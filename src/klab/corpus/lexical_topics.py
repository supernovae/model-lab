"""Train-only lexical anchors for paired sequence ordering."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import struct
import unicodedata
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import numpy as np

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import capture
from klab.corpus.build_stages import implementation_digests
from klab.corpus.mixture_index import ELIGIBLE
from klab.corpus.mixture_pipeline import mixture_writer
from klab.corpus.release import open_release
from klab.corpus.release_verify import current_settings, scratch_root
from klab.corpus.split_runs import SplitStages, checkpoint_stamp
from klab.data.verification import HashingWriter
from klab.progress import progress_phase
from klab.safe_io import symlinked
from klab.workdir import resolve_work_dir
from klab.workspace_preflight import check_storage, require_storage

_WORD = re.compile(r"[^\W\d_]+")
_PREFIX_CHARS = 4096
_TERMS_PER_DOCUMENT = 4
_ENTRY = struct.Struct("<64sQ")
_DTYPE = np.dtype([("id", "S64"), ("anchor", "<u8")])
_FILE = "topics.bin"


def _terms(text: str) -> list[bytes]:
    counts = Counter(
        word
        for word in _WORD.findall(text[:_PREFIX_CHARS].casefold())
        if 4 <= len(word) <= 32
    )
    chosen = sorted(counts, key=lambda word: (-counts[word], -len(word), word))
    return [
        hashlib.blake2b(word.encode(), digest_size=8, person=b"klab-lexical").digest()
        for word in chosen[:_TERMS_PER_DOCUMENT]
    ]


def _train_documents(releases, source_strata):
    for release in releases:
        for doc in release.rows("documents.jsonl"):
            stratum = source_strata.get(doc["source_id"])
            if (
                doc["split"] == "train"
                and doc["drop_reason"] is None
                and doc["rights"]["training_eligibility"] in ELIGIBLE
                and stratum in doc["domains"]
            ):
                yield doc["document_id"], stratum, doc["text"]


def build_lexical_index(directory, database, documents, *, max_topics, max_documents):
    """Spend one text pass and reduce narrow exact postings on disk."""
    page_budget = (4 * 1024**2 + 768 * max_documents) // 4096
    index_bytes = _ENTRY.size * max_documents
    scratch_bytes = 8192 * page_budget
    if database.parent.stat().st_dev == directory.stat().st_dev:
        require_storage(
            [
                check_storage(
                    database.parent, projected_bytes=scratch_bytes + index_bytes
                )
            ]
        )
    else:
        require_storage(
            [
                check_storage(database.parent, projected_bytes=scratch_bytes),
                check_storage(directory, projected_bytes=index_bytes),
            ]
        )
    if symlinked(database):
        raise ValueError("unsafe lexical index scratch path")
    database.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(database)) as db:
            db.executescript(f"""
                PRAGMA journal_mode=OFF;
                PRAGMA page_size=4096;
                PRAGMA cache_size=-8192;
                PRAGMA temp_store=FILE;
                PRAGMA max_page_count={page_budget};
                CREATE TABLE documents(ordinal INTEGER PRIMARY KEY, id BLOB NOT NULL);
                CREATE TABLE postings(ordinal INTEGER, rank INTEGER, stratum TEXT,
                    term BLOB, PRIMARY KEY(ordinal, rank)) WITHOUT ROWID;
                CREATE TABLE anchors(stratum TEXT, term BLOB, ordinal INTEGER,
                    PRIMARY KEY(stratum, term)) WITHOUT ROWID;
            """)
            return _reduce_index(db, directory, documents, max_topics, max_documents)
    finally:
        database.unlink(missing_ok=True)


def _reduce_index(db, directory, documents, max_topics, max_documents):
    counts: Counter[str] = Counter()
    with progress_phase("corpus_lexical_terms", unit="documents") as progress:
        for number, (document_id, stratum, text) in enumerate(documents, 1):
            if number > max_documents or len(document_id) != 64:
                raise ValueError("lexical index exceeds its verified document scope")
            db.execute(
                "INSERT INTO documents VALUES (?, ?)",
                (number, document_id.encode("ascii")),
            )
            db.executemany(
                "INSERT INTO postings VALUES (?, ?, ?, ?)",
                (
                    (number, rank, stratum, term)
                    for rank, term in enumerate(_terms(text))
                ),
            )
            counts[stratum] += 1
            if number % 8192 == 0:
                progress.update(completed_work=number, emit=False)
        progress.update(completed_work=sum(counts.values()))
    anchors: dict[str, int] = {}
    with progress_phase("corpus_lexical_topics", unit="strata") as progress:
        db.execute("CREATE INDEX term_counts ON postings(stratum, term)")
        for stratum, count in sorted(counts.items()):
            terms = db.execute(
                """SELECT term FROM postings WHERE stratum=? GROUP BY term
                   HAVING COUNT(*) BETWEEN 2 AND ?
                   ORDER BY COUNT(*) DESC, term LIMIT ?""",
                (stratum, max(2, count // 2), max_topics),
            ).fetchall()
            db.executemany(
                "INSERT INTO anchors VALUES (?, ?, ?)",
                ((stratum, term, ordinal) for ordinal, (term,) in enumerate(terms, 1)),
            )
            anchors[stratum] = len(terms)
            progress.update(completed_work=len(anchors), emit=False)
        rows = db.execute("""
            SELECT d.id, COALESCE((SELECT a.ordinal FROM postings p JOIN anchors a
                ON a.stratum=p.stratum AND a.term=p.term WHERE p.ordinal=d.ordinal
                ORDER BY p.rank LIMIT 1), 0) FROM documents d ORDER BY d.id
        """)
        path = directory / _FILE
        assigned = 0
        previous = None
        with path.open("xb") as output:
            writer = HashingWriter(output)
            for document_id, group in rows:
                if document_id == previous:
                    raise ValueError("duplicate document in train lexical index")
                previous = document_id
                writer.write(_ENTRY.pack(document_id, group))
                assigned += bool(group)
            proof = writer.seal(path)
    return {
        "index_sha256": proof.sha256,
        "indexed_documents": sum(counts.values()),
        "assigned_documents": assigned,
        "anchors_by_stratum": anchors,
        "files": [
            {"path": str(path), "sha256": proof.sha256, "bytes": proof.size_bytes}
        ],
    }


class LexicalTopics:
    def __init__(self, result, request_id, max_topics):
        path = Path(result["files"][0]["path"])
        self.path = path
        self.generation = checkpoint_stamp(path)
        self.max_topics = max_topics
        if self.generation[2] != result["indexed_documents"] * _ENTRY.size:
            raise ValueError("lexical index size disagrees with its receipt")
        self.rows = (
            np.memmap(path, dtype=_DTYPE, mode="r")
            if self.generation[2]
            else np.empty(0, dtype=_DTYPE)
        )
        self.ids = self.rows["id"]
        self.receipt = {
            "id": request_id,
            "max_topics": max_topics,
            "eligible_rights": sorted(ELIGIBLE),
            "unicode_version": unicodedata.unidata_version,
            **{key: value for key, value in result.items() if key != "files"},
        }

    def group(self, document_id: str) -> int:
        key = document_id.encode("ascii")
        i = int(self.ids.searchsorted(key))
        if i == len(self.ids) or self.ids[i] != key:
            raise ValueError(
                "scheduled document is missing from the train lexical index"
            )
        return int(self.rows["anchor"][i])

    def require_unchanged(self):
        if checkpoint_stamp(self.path) != self.generation:
            raise ValueError("lexical index changed during sequence ordering")


@contextmanager
def lexical_topics(spec, base, *, audit_directory=None):
    """Open a scoped native phase, or independently rebuild it during audit."""
    with ExitStack() as stack:
        releases = [
            stack.enter_context(open_release(path))
            for path in (
                spec.release_path,
                *(a.release_path for a in spec.additional_releases),
            )
        ]
        paths = [release.member("documents.jsonl") for release in releases]
        max_documents = sum(
            json.loads(release.member("report.json").read_bytes())["split_counts"][
                "train"
            ]
            for release in releases
        )
        max_topics = spec.sequence_order.max_topics or 16
        request = {
            "producer": implementation_digests("lexical_topics.py"),
            "releases": [
                [
                    release.manifest["release_id"],
                    release.manifest["files"]["documents.jsonl"]["sha256"],
                ]
                for release in releases
            ],
            "family_inventory_sha256": base["family_inventory_sha256"],
            "source_strata": spec.source_strata,
            "max_topics": max_topics,
            "eligible_rights": sorted(ELIGIBLE),
            "unicode_version": unicodedata.unidata_version,
        }
        identity = canonical_digest(request)
        work_root = resolve_work_dir()
        stage_root = work_root / "cache" / "lexical-topics"
        scratch_key = canonical_digest([str(work_root), identity])
        database = scratch_root() / f"lexical-topics-{scratch_key}.sqlite"

        def build(directory):
            result = build_lexical_index(
                directory,
                database,
                _train_documents(releases, spec.source_strata),
                max_topics=max_topics,
                max_documents=max_documents,
            )
            for release in releases:
                release.require_unchanged()
            return result

        with mixture_writer(stage_root / identity):
            if audit_directory is None:
                stages = SplitStages(
                    stage_root,
                    {**request, "inputs": capture(paths)},
                    work_root=work_root,
                    producer_scoped=True,
                )
                result = stages.phase("train-topics", build)
                echo = current_settings().echo
                if echo is not None:
                    state = "reused" if stages.reused else "built"
                    print(f"Lexical index {identity[:16]} {state}", file=echo)
            else:
                result = build(audit_directory)
            topics = LexicalTopics(result, identity, max_topics)
        yield topics
        topics.require_unchanged()
