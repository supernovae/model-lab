"""Disposable disk indexes for deterministic mixture construction.

Source JSONL and native verification remain authoritative. Indexes and encoded
rows are temporary derived storage, charged to the caller's workspace monitor.
SQLite's bounded cache is not a hard process RSS limit.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from klab.canonical import canonical_json
from klab.training.manifest import sha256_file

MAX_ROW_BYTES = 64 * 1024**2
MAX_RECEIPT_BYTES = 256 * 1024**2


def rows(path: Path):
    with path.open("rb") as stream:
        while line := stream.readline(MAX_ROW_BYTES + 1):
            if len(line) > MAX_ROW_BYTES:
                raise ValueError("mixture record-byte cap exceeded")
            if line.strip():
                try:
                    yield json.loads(line)
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"invalid mixture JSONL at {path}") from error


class MixtureIndex:
    """Index source offsets, family constraints and token-spool offsets on disk."""

    def __init__(self, root: Path, documents: Path, *, prepared=None):
        self.input_digests = {documents: sha256_file(documents)}
        self.documents = documents.open("rb")
        self.tokens = (root / "encoded.jsonl").open("xb+")
        self.prepared = prepared is not None
        if prepared is not None:
            shutil.copyfile(prepared, root / "mixture.sqlite")
        self.db = sqlite3.connect(root / "mixture.sqlite")
        if self.prepared:
            self.db.executescript(
                "PRAGMA cache_size=-4096; PRAGMA temp_store=FILE; PRAGMA journal_mode=OFF;"
            )
            return
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA temp_store=FILE;
            PRAGMA journal_mode=OFF;
            CREATE TABLE docs(id TEXT PRIMARY KEY, offset INTEGER, size INTEGER, split TEXT,
                content TEXT, source TEXT, family TEXT, stratum TEXT, rank TEXT,
                inventoried INTEGER DEFAULT 0, selected INTEGER DEFAULT 0);
            CREATE TABLE families(id TEXT PRIMARY KEY, split TEXT);
            CREATE TABLE contents(id TEXT PRIMARY KEY, split TEXT, stratum TEXT);
            CREATE TABLE encoded(ordinal INTEGER PRIMARY KEY, offset INTEGER,
                family TEXT, stratum TEXT);
            CREATE TABLE seen(content TEXT PRIMARY KEY);
            CREATE INDEX ordered_candidates ON docs(stratum,rank,id);
            CREATE INDEX encoded_order ON encoded(stratum,ordinal);
            CREATE INDEX encoded_families ON encoded(stratum,family);
        """)

    def _load_documents(self):
        while True:
            offset = self.documents.tell()
            line = self.documents.readline(MAX_ROW_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_ROW_BYTES:
                raise ValueError("mixture record-byte cap exceeded")
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc["drop_reason"] is None:
                # Match the old mapping's last-value behavior for duplicate IDs.
                self.db.execute(
                    "INSERT OR REPLACE INTO docs(id,offset,size,split,content,source) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        doc["document_id"],
                        offset,
                        len(line),
                        doc["split"],
                        doc["content_sha256"],
                        doc["source_id"],
                    ),
                )

    def __enter__(self):
        try:
            if not self.prepared:
                self._load_documents()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        self.documents.close()
        self.tokens.close()
        self.db.close()

    def document(self, offset):
        self.documents.seek(offset)
        return json.loads(self.documents.readline(MAX_ROW_BYTES + 1))

    def verify_inputs(self):
        for path, digest in self.input_digests.items():
            if sha256_file(path) != digest:
                raise ValueError("mixture input changed during materialization")

    def inventory(self, path, strata, rank):
        self.input_digests[path] = sha256_file(path)
        for row in rows(path):
            if (
                not isinstance(row, dict)
                or set(row)
                != {"document_id", "family_id", "split", "stratum", "content_sha256"}
                or any(not isinstance(row[key], str) or not row[key] for key in row)
                or row["split"] not in {"train", "validation", "test"}
            ):
                raise ValueError("invalid family inventory row")
            found = self.db.execute(
                "SELECT offset,inventoried FROM docs WHERE id=?", (row["document_id"],)
            ).fetchone()
            if found is None or found[1]:
                raise ValueError("family inventory has unknown or duplicate document")
            doc = self.document(found[0])
            if (
                doc["split"] != row["split"]
                or doc["content_sha256"] != row["content_sha256"]
                or strata.get(doc["source_id"]) != row["stratum"]
                or row["stratum"] not in doc["domains"]
            ):
                raise ValueError("family inventory disagrees with verified release")
            family = self.db.execute(
                "SELECT split FROM families WHERE id=?", (row["family_id"],)
            ).fetchone()
            if family is not None and family[0] != row["split"]:
                raise ValueError("document family leaks across splits")
            self.db.execute(
                "INSERT OR IGNORE INTO families VALUES(?,?)",
                (row["family_id"], row["split"]),
            )
            content = self.db.execute(
                "SELECT split,stratum FROM contents WHERE id=?",
                (row["content_sha256"],),
            ).fetchone()
            if content is not None:
                if content[0] != row["split"]:
                    raise ValueError("identical content leaks across splits")
                if content[1] != row["stratum"]:
                    raise ValueError("identical content assigned to different strata")
            self.db.execute(
                "INSERT OR IGNORE INTO contents VALUES(?,?,?)",
                (row["content_sha256"], row["split"], row["stratum"]),
            )
            self.db.execute(
                "UPDATE docs SET family=?,stratum=?,rank=?,inventoried=1 WHERE id=?",
                (row["family_id"], row["stratum"], rank(row), row["document_id"]),
            )
        if self.db.execute("SELECT 1 FROM docs WHERE inventoried=0 LIMIT 1").fetchone():
            raise ValueError("family inventory does not cover every kept document")

    def select(self, lineage):
        self.input_digests[lineage] = sha256_file(lineage)
        for row in rows(lineage):
            found = self.db.execute(
                "SELECT split FROM docs WHERE id=?", (row["record_id"],)
            ).fetchone()
            if found is None or found[0] != "train":
                raise ValueError(
                    "train LM view does not cover exactly eligible train documents"
                )
            self.db.execute(
                "UPDATE docs SET selected=1 WHERE id=?", (row["record_id"],)
            )
        if self.db.execute(
            "SELECT 1 FROM docs WHERE split='train' AND selected=0 LIMIT 1"
        ).fetchone():
            raise ValueError(
                "train LM view does not cover exactly eligible train documents"
            )
        for (offset,) in self.db.execute("SELECT offset FROM docs WHERE selected=1"):
            if self.document(offset).get("rights", {}).get(
                "training_eligibility"
            ) not in {"eligible", "eligible_with_obligations"}:
                raise ValueError("train document lacks admitted file or record rights")
        self.db.commit()

    def candidates(self, stratum, frontier=None, *, workers=1):
        if frontier is None:
            self.db.execute("DELETE FROM seen")
        from klab.corpus.mixture_pipeline import ordered_fetch

        cursor = self.db.execute(
            "SELECT offset,family,rank,id,size,content,source FROM docs WHERE selected=1 AND stratum=? "
            "AND (rank,id) > (?,?) ORDER BY rank,id",
            (stratum, *(frontier or ["", ""])),
        )

        def fetch(row):
            # Independent file descriptions: seek positions are never shared.
            with open(self.documents.name, "rb") as stream:
                stream.seek(row[0])
                raw = stream.readline(MAX_ROW_BYTES + 1)
            if len(raw) > MAX_ROW_BYTES:
                raise ValueError("mixture record-byte cap exceeded")
            return json.loads(raw)

        try:
            for row, doc in ordered_fetch(
                cursor,
                fetch,
                workers=workers,
                size=lambda row: self.candidate_bytes(row[4], row[5]),
                metrics=getattr(self, "pipeline_metrics", None),
                cached=lambda row: self.cached_document(row[3], row[6], row[5]),
            ):
                self.frontier = [row[2], row[3]]
                yield doc, row[1]
        finally:
            cursor.close()

    def candidate_bytes(self, source_bytes, content):
        lookup = getattr(self, "token_bytes", None)
        return max(source_bytes, lookup(content) if lookup is not None else 0)

    def cached_document(self, doc_id, source, content):
        lookup = getattr(self, "token_lookup", None)
        ids = (
            lookup(content, count_miss=False, decode=False)
            if lookup is not None
            else None
        )
        if ids is None:
            return None
        return {
            "document_id": doc_id,
            "source_id": source,
            "content_sha256": content,
            "text": "",
            "cached_payload": ids,
        }

    def seen(self, digest):
        duplicate = (
            self.db.execute("INSERT OR IGNORE INTO seen VALUES(?)", (digest,)).rowcount
            == 0
        )
        if not duplicate:
            self.record_event(["seen", digest])
        return duplicate

    def record_event(self, event):
        sink = getattr(self, "event_sink", None)
        if sink is not None:
            sink(event)

    def restore_event(self, event):
        kind, value = event
        if kind == "seen":
            self.db.execute("INSERT OR IGNORE INTO seen VALUES(?)", (value,))
        elif kind == "append":
            self.append(value)
        elif kind == "family":
            self.record_family(*value)
        elif kind == "reset":
            self.db.execute("DELETE FROM seen")
        else:
            raise ValueError("invalid mixture checkpoint index event")

    def append(self, row):
        self.record_event(["append", row])
        encoded = canonical_json(row) + b"\n"
        if len(encoded) > MAX_ROW_BYTES:
            raise ValueError("mixture encoded record-byte cap exceeded")
        self.tokens.seek(0, 2)
        offset = self.tokens.tell()
        self.tokens.write(encoded)
        self.db.execute(
            "INSERT INTO encoded(offset,family,stratum) VALUES(?,?,?)",
            (offset, row["family_id"], row["stratum"]),
        )

    def encoded(self, stratum, after=0):
        self.tokens.flush()
        cursor = self.db.execute(
            "SELECT offset,ordinal FROM encoded WHERE stratum=? AND ordinal>? ORDER BY ordinal",
            (stratum, after),
        )
        try:
            for offset, ordinal in cursor:
                self.encoded_frontier = ordinal
                self.tokens.seek(offset)
                yield json.loads(self.tokens.readline(MAX_ROW_BYTES + 1))
        finally:
            cursor.close()
        self.tokens.seek(0, 2)

    def family_count(self, stratum):
        return self.db.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT family FROM encoded "
            "INDEXED BY encoded_families WHERE stratum=?)",
            (stratum,),
        ).fetchone()[0]
