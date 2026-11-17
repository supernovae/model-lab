"""Disposable disk indexes for deterministic mixture construction.

Source JSONL and native verification remain authoritative. The selection index
is derived storage: the train documents a mixture may draw from, sorted by
stratum and rank, as memory-mapped columns beside the byte offset of each
source record. It is built by three sharded passes (`documents.jsonl`, the
family inventory and the TRAIN lineage), each merged in file order, so it does
not depend on `--workers`. A small SQLite file keeps only the mutable schedule
state (encoded rows and the seen-content set), charged to the caller's
workspace monitor. SQLite's bounded cache is not a hard process RSS limit.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.shards import declare_stage, plan_ranges, run_sharded
from klab.training.manifest import sha256_file

MAX_ROW_BYTES = 64 * 1024**2
MAX_RECEIPT_BYTES = 256 * 1024**2

DOCUMENTS_STAGE = declare_stage("mixture index: documents")
INVENTORY_STAGE = declare_stage("mixture index: family inventory")
LINEAGE_STAGE = declare_stage("mixture index: TRAIN lineage")

SPLITS = ("train", "validation", "test")
ELIGIBLE = frozenset({"eligible", "eligible_with_obligations"})
INVENTORY_KEYS = frozenset(
    {"document_id", "family_id", "split", "stratum", "content_sha256"}
)
# Identifiers and digests are stored as fixed-width byte columns.
MAX_TEXT_BYTES = 255
MAX_STRATA = 200
# Stratum codes that match nothing: a document whose source or domains do not
# give a mixture stratum, and an inventory row naming a stratum the mixture lacks.
NO_STRATUM, UNKNOWN_STRATUM = 255, 254
BATCH_ROWS = 4096
COVERAGE_ERROR = "train LM view does not cover exactly eligible train documents"


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


def document_rank(
    seed: int, stratum: str, family: str, doc_id: str, order_policy: str | None = None
) -> str:
    """The sort key that orders a stratum's documents for the schedule."""
    if order_policy == "kml-card03-v1":
        return hashlib.sha256(
            f"kml-card03-v1 | {stratum} | {doc_id}".encode()
        ).hexdigest()
    return canonical_digest([seed, stratum, family, doc_id])


def _range_rows(path: Path, start: int, end: int) -> Iterator[tuple[Any, int, int]]:
    """`(value, offset, size)` of each nonblank JSON line starting in a byte range."""
    with path.open("rb") as stream:
        stream.seek(start)
        offset = start
        while offset < end:
            line = stream.readline(MAX_ROW_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_ROW_BYTES:
                raise ValueError("mixture record-byte cap exceeded")
            position, offset = offset, offset + len(line)
            if line.strip():
                try:
                    yield json.loads(line), position, len(line)
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"invalid mixture JSONL at {path}") from error


def _text(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - invalid JSONL content
            "mixture document id and content digest must be strings"
        )
    raw = value.encode("utf-8")
    if len(raw) > MAX_TEXT_BYTES or b"\0" in raw:
        raise ValueError(
            f"mixture document id or content digest is longer than {MAX_TEXT_BYTES} "
            "bytes or contains NUL"
        )
    return raw


def _pack(values: np.ndarray | list, dtype: str) -> dict[str, str]:
    array = np.asarray(values, dtype=dtype)
    return {
        "dtype": array.dtype.str,
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _unpack(packed: dict[str, str]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(packed["data"]), dtype=packed["dtype"])


# Column dtypes of each pass; byte strings are sized by their longest value.
DOCUMENT_COLUMNS = {
    "id": "S",
    "content": "S",
    "offset": "<u8",
    "size": "<u4",
    "split": "u1",
    "stratum": "u1",
    "eligible": "?",
    "source": "<u4",
}
INVENTORY_COLUMNS = {
    "id": "S",
    "content": "S",
    "rank": "S64",
    "family": "<u4",
    "split": "u1",
    "stratum": "u1",
}


class _Columns:
    """Concatenate shard results in shard order, numbering labelled columns globally."""

    def __init__(self, schema: dict[str, str], labelled: tuple[str, ...] = ()):
        self.schema = schema
        self.parts: dict[str, list[np.ndarray]] = defaultdict(list)
        self.tables: dict[str, dict[str, int]] = {name: {} for name in labelled}

    def add(self, result: dict[str, Any]) -> None:
        for name, packed in result["columns"].items():
            values = _unpack(packed)
            table = self.tables.get(name)
            if table is not None:
                remap = np.array(
                    [
                        table.setdefault(label, len(table))
                        for label in result["labels"][name]
                    ],
                    dtype="<u4",
                )
                values = remap[values]
            self.parts[name].append(values)

    def finish(self) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        columns = {
            name: (
                np.concatenate(self.parts.pop(name))
                if self.parts.get(name)
                else np.empty(0, dtype=dtype)
            )
            for name, dtype in self.schema.items()
        }
        return columns, {name: list(table) for name, table in self.tables.items()}


def _collect(plan, function, schema, labelled=()):
    columns = _Columns(schema, labelled)
    for result in run_sharded(plan, function, ordered=True):
        columns.add(result)
    return columns.finish()


def _locate(sorted_ids: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Position of each query among the sorted unique ids, or -1."""
    if not len(sorted_ids):
        return np.full(len(queries), -1, dtype=np.int64)
    position = np.minimum(np.searchsorted(sorted_ids, queries), len(sorted_ids) - 1)
    return np.where(sorted_ids[position] == queries, position, -1)


def _plan(stage: str, path: Path, *extra: Any):
    return plan_ranges(
        stage, path, *extra, inputs={"producer": sha256_file(Path(__file__))}
    )


def _stratum_names(strata: dict[str, str]) -> list[str]:
    names = sorted(set(strata.values()))
    if len(names) > MAX_STRATA:
        raise ValueError(f"a mixture has at most {MAX_STRATA} strata")
    return names


def _documents_range(path: str, start: int, end: int, strata: dict[str, str]):
    """The kept documents of one byte range: where they are and what they say."""
    code = {name: index for index, name in enumerate(_stratum_names(strata))}
    split_code = {name: index for index, name in enumerate(SPLITS)}
    columns: dict[str, list] = {name: [] for name in DOCUMENT_COLUMNS}
    labels: dict[str, int] = {}
    for doc, offset, size in _range_rows(Path(path), start, end):
        if doc["drop_reason"] is not None:
            continue
        source = doc["source_id"]
        expected = strata.get(source)
        columns["id"].append(_text(doc["document_id"]))
        columns["content"].append(_text(doc["content_sha256"]))
        columns["offset"].append(offset)
        columns["size"].append(size)
        columns["split"].append(split_code.get(doc["split"], len(SPLITS)))
        columns["stratum"].append(
            code[expected]
            if expected is not None and expected in doc.get("domains", ())
            else NO_STRATUM
        )
        columns["eligible"].append(
            doc.get("rights", {}).get("training_eligibility") in ELIGIBLE
        )
        columns["source"].append(labels.setdefault(source, len(labels)))
    return {
        "columns": {
            name: _pack(columns[name], dtype)
            for name, dtype in DOCUMENT_COLUMNS.items()
        },
        "labels": {"source": list(labels)},
    }


def _inventory_range(
    path: str, start: int, end: int, names: list[str], seed: int, order_policy
):
    """Validated inventory rows of one byte range, each with its schedule rank."""
    code = {name: index for index, name in enumerate(names)}
    split_code = {name: index for index, name in enumerate(SPLITS)}
    columns: dict[str, list] = {name: [] for name in INVENTORY_COLUMNS}
    labels: dict[str, int] = {}
    for row, _, _ in _range_rows(Path(path), start, end):
        if (
            not isinstance(row, dict)
            or set(row) != INVENTORY_KEYS
            or any(not isinstance(row[key], str) or not row[key] for key in row)
            or row["split"] not in split_code
        ):
            raise ValueError("invalid family inventory row")
        columns["id"].append(_text(row["document_id"]))
        columns["content"].append(_text(row["content_sha256"]))
        columns["rank"].append(
            document_rank(
                seed, row["stratum"], row["family_id"], row["document_id"], order_policy
            ).encode("ascii")
        )
        columns["family"].append(labels.setdefault(row["family_id"], len(labels)))
        columns["split"].append(split_code[row["split"]])
        columns["stratum"].append(code.get(row["stratum"], UNKNOWN_STRATUM))
    return {
        "columns": {
            name: _pack(columns[name], dtype)
            for name, dtype in INVENTORY_COLUMNS.items()
        },
        "labels": {"family": list(labels)},
    }


def _lineage_range(path: str, start: int, end: int):
    """The distinct record IDs of one byte range of the TRAIN lineage view."""
    ids = set()
    for row, _, _ in _range_rows(Path(path), start, end):
        record = row.get("record_id") if isinstance(row, dict) else None
        if not isinstance(record, str):
            raise ValueError(COVERAGE_ERROR)  # noqa: TRY004 - invalid JSONL content
        ids.add(_text(record))
    return {"columns": {"id": _pack(sorted(ids), "S")}}


def _kept_documents(path: Path, strata: dict[str, str]):
    """Kept documents sorted by ID; a repeated ID keeps the last row of the file."""
    columns, labels = _collect(
        _plan(DOCUMENTS_STAGE, path, strata),
        _documents_range,
        DOCUMENT_COLUMNS,
        labelled=("source",),
    )
    order = np.argsort(columns["id"], kind="stable")
    ids = columns["id"][order]
    last = np.append(ids[1:] != ids[:-1], True) if len(ids) else np.empty(0, bool)
    keep = order[last]
    return {name: values[keep] for name, values in columns.items()}, labels["source"]


def _join_inventory(path, documents, strata, seed, order_policy):
    """Check the inventory against the kept documents and attach family and rank."""
    columns, labels = _collect(
        _plan(INVENTORY_STAGE, path, _stratum_names(strata), seed, order_policy),
        _inventory_range,
        INVENTORY_COLUMNS,
        labelled=("family",),
    )
    position = _locate(documents["id"], columns["id"])
    if (position < 0).any() or len(np.unique(position)) != len(position):
        raise ValueError("family inventory has unknown or duplicate document")
    if (
        (documents["split"][position] != columns["split"])
        | (documents["content"][position] != columns["content"])
        | (documents["stratum"][position] != columns["stratum"])
    ).any():
        raise ValueError("family inventory disagrees with verified release")
    pairs = np.unique(columns["family"].astype("<u8") * len(SPLITS) + columns["split"])
    if len(np.unique(pairs // len(SPLITS))) != len(pairs):
        raise ValueError("document family leaks across splits")
    by_content = np.argsort(columns["content"])
    content = columns["content"][by_content]
    first = np.flatnonzero(
        np.append(True, content[1:] != content[:-1]) if len(content) else []
    )
    for column, message in (
        ("split", "identical content leaks across splits"),
        ("stratum", "identical content assigned to different strata"),
    ):
        values = columns[column][by_content]
        if (
            np.minimum.reduceat(values, first) != np.maximum.reduceat(values, first)
        ).any():
            raise ValueError(message)
    if len(position) != len(documents["id"]):
        raise ValueError("family inventory does not cover every kept document")
    placed = np.empty(len(position), dtype=np.int64)
    placed[position] = np.arange(len(position))
    return {
        **documents,
        "family": columns["family"][placed],
        "rank": columns["rank"][placed],
    }, labels["family"]


def _train_view(path: Path, table):
    """Mark the documents of the TRAIN lineage; it must be exactly the eligible train set."""
    columns, _ = _collect(_plan(LINEAGE_STAGE, path), _lineage_range, {"id": "S"})
    ids = np.unique(columns["id"])
    position = _locate(table["id"], ids)
    if (position < 0).any() or (table["split"][position] != 0).any():
        raise ValueError(COVERAGE_ERROR)
    selected = np.zeros(len(table["id"]), dtype=bool)
    selected[position] = True
    if ((table["split"] == 0) & ~selected).any():
        raise ValueError(COVERAGE_ERROR)
    if not table["eligible"][selected].all():
        raise ValueError("train document lacks admitted file or record rights")
    return np.flatnonzero(selected)


class StringTable:
    """Strings by position from two memory-mapped files: UTF-8 bytes and end offsets."""

    def __init__(self, directory: Path, name: str):
        self.data = _map(directory / f"{name}.bin", "u1")
        self.ends = _map(directory / f"{name}.ends", "<u8")

    @staticmethod
    def write(directory: Path, name: str, values: list[str]) -> None:
        encoded = [value.encode("utf-8") for value in values]
        (directory / f"{name}.bin").write_bytes(b"".join(encoded))
        np.cumsum([len(item) for item in encoded], dtype="<u8").tofile(
            directory / f"{name}.ends"
        )

    @functools.lru_cache(maxsize=65536)  # noqa: B019 - one table per index
    def __getitem__(self, index: int) -> str:
        start = int(self.ends[index - 1]) if index else 0
        return bytes(self.data[start : int(self.ends[index])]).decode("utf-8")


def _map(path: Path, dtype: str) -> np.ndarray:
    if not path.stat().st_size:
        return np.empty(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


class Selection:
    """The train documents a mixture draws from, sorted by (stratum, rank, id).

    Each column is one flat file; `strata` gives the row range of each stratum.
    """

    COLUMNS: ClassVar[dict[str, str]] = {
        "rank": "S64",
        "id": "S{id_width}",
        "content": "S{content_width}",
        "offset": "<u8",
        "size": "<u4",
        "family": "<u4",
        "source": "<u4",
    }

    def __init__(self, directory: Path):
        self.directory = directory
        meta = json.loads((directory / "meta.json").read_bytes())
        self.strata: dict[str, list[int]] = meta["strata"]
        count = meta["rows"]
        self.columns = {}
        for name, dtype in self.COLUMNS.items():
            dtype = dtype.format(**meta)
            path = directory / f"{name}.bin"
            if path.stat().st_size != count * np.dtype(dtype).itemsize:
                raise ValueError("mixture selection index is damaged")
            self.columns[name] = _map(path, dtype)
        self.families = StringTable(directory, "family-labels")
        self.sources = StringTable(directory, "source-labels")

    @classmethod
    def write(cls, directory: Path, table, order, names, families, sources):
        """Write the rows at `order` (already sorted by stratum, rank, id)."""
        directory.mkdir()
        meta = {
            "rows": len(order),
            "id_width": table["id"].dtype.itemsize,
            "content_width": table["content"].dtype.itemsize,
            "strata": {},
        }
        start = 0
        for code, name in enumerate(names):
            count = int((table["stratum"][order] == code).sum())
            meta["strata"][name] = [start, start + count]
            start += count
        for name in cls.COLUMNS:
            table[name][order].tofile(directory / f"{name}.bin")
        StringTable.write(directory, "family-labels", families)
        StringTable.write(directory, "source-labels", sources)
        (directory / "meta.json").write_bytes(canonical_json(meta))
        return cls(directory)

    def candidates(self, stratum: str, frontier=None) -> Iterator[tuple]:
        """`(offset, family, rank, id, size, content, source)` after the frontier."""
        start, end = self.strata.get(stratum, (0, 0))
        if frontier is not None:
            rank, doc_id = (item.encode("utf-8") for item in frontier)
            start += int(np.searchsorted(self.columns["rank"][start:end], rank))
            while (
                start < end
                and self.columns["rank"][start] == rank
                and self.columns["id"][start] <= doc_id
            ):
                start += 1
        for low in range(start, end, BATCH_ROWS):
            high = min(low + BATCH_ROWS, end)
            batch = {
                name: column[low:high].tolist() for name, column in self.columns.items()
            }
            for index in range(high - low):
                yield (
                    batch["offset"][index],
                    self.families[batch["family"][index]],
                    batch["rank"][index].decode("ascii"),
                    batch["id"][index].decode("utf-8"),
                    batch["size"][index],
                    batch["content"][index].decode("utf-8"),
                    self.sources[batch["source"][index]],
                )


class MixtureIndex:
    """Index source offsets and family constraints; keep schedule state in SQLite."""

    def __init__(
        self,
        root: Path,
        documents: Path,
        digests: dict[Path, str] | None = None,
        *,
        prepared=None,
    ):
        """`digests` are the input files' SHA-256 the caller already verified (from
        the release manifest); an input without one is hashed when it is first used.
        """
        self.root = Path(root)
        self.documents_path = documents
        self.input_digests: dict[Path, str] = {}
        self.stamps: dict[Path, tuple] = {}
        for path in (documents, *(digests or {})):
            self.bind(path, (digests or {}).get(path))
        self.documents = documents.open("rb")
        self.tokens = (self.root / "encoded.jsonl").open("xb+")
        self.prepared = prepared is not None
        self.selection = Selection(Path(prepared)) if self.prepared else None
        self.staged = None
        self.db = sqlite3.connect(self.root / "mixture.sqlite")
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA temp_store=FILE;
            PRAGMA journal_mode=OFF;
            CREATE TABLE encoded(ordinal INTEGER PRIMARY KEY, offset INTEGER,
                family TEXT, stratum TEXT);
            CREATE TABLE seen(content TEXT PRIMARY KEY);
            CREATE INDEX encoded_order ON encoded(stratum,ordinal);
            CREATE INDEX encoded_families ON encoded(stratum,family);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.documents.close()
        self.tokens.close()
        self.db.close()
        self.selection = self.staged = None

    def bind(self, path: Path, digest: str | None = None) -> None:
        """Record an input's digest and file identity; `digest` None hashes the file."""
        self.input_digests[path] = digest or sha256_file(path)
        info = path.stat()
        self.stamps[path] = (
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def verify_inputs(self):
        """Refuse if an input was replaced or rewritten since it was bound.

        Digests come from the verified release; rereading hundreds of GB to
        confirm them again would repeat that verification.
        """
        for path, stamp in self.stamps.items():
            info = path.stat()
            if stamp != (
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ):
                raise ValueError("mixture input changed during materialization")

    def inventory(self, path, strata, seed, order_policy=None):
        """Check the family inventory against the kept documents; return its row count."""
        if path not in self.input_digests:
            self.bind(path)
        documents, sources = _kept_documents(self.documents_path, strata)
        table, families = _join_inventory(path, documents, strata, seed, order_policy)
        self.staged = (table, strata, families, sources)
        return len(table["id"])

    def select(self, lineage):
        """Keep the documents of the TRAIN lineage view, ordered for the schedule."""
        if self.staged is None:
            raise ValueError("select needs the family inventory first")
        if lineage not in self.input_digests:
            self.bind(lineage)
        table, strata, families, sources = self.staged
        chosen = _train_view(lineage, table)
        names = _stratum_names(strata)
        ordered = []
        for code in range(len(names)):
            rows_ = chosen[table["stratum"][chosen] == code]
            ordered.append(
                rows_[np.lexsort((table["id"][rows_], table["rank"][rows_]))]
            )
        order = np.concatenate(ordered) if ordered else np.empty(0, dtype=np.int64)
        self.selection = Selection.write(
            self.root / "selection", table, order, names, families, sources
        )
        self.staged = None

    def candidates(self, stratum, frontier=None, *, workers=1):
        """Yield ``(doc, family)`` in rank order without reading any text.

        A document is its identity plus ``path`` and ``location``, the byte
        range of its row, for the encoder to read; a document whose IDs are
        cached carries them instead. ``workers`` is unused here: reading is
        the encoder's.
        """
        if frontier is None:
            self.db.execute("DELETE FROM seen")
        if self.selection is None:
            raise ValueError("the mixture index has no selection yet")
        cursor = self.selection.candidates(stratum, frontier)
        path = str(Path(self.documents.name).resolve())
        try:
            for offset, family, rank, doc_id, size, content, source in cursor:
                if size > MAX_ROW_BYTES:
                    raise ValueError("mixture record-byte cap exceeded")
                self.frontier = [rank, doc_id]
                yield (
                    self.cached_document(doc_id, source, content)
                    or {
                        "document_id": doc_id,
                        "source_id": source,
                        "content_sha256": content,
                        "path": path,
                        "location": (offset, size),
                    },
                    family,
                )
        finally:
            cursor.close()

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

    def record_family(self, stratum, family):
        """Count an eligible family without retaining a second token copy."""
        self.record_event(["family", [stratum, family]])
        self.db.execute(
            "INSERT INTO encoded(offset,family,stratum) VALUES(NULL,?,?)",
            (family, stratum),
        )

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
