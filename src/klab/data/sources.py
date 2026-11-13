"""Dataset-neutral pinned declarations and immutable, replayable text snapshots.

Public API: load_source -> DatasetSource, lock_source -> lock Path,
snapshot_source -> manifest Path, verify_snapshot -> dict, iter_snapshot -> text.
Locking checks Hub metadata only. Snapshotting streams standard datasets (datasets
4 never executes repository Python), or explicitly named verified Forge files.
Resume replays and authenticates the committed prefix; it does not assume upstream
seek support. SQLite is the transactional journal and disk-backed exact-text index.
Resource ceilings are safety stops, never evidence of full-source exhaustion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.corpus import acquisition as forge
from klab.corpus.project import safe_name
from klab.documents import read_document
from klab.engram.packs import _rename_noreplace
from klab.hf_auth import hub_access_errors, hub_auth_kwargs, raise_for_hub_auth
from klab.result_publication import sync_directory
from klab.training.manifest import sha256_file
from klab.verification_proofs import VerificationMode


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResourceBounds(_Model):
    """Explicit ceilings; records are per split, bytes cover all splits.

    max_work_bytes bounds journal + publication disk usage, not Hub library cache
    traffic. min_free_bytes is checked on both actual output and cache filesystems.
    max_record_bytes bounds serialized source rows after the library decodes them.
    """

    max_source_records: int = Field(gt=0, strict=True)
    max_text_bytes: int = Field(gt=0, strict=True)
    max_record_bytes: int = Field(gt=0, strict=True)
    max_work_bytes: int = Field(gt=0, strict=True)
    min_free_bytes: int = Field(ge=0, strict=True)


class Selection(_Model):
    mode: Literal["bounded", "exhaustion"]
    documents: dict[str, Annotated[int, Field(strict=True, gt=0)]] = Field(
        default_factory=dict
    )


class DedupPolicy(_Model):
    within_split: Literal["exclude", "error", "keep"] = "exclude"
    overlap: Literal["exclude", "error", "keep"] = "exclude"
    priority: tuple[str, ...] = ("validation", "train")


class ForgeFiles(_Model):
    snapshot_path: Path
    files: tuple[str, ...]


class DatasetSource(_Model):
    schema_version: Literal[1] = 1
    kind: Literal["huggingface", "forge_files"] = "huggingface"
    repo_id: str
    revision: str
    config: str
    splits: dict[str, str]
    text_field: str
    attribution: str
    license: str
    license_url: str | None = None
    selection: Selection
    resources: ResourceBounds
    dedup: DedupPolicy = Field(default_factory=DedupPolicy)
    forge_files: dict[str, ForgeFiles] = Field(default_factory=dict)

    @model_validator(mode="after")
    def contract(self) -> DatasetSource:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision):
            raise ValueError("revision must be an immutable lowercase commit hash")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo_id):
            raise ValueError("repo_id must be owner/repository")
        for value in (self.config, self.text_field, self.attribution, self.license):
            if not value.strip():
                raise ValueError("source identity fields must be nonblank")
        if not self.splits or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", key)
            or key in {"events", "excluded", "manifest"}
            or not value.strip()
            for key, value in self.splits.items()
        ):
            raise ValueError(
                "split mapping needs safe logical names and upstream splits"
            )
        if len(set(self.splits.values())) != len(self.splits):
            raise ValueError("upstream split mapping must be distinct")
        if set(self.dedup.priority) != set(self.splits) or len(
            self.dedup.priority
        ) != len(self.splits):
            raise ValueError("dedup priority must name every split exactly once")
        if self.selection.mode == "bounded":
            if set(self.selection.documents) != set(self.splits) or any(
                type(n) is not int or n <= 0 or n > self.resources.max_source_records
                for n in self.selection.documents.values()
            ):
                raise ValueError(
                    "bounded selection requires positive per-split documents within resource bounds"
                )
        elif self.selection.documents:
            raise ValueError("exhaustion cannot declare document targets")
        if (self.kind == "forge_files") != bool(self.forge_files):
            raise ValueError("forge_files requires explicit Forge file bindings")
        if self.forge_files:
            if set(self.forge_files) != set(self.splits):
                raise ValueError("Forge files must bind every split")
            for binding in self.forge_files.values():
                if not binding.files or len(set(binding.files)) != len(binding.files):
                    raise ValueError("Forge files must be nonempty and distinct")
                for name in binding.files:
                    safe_name(name)
                    if not name.endswith((".jsonl", ".jsonl.gz", ".parquet")):
                        raise ValueError("Forge files must be JSONL or Parquet")
        return self


def seal_snapshot_payload(value: dict, key: str) -> dict:
    return {**value, key: canonical_digest(value)}


def _read(path: Path, key: str) -> dict:
    forge._regular(path)
    value = read_document(path)
    if not isinstance(value, dict) or key not in value:
        raise ValueError("missing artifact identity")
    body = {k: v for k, v in value.items() if k != key}
    if value[key] != canonical_digest(body):
        raise ValueError("artifact manifest hash mismatch")
    return value


def load_source(path: str | Path) -> DatasetSource:
    """Read strict YAML; resolve explicit Forge paths relative to the declaration."""
    path = Path(path)
    source = DatasetSource.model_validate(read_document(path))
    bindings = {
        split: binding.model_copy(
            update={"snapshot_path": (path.parent / binding.snapshot_path).resolve()}
        )
        for split, binding in source.forge_files.items()
    }
    return source.model_copy(update={"forge_files": bindings})


def forge_bindings(source: DatasetSource) -> dict:
    result = {}
    for split, binding in source.forge_files.items():
        manifest = forge.verify_snapshot(binding.snapshot_path)
        declaration = manifest["declaration"]
        if (
            declaration["revision"] != source.revision
            or declaration["canonical_uri"]
            != f"https://huggingface.co/datasets/{source.repo_id}"
        ):
            raise ValueError("Forge source provenance mismatch")
        spec = declaration["acquisition"]
        if (
            spec["config"],
            spec["split"],
            spec["text_field"],
            declaration["license"],
        ) != (source.config, source.splits[split], source.text_field, source.license):
            raise ValueError("Forge selection provenance mismatch")
        inventory = {row["path"] for row in manifest["files"]}
        if not set(binding.files) <= inventory:
            raise ValueError("Forge file missing from verified inventory")
        result[split] = manifest["snapshot_sha256"]
    return result


def _hub_file(source: DatasetSource, url: str) -> str:
    """Accept only a file in this exact immutable Hub repository revision."""
    parsed = urlsplit(url)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("source data file must be a pinned Hub file")
    if parsed.scheme == "hf" and parsed.netloc == "datasets":
        prefix = f"/{source.repo_id}@{source.revision}/"
    elif parsed.scheme == "https" and parsed.netloc == "huggingface.co":
        prefix = f"/datasets/{source.repo_id}/resolve/{source.revision}/"
    else:
        raise ValueError(
            "external data URLs are not supported; use verified Forge files"
        )
    path = unquote(parsed.path)
    if not path.startswith(prefix):
        raise ValueError("data file repository/revision differs from pinned source")
    return safe_name(path[len(prefix) :])


def _hub_identity(source: DatasetSource, *, cache_dir: Path | None = None) -> dict:
    from datasets import load_dataset_builder
    from huggingface_hub import HfApi

    auth = hub_auth_kwargs()
    try:
        info = HfApi().dataset_info(
            source.repo_id, revision=source.revision, files_metadata=True, **auth
        )
        builder = load_dataset_builder(
            source.repo_id,
            name=source.config,
            revision=source.revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            **auth,
        )
    except hub_access_errors() as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))
    if info.sha != source.revision:
        raise ValueError("Hub resolved revision differs from pinned commit")
    loader = builder.info.builder_name
    if loader not in {"parquet", "json", "text", "csv", "arrow"}:
        raise ValueError(
            "source requires a standard file loader or verified Forge files"
        )
    options = {}
    if is_dataclass(builder.config):
        for field in fields(builder.config):
            if field.name in {
                "name",
                "version",
                "data_dir",
                "data_files",
                "description",
            }:
                continue
            value = getattr(builder.config, field.name)
            if field.name == "features" and value is not None:
                value = value.to_dict()
            try:
                canonical_json(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"unsupported standard loader option: {field.name}"
                ) from error
            options[field.name] = value
    if (
        options.get("on_bad_files", "error") != "error"
        or options.get("on_bad_lines", "error") != "error"
    ):
        raise ValueError("source loader must fail rather than skip malformed data")
    siblings = {item.rfilename: item for item in info.siblings or []}
    files = {}
    for logical, upstream in source.splits.items():
        selected = (builder.config.data_files or {}).get(upstream)
        if not selected:
            raise ValueError(f"missing source split data files: {upstream}")
        rows = []
        for url in selected:
            name = _hub_file(source, str(url))
            entry = siblings.get(name)
            if entry is None or type(entry.size) is not int or entry.size < 0:
                raise ValueError(f"missing Hub file metadata: {name}")
            blob = entry.blob_id
            if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
                raise ValueError(f"missing Hub Git blob identity: {name}")
            row = {"path": name, "size": entry.size, "blob_id": blob}
            lfs = getattr(entry, "lfs", None)
            if lfs is not None:
                digest = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise ValueError(f"invalid Hub LFS content identity: {name}")
                row["lfs_sha256"] = digest
            rows.append(row)
        if len({row["path"] for row in rows}) != len(rows):
            raise ValueError("duplicate source data file")
        files[logical] = rows
    return {
        "repo_id": source.repo_id,
        "revision": info.sha,
        "loader": loader,
        "files": files,
        "loader_options": options,
    }


def _source_payload(source: DatasetSource) -> dict:
    value = source.model_dump(mode="json")
    for binding in value["forge_files"].values():
        binding.pop("snapshot_path")
    return value


def _locked_source(lock: dict, locations: dict | None = None) -> DatasetSource:
    value = {
        **lock["source"],
        "forge_files": {
            split: {**binding, "snapshot_path": (locations or {}).get(split, ".")}
            for split, binding in lock["source"].get("forge_files", {}).items()
        },
    }
    return DatasetSource.model_validate(value)


def _verify_lock_payload(lock: dict) -> DatasetSource:
    if lock.get("format") != "klab-dataset-lock-v1" or lock.get(
        "lock_sha256"
    ) != canonical_digest(
        {key: value for key, value in lock.items() if key != "lock_sha256"}
    ):
        raise ValueError("unsupported or changed snapshot lock")
    source = _locked_source(lock)
    if lock["source"] != _source_payload(source):
        raise ValueError("lock source must use canonical location-independent fields")
    provenance = lock["provenance"]
    if source.kind == "forge_files":
        if set(provenance) != set(source.splits) or any(
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in provenance.values()
        ):
            raise ValueError("invalid Forge snapshot identities")
    else:
        if (provenance.get("repo_id"), provenance.get("revision")) != (
            source.repo_id,
            source.revision,
        ):
            raise ValueError("lock provenance mismatch")
        if provenance.get("loader") not in {"parquet", "json", "text", "csv", "arrow"}:
            raise ValueError("invalid locked data loader")
        options = provenance.get("loader_options", {})
        if not isinstance(options, dict) or set(options) & {
            "path",
            "name",
            "data_files",
            "data_dir",
            "revision",
            "token",
            "cache_dir",
            "streaming",
            "split",
            "storage_options",
            "trust_remote_code",
        }:
            raise ValueError("invalid locked loader options")
        if (
            options.get("on_bad_files", "error") != "error"
            or options.get("on_bad_lines", "error") != "error"
        ):
            raise ValueError("locked source cannot skip malformed files or rows")
        if set(provenance.get("files", {})) != set(source.splits):
            raise ValueError("lock split inventory mismatch")
        for rows in provenance["files"].values():
            if not isinstance(rows, list) or not rows:
                raise ValueError("empty source file inventory")
            names = set()
            for row in rows:
                name = safe_name(row["path"])
                if (
                    name in names
                    or type(row["size"]) is not int
                    or row["size"] < 0
                    or not re.fullmatch(r"[0-9a-f]{40}", row["blob_id"])
                ):
                    raise ValueError("invalid source file inventory")
                if "lfs_sha256" in row and (
                    not isinstance(row["lfs_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["lfs_sha256"])
                ):
                    raise ValueError("invalid locked LFS content identity")
                names.add(name)
    return source


def lock_source(
    source_path: str | Path, output: str | Path, cache_dir: str | Path
) -> Path:
    """Validate pinned availability without downloading dataset bodies; publish exclusively."""
    source = load_source(source_path)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    provenance = (
        forge_bindings(source)
        if source.kind == "forge_files"
        else _hub_identity(source, cache_dir=Path(cache_dir))
    )
    value = seal_snapshot_payload(
        {
            "format": "klab-dataset-lock-v1",
            "source": _source_payload(source),
            "provenance": provenance,
        },
        "lock_sha256",
    )
    _verify_lock_payload(value)
    if source.kind == "forge_files":
        locations = {
            split: str(binding.snapshot_path)
            for split, binding in source.forge_files.items()
        }
        with output.with_name(output.name + ".locations.json").open("xb") as handle:
            handle.write(
                canonical_json(
                    {"lock_sha256": value["lock_sha256"], "locations": locations}
                )
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    # Same-filesystem temporary name is also exclusive; never replace another lock.
    temporary = output.with_name(output.name + ".pending")
    with temporary.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _rename_noreplace(temporary, output)
    sync_directory(output.parent)
    return output


def verify_lock(path: str | Path) -> dict:
    """Cold authenticate an immutable lock without fetching Hub data."""
    lock = _read(Path(path), "lock_sha256")
    source = _verify_lock_payload(lock)
    if source.kind == "forge_files":
        source = _locked_source(lock, _lock_locations(Path(path), lock))
        if forge_bindings(source) != lock["provenance"]:
            raise ValueError("lock provenance mismatch")
    return lock


def _lock_locations(path: Path, lock: dict) -> dict:
    sidecar = path.with_name(path.name + ".locations.json")
    forge._regular(sidecar)
    binding = read_document(sidecar)
    if binding.get("lock_sha256") != lock["lock_sha256"] or set(
        binding.get("locations", {})
    ) != set(lock["source"]["forge_files"]):
        raise ValueError("Forge lock location binding mismatch")
    return {
        split: str((path.parent / location).absolute())
        for split, location in binding["locations"].items()
    }


def _stream_parquet(
    files: list[str], options: dict, source: DatasetSource, cache_dir: Path, auth: dict
) -> Iterator[dict]:
    """Read locked Parquet files without asynchronous Arrow-to-Python callbacks.

    The datasets Parquet scanner uses background I/O even with pre-buffering
    disabled. A bounded consumer can abandon that scanner before its callbacks
    finish, deadlocking Python 3.14 finalization. ParquetFile's synchronous batch
    reader owns no such outstanding scanner work. Preserve the packaged loader's
    row order, filtering, projection, feature casting and decoding here.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features
    from datasets.download.download_config import DownloadConfig
    from datasets.table import table_cast
    from datasets.utils.file_utils import xopen

    allowed = {
        "batch_size",
        "columns",
        "features",
        "filters",
        "fragment_scan_options",
        "on_bad_files",
    }
    if set(options) - allowed or options.get("fragment_scan_options") is not None:
        raise ValueError("unsupported synchronous Parquet loader options")
    if options.get("on_bad_files", "error") != "error":
        raise ValueError("Parquet source cannot skip malformed files")
    features = options.get("features")
    if features is not None:
        features = Features.from_dict(features)
    columns = options.get("columns")
    if features is not None and columns is not None and set(features) != set(columns):
        raise ValueError("Parquet columns and features must contain the same columns")
    filters = options.get("filters")
    expression = (
        pq.filters_to_expression(filters) if isinstance(filters, list) else filters
    )
    download = DownloadConfig(cache_dir=str(cache_dir), **auth)
    for file in files:
        with (
            xopen(file, "rb", download_config=download) as handle,
            pq.ParquetFile(handle, pre_buffer=False, buffer_size=65536) as parquet,
        ):
            if features is None:
                features = Features.from_arrow_schema(parquet.schema_arrow)
                if columns is not None:
                    features = Features(
                        {
                            key: value
                            for key, value in features.items()
                            if key in columns
                        }
                    )
            if not parquet.num_row_groups:
                continue
            # Row groups may span an entire corpus. Bound decoder batches
            # independently of file layout and a larger declared batch size.
            batch_size = min(1024, options.get("batch_size") or 1024)
            batches = parquet.iter_batches(
                batch_size=max(1, batch_size),
                columns=columns if expression is None else None,
                use_threads=False,
            )
            try:
                for batch in batches:
                    table = pa.Table.from_batches([batch])
                    if expression is not None:
                        table = table.filter(expression)
                        if columns is not None:
                            table = table.select(columns)
                    table = table_cast(table, features.arrow_schema)
                    for row in table.to_pylist():
                        yield features.decode_example(
                            row,
                            token_per_repo_id={source.repo_id: auth.get("token")},
                        )
            finally:
                batches.close()


def stream_source(
    source: DatasetSource, split: str, cache_dir: Path, *, inventory: dict | None = None
) -> Iterator[dict]:
    from klab.data.acquisition_scope import current_transport

    transport = current_transport.get()
    if transport is not None:
        yield from transport.stream(source, split, inventory)
        return
    if source.kind == "forge_files":
        binding = source.forge_files[split]
        for name in binding.files:
            yield from (
                row
                for _, row in forge._bounded_hf_rows(
                    binding.snapshot_path / "files" / name,
                    limit=source.resources.max_source_records + 1,
                    max_line_bytes=source.resources.max_record_bytes,
                )
            )
        return
    from datasets import load_dataset

    auth = hub_auth_kwargs()
    if inventory is None:
        raise ValueError("Hub acquisition requires a locked data-file inventory")
    from huggingface_hub import hf_hub_url

    options = dict(inventory.get("loader_options", {}))
    files = [
        hf_hub_url(
            source.repo_id, row["path"], repo_type="dataset", revision=source.revision
        )
        for row in inventory["files"][split]
    ]
    try:
        if inventory["loader"] == "parquet":
            yield from _stream_parquet(files, options, source, cache_dir, auth)
            return
        if options.get("features") is not None:
            from datasets import Features

            options["features"] = Features.from_dict(options["features"])
        yield from load_dataset(
            inventory["loader"],
            data_files={source.splits[split]: files},
            split=source.splits[split],
            streaming=True,
            cache_dir=str(cache_dir),
            **options,
            **auth,
        )
    except hub_access_errors() as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))


def snapshot_decision(
    db: sqlite3.Connection, source: DatasetSource, split: str, ordinal: int, row: dict
) -> dict:
    raw = canonical_json(row)
    if len(raw) > source.resources.max_record_bytes:
        raise ValueError("resource bound: max_record_bytes")
    if source.text_field not in row:
        raise ValueError(f"missing text field: {source.text_field}")
    text = row[source.text_field]
    if text is not None and not isinstance(text, str):
        raise ValueError("text field must be string or null")
    reason = "null" if text is None else "empty" if not text else "selected"
    if reason == "selected":
        digest = hashlib.sha256(text.encode()).hexdigest()
        prior = db.execute(
            "SELECT text, split FROM seen WHERE digest=? ORDER BY (split=?) DESC LIMIT 1",
            (digest, split),
        ).fetchone()
        if prior:
            if prior[0] != text:
                raise ValueError("text digest collision")
            category = "duplicate" if prior[1] == split else "overlap"
            policy = (
                source.dedup.within_split
                if category == "duplicate"
                else source.dedup.overlap
            )
            if policy == "error":
                raise ValueError(f"forbidden {category}")
            if policy == "exclude":
                reason = category
        if reason == "selected":
            db.execute(
                "INSERT OR IGNORE INTO seen VALUES (?, ?, ?)", (digest, text, split)
            )
    return {
        "split": split,
        "ordinal": ordinal,
        "source_sha256": canonical_digest(row),
        "reason": reason,
        "text": text if reason == "selected" else None,
    }


def snapshot_counts() -> dict:
    return {
        "source_records_consumed": 0,
        "count": 0,
        "text_bytes": 0,
        "duplicate": 0,
        "overlap": 0,
        "empty": 0,
        "null": 0,
    }


def count_snapshot_event(counts: dict, event: dict) -> None:
    counts["source_records_consumed"] += 1
    if event["reason"] == "selected":
        counts["count"] += 1
        counts["text_bytes"] += len(event["text"].encode())
    else:
        counts[event["reason"]] += 1


def snapshot_source(
    lock_path: str | Path,
    output: str | Path,
    cache_dir: str | Path,
    resume: bool = False,
) -> Path:
    """Publish a snapshot atomically. Interrupted work remains in .<output>.work.

    Bounded targets may exhaust early, reported as source_exhausted. Resource caps
    fail without publication and retain the journal. Resume replays the same
    pinned declaration; increasing limits requires a new declaration and lock.
    """
    import fcntl

    from klab.data.snapshot_writer import write_snapshot

    lock = verify_lock(lock_path)
    if lock["format"] != "klab-dataset-lock-v1":
        raise ValueError("unsupported lock format")
    source = _locked_source(
        lock,
        _lock_locations(Path(lock_path), lock)
        if lock["source"]["kind"] == "forge_files"
        else None,
    )
    output, cache = Path(output), Path(cache_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    work = output.with_name("." + output.name + ".work")
    if resume:
        if not work.is_dir() or work.is_symlink():
            raise ValueError("missing or unsafe resume journal")
    else:
        try:
            work.mkdir()
        except FileExistsError as error:
            journal = work / "journal.sqlite"
            resumable = (
                work.is_dir()
                and not work.is_symlink()
                and journal.is_file()
                and not journal.is_symlink()
            )
            guidance = (
                "inspect the retained journal, then use --resume with the original "
                "lock and --output for the same snapshot (busy and identity checks "
                "still apply)"
                if resumable
                else "no regular resume journal is observable; inspect retained "
                "evidence and the original failure. Another initializer may still "
                "be active; this does not authorize deleting the work path or "
                "clearing its guard"
            )
            raise FileExistsError(
                error.errno,
                f"snapshot work path already exists; {guidance}",
                error.filename,
            ) from error
    cache.mkdir(parents=True, exist_ok=True)
    guard = work / "guard"
    if guard.is_symlink():
        raise ValueError("unsafe journal guard")
    with guard.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return write_snapshot(lock, source, output, cache, work, resume)


def _manifest_path(config: Any) -> Path:
    if isinstance(config, str | Path):
        return Path(config)
    if getattr(config, "source", None) != "snapshot" or not getattr(
        config, "source_manifest_path", None
    ):
        raise ValueError(
            "runtime config requires source='snapshot' and source_manifest_path"
        )
    return Path(config.source_manifest_path)


_SNAPSHOT_SEAL = object()


@dataclass(frozen=True)
class _VerifiedSnapshot:
    key: tuple
    issuer_pid: int
    seal: object = field(repr=False)

    def __post_init__(self) -> None:
        if self.seal is not _SNAPSHOT_SEAL or self.issuer_pid != os.getpid():
            raise TypeError("snapshot evidence requires process-owned authentication")


@dataclass
class _SnapshotOperation:
    pid: int = field(default_factory=os.getpid)
    evidence: dict[str, _VerifiedSnapshot] = field(default_factory=dict)
    statistics: dict[str, int | float] = field(
        default_factory=lambda: {
            "calls": 0,
            "cold_verifications": 0,
            "reuse_hits": 0,
            "invalidations": 0,
            "verification_seconds": 0.0,
            "cold_verification_seconds": 0.0,
        }
    )


_snapshot_operation: ContextVar[_SnapshotOperation | None] = ContextVar(
    "snapshot_verification_operation", default=None
)
_snapshot_cold: ContextVar[bool] = ContextVar(
    "snapshot_verification_cold", default=False
)


@contextmanager
def snapshot_verification_operation(*, mode: VerificationMode = "verified_reuse"):
    """Authenticate once per unchanged tree, never beyond an operation or PID."""
    if mode not in {"cold", "verified_reuse"}:
        raise ValueError("invalid snapshot verification mode")
    current = _snapshot_operation.get()
    state_token = None
    if current is None or current.pid != os.getpid():
        state_token = _snapshot_operation.set(_SnapshotOperation())
    cold_token = _snapshot_cold.set(_snapshot_cold.get() or mode == "cold")
    try:
        yield
    finally:
        _snapshot_cold.reset(cold_token)
        if state_token is not None:
            _snapshot_operation.reset(state_token)


def snapshot_verification_statistics() -> dict[str, int | float] | None:
    state = _snapshot_operation.get()
    return (
        dict(state.statistics)
        if state is not None and state.pid == os.getpid()
        else None
    )


def _reject_snapshot_symlinks(path: Path) -> None:
    path = path.absolute()
    for member in (path, *path.parents):
        if member.is_symlink():
            raise ValueError(f"symlink snapshot path component: {member}")


def _snapshot_key(path: Path, manifest: dict) -> tuple:
    from klab.experiments.artifacts import _fingerprint

    _reject_snapshot_symlinks(path)
    return (
        str(path.absolute()),
        manifest["format"],
        manifest["manifest_sha256"],
        manifest["lock"]["lock_sha256"],
        _fingerprint(path.parent),
    )


def verify_snapshot(
    config: Any, *, verification_mode: VerificationMode | None = None
) -> dict:
    """Retain full-verifier evidence only inside the current native operation."""
    if verification_mode not in {None, "cold", "verified_reuse"}:
        raise ValueError("invalid snapshot verification mode")
    state = _snapshot_operation.get()
    if state is not None and state.pid != os.getpid():
        state = None
    started = monotonic()
    if state is not None:
        state.statistics["calls"] += 1
    locator = str(_manifest_path(config).absolute())
    from klab.data.acquisition_scope import verification_transport

    transport = verification_transport(Path(locator))
    if transport is not None:
        transport.before_verification(Path(locator))
    try:
        path, manifest, source = _snapshot_preflight(config)
        reuse = (
            state is not None
            and not _snapshot_cold.get()
            and verification_mode != "cold"
        )
        key = _snapshot_key(path, manifest)
        if reuse:
            evidence = state.evidence.get(locator)
            if evidence is not None and evidence.key != key:
                del state.evidence[locator]
                state.statistics["invalidations"] += 1
                evidence = None
            if (
                evidence is not None
                and evidence.seal is _SNAPSHOT_SEAL
                and evidence.issuer_pid == os.getpid()
            ):
                if _snapshot_key(path, manifest) != key:
                    raise ValueError("snapshot changed during verification")
                state.statistics["reuse_hits"] += 1
                return manifest
        cold_started = monotonic()
        if state is not None:
            state.statistics["cold_verifications"] += 1
        try:
            result = _verify_snapshot_cold(path, manifest, source)
        finally:
            if state is not None:
                state.statistics["cold_verification_seconds"] += (
                    monotonic() - cold_started
                )
        if _snapshot_key(path, manifest) != key:
            raise ValueError("snapshot changed during verification")
        if reuse:
            state.evidence[locator] = _VerifiedSnapshot(
                key, os.getpid(), _SNAPSHOT_SEAL
            )
        return result
    except BaseException:
        if state is not None:
            state.evidence.pop(locator, None)
        raise
    finally:
        if state is not None:
            state.statistics["verification_seconds"] += monotonic() - started


def _snapshot_preflight(config: Any) -> tuple[Path, dict, DatasetSource | None]:
    path = _manifest_path(config)
    _reject_snapshot_symlinks(path)
    manifest = _read(path, "manifest_sha256")
    if manifest["format"] != "klab-dataset-snapshot-v1":
        raise ValueError("unsupported snapshot format")
    lock = manifest["lock"]
    source = _verify_lock_payload(lock)
    if not isinstance(config, str | Path):
        if config.revision != source.revision or config.license != source.license:
            raise ValueError("snapshot config revision/license mismatch")
        if config.dataset_config not in {None, source.config}:
            raise ValueError("snapshot dataset configuration mismatch")
        for split in ("train", "validation"):
            configured = getattr(config, split + "_path", None)
            if split not in source.splits or configured is None:
                raise ValueError(
                    "snapshot runtime requires train and validation splits"
                )
            _reject_snapshot_symlinks(Path(configured))
            if Path(configured).resolve() != (path.parent / f"{split}.jsonl").resolve():
                raise ValueError(
                    f"snapshot source paths/provenance mismatch: configured {split} path differs"
                )
            forge._regular(Path(configured))
    if set(manifest["splits"]) != set(source.splits):
        raise ValueError("snapshot split mapping mismatch")
    expected = {
        path.name,
        "events.jsonl",
        "excluded.jsonl",
        *(split + ".jsonl" for split in source.splits),
    }
    if {p.name for p in path.parent.iterdir()} != expected:
        raise ValueError("snapshot inventory mismatch")
    for name in expected:
        forge._regular(path.parent / name)
    return path, manifest, source


def _verify_snapshot_cold(path: Path, manifest: dict, source: DatasetSource) -> dict:
    """Replay the existing scientific authority without memo-dependent semantics."""
    import tempfile

    lock = manifest["lock"]
    if sha256_file(path.parent / "events.jsonl") != manifest["events_sha256"]:
        raise ValueError("snapshot event digest mismatch")
    if manifest["snapshot_source_sha256"] != canonical_digest(lock["source"]):
        raise ValueError("snapshot source identity mismatch")
    excluded = manifest["excluded"]
    if (
        excluded["path"] != "excluded.jsonl"
        or sha256_file(path.parent / "excluded.jsonl") != excluded["sha256"]
    ):
        raise ValueError("snapshot exclusions digest mismatch")
    counts = {split: snapshot_counts() for split in source.splits}
    from klab.data.acquisition_scope import verification_transport

    transport = verification_transport(path.absolute())
    with tempfile.TemporaryDirectory(
        prefix="klab-verify-",
        dir=transport.scratch() if transport is not None else None,
    ) as temporary:
        db = sqlite3.connect(Path(temporary) / "seen.sqlite")
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE seen (digest TEXT, split TEXT, PRIMARY KEY(digest, split))"
            )
            from contextlib import ExitStack

            with ExitStack() as stack:
                handles = {
                    split: stack.enter_context(
                        (path.parent / f"{split}.jsonl").open("rb")
                    )
                    for split in source.splits
                }
                excluded_handle = stack.enter_context(
                    (path.parent / "excluded.jsonl").open("rb")
                )
                excluded_count = 0
                previous_priority = 0
                with (path.parent / "events.jsonl").open("rb") as events:
                    line_limit = source.resources.max_record_bytes + 512
                    for raw in iter(lambda: events.readline(line_limit + 1), b""):
                        if len(raw) > line_limit:
                            raise ValueError("snapshot event exceeds record byte bound")
                        if not raw.endswith(b"\n"):
                            raise ValueError("partial event record")
                        event = json.loads(raw)
                        split = event["split"]
                        priority = source.dedup.priority.index(split)
                        if (
                            priority < previous_priority
                            or type(event["ordinal"]) is not int
                            or counts[split]["source_records_consumed"]
                            >= source.resources.max_source_records
                            or event["ordinal"]
                            != counts[split]["source_records_consumed"]
                        ):
                            raise ValueError("event ordinal/order mismatch")
                        previous_priority = priority
                        reason = event["reason"]
                        if reason not in {
                            "selected",
                            "duplicate",
                            "overlap",
                            "empty",
                            "null",
                        }:
                            raise ValueError("unknown exclusion reason")
                        if not re.fullmatch(r"[0-9a-f]{64}", event["source_sha256"]):
                            raise ValueError("invalid source row digest")
                        if reason == "selected":
                            text = event["text"]
                            if not isinstance(text, str) or not text:
                                raise ValueError("invalid selected text")
                            wanted = (
                                canonical_json(
                                    {"ordinal": event["ordinal"], "text": text}
                                )
                                + b"\n"
                            )
                            if handles[split].readline(len(wanted) + 1) != wanted:
                                raise ValueError("snapshot selected row mismatch")
                            digest = hashlib.sha256(text.encode()).hexdigest()
                            prior = db.execute(
                                "SELECT split FROM seen WHERE digest=? ORDER BY (split=?) DESC LIMIT 1",
                                (digest, split),
                            ).fetchone()
                            if prior:
                                policy = (
                                    source.dedup.within_split
                                    if prior[0] == split
                                    else source.dedup.overlap
                                )
                                if policy != "keep":
                                    raise ValueError(
                                        "snapshot forbidden duplicate/overlap"
                                    )
                            db.execute(
                                "INSERT OR IGNORE INTO seen VALUES (?, ?)",
                                (digest, split),
                            )
                        elif event["text"] is not None:
                            raise ValueError("excluded record contains selected text")
                        if reason != "selected":
                            if (
                                reason == "duplicate"
                                and source.dedup.within_split != "exclude"
                            ):
                                raise ValueError(
                                    "excluded duplicate contradicts policy"
                                )
                            if (
                                reason == "overlap"
                                and source.dedup.overlap != "exclude"
                            ):
                                raise ValueError("excluded overlap contradicts policy")
                            wanted = (
                                canonical_json(
                                    {
                                        k: event[k]
                                        for k in ("split", "ordinal", "reason")
                                    }
                                )
                                + b"\n"
                            )
                            if excluded_handle.readline(len(wanted) + 1) != wanted:
                                raise ValueError("excluded row mismatch")
                            excluded_count += 1
                        counts[split].setdefault("text_bytes_limit", 0)
                        count_snapshot_event(counts[split], event)
                        if (
                            sum(item["text_bytes"] for item in counts.values())
                            > source.resources.max_text_bytes
                        ):
                            raise ValueError("snapshot exceeds text byte bound")
                if excluded_handle.read(1) or excluded_count != excluded["count"]:
                    raise ValueError("excluded count mismatch")
                for split, handle in handles.items():
                    details = manifest["splits"][split]
                    if handle.read(1) or any(
                        type(details.get(k, 0)) is not int or details.get(k, 0) != v
                        for k, v in counts[split].items()
                    ):
                        raise ValueError("snapshot counters/coverage mismatch")
                    if (
                        details["path"] != f"{split}.jsonl"
                        or sha256_file(path.parent / details["path"])
                        != details["sha256"]
                    ):
                        raise ValueError("snapshot split digest/path mismatch")
                    stop = details["stop_reason"]
                    if (
                        stop
                        not in {
                            "source_exhausted",
                            "document_target",
                        }
                        or type(details["source_exhausted"]) is not bool
                        or details["source_exhausted"] != (stop == "source_exhausted")
                    ):
                        raise ValueError("invalid source exhaustion claim")
                    if stop == "document_target" and (
                        source.selection.mode != "bounded"
                        or counts[split]["count"] != source.selection.documents[split]
                    ):
                        raise ValueError("invalid document target claim")
                    if (
                        counts[split]["source_records_consumed"]
                        > source.resources.max_source_records
                    ):
                        raise ValueError("snapshot exceeds source record bound")
                    if source.selection.mode == "bounded" and (
                        counts[split]["count"] > source.selection.documents[split]
                        or (
                            stop == "source_exhausted"
                            and counts[split]["count"]
                            >= source.selection.documents[split]
                        )
                    ):
                        raise ValueError("snapshot stop reason contradicts selection")
        finally:
            try:
                if transport is not None:
                    transport.observe()
            finally:
                db.close()
    if (
        sum(item["text_bytes"] for item in counts.values())
        > source.resources.max_text_bytes
    ):
        raise ValueError("snapshot exceeds text byte bound")
    if type(manifest["source_exhausted"]) is not bool or manifest[
        "source_exhausted"
    ] != all(s["source_exhausted"] for s in manifest["splits"].values()):
        raise ValueError("inconsistent full-source exhaustion claim")
    return manifest


def iter_snapshot(config: Any, split: str) -> Iterator[str]:
    manifest = verify_snapshot(config)
    if split not in manifest["splits"]:
        raise ValueError(f"unknown snapshot split: {split}")
    path = _manifest_path(config).parent / manifest["splits"][split]["path"]
    with path.open("rb") as handle:
        for raw in handle:
            yield json.loads(raw)["text"]
