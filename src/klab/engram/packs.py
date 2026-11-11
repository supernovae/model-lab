"""Versioned, content-addressed Engram pack artifacts."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Literal, cast

import numpy as np
from pydantic import (
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from safetensors import SafetensorError

from klab.address_hash import byte_scheme
from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from klab.config.models import StrictModel
from klab.engram.records import (
    MAX_RECORD_BYTES,
    KnowledgeRecord,
    _reject_constant,
    canonical_record_bytes,
    iter_knowledge_records,
)
from klab.model.portable_engram import load_portable_engram
from klab.result_publication import sync_directory
from klab.training.manifest import ArtifactIdentity, sha256_file


def _validate_model[T: StrictModel](model_type: type[T], value: object) -> T:
    try:
        return model_type.model_validate(value)
    except ValidationError as error:
        reasons = []
        for issue in error.errors(include_input=False):
            location = ".".join(str(part) for part in issue["loc"])
            reasons.append(f"{location or 'record'}: {issue['msg']}")
        raise ValueError("; ".join(reasons)) from error


PACK_FORMAT = "klab-engram-pack"
PACK_VERSION = 1
SEMANTIC_FORMAT = "klab-semantic-assets"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SEMANTIC_METADATA_BYTES = 64 * 1024 * 1024
_TENSOR_SCAN_BYTES = 8 * 1024 * 1024
_ALLOWED_MEMBERS = frozenset(
    {
        "manifest.json",
        "records.jsonl",
        "lexical.engram",
        "semantic_keys.safetensors",
        "semantic_values.safetensors",
        "semantic.json",
    }
)


def _strict_version_one(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("version must be integer 1")
    return value


class EncoderIdentity(StrictModel):
    name: StrictStr
    revision: StrictStr
    sha256: StrictStr

    @model_validator(mode="after")
    def _valid(self) -> EncoderIdentity:
        if not self.name.strip() or not self.revision.strip():
            raise ValueError("encoder name and revision must be nonblank")
        _require_sha(self.sha256, "encoder sha256")
        return self


class SemanticMetadata(StrictModel):
    format: Literal["klab-semantic-assets"]
    format_version: Literal[1]
    record_ids: tuple[StrictStr, ...]
    key_encoder: EncoderIdentity
    value_encoder: EncoderIdentity
    key_normalization: Literal["none", "l2"]

    @field_validator("format_version", mode="before")
    @classmethod
    def _strict_version(cls, value: object) -> object:
        return _strict_version_one(value)

    @model_validator(mode="after")
    def _valid(self) -> SemanticMetadata:
        if not self.record_ids or any(not value.strip() for value in self.record_ids):
            raise ValueError("semantic record_ids must be nonempty and nonblank")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("semantic record_ids must be unique")
        return self


class _ManifestFile(StrictModel):
    relative_path: StrictStr
    sha256: StrictStr
    size_bytes: StrictInt

    @model_validator(mode="after")
    def _valid(self) -> _ManifestFile:
        _validate_member_name(self.relative_path)
        if self.relative_path == "manifest.json":
            raise ValueError("manifest must not inventory itself")
        _require_sha(self.sha256, "file sha256")
        if self.size_bytes < 0:
            raise ValueError("file size_bytes must be nonnegative")
        return self


class _SourceSample(StrictModel):
    source: StrictStr
    revision: StrictStr | None

    @model_validator(mode="after")
    def _valid(self) -> _SourceSample:
        if not self.source.strip():
            raise ValueError("provenance source must be nonblank")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("provenance revision must be nonblank")
        return self


class _Provenance(StrictModel):
    licenses: tuple[StrictStr, ...]
    source_count: StrictInt
    source_sample: tuple[_SourceSample, ...]

    @model_validator(mode="after")
    def _valid(self) -> _Provenance:
        if self.source_count < 0:
            raise ValueError("source_count must be nonnegative")
        if any(not value.strip() for value in self.licenses):
            raise ValueError("licenses must be nonblank")
        if tuple(sorted(set(self.licenses))) != self.licenses:
            raise ValueError("licenses must be sorted and unique")
        if len(self.source_sample) > 16:
            raise ValueError("source_sample exceeds 16 entries")
        return self


class LexicalComponent(StrictModel):
    relative_path: Literal["lexical.engram"]
    format_version: StrictInt
    normalization: StrictStr
    hashing: StrictStr
    ngram_size: StrictInt
    table_size: StrictInt
    embedding_dim: StrictInt
    table_sha256: StrictStr

    @model_validator(mode="after")
    def _valid(self) -> LexicalComponent:
        if self.format_version != 1:
            raise ValueError("unsupported lexical format version")
        if min(self.ngram_size, self.table_size, self.embedding_dim) <= 0:
            raise ValueError("lexical dimensions and ngram_size must be positive")
        if self.normalization != "raw-utf8-v1" or self.hashing != byte_scheme(
            self.table_size
        ):
            raise ValueError("unsupported lexical addressing contract")
        _require_sha(self.table_sha256, "table_sha256")
        return self


class SemanticComponent(StrictModel):
    keys_path: Literal["semantic_keys.safetensors"]
    values_path: Literal["semantic_values.safetensors"]
    metadata_path: Literal["semantic.json"]
    entry_count: StrictInt
    key_dim: StrictInt
    memory_dim: StrictInt
    dtype: Literal["float32"]
    key_normalization: Literal["none", "l2"]
    key_encoder: EncoderIdentity
    value_encoder: EncoderIdentity
    space_id: StrictStr

    @model_validator(mode="after")
    def _valid(self) -> SemanticComponent:
        if min(self.entry_count, self.key_dim, self.memory_dim) <= 0:
            raise ValueError("semantic entry_count and dimensions must be positive")
        _require_sha(self.space_id, "space_id")
        return self


class EngramPackManifest(StrictModel):
    format: Literal["klab-engram-pack"]
    format_version: Literal[1]
    record_version: Literal[1]
    name: StrictStr
    namespace: StrictStr
    created_at: StrictStr
    record_count: StrictInt
    provenance: _Provenance
    lexical: LexicalComponent | None
    semantic: SemanticComponent | None
    files: tuple[_ManifestFile, ...]
    pack_id: StrictStr

    @field_validator("format_version", "record_version", mode="before")
    @classmethod
    def _strict_versions(cls, value: object) -> object:
        return _strict_version_one(value)

    @model_validator(mode="after")
    def _valid(self) -> EngramPackManifest:
        if not self.name.strip() or not self.namespace.strip():
            raise ValueError("name and namespace must be nonblank")
        if self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if len({entry.relative_path for entry in self.files}) != len(self.files):
            raise ValueError("files contains duplicate members")
        if tuple(sorted(self.files, key=lambda item: item.relative_path)) != self.files:
            raise ValueError("files must be sorted by relative_path")
        _require_sha(self.pack_id, "pack_id")
        _canonical_utc_timestamp(self.created_at)
        return self


@dataclass(frozen=True)
class PackVerificationError:
    field: str
    reason: str


@dataclass(frozen=True)
class PackVerificationReport:
    valid: bool
    pack_id: str | None
    errors: tuple[PackVerificationError, ...]
    files: tuple[ArtifactIdentity, ...]


@dataclass(frozen=True)
class EngramPack:
    root: Path
    manifest: EngramPackManifest
    semantic_metadata: SemanticMetadata | None


class PublishedPackDurabilityError(OSError):
    """The pack committed, but syncing its parent directory failed."""

    def __init__(self, path: Path, cause: OSError) -> None:
        self.path = path
        super().__init__(
            f"pack was published at {path}, but parent-directory durability is uncertain: {cause}"
        )


def _require_sha(value: str, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")


def _validate_member_name(value: str) -> None:
    if (
        value not in _ALLOWED_MEMBERS
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or Path(value).is_absolute()
    ):
        raise ValueError(f"unsafe or unsupported pack member name: {value!r}")


def _format_utc(value: datetime) -> str:
    base = value.strftime("%Y-%m-%dT%H:%M:%S")
    fraction = f".{value.microsecond:06d}" if value.microsecond else ""
    return f"{base}{fraction}Z"


def _utc_now() -> str:
    return _format_utc(datetime.now(UTC).replace(microsecond=0))


def _fraction_digits(value: str) -> int:
    match = re.search(r"[Tt ]\d{2}:\d{2}:\d{2}[.,](\d+)", value)
    return len(match.group(1)) if match else 0


def _canonical_utc_timestamp(value: str) -> str:
    if _fraction_digits(value) > 6:
        raise ValueError("timestamp precision finer than microseconds is unsupported")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at timestamp must include timezone")
    canonical = _format_utc(parsed.astimezone(UTC))
    if canonical != value:
        raise ValueError("created_at must be canonical UTC with Z suffix")
    return canonical


def _normalize_created_at(value: str | None) -> str:
    if value is None:
        return _utc_now()
    if _fraction_digits(value) > 6:
        raise ValueError("timestamp precision finer than microseconds is unsupported")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at timestamp must include timezone")
    return _format_utc(parsed.astimezone(UTC))


def _strict_json(data: bytes, label: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label}: {error}") from error


def _read_bounded(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return data


def _manifest_payload(manifest: EngramPackManifest) -> dict[str, object]:
    payload = manifest.model_dump(mode="json")
    payload.pop("pack_id")
    return payload


def _iter_record_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline(MAX_RECORD_BYTES + 2)
            if not line:
                break
            line_number += 1
            if len(line) > MAX_RECORD_BYTES + 1:
                raise ValueError(f"records.jsonl:{line_number}: line exceeds 1 MiB")
            if not line.endswith(b"\n"):
                raise ValueError(f"records.jsonl:{line_number}: missing final LF")
            yield line_number, line


def _manifest_from_bytes(path: Path) -> tuple[EngramPackManifest, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("manifest.json must be a regular nonsymlink file")
    data = _read_bounded(path, MAX_MANIFEST_BYTES, "manifest.json")
    raw = _strict_json(data, "manifest.json")
    if not isinstance(raw, dict):
        raise TypeError("manifest.json must contain an object")
    manifest = _validate_model(EngramPackManifest, raw)
    canonical = canonical_json(manifest.model_dump(mode="json")) + b"\n"
    if data != canonical:
        raise ValueError("manifest.json is not canonical JSON with final LF")
    if manifest.pack_id != canonical_digest(_manifest_payload(manifest)):
        raise ValueError("manifest pack_id self-digest mismatch")
    return manifest, data


def _validate_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("pack root must be a nonsymlink directory")
    return path.resolve(strict=True)


def _directory_members(root: Path) -> dict[str, Path]:
    members: dict[str, Path] = {}
    with os.scandir(root) as entries:
        for entry in entries:
            name = entry.name
            if name not in _ALLOWED_MEMBERS:
                raise ValueError(f"unexpected pack member: {name}")
            member = root / name
            if entry.is_symlink():
                raise ValueError(f"pack member is a symlink: {name}")
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"pack member is not a regular file: {name}")
            members[name] = member
    return members


def _load_semantic_metadata(path: Path) -> tuple[SemanticMetadata, bytes]:
    data = _read_bounded(path, MAX_SEMANTIC_METADATA_BYTES, "semantic.json")
    raw = _strict_json(data, "semantic.json")
    metadata = _validate_model(SemanticMetadata, raw)
    canonical = canonical_json(metadata.model_dump(mode="json")) + b"\n"
    if data != canonical:
        raise ValueError("semantic.json is not canonical JSON with final LF")
    return metadata, data


def _semantic_space_id(component: SemanticComponent | dict[str, object]) -> str:
    if isinstance(component, SemanticComponent):
        identity: dict[str, object] = {
            "key_encoder": component.key_encoder.model_dump(mode="json"),
            "value_encoder": component.value_encoder.model_dump(mode="json"),
            "key_dim": component.key_dim,
            "memory_dim": component.memory_dim,
            "dtype": component.dtype,
            "key_normalization": component.key_normalization,
        }
    else:
        raw = cast(dict[str, Any], component)
        identity = {
            "key_encoder": raw["key_encoder"],
            "value_encoder": raw["value_encoder"],
            "key_dim": raw["key_dim"],
            "memory_dim": raw["memory_dim"],
            "dtype": raw["dtype"],
            "key_normalization": raw["key_normalization"],
        }
    return canonical_digest(identity)


def _scan_tensor(path: Path, name: str, *, l2: bool = False) -> tuple[int, int]:
    from safetensors import safe_open

    with safe_open(path, framework="np", device="cpu") as handle:
        if set(handle.keys()) != {name}:
            raise ValueError(f"{path.name} must contain exactly tensor {name!r}")
        tensor_slice = handle.get_slice(name)
        shape = tuple(tensor_slice.get_shape())
        dtype = tensor_slice.get_dtype()
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"{path.name}:{name} must have positive rank-2 shape")
        if dtype != "F32":
            raise ValueError(f"{path.name}:{name} must be FP32, got {dtype}")
        rows, columns = shape
        max_cells = _TENSOR_SCAN_BYTES // np.dtype(np.float32).itemsize
        norms = np.zeros(rows, dtype=np.float64) if l2 else None
        row_step = max(1, max_cells // min(columns, max_cells))
        column_step = min(columns, max_cells)
        for row_start in range(0, rows, row_step):
            row_end = min(rows, row_start + row_step)
            for column_start in range(0, columns, column_step):
                column_end = min(columns, column_start + column_step)
                tile = tensor_slice[row_start:row_end, column_start:column_end]
                if tile.dtype != np.float32 or not tile.flags.c_contiguous:
                    tile = np.ascontiguousarray(tile, dtype=np.float32)
                if tile.nbytes > _TENSOR_SCAN_BYTES:
                    raise ValueError("tensor scan tile exceeds 8 MiB")
                if not np.isfinite(tile).all():
                    raise ValueError(f"{path.name}:{name} contains nonfinite values")
                if norms is not None:
                    norms[row_start:row_end] += np.einsum(
                        "ij,ij->i", tile, tile, dtype=np.float64
                    )
        if norms is not None:
            norms = np.sqrt(norms)
            if np.any(norms == 0):
                raise ValueError("l2-normalized semantic keys contain a zero vector")
            if not np.all(np.abs(norms - 1.0) <= 1e-5):
                raise ValueError("semantic key norms violate l2 declaration")
        return rows, columns


def _scan_semantic_group(
    keys_path: Path,
    values_path: Path,
    metadata: SemanticMetadata,
) -> SemanticComponent:
    key_rows, key_dim = _scan_tensor(
        keys_path, "keys", l2=metadata.key_normalization == "l2"
    )
    value_rows, memory_dim = _scan_tensor(values_path, "values")
    if key_rows != value_rows or key_rows != len(metadata.record_ids):
        raise ValueError("semantic tensor row counts and record_ids must agree")
    raw = {
        "keys_path": "semantic_keys.safetensors",
        "values_path": "semantic_values.safetensors",
        "metadata_path": "semantic.json",
        "entry_count": key_rows,
        "key_dim": key_dim,
        "memory_dim": memory_dim,
        "dtype": "float32",
        "key_normalization": metadata.key_normalization,
        "key_encoder": metadata.key_encoder.model_dump(mode="json"),
        "value_encoder": metadata.value_encoder.model_dump(mode="json"),
        "space_id": "0" * 64,
    }
    raw["space_id"] = _semantic_space_id(raw)
    return _validate_model(SemanticComponent, raw)


def _copy_regular(source: Path, destination: Path) -> None:
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"input must be a regular nonsymlink file: {source}")
    with source.open("rb") as source_handle, destination.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())


def _write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_parent_if_supported(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is None:
            raise OSError(errno.ENOTSUP, "renamex_np is unavailable")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError(
            errno.ENOTSUP, f"atomic no-replace rename unsupported on {sys.platform}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _read_semantic_input(path: Path) -> SemanticMetadata:
    if path.is_symlink() or not path.is_file():
        raise ValueError("semantic metadata input must be a regular nonsymlink file")
    data = _read_bounded(path, MAX_SEMANTIC_METADATA_BYTES, "semantic metadata input")
    raw = _strict_json(data, "semantic metadata input")
    return _validate_model(SemanticMetadata, raw)


def compile_pack(
    source: Path,
    output: Path,
    *,
    name: str,
    namespace: str,
    default_license: str | None = None,
    source_name: str | None = None,
    source_revision: str | None = None,
    created_at: str | None = None,
    lexical_package: Path | None = None,
    semantic_keys: Path | None = None,
    semantic_values: Path | None = None,
    semantic_metadata: Path | None = None,
) -> EngramPackManifest:
    """Compile local source records and supplied assets into a new immutable pack."""
    if (
        semantic_keys is not None,
        semantic_values is not None,
        semantic_metadata is not None,
    ).count(True) not in {0, 3}:
        raise ValueError("semantic keys, values and metadata must be supplied together")
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"pack output already exists: {output}")
    if not name.strip() or not namespace.strip():
        raise ValueError("pack name and namespace must be nonblank")
    if source_revision is not None and source_name is None:
        raise ValueError("source_revision requires source_name")
    if not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("record input must be a regular nonsymlink file")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    committed = False
    try:
        ids: set[str] = set()
        licenses: set[str] = set()
        sources: set[tuple[str, str | None]] = set()
        record_count = 0
        records_path = temporary / "records.jsonl"
        with records_path.open("xb") as records_handle:
            for record in iter_knowledge_records(
                source,
                namespace=namespace,
                default_license=default_license,
                source_name=source_name,
                source_revision=source_revision,
            ):
                line = canonical_record_bytes(record) + b"\n"
                if len(line) > 1024 * 1024 + 1:
                    raise ValueError(f"canonical record {record.id!r} exceeds 1 MiB")
                records_handle.write(line)
                ids.add(record.id)
                licenses.add(record.license)
                if record.source is not None:
                    sources.add((record.source, record.source_revision))
                record_count += 1
            records_handle.flush()
            os.fsync(records_handle.fileno())
        if record_count == 0:
            raise ValueError("input contains no knowledge records")

        lexical_component = None
        if lexical_package is not None:
            lexical_copy = temporary / "lexical.engram"
            _copy_regular(lexical_package, lexical_copy)
            portable = load_portable_engram(lexical_copy)
            lexical_component = _validate_model(
                LexicalComponent,
                {
                    "relative_path": "lexical.engram",
                    **portable.manifest.as_dict(),
                },
            )
            del portable

        semantic_component = None
        if (
            semantic_keys is not None
            and semantic_values is not None
            and semantic_metadata is not None
        ):
            metadata = _read_semantic_input(semantic_metadata)
            missing_ids = [
                record_id for record_id in metadata.record_ids if record_id not in ids
            ]
            if missing_ids:
                raise ValueError(
                    f"semantic metadata references unknown record IDs: {', '.join(missing_ids[:8])}"
                )
            keys_copy = temporary / "semantic_keys.safetensors"
            values_copy = temporary / "semantic_values.safetensors"
            _copy_regular(semantic_keys, keys_copy)
            _copy_regular(semantic_values, values_copy)
            metadata_path = temporary / "semantic.json"
            _write_file(
                metadata_path, canonical_json(metadata.model_dump(mode="json")) + b"\n"
            )
            semantic_component = _scan_semantic_group(keys_copy, values_copy, metadata)

        payload_paths = [temporary / "records.jsonl"]
        if lexical_component is not None:
            payload_paths.append(temporary / "lexical.engram")
        if semantic_component is not None:
            payload_paths.extend(
                [
                    temporary / "semantic_keys.safetensors",
                    temporary / "semantic_values.safetensors",
                    temporary / "semantic.json",
                ]
            )
        files = tuple(
            _validate_model(
                _ManifestFile,
                {
                    "relative_path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                },
            )
            for path in sorted(payload_paths, key=lambda item: item.name)
        )
        sorted_sources = sorted(
            sources, key=lambda pair: (pair[0], pair[1] is not None, pair[1] or "")
        )
        provenance = _validate_model(
            _Provenance,
            {
                "licenses": tuple(sorted(licenses)),
                "source_count": len(sources),
                "source_sample": tuple(
                    {"source": source_value, "revision": revision}
                    for source_value, revision in sorted_sources[:16]
                ),
            },
        )
        raw_payload: dict[str, object] = {
            "format": PACK_FORMAT,
            "format_version": PACK_VERSION,
            "record_version": 1,
            "name": name,
            "namespace": namespace,
            "created_at": _normalize_created_at(created_at),
            "record_count": record_count,
            "provenance": provenance.model_dump(mode="json"),
            "lexical": lexical_component.model_dump(mode="json")
            if lexical_component
            else None,
            "semantic": semantic_component.model_dump(mode="json")
            if semantic_component
            else None,
            "files": [entry.model_dump(mode="json") for entry in files],
        }
        raw_payload["pack_id"] = canonical_digest(raw_payload)
        manifest = _validate_model(EngramPackManifest, raw_payload)
        manifest_path = temporary / "manifest.json"
        _write_file(
            manifest_path, canonical_json(manifest.model_dump(mode="json")) + b"\n"
        )
        sync_directory(temporary)

        report = verify_pack(temporary, expected_pack_id=manifest.pack_id)
        if not report.valid:
            raise ValueError(
                "compiled pack failed full verification: "
                + "; ".join(f"{item.field}: {item.reason}" for item in report.errors)
            )
        _rename_noreplace(temporary, output)
        committed = True
        try:
            _fsync_parent_if_supported(output.parent)
        except OSError as error:
            raise PublishedPackDurabilityError(output, error) from error
        return manifest
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _inspect_manifest(path: Path) -> tuple[Path, EngramPackManifest]:
    root = _validate_root(path)
    manifest, _ = _manifest_from_bytes(root / "manifest.json")
    return root, manifest


def inspect_pack(path: Path) -> dict[str, object]:
    """Read manifest identity and schema without verifying any payload bytes."""
    _, manifest = _inspect_manifest(path)
    return {
        **manifest.model_dump(mode="json"),
        "verification_status": "not_verified",
    }


def _verify_internal(
    path: Path,
    *,
    expected_pack_id: str | None,
) -> tuple[
    PackVerificationReport, EngramPackManifest | None, SemanticMetadata | None, set[str]
]:
    errors: list[PackVerificationError] = []
    verified: list[ArtifactIdentity] = []
    manifest: EngramPackManifest | None = None
    semantic_metadata_value: SemanticMetadata | None = None
    record_ids: set[str] = set()
    pack_id: str | None = None
    try:
        root = _validate_root(path)
        members = _directory_members(root)
        manifest, _ = _manifest_from_bytes(root / "manifest.json")
        pack_id = manifest.pack_id
        if expected_pack_id is not None and expected_pack_id != pack_id:
            errors.append(PackVerificationError("expected_pack_id", "pack ID mismatch"))
        inventory = {identity.relative_path: identity for identity in manifest.files}
        expected_members = set(inventory) | {"manifest.json"}
        if set(members) != expected_members:
            errors.append(
                PackVerificationError(
                    "files",
                    f"payload inventory mismatch (actual={sorted(members)}, expected={sorted(expected_members)})",
                )
            )
        if "records.jsonl" not in inventory:
            errors.append(PackVerificationError("files", "records.jsonl is required"))
        if (manifest.lexical is None) != ("lexical.engram" not in inventory):
            errors.append(
                PackVerificationError(
                    "lexical", "component and payload inventory disagree"
                )
            )
        semantic_names = {
            "semantic_keys.safetensors",
            "semantic_values.safetensors",
            "semantic.json",
        }
        if (manifest.semantic is None) != (not semantic_names.intersection(inventory)):
            errors.append(
                PackVerificationError(
                    "semantic", "component and payload inventory disagree"
                )
            )
        if manifest.semantic is not None and not semantic_names.issubset(inventory):
            errors.append(
                PackVerificationError(
                    "semantic", "semantic payload group is incomplete"
                )
            )
        if set(inventory) - (_ALLOWED_MEMBERS - {"manifest.json"}):
            errors.append(
                PackVerificationError(
                    "files", "manifest inventories unsupported payload names"
                )
            )

        for relative_path, identity in sorted(inventory.items()):
            member = members.get(relative_path)
            if member is None:
                continue
            try:
                info = member.stat(follow_symlinks=False)
                if info.st_size != identity.size_bytes:
                    raise ValueError("size mismatch")
                digest = sha256_file(member)
                if digest != identity.sha256:
                    raise ValueError("SHA-256 mismatch")
                verified.append(ArtifactIdentity(relative_path, digest, info.st_size))
            except (OSError, ValueError) as error:
                errors.append(PackVerificationError(relative_path, str(error)))

        records_path = members.get("records.jsonl")
        if records_path is not None:
            try:
                observed_count = 0
                licenses: set[str] = set()
                sources: set[tuple[str, str | None]] = set()
                for line_number, line in _iter_record_lines(records_path):
                    parsed = _strict_json(line[:-1], f"records.jsonl:{line_number}")
                    record = _validate_model(KnowledgeRecord, parsed)
                    if line != canonical_record_bytes(record) + b"\n":
                        raise ValueError(f"line {line_number} is not canonical")
                    if record.id in record_ids:
                        raise ValueError(
                            f"line {line_number} has duplicate id {record.id!r}"
                        )
                    record_ids.add(record.id)
                    licenses.add(record.license)
                    if record.source is not None:
                        sources.add((record.source, record.source_revision))
                    observed_count += 1
                if observed_count != manifest.record_count:
                    errors.append(
                        PackVerificationError(
                            "record_count", "records.jsonl count mismatch"
                        )
                    )
                for line_number, line in _iter_record_lines(records_path):
                    raw_value = _strict_json(line[:-1], f"records.jsonl:{line_number}")
                    raw = cast(dict[str, Any], raw_value)
                    negative_ids = tuple(raw["hard_negative_ids"])
                    missing = [
                        value for value in negative_ids if value not in record_ids
                    ]
                    if missing:
                        raise ValueError(
                            f"line {line_number}: record {raw['id']!r} "
                            "has dangling hard negatives"
                        )
                sorted_sources = sorted(
                    sources,
                    key=lambda pair: (pair[0], pair[1] is not None, pair[1] or ""),
                )
                expected_sample = [
                    {"source": source_value, "revision": revision}
                    for source_value, revision in sorted_sources[:16]
                ]
                provenance = manifest.provenance
                if sorted(licenses) != list(provenance.licenses):
                    errors.append(
                        PackVerificationError(
                            "provenance.licenses", "license summary mismatch"
                        )
                    )
                if len(sources) != provenance.source_count or expected_sample != [
                    item.model_dump(mode="json") for item in provenance.source_sample
                ]:
                    errors.append(
                        PackVerificationError("provenance", "source summary mismatch")
                    )
            except (OSError, TypeError, KeyError, ValueError) as error:
                errors.append(PackVerificationError("records.jsonl", str(error)))

        if manifest.lexical is not None and "lexical.engram" in members:
            try:
                lexical = manifest.lexical
                package = load_portable_engram(
                    members["lexical.engram"],
                    expected_shape=(lexical.table_size, lexical.embedding_dim),
                    expected_ngram_size=lexical.ngram_size,
                )
                actual = package.manifest.as_dict()
                expected = {
                    key: value
                    for key, value in lexical.model_dump(mode="json").items()
                    if key != "relative_path"
                }
                if actual != expected:
                    raise ValueError("lexical descriptor differs from embedded package")
                del package
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                KeyError,
                EOFError,
                UnpicklingError,
            ) as error:
                errors.append(PackVerificationError("lexical", str(error)))

        if manifest.semantic is not None and semantic_names.issubset(members):
            try:
                semantic_metadata_value, _ = _load_semantic_metadata(
                    members["semantic.json"]
                )
                component = manifest.semantic
                if semantic_metadata_value.key_encoder != component.key_encoder:
                    raise ValueError(
                        "key encoder identity differs from semantic metadata"
                    )
                if semantic_metadata_value.value_encoder != component.value_encoder:
                    raise ValueError(
                        "value encoder identity differs from semantic metadata"
                    )
                if (
                    semantic_metadata_value.key_normalization
                    != component.key_normalization
                ):
                    raise ValueError("key normalization differs from semantic metadata")
                missing = [
                    value
                    for value in semantic_metadata_value.record_ids
                    if value not in record_ids
                ]
                if missing:
                    raise ValueError("semantic metadata references unknown record IDs")
                actual_component = _scan_semantic_group(
                    members["semantic_keys.safetensors"],
                    members["semantic_values.safetensors"],
                    semantic_metadata_value,
                )
                if actual_component != component:
                    raise ValueError(
                        "semantic component descriptor differs from payload"
                    )
                if component.space_id != _semantic_space_id(component):
                    raise ValueError("semantic space_id mismatch")
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                KeyError,
                SafetensorError,
            ) as error:
                errors.append(PackVerificationError("semantic", str(error)))

    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        errors.append(PackVerificationError("manifest_or_root", str(error)))
    report = PackVerificationReport(not errors, pack_id, tuple(errors), tuple(verified))
    return report, manifest, semantic_metadata_value, record_ids


def verify_pack(
    path: Path,
    *,
    expected_pack_id: str | None = None,
) -> PackVerificationReport:
    """Verify identity, exact inventory, canonical records and component payloads."""
    report, _, _, _ = _verify_internal(path, expected_pack_id=expected_pack_id)
    return report


def load_pack(
    path: Path,
    *,
    expected_pack_id: str | None = None,
) -> EngramPack:
    """Return a verified immutable descriptor without retaining arrays or records."""
    report, manifest, metadata, _ = _verify_internal(
        path, expected_pack_id=expected_pack_id
    )
    if not report.valid or manifest is None:
        detail = "; ".join(f"{item.field}: {item.reason}" for item in report.errors)
        raise ValueError(f"invalid Engram pack: {detail}")
    return EngramPack(_validate_root(path), manifest, metadata)
