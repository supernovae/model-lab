"""Bounded previews of explicitly pinned acquired JSONL bytes, never admission."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys


class Pin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def relative(self):
        p = Path(self.path)
        if (
            p.is_absolute()
            or not p.parts
            or ".." in p.parts
            or p.as_posix() != self.path
        ):
            raise ValueError("pin path must be a canonical relative path")
        return self


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_field: str = Field(default="text", min_length=1, max_length=128)
    manifest: Pin
    data: Pin

    @model_validator(mode="after")
    def snapshot_paths(self):
        parent = Path(self.manifest.path).parent
        if (
            Path(self.manifest.path).name != "manifest.json"
            or parent.name != self.snapshot_sha256
            or not Path(self.data.path).is_relative_to(parent / "files")
        ):
            raise ValueError("pins must identify one acquired snapshot")
        return self


class Preview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format: Literal["klab-corpus-preview-v1"]
    seed: str = Field(min_length=1, max_length=128)
    sources: list[Source] = Field(min_length=1, max_length=64)
    max_read_bytes: int = Field(gt=0, le=2**63 - 1)
    max_record_bytes: int = Field(gt=0, le=8 * 1024**2)
    samples_per_source: int = Field(gt=0, le=128)
    max_excerpt_bytes: int = Field(gt=0, le=4096)
    max_output_bytes: int = Field(ge=8192, le=4 * 1024**2)
    max_wall_seconds: int = Field(gt=0, le=86400)

    @model_validator(mode="after")
    def unique(self):
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate source")
        paths = [p.path for s in self.sources for p in (s.manifest, s.data)]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate input path")
        return self


def _regular(path: Path) -> Path:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked preview input or output")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("preview input must be a regular file")
    return path


class Reader:
    """Account every requested input byte, including the declaration and manifests."""

    def __init__(self, maximum: int, deadline: float, initial: int):
        self.maximum, self.deadline, self.used = maximum, deadline, initial

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("preview wall deadline reached")
        if self.used > self.maximum:
            raise ValueError("preview aggregate read budget exceeded")

    def read(self, stream, count: int, *, line: bool = False) -> bytes:
        self.check()
        allowed = min(count, self.maximum - self.used)
        if allowed <= 0:
            raise ValueError("preview aggregate read budget exhausted")
        data = stream.readline(allowed) if line else stream.read(allowed)
        self.used += len(data)
        self.check()
        return data


def _scan(root: Path, pin: Pin, reader: Reader, consume=None, record_limit=0):
    path = _regular(root / pin.path)
    if not path.resolve().is_relative_to(root):
        raise ValueError("preview input escapes root")
    digest = hashlib.sha256()
    # Unbuffered reads make the byte counter include all physical API reads.
    with path.open("rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != pin.bytes:
            raise ValueError("preview input size differs from pin")
        remaining, physical_line = pin.bytes, 0
        pending = bytearray()
        while remaining:
            raw = reader.read(stream, min(remaining, 1024**2))
            if not raw:
                raise ValueError("preview input truncated")
            digest.update(raw)
            remaining -= len(raw)
            if consume:
                pending.extend(raw)
                start = 0
                while (end := pending.find(b"\n", start)) >= 0:
                    if end + 1 - start > record_limit:
                        raise ValueError("preview record-byte cap exceeded")
                    physical_line += 1
                    consume(bytes(pending[start : end + 1]), physical_line)
                    start = end + 1
                del pending[:start]
                if len(pending) > record_limit:
                    raise ValueError("preview record-byte cap exceeded")
        if consume and pending:
            physical_line += 1
            consume(bytes(pending), physical_line)
        after = os.fstat(stream.fileno())
    reader.check()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(path.stat()):
        raise ValueError("preview input changed while reading")
    if digest.hexdigest() != pin.sha256:
        raise ValueError("preview input SHA256 differs from pin")


def _source(root: Path, source: Source, spec: Preview, reader: Reader):
    _scan(root, source.manifest, reader)
    chosen = []
    rows, blanks, previous = 0, 0, -1

    def consume(raw, line):
        nonlocal rows, blanks, previous
        if not raw.strip():
            blanks += 1
            return
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or not isinstance(
            value.get(source.text_field), str
        ):
            raise TypeError(f"preview row must contain {source.text_field} as a string")
        envelope = value.pop("_klab_source", None)
        if not isinstance(envelope, dict):
            raise TypeError("preview row lacks acquired provenance")
        if "source_row_index" not in envelope:
            raise ValueError("unsupported acquisition envelope: no source_row_index")
        index = envelope["source_row_index"]
        digest = canonical_digest(value)
        if (
            type(index) is not int
            or index <= previous
            or envelope.get("source_row_sha256") != digest
        ):
            raise ValueError("preview row provenance mismatch or unordered index")
        previous = index
        rows += 1
        rank = hashlib.sha256(
            canonical_json(
                [spec.seed, source.source_id, source.snapshot_sha256, index, digest]
            )
        ).hexdigest()
        key = (rank, index)
        if len(chosen) < spec.samples_per_source or key < chosen[-1][0]:
            text = value[source.text_field].encode("utf-8")
            item = {
                "source_row_index": index,
                "source_row_sha256": digest,
                "physical_line": line,
                "text_bytes": len(text),
                "excerpt": text[: spec.max_excerpt_bytes].decode(
                    "utf-8", errors="ignore"
                ),
                "truncated": len(text) > spec.max_excerpt_bytes,
            }
            chosen.append((key, item))
            chosen.sort(key=lambda item: item[0])
            del chosen[spec.samples_per_source :]

    _scan(root, source.data, reader, consume, spec.max_record_bytes)
    return {
        "source_id": source.source_id,
        "text_field": source.text_field,
        "snapshot_sha256": source.snapshot_sha256,
        "manifest": source.manifest.model_dump(),
        "data": source.data.model_dump(),
        "rows": rows,
        "blank_lines": blanks,
        "hashes_verified": True,
        "items": [item for _, item in chosen],
    }


def preview(declaration: Path, input_root: Path, output: Path) -> dict:
    started = time.monotonic()
    _regular(declaration)
    if declaration.stat().st_size > 65536:
        raise ValueError("preview declaration exceeds 64 KiB")
    with declaration.open("rb", buffering=0) as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("preview declaration exceeds 64 KiB")

    spec = Preview.model_validate(
        json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    )
    reader = Reader(spec.max_read_bytes, started + spec.max_wall_seconds, len(raw))
    reader.check()
    # Refuse before scanning when even one complete pass cannot fit.
    if (
        len(raw) + sum(p.bytes for s in spec.sources for p in (s.manifest, s.data))
        > spec.max_read_bytes
    ):
        raise ValueError("pinned input bytes exceed aggregate read budget")
    if any(
        p.is_symlink()
        for p in (input_root, *input_root.parents, output, *output.parents)
    ):
        raise ValueError("symlinked preview root or output")
    root = input_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("input root must be a directory")
    if any(
        output.resolve().is_relative_to(root / Path(s.manifest.path).parent)
        for s in spec.sources
    ):
        raise ValueError("preview output must be outside acquired snapshots")
    output.mkdir()  # Exclusive: an interrupted or previous preview is never overwritten.
    base = {
        "format": "klab-corpus-preview-receipt-v1",
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec.model_dump(),
        "verification_scope": "pinned-file-bytes-and-row-envelopes",
        "admission": False,
    }
    written = 0

    def write(name, payload):
        nonlocal written
        data = canonical_json(payload) + b"\n"
        reserve = 0 if name == "failure.json" else 4096
        if written + len(data) + reserve > spec.max_output_bytes:
            raise ValueError("preview output-byte cap exceeded")
        with (output / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        written += len(data)

    write("started.json", {**base, "status": "incomplete"})
    results = []
    try:
        for source in spec.sources:
            results.append(_source(root, source, spec, reader))
        reader.check()
        report = {
            **base,
            "status": "complete",
            "read_bytes": reader.used,
            "elapsed_seconds": time.monotonic() - started,
            "sources": results,
        }
        write("report.json", report)
        return {"output": str(output), "status": "complete", "read_bytes": reader.used}
    except BaseException as error:
        write(
            "failure.json",
            {
                "status": "incomplete",
                "read_bytes": reader.used,
                "elapsed_seconds": time.monotonic() - started,
                "completed_source_count": len(results),
                "completed_sources_sha256": hashlib.sha256(
                    canonical_json([s["source_id"] for s in results])
                ).hexdigest(),
                # Bound UTF-8 bytes, including worst-case JSON escaping (6x),
                # so even 64 long source IDs cannot consume the 4096-byte reserve.
                "error_type": type(error).__name__[:128],
                "error": str(error)
                .encode("utf-8")[:384]
                .decode("utf-8", errors="ignore"),
            },
        )
        raise
