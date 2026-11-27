"""A bounded reader and writer for immutable mixture segment ranges."""

from __future__ import annotations

import hashlib
import re
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.mixture_index import MAX_ROW_BYTES, rows
from klab.corpus.release_verify import current_settings
from klab.data.token_segments import (
    SEGMENT_DTYPES,
    TOKEN_SEGMENT_MANIFEST,
    read_token_segment,
)
from klab.data.verification import VerifiedFile, verify_file
from klab.owned_copy import owned_copy
from klab.safe_io import read_file_range, symlinked
from klab.verification_proofs import ProofStore, artifact_fingerprint

TRAIN_SCHEDULE = "train.schedule.bin"
PORTABLE_SEGMENTS = "token-segments"
SCHEDULE_RANGE = struct.Struct(
    "<IQI"
)  # segment ordinal, token offset, targets including EOS
_SHA256 = re.compile("[0-9a-f]{64}")
_FORMAT = "int32-segment-ranges"


def token_cache_root(release: Path, output_parent: Path) -> Path:
    from klab.corpus.preparation_inputs import corpus_work_root

    try:
        return corpus_work_root(release) / "cache" / "source-tokens"
    except ValueError:
        return output_parent / ".source-tokens"


def write_schedule(path: Path, records: Path, cache, eos: int) -> dict:
    """Resolve committed content keys after the cache's active segment seals."""
    from os import fsync

    segments = {}
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as stream:
        for row in rows(records):
            segment, offset, available = cache.segment_range(row["content_sha256"])
            count = row["tokens"]
            if not 1 <= count <= available + 1:
                raise ValueError("mixture token range exceeds the encoded document")
            ordinal = segments.setdefault(segment, len(segments))
            raw = SCHEDULE_RANGE.pack(ordinal, offset, count)
            stream.write(raw)
            digest.update(raw)
            size += len(raw)
        stream.flush()
        fsync(stream.fileno())
    # Any index used to resolve a range must have stayed fixed across the pass.
    cache.finish()
    return {
        "format": _FORMAT,
        "encoding": cache.path.name,
        "segments": list(segments),
        "eos": eos,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def require_schedule_generation(proof: VerifiedFile, descriptor: int | None = None):
    if artifact_fingerprint(proof.path if descriptor is None else descriptor) != (
        (".", *proof.fingerprint),
    ):
        raise ValueError(f"verified mixture schedule member changed: {proof.path}")


@contextmanager
def open_verified_schedule_member(proof: VerifiedFile):
    with proof.path.open("rb") as stream:
        require_schedule_generation(proof, stream.fileno())
        yield stream
        require_schedule_generation(proof, stream.fileno())


@dataclass(frozen=True)
class VerifiedSchedule:
    cache_root: Path
    schedule: VerifiedFile
    tokens: tuple[VerifiedFile, ...]
    indexes: tuple[VerifiedFile, ...]
    manifest_sha256: tuple[str, ...]
    manifest_sizes: tuple[int, ...]
    token_dtype: np.dtype
    eos: int
    target_count: int
    projection_files: tuple[VerifiedFile, ...] = ()

    def copy_dependencies(self, destination: Path) -> dict[str, VerifiedFile]:
        """Own just the referenced sealed segments, reusing authenticated bytes."""
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"token segment destination exists: {destination}")
        from klab.workspace_preflight import check_storage, require_storage

        size, files = self.dependency_footprint()
        require_storage(
            [
                check_storage(
                    destination.parent, projected_bytes=size, projected_inodes=files * 2
                )
            ]
        )
        copied = {}
        if self.projection_files:
            for proof in self.projection_files:
                name = proof.path.relative_to(self.cache_root).as_posix()
                copied[name] = owned_copy(
                    proof.path, destination / name, proof=proof
                ).proof
            return copied
        for tokens, index, digest in zip(
            self.tokens, self.indexes, self.manifest_sha256, strict=True
        ):
            for proof in (tokens, index):
                name = proof.path.relative_to(self.cache_root).as_posix()
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                copied[name] = owned_copy(proof.path, target, proof=proof).proof
            manifest = tokens.path.with_name(TOKEN_SEGMENT_MANIFEST)
            name = manifest.relative_to(self.cache_root).as_posix()
            copied[name] = owned_copy(
                manifest, destination / name, expected_sha256=digest
            ).proof
        return copied

    def dependency_footprint(self) -> tuple[int, int]:
        """Count the authenticated payload, index and manifest bytes and files."""
        if self.projection_files:
            return sum(proof.size_bytes for proof in self.projection_files), len(
                self.projection_files
            )
        return (
            sum(proof.size_bytes for proof in (*self.tokens, *self.indexes))
            + sum(self.manifest_sizes),
            3 * len(self.tokens),
        )

    def require_unchanged(self):
        for proof in self.projection_files or (
            self.schedule,
            *self.tokens,
            *self.indexes,
        ):
            require_schedule_generation(proof)

    def ranges(self):
        total = 0
        seen = set()
        width = self.token_dtype.itemsize
        with open_verified_schedule_member(self.schedule) as stream:
            while raw := stream.read(SCHEDULE_RANGE.size * 4096):
                for segment, offset, count in SCHEDULE_RANGE.iter_unpack(raw):
                    if (
                        segment >= len(self.tokens)
                        or not 1 <= count <= MAX_ROW_BYTES // 4
                        or offset + count - 1 > self.tokens[segment].size_bytes // width
                    ):
                        raise ValueError("invalid mixture token range")
                    total += count
                    seen.add(segment)
                    yield segment, offset, count
        if total != self.target_count or len(seen) != len(self.tokens):
            raise ValueError("mixture schedule totals or segment inventory differ")

    def validate_counts(self, lengths):
        for (_, _, count), expected in zip(self.ranges(), lengths, strict=True):
            if count != expected:
                raise ValueError("mixture schedule range differs from its record")

    def blocks(self, block_bytes: int, *, lengths=None):
        """Read scheduled ranges with bounded open files and append one EOS each."""
        width = self.token_dtype.itemsize
        limit = block_bytes // 4 * width
        eos = self.eos.to_bytes(width, "little")
        buffer = bytearray()
        streams = OrderedDict()

        def logical_tokens(values):
            if width == 4:
                return values
            widened = np.frombuffer(values, dtype=self.token_dtype).astype("<i4")
            return memoryview(widened).cast("B")

        def close_streams():
            with ExitStack() as opened:
                for _, stack in streams.values():
                    opened.callback(stack.close)

        with ExitStack() as closing:
            closing.callback(close_streams)
            ranges = self.ranges()
            closing.callback(ranges.close)
            entries = (
                ((item, item[2]) for item in ranges)
                if lengths is None
                else zip(ranges, lengths, strict=True)
            )
            for (segment, offset, count), expected in entries:
                if count != expected:
                    raise ValueError("mixture schedule range differs from its record")
                if len(buffer) + width * count > limit and buffer:
                    yield logical_tokens(buffer)
                    buffer = bytearray()
                if count > 1:
                    if segment not in streams:
                        if len(streams) == 16:
                            streams.popitem(last=False)[1][1].close()
                        with ExitStack() as opened:
                            stream = opened.enter_context(
                                open_verified_schedule_member(self.tokens[segment])
                            )
                            streams[segment] = stream, opened.pop_all()
                    streams.move_to_end(segment)
                    buffer.extend(
                        read_file_range(
                            streams[segment][0], offset * width, (count - 1) * width
                        )
                    )
                buffer.extend(eos)
            if buffer:
                yield logical_tokens(buffer)
            self.require_unchanged()


def _is_digest(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def schedule_metadata(receipt: dict) -> tuple[dict, int]:
    """Validate the logical schedule contract shared by whole and compact inputs."""
    schedule = receipt["token_schedule"]
    targets = sum(receipt["actual_target_tokens"].values())
    if (
        not isinstance(schedule, dict)
        or schedule.keys()
        != {"format", "encoding", "segments", "eos", "bytes", "sha256"}
        or schedule["format"] != _FORMAT
        or not _is_digest(schedule["encoding"])
        or type(schedule["eos"]) is not int
        or not 0 <= schedule["eos"] < 2**31
        or not _is_digest(schedule["sha256"])
        or not isinstance(schedule["segments"], list)
        or not 0 < len(schedule["segments"]) <= receipt["records"]
        or any(not _is_digest(segment) for segment in schedule["segments"])
        or len(set(schedule["segments"])) != len(schedule["segments"])
        or schedule["bytes"] != SCHEDULE_RANGE.size * receipt["records"]
        or receipt["train_tokens_bytes"] != 4 * targets
    ):
        raise ValueError("invalid mixture token schedule receipt")
    return schedule, targets


def verify_schedule(
    output: Path,
    cache_root: Path,
    receipt: dict,
    eos: int,
    *,
    memo: dict[object, VerifiedFile] | None = None,
    portable: bool = False,
) -> VerifiedSchedule:
    """Verify the schedule bytes and exact sealed dependencies, including unused bytes."""
    schedule, targets = schedule_metadata(receipt)
    if schedule["eos"] != eos:
        raise ValueError("invalid mixture token schedule receipt")
    path = output / TRAIN_SCHEDULE
    if symlinked(path):
        raise ValueError("unsafe mixture schedule path")
    schedule_file = verify_file(path, expected_sha256=schedule["sha256"], memo=memo)
    if schedule_file.size_bytes != schedule["bytes"]:
        raise ValueError("mixture schedule size differs from receipt")
    store = cache_root / "sealed" / schedule["encoding"] / "segments"
    if symlinked(store):
        raise ValueError("unsafe mixture token segment store")
    if portable:
        members = {
            store / segment / name
            for segment in schedule["segments"]
            for name in (TOKEN_SEGMENT_MANIFEST, "tokens.ids", "index.bin")
        }
        if {
            path
            for path in cache_root.rglob("*")
            if path.is_file() or path.is_symlink()
        } != members:
            raise ValueError("portable token segment inventory differs from schedule")
    settings = current_settings()
    proofs = settings.proof_store
    if proofs is None or not proofs.trusted(store):
        proofs = None if portable else ProofStore(cache_root)

    def check(segment):
        directory = store / segment
        manifest = read_token_segment(
            directory, schedule["encoding"], receipt["tokenizer_sha256"]
        )
        if canonical_digest(manifest) != segment:
            raise ValueError("mixture token segment manifest differs from its identity")
        files = []
        for name in ("tokens.ids", "index.bin"):
            entry = manifest["files"][name]
            proof = verify_file(
                directory / name,
                expected_sha256=entry["sha256"],
                memo=memo,
                proof_store=proofs,
                verification_mode=settings.verification_mode,
                chunk_size=4 * 1024**2,
            )
            if proof.size_bytes != entry["size"]:
                raise ValueError("mixture token segment size differs from its manifest")
            files.append(proof)
        encoded = canonical_json(manifest) + b"\n"
        return (
            *files,
            hashlib.sha256(encoded).hexdigest(),
            len(encoded),
            manifest["encoding"]["dtype"],
        )

    workers = min(8, settings.workers, len(schedule["segments"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(check, schedule["segments"], buffersize=workers))
    tokens, indexes, manifests, manifest_sizes, encodings = zip(*files, strict=True)
    dtype = SEGMENT_DTYPES[encodings[0]]
    if len(set(encodings)) != 1 or eos >= 2 ** (8 * dtype.itemsize):
        raise ValueError("invalid physical token encoding for the mixture schedule")
    return VerifiedSchedule(
        cache_root.resolve(),
        schedule_file,
        tokens,
        indexes,
        manifests,
        manifest_sizes,
        dtype,
        eos,
        targets,
    )
