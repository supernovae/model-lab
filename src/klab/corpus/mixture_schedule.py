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

from klab.canonical import canonical_digest
from klab.corpus.mixture_index import MAX_ROW_BYTES, rows
from klab.corpus.release_verify import current_settings
from klab.data.token_segments import read_token_segment
from klab.data.verification import VerifiedFile, verify_file
from klab.safe_io import read_file_range, symlinked
from klab.verification_proofs import ProofStore, artifact_fingerprint

TRAIN_SCHEDULE = "train.schedule.bin"
SCHEDULE_RANGE = struct.Struct(
    "<IQI"
)  # segment ordinal, token offset, targets including EOS
_TOKEN = struct.Struct("<i")
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
    schedule: VerifiedFile
    tokens: tuple[VerifiedFile, ...]
    indexes: tuple[VerifiedFile, ...]
    eos: int
    target_count: int

    def require_unchanged(self):
        for proof in (self.schedule, *self.tokens, *self.indexes):
            require_schedule_generation(proof)

    def ranges(self):
        total = 0
        seen = set()
        with open_verified_schedule_member(self.schedule) as stream:
            while raw := stream.read(SCHEDULE_RANGE.size * 4096):
                for segment, offset, count in SCHEDULE_RANGE.iter_unpack(raw):
                    if (
                        segment >= len(self.tokens)
                        or not 1 <= count <= MAX_ROW_BYTES // 4
                        or offset + count - 1 > self.tokens[segment].size_bytes // 4
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
        eos = _TOKEN.pack(self.eos)
        buffer = bytearray()
        streams = OrderedDict()

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
                if len(buffer) + 4 * count > block_bytes and buffer:
                    yield buffer
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
                            streams[segment][0], offset * 4, (count - 1) * 4
                        )
                    )
                buffer.extend(eos)
            if buffer:
                yield buffer
            self.require_unchanged()


def _is_digest(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def verify_schedule(
    output: Path, cache_root: Path, receipt: dict, eos: int
) -> VerifiedSchedule:
    """Verify the schedule bytes and exact sealed dependencies, including unused bytes."""
    schedule = receipt["token_schedule"]
    targets = sum(receipt["actual_target_tokens"].values())
    if (
        not isinstance(schedule, dict)
        or schedule.keys()
        != {"format", "encoding", "segments", "eos", "bytes", "sha256"}
        or schedule["format"] != _FORMAT
        or not _is_digest(schedule["encoding"])
        or type(schedule["eos"]) is not int
        or schedule["eos"] != eos
        or not 0 <= eos < 2**31
        or not _is_digest(schedule["sha256"])
        or not isinstance(schedule["segments"], list)
        or not 0 < len(schedule["segments"]) <= receipt["records"]
        or any(not _is_digest(segment) for segment in schedule["segments"])
        or len(set(schedule["segments"])) != len(schedule["segments"])
        or schedule["bytes"] != SCHEDULE_RANGE.size * receipt["records"]
        or receipt["train_tokens_bytes"] != 4 * targets
    ):
        raise ValueError("invalid mixture token schedule receipt")
    path = output / TRAIN_SCHEDULE
    if symlinked(path):
        raise ValueError("unsafe mixture schedule path")
    schedule_file = verify_file(path, expected_sha256=schedule["sha256"])
    if schedule_file.size_bytes != schedule["bytes"]:
        raise ValueError("mixture schedule size differs from receipt")
    store = cache_root / "sealed" / schedule["encoding"] / "segments"
    if symlinked(store):
        raise ValueError("unsafe mixture token segment store")
    settings = current_settings()
    proofs = settings.proof_store
    if proofs is None or not proofs.trusted(store):
        proofs = ProofStore(cache_root)

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
                proof_store=proofs,
                verification_mode=settings.verification_mode,
                chunk_size=4 * 1024**2,
            )
            if proof.size_bytes != entry["size"]:
                raise ValueError("mixture token segment size differs from its manifest")
            files.append(proof)
        return files

    workers = min(8, settings.workers, len(schedule["segments"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(check, schedule["segments"], buffersize=workers))
    tokens, indexes = zip(*files, strict=True)
    return VerifiedSchedule(
        schedule_file,
        tokens,
        indexes,
        eos,
        targets,
    )
