"""Portable selected-range token payloads and replay against prepared data."""

from __future__ import annotations

import hashlib
import tempfile
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from klab.canonical import canonical_json
from klab.corpus.mixture_index import MAX_RECEIPT_BYTES
from klab.corpus.mixture_schedule import (
    SCHEDULE_RANGE,
    VerifiedSchedule,
    open_verified_schedule_member,
    schedule_metadata,
)
from klab.corpus.mixture_transfer import selected_intervals
from klab.corpus.release_verify import current_settings, scratch_root
from klab.data.token_store import CONTENT_TOKEN_DTYPES
from klab.data.verification import HashingWriter, VerifiedFile, verify_file
from klab.result_publication import read_canonical, sync_directory
from klab.safe_io import read_file_range, symlinked
from klab.workspace_preflight import check_storage, require_storage

if TYPE_CHECKING:
    from klab.data.packing import PreparedData
    from klab.verification_proofs import ProofStore, VerificationMode

PROJECTION_MANIFEST = "projection.json"
_SCHEDULE = "projected.schedule.bin"
_RANGES = np.dtype([("segment", "<u4"), ("offset", "<u8"), ("count", "<u4")])
_MAP = np.dtype([("start", "<u8"), ("stop", "<u8"), ("packed", "<u8")])
_COPY_BYTES = 4 * 1024**2


def _token_name(ordinal: int) -> str:
    return f"segment-{ordinal:08d}.ids"


def _project_tokens(schedule, destination, index, intervals):
    """Write the union once, recording a disposable source-to-packed map."""
    groups = iter(groupby(intervals, key=itemgetter(0)))
    group = next(groups, None)
    width = schedule.token_dtype.itemsize
    copied, bounds = {}, []
    row = 0
    mapping = np.empty(1, dtype=_MAP)
    for ordinal, token_file in enumerate(schedule.tokens):
        start_row = row
        name = _token_name(ordinal)
        path = destination / name
        with path.open("xb") as output:
            writer = HashingWriter(output)
            if group is not None and group[0] == ordinal:
                with open_verified_schedule_member(token_file) as source:
                    for _, start, stop in group[1]:
                        mapping[0] = start, stop, writer.size // width
                        index.write(mapping.tobytes())
                        row += 1
                        past = stop * width
                        for offset in range(start * width, past, _COPY_BYTES):
                            writer.write(
                                read_file_range(
                                    source, offset, min(_COPY_BYTES, past - offset)
                                )
                            )
                group = next(groups, None)
            copied[name] = writer.seal(path)
        bounds.append((start_row, row))
    return copied, bounds


def _rebase_schedule(schedule, mapping_path, bounds, target):
    mapping = (
        np.memmap(mapping_path, dtype=_MAP, mode="r")
        if mapping_path.stat().st_size
        else np.empty(0, dtype=_MAP)
    )
    with (
        open_verified_schedule_member(schedule.schedule) as source,
        target.open("xb") as output,
    ):
        writer = HashingWriter(output)
        while raw := source.read(SCHEDULE_RANGE.size * 16_384):
            records = np.frombuffer(raw, dtype=_RANGES).copy()
            active = np.flatnonzero(records["count"] > 1)
            ordinals = records["segment"][active]
            order = np.argsort(ordinals, kind="stable")
            active, ordinals = active[order], ordinals[order]
            cuts = np.flatnonzero(ordinals[1:] != ordinals[:-1]) + 1
            for group in np.split(active, cuts) if len(active) else ():
                lo, hi = bounds[int(records["segment"][group[0]])]
                spans = mapping[lo:hi]
                offsets = records["offset"][group]
                found = np.searchsorted(spans["start"], offsets, side="right") - 1
                if np.any(found < 0) or np.any(
                    offsets + records["count"][group] - 1 > spans["stop"][found]
                ):
                    raise ValueError(
                        "selected token range is missing from the projection"
                    )
                records["offset"][group] = spans["packed"][found] + (
                    offsets - spans["start"][found]
                )
            records["offset"][records["count"] == 1] = 0
            writer.write(records.tobytes())
        return writer.seal(target)


def project_dependencies(
    schedule: VerifiedSchedule, destination: Path, manifest_sha256: str
) -> dict[str, VerifiedFile]:
    """Compact a verified source schedule inside the dispatch writer's staging area."""
    if schedule.projection_files:
        return schedule.copy_dependencies(destination)
    scratch = scratch_root()
    records = schedule.schedule.size_bytes // SCHEDULE_RANGE.size
    output_bytes = (
        min(
            sum(file.size_bytes for file in schedule.tokens),
            (schedule.target_count - records) * schedule.token_dtype.itemsize,
        )
        + schedule.schedule.size_bytes
        + 192 * len(schedule.tokens)
        + 4096
    )
    alongside_sort = records * _MAP.itemsize
    if scratch.stat().st_dev == destination.parent.stat().st_dev:
        alongside_sort += output_bytes
    require_storage(
        [
            check_storage(
                destination.parent,
                projected_bytes=output_bytes,
                projected_inodes=len(schedule.tokens) + 3,
            ),
        ]
    )
    destination.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix="project-mixture-", dir=scratch) as temp:
        mapping = Path(temp) / "ranges.bin"
        with (
            selected_intervals(schedule, additional_scratch_bytes=alongside_sort) as (
                intervals,
                _,
            ),
            mapping.open("wb") as index,
        ):
            files, bounds = _project_tokens(schedule, destination, index, intervals)
        files[_SCHEDULE] = _rebase_schedule(
            schedule, mapping, bounds, destination / _SCHEDULE
        )
    schedule.require_unchanged()
    marker = destination / PROJECTION_MANIFEST
    with marker.open("xb") as output:
        writer = HashingWriter(output)
        writer.write(
            canonical_json(
                {
                    "prepared_manifest_sha256": manifest_sha256,
                    "source_schedule_sha256": schedule.schedule.sha256,
                    "token_dtype": schedule.token_dtype.name,
                    "files": {
                        name: {"sha256": proof.sha256, "bytes": proof.size_bytes}
                        for name, proof in files.items()
                    },
                }
            )
            + b"\n"
        )
        files[PROJECTION_MANIFEST] = writer.seal(marker)
    sync_directory(destination)
    return files


def load_projection(
    store: Path, manifest: dict, *, memo: dict[object, VerifiedFile]
) -> VerifiedSchedule:
    """Verify the exact compact file inventory before replay can read it."""
    marker = store / PROJECTION_MANIFEST
    if symlinked(marker) or marker.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("unsafe selected token projection manifest")
    projection = read_canonical(marker)
    mixture = manifest["token_mixture"]
    source, targets = schedule_metadata(mixture)
    dtype_name = projection.get("token_dtype")
    files = projection.get("files")
    if (
        set(projection)
        != {
            "prepared_manifest_sha256",
            "source_schedule_sha256",
            "token_dtype",
            "files",
        }
        or projection["prepared_manifest_sha256"] != manifest["manifest_sha256"]
        or projection["source_schedule_sha256"] != source["sha256"]
        or not isinstance(dtype_name, str)
        or dtype_name not in CONTENT_TOKEN_DTYPES
        or not isinstance(files, dict)
    ):
        raise ValueError("selected token projection differs from prepared data")
    names = [_token_name(i) for i in range(len(source["segments"]))]
    actual = list(store.iterdir())
    if (
        set(files) != {*names, _SCHEDULE}
        or {path.name for path in actual} != {*files, PROJECTION_MANIFEST}
        or any(path.is_symlink() or not path.is_file() for path in actual)
    ):
        raise ValueError(
            "selected token projection inventory differs from its manifest"
        )
    width = CONTENT_TOKEN_DTYPES[dtype_name].itemsize
    eos = source["eos"]
    if eos >= 2 ** (8 * width):
        raise ValueError("invalid EOS for selected token projection")
    settings = current_settings()

    def check(path, digest):
        return verify_file(
            path,
            expected_sha256=digest,
            memo=memo,
            proof_store=settings.proof_store,
            verification_mode=settings.verification_mode,
        )

    marker_proof = check(
        marker, hashlib.sha256(canonical_json(projection) + b"\n").hexdigest()
    )
    proofs = {}
    for name, entry in files.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sha256", "bytes"}
            or not isinstance(entry["sha256"], str)
            or type(entry["bytes"]) is not int
            or entry["bytes"] < 0
            or entry["bytes"] % width
        ):
            raise ValueError("invalid selected token projection member")
        proof = check(store / name, entry["sha256"])
        if proof.size_bytes != entry["bytes"]:
            raise ValueError("selected token projection member has the wrong length")
        proofs[name] = proof
    schedule = proofs[_SCHEDULE]
    if schedule.size_bytes != source["bytes"]:
        raise ValueError("selected token projection changed the schedule length")
    return VerifiedSchedule(
        store.resolve(),
        schedule,
        tuple(proofs[name] for name in names),
        (),
        (),
        (),
        CONTENT_TOKEN_DTYPES[dtype_name],
        eos,
        targets,
        projection_files=(marker_proof, *proofs.values()),
    )


def verify_projection_replay(
    data: PreparedData,
    *,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> None:
    """Check logical equivalence once per digest pair in verified-reuse mode."""
    schedule = data.mixture_schedule
    assert schedule is not None and schedule.projection_files
    request = {
        "prepared": data.receipt.manifest_sha256,
        "projection": schedule.projection_files[0].sha256,
    }
    cache = None
    if verification_mode == "verified_reuse" and proof_store is not None:
        from klab.corpus.admission_inspection_cache import InspectionCache
        from klab.verifier_authority import scoped_authority

        cache = InspectionCache(
            proof_store.root,
            proof_store=proof_store,
            phase_producer=scoped_authority("selected_mixture"),
        )
        if cache.load("selected-mixture-replay", request, limit=8192) == (
            {"verified": True},
            {},
        ):
            schedule.require_unchanged()
            return
    with selected_intervals(schedule) as (intervals, _):
        covered = sum(stop - start for _, start, stop in intervals)
    if covered * schedule.token_dtype.itemsize != sum(
        proof.size_bytes for proof in schedule.tokens
    ):
        raise ValueError("selected token projection contains unselected bytes")
    data.verify_logical_train()
    schedule.require_unchanged()
    if cache is not None:
        cache.publish(
            "selected-mixture-replay", request, {"verified": True}, {}, limit=8192
        )
