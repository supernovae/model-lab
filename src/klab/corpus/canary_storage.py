"""Read-only storage calibration from an explicitly selected native canary."""

from __future__ import annotations

import os
import re
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.bounded_metadata import read_inventory_metadata
from klab.corpus.normalized_spans import SPAN_MEMBER
from klab.corpus.shards import range_file_identity
from klab.safe_io import symlinked

_KEY = re.compile(r"[0-9a-f]{64}")
_MAX_ENTRIES = 8192
_MEMBERS = ("docs.jsonl", SPAN_MEMBER, "rejected.jsonl")
_PLACEMENT = {"reference_normalized_spans", "reference_normalized_documents"}


@dataclass(frozen=True)
class StorageCanary:
    path: Path
    manifest: dict
    generation: list[int]

    @property
    def workspace(self) -> Path:
        return self.path.parents[1]


@dataclass(frozen=True)
class PartitionSize:
    source_bytes: int
    documents: int
    members: tuple[int, ...]

    def project(self, size: int) -> PartitionSize:
        return PartitionSize(
            size,
            _scale(self.documents, size, self.source_bytes),
            tuple(_scale(value, size, self.source_bytes) for value in self.members),
        )


def _scale(value: int, total: int, sampled: int) -> int:
    return -(-(value * total) // sampled)


def _combine(sizes) -> PartitionSize:
    totals = [0] * (2 + len(_MEMBERS))
    for size in sizes:
        for index, value in enumerate(
            (size.source_bytes, size.documents, *size.members)
        ):
            totals[index] += value
    return PartitionSize(totals[0], totals[1], tuple(totals[2:]))


def _partition(path: Path):
    receipt = read_inventory_metadata(path / "receipt.json")
    identity = receipt["identity"]
    files = receipt["files"]
    counts = receipt["counts"]
    source_bytes, documents = counts["input_bytes"], counts["documents"]
    if (
        receipt["schema_version"] != 1
        or identity["artifact_type"] != "normalized_source_partition"
        or identity["source_id"] != path.parent.name
        or receipt["key"] != path.name
        or canonical_digest(identity) != path.name
        or files.keys() != set(_MEMBERS)
        or type(source_bytes) is not int
        or not 0 <= source_bytes <= identity["file_size"]
        or type(documents) is not int
        or documents < 0
    ):
        raise ValueError(f"invalid normalized storage receipt: {path}")
    sizes = []
    for name in _MEMBERS:
        member = path / name
        info = member.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or type(files[name]["size"]) is not int
            or info.st_size != files[name]["size"]
        ):
            raise ValueError(f"normalized storage member changed: {member}")
        sizes.append(info.st_size)
    return identity, PartitionSize(source_bytes, documents, tuple(sizes))


def open_storage_canary(path: Path, work_root: Path, identity: dict) -> StorageCanary:
    """Bind only a published canary for the same pins and non-placement policy."""
    path = path.absolute()
    root = work_root / "scratch" / "canary" / identity["project_id"] / "builds"
    if path.parent != root or symlinked(path) or not _KEY.fullmatch(path.name):
        raise ValueError(
            "storage canary must be a build in this project's canary workspace"
        )
    control = path / "build.json"
    generation = range_file_identity(control)
    manifest = read_inventory_metadata(control)
    sample = manifest["identity"].get("canary")
    expected = {key: value for key, value in identity.items() if key not in _PLACEMENT}
    if (
        manifest["schema_version"] != 1
        or manifest["build_id"] != path.name
        or canonical_digest(manifest["identity"]) != path.name
        or not isinstance(sample, dict)
        or set(sample) != {"files_per_source", "rows_per_file", "bytes_per_source"}
        or any(type(value) is not int or value < 1 for value in sample.values())
        or manifest["identity"] != {**expected, "canary": sample}
    ):
        raise ValueError("storage canary does not match this build's pins and policy")
    return StorageCanary(path, manifest, generation)


def _canary_partitions(root, sample, sources):
    """Match sampled partitions by their full-build keys, with a fixed scan budget."""
    samples = {}
    visits = 0
    for source in sources:
        directory = root / "prepared" / source["source_id"]
        if symlinked(directory):
            raise ValueError(f"symlinked canary prepared namespace: {directory}")
        if not directory.exists():
            continue
        full_keys = {
            row["path"]: row["computation_key"] for row in source["partitions"]
        }
        with os.scandir(directory) as entries:
            for entry in entries:
                visits += 1
                if visits > _MAX_ENTRIES:
                    raise ValueError(
                        "canary prepared metadata exceeds the planning entry budget"
                    )
                if not _KEY.fullmatch(entry.name):
                    continue
                path = Path(entry.path)
                if not entry.is_dir(follow_symlinks=False):
                    raise ValueError(f"unsafe canary prepared partition: {path}")
                identity, size = _partition(path)
                key = source["source_id"], identity["path"]
                if (
                    identity["row_limit"] != sample["rows_per_file"]
                    or canonical_digest({**identity, "row_limit": None})
                    != full_keys.get(key[1])
                    or key in samples
                ):
                    raise ValueError(
                        f"canary partition does not match the pinned input: {path}"
                    )
                samples[key] = size
    return samples


def _normalization(sources, samples):
    by_format = defaultdict(list)
    for (source, path), sample in samples.items():
        if sample.source_bytes:
            by_format[source, Path(path).suffix.lower()].append(sample)
    pooled = {key: _combine(sizes) for key, sizes in by_format.items()}
    records = []
    predictions = []
    for source in sources:
        source_id = source["source_id"]
        pending = covered = pending_bytes = covered_bytes = direct = 0
        for partition in source["partitions"]:
            if partition["action"] not in {"normalize", "resume_candidate"}:
                continue
            size = partition["source_bytes"]
            path = partition["path"]
            suffix = Path(path).suffix.lower()
            sample = samples.get((source_id, path))
            pending += 1
            pending_bytes += size
            if sample is not None and sample.source_bytes:
                direct += 1
            else:
                sample = pooled.get((source_id, suffix))
            if sample is None:
                continue
            predictions.append(sample.project(size))
            covered += 1
            covered_bytes += size
        if pending:
            records.append(
                {
                    "source_id": source_id,
                    "pending_files": pending,
                    "covered_files": covered,
                    "directly_sampled_files": direct,
                    "pending_source_bytes": pending_bytes,
                    "covered_source_bytes": covered_bytes,
                }
            )
    predicted = _combine(predictions)
    complete = all(row["covered_files"] == row["pending_files"] for row in records)
    return predicted, complete, records


def _member_sizes(sample: PartitionSize) -> dict[str, int]:
    return dict(zip(_MEMBERS, sample.members, strict=True))


def _stages(observation, canary, full, steps, storage, *, inline):
    """Model sequential named build occupancy, retaining new stage payloads."""
    estimate = {}
    combined = 0
    retained = 0
    for name in ("index", "deduplicate", "views"):
        step = steps[name]
        payload = storage[name]["payload_bytes"]
        peak = None
        basis = storage[name]["basis"]
        if basis in {"existing_build", "retained_candidate"}:
            peak = 0
        elif (
            full is not None
            and step["action"]
            in {"run", "await_normalization", "await_index", "await_deduplication"}
            and (name != "views" or inline)
        ):
            observed = observation["stage_outputs"].get(name)
            named = observation["stage_workspaces"].get(name)
            if observed is not None:
                sample_size = canary.members[0] if name == "views" else canary.documents
                full_size = full.members[0] if name == "views" else full.documents
                if sample_size:
                    if payload is None:
                        payload = _scale(
                            observed["payload_bytes"], full_size, sample_size
                        )
                        basis = (
                            "canary_document_bytes"
                            if name == "views"
                            else "canary_document_count"
                        )
                    if (
                        named is not None
                        and named["computation_key"] == observed["computation_key"]
                        and named["completed"]
                        and named["incomplete_samples"] == 0
                    ):
                        peak = _scale(
                            named["observed_peak_allocated_bytes"],
                            full_size,
                            sample_size,
                        )
        if step["action"] in {"run", "resume_candidate"}:
            prior_peak = storage[name]["prior_named_peak_allocated_bytes"]
            if prior_peak is not None:
                peak = prior_peak
        build_peak = (
            max(payload, peak) if payload is not None and peak is not None else None
        )
        estimate[name] = {
            "basis": basis,
            "modeled_payload_bytes": payload,
            "modeled_named_peak_bytes": build_peak,
        }
        if combined is not None:
            if build_peak is None or step["action"] == "resume_candidate":
                combined = None
            else:
                combined = max(combined, retained + build_peak)
                retained += payload
    return estimate, combined


def calibrate_storage(
    canary_build: StorageCanary,
    sources,
    prepared,
    observation,
    steps,
    storage,
):
    path, manifest = canary_build.path, canary_build.manifest
    sample = manifest["identity"]["canary"]
    samples = _canary_partitions(path.parent, sample, sources)
    report = path / "report.json"
    if (
        symlinked(report)
        or report.stat().st_size != manifest["files"]["report.json"]["size"]
    ):
        raise ValueError("canary report differs from its published inventory")
    observed_count = read_inventory_metadata(report)["document_count"]
    if range_file_identity(path / "build.json") != canary_build.generation:
        raise ValueError("storage canary changed during planning")
    canary = _combine(samples.values())
    if (
        canary.documents != observed_count
        or canary.members[_MEMBERS.index(SPAN_MEMBER)]
        != manifest["files"][SPAN_MEMBER]["size"]
    ):
        raise ValueError("canary normalized inventory differs from the published build")
    fresh, complete, coverage = _normalization(sources, samples)
    full = (
        _combine([fresh, *(_partition(directory)[1] for directory in prepared)])
        if complete
        else None
    )
    stages, build_peak = _stages(
        observation,
        canary,
        full,
        steps,
        storage["stage_storage_estimates"],
        inline=not steps["views"].get("reference_normalized_spans"),
    )
    normalization = sum(fresh.members) if complete else None
    filesystems = storage["filesystems"]
    same_device = filesystems["build"]["device"] == filesystems["normalized"]["device"]
    peak = None
    if normalization is not None and build_peak is not None:
        if same_device:
            peak = {"build_and_normalized": build_peak + normalization}
        else:
            peak = {"build": build_peak, "normalized": normalization}
    return {
        "build_id": manifest["build_id"],
        "sample": sample,
        "sampled_files": len(samples),
        "sampled_input_bytes": canary.source_bytes,
        "sampled_documents": canary.documents,
        "coverage": coverage,
        "covered_normalization_payload_bytes": sum(fresh.members),
        "modeled_normalization_payload_bytes": normalization,
        "modeled_normalization_members": _member_sizes(fresh) if complete else None,
        "stages": stages,
        "modeled_named_peak_by_filesystem": peak,
    }
