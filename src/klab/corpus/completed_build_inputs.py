"""Reviewed admission tables and source rendering for completed preparation."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import struct
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from klab.canonical import canonical_json
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.bounded_metadata import _HashedInput, _Reader
from klab.corpus.build_stages import implementation_digests
from klab.corpus.prepared_shards import prepare_file, prepared_shard_implementation
from klab.corpus.rights import FileRights, RightsPolicy, resolve_file_rights
from klab.corpus.split_runs import (
    SplitStages,
    checkpoint_stamp,
    external_sort,
    rows,
)
from klab.corpus.streaming_split_freeze import _AdmissionSpools
from klab.training.manifest import sha256_file

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DECISION_ENTRY = struct.Struct("<QQQ")
_INPUT_PHASE_FILES = (
    "admission_inspection_cache.py",
    "build_stages.py",
    "completed_build_inputs.py",
    "split_runs.py",
)


class _FullDecisions(_AdmissionSpools):
    def append(self, spool, item):
        if (
            not isinstance(item, dict)
            or type(item.get("source_row_index")) is not int
            or item["source_row_index"] < 0
            or item.get("decision") not in {"qualify", "exclude", "quarantine"}
            or not isinstance(item.get("source_row_sha256"), str)
            or not _HEX.fullmatch(item["source_row_sha256"])
        ):
            raise ValueError("invalid completed row decision")
        spool.stream.write(canonical_json(item) + b"\n")
        spool.count += 1


def _decision_table(directory, spool):
    ordered = external_sort(
        rows(spool.path), directory / "sorted", key=lambda row: row["source_row_index"]
    )
    data, keys = directory / "decisions.jsonl", directory / "keys.bin"
    previous, count, qualified, offset = None, 0, 0, 0
    data_sha, keys_sha = hashlib.sha256(), hashlib.sha256()
    with data.open("xb") as target, keys.open("xb") as index:
        for row in rows(ordered["path"]):
            key = row["source_row_index"]
            if key == previous:
                raise ValueError("duplicate completed admission row")
            raw = canonical_json(row) + b"\n"
            entry = _DECISION_ENTRY.pack(key, offset, len(raw))
            target.write(raw)
            data_sha.update(raw)
            index.write(entry)
            keys_sha.update(entry)
            offset += len(raw)
            previous, count = key, count + 1
            qualified += row["decision"] == "qualify"
        for stream in (target, index):
            stream.flush()
            os.fsync(stream.fileno())
    files = [
        ordered,
        {
            "path": str(data),
            "sha256": data_sha.hexdigest(),
            "bytes": data.stat().st_size,
        },
        {
            "path": str(keys),
            "sha256": keys_sha.hexdigest(),
            "bytes": keys.stat().st_size,
        },
    ]
    return {
        "files": files,
        "data": str(data),
        "keys": str(keys),
        "data_sha256": data_sha.hexdigest(),
        "keys_sha256": keys_sha.hexdigest(),
        "rows": count,
        "qualified": qualified,
    }


class _DecisionLookup(Mapping):
    def __init__(self, reference):
        self.count = reference["rows"]
        with ExitStack() as stack:
            self.stream = stack.enter_context(Path(reference["data"]).open("rb"))
            source = stack.enter_context(Path(reference["keys"]).open("rb"))
            self.keys = (
                stack.enter_context(
                    mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                )
                if self.count
                else None
            )
            self.owned = stack.pop_all()
        self.previous = None
        self.cached = None

    def __getitem__(self, key):
        if type(key) is not int:
            raise KeyError(key)
        if key == self.previous:
            return self.cached
        left, right = 0, self.count
        while left < right:
            middle = (left + right) // 2
            observed = struct.unpack_from(
                "<Q", self.keys, middle * _DECISION_ENTRY.size
            )[0]
            if observed < key:
                left = middle + 1
            else:
                right = middle
        if left == self.count:
            raise KeyError(key)
        observed, offset, length = _DECISION_ENTRY.unpack_from(
            self.keys, left * _DECISION_ENTRY.size
        )
        if observed != key:
            raise KeyError(key)
        self.stream.seek(offset)
        self.previous, self.cached = key, json.loads(self.stream.read(length))
        return self.cached

    def __iter__(self):
        for ordinal in range(self.count):
            yield struct.unpack_from("<Q", self.keys, ordinal * _DECISION_ENTRY.size)[0]

    def __len__(self):
        return self.count

    def close(self):
        self.owned.close()


def _admissions(directory, path, expected, declarations):
    before = checkpoint_stamp(path)
    store = _FullDecisions(directory)
    try:
        with path.open("rb") as stream:
            hashed = _HashedInput(stream)
            manifest = _Reader(
                hashed, store, path.stat().st_size + 1, 4 * 1024**2, 8 * 1024**2
            ).read()
        if hashed.digest.hexdigest() != expected or checkpoint_stamp(path) != before:
            raise ValueError("completed admission digest mismatch")
        if (
            set(manifest) != {"schema_version", "policy_id", "policy_sha256", "sources"}
            or manifest["schema_version"] != 2
            or not manifest["policy_id"]
            or not _HEX.fullmatch(manifest["policy_sha256"])
        ):
            raise ValueError("complete reviewed v2 admission required")
        store.close(True)
    except BaseException:
        store.close(False)
        raise
    sources, files = {}, []
    for item in manifest["sources"]:
        source_id = item["source_id"]
        if source_id not in declarations or source_id in sources:
            raise ValueError("completed admission source inventory differs")
        source = declarations[source_id]
        if (
            source["kind"] != "huggingface_dataset"
            or item["source_revision"] != source["revision"]
        ):
            raise ValueError("completed build requires unchanged HF sources")
        for key in ("snapshot_sha256", "sample_sha256"):
            if not _HEX.fullmatch(item[key]):
                raise ValueError("invalid completed source digest")
        policy = RightsPolicy.model_validate(item["rights"])
        rights = resolve_file_rights(
            policy, item["sample_path"], b"", prospective_private_research=True
        )
        if (
            policy.training_eligibility not in {"eligible", "eligible_with_obligations"}
            or policy.redistribution_mode != "metadata_reconstruction_only"
            or not policy.license_references
            or rights.training_eligibility
            not in {"eligible", "eligible_with_obligations"}
        ):
            raise ValueError("completed source lacks admitted local training rights")
        subdirectory = directory / source_id
        subdirectory.mkdir()
        decisions = _decision_table(subdirectory, item["records"])
        files.extend(decisions.pop("files"))
        sources[source_id] = {
            "snapshot_sha256": item["snapshot_sha256"],
            "sample_path": item["sample_path"],
            "sample_sha256": item["sample_sha256"],
            "rights": rights.model_dump(mode="json"),
            "decisions": decisions,
            "license_label": item["license_label"],
            "policy_spdx_expression": policy.spdx_expression,
            "policy_id": manifest["policy_id"],
            "policy_sha256": manifest["policy_sha256"],
        }
    if set(sources) != set(declarations):
        raise ValueError("completed admission source inventory differs")
    return {"files": files, "sources": sources}


def _prepared_source(directory, source, header, path, build_id, progress, release):
    rights = FileRights.model_validate(header["rights"])
    decision_paths = [Path(header["decisions"][name]) for name in ("data", "keys")]
    decision_dependencies = capture(decision_paths)
    for name, decision_path in zip(("data", "keys"), decision_paths, strict=True):
        if sha256_file(decision_path) != header["decisions"][name + "_sha256"]:
            raise ValueError("source admission decisions changed before rendering")
    lookup = _DecisionLookup(header["decisions"])
    try:
        admitted = {
            name: header[name]
            for name in (
                "license_label",
                "policy_spdx_expression",
                "policy_id",
                "policy_sha256",
            )
        } | {"rights": rights, "decisions": lookup}
        prepared = prepare_file(
            build_id=build_id,
            prepared_root=directory,
            source=source,
            file={
                "path": header["sample_path"],
                "sha256": header["sample_sha256"],
                "size": path.stat().st_size,
            },
            snapshot_sha=header["snapshot_sha256"],
            source_path=path,
            decision=rights,
            progress=progress,
            normalizer_options=release.normalizer,
            document_policy=release.policy.documents,
            record_admission=admitted,
            admission_sha256=header["admission_sha256"],
        )
    finally:
        lookup.close()
    unchanged(decision_dependencies)
    receipt_path = prepared / "receipt.json"
    checkpoint_stamp(receipt_path)
    if receipt_path.stat().st_size > 4 * 1024**2:
        raise ValueError("prepared source receipt exceeds the control metadata limit")
    receipt = json.loads(receipt_path.read_bytes())
    if receipt["counts"]["documents"] != header["decisions"]["qualified"]:
        raise ValueError("admitted source row inventory differs from documents")
    outputs = [
        {
            "path": str(prepared / name),
            "sha256": value["sha256"],
            "bytes": value["size"],
        }
        for name, value in receipt["files"].items()
    ]
    outputs.append(
        {
            "path": str(prepared / "receipt.json"),
            "sha256": sha256_file(prepared / "receipt.json"),
            "bytes": (prepared / "receipt.json").stat().st_size,
        }
    )
    return {"files": outputs, "prepared": str(prepared), "counts": receipt["counts"]}


def materialize_admission(directory, work_root, path, expected, declarations):
    """Reuse authenticated reviewed rows across changes to downstream phases."""
    request = {
        "producer": implementation_digests(
            *_INPUT_PHASE_FILES,
            "bounded_metadata.py",
            "project.py",
            "rights.py",
            "streaming_split_freeze.py",
        ),
        "admission_sha256": expected,
        "sources": declarations,
        "inputs": capture([path]),
    }
    stages = SplitStages(directory, request, work_root=work_root, producer_scoped=True)
    if stages.cache.proofs is None or not stages.cache.writable:
        raise ValueError(
            "completed preparation requires existing trusted native completion storage"
        )
    result = stages.phase(
        "build-admission-materialization",
        lambda d: _admissions(d, path, expected, declarations),
    )
    unchanged(request["inputs"])
    return result


def source_render_stage(directory, work_root, declaration, header, path, release):
    """Bind rendered bytes to only this source's reviewed inputs and renderer."""
    decisions = header["decisions"]
    request = {
        "producer": {
            **prepared_shard_implementation(),
            **implementation_digests(*_INPUT_PHASE_FILES),
        },
        "source": declaration,
        "normalizer": release.normalizer.model_dump(mode="json"),
        "document_policy": release.policy.documents.model_dump(mode="json"),
        "admission": {
            **header,
            "decisions": {
                key: decisions[key]
                for key in ("rows", "qualified", "data_sha256", "keys_sha256")
            },
        },
        "inputs": capture([path]),
    }
    return SplitStages(directory, request, work_root=work_root, producer_scoped=True)


def render_source(stages, source, header, path, progress, release):
    result = stages.phase(
        "native-shard-render",
        lambda directory: _prepared_source(
            directory, source, header, path, stages.identity, progress, release
        ),
    )
    unchanged(stages.request["inputs"])
    if "native-shard-render" in stages.reused:
        progress.record("reused_completed_source_render")
    return result
