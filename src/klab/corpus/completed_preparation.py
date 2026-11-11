"""Explicit local reuse of completed preparation, not a full source re-audit.

This route retains the native renderer, v2 duplicate policy and release formats.
The existing native completion cache authenticates work actually executed here;
it does not turn a checksum into historical execution or redistribution rights.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import struct
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import (
    InspectionCache,
    capture,
    unchanged,
)
from klab.corpus.bounded_metadata import _HashedInput, _Reader
from klab.corpus.preparation_inputs import (
    corpus_work_root,
    local_dependencies,
    local_scope,
)
from klab.corpus.project import (
    load_project,
    project_path,
    source_declaration_payload,
)
from klab.corpus.rights import RightsPolicy, resolve_file_rights
from klab.corpus.split_runs import (
    SplitStages,
    _stamp,
    external_sort,
    rows,
    write_rows,
)
from klab.corpus.streaming_build_rows import StreamBuildRows
from klab.corpus.streaming_split_freeze import _AdmissionSpools
from klab.training.manifest import sha256_file

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PHASE = "native-build-publication"
_POLICY = "completed_local_preparation"


def _producer():
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "completed_preparation.py",
            "preparation_inputs.py",
            "streaming_build_rows.py",
            "split_runs.py",
            "large_build.py",
            "pipeline.py",
            "rights.py",
            "project.py",
            "jsonl_records.py",
        )
    }


def _small(path):
    _stamp(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("completed preparation control metadata too large")
    return json.loads(path.read_bytes())


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
            entry = struct.pack("<QQQ", key, offset, len(raw))
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
            observed = struct.unpack_from("<Q", self.keys, middle * 24)[0]
            if observed < key:
                left = middle + 1
            else:
                right = middle
        if left == self.count:
            raise KeyError(key)
        observed, offset, length = struct.unpack_from("<QQQ", self.keys, left * 24)
        if observed != key:
            raise KeyError(key)
        self.stream.seek(offset)
        self.previous, self.cached = key, json.loads(self.stream.read(length))
        return self.cached

    def __iter__(self):
        for ordinal in range(self.count):
            yield struct.unpack_from("<Q", self.keys, ordinal * 24)[0]

    def __len__(self):
        return self.count

    def close(self):
        self.owned.close()


def _admissions(directory, path, expected, declarations):
    before = _stamp(path)
    store = _FullDecisions(directory)
    try:
        with path.open("rb") as stream:
            hashed = _HashedInput(stream)
            manifest = _Reader(
                hashed, store, path.stat().st_size + 1, 4 * 1024**2, 8 * 1024**2
            ).read()
        if hashed.digest.hexdigest() != expected or _stamp(path) != before:
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
            or source["schema_version"] != 2
            or item["source_revision"] != source["revision"]
        ):
            raise ValueError("completed build requires unchanged v2 HF sources")
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


def _prepared_source(directory, source, header, path, build_id, progress, normalizer):
    from klab.corpus.large_build import _prepare_file
    from klab.corpus.rights import FileRights

    rights = FileRights.model_validate(header["rights"])
    decision_paths = [Path(header["decisions"][name]) for name in ("data", "keys")]
    decision_dependencies = capture(decision_paths)
    for name, decision_path in zip(("data", "keys"), decision_paths, strict=True):
        if sha256_file(decision_path) != header["decision_digests"][name + "_sha256"]:
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
        prepared = _prepare_file(
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
            normalizer_version=normalizer,
            record_admission=admitted,
            admission_sha256=header["admission_sha256"],
        )
    finally:
        lookup.close()
    unchanged(decision_dependencies)
    receipt = _small(prepared / "receipt.json")
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


def _check_completed_producer(project, producer):
    """Keep producer source identity; inventory-only runs have no admission."""
    declarations = {
        source.id: source_declaration_payload(source) for source in project.sources
    }
    producer_declarations = {
        source.id: source_declaration_payload(source) for source in producer.sources
    }
    current_admission = project.release.record_admission
    producer_admission = producer.release.record_admission
    if (
        current_admission is None
        or project.config.id != producer.config.id
        or declarations != producer_declarations
        or (
            producer_admission is not None
            and (
                current_admission.sha256 != producer_admission.sha256
                or project_path(project.root, current_admission.path)
                != project_path(producer.root, producer_admission.path)
            )
        )
    ):
        raise ValueError("build source/admission differs from completed producer")
    return declarations


def build_completed(project, work_root):
    from klab.corpus.acquisition import declaration_sha256
    from klab.corpus.large_build import _emit_build
    from klab.corpus.pipeline import (
        _model,
        _ordered_transforms,
        _uses_disk_lm_builder,
    )
    from klab.corpus.progress import BuildProgress
    from klab.corpus.split_assignments import copy_assignments
    from klab.corpus.split_inventory_adoption import (
        _inside,
        adopt_completed_inventory,
    )

    root = Path(work_root).resolve()
    transforms = _ordered_transforms(project)
    if (
        not _uses_disk_lm_builder(project, transforms)
        or project.release.schema_version != 2
        or project.splits.unit != "document"
        or any(source.kind != "huggingface_dataset" for source in project.sources)
    ):
        raise ValueError(
            "completed preparation build requires v2 HF document-split LM inputs"
        )
    split = project_path(project.root, project.config.splits)
    receipt_path = split.with_name(split.stem + ".receipt.json")
    frozen = _small(receipt_path)
    if (
        project.release.record_admission is None
        or frozen.get("split_sha256") != sha256_file(split)
        or frozen.get("admission_sha256") != project.release.record_admission.sha256
        or not frozen.get("preparation_reuse")
    ):
        raise ValueError("completed frozen preparation evidence differs")
    completion = frozen["preparation_reuse"]["declaration"]
    declaration_path = _inside(root, completion["path"])
    inventory_evidence = _small(declaration_path)
    producer_project = _inside(root, inventory_evidence["project"]["path"])
    producer = load_project(producer_project, verify_inputs=False)
    inventory = _inside(root, inventory_evidence["inventory"]["path"])
    declarations = _check_completed_producer(project, producer)
    admission_path = project_path(project.root, project.release.record_admission.path)
    candidates = split.with_name(split.stem + ".family-candidates.jsonl")
    family_reference = {
        "path": str(candidates),
        "sha256": frozen["family_candidate_sha256"],
    }
    inputs = [
        admission_path,
        split,
        receipt_path,
        declaration_path,
        producer_project,
        candidates,
        *[
            project_path(project.root, p)
            for p in (
                *project.config.sources,
                *project.config.transforms,
                project.config.release,
            )
        ],
    ]
    assignment = project.splits.model_dump(mode="json").get("assignment_file")
    if assignment is not None:
        inputs.append(split.parent / ("assignments-" + assignment["sha256"] + ".jsonl"))
    invocation = {
        "producer": _producer(),
        "project": project.model_dump(mode="json"),
        "inputs": capture(inputs),
    }
    invocation_cache = InspectionCache(root)
    prior = invocation_cache.load(
        "completed-build-invocation", invocation, limit=4 * 1024**2
    )
    if prior is not None:
        path = Path(prior[0]["build_path"])
        verify_completed_build(path)
        return path
    adopted = adopt_completed_inventory(producer, root, inventory, completion)
    provenance = {
        **adopted.provenance,
        "source_validation": "Completed producer evidence reused; row admission was not replayed by this local build.",
    }
    base_request = {
        "producer": _producer(),
        "project": project.model_dump(mode="json"),
        "inputs": {**capture(inputs), **adopted.dependencies},
        "inventory_execution": provenance,
    }
    workspace = root / "corpora" / project.config.id
    admission_stages = SplitStages(
        workspace / "build-input-stages", base_request, work_root=root
    )
    if admission_stages.cache.proofs is None or not admission_stages.cache.writable:
        raise ValueError(
            "completed preparation requires existing trusted native completion storage"
        )
    admission = admission_stages.phase(
        "build-admission-materialization",
        lambda d: _admissions(
            d, admission_path, project.release.record_admission.sha256, declarations
        ),
    )
    source_paths, lock = {}, {"sources": {}}
    for source in project.sources:
        header = admission["sources"][source.id]
        snapshot = workspace / "snapshots" / source.id / header["snapshot_sha256"]
        path = snapshot / "files" / header["sample_path"]
        if not path.resolve().is_relative_to(snapshot / "files"):
            raise ValueError("completed source shard escapes snapshot")
        _stamp(path)
        source_paths[source.id] = path
        lock["sources"][source.id] = {
            "declaration_sha256": declaration_sha256(source),
            "snapshot_sha256": header["snapshot_sha256"],
            "snapshot_path": str(snapshot),
        }
    # The source hashes are checked during rendering, not in a prior body scan.
    source_configuration = capture(source_paths.values())
    request = {
        **base_request,
        "source_bindings": source_configuration,
        "inputs": {**base_request["inputs"], **source_configuration},
    }
    stages = SplitStages(workspace / "build-stages", request, work_root=root)
    source_stages_by_id = {}
    for source in project.sources:
        header = admission["sources"][source.id]
        decisions = header["decisions"]
        source_request = {
            "producer": _producer(),
            "source": declarations[source.id],
            "normalizer": project.release.normalizer,
            "admission": {
                **{key: value for key, value in header.items() if key != "decisions"},
                "decisions": {
                    "rows": decisions["rows"],
                    "qualified": decisions["qualified"],
                    "data_sha256": sha256_file(Path(decisions["data"])),
                    "keys_sha256": sha256_file(Path(decisions["keys"])),
                },
            },
            "inputs": capture([source_paths[source.id]]),
        }
        source_stages_by_id[source.id] = SplitStages(
            workspace / "source-build-stages" / source.id,
            source_request,
            work_root=root,
        )
    scope = {
        "source_stage_ids": {
            key: value.identity for key, value in source_stages_by_id.items()
        },
        "policy": _POLICY,
        "request": request,
        "source_bindings": source_configuration,
        "source_bytes_verified": "during_native_shard_rendering",
        "deferred_verification": [
            "standalone acquisition/snapshot re-audit",
            "detailed row admission replay and row-envelope proof reconstruction",
        ],
        "reviewed_rights_decisions_reused": True,
    }
    identity = {
        "project_id": project.config.id,
        "project": _model(project.config),
        "lock": {
            key: {k: value[k] for k in ("declaration_sha256", "snapshot_sha256")}
            for key, value in sorted(lock["sources"].items())
        },
        "transforms": [_model(t) for t in transforms],
        "split": _model(project.splits),
        "release": _model(project.release),
        "implementation_sha256": sha256_file(Path(__file__).with_name("pipeline.py")),
        "schema_implementation_sha256": sha256_file(
            Path(__file__).with_name("project.py")
        ),
        "provenance_implementation_sha256": sha256_file(
            Path(__file__).with_name("provenance.py")
        ),
        "rights_implementation_sha256": sha256_file(
            Path(__file__).with_name("rights.py")
        ),
        "large_builder_implementation_sha256": sha256_file(
            Path(__file__).with_name("large_build.py")
        ),
        "completed_preparation": scope,
    }
    build_id = canonical_digest(identity)
    target = workspace / "builds" / build_id
    progress = BuildProgress(workspace / "progress" / (build_id + ".jsonl"), build_id)
    if _PHASE in stages.authenticated:
        result = stages.phase(_PHASE, lambda _: None)
        if result["build_path"] != str(target) or result["build_id"] != build_id:
            raise ValueError("completed native build identity differs")
        progress.record("reused_completed_build")
        return target
    prepared, rights_files = [], []
    for source in sorted(project.sources, key=lambda value: value.id):
        header = {
            **admission["sources"][source.id],
            "admission_sha256": project.release.record_admission.sha256,
        }
        source_stages = source_stages_by_id[source.id]
        header["decision_digests"] = source_stages.request["admission"]["decisions"]
        result = source_stages.phase(
            "native-shard-render",
            lambda d, source=source, header=header, source_identity=source_stages.identity: (
                _prepared_source(
                    d,
                    source,
                    header,
                    source_paths[source.id],
                    source_identity,
                    progress,
                    project.release.normalizer or "normalizer-nfc-markdown-v1",
                )
            ),
        )
        prepared.append(Path(result["prepared"]))
        rights_files.append(
            {
                "source_id": source.id,
                "path": header["sample_path"],
                "sha256": header["sample_sha256"],
                "size": source_paths[source.id].stat().st_size,
                "role": "document",
                "canonical_uri": source.canonical_uri,
                "revision": source.revision,
                "license_url": source.license_url,
                "rights": header["rights"],
                "admission_policy_id": header["policy_id"],
                "admission_policy_sha256": header["policy_sha256"],
                "admission_license_label": header["license_label"],
                "admission_policy_spdx_expression": header["policy_spdx_expression"],
            }
        )
    ordered = StreamBuildRows.create(
        stages, prepared, project, family_reference=family_reference
    )
    families = stages.phase(
        "build-final-family-inventory",
        lambda d: {
            "files": [
                write_rows(d / "family-inventory.jsonl", ordered.family_inventory())
            ]
        },
    )

    def emit(directory):
        if target.exists():
            # No unsigned existing output is mistaken for executed completion.
            raise ValueError(
                "unsealed completed build exists; preserve it for explicit recovery"
            )
        staging = directory / "build"
        staging.mkdir()
        copy_assignments(project, staging)
        shutil.copyfile(admission_path, staging / "record-admission.json")
        shutil.copyfile(
            families["files"][0]["path"], staging / "family-inventory.jsonl"
        )
        _emit_build(
            connection=None,
            staging=staging,
            project=project,
            lock=lock,
            identity=identity,
            build_id=build_id,
            transform=transforms[0],
            prepared=prepared,
            rights_files=rights_files,
            source_events=prepared,
            groups_path=Path(ordered.deduplicated["groups"]),
            group_count=ordered.deduplicated["group_count"],
            dropped=ordered.deduplicated["dropped"],
            progress=progress,
            ordered_rows=ordered,
            descriptive_census=False,
        )
        unchanged(request["inputs"])
        unchanged(source_configuration)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
        manifest = _small(target / "build.json")
        outputs = [
            {
                "path": str(target / name),
                "sha256": item["sha256"],
                "bytes": item["size"],
            }
            for name, item in manifest["files"].items()
        ]
        outputs.append(
            {
                "path": str(target / "build.json"),
                "sha256": sha256_file(target / "build.json"),
                "bytes": (target / "build.json").stat().st_size,
            }
        )
        return {"files": outputs, "build_path": str(target), "build_id": build_id}

    stages.phase(_PHASE, emit)
    invocation_cache.publish(
        "completed-build-invocation",
        invocation,
        {"build_path": str(target)},
        capture([target, *request["inputs"]]),
        limit=4 * 1024**2,
    )
    progress.record("complete")
    return target


def verify_completed_build(path, manifest=None):
    path = Path(path).resolve()
    manifest = manifest or _small(path / "build.json")
    scope = local_scope(manifest)
    if scope is None or canonical_digest(manifest["identity"]) != path.name:
        raise ValueError("invalid completed local build identity")
    root = corpus_work_root(path)
    local_dependencies(manifest, work_root=root)
    if scope["request"]["producer"] != _producer():
        raise ValueError("completed local build producer changed")
    unchanged(scope["source_bindings"])
    directory = path.parent.parent / "build-stages"
    request_id = canonical_digest(scope["request"])
    if not (directory / request_id).is_dir():
        raise ValueError(
            "completed native build evidence unavailable; no automatic source replay"
        )
    stages = SplitStages(directory, scope["request"], work_root=root)
    result = stages.completed_phase(_PHASE)
    if result is None:
        raise ValueError(
            "completed native build evidence unavailable; no automatic source replay"
        )
    if result["build_path"] != str(path) or result["build_id"] != manifest["build_id"]:
        raise ValueError("completed local build execution binding differs")
    return manifest


def verify_completed_release(path, manifest, *, expected_id=None):
    from klab.corpus.release import _files, _manifest_payload

    path = Path(path).resolve()
    payload = {key: value for key, value in manifest.items() if key != "release_id"}
    expected = expected_id or path.name
    if manifest.get("release_id") != expected or canonical_digest(payload) != expected:
        raise ValueError("invalid completed local release identity")
    build_path = path.parent.parent / "builds" / manifest["build_id"]
    build = verify_completed_build(build_path)
    if payload != _manifest_payload(build):
        raise ValueError("local release differs from completed native build")
    root = corpus_work_root(path)
    cache = InspectionCache(root)
    request = {
        "path": str(path),
        "release": manifest["release_id"],
        "manifest_sha256": sha256_file(path / "manifest.json"),
    }
    cached = cache.load("completed-local-release", request, limit=4 * 1024**2)
    if cached is not None and cached[0] == manifest:
        return manifest
    # This is copy verification against authenticated executed build output,
    # not a fresh claim that source admission was replayed here.
    _files(
        path,
        {
            **manifest["files"],
            "manifest.json": {
                "sha256": request["manifest_sha256"],
                "size": (path / "manifest.json").stat().st_size,
            },
        },
    )
    cache.publish(
        "completed-local-release",
        request,
        manifest,
        capture([path, build_path, *local_dependencies(manifest, work_root=root)]),
        limit=4 * 1024**2,
    )
    return manifest
