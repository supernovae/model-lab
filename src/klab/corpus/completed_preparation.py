"""Explicit local reuse of completed preparation, not a full source re-audit.

This route retains the native renderer, v2 duplicate policy and release formats.
The existing native completion cache authenticates work actually executed here;
it does not turn a checksum into historical execution or redistribution rights.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from copy import copy
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import (
    InspectionCache,
    capture,
    unchanged,
)
from klab.corpus.build_outputs import (
    publish_staging,
    rights_file_entry,
    stage_declared_inputs,
)
from klab.corpus.completed_build_inputs import (
    materialize_admission,
    render_source,
    source_render_stage,
)
from klab.corpus.preparation_inputs import (
    corpus_work_root,
    local_dependencies,
    local_scope,
)
from klab.corpus.project import (
    declaration_payload,
    load_project,
    project_path,
    source_build_declarations,
)
from klab.corpus.split_runs import (
    SplitStages,
    checkpoint_stamp,
    write_rows,
)
from klab.corpus.streaming_build_rows import StreamBuildRows
from klab.corpus.view_checkpoints import ViewCheckpoints
from klab.result_publication import sync_directory
from klab.safe_io import symlinked
from klab.training.manifest import sha256_file
from klab.verification_proofs import artifact_fingerprint

_PHASE = "native-build-publication"
_POLICY = "completed_local_preparation"


def _producer():
    """Digests of every corpus module a completed build can run.

    Rendering reaches cleaning, qualification, provenance and admission code
    through these entry modules, so all of them key the completed phases.
    """
    from klab.verifier_authority import local_import_closure

    return local_import_closure(
        tuple(
            "corpus." + name
            for name in (
                "completed_preparation",
                "preparation_inputs",
                "streaming_build_rows",
                "split_runs",
                "large_build",
                "pipeline",
                "rights",
                "project",
                "jsonl_records",
            )
        ),
        within="corpus",
        exclude=frozenset({"corpus.cli"}),
    )


def _small(path):
    checkpoint_stamp(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("completed preparation control metadata too large")
    return json.loads(path.read_bytes())


def _check_completed_producer(project, producer):
    """Keep producer source identity; inventory-only runs have no admission."""
    declarations = source_build_declarations(project.sources)
    producer_declarations = source_build_declarations(producer.sources)
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
    root = Path(work_root).resolve()
    workspace = root / "corpora" / project.config.id
    if symlinked(workspace):
        raise ValueError("symlinked completed preparation workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        workspace / ".completed-build.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another completed preparation build is running"
            ) from error
        return _build_completed(project, root, workspace)
    finally:
        os.close(descriptor)


def _build_completed(project, root, workspace):
    from klab.corpus.acquisition import declaration_sha256
    from klab.corpus.dedup_policy import is_strict_chain
    from klab.corpus.large_build import _emit_build
    from klab.corpus.pipeline import (
        _ordered_transforms,
        _uses_disk_lm_builder,
    )
    from klab.corpus.progress import BuildProgress
    from klab.corpus.split_inventory_adoption import (
        _inside,
        adopt_completed_inventory,
    )

    transforms = _ordered_transforms(project)
    if (
        not _uses_disk_lm_builder(project, transforms)
        or not is_strict_chain(project.release.policy.duplicates)
        or project.splits.unit != "document"
        or any(source.kind != "huggingface_dataset" for source in project.sources)
    ):
        raise ValueError(
            "completed preparation build requires HF document-split LM inputs"
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
    producer_implementation = _producer()
    invocation = {
        "producer": producer_implementation,
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
        "producer": producer_implementation,
        "project": project.model_dump(mode="json"),
        "inputs": {**invocation["inputs"], **adopted.dependencies},
        "inventory_execution": provenance,
    }
    admission = materialize_admission(
        workspace / "build-input-stages",
        root,
        admission_path,
        project.release.record_admission.sha256,
        declarations,
    )
    source_paths, lock = {}, {"sources": {}}
    for source in project.sources:
        header = admission["sources"][source.id]
        snapshot = workspace / "snapshots" / source.id / header["snapshot_sha256"]
        path = snapshot / "files" / header["sample_path"]
        if not path.resolve().is_relative_to(snapshot / "files"):
            raise ValueError("completed source shard escapes snapshot")
        checkpoint_stamp(path)
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
    source_stages_by_id = {
        source.id: source_render_stage(
            workspace / "source-build-stages" / source.id,
            root,
            declarations[source.id],
            admission["sources"][source.id],
            source_paths[source.id],
            project.release,
        )
        for source in project.sources
    }
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
        "project": declaration_payload(project.config),
        "lock": {
            key: {k: value[k] for k in ("declaration_sha256", "snapshot_sha256")}
            for key, value in sorted(lock["sources"].items())
        },
        "sources": declarations,
        "transforms": [declaration_payload(t) for t in transforms],
        "split": declaration_payload(project.splits),
        "release": declaration_payload(project.release),
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
    result = stages.completed_phase(_PHASE)
    if result is not None:
        if result["build_path"] != str(target) or result["build_id"] != build_id:
            raise ValueError("completed native build identity differs")
        _clean_completed_views(stages)
        progress.record("reused_completed_build")
        return target
    prepared, rights_files, rendered_sources = [], [], []
    for source in sorted(project.sources, key=lambda value: value.id):
        header = {
            **admission["sources"][source.id],
            "admission_sha256": project.release.record_admission.sha256,
        }
        source_stages = source_stages_by_id[source.id]
        result = render_source(
            source_stages,
            source,
            header,
            source_paths[source.id],
            progress,
            project.release,
        )
        prepared.append(Path(result["prepared"]))
        rendered_sources.extend(result["files"])
        rights_files.append(
            rights_file_entry(
                source,
                path=header["sample_path"],
                sha256=header["sample_sha256"],
                size=source_paths[source.id].stat().st_size,
                role="document",
                rights=header["rights"],
                admitted=header,
            )
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

    def emit(staging, checkpoints):
        stage_declared_inputs(project, staging, admission_path, resume=True)
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
            checkpoints=checkpoints,
            ordered_rows=ordered,
            descriptive_census=False,
        )

    input_files = [
        item["path"]
        for item in (
            *rendered_sources,
            *ordered.assigned["files"],
            *ordered.deduplicated["files"],
            *families["files"],
        )
    ]
    _publish_completed_views(stages, target, input_files, emit, progress)
    invocation_cache.publish(
        "completed-build-invocation",
        invocation,
        {"build_path": str(target)},
        capture([target, *request["inputs"]]),
        limit=4 * 1024**2,
    )
    progress.record("complete")
    return target


def _build_members(path):
    if symlinked(path) or not path.is_dir():
        raise ValueError(f"unsafe completed build directory: {path}")
    # Rename updates metadata on the containing directory, not its members.
    return [list(member) for member in artifact_fingerprint(path)[1:]]


def _clean_completed_views(stages):
    view_root = stages.root / "view-emission"
    if view_root.exists():
        shutil.rmtree(view_root)


def _publish_completed_views(stages, target, input_files, emit, progress):
    view_root = stages.root / "view-emission"
    staging = view_root / "build"
    if symlinked(staging) or symlinked(target):
        raise ValueError("symlinked completed build output")
    inputs = stages.request["inputs"]
    views = capture(input_files)
    binding = {"build_path": str(target), "views": canonical_digest(views)}
    kind = "completed-build-ready"
    limit = 4 * 1024**2
    # An optional ready receipt failure must leave the phase signer enabled.
    ready_cache = copy(stages.cache)

    def publish(directory):
        # View staging is stable across attempts. The phase still owns completion.
        directory.rmdir()
        unchanged(inputs)
        ready = ready_cache.load(kind, binding, limit=limit)
        if target.exists():
            if (
                staging.exists()
                or ready is None
                or _build_members(target) != ready[0]["members"]
            ):
                raise ValueError(
                    "unsealed completed build exists; preserve it for explicit recovery"
                )
            sync_directory(target.parent)
            progress.record("recovered_completed_build_publication")
            return ready[0]["result"]

        if (
            ready is not None
            and staging.is_dir()
            and _build_members(staging) == ready[0]["members"]
        ):
            payload = ready[0]
            progress.record("reused_completed_build_views")
        else:
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "build.json").unlink(missing_ok=True)
            progress.update(phase="writing_views")
            emit(staging, ViewCheckpoints(view_root, canonical_digest(binding)))
            manifest_path = staging / "build.json"
            manifest = _small(manifest_path)
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
                    "sha256": sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                }
            )
            payload = {
                "members": _build_members(staging),
                "result": {
                    "files": outputs,
                    "build_path": str(target),
                    "build_id": target.name,
                },
            }

        def seal():
            unchanged(inputs)
            unchanged(views)
            if _build_members(staging) != payload["members"]:
                raise ValueError("completed build changed before publication")
            if ready is None or ready[0] != payload:
                ready_cache.publish(kind, binding, payload, views, limit=limit)
                if not ready_cache.writable:
                    progress.record("completed_build_ready_proof_unavailable")

        progress.update(phase="publishing")
        publish_staging(staging, target, before_rename=seal)
        return payload["result"]

    stages.phase(_PHASE, publish)
    _clean_completed_views(stages)


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
