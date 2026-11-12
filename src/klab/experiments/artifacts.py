"""Verify existing typed experiment inputs without acquiring or preparing data."""

from __future__ import annotations

import json
import os
import stat
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from klab.canonical import canonical_json
from klab.experiments.artifact_paths import artifact_path
from klab.training.manifest import sha256_file
from klab.verification_proofs import (
    ProofStore,
    VerificationMode,
    artifact_binding,
    validate_mode,
)

if TYPE_CHECKING:
    from klab.config.models import DatasetConfig
    from klab.data.verification import VerifiedPreparedData
    from klab.experiments.plan import Artifact, ExperimentPlan


def _safe_path(reference: str, source: Path) -> Path:
    target = artifact_path(reference, source)
    if not target.exists():
        raise ValueError(f"missing artifact: {target}")
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked artifact path: {component}")
    if target.is_dir():
        for member in target.rglob("*"):
            if member.is_symlink():
                raise ValueError(f"symlinked artifact member: {member}")
    elif not target.is_file():
        raise ValueError(f"artifact is not a regular file or directory: {target}")
    return target


def _json_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"artifact manifest must be an object: {path}")
    return raw


def _require_directory(path: Path, manifest: str) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"expected a directory with {manifest}: {path}")
    return _json_file(path / manifest)


def _identity(artifact: Artifact, actual: object, digest: object) -> None:
    if actual != artifact.identifier:
        raise ValueError(
            f"logical identifier mismatch: expected {artifact.identifier}, got {actual}"
        )
    if digest != artifact.sha256:
        raise ValueError(
            f"domain digest mismatch: expected {artifact.sha256}, got {digest}"
        )


def _check_snapshot_paths(path: Path, manifest: dict[str, Any]) -> None:
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list):
        raise TypeError("missing build/release snapshot inventory")
    for item in snapshots:
        if not isinstance(item, dict):
            raise TypeError("invalid snapshot inventory entry")
        source_id, digest = item.get("source_id"), item.get("sha256")
        if (
            not isinstance(source_id, str)
            or not source_id
            or Path(source_id).name != source_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid snapshot identity")
        _safe_path(str(path.parent.parent / "snapshots" / source_id / digest), path)


def _verify_domain(
    artifact: Artifact,
    path: Path,
    *,
    dataset: DatasetConfig | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    kind = artifact.kind
    if kind == "dataset_snapshot":
        from klab.data.sources import verify_snapshot

        if artifact.version != 1 or not path.is_file() or path.name != "manifest.json":
            raise ValueError("dataset snapshot requires version 1 and manifest.json")
        verified = verify_snapshot(path, verification_mode="cold")
        if verified.get("format") != "klab-dataset-snapshot-v1":
            raise ValueError("dataset snapshot requires a native source-lock manifest")
        _identity(
            artifact, verified["lock"]["lock_sha256"], verified["manifest_sha256"]
        )
    elif kind == "source_snapshot":
        from klab.corpus.acquisition import verify_snapshot

        manifest = _require_directory(path, "manifest.json")
        verified = verify_snapshot(
            path,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
        )
        if manifest != verified or manifest.get("schema_version") != artifact.version:
            raise ValueError("snapshot manifest/version mismatch")
        _identity(artifact, verified["source_id"], verified["snapshot_sha256"])
    elif kind == "corpus_build":
        from klab.corpus.release import verify_build

        manifest = _require_directory(path, "build.json")
        if manifest.get("schema_version") != artifact.version:
            raise ValueError("build format version mismatch")
        _check_snapshot_paths(path, manifest)
        verified = verify_build(
            path,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
        )
        _identity(artifact, verified["build_id"], verified["build_id"])
    elif kind == "corpus_release":
        from klab.corpus.release import verify_release

        manifest = _require_directory(path, "manifest.json")
        if manifest.get("schema_version") != artifact.version:
            raise ValueError("release format version mismatch")
        _check_snapshot_paths(path, manifest)
        verified = verify_release(
            path,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
        )
        _identity(artifact, verified["release_id"], verified["release_id"])
    elif kind == "corpus_export":
        from klab.config.models import RunConfig
        from klab.corpus.export import verify_release_export

        manifest = _require_directory(path, "export.json")
        if manifest.get("schema_version") != artifact.version:
            raise ValueError("export format version mismatch")
        run = RunConfig.model_validate(
            yaml.safe_load((path / "run.yaml").read_text(encoding="utf-8"))
        )
        if run.dataset.corpus_export_path != path:
            raise ValueError(
                "export run configuration references another export directory"
            )
        if run.dataset.corpus_release_path is None:
            raise ValueError("export lacks a pinned release")
        _safe_path(str(run.dataset.corpus_release_path), path)
        verified = verify_release_export(
            run.dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
        )
        _identity(artifact, path.name, path.name)
        if verified["export_sha256"] != sha256_file(path / "export.json"):
            raise ValueError("export sidecar changed")
    elif kind == "tokenizer":
        from klab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact

        if not path.is_file() or path.name != "tokenizer.json":
            raise ValueError(
                "tokenizer requires tokenizer.json and its provenance manifest"
            )
        manifest = _json_file(
            _safe_path(str(path.with_name("tokenizer_manifest.json")), path)
        )
        if artifact.version != 1:
            raise ValueError("unsupported tokenizer artifact version")
        source, revision, vocab = (
            manifest.get("source"),
            manifest.get("revision"),
            manifest.get("vocab_size"),
        )
        if not isinstance(source, str) or type(vocab) is not int:
            raise ValueError("invalid tokenizer provenance")
        supplied_dataset = dataset
        dataset = None
        bakeoff = manifest.get("corpus_forge_bakeoff")
        if bakeoff is not None:
            if (
                source != "local_text"
                or not isinstance(bakeoff, dict)
                or not isinstance(bakeoff.get("release_id"), str)
            ):
                raise ValueError("invalid bakeoff tokenizer provenance")
            _safe_path(str(path.parent.parent.parent), path)
            report = _json_file(path.parent.parent.parent / "report.json")
            _safe_path(report["identity"]["release_path"], path)
        elif manifest.get("corpus_export") is not None:
            from klab.config.models import RunConfig

            export = path.parent.parent
            _safe_path(str(export), path)
            run = RunConfig.model_validate(
                yaml.safe_load((export / "run.yaml").read_text(encoding="utf-8"))
            )
            if run.tokenizer.path != path or run.dataset.corpus_export_path != export:
                raise ValueError("tokenizer is not bound to its corpus export")
            if run.dataset.corpus_release_path is None:
                raise ValueError("corpus tokenizer lacks a pinned release")
            _safe_path(str(run.dataset.corpus_release_path), path)
            dataset = run.dataset
        elif source == "snapshot":
            if supplied_dataset is None or supplied_dataset.source != source:
                raise ValueError(f"{source} tokenizer requires a pinned source dataset")
            dataset = supplied_dataset
        verify_tokenizer_artifact(
            path,
            source=source,
            revision=revision,
            vocab_size=vocab,
            dataset=dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
        )
        tokenizer = load_tokenizer(path)
        if tokenizer.get_vocab_size() != vocab:
            raise ValueError("tokenizer vocabulary differs from manifest")
        _identity(
            artifact,
            path.parent.name,
            sha256_file(path),
        )
    elif kind == "prepared_data":
        _load_verified_prepared(
            artifact, path, proof_store=proof_store, verification_mode=verification_mode
        )
    elif kind == "stage_bundle":
        from klab.config.models import RunConfig
        from klab.staging import verify_stage_bundle

        inputs = _require_directory(path, "inputs.json")
        if artifact.version != 1:
            raise ValueError("unsupported stage-bundle version")
        config = RunConfig.model_validate(inputs["requested_config"])
        verified = verify_stage_bundle(
            path, config, proof_store=proof_store, verification_mode=verification_mode
        )
        _identity(artifact, verified.get("sha256"), verified.get("sha256"))
    elif kind == "checkpoint":
        from klab.training.checkpoints import FORMAT_VERSION, CheckpointManager

        raw = _require_directory(path, "manifest.json")
        if (
            path.name in {"latest.json", "best.json"}
            or artifact.version != FORMAT_VERSION
        ):
            raise ValueError("checkpoint must pin a supported immutable generation")
        if path.parent.name != "checkpoints":
            raise ValueError(
                "checkpoint must be a generation under a run's checkpoints directory"
            )
        if artifact.state not in {"full", "weights"}:
            raise ValueError("checkpoint state must be full or weights")
        expected_level = "full" if artifact.state == "full" else "weights_only"
        if raw.get("resume_level") != expected_level:
            raise ValueError(f"checkpoint does not have {expected_level} state")
        report = CheckpointManager(path.parent.parent).verify(
            path, require_training_state=artifact.state == "full"
        )
        if not report.valid:
            raise ValueError(f"checkpoint generation invalid: {report.errors}")
        if raw.get("format_version") != artifact.version:
            raise ValueError("checkpoint format version mismatch")
        _identity(artifact, path.name, raw.get("sha256"))
    elif kind == "capability_card":
        from klab.evaluation.capabilities import load_capability_card

        if not path.is_file():
            raise ValueError("capability card must be a JSON file")
        card = load_capability_card(path)
        if card.version != artifact.version:
            raise ValueError("capability card version mismatch")
        _identity(artifact, card.name, sha256_file(path))
    elif kind == "prompt_set":
        if artifact.version != 1 or not path.is_file() or path.suffix != ".json":
            raise ValueError("prompt_set v1 requires a JSON file")
        panel = _json_file(path)
        prompts = panel.get("prompts")
        if (
            panel.get("format") != "dense_lm_decoding_prompts_v1"
            or panel.get("split") not in {"development", "test"}
            or not isinstance(prompts, list)
            or not prompts
        ):
            raise ValueError("unsupported prompt-set format or empty panel")
        ids: set[str] = set()
        for row in prompts:
            if (
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(key), str) or not row[key]
                    for key in ("id", "text", "category")
                )
                or row["id"] in ids
            ):
                raise ValueError("invalid or duplicate prompt in frozen panel")
            ids.add(row["id"])
        _identity(artifact, path.stem, sha256_file(path))
    else:
        raise ValueError(f"no immutable external verifier for {kind}")


def _load_verified_prepared(
    artifact: Artifact,
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> VerifiedPreparedData:
    from klab.data.packing import load_prepared_data

    manifest = _require_directory(path, "manifest.json")
    if artifact.version != 1:
        raise ValueError("unsupported prepared-data artifact version")
    data = load_prepared_data(
        path,
        byte_enabled=manifest.get("byte_addressing") is not None,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    _identity(
        artifact, manifest.get("settings_sha256"), manifest.get("manifest_sha256")
    )
    return data.receipt


def verify_prepared_artifact(
    artifact: Artifact,
    source: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> VerifiedPreparedData:
    """Authenticate a typed existing prepared artifact and return its sealed proof."""
    if artifact.kind != "prepared_data" or artifact.from_phase is not None:
        raise ValueError(
            "prepared storage planning requires an existing prepared artifact"
        )
    if artifact.path is None or artifact.sha256 is None:
        raise ValueError("prepared storage planning requires pinned path and digest")
    return _load_verified_prepared(
        artifact,
        _safe_path(artifact.path, source),
        proof_store=proof_store,
        verification_mode=verification_mode,
    )


_MEMO_SEAL = object()


class _VerifiedArtifact:
    __slots__ = (
        "__weakref__",
        "_seal",
        "cold_verified",
        "identity",
        "issuer_pid",
        "key",
    )

    def __init__(
        self,
        identity: dict[str, object],
        key: tuple[object, ...],
        *,
        _seal: object = None,
        cold_verified: bool = True,
    ) -> None:
        if _seal is not _MEMO_SEAL:
            raise TypeError("artifact evidence cannot be constructed from metadata")
        object.__setattr__(self, "identity", MappingProxyType(dict(identity)))
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "_seal", _MEMO_SEAL)
        object.__setattr__(self, "issuer_pid", os.getpid())
        object.__setattr__(self, "cold_verified", cold_verified)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("sealed artifact evidence is immutable")


_MINTED_ARTIFACTS: weakref.WeakSet[_VerifiedArtifact] = weakref.WeakSet()


def _fingerprint(path: Path) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
    members = [path] if path.is_file() else [path, *sorted(path.rglob("*"))]
    result = []
    for member in members:
        info = member.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"nonregular artifact member: {member}")
        result.append(
            (
                member.relative_to(path).as_posix() if member != path else ".",
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        )
    return tuple(result)


def _snapshot_dependencies(path: Path, manifest_name: str) -> tuple[Path, ...]:
    manifest = _json_file(path / manifest_name)
    from klab.corpus.preparation_inputs import (
        corpus_work_root,
        local_dependencies,
        local_scope,
    )

    if local_scope(manifest) is not None:
        dependencies = local_dependencies(manifest, work_root=corpus_work_root(path))
        if manifest_name == "manifest.json":
            return (*dependencies, path.parent.parent / "builds" / manifest["build_id"])
        return dependencies
    _check_snapshot_paths(path, manifest)
    return tuple(
        path.parent.parent / "snapshots" / item["source_id"] / item["sha256"]
        for item in manifest["snapshots"]
    )


def _dependent_paths(artifact: Artifact, path: Path) -> tuple[Path, ...]:
    """Track external files read by domain verifiers alongside the artifact tree."""
    if artifact.kind == "source_snapshot":
        manifest = _json_file(path / "manifest.json")
        retained = manifest["declaration"]["acquisition"].get("retained_subset")
        if retained is not None:
            return (Path(retained["snapshot_path"]),)
    if artifact.kind == "dataset_snapshot":
        # The manifest identity covers sibling split/exclusion/event payloads.
        # Include all of them in memo and persistent proof invalidation.
        return (path.parent,)
    if artifact.kind == "tokenizer":
        manifest = path.with_name("tokenizer_manifest.json")
        paths = [manifest]
        provenance = _json_file(manifest)
        release = None
        if provenance.get("corpus_forge_bakeoff") is not None:
            selection = path.parent.parent.parent
            paths.append(selection)
            release = Path(
                _json_file(selection / "report.json")["identity"]["release_path"]
            )
        elif provenance.get("corpus_export") is not None:
            export = path.parent.parent
            paths.append(export)
            run = yaml.safe_load((export / "run.yaml").read_text(encoding="utf-8"))
            release = Path(run["dataset"]["corpus_release_path"])
        if release is not None:
            paths.append(release)
            paths.extend(_snapshot_dependencies(release, "manifest.json"))
        return tuple(paths)
    if artifact.kind in {"corpus_build", "corpus_release"}:
        return _snapshot_dependencies(
            path, "build.json" if artifact.kind == "corpus_build" else "manifest.json"
        )
    if artifact.kind == "corpus_export":
        run = yaml.safe_load((path / "run.yaml").read_text(encoding="utf-8"))
        release = Path(run["dataset"]["corpus_release_path"])
        return (release, *_snapshot_dependencies(release, "manifest.json"))
    return ()


def _manifest_binding(artifact: Artifact, path: Path) -> str:
    """Bind manifests and exact typed upstream identities, not an invented DAG."""
    manifests = []
    upstream = []
    for member in (path, *_dependent_paths(artifact, path)):
        if member.is_file():
            candidates = (
                [member.with_name("tokenizer_manifest.json")]
                if member.name == "tokenizer.json"
                else []
            )
        else:
            candidates = [
                member / name
                for name in ("manifest.json", "build.json", "export.json")
                if (member / name).is_file()
            ]
        for candidate in candidates:
            raw = _json_file(candidate)
            manifests.append((str(candidate.absolute()), sha256_file(candidate)))
            if "release_id" in raw:
                upstream.append(
                    {
                        "kind": "corpus_release",
                        "version": raw.get("schema_version"),
                        "identifier": raw["release_id"],
                        "sha256": raw["release_id"],
                    }
                )
            elif "snapshot_sha256" in raw:
                upstream.append(
                    {
                        "kind": "source_snapshot",
                        "version": raw.get("schema_version"),
                        "identifier": raw.get("source_id"),
                        "sha256": raw["snapshot_sha256"],
                    }
                )
            elif "build_id" in raw:
                upstream.append(
                    {
                        "kind": "corpus_build",
                        "version": raw.get("schema_version"),
                        "identifier": raw["build_id"],
                        "sha256": raw["build_id"],
                    }
                )
            elif "export_id" in raw:
                upstream.append(
                    {
                        "kind": "corpus_export",
                        "version": raw.get("schema_version"),
                        "identifier": raw["export_id"],
                        "sha256": raw["export_id"],
                    }
                )
    return canonical_json({"manifests": manifests, "upstream": upstream}).decode(
        "utf-8"
    )


def _artifact_key(
    artifact: Artifact, path: Path, dataset: DatasetConfig | None = None
) -> tuple[object, ...]:
    dependencies = tuple(
        (str(dependency.absolute()), _fingerprint(_safe_path(str(dependency), path)))
        for dependency in _dependent_paths(artifact, path)
    )
    manifest_binding = _manifest_binding(artifact, path)
    if (
        artifact.kind == "tokenizer"
        and _json_file(path.with_name("tokenizer_manifest.json")).get("source")
        == "snapshot"
    ):
        source = "snapshot"
        if dataset is None or dataset.source != source:
            raise ValueError(f"{source} tokenizer requires a pinned source dataset")
        assert dataset.source_manifest_path is not None
        manifest = _json_file(_safe_path(str(dataset.source_manifest_path), path))
        members = (
            dataset.source_manifest_path,
            dataset.train_path,
            dataset.validation_path,
            dataset.source_manifest_path.parent / manifest["excluded"]["path"],
        )
        # The generic verifier authenticates the whole snapshot inventory,
        # including its event journal and any additional declared splits.
        members += (dataset.source_manifest_path.parent,)
        dependencies += tuple(
            (str(member), _fingerprint(_safe_path(str(member), path)))
            for member in members
        )
        manifest_binding = canonical_json(
            {
                **json.loads(manifest_binding),
                "dataset_context": {
                    "source": dataset.source,
                    "revision": dataset.revision,
                    "license": dataset.license,
                    "source_manifest_sha256": sha256_file(dataset.source_manifest_path),
                },
            }
        ).decode("utf-8")
    return (
        path.resolve(strict=True),
        artifact.kind,
        artifact.version,
        artifact.identifier,
        artifact.sha256,
        _fingerprint(path),
        dependencies,
        artifact.state,
        manifest_binding,
    )


def _reuse_verified_artifact(
    artifact: Artifact,
    source: Path,
    memo: dict[tuple[object, ...], _VerifiedArtifact],
    *,
    dataset: DatasetConfig | None = None,
) -> dict[str, object]:
    """Fail closed if a resolved artifact or its path changed before publication."""
    if artifact.from_phase is not None or artifact.path is None:
        raise ValueError("unresolved artifact cannot reuse verification")
    path = _safe_path(artifact.path, source)
    key = _artifact_key(artifact, path, dataset)
    evidence = memo.get(key)
    if (
        not isinstance(evidence, _VerifiedArtifact)
        or evidence._seal is not _MEMO_SEAL
        or evidence.issuer_pid != os.getpid()
        or evidence not in _MINTED_ARTIFACTS
        or evidence.key != key
        or evidence.identity
        != {
            "kind": artifact.kind,
            "version": artifact.version,
            "identifier": artifact.identifier,
            "sha256": artifact.sha256,
            "path": str(path),
        }
    ):
        raise ValueError(f"artifact changed since resolution: {path}")
    return dict(evidence.identity)


def verify_artifact(
    artifact: Artifact,
    source: Path,
    *,
    dataset: DatasetConfig | None = None,
    memo: dict[tuple[object, ...], _VerifiedArtifact] | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    """Verify one pinned external identity and return JSON-native identity/availability."""
    validate_mode(verification_mode)
    if artifact.from_phase is not None:
        raise ValueError(
            f"{artifact.kind} planned output {artifact.identifier} is unresolved until execution"
        )
    if artifact.path is None or artifact.sha256 is None:
        raise ValueError(f"{artifact.kind} external artifact lacks path or digest")
    try:
        path = _safe_path(artifact.path, source)
        key = _artifact_key(artifact, path, dataset)
        cached = memo.get(key) if memo is not None else None
        if (
            isinstance(cached, _VerifiedArtifact)
            and cached._seal is _MEMO_SEAL
            and cached.issuer_pid == os.getpid()
            and cached.key == key
            and cached in _MINTED_ARTIFACTS
            and (verification_mode != "cold" or cached.cold_verified)
        ):
            return dict(cached.identity)
        identity = {
            "kind": artifact.kind,
            "version": artifact.version,
            "identifier": artifact.identifier,
            "sha256": artifact.sha256,
            "path": str(path),
        }
        binding = artifact_binding(key, identity)
        hit = (
            verification_mode == "verified_reuse"
            and proof_store is not None
            and proof_store.lookup(binding)
        )
        if not hit:
            if proof_store is None or verification_mode == "cold":
                if (
                    artifact.kind == "tokenizer"
                    and dataset is not None
                    and dataset.source == "snapshot"
                ):
                    _verify_domain(artifact, path, dataset=dataset)
                else:
                    _verify_domain(artifact, path)
            else:
                _verify_domain(
                    artifact,
                    path,
                    dataset=dataset,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
        if _artifact_key(artifact, path, dataset) != key:
            raise ValueError(f"artifact changed during verification: {path}")
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        raise ValueError(
            f"invalid {artifact.kind} artifact at {artifact.path}: {error}"
        ) from error
    verified = {
        "kind": artifact.kind,
        "version": artifact.version,
        "identifier": artifact.identifier,
        "sha256": artifact.sha256,
        "path": str(path),
    }
    evidence = _VerifiedArtifact(verified, key, _seal=_MEMO_SEAL, cold_verified=not hit)
    _MINTED_ARTIFACTS.add(evidence)
    if not hit and verification_mode == "verified_reuse" and proof_store is not None:
        proof_store.record(binding, evidence)
    if memo is not None:
        memo[key] = evidence
    return dict(verified)


def verify_inputs(
    plan: ExperimentPlan,
    source: Path,
    *,
    dataset: DatasetConfig | None = None,
    memo: dict[tuple[object, ...], _VerifiedArtifact] | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, dict[str, object]]:
    """Resolve named plan inputs only; planned phase outputs remain unresolved."""
    return {
        name: verify_artifact(
            plan.artifacts[reference],
            source,
            dataset=dataset,
            memo=memo,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        for name, reference in plan.inputs.items()
    }
