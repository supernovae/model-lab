"""Portable, immutable preflight evidence; pilots execute only in subprocesses."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from sparselab.canonical import canonical_digest, reject_duplicate_keys
from sparselab.config.models import RunConfig
from sparselab.data.allocation import (
    copy_allocation_bundle,
    load_allocation_manifest,
    load_semantic_retriever,
)
from sparselab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from sparselab.data.packing import (
    PreparedData,
    _publish_prepared_directory,
    _tokenizer_sha256,
    load_prepared_data,
    prepare_data,
)
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.data.tokenizer import load_tokenizer
from sparselab.data.verification import (
    VerifiedFile,
    _fingerprint,
    _owned_file,
    _receipt_from_proofs,
    _relocate_proofs,
    _verify_file_with_hasher,
    _written_file,
    verify_file,
)
from sparselab.memory import (
    calibrated_estimate,
    calibration_key,
    estimate_memory,
    parameter_inventory,
    plan_memory,
    write_resource_proposal,
)
from sparselab.model.inspection import inspection_report, named_tensor_inventory
from sparselab.model.portable_engram import load_portable_engram
from sparselab.owned_copy import owned_copy
from sparselab.resource_envelope import (
    ResourceEnvelope,
    check_envelope,
    current_process_rss_bytes,
)
from sparselab.runtime import (
    RuntimeInfo,
    discover_runtimes,
    select_device,
    validate_runtime,
)
from sparselab.runtime_attention import attention_probe
from sparselab.runtime_forecasting import (
    runtime_forecast_planning,
    warmup_estimate,
)
from sparselab.runtime_profile import RuntimeAuthorization, require_authorization
from sparselab.training.checkpoints import _atomic_json, _fsync_directory, _safe_member
from sparselab.training.manifest import config_sha256, sha256_file, source_identity
from sparselab.training.metrics import SCHEMA_VERSION, ExperimentStore
from sparselab.training.pilot_deadline import (
    PilotCancelled,
    PilotDeadlinePolicy,
    PilotProcessFailure,
    PilotSupervisorError,
    supervise_pilot,
)
from sparselab.training.preparation_guard import require_model_runtime_allowed
from sparselab.training.stages import ExperimentStage, StageHistory
from sparselab.verification_proofs import file_binding
from sparselab.workdir import ensure_work_dir

if TYPE_CHECKING:
    from sparselab.bottleneck_observations import BottleneckObserver
    from sparselab.verification_proofs import ProofStore


_LOGGER = logging.getLogger(__name__)

_LEVELS = {"inspect": 1, "validate": 2, "smoke": 3, "warmup": 4}
_MAX_REPORT_BYTES = 16 * 1024 * 1024


def inspect_runtime(config: RunConfig) -> RuntimeInfo:
    """Resolve passive readings without validating or allocating model tensors."""
    runtimes = discover_runtimes()
    backend = config.runtime.backend
    if backend == "auto":
        selected = str(select_device("auto"))
        backend = next(
            info.backend
            for info in runtimes
            if info.engine == "pytorch" and info.torch_device == selected
        )
    runtime = next(
        info
        for info in runtimes
        if info.engine == config.runtime.engine and info.backend == backend
    )
    if config.runtime.device_index != runtime.device_index:
        runtime = replace(
            runtime,
            device_index=config.runtime.device_index,
            torch_device=None,
            device_name=None,
            physical_device_id=None,
            device_total_bytes=None,
            device_free_bytes=None,
            device_recommended_bytes=None,
            device_driver_allocated_bytes=None,
            limitations=(
                *runtime.limitations,
                "passive memory readings for this device index are unavailable",
            ),
        )
    return runtime


def _seal(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = {**payload, "sha256": canonical_digest(payload)}
    _atomic_json(path, value)
    return value


def _read_sealed(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAX_REPORT_BYTES
    ):
        raise ValueError(f"invalid or oversized stage manifest: {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(payload, dict):
        raise TypeError("stage manifest must be an object")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_digest(body):
        raise ValueError("stage manifest digest mismatch")
    if payload.get("format_version") != 1:
        raise ValueError("unsupported stage bundle version")
    return payload


def _inventory(
    root: Path, *, copied: dict[str, VerifiedFile] | None = None
) -> list[dict[str, object]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in immutable stage assets: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            proof = copied.get(name) if copied is not None else None
            if proof is not None and (
                not _owned_file(proof)
                or proof.path != path.resolve(strict=True)
                or proof.fingerprint != _fingerprint(path)
            ):
                raise ValueError(f"copied stage asset changed: {path}")
            result.append(
                {
                    "relative_path": name,
                    "sha256": proof.sha256 if proof is not None else sha256_file(path),
                    "size_bytes": (
                        proof.size_bytes if proof is not None else path.stat().st_size
                    ),
                }
            )
    return result


def _verify_inventory(
    root: Path,
    inventory: object,
    *,
    memo: dict[object, VerifiedFile] | None = None,
    published_manifest: str | None = None,
    manifest_sha256: str,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
) -> dict[str, VerifiedFile]:
    if not isinstance(inventory, list):
        raise TypeError("stage inventory must be a list")
    seen: set[str] = set()
    closure = {
        "manifest_sha256": manifest_sha256,
        "inventory_sha256": canonical_digest(inventory),
        "published_manifest": published_manifest,
    }
    proofs: dict[str, VerifiedFile] = {}
    from sparselab.training.pilot_progress import current_pilot_progress

    active = current_pilot_progress() is not None
    if active:

        def hash_staged_file(member: Path) -> str:
            return sha256_file(member, progress_phase="stage_bundle_verification")

    for item in inventory:
        if not isinstance(item, dict):
            raise TypeError("invalid stage inventory row")
        name = item.get("relative_path")
        path = _safe_member(root, name)
        if (
            not isinstance(name, str)
            or path is None
            or not path.is_file()
            or name in seen
        ):
            raise ValueError(f"unsafe or duplicate stage member: {name}")
        seen.add(name)
        if (
            type(item.get("size_bytes")) is not int
            or path.stat().st_size != item["size_bytes"]
        ):
            raise ValueError(f"stage member length mismatch: {name}")
        if not isinstance(item.get("sha256"), str):
            raise ValueError(f"invalid stage member digest: {name}")  # noqa: TRY004 - invalid serialized schema
        binding = (
            file_binding(
                path,
                item["sha256"],
                kind="stage_inventory_file",
                identifier=name,
                closure=closure,
            )
            if verification_mode == "verified_reuse" and proof_store is not None
            else None
        )
        if active:
            proofs[name] = _verify_file_with_hasher(
                path,
                expected_sha256=item["sha256"],
                memo=memo,
                hash_file=hash_staged_file,
                proof_store=proof_store,
                verification_mode=verification_mode,
                binding=binding,
            )
        else:
            proofs[name] = verify_file(
                path,
                expected_sha256=item["sha256"],
                memo=memo,
                proof_store=proof_store,
                verification_mode=verification_mode,
                binding=binding,
            )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if published_manifest is not None:
        actual.discard(published_manifest)
    if actual != seen:
        raise ValueError("stage inventory differs from actual files")
    return proofs


def _load_inventory_prepared(
    assets: Path, proofs: dict[str, VerifiedFile], *, byte_enabled: bool
) -> Any:
    root = assets / "data"
    manifest_proof = proofs["data/manifest.json"]
    verify_file(
        root / "manifest.json",
        expected_sha256=manifest_proof.sha256,
        memo={(manifest_proof.path, manifest_proof.fingerprint): manifest_proof},
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = {
        name.removeprefix("data/"): proof
        for name, proof in proofs.items()
        if name.startswith("data/") and name.endswith(".npy")
    }
    receipt = _receipt_from_proofs(root, manifest, arrays)
    return load_prepared_data(
        root, byte_enabled=byte_enabled, verification="structural", receipt=receipt
    )


def _pilot_story_dataset(
    config: RunConfig, data: PreparedData, eos: int | None
) -> MatchedStoryDataset:
    matched = config.training.matched_stories
    if matched is None or eos is None:
        raise ValueError("matched pilot requires story configuration and EOS")
    return MatchedStoryDataset(
        data.train,
        config.training.seq_len,
        eos,
        layout=matched.layout,
        targets_per_update=matched.targets_per_update,
    )


def pilot_config(
    config: RunConfig,
    purpose: str,
    root: Path,
    *,
    matched_dataset: MatchedStoryDataset | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
) -> RunConfig:
    if purpose not in {"smoke", "warmup"}:
        raise ValueError("invalid pilot purpose")
    steps = getattr(config.staging, f"{purpose}_steps")
    targets = (
        steps
        * config.training.micro_batch_size
        * config.training.gradient_accumulation
        * config.training.seq_len
    )
    if config.training.matched_stories is not None:
        if matched_dataset is None:
            # Derive from authenticated staged bytes, never the mutable source
            # cache. Bundle verification can supply its already verified dataset.
            proofs: dict[str, VerifiedFile] = {}
            verify_prepared_inputs(
                root,
                config,
                proof_store=proof_store,
                verification_mode=verification_mode,
                _proofs=proofs,
            )
            data = _load_inventory_prepared(
                root / "assets",
                proofs,
                byte_enabled=config.model.memory in {"byte", "portable"},
            )
            tokenizer = load_tokenizer(root / "assets" / "tokenizer.json")
            matched_dataset = _pilot_story_dataset(
                config, data, tokenizer.token_to_id("<eos>")
            )
        targets = matched_dataset.targets_for_updates(steps)
    payload = config.model_dump(mode="json")
    payload["training"].update(max_steps=steps, max_tokens=targets)
    payload["optimizer"]["warmup_steps"] = min(config.optimizer.warmup_steps, steps - 1)
    if payload["optimizer"].get("decay_steps") is not None:
        payload["optimizer"]["decay_steps"] = min(
            payload["optimizer"]["decay_steps"], steps
        )
    payload["logging"]["root_dir"] = str(root / "pilots" / purpose)
    if config.dataset.allocation_manifest_path is not None:
        payload["dataset"]["allocation_manifest_path"] = str(
            root
            / "assets"
            / "allocation"
            / config.dataset.allocation_manifest_path.name
        )
    return RunConfig.model_validate(payload)


def verify_stage_bundle(
    root: Path,
    config: RunConfig,
    *,
    purpose: str = "training",
    allow_runtime_drift: bool = False,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
    _proofs: dict[str, VerifiedFile] | None = None,
) -> dict[str, Any]:
    """Verify frozen inputs; a pilot may read only the sealed pre-pilot input manifest."""
    root = root.resolve(strict=True)
    inputs = _read_sealed(root / "inputs.json")
    base = RunConfig.model_validate(inputs["requested_config"])
    matched_pilot = purpose != "training" and base.training.matched_stories is not None
    expected = (
        None
        if matched_pilot
        else base
        if purpose == "training"
        else pilot_config(base, purpose, root)
    )
    if expected is not None and config_sha256(
        config.model_dump(mode="json")
    ) != config_sha256(expected.model_dump(mode="json")):
        raise ValueError("stage bundle configuration differs from requested execution")
    if (
        inputs.get("source_identity_sha256") != source_identity()["sha256"]
        and not allow_runtime_drift
    ):
        raise ValueError(
            "stage bundle executable source identity differs; restage explicitly"
        )
    proof_memo: dict[object, VerifiedFile] = {}
    assets = root / "assets"
    proofs = _verify_inventory(
        assets,
        inputs.get("artifacts"),
        memo=proof_memo,
        manifest_sha256=inputs["sha256"],
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    data = _load_inventory_prepared(
        assets, proofs, byte_enabled=base.model.memory in {"byte", "portable"}
    )
    tokenizer = load_tokenizer(root / "assets" / "tokenizer.json")
    if matched_pilot:
        expected = pilot_config(
            base,
            purpose,
            root,
            matched_dataset=_pilot_story_dataset(
                base, data, tokenizer.token_to_id("<eos>")
            ),
        )
        if config_sha256(config.model_dump(mode="json")) != config_sha256(
            expected.model_dump(mode="json")
        ):
            raise ValueError(
                "stage bundle configuration differs from requested execution"
            )
    if tokenizer.get_vocab_size() != base.model.vocab_size:
        raise ValueError("staged tokenizer vocabulary does not match model")
    if base.dataset.corpus_release_path is not None:
        from sparselab.workers.bundles import verify_portable_corpus_binding

        verify_portable_corpus_binding(base, assets)
    _verify_allocation_assets(
        root / "assets",
        config,
        data,
        inputs.get("source_identity_sha256"),
        tokenizer,
    )
    if not len(data.train) or not len(data.validation):
        raise ValueError("staged data is empty")
    if purpose == "training":
        bundle = _read_sealed(root / "bundle.json")
        if bundle.get("status") != "complete" or bundle.get("through") not in {
            "validate",
            "smoke",
            "warmup",
        }:
            raise ValueError("stage bundle has not completed asset validation")
        _verify_inventory(
            root,
            bundle.get("artifacts"),
            memo=proof_memo,
            published_manifest="bundle.json",
            manifest_sha256=bundle["sha256"],
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if bundle.get("inputs_sha256") != inputs["sha256"]:
            raise ValueError("stage bundle input identity mismatch")
    else:
        bundle = {"sha256": inputs["sha256"], "pilot_reports": [], "stages": []}
    if _proofs is not None:
        _proofs.update(proofs)
    return {**bundle, "assets_root": str(root / "assets")}


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    verified: dict[str, VerifiedFile] | None = None,
    copy_observations: list[dict[str, object]] | None = None,
    observer: BottleneckObserver | None = None,
) -> dict[str, VerifiedFile]:
    """Make isolated owned copies with destination SHA and source fingerprint checks."""
    from sparselab.training.pilot_progress import (
        current_pilot_progress,
        emit_pilot_progress,
    )

    if source.is_symlink():
        raise ValueError(f"symlink asset root: {source}")
    if source.is_dir():
        paths = sorted(source.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError(f"symlink in immutable stage assets: {source}")
        files = [path for path in paths if path.is_file()]
        if any(not path.is_file() and not path.is_dir() for path in paths):
            raise ValueError(f"unsupported stage asset member: {source}")
        destination.mkdir()
        for path in paths:
            if path.is_dir():
                (destination / path.relative_to(source)).mkdir(exist_ok=True)
    else:
        files = [source]
        destination.parent.mkdir(parents=True, exist_ok=True)
    if verified is not None and not set(verified).issubset(
        {path.relative_to(source).as_posix() for path in files}
    ):
        raise ValueError("verified source inventory differs from copied files")
    active = current_pilot_progress() is not None
    total = sum(path.stat().st_size for path in files) if active else 0
    subject = None
    if active:
        import secrets

        subject = secrets.token_hex(8)
    copied_bytes = 0
    reported = 0
    proofs: dict[str, VerifiedFile] = {}
    for path in files:
        name = (
            path.relative_to(source).as_posix() if source.is_dir() else destination.name
        )
        target = destination / name if source.is_dir() else destination
        old = verified.get(name) if verified is not None else None
        base = copied_bytes

        def progress(count: int, *, base: int = base) -> None:
            nonlocal copied_bytes, reported
            copied_bytes = base + count
            if active and (
                copied_bytes - reported >= 64 * 1024 * 1024 or copied_bytes == total
            ):
                emit_pilot_progress(
                    "progress",
                    "run_input_materialization",
                    counter="bytes",
                    value=copied_bytes,
                    total=total,
                    subject=subject,
                )
                reported = copied_bytes

        with (
            observer.phase("stage_owned_copy", host_kind="copy_bound")
            if observer is not None
            else nullcontext()
        ):
            result = owned_copy(path, target, proof=old, progress=progress)
        proofs[name] = result.proof
        if copy_observations is not None:
            copy_observations.append(
                {"mechanism": result.mechanism, "logical_bytes": result.logical_bytes}
            )
    return proofs


def _link_stage_assets(
    work: Path,
    proofs: dict[str, VerifiedFile],
    *,
    copy_observations: list[dict[str, object]] | None = None,
    observer: BottleneckObserver | None = None,
) -> tuple[dict[str, VerifiedFile], dict[str, VerifiedFile]]:
    """Link only freshly copied, stage-owned private assets, never canonical input."""
    source, destination = work / "prepared" / "assets", work / "assets"
    if work.stat().st_uid != os.getuid() or work.stat().st_mode & 0o077:
        raise ValueError("stage linking requires a private owned staging directory")
    destination.mkdir()
    result, refreshed = {}, {}
    for path in sorted(source.rglob("*")):
        target = destination / path.relative_to(source)
        if path.is_symlink():
            raise ValueError("symlink in private stage assets")
        if path.is_dir():
            target.mkdir()
            continue
        name = path.relative_to(source).as_posix()
        proof = proofs.get(name) or verify_file(path)
        before = _fingerprint(path)
        if (
            not _owned_file(proof)
            or proof.path != path.resolve()
            or proof.fingerprint != before
        ):
            raise ValueError("private stage source seal changed before linking")
        with (
            observer.phase("stage_private_asset_link", host_kind="copy_bound")
            if observer is not None
            else nullcontext()
        ):
            try:
                os.link(path, target)
            except OSError:
                copied = owned_copy(path, target, proof=proof)
                result[name], refreshed[name] = copied.proof, proof
                mechanism = copied.mechanism
            else:
                after = _fingerprint(path)
                if after[:5] != before[:5] or _fingerprint(target) != after:
                    raise ValueError("private stage asset changed while linking")
                # Linking changes ctime. Authenticate this intentional transition;
                # a changed fingerprint alone is never sufficient authority.
                linked = verify_file(target, expected_sha256=proof.sha256)
                if _fingerprint(path) != after:
                    raise ValueError(
                        "private stage source changed during link authentication"
                    )
                refreshed[name] = _written_file(path, linked.sha256, linked.size_bytes)
                result[name], mechanism = linked, "stage_private_hardlink"
        if copy_observations is not None:
            copy_observations.append(
                {"mechanism": mechanism, "logical_bytes": result[name].size_bytes}
            )
    return result, refreshed


def _verify_allocation_assets(
    assets: Path,
    config: RunConfig,
    data: Any,
    source_sha256: object,
    tokenizer: Any,
) -> None:
    allocation_path = config.dataset.allocation_manifest_path
    metadata = data.manifest.get("allocation")
    if allocation_path is None:
        if metadata is not None or (assets / "allocation").exists():
            raise ValueError("prepared inputs contain an unexpected allocation")
        return
    if not isinstance(source_sha256, str):
        raise TypeError("prepared allocation source identity must be a string")
    if not isinstance(metadata, dict):
        raise TypeError("prepared allocation metadata must be an object")
    allocation = load_allocation_manifest(
        assets / "allocation" / allocation_path.name,
        source_identity_sha256=source_sha256,
        tokenizer_sha256=_tokenizer_sha256(tokenizer),
    )
    if allocation.sha256 != metadata.get("manifest_sha256"):
        raise ValueError("prepared data and allocation manifests differ")
    for split, values in (("train", data.train), ("validation", data.validation)):
        allocation.split(split, token_count=len(values))
    retriever = load_semantic_retriever(allocation)
    if (retriever is None) != (config.model.semantic_memory_dim is None):
        raise ValueError("model.semantic_memory_dim must match the allocation pack")
    if (
        retriever is not None
        and retriever.memory_dim != config.model.semantic_memory_dim
    ):
        raise ValueError("model.semantic_memory_dim differs from semantic pack width")


def materialize_prepared_inputs(
    config: RunConfig,
    destination: Path,
    *,
    prepared_inputs: Path | None = None,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
    _proofs: dict[str, VerifiedFile] | None = None,
    observer: BottleneckObserver | None = None,
    copy_observations: list[dict[str, object]] | None = None,
) -> Path:
    """Publish immutable, engine-neutral training inputs without probing a runtime.

    ``prepared_inputs`` is a prior materialized root (or its ``assets`` directory).
    It is verified before copying, so callers never use controller-local source paths
    after this boundary.
    """
    validate_tokenizer_batch_limits(
        tokenizer_batch_documents, tokenizer_batch_source_bytes
    )
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=destination,
            rss_bytes=current_process_rss_bytes(),
        )
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"prepared input destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        work = Path(temporary)
        assets = work / "assets"
        copied: dict[str, VerifiedFile] = {}
        source_digest = source_identity()["sha256"]
        if prepared_inputs is None:
            tokenizer = load_tokenizer(config.tokenizer.path)
            if tokenizer.get_vocab_size() != config.model.vocab_size:
                raise ValueError("tokenizer vocabulary does not match model")
            data = prepare_data(
                config,
                tokenizer,
                resource_envelope=resource_envelope,
                tokenizer_batch_documents=tokenizer_batch_documents,
                tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                proof_store=proof_store,
                verification_mode=verification_mode,
                observer=observer,
            )
            assets.mkdir()
            copied.update(
                _copy_tree(
                    config.tokenizer.path,
                    assets / "tokenizer.json",
                    copy_observations=copy_observations,
                    observer=observer,
                )
            )
            tokenizer_manifest = config.tokenizer.path.with_name(
                "tokenizer_manifest.json"
            )
            if tokenizer_manifest.is_file():
                copied.update(
                    _copy_tree(
                        tokenizer_manifest,
                        assets / tokenizer_manifest.name,
                        copy_observations=copy_observations,
                        observer=observer,
                    )
                )
            if config.dataset.corpus_release_path is not None:
                from sparselab.workers.bundles import _copy_portable_corpus

                # Pilots consume these inputs before dispatch, and must not reopen
                # controller-local export configs to authenticate the tokenizer.
                _copy_portable_corpus(config, assets)
            copied.update(
                {
                    f"data/{name}": proof
                    for name, proof in _copy_tree(
                        data.root,
                        assets / "data",
                        verified=dict(data.receipt.proofs),
                        copy_observations=copy_observations,
                        observer=observer,
                    ).items()
                }
            )
            from sparselab.evaluation.panel_data import copy_fixed_panels

            copy_fixed_panels(config, assets, training_manifest=data.manifest)
            allocation_source = config.dataset.allocation_manifest_path
            if allocation_source is not None:
                allocation = load_allocation_manifest(
                    allocation_source,
                    source_identity_sha256=str(source_digest),
                    tokenizer_sha256=_tokenizer_sha256(tokenizer),
                )
                prepared_allocation = data.manifest.get("allocation")
                if (
                    not isinstance(prepared_allocation, dict)
                    or prepared_allocation.get("manifest_sha256") != allocation.sha256
                ):
                    raise ValueError("prepared data differs from allocation manifest")
                copy_allocation_bundle(allocation, assets / "allocation")
            if config.model.memory_package_path is not None:
                copied.update(
                    {
                        f"portable_package/{name}"
                        if config.model.memory_package_path.is_dir()
                        else "portable_package": proof
                        for name, proof in _copy_tree(
                            config.model.memory_package_path,
                            assets / "portable_package",
                            copy_observations=copy_observations,
                            observer=observer,
                        ).items()
                    }
                )
                load_portable_engram(
                    assets / "portable_package",
                    expected_shape=(
                        config.model.memory_table_size,
                        config.model.memory_dim,
                    ),
                    expected_ngram_size=config.model.memory_ngram_size,
                )
        else:
            source = prepared_inputs.resolve(strict=True)
            if source.name == "assets":
                source = source.parent
            verified: dict[str, VerifiedFile] = {}
            with (
                observer.phase(
                    "prepared_input_verification", host_kind="verification_bound"
                )
                if observer is not None
                else nullcontext()
            ):
                verify_prepared_inputs(
                    source,
                    config,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                    _proofs=verified,
                )
            copied.update(
                _copy_tree(
                    source / "assets",
                    assets,
                    verified=verified,
                    copy_observations=copy_observations,
                    observer=observer,
                )
            )
        _seal(
            work / "inputs.json",
            {
                "format_version": 1,
                "requested_config": config.model_dump(mode="json"),
                "source_identity_sha256": source_digest,
                "artifacts": _inventory(assets, copied=copied),
            },
        )
        if resource_envelope is not None:
            check_envelope(
                resource_envelope,
                workspace=destination,
                rss_bytes=current_process_rss_bytes(),
            )
        _publish_prepared_directory(work, destination)
        _fsync_directory(destination.parent)
        if _proofs is not None:
            _proofs.update(_relocate_proofs(destination / "assets", copied))
    return destination


def _prepared_config_matches(saved: RunConfig, requested: RunConfig) -> bool:
    """Allow only a worker's recorded auto-runtime resolution over frozen inputs."""
    if config_sha256(saved.model_dump(mode="json")) == config_sha256(
        requested.model_dump(mode="json")
    ):
        return True
    base = saved.model_dump(mode="json")
    effective = requested.model_dump(mode="json")
    runtime = base["runtime"]
    resolved = effective["runtime"]
    if (
        runtime["engine"] != resolved["engine"]
        or runtime["device_index"] != resolved["device_index"]
    ):
        return False
    if runtime["backend"] == "auto":
        runtime["backend"] = resolved["backend"]
    if runtime["precision"] == "auto":
        runtime["precision"] = resolved["precision"]
    return config_sha256(base) == config_sha256(effective)


def verify_prepared_inputs(
    root: Path,
    config: RunConfig,
    *,
    allow_runtime_drift: bool = False,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
    _proofs: dict[str, VerifiedFile] | None = None,
    _seals: dict[str, VerifiedFile] | None = None,
) -> dict[str, Any]:
    """Verify a materialized input root without acquiring a target accelerator."""
    root = root.resolve(strict=True)
    inputs = _read_sealed(root / "inputs.json")
    saved = RunConfig.model_validate(inputs["requested_config"])
    if not _prepared_config_matches(saved, config):
        raise ValueError(
            "prepared inputs configuration differs from requested execution"
        )
    if (
        inputs.get("source_identity_sha256") != source_identity()["sha256"]
        and not allow_runtime_drift
    ):
        raise ValueError("prepared inputs executable source identity differs")
    assets = root / "assets"
    proofs = _verify_inventory(
        assets,
        inputs.get("artifacts"),
        memo={
            (proof.path, proof.fingerprint): proof for proof in (_seals or {}).values()
        },
        manifest_sha256=inputs["sha256"],
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    data = _load_inventory_prepared(
        assets, proofs, byte_enabled=config.model.memory in {"byte", "portable"}
    )
    tokenizer = load_tokenizer(root / "assets" / "tokenizer.json")
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("prepared tokenizer vocabulary does not match model")
    if config.dataset.corpus_release_path is not None:
        from sparselab.workers.bundles import verify_portable_corpus_binding

        verify_portable_corpus_binding(config, assets)
    if not len(data.train) or not len(data.validation):
        raise ValueError("prepared data is empty")
    _verify_allocation_assets(
        root / "assets",
        config,
        data,
        inputs.get("source_identity_sha256"),
        tokenizer,
    )
    if config.model.memory_package_path is not None:
        load_portable_engram(
            root / "assets" / "portable_package",
            expected_shape=(config.model.memory_table_size, config.model.memory_dim),
            expected_ngram_size=config.model.memory_ngram_size,
        )
    if _proofs is not None:
        _proofs.update(proofs)
    return {**inputs, "assets_root": str(root / "assets")}


def _run_pilot(
    root: Path,
    purpose: str,
    *,
    inherited_fds: tuple[int, ...] = (),
    cancel_path: Path | None = None,
    pilot_deadline_policy: PilotDeadlinePolicy | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
) -> dict[str, Any]:
    if cancel_path is not None and cancel_path.exists():
        raise InterruptedError("staging cancelled before pilot initialization")
    directory = root / "pilots" / purpose
    directory.mkdir(parents=True)
    log_path = directory / "execution.log"
    ensure_work_dir()
    requested = RunConfig.model_validate(
        _read_sealed(root / "inputs.json")["requested_config"]
    )
    expected_steps = getattr(requested.staging, f"{purpose}_steps")
    try:
        supervise_pilot(
            [
                sys.executable,
                "-m",
                "sparselab.training.pilot",
                str(root),
                purpose,
                *([] if cancel_path is None else ["--cancel-path", str(cancel_path)]),
                *(
                    ["--verification-root", str(proof_store.root)]
                    if verification_mode == "verified_reuse" and proof_store is not None
                    else ["--cold-verify"]
                ),
            ],
            purpose=purpose,
            directory=directory,
            policy=pilot_deadline_policy,
            cancel_path=cancel_path,
            inherited_fds=inherited_fds,
            expected_steps=expected_steps,
        )
    except PilotCancelled as error:
        raise InterruptedError("staging pilot cancelled") from error
    except PilotProcessFailure as error:
        failure_path = directory / "failure.json"
        if failure_path.is_file():
            failure = _read_sealed(failure_path)
            if failure.get("kind") == "out_of_memory":
                raise MemoryError(
                    f"{purpose} pilot exhausted memory: {failure['message']}"
                ) from error
            if failure.get("kind") == "cancelled":
                raise InterruptedError("staging pilot cancelled") from error
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            detail = log.read(65536).decode("utf-8", errors="replace")
        raise RuntimeError(f"{purpose} pilot failed: {detail}") from error
    if cancel_path is not None and cancel_path.exists():
        raise InterruptedError("staging cancelled at pilot boundary")
    report = _read_sealed(directory / "report.json")
    if report.get("purpose") != purpose or report.get("status") != "complete":
        raise ValueError("pilot returned incomplete or mismatched evidence")
    if (
        report.get("step") != expected_steps
        or report.get("timed_updates") != expected_steps
    ):
        raise ValueError("pilot report does not account for configured updates")
    return {
        "run_id": report["run_id"],
        "purpose": purpose,
        "relative_path": (directory / "report.json").relative_to(root).as_posix(),
        "sha256": report["sha256"],
        "report": report,
        "supervision": _read_sealed(directory / "supervisor-completion.json"),
    }


def stage(
    config: RunConfig,
    output: Path,
    through: str = "smoke",
    *,
    prepared_inputs: Path | None = None,
    allow_runtime_drift: bool = False,
    inherited_fds: tuple[int, ...] = (),
    cancel_path: Path | None = None,
    authorization: RuntimeAuthorization | None = None,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    pilot_deadline_policy: PilotDeadlinePolicy | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
    copy_observations: list[dict[str, object]] | None = None,
    observer: BottleneckObserver | None = None,
) -> Path:
    require_model_runtime_allowed()
    if through not in _LEVELS:
        raise ValueError("through must be inspect, validate, smoke, or warmup")
    if prepared_inputs is not None and allow_runtime_drift:
        raise ValueError("existing prepared inputs cannot allow source/runtime drift")
    validate_tokenizer_batch_limits(
        tokenizer_batch_documents, tokenizer_batch_source_bytes
    )
    if _LEVELS[through] >= 2:
        require_authorization(config, authorization)
    output = output.absolute()
    if prepared_inputs is not None and (output.exists() or output.is_symlink()):
        raise FileExistsError("existing-input staging requires a new output")
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=output,
            rss_bytes=current_process_rss_bytes(),
        )
    if _LEVELS[through] >= 3:
        # Validation publishes prepared assets only. Checkpoint space is needed
        # only for the model-executing smoke/warmup stages.
        from sparselab.workspace_preflight import check_storage, require_storage

        checkpoint = int(inspection_report(config)["estimated_checkpoint_bytes"])
        require_storage([check_storage(output, projected_bytes=4 * checkpoint)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(f".{output.name}.stage.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError("another writer owns this stage output") from error
        prepared_proofs: dict[str, VerifiedFile] = {}
        if prepared_inputs is not None:
            if output.exists() or output.is_symlink():
                raise FileExistsError("existing-input staging requires a new output")
            prepared_root = prepared_inputs.resolve(strict=True)
            if prepared_root.name == "assets":
                prepared_root = prepared_root.parent
            with (
                observer.phase(
                    "stage_prepared_input_verification",
                    host_kind="verification_bound",
                )
                if observer is not None
                else nullcontext()
            ):
                prepared_identity = verify_prepared_inputs(
                    prepared_root,
                    config,
                    _proofs=prepared_proofs,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
        if output.exists():
            bundle = _read_sealed(output / "bundle.json")
            if bundle.get("status") != "complete" or bundle.get("through") != through:
                raise FileExistsError("stage output is conflicting or incomplete")
            if bundle.get("config_sha256") != config_sha256(
                config.model_dump(mode="json")
            ):
                raise FileExistsError("stage output belongs to another configuration")
            if (
                bundle.get("source_identity_sha256") != source_identity()["sha256"]
                and not allow_runtime_drift
            ):
                raise FileExistsError(
                    "stage output belongs to another executable source"
                )
            with (
                observer.phase(
                    "stage_reuse_verification",
                    host_kind="verification_bound",
                    cache_event="hit",
                )
                if observer is not None
                else nullcontext()
            ):
                _verify_inventory(
                    output,
                    bundle.get("artifacts"),
                    published_manifest="bundle.json",
                    manifest_sha256=bundle["sha256"],
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            if through != "inspect":
                verify_stage_bundle(
                    output,
                    config,
                    allow_runtime_drift=allow_runtime_drift,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                if config.tokenizer.path.exists() and sha256_file(
                    config.tokenizer.path
                ) != sha256_file(output / "assets" / "tokenizer.json"):
                    raise FileExistsError("requested tokenizer changed since staging")
            return output
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.", dir=output.parent
        ) as temporary:
            work = Path(temporary)
            history = StageHistory()
            reports: list[dict[str, Any]] = []
            copied_assets: dict[str, VerifiedFile] = {}
            runtime = estimate = None
            report: dict[str, Any] = {
                "format_version": 1,
                "requested_config": config.model_dump(mode="json"),
                "effective_config": config.model_dump(mode="json"),
                "through": through,
                "metric_schema_version": SCHEMA_VERSION,
            }
            try:
                history.start(ExperimentStage.CONFIGURED)
                history.finish()
                history.start(ExperimentStage.INSPECTED)
                runtime = inspect_runtime(config)
                inventory = parameter_inventory(config)
                estimate = estimate_memory(config, runtime, inventory)
                forecast = runtime_forecast_planning(
                    config.logging.root_dir,
                    config,
                    runtime,
                    total_targets=config.training.max_tokens,
                )
                source_digest = source_identity()["sha256"]
                report.update(
                    inventory=asdict(inventory),
                    runtime=runtime.as_dict(),
                    runtime_authorization=(
                        authorization.as_dict() if authorization is not None else None
                    ),
                    resource_envelope=(
                        resource_envelope.model_dump(mode="json")
                        if resource_envelope is not None
                        else None
                    ),
                    tokenizer_batch_documents=tokenizer_batch_documents,
                    tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                    pilot_deadline_policy=(
                        (pilot_deadline_policy or PilotDeadlinePolicy()).model_dump(
                            mode="json"
                        )
                    ),
                    estimate=asdict(estimate),
                    source_identity_sha256=source_digest,
                    tensor_inventory={
                        name: asdict(spec)
                        for name, spec in named_tensor_inventory(
                            config.model,
                            config.attention,
                            trainable_parameters=config.training.trainable_parameters,
                        ).items()
                    },
                    runtime_forecast={
                        "schema_version": forecast["schema_version"],
                        "runtime_signature": forecast["runtime_signature"],
                        "planning": forecast["planning"],
                        "warmup_calibrated": None,
                    },
                )
                history.finish()
                inputs_digest = None
                calibration_identity: str | None = None
                observations: list[dict[str, object]] = []
                if _LEVELS[through] >= 2:
                    history.start(ExperimentStage.VALIDATED)
                    runtime = validate_runtime(config, authorization=authorization)
                    forecast = runtime_forecast_planning(
                        config.logging.root_dir,
                        config,
                        runtime,
                        total_targets=config.training.max_tokens,
                    )
                    calibration_identity = calibration_key(
                        config, runtime, source_digest=str(source_digest)
                    )
                    observations = ExperimentStore.get_calibration(
                        config.logging.root_dir, calibration_identity
                    )
                    estimate = calibrated_estimate(
                        config, runtime, inventory, observations
                    )
                    report.update(runtime=runtime.as_dict(), estimate=asdict(estimate))
                    report["runtime_forecast"] = {
                        "schema_version": forecast["schema_version"],
                        "runtime_signature": forecast["runtime_signature"],
                        "planning": forecast["planning"],
                        "warmup_calibrated": None,
                    }
                    effective = config.model_dump(mode="json")
                    effective["runtime"]["backend"] = runtime.backend
                    if effective["runtime"]["precision"] == "auto":
                        effective["runtime"]["precision"] = "fp32"
                    if (
                        runtime.backend == "cuda"
                        and config.attention.kind == "dense"
                        and (
                            config.model.num_kv_heads is None
                            or config.model.num_kv_heads == config.model.num_heads
                        )
                    ):
                        report["attention_probe"] = attention_probe(
                            config, authorization=authorization
                        )
                    report["effective_config"] = effective
                    if estimate.result == "LIKELY_TO_EXCEED":
                        if config.runtime.memory.estimate_policy == "enforce":
                            write_resource_proposal(
                                plan_memory(config, runtime, estimate),
                                work / "proposal.yaml",
                            )
                            raise MemoryError(
                                "estimated training memory exceeds the explicit safe ceiling"
                            )
                        _LOGGER.warning(
                            "Estimated training memory exceeds the planning ceiling; "
                            "estimate_policy=observe continues with the unchanged configuration"
                        )
                    if prepared_inputs is None:
                        prepared_root = materialize_prepared_inputs(
                            config,
                            work / "prepared",
                            resource_envelope=resource_envelope,
                            tokenizer_batch_documents=tokenizer_batch_documents,
                            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                            proof_store=proof_store,
                            verification_mode=verification_mode,
                            _proofs=prepared_proofs,
                            copy_observations=copy_observations,
                            observer=observer,
                        )
                    if prepared_inputs is None:
                        prepared_identity = _read_sealed(prepared_root / "inputs.json")
                        copied_assets, prepared_proofs = _link_stage_assets(
                            work,
                            prepared_proofs,
                            copy_observations=copy_observations,
                            observer=observer,
                        )
                    else:
                        copied_assets = _copy_tree(
                            prepared_root / "assets",
                            work / "assets",
                            verified=prepared_proofs,
                            copy_observations=copy_observations,
                            observer=observer,
                        )
                    prepared_manifest = json.loads(
                        (work / "assets" / "data" / "manifest.json").read_text()
                    )
                    inputs = _seal(
                        work / "inputs.json",
                        {
                            "format_version": 1,
                            "requested_config": config.model_dump(mode="json"),
                            "source_identity_sha256": source_digest,
                            "producer_source_identity_sha256": prepared_identity.get(
                                "producer_source_identity_sha256",
                                prepared_identity["source_identity_sha256"],
                            ),
                            "prepared_production_source_identity_sha256": prepared_manifest[
                                "source_identity_sha256"
                            ],
                            "verification_source_identity_sha256": source_digest,
                            "parent_inputs_sha256": prepared_identity["sha256"],
                            "runtime_authorization": (
                                authorization.as_dict()
                                if authorization is not None
                                else None
                            ),
                            "resource_envelope": (
                                resource_envelope.model_dump(mode="json")
                                if resource_envelope is not None
                                else None
                            ),
                            "tokenizer_batch_documents": tokenizer_batch_documents,
                            "tokenizer_batch_source_bytes": tokenizer_batch_source_bytes,
                            "artifacts": _inventory(
                                work / "assets", copied=copied_assets
                            ),
                        },
                    )
                    inputs_digest = inputs["sha256"]
                    history.finish()
                for purpose, level, stage_name in (
                    ("smoke", 3, ExperimentStage.SMOKE_TEST),
                    ("warmup", 4, ExperimentStage.WARMUP),
                ):
                    if _LEVELS[through] < level:
                        continue
                    if calibration_identity is None:
                        raise RuntimeError(
                            "pilot stage lacks validated calibration identity"
                        )
                    history.start(stage_name)
                    pilot = _run_pilot(
                        work,
                        purpose,
                        inherited_fds=inherited_fds,
                        cancel_path=cancel_path,
                        pilot_deadline_policy=pilot_deadline_policy,
                        proof_store=proof_store,
                        verification_mode=verification_mode,
                    )
                    reports.append(pilot)
                    observations.extend(
                        ExperimentStore.get_calibration(
                            work / "pilots" / purpose, calibration_identity
                        )
                    )
                    estimate = calibrated_estimate(
                        config, runtime, inventory, observations
                    )
                    report["estimate"] = asdict(estimate)
                    peak = pilot["report"].get("observed_peak_bytes")
                    if estimate.result == "LIKELY_TO_EXCEED" or (
                        peak is not None
                        and estimate.capacity_ceiling_bytes is not None
                        and peak > estimate.capacity_ceiling_bytes
                    ):
                        if config.runtime.memory.estimate_policy == "enforce":
                            write_resource_proposal(
                                plan_memory(config, runtime, estimate),
                                work / "proposal.yaml",
                            )
                            raise MemoryError(
                                f"{purpose} measured or calibrated memory exceeds safe ceiling"
                            )
                        _LOGGER.warning(
                            "%s memory exceeds the planning ceiling; "
                            "estimate_policy=observe retains the measurements and unchanged configuration",
                            purpose,
                        )
                    if purpose == "warmup":
                        raw_updates = pilot["report"].get("update_observations", [])
                        updates: list[tuple[int, float]] = []
                        if isinstance(raw_updates, list):
                            for item in raw_updates:
                                targets = (
                                    item.get("targets")
                                    if isinstance(item, dict)
                                    else None
                                )
                                update_seconds = (
                                    item.get("update_seconds")
                                    if isinstance(item, dict)
                                    else None
                                )
                                if (
                                    type(targets) is int
                                    and isinstance(update_seconds, (int, float))
                                    and not isinstance(update_seconds, bool)
                                ):
                                    updates.append((targets, float(update_seconds)))
                        runtime_forecast = report["runtime_forecast"]
                        runtime_forecast["warmup_calibrated"] = warmup_estimate(
                            config.training.max_tokens,
                            updates,
                            pilot_run_id=str(pilot["run_id"]),
                        )
                    history.finish(
                        payload={
                            "pilot_run_id": pilot["run_id"],
                            "report_sha256": pilot["sha256"],
                        }
                    )
                report.update(
                    status="complete",
                    stages=[asdict(item) for item in history.records],
                    pilot_reports=reports,
                )
                _seal(work / "stage.json", report)
                _seal(
                    work / "bundle.json",
                    {
                        "format_version": 1,
                        "status": "complete",
                        "through": through,
                        "config_sha256": config_sha256(config.model_dump(mode="json")),
                        "source_identity_sha256": source_digest,
                        "inputs_sha256": inputs_digest,
                        "stages": report["stages"],
                        "pilot_reports": reports,
                        "runtime_forecast": report["runtime_forecast"],
                        "artifacts": _inventory(
                            work,
                            copied={
                                **{
                                    f"assets/{name}": proof
                                    for name, proof in copied_assets.items()
                                },
                                **(
                                    {
                                        f"prepared/assets/{name}": proof
                                        for name, proof in prepared_proofs.items()
                                    }
                                    if prepared_inputs is None
                                    else {}
                                ),
                            },
                        ),
                    },
                )
            except BaseException as error:
                if history.current is not None:
                    history.finish("failed", reason=str(error))
                if history.records:
                    history.start(ExperimentStage.FAILED)
                    history.finish("failed", reason=str(error))
                report.update(
                    status="failed",
                    reason=str(error),
                    stages=[asdict(item) for item in history.records],
                    pilot_reports=reports,
                )
                pilot_error = (
                    error
                    if isinstance(error, PilotSupervisorError)
                    else error.__cause__
                )
                if isinstance(pilot_error, PilotSupervisorError):
                    failure = pilot_error.evidence
                    report["pilot_failure"] = {
                        "error_type": type(error).__name__,
                        "timeout_kind": failure.get("timeout_kind"),
                        "reason": failure["reason"],
                    }
                    purpose = failure.get("purpose")
                    if purpose in {"smoke", "warmup"} and "sha256" in failure:
                        report["pilot_failure"].update(
                            ref=f"pilots/{purpose}/supervisor-failure.json",
                            sha256=failure["sha256"],
                        )
                _seal(work / "stage.json", report)
                if (
                    isinstance(error, MemoryError)
                    and runtime is not None
                    and estimate is not None
                    and not (work / "proposal.yaml").exists()
                ):
                    write_resource_proposal(
                        plan_memory(config, runtime, estimate), work / "proposal.yaml"
                    )
                _publish_prepared_directory(work, output)
                _fsync_directory(output.parent)
                raise
            if resource_envelope is not None:
                check_envelope(
                    resource_envelope,
                    workspace=output,
                    rss_bytes=current_process_rss_bytes(),
                )
            _publish_prepared_directory(work, output)
            _fsync_directory(output.parent)
    return output
