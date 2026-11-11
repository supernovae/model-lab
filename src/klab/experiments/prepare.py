"""Explicit, offline corpus variant preparation; never launches training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from klab.config.loading import load_config, load_tokenizer_config
from klab.config.models import RunConfig
from klab.corpus.acquisition import acquire, verify_acquisition
from klab.corpus.export import export_release, verify_release_export
from klab.corpus.pipeline import build
from klab.corpus.project import (
    ReleaseDeclaration,
    load_project,
    verify_fraction_tokenizer,
)
from klab.corpus.release import freeze, verify_release
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from klab.data.packing import prepare_data
from klab.data.tokenizer import load_tokenizer, train_tokenizer
from klab.experiments.artifacts import verify_artifact
from klab.experiments.plan import Artifact, CorpusVariant, ExperimentPlan
from klab.resource_envelope import (
    ResourceEnvelope,
    check_envelope,
    current_process_rss_bytes,
)
from klab.training.manifest import config_sha256, sha256_file
from klab.verification_proofs import ProofStore, VerificationMode
from klab.workdir import ensure_work_dir
from klab.workspace_preflight import require_storage, training_storage_checks


def verified_reuse_tokenizer(
    variant: CorpusVariant,
    artifacts: dict[str, Artifact],
    source: Path,
    release: ReleaseDeclaration,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    """Bind a reused tokenizer and any fraction selector to verified artifacts."""
    name = variant.tokenizer_artifact
    if name is None:
        raise ValueError("variant does not reuse a tokenizer")
    spec = artifacts[name]
    if spec.kind != "tokenizer":
        raise ValueError(f"{name} is not a tokenizer artifact")
    identity = verify_artifact(
        spec,
        source,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    fraction = release.fraction
    if variant.fraction_tokenizer is not None and fraction is None:
        raise ValueError("fraction tokenizer override requires a fractional release")
    if fraction is not None:
        selector = identity
        if variant.fraction_tokenizer is not None:
            selector_spec = artifacts[variant.fraction_tokenizer]
            if selector_spec.kind != "tokenizer":
                raise ValueError("fraction selector is not a tokenizer artifact")
            selector = verify_artifact(
                selector_spec,
                source,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
        if fraction.tokenizer_sha256.lower() != selector["sha256"]:
            raise ValueError("fraction tokenizer digest differs from verified selector")
    return identity


def prepare_variant(
    variant: CorpusVariant,
    source: Path,
    base_run_path: Path,
    workspace: Path,
    *,
    artifacts: dict[str, Artifact] | None = None,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Reuse one project's pinned acquisition and independently freeze its release."""
    validate_tokenizer_batch_limits(
        tokenizer_batch_documents, tokenizer_batch_source_bytes
    )
    if not base_run_path.is_file():
        raise ValueError(f"corpus variant {variant.id} needs a file-backed base_run")
    from klab.recovery.provenance import declaration_reference

    project_path = declaration_reference(source, variant.project)
    project = load_project(project_path)
    values = project.release.model_dump(mode="python")
    values.update(variant.release_set)
    release = ReleaseDeclaration.model_validate(values)
    reused = (
        verified_reuse_tokenizer(
            variant,
            artifacts or {},
            source,
            release,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if variant.tokenizer_artifact is not None
        else None
    )
    candidate = project.model_copy(update={"release": release})
    verify_fraction_tokenizer(candidate)
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=workspace,
            rss_bytes=current_process_rss_bytes(),
        )
    ensure_work_dir(workspace)
    acquisition_path = workspace / "corpora" / candidate.config.id / "acquisition.json"
    if acquisition_path.exists():
        acquisition = verify_acquisition(
            candidate,
            workspace,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    else:
        if any(
            item.kind not in {"local", "deterministic_generator"}
            for item in candidate.sources
            if item.redistribution != "rejected"
        ):
            raise ValueError(
                f"corpus variant {variant.id} needs pre-acquired pinned external sources"
            )
        acquisition = acquire(
            candidate,
            workspace,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    built = build(
        candidate,
        workspace,
        offline=True,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    frozen = freeze(
        built, workspace, proof_store=proof_store, verification_mode=verification_mode
    )
    verified = verify_release(
        frozen, proof_store=proof_store, verification_mode=verification_mode
    )
    exported = export_release(
        frozen,
        variant.view,
        base_run_path,
        (
            variant.vocab_size
            if reused is None
            else load_tokenizer(Path(reused["path"])).get_vocab_size()
        ),
        workspace,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    config = load_config(exported / "run.yaml")
    export_record = verify_release_export(
        config.dataset, proof_store=proof_store, verification_mode=verification_mode
    )
    if reused is None:
        tokenizer = train_tokenizer(
            load_tokenizer_config(exported / "tokenizer.yaml"),
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    else:
        tokenizer = Path(reused["path"])
        config = RunConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "tokenizer": {
                    **config.tokenizer.model_dump(mode="python"),
                    "path": tokenizer,
                },
            }
        )
        if config.model.vocab_size != load_tokenizer(tokenizer).get_vocab_size():
            raise ValueError("reused tokenizer vocabulary differs from prepared model")
    prepared = prepare_data(
        config,
        load_tokenizer(tokenizer),
        resource_envelope=resource_envelope,
        tokenizer_batch_documents=tokenizer_batch_documents,
        tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    return {
        "id": variant.id,
        "acquisition": acquisition,
        "release_id": verified["release_id"],
        "release_path": str(frozen),
        "export": export_record,
        "export_path": str(exported),
        "tokenizer_sha256": sha256_file(tokenizer),
        "tokenizer_path": str(tokenizer),
        "prepared_path": str(prepared.root),
        "prepared_manifest_sha256": sha256_file(prepared.root / "manifest.json"),
        "config_sha256": config_sha256(config.model_dump(mode="json")),
        "config": config.model_dump(mode="json"),
        **(
            {"tokenizer_artifact": variant.tokenizer_artifact}
            if reused is not None
            else {}
        ),
    }


def prepare_plan(
    plan: ExperimentPlan,
    source: Path,
    workspace: Path,
    *,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Materialize declared variants only, after storage preflight and pinned sources."""
    validate_tokenizer_batch_limits(
        tokenizer_batch_documents, tokenizer_batch_source_bytes
    )
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=workspace,
            rss_bytes=current_process_rss_bytes(),
        )
    from klab.experiments.plan import base_run_config

    base = base_run_config(plan, source)
    require_storage(training_storage_checks(base))
    ensure_work_dir()
    workspace.mkdir(parents=True, exist_ok=True)
    if not isinstance(plan.base_run, str):
        if plan.corpus_variants:
            raise ValueError("corpus variants need a file-backed base_run")
        return {"format": "experiment-preparation-v1", "id": plan.id, "variants": []}
    from klab.recovery.provenance import declaration_reference

    base_path = declaration_reference(source, plan.base_run)
    records = [
        prepare_variant(
            variant,
            source,
            base_path,
            workspace,
            artifacts=plan.artifacts,
            resource_envelope=resource_envelope,
            tokenizer_batch_documents=tokenizer_batch_documents,
            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        for variant in plan.corpus_variants
    ]
    return {"format": "experiment-preparation-v1", "id": plan.id, "variants": records}
