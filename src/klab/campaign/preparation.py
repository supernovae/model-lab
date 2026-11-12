"""Operation-local preparation adapters; bindings belong to the compiled plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.campaign.preparation_plan import (
    PreparationArtifact,
    PreparationRequest,
    PreparationResult,
)
from klab.experiments.artifacts import verify_artifact
from klab.experiments.plan import Artifact
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


def _artifact(
    request: PreparationRequest,
    path: Path,
    output: PreparationArtifact | None = None,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> PreparationArtifact:
    if output is None:
        if request.operation.kind == "tokenizer_train":
            output = PreparationArtifact(
                kind="tokenizer", identifier=path.parent.name, sha256=sha256_file(path)
            )
        else:
            manifest = json.loads((path / "manifest.json").read_text())
            output = PreparationArtifact(
                kind="prepared_data",
                identifier=manifest["settings_sha256"],
                sha256=manifest["manifest_sha256"],
            )
    if output.kind != request.operation.output.kind:
        raise ValueError("preparation output differs from compiled kind")
    verified = verify_artifact(
        Artifact.model_validate(
            {
                **output.model_dump(mode="json"),
                "version": 1,
                "producer": f"klab {request.operation.kind}",
                "path": str(path),
            }
        ),
        request.source,
        dataset=request.config.dataset,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    return PreparationArtifact.model_validate(
        {key: verified[key] for key in ("kind", "identifier", "sha256")}
    )


def verify(
    request: PreparationRequest,
    result: PreparationResult,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    config, path = request.config, result.path
    if (
        request.operation.kind == "tokenizer_train"
        and path != config.output_dir / "tokenizer.json"
    ):
        raise ValueError("tokenizer output differs from declaration")
    if (
        request.operation.kind == "data_prepare"
        and path.parent != config.dataset.cache_dir
    ):
        raise ValueError("prepared output differs from declaration")
    output = _artifact(
        request,
        path,
        result.artifact,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    if result.artifact != output:
        raise ValueError("preparation output identity changed")


def dispatch(
    request: PreparationRequest,
    *,
    observer: Any | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> PreparationResult:
    from klab.data.tokenizer import load_tokenizer, train_tokenizer

    config = request.config
    if request.operation.kind == "tokenizer_train":
        path = train_tokenizer(config)
    else:
        from klab.data.packing import prepare_data

        path = prepare_data(
            config,
            load_tokenizer(config.tokenizer.path),
            observer=observer,
            proof_store=proof_store,
            verification_mode=verification_mode,
        ).root
    identity = _artifact(
        request, path, proof_store=proof_store, verification_mode=verification_mode
    )
    return PreparationResult(artifact=identity, path=path)
