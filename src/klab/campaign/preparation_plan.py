"""Typed preparation projection of CampaignPlan; no second graph or execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from klab.campaign.plan import CampaignPlan, DataPrepare, TokenizerTrain
from klab.config.models import RunConfig, StrictModel, TokenizerTrainConfig


class PreparationInput(StrictModel):
    port: Literal["tokenizer", "snapshot"]
    producer: str = Field(min_length=1)
    output_port: Literal["artifact"] = "artifact"
    kind: Literal["tokenizer", "dataset_snapshot"]

    @model_validator(mode="after")
    def valid_kind(self) -> PreparationInput:
        expected = "tokenizer" if self.port == "tokenizer" else "dataset_snapshot"
        if self.kind != expected:
            raise ValueError(f"{self.port} requires {expected} input")
        return self


class PreparationOutput(StrictModel):
    port: Literal["artifact"] = "artifact"
    kind: Literal["tokenizer", "prepared_data"]


class PreparationOperation(StrictModel):
    operation_version: Literal[1] = 1
    stage_id: str
    kind: Literal["tokenizer_train", "data_prepare"]
    config_reference: str
    dependencies: tuple[str, ...]
    inputs: tuple[PreparationInput, ...]
    output: PreparationOutput

    @model_validator(mode="after")
    def valid_ports(self) -> PreparationOperation:
        ports = [item.port for item in self.inputs]
        if len(set(ports)) != len(ports):
            raise ValueError("duplicate preparation input port")
        if any(item.producer not in self.dependencies for item in self.inputs):
            raise ValueError("preparation producer must be a declared dependency")
        if ("tokenizer" in ports) != (self.kind == "data_prepare"):
            raise ValueError("only data_prepare requires a tokenizer input")
        expected = "tokenizer" if self.kind == "tokenizer_train" else "prepared_data"
        if self.output.kind != expected:
            raise ValueError(f"{self.kind} requires {expected} output")
        return self


class CompiledPreparation(StrictModel):
    # Ordering comes exclusively from CampaignPlan. Non-preparation stages keep
    # their native executor route and appear here only in the complete ordering.
    ordered_stage_ids: tuple[str, ...]
    operations: tuple[PreparationOperation, ...]

    def operation(self, stage_id: str) -> PreparationOperation:
        for operation in self.operations:
            if operation.stage_id == stage_id:
                return operation
        raise ValueError(f"no compiled preparation operation: {stage_id}")


def compile_preparation(plan: CampaignPlan) -> CompiledPreparation:
    """Project validated native edges without opening configs, artifacts or stores."""
    ordered = plan.ordered_stages()
    operations = []
    for stage in ordered:
        if not isinstance(stage, (TokenizerTrain, DataPrepare)):
            continue
        inputs = []
        if stage.snapshot is not None:
            inputs.append(
                PreparationInput(
                    port="snapshot", producer=stage.snapshot, kind="dataset_snapshot"
                )
            )
        if isinstance(stage, DataPrepare):
            inputs.append(
                PreparationInput(
                    port="tokenizer", producer=stage.tokenizer, kind="tokenizer"
                )
            )
        operations.append(
            PreparationOperation(
                stage_id=stage.id,
                kind=stage.kind,
                config_reference=stage.config,
                dependencies=stage.requires,
                inputs=tuple(inputs),
                output=PreparationOutput(
                    kind="tokenizer"
                    if isinstance(stage, TokenizerTrain)
                    else "prepared_data"
                ),
            )
        )
    return CompiledPreparation(
        ordered_stage_ids=tuple(stage.id for stage in ordered),
        operations=tuple(operations),
    )


class PreparationArtifact(StrictModel):
    kind: Literal["tokenizer", "prepared_data", "dataset_snapshot"]
    identifier: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BoundPreparationInput(StrictModel):
    reference: PreparationInput
    path: Path
    artifact: PreparationArtifact

    @model_validator(mode="after")
    def valid_artifact(self) -> BoundPreparationInput:
        if self.artifact.kind != self.reference.kind:
            raise ValueError("bound preparation artifact has wrong kind")
        return self


class PreparationRequest(StrictModel):
    operation: PreparationOperation
    source: Path
    config: RunConfig | TokenizerTrainConfig
    inputs: tuple[BoundPreparationInput, ...]

    @model_validator(mode="after")
    def valid_bindings(self) -> PreparationRequest:
        if tuple(item.reference for item in self.inputs) != self.operation.inputs:
            raise ValueError("bound inputs differ from compiled preparation ports")
        expected = (
            TokenizerTrainConfig
            if self.operation.kind == "tokenizer_train"
            else RunConfig
        )
        if not isinstance(self.config, expected):
            raise TypeError("preparation config type differs from compiled operation")
        for item in self.inputs:
            if item.reference.port == "snapshot":
                if (
                    self.config.dataset.source != "snapshot"
                    or self.config.dataset.source_manifest_path != item.path
                ):
                    raise ValueError(
                        "configured dataset differs from upstream snapshot"
                    )
            elif self.config.tokenizer.path != item.path:
                raise ValueError("configured tokenizer differs from upstream tokenizer")
        return self


class PreparationResult(StrictModel):
    artifact: PreparationArtifact
    path: Path
