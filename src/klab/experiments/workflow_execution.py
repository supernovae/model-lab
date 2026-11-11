"""Current admission of inventoried prepared-input CPU Campaigns.

The authored native Campaign owns connections; this context authenticates them.
A reviewed commit is a caller assertion, not author attestation or compute credit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, StrictInt, StrictStr, model_validator

from klab.campaign.plan import CampaignPlan
from klab.config.models import StrictModel
from klab.experiments.compiler import expand_axes
from klab.experiments.plan import recipe_run_config
from klab.experiments.preparation_execution import _environment
from klab.experiments.recipe_admission import RecipeAdmission
from klab.experiments.workflow_recipe import (
    _location,
    load_workflow,
    locked_workflow_context,
)
from klab.recovery.provenance import repository_root


class WorkflowLimits(StrictModel):
    """Declared ceilings, not allocation or an operating-system resource quota."""

    max_training_updates: StrictInt = Field(gt=0)
    max_target_positions: StrictInt = Field(gt=0)
    max_eval_batches: StrictInt = Field(gt=0)


class WorkflowAdmission(RecipeAdmission):
    recipe_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_python: Path
    limits: WorkflowLimits

    @model_validator(mode="before")
    @classmethod
    def current_contract(cls, value):
        if isinstance(value, dict) and (
            "workflow_admission_version" in value or "pair_limits" in value
        ):
            raise ValueError(
                "obsolete workflow admission; regenerate admission with explicit limits "
                "for the current authored workflow and reviewed commit"
            )
        return value


class WorkflowExecution:
    def __init__(
        self, source: Path, bindings: Path, work_dir: Path, admission: WorkflowAdmission
    ):
        self.source = Path(source).absolute()
        self.bindings = Path(bindings).absolute()
        if not Path(work_dir).is_absolute():
            raise ValueError("workflow requires an explicit absolute work root")
        self.work_dir = _location(self.bindings, str(work_dir))
        self.admission = admission.model_copy(deep=True)
        self.loaded = load_workflow(self.source, self.bindings)
        self.environment = _environment()
        self.identity = {
            "format": "klab-workflow-execution-v1",
            "source": str(self.source),
            "bindings": str(self.bindings),
            "work_dir": str(self.work_dir),
            "admission": self.admission.model_dump(mode="json"),
            "environment": self.environment,
        }
        loaded = self.loaded
        if loaded.descriptor.campaign is None:
            raise ValueError("workflow execution requires an inventoried Campaign")
        self.campaign_source = loaded.context.reference(
            loaded.source, loaded.descriptor.campaign
        )
        document = loaded.context.read_document(self.campaign_source)
        # Artifact identity is portable; its location comes exclusively from bindings.
        for stage in document.get("stages", []):
            if stage.get("kind") == "artifact_reference":
                name = stage.get("id")
                if name not in loaded.plan.artifacts or stage.get(
                    "artifact"
                ) != loaded.descriptor.artifacts[name].model_dump(mode="json"):
                    raise ValueError("Campaign artifact differs from workflow identity")
                stage["artifact"] = loaded.plan.artifacts[name].model_dump(mode="json")
        self.campaign = CampaignPlan.model_validate(document)
        self._validate_workflow()
        self.refresh()

    def _validate_workflow(self):
        loaded = self.loaded
        plan = loaded.plan
        paired = bool(plan.axes or plan.comparisons)
        if paired:
            if (
                len(plan.axes) != 1
                or len(plan.axes[0].choices) != 2
                or len(plan.comparisons) != 1
                or plan.comparisons[0].mode != "controlled"
                or len(plan.comparisons[0].interventions) != 1
                or plan.comparisons[0].phases
            ):
                raise ValueError(
                    "paired workflow requires two choices and one controlled comparison"
                )
            comparison = plan.comparisons[0]
            intervention = comparison.interventions[0]
            if intervention.split(".")[0] not in {"model", "attention", "optimizer"}:
                raise ValueError(
                    "paired workflow intervention must preserve data, seed, runtime and evaluation"
                )
        self.config = recipe_run_config(plan, loaded.source, context=loaded.context)
        expanded = expand_axes(
            self.config, plan.axes, max_runs=2, base_dir=loaded.source.parent
        )
        coordinates = [coordinate for _, coordinate in expanded]
        if paired and (
            comparison.baseline not in coordinates
            or comparison.variant not in coordinates
        ):
            raise ValueError(
                "paired comparison must select both concrete configurations"
            )
        # Data and graph determine supported capabilities; admission never selects
        # a historical execution behavior.
        expected_source = "snapshot" if paired else "synthetic"
        limits = self.admission.limits
        updates = positions = 0
        for config, _ in expanded:
            if (
                self.config.dataset.source != expected_source
                or config.dataset != self.config.dataset
                or config.tokenizer != self.config.tokenizer
                or config.dataset.source != expected_source
                or config.runtime.backend != "cpu"
                or config.runtime.engine != "pytorch"
                or plan.execution.backend not in (None, "cpu")
            ):
                raise ValueError(
                    f"workflow supports {expected_source} CPU PyTorch only"
                )
            count = (
                config.training.max_steps
                + config.staging.smoke_steps
                + config.staging.warmup_steps
            )
            updates += count
            positions += (
                count
                * config.training.seq_len
                * config.training.micro_batch_size
                * config.training.gradient_accumulation
            )
            if config.evaluation.batch_limit > limits.max_eval_batches:
                raise ValueError("workflow exceeds declared evaluation limit")
        if (
            updates > limits.max_training_updates
            or positions > limits.max_target_positions
        ):
            raise ValueError(
                "workflow exceeds declared training limits (including pilots)"
            )
        groups = {}
        for stage in self.campaign.stages:
            groups.setdefault(stage.kind, []).append(stage)
        finals = [s for s in groups.get("experiment_collect", []) if s.evaluations]
        if len(finals) != 1:
            raise ValueError(
                "current workflow requires one final collection; regenerate recipe"
            )
        groups["experiment_collect"] = [
            s for s in groups["experiment_collect"] if not s.evaluations
        ]
        cell_count = len(expanded)
        counts = {kind: len(stages) for kind, stages in groups.items()}
        if (
            counts
            != {
                "artifact_reference": 2,
                "experiment_plan": 1,
                "runtime_acceptance": 1,
                "approval": 1,
                "experiment_run": cell_count,
                "experiment_collect": cell_count,
                "evaluation": cell_count,
            }
            or self.campaign.recovery is not None
        ):
            raise ValueError(
                "workflow requires one native plan and one run/collect/evaluation branch per cell"
            )
        planning = groups["experiment_plan"][0]
        runtime = groups["runtime_acceptance"][0]
        gate = groups["approval"][0]
        runs = groups["experiment_run"]
        collects = groups["experiment_collect"]
        evaluations = groups["evaluation"]
        run_ids = {run.id for run in runs}
        expected_cells = (
            {
                f"main:{plan.axes[0].name}={choice.label}"
                for choice in plan.axes[0].choices
            }
            if paired
            else {"main:single"}
        )
        base_dependencies = {planning.id, runtime.id, gate.id}
        first = [run for run in runs if set(run.requires) == base_dependencies]
        if (
            planning.source != loaded.descriptor.entrypoint
            or planning.mode != "lock"
            or planning.tokenizer != plan.inputs["tokenizer"]
            or planning.prepared != plan.inputs["prepared_data"]
            or set(planning.requires) != set(plan.inputs.values())
            or runtime.plan != planning.id
            or set(runtime.requires) != {planning.id}
            or runtime.profile_id is not None
            or runtime.worker is not None
            or set(gate.bind) != {planning.id, runtime.id}
            or set(gate.requires) != {planning.id, runtime.id}
            or {run.cell for run in runs} != expected_cells
            or any(run.plan != planning.id or run.runtime != runtime.id for run in runs)
            or len(first) != 1
            or any(
                set(run.requires) != base_dependencies | {first[0].id}
                for run in runs
                if run not in first
            )
            or {collect.run for collect in collects} != run_ids
            or any(
                collect.plan != planning.id
                or set(collect.requires) != {planning.id, *run_ids}
                for collect in collects
            )
            or {evaluation.collect for evaluation in evaluations}
            != {collect.id for collect in collects}
            or any(
                evaluation.backend not in (None, "cpu")
                or evaluation.runtime is not None
                or evaluation.suite != plan.evaluation_suite
                or set(evaluation.requires) != {evaluation.collect}
                for evaluation in evaluations
            )
        ):
            raise ValueError("unsupported portable Campaign connections")
        if finals:
            final = finals[0]
            evaluation_ids = {s.id for s in evaluations}
            if (
                final.plan != planning.id
                or final.run not in run_ids
                or set(final.evaluations) != evaluation_ids
                or set(final.requires) != {planning.id, *run_ids, *evaluation_ids}
                or any(final.id in s.requires for s in self.campaign.stages)
            ):
                raise ValueError(
                    "final collection must follow all evaluations and be terminal"
                )

    def refresh(self):
        current = load_workflow(self.source, self.bindings)
        if (
            current != self.loaded
            or current.recipe_sha256 != self.admission.recipe_sha256
            or current.binding_sha256 != self.admission.binding_sha256
        ):
            raise ValueError("WORKFLOW_CONTEXT_CHANGED: recipe or bindings differ")
        if (
            self.admission.cpu_python.absolute() != Path(sys.executable).absolute()
            or _environment() != self.environment
        ):
            raise ValueError(
                "WORKFLOW_CONTEXT_CHANGED: selected CPU environment differs"
            )
        _location(self.bindings, str(self.work_dir))
        if repository_root(self.work_dir) is not None:
            raise ValueError("workflow work root must be outside Git")
        provenance = self.admission.provenance(
            self.source, current.context, operation="workflow"
        )
        return {
            **provenance,
            "allow_uncommitted_declaration": False,
            "workflow_context": self.identity,
            "closure": sorted(
                (
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in provenance["declarations"]
                ),
                key=lambda item: item["path"],
            ),
        }

    def authorize_lock(self, lock):
        self.refresh()
        loaded = locked_workflow_context(lock, self.loaded.source)
        if loaded != self.loaded:
            raise ValueError("WORKFLOW_CONTEXT_CHANGED: lock context differs")

    def engine(self, **kwargs):
        from klab.campaign.engine import CampaignEngine

        kwargs.setdefault("cold_verify", True)
        return CampaignEngine(
            self.campaign_source, self.work_dir, workflow_context=self, **kwargs
        )
