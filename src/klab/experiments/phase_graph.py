"""Shared native declaration connectivity checks; no payload or runtime access."""

from klab.experiments.plan import ExperimentPlan, Phase


def validate_phase_graph(plan: ExperimentPlan) -> tuple[Phase, ...]:
    phases = plan.phases or (Phase(id="main"),)
    positions = {phase.id: index for index, phase in enumerate(phases)}
    for index, phase in enumerate(phases):
        if phase.parent is not None:
            if phase.parent not in positions or positions[phase.parent] >= index:
                raise ValueError(
                    f"phase {phase.id} parent must be an earlier declared phase (acyclic DAG)"
                )
            if phase.selector == "terminal" and phase.at_step is None:
                raise ValueError(f"phase {phase.id} terminal parent needs an update")
            if phase.selector == "best_validation" and phase.at_step is not None:
                raise ValueError(
                    f"phase {phase.id} best-validation selection cannot also pin a terminal update"
                )
        if phase.checkpoint is not None:
            artifact = plan.artifacts.get(phase.checkpoint)
            if (
                artifact is None
                or artifact.kind != "checkpoint"
                or artifact.from_phase is not None
            ):
                raise ValueError(
                    f"phase {phase.id} needs a pinned external checkpoint generation"
                )
            if (
                phase.transition in {"resume", "extend_budget"}
                and artifact.state != "full"
            ):
                raise ValueError(
                    f"phase {phase.id} {phase.transition} requires full checkpoint state"
                )
            if phase.transition == "promote" and artifact.state not in {
                "full",
                "weights",
            }:
                raise ValueError(
                    f"phase {phase.id} promotion requires pinned checkpoint weights"
                )
            if phase.selector is not None or phase.at_step is not None:
                raise ValueError(
                    f"phase {phase.id} external checkpoint is already a pinned generation"
                )
        if phase.transition == "fresh" and (
            phase.selector is not None or phase.at_step is not None
        ):
            raise ValueError(
                f"fresh phase {phase.id} cannot select a parent generation"
            )
    return phases
