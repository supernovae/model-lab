"""``klab probe``: run the battery on a run or checkpoint, write a record."""

from __future__ import annotations

import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.lab_context import LabContext, LabSignal
from klab.lab_records import PROBE_FORMAT, read_lab_record, write_sealed
from klab.lab_results import (
    CheckpointTarget,
    ResultLookupError,
    baseline_provenance,
    checkpoint_from_result,
    resolve_checkpoint,
    resolve_result,
)
from klab.probes.runner import Arm, lm_eval_spec, progress_writer, run_battery
from klab.reference_models import load_reference, require_reference_extras


def load_target(
    spec: str,
    *,
    lab_dir: Path,
    runs_dir: Path | None,
    backend: str | None,
    authorization: Any,
    selected: CheckpointTarget | None = None,
) -> Any:
    from klab.evaluation.inference import load_run

    target = selected or resolve_checkpoint(spec, lab_dir=lab_dir, runs_dir=runs_dir)
    return load_run(
        target.run.name,
        target.run.parent,
        target.checkpoint,
        backend,
        authorization=authorization,
        expected_checkpoint_sha256=target.expected_sha256,
    )


def run_probe(
    target_spec: str,
    baseline_spec: str | None,
    *,
    lab_dir: Path,
    runs_dir: Path | None = None,
    tier: str = "fast",
    fast_fail: bool = True,
    backend: str | None = None,
    authorization: Any = None,
    resource_envelope: Any = None,
    lm_eval_tasks: list[str] | None = None,
    lm_eval_limit: int | None = None,
    lm_eval_protocol: Any = None,
    final: bool = False,
    request: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run the battery and publish ``LAB/probes/<id>/probe.json`` (sealed).

    The same :class:`LabContext` as ``klab try`` applies: touch
    ``LAB/probes/<id>/CANCEL`` (or Ctrl-C) to stop at a safe point; a resource
    envelope is checked before every probe. A stopped battery is still
    published, marked incomplete. FINAL scores the held-back final split: the
    final verdict on a chosen candidate, never a selection signal.
    """
    from klab.reference_models import is_reference, reference_for
    from klab.resource_envelope import require_envelope_applicability

    require_envelope_applicability(resource_envelope, operation="probe")
    lm_spec = lm_eval_spec(
        lm_eval_tasks, lm_eval_limit, lm_eval_protocol
    )  # bad options fail first
    if baseline_spec is not None and is_reference(baseline_spec):
        raise ValueError(
            "a reference model is not a probe baseline (different tokenizer and "
            f"data); probe it on its own and use `klab compare {target_spec} "
            f"{baseline_spec}`"
        )
    if is_reference(target_spec):
        reference_for(target_spec)  # unknown names fail before any work
        if tier != "full":
            raise ValueError(
                "reference models are scored on lm-eval tasks only: use --config full"
            )
        require_reference_extras()
    if final and tier == "fast":
        raise ValueError(
            "final evaluation scores the held-back text probes: use --config final or final-full"
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    probe_id = f"probe-{stamp}-{secrets.token_hex(4)}"
    folder = lab_dir / "probes" / probe_id
    folder.mkdir(parents=True, exist_ok=False)
    progress = progress_writer(folder / "progress.json")
    context = LabContext(
        folder / "CANCEL", resource_envelope=resource_envelope, workspace=lab_dir
    )

    request = dict(request) if request is not None else {"baseline": baseline_spec}
    provenance = baseline_provenance(baseline_spec)
    setup_record = {
        "format": PROBE_FORMAT,
        "probe_id": probe_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "started",
        "tier": tier,
        "final": final,
        "request": request,
        "baseline_provenance": provenance,
        "target": None,
        "baseline": None,
    }
    path = folder / "probe.json"

    def own_started(existing: Path) -> bool:
        current = read_probe(existing)
        return (
            current.get("status") == "started" and current.get("probe_id") == probe_id
        )

    def arm(spec: str, *, baseline: bool = False) -> Arm:
        if is_reference(spec):
            return Arm(load=lambda: load_reference(spec), reference=True)
        # Resolve now (fail fast on a typo); load lazily, one arm at a time.
        try:
            found = resolve_result(spec, lab_dir=lab_dir, runs_dir=runs_dir)
            selected = checkpoint_from_result(spec, found)
        except ResultLookupError as error:
            if baseline:
                provenance.update(status=error.status, error=error.as_json())
            raise
        if baseline:
            provenance.update(
                status="resolved",
                resolved={
                    "kind": found.kind
                    or ("run" if found.path == found.run else "checkpoint"),
                    "path": str(found.path),
                    "run_dir": str(selected.run),
                    "run_id": selected.run.name,
                    "checkpoint": selected.checkpoint,
                    "expected_checkpoint_sha256": selected.expected_sha256,
                    **(
                        {"record_sha256": found.record["record_sha256"]}
                        if found.record
                        else {}
                    ),
                },
            )
            write_sealed(path, setup_record, may_replace=own_started)
        return Arm(
            load=lambda: load_target(
                spec,
                lab_dir=lab_dir,
                runs_dir=runs_dir,
                backend=backend,
                authorization=authorization,
                selected=selected,
            )
        )

    context.install()
    try:
        try:
            write_sealed(path, setup_record)
            progress({"state": "loading", "tier": tier, "target": target_spec})
            print(
                f"probe {probe_id}: touch {context.cancel_path} (or Ctrl-C) to stop",
                file=sys.stderr,
            )
            result = run_battery(
                arm(target_spec),
                arm(baseline_spec, baseline=True)
                if baseline_spec is not None
                else None,
                tier=tier,
                fast_fail=fast_fail,
                progress=progress,
                context=context,
                lm_eval=lm_spec,
                final=final,
            )
        except (LabSignal, KeyboardInterrupt, Exception) as error:
            # Only failures outside the battery's safe points land here
            # (run_battery finalizes signals and stops itself).
            context.finalizing = True
            name = error.name if isinstance(error, LabSignal) else type(error).__name__
            progress({"state": "failed", "error": f"{name}: {error}"})
            setup_record.update(
                status="interrupted"
                if isinstance(error, (LabSignal, KeyboardInterrupt))
                else "failed",
                failure=f"{name}: {error}",
            )
            write_sealed(path, setup_record, may_replace=own_started)
            if isinstance(error, (LabSignal, KeyboardInterrupt)):
                raise KeyboardInterrupt from None
            raise
        # Signals are only noted from here on, so the record is always written.
        context.finalizing = True
        sealed = write_sealed(
            path,
            {
                **result,
                "probe_id": probe_id,
                "request": request,
                "baseline_provenance": provenance,
            },
            may_replace=own_started,
        )
    finally:
        context.restore()
    if context.signals or result["stop"]["kind"] == "interrupted":
        # The partial battery is published (incomplete); now honor the signal.
        print(f"probe {probe_id}: interrupted; partial record: {path}", file=sys.stderr)
        raise KeyboardInterrupt
    return sealed, path


def read_probe(path: Path) -> dict[str, Any]:
    """Verified probe record (the shared lab-record reader)."""
    return read_lab_record(path, "probe")[1]
