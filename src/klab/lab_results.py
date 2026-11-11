"""Resolve everyday lab inputs without guessing a record or checkpoint.

Sealed record IDs are the common handle: readers retain the whole record;
checkpoint consumers select its recorded candidate (try) or target (probe).
Runs do not identify a unique scientific record, so readers require a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from klab.lab_records import KINDS, read_lab_record


class ResultLookupError(ValueError):
    """One actionable lookup failure, shared by text and JSON callers."""

    def __init__(self, spec: str, status: str, reason: str, hint: str) -> None:
        self.spec = spec
        self.status = status
        self.reason = reason
        self.hint = hint
        super().__init__(f"{status}: {spec}: {reason}; {hint}")

    def as_json(self) -> dict[str, str]:
        return {
            "status": self.status,
            "result": self.spec,
            "reason": self.reason,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class LabResult:
    path: Path
    run: Path | None = None
    kind: str | None = None
    record: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckpointTarget:
    run: Path
    checkpoint: str | None
    expected_sha256: str | None = None


def _run_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / "manifest.json").is_file() and (
            candidate / "resolved_config.yaml"
        ).is_file():
            return candidate
    return None


def resolve_result(
    spec: str, *, lab_dir: Path, runs_dir: Path | None = None
) -> LabResult:
    """Locate a record, run, or checkpoint; colliding IDs require a path."""
    requested = Path(spec)
    candidates = [requested]
    if requested.parent == Path() and not requested.is_absolute():
        candidates += [lab_dir / folder / spec for _, folder, _ in KINDS.values()]
        candidates.append(lab_dir / "runs" / spec)
        if runs_dir is not None:
            candidates.append(runs_dir / spec)
    found = sorted({p.resolve() for p in candidates if p.exists()})
    if not found:
        raise ResultLookupError(
            spec,
            "missing_evidence",
            "no saved record, run or checkpoint found",
            "check the ID and --work-dir/--lab-dir, or pass an explicit path",
        )
    if len(found) > 1:
        raise ResultLookupError(
            spec,
            "ambiguous_result",
            "multiple locations match",
            "pass one explicit path: " + ", ".join(map(str, found)),
        )
    path = found[0]
    record_files = (
        [path / name for _, _, name in KINDS.values() if (path / name).is_file()]
        if path.is_dir()
        else []
    )
    if record_files or path.name in {name for _, _, name in KINDS.values()}:
        if len(record_files) > 1:
            raise ResultLookupError(
                spec,
                "ambiguous_result",
                "multiple records in this folder",
                "pass try.json or probe.json explicitly",
            )
        path = record_files[0] if record_files else path
        return _record_result(spec, path)
    # An exported record can live inside a run. Only native pointer/manifest
    # JSON names belong to checkpoint resolution; don't treat a renamed record
    # as a checkpoint simply because one of its ancestors is a run directory.
    if (
        path.is_file()
        and path.suffix == ".json"
        and path.name not in {"latest.json", "best.json", "manifest.json"}
    ):
        return _record_result(spec, path)
    run = _run_root(path if path.is_dir() else path.parent)
    if run is not None:
        return LabResult(path=path, run=run)
    # Exported sealed records may have a user-chosen filename.
    if path.is_file():
        return _record_result(spec, path)
    raise ResultLookupError(
        spec,
        "missing_evidence",
        "no lab record or Model Lab run at this path",
        "pass a sealed try/probe record or a run/checkpoint path",
    )


def _record_result(spec: str, path: Path) -> LabResult:
    try:
        kind, record = read_lab_record(path)
    except (ValueError, OSError) as error:
        raise ResultLookupError(
            spec,
            "invalid_evidence",
            str(error),
            "restore the original sealed record; do not edit its digest",
        ) from None
    return LabResult(path=path, kind=kind, record=record)


def resolve_record(spec: str, lab_dir: Path) -> Path:
    result = resolve_result(spec, lab_dir=lab_dir)
    if result.record is None:
        raise ResultLookupError(
            spec,
            "missing_evidence",
            "a run/checkpoint does not select a unique sealed result",
            "pass a try/probe record ID or path to report/compare; "
            "use probe to produce evidence, or explore for checkpoint diagnostics",
        )
    return result.path


def resolve_checkpoint(
    spec: str, *, lab_dir: Path, runs_dir: Path | None = None
) -> CheckpointTarget:
    result = resolve_result(spec, lab_dir=lab_dir, runs_dir=runs_dir)
    return checkpoint_from_result(spec, result)


def checkpoint_from_result(spec: str, result: LabResult) -> CheckpointTarget:
    """Keep checkpoint selection and provenance on the same native lookup."""
    if result.run is not None:
        return CheckpointTarget(
            result.run, None if result.path == result.run else str(result.path)
        )
    assert result.record is not None
    row = (
        (result.record.get("arms") or {}).get("candidate")
        if result.kind == "try"
        else result.record.get("target")
    ) or {}
    checkpoint = row.get("checkpoint")
    digest = row.get("checkpoint_sha256")
    if not row.get("run_dir") or not checkpoint or not digest or row.get("reference"):
        raise ResultLookupError(
            spec,
            "missing_evidence",
            "record has no local scored candidate checkpoint",
            "use report to inspect the retained stop/missing evidence; "
            "pass a local run/checkpoint explicitly to start new work",
        )
    run = Path(row["run_dir"])
    if not run.is_absolute():
        raise ResultLookupError(
            spec,
            "missing_evidence",
            "recorded run location is relative; original working directory is unknown",
            "report can read the historical record; pass an explicit checkpoint path "
            "for new work rather than inferring a directory from the reader",
        )
    run = run.resolve()
    relative = Path(checkpoint)
    # A record must name a generation, never a mutable latest/best pointer.
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "checkpoints"
        or relative.parts[1] in {"..", "latest.json", "best.json"}
    ):
        raise ResultLookupError(
            spec,
            "invalid_evidence",
            "record does not name a checkpoint generation",
            "pass a checkpoint explicitly; never substitute latest for recorded evidence",
        )
    selected = run / relative
    if _run_root(run) != run or not selected.is_dir():
        raise ResultLookupError(
            spec,
            "missing_evidence",
            f"recorded checkpoint is unavailable: {selected}",
            "restore the recorded run/checkpoint; report/compare can still read the record",
        )
    if selected.resolve().parent != run / "checkpoints":
        raise ResultLookupError(
            spec,
            "invalid_evidence",
            "recorded checkpoint escapes its run",
            "restore the recorded checkpoint generation inside its run",
        )
    return CheckpointTarget(run, str(selected), digest)


def baseline_provenance(spec: str | None) -> dict[str, Any]:
    """Capture selector context before setup; never interpret an ID as a path."""
    return {
        "selector": spec,
        "working_directory": str(Path.cwd()),
        "status": "not_requested" if spec is None else "unresolved",
    }


def baseline_text(record: Any) -> str:
    """Read-only presentation; historical relative selectors stay uncertain."""
    provenance = record.get("baseline_provenance")
    request = record.get("request") or {}
    if (provenance or {}).get("status") == "not_requested" or (
        provenance is None and "baseline" in request and request["baseline"] is None
    ):
        return "  baseline: none requested"
    if provenance is None:
        selector = (record.get("request") or {}).get("baseline")
        return (
            f"  baseline selector: {selector!r}; provenance unavailable "
            "(historical record; original working directory unknown)"
        )
    line = f"  baseline selector: {provenance['selector']!r}; {provenance['status']}"
    if provenance.get("resolved"):
        selected = provenance["resolved"]
        line += f"; {selected['kind']}: {selected['path']}"
        if selected.get("run_id"):
            line += f"; run {selected['run_id']}"
        if selected.get("checkpoint"):
            line += f"; checkpoint {selected['checkpoint']}"
    return line
