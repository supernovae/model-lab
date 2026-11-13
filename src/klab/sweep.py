"""Sweeps: one baseline, several values of a setting, each run as an ordinary try.

A sweep declaration names a baseline config, then either one or more dotted
fields with lists of values (``vary``: every combination is a point) or an
explicit list of points (``points``: each sets several fields together, such as
a tokenizer and its vocabulary size), and optionally seeds. Every point runs as
a ``klab try`` against the baseline at every seed. The baseline trains once per
seed and is reused for every point, because ``try`` reuses a verified baseline
with the same inputs. The sweep writes one sealed record listing each try and
summarizing each point across its seeds.
"""

from __future__ import annotations

import csv
import itertools
import math
import secrets
import statistics
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from klab.canonical import canonical_json
from klab.config.models import RunSeed, StrictModel
from klab.lab_records import KINDS, read_lab_record, write_sealed
from klab.training.manifest import sha256_file

SWEEP_VERSION = 1
MAX_POINTS = 256
EXIT_INTERRUPTED = 130


class SweepConfig(StrictModel):
    sweep_version: Literal[1] = 1
    baseline: Path
    question: str | None = None
    vary: dict[str, list[Any]] | None = None
    points: list[dict[str, Any]] | None = None
    seeds: list[RunSeed] | None = None
    config: str = Field(default="fast", min_length=1)

    @field_validator("vary")
    @classmethod
    def valid_vary(
        cls, vary: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]] | None:
        from klab.derivation import _settings

        if vary is None:
            return None
        if not vary:
            raise ValueError("vary needs at least one field")
        _settings({field: values for field, values in vary.items()})
        for field, values in vary.items():
            if not values:
                raise ValueError(f"vary.{field} needs at least one value")
            encoded = [canonical_json(value) for value in values]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"vary.{field} repeats a value")
        return vary

    @field_validator("points")
    @classmethod
    def valid_points(
        cls, points: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        from klab.derivation import _settings

        if points is None:
            return None
        if not points:
            raise ValueError("points needs at least one point")
        for point in points:
            if not point:
                raise ValueError("every point sets at least one field")
            _settings(point)
        encoded = [canonical_json(point) for point in points]
        if len(set(encoded)) != len(encoded):
            raise ValueError("points repeats a point")
        return points

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, seeds: list[int] | None) -> list[int] | None:
        if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
            raise ValueError("seeds must be a nonempty list of distinct seeds")
        return seeds

    @model_validator(mode="after")
    def bounded(self) -> SweepConfig:
        if (self.vary is None) == (self.points is None):
            raise ValueError("a sweep sets exactly one of vary and points")
        if len(points(self)) * len(self.seeds or [None]) > MAX_POINTS:
            raise ValueError(f"a sweep runs at most {MAX_POINTS} tries")
        return self


def load_sweep(path: Path) -> SweepConfig:
    """Read a declaration; relative paths resolve from its own directory."""
    path = Path(path).absolute()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read sweep {path}: {error.strerror or error}"
        ) from None
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in sweep {path}: {error}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"sweep {path} must contain a YAML mapping")  # noqa: TRY004 - serialized schema
    try:
        config = SweepConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid sweep {path}: {error}") from None
    base = path.parent
    from klab.try_config import PRESETS

    return config.model_copy(
        update={
            "baseline": (base / config.baseline.expanduser()).resolve(),
            "config": config.config
            if config.config in PRESETS
            else str((base / Path(config.config).expanduser()).resolve()),
        }
    )


def points(config: SweepConfig) -> list[dict[str, Any]]:
    """The declared points, or every combination of the varied values."""
    if config.points is not None:
        return [dict(point) for point in config.points]
    assert config.vary is not None
    fields = list(config.vary)
    return [
        dict(zip(fields, values, strict=True))
        for values in itertools.product(*(config.vary[field] for field in fields))
    ]


def label(settings: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{field}={canonical_json(value).decode()}" for field, value in settings.items()
    )


def _baseline_values(baseline: Path, fields: list[str]) -> dict[str, Any]:
    """The baseline's own value for each varied field (missing fields absent)."""
    from klab.config.loading import load_config
    from klab.experiments.compiler import patchable_config

    view = patchable_config(load_config(baseline))
    values: dict[str, Any] = {}
    for field in fields:
        node: Any = view
        for part in field.split("."):
            if not isinstance(node, dict) or part not in node:
                break
            node = node[part]
        else:
            values[field] = node
    return values


def _is_baseline(settings: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return all(
        field in baseline and canonical_json(baseline[field]) == canonical_json(value)
        for field, value in settings.items()
    )


def _entry(
    index: int,
    settings: Mapping[str, Any],
    seed: int | None,
    record: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    arms = record.get("arms") or {}
    baseline = arms.get("baseline") or {}
    candidate = arms.get("candidate") or {}
    comparison = record.get("comparison") or {}
    probe = record.get("probe") or {}
    verdict = probe.get("verdict") if isinstance(probe, Mapping) else None
    return {
        "point": index,
        "label": label(settings),
        "set": dict(settings),
        "seed": seed,
        "try_id": record.get("try_id"),
        "record": str(path),
        "status": record.get("status"),
        "exit_status": record.get("exit_status"),
        "verdict": comparison.get("verdict"),
        "metric": comparison.get("metric"),
        "heldout_delta": comparison.get("heldout_delta"),
        "baseline_loss": (baseline.get("heldout") or {}).get("loss"),
        "candidate_loss": (candidate.get("heldout") or {}).get("loss"),
        "baseline_bits_per_byte": _bits(baseline),
        "candidate_bits_per_byte": _bits(candidate),
        "baseline_kernel_norm_fraction": _kernel(baseline),
        "candidate_kernel_norm_fraction": _kernel(candidate),
        "baseline_reused": bool(baseline.get("reused")),
        "parameters": candidate.get("parameters"),
        "baseline_parameters": baseline.get("parameters"),
        "tokens_seen": candidate.get("tokens_seen"),
        "wall_seconds": (record.get("resources") or {}).get("wall_seconds"),
        "probe_action": (verdict or {}).get("action")
        if isinstance(verdict, Mapping)
        else None,
        "failure": record.get("failure"),
    }


def _bits(row: Mapping[str, Any]) -> float | None:
    text = (row.get("heldout") or {}).get("text")
    return text.get("bits_per_byte") if isinstance(text, Mapping) else None


def _kernel(row: Mapping[str, Any]) -> float | None:
    measured = (row.get("heldout") or {}).get("gradient_bottleneck")
    return (
        measured.get("kernel_norm_fraction") if isinstance(measured, Mapping) else None
    )


def summarize_point(
    index: int, settings: Mapping[str, Any], entries: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """One point across seeds: paired differences and their spread.

    A point's seeds compare on one metric: held-out loss, or bits per byte
    when the point changes the tokenizer. Seeds that compared on another
    metric than the point's first are left out and counted.
    """
    metrics = [
        entry.get("metric")
        for entry in entries
        if isinstance(entry.get("heldout_delta"), (int, float))
        and entry.get("status") != "baseline"
    ]
    metric = metrics[0] if metrics else "loss"
    deltas = [
        float(entry["heldout_delta"])
        for entry in entries
        if isinstance(entry.get("heldout_delta"), (int, float))
        and (entry.get("status") == "baseline" or entry.get("metric") == metric)
    ]
    n = len(deltas)
    mean = statistics.fmean(deltas) if deltas else None
    sd = statistics.stdev(deltas) if n >= 2 else None
    se = sd / math.sqrt(n) if sd is not None else None
    if entries and all(entry.get("status") == "baseline" for entry in entries):
        verdict = "BASELINE"
    elif mean is None:
        verdict = "NO_EVIDENCE"
    elif n == 1:
        verdict = (
            "SINGLE_SEED_LOWER"
            if mean < 0
            else "SINGLE_SEED_HIGHER"
            if mean > 0
            else "TIE"
        )
    elif se is not None and mean + 2 * se < 0:
        verdict = "LOWER_LOSS"
    elif se is not None and mean - 2 * se > 0:
        verdict = "HIGHER_LOSS"
    else:
        verdict = "WITHIN_SEED_NOISE"
    walls = [e["wall_seconds"] for e in entries if e.get("wall_seconds") is not None]
    kernels = [
        e["candidate_kernel_norm_fraction"]
        for e in entries
        if isinstance(e.get("candidate_kernel_norm_fraction"), (int, float))
    ]
    candidate = next((e for e in entries if e.get("parameters") is not None), {})
    return {
        "point": index,
        "label": label(settings),
        "set": dict(settings),
        "tries": len(entries),
        "comparable": n,
        "metric": metric,
        "mixed_metric": sum(m != metric for m in metrics),
        "heldout_delta_mean": mean,
        "heldout_delta_sd": sd,
        "heldout_delta_se": se,
        "verdict": verdict,
        "parameters": candidate.get("parameters"),
        "tokens_seen": candidate.get("tokens_seen"),
        "wall_seconds_mean": statistics.fmean(walls) if walls else None,
        # Measured only when the baseline sets evaluation.gradient_bottleneck.
        "kernel_norm_fraction_mean": statistics.fmean(kernels) if kernels else None,
        "not_comparable": sum(e.get("verdict") == "NOT_COMPARABLE" for e in entries),
        "failed": sum(e.get("status") == "failed" for e in entries),
    }


def _write_series(path: Path, summary: list[Mapping[str, Any]]) -> None:
    fields = [
        "point",
        "label",
        "comparable",
        "metric",
        "heldout_delta_mean",
        "heldout_delta_se",
        "verdict",
        "parameters",
        "tokens_seen",
        "wall_seconds_mean",
        "kernel_norm_fraction_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)


def run_sweep(
    declaration: Path,
    *,
    work_dir: Path,
    lab_dir: Path | None = None,
    backend: str | None = None,
    authorization: Any = None,
    resource_envelope: Any = None,
    request: Mapping[str, Any] | None = None,
    try_fn: Callable[..., tuple[dict[str, Any], Path]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run every point at every seed as a try; write one sealed sweep record."""
    from klab.lab_mode import lab_root, run_try
    from klab.try_config import load_try_config

    declaration = Path(declaration).absolute()
    config = load_sweep(declaration)
    policy = load_try_config(config.config)
    if policy.seed is not None and config.seeds is not None:
        raise ValueError("set seeds in the sweep or in its try config, not both")
    try_fn = try_fn or run_try
    root = lab_root(work_dir, policy.lab_dir or lab_dir)
    started_at = (now or (lambda: datetime.now(UTC)))()
    sweep_id = f"sweep-{started_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
    folder = root / KINDS["sweep"][1] / sweep_id
    folder.mkdir(parents=True)
    record_path = folder / KINDS["sweep"][2]
    grid = points(config)
    seeds: list[int | None] = list(config.seeds) if config.seeds else [policy.seed]
    deltas = folder / "deltas"
    deltas.mkdir()
    delta_paths = []
    for index, settings in enumerate(grid):
        path = deltas / f"point-{index:03d}.yaml"
        body = {"lab_delta_version": 1, "set": settings}
        if config.question:
            body["question"] = config.question
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        delta_paths.append(path)
    record: dict[str, Any] = {
        "format": KINDS["sweep"][0],
        "sweep_id": sweep_id,
        "created_at": started_at.isoformat(),
        "declaration": {
            "path": str(declaration),
            "sha256": sha256_file(declaration),
            **config.model_dump(mode="json"),
        },
        "baseline": str(config.baseline),
        "seeds": seeds,
        "points": [
            {"point": index, "label": label(settings), "set": settings}
            for index, settings in enumerate(grid)
        ],
        "tries": [],
        "status": "started",
    }
    if request is not None:
        record["request"] = dict(request)

    def own_started(existing: Path) -> bool:
        current = read_lab_record(existing, "sweep")[1]
        return (
            current.get("status") == "started" and current.get("sweep_id") == sweep_id
        )

    write_sealed(record_path, record)
    interrupted = False
    try:
        own = _baseline_values(
            config.baseline, sorted({field for point in grid for field in point})
        )
    except ValueError, OSError:
        own = {}
    for seed in seeds:
        for index, (settings, delta) in enumerate(zip(grid, delta_paths, strict=True)):
            if _is_baseline(settings, own):
                # The baseline itself: nothing to train, by definition no change.
                record["tries"].append(
                    {
                        "point": index,
                        "label": label(settings),
                        "set": dict(settings),
                        "seed": seed,
                        "status": "baseline",
                        "verdict": "BASELINE",
                        "heldout_delta": 0.0,
                    }
                )
                continue
            try:
                result, path = try_fn(
                    delta,
                    config.baseline,
                    work_dir=work_dir,
                    lab_dir=root,
                    seed=seed,
                    backend=backend,
                    authorization=authorization,
                    resource_envelope=resource_envelope,
                    fresh_baseline=policy.fresh_baseline,
                    repeat_baseline=policy.repeat_baseline,
                    continuation=policy.continuation,
                    probe_tier=None
                    if policy.probe_tier == "none"
                    else policy.probe_tier,
                    request={
                        **policy.model_dump(mode="json"),
                        "sweep_id": sweep_id,
                        "point": index,
                        "baseline": str(config.baseline),
                        "backend_override": backend,
                    },
                )
            except (ValueError, OSError) as error:
                record["tries"].append(
                    {
                        "point": index,
                        "label": label(settings),
                        "set": dict(settings),
                        "seed": seed,
                        "status": "failed",
                        "failure": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            record["tries"].append(_entry(index, settings, seed, result, path))
            if result.get("exit_status") == EXIT_INTERRUPTED:
                interrupted = True
                break
        if interrupted:
            break
    by_point: dict[int, list[dict[str, Any]]] = {}
    for entry in record["tries"]:
        by_point.setdefault(entry["point"], []).append(entry)
    record["summary"] = [
        summarize_point(index, settings, by_point.get(index, []))
        for index, settings in enumerate(grid)
    ]
    baseline_parameters = next(
        (
            entry["baseline_parameters"]
            for entry in record["tries"]
            if entry.get("baseline_parameters") is not None
        ),
        None,
    )
    for point in record["summary"]:
        if point["verdict"] == "BASELINE":
            point["parameters"] = baseline_parameters
    failed = sum(entry.get("status") == "failed" for entry in record["tries"])
    record["status"] = (
        "interrupted"
        if interrupted
        else "failed"
        if failed == len(record["tries"])
        else "completed"
    )
    record["exit_status"] = EXIT_INTERRUPTED if interrupted else 1 if failed else 0
    _write_series(folder / "series.csv", record["summary"])
    sealed = write_sealed(record_path, record, may_replace=own_started)
    return sealed, record_path


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def summarize_sweep(record: Mapping[str, Any], path: Path | None = None) -> str:
    """The series as a table: one row per point, paired across seeds."""
    seeds = record.get("seeds") or [None]
    shown_seeds = ", ".join("baseline" if s is None else str(s) for s in seeds)
    lines = [
        f"SWEEP {record['sweep_id']}  status: {record['status']}",
        f"  baseline: {record['baseline']}",
        f"  seeds: {shown_seeds}",
    ]
    question = (record.get("declaration") or {}).get("question")
    if question:
        lines.append(f"  question: {question}")
    lines.append("")
    header = ("point", "n", "metric", "Δ", "± se", "verdict", "params", "wall s")
    rows = [header]
    for point in record.get("summary") or []:
        rows.append(
            (
                point["label"],
                str(point["comparable"]),
                "bits/byte" if point.get("metric") == "bits_per_byte" else "loss",
                _number(point["heldout_delta_mean"]),
                _number(point["heldout_delta_se"]),
                point["verdict"],
                _number(point["parameters"]),
                _number(point["wall_seconds_mean"], 1),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        lines.append(
            (
                "  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            ).rstrip()
        )
    lines.append("")
    lines.append(
        "  Δ is candidate minus baseline at the same seed, in held-out loss, or "
        "in bits per byte where the point changes the tokenizer; negative is "
        "better. A verdict needs two or more seeds: LOWER_LOSS or HIGHER_LOSS "
        "when the mean is more than two standard errors from zero."
    )
    lines.append("")
    for entry in record.get("tries") or []:
        seed = "baseline" if entry.get("seed") is None else entry["seed"]
        status = entry.get("verdict") or entry.get("status")
        lines.append(
            f"  {entry.get('try_id') or '-'}  {entry['label']}  seed {seed}  {status}"
        )
    if path is not None:
        lines.append(f"  record: {path}")
        lines.append(f"  series: {path.parent / 'series.csv'}")
    return "\n".join(lines)


def prepare_sweep(args: Any) -> None:
    """Expose the sweep's baseline and policy the way ``try`` declares them."""
    from klab.try_config import prepare_try_config

    if getattr(args, "sweep_loaded", None) is not None:
        return
    config = load_sweep(Path(args.sweep))
    args.sweep_loaded = config
    args.vs = str(config.baseline)
    args.config = config.config
    prepare_try_config(args)
    if args.seed is not None and config.seeds is not None:
        raise ValueError("set seeds in the sweep or in its try config, not both")
