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
import json
import math
import secrets
import statistics
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
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
            values[field] = str(node) if isinstance(node, Path) else node
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


class _Unit:
    """One try to run: a point at a seed."""

    __slots__ = ("delta", "index", "order", "seed", "settings")

    def __init__(
        self,
        order: tuple[int, int],
        index: int,
        settings: Mapping[str, Any],
        seed: int | None,
        delta: Path,
    ) -> None:
        self.order = order
        self.index = index
        self.settings = settings
        self.seed = seed
        self.delta = delta


class _Plan:
    """Which try may start next; shared by every worker.

    Tries are handed out in declaration order. A seed's baseline is trained by
    the first try at that seed, and only reused by the tries after it, so the
    others wait until that first try has completed (a failed first try hands
    the role to the next one). Seeds do not wait for each other.
    """

    def __init__(self, units: list[_Unit], *, share_baseline: bool) -> None:
        self._condition = threading.Condition()
        self._pending = list(units)
        self._state = {
            unit.order[0]: "untrained" if share_baseline else "ready" for unit in units
        }
        self.stopped = False

    def acquire(self) -> _Unit | None:
        """The next try this worker may run, waiting for one; None when done."""
        with self._condition:
            while not self.stopped and self._pending:
                for position, unit in enumerate(self._pending):
                    state = self._state[unit.order[0]]
                    if state == "training":
                        continue
                    if state == "untrained":
                        self._state[unit.order[0]] = "training"
                    del self._pending[position]
                    return unit
                self._condition.wait()
            return None

    def release(self, unit: _Unit, *, completed: bool) -> None:
        with self._condition:
            if self._state[unit.order[0]] == "training":
                self._state[unit.order[0]] = "ready" if completed else "untrained"
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self.stopped = True
            self._condition.notify_all()


def _device_try(
    *,
    folder: Path,
    policy: Any,
    root: Path,
    seeds: list[int | None],
    device_backend: str,
    runtime_id: str | None,
) -> Callable[..., tuple[dict[str, Any], Path]]:
    """A ``try_fn`` that runs ``klab try`` in a process confined to one device.

    A process fixes the devices it sees when its device runtime first
    enumerates them, so a try on another device needs a process of its own.
    The configured ``runtime.device_index`` stays what the baseline declares
    (0) and the process sees its card as that device, which keeps the
    baseline's identity the same on every device.
    """
    from klab.device_visibility import selected_environment

    policies = folder / "policies"
    policies.mkdir()
    logs = folder / "logs"
    logs.mkdir()
    files: dict[int | None, Path] = {}
    for seed in seeds:
        update = {"lab_dir": root} | ({} if seed is None else {"seed": seed})
        path = policies / f"seed-{seed if seed is not None else 'default'}.yaml"
        body = policy.model_copy(update=update).model_dump(mode="json")
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        files[seed] = path

    def run(
        delta: Path,
        baseline: Path,
        *,
        device: int,
        work_dir: Path,
        seed: int | None,
        backend: str | None,
        request: Mapping[str, Any],
        **_authorized_by_the_child: Any,
    ) -> tuple[dict[str, Any], Path]:
        command = [
            sys.executable,
            "-m",
            "klab",
            "--work-dir",
            str(work_dir),
            "try",
            str(delta),
            "--vs",
            str(baseline),
            "--config",
            str(files[seed]),
            "--sweep-id",
            str(request["sweep_id"]),
            "--sweep-point",
            str(request["point"]),
            "--json",
        ]
        if backend is not None:
            command += ["--backend", backend]
        if runtime_id is not None:
            command += ["--runtime", runtime_id]
        log = logs / f"{delta.stem}-seed-{seed}-device-{device}.log"
        with log.open("wb") as stderr:
            process = subprocess.run(
                command,
                env=selected_environment(device_backend, device),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr,
                check=False,
            )
        try:
            record = json.loads(process.stdout)
            path = Path(record.pop("record"))
        except ValueError, KeyError, TypeError:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            last = lines[-1] if lines else ""
            if last.startswith("klab try: "):
                raise ValueError(last.removeprefix("klab try: ")) from None
            raise RuntimeError(
                f"klab try exited {process.returncode} on device {device} "
                f"without a record; see {log}"
            ) from None
        return record, path

    return run


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
    devices: Sequence[int] | None = None,
    runtime_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run every point at every seed as a try; write one sealed sweep record.

    With ``devices`` the tries run concurrently, one at a time per device, each
    in a process confined to its device; ``try_fn`` then also receives
    ``device``. An unexpected exception seals the record ``failed``, keeping
    the finished tries and the error, and is raised again.
    """
    from klab.lab_mode import lab_root, run_try
    from klab.try_config import load_try_config

    declaration = Path(declaration).absolute()
    config = load_sweep(declaration)
    policy = load_try_config(config.config)
    if policy.seed is not None and config.seeds is not None:
        raise ValueError("set seeds in the sweep or in its try config, not both")
    device_backend = (
        require_devices(devices, config.baseline, backend)
        if devices is not None
        else None
    )
    root = lab_root(work_dir, policy.lab_dir or lab_dir)
    started_at = (now or (lambda: datetime.now(UTC)))()
    sweep_id = f"sweep-{started_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
    folder = root / KINDS["sweep"][1] / sweep_id
    folder.mkdir(parents=True)
    record_path = folder / KINDS["sweep"][2]
    grid = points(config)
    seeds: list[int | None] = list(config.seeds) if config.seeds else [policy.seed]
    if try_fn is None:
        try_fn = (
            run_try
            if devices is None
            else _device_try(
                folder=folder,
                policy=policy,
                root=root,
                seeds=seeds,
                device_backend=device_backend,
                runtime_id=runtime_id,
            )
        )
    deltas = folder / "deltas"
    deltas.mkdir()
    delta_paths = []
    for index, settings in enumerate(grid):
        path = deltas / f"point-{index:03d}.yaml"
        body = {"kind": "delta", "set": settings}
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
        "devices": None if devices is None else list(devices),
        "points": [
            {"point": index, "label": label(settings), "set": settings}
            for index, settings in enumerate(grid)
        ],
        "tries": [],
        "status": "started",
    }
    if request is not None:
        record["request"] = dict(request)

    write_sealed(record_path, record)
    results: list[tuple[tuple[int, int], dict[str, Any]]] = []
    try:
        interrupted = _run_tries(
            config=config,
            policy=policy,
            grid=grid,
            seeds=seeds,
            delta_paths=delta_paths,
            results=results,
            try_fn=try_fn,
            devices=devices,
            sweep_id=sweep_id,
            work_dir=work_dir,
            root=root,
            backend=backend,
            authorization=authorization,
            resource_envelope=resource_envelope,
        )
        record["tries"] = [entry for _, entry in sorted(results, key=_by_order)]
        sealed = _seal_finished(record, grid, interrupted, folder, record_path)
    except Exception as error:
        record["tries"] = [entry for _, entry in sorted(results, key=_by_order)]
        record.update(
            status="failed",
            exit_status=1,
            failure=f"{type(error).__name__}: {error}",
        )
        try:
            write_sealed(record_path, record, may_replace=_own_started(sweep_id))
        except Exception as sealing:  # noqa: BLE001 - the original error is raised
            error.add_note(f"the sweep record could not be sealed as failed: {sealing}")
        raise
    return sealed, record_path


def _own_started(sweep_id: str) -> Callable[[Path], bool]:
    def own_started(existing: Path) -> bool:
        current = read_lab_record(existing, "sweep")[1]
        return (
            current.get("status") == "started" and current.get("sweep_id") == sweep_id
        )

    return own_started


def _by_order(item: tuple[tuple[int, int], Any]) -> tuple[int, int]:
    return item[0]


def _run_tries(
    *,
    config: SweepConfig,
    policy: Any,
    grid: list[dict[str, Any]],
    seeds: list[int | None],
    delta_paths: list[Path],
    results: list[tuple[tuple[int, int], dict[str, Any]]],
    try_fn: Callable[..., tuple[dict[str, Any], Path]],
    devices: Sequence[int] | None,
    sweep_id: str,
    work_dir: Path,
    root: Path,
    backend: str | None,
    authorization: Any,
    resource_envelope: Any,
) -> bool:
    """Run the tries, adding each finished one to ``results``; True if interrupted."""
    try:
        own = _baseline_values(
            config.baseline, sorted({field for point in grid for field in point})
        )
    except ValueError, OSError:
        own = {}
    units: list[_Unit] = []
    for seed_index, seed in enumerate(seeds):
        for index, (settings, delta) in enumerate(zip(grid, delta_paths, strict=True)):
            order = (seed_index, index)
            if _is_baseline(settings, own):
                # The baseline itself: nothing to train, by definition no change.
                results.append(
                    (
                        order,
                        {
                            "point": index,
                            "label": label(settings),
                            "set": dict(settings),
                            "seed": seed,
                            "status": "baseline",
                            "verdict": "BASELINE",
                            "heldout_delta": 0.0,
                        },
                    )
                )
            else:
                units.append(_Unit(order, index, settings, seed, delta))
    plan = _Plan(
        units,
        share_baseline=not (
            policy.fresh_baseline
            or policy.repeat_baseline
            or policy.continuation is not None
        ),
    )
    lock = threading.Lock()
    state: dict[str, Any] = {"interrupted": False, "error": None}

    def work(device: int | None) -> None:
        while (unit := plan.acquire()) is not None:
            completed = False
            try:
                extra = {} if device is None else {"device": device}
                result, path = try_fn(
                    unit.delta,
                    config.baseline,
                    work_dir=work_dir,
                    lab_dir=root,
                    seed=unit.seed,
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
                        "point": unit.index,
                        "baseline": str(config.baseline),
                        "backend_override": backend,
                    },
                    **extra,
                )
                entry = _entry(unit.index, unit.settings, unit.seed, result, path)
                completed = result.get("status") == "completed"
                if result.get("exit_status") == EXIT_INTERRUPTED:
                    state["interrupted"] = True
                    plan.stop()
            except (ValueError, OSError) as error:
                entry = {
                    "point": unit.index,
                    "label": label(unit.settings),
                    "set": dict(unit.settings),
                    "seed": unit.seed,
                    "status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                }
            except Exception as error:  # noqa: BLE001 - re-raised by run_sweep
                with lock:
                    state["error"] = state["error"] or error
                plan.stop()
                plan.release(unit, completed=False)
                return
            if device is not None:
                entry["device"] = device
            with lock:
                results.append((unit.order, entry))
            plan.release(unit, completed=completed)

    if devices is None:
        work(None)
    else:
        threads = [
            threading.Thread(target=work, args=(device,), name=f"sweep-device-{device}")
            for device in devices
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():
                try:
                    thread.join()
                except KeyboardInterrupt:
                    # The terminal signalled every try's process too; let the
                    # running ones record how far they got.
                    state["interrupted"] = True
                    plan.stop()
    if state["error"] is not None:
        raise state["error"]
    return state["interrupted"]


def _seal_finished(
    record: dict[str, Any],
    grid: list[dict[str, Any]],
    interrupted: bool,
    folder: Path,
    record_path: Path,
) -> dict[str, Any]:
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
    return write_sealed(
        record_path, record, may_replace=_own_started(record["sweep_id"])
    )


def require_devices(devices: Sequence[int], baseline: Path, backend: str | None) -> str:
    """The backend the devices belong to, refusing a baseline they cannot serve."""
    from klab.config.loading import load_config
    from klab.device_visibility import SELECTION_VARIABLE

    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError("--devices needs distinct device indices, such as 0,1,2")
    runtime = load_config(baseline).runtime
    name = backend or runtime.backend
    if name not in SELECTION_VARIABLE:
        raise ValueError(
            f"--devices selects accelerator cards on xpu, cuda and rocm; the "
            f"baseline's backend is {name!r} (set runtime.backend or pass --backend)"
        )
    if runtime.device_index != 0 or runtime.secondary_device_indices:
        raise ValueError(
            "--devices gives each try one card that it sees as device 0; the "
            "baseline must set runtime.device_index 0 and no "
            "secondary_device_indices"
        )
    return name


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
    if record.get("devices"):
        lines.append("  devices: " + ", ".join(str(d) for d in record["devices"]))
    if record.get("failure"):
        lines.append(f"  failure: {record['failure']}")
        lines.append(
            "  The sweep stopped before it finished: no summary, only the tries "
            "that completed are listed below."
        )
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
        device = f"  device {entry['device']}" if "device" in entry else ""
        lines.append(
            f"  {entry.get('try_id') or '-'}  {entry['label']}  seed {seed}  "
            f"{status}{device}"
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
