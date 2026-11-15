"""Compose explicit research and lesson folders without executing experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

from klab.canonical import canonical_json
from klab.config.models import RunConfig, TokenizerTrainConfig
from klab.experiments.compiler import apply_patch, patchable_config
from klab.research.catalog import (
    MechanismLesson,
    load_dataset_source,
    load_datasets,
    load_lesson,
    load_profiles,
    resource_path,
    resource_sha256,
)
from klab.safe_io import rename_noreplace
from klab.training.manifest import config_sha256, sha256_file
from klab.workdir import resolve_work_dir


def _dataset_names(*, root: Path | None = None) -> tuple[str, ...]:
    return tuple(sorted(load_datasets(root=root).datasets))


_BACKENDS = frozenset({"cpu", "mps", "cuda", "rocm", "xpu"})
_PATH_KEYS = frozenset(
    {
        "path",
        "cache_dir",
        "root_dir",
        "output_dir",
        "train_path",
        "validation_path",
        "source_manifest_path",
    }
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = data.encode("utf-8") if isinstance(data, str) else data
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _yaml_bytes(value: object) -> bytes:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")


def _dataset_mapping(name: str, *, root: Path | None = None) -> dict[str, object]:
    try:
        profile = load_datasets(root=root).datasets[name]
    except KeyError as error:
        choices = ", ".join(_dataset_names(root=root))
        raise ValueError(
            f"unknown dataset profile {name!r}; choose {choices}"
        ) from error
    result = {
        "source": profile.source,
        "revision": profile.revision,
        "dataset_config": profile.dataset_config,
        "cache_dir": "artifacts/data",
        "train_max_documents": profile.train_max_documents,
        "validation_max_documents": profile.validation_max_documents,
        "train_max_tokens": profile.train_max_tokens,
        "validation_max_tokens": profile.validation_max_tokens,
        "synthetic_seed": profile.dataset_seed,
        "train_path": None,
        "validation_path": None,
        "license": None,
    }
    if profile.synthetic_task is not None:
        result["synthetic_task"] = profile.synthetic_task
    if profile.dataset_id is not None:
        result.update(dataset_id=profile.dataset_id, license=profile.license)
    if profile.source_declaration is not None:
        source = load_dataset_source(profile, root=root)
        result.update(
            revision=source["revision"],
            license=source["license"],
            train_path="artifacts/snapshot/train.jsonl",
            validation_path="artifacts/snapshot/validation.jsonl",
            source_manifest_path="artifacts/snapshot/manifest.json",
        )
    return result


def _tokenizer_values(name: str, *, root: Path | None = None) -> dict[str, object]:
    profile = load_datasets(root=root).datasets[name]
    dataset = {
        **_dataset_mapping(name, root=root),
        "train_max_documents": profile.tokenizer_max_documents,
        "validation_max_documents": profile.tokenizer_validation_max_documents,
        "train_max_tokens": profile.tokenizer_train_max_tokens,
        "validation_max_tokens": profile.tokenizer_validation_max_tokens,
    }
    return {
        "schema_version": 1,
        "vocab_size": profile.vocab_size,
        "min_frequency": profile.min_frequency,
        "max_documents": profile.tokenizer_max_documents,
        "output_dir": "artifacts/tokenizer",
        "dataset": dataset,
    }


def _write_dataset_source(
    directory: Path, data: str, *, root: Path | None = None
) -> list[str]:
    profile = load_datasets(root=root).datasets[data]
    if profile.source_declaration is None:
        return []
    source = load_dataset_source(profile, root=root)
    _write(directory / "source.yaml", _yaml_bytes(source))
    relative = "research_sources/" + profile.source_declaration
    _write(
        directory / relative,
        resource_path(profile.source_declaration, root=root).read_bytes(),
    )
    return ["source.yaml", relative]


def _acquisition_commands(data: str, command: str, *, root: Path | None = None) -> str:
    if load_datasets(root=root).datasets[data].source_declaration is None:
        return ""
    return (
        "# Acquire once before tokenizer fitting; reuse the verified snapshot thereafter.\n"
        "mkdir -p artifacts\n"
        f"{command} klab data lock source.yaml --output artifacts/source.lock.json --cache-dir artifacts/hf-cache\n"
        f"{command} klab data snapshot artifacts/source.lock.json --output artifacts/snapshot --cache-dir artifacts/hf-cache\n"
    )


def _scale_patch(
    scale: str, data: str, backend: str, *, root: Path | None = None
) -> dict[str, object]:
    try:
        profile = load_profiles(root=root).scales[scale]
    except KeyError as error:
        choices = ", ".join(load_profiles(root=root).scales)
        raise ValueError(
            f"unsupported runnable scale {scale!r}; choose {choices}. Larger classes are recommendation-only."
        ) from error
    try:
        dataset = load_datasets(root=root).datasets[data]
    except KeyError as error:
        choices = ", ".join(_dataset_names(root=root))
        raise ValueError(f"unsupported dataset {data!r}; choose {choices}") from error
    return {
        "model.vocab_size": dataset.vocab_size,
        "model.hidden_dim": profile.hidden_dim,
        "model.num_layers": profile.num_layers,
        "model.num_heads": profile.num_heads,
        "model.ffn_dim": profile.ffn_dim,
        "model.max_seq_len": profile.max_seq_len,
        "tokenizer.path": "artifacts/tokenizer/tokenizer.json",
        "dataset": _dataset_mapping(data, root=root),
        "training.seq_len": profile.seq_len,
        "training.max_steps": profile.max_steps,
        "training.max_tokens": profile.max_tokens,
        "training.micro_batch_size": profile.micro_batch_size,
        "training.gradient_accumulation": profile.gradient_accumulation,
        "optimizer.peak": 0.003 if scale == "smoke" else 0.0003,
        "optimizer.floor": 0.0003 if scale == "smoke" else 0.00003,
        "optimizer.warmup_steps": 4 if scale == "smoke" else 10,
        "evaluation.every_steps": profile.validation_steps,
        "checkpoint.every_steps": profile.validation_steps,
        "runtime.backend": backend,
        "logging.root_dir": "klab-work/runs",
    }


def _apply_overlay(
    config: dict[str, object], patch: dict[str, object]
) -> dict[str, object]:
    nested = {key: value for key, value in patch.items() if "." in key}
    direct = {key: value for key, value in patch.items() if "." not in key}
    if direct:
        for key, value in direct.items():
            if key not in config:
                raise ValueError(f"unknown top-level config patch: {key}")
            config[key] = value
    return apply_patch(config, nested) if nested else config


def _load_base(*, root: Path | None = None) -> dict[str, object]:
    value = yaml.safe_load(
        resource_path("base.yaml", root=root).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise TypeError("packaged base config must contain a mapping")
    patchable = patchable_config(value)
    if not isinstance(patchable, dict):
        raise TypeError("packaged base config must project to a mapping")
    return patchable


def _validate_run(value: dict[str, object]) -> RunConfig:
    try:
        return RunConfig.model_validate(value)
    except ValidationError as error:
        raise ValueError(f"generated run configuration is invalid: {error}") from error


def _destination(output: Path) -> tuple[Path, Path]:
    destination = output.expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"output destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    return destination, temporary


def _publish(temporary: Path, destination: Path) -> Path:
    rename_noreplace(temporary, destination)
    return destination


def _cleanup(temporary: Path) -> None:
    if temporary.exists():
        shutil.rmtree(temporary)


def _command_environment(backend: str) -> tuple[str, str]:
    if backend in {"cpu", "mps"}:
        return "uv run --locked --extra cpu", ""
    return (
        "uv run --locked --no-sync",
        (
            "Before running commands, set `UV_PROJECT_ENVIRONMENT` to the absolute path "
            "of your **existing vendor environment prefix** (not its `bin/python`). "
            "For example, `export UV_PROJECT_ENVIRONMENT=/absolute/path/to/vendor-prefix`. "
            "Do not run `uv sync` into that environment: it would replace vendor wheels."
        ),
    )


def _lesson_readme(
    lesson: MechanismLesson,
    scale: str,
    data: str,
    backend: str,
    *,
    workspace: str | None = None,
    root: Path | None = None,
) -> str:
    command, environment_note = _command_environment(backend)
    return f"""# {lesson.title}

{lesson.summary}

**Selection:** `{scale}` / `{data}` / `{backend}`. This standalone path can be used without a worker.

`model.yaml`, `tokenizer.yaml`, and `lesson.json` are the versioned source. `$WORK/staging/` holds disposable pilots and `$WORK/runs/` holds mutable runs and checkpoints.

{environment_note}

## Predict, inspect, probe, change one knob

Before running the probe, predict the output shape and diagnostic values from the shape walkthrough in `lesson.json`. Then inspect the concrete recipe and compare the observed forward. Change exactly one config knob and explain any mismatch.

```sh
WORK="{workspace}"
mkdir -p "$WORK"
export KLAB_WORK_DIR="$WORK"
{_acquisition_commands(data, command, root=root)}\
{command} klab tokenizer train tokenizer.yaml
{command} klab data prepare model.yaml
{command} klab inspect model.yaml --json
{command} klab workspace preflight model.yaml
{command} klab stage model.yaml --through smoke --output "$WORK/staging/lesson-{lesson.id}"
{command} klab learn probe model.yaml --prompt \"A short input asks about an object.\" --json
{command} klab train model.yaml --run-id lesson-{lesson.id} --runs-dir "$WORK/runs" --stop-after-step 2
{command} klab checkpoint verify "$WORK/runs/lesson-{lesson.id}/checkpoints/latest.json" --json
{command} klab eval lesson-{lesson.id} --runs-dir "$WORK/runs"
```

Acquisition, tokenizer fitting and dataset preparation are explicit. Snapshot profiles declare acquisition and exclusion policies in `source.yaml`; strict whole-document budgets may differ from retired direct-loading selections. The probe is a freshly initialized CPU FP32 reference forward, not a hardware benchmark, learned result, or score prediction. Training/evaluation checkpoints and dashboard diagnostics provide learned observations later.

Existing numeric budgets are retained. If strict whole-document preparation rejects a token cap, review the declared document selection and token limits in a new config; the scaffold does not silently resize them.

## Walkthrough

{chr(10).join(f"- **{step.source_path}:{step.source_symbol}** — {step.explanation} Shapes: {'; '.join(step.shapes)} Observe: {', '.join(step.observe)}." for step in lesson.steps)}

**Try changes:** {"; ".join(lesson.try_changes)}

**Limits:** {"; ".join(lesson.limits)}

## Source and follow-up

{chr(10).join(f"- {doc}" for doc in lesson.docs)}
"""


def scaffold_lesson(
    identifier: str,
    output: Path,
    *,
    scale: str = "smoke",
    data: str = "offline",
    backend: str = "cpu",
    root: Path | None = None,
) -> Path:
    """Write a single-mechanism lesson configuration."""
    lesson = load_lesson(identifier, root=root)
    if data not in _dataset_names(root=root):
        raise ValueError(
            f"unsupported dataset {data!r}; choose {', '.join(_dataset_names(root=root))}"
        )
    if backend not in _BACKENDS:
        raise ValueError(
            f"unsupported backend {backend!r}; choose {', '.join(sorted(_BACKENDS))}"
        )
    if scale not in load_profiles(root=root).scales:
        raise ValueError(
            f"unsupported runnable scale {scale!r}; larger named classes are recommendation-only"
        )
    if scale not in lesson.patches_by_scale:
        raise ValueError(f"lesson {identifier} has no patch for scale {scale}")

    destination, temporary = _destination(output)
    workspace = resolve_work_dir() / "experiments" / f"lesson-{identifier}"
    try:
        lesson_payload: dict[str, object] = {
            "format": "klab-lesson-scaffold",
            "version": 1,
            "lesson": lesson.model_dump(mode="json"),
            "lesson_sha256": resource_sha256(f"lessons/{identifier}.json", root=root),
            "selection": {"scale": scale, "data": data, "backend": backend},
        }
        base = _load_base(root=root)
        base = _apply_overlay(base, _scale_patch(scale, data, backend, root=root))
        patch = lesson.patches_by_scale.get(scale, {})
        base = _apply_overlay(base, patch)
        base = _apply_overlay(
            base,
            {
                "logging.root_dir": Path(
                    os.path.relpath(workspace / "runs", temporary)
                ).as_posix()
            },
        )
        base["name"] = f"lesson-{identifier}-{scale}-{data}"
        config = _validate_run(base)
        tokenizer_values = _tokenizer_values(data, root=root)
        TokenizerTrainConfig.model_validate(tokenizer_values)
        _write(temporary / "model.yaml", _yaml_bytes(base))
        _write(temporary / "tokenizer.yaml", _yaml_bytes(tokenizer_values))
        dataset_inputs = _write_dataset_source(temporary, data, root=root)
        if dataset_inputs:
            lesson_payload.setdefault("inputs", []).extend(
                {"path": relative, "sha256": sha256_file(temporary / relative)}
                for relative in dataset_inputs
            )
        lesson_payload["config_sha256"] = config_sha256(config.model_dump(mode="json"))
        lesson_payload["tokenizer_sha256"] = hashlib.sha256(
            _yaml_bytes(tokenizer_values)
        ).hexdigest()
        _write(
            temporary / "README.md",
            _lesson_readme(
                lesson,
                scale,
                data,
                backend,
                workspace=Path(
                    os.path.relpath(
                        workspace,
                        destination.resolve(),
                    )
                ).as_posix(),
                root=root,
            ),
        )
        source_inputs = {"lesson.json": f"lessons/{identifier}.json"}
        source_inputs.update(
            {name: name for name in ("base.yaml", "profiles.json", "datasets.json")}
        )
        for name, source in source_inputs.items():
            relative = f"research_sources/{name}"
            _write(temporary / relative, resource_path(source, root=root).read_bytes())
            lesson_payload.setdefault("inputs", []).append(
                {"path": relative, "sha256": sha256_file(temporary / relative)}
            )
        lesson_payload["lesson_scaffold_sha256"] = hashlib.sha256(
            canonical_json(lesson_payload)
        ).hexdigest()
        _write(temporary / "lesson.json", _json_bytes(lesson_payload))
        return _publish(temporary, destination)
    except BaseException:
        _cleanup(temporary)
        raise
