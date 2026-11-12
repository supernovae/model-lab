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

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig, TokenizerTrainConfig
from klab.engram.packs import _rename_noreplace
from klab.experiments.compiler import apply_patch, patchable_config
from klab.model.portable_engram import load_portable_engram
from klab.research.catalog import (
    MechanismLesson,
    load_dataset_source,
    load_datasets,
    load_lesson,
    load_profiles,
    resource_path,
    resource_sha256,
)
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
        "memory_package_path",
        "train_path",
        "validation_path",
        "allocation_manifest_path",
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
        "allocation_manifest_path": None,
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
    dataset.pop("allocation_manifest_path", None)
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
    _rename_noreplace(temporary, destination)
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


def _semantic_lesson_workspace(directory: Path) -> dict[str, object]:
    """Materialize a verified semantic pack and its executable lesson probe."""
    import numpy as np
    from safetensors.numpy import save_file

    from klab.engram.packs import compile_pack

    key_indices = {
        ("atlas", "maps_to"): 0,
        ("beacon", "maps_to"): 1,
        ("cipher", "maps_to"): 2,
        ("harbor", "label"): 3,
        ("vault", "label"): 4,
        ("fork", "maps_to"): 5,
    }
    value_indices = {
        "beacon": 0,
        "amber": 1,
        "delta": 2,
        "open": 3,
        "closed": 4,
        "left": 5,
        "right": 6,
    }
    facts = (
        ("lesson-atlas-maps-to-beacon", "atlas", "maps_to", "beacon", None, None),
        ("lesson-beacon-maps-to-amber", "beacon", "maps_to", "amber", None, None),
        ("lesson-cipher-maps-to-delta", "cipher", "maps_to", "delta", None, None),
        (
            "lesson-harbor-label-open",
            "harbor",
            "label",
            "open",
            "2025-01-01",
            "2025-12-31",
        ),
        ("lesson-vault-label-closed", "vault", "label", "closed", None, "2020-12-31"),
        ("lesson-fork-left", "fork", "maps_to", "left", None, None),
        ("lesson-fork-right", "fork", "maps_to", "right", None, None),
    )
    record_rows = [
        {
            "record_version": 1,
            "id": record_id,
            "namespace": "semantic-retrieval-lesson",
            "subject": subject,
            "relation": relation,
            "value": value,
            "license": "CC0-1.0",
            "source": "Model Lab original tutorial fixture",
            "source_revision": "frozen-v1",
            "created_at": "2026-09-23T00:00:00Z",
            "valid_from": valid_from,
            "valid_until": valid_until,
        }
        for record_id, subject, relation, value, valid_from, valid_until in facts
    ]
    records_bytes = b"".join(canonical_json(row) + b"\n" for row in record_rows)
    records_path = directory / "records.jsonl"
    _write(records_path, records_bytes)

    key_specification = {
        "algorithm": "fixed-one-hot-structured-key-v1",
        "dimension": len(key_indices),
        "indices": [
            {"subject": subject, "relation": relation, "index": index}
            for (subject, relation), index in sorted(key_indices.items())
        ],
    }
    value_specification = {
        "algorithm": "fixed-one-hot-value-v1",
        "dimension": len(value_indices),
        "indices": [
            {"value": value, "index": index}
            for value, index in sorted(value_indices.items())
        ],
    }
    key_encoder = {
        "name": "tutorial-fixed-structured-key",
        "revision": "frozen-v1",
        "sha256": canonical_digest(key_specification),
    }
    value_encoder = {
        "name": "tutorial-fixed-value",
        "revision": "frozen-v1",
        "sha256": canonical_digest(value_specification),
    }
    keys = np.zeros((len(facts), len(key_indices)), dtype=np.float32)
    values = np.zeros((len(facts), len(value_indices)), dtype=np.float32)
    for row_index, (_, subject, relation, value, _, _) in enumerate(facts):
        keys[row_index, key_indices[(subject, relation)]] = 1.0
        values[row_index, value_indices[value]] = 1.0
    keys_path = directory / "semantic_keys.safetensors"
    values_path = directory / "semantic_values.safetensors"
    save_file({"keys": keys}, keys_path)
    save_file({"values": values}, values_path)
    metadata_path = directory / "semantic.json"
    _write(
        metadata_path,
        canonical_json(
            {
                "format": "klab-semantic-assets",
                "format_version": 1,
                "record_ids": [row["id"] for row in record_rows],
                "key_encoder": key_encoder,
                "value_encoder": value_encoder,
                "key_normalization": "none",
            }
        )
        + b"\n",
    )
    pack_path = directory / "semantic-pack"
    manifest = compile_pack(
        records_path,
        pack_path,
        name="semantic-retrieval-lesson",
        namespace="semantic-retrieval-lesson",
        default_license="CC0-1.0",
        source_name="Model Lab original tutorial fixture",
        source_revision="frozen-v1",
        created_at="2026-09-23T00:00:00Z",
        semantic_keys=keys_path,
        semantic_values=values_path,
        semantic_metadata=metadata_path,
    )
    semantic = manifest.semantic
    if semantic is None:
        raise RuntimeError("compiled semantic lesson pack lacks its semantic component")
    demo = """from __future__ import annotations

import json
from pathlib import Path

import torch

from klab.config.models import AttentionConfig, ModelConfig
from klab.engram.semantic import SemanticQueryBatch, SemanticRetriever
from klab.model.transformer import DenseLM

ROOT = Path(__file__).resolve().parent
PACK_ID = "__PACK_ID__"
KEY_DIM = 6
KEY_INDEX = __KEY_INDEX__
VALUE_INDEX = __VALUE_INDEX__
VALUE_BY_INDEX = {index: value for value, index in VALUE_INDEX.items()}


def encode_key(subject: str, relation: str) -> torch.Tensor:
    vector = torch.zeros(KEY_DIM, dtype=torch.float32)
    index = KEY_INDEX.get((subject, relation))
    if index is not None:
        vector[index] = 1.0
    return vector


retriever = SemanticRetriever.from_pack(
    ROOT / "semantic-pack", expected_pack_id=PACK_ID
)
query_examples = {
    "hit": ("atlas", "maps_to", None),
    "unknown": ("missing", "maps_to", None),
    "conflict": ("fork", "maps_to", None),
    "temporal_hit": ("harbor", "label", "2025-06-01"),
    "temporal_miss": ("vault", "label", "2025-06-01"),
}
retrieval = {}
for name, (subject, relation, as_of) in query_examples.items():
    outcome = retriever.retrieve(
        encode_key(subject, relation),
        key_encoder=retriever.key_encoder,
        top_k=2,
        min_score=0.9,
        as_of=as_of,
    )
    retrieval[name] = {
        "status": outcome.status,
        "hits": [
            {
                "record_id": hit.record_id,
                "score": hit.score,
                "value_shape": list(hit.value_vector.shape),
            }
            for hit in outcome.hits
        ],
        "trace": {
            "pack_id": outcome.trace.pack_id,
            "metric": outcome.trace.metric,
            "comparison_count": outcome.trace.comparison_count,
            "candidate_count": outcome.trace.candidate_count,
            "temporal_excluded_count": outcome.trace.temporal_excluded_count,
            "best_record_id": outcome.trace.best_record_id,
            "best_score": outcome.trace.best_score,
            "tie_count": outcome.trace.tie_count,
            "tied_record_ids": outcome.trace.tied_record_ids,
        },
    }

first_hop = retriever.retrieve(
    encode_key("atlas", "maps_to"),
    key_encoder=retriever.key_encoder,
    min_score=0.9,
)
intermediate_index = int(first_hop.hits[0].value_vector.argmax())
intermediate = VALUE_BY_INDEX[intermediate_index]
second_hop = retriever.retrieve(
    encode_key(intermediate, "maps_to"),
    key_encoder=retriever.key_encoder,
    min_score=0.9,
)

model = DenseLM(
    ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        max_seq_len=8,
    ),
    AttentionConfig(),
)
adapter = model.add_semantic_memory(
    "lesson",
    retriever,
    site="after_block",
    block_index=0,
    min_score=0.9,
)
trainable_adapter_parameters = sum(
    parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
)
model.eval()
model.requires_grad_(False)
adapter.freeze()
tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
queries = SemanticQueryBatch(
    retriever.key_encoder, encode_key("atlas", "maps_to").reshape(1, KEY_DIM)
)
with torch.inference_mode():
    logits = model(tokens, semantic_queries=queries)
    cached_logits, _ = model.forward_cached(
        tokens, cache_capacity=8, semantic_queries=queries
    )
torch.testing.assert_close(cached_logits, logits, atol=1e-5, rtol=1e-5)
metrics = {
    name: float(value.detach().cpu())
    for name, value in model.architecture_metric_tensors().items()
}
report = {
    "pack_id": retriever.pack_id,
    "dimensions": {
        "keys": [retriever.entry_count, retriever.key_dim],
        "values": [retriever.entry_count, retriever.memory_dim],
        "queries": [1, retriever.key_dim],
        "hidden": [1, tokens.shape[1], model.config.hidden_dim],
        "logits": list(logits.shape),
    },
    "retrieval": retrieval,
    "two_hop": {
        "status": "retrieved" if second_hop.status == "hit" else second_hop.status,
        "intermediate": intermediate,
        "answer": VALUE_BY_INDEX[int(second_hop.hits[0].value_vector.argmax())],
        "record_ids": [first_hop.hits[0].record_id, second_hop.hits[0].record_id],
    },
    "adapter": {
        "site": adapter.site,
        "block_index": adapter.block_index,
        "trainable_parameter_count_before_freeze": trainable_adapter_parameters,
        "frozen_after_attachment": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "traces": [
            {
                "status": trace.status,
                "best_record_id": trace.best_record_id,
                "best_score": trace.best_score,
                "comparison_count": trace.comparison_count,
            }
            for trace in adapter.last_traces
        ],
        "metrics": metrics,
    },
    "cached_forward_parity": True,
    "parameter_gradients_absent": all(
        parameter.grad is None for parameter in model.parameters()
    ),
}
print(json.dumps(report, indent=2, sort_keys=True))
"""
    demo = (
        demo.replace("__PACK_ID__", manifest.pack_id)
        .replace("__KEY_INDEX__", repr(key_indices))
        .replace("__VALUE_INDEX__", repr(value_indices))
    )
    demo_path = directory / "demo.py"
    _write(demo_path, demo)
    query_vectors = [
        [
            1.0 if index == key_indices[("atlas", "maps_to")] else 0.0
            for index in range(len(key_indices))
        ],
        [0.0 for _ in key_indices],
        [
            1.0 if index == key_indices[("fork", "maps_to")] else 0.0
            for index in range(len(key_indices))
        ],
        [
            1.0 if index == key_indices[("harbor", "label")] else 0.0
            for index in range(len(key_indices))
        ],
        [
            1.0 if index == key_indices[("vault", "label")] else 0.0
            for index in range(len(key_indices))
        ],
        [
            1.0 if index == key_indices[("atlas", "maps_to")] else 0.0
            for index in range(len(key_indices))
        ],
    ]
    probe = {
        "format": "klab-semantic-probe-v1",
        "model": {
            "kind": "initialized",
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 32,
                "max_seq_len": 8,
            },
            "attention": {},
            "seed": 17,
        },
        "input_tokens": [[1, 2, 3, 4, 5, 6]],
        "queries": [
            {
                "id": "lesson-keys",
                "encoder": key_encoder,
                "vectors": [query_vectors],
                "mask": [[True, True, True, True, True, False]],
                "as_of": [None, None, None, "2025-06-01", "2025-06-01", None],
            }
        ],
        "attachments": [
            {
                "name": "lesson",
                "pack": "semantic-pack",
                "expected_pack_id": manifest.pack_id,
                "query": "lesson-keys",
                "site": "after_block",
                "block_index": 0,
                "min_score": 0.9,
                "weights": {"kind": "initialized", "seed": 23},
            }
        ],
    }
    _write(directory / "probe.yaml", _yaml_bytes(probe))

    inputs = [
        {"path": name, "sha256": sha256_file(directory / name)}
        for name in (
            "records.jsonl",
            "semantic_keys.safetensors",
            "semantic_values.safetensors",
            "semantic.json",
            "demo.py",
            "probe.yaml",
        )
    ]
    pack_manifest_path = pack_path / "manifest.json"
    inputs.append(
        {
            "path": "semantic-pack/manifest.json",
            "sha256": sha256_file(pack_manifest_path),
        }
    )
    inputs.extend(
        {
            "path": f"semantic-pack/{item.relative_path}",
            "sha256": item.sha256,
        }
        for item in manifest.files
    )
    return {
        "records_sha256": sha256_file(records_path),
        "inputs": inputs,
        "semantic_pack": {
            "path": "semantic-pack",
            "pack_id": manifest.pack_id,
            "manifest_sha256": sha256_file(pack_manifest_path),
            "entry_count": semantic.entry_count,
            "key_dim": semantic.key_dim,
            "value_dim": semantic.memory_dim,
            "key_encoder": key_encoder,
            "value_encoder": value_encoder,
        },
    }


def _lesson_readme(
    lesson: MechanismLesson,
    scale: str,
    data: str,
    backend: str,
    *,
    artifact: bool,
    workspace: str | None = None,
    root: Path | None = None,
) -> str:
    command, environment_note = _command_environment("cpu" if artifact else backend)
    if artifact and lesson.id == "semantic-retrieval":
        walkthrough = "\n".join(
            f"- **{step.source_path}:{step.source_symbol}** — {step.explanation} "
            f"Shapes: {'; '.join(step.shapes)} Observe: {', '.join(step.observe)}."
            for step in lesson.steps
        )
        source_links = "\n".join(f"- {doc}" for doc in lesson.docs)
        return f"""# {lesson.title}

{lesson.summary}

This standalone workspace contains a verified semantic EngramPack, a ready
supplied-vector probe declaration, and an optional deeper Python API lesson.
Query vectors are explicit structured one-hot fixtures, not natural-language
embeddings.

## Run and inspect

```sh
{command} klab semantic probe probe.yaml --json
{command} klab engram pack inspect semantic-pack
{command} klab engram pack verify semantic-pack
```

The ordinary probe reports the verified pack identity, canonical supplied
vectors, retrieval statuses, temporal exclusions, deterministic tie IDs,
adapter site, and per-position traces, including its masked row. `demo.py` is
an explicitly optional deeper API example: run `{command} python demo.py` for
cached-forward parity and a two-edge structured lookup. No tokenizer, model
training run, external dataset, text encoder, or network access is required.

## Tensor walkthrough

This fixture uses keys `[7,6]`, values `[7,7]`, queries `[1,6]`, and backbone hidden states `[1,3,16]`. The adapter maps value width `7` to hidden width `16`; the key encoder width `6` is independent. The demo prints actual output and diagnostic shapes.

{walkthrough}

**Try changes:** {"; ".join(lesson.try_changes)}

**Limits:** {"; ".join(lesson.limits)}

## Source and follow-up

{source_links}
"""
    if artifact:
        return f"""# {lesson.title}

{lesson.summary}

These records are original tutorial-only CC0 triples. They are not a sealed evaluation card, a dynamic-world dataset, or executable model retrieval.

```sh
{command} klab engram pack compile records.jsonl --output artifacts/tutorial-pack --name tutorial-map --namespace tutorial --license CC0-1.0 --source-name original-tutorial-records --created-at 2026-09-23T00:00:00Z
{command} klab engram pack inspect artifacts/tutorial-pack
{command} klab engram pack verify artifacts/tutorial-pack
```

`compile`, `inspect`, and `verify` exercise the artifact format only. There is no `model.yaml`, `tokenizer.yaml`, or probe path in this artifact lesson.
"""
    return f"""# {lesson.title}

{lesson.summary}

**Selection:** `{scale}` / `{data}` / `{backend}`. This standalone path can be used without a campaign or worker.

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
    memory_package: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Write a single-mechanism lesson configuration or an artifact walkthrough."""
    lesson = load_lesson(identifier, root=root)
    if data not in _dataset_names(root=root):
        raise ValueError(
            f"unsupported dataset {data!r}; choose {', '.join(_dataset_names(root=root))}"
        )
    if backend not in _BACKENDS:
        raise ValueError(
            f"unsupported backend {backend!r}; choose {', '.join(sorted(_BACKENDS))}"
        )
    if lesson.mechanism_kind == "model":
        if scale not in load_profiles(root=root).scales:
            raise ValueError(
                f"unsupported runnable scale {scale!r}; larger named classes are recommendation-only"
            )
        if scale not in lesson.patches_by_scale:
            raise ValueError(f"lesson {identifier} has no patch for scale {scale}")
    if identifier == "portable-engram":
        if memory_package is None:
            raise ValueError(
                "portable-engram requires --memory-package PATH. First run: "
                "klab learn scaffold byte-engram --scale smoke --data offline --backend cpu --output experiments/byte-engram; "
                "klab tokenizer train experiments/byte-engram/tokenizer.yaml; "
                "klab data prepare experiments/byte-engram/model.yaml; "
                "klab train experiments/byte-engram/model.yaml --run-id byte-engram --runs-dir klab-work/experiments/byte-engram/runs; "
                "klab engram export byte-engram --runs-dir klab-work/experiments/byte-engram/runs --output experiments/byte-engram/memory.engram; "
                "klab learn scaffold portable-engram --memory-package experiments/byte-engram/memory.engram --output experiments/portable-engram"
            )
        if memory_package.is_symlink() or not memory_package.is_file():
            raise ValueError(
                "--memory-package must be an existing regular nonsymlink exported table"
            )
        package = load_portable_engram(memory_package)
    else:
        package = None

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
        if lesson.mechanism_kind == "artifact":
            if identifier == "semantic-retrieval":
                lesson_payload.update(_semantic_lesson_workspace(temporary))
            else:
                record_rows = [
                    {
                        "record_version": 1,
                        "id": "tutorial-atlas-maps-to-beacon",
                        "namespace": "tutorial-map",
                        "subject": "atlas",
                        "relation": "maps_to",
                        "value": "beacon",
                        "license": "CC0-1.0",
                    },
                    {
                        "record_version": 1,
                        "id": "tutorial-beacon-maps-to-amber",
                        "namespace": "tutorial-map",
                        "subject": "beacon",
                        "relation": "maps_to",
                        "value": "amber",
                        "license": "CC0-1.0",
                    },
                    {
                        "record_version": 1,
                        "id": "tutorial-cipher-maps-to-delta",
                        "namespace": "tutorial-map",
                        "subject": "cipher",
                        "relation": "maps_to",
                        "value": "delta",
                        "license": "CC0-1.0",
                    },
                ]
                records = b"".join(canonical_json(row) + b"\n" for row in record_rows)
                _write(temporary / "records.jsonl", records)
                records_sha256 = hashlib.sha256(records).hexdigest()
                lesson_payload["records_sha256"] = records_sha256
                lesson_payload["inputs"] = [
                    {"path": "records.jsonl", "sha256": records_sha256}
                ]
            _write(
                temporary / "README.md",
                _lesson_readme(lesson, scale, data, backend, artifact=True, root=root),
            )
        else:
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
            if package is not None:
                asset_dir = temporary / "assets"
                asset_dir.mkdir()
                if memory_package is None:
                    raise ValueError("portable package disappeared before copy")
                shutil.copyfile(memory_package, asset_dir / "memory.engram")
                source = package.manifest
                base = _apply_overlay(
                    base,
                    {
                        "model.memory": "portable",
                        "model.memory_injection": "final",
                        "model.memory_table_size": source.table_size,
                        "model.memory_ngram_size": source.ngram_size,
                        "model.memory_dim": source.embedding_dim,
                        "model.memory_package_path": "assets/memory.engram",
                        "model.memory_ngram_orders": [],
                        "model.memory_hash_heads": 1,
                    },
                )
                lesson_payload["memory_package"] = {
                    "table_sha256": source.table_sha256,
                    "table_size": source.table_size,
                    "embedding_dim": source.embedding_dim,
                    "ngram_size": source.ngram_size,
                }
                lesson_payload["inputs"] = [
                    {
                        "path": "assets/memory.engram",
                        "sha256": sha256_file(asset_dir / "memory.engram"),
                    }
                ]
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
            lesson_payload["config_sha256"] = config_sha256(
                config.model_dump(mode="json")
            )
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
                    artifact=False,
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
        if lesson.mechanism_kind == "model":
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
