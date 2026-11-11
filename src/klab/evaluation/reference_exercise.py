"""Deterministic, checkpoint-bound observations for dense language-model references."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal, TypedDict

import torch

from klab.canonical import canonical_json
from klab.evaluation.capabilities import capability_card, evaluate_capability
from klab.evaluation.generation import generate
from klab.evaluation.inference import InferenceRun, load_run

Status = Literal["PASS", "FAIL", "NOT_APPLICABLE", "UNAVAILABLE"]


class ExerciseGroup(TypedDict):
    status: Status
    details: dict[str, object]


class LearningObservation(TypedDict):
    format: Literal["klab_learning_observation_v1"]
    identity: str
    suite_version: int
    run_id: str
    checkpoint: dict[str, object]
    groups: dict[str, ExerciseGroup]
    promotion_eligible: bool


PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "simple-continuation",
        "Once upon a time, a little rabbit found a shiny red apple.",
    ),
    (
        "named-character-continuity",
        "Mia and her brother Leo built a small boat together. Mia carried the boat to the pond, and Leo",
    ),
    (
        "color-object-continuity",
        "A blue kite flew over the green meadow. The kite was blue, and the meadow was",
    ),
    (
        "temporal-causal-continuation",
        "After the rain stopped, Sam opened the door. The path was wet, so Sam",
    ),
    (
        "ordinary-sentence-completion",
        "The baker put warm bread on the table, and everyone",
    ),
    (
        "odd-out-of-domain",
        "A tiny moon made of brass hummed quietly beside the library clock.",
    ),
)

_REQUIRED_GROUPS = (
    "integrity",
    "held_out_lm",
    "deterministic_generation",
    "fixed_prompt_panel",
    "degeneration_diagnostics",
)


def promotion_gate(groups: dict[str, ExerciseGroup]) -> bool:
    """Missing or non-passing required gates block promotion."""
    return all(
        name in groups and groups[name]["status"] == "PASS" for name in _REQUIRED_GROUPS
    )


def _group(status: Status, **details: object) -> ExerciseGroup:
    return {"status": status, "details": details}


def _finite_state(run: InferenceRun) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in run.model.state_dict().values()
        if value.is_floating_point()
    )


def _diagnostics(run: InferenceRun, prompt: str, output: str) -> dict[str, object]:
    encoded = run.tokenizer.encode(output, add_special_tokens=False).ids
    counts = Counter(encoded)
    repeated_bigrams = sum(
        count - 1
        for count in Counter(itertools.pairwise(encoded)).values()
        if count > 1
    )
    repeated_trigrams = sum(
        count - 1
        for count in Counter(zip(encoded, encoded[1:], encoded[2:])).values()
        if count > 1
    )
    special = {
        run.tokenizer.token_to_id(token)
        for token in ("<pad>", "<bos>", "<eos>", "<unk>")
    }
    special.discard(None)
    return {
        "empty": not output.strip(),
        "token_count": len(encoded),
        "distinct_token_count": len(counts),
        "distinct_token_ratio": len(counts) / len(encoded) if encoded else 0.0,
        "repeated_token_excess": sum(max(0, count - 1) for count in counts.values()),
        "repeated_bigram_excess": repeated_bigrams,
        "repeated_trigram_excess": repeated_trigrams,
        "contains_eos": run.tokenizer.token_to_id("<eos>") in encoded,
        "contains_bos": run.tokenizer.token_to_id("<bos>") in encoded,
        "special_token_share": sum(token in special for token in encoded) / len(encoded)
        if encoded
        else 0.0,
        "prompt_token_count": len(
            run.tokenizer.encode(prompt, add_special_tokens=False).ids
        ),
        "decode_error": False,
    }


def exercise_checkpoint(
    run_id: str,
    checkpoint: Path,
    *,
    runs_dir: Path,
    backend: str | None = None,
    prompt_panel: Path | None = None,
) -> LearningObservation:
    """Evaluate exactly one explicit immutable checkpoint; pointers are rejected."""
    if not checkpoint:
        raise ValueError("an explicit immutable checkpoint is required")
    selected = checkpoint.resolve()
    if selected.name in {"best.json", "latest.json"} or selected.suffix == ".json":
        raise ValueError(
            "checkpoint must be an explicit immutable generation directory"
        )
    run = load_run(run_id, runs_dir, str(selected), backend)
    if selected != (run.run / "checkpoints" / selected.name).resolve():
        raise ValueError("checkpoint must belong to the selected run")
    identity = dict(run.identity)
    if not _finite_state(run):
        raise ValueError("checkpoint contains nonfinite model weights")
    if (
        run.config.attention.kind != "dense"
        or run.config.model.ffn != "dense"
        or not run.config.model.tie_embeddings
        or run.config.model.memory != "none"
    ):
        raise ValueError(
            "checkpoint architecture differs from the dense reference contract"
        )
    expected_tokens = (
        identity["step"]
        * run.config.training.micro_batch_size
        * run.config.training.gradient_accumulation
        * run.config.training.seq_len
    )
    if identity["tokens_seen"] != expected_tokens:
        raise ValueError(
            "checkpoint step/token coordinates violate the frozen batch contract"
        )

    coordinate = {
        "step": identity.get("step"),
        "tokens_seen": identity.get("tokens_seen"),
    }
    integrity = _group(
        "PASS",
        checkpoint_sha256=identity.get("checkpoint_sha256"),
        checkpoint_relative_path=identity.get("checkpoint_relative_path"),
        source_identity_sha256=identity.get("source_identity_sha256"),
        tokenizer_sha256=identity.get("tokenizer_sha256"),
        data_sha256=identity.get("data_sha256"),
        coordinate=coordinate,
        architecture=run.config.model.model_dump(mode="json"),
        finite_weights=True,
    )

    validation = run.evaluate()
    loss = validation.get("loss")
    targets = validation.get("valid_targets")
    lm_pass = (
        isinstance(loss, (float, int))
        and math.isfinite(float(loss))
        and isinstance(targets, int)
        and targets > 0
    )
    groups: dict[str, ExerciseGroup] = {
        "integrity": integrity,
        "held_out_lm": _group("PASS" if lm_pass else "FAIL", validation=validation),
    }

    prompts = PROMPTS
    if prompt_panel is not None:
        payload = json.loads(prompt_panel.read_text())
        if (
            payload.get("format") != "dense_lm_prompt_panel_v1"
            or payload.get("version") != 1
        ):
            raise ValueError("unrecognized prompt panel identity")
        prompts = tuple((row["id"], row["text"]) for row in payload["prompts"])
        if prompts != PROMPTS:
            raise ValueError("prompt panel differs from the frozen suite panel")

    panel: list[dict[str, object]] = []
    all_nonempty = True
    all_repeatable = True
    all_cache_equal = True
    cache_capability = getattr(run.model, "incremental_cache_capability", None)
    cache_available = callable(getattr(run.model, "forward_cached", None)) and (
        cache_capability is None or bool(cache_capability.supported)
    )
    assert isinstance(run.device, torch.device)
    for prompt_id, prompt in prompts:
        options = {
            "temperature": 0.0,
            "top_k": 0,
            "seed": 42042,
            "strict_context": True,
        }
        try:
            output = generate(
                run.model,
                run.tokenizer,
                prompt,
                run.config.model.max_seq_len,
                32,
                run.device,
                **options,
            )
            repeated = generate(
                run.model,
                run.tokenizer,
                prompt,
                run.config.model.max_seq_len,
                32,
                run.device,
                **options,
            )
            cached = generate(
                run.model,
                run.tokenizer,
                prompt,
                run.config.model.max_seq_len,
                32,
                run.device,
                **options,
                use_cache=True,
            )
            full = generate(
                run.model,
                run.tokenizer,
                prompt,
                run.config.model.max_seq_len,
                32,
                run.device,
                **options,
                use_cache=False,
            )
        except Exception as error:  # noqa: BLE001 - preserve generation failures as evidence.
            panel.append(
                {
                    "id": prompt_id,
                    "prompt": prompt,
                    "output": None,
                    "error": f"{type(error).__name__}: {error}",
                    "diagnostics": {"decode_error": True},
                }
            )
            all_nonempty = all_repeatable = all_cache_equal = False
            continue
        completion = output[len(prompt) :] if output.startswith(prompt) else ""
        output_tokens = len(
            run.tokenizer.encode(completion, add_special_tokens=False).ids
        )
        row = {
            "id": prompt_id,
            "prompt": prompt,
            "output": output,
            "output_token_count": output_tokens,
            "boundary_valid": output.startswith(prompt),
            "repeatable": output == repeated,
            "cache_full_prefix_equal": cached == full,
            "diagnostics": _diagnostics(run, prompt, completion),
        }
        panel.append(row)
        all_nonempty &= (
            bool(completion.strip())
            and 0 < output_tokens <= 32
            and output.startswith(prompt)
        )
        all_repeatable &= output == repeated
        all_cache_equal &= cached == full
    groups["deterministic_generation"] = _group(
        "PASS"
        if all_nonempty and all_repeatable and (all_cache_equal or not cache_available)
        else "FAIL",
        settings={"temperature": 0.0, "top_k": 0, "seed": 42042, "max_new_tokens": 32},
        cache_supported=cache_available,
        cache_reference_equal=all_cache_equal if cache_available else None,
    )
    groups["fixed_prompt_panel"] = _group(
        "PASS" if all_nonempty else "FAIL", cases=panel
    )
    groups["degeneration_diagnostics"] = _group(
        "PASS" if all_nonempty else "FAIL",
        cases=[{"id": item["id"], **item["diagnostics"]} for item in panel],
    )
    card_results: list[dict[str, object]] = []
    for name in (
        "chat-alias-retention-v1",
        "chat-alias-recall-v1",
        "chat-context-override-v1",
    ):
        try:
            card_results.append(
                {
                    "applicability": "OUT_OF_DOMAIN",
                    **evaluate_capability(
                        capability_card(name),
                        run.model,
                        run.tokenizer,
                        run.config.model.max_seq_len,
                        run.device,
                        engine=run.engine,
                    ),
                }
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            card_results.append(
                {
                    "card": name,
                    "applicability": "OUT_OF_DOMAIN",
                    "valid": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    groups["capability_cards"] = _group(
        "NOT_APPLICABLE",
        note="No current card measures TinyStories language-model behavior; these raw outcomes are out-of-domain and cannot gate promotion.",
        outcomes=card_results,
    )
    groups["resource_capture"] = _group(
        "UNAVAILABLE",
        note="This checkpoint exercise has no synchronized training timing or allocator peak capture.",
        backend=identity.get("training_runtime"),
        measured_weight_bytes=sum(
            t.numel() * t.element_size() for t in run.model.state_dict().values()
        ),
        optimizer_bytes=None,
        checkpoint_bytes=None,
    )
    promotable = promotion_gate(groups)
    record: LearningObservation = {
        "format": "klab_learning_observation_v1",
        "identity": "",
        "suite_version": 1,
        "run_id": run_id,
        "checkpoint": identity,
        "groups": groups,
        "promotion_eligible": promotable,
    }
    record["identity"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in record.items() if key != "identity"}
        )
    ).hexdigest()
    return record


def write_observation(observation: LearningObservation, output: Path) -> Path:
    """Write a deterministic content-addressed observation without replacing data."""
    content = canonical_json(observation) + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    destination = (
        output / f"{digest}.json" if output.is_dir() or not output.suffix else output
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != content:
        raise FileExistsError(f"observation destination collision: {destination}")
    if not destination.exists():
        destination.write_bytes(content)
    return destination
