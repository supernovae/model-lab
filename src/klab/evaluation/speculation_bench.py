"""Measure speculative drafters against plain greedy decoding on fixed prompts.

Every drafter is held to one criterion before any speed is reported: its
completion must equal plain greedy decoding's, token for token. Then each
reports the share of drafted tokens the target accepted, the verifying model
calls it used and wall-time tokens per second against the plain decoding of the
same prompts on the same device.

Wall time is a property of the host, the backend, the model size and the
prompts; the record names them and nothing here extrapolates to another device.
"""

from __future__ import annotations

import json
import platform
import secrets
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from klab.canonical import canonical_digest
from klab.evaluation.generation import generate_with_token_ids
from klab.evaluation.speculation import Drafter
from klab.lab_records import KINDS, read_lab_record, write_sealed

SPECULATION_FORMAT = KINDS["speculation"][0]
PROTOCOL = "speculation-bench-v1"
COUNTERS = ("model_calls", "drafted_tokens", "accepted_tokens", "draft_model_calls")


def validation_prompts(
    run: Path, tokenizer: Any, *, count: int, tokens: int
) -> list[str]:
    """Take ``count`` evenly spaced ``tokens``-token windows of the validation split.

    The split is held out from training, so a corpus table built from the
    training split has not seen these prompts, and the prompts are a function of
    the run's data alone.
    """
    if count < 1 or tokens < 1:
        raise ValueError("prompt count and prompt tokens must be at least 1")
    split = np.load(run / "data" / "validation.npy", mmap_mode="r", allow_pickle=False)
    if len(split) < count * tokens:
        raise ValueError(
            f"the validation split holds {len(split)} tokens, fewer than "
            f"{count} prompts of {tokens} tokens"
        )
    stride = len(split) // count
    return [
        tokenizer.decode(
            [int(token) for token in split[index * stride : index * stride + tokens]],
            skip_special_tokens=True,
        )
        for index in range(count)
    ]


def read_prompts(path: Path) -> list[str]:
    """Read a JSON list of non-empty prompt strings."""
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(item, str) and item for item in prompts)
    ):
        raise ValueError("a prompts file must hold a non-empty JSON list of strings")
    return prompts


def _decode(
    loaded: Any, prompt: str, max_new_tokens: int, drafter: Drafter | None
) -> tuple[list[int], float, dict[str, int]]:
    before = {name: getattr(drafter, name, 0) for name in COUNTERS}
    started = time.perf_counter()
    _, tokens = generate_with_token_ids(
        loaded.model,
        loaded.tokenizer,
        prompt,
        loaded.config.model.max_seq_len,
        max_new_tokens,
        loaded.device,
        speculation=drafter,
    )
    seconds = time.perf_counter() - started
    used = {name: getattr(drafter, name, 0) - before[name] for name in COUNTERS}
    return tokens, seconds, used


def measure(
    loaded: Any,
    prompts: Sequence[str],
    drafters: Mapping[str, Drafter],
    *,
    max_new_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    """Run plain greedy and each drafter over ``prompts``; return their summaries.

    One untimed pass per decoder comes first (warm-up: kernels, caches). Each
    timed prompt is decoded ``repeats`` times and its median kept. Token
    equality with plain decoding is checked on every run.
    """
    if repeats < 1 or max_new_tokens < 1:
        raise ValueError("repeats and max_new_tokens must be at least 1")
    loaded.model.eval()
    decoders: dict[str, Drafter | None] = {"plain": None, **drafters}
    for decoder in decoders.values():
        _decode(loaded, prompts[0], max_new_tokens, decoder)
    reference: list[list[int]] = []
    results: dict[str, dict[str, Any]] = {}
    for name, decoder in decoders.items():
        rows = []
        for index, prompt in enumerate(prompts):
            runs = [
                _decode(loaded, prompt, max_new_tokens, decoder) for _ in range(repeats)
            ]
            tokens, _, used = runs[-1]
            if name == "plain":
                reference.append(tokens)
            rows.append(
                {
                    "tokens": len(tokens),
                    "wall_seconds": statistics.median(run[1] for run in runs),
                    "matches_plain": all(run[0] == reference[index] for run in runs),
                    **used,
                }
            )
        generated = sum(row["tokens"] for row in rows)
        wall = sum(row["wall_seconds"] for row in rows)
        summary: dict[str, Any] = {
            "prompts": rows,
            "generated_tokens": generated,
            "wall_seconds": wall,
            "tokens_per_second": generated / wall if wall else 0.0,
            "matches_plain": all(row["matches_plain"] for row in rows),
        }
        if decoder is not None:
            totals = {name: sum(row[name] for row in rows) for name in COUNTERS}
            summary.update(totals)
            summary["acceptance_rate"] = (
                totals["accepted_tokens"] / totals["drafted_tokens"]
                if totals["drafted_tokens"]
                else 0.0
            )
        results[name] = summary
    plain_rate = results["plain"]["tokens_per_second"]
    for name, summary in results.items():
        summary["speedup_vs_plain"] = (
            summary["tokens_per_second"] / plain_rate if plain_rate else 0.0
        )
    return results


def host_description(loaded: Any) -> dict[str, Any]:
    return {
        "backend": loaded.config.runtime.backend,
        "device": str(loaded.device),
        "precision": loaded.config.runtime.precision,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }


def run_speculation_bench(
    loaded: Any,
    drafters: Mapping[str, Drafter],
    drafter_identities: Mapping[str, Any],
    *,
    prompts: Sequence[str],
    prompt_source: Mapping[str, Any],
    lab_dir: Path,
    request: Mapping[str, Any],
    max_new_tokens: int,
    repeats: int,
    arm_identity: Callable[[Any], dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    """Measure and publish ``LAB/speculation/<id>/speculation.json`` (sealed).

    A drafter whose output differs from plain greedy decoding makes the record
    ``mismatch``; it is still published, because a failure is a result.
    """
    if loaded.engine is not None:
        raise ValueError("speculation is measured on the PyTorch engine only")
    if not drafters:
        raise ValueError("choose at least one drafter")
    results = measure(
        loaded, prompts, drafters, max_new_tokens=max_new_tokens, repeats=repeats
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    record_id = f"speculation-{stamp}-{secrets.token_hex(4)}"
    folder = lab_dir / KINDS["speculation"][1] / record_id
    folder.mkdir(parents=True, exist_ok=False)
    record = {
        "format": SPECULATION_FORMAT,
        "speculation_id": record_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "complete"
        if all(result["matches_plain"] for result in results.values())
        else "mismatch",
        "request": dict(request),
        "target": arm_identity(loaded),
        "host": host_description(loaded),
        "protocol": {
            "protocol": PROTOCOL,
            "decoding": "greedy",
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "timing": "median wall seconds per prompt after one untimed warm-up",
            "prompt_source": dict(prompt_source),
            "prompts": len(prompts),
            "prompts_sha256": canonical_digest(list(prompts)),
        },
        "drafters": dict(drafter_identities),
        "results": results,
    }
    path = folder / KINDS["speculation"][2]
    return write_sealed(path, record), path


def read_speculation(path: Path) -> dict[str, Any]:
    return read_lab_record(path, "speculation")[1]


def render(record: Mapping[str, Any]) -> str:
    """Plain-text report of one sealed speculation record."""
    target, host, protocol = record["target"], record["host"], record["protocol"]
    kind = protocol["prompt_source"]["kind"]
    lines = [
        (
            f"SPECULATION {record['speculation_id']}  "
            f"{target.get('run_id')}@{target.get('step')} · {record['status'].upper()}"
        ),
        (
            f"  host  {host['backend']} ({host['device']}) · torch {host['torch']} · "
            f"{host['cpu']} · {host['torch_threads']} threads"
        ),
        (
            f"  protocol  {protocol['prompts']} prompts from {kind} · up to "
            f"{protocol['max_new_tokens']} new tokens · median of "
            f"{protocol['repeats']} · greedy"
        ),
        "",
        f"  {'decoder':<30}{'match':>6}{'accept':>8}{'calls':>7}{'tok/s':>9}{'x plain':>9}",
    ]
    for name, result in record["results"].items():
        accept = result.get("acceptance_rate")
        calls = result.get("model_calls")
        shown = {
            "match": "yes" if result["matches_plain"] else "NO",
            "accept": "-" if accept is None else f"{accept:.2f}",
            "calls": "-" if calls is None else str(calls),
        }
        lines.append(
            f"  {name:<30}{shown['match']:>6}{shown['accept']:>8}{shown['calls']:>7}"
            f"{result['tokens_per_second']:>9.1f}{result['speedup_vs_plain']:>9.2f}"
        )
    lines.append(
        "  wall times describe this host and backend only; rerun on the device "
        "you care about"
    )
    return "\n".join(lines)
