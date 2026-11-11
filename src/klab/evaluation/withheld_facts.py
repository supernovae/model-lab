"""Strict held-out completion evaluation for verified fact manifests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from klab.data.byte_hash import table_address
from klab.data.withheld_facts import verify_manifest
from klab.evaluation.generation import generate


def _candidate_log_probability(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    value: str,
    device: torch.device,
) -> tuple[float, int]:
    prompt_encoding = tokenizer.encode(prompt, add_special_tokens=False)
    full_text = f"{prompt} {value}"
    full_encoding = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_count = len(prompt_encoding.ids)
    value_ids = full_encoding.ids[prompt_count:]
    if prompt_count < 1 or not value_ids:
        raise ValueError(
            "withheld candidate must encode to non-empty prompt and value IDs"
        )
    input_ids = torch.tensor(
        [full_encoding.ids[:-1]],
        dtype=torch.long,
        device=device,
    )
    byte_addresses = None
    if model.config.memory in {"byte", "portable"}:
        addresses = [
            table_address(
                full_text[:end].encode("utf-8")[-model.config.memory_ngram_size :],
                model.config.memory_table_size,
            )
            for _, end in full_encoding.offsets
        ]
        byte_addresses = torch.tensor([addresses[:-1]], dtype=torch.long, device=device)
    logits = model(input_ids, byte_addresses=byte_addresses)[0, prompt_count - 1 :]
    targets = torch.tensor(value_ids, dtype=torch.long, device=device)
    log_probability = (
        torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).sum()
    )
    return float(log_probability / len(value_ids)), len(value_ids)


def evaluate_withheld_facts(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    manifest_path: Path,
    *,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, object]:
    """Generate each verified held-out completion and score exact continuations."""
    manifest = verify_manifest(manifest_path)
    held_out_cases = manifest["held_out_cases"]
    if not isinstance(held_out_cases, list):
        raise TypeError("verified manifest has invalid held-out cases")
    candidate_values = [case["expected_value"] for case in held_out_cases]
    if not all(isinstance(value, str) for value in candidate_values):
        raise TypeError("verified manifest has invalid expected values")
    model_was_training = model.training
    model.eval()
    cases: list[dict[str, object]] = []
    with torch.inference_mode():
        for case in held_out_cases:
            assert isinstance(case, dict)
            prompt = case["prompt"]
            expected = case["expected_value"]
            assert isinstance(prompt, str)
            assert isinstance(expected, str)
            completion = generate(
                model, tokenizer, prompt, max_seq_len, max_new_tokens, device
            )
            predicted = completion.removeprefix(prompt).strip()
            candidate_scores = [
                _candidate_log_probability(model, tokenizer, prompt, value, device)[0]
                for value in candidate_values
            ]
            expected_index = candidate_values.index(expected)
            expected_score = candidate_scores[expected_index]
            rank = 1 + sum(
                score > expected_score
                for index, score in enumerate(candidate_scores)
                if index != expected_index
            )
            cases.append(
                {
                    "candidate_rank": rank,
                    "candidate_count": len(candidate_values),
                    "exact_match": predicted == expected,
                    "expected_log_probability": expected_score,
                    "expected_value": expected,
                    "prediction": predicted,
                    "prompt": prompt,
                }
            )
    model.train(model_was_training)
    exact_match_count = sum(bool(case["exact_match"]) for case in cases)
    return {
        "case_count": len(cases),
        "cases": cases,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / len(cases),
        "mean_candidate_reciprocal_rank": sum(
            1 / int(case["candidate_rank"]) for case in cases
        )
        / len(cases),
        "mean_expected_log_probability": sum(
            float(case["expected_log_probability"]) for case in cases
        )
        / len(cases),
        "manifest_sha256": manifest["sha256"],
    }


def write_withheld_evaluation(path: Path, report: dict[str, object]) -> None:
    """Atomically retain immutable held-out completion evidence."""
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"conflicting withheld-fact evaluation exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
