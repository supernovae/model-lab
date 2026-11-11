"""Source-bound, teacher-forced scoring of a frozen base-language profile.

The adapter uses the native release, tokenizer and checkpoint verifiers. It does
not choose checkpoints, decode text, or modify training/evaluation declarations.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional

from klab.config.loading import load_tokenizer_config
from klab.corpus.release import verify_release
from klab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact
from klab.evaluation.generation_request import generate_result
from klab.training.attempt_budget import AttemptBudget
from klab.training.manifest import sha256_file
from klab.verification_proofs import verification_options


@dataclass(frozen=True)
class BoundFixedSlices:
    profile: dict[str, Any]
    profile_sha256: str
    tokenizer: Any
    ids: dict[str, list[int]]
    documents: dict[str, dict[str, Any]]
    inventory: dict[str, dict[str, Any]]


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def bind_fixed_slices(
    profile_path: Path,
    release: Path,
    tokenizer_path: Path,
    tokenizer_config: Path,
    family_inventory: Path,
    *,
    expected_profile_sha256: str | None = None,
    expected_family_sha256: str | None = None,
) -> BoundFixedSlices:
    """Cold-bind exact source bytes, lineage and tokenizer offsets before scoring."""
    profile_sha = sha256_file(profile_path)
    if expected_profile_sha256 is not None and profile_sha != expected_profile_sha256:
        raise ValueError("fixed profile digest differs from declaration")
    if (
        expected_family_sha256 is not None
        and sha256_file(family_inventory) != expected_family_sha256
    ):
        raise ValueError("fixed family inventory digest differs from declaration")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        release.parent.name not in {"builds", "releases"}
        or release.parents[2].name != "corpora"
    ):
        raise ValueError("fixed profile release must reside in a native corpus root")
    verification = verification_options(release.parents[3])
    manifest = verify_release(release, **verification)
    if (
        profile.get("schema_version") != 1
        or profile.get("release_id") != manifest["release_id"]
        or profile.get("release_manifest_sha256")
        != sha256_file(release / "manifest.json")
        or profile.get("documents_sha256") != sha256_file(release / "documents.jsonl")
        or profile.get("tokenizer_sha256") != sha256_file(tokenizer_path)
        or profile.get("tokenization")
        != {"add_special_tokens": False, "padding": False, "truncation": False}
    ):
        raise ValueError("fixed profile release or tokenizer binding mismatch")
    slices = profile.get("loss_slices")
    pairs = profile.get("utility_pairs")
    continuations = profile.get("continuations")
    if (
        not isinstance(slices, list)
        or not isinstance(pairs, list)
        or not isinstance(continuations, list)
    ):
        raise TypeError("fixed profile item collections are required")
    ids = [row.get("id") for row in slices]
    if (
        len(ids) != 24
        or len(set(ids)) != 24
        or any(not isinstance(item, str) for item in ids)
    ):
        raise ValueError("fixed profile requires 24 unique slice IDs")
    coverage = Counter((row.get("split"), row.get("stratum")) for row in slices)
    if coverage != Counter(
        {
            (split, stratum): 4
            for split in ("validation", "test")
            for stratum in (
                "general_prose",
                "explanatory_prose",
                "incident_response_docs",
            )
        }
    ):
        raise ValueError("fixed profile split/stratum coverage mismatch")
    if len(pairs) != 24 or {row.get("slice_id") for row in pairs} != set(ids):
        raise ValueError("utility pairs must cover every slice exactly once")
    if len({row.get("slice_id") for row in pairs}) != len(pairs):
        raise ValueError("duplicate utility pair")
    if (
        len(continuations) != 8
        or len({row.get("slice_id") for row in continuations}) != 8
    ):
        raise ValueError("fixed profile requires eight distinct continuations")
    slice_map = {row["id"]: row for row in slices}
    for row in slices:
        if (
            row.get("input_tokens") != 257
            or row.get("scored_targets") != 256
            or type(row.get("start_token")) is not int
            or row["start_token"] < 0
        ):
            raise ValueError("invalid fixed loss window")
    for row in pairs:
        if (
            row.get("context_tokens"),
            row.get("true_next_tokens"),
            row.get("decoy_tokens"),
        ) != (64, 32, 32):
            raise ValueError("invalid fixed utility window")
        if any(
            type(row.get(key)) is not int or row[key] < 0
            for key in ("context_start_token", "decoy_start_token")
        ):
            raise ValueError("invalid utility offset")
    for row in continuations:
        source = slice_map.get(row.get("slice_id"))
        if (
            source is None
            or source["stratum"] not in {"general_prose", "explanatory_prose"}
            or (row.get("prompt_tokens"), row.get("max_new_tokens")) != (64, 64)
            or type(row.get("prompt_start_token")) is not int
            or row["prompt_start_token"] < 0
        ):
            raise ValueError("invalid continuation binding")
    needed = {row["document_id"] for row in slices} | {
        row["decoy_document_id"] for row in pairs
    }
    documents = {
        row["document_id"]: row
        for row in _jsonl(release / "documents.jsonl")
        if row["document_id"] in needed
    }
    inventory = {}
    for row in _jsonl(family_inventory):
        document_id = row["document_id"]
        if document_id in inventory:
            raise ValueError("duplicate fixed family-inventory document ID")
        if document_id in needed:
            inventory[document_id] = row
    if set(documents) != needed or set(inventory) != needed:
        raise ValueError("fixed profile document or family inventory missing")
    fit = load_tokenizer_config(tokenizer_config)
    if (fit.output_dir / "tokenizer.json").resolve() != tokenizer_path.resolve():
        raise ValueError("fixed profile tokenizer config points elsewhere")
    verify_tokenizer_artifact(
        tokenizer_path,
        source=fit.dataset.source,
        revision=fit.dataset.revision,
        vocab_size=fit.vocab_size,
        dataset=fit.dataset,
        **verification,
    )
    tokenizer = load_tokenizer(tokenizer_path)
    token_ids = {}
    for document_id in needed:
        doc = documents[document_id]
        line = inventory[document_id]
        if (
            doc["drop_reason"] is not None
            or doc["split"] not in {"validation", "test"}
            or any(
                line.get(key) != value
                for key, value in (
                    ("document_id", document_id),
                    ("split", doc["split"]),
                    ("content_sha256", doc["content_sha256"]),
                )
            )
            or line.get("family_id") is None
        ):
            raise ValueError(
                "fixed profile document is not an admitted held-out family"
            )
        token_ids[document_id] = tokenizer.encode(
            doc["text"], add_special_tokens=False
        ).ids
    for row in slices:
        doc_id = row["document_id"]
        doc = documents[doc_id]
        line = inventory[doc_id]
        if any(
            (row[key] != expected)
            for key, expected in (
                ("content_sha256", doc["content_sha256"]),
                ("source_id", doc["source_id"]),
                ("split", doc["split"]),
                ("stratum", line["stratum"]),
                ("family_id", line["family_id"]),
                ("document_tokens", len(token_ids[doc_id])),
            )
        ) or row["start_token"] + 257 > len(token_ids[doc_id]):
            raise ValueError("fixed loss slice differs from authenticated source")
    for row in pairs:
        source = slice_map[row["slice_id"]]
        decoy_id = row["decoy_document_id"]
        decoy = inventory[decoy_id]
        if (
            decoy_id == source["document_id"]
            or decoy["family_id"] == source["family_id"]
            or decoy["split"] != source["split"]
            or decoy["stratum"] != source["stratum"]
            or row["decoy_start_token"] + 32 > len(token_ids[decoy_id])
            or row["context_start_token"] + 96 > len(token_ids[source["document_id"]])
        ):
            raise ValueError(
                "utility decoy is not independent held-out same-stratum text"
            )
    for row in continuations:
        source = slice_map[row["slice_id"]]
        source_ids = token_ids[source["document_id"]]
        if row["prompt_start_token"] + 64 > len(source_ids):
            raise ValueError("continuation prompt exceeds held-out document")
        prompt_ids = source_ids[
            row["prompt_start_token"] : row["prompt_start_token"] + 64
        ]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if tokenizer.encode(prompt, add_special_tokens=False).ids != prompt_ids:
            raise ValueError("continuation prompt does not reproduce frozen token IDs")
    return BoundFixedSlices(
        profile, profile_sha, tokenizer, token_ids, documents, inventory
    )


def score_window(
    model: torch.nn.Module,
    token_ids: list[int],
    *,
    first_target: int,
    target_count: int,
    device: torch.device,
) -> float:
    """Sum next-token NLL over explicitly selected targets; inputs are charged whole."""
    if not 0 < first_target <= len(token_ids) - target_count:
        raise ValueError("invalid target positions")
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(x)
        if logits.ndim != 3 or logits.shape[:2] != x.shape:
            raise ValueError(
                "fixed scorer requires [batch, input length, vocabulary] logits"
            )
        chosen = logits[:, first_target - 1 : first_target + target_count - 1]
        targets = x[:, first_target : first_target + target_count]
        loss = functional.cross_entropy(
            chosen.float().reshape(-1, chosen.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
    if not torch.isfinite(loss):
        raise ValueError("nonfinite fixed-slice loss")
    return float(loss)


def score_fixed_slices(
    bound: BoundFixedSlices,
    model: torch.nn.Module,
    device: torch.device,
    mode: Literal["validation", "test", "utility"],
    max_forward_positions: int,
) -> dict[str, Any]:
    """Score one checkpoint without generation, rejecting cap overshoot before a forward."""
    if type(max_forward_positions) is not int or max_forward_positions <= 0:
        raise ValueError("positive forward-input cap required")
    rows = {row["id"]: row for row in bound.profile["loss_slices"]}
    used = 0
    targets = 0
    output = []
    was_training = model.training

    def charged(label: str, ids: list[int], first_target: int, count: int) -> float:
        nonlocal used, targets
        if used + len(ids) > max_forward_positions:
            raise ValueError("fixed profile forward-input cap exhausted")
        if AttemptBudget.forward_allocation_active_from_environment():
            AttemptBudget.reserve_forward_from_environment(
                f"fixed:{bound.profile_sha256}:{mode}:{label}",
                kind="fixed_profile",
                positions=len(ids),
            )
        used += len(ids)
        nll = score_window(
            model, ids, first_target=first_target, target_count=count, device=device
        )
        targets += count
        return nll

    try:
        model.eval()
        if mode in {"validation", "test"}:
            for row in bound.profile["loss_slices"]:
                if row["split"] != mode:
                    continue
                tokens = bound.ids[row["document_id"]][
                    row["start_token"] : row["start_token"] + 257
                ]
                nll = charged(row["id"], tokens, 1, 256)
                output.append(
                    {
                        "id": row["id"],
                        "stratum": row["stratum"],
                        "nll_nats": nll,
                        "targets": 256,
                    }
                )
        elif mode == "utility":
            for pair in bound.profile["utility_pairs"]:
                source = rows[pair["slice_id"]]
                start = pair["context_start_token"]
                context = bound.ids[source["document_id"]][start : start + 64]
                true = bound.ids[source["document_id"]][start + 64 : start + 96]
                decoy = bound.ids[pair["decoy_document_id"]][
                    pair["decoy_start_token"] : pair["decoy_start_token"] + 32
                ]
                true_nll = charged(pair["slice_id"] + ":true", context + true, 64, 32)
                decoy_nll = charged(
                    pair["slice_id"] + ":decoy", context + decoy, 64, 32
                )
                output.append(
                    {
                        "id": pair["slice_id"],
                        "stratum": source["stratum"],
                        "true_nll_nats": true_nll,
                        "decoy_nll_nats": decoy_nll,
                        "prefers_true": true_nll < decoy_nll,
                    }
                )
        else:
            raise ValueError("unknown fixed-profile scoring mode")
    finally:
        model.train(was_training)
    if (
        not output
        or (mode == "utility" and (len(output), used, targets) != (24, 4608, 1536))
        or (mode != "utility" and (len(output), used, targets) != (12, 3084, 3072))
    ):
        raise ValueError("fixed-profile coverage or accounting mismatch")
    result: dict[str, Any] = {
        "mode": mode,
        "profile_sha256": bound.profile_sha256,
        "forward_input_positions": used,
        "scored_targets": targets,
        "items": output,
    }
    if mode != "utility":
        total = sum(row["nll_nats"] for row in output)
        result["loss"] = total / targets
        result["per_stratum_loss"] = {
            stratum: sum(row["nll_nats"] for row in output if row["stratum"] == stratum)
            / sum(row["targets"] for row in output if row["stratum"] == stratum)
            for stratum in (
                "general_prose",
                "explanatory_prose",
                "incident_response_docs",
            )
        }
        if not math.isfinite(result["loss"]):
            raise ValueError("nonfinite fixed-profile aggregate loss")
    else:
        result["preferred_true"] = sum(row["prefers_true"] for row in output)
    return result


def generate_fixed_continuations(
    bound: BoundFixedSlices,
    model: torch.nn.Module,
    device: torch.device,
    max_seq_len: int,
) -> dict[str, Any]:
    """Generate the eight frozen plain-prose prompts with contracted decoding."""
    if max_seq_len < 128:
        raise ValueError("fixed continuation needs 128 context positions")
    rows = {row["id"]: row for row in bound.profile["loss_slices"]}
    output = []
    actual_forward = 0
    for continuation in bound.profile["continuations"]:
        source = rows[continuation["slice_id"]]
        start = continuation["prompt_start_token"]
        prompt_ids = bound.ids[source["document_id"]][start : start + 64]
        if len(prompt_ids) != 64:
            raise ValueError("fixed continuation prompt is incomplete")
        prompt = bound.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if bound.tokenizer.encode(prompt, add_special_tokens=False).ids != prompt_ids:
            raise ValueError("fixed continuation prompt IDs changed")
        result = generate_result(
            model,
            bound.tokenizer,
            prompt,
            max_seq_len,
            64,
            device,
            temperature=0,
            top_k=0,
            strict_context=True,
            use_cache=True,
            accounting_label=f"fixed:{bound.profile_sha256}:continuation:{continuation['slice_id']}",
        )
        actual_forward += result.forward_input_positions
        output.append(
            {
                "slice_id": continuation["slice_id"],
                "stratum": source["stratum"],
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "completion": result.text[len(prompt) :],
                "completion_ids": result.token_ids,
                "finish_reason": result.finish_reason,
                "forward_input_positions": result.forward_input_positions,
                "cache_used": result.cache_used,
            }
        )
    if len(output) != 8 or actual_forward > 8 * 6112:
        raise ValueError("fixed continuation coverage or accounting mismatch")
    return {
        "mode": "continuations",
        "profile_sha256": bound.profile_sha256,
        "generation_calls": len(output),
        "generated_tokens": sum(len(row["completion_ids"]) for row in output),
        "requested_new_tokens": 64 * len(output),
        "forward_input_positions_actual": actual_forward,
        "forward_input_positions_reserved": 6112 * len(output),
        "items": output,
    }
