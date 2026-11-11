"""Read-only adapter from verified immutable Tier-1 reports to sealed surface cells."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from klab.evaluation.post_train_triage import read_triage
from klab.evaluation.surface_review import create_surface_bundle

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMMON = (
    "prompt_adherence",
    "repetition",
    "readability_coherence",
    "overall_preference",
)
_CATEGORY_DIMENSIONS = {
    "named-character-continuity": ("entity_continuity",),
    "color-object-continuity": ("attribute_consistency",),
    "temporal-causal-continuation": ("causal_temporal_coherence",),
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _hex(value: object, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"triage report missing valid {field}")
    return value


def _source(report: dict[str, Any], run_id: str) -> dict[str, str]:
    inputs = report["inputs"]
    identity = report["identity"]
    endpoint = report["core"]["integrity"]
    if endpoint.get("status") != "PASS" or report["tier1"].get("status") != "OBSERVED":
        raise ValueError(
            f"triage run {run_id} has no verified observed Tier-1 endpoint"
        )
    manifest = _hex(inputs.get("manifest_sha256"), "manifest_sha256")
    checkpoint = _hex(
        inputs.get("generation_manifest_sha256"), "generation_manifest_sha256"
    )
    if (
        identity.get("run_id") != run_id
        or identity.get("manifest_sha256") != manifest
        or identity.get("generation_manifest_sha256") != checkpoint
        or endpoint.get("generation_manifest_sha256") != checkpoint
    ):
        raise ValueError(f"triage run {run_id} has conflicting endpoint identity")
    generation = endpoint.get("generation")
    if not isinstance(generation, str) or not re.fullmatch(
        r"step_\d+_gen_\d+", generation
    ):
        raise ValueError(f"triage run {run_id} missing checkpoint generation")
    source = {
        "kind": "klab_checkpoint",
        "run_id": run_id,
        "generation": generation,
        "manifest_sha256": manifest,
        "checkpoint_sha256": checkpoint,
        "report_sha256": _hex(identity.get("sha256"), "identity.sha256"),
    }
    # Older verified reports do not necessarily carry these artifact digests.
    # The content-addressed report and run/checkpoint hashes remain mandatory.
    for field in (
        "tokenizer_sha256",
        "source_identity_sha256",
        "effective_config_sha256",
    ):
        if field in inputs:
            source[field] = _hex(inputs[field], field)
    return source


def _decoder(settings: object, policy: str) -> tuple[dict[str, object], int]:
    if not isinstance(settings, dict) or set(settings) != {
        "temperature",
        "top_k",
        "seed",
        "max_new_tokens",
    }:
        raise ValueError("observed Tier-1 row has malformed decoder settings")
    temperature, top_k = settings["temperature"], settings["top_k"]
    seed, maximum = settings["seed"], settings["max_new_tokens"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (float, int))
        or not math.isfinite(temperature)
        or temperature < 0
        or type(top_k) is not int
        or top_k < 0
        or type(seed) is not int
        or type(maximum) is not int
        or maximum <= 0
        or (policy == "greedy" and (temperature != 0 or top_k != 0))
        or (policy == "sampled" and temperature <= 0)
    ):
        raise ValueError("observed Tier-1 row has invalid decoder settings")
    return {"temperature": temperature, "top_k": top_k, "max_new_tokens": maximum}, seed


def _rows(
    report: dict[str, Any], run_id: str
) -> dict[tuple[str, str, str, int], dict[str, object]]:
    result: dict[tuple[str, str, str, int], dict[str, object]] = {}
    prompts: dict[str, str] = {}
    for policy in ("greedy", "sampled"):
        rows = report["tier1"][policy]
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"triage run {run_id} has malformed Tier-1 row")
            if row.get("status") != "OBSERVED":
                continue
            prompt_id, prompt, response = (
                row.get("id"),
                row.get("prompt"),
                row.get("text"),
            )
            # Sampled reports omit prompt text; bind to the observed greedy row.
            if policy == "sampled":
                prompt = prompts.get(prompt_id)
            if not all(
                isinstance(value, str) and value
                for value in (prompt_id, prompt, response)
            ):
                raise ValueError(
                    f"triage run {run_id} has incomplete observed {policy} row"
                )
            if policy == "greedy":
                if prompt_id in prompts and prompts[prompt_id] != prompt:
                    raise ValueError(f"triage run {run_id} has conflicting prompt text")
                prompts[prompt_id] = prompt
            decoder, seed = _decoder(row.get("settings"), policy)
            expected = report["tier1"].get("settings", {}).get(policy)
            if policy == "greedy" and expected != row["settings"]:
                raise ValueError(
                    f"triage run {run_id} has conflicting greedy decoder settings"
                )
            if policy == "sampled" and (
                not isinstance(expected, dict)
                or any(
                    row["settings"][key] != expected.get(key)
                    for key in ("temperature", "max_new_tokens")
                )
                or row["settings"]["top_k"] not in expected.get("top_k", ())
                or seed not in expected.get("seeds", ())
            ):
                raise ValueError(
                    f"triage run {run_id} has conflicting sampled decoder settings"
                )
            key = (prompt_id, prompt, _digest(decoder), seed)
            if key in result:
                raise ValueError(f"triage run {run_id} has duplicate Tier-1 coordinate")
            result[key] = {"decoder": decoder, "response": response, "policy": policy}
    return result


def import_triage_reports(
    report_paths: Iterable[tuple[str, str | Path]],
    output_dir: str | Path,
    profile: str,
    selection_seed: int,
    presentation_seed: int,
) -> dict[str, Any]:
    """Compare only coordinates observed in two or more explicitly supplied runs.

    ``report_paths`` contains ``(run_id, runs_dir)`` pairs, not arbitrary report
    files: read_triage verifies the immutable report and its content address.
    """
    supplied = list(report_paths)
    if len(supplied) < 2:
        raise ValueError("triage import requires at least two distinct runs")
    sources: dict[
        str, tuple[dict[str, str], dict[tuple[str, str, str, int], dict[str, object]]]
    ] = {}
    artifacts: list[dict[str, object]] = []
    for run_id, directory in supplied:
        runs_dir = Path(directory)
        report = read_triage(run_id, runs_dir)
        if report is None:
            raise ValueError(f"missing immutable post-train triage report for {run_id}")
        source = _source(report, run_id)
        identity_key = _digest(source)
        if identity_key in sources or any(
            previous[0]["run_id"] == run_id for previous in sources.values()
        ):
            raise ValueError(f"duplicate triage source: {run_id}")
        folder = runs_dir / run_id / "post-train-triage"
        entries = list(folder.iterdir())
        if len(entries) != 1 or entries[0].is_symlink():
            raise ValueError(f"triage report artifact changed for {run_id}")
        raw = entries[0].read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if entries[0].name != f"{digest}.json" or json.loads(raw) != report:
            raise ValueError(f"triage report artifact changed for {run_id}")
        artifacts.append(
            {"path": str(entries[0].absolute()), "sha256": digest, "size": len(raw)}
        )
        sources[identity_key] = source, _rows(report, run_id)
    cells: list[dict[str, object]] = []
    for (left_id, (left_source, left_rows)), (
        right_id,
        (right_source, right_rows),
    ) in itertools.combinations(sorted(sources.items()), 2):
        for prompt_id, prompt, decoder_digest, seed in sorted(
            left_rows.keys() & right_rows.keys()
        ):
            key = (prompt_id, prompt, decoder_digest, seed)
            left, right = left_rows[key], right_rows[key]
            if left["policy"] != right["policy"] or left["decoder"] != right["decoder"]:
                raise ValueError("conflicting matched Tier-1 decoder coordinate")
            cells.append(
                {
                    "candidate_id": _digest(
                        [prompt_id, prompt, left["decoder"], seed, left_id, right_id]
                    ),
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": prompt_id,
                    "decoder": left["decoder"],
                    "rng_seed": seed,
                    "a": {"source": left_source, "response": left["response"]},
                    "b": {"source": right_source, "response": right["response"]},
                    "dimensions": [*_COMMON, *_CATEGORY_DIMENSIONS.get(prompt_id, ())],
                }
            )
    if not cells:
        raise ValueError(
            "no matched observed Tier-1 prompt/decoder cells across distinct verified runs"
        )
    return create_surface_bundle(
        cells,
        profile=profile,
        selection_seed=selection_seed,
        presentation_seed=presentation_seed,
        output_dir=output_dir,
        source_artifacts=artifacts,
    )
