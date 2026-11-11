"""Independent, read-only text-kind probes for an already selected tokenizer.

Probe samples are authored fixtures, never Corpus Forge source documents or fitting data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klab.corpus.tokenizer_bakeoff import GROUPS
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file

_PROVENANCE = frozenset({"independently_authored", "synthetic_syntax"})
_SUITE_KEYS = frozenset({"schema_version", "suite_id", "samples"})
_SAMPLE_KEYS = frozenset({"id", "group", "text", "provenance"})


def load_probe_suite(suite_path: Path) -> dict[str, Any]:
    """Validate a versioned, self-contained probe suite without reading any corpus."""
    try:
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("probe suite must be valid UTF-8 JSON") from exc
    if not isinstance(suite, dict) or suite.keys() != _SUITE_KEYS:
        raise ValueError(
            "probe suite must contain only schema_version, suite_id, samples"
        )
    if type(suite["schema_version"]) is not int or suite["schema_version"] != 1:
        raise ValueError("unsupported probe suite schema_version")
    if not isinstance(suite["suite_id"], str) or not suite["suite_id"].strip():
        raise ValueError("probe suite requires a nonempty suite_id")
    try:
        suite["suite_id"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("probe suite_id is not valid UTF-8 text") from exc
    samples = suite["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("probe suite requires samples")
    seen: set[str] = set()
    groups: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.keys() != _SAMPLE_KEYS:
            raise ValueError(
                "probe sample must contain only id, group, text, provenance; "
                "source IDs are forbidden"
            )
        sample_id = sample["id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("probe sample requires a nonempty id")
        if sample_id in seen:
            raise ValueError(f"duplicate probe sample id: {sample_id}")
        seen.add(sample_id)
        group = sample["group"]
        if not isinstance(group, str) or group not in GROUPS:
            raise ValueError(f"invalid probe group: {group!r}")
        groups.add(group)
        if not isinstance(sample["text"], str) or not sample["text"].strip():
            raise ValueError(f"probe sample {sample_id} requires nonempty text")
        try:
            sample["text"].encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"probe sample {sample_id} is not valid UTF-8 text"
            ) from exc
        provenance = sample["provenance"]
        if not isinstance(provenance, str) or provenance not in _PROVENANCE:
            raise ValueError(f"invalid probe provenance for {sample_id}")
        if group == "logs" and sample["provenance"] != "synthetic_syntax":
            raise ValueError("log probes must be labeled synthetic_syntax")
    missing = set(GROUPS) - groups
    if missing:
        raise ValueError(f"probe suite missing groups: {', '.join(sorted(missing))}")
    return suite


def probe_tokenizer(tokenizer_path: Path, suite_path: Path) -> dict[str, Any]:
    """Score independent probe text with an existing tokenizer; never fit or mutate it."""
    suite = load_probe_suite(suite_path)
    tokenizer = load_tokenizer(tokenizer_path)
    by_kind: dict[str, dict[str, int | float]] = {
        group: {"samples": 0, "utf8_bytes": 0, "tokens": 0} for group in GROUPS
    }
    for sample in suite["samples"]:
        row = by_kind[sample["group"]]
        row["samples"] += 1
        row["utf8_bytes"] += len(sample["text"].encode("utf-8"))
        row["tokens"] += len(tokenizer.encode(sample["text"]).ids)
    return {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "suite_sha256": sha256_file(suite_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "per_kind": {
            group: {
                **row,
                "bytes_per_token": row["utf8_bytes"] / row["tokens"]
                if row["tokens"]
                else None,
            }
            for group, row in by_kind.items()
        },
    }
