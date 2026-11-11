"""Sealed, local, single-reviewer pairwise comparisons (not scientific promotion)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from klab.canonical import reject_duplicate_keys

BLIND_FORMAT = "klab_surface_blind_v1"
PROVENANCE_FORMAT = "klab_surface_provenance_v1"
JUDGMENT_FORMAT = "klab_surface_judgment_v1"
DIMENSIONS = (
    "prompt_adherence",
    "entity_continuity",
    "attribute_consistency",
    "causal_temporal_coherence",
    "repetition",
    "readability_coherence",
    "overall_preference",
)
ISSUES = (
    "entity_changed",
    "attribute_changed",
    "object_changed",
    "causal_contradiction",
    "repetition",
    "nonsensical_drift",
    "premature_ending",
    "prompt_ignored",
    "other",
)
CHOICES = ("A", "B", "tie", "neither", "cannot_tell")
MAX_CASES = 10000
MAX_FILE = 32 * 1024 * 1024
_ID = re.compile(r"^[a-f0-9]{32}$")


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sealed_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError(f"missing, symlinked or oversized Surface Review file: {path}")
    return path.read_bytes()


def _load(path: Path) -> Any:
    data = _sealed_bytes(path)

    return json.loads(data, object_pairs_hook=reject_duplicate_keys), data


def _publish(path: Path, value: Any) -> None:
    """Exclusive atomic file publication, including directory fsync."""
    data = _bytes(value)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temp = tempfile.mkstemp(prefix=".surface-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)  # EEXIST rejects racing browser tabs
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.unlink(temp)


def _seed(value: int) -> int:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("seed must be a signed 64-bit integer")
    return value


def _dimension_ids(category: str) -> list[str]:
    ids = [
        "prompt_adherence",
        "repetition",
        "readability_coherence",
        "overall_preference",
    ]
    if category in ("entity_continuity", "named_character_continuity"):
        ids.append("entity_continuity")
    if category in ("object_continuity", "color_attribute_continuity"):
        ids.append("attribute_consistency")
    if category in ("cause_effect", "temporal_ordering", "location_permanence"):
        ids.append("causal_temporal_coherence")
    return ids


def create_surface_bundle(
    cells: list[dict[str, Any]],
    *,
    profile: str,
    selection_seed: int,
    presentation_seed: int,
    output_dir: str | Path,
    dimensions: list[str] | None = None,
    source_artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Select candidates before review; publish blinded and private views once."""
    if profile not in ("quick", "standard", "full"):
        raise ValueError("unknown review sample profile")
    _seed(selection_seed)
    _seed(presentation_seed)
    if not cells or len(cells) > MAX_CASES:
        raise ValueError("eligible count out of bounds")
    if dimensions is not None and (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or set(dimensions) - set(DIMENSIONS)
    ):
        raise ValueError("invalid dimension IDs")
    keys: set[str] = set()
    for cell in cells:
        if set(cell) - {
            "candidate_id",
            "prompt_id",
            "prompt",
            "category",
            "decoder",
            "rng_seed",
            "a",
            "b",
            "dimensions",
            "legacy",
        }:
            raise ValueError("unexpected candidate fields")
        key = cell["candidate_id"]
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("duplicate or invalid candidate ID")
        keys.add(key)
        if not all(
            isinstance(cell[x], str) and cell[x]
            for x in ("prompt_id", "prompt", "category")
        ):
            raise ValueError("incomplete prompt coordinate")
        if (
            not isinstance(cell["decoder"], dict)
            or cell["rng_seed"] is not None
            and type(cell["rng_seed"]) is not int
        ):
            raise ValueError("invalid decoder coordinate")
        for side in ("a", "b"):
            entry = cell[side]
            if (
                set(entry) != {"source", "response"}
                or not isinstance(entry["source"], dict)
                or not isinstance(entry["response"], str)
            ):
                raise ValueError("invalid response source")
            source = entry["source"]
            if source.get("kind") in (
                "imported_external",
                "external_local_endpoint",
            ) and any(
                not isinstance(source.get(field), str) or not source[field]
                for field in (
                    "model_uri",
                    "revision",
                    "tokenizer_sha256",
                    "output_provenance",
                )
            ):
                raise ValueError(
                    "external source requires stable URI, revision, tokenizer and output provenance"
                )
        if cell["a"]["source"] == cell["b"]["source"]:
            raise ValueError("same source on both sides")
        ids = cell.get("dimensions", dimensions or _dimension_ids(cell["category"]))
        if not ids or len(ids) != len(set(ids)) or set(ids) - set(DIMENSIONS):
            raise ValueError("invalid applicable dimensions")
    cells = sorted(
        cells,
        key=lambda c: (
            c["prompt_id"],
            _bytes(c["decoder"]),
            str(c["rng_seed"]),
            _bytes(
                sorted(
                    (
                        _bytes(c["a"]["source"]).decode(),
                        _bytes(c["b"]["source"]).decode(),
                    )
                )
            ),
            c["candidate_id"],
        ),
    )
    limit = {"quick": 12, "standard": 32, "full": len(cells)}[profile]
    remaining = list(cells)
    chosen: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    pairings: Counter[tuple[str, str]] = Counter()
    decoders: Counter[bytes] = Counter()
    rngs: Counter[str] = Counter()
    prompts: Counter[str] = Counter()

    def pairing(c: dict[str, Any]) -> tuple[str, str]:
        return tuple(
            sorted(
                (_bytes(c["a"]["source"]).decode(), _bytes(c["b"]["source"]).decode())
            )
        )  # type: ignore[return-value]

    for _ in range(min(limit, len(cells))):
        c = min(
            remaining,
            key=lambda x: (
                categories[x["category"]],
                pairings[pairing(x)],
                decoders[_bytes(x["decoder"])],
                rngs[str(x["rng_seed"])],
                prompts[x["prompt_id"]],
                _sha(
                    b"surface-selection-v1\0"
                    + str(selection_seed).encode()
                    + b"\0"
                    + x["candidate_id"].encode()
                ),
            ),
        )
        remaining.remove(c)
        chosen.append(c)
        categories[c["category"]] += 1
        pairings[pairing(c)] += 1
        decoders[_bytes(c["decoder"])] += 1
        rngs[str(c["rng_seed"])] += 1
        prompts[c["prompt_id"]] += 1
    rng = random.Random(presentation_seed)
    rng.shuffle(chosen)
    blind_cases = []
    private_cases = []
    for c in chosen:
        case_id = _sha(
            b"surface-case-v1\0"
            + str(presentation_seed).encode()
            + b"\0"
            + c["candidate_id"].encode()
        )[:32]
        flip = bool(rng.getrandbits(1))
        a, b = (c["b"], c["a"]) if flip else (c["a"], c["b"])
        blind_cases.append(
            {
                "blind_case_id": case_id,
                "prompt": c["prompt"],
                "category": c["category"],
                "responses": [
                    {"side": "A", "response": a["response"]},
                    {"side": "B", "response": b["response"]},
                ],
                "dimensions": c.get(
                    "dimensions", dimensions or _dimension_ids(c["category"])
                ),
                "issue_tags": list(ISSUES),
            }
        )
        private_cases.append(
            {
                "blind_case_id": case_id,
                "candidate_id": c["candidate_id"],
                "prompt_id": c["prompt_id"],
                "decoder": c["decoder"],
                "rng_seed": c["rng_seed"],
                "sides": {"A": a["source"], "B": b["source"]},
                "original_sides": {"a": c["a"]["source"], "b": c["b"]["source"]},
                **({"legacy": c["legacy"]} if "legacy" in c else {}),
            }
        )
    blind = {"format": BLIND_FORMAT, "version": 1, "cases": blind_cases}
    provenance = {
        "format": PROVENANCE_FORMAT,
        "version": 1,
        "profile": profile,
        "selection_seed": selection_seed,
        "presentation_seed": presentation_seed,
        "eligible_keys": [c["candidate_id"] for c in cells],
        "selected_keys": [c["candidate_id"] for c in chosen],
        "source_artifacts": list(source_artifacts),
        "cases": private_cases,
        "prompt_set_digest": _sha(
            _bytes(
                sorted({(c["prompt_id"], c["prompt"], c["category"]) for c in cells})
            )
        ),
    }
    manifest = {
        "format": "klab_surface_manifest_v1",
        "version": 1,
        "eligible_count": len(cells),
        "selected_count": len(chosen),
        "blind_sha256": _sha(_bytes(blind)),
        "provenance_sha256": _sha(_bytes(provenance)),
    }
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    _publish(folder / "blind.json", blind)
    _publish(folder / "provenance.json", provenance)
    _publish(folder / "manifest.json", manifest)
    open_surface_bundle(folder, private=True)
    return manifest


def open_surface_bundle(
    bundle_dir: str | Path, *, private: bool = False
) -> dict[str, Any]:
    """Validate exact sealed bytes and all structural limits before returning a view."""
    folder = Path(bundle_dir)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("missing or symlinked bundle")
    manifest, manifest_bytes = _load(folder / "manifest.json")
    blind, blind_bytes = _load(folder / "blind.json")
    # The review path checks private bytes against the seal but does not parse
    # identities into its in-memory view until an explicit revealed operation.
    private_bytes = _sealed_bytes(folder / "provenance.json")
    if any(
        _bytes(obj) != data
        for obj, data in ((manifest, manifest_bytes), (blind, blind_bytes))
    ):
        raise ValueError("noncanonical sealed JSON")
    if (
        set(manifest)
        != {
            "format",
            "version",
            "eligible_count",
            "selected_count",
            "blind_sha256",
            "provenance_sha256",
        }
        or manifest["format"] != "klab_surface_manifest_v1"
        or manifest["version"] != 1
    ):
        raise ValueError("invalid Surface Review manifest")
    if manifest["blind_sha256"] != _sha(blind_bytes) or manifest[
        "provenance_sha256"
    ] != _sha(private_bytes):
        raise ValueError("changed sealed bundle")
    if (
        set(blind) != {"format", "version", "cases"}
        or blind["format"] != BLIND_FORMAT
        or blind["version"] != 1
    ):
        raise ValueError("invalid blinded format")
    count = manifest["selected_count"]
    if (
        type(count) is not int
        or type(manifest["eligible_count"]) is not int
        or not 1 <= count <= manifest["eligible_count"] <= MAX_CASES
        or len(blind["cases"]) != count
    ):
        raise ValueError("invalid sealed case counts")
    ids = set()
    for visible in blind["cases"]:
        if set(visible) != {
            "blind_case_id",
            "prompt",
            "category",
            "responses",
            "dimensions",
            "issue_tags",
        }:
            raise ValueError("invalid sealed case shape")
        cid = visible["blind_case_id"]
        if not isinstance(cid, str) or not _ID.fullmatch(cid) or cid in ids:
            raise ValueError("invalid or duplicate blinded case")
        ids.add(cid)
        if (
            not isinstance(visible["prompt"], str)
            or not isinstance(visible["category"], str)
            or not isinstance(visible["responses"], list)
            or len(visible["responses"]) != 2
            or any(
                set(r) != {"side", "response"}
                or r["side"] != side
                or not isinstance(r["response"], str)
                for r, side in zip(visible["responses"], ("A", "B"), strict=True)
            )
            or not isinstance(visible["dimensions"], list)
            or not visible["dimensions"]
            or len(set(visible["dimensions"])) != len(visible["dimensions"])
            or set(visible["dimensions"]) - set(DIMENSIONS)
            or visible["issue_tags"] != list(ISSUES)
        ):
            raise ValueError("invalid blinded response or dimensions")
    if not private:
        return {"manifest": manifest, "blind": blind}
    provenance, parsed_bytes = _load(folder / "provenance.json")
    if (
        parsed_bytes != private_bytes
        or _bytes(provenance) != private_bytes
        or set(provenance)
        != {
            "format",
            "version",
            "profile",
            "selection_seed",
            "presentation_seed",
            "eligible_keys",
            "selected_keys",
            "source_artifacts",
            "cases",
            "prompt_set_digest",
        }
        or provenance["format"] != PROVENANCE_FORMAT
        or provenance["version"] != 1
    ):
        raise ValueError("invalid private format")
    if (
        len(provenance["cases"]) != count
        or len(provenance["selected_keys"]) != count
        or len(provenance["eligible_keys"]) != manifest["eligible_count"]
    ):
        raise ValueError("invalid private case counts")
    for visible, hidden in zip(blind["cases"], provenance["cases"], strict=True):
        required = {
            "blind_case_id",
            "candidate_id",
            "prompt_id",
            "decoder",
            "rng_seed",
            "sides",
            "original_sides",
        }
        if (
            not isinstance(hidden, dict)
            or not required <= set(hidden)
            or set(hidden) - required - {"legacy"}
            or hidden["blind_case_id"] != visible["blind_case_id"]
            or set(hidden["sides"]) != {"A", "B"}
        ):
            raise ValueError("invalid private case mapping")
    return {"manifest": manifest, "blind": blind, "provenance": provenance}


def read_surface_answers(bundle_dir: str | Path) -> dict[str, dict[str, Any]]:
    view = open_surface_bundle(bundle_dir)
    expected = {c["blind_case_id"]: c for c in view["blind"]["cases"]}
    folder = Path(bundle_dir) / "answers"
    if folder.is_symlink():
        raise ValueError("symlinked answers directory")
    if not folder.exists():
        return {}
    if not folder.is_dir():
        raise ValueError("invalid answers directory")
    answers = {}
    for path in folder.iterdir():
        if not path.name.endswith(".json") or path.stem not in expected:
            raise ValueError(f"unexpected answer state: {path}")
        answer, data = _load(path)
        if (
            _bytes(answer) != data
            or set(answer)
            != {
                "format",
                "blind_sha256",
                "blind_case_id",
                "choices",
                "issue_tags",
                "note",
                "timestamp_utc",
                "digest",
            }
            or answer["format"] != JUDGMENT_FORMAT
            or answer["blind_sha256"] != view["manifest"]["blind_sha256"]
            or answer["blind_case_id"] != path.stem
            or answer["digest"]
            != _sha(
                _bytes({key: value for key, value in answer.items() if key != "digest"})
            )
        ):
            raise ValueError(f"invalid answer digest/shape: {path}")
        _validate_vote(
            expected[path.stem],
            answer["choices"],
            answer["issue_tags"],
            answer["note"],
            answer["timestamp_utc"],
        )
        answers[path.stem] = answer
    return answers


def _validate_vote(
    case: dict[str, Any], choices: Any, issue_tags: Any, note: Any, timestamp: Any
) -> None:
    if (
        not isinstance(choices, dict)
        or set(choices) != set(case["dimensions"])
        or any(choice not in CHOICES for choice in choices.values())
    ):
        raise ValueError("vote must choose every applicable dimension exactly once")
    if (
        not isinstance(issue_tags, list)
        or len(issue_tags) != len(set(issue_tags))
        or set(issue_tags) - set(ISSUES)
    ):
        raise ValueError("invalid issue tags")
    if not isinstance(note, str) or len(note) > 4000:
        raise ValueError("invalid note")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ValueError("timestamp must be UTC ISO-8601")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("invalid UTC timestamp") from exc


def record_surface_judgment(
    bundle_dir: str | Path,
    blind_case_id: str,
    choices: dict[str, str],
    issue_tags: list[str],
    timestamp_utc: str,
    note: str = "",
) -> dict[str, Any]:
    view = open_surface_bundle(bundle_dir)
    if (Path(bundle_dir) / "review.json").exists():
        raise ValueError("review already completed")
    cases = {c["blind_case_id"]: c for c in view["blind"]["cases"]}
    if blind_case_id not in cases:
        raise ValueError("unknown blinded case")
    read_surface_answers(bundle_dir)
    _validate_vote(cases[blind_case_id], choices, issue_tags, note, timestamp_utc)
    folder = Path(bundle_dir) / "answers"
    folder.mkdir(exist_ok=True)
    body = {
        "format": JUDGMENT_FORMAT,
        "blind_sha256": view["manifest"]["blind_sha256"],
        "blind_case_id": blind_case_id,
        "choices": choices,
        "issue_tags": issue_tags,
        "note": note,
        "timestamp_utc": timestamp_utc,
    }
    answer = {**body, "digest": _sha(_bytes(body))}
    _publish(folder / f"{blind_case_id}.json", answer)
    return answer


def _review(bundle_dir: str | Path) -> dict[str, Any]:
    view = open_surface_bundle(bundle_dir)
    review, data = _load(Path(bundle_dir) / "review.json")
    if (
        _bytes(review) != data
        or set(review)
        != {
            "format",
            "reviewer_mode",
            "version",
            "blind_sha256",
            "judgments",
            "reveal_state",
            "digest",
        }
        or review["format"] != "klab_surface_review_v1"
        or review["reviewer_mode"] != "single_reviewer_self_blind"
        or review["version"] != 1
        or review["blind_sha256"] != view["manifest"]["blind_sha256"]
        or review["reveal_state"] is not False
        or _sha(_bytes({k: v for k, v in review.items() if k != "digest"}))
        != review["digest"]
    ):
        raise ValueError("changed or invalid completed review")
    answers = read_surface_answers(bundle_dir)
    if review["judgments"] != [
        answers[c["blind_case_id"]] for c in view["blind"]["cases"]
    ] or len(answers) != len(view["blind"]["cases"]):
        raise ValueError("changed answers since review completion")
    return review


def complete_surface_review(bundle_dir: str | Path) -> dict[str, Any]:
    view = open_surface_bundle(bundle_dir)
    answers = read_surface_answers(bundle_dir)
    cases = view["blind"]["cases"]
    if len(answers) != len(cases):
        raise ValueError("every case must be voted before completion")
    body = {
        "format": "klab_surface_review_v1",
        "version": 1,
        "reviewer_mode": "single_reviewer_self_blind",
        "blind_sha256": view["manifest"]["blind_sha256"],
        "judgments": [answers[c["blind_case_id"]] for c in cases],
        "reveal_state": False,
    }
    review = {**body, "digest": _sha(_bytes(body))}
    _publish(Path(bundle_dir) / "review.json", review)
    return review


def reveal_surface_review(bundle_dir: str | Path) -> dict[str, Any]:
    review = _review(bundle_dir)
    view = open_surface_bundle(bundle_dir, private=True)
    body = {
        "format": "klab_surface_reveal_v1",
        "version": 1,
        "reveal_state": True,
        "review_digest": review["digest"],
        "provenance_sha256": view["manifest"]["provenance_sha256"],
        "sides": [
            {"blind_case_id": c["blind_case_id"], "sides": c["sides"]}
            for c in view["provenance"]["cases"]
        ],
    }
    result = {**body, "digest": _sha(_bytes(body))}
    _publish(Path(bundle_dir) / "reveal.json", result)
    return result


def surface_results(bundle_dir: str | Path) -> dict[str, Any]:
    review = _review(bundle_dir)
    view = open_surface_bundle(bundle_dir, private=True)
    reveal, data = _load(Path(bundle_dir) / "reveal.json")
    if (
        _bytes(reveal) != data
        or set(reveal)
        != {
            "format",
            "version",
            "reveal_state",
            "review_digest",
            "provenance_sha256",
            "sides",
            "digest",
        }
        or reveal["format"] != "klab_surface_reveal_v1"
        or reveal["version"] != 1
        or reveal["reveal_state"] is not True
        or reveal["review_digest"] != review["digest"]
        or reveal["provenance_sha256"] != view["manifest"]["provenance_sha256"]
        or reveal["sides"]
        != [
            {"blind_case_id": c["blind_case_id"], "sides": c["sides"]}
            for c in view["provenance"]["cases"]
        ]
        or reveal["digest"]
        != _sha(_bytes({k: v for k, v in reveal.items() if k != "digest"}))
    ):
        raise ValueError("changed or invalid reveal")
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for judgment, case in zip(
        review["judgments"], view["provenance"]["cases"], strict=True
    ):
        sources = case["sides"]
        pair = " vs ".join(
            sorted((_bytes(sources["A"]).decode(), _bytes(sources["B"]).decode()))
        )
        for dimension, choice in judgment["choices"].items():
            key = _bytes(sources[choice]).decode() if choice in ("A", "B") else choice
            counts[pair][dimension][key] += 1
    return {
        pair: {dimension: dict(votes) for dimension, votes in dims.items()}
        for pair, dims in counts.items()
    }
