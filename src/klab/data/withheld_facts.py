"""Deterministic fact-identity splits for leakage diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    value: str

    @property
    def key(self) -> tuple[str, str]:
        return self.subject, self.relation

    def statement(self) -> str:
        return f"{self.subject}'s {self.relation} is {self.value}."

    def prompt(self) -> str:
        return f"{self.subject}'s {self.relation} is"


_FACTS = (
    Fact("Ava", "favorite color", "red"),
    Fact("Ben", "favorite color", "blue"),
    Fact("Cora", "favorite color", "green"),
    Fact("Dax", "favorite color", "gold"),
    Fact("Eli", "favorite animal", "otter"),
    Fact("Fae", "favorite animal", "panda"),
    Fact("Gus", "favorite animal", "falcon"),
    Fact("Hana", "favorite animal", "turtle"),
)


def split_facts(seed: int = 0) -> tuple[tuple[Fact, ...], tuple[Fact, ...]]:
    """Return a stable 75/25 split by whole fact identity."""
    rotated = _FACTS[seed % len(_FACTS) :] + _FACTS[: seed % len(_FACTS)]
    return rotated[:6], rotated[6:]


def training_documents(seed: int = 0) -> tuple[str, ...]:
    train, held_out = split_facts(seed)
    assert not {fact.key for fact in train} & {fact.key for fact in held_out}
    return tuple(fact.statement() for fact in train)


def evaluation_cases(seed: int = 0) -> tuple[tuple[str, str], ...]:
    _, held_out = split_facts(seed)
    return tuple((fact.prompt(), fact.value) for fact in held_out)


def diagnostic_manifest(seed: int = 0) -> dict[str, object]:
    """Return canonical split evidence without training or model artifacts."""
    train, held_out = split_facts(seed)
    payload = {
        "format_version": 1,
        "seed": seed,
        "training_statements": [fact.statement() for fact in train],
        "held_out_cases": [
            {"prompt": fact.prompt(), "expected_value": fact.value} for fact in held_out
        ],
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def write_manifest(path: Path, seed: int = 0) -> None:
    """Atomically write immutable diagnostic evidence."""
    target = path
    manifest = diagnostic_manifest(seed)
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"conflicting diagnostic manifest exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)


def verify_manifest(path: Path) -> dict[str, object]:
    """Read and verify a manifest against its digest and fixture seed."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid diagnostic manifest JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"diagnostic manifest must be an object: {path}")
    digest = manifest.pop("sha256", None)
    if not isinstance(digest, str):
        raise TypeError(f"diagnostic manifest has no SHA-256 digest: {path}")
    canonical = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if digest != actual:
        raise ValueError(f"diagnostic manifest digest mismatch: {path}")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"diagnostic manifest has invalid seed: {path}")
    expected = diagnostic_manifest(seed)
    if manifest != {key: value for key, value in expected.items() if key != "sha256"}:
        raise ValueError(f"diagnostic manifest does not match fixture: {path}")
    return {**manifest, "sha256": digest}


def audit_manifest(path: Path) -> dict[str, object]:
    """Return a concise, verified report of fixture split evidence."""
    manifest = verify_manifest(path)
    statements = manifest["training_statements"]
    cases = manifest["held_out_cases"]
    assert isinstance(statements, list)
    assert isinstance(cases, list)
    training_text = " ".join(statements)
    held_out_values_absent = all(
        isinstance(case, dict)
        and isinstance(case.get("expected_value"), str)
        and case["expected_value"] not in training_text
        for case in cases
    )
    return {
        "format_version": manifest["format_version"],
        "held_out_case_count": len(cases),
        "held_out_values_absent_from_training": held_out_values_absent,
        "seed": manifest["seed"],
        "sha256": manifest["sha256"],
        "training_statement_count": len(statements),
        "valid": True,
    }
