"""Exact native command shapes accepted by a contracted attempt.

This is deliberately smaller than the public CLI. A new command or option does
not inherit a zero-model-work classification merely because its parent verb is
listed here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptCommand:
    args: tuple[str, ...]
    operation: str
    effect: str  # preparation, inspection, stage or train
    work_dir: Path | None
    options: dict[str, str | bool]
    monitor: dict[str, str] | None = None


def phase_output_paths(
    attempt_root: Path, phase: str, leaf_name: str | None = None
) -> dict[str, Path]:
    """Derive separate namespaces for one attempt leaf and its supervisors."""
    if not phase or not all(c.isalnum() or c in "-_" for c in phase):
        raise ValueError("invalid attempt phase name")
    paths = {
        "completion": attempt_root / "receipts" / f"{phase}.outer-owned.json",
        "inner_monitor": attempt_root / "logs" / f"{phase}.inner-monitor",
    }
    if leaf_name is not None:
        leaf = Path(leaf_name)
        if (
            leaf.is_absolute()
            or not leaf.parts
            or any(part in {".", ".."} for part in leaf.parts)
            or str(leaf) != leaf_name
        ):
            raise ValueError("leaf name must be a relative path beneath prep")
        paths["leaf"] = attempt_root / "prep" / leaf
    return paths


# Positionals, valued options, boolean options, effect. No abbreviated flags,
# duplicate flags, extra operands or hidden shell/Python dispatch are accepted.
_SHAPES: dict[str, tuple[int, set[str], set[str], str]] = {
    "corpus render-declaration": (
        0,
        {"--template", "--values-json", "--output", "--workspace-baseline"},
        set(),
        "preparation",
    ),
    "tokenizer train": (1, set(), set(), "preparation"),
    "corpus budget-init": (1, set(), set(), "preparation"),
    "corpus budget-status": (1, set(), set(), "inspection"),
    "corpus alias-snapshot": (1, {"--source-id", "--snapshot"}, set(), "preparation"),
    "corpus preview": (1, {"--input-root", "--output"}, set(), "preparation"),
    "corpus acquire": (
        1,
        {"--source-id", "--max-new-bytes", "--min-free-bytes"},
        {"--offline", "--plan"},
        "preparation",
    ),
    "corpus admission-draft": (
        1,
        {"--template", "--policy-document", "--output"},
        set(),
        "preparation",
    ),
    "corpus inspect-admission": (
        1,
        {
            "--draft",
            "--policy-document",
            "--selection",
            "--output",
            "--max-input-bytes",
            "--max-excerpt-bytes",
            "--max-output-bytes",
        },
        set(),
        "preparation",
    ),
    "corpus split-inventory": (1, {"--output"}, {"--json"}, "preparation"),
    "corpus freeze-splits": (
        1,
        {"--inventory", "--clusters", "--output"},
        {"--json"},
        "preparation",
    ),
    "corpus build": (1, set(), {"--offline"}, "preparation"),
    "corpus freeze": (1, set(), set(), "preparation"),
    "corpus audit": (1, set(), set(), "inspection"),
    "corpus near-duplicates": (
        1,
        {
            "--max-heldout-documents",
            "--max-heldout-shingles",
            "--max-candidates",
        },
        set(),
        "inspection",
    ),
    "corpus finalize-family-inventory": (
        1,
        {"--splits", "--output"},
        {"--json"},
        "preparation",
    ),
    "corpus audit-protected-lineage": (
        0,
        {
            "--prior-release",
            "--candidate-release",
            "--prior-inventory",
            "--candidate-inventory",
            "--profile",
            "--suite",
            "--output",
        },
        {"--json"},
        "preparation",
    ),
    "corpus measure-tokens": (
        1,
        {
            "--tokenizer",
            "--tokenizer-origin-release",
            "--family-inventory",
            "--policy",
            "--output",
            "--batch-source-bytes",
        },
        {"--json"},
        "preparation",
    ),
    "corpus materialize-mixture": (1, {"--output"}, {"--json"}, "preparation"),
    "corpus verify-mixture": (1, {"--output"}, {"--json"}, "inspection"),
    "corpus export": (
        1,
        {"--view", "--base-run-config", "--vocab-size", "--min-frequency"},
        set(),
        "preparation",
    ),
    "corpus verify-export": (1, set(), set(), "inspection"),
    "data prepared-inputs publish": (
        1,
        {"--output", "--resource-envelope", "--tokenizer-batch-source-bytes"},
        set(),
        "preparation",
    ),
    "data prepared-inputs verify": (2, set(), set(), "inspection"),
    "inspect": (1, set(), {"--json"}, "inspection"),
    "workspace preflight": (1, set(), set(), "inspection"),
    "stage": (
        1,
        {
            "--through",
            "--prepared-inputs",
            "--output",
            "--runtime",
            "--resource-envelope",
        },
        {"--cold-verify"},
        "stage",
    ),
    "train": (
        1,
        {
            "--stage-bundle",
            "--run-id",
            "--runs-dir",
            "--runtime",
            "--resource-envelope",
            "--backend",
            "--resume",
            "--extend-budget",
            "--promote",
            "--recover",
            "--stop-after-step",
        },
        {"--allow-runtime-drift"},
        "train",
    ),
    "monitor-baseline": (1, {"--output", "--seconds"}, {"--json"}, "preparation"),
    "report": (
        1,
        {"--kind", "--runs-dir", "--lab-dir", "--config"},
        {"--json"},
        "inspection",
    ),
    "checkpoint verify": (1, {"--config"}, {"--json"}, "inspection"),
    "attempt status": (0, {"--ledger"}, set(), "inspection"),
}

_REQUIRED: dict[str, set[str]] = {
    "corpus export": {"--view", "--base-run-config", "--vocab-size"},
    "corpus render-declaration": {"--template", "--values-json", "--output"},
    "corpus preview": {"--input-root", "--output"},
    "corpus alias-snapshot": {"--source-id", "--snapshot"},
    "corpus admission-draft": {"--template", "--policy-document", "--output"},
    "corpus inspect-admission": {
        "--draft",
        "--policy-document",
        "--selection",
        "--output",
        "--max-input-bytes",
        "--max-excerpt-bytes",
        "--max-output-bytes",
    },
    "corpus split-inventory": {"--output"},
    "corpus freeze-splits": {"--inventory", "--clusters", "--output"},
    "corpus finalize-family-inventory": {"--splits", "--output"},
    "corpus audit-protected-lineage": {
        "--prior-release",
        "--candidate-release",
        "--prior-inventory",
        "--candidate-inventory",
        "--profile",
        "--suite",
        "--output",
    },
    "corpus measure-tokens": {
        "--tokenizer",
        "--tokenizer-origin-release",
        "--family-inventory",
        "--policy",
        "--output",
        "--batch-source-bytes",
    },
    "corpus materialize-mixture": {"--output"},
    "corpus verify-mixture": {"--output"},
    "data prepared-inputs publish": {
        "--output",
        "--resource-envelope",
        "--tokenizer-batch-source-bytes",
    },
    "stage": {"--through", "--output"},
    "train": {"--run-id", "--runs-dir"},
    "monitor-baseline": {"--output", "--seconds"},
    "attempt status": {"--ledger"},
}


def _native_words(command: list[str]) -> tuple[list[str], Path | None]:
    if not command:
        raise ValueError("empty attempt command")
    if command[0] == "uv":
        if command[1:5] != ["run", "--locked", "--no-sync", "klab"]:
            raise ValueError("unreviewed uv invocation")
        words = command[5:]
    elif command[0] == "klab":
        words = command[1:]
    elif (
        len(command) >= 3
        and Path(command[0]).resolve() == Path(sys.executable).resolve()
        and command[1:3] == ["-m", "klab"]
    ):
        words = command[3:]
    else:
        raise ValueError("attempt requires a direct native klab invocation")
    work_dir = None
    if words[:1] == ["--work-dir"] and len(words) >= 3:
        work_dir = Path(words[1])
        if not work_dir.is_absolute() or work_dir.is_symlink():
            raise ValueError("attempt work directory must be absolute and unlinked")
        words = words[2:]
    if not words or words[0].startswith("-"):
        raise ValueError("missing native operation")
    return words, work_dir


def _options(
    words: list[str], count: int, valued: set[str], boolean: set[str]
) -> dict[str, str | bool]:
    if len(words) < count or any(word.startswith("-") for word in words[:count]):
        raise ValueError("missing native positional input")
    values: dict[str, str | bool] = {}
    index = count
    while index < len(words):
        key = words[index]
        if key in values or key not in valued | boolean:
            raise ValueError(f"unreviewed or duplicate native option: {key}")
        if key in boolean:
            values[key] = True
            index += 1
        else:
            if index + 1 >= len(words) or words[index + 1].startswith("-"):
                raise ValueError(f"missing value for {key}")
            values[key] = words[index + 1]
            index += 2
    return values


def classify_attempt_command(command: list[str], *, depth: int = 0) -> AttemptCommand:
    """Reject every dispatch shape outside the explicitly reviewed native grammar."""
    words, work_dir = _native_words(command)
    if words[0] == "monitor":
        if depth or words.count("--") != 1:
            raise ValueError("only one bounded native monitor wrapper is supported")
        separator = words.index("--")
        parsed = _options(
            words[1:separator],
            0,
            {"--policy", "--log-dir", "--workspace", "--baseline"},
            set(),
        )
        if set(parsed) != {"--policy", "--log-dir", "--workspace", "--baseline"}:
            raise ValueError("incomplete nested monitor binding")
        monitor = {key: str(value) for key, value in parsed.items()}
        if any(
            not Path(value).is_absolute() or Path(value).is_symlink()
            for value in monitor.values()
        ):
            raise ValueError("nested monitor paths must be absolute and unlinked")
        inner = classify_attempt_command(words[separator + 1 :], depth=1)
        if inner.work_dir != work_dir:
            raise ValueError("nested native work directory differs")
        if inner.effect in {"train", "stage"}:
            raise ValueError("training or staging cannot be hidden in a nested monitor")
        return AttemptCommand(
            inner.args, inner.operation, inner.effect, work_dir, inner.options, monitor
        )
    for size in (3, 2, 1):
        name = " ".join(words[:size])
        if name in _SHAPES:
            count, valued, boolean, effect = _SHAPES[name]
            options = _options(words[size:], count, valued, boolean)
            if not _REQUIRED.get(name, set()).issubset(options):
                raise ValueError("required native option is missing")
            if name == "stage" and options.get("--through") != "validate":
                raise ValueError("attempt stage must stop at validation")
            if name == "corpus acquire":
                step_options = {"--source-id", "--max-new-bytes", "--min-free-bytes"}
                required_step = {"--source-id", "--min-free-bytes"}
                if (
                    step_options.intersection(options) or "--plan" in options
                ) and not required_step.issubset(options):
                    raise ValueError(
                        "acquisition step requires source ID and free-space floor"
                    )
                if required_step.issubset(options):
                    if "--offline" in options:
                        raise ValueError("offline acquisition cannot use step limits")
                    if not str(options["--source-id"]).strip():
                        raise ValueError("acquisition step source ID is empty")
                    try:
                        max_new_bytes = (
                            int(str(options["--max-new-bytes"]))
                            if "--max-new-bytes" in options
                            else None
                        )
                        min_free_bytes = int(str(options["--min-free-bytes"]))
                    except ValueError as error:
                        raise ValueError(
                            "acquisition step limits must be integers"
                        ) from error
                    if (
                        max_new_bytes is not None and max_new_bytes <= 0
                    ) or min_free_bytes < 0:
                        raise ValueError(
                            "acquisition step limits are outside their bounds"
                        )
            return AttemptCommand(tuple(words), name, effect, work_dir, options)
    raise ValueError("unreviewed native attempt operation")
