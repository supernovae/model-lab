"""Data-only preparation argument expansion for the existing native dispatcher.

No scheduling, execution, review decisions or artifact discovery happens here.
Late bindings must come from verified receipts, not guessed future identities.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from klab.training.attempt_commands import (
    classify_attempt_command,
    phase_output_paths,
)

_SLOT = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}\Z")


def load_preparation(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("unsafe preparation declaration")
    plan = json.loads(path.read_text())
    if (
        set(plan) != {"format", "defaults", "phases"}
        or plan["format"] != "klab-preparation-v1"
    ):
        raise ValueError("unsupported preparation declaration")
    if (
        not isinstance(plan["defaults"], dict)
        or not isinstance(plan["phases"], dict)
        or not plan["phases"]
    ):
        raise ValueError("invalid preparation declaration")
    leaves = []
    for label, phase in plan["phases"].items():
        if set(phase) - {"leaf", "args", "template", "template_bindings"} or not {
            "leaf",
            "args",
        } <= set(phase):
            raise ValueError("invalid preparation phase")
        paths = phase_output_paths(Path("/attempt"), label, phase["leaf"])
        if "leaf" in paths:
            leaves.append(paths["leaf"])
        if not isinstance(phase["args"], list) or not all(
            isinstance(word, str) for word in phase["args"]
        ):
            raise ValueError("invalid phase arguments")
    if len(set(leaves)) != len(leaves):
        raise ValueError("duplicate preparation leaf")
    return plan


def compile_phase(
    plan_path: Path, attempt_root: Path, label: str, bindings: dict[str, str]
) -> dict:
    """Return argv and disjoint paths; never invoke a command or write an artifact."""
    if not attempt_root.is_absolute() or attempt_root.is_symlink():
        raise ValueError("attempt root must be absolute and unlinked")
    plan = load_preparation(plan_path)
    if label not in plan["phases"]:
        raise ValueError("unknown preparation phase")
    if not isinstance(bindings, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in bindings.items()
    ):
        raise ValueError("invalid preparation bindings")
    phase = plan["phases"][label]
    paths = phase_output_paths(attempt_root, label, phase["leaf"])

    def reference(word: str) -> str:
        if word.startswith("@"):
            target = plan["phases"].get(word[1:])
            if target is None or target["leaf"] is None:
                raise ValueError("unknown preparation leaf reference")
            return str(
                phase_output_paths(attempt_root, word[1:], target["leaf"])["leaf"]
            )
        return word

    defaults = {
        key: str((plan_path.parent / value[2:]).resolve())
        if value.startswith("./")
        else value
        for key, value in plan["defaults"].items()
    }
    if "template" in phase:
        template = phase["template"]
        defaults["TEMPLATE"] = (
            reference(template)
            if template.startswith("@")
            else str((plan_path.parent / template).resolve())
        )
        defaults["VALUES_JSON"] = "{}"
    required = {match[1] for word in phase["args"] if (match := _SLOT.fullmatch(word))}
    if set(bindings) - required:
        raise ValueError("unused preparation bindings")
    values = {**defaults, **bindings}
    if required - values.keys():
        raise ValueError(
            f"missing preparation bindings: {sorted(required - values.keys())}"
        )
    if "template_bindings" in phase:
        rendered_values = json.loads(values["VALUES_JSON"])
        if not isinstance(rendered_values, dict):
            raise ValueError("invalid template bindings")
        for key, target in phase["template_bindings"].items():
            if key in rendered_values:
                raise ValueError("cannot override a phase path binding")
            rendered_values[key] = str(
                Path(reference(target)).relative_to(paths["leaf"].parent)
            )
        values["VALUES_JSON"] = json.dumps(rendered_values, sort_keys=True)
    args = [
        values[match[1]] if (match := _SLOT.fullmatch(word)) else reference(word)
        for word in phase["args"]
    ]
    if any("${" in word for word in args if word != values.get("VALUES_JSON")):
        raise ValueError("unresolved preparation binding")
    command = classify_attempt_command(["klab", *args])
    if command.effect not in {"preparation", "inspection"}:
        raise ValueError("preparation declaration cannot dispatch model work")
    if (
        "--output" in command.options
        and "leaf" in paths
        and Path(str(command.options["--output"])) != paths["leaf"]
    ):
        raise ValueError("phase output differs from declared leaf")
    return {**{key: str(value) for key, value in paths.items()}, "args": args}


def supervised_phase_command(
    compiled: dict,
    *,
    work_root: Path,
    ledger: Path,
    label: str,
    content_identity: str,
    whole_policy: Path,
    preparation_policy: Path,
    baseline: Path,
) -> list[str]:
    """Build the single zero-counter invocation of native attempt + monitor."""
    native = ["klab", "--work-dir", str(work_root)]
    return [
        *native,
        "attempt",
        "run",
        "--ledger",
        str(ledger),
        "--label",
        label,
        "--activity",
        "inspect",
        "--content-identity-sha256",
        content_identity,
        "--policy",
        str(whole_policy),
        "--baseline",
        str(baseline),
        "--workspace",
        str(work_root),
        "--completion",
        compiled["completion"],
        "--updates",
        "0",
        "--target-positions",
        "0",
        "--generation-calls",
        "0",
        "--generated-tokens",
        "0",
        "--receipt-kind",
        "none",
        "--",
        *native,
        "monitor",
        "--policy",
        str(preparation_policy),
        "--log-dir",
        compiled["inner_monitor"],
        "--workspace",
        str(work_root),
        "--baseline",
        str(baseline),
        "--",
        *native,
        *compiled["args"],
    ]
