"""Lab ships no downstream project names and no former product name."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Built from fragments so this file does not contain the names it forbids.
FORBIDDEN = re.compile(
    "|".join(("dev" + "mind", "tiny" + "stories", "sparse" + "lab")), re.IGNORECASE
)
# Tracked files that may keep a forbidden name. Empty: third-party names that
# must survive upstream (a model or dataset id) are not on the forbidden list.
ALLOWLIST: frozenset[str] = frozenset()


def test_no_legacy_project_names_in_tracked_files() -> None:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        pytest.skip("not a git checkout")
    hits = []
    for name in listed.decode().split("\0"):
        path = ROOT / name
        if not name or name in ALLOWLIST or not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        if FORBIDDEN.search(text) or FORBIDDEN.search(name):
            hits.append(name)
    assert not hits, f"legacy names in: {sorted(hits)}"
