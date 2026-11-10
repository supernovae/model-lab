"""Explicit dashboard run-store locations; no database opens or filesystem writes."""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache
from pathlib import Path

PRIMARY_STORE = "Primary run store"


def parse_run_store(value: str) -> tuple[str, str]:
    label, separator, directory = value.partition("=")
    label, directory = label.strip(), directory.strip()
    if not separator or not label or not directory:
        raise argparse.ArgumentTypeError(
            "expected LABEL=DIRECTORY with a nonempty label and path"
        )
    if label == PRIMARY_STORE:
        raise argparse.ArgumentTypeError(
            f"{PRIMARY_STORE!r} is reserved for --runs-dir"
        )
    return label, directory


def configured_stores(primary: Path, extra: list[tuple[str, str]]) -> dict[str, Path]:
    stores = {PRIMARY_STORE: primary.expanduser().resolve()}
    for label, directory in extra:
        if label in stores:
            raise ValueError(f"Duplicate run-store label: {label}")
        path = Path(directory).expanduser().resolve()
        if path in stores.values():
            raise ValueError(
                f"Run store {label!r} repeats an already configured directory"
            )
        stores[label] = path
    return stores


# Opt-in roots are deployment configuration, never inferred from the filesystem.
def discovery_roots() -> tuple[str, ...]:
    import json
    import os

    roots = json.loads(os.environ.get("SPARSELAB_DASHBOARD_DISCOVERY_ROOTS", "[]"))
    if (
        not isinstance(roots, list)
        or len(roots) > 8
        or any(
            not isinstance(root, str) or not Path(root).is_absolute() for root in roots
        )
    ):
        raise ValueError("Dashboard discovery requires at most eight absolute roots")
    return tuple(roots)


@lru_cache(maxsize=16)
def _discover(roots: tuple[str, ...], epoch: int):
    found, warnings = {}, []
    for root_text in roots:
        root = Path(root_text)
        try:
            if root.is_symlink():
                raise OSError("symlink root is not allowed")
            root = root.resolve(strict=True)
            with os.scandir(root) as entries:
                children = []
                for entry in entries:
                    if len(children) == 256:
                        raise OSError("more than 256 entries; use explicit stores")
                    children.append(entry.name)
            for name in sorted(children):
                child = root / name
                runs = child / "runs"
                db = runs / "experiments.sqlite3"
                try:
                    if any(p.is_symlink() for p in (child, runs, db)):
                        continue
                    if not child.is_dir() or not runs.is_dir() or not db.is_file():
                        continue
                    if not db.resolve(strict=True).is_relative_to(root):
                        continue
                    if not os.access(db, os.R_OK) or not os.access(
                        runs, os.R_OK | os.X_OK
                    ):
                        warnings.append(f"Unreadable run store: {runs}")
                        continue
                    found[f"Discovered: {name} [{root}]"] = runs
                except OSError as error:
                    warnings.append(f"Cannot inspect {runs}: {error}")
        except OSError as error:
            warnings.append(f"Discovery unavailable for {root_text}: {error}")
    return tuple(found.items()), tuple(warnings)


def discovered_stores(roots: tuple[str, ...]):
    """Inspect at most 256 direct children per root, once per 60-second bucket.

    No database is opened. Explicit stores remain authoritative in the caller.
    """
    return _discover(roots, int(time.monotonic() // 60))


def available_stores(primary: Path, extra: list[tuple[str, str]]):
    stores = configured_stores(primary, extra)
    discovered, warnings = discovered_stores(discovery_roots())
    for label, path in discovered:
        if path not in stores.values() and label not in stores:
            stores[label] = path
    return stores, warnings
