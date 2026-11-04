"""Explicit dashboard run-store locations; no database opens or filesystem writes."""

from __future__ import annotations

import argparse
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
