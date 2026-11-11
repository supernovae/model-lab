"""Fail closed at model-update entry points during preparation-only attempts."""

from __future__ import annotations

import os


def require_model_runtime_allowed() -> None:
    if os.environ.get("KLAB_PREPARATION_ONLY") == "1":
        raise RuntimeError(
            "model training and staging are disabled by KLAB_PREPARATION_ONLY"
        )
