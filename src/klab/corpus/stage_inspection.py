"""Inspect immutable Corpus stages using the host-local verification proof."""

import hashlib
import json
import re
from contextlib import contextmanager

from klab.canonical import canonical_digest
from klab.corpus.build_stages import (
    RECEIPT,
    StageStore,
    implementation_digests,
    verify_stage,
)
from klab.corpus.release_receipt import receipt_first, stat_index
from klab.corpus.release_verify import current_settings
from klab.safe_io import symlinked

_KEY = re.compile(r"[0-9a-f]{64}")


@contextmanager
def open_verified_stage(store: StageStore, stage: str, key: str):
    """Read a published stage by key using an unchanged, verified generation."""
    if not _KEY.fullmatch(key):
        raise ValueError("stage inspection requires a full SHA-256 computation key")
    directory = store.directory(stage, key)
    receipt_path = directory / RECEIPT
    if symlinked(receipt_path) or not receipt_path.is_file():
        raise ValueError(f"missing or unsafe retained Corpus {stage} stage: {key}")
    before, _ = stat_index([directory], strict=False)
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    if (
        receipt["key"] != key
        or receipt["stage"] != stage
        or canonical_digest(receipt["inputs"]) != key
    ):
        raise ValueError(f"retained Corpus {stage} stage identity differs")
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    identity = canonical_digest(
        [
            receipt_sha256,
            implementation_digests("stage_inspection.py", "build_stages.py"),
        ]
    )
    settings = current_settings()
    _, verified = receipt_first(
        "stage_output",
        directory,
        identity,
        [],
        settings.proof_store,
        settings.verification_mode,
        verify=lambda: verify_stage(directory, stage, key, receipt["inputs"]),
        reused=lambda: receipt,
    )
    if before != verified:
        raise ValueError(f"Corpus {stage} stage changed during verification")
    yield directory, receipt, receipt_sha256
    if stat_index([directory], strict=False)[0] != before:
        raise ValueError(f"Corpus {stage} stage changed during inspection")
