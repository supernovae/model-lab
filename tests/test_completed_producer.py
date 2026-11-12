"""Completed-build phases are keyed by every corpus module they can run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import klab
from klab.corpus.completed_preparation import _producer
from klab.verifier_authority import local_import_closure


def test_render_code_beyond_the_entry_modules_keys_completed_phases() -> None:
    producer = _producer()
    # Rendering runs these through the entry modules; a change to any of them
    # must invalidate completed phases instead of reusing stale shards.
    for module in (
        "corpus.structure_cleaning",
        "corpus.row_qualification",
        "corpus.provenance",
        "corpus.bounded_metadata",
        "corpus.streaming_split_freeze",
        "corpus.admission_inspection_cache",
    ):
        path = Path(klab.__file__).parent / (module.replace(".", "/") + ".py")
        assert producer[module] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "corpus.cli" not in producer
    assert all(name.startswith("corpus.") for name in producer)


def test_closure_follows_imports_only_inside_its_package(tmp_path: Path) -> None:
    closure = local_import_closure(("corpus.rights",), within="corpus")
    assert "corpus.rights" in closure
    assert not any(name.startswith(("data.", "training.")) for name in closure)
