"""Acquired snapshots are keyed on `declaration_sha256`; it must not drift.

The fixtures are two real source declarations (one Hugging Face dataset, one
pinned Git repository). Their digests were recorded from the code before the
normalizer options and the removal of `generator_version`; a change to a source
model that moves them would orphan every snapshot acquired under the old digest.
"""

from pathlib import Path

import pytest
import yaml

from klab.corpus.acquisition import declaration_sha256
from klab.corpus.project import SourceDeclaration

PINNED = {
    "code_python": "4da756eca7abe707de55a903bace022b786bfffbb85f7e9c88b84f14694fb6c4",
    "ref_python_docs": "ddce7d3134ba386e0f1937d2604f5de1664f8507d11929495224a585c5078d15",
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_source_declaration_digest_is_unchanged(name: str) -> None:
    path = Path(__file__).parent / "fixtures/declarations" / f"{name}.yaml"
    source = SourceDeclaration.model_validate(yaml.safe_load(path.read_text()))
    assert declaration_sha256(source) == PINNED[name]
