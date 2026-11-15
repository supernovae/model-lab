"""Acquired snapshots are keyed on `declaration_sha256`; it must not drift.

The fixtures are two real source declarations (one Hugging Face dataset, one
pinned Git repository). Their digests cover only the acquisition identity; a
change to a source model or to the identity definition that moves them would
orphan every snapshot acquired under the old digest.
"""

from pathlib import Path

import pytest
import yaml

from klab.corpus.acquisition import declaration_sha256
from klab.corpus.project import SourceDeclaration

PINNED = {
    "code_python": "2df39f7b43dbea9b7b62a1d345b81044035183e0328c57ef68914982c64186e5",
    "ref_python_docs": "f0bda185e284c5fc485eb7e5528670db56a4f83575b3a62180c95f11f3dfa74b",
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_source_declaration_digest_is_unchanged(name: str) -> None:
    path = Path(__file__).parent / "fixtures/declarations" / f"{name}.yaml"
    source = SourceDeclaration.model_validate(yaml.safe_load(path.read_text()))
    assert declaration_sha256(source) == PINNED[name]
