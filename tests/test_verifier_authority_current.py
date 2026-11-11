"""Every registered verifier kind must authorize on the current source tree.

Without this, an edit that drifts a pinned exclusion signature or pulls a
dynamic import into a verifier closure silently turns every signed reuse into
a full rehash (reason "unknown"), and only slow end-to-end tests notice.
When this fails, re-audit the reported edge before re-pinning it.
"""

from __future__ import annotations

import pytest

from klab import verifier_authority as authority


@pytest.mark.parametrize("kind", sorted(authority._CLOSURES))
def test_registered_verifier_kind_authorizes_current_source(kind: str) -> None:
    result = authority.verifier_authority(kind, 1)
    assert result


@pytest.mark.parametrize("kind", ["checkpoint", "prepared_array", "prepared_data"])
def test_memory_configuration_validators_remain_in_authority(kind: str) -> None:
    modules = authority.verifier_authority(kind, 1)["modules"]
    # These validators authenticate architecture-bound maps and hash tables.
    # Producer-only packing exclusions must not remove config semantics.
    assert "memory_tokenizer" in modules
    assert "address_hash" in modules
