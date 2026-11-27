"""Locations and stable names of the typed Corpus stage stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TypedArtifactKind:
    directory: str
    stage: str
    shard_producer: str | None = None

    @property
    def cache_shard(self) -> bool:
        return self.shard_producer is not None


KINDS = {
    "relation-index": TypedArtifactKind("relations/relation-index", "relation-index"),
    "relation-sequences": TypedArtifactKind(
        "relations/relation-sequences", "relation-sequences"
    ),
    "relation-source-shard": TypedArtifactKind(
        "relations/extraction/files/shard", "shard", "relation source extraction"
    ),
    "procedural-worlds": TypedArtifactKind(
        "mechanisms/procedural-worlds", "procedural-worlds"
    ),
    "procedural-sequences": TypedArtifactKind(
        "mechanisms/procedural-sequences", "procedural-sequences"
    ),
    "procedural-source-shard": TypedArtifactKind(
        "mechanisms/simulation/files/shard", "shard", "procedural world simulation"
    ),
    "mechanism-memberships": TypedArtifactKind(
        "mechanisms/mechanism-memberships", "mechanism-memberships"
    ),
    "command-output": TypedArtifactKind(
        "command-operators/command-output", "command-output"
    ),
}

_TYPED_PUBLISHERS = {
    tuple(Path(kind.directory).parts) for kind in KINDS.values() if not kind.cache_shard
}


def corpus_cache(root: Path) -> Path:
    return root / "cache" / "corpus"


def typed_publisher_work_root(payload: Path) -> Path | None:
    """Select the existing release-retention lease for a typed publisher."""
    if (
        payload.parts[-5:-3] == ("cache", "corpus")
        and payload.parts[-3:-1] in _TYPED_PUBLISHERS
    ):
        return payload.parents[4]
    return None
