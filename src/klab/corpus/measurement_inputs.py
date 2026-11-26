"""Lightweight input binding for Corpus measurements and their completed results."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from klab.corpus.shards import range_file_identity
from klab.training.manifest import sha256_file

TokenizerGeneration = tuple[str, tuple[int, ...]]


@dataclass(frozen=True)
class MeasurementTokenizer:
    """A tokenizer digest and the local file generation it was read from."""

    digest: str
    source: TokenizerGeneration
    implementation: str

    @classmethod
    def from_path(cls, path: Path) -> MeasurementTokenizer:
        path = Path(path).resolve(strict=True)
        source = str(path), tuple(range_file_identity(path))
        bound = cls(sha256_file(path), source, version("tokenizers"))
        bound.require_unchanged()
        return bound

    def require_unchanged(self) -> None:
        path, generation = self.source
        if tuple(range_file_identity(Path(path))) != generation:
            raise ValueError("tokenizer changed during Corpus measurement")

    def identity(self) -> dict[str, str]:
        return {"sha256": self.digest, "tokenizers": self.implementation}
