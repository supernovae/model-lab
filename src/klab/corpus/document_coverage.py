"""Compact membership checks over a verified document index's row IDs."""

from __future__ import annotations

import mmap
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentCoverage:
    bits: mmap.mmap
    documents: int
    duplicate: str
    count: int = 0

    def add(self, rows: Iterable[int]) -> None:
        for row in rows:
            if not 1 <= row <= self.documents:
                raise ValueError("document replay index has an invalid row ID")
            offset, bit = divmod(row - 1, 8)
            flag = 1 << bit
            present = self.bits[offset]
            if present & flag:
                raise ValueError(self.duplicate)
            self.bits[offset] = present | flag
            self.count += 1


@contextmanager
def document_coverage(
    path: Path, documents: int, *, duplicate: str
) -> Iterator[DocumentCoverage]:
    """Mark one membership relation in a resettable file-backed bitmap."""
    with path.open("w+b") as stream:
        stream.truncate(max(1, (documents + 7) // 8))
        with mmap.mmap(stream.fileno(), 0) as bits:
            yield DocumentCoverage(bits, documents, duplicate)
