"""Explicit, read-only declaration roots; no repository discovery or execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

from klab.documents import parse_document


def safe_path(base: Path, reference: str) -> Path:
    """Resolve a declaration-relative path without accepting traversal or symlinks.

    The target need not exist: availability is checked by the consumer, not by
    declaration validation. Existing components beneath the base may not be links.
    """
    if (
        not reference
        or "\\" in reference
        or "\x00" in reference
        or Path(reference).is_absolute()
        or PureWindowsPath(reference).is_absolute()
        or bool(PureWindowsPath(reference).drive)
        or any(part == ".." for part in Path(reference).parts)
        or not Path(reference).parts
        or Path(reference) == Path(".")
    ):
        raise ValueError(f"unsafe declaration-relative path: {reference!r}")
    root = base.resolve(strict=False)
    parts = Path(reference).parts
    target = root / reference
    for index in range(len(parts) + 1):
        component = root.joinpath(*parts[:index])
        if component.is_symlink():
            raise ValueError(f"symlinked declaration path: {component}")
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError(f"path escapes declaration directory: {reference!r}")
    return target


def canonical_reference(reference: str) -> str:
    """Require one portable POSIX spelling, independent of host normalization."""
    if (
        not reference
        or "\\" in reference
        or "\x00" in reference
        or PureWindowsPath(reference).drive
        or PurePosixPath(reference).is_absolute()
        or ".." in PurePosixPath(reference).parts
        or reference == "."
        or PurePosixPath(reference).as_posix() != reference
    ):
        raise ValueError(f"unsafe or ambiguous recipe reference: {reference!r}")
    return reference


@dataclass(frozen=True)
class RecipeContext:
    """An explicit root and authenticated declaration-byte snapshot.

    Only callers that opt into this context use recipe-root references. Legacy
    declaration lookup continues to discover its own repository root. Optional
    config_paths bind explicitly selected native fields to local locations; they
    do not alter the authenticated declaration snapshots or their byte identity.
    """

    root: Path
    documents: Mapping[str, bytes]
    config_paths: Mapping[str, Mapping[str, Path]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = self.root.absolute()
        for component in (root, *root.parents):
            if component.is_symlink():
                raise ValueError(f"symlinked recipe root: {component}")
        for name in self.documents:
            canonical_reference(name)
            safe_path(root, name)
        for name in self.config_paths:
            if name not in self.documents:
                raise ValueError(f"bindings for undeclared config: {name}")
        object.__setattr__(
            self,
            "config_paths",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(paths))
                    for name, paths in self.config_paths.items()
                }
            ),
        )
        object.__setattr__(self, "root", root.resolve())
        object.__setattr__(self, "documents", MappingProxyType(dict(self.documents)))

    def reference(self, source: Path, reference: str) -> Path:
        self.document_name(source)
        reference = canonical_reference(reference)
        if reference not in self.documents:
            raise ValueError(f"undeclared recipe reference: {reference!r} in {source}")
        return safe_path(self.root, reference)

    def document_name(self, path: Path) -> str:
        try:
            name = path.absolute().relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError(f"declaration outside recipe: {path}") from error
        if name not in self.documents:
            raise ValueError(f"undeclared recipe document: {path}")
        return name

    def read_document(self, path: Path) -> dict:
        name = self.document_name(path)
        try:
            text = self.documents[name].decode("utf-8")
        except UnicodeError as error:
            raise ValueError(f"invalid recipe document {path}: {error}") from error
        return parse_document(text, source=path)
