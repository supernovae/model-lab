"""Native artifact location resolution without opening payloads."""

from pathlib import Path


def artifact_path(reference: str, source: Path) -> Path:
    declared = Path(reference)
    if ".." in declared.parts:
        raise ValueError("artifact path cannot contain parent traversal")
    target = declared if declared.is_absolute() else source.absolute().parent / declared
    return target.absolute()
