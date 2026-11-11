"""Copy authenticated native recipe declarations, never artifacts or authority."""

import json
from pathlib import Path

from klab.documents import parse_document
from klab.experiments.recipe import (
    PortableRecipe,
    _metadata_bytes,
    snapshot_inventory,
)


def export_recipe(source: Path, destination: Path) -> dict:
    """Export existing inventory exactly; validate all input before any output.

    Exclusive destination creation refuses clobber. On an I/O interruption the
    new directory may remain partial; its descriptor is written last.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    raw = _metadata_bytes(source)
    document = parse_document(raw.decode("utf-8"), source=source)
    if "portable_preparation_version" in document:
        from klab.experiments.preparation_recipe import PreparationRecipe

        model = PreparationRecipe
    elif "portable_workflow_version" in document:
        from klab.experiments.workflow_recipe import WorkflowRecipe

        model = WorkflowRecipe
    else:
        model = PortableRecipe
    descriptor = model.model_validate(document)
    context = snapshot_inventory(source, raw, descriptor.files)
    publish_inventory(destination, source.name, raw, context.documents)
    return {
        "format": "klab-recipe-export-v1",
        "source": str(source),
        "output": str(destination / source.name),
        "files": len(context.documents) + 1,
        "executable": False,
        "limitations": [
            "inventory integrity only; run native inspection with new explicit bindings",
            "external artifacts, bindings, admissions and execution receipts are not exported unless explicitly inventoried as metadata",
        ],
    }


def publish_inventory(
    destination: Path, descriptor_name: str, raw: bytes, documents
) -> None:
    """Publish bounded validated metadata exclusively, with descriptor last."""
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked export destination: {component}")
    if not destination.parent.is_dir():
        raise ValueError("export destination parent must already exist")
    destination.mkdir()  # Never merge into an existing directory.
    for name, data in documents.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    output = destination / descriptor_name
    with output.open("xb") as stream:
        stream.write(raw)


def handle(args):
    try:
        result = export_recipe(args.source, args.output)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))
