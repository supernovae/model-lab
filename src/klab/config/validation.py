"""Read-only native configuration schema validation, without runtime imports."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from klab.config.models import (
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TokenizerTrainConfig,
)
from klab.documents import parse_document

KINDS = {
    "model": ModelConfig,
    "optimizer": OptimizerConfig,
    "run": RunConfig,
    "tokenizer": TokenizerTrainConfig,
}


def validate_config(source: Path, kind: str) -> dict:
    # Use the same bounded, symlink-refusing metadata reader as portable recipes.
    from klab.experiments.recipe import _metadata_bytes

    source = Path(source).absolute()
    raw = _metadata_bytes(source)
    document = parse_document(raw.decode("utf-8"), source=source)
    config = TypeAdapter(KINDS[kind]).validate_python(document)
    return {
        "format": "klab-config-validation-v1",
        "kind": kind,
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_valid": True,
        "executable": False,
        "config": config.model_dump(mode="json"),
        "limitations": [
            "schema validation only; no artifact content, runtime or capacity verification",
            "no acquisition, training, evaluation or scientific readiness evidence",
            "a section declaration does not establish cross-section run compatibility",
        ],
    }


_SUFFIXES = {".json", ".yaml", ".yml"}
_MAX_DECLARATION_BYTES = 1024 * 1024


def declaration_kind(document: object) -> str | None:
    """Name the native schema a complete declaration is written for, if any.

    Partial sections, plans, campaigns and other documents are not recognized;
    a directory check reports them as skipped, never as valid.
    """
    if not isinstance(document, dict):
        return None
    keys = document.keys()
    if {"model", "training", "dataset", "tokenizer"} <= keys:
        return "run"
    if {"vocab_size", "output_dir", "dataset"} <= keys:
        return "tokenizer"
    if {"hidden_dim", "num_layers", "num_heads"} <= keys:
        return "model"
    if {"name", "peak"} <= keys and document["name"] in {"adamw", "adafactor"}:
        return "optimizer"
    return None


def validate_tree(root: Path) -> dict:
    """Validate every recognized declaration under ``root`` against its schema.

    This is the check a recipe repository runs to learn whether its
    declarations still load with the installed Lab.
    """
    root = Path(root).absolute()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    valid, invalid, skipped = [], [], 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            path.suffix.lower() not in _SUFFIXES
            or any(part.startswith(".") for part in relative.parts)
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        if path.stat().st_size > _MAX_DECLARATION_BYTES:
            skipped += 1
            continue
        try:
            kind = declaration_kind(
                parse_document(path.read_text(encoding="utf-8"), source=path)
            )
        except OSError, TypeError, ValueError:
            kind = None
        if kind is None:
            skipped += 1
            continue
        try:
            validate_config(path, kind)
        except (OSError, TypeError, ValueError) as error:
            invalid.append({"path": str(relative), "kind": kind, "error": str(error)})
        else:
            valid.append({"path": str(relative), "kind": kind})
    return {
        "format": "klab-config-tree-validation-v1",
        "root": str(root),
        "schema_valid": not invalid,
        "executable": False,
        "valid": valid,
        "invalid": invalid,
        "skipped_documents": skipped,
        "limitations": [
            "schema validation only; no artifact content, runtime or capacity verification",
            "only complete run, tokenizer, model and optimizer declarations are recognized",
        ],
    }


def add_arguments(parser):
    parser.add_argument(
        "source", type=Path, help="one declaration, or a directory to check"
    )
    parser.add_argument(
        "--kind", choices=tuple(KINDS), help="required for a single declaration"
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=handle)


def handle(args):
    try:
        if args.source.is_dir():
            if args.kind is not None:
                raise ValueError("--kind applies to one declaration, not a directory")
            result = validate_tree(args.source)
        elif args.kind is None:
            raise ValueError("--kind is required for a single declaration")
        else:
            result = validate_config(args.source, args.kind)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif "valid" in result:
        for row in result["valid"]:
            print(f"valid    {row['kind']:<9} {row['path']}")
        for row in result["invalid"]:
            first = row["error"].splitlines()[0]
            print(f"INVALID  {row['kind']:<9} {row['path']}: {first}")
        print(
            f"{len(result['valid'])} valid, {len(result['invalid'])} invalid, "
            f"{result['skipped_documents']} other documents skipped"
        )
    else:
        print(f"Valid {args.kind} schema: {args.source}")
        for limitation in result["limitations"]:
            print(limitation)
    if not result["schema_valid"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="klab config validate")
    add_arguments(parser)
    handle(parser.parse_args(argv))
