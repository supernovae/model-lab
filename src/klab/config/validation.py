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


def add_arguments(parser):
    parser.add_argument("source", type=Path)
    parser.add_argument("--kind", choices=tuple(KINDS), required=True)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=handle)


def handle(args):
    try:
        result = validate_config(args.source, args.kind)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        print(f"Valid {args.kind} schema: {args.source}")
        for limitation in result["limitations"]:
            print(limitation)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="klab config validate")
    add_arguments(parser)
    handle(parser.parse_args(argv))
