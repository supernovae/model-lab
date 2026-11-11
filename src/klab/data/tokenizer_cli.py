"""Read-only configured tokenizer verification; never acquire or fit inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from klab.config.loading import load_config
from klab.data.tokenizer import (
    SPECIAL_TOKENS,
    load_tokenizer,
    verify_run_tokenizer,
)
from klab.training.manifest import sha256_file


def verify_configured_tokenizer(config_path: Path) -> dict[str, object]:
    """Authenticate configured artifact bytes, immutable origin, and vocabulary."""
    config = load_config(config_path)
    mixture_origin = (
        config.tokenizer.origin is not None
        and config.tokenizer.origin.mixture_declaration_path is not None
    )
    dataset = (
        config.tokenizer.origin.dataset
        if config.tokenizer.origin is not None
        and config.tokenizer.origin.dataset is not None
        else config.dataset
    )
    if (
        not mixture_origin
        and dataset.corpus_release_path is None
        and dataset.source
        not in {
            "snapshot",
            "local_stories",
            "local_token_mixture",
        }
    ):
        raise ValueError(
            "read-only tokenizer origin verification requires a frozen corpus "
            "export, snapshot, or verified token mixture; unbound sources cannot "
            "be authenticated without replay"
        )
    path = config.tokenizer.path
    manifest = verify_run_tokenizer(config, verification_mode="cold")
    tokenizer = load_tokenizer(path)
    actual = tokenizer.get_vocab_size()
    if actual != config.model.vocab_size or actual != manifest.get("vocab_size"):
        raise ValueError(
            "tokenizer actual vocabulary differs from configured vocabulary"
        )
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if (
        list(special_ids.values()) != [0, 1, 2, 3]
        or manifest.get("special_ids") != special_ids
    ):
        raise ValueError(
            "tokenizer special token IDs differ from required manifest IDs"
        )
    return {
        "tokenizer": str(path),
        "sha256": manifest["sha256"],
        "manifest_sha256": sha256_file(path.with_name("tokenizer_manifest.json")),
        "vocab_size": actual,
        "special_ids": special_ids,
        "origin": {
            key: manifest[key]
            for key in (
                "source",
                "revision",
                "source_manifest_sha256",
                "corpus_export",
                "corpus_forge_bakeoff",
            )
            if key in manifest
        },
        "verification_mode": "cold",
    }


def _verify(args: argparse.Namespace) -> None:
    result = verify_configured_tokenizer(Path(args.config))
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))


def register_verify_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "verify", help="Verify a run's tokenizer without fitting."
    )
    command.add_argument(
        "config", help="RunConfig binding tokenizer, dataset and vocabulary"
    )
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_verify)


def _memory_map(args: argparse.Namespace) -> None:
    from klab.canonical import canonical_json
    from klab.derivation import derive_config
    from klab.memory_tokenizer import build_memory_token_map

    if bool(args.bind_config) != bool(args.config_output):
        raise ValueError("--bind-config and --config-output must be supplied together")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    mapping = build_memory_token_map(Path(args.tokenizer), mode=args.mode)
    receipt = None
    if args.bind_config:
        config = load_config(Path(args.bind_config))
        if sha256_file(config.tokenizer.path) != mapping.tokenizer_sha256:
            raise ValueError("bound config tokenizer differs from mapping tokenizer")
        receipt = derive_config(
            Path(args.bind_config),
            Path(args.config_output),
            {"model.memory_token_mapping": mapping.to_dict()},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json(mapping.to_dict()) + b"\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "mapping_sha256": mapping.mapping_sha256,
                "vocab_size": mapping.vocab_size,
                "memory_vocab_size": mapping.memory_vocab_size,
                "derivation": receipt,
            }
        )
    )


def register_memory_map_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "memory-map",
        help="Build a tokenizer-bound memory ID map; preserve input/output tokenization.",
    )
    command.add_argument("tokenizer")
    command.add_argument(
        "--mode",
        choices=("identity", "deepseek_v1", "deepseek_safe_v1"),
        default="identity",
    )
    command.add_argument("--output", required=True)
    command.add_argument("--bind-config")
    command.add_argument("--config-output")
    command.set_defaults(handler=_memory_map)
