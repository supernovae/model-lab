"""Deterministic memory artifacts and recipient inputs for portability campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from klab.address_hash import token_addressing_marker, token_ngram_address
from klab.canonical import canonical_digest, canonical_json
from klab.data.allocation import (
    OWNER_LEXICAL,
    OWNER_NEURAL,
    OWNER_SEMANTIC,
    build_allocation_manifest,
)
from klab.data.byte_hash import table_address
from klab.data.conversations import iter_rendered_conversations
from klab.data.portability_worlds import _write_jsonl
from klab.data.tokenizer import load_tokenizer
from klab.data.toy_worlds import (
    FrozenStructuredKeyEncoder,
    FrozenValueEncoder,
    ProducerFact,
    StructuredKey,
    _write_pack_inputs,
    frozen_encoders,
)
from klab.engram.packs import _rename_noreplace, compile_pack, verify_pack
from klab.engram.semantic import SemanticRetriever
from klab.evaluation.chat import format_chat_prompt
from klab.model.portable_engram import export_portable_engram
from klab.training.manifest import sha256_file, source_identity

_ASSET_FORMAT = "klab-portability-memory-assets"
_ASSET_VERSION = 1
_TABLE_SIZE = 8192
_VALUE_DIM = 8
_TOKEN_ORDER = 32
_BYTE_ORDER = 32
_SYMBOLS = "ABCDEFGHIJKLMNOP"
_CREATED_AT = "2026-09-25T00:00:00Z"


def _file_descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_existing_assets(root: Path, world_digest: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise FileExistsError(
            f"portability asset root is not a regular directory: {root}"
        )
    manifest_path = root / "memory_assets.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileExistsError(f"portability asset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.get("sha256")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if (
        manifest.get("format") != _ASSET_FORMAT
        or manifest.get("version") != _ASSET_VERSION
        or manifest.get("world_manifest_sha256") != world_digest
        or canonical_digest(body) != digest
    ):
        raise FileExistsError(
            "existing portability memory assets bind different inputs"
        )
    expected: set[str] = set()
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise FileExistsError(
                "existing portability memory asset inventory is invalid"
            )
        relative = entry["path"]
        if not isinstance(relative, str) or relative in expected:
            raise FileExistsError(
                "existing portability memory asset path is duplicated"
            )
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size_bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise FileExistsError(
                f"existing portability memory asset failed verification: {relative}"
            )
        expected.add(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if actual != expected:
        raise FileExistsError("existing portability memory asset inventory differs")
    return manifest_path


def _load_fact_rows(world_root: Path, world_id: str) -> list[dict[str, Any]]:
    path = world_root / "producers" / f"{world_id}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(row.get("world_id") != world_id for row in rows):
        raise ValueError(f"producer rows contain a foreign world id: {world_id}")
    return rows


def _fact(row: dict[str, Any], value: str | None = None) -> ProducerFact:
    return ProducerFact(
        str(row["id"]),
        str(row["world_id"]),
        StructuredKey(str(row["subject"]), str(row["relation"])),
        str(row["value"] if value is None else value),
        row.get("valid_from"),
        row.get("valid_until"),
    )


def _controlled_value(fact: ProducerFact, condition: str) -> str:
    if condition == "real":
        return fact.value
    if condition == "corrupt":
        if fact.value not in _SYMBOLS:
            return fact.value
        return _SYMBOLS[(_SYMBOLS.index(fact.value) + 1) % len(_SYMBOLS)]
    if condition == "random":
        key = f"portability-random-v1|{fact.key.subject}|{fact.key.relation}"
        index = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        return _SYMBOLS[index % len(_SYMBOLS)]
    raise ValueError(f"unsupported memory control: {condition}")


def _key_prompt(fact: ProducerFact) -> tuple[str, str]:
    subject = fact.key.subject
    _, node_nonce = subject.split(".", 1)
    node = node_nonce.rsplit("~", 1)[0]
    lookup = f"{subject}.{fact.key.relation}"
    encoded = lookup.encode("ascii")
    if len(encoded) > 32:
        raise ValueError(f"structured key exceeds the fixed lookup suffix: {lookup}")
    suffix = "|" * (32 - len(encoded)) + lookup
    prompt = (
        f"Recipient adapter training association: report {node}'s supplied "
        f"{fact.key.relation} value.\n{suffix}"
    )
    return prompt, suffix


def _token_address(token_ids: list[int], order: int, table_size: int) -> int:
    key = (
        token_ids[-1 - offset] if offset < len(token_ids) else 0
        for offset in range(order)
    )
    return token_ngram_address(key, table_size, 0)


def _memory_artifact(
    root: Path,
    *,
    representation: str,
    scope: str,
    condition: str,
    facts: list[ProducerFact],
    tokenizer: Any,
    tokenizer_sha256: str,
    value_encoder: FrozenValueEncoder,
) -> tuple[Path, str, dict[str, object], dict[str, object]]:
    table = torch.zeros((_TABLE_SIZE, _VALUE_DIM), dtype=torch.float32)
    address_owner: dict[int, str] = {}
    colliding_facts: list[dict[str, str]] = []
    fact_rows: list[dict[str, object]] = []
    for fact in facts:
        controlled = _controlled_value(fact, condition)
        prompt, suffix = _key_prompt(fact)
        rendered = format_chat_prompt([], prompt)
        if representation == "token":
            token_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            address = _token_address(token_ids, _TOKEN_ORDER, _TABLE_SIZE)
        elif representation == "byte":
            address = table_address(
                rendered.encode("utf-8")[-_BYTE_ORDER:], _TABLE_SIZE
            )
        else:
            raise ValueError(f"unsupported lexical representation: {representation}")
        key_id = f"{fact.key.subject}|{fact.key.relation}"
        prior = address_owner.get(address)
        if prior is not None and prior != key_id:
            colliding_facts.append(
                {"address": str(address), "first_key": prior, "second_key": key_id}
            )
        address_owner[address] = key_id
        vector = torch.from_numpy(value_encoder.encode(controlled).copy())
        if vector.shape != (_VALUE_DIM,):
            raise ValueError(
                "value encoder dimension differs from portability table width"
            )
        table[address] = vector
        fact_rows.append(
            {
                "fact_id": fact.id,
                "key": key_id,
                "value": controlled,
                "address": address,
                "lookup_suffix": suffix,
            }
        )
    if colliding_facts:
        raise ValueError(
            f"{representation} address collisions between distinct structured keys: "
            f"{colliding_facts[:8]}"
        )
    path = root / "memory" / representation / condition / f"{scope}."
    if representation == "token":
        artifact = path.with_suffix(".safetensors")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_file({"memory.table.weight": table.contiguous()}, str(artifact))
        addressing = {
            "tokenizer_sha256": tokenizer_sha256,
            "order": _TOKEN_ORDER,
            "hash_heads": 1,
            "rows": _TABLE_SIZE,
            "embedding_dim": _VALUE_DIM,
            **token_addressing_marker(_TABLE_SIZE, 1),
        }
        tensor_sha256 = hashlib.sha256(table.numpy().tobytes()).hexdigest()
    else:
        artifact = path.with_suffix(".enbyte")
        export_portable_engram(table, artifact, ngram_size=_BYTE_ORDER)
        addressing = {
            "format_version": 1,
            "normalization": "raw-utf8-v1",
            "hashing": "poly257-terminal-v1",
            "ngram_size": _BYTE_ORDER,
            "table_size": _TABLE_SIZE,
            "embedding_dim": _VALUE_DIM,
        }
        tensor_sha256 = hashlib.sha256(table.numpy().tobytes()).hexdigest()
    details = {
        "representation": representation,
        "scope": scope,
        "control": condition,
        "table_size": _TABLE_SIZE,
        "value_dimension": _VALUE_DIM,
        "record_count": len(facts),
        "distinct_addresses": len(address_owner),
        "duplicate_key_address_count": len(facts) - len(address_owner),
        "cross_key_collision_count": len(colliding_facts),
        "addressing": addressing,
        "records": fact_rows,
    }
    return artifact, tensor_sha256, addressing, details


def _compile_semantic_pack(
    root: Path,
    *,
    scope: str,
    condition: str,
    facts: list[ProducerFact],
    key_encoder: FrozenStructuredKeyEncoder,
    value_encoder: FrozenValueEncoder,
) -> Path:
    destination = root / "memory" / "semantic" / condition / f"{scope}.enpack"
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_root = root / ".pack-inputs" / f"{condition}-{scope}"
    source, keys, values, metadata = _write_pack_inputs(
        input_root, facts, key_encoder, value_encoder, random_keys=False
    )
    compile_pack(
        source,
        destination,
        name=f"engram-portability-{condition}-{scope}",
        namespace="engram-portability-v1",
        default_license="CC0-1.0",
        source_name="generated",
        source_revision="controlled-memory-compiler-v1",
        created_at=_CREATED_AT,
        semantic_keys=keys,
        semantic_values=values,
        semantic_metadata=metadata,
    )
    shutil.rmtree(input_root)
    report = verify_pack(destination)
    if not report.valid:
        raise ValueError(
            f"compiled portability pack failed verification: {scope}/{condition}"
        )
    return destination


def _world_packs(world_root: Path, world_manifest: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    inventory = world_manifest["semantic_packs"]
    for scope, descriptor in inventory.items():
        result[scope] = world_root / descriptor["path"]
    return result


def _training_facts(
    world_root: Path, world_manifest: dict[str, Any]
) -> list[ProducerFact]:
    eligible = set(world_manifest["recipient_adapter_fact_ids"])
    result = [
        _fact(row)
        for world in world_manifest["worlds"]
        if world["partition"] == "train"
        for row in _load_fact_rows(world_root, world["world_id"])
        if row["id"] in eligible
    ]
    if {fact.id for fact in result} != eligible:
        raise ValueError(
            "training producer rows differ from eligible recipient fact inventory"
        )
    return result


def build_portability_memory_assets(
    world_manifest_path: Path, asset_root: Path
) -> Path:
    """Compile frozen lexical and semantic artifacts from verified generated worlds.

    Token and byte artifacts are deterministic direct compiles of the declared
    structured fact records; they are not source-model-trained tables. Semantic
    artifacts are verified EngramPacks. Replacement assignments are compiled
    only as immutable evaluation attachments and never enter recipient training.
    """
    world_manifest_path = Path(world_manifest_path)
    world_root = world_manifest_path.parent
    world_manifest = json.loads(world_manifest_path.read_text(encoding="utf-8"))
    expected = world_manifest.get("sha256")
    body = {key: value for key, value in world_manifest.items() if key != "sha256"}
    if canonical_digest(body) != expected:
        raise ValueError("portability world manifest hash mismatch")
    if asset_root.exists() or asset_root.is_symlink():
        _verify_existing_assets(asset_root, str(expected))
        return asset_root / "memory_assets.json"
    asset_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{asset_root.name}.memory-assets-", dir=asset_root.parent
        )
    )
    try:
        tokenizer_path = world_root / world_manifest["tokenizer"]["path"]
        tokenizer = load_tokenizer(tokenizer_path)
        key_encoder, value_encoder = frozen_encoders()
        train_facts = _training_facts(world_root, world_manifest)
        train_ids = set(world_manifest["recipient_adapter_fact_ids"])
        world_facts: dict[str, list[ProducerFact]] = {
            world["world_id"]: _facts_for_world(world_root, world["world_id"])
            for world in world_manifest["worlds"]
            if world["partition"] != "train"
        }
        outputs: dict[str, dict[str, dict[str, Any]]] = {
            representation: {
                condition: {} for condition in ("real", "random", "corrupt")
            }
            for representation in ("token", "byte", "semantic")
        }
        address_rows: list[dict[str, object]] = []
        for representation in ("token", "byte", "semantic"):
            for condition in ("real", "random", "corrupt"):
                scopes = {"training": train_facts}
                scopes.update(
                    {
                        world_id: facts
                        for world_id, facts in world_facts.items()
                        if world_id != "training"
                    }
                )
                for scope, facts in scopes.items():
                    selected = [
                        fact
                        for fact in facts
                        if scope != "training" or fact.id in train_ids
                    ]
                    controlled = [
                        ProducerFact(
                            fact.id,
                            fact.world_id,
                            fact.key,
                            _controlled_value(fact, condition),
                            fact.valid_from,
                            fact.valid_until,
                        )
                        for fact in selected
                    ]
                    if representation == "semantic":
                        if condition == "real":
                            source = _world_packs(world_root, world_manifest)[scope]
                            artifact = (
                                staging
                                / "memory"
                                / "semantic"
                                / condition
                                / f"{scope}.enpack"
                            )
                            artifact.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(source, artifact)
                        else:
                            artifact = _compile_semantic_pack(
                                staging,
                                scope=scope,
                                condition=condition,
                                facts=controlled,
                                key_encoder=key_encoder,
                                value_encoder=value_encoder,
                            )
                        pack_report = verify_pack(artifact)
                        if not pack_report.valid:
                            raise ValueError(
                                f"portability pack failed verification: {artifact}"
                            )
                        retriever = SemanticRetriever.from_pack(
                            artifact, expected_pack_id=pack_report.pack_id
                        )
                        outputs[representation][condition][scope] = {
                            "artifact": artifact,
                            "pack_id": pack_report.pack_id,
                            "tensor_sha256": None,
                            "addressing": None,
                            "encoder_contract": {
                                "key_encoder": retriever.key_encoder.model_dump(
                                    mode="json"
                                ),
                                "value_encoder": retriever.value_encoder.model_dump(
                                    mode="json"
                                ),
                                "key_dim": retriever.key_dim,
                                "value_dim": retriever.memory_dim,
                                "normalization": retriever.key_normalization,
                                "representation_space_id": retriever.space_id,
                            },
                        }
                    else:
                        artifact, tensor_sha256, addressing, details = _memory_artifact(
                            staging,
                            representation=representation,
                            scope=scope,
                            condition=condition,
                            facts=controlled,
                            tokenizer=tokenizer,
                            tokenizer_sha256=sha256_file(tokenizer_path),
                            value_encoder=value_encoder,
                        )
                        outputs[representation][condition][scope] = {
                            "artifact": artifact,
                            "pack_id": None,
                            "tensor_sha256": tensor_sha256,
                            "addressing": addressing,
                            "encoder_contract": None,
                        }
                        address_rows.append(details)
        address_manifest = {
            "format": "klab-portability-address-observations",
            "version": 1,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "addressing": {
                "token": "causal-token-polynomial-v1",
                "byte": "raw-utf8-poly257-terminal-v1",
                "token_order": _TOKEN_ORDER,
                "byte_order": _BYTE_ORDER,
                "table_size": _TABLE_SIZE,
            },
            "records": address_rows,
        }
        (staging / "address_observations.json").write_bytes(
            canonical_json(address_manifest) + b"\n"
        )
        files = [
            _file_descriptor(staging, path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest_body = {
            "format": _ASSET_FORMAT,
            "version": _ASSET_VERSION,
            "world_manifest_sha256": expected,
            "compiler": "direct-structured-memory-compiler-v1",
            "table_size": _TABLE_SIZE,
            "value_dimension": _VALUE_DIM,
            "token_order": _TOKEN_ORDER,
            "byte_order": _BYTE_ORDER,
            "representations": {
                representation: {
                    condition: {
                        scope: {
                            **{
                                key: value
                                for key, value in descriptor.items()
                                if key != "artifact"
                            },
                            "artifact": _file_descriptor(
                                staging, descriptor["artifact"]
                            )
                            if descriptor["artifact"].is_file()
                            else {
                                "path": descriptor["artifact"]
                                .relative_to(staging)
                                .as_posix(),
                                "files": [
                                    _file_descriptor(descriptor["artifact"], member)
                                    for member in sorted(
                                        descriptor["artifact"].rglob("*")
                                    )
                                    if member.is_file()
                                ],
                            },
                        }
                        for scope, descriptor in by_scope.items()
                    }
                    for condition, by_scope in by_condition.items()
                }
                for representation, by_condition in outputs.items()
            },
            "files": files,
            "address_observations": "address_observations.json",
        }
        digest = canonical_digest(manifest_body)
        (staging / "memory_assets.json").write_bytes(
            canonical_json({**manifest_body, "sha256": digest}) + b"\n"
        )
        _rename_noreplace(staging, asset_root)
        return asset_root / "memory_assets.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _facts_for_world(world_root: Path, world_id: str) -> list[ProducerFact]:
    return [_fact(row) for row in _load_fact_rows(world_root, world_id)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def prepare_recipient_initialization_split(
    world_root: Path, output_root: Path
) -> tuple[Path, Path]:
    """Split independent preparation prompts into disjoint initialization data."""
    rows = _read_jsonl(world_root / "recipient_preparation_conversations.jsonl")
    if len(rows) < 6:
        raise ValueError("recipient preparation corpus is too small to split")
    train_rows, validation_rows = rows[:-5], rows[-5:]
    if output_root.exists() or output_root.is_symlink():
        if output_root.is_symlink() or not output_root.is_dir():
            raise FileExistsError(
                f"recipient preparation split is not a regular directory: {output_root}"
            )
        train_path, validation_path = (
            output_root / "train.jsonl",
            output_root / "validation.jsonl",
        )
        if (
            not train_path.is_file()
            or not validation_path.is_file()
            or _read_jsonl(train_path) != train_rows
            or _read_jsonl(validation_path) != validation_rows
        ):
            raise FileExistsError(
                "existing recipient preparation split differs from source"
            )
        return train_path, validation_path
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True)
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    return train_path, validation_path


def _allocation_arrays(
    path: Path,
    fact_ids: list[str | None],
    tokenizer: Any,
    key_by_fact_id: dict[str, StructuredKey],
    key_encoder: FrozenStructuredKeyEncoder,
    owner_code: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    documents = list(iter_rendered_conversations(path))
    if len(documents) != len(fact_ids):
        raise ValueError(
            "conversation rows and allocation provenance are not one-to-one"
        )
    encodings = [
        tokenizer.encode(document.text, add_special_tokens=False)
        for document in documents
    ]
    token_count = sum(len(encoding.ids) + 1 for encoding in encodings)
    owners = np.zeros(token_count, dtype=np.uint8)
    semantic_queries: np.ndarray | None = None
    semantic_mask: np.ndarray | None = None
    if owner_code == OWNER_SEMANTIC:
        semantic_queries = np.zeros(
            (token_count, key_encoder.dimension), dtype=np.float32
        )
        semantic_mask = np.zeros(token_count, dtype=np.bool_)
    offset = 0
    for document, encoding, fact_id in zip(documents, encodings, fact_ids, strict=True):
        if fact_id is not None:
            key = key_by_fact_id.get(fact_id)
            if key is None:
                raise ValueError(
                    f"adapter example references an ineligible fact: {fact_id}"
                )
            for index, (start, end) in enumerate(encoding.offsets):
                supervised = any(
                    start < end and span_start <= start and end <= span_end
                    for span_start, span_end in document.supervision_spans
                )
                if not supervised:
                    continue
                target_position = offset + index
                owners[target_position] = owner_code
                if semantic_queries is not None and semantic_mask is not None:
                    query_position = target_position - 1
                    if query_position < 0:
                        raise ValueError("semantic answer has no causal query position")
                    semantic_queries[query_position] = key_encoder.encode(key)
                    semantic_mask[query_position] = True
        offset += len(encoding.ids) + 1
    return owners, semantic_queries, semantic_mask


def build_portability_allocation(
    world_manifest_path: Path,
    output_root: Path,
    *,
    representation: str,
    condition: str,
    semantic_pack_path: Path | None,
) -> dict[str, object]:
    """Write token-aligned ownership/query sidecars for one recipient arm."""
    if representation not in {"token", "byte", "semantic"}:
        raise ValueError("unknown portability representation")
    if condition not in {
        "adapter-tuned",
        "frozen-only",
        "joint",
        "random",
        "corrupt",
        "disabled",
        "native",
    }:
        raise ValueError("unsupported portability condition")
    if (representation == "semantic" and condition == "native") or (
        condition == "disabled" and semantic_pack_path is not None
    ):
        raise ValueError("condition cannot use the requested semantic attachment")
    if (
        representation == "semantic"
        and condition != "disabled"
        and semantic_pack_path is None
    ):
        raise ValueError("semantic allocation requires the selected training pack")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"portability allocation already exists: {output_root}")
    world_root = Path(world_manifest_path).parent
    world_manifest = json.loads(world_manifest_path.read_text(encoding="utf-8"))
    world_digest = world_manifest.get("sha256")
    if (
        hashlib.sha256(
            canonical_json(
                {key: value for key, value in world_manifest.items() if key != "sha256"}
            )
        ).hexdigest()
        != world_digest
    ):
        raise ValueError("portability world manifest hash mismatch")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.allocation-", dir=output_root.parent
        )
    )
    try:
        wrappers = _read_jsonl(world_root / "recipient_adapter_examples.jsonl")
        conversations: list[dict[str, Any]] = []
        training_fact_ids: list[str | None] = []
        for row in wrappers:
            conversation = row.get("conversation")
            if not isinstance(conversation, dict):
                raise TypeError(
                    "adapter corpus contains an invalid conversation envelope"
                )
            conversations.append(conversation)
            fact_id = row.get("fact_id")
            training_fact_ids.append(fact_id if isinstance(fact_id, str) else None)
        train_path = staging / "train.jsonl"
        _write_jsonl(train_path, conversations)

        preparation = _read_jsonl(
            world_root / "recipient_preparation_conversations.jsonl"
        )
        validation_conversations = preparation[-5:]
        validation_path = staging / "validation.jsonl"
        _write_jsonl(validation_path, validation_conversations)
        validation_fact_ids: list[str | None] = [None] * len(validation_conversations)

        tokenizer_path = world_root / world_manifest["tokenizer"]["path"]
        tokenizer = load_tokenizer(tokenizer_path)
        key_encoder, _ = frozen_encoders()
        eligible = set(world_manifest["recipient_adapter_fact_ids"])
        key_by_fact_id = {
            row["id"]: StructuredKey(row["subject"], row["relation"])
            for world in world_manifest["worlds"]
            if world["partition"] == "train"
            for row in _load_fact_rows(world_root, world["world_id"])
            if row["id"] in eligible
        }
        if set(key_by_fact_id) != eligible:
            raise ValueError("eligible training fact keys are not one-to-one")
        owner_code = (
            OWNER_NEURAL
            if condition == "disabled"
            else OWNER_SEMANTIC
            if representation == "semantic"
            else OWNER_LEXICAL
        )
        train_owner, train_queries, train_mask = _allocation_arrays(
            train_path,
            training_fact_ids,
            tokenizer,
            key_by_fact_id,
            key_encoder,
            owner_code,
        )
        validation_owner, validation_queries, validation_mask = _allocation_arrays(
            validation_path,
            validation_fact_ids,
            tokenizer,
            key_by_fact_id,
            key_encoder,
            OWNER_SEMANTIC
            if representation == "semantic" and condition != "disabled"
            else OWNER_NEURAL,
        )
        semantic: dict[str, object] | None = None
        if representation == "semantic" and condition != "disabled":
            assert semantic_pack_path is not None
            pack_target = staging / "semantic_memory.enpack"
            shutil.copytree(semantic_pack_path, pack_target)
            retriever = SemanticRetriever.from_pack(pack_target)
            semantic = {
                "pack_path": pack_target.name,
                "pack_sha256": sha256_file(pack_target / "manifest.json"),
                "pack_id": retriever.pack_id,
                "key_encoder": retriever.key_encoder.model_dump(mode="json"),
            }
        allocation = build_allocation_manifest(
            staging / "allocation.json",
            source_identity_sha256=str(source_identity()["sha256"]),
            tokenizer_sha256=hashlib.sha256(
                tokenizer.to_str().encode("utf-8")
            ).hexdigest(),
            train_jsonl_sha256=sha256_file(train_path),
            validation_jsonl_sha256=sha256_file(validation_path),
            train_owner=train_owner,
            validation_owner=validation_owner,
            semantic=semantic,
            train_queries=train_queries,
            validation_queries=validation_queries,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )
        _rename_noreplace(staging, output_root)
        return {
            "root": output_root,
            "train_path": output_root / train_path.name,
            "validation_path": output_root / validation_path.name,
            "allocation_manifest_path": output_root / "allocation.json",
            "training_fact_ids": sorted(eligible),
            "allocation_sha256": allocation.sha256,
            "train_token_count": int(train_owner.shape[0]),
            "validation_token_count": int(validation_owner.shape[0]),
            "semantic_pack_id": None if semantic is None else semantic["pack_id"],
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _directory_descriptor(root: Path, path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"portability checkpoint must be a regular directory: {path}")
    files = [
        _file_descriptor(path, member)
        for member in sorted(path.rglob("*"))
        if member.is_file()
    ]
    if not files:
        raise ValueError(f"portability checkpoint directory is empty: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "files": files,
    }


def _campaign_config(
    *,
    world_manifest_path: Path,
    training_path: Path,
    validation_path: Path,
    cache_dir: Path,
    logging_root: Path,
    seed: int,
    recipient: str,
    updates: int,
    representation: str | None,
    condition: str | None,
    allocation_manifest_path: Path | None,
    portability_manifest_path: Path | None,
    trainable_parameters: tuple[str, ...] | None,
    prep: bool = False,
) -> Any:
    from importlib.resources import files

    from klab.config.loading import load_config
    from klab.config.models import RunConfig

    world_root = world_manifest_path.parent
    world_manifest = json.loads(world_manifest_path.read_text(encoding="utf-8"))
    width = int(recipient.removeprefix("width"))
    if width not in (32, 64):
        raise ValueError(f"unsupported portability recipient: {recipient}")
    model = {
        "vocab_size": world_manifest["tokenizer"]["vocab_size"],
        "hidden_dim": width,
        "num_layers": 1,
        "num_heads": 4,
        "ffn_dim": width * 2,
        "max_seq_len": 128,
        "rms_norm_eps": 1e-6,
        "tie_embeddings": True,
        "ffn": "dense",
        "num_experts": 1,
        "experts_per_token": 1,
        "shared_expert": False,
        "router_aux_loss_coefficient": 0.0,
        "memory": "none",
        "memory_injection": "final",
        "memory_table_size": 0,
        "memory_ngram_size": 0,
        "memory_dim": 0,
        "memory_package_path": None,
        "memory_ngram_orders": [],
        "memory_hash_heads": 1,
        "semantic_memory_dim": None,
    }
    if not prep:
        assert representation is not None and condition is not None
        if condition != "disabled":
            model.update(
                memory_table_size=_TABLE_SIZE,
                memory_ngram_size=_TOKEN_ORDER,
                memory_dim=_VALUE_DIM,
            )
            if representation == "token":
                model["memory"] = "ngram"
            elif representation == "byte":
                model["memory"] = "byte"
            elif representation == "semantic":
                model.update(
                    memory="none",
                    memory_table_size=0,
                    memory_ngram_size=0,
                    memory_dim=0,
                    semantic_memory_dim=_VALUE_DIM,
                )
            else:
                raise ValueError(
                    f"unsupported portability representation: {representation}"
                )
    base_path = Path(str(files("klab.research").joinpath("resources/base.yaml")))
    base = load_config(base_path).model_dump(mode="python")
    base["name"] = (
        f"portability-prep-{recipient}-s{seed}"
        if prep
        else f"portability-{recipient}-{representation}-{condition}-s{seed}"
    )
    base["seed"] = seed
    base["model"] = model
    base["tokenizer"] = {"path": world_root / world_manifest["tokenizer"]["path"]}
    base["dataset"] = {
        "source": "local_chat",
        "cache_dir": cache_dir,
        "train_max_documents": 4096,
        "validation_max_documents": 512,
        "train_max_tokens": 1_000_000,
        "validation_max_tokens": 1_000_000,
        "synthetic_seed": seed,
        "train_path": training_path,
        "validation_path": validation_path,
        "license": "CC0-1.0",
        "allocation_manifest_path": allocation_manifest_path,
    }
    base["training"] = {
        "micro_batch_size": 1,
        "gradient_accumulation": 1,
        "seq_len": 64,
        "max_steps": updates,
        "max_tokens": updates * 64,
        "grad_clip_norm": 1.0,
        "neural_loss_weight": 1.0,
        "deterministic": True,
        "trainable_parameters": (
            trainable_parameters
            if trainable_parameters is not None
            else ("embedding.weight",)
            if portability_manifest_path is not None
            else None
        ),
        "portability_manifest_path": portability_manifest_path,
    }
    base["optimizer"] = {
        "name": "adamw",
        "peak": 3e-4,
        "floor": 3e-5,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "state_offload": False,
    }
    base["evaluation"] = {"every_steps": 1, "max_batches": 2}
    base["checkpoint"] = {
        "every_steps": 1,
        "every_tokens": None,
        "every_minutes": None,
        "keep_periodic": True,
    }
    base["logging"] = {
        "root_dir": logging_root,
        "every_steps": 1,
        "architecture_diagnostics": "scalar",
    }
    base["runtime"] = {
        **base["runtime"],
        "engine": "pytorch",
        "backend": "cpu",
        "precision": "fp32",
    }
    return RunConfig.model_validate(base)


def _directory_descriptor(root: Path, path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"portability checkpoint must be a regular directory: {path}")
    files = [
        _file_descriptor(path, member)
        for member in sorted(path.rglob("*"))
        if member.is_file()
    ]
    if not files:
        raise ValueError(f"portability checkpoint directory is empty: {path}")
    return {"path": path.relative_to(root).as_posix(), "files": files}


def _write_immutable_bytes(path: Path, encoded: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(f"immutable portability file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise FileExistsError(f"immutable portability file differs: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    digest = canonical_digest(body)
    _write_immutable_bytes(path, canonical_json({**body, "sha256": digest}) + b"\n")
    return digest


def _observation_inputs(world_root: Path, partition: str, output: Path) -> Path:
    manifest = json.loads(
        (world_root / "world_manifest.json").read_text(encoding="utf-8")
    )
    all_queries = _read_jsonl(world_root / f"{partition}_queries.jsonl")
    all_scores = {
        row["case_id"]: row
        for row in _read_jsonl(world_root / f"{partition}_scores.jsonl")
    }
    cases: list[dict[str, Any]] = []
    for world in manifest["worlds"]:
        if world["partition"] != partition:
            continue
        world_id = world["world_id"]
        queries = [row for row in all_queries if row["world_id"] == world_id]
        if any(row["case_id"] not in all_scores for row in queries):
            raise ValueError(f"query/scorer rows are misaligned for {world_id}")
        cases.extend(
            {**query, "scorer": all_scores[query["case_id"]]} for query in queries
        )
    if not cases:
        raise ValueError(f"no {partition} portability observations were materialized")
    encoded = b"".join(canonical_json(row) + b"\n" for row in cases)
    _write_immutable_bytes(output, encoded)
    return output


def build_portability_campaign(
    output_root: Path,
    *,
    seed: int = 20260925,
    scale: str = "smoke",
    updates: int = 4,
) -> Path:
    """Materialize the immutable protocol, worlds, assets, and query observations."""
    if updates <= 0:
        raise ValueError("portability updates must be positive")
    from klab.data.portability_worlds import materialize_portability_worlds

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    world_path = materialize_portability_worlds(
        output_root / "worlds", seed=seed, scale=scale
    )
    asset_path = build_portability_memory_assets(world_path, output_root / "assets")
    world = json.loads(world_path.read_text(encoding="utf-8"))
    assets = json.loads(asset_path.read_text(encoding="utf-8"))
    address_path = asset_path.parent / assets["address_observations"]
    _observation_inputs(
        world_path.parent,
        "development",
        output_root / "observations" / "development.jsonl",
    )
    _observation_inputs(
        world_path.parent, "final", output_root / "observations" / "final.jsonl"
    )
    protocol = {
        "format": "klab-portability-protocol",
        "version": 1,
        "campaign_id": f"engram-portability-v1-{scale}-seed-{seed}",
        "seed": seed,
        "scale": scale,
        "world_manifest_sha256": world["sha256"],
        "memory_assets_sha256": assets["sha256"],
        "address_observations_sha256": sha256_file(address_path),
        "query_producer": (
            "generator-provided structured keys and frozen value vectors; "
            "not a natural-language query encoder"
        ),
        "compiler": (
            "token and byte tables are deterministic direct compiles of declared "
            "structured records; not source-model-trained representations"
        ),
        "recipient_seeds": [17, 41, 73],
        "recipient_widths": [32, 64],
        "representations": ["token", "byte", "semantic"],
        "conditions": {
            "token": [
                "native",
                "adapter-tuned",
                "frozen-only",
                "joint",
                "random",
                "corrupt",
                "disabled",
            ],
            "byte": [
                "native",
                "adapter-tuned",
                "frozen-only",
                "joint",
                "random",
                "corrupt",
                "disabled",
            ],
            "semantic": [
                "adapter-tuned",
                "frozen-only",
                "joint",
                "random",
                "corrupt",
                "disabled",
            ],
        },
        "training": {
            "optimizer": "AdamW",
            "planned_updates": updates,
            "recipient_preparation_updates": updates,
            "recipient_preparation_only": True,
            "frozen_conditions": ["adapter-tuned", "random", "corrupt"],
            "threshold": {
                "metric": "development_exact_answer_accuracy",
                "minimum": 0.95,
                "censoring": "right-censored at final planned update; no extrapolation",
            },
        },
        "limitations": [
            "DenseLM width changes are cross-width, not cross-architecture evidence.",
            "Structured semantic vectors are supplied by the generator; ordinary text-query production is unproven.",
            "Directly compiled token and byte tables are not learned source-producer checkpoints.",
            "No teacher-derived hidden-state compilation is performed.",
            "A successful retrieval trace is not evidence of learned recipient behavior.",
        ],
    }
    protocol_path = output_root / "portability_protocol.json"
    _write_immutable_json(protocol_path, protocol)
    plan = {
        "format": "klab-portability-plan",
        "version": 1,
        "campaign_id": protocol["campaign_id"],
        "protocol_sha256": json.loads(protocol_path.read_text(encoding="utf-8"))[
            "sha256"
        ],
        "arms": [
            {
                "recipient": f"width{width}",
                "representation": representation,
                "condition": condition,
                "seed": recipient_seed,
            }
            for width in protocol["recipient_widths"]
            for representation, conditions in protocol["conditions"].items()
            for condition in conditions
            for recipient_seed in protocol["recipient_seeds"]
        ],
    }
    _write_immutable_json(output_root / "plan.json", plan)
    return protocol_path


def _run_if_needed(config: Any, run_id: str) -> Path:
    from klab.training.checkpoints import CheckpointManager
    from klab.training.manifest import config_sha256, read_manifest
    from klab.training.trainer import train

    run_dir = config.logging.root_dir / run_id
    if not run_dir.exists():
        train(config, run_id=run_id)
    else:
        manifest = read_manifest(run_dir / "manifest.json")
        if manifest["effective_config_sha256"] != config_sha256(
            config.model_dump(mode="json")
        ):
            raise FileExistsError(
                f"existing portability run has a different config: {run_dir}"
            )
        latest = CheckpointManager(run_dir).recovery_report().record
        if latest is None or latest.step < config.training.max_steps:
            train(config, recover=run_dir, run_id=run_id)
    if not (run_dir / "manifest.json").is_file():
        raise RuntimeError(f"training did not create a run manifest: {run_dir}")
    latest = CheckpointManager(run_dir).recovery_report().record
    if latest is None or latest.step < config.training.max_steps:
        raise RuntimeError(
            f"portability training did not reach planned updates: {run_dir}"
        )
    return run_dir


def _prepare_recipient(
    campaign_root: Path,
    world_manifest_path: Path,
    preparation_train: Path,
    preparation_validation: Path,
    *,
    recipient: str,
    seed: int,
    updates: int,
) -> dict[str, object]:
    from klab.training.checkpoints import CheckpointManager
    from klab.training.manifest import read_manifest

    run_id = f"preparation-s{seed}"
    logging_root = campaign_root / "runs" / "preparation" / recipient / f"s{seed}"
    config = _campaign_config(
        world_manifest_path=world_manifest_path,
        training_path=preparation_train,
        validation_path=preparation_validation,
        cache_dir=campaign_root / "cache" / "preparation" / recipient / f"s{seed}",
        logging_root=logging_root,
        seed=seed,
        recipient=recipient,
        updates=updates,
        representation=None,
        condition=None,
        allocation_manifest_path=None,
        portability_manifest_path=None,
        trainable_parameters=None,
        prep=True,
    )
    run_dir = _run_if_needed(config, run_id)
    manifest = read_manifest(run_dir / "manifest.json")
    manager = CheckpointManager(run_dir)
    record = manager.recovery_report().record
    if record is None:
        raise RuntimeError(
            f"recipient preparation has no verified checkpoint: {run_dir}"
        )
    checkpoint = run_dir / "checkpoints" / record.relative_path
    snapshot = manager.load(checkpoint, mode="promote")
    if snapshot.step != 0 or not snapshot.checkpoint_sha256:
        raise ValueError(
            "preparation checkpoint promotion did not yield immutable weights"
        )
    if manifest["effective_config"]["model"]["hidden_dim"] != int(
        recipient.removeprefix("width")
    ):
        raise ValueError(
            "recipient preparation checkpoint width differs from its label"
        )
    return {
        "checkpoint": _directory_descriptor(campaign_root, checkpoint),
        "checkpoint_sha256": snapshot.checkpoint_sha256,
        "architecture_sha256": manifest["architecture_sha256"],
        "tokenizer_sha256": json.loads(world_manifest_path.read_text(encoding="utf-8"))[
            "tokenizer"
        ]["sha256"],
    }


def _trainable_names(
    config: Any, *, condition: str, semantic_pack_path: Path | None
) -> tuple[str, ...]:
    from klab.engram.semantic import SemanticRetriever
    from klab.model.inspection import named_tensor_inventory
    from klab.model.transformer import DenseLM

    model = DenseLM(config.model, config.attention)
    if semantic_pack_path is not None:
        model.add_semantic_memory(
            "allocation",
            SemanticRetriever.from_pack(semantic_pack_path),
            site="final",
            min_score=3.5,
        )
    named = dict(model.named_parameters())
    if condition in {"joint", "native", "disabled"}:
        selected = tuple(
            name for name, parameter in named.items() if parameter.requires_grad
        )
    else:
        selected = tuple(
            name
            for name, parameter in named.items()
            if parameter.requires_grad
            and name.endswith(
                (
                    "memory.output.weight",
                    "memory.gate.weight",
                    "semantic_memories.allocation.output.weight",
                    "semantic_memories.allocation.gate.weight",
                )
            )
        )
    if not selected:
        raise ValueError(f"no trainable model parameters selected for {condition}")
    inventory = named_tensor_inventory(
        config.model, config.attention, trainable_parameters=selected
    )
    if any(name not in inventory or not inventory[name].trainable for name in selected):
        raise ValueError(
            "selected portability names differ from canonical model inventory"
        )
    return selected


def _asset_descriptor_from_manifest(
    campaign_root: Path, asset_root: Path, descriptor: dict[str, Any]
) -> dict[str, object]:
    artifact = asset_root / descriptor["artifact"]["path"]
    if "files" in descriptor["artifact"]:
        return _directory_descriptor(campaign_root, artifact)
    return _file_descriptor(campaign_root, artifact)


def build_portability_run_manifest(
    campaign_root: Path,
    *,
    recipient: str,
    representation: str,
    condition: str,
    seed: int,
    updates: int,
) -> tuple[Path, Any, dict[str, object]]:
    """Bind one coordinate to immutable inputs and its canonical optimizer names."""
    campaign_root = Path(campaign_root)
    protocol_path = campaign_root / "portability_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if updates != protocol["training"]["planned_updates"]:
        raise ValueError(
            "requested updates differ from the immutable campaign protocol"
        )
    world_path = campaign_root / "worlds" / "world_manifest.json"
    world = json.loads(world_path.read_text(encoding="utf-8"))
    asset_root = campaign_root / "assets"
    asset_manifest_path = asset_root / "memory_assets.json"
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if world["sha256"] != protocol["world_manifest_sha256"]:
        raise ValueError("protocol and world manifest identities differ")
    if asset_manifest["sha256"] != protocol["memory_assets_sha256"]:
        raise ValueError("protocol and memory asset identities differ")
    if (
        representation not in protocol["conditions"]
        or condition not in protocol["conditions"][representation]
    ):
        raise ValueError("coordinate is not declared in the immutable protocol")
    if seed not in protocol["recipient_seeds"]:
        raise ValueError("recipient seed is not declared in the immutable protocol")
    if recipient not in {f"width{width}" for width in protocol["recipient_widths"]}:
        raise ValueError("recipient is not declared in the immutable protocol")

    preparation_train, preparation_validation = prepare_recipient_initialization_split(
        world_path.parent, campaign_root / "preparation" / "recipient"
    )
    backbone_by_seed: dict[str, dict[str, object] | None] = {}
    for candidate_seed in protocol["recipient_seeds"]:
        backbone_by_seed[str(candidate_seed)] = (
            None
            if condition == "native"
            else _prepare_recipient(
                campaign_root,
                world_path,
                preparation_train,
                preparation_validation,
                recipient=recipient,
                seed=candidate_seed,
                updates=updates,
            )
        )

    asset_condition = condition if condition in {"random", "corrupt"} else "real"
    artifacts: dict[str, Any] = {}
    if condition not in {"disabled", "native"}:
        artifacts = asset_manifest["representations"][representation][asset_condition]
    semantic_pack = (
        asset_root / artifacts["training"]["artifact"]["path"]
        if representation == "semantic" and condition != "disabled"
        else None
    )
    allocation_root = campaign_root / "allocations" / f"{representation}-{condition}"
    allocation = (
        build_portability_allocation(
            world_path,
            allocation_root,
            representation=representation,
            condition=condition,
            semantic_pack_path=semantic_pack,
        )
        if not allocation_root.exists()
        else {
            "root": allocation_root,
            "train_path": allocation_root / "train.jsonl",
            "validation_path": allocation_root / "validation.jsonl",
            "allocation_manifest_path": allocation_root / "allocation.json",
        }
    )

    if condition == "disabled":
        memory_entry: dict[str, object] = {
            "kind": representation,
            "artifact": None,
            "pack_id": None,
            "tensor_sha256": None,
            "addressing": None,
            "encoder_contract": None,
            "replacements": {},
        }
    elif condition == "native":
        addressing = (
            {
                "tokenizer_sha256": world["tokenizer"]["sha256"],
                "order": _TOKEN_ORDER,
                "hash_heads": 1,
                "rows": _TABLE_SIZE,
                "embedding_dim": _VALUE_DIM,
                **token_addressing_marker(_TABLE_SIZE, 1),
            }
            if representation == "token"
            else {
                "format_version": 1,
                "normalization": "raw-utf8-v1",
                "hashing": "poly257-terminal-v1",
                "ngram_size": _BYTE_ORDER,
                "table_size": _TABLE_SIZE,
                "embedding_dim": _VALUE_DIM,
            }
        )
        memory_entry = {
            "kind": representation,
            "artifact": None,
            "pack_id": None,
            "tensor_sha256": None,
            "addressing": addressing,
            "encoder_contract": None,
            "replacements": {},
        }
    else:
        training_asset = artifacts["training"]
        replacements = {
            world_id: {
                "artifact": _asset_descriptor_from_manifest(
                    campaign_root, asset_root, descriptor
                ),
                "pack_id": descriptor["pack_id"],
                "tensor_sha256": descriptor["tensor_sha256"],
            }
            for world_id, descriptor in artifacts.items()
            if world_id != "training"
        }
        memory_entry = {
            "kind": representation,
            "artifact": _asset_descriptor_from_manifest(
                campaign_root, asset_root, training_asset
            ),
            "pack_id": training_asset["pack_id"],
            "tensor_sha256": training_asset["tensor_sha256"],
            "addressing": training_asset["addressing"],
            "encoder_contract": training_asset["encoder_contract"],
            "replacements": replacements,
        }
    memory_by_seed = {
        str(candidate_seed): memory_entry
        for candidate_seed in protocol["recipient_seeds"]
    }
    observation_descriptors = {
        partition: _file_descriptor(
            campaign_root, campaign_root / "observations" / f"{partition}.jsonl"
        )
        for partition in ("development", "final")
    }
    manifest_value = {
        "format": "klab-portability-run",
        "version": 1,
        "protocol": _file_descriptor(campaign_root, protocol_path),
        "world_manifest": _file_descriptor(campaign_root, world_path),
        "coordinate": {
            "recipient": recipient,
            "representation": representation,
            "condition": condition,
        },
        "seeds": [17, 41, 73],
        "initial_backbone": backbone_by_seed,
        "memory": memory_by_seed,
        "observations": observation_descriptors,
        "training_fact_ids": (
            []
            if condition == "frozen-only"
            else sorted(world["recipient_adapter_fact_ids"])
        ),
    }
    manifest_path = campaign_root / f"run-{recipient}-{representation}-{condition}.json"
    _write_immutable_json(manifest_path, manifest_value)
    config = _campaign_config(
        world_manifest_path=world_path,
        training_path=allocation["train_path"],
        validation_path=allocation["validation_path"],
        cache_dir=campaign_root
        / "cache"
        / f"{recipient}-{representation}-{condition}"
        / f"s{seed}",
        logging_root=campaign_root
        / "runs"
        / recipient
        / representation
        / condition
        / f"s{seed}",
        seed=seed,
        recipient=recipient,
        updates=updates,
        representation=representation,
        condition=condition,
        allocation_manifest_path=allocation["allocation_manifest_path"],
        portability_manifest_path=manifest_path,
        trainable_parameters=None,
    )
    semantic_path = (
        allocation_root / "semantic_memory.enpack"
        if representation == "semantic" and condition != "disabled"
        else None
    )
    trainable = _trainable_names(
        config, condition=condition, semantic_pack_path=semantic_path
    )
    config = type(config).model_validate(
        config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={"trainable_parameters": trainable}
                )
            }
        ).model_dump(mode="python")
    )
    return manifest_path, config, allocation
