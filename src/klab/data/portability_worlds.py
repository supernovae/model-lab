"""Leakage-audited synthetic worlds for Engram portability experiments."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import DatasetConfig, TokenizerTrainConfig
from klab.data.conversations import iter_conversations
from klab.data.tokenizer import load_tokenizer, train_tokenizer
from klab.data.toy_worlds import (
    FrozenStructuredKeyEncoder,
    ProducerFact,
    StructuredKey,
    _record_payload,
    _write_pack_inputs,
    frozen_encoders,
)
from klab.engram.packs import (
    _rename_noreplace,
    compile_pack,
    verify_pack,
)
from klab.training.manifest import sha256_file

_SYMBOLS = tuple("ABCDEFGHIJKLMNOP")
_RELATIONS = ("next", "label")
_SCALE_TRAIN_WORLDS = {"smoke": 2, "nano": 2, "micro": 8, "tiny": 8}
_CREATED_AT = "2026-09-25T00:00:00Z"
_FORMAT = "klab-portability-worlds"
_VERSION = 1


def _stream_seed(seed: int, scale: str, purpose: str) -> int:
    value = f"engram-portability-v1|generator-v1|{seed}|{scale}|{purpose}"
    return int.from_bytes(hashlib.sha256(value.encode("ascii")).digest()[:8], "big")


def _hash_key(encoder: FrozenStructuredKeyEncoder, key: StructuredKey) -> bytes:
    return np.ascontiguousarray(encoder.encode(key), dtype=np.float32).tobytes()


def _permutation(rng: random.Random) -> dict[str, str]:
    while True:
        values = list(_SYMBOLS)
        rng.shuffle(values)
        mapping = dict(zip(_SYMBOLS, values, strict=True))
        unseen = set(_SYMBOLS)
        valid = True
        while unseen:
            start = min(unseen)
            current = start
            length = 0
            while current in unseen:
                unseen.remove(current)
                current = mapping[current]
                length += 1
            if current != start or length < 4:
                valid = False
                break
        if valid:
            return mapping


def _walk(mapping: dict[str, str], start: str, hops: int) -> str:
    for _ in range(hops):
        start = mapping[start]
    return start


def _world_specs(seed: int, scale: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    specs: list[dict[str, Any]] = []
    streams: dict[str, int] = {}
    training_count = _SCALE_TRAIN_WORLDS[scale]

    def add_single(world_id: str, partition: str, slot: int, purpose: str) -> None:
        p_seed = _stream_seed(seed, scale, f"{purpose}|next")
        l_seed = _stream_seed(seed, scale, f"{purpose}|label")
        streams[f"{purpose}|next"] = p_seed
        streams[f"{purpose}|label"] = l_seed
        next_map = _permutation(random.Random(p_seed))
        label_values = list(_SYMBOLS)
        random.Random(l_seed).shuffle(label_values)
        specs.append(
            {
                "world_id": world_id,
                "partition": partition,
                "slot": slot,
                "next": next_map,
                "label": dict(zip(_SYMBOLS, label_values, strict=True)),
            }
        )

    for index in range(training_count):
        add_single(f"train-{index:02d}", "train", index, f"train-{index:02d}")

    for partition, base_slot in (("development", 32), ("final", 48)):
        for index in range(2):
            slot = base_slot + index
            slot_name = f"{partition}-{index:02d}"
            next_seed = _stream_seed(seed, scale, f"{slot_name}|next-pair")
            label_seed = _stream_seed(seed, scale, f"{slot_name}|label-pair")
            streams[f"{slot_name}|next-pair"] = next_seed
            streams[f"{slot_name}|label-pair"] = label_seed
            pair_rng = random.Random(next_seed)
            next_a = _permutation(pair_rng)
            next_b = _permutation(pair_rng)
            # Fixed probe A has distinct 1/2/3-hop endpoints across replacement worlds.
            while any(
                _walk(next_a, "A", hops) == _walk(next_b, "A", hops)
                for hops in (1, 2, 3)
            ):
                next_b = _permutation(pair_rng)
            label_rng = random.Random(label_seed)
            label_a = list(_SYMBOLS)
            label_b = list(_SYMBOLS)
            label_rng.shuffle(label_a)
            label_rng.shuffle(label_b)
            while any(a == b for a, b in zip(label_a, label_b, strict=True)):
                label_b = list(_SYMBOLS)
                label_rng.shuffle(label_b)
            for side, next_map, labels in (
                ("a", next_a, label_a),
                ("b", next_b, label_b),
            ):
                specs.append(
                    {
                        "world_id": f"{partition}-{index:02d}-{side}",
                        "partition": partition,
                        "slot": slot,
                        "replacement": side,
                        "next": next_map.copy(),
                        "label": dict(zip(_SYMBOLS, labels, strict=True)),
                    }
                )
    return specs, streams


def _key_subject(slot: int, node: str, nonce: int) -> str:
    return f"w{slot:02d}.{node}~{nonce}"


def _assign_key_nonces(
    specs: list[dict[str, Any]], encoder: FrozenStructuredKeyEncoder
) -> tuple[dict[tuple[int, str, str], int], int]:
    keys: dict[tuple[int, str, str], int] = {}
    address_owners: dict[bytes, tuple[int, str, str]] = {}
    initial_collisions = 0
    for spec in specs:
        slot = int(spec["slot"])
        nodes = (*_SYMBOLS, "C0", "T0", "D0", "D1", "D2", "D3", "M0", "M1", "M2", "M3")
        for node in nodes:
            for relation in _RELATIONS:
                key_id = (slot, node, relation)
                if key_id in keys:
                    continue
                nonce = 0
                while True:
                    key = StructuredKey(_key_subject(slot, node, nonce), relation)
                    address = _hash_key(encoder, key)
                    owner = address_owners.get(address)
                    if owner is None or owner == key_id:
                        keys[key_id] = nonce
                        address_owners[address] = key_id
                        break
                    if nonce == 0:
                        initial_collisions += 1
                    nonce += 1
                    if nonce > 1_000_000:
                        raise ValueError("could not resolve structured key collision")
    return keys, initial_collisions


def _prompt(
    slot: int, node: str, relation: str, nonce: int, wording: str
) -> tuple[str, str]:
    lookup = f"w{slot:02d}.{node}~{nonce}.{relation}"
    if len(lookup) > 32:
        raise ValueError(f"lookup address exceeds fixed 32-byte suffix: {lookup}")
    suffix = "|" * (32 - len(lookup)) + lookup
    return f"{wording}\n{suffix}", suffix


def _conversation(prompt: str, answer: str) -> dict[str, object]:
    return {
        "format_version": 2,
        "loss_mode": "assistant_only",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))


def _fact_row(
    fact: ProducerFact, role: str, *, permitted: list[str]
) -> dict[str, object]:
    row = _record_payload(fact)
    row.update(
        {
            "world_id": fact.world_id,
            "partition": role,
            "permitted_consumer_roles": permitted,
            "source": "generated",
            "license": "CC0-1.0",
        }
    )
    row["content_digest_sha256"] = canonical_digest(row)
    return row


def _make_facts(
    spec: dict[str, Any], nonces: dict[tuple[int, str, str], int]
) -> tuple[list[ProducerFact], dict[tuple[str, str], str]]:
    world_id = str(spec["world_id"])
    partition = str(spec["partition"])
    slot = int(spec["slot"])
    facts: list[ProducerFact] = []
    eligible_ids: dict[tuple[str, str], str] = {}

    def add(
        node: str, relation: str, value: str, suffix: str, **times: str | None
    ) -> ProducerFact:
        nonce = nonces[(slot, node, relation)]
        subject = _key_subject(slot, node, nonce)
        fact = ProducerFact(
            f"portability:{world_id}:{suffix}",
            world_id,
            StructuredKey(subject, relation),
            value,
            **times,
        )
        facts.append(fact)
        return fact

    for node in _SYMBOLS:
        next_fact = add(node, "next", spec["next"][node], f"next-{node}")
        label_fact = add(node, "label", spec["label"][node], f"label-{node}")
        eligible_symbols = set(
            "ABCDEFGH" if int(spec.get("slot", 0)) % 2 == 0 else "IJKLMNOP"
        )
        if partition == "train":
            if next_fact.value in eligible_symbols:
                eligible_ids[(node, "next")] = next_fact.id
            if label_fact.value in eligible_symbols:
                eligible_ids[(node, "label")] = label_fact.id

    add("C0", "label", "A", "conflict-a")
    add("C0", "label", "B", "conflict-b")
    add(
        "T0",
        "label",
        "C",
        "temporal-old",
        valid_from="2020-01-01",
        valid_until="2024-12-31",
    )
    add("T0", "label", "D", "temporal-current", valid_from="2025-01-01")
    for index, value in enumerate(("E", "H", "K", "N")):
        add(f"D{index}", "label", value, f"distractor-{index}")
    return facts, eligible_ids


def _cases_for_world(
    spec: dict[str, Any], nonces: dict[tuple[int, str, str], int]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    world_id = str(spec["world_id"])
    partition = str(spec["partition"])
    slot = int(spec["slot"])
    template_prefix = {
        "train": "training",
        "development": "development",
        "final": "final",
    }[partition]
    queries: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []

    def add(
        case_type: str,
        node: str,
        relations: list[str],
        expected: str,
        *,
        wording: str | None = None,
        alias: bool = False,
        as_of: str | None = None,
        path: list[str] | None = None,
    ) -> None:
        case_id = f"case:{world_id}:{len(queries):03d}:{case_type}"
        relation = relations[0]
        nonce = nonces[(slot, node, relation)]
        phrasing = (
            wording
            or f"{template_prefix} template {case_type}: report the supplied one-character answer."
        )
        if alias:
            phrasing = (
                f"{template_prefix} alias {partition.upper()}-{node}-ALT: {phrasing}"
            )
        address_node = f"ALT{node}" if alias else node
        prompt, suffix = _prompt(slot, address_node, relation, nonce, phrasing)
        queries.append(
            {
                "case_id": case_id,
                "world_id": world_id,
                "partition": partition,
                "task": case_type,
                "template": f"{template_prefix}-v1-{case_type}",
                "alias": f"{partition}-alias-{node}" if alias else None,
                "prompt": prompt,
                "lookup_suffix": suffix,
                "start_key": {
                    "subject": _key_subject(slot, node, nonce),
                    "relation": relation,
                },
                "relations": relations,
                "as_of": as_of,
                "address_generalization": alias,
            }
        )
        scores.append(
            {
                "case_id": case_id,
                "expected_answer": expected,
                "expected_path": path or [],
                "expected_status": "conflict"
                if case_type == "conflict"
                else "temporal_miss"
                if case_type == "temporal-miss"
                else "unknown"
                if case_type == "missing"
                else "retrieved",
                "changed_assignment": bool(spec.get("replacement"))
                and case_type in {"next", "two-hop", "three-hop", "label"},
            }
        )

    next_map: dict[str, str] = spec["next"]
    labels: dict[str, str] = spec["label"]
    for node in _SYMBOLS:
        add("next", node, ["next"], next_map[node], path=[node, next_map[node]])
    for node in _SYMBOLS:
        add("label", node, ["label"], labels[node], path=[node, labels[node]])
    for hops in (2, 3):
        for node in _SYMBOLS:
            chain = [node]
            for _ in range(hops):
                chain.append(next_map[chain[-1]])
            add(
                "two-hop" if hops == 2 else "three-hop",
                node,
                ["next"] * hops,
                chain[-1],
                path=chain,
            )
    for node in _SYMBOLS:
        add(
            "alias",
            node,
            ["label"],
            labels[node],
            alias=True,
            path=[node, labels[node]],
        )
    for node in _SYMBOLS:
        add(
            "changed-wording",
            node,
            ["label"],
            labels[node],
            wording=f"{template_prefix} alternate phrasing {node}: identify its label.",
            path=[node, labels[node]],
        )
    for node in _SYMBOLS:
        add(
            "distractor-label",
            node,
            ["label"],
            labels[node],
            wording=f"{template_prefix} ignore unrelated records and identify {node}'s label.",
            path=[node, labels[node]],
        )
    for index, node in enumerate(_SYMBOLS):
        override = _SYMBOLS[(index + 7) % len(_SYMBOLS)]
        add(
            "context-override",
            node,
            ["label"],
            override,
            wording=f"{template_prefix} for this turn only, override the label of {node} as {override}; repeat the overridden value.",
            path=[override],
        )
    add("conflict", "C0", ["label"], "!", path=[])
    add("temporal", "T0", ["label"], "C", as_of="2023-06-01", path=["C"])
    add("temporal", "T0", ["label"], "D", as_of="2026-06-01", path=["D"])
    add("temporal-miss", "T0", ["label"], "?", as_of="2019-06-01", path=[])
    for index in range(4):
        add("missing", f"M{index}", ["label"], "?", path=[])
    if len(queries) != 136:
        raise AssertionError(
            f"world {world_id} generated {len(queries)} cases, expected 136"
        )
    return queries, scores


def _verify_existing(root: Path, expected_manifest_sha256: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise FileExistsError(
            f"existing portability world is not a regular directory: {root}"
        )
    path = root / "world_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FileExistsError(
            f"existing portability world is not verifiable: {root}"
        ) from error
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    actual = canonical_digest(content)
    if actual != manifest.get("sha256") or actual != expected_manifest_sha256:
        raise FileExistsError(
            f"existing portability world differs from requested replay: {root}"
        )
    expected_files = {entry["path"]: entry for entry in manifest["files"]}
    actual_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "world_manifest.json"
    }
    if actual_files != set(expected_files):
        raise FileExistsError(
            f"existing portability world file inventory differs: {root}"
        )
    for relative, entry in expected_files.items():
        item = root / relative
        if (
            item.is_symlink()
            or item.stat().st_size != entry["size_bytes"]
            or sha256_file(item) != entry["sha256"]
        ):
            raise FileExistsError(
                f"existing portability world asset failed verification: {relative}"
            )
    return path


def materialize_portability_worlds(
    root: Path, *, seed: int = 20260925, scale: str = "micro"
) -> Path:
    """Create immutable producer worlds, recipient inputs, packs, and audit manifests.

    The returned path is the versioned world manifest. Existing identical materializations
    are verified and reused; a differing or damaged destination is never overwritten.
    """
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if scale not in _SCALE_TRAIN_WORLDS:
        raise ValueError(f"unsupported portability scale: {scale}")
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.portability-", dir=root.parent)
    )
    try:
        specs, streams = _world_specs(seed, scale)
        key_encoder, value_encoder = frozen_encoders()
        nonces, initial_collision_count = _assign_key_nonces(specs, key_encoder)
        all_facts: dict[str, list[ProducerFact]] = {}
        eligible_by_world: dict[str, dict[tuple[str, str], str]] = {}
        queries_by_world: dict[str, list[dict[str, object]]] = {}
        scores_by_world: dict[str, list[dict[str, object]]] = {}
        producer_rows: dict[str, list[dict[str, object]]] = {}
        train_facts: list[ProducerFact] = []
        recipient_train_ids: set[str] = set()
        producer_root = staging / "producers"
        for spec in specs:
            world_id = str(spec["world_id"])
            facts, eligible = _make_facts(spec, nonces)
            queries, scores = _cases_for_world(spec, nonces)
            all_facts[world_id] = facts
            eligible_by_world[world_id] = eligible
            queries_by_world[world_id] = queries
            scores_by_world[world_id] = scores
            permissions: list[str]
            if spec["partition"] == "train":
                permissions = ["source_training", "recipient_semantic_compile"]
                train_facts.extend(facts)
                recipient_train_ids.update(eligible.values())
            else:
                permissions = ["evaluation_attachment"]
            eligible_ids = set(eligible.values())
            rows = [
                _fact_row(
                    fact,
                    str(spec["partition"]),
                    permitted=(
                        [*permissions, "recipient_adapter_training"]
                        if fact.id in eligible_ids
                        else permissions
                    ),
                )
                for fact in facts
            ]
            producer_rows[world_id] = rows
            _write_jsonl(producer_root / f"{world_id}.jsonl", rows)

        source_conversations: list[dict[str, object]] = []
        source_conversation_fact_ids: list[str] = []
        for spec in specs:
            if spec["partition"] != "train":
                continue
            slot = int(spec["slot"])
            for fact in all_facts[str(spec["world_id"])]:
                node = next(
                    (
                        candidate
                        for candidate in (*_SYMBOLS, "C0", "T0", "D0", "D1", "D2", "D3")
                        if _key_subject(
                            slot,
                            candidate,
                            nonces[(slot, candidate, fact.key.relation)],
                        )
                        == fact.key.subject
                    ),
                    "C0",
                )
                temporal_text = ""
                if fact.valid_from is not None or fact.valid_until is not None:
                    temporal_text = (
                        f" Valid from {fact.valid_from or 'unbounded'} "
                        f"through {fact.valid_until or 'unbounded'}."
                    )
                wording = (
                    f"Source record: the value for {node} and {fact.key.relation} "
                    f"is supplied.{temporal_text}"
                )
                prompt, _ = _prompt(
                    slot,
                    node,
                    fact.key.relation,
                    nonces[(slot, node, fact.key.relation)],
                    wording,
                )
                source_conversations.append(_conversation(prompt, fact.value))
                source_conversation_fact_ids.append(fact.id)
        _write_jsonl(
            staging / "source_training_conversations.jsonl", source_conversations
        )

        preparation: list[dict[str, object]] = []
        for symbol in (*_SYMBOLS, "?", "!"):
            preparation.append(
                _conversation(
                    f"Repeat the one-character value supplied in this message: {symbol}.",
                    symbol,
                )
            )
        preparation.extend(
            [
                _conversation(
                    "No matching information was supplied. Return the missing-information marker ?.",
                    "?",
                ),
                _conversation(
                    "Two incompatible values remain unresolved. Return the conflict marker !.",
                    "!",
                ),
                _conversation(
                    "A turn-local override replaces any earlier supplied value. Repeat this override: A.",
                    "A",
                ),
            ]
        )
        _write_jsonl(staging / "recipient_preparation_conversations.jsonl", preparation)

        adapter_examples: list[dict[str, object]] = []
        adapter_fact_ids: list[str] = []
        for spec in specs:
            if spec["partition"] != "train":
                continue
            slot = int(spec["slot"])
            world_id = str(spec["world_id"])
            fact_lookup = {fact.id: fact for fact in all_facts[world_id]}
            for (node, relation), fact_id in sorted(
                eligible_by_world[world_id].items()
            ):
                fact = fact_lookup[fact_id]
                wording = f"Recipient adapter training association: report {node}'s supplied {relation} value."
                prompt, _ = _prompt(
                    slot, node, relation, nonces[(slot, node, relation)], wording
                )
                adapter_examples.append(
                    {
                        "example_id": f"adapter:{world_id}:{node}:{relation}",
                        "fact_id": fact_id,
                        "world_id": world_id,
                        "permitted_consumer_role": "recipient_adapter_training",
                        "conversation": _conversation(prompt, fact.value),
                    }
                )
                adapter_fact_ids.append(fact_id)
            for index in range(4):
                node = f"M{index}"
                prompt, _ = _prompt(
                    slot,
                    node,
                    "label",
                    nonces[(slot, node, "label")],
                    "Training-only missing-key convention example.",
                )
                adapter_examples.append(
                    {
                        "example_id": f"adapter:{world_id}:missing:{index}",
                        "fact_id": None,
                        "world_id": world_id,
                        "permitted_consumer_role": "recipient_adapter_training",
                        "conversation": _conversation(prompt, "?"),
                    }
                )
            for index, node in enumerate(_SYMBOLS[:4]):
                override = _SYMBOLS[(index + 9) % len(_SYMBOLS)]
                prompt, _ = _prompt(
                    slot,
                    node,
                    "label",
                    nonces[(slot, node, "label")],
                    f"Training-only context override: for this turn use {override} as {node}'s label; report the override.",
                )
                adapter_examples.append(
                    {
                        "example_id": f"adapter:{world_id}:override:{node}",
                        "fact_id": None,
                        "permitted_consumer_role": "recipient_adapter_training",
                        "conversation": _conversation(prompt, override),
                    }
                )
        _write_jsonl(staging / "recipient_adapter_examples.jsonl", adapter_examples)

        for partition, filename in (
            ("development", "development"),
            ("final", "final"),
            ("train", "training"),
        ):
            queries = [
                row
                for spec in specs
                if spec["partition"] == partition
                for row in queries_by_world[str(spec["world_id"])]
            ]
            scores = [
                row
                for spec in specs
                if spec["partition"] == partition
                for row in scores_by_world[str(spec["world_id"])]
            ]
            _write_jsonl(staging / f"{filename}_queries.jsonl", queries)
            _write_jsonl(staging / f"{filename}_scores.jsonl", scores)

        packs_root = staging / "packs"
        packs_root.mkdir()
        pack_inventory: dict[str, dict[str, object]] = {}
        recipient_training_facts = tuple(
            fact
            for spec in specs
            if spec["partition"] == "train"
            for fact in all_facts[str(spec["world_id"])]
            if fact.id in recipient_train_ids
        )
        pack_specs: list[tuple[str, tuple[ProducerFact, ...]]] = [
            ("training", recipient_training_facts)
        ]
        pack_specs.extend(
            (str(spec["world_id"]), tuple(all_facts[str(spec["world_id"])]))
            for spec in specs
            if spec["partition"] != "train"
        )
        for pack_name, facts in pack_specs:
            inputs = staging / f".{pack_name}-inputs"
            source, keys_path, values_path, metadata_path = _write_pack_inputs(
                inputs, facts, key_encoder, value_encoder, random_keys=False
            )
            output = packs_root / f"{pack_name}.enpack"
            pack_manifest = compile_pack(
                source,
                output,
                name=f"engram-portability-{pack_name}",
                namespace="engram-portability-v1",
                default_license="CC0-1.0",
                source_name="generated",
                source_revision="generator-v1",
                created_at=_CREATED_AT,
                semantic_keys=keys_path,
                semantic_values=values_path,
                semantic_metadata=metadata_path,
            )
            shutil.rmtree(inputs)
            verified = verify_pack(output, expected_pack_id=pack_manifest.pack_id)
            if not verified.valid:
                raise ValueError(f"portability pack failed verification: {pack_name}")
            pack_inventory[pack_name] = {
                "path": output.relative_to(staging).as_posix(),
                "pack_id": pack_manifest.pack_id,
                "record_ids": [fact.id for fact in facts],
            }

        tokenizer_train_path = staging / "tokenizer_training_conversations.jsonl"
        tokenizer_rows = [*source_conversations, *preparation]
        _write_jsonl(tokenizer_train_path, tokenizer_rows)
        validation_path = staging / "tokenizer_validation_conversations.jsonl"
        _write_jsonl(validation_path, [preparation[0]])
        train_bytes = sum(len(canonical_json(row)) + 1 for row in tokenizer_rows)
        tokenizer_config = TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            min_frequency=1,
            max_documents=max(1, len(tokenizer_rows)),
            output_dir=staging / "tokenizer",
            dataset=DatasetConfig(
                source="local_chat",
                cache_dir=staging / "cache",
                train_max_documents=max(1, len(tokenizer_rows)),
                validation_max_documents=1,
                train_max_tokens=max(1, train_bytes),
                validation_max_tokens=1,
                train_path=tokenizer_train_path,
                validation_path=validation_path,
                license="CC0-1.0",
            ),
        )
        tokenizer_path = train_tokenizer(tokenizer_config)
        tokenizer = load_tokenizer(tokenizer_path)

        raw_text_parts: list[str] = []
        lexical_rows: list[dict[str, object]] = []
        byte_offset = 0
        token_offset = 0
        for conversation, fact_id in zip(
            source_conversations, source_conversation_fact_ids, strict=True
        ):
            temporary_jsonl = staging / ".one-conversation.jsonl"
            _write_jsonl(temporary_jsonl, [conversation])
            rendered = next(iter_conversations(temporary_jsonl))
            temporary_jsonl.unlink()
            separator = "\n\n" if raw_text_parts else ""
            byte_offset += len(separator.encode("utf-8"))
            token_offset += len(tokenizer.encode(separator).ids) if separator else 0
            raw_text_parts.append(rendered)
            byte_start = byte_offset
            byte_end = byte_start + len(rendered.encode("utf-8"))
            encoded = tokenizer.encode(rendered)
            token_start = token_offset
            token_end = token_start + len(encoded.ids)
            lexical_rows.append(
                {
                    "fact_id": fact_id,
                    "rendered_record_id": f"source:{fact_id}",
                    "byte_span": [byte_start, byte_end],
                    "token_span": [token_start, token_end],
                    "tokenizer_sha256": sha256_file(tokenizer_path),
                    "lexical_source": True,
                    "semantic_row_id": fact_id,
                    "temporal_filter": "absent",
                }
            )
            byte_offset = byte_end
            token_offset = token_end
        lexical_text = "\n\n".join(raw_text_parts)
        (staging / "lexical_training.txt").write_bytes(lexical_text.encode("utf-8"))

        for row in adapter_examples:
            conversation = row["conversation"]
            if not isinstance(conversation, dict):
                raise TypeError("adapter example has invalid serialized conversation")
            messages = conversation.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(messages[0], dict)
            ):
                raise ValueError("adapter example lacks its user prompt")
            rendered_prompt = messages[0].get("content")
            if not isinstance(rendered_prompt, str):
                raise TypeError("adapter example has non-text user prompt")
            # Validate the exact serialized prompt/query alignment and trailing address context.
            one = staging / ".adapter-check.jsonl"
            _write_jsonl(one, [conversation])
            rendered_doc = next(iter_conversations(one))
            one.unlink()
            answer_start = rendered_doc.rfind(" ")
            prefix = rendered_doc[:answer_start]
            encoded_prefix = tokenizer.encode(prefix)
            key = rendered_prompt.rsplit("\n", 1)[-1].lstrip("|")
            key_start = prefix.rfind(key)
            if key_start < 0:
                raise ValueError("serialized adapter prompt lost its lookup address")
            key_tokens = [
                index
                for index, (start, end) in enumerate(encoded_prefix.offsets)
                if end > key_start and start < key_start + len(key)
            ]
            if not key_tokens or key_tokens[0] < max(0, len(encoded_prefix.ids) - 32):
                raise ValueError(
                    "complete lookup key is not in the last 32 causal tokens"
                )

        rep_rows: list[dict[str, object]] = []
        train_fact_set = {fact.id for fact in train_facts}
        for spec in specs:
            for row in producer_rows[str(spec["world_id"])]:
                fact_id = str(row["id"])
                is_train = fact_id in train_fact_set
                lexical_row = next(
                    (item for item in lexical_rows if item["fact_id"] == fact_id),
                    None,
                )
                rep_rows.append(
                    {
                        "fact_id": fact_id,
                        "world_id": row["world_id"],
                        "partition": spec["partition"],
                        "source_fact_ids_included": [fact_id] if is_train else [],
                        "source_fact_ids_omitted": [] if is_train else [fact_id],
                        "omission_reason": None
                        if is_train
                        else "replacement-world lexical training forbidden",
                        "rendered_record_id": f"source:{fact_id}" if is_train else None,
                        "byte_span": lexical_row["byte_span"] if lexical_row else None,
                        "token_span": lexical_row["token_span"]
                        if lexical_row
                        else None,
                        "semantic_row_id": fact_id,
                        "semantic_pack_role": "source-training"
                        if is_train
                        else "evaluation-attachment",
                        "permitted_consumer_roles": row["permitted_consumer_roles"],
                    }
                )
        if len(raw_text_parts) != len(set(raw_text_parts)):
            raise ValueError("duplicate rendered lexical producer content")
        producer_ids = {
            str(row["id"]) for rows in producer_rows.values() for row in rows
        }
        query_ids = {
            str(row["case_id"]) for rows in queries_by_world.values() for row in rows
        }
        if len(producer_ids) != sum(len(rows) for rows in producer_rows.values()):
            raise ValueError("duplicate producer fact ID")
        if producer_ids & query_ids:
            raise ValueError("producer and query IDs must be disjoint")
        if query_ids != {
            str(row["case_id"]) for rows in scores_by_world.values() for row in rows
        }:
            raise ValueError("query and scorer case IDs are not one-to-one")
        if set(adapter_fact_ids) != recipient_train_ids:
            raise ValueError("adapter examples do not exactly match eligible fact IDs")
        partitions = {
            partition: [
                row
                for spec in specs
                if spec["partition"] == partition
                for row in queries_by_world[str(spec["world_id"])]
            ]
            for partition in ("train", "development", "final")
        }
        templates = {
            partition: {str(row["template"]) for row in rows}
            for partition, rows in partitions.items()
        }
        aliases = {
            partition: {str(row["alias"]) for row in rows if row["alias"] is not None}
            for partition, rows in partitions.items()
        }
        if (
            templates["train"] & templates["development"]
            or templates["train"] & templates["final"]
            or templates["development"] & templates["final"]
            or aliases["train"] & aliases["development"]
            or aliases["train"] & aliases["final"]
            or aliases["development"] & aliases["final"]
        ):
            raise ValueError("query templates and aliases must be partition-disjoint")
        representation_manifest = {
            "format": "klab-portability-representations",
            "version": 1,
            "lexical_corpus_sha256": sha256_file(staging / "lexical_training.txt"),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "corpus_split": "training-world producer facts only",
            "records": rep_rows,
        }
        (staging / "representation_manifest.json").write_bytes(
            canonical_json(representation_manifest) + b"\n"
        )

        for path in (
            staging / "source_training_conversations.jsonl",
            staging / "recipient_preparation_conversations.jsonl",
            staging / "recipient_adapter_examples.jsonl",
        ):
            if not path.is_file():
                raise ValueError(f"missing required generated input: {path.name}")
        for fact_id in recipient_train_ids:
            matching = [
                row for row in adapter_examples if row.get("fact_id") == fact_id
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"recipient association eligibility is not one-to-one: {fact_id}"
                )
        replacement_ids = {
            str(row["id"])
            for spec in specs
            if spec["partition"] != "train"
            for row in producer_rows[str(spec["world_id"])]
        }
        if replacement_ids & recipient_train_ids:
            raise ValueError("replacement facts entered recipient adapter training")

        # Record the exact generated world/key specification and all immutable assets.
        file_entries = []
        for item in sorted(staging.rglob("*")):
            if item.is_file() and item.name != "world_manifest.json":
                file_entries.append(
                    {
                        "path": item.relative_to(staging).as_posix(),
                        "sha256": sha256_file(item),
                        "size_bytes": item.stat().st_size,
                    }
                )
        world_spec = [
            {
                "world_id": spec["world_id"],
                "partition": spec["partition"],
                "slot": spec["slot"],
                "replacement": spec.get("replacement"),
                "next": spec["next"],
                "label": spec["label"],
                "producer_fact_ids": [
                    fact.id for fact in all_facts[str(spec["world_id"])]
                ],
                "eligible_adapter_fact_ids": sorted(
                    eligible_by_world[str(spec["world_id"])].values()
                ),
            }
            for spec in specs
        ]
        manifest_content: dict[str, object] = {
            "format": _FORMAT,
            "version": _VERSION,
            "generator": "generator-v1",
            "seed": seed,
            "scale": scale,
            "rng_derivation": "SHA-256 of engram-portability-v1|generator-v1|data-seed|scale|purpose; first 8 bytes big-endian",
            "rng_stream_seeds": streams,
            "worlds": world_spec,
            "key_encoder": key_encoder.identity.model_dump(mode="json"),
            "value_encoder": value_encoder.identity.model_dump(mode="json"),
            "structured_key_dimension": key_encoder.dimension,
            "structured_value_dimension": value_encoder.dimension,
            "key_nonce_by_slot_node_relation": {
                f"{slot}|{node}|{relation}": nonce
                for (slot, node, relation), nonce in sorted(nonces.items())
            },
            "original_distinct_key_collision_count": initial_collision_count,
            "symbol_alphabet": [*_SYMBOLS, "?", "!"],
            "training_fact_count": len(train_facts),
            "recipient_memory_fact_count": len(recipient_training_facts),
            "recipient_adapter_fact_ids": sorted(recipient_train_ids),
            "query_case_count_by_world": 136,
            "query_case_counts": {
                partition: sum(
                    len(queries_by_world[str(spec["world_id"])])
                    for spec in specs
                    if spec["partition"] == partition
                )
                for partition in ("train", "development", "final")
            },
            "semantic_packs": pack_inventory,
            "tokenizer": {
                "path": tokenizer_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(tokenizer_path),
                "vocab_size": tokenizer.get_vocab_size(),
                "training_input_sha256": sha256_file(tokenizer_train_path),
                "training_split": "train-only generated facts and recipient preparation",
            },
            "lexical_corpus": {
                "path": "lexical_training.txt",
                "sha256": sha256_file(staging / "lexical_training.txt"),
                "tokenizer_sha256": sha256_file(tokenizer_path),
                "temporal_filter": "absent; temporal producer values are rendered without a lexical time filter",
            },
            "files": file_entries,
        }
        digest = canonical_digest(manifest_content)
        manifest = {**manifest_content, "sha256": digest}
        (staging / "world_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        if root.exists() or root.is_symlink():
            _verify_existing(root, digest)
            return root / "world_manifest.json"
        try:
            _rename_noreplace(staging, root)
        except FileExistsError:
            _verify_existing(root, digest)
        return root / "world_manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
