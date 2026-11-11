# Existing learned-data manifests use ValueError for malformed schemas.
# ruff: noqa: TRY004
"""Deterministic, leakage-audited inputs for learned Engram portability v1.

This module deliberately produces ordinary assistant-supervised conversations rather
than a compiled memory artifact.  The scorer is the only published label channel for
queries, so consumers cannot accidentally learn labels from evaluation inputs.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import DatasetConfig, TokenizerTrainConfig
from klab.data.byte_hash import table_address, token_bytes
from klab.data.tokenizer import load_tokenizer, train_tokenizer
from klab.engram.packs import _rename_noreplace
from klab.evaluation.chat import format_chat_prompt
from klab.training.manifest import sha256_file

_FORMAT = "klab-learned-portability-data"
_VERSION = 1
_NAMESPACE = "learned-engram-portability-v1|data-v1"
_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
_TABLE_SIZE = 65_521
_NGRAM_SIZE = 32
_EMBEDDING_DIM = 32
_MAX_NONCE = 0xFFFF

# Template text is an immutable part of the ownership/auditing contract.
_TEMPLATES: dict[str, dict[str, str]] = {
    "source_lookup": {"role": "source_training", "text": "Lookup the stored symbol."},
    "source_recall": {"role": "source_training", "text": "Recall the stored symbol."},
    "adapter_return": {
        "role": "recipient_calibration",
        "text": "Return the stored symbol.",
    },
    "adapter_give": {
        "role": "recipient_calibration",
        "text": "Give the stored symbol.",
    },
    "source_monitor": {"role": "source_gate", "text": "Which symbol is assigned?"},
    "final_report": {
        "role": "held_out_evaluation",
        "text": "Report the assigned symbol.",
    },
    "final_state": {
        "role": "held_out_evaluation",
        "text": "State the assigned symbol.",
    },
    "preparation_copy": {
        "role": "preparation",
        "text": "Copy the supplied symbol {symbol}.",
    },
}


def _stream(seed: int, fact_count: int, purpose: str) -> bytes:
    text = f"{_NAMESPACE}|{seed}|{fact_count}|{purpose}"
    return hashlib.sha256(text.encode("ascii")).digest()


def _rng(seed: int, fact_count: int, purpose: str) -> random.Random:
    return random.Random(int.from_bytes(_stream(seed, fact_count, purpose)[:8], "big"))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))


def _conversation(message: str, answer: str) -> dict[str, object]:
    return {
        "format_version": 2,
        "loss_mode": "assistant_only",
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ],
    }


def _key(seed: int, fact_count: int, purpose: str, index: int, nonce: int) -> str:
    # The nonce is independently selected before labels are allocated.
    stem = hashlib.sha256(
        _stream(seed, fact_count, purpose) + index.to_bytes(8, "big")
    ).hexdigest()[:12]
    return f"k{stem}.{nonce:04x}"


def _suffix(key: str) -> str:
    if len(key.encode("ascii")) != 18:
        raise AssertionError("learned portability key must be exactly 18 ASCII bytes")
    return "|" * 14 + key


def _message(
    template_id: str,
    key: str,
    *,
    answer: str | None = None,
    filler: str = "",
) -> str:
    template = _TEMPLATES[template_id]["text"]
    wording = template.format(symbol=answer) if answer is not None else template
    # Filler stays before the lookup suffix, never between it and the assistant marker.
    return f"{wording}\n{filler}{_suffix(key)}"


def _prompt(
    template_id: str, key: str, *, answer: str | None = None, filler: str = ""
) -> str:
    return format_chat_prompt(
        [], _message(template_id, key, answer=answer, filler=filler)
    )


def _query_prompt(
    tokenizer: Tokenizer,
    template_id: str,
    key: str,
    *,
    answer: str | None = None,
) -> str:
    template = _TEMPLATES[template_id]["text"]
    if "{symbol}" in template and answer is None:
        raise ValueError("learned query template requires a supplied symbol")
    base = _prompt(template_id, key, answer=answer)
    base_tokens = len(_token_ids(tokenizer, base + " "))
    filler_count = 126 - base_tokens
    if filler_count < 0:
        raise ValueError("learned query prefix exceeds the fixed model context")
    prompt = _prompt(template_id, key, answer=answer, filler="~" * filler_count)
    if len(_token_ids(tokenizer, prompt + " ")) != 126:
        raise ValueError("query padding did not produce equal 126-token prefixes")
    return prompt


def _answer_prefix(
    template_id: str, key: str, *, answer: str | None = None, filler: str = ""
) -> str:
    return _prompt(template_id, key, answer=answer, filler=filler) + " "


def _address(prefix: str) -> int:
    raw = prefix.encode("utf-8")
    if raw[-32:] != b"|" + raw[-31:]:
        # Kept explicit to make accidental prompt format changes fail visibly.
        raise ValueError("answer prefix does not retain a complete byte lookup suffix")
    return table_address(raw[-_NGRAM_SIZE:], _TABLE_SIZE)


def _assert_suffix_contract(prefix: str, key: str) -> None:
    expected = b"|" + key.encode("ascii") + b"\n\nAssistant: "
    if prefix.encode("utf-8")[-32:] != expected:
        raise ValueError("answer prefix violates the fixed 32-byte address contract")


def _token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    values = list(encoded.ids)
    if b"".join(token_bytes(tokenizer, value) for value in values) != text.encode(
        "utf-8"
    ):
        raise ValueError("tokenizer does not reconstruct serialized UTF-8 bytes")
    return values


def _address_stream(tokenizer: Tokenizer, text: str) -> list[int]:
    prefix = bytearray()
    rows: list[int] = []
    for value in _token_ids(tokenizer, text):
        prefix.extend(token_bytes(tokenizer, value))
        rows.append(table_address(bytes(prefix[-_NGRAM_SIZE:]), _TABLE_SIZE))
    return rows


def _padded_record(
    tokenizer: Tokenizer, template_id: str, key: str, answer: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Return one exactly-127-token document and its precise answer-side audit."""
    base = _answer_prefix(template_id, key, answer=answer)
    base_document = base + answer
    base_count = len(_token_ids(tokenizer, base_document))
    filler_count = 127 - base_count
    if filler_count < 0:
        raise ValueError("fixed learned portability conversation exceeds 127 tokens")
    filler = "~" * filler_count
    prefix = _answer_prefix(template_id, key, answer=answer, filler=filler)
    document = prefix + answer
    ids = _token_ids(tokenizer, document)
    if len(ids) != 127:
        raise ValueError("padding did not produce the required 127-token conversation")
    _assert_suffix_contract(prefix, key)
    answer_id = tokenizer.token_to_id(answer)
    if answer_id is None or _token_ids(tokenizer, answer) != [answer_id]:
        raise ValueError("answer symbols must each be exactly one tokenizer token")
    answer_position = len(_token_ids(tokenizer, prefix)) - 1
    stream = _address_stream(tokenizer, document)
    address = _address(prefix)
    if stream[answer_position] != address:
        raise ValueError(
            "prepared answer target does not use the manifest byte address"
        )
    return _conversation(
        _message(template_id, key, answer=answer, filler=filler), answer
    ), {
        "token_count_before_eos": 127,
        "answer_input_position": answer_position,
        "answer_target_position": answer_position + 1,
        "address": address,
        "whole_prefix_addresses": stream,
    }


def _candidate_access_rows(
    tokenizer: Tokenizer, template_id: str, key: str
) -> set[int]:
    """Return prompt rows plus the possible one-token answer-ending rows."""
    conversation, audit = _padded_record(tokenizer, template_id, key, _SYMBOLS[0])
    messages = conversation["messages"]
    assert isinstance(messages, list) and isinstance(messages[0], dict)
    prefix = format_chat_prompt([], str(messages[0]["content"])) + " "
    prefix_ids = _token_ids(tokenizer, prefix)
    if len(prefix_ids) != 126:
        raise ValueError("candidate answer prefix must contain exactly 126 tokens")
    rows = {int(value) for value in audit["whole_prefix_addresses"][:-1]}
    prefix_bytes = prefix.encode("utf-8")
    for symbol in _SYMBOLS:
        answer_id = tokenizer.token_to_id(symbol)
        if answer_id is None:
            raise ValueError("answer symbol is not a tokenizer token")
        ending = token_bytes(tokenizer, answer_id)
        rows.add(table_address((prefix_bytes + ending)[-_NGRAM_SIZE:], _TABLE_SIZE))
    return rows


def _sentinel_key(split: str) -> str:
    return f"k{hashlib.sha256(split.encode('ascii')).hexdigest()[:12]}.0000"


def _sentinel(
    tokenizer: Tokenizer, split: str
) -> tuple[dict[str, object], dict[str, object]]:
    # The split-specific generated key keeps train/validation conversations disjoint.
    return _padded_record(tokenizer, "source_lookup", _sentinel_key(split), "!")


def _pack_split(
    staging: Path,
    tokenizer: Tokenizer,
    name: str,
    records: list[tuple[str, dict[str, object], dict[str, object]]],
) -> list[dict[str, object]]:
    """Write exact EOS-packed arrays plus records for one supervised role."""
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer lacks <eos>")
    values: list[int] = []
    supervision: list[bool] = []
    addresses: list[int] = []
    block_map: list[dict[str, object]] = []
    for block, (record_id, conversation, audit) in enumerate(records):
        messages = conversation["messages"]
        assert isinstance(messages, list)
        user = messages[0]
        assistant = messages[1]
        assert isinstance(user, dict) and isinstance(assistant, dict)
        prompt = format_chat_prompt([], str(user["content"]))
        document = prompt + " " + str(assistant["content"])
        ids = _token_ids(tokenizer, document)
        stream = _address_stream(tokenizer, document)
        if len(ids) != 127 or len(stream) != 127:
            raise ValueError("packed learned document is not one full pre-EOS block")
        answer_position = int(audit["answer_input_position"])
        if stream[answer_position] != int(audit["address"]):
            raise ValueError("packed answer address differs from audited address")
        values.extend(ids + [eos])
        # The assistant separator space and one symbol are the only supervised targets.
        mask = [False] * 127 + [False]
        mask[answer_position] = True
        mask[answer_position + 1] = True
        supervision.extend(mask)
        addresses.extend(stream + [0])
        block_map.append(
            {
                "split": name,
                "block_index": block,
                "record_id": record_id,
                "fact_id": audit.get("fact_id"),
                "ownership": audit.get("ownership"),
                "template_id": audit.get("template_id"),
                "address": audit["address"],
                "answer_input_position": answer_position,
                "answer_target_position": audit["answer_target_position"],
                "whole_prefix_addresses": audit["whole_prefix_addresses"],
                "token_count_before_eos": 127,
            }
        )
    # One unscored generic document is enough to preserve every real 128-token block.
    sentinel, _ = _sentinel(tokenizer, name)
    sentinel_user = sentinel["messages"][0]
    sentinel_assistant = sentinel["messages"][1]
    assert isinstance(sentinel_user, dict) and isinstance(sentinel_assistant, dict)
    sentinel_doc = (
        format_chat_prompt([], str(sentinel_user["content"]))
        + " "
        + str(sentinel_assistant["content"])
    )
    sentinel_ids = _token_ids(tokenizer, sentinel_doc)
    values.extend(sentinel_ids + [eos])
    supervision.extend([False] * 128)
    addresses.extend(_address_stream(tokenizer, sentinel_doc) + [0])
    array_values = np.asarray(values, dtype=np.int32)
    array_supervision = np.asarray(supervision, dtype=bool)
    array_addresses = np.asarray(addresses, dtype=np.int32)
    if len(array_values) != (len(records) + 1) * 128:
        raise ValueError("packed block length is inconsistent")
    # TokenBlockDataset receives N candidate blocks, precisely the N factual rows.
    if (len(array_values) - 1) // 128 != len(records):
        raise ValueError("sentinel failed to retain every factual token block")
    np.save(staging / f"{name}_tokens.npy", array_values, allow_pickle=False)
    np.save(staging / f"{name}_supervision.npy", array_supervision, allow_pickle=False)
    np.save(staging / f"{name}_byte_addresses.npy", array_addresses, allow_pickle=False)
    return block_map


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_existing(root: Path, *, seed: int, fact_count: int) -> Path:
    manifest_path = root / "manifest.json"
    try:
        if (
            root.is_symlink()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise ValueError("manifest is missing or symlinked")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
        payload = dict(manifest)
        digest = payload.pop("sha256")
        if (
            digest != canonical_digest(payload)
            or type(manifest.get("seed")) is not int
            or manifest.get("seed") != seed
            or type(manifest.get("fact_count")) is not int
            or manifest.get("fact_count") != fact_count
        ):
            raise ValueError("manifest identity differs")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("manifest inventory is missing")
        expected: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise ValueError("manifest file descriptor malformed")
            relative = entry["path"]
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in expected
            ):
                raise ValueError("manifest file path is unsafe or duplicated")
            expected.add(relative)
            candidate = root / relative
            current = root
            for part in Path(relative).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError("manifest file traverses a symlink")
            if (
                not candidate.is_file()
                or not candidate.resolve().is_relative_to(root.resolve())
                or _descriptor(candidate, root) != entry
            ):
                raise ValueError("manifest file inventory mismatch")
        actual: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError("learned data root contains a symlink")
            if candidate.is_file() and candidate != manifest_path:
                actual.add(candidate.relative_to(root).as_posix())
        if actual != expected:
            raise ValueError("manifest inventory does not cover exact data contents")
        ownership_file = manifest.get("ownership_file")
        if not isinstance(ownership_file, dict):
            raise ValueError("ownership manifest file descriptor is missing")
        ownership_path = root / str(ownership_file.get("path"))
        if _descriptor(ownership_path, root) != ownership_file or json.loads(
            ownership_path.read_text(encoding="utf-8")
        ) != manifest.get("ownership"):
            raise ValueError("ownership file differs from manifest")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise FileExistsError(
            f"existing learned portability data is invalid: {root}"
        ) from error
    return manifest_path


def materialize_learned_portability_data(
    root: Path, *, seed: int = 20260925, fact_count: int
) -> Path:
    """Materialize immutable learned-association data and return ``manifest.json``.

    ``fact_count`` must give every ownership partition an independently balanced
    32-symbol assignment; the experimental counts (128, 512, 1024, 2048) satisfy it.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(fact_count, bool)
        or not isinstance(fact_count, int)
        or fact_count not in {128, 512, 1024, 2048}
    ):
        raise ValueError(
            "learned portability data requires a supported integer seed and fact count"
        )
    if root.exists() or root.is_symlink():
        return _verify_existing(root, seed=seed, fact_count=fact_count)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        # Key/nonce ownership is fixed before labels.  Preparation is allocated first
        # so source-monitor/held-out streams can fail closed on any leak.
        partitions = [
            ("calibration", fact_count // 4),
            ("source_monitor", fact_count // 4),
            ("held_out", fact_count // 2),
        ]

        occupied_targets: set[int] = set()
        conservative_forbidden: set[int] = set()

        prep_specs = [("preparation_train", 512), ("preparation_validation", 128)]
        prep_keys: dict[str, list[tuple[str, int]]] = {
            name: [] for name, _ in prep_specs
        }
        for purpose, count in prep_specs:
            for index in range(count):
                for nonce in range(_MAX_NONCE + 1):
                    key = _key(seed, fact_count, purpose, index, nonce)
                    # Every possible preparation symbol is audited as a causal input.
                    rows: set[int] = set()
                    for symbol in _SYMBOLS:
                        _, audit = _padded_record_placeholder(
                            tokenizer=None,
                            template_id="preparation_copy",
                            key=key,
                            answer=symbol,
                        )
                        rows.add(int(audit["address"]))
                    if (
                        len(rows) == 1
                        and next(iter(rows))
                        and next(iter(rows)) not in occupied_targets
                    ):
                        prep_keys[purpose].append((key, nonce))
                        occupied_targets.add(next(iter(rows)))
                        break
                else:
                    raise ValueError(
                        "addressing-capacity error while selecting preparation nonce"
                    )

        # Train tokenizer only on preparation text, with exact complete-corpus limits.
        tokenizer_train_rows: list[dict[str, object]] = []
        tokenizer_validation_rows: list[dict[str, object]] = []
        for purpose, target in (
            ("preparation_train", tokenizer_train_rows),
            ("preparation_validation", tokenizer_validation_rows),
        ):
            for index, (key, _) in enumerate(prep_keys[purpose]):
                target.append(
                    _conversation(
                        _message(
                            "preparation_copy",
                            key,
                            answer=_SYMBOLS[index % len(_SYMBOLS)],
                        ),
                        _SYMBOLS[index % len(_SYMBOLS)],
                    )
                )
        _write_jsonl(
            staging / "tokenizer_training_conversations.jsonl", tokenizer_train_rows
        )
        _write_jsonl(
            staging / "tokenizer_validation_conversations.jsonl",
            tokenizer_validation_rows,
        )
        train_bytes = (
            (staging / "tokenizer_training_conversations.jsonl").stat().st_size
        )
        tokenizer_path = train_tokenizer(
            TokenizerTrainConfig(
                schema_version=1,
                vocab_size=260,
                min_frequency=1,
                max_documents=len(tokenizer_train_rows),
                output_dir=staging / "tokenizer",
                dataset=DatasetConfig(
                    source="local_chat",
                    cache_dir=staging / "cache",
                    train_max_documents=len(tokenizer_train_rows),
                    validation_max_documents=len(tokenizer_validation_rows),
                    train_max_tokens=train_bytes,
                    validation_max_tokens=(
                        staging / "tokenizer_validation_conversations.jsonl"
                    )
                    .stat()
                    .st_size,
                    train_path=staging / "tokenizer_training_conversations.jsonl",
                    validation_path=staging
                    / "tokenizer_validation_conversations.jsonl",
                    license="CC0-1.0",
                ),
            )
        )
        tokenizer = load_tokenizer(tokenizer_path)
        if tokenizer.get_vocab_size() != 260:
            raise ValueError(
                "learned portability tokenizer is not the required 260-entry BPE"
            )
        model_payload = json.loads(tokenizer_path.read_text(encoding="utf-8")).get(
            "model", {}
        )
        if model_payload.get("merges") != []:
            raise ValueError("260-entry ByteLevel tokenizer must have no merge rules")
        for symbol in _SYMBOLS:
            if len(_token_ids(tokenizer, symbol)) != 1:
                raise ValueError("answer symbol is not a single tokenizer token")

        # Actual preparation/validation streams and their sentinels are forbidden
        # before any transferred fact key/label assignment.
        for purpose, entries in prep_keys.items():
            for index, (key, _) in enumerate(entries):
                symbol = _SYMBOLS[index % len(_SYMBOLS)]
                _, audit = _padded_record(tokenizer, "preparation_copy", key, symbol)
                conservative_forbidden.update(audit["whole_prefix_addresses"])
        for split in ("preparation_train", "preparation_validation"):
            _, audit = _sentinel(tokenizer, split)
            conservative_forbidden.update(audit["whole_prefix_addresses"])
        # Calibration receives every possible answer across both actual templates.
        calibration_keys: list[tuple[str, int]] = []
        for index in range(fact_count // 4):
            for nonce in range(_MAX_NONCE + 1):
                key = _key(seed, fact_count, "fact", index, nonce)
                candidates = {
                    _address(_answer_prefix(template, key))
                    for template in ("adapter_return", "adapter_give")
                }
                if (
                    len(candidates) == 1
                    and next(iter(candidates))
                    and next(iter(candidates)) not in occupied_targets
                ):
                    calibration_keys.append((key, nonce))
                    occupied_targets.add(next(iter(candidates)))
                    for template in ("adapter_return", "adapter_give"):
                        conservative_forbidden.update(
                            _candidate_access_rows(tokenizer, template, key)
                        )
                    break
            else:
                raise ValueError(
                    "addressing-capacity error while selecting calibration nonce"
                )
        _, calibration_sentinel_audit = _sentinel(tokenizer, "adapter_calibration")
        conservative_forbidden.update(
            calibration_sentinel_audit["whole_prefix_addresses"]
        )

        fact_keys: dict[str, list[tuple[str, int]]] = {
            "calibration": calibration_keys,
            "source_monitor": [],
            "held_out": [],
        }
        for ownership, count in partitions[1:]:
            for index in range(count):
                ordinal = len(calibration_keys) + sum(
                    len(fact_keys[name]) for name in ("source_monitor", "held_out")
                )
                for nonce in range(_MAX_NONCE + 1):
                    key = _key(seed, fact_count, "fact", ordinal, nonce)
                    address = _address(_answer_prefix("adapter_return", key))
                    # Final-site injection makes only this supervised answer row
                    # relevant; calibration and preparation streams stay forbidden.
                    if (
                        address
                        and address not in occupied_targets
                        and address not in conservative_forbidden
                    ):
                        fact_keys[ownership].append((key, nonce))
                        occupied_targets.add(address)
                        break
                else:
                    raise ValueError(
                        "addressing-capacity error while selecting source/held-out nonce"
                    )

        facts: list[dict[str, object]] = []
        labels: dict[str, str] = {}
        for ownership, _ in partitions:
            entries = fact_keys[ownership]
            assigned = list(_SYMBOLS) * (len(entries) // len(_SYMBOLS))
            _rng(seed, fact_count, f"labels|{ownership}").shuffle(assigned)
            for index, ((key, nonce), symbol) in enumerate(
                zip(entries, assigned, strict=True)
            ):
                fact_id = f"fact-{len(facts):04d}"
                address = _address(_answer_prefix("adapter_return", key))
                roles = ["source_training"]
                if ownership == "source_monitor":
                    roles.append("source_monitor")
                elif ownership == "calibration":
                    roles.append("recipient_calibration")
                else:
                    roles.append("held_out_evaluation")
                fact = {
                    "fact_id": fact_id,
                    "ownership": ownership,
                    "key": key,
                    "nonce": nonce,
                    "target_row": address,
                    "assigned_symbol": symbol,
                    "provenance": {
                        "source": "generated",
                        "license": "CC0-1.0",
                        "namespace": _NAMESPACE,
                    },
                    "permitted_roles": roles,
                }
                fact["content_digest_sha256"] = canonical_digest(fact)
                facts.append(fact)
                labels[fact_id] = symbol
        if len({int(row["target_row"]) for row in facts}) != fact_count or 0 in {
            int(row["target_row"]) for row in facts
        }:
            raise ValueError("fact target rows must be unique and nonzero")

        # Create padded v2 inputs; source sees all facts/two literal templates.
        source_rows: list[dict[str, object]] = []
        calibration_rows: list[dict[str, object]] = []
        preparation_rows: dict[str, list[dict[str, object]]] = {
            "preparation_train": [],
            "preparation_validation": [],
        }
        preparation_facts: list[dict[str, object]] = []
        packed: dict[str, list[tuple[str, dict[str, object], dict[str, object]]]] = {
            "source_train": [],
            "adapter_calibration": [],
            "preparation_train": [],
            "preparation_validation": [],
        }
        for fact in facts:
            fact_id, key, symbol, ownership = (
                str(fact["fact_id"]),
                str(fact["key"]),
                str(fact["assigned_symbol"]),
                str(fact["ownership"]),
            )
            for template in ("source_lookup", "source_recall"):
                conversation, audit = _padded_record(tokenizer, template, key, symbol)
                record_id = f"source:{fact_id}:{template}"
                source_rows.append(conversation)
                packed["source_train"].append(
                    (
                        record_id,
                        conversation,
                        {
                            **audit,
                            "fact_id": fact_id,
                            "ownership": ownership,
                            "template_id": template,
                        },
                    )
                )
            if ownership == "calibration":
                for template in ("adapter_return", "adapter_give"):
                    conversation, audit = _padded_record(
                        tokenizer, template, key, symbol
                    )
                    record_id = f"calibration:{fact_id}:{template}"
                    calibration_rows.append(conversation)
                    packed["adapter_calibration"].append(
                        (
                            record_id,
                            conversation,
                            {
                                **audit,
                                "fact_id": fact_id,
                                "ownership": ownership,
                                "template_id": template,
                            },
                        )
                    )
        for purpose, entries in prep_keys.items():
            split = "training" if purpose == "preparation_train" else "validation"
            for index, (key, nonce) in enumerate(entries):
                symbol = _SYMBOLS[index % len(_SYMBOLS)]
                conversation, audit = _padded_record(
                    tokenizer, "preparation_copy", key, symbol
                )
                record_id = f"prep-{split}-{index:04d}"
                fact = {
                    "fact_id": record_id,
                    "split": split,
                    "key": key,
                    "nonce": nonce,
                    "target_row": int(audit["address"]),
                    "assigned_symbol": symbol,
                    "provenance": {
                        "source": "generated",
                        "license": "CC0-1.0",
                        "namespace": _NAMESPACE,
                        "purpose": "preparation_only",
                    },
                }
                fact["content_digest_sha256"] = canonical_digest(fact)
                preparation_facts.append(fact)
                preparation_rows[purpose].append(conversation)
                packed[purpose].append(
                    (
                        record_id,
                        conversation,
                        {
                            **audit,
                            "fact_id": record_id,
                            "ownership": purpose,
                            "template_id": "preparation_copy",
                        },
                    )
                )
        for split, rows in (
            ("source_train", source_rows),
            ("adapter_calibration", calibration_rows),
            ("preparation_train", preparation_rows["preparation_train"]),
            ("preparation_validation", preparation_rows["preparation_validation"]),
        ):
            rows.append(_sentinel(tokenizer, split)[0])
        _write_jsonl(staging / "facts.jsonl", facts)
        _write_jsonl(staging / "preparation_facts.jsonl", preparation_facts)
        _write_jsonl(staging / "source_train.jsonl", source_rows)
        _write_jsonl(staging / "adapter_calibration.jsonl", calibration_rows)
        _write_jsonl(
            staging / "preparation_train.jsonl", preparation_rows["preparation_train"]
        )
        _write_jsonl(
            staging / "preparation_validation.jsonl",
            preparation_rows["preparation_validation"],
        )
        block_map: list[dict[str, object]] = []
        for split, rows in packed.items():
            block_map.extend(_pack_split(staging, tokenizer, split, rows))
        _write_jsonl(staging / "block_map.jsonl", block_map)

        queries: list[dict[str, object]] = []
        scorer: list[dict[str, object]] = []
        for fact in facts:
            fact_id, key, symbol, ownership = (
                str(fact["fact_id"]),
                str(fact["key"]),
                str(fact["assigned_symbol"]),
                str(fact["ownership"]),
            )
            template_ids = ["source_lookup", "source_recall"]
            if ownership == "source_monitor":
                template_ids.append("source_monitor")
            if ownership in {"calibration", "held_out"}:
                template_ids.append("adapter_return")
            if ownership == "held_out":
                template_ids.extend(("final_report", "final_state"))
            for template_id in template_ids:
                prompt = _query_prompt(tokenizer, template_id, key)
                answer_prefix = prompt + " "
                _assert_suffix_contract(answer_prefix, key)
                address = _address(answer_prefix)
                case_id = f"case:{fact_id}:{template_id}"
                queries.append(
                    {
                        "case_id": case_id,
                        "fact_id": fact_id,
                        "ownership": ownership,
                        "template_id": template_id,
                        "prompt": prompt,
                        "answer_prefix": answer_prefix,
                        "address": address,
                    }
                )
                scorer.append(
                    {"case_id": case_id, "fact_id": fact_id, "expected_symbol": symbol}
                )
        if {str(row["case_id"]) for row in queries} != {
            str(row["case_id"]) for row in scorer
        }:
            raise ValueError("query and scorer IDs must match exactly")
        if any("expected" in key or "symbol" in key for row in queries for key in row):
            raise ValueError("query records must not contain labels")
        _write_jsonl(staging / "queries.jsonl", queries)
        _write_jsonl(staging / "scorer.jsonl", scorer)
        target_rows = {int(row["target_row"]) for row in facts}
        protected_rows = {
            int(row["target_row"])
            for row in facts
            if row["ownership"] in {"source_monitor", "held_out"}
        }
        calibration_rows = np.load(
            staging / "adapter_calibration_byte_addresses.npy", allow_pickle=False
        )
        preparation_accesses = np.concatenate(
            [
                np.load(staging / f"{split}_byte_addresses.npy", allow_pickle=False)
                for split in ("preparation_train", "preparation_validation")
            ]
        )
        forbidden_actual = {
            int(value)
            for value in np.concatenate((calibration_rows, preparation_accesses))
        }
        if protected_rows.intersection(forbidden_actual):
            raise ValueError(
                "source-monitor/held-out target rows appear in calibration or preparation address streams"
            )
        factual_accesses = [
            int(address)
            for row in block_map
            for address in row["whole_prefix_addresses"]
        ]
        distinct_accessed_rows = set(factual_accesses)
        selected_nonces = [
            nonce
            for entries in [*fact_keys.values(), *prep_keys.values()]
            for _, nonce in entries
        ]
        address_audit = {
            "nonce_retries": sum(selected_nonces),
            "distinct_access_addresses": len(distinct_accessed_rows),
            "repeated_address_accesses": int(
                len(factual_accesses) - len(distinct_accessed_rows)
            ),
            "true_distinct_key_aliases": fact_count - len(target_rows),
            "occupancy": len(distinct_accessed_rows) / _TABLE_SIZE,
            "target_coverage": {
                "declared_facts": fact_count,
                "unique_nonzero_target_rows": len(target_rows),
            },
        }

        template_manifest = {
            name: {**value, "digest_sha256": canonical_digest(value)}
            for name, value in _TEMPLATES.items()
        }
        ownership_content = {
            name: {
                "count": count,
                "fact_ids": [
                    str(row["fact_id"]) for row in facts if row["ownership"] == name
                ],
                "permitted_roles": sorted(
                    {
                        role
                        for row in facts
                        if row["ownership"] == name
                        for role in row["permitted_roles"]
                    }
                ),
            }
            for name, count in partitions
        }
        ownership_path = staging / "ownership.json"
        ownership_path.write_bytes(canonical_json(ownership_content) + b"\n")
        ownership_descriptor = _descriptor(ownership_path, staging)
        files = [
            _descriptor(path, staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
        fact_descriptor = _descriptor(staging / "facts.jsonl", staging)
        manifest_content: dict[str, object] = {
            "format": _FORMAT,
            "version": _VERSION,
            "seed": seed,
            "fact_count": fact_count,
            "rng_derivation": "SHA-256 of learned-engram-portability-v1|data-v1|20260925|fact_count|purpose",
            "symbols": list(_SYMBOLS),
            "addressing": {
                "kind": "raw-utf8",
                "hash": "poly257-terminal-v1",
                "table_size": _TABLE_SIZE,
                "ngram_size": _NGRAM_SIZE,
                "embedding_dim": _EMBEDDING_DIM,
                "answer_prefix_contract": "hash(raw UTF-8 last 32 bytes of answer_prefix); answer_prefix == prompt + ' '",
            },
            "address_audit": address_audit,
            "tokenizer": {
                "path": tokenizer_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(tokenizer_path),
                "vocab_size": tokenizer.get_vocab_size(),
                "merges": [],
                "training": "preparation-only local_chat CC0-1.0",
            },
            "templates": template_manifest,
            "ownership": ownership_content,
            "ownership_file": ownership_descriptor,
            "facts": {**fact_descriptor, "count": len(facts)},
            "preparation": {
                "training_fact_ids": [
                    str(row["fact_id"])
                    for row in preparation_facts
                    if row["split"] == "training"
                ],
                "validation_fact_ids": [
                    str(row["fact_id"])
                    for row in preparation_facts
                    if row["split"] == "validation"
                ],
            },
            "preparation_facts": {
                **_descriptor(staging / "preparation_facts.jsonl", staging),
                "count": len(preparation_facts),
            },
            "queries": {
                **_descriptor(staging / "queries.jsonl", staging),
                "count": len(queries),
            },
            "scorer": {
                **_descriptor(staging / "scorer.jsonl", staging),
                "count": len(scorer),
            },
            "blocks": {
                **_descriptor(staging / "block_map.jsonl", staging),
                "count": len(block_map),
                "seq_len": 128,
                "sentinel": "one unscored generic 128-token document per packed split",
            },
            "files": files,
        }
        manifest = {**manifest_content, "sha256": canonical_digest(manifest_content)}
        (staging / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        try:
            _rename_noreplace(staging, root)
        except FileExistsError:
            return _verify_existing(root, seed=seed, fact_count=fact_count)
        return root / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _padded_record_placeholder(
    tokenizer: Tokenizer | None, template_id: str, key: str, answer: str
) -> tuple[None, dict[str, object]]:
    """Address-only helper used before tokenizer construction.

    The address is solely an answer-prefix byte hash, so its pre-tokenizer audit is
    exact and cannot be influenced by labels or BPE behavior.
    """
    prefix = _answer_prefix(template_id, key)
    _assert_suffix_contract(prefix, key)
    return None, {"address": _address(prefix)}
