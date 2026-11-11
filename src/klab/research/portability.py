# Malformed portability bundles retain one ValueError boundary.
# ruff: noqa: TRY004
"""Verified inputs and model initialization for Engram portability runs."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors.torch import load_file

from klab.address_hash import byte_scheme, token_addressing_marker
from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig
from klab.engram.packs import verify_pack
from klab.model.inspection import named_tensor_inventory
from klab.training.manifest import architecture_sha256, read_manifest, sha256_file

_FORMAT = "klab-portability-run"
_VERSION = 1
_SEEDS = (17, 41, 73)
_MEMORY_MODEL_FIELDS = frozenset(
    {
        "memory",
        "memory_injection",
        "memory_variant",
        "memory_gate_transform",
        "memory_addressing",
        "memory_table_sizes",
        "memory_hash_seed",
        "memory_token_mapping",
        "memory_token_mapping.algorithm",
        "memory_token_mapping.token_ids",
        "memory_token_mapping.vocab_size",
        "memory_token_mapping.memory_vocab_size",
        "memory_token_mapping.tokenizer_sha256",
        "memory_token_mapping.mapping_sha256",
        "memory_padding_token_id",
        "memory_independent_init",
        "memory_conv_kernel_size",
        "memory_block_index",
        "memory_block_indices",
        "memory_table_size",
        "memory_ngram_size",
        "memory_dim",
        "memory_package_path",
        "memory_ngram_orders",
        "memory_hash_heads",
        "semantic_memory_dim",
    }
)


@dataclass(frozen=True, slots=True)
class PortabilityRun:
    path: Path
    root: Path
    payload: dict[str, Any]
    backbone: dict[str, Any] | None
    memory: dict[str, Any]
    config: RunConfig | None = None

    @property
    def coordinate(self) -> dict[str, Any]:
        coordinate = self.payload["coordinate"]
        if self.payload.get("version") == 2:
            return {**coordinate, "seed": self.payload["seed"]}
        return coordinate


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(
            "portability asset path must be a nonempty POSIX relative path"
        )
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("portability asset path is unsafe")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("portability asset path traverses a symlink")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("portability asset escapes its manifest directory") from error
    return path


def _verify_file(root: Path, descriptor: object) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("file descriptor must contain path, sha256, and size_bytes")
    path = _relative_path(root, descriptor["path"])
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("portability file asset is not a regular file")
    expected_size = descriptor["size_bytes"]
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("portability file size_bytes must be a nonnegative integer")
    expected_digest = _digest(descriptor["sha256"], "file sha256")
    if info.st_size != expected_size or sha256_file(path) != expected_digest:
        raise ValueError(
            f"portability file asset failed integrity verification: {path}"
        )
    return path


def _verify_directory(root: Path, descriptor: object) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "files"}:
        raise ValueError("directory descriptor must contain path and files")
    directory = _relative_path(root, descriptor["path"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("portability directory asset is not a regular directory")
    files = descriptor["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("directory descriptor requires a nonempty file inventory")
    expected: dict[str, object] = {}
    for member in files:
        if not isinstance(member, dict) or set(member) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("directory member descriptor is invalid")
        member_path = member["path"]
        if not isinstance(member_path, str) or member_path in expected:
            raise ValueError("directory member path is invalid or duplicated")
        expected[member_path] = member
        _verify_file(directory, member)
    actual: set[str] = set()
    for item in directory.rglob("*"):
        if item.is_symlink():
            raise ValueError("portability directory contains a symlink")
        if item.is_file():
            actual.add(item.relative_to(directory).as_posix())
    if actual != set(expected):
        raise ValueError("portability directory inventory differs from its descriptor")
    return directory


def _verify_asset(root: Path, descriptor: object) -> Path:
    if isinstance(descriptor, dict) and set(descriptor) == {
        "path",
        "sha256",
        "size_bytes",
    }:
        return _verify_file(root, descriptor)
    return _verify_directory(root, descriptor)


def _verify_world_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("world manifest is invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "klab-portability-worlds"
    ):
        raise ValueError("unsupported portability world manifest")
    expected = _digest(payload.get("sha256"), "world manifest sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if canonical_digest(content) != expected:
        raise ValueError("portability world manifest hash mismatch")
    files = payload.get("files")
    if not isinstance(files, list):
        raise TypeError("world manifest file inventory must be an array")
    expected_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise TypeError("world manifest file descriptor is invalid")
        _verify_file(path.parent, entry)
        expected_paths.add(entry["path"])
    actual_paths: set[str] = set()
    for item in path.parent.rglob("*"):
        if item.is_symlink():
            raise ValueError("world assets contain a symlink")
        if item.is_file() and item != path:
            actual_paths.add(item.relative_to(path.parent).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("world manifest file inventory differs from its directory")
    return payload


def _load_portability_manifest_v1(config: RunConfig) -> PortabilityRun:
    """Validate the complete seed-selected portability input contract before training."""
    manifest_path = config.training.portability_manifest_path
    if manifest_path is None:
        raise ValueError("training.portability_manifest_path is not configured")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("portability run manifest must be a regular nonsymlink file")
    root = manifest_path.parent.resolve(strict=True)
    path = manifest_path.resolve(strict=True)
    if path.parent != root:
        raise ValueError("portability run manifest must reside in its asset root")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("portability run manifest is invalid JSON") from error
    required = {
        "format",
        "version",
        "protocol",
        "world_manifest",
        "coordinate",
        "seeds",
        "initial_backbone",
        "memory",
        "observations",
        "training_fact_ids",
        "sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("portability run manifest has an invalid top-level schema")
    if (
        payload["format"] != _FORMAT
        or type(payload["version"]) is not int
        or payload["version"] != _VERSION
    ):
        raise ValueError("unsupported portability run manifest version")
    expected = _digest(payload["sha256"], "portability run manifest sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if canonical_digest(content) != expected:
        raise ValueError("portability run manifest hash mismatch")
    if payload["seeds"] != list(_SEEDS) or config.seed not in _SEEDS:
        raise ValueError("portability seed axis must be exactly [17, 41, 73]")

    _verify_file(root, payload["protocol"])
    world_path = _verify_file(root, payload["world_manifest"])
    world = _verify_world_manifest(world_path)
    coordinate = payload["coordinate"]
    if not isinstance(coordinate, dict) or set(coordinate) != {
        "recipient",
        "representation",
        "condition",
    }:
        raise ValueError(
            "portability coordinate must name recipient, representation, and condition"
        )
    if not all(
        isinstance(coordinate[key], str) and coordinate[key] for key in coordinate
    ):
        raise ValueError("portability coordinate fields must be nonempty strings")
    representation = coordinate["representation"]
    condition = coordinate["condition"]
    if representation not in {"token", "byte", "semantic"}:
        raise ValueError("unknown portability representation")
    if condition not in {
        "adapter-tuned",
        "frozen-only",
        "joint",
        "native",
        "random",
        "corrupt",
        "disabled",
    }:
        raise ValueError("unsupported portability training condition")

    seed_key = str(config.seed)
    backbones = payload["initial_backbone"]
    memories = payload["memory"]
    seed_keys = {str(seed) for seed in _SEEDS}
    if not isinstance(backbones, dict) or set(backbones) != seed_keys:
        raise ValueError(
            "initial_backbone must be keyed by all declared decimal seed strings"
        )
    if not isinstance(memories, dict) or set(memories) != seed_keys:
        raise ValueError("memory must be keyed by all declared decimal seed strings")
    for candidate in backbones.values():
        if candidate is None:
            continue
        if not isinstance(candidate, dict) or set(candidate) != {
            "checkpoint",
            "checkpoint_sha256",
            "architecture_sha256",
            "tokenizer_sha256",
        }:
            raise ValueError("initial backbone entry has an invalid schema")
        _verify_directory(root, candidate["checkpoint"])
        for field in ("checkpoint_sha256", "architecture_sha256", "tokenizer_sha256"):
            _digest(candidate[field], field)
    backbone = backbones[seed_key]
    if (condition == "native") != (backbone is None):
        raise ValueError("only native-memory conditions may omit preparation backbones")

    evaluation_world_ids = {
        item["world_id"] for item in world["worlds"] if item["partition"] != "train"
    }
    for candidate in memories.values():
        if not isinstance(candidate, dict) or set(candidate) != {
            "kind",
            "artifact",
            "pack_id",
            "tensor_sha256",
            "addressing",
            "encoder_contract",
            "replacements",
        }:
            raise ValueError("memory entry has an invalid schema")
        if candidate["kind"] != representation:
            raise ValueError("memory kind differs from the declared representation")
        artifact_value = candidate["artifact"]
        native_local_table = condition == "native" and representation in {
            "token",
            "byte",
        }
        disabled = condition == "disabled"
        if (condition == "native" and representation == "semantic") or (
            artifact_value is None and not native_local_table and not disabled
        ):
            raise ValueError(
                "portability memory artifact is required for this condition"
            )
        memory_path = (
            None if artifact_value is None else _verify_asset(root, artifact_value)
        )
        if disabled:
            if any(
                candidate[field] is not None
                for field in (
                    "pack_id",
                    "tensor_sha256",
                    "addressing",
                    "encoder_contract",
                )
            ):
                raise ValueError(
                    "disabled conditions cannot declare an attached memory asset"
                )
        elif representation == "semantic":
            if (
                memory_path is None
                or candidate["pack_id"] is None
                or candidate["tensor_sha256"] is not None
                or not isinstance(candidate["encoder_contract"], dict)
                or candidate["addressing"] is not None
            ):
                raise ValueError(
                    "semantic memory must bind a verified pack and encoder contract"
                )
            pack_report = verify_pack(
                memory_path,
                expected_pack_id=_digest(candidate["pack_id"], "pack_id"),
            )
            if not pack_report.valid:
                raise ValueError("portability semantic pack failed verification")
        else:
            if (
                candidate["pack_id"] is not None
                or candidate["encoder_contract"] is not None
            ):
                raise ValueError(
                    "lexical memory must not claim a semantic pack contract"
                )
            if not isinstance(candidate["addressing"], dict):
                raise ValueError("lexical memory must bind its addressing identity")
            if artifact_value is None:
                if candidate["tensor_sha256"] is not None:
                    raise ValueError(
                        "native local memory must not pin a frozen table digest"
                    )
            else:
                _digest(candidate["tensor_sha256"], "memory tensor_sha256")
        replacements = candidate["replacements"]
        if not isinstance(replacements, dict):
            raise TypeError("memory replacements must be an object")
        if (
            not disabled
            and condition != "native"
            and set(replacements) != evaluation_world_ids
        ):
            raise ValueError(
                "replacement memory inventory must cover every evaluation world"
            )
        if (disabled or condition == "native") and replacements:
            raise ValueError(
                "conditions without transferred memory cannot declare replacements"
            )
        for replacement in replacements.values():
            if not isinstance(replacement, dict) or set(replacement) != {
                "artifact",
                "pack_id",
                "tensor_sha256",
            }:
                raise ValueError("replacement memory descriptor has an invalid schema")
            _verify_asset(root, replacement["artifact"])
            if representation == "semantic":
                _digest(replacement["pack_id"], "replacement pack_id")
                if replacement["tensor_sha256"] is not None:
                    raise ValueError(
                        "semantic replacements cannot declare tensor digests"
                    )
            else:
                if replacement["pack_id"] is not None:
                    raise ValueError("lexical replacements cannot declare pack ids")
                _digest(replacement["tensor_sha256"], "replacement tensor_sha256")
    memory = memories[seed_key]
    if condition != "native" and backbone is None:
        raise ValueError(
            "transferred portability conditions require a preparation checkpoint"
        )

    observations = payload["observations"]
    if not isinstance(observations, dict) or set(observations) != {
        "development",
        "final",
    }:
        raise ValueError(
            "observations must reference development and final case manifests"
        )
    for descriptor in observations.values():
        _verify_file(root, descriptor)
    training_fact_ids = payload["training_fact_ids"]
    if (
        not isinstance(training_fact_ids, list)
        or any(not isinstance(value, str) or not value for value in training_fact_ids)
        or len(training_fact_ids) != len(set(training_fact_ids))
    ):
        raise ValueError("training_fact_ids must be unique nonempty strings")
    training_world_ids = {
        fact_id
        for item in world["worlds"]
        if item["partition"] == "train"
        for fact_id in item["producer_fact_ids"]
    }
    if not set(training_fact_ids) <= training_world_ids:
        raise ValueError("portability manifest includes non-training-world facts")
    permitted_ids = {
        fact_id
        for item in world["worlds"]
        if item["partition"] == "train"
        for fact_id in (
            item["eligible_adapter_fact_ids"]
            if condition != "native"
            else item["producer_fact_ids"]
        )
    }
    if not set(training_fact_ids) <= permitted_ids:
        raise ValueError(
            "portability manifest includes recipient-ineligible fact associations"
        )
    tokenizer_digest = sha256_file(config.tokenizer.path)
    if backbone is not None and tokenizer_digest != backbone["tokenizer_sha256"]:
        raise ValueError(
            "recipient tokenizer differs from its preparation checkpoint identity"
        )
    return PortabilityRun(path, root, payload, backbone, memory)


_V2_FORMAT = "klab-learned-portability-data"
_V2_ADDRESSING = {
    "format_version": 1,
    "normalization": "raw-utf8-v1",
    "hashing": "poly257-terminal-v1",
    "ngram_size": 32,
    "table_size": 65521,
    "embedding_dim": 32,
}
_V2_ROLES = frozenset({"source", "preparation", "recipient", "native", "calibration"})
_V2_SCIENTIFIC_CONDITIONS = {
    "source": {"source-real", "source-dense"},
    "preparation": {"prepare"},
    "recipient": {
        "baseline",
        "constant",
        "random",
        "permuted",
        "real-zero-shot",
        "real-adapter",
    },
    "native": {"native"},
}
_V2_TIMING_CONDITIONS = frozenset(
    {
        "source-byte-128",
        "dense-128",
        "native-64",
        "preparation-64",
        "adapter-64",
        "adapter-128",
    }
)


def _v2_descriptor(
    root: Path, value: object, *, extra: frozenset[str] = frozenset()
) -> Path:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "size_bytes"} | extra
    ):
        raise ValueError("learned portability descriptor has an invalid schema")
    return _verify_file(
        root, {key: value[key] for key in ("path", "sha256", "size_bytes")}
    )


def _within(directory: Path, path: Path, name: str) -> None:
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise ValueError(f"{name} is not owned by the portability bundle") from error


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid JSONL") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{name} rows must be objects")
    return rows


def _verify_learned_data_manifest(path: Path) -> dict[str, Any]:
    data = _read_json(path, "learned portability data manifest")
    required = {
        "format",
        "version",
        "seed",
        "fact_count",
        "rng_derivation",
        "symbols",
        "addressing",
        "address_audit",
        "tokenizer",
        "templates",
        "ownership",
        "ownership_file",
        "facts",
        "preparation",
        "preparation_facts",
        "queries",
        "scorer",
        "blocks",
        "files",
        "sha256",
    }
    if set(data) != required or data["format"] != _V2_FORMAT or data["version"] != 1:
        raise ValueError("unsupported learned portability data manifest")
    expected = _digest(data["sha256"], "learned data manifest sha256")
    if (
        hashlib.sha256(
            canonical_json({k: v for k, v in data.items() if k != "sha256"})
        ).hexdigest()
        != expected
    ):
        raise ValueError("learned portability data manifest hash mismatch")
    if type(data["fact_count"]) is not int or data["fact_count"] <= 0:
        raise ValueError("learned portability fact_count is invalid")
    if data["addressing"] != {
        "kind": "raw-utf8",
        "hash": "poly257-terminal-v1",
        "table_size": 65521,
        "ngram_size": 32,
        "embedding_dim": 32,
        "answer_prefix_contract": "hash(raw UTF-8 last 32 bytes of answer_prefix); answer_prefix == prompt + ' '",
    }:
        raise ValueError("learned portability data addressing contract differs")
    files = data["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("learned portability data file inventory is invalid")
    expected_paths: set[str] = set()
    for descriptor in files:
        file_path = _v2_descriptor(path.parent, descriptor)
        relative = file_path.relative_to(path.parent).as_posix()
        if relative in expected_paths:
            raise ValueError("learned portability data inventory has duplicate paths")
        expected_paths.add(relative)
    actual_paths: set[str] = set()
    for item in path.parent.rglob("*"):
        if item.is_symlink():
            raise ValueError("learned portability data contains a symlink")
        if item.is_file() and item != path:
            actual_paths.add(item.relative_to(path.parent).as_posix())
    if actual_paths != expected_paths:
        raise ValueError(
            "learned portability data inventory differs from its directory"
        )
    facts_path = _v2_descriptor(path.parent, data["facts"], extra=frozenset({"count"}))
    queries_path = _v2_descriptor(
        path.parent, data["queries"], extra=frozenset({"count"})
    )
    scorer_path = _v2_descriptor(
        path.parent, data["scorer"], extra=frozenset({"count"})
    )
    facts, queries, scorer = (
        _read_jsonl(facts_path, "learned facts"),
        _read_jsonl(queries_path, "learned queries"),
        _read_jsonl(scorer_path, "learned scorer"),
    )
    if any(
        data[key]["count"] != len(rows)
        for key, rows in (("facts", facts), ("queries", queries), ("scorer", scorer))
    ):
        raise ValueError("learned data descriptor count differs from content")
    fact_ids: set[str] = set()
    fact_by_id: dict[str, dict[str, Any]] = {}
    ownership_ids: dict[str, set[str]] = {
        name: set() for name in ("calibration", "source_monitor", "held_out")
    }
    expected_roles = {
        "calibration": ["recipient_calibration", "source_training"],
        "source_monitor": ["source_monitor", "source_training"],
        "held_out": ["held_out_evaluation", "source_training"],
    }
    for fact in facts:
        required_fact = {
            "fact_id",
            "ownership",
            "key",
            "nonce",
            "target_row",
            "assigned_symbol",
            "provenance",
            "permitted_roles",
            "content_digest_sha256",
        }
        if (
            set(fact) != required_fact
            or not isinstance(fact["fact_id"], str)
            or fact["fact_id"] in fact_ids
        ):
            raise ValueError("learned fact schema or identity is invalid")
        ownership = fact["ownership"]
        provenance = fact["provenance"]
        if (
            ownership not in ownership_ids
            or sorted(fact["permitted_roles"]) != expected_roles[ownership]
            or not isinstance(provenance, dict)
            or provenance.get("source") != "generated"
            or provenance.get("license") != "CC0-1.0"
            or not isinstance(fact["key"], str)
            or type(fact["nonce"]) is not int
        ):
            raise ValueError("learned fact ownership is invalid")
        if (
            not isinstance(fact["assigned_symbol"], str)
            or len(fact["assigned_symbol"]) != 1
        ):
            raise ValueError("learned fact symbol is invalid")
        if type(fact["target_row"]) is not int or not 0 < fact["target_row"] < 65521:
            raise ValueError("learned fact target row is invalid")
        content = {
            key: value for key, value in fact.items() if key != "content_digest_sha256"
        }
        if canonical_digest(content) != _digest(
            fact["content_digest_sha256"], "fact content digest"
        ):
            raise ValueError("learned fact content digest mismatch")
        fact_ids.add(fact["fact_id"])
        fact_by_id[fact["fact_id"]] = fact
        ownership_ids[ownership].add(fact["fact_id"])
    if len(facts) != data["fact_count"] or len(
        {row["target_row"] for row in facts}
    ) != len(facts):
        raise ValueError("learned facts do not have unique declared target rows")
    ownership = data["ownership"]
    if not isinstance(ownership, dict) or set(ownership) != set(ownership_ids):
        raise ValueError("learned ownership manifest is invalid")
    for name, ids in ownership_ids.items():
        item = ownership[name]
        if (
            not isinstance(item, dict)
            or set(item) != {"count", "fact_ids", "permitted_roles"}
            or item["count"] != len(ids)
            or item["fact_ids"]
            != [row["fact_id"] for row in facts if row["ownership"] == name]
        ):
            raise ValueError("learned ownership facts differ from manifest")
    ownership_file = _v2_descriptor(path.parent, data["ownership_file"])
    if _read_json(ownership_file, "learned ownership manifest") != ownership:
        raise ValueError("learned ownership file differs from the data manifest")
    fact_target_rows = {int(row["target_row"]) for row in facts}
    preparation = data["preparation"]
    prep_path = _v2_descriptor(
        path.parent, data["preparation_facts"], extra=frozenset({"count"})
    )
    prep_facts = _read_jsonl(prep_path, "preparation facts")
    expected_training_ids = [f"prep-training-{index:04d}" for index in range(512)]
    expected_validation_ids = [f"prep-validation-{index:04d}" for index in range(128)]
    if (
        not isinstance(preparation, dict)
        or set(preparation) != {"training_fact_ids", "validation_fact_ids"}
        or preparation["training_fact_ids"] != expected_training_ids
        or preparation["validation_fact_ids"] != expected_validation_ids
        or data["preparation_facts"]["path"] != "preparation_facts.jsonl"
        or data["preparation_facts"]["count"] != 640
        or len(prep_facts) != 640
    ):
        raise ValueError("learned preparation fact inventory differs")
    prep_ids: set[str] = set()
    prep_rows: set[int] = set()
    prep_keys: set[str] = set()
    fact_keys = {str(row["key"]) for row in facts}
    for index, fact in enumerate(prep_facts):
        split = "training" if index < 512 else "validation"
        expected_id = (
            expected_training_ids[index]
            if split == "training"
            else expected_validation_ids[index - 512]
        )
        if (
            not isinstance(fact, dict)
            or set(fact)
            != {
                "fact_id",
                "split",
                "key",
                "nonce",
                "target_row",
                "assigned_symbol",
                "provenance",
                "content_digest_sha256",
            }
            or fact["fact_id"] != expected_id
            or fact["split"] != split
            or not isinstance(fact["key"], str)
            or re.fullmatch(r"k[0-9a-f]{12}\.[0-9a-f]{4}", fact["key"]) is None
            or type(fact["nonce"]) is not int
            or fact["nonce"] != int(fact["key"].rsplit(".", 1)[1], 16)
            or type(fact["target_row"]) is not int
            or not 0 < fact["target_row"] < 65521
            or fact["assigned_symbol"] not in data["symbols"]
            or fact["provenance"]
            != {
                "source": "generated",
                "license": "CC0-1.0",
                "namespace": "learned-engram-portability-v1|data-v1",
                "purpose": "preparation_only",
            }
        ):
            raise ValueError("learned preparation fact schema or identity differs")
        content = {
            key: value for key, value in fact.items() if key != "content_digest_sha256"
        }
        if canonical_digest(content) != _digest(
            fact["content_digest_sha256"], "preparation fact content digest"
        ):
            raise ValueError("learned preparation fact digest mismatch")
        fact_id, target_row, key = (
            str(fact["fact_id"]),
            int(fact["target_row"]),
            str(fact["key"]),
        )
        if (
            fact_id in prep_ids
            or target_row in prep_rows
            or target_row in fact_target_rows
            or key in prep_keys
            or key in fact_keys
        ):
            raise ValueError("learned preparation fact overlaps another data owner")
        prep_ids.add(fact_id)
        prep_rows.add(target_row)
        prep_keys.add(key)
    query_ids: set[str] = set()
    for row in queries:
        if (
            set(row)
            != {
                "case_id",
                "fact_id",
                "ownership",
                "template_id",
                "prompt",
                "answer_prefix",
                "address",
            }
            or row["fact_id"] not in fact_ids
            or row["ownership"] != fact_by_id[row["fact_id"]]["ownership"]
            or not isinstance(row["template_id"], str)
            or type(row["address"]) is not int
            or any("expected" in key or "symbol" in key for key in row)
        ):
            raise ValueError("learned query schema leaks labels or identities")
        if not isinstance(row["case_id"], str) or row["case_id"] in query_ids:
            raise ValueError("learned query IDs are invalid")
        query_ids.add(row["case_id"])
    scorer_ids: set[str] = set()
    for row in scorer:
        if (
            set(row) != {"case_id", "fact_id", "expected_symbol"}
            or row["fact_id"] not in fact_ids
            or row["case_id"] in scorer_ids
            or row["expected_symbol"] != fact_by_id[row["fact_id"]]["assigned_symbol"]
        ):
            raise ValueError("learned scorer schema is invalid")
        scorer_ids.add(row["case_id"])
    if query_ids != scorer_ids or len(query_ids) != len(queries):
        raise ValueError("learned query/scorer IDs differ")
    return {
        "payload": data,
        "fact_ids": fact_ids,
        "ownership": ownership_ids,
        "preparation_fact_ids": prep_ids,
    }


def _load_portability_manifest_v2(
    config: RunConfig, path: Path, root: Path, payload: dict[str, Any]
) -> PortabilityRun:
    required = {
        "format",
        "version",
        "experiment",
        "protocol",
        "world_manifest",
        "coordinate",
        "seed",
        "initialization",
        "initial_backbone",
        "memory",
        "source",
        "observations",
        "training_fact_ids",
        "sha256",
    }
    if (
        set(payload) != required
        or payload["format"] != _FORMAT
        or payload["version"] != 2
        or payload["experiment"] != "learned-engram-portability-v1"
    ):
        raise ValueError("learned portability v2 manifest has an invalid schema")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if (
        canonical_digest(body)
        != _digest(payload["sha256"], "portability run manifest sha256")
        or path.read_bytes() != canonical_json(payload) + b"\n"
    ):
        raise ValueError("learned portability run manifest hash or encoding differs")

    bundle = root / "portability"
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("learned portability bundle is missing or symlinked")
    protocol_path = _v2_descriptor(root, payload["protocol"])
    world_path = _v2_descriptor(root, payload["world_manifest"])
    if (
        payload["protocol"]["path"] != "portability/protocol.json"
        or payload["world_manifest"]["path"] != "portability/data/manifest.json"
    ):
        raise ValueError("learned portability protocol/world paths are invalid")
    _within(bundle, protocol_path, "protocol")
    _within(bundle, world_path, "world manifest")
    protocol = _read_json(protocol_path, "learned portability protocol")
    if (
        protocol.get("experiment") != "learned-engram-portability-v1"
        or protocol.get("format") != "klab-portability-protocol"
        or type(protocol.get("version")) is not int
        or protocol["version"] != 2
    ):
        raise ValueError("learned portability protocol identity differs")
    data = _verify_learned_data_manifest(world_path)

    coordinate = payload["coordinate"]
    if (
        not isinstance(coordinate, dict)
        or set(coordinate) != {"role", "recipient", "representation", "condition"}
        or coordinate.get("representation") != "byte"
    ):
        raise ValueError("learned portability coordinate schema is invalid")
    role, recipient, condition = (
        coordinate["role"],
        coordinate["recipient"],
        coordinate["condition"],
    )
    seed = payload["seed"]
    init = payload["initialization"]
    if not isinstance(init, dict) or set(init) != {
        "purpose",
        "derivation_version",
        "pair_seed",
        "width",
        "families",
    }:
        raise ValueError("learned portability initialization schema is invalid")
    purpose = init["purpose"]
    is_timing = purpose == "timing"
    if purpose not in {"scientific", "timing"}:
        raise ValueError("learned portability initialization purpose is invalid")
    if is_timing:
        if (
            role != "calibration"
            or condition not in _V2_TIMING_CONDITIONS
            or type(seed) is not int
            or seed != 20260925
            or config.seed != seed
        ):
            raise ValueError("timing manifest coordinate or seed is invalid")
        expected_families = {
            "source-byte-128": ["source-backbone", "source-memory"],
            "dense-128": ["source-backbone"],
            "native-64": ["native-memory"],
            "preparation-64": ["preparation-backbone"],
            "adapter-64": ["recipient-adapter"],
            "adapter-128": ["recipient-adapter"],
        }[condition]
    else:
        if (
            role not in _V2_SCIENTIFIC_CONDITIONS
            or condition not in _V2_SCIENTIFIC_CONDITIONS[role]
            or type(seed) is not int
            or seed not in _SEEDS
            or config.seed != seed
        ):
            raise ValueError("learned scientific role, condition, or seed is invalid")
        expected_families = {
            ("source", "source-real"): ["source-backbone", "source-memory"],
            ("source", "source-dense"): ["source-backbone"],
            ("preparation", "prepare"): ["preparation-backbone"],
            ("recipient", "baseline"): ["recipient-adapter"],
            ("recipient", "constant"): ["recipient-adapter"],
            ("recipient", "random"): ["recipient-adapter"],
            ("recipient", "permuted"): ["recipient-adapter"],
            ("recipient", "real-zero-shot"): ["recipient-adapter"],
            ("recipient", "real-adapter"): ["recipient-adapter"],
            ("native", "native"): ["native-memory"],
        }[(role, condition)]
    width = (
        128
        if recipient == "source"
        else 64
        if recipient == "width64"
        else 128
        if recipient == "width128"
        else 0
    )
    if (
        recipient not in {"source", "width64", "width128"}
        or type(init["pair_seed"]) is not int
        or init["pair_seed"] != seed
        or type(init["width"]) is not int
        or init["width"] != width
        or init["derivation_version"] != "init-v1"
        or init["families"] != expected_families
    ):
        raise ValueError("learned per-tensor initialization identity differs")

    memory = payload["memory"]
    memory_fields = {
        "kind",
        "artifact",
        "pack_id",
        "tensor_sha256",
        "addressing",
        "encoder_contract",
        "replacements",
    }
    if (
        not isinstance(memory, dict)
        or set(memory) != memory_fields
        or memory["kind"] != "byte"
        or memory["pack_id"] is not None
        or memory["encoder_contract"] is not None
        or memory["replacements"] != []
        or memory["addressing"] != _V2_ADDRESSING
    ):
        raise ValueError("learned byte-memory identity is invalid")
    transferred = (
        not is_timing
        and role == "recipient"
        and condition
        in {
            "constant",
            "random",
            "permuted",
            "real-zero-shot",
            "real-adapter",
        }
    )
    artifact = memory["artifact"]
    expected_model_memory = (
        "portable"
        if transferred
        else "byte"
        if (role in {"source", "native"} and condition in {"source-real", "native"})
        or (
            is_timing
            and condition
            in {"source-byte-128", "native-64", "adapter-64", "adapter-128"}
        )
        else "none"
    )
    if config.model.memory != expected_model_memory:
        raise ValueError("learned model attachment differs from coordinate")
    if transferred:
        artifact_path = _v2_descriptor(root, artifact)
        if not artifact_path.is_relative_to(bundle):
            raise ValueError(
                "learned memory artifact is outside owned portability assets"
            )
        table_sha = _digest(memory["tensor_sha256"], "learned memory tensor sha256")
        from klab.model.portable_engram import load_portable_engram

        package = load_portable_engram(
            artifact_path, expected_shape=(65521, 32), expected_ngram_size=32
        )
        if package.manifest.table_sha256 != table_sha:
            raise ValueError("learned package table digest differs")
        if (
            config.model.memory_package_path is None
            or config.model.memory_package_path.resolve(strict=True)
            != artifact_path.resolve(strict=True)
        ):
            raise ValueError("learned package path differs from owned memory artifact")
    elif artifact is not None or memory["tensor_sha256"] is not None:
        raise ValueError("nontransferred learned memory must not import a package")

    observations = payload["observations"]
    if not isinstance(observations, dict) or set(observations) != {
        "schedule",
        "queries",
        "scorer",
        "ownership",
    }:
        raise ValueError("learned observation bindings are invalid")
    schedule_path = _v2_descriptor(root, observations["schedule"])
    query_path = _v2_descriptor(root, observations["queries"])
    scorer_path = _v2_descriptor(root, observations["scorer"])
    ownership_path = _v2_descriptor(root, observations["ownership"])
    for item, name in (
        (schedule_path, "observation schedule"),
        (query_path, "query corpus"),
        (scorer_path, "scorer corpus"),
        (ownership_path, "ownership file"),
    ):
        _within(bundle, item, name)
    data_root = world_path.parent
    if (
        query_path != data_root / data["payload"]["queries"]["path"]
        or scorer_path != data_root / data["payload"]["scorer"]["path"]
        or ownership_path != data_root / data["payload"]["ownership_file"]["path"]
    ):
        raise ValueError(
            "learned observation data bindings differ from the world manifest"
        )
    schedule = _read_json(schedule_path, "learned observation schedule")
    if (
        schedule.get("format") != "klab-portability-observation-schedule"
        or schedule.get("version") != 1
        or schedule.get("role") != role
        or schedule.get("condition") != condition
    ):
        raise ValueError("learned observation schedule coordinate differs")

    backbone = payload["initial_backbone"]
    needs_backbone = not is_timing and (role == "recipient" or role == "native")
    if needs_backbone:
        if not isinstance(backbone, dict) or set(backbone) != {
            "checkpoint",
            "checkpoint_sha256",
            "architecture_sha256",
            "tokenizer_sha256",
            "run_manifest",
        }:
            raise ValueError("recipient/native preparation backbone is missing")
        checkpoint = _verify_directory(root, backbone["checkpoint"])
        run_manifest = _v2_descriptor(root, backbone["run_manifest"])
        _within(bundle, checkpoint, "preparation checkpoint")
        _within(bundle, run_manifest, "preparation run manifest")
        if sha256_file(config.tokenizer.path) != _digest(
            backbone["tokenizer_sha256"], "preparation tokenizer sha256"
        ):
            raise ValueError("recipient tokenizer differs from preparation")
        if run_manifest != checkpoint.parents[1] / "manifest.json":
            raise ValueError("preparation run manifest is not checkpoint parent")
        checkpoint_manifest = _read_json(
            checkpoint / "manifest.json", "preparation checkpoint manifest"
        )
        prepared_manifest_payload = _read_json(run_manifest, "preparation run manifest")
        prepared_manifest = read_manifest(run_manifest)
        if (
            checkpoint_manifest.get("sha256")
            != _digest(backbone["checkpoint_sha256"], "preparation checkpoint sha256")
            or checkpoint_manifest.get("manifest_sha256")
            != prepared_manifest_payload.get("sha256")
            or checkpoint_manifest.get("architecture_sha256")
            != _digest(
                backbone["architecture_sha256"], "preparation architecture sha256"
            )
            or prepared_manifest.get("architecture_sha256")
            != backbone["architecture_sha256"]
        ):
            raise ValueError("preparation checkpoint binding differs")
    elif backbone is not None:
        raise ValueError("source/preparation/timing runs cannot bind a backbone")

    source = payload["source"]
    if transferred:
        source_fields = {
            "run_id",
            "checkpoint_sha256",
            "architecture_sha256",
            "table_sha256",
            "gate",
            "run_manifest",
            "checkpoint_manifest",
            "audit",
            "provenance",
        }
        if not isinstance(source, dict) or set(source) != source_fields:
            raise ValueError("transferred recipient lacks source gate provenance")
        for key in (
            "run_manifest",
            "checkpoint_manifest",
            "audit",
            "provenance",
            "gate",
        ):
            file_path = _v2_descriptor(root, source[key])
            _within(bundle, file_path, f"source {key}")
        gate = _read_json(_v2_descriptor(root, source["gate"]), "source gate")
        if gate.get("status") != "passed":
            raise ValueError("recipient source provenance does not pass all seed gates")
        _digest(source["checkpoint_sha256"], "source checkpoint sha256")
        _digest(source["architecture_sha256"], "source architecture sha256")
        _digest(source["table_sha256"], "source table sha256")
    elif source is not None:
        raise ValueError("nontransferred learned coordinate cannot bind a source")

    all_fact_ids = data["fact_ids"]
    ownership_ids = data["ownership"]
    if is_timing:
        if condition in {"source-byte-128", "dense-128", "native-64"}:
            expected_ids = all_fact_ids
        elif condition in {"adapter-64", "adapter-128"}:
            expected_ids = ownership_ids["calibration"]
        else:
            expected_ids = set(data["payload"]["preparation"]["training_fact_ids"])
    elif role in {"source", "native"}:
        expected_ids = all_fact_ids
    elif role == "preparation":
        expected_ids = set(data["payload"]["preparation"]["training_fact_ids"])
    elif condition in {"constant", "random", "permuted", "real-adapter"}:
        expected_ids = ownership_ids["calibration"]
    else:
        expected_ids = set()
    training = payload["training_fact_ids"]
    if (
        not isinstance(training, list)
        or any(not isinstance(item, str) for item in training)
        or len(training) != len(set(training))
        or set(training) != expected_ids
    ):
        raise ValueError("learned training facts differ from permitted ownership")

    expected_paths = {
        str(payload[key]["path"]) for key in ("protocol", "world_manifest")
    }
    expected_paths.add(str(observations["schedule"]["path"]))
    expected_paths.update(
        f"{world_path.parent.relative_to(root).as_posix()}/{entry['path']}"
        for entry in data["payload"]["files"]
    )
    if transferred:
        expected_paths.add(str(artifact["path"]))
        expected_paths.update(
            str(source[key]["path"])
            for key in (
                "gate",
                "run_manifest",
                "checkpoint_manifest",
                "audit",
                "provenance",
            )
        )
    if needs_backbone:
        expected_paths.add(str(backbone["run_manifest"]["path"]))
        expected_paths.update(
            f"{backbone['checkpoint']['path']}/{entry['path']}"
            for entry in backbone["checkpoint"]["files"]
        )
    actual_paths = {
        item.relative_to(root).as_posix()
        for item in bundle.rglob("*")
        if item.is_file() and not item.is_symlink()
    }
    if actual_paths != expected_paths:
        raise ValueError("learned portability bundle inventory differs from bindings")
    tokenizer_sha = data["payload"]["tokenizer"]["sha256"]
    if (
        config.tokenizer.path.resolve(strict=True)
        != (root / "tokenizer.json").resolve(strict=True)
        or sha256_file(config.tokenizer.path) != tokenizer_sha
    ):
        raise ValueError("learned run tokenizer differs from owned data tokenizer")
    return PortabilityRun(
        path,
        root,
        payload,
        backbone if isinstance(backbone, dict) else None,
        memory,
        config,
    )


def load_portability_manifest(config: RunConfig) -> PortabilityRun:
    """Dispatch exact portability manifest versions before applying their schemas."""
    manifest_path = config.training.portability_manifest_path
    if manifest_path is None:
        raise ValueError("training.portability_manifest_path is not configured")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("portability run manifest must be a regular nonsymlink file")
    root, path = (
        manifest_path.parent.resolve(strict=True),
        manifest_path.resolve(strict=True),
    )
    payload = _read_json(path, "portability run manifest")
    if payload.get("format") != _FORMAT or type(payload.get("version")) is not int:
        raise ValueError("unsupported portability run manifest version")
    if payload["version"] == 1:
        return _load_portability_manifest_v1(config)
    if payload["version"] == 2:
        return _load_portability_manifest_v2(config, path, root, payload)
    raise ValueError("unsupported portability run manifest version")


def _nonmemory_name(name: str) -> bool:
    return not name.startswith(("memory.", "semantic_memories."))


def initialize_recipient_backbone(
    model: torch.nn.Module, config: RunConfig, run: PortabilityRun
) -> dict[str, torch.Tensor] | None:
    """Load only verified non-memory tensors from this recipient's own checkpoint."""
    if run.backbone is None:
        return None
    from klab.training.checkpoints import CheckpointManager

    descriptor = run.backbone["checkpoint"]
    checkpoint = _verify_directory(run.root, descriptor)
    manager = CheckpointManager(checkpoint.parents[1])
    snapshot = manager.load(checkpoint, mode="promote")
    if snapshot.checkpoint_sha256 != run.backbone["checkpoint_sha256"]:
        raise ValueError("recipient preparation checkpoint digest mismatch")
    checkpoint_manifest = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    run_manifest_path = checkpoint.parents[1] / "manifest.json"
    run_manifest = read_manifest(run_manifest_path)
    run_manifest_digest = json.loads(run_manifest_path.read_text(encoding="utf-8"))[
        "sha256"
    ]
    if checkpoint_manifest.get("manifest_sha256") != run_manifest_digest:
        raise ValueError("recipient checkpoint is not bound to its run manifest")
    prepared_config = run_manifest.get("effective_config")
    if not isinstance(prepared_config, dict):
        raise TypeError("recipient run manifest has no effective configuration")
    if (
        architecture_sha256(prepared_config) != run.backbone["architecture_sha256"]
        or checkpoint_manifest.get("architecture_sha256")
        != run.backbone["architecture_sha256"]
    ):
        raise ValueError("recipient preparation architecture identity mismatch")
    checkpoint_config = RunConfig.model_validate(prepared_config)
    if checkpoint_config.attention.model_dump(
        mode="json"
    ) != config.attention.model_dump(mode="json"):
        raise ValueError("recipient preparation attention configuration mismatch")
    prepared_model = checkpoint_config.model.model_dump(mode="json")
    recipient_model = config.model.model_dump(mode="json")
    for field in _MEMORY_MODEL_FIELDS:
        prepared_model.pop(field, None)
        recipient_model.pop(field, None)
    if prepared_model != recipient_model:
        raise ValueError("recipient preparation backbone configuration mismatch")
    expected = {
        name: spec
        for name, spec in named_tensor_inventory(config.model, config.attention).items()
        if _nonmemory_name(name)
    }
    prepared = {
        name: tensor for name, tensor in snapshot.model.items() if _nonmemory_name(name)
    }
    if set(prepared) != set(expected):
        raise ValueError(
            "recipient preparation non-memory tensor names or aliases differ"
        )
    for name, spec in expected.items():
        if tuple(prepared[name].shape) != spec.shape:
            raise ValueError(f"recipient preparation tensor shape mismatch: {name}")
    model_state = model.state_dict()
    current_nonmemory = {name for name in model_state if _nonmemory_name(name)}
    if current_nonmemory != set(expected):
        raise ValueError("recipient model non-memory tensor inventory differs")
    result = model.load_state_dict(prepared, strict=False)
    allowed_missing = {name for name in model_state if not _nonmemory_name(name)}
    if result.unexpected_keys or set(result.missing_keys) != allowed_missing:
        raise ValueError("recipient checkpoint attachment inventory differs")
    return prepared


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def initialize_memory_artifact(
    model: torch.nn.Module, config: RunConfig, run: PortabilityRun
) -> None:
    """Load or verify the manifest-selected frozen memory attachment."""
    if run.payload.get("version") == 2:
        from klab.model.memory import ByteAddressMemory
        from klab.model.portable_engram import (
            PortableEngramAdapter,
            load_portable_engram,
        )

        memory = run.memory
        expected_addressing = _V2_ADDRESSING
        if memory["addressing"] != expected_addressing:
            raise ValueError("learned memory address contract differs from protocol")
        if config.model.memory == "none":
            if (
                model.memory is not None
                or memory["artifact"] is not None
                or memory["tensor_sha256"] is not None
            ):
                raise ValueError("memory-free coordinate attached transferred memory")
            return
        if (
            config.model.memory_table_size != 65521
            or config.model.memory_ngram_size != 32
            or config.model.memory_dim != 32
        ):
            raise ValueError("learned byte memory configuration differs from protocol")
        if config.model.memory == "byte":
            if (
                not isinstance(model.memory, ByteAddressMemory)
                or memory["artifact"] is not None
                or memory["tensor_sha256"] is not None
            ):
                raise ValueError("local byte memory must be a fresh trainable table")
            return
        if (
            config.model.memory != "portable"
            or not isinstance(model.memory, PortableEngramAdapter)
            or memory["artifact"] is None
        ):
            raise ValueError("transferred learned memory requires a portable adapter")
        package_path = _v2_descriptor(run.root, memory["artifact"])
        if (
            config.model.memory_package_path is None
            or config.model.memory_package_path.resolve(strict=True)
            != package_path.resolve(strict=True)
        ):
            raise ValueError("learned model package path differs from owned artifact")
        package = load_portable_engram(
            package_path,
            expected_shape=(65521, 32),
            expected_ngram_size=32,
        )
        expected_digest = _digest(memory["tensor_sha256"], "learned table sha256")
        if (
            package.manifest.table_sha256 != expected_digest
            or _tensor_digest(package.table) != expected_digest
            or _tensor_digest(model.memory.embedding.weight) != expected_digest
        ):
            raise ValueError("learned portable table bytes differ from source package")
        return

    memory = run.memory
    kind = memory["kind"]
    descriptor = memory["artifact"]
    if descriptor is None:
        condition = run.payload["coordinate"]["condition"]
        if condition == "disabled":
            if config.model.memory != "none" or model.memory is not None:
                raise ValueError("disabled condition must not attach lexical memory")
            semantic_modules = getattr(model, "semantic_memories", None)
            if semantic_modules is not None and len(semantic_modules):
                raise ValueError("disabled condition must not attach semantic memory")
            return
        if condition != "native" or kind not in {"token", "byte"}:
            raise ValueError(
                "this portability condition requires a frozen memory artifact"
            )
        if kind == "token":
            if config.model.memory != "ngram" or model.memory is None:
                raise ValueError(
                    "native token memory requires a token n-gram attachment"
                )
            expected = {
                "tokenizer_sha256": sha256_file(config.tokenizer.path),
                "order": config.model.memory_ngram_size,
                "hash_heads": config.model.memory_hash_heads,
                "rows": config.model.memory_table_size,
                "embedding_dim": config.model.memory_dim,
                **token_addressing_marker(
                    config.model.memory_table_size, config.model.memory_hash_heads
                ),
            }
        else:
            if config.model.memory != "byte" or model.memory is None:
                raise ValueError(
                    "native byte memory requires a local byte-table attachment"
                )
            expected = {
                "format_version": 1,
                "normalization": "raw-utf8-v1",
                "hashing": byte_scheme(config.model.memory_table_size),
                "ngram_size": config.model.memory_ngram_size,
                "table_size": config.model.memory_table_size,
                "embedding_dim": config.model.memory_dim,
            }
        if memory["addressing"] != expected:
            raise ValueError(
                "native memory addressing identity differs from recipient config"
            )
        return

    artifact = _verify_asset(run.root, descriptor)
    if kind == "token":
        if config.model.memory != "ngram" or model.memory is None:
            raise ValueError(
                "token portability requires a token n-gram memory attachment"
            )
        loaded = load_file(artifact, device="cpu")
        if set(loaded) != {"memory.table.weight"}:
            raise ValueError(
                "token memory safetensors must contain only memory.table.weight"
            )
        table = loaded["memory.table.weight"]
        module_table = model.memory.table.weight
        if table.dtype != module_table.dtype or tuple(table.shape) != tuple(
            module_table.shape
        ):
            raise ValueError(
                "token memory table shape or dtype differs from recipient configuration"
            )
        if _tensor_digest(table) != memory["tensor_sha256"]:
            raise ValueError("token memory table tensor digest mismatch")
        with torch.no_grad():
            module_table.copy_(table.to(device=module_table.device))
        expected = {
            "tokenizer_sha256": sha256_file(config.tokenizer.path),
            "order": config.model.memory_ngram_size,
            "hash_heads": config.model.memory_hash_heads,
            "rows": config.model.memory_table_size,
            "embedding_dim": config.model.memory_dim,
            **token_addressing_marker(
                config.model.memory_table_size, config.model.memory_hash_heads
            ),
        }
        if memory["addressing"] != expected:
            raise ValueError(
                "token memory addressing identity differs from recipient config"
            )
        return
    if kind == "byte":
        from klab.model.portable_engram import load_portable_engram

        if config.model.memory != "byte" or model.memory is None:
            raise ValueError("transferred byte memory requires a byte-table attachment")
        package = load_portable_engram(
            artifact,
            expected_shape=(config.model.memory_table_size, config.model.memory_dim),
            expected_ngram_size=config.model.memory_ngram_size,
        )
        if package.manifest.table_sha256 != memory["tensor_sha256"]:
            raise ValueError("portable byte table tensor digest differs from manifest")
        table = model.memory.table.weight
        if (
            tuple(table.shape) != tuple(package.table.shape)
            or table.dtype != package.table.dtype
        ):
            raise ValueError(
                "portable byte table shape or dtype differs from recipient configuration"
            )
        if _tensor_digest(package.table) != memory["tensor_sha256"]:
            raise ValueError("portable byte table tensor digest mismatch")
        with torch.no_grad():
            table.copy_(package.table.to(device=table.device))
        expected = {
            "format_version": 1,
            "normalization": "raw-utf8-v1",
            "hashing": byte_scheme(config.model.memory_table_size),
            "ngram_size": config.model.memory_ngram_size,
            "table_size": config.model.memory_table_size,
            "embedding_dim": config.model.memory_dim,
        }
        if memory["addressing"] != expected:
            raise ValueError(
                "portable byte addressing identity differs from recipient config"
            )
        return
    if kind == "semantic":
        if "allocation" not in model.semantic_memories:
            raise ValueError(
                "semantic portability requires the verified allocation attachment"
            )
        adapter = model.semantic_memories["allocation"]
        retriever = adapter.retriever
        if retriever.pack_id != memory["pack_id"]:
            raise ValueError("attached semantic pack differs from manifest identity")
        contract = memory["encoder_contract"]
        actual = {
            "key_encoder": retriever.key_encoder.model_dump(mode="json"),
            "value_encoder": retriever.value_encoder.model_dump(mode="json"),
            "key_dim": retriever.key_dim,
            "value_dim": retriever.memory_dim,
            "normalization": retriever.key_normalization,
            "representation_space_id": retriever.space_id,
        }
        if contract != actual:
            raise ValueError("semantic encoder contract differs from verified pack")
        return
    raise ValueError(f"unsupported portability memory kind: {kind}")


def apply_trainable_parameter_filter(
    model: torch.nn.Module,
    config: RunConfig,
    run: PortabilityRun | None = None,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Freeze every nonselected storage and return optimizer parameters by canonical name."""
    names = config.training.trainable_parameters
    if names is None:
        return tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    inventory = named_tensor_inventory(
        config.model,
        config.attention,
        trainable_parameters=names,
    )
    if (
        run is not None
        and run.memory["kind"] == "token"
        and run.memory["artifact"] is not None
        and run.payload["coordinate"]["condition"] != "joint"
        and "memory.table.weight" in names
    ):
        raise ValueError(
            "a transferred token table is trainable only in the joint condition"
        )
    parameters = dict(model.named_parameters())
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name in names:
        if name not in parameters or not inventory[name].trainable:
            raise ValueError(
                f"configured trainable parameter is not a canonical model storage: {name}"
            )
        selected.append((name, parameters[name]))
    selected_names = set(names)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected_names)
    return tuple(selected)


def verify_portability_assets_unchanged(run: PortabilityRun) -> None:
    """Reverify every pinned run input after training or attachment-only evaluation."""
    payload = run.payload
    if payload.get("version") == 2:
        if run.config is None:
            raise ValueError(
                "learned portability run lacks its validated configuration"
            )
        current = _read_json(run.path, "learned portability run manifest")
        reloaded = _load_portability_manifest_v2(
            run.config, run.path, run.root, current
        )
        if reloaded.payload != payload:
            raise ValueError("learned portability run manifest changed after loading")
        return
    _verify_file(run.root, payload["protocol"])
    _verify_world_manifest(_verify_file(run.root, payload["world_manifest"]))
    for entry in payload["initial_backbone"].values():
        if entry is not None:
            _verify_directory(run.root, entry["checkpoint"])
    for entry in payload["memory"].values():
        if entry["artifact"] is not None:
            path = _verify_asset(run.root, entry["artifact"])
            if (
                entry["pack_id"] is not None
                and not verify_pack(path, expected_pack_id=entry["pack_id"]).valid
            ):
                raise ValueError("pinned semantic pack changed after training")
        for replacement in entry["replacements"].values():
            path = _verify_asset(run.root, replacement["artifact"])
            if (
                replacement["pack_id"] is not None
                and not verify_pack(path, expected_pack_id=replacement["pack_id"]).valid
            ):
                raise ValueError(
                    "pinned replacement semantic pack changed after training"
                )
    for descriptor in payload["observations"].values():
        _verify_file(run.root, descriptor)
