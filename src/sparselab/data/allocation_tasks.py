"""Build provenance-bound path-domain allocation profiles and query cards."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer

from sparselab.canonical import canonical_digest, canonical_json
from sparselab.config.models import DatasetConfig
from sparselab.data.allocation import (
    ALLOCATION_REGIMES,
    OWNER_HYBRID,
    OWNER_LEXICAL,
    OWNER_NEURAL,
    OWNER_SEMANTIC,
    build_allocation_manifest,
    load_semantic_retriever,
)
from sparselab.data.packing import (
    _collect,
    _source_documents,
    canonical_tokenizer_sha256,
)
from sparselab.data.tokenizer import load_tokenizer
from sparselab.data.toy_worlds import StructuredKey, frozen_encoders
from sparselab.engram.packs import (
    SemanticMetadata,
    _rename_noreplace,
    compile_pack,
)
from sparselab.evaluation.capabilities import _card_from_mapping, load_capability_card
from sparselab.training.manifest import sha256_file, source_identity

_LICENSE = "MIT; original synthetic path_domain_v1 corpus; see data/path_domain_v1/provenance.json"
_PACK_CREATED_AT = "2026-09-23T00:00:00Z"
_DOCUMENT_CAPS = {"train": 24, "validation": 8}
_TOKEN_CAPS = {"train": 16_384, "validation": 4_096}
_PROFILE_NEURAL_FRACTIONS = {
    "n100": 1.0,
    "n75": 0.75,
    "n50": 0.5,
    "n25": 0.25,
    "n0": 0.0,
}
_CARD_FILES = {
    "path-domain-acquisition-v1": ("path-domain-acquisition-v1.json", "train"),
    "path-domain-development-v1": ("path-domain-development-v1.json", "development"),
    "path-domain-frozen-v1": ("path-domain-frozen-v1.json", "frozen_evaluation"),
}


def _canonical_key(value: dict[str, Any]) -> str:
    return canonical_json(value).decode("utf-8")


def _key_identity(value: dict[str, Any]) -> tuple[str, str]:
    subject = _canonical_key(value)
    return subject, "path-domain-v1"


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"allocation input must be a regular nonsymlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"allocation JSON input must be an object: {path}")
    return value


def _read_audit(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("format") != "path_domain_oracle_audit_v1":
        raise ValueError("path-domain oracle audit format is unsupported")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("path-domain oracle audit has no records")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("matches_oracle") is not True
            or not isinstance(record.get("split"), str)
            or type(record.get("line")) is not int
            or not isinstance(record.get("semantic_key"), dict)
            or not isinstance(record.get("rendered_assistant_label"), str)
        ):
            raise ValueError("path-domain oracle audit record is malformed")
        identity = (record["split"], record["line"])
        if identity in result:
            raise ValueError("path-domain oracle audit repeats a split line")
        key = record["semantic_key"]
        if set(key) != {"argument", "operation", "path"}:
            raise ValueError("path-domain oracle semantic key has unsupported fields")
        result[identity] = record
    for split in ("train", "development", "frozen_evaluation"):
        lines = sorted(line for name, line in result if name == split)
        if lines != list(range(1, len(lines) + 1)):
            raise ValueError(
                f"path-domain oracle audit has noncontiguous {split} lines"
            )
    return result


def _source_key_maps(
    audit: dict[tuple[str, int], dict[str, Any]], key_encoder: Any
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    vectors: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, Any]] = {}
    for (split, line), item in sorted(audit.items()):
        record_id = f"path-domain-{split}-{line:02d}"
        record = {**item, "record_id": record_id}
        records[record_id] = record
        subject, relation = _key_identity(item["semantic_key"])
        vectors[record_id] = key_encoder.encode(StructuredKey(subject, relation))
    training_vectors = [
        vectors[f"path-domain-train-{line:02d}"] for line in range(1, 25)
    ]
    if len({value.tobytes() for value in training_vectors}) != len(training_vectors):
        raise ValueError(
            "path-domain structured-key encoder collides on training records"
        )
    return vectors, records


def _operation_from_identifier(identifier: str) -> str:
    normalized = f"-{identifier}-"
    for operation in (
        "with_suffix",
        "with_name",
        "with_stem",
        "parent",
        "name",
        "stem",
        "suffix",
    ):
        if f"-{operation.replace('_', '-')}-" in normalized:
            return operation
    raise ValueError(f"capability case does not declare a path operation: {identifier}")


def _card_record(
    case: dict[str, Any], split: str, audit: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    operation = _operation_from_identifier(case["identifier"])
    candidates = [
        record
        for (record_split, _), record in audit.items()
        if record_split == split
        and record["semantic_key"]["operation"] == operation
        and record["rendered_assistant_label"] == case["expected"]
        and record["semantic_key"]["path"] in case["prompt"]
        and (
            record["semantic_key"]["argument"] is None
            or str(record["semantic_key"]["argument"]) in case["prompt"]
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"capability case {case['identifier']} maps to {len(candidates)} audited records"
        )
    line = candidates[0]["line"]
    return {**candidates[0], "record_id": f"path-domain-{split}-{line:02d}"}


def _build_query_cards(
    staging: Path,
    cards_dir: Path,
    audit: dict[tuple[str, int], dict[str, Any]],
    key_encoder: Any,
) -> list[dict[str, Any]]:
    output = staging / "cards"
    output.mkdir()
    entries: list[dict[str, Any]] = []
    for card_name, (filename, split) in _CARD_FILES.items():
        source = cards_dir / filename
        load_capability_card(source)
        payload = _read_json(source)
        query_count = 0
        for case in payload["cases"]:
            record = _card_record(case, split, audit)
            subject, relation = _key_identity(record["semantic_key"])
            vector = key_encoder.encode(StructuredKey(subject, relation))
            case["semantic_query"] = {
                "encoder": key_encoder.identity.model_dump(mode="json"),
                "vector": [float(value) for value in vector],
            }
            query_count += 1
        payload.pop("digest", None)
        card = _card_from_mapping(payload)
        payload["digest"] = card.digest
        destination = output / filename
        content = (
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        with destination.open("xb") as handle:
            handle.write(content)
            handle.flush()
        verified = load_capability_card(destination)
        if verified.digest != card.digest or any(
            case.semantic_query is None for case in verified.cases
        ):
            raise ValueError(
                f"query-bound capability card failed round-trip: {card_name}"
            )
        entries.append(
            {
                "path": f"cards/{filename}",
                "sha256": sha256_file(destination),
                "name": card_name,
                "case_count": len(card.cases),
                "semantic_query_count": query_count,
                "key_encoder": key_encoder.identity.model_dump(mode="json"),
            }
        )
    return entries


def _collect_provenance(
    dataset: DatasetConfig,
    tokenizer: Tokenizer,
    split: str,
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids, supervision, _, _ = _collect(dataset, tokenizer, split)
    max_documents = _DOCUMENT_CAPS[split]
    max_tokens = _TOKEN_CAPS[split]
    token_records: list[str] = []
    offset = 0
    acquired = 0
    selected_records = 0
    for record_index, rendered in enumerate(_source_documents(dataset, split)):
        if acquired >= max_documents or offset >= max_tokens:
            break
        acquired += 1
        if record_index >= len(records):
            raise ValueError(f"{split} corpus has more lines than its oracle audit")
        source_record = records[record_index]
        if not rendered.text:
            continue
        if rendered.loss_mode != "assistant_only" or not rendered.supervision_spans:
            raise ValueError(
                "path-domain local_chat records require assistant-only loss"
            )
        answer_start, answer_end = rendered.supervision_spans[-1]
        if (
            rendered.text[answer_start:answer_end].strip()
            != source_record["rendered_assistant_label"]
        ):
            raise ValueError(f"{split} source answer differs from its oracle audit")
        semantic_key = source_record["semantic_key"]
        if (
            semantic_key["path"] not in rendered.text
            or semantic_key["operation"] not in rendered.text
            or (
                semantic_key["argument"] is not None
                and str(semantic_key["argument"]) not in rendered.text
            )
        ):
            raise ValueError(
                f"{split} source prompt differs from its structured audit key"
            )
        encoding = tokenizer.encode(rendered.text, add_special_tokens=False)
        remaining = max_tokens - offset
        if remaining <= 1:
            break
        selected_count = min(len(encoding.ids), remaining - 1)
        token_count = selected_count + 1
        token_records.extend([source_record["record_id"]] * token_count)
        offset += token_count
        selected_records += 1
    if selected_records != min(len(records), max_documents) or len(
        token_records
    ) != len(ids):
        raise ValueError(
            f"{split} provenance spans do not match packed local_chat tokens"
        )
    return ids, supervision, token_records


def _owners_for_targets(target_count: int, profile: str, seed: int) -> np.ndarray:
    fraction = _PROFILE_NEURAL_FRACTIONS[profile]
    neural_count = int(target_count * fraction + 0.5)
    remaining = target_count - neural_count
    base, extra = divmod(remaining, 3)
    owners = [OWNER_NEURAL] * neural_count
    for owner, count in zip(
        (OWNER_LEXICAL, OWNER_SEMANTIC, OWNER_HYBRID),
        (base + int(extra > 0), base + int(extra > 1), base),
        strict=True,
    ):
        owners.extend([owner] * count)
    random.Random(seed).shuffle(owners)
    return np.asarray(owners, dtype=np.uint8)


def _split_sidecars(
    ids: np.ndarray,
    supervision: np.ndarray,
    token_records: list[str],
    profile: str,
    seed: int,
    vectors: dict[str, np.ndarray],
    key_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(ids) != len(supervision) or len(ids) != len(token_records):
        raise ValueError("packed IDs, supervision, and provenance are not aligned")
    owner = np.full(len(ids), OWNER_NEURAL, dtype=np.uint8)
    target_positions = np.flatnonzero(supervision)
    owner[target_positions] = _owners_for_targets(len(target_positions), profile, seed)
    queries = np.zeros((len(ids), key_dim), dtype=np.float32)
    mask = np.zeros(len(ids), dtype=bool)
    for target_position in target_positions:
        if owner[target_position] not in (OWNER_SEMANTIC, OWNER_HYBRID):
            continue
        input_position = int(target_position) - 1
        if input_position < 0:
            raise ValueError("a supervised first token has no causal input position")
        vector = vectors.get(token_records[target_position])
        if vector is None:
            raise ValueError("supervised target lacks an audited structured query")
        queries[input_position] = vector
        mask[input_position] = True
    return owner, queries, mask


def _pack_source(
    work: Path,
    records: dict[str, dict[str, Any]],
    key_encoder: Any,
    value_encoder: Any,
) -> tuple[Path, dict[str, Any]]:
    pack_records = work / "semantic-records.jsonl"
    keys_path = work / "semantic-keys.safetensors"
    values_path = work / "semantic-values.safetensors"
    metadata_path = work / "semantic-metadata.json"
    record_ids = [f"path-domain-train-{line:02d}" for line in range(1, 25)]
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with pack_records.open("xb") as handle:
        for record_id in record_ids:
            audit = records[record_id]
            subject, relation = _key_identity(audit["semantic_key"])
            label = audit["rendered_assistant_label"]
            source_record = {
                "id": record_id,
                "namespace": "path-domain-v1",
                "subject": subject,
                "relation": relation,
                "value": label,
                "source": "path_domain_v1",
                "source_revision": "1.1.0",
                "license": _LICENSE,
            }
            handle.write(canonical_json(source_record) + b"\n")
            keys.append(key_encoder.encode(StructuredKey(subject, relation)))
            values.append(value_encoder.encode(label))
        handle.flush()
        os.fsync(handle.fileno())
    key_array = np.ascontiguousarray(np.stack(keys), dtype=np.float32)
    value_array = np.ascontiguousarray(np.stack(values), dtype=np.float32)
    save_file({"keys": key_array}, str(keys_path))
    save_file({"values": value_array}, str(values_path))
    metadata = SemanticMetadata(
        format="sparselab-semantic-assets",
        format_version=1,
        record_ids=tuple(record_ids),
        key_encoder=key_encoder.identity,
        value_encoder=value_encoder.identity,
        key_normalization="none",
    )
    with metadata_path.open("xb") as handle:
        handle.write(canonical_json(metadata.model_dump(mode="json")) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    pack_root = work / "semantic-pack"
    manifest = compile_pack(
        pack_records,
        pack_root,
        name="path-domain-train-semantic-v1",
        namespace="path-domain-v1",
        default_license=_LICENSE,
        source_name="path_domain_v1",
        source_revision="1.1.0",
        created_at=_PACK_CREATED_AT,
        semantic_keys=keys_path,
        semantic_values=values_path,
        semantic_metadata=metadata_path,
    )
    return pack_root, {
        "pack_id": manifest.pack_id,
        "pack_sha256": sha256_file(pack_root / "manifest.json"),
        "key_encoder": key_encoder.identity.model_dump(mode="json"),
        "key_dim": key_array.shape[1],
        "memory_dim": value_array.shape[1],
        "entry_count": len(record_ids),
    }


def _build_bundle(
    staging: Path,
    *,
    tokenizer_path: Path,
    train_path: Path,
    validation_path: Path,
    audit_path: Path,
    provenance_path: Path,
    cards_dir: Path,
) -> None:
    inputs = (tokenizer_path, train_path, validation_path, audit_path, provenance_path)
    if any(path.is_symlink() or not path.is_file() for path in inputs):
        raise ValueError("allocation sources must be regular nonsymlink files")
    tokenizer = load_tokenizer(tokenizer_path)
    key_encoder, value_encoder = frozen_encoders()
    audit = _read_audit(audit_path)
    source_vectors, source_records = _source_key_maps(audit, key_encoder)
    expected_counts = {"train": 24, "development": 8, "frozen_evaluation": 12}
    for split, count in expected_counts.items():
        if sum(name == split for name, _ in audit) != count:
            raise ValueError(f"path-domain audit must contain {count} {split} records")
    dataset = DatasetConfig(
        source="local_chat",
        cache_dir=staging / "cache",
        train_max_documents=_DOCUMENT_CAPS["train"],
        validation_max_documents=_DOCUMENT_CAPS["validation"],
        train_max_tokens=_TOKEN_CAPS["train"],
        validation_max_tokens=_TOKEN_CAPS["validation"],
        train_path=train_path,
        validation_path=validation_path,
        license=_LICENSE,
    )
    split_data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split, records_split in (("train", "train"), ("validation", "development")):
        records = [
            source_records[f"path-domain-{records_split}-{line:02d}"]
            for line in range(1, expected_counts[records_split] + 1)
        ]
        split_data[split] = _collect_provenance(dataset, tokenizer, split, records)
    work = staging / ".work"
    work.mkdir()
    pack_root, pack_identity = _pack_source(
        work, source_records, key_encoder, value_encoder
    )
    bundle_cards = _build_query_cards(staging, cards_dir, audit, key_encoder)
    provenance = _read_json(provenance_path)
    provenance_splits = provenance.get("splits")
    license_info = provenance.get("license")
    oracle_info = provenance.get("oracle")
    if (
        provenance.get("format") != "path_domain_provenance_v1"
        or provenance.get("dataset_name") != "path_domain_v1"
        or provenance.get("dataset_version") != "1.1.0"
        or not isinstance(provenance.get("record_format"), dict)
        or provenance["record_format"].get("loss_mode") != "assistant_only"
        or not isinstance(provenance_splits, dict)
        or not isinstance(provenance_splits.get("training"), dict)
        or not isinstance(provenance_splits.get("development"), dict)
        or provenance_splits["training"].get("path") != train_path.name
        or provenance_splits["development"].get("path") != validation_path.name
        or not isinstance(license_info, dict)
        or license_info.get("spdx") != "MIT"
        or not isinstance(oracle_info, dict)
        or oracle_info.get("audit") != audit_path.name
    ):
        raise ValueError("path-domain provenance does not match supported inputs")
    source_digest = source_identity().get("sha256")
    if not isinstance(source_digest, str):
        raise TypeError("source identity has no SHA-256 digest")
    source_sha = source_digest
    tokenizer_sha = canonical_tokenizer_sha256(tokenizer)
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    profiles: dict[str, dict[str, Any]] = {}
    for regime in sorted(ALLOCATION_REGIMES):
        regime_dir = staging / regime
        regime_dir.mkdir()
        for profile, fraction in _PROFILE_NEURAL_FRACTIONS.items():
            seed = 20_260_923 + int(fraction * 100)
            train_ids, train_supervision, train_records = split_data["train"]
            val_ids, val_supervision, val_records = split_data["validation"]
            train_owner, train_queries, train_mask = _split_sidecars(
                train_ids,
                train_supervision,
                train_records,
                profile,
                seed,
                source_vectors,
                pack_identity["key_dim"],
            )
            val_owner, val_queries, val_mask = _split_sidecars(
                val_ids,
                val_supervision,
                val_records,
                profile,
                seed + 1,
                source_vectors,
                pack_identity["key_dim"],
            )
            pack_name = f"{profile}-semantic-pack"
            shutil.copytree(pack_root, regime_dir / pack_name)
            manifest_path = regime_dir / (
                f"{profile}-owners-neural-lexical-semantic-hybrid.json"
            )
            manifest = build_allocation_manifest(
                manifest_path,
                source_identity_sha256=source_sha,
                tokenizer_sha256=tokenizer_sha,
                train_jsonl_sha256=train_sha,
                validation_jsonl_sha256=validation_sha,
                train_owner=train_owner,
                validation_owner=val_owner,
                train_queries=train_queries,
                validation_queries=val_queries,
                train_mask=train_mask,
                validation_mask=val_mask,
                semantic={
                    "pack_path": pack_name,
                    "pack_sha256": pack_identity["pack_sha256"],
                    "pack_id": pack_identity["pack_id"],
                    "key_encoder": pack_identity["key_encoder"],
                },
                resource_regime=regime,
                ownership_profile=profile,
                sidecar_prefix=profile,
            )
            retriever = load_semantic_retriever(manifest)
            if retriever is None or retriever.memory_dim != pack_identity["memory_dim"]:
                raise ValueError("allocation bundle semantic pack failed verification")
            profiles[f"{regime}/{profile}"] = {
                "manifest": f"{regime}/{manifest_path.name}",
                "manifest_sha256": manifest.sha256,
                "train_raw_tokens": len(train_ids),
                "validation_raw_tokens": len(val_ids),
                "train_supervised_targets": int(train_supervision.sum()),
                "validation_supervised_targets": int(val_supervision.sum()),
                "train_owner_targets": {
                    str(owner): int(
                        np.count_nonzero(train_owner[train_supervision] == owner)
                    )
                    for owner in (
                        OWNER_NEURAL,
                        OWNER_LEXICAL,
                        OWNER_SEMANTIC,
                        OWNER_HYBRID,
                    )
                },
                "validation_owner_targets": {
                    str(owner): int(
                        np.count_nonzero(val_owner[val_supervision] == owner)
                    )
                    for owner in (
                        OWNER_NEURAL,
                        OWNER_LEXICAL,
                        OWNER_SEMANTIC,
                        OWNER_HYBRID,
                    )
                },
                "train_semantic_query_positions": int(train_mask.sum()),
                "validation_semantic_query_positions": int(val_mask.sum()),
                "neural_profile_fraction": fraction,
                "semantic_pack_id": pack_identity["pack_id"],
            }
    shutil.rmtree(work)
    files: list[dict[str, Any]] = []
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"allocation bundle contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    payload = {
        "format": "sparselab-memory-allocation-bundle",
        "version": 1,
        "source_identity_sha256": source_sha,
        "tokenizer_sha256": tokenizer_sha,
        "corpus": {
            "train_jsonl_sha256": train_sha,
            "validation_jsonl_sha256": validation_sha,
            "oracle_audit_sha256": sha256_file(audit_path),
            "provenance_sha256": sha256_file(provenance_path),
        },
        "assignment": {
            "unit": "assistant-supervised target token",
            "profile_fractions": _PROFILE_NEURAL_FRACTIONS,
            "remaining_owners": ["lexical", "semantic", "hybrid"],
            "algorithm": "rounded neural quota; balanced residual; seeded shuffle",
            "seed_base": 20_260_923,
        },
        "semantic_pack": pack_identity,
        "cards": bundle_cards,
        "profiles": profiles,
        "files": files,
    }
    payload["sha256"] = canonical_digest(payload)
    with (staging / "manifest.json").open("xb") as handle:
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_memory_allocation(
    output: Path,
    *,
    tokenizer_path: Path,
    train_path: Path,
    validation_path: Path,
    audit_path: Path,
    provenance_path: Path,
    cards_dir: Path,
) -> Path:
    """Build all five resource regimes and query-bound capability cards atomically."""
    output = output.absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite allocation bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.allocation-", dir=output.parent)
    )
    try:
        _build_bundle(
            staging,
            tokenizer_path=tokenizer_path,
            train_path=train_path,
            validation_path=validation_path,
            audit_path=audit_path,
            provenance_path=provenance_path,
            cards_dir=cards_dir,
        )
        _rename_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"
