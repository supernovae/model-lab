"""Immutable contiguous-EOS packing with optional causal byte-address arrays."""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tokenizers import Tokenizer

from klab.canonical import canonical_json
from klab.config.models import DatasetConfig, RunConfig
from klab.corpus.export import verify_release_export
from klab.data.allocation import AllocationManifest, load_allocation_manifest
from klab.data.byte_hash import table_address, token_bytes
from klab.data.conversations import (
    RenderedConversation,
    iter_rendered_conversations,
)
from klab.data.datasets import iter_documents
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    PreparationEncoder,
    validate_tokenizer_batch_limits,
)
from klab.data.local_stories import verify_snapshot
from klab.data.preparation_chunks import PreparationChunks
from klab.data.preparation_telemetry import PreparationTelemetry
from klab.data.preparation_workers import (
    PreparationWorkers,
    collect_parallel_stream,
)
from klab.data.token_store import TokenStore
from klab.data.verification import (
    HashingWriter,
    VerifiedFile,
    VerifiedPreparedData,
    _receipt_from_proofs,
    _relocate_proofs,
    _verify_file_with_hasher,
    _written_file,
    required_arrays,
)
from klab.host_capacity import run_ordered, sha_work_plan
from klab.progress import progress_phase
from klab.resource_envelope import (
    ResourceEnvelope,
    check_envelope,
    current_process_rss_bytes,
)
from klab.training import manifest as manifest_module
from klab.training.manifest import source_identity
from klab.verification_proofs import ProofStore, VerificationMode, file_binding
from klab.workdir import ensure_work_dir
from klab.workspace_cleanup import campaign_lock, mark_prepared_cache

if TYPE_CHECKING:
    from klab.bottleneck_observations import BottleneckObserver
    from klab.data.preparation_chunks import SplitChunks

PACKING_VERSION = "contiguous-eos-v6"
PREVIOUS_PACKING_VERSION = "contiguous-eos-v5"
HISTORICAL_PACKING_VERSION = "contiguous-eos-v4"
_MAX_MIXTURE_ROW_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class PreparedData:
    root: Path
    train: np.ndarray
    validation: np.ndarray
    train_supervision: np.ndarray | None
    validation_supervision: np.ndarray | None
    train_byte_addresses: np.ndarray | None
    validation_byte_addresses: np.ndarray | None
    train_owner_ids: np.ndarray | None
    validation_owner_ids: np.ndarray | None
    train_semantic_queries: np.ndarray | None
    validation_semantic_queries: np.ndarray | None
    train_semantic_mask: np.ndarray | None
    validation_semantic_mask: np.ndarray | None
    allocation: AllocationManifest | None
    manifest: dict[str, object]
    receipt: VerifiedPreparedData


@dataclass(frozen=True)
class PreparedPrefix:
    """Explicitly pinned immutable parent for a whole-document token prefix.

    The parent must have the same preparation source fingerprint. A pin binds
    bytes; it does not establish equivalence between different producers.
    """

    root: Path
    manifest_sha256: str


@dataclass(frozen=True)
class PreparedAppend:
    """Same-producer parent for an authenticated fresh local_text tail."""

    root: Path
    manifest_sha256: str


def prepared_dataset_identity(dataset: DatasetConfig) -> dict[str, object]:
    """Dataset selection identity; verified content binds operational locations."""
    return {
        key: value
        for key, value in dataset.model_dump(mode="json").items()
        if key
        not in {
            "cache_dir",
            "train_path",
            "validation_path",
            "source_manifest_path",
            "corpus_release_path",
            "corpus_export_path",
            "mixture_declaration_path",
            "mixture_output_path",
        }
    }


def preparation_source_identity(inventory: dict[str, str] | None = None) -> str:
    """Conservative reader/transform/packing fingerprint, not acquisition code.

    snapshot_writer is publication-only. Snapshot content, selection and reader
    semantics remain bound by the verified manifest and sources.py fingerprint.
    All other data modules (including newly added ones) remain fail-closed.
    """
    if inventory is None:
        root = Path(__file__).parents[1]
        paths = sorted((root / "data").glob("*.py"))
        paths += [root / "config" / "models.py"]
        inventory = {str(path.relative_to(root)): _sha256(path) for path in paths}
    names = sorted(
        name
        for name in inventory
        if name.startswith("data/")
        and name.endswith(".py")
        and name.count("/") == 1
        and name != "data/snapshot_writer.py"
    )
    names.append("config/models.py")
    return hashlib.sha256(
        canonical_json([(name, inventory[name]) for name in names])
    ).hexdigest()


def prepared_source_matches(identity: dict, package_sha256: str) -> bool:
    """Match preparation semantics, retaining explicit package-bound artifacts."""
    from klab.experiments.source_compatibility import source_identities_compatible

    recorded = identity["source_identity_sha256"]
    if (
        identity.get("allocation_manifest_sha256") is None
        and recorded == preparation_source_identity()
    ):
        return True
    return source_identities_compatible(recorded, package_sha256)


def _sha256(path: Path) -> str:
    return manifest_module.sha256_file(path)


def canonical_tokenizer_sha256(tokenizer: Tokenizer) -> str:
    """Hash the complete tokenizer, not merely its vocabulary."""
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def supervision_requires_mask(manifest: dict[str, object]) -> bool:
    required_arrays(manifest)
    return (
        manifest.get("packing_version") != HISTORICAL_PACKING_VERSION
        and manifest["supervision"]["kind"] == "token-loss-mask-v1"
    )


def _verified_receipt(
    root: Path,
    manifest: dict[str, object],
    identity: dict[str, object],
    *,
    byte_enabled: bool,
    verification: str,
    receipt: VerifiedPreparedData | None,
    telemetry: PreparationTelemetry | None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> VerifiedPreparedData:
    if (
        manifest.get("cache_identity") != identity
        or manifest.get("settings_sha256")
        != hashlib.sha256(canonical_json(identity)).hexdigest()
    ):
        raise ValueError("prepared cache identity mismatch")
    required = required_arrays(manifest)
    byte = manifest.get("byte_addressing")
    if (byte is not None) != byte_enabled:
        raise ValueError("prepared byte-address mode mismatch")
    if byte is not None and (
        byte.get("table_size") != identity["packing"]["memory_table_size"]
        or byte.get("ngram_size") != identity["packing"]["memory_ngram_size"]
    ):
        raise ValueError("prepared byte-address settings mismatch")
    if verification == "structural":
        if (
            not isinstance(receipt, VerifiedPreparedData)
            or receipt._seal is not _verification_seal()
            or receipt.root != root.resolve(strict=True)
            or receipt.manifest_sha256 != manifest.get("manifest_sha256")
        ):
            raise ValueError(
                "structural loading requires a matching in-process sealed receipt"
            )
        return _receipt_from_proofs(root, manifest, receipt.proofs)
    if verification != "deep":
        raise ValueError(f"unknown prepared verification mode: {verification}")

    def authenticate(name: str) -> tuple[str, VerifiedFile, float]:
        metadata, _, _ = required[name]
        started = time.monotonic()
        proof = _verify_file_with_hasher(
            root / name,
            expected_sha256=metadata.get("sha256"),
            hash_file=_sha256,
            proof_store=proof_store,
            verification_mode=verification_mode,
            binding=file_binding(
                root / name,
                metadata.get("sha256"),
                kind="prepared_array",
                closure={"metadata": metadata, "cache_identity": identity},
            ),
        )
        return name, proof, time.monotonic() - started

    # One full native SHA per independent member; never a chunk-tree digest.
    # Serial default retained until comparable storage profiling proves otherwise.
    proofs: dict[str, VerifiedFile] = {}
    for name, proof, elapsed in run_ordered(
        sorted(required), authenticate, plan=sha_work_plan()
    ):
        proofs[name] = proof
        if telemetry is not None:
            telemetry.add("deep_verification_hash_seconds", elapsed)
    return _receipt_from_proofs(root, manifest, proofs)


def _verification_seal() -> object:
    # No public unsigned receipt mint: the verification module owns this identity.
    from klab.data import verification

    return verification._SEAL


def load_prepared_data(
    root: Path,
    *,
    byte_enabled: bool,
    expected_identity: dict[str, object] | None = None,
    telemetry: PreparationTelemetry | None = None,
    verification: str = "deep",
    receipt: VerifiedPreparedData | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> PreparedData:
    """Open a verified immutable cache or run-owned copy without reacquiring data."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"prepared-data manifest must be an object: {root}")
    identity = (
        expected_identity
        if expected_identity is not None
        else manifest.get("cache_identity")
    )
    train, validation = root / "train.npy", root / "validation.npy"
    train_supervision, validation_supervision = (
        root / "train_supervision.npy",
        root / "validation_supervision.npy",
    )
    train_byte, validation_byte = (
        root / "train_byte_addresses.npy",
        root / "validation_byte_addresses.npy",
    )
    if not isinstance(identity, dict):
        raise ValueError(f"prepared-data cache identity check failed: {root}")  # noqa: TRY004 - invalid serialized schema
    verified = _verified_receipt(
        root,
        manifest,
        identity,
        byte_enabled=byte_enabled,
        verification=verification,
        receipt=receipt,
        telemetry=telemetry,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    has_supervision = supervision_requires_mask(manifest)
    allocation_enabled = isinstance(manifest.get("allocation"), dict)
    paths = [
        root / name
        for name in (
            "train_owner_ids.npy",
            "validation_owner_ids.npy",
            "train_semantic_queries.npy",
            "validation_semantic_queries.npy",
            "train_semantic_mask.npy",
            "validation_semantic_mask.npy",
        )
    ]
    allocation_arrays: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ] = (None, None, None, None, None, None)
    if allocation_enabled:
        allocation_metadata = manifest["allocation"]
        assert isinstance(allocation_metadata, dict)
        semantic = allocation_metadata.get("semantic")
        if allocation_metadata.get("format") != "klab-prepared-allocation-v1":
            raise ValueError("prepared allocation metadata is malformed")
        expected_paths = paths[:2] if semantic is None else paths
        if not all(path.is_file() for path in expected_paths) or (
            semantic is None and any(path.exists() for path in paths[2:])
        ):
            raise ValueError("prepared allocation sidecars are missing or unexpected")
        loaded: list[np.ndarray] = []
        for split, owner_path, query_path, mask_path in (
            ("train", paths[0], paths[2], paths[4]),
            ("validation", paths[1], paths[3], paths[5]),
        ):
            split_metadata = allocation_metadata.get(split)
            if not isinstance(split_metadata, dict) or not isinstance(
                split_metadata.get("owner"), dict
            ):
                raise TypeError("prepared allocation metadata is malformed")
            packed_ids = np.load(
                train if split == "train" else validation,
                mmap_mode="r",
                allow_pickle=False,
            )
            owner = np.load(owner_path, mmap_mode="r", allow_pickle=False)
            if (
                owner.ndim != 1
                or owner.shape != packed_ids.shape
                or owner.dtype != np.dtype(np.uint8)
                or np.any(owner > 3)
            ):
                raise ValueError("prepared allocation owner sidecar integrity failed")
            loaded.append(owner)
            if semantic is None:
                continue
            query_metadata = split_metadata.get("semantic_queries")
            mask_metadata = split_metadata.get("semantic_mask")
            if not isinstance(query_metadata, dict) or not isinstance(
                mask_metadata, dict
            ):
                raise TypeError("prepared semantic allocation metadata is malformed")
            query = np.load(query_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if (
                query.ndim != 2
                or query.dtype != np.dtype(np.float32)
                or query.shape[0] != len(owner)
                or query.shape[1] <= 0
                or mask.dtype != np.dtype(bool)
                or mask.shape != owner.shape
                or mask[-1]
                or np.any(mask[:-1] & ~np.isin(owner[1:], (2, 3)))
            ):
                raise ValueError(
                    "prepared semantic allocation sidecar integrity failed"
                )
            loaded.extend((query, mask))
        if semantic is None:
            allocation_arrays = (loaded[0], loaded[1], None, None, None, None)
        else:
            allocation_arrays = (
                loaded[0],
                loaded[3],
                loaded[1],
                loaded[4],
                loaded[2],
                loaded[5],
            )
    return PreparedData(
        root,
        np.load(train, mmap_mode="r", allow_pickle=False),
        np.load(validation, mmap_mode="r", allow_pickle=False),
        np.load(train_supervision, mmap_mode="r", allow_pickle=False)
        if has_supervision
        else None,
        np.load(validation_supervision, mmap_mode="r", allow_pickle=False)
        if has_supervision
        else None,
        np.load(train_byte, mmap_mode="r", allow_pickle=False)
        if byte_enabled
        else None,
        np.load(validation_byte, mmap_mode="r", allow_pickle=False)
        if byte_enabled
        else None,
        *allocation_arrays,
        None,
        manifest,
        verified,
    )


def _encoded_token_bytes(
    tokenizer: Tokenizer, ids: list[int], document: str
) -> list[bytes]:
    pieces = [token_bytes(tokenizer, token_id) for token_id in ids]
    if b"".join(pieces) != document.encode("utf-8"):
        raise ValueError("tokenizer token bytes do not reconstruct the source document")
    return pieces


def _local_chat_identity(config: DatasetConfig) -> dict[str, str] | None:
    if config.source not in {"local_chat", "local_text"}:
        return None
    paths = (
        getattr(config, "train_path", None),
        getattr(config, "validation_path", None),
    )
    if not all(isinstance(path, Path) and path.is_file() for path in paths):
        raise ValueError(
            f"{config.source} requires readable train_path and validation_path"
        )
    return {
        "train_sha256": manifest_module.sha256_file(paths[0]),
        "validation_sha256": manifest_module.sha256_file(paths[1]),
    }


def _assert_local_chat_disjoint(
    config: DatasetConfig, *, local_text_source_bytes: int | None = None
) -> None:
    if config.source not in {"local_chat", "local_text"}:
        return
    train = {
        hashlib.sha256(document.encode("utf-8")).digest()
        for document in iter_documents(
            config, "train", local_text_source_bytes=local_text_source_bytes
        )
    }
    overlap = next(
        (
            document
            for document in iter_documents(
                config, "validation", local_text_source_bytes=local_text_source_bytes
            )
            if hashlib.sha256(document.encode("utf-8")).digest() in train
        ),
        None,
    )
    if overlap is not None:
        raise ValueError(
            "local_chat train and validation contain an overlapping conversation"
        )


def _assert_local_chat_supervision_consistent(config: DatasetConfig) -> None:
    if config.source != "local_chat":
        return
    assert config.train_path is not None and config.validation_path is not None
    train_modes = {
        item.loss_mode for item in iter_rendered_conversations(config.train_path)
    }
    validation_modes = {
        item.loss_mode for item in iter_rendered_conversations(config.validation_path)
    }
    if (
        len(train_modes) != 1
        or len(validation_modes) != 1
        or train_modes != validation_modes
    ):
        raise ValueError(
            "local_chat training and validation must use one matching supervision mode"
        )


def _supervision_for_encoding(
    document: RenderedConversation | None,
    offsets: list[tuple[int, int]],
    token_count: int,
) -> list[bool]:
    if document is None or document.loss_mode == "all_tokens":
        return [True] * token_count
    spans = document.supervision_spans
    # A token crossing a role boundary cannot be partially supervised. Spans
    # include the assistant's leading separator space, but never role markers.
    return [
        any(
            start < end and span_start <= start and end <= span_end
            for span_start, span_end in spans
        )
        for start, end in offsets[:token_count]
    ]


def _source_documents(
    config: DatasetConfig,
    split: str,
    *,
    local_text_source_bytes: int | None = None,
) -> Iterator[RenderedConversation]:
    if config.source == "local_chat":
        path = config.train_path if split == "train" else config.validation_path
        assert path is not None
        yield from iter_rendered_conversations(path)
        return
    documents = (
        iter_documents(config, split)
        if local_text_source_bytes is None
        else iter_documents(
            config, split, local_text_source_bytes=local_text_source_bytes
        )
    )
    for text in documents:
        yield RenderedConversation(text, ((0, len(text)),), "all_tokens")


def _collect(
    config: DatasetConfig,
    tokenizer: Tokenizer,
    split: str,
    *,
    root: Path | None = None,
    proofs: dict[str, VerifiedFile] | None = None,
    byte_table_size: int | None = None,
    byte_ngram_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    max_documents = (
        config.train_max_documents
        if split == "train"
        else config.validation_max_documents
    )
    max_tokens = (
        config.train_max_tokens if split == "train" else config.validation_max_tokens
    )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> special token")
    values = (
        [] if root is None else _ArraySpool(root / f"{split}.npy", np.dtype(np.int32))
    )
    supervision = (
        []
        if root is None
        else _ArraySpool(root / f"{split}_supervision.npy", np.dtype(bool))
    )
    byte_addresses = (
        (
            []
            if root is None
            else _ArraySpool(root / f"{split}_byte_addresses.npy", np.dtype(np.int32))
        )
        if byte_table_size is not None
        else None
    )
    stats = {
        "acquired_documents": 0,
        "retained_documents": 0,
        "skipped_documents": 0,
        "truncated_documents": 0,
    }
    documents = iter(
        _source_documents(
            config,
            split,
            local_text_source_bytes=TOKENIZER_BATCH_SOURCE_BYTES
            if root is not None
            else None,
        )
    )
    source_wait_seconds = 0.0
    encoding_seconds = 0.0
    source_bytes = 0
    encoded_tokens = 0
    source_digest = hashlib.sha256()
    started = time.monotonic()
    with progress_phase(
        f"data_{split}_document_iteration_and_collection",
        completed_work=0,
        unit="documents",
        raw_counters={**stats, "source_bytes": 0, "encoded_tokens": 0},
    ) as progress:
        while stats["acquired_documents"] < max_documents and len(values) < max_tokens:
            read_started = time.monotonic()
            try:
                rendered = next(documents)
            except StopIteration:
                break
            finally:
                source_wait_seconds += time.monotonic() - read_started
            stats["acquired_documents"] += 1
            document = rendered.text
            if not document:
                stats["skipped_documents"] += 1
                continue
            encoded_bytes = document.encode("utf-8")
            source_bytes += len(encoded_bytes)
            source_digest.update(encoded_bytes)
            source_digest.update(b"\0")
            encode_started = time.monotonic()
            encoding = tokenizer.encode(document, add_special_tokens=False)
            encoding_seconds += time.monotonic() - encode_started
            encoded_tokens += len(encoding.ids)
            remaining = max_tokens - len(values)
            if remaining <= 1:
                break
            selected_count = min(len(encoding.ids), remaining - 1)
            selected = encoding.ids[:selected_count] + [eos]
            selected_supervision = _supervision_for_encoding(
                rendered, encoding.offsets, selected_count
            )
            selected_supervision.append(
                rendered.loss_mode == "all_tokens"
                or (
                    selected_count == len(encoding.ids)
                    and bool(rendered.supervision_spans)
                )
            )
            if selected_count < len(encoding.ids):
                stats["truncated_documents"] += 1
            if byte_addresses is not None:
                assert byte_ngram_size is not None
                prefix = bytearray()
                for token_piece in _encoded_token_bytes(
                    tokenizer, encoding.ids, document
                )[:selected_count]:
                    prefix.extend(token_piece)
                    byte_addresses.append(
                        table_address(bytes(prefix[-byte_ngram_size:]), byte_table_size)
                    )
                byte_addresses.append(0)
            values.extend(selected)
            supervision.extend(selected_supervision)
            stats["retained_documents"] += 1
            if stats["acquired_documents"] % 500 == 0:
                progress.update(
                    completed_work=stats["acquired_documents"],
                    unit="documents",
                    raw_counters={
                        **stats,
                        "source_bytes": source_bytes,
                        "encoded_tokens": encoded_tokens,
                        "retained_tokens": len(values),
                    },
                )
        elapsed = max(time.monotonic() - started, 1e-9)
        construction_seconds = max(
            0.0, elapsed - source_wait_seconds - encoding_seconds
        )
        progress.update(
            completed_work=stats["acquired_documents"],
            unit="documents",
            raw_counters={
                **stats,
                "source_bytes": source_bytes,
                "encoded_tokens": encoded_tokens,
                "source_content_sha256": source_digest.hexdigest(),
                "encoded_tokens_per_second": encoded_tokens
                / max(encoding_seconds, 1e-9),
                "construction_tokens_per_second": len(values)
                / max(construction_seconds, 1e-9),
                "output_tokens": len(values),
                "source_iteration_seconds": source_wait_seconds,
                "tokenizer_encoding_seconds": encoding_seconds,
                "python_array_construction_seconds": construction_seconds,
                "documents_per_second": stats["acquired_documents"] / elapsed,
                "source_bytes_per_second": source_bytes / elapsed,
                "output_tokens_per_second": len(values) / elapsed,
            },
        )
    if root is not None:
        for spool in (values, supervision, byte_addresses):
            if spool is not None:
                proof = spool.finish()
                if proofs is not None:
                    proofs[spool.path.name] = proof
        return (
            np.load(values.path, mmap_mode="r", allow_pickle=False),
            np.load(supervision.path, mmap_mode="r", allow_pickle=False),
            None
            if byte_addresses is None
            else np.load(byte_addresses.path, mmap_mode="r", allow_pickle=False),
            stats,
        )
    return (
        np.asarray(values, dtype=np.int32),
        np.asarray(supervision, dtype=bool),
        None if byte_addresses is None else np.asarray(byte_addresses, dtype=np.int32),
        stats,
    )


def _collect_token_mixture(
    config: RunConfig,
    tokenizer: Tokenizer,
    receipt: dict[str, object],
    root: Path,
    *,
    proofs: dict[str, VerifiedFile] | None = None,
    chunks: SplitChunks | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Pack exact scheduled targets with BOS and masked block-end padding."""
    dataset = config.dataset
    assert dataset.train_path is not None
    target_count = sum(receipt["actual_target_tokens"].values())
    seq_len = config.training.seq_len
    padded_targets = ((target_count + seq_len - 1) // seq_len) * seq_len
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    if any(value is None for value in (bos, eos, pad)):
        raise ValueError("token mixture tokenizer requires BOS, EOS and PAD")
    ids = (
        _ArraySpool(root / "train.npy", np.dtype(np.int32)) if chunks is None else None
    )
    mask = (
        _ArraySpool(root / "train_supervision.npy", np.dtype(bool))
        if chunks is None
        else None
    )
    if chunks is None:
        ids.append([bos])
        mask.append([False])
    cursor = 1
    records = 0
    stream_digest = hashlib.sha256()
    by_stratum: Counter[str] = Counter()
    with dataset.train_path.open("rb") as stream:
        while line := stream.readline(_MAX_MIXTURE_ROW_BYTES + 1):
            if len(line) > _MAX_MIXTURE_ROW_BYTES:
                raise ValueError("mixture record-byte cap exceeded")
            if not line.strip():
                raise ValueError("empty token mixture JSONL row")
            stream_digest.update(line)
            row = json.loads(line)
            tokens = row["token_ids"]
            if (
                not isinstance(tokens, list)
                or not tokens
                or tokens[-1] != eos
                or any(
                    type(value) is not int or not 0 <= value < config.model.vocab_size
                    for value in tokens
                )
                or cursor + len(tokens) > target_count + 1
            ):
                raise ValueError("invalid token mixture target row")
            if chunks is None:
                ids.append(tokens)
                mask.append([True] * len(tokens))
            elif records >= chunks.state["acquired_documents"]:
                selected = ([bos] if records == 0 else []) + tokens
                supervision = ([False] if records == 0 else []) + [True] * len(tokens)
                if records + 1 == receipt["records"]:
                    padding = padded_targets - target_count
                    selected += [pad] * padding
                    supervision += [False] * padding
                chunks.append_record(
                    records + 1, {"ids": selected, "supervision": supervision}
                )
            cursor += len(tokens)
            records += 1
            by_stratum[row["stratum"]] += len(tokens)
    if (
        cursor != target_count + 1
        or records != receipt["records"]
        or dict(by_stratum) != receipt["actual_target_tokens"]
        or stream_digest.hexdigest() != receipt["train_tokens_sha256"]
    ):
        raise ValueError("token mixture target count differs from verified receipt")
    remaining_padding = padded_targets - target_count
    while chunks is None and remaining_padding:
        count = min(remaining_padding, 65_536)
        ids.append([pad] * count)
        mask.append([False] * count)
        remaining_padding -= count
    if chunks is None:
        for spool in (ids, mask):
            proof = spool.finish()
            if proofs is not None:
                proofs[spool.path.name] = proof
    else:
        chunk_proofs, _ = chunks.finish()
        for name, proof in chunk_proofs.items():
            destination = root / name
            proof.path.replace(destination)
            if proofs is not None:
                proofs[name] = _written_file(
                    destination, proof.sha256, proof.size_bytes
                )
    stats = {
        "acquired_documents": records,
        "retained_documents": records,
        "skipped_documents": 0,
        "truncated_documents": 0,
        "scheduled_target_positions": target_count,
        "supervised_target_positions": target_count,
        "supervised_target_positions_by_stratum": receipt["actual_target_tokens"],
        "masked_padding_positions": padded_targets - target_count,
    }
    return (
        np.load(root / "train.npy", mmap_mode="r", allow_pickle=False),
        np.load(root / "train_supervision.npy", mmap_mode="r", allow_pickle=False),
        stats,
    )


class _ArraySpool:
    """Bounded in-memory chunks with a disk-backed, exactly sized final array."""

    def __init__(
        self, path: Path, dtype: np.dtype, telemetry: PreparationTelemetry | None = None
    ) -> None:
        self.telemetry = telemetry
        self.path = path
        self.dtype = np.dtype(dtype)
        self.raw_path = path.with_suffix(".raw")
        self.raw = self.raw_path.open("xb")
        self.buffer: list[int | bool] = []
        self.count = 0

    def __len__(self):
        return self.count

    def append(self, values: list[int] | list[bool] | int | bool) -> None:
        if isinstance(values, (int, bool)):
            values = [values]
        self.buffer.extend(values)
        self.count += len(values)
        if len(self.buffer) >= 65_536:
            self.flush()

    extend = append

    def append_array(self, values: np.ndarray) -> None:
        """Copy verified contiguous IDs without per-token Python conversion."""
        if (
            values.ndim != 1
            or values.dtype != self.dtype
            or not values.flags.c_contiguous
        ):
            raise ValueError(
                "spool array must have matching contiguous one-dimensional dtype"
            )
        self.flush()
        self.raw.write(memoryview(values).cast("B"))
        self.count += len(values)

    def flush(self) -> None:
        if self.buffer:
            started = time.monotonic()
            self.raw.write(np.asarray(self.buffer, dtype=self.dtype).tobytes())
            self.buffer.clear()
            if self.telemetry is not None:
                self.telemetry.add("spool_write_seconds", time.monotonic() - started)

    def finish(self) -> VerifiedFile:
        self.flush()
        started = time.monotonic()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        temporary = self.path.with_name(self.path.stem + ".tmp.npy")
        with temporary.open("xb") as output, self.raw_path.open("rb") as source:
            writer = HashingWriter(output)
            np.lib.format.write_array_header_1_0(
                writer,
                {
                    "descr": np.lib.format.dtype_to_descr(self.dtype),
                    "fortran_order": False,
                    "shape": (self.count,),
                },
            )
            copy_started = time.monotonic()
            shutil.copyfileobj(source, writer, length=1024 * 1024)
            copy_seconds = time.monotonic() - copy_started
            if self.telemetry is not None:
                self.telemetry.add("spool_write_seconds", copy_seconds)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)
        self.raw_path.unlink()
        if self.telemetry is not None:
            self.telemetry.add(
                "finalize_fsync_seconds", time.monotonic() - started - copy_seconds
            )
        return _written_file(self.path, writer.digest.hexdigest(), writer.size)


def _streaming_values(
    tokenizer, encoded_ids, text, *, eos, byte_table_size=None, byte_ngram_size=None
):
    values = {"ids": encoded_ids + [eos]}
    if byte_table_size is not None:
        assert byte_ngram_size is not None
        prefix = bytearray()
        addresses = []
        for piece in _encoded_token_bytes(tokenizer, encoded_ids, text):
            prefix.extend(piece)
            addresses.append(
                table_address(bytes(prefix[-byte_ngram_size:]), byte_table_size)
            )
        addresses.append(0)
        values["byte_addresses"] = addresses
    return values


def _collect_streaming(
    config: DatasetConfig,
    tokenizer: Tokenizer,
    split: str,
    root: Path,
    *,
    selected_documents: int,
    byte_table_size: int | None = None,
    byte_ngram_size: int | None = None,
    resource_envelope: ResourceEnvelope | None = None,
    telemetry: PreparationTelemetry | None = None,
    encoder: PreparationEncoder | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    proofs: dict[str, VerifiedFile] | None = None,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    chunks: SplitChunks | None = None,
) -> dict[str, int]:
    """Pack every selected story, refusing a partial document or short source."""
    tokenizer_batch_documents, tokenizer_batch_source_bytes = (
        validate_tokenizer_batch_limits(
            tokenizer_batch_documents, tokenizer_batch_source_bytes
        )
    )
    if resource_envelope is not None:
        if not resource_envelope.spill_to_disk:
            raise ValueError(
                "streaming preparation requires resource_envelope.spill_to_disk=true"
            )
        check_envelope(
            resource_envelope,
            workspace=root,
            rss_bytes=current_process_rss_bytes()
            if telemetry is None
            else telemetry.snapshot()["current_rss_bytes"],
            pending_workers=0,
            queue_depth=0,
        )
    if encoder is None:
        with PreparationEncoder(
            tokenizer,
            max_workers=None
            if resource_envelope is None
            else resource_envelope.max_workers,
        ) as owned_encoder:
            return _collect_streaming(
                config,
                tokenizer,
                split,
                root,
                selected_documents=selected_documents,
                byte_table_size=byte_table_size,
                byte_ngram_size=byte_ngram_size,
                resource_envelope=resource_envelope,
                telemetry=telemetry,
                encoder=owned_encoder,
                tokenizer_batch_documents=tokenizer_batch_documents,
                proofs=proofs,
                tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                chunks=chunks,
            )
    if telemetry is not None:
        telemetry.tokenizer_rayon_threads = encoder.rayon_threads
        telemetry.tokenizer_host_work_plan = asdict(encoder.host_work_plan)
    tokenizer_spec = json.loads(tokenizer.to_str())
    byte_token_bound = (
        tokenizer_spec.get("model", {}).get("type") == "BPE"
        and tokenizer_spec.get("pre_tokenizer", {}).get("type") == "ByteLevel"
        and tokenizer_spec.get("normalizer") is None
    )
    max_tokens = (
        config.train_max_tokens if split == "train" else config.validation_max_tokens
    )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> special token")
    ids = (
        _ArraySpool(root / f"{split}.npy", np.dtype(np.int32), telemetry)
        if chunks is None
        else None
    )
    supervision = None
    byte_addresses = (
        _ArraySpool(root / f"{split}_byte_addresses.npy", np.dtype(np.int32), telemetry)
        if byte_table_size is not None and chunks is None
        else None
    )
    stats = (
        {
            "acquired_documents": 0,
            "retained_documents": 0,
            "skipped_documents": 0,
            "truncated_documents": 0,
        }
        if chunks is None
        else {
            key: chunks.state[key]
            for key in (
                "acquired_documents",
                "retained_documents",
                "skipped_documents",
                "truncated_documents",
            )
        }
    )
    output_tokens = 0 if chunks is None else chunks.state["output_tokens"]
    with progress_phase(
        f"data_{split}_document_iteration_and_collection",
        completed_work=0,
        total_work=selected_documents,
        unit="documents",
        raw_counters=stats
        if telemetry is None
        else {**telemetry.snapshot(), **stats, "output_tokens": output_tokens},
    ) as progress:
        documents = iter(
            _source_documents(
                config,
                split,
                local_text_source_bytes=(
                    tokenizer_batch_source_bytes
                    if config.source == "local_text"
                    else None
                ),
            )
        )
        if chunks is not None:
            if telemetry is not None:
                telemetry.output_tokens += output_tokens
            for skipped_index in range(stats["acquired_documents"]):
                read_started = time.monotonic()
                try:
                    restored = next(documents)
                except StopIteration as error:
                    raise ValueError(
                        f"{split} source ended before recovered record {skipped_index + 1}"
                    ) from error
                if telemetry is not None:
                    telemetry.add(
                        "source_iteration_seconds", time.monotonic() - read_started
                    )
                    source_bytes = len(restored.text.encode("utf-8"))
                    telemetry.records += 1
                    telemetry.source_bytes += source_bytes
                    telemetry.logical_input_bytes += source_bytes
        split_started = time.monotonic()
        pending: list[tuple[RenderedConversation, int]] = []
        pending_bytes = 0
        pending_upper_tokens = 0

        def flush() -> None:
            nonlocal pending_bytes, pending_upper_tokens, output_tokens
            if not pending:
                return
            if resource_envelope is not None:
                check_envelope(
                    resource_envelope,
                    workspace=root,
                    rss_bytes=current_process_rss_bytes(),
                    pending_workers=0,
                    queue_depth=1,
                )
            encode_started = time.monotonic()
            assert encoder is not None
            encodings = encoder.encode([rendered.text for rendered, _ in pending])
            if telemetry is not None:
                telemetry.add(
                    "tokenizer_encoding_seconds", time.monotonic() - encode_started
                )
            for (rendered, acquired_index), encoded_ids in zip(
                pending, encodings, strict=True
            ):
                bookkeeping_started = time.monotonic()
                values = _streaming_values(
                    tokenizer,
                    encoded_ids,
                    rendered.text,
                    eos=eos,
                    byte_table_size=byte_table_size,
                    byte_ngram_size=byte_ngram_size,
                )
                selected = values["ids"]
                if output_tokens + len(selected) > max_tokens:
                    raise ValueError(
                        f"{split} token cap would truncate selected story "
                        f"{acquired_index}: {output_tokens + len(selected)} > {max_tokens}"
                    )
                if chunks is None:
                    assert ids is not None
                    ids.append(selected)
                    if byte_addresses is not None:
                        byte_addresses.append(values["byte_addresses"])
                else:
                    chunks.append_record(acquired_index, values)
                output_tokens += len(selected)
                stats["retained_documents"] += 1
                if telemetry is not None:
                    telemetry.output_tokens += len(selected)
                    telemetry.add(
                        "python_bookkeeping_seconds",
                        time.monotonic() - bookkeeping_started,
                    )
                if stats["acquired_documents"] % 500 == 0:
                    progress.update(
                        completed_work=stats["retained_documents"],
                        total_work=selected_documents,
                        unit="documents",
                        raw_counters={
                            **({} if telemetry is None else telemetry.snapshot()),
                            **stats,
                            "output_tokens": output_tokens,
                        },
                    )
            pending.clear()
            pending_bytes = 0
            pending_upper_tokens = 0

        while stats["retained_documents"] + len(pending) < selected_documents:
            # UTF-8 bytes plus EOS bound unnormalized ByteLevel BPE outputs.
            # Other tokenizers are encoded before acquiring the next record.
            if pending and (
                not byte_token_bound
                or output_tokens + pending_upper_tokens >= max_tokens
            ):
                flush()
            read_started = time.monotonic()
            try:
                rendered = next(documents)
            except StopIteration:
                break
            finally:
                if telemetry is not None:
                    telemetry.add(
                        "source_iteration_seconds", time.monotonic() - read_started
                    )
            stats["acquired_documents"] += 1
            bookkeeping_started = time.monotonic()
            document = rendered.text
            if telemetry is not None:
                telemetry.records += 1
            if not document:
                if chunks is not None:
                    flush()
                    chunks.append_record(stats["acquired_documents"], None)
                stats["skipped_documents"] += 1
                if telemetry is not None:
                    telemetry.add(
                        "python_bookkeeping_seconds",
                        time.monotonic() - bookkeeping_started,
                    )
                continue
            encoded_bytes = len(document.encode("utf-8"))
            if encoded_bytes > tokenizer_batch_source_bytes:
                raise ValueError(
                    f"{split} source document {stats['acquired_documents']} exceeds "
                    f"tokenizer_batch_source_bytes ({encoded_bytes} > "
                    f"{tokenizer_batch_source_bytes})"
                )
            if telemetry is not None:
                telemetry.source_bytes += encoded_bytes
                telemetry.logical_input_bytes += encoded_bytes
                telemetry.add(
                    "python_bookkeeping_seconds", time.monotonic() - bookkeeping_started
                )
            if pending and (
                len(pending) == tokenizer_batch_documents
                or pending_bytes + encoded_bytes > tokenizer_batch_source_bytes
                or output_tokens + pending_upper_tokens + encoded_bytes + 1 > max_tokens
            ):
                flush()
            pending.append((rendered, stats["acquired_documents"]))
            pending_bytes += encoded_bytes
            pending_upper_tokens += encoded_bytes + 1
            if (
                len(pending) == tokenizer_batch_documents
                or pending_bytes == tokenizer_batch_source_bytes
                or output_tokens + pending_upper_tokens >= max_tokens
            ):
                flush()
        flush()
        if stats["retained_documents"] != selected_documents:
            raise ValueError(
                f"{split} snapshot has only {stats['retained_documents']} "
                f"selected distinct stories; required {selected_documents}"
            )
        if resource_envelope is not None:
            check_envelope(
                resource_envelope,
                workspace=root,
                rss_bytes=(
                    None
                    if telemetry is None
                    else telemetry.snapshot()["current_rss_bytes"]
                ),
                pending_workers=0,
                queue_depth=0,
            )
        if chunks is None:
            finished = {}
            for spool in (ids, supervision, byte_addresses):
                if spool is not None:
                    finished[spool.path.name] = spool.finish()
            stats["output_tokens"] = output_tokens
            stats["artifact_bytes"] = sum(
                proof.size_bytes for proof in finished.values()
            )
            stats["artifact_files"] = len(finished)
        else:
            finished, stats = chunks.finish()
        if proofs is not None:
            proofs.update(finished)
        if telemetry is not None and chunks is None:
            telemetry.logical_output_bytes += stats["artifact_bytes"]
        progress.update(
            completed_work=selected_documents,
            total_work=selected_documents,
            unit="documents",
            raw_counters={
                **({} if telemetry is None else telemetry.snapshot()),
                **stats,
                "split_elapsed_seconds": time.monotonic() - split_started,
            },
        )
    return stats


def _documents_encode_independently(tokenizer: Tokenizer) -> bool:
    """Whether batched, parallel and one-at-a-time encoding give the same IDs."""
    tokenizer_spec = json.loads(tokenizer.to_str())
    bpe = tokenizer_spec.get("model", {})
    return (
        bpe.get("type") == "BPE"
        and bpe.get("dropout") in (None, 0.0)
        and not bpe.get("byte_fallback", False)
        and bpe.get("continuing_subword_prefix") in (None, "")
        and bpe.get("end_of_word_suffix") in (None, "")
        and tokenizer_spec.get("pre_tokenizer", {}).get("type") == "ByteLevel"
        and tokenizer_spec.get("normalizer") is None
        and tokenizer_spec.get("pre_tokenizer", {}).get("add_prefix_space") is False
        and tokenizer_spec.get("padding") is None
    )


# Staging directories of the token-store path, removed if preparation fails.
_STORE_STAGING: ContextVar[list[Path] | None] = ContextVar(
    "klab_token_store_staging", default=None
)
# The reusable token store is used for eligible sources; tests compare it with
# the chunked path by turning it off.
TOKEN_STORE_ENABLED = True


def _discard_store_staging_on_failure(function):
    """Leave no token-store staging directory behind when preparation fails."""

    @wraps(function)
    def guarded(*args, **kwargs):
        token = _STORE_STAGING.set([])
        try:
            return function(*args, **kwargs)
        except BaseException:
            for staging in _STORE_STAGING.get() or []:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            _STORE_STAGING.reset(token)

    return guarded


def token_store_root(cache_dir: Path) -> Path:
    """Where a dataset cache directory keeps its reusable token shards.

    The store sits beside the cache directory, so every run that shares a cache
    directory shares its tokens, and a cache directory in a private or
    temporary location keeps its tokens there too.
    """
    return cache_dir.with_name(cache_dir.name + "-tokens")


def _token_stores(
    config: RunConfig,
    tokenizer: Tokenizer,
    source_sha256: dict[str, str],
    *,
    tokenizer_batch_source_bytes: int,
    resource_envelope: ResourceEnvelope | None,
    telemetry: PreparationTelemetry,
) -> dict[str, tuple[TokenStore, int, int]] | None:
    """Build or extend the token store for both splits of a local_text source.

    Returns each split's store with its selected document and token counts, or
    ``None`` when the source is not eligible and the chunked path must run. The
    checks and messages are those of the chunked path, so a selection it would
    refuse is refused here in the same words.
    """
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> special token")
    from klab.data.encoding import tokenizer_work_plan
    from klab.workspace_preflight import check_storage, require_storage

    workers = tokenizer_work_plan(
        None if resource_envelope is None else resource_envelope.max_workers,
        default_cap=None,
    ).workers
    base = token_store_root(config.dataset.cache_dir)
    selected = {}
    for split in ("train", "validation"):
        source = getattr(config.dataset, f"{split}_path")
        documents = getattr(config.dataset, f"{split}_max_documents")
        max_tokens = getattr(config.dataset, f"{split}_max_tokens")
        store = TokenStore.open(
            base,
            source_sha256=source_sha256[f"{split}_sha256"],
            tokenizer_sha256=canonical_tokenizer_sha256(tokenizer),
            eos=eos,
        )
        if not store.complete and store.documents < documents:
            # ByteLevel BPE never yields more tokens than bytes; real text is
            # near four bytes per token, so two is a safe planning bound.
            unread = source.stat().st_size - (
                int(store.shards[-1]["source_end"]) if store.shards else 0
            )
            require_storage(
                [
                    check_storage(
                        base,
                        projected_bytes=min(
                            4 * max(max_tokens - store.tokens, 0), 2 * unread
                        ),
                    )
                ]
            )
        started = time.monotonic()
        with progress_phase(
            f"data_{split}_token_store",
            completed_work=store.documents,
            total_work=documents,
            unit="documents",
        ) as progress:
            store.extend(
                source,
                tokenizer,
                documents=documents,
                max_tokens=max_tokens,
                source_bytes=tokenizer_batch_source_bytes,
                workers=workers,
                split=split,
                progress=lambda done, tokens, progress=progress, total=documents: (
                    progress.update(
                        completed_work=done,
                        total_work=total,
                        unit="documents",
                        raw_counters={"output_tokens": tokens},
                    )
                ),
            )
        telemetry.add("tokenizer_encoding_seconds", time.monotonic() - started)
        ends = store.ends()[:documents]
        if len(ends) and int(ends[-1]) > max_tokens:
            first = int(np.searchsorted(ends, max_tokens, side="right"))
            raise ValueError(
                f"{split} token cap would truncate selected story "
                f"{first + 1}: {int(ends[first])} > {max_tokens}"
            )
        if len(ends) != documents:
            raise ValueError(
                f"{split} snapshot has only {len(ends)} "
                f"selected distinct stories; required {documents}"
            )
        if store.max_document_bytes(documents) > tokenizer_batch_source_bytes:
            # Stored under a looser document cap: let the chunked path name the
            # offending document.
            return None
        selected[split] = (store, documents, int(ends[-1]))
    return selected


def _copy_from_token_store(
    store: TokenStore, tokens: int, path: Path, telemetry: PreparationTelemetry
) -> VerifiedFile:
    """Write the first ``tokens`` stored IDs as a prepared ``.npy`` array."""
    temporary = path.with_name(path.stem + ".tmp.npy")
    started = time.monotonic()
    with temporary.open("xb") as output:
        writer = HashingWriter(output)
        np.lib.format.write_array_header_1_0(
            writer,
            {
                "descr": np.lib.format.dtype_to_descr(np.dtype(np.int32)),
                "fortran_order": False,
                "shape": (tokens,),
            },
        )
        store.copy_tokens(tokens, writer.write)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    telemetry.add("spool_write_seconds", time.monotonic() - started)
    return _written_file(path, writer.digest.hexdigest(), writer.size)


def _atomic_array(path: Path, values: np.ndarray) -> VerifiedFile:
    temporary = path.with_name(path.stem + ".tmp.npy")
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    with progress_phase(
        f"data_array_write_fsync_{path.name}",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": path.name},
    ) as progress:
        with temporary.open("xb") as output:
            writer = HashingWriter(output)
            np.lib.format.write_array_header_1_0(
                writer,
                {
                    "descr": np.lib.format.dtype_to_descr(values.dtype),
                    "fortran_order": False,
                    "shape": values.shape,
                },
            )
            writer.write(memoryview(values).cast("B"))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        progress.update(
            completed_work=writer.size,
            total_work=writer.size,
            unit="bytes",
            raw_counters={"artifact": path.name, "artifact_bytes": writer.size},
        )
    return _written_file(path, writer.digest.hexdigest(), writer.size)


def _write_preparation_receipt(
    root: Path, manifest_sha256: str, telemetry: PreparationTelemetry, *, reused: bool
) -> None:
    receipt = root.with_name(root.name + ".preparation.json")
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "cache_reused": reused,
        **telemetry.snapshot(),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root.parent, prefix=receipt.name + ".", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(receipt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_prepared_directory(staged: Path, destination: Path) -> None:
    """Atomically publish without replacing a previously published cache."""
    if os.name == "nt":
        # Windows rename fails whenever the destination already exists.
        os.rename(staged, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError(
                "atomic no-replace preparation publication requires renamex_np"
            )
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staged), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError(
                "atomic no-replace preparation publication requires renameat2"
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(
            error,
            f"cannot publish prepared cache without replacing existing destination: {destination}",
        )


def _prepare_verified_prefix(
    config: RunConfig,
    tokenizer: Tokenizer,
    request: PreparedPrefix,
    *,
    identity: dict,
    root: Path,
    telemetry: PreparationTelemetry,
    resource_envelope: ResourceEnvelope | None,
) -> PreparedData:
    """Derive exact IDs from a cold-verified parent, without calling an encoder."""
    if (
        config.dataset.source != "local_text"
        or config.dataset.corpus_release_path is not None
        or config.dataset.allocation_manifest_path is not None
        or config.model.memory in {"byte", "portable"}
    ):
        raise ValueError("prepared prefix requires plain all-token local_text")
    parent_manifest = json.loads((request.root / "manifest.json").read_text())
    if parent_manifest.get("manifest_sha256") != request.manifest_sha256:
        raise ValueError("prepared prefix parent manifest pin mismatch")
    parent = load_prepared_data(request.root, byte_enabled=False)
    manifest = parent.manifest
    if manifest["manifest_sha256"] != request.manifest_sha256:
        raise ValueError(
            "prepared prefix parent manifest pin changed during verification"
        )
    if (
        manifest["packing_version"] != PACKING_VERSION
        or manifest.get("supervision") != {"kind": "all_tokens"}
        or manifest.get("allocation") is not None
    ):
        raise ValueError(
            "prepared prefix requires current all-token arrays without sidecars"
        )
    parent_identity = manifest["cache_identity"]
    expected = {**identity, "dataset": dict(identity["dataset"])}
    # Only selection size may differ. A pin alone cannot authorize relabeling
    # another producer's IDs under the current ordinary preparation cache key.
    for key in ("train_max_documents", "train_max_tokens"):
        expected["dataset"][key] = parent_identity["dataset"][key]
    if expected != parent_identity:
        raise ValueError("prepared prefix source/tokenizer/selection identity mismatch")
    count = config.dataset.train_max_documents
    tokens = config.dataset.train_max_tokens
    parent_count = parent_identity["dataset"]["train_max_documents"]
    if not (0 < count < parent_count and 0 < tokens < len(parent.train)):
        raise ValueError("prepared prefix must reduce both document and token counts")
    for split, values in (("train", parent.train), ("validation", parent.validation)):
        stats = manifest[split]
        documents = parent_identity["dataset"][f"{split}_max_documents"]
        if (
            stats.get("acquired_documents") != documents
            or stats.get("retained_documents") != documents
            or stats.get("skipped_documents") != 0
            or stats.get("truncated_documents") != 0
            or stats.get("tokens") != len(values)
        ):
            raise ValueError(
                "prepared prefix requires complete unskipped parent documents"
            )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None or parent.train[tokens - 1] != eos:
        raise ValueError(
            "prepared prefix token cap is not a whole-document EOS boundary"
        )
    # Bound temporary memory even for billion-token parents. Counting the whole
    # parent also rejects embedded EOS tokens, which cannot identify documents.
    parent_eos = prefix_eos = 0
    for start in range(0, len(parent.train), 1_048_576):
        end = min(start + 1_048_576, len(parent.train))
        parent_eos += int(np.count_nonzero(parent.train[start:end] == eos))
        if start < tokens:
            prefix_eos += int(
                np.count_nonzero(parent.train[start : min(end, tokens)] == eos)
            )
        if resource_envelope is not None:
            check_envelope(
                resource_envelope,
                workspace=root.parent,
                rss_bytes=current_process_rss_bytes(),
            )
    if parent_eos != parent_count or prefix_eos != count:
        raise ValueError("prepared prefix document count/EOS boundary mismatch")
    matched = config.training.matched_stories
    if matched is not None and matched.unpadded_documents:
        from klab.data.story_packing import MatchedStoryDataset

        for values in (parent.train[:tokens], parent.validation):
            MatchedStoryDataset(
                values,
                config.training.seq_len,
                eos,
                layout="story",
                targets_per_update=1,
                unpadded_documents=True,
            )
    elif (
        tokens < config.training.seq_len + 1
        or len(parent.validation) < config.training.seq_len + 1
    ):
        raise ValueError("prepared split lacks a full next-token block")
    if (root / "manifest.json").is_file():
        return load_prepared_data(root, byte_enabled=False, expected_identity=identity)
    if root.exists() or root.is_symlink():
        raise RuntimeError(f"unverified prepared-data destination exists: {root}")
    from klab.workspace_preflight import check_storage, require_storage

    require_storage(
        [
            check_storage(
                root, projected_bytes=(tokens + len(parent.validation)) * 4 + 65536
            )
        ]
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    # Separate staging leaves any failed streaming chunks intact and auditable.
    staged = Path(tempfile.mkdtemp(prefix=root.name + ".prefix-", dir=root.parent))
    proofs = {}
    result = dict(manifest)
    for split, values in (
        ("train", parent.train[:tokens]),
        ("validation", parent.validation),
    ):
        if resource_envelope is not None:
            check_envelope(
                resource_envelope,
                workspace=staged,
                rss_bytes=current_process_rss_bytes(),
            )
        proof = _atomic_array(staged / f"{split}.npy", values)
        telemetry.logical_output_bytes += proof.size_bytes
        telemetry.output_tokens += len(values)
        proofs[f"{split}.npy"] = proof
        result[split] = {
            **manifest[split],
            "shape": [len(values)],
            "tokens": len(values),
            "sha256": proof.sha256,
            "size_bytes": proof.size_bytes,
            "artifact_bytes": proof.size_bytes,
            "output_tokens": len(values),
        }
        if split == "train":
            result[split].update(acquired_documents=count, retained_documents=count)
    # Revalidate the sealed parent's file fingerprints after reading/copying.
    load_prepared_data(
        request.root,
        byte_enabled=False,
        verification="structural",
        receipt=parent.receipt,
    )
    result.update(
        cache_identity=identity,
        settings_sha256=hashlib.sha256(canonical_json(identity)).hexdigest(),
        source_identity_sha256=source_identity()["sha256"],
        prefix_derivation={
            "kind": "verified-whole-document-prefix-v1",
            "parent_manifest_sha256": request.manifest_sha256,
            "parent_cache_identity": parent_identity,
            "parent_train_sha256": manifest["train"]["sha256"],
            "train_documents": count,
            "train_tokens": tokens,
        },
    )
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    with (staged / "manifest.json").open("xb") as handle:
        handle.write(canonical_json(result) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if resource_envelope is not None:
        check_envelope(
            resource_envelope, workspace=staged, rss_bytes=current_process_rss_bytes()
        )
    _publish_prepared_directory(staged, root)
    fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    sealed = _receipt_from_proofs(root, result, _relocate_proofs(root, proofs))
    prepared = load_prepared_data(
        root,
        byte_enabled=False,
        expected_identity=identity,
        verification="structural",
        receipt=sealed,
    )
    _write_preparation_receipt(root, result["manifest_sha256"], telemetry, reused=False)
    return prepared


def _prepare_verified_append(
    config: RunConfig,
    tokenizer: Tokenizer,
    request: PreparedAppend,
    *,
    identity: dict,
    root: Path,
    telemetry: PreparationTelemetry,
    resource_envelope: ResourceEnvelope | None,
    tokenizer_batch_documents: int,
    tokenizer_batch_source_bytes: int,
) -> PreparedData:
    """Authenticate old raw bytes/IDs, then encode only the fresh JSONL tail."""
    if (
        config.dataset.source != "local_text"
        or config.dataset.corpus_release_path is not None
        or config.dataset.allocation_manifest_path is not None
        or config.model.memory in {"byte", "portable"}
    ):
        raise ValueError("prepared append requires plain all-token local_text")
    parent = load_prepared_data(request.root, byte_enabled=False)
    manifest = parent.manifest
    if manifest["manifest_sha256"] != request.manifest_sha256:
        raise ValueError("prepared append parent manifest pin mismatch")
    if (
        manifest["packing_version"] != PACKING_VERSION
        or manifest.get("supervision") != {"kind": "all_tokens"}
        or manifest.get("allocation") is not None
    ):
        raise ValueError(
            "prepared append requires current all-token arrays without sidecars"
        )
    parent_identity = manifest["cache_identity"]
    expected = {
        **identity,
        "dataset": dict(identity["dataset"]),
        "local_chat_source": dict(identity["local_chat_source"]),
    }
    for key in ("revision", "train_max_documents", "train_max_tokens"):
        expected["dataset"][key] = parent_identity["dataset"][key]
    expected["local_chat_source"]["train_sha256"] = parent_identity[
        "local_chat_source"
    ]["train_sha256"]
    if expected != parent_identity:
        raise ValueError(
            "prepared append source/tokenizer/producer/validation identity mismatch"
        )
    old_count = parent_identity["dataset"]["train_max_documents"]
    count = config.dataset.train_max_documents
    tokens = config.dataset.train_max_tokens
    if not (0 < old_count < count and len(parent.train) < tokens):
        raise ValueError("prepared append must increase both document and token counts")
    for split, values in (("train", parent.train), ("validation", parent.validation)):
        stats = manifest[split]
        documents = parent_identity["dataset"][f"{split}_max_documents"]
        if (
            stats.get("acquired_documents") != documents
            or stats.get("retained_documents") != documents
            or stats.get("skipped_documents") != 0
            or stats.get("truncated_documents") != 0
            or stats.get("tokens") != len(values)
        ):
            raise ValueError(
                "prepared append requires complete unskipped parent documents"
            )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None or parent.train[-1] != eos:
        raise ValueError("prepared append parent must end at EOS")
    matched = config.training.matched_stories
    if matched is not None and len(parent.train) not in matched.frozen_update_ends:
        raise ValueError("prepared append requires the frozen parent update boundary")

    def envelope():
        if resource_envelope is not None:
            check_envelope(
                resource_envelope,
                workspace=root.parent,
                rss_bytes=current_process_rss_bytes(),
            )

    parent_eos = 0
    for start in range(0, len(parent.train), 1_048_576):
        parent_eos += int(
            np.count_nonzero(parent.train[start : start + 1_048_576] == eos)
        )
        envelope()
    if parent_eos != old_count:
        raise ValueError("prepared append parent document/EOS count mismatch")
    cached = (root / "manifest.json").is_file()
    if not cached and (root.exists() or root.is_symlink()):
        raise RuntimeError(f"unverified prepared-data destination exists: {root}")
    from klab.workspace_preflight import check_storage, require_storage

    path = config.dataset.train_path
    assert path is not None
    # Final plus spool, tail array plus spool, validation and raw tail staging.
    projected = (
        (2 * tokens + 2 * (tokens - len(parent.train)) + len(parent.validation)) * 4
        + path.stat().st_size
        + 65536
    )
    staged = tail_dir = tail_file = None
    if not cached:
        require_storage([check_storage(root, projected_bytes=projected)])
        root.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=root.name + ".append-", dir=root.parent))
        tail_dir = staged / "tail"
        tail_dir.mkdir()
        tail_file = tail_dir / "source.jsonl"
    prefix_digest = hashlib.sha256()
    tail_digest = hashlib.sha256()
    whole_digest = hashlib.sha256()
    raw_limit = 6 * tokenizer_batch_source_bytes + 1024
    raw_prefix_bytes = 0
    # One record per line, as required by the native local_text reader. Hash the
    # exact first N lines against the parent's whole raw-source digest; parents
    # that selected only part of a larger raw file are deliberately ineligible.
    with (
        path.open("rb") as source,
        tail_file.open("xb") if tail_file else nullcontext(None) as output,
    ):
        for number in range(count):
            line = source.readline(raw_limit + 1)
            if not line or len(line) > raw_limit or not line.endswith(b"\n"):
                raise ValueError(
                    "prepared append requires bounded complete JSONL records"
                )
            whole_digest.update(line)
            if number < old_count:
                prefix_digest.update(line)
                raw_prefix_bytes += len(line)
            else:
                tail_digest.update(line)
                if output is not None:
                    output.write(line)
            if number % 1024 == 0:
                envelope()
        if source.read(1):
            raise ValueError(
                "prepared append requires the exact complete selected raw source"
            )
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    if (
        prefix_digest.hexdigest()
        != parent_identity["local_chat_source"]["train_sha256"]
    ):
        raise ValueError("prepared append historical raw prefix digest mismatch")
    if whole_digest.hexdigest() != identity["local_chat_source"]["train_sha256"]:
        raise ValueError("prepared append source changed during verification")
    if cached:
        return load_prepared_data(root, byte_enabled=False, expected_identity=identity)
    assert staged is not None and tail_dir is not None and tail_file is not None
    tail_config = config.dataset.model_copy(
        update={
            "train_path": tail_file,
            "train_max_documents": count - old_count,
            "train_max_tokens": tokens - len(parent.train),
        }
    )
    # The authenticated parent already checked its history against the identical
    # validation source. Apply that same native check to every new tail document.
    _assert_local_chat_disjoint(
        tail_config, local_text_source_bytes=tokenizer_batch_source_bytes
    )
    tail_proofs: dict[str, VerifiedFile] = {}
    _collect_streaming(
        tail_config,
        tokenizer,
        "train",
        tail_dir,
        selected_documents=count - old_count,
        resource_envelope=resource_envelope,
        telemetry=telemetry,
        tokenizer_batch_documents=tokenizer_batch_documents,
        tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
        proofs=tail_proofs,
    )
    tail = np.load(tail_dir / "train.npy", mmap_mode="r", allow_pickle=False)
    tail_eos = 0
    for start in range(0, len(tail), 1_048_576):
        tail_eos += int(np.count_nonzero(tail[start : start + 1_048_576] == eos))
        envelope()
    if len(parent.train) + len(tail) != tokens or tail_eos != count - old_count:
        raise ValueError("prepared append tail token/document count mismatch")
    spool = _ArraySpool(staged / "train.npy", np.dtype(np.int32), telemetry)
    for values in (parent.train, tail):
        for start in range(0, len(values), 1_048_576):
            spool.append_array(values[start : start + 1_048_576])
            envelope()
    proofs = {
        "train.npy": spool.finish(),
        "validation.npy": _atomic_array(staged / "validation.npy", parent.validation),
    }
    train = np.load(staged / "train.npy", mmap_mode="r", allow_pickle=False)
    if matched is not None and matched.unpadded_documents:
        from klab.data.story_packing import MatchedStoryDataset

        for values in (train, parent.validation):
            MatchedStoryDataset(
                values,
                config.training.seq_len,
                eos,
                layout="story",
                targets_per_update=1,
                unpadded_documents=True,
            )
    elif min(len(train), len(parent.validation)) < config.training.seq_len + 1:
        raise ValueError("prepared split lacks a full next-token block")
    # Verify sources and parent file fingerprints again before publication.
    if _local_chat_identity(config.dataset) != identity["local_chat_source"]:
        raise ValueError("prepared append source changed before publication")
    load_prepared_data(
        request.root,
        byte_enabled=False,
        verification="structural",
        receipt=parent.receipt,
    )
    result = dict(manifest)
    result.pop("prefix_derivation", None)
    result.update(
        revision=config.dataset.revision,
        cache_identity=identity,
        settings_sha256=hashlib.sha256(canonical_json(identity)).hexdigest(),
        source_identity_sha256=source_identity()["sha256"],
        train={
            **manifest["train"],
            "shape": [tokens],
            "tokens": tokens,
            "acquired_documents": count,
            "retained_documents": count,
            "output_tokens": tokens,
            "artifact_bytes": proofs["train.npy"].size_bytes,
            "sha256": proofs["train.npy"].sha256,
            "size_bytes": proofs["train.npy"].size_bytes,
        },
        append_derivation={
            "kind": "verified-whole-document-append-v1",
            "parent_manifest_sha256": request.manifest_sha256,
            "parent_cache_identity": parent_identity,
            "parent_train_sha256": manifest["train"]["sha256"],
            "parent_train_tokens": len(parent.train),
            "parent_train_documents": old_count,
            "parent_raw_prefix_bytes": raw_prefix_bytes,
            "parent_raw_prefix_sha256": prefix_digest.hexdigest(),
            "tail_raw_sha256": tail_digest.hexdigest(),
            "tail_array_sha256": tail_proofs["train.npy"].sha256,
            "tail_documents": count - old_count,
            "tail_tokens": len(tail),
        },
    )
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    with (staged / "manifest.json").open("xb") as handle:
        handle.write(canonical_json(result) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Tail files are owned scratch, not extra published artifact dependencies.
    del tail
    shutil.rmtree(tail_dir)
    fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    envelope()
    _publish_prepared_directory(staged, root)
    fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    sealed = _receipt_from_proofs(root, result, _relocate_proofs(root, proofs))
    prepared = load_prepared_data(
        root,
        byte_enabled=False,
        expected_identity=identity,
        verification="structural",
        receipt=sealed,
    )
    _write_preparation_receipt(root, result["manifest_sha256"], telemetry, reused=False)
    return prepared


def _prepare_data(
    config: RunConfig,
    tokenizer: Tokenizer,
    *,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    observer: BottleneckObserver | None = None,
    prefix_from: PreparedPrefix | None = None,
    append_from: PreparedAppend | None = None,
    require_prepared: bool = False,
) -> PreparedData:
    """Prepare immutable IDs and causal sidecars with an optional allocation."""
    tokenizer_batch_documents, tokenizer_batch_source_bytes = (
        validate_tokenizer_batch_limits(
            tokenizer_batch_documents, tokenizer_batch_source_bytes
        )
    )
    streaming = config.dataset.source in {"local_stories", "local_text", "snapshot"}
    if (
        streaming
        and resource_envelope is not None
        and not resource_envelope.spill_to_disk
    ):
        raise ValueError(
            "streaming preparation requires resource_envelope.spill_to_disk=true"
        )
    telemetry = PreparationTelemetry(config.dataset.cache_dir, observer=observer)
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=config.dataset.cache_dir,
            rss_bytes=telemetry.snapshot()["current_rss_bytes"],
        )
    local_stories = (
        verify_snapshot(config.dataset)
        if config.dataset.source == "local_stories"
        else None
    )
    generic_snapshot = None
    token_mixture = None
    if config.dataset.source == "local_token_mixture":
        from klab.corpus.mixture import verify_mixture_dataset

        if config.model.memory in {"byte", "portable"}:
            raise ValueError("token-ID mixture has no verified byte-address sidecar")
        token_mixture = verify_mixture_dataset(
            config.dataset, config.tokenizer.path, config.model.vocab_size
        )
        if (
            tokenizer.to_str()
            != Tokenizer.from_file(str(config.tokenizer.path)).to_str()
        ):
            raise ValueError(
                "supplied tokenizer differs from verified token mixture tokenizer"
            )
    if config.dataset.source == "snapshot":
        from klab.data.sources import verify_snapshot as verify_generic_snapshot

        generic_snapshot = verify_generic_snapshot(config.dataset)
    if config.dataset.source == "snapshot" or config.tokenizer.origin is not None:
        from klab.data.tokenizer import verify_run_tokenizer

        verify_run_tokenizer(
            config, proof_store=proof_store, verification_mode=verification_mode
        )
        if (
            tokenizer.to_str()
            != Tokenizer.from_file(str(config.tokenizer.path)).to_str()
        ):
            raise ValueError(
                "supplied tokenizer differs from verified snapshot tokenizer"
            )
    selected_snapshot = generic_snapshot or local_stories
    corpus_export = (
        verify_release_export(
            config.dataset, proof_store=proof_store, verification_mode=verification_mode
        )
        if config.dataset.corpus_release_path is not None
        else None
    )
    if (
        corpus_export is not None
        and corpus_export["vocab_size"] != config.model.vocab_size
    ):
        raise ValueError("corpus export vocabulary size does not match run model")
    local_chat = _local_chat_identity(config.dataset)
    # Allocation attestations retain their declared package identity. Ordinary
    # data caches depend on preparation implementation, not training/report/docs.
    source_digest = (
        source_identity()["sha256"]
        if config.dataset.allocation_manifest_path is not None
        else preparation_source_identity()
    )
    allocation = None
    if config.dataset.allocation_manifest_path is not None:
        if config.dataset.source != "local_chat" or local_chat is None:
            raise ValueError("allocation manifests require a local_chat dataset")
        allocation = load_allocation_manifest(
            config.dataset.allocation_manifest_path,
            source_identity_sha256=source_digest,
            tokenizer_sha256=canonical_tokenizer_sha256(tokenizer),
        )
        corpus = allocation.payload["corpus"]
        assert isinstance(corpus, dict)
        if (
            corpus["train_jsonl_sha256"] != local_chat["train_sha256"]
            or corpus["validation_jsonl_sha256"] != local_chat["validation_sha256"]
        ):
            raise ValueError(
                "allocation manifest corpus digests do not match local_chat"
            )
    cache_identity = {
        "dataset": prepared_dataset_identity(config.dataset),
        "local_chat_source": local_chat,
        **(
            {
                "token_mixture_receipt_sha256": _sha256(
                    config.dataset.mixture_output_path / "receipt.json"
                )
            }
            if token_mixture is not None
            else {}
        ),
        **({"corpus_export": corpus_export} if corpus_export is not None else {}),
        **(
            {
                "local_stories_source_sha256": hashlib.sha256(
                    canonical_json(local_stories)
                ).hexdigest()
            }
            if local_stories is not None
            else {}
        ),
        **(
            {
                "snapshot_source_sha256": hashlib.sha256(
                    canonical_json(generic_snapshot)
                ).hexdigest()
            }
            if generic_snapshot is not None
            else {}
        ),
        "allocation_manifest_sha256": None if allocation is None else allocation.sha256,
        "packing": {
            "memory": config.model.memory,
            "memory_table_size": config.model.memory_table_size,
            "memory_ngram_size": config.model.memory_ngram_size,
        },
        "packing_version": PACKING_VERSION,
        "source_identity_sha256": source_digest,
        "tokenizer_sha256": canonical_tokenizer_sha256(tokenizer),
    }
    from klab.experiments.source_compatibility import active_source_compatibility

    compatibility = active_source_compatibility()
    if compatibility is not None and (
        prefix_from is not None or append_from is not None
    ):
        raise ValueError("prepared prefix cannot use source compatibility override")
    if compatibility is not None:
        baseline = compatibility["baseline_source_identity"]
        historical_source = (
            baseline["sha256"]
            if config.dataset.allocation_manifest_path is not None
            else preparation_source_identity(dict(baseline["files"]))
        )
        if (
            historical_source != source_digest
            or config.dataset.allocation_manifest_path is None
        ):
            historical_identity = {
                **cache_identity,
                "source_identity_sha256": historical_source,
            }
            historical_root = (
                config.dataset.cache_dir
                / hashlib.sha256(canonical_json(historical_identity)).hexdigest()[:16]
            )
            if not (historical_root / "manifest.json").is_file():
                raise ValueError(
                    "authenticated source compatibility requires an existing "
                    "historical prepared cache; new preparation is not authorized"
                )
            cached = load_prepared_data(
                historical_root,
                byte_enabled=config.model.memory in {"byte", "portable"},
                expected_identity=historical_identity,
                telemetry=telemetry,
                verification="deep",
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            if resource_envelope is not None:
                check_envelope(
                    resource_envelope,
                    workspace=historical_root,
                    rss_bytes=telemetry.snapshot()["current_rss_bytes"],
                )
            return cached
    binding = (
        {
            "dataset": {
                key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in config.dataset.model_dump(mode="python").items()
                if key
                in {
                    "source",
                    "train_path",
                    "validation_path",
                    "source_manifest_path",
                    "corpus_release_path",
                    "corpus_export_path",
                    "train_max_documents",
                    "validation_max_documents",
                    "train_max_tokens",
                    "validation_max_tokens",
                }
            },
            "tokenizer_path": str(config.tokenizer.path.resolve()),
            "packing": cache_identity["packing"],
            "packing_version": PACKING_VERSION,
        }
        if streaming
        else None
    )
    root = (
        config.dataset.cache_dir
        / hashlib.sha256(canonical_json(cache_identity)).hexdigest()[:16]
    )
    if append_from is not None:
        return _prepare_verified_append(
            config,
            tokenizer,
            append_from,
            identity=cache_identity,
            root=root,
            telemetry=telemetry,
            resource_envelope=resource_envelope,
            tokenizer_batch_documents=tokenizer_batch_documents,
            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
        )
    if prefix_from is not None:
        return _prepare_verified_prefix(
            config,
            tokenizer,
            prefix_from,
            identity=cache_identity,
            root=root,
            telemetry=telemetry,
            resource_envelope=resource_envelope,
        )
    manifest_path = root / "manifest.json"
    _train_path, _validation_path = root / "train.npy", root / "validation.npy"
    _train_supervision_path, _validation_supervision_path = (
        root / "train_supervision.npy",
        root / "validation_supervision.npy",
    )
    byte_enabled = config.model.memory in {"byte", "portable"}
    _train_byte_path, _validation_byte_path = (
        root / "train_byte_addresses.npy",
        root / "validation_byte_addresses.npy",
    )
    chunk_arrays = {"ids": np.dtype(np.int32)}
    if byte_enabled:
        chunk_arrays["byte_addresses"] = np.dtype(np.int32)
    recovered_receipt = None
    if streaming:
        assert binding is not None
        PreparationChunks.check_binding_collision(
            root.with_name(root.name + ".tmp"),
            cache_identity=cache_identity,
            binding=binding,
        )
        if (root / "staging.json").exists() or (root / "staging.json").is_symlink():
            recovered_receipt = PreparationChunks.recover_published(
                root,
                cache_identity=cache_identity,
                binding=binding,
                arrays=chunk_arrays,
                telemetry=telemetry,
            )
    if manifest_path.is_file():
        with (
            progress_phase(
                "data_cache_validation",
                completed_work=0,
                total_work=1,
                unit="cache_verifications",
                raw_counters={
                    "cache_reused": True,
                    "verification_only": True,
                    **telemetry.snapshot(),
                },
            ) as progress,
            (
                observer.phase(
                    "prepared_cache_validation",
                    host_kind="verification_bound",
                    cache_event="hit",
                )
                if observer is not None
                else nullcontext()
            ),
        ):
            cached = load_prepared_data(
                root,
                byte_enabled=byte_enabled,
                expected_identity=cache_identity,
                telemetry=telemetry,
                verification="deep" if recovered_receipt is None else "structural",
                receipt=recovered_receipt,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            _write_preparation_receipt(
                root, cached.manifest["manifest_sha256"], telemetry, reused=True
            )
            progress.update(
                completed_work=1,
                total_work=1,
                unit="cache_verifications",
                raw_counters={
                    "cache_reused": True,
                    "verification_only": True,
                    **telemetry.snapshot(),
                },
            )
            return cached
    if require_prepared:
        raise PreparedDataMissing(
            f"prepared data is not published at {root}; run `klab data prepare` "
            "with this config before training with --require-prepared"
        )
    if root.exists() or root.is_symlink():
        raise RuntimeError(f"unverified prepared-data destination exists: {root}")
    _assert_local_chat_disjoint(
        config.dataset,
        local_text_source_bytes=(
            tokenizer_batch_source_bytes
            if config.dataset.source == "local_text"
            else None
        ),
    )
    _assert_local_chat_supervision_consistent(config.dataset)
    from klab.workspace_preflight import (
        check_storage,
        projected_data_bytes,
        require_storage,
    )

    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=root.parent,
            rss_bytes=telemetry.snapshot()["current_rss_bytes"],
        )
    writable_ancestor = root.parent
    while not writable_ancestor.exists():
        writable_ancestor = writable_ancestor.parent
    if not writable_ancestor.is_dir() or not os.access(
        writable_ancestor, os.W_OK | os.X_OK
    ):
        raise PermissionError(
            f"prepared-data spill root is not writable: {root.parent}"
        )
    require_storage([check_storage(root, projected_bytes=projected_data_bytes(config))])
    temporary_root = root.with_name(root.name + ".tmp")
    chunks_owner = None
    token_stores = None
    if (
        TOKEN_STORE_ENABLED
        and config.dataset.source == "local_text"
        and not byte_enabled
        and allocation is None
        and _documents_encode_independently(tokenizer)
    ):
        assert local_chat is not None
        token_stores = _token_stores(
            config,
            tokenizer,
            local_chat,
            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
            resource_envelope=resource_envelope,
            telemetry=telemetry,
        )
    if token_stores is not None:
        # Tokens come from the reusable store, so staging holds only the final
        # arrays; it is removed if anything fails before publication.
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent)
        )
        staged = _STORE_STAGING.get()
        if staged is not None:
            staged.append(temporary_root)
    elif streaming:
        assert binding is not None
        chunks_owner = PreparationChunks(
            temporary_root,
            cache_identity=cache_identity,
            binding=binding,
            arrays=chunk_arrays,
            telemetry=telemetry,
        )
    elif token_mixture is not None:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent)
        )
    else:
        if temporary_root.exists() or temporary_root.is_symlink():
            raise RuntimeError(
                f"incomplete prepared-data sibling exists: {temporary_root}"
            )
        temporary_root.mkdir(parents=True, exist_ok=False)
    proofs: dict[str, VerifiedFile] = {}
    settings = (
        {
            "byte_table_size": config.model.memory_table_size,
            "byte_ngram_size": config.model.memory_ngram_size,
        }
        if byte_enabled
        else {}
    )
    if token_stores is not None:
        results = {}
        for split, (store, documents, tokens) in token_stores.items():
            proofs[f"{split}.npy"] = _copy_from_token_store(
                store, tokens, temporary_root / f"{split}.npy", telemetry
            )
            results[split] = {
                "acquired_documents": documents,
                "retained_documents": documents,
                "skipped_documents": 0,
                "truncated_documents": 0,
                "output_tokens": tokens,
                "artifact_bytes": proofs[f"{split}.npy"].size_bytes,
                "artifact_files": 1,
            }
            telemetry.records += documents
            telemetry.output_tokens += tokens
            telemetry.logical_output_bytes += proofs[f"{split}.npy"].size_bytes
            telemetry.shards[split] = {
                "token_store": str(store.root),
                "token_store_shards": len(store.shards),
            }
        train_stats = results["train"]
        validation_stats = results["validation"]
        train = np.load(temporary_root / "train.npy", mmap_mode="r", allow_pickle=False)
        validation = np.load(
            temporary_root / "validation.npy", mmap_mode="r", allow_pickle=False
        )
    elif config.dataset.source in {"local_stories", "local_text", "snapshot"}:
        if config.dataset.source == "local_stories":
            assert local_stories is not None
        assert chunks_owner is not None
        parallel_supported = _documents_encode_independently(tokenizer)
        results = {}
        with (
            chunks_owner,
            PreparationWorkers(
                tokenizer,
                temporary_root,
                envelope=resource_envelope,
                parallel_supported=parallel_supported,
                token_budget=config.dataset.train_max_tokens
                + config.dataset.validation_max_tokens,
            ) as workers,
        ):
            telemetry.shard_workers = workers.count
            telemetry.tokenizer_host_work_plan = asdict(workers.plan)
            telemetry.tokenizer_rayon_threads = sum(
                e.rayon_threads for e in workers.encoders
            )
            for split in ("train", "validation"):
                chunks = chunks_owner.open_split(split)
                measured = PreparationTelemetry(temporary_root)
                chunks.telemetry = measured
                count = getattr(config.dataset, f"{split}_max_documents")
                if selected_snapshot is not None:
                    count = min(count, selected_snapshot["splits"][split]["count"])
                diagnostics = {}
                if workers.count > 1:
                    eos = tokenizer.token_to_id("<eos>")
                    if eos is None:
                        raise ValueError("tokenizer has no <eos> special token")
                    stats, shard_proofs, diagnostics = collect_parallel_stream(
                        _source_documents(
                            config.dataset,
                            split,
                            local_text_source_bytes=tokenizer_batch_source_bytes
                            if config.dataset.source == "local_text"
                            else None,
                        ),
                        chunks,
                        workers,
                        selected_documents=count,
                        max_tokens=getattr(config.dataset, f"{split}_max_tokens"),
                        batch_documents=tokenizer_batch_documents,
                        batch_source_bytes=tokenizer_batch_source_bytes,
                        transform=partial(_streaming_values, eos=eos, **settings),
                        telemetry=measured,
                    )
                    proofs.update(shard_proofs)
                else:
                    stats = _collect_streaming(
                        config.dataset,
                        tokenizer,
                        split,
                        temporary_root,
                        selected_documents=count,
                        resource_envelope=resource_envelope,
                        telemetry=measured,
                        encoder=workers.encoders[0],
                        tokenizer_batch_documents=tokenizer_batch_documents,
                        tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                        proofs=proofs,
                        chunks=chunks,
                        **settings,
                    )
                results[split] = stats
                telemetry.shards[split] = {**measured.snapshot(), **diagnostics}
                for name in (
                    "records",
                    "source_bytes",
                    "output_tokens",
                    "logical_input_bytes",
                    "logical_output_bytes",
                ):
                    setattr(
                        telemetry,
                        name,
                        getattr(telemetry, name) + getattr(measured, name),
                    )
                for name, elapsed in measured.durations.items():
                    telemetry.add(name, elapsed)
            telemetry.workers = workers.snapshot()
        train_stats = results["train"]
        validation_stats = results["validation"]
        train = np.load(temporary_root / "train.npy", mmap_mode="r", allow_pickle=False)
        validation = np.load(
            temporary_root / "validation.npy", mmap_mode="r", allow_pickle=False
        )
    elif token_mixture is not None:
        mixture_chunks = PreparationChunks(
            root.with_name(root.name + ".mixture.tmp"),
            cache_identity=cache_identity,
            binding={"stage": "mixture-target-packing", "identity": cache_identity},
            arrays={"ids": np.dtype(np.int32), "supervision": np.dtype(bool)},
            raw_byte_limit=128 * 1024**2,
            telemetry=telemetry,
        )
        with mixture_chunks:
            train, train_supervision, train_stats = _collect_token_mixture(
                config,
                tokenizer,
                token_mixture,
                temporary_root,
                proofs=proofs,
                chunks=mixture_chunks.open_split("train"),
            )
        train_byte = None
        validation, validation_supervision, validation_byte, validation_stats = (
            _collect(
                config.dataset,
                tokenizer,
                "validation",
                root=temporary_root,
                proofs=proofs,
                **settings,
            )
        )
    else:
        train, train_supervision, train_byte, train_stats = _collect(
            config.dataset, tokenizer, "train", **settings
        )
        validation, validation_supervision, validation_byte, validation_stats = (
            _collect(config.dataset, tokenizer, "validation", **settings)
        )
    unpadded = (
        config.training.matched_stories is not None
        and config.training.matched_stories.unpadded_documents
    )
    if unpadded:
        from klab.data.story_packing import MatchedStoryDataset

        eos = tokenizer.token_to_id("<eos>")
        if eos is None:
            raise ValueError("whole-document preparation requires EOS")
        for values in (train, validation):
            MatchedStoryDataset(
                values,
                config.training.seq_len,
                eos,
                layout="story",
                targets_per_update=1,
                unpadded_documents=True,
            )
    elif (
        len(train) < config.training.seq_len + 1
        or len(validation) < config.training.seq_len + 1
    ):
        raise ValueError("prepared split lacks a full next-token block")
    mask_enabled = config.dataset.source == "local_token_mixture" or (
        config.dataset.source == "local_chat"
        and (
            not bool(np.all(train_supervision))
            or not bool(np.all(validation_supervision))
        )
    )
    allocation_sides = None
    if allocation is not None:
        train_sides = allocation.split("train", token_count=len(train))
        validation_sides = allocation.split("validation", token_count=len(validation))
        allocation_sides = (train_sides, validation_sides)
        for name, values in (
            ("train_owner_ids.npy", train_sides.owner),
            ("validation_owner_ids.npy", validation_sides.owner),
            ("train_semantic_queries.npy", train_sides.semantic_queries),
            ("validation_semantic_queries.npy", validation_sides.semantic_queries),
            ("train_semantic_mask.npy", train_sides.semantic_mask),
            ("validation_semantic_mask.npy", validation_sides.semantic_mask),
        ):
            if values is not None:
                proofs[name] = _atomic_array(temporary_root / name, values)
    if config.dataset.source not in {
        "local_stories",
        "local_text",
        "snapshot",
        "local_token_mixture",
    }:
        proofs["train.npy"] = _atomic_array(temporary_root / "train.npy", train)
        proofs["validation.npy"] = _atomic_array(
            temporary_root / "validation.npy", validation
        )
        if mask_enabled:
            proofs["train_supervision.npy"] = _atomic_array(
                temporary_root / "train_supervision.npy", train_supervision
            )
            proofs["validation_supervision.npy"] = _atomic_array(
                temporary_root / "validation_supervision.npy", validation_supervision
            )
        if byte_enabled:
            assert train_byte is not None and validation_byte is not None
            proofs["train_byte_addresses.npy"] = _atomic_array(
                temporary_root / "train_byte_addresses.npy", train_byte
            )
            proofs["validation_byte_addresses.npy"] = _atomic_array(
                temporary_root / "validation_byte_addresses.npy", validation_byte
            )
    attributions = {
        "tinystories": {
            "license": "CDLA-Sharing-1.0",
            "source_attribution": "roneneldan/TinyStories",
        },
        "fineweb_edu": {
            "license": "ODC-By-1.0; Common Crawl Terms of Use",
            "source_attribution": "HuggingFaceFW/fineweb-edu",
        },
        "cosmopedia": {
            "license": "Apache-2.0",
            "source_attribution": "HuggingFaceTB/cosmopedia dataset card",
        },
        "local_chat": {
            "license": config.dataset.license,
            "source_attribution": "user-provided local_chat",
        },
        "local_text": {
            "license": config.dataset.license,
            "source_attribution": "frozen corpus local_text export",
        },
        "local_token_mixture": {
            "license": config.dataset.license,
            "source_attribution": "verified frozen corpus token mixture",
        },
        "snapshot": {
            "license": config.dataset.license,
            "source_attribution": "declarative pinned dataset snapshot",
        },
        "local_stories": {
            "license": "CDLA-Sharing-1.0",
            "source_attribution": "roneneldan/TinyStories pinned local snapshot",
        },
        "synthetic": {
            "license": "synthetic fixture",
            "source_attribution": "klab synthetic",
        },
        "instruction_reference": {
            "license": "synthetic fixture",
            "source_attribution": "klab instruction_reference",
        },
        "chat_recall": {
            "license": "synthetic fixture",
            "source_attribution": "klab chat_recall",
        },
        "engram_recall": {
            "license": "synthetic fixture",
            "source_attribution": "klab engram_recall",
        },
        "withheld_facts": {
            "license": "project fixture",
            "source_attribution": "klab withheld_facts",
        },
    }

    def metadata(path: Path, **kwargs: object) -> dict[str, object]:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        proof = proofs[path.name]
        expected_dtype = np.dtype(kwargs.get("dtype", np.int32))
        if values.ndim != kwargs.get("dimensions", 1) or values.dtype != expected_dtype:
            raise ValueError(f"packed array has unexpected shape or dtype: {path}")
        return {
            "dtype": values.dtype.name,
            "shape": list(values.shape),
            "tokens": int(values.shape[0]),
            "sha256": proof.sha256,
            "size_bytes": proof.size_bytes,
        }

    manifest = {
        "packing_version": PACKING_VERSION,
        "cache_identity": cache_identity,
        **(
            {"dataset_snapshot": generic_snapshot}
            if generic_snapshot is not None
            else {}
        ),
        "source": config.dataset.source,
        "revision": config.dataset.revision,
        "dataset_config": config.dataset.dataset_config,
        **({"token_mixture": token_mixture} if token_mixture is not None else {}),
        **attributions[config.dataset.source],
        **({"corpus_export": corpus_export} if corpus_export is not None else {}),
        "tokenizer_sha256": canonical_tokenizer_sha256(tokenizer),
        "settings_sha256": hashlib.sha256(canonical_json(cache_identity)).hexdigest(),
        "source_identity_sha256": source_identity()["sha256"],
        "train": {
            **train_stats,
            **metadata(temporary_root / "train.npy"),
        },
        "validation": {
            **validation_stats,
            **metadata(temporary_root / "validation.npy"),
        },
        "supervision": {"kind": "all_tokens"}
        if not mask_enabled
        else {
            "kind": "token-loss-mask-v1",
            "train": metadata(
                temporary_root / "train_supervision.npy", dtype=np.dtype(bool)
            ),
            "validation": metadata(
                temporary_root / "validation_supervision.npy", dtype=np.dtype(bool)
            ),
        },
        "byte_addressing": None
        if not byte_enabled
        else {
            "kind": "raw-utf8-suffix-v1",
            "table_size": config.model.memory_table_size,
            "ngram_size": config.model.memory_ngram_size,
            "train": metadata(temporary_root / "train_byte_addresses.npy"),
            "validation": metadata(temporary_root / "validation_byte_addresses.npy"),
        },
        "allocation": None
        if allocation is None
        else {
            "format": "klab-prepared-allocation-v1",
            "manifest_sha256": allocation.sha256,
            "owner_codes": {"neural": 0, "lexical": 1, "semantic": 2, "hybrid": 3},
            "resource_regime": allocation.payload.get("resource_regime"),
            "ownership_profile": allocation.payload.get("ownership_profile"),
            "semantic": allocation.semantic,
            "train": {
                "owner": metadata(
                    temporary_root / "train_owner_ids.npy", dtype=np.dtype(np.uint8)
                ),
                **(
                    {}
                    if allocation_sides is None
                    or allocation_sides[0].semantic_queries is None
                    else {
                        "semantic_queries": metadata(
                            temporary_root / "train_semantic_queries.npy",
                            dtype=np.dtype(np.float32),
                            dimensions=2,
                        ),
                        "semantic_mask": metadata(
                            temporary_root / "train_semantic_mask.npy",
                            dtype=np.dtype(bool),
                        ),
                    }
                ),
            },
            "validation": {
                "owner": metadata(
                    temporary_root / "validation_owner_ids.npy",
                    dtype=np.dtype(np.uint8),
                ),
                **(
                    {}
                    if allocation_sides is None
                    or allocation_sides[1].semantic_queries is None
                    else {
                        "semantic_queries": metadata(
                            temporary_root / "validation_semantic_queries.npy",
                            dtype=np.dtype(np.float32),
                            dimensions=2,
                        ),
                        "semantic_mask": metadata(
                            temporary_root / "validation_semantic_mask.npy",
                            dtype=np.dtype(bool),
                        ),
                    }
                ),
            },
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=temporary_root,
            rss_bytes=telemetry.snapshot()["current_rss_bytes"],
            pending_workers=0,
            queue_depth=0,
        )
    manifest_path = temporary_root / "manifest.json"
    finalize_started = time.monotonic()
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging_fd = os.open(temporary_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(staging_fd)
    finally:
        os.close(staging_fd)
    telemetry.add("finalize_fsync_seconds", time.monotonic() - finalize_started)
    root.parent.mkdir(parents=True, exist_ok=True)
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=root.parent,
            rss_bytes=telemetry.snapshot()["current_rss_bytes"],
            pending_workers=0,
            queue_depth=0,
        )
    finalize_started = time.monotonic()
    _publish_prepared_directory(temporary_root, root)
    directory_fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    telemetry.add("finalize_fsync_seconds", time.monotonic() - finalize_started)
    if chunks_owner is not None:
        chunks_owner.cleanup_published(root)
    if token_mixture is not None:
        # Published arrays are independently owned and verified; raw chunks are
        # only retained until the complete prepared bundle commits.
        shutil.rmtree(mixture_chunks.root)
    with (
        progress_phase(
            "data_cache_validation",
            completed_work=0,
            total_work=1,
            unit="cache_verifications",
            raw_counters=telemetry.snapshot(),
        ) as progress,
        (
            observer.phase(
                "prepared_publication_validation",
                host_kind="verification_bound",
                cache_event="miss",
            )
            if observer is not None
            else nullcontext()
        ),
    ):
        sealed = _receipt_from_proofs(root, manifest, _relocate_proofs(root, proofs))
        prepared = load_prepared_data(
            root,
            byte_enabled=byte_enabled,
            expected_identity=cache_identity,
            telemetry=telemetry,
            verification="structural",
            receipt=sealed,
        )
        _write_preparation_receipt(
            root, manifest["manifest_sha256"], telemetry, reused=False
        )
        progress.update(
            completed_work=1,
            total_work=1,
            unit="cache_verifications",
            raw_counters=telemetry.snapshot(),
        )
    return prepared


@_discard_store_staging_on_failure
def prepare_data(
    config: RunConfig,
    tokenizer: Tokenizer,
    *,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    observer: BottleneckObserver | None = None,
    prefix_from: PreparedPrefix | None = None,
    append_from: PreparedAppend | None = None,
    require_prepared: bool = False,
) -> PreparedData:
    """Prepare data and mark only new caches owned by the selected workspace.

    ``prefix_from`` explicitly derives a smaller, whole-document local_text
    selection from a pinned parent cache. Sources and parent bytes are verified;
    validation is unchanged and no document is re-encoded. Other dataset kinds,
    masks, sidecars and source-compatibility overrides are deliberately rejected.

    ``append_from`` authenticates a same-producer complete raw JSONL prefix,
    copies historical IDs unchanged and encodes only the fresh complete tail.
    Matched-story callers must freeze the parent update boundary; full-state
    continuation additionally verifies the historical group settings/cursor.

    ``require_prepared`` refuses to tokenize or pack: an already published cache
    is verified and returned, and a miss raises ``PreparedDataMissing``.
    """
    if prefix_from is not None and append_from is not None:
        raise ValueError("prepared prefix and append are mutually exclusive")
    if config.model.memory_token_mapping is not None:
        from klab.memory_tokenizer import validate_memory_token_map

        validate_memory_token_map(
            config.model.memory_token_mapping.model_dump(mode="json"),
            tokenizer_path=config.tokenizer.path,
        )
        from klab.data.tokenizer import load_tokenizer

        if canonical_tokenizer_sha256(tokenizer) != canonical_tokenizer_sha256(
            load_tokenizer(config.tokenizer.path)
        ):
            raise ValueError(
                "memory mapping tokenizer object differs from configured tokenizer"
            )
    tokenizer_batch_documents, tokenizer_batch_source_bytes = (
        validate_tokenizer_batch_limits(
            tokenizer_batch_documents, tokenizer_batch_source_bytes
        )
    )
    if resource_envelope is not None:
        if (
            config.dataset.source in {"local_stories", "local_text", "snapshot"}
            and not resource_envelope.spill_to_disk
        ):
            raise ValueError(
                "streaming preparation requires resource_envelope.spill_to_disk=true"
            )
        check_envelope(
            resource_envelope,
            workspace=config.dataset.cache_dir,
            rss_bytes=current_process_rss_bytes(),
        )
    workspace = ensure_work_dir()
    base = config.dataset.cache_dir
    if (
        base.is_symlink()
        or not base.resolve().is_relative_to(workspace)
        or base.resolve() == workspace
    ):
        with (
            campaign_lock(base),
            observer.phase("data_prepare") if observer is not None else nullcontext(),
        ):
            return _prepare_data(
                config,
                tokenizer,
                resource_envelope=resource_envelope,
                tokenizer_batch_documents=tokenizer_batch_documents,
                tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                proof_store=proof_store,
                verification_mode=verification_mode,
                observer=observer,
                prefix_from=prefix_from,
                append_from=append_from,
                require_prepared=require_prepared,
            )
    with (
        campaign_lock(base),
        campaign_lock(workspace),
        observer.phase("data_prepare") if observer is not None else nullcontext(),
    ):
        existing = {child.name for child in base.iterdir()} if base.is_dir() else set()
        prepared = _prepare_data(
            config,
            tokenizer,
            resource_envelope=resource_envelope,
            tokenizer_batch_documents=tokenizer_batch_documents,
            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
            proof_store=proof_store,
            verification_mode=verification_mode,
            observer=observer,
            prefix_from=prefix_from,
            append_from=append_from,
            require_prepared=require_prepared,
        )
        if prepared.root.name not in existing:
            mark_prepared_cache(workspace, prepared.root)
        return prepared


def _supervised_blocks(
    supervision: np.ndarray, candidates: int, seq_len: int
) -> np.ndarray:
    """Indices of full blocks with at least one supervised target, in bounded chunks."""
    chunk = max(1, (1 << 26) // seq_len)
    kept = [np.empty(0, dtype=np.int64)]
    for first in range(0, candidates, chunk):
        last = min(candidates, first + chunk)
        targets = np.asarray(supervision[first * seq_len + 1 : last * seq_len + 1])
        kept.append(
            first + np.flatnonzero(targets.reshape(last - first, seq_len).any(axis=1))
        )
    return np.concatenate(kept).astype(np.int64, copy=False)


class TokenBlockDataset:
    def __init__(
        self,
        ids: np.ndarray,
        seq_len: int,
        byte_addresses: np.ndarray | None = None,
        supervision: np.ndarray | None = None,
        owner_ids: np.ndarray | None = None,
        semantic_queries: np.ndarray | None = None,
        semantic_mask: np.ndarray | None = None,
    ) -> None:
        if ids.ndim != 1 or ids.dtype != np.dtype(np.int32):
            raise ValueError("packed IDs must be one-dimensional int32")
        if supervision is not None and (
            supervision.ndim != 1
            or supervision.dtype != np.dtype(bool)
            or len(supervision) != len(ids)
        ):
            raise ValueError(
                "supervision mask must be one-dimensional bool aligned with IDs"
            )
        if byte_addresses is not None and len(byte_addresses) != len(ids):
            raise ValueError("byte address array must align with packed IDs")
        if owner_ids is not None and (
            owner_ids.ndim != 1
            or owner_ids.dtype != np.dtype(np.uint8)
            or len(owner_ids) != len(ids)
        ):
            raise ValueError("owner IDs must be one-dimensional uint8 aligned with IDs")
        if semantic_queries is not None and (
            semantic_queries.ndim != 2
            or semantic_queries.dtype != np.dtype(np.float32)
            or semantic_queries.shape[0] != len(ids)
            or semantic_mask is None
        ):
            raise ValueError(
                "semantic queries must be float32 [tokens,key_dim] with mask"
            )
        if semantic_mask is not None and (
            semantic_mask.ndim != 1
            or semantic_mask.dtype != np.dtype(bool)
            or len(semantic_mask) != len(ids)
        ):
            raise ValueError("semantic query mask must be bool aligned with IDs")
        self.ids, self.seq_len = ids, seq_len
        self.byte_addresses, self.supervision = byte_addresses, supervision
        self.owner_ids = owner_ids
        self.semantic_queries, self.semantic_mask = semantic_queries, semantic_mask
        candidates = (len(ids) - 1) // seq_len
        self.block_indices = (
            np.arange(candidates, dtype=np.int64)
            if supervision is None
            else _supervised_blocks(supervision, candidates, seq_len)
        )

    def __len__(self) -> int:
        return len(self.block_indices)

    def exposure_supply(self) -> dict[str, int]:
        """One prepared-data pass, using the same full blocks and target mask."""
        full_blocks = max(0, (len(self.ids) - 1) // self.seq_len)
        end = 1 + full_blocks * self.seq_len
        targets = (
            full_blocks * self.seq_len
            if self.supervision is None
            else int(np.count_nonzero(self.supervision[1:end]))
        )
        return {
            "raw_packed_positions": len(self.ids),
            "full_blocks": full_blocks,
            "eligible_blocks": len(self),
            "zero_target_blocks_skipped": full_blocks - len(self),
            "trailing_target_positions": max(0, len(self.ids) - end),
            "usable_supervised_targets_per_pass": targets,
        }

    def advise(self, indices: Sequence[int]) -> None:
        """Ask the kernel to start reading these blocks; never changes results.

        Training reads each update's blocks at random offsets of a memory-mapped
        file. Cold pages cost a disk read on the critical path while every
        device waits, so the trainer announces the next update's blocks and the
        kernel reads them while the current update computes. Arrays that are not
        memory-mapped, and platforms without ``madvise``, are left alone.
        """
        for array in (self.ids, self.supervision, self.byte_addresses):
            mapped = getattr(array, "_mmap", None)
            if mapped is None or not hasattr(mapped, "madvise"):
                continue
            base = getattr(array, "offset", 0)
            width = array.dtype.itemsize
            for index in indices:
                start = base + int(self.block_indices[index]) * self.seq_len * width
                aligned = start - start % mmap.ALLOCATIONGRANULARITY
                length = min(
                    start + (self.seq_len + 1) * width - aligned,
                    mapped.size() - aligned,
                )
                if length > 0:
                    try:
                        mapped.madvise(mmap.MADV_WILLNEED, aligned, length)
                    except OSError, ValueError:
                        return

    def numpy_block(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        block = int(self.block_indices[index])
        start = block * self.seq_len
        values = np.asarray(self.ids[start : start + self.seq_len + 1], dtype=np.int64)
        targets = values[1:].copy()
        if self.supervision is not None:
            targets[~self.supervision[start + 1 : start + self.seq_len + 1]] = -100
        addresses = (
            None
            if self.byte_addresses is None
            else np.asarray(
                self.byte_addresses[start : start + self.seq_len], dtype=np.int64
            )
        )
        return values[:-1], targets, addresses

    def numpy_microblock(
        self, index: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        inputs, targets, addresses = self.numpy_block(index)
        start = int(self.block_indices[index]) * self.seq_len
        owners = (
            None
            if self.owner_ids is None
            else np.asarray(
                self.owner_ids[start + 1 : start + self.seq_len + 1], dtype=np.uint8
            )
        )
        queries = (
            None
            if self.semantic_queries is None
            else np.asarray(
                self.semantic_queries[start : start + self.seq_len], dtype=np.float32
            )
        )
        mask = (
            None
            if self.semantic_mask is None
            else np.asarray(
                self.semantic_mask[start : start + self.seq_len], dtype=bool
            )
        )
        return inputs, targets, addresses, owners, queries, mask

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        inputs, targets, addresses = self.numpy_block(index)
        if addresses is None:
            return torch.from_numpy(inputs), torch.from_numpy(targets)
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(addresses),
        )


class PreparedDataMissing(RuntimeError):
    """Training required an already prepared cache and none is published."""


@dataclass(frozen=True)
class BatchCursor:
    epoch: int = 0
    next_block: int = 0


@lru_cache(maxsize=2)
def epoch_order(num_blocks: int, seed: int, epoch: int) -> torch.Tensor:
    """One epoch's block permutation, kept so each update does not redraw it.

    The permutation is a pure function of its arguments and costs O(num_blocks),
    which is seconds and hundreds of MB at tens of billions of tokens. Callers
    only index the shared result.
    """
    return torch.randperm(
        num_blocks, generator=torch.Generator().manual_seed(seed + epoch)
    )
