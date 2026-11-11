"""Native, bounded supplied-vector semantic mechanism probes."""

from __future__ import annotations

import hashlib
import math
import re
import stat
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import AttentionConfig, ModelConfig, StrictModel
from klab.documents import read_document
from klab.engram.packs import EncoderIdentity
from klab.engram.semantic import (
    MAX_ADAPTER_TRACE_ITEMS,
    SemanticQueryBatch,
    SemanticRetriever,
)
from klab.evaluation.inference import load_run
from klab.model.transformer import DenseLM
from klab.training.manifest import sha256_file

_MAX_INPUT_BYTES = 2 * 1024 * 1024
_MAX_ITEMS = 16


class _InitializedModel(StrictModel):
    kind: Literal["initialized"]
    model: ModelConfig
    attention: AttentionConfig
    seed: StrictInt

    @field_validator("seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be nonnegative")
        return value


class _CheckpointModel(StrictModel):
    kind: Literal["checkpoint"]
    run_id: StrictStr
    runs_dir: StrictStr
    checkpoint: StrictStr | None

    @model_validator(mode="after")
    def _required(self) -> _CheckpointModel:
        if (
            not self.run_id.strip()
            or not self.runs_dir.strip()
            or self.checkpoint is None
        ):
            raise ValueError(
                "checkpoint probes require run_id, runs_dir, and checkpoint"
            )
        return self


class _Query(StrictModel):
    id: StrictStr
    encoder: EncoderIdentity
    vectors: Any
    mask: Any | None = None
    as_of: Any | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query id must be nonblank")
        return value


class _InitializedWeights(StrictModel):
    kind: Literal["initialized"]
    seed: StrictInt

    @field_validator("seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("adapter seed must be nonnegative")
        return value


class _CheckpointWeights(StrictModel):
    kind: Literal["checkpoint"]


class _Attachment(StrictModel):
    name: StrictStr
    pack: StrictStr
    expected_pack_id: StrictStr
    query: StrictStr
    site: Literal["embedding", "after_block", "final"]
    block_index: StrictInt | None = None
    min_score: StrictFloat | StrictInt | None = None
    weights: Annotated[
        _InitializedWeights | _CheckpointWeights, Field(discriminator="kind")
    ]

    @field_validator("name", "pack", "expected_pack_id", "query")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("attachment strings must be nonblank")
        return value


class _Declaration(StrictModel):
    format: Literal["klab-semantic-probe-v1"]
    model: Annotated[_InitializedModel | _CheckpointModel, Field(discriminator="kind")]
    input_tokens: Any
    queries: tuple[_Query, ...]
    attachments: tuple[_Attachment, ...]

    @model_validator(mode="after")
    def _bounds(self) -> _Declaration:
        if not self.queries or len(self.queries) > _MAX_ITEMS:
            raise ValueError("queries must contain 1 through 16 entries")
        if not self.attachments or len(self.attachments) > _MAX_ITEMS:
            raise ValueError("attachments must contain 1 through 16 entries")
        if len({item.id for item in self.queries}) != len(self.queries):
            raise ValueError("query ids must be unique")
        if len({item.name for item in self.attachments}) != len(self.attachments):
            raise ValueError("attachment names must be unique")
        return self


def _failure(error: ValidationError) -> ValueError:
    return ValueError(
        "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in error.errors(include_input=False)
        )
    )


def _safe_path(source: Path, value: str, *, directory: bool = False) -> Path:
    path = Path(value)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"unsafe path traversal: {value}")
    if not path.is_absolute():
        path = source.parent / path
    # Do not resolve first: it would conceal a symlink in the supplied spelling.
    current = path.anchor and Path(path.anchor) or Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink path component is not allowed: {current}")
    try:
        info = path.stat()
    except OSError as error:
        raise ValueError(f"missing path: {path}") from error
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not expected:
        raise ValueError(
            f"path is not a regular {'directory' if directory else 'file'}: {path}"
        )
    return path.resolve()


def _tensor_from_nested(
    value: Any, *, name: str, boolean: bool = False
) -> torch.Tensor:
    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif boolean:
            if type(item) is not bool:
                raise ValueError(f"{name} must contain only booleans")
        elif type(item) not in (int, float) or isinstance(item, bool):
            raise ValueError(f"{name} must contain only finite numeric values")
        else:
            try:
                number = float(item)
            except OverflowError as error:
                raise ValueError(f"{name} contains an overflowing value") from error
            if (
                not math.isfinite(number)
                or abs(number) > torch.finfo(torch.float32).max
            ):
                raise ValueError(f"{name} contains a nonfinite or overflowing value")

    visit(value)
    try:
        result = torch.tensor(value, dtype=torch.bool if boolean else torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be rectangular") from error
    if not boolean and (not torch.isfinite(result).all()):
        raise ValueError(f"{name} contains a nonfinite value")
    return result.contiguous()


def _tokens(value: Any, vocab_size: int, max_seq_len: int) -> torch.Tensor:
    def walk(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
        elif type(item) is not int:
            raise ValueError("input_tokens must contain strict integers")

    walk(value)
    try:
        tokens = torch.tensor(value, dtype=torch.long)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("input_tokens must be a rectangular [B,T] array") from error
    if tokens.ndim != 2 or not all(tokens.shape):
        raise ValueError("input_tokens must be a nonempty rectangular [B,T] array")
    if tokens.numel() > MAX_ADAPTER_TRACE_ITEMS or tokens.shape[1] > max_seq_len:
        raise ValueError("input_tokens exceed semantic probe trace or sequence bounds")
    if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
        raise ValueError("input_tokens contain an out-of-vocabulary id")
    return tokens.contiguous()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Canonical raw little-endian bytes without changing tensor dtype."""
    raw = tensor.detach().to(device="cpu").contiguous()
    if raw.dtype == torch.bfloat16:
        return raw.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    try:
        array = raw.numpy()
    except TypeError:
        return raw.view(torch.uint8).numpy().tobytes()
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array.tobytes()


def _digest_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _state_digest(module: torch.nn.Module) -> str:
    rows = []
    for name, tensor in sorted(module.state_dict().items()):
        raw = tensor.detach().to(device="cpu").contiguous()
        rows.append(
            {
                "name": name,
                "dtype": str(raw.dtype),
                "shape": list(raw.shape),
                "sha256": _digest_tensor(raw),
            }
        )
    return canonical_digest(rows)


def _validate_attachment_shape(attachment: _Attachment, model: DenseLM) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", attachment.name):
        raise ValueError("semantic memory name must be lowercase kebab/snake case")
    if attachment.site == "after_block":
        if attachment.block_index is None or not 0 <= attachment.block_index < len(
            model.blocks
        ):
            raise ValueError("after_block site needs a valid zero-based block_index")
    elif attachment.block_index is not None:
        raise ValueError("block_index is valid only for after_block site")
    if attachment.min_score is not None and not math.isfinite(
        float(attachment.min_score)
    ):
        raise ValueError("min_score must be finite")


def _pack_inventory_digest(retriever: SemanticRetriever) -> str:
    semantic = retriever.pack.manifest.semantic
    if semantic is None:
        raise ValueError("semantic retriever lacks a semantic component")
    payload = {
        "pack_id": retriever.pack_id,
        "semantic": semantic.model_dump(mode="json"),
        "files": [
            entry.model_dump(mode="json") for entry in retriever.pack.manifest.files
        ],
    }
    return canonical_digest(payload)


def _query_identity(query: SemanticQueryBatch, tokens: torch.Tensor) -> dict[str, Any]:
    mask = (
        query.mask
        if query.mask is not None
        else torch.ones(query.vectors.shape[:-1], dtype=torch.bool)
    )
    as_of = query.as_of
    if isinstance(as_of, datetime):
        time_value: Any = as_of.isoformat()
    elif isinstance(as_of, tuple):
        time_value = [
            value.isoformat() if value is not None else None for value in as_of
        ]
    elif as_of is None:
        time_value = None
    else:
        time_value = as_of.isoformat()
    body = {
        "encoder": query.encoder.model_dump(mode="json"),
        "vectors": {
            "dtype": "float32",
            "shape": list(query.vectors.shape),
            "sha256": _digest_tensor(query.vectors),
        },
        "mask": {
            "dtype": "bool",
            "shape": list(mask.shape),
            "sha256": _digest_tensor(mask),
        },
        "tokens": {
            "dtype": "int64",
            "shape": list(tokens.shape),
            "sha256": hashlib.sha256(
                tokens.numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "as_of": time_value,
    }
    body["identity_sha256"] = canonical_digest(body)
    return body


def _trace(trace: Any) -> dict[str, Any]:
    report = asdict(trace)
    report["tied_record_ids"] = list(report["tied_record_ids"])
    return report


def run_semantic_probe(source: Path) -> dict[str, Any]:
    """Authenticate supplied assets then execute one CPU semantic forward."""
    source = _safe_path(Path(source).absolute(), str(Path(source).absolute()))
    declaration_bytes = source.read_bytes()
    if len(declaration_bytes) > _MAX_INPUT_BYTES:
        raise ValueError("semantic probe declaration exceeds 2 MiB")
    declaration_digest = hashlib.sha256(declaration_bytes).hexdigest()
    try:
        declaration = _Declaration.model_validate(read_document(source))
    except ValidationError as error:
        raise _failure(error) from error
    if len(canonical_json(declaration.model_dump(mode="json"))) > _MAX_INPUT_BYTES:
        raise ValueError("aggregate semantic probe input exceeds 2 MiB")

    initialized = isinstance(declaration.model, _InitializedModel)
    initialized_assets: list[tuple[Path, str]] = []
    if initialized:
        model_spec = declaration.model
        assert isinstance(model_spec, _InitializedModel)
        if model_spec.model.memory_package_path is not None:
            package = _safe_path(source, str(model_spec.model.memory_package_path))
            initialized_assets.append((package, sha256_file(package)))
            model_spec = model_spec.model_copy(
                update={
                    "model": model_spec.model.model_copy(
                        update={"memory_package_path": package}
                    )
                }
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_spec.seed)
            model: DenseLM = DenseLM(model_spec.model, model_spec.attention)
        model_identity: dict[str, Any] = {
            "kind": "initialized",
            "model": model_spec.model.model_dump(mode="json"),
            "attention": model_spec.attention.model_dump(mode="json"),
            "seed": model_spec.seed,
            "torch_version": torch.__version__,
            "state_sha256": _state_digest(model),
        }
        if initialized_assets:
            model_identity["assets"] = [
                {"path": str(path), "file_sha256": digest}
                for path, digest in initialized_assets
            ]
    else:
        model_spec = declaration.model
        assert isinstance(model_spec, _CheckpointModel)
        runs_dir = _safe_path(source, model_spec.runs_dir, directory=True)
        loaded = load_run(
            model_spec.run_id,
            runs_dir,
            model_spec.checkpoint,
            backend="cpu",
            verification_mode="cold",
        )
        if loaded.engine is not None or not isinstance(loaded.model, DenseLM):
            raise ValueError("semantic probes require a PyTorch checkpoint model")
        model = loaded.model
        model_identity = {"kind": "checkpoint", "identity": loaded.identity}
    if initialized and any(
        isinstance(item.weights, _CheckpointWeights) for item in declaration.attachments
    ):
        raise ValueError("initialized probes require initialized adapter weights")

    tokens = _tokens(
        declaration.input_tokens, model.config.vocab_size, model.config.max_seq_len
    )
    query_batches: dict[str, SemanticQueryBatch] = {}
    query_reports: dict[str, dict[str, Any]] = {}
    query_by_id: dict[str, SemanticQueryBatch] = {}
    for item in declaration.queries:
        vectors = _tensor_from_nested(item.vectors, name=f"query {item.id}.vectors")
        if vectors.ndim not in (2, 3) or vectors.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"query {item.id} vector batch shape does not match input_tokens"
            )
        if vectors.ndim == 3 and vectors.shape[1] != tokens.shape[1]:
            raise ValueError(
                f"query {item.id} vector sequence shape does not match input_tokens"
            )
        mask = (
            None
            if item.mask is None
            else _tensor_from_nested(
                item.mask, name=f"query {item.id}.mask", boolean=True
            )
        )
        if mask is not None and tuple(mask.shape) != tuple(vectors.shape[:-1]):
            raise ValueError(f"query {item.id} mask shape does not match vectors")
        batch = SemanticQueryBatch(item.encoder, vectors, mask, item.as_of)
        if item.encoder.sha256 in query_batches:
            prior = query_batches[item.encoder.sha256]
            if prior.encoder != item.encoder:
                raise ValueError(
                    "one encoder SHA-256 cannot name different query identities"
                )
            raise ValueError("each encoder SHA-256 must use one query object")
        query_batches[item.encoder.sha256] = batch
        query_by_id[item.id] = batch
        query_reports[item.id] = _query_identity(batch, tokens)

    attachment_reports: list[dict[str, Any]] = []
    pack_bindings: list[tuple[Path, str, str]] = []
    used_queries: set[str] = set()
    declared_checkpoint_names: set[str] = set()
    for attachment in declaration.attachments:
        _validate_attachment_shape(attachment, model)
        if attachment.query not in query_by_id:
            raise ValueError(f"attachment {attachment.name} names an unknown query")
        pack = _safe_path(source, attachment.pack, directory=True)
        retriever = SemanticRetriever.from_pack(
            pack, expected_pack_id=attachment.expected_pack_id
        )
        batch = query_by_id[attachment.query]
        if (
            batch.encoder != retriever.key_encoder
            or batch.vectors.shape[-1] != retriever.key_dim
        ):
            raise ValueError(
                f"attachment {attachment.name} query does not match pack key encoder"
            )
        active = (
            batch.mask
            if batch.mask is not None
            else torch.ones(batch.vectors.shape[:-1], dtype=torch.bool)
        )
        if int(active.sum()) > MAX_ADAPTER_TRACE_ITEMS:
            raise ValueError(
                f"attachment {attachment.name} has too many active positions"
            )
        used_queries.add(attachment.query)
        existing = (
            model.semantic_memories[attachment.name]  # noqa: SIM401 -- ModuleDict does not implement get()
            if attachment.name in model.semantic_memories
            else None
        )
        if isinstance(attachment.weights, _CheckpointWeights):
            if existing is None:
                raise ValueError(
                    f"checkpoint adapter {attachment.name!r} is not restored"
                )
            if (
                existing.retriever.pack_id != retriever.pack_id
                or existing.retriever.key_encoder != retriever.key_encoder
                or existing.site != attachment.site
                or existing.block_index != attachment.block_index
                or existing.min_score != attachment.min_score
            ):
                raise ValueError(
                    f"checkpoint adapter {attachment.name!r} does not match its declaration"
                )
            declared_checkpoint_names.add(attachment.name)
            origin: dict[str, Any] = {"kind": "checkpoint"}
        else:
            if existing is not None:
                raise ValueError(
                    f"initialized adapter {attachment.name!r} collides with a restored adapter"
                )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(attachment.weights.seed)
                adapter = model.add_semantic_memory(
                    attachment.name,
                    retriever,
                    site=attachment.site,
                    block_index=attachment.block_index,
                    min_score=attachment.min_score,
                )
            origin = {
                "kind": "initialized",
                "seed": attachment.weights.seed,
                "state_sha256": _state_digest(adapter),
            }
        inventory_digest = _pack_inventory_digest(retriever)
        pack_bindings.append((pack, attachment.expected_pack_id, inventory_digest))
        attachment_reports.append(
            {
                "name": attachment.name,
                "pack_id": retriever.pack_id,
                "component_inventory_sha256": inventory_digest,
                "pack_manifest_sha256": sha256_file(pack / "manifest.json"),
                "key_encoder": retriever.key_encoder.model_dump(mode="json"),
                "value_encoder": retriever.value_encoder.model_dump(mode="json"),
                "query": attachment.query,
                "site": attachment.site,
                "block_index": attachment.block_index,
                "min_score": attachment.min_score,
                "weights": origin,
            }
        )

    if used_queries != set(query_by_id):
        raise ValueError("every declared query must be used by an attachment")
    restored = set(model.semantic_memories) - {
        item.name
        for item in declaration.attachments
        if isinstance(item.weights, _InitializedWeights)
    }
    if not initialized and restored != declared_checkpoint_names:
        raise ValueError(
            "every restored checkpoint adapter must be explicitly declared"
        )
    if set(query_batches) != {
        adapter.retriever.key_encoder.sha256
        for adapter in model.semantic_memories.values()
    }:
        raise ValueError("query coverage does not match attached semantic encoders")

    # All declaration-dependent validation precedes this single forward.
    model.eval().requires_grad_(False)
    for adapter in model.semantic_memories.values():
        adapter.freeze()
    with torch.inference_mode():
        logits = model(tokens, semantic_queries=query_batches)

    by_name = {item["name"]: item for item in attachment_reports}
    for name, adapter in model.semantic_memories.items():
        report = by_name[name]
        attachment = next(item for item in declaration.attachments if item.name == name)
        batch = query_by_id[attachment.query]
        mask = (
            batch.mask
            if batch.mask is not None
            else torch.ones(batch.vectors.shape[:-1], dtype=torch.bool)
        )
        traces = iter(adapter.last_traces)
        positions = []
        for batch_index in range(tokens.shape[0]):
            for token_index in range(tokens.shape[1]):
                enabled = bool(
                    mask[batch_index]
                    if mask.ndim == 1
                    else mask[batch_index, token_index]
                )
                positions.append(
                    {
                        "batch_index": batch_index,
                        "token_index": token_index,
                        "masked": not enabled,
                        "trace": _trace(next(traces)) if enabled else None,
                    }
                )
        report["traces"] = positions
        report["diagnostics"] = (
            {
                key: value.detach().cpu().tolist()
                for key, value in asdict(adapter.last_diagnostics).items()
            }
            if adapter.last_diagnostics is not None
            else None
        )

    for pack, expected_pack_id, inventory_digest in pack_bindings:
        rechecked = SemanticRetriever.from_pack(pack, expected_pack_id=expected_pack_id)
        if _pack_inventory_digest(rechecked) != inventory_digest:
            raise ValueError("semantic pack changed during probe execution")
    for query_id, batch in query_by_id.items():
        if _query_identity(batch, tokens) != query_reports[query_id]:
            raise ValueError("semantic query input changed during probe execution")
    for asset, digest in initialized_assets:
        _safe_path(source, str(asset))
        if sha256_file(asset) != digest:
            raise ValueError("initialized model asset changed during probe execution")
    _safe_path(source, str(source))
    if hashlib.sha256(source.read_bytes()).hexdigest() != declaration_digest:
        raise ValueError("semantic probe declaration changed during execution")
    combined_query_identity = hashlib.sha256(
        canonical_json(
            [
                {
                    "id": query_id,
                    "identity_sha256": query_reports[query_id]["identity_sha256"],
                }
                for query_id in sorted(query_reports)
            ]
        )
    ).hexdigest()

    return {
        "format": "klab-semantic-probe-report-v1",
        "role": "supplied_vector_mechanism_probe",
        "declaration": {"path": str(source), "file_sha256": declaration_digest},
        "model": model_identity,
        "input_tokens": {
            "dtype": "int64",
            "shape": list(tokens.shape),
            "sha256": hashlib.sha256(
                tokens.numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "queries": query_reports,
        "query_identity_sha256": combined_query_identity,
        "attachments": attachment_reports,
        "logits_shape": list(logits.shape),
    }
