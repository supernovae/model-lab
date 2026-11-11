"""Verified exact retrieval over semantic Engram pack assets."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from safetensors.torch import load as load_safetensors
from torch import Tensor, nn

if TYPE_CHECKING:
    from klab.engram.packs import EncoderIdentity, EngramPack

from klab.engram.records import MAX_RECORD_BYTES, KnowledgeRecord

MAX_SEMANTIC_ENTRIES = 65_536
MAX_SEMANTIC_ASSET_BYTES = 256 * 1024 * 1024
MAX_SEMANTIC_COMPARISONS = 4_194_304
MAX_SEMANTIC_RESULT_BYTES = 64 * 1024 * 1024
MAX_ADAPTER_TRACE_ITEMS = 64
MAX_TOP_K = 16
_VERIFIED_RETRIEVER_TOKEN = object()


def _coerce_encoder_identity(value: object) -> EncoderIdentity:
    from klab.engram.packs import EncoderIdentity

    if isinstance(value, EncoderIdentity):
        return value
    return EncoderIdentity.model_validate(value)


@dataclass(frozen=True)
class SemanticQueryBatch:
    """Already-encoded query vectors; this type does not encode text."""

    encoder: EncoderIdentity
    vectors: Tensor
    mask: Tensor | None = None
    as_of: str | date | datetime | Sequence[str | date | datetime | None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder", _coerce_encoder_identity(self.encoder))
        if not isinstance(self.vectors, Tensor):
            raise TypeError("semantic query vectors must be a torch.Tensor")
        if self.vectors.ndim not in (2, 3):
            raise ValueError(
                "semantic query vectors must have shape [batch,key_dim] or [batch,sequence,key_dim]"
            )
        if not self.vectors.is_floating_point():
            raise TypeError("semantic query vectors must use a floating-point dtype")
        if self.mask is not None:
            if not isinstance(self.mask, Tensor) or self.mask.dtype != torch.bool:
                raise TypeError("semantic query mask must be a boolean tensor")
            expected = self.vectors.shape[:-1]
            if self.mask.shape != expected:
                raise ValueError(
                    "semantic query mask must match query batch and sequence dimensions"
                )
        if isinstance(self.as_of, Sequence) and not isinstance(
            self.as_of, (str, bytes)
        ):
            expected_count = (
                self.vectors.shape[0]
                if self.vectors.ndim == 2
                else self.vectors.shape[0] * self.vectors.shape[1]
            )
            if len(self.as_of) != expected_count:
                raise ValueError("semantic as_of sequence must match query positions")
            normalized_as_of: datetime | tuple[datetime | None, ...] | None = tuple(
                _time_instant(value) for value in self.as_of
            )
        else:
            normalized_as_of = _time_instant(self.as_of)
        object.__setattr__(self, "as_of", normalized_as_of)


@dataclass(frozen=True)
class SemanticRetrievalHit:
    record_id: str
    score: float
    value_vector: Tensor


@dataclass(frozen=True)
class SemanticRetrievalTrace:
    pack_id: str
    key_encoder_sha256: str
    metric: Literal["dot", "cosine"]
    comparison_count: int
    candidate_count: int
    temporal_excluded_count: int
    best_record_id: str | None
    best_score: float | None
    tie_count: int
    tied_record_ids: tuple[str, ...]
    status: Literal["hit", "unknown", "conflict", "temporal_miss"]
    min_score: float | None
    as_of: str | None


@dataclass(frozen=True)
class SemanticRetrievalOutcome:
    status: Literal["hit", "unknown", "conflict", "temporal_miss"]
    hits: tuple[SemanticRetrievalHit, ...]
    trace: SemanticRetrievalTrace


@dataclass(frozen=True, init=False)
class SemanticRetriever:
    """Immutable, verified semantic assets with exact bounded top-k lookup."""

    pack: EngramPack
    _keys: Tensor
    _values: Tensor
    _record_ids: tuple[str, ...]
    _records: dict[str, KnowledgeRecord]
    max_comparisons: int

    def __init__(
        self,
        pack: EngramPack,
        keys: Tensor,
        values: Tensor,
        record_ids: tuple[str, ...],
        records: dict[str, KnowledgeRecord],
        max_comparisons: int,
        *,
        _verification_token: object,
    ) -> None:
        if _verification_token is not _VERIFIED_RETRIEVER_TOKEN:
            raise TypeError("SemanticRetriever instances must be loaded with from_pack")
        object.__setattr__(self, "pack", pack)
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_record_ids", record_ids)
        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "max_comparisons", max_comparisons)

    @classmethod
    def from_pack(
        cls,
        path: Path,
        *,
        expected_pack_id: str | None = None,
        max_entries: int = MAX_SEMANTIC_ENTRIES,
        max_asset_bytes: int = MAX_SEMANTIC_ASSET_BYTES,
        max_comparisons: int = MAX_SEMANTIC_COMPARISONS,
    ) -> SemanticRetriever:
        """Verify the complete pack before loading a size-bounded semantic group."""
        from klab.engram.packs import load_pack

        for name, value in (
            ("max_entries", max_entries),
            ("max_asset_bytes", max_asset_bytes),
            ("max_comparisons", max_comparisons),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        pack = load_pack(path, expected_pack_id=expected_pack_id)
        component = pack.manifest.semantic
        metadata = pack.semantic_metadata
        if component is None or metadata is None:
            raise ValueError("Engram pack has no semantic component")
        if component.entry_count > max_entries:
            raise ValueError(
                f"semantic pack has {component.entry_count} entries; limit is {max_entries}"
            )
        expected_bytes = (
            4 * component.entry_count * (component.key_dim + component.memory_dim)
        )
        if expected_bytes > max_asset_bytes:
            raise ValueError(
                f"semantic tensor assets need {expected_bytes} bytes; limit is {max_asset_bytes}"
            )
        if component.entry_count > max_comparisons:
            raise ValueError(
                "semantic pack entry count exceeds the exact comparison bound"
            )
        identities = {item.relative_path: item for item in pack.manifest.files}
        tensor_asset_bytes = sum(
            identities[name].size_bytes
            for name in (component.keys_path, component.values_path)
        )
        if tensor_asset_bytes > max_asset_bytes:
            raise ValueError(
                f"semantic tensor files need {tensor_asset_bytes} bytes; limit is {max_asset_bytes}"
            )

        def read_verified_asset(name: str) -> bytes:
            identity = identities[name]
            data = (pack.root / name).read_bytes()
            if (
                len(data) != identity.size_bytes
                or hashlib.sha256(data).hexdigest() != identity.sha256
            ):
                raise ValueError(
                    f"semantic tensor asset changed after pack verification: {name}"
                )
            return data

        keys_payload = load_safetensors(read_verified_asset(component.keys_path))
        values_payload = load_safetensors(read_verified_asset(component.values_path))
        if set(keys_payload) != {"keys"} or set(values_payload) != {"values"}:
            raise ValueError("semantic safetensors files contain unexpected tensors")
        keys_f32 = keys_payload["keys"]
        values = values_payload["values"].contiguous()
        if (
            keys_f32.shape != (component.entry_count, component.key_dim)
            or values.shape != (component.entry_count, component.memory_dim)
            or keys_f32.dtype != torch.float32
            or values.dtype != torch.float32
            or not torch.isfinite(keys_f32).all()
            or not torch.isfinite(values).all()
        ):
            raise ValueError("semantic tensors conflict with the verified descriptor")
        # Score in FP64 on CPU so ties do not depend on accelerator reductions.
        # The pack's FP32 values remain immutable and outside state_dict.
        keys = keys_f32.to(torch.float64)
        if component.key_normalization == "l2":
            norms = torch.linalg.vector_norm(keys, dim=1, keepdim=True)
            if not torch.isfinite(norms).all() or torch.any(norms == 0):
                raise ValueError("semantic key vectors must have finite nonzero norms")
            keys.div_(norms)
        del keys_f32, keys_payload, values_payload

        ordered_indices = sorted(
            range(len(metadata.record_ids)), key=metadata.record_ids.__getitem__
        )
        order = torch.tensor(ordered_indices, dtype=torch.long)
        keys = keys.index_select(0, order).contiguous()
        values = values.index_select(0, order).contiguous()
        record_ids = tuple(metadata.record_ids[index] for index in ordered_indices)
        semantic_id_set = set(record_ids)
        records: dict[str, KnowledgeRecord] = {}
        record_identity = identities["records.jsonl"]
        digest = hashlib.sha256()
        with (pack.root / "records.jsonl").open("rb") as handle:
            while line := handle.readline(MAX_RECORD_BYTES + 2):
                if len(line) > MAX_RECORD_BYTES + 1 or not line.endswith(b"\n"):
                    raise ValueError(
                        "verified records.jsonl changed after pack verification"
                    )
                digest.update(line)
                record = KnowledgeRecord.model_validate_json(line)
                if record.id in semantic_id_set:
                    records[record.id] = record
        if digest.hexdigest() != record_identity.sha256:
            raise ValueError("records.jsonl changed after pack verification")
        if set(records) != semantic_id_set:
            raise ValueError("semantic metadata does not resolve to verified records")
        return cls(
            pack,
            keys,
            values,
            record_ids,
            records,
            max_comparisons,
            _verification_token=_VERIFIED_RETRIEVER_TOKEN,
        )

    @property
    def pack_id(self) -> str:
        return self.pack.manifest.pack_id

    @property
    def key_encoder(self) -> EncoderIdentity:
        component = self.pack.manifest.semantic
        assert component is not None
        return component.key_encoder

    @property
    def value_encoder(self) -> EncoderIdentity:
        component = self.pack.manifest.semantic
        assert component is not None
        return component.value_encoder

    @property
    def key_dim(self) -> int:
        component = self.pack.manifest.semantic
        assert component is not None
        return component.key_dim

    @property
    def memory_dim(self) -> int:
        component = self.pack.manifest.semantic
        assert component is not None
        return component.memory_dim

    @property
    def key_normalization(self) -> Literal["none", "l2"]:
        component = self.pack.manifest.semantic
        assert component is not None
        return component.key_normalization

    @property
    def space_id(self) -> str:
        component = self.pack.manifest.semantic
        assert component is not None
        return component.space_id

    @property
    def entry_count(self) -> int:
        return len(self._record_ids)

    def retrieve(
        self,
        query: Tensor,
        *,
        key_encoder: EncoderIdentity,
        top_k: int = 1,
        min_score: float | None = None,
        as_of: str | date | datetime | None = None,
    ) -> SemanticRetrievalOutcome:
        """Return exact neighbors using the pack's declared dot/cosine contract."""
        if not isinstance(query, Tensor) or query.ndim != 1:
            raise ValueError("one semantic query must have shape [key_dim]")
        outcomes = self.retrieve_many(
            query.reshape(1, -1),
            key_encoder=key_encoder,
            top_k=top_k,
            min_score=min_score,
            as_of=as_of,
        )
        return outcomes[0]

    def retrieve_many(
        self,
        queries: Tensor,
        *,
        key_encoder: EncoderIdentity,
        top_k: int = 1,
        min_score: float | None = None,
        as_of: str
        | date
        | datetime
        | Sequence[str | date | datetime | None]
        | None = None,
    ) -> tuple[SemanticRetrievalOutcome, ...]:
        """Search a finite query matrix under hard comparison and result bounds."""
        key_encoder = _coerce_encoder_identity(key_encoder)
        if key_encoder != self.key_encoder:
            raise ValueError("semantic query encoder identity does not match the pack")
        if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be an integer in [1, {MAX_TOP_K}]")
        if min_score is not None and (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(float(min_score))
        ):
            raise ValueError("min_score must be a finite number or None")
        if not isinstance(queries, Tensor) or queries.ndim != 2:
            raise ValueError("semantic queries must have shape [query_count,key_dim]")
        if queries.shape[1] != self.key_dim:
            raise ValueError(
                "semantic query width does not match the pack key dimension"
            )
        if not queries.is_floating_point():
            raise TypeError("semantic queries must use a floating-point dtype")
        if not torch.isfinite(queries).all():
            raise ValueError("semantic queries contain nonfinite values")
        query_count = queries.shape[0]
        comparisons = query_count * self.entry_count
        if comparisons > self.max_comparisons:
            raise ValueError(
                f"exact retrieval needs {comparisons} comparisons; "
                f"limit is {self.max_comparisons}"
            )
        result_bytes = query_count * min(top_k, self.entry_count) * self.memory_dim * 4
        if result_bytes > MAX_SEMANTIC_RESULT_BYTES:
            raise ValueError(
                f"semantic results need up to {result_bytes} bytes; "
                f"limit is {MAX_SEMANTIC_RESULT_BYTES}"
            )
        if isinstance(as_of, Sequence) and not isinstance(as_of, (str, bytes)):
            if len(as_of) != query_count:
                raise ValueError("as_of sequence must match the query count")
            instants = tuple(_time_instant(value) for value in as_of)
        else:
            instant = _time_instant(as_of)
            instants = (instant,) * query_count
        query_values = (
            queries.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        if self.key_normalization == "l2":
            norms = torch.linalg.vector_norm(query_values, dim=1, keepdim=True)
            if not torch.isfinite(norms).all() or torch.any(norms == 0):
                raise ValueError(
                    "cosine semantic queries must have finite nonzero norms"
                )
            query_values.div_(norms)
        scores = query_values @ self._keys.transpose(0, 1)
        metric: Literal["dot", "cosine"] = (
            "cosine" if self.key_normalization == "l2" else "dot"
        )
        results: list[SemanticRetrievalOutcome] = []
        for row, instant in enumerate(instants):
            eligible_mask = torch.tensor(
                [
                    instant is None or _is_valid_at(self._records[record_id], instant)
                    for record_id in self._record_ids
                ],
                dtype=torch.bool,
            )
            candidate_count = int(eligible_mask.sum())
            temporal_excluded = self.entry_count - candidate_count
            all_scores = scores[row]
            all_best_index = int(all_scores.argmax())
            all_best_score = float(all_scores[all_best_index])
            if candidate_count == 0:
                threshold = -math.inf if min_score is None else float(min_score)
                status: Literal["unknown", "temporal_miss"] = (
                    "temporal_miss" if all_best_score >= threshold else "unknown"
                )
                tie_count = int((all_scores == all_best_score).sum())
                tied_indices = (
                    torch.nonzero(all_scores == all_best_score, as_tuple=False)
                    .flatten()[:MAX_TOP_K]
                    .tolist()
                )
                results.append(
                    self._outcome(
                        status=status,
                        hits=(),
                        best_id=self._record_ids[all_best_index],
                        best_score=all_best_score,
                        tie_count=tie_count,
                        ties=tuple(self._record_ids[index] for index in tied_indices),
                        comparison_count=self.entry_count,
                        candidate_count=0,
                        temporal_excluded=temporal_excluded,
                        metric=metric,
                        min_score=min_score,
                        as_of=instant,
                    )
                )
                continue
            row_scores = all_scores.masked_fill(~eligible_mask, -torch.inf)
            ranked = torch.argsort(row_scores, descending=True, stable=True)
            best_index = int(ranked[0])
            best_score = float(row_scores[best_index])
            tie_count = int((row_scores == best_score).sum())
            tied_indices = ranked[: min(tie_count, MAX_TOP_K)].tolist()
            tied_ids = tuple(self._record_ids[index] for index in tied_indices)
            if min_score is not None and best_score < float(min_score):
                excluded_best_score = (
                    float(all_scores[~eligible_mask].max())
                    if temporal_excluded
                    else -math.inf
                )
                temporal_miss = (
                    instant is not None
                    and temporal_excluded > 0
                    and excluded_best_score >= float(min_score)
                )
                if temporal_miss:
                    trace_best_id = self._record_ids[all_best_index]
                    trace_best_score = all_best_score
                    trace_tie_count = int((all_scores == all_best_score).sum())
                    trace_tied_indices = (
                        torch.nonzero(all_scores == all_best_score, as_tuple=False)
                        .flatten()[:MAX_TOP_K]
                        .tolist()
                    )
                    trace_tied_ids = tuple(
                        self._record_ids[index] for index in trace_tied_indices
                    )
                else:
                    trace_best_id = self._record_ids[best_index]
                    trace_best_score = best_score
                    trace_tie_count = tie_count
                    trace_tied_ids = tied_ids
                results.append(
                    self._outcome(
                        status="temporal_miss" if temporal_miss else "unknown",
                        hits=(),
                        best_id=trace_best_id,
                        best_score=trace_best_score,
                        tie_count=trace_tie_count,
                        ties=trace_tied_ids,
                        comparison_count=self.entry_count,
                        candidate_count=candidate_count,
                        temporal_excluded=temporal_excluded,
                        metric=metric,
                        min_score=min_score,
                        as_of=instant,
                    )
                )
                continue
            selected_indices = ranked[: min(top_k, candidate_count)].tolist()
            hits = tuple(
                SemanticRetrievalHit(
                    self._record_ids[index],
                    float(row_scores[index]),
                    self._values[index].detach().clone(),
                )
                for index in selected_indices
            )
            status: Literal["hit", "conflict"] = "conflict" if tie_count > 1 else "hit"
            results.append(
                self._outcome(
                    status=status,
                    hits=hits,
                    best_id=self._record_ids[best_index],
                    best_score=best_score,
                    tie_count=tie_count,
                    ties=tied_ids,
                    comparison_count=self.entry_count,
                    candidate_count=candidate_count,
                    temporal_excluded=temporal_excluded,
                    metric=metric,
                    min_score=min_score,
                    as_of=instant,
                )
            )
        return tuple(results)

    def _outcome(
        self,
        *,
        status: Literal["hit", "unknown", "conflict", "temporal_miss"],
        hits: tuple[SemanticRetrievalHit, ...],
        best_id: str | None,
        best_score: float | None,
        tie_count: int,
        ties: tuple[str, ...],
        comparison_count: int,
        candidate_count: int,
        temporal_excluded: int,
        metric: Literal["dot", "cosine"],
        min_score: float | None,
        as_of: datetime | None,
    ) -> SemanticRetrievalOutcome:
        trace = SemanticRetrievalTrace(
            pack_id=self.pack_id,
            key_encoder_sha256=self.key_encoder.sha256,
            metric=metric,
            comparison_count=comparison_count,
            candidate_count=candidate_count,
            temporal_excluded_count=temporal_excluded,
            best_record_id=best_id,
            best_score=best_score,
            tie_count=tie_count,
            tied_record_ids=ties,
            status=status,
            min_score=None if min_score is None else float(min_score),
            as_of=None if as_of is None else as_of.isoformat(),
        )
        return SemanticRetrievalOutcome(status, hits, trace)


def _time_instant(value: str | date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time(), UTC)
    elif isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            parsed = datetime.combine(
                date.fromisoformat(value), datetime.min.time(), UTC
            )
        else:
            parsed = datetime.fromisoformat(
                value[:-1] + "+00:00" if value.endswith("Z") else value
            )
    else:
        raise TypeError("as_of must be an ISO-8601 string, date, datetime, or None")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("as_of timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _record_instant(value: str | None) -> datetime | None:
    return _time_instant(value)


def _is_valid_at(record: KnowledgeRecord, instant: datetime) -> bool:
    start = _record_instant(record.valid_from)
    end = _record_instant(record.valid_until)
    return (start is None or instant >= start) and (end is None or instant <= end)


@dataclass(frozen=True)
class SemanticMemoryDiagnostics:
    lookup_count: Tensor
    hit_count: Tensor
    unknown_count: Tensor
    conflict_count: Tensor
    temporal_miss_count: Tensor
    mean_score: Tensor
    gate_mean: Tensor
    value_norm: Tensor
    hidden_norm: Tensor


class SemanticMemoryAdapter(nn.Module):
    """Frozen retrieval assets plus a trainable projection and hidden-state gate."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        hidden_dim: int,
        *,
        min_score: float | None = None,
        site: Literal["embedding", "after_block", "final"] = "final",
        block_index: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(retriever, SemanticRetriever):
            raise TypeError("semantic memory requires a verified SemanticRetriever")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if site not in ("embedding", "after_block", "final"):
            raise ValueError(f"unsupported semantic injection site: {site!r}")
        if site == "after_block":
            if type(block_index) is not int or block_index < 0:
                raise ValueError(
                    "after_block semantic memory requires a nonnegative block_index"
                )
        elif block_index is not None:
            raise ValueError(
                "block_index is valid only for after_block semantic memory"
            )
        if min_score is not None and (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(float(min_score))
        ):
            raise ValueError("min_score must be a finite number or None")
        self.retriever = retriever
        self.hidden_dim = hidden_dim
        self.min_score = None if min_score is None else float(min_score)
        self.site = site
        self.block_index = block_index
        self.output = nn.Linear(retriever.memory_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: SemanticMemoryDiagnostics | None = None
        self.last_traces: tuple[SemanticRetrievalTrace, ...] = ()

    def replace_retriever(self, retriever: SemanticRetriever) -> None:
        """Attach a new pack without changing trainable adapter parameters."""
        if not isinstance(retriever, SemanticRetriever):
            raise TypeError("replacement must be a verified SemanticRetriever")
        prior = self.retriever
        if (
            retriever.key_encoder != prior.key_encoder
            or retriever.value_encoder != prior.value_encoder
            or retriever.key_dim != prior.key_dim
            or retriever.memory_dim != prior.memory_dim
            or retriever.space_id != prior.space_id
        ):
            raise ValueError(
                "replacement pack changes the frozen semantic feature contract"
            )
        self.retriever = retriever
        self.last_diagnostics = None
        self.last_traces = ()

    def freeze(self) -> None:
        """Freeze only this adapter's learned projection and gate."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: Tensor, queries: SemanticQueryBatch) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                "semantic hidden states must have shape [batch,sequence,hidden_dim]"
            )
        if queries.encoder != self.retriever.key_encoder:
            raise ValueError(
                "semantic query encoder identity does not match the attached pack"
            )
        vectors = queries.vectors
        batch, sequence, _ = hidden.shape
        if vectors.ndim == 2:
            if vectors.shape != (batch, self.retriever.key_dim):
                raise ValueError("semantic queries must have shape [batch,key_dim]")
            vectors = vectors[:, None, :].expand(
                batch, sequence, self.retriever.key_dim
            )
        elif vectors.shape != (batch, sequence, self.retriever.key_dim):
            raise ValueError("semantic queries must match [batch,sequence,key_dim]")
        if queries.mask is None:
            mask = torch.ones(
                (batch, sequence), dtype=torch.bool, device=vectors.device
            )
        elif queries.mask.ndim == 1:
            mask = queries.mask[:, None].expand(batch, sequence)
        else:
            mask = queries.mask
        mask = mask.to(device=hidden.device)
        flat_vectors = vectors.reshape(batch * sequence, self.retriever.key_dim)
        active = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        values_cpu = torch.zeros(
            (batch * sequence, self.retriever.memory_dim), dtype=torch.float32
        )
        outcomes: tuple[SemanticRetrievalOutcome, ...] = ()
        if active.numel():
            selected_queries = flat_vectors.index_select(
                0, active.to(flat_vectors.device)
            )
            if isinstance(queries.as_of, tuple):
                active_indices = active.to(device="cpu").tolist()
                if queries.vectors.ndim == 2:
                    query_as_of = tuple(
                        queries.as_of[index // sequence] for index in active_indices
                    )
                else:
                    query_as_of = tuple(
                        queries.as_of[index] for index in active_indices
                    )
            else:
                query_as_of = queries.as_of
            outcomes = self.retriever.retrieve_many(
                selected_queries,
                key_encoder=queries.encoder,
                top_k=1,
                min_score=self.min_score,
                as_of=query_as_of,
            )
            active_indices = active.to(device="cpu").tolist()
            for flat_index, outcome in zip(active_indices, outcomes, strict=True):
                if outcome.hits:
                    values_cpu[flat_index] = outcome.hits[0].value_vector
        values = values_cpu.to(device=hidden.device, dtype=hidden.dtype).reshape(
            batch, sequence, self.retriever.memory_dim
        )
        projected = self.output(values)
        gate = torch.sigmoid(self.gate(hidden))
        self.last_diagnostics = _semantic_diagnostics(hidden, values, gate, outcomes)
        self.last_traces = tuple(
            outcome.trace for outcome in outcomes[-MAX_ADAPTER_TRACE_ITEMS:]
        )
        return hidden + gate * projected


def _semantic_diagnostics(
    hidden: Tensor,
    values: Tensor,
    gate: Tensor,
    outcomes: tuple[SemanticRetrievalOutcome, ...],
) -> SemanticMemoryDiagnostics:
    device = hidden.device
    count = len(outcomes)
    statuses = [outcome.status for outcome in outcomes]
    scores = [
        outcome.trace.best_score
        for outcome in outcomes
        if outcome.trace.best_score is not None
    ]
    return SemanticMemoryDiagnostics(
        lookup_count=torch.tensor(count, device=device, dtype=torch.float32).detach(),
        hit_count=torch.tensor(
            statuses.count("hit"), device=device, dtype=torch.float32
        ).detach(),
        unknown_count=torch.tensor(
            statuses.count("unknown"), device=device, dtype=torch.float32
        ).detach(),
        conflict_count=torch.tensor(
            statuses.count("conflict"), device=device, dtype=torch.float32
        ).detach(),
        temporal_miss_count=torch.tensor(
            statuses.count("temporal_miss"), device=device, dtype=torch.float32
        ).detach(),
        mean_score=torch.tensor(
            sum(scores) / len(scores) if scores else 0.0, device=device
        ).detach(),
        gate_mean=gate.mean().detach(),
        value_norm=values.norm(dim=-1).mean().detach(),
        hidden_norm=hidden.norm(dim=-1).mean().detach(),
    )
