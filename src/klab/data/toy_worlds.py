"""Deterministic synthetic worlds for exact, structured semantic-memory checks.

This module deliberately does *not* encode prompts.  Its vectors encode canonical
``(subject, relation)`` keys, while prompt template names are retained only as a
held-out wording partition.  It is therefore an executable retrieval fixture, not
evidence of natural-language semantic generalization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from safetensors.numpy import save_file

from klab.canonical import canonical_digest, canonical_json
from klab.engram.packs import EncoderIdentity, EngramPackManifest, compile_pack

_CREATED_AT = "2026-09-23T00:00:00Z"
_NAMESPACE = "synthetic-toy-world-v1"
_KEY_DIM = 32
_VALUE_DIM = 8


class MemoryCondition(StrEnum):
    """The attached external-memory condition; no condition trains a model."""

    CORRECT = "correct"
    DISABLED = "disabled"
    RANDOM = "random"
    CONFLICTING = "conflicting"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class StructuredKey:
    """Canonical lookup input.  Textual prompt wording is intentionally absent."""

    subject: str
    relation: str


@dataclass(frozen=True, slots=True)
class FrozenStructuredKeyEncoder:
    """A small versioned, deterministic sparse-sign encoder for synthetic keys."""

    identity: EncoderIdentity
    dimension: int = _KEY_DIM

    def encode(self, key: StructuredKey) -> np.ndarray:
        if not key.subject or not key.relation:
            raise ValueError("structured keys require nonblank subject and relation")
        payload = canonical_json(
            {
                "format": "toy-world-structured-key",
                "relation": key.relation,
                "subject": key.subject,
            }
        )
        digest = hashlib.sha256(payload).digest()
        result = np.zeros(self.dimension, dtype=np.float32)
        # A feature from each disjoint zone makes a true key score exactly four.
        for offset in range(0, 16, 4):
            zone = offset // 4
            width = self.dimension // 4
            index = zone * width + (
                int.from_bytes(digest[offset : offset + 2], "big") % width
            )
            result[index] = 1.0 if digest[offset + 2] & 1 else -1.0
        return result


@dataclass(frozen=True, slots=True)
class FrozenValueEncoder:
    """Declared deterministic value encoder used only to satisfy pack assets."""

    identity: EncoderIdentity
    dimension: int = _VALUE_DIM

    def encode(self, value: str) -> np.ndarray:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return np.asarray(
            [
                int.from_bytes(digest[index : index + 4], "big") / 2**32
                for index in range(0, 32, 4)
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class ProducerFact:
    """A source record used to compile a pack, never a task-case identifier."""

    id: str
    world_id: str
    key: StructuredKey
    value: str
    valid_from: str | None = None
    valid_until: str | None = None


@dataclass(frozen=True, slots=True)
class QueryCase:
    """A query/answer artifact separate from producer records."""

    id: str
    world_id: str
    template_name: str
    keys: tuple[StructuredKey, ...]
    expected_value: str | None
    expected_status: str
    as_of: str | None = None
    changed_assignment: bool = False


@dataclass(frozen=True, slots=True)
class World:
    id: str
    partition: str
    assignment: str
    producer_facts: tuple[ProducerFact, ...]
    query_cases: tuple[QueryCase, ...]


@dataclass(frozen=True, slots=True)
class PackMaterialization:
    condition: MemoryCondition
    path: Path | None
    pack_id: str | None
    record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToyWorldAudit:
    """Leakage-audit identities; wording labels are partitions, never encoder input."""

    benchmark_id: str
    source_identity: str
    train_producer_record_ids: tuple[str, ...]
    heldout_producer_record_ids: tuple[str, ...]
    query_case_ids: tuple[str, ...]
    train_world_ids: tuple[str, ...]
    heldout_world_ids: tuple[str, ...]
    train_templates: tuple[str, ...]
    heldout_templates: tuple[str, ...]
    pack_ids: tuple[tuple[str, MemoryCondition, str | None], ...]


@dataclass(frozen=True, slots=True)
class ToyWorldBenchmark:
    """Immutable benchmark identity and auditable source/query partitions."""

    seed: int
    key_encoder: FrozenStructuredKeyEncoder
    value_encoder: FrozenValueEncoder
    worlds: tuple[World, ...]
    packs: Mapping[str, Mapping[MemoryCondition, PackMaterialization]]
    identity: str

    @property
    def producer_ids(self) -> frozenset[str]:
        return frozenset(
            fact.id for world in self.worlds for fact in world.producer_facts
        )

    @property
    def query_ids(self) -> frozenset[str]:
        return frozenset(case.id for world in self.worlds for case in world.query_cases)

    def pack_for(
        self, case: QueryCase, condition: MemoryCondition
    ) -> PackMaterialization:
        return self.packs[case.world_id][condition]

    def cases(self, partition: str | None = None) -> tuple[QueryCase, ...]:
        return tuple(
            case
            for world in self.worlds
            if partition is None or world.partition == partition
            for case in world.query_cases
        )

    def audit(self) -> ToyWorldAudit:
        train = tuple(world for world in self.worlds if world.partition == "train")
        heldout = tuple(world for world in self.worlds if world.partition == "heldout")
        return ToyWorldAudit(
            benchmark_id=self.identity,
            source_identity="toy-world-generator@frozen-v1",
            train_producer_record_ids=tuple(
                sorted(fact.id for world in train for fact in world.producer_facts)
            ),
            heldout_producer_record_ids=tuple(
                sorted(fact.id for world in heldout for fact in world.producer_facts)
            ),
            query_case_ids=tuple(sorted(self.query_ids)),
            train_world_ids=tuple(world.id for world in train),
            heldout_world_ids=tuple(world.id for world in heldout),
            train_templates=tuple(
                sorted(
                    {
                        case.template_name
                        for world in train
                        for case in world.query_cases
                    }
                )
            ),
            heldout_templates=tuple(
                sorted(
                    {
                        case.template_name
                        for world in heldout
                        for case in world.query_cases
                    }
                )
            ),
            pack_ids=tuple(
                (scope, condition, pack.pack_id)
                for scope, by_condition in self.packs.items()
                for condition, pack in by_condition.items()
            ),
        )


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    condition: MemoryCondition
    status: str
    value: str | None
    followed_attached_pack: bool
    trace_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeSwapRatio:
    value: float | None
    denominator: int
    reason: str | None


def _identity(
    name: str, revision: str, specification: Mapping[str, object]
) -> EncoderIdentity:
    return EncoderIdentity(
        name=name,
        revision=revision,
        sha256=canonical_digest(dict(specification)),
    )


def frozen_encoders() -> tuple[FrozenStructuredKeyEncoder, FrozenValueEncoder]:
    """Return frozen public encoder identities shared by every materialized pack."""
    key = FrozenStructuredKeyEncoder(
        _identity(
            "toy-world-structured-key",
            "frozen-v1",
            {
                "algorithm": "sha256-four-signed-features",
                "dimension": _KEY_DIM,
                "format": 1,
            },
        )
    )
    value = FrozenValueEncoder(
        _identity(
            "toy-world-value",
            "frozen-v1",
            {"algorithm": "sha256-u32-fraction", "dimension": _VALUE_DIM, "format": 1},
        )
    )
    return key, value


def _fact(
    world_id: str,
    suffix: str,
    subject: str,
    relation: str,
    value: str,
    **times: str | None,
) -> ProducerFact:
    return ProducerFact(
        f"producer:{world_id}:{suffix}",
        world_id,
        StructuredKey(subject, relation),
        value,
        **times,
    )


def _case(
    world_id: str,
    suffix: str,
    template: str,
    keys: Sequence[StructuredKey],
    expected: str | None,
    status: str,
    **kwargs: object,
) -> QueryCase:
    return QueryCase(
        f"query:{world_id}:{suffix}",
        world_id,
        template,
        tuple(keys),
        expected,
        status,
        **kwargs,
    )


def _world(seed: int, ordinal: int, partition: str) -> World:
    world_id = f"{partition}-world-{ordinal}"
    # Every world binds the same traversal role to a distinct terminal assignment.
    assignment = f"artifact-{(seed + ordinal * 17) % 997:03d}"
    a = "shared-root"
    b, c = f"{world_id}-b", f"{world_id}-c"
    facts = (
        _fact(world_id, "one", a, "next", b),
        _fact(world_id, "two", b, "next", c),
        _fact(world_id, "three", c, "next", assignment),
        _fact(
            world_id,
            "temporal-current",
            f"{world_id}-season",
            "label",
            "summer",
            valid_from="2030-01-01",
            valid_until="2030-12-31",
        ),
        _fact(
            world_id,
            "temporal-expired",
            f"{world_id}-archive",
            "label",
            "winter",
            valid_until="2029-12-31",
        ),
        _fact(world_id, "conflict-a", f"{world_id}-conflict", "label", "left"),
        _fact(world_id, "conflict-b", f"{world_id}-conflict", "label", "right"),
    )
    template = "train-canonical-v1" if partition == "train" else "heldout-paraphrase-v1"
    keys = (
        StructuredKey(a, "next"),
        StructuredKey(b, "next"),
        StructuredKey(c, "next"),
    )
    cases = (
        _case(
            world_id,
            "one-hop",
            template,
            keys[:1],
            b,
            "retrieved",
            changed_assignment=True,
        ),
        _case(
            world_id,
            "two-hop",
            template,
            keys[:2],
            c,
            "retrieved",
            changed_assignment=True,
        ),
        _case(
            world_id,
            "three-hop",
            template,
            keys,
            assignment,
            "retrieved",
            changed_assignment=True,
        ),
        _case(
            world_id,
            "conflict",
            template,
            (StructuredKey(f"{world_id}-conflict", "label"),),
            None,
            "conflict",
        ),
        _case(
            world_id,
            "temporal",
            template,
            (StructuredKey(f"{world_id}-season", "label"),),
            "summer",
            "retrieved",
            as_of="2030-06-01",
        ),
        _case(
            world_id,
            "temporal-miss",
            template,
            (StructuredKey(f"{world_id}-archive", "label"),),
            None,
            "temporal_miss",
            as_of="2030-06-01",
        ),
        _case(
            world_id,
            "unknown",
            template,
            (StructuredKey(f"{world_id}-missing", "label"),),
            None,
            "unknown",
        ),
    )
    return World(world_id, partition, assignment, facts, cases)


def _validate_worlds(worlds: Sequence[World]) -> None:
    world_ids = [world.id for world in worlds]
    if len(world_ids) != len(set(world_ids)):
        raise ValueError("world IDs must be unique")
    train = {world.id for world in worlds if world.partition == "train"}
    heldout = {world.id for world in worlds if world.partition == "heldout"}
    if not train or not heldout or train & heldout:
        raise ValueError("train and held-out worlds must be nonempty and disjoint")
    templates = {"train": set(), "heldout": set()}
    producer_ids: set[str] = set()
    query_ids: set[str] = set()
    assignments: set[str] = set()
    for world in worlds:
        templates[world.partition].update(
            case.template_name for case in world.query_cases
        )
        assignments.add(world.assignment)
        producer_ids.update(fact.id for fact in world.producer_facts)
        query_ids.update(case.id for case in world.query_cases)
        edges = {
            (fact.key.subject, fact.key.relation, fact.value)
            for fact in world.producer_facts
        }
        # Report a direct start-to-final edge before duplicate-edge failures.
        for case in world.query_cases:
            if len(case.keys) > 1 and case.expected_value is not None:
                start = case.keys[0].subject
                if (start, case.keys[0].relation, case.expected_value) in edges:
                    raise ValueError(f"shortcut edge for {case.id}")
        for case in world.query_cases:
            if len(case.keys) > 1 and case.expected_value is not None:
                expected_subject = case.keys[0].subject
                for hop in case.keys:
                    if hop.subject != expected_subject:
                        raise ValueError(
                            f"case {case.id} does not declare the prior hop value"
                        )
                    matches = [
                        fact
                        for fact in world.producer_facts
                        if fact.key == StructuredKey(expected_subject, hop.relation)
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"case {case.id} does not have one unambiguous edge per hop"
                        )
                    expected_subject = matches[0].value
    if templates["train"] & templates["heldout"]:
        raise ValueError("train and held-out prompt templates must be disjoint")
    if producer_ids & query_ids:
        raise ValueError("producer record IDs and query case IDs must be disjoint")
    if len(assignments) != len(worlds):
        raise ValueError("assignments must change between worlds")


def validate_toy_worlds(worlds: Sequence[World]) -> None:
    """Reject leaking partitions, reused assignments, and multi-hop shortcut edges."""
    _validate_worlds(worlds)


def _record_payload(fact: ProducerFact) -> dict[str, object]:
    return {
        "record_version": 1,
        "id": fact.id,
        "namespace": _NAMESPACE,
        "subject": fact.key.subject,
        "relation": fact.key.relation,
        "value": fact.value,
        "license": "CC0-1.0",
        "source": "toy-world-generator",
        "source_revision": "frozen-v1",
        "created_at": _CREATED_AT,
        "valid_from": fact.valid_from,
        "valid_until": fact.valid_until,
    }


def _write_pack_inputs(
    directory: Path,
    facts: Sequence[ProducerFact],
    key_encoder: FrozenStructuredKeyEncoder,
    value_encoder: FrozenValueEncoder,
    *,
    random_keys: bool,
) -> tuple[Path, Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "records.jsonl"
    source.write_bytes(
        b"".join(canonical_json(_record_payload(fact)) + b"\n" for fact in facts)
    )
    keys = np.stack([key_encoder.encode(fact.key) for fact in facts]).astype(np.float32)
    if random_keys:
        # A deterministic row rotation reassigns keys to different record IDs.
        keys = np.roll(keys, 1, axis=0)
    values = np.stack([value_encoder.encode(fact.value) for fact in facts]).astype(
        np.float32
    )
    keys_path, values_path, metadata_path = (
        directory / "keys.safetensors",
        directory / "values.safetensors",
        directory / "semantic.json",
    )
    save_file({"keys": keys}, keys_path)
    save_file({"values": values}, values_path)
    metadata_path.write_bytes(
        canonical_json(
            {
                "format": "klab-semantic-assets",
                "format_version": 1,
                "record_ids": [fact.id for fact in facts],
                "key_encoder": key_encoder.identity.model_dump(mode="json"),
                "value_encoder": value_encoder.identity.model_dump(mode="json"),
                "key_normalization": "none",
            }
        )
        + b"\n"
    )
    return source, keys_path, values_path, metadata_path


def _compile_condition(
    root: Path,
    condition: MemoryCondition,
    facts: tuple[ProducerFact, ...],
    key_encoder: FrozenStructuredKeyEncoder,
    value_encoder: FrozenValueEncoder,
) -> PackMaterialization:
    if condition is MemoryCondition.DISABLED:
        return PackMaterialization(condition, None, None, ())
    selected = facts
    if condition is MemoryCondition.INCOMPLETE:
        selected = tuple(fact for fact in facts if not fact.id.endswith(":three"))
    if condition is MemoryCondition.CONFLICTING:
        target = next(fact for fact in facts if fact.id.endswith(":one"))
        selected = (
            *facts,
            ProducerFact(
                target.id + ":override",
                target.world_id,
                target.key,
                "incorrect-override",
            ),
        )
    inputs = root / f".{condition.value}-inputs"
    source, keys, values, metadata = _write_pack_inputs(
        inputs,
        selected,
        key_encoder,
        value_encoder,
        random_keys=condition is MemoryCondition.RANDOM,
    )
    pack_path = root / f"{condition.value}.enpack"
    manifest: EngramPackManifest = compile_pack(
        source,
        pack_path,
        name=f"toy-world-{condition.value}",
        namespace=_NAMESPACE,
        default_license="CC0-1.0",
        source_name="toy-world-generator",
        source_revision="frozen-v1",
        created_at=_CREATED_AT,
        semantic_keys=keys,
        semantic_values=values,
        semantic_metadata=metadata,
    )
    return PackMaterialization(
        condition, pack_path, manifest.pack_id, tuple(fact.id for fact in selected)
    )


def materialize_toy_worlds(root: Path, *, seed: int = 0) -> ToyWorldBenchmark:
    """Generate isolated train and held-out attachment packs below ``root``."""
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"toy-world output already exists: {root}")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    root.mkdir(parents=True)
    worlds = (
        _world(seed, 0, "train"),
        _world(seed, 1, "train"),
        _world(seed, 2, "heldout"),
        _world(seed, 3, "heldout"),
    )
    _validate_worlds(worlds)
    key_encoder, value_encoder = frozen_encoders()
    scopes = {world.id: world.producer_facts for world in worlds}
    packs = MappingProxyType(
        {
            scope: MappingProxyType(
                {
                    condition: _compile_condition(
                        root / scope,
                        condition,
                        facts,
                        key_encoder,
                        value_encoder,
                    )
                    for condition in MemoryCondition
                }
            )
            for scope, facts in scopes.items()
        }
    )
    identity = hashlib.sha256(
        canonical_json(
            {
                "seed": seed,
                "worlds": [world.id for world in worlds],
                "packs": {
                    scope: {
                        condition.value: pack.pack_id
                        for condition, pack in by_condition.items()
                    }
                    for scope, by_condition in packs.items()
                },
            }
        )
    ).hexdigest()
    return ToyWorldBenchmark(seed, key_encoder, value_encoder, worlds, packs, identity)


def evaluate_case(
    benchmark: ToyWorldBenchmark,
    case: QueryCase,
    condition: MemoryCondition,
    *,
    attachment_scope: str | None = None,
) -> CaseEvaluation:
    """Evaluate linked structured retrieval through the public semantic runtime API."""
    scope = case.world_id if attachment_scope is None else attachment_scope
    if scope not in benchmark.packs:
        raise ValueError(f"unknown toy-world attachment scope: {scope}")
    pack = benchmark.packs[scope][condition]
    if pack.path is None:
        return CaseEvaluation(case.id, condition, "disabled", None, False, ())
    from klab.engram.semantic import SemanticRetriever

    retriever = SemanticRetriever.from_pack(pack.path, expected_pack_id=pack.pack_id)
    scope_world = next(world for world in benchmark.worlds if world.id == scope)
    values = {
        fact.id: fact.value
        for fact in scope_world.producer_facts
        if fact.id in pack.record_ids
    }
    trace: list[str] = []
    value: str | None = None
    task_status = "unknown"
    next_subject: str | None = None
    for index, declared_key in enumerate(case.keys):
        key = (
            declared_key
            if index == 0
            else StructuredKey(next_subject or "", declared_key.relation)
        )
        outcome = retriever.retrieve(
            torch.from_numpy(benchmark.key_encoder.encode(key)),
            key_encoder=benchmark.key_encoder.identity,
            top_k=1,
            min_score=3.5,
            as_of=case.as_of,
        )
        if outcome.status != "hit":
            task_status = outcome.status
            break
        record_id = outcome.hits[0].record_id
        trace.append(record_id)
        value = values.get(record_id)
        if value is None:
            task_status = "unknown"
            break
        next_subject = value
        task_status = "retrieved"
    task_kind = case.id.rsplit(":", 1)[-1]
    attached_expected = next(
        candidate.expected_value
        for candidate in scope_world.query_cases
        if candidate.id.rsplit(":", 1)[-1] == task_kind
    )
    follows = task_status == "retrieved" and value == attached_expected
    return CaseEvaluation(case.id, condition, task_status, value, follows, tuple(trace))


def knowledge_swap_ratio(
    cases: Sequence[QueryCase], evaluations: Mapping[str, CaseEvaluation]
) -> KnowledgeSwapRatio:
    """Return one task family's attached-pack following rate after a knowledge swap."""
    eligible = [case for case in cases if case.changed_assignment]
    if not eligible:
        return KnowledgeSwapRatio(None, 0, "no changed-assignment query cases")
    missing = [case.id for case in eligible if case.id not in evaluations]
    if missing:
        raise ValueError("missing evaluations for changed-assignment cases")
    task_kinds = {case.id.rsplit(":", 1)[-1] for case in eligible}
    if len(task_kinds) != 1:
        raise ValueError("knowledge-swap ratio requires one task kind")
    conditions = {evaluations[case.id].condition for case in eligible}
    if len(conditions) != 1:
        raise ValueError("knowledge-swap ratio requires one attachment condition")
    followed = sum(evaluations[case.id].followed_attached_pack for case in eligible)
    return KnowledgeSwapRatio(followed / len(eligible), len(eligible), None)


def materialize_portability_worlds(
    root: Path, *, seed: int = 20260925, scale: str = "micro"
) -> Path:
    """Materialize the immutable, leakage-audited Engram portability inputs."""
    from klab.data.portability_worlds import (
        materialize_portability_worlds as build,
    )

    return build(root, seed=seed, scale=scale)
