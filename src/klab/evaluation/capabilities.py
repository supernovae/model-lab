"""Versioned, bounded capability cards and evidence-bound comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from klab.data.chat_recall import cases as chat_recall_cases
from klab.data.chat_recall import context_override_cases
from klab.data.engram_recall import recall_case
from klab.engines.mlx import MLXEngine
from klab.engram.packs import EncoderIdentity
from klab.engram.semantic import SemanticQueryBatch
from klab.evaluation.chat import assistant_reply
from klab.evaluation.generation import generate
from klab.experiments.compiler import (
    compare_configs,
    flatten_config,
    is_dotted_path,
)
from klab.result_publication import write_bytes_exclusive
from klab.training.manifest import source_identity

_CARD_FORMAT = "capability_card_v2"
_FAMILY_CARD_FORMAT = "capability_card_v3"
_RESULT_FORMAT = "capability_result_v2"
_EXACT_SCORER = "normalized_full_answer_exact_v1"
_LITERAL_SCORER = "literal_full_answer_exact_v1"
_LEGACY_SCORER = "first_normalized_word_exact_v1"
_GENERATION = {
    "max_new_tokens": 8,
    "temperature": 0.0,
    "top_k": 0,
    "seed": 0,
    "stop_sequences": ("\nUser:", "\nSystem:", "\nAssistant:"),
}
_CASE_KINDS = frozenset(
    {
        "recall",
        "alias_recall",
        "context_override",
        "custom",
        "math",
        "api",
        "factual_recall",
        "paraphrase",
        "application",
        "composition",
        "long_context",
        "instruction",
        "conversation",
        "stale_source",
        "conflicting_source",
        "missing_source",
    }
)
_MEMORY_FIELDS = frozenset(
    {
        "memory",
        "memory_table_size",
        "memory_ngram_size",
        "memory_dim",
        "memory_package_path",
        "memory_ngram_orders",
        "memory_hash_heads",
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
    }
)
_FFN_FIELDS = frozenset(
    {
        "ffn",
        "num_experts",
        "experts_per_token",
        "shared_expert",
        "router_aux_loss_coefficient",
    }
)
_SCALE_FIELDS = frozenset(
    {"hidden_dim", "num_layers", "num_heads", "ffn_dim", "max_seq_len"}
)


@dataclass(frozen=True)
class CapabilitySemanticQuery:
    encoder: EncoderIdentity
    vector: tuple[float, ...]


@dataclass(frozen=True)
class CapabilityCaseLineage:
    split: str
    source_document_family: str | None
    world_id: str | None
    template_family: str
    parent_content_hashes: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityCase:
    identifier: str
    prompt: str
    expected: str
    kind: str = "recall"
    semantic_query: CapabilitySemanticQuery | None = None
    lineage: CapabilityCaseLineage | None = None


@dataclass(frozen=True)
class CapabilityCard:
    name: str
    version: int
    hypothesis: str
    scorer: str
    cases: tuple[CapabilityCase, ...]
    generation: Mapping[str, Any]
    scoring: Mapping[str, str]
    controls: Mapping[str, float]
    limitations: str

    format: str = _CARD_FORMAT

    @property
    def digest(self) -> str:
        return _digest(_card_payload(self))


def _case_payload(case: CapabilityCase) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "identifier": case.identifier,
        "prompt": case.prompt,
        "expected": case.expected,
        "kind": case.kind,
    }
    if case.semantic_query is not None:
        payload["semantic_query"] = {
            "encoder": case.semantic_query.encoder.model_dump(mode="json"),
            "vector": list(case.semantic_query.vector),
        }
    if case.lineage is not None:
        payload["lineage"] = {
            "split": case.lineage.split,
            "source_document_family": case.lineage.source_document_family,
            "world_id": case.lineage.world_id,
            "template_family": case.lineage.template_family,
            "parent_content_hashes": list(case.lineage.parent_content_hashes),
        }
    return payload


def _card_payload(card: CapabilityCard) -> dict[str, Any]:
    generation = dict(card.generation)
    generation["stop_sequences"] = list(generation["stop_sequences"])
    return {
        "name": card.name,
        "version": card.version,
        "hypothesis": card.hypothesis,
        "scorer": card.scorer,
        "cases": [_case_payload(case) for case in card.cases],
        "generation": generation,
        "scoring": dict(card.scoring),
        "controls": dict(card.controls),
        "limitations": card.limitations,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _first_word(value: str) -> str:
    words = _normalize_answer(value).split()
    return words[0] if words else ""


def _legacy_card() -> CapabilityCard:
    return CapabilityCard(
        "engram-recall-v1",
        1,
        "Diagnostic only: fixed association completion under matched controls.",
        _LEGACY_SCORER,
        tuple(
            CapabilityCase(f"recall-{index:02d}", *recall_case(42, 2_000_000 + index))
            for index in range(16)
        ),
        _GENERATION,
        {"normalization": "first_word_casefold", "match": "first_word"},
        {"uniform_chance": 1 / 16, "majority_answer": 1 / 16},
        "The answer appears in every prompt and only its first word is scored; this is not held-out knowledge evidence.",
    )


def _chat_card(name: str) -> CapabilityCard:
    if name == "chat-alias-retention-v1":
        source_cases = chat_recall_cases("train")[::3]
        hypothesis = "A trained model can retain the fixed alias associations in its training query format."
        limitations = "Training-seen prompts: an acquisition sanity check, not held-out generalization or an Engram superiority claim."
    elif name == "chat-alias-recall-v1":
        # Validation is used for held-out loss/checkpoint selection.  Card evidence
        # therefore uses only the independently held-out test prompts.
        source_cases = chat_recall_cases("test")
        hypothesis = "A model recalls trained alias-to-value associations under held-out chat wording."
        limitations = "Mappings are learned during training; this tests held-out wording, not novel-world knowledge."
    elif name == "chat-context-override-v1":
        source_cases = context_override_cases()
        hypothesis = "A model follows an explicit conversation-local mapping instead of a static learned lookup."
        limitations = "This is a control for context following, not evidence that static memory provides novel knowledge."
    else:
        raise ValueError(f"unknown capability card: {name}")
    answer_counts = {
        case.expected: sum(other.expected == case.expected for other in source_cases)
        for case in source_cases
    }
    return CapabilityCard(
        name,
        1,
        hypothesis,
        _EXACT_SCORER,
        tuple(
            CapabilityCase(case.identifier, case.prompt, case.expected, case.kind)
            for case in source_cases
        ),
        _GENERATION,
        {
            "normalization": "unicode_nfc_casefold_trim_collapse_whitespace",
            "match": "full_answer",
        },
        {
            "uniform_chance": 1 / len(answer_counts),
            "majority_answer": max(answer_counts.values()) / len(source_cases),
        },
        limitations,
    )


def _task_card(
    name: str,
    hypothesis: str,
    limitations: str,
    cases: Any,
) -> CapabilityCard:
    normalized_cases = tuple(
        CapabilityCase(
            case.identifier,
            case.prompt
            if case.prompt.rstrip().endswith("Assistant:")
            else f"User: {case.prompt.strip()}\nAssistant:",
            case.expected,
            case.kind,
        )
        for case in cases
    )
    if not normalized_cases:
        raise ValueError(f"capability card {name} requires test cases")
    answer_counts = Counter(case.expected for case in normalized_cases)
    return CapabilityCard(
        name,
        1,
        hypothesis,
        _EXACT_SCORER,
        normalized_cases,
        _GENERATION,
        {
            "normalization": "unicode_nfc_casefold_trim_collapse_whitespace",
            "match": "full_answer",
        },
        {
            "uniform_chance": 1 / len(answer_counts),
            "majority_answer": max(answer_counts.values()) / len(normalized_cases),
        },
        limitations,
    )


def _phase_e_cards() -> tuple[CapabilityCard, ...]:
    from klab.data.phase_e_tasks import (
        math_identity_test_cases,
        python_stdlib_test_cases,
    )

    alias_test_cases = chat_recall_cases("test")
    lexical_cases = tuple(
        CapabilityCase(
            case.identifier,
            "User: Look up the stored value for alias "
            f"{case.identifier.removeprefix('test-alias-')}. Return the value only.\nAssistant:",
            case.expected,
            "recall",
        )
        for case in alias_test_cases
    )
    paraphrase_cases = tuple(
        CapabilityCase(
            case.identifier,
            "System: Answer the requested alias with only its value.\n\nUser: "
            + case.prompt.rsplit("\n\nUser: ", 1)[-1].split("\n\nAssistant:", 1)[0]
            + "\n\nAssistant:",
            case.expected,
            "paraphrase",
        )
        for case in alias_test_cases
    )
    conversation_cases = tuple(
        CapabilityCase(
            case.identifier,
            case.prompt,
            case.expected,
            "conversation",
        )
        for case in alias_test_cases
        if "Assistant: Understood.\n\nUser:" in case.prompt
    )

    cards = [
        _task_card(
            "math-identity-novel-operands-v1",
            "Applies practiced arithmetic identities to disjoint held-out operands.",
            "Synthetic integer tasks measure this fixed identity set, not broad mathematical reasoning.",
            math_identity_test_cases(),
        ),
        _task_card(
            "python-stdlib-api-v1",
            "Applies documented Python standard-library operations to held-out inputs.",
            "Expected outputs are computed by an allowlisted trusted standard-library implementation; no generated Python is executed.",
            python_stdlib_test_cases(),
        ),
        _task_card(
            "lexical-recall-v1",
            "Retrieves held-out key-value associations in a canonical lookup format.",
            "Synthetic alias mappings measure string-association recall, not factual knowledge.",
            lexical_cases,
        ),
        _task_card(
            "conversation-followup-v1",
            "Uses prior dialogue context to answer held-out alias follow-ups.",
            "Six synthetic multi-turn cases share their alias domain with other suite cards.",
            conversation_cases,
        ),
        _task_card(
            "paraphrase-recall-v1",
            "Recalls trained alias associations under held-out paraphrased wording.",
            "Synthetic paraphrases probe wording variation on the same alias domain; they are not independent knowledge samples.",
            paraphrase_cases,
        ),
        _task_card(
            "application-task-v1",
            "Applies supplied reference facts to a small bounded task.",
            "The facts are present in each prompt; this measures instructed application, not retained knowledge.",
            (
                CapabilityCase(
                    "application-format-01",
                    "User: Reference: the station code for North is 4821. Return the code followed by '-ready', and nothing else.\nAssistant:",
                    "4821-ready",
                    "application",
                ),
                CapabilityCase(
                    "application-format-02",
                    "User: Reference: the station code for South is 7354. Return the code followed by '-ready', and nothing else.\nAssistant:",
                    "7354-ready",
                    "application",
                ),
            ),
        ),
        _task_card(
            "composition-task-v1",
            "Composes two explicit links into a two-hop result.",
            "Both links are supplied in the prompt; this is a bounded composition control, not unseen-graph recall.",
            (
                CapabilityCase(
                    "composition-two-hop-01",
                    "User: In this chain, Aster points to Birch and Birch points to Cedar. Follow two links from Aster. Give only the destination.\nAssistant:",
                    "Cedar",
                    "composition",
                ),
                CapabilityCase(
                    "composition-two-hop-02",
                    "User: In this chain, Delta points to Elm and Elm points to Finch. Follow two links from Delta. Give only the destination.\nAssistant:",
                    "Finch",
                    "composition",
                ),
            ),
        ),
        _task_card(
            "long-context-recall-v1",
            "Retrieves a named value after a longer supplied context and distractors.",
            "A fixed distractor prompt stresses retrieval near a 128-token context limit; it does not establish long-context scaling or external-fact recall.",
            (
                CapabilityCase(
                    "long-context-target-01",
                    "User: Dispatch codes: Kestrel=cobalt; Heron=amber; Wren=ivory; Lark=violet; Finch=coral; Robin=jade; Tern=linen; Gull=orchid. Kestrel code? Reply only with the code.\nAssistant:",
                    "cobalt",
                    "long_context",
                ),
            ),
        ),
        _task_card(
            "stale-source-retrieval-v1",
            "Rejects stale evidence instead of presenting it as current.",
            "One synthetic dated source tests freshness handling; it does not establish source authority.",
            (
                CapabilityCase(
                    "stale-source-01",
                    "User: Evidence policy: only sources dated 2024 or later can establish the current value. The only available record is dated 2019 and says the value is jade. Is a current value established? Reply only STALE or CURRENT.\nAssistant:",
                    "STALE",
                    "stale_source",
                ),
            ),
        ),
        _task_card(
            "conflicting-source-retrieval-v1",
            "Reports unresolved disagreement between equally current sources.",
            "Synthetic equal-date records test explicit conflict handling; they do not evaluate real-source credibility.",
            (
                CapabilityCase(
                    "conflicting-source-01",
                    "User: Two equally current records disagree: source A says the value is coral; source B says the value is indigo. No tie-breaker is available. Return only CONFLICT or a value.\nAssistant:",
                    "CONFLICT",
                    "conflicting_source",
                ),
            ),
        ),
        _task_card(
            "missing-source-retrieval-v1",
            "Abstains when the supplied source set does not contain the requested fact.",
            "The expected abstention is task-defined; it is not a calibrated uncertainty estimate.",
            (
                CapabilityCase(
                    "missing-source-01",
                    "User: Use only the evidence below; do not answer from prior knowledge. Evidence: no entry for the capital of the fictional country Meridia. What is Meridia's capital? If absent, reply only UNKNOWN.\nAssistant:",
                    "UNKNOWN",
                    "missing_source",
                ),
            ),
        ),
        _task_card(
            "instruction-over-memory-v1",
            "Follows an explicit evidence-only instruction over likely memorized associations.",
            "One familiar-fact abstention case; it does not establish general instruction following or eliminate training contamination.",
            (
                CapabilityCase(
                    "instruction-memory-01",
                    "System: Answer only from the supplied evidence. Ignore memorized or outside answers. If the fact is absent, reply exactly UNKNOWN.\nUser: Evidence: the document has no entry for the capital of France. What is the capital of France?\nAssistant:",
                    "UNKNOWN",
                    "instruction",
                ),
            ),
        ),
    ]
    return tuple(cards)


def capability_cards() -> tuple[CapabilityCard, ...]:
    return (
        _legacy_card(),
        _chat_card("chat-alias-retention-v1"),
        _chat_card("chat-alias-recall-v1"),
        _chat_card("chat-context-override-v1"),
        *_phase_e_cards(),
    )


def phase_e_behavior_suite() -> dict[str, Any]:
    """Describe separated outcomes without collapsing unlike metrics."""
    cards = {card.name: card for card in capability_cards()}
    categories = [
        {
            "id": "language_model",
            "measurement": "held-out validation token loss",
            "cards": [],
        },
        {"id": "lexical_recall", "cards": ["lexical-recall-v1"]},
        {
            "id": "factual_recall",
            "builder": "research tasks build wikidata-mini --output DIR",
            "card_file": "cards/wikidata-mini-factual-recall-v1.json",
        },
        {
            "id": "paraphrase",
            "cards": ["paraphrase-recall-v1"],
            "generated_card_file": "cards/wikidata-mini-paraphrase-v1.json",
        },
        {"id": "api_behavior", "cards": ["python-stdlib-api-v1"]},
        {"id": "math", "cards": ["math-identity-novel-operands-v1"]},
        {"id": "application", "cards": ["application-task-v1"]},
        {"id": "composition", "cards": ["composition-task-v1"]},
        {"id": "long_context", "cards": ["long-context-recall-v1"]},
        {"id": "override", "cards": ["chat-context-override-v1"]},
        {"id": "instruction_following", "cards": ["instruction-over-memory-v1"]},
        {"id": "conversation", "cards": ["conversation-followup-v1"]},
        {"id": "stale_source", "cards": ["stale-source-retrieval-v1"]},
        {"id": "conflicting_source", "cards": ["conflicting-source-retrieval-v1"]},
        {"id": "missing_source", "cards": ["missing-source-retrieval-v1"]},
    ]
    return {
        "format": "klab-phase-e-behavior-suite",
        "version": 1,
        "categories": categories,
        "card_digests": {
            card_name: cards[card_name].digest
            for category in categories
            for card_name in category.get("cards", [])
        },
        "claim_boundary": (
            "Report held-out language-model loss separately from each frozen exact-answer "
            "card. Category scores are not interchangeable and are not combined."
        ),
    }


_PHASE_E_CARD_NAMES = frozenset(
    {
        "math-identity-novel-operands-v1",
        "python-stdlib-api-v1",
        "lexical-recall-v1",
        "conversation-followup-v1",
        "paraphrase-recall-v1",
        "application-task-v1",
        "composition-task-v1",
        "long-context-recall-v1",
        "stale-source-retrieval-v1",
        "conflicting-source-retrieval-v1",
        "missing-source-retrieval-v1",
        "instruction-over-memory-v1",
    }
)


def list_capability_cards() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": card.name,
            "version": card.version,
            "digest": card.digest,
            "hypothesis": card.hypothesis,
            "limitations": card.limitations,
        }
        for card in capability_cards()
    )


def task_card_payload(
    name: str,
    hypothesis: str,
    limitations: str,
    cases: Any,
) -> dict[str, Any]:
    """Serialize generated cases through the existing strict capability schema."""
    card = _task_card(name, hypothesis, limitations, cases)
    return capability_card_payload(card)


def capability_card_payload(card: CapabilityCard) -> dict[str, Any]:
    """Serialize and validate a v2 or family-bound v3 card."""
    payload = {"format": card.format, **_card_payload(card), "digest": card.digest}
    _card_from_mapping(payload)
    return payload


def describe_capability_card(name: str) -> dict[str, Any]:
    card = capability_card(name)
    return capability_card_payload(card)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _case_lineage(raw: Any, *, split: str = "test") -> CapabilityCaseLineage:
    keys = {
        "split",
        "source_document_family",
        "world_id",
        "template_family",
        "parent_content_hashes",
    }
    if not isinstance(raw, dict) or set(raw) != keys or raw["split"] != split:
        raise ValueError("capability case lineage requires the declared split")
    source, world, template, parents = (
        raw["source_document_family"],
        raw["world_id"],
        raw["template_family"],
        raw["parent_content_hashes"],
    )
    if (
        any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (source, world)
        )
        or (source is None and world is None)
        or not isinstance(template, str)
        or not template.strip()
        or not isinstance(parents, list)
        or not parents
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in parents
        )
        or len(set(parents)) != len(parents)
    ):
        raise ValueError(
            "capability case lineage requires family IDs and SHA-256 parents"
        )
    return CapabilityCaseLineage(split, source, world, template, tuple(parents))


def check_capability_family_exclusion(
    card: CapabilityCard, ledger: Mapping[str, list[dict[str, Any]]]
) -> None:
    """Reject v3 test cases sharing families, worlds or parent bytes with train/validation.

    Ledger rows use the same lineage schema as card cases, with their own split.
    This checks declared identities; the caller must independently audit near-duplicates.
    """
    if card.format != _FAMILY_CARD_FORMAT:
        raise ValueError("family exclusion requires a v3 capability card")
    capability_card_payload(card)
    if not isinstance(ledger, Mapping) or set(ledger) != {"train", "validation"}:
        raise ValueError("family ledger requires train and validation rows")
    heldout = [case.lineage for case in card.cases]
    if any(lineage is None for lineage in heldout):
        raise ValueError("v3 cases require lineage")
    for split, rows in ledger.items():
        if not isinstance(rows, list):
            raise TypeError("family ledger rows must be lists")
        for row in rows:
            prior = _case_lineage(row, split=split)
            for current in heldout:
                assert current is not None
                if (
                    (
                        prior.source_document_family is not None
                        and prior.source_document_family
                        == current.source_document_family
                    )
                    or (
                        prior.world_id is not None
                        and prior.world_id == current.world_id
                    )
                    or prior.template_family == current.template_family
                    or bool(
                        set(prior.parent_content_hashes)
                        & set(current.parent_content_hashes)
                    )
                ):
                    raise ValueError(f"capability test lineage overlaps {split}")


def _card_from_mapping(raw: Mapping[str, Any]) -> CapabilityCard:
    required = {
        "format",
        "name",
        "version",
        "hypothesis",
        "scorer",
        "cases",
        "generation",
        "scoring",
        "controls",
        "limitations",
    }
    unknown, missing = set(raw) - (required | {"digest"}), required - set(raw)
    card_format = raw.get("format")
    if unknown or missing or card_format not in {_CARD_FORMAT, _FAMILY_CARD_FORMAT}:
        raise ValueError(
            f"invalid capability card schema: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    data = {key: value for key, value in raw.items() if key not in {"format", "digest"}}
    if "digest" in raw and (
        not isinstance(raw["digest"], str) or raw["digest"] != _digest(data)
    ):
        raise ValueError("capability card digest does not match content")
    if (
        not isinstance(data["name"], str)
        or not data["name"].strip()
        or "/" in data["name"]
        or "\\" in data["name"]
    ):
        raise ValueError("capability card name must be a safe nonempty string")
    if not _is_positive_int(data["version"]):
        raise ValueError("capability card version must be a positive integer")
    if card_format == _FAMILY_CARD_FORMAT and data["version"] != 3:
        raise ValueError("v3 capability cards require version 3")
    if data["scorer"] not in (_EXACT_SCORER, _LITERAL_SCORER):
        raise ValueError("user cards require a supported full-answer scorer")
    generation = data["generation"]
    if not isinstance(generation, dict) or set(generation) != set(_GENERATION):
        raise ValueError(
            "user card generation must use the fixed deterministic protocol"
        )
    if (
        not _is_positive_int(generation["max_new_tokens"])
        or isinstance(generation["temperature"], bool)
        or generation["temperature"] != 0.0
        or isinstance(generation["top_k"], bool)
        or generation["top_k"] != 0
        or isinstance(generation["seed"], bool)
        or generation["seed"] != 0
        or not isinstance(generation["stop_sequences"], list)
        or tuple(generation["stop_sequences"]) != _GENERATION["stop_sequences"]
    ):
        raise ValueError(
            "user card generation must use the fixed deterministic protocol"
        )
    expected_scoring = {
        "normalization": (
            "trim_outer_whitespace"
            if data["scorer"] == _LITERAL_SCORER
            else "unicode_nfc_casefold_trim_collapse_whitespace"
        ),
        "match": "full_answer",
    }
    if data["scoring"] != expected_scoring:
        raise ValueError("user card scoring must use the fixed full-answer schema")
    if (
        not isinstance(data["controls"], dict)
        or set(data["controls"]) != {"uniform_chance", "majority_answer"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            for value in data["controls"].values()
        )
    ):
        raise ValueError(
            "card controls must declare bounded uniform and majority baselines"
        )
    if not all(
        isinstance(data[key], str) and data[key].strip()
        for key in ("hypothesis", "limitations")
    ):
        raise ValueError("capability card text fields must be nonempty strings")
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("capability card must contain nonempty cases")
    cases: list[CapabilityCase] = []
    for item in data["cases"]:
        allowed = {"identifier", "prompt", "expected", "kind", "semantic_query"}
        if card_format == _FAMILY_CARD_FORMAT:
            allowed.add("lineage")
        if (
            not isinstance(item, dict)
            or set(item) - allowed
            or not {"identifier", "prompt", "expected", "kind"} <= set(item)
            or (card_format == _FAMILY_CARD_FORMAT and "lineage" not in item)
        ):
            raise ValueError("capability card case schema is invalid")
        if (
            not all(
                isinstance(item[key], str) and item[key].strip()
                for key in ("identifier", "prompt", "expected", "kind")
            )
            or item["kind"] not in _CASE_KINDS
        ):
            raise ValueError("capability card case text or kind is invalid")
        semantic_query = None
        if "semantic_query" in item:
            raw_query = item["semantic_query"]
            if (
                not isinstance(raw_query, dict)
                or set(raw_query) != {"encoder", "vector"}
                or not isinstance(raw_query["encoder"], dict)
                or not isinstance(raw_query["vector"], list)
                or not raw_query["vector"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or abs(float(value)) > torch.finfo(torch.float32).max
                    for value in raw_query["vector"]
                )
            ):
                raise ValueError("capability semantic query schema is invalid")
            semantic_query = CapabilitySemanticQuery(
                EncoderIdentity.model_validate(raw_query["encoder"]),
                tuple(float(value) for value in raw_query["vector"]),
            )
        lineage = (
            _case_lineage(item["lineage"])
            if card_format == _FAMILY_CARD_FORMAT
            else None
        )
        cases.append(
            CapabilityCase(
                item["identifier"],
                item["prompt"],
                item["expected"],
                item["kind"],
                semantic_query,
                lineage,
            )
        )
    if len({case.identifier for case in cases}) != len(cases):
        raise ValueError("capability card case identifiers must be unique")
    if card_format == _FAMILY_CARD_FORMAT:
        worlds: set[str] = set()
        parents: set[str] = set()
        for case in cases:
            assert case.lineage is not None
            if case.lineage.world_id is not None:
                if case.lineage.world_id in worlds:
                    raise ValueError("capability card world appears in multiple cases")
                worlds.add(case.lineage.world_id)
            if parents.intersection(case.lineage.parent_content_hashes):
                raise ValueError("capability card parent appears in multiple cases")
            parents.update(case.lineage.parent_content_hashes)
    return CapabilityCard(
        data["name"],
        data["version"],
        data["hypothesis"],
        data["scorer"],
        tuple(cases),
        {**generation, "stop_sequences": tuple(generation["stop_sequences"])},
        data["scoring"],
        data["controls"],
        data["limitations"],
        card_format,
    )


def load_capability_card(path: Path) -> CapabilityCard:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load capability card {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError("capability card JSON must be an object")
    return _card_from_mapping(raw)


def capability_card(name: str) -> CapabilityCard:
    if name == "engram-recall-v1":
        return _legacy_card()
    if name in {
        "chat-alias-retention-v1",
        "chat-alias-recall-v1",
        "chat-context-override-v1",
    }:
        return _chat_card(name)
    if name in _PHASE_E_CARD_NAMES:
        return next(card for card in _phase_e_cards() if card.name == name)
    path = Path(name)
    if path.suffix == ".json" or path.exists():
        return load_capability_card(path)
    raise ValueError(f"unknown capability card: {name}")


def _normalize_answer(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().strip().split())


def _passes(card: CapabilityCard, response: str, expected: str) -> bool:
    if card.scorer == _LEGACY_SCORER:
        return _first_word(response) == _first_word(expected)
    if card.scorer == _LITERAL_SCORER:
        return response.strip() == expected.strip()
    if card.scorer == _EXACT_SCORER:
        return _normalize_answer(response) == _normalize_answer(expected)
    raise ValueError(f"unsupported capability scorer: {card.scorer}")


def evaluate_capability(
    card: CapabilityCard,
    model: Any,
    tokenizer: Tokenizer,
    max_seq_len: int,
    device: torch.device | str,
    *,
    engine: MLXEngine | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for case in card.cases:
        prompt_tokens = len(tokenizer.encode(case.prompt, add_special_tokens=False).ids)
        if prompt_tokens + card.generation["max_new_tokens"] > max_seq_len:
            failures.append(
                {
                    "id": case.identifier,
                    "reason": "prompt_and_response_exceed_max_context",
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": card.generation["max_new_tokens"],
                    "max_seq_len": max_seq_len,
                }
            )
    protocol = {"max_seq_len": max_seq_len, **card.generation}
    common = {
        "format": _RESULT_FORMAT,
        "card": card.name,
        "card_digest": card.digest,
        "evaluation_source_sha256": source_identity()["sha256"],
        "hypothesis": card.hypothesis,
        "limitations": card.limitations,
        "scorer": card.scorer,
        "controls": dict(card.controls),
        "generation": protocol,
        "case_count": len(card.cases),
    }
    if failures:
        return {
            **common,
            "valid": False,
            "score": None,
            "failures": failures,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    for case in card.cases:
        semantic_queries = None
        if case.semantic_query is not None:
            if engine is not None or not isinstance(device, torch.device):
                raise ValueError("semantic capability queries require a PyTorch device")
            semantic_queries = SemanticQueryBatch(
                case.semantic_query.encoder,
                torch.tensor(
                    [case.semantic_query.vector], dtype=torch.float32, device=device
                ),
            )
        completion = generate(
            model,
            tokenizer,
            case.prompt,
            max_seq_len,
            card.generation["max_new_tokens"],
            device,
            temperature=card.generation["temperature"],
            top_k=card.generation["top_k"],
            seed=card.generation["seed"],
            strict_context=True,
            stop_sequences=card.generation["stop_sequences"],
            engine=engine,
            semantic_queries=semantic_queries,
        )
        response = assistant_reply(completion, case.prompt)
        results.append(
            {
                "id": case.identifier,
                "kind": case.kind,
                "prompt": case.prompt,
                "expected": case.expected,
                "response": response,
                "prompt_tokens": len(
                    tokenizer.encode(case.prompt, add_special_tokens=False).ids
                ),
                "response_tokens": len(
                    tokenizer.encode(response, add_special_tokens=False).ids
                ),
                "passed": _passes(card, response, case.expected),
            }
        )
    passed = sum(int(item["passed"]) for item in results)
    return {
        **common,
        "valid": True,
        "passed": passed,
        "score": passed / len(results),
        "results": results,
    }


def write_capability_result(run: Path, result: dict[str, Any]) -> Path:
    """Write immutable content-addressed evidence without trusting card/path text."""
    result_digest = _digest(result)
    identity = result.get("identity")
    checkpoint = (
        identity.get("checkpoint_sha256")
        if isinstance(identity, dict)
        else result.get("checkpoint_sha256")
    )
    checkpoint_component = _digest(str(checkpoint or "unbound"))[:12]
    payload = {**result, "result_digest": result_digest}
    evaluations = run / "evaluations"
    path = evaluations / f"capability-{checkpoint_component}-{result_digest}.json"
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        write_bytes_exclusive(path, encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError(f"refusing conflicting capability result at {path}")
    return path


def _scrub_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    if not path and not isinstance(value, dict):
        return value
    if isinstance(value, dict):
        ignored_top_level = {"name", "logging", "checkpoint", "evaluation", "staging"}
        return {
            key: _scrub_config(item, path + (key,))
            for key, item in value.items()
            if not (not path and key in ignored_top_level)
        }
    if isinstance(value, list):
        return [_scrub_config(item, path) for item in value]
    if path == ("dataset", "allocation_manifest_path") and isinstance(
        value, (str, Path)
    ):
        # Keep allocation profiles comparable without leaking machine-local roots.
        return Path(value).name
    if path and (
        path[-1] == "path"
        or path[-1].endswith("_path")
        or path[-1] in {"cache_dir", "root_dir", "output_dir"}
    ):
        return None
    return value


def _identity(result: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = result.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("capability result lacks immutable identity")
    required = {
        "checkpoint_sha256",
        "step",
        "tokens_seen",
        "config",
        "tokenizer_sha256",
        "data_sha256",
        "source_identity_sha256",
        "runtime",
    }
    missing = required - set(identity)
    if missing:
        raise ValueError(f"capability result identity missing {sorted(missing)}")
    return identity


def compare_results(
    base: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    vary: str = "memory",
    vary_fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if vary not in {"memory", "attention", "ffn", "scale", "none", "custom"}:
        raise ValueError("vary must be memory, attention, ffn, scale, none, or custom")
    if vary == "custom":
        if (
            not vary_fields
            or any(not is_dotted_path(field) for field in vary_fields)
            or len(vary_fields) != len(set(vary_fields))
        ):
            raise ValueError("custom comparisons require unique dotted vary_fields")
    elif vary_fields is not None:
        raise ValueError("vary_fields are only valid with vary='custom'")
    if base.get("card_digest") != variant.get("card_digest"):
        raise ValueError("capability cards differ")
    if base.get("evaluation_source_sha256") != variant.get("evaluation_source_sha256"):
        raise ValueError("capability evaluator source differs")
    base_identity, variant_identity = _identity(base), _identity(variant)
    keys: tuple[str, ...] = (
        "step",
        "tokens_seen",
        "tokenizer_sha256",
        "data_sha256",
        "source_identity_sha256",
        "runtime",
    )
    if "training_runtime" in base_identity or "training_runtime" in variant_identity:
        keys += ("training_runtime",)
    for key in keys:
        if base_identity.get(key) != variant_identity.get(key):
            raise ValueError(f"comparison identity differs at {key}")
    base_config_data = base_identity["config"]
    variant_config_data = variant_identity["config"]
    if (
        not isinstance(base_config_data, Mapping)
        or not isinstance(variant_config_data, Mapping)
        or base_config_data.get("seed") != variant_config_data.get("seed")
    ):
        raise ValueError("comparison identity differs at seed")
    base_config = _scrub_config(base_identity["config"])
    variant_config = _scrub_config(variant_identity["config"])
    available = set(flatten_config(base_config)) | set(flatten_config(variant_config))
    permitted = {
        "memory": {f"model.{field}" for field in _MEMORY_FIELDS},
        "attention": {key for key in available if key.startswith("attention.")},
        "ffn": {f"model.{field}" for field in _FFN_FIELDS},
        "scale": {f"model.{field}" for field in _SCALE_FIELDS},
        "none": set(),
        "custom": set(vary_fields or ()),
    }[vary]
    declared = permitted
    try:
        differences = compare_configs(
            base_config,
            variant_config,
            expected=permitted & available,
            require_all=False,
        )
    except ValueError as error:
        if str(error).startswith("undeclared changed fields:"):
            raise ValueError(
                f"comparison varies controls outside {vary}: "
                f"{str(error).split(': ', 1)[1]}"
            ) from error
        raise
    if not differences and vary != "none":
        raise ValueError("comparison configurations are identical")
    if vary == "custom":
        unchanged = sorted(declared - set(differences))
        if unchanged:
            raise ValueError(
                f"custom vary_fields include unchanged controls: {', '.join(unchanged)}"
            )
    if (
        not base.get("valid")
        or not variant.get("valid")
        or not isinstance(base.get("score"), (int, float))
        or not isinstance(variant.get("score"), (int, float))
    ):
        outcome, delta = "inconclusive", None
    else:
        delta = variant["score"] - base["score"]
        outcome = "supported_observation" if delta > 0 else "no_improvement"
    base_cases, variant_cases = (
        {item["id"]: item for item in base.get("results", [])},
        {item["id"]: item for item in variant.get("results", [])},
    )
    common = sorted(set(base_cases) & set(variant_cases))
    gains = sum(
        not base_cases[item]["passed"] and variant_cases[item]["passed"]
        for item in common
    )
    losses = sum(
        base_cases[item]["passed"] and not variant_cases[item]["passed"]
        for item in common
    )
    return {
        "format": "capability_comparison_v3"
        if vary == "custom"
        else "capability_comparison_v2",
        "card": base.get("card"),
        "card_digest": base.get("card_digest"),
        "vary": vary,
        "differences": differences,
        **({"vary_fields": list(vary_fields or ())} if vary == "custom" else {}),
        "base_score": base.get("score"),
        "variant_score": variant.get("score"),
        "score_delta": delta,
        "paired_case_count": len(common),
        "paired_gains": gains,
        "paired_losses": losses,
        "checkpoints": {
            "base": {
                key: base_identity.get(key)
                for key in (
                    "run_id",
                    "checkpoint_sha256",
                    "checkpoint_relative_path",
                    "step",
                    "tokens_seen",
                )
            },
            "variant": {
                key: variant_identity.get(key)
                for key in (
                    "run_id",
                    "checkpoint_sha256",
                    "checkpoint_relative_path",
                    "step",
                    "tokens_seen",
                )
            },
        },
        "parameter_inventory": {
            "base": base_identity.get("parameter_inventory"),
            "variant": variant_identity.get("parameter_inventory"),
        },
        "outcome": outcome,
        "interpretation": "Matched observation only; no universal benefit, efficiency claim, or statistical significance is inferred.",
    }
