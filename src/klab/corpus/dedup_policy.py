"""Duplicate-resolution rules shared by the in-memory and disk-backed builders.

Both builders find duplicate groups their own way (dictionaries versus an
on-disk index) but decide every group here, so a policy option behaves the same
on either path.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from klab.canonical import canonical_json
from klab.corpus.project import CURATED_SOURCE_KINDS, DuplicatePolicy

CROSS_SPLIT_OVERLAP = "cross-split exact text overlap"
RIGHTS_CONFLICT = "duplicate source text has incompatible rights evidence"
PAGE_OVERLAP = "validation/test page or paper origin overlap"
CLEANING_CONFLICT = "discordant metadata-only cleaning decisions in duplicate component"
METADATA_ONLY = "exclude_lm_metadata_only"

_RIGHTS_FIELDS = (
    "training_eligibility",
    "license_references",
    "notices",
    "training_restriction",
)


STRICT_CHAIN = DuplicatePolicy(
    heldout_overlap="error",
    keys=("raw", "normalized"),
    resolution="chain_smallest_id",
    rights_conflict="error",
)


def is_strict_chain(policy: DuplicatePolicy) -> bool:
    """The only duplicate policy the completed-preparation build implements."""
    return policy == STRICT_CHAIN


def source_rank(source: Any) -> int:
    """Curated sources outrank pes2o, which outranks the rest."""
    if source.kind in CURATED_SOURCE_KINDS:
        return 0
    return 1 if source.id.startswith("pes2o") else 2


def rights_signature(doc: dict[str, Any]) -> bytes:
    """The rights evidence two copies of the same text must agree on."""
    rights = doc["rights"]
    return hashlib.sha256(
        canonical_json(
            {
                "license": doc["license"],
                "redistribution": doc["redistribution"],
                **{name: rights[name] for name in _RIGHTS_FIELDS},
            }
        )
    ).digest()


def is_metadata_only(doc: dict[str, Any]) -> bool:
    return (doc.get("structure") or {}).get("decision") == METADATA_ONLY


@dataclass(frozen=True)
class Member:
    """One document in a duplicate group."""

    id: str
    split: str
    rank: int
    kind: str
    signature: bytes | None = None


@dataclass(frozen=True)
class Group:
    representative: str
    splits: list[str]
    error: str | None
    # Train copies of text that a held-out split also contains.
    contaminated: bool


def ranked_choice(
    split: str, priority: str | None, rank: int, kind: str, identifier: str
) -> tuple[Any, ...]:
    """Sort key for `component_ranked`: lower wins."""
    return (split != priority if priority else False, rank, kind != "paper", identifier)


def resolve_group(
    method: str, members: Iterable[Member], policy: DuplicatePolicy
) -> Group:
    """Decide one duplicate group (documents sharing one key)."""
    ordered = sorted(members, key=lambda member: member.id)
    splits = {member.split for member in ordered}
    heldout = splits - {"train"}
    contaminated = (
        len(splits) > 1
        and policy.heldout_overlap == "drop_train_copy"
        and "train" in splits
        and len(heldout) == 1
    )
    priority = next(iter(heldout)) if heldout else None
    if policy.resolution == "component_ranked":
        representative = min(
            ordered,
            key=lambda m: ranked_choice(m.split, priority, m.rank, m.kind, m.id),
        ).id
    else:
        kept = [m for m in ordered if not contaminated or m.split != "train"]
        representative = kept[0].id
    error = None
    if len(splits) > 1 and not contaminated:
        error = CROSS_SPLIT_OVERLAP
    elif (
        method == "normalized"
        and policy.rights_conflict == "error"
        and len({member.signature for member in ordered}) > 1
    ):
        error = RIGHTS_CONFLICT
    return Group(representative, sorted(splits), error, contaminated)


def drop_reason(
    *,
    metadata_only: bool,
    split: str,
    is_representative: bool,
    representative_split: str,
) -> str | None:
    """Why a document leaves the language-model view, if it does."""
    if metadata_only:
        return "metadata_only_front_matter"
    if is_representative:
        return None
    if split == "train" and representative_split != "train":
        return "contaminated_heldout"
    return "duplicate"


class DuplicateError(ValueError):
    """A duplicate policy violation, with the groups observed before it."""

    def __init__(self, message: str, groups: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.groups = groups


def find_root(parents, item):
    """Union-find root of `item` in a parent array or mapping, compressing the path."""
    while parents[item] != item:
        parents[item] = parents[parents[item]]
        item = parents[item]
    return item


def deduplicate_documents(
    documents: list[dict[str, Any]],
    ranks: dict[str, int],
    policy: DuplicatePolicy,
    origin_keys: Callable[[dict[str, Any]], Iterable[str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Resolve duplicates for documents held in memory.

    Returns the duplicate groups (in canonical order) and each document's final
    representative. The disk-backed builder applies the same rules through
    `resolve_group`, `ranked_choice` and `drop_reason`.
    """
    by_id = {doc["document_id"]: doc for doc in documents}
    methods = [
        method
        for method in ("raw", "normalized", "canonical_origin")
        if method in policy.keys
    ]
    keyed: dict[str, dict[str, list[str]]] = {
        method: defaultdict(list) for method in methods
    }
    for doc in documents:
        identifier = doc["document_id"]
        if "raw" in keyed:
            keyed["raw"][doc["raw_content_sha256"]].append(identifier)
        if "normalized" in keyed:
            keyed["normalized"][doc["content_sha256"]].append(identifier)
        if "canonical_origin" in keyed:
            for key in origin_keys(doc):
                keyed["canonical_origin"][key].append(identifier)
    groups: list[dict[str, Any]] = []
    representative = {identifier: identifier for identifier in by_id}
    for method in methods:
        needs_rights = method == "normalized" and policy.rights_conflict == "error"
        for content_id, ids in sorted(keyed[method].items()):
            if len(ids) < 2:
                continue
            ids = sorted(ids)
            group = resolve_group(
                method,
                (
                    Member(
                        i,
                        by_id[i]["split"],
                        ranks[by_id[i]["source_id"]],
                        by_id[i]["document_kind"],
                        rights_signature(by_id[i]) if needs_rights else None,
                    )
                    for i in ids
                ),
                policy,
            )
            groups.append(
                {
                    "method": method,
                    "sha256": (
                        hashlib.sha256(content_id.encode()).hexdigest()
                        if method == "canonical_origin"
                        else content_id
                    ),
                    "origins": ids,
                    "splits": group.splits,
                    "representative": group.representative,
                }
            )
            if group.error:
                raise DuplicateError(group.error, groups)
            if policy.resolution == "chain_smallest_id":
                for i in ids:
                    if group.contaminated and by_id[i]["split"] == "train":
                        # Train copies follow the held-out document.
                        representative[i] = group.representative
                    else:
                        representative[i] = min(representative[i], group.representative)

    if policy.resolution == "component_ranked":
        # Raw, normalized, page-URL and paper-ID groups can be connected
        # transitively. Resolve each connected component once: a chain of
        # locally chosen representatives would otherwise introduce cycles.
        parents = {identifier: identifier for identifier in by_id}

        for group in groups:
            first = find_root(parents, group["origins"][0])
            for identifier in group["origins"][1:]:
                parents[find_root(parents, identifier)] = first
        components: dict[str, list[str]] = defaultdict(list)
        for identifier in by_id:
            components[find_root(parents, identifier)].append(identifier)
        for ids in components.values():
            heldout = {by_id[i]["split"] for i in ids} - {"train"}
            if len(heldout) > 1:
                raise DuplicateError(PAGE_OVERLAP, groups)
            if len({is_metadata_only(by_id[i]) for i in ids}) > 1:
                raise DuplicateError(CLEANING_CONFLICT, groups)
            priority = next(iter(heldout)) if heldout else None
            selected = min(
                ids,
                key=lambda i: ranked_choice(
                    by_id[i]["split"],
                    priority,
                    ranks[by_id[i]["source_id"]],
                    by_id[i]["document_kind"],
                    i,
                ),
            )
            for identifier in ids:
                representative[identifier] = selected

    def chosen(identifier: str) -> str:
        current = identifier
        while representative[current] != current:
            current = representative[current]
        representative[identifier] = current
        return current

    for identifier in representative:
        chosen(identifier)
    return groups, representative
