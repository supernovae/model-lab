"""Release outputs shared by every corpus builder.

The in-memory builder (`pipeline.build`) and the bounded-memory builder
(`large_build.build_large`) decide which documents exist differently, but they
describe, report and publish a build identically. That description lives here
so the two cannot drift.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json
from klab.corpus.project import declaration_payload
from klab.corpus.rights import reported_spdx_expression
from klab.corpus.split_assignments import SPLITS, UNIT_FIELDS, copy_assignments
from klab.training.manifest import sha256_file

_UNREPRODUCIBLE_MODES = frozenset(
    {
        "reference_only",
        "derived_only",
        "unknown",
        "rejected",
        "metadata_reconstruction_only",
        "not_redistributable",
        "review_required",
    }
)
_ELIGIBILITY_STATES = (
    "eligible",
    "eligible_with_obligations",
    "review_required",
    "ineligible",
)
_RIGHTS_SUMMARY_KEYS = frozenset(
    {
        "training_eligibility",
        "spdx_expressions",
        "redistribution_modes",
        "unresolved_rights_files",
        "weight_license_status",
        "publication_mode",
        "release_policy",
    }
)
_SCALE_KEYS = ("documents", "utf8_bytes", "characters", "whitespace_words")
ADMISSION_COPY = "record-admission.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def stage_declared_inputs(
    project: Any, staging: Path, admission_path: Path | None
) -> None:
    """Copy the frozen split assignments and reviewed admission into a staging build."""
    copy_assignments(project, staging)
    if admission_path is None:
        return
    copy = staging / ADMISSION_COPY
    shutil.copyfile(admission_path, copy)
    if sha256_file(copy) != project.release.record_admission.sha256.lower():
        raise ValueError("record admission changed during build")


def publish_staging(staging: Path, target: Path) -> None:
    """Make the staged files durable, then rename the directory into place."""
    from klab.result_publication import sync_directory
    from klab.safe_io import rename_noreplace

    for entry in sorted(staging.rglob("*"), reverse=True):
        if entry.is_file():
            with entry.open("rb") as handle:
                os.fsync(handle.fileno())
        elif entry.is_dir():
            sync_directory(entry)
    sync_directory(staging)
    target.parent.mkdir(parents=True, exist_ok=True)
    rename_noreplace(staging, target)
    sync_directory(target.parent)


def source_declarations(project: Any, lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": source.id,
            "kind": source.kind,
            "canonical_uri": source.canonical_uri,
            "revision": source.revision,
            "license": source.license,
            "redistribution": source.rights.redistribution_mode,
            "license_url": source.license_url,
            "rights_policy": source.rights.model_dump(mode="json"),
            **(
                {"explicit_training_restriction": source.explicit_training_restriction}
                if source.explicit_training_restriction is not None
                else {}
            ),
            "origin": source.origin,
            "source_family": source.source_family,
            "snapshot_sha256": lock["sources"]
            .get(source.id, {})
            .get("snapshot_sha256"),
            "reproducibility_class": (
                "generator_dependent"
                if source.kind == "inference_generator"
                else "script_reproducible_not_redistributed"
                if source.rights.redistribution_mode in _UNREPRODUCIBLE_MODES
                else "fully_reproducible"
            ),
        }
        for source in project.sources
    ]


def license_metadata_entry(
    source: Any, *, path: str, sha256: str, size: int
) -> dict[str, Any]:
    return {
        "source_id": source.id,
        "path": path,
        "sha256": sha256,
        "size": size,
        "role": "license_metadata",
        "canonical_uri": source.canonical_uri,
        "revision": source.revision,
    }


def rights_file_entry(
    source: Any,
    *,
    path: str,
    sha256: str,
    size: int,
    role: str,
    rights: Mapping[str, Any],
    admitted: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """One file's rights decision; `admitted` is its reviewed admission, if any."""
    return {
        "source_id": source.id,
        "path": path,
        "sha256": sha256,
        "size": size,
        "role": role,
        "canonical_uri": source.canonical_uri,
        "revision": source.revision,
        "license_url": source.license_url,
        "rights": dict(rights),
        **(
            {
                "admission_policy_id": admitted["policy_id"],
                "admission_policy_sha256": admitted["policy_sha256"],
                "admission_license_label": admitted["license_label"],
                "admission_policy_spdx_expression": admitted["policy_spdx_expression"],
            }
            if admitted is not None
            else {}
        ),
    }


def rights_report(
    project: Any, sources: list[dict[str, Any]], rights_files: list[dict[str, Any]]
) -> dict[str, Any]:
    files = [row for row in rights_files if row["role"] == "document"]
    eligibility = Counter(row["rights"]["training_eligibility"] for row in files)
    bytes_by_state: Counter[str] = Counter()
    expressions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    source_spdx = {
        source.id: source.rights.spdx_expression for source in project.sources
    }
    for item in files:
        rights = item["rights"]
        bytes_by_state[rights["training_eligibility"]] += item["size"]
        expressions[reported_spdx_expression(item, source_spdx[item["source_id"]])] += 1
        modes[rights["redistribution_mode"]] += 1
    return {
        "schema_version": 1,
        "publication_mode": project.release.publication_mode,
        "release_policy": project.release.policy.model_dump(mode="json"),
        "weight_license_status": "separate_analysis_required",
        "sources": sources,
        "files": rights_files,
        "training_eligibility": {
            state: {
                "files": eligibility[state],
                "source_bytes": bytes_by_state[state],
                "source_tokens": None,
                "token_count_reason": "tokenizer_not_declared",
            }
            for state in _ELIGIBILITY_STATES
        },
        "spdx_expressions": dict(expressions),
        "redistribution_modes": dict(modes),
        "unresolved_rights_files": eligibility["review_required"],
        "advisory": "Source/derived-data rights and model-weight licensing are separate decisions; not a legal conclusion.",
    }


def record_overlap(
    leakage: list[dict[str, Any]],
    warnings: list[str],
    *,
    unit: str,
    field: str,
    family: str,
    splits: Iterable[str],
    diagnostics: Path,
) -> None:
    """Record a family seen in several splits; refuse it if it is the split unit."""
    splits = sorted(splits)
    leakage.append({"family_type": field, "family_id": family, "splits": splits})
    if UNIT_FIELDS[unit] == field:
        write_json(
            diagnostics, {"error": "shared split policy unit", "leakage": leakage}
        )
        raise ValueError(f"split policy unit {family} spans splits")
    warnings.append(f"{field} {family} overlaps {', '.join(splits)}")


class Tally:
    """Counts of the retained documents that the release report describes."""

    def __init__(self) -> None:
        self.by_split: Counter[str] = Counter()
        self.by_source: Counter[str] = Counter()
        self.by_kind: Counter[str] = Counter()
        self.by_domain: dict[str, dict[str, int]] = defaultdict(
            lambda: {"documents": 0, "utf8_bytes": 0}
        )
        self.source_scale: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)

    def add(self, doc: Mapping[str, Any]) -> None:
        text = doc["text"]
        size = len(text.encode("utf-8"))
        split = doc["split"]
        self.by_split[split] += 1
        self.by_source[doc["source_id"]] += 1
        self.by_kind[doc["document_kind"]] += 1
        for domain in doc["domains"]:
            self.by_domain[domain]["documents"] += 1
            self.by_domain[domain]["utf8_bytes"] += size
        counts = self.source_scale[doc["source_id"]].setdefault(
            split, dict.fromkeys(_SCALE_KEYS, 0)
        )
        counts["documents"] += 1
        counts["utf8_bytes"] += size
        counts["characters"] += len(text)
        counts["whitespace_words"] += len(text.split())


def release_report(
    project: Any,
    rights: Mapping[str, Any],
    tally: Tally,
    *,
    document_count: int,
    view_counts: Mapping[str, Mapping[str, int]],
    duplicate_groups: int,
    warnings: list[str],
    lexical: int = 0,
    semantic: int = 0,
    chat: int = 0,
    generation_statuses: Mapping[str, int] | None = None,
    generator_count: int = 0,
    scenario_count: int = 0,
    tokenized: bool = False,
) -> dict[str, Any]:
    """The `report.json` body, before its measurement section."""
    kept = sum(tally.by_split.values())
    mixture = declaration_payload(project.release).get("mixture", {})
    statuses = Counter(generation_statuses or {})
    statuses["oracle_verified"] += scenario_count
    status_total = sum(statuses.values())
    train_scale = {
        key: sum(
            per_split.get("train", {}).get(key, 0)
            for per_split in tally.source_scale.values()
        )
        for key in _SCALE_KEYS
    }
    return {
        "schema_version": 1,
        "release_policy": project.release.policy.model_dump(mode="json"),
        "normalizer": project.release.normalizer.model_dump(mode="json"),
        "corpus_id": project.config.id,
        "requested_mixture": mixture,
        "actual_mixture": {
            domain: {
                "documents": tally.by_domain[domain]["documents"],
                "utf8_bytes": tally.by_domain[domain]["utf8_bytes"],
                "actual_tokens": None,
                "token_count_reason": "tokenizer_not_declared",
            }
            for domain in sorted(set(mixture) | set(tally.by_domain))
        },
        "document_count": document_count,
        "kept_document_count": kept,
        "split_counts": {split: tally.by_split[split] for split in SPLITS},
        "view_counts": view_counts,
        "source_counts": dict(tally.by_source),
        "source_concentration": {
            source: count / kept for source, count in sorted(tally.by_source.items())
        }
        if kept
        else {},
        "kind_counts": dict(tally.by_kind),
        "lexical_count": lexical,
        "semantic_count": semantic,
        "chat_count": chat,
        "generation_statuses": dict(statuses),
        "generation_validation_ratios": {
            status: count / status_total for status, count in sorted(statuses.items())
        }
        if status_total
        else {},
        "generator_count": generator_count,
        "scenario_count": scenario_count,
        "dedupe": {"groups": duplicate_groups, "dropped": document_count - kept},
        "warnings": warnings,
        "token_count_reason": None if tokenized else "tokenizer_not_declared",
        "source_scale": {
            source: dict(sorted(splits.items()))
            for source, splits in sorted(tally.source_scale.items())
        },
        "train_source_scale": {
            **train_scale,
            "utf8_bytes_div4_proxy": train_scale["utf8_bytes"] // 4,
            "actual_tokens": None,
            "token_count_reason": "final_tokenizer_not_fitted",
            "proxy_warning": (
                "Whitespace words and UTF-8 bytes/4 are tokenizer-independent "
                "descriptors, not measured token counts."
            ),
        },
        "rights": {
            key: value for key, value in rights.items() if key in _RIGHTS_SUMMARY_KEYS
        },
    }


def write_release_files(
    staging: Path,
    project: Any,
    report: dict[str, Any],
    *,
    build_id: str,
    identity: Mapping[str, Any],
    stages: list[dict[str, Any]],
    lock: Mapping[str, Any],
    tokenizer_sha256: str | None = None,
) -> None:
    """Write `report.json`, `report.md` and the file inventory `build.json`."""
    write_json(staging / "report.json", report)
    kept = report["kept_document_count"]
    (staging / "report.md").write_text(
        f"# Corpus {project.config.id}\n\nDocuments: {kept} retained / "
        f"{report['document_count']} total.\n\n"
        + (
            f"Training tokens: {report['measurement']['training_mixture']['actual_tokens']} with pinned tokenizer {tokenizer_sha256}.\n"
            if tokenizer_sha256 is not None
            else "No tokenizer declared; token counts unavailable.\n"
        ),
        encoding="utf-8",
    )
    write_json(
        staging / "build.json",
        {
            "schema_version": 1,
            "build_id": build_id,
            "identity": identity,
            "stages": stages,
            "files": {
                str(path.relative_to(staging)): {
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
            "snapshots": [
                {"source_id": source_id, "sha256": entry["snapshot_sha256"]}
                for source_id, entry in sorted(lock["sources"].items())
                if entry.get("snapshot_path")
            ],
        },
    )
