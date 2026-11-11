"""Conservative admission draft from verified bounded acquisition snapshots.

This prepares decisions for policy review. It never turns a flagged or
conflicting record into an eligible one, and the existing admission verifier
remains the final build/release gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from klab.canonical import canonical_digest
from klab.corpus.acquisition import declaration_sha256, verify_acquisition
from klab.corpus.bounded_metadata import metadata_operation, read_metadata
from klab.corpus.project import Project, source_declaration_payload
from klab.corpus.rights import (
    RightsPolicy,
    _admission_exceptions,
    _admission_profile,
    _git_file_exceptions,
    resolve_file_rights,
    verify_record_admission,
)
from klab.training.manifest import sha256_file
from klab.verification_proofs import verification_options

_MAX_RECORD_BYTES = 8 * 1024**2


def _decision(value: str, reason: str) -> dict[str, Any]:
    return {"decision": value, "reason": reason, "issues": []}


def _issue(field: str, remedy: str) -> dict[str, str]:
    return {
        "field": field,
        "owner": "KML Card 03 source audit",
        "remedy": remedy,
        "decision_impact": "Optional descriptive gap; source-policy coverage and pinned row identity still govern local use.",
    }


def _optional_issues(record: dict[str, Any], profile: str) -> list[dict[str, str]]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return []
    issues: list[dict[str, str]] = []
    if profile == "gutenberg":
        if not metadata.get("edition"):
            issues.append(
                _issue(
                    "metadata.edition",
                    "Reconcile edition from the retained Gutenberg book ID and source URL before later redistribution.",
                )
            )
    else:
        authors = metadata.get("authors")
        if not isinstance(authors, list) or len(authors) <= 1:
            issues.append(
                _issue(
                    "metadata.authors",
                    "Use the canonical page history for attribution; reconcile the optional author list before later redistribution.",
                )
            )
        locator = metadata.get("url")
        if isinstance(locator, str):
            try:
                has_revision = "oldid" in parse_qs(urlsplit(locator).query)
            except ValueError:
                has_revision = False
            if not has_revision:
                issues.append(
                    _issue(
                        "metadata.url.oldid",
                        "Retain the pinned row digest and canonical page/history route; recover the optional revision query if needed.",
                    )
                )
    return issues


def _additional_draft_flags(record: dict[str, Any], profile: str) -> tuple[str, ...]:
    """Conservatively hold observed scale-audit patterns for manual review.

    These flags only narrow a draft; they do not change the historical rights
    verifier or turn a flagged row into an eligible one.
    """
    if profile != "wikimedia":
        return ()
    content = record.get("text")
    if not isinstance(content, str):
        return ()
    flags: list[str] = []
    if re.search(r'["“”]', content):
        flags.append("quoted_text_context")
    if re.search(
        r"(?i)\b(?:sexual orientation|coming out|came out|partner|"
        r"accused|alleged|allegations?|misconduct|scandal)\b",
        content,
    ):
        flags.append("sensitive_biography_or_allegation_context")
    return tuple(flags)


def _draft_admission_manifest(
    project: Project,
    work_root: Path,
    template_path: Path,
    policy_document: Path,
    output_path: Path,
    store,
) -> dict[str, Any]:
    """Draft one complete v2 manifest; flagged cases remain quarantined.

    The policy template supplies reviewed source policies, not row decisions:
    ``policy_id``, ``policy_sha256`` and one item per source with ``source_id``,
    ``license_label`` and ``rights``. The result binds immutable snapshots and
    covers every selected HF row and Git file. No rights review is inferred
    from this mechanical draft.
    """
    root = work_root.resolve()
    output = output_path.resolve()
    if not output.is_relative_to(root) or output.exists() or output_path.is_symlink():
        raise ValueError("admission draft requires a new output inside the work root")
    if template_path.is_symlink() or not template_path.is_file():
        raise ValueError("admission policy template is missing or symlinked")
    if policy_document.is_symlink() or not policy_document.is_file():
        raise ValueError("admission policy document is missing or symlinked")
    template = read_metadata(template_path)
    if (
        not isinstance(template, dict)
        or not {"policy_id", "policy_sha256", "sources"}.issubset(template)
        or set(template)
        - {"policy_id", "policy_sha256", "sources", "application_binding"}
        or not isinstance(template["sources"], list)
        or any(
            not isinstance(item, dict)
            or set(item) - {"row_use_policy"}
            != {"source_id", "license_label", "rights"}
            for item in template["sources"]
        )
    ):
        raise ValueError("invalid admission policy template")
    if sha256_file(policy_document) != template["policy_sha256"]:
        raise ValueError("admission policy document SHA-256 mismatch")
    policies = {item["source_id"]: item for item in template["sources"]}
    if len(policies) != len(template["sources"]) or set(policies) != {
        source.id for source in project.sources
    }:
        raise ValueError("admission policy template source mismatch")

    lock = verify_acquisition(project, root, **verification_options(root))
    if "application_binding" in template:
        binding = template["application_binding"]
        sources = binding.get("sources") if isinstance(binding, dict) else None
        if (
            not isinstance(binding, dict)
            or set(binding) != {"project_sha256", "acquisition_lock_sha256", "sources"}
            or not isinstance(sources, dict)
            or set(sources) != {source.id for source in project.sources}
            or binding["project_sha256"] != lock.get("project_sha256")
            or binding["acquisition_lock_sha256"]
            != sha256_file(root / "corpora" / project.config.id / "acquisition.json")
        ):
            raise ValueError("application-policy project or lock binding mismatch")
        for source in project.sources:
            expected = sources[source.id]
            entry = lock["sources"][source.id]
            if (
                not isinstance(expected, dict)
                or set(expected) != {"declaration_sha256", "snapshot_sha256"}
                or expected["declaration_sha256"] != declaration_sha256(source)
                or expected["declaration_sha256"] != entry["declaration_sha256"]
                or expected["snapshot_sha256"] != entry["snapshot_sha256"]
            ):
                raise ValueError(
                    f"application-policy source binding mismatch: {source.id}"
                )
    snapshots: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    counts = {"qualify": 0, "exclude": 0, "quarantine": 0}
    for source in project.sources:
        source_id = source.id
        row = lock["sources"][source_id]
        if row["snapshot_path"] is None:
            raise ValueError(f"missing acquired snapshot: {source_id}")
        snapshot_dir = Path(row["snapshot_path"])
        snapshot = read_metadata(snapshot_dir / "manifest.json")
        snapshots[source_id] = snapshot
        policy = policies[source_id]
        rights = RightsPolicy.model_validate(policy["rights"])
        from klab.corpus.row_qualification import (
            RowQualification,
            RowUsePolicy,
            use_evidence,
        )

        row_use = (
            RowUsePolicy.model_validate(policy["row_use_policy"])
            if "row_use_policy" in policy
            else None
        )
        if row_use is not None and (
            source.kind != "huggingface_dataset"
            or source.acquisition.qualification is None
        ):
            raise ValueError("row use policy requires declared HF row qualification")
        entry: dict[str, Any] = {
            "source_id": source_id,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "source_revision": source.revision,
            "license_label": policy["license_label"],
            "rights": rights.model_dump(mode="json"),
        }
        if row_use is not None:
            entry["row_use_policy"] = row_use.model_dump(mode="json")
        if source.kind == "huggingface_dataset":
            if len(snapshot["files"]) != 1 or len(snapshot["retrieval"]["shards"]) != 1:
                raise ValueError("admission draft requires one selected HF shard")
            file = snapshot["files"][0]
            shard = snapshot["retrieval"]["shards"][0]
            entry.update(
                sample_path=file["path"],
                sample_sha256=file["sha256"],
                source_shard_path=shard["source_shard_path"],
                source_shard_sha256=shard["source_shard_sha256"],
            )
            selected = shard["selected_rows"]
            decisions = store.new_rows()
            profile = _admission_profile(source_declaration_payload(source))
            with (snapshot_dir / "files" / file["path"]).open("rb") as stream:
                from klab.corpus.jsonl_records import iter_jsonl_records

                rows = iter_jsonl_records(
                    stream, source=file["path"], max_record_bytes=_MAX_RECORD_BYTES
                )
                for receipt in selected:
                    row = next(rows, None)
                    if row is None:
                        raise ValueError("selected HF rows exceed snapshot file")
                    record = row.value
                    envelope = record.pop("_klab_source", None)
                    original_sha = canonical_digest(record)
                    if (
                        not isinstance(envelope, dict)
                        or envelope.get("source_row_index")
                        != receipt["source_row_index"]
                        or original_sha != receipt["source_row_sha256"]
                    ):
                        raise ValueError("HF row differs from acquisition receipt")
                    conflict, flags = _admission_exceptions(
                        record,
                        profile,
                        source_declaration_payload(source),
                        row_use=row_use,
                    )
                    flags = tuple(
                        sorted(
                            set(flags) | set(_additional_draft_flags(record, profile))
                        )
                    )
                    if conflict is not None:
                        state = (
                            "exclude"
                            if "outside content namespace" in conflict
                            or conflict
                            == "explicit training-use restriction on source row"
                            else "quarantine"
                        )
                        choice = _decision(state, conflict)
                    elif flags:
                        choice = _decision(
                            "quarantine",
                            "manual exception review required: " + ", ".join(flags),
                        )
                    else:
                        choice = _decision(
                            "qualify",
                            "covered by reviewed source policy; no automated exception",
                        )
                    choice["issues"] = (
                        []
                        if profile == "declared_rows"
                        else _optional_issues(record, profile)
                    )
                    choice.update(
                        source_row_index=receipt["source_row_index"],
                        source_row_sha256=original_sha,
                    )
                    if row_use is not None:
                        choice["row_evidence"] = use_evidence(
                            record,
                            RowQualification.model_validate(
                                source.acquisition.qualification
                            ),
                            source.acquisition.text_field,
                            row_use,
                        )
                    store.append(decisions, choice)
                    counts[state if conflict is not None else choice["decision"]] += 1
                if next(rows, None) is not None:
                    raise ValueError("snapshot has undeclared HF rows")
            entry["records"] = decisions
        elif source.kind == "git":
            if rights.spdx_expression is None:
                raise ValueError("Git admission policy requires SPDX expression")
            decisions = []
            for file in snapshot["files"]:
                path = file["path"]
                raw = (snapshot_dir / "files" / path).read_bytes()
                if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                    raise ValueError("Git file differs from verified snapshot")
                material, flagged = _git_file_exceptions(raw, rights.spdx_expression)
                checked = resolve_file_rights(
                    rights,
                    path,
                    b"\n".join(raw.splitlines()[:30]) + b"\n",
                    prospective_private_research=True,
                )
                if path == "LICENSE" or Path(path).name == "README.md":
                    choice = _decision(
                        "exclude", "rights context, not incident training prose"
                    )
                elif material or checked.training_eligibility not in {
                    "eligible",
                    "eligible_with_obligations",
                }:
                    choice = _decision(
                        "quarantine", "material or unknown file-level rights exception"
                    )
                elif flagged:
                    choice = _decision(
                        "quarantine", "manual third-party/notice review required"
                    )
                else:
                    choice = _decision(
                        "qualify",
                        "covered by reviewed source policy; no automated exception",
                    )
                choice.update(path=path, sha256=file["sha256"])
                decisions.append(choice)
                counts[choice["decision"]] += 1
            entry["files"] = decisions
        else:
            raise ValueError("admission draft source kind is unsupported")
        entries.append(entry)

    manifest = {
        "schema_version": 2,
        "policy_id": template["policy_id"],
        "policy_sha256": template["policy_sha256"],
        "sources": entries,
    }
    verify_record_admission(
        manifest,
        {source.id: source_declaration_payload(source) for source in project.sources},
        snapshots,
        root / "corpora" / project.config.id / "snapshots",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Canonical bytes stay identical; serialize row arrays directly from disk.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".draft-", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            encoder = json.JSONEncoder(
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            for part in encoder.iterencode(manifest):
                raw = part.encode("utf-8")
                stream.write(raw)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "schema_version": 2,
        "path": str(output),
        "sha256": sha256_file(output),
        "counts": counts,
    }


def draft_admission_manifest(
    project: Project,
    work_root: Path,
    template_path: Path,
    policy_document: Path,
    output_path: Path,
) -> dict[str, Any]:
    with metadata_operation(
        Path(work_root) / "corpora" / project.config.id / "metadata-scratch",
        max_database_bytes=None,
    ) as store:
        return _draft_admission_manifest(
            project, work_root, template_path, policy_document, output_path, store
        )
