"""Conservative, file-level rights decisions for declared corpus sources.

``nested_metadata`` maps repository-relative ``license-metadata.json`` paths
to decoded JSON documents. Supports Rust REUSE's hierarchical ``files`` root
with directory/file/group nodes, nested children, and license ``spdx`` objects,
as well as flat ``files`` lists with path(s), license and directory prefixes.
The most specific matching file/group/directory overrides ancestor metadata.
Contradictions at equal specificity require review. Metadata is never loaded
implicitly from the filesystem.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from contextlib import closing
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import field_validator, model_validator

from klab.canonical import canonical_digest
from klab.config.models import StrictModel

TrainingEligibility = Literal[
    "eligible", "eligible_with_obligations", "review_required", "ineligible"
]
RedistributionMode = Literal[
    "redistributable_under_source_terms",
    "metadata_reconstruction_only",
    "derived_only",
    "not_redistributable",
    "review_required",
]
WeightLicenseStatus = Literal["separate_analysis_required"]


class TrainingRestriction(StrictModel):
    kind: Literal["prohibited", "conditional"]
    basis: str

    @field_validator("basis")
    @classmethod
    def nonblank_basis(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("training restriction requires a nonblank basis")
        return value


class RightsPolicy(StrictModel):
    training_eligibility: TrainingEligibility
    redistribution_mode: RedistributionMode
    weight_license_status: WeightLicenseStatus = "separate_analysis_required"
    spdx_expression: str | None = None
    license_references: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    training_restriction: TrainingRestriction | None = None
    allowed_boundary_paths: tuple[str, ...] = ()
    nested_metadata_path: str | None = None

    @field_validator("nested_metadata_path")
    @classmethod
    def safe_metadata_path(cls, value: str | None) -> str | None:
        if value is not None:
            _safe_path(value)
            if PurePosixPath(value).name != "license-metadata.json":
                raise ValueError("nested_metadata_path must name license-metadata.json")
        return value

    @field_validator("spdx_expression")
    @classmethod
    def nonblank_spdx(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("SPDX expression must not be blank")
        return value

    @field_validator("license_references", "notices")
    @classmethod
    def nonblank_entries(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("rights references and notices must not be blank")
        return values

    @field_validator("allowed_boundary_paths")
    @classmethod
    def safe_allowlist(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _safe_path(value.rstrip("/"))
        return values

    @model_validator(mode="after")
    def consistent_restriction(self) -> RightsPolicy:
        if (
            self.training_restriction is not None
            and self.training_restriction.kind == "prohibited"
            and self.training_eligibility != "ineligible"
        ):
            raise ValueError("prohibited training requires ineligible policy")
        if (
            self.training_restriction is not None
            and self.training_restriction.kind == "conditional"
            and self.training_eligibility in ("eligible", "eligible_with_obligations")
        ):
            raise ValueError("conditional training cannot be declared eligible")
        return self


class FileRights(StrictModel):
    path: str
    detected_spdx_expression: str | None
    training_eligibility: TrainingEligibility
    redistribution_mode: RedistributionMode
    weight_license_status: WeightLicenseStatus
    license_references: tuple[str, ...]
    notices: tuple[str, ...]
    training_restriction: TrainingRestriction | None
    boundary_flags: tuple[str, ...]
    reason: str


_MAX_RECORD_BYTES = 8 * 1024**2


def reported_spdx_expression(file: Mapping[str, Any], source_spdx: str | None) -> str:
    """Report inherited admission SPDX without inventing one for public domain."""
    return (
        file["rights"]["detected_spdx_expression"]
        or file.get("admission_policy_spdx_expression")
        or source_spdx
        or "unknown"
    )


# The registry is intentionally finite: absence of recognition is a review, not
# an inference from a project name or from a nearly matching identifier.
_PERMISSIVE = frozenset(
    {
        "MIT",
        "MIT-0",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "ISC",
        "Zlib",
        "Unlicense",
        "PSF-2.0",
        "Python-2.0",
        "CC0-1.0",
        "BSL-1.0",
        "Artistic-2.0",
    }
)
_ATTRIBUTION = frozenset(
    {"CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0", "ODC-By-1.0"}
)
_COPYLEFT = frozenset(
    {
        "MPL-2.0",
        "GPL-2.0-only",
        "GPL-2.0",
        "GPL-2.0+",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0",
        "GPL-3.0+",
        "GPL-3.0-or-later",
        "LGPL-2.1-only",
        "LGPL-2.1",
        "LGPL-2.1+",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0",
        "LGPL-3.0+",
        "LGPL-3.0-or-later",
        "AGPL-3.0-only",
        "AGPL-3.0",
        "AGPL-3.0+",
        "AGPL-3.0-or-later",
    }
)
_RESTRICTED = frozenset(
    {
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-ND-4.0",
        "CC-BY-NC-ND-4.0",
        "CC-BY-NC-3.0",
        "CC-BY-NC-SA-3.0",
        "CC-BY-ND-3.0",
        "CC-BY-NC-ND-3.0",
    }
)
_TOKEN = re.compile(r"\(|\)|AND\b|OR\b|WITH\b|[A-Za-z0-9][A-Za-z0-9.\-+]*", re.ASCII)
_SPDX_LINE = re.compile(r"SPDX-License-Identifier\s*:\s*([^\r\n]+)", re.IGNORECASE)
_COPYRIGHT_LINE = re.compile(r"\b(?:SPDX-FileCopyrightText|Copyright\b)", re.IGNORECASE)
_BOUNDARIES = frozenset(
    {
        "vendor",
        "third_party",
        "third-party",
        "external",
        "generated",
        "bundled",
        "fixtures",
    }
)


def _safe_path(path: str) -> str:
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError(f"unsafe rights path: {path!r}")
    return path


def _license_class(expression: str) -> str:
    """Parse a bounded SPDX grammar; no best-case treatment of OR branches."""
    tokens = _TOKEN.findall(expression)
    if not tokens or "".join(tokens) != re.sub(r"\s+", "", expression):
        return "unknown"
    position = 0

    def atom() -> set[str]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("missing license")
        token = tokens[position]
        position += 1
        if token == "(":
            result = disjunction()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unclosed SPDX expression")
            position += 1
            return result
        if token in ("AND", "OR", "WITH", ")"):
            raise ValueError("missing license identifier")
        if position < len(tokens) and tokens[position] == "WITH":
            position += 1
            if position >= len(tokens):
                raise ValueError("missing SPDX exception")
            exception = tokens[position]
            position += 1
            if exception != "Linux-syscall-note" or token not in {
                "GPL-2.0",
                "GPL-2.0-only",
                "GPL-2.0+",
                "GPL-2.0-or-later",
            }:
                return {"unknown"}
        if token in _PERMISSIVE:
            return {"permissive"}
        if token in _ATTRIBUTION or token in _COPYLEFT:
            return {"obligations"}
        if token in _RESTRICTED:
            return {"restricted"}
        return {"unknown"}

    def conjunction() -> set[str]:
        nonlocal position
        result = atom()
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            result |= atom()
        return result

    def disjunction() -> set[str]:
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            result |= conjunction()
        return result

    try:
        classes = disjunction()
        if position != len(tokens):
            return "unknown"
    except ValueError:
        return "unknown"
    if "restricted" in classes:
        return "restricted"
    if "unknown" in classes:
        return "unknown"
    if "obligations" in classes:
        return "obligations"
    return "permissive"


def _metadata_expression(
    path: str, nested_metadata: Mapping[str, Any]
) -> tuple[str | None, bool]:
    matches: list[tuple[int, int, str]] = []

    def add(
        scope: str, license_id: Any, *, directory: bool, metadata_depth: int
    ) -> None:
        if isinstance(license_id, Mapping):
            license_id = license_id.get("spdx")
        if not isinstance(license_id, str) or not license_id.strip():
            raise ValueError("license metadata requires a nonblank SPDX expression")
        if path == scope or (directory and (not scope or path.startswith(scope + "/"))):
            matches.append((metadata_depth, len(scope), license_id))

    def walk(node: Any, base: str, metadata_depth: int) -> None:
        if not isinstance(node, Mapping):
            raise TypeError("license metadata nodes must be objects")
        kind = node.get("type")
        if kind == "root":
            for child in node.get("children", []):
                walk(child, base, metadata_depth)
            return
        if kind in ("directory", "file"):
            name = node.get("name")
            if not isinstance(name, str):
                raise ValueError("license metadata node requires a name")
            scope = (
                base
                if name == "."
                else "/".join(filter(None, (base, _safe_path(name))))
            )
            if "license" in node:
                add(
                    scope,
                    node["license"],
                    directory=kind == "directory",
                    metadata_depth=metadata_depth,
                )
            for child in node.get("children", []):
                walk(child, scope, metadata_depth)
            return
        if kind == "group":
            for name in node.get("files", []):
                add(
                    "/".join(filter(None, (base, _safe_path(name)))),
                    node.get("license"),
                    directory=False,
                    metadata_depth=metadata_depth,
                )
            for name in node.get("directories", []):
                add(
                    "/".join(filter(None, (base, _safe_path(name.rstrip("/"))))),
                    node.get("license"),
                    directory=True,
                    metadata_depth=metadata_depth,
                )
            return
        raise ValueError(f"unrecognized license metadata node: {kind!r}")

    for metadata_path, document in nested_metadata.items():
        _safe_path(metadata_path)
        if PurePosixPath(metadata_path).name != "license-metadata.json":
            raise ValueError("nested metadata keys must name license-metadata.json")
        if not isinstance(document, Mapping):
            raise TypeError("license metadata must be an object")
        entries = document.get("files")
        parent = PurePosixPath(metadata_path).parent
        base = "" if parent.as_posix() == "." else parent.as_posix()
        depth = len(parent.parts)
        if isinstance(entries, Mapping):
            walk(entries, base, depth)
        elif isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise TypeError("license metadata entries must be objects")
                license_id = entry.get("license", entry.get("spdx_expression"))
                paths = entry.get(
                    "paths", entry.get("files", entry.get("path", entry.get("name")))
                )
                if not isinstance(paths, (str, list, tuple)):
                    raise TypeError("license metadata entry requires path(s)")
                if isinstance(paths, str):
                    paths = [paths]
                for relative in paths:
                    if not isinstance(relative, str):
                        raise TypeError("license metadata path must be text")
                    directory = relative.endswith("/")
                    clean = _safe_path(relative.rstrip("/"))
                    add(
                        "/".join(filter(None, (base, clean))),
                        license_id,
                        directory=directory,
                        metadata_depth=depth,
                    )
        else:
            raise TypeError("license metadata requires a files root or list")
    if not matches:
        return None, False
    best = max((depth, length) for depth, length, _ in matches)
    selected = {value for depth, length, value in matches if (depth, length) == best}
    return (next(iter(selected)), False) if len(selected) == 1 else (None, True)


def resolve_file_rights(
    policy: RightsPolicy,
    path: str,
    raw_bytes: bytes,
    *,
    nested_metadata: Mapping[str, Any] | None = None,
    prospective_private_research: bool = False,
) -> FileRights:
    """Resolve pinned file evidence under the declared release training-use policy."""
    _safe_path(path)
    flags = tuple(
        sorted({part for part in path.split("/") if part.lower() in _BOUNDARIES})
    )
    allowed = any(
        path == item.rstrip("/") or (item.endswith("/") and path.startswith(item))
        for item in policy.allowed_boundary_paths
    )
    # Dataset rows are not file-level license headers. Avoid copying every line
    # of a multi-gigabyte selected shard just to inspect a handful of notices.
    header: list[bytes] = []
    if not path.endswith((".jsonl", ".json", ".parquet")):
        offset = 0
        for _ in range(30):
            end = raw_bytes.find(b"\n", offset)
            if end < 0:
                header.append(raw_bytes[offset:])
                break
            header.append(raw_bytes[offset:end])
            offset = end + 1
    expressions = set()
    file_notices: list[str] = []
    for line in header:
        text = line.decode("utf-8", errors="replace").strip()
        match = _SPDX_LINE.search(text)
        if match:
            expressions.add(
                re.sub(r"(?:\s*\*/|\s*-->)\s*$", "", match.group(1)).strip()
            )
        if _COPYRIGHT_LINE.search(text):
            file_notices.append(text)
    metadata_license, metadata_conflict = _metadata_expression(
        path, nested_metadata or {}
    )
    detected = next(iter(expressions)) if len(expressions) == 1 else None
    if metadata_license is not None:
        if detected is not None and detected != metadata_license:
            metadata_conflict = True
        detected = metadata_license if detected is None else detected
    effective = detected or policy.spdx_expression
    reasons: list[str] = []
    if flags and not allowed:
        reasons.append("external or generated path boundary: " + ", ".join(flags))
    if len(expressions) > 1 or metadata_conflict:
        reasons.append("conflicting file license evidence")
    # A recognized file-level expression supersedes a repository default.
    # Only contradictory file-vs-metadata evidence is a conflict; legitimate
    # per-file license exceptions must remain separately classifiable.
    classification = _license_class(effective) if effective is not None else "unknown"
    if classification == "unknown" and not prospective_private_research:
        reasons.append("unknown or unrecognized SPDX expression")
    elif classification == "restricted" and not prospective_private_research:
        reasons.append("restrictive SPDX license requires review")
    if policy.training_restriction is not None:
        reasons.append(
            f"{policy.training_restriction.kind} ML training: {policy.training_restriction.basis}"
        )
    if policy.training_eligibility == "ineligible" or (
        policy.training_restriction is not None
        and policy.training_restriction.kind == "prohibited"
    ):
        training: TrainingEligibility = "ineligible"
    elif policy.training_eligibility == "review_required" or reasons:
        training = "review_required"
    elif (
        policy.training_eligibility == "eligible_with_obligations"
        or classification == "obligations"
        or (
            prospective_private_research and classification in {"unknown", "restricted"}
        )
        or (classification == "permissive" and not policy.notices)
    ):
        training = "eligible_with_obligations"
    else:
        training = "eligible"
    redistribution = policy.redistribution_mode
    if reasons and redistribution == "redistributable_under_source_terms":
        redistribution = "review_required"
    if not reasons:
        reasons.append(
            "private research candidate; license obligations and explicit source terms retained"
            if prospective_private_research
            else "recognized SPDX under declared source terms"
        )
    return FileRights(
        path=path,
        detected_spdx_expression=detected,
        training_eligibility=training,
        redistribution_mode=redistribution,
        weight_license_status=policy.weight_license_status,
        license_references=policy.license_references,
        notices=(*policy.notices, *file_notices),
        training_restriction=policy.training_restriction,
        boundary_flags=flags,
        reason="; ".join(reasons),
    )


def _admission_profile(source: Mapping[str, Any]) -> str:
    """Select the legacy or explicitly declared record admission policy."""
    from klab.corpus.row_qualification import LEGACY_HF_ROW_URIS

    if (
        source.get("canonical_uri") in LEGACY_HF_ROW_URIS
        and source.get("acquisition", {}).get("qualification") is not None
    ):
        raise ValueError("legacy HF row policies do not accept qualification overrides")
    uri = source["canonical_uri"]
    if uri == "https://huggingface.co/datasets/common-pile/project_gutenberg_filtered":
        return "gutenberg"
    if uri == "https://huggingface.co/datasets/common-pile/wikimedia_filtered":
        return "wikimedia"
    if source.get("acquisition", {}).get("qualification") is not None:
        return "declared_rows"
    raise ValueError("record admission source is outside supported policy scope")


def _admission_exceptions(
    row: Mapping[str, Any],
    profile: str,
    source: Mapping[str, Any] | None = None,
    *,
    row_use=None,
) -> tuple[str | None, tuple[str, ...]]:
    """Return a material conflict or review flags from the original retained row."""
    if profile == "declared_rows":
        from klab.corpus.row_qualification import RowQualification, exceptions

        if source is None:
            raise ValueError("declared row policy requires source declaration")
        spec = source["acquisition"]
        return exceptions(
            dict(row),
            RowQualification.model_validate(spec["qualification"]),
            spec["text_field"],
            row_use=row_use,
        )
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return "missing source metadata", ()
    if not isinstance(metadata.get("title"), str) or not metadata["title"].strip():
        return "missing source title", ()
    language = metadata.get("language")
    if language is not None and language != "en":
        return "record outside English source policy", ()
    url = metadata.get("url")
    if not isinstance(url, str):
        return "missing source locator", ()
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return "invalid source locator", ()
    if parsed.scheme != "https" or not host:
        return "invalid source locator", ()
    license_claim = metadata.get("license")
    if not isinstance(license_claim, str):
        return "missing source license claim", ()
    if profile == "gutenberg":
        if host not in {"www.gutenberg.org", "gutenberg.org"} or not re.fullmatch(
            r"/ebooks/\d+(?:\.txt(?:\.utf-8)?)?", parsed.path
        ):
            return "source locator outside Gutenberg work scope", ()
        if parsed.path.removeprefix("/ebooks/").split(".", 1)[0] != row.get("id"):
            return "Gutenberg book ID differs from source locator", ()
        if license_claim.strip().casefold() != "public domain":
            return "Gutenberg source license conflicts with policy", ()
    else:
        if host not in {
            "wikipedia.com",
            "www.wikipedia.com",
            "wikipedia.org",
            "en.wikipedia.org",
        } or not parsed.path.startswith("/wiki/"):
            return "source locator outside Wikimedia page scope", ()
        if str(metadata.get("namespace")) != "0":
            return "Wikimedia record outside content namespace", ()
        if (
            unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
            != metadata["title"]
        ):
            return "Wikimedia title differs from page locator", ()
        if "creativecommons.org/licenses/by-sa/4.0/" not in license_claim.casefold():
            return "Wikimedia source license conflicts with policy", ()
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return "empty or nontext source row", ()
    if _ADMISSION_PRIVATE.search(text):
        return "private identifier or credential in source row", ()
    flags: set[str] = set()
    if profile == "gutenberg" and re.search(
        r"(?i)\b(?:by permission of|printed from.{0,60}by permission|"
        r"permission to (?:reprint|reproduce|publish)|copyright notice|"
        r"all rights reserved)\b",
        text,
    ):
        flags.add("gutenberg_rights_notice_context")
    if profile == "wikimedia":
        if re.search(r"(?i)\bpersonal life\b", text) and re.search(
            r"(?i)\bsuicide\b", text
        ):
            return "sensitive biography and self-harm detail", ()
        if re.search(r"(?i)\b(?:personal life|family life)\b", text):
            flags.add("biography_privacy_context")
        if re.search(
            r"(?i)\b(?:description in seitz|imported text|copied from|public domain text)\b",
            text,
        ):
            flags.add("imported_text_context")
    return None, tuple(sorted(flags))


_ADMISSION_PRIVATE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|\bAKIA[A-Z0-9]{16}\b"
    r"|\b(?:ghp|gho|github_pat)_[A-Za-z0-9_]{30,}\b"
    r"|\b\d{3}-\d{2}-\d{4}\b",
    re.IGNORECASE,
)


def _verify_record_admission_v1(
    manifest: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    snapshot_root: Path,
    *,
    _visit=None,
    _max_record_bytes: int | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Recheck a complete admission inventory against immutable selected HF rows.

    The manifest is a release-level review of already acquired rows. It never
    changes the source declaration or its acquisition receipt.
    """
    if (
        set(manifest) != {"schema_version", "policy_id", "policy_sha256", "sources"}
        or manifest["schema_version"] != 1
        or not isinstance(manifest["policy_id"], str)
        or not manifest["policy_id"].strip()
        or not isinstance(manifest["policy_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest["policy_sha256"])
        or not isinstance(manifest["sources"], list)
    ):
        raise ValueError("invalid versioned record admission manifest")
    rows = manifest["sources"]
    if {item.get("source_id") for item in rows if isinstance(item, dict)} != set(
        sources
    ) or len(rows) != len(sources):
        raise ValueError("record admission source inventory mismatch")
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) - {"row_use_policy"} != {
            "source_id",
            "snapshot_sha256",
            "source_revision",
            "sample_path",
            "sample_sha256",
            "source_shard_path",
            "source_shard_sha256",
            "license_label",
            "rights",
            "records",
        }:
            raise ValueError("invalid record admission source entry")
        source_id = item["source_id"]
        source = sources[source_id]
        snapshot = snapshots[source_id]
        if source["kind"] != "huggingface_dataset" or source["schema_version"] != 2:
            raise ValueError("record admission requires v2 bounded HF source")
        profile = _admission_profile(source)
        from klab.corpus.row_qualification import (
            RowQualification,
            RowUsePolicy,
            use_evidence,
        )

        row_use = (
            RowUsePolicy.model_validate(item["row_use_policy"])
            if "row_use_policy" in item
            else None
        )
        if row_use is not None and profile != "declared_rows":
            raise ValueError("row use policy requires declared row qualification")
        if (
            item["snapshot_sha256"] != snapshot["snapshot_sha256"]
            or item["source_revision"] != source["revision"]
            or snapshot["declaration"]["id"] != source_id
            or snapshot["declaration"]["revision"] != source["revision"]
            or not isinstance(item["license_label"], str)
            or not item["license_label"].strip()
        ):
            raise ValueError("record admission source provenance mismatch")
        if profile == "gutenberg" and "public domain" not in item[
            "license_label"
        ].casefold().replace("-", " "):
            raise ValueError("Gutenberg admission lacks public-domain basis")
        if (
            profile == "wikimedia"
            and "cc-by-sa-4.0" not in item["license_label"].casefold()
        ):
            raise ValueError("Wikimedia admission lacks license basis")
        retrieval = snapshot["retrieval"]
        shards = retrieval.get("shards", [])
        if len(shards) != 1 or len(snapshot["files"]) != 1:
            raise ValueError("record admission requires one complete selected shard")
        shard = shards[0]
        file = snapshot["files"][0]
        if any(
            item[key] != expected
            for key, expected in (
                ("sample_path", file["path"]),
                ("sample_path", shard["output_path"]),
                ("sample_sha256", file["sha256"]),
                ("source_shard_path", shard["source_shard_path"]),
                ("source_shard_sha256", shard["source_shard_sha256"]),
            )
        ):
            raise ValueError("record admission shard provenance mismatch")
        policy = RightsPolicy.model_validate(item["rights"])
        if (
            policy.training_eligibility not in {"eligible", "eligible_with_obligations"}
            or policy.redistribution_mode != "metadata_reconstruction_only"
            or not policy.license_references
            or profile == "gutenberg"
            and policy.spdx_expression is not None
            or profile == "wikimedia"
            and policy.spdx_expression != "CC-BY-SA-4.0"
        ):
            raise ValueError("record admission policy is outside local research scope")
        file_rights = resolve_file_rights(
            policy, file["path"], b"", prospective_private_research=True
        )
        if file_rights.training_eligibility not in {
            "eligible",
            "eligible_with_obligations",
        }:
            raise ValueError("record admission policy does not qualify its shard")
        decisions = item["records"]
        if not isinstance(decisions, list):
            raise TypeError("record admission decisions must be a list")
        from klab.corpus.bounded_metadata import decision_index

        indexed = decision_index()
        for decision in decisions:
            if (
                not isinstance(decision, dict)
                or not set(decision).issubset(
                    {
                        "source_row_index",
                        "source_row_sha256",
                        "decision",
                        "reason",
                        "issues",
                        "manual_review",
                        "row_evidence",
                    }
                )
                or not {
                    "source_row_index",
                    "source_row_sha256",
                    "decision",
                    "reason",
                    "issues",
                }.issubset(decision)
            ):
                raise ValueError("invalid record admission decision")
            index = decision["source_row_index"]
            if type(index) is not int or index < 0 or index in indexed:
                raise ValueError("duplicate or invalid record admission row index")
            if (
                not isinstance(decision["source_row_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", decision["source_row_sha256"])
                or decision["decision"] not in {"qualify", "exclude", "quarantine"}
                or not isinstance(decision["reason"], str)
                or not decision["reason"].strip()
                or not isinstance(decision["issues"], list)
                or any(
                    not isinstance(issue, dict)
                    or set(issue) != {"field", "owner", "remedy", "decision_impact"}
                    or any(
                        not isinstance(value, str) or not value.strip()
                        for value in issue.values()
                    )
                    for issue in decision["issues"]
                )
            ):
                raise ValueError("invalid record admission decision value")
            manual = decision.get("manual_review")
            if manual is not None and (
                not isinstance(manual, dict)
                or set(manual) != {"reviewer", "reviewed_on", "evidence_url", "finding"}
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in manual.values()
                )
                or not manual["evidence_url"].startswith("https://")
            ):
                raise ValueError("invalid record admission manual review")
            indexed[index] = decision
        selected = shard["selected_rows"]
        if len(indexed) != len(selected):
            raise ValueError("record admission decisions do not cover selected rows")
        selected_indices = decision_index()
        for receipt in selected:
            index = receipt["source_row_index"]
            if index not in indexed or index in selected_indices:
                raise ValueError(
                    "record admission decisions do not cover selected rows"
                )
            selected_indices[index] = None
        sample = (
            snapshot_root
            / source_id
            / snapshot["snapshot_sha256"]
            / "files"
            / file["path"]
        )
        if sample.is_symlink():
            raise ValueError("record admission sample bytes differ from snapshot")
        from klab.corpus.jsonl_records import records_from_path

        # Keep source text bounded to the current record. Decisions and acquisition
        # receipts use disk indexes inside a metadata operation; other callers
        # retain ordinary mappings. Close the source even when validation fails.
        with closing(
            records_from_path(
                sample,
                max_record_bytes=(
                    _MAX_RECORD_BYTES
                    if _max_record_bytes is None
                    else min(_MAX_RECORD_BYTES, _max_record_bytes)
                ),
                expected_sha256=file["sha256"],
            )
        ) as sample_rows:
            for ordinal, (record, receipt) in enumerate(
                zip_longest(sample_rows, selected)
            ):
                if record is None or receipt is None:
                    raise ValueError("record admission sample row count mismatch")
                row = record.value
                if not isinstance(row, dict) or not isinstance(
                    row.get("_klab_source"), dict
                ):
                    raise TypeError("record admission row lacks acquired provenance")
                envelope = row["_klab_source"]
                original = {
                    key: value for key, value in row.items() if key != "_klab_source"
                }
                row_hash = canonical_digest(original)
                index = envelope.get("source_row_index")
                decision = indexed.get(index)
                metadata = original.get("metadata")
                provenance = (
                    metadata.get("provenance") if isinstance(metadata, dict) else None
                )
                if (
                    type(index) is not int
                    or index != receipt["source_row_index"]
                    or row_hash != receipt["source_row_sha256"]
                    or row_hash != envelope.get("source_row_sha256")
                    or decision is None
                    or row_hash != decision["source_row_sha256"]
                    or envelope.get("dataset_revision") != source["revision"]
                    or envelope.get("source_shard_path") != item["source_shard_path"]
                    or envelope.get("source_shard_sha256")
                    != item["source_shard_sha256"]
                    or (
                        profile != "declared_rows"
                        and (
                            not isinstance(original.get("id"), str)
                            or original["id"] != envelope.get("id")
                            or not isinstance(provenance, str)
                            or re.fullmatch(
                                re.escape(item["source_shard_path"]) + r":\d+",
                                provenance,
                            )
                            is None
                        )
                    )
                ):
                    raise ValueError("record admission row provenance mismatch")
                conflict, flags = _admission_exceptions(
                    original, profile, source, row_use=row_use
                )
                if row_use is not None:
                    expected = use_evidence(
                        original,
                        RowQualification.model_validate(
                            source["acquisition"]["qualification"]
                        ),
                        source["acquisition"]["text_field"],
                        row_use,
                    )
                    if decision.get("row_evidence") != expected:
                        raise ValueError(
                            "record attribution evidence differs from original row"
                        )
                elif "row_evidence" in decision:
                    raise ValueError("row evidence requires a bound row use policy")
                if decision["decision"] == "qualify":
                    if conflict:
                        raise ValueError(
                            f"qualified record has material exception: {conflict}"
                        )
                    if flags and "manual_review" not in decision:
                        raise ValueError("qualified flagged record lacks manual review")
                elif (
                    conflict
                    and decision["decision"] == "exclude"
                    and conflict
                    not in {
                        "Wikimedia record outside content namespace",
                        "explicit training-use restriction on source row",
                    }
                ):
                    raise ValueError("material rights exception must be quarantined")
                if _visit is not None:
                    _visit(source_id, file["path"], decision, record, ordinal)
        if row_use is not None:
            unmatched = set(row_use.restricted_rows)
            for receipt in selected:
                unmatched.discard(receipt["source_row_sha256"])
            if unmatched:
                raise ValueError("row restriction is outside acquired row inventory")
        resolved[source_id, file["path"]] = {
            "rights": file_rights,
            "license_label": item["license_label"],
            "policy_spdx_expression": policy.spdx_expression,
            "decisions": indexed,
            "policy_id": manifest["policy_id"],
            "policy_sha256": manifest["policy_sha256"],
        }
    return resolved


_GIT_PRIVATE = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    rb"|\bAKIA[A-Z0-9]{16}\b"
    rb"|\b(?:ghp|gho|github_pat)_[A-Za-z0-9_]{30,}\b"
    rb"|\b\d{3}-\d{2}-\d{4}\b"
    rb"|\b(?:password|api[_-]?key|secret[_-]?key)\s*[:=]\s*"
    rb"['\"]?[A-Za-z0-9/+_=]{24,}",
    re.IGNORECASE,
)
_GIT_THIRD_PARTY = re.compile(
    rb"\b(?:third[- ]party|copied from|reproduced with permission|"
    rb"all rights reserved|fair use|copyright|courtesy of)\b",
    re.IGNORECASE,
)


def _git_file_exceptions(raw: bytes, policy_spdx: str) -> tuple[bool, bool]:
    """Screen the complete pinned file; return material conflict and review flag."""
    expressions = {
        re.sub(r"(?:\s*\*/|\s*-->)\s*$", "", match.group(1)).strip()
        for line in raw.splitlines()
        if (match := _SPDX_LINE.search(line.decode("utf-8", errors="replace")))
    }
    conflict = bool(_GIT_PRIVATE.search(raw)) or bool(expressions - {policy_spdx})
    flag = bool(_GIT_THIRD_PARTY.search(raw))
    return conflict, flag


def verify_record_admission(
    manifest: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    snapshot_root: Path,
    *,
    _visit=None,
    _max_record_bytes: int | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Verify reviewed HF rows and, in v2, each pinned Git file.

    The v1 path is left intact for historical manifests and release identities.
    A v2 manifest covers every project source; no unreviewed Git file can fall
    through to a permissive source default.
    """
    if manifest.get("schema_version") == 1:
        return _verify_record_admission_v1(
            manifest,
            sources,
            snapshots,
            snapshot_root,
            _visit=_visit,
            _max_record_bytes=_max_record_bytes,
        )
    if (
        set(manifest) != {"schema_version", "policy_id", "policy_sha256", "sources"}
        or manifest.get("schema_version") != 2
        or not isinstance(manifest.get("policy_id"), str)
        or not manifest["policy_id"].strip()
        or not isinstance(manifest.get("policy_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["policy_sha256"]) is None
        or not isinstance(manifest.get("sources"), list)
        or set(sources) != set(snapshots)
    ):
        raise ValueError("invalid versioned record admission manifest")
    entries = manifest["sources"]
    if (
        len(entries) != len(sources)
        or any(not isinstance(item, dict) for item in entries)
        or {item.get("source_id") for item in entries} != set(sources)
    ):
        raise ValueError("record admission source inventory mismatch")
    hf_entries = [
        item
        for item in entries
        if sources[item["source_id"]]["kind"] == "huggingface_dataset"
    ]
    git_entries = [
        item for item in entries if sources[item["source_id"]]["kind"] == "git"
    ]
    if len(hf_entries) + len(git_entries) != len(entries):
        raise ValueError("record admission source kind is unsupported")
    hf_ids = {item["source_id"] for item in hf_entries}
    resolved = _verify_record_admission_v1(
        {**manifest, "schema_version": 1, "sources": hf_entries},
        {key: sources[key] for key in hf_ids},
        {key: snapshots[key] for key in hf_ids},
        snapshot_root,
        _visit=_visit,
        _max_record_bytes=_max_record_bytes,
    )
    for item in git_entries:
        if set(item) != {
            "source_id",
            "snapshot_sha256",
            "source_revision",
            "license_label",
            "rights",
            "files",
        }:
            raise ValueError("invalid Git file admission source entry")
        source_id = item["source_id"]
        source = sources[source_id]
        snapshot = snapshots[source_id]
        if (
            source["schema_version"] != 2
            or item["snapshot_sha256"] != snapshot["snapshot_sha256"]
            or item["source_revision"] != source["revision"]
            or snapshot["declaration"]["id"] != source_id
            or snapshot["declaration"]["revision"] != source["revision"]
            or not isinstance(item["license_label"], str)
            or not item["license_label"].strip()
            or not isinstance(item["files"], list)
        ):
            raise ValueError("Git file admission source provenance mismatch")
        policy = RightsPolicy.model_validate(item["rights"])
        if (
            policy.training_eligibility not in {"eligible", "eligible_with_obligations"}
            or policy.redistribution_mode != "metadata_reconstruction_only"
            or not policy.spdx_expression
            or not policy.license_references
            or policy.training_restriction is not None
            or source["license_url"] not in policy.license_references
            or policy.spdx_expression not in item["license_label"]
        ):
            raise ValueError(
                "Git file admission policy is outside local research scope"
            )
        files = {file["path"]: file for file in snapshot["files"]}
        decisions = item["files"]
        if (
            len(files) != len(snapshot["files"])
            or len(decisions) != len(files)
            or any(not isinstance(decision, dict) for decision in decisions)
            or {decision.get("path") for decision in decisions} != set(files)
        ):
            raise ValueError("Git file admission decisions do not cover pinned files")
        for ordinal, decision in enumerate(decisions):
            if (
                not set(decision).issubset(
                    {"path", "sha256", "decision", "reason", "issues", "manual_review"}
                )
                or not {"path", "sha256", "decision", "reason", "issues"}.issubset(
                    decision
                )
                or decision["decision"] not in {"qualify", "exclude", "quarantine"}
                or not isinstance(decision["reason"], str)
                or not decision["reason"].strip()
                or not isinstance(decision["issues"], list)
                or any(
                    not isinstance(issue, dict)
                    or set(issue) != {"field", "owner", "remedy", "decision_impact"}
                    or any(
                        not isinstance(value, str) or not value.strip()
                        for value in issue.values()
                    )
                    for issue in decision["issues"]
                )
            ):
                raise ValueError("invalid Git file admission decision")
            path = decision["path"]
            file = files[path]
            if decision["sha256"] != file["sha256"]:
                raise ValueError("Git file admission SHA-256 mismatch")
            sample = (
                snapshot_root / source_id / snapshot["snapshot_sha256"] / "files" / path
            )
            if sample.is_symlink():
                raise ValueError("Git file admission bytes differ from snapshot")
            raw = sample.read_bytes()
            if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                raise ValueError("Git file admission bytes differ from snapshot")
            header = b"\n".join(raw.splitlines()[:30]) + b"\n"
            material, flagged = _git_file_exceptions(raw, policy.spdx_expression)
            checked = resolve_file_rights(
                policy, path, header, prospective_private_research=True
            )
            manual = decision.get("manual_review")
            if manual is not None and (
                not isinstance(manual, dict)
                or set(manual) != {"reviewer", "reviewed_on", "evidence_url", "finding"}
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in manual.values()
                )
                or not manual["evidence_url"].startswith("https://")
            ):
                raise ValueError("invalid Git file admission manual review")
            if decision["decision"] == "qualify":
                if (
                    checked.training_eligibility
                    not in {"eligible", "eligible_with_obligations"}
                    or checked.boundary_flags
                    or checked.detected_spdx_expression
                    not in {None, policy.spdx_expression}
                    or material
                ):
                    raise ValueError("qualified Git file has material rights exception")
                if flagged and manual is None:
                    raise ValueError("qualified flagged Git file lacks manual review")
                file_rights = checked
            else:
                source_policy = RightsPolicy.model_validate(source["rights"])
                file_rights = resolve_file_rights(
                    source_policy, path, header, prospective_private_research=True
                )
                if file_rights.training_eligibility in {
                    "eligible",
                    "eligible_with_obligations",
                }:
                    raise ValueError("excluded Git file unexpectedly eligible")
            resolved[source_id, path] = {
                "rights": file_rights,
                "license_label": item["license_label"],
                "policy_spdx_expression": policy.spdx_expression,
                "decision": decision["decision"],
                "policy_id": manifest["policy_id"],
                "policy_sha256": manifest["policy_sha256"],
            }
            if _visit is not None:
                _visit(source_id, path, decision, raw, ordinal)
    return resolved
