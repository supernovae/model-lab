"""Declared HF row screening and family provenance; never grants review authority."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel

LEGACY_HF_ROW_URIS = {
    "https://huggingface.co/datasets/common-pile/project_gutenberg_filtered",
    "https://huggingface.co/datasets/common-pile/wikimedia_filtered",
}

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}


class RowQualification(StrictModel):
    profile: Literal["web_text", "synthetic_text", "code"]
    family_fields: tuple[str, ...] = Field(min_length=1, max_length=8)
    language_field: str | None = None
    allowed_languages: tuple[str, ...] = ()
    license_field: str | None = None
    license_labels: dict[str, str] = Field(default_factory=dict)
    repository_field: str | None = None
    path_field: str | None = None
    synthetic_provenance_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def complete(self):
        names = (
            *self.family_fields,
            *self.synthetic_provenance_fields,
            self.language_field,
            self.license_field,
            self.repository_field,
            self.path_field,
        )
        if any(
            name is not None
            and (
                len(name) > 512
                or len(name.split(".")) > 8
                or any(
                    not part
                    or part != part.strip()
                    or len(part) > 128
                    or part.startswith("_klab")
                    for part in name.split(".")
                )
            )
            for name in names
        ):
            raise ValueError("qualification requires bounded original row field names")
        if len(set(self.family_fields)) != len(self.family_fields):
            raise ValueError("duplicate qualification family field")
        if bool(self.language_field) != bool(self.allowed_languages) or any(
            not value.strip() for value in self.allowed_languages
        ):
            raise ValueError(
                "language field and explicit allowed values are required together"
            )
        if self.profile != "code" and any(
            (
                self.license_field,
                self.license_labels,
                self.repository_field,
                self.path_field,
            )
        ):
            raise ValueError("code-only qualification fields require the code profile")
        if self.profile != "synthetic_text" and self.synthetic_provenance_fields:
            raise ValueError(
                "synthetic provenance fields require the synthetic_text profile"
            )
        if self.profile == "synthetic_text" and not self.synthetic_provenance_fields:
            raise ValueError("synthetic rows require original provenance fields")
        if self.profile == "code" and (
            not all(
                (
                    self.language_field,
                    self.license_field,
                    self.repository_field,
                    self.path_field,
                )
            )
            or not self.license_labels
            or any(
                not key.strip() or value not in PERMISSIVE
                for key, value in self.license_labels.items()
            )
            or self.family_fields != (self.repository_field,)
        ):
            raise ValueError(
                "code qualification requires language, permissive license mapping, path and repository family"
            )
        return self


class RowUsePolicy(StrictModel):
    """Prospective application policy; never mutates acquired source declarations."""

    mode: Literal["record_attribution"]
    restricted_rows: dict[str, dict[str, str]] = Field(
        default_factory=dict, max_length=100000
    )

    @model_validator(mode="after")
    def explicit_restrictions(self):
        import re

        for digest, evidence in self.restricted_rows.items():
            if (
                re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or set(evidence) != {"basis", "evidence"}
                or any(
                    not value.strip() or len(value) > 4096
                    for value in evidence.values()
                )
            ):
                raise ValueError(
                    "row restriction requires exact row digest, basis and evidence"
                )
        return self


def use_evidence(
    row: dict, qualification: RowQualification, text_field: str, policy: RowUsePolicy
) -> dict:
    """Retain labels and notice pointers; do not classify licenses or infer permission."""
    digest = canonical_digest(row)
    label = field_value(row, qualification.license_field)
    text = row.get(text_field)
    # Iterate LF physical lines without materializing a split list. Extracted
    # notices are bounded pointers; the original row remains authoritative.
    from io import StringIO

    notices = []
    truncated = False
    if isinstance(text, str):
        for number, line in enumerate(StringIO(text), 1):
            line = line.removesuffix("\n")
            if (
                "copyright" not in line.casefold()
                and "spdx-license-identifier:" not in line.casefold()
            ):
                continue
            if len(notices) == 64:
                truncated = True
                break
            notices.append({"line": number, "text": line[:1024]})
            truncated |= len(line) > 1024
    return {
        "source_row_sha256": digest,
        "license_label": label,
        "license_label_status": "recorded" if _present(label) else "unknown",
        "notice_lines": notices,
        "notice_lines_truncated": truncated,
        "restriction": policy.restricted_rows.get(digest),
        "scope": "declared source policy and explicit row restrictions; not legal clearance",
    }


def _present(value: object) -> bool:
    return (
        value is not None
        and (not isinstance(value, str) or bool(value.strip()))
        and value != []
        and value != {}
    )


def field_value(row: dict, path: str | None):
    """Resolve an explicit dotted path through dictionaries, never arrays."""
    if path is None:
        return None
    value = row
    for segment in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(segment)
    return value


def family_hint(row: dict, policy: RowQualification) -> str:
    values = [field_value(row, field) for field in policy.family_fields]
    if policy.profile == "code" and isinstance(values[0], str):
        values[0] = values[0].casefold()
    if any(not _present(value) for value in values):
        raise ValueError("missing declared family provenance")
    return policy.profile + "-family:" + canonical_digest(values)


def provenance(row: dict, policy: RowQualification) -> dict:
    result = {"profile": policy.profile, "family_hint": family_hint(row, policy)}
    if policy.profile == "synthetic_text":
        values = {
            key: field_value(row, key) for key in policy.synthetic_provenance_fields
        }
        if any(not _present(value) for value in values.values()):
            raise ValueError("missing synthetic provenance")
        result["synthetic_provenance_sha256"] = hashlib.sha256(
            canonical_json(values)
        ).hexdigest()
        result["synthetic_provenance_fields"] = list(values)
    if policy.profile == "code":
        result["attribution"] = {
            "repository": field_value(row, policy.repository_field),
            "path": field_value(row, policy.path_field),
            "license_label": field_value(row, policy.license_field),
            "language": field_value(row, policy.language_field),
            "license_expression": policy.license_labels.get(
                field_value(row, policy.license_field)
            )
            if isinstance(field_value(row, policy.license_field), str)
            else None,
        }
    return result


def exceptions(
    row: dict,
    policy: RowQualification,
    text_field: str,
    *,
    row_use: RowUsePolicy | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    from klab.corpus.rights import (
        _ADMISSION_PRIVATE,
        _GIT_PRIVATE,
        _git_file_exceptions,
    )

    text = row.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return "empty or nontext declared content field", ()
    if _ADMISSION_PRIVATE.search(text) or _GIT_PRIVATE.search(text.encode("utf-8")):
        return "private identifier or credential in source row", ()
    try:
        provenance(row, policy)
    except TypeError, ValueError:
        return "missing or invalid declared family/synthetic provenance", ()
    if (
        policy.language_field
        and field_value(row, policy.language_field) not in policy.allowed_languages
    ):
        return "record outside declared language policy", ()
    if row_use is not None and canonical_digest(row) in row_use.restricted_rows:
        return "explicit training-use restriction on source row", ()
    if policy.profile != "code":
        return None, ()
    license_value = field_value(row, policy.license_field)
    expression = (
        policy.license_labels.get(license_value)
        if isinstance(license_value, str)
        else None
    )
    if expression is None and row_use is None:
        return "missing or incompatible repository license label", ()
    repository = field_value(row, policy.repository_field)
    path = field_value(row, policy.path_field)
    if (
        not isinstance(repository, str)
        or not repository.strip()
        or not isinstance(path, str)
    ):
        return "missing code repository or path", ()
    parts = PurePosixPath(path).parts
    if (
        not parts
        or path.startswith("/")
        or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        return "unsafe code path", ()
    if any(
        part.casefold()
        in {
            "vendor",
            "vendored",
            "node_modules",
            "third_party",
            "third-party",
            "generated",
            "dist",
            "build",
        }
        for part in parts
    ):
        return "vendor or generated code path", ()
    material, flags = _git_file_exceptions(text.encode("utf-8"), expression)
    if material and row_use is None:
        return "material or conflicting file-level code notice", ()
    if "generated" in text[:4096].casefold() and any(
        marker in text[:4096].casefold()
        for marker in ("do not edit", "automatically generated", "auto-generated")
    ):
        return "generated code notice", ()
    return None, ("code_notice_review",) if flags and row_use is None else ()
