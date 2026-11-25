"""Deterministic source document and evidence normalization."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.corpus.project import ROW_SOURCE_KINDS, NormalizerOptions, source_language
from klab.corpus.rights import FileRights

NORMALIZATION_IMPLEMENTATION_FILES = (
    "document_normalization.py",
    "jsonl_records.py",
    "project.py",
    "rights.py",
    "row_qualification.py",
    "structure_cleaning.py",
)


class EmptyDocumentError(ValueError):
    """An empty or non-text passage under `documents.empty_text: error`.

    The policy says such a passage fails the build, so builders must not turn it
    into the quiet rejection of the file that contains it.
    """


class NormalizedDocument(StrictModel):
    """Versioned document retaining source attribution and raw evidence identity."""

    document_id: str
    source_id: str
    modality: Literal["text"]
    source_revision: str
    source_location: str
    license: str
    redistribution: str
    domains: list[str]
    document_kind: str
    title: str
    section_path: list[str]
    language: str
    text: str
    content_sha256: str
    raw_content_sha256: str
    source_family: str
    file_sha256: str | None = None
    rights: dict[str, Any] | None = None
    metadata: dict[str, str | int | float | bool | None] | None = None
    structure: dict[str, Any] | None = None
    split: str | None = None
    representative_id: str | None = None
    drop_reason: str | None = None


HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _cnxml_passages(raw: bytes) -> list[tuple[str, list[str], int, int]]:
    """Extract prose; accept predefined XML escapes, never expand custom entities."""
    if re.search(
        rb"<!\s*(?:DOCTYPE|ENTITY)\b|&(?!amp;|lt;|gt;|quot;|apos;)",
        raw,
        re.IGNORECASE,
    ):
        raise ValueError("CNXML custom entities and DTDs are not permitted")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("invalid CNXML") from exc
    excluded = {"media", "image", "video", "audio", "figure", "download"}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] in excluded and any(
            value.startswith(("http:", "https:", "//"))
            for key, value in element.attrib.items()
            if key in {"src", "href", "url"}
        ):
            raise ValueError("CNXML references external media")
    content = root.find(".//{*}content")
    if content is None:
        raise ValueError("CNXML content missing")
    blocks = {"para", "title", "code", "item"}
    parts: list[str] = []

    def visible_text(element: ET.Element) -> str:
        pieces = [element.text or ""]
        for child in element:
            if child.tag.rsplit("}", 1)[-1] not in excluded:
                pieces.append(visible_text(child))
            pieces.append(child.tail or "")
        return "".join(pieces)

    def visit(element: ET.Element) -> None:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in excluded:
            return
        if tag in blocks:
            value = normalize_text(" ".join(visible_text(element).split()))
            if value:
                parts.append(value)
            return
        for child in element:
            visit(child)

    visit(content)
    if not parts:
        raise ValueError("CNXML has no prose")
    return [("\n".join(parts), [], 1, raw.count(b"\n") + int(not raw.endswith(b"\n")))]


def markdown_sections(
    text: str, markdown: bool
) -> list[tuple[str, list[str], int, int]]:
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    if not markdown:
        return [(text, [], 1, max(1, len(lines)))]
    starts = [(0, [])]
    ancestry: list[str] = []
    fenced = False
    fence_char = ""
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        fence = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            if not fenced:
                fenced, fence_char = True, fence.group(1)[0]
            elif fence.group(1)[0] == fence_char:
                fenced = False
        if fenced or fence:
            continue
        match = HEADING.match(line.rstrip("\r\n"))
        if match:
            level = len(match.group(1))
            ancestry = ancestry[: level - 1] + [match.group(2).strip()]
            if index == 0:
                starts[0] = (0, ancestry.copy())
            else:
                starts.append((index, ancestry.copy()))
    result = []
    for position, (start, path) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        passage = "".join(lines[start:end])
        if passage.strip() and not (
            markdown
            and (
                (path and path[-1] == '{{% heading "whatsnext" %}}')
                or all(
                    not line.strip()
                    or HEADING.fullmatch(line.strip())
                    or line.strip().startswith(("{{", "<!--"))
                    for line in passage.splitlines()
                )
            )
        ):
            result.append((passage, path, start + 1, end))
    return result


_SECRET_OR_PRIVATE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|\bAKIA[A-Z0-9]{16}\b"
    r"|\b(?:ghp|gho|github_pat)_[A-Za-z0-9_]{30,}\b"
    r"|\b\d{3}-\d{2}-\d{4}\b"
    r"|\b(?:password|api[_-]?key|secret[_-]?key)\s*[:=]\s*"
    r"['\"]?[A-Za-z0-9/+_=]{24,}",
    re.IGNORECASE,
)
_EXCLUDED_WEB_HOSTS = frozenset(
    {"stackoverflow.com", "stackexchange.com", "reddit.com", "pastebin.com"}
)


def _excluded_research_record(
    content: str, metadata: dict[str, Any], source_id: str
) -> str | None:
    """Drop identifiable confidential material and disallowed platform mirrors."""
    if _SECRET_OR_PRIVATE.search(content):
        return "secret_or_private_identifier"
    url = metadata.get("url")
    if isinstance(url, str):
        try:
            host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
        except ValueError:
            return "unknown_page_provenance"
        if any(
            host == banned or host.endswith("." + banned)
            for banned in _EXCLUDED_WEB_HOSTS
        ):
            return "excluded_platform_terms_or_private_paste"
        if not host:
            return "unknown_page_provenance"
    elif source_id.startswith(("fineweb", "openwebmath")):
        return "unknown_page_provenance"
    if source_id.startswith("pes2o") and metadata.get("source") not in (
        "s2orc",
        "s2orc/train",
        "s2ag/train",
    ):
        return "unreviewed_academic_origin"
    if len(content.strip()) < 100 and source_id.startswith(
        ("fineweb", "pes2o", "openwebmath")
    ):
        return "insufficient_content"
    return None


def records_for_file(
    raw: bytes,
    name: str,
    source: Any,
    snapshot_id: str,
    *,
    file_rights: FileRights | None = None,
    rejected_records: list[dict[str, Any]] | None = None,
    full_file_sha256: str | None = None,
    first_row_index: int = 1,
    record_admission: dict[str, Any] | None = None,
    normalizer_options: NormalizerOptions,
    document_policy: Any,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    suffix = Path(name).suffix.lower()
    streaming_rows = (
        document_policy.row_files == "streaming"
        and suffix == ".jsonl"
        and source.kind in ROW_SOURCE_KINDS
    )
    jsonl_rows = suffix == ".jsonl" and source.kind in ROW_SOURCE_KINDS
    raw_text = ""
    if suffix != ".parquet":
        if b"\x00" in raw:
            raise ValueError("binary input")
        raw_text = "" if jsonl_rows else raw.decode("utf-8", errors="strict")
        if (
            not raw
            or (jsonl_rows and not raw.strip())
            or (not jsonl_rows and not raw_text.strip())
        ):
            raise ValueError("empty input")
    text = (
        ""
        if streaming_rows or suffix in {".parquet", ".md", ".markdown"}
        else normalize_text(raw_text)
    )
    if suffix in {".jsonl", ".json", ".parquet"} and source.kind in ROW_SOURCE_KINDS:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            objects = pq.read_table(io.BytesIO(raw)).to_pylist()
        elif streaming_rows or suffix == ".jsonl":
            from klab.corpus.jsonl_records import records_from_bytes

            objects = (record.value for record in records_from_bytes(raw, source=name))
        else:
            loaded = json.loads(text)
            objects = loaded if isinstance(loaded, list) else [loaded]
        field = source.acquisition.text_field
        max_rows = source.acquisition.max_rows
        passages = []
        for n, item in enumerate(itertools.islice(objects, max_rows), first_row_index):
            upstream = item.get("_klab_source", {})
            if record_admission is not None:
                decision = record_admission["decisions"].get(
                    upstream.get("source_row_index")
                )
                if decision is None:
                    raise ValueError("record admission lacks selected source row")
                if decision["decision"] != "qualify":
                    if rejected_records is not None:
                        rejected_records.append(
                            {
                                "source_id": source.id,
                                "path": name,
                                "row": upstream["source_row_index"],
                                "reason": "record admission "
                                + decision["decision"]
                                + ": "
                                + decision["reason"],
                            }
                        )
                    continue
            metadata = {
                key: value
                for key, value in {**item, **upstream}.items()
                if key
                in {
                    "url",
                    "id",
                    "dump",
                    "date",
                    "file_path",
                    "language",
                    "score",
                    "int_score",
                    "token_count",
                    "language_score",
                    "title",
                    "license",
                    "license_type",
                    "path",
                    "repo_name",
                    "blob_id",
                    "source_row_index",
                    "source_shard_path",
                    "source_row_sha256",
                    "source_shard_sha256",
                    "source",
                    "corpusid",
                    "doi",
                    "year",
                    "page_uri",
                    "page_title",
                    "page_id",
                    "revision_id",
                    "revision_timestamp",
                    "source_uri",
                    "dump_revision",
                }
                and isinstance(value, (str, int, float, bool))
            }
            qualification = getattr(source.acquisition, "qualification", None)
            if qualification is not None:
                from klab.corpus.row_qualification import provenance

                try:
                    evidence = provenance(item, qualification)
                    if record_admission is not None and "row_evidence" in decision:
                        metadata["row_attribution_json"] = canonical_json(
                            decision["row_evidence"]
                        ).decode("utf-8")
                    metadata["qualification_profile"] = evidence["profile"]
                    metadata["qualification_family_hint"] = evidence["family_hint"]
                    if "synthetic_provenance_sha256" in evidence:
                        metadata["synthetic_provenance_sha256"] = evidence[
                            "synthetic_provenance_sha256"
                        ]
                    for key, value in evidence.get("attribution", {}).items():
                        if isinstance(value, str):
                            metadata["qualification_" + key] = value
                except TypeError, ValueError:
                    metadata["qualification_profile"] = qualification.profile
                    metadata["qualification_status"] = "unresolved"
            content = (
                normalize_text(item[field])
                if suffix == ".jsonl" and isinstance(item[field], str)
                else item[field]
            )
            passages.append((content, [], n, n, metadata))
    elif suffix == ".cnxml":
        passages = _cnxml_passages(raw)
    elif suffix in {".md", ".markdown"}:
        # Section line numbers refer to LF-delimited raw lines, not Unicode
        # separators or newlines introduced by normalizing lone CR bytes.
        passages = [
            (normalize_text(content), ancestry, start, end)
            for content, ancestry, start, end in markdown_sections(
                raw_text, source.markdown_documents == "sections"
            )
        ]
    else:
        line_end = raw.count(b"\n") + int(not raw.endswith(b"\n"))
        passages = [(text, [], 1, line_end)]
    rows = []
    raw_sha = full_file_sha256 or hashlib.sha256(raw).hexdigest()
    rights_payload = file_rights.model_dump(mode="json") if file_rights else None
    file_license = (
        record_admission["license_label"]
        if record_admission is not None
        else (
            file_rights.detected_spdx_expression
            or source.rights.spdx_expression
            or source.license
        )
        if file_rights
        else source.license
    )
    byte_offsets = [] if source.kind in ROW_SOURCE_KINDS else [0]
    if source.kind not in ROW_SOURCE_KINDS:
        offset = 0
        while (end := raw.find(b"\n", offset)) >= 0:
            byte_offsets.append(end + 1)
            offset = end + 1
        if offset < len(raw):
            byte_offsets.append(len(raw))
    for passage in passages:
        content, ancestry, start, end = passage[:4]
        metadata = passage[4] if len(passage) > 4 else None
        raw_content = content
        if not isinstance(content, str) or not content.strip():
            if document_policy.empty_text == "error":
                raise EmptyDocumentError(
                    f"empty/nontext document in source {source.id}, file {name}, "
                    f"row or line {start}; "
                    "release.policy.documents.empty_text: skip records and drops it"
                )
            reason = "empty_or_nontext"
        else:
            reason = (
                _excluded_research_record(content, metadata or {}, source.id)
                if document_policy.exclusions == "research_privacy"
                else None
            )
        if reason:
            if rejected_records is not None:
                rejected_records.append(
                    {
                        "source_id": source.id,
                        "path": name,
                        "row": start if source.kind in ROW_SOURCE_KINDS else None,
                        "reason": reason,
                    }
                )
            continue
        structure = None
        if normalizer_options.structure_cleaning and suffix != ".cnxml":
            from klab.corpus.structure_cleaning import clean_structure

            cleaned = clean_structure(
                content, source_id=source.id, options=normalizer_options
            )
            content, structure = cleaned["text"], cleaned["structure"]
        location = (
            f"{name}#row={start}"
            if source.kind in ROW_SOURCE_KINDS
            else f"{name}#lines={start}-{end}"
        )
        normalizer = (
            "cnxml-text-v1" if suffix == ".cnxml" else normalizer_options.identity
        )
        identity = canonical_digest([snapshot_id, location, raw_sha, normalizer])
        byte_start = byte_offsets[start - 1] if byte_offsets else None
        byte_end = byte_offsets[end] if byte_offsets else None
        raw_passage = (
            raw[byte_start:byte_end]
            if byte_start is not None and byte_end is not None
            else raw_content.encode("utf-8")
        )
        raw_content_sha = hashlib.sha256(raw_passage).hexdigest()
        document = {
            "document_id": identity,
            "source_id": source.id,
            "modality": source.modality,
            "source_revision": source.revision,
            "source_location": location,
            "license": file_license,
            "redistribution": (
                file_rights.redistribution_mode
                if file_rights
                else source.rights.redistribution_mode
            ),
            "domains": list(source.domains),
            "document_kind": (
                "prose"
                if document_policy.markdown_kind == "prose_when_declared"
                and suffix in {".md", ".markdown"}
                and "prose" in source.document_kinds
                else next(iter(source.document_kinds))
            ),
            "title": (
                metadata.get("page_title")
                if metadata and metadata.get("page_title")
                else ancestry[-1]
                if ancestry
                else Path(name).name
            ),
            "section_path": ancestry,
            "language": source_language(source),
            "text": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "raw_content_sha256": raw_content_sha,
            "source_family": source.source_family,
            **({"structure": structure} if structure is not None else {}),
            **(
                {"drop_reason": "metadata_only_front_matter"}
                if structure and structure["decision"] == "exclude_lm_metadata_only"
                else {}
            ),
            **(
                {
                    "metadata": {
                        **metadata,
                        **({"normalizer": normalizer} if structure is not None else {}),
                    }
                }
                if metadata
                else {"metadata": {"normalizer": normalizer}}
                if structure is not None
                else {"metadata": {"normalizer": normalizer}}
                if suffix == ".cnxml"
                else {}
            ),
            **(
                {"file_sha256": raw_sha, "rights": rights_payload}
                if file_rights
                else {}
            ),
        }
        document = NormalizedDocument.model_validate(document).model_dump(
            mode="json", exclude_none=True
        )
        evidence = {
            "record_id": identity,
            "snapshot_sha256": snapshot_id,
            "source_id": source.id,
            "raw_path": name,
            "raw_sha256": raw_sha,
            "raw_content_sha256": raw_content_sha,
            "line_start": start,
            "line_end": end,
            "byte_start": byte_start,
            "byte_end": byte_end,
            "normalizer": normalizer,
            "section_path": ancestry,
        }
        rows.append((document, evidence))
    return rows
