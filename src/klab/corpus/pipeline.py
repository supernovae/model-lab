"""Deterministic, evidence-preserving corpus normalization and derivation."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.corpus.project import (
    CURATED_SOURCE_KINDS,
    ROW_SOURCE_KINDS,
    ReleaseDeclaration,
    release_declaration_payload,
)
from klab.corpus.provenance import (
    DERIVED_ORIGIN,
    DETERMINISTIC_ORIGIN,
    HUMAN_ORIGIN,
    INFERENCE_ORIGIN,
    MULTI_SOURCE_ORIGIN,
    SOURCE_ORIGIN,
    rendered_digest,
    shape_for_record,
    validate_lineage,
    verification,
)
from klab.corpus.rights import (
    FileRights,
    reported_spdx_expression,
    resolve_file_rights,
    verify_record_admission,
)
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


class NormalizedDocument(StrictModel):
    """Versioned document retaining source attribution and raw evidence identity."""

    schema_version: Literal[1, 2, 3]
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


def _require_consistent_cleaning_decisions(documents: list[dict[str, Any]]) -> None:
    """Reject duplicate source rules that disagree about LM eligibility."""
    decisions = {
        (doc.get("structure") or {}).get("decision") == "exclude_lm_metadata_only"
        for doc in documents
    }
    if len(decisions) > 1:
        raise ValueError(
            "discordant metadata-only cleaning decisions in duplicate component"
        )


class GenerationRecord(StrictModel):
    """Captured response with explicit, non-self-attested validation status."""

    schema_version: Literal[1]
    request_id: str
    generator: dict[str, Any]
    generator_identity: str
    source_document_ids: list[str]
    raw_output: str
    parsed_output: dict[str, Any] | None
    validation_status: Literal[
        "unverified",
        "schema_validated",
        "source_entailed",
        "oracle_verified",
        "cross_source_verified",
        "human_reviewed",
        "rejected",
    ]
    rejection_reason: str | None
    evidence: dict[str, Any] | None
    seed: int
    transform_id: str
    record_id: str
    split: Literal["train", "validation", "test"]


class ScenarioRecord(StrictModel):
    """Deterministic, filesystem-free oracle world."""

    schema_version: Literal[1]
    generator_id: str
    generator_version: str
    world_seed: int
    scenario_family_id: str
    generator_world_id: str
    world_state: dict[str, Any]
    oracle_answer: str
    rendered_example: dict[str, str]
    template_family_id: str | None = None
    oracle_receipt: dict[str, Any] | None = None
    interpreter: str
    transform_id: str
    scenario_id: str
    split: Literal["train", "validation", "test"] | None = None


class ToolEpisode(StrictModel):
    """Static inert tool transcript bound to an oracle or source."""

    schema_version: Literal[1]
    goal: str
    tool_call: dict[str, Any]
    tool_result: str
    interpretation: str
    next_bounded_action: str
    verification: str
    final_response: str
    source_document_ids: list[str]
    scenario_id: str
    transform_id: str
    record_id: str


class Generator(Protocol):
    """Provider-neutral response interface; receipts, not providers, define identity."""

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...


class RecordedResponses:
    """Replay a pinned ledger; never infer a missing or changed response."""

    adapter_id = "recorded_responses"
    adapter_version = "1"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = {row["request_id"]: row for row in responses}
        if len(self._responses) != len(responses):
            raise ValueError("duplicate inference request ID")

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        response = self._responses.get(request["request_id"])
        if response is None or any(
            response.get(name) != request[name]
            for name in (
                "prompt_sha256",
                "source_document_ids",
                "provider",
                "model",
                "model_revision",
            )
        ):
            raise ValueError("recorded response request/prompt mismatch")
        raw = response.get("raw_output")
        if not isinstance(raw, str) or not isinstance(response.get("seed"), int):
            raise TypeError("recorded response requires raw output and seed")
        return raw, response


SPLITS = ("train", "validation", "test")
KINDS = frozenset(
    {
        "lm_text",
        "lexical_candidates",
        "semantic_candidates",
        "deterministic_scenarios",
        "inference_qa",
        "source_qa",
        "manual_semantic",
        "chat_sft",
        "tool_episode",
    }
)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
LEXEME = re.compile(
    r"(?<![\w-])--[\w-]+|\b[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+\b|\b[A-Z][A-Z_0-9]{2,}\b|\b[a-zA-Z_][\w]*(?:_[\w]+)+\b"
)
KEY_VALUE = re.compile(
    r"^[ \t]*([A-Za-z_][\w.-]*)[ \t]*[:=][ \t]*(\S.*?)[ \t]*$", re.MULTILINE
)


def digest(value: Any) -> str:
    return canonical_digest(value)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_json(row) + b"\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    from klab.corpus.jsonl_records import records_from_path

    return [record.value for record in records_from_path(path)]


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def _model(value: Any) -> dict[str, Any]:
    if isinstance(value, ReleaseDeclaration):
        return release_declaration_payload(value)
    return (
        value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
    )


def _normalized(text: str) -> str:
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
            value = _normalized(" ".join(visible_text(element).split()))
            if value:
                parts.append(value)
            return
        for child in element:
            visit(child)

    visit(content)
    if not parts:
        raise ValueError("CNXML has no prose")
    return [("\n".join(parts), [], 1, raw.count(b"\n") + int(not raw.endswith(b"\n")))]


def _sections(text: str, markdown: bool) -> list[tuple[str, list[str], int, int]]:
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


def _origin_keys(document: dict[str, Any]) -> tuple[str, ...]:
    """Identify a source page or paper without grouping unrelated host pages."""
    metadata = document.get("metadata") or {}
    keys: set[str] = set()
    for field in ("url", "page_uri"):
        value = metadata.get(field)
        if not isinstance(value, str):
            continue
        try:
            parsed = urlsplit(value)
            host = (parsed.hostname or "").lower().removeprefix("www.")
        except ValueError:
            continue
        if not host:
            continue
        query = urlencode(
            sorted(
                (key, val)
                for key, val in parse_qsl(parsed.query, keep_blank_values=True)
                if not key.lower().startswith("utm_")
                and key.lower() not in {"fbclid", "gclid"}
            )
        )
        path = parsed.path.rstrip("/") or "/"
        keys.add("url:" + host + path + ("?" + query if query else ""))
        if host in {"en.wikipedia.org", "en.wikibooks.org"} and path.startswith(
            "/wiki/"
        ):
            keys.add(
                "wiki:" + host + ":" + unquote(path[6:]).replace("_", " ").casefold()
            )
    if document["source_id"].startswith("pes2o") and metadata.get("id"):
        keys.add("paper:semantic_scholar:" + str(metadata["id"]))
    page_title = metadata.get("page_title")
    if isinstance(page_title, str) and document["source_id"] in {
        "wikipedia_20260901",
        "wikibooks_20260901",
    }:
        wiki_host = (
            "en.wikipedia.org"
            if document["source_id"] == "wikipedia_20260901"
            else "en.wikibooks.org"
        )
        keys.add("wiki:" + wiki_host + ":" + page_title.replace("_", " ").casefold())
    return tuple(sorted(keys))


def _records_for_file(
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
    normalizer_version: str = "normalizer-nfc-markdown-v1",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    from klab.corpus.structure_cleaning import (
        NORMALIZER_V1,
        NORMALIZER_V2,
        NORMALIZER_V3,
    )

    if normalizer_version not in {NORMALIZER_V1, NORMALIZER_V2, NORMALIZER_V3}:
        raise ValueError("unsupported corpus normalizer")
    suffix = Path(name).suffix.lower()
    streaming_rows = (
        source.schema_version == 3
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
        else _normalized(raw_text)
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
                _normalized(item[field])
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
            (_normalized(content), ancestry, start, end)
            for content, ancestry, start, end in _sections(raw_text, True)
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
            if source.schema_version != 3:
                raise ValueError("empty/nontext document")
            reason = "empty_or_nontext"
        else:
            reason = (
                _excluded_research_record(content, metadata or {}, source.id)
                if source.schema_version == 3
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
        if normalizer_version in {NORMALIZER_V2, NORMALIZER_V3} and suffix != ".cnxml":
            from klab.corpus.structure_cleaning import clean_structure

            cleaned = clean_structure(
                content, source_id=source.id, normalizer=normalizer_version
            )
            content, structure = cleaned["text"], cleaned["structure"]
        location = (
            f"{name}#row={start}"
            if source.kind in ROW_SOURCE_KINDS
            else f"{name}#lines={start}-{end}"
        )
        normalizer = "cnxml-text-v1" if suffix == ".cnxml" else normalizer_version
        identity = digest([snapshot_id, location, raw_sha, normalizer])
        byte_start = byte_offsets[start - 1] if byte_offsets else None
        byte_end = byte_offsets[end] if byte_offsets else None
        raw_passage = (
            raw[byte_start:byte_end]
            if byte_start is not None and byte_end is not None
            else raw_content.encode("utf-8")
        )
        raw_content_sha = hashlib.sha256(raw_passage).hexdigest()
        document = {
            "schema_version": source.schema_version,
            "document_id": identity,
            "source_id": source.id,
            "modality": source.modality,
            "source_revision": source.revision,
            "source_location": location,
            "license": file_license,
            "redistribution": (
                file_rights.redistribution_mode
                if file_rights
                else source.redistribution
            ),
            "domains": list(source.domains),
            "document_kind": (
                "prose"
                if source.schema_version == 3
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
            "language": "en",
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


def _split(record: dict[str, Any], policy: dict[str, Any]) -> str:
    unit = policy["unit"]
    if unit == "document":
        keys = record.get("parent_document_ids") or [
            record.get("document_id") or record.get("scenario_family_id")
        ]
    elif unit in {"source_repository", "source_document_family"}:
        keys = record.get("source_family_ids") or [
            record.get("source_family") or record.get("scenario_family_id")
        ]
    else:
        field = {
            "scenario_family": "scenario_family_id",
            "generator_world": "generator_world_id",
            "template_family": "template_family_id",
        }[unit]
        keys = [record.get(field)]
    keys = [k for k in keys if k]
    if not keys:
        raise ValueError(f"missing {unit} split lineage")
    assignments = policy["assignments"]
    choices = {assignments[k] for k in keys if k in assignments}
    if len(choices) != 1 or any(k not in assignments for k in keys):
        raise ValueError(f"missing/conflicting split assignment: {keys}")
    choice = choices.pop()
    if choice not in SPLITS:
        raise ValueError("invalid split")
    return choice


def _lineage(
    record: dict[str, Any],
    parents: list[dict[str, Any]],
    stage: str,
    *,
    family: str | None = None,
    world: str | None = None,
    template: str | None = None,
) -> dict[str, Any]:
    ids = sorted({p["document_id"] for p in parents})
    families = sorted({p["source_family"] for p in parents})
    return {
        "record_id": record["record_id"],
        "record_kind": record["kind"],
        "transform_id": stage,
        "parent_document_ids": ids,
        "original_parent_document_ids": ids,
        "source_family_ids": families,
        "scenario_family_id": family,
        "generator_world_id": world,
        "template_family_id": template,
        "domains": ["systems_scenarios"]
        if world
        else sorted({domain for parent in parents for domain in parent["domains"]}),
        "validation_status": record.get("validation_status", "schema_validated"),
    }


def _path_scenario(seed: int, family: str, stage: str) -> dict[str, Any]:
    from random import Random

    rng = Random(seed)
    parts = [
        "",
        "workspace",
        "project",
        f"module{rng.randrange(1000)}",
        f"item{rng.randrange(1000)}.py",
    ]
    path = "/".join(parts)
    suffix = PurePosixPath(path).suffix
    world = {"path": path, "operation": "suffix"}
    result = {
        "schema_version": 1,
        "generator_id": "pathlib_path_suffix_v1",
        "generator_version": "1",
        "world_seed": seed,
        "scenario_family_id": family,
        "generator_world_id": f"{family}:{seed}",
        "world_state": world,
        "oracle_answer": suffix,
        "rendered_example": {
            "question": f"What is the suffix of {path}?",
            "answer": suffix,
        },
        "interpreter": "pathlib.PurePosixPath",
        "transform_id": stage,
    }
    result["scenario_id"] = digest(result)
    return ScenarioRecord.model_validate(result).model_dump(
        mode="json", exclude_none=True
    )


def _scenario_v2(
    generator: str, seed: int, family: str, template: str, stage: str
) -> dict[str, Any]:
    """Construct distinct causal configurations, not seed-labelled copies."""
    if not 0 <= seed < 1000:
        raise ValueError("v2 world seeds must be in the declared 0–999 partition")
    a, b, c = seed % 10, (seed // 10) % 10, seed // 100
    if generator == "filesystem_judgment_v2":
        targets = (
            ("outside workspace", False, False),
            ("symlink escaping workspace", False, False),
            ("generated build cache", True, True),
            ("generated coverage output", True, True),
            ("generated test scratch", True, True),
            ("tracked source directory", True, False),
            ("tracked configuration directory", True, False),
            ("empty target", True, False),
            ("generated deployment bundle", True, True),
            ("generated package artifacts", True, True),
        )
        evidence_states = (
            "exact path and owner confirmed",
            "target path unset",
            "two candidate paths",
            "listing contradicts request",
            "symlink destination unverified",
            "owner confirmed by manifest",
            "target confirmed by dry-run",
            "stale listing",
            "build manifest confirms target",
            "request conflicts with manifest",
        )
        operations = (
            ("remove disposable output", True, True),
            ("replace generated cache", True, True),
            ("archive before cleanup", True, True),
            ("delete without backup", False, True),
            ("modify source", True, False),
            ("reset tracked files", False, False),
            ("clear temporary outputs", True, True),
            ("overwrite published artifacts", False, True),
            ("rename with restore point", True, True),
            ("prune generated files", True, True),
        )
        target, in_scope, generated = targets[a]
        evidence = evidence_states[b]
        operation, reversible, allowed_kind = operations[c]
        authorized = (a + b + c) % 4 != 0
        world = {
            "target": target,
            "in_workspace": in_scope,
            "generated": generated,
            "evidence": evidence,
            "operation": operation,
            "reversible": reversible,
            "authorized": authorized,
        }
        judgment = (
            "stop"
            if not in_scope
            or not reversible
            or not allowed_kind
            or target.startswith("tracked")
            else "ask"
            if not authorized
            else "inspect"
            if target == "empty target" or b not in (0, 5, 6, 8)
            else "proceed"
        )
        diagnostic = (
            f"Check the declared {target} against {evidence} before {operation}."
        )
        risk = "An out-of-scope or irreversible filesystem change is unsafe."
        verification = (
            f"Confirm {target} remains within the requested workspace boundary."
        )
        rollback = (
            "Restore the saved prior state." if reversible else "No verified rollback."
        )
    elif generator == "platform_fault_v2":
        causes = (
            ("Pending workload", "scheduling events"),
            ("unbound storage", "claim binding events"),
            ("bad probe", "probe configuration"),
            ("image pull", "image reference and pull events"),
            ("DNS", "resolver responses"),
            ("resource exhaustion", "resource limits and usage"),
            ("network policy", "selected policy rules"),
            ("config mismatch", "declared and effective configuration"),
        )
        observations = (
            "first failure after rollout",
            "repeated failure under load",
            "failure on one replica",
            "failure on every replica",
            "intermittent failure",
        )
        workloads = ("api service", "worker", "controller", "scheduled job", "gateway")
        scopes = (
            "single namespace",
            "one node",
            "all nodes",
            "new revision",
            "one zone",
        )
        cause, check = causes[seed % 8]
        observation = observations[(seed // 8) % 5]
        context = workloads[(seed // 40) % 5]
        scope = scopes[seed // 200]
        world = {
            "cause": cause,
            "evidence": f"{cause} observed: {observation}",
            "resource": context,
            "scope": scope,
            "intervention": "restart all workloads",
        }
        judgment = "inspect"
        diagnostic = f"Inspect {check} for {context} in {scope}."
        risk = "A broad restart is unsupported by these observations."
        verification = (
            f"After a targeted correction, recheck {cause} for {context} in {scope}."
        )
        rollback = "Revert the targeted correction if the observation persists."
    elif generator == "deployment_change_v2":
        preflights = (
            "preflight passed",
            "preflight absent",
            "config check failed",
            "dependency unhealthy",
            "migration untested",
            "capacity confirmed",
            "capacity unknown",
            "health baseline passed",
            "health baseline failed",
            "goal already verified",
        )
        radii = (
            "one canary",
            "one replica",
            "one namespace",
            "one shard",
            "one zone",
            "all regions",
            "all tenants",
            "one background worker",
            "one staging environment",
            "entire production cluster",
        )
        recovery = (
            "rollback rehearsed",
            "rollback unavailable",
            "rollback untested",
            "prior version retained",
            "data migration irreversible",
            "rollback timed out",
            "snapshot verified",
            "snapshot absent",
            "canary abort verified",
            "rollback approval missing",
        )
        world = {
            "condition": preflights[a],
            "blast_radius": radii[b],
            "rollback_state": recovery[c],
        }
        judgment = (
            "stop"
            if a == 9 or a not in (0, 5, 7) or b in (5, 6, 9) or c not in (0, 3, 6, 8)
            else "proceed"
        )
        diagnostic = f"Review {preflights[a]}, {radii[b]} and {recovery[c]}."
        risk = f"The proposed rollout affects {radii[b]}."
        verification = (
            f"Check deployment health and user-visible success for {radii[b]}."
        )
        rollback = f"Recovery condition: {recovery[c]}."
    elif generator == "code_test_workflow_v2":
        failures = (
            "focused assertion fails",
            "type check fails",
            "configuration parse fails",
            "integration assertion fails",
            "build fails",
            "focused test passes",
            "broad regression fails",
            "broad suite passes",
            "goal verified",
            "contradictory test results",
        )
        edits = (
            "guard empty input",
            "correct parser branch",
            "update configuration key",
            "repair timeout handling",
            "fix return value",
            "restore error propagation",
            "correct bounds check",
            "repair fixture setup",
            "fix state transition",
            "remove invalid retry",
        )
        checks = (
            "focused test not run",
            "focused test failed",
            "focused test passed",
            "broad suite not run",
            "broad suite failed",
            "broad suite passed",
            "regression test not run",
            "regression test passed",
            "verification conflicts",
            "both focused and broad tests passed",
        )
        world = {
            "fixture": "inert disposable fixture",
            "condition": failures[a],
            "candidate_edit": edits[b],
            "verification_state": checks[c],
        }
        judgment = (
            "stop"
            if a == 8 and c == 9
            else "inspect"
            if a == 8 or c in (1, 4, 8) or a in (0, 1, 2, 3, 4, 6, 9)
            else "proceed"
        )
        diagnostic = f"Inspect {failures[a]} and verify {edits[b]} using {checks[c]}."
        risk = "A candidate edit without consistent focused and broad checks is not a verified fix."
        verification = f"Require focused and broad verification of {edits[b]}."
        rollback = "Restore the disposable fixture if checks fail."
    else:
        raise ValueError(f"unregistered v2 scenario generator: {generator}")
    answer = f"{judgment}: {diagnostic}"
    receipt = {
        "schema_version": 1,
        "generator_id": generator,
        "generator_version": "2",
        "world_id": f"{generator}:{seed}",
        "scenario_family_id": family,
        "template_family_id": template,
        "world_facts": world,
        "judgment": judgment,
        "evidence": world.get("evidence", world.get("condition", world.get("target"))),
        "next_diagnostic": diagnostic,
        "risk": risk,
        "verification": verification,
        "rollback": rollback,
    }
    result = {
        "schema_version": 1,
        "generator_id": generator,
        "generator_version": "2",
        "world_seed": seed,
        "scenario_family_id": family,
        "generator_world_id": receipt["world_id"],
        "template_family_id": template,
        "world_state": world,
        "oracle_answer": answer,
        "oracle_receipt": receipt,
        "rendered_example": {
            "question": f"Inert {generator} world: {world}. Select the next bounded response.",
            "answer": answer,
        },
        "interpreter": generator,
        "transform_id": stage,
    }
    result["scenario_id"] = digest(result)
    return ScenarioRecord.model_validate(result).model_dump(
        mode="json", exclude_none=True
    )


GENERATORS = frozenset(
    {
        "pathlib_path_suffix_v1",
        "filesystem_judgment_v1",
        "platform_fault_v1",
        "deployment_change_v1",
        "code_test_workflow_v1",
        "filesystem_judgment_v2",
        "platform_fault_v2",
        "deployment_change_v2",
        "code_test_workflow_v2",
    }
)


def _scenario(
    generator: str, seed: int, family: str, template: str, stage: str
) -> dict[str, Any]:
    """Pure world constructors: no host inspection, commands, or mutable state."""
    if generator.endswith("_v2"):
        return _scenario_v2(generator, seed, family, template, stage)
    if generator == "filesystem_judgment_v1":
        cases = (
            ("outside", "no", "known", "yes", "reversible"),
            ("source", "no", "known", "yes", "reversible"),
            ("source", "yes", "unset", "yes", "reversible"),
            ("generated", "yes", "ambiguous", "yes", "reversible"),
            ("generated", "yes", "known", "yes", "irreversible"),
            ("generated", "yes", "known", "yes", "reversible"),
            ("source", "yes", "known", "yes", "reversible"),
            ("empty", "yes", "unset", "yes", "reversible"),
            ("generated", "yes", "contradictory", "yes", "reversible"),
        )
        target, authorized, evidence, bounded, operation = cases[seed % len(cases)]
        world = {
            "target": target,
            "authorized": authorized,
            "evidence": evidence,
            "bounded": bounded,
            "operation": operation,
        }
        judgment = (
            "stop"
            if target == "outside" or bounded == "no" or operation == "irreversible"
            else "ask"
            if authorized == "no"
            else "inspect"
            if evidence != "known"
            else "proceed"
        )
        answer = (
            f"{judgment}: "
            + {
                "stop": "Target or operation is outside the safe bounded workspace.",
                "ask": "Obtain explicit authorization before changing the target.",
                "inspect": "Resolve the exact target and conflicting evidence before acting.",
                "proceed": "Only the authorized, reversible in-workspace operation is justified.",
            }[judgment]
        )
        diagnostic = "Confirm exact target, authorization and reversibility."
        risk = "Never infer permission from a tool transcript."
        verification = "Compare the intended target with the resulting workspace state."
        rollback = (
            "Restore the reversible change from its prior state."
            if operation == "reversible"
            else "No verified rollback."
        )
    elif generator == "platform_fault_v1":
        causes = (
            (
                "Pending workload",
                "scheduler reports unschedulable",
                "inspect scheduling events",
            ),
            (
                "unbound storage",
                "claim remains Pending",
                "inspect claim and volume binding events",
            ),
            (
                "bad probe",
                "readiness probe fails",
                "inspect probe configuration and container health",
            ),
            (
                "image pull",
                "image pull reports failure",
                "inspect image reference and pull events",
            ),
            (
                "DNS",
                "name resolution fails",
                "inspect resolver and DNS service responses",
            ),
            (
                "resource exhaustion",
                "container reports out-of-memory",
                "inspect memory limits and usage",
            ),
            (
                "network policy",
                "connection denied by policy",
                "inspect the selected ingress/egress rules",
            ),
            (
                "config mismatch",
                "effective config differs from declared config",
                "compare effective and declared configuration",
            ),
        )
        cause, evidence, diagnostic = causes[seed % len(causes)]
        world = {
            "cause": cause,
            "evidence": evidence,
            "intervention": "restart everything",
        }
        judgment = "inspect"
        answer = f"inspect: {cause}: {evidence}; next diagnostic: {diagnostic}. Do not restart everything."
        risk = "Broad restart is unsupported by this evidence."
        verification = f"Recheck {evidence} after a targeted correction."
        rollback = "Revert any targeted configuration change if verification fails."
    elif generator == "deployment_change_v1":
        failures = (
            "missing preflight",
            "excessive blast radius",
            "failed verification",
            "healthy rollout",
            "no rollback",
        )
        condition = failures[seed % len(failures)]
        world = {
            "condition": condition,
            "rollback_available": "no" if condition == "no rollback" else "yes",
            "blast_radius": "unbounded"
            if condition == "excessive blast radius"
            else "bounded",
        }
        judgment = "proceed" if condition == "healthy rollout" else "stop"
        diagnostic = "Confirm preflight, bounded blast radius and rollback readiness."
        answer = f"{judgment}: {condition}; {diagnostic}"
        risk = "Deployment may affect users outside the intended scope."
        verification = "Check rollout health and user-visible success criteria."
        rollback = (
            "No verified rollback is available."
            if condition == "no rollback"
            else "Return to the last verified deployment if health fails."
        )
    elif generator == "code_test_workflow_v1":
        conditions = (
            "focused test fails",
            "candidate edit unverified",
            "focused test passes",
            "broad regression fails",
            "goal verified",
        )
        condition = conditions[seed % len(conditions)]
        world = {
            "fixture": "inert disposable fixture",
            "condition": condition,
            "candidate_edit": "local reversible edit",
        }
        judgment = "stop" if condition == "goal verified" else "inspect"
        diagnostic = (
            "Inspect the failure and run focused then broad verification."
            if judgment != "stop"
            else "No further edits needed."
        )
        answer = f"{judgment}: {condition}; {diagnostic}"
        risk = "An unverified change is not a successful fix."
        verification = "Focused and broad checks must agree before concluding success."
        rollback = "Revert the disposable fixture edit if regression persists."
    else:
        raise ValueError(f"unregistered scenario generator: {generator}")
    world["fixture_id"] = f"{generator}:{seed}"
    receipt = {
        "schema_version": 1,
        "generator_id": generator,
        "generator_version": "1",
        "world_id": f"{generator}:{seed}",
        "scenario_family_id": family,
        "template_family_id": template,
        "world_facts": world,
        "judgment": judgment,
        "evidence": world.get("evidence", world.get("condition", world.get("target"))),
        "next_diagnostic": diagnostic,
        "risk": risk,
        "verification": verification,
        "rollback": rollback,
    }
    result = {
        "schema_version": 1,
        "generator_id": generator,
        "generator_version": "1",
        "world_seed": seed,
        "scenario_family_id": family,
        "generator_world_id": receipt["world_id"],
        "template_family_id": template,
        "world_state": world,
        "oracle_answer": answer,
        "oracle_receipt": receipt,
        "rendered_example": {
            "question": (
                f"Given this declared {generator} world {world}, what is the next bounded judgment?"
                if seed < 800
                else f"Review only the stated {generator} facts {world}. What should happen next?"
                if seed < 900
                else f"Evaluate the evidence in {world} for {generator}; select a bounded response."
            ),
            "answer": answer,
        },
        "interpreter": generator,
        "transform_id": stage,
    }
    result["scenario_id"] = digest(result)
    return ScenarioRecord.model_validate(result).model_dump(
        mode="json", exclude_none=True
    )


def _scenario_messages(scenario: dict[str, Any], tool: bool) -> list[dict[str, Any]]:
    question = scenario["rendered_example"]["question"]
    answer = scenario["oracle_answer"]
    if not question or scenario["rendered_example"].get("answer") != answer:
        raise ValueError("scenario lacks a verified rendered answer")
    if not tool:
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    call_id = scenario["scenario_id"][:16]
    if scenario["generator_id"] == "pathlib_path_suffix_v1":
        name = "pathlib_suffix"
        arguments = {"path": scenario["world_state"]["path"]}
        instruction = "Inspect the declared path world."
        final = f"The tool result is {answer}; checking against the declared path suffix gives {answer}. Final answer: {answer}"
    else:
        receipt = scenario.get("oracle_receipt")
        if (
            not receipt
            or not receipt.get("evidence")
            or not receipt.get("verification")
        ):
            raise ValueError("scenario has no complete inert oracle tool receipt")
        name = "declared_world_inspection_v1"
        arguments = {
            "world_id": receipt["world_id"],
            "generator_version": receipt["generator_version"],
        }
        instruction = "Inspect only the inert declared world receipt."
        final = f"Evidence: {receipt['evidence']}. {answer} Verification: {receipt['verification']} Rollback: {receipt['rollback']}"
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": instruction,
            "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": answer if name == "pathlib_suffix" else str(receipt["evidence"]),
        },
        {"role": "assistant", "content": final},
    ]


def _scenario_shape(scenario: dict[str, Any], kind: str) -> str:
    if kind == "tool_episode":
        return "tool_trace"
    return {
        "filesystem_judgment_v1": "stop_or_abstain",
        "platform_fault_v1": "error_diagnosis",
        "deployment_change_v1": "verification_episode",
        "code_test_workflow_v1": "multi_turn_dialogue",
        "filesystem_judgment_v2": "stop_or_abstain",
        "platform_fault_v2": "error_diagnosis",
        "deployment_change_v2": "verification_episode",
        "code_test_workflow_v2": "multi_turn_dialogue",
    }.get(scenario.get("generator_id"), "troubleshooting_scenario")


def _snapshot_file(
    project: Any,
    lock: dict[str, Any],
    source_id: str,
    name: str,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> bytes:
    entry = lock["sources"][source_id]
    root = Path(entry["snapshot_path"])
    from klab.corpus.acquisition import verify_snapshot

    manifest = verify_snapshot(
        root, proof_store=proof_store, verification_mode=verification_mode
    )
    matches = [item for item in manifest["files"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"ambiguous/missing snapshotted input: {source_id}/{name}")
    return (root / "files" / matches[0]["path"]).read_bytes()


def _ordered_transforms(project: Any) -> list[Any]:
    sources = {source.id for source in project.sources}
    stages = {stage.id: stage for stage in project.transforms}
    if len(stages) != len(project.transforms) or sources.intersection(stages):
        raise ValueError("ambiguous transform/source ID")
    ordered: list[Any] = []
    pending = dict(stages)
    while pending:
        ready = [
            stage
            for stage in pending.values()
            if all(
                dependency in sources or dependency in {item.id for item in ordered}
                for dependency in stage.inputs
            )
        ]
        if not ready:
            raise ValueError("unknown or cyclic transform dependency")
        stage = min(ready, key=lambda value: value.id)
        ordered.append(stage)
        del pending[stage.id]
    return ordered


@contextmanager
def _staged_build(root: Path, build_id: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temporary:
        staging = Path(temporary)
        try:
            yield staging
        except Exception as exc:
            diagnostic = root / "diagnostics" / f"{build_id}.json"
            details = (
                json.loads(diagnostic.read_text(encoding="utf-8"))
                if diagnostic.exists()
                else {}
            )
            details.setdefault("error", str(exc))
            details["build_id"] = build_id
            for name in ("rejected.jsonl", "generations.jsonl", "audit.json"):
                source = staging / name
                if source.exists():
                    details[name] = (
                        _rows(source)
                        if name.endswith(".jsonl")
                        else json.loads(source.read_text(encoding="utf-8"))
                    )
            _json(diagnostic, details)
            raise


def _uses_disk_lm_builder(project: Any, transforms: list[Any]) -> bool:
    return (
        project.release.schema_version in (2, 3)
        and len(transforms) == 1
        and transforms[0].kind == "lm_text"
        and project.release.lm.selected
        and not project.release.chat.selected
        and all(
            source.schema_version == project.release.schema_version
            for source in project.sources
        )
        and (
            project.release.schema_version == 3
            or (
                project.release.record_admission is not None
                and project.release.fraction is None
                and all(
                    source.kind in {"huggingface_dataset", "git"}
                    for source in project.sources
                )
            )
        )
    )


def build(
    project: Any,
    work_root: Path,
    offline: bool = False,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    reuse_completed_preparation: bool = False,
) -> Path:
    """Keep bounded metadata alive through native admission and build consumers."""
    if reuse_completed_preparation:
        from klab.corpus.completed_preparation import build_completed

        return build_completed(project, work_root)
    from klab.corpus.bounded_metadata import metadata_operation

    with metadata_operation(
        Path(work_root) / "corpora" / project.config.id / "metadata-scratch",
        max_database_bytes=None,
    ):
        return _build_stored(
            project,
            work_root,
            offline,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _build_stored(
    project: Any,
    work_root: Path,
    offline: bool = False,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> Path:
    """Build an immutable derived inventory from the exact acquisition lock."""
    from klab.corpus.acquisition import verify_acquisition

    lock = verify_acquisition(
        project, work_root, proof_store=proof_store, verification_mode=verification_mode
    )
    record_admission: dict[tuple[str, str], dict[str, Any]] = {}
    admission_path: Path | None = None
    if project.release.record_admission is not None:
        from klab.corpus.project import project_path, source_declaration_payload

        reference = project.release.record_admission
        admission_path = project_path(project.root, reference.path)
        if sha256_file(admission_path) != reference.sha256.lower():
            raise ValueError("record admission SHA-256 mismatch")
        from klab.corpus.bounded_metadata import read_metadata

        admission_manifest = read_metadata(admission_path)
        record_admission = verify_record_admission(
            admission_manifest,
            {
                source.id: source_declaration_payload(source)
                for source in project.sources
            },
            {
                source_id: read_metadata(Path(row["snapshot_path"]) / "manifest.json")
                for source_id, row in lock["sources"].items()
            },
            Path(work_root) / "corpora" / project.config.id / "snapshots",
        )
    root = Path(work_root) / "corpora" / project.config.id / "builds"
    root.mkdir(parents=True, exist_ok=True)
    transforms = _ordered_transforms(project)
    stage_specs = [_model(t) for t in transforms]
    for t in stage_specs:
        if t["kind"] not in KINDS:
            raise ValueError(f"unregistered transform: {t['kind']}")
    identity = {
        "project_id": project.config.id,
        "project": _model(project.config),
        "lock": {
            key: {
                "declaration_sha256": value["declaration_sha256"],
                "snapshot_sha256": value.get("snapshot_sha256"),
            }
            for key, value in sorted(lock["sources"].items())
        },
        "transforms": stage_specs,
        "split": _model(project.splits),
        "release": _model(project.release),
        "implementation_sha256": sha256_file(Path(__file__)),
        "schema_implementation_sha256": sha256_file(
            Path(__file__).with_name("project.py")
        ),
        "provenance_implementation_sha256": sha256_file(
            Path(__file__).with_name("provenance.py")
        ),
        **(
            {
                "rights_implementation_sha256": sha256_file(
                    Path(__file__).with_name("rights.py")
                )
            }
            if project.release.schema_version in (2, 3)
            else {}
        ),
    }
    large_lm_build = _uses_disk_lm_builder(project, transforms)
    if large_lm_build:
        identity["large_builder_implementation_sha256"] = sha256_file(
            Path(__file__).with_name("large_build.py")
        )
    build_id = digest(identity)
    target = root / build_id
    if target.exists():
        from klab.corpus.release import verify_build

        verify_build(
            target, proof_store=proof_store, verification_mode=verification_mode
        )
        return target
    if large_lm_build:
        from klab.corpus.large_build import build_large
        from klab.corpus.progress import BuildProgress

        workspace = Path(work_root) / "corpora" / project.config.id
        progress = BuildProgress(workspace / "progress" / f"{build_id}.jsonl", build_id)
        return build_large(
            project,
            workspace,
            lock,
            identity,
            build_id,
            target,
            progress,
            proof_store=proof_store,
            verification_mode=verification_mode,
            record_admission=record_admission,
            admission_path=admission_path,
        )
    with _staged_build(root, build_id) as staging:
        from klab.corpus.split_assignments import copy_assignments

        copy_assignments(project, staging)
        if admission_path is not None:
            import shutil

            destination = staging / "record-admission.json"
            shutil.copyfile(admission_path, destination)
            if (
                sha256_file(destination)
                != project.release.record_admission.sha256.lower()
            ):
                raise ValueError("record admission changed during build")
        documents: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        rights_files: list[dict[str, Any]] = []
        auxiliary_files: set[tuple[str, str]] = set()
        for spec in stage_specs:
            params = spec["parameters"]
            if spec["kind"] == "inference_qa":
                if not all(
                    params.get(name)
                    for name in (
                        "generator_source_id",
                        "prompt_template_path",
                        "responses_path",
                    )
                ):
                    raise ValueError("incomplete recorded inference generator")
                auxiliary_files.update(
                    (params["generator_source_id"], params[name])
                    for name in ("prompt_template_path", "responses_path")
                )
            elif spec["kind"] == "manual_semantic":
                auxiliary_files.add((params["source_id"], params["path"]))
        for source in sorted(project.sources, key=lambda s: s.id):
            entry = lock["sources"].get(source.id)
            if (
                source.redistribution == "rejected"
                or not entry
                or not entry.get("snapshot_path")
            ):
                continue
            from klab.corpus.acquisition import verify_snapshot

            snapshot = verify_snapshot(
                Path(entry["snapshot_path"]),
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            nested_path = source.rights.nested_metadata_path if source.rights else None
            nested_metadata = (
                {
                    nested_path: json.loads(
                        (
                            Path(entry["snapshot_path"]) / "files" / nested_path
                        ).read_text(encoding="utf-8")
                    )
                }
                if nested_path
                else {}
            )
            for file in sorted(snapshot["files"], key=lambda f: f["path"]):
                name = file["path"]
                if name == nested_path:
                    rights_files.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "sha256": file["sha256"],
                            "size": file["size"],
                            "role": "license_metadata",
                            "canonical_uri": source.canonical_uri,
                            "revision": source.revision,
                        }
                    )
                    continue
                raw = (Path(entry["snapshot_path"]) / "files" / name).read_bytes()
                decision = (
                    resolve_file_rights(
                        source.rights,
                        name,
                        raw,
                        nested_metadata=nested_metadata,
                        prospective_private_research=source.schema_version == 3,
                    )
                    if source.rights
                    else None
                )
                admitted = record_admission.get((source.id, name))
                if admitted is not None:
                    decision = admitted["rights"]
                if decision is not None:
                    rights_files.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "sha256": file["sha256"],
                            "size": file["size"],
                            "role": "transform_input"
                            if (source.id, name) in auxiliary_files
                            else "document",
                            "canonical_uri": source.canonical_uri,
                            "revision": source.revision,
                            "license_url": source.license_url,
                            "rights": decision.model_dump(mode="json"),
                            **(
                                {
                                    "admission_policy_id": admitted["policy_id"],
                                    "admission_policy_sha256": admitted[
                                        "policy_sha256"
                                    ],
                                    "admission_license_label": admitted[
                                        "license_label"
                                    ],
                                    "admission_policy_spdx_expression": admitted[
                                        "policy_spdx_expression"
                                    ],
                                }
                                if admitted is not None
                                else {}
                            ),
                        }
                    )
                    if decision.training_eligibility not in {
                        "eligible",
                        "eligible_with_obligations",
                    }:
                        rejected.append(
                            {
                                "source_id": source.id,
                                "path": name,
                                "reason": f"rights {decision.training_eligibility}: {decision.reason}",
                            }
                        )
                        if (source.id, name) in auxiliary_files:
                            raise ValueError(
                                f"unresolved rights for transform input: {source.id}/{name}"
                            )
                        continue
                if (source.id, name) in auxiliary_files:
                    continue
                try:
                    pairs = _records_for_file(
                        raw,
                        name,
                        source,
                        entry["snapshot_sha256"],
                        file_rights=decision,
                        rejected_records=rejected,
                        record_admission=admitted,
                        normalizer_version=project.release.normalizer
                        or "normalizer-nfc-markdown-v1",
                    )
                    documents.extend(doc for doc, _ in pairs)
                    evidence.extend(span for _, span in pairs)
                except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                    rejected.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "reason": str(exc),
                        }
                    )
        if record_admission:
            expected_rows = {
                (source_id, path, index)
                for (source_id, path), item in record_admission.items()
                for index, choice in item.get("decisions", {}).items()
                if choice["decision"] == "qualify"
            }
            observed_rows = {
                (
                    doc["source_id"],
                    doc["source_location"].split("#", 1)[0],
                    doc.get("metadata", {}).get("source_row_index"),
                )
                for doc in documents
                if "decisions"
                in record_admission.get(
                    (doc["source_id"], doc["source_location"].split("#", 1)[0]),
                    {},
                )
            }
            if observed_rows != expected_rows:
                raise ValueError("admitted source row inventory differs from documents")
        _jsonl(staging / "rejected.jsonl", rejected)
        documents.sort(key=lambda d: d["document_id"])
        from klab.corpus.split_assignments import split_policy

        policy = split_policy(project)
        for doc in documents:
            doc["split"] = _split(doc, policy)
        groups: dict[str, list[str]] = defaultdict(list)
        normalized: dict[str, list[str]] = defaultdict(list)
        origin: dict[str, list[str]] = defaultdict(list)
        by_id = {d["document_id"]: d for d in documents}
        source_rank = {
            source.id: (
                0
                if source.kind in CURATED_SOURCE_KINDS
                else 1
                if source.id.startswith("pes2o")
                else 2
            )
            for source in project.sources
        }
        for doc in documents:
            groups[doc["raw_content_sha256"]].append(doc["document_id"])
            normalized[doc["content_sha256"]].append(doc["document_id"])
            if project.release.schema_version == 3:
                for key in _origin_keys(doc):
                    origin[key].append(doc["document_id"])
        duplicate_groups = []
        representative = {d["document_id"]: d["document_id"] for d in documents}
        mappings = [("raw", groups), ("normalized", normalized)]
        if project.release.schema_version == 3:
            mappings.append(("canonical_origin", origin))
        for method, mapping in mappings:
            for content_id, ids in sorted(mapping.items()):
                if len(ids) < 2:
                    continue
                ids = sorted(ids)
                splits = {by_id[i]["split"] for i in ids}
                heldout = splits - {"train"}
                priority = next(iter(heldout)) if heldout else None
                selected = (
                    min(
                        ids,
                        key=lambda i: (
                            by_id[i]["split"] != priority if priority else False,
                            source_rank[by_id[i]["source_id"]],
                            by_id[i]["document_kind"] != "paper",
                            i,
                        ),
                    )
                    if project.release.schema_version == 3
                    else ids[0]
                )
                duplicate_groups.append(
                    {
                        "method": method,
                        "sha256": (
                            hashlib.sha256(content_id.encode()).hexdigest()
                            if method == "canonical_origin"
                            else content_id
                        ),
                        "origins": ids,
                        "splits": sorted(splits),
                        "representative": selected,
                    }
                )
                if len(splits) > 1 and (
                    project.release.schema_version != 3 or len(heldout) != 1
                ):
                    diagnostic = {
                        "duplicates": duplicate_groups,
                        "rejected": rejected,
                        "error": "cross-split exact text overlap",
                    }
                    _json(staging / "audit.json", diagnostic)
                    _json(root / "diagnostics" / f"{build_id}.json", diagnostic)
                    raise ValueError("cross-split exact text overlap")
                if method == "normalized" and project.release.schema_version == 2:
                    rights_signatures = {
                        canonical_json(
                            {
                                "license": by_id[i]["license"],
                                "redistribution": by_id[i]["redistribution"],
                                "training_eligibility": by_id[i]["rights"][
                                    "training_eligibility"
                                ],
                                "license_references": by_id[i]["rights"][
                                    "license_references"
                                ],
                                "notices": by_id[i]["rights"]["notices"],
                                "training_restriction": by_id[i]["rights"][
                                    "training_restriction"
                                ],
                            }
                        )
                        for i in ids
                    }
                    if len(rights_signatures) > 1:
                        diagnostic = {
                            "duplicates": duplicate_groups,
                            "rejected": rejected,
                            "error": "duplicate source text has incompatible rights evidence",
                        }
                        _json(staging / "audit.json", diagnostic)
                        _json(root / "diagnostics" / f"{build_id}.json", diagnostic)
                        raise ValueError(
                            "duplicate source text has incompatible rights evidence"
                        )
                if project.release.schema_version == 3:
                    if method == "normalized":
                        for i in ids:
                            representative[i] = selected
                else:
                    for i in ids[1:]:
                        representative[i] = min(representative[i], ids[0])

        if project.release.schema_version == 3:
            # Raw, normalized, page-URL and paper-ID groups can be connected
            # transitively. Resolve each connected component once: a chain of
            # locally chosen representatives would otherwise introduce cycles.
            parents = {identifier: identifier for identifier in by_id}

            def root_of(identifier: str) -> str:
                while parents[identifier] != identifier:
                    parents[identifier] = parents[parents[identifier]]
                    identifier = parents[identifier]
                return identifier

            for group in duplicate_groups:
                first = root_of(group["origins"][0])
                for identifier in group["origins"][1:]:
                    parents[root_of(identifier)] = first
            components: dict[str, list[str]] = defaultdict(list)
            for identifier in by_id:
                components[root_of(identifier)].append(identifier)
            for ids in components.values():
                heldout = {by_id[i]["split"] for i in ids} - {"train"}
                if len(heldout) > 1:
                    diagnostic = {
                        "duplicates": duplicate_groups,
                        "rejected": rejected,
                        "error": "validation/test page or paper origin overlap",
                    }
                    _json(staging / "audit.json", diagnostic)
                    _json(root / "diagnostics" / f"{build_id}.json", diagnostic)
                    raise ValueError("validation/test page or paper origin overlap")
                try:
                    _require_consistent_cleaning_decisions([by_id[i] for i in ids])
                except ValueError as error:
                    diagnostic = {
                        "duplicates": duplicate_groups,
                        "rejected": rejected,
                        "error": str(error),
                    }
                    _json(staging / "audit.json", diagnostic)
                    _json(root / "diagnostics" / f"{build_id}.json", diagnostic)
                    raise
                priority = next(iter(heldout)) if heldout else None
                selected = min(
                    ids,
                    key=lambda i: (
                        by_id[i]["split"] != priority if priority else False,
                        source_rank[by_id[i]["source_id"]],
                        by_id[i]["document_kind"] != "paper",
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
        for doc in documents:
            doc["representative_id"] = representative[doc["document_id"]]
            doc["drop_reason"] = (
                "metadata_only_front_matter"
                if (doc.get("structure") or {}).get("decision")
                == "exclude_lm_metadata_only"
                else (
                    "contaminated_heldout"
                    if project.release.schema_version == 3
                    and doc["split"] == "train"
                    and by_id[doc["representative_id"]]["split"] != "train"
                    else "duplicate"
                )
                if doc["representative_id"] != doc["document_id"]
                else None
            )
        kept = [d for d in documents if not d["drop_reason"]]
        stages: list[dict[str, Any]] = []
        lexical: list[dict[str, Any]] = []
        semantic: list[dict[str, Any]] = []
        scenarios: list[dict[str, Any]] = []
        generations: list[dict[str, Any]] = []
        chats: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        template_splits: dict[str, str] = {}
        world_splits: dict[str, str] = {}
        for transform in transforms:
            spec = _model(transform)
            kind, stage_id = spec["kind"], spec["id"]
            params = spec.get("parameters", {})
            inputs = spec.get("inputs", [])
            selected_sources = set(inputs) & {source.id for source in project.sources}
            selected = [
                doc
                for doc in kept
                if not selected_sources or doc["source_id"] in selected_sources
            ]
            if kind == "lm_text":
                output = [
                    {
                        "record_id": d["document_id"],
                        "text": d["text"],
                        "split": d["split"],
                    }
                    for d in selected
                ]
            elif kind == "lexical_candidates":
                terms: dict[str, dict[str, Any]] = {}
                for doc in selected:
                    for term in LEXEME.findall(doc["text"]):
                        item = terms.setdefault(
                            term,
                            {
                                "term": term,
                                "occurrences": 0,
                                "document_ids": set(),
                                "domains": Counter(),
                            },
                        )
                        item["occurrences"] += 1
                        item["document_ids"].add(doc["document_id"])
                        item["domains"].update(doc["domains"])
                output = [
                    {
                        "record_id": digest([stage_id, term]),
                        "term": term,
                        "occurrences": item["occurrences"],
                        "document_ids": sorted(item["document_ids"]),
                        "domain_counts": dict(sorted(item["domains"].items())),
                    }
                    for term, item in sorted(terms.items())
                ]
                for candidate in output:
                    parents = [
                        by_id[identifier] for identifier in candidate["document_ids"]
                    ]
                    record = _lineage(
                        {
                            "record_id": candidate["record_id"],
                            "kind": "lexical_candidate",
                        },
                        parents,
                        stage_id,
                    )
                    splits = {parent["split"] for parent in parents}
                    record["split"] = splits.pop() if len(splits) == 1 else "inventory"
                    lineage.append(record)
                lexical.extend(output)
            elif kind == "semantic_candidates":
                output = []
                for doc in selected:
                    for match in KEY_VALUE.finditer(doc["text"]):
                        passage = match.group(0)
                        row = {
                            "subject": match.group(1),
                            "relation": "has_value",
                            "value": match.group(2),
                            "preconditions": [],
                            "consequences": [],
                            "product": None,
                            "version": doc["source_revision"],
                            "validity_interval": None,
                            "method": "deterministic",
                            "evidence_document_id": doc["document_id"],
                            "evidence_passage": passage,
                            "evidence_span": [match.start(), match.end()],
                            "validation_status": "source_entailed",
                        }
                        row["record_id"] = digest([stage_id, row])
                        output.append(row)
                        lineage.append(
                            _lineage(
                                {
                                    "record_id": row["record_id"],
                                    "kind": "semantic_candidate",
                                },
                                [doc],
                                stage_id,
                            )
                        )
                semantic.extend(output)
            elif kind == "manual_semantic":
                raw = _snapshot_file(
                    project,
                    lock,
                    params["source_id"],
                    params["path"],
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                output = []
                from klab.corpus.jsonl_records import records_from_bytes

                for parsed in records_from_bytes(raw, source=params["path"]):
                    entry = parsed.value
                    doc = by_id[entry["evidence_document_id"]]
                    start, end = entry["evidence_span"]
                    if doc["text"][start:end] != entry["evidence_passage"]:
                        raise ValueError("manual semantic evidence span mismatch")
                    row = {
                        **entry,
                        "method": "manual",
                        "validation_status": "source_entailed",
                    }
                    row["record_id"] = digest([stage_id, row])
                    lineage.append(
                        _lineage(
                            {
                                "record_id": row["record_id"],
                                "kind": "semantic_candidate",
                            },
                            [doc],
                            stage_id,
                        )
                    )
                    output.append(row)
                semantic.extend(output)
            elif kind == "deterministic_scenarios":
                generator = params["generator"]
                if generator not in GENERATORS:
                    raise ValueError("unregistered scenario generator")
                generator_version = generator.rsplit("_v", 1)[-1]
                if spec["version"] != generator_version:
                    raise ValueError("scenario generator transform version mismatch")
                if "seed_ranges" in params:
                    if generator == "pathlib_path_suffix_v1" or set(params) != {
                        "generator",
                        "seed_ranges",
                    }:
                        raise ValueError("invalid compact generator declaration")
                    ranges = params["seed_ranges"]
                    if not isinstance(ranges, list) or not ranges:
                        raise ValueError("seed_ranges must be a nonempty list")
                    seeds, families, templates = [], [], []
                    for interval in ranges:
                        if not isinstance(interval, dict) or set(interval) != {
                            "start",
                            "end",
                            "scenario_family",
                            "template_family",
                        }:
                            raise ValueError("invalid generator seed range")
                        start, end = interval["start"], interval["end"]
                        if (
                            type(start) is not int
                            or type(end) is not int
                            or not 0 <= start <= end <= 999
                        ):
                            raise ValueError("invalid inclusive generator seed range")
                        family, template = (
                            interval["scenario_family"],
                            interval["template_family"],
                        )
                        if (
                            not isinstance(family, str)
                            or not family
                            or not isinstance(template, str)
                            or not template
                        ):
                            raise ValueError("missing scenario/template family")
                        for seed in range(start, end + 1):
                            seeds.append(seed)
                            families.append(family)
                            templates.append(template)
                    if len(set(seeds)) != len(seeds):
                        raise ValueError("overlapping generator seed ranges")
                    if len(seeds) != 1000 or set(seeds) != set(range(1000)):
                        raise ValueError(
                            "compact generator declaration must partition all 0–999 seeds"
                        )
                else:
                    seeds, families = params["world_seeds"], params["scenario_families"]
                    templates = (
                        params.get("template_families")
                        if generator != "pathlib_path_suffix_v1"
                        else None
                    )
                if (
                    not seeds
                    or len(seeds) != len(families)
                    or len(set(seeds)) != len(seeds)
                    or (
                        generator != "pathlib_path_suffix_v1"
                        and (
                            not isinstance(templates, list)
                            or len(templates) != len(seeds)
                        )
                    )
                ):
                    raise ValueError("invalid scenario generator configuration")
                declared = [
                    s
                    for s in project.sources
                    if s.kind == "deterministic_generator"
                    and s.acquisition.generator == generator
                    and s.acquisition.generator_version == generator_version
                ]
                if generator != "pathlib_path_suffix_v1" and not declared:
                    raise ValueError(
                        "scenario generator has no pinned source declaration"
                    )
                output = [
                    _path_scenario(seed, family, stage_id)
                    if generator == "pathlib_path_suffix_v1"
                    else _scenario(generator, seed, family, template, stage_id)
                    for seed, family, template in zip(
                        seeds, families, templates or [None] * len(seeds), strict=True
                    )
                ]
                for row in output:
                    row["split"] = _split(row, policy)
                    if generator != "pathlib_path_suffix_v1":
                        seed = row["world_seed"]
                        expected = (
                            "train"
                            if 0 <= seed <= 799
                            else "validation"
                            if 800 <= seed <= 899
                            else "test"
                            if 900 <= seed <= 999
                            else None
                        )
                        if row["split"] != expected:
                            raise ValueError(
                                "scenario seed and explicit family split disagree"
                            )
                        for register, key in (
                            (template_splits, "template_family_id"),
                            (world_splits, "generator_world_id"),
                        ):
                            identifier = row[key]
                            if (
                                identifier in register
                                and register[identifier] != expected
                            ):
                                raise ValueError(f"{key} leaks across splits")
                            register[identifier] = expected
                    ScenarioRecord.model_validate(row)
                    lineage.append(
                        {
                            **_lineage(
                                {
                                    "record_id": row["scenario_id"],
                                    "kind": "scenario",
                                    "validation_status": "oracle_verified",
                                },
                                [],
                                stage_id,
                                family=row["scenario_family_id"],
                                world=row["generator_world_id"],
                                template=row.get("template_family_id")
                                or digest([stage_id, "path_suffix_prompt_v1"]),
                            ),
                            "scenario_id": row["scenario_id"],
                            "generator_identity": digest(
                                [row["generator_id"], row["generator_version"]]
                            ),
                            "oracle_identity": digest(
                                [row["world_state"], row["oracle_answer"]]
                            ),
                        }
                    )
                scenarios.extend(output)
            elif kind == "source_qa":
                if spec["version"] != "1" or params != {"extractor": "literal_span_v1"}:
                    raise ValueError("source_qa requires version 1 literal_span_v1")
                output = []
                for doc in selected:
                    matches = list(KEY_VALUE.finditer(doc["text"]))
                    if not matches:
                        rejected.append(
                            {
                                "source_id": doc["source_id"],
                                "document_id": doc["document_id"],
                                "reason": "no literal key/value answer in normalized source",
                            }
                        )
                    counts = Counter(match.group(1) for match in matches)
                    char_cursor = 0
                    byte_cursor = 0
                    for match in matches:
                        byte_start = byte_cursor + len(
                            doc["text"][char_cursor : match.start()].encode("utf-8")
                        )
                        byte_end = byte_start + len(match.group(0).encode("utf-8"))
                        char_cursor, byte_cursor = match.end(), byte_end
                        key, answer = match.group(1), match.group(2).strip()
                        if (
                            counts[key] != 1
                            or not answer
                            or len(answer) > 200
                            or len(key) > 100
                        ):
                            rejected.append(
                                {
                                    "source_id": doc["source_id"],
                                    "document_id": doc["document_id"],
                                    "reason": "ambiguous or unbounded literal source answer",
                                    "key": key,
                                }
                            )
                            continue
                        passage = match.group(0)
                        for shape_id, question in (
                            (
                                "source_grounded_qa",
                                f"According to this source, what value is set for {key}?",
                            ),
                            (
                                "paraphrased_qa",
                                f"In the cited configuration, give the value of {key}.",
                            ),
                        ):
                            evidence_row = {
                                "document_id": doc["document_id"],
                                "passage": passage,
                                "span": [match.start(), match.end()],
                                "byte_span": [byte_start, byte_end],
                                "answer": answer,
                            }
                            parsed = {
                                "question": f"{question}\nSource [{doc['document_id']}]: {passage}",
                                "answer": answer,
                                "citation_id": doc["document_id"],
                                "shape": shape_id,
                            }
                            record = {
                                "schema_version": 1,
                                "request_id": digest(
                                    [
                                        stage_id,
                                        doc["document_id"],
                                        match.start(),
                                        shape_id,
                                    ]
                                ),
                                "generator": {
                                    "id": "source_qa_v1",
                                    "version": "1",
                                    "extractor": "literal_span_v1",
                                    "implementation_sha256": identity[
                                        "implementation_sha256"
                                    ],
                                },
                                "source_document_ids": [doc["document_id"]],
                                "raw_output": canonical_json(parsed).decode(),
                                "parsed_output": parsed,
                                "validation_status": "source_entailed",
                                "rejection_reason": None,
                                "evidence": evidence_row,
                                "seed": 0,
                                "transform_id": stage_id,
                                "split": doc["split"],
                            }
                            record["generator_identity"] = digest(record["generator"])
                            record["record_id"] = digest(record)
                            GenerationRecord.model_validate(record)
                            output.append(record)
                            lineage.append(
                                {
                                    **_lineage(
                                        {
                                            "record_id": record["record_id"],
                                            "kind": "generation",
                                            "validation_status": "source_entailed",
                                        },
                                        [doc],
                                        stage_id,
                                        template=digest(
                                            [
                                                "literal_span_v1",
                                                shape_id,
                                                doc["source_family"],
                                            ]
                                        ),
                                    ),
                                    "generation_id": record["record_id"],
                                    "generator_identity": record["generator_identity"],
                                }
                            )
                generations.extend(output)
            elif kind == "inference_qa":
                required = (
                    "generator_source_id",
                    "backend",
                    "provider",
                    "model",
                    "model_revision",
                    "prompt_template_path",
                    "responses_path",
                    "max_input_chars",
                    "generation_config",
                )
                if (
                    any(not params.get(key) for key in required)
                    or params["backend"] != "recorded_responses"
                ):
                    raise ValueError("incomplete recorded inference generator")
                source_id = params["generator_source_id"]
                template = _snapshot_file(
                    project,
                    lock,
                    source_id,
                    params["prompt_template_path"],
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                response_bytes = _snapshot_file(
                    project,
                    lock,
                    source_id,
                    params["responses_path"],
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                output = []
                from klab.corpus.jsonl_records import records_from_bytes

                responses = [
                    record.value
                    for record in records_from_bytes(
                        response_bytes, source=params["responses_path"]
                    )
                ]
                adapter = RecordedResponses(responses)
                replay: Generator = adapter
                for response in responses:
                    request_id = response["request_id"]
                    parents = [by_id[item] for item in response["source_document_ids"]]
                    prompt = template.decode("utf-8").format(
                        passages="\n".join(
                            f"[{d['document_id']}] {d['text'][: params['max_input_chars']]}"
                            for d in parents
                        )
                    )
                    request = {
                        "request_id": request_id,
                        "source_document_ids": response["source_document_ids"],
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        **{
                            name: params[name]
                            for name in ("provider", "model", "model_revision")
                        },
                    }
                    try:
                        raw_output, response = replay.generate(request)
                    except ValueError as exc:
                        _json(
                            root / "diagnostics" / f"{build_id}.json",
                            {
                                "error": str(exc),
                                "request_id": request_id,
                                "source_document_ids": response["source_document_ids"],
                                "raw_output": response.get("raw_output"),
                            },
                        )
                        raise
                    generator = {
                        "adapter_id": adapter.adapter_id,
                        "adapter_version": adapter.adapter_version,
                        "adapter_module_sha256": identity["implementation_sha256"],
                        "provider": params["provider"],
                        "model": params["model"],
                        "model_revision": params["model_revision"],
                        "template_id": params["prompt_template_path"],
                        "template_sha256": hashlib.sha256(template).hexdigest(),
                        "generation_config": params["generation_config"],
                        "seed": response["seed"],
                        "raw_output_sha256": hashlib.sha256(
                            raw_output.encode()
                        ).hexdigest(),
                    }
                    try:
                        parsed = json.loads(raw_output)
                        answer, cited = parsed["answer"], parsed["citation_id"]
                        if (
                            not isinstance(answer, str)
                            or not answer.strip()
                            or cited not in {p["document_id"] for p in parents}
                        ):
                            raise ValueError("invalid answer/citation")
                        parent = by_id[cited]
                        start = parent["text"][: params["max_input_chars"]].find(answer)
                        status = "source_entailed" if start >= 0 else "unverified"
                        grounding_evidence = (
                            {
                                "document_id": cited,
                                "passage": answer,
                                "span": [start, start + len(answer)],
                            }
                            if start >= 0
                            else None
                        )
                        reason = (
                            None
                            if status == "source_entailed"
                            else "answer absent from cited passage"
                        )
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        json.JSONDecodeError,
                    ) as exc:
                        parsed, status, reason, grounding_evidence = (
                            None,
                            "rejected",
                            str(exc),
                            None,
                        )
                    record = {
                        "schema_version": 1,
                        "request_id": request_id,
                        "generator": generator,
                        "generator_identity": digest(generator),
                        "source_document_ids": response["source_document_ids"],
                        "raw_output": raw_output,
                        "parsed_output": parsed,
                        "validation_status": status,
                        "rejection_reason": reason,
                        "evidence": grounding_evidence,
                        "seed": response["seed"],
                        "transform_id": stage_id,
                    }
                    record["record_id"] = digest(record)
                    record["split"] = _split(
                        _lineage(
                            {"record_id": record["record_id"], "kind": "generation"},
                            parents,
                            stage_id,
                        ),
                        policy,
                    )
                    GenerationRecord.model_validate(record)
                    output.append(record)
                    if status == "rejected":
                        rejected.append(
                            {
                                "record_id": record["record_id"],
                                "request_id": request_id,
                                "raw_output": raw_output,
                                "generator_identity": record["generator_identity"],
                                "source_document_ids": record["source_document_ids"],
                                "reason": reason,
                            }
                        )
                    lineage.append(
                        {
                            **_lineage(
                                {
                                    "record_id": record["record_id"],
                                    "kind": "generation",
                                    "validation_status": status,
                                },
                                parents,
                                stage_id,
                                template=hashlib.sha256(template).hexdigest(),
                            ),
                            "generation_id": record["record_id"],
                            "generator_identity": record["generator_identity"],
                        }
                    )
                generations.extend(output)
            elif kind in {"chat_sft", "tool_episode"}:
                output = []
                accepted = set(
                    _model(project.release).get(
                        "accepted_generation_statuses",
                        ["source_entailed", "oracle_verified", "human_reviewed"],
                    )
                )
                for scenario in scenarios:
                    messages = _scenario_messages(scenario, kind == "tool_episode")
                    row = {
                        "record_id": digest(
                            [stage_id, scenario["scenario_id"], messages]
                        ),
                        "kind": kind,
                        "format_version": 2,
                        "loss_mode": "assistant_only",
                        "messages": messages,
                        "scenario_id": scenario["scenario_id"],
                        "validation_status": "oracle_verified",
                        **(
                            {"semantic_shape": _scenario_shape(scenario, kind)}
                            if scenario["generator_id"] != "pathlib_path_suffix_v1"
                            and kind == "chat_sft"
                            else {}
                        ),
                        "split": scenario["split"],
                    }
                    output.append(row)
                    lineage.append(
                        {
                            **_lineage(
                                row,
                                [],
                                stage_id,
                                family=scenario["scenario_family_id"],
                                world=scenario["generator_world_id"],
                                template=scenario.get("template_family_id")
                                or digest([stage_id, kind]),
                            ),
                            "scenario_id": scenario["scenario_id"],
                            "generator_identity": digest(
                                [
                                    scenario["generator_id"],
                                    scenario["generator_version"],
                                ]
                            ),
                            "oracle_identity": digest(
                                [scenario["world_state"], scenario["oracle_answer"]]
                            ),
                        }
                    )
                    if kind == "tool_episode":
                        tools.append(
                            ToolEpisode.model_validate(
                                {
                                    "schema_version": 1,
                                    "goal": messages[0]["content"],
                                    "tool_call": messages[1]["tool_calls"][0],
                                    "tool_result": messages[2]["content"],
                                    "interpretation": "Inert declared-world evidence, not a live action",
                                    "next_bounded_action": (
                                        scenario.get("oracle_receipt") or {}
                                    ).get("next_diagnostic", "Compare the path suffix"),
                                    "verification": (
                                        scenario.get("oracle_receipt") or {}
                                    ).get("verification", scenario["oracle_answer"]),
                                    "final_response": messages[-1]["content"],
                                    "source_document_ids": [],
                                    "scenario_id": scenario["scenario_id"],
                                    "transform_id": stage_id,
                                    "record_id": row["record_id"],
                                }
                            ).model_dump(mode="json")
                        )
                if kind == "chat_sft":
                    generation_lineage = {
                        item["record_id"]: item
                        for item in lineage
                        if item["record_kind"] == "generation"
                    }
                    for gen in generations:
                        if (
                            gen["validation_status"] not in accepted
                            or gen["validation_status"] == "rejected"
                        ):
                            continue
                        parsed = gen["parsed_output"]
                        row = {
                            "record_id": digest([stage_id, gen["record_id"]]),
                            "kind": kind,
                            "format_version": 2,
                            "loss_mode": "assistant_only",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": parsed.get("question")
                                    or f"Based on source passage {parsed['citation_id']}, answer the question.",
                                },
                                {"role": "assistant", "content": parsed["answer"]},
                            ],
                            "generation_id": gen["record_id"],
                            **(
                                {"semantic_shape": parsed["shape"]}
                                if parsed.get("shape")
                                else {}
                            ),
                            "validation_status": gen["validation_status"],
                            "split": gen["split"],
                        }
                        output.append(row)
                        source_lineage = generation_lineage.get(gen["record_id"])
                        if source_lineage is None:
                            raise ValueError("missing generation lineage")
                        lineage.append(
                            {
                                **source_lineage,
                                "record_id": row["record_id"],
                                "record_kind": kind,
                                "transform_id": stage_id,
                            }
                        )
                    requested_shapes = params.get("semantic_shapes", [])
                    if params.get("include_semantic_qa", False):
                        requested_shapes = list(
                            dict.fromkeys([*requested_shapes, "direct_qa"])
                        )
                    allowed_shapes = {
                        "direct_qa",
                        "troubleshooting_scenario",
                        "decision_record",
                    }
                    if not isinstance(requested_shapes, list) or (
                        set(requested_shapes) - allowed_shapes
                    ):
                        raise ValueError("unknown semantic chat shape")
                    for fact in semantic:
                        doc = by_id[fact["evidence_document_id"]]
                        if (
                            selected_sources
                            and doc["source_id"] not in selected_sources
                        ):
                            continue
                        key, value = fact["subject"], fact["value"]
                        if value not in fact["evidence_passage"]:
                            raise ValueError(
                                "semantic answer absent from source passage"
                            )
                        for shape_id in requested_shapes:
                            question = {
                                "direct_qa": f"What value is recorded for {key}?",
                                "troubleshooting_scenario": (
                                    f"Checking the recorded configuration: which setting has value {value}?"
                                ),
                                "decision_record": (
                                    f"For the recorded setting {key}, which recorded value should be used?"
                                ),
                            }[shape_id]
                            answer = {
                                "direct_qa": value,
                                "troubleshooting_scenario": key,
                                "decision_record": value,
                            }[shape_id]
                            messages = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer},
                            ]
                            row = {
                                "record_id": digest(
                                    [stage_id, fact["record_id"], shape_id, messages]
                                ),
                                "kind": kind,
                                "format_version": 2,
                                "loss_mode": "assistant_only",
                                "messages": messages,
                                "semantic_id": fact["record_id"],
                                "semantic_shape": shape_id,
                                "validation_status": "source_entailed",
                                "split": doc["split"],
                            }
                            output.append(row)
                            lineage.append(
                                {
                                    **_lineage(
                                        row,
                                        [doc],
                                        stage_id,
                                        template=digest(
                                            [stage_id, shape_id, "semantic_v1"]
                                        ),
                                    ),
                                    "semantic_id": fact["record_id"],
                                    "generator_identity": digest(
                                        ["semantic_chat_v1", stage_id, shape_id]
                                    ),
                                }
                            )
                chats.extend(output)
            else:
                raise ValueError(f"unregistered transform: {kind}")
            _jsonl(staging / "stages" / f"{stage_id}.jsonl", output)
            stages.append(
                {
                    "id": stage_id,
                    "kind": kind,
                    "version": spec["version"],
                    "parameters_sha256": digest(params),
                    "inputs": inputs,
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "output_sha256": sha256_file(
                        staging / "stages" / f"{stage_id}.jsonl"
                    ),
                    "output_count": len(output),
                    "output_id": digest([stage_id, output]),
                }
            )
        for doc in documents:
            lineage.append(
                {
                    "record_id": doc["document_id"],
                    "record_kind": "document",
                    "transform_id": doc.get("metadata", {}).get(
                        "normalizer", "normalizer-nfc-markdown-v1"
                    ),
                    "parent_document_ids": [doc["document_id"]],
                    "original_parent_document_ids": [doc["document_id"]],
                    "representative_id": doc["representative_id"],
                    "drop_reason": doc["drop_reason"],
                    "source_family_ids": [doc["source_family"]],
                    "split": doc["split"],
                }
            )
        for item in lineage:
            if "split" not in item:
                item["split"] = _split(item, policy)
            for parent in item["parent_document_ids"]:
                if parent not in by_id:
                    raise ValueError("dangling parent document")
            item["representative_parent_document_ids"] = sorted(
                {representative[p] for p in item["parent_document_ids"]}
            )
        payloads = {
            row["record_id"]: row
            for collection in (lexical, semantic, generations, chats)
            for row in collection
        }
        payloads.update({row["scenario_id"]: row for row in scenarios})
        source_origins = {source.id: source.origin for source in project.sources}
        for item in lineage:
            kind = item["record_kind"]
            record_id = item["record_id"]
            payload = by_id[record_id] if kind == "document" else payloads[record_id]
            if kind == "document":
                origin = (
                    HUMAN_ORIGIN
                    if source_origins[payload["source_id"]] == HUMAN_ORIGIN
                    else SOURCE_ORIGIN
                )
                shape_id = "raw_document"
                item["domains"] = payload["domains"]
                status, method, proof = (
                    "schema_validated",
                    payload.get("metadata", {}).get("normalizer", "normalizer_v1"),
                    {"schema_id": "normalized_document_v1"},
                )
                rendered = rendered_digest(payload["text"], text=True)
            else:
                source_ids = {
                    by_id[doc_id]["source_id"] for doc_id in item["parent_document_ids"]
                }
                origin = (
                    DETERMINISTIC_ORIGIN
                    if kind in {"scenario", "tool_episode"}
                    or (kind == "chat_sft" and "scenario_id" in payload)
                    else MULTI_SOURCE_ORIGIN
                    if len(source_ids) > 1
                    else INFERENCE_ORIGIN
                    if kind == "generation"
                    and not source_ids
                    or kind == "chat_sft"
                    and "generation_id" in payload
                    and not source_ids
                    else DERIVED_ORIGIN
                )
                shape_id = {
                    "lexical_candidate": "lexical_inventory",
                    "semantic_candidate": "definition",
                    "scenario": _scenario_shape(payload, "scenario"),
                    "generation": (payload.get("parsed_output") or {}).get(
                        "shape", "direct_qa"
                    ),
                    "tool_episode": "tool_trace",
                    "chat_sft": payload.get(
                        "semantic_shape",
                        "troubleshooting_scenario"
                        if "scenario_id" in payload
                        else "direct_qa",
                    ),
                }[kind]
                evidence_row = (
                    payloads[payload["semantic_id"]]
                    if "semantic_id" in payload
                    else payloads[payload["generation_id"]]
                    if "generation_id" in payload
                    else payload
                )
                status = evidence_row.get(
                    "validation_status", item["validation_status"]
                )
                if status == "source_entailed":
                    proof = {
                        "document_id": evidence_row.get("evidence_document_id")
                        or evidence_row["evidence"]["document_id"],
                        "passage": evidence_row.get("evidence_passage")
                        or evidence_row["evidence"]["passage"],
                        "span": evidence_row.get("evidence_span")
                        or evidence_row["evidence"]["span"],
                        "answer": (
                            payload["messages"][-1]["content"]
                            if kind == "chat_sft"
                            else evidence_row["parsed_output"]["answer"]
                            if kind == "generation"
                            else evidence_row["value"]
                        ),
                        **(
                            {"byte_span": evidence_row["evidence"]["byte_span"]}
                            if evidence_row.get("evidence")
                            and "byte_span" in evidence_row["evidence"]
                            else {}
                        ),
                    }
                    method = "verbatim_source_span"
                elif status == "oracle_verified":
                    scenario = (
                        payloads[payload["scenario_id"]]
                        if "scenario_id" in payload
                        else payload
                    )
                    proof = {
                        "oracle_identity": digest(
                            [scenario["world_state"], scenario["oracle_answer"]]
                        ),
                        "generator_world_id": scenario["generator_world_id"],
                        "oracle_answer": scenario["oracle_answer"],
                        "oracle_implementation": identity["implementation_sha256"],
                        "oracle_version": scenario["generator_version"],
                        "interpreter": scenario["interpreter"],
                        "world_state": scenario["world_state"],
                        "actual_result": scenario["oracle_answer"],
                        "comparison_status": "match",
                        **(
                            {"receipt": scenario["oracle_receipt"]}
                            if scenario.get("oracle_receipt")
                            else {}
                        ),
                    }
                    method = (
                        "pathlib_pureposix_oracle_v1"
                        if scenario["generator_id"] == "pathlib_path_suffix_v1"
                        else scenario["generator_id"]
                    )
                elif status in {"rejected", "unverified"}:
                    proof = {
                        "reason": evidence_row.get("rejection_reason")
                        or "response not entailed by cited passage"
                    }
                    method = "recorded_response_validation"
                else:
                    proof = {"schema_id": f"{kind}_v1"}
                    method = "schema_validation"
                if kind in {"chat_sft", "tool_episode"}:
                    from klab.data.conversations import _v2_document

                    chat_row = {
                        field: payload[field]
                        for field in ("format_version", "loss_mode", "messages")
                    }
                    rendered = rendered_digest(
                        _v2_document(chat_row, Path("<corpus-record>"), 1).text,
                        text=True,
                    )
                else:
                    rendered = rendered_digest(payload)
            item["origin"] = origin
            item["modalities"] = sorted(
                {by_id[parent]["modality"] for parent in item["parent_document_ids"]}
                or {"text"}
            )
            item["origin_schema_version"] = 1
            item["shape"] = shape_for_record(
                kind,
                shape_id,
                domains=item.get("domains", []),
                parent_document_ids=item["parent_document_ids"],
                scenario_id=item.get("scenario_id"),
            )
            item["verification"] = verification(status, method, proof)
            item["validation_status"] = status
            item["rendered_sha256"] = rendered
            if kind == "chat_sft" and "generation_id" in payload:
                item["generator_identity"] = payloads[payload["generation_id"]][
                    "generator_identity"
                ]
                item["generator"] = payloads[payload["generation_id"]]["generator"]
            elif kind == "generation":
                item["generator"] = payload["generator"]
            validate_lineage(item, documents=by_id)
        _jsonl(staging / "documents.jsonl", documents)
        _jsonl(staging / "spans.jsonl", evidence)
        _jsonl(staging / "lineage.jsonl", sorted(lineage, key=lambda r: r["record_id"]))
        _jsonl(staging / "lexical/candidates.jsonl", lexical)
        _jsonl(staging / "semantic/candidates.jsonl", semantic)
        _jsonl(staging / "scenarios.jsonl", scenarios)
        _jsonl(staging / "generations.jsonl", generations)
        _jsonl(staging / "tool_episodes.jsonl", tools)
        _jsonl(staging / "rejected.jsonl", rejected)
        _jsonl(staging / "chat/records.jsonl", chats)
        ledger_by_id = {item["record_id"]: item for item in lineage}
        release_spec = _model(project.release)
        include_shapes = release_spec.get("include_shapes")
        include_origins = release_spec.get("include_origins")

        def included(record_id: str) -> bool:
            item = ledger_by_id[record_id]
            return (
                include_shapes is None or item["shape"]["id"] in include_shapes
            ) and (include_origins is None or item["origin"] in include_origins)

        for split in SPLITS:
            chosen_lm = [
                d
                for d in kept
                if d["split"] == split
                and any(
                    t.kind == "lm_text" and (not t.inputs or d["source_id"] in t.inputs)
                    for t in transforms
                )
                and included(d["document_id"])
            ]
            chosen_chat = [
                row
                for row in chats
                if row["split"] == split and included(row["record_id"])
            ]
            _jsonl(
                staging / "lm" / f"{split}.jsonl",
                [{"text": d["text"]} for d in chosen_lm],
            )
            _jsonl(
                staging / "lm" / f"{split}.lineage.jsonl",
                [{"record_id": d["document_id"], "split": split} for d in chosen_lm],
            )
            _jsonl(
                staging / "chat" / f"{split}.jsonl",
                [
                    {k: row[k] for k in ("format_version", "loss_mode", "messages")}
                    for row in chosen_chat
                ],
            )
            _jsonl(
                staging / "chat" / f"{split}.lineage.jsonl",
                [
                    {"record_id": row["record_id"], "split": split}
                    for row in chosen_chat
                ],
            )
        fraction = project.release.fraction
        if fraction is not None:
            from tokenizers import Tokenizer

            from klab.corpus.project import project_path
            from klab.corpus.selection import select_fraction
            from klab.data.conversations import iter_rendered_conversations

            tokenizer_path = project_path(project.root, fraction.tokenizer_path)
            if sha256_file(tokenizer_path) != fraction.tokenizer_sha256.lower():
                raise ValueError("fraction tokenizer SHA-256 changed during build")
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            candidates = []
            for view in ("lm", "chat"):
                selected_view = getattr(project.release, view)
                if (
                    not selected_view.selected
                    or "train" not in selected_view.training_splits
                ):
                    continue
                payload_path = staging / view / "train.jsonl"
                links = _rows(staging / view / "train.lineage.jsonl")
                texts = (
                    [row["text"] for row in _rows(payload_path)]
                    if view == "lm"
                    else [row.text for row in iter_rendered_conversations(payload_path)]
                )
                for link, text in zip(links, texts, strict=True):
                    record = ledger_by_id[link["record_id"]]
                    candidates.append(
                        {
                            "record_id": link["record_id"],
                            "tokens": len(tokenizer.encode(text).ids),
                            "origin": record["origin"],
                            "source_family_ids": record["source_family_ids"],
                        }
                    )
            chosen_ids = select_fraction(
                candidates, fraction.generated_share, fraction.train_tokens
            )
            for view in ("lm", "chat"):
                selected_view = getattr(project.release, view)
                if (
                    not selected_view.selected
                    or "train" not in selected_view.training_splits
                ):
                    continue
                path = staging / view / "train.jsonl"
                links_path = staging / view / "train.lineage.jsonl"
                pairs = [
                    (row, link)
                    for row, link in zip(_rows(path), _rows(links_path), strict=True)
                    if link["record_id"] in chosen_ids
                ]
                _jsonl(path, [row for row, _ in pairs])
                _jsonl(links_path, [link for _, link in pairs])
        from klab.data.conversations import iter_rendered_conversations

        for split in SPLITS:
            list(iter_rendered_conversations(staging / "chat" / f"{split}.jsonl"))
        warnings = []
        leakage = []
        for field in (
            "parent_document_ids",
            "source_family_ids",
            "scenario_family_id",
            "generator_world_id",
            "template_family_id",
        ):
            families: dict[str, set[str]] = defaultdict(set)
            for item in lineage:
                if item["split"] == "inventory":
                    continue
                value = item.get(field)
                for family in (
                    value if isinstance(value, list) else ([value] if value else [])
                ):
                    families[family].add(item["split"])
            for family, assignments in sorted(families.items()):
                if len(assignments) < 2:
                    continue
                overlap = {
                    "family_type": field,
                    "family_id": family,
                    "splits": sorted(assignments),
                }
                leakage.append(overlap)
                if (
                    (policy["unit"] == "document" and field == "parent_document_ids")
                    or (
                        policy["unit"]
                        in {"source_document_family", "source_repository"}
                        and field == "source_family_ids"
                    )
                    or (
                        policy["unit"] == "scenario_family"
                        and field == "scenario_family_id"
                    )
                    or (
                        policy["unit"] == "generator_world"
                        and field == "generator_world_id"
                    )
                    or (
                        policy["unit"] == "template_family"
                        and field == "template_family_id"
                    )
                ):
                    _json(
                        root / "diagnostics" / f"{build_id}.json",
                        {"error": "shared split policy unit", "leakage": leakage},
                    )
                    raise ValueError(f"split policy unit {family} spans splits")
                warnings.append(
                    f"{field} {family} overlaps {', '.join(sorted(assignments))}"
                )
        audit = {
            "duplicates": duplicate_groups,
            "rejected": rejected,
            "warnings": warnings,
            "drop_count": len(documents) - len(kept),
            "leakage": leakage,
        }
        _json(staging / "audit.json", audit)
        _json(staging / "splits.json", _model(project.splits))
        sources = [
            {
                "id": s.id,
                "kind": s.kind,
                "canonical_uri": s.canonical_uri,
                "revision": s.revision,
                "license": s.license,
                "redistribution": s.rights.redistribution_mode
                if s.rights
                else s.redistribution,
                **(
                    {
                        "license_url": s.license_url,
                        "rights_policy": s.rights.model_dump(mode="json"),
                    }
                    if s.rights
                    else {}
                ),
                **(
                    {"explicit_training_restriction": s.explicit_training_restriction}
                    if s.schema_version == 3
                    else {}
                ),
                "origin": s.origin,
                "source_family": s.source_family,
                "snapshot_sha256": lock["sources"].get(s.id, {}).get("snapshot_sha256"),
                "reproducibility_class": (
                    "generator_dependent"
                    if s.kind == "inference_generator"
                    else "script_reproducible_not_redistributed"
                    if (s.rights.redistribution_mode if s.rights else s.redistribution)
                    in {
                        "reference_only",
                        "derived_only",
                        "unknown",
                        "rejected",
                        "metadata_reconstruction_only",
                        "not_redistributable",
                        "review_required",
                    }
                    else "fully_reproducible"
                ),
            }
            for s in project.sources
        ]
        _json(staging / "sources.json", sources)
        if project.release.schema_version in (2, 3):
            file_decisions = [
                item for item in rights_files if item["role"] == "document"
            ]
            rights_counts = Counter(
                item["rights"]["training_eligibility"] for item in file_decisions
            )
            bytes_by_state: Counter[str] = Counter()
            spdx_counts: Counter[str] = Counter()
            modes: Counter[str] = Counter()
            source_spdx = {s.id: s.rights.spdx_expression for s in project.sources}
            for item in file_decisions:
                decision = item["rights"]
                bytes_by_state[decision["training_eligibility"]] += item["size"]
                spdx_counts[
                    reported_spdx_expression(item, source_spdx[item["source_id"]])
                ] += 1
                modes[decision["redistribution_mode"]] += 1
            rights_report = {
                "schema_version": project.release.schema_version,
                "publication_mode": project.release.publication_mode,
                **(
                    {"training_use_policy": project.release.training_use_policy}
                    if project.release.schema_version == 3
                    else {}
                ),
                "weight_license_status": "separate_analysis_required",
                "sources": sources,
                "files": rights_files,
                "training_eligibility": {
                    state: {
                        "files": rights_counts[state],
                        "source_bytes": bytes_by_state[state],
                        "source_tokens": None,
                        "token_count_reason": "tokenizer_not_declared",
                    }
                    for state in (
                        "eligible",
                        "eligible_with_obligations",
                        "review_required",
                        "ineligible",
                    )
                },
                "spdx_expressions": dict(spdx_counts),
                "redistribution_modes": dict(modes),
                "unresolved_rights_files": rights_counts["review_required"],
                "advisory": "Source/derived-data rights and model-weight licensing are separate decisions; not a legal conclusion.",
            }
        else:
            rights_report = {
                "sources": sources,
                "redistribution_classes": dict(
                    Counter(s.redistribution for s in project.sources)
                ),
                "advisory": "Declared attribution and redistribution classifications are inventory metadata, not legal review.",
            }
        _json(staging / "license-report.json", rights_report)
        mixture = _model(project.release).get("mixture", {})
        actual = {
            domain: {
                "documents": sum(domain in d["domains"] for d in kept),
                "utf8_bytes": sum(
                    len(d["text"].encode()) for d in kept if domain in d["domains"]
                ),
                "actual_tokens": None,
                "token_count_reason": "tokenizer_not_declared",
            }
            for domain in sorted(
                set(mixture) | {domain for d in kept for domain in d["domains"]}
            )
        }
        generation_statuses = Counter(g["validation_status"] for g in generations)
        generation_statuses["oracle_verified"] += len(scenarios)
        generation_total = len(generations) + len(scenarios)
        source_counts = Counter(d["source_id"] for d in kept)
        source_scale: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        if project.release.schema_version == 3:
            for doc in kept:
                split = doc["split"]
                counts = source_scale[doc["source_id"]].setdefault(
                    split,
                    {
                        "documents": 0,
                        "utf8_bytes": 0,
                        "characters": 0,
                        "whitespace_words": 0,
                    },
                )
                text = doc["text"]
                counts["documents"] += 1
                counts["utf8_bytes"] += len(text.encode("utf-8"))
                counts["characters"] += len(text)
                counts["whitespace_words"] += len(text.split())
        report = {
            "schema_version": project.release.schema_version,
            "corpus_id": project.config.id,
            "requested_mixture": mixture,
            "actual_mixture": actual,
            "document_count": len(documents),
            "kept_document_count": len(kept),
            "split_counts": {s: sum(d["split"] == s for d in kept) for s in SPLITS},
            "view_counts": {
                view: {
                    split: (
                        _line_count(staging / view / f"{split}.jsonl")
                        if project.release.schema_version == 3
                        else len(_rows(staging / view / f"{split}.jsonl"))
                    )
                    for split in SPLITS
                }
                for view in ("lm", "chat")
            },
            "source_counts": dict(source_counts),
            "source_concentration": {
                source: count / len(kept)
                for source, count in sorted(source_counts.items())
            }
            if kept
            else {},
            "kind_counts": dict(Counter(d["document_kind"] for d in kept)),
            "lexical_count": len(lexical),
            "semantic_count": len(semantic),
            "chat_count": len(chats),
            "generation_statuses": dict(generation_statuses),
            "generation_validation_ratios": {
                status: count / generation_total
                for status, count in sorted(generation_statuses.items())
            }
            if generation_total
            else {},
            "generator_count": len({g["generator_identity"] for g in generations}),
            "scenario_count": len(scenarios),
            "dedupe": {
                "groups": len(duplicate_groups),
                "dropped": len(documents) - len(kept),
            },
            "warnings": warnings,
            "token_count_reason": None
            if fraction is not None
            else "tokenizer_not_declared",
        }
        if project.release.schema_version == 3:
            report["source_scale"] = {
                source: dict(sorted(splits.items()))
                for source, splits in sorted(source_scale.items())
            }
            train_scale = {
                key: sum(
                    per_split.get("train", {}).get(key, 0)
                    for per_split in source_scale.values()
                )
                for key in ("documents", "utf8_bytes", "characters", "whitespace_words")
            }
            report["train_source_scale"] = {
                **train_scale,
                "utf8_bytes_div4_proxy": train_scale["utf8_bytes"] // 4,
                "actual_tokens": None,
                "token_count_reason": "final_tokenizer_not_fitted",
                "proxy_warning": (
                    "Whitespace words and UTF-8 bytes/4 are tokenizer-independent "
                    "descriptors, not measured token counts."
                ),
            }
        if project.release.schema_version in (2, 3):
            report["rights"] = {
                key: value
                for key, value in rights_report.items()
                if key
                in {
                    "training_eligibility",
                    "spdx_expressions",
                    "redistribution_modes",
                    "unresolved_rights_files",
                    "weight_license_status",
                    "publication_mode",
                    "training_use_policy",
                }
            }
        from klab.corpus.measurement import summarize_release

        report["measurement"] = summarize_release(
            staging,
            release_spec=release_spec,
            tokenizer=tokenizer_path if fraction is not None else None,
        )
        _json(staging / "report.json", report)
        (staging / "report.md").write_text(
            f"# Corpus {project.config.id}\n\nDocuments: {len(kept)} retained / {len(documents)} total.\n\n"
            + (
                f"Training tokens: {report['measurement']['training_mixture']['actual_tokens']} with pinned tokenizer {fraction.tokenizer_sha256}.\n"
                if fraction is not None
                else "No tokenizer declared; token counts unavailable.\n"
            ),
            encoding="utf-8",
        )
        _json(
            staging / "build.json",
            {
                "schema_version": 1,
                "build_id": build_id,
                "identity": identity,
                "stages": stages,
                "files": {
                    str(p.relative_to(staging)): {
                        "sha256": sha256_file(p),
                        "size": p.stat().st_size,
                    }
                    for p in sorted(staging.rglob("*"))
                    if p.is_file()
                },
                "snapshots": [
                    {"source_id": key, "sha256": v["snapshot_sha256"]}
                    for key, v in sorted(lock["sources"].items())
                    if v.get("snapshot_path")
                ],
            },
        )
        from klab.engram.packs import _rename_noreplace
        from klab.result_publication import sync_directory

        for entry in sorted(staging.rglob("*"), reverse=True):
            if entry.is_file():
                with entry.open("rb") as handle:
                    os.fsync(handle.fileno())
            elif entry.is_dir():
                sync_directory(entry)
        sync_directory(staging)
        _rename_noreplace(staging, target)
        sync_directory(root)
    return target
