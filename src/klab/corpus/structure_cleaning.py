"""Conservative, replayable text cleaning for a prospective corpus release.

The source snapshot remains the authority. This transformation removes only
recognised wrappers and records the original line ranges for every removal.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

NORMALIZER_V1 = "normalizer-nfc-markdown-v1"
NORMALIZER_V2 = "normalizer-structure-v2"
NORMALIZER_V3 = "normalizer-structure-v3"

_FRONT_MATTER_KEYS = frozenset(
    {
        "title",
        "weight",
        "categories",
        "category",
        "tags",
        "date",
        "draft",
        "slug",
        "description",
    }
)
_PAGERDUTY_FRONT_MATTER_KEYS = frozenset(
    {"cover", "hero", "hero_alt_text", "style", "pdf"}
)
_GUTENBERG_START = re.compile(
    r"^\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG (?:EBOOK|ETEXT)\b.*\*\*\*\s*$",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"^\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG (?:EBOOK|ETEXT)\b.*\*\*\*\s*$",
    re.IGNORECASE,
)
_GUTENBERG_CREDIT = re.compile(r"^(?:This eBook was )?Produced by\b", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})([^\n]*)$")


def _lines(text: str) -> list[str]:
    """Split physical LF lines, preserving endings and Unicode separators."""
    if not text:
        return []
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _front_matter_end(
    lines: list[str], *, source_id: str, normalizer: str
) -> tuple[int | None, str | None]:
    if not lines or lines[0].strip() != "---":
        return None, None
    closing = next(
        (n for n, line in enumerate(lines[1:81], 1) if line.strip() in {"---", "..."}),
        None,
    )
    if closing is None:
        return None, "unclosed_yaml_header"
    keys: set[str] = set()
    for line in lines[1:closing]:
        value = line.strip()
        if not value or value.startswith(("#", "- ")):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):(?:\s*.*)?", value)
        if match is None:
            return None, "ambiguous_yaml_header"
        keys.add(match.group(1).lower())
    allowed = _FRONT_MATTER_KEYS
    if normalizer == NORMALIZER_V3 and source_id == "kml_scale_pagerduty":
        allowed = allowed | _PAGERDUTY_FRONT_MATTER_KEYS
    if not keys or not keys <= allowed:
        return None, "configuration_or_ambiguous_yaml"
    return closing + 1, None


def clean_structure(
    text: str, *, source_id: str, normalizer: str = NORMALIZER_V2
) -> dict[str, Any]:
    """Return text and structural provenance without rewriting useful syntax.

    Ranges are one-based physical lines within the source passage or JSONL text
    field. The enclosing document source span still identifies the snapshot row
    or file-line interval. Output offsets are character offsets in cleaned text.
    """
    if normalizer not in {NORMALIZER_V2, NORMALIZER_V3}:
        raise ValueError("unsupported structured normalizer")
    normalized = unicodedata.normalize(
        "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = _lines(normalized)
    keep = [True] * len(lines)
    removed: list[dict[str, Any]] = []
    flags: list[str] = []

    def remove(start: int, end: int, reason: str) -> None:
        if start >= end:
            return
        for index in range(start, end):
            keep[index] = False
        removed.append(
            {
                "raw_line_start": start + 1,
                "raw_line_end": end,
                "reason": reason,
                "sha256": hashlib.sha256(
                    "".join(lines[start:end]).encode()
                ).hexdigest(),
            }
        )

    front_end, front_flag = _front_matter_end(
        lines, source_id=source_id, normalizer=normalizer
    )
    if front_flag:
        flags.append(front_flag)
    if front_end is not None:
        remove(0, front_end, "recognized_yaml_front_matter")

    if source_id.startswith("project_gutenberg") or (
        normalizer == NORMALIZER_V3 and source_id == "kml_scale_project_gutenberg"
    ):
        start = next(
            (
                n
                for n, line in enumerate(lines[:300])
                if _GUTENBERG_START.match(line.strip())
            ),
            None,
        )
        end = next(
            (n for n, line in enumerate(lines) if _GUTENBERG_END.match(line.strip())),
            None,
        )
        if start is not None:
            remove(0, start + 1, "gutenberg_header_through_start_marker")
        else:
            first = next((n for n, line in enumerate(lines[:8]) if line.strip()), None)
            if first is not None and _GUTENBERG_CREDIT.match(lines[first].strip()):
                boundary = next(
                    (
                        n
                        for n in range(first + 1, min(first + 12, len(lines)))
                        if not lines[n].strip()
                    ),
                    None,
                )
                if (
                    boundary is not None
                    and sum(len(line) for line in lines[first:boundary]) <= 500
                    and any(line.strip() for line in lines[boundary + 1 :])
                ):
                    remove(0, boundary + 1, "gutenberg_production_credit")
                else:
                    flags.append("ambiguous_gutenberg_credit")
        if start is None and any(
            "project gutenberg" in line.casefold() for line in lines[:80]
        ):
            flags.append("ambiguous_gutenberg_header")
        if end is not None and (start is None or end > start):
            remove(end, len(lines), "gutenberg_footer_from_end_marker")
        elif end is not None:
            flags.append("ambiguous_gutenberg_markers")

    if "\ufffd" in normalized or "\x00" in normalized:
        flags.append("possible_extraction_damage")

    retained: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    output: list[str] = []
    offset = 0
    fence_char: str | None = None
    fence_length = 0
    segment_start: int | None = None
    segment_offset = 0
    for index, line in enumerate(lines):
        if not keep[index]:
            if segment_start is not None:
                retained.append(
                    {
                        "raw_line_start": segment_start + 1,
                        "raw_line_end": index,
                        "output_start": segment_offset,
                        "output_end": offset,
                    }
                )
                segment_start = None
            continue
        if segment_start is None:
            segment_start, segment_offset = index, offset
        stripped = line.rstrip("\n")
        fence = _FENCE.match(stripped)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
                kind = "fence_open"
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char, fence_length = None, 0
                kind = "fence_close"
            else:
                kind = "code"
        elif fence_char is not None:
            kind = "code"
        elif _HEADING.match(stripped):
            kind = "heading"
        elif _LIST.match(stripped):
            kind = "list"
        elif stripped.lstrip().startswith(("$$", "\\[", "\\]")):
            kind = "equation"
        elif stripped.startswith(("    ", "\t")):
            kind = "indented"
        else:
            kind = "prose"
        if kind != "prose":
            blocks.append(
                {
                    "kind": kind,
                    "raw_line": index + 1,
                    "output_start": offset,
                    "output_end": offset + len(line),
                }
            )
        output.append(line)
        offset += len(line)
    if segment_start is not None:
        retained.append(
            {
                "raw_line_start": segment_start + 1,
                "raw_line_end": len(lines),
                "output_start": segment_offset,
                "output_end": offset,
            }
        )
    result = "".join(output)
    metadata_only = front_end is not None and not result.strip()
    if metadata_only:
        # Keep the admitted source row and its identity in the canonical layer;
        # the build excludes it from LM views with an explicit drop reason.
        result = normalized
        retained = [
            {
                "raw_line_start": 1,
                "raw_line_end": len(lines),
                "output_start": 0,
                "output_end": len(result),
            }
        ]
        removed = []
        flags.append("metadata_only_front_matter")
    return {
        "text": result,
        "structure": {
            "normalizer": normalizer,
            "retained_spans": retained,
            "removed_spans": removed,
            "blocks": blocks,
            "flags": flags,
            "decision": "exclude_lm_metadata_only" if metadata_only else "retain",
        },
        "drop_reason": "metadata_only_front_matter" if metadata_only else None,
    }
