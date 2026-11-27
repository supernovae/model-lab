"""Deterministic contact-detail candidates in surviving normalized text."""

import re
import string
import struct
from pathlib import Path

from klab.corpus.normalized_annotations import NormalizedAnnotator, annotate_partition

# Search starts at the literal at-sign. The local-part check looks back at
# most 65 characters, and the host expression bounds every label and its count.
_EMAIL_HOST = re.compile(
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.){1,4}"
    r"[A-Za-z]{2,63}(?![A-Za-z0-9-]|\.[A-Za-z0-9-])"
)
_EMAIL_LOCAL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9_%+-]"
    r"(?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9_%+-])?\Z"
)
_EMAIL_FINAL = string.ascii_letters + string.digits + "_%+-"
# Search the literal final hyphen before checking a bounded number prefix.
_PHONE_END = re.compile(r"-[0-9]{4}(?![A-Za-z0-9])")
_PHONE_PREFIX = re.compile(
    r"(?<![A-Za-z0-9+])(?:\+1[ -])?"
    r"(?:\([2-9][0-9]{2}\) ?|[2-9][0-9]{2}-)"
    r"[2-9][0-9]{2}\Z"
)
PRIVACY_FIELDS = ("email_like", "phone_like")
PRIVACY_ROW = struct.Struct("<32sB")


def _email_like(text):
    end = text.find("@")
    while end >= 0:
        if (
            end
            and text[end - 1] in _EMAIL_FINAL
            and _EMAIL_HOST.match(text, end)
            and _EMAIL_LOCAL.search(text, max(0, end - 65), end)
        ):
            return True
        end = text.find("@", end + 1)
    return False


def _phone_like(text):
    for match in _PHONE_END.finditer(text):
        end = match.start()
        if _PHONE_PREFIX.search(text, max(0, end - 12), end):
            return True
    return False


def privacy_flags(text, mask=3):
    """Evaluate the requested categories with the normalized annotator."""
    email = bool(mask & 1) and _email_like(text)
    phone = bool(mask & 2) and _phone_like(text)
    return email | (phone << 1)


def _render(writer, documents, _source):
    emails = phones = candidates = 0
    for doc_id, row in documents:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("invalid document in normalized partition")  # noqa: TRY004 - invalid serialized artifact
        email = _email_like(text)
        phone = _phone_like(text)
        writer.write(PRIVACY_ROW.pack(doc_id, email | (phone << 1)))
        emails += email
        phones += phone
        candidates += email or phone
    return {"totals": {"email_like": emails, "phone_like": phones, "any": candidates}}


def _decode(values, _summary):
    doc_id, flags = values
    if flags >= 1 << len(PRIVACY_FIELDS):
        raise ValueError("invalid privacy candidate flags")
    return {
        "document_id": doc_id.hex(),
        "candidates": [
            name for bit, name in enumerate(PRIVACY_FIELDS) if flags & (1 << bit)
        ],
    }


_ANNOTATOR = NormalizedAnnotator(
    "privacy-candidates", "privacy_candidates.py", PRIVACY_ROW, _render, _decode
)


def annotate_privacy_candidates(
    work_root: Path, partition: str, *, offset=0, limit=0
) -> dict:
    """Publish or page document-level candidate categories without matched text."""
    return annotate_partition(
        work_root, partition, _ANNOTATOR, offset=offset, limit=limit
    )
