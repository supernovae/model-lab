"""Source and document signals used to index and compare duplicates."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from klab.canonical import canonical_json
from klab.corpus.project import CURATED_SOURCE_KINDS

METADATA_ONLY = "exclude_lm_metadata_only"

_RIGHTS_FIELDS = (
    "training_eligibility",
    "license_references",
    "notices",
    "training_restriction",
)


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


def origin_keys(document: dict[str, Any]) -> tuple[str, ...]:
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
