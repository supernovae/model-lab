"""Pinned source adapters for Corpus relations. Workers never evaluate source code."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.corpus.jsonl_records import records_from_path

Identifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/%-]{0,191}$")
]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, StringConstraints(min_length=1)]
Split = Literal["train", "validation", "test"]
ClaimClass = Literal["explicit", "derived", "candidate"]
MAX_ROW_BYTES = 4 * 1024**2


class PinnedRows(StrictModel):
    path: Path
    sha256: Digest
    max_rows: Annotated[int, Field(gt=0, strict=True)]
    max_record_bytes: Annotated[int, Field(gt=0, le=MAX_ROW_BYTES, strict=True)] = (
        1024**2
    )

    def rows(self, base: Path):
        path = base / self.path
        if not path.is_file():
            raise ValueError(f"expected a pinned local JSONL file at {path}")
        for count, record in enumerate(
            records_from_path(
                path,
                expected_sha256=self.sha256,
                max_record_bytes=self.max_record_bytes,
            ),
            1,
        ):
            if count > self.max_rows:
                raise ValueError(f"row limit exceeded for {self.path}")
            yield record


class RelationSource(PinnedRows):
    id: Identifier
    format: Literal["records", "wikipedia", "wikidata"]
    uri: Nonempty
    license: Nonempty
    origin: Literal["human", "structured", "procedural", "teacher", "unknown", "mixed"]
    site: Identifier | None = None

    @model_validator(mode="after")
    def wikipedia_site(self):
        if (self.format == "wikipedia") != (self.site is not None):
            raise ValueError("site is required only for Wikipedia page records")
        return self


class RelationSpec(StrictModel):
    scope: PinnedRows
    sources: Annotated[tuple[RelationSource, ...], Field(min_length=1, max_length=4096)]
    max_index_bytes: Annotated[int, Field(ge=1024**2, le=2**40, strict=True)]
    max_shard_output_bytes: Annotated[int, Field(ge=1024**2, le=2**40, strict=True)]

    @model_validator(mode="after")
    def unique_sources(self):
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("relation source IDs must be unique")
        return self


class ScopedEntity(StrictModel):
    id: Identifier
    family: Identifier
    split: Split


class EntityValue(StrictModel):
    kind: Literal["entity"]
    id: Identifier


class TextValue(StrictModel):
    kind: Literal["text"]
    text: str


class TimeValue(StrictModel):
    kind: Literal["time"]
    time: Nonempty
    calendar: Nonempty
    precision: Annotated[int, Field(ge=0, le=14, strict=True)]
    timezone: Annotated[int, Field(strict=True)]
    before: Annotated[int, Field(ge=0, strict=True)]
    after: Annotated[int, Field(ge=0, strict=True)]


Value = Annotated[EntityValue | TextValue | TimeValue, Field(discriminator="kind")]


class Snak(StrictModel):
    predicate: Identifier
    value: Value


class Reference(StrictModel):
    id: Nonempty | None = None
    snaks: tuple[Snak, ...]


class Assertion(StrictModel):
    subject: Identifier
    predicate: Identifier
    value: Value
    qualifiers: tuple[Snak, ...] = ()

    def canonical(self):
        record = self.model_dump(mode="json")
        record["qualifiers"].sort(key=canonical_json)
        return record

    def scoped_ids(self):
        yield self.subject
        if isinstance(self.value, EntityValue):
            yield self.value.id
        for snak in self.qualifiers:
            if isinstance(snak.value, EntityValue):
                yield snak.value.id


class Evidence(StrictModel):
    claim_class: ClaimClass = "explicit"
    source_id: Nonempty | None = None
    rank: Literal["preferred", "normal", "deprecated"] | None = None
    references: tuple[Reference, ...] = ()


class ExplicitRecord(Assertion, Evidence):
    pass


class Extracted(StrictModel):
    line: Annotated[int, Field(gt=0, strict=True)]
    location: Nonempty
    assertion: Assertion
    evidence: Evidence


def wikipedia_entity(site: str, namespace: int, title: str) -> str:
    """Identity from a site's returned title and namespace, without redirect inference."""
    normalized = unicodedata.normalize("NFC", title.replace("_", " "))
    return "wikipedia:" + canonical_digest([site, namespace, normalized])


def _wiki_id(item, site):
    ns, title = item["ns"], item["title"]
    if type(ns) is not int or not isinstance(title, str) or not title:
        raise ValueError("invalid Wikipedia title or namespace")
    return wikipedia_entity(site, ns, title)


def _wiki_claims(row, site):
    subject = _wiki_id(row, site)
    page_id = row["pageid"]
    if type(page_id) is not int or page_id <= 0:
        raise ValueError("invalid Wikipedia page ID")
    for group in ("links", "categories"):
        values = row.get(group, [])
        if not isinstance(values, list):
            yield f"/{group}", "invalid_page_links"
            continue
        for index, item in enumerate(values):
            location = f"/{group}/{index}"
            try:
                value = EntityValue(kind="entity", id=_wiki_id(item, site))
                assertion = Assertion(
                    subject=subject, predicate=f"wikipedia:{group}", value=value
                )
            except KeyError, TypeError, ValueError:
                yield location, "invalid_page_link"
            else:
                yield location, (assertion, Evidence(source_id=str(page_id)))


def _wikidata_value(snak, predicate):
    if snak["property"] != predicate or snak["snaktype"] != "value":
        raise ValueError("unsupported snak property or value state")
    data = snak["datavalue"]
    kind, value = data["type"], data["value"]
    if kind == "wikibase-entityid" and value["entity-type"] in {"item", "property"}:
        entity_id = value["id"]
        prefix = "Q" if value["entity-type"] == "item" else "P"
        if (
            not isinstance(entity_id, str)
            or not entity_id.startswith(prefix)
            or not entity_id[1:].isascii()
            or not entity_id[1:].isdigit()
        ):
            raise ValueError("invalid Wikidata entity ID")
        return EntityValue(kind="entity", id=f"wikidata:{entity_id}")
    if kind == "string":
        return TextValue(kind="text", text=value)
    if kind == "time":
        return TimeValue(
            kind="time",
            time=value["time"],
            calendar=value["calendarmodel"],
            precision=value["precision"],
            timezone=value["timezone"],
            before=value["before"],
            after=value["after"],
        )
    raise ValueError("unsupported Wikidata datavalue")


def _snaks(groups: dict) -> Iterable[Snak]:
    for prop, snaks in groups.items():
        if not isinstance(snaks, list):
            raise TypeError("invalid Wikidata snak group")
        for snak in snaks:
            yield Snak(predicate=f"wikidata:{prop}", value=_wikidata_value(snak, prop))


def _wikidata_claims(row):
    entity_id, claims = row["id"], row["claims"]
    if (
        row["type"] != "item"
        or not isinstance(entity_id, str)
        or not entity_id.startswith("Q")
        or not entity_id[1:].isascii()
        or not entity_id[1:].isdigit()
        or not isinstance(claims, dict)
    ):
        raise ValueError("expected a Wikidata item with claims")
    for prop, statements in claims.items():
        escaped = prop.replace("~", "~0").replace("/", "~1")
        if not isinstance(statements, list):
            yield f"/claims/{escaped}", "invalid_statement_group"
            continue
        for index, statement in enumerate(statements):
            location = f"/claims/{escaped}/{index}"
            try:
                if (
                    statement["type"] != "statement"
                    or not prop.startswith("P")
                    or not prop[1:].isascii()
                    or not prop[1:].isdigit()
                ):
                    raise ValueError("unsupported Wikidata statement")
                references = statement.get("references", [])
                if not isinstance(references, list):
                    raise TypeError("invalid Wikidata references")
                assertion = Assertion(
                    subject=f"wikidata:{entity_id}",
                    predicate=f"wikidata:{prop}",
                    value=_wikidata_value(statement["mainsnak"], prop),
                    qualifiers=tuple(_snaks(statement.get("qualifiers", {}))),
                )
                evidence = Evidence(
                    source_id=statement["id"],
                    rank=statement["rank"],
                    references=tuple(
                        Reference(
                            id=reference["hash"],
                            snaks=tuple(_snaks(reference["snaks"])),
                        )
                        for reference in references
                    ),
                )
            except AttributeError, KeyError, TypeError, ValueError:
                yield location, "unsupported_statement"
            else:
                yield location, (assertion, evidence)


def _records(row, source):
    if source.format == "wikipedia":
        yield from _wiki_claims(row, source.site)
    elif source.format == "wikidata":
        yield from _wikidata_claims(row)
    else:
        record = ExplicitRecord.model_validate(row)
        yield (
            "/",
            (
                Assertion(**record.model_dump(include=set(Assertion.model_fields))),
                Evidence(**record.model_dump(include=set(Evidence.model_fields))),
            ),
        )


def extract_relation_source(
    staging: Path, path: str, declaration: dict, max_bytes: int
):
    """ShardRunner producer. Keep neither source bodies nor process-wide row lists."""
    source = RelationSource.model_validate({**declaration, "path": path})
    rows = claims = unresolved = bytes_written = 0
    with (
        (staging / "claims.jsonl").open("wb") as accepted,
        (staging / "unresolved.jsonl").open("wb") as rejected,
    ):
        for record in source.rows(Path(".")):
            rows += 1
            try:
                for location, result in _records(record.value, source):
                    if isinstance(result, str):
                        output = rejected
                        row = {
                            "line": record.physical_line,
                            "location": location,
                            "reason": result,
                        }
                        unresolved += 1
                    else:
                        output = accepted
                        assertion, evidence = result
                        row = Extracted(
                            line=record.physical_line,
                            location=location,
                            assertion=assertion,
                            evidence=evidence,
                        ).model_dump(mode="json")
                        claims += 1
                    raw = canonical_json(row) + b"\n"
                    bytes_written += len(raw)
                    if bytes_written > max_bytes:
                        raise OSError(
                            f"relation extraction output limit exceeded for {source.id}"
                        )
                    output.write(raw)
            except AttributeError, KeyError, TypeError, ValueError:
                if source.format == "records":
                    raise ValueError(
                        f"invalid explicit record in {source.id} line {record.physical_line}"
                    ) from None
                unresolved += 1
                raw = (
                    canonical_json(
                        {
                            "line": record.physical_line,
                            "location": "/",
                            "reason": "unsupported_source_row",
                        }
                    )
                    + b"\n"
                )
                bytes_written += len(raw)
                if bytes_written > max_bytes:
                    raise OSError(
                        f"relation extraction output limit exceeded for {source.id}"
                    )
                rejected.write(raw)
    return {
        "rows": rows,
        "claims": claims,
        "unresolved": unresolved,
        "bytes": bytes_written,
    }
