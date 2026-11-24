"""Verify physical Wikimedia multistream ranges and their pinned page index."""

from __future__ import annotations

import bz2
import hashlib
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from xml.parsers import expat

from klab.corpus.acquisition import declaration_sha256
from klab.corpus.project import SourceDeclaration, WikimediaDumpAcquisition
from klab.corpus.shards import range_file_identity
from klab.safe_io import symlinked
from klab.training.manifest import sha256_file

_CHUNK = 64 * 1024
_MAX_INDEX_LINE = 64 * 1024
_MAX_XML_DEPTH = 64
_MAX_IDENTIFIER = 256
_XML_NAMESPACE = "http://www.mediawiki.org/xml/export-0.11/"
_NS = _XML_NAMESPACE + "}"
_XML_LANG = "http://www.w3.org/XML/1998/namespace}lang"


@dataclass(frozen=True)
class _PageGroup:
    count: int
    sha256: str


@dataclass(frozen=True)
class WikimediaRange:
    """One complete bzip2 member. Offsets are compressed and end-exclusive."""

    start: int
    end: int
    expanded_bytes: int
    pages: int
    page_ids_sha256: str
    opens_document: bool
    closes_document: bool
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class WikimediaRangePlan:
    source_id: str
    declaration_sha256: str
    dump_sha256: str
    index_sha256: str
    compressed_bytes: int
    expanded_bytes: int
    indexed_pages: int
    xml_namespace: str
    dbname: str
    language: str
    root_offset: int
    header_bytes: int
    ranges: tuple[WikimediaRange, ...]

    @classmethod
    def from_record(cls, record: dict) -> WikimediaRangePlan:
        return cls(
            **{
                **record,
                "ranges": tuple(WikimediaRange(**item) for item in record["ranges"]),
            }
        )


def wikimedia_xml_context(
    dump: Path, plan: WikimediaRangePlan, *, limit: int
) -> tuple[bytes, bytes]:
    """Recover the verified prefix and its matching XML closing tag."""
    if not 0 <= plan.root_offset < plan.header_bytes <= limit:
        raise ValueError("Wikimedia XML header exceeds its byte bound")
    with bz2.open(dump, "rb") as stream:
        prefix = stream.read(plan.header_bytes)
    tag = re.match(rb"<([A-Za-z_][A-Za-z0-9_.:-]*)[\s>]", prefix[plan.root_offset :])
    if len(prefix) != plan.header_bytes or tag is None:
        raise ValueError("unsupported Wikimedia XML header for indexed extraction")
    return prefix, b"</" + tag[1] + b">"


def read_wikimedia_member(
    dump: Path, member: WikimediaRange, *, limit: int
) -> bytearray:
    """Read one verified member within the declared worker memory envelope."""
    if not 0 < member.expanded_bytes <= limit:
        raise ValueError("Wikimedia member exceeds its expanded byte bound")
    if not 0 <= member.start < member.end or (
        member.page_start is None
        or member.page_end is None
        or not 0 <= member.page_start < member.page_end <= member.expanded_bytes
    ):
        raise ValueError("invalid Wikimedia member extraction bounds")
    decoder = bz2.BZ2Decompressor()
    expanded = bytearray()
    with dump.open("rb") as stream:
        stream.seek(member.start)
        while not decoder.eof:
            chunk = b""
            if decoder.needs_input:
                chunk = stream.read(min(_CHUNK, member.end - stream.tell()))
                if not chunk:
                    raise ValueError("incomplete Wikimedia bzip2 member")
            expanded.extend(
                decoder.decompress(
                    chunk,
                    max_length=min(_CHUNK, member.expanded_bytes - len(expanded) + 1),
                )
            )
            if len(expanded) > member.expanded_bytes:
                raise ValueError("Wikimedia member exceeds its verified expanded size")
        if (
            stream.tell() != member.end
            or decoder.unused_data
            or len(expanded) != member.expanded_bytes
        ):
            raise ValueError("Wikimedia member differs from the verified range")
    return expanded


def _bounded_identity(path: Path, limit: int) -> list[int]:
    if symlinked(path.absolute()) or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(
            f"Wikimedia range input must be a regular nonsymlink file: {path}"
        )
    identity = range_file_identity(path)
    if not 0 < identity[2] <= limit:
        raise ValueError(
            f"Wikimedia range input exceeds its compressed byte bound: {path}"
        )
    return identity


def _read_index(
    index: Path, *, dump_size: int, max_expanded_bytes: int, max_members: int
) -> dict[int, _PageGroup]:
    groups = {}
    previous = -1
    digest = hashlib.sha256()
    pages = expanded = 0
    with bz2.open(index, "rb") as stream:
        while line := stream.readline(
            min(_MAX_INDEX_LINE + 1, max_expanded_bytes - expanded + 1)
        ):
            expanded += len(line)
            if expanded > max_expanded_bytes:
                raise ValueError("Wikimedia index exceeds its expanded byte bound")
            if len(line) > _MAX_INDEX_LINE:
                raise ValueError("Wikimedia index line exceeds its byte bound")
            fields = line.rstrip(b"\r\n").split(b":", 2)
            if len(fields) != 3 or not fields[2]:
                raise ValueError("invalid Wikimedia index record")
            offset, page_id, _ = fields
            if (
                not (offset.isdigit() and page_id.isdigit())
                or max(len(offset), len(page_id)) > 20
            ):
                raise ValueError("invalid Wikimedia index offset or page ID")
            start, page = int(offset), int(page_id)
            if not previous <= start < dump_size or page < 1:
                raise ValueError("invalid Wikimedia index offset or page ID")
            if start != previous:
                if pages:
                    groups[previous] = _PageGroup(pages, digest.hexdigest())
                if len(groups) == max_members:
                    raise ValueError("Wikimedia index exceeds its member bound")
                previous, pages, digest = start, 0, hashlib.sha256()
            digest.update(f"{page}\n".encode("ascii"))
            pages += 1
    if not pages:
        raise ValueError("Wikimedia index contains no pages")
    groups[previous] = _PageGroup(pages, digest.hexdigest())
    return groups


def _reject_doctype(*_args) -> None:
    raise ValueError("Wikimedia XML DTD/entities are forbidden")


class _WikimediaXML:
    """Check shared XML context and page IDs without materializing page text."""

    def __init__(self, dbname: str):
        self.parser = expat.ParserCreate(namespace_separator="}")
        self.parser.StartElementHandler = self.start
        self.parser.EndElementHandler = self.end
        self.parser.CharacterDataHandler = self.characters
        self.parser.StartDoctypeDeclHandler = _reject_doctype
        self.dbname = dbname
        self.elements: list[str] = []
        self.root_opened = self.root_closed = self.site_verified = False
        self.root_language = ""
        self.root_offset = 0
        self.root_end = 0
        self.header_bytes: int | None = None
        self.member_page_start: int | None = None
        self.field: str | None = None
        self.value = ""
        self.site: dict[str, str] = {}
        self.page_id: int | None = None
        self.pages = 0
        self.page_ids = hashlib.sha256()

    def start(self, name: str, attrs: dict[str, str]) -> None:
        depth = len(self.elements)
        if depth == _MAX_XML_DEPTH:
            raise ValueError("Wikimedia XML exceeds its nesting bound")
        if self.field is not None:
            raise ValueError("invalid nested Wikimedia identifier")
        if depth == 0:
            if name != _NS + "mediawiki":
                raise ValueError("unsupported Wikimedia XML namespace")
            self.root_opened = True
            self.root_offset = self.parser.CurrentByteIndex
            self.root_language = attrs.get(_XML_LANG, "")
        elif depth == 1:
            if name == _NS + "siteinfo" and not self.site_verified:
                self.site.clear()
            elif name == _NS + "page" and self.site_verified:
                self.page_id = None
                offset = self.parser.CurrentByteIndex
                if self.header_bytes is None:
                    self.header_bytes = offset
                if self.member_page_start is None:
                    self.member_page_start = offset
            else:
                raise ValueError("invalid Wikimedia siteinfo or page context")
        elif depth == 2:
            parent = self.elements[-1]
            if (
                parent == _NS + "siteinfo" and name in {_NS + "dbname", _NS + "lang"}
            ) or (parent == _NS + "page" and name == _NS + "id"):
                self.field, self.value = name, ""
        self.elements.append(name)

    def characters(self, text: str) -> None:
        if self.field is not None:
            if len(text) > _MAX_IDENTIFIER - len(self.value):
                raise ValueError("Wikimedia XML identifier exceeds its bound")
            self.value += text

    def end(self, name: str) -> None:
        self.elements.pop()
        if self.field is not None:
            if self.field == _NS + "id":
                if self.page_id is not None or not re.fullmatch(
                    r"[0-9]{1,20}", self.value
                ):
                    raise ValueError("invalid Wikimedia page ID")
                self.page_id = int(self.value)
            else:
                if self.field in self.site:
                    raise ValueError("duplicate Wikimedia siteinfo field")
                self.site[self.field] = self.value
            self.field = None
        if not self.elements:
            self.root_closed = True
            self.root_end = self.parser.CurrentByteIndex
        elif len(self.elements) == 1:
            if name == _NS + "siteinfo":
                language = self.site.get(_NS + "lang") or self.root_language
                if language != "en" or self.site.get(_NS + "dbname") != self.dbname:
                    raise ValueError(
                        "Wikimedia dump site identity or language mismatch"
                    )
                self.site_verified = True
            else:
                if not self.page_id:
                    raise ValueError("Wikimedia page lacks a valid ID")
                self.page_ids.update(f"{self.page_id}\n".encode("ascii"))
                self.pages += 1


def _scan_dump(
    dump: Path,
    *,
    dump_size: int,
    max_expanded_bytes: int,
    max_members: int,
    xml: _WikimediaXML,
    indexed: dict[int, _PageGroup],
) -> tuple[WikimediaRange, ...]:
    ranges = []
    start = expanded = 0
    pending = b""
    with dump.open("rb") as stream:
        while start < dump_size:
            if len(ranges) == max_members:
                raise ValueError("Wikimedia dump exceeds its member bound")
            before = expanded
            opened, closed = xml.root_opened, xml.root_closed
            xml.pages, xml.page_ids = 0, hashlib.sha256()
            xml.member_page_start = None
            decoder = bz2.BZ2Decompressor()
            while not decoder.eof:
                if decoder.needs_input:
                    compressed = pending or stream.read(
                        min(_CHUNK, dump_size - stream.tell())
                    )
                    pending = b""
                    if not compressed:
                        raise ValueError("incomplete Wikimedia bzip2 member")
                else:
                    compressed = b""
                chunk = decoder.decompress(
                    compressed,
                    max_length=min(_CHUNK, max_expanded_bytes - expanded + 1),
                )
                expanded += len(chunk)
                if expanded > max_expanded_bytes:
                    raise ValueError("Wikimedia dump exceeds its expanded byte bound")
                xml.parser.Parse(chunk, False)
            end = stream.tell() - len(decoder.unused_data)
            pending = decoder.unused_data
            if len(xml.elements) > 1 or xml.parser.CurrentByteIndex != expanded:
                raise ValueError("Wikimedia XML crosses a compressed member boundary")
            group = _PageGroup(xml.pages, xml.page_ids.hexdigest())
            expected = indexed.pop(start, None)
            if expected != (group if xml.pages else None):
                raise ValueError(
                    f"Wikimedia index page IDs or offset differ at member {start}"
                )
            ranges.append(
                WikimediaRange(
                    start,
                    end,
                    expanded - before,
                    group.count,
                    group.sha256,
                    xml.root_opened and not opened,
                    xml.root_closed and not closed,
                    xml.member_page_start - before
                    if xml.member_page_start is not None
                    else None,
                    (xml.root_end if xml.root_closed and not closed else expanded)
                    - before
                    if xml.member_page_start is not None
                    else None,
                )
            )
            start = end
    xml.parser.Parse(b"", True)
    if indexed or not xml.site_verified or not xml.root_closed:
        raise ValueError("Wikimedia index or XML context is incomplete")
    return tuple(ranges)


def plan_wikimedia_ranges(
    source: SourceDeclaration,
    dump: Path,
    index: Path,
    *,
    index_sha256: str,
    max_index_bytes: int,
    max_index_expanded_bytes: int,
    max_members: int,
) -> WikimediaRangePlan:
    """Check local pinned bytes, then verify each bzip2 member against the index."""
    spec = source.acquisition
    if source.kind != "wikimedia_dump":
        raise ValueError("a declared Wikimedia dump source is required")
    assert isinstance(spec, WikimediaDumpAcquisition)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", index_sha256):
        raise ValueError("Wikimedia index requires a SHA-256 pin")
    if min(max_index_bytes, max_index_expanded_bytes, max_members) < 1:
        raise ValueError("Wikimedia index bounds must be positive")
    try:
        initial = (
            _bounded_identity(dump, spec.max_compressed_bytes),
            _bounded_identity(index, max_index_bytes),
        )
        dump_size = initial[0][2]
        sha1 = hashlib.sha1()
        sha256 = sha256_file(dump, hash_chunk=sha1.update)
        if sha1.hexdigest() != spec.expected_sha1.lower() or (
            spec.expected_sha256 is not None and sha256 != spec.expected_sha256.lower()
        ):
            raise ValueError("Wikimedia dump digest differs from the declaration")
        index_digest = sha256_file(index)
        if index_digest != index_sha256.lower():
            raise ValueError("Wikimedia index SHA-256 mismatch")
        indexed = _read_index(
            index,
            dump_size=dump_size,
            max_expanded_bytes=max_index_expanded_bytes,
            max_members=max_members,
        )
        dbname = urlparse(source.canonical_uri).path.split("/")[1]
        xml = _WikimediaXML(dbname)
        ranges = _scan_dump(
            dump,
            dump_size=dump_size,
            max_expanded_bytes=spec.max_decompressed_bytes,
            max_members=max_members,
            xml=xml,
            indexed=indexed,
        )
        if (range_file_identity(dump), range_file_identity(index)) != initial:
            raise ValueError("Wikimedia range input changed during verification")
    except (OSError, EOFError, expat.ExpatError) as error:
        raise ValueError(f"cannot verify Wikimedia dump or index: {error}") from error
    assert xml.header_bytes is not None
    return WikimediaRangePlan(
        source.id,
        declaration_sha256(source),
        sha256,
        index_digest,
        dump_size,
        sum(member.expanded_bytes for member in ranges),
        sum(member.pages for member in ranges),
        _XML_NAMESPACE,
        dbname,
        "en",
        xml.root_offset,
        xml.header_bytes,
        ranges,
    )
