"""Verified, pre-build document IDs for frozen family split assignments."""

from __future__ import annotations

import hashlib
from collections import Counter
from contextlib import closing
from itertools import zip_longest
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.acquisition import verify_acquisition
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.bounded_metadata import (
    decision_index,
    metadata_operation,
    read_metadata,
)
from klab.corpus.build_stages import implementation_digests
from klab.corpus.document_normalization import (
    NORMALIZATION_IMPLEMENTATION_FILES,
    records_for_file,
)
from klab.corpus.inventory_writer import INVENTORY_WINDOW_BYTES, inventory_writer
from klab.corpus.jsonl_records import records_from_bytes, records_from_path
from klab.corpus.project import Project
from klab.corpus.release_verify import current_settings
from klab.corpus.rights import resolve_file_rights
from klab.training.manifest import sha256_file
from klab.verification_proofs import verification_options

_MAX_RECORD_BYTES = 8 * 1024**2
_MAX_FILE_BYTES = 8 * 1024**2


def _verified_hf_records(
    parsed_rows,
    file: dict[str, Any],
    retrieval: dict[str, Any],
    revision: str,
    *,
    original_rows: bool = False,
):
    """Yield one authenticated original row and metadata without retaining its text."""
    matching = [
        shard
        for shard in retrieval.get("shards", [])
        if shard.get("output_path") == file["path"]
    ]
    if len(matching) != 1:
        raise ValueError("split inventory lacks one pinned HF shard receipt")
    shard = matching[0]
    selected = shard["selected_rows"]
    seen = decision_index()
    for parsed, receipt in zip_longest(parsed_rows, selected):
        if parsed is None or receipt is None:
            raise ValueError("split inventory HF row count differs from receipt")
        row = parsed.value
        envelope = row.get("_klab_source")
        if not isinstance(envelope, dict):
            raise TypeError("split inventory HF row lacks provenance")
        original = {key: value for key, value in row.items() if key != "_klab_source"}
        digest = canonical_digest(original)
        index = receipt["source_row_index"]
        if (
            type(index) is not int
            or index < 0
            or index in seen
            or envelope.get("source_row_index") != index
            or envelope.get("source_row_sha256") != digest
            or receipt["source_row_sha256"] != digest
            or envelope.get("dataset_revision") != revision
            or envelope.get("source_shard_path") != shard["source_shard_path"]
            or envelope.get("source_shard_sha256") != shard["source_shard_sha256"]
        ):
            raise ValueError("split inventory HF row provenance mismatch")
        metadata = original.get("metadata")
        seen[index] = None
        yield (
            parsed,
            index,
            (
                original
                if original_rows
                else metadata
                if isinstance(metadata, dict)
                else {}
            ),
        )


def _verified_hf_metadata(
    raw: bytes,
    file: dict[str, Any],
    retrieval: dict[str, Any],
    revision: str,
    *,
    original_rows: bool = False,
) -> dict[int, dict[str, Any]]:
    """Materialize metadata for callers with explicitly small in-memory fixtures."""
    return {
        index: metadata
        for _, index, metadata in _verified_hf_records(
            records_from_bytes(raw, source=file["path"]),
            file,
            retrieval,
            revision,
            original_rows=original_rows,
        )
    }


def _file_documents(
    source, file, snapshot_path, snapshot_sha, retrieval, normalizer, document_policy
):
    """Preserve whole-shard document identities while bounding source-text reads."""
    path = snapshot_path / "files" / file["path"]
    if source.kind == "huggingface_dataset" and file["path"].endswith(".jsonl"):
        rights = (
            resolve_file_rights(source.rights, file["path"], b"")
            if source.rights
            else None
        )
        with closing(
            records_from_path(path, max_record_bytes=_MAX_RECORD_BYTES)
        ) as records:
            for ordinal, (record, index, metadata) in enumerate(
                _verified_hf_records(
                    records,
                    file,
                    retrieval,
                    source.revision,
                    original_rows=source.acquisition.qualification is not None,
                ),
                1,
            ):
                # Validate every acquired row, including any beyond the parser limit.
                if ordinal > source.acquisition.max_rows:
                    continue
                for document, span in records_for_file(
                    record.raw,
                    file["path"],
                    source,
                    snapshot_sha,
                    file_rights=rights,
                    full_file_sha256=file["sha256"],
                    first_row_index=ordinal,
                    normalizer_options=normalizer,
                    document_policy=document_policy,
                ):
                    if document.get("metadata", {}).get("source_row_index") != index:
                        raise ValueError("split inventory document lacks pinned HF row")
                    yield document, span, metadata
        if sha256_file(path) != file["sha256"]:
            raise ValueError("split inventory shard changed during parsing")
        return
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("split inventory file-byte cap exceeded")
    with path.open("rb") as handle:
        raw = handle.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise ValueError("split inventory file-byte cap exceeded")
    if hashlib.sha256(raw).hexdigest() != file["sha256"]:
        raise ValueError("split inventory file changed during parsing")
    metadata = (
        _verified_hf_metadata(
            raw,
            file,
            retrieval,
            source.revision,
            original_rows=getattr(source.acquisition, "qualification", None)
            is not None,
        )
        if source.kind == "huggingface_dataset"
        else {}
    )
    rights = (
        resolve_file_rights(source.rights, file["path"], raw) if source.rights else None
    )
    for document, span in records_for_file(
        raw,
        file["path"],
        source,
        snapshot_sha,
        file_rights=rights,
        normalizer_options=normalizer,
        document_policy=document_policy,
    ):
        index = document.get("metadata", {}).get("source_row_index")
        if source.kind == "huggingface_dataset" and index not in metadata:
            raise ValueError("split inventory document lacks pinned HF row")
        yield document, span, metadata.get(index, {})


def _wiki_locator_hint(locator: str) -> str | None:
    try:
        parsed = urlsplit(locator)
        host = (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError:
        return None
    if parsed.scheme != "https" or not (
        host in {"wikipedia.com", "wikipedia.org"}
        or host.endswith((".wikipedia.org", ".wikibooks.org", ".wikimedia.org"))
    ):
        return None
    if parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path[6:]).replace("_", " ").casefold()
    elif parsed.path == "/w/index.php":
        title = (
            parse_qs(parsed.query).get("title", [""])[0].replace("_", " ").casefold()
        )
    else:
        return None
    return f"wikimedia-page:{host}:{title}" if title else None


def _family_hint(
    source: Any,
    document: dict[str, Any],
    source_metadata: dict[str, Any],
    *,
    qualification=None,
) -> str:
    """Conservative grouping hint; a reviewer must merge mirrored/topic siblings."""
    metadata = document.get("metadata") or {}
    if qualification is None:
        qualification = getattr(
            getattr(source, "acquisition", None), "qualification", None
        )
    if qualification is not None:
        from klab.corpus.row_qualification import family_hint

        try:
            return family_hint(source_metadata, qualification)
        except TypeError, ValueError:
            return f"unresolved-family:{document['document_id']}"
    if source.kind == "huggingface_dataset":
        if "project_gutenberg" in source.canonical_uri:
            identifier = source_metadata.get("id") or metadata.get("id")
            if identifier:
                return f"gutenberg-work:{identifier}"
            locator = source_metadata.get("url") or metadata.get("url")
            if isinstance(locator, str):
                path = urlsplit(locator).path
                if path.startswith("/ebooks/"):
                    return f"gutenberg-work:{path.split('/')[2].split('.')[0]}"
        else:
            page = source_metadata.get("page_id") or metadata.get("page_id")
            if page:
                return f"wikimedia-page:{page}"
            locator = (
                source_metadata.get("page_uri")
                or source_metadata.get("url")
                or metadata.get("page_uri")
                or metadata.get("url")
            )
            if isinstance(locator, str) and (hint := _wiki_locator_hint(locator)):
                return hint
    path = document["source_location"].split("#", 1)[0]
    if source.kind == "git":
        return f"git-file:{source.id}:{path}"
    if source.kind == "archive":
        return f"archive-member:{source.id}:{path}"
    if source.kind == "pdf_documents":
        return f"pdf-document:{source.id}:{path}"
    if source.kind == "mediawiki_dump":
        page = source_metadata.get("page_id") or metadata.get("page_id")
        if page:
            return f"mediawiki-page:{source.acquisition.dbname}:{page}"
    return f"unresolved-family:{document['document_id']}"


def write_split_inventory_stored(
    project: Project, work_root: Path, output: Path, *, completion_cache=None
) -> dict[str, Any]:
    """Write metadata-only IDs from cold-verified immutable acquisition bytes.

    Candidate family hints are not final assignments. The reviewed mapping can
    group multiple hints together before it is frozen into `splits.yaml`.
    """
    root = work_root.resolve()
    destination = output.resolve()
    if not destination.is_relative_to(root) or destination.exists():
        raise ValueError("split inventory requires a new path within the work root")
    lock = verify_acquisition(project, root, **verification_options(root))
    from klab.corpus.split_inventory_cache import InventoryCache, inventory_inputs

    # Completion caching is optional; a native producer can still run when
    # metadata cannot be fingerprinted, but must not publish a reuse proof.
    try:
        dependencies = inventory_inputs(project, root, lock)
    except OSError:
        dependencies = None
    cache = (
        (completion_cache if completion_cache is not None else InventoryCache(root))
        if dependencies is not None
        else None
    )
    if cache is not None and cache.root != root:
        raise ValueError("inventory completion cache work root differs")
    identity = canonical_digest(
        {
            "request": cache.request(project, destination)
            if cache is not None
            else uuid4().hex,
            "documents": project.release.policy.documents.model_dump(mode="json"),
            "inputs": dependencies,
            "producer": implementation_digests(
                "split_inventory.py",
                "inventory_writer.py",
                "view_checkpoints.py",
                *NORMALIZATION_IMPLEMENTATION_FILES,
            ),
        }
    )
    with inventory_writer(destination, identity) as (checkpoints, view, state):
        counts = Counter(state.get("counts", {}))
        echo = current_settings().echo
        if state and echo is not None:
            print(
                f"split inventory: reused {sum(counts.values())} document rows",
                file=echo,
            )
        if not state.get("complete"):
            source_start = state.get("source", 0)
            pending_bytes = pending_files = 0
            for number, source in enumerate(
                project.sources[source_start:], start=source_start
            ):
                qualification = getattr(source.acquisition, "qualification", None)
                overrides = getattr(
                    getattr(project, "splits", None), "family_fields_by_source", {}
                )
                if source.id in overrides:
                    from klab.corpus.row_qualification import RowQualification

                    qualification = RowQualification.model_validate(
                        {
                            **qualification.model_dump(mode="json"),
                            "family_fields": overrides[source.id],
                        }
                    )
                entry = lock["sources"][source.id]
                if entry["snapshot_path"] is None:
                    continue
                snapshot_path = Path(entry["snapshot_path"])
                manifest = read_metadata(snapshot_path / "manifest.json")
                files = sorted(manifest["files"], key=lambda item: item["path"])
                first_file = state.get("file", 0) if number == source_start else 0
                for file_number in range(first_file, len(files)):
                    file = files[file_number]
                    path = file["path"]
                    if source.rights and path == source.rights.nested_metadata_path:
                        continue
                    normalizer = project.release.normalizer
                    with closing(
                        _file_documents(
                            source,
                            file,
                            snapshot_path,
                            entry["snapshot_sha256"],
                            manifest["retrieval"],
                            normalizer,
                            project.release.policy.documents,
                        )
                    ) as documents:
                        for document, _, source_metadata in documents:
                            index = document.get("metadata", {}).get("source_row_index")
                            row = {
                                "document_id": document["document_id"],
                                "source_id": source.id,
                                "snapshot_sha256": entry["snapshot_sha256"],
                                "source_location": document["source_location"],
                                "content_sha256": document["content_sha256"],
                                "raw_content_sha256": document["raw_content_sha256"],
                                "family_hint": _family_hint(
                                    source,
                                    document,
                                    source_metadata,
                                    qualification=qualification,
                                ),
                                "source_row_index": index,
                            }
                            view.write(canonical_json(row) + b"\n")
                            counts[source.id] += 1
                    pending_bytes += file["size"]
                    pending_files += 1
                    if pending_bytes >= INVENTORY_WINDOW_BYTES or pending_files >= 128:
                        checkpoints.commit(
                            {
                                "source": number,
                                "file": file_number + 1,
                                "counts": dict(counts),
                            }
                        )
                        pending_bytes = pending_files = 0
            if dependencies is not None:
                unchanged(dependencies)
            checkpoints.commit({"complete": True, "counts": dict(counts)})
        elif dependencies is not None:
            unchanged(dependencies)
        digest = view.file_digest().sha256
    report = {
        "schema_version": 1,
        "project_id": project.config.id,
        "path": str(destination),
        "sha256": digest,
        "counts": dict(sorted(counts.items())),
        "documents": sum(counts.values()),
    }
    if dependencies is not None:
        dependencies.update(capture([destination]))
        cache.record_completed(project, destination, report, dependencies)
    return report


def write_split_inventory(
    project: Project, work_root: Path, output: Path, *, completion_cache=None
) -> dict[str, Any]:
    with metadata_operation(
        Path(work_root) / "corpora" / project.config.id / "metadata-scratch",
        max_database_bytes=None,
    ):
        return write_split_inventory_stored(
            project, work_root, output, completion_cache=completion_cache
        )
