"""Content-addressed normalized shards shared by the disk build paths."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.build_stages import implementation_digests
from klab.corpus.document_normalization import (
    NORMALIZATION_IMPLEMENTATION_FILES,
    EmptyDocumentError,
    records_for_file,
)
from klab.corpus.progress import BuildProgress
from klab.corpus.project import ROW_SOURCE_KINDS, NormalizerOptions
from klab.corpus.rights import FILE_ADMISSION_KINDS
from klab.safe_io import rename_noreplace, write_canonical_json
from klab.training.manifest import sha256_file

_PREPARED_FILES = ("docs.jsonl", "spans.jsonl", "rejected.jsonl")
_ADMITTED_RECORD_MAX_BYTES = 8 * 1024**2


def prepared_shard_implementation() -> dict[str, str]:
    """Source digests for normalization and the writer of its shard receipt."""
    return implementation_digests(
        "prepared_shards.py", *NORMALIZATION_IMPLEMENTATION_FILES
    )


def _output(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + b"\n")


def _receipt_identity(
    source: Any, snapshot_sha: str, file: dict[str, Any]
) -> dict[str, Any]:
    return {
        "stage_version": "normalized-source-v2",
        "implementation": prepared_shard_implementation(),
        "source": source.model_dump(mode="json"),
        "source_id": source.id,
        "revision": source.revision,
        "snapshot_sha256": snapshot_sha,
        "path": file["path"],
        "file_sha256": file["sha256"],
        "file_size": file["size"],
    }


def _verify_prepared(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("prepared source shard is not a directory")
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    if receipt["identity"] != identity or set(receipt["files"]) != set(_PREPARED_FILES):
        raise ValueError("prepared source shard provenance mismatch")
    for name, expected in receipt["files"].items():
        item = path / name
        if (
            item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != expected["size"]
            or sha256_file(item) != expected["sha256"]
        ):
            raise ValueError(f"prepared source shard changed: {name}")
    return receipt


class SourceParseError(ValueError):
    """Input file rejected by the normalizer, not a broken receipt or build."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0


def prepare_file(
    *,
    build_id: str,
    prepared_root: Path,
    source: Any,
    file: dict[str, Any],
    snapshot_sha: str,
    source_path: Path,
    decision: Any,
    progress: BuildProgress,
    normalizer_options: NormalizerOptions,
    document_policy: Any,
    record_admission: dict[str, Any] | None = None,
    admission_sha256: str | None = None,
    row_limit: int | None = None,
) -> Path:
    # Internal primitive: callers must first use verify_record_admission against
    # the pinned original rows. The digest binds that complete reviewed artifact,
    # not a newly inferred policy or an approval supplied by this helper.
    if (record_admission is None) != (admission_sha256 is None):
        raise ValueError(
            "admitted shard requires verified decisions and manifest digest"
        )
    if record_admission is not None:
        if not isinstance(admission_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", admission_sha256
        ):
            raise ValueError("invalid admission manifest digest")
        if "decisions" in record_admission:
            if source.kind != "huggingface_dataset" or not file["path"].endswith(
                ".jsonl"
            ):
                raise ValueError("streamed record admission requires an HF JSONL shard")
        elif source.kind not in FILE_ADMISSION_KINDS:
            raise ValueError("file admission requires a file-admitted source")
        if decision != record_admission["rights"]:
            raise ValueError("admitted shard file rights differ from verified policy")
    identity = _receipt_identity(source, snapshot_sha, file)
    identity["normalizer"] = normalizer_options.model_dump(mode="json")
    identity["document_policy"] = document_policy.model_dump(mode="json")
    identity["rights"] = decision.model_dump(mode="json")
    if row_limit is not None:
        identity["row_limit"] = row_limit
    if record_admission is not None:
        # Bind only this verified file's decisions, not the admission plan for
        # unrelated sources. Stream disk-backed decisions without materializing.
        digest = hashlib.sha256()
        for key in sorted(record_admission):
            value = record_admission[key]
            digest.update(canonical_json(key))
            if key == "decisions":
                for row_index, row in value.items():
                    digest.update(canonical_json([row_index, row]))
            else:
                if hasattr(value, "model_dump"):
                    value = value.model_dump(mode="json")
                digest.update(canonical_json(value))
        identity["file_admission_sha256"] = digest.hexdigest()
    parent = prepared_root / source.id
    destination = parent / canonical_digest(identity)
    if destination.exists():
        receipt = _verify_prepared(destination, identity)
        progress.update(
            documents=receipt["counts"]["documents"],
            input_bytes=receipt["counts"]["input_bytes"],
            output_records=receipt["counts"]["documents"],
        )
        progress.record("reused_verified_shard")
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".incomplete-", dir=parent) as temporary:
        staging = Path(temporary)
        counts = {"documents": 0, "rejected": 0, "input_bytes": 0}
        with ExitStack() as stack:
            docs = stack.enter_context((staging / "docs.jsonl").open("wb"))
            spans = stack.enter_context((staging / "spans.jsonl").open("wb"))
            rejected = stack.enter_context((staging / "rejected.jsonl").open("wb"))
            if source.kind in ROW_SOURCE_KINDS and file["path"].endswith(".jsonl"):
                with source_path.open("rb") as stream:
                    index = 0
                    from klab.corpus.jsonl_records import iter_lf_lines

                    input_digest = hashlib.sha256()
                    for _, raw in iter_lf_lines(
                        stream,
                        max_record_bytes=(
                            _ADMITTED_RECORD_MAX_BYTES
                            if record_admission is not None
                            else None
                        ),
                    ):
                        input_digest.update(raw)
                        counts["input_bytes"] += len(raw)
                        if not raw.strip():
                            progress.update(input_bytes=len(raw))
                            continue
                        index += 1
                        if row_limit is not None and index > row_limit:
                            break
                        if index > source.acquisition.max_rows:
                            progress.update(input_bytes=len(raw))
                            continue
                        dropped: list[dict[str, Any]] = []
                        try:
                            rows = records_for_file(
                                raw,
                                file["path"],
                                source,
                                snapshot_sha,
                                file_rights=decision,
                                record_admission=record_admission,
                                rejected_records=dropped,
                                full_file_sha256=file["sha256"],
                                first_row_index=index,
                                normalizer_options=normalizer_options,
                                document_policy=document_policy,
                            )
                        except EmptyDocumentError:
                            raise
                        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                            raise SourceParseError(str(exc)) from exc
                        for doc, span in rows:
                            _output(docs, doc)
                            _output(spans, span)
                            counts["documents"] += 1
                        for row in dropped:
                            _output(rejected, row)
                            counts["rejected"] += 1
                        progress.update(
                            documents=1, input_bytes=len(raw), output_records=len(rows)
                        )
                    if (
                        record_admission is not None
                        and input_digest.hexdigest() != file["sha256"]
                    ):
                        raise ValueError(
                            "admitted source shard bytes differ from acquisition"
                        )
            else:
                if record_admission is not None:
                    with source_path.open("rb") as source_stream:
                        raw = source_stream.read(_ADMITTED_RECORD_MAX_BYTES + 1)
                    if len(raw) > _ADMITTED_RECORD_MAX_BYTES:
                        raise ValueError("admitted file-byte cap exceeded")
                    if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                        raise ValueError(
                            "admitted source bytes differ from acquisition"
                        )
                else:
                    raw = source_path.read_bytes()
                dropped = []
                try:
                    rows = records_for_file(
                        raw,
                        file["path"],
                        source,
                        snapshot_sha,
                        file_rights=decision,
                        record_admission=record_admission,
                        rejected_records=dropped,
                        normalizer_options=normalizer_options,
                        document_policy=document_policy,
                    )
                except EmptyDocumentError:
                    raise
                except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                    raise SourceParseError(str(exc)) from exc
                for doc, span in rows:
                    _output(docs, doc)
                    _output(spans, span)
                for row in dropped:
                    _output(rejected, row)
                counts = {
                    "documents": len(rows),
                    "rejected": len(dropped),
                    "input_bytes": len(raw),
                }
                progress.update(
                    documents=len(rows), input_bytes=len(raw), output_records=len(rows)
                )
            for output in (docs, spans, rejected):
                output.flush()
                os.fsync(output.fileno())
        if row_limit is None and counts["input_bytes"] != file["size"]:
            raise ValueError(
                "prepared source shard input size differs from acquisition"
            )
        receipt = {
            "schema_version": 1,
            "identity": identity,
            "provenance": {"build_id": build_id, "admission_sha256": admission_sha256},
            "counts": counts,
            "files": {
                name: {
                    "size": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in _PREPARED_FILES
            },
        }
        write_canonical_json(staging / "receipt.json", receipt)
        if destination.exists():
            _verify_prepared(destination, identity)
        else:
            rename_noreplace(staging, destination)
    _verify_prepared(destination, identity)
    return destination
