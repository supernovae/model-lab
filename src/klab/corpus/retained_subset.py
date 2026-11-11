"""Deterministic bounded derivatives of retained HF rows with cold ancestry replay."""

import hashlib
from contextlib import contextmanager, nullcontext
from itertools import zip_longest
from pathlib import Path

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.bounded_metadata import metadata_operation
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.project import SourceDeclaration, source_declaration_payload
from klab.training.manifest import sha256_file


@contextmanager
def _parent_metadata(directory):
    # Parent row indexes are replay scratch, not output. Release their rows before
    # later admission calls in the same operation, preserving the shared DB cap.
    with metadata_operation(Path(directory)) as store:
        first = store.next_identifier
        try:
            yield
        finally:
            store.release_after(first)


def derive_subset(source, *, scratch_directory, staging=None):
    from klab.corpus.acquisition import _safe_snapshot_path, verify_snapshot

    spec = source.acquisition.retained_subset
    parent = spec.snapshot_path
    _safe_snapshot_path(parent)
    if (parent / "manifest.json").stat().st_size > 256 * 1024**2:
        raise ValueError("retained parent metadata input-byte cap exceeded")
    if sha256_file(parent / "manifest.json") != spec.manifest_sha256:
        raise ValueError("retained parent manifest digest mismatch")
    # Nested derivatives are deliberately unsupported. Check before recursive verification.
    from klab.corpus.bounded_metadata import read_metadata

    with _parent_metadata(scratch_directory):
        header = read_metadata(parent / "manifest.json")
        if header["declaration"]["acquisition"].get("retained_subset") is not None:
            raise ValueError("nested retained subsets are unsupported")
        parent_source = SourceDeclaration.model_validate(header["declaration"])
        if (
            parent_source.kind != "huggingface_dataset"
            or len(header["files"]) != 1
            or header["files"][0]["size"]
            > min(parent_source.acquisition.max_bytes, 8 * 1024**3)
        ):
            raise ValueError(
                "retained parent input-byte cap or source shape unsupported"
            )
        manifest = verify_snapshot(parent)
        if manifest["snapshot_sha256"] != spec.snapshot_sha256:
            raise ValueError("retained parent snapshot identity mismatch")
        original = SourceDeclaration.model_validate(manifest["declaration"])
        expected = source_declaration_payload(original)
        actual = source_declaration_payload(source)
        actual["acquisition"].pop("retained_subset")
        for field in ("max_rows", "max_bytes"):
            actual["acquisition"][field] = expected["acquisition"][field]
        if actual != expected or original.kind != "huggingface_dataset":
            raise ValueError("retained derivative changed parent source policy")
        if len(manifest["files"]) != 1 or len(manifest["retrieval"]["shards"]) != 1:
            raise ValueError("retained subset requires one parent shard")
        file = manifest["files"][0]
        shard = manifest["retrieval"]["shards"][0]
        if file["path"] != shard["output_path"] or not file["path"].endswith(".jsonl"):
            raise ValueError("retained subset requires parent JSONL")
        raw_path = parent / "files" / file["path"]
        output = Path(staging) / "files" / file["path"] if staging is not None else None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
        selected, count, size, excluded = [], 0, 0, 0
        digest = hashlib.sha256()
        with output.open("xb") if output is not None else nullcontext(None) as stream:
            for record, receipt in zip_longest(
                records_from_path(raw_path, max_record_bytes=8 * 1024**2),
                shard["selected_rows"],
            ):
                if record is None or receipt is None:
                    raise ValueError("retained parent row inventory mismatch")
                row = record.value
                envelope = row.get("_klab_source")
                original_row = {
                    key: value for key, value in row.items() if key != "_klab_source"
                }
                row_sha = canonical_digest(original_row)
                index = receipt["source_row_index"]
                if (
                    not isinstance(envelope, dict)
                    or envelope.get("source_row_index") != index
                    or envelope.get("source_row_sha256") != row_sha
                    or receipt["source_row_sha256"] != row_sha
                    or envelope.get("source_shard_path") != shard["source_shard_path"]
                    or envelope.get("source_shard_sha256")
                    != shard["source_shard_sha256"]
                    or envelope.get("dataset_revision") != original.revision
                ):
                    raise ValueError("retained parent row ancestry mismatch")
                count += 1
                if count > original.acquisition.max_rows:
                    raise ValueError("retained parent row cap exceeded")
                rank = hashlib.sha256(
                    canonical_json(
                        [
                            spec.profile_id,
                            spec.seed,
                            original.id,
                            spec.snapshot_sha256,
                            index,
                            row_sha,
                        ]
                    )
                ).hexdigest()
                if int(rank, 16) % 100:
                    continue
                if len(record.raw) > spec.max_record_bytes:
                    excluded += 1
                    continue
                size += len(record.raw)
                if (
                    size > source.acquisition.max_bytes
                    or len(selected) >= source.acquisition.max_rows
                ):
                    raise ValueError("retained subset row/byte cap exceeded")
                if not record.raw.endswith(b"\n"):
                    raise ValueError("selected retained row must end with LF")
                digest.update(record.raw)
                if stream is not None:
                    stream.write(record.raw)
                selected.append(dict(receipt))
            if stream is not None:
                import os

                stream.flush()
                os.fsync(stream.fileno())
        if not selected:
            raise ValueError("retained subset selected no rows")
        if (
            sha256_file(raw_path) != file["sha256"]
            or sha256_file(parent / "manifest.json") != spec.manifest_sha256
        ):
            raise ValueError("retained parent changed during selection")
        files = [{"path": file["path"], "size": size, "sha256": digest.hexdigest()}]
        retrieval = {
            "retained_subset": {
                "parent_snapshot_sha256": spec.snapshot_sha256,
                "parent_manifest_sha256": spec.manifest_sha256,
                "parent_declaration_sha256": manifest["declaration_sha256"],
                "profile_id": spec.profile_id,
                "seed": spec.seed,
                "hash_modulus": 100,
                "scanned_rows": count,
                "selected_rows": len(selected),
                "excluded_long_rows": excluded,
                "max_record_bytes": spec.max_record_bytes,
            },
            "shards": [
                {
                    "output_path": file["path"],
                    "source_shard_path": shard["source_shard_path"],
                    "source_shard_sha256": shard["source_shard_sha256"],
                    "selected_rows": selected,
                }
            ],
        }
        return files, retrieval
