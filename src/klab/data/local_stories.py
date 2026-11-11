"""Pinned, offline-verifiable TinyStories snapshot acquisition and replay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

from klab.config.models import DatasetConfig
from klab.hf_auth import HUB_ACCESS_ERRORS, hub_auth_kwargs, raise_for_hub_auth
from klab.local_stories_pins import LICENSE, REVISION

DATASET = "roneneldan/TinyStories"


def _hub_stream(split: str, cache_dir: Path) -> Iterator[dict[str, object]]:
    from datasets import load_dataset

    auth = hub_auth_kwargs()
    try:
        stream = load_dataset(
            DATASET,
            name="default",
            split=split,
            revision=REVISION,
            streaming=True,
            cache_dir=str(cache_dir),
            **auth,
        )
        yield from stream
    except HUB_ACCESS_ERRORS as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _row(raw: bytes, split: str) -> tuple[int, str, bytes]:
    try:
        record = json.loads(raw.decode("utf-8"))
        if (
            set(record) != {"ordinal", "text"}
            or type(record["ordinal"]) is not int
            or not isinstance(record["text"], str)
            or not record["text"]
        ):
            raise ValueError("invalid story record")
        return record["ordinal"], record["text"], record["text"].encode("utf-8")
    except (UnicodeError, json.JSONDecodeError, TypeError, KeyError) as error:
        raise ValueError(f"invalid {split} story record") from error


def verify_snapshot(config: DatasetConfig) -> dict[str, object]:
    """Verify all saved bytes, order, provenance, counts and split disjointness."""
    if config.source != "local_stories":
        raise ValueError("snapshot verification requires local_stories")
    assert config.source_manifest_path is not None
    try:
        manifest = json.loads(config.source_manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["schema_version"] != 1
            or manifest["source"] != DATASET
            or manifest["revision"] != config.revision
            or manifest["license"] != config.license
            or set(manifest["splits"]) != {"train", "validation"}
        ):
            raise ValueError("snapshot manifest provenance mismatch")
        seen: set[bytes] = set()
        ordinals: dict[str, set[int]] = {"train": set(), "validation": set()}
        for split, path in (
            ("train", config.train_path),
            ("validation", config.validation_path),
        ):
            assert path is not None
            details = manifest["splits"][split]
            if (
                path.resolve()
                != (config.source_manifest_path.parent / details["path"]).resolve()
            ):
                raise ValueError(f"{split} snapshot path mismatch")
            file_hash = hashlib.sha256()
            content_hash = hashlib.sha256()
            count = text_bytes = previous = 0
            with path.open("rb") as handle:
                for raw in handle:
                    if not raw.endswith(b"\n"):
                        raise ValueError(f"{split} snapshot has partial final record")
                    file_hash.update(raw)
                    ordinal, _, encoded = _row(raw, split)
                    if ordinal < previous:
                        raise ValueError(f"{split} source ordinals out of order")
                    if (
                        ordinal >= details["source_records_consumed"]
                        or ordinal in ordinals[split]
                    ):
                        raise ValueError(f"{split} duplicate or invalid source ordinal")
                    ordinals[split].add(ordinal)
                    previous = ordinal + 1
                    digest = hashlib.sha256(encoded).digest()
                    if digest in seen:
                        raise ValueError(
                            "snapshot contains duplicate or cross-split story"
                        )
                    seen.add(digest)
                    content_hash.update(len(encoded).to_bytes(8, "big"))
                    content_hash.update(encoded)
                    text_bytes += len(encoded)
                    count += 1
            if (
                count != details["count"]
                or text_bytes != details["text_bytes"]
                or file_hash.hexdigest() != details["sha256"]
                or content_hash.hexdigest() != details["content_sha256"]
            ):
                raise ValueError(f"{split} snapshot content or digest mismatch")
        excluded = config.source_manifest_path.parent / manifest["excluded"]["path"]
        if _digest(excluded) != manifest["excluded"]["sha256"]:
            raise ValueError("snapshot excluded ordinal digest mismatch")
        excluded_counts = {
            "train": {"duplicate": 0, "overlap": 0, "empty": 0},
            "validation": {"duplicate": 0, "overlap": 0, "empty": 0},
        }
        with excluded.open("rb") as handle:
            excluded_count = 0
            for raw in handle:
                try:
                    entry = json.loads(raw.decode("utf-8"))
                    split = entry["split"]
                    reason = entry["reason"]
                    ordinal = entry["ordinal"]
                    if (
                        not raw.endswith(b"\n")
                        or set(entry) != {"split", "ordinal", "reason"}
                        or type(ordinal) is not int
                        or ordinal < 0
                        or ordinal
                        >= manifest["splits"][split]["source_records_consumed"]
                    ):
                        raise ValueError("invalid excluded source ordinal")
                    if ordinal in ordinals[split]:
                        raise ValueError(f"{split} repeated source ordinal")
                    ordinals[split].add(ordinal)
                    excluded_counts[split][reason] += 1
                except (
                    KeyError,
                    TypeError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as error:
                    raise ValueError("invalid excluded source ordinal") from error
                excluded_count += 1
        if excluded_count != manifest["excluded"]["count"]:
            raise ValueError("snapshot excluded ordinal count mismatch")
        for split in ("train", "validation"):
            details = manifest["splits"][split]
            if (
                details["duplicates"] != excluded_counts[split]["duplicate"]
                or details["overlap"] != excluded_counts[split]["overlap"]
                or details["empty"] != excluded_counts[split]["empty"]
                or details["count"] + sum(excluded_counts[split].values())
                != details["source_records_consumed"]
            ):
                raise ValueError(f"{split} snapshot excluded ordinals mismatch")
            if len(ordinals[split]) != details["source_records_consumed"]:
                raise ValueError(f"{split} source ordinal coverage mismatch")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("incomplete or invalid snapshot manifest/files") from error
    return manifest


def iter_local_stories(config: DatasetConfig, split: str) -> Iterator[str]:
    verify_snapshot(config)
    path = config.train_path if split == "train" else config.validation_path
    assert path is not None
    with path.open("rb") as handle:
        for raw in handle:
            yield _row(raw, split)[1]


def snapshot(
    output_dir: Path,
    *,
    train_count: int = 1_000_000,
    validation_count: int = 10_000,
    cache_dir: Path | None = None,
) -> Path:
    """Acquire two pinned streams once; publish only an exact, verified snapshot."""
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("snapshot counts must be positive")
    if output_dir.exists():
        raise FileExistsError(f"snapshot destination already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(
            f"snapshot parent directory is missing: {output_dir.parent}"
        )

    from klab.engram.packs import _rename_noreplace

    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=output_dir.parent))
    try:
        connection = sqlite3.connect(staging / "dedup.sqlite")
        connection.execute(
            "CREATE TABLE stories (digest BLOB PRIMARY KEY, text BLOB NOT NULL, split TEXT NOT NULL)"
        )
        splits: dict[str, dict[str, object]] = {}
        excluded_count = 0
        with (staging / "excluded.jsonl").open("wb") as excluded:
            for split, target in (
                ("train", train_count),
                ("validation", validation_count),
            ):
                path = staging / f"{split}.jsonl"
                content_hash = hashlib.sha256()
                count = text_bytes = duplicates = overlap = empty = 0
                stream = _hub_stream(split, cache_dir or output_dir.parent / "hf-cache")
                with path.open("wb") as handle:
                    for ordinal, record in enumerate(stream):
                        text = record.get("text")
                        if not isinstance(text, str):
                            raise TypeError(
                                f"{split} source has non-string text at ordinal {ordinal}"
                            )
                        encoded = text.encode("utf-8")
                        if not encoded:
                            reason = "empty"
                        else:
                            digest = hashlib.sha256(encoded).digest()
                            prior = connection.execute(
                                "SELECT text, split FROM stories WHERE digest = ?",
                                (digest,),
                            ).fetchone()
                            if prior is not None:
                                if prior[0] != encoded:
                                    raise ValueError(
                                        "SHA-256 collision between different stories"
                                    )
                                reason = "overlap" if prior[1] != split else "duplicate"
                            else:
                                reason = ""
                        if reason:
                            excluded.write(
                                (
                                    json.dumps(
                                        {
                                            "split": split,
                                            "ordinal": ordinal,
                                            "reason": reason,
                                        }
                                    )
                                    + "\n"
                                ).encode("utf-8")
                            )
                            excluded_count += 1
                            duplicates += reason == "duplicate"
                            empty += reason == "empty"
                            overlap += reason == "overlap"
                            continue
                        connection.execute(
                            "INSERT INTO stories VALUES (?, ?, ?)",
                            (digest, encoded, split),
                        )
                        raw = (
                            json.dumps(
                                {"ordinal": ordinal, "text": text},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                        handle.write(raw)
                        content_hash.update(len(encoded).to_bytes(8, "big"))
                        content_hash.update(encoded)
                        count += 1
                        text_bytes += len(encoded)
                        if count == target:
                            break
                    if count != target:
                        raise ValueError(
                            f"{split} stream exhausted: {count} distinct stories, need {target}"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                splits[split] = {
                    "path": path.name,
                    "count": count,
                    "text_bytes": text_bytes,
                    "sha256": _digest(path),
                    "content_sha256": content_hash.hexdigest(),
                    "duplicates": duplicates,
                    "overlap": overlap,
                    "empty": empty,
                    "source_records_consumed": ordinal + 1,
                }
            excluded.flush()
            os.fsync(excluded.fileno())
        connection.close()
        (staging / "dedup.sqlite").unlink()
        excluded_path = staging / "excluded.jsonl"
        manifest = {
            "schema_version": 1,
            "source": DATASET,
            "revision": REVISION,
            "license": LICENSE,
            "splits": splits,
            "excluded": {
                "path": excluded_path.name,
                "count": excluded_count,
                "sha256": _digest(excluded_path),
            },
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("wb") as handle:
            handle.write(
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        config = DatasetConfig(
            source="local_stories",
            revision=REVISION,
            license=LICENSE,
            cache_dir=output_dir.parent,
            train_path=staging / "train.jsonl",
            validation_path=staging / "validation.jsonl",
            source_manifest_path=manifest_path,
            train_max_documents=train_count,
            validation_max_documents=validation_count,
            train_max_tokens=1,
            validation_max_tokens=1,
        )
        verify_snapshot(config)
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(staging, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return output_dir / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)
