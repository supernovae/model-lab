"""Bounded, authenticated source-token denominator for a frozen corpus release.

The SQLite index and tokenizer batches live outside the immutable release. Source
labels overlap intentionally: a document is deduplicated within each domain.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.bounded_metadata import metadata_operation
from klab.corpus.build_stages import implementation_digests
from klab.corpus.family_inventory_rows import join_family_documents
from klab.corpus.progress import memory_bytes
from klab.corpus.release import ReleaseReader
from klab.corpus.shards import range_file_identity
from klab.corpus.token_denominator_identity import (
    TokenInputs,
    authenticate_token_inputs,
    safe_measurement_path,
)
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.corpus.readiness_policy import CorpusReadinessPolicy

_VERSION = 1
_SELECTION = {
    "split": "train",
    "drop_reason": None,
    "deduplicate_by": ["domain", "content_sha256"],
    "add_special_tokens": False,
    "padding": False,
    "truncation": False,
}
_DOMAIN_FIELDS = (
    "eligible_documents",
    "distinct_documents",
    "source_bytes",
    "source_tokens",
)


def _policy(path: Path) -> tuple[CorpusReadinessPolicy, str]:
    # Lazy: readiness_policy imports this module to share the streaming primitive.
    from klab.corpus.readiness_policy import CorpusReadinessPolicy
    from klab.documents import _ExactUniqueLoader

    raw = path.read_bytes()
    policy = CorpusReadinessPolicy.model_validate(
        yaml.load(raw, Loader=_ExactUniqueLoader)
    )
    if not policy.min_unique_train_tokens_by_domain and not (
        policy.passes is not None and policy.passes.basis == "tokens"
    ):
        raise ValueError("source-token measurement requires a token-domain policy")
    return policy, hashlib.sha256(raw).hexdigest()


def _domains(policy: CorpusReadinessPolicy) -> list[str]:
    requested = set(policy.min_unique_train_bytes_by_domain)
    requested.update(policy.min_unique_train_tokens_by_domain)
    if policy.passes is not None:
        requested.update(policy.passes.mixture)
    return sorted(requested)


def _scratch_dir() -> Path:
    from klab.workdir import resolve_work_dir

    return resolve_work_dir(None) / "scratch" / "source-token-denominator"


def _scan_documents(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class _DocumentScan:
    """Track every byte and record in a single pass over a release member."""

    def __init__(self, path: Path):
        self.path = path
        self.digest = hashlib.sha256()
        self.size = self.rows_scanned = self.kept_documents = 0

    def rows(self) -> Iterator[dict]:
        with self.path.open("rb") as stream:
            for line in stream:
                self.digest.update(line)
                self.size += len(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows_scanned += 1
                self.kept_documents += row["drop_reason"] is None
                yield row


def _inventory(manifest: dict[str, Any], digest: str, size: int) -> None:
    document = manifest["files"]["documents.jsonl"]
    if document["sha256"] != digest or document["size"] != size:
        raise ValueError("streamed documents differ from release inventory")


def _implementation() -> tuple[str, str]:
    return sha256_file(Path(__file__)), canonical_digest(
        implementation_digests(
            "token_denominator_identity.py", "family_inventory_rows.py"
        )
    )


def _scientific(receipt: dict[str, Any]) -> str:
    fields = (
        "measurement_version",
        "release_id",
        "release_manifest_sha256",
        "documents_sha256",
        "documents_size",
        "tokenizer_sha256",
        "tokenizer_manifest_sha256",
        "policy_sha256",
        "policy",
        "selection_definition",
        "implementation_sha256",
        "identity_sha256",
        "domains",
        "rows_scanned",
    )
    from klab.canonical import domain_digest

    bound = (
        {
            key: receipt[key]
            for key in (
                "tokenizer_origin_release_id",
                "tokenizer_origin_manifest_sha256",
                "tokenizer_origin_documents_sha256",
                "tokenizer_origin_documents_size",
                "tokenizer_config_sha256",
                "family_inventory_sha256",
                "family_inventory_rows",
            )
        }
        if "tokenizer_origin_release_id" in receipt
        else {}
    )
    return domain_digest(
        "source-token-denominator-v1",
        {
            **{name: receipt[name] for name in fields},
            **bound,
            "evidence": None,
        },
    )


def _measure_source_domains(
    release: ReleaseReader,
    tokenizer: Path | None,
    policy: CorpusReadinessPolicy,
    *,
    batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    scratch: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    all_domains: bool = False,
    max_document_source_bytes: int | None = TOKENIZER_BATCH_SOURCE_BYTES,
    family_inventory: tuple[Path, str] | None = None,
) -> dict[str, Any]:
    """Stream and authenticate documents once; keep dedup keys on disk.

    Caller authenticates release and tokenizer provenance. ``all_domains``
    preserves mixed-readiness reporting; unlimited single-record encoding is
    only for that legacy path, never the public denominator.
    """
    batch_documents, batch_source_bytes = validate_tokenizer_batch_limits(
        batch_documents, batch_source_bytes
    )
    document_path = safe_measurement_path(release.member("documents.jsonl"))
    manifest = release.manifest
    model = (
        load_tokenizer(safe_measurement_path(Path(tokenizer)))
        if tokenizer is not None
        else None
    )
    if model is not None:
        model.no_padding()
        model.no_truncation()
    requested = _domains(policy)
    domains: dict[str, dict[str, int | None]] = {
        domain: dict.fromkeys(_DOMAIN_FIELDS, 0) for domain in requested
    }
    if model is None:
        for counts in domains.values():
            counts["source_tokens"] = None
    scratch_path = safe_measurement_path(scratch or _scratch_dir())
    scratch_path.mkdir(parents=True, exist_ok=True)
    next_report = 10_000
    texts: list[str] = []
    labels: list[list[str]] = []
    batch_bytes = 0
    batch_count = 0
    scan = _DocumentScan(document_path)

    def flush() -> None:
        nonlocal batch_bytes, batch_count
        if not texts:
            return
        assert model is not None
        encoded = (
            model.encode(texts[0], add_special_tokens=False)
            if len(texts) == 1
            else model.encode_batch(texts, add_special_tokens=False)
        )
        encodings = [encoded] if len(texts) == 1 else encoded
        for encoding, memberships in zip(encodings, labels, strict=True):
            count = len(encoding.ids)
            for domain in memberships:
                current = domains[domain]["source_tokens"]
                assert current is not None
                domains[domain]["source_tokens"] = current + count
        texts.clear()
        labels.clear()
        batch_bytes = 0
        batch_count += 1

    with tempfile.TemporaryDirectory(prefix="measurement-", dir=scratch_path) as temp:
        connection = sqlite3.connect(str(Path(temp) / "seen.sqlite"))
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute(
                "CREATE TABLE seen (domain TEXT NOT NULL, digest TEXT NOT NULL, "
                "PRIMARY KEY (domain, digest)) WITHOUT ROWID"
            )
            with ExitStack() as streams:
                documents = streams.enter_context(closing(scan.rows()))
                checked = (
                    streams.enter_context(
                        closing(
                            _checked_family_documents(
                                documents, *family_inventory, policy, connection
                            )
                        )
                    )
                    if family_inventory is not None
                    else documents
                )
                for row in checked:
                    if progress is not None and scan.rows_scanned >= next_report:
                        progress(
                            {"rows_scanned": scan.rows_scanned, "domains": domains}
                        )
                        next_report = scan.rows_scanned + 10_000
                    if row["split"] != "train" or row["drop_reason"] is not None:
                        continue
                    applicable = (
                        set(row["domains"])
                        if all_domains
                        else set(row["domains"]).intersection(domains)
                    )
                    if not applicable:
                        continue
                    for domain in applicable:
                        if domain not in domains:
                            domains[domain] = {
                                "eligible_documents": 0,
                                "distinct_documents": 0,
                                "source_bytes": 0,
                                "source_tokens": 0 if model is not None else None,
                            }
                    text = row["text"]
                    length = len(text.encode("utf-8"))
                    if (
                        model is not None
                        and max_document_source_bytes is not None
                        and length > max_document_source_bytes
                    ):
                        raise ValueError("eligible source document exceeds 1 MiB")
                    new_domains: list[str] = []
                    for domain in sorted(applicable):
                        counts = domains[domain]
                        counts["eligible_documents"] += 1  # type: ignore[operator]
                        cursor = connection.execute(
                            "INSERT OR IGNORE INTO seen VALUES (?, ?)",
                            (domain, row["content_sha256"]),
                        )
                        if cursor.rowcount:
                            new_domains.append(domain)
                            counts["distinct_documents"] += 1  # type: ignore[operator]
                    if not new_domains:
                        continue
                    for domain in new_domains:
                        counts = domains[domain]
                        counts["source_bytes"] += length  # type: ignore[operator]
                    if model is not None:
                        if texts and (
                            len(texts) >= batch_documents
                            or batch_bytes + length > batch_source_bytes
                        ):
                            flush()
                        if length > batch_source_bytes:
                            if max_document_source_bytes is not None:
                                raise ValueError(
                                    "source document exceeds tokenizer batch byte limit"
                                )
                            count = len(
                                model.encode(text, add_special_tokens=False).ids
                            )
                            for domain in new_domains:
                                current = domains[domain]["source_tokens"]
                                assert current is not None
                                domains[domain]["source_tokens"] = current + count
                            batch_count += 1
                            continue
                        texts.append(text)
                        labels.append(new_domains)
                        batch_bytes += length
                        if (
                            len(texts) >= batch_documents
                            or batch_bytes >= batch_source_bytes
                        ):
                            flush()
            flush()
        finally:
            connection.close()
    documents_sha256 = scan.digest.hexdigest()
    _inventory(manifest, documents_sha256, scan.size)
    if progress is not None:
        progress({"rows_scanned": scan.rows_scanned, "domains": domains})
    return {
        "domains": domains,
        "rows_scanned": scan.rows_scanned,
        "documents_sha256": documents_sha256,
        "documents_size": scan.size,
        "kept_documents": scan.kept_documents,
        "batch_count": batch_count,
    }


def _check_locations(
    release: Path,
    tokenizer: Path,
    policy_path: Path,
    output: Path,
    *,
    tokenizer_origin_release: Path | None = None,
    family_inventory: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path | None, Path | None]:
    release, tokenizer, policy_path, output = (
        safe_measurement_path(Path(item))
        for item in (release, tokenizer, policy_path, output)
    )
    if (tokenizer_origin_release is None) != (family_inventory is None):
        raise ValueError(
            "tokenizer origin and family inventory must be declared together"
        )
    origin = (
        safe_measurement_path(Path(tokenizer_origin_release))
        if tokenizer_origin_release is not None
        else None
    )
    inventory = (
        safe_measurement_path(Path(family_inventory)) if family_inventory else None
    )
    if output.suffix in {".partial", ".ready"}:
        raise ValueError("source-token receipt is not complete: reserved partial path")
    sources = (
        release,
        release / "manifest.json",
        tokenizer,
        tokenizer.with_name("tokenizer_manifest.json"),
        policy_path,
        *(
            (
                origin,
                origin / "manifest.json",
                tokenizer.parent.parent / "tokenizer.yaml",
            )
            if origin
            else ()
        ),
        *((inventory,) if inventory else ()),
    )
    if (
        output in sources
        or output.is_relative_to(release)
        or (origin is not None and output.is_relative_to(origin))
        or any(source.is_relative_to(output) for source in sources)
    ):
        raise ValueError("source and output paths conflict")
    return release, tokenizer, policy_path, output, origin, inventory


def _checked_family_documents(
    documents: Iterator[dict],
    inventory: Path,
    expected_digest: str,
    policy: CorpusReadinessPolicy,
    connection: sqlite3.Connection,
) -> Iterator[dict]:
    """Check split and stratum isolation in the same pass that consumes documents."""
    requested = set(_domains(policy))
    connection.execute(
        "CREATE TABLE family_split (id TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE content_split (digest TEXT PRIMARY KEY, split TEXT NOT NULL, "
        "stratum TEXT NOT NULL) WITHOUT ROWID"
    )
    with closing(
        join_family_documents(documents, inventory, expected_sha256=expected_digest)
    ) as pairs:
        for doc, row in pairs:
            family, split, stratum = row["family_id"], row["split"], row["stratum"]
            if any(not value for value in row.values()) or split not in {
                "train",
                "validation",
                "test",
            }:
                raise ValueError("invalid family inventory row")
            inserted = connection.execute(
                "INSERT OR IGNORE INTO family_split VALUES (?, ?)", (family, split)
            )
            if (
                not inserted.rowcount
                and connection.execute(
                    "SELECT split FROM family_split WHERE id = ?", (family,)
                ).fetchone()[0]
                != split
            ):
                raise ValueError("family inventory leaks across splits")
            content = row["content_sha256"]
            inserted = connection.execute(
                "INSERT OR IGNORE INTO content_split VALUES (?, ?, ?)",
                (content, split, stratum),
            )
            if not inserted.rowcount:
                previous_split, previous_stratum = connection.execute(
                    "SELECT split, stratum FROM content_split WHERE digest = ?",
                    (content,),
                ).fetchone()
                if previous_split != split:
                    raise ValueError("identical content leaks across splits")
                if previous_stratum != stratum:
                    raise ValueError("identical content spans measured strata")
            if split == "train" and requested.intersection(doc["domains"]) != {stratum}:
                raise ValueError("training inventory has ambiguous measured stratum")
            yield doc


def read_source_token_receipt(
    path: Path,
    release: Path,
    tokenizer: Path,
    policy_path: Path,
    *,
    tokenizer_origin_release: Path | None = None,
    family_inventory: Path | None = None,
) -> dict[str, Any]:
    """Authenticate a COMPLETE result, including fresh streamed input identity."""
    release, tokenizer, policy_path, path, origin, inventory = _check_locations(
        release,
        tokenizer,
        policy_path,
        path,
        tokenizer_origin_release=tokenizer_origin_release,
        family_inventory=family_inventory,
    )
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (
        not isinstance(receipt, dict)
        or raw != canonical_json(receipt) + b"\n"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("measurement_version") != _VERSION
    ):
        raise ValueError("source-token receipt is not complete or canonical")
    if origin is None and "tokenizer_origin_release_id" in receipt:
        raise ValueError("source-token receipt requires its tokenizer origin")
    policy, policy_sha = _policy(policy_path)
    evidence = receipt.get("evidence")
    if evidence is not None:
        raise ValueError("source-token receipt requires cold input authentication")
    with authenticate_token_inputs(
        release, tokenizer, tokenizer_origin_release=origin
    ) as inputs:
        return _verify_source_receipt(receipt, policy, policy_sha, inputs, inventory)


def _verify_source_receipt(
    receipt: dict[str, Any],
    policy: CorpusReadinessPolicy,
    policy_sha: str,
    inputs: TokenInputs,
    inventory: Path | None,
) -> dict[str, Any]:
    release, bindings = inputs.release, inputs.bindings
    manifest = release.manifest
    implementation, identity = _implementation()
    expected = {
        "release_id": manifest["release_id"],
        "release_manifest_sha256": bindings["release_manifest_sha256"],
        "tokenizer_sha256": bindings["tokenizer_sha256"],
        "tokenizer_manifest_sha256": bindings["tokenizer_manifest_sha256"],
        "policy_sha256": policy_sha,
        "policy": policy.model_dump(mode="json"),
        "selection_definition": _SELECTION,
        "implementation_sha256": implementation,
        "identity_sha256": identity,
        "evidence": bindings["evidence"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "source-token receipt input or implementation binding mismatch"
        )
    inventory_generation = None
    document_path = safe_measurement_path(release.member("documents.jsonl"))
    if inputs.origin is not None:
        assert inventory is not None
        inventory_sha = receipt.get("family_inventory_sha256")
        if not isinstance(inventory_sha, str):
            raise ValueError("source-token receipt family inventory digest is missing")
        inventory_generation = range_file_identity(inventory)
        if any(
            receipt.get(key) != bindings[key]
            for key in (
                "tokenizer_origin_release_id",
                "tokenizer_origin_manifest_sha256",
                "tokenizer_origin_documents_sha256",
                "tokenizer_origin_documents_size",
                "tokenizer_config_sha256",
            )
        ):
            raise ValueError(
                "source-token receipt origin or inventory binding mismatch"
            )
        scan = _DocumentScan(document_path)
        with (
            metadata_operation(_scratch_dir(), max_database_bytes=None) as store,
            closing(scan.rows()) as documents,
            closing(
                _checked_family_documents(
                    documents, inventory, inventory_sha, policy, store.connection
                )
            ) as checked,
        ):
            for _ in checked:
                pass
        if receipt.get("family_inventory_rows") != scan.kept_documents:
            raise ValueError("source-token receipt family inventory count mismatch")
        digest, size = scan.digest.hexdigest(), scan.size
    else:
        digest, size = _scan_documents(document_path)
    _inventory(manifest, digest, size)
    if (receipt.get("documents_sha256"), receipt.get("documents_size")) != (
        digest,
        size,
    ):
        raise ValueError("source-token receipt documents changed")
    domain_counts = receipt.get("domains")
    if not isinstance(domain_counts, dict) or set(domain_counts) != set(
        _domains(policy)
    ):
        raise ValueError("source-token receipt domain inventory mismatch")
    for counts in domain_counts.values():
        if (
            not isinstance(counts, dict)
            or set(counts) != set(_DOMAIN_FIELDS)
            or any(
                type(counts[key]) is not int or counts[key] < 0
                for key in _DOMAIN_FIELDS
            )
            or counts["distinct_documents"] > counts["eligible_documents"]
        ):
            raise ValueError("invalid source-token receipt domain counts")
    if (
        type(receipt.get("rows_scanned")) is not int
        or receipt["rows_scanned"] < 0
        or receipt.get("scientific_sha256") != _scientific(receipt)
    ):
        raise ValueError("source-token receipt scientific identity mismatch")
    if inventory is not None and range_file_identity(inventory) != inventory_generation:
        raise ValueError("measurement inventory changed during source scan")
    inputs.require_unchanged()
    return receipt


def measure_source_tokens(
    release: Path,
    tokenizer: Path,
    policy_path: Path,
    output: Path,
    *,
    tokenizer_origin_release: Path | None = None,
    family_inventory: Path | None = None,
    batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
) -> dict[str, Any]:
    """Publish an exclusive, full-EOF result or leave an interrupted partial."""
    start = time.monotonic()
    batch_documents, batch_source_bytes = validate_tokenizer_batch_limits(
        batch_documents, batch_source_bytes
    )
    release, tokenizer, policy_path, output, origin, inventory = _check_locations(
        release,
        tokenizer,
        policy_path,
        output,
        tokenizer_origin_release=tokenizer_origin_release,
        family_inventory=family_inventory,
    )
    if output.exists():
        return read_source_token_receipt(
            output,
            release,
            tokenizer,
            policy_path,
            tokenizer_origin_release=origin,
            family_inventory=inventory,
        )
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    policy, policy_sha = _policy(policy_path)
    with authenticate_token_inputs(
        release, tokenizer, tokenizer_origin_release=origin
    ) as inputs:
        return _publish_source_measurement(
            inputs,
            tokenizer,
            policy,
            policy_path,
            policy_sha,
            output,
            inventory,
            batch_documents=batch_documents,
            batch_source_bytes=batch_source_bytes,
            start=start,
        )


def _publish_source_measurement(
    inputs: TokenInputs,
    tokenizer: Path,
    policy: CorpusReadinessPolicy,
    policy_path: Path,
    policy_sha: str,
    output: Path,
    inventory: Path | None,
    *,
    batch_documents: int,
    batch_source_bytes: int,
    start: float,
) -> dict[str, Any]:
    release, bindings = inputs.release, inputs.bindings
    manifest = release.manifest
    scratch = output.parent / "scratch" / "source-token-denominator"
    inventory_input = None
    inventory_generation = None
    if inventory is not None:
        inventory_generation = range_file_identity(inventory)
        inventory_input = (inventory, sha256_file(inventory))
    implementation, identity = _implementation()
    launch = uuid.uuid4().hex
    partial = safe_measurement_path(
        output.with_name(f".{output.name}.{launch}.partial")
    )
    ready = safe_measurement_path(output.with_name(f".{output.name}.{launch}.ready"))
    progress_path = safe_measurement_path(
        output.with_name(f".{output.name}.{launch}.progress.jsonl")
    )
    last_report = start
    last_update: dict[str, Any] = {}

    def report(update: dict[str, Any], *, status: str = "RUNNING") -> None:
        nonlocal last_report, last_update
        last_update = update
        now = time.monotonic()
        if status == "RUNNING" and now - last_report < 10:
            return
        last_report = now
        totals = update.get("domains", {})
        message = {
            "launch": launch,
            "status": status,
            "rows_scanned": update.get("rows_scanned", 0),
            "eligible_documents": sum(
                item["eligible_documents"] for item in totals.values()
            ),
            "distinct_documents": sum(
                item["distinct_documents"] for item in totals.values()
            ),
            "source_bytes": sum(item["source_bytes"] for item in totals.values()),
            "source_tokens": sum(
                item["source_tokens"] or 0 for item in totals.values()
            ),
            "elapsed_seconds": now - start,
            "rows_per_second": update.get("rows_scanned", 0) / max(now - start, 1e-9),
            "peak_rss_kib": memory_bytes()[1] // 1024,
        }
        record = canonical_json(message) + b"\n"
        with progress_path.open("ab") as log:
            log.write(record)
            log.flush()
            os.fsync(log.fileno())
        print(record.decode().rstrip(), file=sys.stderr, flush=True)

    with partial.open("xb") as temporary:
        temporary.write(
            canonical_json(
                {
                    "measurement_version": _VERSION,
                    "status": "INTERRUPTED",
                    "launch": launch,
                }
            )
            + b"\n"
        )
        temporary.flush()
        os.fsync(temporary.fileno())
    report({}, status="STARTED")
    published_identity: tuple[int, int] | None = None
    try:
        measured = _measure_source_domains(
            release,
            tokenizer,
            policy,
            batch_documents=batch_documents,
            batch_source_bytes=batch_source_bytes,
            max_document_source_bytes=batch_source_bytes,
            progress=report,
            scratch=scratch,
            family_inventory=inventory_input,
        )
        receipt = {
            "measurement_version": _VERSION,
            "status": "COMPLETE",
            "release_id": manifest["release_id"],
            "release_manifest_sha256": bindings["release_manifest_sha256"],
            "documents_sha256": measured["documents_sha256"],
            "documents_size": measured["documents_size"],
            "tokenizer_sha256": bindings["tokenizer_sha256"],
            "tokenizer_manifest_sha256": bindings["tokenizer_manifest_sha256"],
            "policy_sha256": policy_sha,
            "policy": policy.model_dump(mode="json"),
            "selection_definition": _SELECTION,
            "implementation_sha256": implementation,
            "identity_sha256": identity,
            "domains": measured["domains"],
            "rows_scanned": measured["rows_scanned"],
            "evidence": bindings["evidence"],
            "policy_path": str(policy_path),
            "operational": {
                "launch": launch,
                "wall_seconds": time.monotonic() - start,
                "peak_rss_kib": memory_bytes()[1] // 1024,
                "batch_documents": batch_documents,
                "batch_source_bytes": batch_source_bytes,
                "batch_count": measured["batch_count"],
                "progress_path": str(progress_path),
            },
        }
        if inputs.origin is not None:
            assert inventory_input is not None
            receipt.update(
                {
                    key: bindings[key]
                    for key in (
                        "tokenizer_origin_release_id",
                        "tokenizer_origin_manifest_sha256",
                        "tokenizer_origin_documents_sha256",
                        "tokenizer_origin_documents_size",
                        "tokenizer_config_sha256",
                    )
                }
            )
            receipt.update(
                family_inventory_sha256=inventory_input[1],
                family_inventory_rows=measured["kept_documents"],
            )
        receipt["scientific_sha256"] = _scientific(receipt)
        # The interrupted marker is never rewritten as COMPLETE. A separate
        # fsynced sibling is linked exclusively into place after all checks.
        with ready.open("xb") as temporary:
            temporary.write(canonical_json(receipt) + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            ready_stat = os.fstat(temporary.fileno())
        if sha256_file(
            safe_measurement_path(policy_path)
        ) != policy_sha or _implementation() != (implementation, identity):
            raise ValueError("measurement input changed during source scan")
        if (
            inventory is not None
            and range_file_identity(inventory) != inventory_generation
        ):
            raise ValueError("measurement inventory changed during source scan")
        inputs.require_unchanged()
        published_identity = (ready_stat.st_dev, ready_stat.st_ino)
        os.link(ready, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        report(measured, status="COMPLETE")
        ready.unlink()
        partial.unlink()
        return receipt
    except BaseException:
        if published_identity is not None:
            try:
                final_stat = output.lstat()
            except FileNotFoundError:
                pass
            else:
                if (final_stat.st_dev, final_stat.st_ino) == published_identity:
                    output.unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        report(last_update, status="INTERRUPTED")
        raise
