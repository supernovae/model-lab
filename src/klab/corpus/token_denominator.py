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
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from klab.canonical import canonical_json
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.progress import memory_bytes
from klab.corpus.release import _verification_operation
from klab.corpus.token_denominator_identity import _authenticate_inputs, _safe_path
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.campaign.policy import CorpusReadinessPolicy

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


def _operation(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with _verification_operation():
            return function(*args, **kwargs)

    return guarded


def _policy(path: Path) -> tuple[CorpusReadinessPolicy, str]:
    # Lazy: campaign.policy imports this module to share the streaming primitive.
    from klab.campaign.policy import CorpusReadinessPolicy
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


def _inventory(manifest: dict[str, Any], digest: str, size: int) -> None:
    document = manifest["files"]["documents.jsonl"]
    if document["sha256"] != digest or document["size"] != size:
        raise ValueError("streamed documents differ from release inventory")


def _implementation() -> tuple[str, str]:
    return sha256_file(Path(__file__)), sha256_file(
        Path(__file__).with_name("token_denominator_identity.py")
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
    from klab.campaign.state import digest

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
    return digest(
        "source-token-denominator-v1",
        {
            **{name: receipt[name] for name in fields},
            **bound,
            "evidence": None,
        },
    )


def _measure_source_domains(
    release: Path,
    tokenizer: Path | None,
    policy: CorpusReadinessPolicy,
    *,
    batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    scratch: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    all_domains: bool = False,
    max_document_source_bytes: int | None = TOKENIZER_BATCH_SOURCE_BYTES,
) -> dict[str, Any]:
    """Stream and authenticate documents once; keep dedup keys on disk.

    Caller authenticates release and tokenizer provenance. ``all_domains``
    preserves mixed-readiness reporting; unlimited single-record encoding is
    only for that legacy path, never the public denominator.
    """
    batch_documents, batch_source_bytes = validate_tokenizer_batch_limits(
        batch_documents, batch_source_bytes
    )
    release = _safe_path(Path(release))
    document_path = _safe_path(release / "documents.jsonl")
    manifest_path = _safe_path(release / "manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = (
        load_tokenizer(_safe_path(Path(tokenizer))) if tokenizer is not None else None
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
    scratch_path = _safe_path(scratch or _scratch_dir())
    scratch_path.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    labels: list[list[str]] = []
    batch_bytes = 0
    batch_count = 0
    rows_scanned = 0
    document_size = 0
    digest = hashlib.sha256()

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
            with document_path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    document_size += len(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_scanned += 1
                    if progress is not None and rows_scanned % 10_000 == 0:
                        progress({"rows_scanned": rows_scanned, "domains": domains})
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
    documents_sha256 = digest.hexdigest()
    _inventory(manifest, documents_sha256, document_size)
    if progress is not None:
        progress({"rows_scanned": rows_scanned, "domains": domains})
    return {
        "domains": domains,
        "rows_scanned": rows_scanned,
        "documents_sha256": documents_sha256,
        "documents_size": document_size,
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
        _safe_path(Path(item)) for item in (release, tokenizer, policy_path, output)
    )
    if (tokenizer_origin_release is None) != (family_inventory is None):
        raise ValueError(
            "tokenizer origin and family inventory must be declared together"
        )
    origin = (
        _safe_path(Path(tokenizer_origin_release))
        if tokenizer_origin_release is not None
        else None
    )
    inventory = _safe_path(Path(family_inventory)) if family_inventory else None
    if output.suffix in {".partial", ".ready"}:
        raise ValueError("source-token receipt is not complete: reserved partial path")
    sources = (
        release,
        release / "manifest.json",
        release / "documents.jsonl",
        tokenizer,
        tokenizer.with_name("tokenizer_manifest.json"),
        policy_path,
        *(
            (
                origin,
                origin / "manifest.json",
                origin / "documents.jsonl",
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


def _family_inventory_binding(
    release: Path, inventory: Path, policy: CorpusReadinessPolicy
) -> tuple[str, int]:
    """Bind every kept document to one reviewed family, split and measured stratum."""
    documents = {}
    for record in records_from_path(release / "documents.jsonl"):
        doc = record.value
        if doc["drop_reason"] is not None:
            continue
        document_id = doc["document_id"]
        if document_id in documents:
            raise ValueError("duplicate kept document ID")
        documents[document_id] = (
            doc["split"],
            doc["content_sha256"],
            set(doc["domains"]),
        )
    families: dict[str, str] = {}
    content_splits: dict[str, str] = {}
    content_strata: dict[str, str] = {}
    requested = set(_domains(policy))
    count = 0
    for record in records_from_path(inventory):
        row = record.value
        if (
            not isinstance(row, dict)
            or set(row)
            != {"document_id", "family_id", "split", "stratum", "content_sha256"}
            or any(not isinstance(value, str) or not value for value in row.values())
            or row["split"] not in {"train", "validation", "test"}
        ):
            raise ValueError("invalid family inventory row")
        doc = documents.pop(row["document_id"], None)
        if (
            doc is None
            or row["split"] != doc[0]
            or row["content_sha256"] != doc[1]
            or row["stratum"] not in doc[2]
        ):
            raise ValueError("family inventory differs from measured release")
        family = row["family_id"]
        if family in families and families[family] != row["split"]:
            raise ValueError("family inventory leaks across splits")
        families[family] = row["split"]
        content = row["content_sha256"]
        if content in content_splits and content_splits[content] != row["split"]:
            raise ValueError("identical content leaks across splits")
        content_splits[content] = row["split"]
        if content in content_strata and content_strata[content] != row["stratum"]:
            raise ValueError("identical content spans measured strata")
        content_strata[content] = row["stratum"]
        if row["split"] == "train" and doc[2].intersection(requested) != {
            row["stratum"]
        }:
            raise ValueError("training inventory has ambiguous measured stratum")
        count += 1
    if documents:
        raise ValueError("family inventory omits kept documents")
    return sha256_file(inventory), count


@_operation
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
    manifest, bindings = _authenticate_inputs(
        release,
        tokenizer,
        tokenizer_origin_release=origin,
    )
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
    if origin is not None:
        assert inventory is not None
        inventory_sha, inventory_rows = _family_inventory_binding(
            release, inventory, policy
        )
        expected.update(
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
        expected.update(
            family_inventory_sha256=inventory_sha,
            family_inventory_rows=inventory_rows,
        )
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "source-token receipt input or implementation binding mismatch"
        )
    digest, size = _scan_documents(_safe_path(release / "documents.jsonl"))
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
    return receipt


@_operation
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
    policy, policy_sha = _policy(policy_path)
    manifest, bindings = _authenticate_inputs(
        release,
        tokenizer,
        tokenizer_origin_release=origin,
    )
    inventory_binding = (
        _family_inventory_binding(release, inventory, policy)
        if inventory is not None
        else None
    )
    if output.exists():
        existing = read_source_token_receipt(
            output,
            release,
            tokenizer,
            policy_path,
            tokenizer_origin_release=origin,
            family_inventory=inventory,
        )
        if existing["evidence"] != bindings["evidence"]:
            raise ValueError("existing source-token receipt evidence mode mismatch")
        return existing
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    implementation, identity = _implementation()
    launch = uuid.uuid4().hex
    partial = _safe_path(output.with_name(f".{output.name}.{launch}.partial"))
    ready = _safe_path(output.with_name(f".{output.name}.{launch}.ready"))
    progress_path = _safe_path(
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
            scratch=output.parent / "scratch" / "source-token-denominator",
        )
        stable_files = {
            release / "manifest.json": bindings["release_manifest_sha256"],
            release / "documents.jsonl": measured["documents_sha256"],
            tokenizer: bindings["tokenizer_sha256"],
            tokenizer.with_name("tokenizer_manifest.json"): bindings[
                "tokenizer_manifest_sha256"
            ],
            policy_path: policy_sha,
            Path(__file__): implementation,
            Path(__file__).with_name("token_denominator_identity.py"): identity,
        }
        if origin is not None:
            assert inventory is not None and inventory_binding is not None
            stable_files.update(
                {
                    origin / "manifest.json": bindings[
                        "tokenizer_origin_manifest_sha256"
                    ],
                    origin / "documents.jsonl": bindings[
                        "tokenizer_origin_documents_sha256"
                    ],
                    tokenizer.parent.parent / "tokenizer.yaml": bindings[
                        "tokenizer_config_sha256"
                    ],
                    inventory: inventory_binding[0],
                }
            )
        if any(
            sha256_file(_safe_path(path)) != expected
            for path, expected in stable_files.items()
        ):
            raise ValueError("measurement input changed during source scan")
        if origin is not None:
            post_manifest, post_bindings = _authenticate_inputs(
                release, tokenizer, tokenizer_origin_release=origin
            )
            if post_manifest != manifest or post_bindings != bindings:
                raise ValueError("measurement input changed during source scan")
            assert inventory is not None and inventory_binding is not None
            if (
                _family_inventory_binding(release, inventory, policy)
                != inventory_binding
            ):
                raise ValueError("measurement inventory changed during source scan")
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
        if origin is not None:
            assert inventory_binding is not None
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
                family_inventory_sha256=inventory_binding[0],
                family_inventory_rows=inventory_binding[1],
            )
        receipt["scientific_sha256"] = _scientific(receipt)
        # The interrupted marker is never rewritten as COMPLETE. A separate
        # fsynced sibling is linked exclusively into place after all checks.
        with ready.open("xb") as temporary:
            temporary.write(canonical_json(receipt) + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            ready_stat = os.fstat(temporary.fileno())
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
