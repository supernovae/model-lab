"""Pinned source acquisition and independently verifiable immutable byte snapshots."""

from __future__ import annotations

import bz2
import fnmatch
import gzip
import hashlib
import html
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import urllib.request
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urljoin, urlparse

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.acquisition_parts import (
    AcquisitionParts,
    CheckedCheckpoint,
    HFParts,
    PDFParts,
    PinnedBody,
    PinnedBodyParts,
    acquisition_staging,
    check_checkpoint,
    pinned_pdf_path,
    reclaim_disposable_staging,
)
from klab.corpus.bounded_metadata import read_metadata
from klab.corpus.build_stages import StageStore, run_stage
from klab.corpus.project import (
    IMMUTABLE_SOURCE_KINDS,
    RETAINED_MAX_BYTES,
    RETAINED_MAX_ROWS,
    AcquisitionIdentity,
    ArchiveAcquisition,
    GitAcquisition,
    HttpAcquisition,
    HuggingFaceAcquisition,
    LocalAcquisition,
    MediaWikiDumpAcquisition,
    PdfAcquisition,
    Project,
    SourceDeclaration,
    WikimediaDumpAcquisition,
    acquisition_identity,
    acquisition_identity_payload,
    project_path,
    safe_name,
)
from klab.corpus.release_verify import worker_count
from klab.corpus.shards import ShardPlan, declare_stage, run_sharded
from klab.corpus.transport_budget import (
    PINNED_FILE_KINDS,
    TransportBudget,
    pinned_transfers,
    source_file_name,
)
from klab.corpus.transport_cache import (
    BodyKey,
    TransportCache,
    git_blob_hasher,
    resolve_transport_cache,
)
from klab.hf_auth import hub_access_errors, hub_auth_kwargs, raise_for_hub_auth
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked, write_canonical_json
from klab.training.manifest import sha256_file
from klab.verification_proofs import artifact_fingerprint

if TYPE_CHECKING:
    from klab.corpus.wikimedia_ranges import WikimediaRange
    from klab.verification_proofs import ProofStore, VerificationMode

ADAPTER_VERSION = "corpus-acquisition-v1"
_WIKIMEDIA_DECODE_STAGE = declare_stage("wikimedia member decode")
_WIKIMEDIA_PARSE_STAGE = declare_stage("wikimedia member parse")


class _AcquisitionResources:
    """Logical file bytes and inode entries in this source's private staging tree.

    Observe at every growth boundary before deletion. Expanded bytes are bytes
    returned by the JSONL decompressor, including a rejected overlong line;
    Parquet has no equivalent stream reading and reports unavailable explicitly.
    """

    def __init__(
        self,
        root: Path,
        budget: TransportBudget | None = None,
        *,
        journal_parent: Path | None = None,
    ):
        self.root = root
        self.budget = budget
        self.index: int | None = None
        self.expanded: int | None = 0
        self.peak_bytes = 0
        self.peak_inodes = 0
        self.journal: Path | None = None
        self.journal_parent = journal_parent or root.parent
        self.binding: dict[str, Any] = {}
        self.status = "started"

    def retain(self, source: SourceDeclaration, shard_path: str | None = None) -> None:
        directory = self.journal_parent / ".resource-receipts"
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(
                "resource receipt directory must be a nonsymlink directory"
            )
        descriptor, name = tempfile.mkstemp(
            prefix="attempt-", suffix=".json", dir=directory
        )
        os.close(descriptor)
        self.journal = Path(name)
        self.binding = {
            "source_id": source.id,
            "declaration_sha256": declaration_sha256(source),
            "staging_name": self.root.relative_to(self.journal_parent).as_posix(),
        }
        if shard_path is not None:
            self.binding["source_shard_path"] = shard_path
        self.persist()

    @contextmanager
    def attempt(self, source: SourceDeclaration, shard_path: str) -> Iterator[None]:
        if self.budget is None:
            self.retain(source, shard_path)
        try:
            yield
        except BaseException as error:
            self.status = (
                "interrupted"
                if isinstance(error, (OSError, EOFError, KeyboardInterrupt, SystemExit))
                else "failed"
            )
            self.persist()
            raise
        else:
            self.status = "complete"
            self.persist()

    def observe(self) -> None:
        if self.budget is not None and self.budget.observes_storage:
            self.budget.check_disk(self.root, 0)
        try:
            root_info = self.root.lstat()
        except OSError as error:
            self._unavailable(error)
        if not stat.S_ISDIR(root_info.st_mode):
            raise ValueError("required staging root must be a nonsymlink directory")
        sizes = 0
        inodes = 1  # the task-owned staging root itself
        for directory, dirs, files in os.walk(self.root, onerror=self._unavailable):
            for name in dirs + files:
                info = (Path(directory) / name).lstat()
                if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    raise ValueError("staging resource reading contains unsafe entry")
                inodes += 1
                if stat.S_ISREG(info.st_mode):
                    sizes += info.st_size
        self.peak_bytes = max(self.peak_bytes, sizes)
        self.peak_inodes = max(self.peak_inodes, inodes)
        self.persist()

    @staticmethod
    def _unavailable(error: OSError) -> None:
        raise ValueError("required staging resource reading unavailable") from error

    def persist(self) -> None:
        if self.budget is not None and self.index is not None:
            self.budget.record_resources(self.index, self.receipt())
        if self.journal is not None:
            descriptor, name = tempfile.mkstemp(
                prefix=".write-", dir=self.journal.parent
            )
            os.close(descriptor)
            temporary = Path(name)
            try:
                write_canonical_json(
                    temporary,
                    {
                        "schema_version": 1,
                        **self.binding,
                        "status": self.status,
                        "resources": self.receipt(),
                    },
                )
                os.replace(temporary, self.journal)
                sync_directory(self.journal.parent)
            finally:
                temporary.unlink(missing_ok=True)

    def receipt(self) -> dict[str, Any]:
        return {
            "expanded_bytes": self.expanded,
            "peak_staging_bytes": self.peak_bytes,
            "peak_staging_inodes": self.peak_inodes,
        }

    @contextmanager
    def output(self, target: Path) -> Iterator[Any]:
        with target.open("wb") as output:
            try:
                yield output
            finally:
                output.flush()
                self.observe()


def _identity_declaration(
    source: SourceDeclaration | AcquisitionIdentity,
) -> dict[str, Any]:
    declaration = acquisition_identity_payload(source)
    if source.kind == "local":
        declaration["acquisition"]["files"] = [
            {"name": entry.name} for entry in source.acquisition.files
        ]
    return declaration


def declaration_sha256(source: SourceDeclaration | AcquisitionIdentity) -> str:
    """Digest of the acquisition identity: only what determines the acquired bytes."""
    return canonical_digest(_identity_declaration(source))


def _adapter(source: SourceDeclaration | AcquisitionIdentity) -> dict[str, str]:
    return {
        "id": source.kind,
        "version": ADAPTER_VERSION,
        "module_sha256": sha256_file(Path(__file__)),
    }


def _reusable_immutable_adapter(
    source: SourceDeclaration | AcquisitionIdentity, manifest: dict[str, Any]
) -> bool:
    """A pinned, verified snapshot survives incidental adapter-module edits.

    The version still binds adapter semantics; the module digest stays recorded
    in the snapshot for provenance rather than forcing a new network retrieval.
    Mutable local and HTTP sources must still be reacquired.
    """
    actual = manifest["adapter"]
    expected = _adapter(source)
    return actual == expected or (
        source.kind in IMMUTABLE_SOURCE_KINDS
        and actual.get("id") == expected["id"]
        and actual.get("version") == expected["version"]
    )


def _copy_stream(
    source: Any,
    target: Path,
    remaining: int,
    resources: _AcquisitionResources | None = None,
    *,
    exact: bool = False,
    hash_chunk: Callable[[bytes], None] | None = None,
    on_written: Callable[[int], None] | None = None,
) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    limit = remaining if exact else remaining + 1
    with target.open("xb") as output:
        if resources is not None:
            resources.observe()
        while size < limit and (chunk := source.read(min(1024 * 1024, limit - size))):
            size += len(chunk)
            if size > remaining:
                raise ValueError("source exceeds declared max_bytes")
            digest.update(chunk)
            if hash_chunk is not None:
                hash_chunk(chunk)
            output.write(chunk)
            if resources is not None:
                output.flush()
                resources.observe()
        if exact and size != remaining:
            raise ValueError("source ended before declared size")
        output.flush()
        os.fsync(output.fileno())
        if on_written is not None:
            on_written(output.fileno())
    return digest.hexdigest(), size


def _regular(path: Path) -> None:
    if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"source is not a regular, nonsymlink file: {path}")


def _acquire_local(
    source: SourceDeclaration, root: Path, staging: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = source.acquisition
    assert isinstance(spec, LocalAcquisition)
    total = 0
    inventory = []
    for item in sorted(spec.files, key=lambda f: f.name):
        location = Path(item.path)
        location = location if location.is_absolute() else project_path(root, item.path)
        _regular(location)
        with location.open("rb") as stream:
            digest, size = _copy_stream(
                stream, staging / "files" / item.name, spec.max_bytes - total
            )
        total += size
        inventory.append({"path": item.name, "sha256": digest, "size": size})
    return inventory, {"acquisition_paths": [item.model_dump() for item in spec.files]}


def _git(args: list[str], *, cwd: Path | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True
    ).stdout


def _ensure_git_repository(
    source: SourceDeclaration, cache_root: Path, transport: TransportCache | None
) -> tuple[Path, str]:
    """A bare repository holding the pinned commit, and that commit.

    With a transport cache the repository is reused, without any request, when
    `git fsck` re-hashes every object and the pinned commit is reachable;
    otherwise it is rebuilt from the network.
    """
    key = BodyKey("git_repository", source.canonical_uri, revision=source.revision)
    present = False
    with transport.git_lock(key) if transport is not None else nullcontext():
        if transport is not None:
            present = transport.git_has_commit(key)
            repository = transport.git_repository(key, fresh=not present)
        else:
            cache_root.mkdir(parents=True, exist_ok=True)
            repository = cache_root / canonical_digest(
                {"uri": source.canonical_uri, "revision": source.revision}
            )
            if not repository.exists():
                repository.mkdir()
                _git(["init", "--bare", str(repository)])
        if not present:
            _git(
                [
                    "--git-dir",
                    str(repository),
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    source.canonical_uri,
                    source.revision,
                ]
            )
        revision = f"{source.revision}^{{commit}}" if present else "FETCH_HEAD^{commit}"
        commit = (
            _git(["--git-dir", str(repository), "rev-parse", revision]).decode().strip()
        )
        if commit.lower() != source.revision.lower():
            raise ValueError("Git fetched commit differs from pinned revision")
        if transport is not None:
            transport.note_git(key, served_from="cache" if present else "network")
    return repository, commit


def _copy_git_blob(
    process: subprocess.Popen[bytes],
    blob: str,
    target: Path,
    remaining: int,
    *,
    verify_oid: bool,
) -> tuple[str, int]:
    assert process.stdin is not None and process.stdout is not None
    oid = blob.encode("ascii")
    process.stdin.write(oid + b"\n")
    process.stdin.flush()
    header = process.stdout.readline(128).split()
    if len(header) != 3 or header[:2] != [oid, b"blob"] or not header[2].isdigit():
        raise ValueError(f"Git blob header mismatch: {target}")
    size = int(header[2])
    if size > remaining:
        raise ValueError("Git selection exceeds max_bytes")
    # A cached repository is trusted no further than its object ids.
    git_hash = git_blob_hasher(size) if verify_oid else None
    digest, copied = _copy_stream(
        process.stdout,
        target,
        size,
        exact=True,
        hash_chunk=git_hash.update if git_hash is not None else None,
    )
    if process.stdout.read(1) != b"\n":
        raise ValueError(f"Git blob read failed: {target}")
    if git_hash is not None and git_hash.hexdigest() != blob:
        raise ValueError(f"cached Git blob differs from its object id: {target}")
    return digest, copied


def _acquire_git(
    source: SourceDeclaration,
    staging: Path,
    cache_root: Path,
    offline: bool,
    budget: TransportBudget | None = None,
    transport: TransportCache | None = None,
    *,
    parts: AcquisitionParts | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Git")
    spec = source.acquisition
    assert isinstance(spec, GitAcquisition)
    if spec.bounded_blobs is not None:
        if budget is None:
            raise ValueError("bounded Git acquisition requires a transport ledger")
        assert parts is not None
        return _bounded_git_acquire(source, spec, budget, transport, parts=parts)
    repository, commit = _ensure_git_repository(source, cache_root, transport)
    tree = _git(["--git-dir", str(repository), "ls-tree", "-rz", "--full-tree", commit])
    inventory = []
    total = 0
    selected = False
    metadata_path = acquisition_identity(source).nested_metadata_path
    metadata_seen = False
    with subprocess.Popen(
        ["git", "--git-dir", str(repository), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, blob = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="strict")
            is_metadata = path == metadata_path
            if not is_metadata and not any(
                fnmatch.fnmatchcase(path, pattern) for pattern in spec.include
            ):
                continue
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in spec.exclude):
                if is_metadata:
                    raise ValueError(
                        "rights metadata excluded from pinned Git selection"
                    )
                continue
            safe_name(path)
            if mode not in ("100644", "100755") or kind != "blob":
                raise ValueError(
                    f"Git selection contains unsafe symlink/submodule: {path}"
                )
            if is_metadata:
                metadata_seen = True
            else:
                selected = True
            digest, copied = _copy_git_blob(
                process,
                blob,
                staging / "files" / path,
                spec.max_bytes - total,
                verify_oid=transport is not None,
            )
            total += copied
            inventory.append(
                {"path": path, "sha256": digest, "size": copied, "git_blob_id": blob}
            )
        assert process.stdin is not None
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError("Git batch blob read failed")
    if not selected:
        raise ValueError("Git include patterns selected no files")
    if metadata_path and not metadata_seen:
        raise ValueError("pinned Git rights metadata file is missing")
    return sorted(inventory, key=lambda item: item["path"]), {"commit": commit}


_GIT_HEADERS = {
    "User-Agent": "KybernModelLab-Corpus-Forge/1",
    "Accept": "application/vnd.github+json",
}


def _github_origin(source: SourceDeclaration) -> tuple[str, str]:
    origin = urlparse(source.canonical_uri)
    parts = origin.path.strip("/").removesuffix(".git").split("/")
    if (
        origin.scheme != "https"
        or origin.netloc != "github.com"
        or origin.query
        or origin.fragment
        or len(parts) != 2
    ):
        raise ValueError("bounded Git requires an exact GitHub repository URL")
    owner, repository = parts
    if not all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in {".", ".."}
        for part in parts
    ):
        raise ValueError("bounded Git repository path is unsafe")
    return owner, repository


def _check_git_metadata(
    source: SourceDeclaration,
    spec: GitAcquisition,
    owner: str,
    repository: str,
    budget: TransportBudget,
) -> None:
    api = f"https://api.github.com/repos/{owner}/{repository}/git"
    metadata = []
    for url in (
        f"{api}/commits/{source.revision}",
        f"{api}/trees/{spec.tree_oid}?recursive=1",
    ):
        response, _ = _open_accounted(url, dict(_GIT_HEADERS), budget)
        with response:
            if urlparse(response.url).hostname != "api.github.com":
                raise ValueError("pinned Git metadata changed origin")
            try:
                metadata.append(json.loads(_read_metadata_body(response, budget)))
            except json.JSONDecodeError as error:
                raise ValueError("invalid pinned Git metadata") from error
    commit, tree = metadata
    if (
        not isinstance(commit, dict)
        or commit.get("sha", "").lower() != source.revision.lower()
        or not isinstance(commit.get("tree"), dict)
        or commit["tree"].get("sha", "").lower() != spec.tree_oid.lower()
        or not isinstance(tree, dict)
        or tree.get("sha", "").lower() != spec.tree_oid.lower()
        or tree.get("truncated") is not False
        or not isinstance(tree.get("tree"), list)
    ):
        raise ValueError("pinned Git commit/tree metadata mismatch or incomplete")
    for blob in spec.bounded_blobs:
        matches = [
            item
            for item in tree["tree"]
            if isinstance(item, dict) and item.get("path") == blob.path
        ]
        if (
            len(matches) != 1
            or matches[0].get("type") != "blob"
            or matches[0].get("mode") not in {"100644", "100755"}
            or matches[0].get("sha", "").lower() != blob.git_blob_oid.lower()
            or matches[0].get("size") != blob.max_bytes
        ):
            raise ValueError(f"pinned Git blob missing or ambiguous: {blob.path}")


def _git_blob_body(
    source: SourceDeclaration,
    blob: Any,
    url: str,
    target: Path,
    budget: TransportBudget,
    resources: _AcquisitionResources,
    cache: TransportCache | None,
    ensure_metadata: Callable[[], None],
    *,
    verify_only: bool = False,
    parts: AcquisitionParts | None = None,
) -> tuple[str, int, str]:
    """One pinned blob as (SHA-256, size, Git blob id): cached, else downloaded.

    A hit makes no request at all, not even the commit and tree metadata check:
    the cached bytes are re-hashed against the pinned Git blob id.
    """
    key = BodyKey(
        "git_blob", url, revision=source.revision, git_blob_oid=blob.git_blob_oid
    )
    if cache is not None:
        served = cache.lookup(
            key,
            None if verify_only else target,
            limit=blob.max_bytes,
            observe=resources.observe,
        )
        if served is not None:
            if parts is not None:
                parts.sync_path(target)
            return served.sha256, served.size, blob.git_blob_oid.lower()
    ensure_metadata()
    budget.assert_transfer_available(
        source.id, blob.path, blob.max_bytes, allow_verified_retry=True
    )
    transfer_index = budget.reserve_transfer(
        source.id, blob.path, blob.max_bytes, allow_verified_retry=True
    )
    resources.index = transfer_index
    resources.expanded = None  # raw Git blobs have no expansion phase
    resources.observe()
    try:
        response, _ = _open_accounted(url, dict(_GIT_HEADERS), budget)
        with response:
            if _response_length(response) != blob.max_bytes:
                raise ValueError("pinned Git blob Content-Length mismatch")
            git_hash = git_blob_hasher(blob.max_bytes)
            digest, copied = _copy_hf_body(
                response,
                target,
                blob.max_bytes,
                budget,
                transfer_index,
                resources,
                hash_chunk=git_hash.update,
            )
        git_digest = git_hash.hexdigest()
        if git_digest.lower() != blob.git_blob_oid.lower():
            raise ValueError(f"pinned Git blob SHA-1 mismatch: {blob.path}")
        if parts is not None:
            parts.sync_path(target)
        budget.finish_transfer(transfer_index, success=True)
    except OSError, TimeoutError, EOFError:
        budget.finish_transfer(transfer_index, success=False, interrupted=True)
        raise
    except Exception:
        budget.finish_transfer(transfer_index, success=False)
        raise
    except BaseException:
        budget.finish_transfer(transfer_index, success=False, interrupted=True)
        raise
    if cache is not None:
        cache.store(key, target)
    return digest, copied, git_digest


def _git_blob_digests(path: Path, size: int) -> tuple[str, str]:
    git_hash = git_blob_hasher(size)
    content_hash = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            git_hash.update(block)
            content_hash.update(block)
    return content_hash.hexdigest(), git_hash.hexdigest()


def _git_blob_url(
    source: SourceDeclaration, owner: str, repository: str, path: str
) -> str:
    return (
        f"https://raw.githubusercontent.com/{owner}/{repository}/"
        f"{source.revision}/{quote(path, safe='/')}"
    )


def _bounded_git_acquire(
    source: SourceDeclaration,
    spec: GitAcquisition,
    budget: TransportBudget,
    cache: TransportCache | None = None,
    *,
    parts: AcquisitionParts,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    owner, repository = _github_origin(source)
    checked = False

    def ensure_metadata() -> None:
        nonlocal checked
        if not checked:
            _check_git_metadata(source, spec, owner, repository, budget)
            checked = True

    staging = parts.staging
    budget.check_disk(staging, budget.projected_disk_bytes())
    files = []
    reused = 0
    for blob in spec.bounded_blobs:
        budget.remaining_seconds()
        target = staging / "files" / blob.path
        copied = parts.staged_file_size(target)
        if copied == blob.max_bytes:
            digest, git_digest = _git_blob_digests(target, copied)
        else:
            git_digest = None
        if git_digest == blob.git_blob_oid.lower():
            budget.recover_staged_transfer(source.id, blob.path, copied)
            reused += 1
        else:
            if copied is not None:
                print(
                    f"{source.id}: retrying incomplete or damaged Git blob {blob.path}",
                    file=sys.stderr,
                )
                target.unlink()
            resources = _AcquisitionResources(staging, budget)
            digest, copied, git_digest = _git_blob_body(
                source,
                blob,
                _git_blob_url(source, owner, repository, blob.path),
                target,
                budget,
                resources,
                cache,
                ensure_metadata,
                parts=parts,
            )
        files.append(
            {
                "path": blob.path,
                "sha256": digest,
                "size": copied,
                "git_blob_id": git_digest,
            }
        )
    if reused:
        print(f"{source.id}: reused {reused} verified Git blobs", file=sys.stderr)
    return files, {
        "commit": source.revision,
        "tree": spec.tree_oid,
        "transport_budget": budget.receipt(),
    }


class _PrivateHubRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a Hub bearer credential to another origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            before = urlparse(request.full_url)
            after = urlparse(newurl)
            if (before.scheme, before.netloc) != (after.scheme, after.netloc):
                redirected.remove_header("Authorization")
        return redirected


class _ManualRedirect(urllib.request.HTTPRedirectHandler):
    """Return redirects to the caller so their bodies can be charged first."""

    def http_error_302(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _response_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Length") if headers is not None else None
    if raw is None:
        return None
    if not str(raw).isdigit():
        raise ValueError("invalid HTTP Content-Length")
    return int(raw)


def _set_response_deadline(response: Any, budget: TransportBudget) -> None:
    remaining = budget.remaining_seconds()
    stream = response
    for _ in range(5):
        if isinstance(stream, io.BytesIO):
            return  # deterministic local fixture, including HTTPError bodies
        sock = getattr(stream, "_sock", None)
        if sock is not None:
            sock.settimeout(min(30, remaining))
            return
        next_stream = getattr(stream, "fp", None)
        if next_stream is None:
            next_stream = getattr(stream, "raw", None)
        if next_stream is None or next_stream is stream:
            break
        stream = next_stream
    raise ValueError("HTTP response socket cannot enforce transport deadline")


def _read_with_deadline(response: Any, size: int, budget: TransportBudget) -> bytes:
    """Bound a urllib read even when its live response hides the socket."""
    try:
        _set_response_deadline(response, budget)
    except ValueError as error:
        if str(error) != "HTTP response socket cannot enforce transport deadline":
            raise
        if threading.current_thread() is not threading.main_thread():
            raise
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] > 0:
            raise ValueError(
                "cannot replace an existing HTTP deadline timer"
            ) from error
        previous_handler = signal.getsignal(signal.SIGALRM)

        def deadline_expired(_signal: int, _frame: Any) -> None:
            raise TimeoutError("HTTP response read exceeded transport deadline")

        signal.signal(signal.SIGALRM, deadline_expired)
        try:
            signal.setitimer(signal.ITIMER_REAL, min(30, budget.remaining_seconds()))
            return response.read(size)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    return response.read(size)


def _read_metadata_body(response: Any, budget: TransportBudget) -> bytes:
    length = _response_length(response)
    if length is not None:
        budget.reserve_metadata(length)
    chunks: list[bytes] = []
    received = 0
    while True:
        if length is not None:
            allowance = length - received
            if allowance == 0:
                break
            request_size = min(65536, allowance)
        else:
            remaining = (
                budget.spec.max_metadata_body_bytes
                - budget.receipt()["metadata_charged"]
            )
            if remaining <= 0:
                raise ValueError("metadata response length is unverified at body cap")
            request_size = min(65536, remaining)
            budget.reserve_metadata(request_size)
        chunk = _read_with_deadline(response, request_size, budget)
        if not chunk:
            if length is not None:
                raise ValueError("short HTTP metadata body")
            break
        if len(chunk) > request_size:
            raise ValueError("HTTP metadata body exceeded requested read")
        received += len(chunk)
        budget.charge_metadata_actual(len(chunk))
        chunks.append(chunk)
    return b"".join(chunks)


def _open_accounted(
    url: str,
    headers: dict[str, str],
    budget: TransportBudget,
    *,
    data: bytes | None = None,
) -> tuple[Any, str]:
    opener = urllib.request.build_opener(_ManualRedirect())
    current = url
    if urlparse(current).hostname != "huggingface.co":
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() != "authorization"
        }
    for _ in range(6):
        budget.remaining_seconds()
        request = urllib.request.Request(current, headers=headers, data=data)
        try:
            response = opener.open(request, timeout=min(30, budget.remaining_seconds()))
        except HTTPError as error:
            # Even an error response consumes the attempt's metadata allowance.
            with error:
                _read_metadata_body(error, budget)
            raise
        status = getattr(response, "status", None) or getattr(response, "code", 200)
        if status not in {301, 302, 303, 307, 308}:
            if urlparse(response.url).scheme != "https":
                response.close()
                raise ValueError("HF response must use HTTPS")
            return response, current
        try:
            _read_metadata_body(response, budget)
            location = response.headers.get("Location")
        finally:
            response.close()
        if not location:
            raise ValueError("HTTP redirect lacks Location")
        next_url = urljoin(current, location)
        before, after = urlparse(current), urlparse(next_url)
        if after.scheme != "https":
            raise ValueError("HF redirect must use HTTPS")
        if (before.scheme, before.netloc) != (after.scheme, after.netloc):
            headers = {
                key: value
                for key, value in headers.items()
                if key.lower() != "authorization"
            }
            if data is not None:
                raise ValueError("pinned HF metadata POST redirected across origins")
        current = next_url
    raise ValueError("too many HF redirects")


def _hf_pinned_metadata(
    source: SourceDeclaration,
    spec: HuggingFaceAcquisition,
    shard: Any,
    headers: dict[str, str],
    budget: TransportBudget,
) -> dict[str, Any]:
    repo = source.canonical_uri.removeprefix("https://huggingface.co/datasets/")
    tree_url = (
        f"https://huggingface.co/api/datasets/{repo}/paths-info/{source.revision}"
    )
    splits_url = "https://datasets-server.huggingface.co/splits?" + urlencode(
        {"dataset": repo, "revision": source.revision}
    )
    records = []
    for url in (tree_url, splits_url):
        data = (
            urlencode({"paths": shard.path, "expand": "true"}).encode()
            if url == tree_url
            else None
        )
        response, _ = _open_accounted(url, headers.copy(), budget, data=data)
        with response:
            try:
                records.append(json.loads(_read_metadata_body(response, budget)))
            except json.JSONDecodeError as error:
                raise ValueError("invalid pinned HF metadata") from error
    tree, splits = records
    if not isinstance(tree, list) or not isinstance(splits, dict):
        raise TypeError("invalid pinned HF metadata shape")
    matches = [
        item
        for item in tree
        if isinstance(item, dict) and item.get("path") == shard.path
    ]
    if len(matches) != 1:
        raise ValueError("pinned HF shard path missing or ambiguous")
    item = matches[0]
    if item.get("type") != "file" or item.get("size") != shard.max_shard_bytes:
        raise ValueError("pinned HF shard size/type mismatch")
    lfs = item.get("lfs")
    if isinstance(lfs, dict):
        hashes = [value for key in ("sha256", "oid") if (value := lfs.get(key))]
        if (
            not hashes
            or any(
                not isinstance(value, str)
                or value.lower() != shard.expected_sha256.lower()
                for value in hashes
            )
            or lfs.get("size", shard.max_shard_bytes) != shard.max_shard_bytes
        ):
            raise ValueError("pinned HF shard metadata checksum mismatch")
    elif lfs is None:
        # Xet metadata has a CAS hash, not a content SHA-256. The pinned
        # expected SHA-256 is still checked against every downloaded byte.
        xet_hash, oid = item.get("xetHash"), item.get("oid")
        if not (
            isinstance(xet_hash, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", xet_hash)
            and isinstance(oid, str)
            and re.fullmatch(r"[0-9a-fA-F]{40}", oid)
        ):
            raise ValueError("pinned HF shard metadata checksum unavailable")
    else:
        raise ValueError("pinned HF shard metadata checksum unavailable")
    entries = splits.get("splits")
    if not isinstance(entries, list):
        raise TypeError("pinned HF config/split metadata missing")
    if not entries or any(
        not isinstance(entry, dict)
        or entry.get("dataset") != repo
        or not isinstance(entry.get("config"), str)
        or not isinstance(entry.get("split"), str)
        for entry in entries
    ):
        raise ValueError("pinned HF config/split metadata missing or ambiguous")
    # The endpoint lists all dataset configs and splits, not just our selection.
    selected = [
        entry
        for entry in entries
        if (entry["config"], entry["split"]) == (spec.config, spec.split)
    ]
    if len(selected) != 1:
        raise ValueError("pinned HF config/split metadata missing or ambiguous")
    return {"tree_url": tree_url, "splits_url": splits_url, "size": item["size"]}


def _copy_hf_body(
    response: Any,
    target: Path,
    limit: int,
    budget: TransportBudget | None,
    transfer_index: int | None,
    resources: _AcquisitionResources | None = None,
    *,
    hash_chunk: Callable[[bytes], None] | None = None,
) -> tuple[str, int]:
    length = _response_length(response)
    if length is None:
        raise ValueError("HF source requires Content-Length for strict body cap")
    if length is not None and length > limit:
        raise ValueError("HF source exceeds declared max_bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        if resources is not None:
            resources.observe()
        while True:
            remaining = limit - size
            if remaining == 0:
                if length is None or length == size:
                    break
                raise ValueError("HF source exceeds declared max_bytes")
            request_size = min(1024 * 1024, remaining)
            if length is not None:
                if length == size:
                    break
                request_size = min(request_size, length - size)
            chunk = (
                _read_with_deadline(response, request_size, budget)
                if budget is not None
                else response.read(request_size)
            )
            if not chunk:
                if length is not None and size != length:
                    raise ValueError("short HF source body")
                break
            if len(chunk) > request_size:
                raise ValueError("HF source body exceeded requested read")
            if budget is not None:
                assert transfer_index is not None
                budget.charge_transfer_actual(transfer_index, len(chunk))
            digest.update(chunk)
            if hash_chunk is not None:
                hash_chunk(chunk)
            output.write(chunk)
            if resources is not None:
                output.flush()
                resources.observe()
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), size


def _acquire_http(
    source: SourceDeclaration,
    staging: Path,
    offline: bool,
    cache: TransportCache | None = None,
    *,
    verify_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch HTTP")
    spec = source.acquisition
    assert isinstance(spec, HttpAcquisition)
    if urlparse(source.canonical_uri).scheme not in ("http", "https"):
        raise ValueError("HTTP source requires an HTTP(S) canonical URI")
    suffix = Path(urlparse(source.canonical_uri).path).suffix.lower()
    if suffix not in (".txt", ".md", ".markdown"):
        raise ValueError("HTTP source must explicitly name a text/Markdown document")
    name = "document" + suffix
    key = BodyKey("http_document", source.canonical_uri, sha256=spec.expected_sha256)
    if cache is not None:
        served = cache.lookup(
            key,
            None if verify_only else staging / "files" / name,
            limit=spec.max_bytes,
        )
        if served is not None and served.meta is not None:
            # The recorded headers describe the original retrieval.
            return [
                {"path": name, "sha256": served.sha256, "size": served.size}
            ], served.meta
    headers = {"User-Agent": "KybernModelLab-Corpus-Forge/1"}
    origin = urlparse(source.canonical_uri)
    if origin.scheme == "https" and origin.hostname == "huggingface.co":
        token = hub_auth_kwargs().get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(source.canonical_uri, headers=headers)
    try:
        with urllib.request.build_opener(_PrivateHubRedirect()).open(
            request, timeout=30
        ) as response:
            if urlparse(response.url).scheme not in ("http", "https"):
                raise ValueError("HTTP redirect has unsafe scheme")
            digest, size = _copy_stream(
                response, staging / "files" / name, spec.max_bytes
            )
            retrieval = {
                "final_url": response.url,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
    except HTTPError as error:
        if origin.scheme == "https" and origin.hostname == "huggingface.co":
            raise_for_hub_auth(error, credential_supplied="Authorization" in headers)
        raise
    if digest != spec.expected_sha256:
        raise ValueError("HTTP expected SHA-256 mismatch")
    if cache is not None:
        cache.store(key, staging / "files" / name, meta=retrieval)
    return [{"path": name, "sha256": digest, "size": size}], retrieval


def _validate_hf_rows(path: Path, spec: HuggingFaceAcquisition, available: int) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        if spec.text_field not in parquet.schema_arrow.names:
            raise ValueError(f"HF text field missing: {spec.text_field}")
        count = parquet.metadata.num_rows
        if count > available:
            raise ValueError("HF selection exceeds max_rows")
        return count
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ValueError("HF JSON requires object or array of objects")
        if len(records) > available:
            raise ValueError("HF selection exceeds max_rows")
    else:
        with path.open(encoding="utf-8") as handle:
            records = []
            for line in handle:
                if len(records) >= available:
                    raise ValueError("HF selection exceeds max_rows")
                records.append(json.loads(line))
    if any(
        not isinstance(record, dict) or not isinstance(record.get(spec.text_field), str)
        for record in records
    ):
        raise ValueError(f"HF text field missing/non-string: {spec.text_field}")
    return len(records)


def _parquet_json_value(value: Any) -> Any:
    """Encode decoded timestamps as ISO 8601, retaining their numeric UTC offset.

    Arrow units and named zones are not retained in JSON. Naive timestamps
    stay naive: never invent a timezone. Shard SHA-256 keeps
    the original typed bytes; row SHA-256 identifies this JSON representation.
    Other unsupported Arrow values still fail canonical JSON serialization.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _parquet_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_parquet_json_value(item) for item in value]
    return value


def _bounded_hf_rows(
    path: Path,
    *,
    limit: int,
    max_line_bytes: int,
    max_decompressed_bytes: int | None = None,
    budget: TransportBudget | None = None,
    resources: _AcquisitionResources | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.name.endswith(".parquet"):
        if max_decompressed_bytes is not None:
            raise ValueError("required expanded byte reading unavailable for Parquet")
        if resources is not None:
            resources.expanded = None
            resources.persist()
        import pyarrow.parquet as pq

        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=min(limit, 256)):
            for row in batch.to_pylist():
                if index >= limit:
                    return
                yield index, _parquet_json_value(row)
                index += 1
        return
    stream = gzip.open if path.name.endswith(".gz") else open
    decompressed = 0
    with stream(path, "rb") as handle:
        for index in range(limit):
            if budget is not None:
                budget.remaining_seconds()
            line = handle.readline(max_line_bytes + 1)
            if not line:
                break
            decompressed += len(line)
            if resources is not None:
                resources.expanded = decompressed
            if (
                max_decompressed_bytes is not None
                and decompressed > max_decompressed_bytes
            ):
                raise ValueError(
                    "HF decompressed stream exceeds max_decompressed_bytes"
                )
            if len(line) > max_line_bytes:
                raise ValueError("HF decompressed JSONL row exceeds max_bytes")
            try:
                yield index, json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"malformed HF JSONL row {index}") from error


def _hf_shard_body(
    source: SourceDeclaration,
    spec: HuggingFaceAcquisition,
    shard: Any,
    input_path: Path,
    budget: TransportBudget | None,
    resources: _AcquisitionResources,
    cache: TransportCache | None,
    *,
    verify_only: bool = False,
    parts: HFParts | None = None,
) -> tuple[str, int]:
    """One pinned shard body: from the cache when it verifies, else downloaded.

    A hit makes no request, reserves nothing and charges the ledger nothing.
    """
    from huggingface_hub import hf_hub_url

    repo_id = source.canonical_uri.removeprefix("https://huggingface.co/datasets/")
    auth = hub_auth_kwargs()
    url = hf_hub_url(
        repo_id=repo_id,
        filename=shard.path,
        repo_type="dataset",
        revision=source.revision,
    )
    if urlparse(url).scheme != "https" or urlparse(url).hostname != "huggingface.co":
        raise ValueError("HF shard URL must point to the Hugging Face Hub")
    headers = {"User-Agent": "KybernModelLab-Corpus-Forge/1"}
    if auth.get("token"):
        headers["Authorization"] = f"Bearer {auth['token']}"
    request = urllib.request.Request(url, headers=headers)
    key = BodyKey(
        "huggingface_shard",
        url,
        revision=source.revision,
        sha256=shard.expected_sha256,
    )
    if cache is not None:
        served = cache.lookup(
            key,
            None if verify_only else input_path,
            limit=shard.max_shard_bytes,
            observe=resources.observe,
        )
        if served is not None:
            if parts is not None:
                parts.sync_path(input_path)
            return served.sha256, served.size
    transfer_index = None
    try:
        if budget is not None:
            budget.assert_transfer_available(
                source.id,
                shard.path,
                shard.max_shard_bytes,
                allow_verified_retry=parts is not None,
            )
            projected = budget.projected_disk_bytes(source_id=budget.selected_source_id)
            budget.check_disk(resources.root, projected)
            _hf_pinned_metadata(source, spec, shard, headers, budget)
            transfer_index = budget.reserve_transfer(
                source.id,
                shard.path,
                shard.max_shard_bytes,
                allow_verified_retry=parts is not None,
            )
            resources.index = transfer_index
            resources.persist()
            response, _ = _open_accounted(url, headers.copy(), budget)
        else:
            response = urllib.request.build_opener(_PrivateHubRedirect()).open(
                request, timeout=30
            )
        with response:
            if urlparse(response.url).scheme != "https":
                raise ValueError("HF shard redirect must use HTTPS")
            digest, input_bytes = _copy_hf_body(
                response,
                input_path,
                shard.max_shard_bytes,
                budget,
                transfer_index,
                resources,
            )
    except (*hub_access_errors(), HTTPError) as error:
        if budget is not None and transfer_index is not None:
            budget.finish_transfer(transfer_index, success=False, interrupted=False)
        raise_for_hub_auth(error, credential_supplied=bool(auth.get("token")))
    except OSError, TimeoutError, EOFError:
        if budget is not None and transfer_index is not None:
            budget.finish_transfer(transfer_index, success=False, interrupted=True)
        raise
    except Exception:
        if budget is not None and transfer_index is not None:
            budget.finish_transfer(transfer_index, success=False, interrupted=False)
        raise
    except BaseException:
        if budget is not None and transfer_index is not None:
            budget.finish_transfer(transfer_index, success=False, interrupted=True)
        raise
    if digest != shard.expected_sha256.lower():
        if budget is not None and transfer_index is not None:
            budget.finish_transfer(transfer_index, success=False)
        raise ValueError(f"HF shard SHA-256 mismatch: {shard.path}")
    if parts is not None:
        parts.sync_path(input_path)
    if budget is not None and transfer_index is not None:
        budget.finish_transfer(transfer_index, success=True)
    if cache is not None:
        cache.store(key, input_path)
    return digest, input_bytes


def _acquisition_part_identity(
    source: SourceDeclaration,
    budget: TransportBudget | None,
    *,
    adapter: dict[str, str] | None = None,
) -> str:
    return canonical_digest(
        {
            "declaration": declaration_sha256(source),
            "acquisition": source.acquisition.model_dump(mode="json"),
            "adapter": _adapter(source) if adapter is None else adapter,
            "ledger": budget.binding if budget is not None else None,
        }
    )


def bounded_hf_parts(
    parent: Path,
    source: SourceDeclaration,
    budget: TransportBudget | None,
    *,
    receipt_limit: int | None = None,
) -> HFParts | None:
    """Bind resumable parts to this adapter, complete request and ledger scope."""
    if source.kind != "huggingface_dataset":
        return None
    spec = source.acquisition
    if spec.bounded_shards is None or spec.retained_subset is not None:
        return None
    return HFParts(
        parent,
        _acquisition_part_identity(source, budget),
        receipt_limit=receipt_limit,
    )


def pinned_body_parts(
    parent: Path,
    source: SourceDeclaration,
    budget: TransportBudget | None,
    *,
    adapter: dict[str, str] | None = None,
) -> PinnedBodyParts | None:
    """Bind a single pinned body and its derivation to this request."""
    spec = source.acquisition
    if isinstance(spec, WikimediaDumpAcquisition):
        name, max_bytes = (
            source_file_name(source.canonical_uri),
            spec.max_compressed_bytes,
        )
        sha1 = spec.expected_sha1.lower()
    elif isinstance(spec, (ArchiveAcquisition, MediaWikiDumpAcquisition)):
        ((name, max_bytes),) = pinned_transfers(source)
        sha1 = None
    else:
        return None
    return PinnedBodyParts(
        parent,
        _acquisition_part_identity(source, budget, adapter=adapter),
        adapter=source.kind,
        name=name,
        sha256=spec.expected_sha256.lower()
        if spec.expected_sha256 is not None
        else None,
        max_bytes=max_bytes,
        sha1=sha1,
    )


def verified_pinned_derivation(
    source: SourceDeclaration,
    parts: PinnedBodyParts,
    *,
    manifest_limit: int | None,
    previous: CheckedCheckpoint[dict[str, Any]] | None = None,
) -> CheckedCheckpoint[dict[str, Any]] | None:
    size = parts.manifest_size()
    if size is None:
        return None
    if manifest_limit is not None and size > manifest_limit:
        raise ValueError("acquisition step manifest byte cap exceeded")
    if previous is not None and previous.matches(parts.staging):
        return previous
    try:
        checked = check_checkpoint(
            parts.staging, lambda path: verify_snapshot(path, _staged=True)
        )
    except ValueError:
        return None
    assert checked is not None
    manifest = checked.value
    if manifest["declaration"] != acquisition_identity_payload(source) or manifest[
        "adapter"
    ] != _adapter(source):
        return None
    return checked


def _reclaim_published_pinned_parts(
    parent: Path,
    source: SourceDeclaration,
    budget: TransportBudget | None,
    manifest: dict[str, Any],
) -> None:
    # Use the published producer to find the request originally moved by rename.
    parts = pinned_body_parts(parent, source, budget, adapter=manifest["adapter"])
    if parts is not None:
        try:
            parts.reclaim_published()
        except (OSError, ValueError) as error:
            print(
                f"cannot reclaim acquisition staging at {parts.root}: {error}",
                file=sys.stderr,
            )


def _bounded_hf_acquire(
    source: SourceDeclaration,
    spec: HuggingFaceAcquisition,
    staging: Path,
    budget: TransportBudget | None = None,
    cache: TransportCache | None = None,
    *,
    parts: HFParts,
    planned_reuse: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert spec.bounded_shards is not None
    inventory: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    total_bytes = 0
    total_rows = 0
    reused_parts = reused_bodies = 0
    planned_reuse = planned_reuse or {}
    if spec.max_decompressed_bytes is not None and any(
        shard.path.endswith(".parquet") for shard in spec.bounded_shards
    ):
        raise ValueError("required expanded byte reading unavailable for Parquet")
    for shard in spec.bounded_shards:
        if shard.declared_config is not None and budget is None:
            raise ValueError("explicit HF shard metadata requires a transport budget")
        if budget is not None:
            budget.remaining_seconds()
        input_path = parts.body(shard)
        target = parts.output(shard)
        if symlinked(input_path):
            raise ValueError(f"unsafe HF staged body: {input_path}")
        if (record := parts.verified_part(shard, spec.max_bytes)) is not None:
            output_entry, receipt = record
            if output_entry is not None:
                inventory.append(output_entry)
                total_bytes += output_entry["size"]
            total_rows += len(receipt["selected_rows"])
            if total_bytes > spec.max_bytes or total_rows > spec.max_rows:
                raise ValueError("recovered HF selection exceeds declared bounds")
            receipts.append(receipt)
            input_path.unlink(missing_ok=True)
            reused_parts += 1
            continue
        input_bytes = parts.verified_body(shard)
        planned = planned_reuse.get(shard.path)
        if planned == "selection" or (planned == "body" and input_bytes is None):
            raise ValueError(f"HF checkpoint changed after admission: {shard.path}")
        if parts.discard_marker(shard):
            print(
                f"{source.id}: rebuilding damaged HF part {shard.path}", file=sys.stderr
            )
        if symlinked(target):
            raise ValueError(f"unsafe HF selection output: {target}")
        target.unlink(missing_ok=True)
        resources = _AcquisitionResources(
            staging, budget, journal_parent=parts.root.parent
        )
        with resources.attempt(source, shard.path):
            resources.observe()
            if input_bytes is not None:
                digest = shard.expected_sha256.lower()
                if budget is not None:
                    budget.recover_staged_transfer(source.id, shard.path, input_bytes)
                reused_bodies += 1
            else:
                input_path.unlink(missing_ok=True)
                digest, input_bytes = _hf_shard_body(
                    source,
                    spec,
                    shard,
                    input_path,
                    budget,
                    resources,
                    cache,
                    parts=parts,
                )
            output_name = target.relative_to(staging / "files").as_posix()
            target.parent.mkdir(parents=True, exist_ok=True)
            selected: list[dict[str, Any]] = []
            output_digest = hashlib.sha256()
            output_bytes = 0
            scanned = 0
            with resources.output(target) as output:
                for index, row in _bounded_hf_rows(
                    input_path,
                    limit=shard.max_scanned_rows,
                    max_line_bytes=spec.max_bytes,
                    max_decompressed_bytes=spec.max_decompressed_bytes,
                    budget=budget,
                    resources=resources,
                ):
                    scanned += 1
                    if not isinstance(row, dict) or spec.text_field not in row:
                        raise ValueError(
                            f"HF text field missing at {shard.path} row {index}"
                        )
                    if not isinstance(row[spec.text_field], str):
                        raise TypeError(
                            f"HF text field non-string at {shard.path} row {index}"
                        )
                    if "_klab_source" in row:
                        raise ValueError(
                            "HF source row collides with reserved provenance field"
                        )
                    try:
                        row_digest = canonical_digest(row)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"HF row is not canonical JSON: {shard.path} row {index}"
                        ) from error
                    row_hash = hashlib.sha256(
                        canonical_json([source.revision, shard.path, index])
                    ).hexdigest()
                    if (
                        int(row_hash, 16) % shard.hash_modulus
                        not in shard.hash_remainders
                    ):
                        continue
                    if total_rows >= spec.max_rows:
                        raise ValueError("HF bounded selection exceeds max_rows")
                    envelope: dict[str, Any] = {
                        "source_shard_path": shard.path,
                        "source_shard_sha256": digest,
                        "dataset_revision": source.revision,
                        "source_row_index": index,
                        "source_row_sha256": row_digest,
                    }
                    for key in (
                        "url",
                        "id",
                        "dump",
                        "score",
                        "language",
                        "language_score",
                        "token_count",
                        "blob_id",
                        "repo_name",
                        "path",
                        "detected_licenses",
                    ):
                        value = row.get(key)
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            if key in row:
                                envelope[key] = value
                        elif key == "detected_licenses":
                            envelope[key] = canonical_json(value).decode("utf-8")
                    line = canonical_json({**row, "_klab_source": envelope}) + b"\n"
                    if len(line) > spec.max_bytes - total_bytes - output_bytes:
                        raise ValueError("HF bounded selection exceeds max_bytes")
                    output.write(line)
                    output_digest.update(line)
                    output_bytes += len(line)
                    total_rows += 1
                    selected.append(
                        {
                            "source_row_index": index,
                            "source_row_sha256": row_digest,
                            "selection_hash": row_hash,
                        }
                    )
                output.flush()
                os.fsync(output.fileno())
            output_entry = None
            if not selected:
                target.unlink()
            else:
                total_bytes += output_bytes
                output_entry = {
                    "path": output_name,
                    "sha256": output_digest.hexdigest(),
                    "size": output_bytes,
                }
                inventory.append(output_entry)
            receipt = {
                "source_shard_path": shard.path,
                "source_shard_sha256": digest,
                "source_shard_bytes": input_bytes,
                "resources": resources.receipt(),
                "scanned_rows": scanned,
                "selected_rows": selected,
                "output_path": output_name if selected else None,
            }
            parts.commit(shard, output_entry, receipt)
            input_path.unlink()
            # Forgetting the body must survive a crash before snapshot publication.
            sync_directory(input_path.parent)
            receipts.append(receipt)
    if reused_parts or reused_bodies:
        print(
            f"{source.id}: reused {reused_parts} selected HF parts and "
            f"{reused_bodies} verified HF bodies",
            file=sys.stderr,
        )
    if not inventory:
        raise ValueError("HF bounded selection produced no rows")
    retrieval = {
        "config": spec.config,
        "split": spec.split,
        "text_field": spec.text_field,
        "max_rows": spec.max_rows,
        "rows": total_rows,
        "dataset_revision": source.revision,
        "sampling": "sha256-revision-path-zero-index-modulus-v1",
        "shards": receipts,
    }
    if any(shard.path.endswith(".parquet") for shard in spec.bounded_shards):
        retrieval["row_encoding"] = "json-with-iso8601-timestamps-v1"
    if budget is not None:
        retrieval["transport_budget"] = budget.receipt()
    return inventory, retrieval


def _acquire_hf(
    source: SourceDeclaration,
    staging: Path,
    cache_root: Path,
    offline: bool,
    budget: TransportBudget | None = None,
    cache: TransportCache | None = None,
    *,
    parts: HFParts | None = None,
    planned_reuse: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Hugging Face")
    from huggingface_hub import snapshot_download

    spec = source.acquisition
    if spec.retained_subset is not None:
        from klab.corpus.retained_subset import derive_subset

        return derive_subset(source, staging=staging, scratch_directory=staging)
    if spec.bounded_shards is not None:
        assert parts is not None
        return _bounded_hf_acquire(
            source,
            spec,
            staging,
            budget,
            cache,
            parts=parts,
            planned_reuse=planned_reuse,
        )
    assert isinstance(spec, HuggingFaceAcquisition)
    auth = hub_auth_kwargs()
    try:
        location = Path(
            snapshot_download(
                repo_id=source.canonical_uri.removeprefix(
                    "https://huggingface.co/datasets/"
                ),
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=list(spec.include),
                cache_dir=str(cache_root),
                **auth,
            )
        )
    except hub_access_errors() as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))
    if location.name.lower() != source.revision.lower():
        raise ValueError("HF snapshot revision does not match pinned commit")
    inventory = []
    total = rows = 0
    for path in sorted(location.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(location).as_posix()
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in spec.include):
            continue
        safe_name(relative)
        if path.is_symlink():
            # hub cache often provides symlinks to immutable blobs; resolve within the cache.
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(cache_root.resolve()):
                raise ValueError("HF snapshot symlink escapes cache")
            path = resolved
        _regular(path)
        parts = relative.split("/")
        if spec.config not in parts and not Path(relative).name.startswith(
            spec.config + "-"
        ):
            raise ValueError(
                f"HF file is not unambiguously in config {spec.config}: {relative}"
            )
        if spec.split not in parts and not any(
            part.startswith((spec.split + "-", spec.split + ".")) for part in parts
        ):
            raise ValueError(
                f"HF file is not unambiguously in split {spec.split}: {relative}"
            )
        if path.suffix.lower() not in (".jsonl", ".json", ".parquet"):
            raise ValueError(f"unsupported HF file format: {relative}")
        if path.stat().st_size > spec.max_bytes - total:
            raise ValueError("HF selection exceeds max_bytes")
        rows += _validate_hf_rows(path, spec, spec.max_rows - rows)
        with path.open("rb") as stream:
            digest, size = _copy_stream(
                stream, staging / "files" / relative, spec.max_bytes - total
            )
        total += size
        inventory.append({"path": relative, "sha256": digest, "size": size})
    if not inventory:
        raise ValueError("HF include patterns selected no files")
    return inventory, {
        "config": spec.config,
        "split": spec.split,
        "text_field": spec.text_field,
        "max_rows": spec.max_rows,
        "rows": rows,
    }


_WIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
_WIKI_BLOCKED = re.compile(
    r"(?i)\b(?:confidential|copyright violation|copyvio|non-free|"
    r"fair use|do not train|noai|imported from|transwiki)\b"
)
_WIKI_PRIVATE_TITLE = re.compile(r"(?i)\b(?:private secrets?|confidential data)\b")


class _BoundedXML:
    """Bound decompressed XML and reject document types before the XML parser sees them."""

    def __init__(
        self, stream: Any, limit: int, resources: _AcquisitionResources | None = None
    ):
        self.stream = stream
        self.limit = limit
        self.size = 0
        self.previous = b""
        self.resources = resources
        self.persisted_size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(min(65536, size if size >= 0 else 65536))
        self.size += len(chunk)
        if self.resources is not None:
            self.resources.expanded = self.size
            if self.size - self.persisted_size >= 1024 * 1024:
                self.resources.persist()
                self.persisted_size = self.size
        if self.size > self.limit:
            raise ValueError("Wikimedia decompressed XML exceeds limit")
        combined = (self.previous + chunk).upper()
        if b"<!DOCTYPE" in combined or b"<!ENTITY" in combined:
            raise ValueError("Wikimedia XML DTD/entities are forbidden")
        self.previous = combined[-8:]
        return chunk


def _wiki_text(wikitext: str) -> str:
    # Remove entire templates, references, tables, files and categories; never
    # expand templates or fetch links. Keep literal code and math element bodies.
    text = re.sub(r"(?is)<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/>", "", wikitext)
    text = re.sub(r"(?s)\{\|.*?\|\}", "", text)
    while "{{" in text:
        stripped = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if stripped == text:
            return ""
        text = stripped
    text = re.sub(r"(?i)\[\[(?:file|image|category):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[(?:[^]|]*\|)?([^]|]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]", lambda m: m[1] or "", text)
    text = re.sub(r"(?is)<!--.*?-->", "", text)
    text = re.sub(r"(?i)</?(?!math\b|code\b|pre\b)[a-z][^>]*>", "", text)
    text = re.sub(r"(?i)</?(?:math|code|pre)\b[^>]*>", "\n", text)
    text = re.sub(r"(?m)^={2,}\s*(.*?)\s*={2,}\s*$", r"\1", text)
    text = html.unescape(text.replace("'''", "").replace("''", ""))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _wiki_row(
    page: ET.Element, source: SourceDeclaration, ns: str = _WIKI_NS
) -> dict[str, Any] | None:
    spec = source.acquisition
    generic = isinstance(spec, MediaWikiDumpAcquisition)
    namespaces = {str(item) for item in spec.namespaces} if generic else {"0"}
    title = page.findtext(ns + "title")
    page_id = page.findtext(ns + "id")
    if (
        page.findtext(ns + "ns") not in namespaces
        or page.find(ns + "redirect") is not None
        or not title
        or not page_id
        or _WIKI_BLOCKED.search(title)
        or _WIKI_PRIVATE_TITLE.search(title)
    ):
        return None
    revisions = page.findall(ns + "revision")
    if not revisions:
        return None
    revision = max(
        revisions,
        key=lambda item: (
            item.findtext(ns + "timestamp") or "",
            int(item.findtext(ns + "id") or 0),
        ),
    )
    content = revision.find(ns + "text")
    if content is None or content.get("deleted") is not None or not content.text:
        return None
    raw = content.text
    if _WIKI_BLOCKED.search(raw) or re.search(
        r"(?i)\[\[(?:category|file|image):[^\]]*(?:import|non-free|copyright)", raw
    ):
        return None
    text = _wiki_text(raw)
    english = not generic or spec.language == "en"
    if not text or not (
        re.search(r"[a-zA-Z]", text) if english else any(c.isalpha() for c in text)
    ):
        return None
    if english and not text.isascii():
        # Skip predominantly non-Latin prose; retain mathematics/code with English context.
        balance = sum(1 if char.isascii() else -1 for char in text if char.isalpha())
        if balance < 0:
            return None
    revision_id = revision.findtext(ns + "id")
    timestamp = revision.findtext(ns + "timestamp")
    if not revision_id or not timestamp:
        return None
    if generic:
        page_uri = f"{spec.site_url}?curid={page_id}"
    else:
        project = urlparse(source.canonical_uri).path.split("/")[1]
        if project not in {"enwikibooks", "enwiki"}:
            raise ValueError("unsupported Wikimedia project for page attribution")
        site = "wikipedia" if project == "enwiki" else "wikibooks"
        page_uri = f"https://en.{site}.org/?curid={page_id}"
    return {
        "text": text,
        "_klab_source": {
            "source_uri": source.canonical_uri,
            "dump_revision": source.revision,
            "page_title": title,
            "page_id": page_id,
            "revision_id": revision_id,
            "revision_timestamp": timestamp,
            "page_uri": page_uri,
        },
    }


_MEDIAWIKI_ROOT = re.compile(r"\{http://www\.mediawiki\.org/xml/export-0\.\d+/\}")


def _wiki_jsonl(
    source: SourceDeclaration,
    decompressed: Any,
    output_path: Path | BinaryIO,
    resources: _AcquisitionResources,
    *,
    limits: tuple[int, int, int, int] | None = None,
    boundaries: list[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, str]], int, int, str]:
    """Stream pages from a decompressed XML export into bounded JSONL rows."""
    spec = source.acquisition
    expanded_limit, scan_limit, row_limit, output_limit = limits or (
        spec.max_decompressed_bytes,
        spec.max_scanned_pages,
        spec.max_selected_pages,
        spec.max_emitted_bytes,
    )
    generic = isinstance(spec, MediaWikiDumpAcquisition)
    emitted = scanned = 0
    digest = hashlib.sha256()
    rows: list[dict[str, str]] = []
    into_file = isinstance(output_path, Path)
    with (
        resources.output(output_path) if into_file else nullcontext(output_path)
    ) as output:
        guarded = _BoundedXML(decompressed, expanded_limit, resources)
        context = ET.iterparse(guarded, events=("start", "end"))
        root: ET.Element | None = None
        ns = _WIKI_NS
        site_verified = False
        for event, element in context:
            if root is None:
                root = element
                if generic:
                    match = _MEDIAWIKI_ROOT.match(root.tag)
                    if match is None or root.tag != match[0] + "mediawiki":
                        raise ValueError("unsupported MediaWiki XML export namespace")
                    ns = match[0]
                elif root.tag != _WIKI_NS + "mediawiki":
                    raise ValueError("unsupported Wikimedia XML namespace")
            if event == "end" and element.tag == ns + "siteinfo":
                language = element.findtext(ns + "lang") or root.get(
                    "{http://www.w3.org/XML/1998/namespace}lang"
                )
                if generic:
                    expected = (spec.language, spec.dbname)
                else:
                    expected = ("en", urlparse(source.canonical_uri).path.split("/")[1])
                if (language, element.findtext(ns + "dbname")) != expected:
                    raise ValueError(
                        "Wikimedia dump site identity or language mismatch"
                    )
                site_verified = True
            if event != "end" or element.tag != ns + "page":
                continue
            if not site_verified:
                raise ValueError("Wikimedia dump lacks English siteinfo")
            if scanned >= scan_limit or len(rows) >= row_limit:
                break
            scanned += 1
            row = _wiki_row(element, source, ns)
            if row is not None:
                line = canonical_json(row) + b"\n"
                if len(line) > output_limit - emitted:
                    raise ValueError("Wikimedia emitted JSONL exceeds limit")
                output.write(line)
                digest.update(line)
                emitted += len(line)
                if boundaries is not None:
                    boundaries.append((scanned, emitted))
                rows.append(
                    {
                        "page_id": row["_klab_source"]["page_id"],
                        "revision_id": row["_klab_source"]["revision_id"],
                        "row_sha256": hashlib.sha256(line).hexdigest(),
                    }
                )
            root.clear()
        if into_file:
            output.flush()
            os.fsync(output.fileno())
        resources.expanded = guarded.size
    if not rows and limits is None:
        raise ValueError("Wikimedia bounded selection produced no pages")
    return rows, scanned, emitted, digest.hexdigest()


def _wikimedia_body(
    source: SourceDeclaration,
    spec: WikimediaDumpAcquisition,
    compressed: Path,
    resources: _AcquisitionResources,
    cache: TransportCache | None,
    *,
    verify_only: bool = False,
    parts: PinnedBodyParts | None = None,
) -> CheckedCheckpoint[PinnedBody]:
    """The compressed dump: from the cache when it verifies, else downloaded."""
    key = BodyKey(
        "wikimedia_dump",
        source.canonical_uri,
        sha256=spec.expected_sha256 or None,
        sha1=spec.expected_sha1,
    )
    cache_target = parts.cache_download if parts is not None else compressed
    served = (
        cache.lookup(
            key,
            None if verify_only else cache_target,
            limit=spec.max_compressed_bytes,
            observe=resources.observe,
        )
        if cache is not None
        else None
    )
    body = (
        CheckedCheckpoint(
            cache_target,
            PinnedBody(served.sha256, served.size, spec.expected_sha1.lower()),
            served.copy_fingerprint,
        )
        if served is not None
        else _download_wikimedia(source, spec, compressed, resources)
    )
    if parts is not None:
        body = parts.promote_body(body)
    if cache is not None and served is None:
        cache.store(key, parts.body if parts is not None else compressed)
    return body


def _download_wikimedia(
    source: SourceDeclaration,
    spec: WikimediaDumpAcquisition,
    compressed: Path,
    resources: _AcquisitionResources,
) -> CheckedCheckpoint[PinnedBody]:
    name = source_file_name(source.canonical_uri)
    opener = urllib.request.build_opener()
    headers = {
        "User-Agent": (
            "KybernModelLab-Corpus-Forge/1 (+https://github.com/supernovae/model-lab)"
        )
    }
    with opener.open(
        urllib.request.Request(spec.checksum_uri, headers=headers), timeout=30
    ) as response:
        if response.url != spec.checksum_uri:
            raise ValueError("Wikimedia checksum redirect differs from pinned URI")
        checksum = response.read(4_194_305)
    if len(checksum) > 4_194_304:
        raise ValueError("Wikimedia checksum manifest exceeds limit")
    matches = [
        line.split()
        for line in checksum.decode("ascii").splitlines()
        if line.split()[1:] == [name]
    ]
    if len(matches) != 1 or matches[0][0].lower() != spec.expected_sha1.lower():
        raise ValueError("Wikimedia official SHA-1 manifest mismatch")
    source_sha1 = hashlib.sha1()
    fingerprint = None

    def record_copy(descriptor: int) -> None:
        nonlocal fingerprint
        fingerprint = artifact_fingerprint(descriptor)

    with opener.open(
        urllib.request.Request(source.canonical_uri, headers=headers), timeout=60
    ) as response:
        if response.url != source.canonical_uri:
            raise ValueError("Wikimedia dump redirect differs from pinned URI")
        digest, compressed_bytes = _copy_stream(
            response,
            compressed,
            spec.max_compressed_bytes,
            resources,
            hash_chunk=source_sha1.update,
            on_written=record_copy,
        )
    resources.observe()
    if spec.expected_sha256 and digest != spec.expected_sha256.lower():
        compressed.unlink()
        raise ValueError("Wikimedia compressed SHA-256 mismatch")
    if source_sha1.hexdigest() != spec.expected_sha1.lower():
        compressed.unlink()
        raise ValueError("Wikimedia compressed SHA-1 mismatch")
    return CheckedCheckpoint(
        compressed,
        PinnedBody(digest, compressed_bytes, source_sha1.hexdigest()),
        fingerprint,
    )


@dataclass(frozen=True)
class _WikimediaCandidates:
    payload: bytes
    rows: list[dict[str, str]]
    boundaries: list[tuple[int, int]]
    scanned: int

    def write_prefix(
        self, output: BinaryIO, limits: tuple[int, int, int]
    ) -> tuple[list[dict[str, str]], int, int]:
        scan_limit, row_limit, byte_limit = limits
        count = min(
            row_limit, bisect_right(self.boundaries, (scan_limit, len(self.payload)))
        )
        scanned = (
            self.boundaries[count - 1][0]
            if count == row_limit
            else min(scan_limit, self.scanned)
        )
        used = self.boundaries[count - 1][1] if count else 0
        if used > byte_limit:
            raise ValueError("Wikimedia emitted JSONL exceeds limit")
        output.write(memoryview(self.payload)[:used])
        return self.rows[:count], scanned, used


def _wikimedia_member_xml(
    member: WikimediaRange, decoded: bytearray, context: tuple[bytes, bytes]
) -> tuple[io.BytesIO, int]:
    header, footer = context
    xml = io.BytesIO()
    xml.write(header)
    xml.write(memoryview(decoded)[member.page_start : member.page_end])
    xml.write(footer)
    size = xml.tell()
    xml.seek(0)
    return xml, size


def _prepare_wikimedia_member(
    member: WikimediaRange,
    *,
    source: SourceDeclaration,
    dump: Path,
    context: tuple[bytes, bytes],
    parse_limit: int,
) -> tuple[WikimediaRange, bytearray | _WikimediaCandidates | None, int]:
    from klab.corpus.wikimedia_ranges import read_wikimedia_member

    spec = source.acquisition
    assert isinstance(spec, WikimediaDumpAcquisition) and spec.multistream_index
    decoded = read_wikimedia_member(
        dump, member, limit=spec.multistream_index.max_member_expanded_bytes
    )
    expanded = len(decoded)
    if not parse_limit:
        return member, decoded, expanded
    xml, size = _wikimedia_member_xml(member, decoded, context)
    del decoded
    candidates = io.BytesIO()
    boundaries: list[tuple[int, int]] = []
    try:
        rows, scanned, _, _ = _wiki_jsonl(
            source,
            xml,
            candidates,
            _AcquisitionResources(dump.parent),
            limits=(
                size,
                min(member.pages, spec.max_scanned_pages),
                spec.max_selected_pages,
                parse_limit,
            ),
            boundaries=boundaries,
        )
    except OSError, EOFError, ValueError, ET.ParseError:
        # This parse may have reached pages outside the eventual selection.
        # Let the ordered consumer retry with the exact remaining limits.
        return member, None, expanded
    return (
        member,
        _WikimediaCandidates(candidates.getvalue(), rows, boundaries, scanned),
        expanded,
    )


def _extract_indexed_wikimedia(
    source: SourceDeclaration,
    parts: PinnedBodyParts,
    body: CheckedCheckpoint[PinnedBody],
    output_path: Path,
    resources: _AcquisitionResources,
    cache: TransportCache | None,
    prefetch_bytes: int,
    parse_bytes: int,
    checkpoint_members: int,
) -> tuple[list[dict[str, str]], int, int, str, dict[str, Any]]:
    from klab.corpus import wikimedia_ranges

    spec = source.acquisition
    assert (
        isinstance(spec, WikimediaDumpAcquisition)
        and spec.multistream_index is not None
    )
    index = spec.multistream_index
    name = source_file_name(index.uri)
    index_path = parts.body.with_name(name)
    if parts.verified_size(index_path, index.expected_sha256, index.max_bytes) is None:
        index_path.unlink(missing_ok=True)
        _fetch_pinned(
            source,
            index.uri,
            name,
            index.expected_sha256,
            index.max_bytes,
            index_path,
            None,
            resources,
            cache,
            parts=parts,
        )
    store = StageStore(parts.root / "ranges")
    inputs = {
        "request": parts.identity,
        "dump_sha256": body.value.sha256,
        "index_sha256": index.expected_sha256,
        "planner_sha256": sha256_file(Path(wikimedia_ranges.__file__)),
    }
    plan_stage, member_stage = "wikimedia-plan", "wikimedia-member"
    store.prune(plan_stage, keep=1)
    store.prune(member_stage, keep=index.max_members)

    def record_plan_bytes(count: int) -> None:
        resources.expanded += count

    request_digest = canonical_digest(inputs)

    def account(result) -> None:
        resources.expanded += result[2]

    rows: list[dict[str, str]] = []
    scanned = emitted = visited = consumed_until = 0
    context: tuple[bytes, bytes] | None = None
    digest = hashlib.sha256()

    def select_group(
        ranges: Sequence[WikimediaRange], root_offset: int, header_bytes: int
    ) -> None:
        nonlocal context, scanned, emitted, visited, consumed_until, digest
        if scanned >= spec.max_scanned_pages or len(rows) >= spec.max_selected_pages:
            return
        members = []
        planned_pages = scanned
        for member in ranges:
            if planned_pages >= spec.max_scanned_pages:
                break
            if member.pages:
                members.append(member)
                planned_pages += member.pages
        if not members:
            return
        if context is None:
            context = wikimedia_ranges.wikimedia_xml_context(
                parts.body,
                root_offset=root_offset,
                header_bytes=header_bytes,
                limit=index.max_header_bytes,
            )
            resources.expanded += len(context[0])
        largest = max(
            min(member.expanded_bytes, index.max_member_expanded_bytes)
            for member in members
        )
        slots = min(len(members), worker_count(), 1 + prefetch_bytes // largest)
        parse_limit = (
            min(spec.max_emitted_bytes, parse_bytes // slots) if slots > 1 else 0
        )
        groups = []
        for group in batched(members, checkpoint_members):
            key = {
                "plan_inputs_sha256": request_digest,
                "members": [asdict(member) for member in group],
            }
            path = store.directory(member_stage, store.key(key))
            groups.append((group, key, path if path.is_dir() else None))
        jobs = ShardPlan(
            _WIKIMEDIA_PARSE_STAGE if parse_limit else _WIKIMEDIA_DECODE_STAGE,
            [
                (member,)
                for group, _, path in groups
                if path is None
                for member in group
            ],
            sum(member.end - member.start for member in members),
            threaded=not parse_limit,
            operator_cap=slots,
            min_parallel_bytes=0,
            # Process mode also reserves interpreter, parsed text and serialization buffers.
            worker_memory_bytes=(
                (largest + len(context[0])) * 8 + parse_limit * 3 + 128 * 1024**2
            )
            if parse_limit
            else (largest * 4 + 16 * 1024**2),
        )
        prepare_member = partial(
            _prepare_wikimedia_member,
            source=source,
            dump=parts.body,
            context=context,
            parse_limit=parse_limit,
        )
        prepare_serial = partial(prepare_member, parse_limit=0)

        def append_verified(key: dict[str, Any]):
            next_digest = digest.copy()
            output_error = None

            def append_chunk(chunk: bytes) -> None:
                nonlocal output_error
                try:
                    output.write(chunk)
                except OSError as error:
                    output_error = error
                    raise
                next_digest.update(chunk)

            receipt = store.verified(
                member_stage, key, hash_chunks={"rows.jsonl": append_chunk}
            )
            if output_error is not None:
                raise output_error
            if receipt is not None:
                return next_digest
            # The unverified copy never leaves the staged output.
            output.seek(emitted)
            output.truncate()
            return None

        with closing(
            run_sharded(jobs, prepare_member, ordered=True, on_discarded=account)
        ) as results:
            for group, key, path in groups:
                limits = (
                    spec.max_scanned_pages - scanned,
                    spec.max_selected_pages - len(rows),
                    spec.max_emitted_bytes - emitted,
                )

                def extract(
                    staging: Path,
                    _result: dict[str, Any],
                    *,
                    group=group,
                    path=path,
                    limits=limits,
                ) -> None:
                    group_rows: list[dict[str, str]] = []
                    group_scanned = group_emitted = group_members = 0
                    with (staging / "rows.jsonl").open("wb") as member_file:
                        for expected in group:
                            member, decoded, expanded = (
                                next(results)
                                if path is None
                                else prepare_serial(expected)
                            )
                            if member != expected:
                                raise ValueError(
                                    "Wikimedia decode order differs from checkpoint group"
                                )
                            resources.expanded += expanded
                            remaining = (
                                limits[0] - group_scanned,
                                limits[1] - len(group_rows),
                                limits[2] - group_emitted,
                            )
                            if isinstance(decoded, _WikimediaCandidates):
                                selected, page_count, used = decoded.write_prefix(
                                    member_file, remaining
                                )
                            else:
                                if decoded is None:
                                    decoded = wikimedia_ranges.read_wikimedia_member(
                                        parts.body,
                                        member,
                                        limit=index.max_member_expanded_bytes,
                                    )
                                    resources.expanded += len(decoded)
                                xml, size = _wikimedia_member_xml(
                                    member, decoded, context
                                )
                                selected, page_count, used, _ = _wiki_jsonl(
                                    source,
                                    xml,
                                    member_file,
                                    _AcquisitionResources(staging),
                                    limits=(size, *remaining),
                                )
                                del xml
                            del decoded
                            group_rows.extend(selected)
                            group_scanned += page_count
                            group_emitted += used
                            group_members += 1
                            if (
                                group_scanned == limits[0]
                                or len(group_rows) == limits[1]
                            ):
                                break
                    if not body.matches(parts.body):
                        raise ValueError("Wikimedia dump changed during extraction")
                    write_canonical_json(
                        staging / "member.json",
                        {
                            "rows": group_rows,
                            "scanned": group_scanned,
                            "emitted": group_emitted,
                            "members": group_members,
                            "limits": limits,
                        },
                    )

                next_digest = append_verified(key) if path is not None else None
                if next_digest is None:
                    completed, _ = run_stage(store, member_stage, key, extract)
                    next_digest = digest.copy()
                    with (completed / "rows.jsonl").open("rb") as stream:
                        while chunk := stream.read(1024**2):
                            output.write(chunk)
                            next_digest.update(chunk)
                else:
                    completed = path
                selection = json.loads((completed / "member.json").read_bytes())
                group_members = selection["members"]
                if (
                    selection["limits"] != list(limits)
                    or not 0 < group_members <= len(group)
                    or not 0
                    < selection["scanned"]
                    <= min(
                        limits[0], sum(member.pages for member in group[:group_members])
                    )
                    or len(selection["rows"]) > min(selection["scanned"], limits[1])
                    or not 0 <= selection["emitted"] <= limits[2]
                    or output.tell() - emitted != selection["emitted"]
                ):
                    raise ValueError(
                        "Wikimedia member selection differs from its bounds"
                    )
                digest = next_digest
                rows.extend(selection["rows"])
                scanned += selection["scanned"]
                emitted += selection["emitted"]
                visited += group_members
                if (
                    scanned >= spec.max_scanned_pages
                    or len(rows) >= spec.max_selected_pages
                ):
                    break

        consumed_until = ranges[-1].end

    def prepare(staging: Path, _result: dict[str, Any]) -> None:
        plan = wikimedia_ranges.plan_wikimedia_ranges(
            source,
            parts.body,
            index_path,
            index_sha256=index.expected_sha256,
            max_index_bytes=index.max_bytes,
            max_index_expanded_bytes=index.max_expanded_bytes,
            max_members=index.max_members,
            checkpoint_store=store,
            on_expanded=record_plan_bytes,
            verified_dump=body,
            on_verified_group=select_group,
        )
        write_canonical_json(staging / "plan.json", asdict(plan))

    with resources.output(output_path) as output:
        plan_directory, _ = run_stage(store, plan_stage, inputs, prepare)
        plan_record = json.loads((plan_directory / "plan.json").read_bytes())
        plan = wikimedia_ranges.WikimediaRangePlan.from_record(plan_record)
        batches = (
            batched(plan.ranges, wikimedia_ranges.PLAN_GROUP_MEMBERS)
            if checkpoint_members > 1
            else (plan.ranges,)
        )
        for batch in batches:
            select_group(
                [member for member in batch if member.start >= consumed_until],
                plan.root_offset,
                plan.header_bytes,
            )
        output.flush()
        os.fsync(output.fileno())
    if not rows:
        raise ValueError("Wikimedia bounded selection produced no pages")
    return (
        rows,
        scanned,
        emitted,
        digest.hexdigest(),
        {
            "index_sha256": index.expected_sha256,
            "plan_sha256": canonical_digest(plan_record),
            "selected_members": visited,
        },
    )


def _acquire_wikimedia(
    source: SourceDeclaration,
    staging: Path,
    parts: PinnedBodyParts,
    body: CheckedCheckpoint[PinnedBody],
    resources: _AcquisitionResources,
    cache: TransportCache | None,
    prefetch_bytes: int,
    parse_bytes: int,
    checkpoint_members: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = source.acquisition
    assert isinstance(spec, WikimediaDumpAcquisition)
    if not body.matches(parts.body):
        raise ValueError("Wikimedia dump changed after pin verification")
    output_name = parts.name + ".sample.jsonl"
    output_path = staging / "files" / output_name
    if spec.multistream_index is None:
        with bz2.open(parts.body, "rb") as decompressed:
            rows, scanned, emitted, digest = _wiki_jsonl(
                source, decompressed, output_path, resources
            )
        evidence = {"resources": resources.receipt()}
    else:
        rows, scanned, emitted, digest, indexed = _extract_indexed_wikimedia(
            source,
            parts,
            body,
            output_path,
            resources,
            cache,
            prefetch_bytes,
            parse_bytes,
            checkpoint_members,
        )
        evidence = {"multistream_index": indexed}
    if not body.matches(parts.body):
        raise ValueError("Wikimedia dump changed during extraction")
    return [
        {
            "path": output_name,
            "sha256": digest,
            "size": emitted,
        }
    ], {
        "source_uri": source.canonical_uri,
        "checksum_uri": spec.checksum_uri,
        "source_sha1": spec.expected_sha1.lower(),
        "source_sha256": body.value.sha256,
        "compressed_bytes": body.value.size,
        **evidence,
        "scanned_pages": scanned,
        "selected_pages": rows,
        "output_path": output_name,
    }


_USER_AGENT = {"User-Agent": "KybernModelLab-Corpus-Forge/1"}


def _fetch_pinned(
    source: SourceDeclaration,
    url: str,
    key_name: str,
    expected_sha256: str,
    limit: int,
    target: Path,
    budget: TransportBudget | None,
    resources: _AcquisitionResources,
    cache: TransportCache | None = None,
    *,
    verify_only: bool = False,
    parts: AcquisitionParts | None = None,
) -> int:
    """One content-pinned body: from the cache when it verifies, else downloaded.

    A download charges the ledger when one is declared; a hit charges nothing.
    """
    key = BodyKey(source.kind, url, sha256=expected_sha256)
    if cache is not None:
        served = cache.lookup(
            key,
            None if verify_only else target,
            limit=limit,
            observe=resources.observe,
        )
        if served is not None:
            if parts is not None:
                parts.sync_path(target)
            return served.size
    size = _download_pinned(
        source,
        url,
        key_name,
        expected_sha256,
        limit,
        target,
        budget,
        resources,
        parts=parts,
    )
    if cache is not None:
        cache.store(key, target)
    return size


def _download_pinned(
    source: SourceDeclaration,
    url: str,
    key: str,
    expected_sha256: str,
    limit: int,
    target: Path,
    budget: TransportBudget | None,
    resources: _AcquisitionResources,
    *,
    parts: AcquisitionParts | None = None,
) -> int:
    if budget is None:
        request = urllib.request.Request(url, headers=_USER_AGENT)
        with urllib.request.build_opener().open(request, timeout=60) as response:
            if urlparse(response.url).scheme not in ("http", "https"):
                raise ValueError("pinned download redirect has unsafe scheme")
            digest, size = _copy_stream(response, target, limit, resources)
        if digest != expected_sha256:
            raise ValueError(f"pinned download SHA-256 mismatch: {key}")
        if parts is not None:
            parts.sync_path(target)
        return size
    budget.assert_transfer_available(source.id, key, limit, allow_verified_retry=True)
    index = budget.reserve_transfer(source.id, key, limit, allow_verified_retry=True)
    resources.index = index
    try:
        response, _ = _open_accounted(url, dict(_USER_AGENT), budget)
        with response:
            digest, size = _copy_hf_body(
                response, target, limit, budget, index, resources
            )
        if digest != expected_sha256:
            raise ValueError(f"pinned download SHA-256 mismatch: {key}")
        if parts is not None:
            parts.sync_path(target)
        budget.finish_transfer(index, success=True)
    except OSError, TimeoutError, EOFError:
        budget.finish_transfer(index, success=False, interrupted=True)
        raise
    except Exception:
        budget.finish_transfer(index, success=False)
        raise
    except BaseException:
        budget.finish_transfer(index, success=False, interrupted=True)
        raise
    return size


def _resume_pinned_body(
    source: SourceDeclaration,
    parts: PinnedBodyParts,
    budget: TransportBudget | None,
    resources: _AcquisitionResources,
    cache: TransportCache | None,
    *,
    admit_body: bool,
    previous: CheckedCheckpoint[PinnedBody] | None,
) -> CheckedCheckpoint[PinnedBody]:
    checked = parts.checked_body(previous)
    if admit_body and checked is None:
        raise ValueError(f"pinned body checkpoint changed after admission: {source.id}")
    if checked is None:
        parts.body.unlink(missing_ok=True)
    # The download candidate is written only after the official checksum manifest passes.
    if (
        checked is None
        and isinstance(source.acquisition, WikimediaDumpAcquisition)
        and (recovered := parts.recover_download()) is not None
    ):
        resources.observe()
        return recovered
    resources.observe()
    if budget is not None:
        budget.remaining_seconds()
        expected = budget.projected_disk_bytes(source_id=source.id)
        budget.check_disk(
            parts.root, expected - (parts.max_bytes if checked is not None else 0)
        )
    if checked is not None:
        if budget is not None:
            budget.recover_staged_transfer(source.id, parts.name, checked.value.size)
        return checked
    parts.body.parent.mkdir(exist_ok=True)
    sync_directory(parts.root)
    if isinstance(source.acquisition, WikimediaDumpAcquisition):
        return _wikimedia_body(
            source, source.acquisition, parts.download, resources, cache, parts=parts
        )
    assert parts.sha256 is not None
    size = _fetch_pinned(
        source,
        source.canonical_uri,
        parts.name,
        parts.sha256,
        parts.max_bytes,
        parts.body,
        budget,
        resources,
        cache,
        parts=parts,
    )
    return CheckedCheckpoint(parts.body, PinnedBody(parts.sha256, size), None)


class _CountedStream:
    """Bound the bytes an archive decompressor returns, before tar parses them."""

    def __init__(self, stream: Any, limit: int, resources: _AcquisitionResources):
        self.stream = stream
        self.limit = limit
        self.size = 0
        self.resources = resources

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(min(1024 * 1024, size if size >= 0 else 1024 * 1024))
        self.size += len(chunk)
        self.resources.expanded = self.size
        if self.size > self.limit:
            raise ValueError("archive expanded bytes exceed max_expanded_bytes")
        return chunk


def _archive_member(name: str) -> str | None:
    """An archive member's logical path; any unsafe path refuses the whole archive."""
    logical = name
    while logical.startswith("./"):
        logical = logical[2:]
    logical = logical.removesuffix("/")
    if logical in ("", "."):
        return None  # the archive root directory entry
    try:
        return safe_name(logical)
    except ValueError as error:
        raise ValueError(f"archive member has an unsafe path: {name!r}") from error


def _acquire_archive(
    source: SourceDeclaration,
    staging: Path,
    body: Path,
    archive_bytes: int,
    resources: _AcquisitionResources,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import tarfile
    import zipfile
    import zlib

    spec = source.acquisition
    assert isinstance(spec, ArchiveAcquisition)
    inventory: list[dict[str, Any]] = []
    names: set[str] = set()
    parents: set[str] = set()
    scanned = 0

    def selected(member: str) -> bool:
        return any(
            fnmatch.fnmatchcase(member, item) for item in spec.include
        ) and not any(fnmatch.fnmatchcase(member, item) for item in spec.exclude)

    def admit(member: str, size: int) -> int:
        ancestors = {
            "/".join(member.split("/")[:n]) for n in range(1, member.count("/") + 1)
        }
        if member in names or member in parents or ancestors & names:
            raise ValueError(f"archive members collide: {member}")
        if len(names) >= spec.max_members:
            raise ValueError("archive selection exceeds max_members")
        if size > spec.max_member_bytes:
            raise ValueError(f"archive member exceeds max_member_bytes: {member}")
        names.add(member)
        parents.update(ancestors)
        return min(
            spec.max_member_bytes, spec.max_bytes - sum(i["size"] for i in inventory)
        )

    def copy(stream: Any, member: str, limit: int) -> None:
        digest, size = _copy_stream(
            stream, staging / "files" / member, limit, resources
        )
        inventory.append({"path": member, "sha256": digest, "size": size})

    try:
        if spec.format == "zip":
            with zipfile.ZipFile(body) as archive:
                expanded = 0
                for info in archive.infolist():
                    scanned += 1
                    member = _archive_member(info.filename)
                    if member is None or info.is_dir() or not selected(member):
                        continue
                    mode = info.external_attr >> 16
                    if info.flag_bits & 0x1 or (mode and not stat.S_ISREG(mode)):
                        raise ValueError(
                            f"archive member is not a plain regular file: {member}"
                        )
                    limit = admit(member, info.file_size)
                    expanded += info.file_size
                    resources.expanded = expanded
                    if expanded > spec.max_expanded_bytes:
                        raise ValueError(
                            "archive expanded bytes exceed max_expanded_bytes"
                        )
                    with archive.open(info) as stream:
                        copy(stream, member, limit)
        else:
            with body.open("rb") as raw:
                stream = gzip.GzipFile(fileobj=raw) if spec.format == "tar.gz" else raw
                counted = _CountedStream(stream, spec.max_expanded_bytes, resources)
                with tarfile.open(fileobj=counted, mode="r|") as archive:
                    for info in archive:
                        scanned += 1
                        member = _archive_member(info.name)
                        if member is None or info.isdir() or not selected(member):
                            continue
                        if not info.isreg() or info.issparse():
                            raise ValueError(
                                f"archive member is not a plain regular file: {member}"
                            )
                        limit = admit(member, info.size)
                        extracted = archive.extractfile(info)
                        assert extracted is not None
                        with extracted:
                            copy(extracted, member, limit)
    except (
        tarfile.TarError,
        zipfile.BadZipFile,
        gzip.BadGzipFile,
        zlib.error,
    ) as error:
        raise ValueError(f"invalid {spec.format} archive: {error}") from error
    resources.observe()
    if not inventory:
        raise ValueError("archive include patterns selected no files")
    return sorted(inventory, key=lambda item: item["path"]), {
        "archive_sha256": spec.expected_sha256,
        "archive_bytes": archive_bytes,
        "format": spec.format,
        "scanned_members": scanned,
        "resources": resources.receipt(),
    }


def _verify_archive_files(source: SourceDeclaration, manifest: dict[str, Any]) -> None:
    spec = source.acquisition
    assert isinstance(spec, ArchiveAcquisition)
    files = manifest["files"]
    if (
        not files
        or len(files) > spec.max_members
        or sum(item["size"] for item in files) > spec.max_bytes
        or any(
            item["size"] > spec.max_member_bytes
            or not any(fnmatch.fnmatchcase(item["path"], p) for p in spec.include)
            or any(fnmatch.fnmatchcase(item["path"], p) for p in spec.exclude)
            for item in files
        )
    ):
        raise ValueError("archive snapshot differs from its declared selection")


def _pypdf(spec: PdfAcquisition) -> Any:
    try:
        import pypdf
    except ImportError as error:
        raise ValueError(
            "pdf_documents sources need the optional `pdf` extra (uv sync --extra pdf)"
        ) from error
    if pypdf.__version__ != spec.extraction.version:
        raise ValueError(
            f"PDF text extraction is pinned to pypdf {spec.extraction.version}; "
            f"installed pypdf is {pypdf.__version__}"
        )
    return pypdf


def _pdf_text(path: Path, spec: PdfAcquisition, name: str) -> tuple[bytes, int]:
    """Extract one PDF in a child process killed at the declared deadline."""
    import sys

    from klab.runtime_env_subprocess import run_bounded

    extraction = spec.extraction
    command = [
        sys.executable,
        "-I",
        "-m",
        "klab.corpus.pdf_text",
        str(path),
        extraction.version,
        str(spec.max_pages),
    ]
    try:
        result = run_bounded(
            command,
            timeout=extraction.max_seconds,
            output_limit=spec.max_text_bytes + 64,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            f"PDF text extraction exceeded {extraction.max_seconds} s: {name}"
        ) from error
    except ValueError as error:
        raise ValueError(
            f"PDF extracted text exceeds max_text_bytes: {name}"
        ) from error
    if result.returncode != 0:
        reason = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"PDF text extraction refused {name}: {reason}")
    pages, _, text = result.stdout.partition(b"\n")
    return text, int(pages)


def pdf_parts(
    parent: Path,
    source: SourceDeclaration,
    budget: TransportBudget | None,
    *,
    receipt_limit: int | None = None,
) -> PDFParts:
    """Bind PDF checkpoints to the pinned parser and the acquisition request."""
    spec = source.acquisition
    assert isinstance(spec, PdfAcquisition)
    if receipt_limit is not None:
        # The fixed envelope, SHA-256 fields and counters fit within 512 bytes
        # beyond the encoded document name.
        receipt_limit = min(
            receipt_limit,
            sum(512 + len(canonical_json(d.name)) for d in spec.documents),
        )
    return PDFParts(
        parent,
        canonical_digest(
            {
                "request": _acquisition_part_identity(source, budget),
                "extractor_sha256": sha256_file(
                    Path(__file__).with_name("pdf_text.py")
                ),
            }
        ),
        receipt_limit=receipt_limit,
    )


def _acquire_pdf(
    source: SourceDeclaration,
    budget: TransportBudget | None,
    resources: _AcquisitionResources,
    cache: TransportCache | None = None,
    *,
    parts: PDFParts,
    planned_reuse: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = source.acquisition
    assert isinstance(spec, PdfAcquisition)
    _pypdf(spec)  # refuse a missing extra or another version before any work
    resources.expanded = None  # PDF bodies are kept whole, not expanded
    resources.observe()
    if budget is not None:
        budget.check_disk(
            parts.staging, budget.projected_disk_bytes(source_id=source.id)
        )
    inventory = []
    receipts = []
    text_bytes = 0
    reused_bodies = reused_text = 0
    planned_reuse = planned_reuse or {}
    for document in spec.documents:
        resources.index = None
        if budget is not None:
            budget.remaining_seconds()
        pinned, target = parts.body(document), parts.output(document)
        size, selected = parts.verified_document(document, spec)
        planned = planned_reuse.get(document.name)
        if (planned == "text" and selected is None) or (
            planned == "body" and size is None
        ):
            raise ValueError(f"PDF checkpoint changed after admission: {document.name}")
        if size is None:
            pinned.unlink(missing_ok=True)
            size = _fetch_pinned(
                source,
                document.url,
                document.name,
                document.expected_sha256,
                document.max_bytes,
                pinned,
                budget,
                resources,
                cache,
                parts=parts,
            )
        else:
            if budget is not None:
                budget.recover_staged_transfer(source.id, document.name, size)
            reused_bodies += 1
        if selected is None:
            if parts.discard_marker(document):
                print(
                    f"{source.id}: rebuilding damaged PDF text {document.name}",
                    file=sys.stderr,
                )
            parts.staged_file_size(target)  # validate a leftover before unlinking it
            target.unlink(missing_ok=True)
            text, pages = _pdf_text(pinned, spec, document.name)
            if len(text) > spec.max_text_bytes - text_bytes:
                raise ValueError("PDF extracted text exceeds max_text_bytes")
            digest, written = _copy_stream(
                io.BytesIO(text), target, len(text), resources
            )
            parts.commit(document, digest, written, pages)
        else:
            digest, written, pages = selected
            reused_text += 1
        text_bytes += written
        if text_bytes > spec.max_text_bytes:
            raise ValueError("PDF extracted text exceeds max_text_bytes")
        inventory.append({"path": document.name, "sha256": digest, "size": written})
        receipts.append(
            {
                "name": document.name,
                "url": document.url,
                "pdf_path": pinned_pdf_path(document.name),
                "pdf_sha256": document.expected_sha256,
                "pdf_bytes": size,
                "pages": pages,
                "text_sha256": digest,
            }
        )
    if reused_bodies:
        print(
            f"{source.id}: reused {reused_bodies} pinned PDFs and {reused_text} extracted texts",
            file=sys.stderr,
        )
    return inventory, {
        "extraction": spec.extraction.model_dump(mode="json"),
        "documents": receipts,
    }


def _verify_pdf_receipt(
    path: Path, source: SourceDeclaration, manifest: dict[str, Any]
) -> None:
    """Pinned PDF bytes must still yield exactly the recorded text."""
    spec = source.acquisition
    assert isinstance(spec, PdfAcquisition)
    receipt = manifest["retrieval"]
    documents = receipt["documents"]
    expected_files = [
        {"path": item["name"], "sha256": item["text_sha256"]} for item in documents
    ]
    if (
        receipt["extraction"] != spec.extraction.model_dump(mode="json")
        or [
            (item["name"], item["url"], item["pdf_sha256"], item["pdf_path"])
            for item in documents
        ]
        != [
            (d.name, d.url, d.expected_sha256, pinned_pdf_path(d.name))
            for d in spec.documents
        ]
        or [{"path": f["path"], "sha256": f["sha256"]} for f in manifest["files"]]
        != expected_files
    ):
        raise ValueError("PDF receipt differs from its declaration")
    pinned_root = path / "pinned"
    actual = {
        item.relative_to(pinned_root).as_posix()
        for item in pinned_root.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if actual != {item["pdf_path"] for item in documents}:
        raise ValueError("pinned PDF inventory mismatch")
    _pypdf(spec)  # refuse a missing extra or another version before any work
    for item in documents:
        pinned = pinned_root / safe_name(item["pdf_path"])
        _regular(pinned)
        if (
            pinned.stat().st_size != item["pdf_bytes"]
            or sha256_file(pinned) != item["pdf_sha256"]
        ):
            raise ValueError(f"pinned PDF byte mismatch: {item['name']}")
        text, pages = _pdf_text(pinned, spec, item["name"])
        if (
            pages != item["pages"]
            or hashlib.sha256(text).hexdigest() != item["text_sha256"]
        ):
            raise ValueError(
                f"PDF text is not reproducible from the pinned bytes: {item['name']}"
            )


def _acquire_mediawiki(
    source: SourceDeclaration,
    staging: Path,
    body: Path,
    compressed_bytes: int,
    resources: _AcquisitionResources,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = source.acquisition
    assert isinstance(spec, MediaWikiDumpAcquisition)
    output_name = body.name + ".sample.jsonl"
    output_path = staging / "files" / output_name
    opener = {"bz2": bz2.open, "gzip": gzip.open, "none": open}[spec.compression]
    with opener(body, "rb") as decompressed:
        rows, scanned, emitted, digest = _wiki_jsonl(
            source, decompressed, output_path, resources
        )
    return [{"path": output_name, "sha256": digest, "size": emitted}], {
        "source_uri": source.canonical_uri,
        "source_sha256": spec.expected_sha256,
        "compressed_bytes": compressed_bytes,
        "scanned_pages": scanned,
        "selected_pages": rows,
        "output_path": output_name,
    }


def _verify_wikimedia_receipt(
    path: Path, source: AcquisitionIdentity, manifest: dict[str, Any]
) -> None:
    receipt = manifest["retrieval"]
    spec = source.acquisition
    if isinstance(spec, WikimediaDumpAcquisition):
        index_receipt = receipt.get("multistream_index")
        if spec.multistream_index is None:
            if index_receipt is not None:
                raise ValueError("Wikimedia snapshot claims an undeclared index")
        elif (
            not isinstance(index_receipt, dict)
            or index_receipt.get("index_sha256")
            != spec.multistream_index.expected_sha256
            or not re.fullmatch(r"[0-9a-f]{64}", index_receipt.get("plan_sha256", ""))
            or type(index_receipt.get("selected_members")) is not int
            or not 0
            < index_receipt["selected_members"]
            <= spec.multistream_index.max_members
        ):
            raise ValueError("Wikimedia multistream index receipt mismatch")
        mismatch = (
            receipt["checksum_uri"] != spec.checksum_uri
            or receipt["source_sha1"] != spec.expected_sha1.lower()
            or (
                spec.expected_sha256
                and receipt["source_sha256"] != spec.expected_sha256.lower()
            )
        )
    else:
        assert isinstance(spec, MediaWikiDumpAcquisition)
        mismatch = receipt["source_sha256"] != spec.expected_sha256
    if (
        mismatch
        or receipt["source_uri"] != source.canonical_uri
        or receipt["compressed_bytes"] > spec.max_compressed_bytes
        or receipt["scanned_pages"] > spec.max_scanned_pages
        or len(receipt["selected_pages"]) > spec.max_selected_pages
        or len(manifest["files"]) != 1
        or receipt["output_path"] != manifest["files"][0]["path"]
        or manifest["files"][0]["size"] > spec.max_emitted_bytes
    ):
        raise ValueError("Wikimedia receipt mismatch")
    entry = manifest["files"][0]
    observed_size = 0
    digest = hashlib.sha256()
    pages = iter(receipt["selected_pages"])
    missing = object()
    with (path / "files" / receipt["output_path"]).open("rb") as handle:
        for line in handle:
            observed_size += len(line)
            digest.update(line)
            row = json.loads(line)
            metadata = row["_klab_source"]
            if (
                not isinstance(row["text"], str)
                or metadata["source_uri"] != source.canonical_uri
                or metadata["dump_revision"] != source.revision
            ):
                raise ValueError("Wikimedia row provenance mismatch")
            if next(pages, missing) != {
                "page_id": metadata["page_id"],
                "revision_id": metadata["revision_id"],
                "row_sha256": hashlib.sha256(line).hexdigest(),
            }:
                raise ValueError("Wikimedia page receipt mismatch")
    if next(pages, missing) is not missing:
        raise ValueError("Wikimedia page receipt mismatch")
    if observed_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"snapshot byte mismatch: {entry['path']}")


def _retrieval_is_identity(source: SourceDeclaration | AcquisitionIdentity) -> bool:
    """Receipts that bind derived bytes (rows, extracted text) to their inputs."""
    return source.kind in {"wikimedia_dump", "mediawiki_dump", "pdf_documents"} or (
        source.kind == "huggingface_dataset"
        and source.acquisition.retained_subset is not None
    )


def verify_snapshot(
    path: Path | str,
    *,
    _staged: bool = False,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, Any]:
    """Authenticate a snapshot; trusted receipts are optional for published roots."""
    path = Path(path)
    if (
        not _staged
        and not _domain_cold
        and proof_store is not None
        and verification_mode == "verified_reuse"
    ):
        from klab.experiments.artifacts import Artifact, verify_artifact

        manifest = read_metadata(path / "manifest.json")
        verify_artifact(
            Artifact(
                kind="source_snapshot",
                version=manifest["schema_version"],
                producer="klab",
                identifier=manifest["source_id"],
                sha256=manifest["snapshot_sha256"],
                path=str(path),
            ),
            path / "manifest.json",
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        return manifest
    return _verify_snapshot_cold(path, _staged=_staged)


def alias_verified_snapshot(
    project: Project, work_root: Path, *, source_id: str, snapshot: Path
) -> dict[str, str]:
    """Reuse exact immutable snapshot bytes across projects without a transfer."""
    if project.config.source_effects is not None:
        raise ValueError("effect-bound reuse is performed by acquire preflight")
    sources = [source for source in project.sources if source.id == source_id]
    if len(sources) != 1 or sources[0].kind not in IMMUTABLE_SOURCE_KINDS:
        raise ValueError("snapshot alias source is missing or not immutable")
    root = work_root.resolve(strict=True)
    if not snapshot.is_absolute() or snapshot.is_symlink():
        raise ValueError("snapshot alias origin must be an absolute directory")
    origin = snapshot.resolve(strict=True)
    if root not in origin.parents or not origin.is_dir():
        raise ValueError("snapshot alias origin is outside the persistent root")
    manifest = verify_snapshot(origin, verification_mode="cold")
    if (
        manifest["source_id"] != source_id
        or manifest["declaration_sha256"] != declaration_sha256(sources[0])
        or manifest["declaration"] != acquisition_identity_payload(sources[0])
    ):
        raise ValueError("snapshot alias differs from current source declaration")
    target = (
        root / "corpora" / project.config.id / "snapshots" / source_id / origin.name
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(origin, target_is_directory=True)
    if (
        verify_snapshot(target, verification_mode="cold")["snapshot_sha256"]
        != origin.name
    ):
        raise ValueError("snapshot alias failed cold readback")
    return {
        "source_id": source_id,
        "snapshot_sha256": origin.name,
        "alias": str(target),
    }


def _verify_snapshot_cold(path: Path, *, _staged: bool = False) -> dict[str, Any]:
    path = Path(path)
    try:
        manifest = read_metadata(path / "manifest.json")
        if (
            manifest["schema_version"] != 1
            or manifest["source_id"] != manifest["declaration"]["id"]
        ):
            raise ValueError("snapshot metadata mismatch")
        source = AcquisitionIdentity.model_validate(manifest["declaration"])
        if (
            declaration_sha256(source) != manifest["declaration_sha256"]
            or manifest["adapter"]["id"] != source.kind
        ):
            raise ValueError("snapshot declaration/adapter identity mismatch")
        identity_fields = {
            "declaration_sha256": manifest["declaration_sha256"],
            "adapter": manifest["adapter"],
            "files": manifest["files"],
        }
        if _retrieval_is_identity(source):
            identity_fields["retrieval"] = manifest["retrieval"]
        expected = canonical_digest(identity_fields)
        if expected != manifest["snapshot_sha256"] or (
            not _staged and path.name != expected
        ):
            raise ValueError("snapshot identity mismatch")
        names = set()
        checks = []
        for entry in manifest["files"]:
            name = safe_name(entry["path"])
            if name in names:
                raise ValueError("duplicate snapshot path")
            names.add(name)
            file = path / "files" / name
            _regular(file)
            checks.append((name, file, entry))

        wiki = source.kind in {"wikimedia_dump", "mediawiki_dump"}

        def check(item) -> None:
            name, file, entry = item
            if (
                not file.resolve().is_relative_to((path / "files").resolve())
                or file.stat().st_size != entry["size"]
                or (not wiki and sha256_file(file) != entry["sha256"])
            ):
                raise ValueError(f"snapshot byte mismatch: {name}")

        if wiki:
            for item in checks:
                check(item)
            _verify_wikimedia_receipt(path, source, manifest)
        else:
            # Hashing releases the GIL, so threads read and hash files concurrently.
            with ThreadPoolExecutor(max_workers=worker_count()) as pool:
                list(pool.map(check, checks))
        actual = {
            p.relative_to(path / "files").as_posix()
            for p in (path / "files").rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if actual != names:
            raise ValueError("snapshot file inventory mismatch")
        if source.kind == "archive":
            _verify_archive_files(source, manifest)
        if source.kind == "pdf_documents":
            _verify_pdf_receipt(path, source, manifest)
        if (
            source.kind == "huggingface_dataset"
            and source.acquisition.retained_subset is not None
        ):
            from klab.corpus.retained_subset import derive_subset

            files, retrieval = derive_subset(source, scratch_directory=path.parent)
            if files != manifest["files"] or retrieval != manifest["retrieval"]:
                raise ValueError("retained subset differs from cold parent replay")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid/incomplete snapshot: {path}") from error


def _project_sha(project: Project) -> str:
    """Only acquisition identities bind the lock; rights and release variants share snapshots."""
    identity = {
        "project_id": project.config.id,
        "sources": [
            acquisition_identity_payload(s)
            for s in sorted(project.sources, key=lambda s: s.id)
        ],
    }
    if project.config.source_effects is not None:
        identity["source_effects"] = [
            item.model_dump(mode="json") for item in project.config.source_effects
        ]
    return canonical_digest(identity)


def _safe_snapshot_path(path: Path) -> None:
    """Keep cold and proof-backed reuse subject to the same path-security rule."""
    if not path.is_absolute() or symlinked(path):
        raise ValueError(f"unsafe snapshot origin or alias: {path}")
    if path.exists():
        from klab.experiments.artifacts import _safe_path

        _safe_path(str(path), path / "manifest.json")


def _retained_origin(work_root: Path, effect: Any) -> Path:
    origin = (
        work_root.absolute()
        / "corpora"
        / effect.origin_project_id
        / "snapshots"
        / effect.source_id
        / effect.snapshot_sha256
    )
    _safe_snapshot_path(origin)
    return origin


def _verify_retained(
    project: Project,
    work_root: Path,
    source: SourceDeclaration,
    effect: Any,
    *,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> tuple[Path, dict[str, Any]]:
    if effect.origin_project_id == project.config.id:
        raise ValueError("reuse-only origin must be another immutable project")
    origin = _retained_origin(work_root, effect)
    try:
        manifest = verify_snapshot(
            origin, proof_store=proof_store, verification_mode=verification_mode
        )
    except OSError as error:
        raise ValueError(f"missing retained snapshot: {source.id}") from error
    if (
        manifest["source_id"] != source.id
        or manifest["snapshot_sha256"] != effect.snapshot_sha256
        or manifest["declaration_sha256"] != effect.declaration_sha256
        or manifest["declaration"] != acquisition_identity_payload(source)
        or not _reusable_immutable_adapter(source, manifest)
    ):
        raise ValueError(f"retained snapshot identity mismatch: {source.id}")
    return origin, manifest


def verify_acquisition(
    project: Project,
    work_root: Path | str,
    *,
    lock_path: Path | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    base = Path(work_root) / "corpora" / project.config.id
    effects = (
        {item.source_id: item for item in project.config.source_effects}
        if project.config.source_effects is not None
        else None
    )
    selected = Path(lock_path) if lock_path is not None else base / "acquisition.json"
    if selected.is_symlink() or any(parent.is_symlink() for parent in selected.parents):
        raise ValueError("symlinked acquisition lock")
    try:
        lock = read_metadata(selected)
        if (
            lock["schema_version"] != 1
            or lock["project_id"] != project.config.id
            or lock["project_sha256"] != _project_sha(project)
            or set(lock["sources"]) != {s.id for s in project.sources}
        ):
            raise ValueError("acquisition lock does not match project")
        for source in project.sources:
            entry = lock["sources"][source.id]
            if entry["declaration_sha256"] != declaration_sha256(source):
                raise ValueError(f"declaration changed: {source.id}")
            effect = effects[source.id] if effects is not None else None
            if effect is not None and effect.effect == "reuse_only":
                origin, _ = _verify_retained(
                    project,
                    Path(work_root),
                    source,
                    effect,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                if entry.get("reuse_origin") != {
                    "project_id": effect.origin_project_id,
                    "snapshot_sha256": effect.snapshot_sha256,
                    "declaration_sha256": effect.declaration_sha256,
                    "path": str(origin),
                }:
                    raise ValueError("retained origin binding mismatch")
            elif effect is not None and "reuse_origin" in entry:
                raise ValueError("acquired source claims an undeclared reuse origin")
            snapshot = base / "snapshots" / source.id / entry["snapshot_sha256"]
            if effect is not None:
                _safe_snapshot_path(snapshot.absolute())
            if entry["snapshot_path"] != str(snapshot.resolve()):
                raise ValueError("snapshot path mismatch")
            manifest = verify_snapshot(
                snapshot, proof_store=proof_store, verification_mode=verification_mode
            )
            if (
                manifest["snapshot_sha256"] != entry["snapshot_sha256"]
                or manifest["declaration_sha256"] != entry["declaration_sha256"]
                or manifest["declaration"] != acquisition_identity_payload(source)
            ):
                raise ValueError("snapshot provenance mismatch")
            status = "generator" if source.kind.endswith("generator") else "acquired"
            if entry["receipt"] != {
                "status": status,
                "retrieval": manifest["retrieval"],
            }:
                raise ValueError("snapshot receipt mismatch")
        return lock
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("missing or invalid acquisition lock") from error


def _sync_snapshot_directories(staging: Path) -> None:
    for directory in sorted(staging.rglob("*"), reverse=True):
        if directory.is_dir():
            sync_directory(directory)
    sync_directory(staging)


def _reusable_snapshot(
    path: Path,
    checked: CheckedCheckpoint[dict[str, Any]] | None,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> dict[str, Any]:
    if checked is not None and checked.matches(path):
        return checked.value
    return verify_snapshot(
        path, proof_store=proof_store, verification_mode=verification_mode
    )


def acquire(
    project: Project,
    work_root: Path | str,
    offline: bool = False,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    source_id: str | None = None,
    max_new_bytes: int | None = None,
    min_free_bytes: int | None = None,
    plan_only: bool = False,
    require_existing: bool = False,
    transport_cache: Path | str | None = None,
    wikimedia_prefetch_bytes: int = 0,
    wikimedia_parse_bytes: int = 0,
    wikimedia_checkpoint_members: int = 1,
) -> dict[str, Any]:
    """Publish verified snapshots independently; replace the active lock only on success.

    `transport_cache` (else `KLAB_TRANSPORT_CACHE`) names operator-local cold
    storage for raw transport bodies; see `klab.corpus.transport_cache`.
    `wikimedia_prefetch_bytes` bounds expanded bytes buffered ahead of the
    current member during indexed extraction.
    `wikimedia_parse_bytes` bounds candidate JSONL in the process worker window.
    `wikimedia_checkpoint_members` bounds physical members per extraction commit.
    """
    if type(wikimedia_prefetch_bytes) is not int or wikimedia_prefetch_bytes < 0:
        raise ValueError("Wikimedia prefetch bytes must be a nonnegative integer")
    if type(wikimedia_parse_bytes) is not int or wikimedia_parse_bytes < 0:
        raise ValueError("Wikimedia parse bytes must be a nonnegative integer")
    if (
        type(wikimedia_checkpoint_members) is not int
        or not 1 <= wikimedia_checkpoint_members <= 64
    ):
        raise ValueError("Wikimedia checkpoint members must be an integer from 1 to 64")
    if require_existing and (offline or source_id is not None or plan_only):
        raise ValueError(
            "existing-snapshot finalization requires the full project without --offline or a source step"
        )
    step = None
    step_budget = None
    step_retained = None
    checked_snapshot = checked_body = None
    active_sources = project.sources
    if source_id is not None:
        from klab.corpus import acquisition_step

        if offline:
            raise ValueError(
                "source steps cannot use --offline; use --plan for admission"
            )
        (source, step_budget, step_retained, step, checked_snapshot, checked_body) = (
            acquisition_step.prepare(
                project,
                Path(work_root),
                source_id,
                max_new_bytes=max_new_bytes,
                min_free_bytes=min_free_bytes,
                proof_store=proof_store,
                verification_mode=verification_mode,
                plan_only=plan_only,
            )
        )
        if plan_only:
            return {"status": "planned", "complete": False, "admission": step}
        active_sources = (source,)
        step_budget.selected_source_id = source_id
        step_budget.bounded_storage = True
    elif max_new_bytes is not None or min_free_bytes is not None or plan_only:
        raise ValueError("step admission options require --source-id")
    base = Path(work_root) / "corpora" / project.config.id
    selected = [
        source
        for source in project.sources
        if source.kind == "huggingface_dataset"
        and source.acquisition.retained_subset is not None
    ]
    if selected and (
        len(selected) != len(project.sources)
        or project.config.transport_budget is not None
        or project.config.source_effects is not None
        or sum(source.acquisition.max_rows for source in selected) > RETAINED_MAX_ROWS
        or sum(source.acquisition.max_bytes for source in selected) > RETAINED_MAX_BYTES
    ):
        raise ValueError(
            "retained subset project must be offline-only within aggregate selection caps"
        )
    if offline or (selected and (base / "acquisition.json").exists()):
        return verify_acquisition(
            project,
            work_root,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    effects = (
        {item.source_id: item for item in project.config.source_effects}
        if project.config.source_effects is not None
        else None
    )
    retained: dict[str, tuple[Path, dict[str, Any]]] = step_retained or {}
    if effects is not None and step_retained is None:
        # Complete rights/source permission and all retained-byte checks happen
        # before the first metadata request, transfer, or acquisition lock write.
        for source in project.sources:
            effect = effects[source.id]
            if effect.effect == "reuse_only":
                retained[source.id] = _verify_retained(
                    project,
                    Path(work_root),
                    source,
                    effect,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
    snapshot_root = base / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    cache_root = resolve_transport_cache(transport_cache)
    cache = TransportCache(cache_root) if cache_root is not None else None
    budget = step_budget or (
        TransportBudget(base / "transport-budget.sqlite", project)
        if project.config.transport_budget is not None
        else None
    )
    entries = {}
    old_sources = None
    for source in active_sources:
        declared_digest = declaration_sha256(source)
        parent = snapshot_root / source.id
        if not require_existing and step is None:
            reclaim_disposable_staging(parent)
        effect = effects[source.id] if effects is not None else None
        if effect is not None and effect.effect == "reuse_only":
            origin, manifest = retained[source.id]
            destination = snapshot_root / source.id / manifest["snapshot_sha256"]
            _safe_snapshot_path(destination.absolute())
            if not destination.exists():
                if require_existing:
                    raise ValueError(
                        f"existing verified snapshot unavailable: {source.id}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with acquisition_staging(destination.parent, None) as staging:
                    shutil.copytree(origin, staging, dirs_exist_ok=True, symlinks=True)
                    if (
                        verify_snapshot(staging, _staged=True)["snapshot_sha256"]
                        != manifest["snapshot_sha256"]
                    ):
                        raise ValueError("retained snapshot copy differs from origin")
                    for member in sorted(staging.rglob("*"), reverse=True):
                        if member.is_file():
                            with member.open("rb") as stream:
                                os.fsync(stream.fileno())
                        elif member.is_dir():
                            sync_directory(member)
                    sync_directory(staging)
                    rename_noreplace(staging, destination)
                    sync_directory(destination.parent)
            copied = verify_snapshot(
                destination,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            if copied != manifest:
                raise ValueError("retained snapshot copy identity mismatch")
            entries[source.id] = {
                "declaration_sha256": declared_digest,
                "snapshot_sha256": manifest["snapshot_sha256"],
                "snapshot_path": str(destination.resolve()),
                "reuse_origin": {
                    "project_id": effect.origin_project_id,
                    "snapshot_sha256": effect.snapshot_sha256,
                    "declaration_sha256": effect.declaration_sha256,
                    "path": str(origin),
                },
                "receipt": {"status": "acquired", "retrieval": manifest["retrieval"]},
            }
            continue
        if source.kind not in ("local", "http_document"):
            # Read the advisory lock once. Reuse still requires snapshot verification.
            if old_sources is None:
                previous = base / "acquisition.json"
                try:
                    old_sources = read_metadata(previous)["sources"]
                    if not isinstance(old_sources, dict):
                        raise TypeError("sources must be a mapping")
                except FileNotFoundError:
                    old_sources = {}
                except (OSError, ValueError, TypeError, KeyError, MemoryError) as error:
                    print(
                        f"cannot read advisory acquisition lock at {previous} "
                        f"({type(error).__name__}): {error}",
                        file=sys.stderr,
                    )
                    old_sources = {}
            if (old := old_sources.get(source.id)) is not None:
                try:
                    old_path = parent / old["snapshot_sha256"]
                    if old["declaration_sha256"] == declared_digest:
                        manifest = _reusable_snapshot(
                            old_path, checked_snapshot, proof_store, verification_mode
                        )
                        if manifest["declaration"] == acquisition_identity_payload(
                            source
                        ) and _reusable_immutable_adapter(source, manifest):
                            _reclaim_published_pinned_parts(
                                parent, source, budget, manifest
                            )
                            entries[source.id] = old
                            continue
                except OSError, ValueError, TypeError, KeyError:
                    pass
        # An interrupted run can publish verified source snapshots before
        # writing its final lock. Only immutable upstream revisions are reusable.
        if source.kind in IMMUTABLE_SOURCE_KINDS and parent.exists():
            for candidate in sorted(parent.iterdir()):
                if not candidate.is_dir() or candidate.name.startswith("."):
                    continue
                try:
                    manifest = _reusable_snapshot(
                        candidate, checked_snapshot, proof_store, verification_mode
                    )
                except OSError, ValueError, KeyError, TypeError, json.JSONDecodeError:
                    continue
                if (
                    manifest["declaration_sha256"] != declared_digest
                    or manifest["declaration"] != acquisition_identity_payload(source)
                    or not _reusable_immutable_adapter(source, manifest)
                ):
                    continue
                _reclaim_published_pinned_parts(parent, source, budget, manifest)
                entries[source.id] = {
                    "declaration_sha256": declared_digest,
                    "snapshot_sha256": manifest["snapshot_sha256"],
                    "snapshot_path": str(candidate.resolve()),
                    "receipt": {
                        "status": "acquired",
                        "retrieval": manifest["retrieval"],
                    },
                }
                break
            if source.id in entries:
                continue
        if require_existing:
            raise ValueError(f"existing verified snapshot unavailable: {source.id}")
        parent.mkdir(parents=True, exist_ok=True)
        receipt_limit = acquisition_step.MANIFEST_BYTES if step is not None else None
        parts: AcquisitionParts | None = bounded_hf_parts(
            parent, source, budget, receipt_limit=receipt_limit
        )
        if (
            source.kind == "git"
            and source.acquisition.bounded_blobs is not None
            and budget is not None
        ):
            parts = AcquisitionParts(
                parent, _acquisition_part_identity(source, budget), adapter="git"
            )
        elif source.kind == "pdf_documents":
            parts = pdf_parts(parent, source, budget, receipt_limit=receipt_limit)
        pinned = pinned_body_parts(parent, source, budget)
        if pinned is not None:
            parts = pinned
        with acquisition_staging(parent, parts) as staging:
            resources = None
            try:
                sealed = None
                planned_pin = step["pinned_checkpoint_reuse"] if step else None
                if pinned is not None:
                    if budget is not None:
                        budget.remaining_seconds()
                    sealed = verified_pinned_derivation(
                        source,
                        pinned,
                        manifest_limit=receipt_limit,
                        previous=checked_snapshot,
                    )
                    if sealed is None:
                        if planned_pin == "derivation":
                            raise ValueError(
                                f"pinned derivation changed after admission: {source.id}"
                            )
                        pinned.reset_derivation()
                (staging / "files").mkdir(exist_ok=True)
                if pinned is not None or source.kind in PINNED_FILE_KINDS:
                    resources = _AcquisitionResources(
                        pinned.root if pinned is not None else staging,
                        budget,
                        journal_parent=parent,
                    )
                    if budget is None:
                        resources.retain(source)
                retrieval: dict[str, Any] = {}
                if sealed is not None:
                    files, retrieval = sealed.value["files"], sealed.value["retrieval"]
                    assert resources is not None
                    resources.observe()
                elif pinned is not None:
                    assert resources is not None
                    body = _resume_pinned_body(
                        source,
                        pinned,
                        budget,
                        resources,
                        cache,
                        admit_body=planned_pin == "body",
                        previous=checked_body,
                    )
                    if source.kind == "wikimedia_dump":
                        files, retrieval = _acquire_wikimedia(
                            source,
                            staging,
                            pinned,
                            body,
                            resources,
                            cache,
                            wikimedia_prefetch_bytes,
                            wikimedia_parse_bytes,
                            wikimedia_checkpoint_members,
                        )
                    else:
                        derive = (
                            _acquire_archive
                            if source.kind == "archive"
                            else _acquire_mediawiki
                        )
                        files, retrieval = derive(
                            source, staging, pinned.body, body.value.size, resources
                        )
                elif source.kind == "local":
                    files, retrieval = _acquire_local(source, project.root, staging)
                elif source.kind == "git":
                    files, retrieval = _acquire_git(
                        source,
                        staging,
                        base / "git-cache",
                        False,
                        budget,
                        cache,
                        parts=parts,
                    )
                elif source.kind == "http_document":
                    files, retrieval = _acquire_http(source, staging, False, cache)
                elif source.kind == "huggingface_dataset":
                    files, retrieval = _acquire_hf(
                        source,
                        staging,
                        base / "hf-cache",
                        False,
                        budget,
                        cache,
                        parts=parts if isinstance(parts, HFParts) else None,
                        planned_reuse=step["hf_checkpoint_reuse"] if step else None,
                    )
                elif source.kind == "pdf_documents":
                    assert isinstance(parts, PDFParts)
                    files, retrieval = _acquire_pdf(
                        source,
                        budget,
                        resources,
                        cache,
                        parts=parts,
                        planned_reuse=step["pdf_checkpoint_reuse"] if step else None,
                    )
                else:
                    files = []
                    retrieval = {
                        "generator_config_sha256": canonical_digest(
                            source.acquisition.model_dump(mode="json")
                        )
                    }
                publication_check = sealed
                if sealed is not None:
                    identity = sealed.value["snapshot_sha256"]
                else:
                    adapter = _adapter(source)
                    identity_fields = {
                        "declaration_sha256": declared_digest,
                        "adapter": adapter,
                        "files": files,
                    }
                    if _retrieval_is_identity(source):
                        identity_fields["retrieval"] = retrieval
                    identity = canonical_digest(identity_fields)
                    manifest = {
                        "schema_version": 1,
                        "source_id": source.id,
                        "declaration": acquisition_identity_payload(source),
                        "declaration_sha256": declared_digest,
                        "adapter": adapter,
                        "files": files,
                        "retrieval": retrieval,
                        "snapshot_sha256": identity,
                    }
                    if (
                        step is not None
                        and len(canonical_json(manifest)) + 1
                        > acquisition_step.MANIFEST_BYTES
                    ):
                        raise ValueError("acquisition step manifest byte cap exceeded")
                    write_canonical_json(staging / "manifest.json", manifest)
                    if pinned is not None:
                        _sync_snapshot_directories(staging)
                        publication_check = check_checkpoint(
                            staging, lambda path: verify_snapshot(path, _staged=True)
                        )
                        assert publication_check is not None
                        staged = publication_check.value
                    else:
                        staged = verify_snapshot(staging, _staged=True)
                    if staged["snapshot_sha256"] != identity:
                        raise ValueError("staged snapshot identity mismatch")
                destination = parent / identity
                if destination.exists():
                    verify_snapshot(
                        destination,
                        proof_store=proof_store,
                        verification_mode=verification_mode,
                    )
                else:
                    if pinned is None or sealed is not None:
                        _sync_snapshot_directories(staging)
                    rename_noreplace(staging, destination)
                    sync_directory(parent)
                    if pinned is not None:
                        pinned.discard_body()
                    if publication_check is None or not publication_check.matches(
                        destination, renamed=True
                    ):
                        verify_snapshot(
                            destination,
                            proof_store=proof_store,
                            verification_mode=verification_mode,
                        )
                entries[source.id] = {
                    "declaration_sha256": declared_digest,
                    "snapshot_sha256": identity,
                    "snapshot_path": str(destination.resolve()),
                    "receipt": {
                        "status": "generator"
                        if not files and source.kind.endswith("generator")
                        else "acquired",
                        "retrieval": retrieval,
                    },
                }
                if cache is not None and (evidence := cache.take_evidence()):
                    # Provenance of the bytes, outside the snapshot identity: the
                    # same snapshot is reached whether the network or the cache served it.
                    entries[source.id]["transport_cache"] = evidence
            except BaseException as error:
                if resources is not None:
                    resources.status = (
                        "interrupted"
                        if isinstance(
                            error, (OSError, EOFError, KeyboardInterrupt, SystemExit)
                        )
                        else "failed"
                    )
                    resources.persist()
                if budget is not None:
                    budget.record_failure(source.id, error)
                raise
            else:
                if resources is not None:
                    resources.status = "complete"
                    resources.persist()
    if effects is not None:
        # An origin changed during the permitted transfer cannot be sealed in
        # the new lock merely because its preflight passed earlier.
        for source in project.sources:
            if effects[source.id].effect != "reuse_only":
                continue
            _, current = _verify_retained(
                project,
                Path(work_root),
                source,
                effects[source.id],
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            if current != retained[source.id][1]:
                raise ValueError(f"retained origin changed before lock: {source.id}")
    if step is not None:
        entry = entries[source_id]
        return {
            "status": "source_verified",
            "complete": False,
            "admission": step,
            "source": {
                key: entry[key]
                for key in (
                    "declaration_sha256",
                    "snapshot_sha256",
                    "snapshot_path",
                    "transport_cache",
                )
                if key in entry
            },
        }
    lock = {
        "schema_version": 1,
        "project_id": project.config.id,
        "project_sha256": _project_sha(project),
        "sources": entries,
    }
    handle, name = tempfile.mkstemp(prefix=".acquisition-", dir=base)
    try:
        with os.fdopen(handle, "wb") as output:
            encoder = json.JSONEncoder(
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            for part in encoder.iterencode(lock):
                output.write(part.encode("utf-8"))
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, base / "acquisition.json")
        sync_directory(base)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return verify_acquisition(
        project, work_root, proof_store=proof_store, verification_mode=verification_mode
    )


def _fill_source(
    source: SourceDeclaration,
    base: Path,
    staging: Path,
    cache: TransportCache,
    budget: TransportBudget | None,
) -> str | None:
    """Put one source's declared bodies in the cache; the reason it cannot, if so."""
    resources = _AcquisitionResources(staging, budget)
    spec = source.acquisition
    kind = source.kind
    if kind == "huggingface_dataset":
        if spec.retained_subset is not None:
            return "derived offline from an existing snapshot"
        if spec.bounded_shards is None:
            return "snapshot_download selects files by pattern; no pinned body"
        for shard in spec.bounded_shards:
            if shard.declared_config is not None and budget is None:
                raise ValueError(
                    "explicit HF shard metadata requires a transport budget"
                )
            _hf_shard_body(
                source,
                spec,
                shard,
                staging / "hf-input" / shard.path,
                budget,
                resources,
                cache,
                verify_only=True,
            )
    elif kind == "git":
        if spec.bounded_blobs is None:
            _ensure_git_repository(source, base / "git-cache", cache)
            return None
        if budget is None:
            raise ValueError("bounded Git acquisition requires a transport ledger")
        owner, repository = _github_origin(source)
        checked = False

        def ensure_metadata() -> None:
            nonlocal checked
            if not checked:
                _check_git_metadata(source, spec, owner, repository, budget)
                checked = True

        for blob in spec.bounded_blobs:
            _git_blob_body(
                source,
                blob,
                _git_blob_url(source, owner, repository, blob.path),
                staging / "files" / blob.path,
                budget,
                _AcquisitionResources(staging, budget),
                cache,
                ensure_metadata,
                verify_only=True,
            )
    elif kind == "http_document":
        _acquire_http(source, staging, False, cache, verify_only=True)
    elif kind == "wikimedia_dump":
        _wikimedia_body(
            source,
            spec,
            staging
            / "wikimedia-input"
            / Path(urlparse(source.canonical_uri).path).name,
            resources,
            cache,
            verify_only=True,
        )
        if spec.multistream_index is not None:
            index = spec.multistream_index
            name = source_file_name(index.uri)
            _fetch_pinned(
                source,
                index.uri,
                name,
                index.expected_sha256,
                index.max_bytes,
                staging / "wikimedia-input" / name,
                None,
                resources,
                cache,
                verify_only=True,
            )
    elif kind in {"archive", "mediawiki_dump"}:
        name = source_file_name(source.canonical_uri)
        limit = (
            spec.max_archive_bytes if kind == "archive" else spec.max_compressed_bytes
        )
        _fetch_pinned(
            source,
            source.canonical_uri,
            name,
            spec.expected_sha256,
            limit,
            staging / "pinned" / name,
            budget,
            resources,
            cache,
            verify_only=True,
        )
    elif kind == "pdf_documents":
        for document in spec.documents:
            _fetch_pinned(
                source,
                document.url,
                document.name,
                document.expected_sha256,
                document.max_bytes,
                staging / "pinned" / pinned_pdf_path(document.name),
                budget,
                resources,
                cache,
                verify_only=True,
            )
    else:
        return "no network transport"
    return None


def fill_transport_cache(
    project: Project,
    work_root: Path | str,
    *,
    transport_cache: Path | str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Download every declared body into the transport cache; make no snapshot.

    Bodies already cached are re-hashed, not fetched. Downloads are charged to
    the project's transport ledger exactly as acquisition would charge them.
    """
    root = resolve_transport_cache(transport_cache)
    if root is None:
        raise ValueError(
            "no transport cache: pass --transport-cache or set KLAB_TRANSPORT_CACHE"
        )
    sources = [s for s in project.sources if source_id in (None, s.id)]
    if not sources:
        raise ValueError(f"unknown source: {source_id}")
    base = Path(work_root) / "corpora" / project.config.id
    cache = TransportCache(root)
    effects = (
        {item.source_id: item for item in project.config.source_effects}
        if project.config.source_effects is not None
        else {}
    )
    budget = (
        TransportBudget(base / "transport-budget.sqlite", project)
        if project.config.transport_budget is not None
        else None
    )
    report = []
    for source in sources:
        if (
            effects.get(source.id) is not None
            and effects[source.id].effect == "reuse_only"
        ):
            report.append(
                {"source_id": source.id, "skipped": "reuses a retained snapshot"}
            )
            continue
        staging = Path(tempfile.mkdtemp(prefix="fill-", dir=root / "tmp"))
        try:
            reason = _fill_source(source, base, staging, cache, budget)
        except BaseException as error:
            if budget is not None:
                budget.record_failure(source.id, error)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        evidence = cache.take_evidence()
        report.append(
            {"source_id": source.id, "skipped": reason}
            if reason is not None
            else {"source_id": source.id, "bodies": evidence}
        )
    return {"transport_cache": str(root), "sources": report}
