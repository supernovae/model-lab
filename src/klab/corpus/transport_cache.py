"""Operator-local, content-addressed cold storage for raw transport bodies.

The cache holds exactly the bytes an acquisition would have downloaded (a
Hugging Face shard, an archive, a PDF, a dump, an HTTP document, a Git blob),
keyed by what was asked for and not by the declaration that asked. A declaration
edit therefore never invalidates it. Nothing in it is trusted: every use
re-hashes the body and re-checks the key and the pins before a byte is copied
into a snapshot, and a body that fails is re-fetched, never repaired in place.

Layout under the cache root::

    bodies/<aa>/<sha256>     raw body, read-only, named by its own SHA-256
    keys/<key-digest>.json   what was asked for, and the body it resolved to
    git/<key-digest>.git     bare repository for a pattern-selected Git source
    tmp/                     private staging for atomic writes
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.result_publication import sync_directory

TRANSPORT_CACHE_ENV = "KLAB_TRANSPORT_CACHE"
_CHUNK = 1024 * 1024
_ORPHAN_GRACE_SECONDS = 3600.0


def resolve_transport_cache(path: str | Path | None = None) -> Path | None:
    """The cache root: explicit path, then `KLAB_TRANSPORT_CACHE`, else none.

    A host setting, never recipe content. Without one, nothing is cached.
    """
    configured = path if path is not None else os.environ.get(TRANSPORT_CACHE_ENV)
    if configured is None or not os.fspath(configured).strip():
        return None
    resolved = Path(configured).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


@dataclass(frozen=True)
class BodyKey:
    """What was asked for. Every field participates in the key digest."""

    kind: str
    url: str
    revision: str | None = None
    sha256: str | None = None
    sha1: str | None = None
    git_blob_oid: str | None = None

    def identity(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "url": self.url,
            "revision": self.revision,
            "sha256": self.sha256.lower() if self.sha256 else None,
            "sha1": self.sha1.lower() if self.sha1 else None,
            "git_blob_oid": self.git_blob_oid.lower() if self.git_blob_oid else None,
        }

    @property
    def digest(self) -> str:
        return canonical_digest({"transport-cache-key": self.identity()})


@dataclass(frozen=True)
class Served:
    """One body and where it came from."""

    served_from: str  # "cache" or "network"
    cache_key: str
    sha256: str
    size: int
    meta: dict[str, Any] | None = None


@dataclass
class _Sums:
    sha256: Any = field(default_factory=hashlib.sha256)
    sha1: Any = None
    size: int = 0


def git_blob_hasher(size: int) -> Any:
    """SHA-1 state seeded with the object's Git blob header."""
    return hashlib.sha1(f"blob {size}\0".encode())


class TransportCache:
    def __init__(self, root: Path):
        self.root = root
        self.evidence: list[dict[str, Any]] = []
        for name in ("bodies", "keys", "tmp"):
            (root / name).mkdir(parents=True, exist_ok=True)
        # A cache that cannot be written must say so before a download, not after.
        try:
            descriptor, probe = tempfile.mkstemp(dir=root / "tmp", prefix="probe-")
        except OSError as error:
            raise OSError(
                f"transport cache is not writable: {root}: {error}"
            ) from error
        os.close(descriptor)
        os.unlink(probe)

    # -- paths -------------------------------------------------------------

    def body_path(self, sha256: str) -> Path:
        return self.root / "bodies" / sha256[:2] / sha256

    def key_path(self, digest: str) -> Path:
        return self.root / "keys" / f"{digest}.json"

    def git_path(self, key: BodyKey) -> Path:
        return self.root / "git" / f"{key.digest}.git"

    def take_evidence(self) -> list[dict[str, Any]]:
        taken, self.evidence = self.evidence, []
        return taken

    def _note(self, key: BodyKey, served: Served) -> None:
        self.evidence.append(
            {
                "kind": key.kind,
                "url": key.url,
                "served_from": served.served_from,
                "cache_key": served.cache_key,
                "sha256": served.sha256,
                "bytes": served.size,
            }
        )

    def _reject(self, key: BodyKey, reason: str) -> None:
        print(
            f"transport cache entry {key.digest} rejected ({reason}); "
            "fetching from the network",
            file=sys.stderr,
        )

    # -- read --------------------------------------------------------------

    def _entry(self, key: BodyKey) -> dict[str, Any] | None:
        import json

        path = self.key_path(key.digest)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            self._reject(key, f"unreadable key entry: {error}")
            return None
        if (
            not isinstance(entry, dict)
            or entry.get("key") != key.identity()
            or not isinstance(entry.get("sha256"), str)
            or type(entry.get("size")) is not int
        ):
            self._reject(key, "key entry does not match the request")
            return None
        if key.sha256 and entry["sha256"] != key.sha256.lower():
            self._reject(key, "entry body differs from the pinned SHA-256")
            return None
        return entry

    def lookup(
        self,
        key: BodyKey,
        target: Path | None,
        *,
        limit: int | None = None,
        observe: Callable[[], None] | None = None,
    ) -> Served | None:
        """Copy a verified body to `target` (or only verify it); None on any doubt."""
        entry = self._entry(key)
        if entry is None:
            return None
        sha256, size = entry["sha256"], entry["size"]
        if limit is not None and size > limit:
            self._reject(key, "body exceeds the declared byte cap")
            return None
        sums = _Sums(sha1=git_blob_hasher(size) if key.git_blob_oid else None)
        plain_sha1 = hashlib.sha1() if key.sha1 else None
        output = None
        try:
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open("xb")
            with self.body_path(sha256).open("rb") as body:
                if observe is not None:
                    observe()
                while chunk := body.read(_CHUNK):
                    sums.size += len(chunk)
                    if sums.size > size:
                        break
                    sums.sha256.update(chunk)
                    if sums.sha1 is not None:
                        sums.sha1.update(chunk)
                    if plain_sha1 is not None:
                        plain_sha1.update(chunk)
                    if output is not None:
                        output.write(chunk)
                        if observe is not None:
                            output.flush()
                            observe()
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            self._discard(target, output)
            self._reject(key, f"unreadable body: {error}")
            return None
        finally:
            if output is not None and not output.closed:
                output.close()
        problem = None
        if sums.size != size:
            problem = "body size differs from its entry"
        elif sums.sha256.hexdigest() != sha256:
            problem = "body hash differs from its entry"
        elif key.git_blob_oid and sums.sha1.hexdigest() != key.git_blob_oid.lower():
            problem = "body differs from the pinned Git blob id"
        elif key.sha1 and plain_sha1.hexdigest() != key.sha1.lower():
            problem = "body differs from the pinned SHA-1"
        if problem is not None:
            self._discard(target, None)
            self._reject(key, problem)
            return None
        served = Served("cache", key.digest, sha256, size, entry.get("meta"))
        self._note(key, served)
        return served

    @staticmethod
    def _discard(target: Path | None, output: Any) -> None:
        if output is not None and not output.closed:
            output.close()
        if target is not None:
            target.unlink(missing_ok=True)

    # -- write -------------------------------------------------------------

    def store(
        self, key: BodyKey, body: Path, *, meta: dict[str, Any] | None = None
    ) -> Served:
        """Publish an already verified download. Safe against concurrent writers."""
        descriptor, name = tempfile.mkstemp(dir=self.root / "tmp", prefix="body-")
        temporary = Path(name)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as output, body.open("rb") as source:
                while chunk := source.read(_CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha256 = digest.hexdigest()
            if key.sha256 and sha256 != key.sha256.lower():
                raise ValueError("refusing to cache a body that differs from its pin")
            temporary.chmod(0o444)
            destination = self.body_path(sha256)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            sync_directory(destination.parent)
            entry: dict[str, Any] = {
                "key": key.identity(),
                "sha256": sha256,
                "size": size,
            }
            if meta is not None:
                entry["meta"] = meta
            self._publish_key(key.digest, entry)
        finally:
            temporary.unlink(missing_ok=True)
        served = Served("network", key.digest, sha256, size, meta)
        self._note(key, served)
        return served

    def _publish_key(self, digest: str, entry: dict[str, Any]) -> None:
        descriptor, name = tempfile.mkstemp(dir=self.root / "tmp", prefix="key-")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(canonical_json(entry) + b"\n")
                output.flush()
                os.fsync(output.fileno())
            destination = self.key_path(digest)
            os.replace(temporary, destination)
            sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

    # -- Git ---------------------------------------------------------------

    @contextlib.contextmanager
    def git_lock(self, key: BodyKey) -> Iterator[None]:
        """Serialize processes that check, repair or fetch one repository."""
        import fcntl

        with (self.root / "tmp" / f"git-{key.digest}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def git_repository(self, key: BodyKey, *, fresh: bool = False) -> Path:
        """The bare repository for this key; `fresh` discards an unusable one."""
        path = self.git_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and path.exists():
            shutil.rmtree(path)
        if not path.exists():
            subprocess.run(
                ["git", "init", "--bare", "-q", str(path)],
                check=True,
                capture_output=True,
            )
        return path

    def git_has_commit(self, key: BodyKey) -> bool:
        """The pinned commit is present and every stored object re-hashes."""
        path = self.git_path(key)
        if not path.is_dir() or not key.revision:
            return False
        for command in (
            ["fsck", "--full", "--no-dangling", "--no-progress"],
            ["rev-list", "--objects", key.revision],
        ):
            done = subprocess.run(
                ["git", "--git-dir", str(path), *command],
                capture_output=True,
                check=False,
            )
            if done.returncode != 0:
                return False
        return True

    def note_git(self, key: BodyKey, *, served_from: str) -> None:
        self._publish_key(
            key.digest,
            {"key": key.identity(), "git_repository": f"git/{key.digest}.git"},
        )
        self.evidence.append(
            {
                "kind": key.kind,
                "url": key.url,
                "served_from": served_from,
                "cache_key": key.digest,
                "commit": key.revision,
            }
        )

    # -- inspection --------------------------------------------------------

    def entries(self, *, verify: bool = False) -> list[dict[str, Any]]:
        """Every key entry; with `verify`, re-hash each body and say why it fails."""
        import json

        listed = []
        for path in sorted((self.root / "keys").glob("*.json")):
            record: dict[str, Any] = {"cache_key": path.stem}
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                record.update(entry["key"])
                if "git_repository" in entry:
                    record["git_repository"] = entry["git_repository"]
                    problem = (
                        None
                        if not verify
                        else self._verify_git(path.stem, entry, entry["key"])
                    )
                else:
                    record["sha256"] = entry["sha256"]
                    record["bytes"] = entry["size"]
                    problem = (
                        None
                        if not verify
                        else self._verify_body(path.stem, entry, entry["key"])
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                problem = f"unreadable key entry: {error}"
            if verify:
                record["status"] = "ok" if problem is None else problem
            listed.append(record)
        return listed

    def _verify_body(self, digest: str, entry: dict[str, Any], key: Any) -> str | None:
        requested = BodyKey(**key)
        if requested.digest != digest:
            return "key entry does not match its name"
        hasher = hashlib.sha256()
        size = 0
        try:
            with self.body_path(entry["sha256"]).open("rb") as body:
                while chunk := body.read(_CHUNK):
                    size += len(chunk)
                    hasher.update(chunk)
        except OSError as error:
            return f"unreadable body: {error}"
        if size != entry["size"] or hasher.hexdigest() != entry["sha256"]:
            return "body does not match its hash"
        return None

    def _verify_git(self, digest: str, entry: dict[str, Any], key: Any) -> str | None:
        requested = BodyKey(**key)
        if requested.digest != digest:
            return "key entry does not match its name"
        return None if self.git_has_commit(requested) else "repository is incomplete"

    def prune(self) -> dict[str, list[str]]:
        """Remove what cannot be used: bad entries, unreferenced bodies, stale scratch.

        A valid entry is never removed; delete cold storage you no longer want
        yourself. Unreferenced bodies younger than an hour are kept, because a
        concurrent writer publishes the body before its key entry.
        """
        removed: dict[str, list[str]] = {"entries": [], "bodies": [], "scratch": []}
        referenced: set[str] = set()
        for record in self.entries(verify=True):
            if record["status"] == "ok":
                if "sha256" in record:
                    referenced.add(record["sha256"])
                continue
            self.key_path(record["cache_key"]).unlink(missing_ok=True)
            if "git_repository" in record:
                shutil.rmtree(self.root / record["git_repository"], ignore_errors=True)
            removed["entries"].append(record["cache_key"])
        now = time.time()
        for body in sorted((self.root / "bodies").glob("*/*")):
            if body.name not in referenced and now - body.stat().st_mtime > (
                _ORPHAN_GRACE_SECONDS
            ):
                body.unlink(missing_ok=True)
                removed["bodies"].append(body.name)
        for scratch in sorted((self.root / "tmp").iterdir()):
            if now - scratch.stat().st_mtime > _ORPHAN_GRACE_SECONDS:
                if scratch.is_dir():
                    shutil.rmtree(scratch, ignore_errors=True)
                else:
                    with contextlib.suppress(OSError):
                        scratch.unlink()
                removed["scratch"].append(scratch.name)
        return removed
