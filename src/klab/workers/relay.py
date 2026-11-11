"""Content-addressed, authenticated relay storage for hosted workers."""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from klab.canonical import canonical_json
from klab.runtime_env_subprocess import run_bounded
from klab.training.manifest import sha256_file
from klab.workdir import ensure_scratch_dir
from klab.workers.relay_models import (
    RelayArtifactIdentity,
    RelayBinding,
    RelayCommitDescriptor,
    RelayLocation,
    RelayProfile,
)
from klab.workers.transport import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    MAX_TOTAL_ATTACHMENT_BYTES,
)
from klab.workspace_preflight import check_storage, require_storage

_CHUNK = 1024 * 1024
_MAX_DESCRIPTOR_BYTES = 1024 * 1024


class RelayUnavailableError(RuntimeError):
    """The configured optional relay tool or credentials are unavailable."""


class RelayVerificationError(ValueError):
    """A relay object or commit did not satisfy its immutable identity."""


@dataclass(frozen=True)
class RelayCommit:
    digest: str
    descriptor: RelayCommitDescriptor


def _identity(value: Any) -> RelayArtifactIdentity:
    if isinstance(value, RelayArtifactIdentity):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif all(
        hasattr(value, field) for field in ("relative_path", "sha256", "size_bytes")
    ):
        value = {
            "relative_path": value.relative_path,
            "sha256": value.sha256,
            "size_bytes": value.size_bytes,
        }
    if not isinstance(value, dict):
        raise TypeError("artifact identity must be an object")
    return RelayArtifactIdentity.model_validate(value)


def _commit_items(
    descriptor: RelayCommitDescriptor,
) -> tuple[RelayArtifactIdentity, ...]:
    return (
        *descriptor.inventory,
        *descriptor.inventory_pages,
        *(page.artifact for page in descriptor.outbox_pages),
    )


def _safe_component(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise RelayVerificationError("unsafe relay path")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RelayVerificationError(f"unsafe relay source: {path}")
    return sha256_file(path)


def _private_key_path(path: Path) -> None:
    for parent in (path.parent, *path.parents):
        if parent.is_symlink():
            raise RelayVerificationError("relay key ancestor must not be a symlink")
    for private in (path.parent, path):
        status = private.stat()
        if status.st_uid != os.getuid() or status.st_mode & 0o077:
            raise RelayVerificationError(
                "relay key and attempt directory must be private and owned"
            )
    if path.is_symlink() or not path.is_file() or path.stat().st_size != 32:
        raise RelayVerificationError("relay key must be a private 32-byte regular file")


def create_attempt_key(controller_root: Path, attempt_id: str) -> Path:
    """Create once, then return an attempt-scoped 32-byte private HMAC key."""
    _safe_component(attempt_id)
    directory = controller_root / ".relay" / attempt_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(parent.is_symlink() for parent in (directory, *directory.parents)):
        raise RelayVerificationError("relay key directory contains a symlink")
    status = directory.stat()
    if status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise RelayVerificationError(
            "relay key attempt directory must be private and owned"
        )
    key_path = directory / "relay-key.bin"
    if key_path.exists():
        _private_key_path(key_path)
        return key_path
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(key_path, flags, 0o600)
    except FileExistsError:
        return create_attempt_key(controller_root, attempt_id)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(os.urandom(32))
        handle.flush()
        os.fsync(handle.fileno())
    _private_key_path(key_path)
    return key_path


def read_attempt_key(path: Path) -> bytes:
    _private_key_path(path)
    key = path.read_bytes()
    if len(key) != 32:
        raise RelayVerificationError("relay key must contain exactly 32 bytes")
    return key


def record_assignment(controller_root: Path, attempt_id: str, worker: Any) -> None:
    """Bind a private attempt key to the original physical worker, once."""
    key_path = create_attempt_key(controller_root, attempt_id)
    path = key_path.with_name("worker.json")
    raw = canonical_json(worker.model_dump(mode="json"))
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if (
            path.is_symlink()
            or path.stat().st_size > 1024 * 1024
            or path.read_bytes() != raw
        ):
            raise RelayVerificationError(
                "relay assignment cannot change during launch replay"
            )
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())


def assigned_worker(controller_root: Path, attempt_id: str):
    from .models import WorkerDefinition

    key_path = (
        controller_root / ".relay" / _safe_component(attempt_id) / "relay-key.bin"
    )
    _private_key_path(key_path)
    path = key_path.with_name("worker.json")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 1024 * 1024
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
    ):
        raise RelayVerificationError(
            "relay assignment must be a private owned bounded file"
        )
    return WorkerDefinition.model_validate_json(path.read_bytes())


def sign_descriptor(unsigned: dict[str, Any], key: bytes) -> str:
    if len(key) != 32:
        raise ValueError("relay HMAC key must contain exactly 32 bytes")
    return hmac.new(key, canonical_json(unsigned), hashlib.sha256).hexdigest()


class RelayStore:
    """One endpoint view over a namespaced immutable content-addressed store."""

    def __init__(
        self,
        location: RelayLocation | RelayBinding | RelayProfile,
        namespace: str | None = None,
        *,
        role: str = "controller",
        timeout_seconds: float | None = None,
        scratch_root: Path | None = None,
    ) -> None:
        if role not in {"controller", "worker"}:
            raise ValueError("relay role must be controller or worker")
        if isinstance(location, RelayProfile):
            endpoint = location.controller if role == "controller" else location.worker
            namespace = location.namespace if namespace is None else namespace
            timeout_seconds = (
                location.transfer_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
        elif isinstance(location, RelayBinding):
            endpoint = location.worker if role == "worker" else location.controller
            if endpoint is None:
                raise ValueError("worker-only relay binding has no controller location")
            namespace = location.namespace if namespace is None else namespace
            timeout_seconds = (
                location.transfer_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
        else:
            endpoint = location
        if namespace is None:
            raise ValueError("relay namespace is required")
        # Validate namespace through the strict profile contract.
        RelayProfile(namespace=namespace, controller=endpoint, worker=endpoint)
        self.location = endpoint
        self.namespace = namespace
        self.timeout_seconds = float(
            1800 if timeout_seconds is None else timeout_seconds
        )
        if not 0 < self.timeout_seconds <= 7200:
            raise ValueError("relay timeout must be finite and bounded")
        self.scratch_root = ensure_scratch_dir(scratch_root)
        require_storage(
            [check_storage(self.scratch_root, projected_bytes=0, projected_inodes=4)]
        )

    @property
    def root(self) -> Path:
        if self.location.kind != "file":
            raise RelayUnavailableError("rclone relay has no local root")
        return Path(self.location.root)

    def _relative(self, *parts: str) -> str:
        for part in parts:
            _safe_component(part)
        return "/".join((self.namespace, *parts))

    def _object_name(self, digest: str) -> str:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RelayVerificationError("invalid object digest")
        return self._relative("objects", "sha256", digest[:2], digest)

    def _local_path(self, relative: str) -> Path:
        root = self.root
        if root.exists() and root.is_symlink():
            raise RelayVerificationError("relay root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        current = root
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise RelayVerificationError("relay path contains a symlink")
        candidate = root / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _rclone(
        self,
        *arguments: str,
        stdout_path: Path | None = None,
        stdout_limit: int | None = None,
    ) -> bytes:
        executable = shutil.which("rclone")
        if executable is None:
            raise RelayUnavailableError(
                "RELAY_UNAVAILABLE: rclone is not installed; install rclone and run `rclone config` for the configured remote"
            )
        command = [
            executable,
            *arguments,
            "--timeout",
            f"{self.timeout_seconds}s",
            "--contimeout",
            f"{self.timeout_seconds}s",
        ]
        if self.location.config is not None:
            command.extend(("--config", str(self.location.config)))
        try:
            result = run_bounded(
                command,
                timeout=self.timeout_seconds,
                output_limit=1024 * 1024,
                stdout_path=stdout_path,
                stdout_limit=stdout_limit,
            )
            if result.returncode:
                raise subprocess.CalledProcessError(
                    result.returncode, command, stderr=result.stderr
                )
            return result.stdout
        except FileNotFoundError as error:  # pragma: no cover - race after which()
            raise RelayUnavailableError(
                "RELAY_UNAVAILABLE: rclone is not installed"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RelayUnavailableError(
                "RELAY_UNAVAILABLE: rclone transfer timed out"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", "replace").strip()
            raise RelayUnavailableError(
                f"RELAY_UNAVAILABLE: rclone transfer failed: {detail or 'check rclone config and remote permissions'}"
            ) from error

    def _remote(self, relative: str) -> str:
        return f"{self.location.root.rstrip('/')}/{relative}"

    def _download(self, relative: str, destination: Path, *, max_bytes: int) -> None:
        _safe_component(relative)
        require_storage(
            [
                check_storage(
                    destination.parent, projected_bytes=max_bytes, projected_inodes=1
                )
            ]
        )
        if self.location.kind == "file":
            source = self.root / relative
            if any(parent.is_symlink() for parent in (source, *source.parents)):
                raise RelayVerificationError("relay object path contains a symlink")
            if not source.is_file() or source.stat().st_size > max_bytes:
                raise RelayVerificationError(
                    "relay object is missing, unsafe or oversized"
                )
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                remaining = max_bytes
                while block := incoming.read(min(_CHUNK, remaining + 1)):
                    if len(block) > remaining:
                        raise RelayVerificationError(
                            "relay object exceeds declared byte bound"
                        )
                    outgoing.write(block)
                    remaining -= len(block)
            return
        try:
            self._rclone(
                "cat",
                self._remote(relative),
                stdout_path=destination,
                stdout_limit=max_bytes,
            )
        except ValueError as error:
            raise RelayVerificationError(
                "relay object exceeds declared byte bound"
            ) from error

    def _upload_immutable(self, source: Path, relative: str) -> None:
        if self.location.kind == "file":
            target = self._local_path(relative)
            if target.exists():
                return
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                expected = source.stat().st_size
                with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                    remaining = expected
                    while block := incoming.read(min(_CHUNK, remaining + 1)):
                        if len(block) > remaining:
                            raise RelayVerificationError(
                                "relay upload source grew during transfer"
                            )
                        outgoing.write(block)
                        remaining -= len(block)
                    if remaining:
                        raise RelayVerificationError(
                            "relay upload source shrank during transfer"
                        )
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass  # Existing immutable bytes are verified by the caller.
            finally:
                temporary.unlink(missing_ok=True)
            return
        self._rclone("copyto", str(source), self._remote(relative), "--immutable")

    def _verified_existing(
        self, relative: str, expected_sha256: str, expected_size: int
    ) -> bool:
        with tempfile.TemporaryDirectory(
            prefix="klab-relay-read-", dir=self.scratch_root
        ) as temporary:
            copy = Path(temporary) / "object"
            try:
                self._download(relative, copy, max_bytes=expected_size)
            except RelayUnavailableError, RelayVerificationError:
                return False
            return (
                copy.stat().st_size == expected_size
                and _sha256(copy) == expected_sha256
            )

    def put_verified(self, source: Path, identity: Any) -> None:
        item = _identity(identity)
        if item.size_bytes > MAX_ATTACHMENT_BYTES:
            raise RelayVerificationError("relay object exceeds item limit")
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != item.size_bytes
            or _sha256(source) != item.sha256
        ):
            raise RelayVerificationError(
                "source does not match declared artifact identity"
            )
        relative = self._object_name(item.sha256)
        if self._verified_existing(relative, item.sha256, item.size_bytes):
            return
        # A present but altered object is a conflict; never overwrite it.
        if self.location.kind == "file" and (self.root / relative).exists():
            raise RelayVerificationError(
                "existing relay object conflicts with declared digest"
            )
        self._upload_immutable(source, relative)
        if not self._verified_existing(relative, item.sha256, item.size_bytes):
            raise RelayVerificationError("relay object failed readback verification")

    def get_verified(self, identity: Any, destination: Path) -> None:
        item = _identity(identity)
        if item.size_bytes > MAX_ATTACHMENT_BYTES:
            raise RelayVerificationError("relay object exceeds item limit")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.relay")
        try:
            self._download(
                self._object_name(item.sha256), temporary, max_bytes=item.size_bytes
            )
            if (
                temporary.is_symlink()
                or temporary.stat().st_size != item.size_bytes
                or _sha256(temporary) != item.sha256
            ):
                raise RelayVerificationError("relay object identity mismatch")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _commit_relative(self, attempt_id: str, sequence: int, digest: str) -> str:
        _safe_component(attempt_id)
        if type(sequence) is not int or sequence < 0:
            raise ValueError("relay commit sequence must be nonnegative integer")
        return self._relative(
            "attempts", attempt_id, "commits", f"{sequence:020d}-{digest}.json"
        )

    def publish_commit(
        self, descriptor: RelayCommitDescriptor | dict[str, Any], *, key: bytes
    ) -> RelayCommit:
        commit = (
            descriptor
            if isinstance(descriptor, RelayCommitDescriptor)
            else RelayCommitDescriptor.model_validate(descriptor)
        )
        if not hmac.compare_digest(
            commit.hmac_sha256, sign_descriptor(commit.unsigned(), key)
        ):
            raise RelayVerificationError(
                "relay commit HMAC does not authenticate descriptor"
            )
        objects = _commit_items(commit)
        if (
            len(objects) > MAX_ATTACHMENTS
            or sum(item.size_bytes for item in objects) > MAX_TOTAL_ATTACHMENT_BYTES
        ):
            raise RelayVerificationError(
                "relay commit inventory exceeds transport limits"
            )
        for item in objects:
            if not self._verified_existing(
                self._object_name(item.sha256), item.sha256, item.size_bytes
            ):
                raise RelayVerificationError(
                    f"commit references missing object {item.sha256}"
                )
        digest = commit.descriptor_digest()
        relative = self._commit_relative(commit.attempt_id, commit.sequence, digest)
        payload = canonical_json(commit.model_dump(mode="json"))
        with tempfile.TemporaryDirectory(
            prefix="klab-relay-commit-", dir=self.scratch_root
        ) as temporary:
            source = Path(temporary) / "descriptor.json"
            source.write_bytes(payload)
            self._upload_immutable(source, relative)
        # Readback catches a competing/conflicting immutable descriptor.
        loaded = self.read_commit_path(relative, key=key)
        if loaded.digest != digest:
            raise RelayVerificationError(
                "published relay descriptor changed during publication"
            )
        return loaded

    def build_commit(self, *, key: bytes, **fields: Any) -> RelayCommitDescriptor:
        unsigned = {"relay_commit_version": 1, **fields}
        unsigned.pop("hmac_sha256", None)
        for field in ("inventory", "inventory_pages"):
            if field in unsigned:
                unsigned[field] = [
                    _identity(item).model_dump(mode="json") for item in unsigned[field]
                ]
        normalized = RelayCommitDescriptor.model_validate(
            {**unsigned, "hmac_sha256": "0" * 64}
        ).unsigned()
        # Descriptor metadata is bounded.  Spill a full inherited inventory to
        # immutable content-addressed pages before signing the final descriptor.
        if len(canonical_json(normalized)) > _MAX_DESCRIPTOR_BYTES:
            inventory = list(normalized["inventory"])
            pages: list[dict[str, Any]] = list(normalized.get("inventory_pages", []))
            with tempfile.TemporaryDirectory(
                prefix="klab-relay-inventory-write-", dir=self.scratch_root
            ) as temporary:
                for offset in range(0, len(inventory), 512):
                    payload = canonical_json(inventory[offset : offset + 512])
                    source = Path(temporary) / f"{offset:08d}.json"
                    source.write_bytes(payload)
                    item = RelayArtifactIdentity(
                        relative_path=f"inventory/{unsigned['attempt_id']}/{unsigned['sequence']}/{offset}.json",
                        sha256=hashlib.sha256(payload).hexdigest(),
                        size_bytes=len(payload),
                    )
                    self.put_verified(source, item)
                    pages.append(item.model_dump(mode="json"))
            normalized["inventory"] = []
            normalized["inventory_pages"] = pages
            normalized = RelayCommitDescriptor.model_validate(
                {**normalized, "hmac_sha256": "0" * 64}
            ).unsigned()
        return RelayCommitDescriptor.model_validate(
            {**normalized, "hmac_sha256": sign_descriptor(normalized, key)}
        )

    def read_commit_path(self, relative: str, *, key: bytes) -> RelayCommit:
        _safe_component(relative)
        with tempfile.TemporaryDirectory(
            prefix="klab-relay-descriptor-", dir=self.scratch_root
        ) as temporary:
            local = Path(temporary) / "descriptor.json"
            self._download(relative, local, max_bytes=_MAX_DESCRIPTOR_BYTES)
            if local.stat().st_size > _MAX_DESCRIPTOR_BYTES:
                raise RelayVerificationError("relay descriptor exceeds metadata limit")
            raw = local.read_bytes()
        try:
            import json

            parsed = json.loads(raw)
        except Exception as error:
            raise RelayVerificationError("relay descriptor is not JSON") from error
        if raw != canonical_json(parsed):
            raise RelayVerificationError("relay descriptor is not canonical JSON")
        try:
            descriptor = RelayCommitDescriptor.model_validate(parsed)
        except Exception as error:
            raise RelayVerificationError(
                "relay descriptor schema is invalid"
            ) from error
        if not hmac.compare_digest(
            descriptor.hmac_sha256, sign_descriptor(descriptor.unsigned(), key)
        ):
            raise RelayVerificationError("relay descriptor HMAC is invalid")
        digest = descriptor.descriptor_digest()
        expected = relative.rsplit("/", 1)[-1].removesuffix(".json").split("-", 1)[-1]
        if digest != expected:
            raise RelayVerificationError("relay descriptor filename digest mismatch")
        return RelayCommit(digest, descriptor)

    def list_commits(
        self, attempt_id: str, *, key: bytes, limit: int = 4096
    ) -> list[RelayCommit]:
        _safe_component(attempt_id)
        if not 1 <= limit <= 4096:
            raise ValueError("relay commit listing limit is out of bounds")
        prefix = self._relative("attempts", attempt_id, "commits")
        if self.location.kind == "file":
            directory = self.root / prefix
            if not directory.exists():
                return []
            if directory.is_symlink() or not directory.is_dir():
                raise RelayVerificationError("relay commit directory is unsafe")
            entries = []
            for entry in directory.iterdir():
                if len(entries) == limit:
                    raise RelayVerificationError("relay commit listing exceeds bound")
                entries.append(entry)
            entries.sort()
            if any(
                entry.is_symlink()
                or not entry.is_file()
                or not entry.name.endswith(".json")
                for entry in entries
            ):
                raise RelayVerificationError(
                    "relay commit directory contains an unsafe entry"
                )
            names = [entry.name for entry in entries]
        else:
            output = self._rclone("lsf", "--files-only", self._remote(prefix))
            if len(output) > limit * 256:
                raise RelayVerificationError(
                    "relay commit listing exceeds metadata bound"
                )
            names = [
                line for line in output.decode("utf-8", "strict").splitlines() if line
            ]
            if len(names) > limit or any(
                "/" in name or not name.endswith(".json") for name in names
            ):
                raise RelayVerificationError(
                    "relay commit listing is malformed or exceeds bound"
                )
        commits = [
            self.read_commit_path(f"{prefix}/{name}", key=key) for name in sorted(names)
        ]
        commits.sort(key=lambda item: item.descriptor.sequence)
        if len({item.descriptor.sequence for item in commits}) != len(commits):
            raise RelayVerificationError("relay commit chain has duplicate sequence")
        return commits

    def verified_chain(self, attempt_id: str, *, key: bytes) -> list[RelayCommit]:
        commits = self.list_commits(attempt_id, key=key)
        previous: str | None = None
        for expected, commit in enumerate(commits):
            if (
                commit.descriptor.sequence != expected
                or commit.descriptor.previous_commit_sha256 != previous
            ):
                raise RelayVerificationError("relay commit chain has a gap or fork")
            if commit.descriptor.attempt_id != attempt_id:
                raise RelayVerificationError("relay commit attempt substitution")
            self.verify_closure(commit)
            previous = commit.digest
        return commits

    def newest_verified_commit(
        self, attempt_id: str, *, key: bytes
    ) -> RelayCommit | None:
        chain = self.verified_chain(attempt_id, key=key)
        return chain[-1] if chain else None

    def flattened_inventory(
        self, commit: RelayCommit | RelayCommitDescriptor
    ) -> tuple[RelayArtifactIdentity, ...]:
        """Return the complete inventory, expanding authenticated bounded pages."""
        descriptor = commit.descriptor if isinstance(commit, RelayCommit) else commit
        items = list(descriptor.inventory)
        for page in descriptor.inventory_pages:
            with tempfile.TemporaryDirectory(
                prefix="klab-relay-inventory-", dir=self.scratch_root
            ) as temporary:
                local = Path(temporary) / "inventory.json"
                self.get_verified(page, local)
                try:
                    import json

                    value = json.loads(local.read_bytes())
                except Exception as error:
                    raise RelayVerificationError(
                        "relay inventory page is not JSON"
                    ) from error
            if not isinstance(value, list) or len(value) > 4096:
                raise RelayVerificationError(
                    "relay inventory page is malformed or unbounded"
                )
            try:
                items.extend(
                    RelayArtifactIdentity.model_validate(item) for item in value
                )
            except Exception as error:
                raise RelayVerificationError(
                    "relay inventory page item is invalid"
                ) from error
        total_items = (
            len(items) + len(descriptor.inventory_pages) + len(descriptor.outbox_pages)
        )
        if (
            total_items > MAX_ATTACHMENTS
            or sum(item.size_bytes for item in items) > MAX_TOTAL_ATTACHMENT_BYTES
        ):
            raise RelayVerificationError(
                "flattened relay inventory exceeds transport bounds"
            )
        if len({item.relative_path for item in items}) != len(items):
            raise RelayVerificationError(
                "flattened relay inventory has duplicate paths"
            )
        return tuple(items)

    def verify_closure(self, commit: RelayCommit | RelayCommitDescriptor) -> None:
        descriptor = commit.descriptor if isinstance(commit, RelayCommit) else commit
        objects = _commit_items(descriptor)
        if (
            len(objects) > MAX_ATTACHMENTS
            or sum(item.size_bytes for item in objects) > MAX_TOTAL_ATTACHMENT_BYTES
        ):
            raise RelayVerificationError(
                "relay recovery closure exceeds transfer bounds"
            )
        for item in objects:
            if not self._verified_existing(
                self._object_name(item.sha256), item.sha256, item.size_bytes
            ):
                raise RelayVerificationError(
                    f"relay recovery closure object is missing: {item.sha256}"
                )
        for item in self.flattened_inventory(descriptor):
            if not self._verified_existing(
                self._object_name(item.sha256), item.sha256, item.size_bytes
            ):
                raise RelayVerificationError(
                    f"relay inventory object is missing: {item.sha256}"
                )
        for page in descriptor.outbox_pages:
            with tempfile.TemporaryDirectory(
                prefix="klab-relay-page-", dir=self.scratch_root
            ) as temporary:
                local = Path(temporary) / "page.json"
                self.get_verified(page.artifact, local)
                try:
                    import json

                    payload = json.loads(local.read_bytes())
                except Exception as error:
                    raise RelayVerificationError(
                        "relay outbox page is not JSON"
                    ) from error
            if not isinstance(payload, dict):
                raise RelayVerificationError("relay outbox page is not an object")
            records = payload.get("records")
            if payload.get("origin_id") != page.origin_id or not isinstance(
                records, list
            ):
                raise RelayVerificationError(
                    "relay outbox page origin or records are invalid"
                )
            sequences = [
                record.get("sequence") for record in records if isinstance(record, dict)
            ]
            if len(sequences) != len(records) or sequences != list(
                range(page.first_sequence, page.last_sequence + 1)
            ):
                raise RelayVerificationError(
                    "relay outbox page sequence range is invalid"
                )

    def fetch_closure(
        self, commit: RelayCommit | RelayCommitDescriptor, destination: Path
    ) -> list[Path]:
        """Materialize a verified immutable closure under a caller-owned directory."""
        descriptor = commit.descriptor if isinstance(commit, RelayCommit) else commit
        self.verify_closure(descriptor)
        result: list[Path] = []
        items = (
            *self.flattened_inventory(descriptor),
            *descriptor.inventory_pages,
            *(page.artifact for page in descriptor.outbox_pages),
        )
        for index, item in enumerate(items):
            name = f"{index:04d}-{item.sha256}"
            target = destination / name
            self.get_verified(item, target)
            result.append(target)
        return result

    def roundtrip_challenge(self, source: Path, identity: Any) -> dict[str, object]:
        """Mutating preflight used by controller/worker binding configuration."""
        item = _identity(identity)
        self.put_verified(source, item)
        storage = check_storage(
            source.parent, projected_bytes=item.size_bytes, projected_inodes=1
        )
        require_storage([storage])
        return {
            "namespace": self.namespace,
            "sha256": item.sha256,
            "storage": storage.__dict__,
        }
