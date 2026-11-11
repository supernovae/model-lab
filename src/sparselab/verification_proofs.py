"""Host-local authenticated operational receipts, never artifact identity authority.

Trust assumes immutable artifacts managed by the service UID. This does not
protect against a compromised same-UID owner. Every miss requires cold checking.
"""

from __future__ import annotations

import functools
import grp
import hashlib
import hmac
import json
import os
import pwd
import secrets
import stat
from collections import Counter
from pathlib import Path
from typing import Literal

from sparselab.canonical import canonical_digest, canonical_json
from sparselab.verifier_authority import (
    AUTHORITY_SCHEMA,
    supports_verifier,
    verifier_authority,
)

VERIFIER_VERSION = AUTHORITY_SCHEMA
VerificationMode = Literal["cold", "verified_reuse"]


def validate_mode(mode: VerificationMode) -> None:
    if mode not in {"cold", "verified_reuse"}:
        raise ValueError(f"unknown verification mode: {mode}")


def _json_native(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    return value


def _fingerprint(path: Path) -> list[int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"nonregular verification path: {path}")
    return [
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]


def writable_by_others(info: os.stat_result) -> bool:
    """Whether anyone but the owner may write this entry.

    Group write is the owner's alone under a user-private group (Ubuntu's
    default umask 002), so it is refused only for shared groups.
    """
    if info.st_mode & 0o002:
        return True
    return bool(info.st_mode & 0o020) and not _private_group(info.st_gid)


@functools.cache
def _private_group(gid: int) -> bool:
    try:
        group = grp.getgrgid(gid)
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return False
    return gid == os.getgid() and group.gr_name == user and set(group.gr_mem) <= {user}


def _trusted(root: Path, path: Path, *, private: bool = False) -> bool:
    """Check selected owned root and children, not system-owned ancestors."""
    try:
        root, path = root.absolute(), path.absolute()
        if root == Path(root.anchor) or not path.is_relative_to(root):
            return False
        # Even a root reached through a link is not a registered trust domain.
        if any(parent.is_symlink() for parent in (root, *root.parents)):
            return False
        entries = [root]
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            entries.append(current)
        for entry in entries:
            info = entry.lstat()
            if info.st_uid != os.getuid() or writable_by_others(info):
                return False
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                return False
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                return False
        if private:
            info = path.lstat()
            expected = 0o700 if stat.S_ISDIR(info.st_mode) else 0o600
            if stat.S_IMODE(info.st_mode) != expected:
                return False
        return True
    except OSError, ValueError:
        return False


def file_binding(
    path: Path,
    sha256: str,
    *,
    kind: str = "file",
    version: int = 1,
    identifier: str | None = None,
    closure: object = None,
) -> dict[str, object]:
    path = path.absolute()
    return {
        "kind": kind,
        "version": version,
        "identifier": identifier or path.name,
        "sha256": sha256,
        "path": str(path),
        "members": [[".", *_fingerprint(path)]],
        "manifest_inventory_sha256": hashlib.sha256(
            canonical_json(closure)
        ).hexdigest(),
        "upstream_identities": _json_native(closure),
        "dependencies": [],
        "success": True,
    }


def artifact_binding(
    key: tuple[object, ...], identity: dict[str, object]
) -> dict[str, object]:
    manifests = json.loads(key[8])
    return {
        **identity,
        "state": key[7],
        "members": _json_native(key[5]),
        "dependencies": _json_native(key[6]),
        "manifest_inventory_sha256": hashlib.sha256(
            canonical_json(manifests)
        ).hexdigest(),
        "upstream_identities": manifests["upstream"],
        "success": True,
    }


def _current_reason(binding: dict[str, object]) -> str | None:
    try:
        path = Path(str(binding["path"]))
        members = binding["members"]
        if not isinstance(members, list):
            return "changed_fingerprint"
        actual = [
            [p.relative_to(path).as_posix() if p != path else ".", *_fingerprint(p)]
            for p in ([path] if path.is_file() else [path, *sorted(path.rglob("*"))])
        ]
        if actual != members:
            return "changed_fingerprint"
        for dependency, fingerprints in binding["dependencies"]:
            root = Path(dependency)
            actual = [
                [p.relative_to(root).as_posix() if p != root else ".", *_fingerprint(p)]
                for p in (
                    [root] if root.is_file() else [root, *sorted(root.rglob("*"))]
                )
            ]
            if actual != fingerprints:
                return "changed_dependency"
        return None
    except OSError, KeyError, TypeError, ValueError:
        return "changed_fingerprint"


def _current(binding: dict[str, object]) -> bool:
    return _current_reason(binding) is None


class ProofStore:
    """One signed receipt index under an explicitly selected task-owned work root."""

    def __init__(self, root: Path, *, read_only: bool = False) -> None:
        self.root = Path(root).absolute()
        self.directory = self.root / "cache" / "verification-v1"
        configured = os.environ.get("XDG_CONFIG_HOME", "")
        self.config_root = (
            Path(configured)
            if configured and Path(configured).is_absolute()
            else Path.home() / ".config"
        )
        self.key_path = self.config_root / "sparselab" / "verification-key-v1"
        self.read_only = read_only
        self.hits = self.misses = self.recorded = 0
        self._events: list[dict[str, object]] = []

    def trusted(self, path: Path) -> bool:
        return _trusted(self.root, path)

    def diagnostics(self) -> dict[str, object]:
        """Operational verification outcomes, separate from artifact identity.

        Hit/miss counters count lookups; reasons also include cold requests and
        receipt-publication failures.
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "recorded": self.recorded,
            "reasons": dict(Counter(event["reason"] for event in self._events)),
            "events": list(self._events),
        }

    def record_event(
        self,
        binding: dict[str, object],
        reason: str,
        *,
        authority: dict[str, object] | None = None,
        bytes_hashed: int | None = None,
        bytes_avoided: int | None = None,
    ) -> None:
        """Append operational diagnostics without granting verification authority."""
        authority_error = None
        if authority is None:
            try:
                authority = verifier_authority(
                    str(binding["kind"]), int(binding["version"])
                )
            except (KeyError, TypeError, ValueError, OSError, SyntaxError) as error:
                authority_error = f"{type(error).__name__}: {error}"
                reason = "unknown"
        event = {
            "path": binding.get("path"),
            "kind": binding.get("kind"),
            "reason": reason,
            "verifier_authority": authority,
            "authority_error": authority_error,
            "bytes_hashed": bytes_hashed,
            "bytes_avoided": bytes_avoided,
        }
        self._events.append(event)

    def cold(
        self, binding: dict[str, object], *, bytes_hashed: int | None = None
    ) -> None:
        """Record an explicitly cold request without treating it as a lookup miss."""
        self.record_event(binding, "explicit_cold", bytes_hashed=bytes_hashed)

    def hashed(self, binding: dict[str, object], size: int) -> None:
        """Account for actual file-verifier cold hashing after a failed lookup."""
        for event in reversed(self._events):
            if event["path"] == binding.get("path"):
                event["bytes_hashed"] = size
                return

    def _safe_binding(self, binding: dict[str, object]) -> bool:
        """Authenticate the whole inventory, with one trust walk per directory.

        The directory set lives only for this invocation. A second lookup check
        and publication each perform a fresh independent walk.
        """
        try:
            if not _trusted(self.root, self.root):
                return False
            checked_dirs = {self.root}
            owner = os.getuid()

            def inventory(root: Path, rows: object) -> bool:
                if not root.is_absolute() or ".." in root.parts:
                    return False
                relative = root.relative_to(self.root)
                parent = self.root
                for part in relative.parts[:-1]:
                    parent = parent / part
                    if parent not in checked_dirs:
                        info = parent.lstat()
                        if (
                            not stat.S_ISDIR(info.st_mode)
                            or info.st_uid != owner
                            or writable_by_others(info)
                        ):
                            return False
                        checked_dirs.add(parent)
                if not isinstance(rows, list) or not rows:
                    return False
                # DFS in Path's lexical order; compare every node against its
                # exact signed fingerprint without materializing another tree.
                pending: list[tuple[Path | os.DirEntry[str], str]] = [(root, ".")]
                index = 0
                while pending:
                    entry, label = pending.pop()
                    info = (
                        entry.lstat()
                        if isinstance(entry, Path)
                        else entry.stat(follow_symlinks=False)
                    )
                    directory = stat.S_ISDIR(info.st_mode)
                    if (
                        (not directory and not stat.S_ISREG(info.st_mode))
                        or info.st_uid != owner
                        or writable_by_others(info)
                        or (not directory and info.st_nlink != 1)
                        or index >= len(rows)
                        or rows[index]
                        != [
                            label,
                            info.st_dev,
                            info.st_ino,
                            info.st_mode,
                            info.st_size,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                        ]
                    ):
                        return False
                    index += 1
                    if directory:
                        directory_path = (
                            entry if isinstance(entry, Path) else Path(entry.path)
                        )
                        checked_dirs.add(directory_path)
                        with os.scandir(directory_path) as children:
                            ordered = sorted(children, key=lambda child: child.name)
                        for child in reversed(ordered):
                            child_label = (
                                child.name if label == "." else f"{label}/{child.name}"
                            )
                            pending.append((child, child_label))
                return index == len(rows)

            root = Path(str(binding["path"]))
            if not inventory(root, binding["members"]):
                return False
            for dependency, rows in binding["dependencies"]:
                if not inventory(Path(dependency), rows):
                    return False
            return True
        except OSError, KeyError, TypeError, ValueError:
            return False

    def _payload(self, binding: dict[str, object]) -> dict[str, object]:
        return {
            "verification_schema": VERIFIER_VERSION,
            "verifier_authority": verifier_authority(
                str(binding["kind"]), int(binding["version"])
            ),
            "binding": binding,
        }

    def receipt_path(self, binding: dict[str, object]) -> Path:
        """Locate a receipt; its existence alone does not authenticate evidence."""
        locator = {
            key: binding.get(key)
            for key in (
                "kind",
                "version",
                "identifier",
                "sha256",
                "path",
            )
        }
        return self.directory / (canonical_digest(locator) + ".json")

    def _key(self, *, create: bool) -> bytes | None:
        if create:
            if any(
                parent.is_symlink()
                for parent in (self.config_root, *self.config_root.parents)
            ):
                return None
            self.config_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not _trusted(self.config_root, self.config_root):
                return None
            self.key_path.parent.mkdir(mode=0o700, exist_ok=True)
            if not _trusted(self.config_root, self.key_path.parent, private=True):
                return None
            if not self.key_path.exists():
                try:
                    fd = os.open(
                        self.key_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(secrets.token_bytes(32))
                        stream.flush()
                        os.fsync(stream.fileno())
                    _sync(self.key_path.parent)
        if not _trusted(self.config_root, self.key_path, private=True):
            return None
        with self.key_path.open("rb") as stream:
            key = stream.read(33)
        return key if len(key) == 32 else None

    def key_available(self, *, create: bool = False) -> bool:
        """Check key access without exposing key bytes; read-only by default.

        ``create=True`` uses the existing publication trust checks and may create
        the private key and its directories. A readable existing key does not
        imply that those stricter publication checks will succeed. This checks
        only key access, not receipt-directory availability.
        """
        return self._key(create=create) is not None

    def migration_signature(self, observation: dict[str, object]) -> str:
        """Authenticate a relocation observation, not a native execution proof.

        As with this store's other receipts, the service UID is trusted. This
        does not protect against that UID fabricating its own observations.
        """
        if observation.get("kind") not in {
            "source-release-migration-v1",
            "target-release-migration-v1",
            "source-release-migration-progress-v1",
            "target-release-migration-progress-v1",
            "source-mixture-stage-migration-v1",
            "target-mixture-stage-migration-v1",
            "target-mixture-stage-migration-progress-v1",
        }:
            raise ValueError("invalid migration observation kind")
        key = self._key(create=False)
        if key is None:
            raise ValueError("existing private verification key unavailable")
        return hmac.new(
            key,
            b"sparselab-release-migration-v1\0" + canonical_json(observation),
            hashlib.sha256,
        ).hexdigest()

    def lookup(self, binding: dict[str, object]) -> bool:
        authority: dict[str, object] | None = None
        reason = "unknown"
        hit = False
        try:
            receipt = self.receipt_path(binding)
            if not supports_verifier(binding.get("kind")):
                reason = "unknown"
            elif not self.trusted(Path(str(binding["path"]))):
                reason = "untrusted_store"
            elif not self._safe_binding(binding):
                reason = _current_reason(binding) or "untrusted_store"
            elif not _trusted(self.root, self.directory, private=True):
                reason = "untrusted_store" if self.directory.exists() else "no_receipt"
            elif not receipt.exists():
                reason = "no_receipt"
            elif not _trusted(self.root, receipt, private=True):
                reason = "untrusted_store"
            else:
                key = self._key(create=False)
                if key is None:
                    reason = "untrusted_store"
                elif receipt.stat().st_size > 16 * 1024 * 1024:
                    reason = "corrupt_or_invalid_receipt"
                else:
                    raw = json.loads(receipt.read_bytes())
                    payload = self._payload(binding)
                    authority = payload["verifier_authority"]
                    assert isinstance(authority, dict)
                    old = raw.get("proof") if isinstance(raw, dict) else None
                    authentic = (
                        isinstance(old, dict)
                        and isinstance(raw.get("hmac"), str)
                        and hmac.compare_digest(
                            raw["hmac"],
                            hmac.new(
                                key, canonical_json(old), hashlib.sha256
                            ).hexdigest(),
                        )
                    )
                    if not authentic:
                        reason = "corrupt_or_invalid_receipt"
                    elif old != payload:
                        previous = old.get("binding")
                        if previous == binding:
                            reason = "verifier_authority_changed"
                        elif isinstance(previous, dict) and all(
                            previous.get(field) == binding.get(field)
                            for field in (
                                "kind",
                                "version",
                                "identifier",
                                "sha256",
                                "path",
                            )
                        ):
                            if previous.get("manifest_inventory_sha256") != binding.get(
                                "manifest_inventory_sha256"
                            ):
                                reason = "changed_manifest_binding"
                            elif previous.get("members") != binding.get("members"):
                                reason = "changed_fingerprint"
                            elif previous.get("dependencies") != binding.get(
                                "dependencies"
                            ):
                                reason = "changed_dependency"
                            else:
                                reason = "corrupt_or_invalid_receipt"
                        else:
                            reason = "corrupt_or_invalid_receipt"
                    elif not self._safe_binding(binding):
                        reason = "changed_fingerprint"
                    else:
                        hit = True
                        reason = "hit"
        except json.JSONDecodeError, UnicodeDecodeError:
            reason = "corrupt_or_invalid_receipt"
        except OSError, ValueError, TypeError, KeyError, SyntaxError:
            reason = "unknown"
        self.hits += int(hit)
        self.misses += int(not hit)
        self.record_event(
            binding,
            reason,
            authority=authority,
            bytes_avoided=self._binding_size(binding) if hit else None,
        )
        return hit

    def lookup_recorded_execution(self, binding: dict[str, object]) -> bool:
        """Authenticate historical evidence without claiming current verification.

        Retained execution readers must separately bind the recorded producer
        and exact operation. This never publishes or upgrades a proof.
        """
        try:
            receipt = self.receipt_path(binding)
            if (
                not supports_verifier(binding.get("kind"))
                or not self._safe_binding(binding)
                or not _trusted(self.root, receipt, private=True)
                or receipt.stat().st_size > 16 * 1024 * 1024
            ):
                return False
            key = self._key(create=False)
            if key is None:
                return False
            raw = json.loads(receipt.read_bytes())
            old = raw.get("proof")
            return (
                isinstance(old, dict)
                and old.get("verification_schema") == VERIFIER_VERSION
                and old.get("binding") == binding
                and isinstance(old.get("verifier_authority"), dict)
                and isinstance(raw.get("hmac"), str)
                and hmac.compare_digest(
                    raw["hmac"],
                    hmac.new(key, canonical_json(old), hashlib.sha256).hexdigest(),
                )
                and self._safe_binding(binding)
            )
        except OSError, ValueError, TypeError, KeyError:
            return False

    @staticmethod
    def _binding_size(binding: dict[str, object]) -> int | None:
        members = binding.get("members")
        if (
            binding.get("kind")
            in {"file", "prepared_array", "run_artifact", "ingested_run_artifact"}
            and isinstance(members, list)
            and len(members) == 1
            and members[0][0] == "."
            and stat.S_ISREG(int(members[0][3]))
        ):
            return int(members[0][4])
        return None

    def record(self, binding: dict[str, object], evidence: object) -> None:
        """Only a successful, exact cold verifier's live seal can publish a proof."""
        from sparselab.data.verification import _owned_file
        from sparselab.experiments.artifacts import (
            _MEMO_SEAL,
            _MINTED_ARTIFACTS,
            _VerifiedArtifact,
        )

        owned = False
        if _owned_file(evidence):
            owned = (
                evidence.cold_verified
                and binding.get("path") == str(evidence.path)
                and binding.get("sha256") == evidence.sha256
                and binding.get("members") == [[".", *_fingerprint(evidence.path)]]
            )
        elif isinstance(evidence, _VerifiedArtifact):
            owned = (
                evidence in _MINTED_ARTIFACTS
                and evidence._seal is _MEMO_SEAL
                and evidence.issuer_pid == os.getpid()
                and evidence.cold_verified
                and artifact_binding(evidence.key, dict(evidence.identity)) == binding
            )
        if not owned or binding.get("success") is not True:
            raise TypeError(
                "verification proof requires exact process-owned cold evidence"
            )
        if self.read_only or not supports_verifier(binding.get("kind")):
            return
        if not self._safe_binding(binding):
            return
        try:
            payload = self._payload(binding)
        except (OSError, ValueError, KeyError, SyntaxError) as error:
            for event in reversed(self._events):
                if event["path"] == binding.get("path"):
                    event["reason"] = "unknown"
                    event["authority_error"] = f"{type(error).__name__}: {error}"
                    break
            return
        try:
            key = self._key(create=True)
            if key is None:
                self.record_event(binding, "key_publication_unavailable")
                return
            self.directory.parent.mkdir(mode=0o700, exist_ok=True)
            if not _trusted(self.root, self.directory.parent):
                return
            self.directory.mkdir(mode=0o700, exist_ok=True)
            if not _trusted(self.root, self.directory, private=True):
                return
            # Authority was computed before any publication or key creation.
            receipt = self.receipt_path(binding)
            temporary = self.directory / (".proof-" + secrets.token_hex(16))
            try:
                with temporary.open("xb") as stream:
                    os.chmod(temporary, 0o600)
                    stream.write(
                        canonical_json(
                            {
                                "proof": payload,
                                "hmac": hmac.new(
                                    key, canonical_json(payload), hashlib.sha256
                                ).hexdigest(),
                            }
                        )
                        + b"\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if not self._safe_binding(binding):
                    raise ValueError("artifact changed during proof publication")
                os.replace(temporary, receipt)
                _sync(self.directory)
                self.recorded += 1
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            # Operational cache unavailable: independently cold-verified result stands.
            return


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verification_options(
    root: Path, *, cold: bool = False, read_only: bool = False
) -> dict[str, object]:
    if cold or not _trusted(Path(root), Path(root)):
        return {"proof_store": None, "verification_mode": "cold"}
    return {
        "proof_store": ProofStore(root, read_only=read_only),
        "verification_mode": "verified_reuse",
    }
