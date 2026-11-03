"""Owned, authenticated inspection work; unsigned or stale state is never reused.

The existing host-local ProofStore trust boundary applies: immutable files owned
by the service UID, not protection against a compromised same-UID owner. Only a
completed native inspection publishes a result. SQLite rows are disposable.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from sparselab.canonical import canonical_digest, canonical_json
from sparselab.corpus.bounded_metadata import current_metadata_store
from sparselab.result_publication import sync_directory
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import (
    _fingerprint,
    _trusted,
    file_binding,
    verification_options,
    writable_by_others,
)
from sparselab.verifier_authority import verifier_authority


def metadata_sha256(value):
    """Stream DiskRows through Python's iterator, never the list C encoder."""
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def capture(paths):
    result = {}
    for path in sorted({Path(p).absolute() for p in paths}):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("unsafe inspection dependency")
        members = [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]
        result[str(path)] = [
            [p.relative_to(path).as_posix() if p != path else ".", *_fingerprint(p)]
            for p in members
        ]
    return result


def unchanged(dependencies):
    if capture(dependencies) != dependencies:
        raise ValueError("inspection inputs changed during execution")


class InspectionCache:
    """Reuse successful native results, including completed sources after a stop."""

    def __init__(self, root, *, proof_store=None):
        self.root = Path(root).resolve(strict=True)
        self.proofs = (
            proof_store
            if proof_store is not None
            else verification_options(self.root)["proof_store"]
        )
        self.directory = self.root / "cache" / "admission-inspection"
        self.writable = self.proofs is not None
        self.unavailable_reason = None if self.writable else "untrusted_work_root"
        self.authority = {}
        if self.proofs is None:
            return
        try:
            self.authority = {
                "corpus": verifier_authority("source_snapshot", 1),
                "inspection": {
                    name: sha256_file(Path(__file__).with_name(name))
                    for name in (
                        "admission_inspection.py",
                        "admission_inspection_cache.py",
                        "rights.py",
                        "jsonl_records.py",
                        "structure_cleaning.py",
                    )
                },
            }
        except OSError, ValueError, TypeError, KeyError, SyntaxError:
            self.proofs = None
            self._unavailable("verifier_authority_unavailable")
            return
        try:
            key_available = self.proofs.key_available(create=True)
        except OSError, ValueError:
            key_available = False
        if not key_available:
            self.proofs = None
            self._unavailable("verification_key_unavailable")
            return
        if (self.proofs.directory.exists() or self.proofs.directory.is_symlink()) and (
            not self.proofs.directory.is_dir()
            or not _trusted(self.root, self.proofs.directory, private=True)
        ):
            self.proofs = None
            self._unavailable("verification_store_unavailable")

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, value):
        from sparselab.corpus.retained_render_adoption import check_render_directory

        value = Path(value)
        check_render_directory(self.root, value)
        self._directory = value

    def _unavailable(self, reason):
        self.writable = False
        self.unavailable_reason = reason

    def _cleanup(self, temporary):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                self._unavailable("cache_cleanup_unavailable")

    def entry_path(self, kind, request):
        """Locate a cache entry without loading or authenticating its contents."""
        key = canonical_digest([kind, request])
        return self.directory / f"{kind}-{key}.json"

    def entry_binding(self, path, raw, request, dependencies):
        """Bind captured entry bytes and dependencies for proof verification."""
        # Declarations may be edited by the owner's group. Their exact captured
        # fingerprints belong in the signed closure, rather than pretending
        # they are immutable artifact members in ProofStore's trust walk.
        immutable, mutable = {}, {}
        for name, members in dependencies.items():
            root = Path(name)
            paths = [root if row[0] == "." else root / row[0] for row in members]
            if any(row[3] & 0o020 for row in members) and any(
                writable_by_others(member.lstat()) for member in paths
            ):
                for member in paths:
                    info = member.lstat()
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                        or info.st_gid != os.getgid()
                        or info.st_nlink != 1
                        or info.st_mode & 0o002
                        or not _trusted(self.root, self.root, private=True)
                        or any(
                            parent.stat().st_uid != os.getuid()
                            or parent.stat().st_mode & 0o002
                            for parent in member.parents
                            if parent == self.root or self.root in parent.parents
                        )
                    ):
                        raise ValueError("unsafe mutable inspection dependency")
                mutable[name] = members
            else:
                immutable[name] = members
        binding = file_binding(
            path,
            hashlib.sha256(raw).hexdigest(),
            closure={
                "request": request,
                "authority": self.authority,
                "mutable_dependencies": mutable,
            },
        )
        binding["dependencies"] = [
            [path, rows] for path, rows in sorted(immutable.items())
        ]
        return binding

    def _native_phase(self, request, payload, dependencies, binding, *, limit):
        """Read an owned atomic native completion, never manufacture a proof.

        SplitStages already checks its chain and recomputes the request identity
        (including input fingerprints). This covers native commits whose old
        optional proof could not publish because declarations were group writable.
        """
        if set(request) != {"request", "completion"} or not self.authority.get(
            "streaming_producer"
        ):
            return None
        root = self.directory.parent
        if root.name != request["request"] or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in request.values()
        ):
            return None
        events = self.proofs.diagnostics()["events"]
        reason = events[-1]["reason"] if events else None
        receipt = self.proofs.receipt_path(binding)
        if (
            receipt.exists()
            or receipt.is_symlink()
            or reason not in ("no_receipt", "untrusted_store")
        ):
            return None
        matches = list(root.glob("completion-*-" + request["completion"] + ".json"))
        if len(matches) != 1:
            return None
        completion = matches[0]
        if (
            not _trusted(self.root, completion)
            or completion.stat().st_nlink != 1
            or completion.stat().st_size > limit
        ):
            return None
        event = json.loads(completion.read_bytes())
        body = event["record"]
        if (
            event["sha256"] != request["completion"]
            or canonical_digest(body) != event["sha256"]
            or body["request"] != request["request"]
            or body["result"] != payload
            or not payload.get("files")
        ):
            return None
        for item in payload["files"]:
            path = Path(item["path"])
            if not path.is_relative_to(root) or not _trusted(self.root, path):
                return None
            info = path.stat()
            if info.st_nlink != 1 or item["stamp"] != [
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ]:
                return None
        unchanged(dependencies)
        self.proofs.record_event(binding, "owned_native_completion")
        return payload, dependencies

    def load(self, kind, request, *, limit):
        from sparselab.corpus.retained_render_adoption import (
            adopt_render,
            requires_adoption,
        )

        explicit = kind == "split-phase" and requires_adoption(
            self.root, self.directory
        )
        if self.proofs is None:
            if explicit:
                raise RuntimeError(
                    "retained render proof store unavailable; refusing replay"
                )
            return None
        path = self.entry_path(kind, request)
        try:
            if not self.proofs.trusted(path) or path.stat().st_size > limit:
                if explicit:
                    raise RuntimeError(
                        "retained render cache unavailable; refusing replay"
                    )
                return None
            raw = path.read_bytes()
            value = json.loads(raw)
            if set(value) != {"dependencies", "payload"}:
                raise ValueError("invalid inspection cache fields")
            paths = value["dependencies"]
            if not isinstance(paths, list) or any(
                not isinstance(p, str)
                or not Path(p).is_absolute()
                or ".." in Path(p).parts
                or not Path(p).is_relative_to(self.root)
                for p in paths
            ):
                raise ValueError("invalid inspection cache dependencies")
            dependencies = capture(paths)
            binding = self.entry_binding(path, raw, request, dependencies)
            if explicit:
                return adopt_render(
                    self, request, value["payload"], dependencies, binding
                )
            if not self.proofs.lookup(binding):
                if kind == "split-phase":
                    completed = self._native_phase(
                        request, value["payload"], dependencies, binding, limit=limit
                    )
                    if completed is not None:
                        return completed
                return None
            unchanged(dependencies)
            return value["payload"], dependencies
        except OSError, ValueError, TypeError, KeyError, json.JSONDecodeError:
            if explicit:
                raise RuntimeError(
                    "retained render metadata changed; refusing replay"
                ) from None
            return None

    def publish(self, kind, request, payload, dependencies, *, limit):
        unchanged(dependencies)
        if self.proofs is None or not self.writable:
            return
        # A proof cannot cover a dependency outside its registered trust root.
        dependencies = {
            path: rows
            for path, rows in dependencies.items()
            if Path(path).is_relative_to(self.root)
        }
        raw = (
            canonical_json({"dependencies": sorted(dependencies), "payload": payload})
            + b"\n"
        )
        if len(raw) > limit:
            return  # Cache size is optional; the independently checked result stands.
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            self._unavailable("untrusted_cache_directory")
            return
        path = self.entry_path(kind, request)
        from sparselab.data.verification import verify_file

        temporary = None
        try:
            self.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.proofs.trusted(self.directory.parent):
                self._unavailable("untrusted_cache_directory")
                return
            self.directory.mkdir(exist_ok=True, mode=0o700)
            if not _trusted(self.root, self.directory, private=True):
                self._unavailable("untrusted_cache_directory")
                return
            with tempfile.NamedTemporaryFile(
                prefix=".inspection-", dir=self.directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._unavailable("cache_write_unavailable")
            self._cleanup(temporary)
            return
        except BaseException:
            self._cleanup(temporary)
            raise
        try:
            unchanged(dependencies)
            try:
                os.replace(temporary, path)
                sync_directory(self.directory)
                binding = self.entry_binding(path, raw, request, dependencies)
                verify_file(
                    path,
                    expected_sha256=binding["sha256"],
                    proof_store=self.proofs,
                    verification_mode="verified_reuse",
                    binding=binding,
                )
            except OSError, ValueError:
                # An optional cache cannot veto the live native-validated result.
                # Keep reads enabled so other authenticated sources can still hit.
                self._unavailable("cache_write_unavailable")
        finally:
            self._cleanup(temporary)


class SourceIndex:
    """One source's offsets and lengths, with bounded SQLite cache and no text."""

    def __init__(self):
        self.connection = current_metadata_store().connection
        self.connection.execute("DROP TABLE IF EXISTS temp.inspection_candidates")
        self.connection.execute(
            "CREATE TEMP TABLE inspection_candidates(ordinal INTEGER PRIMARY KEY, "
            "path TEXT,digest TEXT,row_index INTEGER,physical_line INTEGER,"
            "byte_offset INTEGER,decision TEXT,issues INTEGER,length INTEGER)"
        )
        self.rows = 0

    def add(self, row, length):
        self.connection.execute(
            "INSERT INTO inspection_candidates VALUES(?,?,?,?,?,?,?,?,?)",
            (
                row["_ordinal"],
                row["path"],
                row["digest"],
                row["row_index"],
                row["physical_line"],
                row["_offset"],
                row["decision"],
                bool(row["issues"]),
                length,
            ),
        )
        self.rows += 1

    def median(self):
        count = self.connection.execute(
            "SELECT count(*) FROM inspection_candidates WHERE decision='qualify'"
        ).fetchone()[0]
        if not count:
            return 0, 0
        self.connection.execute(
            "CREATE INDEX inspection_lengths ON inspection_candidates(decision,length)"
        )
        values = self.connection.execute(
            "SELECT length FROM inspection_candidates WHERE decision='qualify' "
            "ORDER BY length LIMIT ? OFFSET ?",
            (2 if count % 2 == 0 else 1, (count - 1) // 2),
        ).fetchall()
        return sum(value[0] for value in values) * (2 if count % 2 else 1), count

    def __iter__(self):
        cursor = self.connection.execute(
            "SELECT * FROM inspection_candidates ORDER BY ordinal"
        )
        try:
            for row in cursor:
                yield dict(
                    zip(
                        (
                            "_ordinal",
                            "path",
                            "digest",
                            "row_index",
                            "physical_line",
                            "_offset",
                            "decision",
                            "issues",
                            "retained_length_bytes",
                        ),
                        row,
                        strict=True,
                    )
                )
        finally:
            cursor.close()
