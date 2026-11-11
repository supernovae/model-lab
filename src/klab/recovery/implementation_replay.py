"""Explicit, isolated replay of a committed historical Corpus Forge implementation.

The source tree is read from Git objects, never from the checkout. Operational
receipts are not inputs to corpus scientific identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from klab.recovery.provenance import declaration_paths, repository_root

_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _package_source(module_name: str) -> Path:
    """Locate an installed Model Lab module's source without importing it.

    Verifier authority refuses non-constant dynamic imports anywhere in a
    verifier's closure, and this module only needs the bytes on disk.
    """
    relative = module_name.removeprefix("klab.").split(".")
    return (
        Path(__file__).resolve().parents[1].joinpath(*relative).with_suffix(".py")
    ).resolve(strict=True)


_COMPONENTS = (
    "acquisition",
    "pipeline",
    "project",
    "provenance",
    "rights",
    "large_build",
    "release",
    "export",
)
_DIRECT = frozenset(
    ("acquisition", "pipeline", "project", "provenance", "rights", "large_build")
)
_FORMAT = "klab-implementation-replay-v1"
_ANCESTRY_FORMAT = "klab-implementation-replay-v2"
_SOURCE_FORMAT = "klab-materialized-source-v1"


class ReplayFailure(ValueError):
    """A failed attempt whose operational receipt was successfully published."""

    def __init__(self, reason: str, receipt_path: Path, record_sha256: str) -> None:
        super().__init__(reason)
        self.receipt_path = receipt_path
        self.record_sha256 = record_sha256


def _reject_symlink_ancestors(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"UNSAFE_REPLAY_ROOT: symlinked destination {component}")


def _source_target(repo: Path, work_root: Path, commit: str) -> Path:
    work = Path(work_root).expanduser().absolute()
    _reject_symlink_ancestors(work)
    work = work.resolve()
    if work.is_relative_to(repo) or repo.is_relative_to(work):
        raise ValueError(
            "UNSAFE_REPLAY_ROOT: replay state must be external to the checkout"
        )
    target = work / "replay" / "source" / commit
    _reject_symlink_ancestors(target)
    return target


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        capture_output=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"MISSING_IMPLEMENTATION: Git object unavailable: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _repository_identity(root: Path, commit: str, project: str) -> dict[str, Any]:
    remote = subprocess.run(
        ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "root": str(root),
        "commit": commit,
        "project": project,
        "tree": _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
        "object_format": _git(root, "rev-parse", "--show-object-format")
        .decode()
        .strip(),
        "origin_url": remote.stdout.strip() if remote.returncode == 0 else None,
    }


def _repository(source: Path, commit: str) -> tuple[Path, str]:
    root = repository_root(source)
    if root is None or not _COMMIT.fullmatch(commit):
        raise ValueError(
            "MISSING_IMPLEMENTATION: exact full commit and local Git repository required"
        )
    resolved = (
        _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    )
    if resolved != commit:
        raise ValueError(
            "MISSING_IMPLEMENTATION: commit must identify an exact commit object"
        )
    return root, resolved


def _relative(root: Path, path: Path) -> str:
    absolute = Path(path).absolute()
    if not absolute.is_relative_to(root):
        raise ValueError(f"SOURCE_COMMIT_MISMATCH: path outside repository: {path}")
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise ValueError(f"SOURCE_COMMIT_MISMATCH: symlinked source: {ancestor}")
        if ancestor == root:
            break
    return absolute.relative_to(root).as_posix()


def _tree(root: Path, commit: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for row in _git(root, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not row:
            continue
        header, raw = row.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split(" ")
        name = raw.decode("utf-8", errors="strict")
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(ord(c) < 32 for c in name)
        ):
            raise ValueError(
                f"UNSAFE_SOURCE_TREE: invalid Git entry {name!r} mode={mode}"
            )
        if name in entries:
            raise ValueError(f"UNSAFE_SOURCE_TREE: duplicate path {name}")
        entries[name] = mode, oid
    if not entries:
        raise ValueError("MISSING_IMPLEMENTATION: empty historical source tree")
    return entries


def _blob_digest(data: bytes, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def _source_inventory(
    root: Path, entries: dict[str, tuple[str, str]], algorithm: str
) -> dict[str, dict[str, str]]:
    inventory: dict[str, dict[str, str]] = {}
    found: set[str] = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            directory = Path(base) / name
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ValueError(f"SOURCE_TAMPERED: unsafe directory {directory}")
        for name in files:
            path = Path(base) / name
            relative = path.relative_to(root).as_posix()
            if relative not in entries or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"SOURCE_TAMPERED: unexpected file {relative}")
            mode, oid = entries[relative]
            data = path.read_bytes()
            if _blob_digest(data, algorithm) != oid or bool(
                path.stat().st_mode & 0o111
            ) != (mode == "100755"):
                raise ValueError(
                    f"SOURCE_TAMPERED: historical bytes or mode differ: {relative}"
                )
            inventory[relative] = {
                "mode": mode,
                "object": oid,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            found.add(relative)
    if found != entries.keys():
        raise ValueError(
            f"SOURCE_TAMPERED: missing historical files: {sorted(entries.keys() - found)}"
        )
    return inventory


def implementation_preflight(
    source: Path,
    commit: str,
    project: Path,
    *,
    historical_source_root: Path | None = None,
) -> dict[str, Any]:
    """Report historical/current component byte identity without creating files."""
    try:
        root, commit = _repository(source, commit)
        tree = _tree(root, commit)
    except ValueError as error:
        if str(error).startswith("MISSING_IMPLEMENTATION"):
            return {
                "status": "MISSING_IMPLEMENTATION",
                "components": {},
                "repository": None,
                "reason": str(error),
            }
        raise
    from klab.corpus.project import load_project

    recipe = load_project(project)
    authored = {Path(project).absolute()}
    authored.update(
        project.parent / relative
        for relative in (
            *recipe.config.sources,
            *recipe.config.transforms,
            recipe.config.splits,
            recipe.config.release,
        )
    )
    project_relative = _relative(historical_source_root or root, project)
    for path in declaration_paths(project, "corpus"):
        relative = _relative(historical_source_root or root, path)
        if not path.is_file() and path not in authored:
            continue  # Missing local payloads remain external prerequisites, not dirty declarations.
        if relative not in tree or not path.is_file():
            raise ValueError(
                f"SOURCE_COMMIT_MISMATCH: missing historical project input {relative}"
            )
        if path.read_bytes() != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(
                f"SOURCE_COMMIT_MISMATCH: dirty or changed project input {relative}"
            )
    execution_root = Path(__file__).resolve().parents[3]
    large_build = (
        recipe.release.schema_version == 3
        and len(recipe.transforms) == 1
        and recipe.transforms[0].kind == "lm_text"
        and recipe.release.lm.selected
        and not recipe.release.chat.selected
        and all(item.schema_version == 3 for item in recipe.sources)
    )
    direct = _DIRECT - {"rights", "large_build"}
    if recipe.release.schema_version in (2, 3):
        direct = direct | {"rights"}
    if large_build:
        direct = direct | {"large_build"}
    components: dict[str, dict[str, Any]] = {}
    for name in _COMPONENTS:
        relative = f"src/klab/corpus/{name}.py"
        if relative not in tree:
            return {
                "status": "MISSING_IMPLEMENTATION",
                "components": components,
                "repository": {"root": str(root), "commit": commit},
                "reason": f"historical source missing {relative}",
            }
        historical = hashlib.sha256(
            _git(root, "show", f"{commit}:{relative}")
        ).hexdigest()
        current_path = _package_source(f"klab.corpus.{name}")
        current = hashlib.sha256(current_path.read_bytes()).hexdigest()
        stage = (
            "acquisition"
            if name == "acquisition"
            else "freeze"
            if name == "release"
            else "export"
            if name == "export"
            else "build"
        )
        components[name] = {
            "role": name,
            "affected_stage": stage,
            "source_commit": commit,
            "path": relative,
            "pinned_sha256": historical,
            "current_sha256": current,
            "direct_identity_hash": name in direct,
            "status": "MATCH"
            if current == historical
            else "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
        }
    for name, relative in (
        ("training_manifest", "src/klab/training/manifest.py"),
        ("project_config", "src/klab/config/models.py"),
        ("pyproject", "pyproject.toml"),
        ("uv_lock", "uv.lock"),
    ):
        if relative not in tree:
            return {
                "status": "MISSING_IMPLEMENTATION",
                "components": components,
                "repository": {"root": str(root), "commit": commit},
                "reason": f"historical source missing {relative}",
            }
        historical = hashlib.sha256(
            _git(root, "show", f"{commit}:{relative}")
        ).hexdigest()
        if name in {"pyproject", "uv_lock"}:
            current_path = execution_root / relative
        else:
            module_name = (
                "klab.training.manifest"
                if name == "training_manifest"
                else "klab.config.models"
            )
            current_path = _package_source(module_name)
        current = (
            hashlib.sha256(current_path.read_bytes()).hexdigest()
            if current_path.is_file()
            else None
        )
        stage = "dependencies" if name in {"pyproject", "uv_lock"} else "build"
        components[name] = {
            "role": name,
            "affected_stage": stage,
            "source_commit": commit,
            "path": relative,
            "pinned_sha256": historical,
            "current_sha256": current,
            "direct_identity_hash": False,
            "status": "MATCH"
            if current == historical
            else "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
        }
    return {
        "status": "MATCH"
        if all(row["status"] == "MATCH" for row in components.values())
        else "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
        "components": components,
        "repository": _repository_identity(root, commit, project_relative),
    }


def verify_materialized_source(
    source_root: Path, commit: str | None = None
) -> dict[str, Any]:
    """Reauthenticate every committed source byte and mode against Git objects."""
    source_root = Path(source_root).absolute()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("SOURCE_TAMPERED: symlinked or missing source tree")
    if any(parent.is_symlink() for parent in source_root.parents):
        raise ValueError("SOURCE_TAMPERED: symlinked source ancestor")
    receipt_path = (
        source_root.parent.parent / "source-manifests" / f"{source_root.name}.json"
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("SOURCE_TAMPERED: missing or symlinked source manifest")
    record = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        record.get("format") != _SOURCE_FORMAT
        or set(record)
        != {
            "format",
            "commit",
            "repository",
            "tree",
            "object_format",
            "files",
            "inventory_sha256",
        }
        or record["commit"] != source_root.name
        or (commit is not None and record["commit"] != commit)
    ):
        raise ValueError("SOURCE_TAMPERED: invalid source manifest")
    root, exact = _repository(Path(record["repository"]), record["commit"])
    entries = _tree(root, exact)
    if _git(root, "rev-parse", f"{exact}^{{tree}}").decode().strip() != record["tree"]:
        raise ValueError("SOURCE_TAMPERED: tree mismatch")
    algorithm = _git(root, "rev-parse", "--show-object-format").decode().strip()
    if algorithm != record["object_format"]:
        raise ValueError("SOURCE_TAMPERED: Git object format mismatch")
    files = _source_inventory(source_root, entries, algorithm)
    if (
        record["files"] != files
        or record["inventory_sha256"]
        != hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("SOURCE_TAMPERED: source manifest differs from Git inventory")
    return record


def materialize_source(source: Path, commit: str, work_root: Path) -> dict[str, Any]:
    """Atomically publish a read-only full tree from the exact local commit."""
    root, commit = _repository(source, commit)
    entries = _tree(root, commit)
    target = _source_target(root, work_root, commit)
    receipt_path = target.parent.parent / "source-manifests" / f"{commit}.json"
    if target.exists() or receipt_path.exists():
        if not target.is_dir() or not receipt_path.is_file():
            raise ValueError("SOURCE_TAMPERED: incomplete existing source publication")
        return {
            "source_root": str(target),
            "manifest_path": str(receipt_path),
            "manifest": verify_materialized_source(target, commit),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    algorithm = _git(root, "rev-parse", "--show-object-format").decode().strip()
    with tempfile.TemporaryDirectory(prefix=".source-", dir=target.parent) as temporary:
        staging = Path(temporary)
        archive = subprocess.Popen(
            [
                "git",
                "-c",
                "tar.umask=0022",
                "-C",
                str(root),
                "archive",
                "--format=tar",
                commit,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        assert archive.stdout is not None
        seen: set[str] = set()
        try:
            with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
                for member in stream:
                    name = member.name.rstrip("/")
                    if member.isdir():
                        if name and (
                            name.startswith("/")
                            or any(part in {"", ".", ".."} for part in name.split("/"))
                        ):
                            raise ValueError(f"UNSAFE_SOURCE_TREE: directory {name!r}")
                        continue
                    if (
                        name not in entries
                        or name in seen
                        or not member.isfile()
                        or member.mode & 0o7000
                    ):
                        raise ValueError(f"UNSAFE_SOURCE_TREE: archive entry {name!r}")
                    mode, oid = entries[name]
                    if bool(member.mode & 0o111) != (mode == "100755"):
                        raise ValueError(f"UNSAFE_SOURCE_TREE: archive mode {name!r}")
                    stream_file = stream.extractfile(member)
                    if stream_file is None:
                        raise ValueError(
                            f"UNSAFE_SOURCE_TREE: unreadable archive entry {name!r}"
                        )
                    data = stream_file.read()
                    if _blob_digest(data, algorithm) != oid:
                        raise ValueError(f"SOURCE_TAMPERED: Git blob mismatch {name}")
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    destination.chmod(0o555 if mode == "100755" else 0o444)
                    seen.add(name)
        finally:
            archive.stdout.close()
            archive.wait()
            if archive.returncode:
                raise ValueError(
                    "MISSING_IMPLEMENTATION: failed to archive pinned Git tree"
                )
        if seen != entries.keys():
            raise ValueError("SOURCE_TAMPERED: incomplete historical archive")
        files = _source_inventory(staging, entries, algorithm)
        record = {
            "format": _SOURCE_FORMAT,
            "commit": commit,
            "repository": str(root),
            "tree": _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
            "object_format": algorithm,
            "files": files,
            "inventory_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for directory, _, _ in os.walk(staging):
            if Path(directory) != staging:
                Path(directory).chmod(0o555)
        # macOS requires write permission on a directory being renamed. Seal
        # its root after publication, before publishing the verification receipt.
        staging.rename(target)
        target.chmod(0o555)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
    receipt_path.chmod(0o444)
    return {
        "source_root": str(target),
        "manifest_path": str(receipt_path),
        "manifest": verify_materialized_source(target, commit),
    }


def _log_identity(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(max(0, stream.tell() - 4096))
        excerpt = stream.read(4096).decode(errors="replace")
    return digest.hexdigest(), excerpt


def _run_logged(
    command: list[str],
    phase: str,
    logs: Path,
    env: dict[str, str],
    cwd: Path | None = None,
) -> dict[str, Any]:
    logs.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stdout_path = logs / f"{token}.{phase}.stdout"
    stderr_path = logs / f"{token}.{phase}.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(
            command, stdout=stdout, stderr=stderr, env=env, cwd=cwd, check=False
        )
    stdout_hash, stdout_excerpt = _log_identity(stdout_path)
    stderr_hash, stderr_excerpt = _log_identity(stderr_path)
    return {
        "phase": phase,
        "argv": command,
        "returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": stdout_hash,
        "stderr_sha256": stderr_hash,
        "stdout": stdout_excerpt,
        "stderr": stderr_excerpt,
    }


def _verify_producer_provenance(
    source: dict[str, Any],
    lock: dict[str, Any],
    build: dict[str, Any],
    release: dict[str, Any] | None,
    inherited_adapter_sha256: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Bind already byte-verified artifacts to the actual materialized producer."""
    files = source["files"]
    acquisition_sha = files["src/klab/corpus/acquisition.py"]["sha256"]
    snapshots = []
    for source_id, entry in lock["sources"].items():
        if entry["snapshot_sha256"] is None:
            continue
        manifest = json.loads(
            (Path(entry["snapshot_path"]) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["adapter"]["module_sha256"] != (inherited_adapter_sha256 or {}).get(
            source_id, acquisition_sha
        ):
            raise ValueError(
                f"IMPLEMENTATION_PROVENANCE_MISMATCH: snapshot {source_id}"
            )
        snapshots.append({"source_id": source_id, "sha256": entry["snapshot_sha256"]})
    identity = build["identity"]
    fields = {
        "implementation_sha256": "pipeline",
        "schema_implementation_sha256": "project",
        "provenance_implementation_sha256": "provenance",
    }
    if identity["release"]["schema_version"] in (2, 3):
        fields["rights_implementation_sha256"] = "rights"
    transforms = identity["transforms"]
    if (
        identity["release"]["schema_version"] == 3
        and len(transforms) == 1
        and transforms[0]["kind"] == "lm_text"
        and identity["release"]["lm"]["selected"]
        and not identity["release"]["chat"]["selected"]
    ):
        fields["large_builder_implementation_sha256"] = "large_build"
    for field, name in fields.items():
        if identity.get(field) != files[f"src/klab/corpus/{name}.py"]["sha256"]:
            raise ValueError(f"IMPLEMENTATION_PROVENANCE_MISMATCH: build {name}")
    if release is not None and (
        release["build_id"] != build["build_id"]
        or release["build_identity"] != identity
    ):
        raise ValueError("IMPLEMENTATION_PROVENANCE_MISMATCH: release/build binding")
    return snapshots


def replay_corpus(
    source: Path,
    commit: str,
    project: Path,
    work_root: Path,
    *,
    allow_network: bool,
    expected_release_sha256: str | None,
    expected_build_sha256: str | None = None,
    corpus_work_root: Path | None = None,
    phase: Literal["build", "release"] = "release",
    parent_receipt: Path | None = None,
    inherited_source_ids: tuple[str, ...] = (),
    expected_unchanged_snapshots: dict[str, str] | None = None,
    expected_changed_snapshots: dict[str, str] | None = None,
    use_historical_project: bool = False,
    expected_project_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute historical producers in a dedicated uv environment and verify outputs."""
    if (
        corpus_work_root is not None
        or phase != "release"
        or parent_receipt is not None
        or inherited_source_ids
        or expected_unchanged_snapshots is not None
        or expected_changed_snapshots is not None
        or use_historical_project
        or expected_project_sha256 is not None
    ):
        return _replay_ancestry(
            source,
            commit,
            project,
            work_root,
            allow_network=allow_network,
            expected_release_sha256=expected_release_sha256,
            expected_build_sha256=expected_build_sha256,
            corpus_work_root=corpus_work_root,
            phase=phase,
            parent_receipt=parent_receipt,
            inherited_source_ids=inherited_source_ids,
            expected_unchanged_snapshots=expected_unchanged_snapshots,
            expected_changed_snapshots=expected_changed_snapshots,
            use_historical_project=use_historical_project,
            expected_project_sha256=expected_project_sha256,
        )
    root = Path(work_root).expanduser().absolute()
    _reject_symlink_ancestors(root)
    root = root.resolve()
    repository, _ = _repository(source, commit)
    _source_target(repository, root, commit)
    receipts = root / "replay" / "receipts"
    _reject_symlink_ancestors(receipts)
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{uuid.uuid4().hex}.json"
    receipt: dict[str, Any] = {
        "format": _FORMAT,
        "status": "FAILED",
        "stage": "preflight",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "requested_source": str(Path(source).absolute()),
        "project": str(Path(project).absolute()),
        "expected_build_sha256": expected_build_sha256,
        "expected_release_sha256": expected_release_sha256,
        "allow_network": allow_network,
        "operations": [],
        "scientific_identity": {},
        "operational_identity": {
            "work_root": str(root),
            "orchestrator": str(Path(__file__).resolve()),
            "orchestrator_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    }
    failure: Exception | None = None
    try:
        for value in (expected_build_sha256, expected_release_sha256):
            if value is not None and not _HEX.fullmatch(value):
                raise ValueError(
                    "INVALID_EXPECTED_DIGEST: expected SHA-256 must be lowercase full hex"
                )
        preflight = implementation_preflight(source, commit, project)
        receipt["preflight"] = preflight
        receipt["repository"] = preflight["repository"]
        if preflight["status"] == "MISSING_IMPLEMENTATION":
            raise ValueError(f"MISSING_IMPLEMENTATION: {preflight['reason']}")
        receipt["stage"] = "materialization"
        source_record = materialize_source(source, commit, root)
        historical = Path(source_record["source_root"])
        verify_materialized_source(historical, commit)
        receipt["source"] = {
            "tree": source_record["manifest"]["tree"],
            "inventory_sha256": source_record["manifest"]["inventory_sha256"],
            "manifest_path": source_record["manifest_path"],
        }
        relative = preflight["repository"]["project"]
        destination = historical / relative
        if not destination.is_file():
            raise ValueError(
                "SOURCE_COMMIT_MISMATCH: missing historical corpus declaration"
            )
        receipt["stage"] = "dependencies"
        environment = root / "replay" / "env" / commit
        _reject_symlink_ancestors(environment)
        environment.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in (
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "UV_ACTIVE",
            "PYTHONHOME",
            "PYTHONPATH",
            "UV_PROJECT_ENVIRONMENT",
        ):
            env.pop(key, None)
        env.update(
            {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "UV_NO_ENV_FILE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "UV_NO_PROGRESS": "1",
            }
        )
        uv = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        )
        if uv.returncode:
            raise ValueError(f"ENVIRONMENT_UNAVAILABLE: uv unavailable: {uv.stderr}")
        receipt["operational_identity"].update(
            {
                "uv": uv.stdout.strip(),
                "python": sys.version,
                "environment": str(environment),
                "lock_sha256": hashlib.sha256(
                    (historical / "uv.lock").read_bytes()
                ).hexdigest(),
            }
        )
        command = [
            "uv",
            "sync",
            "--project",
            str(historical),
            "--locked",
            "--no-install-project",
            "--no-dev",
            "--no-default-groups",
            "--python",
            sys.executable,
        ]
        historical_project = tomllib.loads(
            (historical / "pyproject.toml").read_text(encoding="utf-8")
        )
        if "cpu" in historical_project.get("project", {}).get(
            "optional-dependencies", {}
        ):
            command.extend(["--extra", "cpu"])
        if not allow_network:
            command.append("--offline")
        install = _run_logged(command, "dependencies", root / "replay" / "logs", env)
        receipt["operations"].append(install)
        if install["returncode"]:
            raise ValueError(
                "ENVIRONMENT_UNAVAILABLE: historical locked dependencies could not be installed"
            )
        verify_materialized_source(historical, commit)
        python = environment / "bin" / "python"
        if not python.is_file():
            raise ValueError("ENVIRONMENT_UNAVAILABLE: dedicated uv python missing")
        dependencies = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if dependencies.returncode:
            raise ValueError(
                f"ENVIRONMENT_UNAVAILABLE: unable to inventory dedicated dependencies: {dependencies.stderr}"
            )
        interpreter = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--no-sync",
                "--python",
                str(python),
                "python",
                "-I",
                "-c",
                "import sys; print(sys.version)",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if interpreter.returncode:
            raise ValueError(
                f"ENVIRONMENT_UNAVAILABLE: dedicated interpreter failed: {interpreter.stderr}"
            )
        receipt["operational_identity"]["dependencies"] = (
            dependencies.stdout.splitlines()
        )
        receipt["operational_identity"]["environment_python"] = (
            interpreter.stdout.strip()
        )
        work = root / "replay" / "work" / commit
        work.mkdir(parents=True, exist_ok=True)
        receipt["stage"] = "historical_producers"
        request = {
            "version": 1,
            "source_root": str(historical),
            "project": str(destination),
            "work_root": str(work),
            "allow_network": allow_network,
            "expected_release_sha256": expected_release_sha256,
            "expected_build_sha256": expected_build_sha256,
            "components": preflight["components"],
        }
        with tempfile.TemporaryDirectory(
            prefix=".invocation-", dir=receipts
        ) as temporary:
            request_path = Path(temporary) / "request.json"
            output_path = Path(temporary) / "response.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True), encoding="utf-8"
            )
            worker = Path(__file__).with_name("_implementation_worker.py")
            run = _run_logged(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--no-sync",
                    "--python",
                    str(python),
                    "python",
                    "-I",
                    str(worker),
                    str(request_path),
                    str(output_path),
                ],
                "historical_producers",
                root / "replay" / "logs",
                {**env, "UV_OFFLINE": "1" if not allow_network else "0"},
                work,
            )
            receipt["operations"].append(run)
            result = (
                json.loads(output_path.read_text(encoding="utf-8"))
                if output_path.is_file()
                else None
            )
        if (
            type(result) is not dict
            or type(result.get("version")) is not int
            or result["version"] != 1
            or result.get("status") not in {"MATCH", "FAILED"}
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: invalid historical worker result"
            )
        if result["status"] == "MATCH":
            required = {
                "version",
                "status",
                "path",
                "build_path",
                "build_sha256",
                "release_sha256",
                "snapshots",
                "modules",
            }
            if (
                set(result) != required
                or any(type(result[key]) is not str for key in ("path", "build_path"))
                or any(
                    type(result[key]) is not str or not _HEX.fullmatch(result[key])
                    for key in ("build_sha256", "release_sha256")
                )
                or type(result["snapshots"]) is not list
                or type(result["modules"]) is not dict
            ):
                raise ValueError(
                    "HISTORICAL_EXECUTION_FAILED: malformed successful worker result"
                )
        elif (
            set(result)
            != {"version", "status", "phase", "artifacts", "error", "error_type"}
            or any(
                type(result[key]) is not str for key in ("phase", "error", "error_type")
            )
            or type(result["artifacts"]) is not dict
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: malformed failed worker result"
            )
        receipt["worker"] = result
        if run["returncode"] or result["status"] != "MATCH":
            raise ValueError(
                result.get("error", "HISTORICAL_EXECUTION_FAILED: worker failed")
            )
        receipt["stage"] = "independent_verification"
        from klab.corpus.acquisition import verify_acquisition
        from klab.corpus.project import load_project
        from klab.corpus.release import verify_build, verify_release

        corpus = load_project(Path(project))
        lock = verify_acquisition(corpus, work)
        built = Path(result["build_path"])
        frozen = Path(result["path"])
        if (
            any(
                path.is_symlink()
                for path in (built, *built.parents, frozen, *frozen.parents)
            )
            or not built.resolve(strict=True).is_relative_to(work.resolve(strict=True))
            or not frozen.resolve(strict=True).is_relative_to(work.resolve(strict=True))
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: artifact escaped isolated root"
            )
        build = verify_build(built)
        release = verify_release(frozen)
        if (
            build["build_id"] != result["build_sha256"]
            or release["release_id"] != result["release_sha256"]
            or (expected_build_sha256 and build["build_id"] != expected_build_sha256)
            or (
                expected_release_sha256
                and release["release_id"] != expected_release_sha256
            )
        ):
            raise ValueError(
                "EXPECTED_DIGEST_MISMATCH: independent artifact identity mismatch"
            )
        verified_source = verify_materialized_source(historical, commit)
        snapshots = _verify_producer_provenance(verified_source, lock, build, release)
        if snapshots != result["snapshots"]:
            raise ValueError(
                "IMPLEMENTATION_PROVENANCE_MISMATCH: worker snapshot inventory"
            )
        receipt["scientific_identity"] = {
            "build_sha256": build["build_id"],
            "release_sha256": release["release_id"],
            "snapshots": snapshots,
        }
        receipt["stage"] = "complete"
        receipt["status"] = "MATCH"
        receipt["path"] = str(frozen)
    except Exception as error:  # noqa: BLE001 - publish before raising ReplayFailure
        if (
            isinstance(receipt.get("worker"), dict)
            and receipt["worker"].get("status") == "FAILED"
        ):
            receipt["stage"] = receipt["worker"].get("phase", "historical_producers")
        receipt["error"] = str(error)
        failure = error
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        receipt["record_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with receipt_path.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
    if failure is not None:
        raise ReplayFailure(
            str(failure), receipt_path, receipt["record_sha256"]
        ) from failure
    return {
        "path": receipt["path"],
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def verify_replay_receipt(path: Path) -> dict[str, Any]:
    """Validate operational evidence and independently authenticate published artifacts."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("INVALID_REPLAY_RECEIPT: missing or symlinked receipt")
    with path.open("rb") as stream:
        if json.load(stream).get("format") == _ANCESTRY_FORMAT:
            return _verify_ancestry_receipt(path, set())
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format",
        "status",
        "stage",
        "started_at_utc",
        "finished_at_utc",
        "source_commit",
        "requested_source",
        "project",
        "allow_network",
        "expected_build_sha256",
        "expected_release_sha256",
        "operations",
        "scientific_identity",
        "operational_identity",
        "record_sha256",
    }
    if (
        not isinstance(record, dict)
        or not required.issubset(record)
        or record["format"] != _FORMAT
        or record["status"] not in {"MATCH", "FAILED"}
        or type(record["allow_network"]) is not bool
        or not isinstance(record["operations"], list)
        or not isinstance(record["stage"], str)
    ):
        raise ValueError("INVALID_REPLAY_RECEIPT: invalid versioned record")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if (
        record["record_sha256"]
        != hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("INVALID_REPLAY_RECEIPT: record digest mismatch")
    if record.get("repository"):
        repo = record["repository"]
        root, commit = _repository(Path(repo["root"]), record["source_commit"])
        if (
            commit != repo["commit"]
            or _repository_identity(root, commit, repo["project"])["tree"]
            != repo["tree"]
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: repository tree mismatch")
    for operation in record["operations"]:
        for channel in ("stdout", "stderr"):
            stream = Path(operation[f"{channel}_path"])
            if (
                stream.is_symlink()
                or not stream.is_file()
                or _log_identity(stream)[0] != operation[f"{channel}_sha256"]
            ):
                raise ValueError(f"INVALID_REPLAY_RECEIPT: {channel} log mismatch")
    if record.get("source") and isinstance(record["source"], dict):
        manifest = Path(record["source"]["manifest_path"])
        source_root = manifest.parent.parent / "source" / record["source_commit"]
        verified = verify_materialized_source(source_root, record["source_commit"])
        if verified["inventory_sha256"] != record["source"]["inventory_sha256"]:
            raise ValueError("INVALID_REPLAY_RECEIPT: source digest mismatch")
    if record["status"] == "MATCH":
        if not record.get("source") or not record.get("repository"):
            raise ValueError(
                "INVALID_REPLAY_RECEIPT: missing historical producer identity"
            )
        from klab.corpus.acquisition import verify_acquisition
        from klab.corpus.project import load_project
        from klab.corpus.release import verify_build, verify_release

        worker = record["worker"]
        if worker["status"] != "MATCH" or record["stage"] != "complete":
            raise ValueError("INVALID_REPLAY_RECEIPT: contradictory success")
        work = (
            Path(record["operational_identity"]["work_root"])
            / "replay"
            / "work"
            / record["source_commit"]
        )
        lock = verify_acquisition(load_project(Path(record["project"])), work)
        for key in ("path", "build_path"):
            artifact = Path(worker[key])
            if not artifact.resolve(strict=True).is_relative_to(
                work.resolve(strict=True)
            ) or any(item.is_symlink() for item in (artifact, *artifact.parents)):
                raise ValueError(
                    "INVALID_REPLAY_RECEIPT: artifact outside isolated work root"
                )
        build = verify_build(Path(worker["build_path"]))
        release = verify_release(Path(worker["path"]))
        snapshots = _verify_producer_provenance(verified, lock, build, release)
        if (
            snapshots != record["scientific_identity"]["snapshots"]
            or snapshots != worker["snapshots"]
            or build["build_id"] != record["scientific_identity"]["build_sha256"]
            or release["release_id"] != record["scientific_identity"]["release_sha256"]
            or build["build_id"] != worker["build_sha256"]
            or release["release_id"] != worker["release_sha256"]
            or (
                record["expected_build_sha256"] is not None
                and build["build_id"] != record["expected_build_sha256"]
            )
            or (
                record["expected_release_sha256"] is not None
                and release["release_id"] != record["expected_release_sha256"]
            )
            or record["path"] != worker["path"]
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: scientific identity mismatch")
    elif not isinstance(record.get("error"), str):
        raise ValueError("INVALID_REPLAY_RECEIPT: failure lacks reason")

    # v1 stays byte-compatible; version 2 authenticates archived, phase-specific evidence.
    return record


def _replay_ancestry(
    source: Path,
    commit: str,
    project: Path,
    work_root: Path,
    *,
    allow_network: bool,
    expected_release_sha256: str | None,
    expected_build_sha256: str | None,
    corpus_work_root: Path | None,
    phase: Literal["build", "release"],
    parent_receipt: Path | None,
    inherited_source_ids: tuple[str, ...],
    expected_unchanged_snapshots: dict[str, str] | None,
    expected_changed_snapshots: dict[str, str] | None,
    use_historical_project: bool,
    expected_project_sha256: str | None,
) -> dict[str, Any]:
    from klab.corpus.acquisition import _project_sha, verify_acquisition
    from klab.corpus.project import load_project
    from klab.corpus.release import verify_build, verify_release

    root = Path(work_root).expanduser().absolute()
    _reject_symlink_ancestors(root)
    repository, _ = _repository(source, commit)
    _source_target(repository, root, commit)
    receipts = root / "replay" / "receipts"
    _reject_symlink_ancestors(receipts)
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{uuid.uuid4().hex}.json"
    work = (
        Path(corpus_work_root)
        if corpus_work_root is not None
        else root / "replay" / "work" / commit
    )
    receipt: dict[str, Any] = {
        "format": _ANCESTRY_FORMAT,
        "status": "FAILED",
        "stage": "preflight",
        "phase": phase,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "requested_source": str(Path(source).absolute()),
        "project": str(Path(project).absolute()),
        "corpus_work_root": str(work),
        "project_id": None,
        "expected_project_sha256": expected_project_sha256,
        "expected_build_sha256": expected_build_sha256,
        "expected_release_sha256": expected_release_sha256,
        "expected_unchanged_snapshots": expected_unchanged_snapshots or {},
        "expected_changed_snapshots": expected_changed_snapshots or {},
        "inherited_source_ids": list(inherited_source_ids),
        "parent_receipt": None,
        "allow_network": allow_network,
        "operations": [],
        "scientific_identity": {},
        "operational_identity": {
            "work_root": str(root),
            "orchestrator": str(Path(__file__).resolve()),
            "orchestrator_sha256": _ancestry_digest(Path(__file__)),
        },
    }
    failure: Exception | None = None
    try:
        if phase not in {"build", "release"} or not use_historical_project:
            raise ValueError(
                "INVALID_ANCESTRY_REQUEST: historical project and valid phase required"
            )
        if (phase == "release") != (expected_release_sha256 is not None):
            raise ValueError(
                "INVALID_ANCESTRY_REQUEST: release phase requires SHA; build phase prohibits it"
            )
        if expected_build_sha256 is None:
            raise ValueError("INVALID_ANCESTRY_REQUEST: ancestry build SHA required")
        for value in (
            expected_build_sha256,
            expected_release_sha256,
            expected_project_sha256,
        ):
            if value is not None and (
                type(value) is not str or not _HEX.fullmatch(value)
            ):
                raise ValueError(
                    "INVALID_EXPECTED_DIGEST: lowercase full SHA-256 required"
                )
        for values in (expected_unchanged_snapshots, expected_changed_snapshots):
            if values is not None and (
                type(values) is not dict
                or any(
                    type(key) is not str
                    or type(value) is not str
                    or not _HEX.fullmatch(value)
                    for key, value in values.items()
                )
            ):
                raise ValueError("INVALID_EXPECTED_DIGEST: invalid snapshot map")
        if (
            type(inherited_source_ids) is not tuple
            or any(type(item) is not str for item in inherited_source_ids)
            or len(set(inherited_source_ids)) != len(inherited_source_ids)
            or set(inherited_source_ids) != set(receipt["expected_unchanged_snapshots"])
        ):
            raise ValueError(
                "INVALID_ANCESTRY_REQUEST: exact unchanged inherited IDs required"
            )
        if set(receipt["expected_unchanged_snapshots"]) & set(
            receipt["expected_changed_snapshots"]
        ):
            raise ValueError(
                "INVALID_ANCESTRY_REQUEST: overlapping snapshot expectations"
            )
        project_relative = _relative(repository, project)
        if project_relative not in _tree(repository, commit):
            raise ValueError(
                "SOURCE_COMMIT_MISMATCH: missing historical project Git blob"
            )
        if corpus_work_root is not None:
            if not work.is_absolute():
                raise ValueError(
                    "UNSAFE_REPLAY_ROOT: explicit corpus root must be absolute"
                )
            _reject_symlink_ancestors(work)
            if (
                work.is_relative_to(root / "replay" / "source")
                or work.is_relative_to(root / "replay" / "env")
                or work.is_relative_to(root / "replay" / "receipts")
                or work.is_relative_to(root / "replay" / "logs")
                or work.is_relative_to(repository)
                or repository.is_relative_to(work)
                or root.is_relative_to(work)
            ):
                raise ValueError(
                    "UNSAFE_REPLAY_ROOT: corpus root overlaps protected state"
                )
            if work.exists() and not work.is_dir():
                raise ValueError(
                    "UNSAFE_REPLAY_ROOT: existing non-directory corpus root"
                )
            for other_path in receipts.glob("*.json"):
                other = json.loads(other_path.read_text(encoding="utf-8"))
                prior = Path(
                    other.get(
                        "corpus_work_root",
                        root / "replay" / "work" / other.get("source_commit", ""),
                    )
                )
                if (
                    work == prior
                    or work.is_relative_to(prior)
                    or prior.is_relative_to(work)
                ):
                    allowed = (
                        parent_receipt is not None
                        and other_path == Path(parent_receipt).absolute()
                        and phase == "release"
                        and other.get("phase") == "build"
                        and other.get("status") == "MATCH"
                    )
                    if not allowed:
                        raise ValueError(
                            "UNSAFE_REPLAY_ROOT: colliding prior corpus generation"
                        )
        receipt["stage"] = "materialization"
        source_record = materialize_source(source, commit, root)
        historical = Path(source_record["source_root"])
        destination = historical / project_relative
        receipt["source"] = {
            "source_root": str(historical),
            "tree": source_record["manifest"]["tree"],
            "inventory_sha256": source_record["manifest"]["inventory_sha256"],
            "manifest_path": source_record["manifest_path"],
        }
        receipt["historical_project"] = str(destination)
        preflight = implementation_preflight(
            source,
            commit,
            destination,
            historical_source_root=historical,
        )
        receipt["preflight"] = preflight
        receipt["repository"] = preflight["repository"]
        if preflight["status"] == "MISSING_IMPLEMENTATION":
            raise ValueError(f"MISSING_IMPLEMENTATION: {preflight['reason']}")
        corpus = load_project(destination)
        receipt["project_id"] = corpus.config.id
        project_sha = _project_sha(corpus)
        receipt["scientific_identity"]["project_sha256"] = project_sha
        if (
            expected_project_sha256 is not None
            and project_sha != expected_project_sha256
        ):
            raise ValueError(
                f"EXPECTED_PROJECT_MISMATCH: actual={project_sha} expected={expected_project_sha256}"
            )
        parent, inherited, origins = _ancestry_parent_verified(
            verify_replay_receipt(parent_receipt)
            if parent_receipt is not None
            else None,
            inherited_source_ids,
            corpus,
        )
        if parent is not None:
            if parent["format"] != _ANCESTRY_FORMAT or parent["status"] != "MATCH":
                raise ValueError(
                    "INVALID_PARENT_RECEIPT: successful v2 parent required"
                )
            parent_path = Path(parent_receipt).absolute()
            if parent_path.parent != receipts:
                raise ValueError("INVALID_PARENT_RECEIPT: parent must be task-owned")
            receipt["parent_receipt"] = {
                "path": str(parent_path),
                "sha256": _ancestry_digest(parent_path),
            }
            if set(parent["scientific_identity"]["snapshots"]) != set(
                receipt["expected_unchanged_snapshots"]
            ) | set(receipt["expected_changed_snapshots"]):
                raise ValueError(
                    "INVALID_PARENT_RECEIPT: expected maps must cover entire parent"
                )
            if parent["corpus_work_root"] == str(work) and (
                phase != "release"
                or parent["phase"] != "build"
                or not receipt["expected_changed_snapshots"]
            ):
                raise ValueError(
                    "INVALID_PARENT_RECEIPT: final in-place transition requires build-only parent and changed declarations"
                )
        elif receipt["expected_changed_snapshots"]:
            raise ValueError("INVALID_PARENT_RECEIPT: changed sources require parent")
        for source_id, identity in inherited.items():
            if (
                receipt["expected_unchanged_snapshots"][source_id]
                != identity["snapshot_sha256"]
            ):
                raise ValueError(
                    f"INVALID_PARENT_RECEIPT: inherited identity {source_id}"
                )
        if parent is not None:
            old = parent["scientific_identity"]["snapshots"]
            if any(
                old[key] != value
                for key, value in {
                    **receipt["expected_unchanged_snapshots"],
                    **receipt["expected_changed_snapshots"],
                }.items()
            ):
                raise ValueError(
                    "INVALID_PARENT_RECEIPT: expected map contradicts parent"
                )
        if parent is not None and receipt["expected_changed_snapshots"]:
            _ancestry_changed_declarations(
                parent, corpus, receipt["expected_changed_snapshots"]
            )
        imports = dict(inherited)
        if parent is not None:
            old_corpus = load_project(Path(parent["historical_project"]))
            _, previous, _ = _ancestry_parent_verified(
                parent, tuple(receipt["expected_changed_snapshots"]), old_corpus
            )
            imports.update(previous)
        _stage_ancestry_imports(parent, imports, work, corpus)
        _ancestry_target_imports(parent, imports, work, corpus.config.id)
        receipt["stage"] = "dependencies"
        environment = root / "replay" / "env" / commit
        _reject_symlink_ancestors(environment)
        environment.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in (
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "UV_ACTIVE",
            "PYTHONHOME",
            "PYTHONPATH",
            "UV_PROJECT_ENVIRONMENT",
        ):
            env.pop(key, None)
        env.update(
            {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "UV_NO_ENV_FILE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "UV_NO_PROGRESS": "1",
            }
        )
        uv = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        )
        if uv.returncode:
            raise ValueError(f"ENVIRONMENT_UNAVAILABLE: uv unavailable: {uv.stderr}")
        receipt["operational_identity"].update(
            {
                "uv": uv.stdout.strip(),
                "python": sys.version,
                "environment": str(environment),
                "lock_sha256": _ancestry_digest(historical / "uv.lock"),
            }
        )
        command = [
            "uv",
            "sync",
            "--project",
            str(historical),
            "--locked",
            "--no-install-project",
            "--no-dev",
            "--no-default-groups",
            "--python",
            sys.executable,
        ]
        historical_pyproject = tomllib.loads(
            (historical / "pyproject.toml").read_text(encoding="utf-8")
        )
        if "cpu" in historical_pyproject.get("project", {}).get(
            "optional-dependencies", {}
        ):
            command.extend(["--extra", "cpu"])
        if not allow_network:
            command.append("--offline")
        install = _run_logged(command, "dependencies", root / "replay" / "logs", env)
        receipt["operations"].append(install)
        if install["returncode"]:
            raise ValueError(
                "ENVIRONMENT_UNAVAILABLE: historical locked dependencies unavailable"
            )
        verify_materialized_source(historical, commit)
        python = environment / "bin" / "python"
        if not python.is_file():
            raise ValueError("ENVIRONMENT_UNAVAILABLE: dedicated uv python missing")
        dependencies = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if dependencies.returncode:
            raise ValueError(
                "ENVIRONMENT_UNAVAILABLE: unable to inventory dependencies"
            )
        interpreter = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--no-sync",
                "--python",
                str(python),
                "python",
                "-I",
                "-c",
                "import sys; print(sys.version)",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if interpreter.returncode:
            raise ValueError("ENVIRONMENT_UNAVAILABLE: dedicated interpreter failed")
        receipt["operational_identity"]["dependencies"] = (
            dependencies.stdout.splitlines()
        )
        receipt["operational_identity"]["environment_python"] = (
            interpreter.stdout.strip()
        )
        _reject_symlink_ancestors(work)
        work.mkdir(parents=True, exist_ok=True)
        closure_path = (
            root / "replay" / "acquisition-closures" / f"{receipt_path.stem}.json"
        )
        receipt["stage"] = "historical_producers"
        request = {
            "version": 2,
            "source_root": str(historical),
            "project": str(destination),
            "work_root": str(work),
            "allow_network": allow_network,
            "expected_build_sha256": expected_build_sha256,
            "expected_release_sha256": expected_release_sha256,
            "expected_project_sha256": expected_project_sha256,
            "components": preflight["components"],
            "phase": phase,
            "inherited_snapshots": inherited,
            "expected_unchanged_snapshots": receipt["expected_unchanged_snapshots"],
            "expected_changed_snapshots": receipt["expected_changed_snapshots"],
            "acquisition_closure_path": str(closure_path),
        }
        with tempfile.TemporaryDirectory(
            prefix=".invocation-", dir=receipts
        ) as temporary:
            request_path = Path(temporary) / "request.json"
            output_path = Path(temporary) / "response.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True), encoding="utf-8"
            )
            worker = Path(__file__).with_name("_implementation_worker.py")
            run = _run_logged(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--no-sync",
                    "--python",
                    str(python),
                    "python",
                    "-I",
                    str(worker),
                    str(request_path),
                    str(output_path),
                ],
                "historical_producers",
                root / "replay" / "logs",
                {**env, "UV_OFFLINE": "1" if not allow_network else "0"},
                work,
            )
            receipt["operations"].append(run)
            result = (
                json.loads(output_path.read_text(encoding="utf-8"))
                if output_path.is_file()
                else None
            )
        if closure_path.exists():
            _ancestry_artifact(closure_path, root / "replay" / "acquisition-closures")
            receipt["acquisition_closure"] = {
                "path": str(closure_path),
                "sha256": _ancestry_digest(closure_path),
            }
        if (
            type(result) is not dict
            or result.get("version") != 2
            or result.get("status") not in {"MATCH", "VERIFIED_BUILD", "FAILED"}
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: invalid historical worker result"
            )
        receipt["worker"] = result
        if run["returncode"] or result["status"] == "FAILED":
            raise ValueError(
                result.get("error", "HISTORICAL_EXECUTION_FAILED: worker failed")
            )
        if (
            result["status"] != ("MATCH" if phase == "release" else "VERIFIED_BUILD")
            or result["phase"] != phase
            or result["acquisition_closure"] != receipt.get("acquisition_closure")
        ):
            raise ValueError("HISTORICAL_EXECUTION_FAILED: phase or closure mismatch")
        receipt["stage"] = "independent_verification"
        lock = verify_acquisition(corpus, work, lock_path=closure_path)
        if _project_sha(corpus) != project_sha or lock["project_sha256"] != project_sha:
            raise ValueError(
                "EXPECTED_PROJECT_MISMATCH: declaration changed before build"
            )
        _ancestry_expected(
            lock,
            inherited,
            receipt["expected_unchanged_snapshots"],
            receipt["expected_changed_snapshots"],
            parent,
        )
        built = _ancestry_artifact(Path(result["build_path"]), work)
        build = verify_build(built)
        frozen = None
        release = None
        if phase == "release":
            frozen = _ancestry_artifact(Path(result["path"]), work)
            release = verify_release(frozen)
        verified_source = verify_materialized_source(historical, commit)
        _ancestry_worker_binding(
            result,
            verified_source,
            historical,
            work,
            inherited,
            parent,
            lock,
            build,
            release,
        )
        snapshots = _verify_producer_provenance(
            verified_source, lock, build, release, origins
        )
        observed = {item["source_id"]: item["sha256"] for item in snapshots}
        if (
            observed != _ancestry_maps(lock)
            or result["post_snapshots"] != observed
            or result["snapshots"] != snapshots
            or result["project_sha256"] != project_sha
            or result["build_sha256"] != build["build_id"]
            or (
                expected_build_sha256 is not None
                and expected_build_sha256 != build["build_id"]
            )
            or (
                release is not None and release["release_id"] != expected_release_sha256
            )
            or (
                release is not None
                and release["release_id"] != result["release_sha256"]
            )
            or (release is None and ("path" in result or "release_sha256" in result))
        ):
            raise ValueError("EXPECTED_DIGEST_MISMATCH: independent identity mismatch")
        receipt["scientific_identity"].update(
            {
                "build_sha256": build["build_id"],
                "release_sha256": release["release_id"] if release else None,
                "snapshots": observed,
            }
        )
        receipt["build_id"] = build["build_id"]
        receipt["release"] = str(frozen) if frozen else None
        if frozen:
            receipt["release_id"] = release["release_id"]
        receipt["stage"] = "complete"
        receipt["status"] = "MATCH"
    except Exception as error:  # noqa: BLE001 - publish before raising ReplayFailure
        if (
            isinstance(receipt.get("worker"), dict)
            and receipt["worker"].get("status") == "FAILED"
        ):
            receipt["stage"] = receipt["worker"].get("phase", "historical_producers")
        if receipt.get("acquisition_closure") is not None:
            try:
                _ancestry_partial_artifacts(receipt, record_observed=True)
            except (ValueError, TypeError, KeyError, OSError) as closure_error:
                receipt["closure_verification_error"] = str(closure_error)
        receipt["error"] = str(error)
        failure = error
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        receipt["record_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with receipt_path.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
    if failure is not None:
        raise ReplayFailure(
            str(failure), receipt_path, receipt["record_sha256"]
        ) from failure
    verify_replay_receipt(receipt_path)
    return {
        "path": receipt["release"] or str(built),
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def _ancestry_partial_artifacts(
    record: dict[str, Any], *, record_observed: bool
) -> None:
    """Independently bind all available artifacts of an interrupted phase."""
    from klab.corpus.acquisition import (
        _project_sha,
        verify_acquisition,
        verify_snapshot,
    )
    from klab.corpus.project import load_project
    from klab.corpus.release import verify_build, verify_release

    source = Path(record["source"]["source_root"])
    authenticated = verify_materialized_source(source, record["source_commit"])
    if authenticated["inventory_sha256"] != record["source"]["inventory_sha256"]:
        raise ValueError("INVALID_REPLAY_RECEIPT: incomplete historical source")
    project_path = Path(record["historical_project"])
    if not project_path.is_relative_to(source) or project_path.is_symlink():
        raise ValueError("INVALID_REPLAY_RECEIPT: unsafe historical project")
    corpus = load_project(project_path)
    if record["project_id"] != corpus.config.id:
        raise ValueError("INVALID_REPLAY_RECEIPT: project ID mismatch")
    work = Path(record["corpus_work_root"])
    _reject_symlink_ancestors(work)
    lock = verify_acquisition(
        corpus,
        work,
        lock_path=Path(record["acquisition_closure"]["path"]),
    )
    identity = record["scientific_identity"]
    observed = _ancestry_maps(lock)
    if (
        lock["project_sha256"] != _project_sha(corpus)
        or (
            "project_sha256" in identity
            and identity["project_sha256"] != lock["project_sha256"]
        )
        or ("snapshots" in identity and identity["snapshots"] != observed)
        or (
            not record_observed
            and (
                identity.get("project_sha256") != lock["project_sha256"]
                or identity.get("snapshots") != observed
            )
        )
    ):
        raise ValueError(
            "INVALID_REPLAY_RECEIPT: failed-phase project or snapshot mismatch"
        )
    if record_observed:
        identity["project_sha256"] = lock["project_sha256"]
        identity["snapshots"] = observed
    provenance = {
        source_id: verify_snapshot(Path(entry["snapshot_path"]))["adapter"][
            "module_sha256"
        ]
        for source_id, entry in lock["sources"].items()
        if entry["snapshot_sha256"] is not None
    }
    if record_observed:
        identity["snapshot_provenance"] = provenance
    elif identity.get("snapshot_provenance") != provenance:
        raise ValueError(
            "INVALID_REPLAY_RECEIPT: failed-phase adapter provenance mismatch"
        )
    worker = record.get("worker", {})
    artifacts = (
        worker.get("artifacts", {}) if worker.get("status") == "FAILED" else worker
    )
    if "snapshots" in artifacts and artifacts["snapshots"] != [
        {"source_id": key, "sha256": value["snapshot_sha256"]}
        for key, value in lock["sources"].items()
        if value["snapshot_sha256"] is not None
    ]:
        raise ValueError(
            "INVALID_REPLAY_RECEIPT: failed-phase snapshot inventory mismatch"
        )
    if "post_snapshots" in artifacts and artifacts["post_snapshots"] != observed:
        raise ValueError(
            "INVALID_REPLAY_RECEIPT: failed-phase post-acquisition map mismatch"
        )
    if (
        "project_sha256" in artifacts
        and artifacts["project_sha256"] != lock["project_sha256"]
    ):
        raise ValueError("INVALID_REPLAY_RECEIPT: failed-phase worker project mismatch")
    if "build_path" in artifacts:
        build = verify_build(_ancestry_artifact(Path(artifacts["build_path"]), work))
        if artifacts["build_sha256"] != build["build_id"] or (
            "build_sha256" in identity and identity["build_sha256"] != build["build_id"]
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: failed-phase build mismatch")
        if record_observed:
            identity["build_sha256"] = build["build_id"]
        elif identity.get("build_sha256") != build["build_id"]:
            raise ValueError("INVALID_REPLAY_RECEIPT: missing failed-phase build ID")
        if "path" in artifacts:
            release = verify_release(_ancestry_artifact(Path(artifacts["path"]), work))
            if (
                artifacts["release_sha256"] != release["release_id"]
                or release["build_id"] != build["build_id"]
                or (
                    "release_sha256" in identity
                    and identity["release_sha256"] != release["release_id"]
                )
            ):
                raise ValueError(
                    "INVALID_REPLAY_RECEIPT: failed-phase release mismatch"
                )
            if record_observed:
                identity["release_sha256"] = release["release_id"]
            elif identity.get("release_sha256") != release["release_id"]:
                raise ValueError(
                    "INVALID_REPLAY_RECEIPT: missing failed-phase release ID"
                )


def _ancestry_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ancestry_artifact(path: Path, root: Path) -> Path:
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError("INVALID_REPLAY_RECEIPT: relative artifact path")
    _reject_symlink_ancestors(path)
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("INVALID_REPLAY_RECEIPT: artifact outside corpus work root")
    return path


def _ancestry_maps(lock: dict[str, Any]) -> dict[str, str]:
    return {
        key: value["snapshot_sha256"]
        for key, value in lock["sources"].items()
        if value["snapshot_sha256"] is not None
    }


def _ancestry_worker_binding(
    worker: dict[str, Any],
    source: dict[str, Any],
    source_root: Path,
    work: Path,
    inherited: dict[str, dict[str, str]],
    parent: dict[str, Any] | None,
    lock: dict[str, Any],
    build: dict[str, Any],
    release: dict[str, Any] | None,
) -> None:
    modules = worker["modules"]
    module_paths = {
        "training_manifest": "src/klab/training/manifest.py",
        "project_config": "src/klab/config/models.py",
        **{key: f"src/klab/corpus/{key}.py" for key in _COMPONENTS},
    }
    if set(modules) != set(module_paths):
        raise ValueError("IMPLEMENTATION_PROVENANCE_MISMATCH: imported module set")
    if Path(modules["acquisition"]).parents[3] != source_root:
        raise ValueError("IMPLEMENTATION_PROVENANCE_MISMATCH: imported source root")
    for name, relative in module_paths.items():
        module = Path(modules[name])
        _reject_symlink_ancestors(module)
        if (
            module != source_root / relative
            or _ancestry_digest(module) != source["files"][relative]["sha256"]
        ):
            raise ValueError(
                f"IMPLEMENTATION_PROVENANCE_MISMATCH: imported module {name}"
            )
    if worker["reuse_decisions"] != {
        key: "verified_immutable_reuse" for key in inherited
    }:
        raise ValueError("INHERITED_IDENTITY_MISMATCH: unverified reuse decision")
    pre = worker["pre_snapshots"]
    if type(pre) is not dict:
        raise ValueError("INHERITED_IDENTITY_MISMATCH: invalid pre-acquisition map")
    if parent is not None:
        prior = parent["scientific_identity"]["snapshots"]
        if any(prior.get(key) != value for key, value in pre.items()):
            raise ValueError("INHERITED_IDENTITY_MISMATCH: pre-acquisition identity")
        if parent["corpus_work_root"] == str(work) and pre != prior:
            raise ValueError(
                "INHERITED_IDENTITY_MISMATCH: incomplete in-place prior lock"
            )
    observed = _ancestry_maps(lock)
    if worker["post_snapshots"] != observed or worker["snapshots"] != [
        {"source_id": key, "sha256": value["snapshot_sha256"]}
        for key, value in lock["sources"].items()
        if value["snapshot_sha256"] is not None
    ]:
        raise ValueError("INHERITED_IDENTITY_MISMATCH: worker snapshot inventory")
    if {row["source_id"]: row["sha256"] for row in build["snapshots"]} != observed or (
        release is not None
        and {row["source_id"]: row["sha256"] for row in release["snapshots"]}
        != observed
    ):
        raise ValueError("IMPLEMENTATION_PROVENANCE_MISMATCH: build/lock snapshots")


def _ancestry_changed_declarations(
    parent: dict[str, Any],
    corpus: Any,
    changed: dict[str, str],
) -> None:
    from klab.corpus.project import load_project, source_declaration_payload

    old_project = load_project(Path(parent["historical_project"]))
    old = {
        item.id: (item, (old_project.root / path).read_bytes())
        for item, path in zip(
            old_project.sources, old_project.config.sources, strict=True
        )
    }
    new = {
        item.id: (item, (corpus.root / path).read_bytes())
        for item, path in zip(corpus.sources, corpus.config.sources, strict=True)
    }
    for source_id in changed:
        if (
            source_id not in old
            or source_id not in new
            or old[source_id][1] == new[source_id][1]
            or source_declaration_payload(old[source_id][0])
            == source_declaration_payload(new[source_id][0])
        ):
            raise ValueError(f"CHANGED_DECLARATION_MISMATCH: {source_id}")


def _stage_ancestry_imports(
    parent: dict[str, Any] | None,
    inherited: dict[str, dict[str, str]],
    work: Path,
    corpus: Any,
) -> None:
    """Publish fresh verified inheritance atomically, without any acquisition."""
    from klab.corpus.acquisition import (
        _project_sha,
        _rename_noreplace,
        _write_json,
        declaration_sha256,
        verify_acquisition,
        verify_snapshot,
    )
    from klab.result_publication import sync_directory

    if parent is None or parent["corpus_work_root"] == str(work) or work.exists():
        return
    if corpus.config.source_effects is not None:
        raise ValueError(
            "INVALID_ANCESTRY_REQUEST: effect-bound imports require native acquisition"
        )
    _reject_symlink_ancestors(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inherit-", dir=work.parent) as temp:
        staged = Path(temp) / "work"
        base = staged / "corpora" / corpus.config.id
        entries = {}
        for source_id, identity in inherited.items():
            origin = (
                Path(parent["corpus_work_root"])
                / "corpora"
                / parent["project_id"]
                / "snapshots"
                / source_id
                / identity["snapshot_sha256"]
            )
            _reject_symlink_ancestors(origin)
            for member in origin.rglob("*"):
                mode = member.lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError("INVALID_PARENT_RECEIPT: unsafe inherited member")
            before = verify_snapshot(origin)
            if (
                _ancestry_digest(origin / "manifest.json")
                != identity["manifest_sha256"]
            ):
                raise ValueError("INVALID_PARENT_RECEIPT: inherited manifest changed")
            target = base / "snapshots" / source_id / identity["snapshot_sha256"]
            shutil.copytree(origin, target, symlinks=True)
            for member in target.rglob("*"):
                mode = member.lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError("INVALID_PARENT_RECEIPT: unsafe copied member")
            if (
                verify_snapshot(target) != before
                or _ancestry_digest(target / "manifest.json")
                != identity["manifest_sha256"]
            ):
                raise ValueError("INVALID_PARENT_RECEIPT: inherited copy changed")
            entries[source_id] = {
                "declaration_sha256": before["declaration_sha256"],
                "snapshot_sha256": identity["snapshot_sha256"],
                "snapshot_path": str(
                    work
                    / "corpora"
                    / corpus.config.id
                    / "snapshots"
                    / source_id
                    / identity["snapshot_sha256"]
                ),
                "receipt": {"status": "acquired", "retrieval": before["retrieval"]},
            }
        # Preserve the parent's pre-acquisition inventory even with new sources.
        # Only a complete unchanged inventory can pass offline verification.
        lock = {
            "schema_version": 1,
            "project_id": corpus.config.id,
            "project_sha256": _project_sha(corpus),
            "sources": entries,
        }
        complete_unchanged = set(entries) == {
            item.id for item in corpus.sources
        } and all(
            entries[item.id]["declaration_sha256"] == declaration_sha256(item)
            for item in corpus.sources
        )
        if complete_unchanged:
            for item in entries.values():
                item["snapshot_path"] = str(
                    staged / Path(item["snapshot_path"]).relative_to(work)
                )
            _write_json(base / "acquisition.json", lock)
            verify_acquisition(corpus, staged)
            for item in entries.values():
                item["snapshot_path"] = str(
                    work / Path(item["snapshot_path"]).relative_to(staged)
                )
        _write_json(base / "acquisition.json", lock)
        for member in sorted(staged.rglob("*"), reverse=True):
            if member.is_file():
                with member.open("rb") as stream:
                    os.fsync(stream.fileno())
            elif member.is_dir():
                sync_directory(member)
        sync_directory(staged)
        _rename_noreplace(staged, work)
        sync_directory(work.parent)
    if complete_unchanged:
        verify_acquisition(corpus, work)


def _ancestry_target_imports(
    parent: dict[str, Any] | None,
    inherited: dict[str, dict[str, str]],
    work: Path,
    project_id: str,
) -> None:
    from klab.corpus.acquisition import verify_snapshot

    if parent is None:
        if work.exists() and any(work.iterdir()):
            raise ValueError("UNSAFE_REPLAY_ROOT: non-empty unowned corpus root")
        return
    if parent["corpus_work_root"] == str(work):
        if parent["project_id"] != project_id:
            raise ValueError("INVALID_PARENT_RECEIPT: in-place project ID changed")
        return
    base = work / "corpora" / project_id / "snapshots"
    if work.exists():
        _reject_symlink_ancestors(work)
        for candidate in work.iterdir():
            if candidate != work / "corpora" or candidate.is_symlink():
                raise ValueError(
                    "UNSAFE_REPLAY_ROOT: unrelated corpus workspace content"
                )
        for candidate in (work / "corpora").iterdir():
            if candidate != work / "corpora" / project_id or candidate.is_symlink():
                raise ValueError("UNSAFE_REPLAY_ROOT: unrelated corpus generation")
        for candidate in (work / "corpora" / project_id).iterdir():
            if (
                candidate not in {base, base.parent / "acquisition.json"}
                or candidate.is_symlink()
            ):
                raise ValueError("UNSAFE_REPLAY_ROOT: unexpected existing corpus state")
        if {child.name for child in base.iterdir()} != set(inherited):
            raise ValueError("INVALID_PARENT_RECEIPT: target import ID set differs")
    for source_id, identity in inherited.items():
        target = base / source_id / identity["snapshot_sha256"]
        _reject_symlink_ancestors(target)
        if set((base / source_id).iterdir()) != {target}:
            raise ValueError(
                f"INVALID_PARENT_RECEIPT: conflicting imported snapshot {source_id}"
            )
        manifest = verify_snapshot(target)
        old_path = (
            Path(parent["corpus_work_root"])
            / "corpora"
            / parent["project_id"]
            / "snapshots"
            / source_id
            / identity["snapshot_sha256"]
        )
        earlier = verify_snapshot(old_path)
        if (
            manifest != earlier
            or _ancestry_digest(target / "manifest.json") != identity["manifest_sha256"]
            or _ancestry_digest(old_path / "manifest.json")
            != identity["manifest_sha256"]
        ):
            raise ValueError(
                f"INVALID_PARENT_RECEIPT: imported bytes differ {source_id}"
            )


def _ancestry_expected(
    lock: dict[str, Any],
    inherited: dict[str, dict[str, str]],
    unchanged: dict[str, str],
    changed: dict[str, str],
    parent: dict[str, Any] | None,
) -> None:
    from klab.corpus.acquisition import verify_snapshot

    observed = _ancestry_maps(lock)
    if set(unchanged) & set(changed):
        raise ValueError(
            "INHERITED_IDENTITY_MISMATCH: overlapping changed and unchanged"
        )
    if parent is not None:
        prior = parent["scientific_identity"]["snapshots"]
        if set(prior) != set(unchanged) | set(changed):
            raise ValueError(
                "INHERITED_IDENTITY_MISMATCH: incomplete parent snapshot map"
            )
        if any(prior[key] != value for key, value in {**unchanged, **changed}.items()):
            raise ValueError(
                "INHERITED_IDENTITY_MISMATCH: parent snapshot identity changed"
            )
    for source_id, expected in unchanged.items():
        if observed.get(source_id) != expected:
            raise ValueError(
                f"INHERITED_IDENTITY_MISMATCH: unchanged snapshot {source_id}"
            )
    for source_id, previous in changed.items():
        if source_id not in observed or observed[source_id] == previous:
            raise ValueError(
                f"INHERITED_IDENTITY_MISMATCH: unchanged required change {source_id}"
            )
    for source_id, identity in inherited.items():
        entry = lock["sources"][source_id]
        if entry["snapshot_sha256"] != identity["snapshot_sha256"]:
            raise ValueError(
                f"INHERITED_IDENTITY_MISMATCH: inherited snapshot {source_id}"
            )
        snapshot = Path(entry["snapshot_path"])
        manifest = verify_snapshot(snapshot)
        if (
            _ancestry_digest(snapshot / "manifest.json") != identity["manifest_sha256"]
            or manifest["adapter"]["module_sha256"] != identity["adapter_module_sha256"]
        ):
            raise ValueError(
                f"INHERITED_IDENTITY_MISMATCH: inherited manifest {source_id}"
            )
        if parent is not None:
            previous = Path(parent["corpus_work_root"]) / "corpora"
            previous = (
                previous
                / parent["project_id"]
                / "snapshots"
                / source_id
                / identity["snapshot_sha256"]
            )
            earlier = verify_snapshot(previous)
            if (
                manifest != earlier
                or _ancestry_digest(previous / "manifest.json")
                != identity["manifest_sha256"]
            ):
                raise ValueError(
                    f"INHERITED_IDENTITY_MISMATCH: inherited bytes {source_id}"
                )


def _verify_ancestry_receipt(path: Path, seen: set[Path]) -> dict[str, Any]:
    from klab.corpus.acquisition import _project_sha, verify_acquisition
    from klab.corpus.project import load_project
    from klab.corpus.release import verify_build, verify_release

    path = Path(path).absolute()
    _reject_symlink_ancestors(path)
    if path in seen:
        raise ValueError("INVALID_REPLAY_RECEIPT: cyclic parent chain")
    seen.add(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            type(record) is not dict
            or record.get("format") != _ANCESTRY_FORMAT
            or record.get("status") not in {"MATCH", "FAILED"}
            or record.get("phase") not in {"build", "release"}
            or record.get("record_sha256")
            != hashlib.sha256(
                json.dumps(
                    {k: v for k, v in record.items() if k != "record_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: invalid ancestry record")
        root = Path(record["operational_identity"]["work_root"])
        if (
            not root.is_absolute()
            or path.parent != root / "replay" / "receipts"
            or path.suffix != ".json"
            or record["expected_build_sha256"] is None
            or not _HEX.fullmatch(record["expected_build_sha256"])
            or (record["phase"] == "release")
            != (record["expected_release_sha256"] is not None)
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: invalid task root or phase gate")
        _reject_symlink_ancestors(root)
        source = Path(record["source"]["source_root"]) if record.get("source") else None
        if source is not None:
            if source != root / "replay" / "source" / record["source_commit"]:
                raise ValueError("INVALID_REPLAY_RECEIPT: source root mismatch")
            authenticated = verify_materialized_source(source, record["source_commit"])
            if (
                authenticated["inventory_sha256"]
                != record["source"]["inventory_sha256"]
                or record["source"]["tree"] != authenticated["tree"]
                or record["source"]["manifest_path"]
                != str(
                    root
                    / "replay"
                    / "source-manifests"
                    / f"{record['source_commit']}.json"
                )
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: source inventory mismatch")
            repo, _ = _repository(
                Path(record["requested_source"]), record["source_commit"]
            )
            if (
                repo != Path(authenticated["repository"])
                or record["repository"]["tree"] != authenticated["tree"]
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: repository mismatch")
        for operation in record["operations"]:
            for channel in ("stdout", "stderr"):
                stream = Path(operation[f"{channel}_path"])
                _ancestry_artifact(stream, root / "replay" / "logs")
                if _ancestry_digest(stream) != operation[f"{channel}_sha256"]:
                    raise ValueError("INVALID_REPLAY_RECEIPT: log bytes mismatch")
        if record["status"] == "MATCH" and (
            [operation["phase"] for operation in record["operations"]]
            != ["dependencies", "historical_producers"]
            or any(operation["returncode"] != 0 for operation in record["operations"])
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: incomplete subprocess evidence")
        parent_link = record["parent_receipt"]
        parent = None
        if parent_link is not None:
            parent_path = Path(parent_link["path"])
            if (
                not parent_path.is_absolute()
                or parent_path.parent != root / "replay" / "receipts"
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: invalid parent path")
            parent = _verify_ancestry_receipt(parent_path, seen)
            if (
                parent["status"] != "MATCH"
                or _ancestry_digest(parent_path) != parent_link["sha256"]
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: parent mismatch")
        closure = record.get("acquisition_closure")
        if closure is not None:
            lock_path = Path(closure["path"])
            if (
                lock_path.parent != root / "replay" / "acquisition-closures"
                or lock_path.name != path.name
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: unexpected closure path")
            _ancestry_artifact(lock_path, root / "replay" / "acquisition-closures")
            if _ancestry_digest(lock_path) != closure["sha256"]:
                raise ValueError("INVALID_REPLAY_RECEIPT: closure bytes mismatch")
        if (
            closure is not None
            and record.get("worker", {}).get("status") == "FAILED"
            and record["worker"]["artifacts"].get("acquisition_closure") != closure
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: failure closure mismatch")
        if record["status"] != "MATCH":
            if not isinstance(record.get("error"), str):
                raise ValueError("INVALID_REPLAY_RECEIPT: failure lacks reason")
            if closure is not None:
                if "closure_verification_error" in record:
                    raise ValueError(
                        "INVALID_REPLAY_RECEIPT: archived failure lock was not independently verified"
                    )
                _ancestry_partial_artifacts(record, record_observed=False)
            return record
        if source is None or closure is None or record.get("stage") != "complete":
            raise ValueError("INVALID_REPLAY_RECEIPT: incomplete successful ancestry")
        project_path = Path(record["historical_project"])
        if not project_path.is_relative_to(source) or project_path.is_symlink():
            raise ValueError("INVALID_REPLAY_RECEIPT: unsafe historical project")
        relative = project_path.relative_to(source).as_posix()
        if relative != record["repository"]["project"]:
            raise ValueError("INVALID_REPLAY_RECEIPT: incorrect historical declaration")
        preflight = implementation_preflight(
            Path(record["requested_source"]),
            record["source_commit"],
            project_path,
            historical_source_root=source,
        )
        if preflight["repository"]["project"] != record["repository"]["project"] or any(
            preflight["components"][name]["pinned_sha256"] != entry["pinned_sha256"]
            for name, entry in record["preflight"]["components"].items()
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: declaration or producer changed")
        corpus = load_project(project_path)
        work = Path(record["corpus_work_root"])
        if not work.is_absolute():
            raise ValueError("INVALID_REPLAY_RECEIPT: relative corpus root")
        _reject_symlink_ancestors(work)
        if (
            work.is_relative_to(repo)
            or repo.is_relative_to(work)
            or root.is_relative_to(work)
            or any(
                work.is_relative_to(root / "replay" / component)
                for component in (
                    "source",
                    "env",
                    "receipts",
                    "logs",
                    "source-manifests",
                    "acquisition-closures",
                )
            )
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: overlapping corpus root")
        lock = verify_acquisition(corpus, work, lock_path=lock_path)
        if _project_sha(corpus) != record["scientific_identity"][
            "project_sha256"
        ] or lock["project_sha256"] != _project_sha(corpus):
            raise ValueError("INVALID_REPLAY_RECEIPT: project SHA mismatch")
        if record["expected_project_sha256"] is not None and record[
            "expected_project_sha256"
        ] != _project_sha(corpus):
            raise ValueError("INVALID_REPLAY_RECEIPT: expected project SHA mismatch")
        if parent is not None and record["expected_changed_snapshots"]:
            _ancestry_changed_declarations(
                parent, corpus, record["expected_changed_snapshots"]
            )
        parent_sources = () if parent is None else tuple(record["inherited_source_ids"])
        if set(parent_sources) != set(record["expected_unchanged_snapshots"]):
            raise ValueError("INVALID_REPLAY_RECEIPT: inherited source set changed")
        if parent is None and record["expected_changed_snapshots"]:
            raise ValueError("INVALID_REPLAY_RECEIPT: changed snapshots lack parent")
        if parent is not None and (
            set(parent["scientific_identity"]["snapshots"])
            != set(record["expected_unchanged_snapshots"])
            | set(record["expected_changed_snapshots"])
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: incomplete parent snapshot map")
        if (
            parent is not None
            and parent["corpus_work_root"] == str(work)
            and (
                record["phase"] != "release"
                or parent["phase"] != "build"
                or not record["expected_changed_snapshots"]
            )
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: invalid in-place transition")
        verified_parent, inherited, origins = _ancestry_parent_verified(
            parent, parent_sources, corpus
        )
        _ancestry_expected(
            lock,
            inherited,
            record["expected_unchanged_snapshots"],
            record["expected_changed_snapshots"],
            verified_parent,
        )
        worker = record["worker"]
        if (
            worker["version"] != 2
            or worker["phase"] != record["phase"]
            or worker["status"] not in {"MATCH", "VERIFIED_BUILD"}
            or worker["acquisition_closure"] != closure
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: worker mismatch")
        if worker["project_sha256"] != _project_sha(corpus) or worker[
            "post_snapshots"
        ] != _ancestry_maps(lock):
            raise ValueError("INVALID_REPLAY_RECEIPT: worker snapshot/project mismatch")
        built = _ancestry_artifact(Path(worker["build_path"]), work)
        build = verify_build(built)
        release = None
        if record["phase"] == "build":
            if (
                worker["status"] != "VERIFIED_BUILD"
                or "path" in worker
                or "release_sha256" in worker
                or record.get("release") is not None
                or "release_id" in record
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: build-only phase has release")
        else:
            if worker["status"] != "MATCH" or record.get("release_id") != worker.get(
                "release_sha256"
            ):
                raise ValueError("INVALID_REPLAY_RECEIPT: missing release")
            frozen = _ancestry_artifact(Path(worker["path"]), work)
            release = verify_release(frozen)
            if release["release_id"] != worker["release_sha256"] or record.get(
                "release"
            ) != str(frozen):
                raise ValueError("INVALID_REPLAY_RECEIPT: release mismatch")
        _ancestry_worker_binding(
            worker, authenticated, source, work, inherited, parent, lock, build, release
        )
        snapshots = _verify_producer_provenance(
            authenticated, lock, build, release, origins
        )
        if (
            {item["source_id"]: item["sha256"] for item in snapshots}
            != _ancestry_maps(lock)
            or record["scientific_identity"]["snapshots"] != _ancestry_maps(lock)
            or build["build_id"] != worker["build_sha256"]
            or build["build_id"] != record["build_id"]
            or build["build_id"] != record["scientific_identity"]["build_sha256"]
            or record["scientific_identity"]["release_sha256"]
            != (release["release_id"] if release else None)
            or (
                record["expected_build_sha256"] is not None
                and build["build_id"] != record["expected_build_sha256"]
            )
            or (
                record["expected_release_sha256"]
                != (release["release_id"] if release else None)
            )
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: scientific identity mismatch")
        return record
    finally:
        seen.remove(path)


def _ancestry_parent_verified(
    parent: dict[str, Any] | None,
    ids: tuple[str, ...],
    corpus: Any,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, str]], dict[str, str]]:
    if parent is None:
        if ids:
            raise ValueError("INVALID_PARENT_RECEIPT: inherited sources require parent")
        return None, {}, {}
    from klab.corpus.acquisition import verify_acquisition, verify_snapshot
    from klab.corpus.project import load_project, source_declaration_payload

    old_project = load_project(Path(parent["historical_project"]))
    old_lock = verify_acquisition(
        old_project,
        Path(parent["corpus_work_root"]),
        lock_path=Path(parent["acquisition_closure"]["path"]),
    )
    before = {
        item.id: (item, (old_project.root / path).read_bytes())
        for item, path in zip(
            old_project.sources, old_project.config.sources, strict=True
        )
    }
    after = {
        item.id: (item, (corpus.root / path).read_bytes())
        for item, path in zip(corpus.sources, corpus.config.sources, strict=True)
    }
    if len(set(ids)) != len(ids):
        raise ValueError("INVALID_PARENT_RECEIPT: duplicate inherited source")
    inherited: dict[str, dict[str, str]] = {}
    origins: dict[str, str] = {}
    for source_id in ids:
        if (
            source_id not in before
            or source_id not in after
            or before[source_id][1] != after[source_id][1]
            or source_declaration_payload(before[source_id][0])
            != source_declaration_payload(after[source_id][0])
        ):
            raise ValueError(f"INVALID_PARENT_RECEIPT: changed declaration {source_id}")
        entry = old_lock["sources"][source_id]
        if not entry["snapshot_sha256"]:
            raise ValueError(f"INVALID_PARENT_RECEIPT: rejected source {source_id}")
        snapshot = Path(entry["snapshot_path"])
        manifest = verify_snapshot(snapshot)
        origins[source_id] = manifest["adapter"]["module_sha256"]
        inherited[source_id] = {
            "snapshot_sha256": entry["snapshot_sha256"],
            "manifest_sha256": _ancestry_digest(snapshot / "manifest.json"),
            "adapter_module_sha256": origins[source_id],
        }
    return parent, inherited, origins
