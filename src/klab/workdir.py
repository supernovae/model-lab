"""Persistent state-root selection, disposable scratch, and checkout storage warnings."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO

WORK_DIR_ENV = "KLAB_WORK_DIR"
_TEMP_DIR_ENV_VARS = ("TMPDIR", "TEMP", "TMP")
CHECKOUT_STORAGE_REASON = "STORAGE_INSIDE_GIT_CHECKOUT"


def resolve_work_dir(
    path: str | Path | None = None, *, cwd: Path | None = None
) -> Path:
    """Resolve the persistent root without creating it or tying it to a checkout."""
    current = (cwd if cwd is not None else Path.cwd()).expanduser().resolve()
    configured = path if path is not None else os.environ.get(WORK_DIR_ENV)
    if configured is None or not os.fspath(configured).strip():
        xdg = os.environ.get("XDG_DATA_HOME", "")
        if xdg.strip() and Path(xdg).is_absolute():
            return (Path(xdg) / "klab").resolve()
        return (Path.home().resolve() / ".local" / "share" / "klab").resolve()

    resolved = Path(configured).expanduser()
    if not resolved.is_absolute():
        resolved = current / resolved
    return resolved.resolve()


def ensure_scratch_dir(
    path: str | Path | None = None, *, cwd: Path | None = None
) -> Path:
    """Create only the disposable scratch child of the selected persistent root."""
    root = resolve_work_dir(path, cwd=cwd)
    scratch = root / "scratch"

    def validate_location() -> None:
        if scratch.is_symlink():
            raise ValueError(
                f"scratch directory must not be a symlink: {scratch}; "
                "choose a different --work-dir or KLAB_WORK_DIR with a real "
                "scratch directory"
            )
        # Also reject resolved redirects not identified as symlinks (e.g. junctions).
        if scratch.resolve().parent != root:
            raise ValueError(
                f"scratch directory must remain inside work root: {scratch}"
            )

    # Refuse existing redirects before mkdir, and validate the resulting path.
    # Path checks do not protect against adversarial concurrent replacement.
    validate_location()
    scratch.mkdir(parents=True, exist_ok=True)
    validate_location()
    if not scratch.is_dir():
        raise NotADirectoryError(scratch)
    return scratch


def ensure_work_dir(path: str | Path | None = None, *, cwd: Path | None = None) -> Path:
    """Initialize persistent root and direct process temporary files to scratch."""
    work_dir = resolve_work_dir(path, cwd=cwd)
    scratch = ensure_scratch_dir(work_dir)
    os.environ[WORK_DIR_ENV] = str(work_dir)
    for name in _TEMP_DIR_ENV_VARS:
        os.environ[name] = str(scratch)
    tempfile.tempdir = str(scratch)
    return work_dir


def storage_checks(
    path: str | Path, *, kind: str = "work_root"
) -> list[dict[str, str]]:
    """Read-only checkout-containment warnings for a root or explicit payload path.

    Git is queried from the nearest existing directory, so a not-yet-created
    destination still detects the containing checkout. Resolved paths also catch
    a destination reached through a symlink.
    """
    selected = Path(path).expanduser().resolve()
    ancestor = selected
    while not ancestor.is_dir():
        if ancestor == ancestor.parent:
            return []
        ancestor = ancestor.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    git_root = Path(result.stdout.strip()).resolve()
    if not selected.is_relative_to(git_root):
        return []
    return [
        {
            "reason_code": CHECKOUT_STORAGE_REASON,
            "kind": kind,
            "path": str(selected),
            "git_root": str(git_root),
        }
    ]


def warn_storage_checks(
    checks: list[dict[str, str]], *, stream: TextIO | None = None
) -> list[dict[str, str]]:
    """Emit checkout warnings to stderr while retaining the same structured evidence."""
    destination = sys.stderr if stream is None else stream
    for check in checks:
        print(
            f"{check['reason_code']}: {check['kind']} {check['path']} "
            f"is inside Git checkout {check['git_root']}; "
            "persistent bytes may be lost with this checkout",
            file=destination,
        )
    return checks


def record_storage_observation(
    root: str | Path,
    checks: list[dict[str, str]],
    operation: str,
    *,
    target: str | Path | None = None,
) -> Path:
    """Seal the operational location selected before a long-lived materialization.

    This is execution evidence, not a scientific digest. A replay returns the
    original record and timestamp without writing it again.
    """
    from klab.canonical import domain_digest
    from klab.result_publication import publish_immutable, read_canonical, utc_now

    if not operation or not isinstance(operation, str):
        raise ValueError("storage observation needs an operation")
    selected = Path(root).expanduser().resolve()
    body = {
        "format": "operational-storage-observation-v1",
        "operation": operation,
        "root": str(selected),
        "target": (
            str(Path(target).expanduser().resolve()) if target is not None else None
        ),
        "storage_checks": checks,
    }
    identity = domain_digest("operational-storage-observation-v1", body)
    path = selected / "execution" / f"storage-{identity}.json"
    if path.exists() or path.is_symlink():
        existing = read_canonical(path)
        if (
            set(existing)
            != {*body, "observation_sha256", "observed_at_utc", "receipt_sha256"}
            or existing.get("observation_sha256") != identity
            or {key: existing.get(key) for key in body} != body
            or existing.get("receipt_sha256")
            != domain_digest(
                "operational-storage-receipt-v1",
                {
                    key: value
                    for key, value in existing.items()
                    if key != "receipt_sha256"
                },
            )
        ):
            raise ValueError(f"conflicting storage observation: {path}")
        return path
    record = {**body, "observation_sha256": identity, "observed_at_utc": utc_now()}
    record["receipt_sha256"] = domain_digest("operational-storage-receipt-v1", record)
    publish_immutable(path, record)
    return path
