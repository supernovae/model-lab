"""Sealed lab records: one writer and one verified reader for every lab surface.

Lab tries (``tries/<id>/try.json``), probe runs (``probes/<id>/probe.json``)
and sweeps (``sweeps/<id>/sweep.json``) are self-hashed JSON documents. ``try``, ``probe``, ``report`` and the dashboard
all read them through :func:`read_lab_record`, so an edited record is rejected
everywhere, not only on the command line.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

DIGEST_FIELD = "record_sha256"
TRY_FORMAT = "klab-lab-try-v1"
PROBE_FORMAT = "klab-probe-v1"
SWEEP_FORMAT = "klab-lab-sweep-v1"
# kind -> (format, folder under the lab root, file name in a record folder)
KINDS = {
    "try": (TRY_FORMAT, "tries", "try.json"),
    "probe": (PROBE_FORMAT, "probes", "probe.json"),
    "sweep": (SWEEP_FORMAT, "sweeps", "sweep.json"),
}
_BY_FORMAT = {fmt: kind for kind, (fmt, _, _) in KINDS.items()}


def is_setup_record(record: Mapping[str, Any]) -> bool:
    """A retained probe setup attempt, without a battery's scoring protocol."""
    return (
        record.get("format") == PROBE_FORMAT
        and record.get("status") in {"started", "failed", "interrupted"}
        and "suite" not in record
    )


def canonical(value: Any) -> bytes:
    from klab.canonical import canonical_json

    return canonical_json(value)


def seal(record: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != DIGEST_FIELD}
    return {**body, DIGEST_FIELD: hashlib.sha256(canonical(body)).hexdigest()}


def comparison_group(body: Mapping[str, Any]) -> str:
    """Digest of everything that must match for two measurements to compare."""
    return hashlib.sha256(canonical(dict(body))).hexdigest()


def eval_group(validation: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    """Held-out losses compare only within one group: same validation data,
    tokenizer, loss mask and eval protocol."""
    return comparison_group(
        {"validation": dict(validation), "protocol": dict(protocol)}
    )


def is_sealed(record: Any) -> bool:
    return isinstance(record, dict) and (
        seal(record)[DIGEST_FIELD] == record.get(DIGEST_FIELD)
    )


def _fsync_directory(path: Path) -> None:
    try:
        directory = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def write_json_atomic(
    path: Path,
    value: Any,
    *,
    exclusive: bool = False,
    may_replace: Callable[[Path], bool] | None = None,
) -> None:
    """Publish VALUE at PATH as one complete file.

    ``exclusive`` never replaces an existing file, except when ``may_replace``
    approves the existing one (e.g. a try's own sealed ``started`` record).
    """
    payload = (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        existing = path.exists() or path.is_symlink()
        if not exclusive:
            os.replace(temporary, path)
        elif existing and may_replace is not None:
            if not may_replace(path):
                raise FileExistsError(f"record already final: {path}")
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # exclusive publish of a complete file
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def jsonable(value: Any) -> Any:
    """JSON-native copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        value = item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_sealed(
    path: Path,
    record: dict[str, Any],
    *,
    may_replace: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    sealed = seal(jsonable(record))
    write_json_atomic(path, sealed, exclusive=True, may_replace=may_replace)
    return sealed


def read_lab_record(path: Path, kind: str | None = None) -> tuple[str, dict[str, Any]]:
    """Read a sealed try or probe record (file or folder) and verify its digest."""
    path = Path(path)
    if path.is_dir():
        names = [KINDS[kind][2]] if kind else [n for _, _, n in KINDS.values()]
        found = [path / name for name in names if (path / name).is_file()]
        if not found:
            raise ValueError(f"no lab record in {path}")
        path = found[0]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"unreadable lab record {path}: {error}") from None
    found_kind = (
        _BY_FORMAT.get(record.get("format")) if isinstance(record, dict) else None
    )
    if found_kind is None or (kind is not None and found_kind != kind):
        expected = {"try": "lab try", "probe": "probe", "sweep": "sweep"}.get(
            kind or "", "lab"
        )
        raise ValueError(f"not a {expected} record: {path}")
    if not is_sealed(record):
        raise ValueError(f"{found_kind} record digest mismatch: {path}")
    return found_kind, record


def iter_lab_records(
    lab_dir: Path, limit: int = 500, kinds: tuple[str, ...] = ("try", "probe")
) -> tuple[list[tuple[str, Path, dict[str, Any]]], list[tuple[Path, str]]]:
    """Verified records of KINDS under LAB_DIR plus (path, reason) for rejected ones."""
    accepted: list[tuple[str, Path, dict[str, Any]]] = []
    rejected: list[tuple[Path, str]] = []
    for kind in kinds:
        _, folder, name = KINDS[kind]
        for path in sorted((lab_dir / folder).glob(f"*/{name}"))[-limit:]:
            try:
                accepted.append((kind, path, read_lab_record(path, kind)[1]))
            except (OSError, ValueError) as error:
                rejected.append((path, str(error)))
    return accepted, rejected
