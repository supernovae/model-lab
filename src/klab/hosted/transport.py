"""Finite transfers through the official Colab CLI Contents API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from klab.runtime_env_subprocess import run_bounded
from klab.workdir import ensure_scratch_dir

_CHUNK_BYTES = 16 * 1024**2
_OUTPUT_LIMIT = 32 * 1024
_MAX_PARTS = 4096


@dataclass(frozen=True)
class TransferManifest:
    destination: str
    sha256: str
    size: int
    parts: int

    def __post_init__(self) -> None:
        _remote_path(self.destination)
        if (
            type(self.size) is not int
            or not 0 <= self.size <= _MAX_PARTS * _CHUNK_BYTES
        ):
            raise ValueError("transfer byte count exceeds bound")
        if type(self.parts) is not int or self.parts != max(
            1, (self.size + _CHUNK_BYTES - 1) // _CHUNK_BYTES
        ):
            raise ValueError("transfer shard count differs from declared bytes")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("transfer digest must be lowercase SHA-256")


def _remote_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not Path(value).is_absolute()
        or ".." in Path(value).parts
        or "\0" in value
    ):
        raise ValueError("remote path must be absolute and non-traversing")


def assembly_python(manifest: TransferManifest) -> str:
    """Assemble exact-sized regular shards without overwriting an existing file."""
    payload = json.dumps(manifest.__dict__, sort_keys=True)
    return f"""import hashlib, json, os, pathlib, time
m = json.loads({payload!r})
target = pathlib.Path(m["destination"])
if not target.is_absolute() or ".." in target.parts or any(p.is_symlink() for p in (target, *target.parents)): raise RuntimeError("unsafe transfer destination")
temporary = target.with_name("." + target.name + ".assembling")
deadline = globals().get("_klab_deadline")
digest = hashlib.sha256()
written = 0
try:
 descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
 with os.fdopen(descriptor, "wb") as output:
  for i in range(m["parts"]):
   if deadline is not None and time.monotonic() >= deadline: raise TimeoutError("transfer deadline exhausted")
   part = pathlib.Path(str(target) + f".part-{{i:08d}}")
   expected = min(16 * 1024**2, max(0, m["size"] - i * 16 * 1024**2))
   if part.is_symlink() or not part.is_file() or part.stat().st_size != expected: raise RuntimeError("invalid transfer shard")
   remaining = expected
   with part.open("rb") as incoming:
    while data := incoming.read(min(1024 * 1024, remaining + 1)):
     if deadline is not None and time.monotonic() >= deadline: raise TimeoutError("transfer deadline exhausted")
     if len(data) > remaining: raise RuntimeError("transfer shard grew")
     remaining -= len(data); written += len(data); digest.update(data); output.write(data)
   if remaining: raise RuntimeError("transfer shard shrank")
  output.flush(); os.fsync(output.fileno())
 if written != m["size"] or digest.hexdigest() != m["sha256"]: raise RuntimeError("transfer digest mismatch")
 os.link(temporary, target)
 for i in range(m["parts"]): pathlib.Path(str(target) + f".part-{{i:08d}}").unlink()
finally:
 temporary.unlink(missing_ok=True)
"""


class ColabFileTransport:
    """Contents operations only; this class never executes notebook code."""

    def __init__(
        self,
        session: str,
        *,
        config_path: Path | None = None,
        auth: str = "oauth2",
        timeout: float = 60,
        deadline: float | None = None,
    ) -> None:
        if (
            type(timeout) not in {int, float}
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        self.session, self.config_path, self.auth = session, config_path, auth
        self.timeout = timeout
        self.deadline = deadline if deadline is not None else time.monotonic() + timeout

    def _base(self) -> list[str]:
        command = ["colab"]
        if self.config_path is not None:
            command.extend(["--config", str(self.config_path)])
        return command + ["--auth", self.auth]

    def _run(self, command: list[str]) -> None:
        timeout = min(self.timeout, self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Colab transfer exceeded shared deadline")
        result = run_bounded(command, timeout=timeout, output_limit=_OUTPUT_LIMIT)
        if result.returncode:
            raise ValueError(
                result.stderr.decode("utf-8", "replace")[:1024]
                or "Colab CLI transfer failed"
            )

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def upload(self, source: Path, destination: str) -> TransferManifest:
        if not source.is_file() or source.is_symlink():
            raise ValueError("upload source must be a regular file")
        size = source.stat().st_size
        if size > _MAX_PARTS * _CHUNK_BYTES:
            raise ValueError("upload exceeds bounded shard count")
        manifest = TransferManifest(
            destination,
            self._digest(source),
            size,
            max(1, (size + _CHUNK_BYTES - 1) // _CHUNK_BYTES),
        )
        with (
            tempfile.TemporaryDirectory(
                prefix="colab-transfer-", dir=ensure_scratch_dir()
            ) as directory,
            source.open("rb") as incoming,
        ):
            for index in range(manifest.parts):
                chunk = Path(directory) / f"{index:08d}"
                chunk.write_bytes(incoming.read(_CHUNK_BYTES))
                chunk.chmod(0o600)
                self._run(
                    self._base()
                    + [
                        "upload",
                        "-s",
                        self.session,
                        str(chunk),
                        f"{destination}.part-{index:08d}",
                    ]
                )
                chunk.unlink()
        return manifest

    def upload_exact(self, source: Path, destination: str) -> None:
        _remote_path(destination)
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size > _CHUNK_BYTES
        ):
            raise ValueError("exact upload requires a regular file of at most 16 MiB")
        self._run(
            self._base() + ["upload", "-s", self.session, str(source), destination]
        )

    def download(self, source: str, destination: Path) -> None:
        """Accept only bounded fixed control files; bulk artifacts use the relay."""
        _remote_path(source)
        if not destination.is_absolute():
            raise ValueError("download destination must be absolute")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("owned download staging path already exists")
        try:
            self._run(
                self._base() + ["download", "-s", self.session, source, str(temporary)]
            )
            if (
                temporary.is_symlink()
                or not temporary.is_file()
                or temporary.stat().st_size > _CHUNK_BYTES
            ):
                raise ValueError("Colab control download is unsafe or exceeds 16 MiB")
            temporary.chmod(0o600)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def remove(self, source: str) -> None:
        """Remove a caller-owned request path through Contents, never a runtime."""
        _remote_path(source)
        self._run(self._base() + ["rm", "-s", self.session, source])
