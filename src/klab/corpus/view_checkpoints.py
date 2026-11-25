"""Receipted append boundaries for the disk corpus release writer.

The staged files become the published build. The journal stays beside them so
checkpoints never require a second copy of the rendered text.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from klab.canonical import canonical_digest, canonical_json
from klab.result_publication import (
    read_canonical,
    sync_directory,
    write_bytes_exclusive,
)
from klab.verification_proofs import ArtifactFingerprint, artifact_fingerprint


@dataclass(frozen=True)
class ViewFileDigest:
    """Whole-file SHA-256 tied to the generation observed by the writer."""

    sha256: str
    size: int
    generation: ArtifactFingerprint

    def inventory_entry(self, path: Path) -> dict[str, str | int]:
        if artifact_fingerprint(path) != self.generation:
            raise ValueError(f"corpus view changed before manifest publication: {path}")
        return {"sha256": self.sha256, "size": self.size}


class ViewOutput:
    """Hash appended bytes as the release writer emits them."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.offset = 0
        self.digest = hashlib.sha256()
        self.whole = hashlib.sha256()

    def write(self, data: bytes) -> None:
        if self.stream.write(data) != len(data):
            raise OSError("short corpus view write")
        self.digest.update(data)
        self.whole.update(data)

    def seal(self) -> dict[str, Any]:
        end = self.stream.tell()
        if end != self.offset:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        return {"end": end, "sha256": self.digest.hexdigest()}

    def advance(self, end: int) -> None:
        self.offset = end
        self.digest = hashlib.sha256()

    def matches(self, part: dict[str, Any], observe: Callable[[bytes], None]) -> bool:
        end = part["end"]
        if type(end) is not int or end < self.offset:
            return False
        self.stream.seek(self.offset)
        remaining = end - self.offset
        digest = hashlib.sha256()
        while remaining:
            if not (block := self.stream.read(min(remaining, 1024**2))):
                return False
            remaining -= len(block)
            digest.update(block)
            observe(block)
        return digest.hexdigest() == part["sha256"]

    def file_digest(self) -> ViewFileDigest:
        descriptor = self.stream.fileno()
        if (
            self.stream.tell() != self.offset
            or os.fstat(descriptor).st_size != self.offset
        ):
            raise ValueError(
                "corpus view has uncommitted bytes at manifest publication"
            )
        return ViewFileDigest(
            self.whole.hexdigest(), self.offset, artifact_fingerprint(descriptor)
        )


class ViewCheckpoints:
    """Commit and restore the ordered writer's frontier under its build lock."""

    def __init__(self, root: Path, identity: str):
        self.root = root
        self.identity = identity
        self.outputs: dict[str, ViewOutput] = {}
        self.sequence = 0
        self.previous: str | None = None
        self.damaged = False

    def add(self, name: str, stream: BinaryIO) -> ViewOutput:
        output = self.outputs[name] = ViewOutput(stream)
        return output

    def restore(self) -> dict[str, Any]:
        """Verify the committed prefix, discard damaged or uncommitted tails."""
        state = {}
        generations = {
            name: artifact_fingerprint(out.stream.fileno())
            for name, out in self.outputs.items()
        }
        for path in sorted(self.root.glob("commit-*.json")):
            # Discard candidate hash state for every file if any range is bad.
            candidates = {name: out.whole.copy() for name, out in self.outputs.items()}
            try:
                entry = read_canonical(path)
                body, digest = entry["body"], entry["sha256"]
                parts = body["outputs"]
                valid = (
                    path.name == self._name()
                    and digest == canonical_digest(body)
                    and body["identity"] == self.identity
                    and body["previous"] == self.previous
                    and isinstance(parts, dict)
                    and parts.keys() == self.outputs.keys()
                    and all(
                        out.matches(parts[name], candidates[name].update)
                        for name, out in self.outputs.items()
                    )
                )
                if not valid:
                    break
                state = body["state"]
            except ValueError, TypeError, KeyError:
                break
            for name, output in self.outputs.items():
                output.whole = candidates[name]
                output.advance(parts[name]["end"])
            self.previous = digest
            self.sequence += 1

        for path in self.root.glob("commit-*.json"):
            if path.name >= self._name():
                path.unlink()
                self.damaged = True
        for name, output in self.outputs.items():
            if artifact_fingerprint(output.stream.fileno()) != generations[name]:
                raise ValueError(
                    f"corpus view changed during checkpoint verification: {name}"
                )
            output.stream.seek(output.offset)
            if os.fstat(output.stream.fileno()).st_size != output.offset:
                output.stream.truncate()
        return state

    def file_digests(self) -> dict[str, ViewFileDigest]:
        return {name: out.file_digest() for name, out in self.outputs.items()}

    def commit(self, state: dict[str, Any]) -> None:
        body = {
            "identity": self.identity,
            "previous": self.previous,
            "state": state,
            "outputs": {name: out.seal() for name, out in self.outputs.items()},
        }
        if self.sequence == 0:
            # The first receipt must survive a crash with the output paths intact.
            directories = {(self.root / "build" / name).parent for name in self.outputs}
            for directory in sorted(directories, key=lambda path: -len(path.parts)):
                sync_directory(directory)
            sync_directory(self.root)
            sync_directory(self.root.parent)
        digest = canonical_digest(body)
        write_bytes_exclusive(
            self.root / self._name(),
            canonical_json({"body": body, "sha256": digest}) + b"\n",
        )
        for name, output in self.outputs.items():
            output.advance(body["outputs"][name]["end"])
        self.previous = digest
        self.sequence += 1

    def _name(self) -> str:
        return f"commit-{self.sequence:012d}.json"
