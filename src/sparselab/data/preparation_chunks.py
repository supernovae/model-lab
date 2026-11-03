"""Identity-bound, recoverable raw chunks for streaming prepared-data arrays.

A chunk receipt is the commit record: raw files without that receipt belong only
 to the next unfinished chunk. A cold process hashes every committed raw file
before it can use it; final NPY proofs are accumulated during the final write.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np

from sparselab.canonical import canonical_digest, canonical_json
from sparselab.data.verification import (
    HashingWriter,
    VerifiedFile,
    VerifiedPreparedData,
    _receipt_from_proofs,
    _written_file,
    required_arrays,
    verify_file,
)
from sparselab.data.verification import (
    _fingerprint as _verified_fingerprint,
)
from sparselab.result_publication import sync_directory

_after_sealed_chunk: Callable[[str, int, Path], None] | None = None
_after_unsealed_chunk_write: Callable[[str, int, Path], None] | None = None


def _document(path: Path) -> dict[str, object]:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"staging entry is not a regular file: {path}")
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or canonical_json(document) + b"\n" != raw:
        raise ValueError(f"noncanonical staging receipt: {path}")
    checksum = document.pop("checksum", None)
    if not isinstance(checksum, str) or checksum != canonical_digest(document):
        raise ValueError(f"staging receipt checksum mismatch: {path}")
    return document


def _store_document(path: Path, payload: dict[str, object]) -> None:
    complete = {**payload, "checksum": canonical_digest(payload)}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as output:
        output.write(canonical_json(complete) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"staging entry is not a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class PreparationChunks:
    """Own the exact inventory of one <identity>.tmp staging directory."""

    def __init__(
        self,
        root: Path,
        *,
        cache_identity: dict[str, object],
        binding: dict[str, object],
        arrays: dict[str, np.dtype],
        record_limit: int = 4096,
        raw_byte_limit: int = 16_777_216,
        telemetry: object | None = None,
    ) -> None:
        if (
            not arrays
            or set(arrays) - {"ids", "byte_addresses", "supervision"}
            or "ids" not in arrays
        ):
            raise ValueError(
                "chunk arrays must contain ids and optional byte_addresses/supervision"
            )
        if (
            type(record_limit) is not int
            or record_limit < 1
            or type(raw_byte_limit) is not int
            or raw_byte_limit < 1
        ):
            raise ValueError("chunk limits must be positive integers")
        self.root = root
        self.arrays = {key: np.dtype(value) for key, value in arrays.items()}
        if any(dtype.hasobject or dtype.itemsize < 1 for dtype in self.arrays.values()):
            raise ValueError("chunk arrays require fixed-width dtypes")
        self.record_limit = record_limit
        self.raw_byte_limit = raw_byte_limit
        self.telemetry = telemetry
        self._splits: dict[str, SplitChunks] = {}
        owner = {
            "schema_version": 1,
            "cache_identity": cache_identity,
            "binding": binding,
            "arrays": {name: dtype.str for name, dtype in self.arrays.items()},
        }
        self._owner = owner
        self.check_binding_collision(
            root, cache_identity=cache_identity, binding=binding
        )
        if root.is_symlink():
            raise ValueError(f"staging root is a symlink: {root}")
        if root.exists():
            if not root.is_dir():
                raise ValueError(f"staging root is not a directory: {root}")
            marker = root / "staging.json"
            if _document(marker) != owner:
                raise ValueError(f"staging identity or binding mismatch: {root}")
        else:
            root.mkdir(parents=True, exist_ok=False)
            _store_document(root / "staging.json", owner)
        self._inspect_inventory()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        for split in self._splits.values():
            for stream in split._streams.values():
                stream.close()
            split._streams.clear()

    def work_chunks(
        self, split: str, sequence: int, *, telemetry=None
    ) -> PreparationChunks:
        """Exclusive scratch namespace for one bounded, uncommitted worker batch."""
        if (
            split not in {"train", "validation"}
            or type(sequence) is not int
            or sequence < 0
        ):
            raise ValueError("invalid preparation work coordinate")
        root = self.root / f"work-{split}-{sequence:09d}"
        if root.exists() or root.is_symlink():
            raise ValueError(f"preparation work already exists: {root}")
        return PreparationChunks(
            root,
            cache_identity={"parent_owner_sha256": canonical_digest(self._owner)},
            binding={"split": split, "sequence": sequence},
            arrays=self.arrays,
            record_limit=self.record_limit,
            raw_byte_limit=self.raw_byte_limit,
            telemetry=telemetry,
        )

    def discard_work(self, root: Path) -> None:
        """Discard only owned scratch names, never an unexpected file or symlink.

        Worker receipts are not the parent commit frontier. After a crash they
        may reference raw files already adopted into that frontier.
        """
        if root.parent != self.root or not re.fullmatch(
            r"work-(train|validation)-[0-9]{9,}", root.name
        ):
            raise ValueError("invalid preparation work path")
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"invalid preparation work directory: {root}")
        entries = list(root.iterdir())
        for entry in entries:
            allowed = entry.name in {"staging.json", "staging.json.tmp"}
            match = re.fullmatch(r"(train|validation)-[0-9]{6,}\.(.+)", entry.name)
            if match:
                allowed = match[2] in {"receipt.json", "receipt.json.tmp"} | {
                    f"{name}.raw{suffix}"
                    for name in self.arrays
                    for suffix in ("", ".tmp")
                }
            if not allowed or entry.is_symlink() or not entry.is_file():
                raise ValueError(f"unexpected preparation work entry: {entry}")
        for entry in entries:
            entry.unlink()
        root.rmdir()
        sync_directory(self.root)

    def _inspect_inventory(self) -> None:
        for entry in list(self.root.iterdir()):
            if re.fullmatch(r"work-(train|validation)-[0-9]{9,}", entry.name):
                self.discard_work(entry)
        allowed = {"staging.json", "manifest.json", "manifest.json.tmp"}
        for split in ("train", "validation"):
            prefix = f"{split}-"
            receipts = sorted(
                entry
                for entry in self.root.iterdir()
                if entry.name.startswith(prefix)
                and entry.name.endswith(".receipt.json")
            )
            for index, receipt in enumerate(receipts):
                expected = self.root / f"{split}-{index:06d}.receipt.json"
                if receipt != expected:
                    raise ValueError(f"noncontiguous sealed staging chunk: {receipt}")
                payload = _document(receipt)
                self._validate_receipt(split, index, payload)
                allowed.add(receipt.name)
                allowed.update(
                    f"{split}-{index:06d}.{name}.raw" for name in self.arrays
                )
            # The only unsealed material that can be discarded belongs to the next
            # chunk. A receipt temp is untrusted even when its JSON looks complete.
            next_prefix = f"{split}-{len(receipts):06d}."
            allowed.update({next_prefix + "receipt.json.tmp"})
            for name in self.arrays:
                allowed.update(
                    {next_prefix + f"{name}.raw.tmp", next_prefix + f"{name}.raw"}
                )
            for name in self.arrays:
                final = f"{split}{'' if name == 'ids' else '_' + name}.npy"
                allowed.update({final, final.replace(".npy", ".tmp.npy")})
        for entry in self.root.iterdir():
            if entry.name not in allowed or entry.is_symlink() or not entry.is_file():
                raise ValueError(f"unexpected staging entry: {entry}")
        for split in ("train", "validation"):
            count = sum(
                1
                for entry in self.root.iterdir()
                if entry.name.startswith(f"{split}-")
                and entry.name.endswith(".receipt.json")
            )
            next_prefix = f"{split}-{count:06d}."
            for entry in list(self.root.iterdir()):
                if entry.name.startswith(next_prefix):
                    entry.unlink()
        for stale_name in ("manifest.json", "manifest.json.tmp"):
            (self.root / stale_name).unlink(missing_ok=True)
        sync_directory(self.root)

    def _validate_receipt(
        self, split: str, index: int, payload: dict[str, object]
    ) -> None:
        if (
            set(payload)
            != {
                "schema_version",
                "split",
                "index",
                "first_acquired_index",
                "acquired_documents",
                "retained_documents",
                "skipped_documents",
                "truncated_documents",
                "output_tokens",
                "arrays",
            }
            or payload["schema_version"] != 1
            or payload["split"] != split
            or payload["index"] != index
        ):
            raise ValueError(f"invalid staging chunk receipt: {split} {index}")
        for key in (
            "first_acquired_index",
            "acquired_documents",
            "retained_documents",
            "skipped_documents",
            "truncated_documents",
            "output_tokens",
        ):
            if type(payload[key]) is not int or payload[key] < 0:
                raise ValueError(f"invalid staging chunk {key}: {split} {index}")
        if (
            payload["acquired_documents"] < 1
            or payload["acquired_documents"] > self.record_limit
            or payload["retained_documents"] + payload["skipped_documents"]
            != payload["acquired_documents"]
            or payload["truncated_documents"] != 0
        ):
            raise ValueError(f"invalid staging chunk counts: {split} {index}")
        if payload["output_tokens"] < payload["retained_documents"]:
            raise ValueError(f"invalid staging chunk token count: {split} {index}")
        metadata = payload["arrays"]
        if not isinstance(metadata, dict) or set(metadata) != set(self.arrays):
            raise ValueError(f"invalid staging chunk arrays: {split} {index}")
        total = 0
        for name, dtype in self.arrays.items():
            entry = metadata[name]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"dtype", "length", "sha256"}
                or entry["dtype"] != dtype.str
                or type(entry["length"]) is not int
                or entry["length"] < 0
                or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
            ):
                raise ValueError(
                    f"invalid staging raw metadata: {split} {index} {name}"
                )
            if entry["length"] != payload["output_tokens"] * dtype.itemsize:
                raise ValueError(f"staging raw length mismatch: {split} {index} {name}")
            total += entry["length"]
        if total > self.raw_byte_limit:
            raise ValueError(f"staging chunk exceeds raw-byte limit: {split} {index}")

    def cleanup_published(self, published_root: Path) -> None:
        """Remove only this ledger's staging artifacts after atomic publication."""
        if not published_root.is_dir() or published_root.is_symlink():
            raise ValueError(
                f"published root is not a regular directory: {published_root}"
            )
        if _document(published_root / "staging.json") != self._owner:
            raise ValueError("published staging owner changed")
        manifest = published_root / "manifest.json"
        if not stat.S_ISREG(manifest.lstat().st_mode):
            raise ValueError("published manifest is missing")
        expected = {"manifest.json", "staging.json"}
        owned = set()
        for split in ("train", "validation"):
            ledger = self._splits.get(split)
            if ledger is None or ledger._pending["acquired_documents"]:
                raise ValueError(f"split not finalized before publication: {split}")
            for name in self.arrays:
                expected.add(f"{split}{'' if name == 'ids' else '_' + name}.npy")
            for index in range(len(ledger._sealed)):
                owned.add(f"{split}-{index:06d}.receipt.json")
                owned.update(f"{split}-{index:06d}.{name}.raw" for name in self.arrays)
        expected.update(owned)
        present = {entry.name for entry in published_root.iterdir()}
        for entry in published_root.iterdir():
            if (
                entry.name not in expected | {"staging.json.tmp"}
                or entry.is_symlink()
                or not entry.is_file()
            ):
                raise ValueError(f"unexpected published staging entry: {entry}")
        if present - {"staging.json.tmp"} != expected:
            raise ValueError("published staging inventory is incomplete")
        files = {}
        for split, ledger in self._splits.items():
            for index, (payload, _) in enumerate(ledger._sealed):
                receipt_name = f"{split}-{index:06d}.receipt.json"
                raw = (published_root / receipt_name).read_bytes()
                files[receipt_name] = {
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                for name, metadata in payload["arrays"].items():
                    files[f"{split}-{index:06d}.{name}.raw"] = {
                        "size_bytes": metadata["length"],
                        "sha256": metadata["sha256"],
                    }
        document = json.loads(manifest.read_text(encoding="utf-8"))
        cleanup = {"manifest_sha256": document["manifest_sha256"], "files": files}
        (published_root / "staging.json.tmp").unlink(missing_ok=True)
        # Commit cleanup intent before deleting anything; retain owner LAST.
        _store_document(
            published_root / "staging.json", {**self._owner, "cleanup": cleanup}
        )
        self._finish_cleanup(published_root, cleanup, cold=False)

    def _finish_cleanup(
        self, root: Path, cleanup: dict[str, object], *, cold: bool
    ) -> None:
        if (
            not isinstance(cleanup, dict)
            or set(cleanup) != {"manifest_sha256", "files"}
            or not isinstance(cleanup["files"], dict)
        ):
            raise ValueError("invalid published cleanup receipt")
        files = cleanup["files"]
        for name, metadata in files.items():
            match = re.fullmatch(
                r"(train|validation)-[0-9]{6}\.(receipt\.json|ids\.raw|byte_addresses\.raw|supervision\.raw)",
                name,
            )
            if not match or (
                name.endswith("byte_addresses.raw")
                and "byte_addresses" not in self.arrays
            ):
                raise ValueError("invalid owned cleanup filename")
            if (
                not isinstance(metadata, dict)
                or set(metadata) != {"sha256", "size_bytes"}
                or type(metadata["size_bytes"]) is not int
                or metadata["size_bytes"] < 0
                or not isinstance(metadata["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]) is None
            ):
                raise ValueError("invalid owned cleanup metadata")
        final = {"manifest.json", "staging.json"}
        final.update(
            f"{split}{'' if name == 'ids' else '_' + name}.npy"
            for split in ("train", "validation")
            for name in self.arrays
        )
        present = {entry.name for entry in root.iterdir()}
        if not final <= present or present - (
            final | set(files) | {"staging.json.tmp"}
        ):
            raise ValueError("unexpected published cleanup inventory")
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_file():
                raise ValueError(f"unexpected published cleanup entry: {entry}")
        if cold:
            for name in sorted(set(files) & present):
                proof = verify_file(root / name, expected_sha256=files[name]["sha256"])
                if proof.size_bytes != files[name]["size_bytes"]:
                    raise ValueError("published cleanup raw size mismatch")
        for name in sorted(files):
            (root / name).unlink(missing_ok=True)
        (root / "staging.json.tmp").unlink(missing_ok=True)
        sync_directory(root)
        (root / "staging.json").unlink()
        sync_directory(root)

    @staticmethod
    def check_binding_collision(
        root: Path, *, cache_identity: dict[str, object], binding: dict[str, object]
    ) -> None:
        """Reject an interrupted same-operation sibling with changed identity."""
        if root.parent.exists():
            for sibling in root.parent.glob("*.tmp"):
                if sibling == root or sibling.is_symlink() or not sibling.is_dir():
                    continue
                marker = sibling / "staging.json"
                if not marker.exists() and not marker.is_symlink():
                    continue
                other = _document(marker)
                if (
                    other.get("binding") == binding
                    and other.get("cache_identity") != cache_identity
                ):
                    raise ValueError(
                        f"interrupted preparation for the same source has changed identity: {sibling}"
                    )

    @classmethod
    def recover_published(
        cls,
        root: Path,
        *,
        cache_identity: dict[str, object],
        binding: dict[str, object],
        arrays: dict[str, np.dtype],
        record_limit: int = 4096,
        raw_byte_limit: int = 16_777_216,
        telemetry: object | None = None,
    ) -> VerifiedPreparedData:
        """Complete cleanup after a crash between atomic publish and cleanup.

        No file is removed until both sealed chunks and published scientific
        arrays have passed independent cold hash verification.
        """
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"invalid published staging root: {root}")
        owner = cls.__new__(cls)
        owner.root = root
        owner.arrays = {name: np.dtype(dtype) for name, dtype in arrays.items()}
        owner.record_limit = record_limit
        owner.raw_byte_limit = raw_byte_limit
        owner.telemetry = telemetry
        owner._splits = {}
        owner._owner = {
            "schema_version": 1,
            "cache_identity": cache_identity,
            "binding": binding,
            "arrays": {name: dtype.str for name, dtype in owner.arrays.items()},
        }
        marker = _document(root / "staging.json")
        cleanup = marker.pop("cleanup", None)
        if marker != owner._owner:
            raise ValueError("published staging identity or binding mismatch")
        if cleanup is None:
            for split in ("train", "validation"):
                owner.open_split(split)
        manifest_path = root / "manifest.json"
        if not stat.S_ISREG(manifest_path.lstat().st_mode):
            raise ValueError("published manifest is not a regular file")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("published manifest must be an object")  # noqa: TRY004 - invalid serialized schema
        signed = dict(manifest)
        signature = signed.pop("manifest_sha256", None)
        if (
            not isinstance(signature, str)
            or canonical_digest(signed) != signature
            or manifest.get("cache_identity") != cache_identity
            or manifest.get("settings_sha256") != canonical_digest(cache_identity)
        ):
            raise ValueError("published staging manifest or identity mismatch")
        required = required_arrays(manifest)
        expected_arrays = {
            f"{split}{'' if name == 'ids' else '_' + name}.npy"
            for split in ("train", "validation")
            for name in owner.arrays
        }
        if set(required) != expected_arrays:
            raise ValueError("published staging arrays do not match owner")
        if cleanup is not None:
            if (
                not isinstance(cleanup, dict)
                or cleanup.get("manifest_sha256") != signature
            ):
                raise ValueError("published cleanup manifest identity mismatch")
        else:
            for split in ("train", "validation"):
                entry = manifest[split]
                if not isinstance(entry, dict) or any(
                    entry.get(key) != value
                    for key, value in owner._splits[split].state.items()
                ):
                    raise ValueError(
                        f"published staging scientific counts mismatch: {split}"
                    )
        proofs = {}
        for name, (metadata, dtype, ndim) in required.items():
            started = time.monotonic()
            proof = verify_file(root / name, expected_sha256=metadata.get("sha256"))
            values = np.load(root / name, mmap_mode="r", allow_pickle=False)
            if (
                values.ndim != ndim
                or values.dtype != np.dtype(dtype)
                or metadata.get("shape") != list(values.shape)
                or metadata.get("dtype") != values.dtype.name
                or metadata.get("tokens") != values.shape[0]
                or metadata.get("size_bytes") != proof.size_bytes
                or values.shape[0] == 0
                or _verified_fingerprint(root / name) != proof.fingerprint
            ):
                raise ValueError(f"published staging array metadata mismatch: {name}")
            proofs[name] = proof
            if telemetry is not None:
                telemetry.add(
                    "deep_verification_hash_seconds", time.monotonic() - started
                )
        if cleanup is None:
            owner.cleanup_published(root)
        else:
            owner._finish_cleanup(root, cleanup, cold=True)
        return _receipt_from_proofs(root, manifest, proofs)

    def open_split(self, split: str) -> SplitChunks:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown preparation split: {split}")
        if split not in self._splits:
            self._splits[split] = SplitChunks(self, split)
        return self._splits[split]


class SplitChunks:
    def __init__(self, owner: PreparationChunks, split: str) -> None:
        self.owner = owner
        self.telemetry = owner.telemetry
        self.split = split
        self.root = owner.root
        self._sealed: list[
            tuple[dict[str, object], dict[str, tuple[int, int, int, int]]]
        ] = []
        state = {
            key: 0
            for key in (
                "acquired_documents",
                "retained_documents",
                "skipped_documents",
                "truncated_documents",
                "output_tokens",
            )
        }
        for index in range(1_000_000_000):
            receipt = self._receipt_path(index)
            if not receipt.exists():
                break
            payload = _document(receipt)
            owner._validate_receipt(split, index, payload)
            if payload["first_acquired_index"] != state["acquired_documents"] + 1:
                raise ValueError(
                    f"noncontiguous acquired records in staging: {receipt}"
                )
            fingerprints = {}
            for name in owner.arrays:
                path = self._raw_path(index, name)
                before = _fingerprint(path)
                if before[2] != payload["arrays"][name]["length"]:
                    raise ValueError(f"staging raw length mismatch: {path}")
                started = time.monotonic()
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                if (
                    _fingerprint(path) != before
                    or digest.hexdigest() != payload["arrays"][name]["sha256"]
                ):
                    raise ValueError(f"sealed staging chunk changed: {path}")
                fingerprints[name] = before
                if owner.telemetry is not None:
                    owner.telemetry.add(
                        "deep_verification_hash_seconds", time.monotonic() - started
                    )
            self._sealed.append((payload, fingerprints))
            for key in state:
                state[key] += payload[key]
        self._state = state
        self._pending = {key: 0 for key in state}
        self._streams: dict[str, object] = {}
        self._hashes: dict[str, object] = {}
        self._sizes = {name: 0 for name in owner.arrays}

    @property
    def state(self) -> dict[str, int]:
        """Only committed records count toward resume's tokenizer-free source skip."""
        return dict(self._state)

    def _prefix(self, index: int) -> str:
        return f"{self.split}-{index:06d}"

    def _receipt_path(self, index: int) -> Path:
        return self.root / f"{self._prefix(index)}.receipt.json"

    def _raw_path(self, index: int, name: str) -> Path:
        return self.root / f"{self._prefix(index)}.{name}.raw"

    def append_record(
        self, acquired_index: int, values: dict[str, list[int] | np.ndarray] | None
    ) -> None:
        if (
            type(acquired_index) is not int
            or acquired_index
            != self._state["acquired_documents"]
            + self._pending["acquired_documents"]
            + 1
        ):
            raise ValueError(
                f"noncontiguous acquired record for {self.split}: {acquired_index}"
            )
        data = None
        if values is not None:
            if set(values) != set(self.owner.arrays):
                raise ValueError("record sidecars do not match chunk arrays")
            data = {}
            for name, dtype in self.owner.arrays.items():
                array = np.asarray(values[name], dtype=dtype)
                if array.ndim != 1:
                    raise ValueError("chunk record arrays must be one-dimensional")
                data[name] = array.tobytes()
            if not data["ids"] or any(
                len(raw) // self.owner.arrays[name].itemsize
                != len(data["ids"]) // self.owner.arrays["ids"].itemsize
                for name, raw in data.items()
            ):
                raise ValueError("chunk record sidecar lengths must match nonempty ids")
            additional = sum(map(len, data.values()))
            if additional > self.owner.raw_byte_limit:
                raise ValueError(
                    f"{self.split} encoded document exceeds chunk raw-byte cap ({additional} > {self.owner.raw_byte_limit}); lower the source/token cap"
                )
        else:
            additional = 0
        if self._pending["acquired_documents"] and (
            self._pending["acquired_documents"] == self.owner.record_limit
            or sum(self._sizes.values()) + additional > self.owner.raw_byte_limit
        ):
            self.seal()
        if not self._streams:
            index = len(self._sealed)
            self._streams = {
                name: self._raw_path(index, name)
                .with_name(self._raw_path(index, name).name + ".tmp")
                .open("xb")
                for name in self.owner.arrays
            }
            self._hashes = {name: hashlib.sha256() for name in self.owner.arrays}
        self._pending["acquired_documents"] += 1
        if data is None:
            self._pending["skipped_documents"] += 1
        else:
            self._pending["retained_documents"] += 1
            self._pending["output_tokens"] += (
                len(data["ids"]) // self.owner.arrays["ids"].itemsize
            )
            started = time.monotonic()
            for name, raw in data.items():
                if self._streams[name].write(raw) != len(raw):
                    raise OSError("short staging raw write")
                self._hashes[name].update(raw)
                self._sizes[name] += len(raw)
                if _after_unsealed_chunk_write is not None:
                    self._streams[name].flush()
                    os.fsync(self._streams[name].fileno())
                    _after_unsealed_chunk_write(
                        self.split, len(self._sealed), Path(self._streams[name].name)
                    )
            if self.telemetry is not None:
                self.telemetry.add("spool_write_seconds", time.monotonic() - started)

    def seal(self) -> None:
        if not self._pending["acquired_documents"]:
            return
        index = len(self._sealed)
        started = time.monotonic()
        metadata = {}
        fingerprints = {}
        for name, dtype in self.owner.arrays.items():
            stream = self._streams[name]
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            path = self._raw_path(index, name)
            Path(stream.name).replace(path)
            metadata[name] = {
                "dtype": dtype.str,
                "length": self._sizes[name],
                "sha256": self._hashes[name].hexdigest(),
            }
            fingerprints[name] = _fingerprint(path)
        sync_directory(self.root)
        receipt = self._receipt_path(index)
        payload = {
            "schema_version": 1,
            "split": self.split,
            "index": index,
            "first_acquired_index": self._state["acquired_documents"] + 1,
            **self._pending,
            "arrays": metadata,
        }
        self.owner._validate_receipt(self.split, index, payload)
        _store_document(receipt, payload)
        self._sealed.append((payload, fingerprints))
        for key in self._state:
            self._state[key] += self._pending[key]
        self._pending = {key: 0 for key in self._state}
        self._streams = {}
        self._hashes = {}
        self._sizes = {name: 0 for name in self.owner.arrays}
        if self.telemetry is not None:
            self.telemetry.add("finalize_fsync_seconds", time.monotonic() - started)
        if _after_sealed_chunk is not None:
            _after_sealed_chunk(self.split, index, receipt)

    def adopt(self, work: SplitChunks, *, first_acquired_index: int) -> None:
        """Commit independently written chunks in source order without copying bytes."""
        if (
            self._pending["acquired_documents"]
            or work._pending["acquired_documents"]
            or work.owner.root.parent != self.root
            or work.split != self.split
            or work.owner.arrays != self.owner.arrays
            or work.owner._owner["cache_identity"]
            != {"parent_owner_sha256": canonical_digest(self.owner._owner)}
            or not re.fullmatch(f"work-{self.split}-[0-9]{{9,}}", work.root.name)
            or work.root.is_symlink()
            or _document(work.root / "staging.json") != work.owner._owner
            or first_acquired_index != self._state["acquired_documents"] + 1
        ):
            raise ValueError(
                "worker chunks do not extend the committed source frontier"
            )
        for source_index, (payload, fingerprints) in enumerate(work._sealed):
            if _document(work._receipt_path(source_index)) != payload:
                raise ValueError("worker receipt changed before adoption")
            index = len(self._sealed)
            committed = {
                **payload,
                "index": index,
                "first_acquired_index": first_acquired_index
                + payload["first_acquired_index"]
                - 1,
            }
            self.owner._validate_receipt(self.split, index, committed)
            if (
                committed["first_acquired_index"]
                != self._state["acquired_documents"] + 1
            ):
                raise ValueError("worker chunk source frontier is not contiguous")
            for name in self.owner.arrays:
                source = work._raw_path(source_index, name)
                destination = self._raw_path(index, name)
                if _fingerprint(source) != fingerprints[name]:
                    raise ValueError("worker raw chunk changed before adoption")
                if destination.exists() or destination.is_symlink():
                    raise ValueError("worker adoption destination already exists")
                source.rename(destination)
                if _fingerprint(destination) != fingerprints[name]:
                    raise ValueError("worker raw chunk changed during adoption")
            sync_directory(self.root)
            _store_document(self._receipt_path(index), committed)
            self._sealed.append((committed, fingerprints))
            for key in self._state:
                self._state[key] += committed[key]
            if _after_sealed_chunk is not None:
                _after_sealed_chunk(self.split, index, self._receipt_path(index))
        self.owner.discard_work(work.root)

    def finish(self) -> tuple[dict[str, VerifiedFile], dict[str, int]]:
        self.seal()
        proofs = {}
        for name, dtype in self.owner.arrays.items():
            basename = f"{self.split}{'' if name == 'ids' else '_' + name}.npy"
            path = self.root / basename
            temporary = self.root / basename.replace(".npy", ".tmp.npy")
            # A previous unfinished finalization is never authority. It is safe
            # to replace these exact owned outputs only, after sealed validation.
            for stale in (path, temporary):
                if stale.exists() or stale.is_symlink():
                    if not stat.S_ISREG(stale.lstat().st_mode):
                        raise ValueError(f"unexpected staging output: {stale}")
                    stale.unlink()
            count = self._state["output_tokens"]
            with temporary.open("xb") as output:
                writer = HashingWriter(output)
                np.lib.format.write_array_header_1_0(
                    writer,
                    {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": (count,),
                    },
                )
                copy_started = time.monotonic()
                for index, (receipt, fingerprints) in enumerate(self._sealed):
                    source_path = self._raw_path(index, name)
                    if _fingerprint(source_path) != fingerprints[name]:
                        raise ValueError(
                            f"sealed staging chunk changed before finalization: {source_path}"
                        )
                    size = 0
                    with source_path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            writer.write(block)
                            size += len(block)
                    if (
                        _fingerprint(source_path) != fingerprints[name]
                        or size != receipt["arrays"][name]["length"]
                    ):
                        raise ValueError(
                            f"sealed staging chunk changed during finalization: {source_path}"
                        )
                if self.telemetry is not None:
                    self.telemetry.add(
                        "spool_write_seconds", time.monotonic() - copy_started
                    )
                sync_started = time.monotonic()
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
            sync_directory(self.root)
            if self.telemetry is not None:
                self.telemetry.add(
                    "finalize_fsync_seconds", time.monotonic() - sync_started
                )
            proofs[basename] = _written_file(
                path, writer.digest.hexdigest(), writer.size
            )
        stats = {
            **self.state,
            "artifact_bytes": sum(proof.size_bytes for proof in proofs.values()),
            "artifact_files": len(proofs),
        }
        if self.telemetry is not None:
            self.telemetry.logical_output_bytes += stats["artifact_bytes"]
        return proofs, stats
