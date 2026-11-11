"""In-process evidence for immutable prepared arrays; disk metadata is never authority."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from klab.training import manifest as manifest_module
from klab.verification_proofs import (
    ProofStore,
    VerificationMode,
    file_binding,
    validate_mode,
)

_SEAL = object()
_FILES: weakref.WeakSet[VerifiedFile] = weakref.WeakSet()


def _fingerprint(path: Path) -> tuple[int, int, int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"prepared file is not a regular file: {path}")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True, init=False, eq=False)
class VerifiedFile:
    """A SHA proof bound to a regular file and its identity in this process."""

    path: Path
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]
    cold_verified: bool
    _seal: object
    _issuer_pid: int

    def __init__(
        self,
        path: Path,
        sha256: str,
        fingerprint: tuple[int, int, int, int, int, int],
        *,
        _seal: object = None,
        cold_verified: bool = False,
    ) -> None:
        if _seal is not _SEAL:
            raise TypeError("VerifiedFile cannot be constructed from unsigned metadata")
        object.__setattr__(self, "path", path.resolve(strict=True))
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "_seal", _SEAL)
        object.__setattr__(self, "cold_verified", cold_verified)
        object.__setattr__(self, "_issuer_pid", os.getpid())
        _FILES.add(self)

    @property
    def size_bytes(self) -> int:
        return self.fingerprint[3]


def _owned_file(proof: object) -> bool:
    """Reject metadata, reconstructed objects, and inherited cross-process proofs."""
    return (
        isinstance(proof, VerifiedFile)
        and proof in _FILES
        and proof._issuer_pid == os.getpid()
        and proof._seal is _SEAL
    )


class HashingWriter:
    """Hash exactly the bytes handed to the final temporary file."""

    def __init__(self, handle: object) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        count = self.handle.write(data)
        if count != len(data):
            raise OSError("short prepared array write")
        self.digest.update(data)
        self.size += count
        return count


@dataclass(frozen=True, init=False)
class VerifiedPreparedData:
    root: Path
    manifest_sha256: str
    proofs: Mapping[str, VerifiedFile]
    _seal: object

    def __init__(
        self,
        root: Path,
        manifest_sha256: str,
        proofs: dict[str, VerifiedFile],
        *,
        _seal: object = None,
    ) -> None:
        if _seal is not _SEAL:
            raise TypeError(
                "VerifiedPreparedData cannot be constructed from unsigned metadata"
            )
        object.__setattr__(self, "root", root.resolve(strict=True))
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "proofs", MappingProxyType(dict(proofs)))
        object.__setattr__(self, "_seal", _SEAL)


def verify_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    memo: dict[object, VerifiedFile] | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    binding: dict[str, object] | None = None,
) -> VerifiedFile:
    """Hash actual bytes, or reuse a fingerprint-matched same-operation proof."""
    return _verify_file_with_hasher(
        path,
        expected_sha256=expected_sha256,
        memo=memo,
        hash_file=manifest_module.sha256_file,
        proof_store=proof_store,
        verification_mode=verification_mode,
        binding=binding,
    )


def _verify_file_with_hasher(
    path: Path,
    *,
    expected_sha256: str | None = None,
    memo: dict[object, VerifiedFile] | None = None,
    hash_file: Callable[[Path], str],
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    binding: dict[str, object] | None = None,
) -> VerifiedFile:
    """Internal wrapper for instrumenting the actual file hashing path."""
    validate_mode(verification_mode)
    signed_binding = binding
    if signed_binding is None and expected_sha256 is not None:
        signed_binding = file_binding(path, expected_sha256)
    fingerprint = _fingerprint(path)
    canonical = path.resolve(strict=True)
    if signed_binding is not None and (
        signed_binding.get("path") != str(canonical)
        or signed_binding.get("sha256") != expected_sha256
    ):
        raise ValueError("file proof binding differs from requested path or SHA")
    key = (canonical, fingerprint)
    proof = memo.get(key) if memo is not None else None
    if (
        not _owned_file(proof)
        or proof.fingerprint != fingerprint
        or proof.path != canonical
        or (verification_mode == "cold" and not proof.cold_verified)
    ):
        hit = (
            verification_mode == "verified_reuse"
            and proof_store is not None
            and signed_binding is not None
            and proof_store.lookup(signed_binding)
        )
        digest = expected_sha256 if hit else hash_file(path)
        if not hit and proof_store is not None and signed_binding is not None:
            if verification_mode == "verified_reuse":
                proof_store.hashed(signed_binding, fingerprint[3])
            else:
                proof_store.cold(signed_binding, bytes_hashed=fingerprint[3])
        if _fingerprint(path) != fingerprint:
            raise ValueError(f"prepared file changed during hashing: {path}")
        proof = VerifiedFile(
            path, digest, fingerprint, _seal=_SEAL, cold_verified=not hit
        )
        if (
            not hit
            and verification_mode == "verified_reuse"
            and proof_store is not None
            and signed_binding is not None
        ):
            # Reject mismatches before a cold proof can be published.
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError(f"prepared file digest mismatch: {path}")
            proof_store.record(signed_binding, proof)
        if memo is not None:
            memo[key] = proof
    if expected_sha256 is not None and proof.sha256 != expected_sha256:
        raise ValueError(f"prepared file digest mismatch: {path}")
    return proof


def _written_file(path: Path, digest: str, size: int) -> VerifiedFile:
    """Private mint, called only with a digest accumulated by the final writer."""
    fingerprint = _fingerprint(path)
    if fingerprint[3] != size:
        raise ValueError(f"prepared write size changed: {path}")
    return VerifiedFile(path, digest, fingerprint, _seal=_SEAL, cold_verified=True)


def _relocate_proofs(
    root: Path, proofs: dict[str, VerifiedFile]
) -> dict[str, VerifiedFile]:
    """Rebind proofs after atomic directory rename without re-reading array bytes."""
    relocated = {}
    for name, proof in proofs.items():
        path = root / name
        if not _owned_file(proof) or _fingerprint(path) != proof.fingerprint:
            raise ValueError(f"prepared array changed during publication: {path}")
        relocated[name] = VerifiedFile(
            path,
            proof.sha256,
            proof.fingerprint,
            _seal=_SEAL,
            cold_verified=proof.cold_verified,
        )
    return relocated


def required_arrays(
    manifest: dict[str, object],
) -> dict[str, tuple[dict[str, object], str, int]]:
    """Describe exact required files for historical v4/v5 and current v6."""
    version = manifest.get("packing_version")
    if version not in {"contiguous-eos-v4", "contiguous-eos-v5", "contiguous-eos-v6"}:
        raise ValueError("unknown prepared packing version")
    result: dict[str, tuple[dict[str, object], str, int]] = {}
    for split in ("train", "validation"):
        entry = manifest.get(split)
        if not isinstance(entry, dict):
            raise ValueError("missing prepared split metadata")  # noqa: TRY004 - invalid serialized schema
        result[f"{split}.npy"] = (entry, "int32", 1)
    supervision = manifest.get("supervision")
    if version == "contiguous-eos-v4":
        if supervision is not None:
            raise ValueError("v4 cannot carry supervision metadata")
    elif version == "contiguous-eos-v5" or (
        isinstance(supervision, dict)
        and supervision.get("kind") == "token-loss-mask-v1"
    ):
        if (
            not isinstance(supervision, dict)
            or supervision.get("kind") != "token-loss-mask-v1"
            or set(supervision) != {"kind", "train", "validation"}
        ):
            raise ValueError("invalid supervision descriptor")
        for split in ("train", "validation"):
            entry = supervision.get(split)
            if not isinstance(entry, dict) or entry.get("shape") != manifest[split].get(
                "shape"
            ):
                raise ValueError("supervision shape differs from token IDs")
            result[f"{split}_supervision.npy"] = (entry, "bool", 1)
    elif version == "contiguous-eos-v6" and supervision != {"kind": "all_tokens"}:
        raise ValueError("invalid all-token supervision descriptor")
    byte = manifest.get("byte_addressing")
    if byte is not None:
        if not isinstance(byte, dict) or byte.get("kind") != "raw-utf8-suffix-v1":
            raise ValueError("invalid byte-address descriptor")
        for split in ("train", "validation"):
            entry = byte.get(split)
            if not isinstance(entry, dict) or entry.get("shape") != manifest[split].get(
                "shape"
            ):
                raise ValueError("byte address shape differs from token IDs")
            result[f"{split}_byte_addresses.npy"] = (entry, "int32", 1)
    allocation = manifest.get("allocation")
    if allocation is not None:
        if (
            not isinstance(allocation, dict)
            or allocation.get("format") != "klab-prepared-allocation-v1"
        ):
            raise ValueError("invalid allocation descriptor")
        for split in ("train", "validation"):
            section = allocation.get(split)
            if not isinstance(section, dict):
                raise ValueError("invalid allocation split")  # noqa: TRY004 - invalid serialized schema
            for suffix, key, dtype, ndim in (
                ("owner_ids", "owner", "uint8", 1),
                ("semantic_queries", "semantic_queries", "float32", 2),
                ("semantic_mask", "semantic_mask", "bool", 1),
            ):
                if key != "owner" and allocation.get("semantic") is None:
                    continue
                entry = section.get(key)
                if not isinstance(entry, dict):
                    raise ValueError("missing allocation metadata")  # noqa: TRY004 - invalid serialized schema
                result[f"{split}_{suffix}.npy"] = (entry, dtype, ndim)
    return result


def _receipt_from_proofs(
    root: Path, manifest: dict[str, object], proofs: dict[str, VerifiedFile]
) -> VerifiedPreparedData:
    """Validate file proofs against signed manifest, NPY headers, and exact inventory."""
    from klab.canonical import canonical_digest

    if root.is_symlink():
        raise ValueError("prepared root is a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("prepared root is not a directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("prepared manifest is a symlink")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("prepared manifest changed during verification")
    signed = dict(manifest)
    digest = signed.pop("manifest_sha256", None)
    if not isinstance(digest, str) or canonical_digest(signed) != digest:
        raise ValueError("prepared manifest signature mismatch")
    required = required_arrays(manifest)
    if set(proofs) != set(required):
        raise ValueError("prepared array proofs do not match manifest inventory")
    expected = {*required, "manifest.json"}
    if (root / ".klab-cache-owner.json").exists() or (
        root / ".klab-cache-owner.json"
    ).is_symlink():
        expected.add(".klab-cache-owner.json")
    if {entry.name for entry in root.iterdir()} != expected:
        raise ValueError("unexpected prepared file inventory")
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"prepared inventory contains symlink or non-file: {entry}"
            )
    for name, (metadata, dtype, ndim) in required.items():
        path = root / name
        proof = proofs[name]
        if (
            not _owned_file(proof)
            or proof.path != path
            or proof.fingerprint != _fingerprint(path)
        ):
            raise ValueError(f"prepared proof is stale or forged: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = metadata.get("shape")
        if (
            values.ndim != ndim
            or values.dtype != np.dtype(dtype)
            or len(values) == 0
            or not isinstance(shape, list)
            or len(shape) != ndim
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or shape != list(values.shape)
            or type(metadata.get("dtype")) is not str
            or metadata["dtype"] != values.dtype.name
            or type(metadata.get("tokens")) is not int
            or metadata["tokens"] != values.shape[0]
            or type(metadata.get("sha256")) is not str
            or metadata["sha256"] != proof.sha256
            or (
                "size_bytes" in metadata
                and (
                    type(metadata["size_bytes"]) is not int
                    or metadata["size_bytes"] != proof.size_bytes
                )
            )
            or (
                manifest.get("packing_version") == "contiguous-eos-v6"
                and "size_bytes" not in metadata
            )
        ):
            raise ValueError(f"prepared array metadata mismatch: {path}")
        if proof.fingerprint != _fingerprint(path):
            raise ValueError(f"prepared array changed during header validation: {path}")
    return VerifiedPreparedData(root, digest, proofs, _seal=_SEAL)
