"""Verified ownership-allocation manifests and immutable sidecar arrays."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from klab.canonical import canonical_digest, canonical_json
from klab.training.manifest import sha256_file

ALLOCATION_FORMAT = "klab-allocation-manifest"
ALLOCATION_VERSION = 1
OWNER_NEURAL = 0
OWNER_LEXICAL = 1
OWNER_SEMANTIC = 2
OWNER_HYBRID = 3
OWNER_CODES = {
    "neural": OWNER_NEURAL,
    "lexical": OWNER_LEXICAL,
    "semantic": OWNER_SEMANTIC,
    "hybrid": OWNER_HYBRID,
}

ALLOCATION_REGIMES = frozenset(
    {"iso-neural", "iso-total", "iso-active", "iso-token", "iso-flop"}
)
OWNERSHIP_PROFILES = frozenset({"n100", "n75", "n50", "n25", "n0"})


@dataclass(frozen=True)
class AllocationSidecars:
    owner: np.ndarray
    semantic_queries: np.ndarray | None
    semantic_mask: np.ndarray | None


@dataclass(frozen=True)
class AllocationManifest:
    """A content-addressed allocation declaration, independent of prepared caches.

    Schema v1 requires ``source_identity_sha256``, ``tokenizer_sha256``, and
    ``splits.{train,validation}.owner``.  Array descriptors contain a safe
    relative ``path``, ``sha256``, ``dtype``, and ``shape``.  Semantic sidecars
    are optional as a group and, when present, require a verified pack identity.
    """

    path: Path
    payload: dict[str, Any]
    sha256: str

    @property
    def semantic(self) -> dict[str, Any] | None:
        value = self.payload.get("semantic")
        return value if isinstance(value, dict) else None

    def split(self, name: str, *, token_count: int) -> AllocationSidecars:
        splits = self.payload["splits"]
        assert isinstance(splits, dict)
        spec = splits[name]
        assert isinstance(spec, dict)
        owner = _load_array(
            self.path.parent, spec["owner"], np.dtype("uint8"), token_count
        )
        if np.any(owner > OWNER_HYBRID):
            raise ValueError("allocation owner sidecar contains an unknown owner code")
        semantic = self.semantic
        if semantic is None:
            if np.any(np.isin(owner, (OWNER_SEMANTIC, OWNER_HYBRID))):
                raise ValueError(
                    "semantic or hybrid ownership requires semantic sidecars"
                )
            return AllocationSidecars(owner, None, None)
        queries = _load_array(
            self.path.parent,
            spec["semantic_queries"],
            np.dtype("float32"),
            token_count,
            dimensions=2,
        )
        mask = _load_array(
            self.path.parent, spec["semantic_mask"], np.dtype(bool), token_count
        )
        if not np.isfinite(queries).all():
            raise ValueError("semantic query sidecar contains non-finite values")
        if mask[-1] or np.any(
            mask[:-1] & ~np.isin(owner[1:], (OWNER_SEMANTIC, OWNER_HYBRID))
        ):
            raise ValueError(
                "semantic query mask may select only semantic or hybrid target ownership"
            )
        return AllocationSidecars(owner, queries, mask)


def _safe_relative(root: Path, value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("allocation sidecar path must be a nonempty relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("allocation sidecar path must stay below its manifest")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("allocation bundle root must be a nonsymlink directory")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("allocation sidecar path must not traverse symlinks")
    try:
        path = candidate.resolve(strict=True)
        root_path = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("allocation sidecar is missing or inaccessible") from error
    if not path.is_relative_to(root_path) or (
        not path.is_dir() if directory else not path.is_file()
    ):
        raise ValueError("allocation sidecar is missing or escapes its manifest")
    return path


def _load_array(
    root: Path,
    descriptor: object,
    dtype: np.dtype[Any],
    token_count: int,
    *,
    dimensions: int = 1,
) -> np.ndarray:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "sha256",
        "dtype",
        "shape",
    }:
        raise ValueError(
            "allocation sidecar descriptor must contain path, sha256, dtype, shape"
        )
    path = _safe_relative(root, descriptor["path"])
    if (
        not isinstance(descriptor["sha256"], str)
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise ValueError(f"allocation sidecar digest mismatch: {path.name}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or values.dtype != dtype
        or values.ndim != dimensions
        or list(values.shape) != shape
        or values.shape[0] != token_count
    ):
        raise ValueError(
            "allocation sidecar dtype or shape does not match packed tokens"
        )
    return values


def load_allocation_manifest(
    path: Path, *, source_identity_sha256: str, tokenizer_sha256: str
) -> AllocationManifest:
    """Load a v1 allocation manifest and validate all immutable declarations."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("allocation manifest must be a regular nonsymlink file")
    resolved = path.resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("allocation manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("allocation manifest must be an object")
    digest = payload.pop("sha256", None)
    required = {
        "format",
        "version",
        "source_identity_sha256",
        "tokenizer_sha256",
        "corpus",
        "splits",
    }
    optional = {"semantic", "resource_regime", "ownership_profile"}
    if (
        set(payload) - optional != required
        or payload.get("format") != ALLOCATION_FORMAT
        or payload.get("version") != ALLOCATION_VERSION
    ):
        raise ValueError("unsupported allocation manifest schema")
    if not isinstance(digest, str) or canonical_digest(payload) != digest:
        raise ValueError("allocation manifest digest mismatch")
    if (
        payload["source_identity_sha256"] != source_identity_sha256
        or payload["tokenizer_sha256"] != tokenizer_sha256
    ):
        raise ValueError("allocation manifest source or tokenizer identity mismatch")
    corpus = payload["corpus"]
    if (
        not isinstance(corpus, dict)
        or set(corpus) != {"train_jsonl_sha256", "validation_jsonl_sha256"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or value.lower() != value
            or any(char not in "0123456789abcdef" for char in value)
            for value in corpus.values()
        )
    ):
        raise ValueError(
            "allocation corpus requires lowercase train/validation SHA-256 digests"
        )
    regime = payload.get("resource_regime")
    profile = payload.get("ownership_profile")
    if (regime is None) != (profile is None):
        raise ValueError(
            "resource regime and ownership profile must be declared together"
        )
    if regime is not None and (
        not isinstance(regime, str)
        or not isinstance(profile, str)
        or regime not in ALLOCATION_REGIMES
        or profile not in OWNERSHIP_PROFILES
    ):
        raise ValueError("allocation resource regime or ownership profile is invalid")
    splits = payload["splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("allocation manifest must declare train and validation splits")
    semantic = payload.get("semantic")
    if semantic is not None:
        if not isinstance(semantic, dict) or set(semantic) != {
            "pack_path",
            "pack_sha256",
            "pack_id",
            "key_encoder",
        }:
            raise ValueError(
                "semantic allocation requires verified pack and encoder identity"
            )
        pack = _safe_relative(resolved.parent, semantic["pack_path"], directory=True)
        pack_manifest = _safe_relative(pack, "manifest.json")
        if (
            not isinstance(semantic["pack_sha256"], str)
            or sha256_file(pack_manifest) != semantic["pack_sha256"]
        ):
            raise ValueError("semantic allocation pack digest mismatch")
        if not isinstance(semantic["pack_id"], str) or not isinstance(
            semantic["key_encoder"], dict
        ):
            raise ValueError("semantic allocation identity is invalid")
    for name, spec in splits.items():
        expected_keys = (
            {"owner"}
            if semantic is None
            else {"owner", "semantic_queries", "semantic_mask"}
        )
        if not isinstance(spec, dict) or set(spec) != expected_keys:
            raise ValueError(f"allocation split {name} has unsupported sidecars")
    payload["sha256"] = digest
    return AllocationManifest(resolved, payload, digest)


def load_semantic_retriever(manifest: AllocationManifest):
    """Construct only a verified retriever and re-check its declared identity."""
    semantic = manifest.semantic
    if semantic is None:
        return None
    from klab.engram.semantic import SemanticRetriever

    retriever = SemanticRetriever.from_pack(
        manifest.path.parent / semantic["pack_path"],
        expected_pack_id=semantic["pack_id"],
    )
    if retriever.key_encoder.model_dump(mode="json") != semantic["key_encoder"]:
        raise ValueError("semantic pack key encoder differs from allocation manifest")
    return retriever


def copy_allocation_bundle(
    manifest: AllocationManifest, destination: Path
) -> AllocationManifest:
    """Copy one verified allocation manifest and only its referenced assets."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"allocation destination already exists: {destination}")
    destination.mkdir(parents=True)
    root = manifest.path.parent
    relative_members: set[Path] = set()

    def target_for(relative: Path) -> Path:
        if any(
            relative == existing
            or relative in existing.parents
            or existing in relative.parents
            for existing in relative_members
        ):
            raise ValueError("allocation bundle contains colliding member paths")
        relative_members.add(relative)
        return destination / relative

    for split in ("train", "validation"):
        spec = manifest.payload["splits"][split]
        for descriptor in spec.values():
            source = _safe_relative(root, descriptor["path"])
            relative = Path(descriptor["path"])
            target = target_for(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    semantic = manifest.semantic
    if semantic is not None:
        source_pack = _safe_relative(root, semantic["pack_path"], directory=True)
        if any(member.is_symlink() for member in source_pack.rglob("*")):
            raise ValueError("semantic pack contains a symlink")
        relative_pack = Path(semantic["pack_path"])
        target_pack = target_for(relative_pack)
        target_pack.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_pack, target_pack)
    target_manifest = target_for(Path(manifest.path.name))
    shutil.copy2(manifest.path, target_manifest)
    copied = load_allocation_manifest(
        target_manifest,
        source_identity_sha256=manifest.payload["source_identity_sha256"],
        tokenizer_sha256=manifest.payload["tokenizer_sha256"],
    )
    for split in ("train", "validation"):
        shape = manifest.payload["splits"][split]["owner"]["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] <= 0
        ):
            raise ValueError("allocation owner descriptor has an invalid shape")
        copied.split(split, token_count=shape[0])
    if copied.semantic is not None:
        load_semantic_retriever(copied)
    return copied


def build_allocation_manifest(
    path: Path,
    *,
    source_identity_sha256: str,
    tokenizer_sha256: str,
    train_jsonl_sha256: str,
    validation_jsonl_sha256: str,
    train_owner: np.ndarray,
    validation_owner: np.ndarray,
    semantic: dict[str, Any] | None = None,
    train_queries: np.ndarray | None = None,
    validation_queries: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    validation_mask: np.ndarray | None = None,
    resource_regime: str | None = None,
    ownership_profile: str | None = None,
    sidecar_prefix: str | None = None,
) -> AllocationManifest:
    """Write a canonical v1 manifest for already token-aligned sidecars."""
    sidecars = (train_queries, validation_queries, train_mask, validation_mask)
    if semantic is None:
        if any(value is not None for value in sidecars):
            raise ValueError("semantic sidecars require verified semantic metadata")
    elif any(value is None for value in sidecars):
        raise ValueError("semantic metadata requires both splits' queries and masks")
    if (resource_regime is None) != (ownership_profile is None):
        raise ValueError(
            "resource regime and ownership profile must be declared together"
        )
    if resource_regime is not None and (
        not isinstance(resource_regime, str)
        or not isinstance(ownership_profile, str)
        or resource_regime not in ALLOCATION_REGIMES
        or ownership_profile not in OWNERSHIP_PROFILES
    ):
        raise ValueError("allocation resource regime or ownership profile is invalid")
    if sidecar_prefix is not None and (
        not sidecar_prefix
        or Path(sidecar_prefix).name != sidecar_prefix
        or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for char in sidecar_prefix
        )
    ):
        raise ValueError("allocation sidecar prefix must be a safe lowercase filename")
    for owner in (train_owner, validation_owner):
        if owner.ndim != 1 or owner.dtype != np.dtype("uint8") or owner.size == 0:
            raise ValueError(
                "owner sidecars must be nonempty one-dimensional uint8 arrays"
            )
        if np.any(owner > OWNER_HYBRID):
            raise ValueError("allocation owner sidecar contains an unknown owner code")
    for queries, mask, owner in (
        (train_queries, train_mask, train_owner),
        (validation_queries, validation_mask, validation_owner),
    ):
        if queries is None or mask is None:
            continue
        if (
            queries.ndim != 2
            or queries.shape[0] != owner.shape[0]
            or queries.shape[1] <= 0
            or queries.dtype != np.dtype("float32")
            or mask.ndim != 1
            or mask.dtype != np.dtype(bool)
            or mask.shape[0] != owner.shape[0]
            or not np.isfinite(queries).all()
        ):
            raise ValueError("semantic query sidecars must be finite and token-aligned")
        if mask[-1] or np.any(
            mask[:-1] & ~np.isin(owner[1:], (OWNER_SEMANTIC, OWNER_HYBRID))
        ):
            raise ValueError("semantic query mask disagrees with target ownership")
    path.parent.mkdir(parents=True, exist_ok=True)

    def descriptor(name: str, values: np.ndarray) -> dict[str, Any]:
        filename = f"{sidecar_prefix}-{name}" if sidecar_prefix else name
        target = path.parent / filename
        with target.open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
        return {
            "path": filename,
            "sha256": sha256_file(target),
            "dtype": values.dtype.name,
            "shape": list(values.shape),
        }

    splits: dict[str, Any] = {
        "train": {"owner": descriptor("train_owner_ids.npy", train_owner)},
        "validation": {
            "owner": descriptor("validation_owner_ids.npy", validation_owner)
        },
    }
    if semantic is not None:
        assert train_queries is not None and train_mask is not None
        assert validation_queries is not None and validation_mask is not None
        for split, queries, mask in (
            ("train", train_queries, train_mask),
            ("validation", validation_queries, validation_mask),
        ):
            splits[split]["semantic_queries"] = descriptor(
                f"{split}_semantic_queries.npy", queries
            )
            splits[split]["semantic_mask"] = descriptor(
                f"{split}_semantic_mask.npy", mask
            )
    payload: dict[str, Any] = {
        "format": ALLOCATION_FORMAT,
        "version": ALLOCATION_VERSION,
        "source_identity_sha256": source_identity_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "corpus": {
            "train_jsonl_sha256": train_jsonl_sha256,
            "validation_jsonl_sha256": validation_jsonl_sha256,
        },
        "splits": splits,
    }
    if semantic is not None:
        payload["semantic"] = semantic
    if resource_regime is not None:
        payload["resource_regime"] = resource_regime
        payload["ownership_profile"] = ownership_profile
    payload["sha256"] = canonical_digest(payload)
    with path.open("xb") as handle:
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
    manifest = load_allocation_manifest(
        path,
        source_identity_sha256=source_identity_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )
    manifest.split("train", token_count=len(train_owner))
    manifest.split("validation", token_count=len(validation_owner))
    return manifest
