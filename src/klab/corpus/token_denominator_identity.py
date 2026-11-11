"""Cold-authenticate source-token inputs and explicit tokenizer origins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klab.config.loading import load_tokenizer_config
from klab.corpus.release import _verification_operation, verify_release
from klab.data.tokenizer import verify_tokenizer_artifact
from klab.training.manifest import sha256_file


def _safe_path(path: Path) -> Path:
    """Reject symlinks anywhere in an existing path, before canonicalizing it."""
    path = Path(path)
    if ".." in path.parts:
        raise ValueError(f"parent traversal is not allowed: {path}")
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked input or ancestor: {component}")
    return absolute.resolve()


def _dataset_for_tokenizer(release: Path, tokenizer: Path, metadata: dict):
    """Recover only the export configuration adjacent to a bound tokenizer."""
    if metadata.get("corpus_export") is None:
        return None
    export = _safe_path(tokenizer.parent.parent)
    config = load_tokenizer_config(_safe_path(export / "tokenizer.yaml"))
    dataset = config.dataset
    if (
        _safe_path(config.output_dir) != tokenizer.parent
        or dataset.corpus_release_path is None
        or _safe_path(dataset.corpus_release_path) != release
        or dataset.corpus_export_path is None
        or _safe_path(dataset.corpus_export_path) != export
        or dataset.source != metadata.get("source")
        or dataset.revision != metadata.get("revision")
        or config.vocab_size != metadata.get("vocab_size")
    ):
        raise ValueError("tokenizer export configuration differs from measured inputs")
    return dataset


def _authenticate_inputs(
    release: Path,
    tokenizer: Path,
    *,
    tokenizer_origin_release: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify measured release and tokenizer provenance cold, including an explicit origin."""
    release = _safe_path(release)
    tokenizer = _safe_path(tokenizer)
    origin = (
        _safe_path(tokenizer_origin_release)
        if tokenizer_origin_release is not None
        else None
    )
    if origin is not None and origin == release:
        raise ValueError("tokenizer origin must differ from measured release")
    with _verification_operation():
        manifest = verify_release(release)
        origin_manifest = verify_release(origin) if origin is not None else None
        tokenizer_manifest_path = _safe_path(
            tokenizer.with_name("tokenizer_manifest.json")
        )
        metadata = json.loads(tokenizer_manifest_path.read_bytes())
        if not isinstance(metadata, dict):
            raise TypeError("invalid tokenizer manifest")
        dataset = _dataset_for_tokenizer(origin or release, tokenizer, metadata)
        if origin is not None and (
            dataset is None
            or dataset.source != "local_text"
            or dataset.revision != origin_manifest["release_id"]
        ):
            raise ValueError("tokenizer is not bound to its declared origin release")
        verified = verify_tokenizer_artifact(
            tokenizer,
            source=metadata["source"],
            revision=metadata.get("revision"),
            vocab_size=metadata["vocab_size"],
            dataset=dataset,
        )
        if verified != metadata:
            raise ValueError("tokenizer verifier returned inconsistent manifest")
        tokenizer_sha = sha256_file(tokenizer)
        if verified["sha256"] != tokenizer_sha:
            raise ValueError("tokenizer artifact digest mismatch")
        bindings = {
            "release_manifest_sha256": sha256_file(release / "manifest.json"),
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
            "evidence": None,
        }
        if origin is not None:
            assert origin_manifest is not None
            bindings.update(
                {
                    "tokenizer_origin_release_id": origin_manifest["release_id"],
                    "tokenizer_origin_manifest_sha256": sha256_file(
                        origin / "manifest.json"
                    ),
                    "tokenizer_origin_documents_sha256": origin_manifest["files"][
                        "documents.jsonl"
                    ]["sha256"],
                    "tokenizer_origin_documents_size": origin_manifest["files"][
                        "documents.jsonl"
                    ]["size"],
                    "tokenizer_config_sha256": sha256_file(
                        _safe_path(tokenizer.parent.parent / "tokenizer.yaml")
                    ),
                }
            )
        return manifest, bindings
