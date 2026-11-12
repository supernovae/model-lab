"""Immutable training configs and integrity checks for frozen corpus exports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config, load_tokenizer_config
from klab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


def _release(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    from klab.corpus.release import verify_release

    manifest = verify_release(
        path,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    return manifest


def _licenses(release_dir: Path) -> str:
    sources = json.loads((release_dir / "sources.json").read_text(encoding="utf-8"))
    items = sources.values() if isinstance(sources, dict) else sources
    names: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            declaration = item.get("declaration", item)
            if (
                isinstance(declaration, dict)
                and declaration.get("license")
                and declaration.get("redistribution") != "rejected"
            ):
                names.add(str(declaration["license"]))
    if any(isinstance(item, dict) and "rights_policy" in item for item in items):
        with (release_dir / "documents.jsonl").open(encoding="utf-8") as handle:
            names.update(json.loads(line)["license"] for line in handle)
    if not names:
        raise ValueError("frozen release lacks explicit source licenses")
    return "; ".join(sorted(names))


def _split_stats(path: Path, view: str) -> tuple[int, int]:
    count = size = 0
    if view == "lm":
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid LM record at {path}:{number}") from error
                if (
                    not isinstance(record, dict)
                    or set(record) != {"text"}
                    or not isinstance(record["text"], str)
                    or not record["text"].strip()
                ):
                    raise ValueError(f"invalid LM record at {path}:{number}")
                count += 1
                size += len(record["text"].encode("utf-8"))
    else:
        from klab.data.conversations import iter_conversations

        for text in iter_conversations(path):
            count += 1
            size += len(text.encode("utf-8"))
    return count, size


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_yaml(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _export_request(
    release_id: str, view: str, base_digest: str, vocab_size: int, min_frequency: int
) -> dict[str, object]:
    if type(min_frequency) is not int or min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer")
    return {
        "release_id": release_id,
        "view": view,
        "base_config_sha256": base_digest,
        "vocab_size": vocab_size,
        **({"min_frequency": min_frequency} if min_frequency != 1 else {}),
    }


def _export_metadata(
    release_dir: Path,
    manifest: dict[str, object],
    request: dict[str, object],
    counts: list[tuple[int, int]],
    *,
    run_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, object]:
    """Build the authenticated content closure, independent of storage root."""
    view = request["view"]
    sidecar = {
        "schema_version": 1,
        **request,
        "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
        "report_sha256": sha256_file(release_dir / "report.json"),
        "license_report_sha256": sha256_file(release_dir / "license-report.json"),
        "tokenizer_config_sha256": tokenizer_sha256,
        "run_config_sha256": run_sha256,
        "splits": {
            split: {
                "path": f"{view}/{split}.jsonl",
                "sha256": manifest["files"][f"{view}/{split}.jsonl"]["sha256"],
                "records": count,
                "rendered_bytes": size,
            }
            for split, (count, size) in zip(
                ("train", "validation"), counts, strict=True
            )
        },
    }
    if manifest["build_identity"]["release"]["schema_version"] in (2, 3):
        rights = json.loads(
            (release_dir / "license-report.json").read_text(encoding="utf-8")
        )
        sidecar["publication_mode"] = rights["publication_mode"]
        sidecar["weight_license_status"] = rights["weight_license_status"]
        if rights["schema_version"] == 3:
            sidecar["training_use_policy"] = rights["training_use_policy"]
    return sidecar


def export_release(
    release_dir: Path,
    view: str,
    base_run_config: Path,
    vocab_size: int,
    work_root: Path,
    *,
    min_frequency: int = 1,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> Path:
    """Create two complete configs bound to one verified immutable release."""
    if view not in {"lm", "chat"} or vocab_size < 260:
        raise ValueError("view must be lm or chat and vocab_size must be >= 260")
    release_dir = Path(release_dir).resolve()
    manifest = _release(
        release_dir, proof_store=proof_store, verification_mode=verification_mode
    )
    from klab.corpus.release import _release_closure_key

    release_key = _release_closure_key(release_dir, manifest)
    base = load_config(Path(base_run_config))
    base_digest = sha256_file(Path(base_run_config))
    release_id = str(manifest["release_id"])
    request = _export_request(release_id, view, base_digest, vocab_size, min_frequency)
    identity = canonical_digest(request)
    destination = (
        Path(work_root).resolve()
        / "corpora"
        / str(manifest["corpus_id"])
        / "exports"
        / release_id
        / view
        / identity
    )
    train = release_dir / view / "train.jsonl"
    validation = release_dir / view / "validation.jsonl"
    counts = [_split_stats(path, view) for path in (train, validation)]
    if any(count == 0 for count, _ in counts):
        raise ValueError(f"{view} requires nonempty train and validation")
    if any(
        byte_count + count < base.training.seq_len + 1 for count, byte_count in counts
    ):
        raise ValueError(
            "selected split cannot supply a full next-token block at fixed seq_len"
        )
    source = "local_text" if view == "lm" else "local_chat"
    dataset = base.dataset.model_copy(
        update={
            "source": source,
            "revision": release_id,
            "license": _licenses(release_dir),
            "train_path": train,
            "validation_path": validation,
            "corpus_release_path": release_dir,
            "corpus_export_path": destination,
            "source_manifest_path": None,
            "dataset_config": None,
            "cache_dir": destination / "prepared",
            "train_max_documents": counts[0][0],
            "validation_max_documents": counts[1][0],
            "train_max_tokens": counts[0][1] + counts[0][0] + 1,
            "validation_max_tokens": counts[1][1] + counts[1][0] + 1,
        }
    )
    run = base.model_copy(
        update={
            "model": base.model.model_copy(update={"vocab_size": vocab_size}),
            "dataset": dataset,
            "tokenizer": base.tokenizer.model_copy(
                update={"path": destination / "tokenizer" / "tokenizer.json"}
            ),
        }
    )
    tokenizer = {
        "schema_version": 1,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "max_documents": counts[0][0],
        "output_dir": str(destination / "tokenizer"),
        "dataset": dataset.model_dump(mode="json"),
    }
    run_payload = run.model_dump(mode="json")
    RunConfig.model_validate(run_payload)
    TokenizerTrainConfig.model_validate(tokenizer)
    tokenizer_bytes = yaml.safe_dump(tokenizer, sort_keys=True).encode("utf-8")
    run_bytes = yaml.safe_dump(run_payload, sort_keys=True).encode("utf-8")
    sidecar = _export_metadata(
        release_dir,
        manifest,
        request,
        counts,
        run_sha256=hashlib.sha256(run_bytes).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
    if destination.exists():
        if any(
            not (destination / filename).is_file()
            for filename in ("export.json", "run.yaml", "tokenizer.yaml")
        ):
            raise ValueError("incomplete immutable corpus export")
        if (
            json.loads((destination / "export.json").read_text(encoding="utf-8"))
            != sidecar
            or (destination / "run.yaml").read_bytes() != run_bytes
            or (destination / "tokenizer.yaml").read_bytes() != tokenizer_bytes
        ):
            raise ValueError("existing corpus export has changed")
        if _release_closure_key(release_dir, manifest) != release_key:
            raise ValueError("release changed during export creation")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".export-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        _write_yaml(staging / "tokenizer.yaml", tokenizer)
        _write_yaml(staging / "run.yaml", run_payload)
        _write_json(staging / "export.json", sidecar)
        if destination.exists():
            raise FileExistsError(
                f"corpus export appeared during publication: {destination}"
            )
        if _release_closure_key(release_dir, manifest) != release_key:
            raise ValueError("release changed during export creation")
        staging.rename(destination)
    if _release_closure_key(release_dir, manifest) != release_key:
        raise ValueError("release changed during export creation")
    return destination


def verify_release_export(
    dataset: DatasetConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, object]:
    """Verify an export; typed reuse is restricted to its complete trusted closure."""
    from klab.corpus.release import _verification_operation

    with _verification_operation():
        if (
            not _domain_cold
            and proof_store is not None
            and verification_mode == "verified_reuse"
            and dataset.corpus_export_path is not None
        ):
            from klab.experiments.artifacts import Artifact, verify_artifact

            path = dataset.corpus_export_path
            export = json.loads((path / "export.json").read_text(encoding="utf-8"))
            verify_artifact(
                Artifact(
                    kind="corpus_export",
                    version=export["schema_version"],
                    producer="klab",
                    identifier=path.name,
                    sha256=path.name,
                    path=str(path),
                ),
                path / "export.json",
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            # A signed export receipt authenticates its on-disk run configuration,
            # not an arbitrary caller's dataset object.
            run = load_config(path / "run.yaml")
            if run.dataset != dataset:
                raise ValueError("corpus export generated dataset mismatch")
            return _export_binding(export)
        return _verify_release_export_cold(
            dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _export_binding(export: dict[str, object]) -> dict[str, object]:
    splits = export["splits"]
    return {
        "release_id": export["release_id"],
        "view": export["view"],
        "vocab_size": export["vocab_size"],
        "train_sha256": splits["train"]["sha256"],
        "validation_sha256": splits["validation"]["sha256"],
        "report_sha256": export["report_sha256"],
        "license_report_sha256": export["license_report_sha256"],
        "export_sha256": hashlib.sha256(canonical_json(export) + b"\n").hexdigest(),
    }


def _verify_release_export_cold(
    dataset: DatasetConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    """Verify the release, export request, and exact bytes selected by dataset."""
    if dataset.corpus_release_path is None or dataset.corpus_export_path is None:
        raise ValueError("dataset does not reference a frozen corpus export")
    release_dir = dataset.corpus_release_path
    export_dir = dataset.corpus_export_path
    manifest = _release(
        release_dir,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    export_bytes = (export_dir / "export.json").read_bytes()
    export = json.loads(export_bytes)
    if (
        not isinstance(export, dict)
        or export.get("schema_version") != 1
        or export.get("release_id") != manifest["release_id"]
        or dataset.revision != manifest["release_id"]
    ):
        raise ValueError("corpus export release identity mismatch")
    view = export.get("view")
    if view not in {"lm", "chat"} or dataset.source != (
        "local_text" if view == "lm" else "local_chat"
    ):
        raise ValueError("corpus export view mismatch")
    request = _export_request(
        str(manifest["release_id"]),
        view,
        export.get("base_config_sha256"),
        export.get("vocab_size"),
        export.get("min_frequency", 1),
    )
    if (
        not isinstance(request["base_config_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", request["base_config_sha256"]) is None
        or type(request["vocab_size"]) is not int
        or request["vocab_size"] < 260
        or export_dir.name != canonical_digest(request)
        or export_dir.parts[-6:-1]
        != (
            "corpora",
            str(manifest["corpus_id"]),
            "exports",
            str(manifest["release_id"]),
            view,
        )
        or dataset.license != _licenses(release_dir)
    ):
        raise ValueError("corpus export request or license mismatch")
    for name, key in (
        ("run.yaml", "run_config_sha256"),
        ("tokenizer.yaml", "tokenizer_config_sha256"),
    ):
        if sha256_file(export_dir / name) != export.get(key):
            raise ValueError(f"corpus export config changed: {name}")
    exported_run = load_config(export_dir / "run.yaml")
    exported_tokenizer = load_tokenizer_config(export_dir / "tokenizer.yaml")
    if (
        exported_run.dataset != dataset
        or exported_tokenizer.dataset != dataset
        or exported_run.model.vocab_size != request["vocab_size"]
        or exported_tokenizer.vocab_size != request["vocab_size"]
        or exported_tokenizer.min_frequency != request.get("min_frequency", 1)
    ):
        raise ValueError("corpus export generated dataset or vocabulary mismatch")
    counts = []
    for split, path in (
        ("train", dataset.train_path),
        ("validation", dataset.validation_path),
    ):
        expected = release_dir / view / f"{split}.jsonl"
        if path is None or path.resolve() != expected.resolve():
            raise ValueError(f"corpus export {split} split path mismatch")
        records, rendered_bytes = _split_stats(expected, view)
        counts.append((records, rendered_bytes))
        count = (
            dataset.train_max_documents
            if split == "train"
            else dataset.validation_max_documents
        )
        budget = (
            dataset.train_max_tokens
            if split == "train"
            else dataset.validation_max_tokens
        )
        if count != records or budget != rendered_bytes + records + 1:
            raise ValueError(
                f"corpus export {split} count or rendered-byte budget mismatch"
            )
    expected_export = _export_metadata(
        release_dir,
        manifest,
        request,
        counts,
        run_sha256=sha256_file(export_dir / "run.yaml"),
        tokenizer_sha256=sha256_file(export_dir / "tokenizer.yaml"),
    )
    if export_bytes != canonical_json(expected_export) + b"\n":
        raise ValueError("corpus export content or policy mismatch")
    splits = expected_export["splits"]
    return {
        "release_id": manifest["release_id"],
        "view": view,
        "vocab_size": export["vocab_size"],
        "train_sha256": splits["train"]["sha256"],
        "validation_sha256": splits["validation"]["sha256"],
        "report_sha256": export["report_sha256"],
        "license_report_sha256": export["license_report_sha256"],
        "export_sha256": sha256_file(export_dir / "export.json"),
    }
