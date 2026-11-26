"""Immutable training configs and integrity checks for frozen corpus exports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config, load_tokenizer_config
from klab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from klab.corpus.release_verify import current_settings
from klab.corpus.shards import declare_stage, plan_ranges, run_sharded
from klab.safe_io import rename_noreplace, write_canonical_json
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.corpus.release import ReleaseMembers, ReleaseReader
    from klab.verification_proofs import ProofStore, VerificationMode


LICENSE_SCAN_STAGE = declare_stage("export document licenses")
SPLIT_STATS_STAGE = declare_stage("export split statistics")


def _license_range(path: str, start: int, end: int) -> list[str]:
    """Collect the distinct normalized document labels in one buffered range."""
    labels: set[str] = set()
    with Path(path).open("rb") as stream:
        stream.seek(start)
        offset = start
        while offset < end:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"document file truncated at {path}:{offset}")
            label = json.loads(raw.decode("utf-8"))["license"]
            if not isinstance(label, str):
                raise TypeError(f"document license must be a string at {path}:{offset}")
            labels.add(label)
            offset += len(raw)
    return sorted(labels)


def _export_range_inputs() -> dict[str, str] | None:
    if current_settings().scratch_dir is not None:
        return {"producer": sha256_file(Path(__file__))}
    return None


def release_licenses(sources_path: Path, documents_path: Path) -> str:
    """Collect license labels from authenticated source and document members."""
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    items = sources.values() if isinstance(sources, dict) else sources
    names: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            declaration = item.get("declaration", item)
            if isinstance(declaration, dict) and declaration.get("license"):
                names.add(str(declaration["license"]))
    if any(isinstance(item, dict) and "rights_policy" in item for item in items):
        plan = plan_ranges(
            LICENSE_SCAN_STAGE,
            documents_path,
            inputs=_export_range_inputs(),
        )
        for labels in run_sharded(plan, _license_range):
            names.update(labels)
    if not names:
        raise ValueError("frozen release lacks explicit source licenses")
    return "; ".join(sorted(names))


def _lm_range_stats(path: str, start: int, end: int) -> tuple[int, int, str | None]:
    """Valid records and rendered bytes before the first bad line, if any."""
    count = size = 0
    with Path(path).open("rb") as stream:
        stream.seek(start)
        offset = start
        for raw in stream:
            if offset >= end:
                break
            offset += len(raw)
            try:
                record = json.loads(raw.decode("utf-8"))
            except ValueError:
                return count, size, "invalid LM record"
            if (
                not isinstance(record, dict)
                or set(record) != {"text"}
                or not isinstance(record["text"], str)
                or not record["text"].strip()
            ):
                return count, size, "invalid LM record"
            count += 1
            size += len(record["text"].encode("utf-8"))
    return count, size, None


def _chat_range_stats(path: str, start: int, end: int) -> tuple[int, int, str | None]:
    from klab.data.conversations import (
        ConversationRecordError,
        render_conversation_line,
    )

    count = size = 0
    source = Path(path)
    with source.open("rb") as stream:
        stream.seek(start)
        offset = start
        while offset < end:
            raw = stream.readline()
            if not raw:
                return count, size, "truncated chat split"
            try:
                text = render_conversation_line(raw, source, count + 1).text
                size += len(text.encode("utf-8"))
            except ConversationRecordError as error:
                return count, size, error.detail
            except ValueError as error:
                return count, size, str(error)
            count += 1
            offset += len(raw)
    return count, size, None


def split_stats(path: Path, view: str) -> tuple[int, int]:
    inputs = _export_range_inputs()
    if inputs is not None:
        inputs["view"] = view
        if view == "chat":
            from klab.verifier_authority import CHAT_RENDERER_MODULES, scoped_modules

            inputs["renderer"] = canonical_digest(
                scoped_modules(leaves=(*CHAT_RENDERER_MODULES, "canonical"))
            )
    plan = plan_ranges(SPLIT_STATS_STAGE, path, inputs=inputs)
    worker = _lm_range_stats if view == "lm" else _chat_range_stats
    count = size = 0
    for kept, rendered, error in run_sharded(plan, worker, ordered=True):
        if error is not None:
            location = f"{path}:{count + kept + 1}"
            raise ValueError(
                f"{error} at {location}" if view == "lm" else f"{location}: {error}"
            )
        count += kept
        size += rendered
    return count, size


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
    release: ReleaseReader,
    request: dict[str, object],
    counts: list[tuple[int, int]],
    *,
    run_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, object]:
    """Build the authenticated content closure, independent of storage root."""
    view = request["view"]
    files = release.manifest["files"]
    sidecar = {
        "schema_version": 1,
        **request,
        "release_manifest_sha256": sha256_file(release.root / "manifest.json"),
        "report_sha256": files["report.json"]["sha256"],
        "license_report_sha256": files["license-report.json"]["sha256"],
        "tokenizer_config_sha256": tokenizer_sha256,
        "run_config_sha256": run_sha256,
        "splits": {
            split: {
                "path": f"{view}/{split}.jsonl",
                "sha256": files[f"{view}/{split}.jsonl"]["sha256"],
                "records": count,
                "rendered_bytes": size,
            }
            for split, (count, size) in zip(
                ("train", "validation"), counts, strict=True
            )
        },
    }
    rights = json.loads(
        release.member("license-report.json").read_text(encoding="utf-8")
    )
    sidecar["publication_mode"] = rights["publication_mode"]
    sidecar["weight_license_status"] = rights["weight_license_status"]
    sidecar["release_policy"] = rights["release_policy"]
    return sidecar


def _export_receipt_dependencies(export_dir: Path, release_dir: Path) -> list[Path]:
    from klab.experiments.artifacts import snapshot_dependencies

    if export_dir.is_symlink():
        raise ValueError("symlinked corpus export directory")
    # Tokenizer artifacts and prepared data can be added after the export.
    return [
        export_dir / "run.yaml",
        export_dir / "tokenizer.yaml",
        release_dir,
        *snapshot_dependencies(release_dir, "manifest.json"),
    ]


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
    if view not in {"lm", "chat"} or type(vocab_size) is not int or vocab_size < 260:
        raise ValueError("view must be lm or chat and vocab_size must be >= 260")
    from klab.corpus.release import open_release

    with open_release(
        release_dir, proof_store=proof_store, verification_mode=verification_mode
    ) as release:
        return _publish_export(
            release,
            view,
            base_run_config,
            vocab_size,
            work_root,
            min_frequency=min_frequency,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _publish_export(
    release: ReleaseReader,
    view: str,
    base_run_config: Path,
    vocab_size: int,
    work_root: Path,
    *,
    min_frequency: int,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> Path:
    from klab.corpus import release_receipt

    manifest = release.manifest
    base = load_config(Path(base_run_config))
    base_digest = sha256_file(Path(base_run_config))
    release_id = str(manifest["release_id"])
    corpus_id = manifest["corpus_id"]
    if (
        not isinstance(corpus_id, str)
        or corpus_id in {"", ".", ".."}
        or Path(corpus_id).name != corpus_id
    ):
        raise ValueError("invalid corpus ID in export release")
    request = _export_request(release_id, view, base_digest, vocab_size, min_frequency)
    identity = canonical_digest(request)
    destination = (
        Path(work_root).resolve()
        / "corpora"
        / corpus_id
        / "exports"
        / release_id
        / view
        / identity
    )
    train = release.member(f"{view}/train.jsonl")
    validation = release.member(f"{view}/validation.jsonl")
    existing = destination.exists()
    sidecar_path = destination / "export.json"
    dependencies = None
    reused_index = None
    if existing and proof_store is not None and verification_mode == "verified_reuse":
        dependencies = _export_receipt_dependencies(destination, release.root)
        found = release_receipt.lookup(
            proof_store, "export", sidecar_path, identity, dependencies
        )
        if found.hit:
            reused_index = found.index
    if reused_index is None:
        counts = [split_stats(path, view) for path in (train, validation)]
        licenses = release_licenses(
            release.member("sources.json"), release.member("documents.jsonl")
        )
    else:
        verified = json.loads(sidecar_path.read_bytes())
        counts = [
            (row["records"], row["rendered_bytes"])
            for row in (verified["splits"]["train"], verified["splits"]["validation"])
        ]
        licenses = load_config(destination / "run.yaml").dataset.license
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
            "license": licenses,
            "train_path": train,
            "validation_path": validation,
            "corpus_release_path": release.root,
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
        release,
        request,
        counts,
        run_sha256=hashlib.sha256(run_bytes).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
    if not existing:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".export-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary)
            _write_yaml(staging / "tokenizer.yaml", tokenizer)
            _write_yaml(staging / "run.yaml", run_payload)
            write_canonical_json(staging / "export.json", sidecar, exclusive=True)
            release.require_unchanged()
            rename_noreplace(staging, destination)

    expected = {
        sidecar_path: canonical_json(sidecar) + b"\n",
        destination / "run.yaml": run_bytes,
        destination / "tokenizer.yaml": tokenizer_bytes,
    }
    if existing or proof_store is not None:
        if any(not path.is_file() for path in expected):
            raise ValueError("incomplete immutable corpus export")
        if proof_store is not None and dependencies is None:
            dependencies = _export_receipt_dependencies(destination, release.root)
        paths = [sidecar_path, *dependencies] if dependencies is not None else None
        index = (
            release_receipt.stat_index(paths, strict=False)
            if paths is not None and reused_index is None
            else None
        )
        if any(path.read_bytes() != content for path, content in expected.items()):
            raise ValueError("existing corpus export has changed")
        if reused_index is not None:
            assert paths is not None
            if release_receipt.stat_index(paths, strict=False) != reused_index:
                raise ValueError("corpus export changed during reuse")
        elif proof_store is not None and dependencies is not None:
            release.require_unchanged()
            release_receipt.record(
                proof_store, "export", sidecar_path, identity, dependencies, index=index
            )
    return destination


@dataclass(frozen=True)
class AuthenticatedExport:
    """An export binding plus the file generation authenticated for it."""

    binding: dict[str, object]
    release_root: Path
    sidecar: Path
    dependencies: tuple[Path, ...]
    generation: str

    @contextmanager
    def release_members(self) -> Iterator[ReleaseMembers]:
        """Resolve members under the export proof without verifying the release again."""
        from klab.corpus.release import ReleaseMembers
        from klab.corpus.release_receipt import stat_index

        export_bytes = self.sidecar.read_bytes()
        if hashlib.sha256(export_bytes).hexdigest() != self.binding["export_sha256"]:
            raise ValueError("corpus export changed before reading its release")
        export = json.loads(export_bytes)
        manifest_bytes = (self.release_root / "manifest.json").read_bytes()
        if (
            hashlib.sha256(manifest_bytes).hexdigest()
            != export["release_manifest_sha256"]
        ):
            raise ValueError("corpus export release manifest has changed")
        manifest = json.loads(manifest_bytes)
        if manifest["release_id"] != self.binding["release_id"]:
            raise ValueError("corpus export release identity mismatch")
        yield ReleaseMembers(self.release_root, manifest)
        if (
            stat_index([self.sidecar, *self.dependencies], strict=False)[0]
            != self.generation
        ):
            raise ValueError("corpus export or release changed during member reads")


def verify_release_export(
    dataset: DatasetConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode | None = None,
) -> dict[str, object]:
    """Verify the frozen export and return its portable training binding."""
    return authenticate_release_export(
        dataset, proof_store=proof_store, verification_mode=verification_mode
    ).binding


def authenticate_release_export(
    dataset: DatasetConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode | None = None,
) -> AuthenticatedExport:
    """Verify an export and its release, by signed receipt when one still holds.

    Without a `verification_mode` the receipt store chosen by
    `verification_settings` applies, as it does for `verify_release`.
    """
    from klab.corpus.release import _verification_operation
    from klab.corpus.release_receipt import receipt_first
    from klab.corpus.release_verify import current_settings

    if dataset.corpus_export_path is None or dataset.corpus_release_path is None:
        raise ValueError("dataset does not reference a frozen corpus export")
    if verification_mode is None:
        ambient = current_settings()
        proof_store, verification_mode = (
            ambient.proof_store,
            ambient.verification_mode,
        )
    export_dir, release_dir = dataset.corpus_export_path, dataset.corpus_release_path
    sidecar = export_dir / "export.json"

    def reused() -> dict[str, object]:
        # A receipt vouches for the export directory, not for a caller's dataset.
        if load_config(export_dir / "run.yaml").dataset != dataset:
            raise ValueError("corpus export generated dataset mismatch")
        return _export_binding(json.loads(sidecar.read_bytes()))

    with _verification_operation():
        dependencies = _export_receipt_dependencies(export_dir, release_dir)
        binding, generation = receipt_first(
            "export",
            sidecar,
            export_dir.name,
            dependencies,
            proof_store,
            verification_mode,
            lambda: _verify_release_export_cold(
                dataset, proof_store=proof_store, verification_mode=verification_mode
            ),
            reused,
        )
        return AuthenticatedExport(
            binding, release_dir.resolve(), sidecar, tuple(dependencies), generation
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
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> dict[str, object]:
    """Verify the release, export request, and exact bytes selected by dataset."""
    if dataset.corpus_release_path is None or dataset.corpus_export_path is None:
        raise ValueError("dataset does not reference a frozen corpus export")
    from klab.corpus.release import open_release

    with open_release(
        dataset.corpus_release_path,
        proof_store=proof_store,
        verification_mode=verification_mode,
    ) as release:
        return _verify_export_members(dataset, release, dataset.corpus_export_path)


def _verify_export_members(
    dataset: DatasetConfig, release: ReleaseReader, export_dir: Path
) -> dict[str, object]:
    manifest = release.manifest
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
        or dataset.license
        != release_licenses(
            release.member("sources.json"), release.member("documents.jsonl")
        )
    ):
        raise ValueError("corpus export request or license mismatch")
    config_digests = {}
    for name, key in (
        ("run.yaml", "run_config_sha256"),
        ("tokenizer.yaml", "tokenizer_config_sha256"),
    ):
        config_digests[key] = sha256_file(export_dir / name)
        if config_digests[key] != export.get(key):
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
        expected = release.member(f"{view}/{split}.jsonl")
        if path is None or path.resolve() != expected.resolve():
            raise ValueError(f"corpus export {split} split path mismatch")
        records, rendered_bytes = split_stats(expected, view)
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
        release,
        request,
        counts,
        run_sha256=config_digests["run_config_sha256"],
        tokenizer_sha256=config_digests["tokenizer_config_sha256"],
    )
    if export_bytes != canonical_json(expected_export) + b"\n":
        raise ValueError("corpus export content or policy mismatch")
    return _export_binding(expected_export)
