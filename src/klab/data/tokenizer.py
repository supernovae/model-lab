"""Portable ByteLevel BPE tokenizer artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from klab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from klab.corpus.export import authenticate_release_export
from klab.corpus.shards import Single, declare_stage
from klab.data.datasets import iter_documents
from klab.data.sources import verify_snapshot
from klab.progress import progress_phase
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace
from klab.training.manifest import sha256_file
from klab.verification_proofs import writable_by_others
from klab.workspace_cleanup import workspace_lock

FIT_STAGE = declare_stage(
    "tokenizer fit",
    Single(
        "the Hugging Face BPE trainer is multithreaded natively and selects merges sequentially"
    ),
)

if TYPE_CHECKING:
    from klab.data.tokenizer_sampling import Selection
    from klab.verification_proofs import ProofStore, VerificationMode

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
_FIT_FILES = {"tokenizer.json", "tokenizer_manifest.json"}


def _verify_bakeoff_selection(
    path: Path,
    manifest: dict[str, object],
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> str:
    """Authenticate the original selection and its train/held-out receipts."""
    from klab.corpus.tokenizer_bakeoff import verify_selection

    path = path.absolute()
    report = path.parent.parent.parent / "report.json"
    if (
        path.name != "tokenizer.json"
        or path.parent.parent.name != "candidates"
        or path.parent.name != str(manifest["vocab_size"])
        or any(
            item.is_symlink()
            for item in (
                path,
                path.with_name("tokenizer_manifest.json"),
                path.parent,
                path.parent.parent,
                report.parent,
                report,
                report.with_name("fit.jsonl"),
                report.with_name("validation.jsonl"),
            )
        )
    ):
        raise ValueError("invalid bakeoff candidate path")
    digest, identifier = verify_selection(
        report, proof_store=proof_store, verification_mode=verification_mode
    )
    binding = manifest["corpus_forge_bakeoff"]
    if (
        digest != manifest["sha256"]
        or identifier != f"{binding['release_id']}:{manifest['vocab_size']}"
    ):
        raise ValueError("tokenizer is not the selected bakeoff candidate")
    return identifier


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _fit_staging(output: Path) -> Iterator[Path]:
    """Serialize fitting and publication using the lock in a private sibling."""
    work = output.parent.resolve() / f".{output.name}.tokenizer-work"
    missing = []
    parent = work.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in reversed(missing):
        sync_directory(parent.parent)
    info = work.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or writable_by_others(info)
    ):
        raise ValueError(f"unsafe tokenizer staging workspace: {work}")
    stage = work / "artifact"
    owner = work / ".klab-tokenizer-fit"
    with workspace_lock(work):
        if owner.is_symlink() or (owner.exists() and not owner.is_file()):
            raise ValueError(f"unsafe tokenizer staging marker: {owner}")
        if not owner.exists():
            if stage.exists() or stage.is_symlink():
                raise ValueError(f"unowned tokenizer staging directory: {stage}")
            owner.touch(exist_ok=False)
        sync_directory(work)
        sync_directory(work.parent)
        yield stage


def _staged_members(stage: Path) -> set[Path]:
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError(f"unsafe tokenizer staging directory: {stage}")
    members = set(stage.iterdir())
    for member in members:
        info = member.lstat()
        if (
            member.name not in _FIT_FILES | {"tokenizer_manifest.json.tmp"}
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            raise ValueError(f"unexpected tokenizer staging member: {member}")
    return members


def _matching_fit(
    directory: Path,
    config: TokenizerTrainConfig,
    contract: dict,
    selection: Selection | None,
) -> bool:
    tokenizer = directory / "tokenizer.json"
    manifest_path = directory / "tokenizer_manifest.json"
    if directory.is_symlink() or any(
        file.is_symlink() or not file.is_file() for file in (tokenizer, manifest_path)
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except UnicodeError, json.JSONDecodeError:
        return False
    if (
        not isinstance(manifest, dict)
        or manifest.get("training_contract") != contract
        or any(
            manifest.get(key) != contract.get(key)
            for key in (
                "source",
                "revision",
                "vocab_size",
                "corpus_export",
                "source_manifest_sha256",
            )
        )
        or sha256_file(tokenizer) != manifest.get("sha256")
    ):
        return False
    if selection is not None:
        from klab.data.tokenizer_sampling import audit

        return audit(
            config, load_tokenizer(tokenizer), selection=selection
        ) == manifest.get("sampling_audit")
    return True


def _publish_fit(stage: Path, output: Path, *, reused: bool) -> Path:
    with progress_phase(
        "tokenizer_artifact_publication",
        completed_work=0,
        total_work=1,
        unit="artifacts",
        raw_counters={"reused_staged_fit": reused},
    ) as progress:
        sync_directory(stage)
        sync_directory(stage.parent)
        rename_noreplace(stage, output)
        sync_directory(output.parent)
        sync_directory(stage.parent)
        progress.update(completed_work=1)
    return output / "tokenizer.json"


def _published_fit(
    output: Path,
    config: TokenizerTrainConfig,
    contract: dict,
    selection: Selection | None,
) -> Path | None:
    if not (output.exists() or output.is_symlink()):
        return None
    if not _matching_fit(output, config, contract, selection):
        raise FileExistsError(
            f"tokenizer output at {output} has different or unverifiable training provenance; "
            "use a new output directory, or reference the existing tokenizer explicitly"
        )
    sync_directory(output.parent)
    return output / "tokenizer.json"


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.is_file():
        raise FileNotFoundError(
            f"tokenizer artifact missing: {path}; run `klab tokenizer train CONFIG`"
        )
    return Tokenizer.from_file(str(path))


def _verify_tokenizer_manifest(
    path: Path,
    *,
    source: str,
    revision: str | None,
    vocab_size: int,
    dataset: DatasetConfig | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    """Check a candidate's own bytes and dataset without applying bakeoff selection."""
    manifest_path = path.with_name("tokenizer_manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"complete tokenizer artifact required at {path}; finish `klab tokenizer train CONFIG` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bakeoff = (
        manifest.get("corpus_forge_bakeoff") if isinstance(manifest, dict) else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
        or manifest.get("source") != source
        or (
            manifest.get("revision") != revision
            and not (
                isinstance(bakeoff, dict)
                and dataset is not None
                and dataset.corpus_release_path is not None
                and revision == dataset.revision
            )
        )
        or manifest.get("vocab_size") != vocab_size
    ):
        raise ValueError(f"tokenizer artifact provenance or digest mismatch: {path}")
    if source == "snapshot":
        if dataset is None:
            raise ValueError(
                "snapshot tokenizer verification requires dataset configuration"
            )
        assert dataset.source_manifest_path is not None
        verify_snapshot(dataset)
        snapshot_digest = hashlib.sha256(
            dataset.source_manifest_path.read_bytes()
        ).hexdigest()
        if manifest.get("source_manifest_sha256") != snapshot_digest:
            raise ValueError("tokenizer snapshot identity mismatch")
    verified_export = None
    if dataset is not None and dataset.corpus_release_path is not None:
        verified_export = authenticate_release_export(
            dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        binding = verified_export.binding
        if bakeoff is not None:
            if (
                manifest.get("corpus_export") is not None
                or binding["release_id"] != bakeoff.get("release_id")
                or binding["vocab_size"] != vocab_size
                or dataset.revision != bakeoff.get("release_id")
                or dataset.source != "local_text"
            ):
                raise ValueError("tokenizer bakeoff release/export identity mismatch")
        elif (
            manifest.get("corpus_export") != binding
            or manifest.get("training_contract", {}).get("corpus_export") != binding
        ):
            raise ValueError("tokenizer corpus export identity mismatch")
    elif manifest.get("corpus_export") is not None:
        raise ValueError(
            "corpus tokenizer verification requires frozen export configuration"
        )
    contract = manifest.get("training_contract", {})
    if "sampling" in contract:
        if dataset is None or verified_export is None:
            raise ValueError(
                "sampled tokenizer verification requires its frozen dataset"
            )
        config = TokenizerTrainConfig(
            schema_version=1,
            vocab_size=vocab_size,
            min_frequency=contract["min_frequency"],
            max_documents=contract["max_documents"],
            output_dir=path.parent,
            dataset=dataset,
            sampling=contract["sampling"],
        )
        from klab.data.tokenizer_sampling import audit, documents, select_documents

        # The fit's byte selection and the audit's reserve come from one scan.
        with verified_export.release_members() as release:
            selection = select_documents(config, release)
        stats = {}
        for _ in documents(config, stats, selection=selection):
            pass
        if (
            stats["sampling"] != contract.get("sampling_result")
            or stats["digest"] != contract.get("content_digest_sha256")
            or stats["bytes"] != contract.get("selected_input_bytes_utf8")
        ):
            raise ValueError("tokenizer TRAIN byte-selection identity mismatch")
        if audit(config, load_tokenizer(path), selection=selection) != manifest.get(
            "sampling_audit"
        ):
            raise ValueError("tokenizer disjoint TRAIN audit mismatch")
    return manifest


def verify_tokenizer_artifact(
    path: Path,
    *,
    source: str,
    revision: str | None,
    vocab_size: int,
    dataset: DatasetConfig | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, object]:
    """Verify the tokenizer and its complete provenance closure."""
    if source == "local_token_mixture":
        if dataset is None:
            raise ValueError(
                "token mixture tokenizer verification requires its dataset"
            )
        if revision != dataset.revision:
            raise ValueError("token mixture tokenizer revision mismatch")
        from klab.corpus.mixture import verify_mixture_dataset

        verify_mixture_dataset(dataset, path, vocab_size)
        return json.loads(
            path.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
        )
    from klab.corpus.release import _verification_operation

    with _verification_operation():
        if (
            not _domain_cold
            and proof_store is not None
            and verification_mode == "verified_reuse"
            and dataset is not None
            and dataset.corpus_export_path is not None
        ):
            from klab.experiments.artifacts import Artifact, verify_artifact

            path = Path(path)
            manifest = json.loads(
                path.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
            )
            verify_artifact(
                Artifact(
                    kind="tokenizer",
                    version=1,
                    producer="klab",
                    identifier=path.parent.name,
                    sha256=manifest["sha256"],
                    path=str(path),
                ),
                path.with_name("tokenizer_manifest.json"),
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            if (
                manifest.get("source") != source
                or manifest.get("vocab_size") != vocab_size
                or (
                    manifest.get("revision") != revision
                    and not (
                        manifest.get("corpus_forge_bakeoff") is not None
                        and dataset.revision == revision
                    )
                )
            ):
                raise ValueError("tokenizer artifact provenance mismatch")
            return manifest
        manifest = _verify_tokenizer_manifest(
            path,
            source=source,
            revision=revision,
            vocab_size=vocab_size,
            dataset=dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if manifest.get("corpus_forge_bakeoff") is not None:
            if not isinstance(manifest["corpus_forge_bakeoff"], dict):
                raise ValueError("invalid bakeoff provenance")
            _verify_bakeoff_selection(
                path,
                manifest,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
        return manifest


def verify_run_tokenizer(
    config: RunConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    """Verify a run's tokenizer against its declared origin, never its consumer."""
    from klab.training.manifest import sha256_file

    origin = config.tokenizer.origin
    dataset = (
        origin.dataset
        if origin is not None and origin.dataset is not None
        else config.dataset
    )
    path = config.tokenizer.path
    if origin is not None and (
        sha256_file(path) != origin.sha256
        or sha256_file(path.with_name("tokenizer_manifest.json"))
        != origin.manifest_sha256
    ):
        raise ValueError("explicit tokenizer origin digest mismatch")
    if origin is not None and origin.mixture_declaration_path is not None:
        from klab.corpus.mixture import verify_mixture_tokenizer_origin

        if (
            sha256_file(origin.mixture_declaration_path)
            != origin.mixture_declaration_sha256
        ):
            raise ValueError("tokenizer origin mixture declaration digest mismatch")
        return verify_mixture_tokenizer_origin(
            origin.mixture_declaration_path, path, config.model.vocab_size
        )
    return verify_tokenizer_artifact(
        path,
        source=dataset.source,
        revision=dataset.revision,
        vocab_size=config.model.vocab_size,
        dataset=dataset,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )


def _bounded_documents(
    config: TokenizerTrainConfig,
    stats: dict[str, object],
    *,
    whole_documents: bool = False,
    selection: Selection | None = None,
):
    """Replay the same train prefix without retaining corpus strings in memory."""
    if config.sampling is not None:
        from klab.data.tokenizer_sampling import documents

        assert selection is not None
        yield from documents(config, stats, selection=selection)
        return
    selected = acquired = byte_count = 0
    digest = hashlib.sha256()
    documents = iter(iter_documents(config.dataset, "train"))
    budget = config.dataset.train_max_tokens
    stop_reason = "source_exhausted"
    while acquired < config.max_documents and byte_count < budget:
        try:
            document = next(documents)
        except StopIteration:
            break
        acquired += 1
        if not document:
            continue
        encoded = document.encode("utf-8")
        remaining = budget - byte_count
        if len(encoded) > remaining:
            if whole_documents:
                stop_reason = "byte_limit"
                break
            document = encoded[:remaining].decode("utf-8", errors="ignore")
            encoded = document.encode("utf-8")
            truncated = True
        else:
            truncated = False
        if document:
            digest.update(encoded + b"\0")
            byte_count += len(encoded)
            selected += 1
            yield document
        if truncated:
            stop_reason = "byte_limit"
            break
    if stop_reason == "source_exhausted":
        if acquired >= config.max_documents:
            stop_reason = "document_limit"
        elif byte_count >= budget:
            stop_reason = "byte_limit"
    stats.update(
        acquired=acquired,
        selected=selected,
        bytes=byte_count,
        digest=digest.hexdigest(),
        stop_reason=stop_reason,
    )


def _snapshot_digest(config: TokenizerTrainConfig) -> str | None:
    if config.dataset.source != "snapshot":
        return None
    assert config.dataset.source_manifest_path is not None
    verify_snapshot(config.dataset)
    return hashlib.sha256(config.dataset.source_manifest_path.read_bytes()).hexdigest()


def train_tokenizer(
    config: TokenizerTrainConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> Path:
    """Train BPE from a bounded train-only UTF-8 byte prefix."""
    from klab.workspace_preflight import require_storage, tokenizer_storage_checks

    require_storage(tokenizer_storage_checks(config))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = config.output_dir

    input_byte_budget = (
        config.sampling.max_fit_bytes
        if config.sampling is not None
        else config.dataset.train_max_tokens
    )
    verified_export = (
        authenticate_release_export(
            config.dataset,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if config.dataset.corpus_release_path is not None
        else None
    )
    corpus_export = verified_export.binding if verified_export is not None else None
    if corpus_export is not None and corpus_export["vocab_size"] != config.vocab_size:
        raise ValueError(
            "corpus export vocabulary size does not match tokenizer request"
        )
    source_manifest_sha256 = _snapshot_digest(config)
    whole_documents = config.dataset.source in {"local_text", "snapshot"}
    stats: dict[str, object] = {}
    selection = None
    source_started = time.monotonic()
    with progress_phase(
        "tokenizer_dataset_initialization_and_source_iteration",
        completed_work=0,
        unit="documents",
        raw_counters={
            "documents_acquired": 0,
            "documents_retained": 0,
            "source_bytes": 0,
        },
    ) as progress:
        if config.sampling is not None:
            from klab.data.tokenizer_sampling import select_documents

            assert verified_export is not None
            with verified_export.release_members() as release:
                selection = select_documents(config, release)
        for _ in _bounded_documents(
            config, stats, whole_documents=whole_documents, selection=selection
        ):
            pass
        selected = int(stats["selected"])
        selected_input_bytes = int(stats["bytes"])
        acquired = int(stats["acquired"])
        progress.update(
            completed_work=acquired,
            unit="documents",
            raw_counters={
                "documents_acquired": acquired,
                "documents_retained": selected,
                "source_bytes": selected_input_bytes,
                "source_bytes_per_second": selected_input_bytes
                / max(time.monotonic() - source_started, 1e-9),
            },
        )
    if not selected:
        raise ValueError("tokenizer training selected no non-empty documents")
    training_contract = {
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "source": config.dataset.source,
        "revision": config.dataset.revision,
        "input_byte_budget_utf8": input_byte_budget,
        "selected_input_bytes_utf8": selected_input_bytes,
        "max_documents": config.max_documents,
        "stop_reason": stats["stop_reason"],
        "acquired_documents": acquired,
        "content_digest_sha256": stats["digest"],
        "source_manifest_sha256": source_manifest_sha256,
        **({"corpus_export": corpus_export} if corpus_export is not None else {}),
        **(
            {"chat_special_tokens": list(config.template.special_tokens)}
            if config.template.special_tokens
            else {}
        ),
        **(
            {
                "sampling": config.sampling.model_dump(mode="json"),
                "sampling_result": stats["sampling"],
            }
            if config.sampling is not None
            else {}
        ),
    }
    if published := _published_fit(output, config, training_contract, selection):
        return published
    with _fit_staging(output) as stage:
        if published := _published_fit(output, config, training_contract, selection):
            return published
        if stage.exists() or stage.is_symlink():
            members = _staged_members(stage)
            if {file.name for file in members} == _FIT_FILES and _matching_fit(
                stage, config, training_contract, selection
            ):
                return _publish_fit(stage, output, reused=True)
            # The lock protects these known files from other fit attempts.
            for file in members:
                file.unlink()
            stage.rmdir()
        stage.mkdir()
        _fit_tokenizer(config, stage, training_contract, stats, selection)
        return _publish_fit(stage, output, reused=False)


def _fit_tokenizer(
    config: TokenizerTrainConfig,
    stage: Path,
    training_contract: dict,
    stats: dict[str, object],
    selection: Selection | None,
) -> None:
    json_path = stage / "tokenizer.json"
    manifest_path = stage / "tokenizer_manifest.json"
    selected = int(stats["selected"])
    selected_input_bytes = int(stats["bytes"])
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        initial_alphabet=ByteLevel.alphabet(),
        # The fixed four keep IDs 0..3; a chat template's markers follow them.
        special_tokens=[*SPECIAL_TOKENS, *config.template.special_tokens],
    )
    bpe_started = time.monotonic()
    with progress_phase(
        "tokenizer_bpe_training",
        completed_work=0,
        total_work=1,
        unit="trainer_calls",
        raw_counters={
            "documents": selected,
            "source_bytes": selected_input_bytes,
        },
    ) as progress:
        replay_stats: dict[str, object] = {}
        tokenizer.train_from_iterator(
            _bounded_documents(
                config,
                replay_stats,
                whole_documents=config.dataset.source in {"local_text", "snapshot"},
                selection=selection,
            ),
            trainer=trainer,
            length=selected,
        )
        if replay_stats != stats:
            raise ValueError(
                "tokenizer training source changed between prefix verification and BPE fit"
            )
        bpe_elapsed = max(time.monotonic() - bpe_started, 1e-9)
        progress.update(
            completed_work=1,
            total_work=1,
            unit="trainer_calls",
            raw_counters={
                "documents": selected,
                "source_bytes": selected_input_bytes,
                "bpe_input_bytes_per_second": selected_input_bytes / bpe_elapsed,
            },
        )
    actual = tokenizer.get_vocab_size()
    if actual != config.vocab_size:
        raise ValueError(
            f"BPE produced {actual} entries, requested {config.vocab_size}; increase document budget"
        )
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise RuntimeError("tokenizer special token IDs are not the required 0..3")
    sampling_audit = None
    if config.sampling is not None:
        from klab.data.tokenizer_sampling import audit

        assert selection is not None
        sampling_audit = audit(config, tokenizer, selection=selection)
        if any(
            group["unknown_tokens"] or group["roundtrip_failures"]
            for group in sampling_audit["strata"].values()
        ) or any(
            not probe["roundtrip"] or probe["unknown_tokens"]
            for probe in sampling_audit["fixed_probes"].values()
        ):
            raise ValueError(
                "tokenizer audit found unknown tokens or failed exact roundtrip"
            )
    with progress_phase(
        "tokenizer_artifact_serialization",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": "tokenizer.json"},
    ) as progress:
        tokenizer.save(str(json_path))
        with json_path.open("rb") as artifact:
            os.fsync(artifact.fileno())
        artifact_bytes = json_path.stat().st_size
        progress.update(
            completed_work=artifact_bytes,
            total_work=artifact_bytes,
            unit="bytes",
            raw_counters={
                "artifact": "tokenizer.json",
                "artifact_bytes": artifact_bytes,
            },
        )
    with progress_phase(
        "tokenizer_artifact_hash",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": "tokenizer.json"},
    ) as progress:
        content_sha256 = sha256_file(json_path)
        progress.update(
            completed_work=artifact_bytes,
            total_work=artifact_bytes,
            unit="bytes",
            raw_counters={
                "artifact": "tokenizer.json",
                "artifact_bytes": artifact_bytes,
            },
        )
    with progress_phase(
        "tokenizer_manifest_serialization",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": "tokenizer_manifest.json"},
    ) as progress:
        _atomic_json(
            manifest_path,
            {
                "sha256": content_sha256,
                "training_contract": training_contract,
                **(
                    {"sampling_audit": sampling_audit}
                    if sampling_audit is not None
                    else {}
                ),
                "license": config.dataset.license
                if config.dataset.source
                in {"local_chat", "local_text", "snapshot", "huggingface"}
                else None,
                "requested_vocab_size": config.vocab_size,
                "vocab_size": actual,
                "special_ids": {
                    token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
                },
                "normalizer": None,
                "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
                "decoder": "ByteLevel",
                "source": config.dataset.source,
                "revision": config.dataset.revision,
                "split": "train",
                "selected_documents": selected,
                "input_byte_budget_utf8": training_contract["input_byte_budget_utf8"],
                "selected_input_bytes_utf8": selected_input_bytes,
                "content_digest_sha256": stats["digest"],
                "source_manifest_sha256": training_contract["source_manifest_sha256"],
                **(
                    {"corpus_export": training_contract["corpus_export"]}
                    if "corpus_export" in training_contract
                    else {}
                ),
                "tokenizers_version": __import__("tokenizers").__version__,
            },
        )
        manifest_bytes = manifest_path.stat().st_size
        progress.update(
            completed_work=manifest_bytes,
            total_work=manifest_bytes,
            unit="bytes",
            raw_counters={
                "artifact": "tokenizer_manifest.json",
                "artifact_bytes": manifest_bytes,
            },
        )
