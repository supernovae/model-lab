"""Pinned, safetensors-only observational adapter for Pythia checkpoints.

This module deliberately evaluates a fixed external model on a separately pinned
FineWeb-Edu validation slice.  It does not import, convert, or compare weights
with Model Lab models.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import os
import platform
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from klab.canonical import canonical_json
from klab.config.models import RunConfig
from klab.data.datasets import iter_documents
from klab.reference_models import (
    safe_snapshot,
    validate_model_metadata,
    validate_snapshot,
)
from klab.training.manifest import config_sha256, sha256_file

PYTHIA_MODEL_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_LICENSE = "Apache-2.0"
REPORT_FORMAT = "klab-pythia-reference-trajectory"
REPORT_VERSION = 1


@dataclass(frozen=True)
class PythiaCheckpoint:
    """One official immutable checkpoint revision."""

    step: str
    commit: str


PYTHIA_70M_DEDUPED_CHECKPOINTS: Mapping[str, PythiaCheckpoint] = {
    "step0": PythiaCheckpoint("step0", "c913ae980de9355947d0bf73f9f10d580eb79301"),
    "step10000": PythiaCheckpoint(
        "step10000", "c890c8f6d8f86c36b2af66c3012a14ef1d35d3f3"
    ),
    "step143000": PythiaCheckpoint(
        "step143000", "9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb"
    ),
}
DEFAULT_STEPS = tuple(PYTHIA_70M_DEDUPED_CHECKPOINTS)


def checkpoint_for_step(step: str) -> PythiaCheckpoint:
    """Return a registered checkpoint; arbitrary Hub revisions are forbidden."""
    try:
        return PYTHIA_70M_DEDUPED_CHECKPOINTS[step]
    except KeyError as error:
        raise ValueError(
            f"unsupported Pythia step {step!r}; allowed steps: "
            + ", ".join(DEFAULT_STEPS)
        ) from error


def validate_reference_config(config: RunConfig) -> None:
    """Require the separately pinned FineWeb-Edu pathway for this observation."""
    dataset = config.dataset
    if (
        dataset.source != "fineweb_edu"
        or not dataset.revision
        or not dataset.dataset_config
    ):
        raise ValueError(
            "Pythia reference trajectory requires FineWeb-Edu "
            "dataset.source=fineweb_edu with non-null dataset.revision and "
            "dataset.dataset_config"
        )


def _validate_steps(steps: Sequence[str]) -> tuple[PythiaCheckpoint, ...]:
    if not steps:
        raise ValueError("at least one registered Pythia step is required")
    if len(set(steps)) != len(steps):
        raise ValueError("Pythia steps must not repeat")
    return tuple(checkpoint_for_step(step) for step in steps)


def _validate_snapshot(snapshot: Path) -> list[Path]:
    """Reject every downloaded file outside the safe static/safetensor allowlist."""
    return validate_snapshot(snapshot, "Pythia")


def _validate_model_metadata(snapshot: Path) -> None:
    validate_model_metadata(
        snapshot,
        model_type="gpt_neox",
        architecture="GPTNeoXForCausalLM",
        label="Pythia",
    )


def _validation_documents(config: RunConfig) -> tuple[str, ...]:
    documents = tuple(
        itertools.islice(
            iter_documents(config.dataset, "validation"),
            config.dataset.validation_max_documents,
        )
    )
    if not documents:
        raise ValueError("FineWeb-Edu validation selection is empty")
    return documents


def _bounded_tokenized_documents(
    documents: Sequence[str],
    tokenized: Sequence[Sequence[int]],
    token_limit: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    selected_documents: list[str] = []
    selected_tokens: list[tuple[int, ...]] = []
    consumed = 0
    for document, tokens in zip(documents, tokenized, strict=True):
        if consumed + len(tokens) > token_limit:
            break
        selected_documents.append(document)
        selected_tokens.append(tuple(tokens))
        consumed += len(tokens)
    if not selected_documents:
        raise ValueError("FineWeb-Edu validation limit excludes every document")
    return tuple(selected_documents), tuple(selected_tokens)


def _snapshot(
    checkpoint: PythiaCheckpoint, cache_dir: Path | None
) -> tuple[Path, list[Path]]:
    return safe_snapshot(
        PYTHIA_MODEL_ID,
        checkpoint.commit,
        cache_dir,
        model_type="gpt_neox",
        architecture="GPTNeoXForCausalLM",
        label="Pythia",
    )


def _document_digest(documents: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        encoded = document.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError(f"unsupported Pythia device {device!r}")


def _tokenize_documents(
    tokenizer: Any, documents: Sequence[str]
) -> tuple[tuple[int, ...], ...]:
    encoded = tokenizer(
        list(documents), add_special_tokens=False, return_attention_mask=False
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list) or len(input_ids) != len(documents):
        raise ValueError("Pythia tokenizer returned malformed input IDs")
    result: list[tuple[int, ...]] = []
    for ids in input_ids:
        if not isinstance(ids, list) or any(
            not isinstance(token, int) for token in ids
        ):
            raise ValueError("Pythia tokenizer returned non-integer input IDs")
        result.append(tuple(ids))
    return tuple(result)


def _evaluate_checkpoint(
    checkpoint: PythiaCheckpoint,
    snapshot: Path,
    device: torch.device,
    tokenized_documents: Sequence[Sequence[int]],
    context: int,
) -> dict[str, object]:
    try:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "Pythia reference support requires the optional 'reference' dependency"
        ) from error
    model_config = GPTNeoXConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    if context < 2:
        raise ValueError("Pythia evaluation requires training.seq_len >= 2")
    if context > model_config.max_position_embeddings:
        raise ValueError(
            "configured evaluation context exceeds Pythia max_position_embeddings"
        )
    model = GPTNeoXForCausalLM.from_pretrained(
        snapshot,
        config=model_config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    model.eval()
    loss_sum = 0.0
    target_count = 0
    try:
        with torch.inference_mode():
            for document in tokenized_documents:
                for start in range(0, max(0, len(document) - 1), context - 1):
                    chunk = document[start : start + context]
                    if len(chunk) < 2:
                        continue
                    input_ids = torch.tensor(
                        chunk, dtype=torch.long, device=device
                    ).unsqueeze(0)
                    logits = model(input_ids=input_ids).logits[:, :-1, :]
                    targets = input_ids[:, 1:]
                    loss_sum += float(
                        functional.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            reduction="sum",
                        ).item()
                    )
                    target_count += targets.numel()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    if target_count == 0:
        raise ValueError("FineWeb-Edu validation selection has no next-token targets")
    return {
        "step": checkpoint.step,
        "commit": checkpoint.commit,
        "loss": loss_sum / target_count,
        "loss_sum": loss_sum,
        "target_tokens": target_count,
    }


def _file_inventory(snapshot: Path, files: Sequence[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]


def _write_exclusive_json(output: Path, payload: Mapping[str, object]) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Pythia reference output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output, follow_symlinks=False)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def evaluate_pythia_trajectory(
    config: RunConfig,
    output: Path,
    *,
    cache_dir: Path | None = None,
    device: str = "cpu",
    steps: Sequence[str] = DEFAULT_STEPS,
) -> Path:
    """Download registered safe snapshots and observe them on one pinned data slice."""
    validate_reference_config(config)
    checkpoints = _validate_steps(steps)
    selected_device = _device(device)
    documents = _validation_documents(config)
    document_digest: str | None = None

    tokenizer = None
    tokenized_documents: tuple[tuple[int, ...], ...] | None = None
    tokenizer_inventory: list[dict[str, object]] | None = None
    results: list[dict[str, object]] = []
    snapshot_inventories: dict[str, list[dict[str, object]]] = {}
    for checkpoint in checkpoints:
        snapshot, files = _snapshot(checkpoint, cache_dir)
        if tokenizer is None:
            try:
                from transformers import GPTNeoXTokenizerFast
            except ImportError as error:
                raise RuntimeError(
                    "Pythia reference support requires the optional 'reference' dependency"
                ) from error
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(
                snapshot, local_files_only=True, trust_remote_code=False
            )
            tokenized = _tokenize_documents(tokenizer, documents)
            documents, tokenized_documents = _bounded_tokenized_documents(
                documents, tokenized, config.dataset.validation_max_tokens
            )
            document_digest = _document_digest(documents)
            tokenizer_inventory = _file_inventory(
                snapshot,
                [
                    path
                    for path in files
                    if path.name
                    in {
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "vocab.json",
                        "merges.txt",
                    }
                ],
            )
        assert tokenized_documents is not None
        results.append(
            _evaluate_checkpoint(
                checkpoint,
                snapshot,
                selected_device,
                tokenized_documents,
                config.training.seq_len,
            )
        )
        snapshot_inventories[checkpoint.step] = _file_inventory(snapshot, files)

    assert document_digest is not None
    target_token_count = sum(max(0, len(ids) - 1) for ids in tokenized_documents)
    payload: dict[str, object] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "observation_scope": (
            "Pythia checkpoints observed on the same bounded FineWeb-Edu validation "
            "documents. This is not a controlled comparison with Model Lab models, "
            "tokenizers, data, or training."
        ),
        "model": {
            "id": PYTHIA_MODEL_ID,
            "license": PYTHIA_LICENSE,
            "load_policy": {
                "trust_remote_code": False,
                "use_safetensors": True,
                "local_files_only_after_snapshot": True,
                "pickle_weights": False,
            },
        },
        "dataset": {
            "source": config.dataset.source,
            "revision": config.dataset.revision,
            "dataset_config": config.dataset.dataset_config,
            "validation_max_documents": config.dataset.validation_max_documents,
            "validation_max_tokens": config.dataset.validation_max_tokens,
            "content_sha256": document_digest,
            "license": "ODC-BY-1.0",
            "documents": len(documents),
        },
        "evaluation": {
            "observer_config_sha256": config_sha256(config.model_dump(mode="json")),
            "context_tokens": config.training.seq_len,
            "target_counting": "each within-document shifted next token counted once",
            "tokenized_document_target_tokens": target_token_count,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "tokenizer_files": tokenizer_inventory,
        "snapshots": snapshot_inventories,
        "checkpoints": results,
    }
    return _write_exclusive_json(output, payload)
