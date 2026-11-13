"""Load one frozen checkpoint through its native engine and run-owned artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig
from klab.data.packing import (
    TokenBlockDataset,
    load_prepared_data,
    supervision_requires_mask,
)
from klab.data.tokenizer import load_tokenizer
from klab.data.verification import _receipt_from_proofs
from klab.engines.base import Microbatch
from klab.engines.mlx import MLXEngine, preserve_rng_state
from klab.evaluation.evidence import _validated_artifacts
from klab.model.inspection import inspection_report
from klab.model.transformer import DenseLM
from klab.runtime import discover_runtimes, select_device, torch_device_for
from klab.runtime_profile import RuntimeAuthorization, require_authorization
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import read_manifest
from klab.verification_proofs import ProofStore, VerificationMode


@dataclass(frozen=True)
class InferenceRun:
    run: Path
    config: RunConfig
    model: Any
    tokenizer: Tokenizer
    device: torch.device | str
    identity: dict[str, Any]
    engine: MLXEngine | None = None

    def validation_dataset(self) -> TokenBlockDataset:
        root = self.run / "data"
        addresses = root / "validation_byte_addresses.npy"
        supervision = root / "validation_supervision.npy"
        needs_mask = supervision_requires_mask(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        )
        if supervision.is_file() != needs_mask:
            raise ValueError(
                "validation supervision inventory differs from packing descriptor"
            )
        return TokenBlockDataset(
            np.load(root / "validation.npy", mmap_mode="r", allow_pickle=False),
            self.config.training.seq_len,
            np.load(addresses, mmap_mode="r", allow_pickle=False)
            if self.config.model.memory in {"byte", "portable"}
            else None,
            np.load(supervision, mmap_mode="r", allow_pickle=False)
            if needs_mask
            else None,
        )

    def evaluate(
        self, observer: Callable[[Any, Any], None] | None = None
    ) -> dict[str, float | int | str | None]:
        """Evaluate through the run's native engine without changing its RNG or mode.

        ``observer(logits, targets)`` sees each scored PyTorch batch.
        """
        dataset = self.validation_dataset()
        if self.engine is None:
            from klab.evaluation.language_model import evaluate

            assert isinstance(self.device, torch.device)
            return evaluate(
                self.model,
                dataset,
                batch_size=self.config.evaluation.batch_size
                or self.config.training.micro_batch_size,
                max_batches=self.config.evaluation.max_batches,
                device=self.device,
                observer=observer,
            )
        if observer is not None:
            raise ValueError("evaluation observers need the PyTorch engine")
        batch_size = (
            self.config.evaluation.batch_size or self.config.training.micro_batch_size
        )
        limit = min(len(dataset), batch_size * self.config.evaluation.max_batches)

        def batches() -> Any:
            for start in range(0, limit, batch_size):
                records = [
                    dataset.numpy_block(index)
                    for index in range(start, min(start + batch_size, limit))
                ]
                inputs, targets, addresses = zip(*records, strict=True)
                if any(address is not None for address in addresses):
                    raise ValueError(
                        "MLX evaluation does not support byte-address memory"
                    )
                yield Microbatch(np.stack(inputs), np.stack(targets))

        return self.engine.evaluate(batches()).to_report()

    def text_bits(self, count: int | None = None) -> dict[str, Any] | None:
        """Bits per byte on the first COUNT held-out documents.

        COUNT defaults to ``evaluation.text_documents``; None when neither is set.
        """
        count = count or self.config.evaluation.text_documents
        if count is None:
            return None
        if self.engine is not None:
            raise ValueError("bits per byte needs the PyTorch engine")
        from klab.evaluation.text_bits import heldout_documents, score_documents

        return score_documents(
            self,
            heldout_documents(self.run, self.tokenizer, count),
            batch_size=self.config.evaluation.batch_size
            or self.config.training.micro_batch_size,
        )


def _verified_checkpoint(
    run: Path,
    checkpoint: str | None,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> tuple[RunConfig, dict[str, Any], Path, dict[str, Any], CheckpointManager]:
    config = RunConfig.model_validate_json((run / "resolved_config.yaml").read_text())
    manifest = read_manifest(run / "manifest.json")
    manifest_digest = canonical_digest(manifest)
    if RunConfig.model_validate(manifest["effective_config"]) != config:
        raise ValueError("resolved config differs from the verified run manifest")
    selected = Path(checkpoint or "latest.json")
    if not selected.is_absolute() and not selected.exists():
        selected = (
            run / selected
            if selected.parts[0] == "checkpoints"
            else run / "checkpoints" / selected
        )
    selected = selected.resolve()
    expected_digest = None
    if selected.name in {"latest.json", "best.json"}:
        pointer = json.loads(selected.read_text())
        relative = pointer.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("inference requires a validated v2 checkpoint pointer")
        expected_digest = pointer["manifest_sha256"]
        selected = (selected.parent / relative).resolve()
    if selected.parent != run / "checkpoints" or not selected.is_dir():
        raise ValueError(
            "selected checkpoint must be a generation belonging to this run"
        )
    metadata = json.loads((selected / "manifest.json").read_text())
    if expected_digest is not None and metadata["sha256"] != expected_digest:
        raise ValueError("checkpoint pointer digest does not match selected generation")
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    report = manager.verify(
        selected,
        expected_manifest=manifest_digest,
        require_training_state=False,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    if not report.valid:
        raise ValueError(f"invalid checkpoint: {report.errors}")
    if (
        metadata["engine"] != config.runtime.engine
        or metadata["backend"] != config.runtime.backend
    ):
        raise ValueError(
            "checkpoint runtime differs from the verified run configuration"
        )
    return config, manifest, selected, metadata, manager


def _ephemeral_config(config: RunConfig, backend: str | None) -> RunConfig:
    requested = backend or config.runtime.backend
    if config.runtime.engine == "mlx" and backend not in {None, "auto", "metal"}:
        raise ValueError(
            "MLX inference only supports the stored metal backend; "
            "backend overrides cannot select a PyTorch device"
        )
    return config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"backend": requested, "precision": "fp32", "device_index": 0}
            )
        }
    )


def load_run(
    run_id: str,
    runs_dir: Path,
    checkpoint: str | None = None,
    backend: str | None = None,
    *,
    authorization: RuntimeAuthorization | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    expected_checkpoint_sha256: str | None = None,
    quantized: Path | None = None,
) -> InferenceRun:
    """Load a verified checkpoint; QUANTIZED applies a ``weights quantize``
    artifact made from exactly this checkpoint (PyTorch engine only)."""
    run = (runs_dir / run_id).resolve()
    config, manifest, selected, metadata, manager = _verified_checkpoint(
        run, checkpoint, proof_store=proof_store, verification_mode=verification_mode
    )
    if (
        expected_checkpoint_sha256 is not None
        and metadata["sha256"] != expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint digest differs from the sealed lab record")
    runtime_config = _ephemeral_config(config, backend)
    requested = runtime_config.runtime.backend
    require_authorization(runtime_config, authorization)
    artifacts, array_proofs = _validated_artifacts(
        run, manifest, proof_store=proof_store, verification_mode=verification_mode
    )
    required = {
        "tokenizer.json",
        "data/manifest.json",
        "data/train.npy",
        "data/validation.npy",
    }
    data_manifest = json.loads((run / "data" / "manifest.json").read_text())
    if not isinstance(data_manifest, dict):
        raise TypeError("prepared-data manifest must be an object")
    if supervision_requires_mask(data_manifest):
        required.update(
            {"data/train_supervision.npy", "data/validation_supervision.npy"}
        )
    if config.model.memory in {"byte", "portable"}:
        required.update(
            {"data/train_byte_addresses.npy", "data/validation_byte_addresses.npy"}
        )
    if not required <= artifacts.keys():
        raise ValueError(
            f"run lacks verified artifacts: {sorted(required - artifacts.keys())}"
        )
    receipt = _receipt_from_proofs(run / "data", data_manifest, array_proofs)
    load_prepared_data(
        run / "data",
        byte_enabled=config.model.memory in {"byte", "portable"},
        verification="structural",
        receipt=receipt,
    )

    # The checkpoint and training runtime were verified before authorization.
    if config.model.memory_token_mapping is not None:
        from klab.memory_tokenizer import validate_memory_token_map

        validate_memory_token_map(
            config.model.memory_token_mapping.model_dump(mode="json"),
            tokenizer_path=run / "tokenizer.json",
        )
    tokenizer = load_tokenizer(run / "tokenizer.json")
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("run tokenizer vocabulary differs from model configuration")

    weights = manager.load(
        selected,
        "promote",
        proof_store=proof_store,
        verification_mode=verification_mode,
    ).model
    identity = {
        "run_id": run_id,
        "checkpoint_sha256": metadata["sha256"],
        "checkpoint_relative_path": str(selected.relative_to(run)),
        "step": metadata["step"],
        "tokens_seen": metadata["tokens_seen"],
        "config": config.model_dump(mode="json"),
        "tokenizer_sha256": artifacts["tokenizer.json"],
        "data_sha256": {
            key: artifacts[f"data/{key}.npy"] for key in ("train", "validation")
        },
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "training_runtime": {
            key: manifest["runtime"][key]
            for key in (
                "engine",
                "backend",
                "device_index",
                "device_name",
                "framework_version",
                "os",
            )
        },
    }
    if quantized is not None and config.runtime.engine != "pytorch":
        raise ValueError("quantized weights need the PyTorch engine (not MLX)")
    if config.runtime.engine == "mlx":
        # Native construction changes MLX/Python/NumPy state.  Weight loading
        # does not need caller RNG, and no Torch model is constructed here.
        with torch.random.fork_rng(devices=[]), preserve_rng_state():
            engine = MLXEngine()
            engine.validate(runtime_config, authorization=authorization)
            engine.initialize(runtime_config, initial_weights=weights)
            model = engine.model
            assert model is not None
            model.eval()
            runtime = engine.runtime
        del weights
        assert model is not None and runtime is not None
        identity.update(
            {
                "runtime": {
                    **runtime.as_dict(),
                    "engine": "mlx",
                    "backend": "metal",
                    "device_index": 0,
                    "precision": "fp32",
                },
                "parameter_inventory": inspection_report(config),
            }
        )
        return InferenceRun(run, config, model, tokenizer, "metal", identity, engine)

    if config.runtime.engine != "pytorch":
        raise ValueError(f"unsupported inference engine: {config.runtime.engine}")
    selected_device = select_device(requested)
    actual_backend = (
        "rocm"
        if selected_device.type == "cuda" and torch.version.hip
        else selected_device.type
    )
    device = torch_device_for(actual_backend, runtime_config.runtime.device_index)
    model_config = config.model
    if model_config.memory_package_path is not None:
        if "portable_package" not in artifacts:
            raise ValueError("run lacks a verified portable memory package")
        model_config = model_config.model_copy(
            update={"memory_package_path": run / "portable_package"}
        )
    with torch.random.fork_rng(devices=[]):
        model = DenseLM(model_config, config.attention)
        model.load_state_dict(weights)
    del weights
    quantization = None
    if quantized is not None:
        from klab.evaluation.quantization import apply_artifact

        quantization = apply_artifact(
            model, quantized, checkpoint_sha256=metadata["sha256"]
        )
    model.to(device).eval()
    runtime = next(
        info
        for info in discover_runtimes()
        if info.engine == "pytorch" and info.backend == actual_backend
    )
    device_name = runtime.device_name
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "xpu":
        device_name = torch.xpu.get_device_name(device)
    identity.update(
        {
            "runtime": {
                "engine": "pytorch",
                "backend": actual_backend,
                "device_index": runtime_config.runtime.device_index,
                "device_name": device_name,
                "framework_version": runtime.framework_version,
                "os": runtime.os,
                "precision": "fp32",
            },
            "parameter_inventory": inspection_report(config),
        }
    )
    if quantization is not None:
        identity["quantization"] = quantization
    return InferenceRun(run, config, model, tokenizer, device, identity)


def write_inference_result(run: Path, kind: str, result: dict[str, Any]) -> Path:
    """Content-addressed results preserve earlier checkpoints and observations."""
    content = canonical_json(result) + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    path = run / "evaluations" / f"{kind}-{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"result artifact conflict: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)
    return path
