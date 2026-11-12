"""Offline, validated promotion artifacts for compatible external weights.

It reads one closed source format, a local Hugging Face Llama safetensors
directory. It never downloads a model, imports source code, or guesses a tensor
mapping.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from safetensors import SafetensorError
from safetensors.torch import load_file
from tokenizers import Tokenizer

from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from klab.config.models import RunConfig
from klab.data.tokenizer import SPECIAL_TOKENS
from klab.model.inspection import named_tensor_inventory
from klab.runtime import RuntimeInfo
from klab.training.checkpoints import CheckpointManager, _fsync_directory
from klab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
    architecture_sha256,
    config_sha256,
    sha256_file,
    source_identity,
    write_manifest,
)

_IMPORT_FORMAT_VERSION = 1
_SourceFormat = Literal["hf_llama_safetensors"]


@dataclass(frozen=True)
class WeightImportResult:
    """A finalized promotion-only source run suitable for ``train --promote``."""

    run_dir: Path
    checkpoint: Path
    checkpoint_sha256: str
    source_format: _SourceFormat
    source_config_sha256: str
    source_tokenizer_sha256: str
    provenance: dict[str, str]
    resume_level: Literal["weights_only"] = "weights_only"


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return raw


def _provenance(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise ValueError("provenance must not be a symlink")
    raw = _json(path)
    allowed = {"format_version", "source", "license", "revision", "url"}
    if set(raw) - allowed or raw.get("format_version") != _IMPORT_FORMAT_VERSION:
        raise ValueError("provenance must be format_version 1 with no unknown fields")
    source, license_ = raw.get("source"), raw.get("license")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(license_, str)
        or not license_.strip()
    ):
        raise ValueError("provenance requires non-empty source and license")
    result = {"source": source, "license": license_}
    for name in ("revision", "url"):
        value = raw.get(name)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"provenance.{name} must be a non-empty string or null"
                )
            result[name] = value
    return result


def _tokenizer_identity(path: Path, expected_vocab: int) -> str:
    """Validate the exact Model Lab tokenizer contract before comparing bytes."""
    if path.is_symlink():
        raise ValueError("tokenizer artifact must not be a symlink")
    raw = _json(path)
    model = raw.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("tokenizer must use Model Lab ByteLevel BPE")
    pre, decoder = raw.get("pre_tokenizer"), raw.get("decoder")
    if not isinstance(pre, dict) or pre.get("type") != "ByteLevel":
        raise ValueError("tokenizer must use ByteLevel pre-tokenization")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ValueError("tokenizer must use ByteLevel decoding")
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != expected_vocab:
        raise ValueError("tokenizer vocabulary size differs from destination model")
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise ValueError(
            "tokenizer special-token IDs must be <pad>/<unk>/<bos>/<eos> = 0..3"
        )
    return sha256_file(path)


def _same_tokenizer(source: Path, destination: Path, expected_vocab: int) -> str:
    source_digest = _tokenizer_identity(source, expected_vocab)
    destination_digest = _tokenizer_identity(destination, expected_vocab)
    if source_digest != destination_digest:
        raise ValueError("source and destination tokenizer identities differ")
    return source_digest


def _finite_floating(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} is not a tensor")
    if tensor.layout != torch.strided or tensor.device.type != "cpu":
        raise ValueError(f"{name} is not a CPU strided tensor")
    if tensor.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values")


def _storage_key(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], str]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        str(tensor.dtype),
    )


def _validate_destination_tensors(
    config: RunConfig, tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    expected = named_tensor_inventory(
        config.model,
        config.attention,
        trainable_parameters=config.training.trainable_parameters,
    )
    if set(tensors) != set(expected):
        missing, extra = (
            sorted(set(expected) - set(tensors)),
            sorted(set(tensors) - set(expected)),
        )
        raise ValueError(
            f"mapped tensor inventory differs; missing={missing}, extra={extra}"
        )
    result: dict[str, torch.Tensor] = {}
    for name, spec in expected.items():
        tensor = tensors[name]
        _finite_floating(name, tensor)
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} differs from expected {spec.shape}"
            )
        if spec.alias_of is not None:
            target = tensors[spec.alias_of]
            if _storage_key(tensor) != _storage_key(target):
                raise ValueError(f"{name} must alias {spec.alias_of}")
        result[name] = tensor.detach().contiguous()
    # Recreate aliases after contiguous conversion, preserving a single canonical copy.
    for name, spec in expected.items():
        if spec.alias_of is not None:
            result[name] = result[spec.alias_of]
    return result


def _llama_config(source: Path) -> dict[str, Any]:
    path = source / "config.json"
    if path.is_symlink():
        raise ValueError("Hugging Face source config.json must not be a symlink")
    config = _json(path)
    required = {
        "model_type": "llama",
        "hidden_size": int,
        "intermediate_size": int,
        "num_hidden_layers": int,
        "num_attention_heads": int,
        "num_key_value_heads": int,
        "vocab_size": int,
        "rms_norm_eps": (int, float),
        "rope_theta": (int, float),
        "hidden_act": str,
        "tie_word_embeddings": bool,
        "attention_bias": bool,
        "mlp_bias": bool,
    }
    for name, kind in required.items():
        value = config.get(name)
        if name == "model_type":
            if value != kind:
                raise ValueError("source config is not model_type llama")
        elif (
            not isinstance(value, kind) or isinstance(value, bool) and kind is not bool
        ):
            raise ValueError(f"invalid Llama config field: {name}")
        elif kind is int and value <= 0:
            raise ValueError(f"Llama config field must be positive: {name}")
    if (
        config["num_key_value_heads"] > config["num_attention_heads"]
        or config["num_attention_heads"] % config["num_key_value_heads"]
    ):
        raise ValueError("Llama num_key_value_heads must divide num_attention_heads")
    if config["attention_bias"] or config["mlp_bias"]:
        raise ValueError("biased Llama projections are incompatible with Model Lab")
    if config["hidden_act"] != "silu":
        raise ValueError("only SiLU Llama SwiGLU is compatible")
    if config.get("rope_scaling") not in (None, {}):
        raise ValueError("scaled rotary embeddings are incompatible with Model Lab")
    if config["hidden_size"] % config["num_attention_heads"]:
        raise ValueError("Llama hidden_size must be divisible by num_attention_heads")
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    declared_head_dim = config.get("head_dim", head_dim)
    if type(declared_head_dim) is not int or declared_head_dim != head_dim:
        raise ValueError("Llama head_dim differs from the supported projection layout")
    for name, expected in (
        ("attention_dropout", 0.0),
        ("partial_rotary_factor", 1.0),
        ("pretraining_tp", 1),
    ):
        value = config.get(name, expected)
        if type(value) not in (int, float) or value != expected:
            raise ValueError(f"unsupported Llama config field: {name}")
    for name in ("rope_parameters", "auto_map", "quantization_config"):
        if config.get(name) not in (None, {}):
            raise ValueError(f"unsupported Llama config field: {name}")
    if config.get("architectures") not in (None, ["LlamaForCausalLM"]):
        raise ValueError("unsupported Llama architectures declaration")
    return config


def _llama_weights(
    source: Path, destination: RunConfig
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Hugging Face source must be a non-symlinked local directory")
    config = _llama_config(source)
    model, attention = destination.model, destination.attention
    if (
        destination.ffn_kinds != {"dense"}
        or model.memory != "none"
        or destination.attention_kinds != {"dense"}
    ):
        raise ValueError(
            "Llama import requires dense attention, dense SwiGLU, and no memory adapter"
        )
    checks = {
        "hidden_size": model.hidden_dim,
        "intermediate_size": model.ffn_dim,
        "num_hidden_layers": model.num_layers,
        "num_attention_heads": model.num_heads,
        "num_key_value_heads": model.num_kv_heads or model.num_heads,
        "vocab_size": model.vocab_size,
        "tie_word_embeddings": model.tie_embeddings,
    }
    for name, expected in checks.items():
        if config[name] != expected:
            raise ValueError(f"Llama config {name} differs from destination")
    if not math.isclose(
        float(config["rms_norm_eps"]), model.rms_norm_eps, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("Llama rms_norm_eps differs from destination")
    if not math.isclose(
        float(config["rope_theta"]), attention.rope_base, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("Llama rope_theta differs from destination")
    shards = sorted(
        path
        for path in source.glob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    )
    if not shards:
        raise ValueError("Hugging Face source contains no root safetensors files")
    raw: dict[str, torch.Tensor] = {}
    for shard in shards:
        try:
            loaded = load_file(shard, device="cpu")
        except (SafetensorError, OSError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"invalid safetensors shard {shard.name}: {error}"
            ) from error
        overlap = set(raw).intersection(loaded)
        if overlap:
            raise ValueError(
                f"duplicate tensors across safetensors shards: {sorted(overlap)}"
            )
        raw.update(loaded)
    mapped: dict[str, torch.Tensor] = {
        "embedding.weight": raw.pop("model.embed_tokens.weight", None),
        "norm.weight": raw.pop("model.norm.weight", None),
    }
    if model.tie_embeddings:
        if "lm_head.weight" in raw:
            raise ValueError(
                "tied Llama source must not serialize a separate lm_head.weight"
            )
        mapped["output.weight"] = mapped["embedding.weight"]
    else:
        mapped["output.weight"] = raw.pop("lm_head.weight", None)
    for index in range(model.num_layers):
        source_prefix, target_prefix = f"model.layers.{index}", f"blocks.{index}"
        names = {
            "input_layernorm.weight": "norm1.weight",
            "post_attention_layernorm.weight": "norm2.weight",
            "self_attn.q_proj.weight": "attention.q_proj.weight",
            "self_attn.k_proj.weight": "attention.k_proj.weight",
            "self_attn.v_proj.weight": "attention.v_proj.weight",
            "self_attn.o_proj.weight": "attention.out_proj.weight",
            "mlp.gate_proj.weight": "ffn.gate.weight",
            "mlp.up_proj.weight": "ffn.up.weight",
            "mlp.down_proj.weight": "ffn.down.weight",
        }
        for source_suffix, target_suffix in names.items():
            mapped[f"{target_prefix}.{target_suffix}"] = raw.pop(
                f"{source_prefix}.{source_suffix}", None
            )
    if raw:
        raise ValueError(f"unsupported or unsafe Llama tensor inventory: {sorted(raw)}")
    if any(value is None for value in mapped.values()):
        missing = sorted(name for name, value in mapped.items() if value is None)
        raise ValueError(f"Llama tensor inventory missing: {missing}")
    # HF Llama pairs the first and second head halves in RoPE; Model Lab
    # pairs adjacent coordinates. Permute Q/K output rows, not values/O.
    head_dim = model.hidden_dim // model.num_heads
    for index in range(model.num_layers):
        for projection, heads in (
            ("q_proj", model.num_heads),
            ("k_proj", model.num_kv_heads or model.num_heads),
        ):
            name = f"blocks.{index}.attention.{projection}.weight"
            weight = mapped[name]
            expected_shape = (heads * head_dim, model.hidden_dim)
            if weight.shape != expected_shape:
                raise ValueError(f"Llama rotary projection shape differs: {name}")
            mapped[name] = (
                weight.reshape(heads, 2, head_dim // 2, model.hidden_dim)
                .transpose(1, 2)
                .reshape(expected_shape)
            )
    return (
        _validate_destination_tensors(destination, mapped),
        config,
        canonical_digest(config),
    )


def _import_runtime() -> RuntimeInfo:
    return RuntimeInfo(
        engine="pytorch",
        backend="cpu",
        torch_device="cpu",
        device_index=0,
        device_name="offline weight import",
        physical_device_id=None,
        framework_version=torch.__version__,
        runtime_version=None,
        driver_version=None,
        os="offline-import",
        system_total_bytes=None,
        system_available_bytes=None,
        device_total_bytes=None,
        device_free_bytes=None,
        device_recommended_bytes=None,
        measurement_source="offline-import",
        measured_at=datetime.now(UTC).isoformat(),
        precision_capabilities=("fp32",),
        limitations=("weights imported offline; optimizer state absent",),
    )


def _write_import_checkpoint(
    run_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    config: RunConfig,
    manifest_sha256: str,
) -> tuple[Path, str]:
    manager = CheckpointManager(run_dir, manifest_sha256=manifest_sha256)
    root = run_dir / "checkpoints"
    root.mkdir()
    final = root / "step_00000000_gen_000001"
    temporary = Path(tempfile.mkdtemp(prefix=".import.", dir=root))
    try:
        index, shards = manager._save_weights(
            temporary, tensors
        )  # canonical checkpoint shard writer
        content: dict[str, object] = {
            "format_version": 2,
            "generation_id": 1,
            "step": 0,
            "tokens_seen": 0,
            "created_at": datetime.now(UTC).isoformat(),
            "validation_loss": None,
            "manifest_sha256": manifest_sha256,
            "engine": "pytorch",
            "backend": "cpu",
            "resume_level": "weights_only",
            # A promotion-only import has canonical weights but intentionally no
            # engine-native optimizer, scheduler, cursor, or RNG state.
            "state_codec": None,
            "state_codec_version": None,
            "files": shards,
            "weights": index,
            "config_sha256": config_sha256(config.model_dump(mode="json")),
            "architecture_sha256": architecture_sha256(config.model_dump(mode="json")),
            "source_identity_sha256": None,
            "parent_checkpoint_sha256": None,
            "lineage_best": None,
        }
        digest = canonical_digest(content)
        (temporary / "manifest.json").write_bytes(
            canonical_json({**content, "sha256": digest}) + b"\n"
        )
        with (temporary / "manifest.json").open("rb") as handle:
            os.fsync(handle.fileno())
        report = manager.verify(
            temporary, expected_manifest=manifest_sha256, require_training_state=False
        )
        if not report.valid:
            raise ValueError(
                f"imported checkpoint failed verification: {report.errors}"
            )
        temporary.replace(final)
        _fsync_directory(root)
        return final, digest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def import_weights(
    source: Path,
    destination_run_dir: Path,
    destination_config: RunConfig,
    *,
    source_format: _SourceFormat,
    source_tokenizer: Path,
    provenance: Path,
) -> WeightImportResult:
    """Import a closed local source format as an immutable weights-only run.

    ``destination_run_dir`` must not exist.  The returned checkpoint is accepted
    by the existing explicit promotion path; it cannot be used for full resume.
    """
    if source.is_symlink():
        raise ValueError("weight-import source must not be a symlink")
    source, destination_run_dir = source.resolve(), destination_run_dir.resolve()
    if source_format == "hf_llama_safetensors":
        expected_tokenizer = source / "tokenizer.json"
        if (
            source_tokenizer.is_symlink()
            or source_tokenizer.resolve() != expected_tokenizer.resolve()
        ):
            raise ValueError(
                "Llama import requires the source directory's non-symlinked tokenizer.json"
            )
    if destination_run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite import destination: {destination_run_dir}"
        )
    if destination_config.runtime.engine != "pytorch":
        raise ValueError("current importer writes PyTorch canonical weights only")
    provenance_data = _provenance(provenance)
    tokenizer_digest = _same_tokenizer(
        source_tokenizer,
        destination_config.tokenizer.path,
        destination_config.model.vocab_size,
    )
    if source_format == "hf_llama_safetensors":
        tensors, _source_config, source_config_digest = _llama_weights(
            source, destination_config
        )
    else:
        raise ValueError(f"unsupported import source format: {source_format}")
    parent = destination_run_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_run_dir.name}.import.", dir=parent)
    )
    try:
        tokenizer_target = temporary / "tokenizer.json"
        shutil.copyfile(destination_config.tokenizer.path, tokenizer_target)
        with tokenizer_target.open("rb") as handle:
            os.fsync(handle.fileno())
        (temporary / "resolved_config.yaml").write_text(
            yaml.safe_dump(destination_config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        with (temporary / "resolved_config.yaml").open("rb") as handle:
            os.fsync(handle.fileno())
        artifacts = (
            ArtifactIdentity(
                "tokenizer.json", tokenizer_digest, tokenizer_target.stat().st_size
            ),
            ArtifactIdentity(
                "resolved_config.yaml",
                sha256_file(temporary / "resolved_config.yaml"),
                (temporary / "resolved_config.yaml").stat().st_size,
            ),
        )
        manifest = RunManifest(
            run_id=destination_run_dir.name,
            name=destination_config.name,
            runtime=_import_runtime(),
            requested_config=destination_config.model_dump(mode="json"),
            effective_config=destination_config.model_dump(mode="json"),
            architecture_sha256=architecture_sha256(
                destination_config.model_dump(mode="json")
            ),
            source_identity=source_identity(),
            worker_id="offline-weight-import",
            continuation_kind="PROMOTED",
            purpose="training",
            artifacts=artifacts,
            resource_decisions=(
                {
                    "kind": "weight_import",
                    "source_format": source_format,
                    "source_config_sha256": source_config_digest,
                    "source_tokenizer_sha256": tokenizer_digest,
                    "provenance": provenance_data,
                    "resume_level": "weights_only",
                    "reason": "validated offline canonical weight import",
                },
            ),
        )
        manifest_digest = write_manifest(temporary / "manifest.json", manifest)
        checkpoint, checkpoint_digest = _write_import_checkpoint(
            temporary, tensors, destination_config, manifest_digest
        )
        temporary.replace(destination_run_dir)
        _fsync_directory(parent)
        return WeightImportResult(
            destination_run_dir,
            destination_run_dir / checkpoint.relative_to(temporary),
            checkpoint_digest,
            source_format,
            source_config_digest,
            tokenizer_digest,
            provenance_data,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
