from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from klab.config.models import RunConfig
from klab.model.transformer import DenseLM
from klab.training.checkpoints import CheckpointManager
from klab.training.weight_import import import_weights


def _tokenizer(path: Path, *, add_prefix_space: bool = False) -> None:
    value = Tokenizer(BPE(unk_token="<unk>"))
    value.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)
    value.decoder = ByteLevelDecoder()
    value.train_from_iterator(
        ["tiny corpus"],
        BpeTrainer(
            vocab_size=260,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        ),
    )
    assert value.get_vocab_size() == 260
    value.save(str(path))


def _config(tmp_path: Path, tokenizer: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "import-test",
            "seed": 1,
            "runtime": {
                "engine": "pytorch",
                "backend": "cpu",
                "device_index": 0,
                "precision": "fp32",
                "memory": {
                    "policy": "balanced",
                    "max_device_memory_fraction": 0.9,
                    "budget_bytes": None,
                    "activation_checkpointing": {
                        "enabled": False,
                        "strategy": "transformer_block",
                    },
                    "activation_offload": {"enabled": False},
                    "allowed_sequence_lengths": [],
                    "allowed_optimizers": [],
                },
            },
            "model": {
                "vocab_size": 260,
                "hidden_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 16,
                "max_seq_len": 16,
            },
            "tokenizer": {"path": str(tokenizer)},
            "dataset": {
                "source": "synthetic",
                "cache_dir": str(tmp_path / "data"),
                "train_max_documents": 1,
                "validation_max_documents": 1,
                "train_max_tokens": 32,
                "validation_max_tokens": 32,
            },
            "training": {
                "micro_batch_size": 1,
                "gradient_accumulation": 1,
                "seq_len": 8,
                "max_steps": 2,
                "max_tokens": 16,
            },
            "optimizer": {"name": "adamw", "warmup_steps": 1},
            "logging": {"root_dir": str(tmp_path / "runs")},
        }
    )


def _provenance(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {"format_version": 1, "source": "local fixture", "license": "Apache-2.0"}
        )
    )
    return path


def _hf_llama_rotary_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    head_dim = weight.shape[0] // heads
    return (
        weight.reshape(heads, head_dim // 2, 2, weight.shape[1])
        .transpose(1, 2)
        .reshape_as(weight)
    )


def _llama_export(root: Path, config: RunConfig) -> dict[str, torch.Tensor]:
    root.mkdir()
    state = DenseLM(config.model, config.attention).state_dict()
    source_names = {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "model.layers.0.input_layernorm.weight": "blocks.0.norm1.weight",
        "model.layers.0.post_attention_layernorm.weight": "blocks.0.norm2.weight",
        "model.layers.0.self_attn.q_proj.weight": "blocks.0.attention.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight": "blocks.0.attention.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight": "blocks.0.attention.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight": "blocks.0.attention.out_proj.weight",
        "model.layers.0.mlp.gate_proj.weight": "blocks.0.ffn.gate.weight",
        "model.layers.0.mlp.up_proj.weight": "blocks.0.ffn.up.weight",
        "model.layers.0.mlp.down_proj.weight": "blocks.0.ffn.down.weight",
    }
    exported = {
        source: (
            _hf_llama_rotary_rows(state[target], config.model.num_heads)
            if source.endswith("q_proj.weight")
            else _hf_llama_rotary_rows(
                state[target], config.model.num_kv_heads or config.model.num_heads
            )
            if source.endswith("k_proj.weight")
            else state[target]
        )
        for source, target in source_names.items()
    }
    save_file(exported, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": config.model.hidden_dim,
                "intermediate_size": config.model.ffn_dim,
                "num_hidden_layers": config.model.num_layers,
                "num_attention_heads": config.model.num_heads,
                "num_key_value_heads": config.model.num_kv_heads
                or config.model.num_heads,
                "vocab_size": config.model.vocab_size,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "hidden_act": "silu",
                "tie_word_embeddings": True,
                "attention_bias": False,
                "mlp_bias": False,
            }
        )
    )
    return state


def test_llama_import_preserves_exact_dense_mapping(tmp_path: Path) -> None:
    # Frozen outputs came from actual Transformers 4.57.1 LlamaForCausalLM,
    # not DenseLM with renamed state keys; this detects the rotary basis mismatch.
    fixture = Path(__file__).with_name("fixtures") / "llama_reference"
    source = fixture / "source"
    tokenizer = source / "tokenizer.json"
    config = _config(tmp_path, tokenizer)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "vocab_size": 512,
                    "hidden_dim": 16,
                    "ffn_dim": 32,
                    "max_seq_len": 32,
                }
            )
        }
    )
    result = import_weights(
        source,
        tmp_path / "imported",
        config,
        source_format="hf_llama_safetensors",
        source_tokenizer=tokenizer,
        provenance=fixture / "provenance.json",
    )
    generation_manifest = json.loads(
        (result.checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    assert generation_manifest["resume_level"] == "weights_only"
    assert generation_manifest["state_codec"] is None
    assert generation_manifest["state_codec_version"] is None
    assert (
        CheckpointManager(result.run_dir)
        .verify(result.checkpoint, require_training_state=False)
        .valid
    )
    manager = CheckpointManager(result.run_dir)
    with pytest.raises(ValueError):
        manager.load(result.checkpoint, "resume")
    loaded = manager.load(result.checkpoint, "promote")
    promoted_model = DenseLM(config.model, config.attention).eval()
    promoted_model.load_state_dict(loaded.model)
    reference = load_file(fixture / "reference.safetensors")
    with torch.no_grad():
        actual = promoted_model(reference["input_ids"])
    torch.testing.assert_close(
        actual, reference["expected_logits"], atol=1e-6, rtol=1e-5
    )
    assert loaded.step == loaded.tokens_seen == 0


def test_llama_import_rejects_incompatible_tokenizer_and_unsafe_tensor(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer)
    config = _config(tmp_path, tokenizer)
    source = tmp_path / "llama"
    _llama_export(source, config)
    _tokenizer(source / "tokenizer.json", add_prefix_space=True)
    with pytest.raises(ValueError, match="tokenizer identities"):
        import_weights(
            source,
            tmp_path / "bad-tokenizer",
            config,
            source_format="hf_llama_safetensors",
            source_tokenizer=source / "tokenizer.json",
            provenance=_provenance(tmp_path / "provenance.json"),
        )
    shutil.copyfile(tokenizer, source / "tokenizer.json")
    tensors = dict(load_file(source / "model.safetensors", device="cpu"))
    tensors["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(8)
    save_file(tensors, source / "model.safetensors")
    with pytest.raises(
        ValueError, match="unsupported or unsafe Llama tensor inventory"
    ):
        import_weights(
            source,
            tmp_path / "unsafe-member",
            config,
            source_format="hf_llama_safetensors",
            source_tokenizer=source / "tokenizer.json",
            provenance=tmp_path / "provenance.json",
        )


def test_llama_import_rejects_incompatible_grouped_query_layout(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer)
    config = _config(tmp_path, tokenizer)
    source = tmp_path / "llama"
    _llama_export(source, config)
    shutil.copyfile(tokenizer, source / "tokenizer.json")
    raw = json.loads((source / "config.json").read_text())
    raw["num_key_value_heads"] = 1
    (source / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="num_key_value_heads differs"):
        import_weights(
            source,
            tmp_path / "bad-architecture",
            config,
            source_format="hf_llama_safetensors",
            source_tokenizer=source / "tokenizer.json",
            provenance=_provenance(tmp_path / "provenance.json"),
        )
    raw["num_key_value_heads"] = 2
    (source / "config.json").write_text(json.dumps(raw))
    tensors = dict(load_file(source / "model.safetensors", device="cpu"))
    tensors["model.embed_tokens.weight"][0, 0] = float("nan")
    save_file(tensors, source / "model.safetensors")
    with pytest.raises(ValueError, match="non-finite"):
        import_weights(
            source,
            tmp_path / "nonfinite",
            config,
            source_format="hf_llama_safetensors",
            source_tokenizer=source / "tokenizer.json",
            provenance=tmp_path / "provenance.json",
        )


def test_llama_import_round_trips_grouped_query_weights(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer)
    config = _config(tmp_path, tokenizer)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "hidden_dim": 16,
                    "ffn_dim": 32,
                }
            )
        }
    )
    source = tmp_path / "llama-gqa"
    expected = _llama_export(source, config)
    shutil.copyfile(tokenizer, source / "tokenizer.json")

    result = import_weights(
        source,
        tmp_path / "imported-gqa",
        config,
        source_format="hf_llama_safetensors",
        source_tokenizer=source / "tokenizer.json",
        provenance=_provenance(tmp_path / "provenance.json"),
    )
    loaded = CheckpointManager(result.run_dir).load(result.checkpoint, "promote")
    for name, tensor in expected.items():
        torch.testing.assert_close(loaded.model[name], tensor)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_dim", 8),
        ("attention_dropout", 0.2),
        ("partial_rotary_factor", 0.5),
        ("rope_parameters", {"rope_type": "linear", "factor": 2.0}),
        ("auto_map", {"AutoModelForCausalLM": "custom.Model"}),
    ],
)
def test_llama_import_rejects_unrepresented_source_semantics(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer)
    config = _config(tmp_path, tokenizer)
    source = tmp_path / "llama"
    _llama_export(source, config)
    shutil.copyfile(tokenizer, source / "tokenizer.json")
    raw = json.loads((source / "config.json").read_text())
    raw[field] = value
    (source / "config.json").write_text(json.dumps(raw))
    destination = tmp_path / "incompatible"
    with pytest.raises(ValueError):
        import_weights(
            source,
            destination,
            config,
            source_format="hf_llama_safetensors",
            source_tokenizer=source / "tokenizer.json",
            provenance=_provenance(tmp_path / "provenance.json"),
        )
    assert not destination.exists()


def test_legacy_v1_import_migrates_config_and_stays_promotion_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from klab.cli.main import build_parser

    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer)
    config = _config(tmp_path, tokenizer)
    model = DenseLM(config.model, config.attention)
    legacy_config = {
        "schema_version": 1,
        "name": config.name,
        "seed": config.seed,
        "device": "cpu",
        "model": config.model.model_dump(mode="json"),
        "tokenizer": {"path": str(tokenizer)},
        "dataset": config.dataset.model_dump(mode="json"),
        "training": {
            "batch_size": 1,
            "seq_len": 8,
            "max_steps": 2,
            "max_tokens": 16,
            "grad_clip_norm": 1.0,
            "deterministic": True,
        },
        "optimizer": {
            "learning_rate": 3e-4,
            "min_learning_rate": 3e-5,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "attention": {},
        "evaluation": {"every_steps": 1, "max_batches": 1},
        "logging": {
            "root_dir": str(tmp_path / "legacy-runs"),
            "every_steps": 1,
            "checkpoint_every_steps": 1,
        },
    }
    source = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {},
            "step": 3,
            "tokens_seen": 24,
            "cursor": (0, 3),
            "config": legacy_config,
        },
        source,
    )
    source.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    )
    parser = build_parser()
    arguments = parser.parse_args(["checkpoint", "inspect", str(source), "--json"])
    arguments.handler(arguments)
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["resume_level"] == "weights_only"
    assert (inspected["step"], inspected["tokens_seen"]) == (3, 24)
    arguments = parser.parse_args(["checkpoint", "verify", str(source), "--json"])
    arguments.handler(arguments)
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] and verified["resume_level"] == "weights_only"
    result = import_weights(
        source,
        tmp_path / "legacy-import",
        config,
        source_format="klab_legacy_v1",
        source_tokenizer=tokenizer,
        provenance=_provenance(tmp_path / "provenance.json"),
    )
    promoted = CheckpointManager(result.run_dir).load(result.checkpoint, "promote")
    generation_manifest = json.loads(
        (result.checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    assert generation_manifest["state_codec"] is None
    assert generation_manifest["state_codec_version"] is None
    manager = CheckpointManager(result.run_dir)
    assert manager.verify(result.checkpoint, require_training_state=False).valid
    with pytest.raises(ValueError):
        manager.load(result.checkpoint, "resume")
    assert promoted.step == 0 and torch.equal(
        promoted.model["embedding.weight"], model.state_dict()["embedding.weight"]
    )

    malformed = tmp_path / "malformed.pt"
    state = torch.load(source, weights_only=True)
    state["model"]["embedding.weight"] = state["model"]["embedding.weight"][:1].clone()
    torch.save(state, malformed)
    malformed.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "sha256": hashlib.sha256(malformed.read_bytes()).hexdigest(),
            }
        )
    )
    arguments = parser.parse_args(["checkpoint", "verify", str(malformed), "--json"])
    with pytest.raises(SystemExit) as error:
        arguments.handler(arguments)
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["valid"]
