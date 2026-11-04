from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lab_fixtures import write_smoke_inputs

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig
from sparselab.memory_tokenizer import build_memory_token_map
from sparselab.model.inspection import named_tensor_inventory
from sparselab.model.transformer import DenseLM
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest, sha256_file
from sparselab.training.memory_warmstart import migrate_dense_to_ngram
from sparselab.training.trainer import train


def _configs(
    root: Path, *, tokenizer: bool = False, multiport: bool = False
) -> tuple[RunConfig, RunConfig]:
    baseline = write_smoke_inputs(root, max_steps=2, train_tokenizer_now=tokenizer)
    raw = load_config(baseline).model_dump(mode="json")
    raw["model"].update(
        hidden_dim=16,
        num_layers=2 if multiport else 1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
    )
    raw["training"].update(
        seq_len=8,
        micro_batch_size=1,
        gradient_accumulation=1,
        max_steps=2,
        max_tokens=16,
    )
    raw["dataset"].update(train_max_tokens=65, validation_max_tokens=33)
    raw["optimizer"].update(warmup_steps=0)
    raw["evaluation"].update(every_steps=1, max_batches=1)
    raw["checkpoint"].update(every_steps=1)
    raw["logging"]["root_dir"] = str(root / "runs")
    source = RunConfig.model_validate(raw)
    raw["model"].update(
        memory="ngram",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=4,
        memory_ngram_orders=[2, 3],
        memory_hash_heads=2,
        memory_variant="context_key",
        memory_injection="after_block",
        memory_block_index=0,
        memory_conv_kernel_size=4,
        memory_addressing="prime_xor_v1",
        memory_table_sizes=[31, 37, 41, 43],
        memory_hash_seed=13,
        memory_independent_init=True,
    )
    if multiport:
        raw["model"].update(
            memory_block_index=None,
            memory_block_indices=[0, 1],
            memory_table_sizes=[31, 37, 41, 43, 47, 53, 59, 61],
        )
    if tokenizer:
        raw["model"]["memory_token_mapping"] = build_memory_token_map(
            source.tokenizer.path, "deepseek_v1"
        ).to_dict()
    return source, RunConfig.model_validate(raw)


def _migrate(source: RunConfig, target: RunConfig, model: DenseLM):
    return migrate_dense_to_ngram(
        source,
        target,
        model.state_dict(),
        source_tokenizer_sha256="a" * 64,
        target_tokenizer_sha256="a" * 64,
        source_checkpoint_sha256="b" * 64,
    )


@pytest.mark.parametrize(
    "addressing,gate_transform,multiport",
    [
        ("legacy", "identity", False),
        ("prime_xor_v1", "identity", False),
        ("prime_xor_v1", "identity", True),
        ("prime_xor_v1", "signed_sqrt", False),
    ],
)
def test_zero_impact_warmstart_preserves_trunk_rng_and_learns(
    tmp_path: Path, addressing: str, gate_transform: str, multiport: bool
) -> None:
    source, target = _configs(tmp_path, multiport=multiport)
    changes = {"memory_addressing": addressing, "memory_gate_transform": gate_transform}
    if addressing == "legacy":
        changes.update(memory_table_sizes=(), memory_hash_seed=0)
    target = target.model_copy(
        update={"model": target.model.model_copy(update=changes)}
    )
    dense = DenseLM(source.model, source.attention)
    rng = torch.get_rng_state().clone()
    weights, decision = _migrate(source, target, dense)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in dense.state_dict().items():
        assert torch.equal(value, weights[name])
    memory = DenseLM(target.model, target.attention)
    memory.load_state_dict(weights, strict=True)
    tokens = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    memory.eval()
    with torch.no_grad():
        assert torch.equal(dense(tokens), memory(tokens))
        prefix, cache = memory.forward_cached(tokens[:, :4], cache_capacity=8)
        appended, cache = memory.forward_cached(tokens[:, 4:], cache=cache)
        torch.testing.assert_close(torch.cat((prefix, appended), dim=1), memory(tokens))
    memory.train()
    optimizer = torch.optim.AdamW(memory.parameters(), lr=1e-3)
    assert not optimizer.state
    loss = torch.nn.functional.cross_entropy(
        memory(tokens).flatten(0, 1), tokens.flatten()
    )
    loss.backward()
    ports = [memory.memory, *getattr(memory, "extra_memories", {}).values()]
    assert all(port.output.weight.grad.abs().sum() > 0 for port in ports)
    assert all(port.table.weight.grad.abs().sum() == 0 for port in ports)
    optimizer.step()
    assert all(port.output.weight.abs().sum() > 0 for port in ports)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(
        memory(tokens).flatten(0, 1), tokens.flatten()
    ).backward()
    assert all(port.table.weight.grad.abs().sum() > 0 for port in ports)
    assert decision["optimizer_policy"] == "reset"
    assert decision["zeroed_tensors"] == (
        ["memory.output.weight", "extra_memories.1.output.weight"]
        if multiport
        else ["memory.output.weight"]
    )


@pytest.mark.parametrize(
    "change", ["backbone", "attention", "tokenizer", "non_dense", "weights"]
)
def test_memory_warmstart_refuses_incompatible_sources(
    tmp_path: Path, change: str
) -> None:
    source, target = _configs(tmp_path)
    dense = DenseLM(source.model, source.attention)
    weights = dense.state_dict()
    target_sha = "a" * 64
    if change == "backbone":
        target = target.model_copy(
            update={"model": target.model.model_copy(update={"ffn_dim": 64})}
        )
    elif change == "attention":
        target = target.model_copy(
            update={
                "attention": target.attention.model_copy(update={"rope_base": 20000.0})
            }
        )
    elif change == "tokenizer":
        target_sha = "c" * 64
    elif change == "non_dense":
        source = target
    else:
        weights.pop("norm.weight")
    with pytest.raises(ValueError):
        migrate_dense_to_ngram(
            source,
            target,
            weights,
            source_tokenizer_sha256="a" * 64,
            target_tokenizer_sha256=target_sha,
            source_checkpoint_sha256="b" * 64,
        )


@pytest.mark.parametrize("memory_only", [False, True])
def test_native_promotion_resets_state_then_resumes_exactly(
    tmp_path: Path, memory_only: bool
) -> None:
    source, target = _configs(tmp_path, tokenizer=True, multiport=True)
    if memory_only:
        target = target.model_copy(
            update={
                "training": target.training.model_copy(
                    update={
                        "trainable_parameters": tuple(
                            name
                            for name in named_tensor_inventory(
                                target.model, target.attention
                            )
                            if name.startswith(("memory.", "extra_memories."))
                        )
                    }
                )
            }
        )
    train(source, run_id="dense-source")
    source_dir = source.logging.root_dir / "dense-source"
    manager = CheckpointManager(source_dir)
    source_snapshot = manager.load(source_dir / "checkpoints/latest.json")
    source_checkpoint = (
        source_dir / "checkpoints" / manager.latest_valid().record.relative_path
    )
    if not memory_only:
        with pytest.raises(ValueError, match="architecture semantics"):
            train(target, promote=source_checkpoint, run_id="refused-implicit")
        with pytest.raises(ValueError, match="immutable checkpoint"):
            train(
                target,
                promote=source_dir / "checkpoints/latest.json",
                add_ngram_memory=True,
                run_id="refused-pointer",
            )
        assert not (target.logging.root_dir / "refused-implicit").exists()
        assert not (target.logging.root_dir / "refused-pointer").exists()
    source_files = {
        str(path.relative_to(source_dir)): sha256_file(path)
        for path in source_dir.rglob("*")
        if path.is_file() and path.name != ".writer.lock"
    }
    train(
        target,
        promote=source_checkpoint,
        add_ngram_memory=True,
        run_id="memory-part",
        stop_after_step=1,
    )
    part = target.logging.root_dir / "memory-part"
    partial = CheckpointManager(part).load(part / "checkpoints/latest.json")
    zero_checkpoint = next((part / "checkpoints").glob("step_00000000_gen_*"))
    initial = CheckpointManager(part).load(zero_checkpoint, "promote")
    dense_model = DenseLM(source.model, source.attention).eval()
    dense_model.load_state_dict(source_snapshot.model)
    memory_model = DenseLM(target.model, target.attention).eval()
    memory_model.load_state_dict(initial.model)
    with torch.no_grad():
        tokens = torch.tensor([[4, 5, 6, 7, 8]])
        assert torch.equal(dense_model(tokens), memory_model(tokens))
    assert partial.model["memory.output.weight"].abs().sum() > 0
    assert partial.model["extra_memories.1.output.weight"].abs().sum() > 0
    assert partial.step == 1 and partial.tokens_seen == 8
    assert all(
        float(state["step"]) == 1 for state in partial.optimizer["state"].values()
    )
    if memory_only:
        for name, value in source_snapshot.model.items():
            assert torch.equal(partial.model[name], value)
    assert source_snapshot.step == 2 and source_snapshot.tokens_seen == 16
    manifest = read_manifest(part / "manifest.json")
    migration = next(
        item
        for item in manifest["resource_decisions"]
        if item["kind"] == "dense_to_ngram_warmstart"
    )
    assert migration["optimizer_policy"] == "reset"
    assert migration["parent_checkpoint_sha256"] == source_snapshot.checkpoint_sha256
    assert manifest["continuation_kind"] == "PROMOTED"
    for name, digest in source_files.items():
        assert sha256_file(source_dir / name) == digest
    train(target, resume=part / "checkpoints/latest.json", run_id="memory-resumed")
    train(
        target, promote=source_checkpoint, add_ngram_memory=True, run_id="memory-full"
    )
    snapshots = [
        CheckpointManager(target.logging.root_dir / name).load(
            target.logging.root_dir / name / "checkpoints/latest.json"
        )
        for name in ("memory-resumed", "memory-full")
    ]
    for key in snapshots[0].model:
        assert torch.equal(snapshots[0].model[key], snapshots[1].model[key])
    torch.testing.assert_close(
        snapshots[0].optimizer, snapshots[1].optimizer, rtol=0, atol=0
    )
    assert snapshots[0].step == snapshots[1].step == 2
    assert snapshots[0].tokens_seen == snapshots[1].tokens_seen == 16
