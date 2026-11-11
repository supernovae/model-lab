"""Long-context attention: no quadratic state, and every kind composes with every FFN."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_training import config
from tokenizers import Tokenizer

from klab.config.models import AttentionConfig, ModelConfig
from klab.evaluation.generation import generate_with_token_ids
from klab.model.attention.dense import DenseAttention
from klab.model.attention.latent import LatentAttention
from klab.model.attention.local_selected import LocalSelectedAttention
from klab.model.moe import TopKMoE
from klab.model.transformer import ATTENTION_BUILDERS, DenseLM
from klab.training.trainer import train

ATTENTIONS = {
    "dense_sdpa": {"kind": "dense", "implementation": "sdpa"},
    "sliding_sdpa": {
        "kind": "sliding_window",
        "window_size": 64,
        "implementation": "sdpa",
    },
    "mla_sdpa": {"kind": "mla", "latent_dim": 16, "implementation": "sdpa"},
    "local_selected": {
        "kind": "local_selected",
        "block_size": 16,
        "window_size": 32,
        "selected_blocks": 2,
    },
    "local_selected_latent": {
        "kind": "local_selected",
        "block_size": 16,
        "window_size": 32,
        "selected_blocks": 2,
        "latent_dim": 16,
    },
}
FFNS = {
    "dense": {},
    "moe": {
        "ffn": "moe",
        "num_experts": 4,
        "experts_per_token": 2,
        "shared_expert": True,
        "router_aux_loss_coefficient": 0.01,
    },
}


@pytest.mark.parametrize("latent", [None, 16])
@pytest.mark.parametrize("length", [1, 8, 9, 70])
def test_local_selected_equals_full_attention_when_every_block_is_visible(
    latent: int | None, length: int
) -> None:
    torch.manual_seed(0)
    # Twenty selected blocks cover any earlier block of a 70-token sequence.
    sparse = LocalSelectedAttention(32, 4, 128, 10000.0, 8, 8, 20, latent)
    full = (
        DenseAttention(32, 4, 128, 10000.0)
        if latent is None
        else LatentAttention(32, 4, latent, 128, 10000.0)
    )
    full.load_state_dict(sparse.state_dict())
    values = torch.randn(2, length, 32)
    torch.testing.assert_close(sparse(values), full(values), atol=3e-6, rtol=1e-5)


def test_local_selected_limits_what_distant_queries_see() -> None:
    torch.manual_seed(0)
    attention = LocalSelectedAttention(32, 4, 512, 10000.0, 8, 8, 1)
    values = torch.randn(1, 160, 32)
    query, key, _ = attention._project(values)
    indices, real = attention.visible_blocks(
        query.reshape(4, 20, 8, 8), key.reshape(4, 20, 8, 8)
    )
    # Own block, one local block and one selected block per head.
    assert indices.shape == (4, 20, 3)
    assert real[:, -1].all() and not real[:, 0, 0].any() and not real[:, 1, 2].any()
    assert (indices[:, -1, :2] == torch.tensor([18, 19])).all()
    assert (indices[:, -1, 2] < 18).all()
    seen = set(indices[:, -1].flatten().tolist())
    hidden = next(block for block in range(18) if block not in seen)
    changed = values.clone()
    changed[:, hidden * 8 : (hidden + 1) * 8] += 5.0
    # The last block still selects from the same summaries unless the edit moves
    # them; either way nothing before the edited block may change.
    assert torch.equal(
        attention(changed)[:, : hidden * 8], attention(values)[:, : hidden * 8]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"block_size": 8, "selected_blocks": 1}, "requires block_size, window_size"),
        (
            {"block_size": 8, "window_size": 12, "selected_blocks": 1},
            "multiple of block_size",
        ),
    ],
)
def test_local_selected_config_refuses_incomplete_settings(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        AttentionConfig(kind="local_selected", **kwargs)


def test_windowed_and_latent_sdpa_match_their_reference_paths() -> None:
    torch.manual_seed(0)
    values = torch.randn(2, 150, 32)
    for build in (
        lambda implementation: DenseAttention(
            32, 4, 4096, 10000.0, 16, implementation=implementation
        ),
        lambda implementation: LatentAttention(
            32, 4, 16, 4096, 10000.0, implementation=implementation
        ),
    ):
        reference, fused = build("reference"), build("sdpa")
        fused.load_state_dict(reference.state_dict())
        torch.testing.assert_close(
            fused(values), reference(values), atol=3e-6, rtol=1e-5
        )
        cache = fused.create_cache(2, 256, device=values.device, dtype=values.dtype)
        steps = []
        with torch.no_grad():
            for start in range(0, 150, 7):
                step, cache = fused.forward_cached(values[:, start : start + 7], cache)
                steps.append(step)
        torch.testing.assert_close(
            torch.cat(steps, 1), reference(values), atol=3e-6, rtol=1e-5
        )


@pytest.mark.parametrize("ffn", FFNS)
@pytest.mark.parametrize("attention", ATTENTIONS)
def test_every_attention_composes_with_every_ffn_at_long_context(
    attention: str, ffn: str
) -> None:
    torch.manual_seed(0)
    context = 32768
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=context,
            **FFNS[ffn],
        ),
        AttentionConfig(**ATTENTIONS[attention]),
    )
    assert isinstance(model.blocks[0].ffn, TopKMoE) == (ffn == "moe")
    # The configured context costs nothing until a sequence that long runs.
    assert max(buffer.numel() for buffer in model.buffers()) < context

    ids = torch.randint(0, 260, (2, 1024))
    logits, auxiliary = model.forward_with_aux(ids)
    loss = logits.float().logsumexp(-1).mean() + auxiliary
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if ".experts." not in name
    )

    # Causal at the model level: a later token cannot change earlier logits.
    model.eval()
    with torch.no_grad():
        changed = ids.clone()
        changed[:, 700] = (changed[:, 700] + 1) % 260
        before, after = model(ids), model(changed)
    torch.testing.assert_close(after[:, :700], before[:, :700], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(after[:, 700:], before[:, 700:])


def test_attention_kinds_are_built_from_one_table() -> None:
    kinds = AttentionConfig.model_fields["kind"].annotation.__args__
    assert set(ATTENTION_BUILDERS) == set(kinds)


@pytest.mark.parametrize("ffn", FFNS)
def test_local_selected_trains_checkpoints_and_generates(
    tmp_path: Path, ffn: str
) -> None:
    base = config(tmp_path)
    run = base.model_copy(
        update={
            "model": base.model.model_copy(update=FFNS[ffn]),
            "attention": AttentionConfig(
                kind="local_selected",
                block_size=4,
                window_size=4,
                selected_blocks=1,
                latent_dim=8,
            ),
            "training": base.training.model_copy(
                update={"max_steps": 3, "max_tokens": 96}
            ),
        }
    )
    run_id = train(run, run_id=f"sparse-{ffn}")

    from klab.evaluation.inference import load_run

    loaded = load_run(run_id, run.logging.root_dir, None, "cpu")
    assert isinstance(loaded.model.blocks[0].attention, LocalSelectedAttention)
    assert not loaded.model.incremental_cache_capability.supported
    tokenizer = Tokenizer.from_file(str(run.tokenizer.path))
    _, tokens = generate_with_token_ids(
        loaded.model, tokenizer, "a", run.model.max_seq_len, 6, loaded.device
    )
    assert 0 < len(tokens) <= 6
