"""Long-context attention: no quadratic state, and every kind composes with every FFN."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_training import config
from tokenizers import Tokenizer

from klab.config.models import AttentionConfig, ModelConfig
from klab.evaluation.generation import generate_with_token_ids
from klab.evaluation.speculation import PromptLookupSpeculation
from klab.model.attention.dense import DenseAttention
from klab.model.attention.latent import LatentAttention
from klab.model.attention.local_selected import LocalSelectedAttention
from klab.model.moe import TopKMoE
from klab.model.transformer import DenseLM
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
    "local_selected_learned": {
        "kind": "local_selected",
        "block_size": 16,
        "window_size": 32,
        "selected_blocks": 2,
        "latent_dim": 16,
        "selector": "learned",
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
@pytest.mark.parametrize("length", [8, 70])
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
    query, key = query.reshape(1, 4, 20, 8, 8), key.reshape(1, 4, 20, 8, 8)
    indices = attention.visible_blocks(query, attention._summarize(key))[0]
    # One local block, the own block and one selected block per head.
    assert indices.shape == (4, 20, 3)
    assert (indices[:, 0] == torch.tensor([-1, 0, -1])).all()
    assert (indices[:, 1] == torch.tensor([0, 1, -1])).all()
    assert (indices[:, -1, :2] == torch.tensor([18, 19])).all()
    assert ((indices[:, -1, 2] >= 0) & (indices[:, -1, 2] < 18)).all()
    seen = set(indices[:, -1].flatten().tolist())
    hidden = next(block for block in range(18) if block not in seen)
    changed = values.clone()
    changed[:, hidden * 8 : (hidden + 1) * 8] += 5.0
    # Nothing before an edited block may change, whatever later blocks select.
    assert torch.equal(
        attention(changed)[:, : hidden * 8], attention(values)[:, : hidden * 8]
    )


@pytest.mark.parametrize("selector", ["mean_key", "learned"])
@pytest.mark.parametrize("latent", [None, 16])
def test_local_selected_cached_decoding_matches_full_forward_and_survives_rewind(
    selector: str, latent: int | None
) -> None:
    torch.manual_seed(0)
    attention = LocalSelectedAttention(
        32, 4, 512, 10000.0, 8, 16, 2, latent, selector=selector
    ).eval()
    if selector == "learned":
        with torch.no_grad():
            attention.pool_logits.normal_()
            attention.coarse_gate.normal_()
    values = torch.randn(2, 203, 32)
    with torch.no_grad():
        full = attention(values)
        for width in (1, 7, 203):
            cache = attention.create_cache(
                2, 256, device=values.device, dtype=values.dtype
            )
            steps = []
            for start in range(0, 203, width):
                step, cache = attention.forward_cached(
                    values[:, start : start + width], cache
                )
                steps.append(step)
            torch.testing.assert_close(torch.cat(steps, 1), full, atol=5e-6, rtol=1e-5)
        # Dropping speculative positions and decoding them again gives the same.
        for cut in (202, 100, 9):
            cache.length = cache.position = cut
            step, cache = attention.forward_cached(values[:, cut:], cache)
            torch.testing.assert_close(step, full[:, cut:], atol=5e-6, rtol=1e-5)
    # A token reads its window and selected blocks, not the whole prefix.
    assert cache.extra["selected"].shape == (2, 4, 32, 2)
    # A request smaller than one block still decodes.
    with torch.no_grad():
        small = attention.create_cache(2, 5, device=values.device, dtype=values.dtype)
        step, small = attention.forward_cached(values[:, :5], small)
    torch.testing.assert_close(step, full[:, :5], atol=5e-6, rtol=1e-5)


def test_learned_selector_starts_as_mean_key_and_trains_through_its_gate() -> None:
    torch.manual_seed(0)
    fixed = LocalSelectedAttention(32, 4, 512, 10000.0, 8, 8, 2)
    learned = LocalSelectedAttention(32, 4, 512, 10000.0, 8, 8, 2, selector="learned")
    assert learned.load_state_dict(fixed.state_dict(), strict=False).missing_keys == [
        "pool_logits",
        "coarse_gate",
    ]
    values = torch.randn(2, 150, 32)
    torch.testing.assert_close(learned(values), fixed(values), atol=1e-6, rtol=1e-5)

    learned(values).square().sum().backward()
    assert learned.coarse_gate.grad.abs().sum() > 0
    assert learned.pool_logits.grad.abs().sum() == 0
    with torch.no_grad():
        learned.coarse_gate.fill_(0.5)
        learned.pool_logits.normal_()
    learned.zero_grad()
    learned(values).square().sum().backward()
    assert learned.pool_logits.grad.abs().sum() > 0

    # The coarse branch sees complete earlier blocks only, so it stays causal.
    with torch.no_grad():
        before = learned(values)
        changed = values.clone()
        changed[:, 77] += 5.0
        torch.testing.assert_close(learned(changed)[:, :77], before[:, :77])


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


# Each attention kind runs once; the FFN alternates so both FFNs compose with
# fused, latent and block-selected attention. Attention and FFN do not interact
# inside a block, so the full cross product repeats the same paths.
@pytest.mark.parametrize(
    ("attention", "ffn"),
    [
        ("dense_sdpa", "moe"),
        ("sliding_sdpa", "dense"),
        ("mla_sdpa", "moe"),
        ("local_selected", "dense"),
        ("local_selected_latent", "moe"),
        ("local_selected_learned", "dense"),
    ],
)
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


def test_local_selected_trains_checkpoints_and_generates(tmp_path: Path) -> None:
    ffn = "moe"
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
                selector="learned",
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
    assert loaded.model.incremental_cache_capability.backend == "local_selected"
    tokenizer = Tokenizer.from_file(str(run.tokenizer.path))

    def tokens(**options: object) -> list[int]:
        return generate_with_token_ids(
            loaded.model,
            tokenizer,
            "a b a b a",
            run.model.max_seq_len,
            40,
            loaded.device,
            **options,
        )[1]

    # The KV cache, prefix recomputation and draft-and-verify decoding agree,
    # across several context rollovers of this 32-token model.
    cached = tokens()
    assert len(cached) == 40
    assert cached == tokens(use_cache=False)
    assert cached == tokens(speculation=PromptLookupSpeculation(ngram=1))
