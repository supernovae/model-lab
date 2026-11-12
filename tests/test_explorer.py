"""Model explorer data layer on a tiny in-memory model (no training, CPU)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from klab import explorer
from klab.lab_context import LabCancelled, LabContext
from klab.probes import scoring

ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    "<eos>",
    "once",
    "upon",
    "a",
    "time",
    "the",
    "dog",
    "found",
    "ball",
    "in",
    "park",
    "and",
    "ran",
    "to",
    "it",
]


def _config(**model: Any) -> Any:
    from klab.config.models import RunConfig

    base = (
        "configs/smoke_cpu.yaml"
        if model.get("ffn") == "dense"
        else "configs/smoke_moe_cpu.yaml"
    )
    raw = yaml.safe_load((ROOT / base).read_text())
    raw["model"].update(model)
    return RunConfig.model_validate(raw)


def _tokenizer() -> Any:
    from tokenizers import Tokenizer
    from tokenizers.decoders import WordPiece
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {w: i for i, w in enumerate([*WORDS, "<unk>", ".", ","])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPiece()
    return tokenizer


def _loaded(**model: Any) -> Any:
    from klab.model.transformer import DenseLM

    config = _config(**model)
    torch.manual_seed(0)
    lm = DenseLM(config.model, config.attention).eval()
    return SimpleNamespace(
        model=lm,
        config=config,
        tokenizer=_tokenizer(),
        device=torch.device("cpu"),
        identity={
            "run_id": "tiny",
            "checkpoint_sha256": "c" * 64,
            "tokenizer_sha256": "t" * 64,
            "step": 3,
            "runtime": {
                "engine": "pytorch",
                "backend": "cpu",
                "device_index": 0,
                "precision": "fp32",
            },
        },
    )


TEXT = "once upon a time the dog found a ball in the park and ran to it ."
MOE_NGRAM = {
    "num_kv_heads": 1,  # grouped-query attention: 2 query heads share 1 KV head
    "memory": "ngram",
    "memory_table_size": 97,
    "memory_ngram_size": 2,
    "memory_ngram_orders": (2, 3),
    "memory_dim": 8,
}


def test_architecture_comes_from_the_config_alone() -> None:
    arch = explorer.architecture(_config(**MOE_NGRAM))
    assert [layer["index"] for layer in arch["layers"]] == [0, 1]
    first = arch["layers"][0]
    assert first["attention"]["heads"] == 2 and first["attention"]["kv_heads"] == 1
    assert first["ffn"]["kind"] == "moe" and first["ffn"]["experts"] == 3
    assert first["ffn"]["experts_per_token"] == 2 and first["ffn"]["shared_expert"]
    assert arch["memory"]["kind"] == "ngram" and arch["memory"]["tables"] == 2
    # The per-part counts add up to the inventory's total.
    parts = (
        arch["embedding_parameters"]
        + arch["output_parameters"]
        + sum(
            layer["attention"]["parameters"]
            + layer["ffn"]["parameters"]
            + layer["norm_parameters"]
            for layer in arch["layers"]
        )
        + arch["memory"]["parameters"]
        + arch["hidden_dim"]  # final norm
    )
    assert parts == arch["inventory"]["total"]
    assert explorer.size_check(arch) is None


def test_size_guard_refuses_big_models_before_loading(tmp_path: Path) -> None:
    arch = explorer.architecture(
        _config(
            hidden_dim=1024, num_heads=16, num_layers=24, ffn_dim=4096, vocab_size=32000
        )
    )
    message = explorer.size_check(arch)
    assert message and "limit" in message and "klab probe" in message


def test_attention_probabilities_are_the_forward_passes_own_weights() -> None:
    from klab.model.attention.dense import DenseAttention

    for kv_heads in (2, 1):
        torch.manual_seed(1)
        attention = DenseAttention(32, 2, 16, 10_000.0, num_kv_heads=kv_heads)
        x = torch.randn(1, 7, 32)
        weights = attention.attention_probabilities(x)
        assert weights.shape == (1, 2, 7, 7)
        assert torch.allclose(weights.sum(-1), torch.ones(1, 2, 7))
        assert torch.all(weights.triu(1) == 0)  # causal
        # Rebuild the output from these weights: it equals forward(x).
        value = attention._heads(attention.v_proj, x, attention.num_kv_heads)
        grouped = (
            weights.unflatten(1, (kv_heads, 2 // kv_heads)) if kv_heads < 2 else weights
        )
        mixed = attention._attend_values(grouped, value)
        out = attention.out_proj(mixed.transpose(1, 2).reshape(1, 7, 32))
        assert torch.allclose(out, attention(x), atol=1e-6)


def test_explore_loaded_moe_memory_and_tokens() -> None:
    loaded = _loaded(**MOE_NGRAM)
    result = explorer.explore_loaded(loaded, TEXT)
    assert result["runtime"] == loaded.identity["runtime"]
    ids = [scoring.eot(loaded), *scoring.encode(loaded, TEXT)]
    tokens = result["tokens"]
    assert [t["id"] for t in tokens] == ids and tokens[0]["loss"] is None
    rows = scoring.log_probs(loaded, ids)
    for t in tokens[1:]:
        assert t["loss"] == pytest.approx(-rows[t["position"] - 1, t["id"]])
        assert t["rank"] >= 1 and len(t["next_top"]) == explorer.TOP_K
    assert tokens[2]["text"] == "upon"
    # Attention: every dense layer, heads x T x T, rows sum to 1.
    layers = result["attention"]["layers"]
    assert set(layers) == {"0", "1"} and result["attention"]["note"] is None
    weights = np.asarray(layers["0"])
    assert weights.shape == (2, len(ids), len(ids))
    assert np.allclose(weights.sum(-1), 1, atol=1e-3)
    # Routing: each token picks experts_per_token experts per MoE layer.
    routing = result["routing"]["0"]
    assert len(routing["selected"]) == len(ids)
    assert all(len(choice) == 2 for choice in routing["selected"])
    assert sum(routing["counts"]) == 2 * len(ids)
    # Memory: one address stream per (order, hash head) table, plus the gate.
    memory = result["memory"]
    assert memory["kind"] == "ngram" and len(memory["streams"]) == 2
    for stream in memory["streams"]:
        assert len(stream["addresses"]) == len(ids)
        assert all(0 <= a < 97 for a in stream["addresses"])
        for position, first in enumerate(stream["first_seen_at"]):
            if first is not None:
                assert first < position
                assert stream["addresses"][first] == stream["addresses"][position]
    assert len(memory["gate"]) == len(ids) and all(0 < g < 1 for g in memory["gate"])
    # Weights: every stored tensor with a histogram over its (sampled) values.
    names = [w["name"] for w in result["weights"]]
    assert names == [n for n, _ in loaded.model.named_parameters()]
    for w in result["weights"]:
        assert sum(w["histogram"]["counts"]) == len(
            range(0, w["numel"], w["histogram"]["sampled_every"])
        )


def test_explorer_notes_attention_it_cannot_show() -> None:
    result = explorer.explore_loaded(_loaded(), TEXT)
    assert result["memory"] is None
    mla = _config()
    arch = explorer.architecture(mla)
    arch["layers"][1]["attention"]["kind"] = "mla"
    view = explorer.attention_view(arch, {"0": np.ones((2, 3, 3))})
    assert "mla" in view["note"] and "[1]" in view["note"]


def test_explorer_stops_at_a_cancel_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / "CANCEL"
    sentinel.touch()
    with pytest.raises(LabCancelled):
        explorer.explore_loaded(_loaded(), TEXT, context=LabContext(sentinel))


def test_cache_is_sealed_and_rejects_edits(tmp_path: Path) -> None:
    from klab.lab_records import seal, write_json_atomic

    path = explorer.cache_path(tmp_path, "c" * 64, TEXT)
    assert path != explorer.cache_path(tmp_path, "c" * 64, TEXT + "!")
    path.parent.mkdir(parents=True)
    value = seal({"format": explorer.EXPLORER_FORMAT, "created_at": "x", "text": TEXT})
    write_json_atomic(path, value)
    assert explorer.read_cached(path) == value
    assert [v["text"] for v in explorer.cached_explorations(tmp_path)] == [TEXT]
    write_json_atomic(path, {**value, "text": "edited"})
    assert explorer.read_cached(path) is None
    assert explorer.cached_explorations(tmp_path) == []


def _legacy_collapsed_addresses(
    self: Any, input_ids: torch.Tensor, order: int | None = None, seed: int = 0
) -> torch.Tensor:
    """The pre-fix head-0 hash at 257 rows: the oldest token mod table size."""
    order = self.ngram_size if order is None else order
    shifted = torch.zeros_like(input_ids)
    offset = order - 1
    shifted[:, offset:] = input_ids[:, : input_ids.shape[1] - offset]
    return shifted % self.table_size


def test_memory_view_flags_an_ngram_table_that_reads_one_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detector flags a table whose addresses are one earlier token's id
    (what the legacy 257 multiplier produced at 257 rows)."""
    from klab.model.memory import TokenNgramMemory

    config = {
        "memory": "ngram",
        "memory_table_size": 257,
        "memory_ngram_size": 2,
        "memory_ngram_orders": (2, 3),
        "memory_dim": 8,
    }
    with monkeypatch.context() as patch:
        patch.setattr(TokenNgramMemory, "addresses", _legacy_collapsed_addresses)
        collapsed = explorer.explore_loaded(_loaded(**config), TEXT)
    offsets = [s["single_token_offset"] for s in collapsed["memory"]["streams"]]
    assert sorted(offsets) == [1, 2]
    healthy = explorer.explore_loaded(_loaded(**MOE_NGRAM), TEXT)
    assert all(s["single_token_offset"] is None for s in healthy["memory"]["streams"])


def test_memory_view_no_longer_flags_a_257_row_table() -> None:
    """Regression for the 257 hash fix: the same 257-row table now hashes
    real n-grams, so no stream reads a single earlier token."""
    fixed = explorer.explore_loaded(
        _loaded(memory="ngram", memory_table_size=257, memory_ngram_size=2,
                memory_ngram_orders=(2, 3), memory_dim=8),
        TEXT,
    )  # fmt: skip
    assert all(s["single_token_offset"] is None for s in fixed["memory"]["streams"])


def _fake_run(tmp_path: Path) -> Path:
    lab = tmp_path / "lab"
    run = lab / "runs" / "tiny-run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    (run / "resolved_config.yaml").write_text(_config(**MOE_NGRAM).model_dump_json())
    return lab


def _no_cache(lab: Path) -> None:
    assert explorer.cached_explorations(lab) == []
    assert not list((lab / "explorer").rglob("*.json"))


@pytest.mark.parametrize(
    "error",
    [MemoryError(), torch.OutOfMemoryError("CUDA out of memory")],
    ids=["MemoryError", "torch.OutOfMemoryError"],
)
def test_loader_oom_is_a_clean_outcome_with_cleanup_and_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    """Loading sits inside the protected lifecycle: an OOM while loading is
    a clear 'out of memory' outcome, cleanup runs, nothing is cached."""
    lab = _fake_run(tmp_path)
    released: list[bool] = []
    monkeypatch.setattr(explorer, "release_memory", lambda: released.append(True))

    def load() -> Any:
        raise error

    with pytest.raises(explorer.ExplorerUnavailable, match=r"out of memory .*load"):
        explorer.explore("tiny-run", lab_dir=lab, text=TEXT, load=load)
    assert released == [True]
    _no_cache(lab)
    # Any other loader failure propagates, and cleanup still runs.
    with pytest.raises(ValueError, match="bad checkpoint"):
        explorer.explore(
            "tiny-run",
            lab_dir=lab,
            text=TEXT,
            load=lambda: (_ for _ in ()).throw(ValueError("bad checkpoint")),
        )
    assert released == [True, True]
    _no_cache(lab)


def test_pre_load_stage_check_runs_before_the_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.lab_context import LabResourceExceeded
    from klab.resource_envelope import ResourceEnvelope

    lab = _fake_run(tmp_path)
    calls: list[str] = []
    released: list[bool] = []
    monkeypatch.setattr(explorer, "release_memory", lambda: released.append(True))

    def load() -> Any:
        calls.append("load")
        return _loaded(**MOE_NGRAM)

    # Envelope violated: stops at the pre-load check, the loader never runs.
    envelope = ResourceEnvelope(resource_envelope_version=1, min_disk_bytes=1 << 62)
    with pytest.raises(LabResourceExceeded) as stopped:
        explorer.explore(
            "tiny-run", lab_dir=lab, text=TEXT, resource_envelope=envelope, load=load
        )
    assert stopped.value.phase == "load" and calls == [] and released == [True]
    _no_cache(lab)

    # Cancelled (sentinel touched once the exploration started): same.
    class Cancelled(LabContext):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.cancel_path.touch()

    monkeypatch.setattr(explorer, "LabContext", Cancelled)
    with pytest.raises(LabCancelled) as cancelled:
        explorer.explore("tiny-run", lab_dir=lab, text=TEXT, load=load)
    assert cancelled.value.phase == "load" and calls == [] and released == [True] * 2
    _no_cache(lab)


def test_activation_diagnostics_match_raw_residual_stream_and_leave_no_hooks() -> None:
    for model_overrides in (MOE_NGRAM, {"ffn": "dense"}):
        loaded = _loaded(**model_overrides)
        result = explorer.explore_loaded(loaded, TEXT)
        assert result["target"]["tokenizer_sha256"] == "t" * 64
        act = result["activation"]
        assert act["metric"] == "rms"
        assert act["sites"] == ["residual_pre", "residual_mid", "residual_post"]
        ids, truncated = explorer.tokenize(loaded, TEXT)
        assert not truncated

        # Independently step through the native model modules to compute exact RMS.
        model = loaded.model
        for mod in model.modules():
            assert not mod._forward_pre_hooks and not mod._forward_hooks

        x = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            h = model.embedding(x)
            if model.config.memory_injection == "embedding":
                h = model._apply_memory(h, x, None)
            for idx, block in enumerate(model.blocks):
                pre_rms = h.float().square().mean(dim=-1).sqrt()[0].tolist()
                attn_out = block.attention(block.norm1(h))
                h_mid = h + attn_out
                mid_rms = h_mid.float().square().mean(dim=-1).sqrt()[0].tolist()
                if model_overrides.get("ffn") == "dense":
                    ffn_out = block.ffn(block.norm2(h_mid))
                else:
                    ffn_out, _ = block.ffn.forward_with_aux(
                        block.norm2(h_mid), diagnostics="full"
                    )
                h = h_mid + ffn_out
                post_rms = h.float().square().mean(dim=-1).sqrt()[0].tolist()
                layer_act = act["layers"][str(idx)]
                assert layer_act["residual_pre"] == pytest.approx(pre_rms, rel=1e-6)
                assert layer_act["residual_mid"] == pytest.approx(mid_rms, rel=1e-6)
                assert layer_act["residual_post"] == pytest.approx(post_rms, rel=1e-6)
                if idx + 1 < len(model.blocks):
                    next_pre = act["layers"][str(idx + 1)]["residual_pre"]
                    assert layer_act["residual_post"] == pytest.approx(
                        next_pre, rel=1e-6
                    )
                if model_overrides is MOE_NGRAM:
                    diag = block.ffn.last_diagnostics
                    recorded = result["routing"][str(idx)]
                    assert (
                        recorded["selected"]
                        == diag.selected_experts.cpu().numpy().tolist()
                    )
                    assert np.allclose(
                        recorded["weights"],
                        diag.selected_weights.float().cpu().numpy(),
                    )
            # Final block residual_post is the input to model.norm.
            final_norm_in_rms = h.float().square().mean(dim=-1).sqrt()[0].tolist()
            assert act["layers"][str(len(model.blocks) - 1)][
                "residual_post"
            ] == pytest.approx(final_norm_in_rms, rel=1e-6)

    # Short and truncated inputs keep exact token-aligned activation lengths.
    short_res = explorer.explore_loaded(_loaded(**MOE_NGRAM), "once")
    assert not short_res["truncated"]
    assert len(short_res["tokens"]) == 2
    assert len(short_res["activation"]["layers"]["0"]["residual_pre"]) == 2

    long_text = " ".join(["once upon a time"] * 40)
    trunc_res = explorer.explore_loaded(_loaded(**MOE_NGRAM), long_text)
    assert trunc_res["truncated"] is True
    limit = len(trunc_res["tokens"])
    for site in explorer.ACTIVATION_SITES:
        assert len(trunc_res["activation"]["layers"]["0"][site]) == limit


def test_compare_explorations_deltas_overlap_and_refusals() -> None:
    from klab.lab_records import jsonable, seal

    left_loaded = _loaded(**MOE_NGRAM)
    right_loaded = _loaded(**MOE_NGRAM)
    right_loaded.identity["checkpoint_sha256"] = "d" * 64
    with torch.no_grad():
        for param in right_loaded.model.parameters():
            param.mul_(1.25)

    left = seal(jsonable(explorer.explore_loaded(left_loaded, TEXT)))
    right = seal(jsonable(explorer.explore_loaded(right_loaded, TEXT)))
    comp = explorer.compare_explorations(left, right)
    assert comp["common_layers"] == [0, 1]
    assert comp["only_left"] == [] and comp["only_right"] == []
    assert comp["architecture_matches"] is True
    assert comp["delta"] is not None
    for layer_key in ("0", "1"):
        for site in explorer.ACTIVATION_SITES:
            expected = [
                r - l
                for l, r in zip(
                    left["activation"]["layers"][layer_key][site],
                    right["activation"]["layers"][layer_key][site],
                    strict=True,
                )
            ]
            assert comp["delta"][layer_key][site] == pytest.approx(expected, rel=1e-6)
            assert any(abs(v) > 1e-4 for v in comp["delta"][layer_key][site])

    # Different architecture (3 layers vs 2 layers): overlap layers listed, delta is None.
    deeper_loaded = _loaded(**MOE_NGRAM, num_layers=3)
    deeper_loaded.identity["checkpoint_sha256"] = "e" * 64
    deeper = seal(jsonable(explorer.explore_loaded(deeper_loaded, TEXT)))
    diff_comp = explorer.compare_explorations(left, deeper)
    assert diff_comp["common_layers"] == [0, 1]
    assert diff_comp["only_left"] == []
    assert diff_comp["only_right"] == [2]
    assert diff_comp["architecture_matches"] is False
    assert diff_comp["delta"] is None

    # Explicit refusals for invalid/incompatible pairs.
    with pytest.raises(explorer.ExplorerUnavailable, match="same checkpoint_sha256"):
        explorer.compare_explorations(left, left)

    diff_tok = seal(
        {**right, "target": {**right["target"], "tokenizer_sha256": "z" * 64}}
    )
    with pytest.raises(explorer.ExplorerUnavailable, match="tokenizer_sha256"):
        explorer.compare_explorations(left, diff_tok)

    diff_text = seal({**right, "text": "once upon"})
    with pytest.raises(explorer.ExplorerUnavailable, match="same text"):
        explorer.compare_explorations(left, diff_text)

    diff_ids = seal(
        {
            **right,
            "tokens": [{**right["tokens"][0], "id": 999}, *right["tokens"][1:]],
        }
    )
    with pytest.raises(explorer.ExplorerUnavailable, match="token-ID sequences"):
        explorer.compare_explorations(left, diff_ids)

    truncated = seal({**right, "truncated": True})
    with pytest.raises(explorer.ExplorerUnavailable, match="truncated"):
        explorer.compare_explorations(left, truncated)

    diff_runtime = seal({**right, "runtime": {**right["runtime"], "backend": "rocm"}})
    with pytest.raises(explorer.ExplorerUnavailable, match="different runtimes"):
        explorer.compare_explorations(left, diff_runtime)

    other_format = seal({**right, "format": "klab-explorer-other"})
    with pytest.raises(explorer.ExplorerUnavailable, match="klab-explorer-other"):
        explorer.compare_explorations(left, other_format)

    no_act = seal({k: v for k, v in right.items() if k != "activation"})
    with pytest.raises(explorer.ExplorerUnavailable, match="activation"):
        explorer.compare_explorations(left, no_act)

    unavail = seal({**right, "unavailable": "too big"})
    with pytest.raises(explorer.ExplorerUnavailable, match="unavailable"):
        explorer.compare_explorations(left, unavail)

    with pytest.raises(explorer.ExplorerUnavailable, match="unsealed"):
        explorer.compare_explorations(left, {**right, "text": "tampered"})

    with pytest.raises(explorer.ExplorerUnavailable, match="unverified cache"):
        explorer.compare_explorations(left, {**right, "cache_issue": "bad cache"})


def test_prime_stream_sizes_are_visible_and_not_given_legacy_collapse_diagnosis() -> (
    None
):
    from klab.dashboard import explorer_view

    loaded = _loaded(
        **{
            **MOE_NGRAM,
            "memory_table_size": 97,
            "memory_addressing": "prime_xor_v1",
            "memory_table_sizes": (97, 101),
        }
    )
    architecture = explorer.architecture(loaded.config)
    assert architecture["memory"]["rows_per_stream"] == [97, 101]
    assert architecture["memory"]["rows_scope"] == "first_table"
    assert "rows [97, 101]" in explorer_view.architecture_html(architecture)
    ids = [3, 7, 11, 13, 17]
    # This coincidental short pattern would trigger the legacy-only diagnostic.
    view = explorer.memory_view(loaded, ids, {"memory": {"table": torch.tensor([ids])}})
    assert view["streams"][0]["single_token_offset"] is None


def test_multiport_explorer_captures_every_table_and_gate() -> None:
    from klab.dashboard import explorer_view

    loaded = _loaded(
        **{
            **MOE_NGRAM,
            "num_layers": 3,
            "memory_injection": "after_block",
            "memory_block_indices": [0, 2],
            "memory_variant": "context_key",
            "memory_addressing": "prime_xor_v1",
            "memory_table_sizes": [97, 101, 103, 107],
            "memory_table_size": 97,
        }
    )
    ids = [3, 7, 11]
    forward = explorer._forward_with_hooks(loaded, ids)
    view = explorer.memory_view(loaded, ids, forward)
    assert len(view["streams"]) == 4
    assert set(view["gates_by_port"]) == {"memory", "extra_memories.2"}
    assert view["gate"] == pytest.approx(view["gates_by_port"]["memory"])
    for gates in view["gates_by_port"].values():
        assert len(gates) == len(ids)
        assert all(0 <= gate <= 1 for gate in gates)
    architecture = explorer.architecture(loaded.config)["memory"]
    assert architecture["tables"] == 4
    assert architecture["parameters"] == sum(
        parameter.numel()
        for port in [loaded.model.memory, *loaded.model.extra_memories.values()]
        for parameter in port.parameters()
    )
    assert (
        len(explorer_view.gate_figure(view, explorer.token_view(loaded, ids)).data) == 2
    )
