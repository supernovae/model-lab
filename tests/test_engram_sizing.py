"""Engram sizing measures the real hash: exact counts, closed forms, CLI."""

import json
import math
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from klab.address_hash import (
    engram_multipliers,
    engram_ngram_address,
    engram_table_sizes,
)
from klab.canonical import canonical_digest
from klab.data.engram_sizing import (
    EngramSetup,
    analyze,
    count_orders,
    evaluate_table,
    sample_blocks,
)


def _setup(orders, *, vocab=64, heads=2, seq_len=None, sites=(1,), **extra):
    return EngramSetup(
        orders=orders,
        heads=heads,
        seed=7,
        vocab_size=vocab,
        padding_token_id=3,
        sites=sites,
        memory_dim=8,
        seq_len=seq_len,
        **extra,
    )


def test_counts_and_collisions_match_brute_force_with_windows_and_blocks():
    rng = np.random.default_rng(0)
    tokens = (rng.zipf(1.3, 3000) % 64).astype(np.int64)
    setup = _setup((2, 3), seq_len=50)
    # Two blocks of whole windows: each restarts with the padding token.
    blocks = [tokens[:1500], tokens[1500:]]
    primes = engram_table_sizes(53, setup.heads)
    multipliers = engram_multipliers(7, 3, 64, layer_id=1)
    for stream in count_orders(blocks, setup, 1):
        keys = Counter()
        for block in blocks:
            for start in range(0, len(block), setup.seq_len):
                window = [int(t) for t in block[start : start + setup.seq_len]]
                for i in range(len(window)):
                    keys[
                        tuple(
                            window[i - k] if i - k >= 0 else 3
                            for k in range(stream.order)
                        )
                    ] += 1
        assert stream.distinct == len(keys)
        assert stream.tokens == len(tokens)
        result = evaluate_table(stream, primes)
        for head, prime in zip(result["heads"], primes, strict=True):
            rows = {key: engram_ngram_address(key, prime, multipliers) for key in keys}
            load, members = Counter(), Counter()
            for key, n in keys.items():
                load[rows[key]] += n
                members[rows[key]] += 1
            shared = sum(n for k, n in keys.items() if members[rows[k]] > 1)
            clobbered = sum(n for k, n in keys.items() if n / load[rows[k]] < 0.5)
            assert head["shared_occurrences"] == pytest.approx(shared / stream.tokens)
            assert head["clobbered_occurrences"] == pytest.approx(
                clobbered / stream.tokens
            )
            assert head["occupancy"] == pytest.approx(len(load) / prime)


def test_random_ngrams_collide_as_a_random_hash_would():
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 4096, 400_000).astype(np.int64)
    setup = _setup((2,), vocab=4096)
    (stream,) = count_orders([tokens], setup, 1)
    primes = engram_table_sizes(1_000_003, 2)
    result = evaluate_table(stream, primes)
    for head in result["heads"]:
        expected = head["expected_shared_occurrences"]
        assert 0.2 < expected < 0.4
        assert head["shared_occurrences"] == pytest.approx(expected, abs=0.01)
    # Distinct primes decorrelate the heads: all-head sharing is the product.
    assert result["shared_all_heads"] == pytest.approx(
        result["shared_all_heads_if_independent"], abs=0.01
    )
    assert result["shared_all_heads"] < min(
        h["shared_occurrences"] for h in result["heads"]
    )


def test_zipf_frequency_summary_matches_the_generating_law():
    vocab, exponent, size = 3000, 1.1, 3_000_000
    weights = np.arange(1, vocab + 1) ** -exponent
    weights /= weights.sum()
    tokens = np.random.default_rng(2).choice(vocab, size, p=weights).astype(np.int64)
    (stream,) = count_orders([tokens], _setup((1,), vocab=vocab, heads=1), 1)
    summary = stream.summary()
    assert summary["distinct"] == vocab
    assert summary["zipf_slope"] == pytest.approx(exponent, abs=0.08)
    assert summary["top_mass"]["1000"] == pytest.approx(weights[:1000].sum(), abs=0.01)
    beta = summary["heaps_exponent"]
    assert 0 < beta < 1


def test_report_recommends_smallest_meeting_size_and_costs_it():
    tokens = np.random.default_rng(3).integers(0, 64, 20_000).astype(np.int64)
    setup = _setup((2,), heads=2)
    report = analyze(
        [tokens],
        setup,
        candidates=(64, 256, 65536),
        max_clobbered=0.10,
        target_tokens=1_000_000,
    )
    table = report["sites"][0]["orders"][0]["tables"]
    assert [t["clobbered_all_heads"] for t in table] == sorted(
        (t["clobbered_all_heads"] for t in table), reverse=True
    )
    recommendation = report["recommendation"]
    assert recommendation["met"]
    assert recommendation["per_order_requested_rows"] == {2: 65536}
    sizes = recommendation["memory_table_sizes"]
    assert len(set(sizes)) == len(sizes) == setup.streams
    assert recommendation["cost"]["parameters"] == sum(sizes) * setup.memory_dim
    unreachable = analyze([tokens], setup, candidates=(64,), max_clobbered=0.0)
    assert not unreachable["recommendation"]["met"]


def test_sampling_spans_the_input_and_aligns_windows():
    data = np.arange(100_000, dtype=np.int32)
    blocks = sample_blocks([data], 8_000, 8, block_multiple=100)
    assert all(len(b) % 100 == 0 and b[0] % 100 == 0 for b in blocks)
    assert sum(len(b) for b in blocks) <= 8_000
    assert blocks[0][0] == 0 and blocks[-1][-1] > 90_000


def test_cli_reads_a_prepared_root_and_refuses_out_of_vocabulary_ids(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    ids = np.random.default_rng(4).integers(0, 64, 5_000).astype(np.int32)
    np.save(root / "train.npy", ids)
    (root / "manifest.json").write_text(json.dumps({"manifest_sha256": "ab" * 32}))
    command = [
        sys.executable, "-m", "klab", "data", "engram-sizing", str(root),
        "--orders", "2", "--heads", "2", "--vocab-size", "64",
        "--memory-dim", "8", "--sizes", "64,1024", "--json",
        "--output", str(tmp_path / "sizing.json"),
    ]  # fmt: skip
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(done.stdout)
    assert report["sample_tokens"] == 5_000
    assert report["source"]["prepared_manifest_sha256"] == "ab" * 32
    sealed = json.loads((tmp_path / "sizing.json").read_text())
    assert sealed.pop("sha256") == canonical_digest(sealed)
    assert sealed["source"] == report["source"]
    assert math.isfinite(report["sites"][0]["orders"][0]["distinct"])
    bad = command.copy()
    bad[bad.index("64", bad.index("--vocab-size"))] = "32"
    refused = subprocess.run(bad, capture_output=True, text=True, check=False)
    assert refused.returncode == 1
    assert "outside the model vocabulary" in refused.stderr


def _memory_config(sizes, multiplier=1.0, memory="ngram"):
    from pathlib import Path

    from klab.config.loading import load_config

    config = load_config(Path("configs/smoke_memory_cpu.yaml"))
    model = config.model.model_copy(
        update={
            "memory": memory,
            "memory_table_size": sizes[0],
            "memory_table_sizes": tuple(sizes),
        }
    )
    optimizer = config.optimizer.model_copy(
        update={"memory_table_lr_multiplier": multiplier}
    )
    return config.model_copy(update={"model": model, "optimizer": optimizer})


def test_inspect_advises_on_small_unmeasured_tables_without_refusing():
    from klab.model.inspection import memory_advisories

    small = {
        n["code"]: n for n in memory_advisories(_memory_config([257, 263, 269, 271]))
    }
    assert set(small) == {
        "memory_tables_small_for_vocabulary",
        "memory_tables_never_measured",
        "memory_table_optimizer_defaults",
    }
    assert "klab data engram-sizing" in small["memory_tables_never_measured"]["message"]
    assert "unmeasured" in small["memory_tables_never_measured"]["message"]

    sizes = engram_table_sizes(10_007, 4)  # at least 17.5 rows per vocabulary entry
    large = {n["code"] for n in memory_advisories(_memory_config(sizes, 5.0))}
    assert large == {"memory_tables_never_measured"}

    assert memory_advisories(_memory_config([257], memory="none")) == []
