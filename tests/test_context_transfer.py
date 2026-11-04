"""Long-context mechanics and bounded SDPA buffers on CPU fixtures."""

import numpy as np
import pytest
import torch
from test_engram_integration import model_config

from sparselab.address_hash import engram_table_sizes
from sparselab.config.models import AttentionConfig
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.model.attention.dense import DenseAttention
from sparselab.model.transformer import DenseLM


def test_sdpa_avoids_unused_dense_masks_and_keeps_reference_diagnostics():
    sdpa = DenseAttention(8, 2, 8192, 10000.0, implementation="sdpa")
    reference = DenseAttention(8, 2, 32, 10000.0)
    reference.load_state_dict(sdpa.state_dict())
    assert sdpa.causal_mask.numel() == 0
    x = torch.randn(1, 17, 8)
    torch.testing.assert_close(sdpa(x), reference(x))
    torch.testing.assert_close(
        sdpa.attention_probabilities(x), reference.attention_probabilities(x)
    )
    assert sdpa.causal_mask.numel() == 0
    with pytest.raises(ValueError, match="context"):
        sdpa(torch.empty(1, 8193, 8))


@pytest.mark.parametrize("context", [4096, 8192])
def test_long_document_rope_multiport_and_chunked_cache(tmp_path, context):
    config = model_config(
        tmp_path,
        "deepseek_safe_v1",
        max_seq_len=context,
        memory_block_index=None,
        memory_block_indices=(0, 1),
        memory_table_sizes=engram_table_sizes(31, 8),
    )
    model = DenseLM(config, AttentionConfig(implementation="sdpa")).eval()
    targets = np.r_[np.arange(context - 1) % 256 + 4, 3].astype(np.int32)
    plan = MatchedStoryDataset(
        targets,
        context,
        3,
        layout="story",
        targets_per_update=1,
        unpadded_documents=True,
    )
    x, y, *_ = plan.numpy_microblock(0)
    assert len(x) == context and x[0] == 3 and y[-1] == 3
    ids = torch.from_numpy(x)[None]
    with torch.inference_mode():
        full = model(ids)
        chunks = []
        cache = None
        for start, stop in [(0, 2047), (2047, 2049), (2049, context)]:
            logits, cache = model.forward_cached(
                ids[:, start:stop],
                cache=cache,
                **({"cache_capacity": context} if cache is None else {}),
            )
            chunks.append(logits)
        torch.testing.assert_close(torch.cat(chunks, dim=1), full, rtol=1e-4, atol=1e-5)
        short = ids[:, :32].clone()
        short[:, 8:] = 17
        torch.testing.assert_close(
            model(short)[:, :8], full[:, :8], rtol=1e-4, atol=1e-5
        )
        fresh, _ = model.forward_cached(ids[:, :32], cache_capacity=context)
        torch.testing.assert_close(fresh, full[:, :32], rtol=1e-4, atol=1e-5)
    assert all(block.attention.causal_mask.numel() == 0 for block in model.blocks)
    assert model.blocks[0].attention.rope.cos.shape[2] == context


def test_context_transfer_cli_rejects_implicit_or_unbound_promotion():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "train",
            "configs/smoke_cpu.yaml",
            "--extend-context",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--extend-context requires --add-ngram-memory and --promote" in result.stderr


def test_native_preparation_accepts_short_whole_document_splits(tmp_path):
    from lab_fixtures import write_smoke_inputs

    from sparselab.config.loading import load_config
    from sparselab.config.models import RunConfig
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer

    config = load_config(write_smoke_inputs(tmp_path))
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        '{"text":"A complete first document."}\n{"text":"A complete second document."}\n'
    )
    validation.write_text('{"text":"A separate held out document."}\n')
    raw = config.model_dump(mode="json")
    raw["model"]["max_seq_len"] = 4096
    raw["training"].update(
        seq_len=4096,
        micro_batch_size=1,
        matched_stories={"layout": "story", "unpadded_documents": True},
    )
    raw["dataset"].update(
        source="local_text",
        revision="short-whole-document-fixture",
        license="CC0-1.0",
        train_path=str(train),
        validation_path=str(validation),
        train_max_documents=2,
        validation_max_documents=1,
    )
    measured = RunConfig.model_validate(raw)
    data = prepare_data(measured, load_tokenizer(measured.tokenizer.path))
    assert 0 < len(data.train) < 4096 and 0 < len(data.validation) < 4096
    plan = MatchedStoryDataset(
        data.train,
        4096,
        3,
        layout="story",
        targets_per_update=1,
        unpadded_documents=True,
    )
    assert len(plan) == 2
    assert sum(len(plan.numpy_microblock(i)[1]) for i in range(len(plan))) == len(
        data.train
    )
