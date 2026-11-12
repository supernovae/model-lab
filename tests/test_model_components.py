"""Component registry, per-layer schedule and architecture presets.

The external kinds below live in a throwaway distribution on ``sys.path`` and
reach Lab only through its ``klab.components`` entry points, the way a Workshop
package would.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError
from test_training import config
from tokenizers import Tokenizer

from klab.architectures import (
    count_parameters,
    load_preset,
    parse_parameter_count,
    preset_names,
    scale_preset,
)
from klab.config.loading import load_config
from klab.config.models import RunConfig
from klab.evaluation.generation import generate_with_token_ids
from klab.model.attention.dense import DenseAttention
from klab.model.ffn import SwiGLU
from klab.model.inspection import named_tensor_inventory, parameter_inventory
from klab.model.moe import TopKMoE
from klab.model.transformer import DenseLM
from klab.training.trainer import train

_PACKAGE = '''
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from klab.components import Component


class CosineParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 10.0
    bias: bool = False


class CosineAttention(nn.Module):
    """Causal attention over L2-normalized queries and keys."""

    def __init__(self, hidden, heads, params):
        super().__init__()
        self.heads, self.temperature = heads, params.temperature
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=params.bias)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        batch, length, hidden = x.shape
        q, k, v = (
            part.view(batch, length, self.heads, -1).transpose(1, 2)
            for part in self.qkv(x).chunk(3, dim=-1)
        )
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        scores = self.temperature * q @ k.transpose(-2, -1)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(batch, length, hidden))


class PenalizedMLP(nn.Module):
    """ReLU feed-forward that reports an activation penalty and one metric."""

    def __init__(self, hidden, width):
        super().__init__()
        self.up, self.down = nn.Linear(hidden, width), nn.Linear(width, hidden)
        self.last_penalty = None

    def forward_with_aux(self, x, *, valid_target_mask=None, diagnostics="scalar"):
        inner = torch.relu(self.up(x))
        penalty = 1e-3 * inner.float().pow(2).mean()
        self.last_penalty = penalty.detach()
        return self.down(inner), penalty

    def forward(self, x):
        return self.forward_with_aux(x)[0]

    def metric_tensors(self):
        return {} if self.last_penalty is None else {"penalty": self.last_penalty}


class TokenBias(nn.Module):
    """Adds a learned per-token vector to the hidden state."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.table = nn.Embedding(vocab, hidden)

    def forward(self, hidden, token_ids, *, position_mask=None):
        return hidden + self.table(token_ids)

    def forward_last(self, hidden, token_ids, history):
        return hidden + self.table(token_ids[:, -hidden.shape[1] :]), None


ATTENTION = Component(
    "attention",
    "toy_cosine",
    lambda model, attention, params: CosineAttention(
        model.hidden_dim, model.num_heads, params
    ),
    params=CosineParams,
)
FFN = Component(
    "ffn",
    "toy_mlp",
    lambda model, attention, params: PenalizedMLP(model.hidden_dim, model.ffn_dim),
)
MEMORY = Component(
    "memory",
    "toy_bias",
    lambda model, attention, params: TokenBias(model.vocab_size, model.hidden_dim),
)
MISLABELED = Component("ffn", "something_else", FFN.build)
'''


@pytest.fixture(scope="module", autouse=True)
def toy_distribution(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("site")
    (root / "toy_components.py").write_text(_PACKAGE)
    info = root / "toy_components-0.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: toy-components\nVersion: 0.3\n"
    )
    (info / "entry_points.txt").write_text(
        "[klab.components]\n"
        "attention.toy_cosine = toy_components:ATTENTION\n"
        "ffn.toy_mlp = toy_components:FFN\n"
        "memory.toy_bias = toy_components:MEMORY\n"
        "ffn.toy_mislabeled = toy_components:MISLABELED\n"
    )
    sys.path.insert(0, str(root))
    yield
    sys.path.remove(str(root))
    sys.modules.pop("toy_components", None)


def _variant(base: RunConfig, **sections: dict) -> RunConfig:
    """Re-validate ``base`` with some sections' fields replaced."""
    document = base.model_dump(mode="json")
    for section, fields in sections.items():
        document[section] = {**document[section], **fields}
    return RunConfig.model_validate(document)


def _short(base: RunConfig, **sections: dict) -> RunConfig:
    return _variant(base, training={"max_steps": 3, "max_tokens": 96}, **sections)


def _generate(loaded, run: RunConfig, **options: object) -> list[int]:
    return generate_with_token_ids(
        loaded.model,
        Tokenizer.from_file(str(run.tokenizer.path)),
        "a b a b a",
        run.model.max_seq_len,
        40,
        loaded.device,
        **options,
    )[1]


def _assert_inventory_matches(run: RunConfig, model: DenseLM) -> None:
    stored = {
        name: spec.shape
        for name, spec in named_tensor_inventory(run.model, run.attention).items()
        if spec.alias_of is None
    }
    assert stored == {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    }
    assert parameter_inventory(run).total == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_kinds_from_another_package_train_checkpoint_and_generate(
    tmp_path: Path,
) -> None:
    run = _short(
        config(tmp_path),
        model={"num_layers": 2, "ffn": "toy_mlp", "memory": "toy_bias"},
        attention={"kind": "toy_cosine", "params": {"temperature": 5}},
    )
    # Stored params carry the package's defaults, and the config names its source.
    assert run.attention.params == {"temperature": 5.0, "bias": False}
    assert run.external_components == {
        "attention.toy_cosine": "toy-components 0.3",
        "ffn.toy_mlp": "toy-components 0.3",
        "memory.toy_bias": "toy-components 0.3",
    }

    run_id = train(run, run_id="external")

    from klab.evaluation.inference import load_run

    # The run records which package and version supplied the code.
    with sqlite3.connect(run.logging.root_dir / "experiments.sqlite3") as database:
        recorded = database.execute(
            "SELECT payload_json FROM events WHERE run_id=? AND kind=?",
            (run_id, "external_components"),
        ).fetchone()
    assert json.loads(recorded[0]) == run.external_components

    loaded = load_run(run_id, run.logging.root_dir, None, "cpu")
    block = loaded.model.blocks[1]
    assert type(block.attention).__name__ == "CosineAttention"
    assert type(block.ffn).__name__ == "PenalizedMLP"
    assert type(loaded.model.memory).__name__ == "TokenBias"
    assert block.attention.temperature == 5.0
    _assert_inventory_matches(run, loaded.model)
    # The feed-forward's auxiliary loss reaches the model's loss, and its own
    # metric is reported per layer under its slot.
    _, auxiliary = loaded.model.forward_with_aux(torch.zeros((1, 4), dtype=torch.long))
    assert auxiliary > 0
    assert "ffn/layer_1/penalty" in loaded.model.architecture_metric_tensors()
    # This attention keeps no cache, so generation recomputes the prefix.
    capability = loaded.model.incremental_cache_capability
    assert not capability.supported and "CosineAttention" in capability.reason
    assert len(_generate(loaded, run)) == 40


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"attention": {"kind": "nowhere"}}, "entry point named 'attention.nowhere'"),
        ({"model": {"ffn": "toy_mislabeled"}}, "the two must match"),
        (
            {"attention": {"kind": "toy_cosine", "params": {"heat": 1}}},
            "heat",
        ),
        ({"attention": {"params": {"temperature": 1}}}, "accepts no params"),
        (
            {"attention": {"kind": "toy_cosine", "window_size": 8}},
            "configured through attention.params",
        ),
        (
            {"model": {"memory": "toy_bias", "memory_dim": 8}},
            "configured through model.memory_params",
        ),
        (
            {
                "model": {
                    "layers": [
                        {"blocks": [0], "ffn": {"kind": "moe", "num_experts": 2}},
                        {"blocks": "0:", "ffn": {"kind": "dense"}},
                    ]
                }
            },
            "more than once",
        ),
        (
            {"model": {"layers": [{"blocks": [3], "ffn": {"kind": "dense"}}]}},
            "existing decoder blocks",
        ),
    ],
)
def test_invalid_component_settings_are_refused(
    tmp_path: Path, sections: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _variant(config(tmp_path), **sections)


def test_per_layer_schedule_trains_and_decodes_from_its_cache(tmp_path: Path) -> None:
    run = _short(
        config(tmp_path),
        model={
            "num_layers": 4,
            "layers": [
                {
                    "blocks": "1::2",
                    "attention": {"kind": "sliding_window", "window_size": 8},
                },
                {
                    "blocks": [0, 1],
                    "ffn": {"kind": "moe", "num_experts": 3, "experts_per_token": 2},
                },
            ],
        },
    )
    run_id = train(run, run_id="schedule")

    from klab.evaluation.inference import load_run

    loaded = load_run(run_id, run.logging.root_dir, None, "cpu")
    blocks = loaded.model.blocks
    assert [type(block.ffn) for block in blocks] == [TopKMoE, TopKMoE, SwiGLU, SwiGLU]
    assert all(isinstance(block.attention, DenseAttention) for block in blocks)
    assert [block.attention.window_size for block in blocks] == [None, 8, None, 8]
    _assert_inventory_matches(run, loaded.model)
    assert loaded.model.incremental_cache_capability.backend == "dense+sliding"
    cached = _generate(loaded, run)
    assert len(cached) == 40
    assert cached == _generate(loaded, run, use_cache=False)


def test_default_settings_leave_a_config_identity_unchanged(tmp_path: Path) -> None:
    document = config(tmp_path).model_dump(mode="json")
    assert not {"layers", "ffn_params", "memory_params"} & set(document["model"])
    assert "params" not in document["attention"]


@pytest.mark.parametrize("name", preset_names())
def test_shipped_presets_scale_to_a_target_and_build(name: str) -> None:
    # The memory preset's hash tables alone hold about 33M parameters.
    target = parse_parameter_count("36M" if name == "moe_ngram" else "8M")
    scaled = scale_preset(
        name, load_preset(name)[1], target, vocab_size=512, max_seq_len=128
    )
    assert abs(scaled.relative_error) <= 0.10
    reference = load_preset(name)[1].model
    assert (
        scaled.model.hidden_dim // scaled.model.num_heads
        == reference.hidden_dim // reference.num_heads
    )
    model = DenseLM(scaled.model, scaled.attention)
    assert scaled.parameters == sum(p.numel() for p in model.parameters())
    assert count_parameters(scaled.model, scaled.attention) == scaled.parameters
    logits = model(torch.zeros((1, 8), dtype=torch.long))
    assert logits.shape == (1, 8, 512)


def test_architecture_derive_writes_a_run_config_from_a_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from klab.cli.main import main

    base = config(tmp_path)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base.model_dump(mode="json")))
    output = tmp_path / "derived" / "moe.yaml"
    output.parent.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "architecture",
            "derive",
            "moe",
            "--params",
            "8M",
            "--base",
            str(base_path),
            "--output",
            str(output),
            "--json",
        ],
    )
    main()
    result = json.loads(capsys.readouterr().out)

    derived = load_config(output)
    assert result["status"] == "ok"
    assert parameter_inventory(derived).total == result["parameters"]
    assert abs(result["relative_error"]) <= 0.10
    # The architecture comes from the preset; the vocabulary, context and
    # every other section come from the base.
    assert derived.model.ffn == "moe" and derived.model.num_experts == 8
    assert derived.model.vocab_size == base.model.vocab_size
    assert derived.model.max_seq_len == base.model.max_seq_len
    assert derived.training == base.training
    assert derived.dataset == base.dataset
    assert derived.tokenizer.path == base.tokenizer.path
