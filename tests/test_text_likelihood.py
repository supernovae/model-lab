from __future__ import annotations

import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from klab.evaluation.text_likelihood import score_story


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        ["aé"],
        BpeTrainer(
            vocab_size=260,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        ),
    )
    return tokenizer


class PositionModel(torch.nn.Module):
    """Logits depend on both actual context tokens and reset window positions."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.inputs: list[list[int]] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert not self.training
        assert not torch.is_grad_enabled()
        ids = input_ids.squeeze(0).tolist()
        self.inputs.append(ids)
        logits = torch.zeros(
            (1, len(ids), self.vocabulary_size), device=input_ids.device
        )
        for position, token_id in enumerate(ids):
            logits[0, position, (token_id + position + 1) % self.vocabulary_size] = 2.3
        return logits


def test_long_unicode_story_scores_each_token_and_eos_once_across_short_tail() -> None:
    tokenizer = _tokenizer()
    text = "é" + "a" * 198
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert len(ids) == 200  # UTF-8 é occupies two vocabulary IDs and two bytes.
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    inputs = [eos, *ids]
    targets = [*ids, eos]
    model = PositionModel(tokenizer.get_vocab_size())
    model.train()

    result = score_story(model, tokenizer, text, device=torch.device("cpu"))

    assert model.training
    assert model.inputs == [inputs[:128], inputs[64:192], inputs[128:201]]
    assert [len(window) for window in model.inputs] == [128, 128, 73]
    # Derive log probability independently for each expected target/context pair.
    correct = 0
    for first, last, context_first in [(0, 128, 0), (128, 192, 64), (192, 201, 128)]:
        for index in range(first, last):
            predicted = (
                inputs[index] + index - context_first + 1
            ) % model.vocabulary_size
            correct += predicted == targets[index]
    expected = (
        len(targets) * math.log(model.vocabulary_size - 1 + math.exp(2.3))
        - 2.3 * correct
    )
    assert result["nll_nats"] == pytest.approx(expected, rel=1e-6)
    assert result["token_count"] == 200
    assert result["target_count"] == 201
    assert result["utf8_bytes"] == 200
    assert result["nll_per_byte"] == pytest.approx(expected / 200)
    assert result["bits_per_byte"] == pytest.approx(expected / (200 * math.log(2)))


def test_native_context_scores_same_targets_with_longer_window() -> None:
    tokenizer = _tokenizer()
    model = PositionModel(tokenizer.get_vocab_size())
    text = "a" * 600
    result = score_story(
        model,
        tokenizer,
        text,
        device=torch.device("cpu"),
        window_size=512,
        stride=256,
    )
    assert result["target_count"] == 601
    assert [len(window) for window in model.inputs] == [512, 345]
    inputs = [
        tokenizer.token_to_id("<eos>"),
        *tokenizer.encode(text, add_special_tokens=False).ids,
    ]
    assert model.inputs == [inputs[:512], inputs[256:601]]


@pytest.mark.parametrize(
    "story,expected_lengths",
    [("é", [3]), ("a" * 127, [128]), ("a" * 128, [128, 65])],
)
def test_short_and_exact_window_boundary_score_final_eos(
    story: str, expected_lengths: list[int]
) -> None:
    tokenizer = _tokenizer()
    model = PositionModel(tokenizer.get_vocab_size())
    result = score_story(model, tokenizer, story, device=torch.device("cpu"))
    assert [len(window) for window in model.inputs] == expected_lengths
    assert model.inputs[0][0] == tokenizer.token_to_id("<eos>")
    assert result["target_count"] == result["token_count"] + 1
    assert result["utf8_bytes"] == len(story.encode("utf-8"))


def test_empty_story_is_rejected_without_dividing_by_zero() -> None:
    with pytest.raises(ValueError, match="at least one UTF-8 byte"):
        score_story(PositionModel(260), _tokenizer(), "", device=torch.device("cpu"))
