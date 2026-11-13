"""Compact functional coverage for the optional compiled output/loss path."""

import torch
from torch.nn import functional

from klab.config.models import AttentionConfig, ModelConfig, RuntimeConfig
from klab.engines.pytorch import output_projection_cross_entropy
from klab.model.transformer import DenseLM


def test_compiled_output_loss_matches_eager_with_tied_weights() -> None:
    torch.manual_seed(19)
    model = DenseLM(
        ModelConfig(
            vocab_size=264,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
        ),
        AttentionConfig(),
    )
    assert model.output.weight is model.embedding.weight
    tokens = torch.randint(0, 264, (2, 8))
    labels = torch.randint(0, 264, (2, 8))
    labels[0, -1] = -100
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, auxiliary = model.forward_with_aux(tokens)
    eager = (
        functional.cross_entropy(
            logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        + auxiliary
    )
    eager.backward()
    reference = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        hidden, candidate_auxiliary = model.forward_with_aux(tokens, return_hidden=True)
    compiled = torch.compile(
        output_projection_cross_entropy, backend="eager", dynamic=False
    )
    candidate = compiled(hidden, labels, model.output.weight) + candidate_auxiliary
    candidate.backward()
    torch.testing.assert_close(candidate, eager, rtol=1e-5, atol=1e-5)
    for name, param in model.named_parameters():
        if name in reference:
            torch.testing.assert_close(
                param.grad, reference[name], rtol=1e-4, atol=1e-5
            )


def test_compiled_output_loss_option_preserves_default_identity() -> None:
    assert "compile_output_head_loss" not in RuntimeConfig().model_dump(mode="json")
    enabled = RuntimeConfig(
        engine="pytorch", backend="xpu", precision="bf16", compile_output_head_loss=True
    )
    assert enabled.model_dump(mode="json")["compile_output_head_loss"] is True
    try:
        RuntimeConfig(
            engine="pytorch",
            backend="cpu",
            precision="fp32",
            compile_output_head_loss=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unvalidated runtime was accepted")


def test_chunked_output_loss_matches_whole_head_loss_and_gradients() -> None:
    from klab.engines.output_head import chunked_output_head_cross_entropy

    torch.manual_seed(23)
    model = DenseLM(
        ModelConfig(
            vocab_size=264,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
        ),
        AttentionConfig(),
    )
    tokens = torch.randint(0, 264, (2, 8))
    labels = torch.randint(0, 264, (2, 8))
    labels[0, -1] = -100
    logits, auxiliary = model.forward_with_aux(tokens)
    whole = (
        functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        + auxiliary
    )
    whole.backward()
    reference = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    # Five positions per chunk leaves a short last chunk of the sixteen.
    for chunk_tokens in (5, 16, 64):
        model.zero_grad(set_to_none=True)
        hidden, candidate_auxiliary = model.forward_with_aux(tokens, return_hidden=True)
        chunked = (
            chunked_output_head_cross_entropy(
                hidden, labels, model.output.weight, chunk_tokens, torch.enable_grad
            )
            + candidate_auxiliary
        )
        torch.autograd.backward(chunked, inputs=list(model.parameters()))
        torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-5)
        for name, parameter in model.named_parameters():
            if name in reference:
                torch.testing.assert_close(
                    parameter.grad, reference[name], rtol=1e-4, atol=1e-6
                )
    with torch.no_grad():
        hidden, _ = model.forward_with_aux(tokens, return_hidden=True)
        evaluated = chunked_output_head_cross_entropy(
            hidden, labels, model.output.weight, 5, torch.enable_grad
        )
    torch.testing.assert_close(evaluated + auxiliary.detach(), whole.detach())


def test_chunked_output_head_option_keeps_default_identity_and_lowers_estimate() -> (
    None
):
    assert "output_head_chunk_tokens" not in RuntimeConfig().model_dump(mode="json")
    chunked = RuntimeConfig(output_head_chunk_tokens=2048)
    assert chunked.model_dump(mode="json")["output_head_chunk_tokens"] == 2048
    try:
        RuntimeConfig(
            engine="pytorch",
            backend="xpu",
            precision="bf16",
            compile_output_head_loss=True,
            output_head_chunk_tokens=2048,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("two output-head paths were accepted together")
