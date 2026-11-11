"""CPU guard/capture checks; B70 code generation remains an explicit device gate."""

from copy import deepcopy

import pytest
import torch
from test_document_batching import configured, rows_for

from klab.config.models import RunConfig
from klab.data.document_batching import collate_document_update
from klab.engines.pytorch import output_projection_cross_entropy
from klab.model.rope import RoPE
from klab.model.transformer import DenseLM
from klab.training.continuation import runtime_optimization_transition


def test_dynamic_blocks_and_head_share_graphs_across_padded_shapes(tmp_path):
    torch._dynamo.reset()
    cfg = configured(tmp_path)
    model = DenseLM(cfg.model, cfg.attention)
    reference = deepcopy(model)
    for module in model.modules():
        if isinstance(module, RoPE):
            module.reserve(64, torch.device("cpu"))
    graphs = {"blocks": 0, "head": 0}

    def backend(kind):
        def capture(graph, inputs):
            graphs[kind] += 1
            return graph.forward

        return capture

    for block in model.blocks:
        block.forward_with_aux = torch.compile(
            block.forward_with_aux,
            dynamic=True,
            fullgraph=True,
            backend=backend("blocks"),
        )
    head = torch.compile(
        output_projection_cross_entropy,
        dynamic=True,
        fullgraph=True,
        backend=backend("head"),
    )
    shapes = [(4, 16), (4, 32), (4, 48), (4, 64), (2, 32), (3, 48), (1, 16), (1, 64)]
    with torch._dynamo.config.patch(
        suppress_errors=False, fail_on_recompile_limit_hit=True
    ):
        for repeat in range(2):
            for width, length in shapes:
                rows = rows_for([length - i for i in range(width)])
                batch = collate_document_update(
                    rows, width=4, eos=3, max_length=64, padding_multiple=16
                )[0]
                x, y = torch.from_numpy(batch.inputs), torch.from_numpy(batch.targets)
                mask = torch.from_numpy(batch.position_mask)
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    hidden, aux = model.forward_with_aux(
                        x, position_mask=mask, return_hidden=True
                    )
                    logits, ref_aux = reference.forward_with_aux(x, position_mask=mask)
                loss = head(hidden, y, model.output.weight) + aux
                expected = (
                    torch.nn.functional.cross_entropy(
                        logits.float().flatten(0, 1),
                        y.flatten(),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    + ref_aux
                )
                loss.backward()
                expected.backward()
                torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
                for a, b in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)
            if repeat == 0:
                cold = graphs.copy()
            else:
                assert graphs == cold, "warmed shape family recompiled"
    # B=1 and equal/static-dimension specializations are finite; warm shapes
    # must reuse graphs and stay below the default per-frame limit of eight.
    assert 0 < graphs["blocks"] <= 4 * len(model.blocks)
    assert 0 < graphs["head"] <= 4
    print("Captured full graphs, cold/warm:", graphs, graphs)
    assert not torch.xpu.is_initialized()


def test_dynamic_config_and_continuation_delta(tmp_path):
    source = configured(tmp_path).model_dump(mode="json")
    source["runtime"].update(backend="xpu", precision="bf16")
    source = RunConfig.model_validate(source)
    target = source.model_dump(mode="json")
    target["runtime"].update(
        compile_transformer_blocks=True,
        compile_output_head_loss=True,
        compile_dynamic_shapes=True,
    )
    target = RunConfig.model_validate(target)
    transition = runtime_optimization_transition(source, target, "parent")
    assert {c["field"] for c in transition["changes"]} == {
        "runtime.compile_transformer_blocks",
        "runtime.compile_output_head_loss",
        "runtime.compile_dynamic_shapes",
    }
    invalid = source.model_dump(mode="json")
    invalid["runtime"]["compile_dynamic_shapes"] = True
    with pytest.raises(ValueError, match="requires a compiled"):
        RunConfig.model_validate(invalid)
