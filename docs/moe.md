# Mixture of Experts

Model Lab's local MoE replaces a dense SwiGLU block with independently parameterized SwiGLU experts. For a normalized hidden vector $x \in \mathbb{R}^D$, the router computes logits $r = Wx$ and probabilities $p = \operatorname{softmax}(r)$. It selects the $K$ largest entries and renormalizes their probabilities:

$$y = \sum_{e \in \operatorname{TopK}(p)} \frac{p_e}{\sum_{j \in \operatorname{TopK}(p)} p_j} E_e(x).$$

`model.num_experts` chooses the routed expert count; `model.experts_per_token` chooses $K`; `model.shared_expert` adds a dense SwiGLU contribution for every token. `model.router_aux_loss_coefficient` enables the simple auxiliary load-balancing term $E\sum_e \bar p_e\bar l_e$, where $\bar p$ is mean router probability and $\bar l$ is selected-token load.

The trainer records the auxiliary loss plus per-layer router entropy, maximum expert fraction, and mean selected routing probability. The latest forward pass retains detached router logits, selected experts, and normalized weights for diagnosis. These are local reference mechanics: there is no capacity limit, token dropping, distributed all-to-all exchange, or auxiliary-loss-free balancing strategy.

Run the CPU smoke:

```sh
uv run --locked --extra cpu klab train --runs-dir klab-work/runs configs/smoke_moe_cpu.yaml --run-id moe-smoke
```

Compare it with `configs/smoke_cpu.yaml` only at matching tokenizer, data, and token budget. Total parameters include all routed experts; active-per-token accounting includes only the selected experts and an optional shared expert.

The [`moe` standalone lesson](research/lesson-paths.md) exposes the same router shapes and diagnostics before any training. The [routed-capacity study](research/engram-moe-capacity.md) fixes auxiliary balancing at zero for its controlled recipe and does not establish iso-compute or timing equivalence.
