# Training program example

A two-phase [training program](../../docs/experiment-programs.md) on synthetic
data, small enough for a laptop CPU: `pretrain` trains 20 steps, then
`continue` extends the same run to 40 steps on a longer budget.

Copy this directory into a workspace outside the checkout; `run.yaml` and
`tokenizer.yaml` put the tokenizer, data and runs beside it.

```sh
klab tokenizer train tokenizer.yaml
klab experiment validate offline-chain.yaml --json
klab experiment lock offline-chain.yaml --json
```

`heldout-suite.yaml` is the plan's evaluation suite: held-out loss as a gate.
