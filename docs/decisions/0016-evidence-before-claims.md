# ADR 0016: Pair local quality observations with verified checkpoints

## Status

Accepted.

## Context

A runnable smoke path and unit tests demonstrate that code executes, but they do not by themselves show that a configured experiment is reproducible, uses held-out data, or produces an observation tied to durable model state. The laboratory must support future software and hardware hypotheses without becoming a benchmark platform or a distributed experiment service.

## Decision

Evaluate held-out data at the initial, configured periodic, and terminal training boundaries. Each evaluation forces a validated local checkpoint, records loss and perplexity in the local metric store, and writes an immutable evaluation report naming the checkpoint generation and digest. `klab evidence RUN_ID` verifies the run’s checkpoint generations and presents only these attached observations.

Treat evidence levels distinctly: behavior tests, artifact verification, a checkpointed held-out local experiment, a controlled comparison, and a hardware implementation assay. Hardware or numerical-format work must state its tolerance and frozen reference vectors; an isolated kernel result never substitutes for end-to-end reference comparison.

## Consequences

The project can report exactly what a run demonstrated without claiming broad model quality or untested accelerator correctness. The added work stays local and file-based. Future MXFP4, RISC, or ASIC paths remain explicit implementations with their own configuration, numerical contract, and evidence; they are not inferred from current FP32 reference runs.

## Interface update — 2026-10-11

The historical decision above names the interface used at the time. Its current
result-reading path is `klab report RUN_ID --kind run`, which keeps verified
checkpoint observations separate from advisory triage. See [result reports](../result-reports.md)
for the consolidated readers. This pointer changes neither the historical
claims nor the identities or evidence of executed runs.

## Related decisions

- [ADR 0014](0014-scale-experiments.md) governs bounded scale comparisons.
- [ADR 0015](0015-single-host-extension-boundaries.md) keeps the core local and explicit.
