# Feature roadmap

This is a short, human-readable directional snapshot of pending features, not a
second checklist or a record of shipped capabilities. See the
[README feature matrix](README.md#feature-matrix) for available workflows.

## Where work is tracked

Use [GitHub issues](https://github.com/supernovae/model-lab/issues) for
feature requests, bug reports and proposed code work. Search existing issues
before opening a new one, and describe the use case or expected behavior.
Implementation, review and CI live in
[GitHub pull requests](https://github.com/supernovae/model-lab/pulls) and
their checks. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

This roadmap summarizes direction; issues and pull requests track individual
work items and their status. It is not a second task board or a release promise.

## Current direction

- **A general model-building lab.** Lab is the engine for building consistent,
  repeatable and comparable models at any size; experiments are one use of it.
  Model projects and recipes live in
  [Kybern Workshop](https://github.com/supernovae/kybern-workshop). [Components](docs/components.md) are registered by kind and can come from
  another package, blocks can differ by layer, and named presets scale to a
  parameter target. Pending work: moving the remaining study- and
  corpus-specific schema out of the core
  ([#359](https://github.com/supernovae/model-lab/issues/359)).
- **Scale without idle accelerators.** Prepare data completely on CPU with the
  whole host, then keep devices busy: preparation at 10B to 100B tokens
  ([#356](https://github.com/supernovae/model-lab/issues/356)), checkpoint and
  evaluation dead time ([#354](https://github.com/supernovae/model-lab/issues/354))
  and several devices on one host
  ([#355](https://github.com/supernovae/model-lab/issues/355)), across CPU, CUDA,
  ROCm, Intel XPU and Apple MLX.
- **Long context and sparse attention.** 32k to 256k context without dense
  attention memory ([#353](https://github.com/supernovae/model-lab/issues/353)).
- **The loop after pretraining.** [Post-training](docs/post-training.md)
  stages chain from a parent checkpoint under a registered objective:
  instruction tuning, teacher distillation and preference optimization (DPO),
  with chat templates and Hugging Face Llama export. Pending work: low-rank
  adapters, rollout-based methods and quantization, plus speculative decoding
  with n-gram and draft-model drafters
  ([#358](https://github.com/supernovae/model-lab/issues/358)).
- **Trustworthy evaluation.** [PR #72](https://github.com/supernovae/model-lab/pull/72)
  has landed: held-back final evaluation, text-level needle comparisons, recall
  diagnostics and verdict safeguards are available. Suite-v4 reference resealing
  and across-seed uncertainty intervals remain follow-up work. Keep within-run
  evaluation noise distinct from training variability across paired seeds; see
  the [probe guide](docs/probe-battery.md) for final-evaluation boundaries.
- **Richer comparisons and inspection.** Broader reference trajectories,
  fixed-device latency and benchmark coverage, plus MLX and semantic-pack
  support in probes and the explorer.
- **A simpler, reliable lab loop.** Resolve known test failures, reduce CLI
  choices around the existing experiment journeys, consolidate result lookup
  and reports, and simplify configuration and legacy code while preserving
  evidence identities and useful regression coverage.
- **Stronger experiment workflows.** Support larger local comparisons,
  memory-offload/portability studies and agent iteration with protected final
  evaluation. Scientific readiness and conclusions remain in research records.

## Longer-term directions

Shared read-only experiment-ledger views, served-checkpoint provenance and
broader hardware/provider qualification remain conditional on concrete needs.
A real TPU backend and managed-provider lifecycle support need their own
implementation and measured validation. Multi-host distributed training remains
deferred under the
[single-host decision](docs/decisions/0015-single-host-extension-boundaries.md).

The distinct [scientific research roadmap](docs/research/roadmap.md) and
[original research records](experiments/research/) retain their evidence and
history. Historical mentions of the former `TODO.md` are preserved in those
records; its prior contents remain in Git history. This snapshot grants no
runtime, dataset acquisition, paid-compute or release authority.
