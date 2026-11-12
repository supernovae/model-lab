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
  parameter target. A project composes and checks its declarations with
  `klab config compose` and `klab config validate DIR`. Pending work: moving the
  study-specific code that remains in the package (`src/klab/research/`,
  study-named evaluation modules and study configs) out of the core.
- **Scale without idle accelerators.** Data is prepared on CPU with the whole
  host through a binary [token store](docs/token-store.md), and training
  [keeps devices busy](docs/training.md#keep-accelerators-busy) across
  checkpoints, evaluation and several devices on one host. Pending work:
  remaining preparation costs
  ([#291](https://github.com/supernovae/model-lab/issues/291),
  [#293](https://github.com/supernovae/model-lab/issues/293),
  [#295](https://github.com/supernovae/model-lab/issues/295),
  [#296](https://github.com/supernovae/model-lab/issues/296)) and checking
  storage before packing
  ([#225](https://github.com/supernovae/model-lab/issues/225)), across CPU,
  CUDA, ROCm, Intel XPU and Apple MLX.
- **Long context and sparse attention.**
  [`local_selected`](docs/sparse-attention.md) trains 32k to 131k context and
  was measured faster than dense fused attention from 32k up on an Arc Pro B70.
  Pending work: its memory per token, which is about twice dense's, and a
  measurement of long-range retrieval quality.
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
- **A simpler, reliable lab loop.** Resolve the failing nightly suite
  ([#79](https://github.com/supernovae/model-lab/issues/79)), reduce CLI
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
history; dated records about this repository are under
[`docs/history/`](docs/history/README.md). This snapshot grants no
runtime, dataset acquisition, paid-compute or release authority.
