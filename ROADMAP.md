# Roadmap

Where Lab is heading. Individual work items live in [GitHub
issues](https://github.com/supernovae/model-lab/issues); this page is a short
summary, not a task list or a release promise.

## Now

- **A general model-building engine.** Lab builds consistent, repeatable,
  comparable models at any size. Model projects live in their own repositories,
  such as [Kybern Workshop](https://github.com/supernovae/kybern-workshop), and
  check their declarations with `klab config validate DIR`. Next: move the
  remaining project-specific code out of the package, including the portability
  and allocation hooks in the trainer and the story-specific batching options.
- **Scale without idle accelerators.** Data is prepared on CPU with the whole
  host, and training keeps devices busy across checkpoints, evaluation and
  several GPUs on one host. Next: faster mixture reads and token-cache hits
  ([#291](https://github.com/supernovae/model-lab/issues/291),
  [#293](https://github.com/supernovae/model-lab/issues/293),
  [#295](https://github.com/supernovae/model-lab/issues/295),
  [#296](https://github.com/supernovae/model-lab/issues/296)) and checking
  storage before packing ([#225](https://github.com/supernovae/model-lab/issues/225)).
- **Long context.** [`local_selected`](docs/sparse-attention.md) attention trains
  at 32k to 131k tokens of context. Next: lower its memory per token and measure
  long-range retrieval quality.
- **After pretraining.** [Post-training](docs/post-training.md) covers
  instruction tuning, distillation, preference optimization and Hugging Face
  export. Next: low-rank adapters, rollout-based methods, quantization, and
  speculative decoding with corpus and draft-model drafters
  ([#358](https://github.com/supernovae/model-lab/issues/358)).
- **A reliable loop.** Fix the nightly full suite
  ([#79](https://github.com/supernovae/model-lab/issues/79)) and keep the
  command set small.

## Later

A real TPU backend, managed cloud providers and multi-host training each need
their own design and measurement. Lab runs one model on one host today.
