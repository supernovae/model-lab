# Model scaling and accounting

For the practical progression from a tiny exercise to a useful task, start with [From flashcards to a useful local assistant](from-toy-to-useful.md). More parameters do not by themselves turn story continuation or alias memorization into instruction-following chat.

`total` counts every unique parameter storage once. `trainable` is the trainable subset. `active_per_token` is a direct-use convention, not FLOPs or a context-wide unique-weight measurement.

For a tied dense decoder, all parameter storage is active per token. For an untied dense decoder, the convention excludes the full input-embedding table and includes one input row. Tied output-head storage is attributed to `embedding`; `output_head` is zero.

For local Top-K MoE, `total` includes router, shared expert when configured, and every routed expert. `expert` reports all expert SwiGLU storage; `active_per_token` includes router storage, the selected experts, and a shared expert where enabled. This is an accounting convention, not a claim that router work, indexing, or dispatch has zero cost. `ffn` remains the dense SwiGLU category and is zero for a fully MoE model. When an Engram mode is enabled, its table, projection, and gate are reported as memory storage; they are ordinary local model state, not an external-memory exemption.

Model-weight bytes sum unique parameter tensors once. Estimated checkpoint tensor bytes describe persisted tensor payloads for the configured optimizer after its first update; they exclude gradients, RNG, metadata, and serialization overhead. Inspect the configuration rather than extrapolating a fixed ratio across optimizer or architecture changes.

The supplied 8192-vocabulary tied dense presets are 3,344,064 (`micro_dense.yaml`), 6,917,376 (`dense_7m.yaml`), 10,244,160 (`dense_10m.yaml`), 29,893,120 (`dense_25m.yaml`), and 50,274,752 (`dense_50m.yaml`) parameters. `klab inspect` computes their tensor shapes without allocating a model or opening tokenizer/portable-package assets. Its JSON includes a named-category inventory, memory estimate, and passive runtime readings; the human view separates Model, Estimated training memory, Device, and Result. All scale presets hold TinyStories revision, tokenizer identity, seed, sequence length, token budget, and optimizer settings constant. `configs/smoke_moe_cpu.yaml` is a small local-router plumbing configuration; it is not a quality or scaling benchmark.

`instruction_starter.yaml` reuses the 3,344,064-parameter backbone with the instruction tokenizer/curriculum and a 256,000-target ceiling. `instruction_100m.yaml` has 104,843,648 parameters and a 20-million-target ceiling. These are different development budgets, not a controlled size comparison or a promise of conversational quality. Growing a backbone requires a new run; promotion only accepts compatible architecture and tokenizer semantics.

## Pinned external reference observation

`klab reference pythia trajectory CONFIG --output PATH` is an explicit, optional observation adapter for the official `EleutherAI/pythia-70m-deduped` checkpoints `step0`, `step10000`, and `step143000`. It accepts only a FineWeb-Edu configuration with a non-null pinned dataset revision and config, evaluates the same bounded validation documents for each selected checkpoint, and writes a machine-readable identity report outside Git.

The adapter installs with `kybern-model-lab[reference]`, downloads only the fixed registered Hub commits on this explicit command, and permits only safetensors weights plus fixed model/tokenizer/license/README metadata. It rejects pickle-style weights, Python code, remote-code mappings, and quantization metadata; loading uses built-in `GPTNeoXForCausalLM` and `GPTNeoXTokenizerFast` with remote code disabled.
FineWeb-Edu is identified in the output as ODC-BY-1.0; that database license does not settle rights in independently sourced web content.

Pythia weights are Apache-2.0. This is observation only: it does not download, package, or reproduce The Pile, and it is not a controlled comparison with Model Lab. Pythia and Model Lab can have different tokenizers, data histories, architectures, and training procedures, so their losses must not be presented as matched architecture effects.

## Recorded comparisons

Use the [experiment ledger](research/experiment-ledger.md) to find retained
scaling and lookup comparisons. Their data, checkpoints and hardware describe
those observations; choose your own declared controls and target budget.
