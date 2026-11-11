# Papers and experiments in Kybern Model Lab

Use papers to choose a question, a control and a measurement. Kybern Model Lab
implements small reference mechanisms and declared experiments; it does not
reproduce every training system or headline result below. Pick by what you want
to learn, not a prescribed reading or execution order.

## Choose a question

The linked personal recipe guides are historical Workshop archive pointers, not
current packaged catalog entries. New studies use the
[external-resource authoring contract](docs/external-research-resources.md) and
explicit seeds. The optimizer comparison uses RunConfig/ExperimentPlan.

| Question and motivation | Paper connection | Historical guide or current mechanism | What remains unproven |
| --- | --- | --- | --- |
| Can lookup capacity compensate for a narrower feed-forward network? | [Engram](https://arxiv.org/abs/2601.07372): gated n-gram lookup complements neural computation. | `engram-ffn-substitution-v1`; [guide](docs/research/engram-ffn-substitution.md) | Added tables change total parameters; small reference runs do not inherit large-model gains. |
| Where should conditional capacity go: experts or memory? | [Engram](https://arxiv.org/abs/2601.07372) motivates allocation between lookup and routed computation. | `engram-moe-capacity-v1`; [guide](docs/research/engram-moe-capacity.md) | Selected FFN capacity is not equal measured FLOPs, runtime or total storage. |
| Does memory change the cost of compressing attention? | [DeepSeek-V2](https://arxiv.org/abs/2405.04434): MLA compresses key/value representations alongside MoE. | `engram-mla-compression-v1`; [guide](docs/research/engram-mla-compression.md) | The lab caches expanded keys and latent values; it does not reproduce the paper's compressed-cache savings. |
| How much attention context can a small model discard? | [Native Sparse Attention](https://arxiv.org/abs/2502.11089): compressed, selected and local attention paths. | `engram-sparse-budget-v1`; [guide](docs/research/engram-sparse-budget.md) | The lab's block selector is simpler; full three-path NSA requires new implementation. |
| Do learned memory values transfer between recipients? | [Tokenizer-Agnostic Engram](https://arxiv.org/abs/2607.29065): byte hashing aligns addresses across tokenizations. | [Learned portability protocol](docs/research/learned-engram-portability.md) | The existing study tests widths within one family. Address agreement alone does not establish cross-tokenizer learned transfer. |
| How much optimizer storage can factorization save? | [Adafactor](https://arxiv.org/abs/1804.04235): row/column second-moment factors reduce state. | Declare `optimizer.name: adafactor` versus `adamw`; [training](docs/training.md), [accounting](docs/memory.md#adafactor-state-accounting) | Learning-rate semantics differ. State-byte savings are not total-memory savings or equal-quality evidence. |

## Run through the lab

Start with the [installation and first run](README.md#first-run), then prepare a
complete current local resource closure. Replace the path and `neutral-study`
placeholder with your own resource root and recipe-backed entry ID. This command
only scaffolds declarations; it does not train or authorize execution.

```sh
klab research scaffold neutral-study --resource-root /path/to/resources \
  --scale smoke --data offline --backend cpu --output experiments/new-study
klab study plan experiments/new-study/study.yaml
```

Follow the generated README for tokenizer preparation, actual-config inspection,
staging, submission and collection. See [research execution](docs/research/README.md)
for the catalog and [training programs](docs/experiment-programs.md) for native
input binding, immutable locks and checkpoint chains. A smoke scale checks wiring;
use a separately declared scientific protocol for a quality claim.

Before comparing, hold data, tokenizer, effective batch, token exposure, seed set,
optimizer and evaluation fixed except for declared variables. Record unavoidable
parameter or compute differences. Measure held-out loss and task behavior
separately from peak memory, update time, full-run time and checkpoint cost.
Retain negative, interrupted and unavailable results.

## What the project has learned

The [experiment ledger](docs/research/experiment-ledger.md) links questions to
retained evidence, limits and possible next tests. In particular:

- Lookup/FFN comparisons have mixed or negative same-width results; they have
  not established a general Engram advantage.
- Synthetic context and domain adaptation studies show why acquisition,
  held-out behavior and retention need separate controls.
- Dense-model exposure and size comparisons lower held-out loss under their
  recorded conditions, while fixed-prompt generation remains mixed.
- Runtime and optimizer-state checks establish bounded execution/accounting,
  not model quality or a paper reproduction.

The packaged lifecycle registry is empty and grants no scientific clearance.
Use an explicit current registry for declared lifecycle state,
and `campaign status` for a local execution. Checked-in findings cannot establish
which jobs are currently running on another machine. Scientific next questions
live in the [research roadmap](docs/research/roadmap.md).

## Mechanisms that need extensions

The [detailed paper catalog](docs/research/paper-mechanisms.md) retains the paper
contributions, controls, primary sources and limitations without ranking them.
Choose an extension only when a declared question needs it:

| Direction | Papers / mechanism | Missing lab support |
| --- | --- | --- |
| Conditional adaptation and compiled memory | When to Adapt, Memory Grafting, SCONE, Larimar, User as Engram | Occupancy-aware adapters, donor compilation, exact suffix paths or editable episodic memory, depending on the method. |
| Recurrent and hybrid sequence models | Gated DeltaNet-2, Qwen3.8-Next, Mamba-3, alternating sparse/latent attention | New mixers, layer schedules and method-specific kernels. |
| Learned sparse selection | Full NSA, DeepSeek Sparse Attention | Three-path ablations or a learned indexer and its training protocol. |
| Tools and structured reasoning | ToolkenGPT, PICARD, Octopus, ReaRev | Tool-token training/execution, tokenizer-aware parser constraints or graph reasoning; inert tool transcripts do not provide these. |
| Inference storage | Fiddler, LLM in a flash, KVSwap | Expert placement, flash weight loading or disk KV cache; activation offload moves different state. |
| Optimizer and training memory | 8-bit Adam, Adam-mini, GaLore, APOLLO, Lion, Muon, SCALE, COAT, ZeRO family | Configuration, state accounting, safe checkpoint/resume and runtime integration beyond shipped AdamW/Adafactor. |

A maintained upstream implementation or a published device result is motivation,
not local acceptance. Report missing software through the [contribution workflow](CONTRIBUTING.md#planning-and-contributions), and register
new experiments through the [research contribution workflow](docs/research/contributing.md).
For foundational architecture and dataset references, see [references](docs/references.md).
