# Learn one mechanism

Browse lesson metadata without a campaign or run directory:

```sh
klab learn list
klab learn describe mla
klab learn scaffold mla --scale nano --data offline --backend cpu --output experiments/learn-mla
klab tokenizer train experiments/learn-mla/tokenizer.yaml
klab inspect experiments/learn-mla/model.yaml
klab learn probe experiments/learn-mla/model.yaml --prompt "Once upon a time, a small bird found a key."
```

A probe uses a fresh CPU FP32 initialized model and the selected tokenizer. It reports hook shapes and configured mechanism diagnostics; it is initialized-mechanism evidence, not training or evaluation evidence. Training additionally requires explicit `klab data prepare experiments/learn-mla/model.yaml` and `klab train experiments/learn-mla/model.yaml --run-id learn-mla --stop-after-step 2`.

## Lesson IDs

| ID | Standalone configuration | What to inspect |
| --- | --- | --- |
| `dense` | Dense, no-memory baseline | `[B,T]` IDs to `[B,T,D]` states to `[B,T,V]` logits; causal RoPE attention. |
| `ffn` | Dense decoder with SwiGLU width `2D` | Gate/up `[B,T,F]` projections and down projection back to `D`. |
| `sliding-window` | Causal window 16 | Local visibility; it emits no sparse-selection metric. |
| `mla` | Latent width `D/2` | Latent states, expanded keys, latent values, and actual cache shapes. |
| `sparse-attention` | Blocks of 16, select 2 | Available/selected tokens, ratio, work estimate, and selected positions. |
| `moe` | Four local experts, top-2, auxiliary coefficient `.01` | Router entropy, maximum expert fraction, mean top-k probability, selected experts/weights. |
| `lexical-engram` | Trainable token n-gram table, final injection | Addresses, lookup/reuse/collision, gate, norm, and placement diagnostics. |
| `byte-engram` | Trainable byte-address table, final injection | UTF-8-derived addresses; byte and token suffixes can differ. |
| `portable-engram` | Verified frozen byte package plus trainable adapter | Package-derived table/order/width, frozen values, adapter/gate diagnostics. |
| `engrampack` | Artifact-only tutorial records | Three CC0 provenance-marked triples; no model config or executable retrieval claim. |

`portable-engram` requires `--memory-package PATH`; the scaffold verifies and copies the supplied package rather than inventing dimensions. Create one through the existing byte-memory path and `klab engram export`, then scaffold it. `engrampack` instead creates records for `klab engram pack compile`, `inspect`, and `verify`; a valid records pack does not imply model retrieval or training.

For each lesson, predict a shape or behavior, change one setting, and explain any mismatch. Changing configuration can be invalid—for example, MLA latent width must divide across heads. A trained diagnostic belongs to the ordinary train/dashboard path, not to an initialized probe.
