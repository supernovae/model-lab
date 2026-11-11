# Repository review: from mechanism demos to measured small models

This page preserves the original review and exploratory results; they are not an untouched-test claim. The completed follow-up section below records newer runtime, worker, context/Engram, and domain-adaptation work. Use the [README](../README.md) and [research roadmap](research/roadmap.md) for current capabilities and remaining scientific questions; the [feature roadmap](../ROADMAP.md) summarizes pending implementation directions.

## Project contract

A useful experiment trains an actual checkpoint, exercises it through chat, measures a declared task, compares controlled alternatives, and preserves the conditions needed to repeat or falsify the result. "Useful" starts with a narrow, reliable learned behavior; it does not require mimicking a commercial assistant. The same contract should survive increased model size, data, and training budgets within the supported single-host runtime.

## Gaps closed by this review

| Gap | Resolution and evidence boundary |
|---|---|
| Chat cropped arbitrary tokens and depended on external tokenizers | Completion-reserved whole-turn context management; verified checkpoint selection; run-owned tokenizer/package; interactive reset, seeded sampling, JSON responses and saved transcripts. |
| User chat data required new Python sources | Strict licensed local JSONL conversation ingestion using the same chat formatter; byte-content cache identity and exact train/validation conversation-overlap rejection. |
| First recall card confused record IDs with generalization | Historical card labeled diagnostic. Separate training-seen retention, test-only held-out wording, and context-override cards. |
| No task acquisition control | Step-zero checkpoints and a training-seen retention card distinguish inability to learn from failure under new wording. |
| Scores could outlive or overwrite their checkpoint identity | Frozen generation and content-addressed evidence, full prompts/responses, generation protocol, trained budget, source identities and parameter counts. |
| Configuration equality hid unfair comparisons | Compare actual steps/tokens, seed, content digests and runtimes; explicit memory/attention/FFN/scale axis; exact changed fields and paired gains/losses. |
| Validation/evidence paths accepted incomplete or altered observations | Count-weighted loss, restored evaluation mode/RNG, finite checks, artifact/report/checkpoint binding, explicit rejected and missing reports. |
| Resume/promotion could misrepresent scientific settings or data | Compatibility checks remain mandatory; recovery uses the same checks; promotion uses destination data and compatible architecture/tokenizer semantics. |
| Partial token budgets and performance metrics were misleading | Exact valid-target masking, skipping empty microbatches, synchronized update timing; historical throughput claims withdrawn. |
| Architecture accounting/diagnostics were incomplete | Corrected MLA and multi-table parameter estimates, actual Engram storage versus active rows, persisted last-training-microbatch diagnostics, and pre-renormalization router selection mass. |
| Byte-memory Unicode behavior differed between packing and generation | Shared reversible ByteLevel token-byte conversion with causal UTF-8 regression coverage. |
| Runtime capabilities exceeded implemented behavior | Explicit runtime/precision validation, staged pilots, and MLX/target-hardware limitations documented instead of silent substitution or feature-parity claims. |

## Actual learning experiment

Ran the supplied dense and n-gram configurations on CPU with seed 42, a train-only 512-token BPE, two D64 decoder blocks, 256-token maximum context, and 128-token training blocks. Dense has **139,584 parameters**; Engram has **174,368**. This is a matched-backbone comparison, not a matched-total-parameter efficiency experiment.

| Observation | Dense | Engram |
|---|---:|---:|
| Step-zero retention | 0/24 | 0/24 |
| Retention at 1,536 steps | 24/24 | 24/24 |
| Held-out wording at 1,536 steps | 8/12 | 6/12 |
| Context overrides at 1,536 steps | 0/8 | 0/8 |
| Actual target tokens at 1,536 steps | 391,424 | 391,424 |

Run IDs: `review-chat-dense-long` and `review-chat-engram-long`. The configured token ceiling was 393,216; epoch-end short batches made the step limit bind first. Per-case artifacts and comparisons are under each run's `evaluations/`; run artifacts are intentionally not committed as source.

The earlier 384-step exploratory pair is retained as `review-chat-dense` and `review-chat-engram`: held-out wording was 0/12 versus 1/12, and override was 1/8 versus 0/8. Engram's 1/12 matched the constant-answer baseline. We increased a fixed budget without altering the test cases. Because these cases were inspected during development, this is **exploratory engineering evidence**, not a preregistered untouched-test result.

The trained models genuinely acquired the associations and generalized to some new wording. **This experiment does not support an Engram advantage**: dense performed better on the held-out wording card, and both failed the context-control card. It also does not show broadly useful chat, unseen factual knowledge, parameter efficiency, or statistical significance.

### Chat surface proof

The real interactive CLI, not a mocked response path, produced these replies from `review-chat-engram-long` with the trained system prefix:

```text
User: What value belongs to the alias amber?
Assistant: lumen
User: Please recall the value for birch.
Assistant: orbit
/reset
User: What value belongs to the alias coral?
Assistant: quartz
```

A separate local JSONL fixture completed preparation, training, checkpointing, and `checkpointed_held_out` evidence at 128 targets. That is ingestion/lifecycle proof, not a learned domain-quality claim. Tests cover the harder boundaries: overflow, role stopping, RNG isolation, Unicode addressing, shifted token budgets, comparison mismatches, relocated artifacts, tampered checkpoints/caches/reports, and custom-card schema/response budgets.

The interactive transcript is preserved at `runs/review-chat-engram-long/evaluations/interactive-chat.json`. The local-corpus run is retained as `runs/review-local-corpus`; standalone evaluation still succeeds after removing its original corpus, configuration and cache directory, using 224 run-owned validation targets.

### Original review verification

```sh
uv run --locked ruff check src tests
uv run --locked pytest -m "not cuda and not rocm and not xpu and not network"
uv run --locked klab data prepare configs/chat_recall_dense_cpu.yaml
uv run --locked klab eval review-local-corpus
```

The local suite includes available MPS/MLX coverage; it is not CUDA, ROCm or XPU acceptance. All local file links in the 40 Markdown documents were checked. Temporary corpus/configuration/card fixtures were removed after retaining the run-owned evidence.

## Completed follow-ups — 2026-09-22/23

| Original next step | Executed work and conclusion |
|---|---|
| Broaden context overrides and freeze unseen assignments | The [context/Engram study](context-engram-study.md#execution-results--2026-09-22) executed all 24 planned endpoints. Untouched override results remained 0/8 at every endpoint; the failed control is preserved. |
| Test address order, collisions, matched capacity, seeds and budgets | Seeds 17/41/73, two exact target budgets, dense-total alternatives, and address-order/table-size diagnostics have archived observations and paired deltas. The dense-total model is 160 parameters larger; diagnostics remain confounded and do not establish an Engram advantage. |
| Move to a licensed domain corpus and measure retention | The [path-domain study](path-domain-corpus.md#2026-09-22-execution-record) executed three pretraining/adaptation pairs, audited provenance/semantic separation, and retained acquisition/development/frozen-test outcomes. Acquisition improved, held-out reliability remained poor, and static retention worsened sharply. |
| Add missing training/runtime mechanisms | Assistant-only objectives and versioned inert tool-call transcripts, bounded PyTorch KV caches, native MLX sparse components, block recomputation, accumulation, measured activation offload, and PyTorch Adafactor are implemented within their explicit support boundaries. |
| Make experiments reproducible and independently schedulable | Immutable native checkpoints, full-state child resume, fresh-state promotion, isolated pilots, explicit matrices, local/SSH worker contracts, device leases, and verified controller-local ingestion passed the recorded local acceptance gates. |

The [scientific acceptance record](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/scientific_studies_2026_09_22.json) independently verifies preserved inputs, native endpoints, responses and comparisons. The [single-host](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/single_host_gate_2026_09_22.json) and [worker](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/independent_workers_2026_09_23.json) gates cover actual execution/recovery/installation scenarios; the suite at that acceptance revision passed 335 tests. A passing engineering gate does not make a negative learning result positive.

Further curriculum or scale experiments need a new explicit hypothesis and frozen evaluation; inspected cases must not become “untouched” again. Generic statistically justified model selection and open-ended response grading are not established by these studies. Current hardware boundaries are maintained in the [runtime guide](runtime.md);
the historical review is not an up-to-date support matrix. Distributed training
remains outside the current scope.

See [capability workflow](capabilities.md), [chat-oriented data](instruction-training.md), [evidence](evidence.md), the [research roadmap](research/roadmap.md), and the [feature roadmap](../ROADMAP.md).
