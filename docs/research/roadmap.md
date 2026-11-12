# Capability status and evidence roadmap

This roadmap describes what Model Lab can execute, what has been smoke-tested, what outcomes have been observed, and what still blocks stronger claims. Progress is organized by capability and evidence—not release phases. **Implemented** means the path exists; **smoke-tested** means it ran; neither means a model is useful or an architecture is better. Missing code belongs in [GitHub issues](https://github.com/supernovae/model-lab/issues); scientific questions and next experiments remain here and in the versioned lifecycle.

Durable experiment definitions use the [`experiments/`](../../experiments/)
layout: copyable teaching material under `samples/`, actual campaigns under
`research/`, and mutable execution output under the external persistent
`$KLAB_WORK_DIR/experiments/` tree. GitHub's Code task and Research experiment
issue templates preserve the same boundary.

The [experiment ledger](experiment-ledger.md) is the compact index of retained
lessons. This roadmap owns open questions and their scientific gates; detailed
metrics and historical source/checkpoint identities remain in linked records.

## Decision coordination

The lifecycle sidecar now records bounded Findings, declared next tests, reviewed
promotion decisions, and known-good baseline availability without treating a
metric as an automatic scientific decision. The first dense synthetic alias
reference remains an unpromoted candidate: its one-run capture reached the budget
and acquisition gates, but held-out validation loss rose from 6.176402 to 7.056087.
The [known-good baseline guide](known-good-baselines.md) records the evidence and
the failed gate alongside the bounded offline workflow, run on CPU, and independent branches.
It does not close useful-model, semantic text-query/compiler, full allocation,
independent lexical generalization, representation portability, CUDA/XPU, or
cross-host evidence.

## Capabilities available now

| Capability | Current boundary |
|---|---|
| Model workflow | Train, inspect, evaluate, checkpoint, resume/promote, generate, and chat on documented CPU, MPS, and MLX paths. Worker-based independent runs are supported; no distributed backward or expert sharding. |
| Learn and research | Packaged mechanism lessons, offline/TinyStories/FineWeb-Edu profiles, controlled recipes, explicit scaffolds, capability cards, paired/factorial comparisons, static reports, and read-only research browsing. Preparation, training, and collection remain explicit commands. |
| Architecture | PyTorch dense, sliding-window, block-sparse and MLA attention; local Top-K MoE; token/byte/portable Engram. MLX supports dense and native block-sparse attention, not every PyTorch mechanism. |
| Lexical memory | Token- and byte-addressed trainable tables, placement options, diagnostics, and checkpoint integration. Addressing/reuse/collisions are observable mechanisms, not proof of useful retrieval. |
| Portable memory | Verified byte-table export/load and a frozen table with a trainable target adapter. This is distinct from a semantic EngramPack and from a text encoder. |
| Semantic EngramPack | Verified exact retrieval from caller-supplied vectors, structured toy-world controls, and direct PyTorch `DenseLM` adapters. The standard text chat/training path does not construct semantic queries. |
| Useful model behavior | No independently validated useful domain model yet. Synthetic association and instruction runs provide learning/failure examples, not general assistant capability. |

## Smoke and experiment evidence

- The [FFN-width and lookup smoke and nano reports](sample-report.md) completed 18 runs and 21 comparisons per campaign, using an offline fixture and PyTorch on CPU. All three alias-card scores were zero for every run; lexical memory showed no consistent held-out-loss benefit. The fixed-seed rerun reproduced all 18 losses, but adds no independent seed evidence.
- The [FineWeb-Edu micro study](development-evidence.md#completed-fineweb-edu-micro-study) completed 18 MPS runs at 1,024 updates / 262,144 targets. Narrower FFNs had higher held-out loss; lexical deltas changed by seed/width, and all three alias cards were zero. This is bounded, descriptive evidence, not a scaling law or a general quality result.
- The [MLA analysis smoke](sample-report.md#research-analysis-pipeline-smoke) completed 12 CPU/offline runs and exercised factorial, nondominance, allocation, and boundary-sweep reporting. Its 36 card outcomes were measured zeros; the run verifies analysis paths, not an MLA/memory benefit.
- The [byte-Engram smoke](../byte-addressing.md#verified-smoke-execution) verified nontrivial UTF-8 addresses and two training updates. It does not establish learned byte-memory value.
- The [portable Engram comparison](../portable-engram.md) tested two held-out cases; baseline, random-table, and trained-adapter systems all had zero exact matches, and the trained adapter did not outperform the random control. General transfer remains unverified.
- The [ownership-allocation smoke](../history/path-domain-corpus.md#partial-cpu-allocation-smoke-observation) executed one of 75 configured coordinates. It produced zero card passes at every recorded checkpoint; no allocation optimum or full curve is established.
- The [instruction starter](../from-toy-to-useful.md#measured-starter-example-what-improved-what-did-not) lowered synthetic validation loss, but its actual replies still failed ordinary arithmetic, color, and unrelated questions. This is not a useful assistant.

## Open research questions

These are evidence gaps and proposed investigations, not implementation TODOs.
If an experiment exposes missing or defective software, open a separate Code task
under the [current implementation tracking policy](../../ROADMAP.md#where-work-is-tracked). The main open
questions are:

- **Useful tasks:** establish one low-risk real task against an independent test population and a simple baseline, with declared error handling and human review.
- **Corpus shape and synthetic fraction:** preregister paired releases from the same licensed source snapshots with declared shape/fraction filters, held-out source/world/template units, fixed tokenizer/target token budget and controlled model/seed/evaluation suites. Compare recall, paraphrase, troubleshooting, tool use and abstention separately; report negative and mixed effects rather than a universal synthetic percentage or shape ranking. The [Corpus Forge ledger](../corpus-provenance.md) provides measurement inputs, not an observed outcome.
- **Historical DevMind work:** source expansion, failed recovery and the unpromoted MODEL-0 endpoint are archived in the [research history](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/README.md). The [lessons](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/LESSONS_LEARNED.md#devmind-source-supply-and-recovery) retain unresolved quality, rights, exposure and reproducibility limits. These archived successor proposals are not the current execution plan.
- **Lexical Engram:** test transfer/generalization across independent task data, held-out wording/facts, seeds, and collision/capacity controls; keep token and byte results distinct.
- **Portable byte Engram:** extend the negative two-case result to multiple recipient configurations and held-out cases; prove adapter updates leave source table bytes unchanged and retain disabled/random/frozen-only controls.
- **Semantic EngramPack:** distinguish exact retrieval of supplied vectors from natural-language understanding. Any text-query capability first needs a reproducible encoder/space contract, an exercised query path, leakage-audited tasks, pack controls, and measured encoder/retrieval cost.
- **Other mechanisms and allocation:** run selected full comparisons beyond wiring smokes, preserve every seed/outcome, and separate task quality from parameter/cache estimates and synchronized device timing.
- **Learning and cost evidence:** add a versioned observation protocol for held-out outcomes, actual token/checkpoint boundaries, threshold censoring, wall/device time, and memory. Update duration is not end-to-end experiment time.
- **Hardware:** CPU/Apple evidence cannot close CUDA/HIP/ROCm/XPU or real cross-host execution gates.

## Archived dense and TinyStories findings

The completed token-budget, model-scale, decoding and data-rich TinyStories
studies now live in [research history](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/README.md).
Their [source-linked lessons](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/LESSONS_LEARNED.md)
separate lower held-out loss from mixed prose behavior, repetition from coherence,
and changed exposure from new source supply. Retained reports and identities are
unchanged; old executable study drivers have been removed. An archive is not a
current launch packet or baseline-promotion decision.

Any future continuation needs a new scientific question, explicit controls,
independent review population and separately authorized execution. Opened prompt
sets and historical source revisions remain historical evidence.

## Deferred capability: teacher-derived semantic representations

**Status: not implemented.** Verified semantic retrieval consumes already encoded canonical vectors; it does not encode prompt text. Current toy worlds use structured keys, not natural-language embeddings. Standard training and generation do not construct semantic query batches. These interfaces and tests establish bounded retrieval mechanics only.

The existing `klab engram pack compile` command packages caller-supplied vectors. It does not load a teacher, extract hidden states, choose layers, pool source text, or account for teacher compute. Do not describe manual vectors, record compilation, or the structured toy encoder as teacher knowledge distillation.

A teacher-derived pack remains gated until all of the following are exercised:

1. A frozen, locally available teacher with immutable revision/checkpoint identity; no implicit download or remote code execution.
2. Reproducible query and value representations with declared encoder identities and one explicit compatible feature-space contract.
3. One verified pack used by at least two recipient configurations, with unchanged pack bytes and separate recipient adapters.
4. Held-out evaluation against correct, disabled, random, conflicting, and incomplete pack controls; compiler inputs exclude validation, test, and capability-card content.
5. Source/license/provenance plus measured teacher and compiler costs before any transfer or amortization claim.

Until those conditions hold, do not add an implicit `teacher compile` path, call the synthetic key encoder a text encoder, or describe current retrieval as natural-language understanding.

## Required contract for a future teacher-pack compiler

When the gate is met, compilation is an explicit offline command over already-authorized local assets. Its output is a versioned, verified semantic EngramPack; pack creation never runs implicitly during scaffold, training, evaluation, or dashboard browsing. Any new manifest fields require a versioned schema and must preserve compatibility with existing records-only, lexical, and externally supplied semantic packs.

A compiled pack and its sidecar provenance must bind at least:

- teacher model identifier, immutable revision, checkpoint/config digest, tokenizer digest, architecture, license, and local source identity;
- extraction layer(s), hidden-state selection, pooling/window policy, text template, truncation limits, input-token accounting, dtype, and normalization;
- independent key and value encoder identities, widths, representation-space identity, row/record mapping, and compiler algorithm/version/config/source digest;
- source-record identities, split ownership, license/attribution, skipped/invalid counts, and exact content hashes;
- teacher forward tokens, compiler wall/device time, peak memory, output bytes, and any cached intermediates.

The pack stores representations and provenance, not teacher weights or a promise that the teacher's knowledge is true. A stable hash proves artifact identity, not encoder quality, factual correctness, license sufficiency, retrieval quality, or recipient compatibility. Never infer compatibility from equal vector widths alone.

## Comparisons and cost accounting

A later controlled comparison should hold source split, tokenizer, teacher checkpoint, student initialization, optimizer, student target-token budget, and evaluator fixed. Keep these arms distinct:

1. **Student baseline:** student trained without teacher targets or a semantic pack.
2. **Traditional distillation:** frozen teacher supplies the declared targets; student has no pack.
3. **Teacher + pack / student + same pack:** the same frozen teacher and train-only source produce a verified pack; the student receives that exact pack under the declared training/evaluation protocol.
4. **Frozen-recipient transfer:** freeze each recipient and its adapter, then attach the same verified pack without updating recipient weights.

Report task-level language-model, factual/semantic, reasoning, and unseen-pack results separately. Preserve negative, unknown, conflicting, incomplete, and unavailable outcomes; do not replace them with a pooled quality score. A recipient using a new adapter is a separate trained artifact, not zero-cost transfer.

Account separately for teacher source/input tokens, teacher output/distillation tokens, pack-compiler tokens and forwards, compiler wall time and peak memory, pack storage, student training tokens and targets, student training time, query-encoder and retrieval latency, and recipient adaptation. Record hardware/backend and measurement method. Any amortized figure names the number of recipients and includes pack creation and storage; compiler cost is never treated as free or hidden. Token counts are not interchangeable with FLOPs, energy, wall time, or quality.

## Teaching shapes and boundaries

The intended walkthrough makes the feature path explicit:

`source text → frozen teacher hidden states [B,T,D_T] → declared pooling/window rule → keys [M,K] and values [M,V] → encoded query [B,K] or [B,T,K] → retrieved values [B,T,V] → recipient projection/gate → residual [B,T,D_S]`.

`D_T`, key width `K`, value width `V`, and recipient width `D_S` are independent. The teacher is an offline producer; the pack is frozen; the recipient adapter is trainable only where the experiment says so. Explain which components run at pack-build time, student-training time, and inference time. Do not imply the teacher is absent at query time unless the query path actually avoids loading it.

Current source boundaries: `SemanticQueryBatch` in `src/klab/engram/semantic.py` accepts pre-encoded vectors; `SemanticRetriever` verifies and retrieves from pack assets; `SemanticMemoryAdapter` applies the retrieved values. `src/klab/data/toy_worlds.py` encodes canonical structured keys for deterministic synthetic controls. These are useful interfaces and test fixtures, not a macro-model compiler. See [the current runtime contract](semantic-memory.md) for exact limits and testable behavior.

## Historical research: Kernel Memory Lab

[Kernel Memory Lab](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/README.md) was
retired from the active showcase on 2026-10-18. Its retained
[status](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/STATUS.md) and
[bootstrap](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/BOOTSTRAP.md) describe
historical cards, stopped attempts and scope, not a current release assignment.
Fresh rerun ideas belong to the coordinated backlog with new identities and
current frameworks. The old project is not awaiting qualification as an active
showcase. Historical allocations and archived models authorize no new run;
existing native receipts, accounting and runtime holds remain unchanged.
