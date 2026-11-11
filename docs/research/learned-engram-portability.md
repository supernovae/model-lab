# Learned Engram portability v1 — Experiment A

**Status:** the source-learned portability protocol. Its commands may execute only after the campaign build has materialized and sealed the data, resource selection, configurations, and coordinate plan. This is not the historical compiled-world runner and does not compile facts into the source table. Negative source or transfer outcomes are valid. If any source gate fails, no table is exported and every dependent recipient coordinate is blocked.

## Question and falsifiable hypothesis

Can a byte-addressed N-gram table learned jointly with a source DenseLM transfer held-out fact associations to independent width-64 and width-128 recipients using only recipient-local projection/gate calibration?

The hypothesis is that correctly associated source-learned rows improve recipient-held-out accuracy over **each** matched control at both widths across all three seeds (17, 41, 73). Controls, thresholds, fixed endpoints, and source gates are sealed before behavioral outcomes are inspected. The supportive adapter result additionally requires final held-out accuracy ≥0.50 and ≥0.10 advantage over baseline, constant, random, and permuted controls in all six width/seed groups. Report every score regardless of decision.

This synthetic experiment tests source acquisition and interface portability, not general language competence, cross-architecture transfer, or semantic representation transfer. Width changes retain DenseLM architecture family.

## Data and address contract

The materializer creates one deterministic dataset shared by all model seeds. Candidate fact counts are 512, 1,024, and 2,048; 128-fact smoke is wiring-only and cannot satisfy behavioral acceptance. Facts receive balanced one-character answers from `ABCDEFGHIJKLMNOPQRSTUVWXYZ234567`. Keys use fixed ASCII syntax `k` plus 12 lowercase hexadecimal digits, a dot, and four nonce digits. The last 32 bytes of the prompt suffix are exactly `b"|" + key + b"\n\nAssistant: "`. The scored prefix is `format_chat_prompt([], message) + " "`; the address is the byte-hash N-gram immediately before the answer symbol, not the colon-only position or the position after the symbol.

Raw UTF-8 uses `poly257-terminal-v1`, order 32, 65,521 rows, and E=32 FP32. Facts are assigned disjoint calibration (one quarter), source-monitor (one quarter), and recipient-held-out (one half) ownership. Source SGD/native acquisition sees all N source facts; adapter SGD sees calibration only. Source-monitor facts gate memory-sensitive source behavior using unseen wording; they are not recipient calibration or the primary transfer denominator. “Held out” means held from recipient supervision, not unknown to the trained source.

Preparation uses 512 unrelated copy examples (each prompt names the symbol to copy) and 128 disjoint validation examples; these never contain transferred key→answer associations. Every run uses assistant-only version-2 conversations and existing packing. Training blocks contain 127 tokens plus EOS, exactly 128; only answer-space/symbol positions are supervised. A generic sentinel retains the last real block. The tokenizer has exactly 260 ByteLevel tokens and no merges. Data publication records complete file hashes, fact/block ownership, template identities, addresses, collision/occupancy diagnostics, and role permissions. Address-stream overlap with calibration/preparation is rejected before training.

Source templates: `Lookup the stored symbol.` and `Recall the stored symbol.` Adapter templates: `Return the stored symbol.` and `Give the stored symbol.` Source-monitor wording: `Which symbol is assigned?`. Final unseen wording: `Report the assigned symbol.` and `State the assigned symbol.` Primary transfer curves use `Return the stored symbol.` Query records contain no expected answers; scorer labels remain in separately hashed scorer/result files.

### Run-owned v2 manifest

Every coordinate owns a canonical, SHA-256-bound `klab-portability-run` version-2 manifest. Its exact top-level fields are `format`, `version`, `experiment`, `protocol`, `world_manifest`, `coordinate`, `seed`, `initialization`, `initial_backbone`, `memory`, `source`, `observations`, `training_fact_ids`, and `sha256`. The external coordinate contains only `role`, `recipient`, `representation`, and `condition`; the paired seed remains a separate field. Observation descriptors bind the schedule, query records, scorer, and ownership map.

Initialization uses `init-v1` with explicit family, width, pair seed, purpose, and canonical tensor name. Source real/dense runs share the `source-backbone` initialization; source memory, preparation backbones, recipient adapters, and native memory use their separately named families. Preparation and native runs bind the matching preparation checkpoint. Transferred recipients bind the source run, checkpoint, original table digest, provenance, and passed source gate, but never bind source weights.

The normalized memory object has exactly `kind`, `artifact`, `pack_id`, `tensor_sha256`, `addressing`, `encoder_contract`, and `replacements`. Its kind is `byte` even when a control model has no configured memory; `artifact` is absent or names the copied learned package, and the byte encoder contract is null. Timing-only manifests use role `calibration` and purpose `timing`; their six fixed pilots and costs are excluded from scientific outcomes.

## Models, updates, and source gate

Source DenseLM: D=128, L=2, H=4, FFN=512. Recipients: D=64/L=2/H=4/FFN=256 and D=128/L=2/H=4/FFN=512. All use vocabulary 260, context 128, dense attention, tied embeddings, final memory injection, byte memory E=32/table=65,521/order=32. The source-real arm starts with a random trainable table and jointly trains all assigned DenseLM and table parameters. Source-dense starts from exactly the same nonmemory initialization and trains without memory. No answer-derived initialization or direct table compilation is permitted.

All learned training roles use deterministic FP32 AdamW: peak LR 0.001, floor 0.0001, betas (0.9, 0.95), eps 1e-8, weight decay 0, clipping 1.0, batch 8, accumulation 1, sequence length 128, warmup `min(64, max_steps // 8)`. Source/native present each fact 128 times (64 epochs over two templates); adapters present calibration facts 128 times; preparation trains 64 epochs on 512 examples. A single prespecified half-budget policy is available only if resource projections rule out full policy: 64 source/native presentations per fact, 64 calibration presentations, 32 preparation epochs. No smaller behavioral budget is valid.

Each source seed must satisfy every gate at the fixed source endpoint: full fact exposure and valid asset/update audit; finite tensors and nonzero table delta; at least 90% of factual answer rows have observed nonzero SGD gradients and changed final bytes; source-monitor unseen-wording accuracy ≥0.75; enabled-minus-disabled monitor accuracy ≥0.10; and disabled-minus-enabled answer NLL ≥0.10 nats on the same cases. The ablation compares the same final checkpoint with memory enabled, disabled, and the initial random table. All three seeds must pass before any export or recipient training. A failure publishes `source_signal_not_established`, all three source outcomes, and blocked recipient rows; no threshold tuning or favorable seed selection.

Gradient evidence records actual nonzero gathered table-row gradients at each applied update, before clipping, plus compact row indices and scalar norm/count. Committed checkpoint audits distinguish gradient-observed rows from rows whose bytes changed, and include factual rows, row-delta norms, address frequency, and fact exposure. Resume evidence is aggregated from selected immutable lineage events, never an in-memory union.

## Recipient matrix and outcomes

After the source gate, independently prepare each width/seed model on copy examples. Validation must reach ≥0.95 total exact-symbol accuracy and ≥0.75 for each symbol. Preparation failure blocks dependent conditions. For each width and pair seed use the same preparation checkpoint:

- `baseline`: preparation checkpoint, no memory or extra updates.
- `constant`: identical nonzero vector in every row; freeze it and train only output projection/gate.
- `random`: frozen independently sampled Gaussian rows globally scaled to real factual-row RMS; no per-row norm or answer matching.
- `permuted`: deterministic nonidentity derangement of learned vectors among factual rows within each ownership partition; preserve value multiset and record incidental label coincidences.
- `real-zero-shot`: real table, fresh recipient-local interface, no updates.
- `real-adapter`: freeze preparation backbone and table; train only `memory.output.weight` and `memory.gate.weight` on calibration facts.
- `native`: initialize recipient-local memory and jointly train backbone/table/interface on all source facts; descriptive acquisition reference, not matched adaptation.

The 42 planned trained runs are six source, six preparation, 24 adapter-control, and six native runs. Baseline and zero-shot are observations, not fake training arms. Same-seed source package bytes are used at both recipient widths. Adapter parameter counts are (E+1)D: 2,112 at D=64 and 4,224 at D=128. No source backbone, optimizer, hidden state, or source adapter is transferred.

Adapter observation boundaries are deduplicated `[0, 8, 32, 128, 512, 2048, 8192, endpoint]` within the sealed budget. Source/native use `[0, 32, 128, 512, 2048, 8192, endpoint]`; preparation uses `[0, 128, 512, 2048, endpoint]`. Curves separate calibration and held-out accuracy/NLL. Held-out is never used for selection. Final wording is scored only at fixed endpoints, including archived baseline/zero-shot states. Full-vocabulary argmax is used; invalid/non-symbol output is incorrect. A descriptive 0.75 threshold reports first observed boundary and prior interval or right-censoring; it is distinct from execution status and the historical compiled runner's 0.95 threshold.

## Resource policy and bounded execution

The behavioral ceiling is 43,200 seconds including build/materialization and timing calibration. Timing calibration is at most 1,800 seconds, six 16-update role/width pilots per available backend, with no pilot outcomes entering science. Measure complete data preparation and evaluation at each candidate N. Project all updates, setup, checkpoint/audit, observations, export, and report; apply a 25% margin and require `1.25 × projected remaining work ≤ 0.80 × remaining allowance`. Memory admission applies a 25% margin to the larger of measured process RSS or driver allocation plus candidate-inventory growth beyond N=512, then compares it with measured available host RAM. Disk admission applies the same margin to owned data copies, measured checkpoint generations, projected audit histories, and portable memory packages, then compares it with free space on the campaign volume. These projections use actual candidate inventories and pilot artifacts; resource capacity is recorded in the immutable selection. Select full policy and largest feasible N first; use half policy only if no full policy fits. Backend choice uses complete timings only, with CPU on a tie and no implicit operator fallback. If half-policy N=512 cannot fit, publish a resource-blocked design and do not substitute smoke.

Learned coordinates save full-state checkpoints at registered observation boundaries, plus at most one terminal/interruption checkpoint. Those generations remain available for post-training per-case scoring. Built-in validation runs initially and at the final endpoint; disk projection counts the registered checkpoint boundaries plus one extra generation per coordinate.

`build` and `plan` do not train. Learned `build` accepts smoke/nano and auto/cpu/mps; it creates candidate data inventories and immutable design inputs. `plan` is read-only and reports uncalibrated timing until real pilots exist. Run/continue execute the fixed dependency graph under the remaining sealed allowance; a cooperative deadline may overrun only by a current update/bounded item and safe checkpoint/receipt publication. Continue uses verified new child run IDs, restores checkpoint-owned asset paths while rejecting changed scientific settings, and never resets optimizer state or increases total allowance. Report is read-only and does not infer or train.

```sh
uv run klab research describe learned-engram-portability-v1 --json
uv run klab research portability build --experiment learned-engram-portability-v1 --scale nano --backend auto --output artifacts/learned-engram-portability-v1/behavioral
uv run klab research portability plan --campaign-root artifacts/learned-engram-portability-v1/behavioral
uv run klab research portability run --campaign-root artifacts/learned-engram-portability-v1/behavioral --max-wall-seconds 43200
uv run klab research portability report --campaign-root artifacts/learned-engram-portability-v1/behavioral
```

Use `continue --campaign-root ROOT --max-wall-seconds SECONDS` only with the emitted remaining allowance. The separately labeled wiring smoke uses 128 facts and its fixed small budgets; it cannot close behavioral criteria or select behavior settings. Existing artifact roots fail closed on conflicting contents.

## Evidence and claim boundaries

Protocol, data manifest, execution receipt, source gate/export provenance, native run manifests/checkpoints, observation envelopes, per-case scorer results, and evidence are canonical and hash-bound. Each seed/role/condition coordinate has immutable attempts; partial, blocked, failed, interrupted, and completed states remain distinct. The report bundles verified original inputs/results and re-rendering is read-only. It reports three separate conclusions: **artifact portability** (same bytes verified/loadable), **adapter portability** (the six-group held-out contrasts), and **representation portability** (not tested).

Report source-monitor and source-task scores, preparation gates, per-width/per-seed contrasts against every control, costs, thresholds/censoring, and integrity failures. Do not pool calibration into held-out scores, call held-out facts unseen by the source, or interpret synthetic symbols as ordinary-language quality. Optional source-B A→B→A is deferred until a positive primary result and a separately approved budget.

The version-2 static bundle renders fixed SVG curves for source-monitor, preparation validation, recipient calibration, final-report, and final-state observations. It includes the canonical evidence, protocol, world/data inventory, observation results, execution receipts, source/package provenance, run manifests, and checkpoint evidence; loading verifies the copied inventory without training or inference. Version-1 and no-evidence reports continue through the legacy renderer and serialization path.

## N=2048 nano execution record — 2026-09-26

`/tmp/omp-learned-portability-nano-final5.rTighq` sealed the full CPU policy at N=2,048 and completed 54/54 coordinates (42 trained, 12 observation-only), with no failed or blocked outcomes, in 9,583.7 seconds. All 42 run manifests and 282 checkpoint manifests share source identity `3953d5464572ec5c03d9852f756fc0ee00c8cb9446507f74cd5a61d5bd54d229`, matching the current 151-file package identity. Evidence envelope `5f0b3b678a38db7a1a1f9c7d0fefc79b73842e3a6c215a8dce6fda4c410fa1ae` has raw SHA-256 `fdbbbf354785adfadb3efb4a93e36763555f1744bb439b3258858030fd31be7b`. Two report renders produced bundle `b094e9791fe9c0eb0a635388e52d00782c7317abadad33f9e6b9c4f3e3670bed`; all 1,066 bundle assets matched their recorded hashes.

All three source-real seeds passed the gate: 32,768 updates each; 2,048/2,048 facts each received 128 target exposures (262,144 total), all factual rows had observed nonzero gradients and changed final bytes, and source-monitor accuracy was 512/512. Enabled-minus-disabled accuracy was +0.96875; disabled-minus-enabled answer NLL was 11.713/9.448/12.284 nats for seeds 17/41/73. Each exported table exactly matched the final source checkpoint and both width-64/128 recipient table tensors (nine exact comparisons). All six preparations scored 128/128. Every real adapter updated only `memory.gate.weight` and `memory.output.weight`; backbone, frozen parameters, and assets remained unchanged.

The matched source-dense monitor controls scored 17.19%, 33.59%, and 13.87% for seeds 17, 41, and 73, respectively.

On recipient-held-out facts, baseline, constant, random, permuted, and real-zero-shot controls all scored 3.125% exact accuracy on both final templates. Real adapters averaged 71.70% (answer NLL 1.833) on “Report the assigned symbol,” versus 3.125% (NLL 17.528) for zero-shot; mean gain was 68.57 percentage points. Five of six groups reached 50%, but only three crossed the descriptive 75% threshold, all at step 8,192. On the separate “State the assigned symbol” wording, all six adapters remained at 3.125% accuracy (NLL 12.424 versus 18.044 zero-shot). Native recipients averaged 99.97%/91.76% on report/state and crossed 75% in all groups at 8,192/32,768 updates. Width-64/128 adapter means were 61.88%/81.51% on the report wording, but the three-seed width gap is descriptive only.

Per-seed real-adapter exact accuracy on the final held-out wordings:

| Seed | Width 64 `final_report` | Width 128 `final_report` | `final_state`, both widths |
|---:|---:|---:|---:|
| 17 | 85.45% | 85.06% | 3.125% |
| 41 | 51.07% | 68.26% | 3.125% |
| 73 | 49.12% | 91.21% | 3.125% |


The preregistered supportive adapter criterion (≥0.50 held-out accuracy in all six groups) failed narrowly at width64/seed73 (0.4912); no exact-accuracy gain generalized to the second final wording. This supports source acquisition and byte-exact artifact reuse plus recipient-local calibration on one held-out wording, not robust wording transfer, general language competence, or representation portability. The static evidence report is at `/tmp/omp-learned-portability-nano-final5.rTighq/reports/b094e9791fe9c0eb0a635388e52d00782c7317abadad33f9e6b9c4f3e3670bed/index.html`.


Historical boundary: the prior compiled-world matrix and bounded MiniLM pilot remain as recorded evidence with their original identities/thresholds/scores. The old evaluator started from a colon-only prefix while training appends a separator space before the answer; that alignment limitation is documented prospectively and does not authorize rewriting or rescoring those artifacts.
