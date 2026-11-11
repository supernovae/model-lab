# Engram portability campaign lifecycle

This runner distinguishes implementation from behavior. Its smoke campaign uses generated structured records, direct token/byte table compilation, supplied frozen semantic vectors, 32/64-wide dense-attention `DenseLM` recipients, and seeds 17/41/73. The token and byte tables are **not source-model-trained producer representations**. Semantic query vectors are supplied from the generated structured key; this is **not** a natural-language query encoder. A correct retrieval trace does not repair a wrong recipient prediction.

This guide documents the historical **compiled-world** portability runner: token/byte tables were directly compiled from generated structured facts, while semantic vectors were supplied. It is not source-learned representation transfer. The active Experiment A protocol is documented separately in [Learned Engram portability](learned-engram-portability.md). The staged questions for [compiled knowledge](compiled-knowledge-engram-v1.md) and [compiled initialization](compiled-engram-initialization-v1.md) are documentation only.

The earlier one-token evaluation began at the colon-only assistant prefix, whereas the data conversation adds a separator space before the answer. This is a newly identified alignment limitation for historical scores; preserve the original threshold, identities, artifacts, and scores, and do not rescore them as learned-protocol evidence.

## A — Declare the protocol

Build an immutable campaign before training:

```sh
uv run --locked klab research portability build \
  --output artifacts/engram-portability-smoke \
  --scale smoke --seed 20260925 --updates 4
uv run --locked klab research portability plan \
  --campaign-root artifacts/engram-portability-smoke
```

`plan` is read-only. It reports the 120-arm matrix, sequential CPU memory/checkpoint estimates, architecture dimensions, and runtime calibration only after a completed arm receipt exists. Estimates exclude most Python/runtime overhead and are not performance claims. Do not reuse a campaign root with a changed protocol, source inventory, tokenizer, or generated world.

**Acceptance checkpoint A:** protocol, plan, world manifest, asset manifest, address inventory, development/final observations, and declared limitations are present and hash-bound. Reject an edited or partially replaced artifact.

## B — Inspect the worlds and addresses

The builder separates recipient-preparation conversations, adapter-eligible training facts, development observations, and final replacement worlds. A/B worlds share lookup addresses but assign different values. Token assets pin the complete tokenizer file identity; byte assets pin raw UTF-8 normalization, terminal polynomial hashing, table dimensions, and tensor digest. Semantic assets pin the pack and encoder contract.

Inspect artifact hashes in `assets/memory_assets.json` and exact raw-byte
address records in `assets/address_observations.json`. These builder outputs bind
the raw-UTF-8 hashing scheme, byte order and per-fact table addresses to the
declared world without training. For the hashing mechanism and UTF-8 boundary checks, see
[byte addressing](../byte-addressing.md). Address equality does not imply
equivalent token addresses or useful recipient behavior.

**Acceptance checkpoint B:** source/preparation/training/evaluation permissions are disjoint; each asset descriptor verifies; no cross-key collision is silently accepted; A/B assignments differ while addresses remain bound.

## C — Prepare independent recipients

The first non-native arm for a width materializes three own-preparation runs (one per declared seed). Their checkpoints initialize only their matching recipient. The runner validates architecture and tokenizer identities and loads every nonmemory backbone tensor from that recipient checkpoint. It never imports source-producer or other-width model tensors. Native-memory conditions intentionally start without a preparation checkpoint.

**Acceptance checkpoint C:** each seed has a verified own-preparation checkpoint; all other seed checkpoints remain separately keyed; `initial_backbone` binds checkpoint, architecture, and tokenizer hashes.

## D — Verify the training boundary

Run a single arm first:

```sh
uv run --locked klab research portability run \
  --campaign-root artifacts/engram-portability-smoke \
  --recipient width32 --representation token \
  --condition adapter-tuned --seed 17
```

The adapter-tuned, random, and corrupt controls select only the memory output projection and gate. `frozen-only` attaches without recipient updates. `joint` trains the declared model parameters. `native` learns local memory without an imported artifact. `disabled` has no memory attachment. Each training checkpoint records exact optimizer parameter names; the final portability audit compares tensor digests and verifies frozen tensors/assets.

**Acceptance checkpoint D:** optimizer membership equals the declared canonical names; frozen parameters remain byte-identical; adapter-only runs preserve every backbone tensor; asset hashes still match after training. Any violation stops the arm.

## E — Observe checkpoints and swaps

Every trained arm records step-zero, intermediate, and final immutable checkpoints. Each observation evaluates all development cases against their corresponding A/B replacement artifact; the final checkpoint additionally evaluates the final partition. Multi-hop queries use the model's preceding predicted answer to construct the next structured key. There is no oracle traversal that substitutes expected intermediate nodes. Semantic traces and their `as_of` values are recorded beside recipient predictions.

For each transferred-artifact arm, the A/B/A probe evaluates the same three-hop A query under A, then B, then A again while keeping recipient weights fixed. The final A prediction must reproduce both the exact predicted answer and path from the first A prediction under deterministic decoding. This demonstrates attachment switching/reversion only; it is not evidence of learned language understanding.

The predeclared development threshold is exact-answer accuracy ≥0.95. An arm that reaches the planned final update without reaching threshold is marked right-censored. No time-to-threshold is extrapolated. `frozen-only` has one step-zero observation and is censored if it misses the threshold.

**Acceptance checkpoint E:** checkpoint and observation digests are immutable; each observation records step, tokens, dev accuracy/path accuracy, semantic traces, final-only results, and A/B/A responses; censoring is explicit.

## F — Continue and package evidence

`continue` executes remaining arms in protocol order. Omit `--max-arms` to run the full three-seed matrix; use it for a bounded invocation only:

```sh
uv run --locked klab research portability continue \
  --campaign-root artifacts/engram-portability-smoke --max-arms 1
uv run --locked klab research portability continue \
  --campaign-root artifacts/engram-portability-smoke
uv run --locked klab research portability report \
  --campaign-root artifacts/engram-portability-smoke
```

The report command validates and writes a content-addressed `portability_evidence-<digest>.json` snapshot. A partial snapshot is an incomplete campaign, not a completed matrix. For the existing static study report, pass that exact file with `study report --portability-evidence PATH`; the report bundles the original evidence bytes. Arm receipts, run manifests, checkpoints, and detailed observation files remain in the campaign root.

**Acceptance checkpoint F:** evidence schema/digest validates; completed, failed, censored, and still-planned arms are distinguished; every reported observation hash resolves to its immutable observation file. Claims must state the executed arm count and protocol scale.

## Z — Stop or resume safely

Stop after a bounded run by omitting `continue`; the campaign root is self-contained and no-clobber. Continue only with the exact original protocol, source tree, assets, and update count. A failed invocation records an immutable attempt under `evidence/failures/`; a later `continue` retries arms without a final receipt. Failures are not silently converted to completed or censored results. Do not edit generated files or treat unexecuted plan rows as evidence.

**Acceptance checkpoint Z:** no training is launched before explicit `run`/`continue`; stale or conflicting inputs fail closed; incomplete work remains labeled incomplete. This campaign can establish bounded behavior for its generated task family and DenseLM widths. It cannot establish cross-architecture, cross-tokenizer behavioral, broad language, source-trained representation, or natural-language semantic-query portability.

## Latest recorded smoke result

The completed smoke matrix executed all 120 declared arms for two updates across seeds 17/41/73: 42 token, 42 byte, and 36 semantic arms at widths 32/64. The refreshed evidence report records 120 right-censored arms at the 0.95 development exact-answer threshold, zero failures, and zero threshold completions. Every arm had 0.0 development exact-answer accuracy and 0.0 final exact-answer/path accuracy. The report contains 324 checkpoint observations; all 102 trained-arm audits passed, while 18 `frozen-only` arms used step-zero checkpoints. All 468 saved A/B/A probes reproduced the exact A answer and path; a post-fix semantic-arm smoke also verified exact-answer/path equality reporting.

This is runner and retrieval-workflow evidence, not useful transfer: recipient accuracies stayed at zero. Token/byte artifacts were directly compiled from structured facts, and semantic queries used supplied structured vectors rather than a natural-language query producer or source-model-trained representation. The analytical maximum single-arm RAM estimate was 69,222,400 bytes; total checkpoint storage was estimated at 185,317,632 bytes. Observed median train-plus-checkpoint-observation time was 6.9923 seconds per arm.
