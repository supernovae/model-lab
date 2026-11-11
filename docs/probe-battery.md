# Probe battery: fast-fail checks for native checkpoints

Commands below use the [activated, provisioned environment](../README.md#first-run).

The probe battery is a small, versioned suite of cheap checks that tells you, in
seconds to minutes, whether an idea is worth a longer run. It scores supported native PyTorch
checkpoints, optionally against a baseline checkpoint, and ends with a machine-readable
verdict and a recommended next action.

MLX checkpoints and attached semantic packs are currently refused; text-level
reference models have a separate full-tier path described below.

**Probes screen; they do not prove usefulness.** A pass means "nothing obvious
broke and loss moved the right way on a small fixed sample", nothing more.
Selecting ideas again and again on the held-out probe items slowly turns them
into development data, so keep a separate, untouched final evaluation (a fresh
data split, a longer run, more seeds) for any claim that an idea is better.

```sh
# Every lab try runs the fast tier automatically (disable with --config no-probes).
klab try wider-ffn.yaml --vs BASELINE.yaml

# Any run or checkpoint, any tier.
klab probe RUN_OR_CHECKPOINT --vs BASELINE_RUN --backend cpu --config standard
klab probe RUN --vs BASE --backend cpu --json      # stable schema
klab report probe-20261010T150839Z-ece5a739

# Standard lm-eval tasks (optional extra).
uv sync --locked --extra cpu --extra lmeval
klab probe RUN --vs BASE --backend cpu --config full

# Place a result on the reference curve (reads records; no download, no compute).
klab compare TRY_OR_PROBE_ID --references
```

`--extra cpu` selects dependencies, not the saved checkpoint backend. These
examples explicitly score on CPU; use a registered compatible runtime for GPU
scoring. `explore` defaults to CPU, while standalone `probe` inherits the saved
backend when `--backend` is omitted.

A target is a sealed try/probe record ID or path (its exact candidate checkpoint),
a run directory, a checkpoint directory inside a run, or a run id
under `WORK_DIR/lab/runs` (or declaration `runs_dir`). `--vs` accepts the same local inputs;
a record supplied there selects that record's candidate, not its baseline.
Records never silently select a newer checkpoint; missing generations must be
restored or a different input explicitly chosen. See [result lookup](result-lookup.md).
Results land in
`WORK_DIR/lab/probes/<probe-id>/probe.json`, sealed with `record_sha256` by
the same lab-record writer as `try.json`. `report`, `probe` and the dashboard
read every record through one verified reader (`klab.lab_records`), so an
edited record is rejected everywhere. A try stores its result in `try.json`
under `probe`.

```text
PROBE BATTERY  klab-probe-battery v5 · suite b7f7b01d · tier full (ran fast → standard → full) · 22.6s
  candidate lab-try-20261010T191906Z-6a192ff3  step 60 · 7.7k tokens · 98.7k params (43.4k active)
  baseline  lab-base-43194de82c467490-try-20261010T191906Z-6a192ff3  step 60 · 7.7k tokens · 43.2k params

  ✔ PASS        Held-out loss               3.247 vs 3.278    ···◀◀│·····  Δ -0.031 (-0.9%) ±0.005    ppl 25.7
  ▲ WARN        Calibration (ECE)           0.296 vs 0.263    ·····│▶▶▶··  Δ +0.032                   
             ↳ Confidence no longer tracks accuracy: check the LR schedule end, label smoothing or an output-scale change.
  ✔ PASS        Top-1 agreement             0.964                                                     JS 0.018 · KL 0.073
  ✔ PASS        Degeneration                0.690 vs 0.966    ◀◀◀◀◀│·····  Δ -0.276                   distinct-2 0.03
  ⊘ SKIPPED     Fact recall (from weights)  candidate: inapplicable: the run trained on synthetic, not withheld_facts (no facts manifest to bind)
  ✔ PASS        Fact recall (reworded)      0.250 vs 0.250    ·····│·····  Δ +0.000 ±0.000            chance 0.25
  ✔ PASS        Needle in context           0.167 vs 0.167    ·····│·····  Δ +0.000 ±0.000            by length ▂▂▂ (27/40/55 tok)
  ✔ PASS        Standard tasks (lm-eval)    0.215 vs 0.215    ·····│·····  Δ +0.000 ±0.000            lambada 0.00 hellaswag 0.20 arc 0.16 piqa 0.50

  verdict ▲ WARN  next → LONGER RUN
  Every tier holds up: schedule a longer run (more tokens or seeds) to confirm the gain. Probes screen; they do not prove usefulness.
  guard   verdicts use the held-out split · ok · fact_recall dev +0.00 / held-out +0.00; needle dev +0.00 / held-out +0.00
  legend  meter ◀ better │ worse ▶ (full = fail threshold)
```

The meter is centred on the baseline; a full half-bar is the probe's fail
threshold. Colors follow `NO_COLOR`/`FORCE_COLOR` and are off when piped.

## Probes

Probes run cheapest first: by tier, then declared cost.

| Probe | Tier · cost | Metric | Judged on (warn / fail) | Hard | A failure suggests |
|---|---|---|---|---|---|
| Held-out loss | fast · 1 | mean token cross-entropy on the validation split, same eval protocol as the baseline | relative increase 0.5% / 3% | yes | the change hurts at this budget; revert or retune LR/warmup |
| Calibration (ECE) | fast · 1 | top-1 expected calibration error, 15 equal-width bins (Guo et al. 2017); the non-empty bins are kept as a reliability diagram | +0.02 / +0.05 | no | confidence no longer tracks accuracy (schedule end, output scale) |
| Top-1 agreement | fast · 1 | argmax agreement with the baseline on fixed held-out prompts, plus KL(base‖cand) and JS in nats | below 0.5 (warn only) | no | a large behavior shift; informative only together with loss |
| Degeneration | fast · 2 | seq-rep-4 of greedy continuations (Welleck et al. 2019), distinct-1/2 (Li et al. 2016) | +0.05 / +0.20 | yes | loops: duplicated data, too-high LR, positional change |
| Fact recall (from weights) | standard · 1 | closed-book: ranks the trained value of one of the run's own withheld-facts *training* facts (manifest of its `dataset.synthetic_seed`) among the values of the same relation, with no context; also reports never-trained control facts | −0.05 / −0.15 | no | the facts are not stored in the weights (runs trained on `withheld_facts` only; inapplicable elsewhere) |
| Fact recall (reworded) | standard · 2 | ranks the stated answer among candidates from the same relation (mean token log-prob), asked in different words than the fact was stated; fractional tie credit | −0.05 / −0.15 | no | context handling or memory wiring |
| Needle in context | standard · 3 | retrieve a code word stated at the start of a filler context of exactly 96/192/384 characters (text-level: the same strings for every tokenizer, references included; a context beyond the model's window scores chance); fractional tie credit | −0.05 / −0.15 | no | attention span, positions or sequence-length changes |
| Standard tasks (lm-eval) | full · 4 | mean `acc` over lambada_openai, hellaswag, arc_easy, piqa at `limit=50` | −0.02 / −0.05 | no | treat small moves as noise at tiny scale |

**Credit.** Recall and needle rank candidates by mean token log-prob of
`" " + candidate` after the prompt. When k candidates tie at the top score the
item earns 1/k if the answer is among them, else 0, so a model that scores
everything alike lands at chance, never at 100% (ties used to break toward the
answer; a uniform-score regression test pins this). The per-item `picked`
field shows a tie as `tie: a / b`, never as the first candidate.

**Constant picks (the PR #63 demo finding).** In the demo lab every model
chose the same candidate for every question of a relation (for example
`red`/`panda`, `gold`/`falcon`). This is not a scorer artifact: no item tied,
shuffling the candidate order left every pick unchanged, and each pick matched
the model's context-free preference (its scores with the prompt replaced by
"The"). Those models had seen about 1,500 tokens, so their output was close to
uniform (candidates within 0.1–0.2 nats of each other) and the prompt did not
move the ranking. Accuracy was then exactly chance, as the fractional-credit
rule requires. Recall rows now record `details.constant_pick` (and
`baseline_constant_pick`) when every question with the same candidate set got
the same pick, and the Behaviors page warns about it. Mean token log-prob does
favor multi-token candidates whose later tokens are predictable; that is a
known property of length-normalized ranking, and the same for both arms.

**Noise.** Each probe declares its `uncertainty`. Held-out loss is a ratio
(sum of token NLL / tokens) over evaluation windows, so its delta SE is the
token-weighted, window-clustered delta-method SE of the difference
(`window_clustered_ratio`). Recall and needle compare the same items on both
arms, so their delta SE is the paired SE of the per-item credit differences
(`paired_items`). A change within 2 SE counts as `pass` with a "within noise"
note, never as a win or a regression. The held-out loss SE is **within-run
eval noise**: it says how precisely one checkpoint was scored, not how much a
different seed would move it. At tiny budgets seed-to-seed spread is about
5–10× larger, so the renderer labels the bar "eval noise, 1 seed" and every
promoting verdict (`escalate`, `longer_run`) ends by recommending paired
seeds before a win is claimed.

**Tiny-model honesty.** The reworded fact recall probe is *in-context* (the
fact is stated, then asked in other words), because a tiny model knows no
facts. **Fact recall (from weights)** is the closed-book counterpart. It is
bound to the facts the run actually trained on: the `withheld_facts` source
trains on `split_facts(dataset.synthetic_seed)`
(`klab.data.withheld_facts`), so the probe reads that seed from the run's
resolved config, checks it against the run's prepared-data provenance
(`data/manifest.json`) and builds the same seed's manifest. The held-out split
asks each trained fact with its own prompt, the dev split rewords it, and the
never-trained facts are reported as `control_accuracy`. A model that stored its
training facts beats chance on the held-out split while the control stays at
chance. The manifest seed and digest are recorded in `details.manifest_seed`
and `details.manifest_sha256`; different seeds are different items, so they
never pair. A run trained on another dataset, and every reference model, has
no facts manifest: the probe is `skipped` with an "inapplicable" note (it does
not make the battery incomplete). A `withheld_facts` run whose provenance is
missing or disagrees with its config is `unavailable`, i.e. missing evidence
with the reason, so trained facts are never scored as never-trained. When both models are near-uniform (JS < 0.01)
top-1 agreement is meaningless and is reported as uninformative instead of
warning. lm-eval tasks are borrowed for their established metric definitions,
but tiny models sit at or near chance (≈0.25 hellaswag/arc_easy, 0.5 piqa, ≈0
lambada); at `limit=50` a difference of a few points is noise. The full tier
records each task, the chance level and this caveat in the result.

## Tiers and fast-fail

- `fast` (default for `try`): loss, calibration, agreement, degeneration; well
  under a second on the CPU smoke model.
- `standard`: adds fact recall from weights, reworded fact recall and needle
  retrieval.
- `full`: adds lm-eval. The `lmeval` extra stays optional: without it the probe
  is `unavailable` with the install hint and the battery is **incomplete**
  (see below), never a pass.

The lm-eval adapter scores through the same token-scoring path as the probes
(`klab.probes.scoring`). It follows the harness boundary rules (trailing
context whitespace moves to the continuation; context and continuation are
encoded together and split at the context's token count; an empty context is
the end-of-text token) and scores every continuation token, windowing with
maximal left context when a continuation is longer than `max_seq_len`. A task
that produces no accuracy is an error, i.e. missing evidence.

**Larger limits and more task families.** Save a [probe declaration](probe-config.md):

```yaml
probe_config_version: 1
tier: full
lm_eval_tasks: [hellaswag, arc_challenge, winogrande]
lm_eval_limit: 500
```

`klab probe RUN --config benchmark.yaml` runs any
lm-eval tasks at any limit through the same adapter. Bad options (duplicate
tasks, a limit below 1) are refused before any work. The probe suite and its
digest do not change. The tasks, limit, task versions and per-item
prompts/targets go into `details.benchmark_group`, so these results only ever
compare with results run the same way, never with the default `limit=50`
slice or the packaged references. Tasks the probe does not know get no chance
level rather than a guessed one. An expensive run (thousands of items, many
tasks, a 360M reference) is the same command on a bigger machine; CI tests
the option handling and the benchmark-group separation without running
lm-eval.

A **hard** probe that fails (held-out loss, degeneration) stops the battery at
once; the remaining probes are `skipped`. A NaN/inf loss raised by the native
evaluator is a **numerical failure**, a distinct outcome rather than a probe
error: the held-out loss row fails with `details.numerical_failure`, the battery
stops (`fast-fail: numerical failure in heldout_loss`) and the verdict is
`fail`/`abandon` with a numerical-failure reason and suggestion, even without a
baseline. If the *baseline* is the one that fails numerically, nothing can be
judged against it and the row is `error` (missing evidence). Any other failure finishes the current
tier but the battery does not escalate ("not promising"). Declaration `fast_fail: false` runs
everything. A probe that crashes is recorded as `error`; it never sinks the try,
but it is missing evidence: the battery does not escalate past it and the
verdict is `incomplete`.

## Missing evidence, cancellation and resources

A check that should have produced evidence and did not is never a success.
`unavailable` (optional dependency missing, unsupported engine), `error` (the
probe raised) and `not_comparable` rows, and probes left unrun by a stop, are
listed in `verdict.missing` with the reason, printed as `missing ...` lines and
shown in the dashboard banner. Unless a real failure already decides the
verdict, the status is `incomplete` and the action is `rerun`.

Arms are loaded one at a time: per tier the baseline is loaded, measured and
released, then the candidate is loaded, measured and judged probe by probe, so a
battery never holds two models in memory. Inside `try` the battery runs under
the try's own context (the same `CANCEL` sentinel, declaration `resource_envelope` and
SIGINT/SIGTERM handling as training and scoring); standalone `probe` creates
`LAB/probes/<id>/CANCEL` as its sentinel and reads a native envelope through declaration `resource_envelope: FILE`.
The context is checked before each arm and before every probe. `stop.kind`
records why a battery stopped: `fast_fail` (hard failure, including a
numerical failure), `not_promising`
(a failure in an earlier tier), `missing_evidence` (an earlier tier is
incomplete), `cancelled` (sentinel), `resources` (envelope violated) or `oom`
(host `MemoryError` or a torch out-of-memory error, which stops further probe
work instead of becoming an error row) or `interrupted` (Ctrl-C or SIGTERM).
Everything measured before the stop is kept: on a signal the battery is
finalized (unrun probes listed as missing evidence, verdict `incomplete`), the
progress file is marked `stopped` and the sealed record is published before the
command exits (130 for standalone `probe`; `try` records `interrupted` at
`phase: probing`). Inside `try`, a stopped battery never undoes the training comparison: a
cancel during probing marks the try `interrupted` and keeps both scored arms and
the comparison; resources or OOM leave the try `completed` with an incomplete
battery.

Validation is scored once per arm. Inside `try` the scoring pass doubles as the
probe's validation pass (an observer on the native evaluator collects per-window
sums and top-1 confidences); standalone, held-out loss and calibration share one
pass that is computed only on a cache miss.

## Verdict and next action

`verdict` is `{status, action, next_tier, reasons, missing, suggestion}`;
actions are:

| Condition (first match) | status | action |
|---|---|---|
| numerical failure (NaN/inf held-out loss) | fail | `abandon` |
| inside a try whose comparison is `NOT_COMPARABLE` | incomplete | `rerun` (names the failed held-out checks; never escalates) |
| no baseline, any requested evidence missing (unavailable, error, not comparable, stopped by cancel/resources/OOM/interrupt) | incomplete | `rerun` (names each gap) |
| no baseline, every requested check produced evidence (a reference model only once every task and item is scored) | info | `compare` |
| a hard probe failed | fail | `abandon` |
| any probe failed | fail | `tweak` (that probe's hint) |
| missing evidence (unavailable, error, not comparable, stopped by cancel/resources/OOM/interrupt) | incomplete | `rerun` (names each gap) |
| reference model: a benchmark task or item unscored (error, OOM, partial) | incomplete | `rerun` (names the unscored tasks; never placed on the curve) |
| overfit guard fired | warn | `tweak` |
| held-out loss improved beyond noise, tiers left | pass/warn | `escalate` with `next_tier` |
| held-out loss improved beyond noise, all tiers run | pass/warn | `longer_run` |
| otherwise (no measurable gain) | pass/warn | `tweak` |
| `probe --config final`, any requested evidence missing (whatever the row above, e.g. a soft `fail`), unless a NaN/inf loss or hard failure decided it | incomplete | `rerun` the same final declaration unchanged (names the gaps) |
| `probe --config final`, otherwise (complete, or decided by a NaN/inf loss or hard failure) | status kept | `report` (terminal; `next_tier` null; never escalate, tweak or re-run) |

## Held-out guard and the held-back final split

Every item set has three disjoint, deterministic splits (different facts,
templates, needles, filler and prompts): **dev**, **held-out** and the
held-back **final** split. Iteration verdicts (`try`, `probe`) use the held-out
split. Because every try reads it, a long loop still selects on it, so the
final split exists for the one verdict that must not be selected on:

```sh
# Once, for a candidate you have already chosen. Never to pick between candidates.
klab probe CANDIDATE --vs BASELINE --backend cpu --config final
```

`--config final` (or declaration `final: true`) scores the text probes, agreement and generations on the final split
(held-out loss stays on the run's validation split) and records `final: true`
and `suite.verdict_split: "final"`. Reading the final split anywhere else raises
`FinalSplitLocked`, and `tests/test_eval_integrity.py` checks that only the
probe CLI's final-declaration path can open it. Final items have their own item groups,
so a final result never pairs with an iteration result. Final evidence is also
kept out of every selection view: points carry `final`/`verdict_split`
provenance, and ordinary `compare` (subjects, others and the same-checkpoint
enrichment), the Pareto view, the Models page, dashboard history, trends,
behaviors and next steps exclude final results (Home activity lists them as
`final`). `klab compare --final A B` is the explicit report of two final
results (no references, no ordinary evidence). Arms cache item-scored probes per
split, so reusing an Arm across an ordinary and a final battery always
scores each split afresh. A completed final verdict's action is `report`.
Agents may look at dev results while iterating; the guard flags
`overfit_suspected` when the dev gain is beyond its paired noise and exceeds
the held-out gain by more than `max(0.15, 2 SE)`. Never edit probe items to make
an idea pass: changing any item or threshold changes the suite digest, and a
digest change requires bumping `SUITE_VERSION` (a pinned test enforces this;
v2 introduced fractional tie credit and the declared uncertainty methods).
Compare results only within one suite digest; the dashboard warns when history
mixes suites.

## Result schema (`klab-probe-v1`)

Top level: `format`, `created_at`, `suite` (`name`, `version`, `sha256`, split
digests, `verdict_split`), `tier`, `tiers_run`, `stop` (`stopped`, `kind`, `at`,
`reason`), `target`/`baseline` (run id, checkpoint and its sha256, step, tokens
seen, parameters, parameter bytes, tokenizer and validation digests,
`max_seq_len`, `eval_group`), `comparable`, `protocol`, `probes`, `guard`,
`verdict`, `seconds`; standalone records add `probe_id` and `record_sha256`.
`eval_group` hashes the validation data, tokenizer, loss mask and eval
protocol: losses are only comparable within one group.

Each probe row: `id`, `title`, `tier`, `cost`, `metric`, `higher_is_better`,
`hard`, `thresholds` (`mode`, `warn`, `fail`), `uncertainty`, `suggests`,
`explains`, `reference`, `status`
(`pass|warn|fail|info|skipped|unavailable|not_comparable|error`),
`value`, `baseline_value`, `delta`, `regression` (positive = worse; relative for
loss), `delta_se`, `within_noise`, `improved`, `note`, `details`, `seconds`.
Keys are stable within a format version; `tests/test_probes.py` pins them.

Per-probe details that the dashboard and `compare` read. The ranking probes
(both fact recalls and needle) keep `details.items`: per held-out item, the
`answer`, what the candidate `picked` and its `credit`, `baseline_picked` and
`baseline_credit` when a baseline ran, and the `prompt` (not for needle, whose
prompts are long filler). They also keep `details.item_group`, the digest of
the probe, the ranking protocol and the rendered items. Two results with
different item groups are `not_comparable` ("different items"), never
compared. Calibration keeps `details.reliability` (and `baseline_reliability`):
the non-empty confidence bins with their mean confidence, accuracy and share of
tokens, the same bins ECE is computed from. Suite v3 added fact recall from
weights, the item records and the reliability bins; results from older suites
keep their own digest and are compared only with each other.

## Reference models and `klab compare`

Our runs should sit on a known curve, not only next to each other. Four public
checkpoints in the 70M–360M range are pinned in
`klab.reference_models` (immutable Hugging Face commits, safe snapshots
only: static metadata and `*.safetensors`, no pickles, remote code or
quantization configs; the same rules as the Pythia trajectory adapter):

| Reference | Repository @ commit | Resident / active params | Training tokens (model card) |
|---|---|---|---|
| `ref:pythia-70m-deduped` | `EleutherAI/pythia-70m-deduped@9a7c847e` (step 143000) | 70.4M / 44.7M | 299,892,736,000 |
| `ref:pythia-160m-deduped` | `EleutherAI/pythia-160m-deduped@c54a0e0b` (step 143000) | 162.3M / 123.7M | 299,892,736,000 |
| `ref:SmolLM2-135M` | `HuggingFaceTB/SmolLM2-135M@93efa2f0` | 134.5M / 134.5M | "2T" |
| `ref:SmolLM2-360M` | `HuggingFaceTB/SmolLM2-360M@f8027fd0` | 361.8M / 361.8M | "4T" |

A reference is scored by the same battery, the same token-scoring path and the
same lm-eval adapter as our checkpoints:

```sh
uv sync --locked --extra cpu --extra reference --extra lmeval
klab probe ref:SmolLM2-135M --config full
klab compare --list-references
```

For a machine-readable registry listing, use
`klab compare --list-references --json`. This reads existing records only;
it does not download or score models and needs no subject/result ID. The JSON
listing has its own schema, separate from a comparison report:

- `schema_version`: integer `1`.
- `references`: array containing every pinned registry entry in registry order,
  even when no evidence is available. Each entry has:
  - `target`: string `ref:NAME`, accepted by `probe` and `compare`.
  - `reference`: the full registry declaration. String fields are `name`,
    `family`, `repo_id`, `revision` (full immutable commit, never abbreviated),
    `model_type`, `architecture`, `license` and `training_tokens_source`.
    `training_tokens` is an integer; `step` is an integer or `null`.
  - `lm_eval_available`: boolean, true when a reference point with that name
    has lm-eval metrics in the ordinary collected evidence (verified packaged
    or lab records). This is the same criterion as the human listing's `✔`.
    Recall-only evidence does not count; zero lm-eval accuracy does count.

Registry presence is not evidence availability. With an empty evidence pool,
all entries remain listed with `lm_eval_available: false`; an empty lab may
still have packaged evidence. Availability does not assert comparability with
any particular result, include held-back final evidence, or prove model quality.

A reference runs the probes that are defined at the text level (`--config full`
is required): **fact recall** (reworded, in-context), **fact recall from
weights** and **lm-eval**. Recall items are rendered text (prompt, answer,
candidates) scored by mean token log-prob under each model's own tokenizer, so
`details.item_group` (a digest of the probe, the ranking protocol and the
items) is the same for every model and fact recall compares across references
and lab runs. Its held-out loss on our validation split is never computed or
compared, because it has its own tokenizer and training data. The needle probe
is *not* run on references. Its filler context is sized in the model's own
tokens (~50/75/95% of `max_seq_len`), so its items differ per tokenizer and the
result would never be comparable. A text-level needle with fixed character
lengths is a follow-up. A reference is never a `--vs`
baseline. Without the optional extras the command stops before any work with
`uv sync --extra reference --extra lmeval`.

**Packaged results.** The sealed probe records of all four references ship in
`src/klab/probes/reference_results/` and are verified on read like any lab
record, so `compare`, the dashboard and CI use them without downloading a
model. Measured on CPU (fp32, lm-eval 0.4.13, transformers 4.57.6) with the
`full` tier's tasks at `limit=50`, zero-shot; two independent runs gave
identical records. They were re-sealed for suite v3, which added the two
text-level recall probes; the lm-eval items, per-item outcomes, accuracies,
benchmark group and checkpoint digests did not change:

| Reference | lambada_openai | hellaswag | arc_easy | piqa | mean `acc` | fact recall |
|---|---|---|---|---|---|---|
| pythia-70m-deduped | 0.28 | 0.30 | 0.30 | 0.54 | 0.355 | 0.688 |
| pythia-160m-deduped | 0.36 | 0.42 | 0.34 | 0.60 | 0.430 | 0.938 |
| SmolLM2-135M | 0.36 | 0.44 | 0.52 | 0.62 | 0.485 | 1.000 |
| SmolLM2-360M | 0.42 | 0.46 | 0.64 | 0.74 | 0.565 | 1.000 |

These are 50-item slices, not the published full-task numbers; compare them
only with results in the same benchmark group (below). Fact recall is 16
held-out items with 4 candidates each (chance 0.25). Fact recall from weights
is recorded as `skipped` (inapplicable): a public model was never trained on a
withheld-facts manifest, so there are no trained facts to ask about. To refresh
them, run
`klab probe ref:NAME --config full` for each reference and copy the sealed
`probe.json` to `reference_results/NAME.json`; `tests/test_references.py`
checks every packaged record against the pinned registry.

**Comparison groups.** A number is only compared with numbers measured the same
way:

- held-out loss within one `eval_group` (validation data, tokenizer, loss mask,
  eval protocol), as before;
- lm-eval accuracy within one `benchmark_group`: a digest of the harness, the
  scoring-protocol version (`klab-lmeval-scoring-v1`), tasks, task
  versions, shots, `limit`, the metric used per task and, per item, a hash of
  its document, its rendered requests (context and continuation) and its
  target(s) (`details.benchmark`), so a reworded prompt or a changed target is a
  different group. Each task's per-item outcomes are
  kept (`details.tasks.<task>.items`) so two results in the same group get a
  **paired** standard error: per-task paired SEs combined as `sqrt(Σ se_t²)/T`.
  The battery's own lm-eval row uses the same paired SE, and refuses a
  baseline in another benchmark group (`not_comparable`).

`klab compare RESULT [RESULT…] [--references] [--json]` reads sealed
records only (a try/probe id, a record path or `ref:NAME`; the first is the
subject). Each point is completed with compatible evidence for the **same
checkpoint** (same checkpoint sha256) from every verified record in the lab:
per (comparison group, checkpoint) each field keeps its newest available value
and its source. So when `compare TRY --references` reports missing lm-eval
evidence and suggests `klab probe TRY --config full`, rerunning the same
`compare TRY` picks up that probe and says where the number came from. It and prints one table of points and, per metric, each pair as
better/worse/within noise (|Δ| ≤ 2 paired SE), **NOT COMPARABLE** with the
reason, or **MISSING EVIDENCE** with the command that produces it:

```text
COMPARE  lab-try-20261010T191906Z-6a192ff3@60 vs 4 point(s) · comparisons only within one eval/benchmark/item group

    point                                   resident    active   tokens  held-out loss  lm-eval acc  fact recall
  ▶ lab-try-20261010T191906Z-6a192ff3@60       98.7k     43.4k     7.7k          3.247        0.215        0.250
  ◆ pythia-70m-deduped                         70.4M     44.7M   299.9B              –        0.355        0.688
  ◆ pythia-160m-deduped                       162.3M    123.7M   299.9B              –        0.430        0.938
  ◆ SmolLM2-135M                              134.5M    134.5M     2.0T              –        0.485        1.000
  ◆ SmolLM2-360M                              361.8M    361.8M     4.0T              –        0.565        1.000

  held-out loss (lower is better)
    ≠ NOT COMPARABLE vs pythia-70m-deduped, pythia-160m-deduped, SmolLM2-135M, SmolLM2-360M
      ↳ reference models have their own tokenizer and training data, so held-out loss on our split is never comparable; use lm-eval accuracy or fact recall

  lm-eval accuracy (higher is better)
    ✖ WORSE         vs pythia-70m-deduped                     Δ -0.140 ±0.034
    ✖ WORSE         vs pythia-160m-deduped                    Δ -0.215 ±0.038
    ✖ WORSE         vs SmolLM2-135M                           Δ -0.270 ±0.040
    ✖ WORSE         vs SmolLM2-360M                           Δ -0.350 ±0.040

  fact recall (higher is better)
    ✖ WORSE         vs pythia-70m-deduped                     Δ -0.438 ±0.128
    ✖ WORSE         vs pythia-160m-deduped                    Δ -0.688 ±0.120
    ✖ WORSE         vs SmolLM2-135M                           Δ -0.750 ±0.112
    ✖ WORSE         vs SmolLM2-360M                           Δ -0.750 ±0.112

  reference curve (lm-eval accuracy)
    ▶ lab-try-20261010T191906Z-6a192ff3@60   98.7k params (43.4k active)    0.215 █████████
    ◆ pythia-70m-deduped                     70.4M params (44.7M active)    0.355 ██████████████
    ◆ pythia-160m-deduped                    162.3M params (123.7M active)  0.430 █████████████████
    ◆ SmolLM2-135M                           134.5M params                  0.485 ███████████████████
    ◆ SmolLM2-360M                           361.8M params                  0.565 ███████████████████████
    chance (task mean) 0.250

  legend  Δ = subject − other ± paired SE · within noise = |Δ| ≤ 2 SE · resident = all weights · active = touched per token
```

(A 99k-parameter mixture-of-experts smoke model trained on 7.7k tokens, so it
sits below chance on lm-eval and at chance on fact recall; the point is the
placement, not the score.) `--json` emits
`{subject, points, comparisons, curve}`.

## Dashboard

The [dashboard tour](dashboard.md) walks through every page. Probe results
appear in four of them. **Experiments** shows the verdict banner (with any
missing evidence), per-probe results against the baseline with meters, charts,
per-probe trends, history and a side-by-side compare. **Models** has the
checkpoint catalog and the Pareto view. **Behaviors** shows generations, recall
hits and misses, needle by length and reliability diagrams. **Home** shows the
latest verdict and the next command. Every page reads records through the same
verified reader as `report` and lists rejected (edited or unreadable) records
instead of showing them.

The **Pareto view** plots quality (held-out loss, lm-eval accuracy or fact
recall) against a cost: resident parameters, active parameters per token,
resident or active weight bytes, training tokens or scoring latency (ms/token).
*Resident* counts every weight in memory; *active* counts what one token
touches (one embedding row, the routed experts, the looked-up memory rows), and
a grey bar spans each checkpoint's active → resident range. Points come from
every verified lab record through `klab.probes.points` (shared with
`compare`): each scored arm of every `try`, including tries run with
`--config no-probes`, every probe target and baseline, and the packaged
reference results (purple diamonds, lm-eval and fact recall). One point per
(comparison group, checkpoint): compatible measurements are merged field by
field, each keeping its newest available value with its source (`evidence`),
so a newer record without latency does not drop an older record's latency. It
plots only within one comparison group (`eval_group` for loss,
`benchmark_group` for lm-eval, `item_group` for fact recall), chosen with a
selector that defaults to the selected result's group. Hidden groups, points
without the chosen cost and older records without a group are counted in a
caption, never mixed in. The dotted frontier respects the metric's direction,
the dashed line is the chance level, a log cost axis is on by default when
references are present, and "How to read the Pareto view" explains it for
learners. Plain `klab train` runs join the view once
`klab probe DIR/RUN` scores them; the Models page lists the
ones still unscored, with that command.

![Pareto view: lab runs on the reference curve](assets/dashboard-models-pareto.png)

For pinned tasks and full metric receipts, see [controlled benchmarks](controlled-benchmarks.md).
