# Post-train triage (v1)

`train` runs a bounded, advisory triage **after** a normally completed run. `klab report --kind run RUN_ID --runs-dir PATH [--json]` reads the immutable report; it never reruns inference. Reports live at `runs/RUN_ID/post-train-triage/<sha256>.json`. A missing, malformed or ambiguous report is `UNKNOWN`, not evidence of a failed training run. Neither training status nor immutable checkpoint, run manifest, research registry, Finding, NextTest, or promotion decision is changed by triage.

## Evidence and applicability

Tier 0 reads the completed progress record, manifest, verified final immutable checkpoint, read-only metric database and run-owned held-out reports. The final checkpoint must have verified training state. Recorded loss points without a checkpoint-verified matching report are retained as `recorded_not_checkpoint_verified`; never treat them as scientific evidence for a cross-run trigger. Historical checkpoint metadata can bind a small report without rehashing all historical tensor files. Missing optional reports, cards and reviews mean `NOT_TESTED`, not `FAIL` or zero. Invalid existing reports mean `INVALID`. Explicitly out-of-domain cards mean `NOT_APPLICABLE`. Independent prose quality, causal dependence and human preference cannot be inferred from a loss curve or mechanical repetition.

Targets count **supervised target exposures**, not unique tokens or documents. Tier 0 retains each `(step, targets, loss)`, initial/terminal finite loss, delta, per-milestone deltas and marginal improvement in nats/target per million added target exposures. Four comparable ordered observations are required for a trend; split at the observation nearest the target midpoint, compare early and recent `(loss decrease)/(million additional targets)`. `regressing` means final loss at least 0.01 above the preceding loss. `still_improving` means recent absolute loss decrease at least 0.01. `plateau_possible` requires positive early decrease, nonnegative recent decrease below 0.01, and recent rate below one quarter early rate. Fewer than four points means `insufficient_history`; all remaining cases are `unclear`. These are heuristics, not convergence claims. An interval with no additional targets cannot supply a marginal rate.

Parent comparisons require explicit parent run/checkpoint lineage and matching verified checkpoint, validation dataset, tokenizer, protocol, evaluation target count and metric units. Similar run names are not lineage. Incompatible or absent comparisons are `UNKNOWN` with the mismatch recorded. Runtime numbers distinguish optimizer-update seconds, total wall time and component phase times; allocated/reserved device peaks and sampled memory are different measurement methods.

## Bounded diagnostic tiers

Tier 1 uses the six fixed TinyStories `reference_exercise.PROMPTS` exactly once each at greedy temperature 0, top-k 0, seed 42042, maximum 32 new tokens. Only `simple-continuation` and `named-character-continuity` get sampled probes: temperature 0.8, top-k 0 or 40, seeds 11 or 29, maximum 24 new tokens (eight calls). Compare returned token IDs with the corresponding greedy first-24-token prefix; token/bigram/trigram excess and empty/special-token cases measure mechanical changes, not prose quality. Long prompts are individually `UNKNOWN`, never truncated. TinyStories prompts may be out of domain for a different task. Automatic inference is skipped when parameters exceed 100 million, or on CPU exceed 10 million, or the run deadline has less than 120 seconds left; record the reason and recommend an explicit bounded probe if useful. Backend errors produce `UNKNOWN`, never a failed training result.

Recorded Engram, MoE and sparse-attention telemetry measures usage, not causal dependence. Configured mechanisms without telemetry remain `UNKNOWN`; absent mechanisms are `NOT_APPLICABLE`. A supported MLA cached-forward probe records actual expanded projected K/V cache allocation, **not** theoretical compressed-cache savings. No automatic memory-disabled ablation, expert sweep, held-out rerun, capability-card sweep, decoder ranking, text-quality gate or additional training occurs.

Tier 2 emits transparent post-hoc recommendations only. Every named trigger records `true`, `false` or `null` (missing operands), thresholds and evidence references. Loss/behavior divergence requires compatible parent loss decrease ≥0.05 nats/target plus a mixed, flat or regressed matched mechanical panel. Decoder sensitivity requires ≥3 fewer repeated-trigram excess occurrences versus matched greedy first-24-ID prefix on a probe. Endpoint still learning uses the Tier-0 label, but **more targets are not recommended** without declared comparable improving target behavior; cheap independent behavior measurement comes first. Possible plateau suggests capacity/data/task hypothesis review. Scale without behavior gain requires compatible reference, more parameters, ≥20% higher measured optimizer-update seconds, ≥0.05 lower held-out loss and flat/mixed mechanics. Seed disagreement is applicable only for declared completed sibling seeds of the same sealed matrix with matched non-seed settings and validated comparison protocol; terminal loss difference ≥0.05 or ≥3 trigram-excess difference on at least two matched cases. Missing usage for configured mechanisms recommends the smallest telemetry/inference check, not a causal ablation.

For matched 32-token greedy cases, `improved` means empty/special becomes valid or repeated-trigram excess falls by ≥2 without new invalidity; `worse` means the reverse or excess rises by ≥2. Other cases are `unchanged`. Both directions yield `MIXED`, only worse `REGRESSED`, only improved `IMPROVED`, otherwise `FLAT`. These are mechanical labels, never a human-quality judgment. Missing verified endpoint, comparable identity or required cases yields a `null` trigger, not `false`.

Candidates sort by cost (`negligible`, `low`, `medium`, `high`), then trigger code; the short display gives only the cheapest useful action. JSON retains additional questions, statuses, applicability, missing operands and known unknowns. New training remains `NOT RECOMMENDED YET` while cheap behavior questions remain unresolved. Tier 3 requires a new versioned explicit research workflow for budgets, width, seeds, datasets, architecture, memory compilation, broad suites, human review and full-factorial decoding. No recommendation dispatches a job or changes frozen scientific gates.

## Optional Surface Review overlay

`klab report RUN_ID --kind run --runs-dir RUNS_DIR --config report.yaml`
and `klab dashboard --runs-dir RUNS_DIR --surface-dir DIR` can show a
**verified read-only overlay** for sealed [Surface Review v1](surface-review-v1.md)
bundles. The report declaration is:

```yaml
report_config_version: 1
surface_dir: /path/to/surface-review-bundles
```

Set `RUN_ID` and `RUNS_DIR` to your retained run and its store, and
`SURFACE_DIR` to the review bundles described in the Surface Review guide. Set
`surface_dir` in `report.yaml` to that same directory: `report` reads its YAML
selector, while the dashboard command below uses `$SURFACE_DIR`. The report
declaration does not expand environment variables.

```sh
uv run --locked klab report "$RUN_ID" --kind run \
  --runs-dir "$RUNS_DIR" --config report.yaml --json
uv run --locked klab dashboard --runs-dir "$RUNS_DIR" \
  --surface-dir "$SURFACE_DIR"
```

Run-report JSON keeps the verified advisory report under `triage`, native
checkpoint observations under `evidence`, and the requested overlay under
`surface_review`. Without `surface_dir` in the declaration, it omits the overlay.
The telemetry dashboard retains its existing behavior. Reading changes neither
the report bytes nor its content-addressed filename.

The overlay first calls `read_triage` to verify the original immutable report,
then checks candidate bundles against its run and checkpoint digest. A
matching verified import can show `independent_subjective_quality:
REVIEW_AVAILABLE`; only a valid **completed** single-reviewer judgment
raises this to `OBSERVED_SINGLE_REVIEWER`. No matching verified bundle yields
`UNKNOWN`. More than one matching bundle remains a list of references, not a
chosen winner. An unrelated bundle's votes cannot satisfy this run. Explicit
reveal is separate from completion: neither availability nor a completed
still-blind review silently reveals the A/B identity mapping. Overlay status
does not change the immutable report, `surfaces.human_review`, checkpoint,
run, Tier-2 recommendations, or any promotion gate.

One self-blind reader supplies neither population preference nor inter-rater
agreement. Tier-1 outputs and Tier-2 mechanical triggers remain diagnostic,
not quality or causal proof; a study's nominally held-out prompts do not by
themselves establish train-disjointness. Compare changed data, tokenizer,
budget and architecture without attributing an effect to any one factor.
