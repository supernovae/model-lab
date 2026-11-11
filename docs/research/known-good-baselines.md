# Known-good baselines

A known-good baseline is a reviewed, reproducible, explicitly bounded lifecycle and behavior reference. It is not a best model, evidence of general intelligence, or an automatic replacement when a loss is lower. Declarations live in the lifecycle sidecar; `klab research validate` separately reports what is currently verified, unavailable, or invalid.

## Historical personal references

The original dense-small-v1 failed-gate candidate and dense-lm-v1 learning-reference
outcomes, frozen gates and limitations are preserved in the
[original reference guide](https://github.com/supernovae/kybern-workshop/blob/0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4/provenance/lab/docs/research/known-good-baselines.md).
These historical registry statements are not current Lab defaults or execution
readiness claims. New work requires an explicit current declaration and a fresh
experiment identity; see [external resource authoring](../external-research-resources.md).

The [dense-lm-v1 report manifest](https://github.com/supernovae/kybern-workshop/blob/0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4/provenance/lab/artifacts/research-reports/3f6deec6c11b70750f92219e23fb37c017922017c4730b775b245f27be82c3f0/manifest.json)
is preserved. The failed-gate dense-small-v1 bundle
`1d471241a17eca7b6d9680cfc3f595a921ab0aabe4f8b2e54a60835a6e95bb00`
remains unavailable in the pinned tracked trees, as recorded in the
[lifecycle supplement](https://github.com/supernovae/kybern-workshop/blob/0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4/provenance/lab-lifecycle-supplement.json).
This is not a claim about all host storage or a recovered checkpoint.

## Use a reference in a new experiment

Inspect a retained reference through the native lifecycle before planning a
child. The packaged registry is empty. Set `LIFECYCLE` to your explicit lifecycle
JSON, `EVIDENCE_ROOT` to its evidence root, and `BASELINE_ID` to a baseline declared
in that file. For the report command, save a `report-selectors.yaml` with the same
locations (replace these absolute example paths), and set `REPORT_CONFIG` to it:

```yaml
report_config_version: 1
lifecycle: /path/to/lifecycle.json
evidence_root: /path/to/evidence
```

```sh
uv run --locked --extra cpu klab report --kind research --config "$REPORT_CONFIG" --json
uv run --locked --extra cpu klab research baseline describe "$BASELINE_ID" --lifecycle "$LIFECYCLE" --evidence-root "$EVIDENCE_ROOT"
uv run --locked --extra cpu klab research validate --baseline "$BASELINE_ID" --lifecycle "$LIFECYCLE" --evidence-root "$EVIDENCE_ROOT"
```

`report` takes these selectors through `--config`; the research subcommands take
`--lifecycle` and `--evidence-root` directly. Relative paths in report YAML resolve
beside that YAML file.

Historical evidence and locally available checkpoint bytes are different states.
Use the [training-program guide](../experiment-programs.md) for a new declared
checkpoint chain and [iteration checks](../iteration.md) for authenticated parent,
source, input and runtime gates. Keep outputs in a fresh task directory under
the external persistent root. Do not rerun a failed historical candidate under
its old identity or overwrite its tokenizer, checkpoints or receipts.

## Historical gates and retention

Consult the [preserved original](https://github.com/supernovae/kybern-workshop/blob/0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4/provenance/lab/docs/research/known-good-baselines.md#historical-gates-and-retention)
for the original candidate's exact gates and retention records. Preserve zero,
negative, failed and untested outcomes. Static report validation does not make
omitted checkpoint binaries locally available or runnable.

## Findings, branches, and promotion

A Finding records conditions, evidence, bounded claims, reviewer attribution, disposition, rationale, next action, and reopen conditions. Dispositions are: **rejected** (unsupported under declared conditions), **inconclusive** (unanswered), **learning** (useful evidence without justified compute), **replicate** (independent repetition justified), **scale** (replicated evidence warrants larger compute), and **promote** (eligible for integrated baseline review). Stages are descriptive: mechanism, micro, replicate, scale, confirm, promote. Maturity axes are independent implementation, capability, efficiency, and portability declarations; no axis numerically ranks another.

Use the explicitly selected report and baseline commands above to inspect declared branches and evidence availability. `report --kind research` is a declared next-test view, not optimal planning. A prior-design advisory never blocks planning; changing the sealed study/protocol identity can make equivalence unassessed.

Promotion is an explicit reviewed lifecycle JSON change, never a metric trigger. A known-good baseline needs an accepted promotion and complete preparation, training, checkpoint, validation, capability, generation, resources, and static-report evidence. A replacement also needs its parent’s required regressions, identity-bound passed outcomes, a resource tradeoff, and explicit `supersedes`; parallel promotion leaves its parent active. Run `klab research validate --lifecycle "$LIFECYCLE" --evidence-root "$EVIDENCE_ROOT" --json` before relying on a declaration. Keep independent mechanism branches interpretable; a combined candidate needs its own campaign and regressions.
