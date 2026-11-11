# Contributing an architecture experiment

Catalog membership is optional; ordinary configs remain usable. A forked catalog entry is an explicit local JSON path, not a package ID:

```sh
klab research describe experiments/my-entry.json --json
klab research scaffold experiments/my-entry.json --scale smoke --data offline --backend cpu --output experiments/my-study
klab study plan experiments/my-study/study.yaml
```

A local entry must validate against the packaged strict schema, and its recipe is resolved from the fork root's sibling `recipes/` directory. Unknown fields, duplicate JSON keys, non-finite values, invalid scales/designs, absolute/traversing paths, and unsafe resource paths are rejected. Scaffold publication refuses an existing destination and atomically publishes only validated config files; it never downloads data, trains, opens a worker/store, or overwrites an output.

State the question and hypothesis, exact fixed controls and dotted fields varied, data/tokenizer and packing identities, scale/backend/budget/seeds, card roles and applicability, confounders, and what the result cannot establish. Keep training/tokenizer inputs train-only: validation and capability cards are not tokenizer fitting material. Preserve negative, zero, unavailable, and partial outcomes. Do not reframe cache/accounting estimates or table capacity as runtime, cost, semantic-retrieval, or general-capability evidence.

Scaffolds contain hashes for their declared scientific inputs and coordinates. If a generated input is edited, its metadata no longer binds the changed bytes; re-scaffold instead of silently repairing it. Use the offline smoke route for a small mechanism proof and independently choose a larger scale/data route only when its prerequisites are explicit. TinyStories preparation may use network/cache only after you explicitly run tokenizer/data preparation, and its cards are out-of-domain stress rather than story quality.

## Findings and promotion review

Keep the catalog separate from lifecycle decisions. A Finding cites retained
evidence, conditions, supported and unsupported claims, reviewer attribution,
one disposition, a next action, and explicit reopen conditions. Use the six
dispositions precisely: rejected, inconclusive, learning, replicate, scale,
and promote. Stages and maturity axes are descriptive rather than automatic
progression or rankings. See the [known-good baseline guide](known-good-baselines.md)
for the required record fields and branch workflow.

A promotion is a reviewed lifecycle JSON declaration, never an automatic
response to lower loss. An accepted known-good reference requires identity-bound
preparation, training, checkpoint, validation, capability, generation,
resources, and static-report evidence. Replacement records also enumerate
parent-required regression checks and an explicit resource tradeoff. Run
`klab research validate --json`; missing evidence blocks eligibility, while
documentary citations remain visible without becoming measured proof.
