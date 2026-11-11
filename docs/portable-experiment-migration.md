# Portable experiment ownership

Kybern Model Lab supplies reusable experiment tools. A recipe supplies the
scientific choices: model and data declarations, seeds, budgets, comparisons,
artifact references and supporting documentation. Keep recipes in a local
directory or a Git repository of your choice.

[Kybern Workshop](https://github.com/supernovae/kybern-workshop) contains the
project's active examples and experiments. It is optional: installed Lab can
inspect recipes from other repositories without Workshop or an internal runner.
Preparation, prepared-input workflows and Lab mode support their documented
execution paths; the matrix profile provides metadata previews.

## What belongs in a recipe

Keep DSL, YAML and JSON declarations together with the documentation needed to
understand them. Use recipe-relative paths for declarations and explicit bindings
for external artifacts. Bind actual verified outputs when a later phase needs a
checkpoint, tokenizer or prepared dataset; do not guess future hashes.

Reusable parsing, planning, preparation, model execution, evaluation and report
verification belong in Lab. Experiment repositories should use those native
commands rather than per-experiment Python launchers, validators or test suites.
Generic engine tests should use small neutral fixtures inside Lab.

## Author and validate

Use the [portable recipe guide](portable-recipes.md) for native section
validation, recipe creation/export, configuration derivation and relocation.
Select a supported recipe profile:

- [Preparation recipes](preparation-recipes.md) connect declared source and
  preparation operations with explicit input and output bindings.
- [Prepared-input workflows](workflow-recipes.md) bind verified inputs to native
  resolution, admission, execution and reporting.
- [Lab mode](lab-mode.md) compares a candidate with a baseline using `try` and
  inspects its sealed result with `report`.

Metadata inspection checks declared structure and connections. An
`executable: false` result is a preview, not proof of available inputs, an
installed runtime, successful training or model quality. Follow the selected
profile's verification and execution requirements before running it.

Keep original manifests, hashes, receipts and results with their provenance.
Adapting a recipe to current interfaces does not rewrite historical evidence or
make an old result a new experiment. See the
[contribution policy](../CONTRIBUTING.md#pre-production-migration-policy).
