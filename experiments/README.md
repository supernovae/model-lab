# Experiment workspace

This directory contains durable, reviewable experiment definitions—not mutable
run output.

| Path | Purpose | Expected contents |
|---|---|---|
| [`samples/`](samples/) | Copyable teaching and starter experiments | Small configs, walkthroughs, expected artifact shapes, and explicitly non-scientific example output |
| [`research/`](research/) | Actual Model Lab research campaigns | Identity-bound protocols/config references, preregistered gates, iteration notes, findings, and evidence links |
| `$KLAB_WORK_DIR/experiments/` | External persistent execution workspace | One named root per experiment, with a shared run store, receipt, staging, exercises, captures, and local reports |

Do not promote sample output into research evidence. To turn a sample into a
campaign, copy its editable inputs into a named `research/<campaign>/` directory,
declare the question and gates, and use a separate named work directory.

Reusable tokenizer and prepared-data caches retain their existing shared identity/location semantics; they do not need to live under each experiment. Deliberately retained compact evidence is promoted separately to `artifacts/`. See [workspace policy](../docs/workspaces.md).

Existing packaged lessons and catalog recipes remain under
`src/klab/research/resources/`; existing reference configs remain under
`configs/references/`. This directory indexes how they are used rather than
duplicating them.
