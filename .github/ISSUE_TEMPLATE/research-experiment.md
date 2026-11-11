---
name: Research experiment
about: Track a scientific question, protocol, execution, and evidence review
title: "[experiment] "
labels: ""
assignees: ""
---

## Question and competing explanations

State the falsifiable question and plausible alternative explanations.

## Protocol identity and invariants

Link the checked-in `experiments/research/<campaign>/` record or lifecycle entry.
Declare data/split ownership, controls, seeds, budgets, fixed settings, and
acceptance gates before final outcomes are inspected.

## Execution workspace

Name the configured `KLAB_WORK_DIR`, expected storage envelope, backend, and
hardware. Do not use an anonymous `/tmp` path for a long run.

## Status and evidence

Record planned/running/completed/failed/censored status, verified artifact or
report references, negative results, and unresolved limitations. Open separate
Code task issues for software defects discovered during the experiment.
