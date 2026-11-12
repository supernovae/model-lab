# Sample experiments

Samples teach the workflow and are safe to copy. They are not claims that a
mechanism improves quality or performance.

- [TinyStories microlab](../../docs/tinystories-microlab.md): an optional external
  Workshop recipe; its declarations are no longer copied into Lab.
- [Chained training program](../../docs/experiment-programs.md): an offline
  authored plan with preparation, immutable locking, and a checkpoint-bound
  child phase.

For an advanced research recipe, supply your own complete
[current resource closure](../../docs/external-research-resources.md). The entry
ID and resource path below are placeholders; no personal recipe is packaged.

```sh
klab research scaffold neutral-study --resource-root /path/to/resources \
  --scale smoke --data offline --backend cpu --output experiments/new-study
klab study plan experiments/new-study/study.yaml
```

Every seed must be explicit in the recipe. The generated README describes later
preparation and execution steps; scaffolding itself executes no experiment.
For a checked-in reference layout, see
[`configs/references/dense-small-v1`](../../configs/references/dense-small-v1/).
Copy editable inputs; do not edit a known-good reference or treat smoke output as
research evidence.

A sample should say what it demonstrates, what it deliberately does not prove,
the expected cost class, and which paths can be deleted safely.

## Retained neutral teaching fixtures

The training-program trio, corpus-shape plan, held-out gate and pinned fraction
tokenizer remain active Lab teaching fixtures. They teach native interfaces using
synthetic inputs; preserving copies in Workshop does not make them personal
research or remove their current teaching role. See the
[explicit disposition map](../../docs/history/sample-disposition.json) for byte identities
and separate stage-one, stage-two and guide preservation records.

Original prose is [historical evidence](https://github.com/supernovae/kybern-workshop/blob/df5a43aff682d14c3a3481413b80f12e7894a042/provenance/lab/experiments/samples/README.md),
not authority to run or a guarantee of model quality. Current commands below use
the retained Lab files. Personal research belongs in externally authored recipes.

## Corpus shape and exact generated fraction

[`corpus-shape-fraction.yaml`](corpus-shape-fraction.yaml) is an offline CPU
example. Its four independently frozen releases compare chat training with and
without tool-trace shapes, and an unrestricted LM release with an exact
361/640-generated-token LM release. Both contrasts declare every changed
resolved field; neither is controlled, because the tokenizer and available
training records also change. The target optimizer exposure is 640 tokens per
arm, not proof that the same distinct source tokens were consumed. The pinned
fraction tokenizer is checked in under
[`tests/fixtures/corpus-sample/pinned/`](../../tests/fixtures/corpus-sample/pinned/).
If an exact subset becomes infeasible, preparation fails instead of rounding.

From the repository root, in the locked environment:

```sh
uv run --locked --extra cpu klab experiment validate experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked --extra cpu klab experiment prepare experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked --extra cpu klab experiment diff experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked --extra cpu klab experiment lock experiments/samples/corpus-shape-fraction.yaml --json
```

Preparation materializes sources, releases, exports, tokenizers and packed
datasets but starts no training. Inspect the lock output and the effective
config before submitting any cell. `experiment run LOCK --cell
main:corpus=full --json` enqueues a worker run; run `klab controller run
--store klab-work/experiments/corpus-shape-fraction-sample-v5/controller`
in a second process to dispatch, verify and ingest it. `experiment collect LOCK
--json` writes a content-addressed evidence index; `experiment reconstruct
LOCK --index INDEX --json` checks that index and emits five retrospective views.
Use the paths returned by the commands for `LOCK` and `INDEX`. A single-cell
smoke leaves the remaining cells pending and comparisons unavailable. No
quality claim, review decision or baseline promotion follows from it.

The expected cost class is a small CPU smoke (four 20-update cells only if all
are explicitly submitted). Only task-owned directories under
`klab-work/experiments/corpus-shape-fraction-sample-v5/` may be removed
after confirming no worker is active; the checked-in declaration and pinned
tokenizer are durable inputs.
