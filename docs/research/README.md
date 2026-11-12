# Research workbench

Model Lab has two non-executing entry points: `klab learn list` for one initialized mechanism, and `klab research list` for declared controlled studies. Listing, describing, and scaffolding package metadata/configuration only: they do not train, fit a tokenizer, prepare data, open a run store, or access the network.

Current authoring uses an explicitly supplied resource closure. Prepare a current
entry, recipe with explicit seed options, base, profiles, datasets, source pins,
capability cards and lifecycle registry as described in the
[external-resource contract](../external-research-resources.md). In this example,
`neutral-study` is a placeholder for your own entry ID, not a packaged recipe;
`/path/to/resources` must be your complete local resource root.

```sh
klab learn list
klab research list
klab research scaffold neutral-study --resource-root /path/to/resources \
  --scale smoke --data offline --backend cpu --output experiments/new-study
klab study plan experiments/new-study/study.yaml
```

The packaged research list retains three historical cards pending separate
protocol retirement; it is not a list of scaffoldable recipes. The eight personal
recipe IDs were retired together with their catalog entries. Scaffolding does not
prepare data, run training, or grant authority to execute.

A scaffold publishes editable `base.yaml`, `tokenizer.yaml`, `matrix.yaml`, `study.yaml`, hashed standalone configs, and metadata. It does **not** execute them. Follow the generated README to train the tokenizer, prepare each required config's data, check a listed config before training an arm directly, or explicitly submit and collect a study. See [lesson paths](lesson-paths.md), [the experiment learning cycle](experiment-learning-cycle.md), [external resource authoring](../external-research-resources.md), [evidence and reporting](evidence-and-reporting.md), and [dashboard browsing](dashboard.md).

If you are unsure what to run next, start with the [experiment learning cycle](experiment-learning-cycle.md). It explains how to move from a known-good reference to the cheapest useful follow-up, distinguish broken evidence from a negative result, keep soft limitations from becoming retroactive gates, and decide when to stop, replicate, scale, branch, or promote.

The [research roadmap](roadmap.md) tracks capability evidence and open scientific
questions. Use [`experiments/samples/`](../../experiments/samples/) for neutral
teaching material. Personal project definitions and evidence are discoverable in
[Workshop](https://github.com/supernovae/kybern-workshop), whose current showcase
is basic synthetic CPU, TinyStories and DevMind. Other personal collections,
including Kernel Memory Lab, are retired from its current tree; the
[historical catalog](https://github.com/supernovae/kybern-workshop/tree/fd259e2cad4e9856d6d659e8df34728a98b32502/catalog) preserves original context.
New research belongs in the coordinated backlog with new identities and current
frameworks. Historical results do not establish current readiness or model
quality. Author new campaigns using the resource contract above. The
[retained research index](../../experiments/research/README.md) describes remaining
local records; this documentation change does not remove engine behavior.
Mutable output belongs in the
external persistent root (`$KLAB_WORK_DIR/experiments/`, or the XDG/home default);
declarations and compact verified evidence belong in Git. Report missing code through the
[contribution workflow](../../CONTRIBUTING.md#planning-and-contributions). Verified semantic retrieval accepts pre-encoded
vectors; a teacher-representation compiler and natural-language query encoder are
not shipped.

## Durable research and compact evidence

The [declaration-to-artifact lifecycle](lifecycle-recovery.md) explains commit-before-compute, read-only `research snapshot`, recovery manifests, checkpoint-bound EvaluationSuite/ModelReadiness, human-reviewed ModelFamily actions, and thin/portable archives. Publish small verified references without checking in data/checkpoints:

```sh
uv run --locked klab research evidence export --kind corpus_release <verified-release-dir> \
  --declaration <recovery.yaml> --output <evidence/release.json> --json
git add <evidence/release.json> && git commit -m "Record verified corpus identity"
```

The digest pins bytes and provenance, **not** scientific quality. A tokenizer-selection reference needs a verified selection receipt; a runtime probe is an observation. An absent external source and a lost checkpoint require different responses.

## Known-good decision records

The read-only lifecycle projection links bounded Findings, human dispositions,
declared next tests, and reviewed baseline promotions without changing catalog,
study, run, or report artifacts. Start with the [known-good baseline
guide](known-good-baselines.md) for the exact dense reference commands, gates,
capture requirements, branch workflow, and the distinction between archival
report verification and locally runnable checkpoint files.
The packaged lifecycle registry is empty. `klab report --kind research --json`
and `research validate --json` therefore provide no historical baseline clearance.
Select a current explicit registry to assess its declared evidence; historical
findings and baseline guides remain evidence of their recorded conditions only.

The [experiment ledger](experiment-ledger.md) centralizes retained lessons from
recall, context use, adaptation, architecture, exposure, scale and decoding
studies. Follow its evidence links for exact identities and controls; use the
[roadmap](roadmap.md) for declared next questions. Historical completion does not
imply local checkpoint availability or authorization to rerun a campaign.

Relative path values in matrix axis patches resolve from `matrix.yaml`; paths
in a base config resolve from that YAML file.

## Ownership-aware memory allocation

The personal allocation recipe is historical and no longer available as a
packaged scaffold. Its [archived corpus and allocation guide](../history/path-domain-corpus.md)
preserves the original declarations and limitations. A new allocation experiment
needs a current external resource closure, explicit seed options, actual input
bindings and a separately reviewed plan. No standard seed set is supplied by Lab.

Published worked examples: [FFN-substitution smoke report](https://github.com/supernovae/kybern-workshop/blob/a47389204a30da1bbfec5e0de4418b3e3e72d4f2/provenance/lab/artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/index.html) and [nano/offline FFN follow-up report](https://github.com/supernovae/kybern-workshop/blob/a47389204a30da1bbfec5e0de4418b3e3e72d4f2/provenance/lab/artifacts/research-reports/a444e2869973568e28315aa2cac1a97454d7f9d7b8742fd69de8358e867e7daf/index.html), with their [outcomes and interpretation](sample-report.md).

## Controlled Engram portability

The explicit `research portability` runner binds generated worlds, direct token/byte compiles, semantic packs, independent recipient-preparation checkpoints, update audits, immutable checkpoint observations, A/B/A replacement probes, and censored threshold reports. Start with the [A/B/C/D/E/F/Z lifecycle guide](engram-portability.md). The former personal research recipe is archived in Workshop; it is not a current packaged scaffold.

The historical source-learned Experiment A is separate: [retained learned-portability protocol](learned-engram-portability.md). It trains a source byte table with DenseLM SGD and tests independent width-64/128 recipients; compiled-world artifacts and the MiniLM pilot are historical evidence, not substitutes.

Staged documentation-only definitions: [compiled knowledge Engrams](compiled-knowledge-engram-v1.md) and [compiled Engram initialization](compiled-engram-initialization-v1.md). Neither has a runnable recipe or catalog entry.

See the [research launcher contract audit](launcher-contracts.md) for retained
operations, the removed byte diagnostic wrapper and consolidation boundaries.
