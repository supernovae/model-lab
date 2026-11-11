# Campaigns: declare and execute an experimental workflow

Commands below use the [activated, provisioned environment](../README.md#first-run).

A Campaign is a YAML/JSON DSL that connects the steps of an experiment: inputs,
readiness checks, training plans, runtime selection, collection, evaluation and
review. Use it when a question needs several dependent operations and you want
their inputs, controls and completion conditions to remain explicit and reviewable.

## Why use a Campaign?

Instead of an external Python script choosing files, ordering jobs and deciding
what counts as complete, the declaration names each stage and binds it to typed
inputs. The lab checks those bindings during execution and records the outputs
that downstream stages actually consume. A missing dependency, incompatible
input or incomplete ingestion blocks the dependent step.

The linked [ExperimentPlan](experiments.md#authored-experiment-plans-and-resolved-locks)
declares concrete parameter changes, fixed settings, comparison cells and
checkpoint phases. A Campaign selects those locked cells and connects their
collected checkpoints to the declared evaluations. Together they make it possible
to check that the configurations and artifacts being compared are the ones the
author specified, without separate Python coordination. They do not determine
whether the hypothesis, controls or evaluation are scientifically sufficient.

For example, to compare attention budgets, declare the changed budget in the
plan's cells and hold data, tokenizer, seed, exposure and other model settings
fixed. Add a run stage for each intended cell and bind its collection and
evaluation stages. Review the plan before dispatch and the retained configurations
afterward. Campaigns currently require explicit cell selection; they do not
silently expand an incomplete declaration into all the comparisons you intended.

Corpus Forge owns source acquisition, documents, lineage and frozen releases.
ExperimentPlan owns scientific configurations, artifact identities, comparisons
and locks. Campaigns add dependencies, readiness policies, recoverable execution
and explicit authorization. Individual native CLI operations remain usable for
a simpler workflow.

The DSL is under active development toward more complete campaign execution.
Use the supported [stage forms](#stage-forms-and-binding) to compose a workflow;
[current boundaries](#current-boundaries) describe operations that still need
separate native preparation or additional implementation. Unsupported steps are
not implicitly supplied by a hidden script.

## Declaration format

The required `campaign_version: 1` field identifies the machine-readable schema.
It is not the name of the Campaigns feature or a separate campaign product edition.

The [Campaign JSON Schema](../schemas/campaign-plan-v1.schema.json) describes the
strict authoring format. YAML/JSON is data only: duplicate keys, executable tags,
nonfinite numbers, unknown fields, duplicate stage/dependency IDs, unknown or
self dependencies, cycles and incompatible producer kinds/scopes are rejected.
Scientific declarations (`project`, `source`, `suite`, `policy`, `panel`, `recovery`) must
be declaration-relative without absolute paths, parent traversal or symlinks.
Operational artifact paths and reference-mode lock locations may instead be
absolute locations in the selected persistent root; they preserve their exact
values and still reject traversal or symlinks. A dependency in `requires` is not
an artifact binding: each typed input names its compatible producer.
Approval bindings may name transitive ancestors. Topological execution preserves
authored order among available stages.

## A complete small corpus workflow

[`examples/tiny-campaign.yaml`](../examples/tiny-campaign.yaml) is a valid partial
campaign and a complete, runnable corpus/readiness/approval workflow. Its recipe
contains twelve short local text files: ten train documents in two source families
and two test documents in one heldout family, across `technical_docs` and
`developer` domains. The selected LM training view uses the `raw_document` shape.
It contains no external tokenizer/model artifact placeholders and starts no training.

```yaml
campaign_version: 1
id: tiny-campaign
stages:
  - id: corpus
    kind: corpus_release
    scope: corpus
    project: tiny-campaign/corpus.yaml
  - id: readiness
    kind: corpus_readiness
    scope: corpus
    requires: [corpus]
    corpus: corpus
    policy:
      min_unique_train_bytes_by_domain:
        technical_docs: 100
        developer: 100
      required_nonzero_languages: [en]
      required_nonzero_shapes: [raw_document]
      min_heldout_families: 1
  - id: corpus-approval
    kind: approval
    scope: release
    requires: [readiness]
    bind: [corpus, readiness]
```

Use one named external persistent root (the default is XDG/home, not the
checkout); the global `--work-dir` precedes `campaign`. Commit the full
declaration closure before mutation, including the corpus recipe and local source
files. An uncommitted declaration is blocked before workspace creation; the
explicit `--allow-uncommitted-declaration` override records exact declaration
statuses and hashes but does not bypass artifact, approval, or runtime checks.

```sh
export KLAB_WORK_DIR=/data/klab
klab campaign validate examples/tiny-campaign.yaml --json
klab campaign plan examples/tiny-campaign.yaml --json
klab campaign status examples/tiny-campaign.yaml --json
klab campaign next examples/tiny-campaign.yaml --json
klab campaign apply examples/tiny-campaign.yaml --json
klab campaign explain examples/tiny-campaign.yaml --json
klab campaign approve examples/tiny-campaign.yaml corpus-approval --note 'Reviewed these bound inputs' --json
klab campaign resume examples/tiny-campaign.yaml --json
```

`validate` performs static checks only. `plan`, `next` and `explain` verify requisite
existing inputs and committed receipts without acquiring/building/training or
creating a workspace. `status` preserves historical last-committed stage outcomes
and separately reports fresh `recoverability` availability; the historical
`verified_at` and `verification: last_committed` are not fresh rehashes. A linked
declaration-relative `recovery` manifest supplies classified reconstruction steps,
while an unlinked campaign can only classify known receipt references. A missing
checkpoint is nonreconstructable training state, never permission to retrain.
`campaign reconstruct CAMPAIGN` invokes only deterministic recovery and does not
replay campaign approvals, stage receipts, or training. Before initialization
`status` projects unexecuted stages without creating a workspace.
The campaign CLI's own `PATH` is normalized; authored artifact references still
cannot contain parent traversal. All verbs accept `--json`; stdout is one sorted
`klab-campaign-command-v1` object. Progress and in-checkout storage warnings
go to stderr. Text mode lists stage ID/kind/scope/state/outcome, reasons, deficits,
identities and next action.

For example, after corpus readiness the next action is structurally:

```json
{"action":"approve","stage":"corpus-approval","reason":"approval required","identities":{"outputs":[]}}
```

A gate's stage row separately includes its binding SHA and bound identities.
Scope labels are domain ownership, not measurements of success.

## Stage forms and binding

Every stage has `id`, `kind`, `scope` and optional `requires` (default `[]`).

| Kind | Scope | Explicit inputs/settings |
|---|---|---|
| `artifact_reference` | Derived from artifact kind | External ExperimentPlan `artifact`, including full SHA, identifier, version, producer and verified relative or absolute operational location; `from_phase: null` |
| `corpus_release` | `corpus` | `project`; only local/deterministic-generator sources accepted, never network acquisition |
| `dataset_snapshot` | `corpus` | Pinned dataset `lock`, exclusive `output`, `cache_dir`, optional interrupted-acquisition `resume`; uses native generic acquisition |
| `tokenizer_train` | `tokenizer` | Tokenizer `config`, optional upstream `snapshot`; verifies that the config names that snapshot |
| `data_prepare` | `model` | Run `config`, upstream `tokenizer`, optional `snapshot`; preserves native preparation and provenance checks |
| `corpus_readiness` | `corpus` | `corpus`, `policy`, optional `tokenizer`; token policies require a tokenizer dependency; token-only policies may bind `measurement_receipt` and `measurement_sha256` together |
| `tokenizer_reference` | `tokenizer` | External `artifact` of kind `tokenizer`; provenance manifest required |
| `token_measurement` | `tokenizer` | Verified `corpus` release and `tokenizer`; actual selected-view/split counts |
| `experiment_plan` | `model` | `source`, `mode: lock|reference|bind`, `tokenizer`, `prepared`, optional `corpus`; reference mode requires `lock`, bind mode takes the run config from its `data_prepare` producer |
| `runtime_acceptance` | `runtime` | `plan`, optional mutually exclusive logical `profile_id` or registered `worker`; per-cell fresh runtime bindings and storage-headroom checks |
| `experiment_run` | `model` | `plan`, matching `runtime`, exact `cell` |
| `experiment_collect` | `evaluation` | `plan`, matching `run`; seals the selected cell, complete ingested evidence, and unique highest-step verified checkpoint generation/digest |
| `evaluation` | `evaluation` | `collect`, declaration-relative `suite`, optional `runtime` acceptance reference or explicit `backend: cpu` without a runtime reference; exact collected generation and evaluation runtime |
| `generation_panel` | `evaluation` | Matching `collect`, `evaluation`, accepted `runtime`, declaration-relative `panel`; immutable descriptive observations, never a quality gate |
| `model_readiness` | `model` | `evaluation`, mandatory declaration-relative `policy`, optional operational `review` receipt; verifies typed readiness against the suite index and the exact human review binding |
| `approval` | `release` or `model` | Nonempty `bind` of declared ancestors |

External artifact scopes: source snapshot/build/release/export → corpus;
tokenizer → tokenizer; prepared data/stage bundle/checkpoint → model;
capability card/prompt set → evaluation. Artifacts lacking a domain verifier,
notably `outcome`, are not accepted. Directory presence never proves identity.
Missing local inputs block with their path; existing corrupt/mismatched inputs fail.

Lock mode reuses existing preparation, resolution and immutable lock publication
APIs. Reference mode reopens the supplied lock. Selected cells must bind exactly
the upstream tokenizer and prepared-data artifact digests. Direct snapshot inputs
use the same verified dataset/tokenizer/prepared contract as standalone
ExperimentPlan. Forge-backed cells still require an upstream corpus release whose
identity matches the lock's selected variant; Campaign never silently substitutes
a variant or fabricates a release for a direct dataset. Synthetic cells must omit
`corpus`: a separate corpus/readiness stage can be a research-workflow prerequisite,
but is not thereby the model's training data. Every plan consumer reopens the lock.
An explicit run cell selects exactly one locked cell; there is no automatic
multi-cell dispatch. Local workers use the existing lock-bound request,
controller receipts, checkpoint verification and ingestion machinery.

`tests/test_campaign.py` generates its own full fixture under `tmp_path`, adding
real synthetic tokenizer training (seed 7, vocabulary 260), prepared data, an
ExperimentPlan, CPU worker execution (two steps / 32 supervised targets), a
checkpoint-bound heldout EvaluationSuite and a typed ModelReadiness policy. These
synthetic model inputs are **independent of its Corpus Forge release**. The fixture
threshold of 100.0 is only wiring acceptance, not a generally useful
model-quality threshold. No mock controller completion or checked-in
tokenizer/cache/run artifact is used.

## Corpus policy facts and projected passes

The policy can declare `min_unique_train_bytes_by_domain`,
`min_unique_train_tokens_by_domain`, `required_nonzero_languages`,
`required_nonzero_shapes`, `min_heldout_families`, and/or `passes`. At least one
requirement is necessary; minima are nonnegative integers.

Measurements first verify the frozen release. For each domain, the denominator
is normalized source-document UTF-8 text from `documents.jsonl`, restricted to
`split: train` and `drop_reason: null`, deduplicated by `content_sha256`
independently within that domain. It is **not** raw snapshot bytes or the build
report's overlapping/repeated mixture byte total. Token minima encode those exact
same kept unique documents with the pinned tokenizer after disabling tokenizer
padding and truncation, using `encode(text, add_special_tokens=False)`; tokens
are never inferred from bytes.
Languages use these documents; shapes use selected views that explicitly include
`train` in `training_splits`; heldout coverage counts distinct nonempty
source-family IDs in kept test lineage.
An absent measurement produces a typed deficit, not a pass.

### Canonical source-token receipts

`corpus measure-tokens` measures requested source domains, not the rendered-view
counts returned by `corpus describe` or Campaign `token_measurement`:

```sh
klab --work-dir /data/klab \
  corpus measure-tokens RELEASE --tokenizer TOKENIZER --policy POLICY.yaml \
  --output /data/klab/experiments/TASK/source-tokens.json --json
```

The command streams raw `documents.jsonl` bytes, authenticates them at EOF,
deduplicates `(domain, content_sha256)` on disk and counts raw token IDs without
injected EOS, padding, truncation or packing. Default batches are bounded by
256 documents and 1,048,576 source UTF-8 bytes. An eligible document larger
than 1 MiB is rejected,
not truncated or split. Operational batch choices do not change scientific
identity. Progress is stderr plus a launch-bound operational JSONL log.

`COMPLETE` is published exclusively after input authentication and successful
reduction. Existing results are read-only reuse only after validating release,
tokenizer, policy, implementation and evidence bindings; conflicting results are
never overwritten. Interrupted work is not a receipt and has no chunk resume.

Normal measurement authenticates the complete release and tokenizer through
their existing verifiers, sharing the release proof only within this operation.
Historical evidence overrides are restricted compatibility paths, not a general
way to skip verification. Their scope is preserved in the
[development notes](research/development-evidence.md#retired-campaign-specific-measurement-shortcut).
Use normal native measurement and authenticated receipts for new declarations.

A token-only Campaign readiness policy may consume a canonical receipt instead
of encoding again. Declare both its operational `measurement_receipt` path and
the expected full-file `measurement_sha256` in the committed Campaign. The
normalized inline policy must equal the receipt's policy, and the original policy
file's exact SHA, release, streamed documents, tokenizer, implementation and
evidence must still validate. Completed stages reopen these bindings rather than
treating their historical stage result as fresh authentication. A mismatch fails;
it does not authorize a new measurement or overwrite.

Token-only readiness reports requested source-domain bytes and tokens.
Unrequested language, selected-shape and heldout-family facts are unavailable
(`null`), not zero. Mixed policies retain their language/shape/heldout semantics
and use bounded streaming with disk-backed joins; a token-only receipt cannot
stand in for those other measurements.

### Projected source passes

```yaml
passes:
  basis: bytes  # or tokens, requiring the pinned tokenizer
  requested_total: 1000000
  mixture: {technical_docs: 0.5, developer: 0.5}
  max_required: 4
```

For each requested mixture domain `d`:

```text
required_passes[d] = ceil(requested_total * mixture[d] / unique_available[d])
```

Authored numeric weights retain exact decimal precision; numeric strings are
rejected. Integer rational ceilings avoid rounding across exact boundaries.
Weights must be positive, finite and sum to one; total and maximum are positive
integers. Zero or unavailable denominator blocks. The maximum per-domain estimate
must not exceed `max_required`. This mixture is policy input, never a rewrite of
Corpus Forge's release mixture. Overlap across domains means these are per-domain
upper-bound reuse estimates, **not train-order or actual ingestion passes**.

Satisfied policies return `COMPLETE / READY_FOR_TOKENIZER` with no deficits.
Unsatisfied policies return `BLOCKED / EXPAND_MORE`, sorted deficits with
`dimension`, `observed`, `required` and optional `domain`. Descendants remain
blocked. The engine never adjusts sources, mixture or thresholds automatically.

## Durable state, authorization and replay

State vocabulary is exactly `NOT_STARTED`, `READY`, `RUNNING`, `COMPLETE`,
`BLOCKED`, `AWAITING_APPROVAL`, `INTERRUPTED`, `FAILED`, `INCONCLUSIVE`.
Result outcomes are `EXPAND_MORE`, `READY_FOR_TOKENIZER`, `READY_FOR_NEXT_STAGE`,
`DO_NOT_ADVANCE`, `INCONCLUSIVE`. Next actions are typed objects with reasons and
identities. A prerequisite that cannot advance blocks descendants.

A domain-separated canonical declaration SHA creates its own namespace:

```text
<work-dir>/campaigns/<id>/<declaration-sha>/state.json
```

Stage-input hashes bind the declaration, stage and sorted immutable dependency
results, including transitive input hashes. Machine-local availability is separate
from scientific identities; absolute paths do not enter Campaign declaration,
stage-input or approval-binding digests. A new declaration with changed policy or
artifact SHA gets a new state/approval namespace, even with the same campaign ID.
Old approvals never authorize it.

Stage receipts also record exact scientific declaration-closure path hashes. Before
reusing completed stages or issuing an approval, Campaign compares current closure
bytes against those receipts. A changed corpus source, run config, suite or policy
is `DECLARATION_IDENTITY_CHANGED` even after a new Git commit when the Campaign
YAML text itself is unchanged; unrelated commits and dirty documents outside the
closure do not rebind science. Publish changed intent under a new Campaign
declaration identity rather than overwriting historical receipts.

`RUNNING` is durable before side effects. Immutable canonical stage receipts at
`receipts/<stage>/<input-sha>.json` publish before the atomic fsynced state index.
Resume reconciles receipts after a crash between receipt and index publication;
pre-receipt content-addressed lower-level outputs can be verified and reused.
Interrupted attempts remain visible. A per-campaign file lock serializes mutation.
Noncanonical/tampered records, conflicting output for one input SHA, missing or
changed completed artifacts fail closed: the engine never silently reruns them.
Repeated completed apply verifies without rebuilding, dispatching, training or
rewriting state/receipts.

`campaign apply|resume` will not submit a new optimizer run unless explicitly
passed `--execute-runs`. Without it, the first pending `experiment_run` returns
`next_action: execute_run` before creating any controller submission. Reconciliation
of an already submitted attempt needs no new authorization, but never enqueues a
replacement. The explicit flag does not replace the required bound campaign
approval, committed declaration, verified lock, runtime, or declared evaluation.
In state loss, `campaign reconstruct` restores only declared deterministic inputs;
the later `campaign apply --execute-runs` still requires fresh explicit intent.

`apply` initializes or reuses state; `resume` requires existing state. Both accept
`--max-wait-seconds` (default 120). Timeout leaves `RUNNING` with a `resume` next
action. The foreground budget bounds sequential protocol waits, not local hashing
or verification work. Runtime headroom is checked on the campaign's actual
execution workspace. Before reuse and after completion, the controller attempt
must match the full lock-bound config, phase, coordinate, science digest and
continuation binding, not just lock/cell labels. Multiple/conflicting attempts fail.
There is no automatic optimizer retry after a terminal worker interruption:
the stage remains `INTERRUPTED` with a `wait` action.

Approval waits for complete bound ancestors and binds their immutable outputs,
transitive stage inputs and the gate's own declaration/policy identity. `approve`
defaults to `--decision approve`; `--decision reject` blocks with `DO_NOT_ADVANCE`.
Receipts at `approvals/<gate>/<binding-sha>.json` are immutable. Identical decision
replay retains the original UTC timestamp and note; a conflicting decision needs
a new declaration, not an overwrite. Approval authorizes **only those inputs**;
it is not a quality finding or model promotion.
Resume also reconciles an approval receipt published before its gate-stage receipt,
without changing its original decision, timestamp or note.

Model readiness is assessed by the declared `model-readiness-v1` policy against
the verified checkpoint-bound EvaluationSuite index. Missing gates or coverage
produce `INCONCLUSIVE`, breached numeric gates `DO_NOT_ADVANCE`, required human
review still outstanding `NEEDS_REVIEW`, and satisfied policy
`READY_FOR_NEXT_STAGE`. Campaign evaluation/readiness stages cannot infer a
quality threshold from the last run log, and a campaign approval authorizes
execution only; promotion requires independent reviewed model-quality evidence.

An optional `model_readiness.review` names a separately issued
`model-review-receipt-v1` location (absolute locations under the persistent root
are allowed). It is **not** a Campaign execution approval. The declared review
must identify the same evaluation index and checkpoint; a missing or different
receipt cannot authorize readiness. If Campaign already committed an
`INCONCLUSIVE` assessment because the review was not available, do not replace
that receipt or silently redispatch the stage. Issue the named human review and
assess it explicitly with `klab readiness model POLICY INDEX --review RECEIPT`;
author a new Campaign declaration identity if a later Campaign assessment is
needed.

## Current boundaries

No tokenizer bakeoff, architecture selection, auto-generated evaluation protocol
or worker-farm scheduling stage is provided. Campaigns can prepare generic dataset
snapshots, train tokenizers, prepare arrays or reference verified existing inputs.
See [datasets](datasets.md) and the [optional external recipe example](tinystories-microlab.md).
Accelerator acceptance requires
a matching declared profile ID or named worker; pass `--runtime-profile PROFILE`
to each apply/resume that needs it. Each runtime stage authorizes only its assigned
run cells, and new dispatch revalidates the accepted identity.

Evaluation can reuse a profile/local-worker runtime stage with `runtime: STAGE`.
An accelerator checkpoint without that reference blocks; an explicit
`backend: cpu` evaluation omits the reference and records its CPU override.
Remote registered workers can train, but local suite evaluation has no SSH RPC:
declare explicit CPU evaluation or report that unsupported boundary.
See the [runtime contract acceptance workflow](runtime.md#runtime-contract-acceptance).
Capacity/probe checks establish bounded execution readiness, not evidence of
fit on other hardware, throughput superiority, useful model behavior, causality
or portability.

## Descriptive generation panels

A `generation_panel` stage records one fixed prompt panel against the **same
immutable checkpoint** as an authenticated evaluation index. It requires explicit
collection, evaluation and runtime-acceptance dependencies:

```yaml
- id: descriptions
  kind: generation_panel
  scope: evaluation
  requires: [collect, evaluation, runtime]
  collect: collect
  evaluation: evaluation
  runtime: runtime
  panel: generation-panel.json
```

The referenced declaration uses this strict format:

```json
{
  "generation_panel_version": 1,
  "id": "fixed-descriptions",
  "role": "descriptive_not_quality_gate",
  "prompts": ["def parse_config(path):", "SELECT user_id FROM"],
  "decoder": {"temperature": 0, "top_k": 0, "max_new_tokens": 64, "seed": 42},
  "checkpoint_selection": "same immutable generation as heldout evaluation"
}
```

Commit the panel with the Campaign declaration closure. Its prompt order, exact
text, decoder and seed are bound to the result; changing the panel requires a new
Campaign identity. `checkpoint_selection` is descriptive text, never permission
to resolve a moving `latest` or `best` pointer. The stage uses the collected
immutable generation and SHA authenticated by the evaluation index.

Run through the usual `campaign apply` or `campaign resume`. The stage uses native
checkpoint inference and the declared accepted runtime; it does not dispatch
training. An implicit CPU acceptance is supported for a CPU cell. Accelerator
execution requires a valid local runtime binding. Unsupported remote inference
and inconsistent checkpoint/index/runtime bindings are rejected before generation.

The immutable result preserves ordered prompts, raw returned text, completion
token IDs and per-prompt failures. Empty and repetitive outputs are evidence;
they are neither filtered nor retried. Reopening a completed stage verifies its
bindings and output bytes instead of generating another sample. A
`DESCRIPTIVE_EVIDENCE` outcome records completion of observation only. The panel
does not add quality gates, alter ModelReadiness results, approve a checkpoint or
promote a model. Keep readiness attached to its authenticated evaluation stage.

Each prompt has a durable attempt marker before inference and an immutable result
receipt afterward. Resume reuses recorded observations. A started attempt with
no durable result becomes `FAILED` with `InterruptedAttempt`, unavailable text
and token IDs, and no retry. Per-binding locking prevents concurrent callers
from sampling the same prompt twice. Keep the adjacent
`panel-<binding>.attempts/` journal with the final `panel-<binding>.json`; missing
or changed journal receipts invalidate the evidence. When native generation
raises before returning, its unavailable partial text and token IDs are recorded
as null, with the exception type and message preserved.

Generation panels retain descriptive observations for their bound checkpoint.
They do not change an existing experiment's protocol or turn generated text into
an automatic quality decision.
