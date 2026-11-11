# Capacity-aware execution and verified reuse

Keep execution efficient without changing the declared experiment. The lab can
reuse authenticated inputs, bound preparation work to available host resources,
and materialize private worker/stage inputs while preserving scientific identity.
Use these native paths instead of custom hashing, copying or orchestration scripts.

## When to use it

Repeated iterations often share a corpus, tokenizer and prepared arrays. Rebuilding
those inputs or repeatedly hashing their full unchanged dependency closure can
cost more than the changed training phase. Supported native verification boundaries
reuse signed local proofs when eligible, while retaining input and source checks.
Reuse is an execution optimization, not permission to change data, precision,
effective batch, architecture or the training budget.

| Situation | Verification behavior |
| --- | --- |
| New input, changed identity or unsupported reuse boundary | Cold byte and domain verification. |
| Unchanged input in an authenticated owner-only registered store | Eligible native commands can reuse signed host-local proofs. |
| Transfer to another worker/store trust domain | Authenticate bytes and semantics before publishing the destination. |
| Archive, recovery or model-family verification | Independent cold verification. |

Changed fingerprints, dependencies, implementation identity, unsafe paths,
missing/foreign proofs or invalid signatures require cold fallback. A failed
cold check is an error. Same-user compromise is outside the protection offered
by local proofs. File size, remembered hashes and directory presence are not
substitutes for verification.

## Inspect through the lab

Use [iteration checks](iteration.md#read-only-iteration-check) for the declared
delta, input bindings, parent compatibility, storage and runtime prerequisites.
Use `experiment explain` for a resolved lock and `campaign status`, `next` and
`explain` for orchestration state. These read paths do not launch training.

Supported ExperimentPlan operations, `stage`, and Campaign `plan`, `status`,
`next`, `explain`, `apply`, `resume`, and `approve` expose `--cold-verify`; use
it at a new trust boundary or when independently checking retained bytes.
Reuse and cold-fallback diagnostics belong to operational evidence, outside
scientific/plan identities.

## Bound preparation and materialization

The [runtime resource envelope](runtime.md#operational-resource-envelopes)
controls native host work. Worker counts are bounded by available CPU capacity,
RAM, per-worker requirements and explicit limits. Results are reduced in input
order so concurrency does not redefine prepared data. Unknown capacity is
reported conservatively rather than treated as unlimited.

Stage and worker materialization verify source and destination identities and
publish private copies. Native clone/copy support depends on the destination;
unsupported operations fall back to bounded copying. Physical storage placement
and worker choice remain operational settings. Follow [workspace policy](workspaces.md)
for capacity, ownership, relocation and retention.

## Interpret measurements

Keep preparation, verification, copying, optimizer updates, evaluation and
checkpointing timings separate. Record logical bytes independently of physical
I/O, sampled memory independently of native peaks, and unavailable counters as
unavailable. Avoid treating an isolated copy or hash improvement as a full-run
speedup or model-quality result.

## Opt-in phase observations

`data prepare`, `stage`, and Campaign `plan`, `status`, `next`, `explain`,
`apply`, and `resume` accept `--observations-output PATH.json`. The command
validates that this is a new JSON file under an existing non-symlinked parent
before it selects a runtime or creates a work root. Keep it outside the
prepared cache, stage output, Campaign state, and declared inputs; a sibling
`observations/` directory in an external task workspace is a suitable location.

The published `klab-phase-observations-v1` envelope contains an outer
operation phase, the native nested phase records, controller-host coverage,
runtime-selection metadata, and generic snapshot-verification counters. It is
operational evidence only: the destination and records never enter prepared,
stage, Campaign, or model identities. Missing process counters remain `null`
with an availability reason; no accelerator measurement is inferred from host
information. Publication or sampling failure only writes a warning and does
not replace the command's original result.

A selected or authorized profile is a request, not execution evidence: unresolved
execution remains `null`, including failed stages and controller-only Campaign
operations. Observer setup/collection failures retain an empty record list with
an explicit availability reason rather than fabricated measurements. Campaign
admission also protects referenced tokenizers, experiment-plan run stores and
artifacts, and the native corpus artifact namespace before dispatch.

The [retained development evidence](research/development-evidence.md#capacity-execution-measurements-and-implementation-notes)
contains implementation details, measured comparisons and corrections from prior
workloads. Those workloads do not define the reuse interface or your experiment.

Source-snapshot warm proofs retain acquisition, source declarations and semantic
verifier helpers in their authority closure. The worker package's transport-only
convenience exports are excluded for this proof kind with a pinned AST-use
signature; a new use requires re-audit. Snapshot proofs authenticate warm hits
only in trusted registered stores. Authority changes, changed inputs, foreign
signatures and unsafe paths still force cold verification or rejection. This
does not authorize source acquisition or replace cold recovery authentication.
