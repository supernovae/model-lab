# Compiled native preparation bindings

The first LAB-74 slice compiles `tokenizer_train` and `data_prepare` bindings from
the existing `CampaignPlan`. `compile_preparation(plan)` is a pure projection,
not a scheduler: `CampaignPlan` still validates producer kinds, dependencies and
cycles and supplies the stable topological order. Other stage kinds retain their
native routes. No new CLI, external recipe workflow profile or execution grant
is introduced.

Each immutable operation records its stage ID, operation version, declared config
reference, dependencies, typed input ports and output contract. Tokenizer and
snapshot inputs point to producer stages' artifact outputs. Future output bytes
have no digest or availability claim at compile time. Compiling opens no configs,
artifacts, stores or runtime metadata and does not import optional computation.
It does not certify content, resource availability or admission.

At dispatch or receipt verification, CampaignEngine selects the compiled
operation, loads the native config and uses its existing `_upstream` verification
to authenticate each selected producer. It copies those outputs and locations
into an immutable `PreparationRequest`. Request validation retains the native
config-versus-snapshot/tokenizer path checks. The adapter receives neither an
engine nor the entire stage/row map, and cannot choose another producer or
invoke a downstream workflow operation. It calls native tokenizer training or
packing and native artifact verification. The engine translates its typed result
back to the existing receipt shape.

The preparation operation consumes an already authorized execution context;
compilation and a well-typed request do not authorize execution. Existing apply,
claims, resource, provenance, runtime and attempt gates remain in force. The
snapshot acquisition branch is unchanged in behavior and remains in the native
orchestrator, including interrupted-journal resume and verification of previously
published snapshots. No acquisition path is added to compilation.

Two supported graphs exercise the same data-preparation port:

- `tokenizer_train → data_prepare`: use a newly produced native tokenizer.
- `artifact_reference(tokenizer) → data_prepare`: reuse a declared tokenizer.

The choice belongs to the authored graph and compiled binding, not an adapter
branch or a dispatcher patch. Optional snapshot edges remain explicit. Extra
control dependencies remain in the native graph and are not mistaken for data
ports.

## Compatibility and dependency boundary

The projection and requests are internal immutable values, not additions to v1
serialized campaign plans, receipts, configs or experiment locks. Native stage
IDs, declaration digests, result shapes and scientific hash algorithms remain
unchanged. Golden declaration digests were captured using CampaignPlan from
`b264f41ae560af5767cbcbd3fdc861edce43c914`. Installed source and verifier closure
digests naturally change with implementation; prior proofs still require the
existing authority check. No pinned exclusions or historical evidence is edited.

Readiness-policy computation imports are now local to its existing operations so
CampaignPlan parsing and preparation compilation do not load corpus/data code.
The schema and readiness computation are unchanged. Its token-only no-scan test
now fences the ancillary scanner itself instead of a former eager import alias.
The archive crash injector now intercepts only publication to the final archive
path, allowing newly deferred dependency imports to perform their own link probes;
all three publication crash assertions remain intact.

This is a narrow dependency-direction guarantee for the migrated adapters.
Native artifact/source-compatibility checks remain intact. The broader trainer,
packing and tokenizer dependency inventory in [document contracts](document-contracts.md)
remains open; this slice does not claim repository-wide component separation.

## Change-impact test map

| Boundary | Evidence |
| --- | --- |
| Native graph → typed projection | Two distinct graphs, native ordering, missing/wrong edges, cycles, duplicate producers, version and port refusals; campaigns with no preparation stages |
| Compiled binding → engine request | Dispatch spies reject decoy stage config/producer fields; immutable copies prevent row mutation changing a request; zero/two upstream output refusals |
| Request → native operation | Config/path/type checks; exact native computation and artifact-verification spies; native tiny synthetic train/reuse parity |
| Metadata-only compilation | Fresh-process import/write/network/process fences, including forbidden config/artifact reads |
| Legacy formats and lifecycle | Pre-change declaration hash goldens, unchanged result shape, native replay, recorded versus reverified identity mismatch, prepared-output location refusal before artifact access, parent binding, snapshot interruption and verifier-authority checks |
| Readiness / runtime / recovery | Existing focused policy, runtime, campaign recovery tests; normal safe/fast/tiny CPU lab-loop CI |

Focused mutation checks remove each of the replay identity, upstream cardinality
and prepared-output location guards in isolation. The new tests detect every
removal (two, two and one failing cases respectively); source is restored before
normal validation. These checks exercise refusal boundaries without model work.

LAB-74 is not complete: typed bindings for other operations, full compiled-plan
coverage, broader dependency-direction gates and recorded runtime-capability
preflight remain. The first [LAB-73/PR B workflow slice](workflow-recipes.md) adds explicit external
prepared-input bindings and native lock publication. Preparation-graph context,
evaluation closures, admission and evidence linkage remain open. Actual external CPU/GPU portability proof remains separately
admitted later work. Historical seed and mixture identity decisions are unchanged.

[Portable preparation planning](preparation-recipes.md) now supplies inventoried
external graphs/configs to this same compiler. It adds no engine admission or
execution route; the remaining execution/provenance boundaries still apply.

For a smaller whole-document window of an already encoded all-token local corpus,
use [verified prepared prefixes](prepared-prefix.md) to avoid encoding it again.
### Unpadded whole-document n-gram pilot

For an all-token CPU/PyTorch pilot, set
`training.matched_stories: {layout: story, targets_per_update: 4096,
unpadded_documents: true}`, `training.micro_batch_size: 1` and
`evaluation.batch_size: 1`. Both training and default validation use one complete
EOS-terminated document per row, with a fresh virtual EOS input and fresh model
history. No other document shares its attention or n-gram suffix. Rows end at
the last valid target, so there are no padding slots. The native shared target
groups and cursor still determine updates, budgets and exact resume.

This mode accepts `memory: none` or `memory: ngram`. It refuses overlength
documents and supervision/byte/semantic sidecars instead of silently splitting
context or discarding masks. Select complete source records within the desired
context and retain the original records for a separate longer-context identity.
Use `MatchedStoryDataset.horizon(max_tokens)` for the exact `max_steps`;
the token budget must end at a document boundary. An interior cutoff is rejected.
Changing 4096 to 8192 does not require tokenizing those same documents again.
This is not multi-document concatenation, assistant-only SFT, or a guarantee of
GPU performance; microbatch one and variable shapes should be measured on the
intended runtime before scaling.

A reused tokenizer can bind `tokenizer.origin` to either an existing frozen
snapshot dataset or a `mixture_declaration_path` plus its
`mixture_declaration_sha256`. Both forms also require `sha256` and
`manifest_sha256` for the tokenizer files. The mixture declaration form checks
its native release and fit-tokenizer closure, including an explicitly recorded
post-run observation when applicable. It does not replay the consumer mixture
or assert that its schedule completed. A post-run observation remains bounded
metadata/current-byte evidence, not a historical output-digest attestation.
`klab tokenizer verify CONFIG --json` checks the actual vocabulary and
special IDs, and `klab data prepare CONFIG` uses the same origin check.
Existing service verification context is required for authenticated relocated
releases; do not replace keys or rewrite historical receipts.

### Continue variable document groups and append new data

`train TARGET --extend-budget IMMUTABLE_GENERATION` now supports terminal matched
story runs. Budget caps must equal complete native update boundaries; derive
counts with `MatchedStoryDataset.targets_for_updates` / `horizon`, rather than
multiplying steps by sequence length. AdamW moments, RNG, cursor and cumulative
counters are restored. The target's `optimizer.decay_steps` must retain the
parent's existing decay horizon (or the parent's terminal step on its first
extension). Further extensions keep that same horizon and floor; they do not
restart warmup.

Adding `--append-data` admits a larger, already prepared `local_text` stream for
an unpadded, single-pass parent. The target must preserve every trained token in
its original order, every prior update boundary, and identical validation tokens.
Set `training.matched_stories.frozen_update_ends` to the old stream length, plus
any earlier frozen phase ends. Those document-aligned boundaries prevent the
parent's short terminal update from being merged with newly appended records.
The native validator rejects a missing boundary, changed prefix/heldout stream,
changed grouping or other scientific settings, and records old/new token hashes.
The new single-pass budget includes the consumed prefix; training starts after
that prefix and processes only the new records.

The explicit operation is full-state continuation, not weight promotion. After
its first checkpoint, ordinary `--resume` uses the expanded owned data snapshot;
do not repeat `--append-data` without another terminal budget extension. A code
or runtime change still requires the existing explicit drift policy. Arbitrary
corpus replacement, masked SFT and multi-pass append transitions are not admitted
by this append-only path.

To start a new data phase before the source's planned budget finishes, add
`--append-from-checkpoint` alongside `--extend-budget` and `--append-data`.
This explicit transition admits only a verified complete-update checkpoint from
a single-pass whole-document parent. The new stream must preserve every consumed
token, every consumed update boundary and the validation stream. Its caps must
advance beyond the checkpoint; they may be smaller than the parent's unconsumed
plan. Freeze the consumed token boundary in `frozen_update_ends`. The decision
records consumed counters and the number/location of deferred source tokens.
Original source data and checkpoints are retained. Optimizer, RNG, scheduler and
learned memory remain full-state; this is not a weight-only promotion or an
objective change. Without the new flag, append-data still requires a terminal
parent. Coordinate device/source ownership before launching the child; preparation
and read-only checkpoint validation do not stop or change a live parent.

A separately measured runtime configuration can be composed with a budget/data
extension using `--allow-runtime-optimization`. Both validators run: the extension
must preserve the authenticated consumed prefix, update boundaries, heldouts and
scheduler; the existing runtime policy permits only its supported dense-attention
execution changes and preserves effective batch and ordinary validation batch.
Changed optimizer/scientific settings remain refused. The child records separate
budget-extension and runtime-optimization decisions. Full optimizer/RNG state is
retained, while changed kernels may change the numerical trajectory. This flag is
explicit consent to a supported transition, not evidence that it is faster or
correct; choose settings only after the separate comparison results are reviewed.
