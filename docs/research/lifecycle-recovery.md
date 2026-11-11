# Lifecycle recovery and scientific lineage

A Git commit records *intent*; a content digest records *which bytes*; a verified checkpoint and evaluation record show *what actually ran*. None by itself establishes model quality. Keep the scientific declaration closure (corpus project, source pins and rights, training config, ExperimentPlan, runtime requirement, EvaluationSuite, readiness policy, RecoveryManifest and ModelFamily) in Git. Keep downloads, releases, tokenizer files, prepared arrays, locks, checkpoint generations, receipts and logs in an adequately sized **external persistent state root**. Git branches, worktrees, checkout locations and the state-root path do not define scientific identity. Record the commit object ID and exact declaration hashes instead.

Campaign stage receipts and execution attempts keep their Campaign namespace.
Linked and standalone recovery recipes materialize deterministic artifacts under
the same selected persistent root; switching command families does not silently
select a second recovery store. Historical absolute availability paths remain
recorded and are freshly verified, never rewritten as though bytes moved.
Campaign status reports verified collected checkpoints separately from an
unbound planned logical run name; it maps a recipe checkpoint only when its
scientific binding is unambiguous.

```sh
# Normal development branch, in the existing checkout (no automatic worktree):
git switch main
git pull --ff-only
git switch -c feat/<task>
export KLAB_WORK_DIR=/data/klab
# Commit authored scientific inputs before corpus acquisition, tokenizer fit,
# experiment preparation or model training.
git add <declarations> && git commit -m "Declare scientific inputs"
uv run --locked klab research snapshot <plan-or-campaign> --json
uv run --locked klab recovery inspect <recovery.yaml> --json
uv run --locked klab recovery plan <recovery.yaml> --json
```

`research snapshot` is read-only and reports Git closure status, fresh prerequisite verification, declared evaluation and runtime, and available storage. `BLOCKED` is not a request to proceed blindly: inspect the reason codes and commit the corrected declarations. A mutation with `--allow-uncommitted-declaration` explicitly records exact dirty/untracked hashes but does not relax immutable artifact, runtime, approval or pinned source checks. `RecoveryManifest.source_commit` pins the earlier input commit, not the commit containing its own self-reference. A historic expected digest is a comparison target, not present-byte proof. There is no automatic Git commit, push, network retrieval, training or scientific decision.

Authored capability cards are scientific inputs and belong to the committed
closure. Checkpoint-bound result references and Surface Review bundles can be
declared before their output bytes exist: an absent result is unavailable and an
unfinished human review is skipped, not permission to invent evidence. Existing
result references are authenticated when consumed; invalid present review seals
remain hard failures. A pending review does not abort independent numeric gates.

## From declarations to a model decision

1. Declare and commit the corpus project, source acquisitions and source/redistribution rights; run Corpus Forge `corpus acquire`, `corpus build`, `corpus freeze` explicitly, then `corpus export` to obtain release-bound run/tokenizer configurations. Reopen the immutable release. A remote pinned source requires deliberate network permission; an unavailable source or publication right remains an external blocker.
2. Commit the tokenizer-selection evidence and complete ExperimentPlan/base run, including architecture, objective, token and step budgets, declared evaluation suite and runtime requirement **before** `experiment prepare`. Run `tokenizer train`, `data prepare` or the committed plan's variant preparation, then `experiment lock`. A verified preparation receipt and lock pin resulting content identities; a missing expected digest is `UNSEALED_RESULT`, not a reconstructed historical output.
3. Publish a compact verified reference, for example:

   ```sh
   uv run --locked klab research evidence export --kind corpus_release <verified-release-dir> \
     --declaration <recovery.yaml> --output <tracked-evidence/release.json> --json
   uv run --locked klab research evidence export --kind experiment_lock <verified-lock.json> \
     --declaration <recovery.yaml> --output <tracked-evidence/lock.json> --json
   git add <tracked-evidence> && git commit -m "Record verified scientific identities"
   ```

   Other accepted kinds are `tokenizer_selection`, `prepared_data`, `runtime_probe`, `checkpoint`, and `evaluation_index`. A tokenizer selection needs its real verified bakeoff selection receipt: a tokenizer file alone is not a selection. A runtime probe reference is an observation, not runtime acceptance. Evidence exports retain a verified digest and location reference; they do **not** commit payloads or certify scientific quality. Reusing the output path with a different scientific binding fails rather than overwriting prior evidence.

   For an archive, name the compact records explicitly in
   `RecoveryManifest.evidence` and commit them with the updated recipe. These
   later output records (including a lock's authenticated availability sidecar)
   belong to the **current** committed declaration closure but are not inputs
   pinned to the earlier `source_commit`; publishing output evidence does not
   require rewriting the input pin. The recipe never hashes itself. Changing
   scientific inputs does require a newly reviewed input commit and recipe.
   Name lifecycle receipts under their family node after the decision is issued:
   the node identity excludes operational receipt references, while changes to
   the checked-in family declaration still require reviewing its input pin.
   Supported explicit archive records are compact scientific references,
   evaluation-index/readiness references, canonical reconstruction receipts,
   reviewed lifecycle receipts, and a raw experiment lock together with its
   adjacent `<plan_sha>.availability.json`. Portable model archives include
   the selected run manifest, declared suite/source/result bytes, policy and
   human-review receipt (plus any sealed Surface Review bundle), not just their
   top-level index JSON. Missing required bytes block portable creation.
4. Bind the mandatory EvaluationSuite and ModelReadiness policy in a Campaign; verify corpus readiness separately from model readiness. Use `campaign plan`/`status` for historical stage state and fresh recoverability, then `campaign apply` for verified deterministic stages. Model dispatch requires a committed declaration, bound runtime/lock/approval where declared, and **explicit** `campaign apply <campaign.yaml> --execute-runs` (or `resume --execute-runs`) to enqueue new training. Without that flag it stops at `next_action: execute_run`. Losing a checkpoint or an old approval never grants permission to replace or retrain it.
5. Run the locked ExperimentPlan on a selected local/SSH worker and verify one immutable checkpoint generation. Run a declared suite, not an implicit collection of tests:

   ```sh
   uv run --locked klab evaluation suite run <suite.yaml> <run-id> \
     --checkpoint <step_N_gen_M> --runs-dir "$KLAB_WORK_DIR/runs" --json
   uv run --locked klab readiness model <policy.yaml> <suite-index.json> --json
   uv run --locked klab readiness review <suite-index.json> --reviewer <human-id> \
     --decision approve --note 'Reviewed checkpoint-bound evidence' --output <review.json>
   uv run --locked klab readiness model <policy.yaml> <suite-index.json> \
     --review <review.json> --json
   ```

   Suite roles separate numeric gates, diagnostics, descriptions, exploration and blinded Surface Review. Unsupported evaluators become `UNAVAILABLE`; absent review is `SKIPPED_REVIEW`, never a fabricated judgment. `INCONCLUSIVE`, `DO_NOT_ADVANCE`, `NEEDS_REVIEW` and `READY_FOR_NEXT_STAGE` have distinct meanings. One named reviewer is one person, not consensus. A numeric pass alone cannot substitute for required human approval.
6. Declare ordered parent/child ModelFamily nodes with pinned corpus, tokenizer, plan, architecture, objective, budgets, and optional checkpoint/index/readiness identities. Inspect `family show`, `family graph`, `family compare` and `family verify` before any action. Comparisons report factual ancestry/settings/evaluation identities, not causality. Use `family promote|reject|supersede <family.yaml> <node> --readiness <result.json> --evaluation <index.json> --approval <review.json> --note '<reason>'` (also `--successor <node>` for supersede). These human-reviewed immutable lifecycle actions are **not** ExperimentPlan weight-transition `promote`. Publish compact lifecycle receipts and name them under `lifecycle_receipts` in the family declaration so archives can inventory decisions after loss of local state.
7. Invoke `archive create <recovery.yaml> --mode thin --output <new-archive.tar>` only after reviewing the explicit inventory. `archive verify <archive.tar> --json` checks exact TAR membership and hashes without extracting. Thin archives contain small declarations/evidence and **identify unresolved external payloads**; they do not prove absent data. Portable archives require all locally verified required bytes, sufficient destination space/inodes and rights allowing publication. A `metadata_reconstruction_only` corpus or partial `external_required` recipe cannot become portable. Archives never appear automatically. The archive and its exclusive sibling stage use ordinary file permissions under the caller's umask. An abrupt exit may leave a `.<output-name>.*.tmp` stage; remove only the stage belonging to a writer you have confirmed has stopped, never an active writer's stage or the published archive. If directory durability fails after publication, the error identifies the complete archive that was retained; verify it before deciding how to proceed.

Study reports in `local-reports/` and capability results in `runs/<id>/evaluations/`
also use exclusive sibling stages (`.architecture-*.json.*.tmp` and
`.capability-*.json.*.tmp`). Readers ignore these stages. After an abrupt exit,
remove only a stage belonging to a writer confirmed to have stopped; retain
published result files. A directory-fsync error after publication identifies the
complete result that was retained but whose directory entry may not be durable.
An identical replay syncs the directory before reporting success and preserves
the existing result's bytes.

New archives use `archive-index-v2`; verification still accepts historical v1
archives. For each available, verified corpus release, the index retains a compact
`corpus_identities` explanation: project declaration and ID, every normalized
source declaration (including rejected sources without snapshots), source-ID to
snapshot-SHA bindings, snapshot adapter/version/module and file provenance, the
full original build identity payload, and the original release manifest/identity.
Local acquisition paths are omitted from declaration digest payloads exactly as
in the original acquisition algorithm. No build, snapshot, or release identity
algorithm changes, and no source bytes are added to thin archives.

`archive verify` checks these declared SHA-256 payloads and their cross-bindings,
including the archived release manifest, after relocation and without accessing
original sources. The explanation records
`verification_scope: declared_digest_payloads_only`; its metadata-only Python
validator, `klab.corpus.identity.verify_corpus_identity`, also reports
`source_contents_verified: false` and `release_contents_verified: false`.
These checks explain the declared identity; they do not verify unavailable file
contents, reconstruct a corpus, or establish scientific validity. File digests
and algorithm provenance remain declarations until the corresponding bytes are
available and checked by the full domain verifier. A wholly replaced, internally
consistent identity chain still needs an independently trusted release SHA.
Missing releases retain their external references rather than invented identity
explanations. Metadata and index size limits also apply to these explanations.


Portable verification reopens the bundled recipe and model graph and requires
their pinned payload closure. Rewriting the outer TAR index, relabeling a
checkpoint manifest, or replacing its weight bytes cannot substitute for inner
artifact verification. Reviewed lifecycle actions must still satisfy their
approval/readiness and successor-lineage rules. Thin unresolved references are
not permitted in portable archives. Streaming verification bounds both individual
metadata records and aggregate retained metadata.

## What happens if the persistent state root disappears?

```sh
export KLAB_WORK_DIR=/data/klab
uv run --locked klab recovery inspect <recovery.yaml> --json
uv run --locked klab recovery plan <recovery.yaml> --json
uv run --locked klab research snapshot <plan-or-campaign> --json
# Only after checking source availability, storage and network permission:
uv run --locked klab recovery reconstruct <recovery.yaml> --json
# For pinned remote sources, explicitly add --allow-network after rights review.
uv run --locked klab campaign reconstruct <campaign.yaml> --json
uv run --locked klab campaign status <campaign.yaml> --json
```

The first three commands must not create the root or scratch. `reconstruct` only replays declared deterministic corpus acquisition/build/freeze, export, tokenizer training, data preparation and experiment lock, comparing rebuilt bytes to expected digests. It **never** trains, evaluates, approves or promotes a model. `UNSEALED_RESULT` needs a new explicitly reviewed and committed identity; a different digest is a discrepancy, never an alias for the old release. `campaign reconstruct` cannot recreate lost historical Campaign receipts and does not dispatch training. Fresh recoverability differs from historic last-committed stage outcome.

Recoverable: committed local inputs and deterministic outputs with verified dependencies. Externally required: inaccessible pinned downloads, explicit tokenizer/model/runtime/evaluation decisions not yet authored, or unresolved redistribution rights. Nonreconstructable: a lost checkpoint generation and optimizer state unless a verified independent copy exists. A family parent checkpoint does not recreate a lost child or make unrun planned nodes complete. Restore such bytes from a verified portable archive if rights allow; otherwise retain their pinned missing identity, do **not** silently retrain under it. A deliberately relocated state root needs new location binding/manifest and re-verification of original referenced bytes—not editing old receipts or assuming paths moved. `scratch/` and optional `cache/` are disposable, never scientific evidence. Explicit old output/cache/run paths retain their original meaning, including legacy `klab-work/`; selecting an external root does not migrate them.

A recovery declaration can be partial. If source access, rights, tokenizer,
model configuration, runtime or evaluation bindings are absent, report those
prerequisites as unavailable. An expected release digest alone cannot recover
model weights or authorize new training.

## Explicit historical Corpus Forge implementation replay

Legacy Corpus Forge identities include exact implementation-file hashes:
acquisition snapshots bind `corpus/acquisition.py`; builds bind
`corpus/pipeline.py`, `corpus/project.py`, and `corpus/provenance.py`, with
`corpus/rights.py` for schema 2/3 and `corpus/large_build.py` for the large
schema-3 LM path. Freeze carries those identities and verified file/stage
inventories into the release. Export binds release, configuration and split
bytes rather than adding its own implementation-file hash. Consequently,
replaying acquisition alone is insufficient when historical build bytes differ.

`recovery inspect` and `recovery plan` compare the locally available
`RecoveryManifest.source_commit` implementation with the current implementation
before acquisition. Their per-corpus `implementation` record is operational:
it lists historical/current component hashes and affected stages. Matching
implementation permits normal deterministic reconstruction; a difference reports
`PINNED_IMPLEMENTATION_REPLAY_REQUIRED`, and unavailable historical implementation
reports `MISSING_IMPLEMENTATION`. These read-only calls do not download sources,
materialize a tree, create an environment, or create the selected state root.

Review that inventory and source rights before explicitly authorizing replay:

```sh
export KLAB_WORK_DIR=/data/klab
uv run --locked --extra cpu klab recovery inspect recovery.yaml --json
uv run --locked --extra cpu klab recovery plan recovery.yaml --json
# Explicit code-execution authorization; --allow-network separately authorizes sources:
uv run --locked --extra cpu klab recovery reconstruct recovery.yaml \
  --replay-pinned-implementation --allow-network --json
```

Replay is execution of reviewed historical code, **not a sandbox for untrusted
Git commits**. It exports exact Git object bytes outside the checkout, never a
worktree or dirty working-tree copy. The source tree has a verified inventory
and is read-only after publication where supported. Existing different bytes
fail closed; they are not repaired in place. A dedicated uv-managed data/build
environment is separate from both the checkout `.venv` and registered model
runtimes. Its dependency lock, Python and uv identities are recorded.

The subprocess imports the exported historical source: its producing functions'
`__file__` paths point at historical bytes. It executes acquisition, build and
freeze historically; the current interpreter independently verifies the
resulting snapshots/build/release. No current producer receives substituted
historical hashes. Operational receipts distinguish the current orchestrator,
historical source tree/environment, and each artifact's recorded provenance.
Existing partial current-code acquisition stays untouched. Historical products
live under `$KLAB_WORK_DIR/replay/work/<commit>/`; recovery can resolve the
verified release there without rewriting earlier location receipts.

An optional `expected_build_sha256` on a `corpus_release` recovery step adds a
stop-before-freeze gate. `expected_release_sha256` remains the final immutable
gate. Supply reviewed historical expectations; neither field is automatically
updated from a differing result. Failures retain operational receipts and partial
outputs. A successful corpus replay does not resolve other missing scientific
decisions, authorize training, or establish model quality.

When a replay attempt fails after publishing its receipt, `recovery reconstruct
--json` still exits with status 1 and returns `status: "BLOCKED"` and the original
reason code. Its `failed_replay` object identifies that exact attempt:

```json
{
  "receipt_path": "/absolute/work/replay/receipts/<attempt>.json",
  "record_sha256": "<canonical receipt record SHA-256>"
}
```

Pass `receipt_path` directly to
`klab.recovery.implementation_replay.verify_replay_receipt(Path(receipt_path))`
and compare the returned `record_sha256` with this reference. This is the
receipt's canonical record digest, not a hash of the pretty-printed JSON file
or an expected corpus identity. Verification does not retry producers or scan
for attempts. Keep the receipt and its referenced source/log/artifact evidence
available for independent verification. Python callers receive a `ReplayFailure`
(a `ValueError`) with `receipt_path`, `record_sha256`, and the original exception
as its cause, for both standard and ancestry replay. Failures before receipt
creation or during receipt publication do not claim a published reference.

Replay integration tests install real locked environments. Use adequately sized
external scratch rather than a small `/tmp` tmpfs, for example
`TMPDIR="$KLAB_WORK_DIR/scratch/replay-tests/tmp"` with pytest's
`--basetemp="$KLAB_WORK_DIR/scratch/replay-tests/pytest"`. Create the temporary
directory first; pytest owns and replaces its explicitly selected base directory.
The fixtures remove their dedicated environments after each test while retaining
small source/artifact/receipt diagnostics.

### Bounded ancestry replay and archived acquisition locks

The opt-in Python replay API accepts a historical Git project, an explicit
external corpus work root, a `build` or `release` phase, and a verified parent
receipt with exact unchanged/changed snapshot maps. This emits version-2
operational receipts; ordinary recovery reconstruction and version-1 receipts
retain their original behavior. An ancestry request requires an exact build
gate; a release phase additionally requires an exact release gate.

Historical declarations are authenticated against their Git objects before
dependency installation. Imported directories must preserve declaration bytes,
snapshot identity, complete file inventory, manifest bytes and original adapter
provenance from the recursively verified parent. The selected historical
producer must accept every imported snapshot through its immutable reuse path
before acquisition; source-scoped guards refuse inherited reacquisition. New
snapshots retain the selected producer's acquisition hash. No manifest is
re-stamped with a later producer hash.

Acquisition success archives the **unaltered** lock bytes before build, including
when a later gate fails. `verify_acquisition(project, work_root,
lock_path=archived_lock)` applies the normal project, canonical snapshot path,
file inventory and receipt checks without replacing the active lock.
`verify_replay_receipt` authenticates archived locks, logs, source trees, parent
links and artifacts. An intermediate build-only receipt remains verifiable
after a later acquisition updates the same corpus workspace's active lock.
Build-only success records a real build and `release: null`; it never freezes.

Some historical recovery procedures used task-specific coordinators. Their
[retained records](development-evidence.md#task-scoped-ancestry-replay) describe
those exact attempts, not a general DSL route. New recovery declarations use
the native [snapshot inheritance contract](#declared-snapshot-inheritance) below.
Partial inheritance still needs explicit acquisition authorization for missing
sources; production replay remains a separately allocated execution, not a
missing inheritance interface.

### Future identity design boundary

Historical schema-v1/v2/v3 semantics remain unchanged. A future separately
versioned Corpus Forge identity should use deliberately reviewed semantic
algorithm/version IDs in artifact identity, retaining exact commit/file hashes
as implementation provenance. Algorithm changes require explicit versions and
deterministic regressions. Python AST equality is incident evidence only, not
a semantic identity algorithm. This replay mechanism does not migrate old
artifacts or introduce a new corpus identity schema.

### Declared snapshot inheritance

A `corpus_release` recovery step can bind verified parent replay evidence and
exact source IDs through `snapshot_inheritance`:

```yaml
snapshot_inheritance:
  parent_receipt: replay/receipts/parent.json
  parent_receipt_sha256: <SHA-256 of the exact retained parent receipt bytes>
  snapshots:
    source_a: <unchanged snapshot SHA-256>
  changed_snapshots: {}
```

The parent locator is relative to `--work-dir`, under `replay/receipts`; it is
operational evidence, not a Git-relative source declaration. Pin both
`expected_build_sha256` and `expected_release_sha256` on the step. The maps
must cover the parent's snapshot IDs exactly; `changed_snapshots`, when used,
records the **old** identities whose declarations must change. New source IDs
remain governed by ordinary acquisition permission. Inherited sources must use
immutable Git, Hugging Face, or Wikimedia adapters; effect-bound projects use
the canonical corpus acquisition interface instead.

Use the existing `recovery inspect` and `recovery plan` commands first. They
cold-authenticate the parent receipt, acquisition closure, selected snapshot
bytes, exact source declaration bytes and expected identities. Then explicitly
invoke `recovery reconstruct MANIFEST --replay-pinned-implementation`. Native
replay copies verified snapshots into a fresh owned work root, preserving their
manifest bytes and identities; complete inherited inputs get an acquisition
lock for offline reuse. Changed-source imports preserve prior snapshot evidence
for the native worker's checks. Partial inheritance still requires separately
authorized acquisition for missing sources; `--allow-network` is never implied.
Conflicting existing work, symlinks, tampered parents and mismatched declarations
fail closed. Producer replay is a separate execution allocation: the offline
code tests cover import/staging, worker inheritance preflight and CLI wiring,
not a production release replay.
