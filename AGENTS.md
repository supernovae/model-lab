# Kybern Model Lab agent invariants

## 0. Default to lab mode

Most work is a **normal authorized experiment**: a local or owned-SSH run that
the user asked for, on hardware they already control, inside its resource
envelope. Use the fast path in the [activated environment](README.md#first-run) and do not write ceremony:

```sh
klab try DELTA.yaml --vs BASELINE.yaml
klab report TRY_ID
```

`try` derives the candidate through native `config derive`, trains both arms
with the native trainer, scores both on the same held-out split and writes one
sealed record (config, seed, code revision, data identity, resource envelope,
result). See [lab mode](docs/lab-mode.md). In lab mode, do **not** write
proposals, bindings, ledgers, allocations or stop/result documents, do not lock
ExperimentPlans, and do not request Campaign approvals. Iterate: read the
record, change one setting, try again. A lab record is evidence of one local
comparison, not a release, promotion or scientific conclusion.

For a new user, follow [lab mode](docs/lab-mode.md) with an explicit local recipe.
The optional [Workshop TinyStories example](docs/tinystories-microlab.md) is owned
outside Lab. Prepare its declared inputs, then use `try` and `report`. Keep advanced
continuation, matrices and release instructions in their own guides. Run IDs
belong to `probe`/`explore`; sealed record IDs work across `report`/`compare` and
`probe`/`explore` (the record's candidate checkpoint, never a newer `latest`).
`report`/`compare` require a record rather than guessing one from a run.
See [result lookup](docs/result-lookup.md). Standalone
`probe` inherits the saved backend unless explicitly overridden; the CPU extra
installs a framework but does not override a saved CUDA/ROCm backend.

Use `report RUN_ID --kind run` for verified run evidence and saved triage together;
JSON keeps them under separate `evidence` and `triage` keys, plus an optional
`surface_review` overlay selected by strict `--config` YAML.
`report --kind research` shows declared status and next
tests under separate `status` and `next` keys. See [result reports](docs/result-reports.md).
These views never train, create a run store or grant release/compute authority.

Every try also runs the fast [probe battery](docs/probe-battery.md) tier; use
`klab probe RUN --vs BASE --backend cpu --config standard --json`
(or `--config full`) for more. Act on
`probe.verdict.action`: `abandon` (a hard probe failed, including a NaN/inf
loss reported as a numerical failure: drop or fix the idea),
`tweak` (change one setting; `suggestion` names what the failing probe
implicates), `rerun` (the battery is incomplete: `verdict.missing` names each
check that produced no evidence, such as lm-eval not installed, a probe error,
or a cancel/signal/resource/OOM stop; fix it and re-run, never read it as a
pass), `escalate` (re-probe at `next_tier`), `longer_run` (all tiers pass and loss
improved: worth a larger budget, still in lab mode), `compare` (add `--vs`) or
`report` (a completed final verdict: terminal, report it as is; a final
battery with any missing evidence, with or without `--vs`, says `rerun` the
same final declaration unchanged unless a NaN/inf or hard failure decided it).
Probes are screening signals, not proof that an idea is useful. Selecting on
the held-out probes again and again turns them into development data, so any
claim that an idea is better needs a separate, untouched final evaluation
(fresh split, longer run, more seeds). Iteration verdicts use only the held-out
item split. The **held-back final split** is that untouched evaluation for
probe items: never select on it. Only `klab probe CANDIDATE --vs BASE
--config final` reads it (`try` and ordinary probes cannot; a test
enforces this), once, for the final verdict on a candidate you have already
chosen; never use it to compare or rank candidates, and never re-run it after
changing the idea. Final results never enter ordinary `compare`, enrichment,
Pareto or dashboard history; `klab compare --final A B` reports them
side by side only. A single-seed "beyond noise" loss gain is within-run eval
noise only: confirm it with paired seeds before claiming a win. A try whose
comparison is `NOT_COMPARABLE` is never promoted, whatever its probes say. Do not edit probe
items, thresholds or prompts to make an idea pass, and do not iterate against
the dev split: `guard.overfit_suspected` means you are fitting the probes; change
the idea instead. Any suite change needs a `SUITE_VERSION` bump, and results are
comparable only within one suite digest.

To place a result on a known curve, run `klab compare RESULT
--references` (records only, no compute). It compares a metric only within one
comparison group (held-out loss: same `eval_group`; lm-eval: same
`benchmark_group`; fact recall: same `item_group`) and otherwise reports
`not_comparable` with the reason or `missing_evidence` with the command that
produces it; never compare such pairs by hand. Reference models (`ref:NAME`)
are scored on the text-level probes only (both fact recalls and lm-eval;
`klab probe ref:NAME --config full`, optional `reference` + `lmeval`
extras) and are never a `--vs` baseline. Declaration `lm_eval_tasks`/`lm_eval_limit`
make a new benchmark group, so such results never compare with the default
slice.

To look inside a small checkpoint (≤ 60M parameters), run `klab explore
RUN [--config explore.yaml] [--json]` (CPU by default; `--runtime-profile`/`--runtime`,
`--resource-envelope` use the same native loaders and runtime preparation as
`probe`, whose direct profile/envelope paths are selected through its validated
declaration). It
reports the architecture, per-token loss and top-k, attention maps, weight
statistics, expert routing and memory lookups, cached under `LAB/explorer`. It is descriptive, not a verdict.
`klab dashboard` shows the same data for people (Home, Experiments,
Models, Behaviors, Explorer; see `docs/dashboard.md`).

Write proposal, binding and stop documents **only** for:

- **release runs**: results that will be promoted, published, cited as a
  scientific milestone or used as a parent for a release family; these keep the
  full ExperimentPlan/Campaign path with approvals, locks and reconciliation;
- **paid compute above an explicit budget** the user has stated (rented GPUs,
  hosted providers). Spending always needs the user's explicit authorization;
  never infer it from a plan, a missing receipt or an earlier allocation.

If you are unsure whether a run is lab or release, run it in lab mode and say
so; promoting a result later is a separate, reviewed step.

## 1. Establish scope before execution

Read `README.md`, relevant nearby guidance, `ROADMAP.md` for feature direction, and
`docs/research/` for scientific work. Inspect Git state/history before editing;
preserve other work and use an isolated checkout when experiments have pinned
another checkout. Distinguish code, fixtures, experiments and evidence.
Kernel Memory Lab was retired from the active showcase on 2026-10-18. Its
[bootstrap](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/BOOTSTRAP.md) and retained status
are historical context, not a current release assignment. Fresh rerun ideas belong
to the coordinated backlog with new identities, current frameworks and separately
authorized scope. Historical allocations grant no runtime or spending authority;
this retirement does not release existing runtime holds or reset accounting.

## 2. Use the native, current public path

Reuse native dispatch, renderers, verifiers and monitors described in
[the iteration guide](docs/iteration.md); `try`/`report` are the native lab
path. For research launchers, consult the
[contract audit](docs/research/launcher-contracts.md) before replacing a specialized
protocol with generic dispatch. Mechanism lessons live in `learn`; research
scaffolds emit native study declarations. Keep one current public command/schema
path; update callers together
instead of adding executable version forks, compatibility aliases or a second
runner. Until a production release is tagged, adapt experiment plans, recipes
and models to current Lab behavior; do not add backward compatibility or version
adapters to preserve old execution behavior. Historical evidence may be read and
audited without making old artifacts executable or treating historical claims
as current guarantees. Follow the [migration and review policy](CONTRIBUTING.md#pre-production-migration-policy),
including new experiment identities and the release-tag boundary.
Do not reimplement hashing, checkpoint selection, accounting or workflow
state in task scripts. Before building new training, evaluation or quantization
machinery, check existing libraries (nanoGPT/modded-nanogpt, litgpt,
lm-evaluation-harness, llm-compressor) and wrap them. Record a genuinely
missing operation with inputs and failure criteria in a GitHub issue. Use
GitHub pull requests and their checks for implementation, review and CI;
[ROADMAP.md](ROADMAP.md#where-work-is-tracked) summarizes feature direction.

For release corpus preparation, use the
[canonical corpus-to-bundle interface](docs/corpus-preparation.md) and its
authenticated phase map; lab mode may use any dataset already available
locally, with its digests recorded in the try record.

## 3. Preserve identities, evidence and scientific settings

Never rewrite historical hashes, normalizer identities, receipts, failed runs or
acceptance criteria to fit new code. Preserve negative, censored, interrupted and
unavailable observations, including interrupted lab tries. A changed protocol
needs a new identity; promotion requires review. Paths are locations, not
scientific identities. Keep preparation, updates, evaluation, generation and
reporting measurements separate. Do not silently change effective batch, data,
tokenizer, seed, architecture, precision, token budget or optimizer: a lab delta
states its change explicitly. Never change held-out evaluation data or settings
inside a comparison; `try` refuses such comparisons as `NOT_COMPARABLE`.
Fixtures establish wiring only; successful commands are not scientific
conclusions.

## 4. Bound resources and retain owned outputs

Use a persistent work root outside Git: global `--work-dir` overrides
`KLAB_WORK_DIR`; otherwise use absolute nonempty `XDG_DATA_HOME/klab`
or `~/.local/share/klab`. Lab tries live under `WORK_DIR/lab`. Respect
explicit legacy locations. Only disposable fixture scratch belongs in `/tmp`.
Use the [location/runtime selection guide](docs/location-runtime-controls.md)
before changing selectors. Work defaults, historical lab/run readers and
controller stores have distinct contracts; runtime IDs/profiles select an
interpreter, while backend selects the workload. Keep `runtime env`, including
its global `--work-dir` prefix, Torch-free and preserve source-relative
declaration paths.
Check bytes, inodes and expected growth before expensive work; select a native
resource envelope for anything large (`try`/`probe`: declaration
`resource_envelope`; other commands retain their direct `--resource-envelope`
selector). Do not add arbitrary short wall-time limits to normal experiments;
set a wall limit only when the user asks for one (`try`: envelope
`lab_try.max_wall_seconds_per_arm`). Stop runs safely (Ctrl-C, SIGTERM or the printed `CANCEL` file), never
by killing a process mid-checkpoint. Never mutate an active run, reset a spent
ledger, prune unrelated data, or infer safe shutdown from a missing receipt.
Check in only small durable declarations, summaries and evidence references.

## 5. Verify the changed behavior and its actual callers

Use the locked uv environment (`uv run --locked ...`), repository Ruff settings,
and the documented vendor environment for accelerator work (`--no-sync` where
already provisioned). Do not install packages or create another environment
without authorization. Host OS does not identify the accelerator. Inspect test
fixtures and transitive calls before choosing a bounded explicit test selection;
expand only for a changed behavior or required gate. While iterating, run
the fast suite (`-m 'not slow and not mps and not mlx and not cuda and not rocm
and not xpu and not network'`) plus targeted slow tests for the touched area;
leave the full suite to the nightly workflow (docs/test-speed.md). Prefer deterministic
identities, counters, transitions and failure checks to wall-clock assertions
(the generous lab-loop timer is the deliberate exception).

Changes to release corpus preparation keep connected coverage: load the
checked-in production templates, use public baseline rendering/initialization
and canonical phase arguments, and reach a cold prepared bundle on tiny
fixtures, declaring fixture substitutions explicitly.

Before pruning a test, map its useful guarantees to retained coverage,
including provenance, zero-network reuse, review/sample binding, protected
families, collisions, accounting and shutdown. Review `git diff --check`, Git
status and current documentation examples before handoff; report exact passed,
failed and unrun checks.

## 6. Keep changes and collaboration reviewable

Use one agent for small tasks; substantial independent work may use parallel
helpers with narrow owned files and concise evidence-backed handoffs. Avoid
overlapping device jobs. Let the user choose models, reasoning effort and
concurrency. Keep local commits narrow; publishing, history rewriting and remote
deletion require authorization. Lab records already capture commit, dirty state
and effective config; release runs additionally run from a fixed tested
revision. Keep scientific milestones in research records and proposed code work
in GitHub issues. Keep `ROADMAP.md` a short directional snapshot, not a second
checklist.

## 7. Avoid sprawl in code and tests

- **Reuse the shared primitives; never copy them.** Hash JSON with
  `canonical_digest`, encode with `canonical_json` and hash files with
  `sha256_file` (all in `klab.training.manifest`). Check filesystem trust
  with `verification_proofs.writable_by_others`/`_trusted`. Do not add a
  private `_digest`/`_sha`/`_write_json` wrapper or inline
  `hashlib.sha256(canonical_json(...))`. Never import another module's
  `_private` helper: promote it or use the shared one.
- **One builder per fixture.** Shared test inputs live in `tests/*_fixtures.py`
  (`lab_fixtures.write_smoke_inputs` for lab-loop configs). Parameterize them
  instead of copying, and keep derived settings valid when you shrink a budget
  (for example `warmup_steps < max_steps`).
- **Test outcomes, not trivia.** A test should prove that a lab path works
  (`try` → record → `report`/`probe`) or that a real rail refuses bad input.
  Do not test prose, help text or private helper shapes. Do not keep tests for
  retired contract versions. Do not bundle unrelated assertions into one
  grab-bag test.
- **Size tests for an ordinary laptop.** Use the smallest budget that proves the
  behavior. A cancel or signal test only needs training still running, not an
  hour of steps or tens of GB of storage preflight. Never depend on CI's umask,
  user, disk or `HOME`, and always isolate `KLAB_WORK_DIR`.
- **Pruning is a change too.** After deleting tests, imports or helpers, run
  `ruff check .` (F821 catches names a removal broke), the touched test files
  and the `lab-loop` tests before handoff. A red `main` blocks every other PR.
- **No silent degradation.** An optional cache or proof that cannot publish
  records why. If a rail refuses ordinary local setups (default umask,
  user-private groups), fix the rail. Do not paper over it in tests.


## 8. Report the model people can inspect

When discussing current training, include the dashboard URL, configured run-store
path, model name and exact run ID. Use `/training?store=<URL-encoded-path>&run=<ID>`
for a stable selection. Verify the store and current run from fresh runtime evidence;
do not call the newest record active merely because it sorts first. Report recorded
status and observation time, mark stale/unverified activity explicitly, and say
idle/completed when no training is verified. A stored `running` status alone is not
a heartbeat. Never restart training to make the dashboard look active. Keep training
progress distinct from checkpoint-specific benchmark results and unknown capabilities.
