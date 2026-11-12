# Working in Kybern Model Lab

Instructions for agents changing this repository or running experiments with it.

## 1. Default to lab mode

Most work is an ordinary experiment the user asked for, on hardware they
control. Use the fast path in the [activated environment](README.md#install):

```sh
klab try DELTA.yaml --vs BASELINE.yaml
klab report TRY_ID
```

`try` derives the candidate from the baseline, trains both arms, scores them on
the same held-out split and writes one sealed record: config, seed, code
revision, data identity, resource limits and result. See [lab mode](docs/lab-mode.md).
Iterate by reading the record, changing one setting and trying again. To try
several values of a setting, or to confirm a result over paired seeds, run
`klab sweep SWEEP.yaml` instead of a hand-written loop of tries: it trains the
baseline once per seed and reports every point in one table. Do not write
proposals or approval documents. A lab
record describes one local comparison; it is not a release or a scientific
conclusion.

Record IDs work with `report`, `compare`, `probe` and `explore`, and always mean
the record's own candidate checkpoint. Run IDs work with `probe` and `explore`.
`klab report RUN_ID --kind run` shows a run's verified evidence and its
post-training triage. See [results](docs/results.md).

### Act on probe verdicts

Every `try` runs the fast [probe](docs/probes.md) tier. For more, run
`klab probe RUN --vs BASE --config standard` (or `full`). Act on
`probe.verdict.action`:

- `abandon`: a hard probe failed, including a NaN or infinite loss. Drop or fix the idea.
- `tweak`: change one setting; `suggestion` names what the failing probe implicates.
- `rerun`: evidence is missing (`verdict.missing` says which). Fix it and run again; never read it as a pass.
- `escalate`: probe again at `next_tier`.
- `longer_run`: every tier passed and loss improved. Worth a larger budget, still in lab mode.
- `compare`: add `--vs`.
- `report`: a finished final verdict. Report it as it is.

Probes screen ideas; they do not prove one is useful. Selecting on the held-out
probes repeatedly turns them into development data, so a claim that an idea is
better needs an untouched final evaluation: a fresh split, a longer run, more
seeds. The held-back final split is that evaluation for probe items. Only
`klab probe CANDIDATE --vs BASE --config final` reads it, once, for a candidate
you have already chosen. Never use it to rank candidates, and never run it again
after changing the idea. `klab compare --final A B` shows final results side by
side; they never enter ordinary comparisons or the dashboard.

A single-seed gain is within-run noise until paired seeds confirm it (a sweep with
`seeds: [1, 2, 3]` does this). A try
marked `NOT_COMPARABLE` is never promoted. Do not edit probe items, thresholds
or prompts to make an idea pass; `guard.overfit_suspected` means the idea is
fitting the probes. A probe suite change needs a `SUITE_VERSION` bump.

`klab compare RESULT --references` places a result next to pinned public models,
comparing a metric only within one comparison group and saying why when it
cannot. Reference models (`ref:NAME`) are scored on text-level probes only and
are never a `--vs` baseline. `klab explore RUN` shows a small checkpoint's
internals; it describes, it does not judge.

### Paid compute

Rented GPUs and hosted providers cost money. Spending always needs the user's
explicit authorization; never infer it from an earlier allocation. Run the same
`try`, `train` or queued work there through [workers](docs/workers.md).

## 2. Establish scope first

Read `README.md`, the guides for the area you are changing (listed in
[docs/README.md](docs/README.md)) and `ROADMAP.md`. Check Git state before
editing and preserve other work; use a separate worktree when a running
experiment depends on a checkout. Model projects, named datasets and studies
belong in [Kybern Workshop](https://github.com/supernovae/kybern-workshop) or the
user's own repository, not in Lab.

## 3. Build on the current interfaces

- **One current path.** Each operation has one public command and one schema.
  Change a format by updating it and every caller in the same pull request, and
  say what must be rebuilt. Do not add compatibility readers, version adapters,
  aliases or a second runner to keep old artifacts working.
- **Use what Lab provides.** Do not reimplement hashing, checkpoint selection,
  accounting or workflow state in a script. If Lab lacks an operation, add it to
  Lab with a test, or open an issue stating the inputs, the expected output and
  how to tell it failed.
- **Look before building.** Before writing training, evaluation or
  quantization machinery, check existing libraries (nanoGPT, litgpt,
  lm-evaluation-harness, llm-compressor) and wrap one if it fits.
- **Large corpora** use the [corpus preparation](docs/corpus-preparation.md)
  interface. Lab mode may use any local dataset; the try record keeps its digests.

## 4. Keep results honest

Keep negative, interrupted and failed results. A changed protocol is a new
experiment with a new identity. Paths are locations, not identities. Keep
preparation, training, evaluation and generation measurements separate. Never
change batch, data, tokenizer, seed, architecture, precision, token budget or
optimizer silently: a lab delta states its change. Never change held-out data
or evaluation settings within a comparison; `try` refuses that as
`NOT_COMPARABLE`. A passing fixture shows wiring, not a scientific result.

## 5. Bound resources

- **Work root.** Outputs go under a work directory outside Git: `--work-dir`,
  then `KLAB_WORK_DIR`, then `$XDG_DATA_HOME/klab` or `~/.local/share/klab`.
  Lab tries live under `WORK_DIR/lab`. Only disposable fixtures belong in `/tmp`.
- **Runtimes.** A runtime ID selects an interpreter; `runtime.backend` selects
  where the model runs. Accelerator commands need a registered runtime
  (`--runtime ID`). See the [runtime guide](docs/runtime.md).
- **Size before running.** Check disk space and expected growth before large
  work, and use a resource envelope for anything big. Add a wall-time limit only
  when the user asks for one.
- **Stop safely.** Use Ctrl-C, SIGTERM or the printed `CANCEL` file; never kill
  a process during a checkpoint write. Never modify an active run, reset a
  spent budget or delete data you did not create.
- **Commit small things.** Check in declarations, summaries and references, not
  data or weights.

## 6. Verify the change

- **Environment.** Use `uv run --locked` and the repository's Ruff settings. Use
  the provisioned vendor environment for accelerator work (`--no-sync`). Do not
  install packages or create environments without authorization.
- **Tests.** While iterating, run the fast suite plus the slow tests for the
  area you touched:

  ```sh
  uv run --locked pytest tests -q -m 'not slow and not mps and not mlx and not cuda and not rocm and not xpu and not network'
  ```

  The nightly workflow runs everything ([test speed](docs/test-speed.md)).
  Prefer checks on identities, counters and refusals over wall-clock timing.
- **Corpus preparation changes** keep a connected test that renders the
  checked-in templates and reaches a prepared bundle on tiny fixtures.
- **Docs.** A change to a command, config field or default updates its page in
  the same pull request. Each subject has one page, listed in
  [docs/README.md](docs/README.md). `tests/test_docs.py` checks links, the index
  and every `klab` command a page shows.
- **Report** exactly which checks passed, failed and did not run, and review
  `git diff --check` and Git status before handing off.

## 7. Keep work reviewable

Use one agent for small tasks. Larger independent work may use parallel helpers
with narrow file ownership; let the user choose models, effort and concurrency.
Avoid overlapping jobs on one device. Keep commits narrow. Publishing, history
rewriting and remote deletion need the user's authorization. Proposed code work
goes in GitHub issues; keep `ROADMAP.md` short and directional.

## 8. Avoid sprawl

- **Reuse shared primitives.** Hash JSON with `canonical_digest`, encode with
  `canonical_json` and hash files with `sha256_file` (all in
  `klab.training.manifest`). Check filesystem trust with
  `verification_proofs.writable_by_others`. Never import another module's
  private helper: promote it or use the shared one.
- **One builder per fixture.** Shared test inputs live in `tests/*_fixtures.py`
  (`lab_fixtures.write_smoke_inputs` for lab-loop configs). Parameterize them
  rather than copying, and keep derived settings valid when shrinking a budget.
- **A test earns its place.** Keep a test only if its failure would mean that
  something a user relies on broke, a measured performance property regressed,
  a trust guarantee (identity, verification, refusal of bad input) stopped
  holding, or a bug that happened came back. Do not test wording, help text,
  private helpers or shapes that ordinary edits change.
- **Few tests per change.** A feature usually gets one test that runs its
  public path end to end and one for its most important refusal. Extend an
  existing outcome test before adding a new one. A bug fix gets the test that
  would have caught it.
- **Parametrize over code paths, not values.** Each case must take a different
  branch. Inputs that all take the same path are one case.
- **Delete as well as add.** Remove tests that duplicate another test's
  coverage or cover removed code in the same pull request. A rising test count
  is not progress; CI time and maintenance are the cost.
- **Laptop-sized tests.** Use the smallest budget that proves the behavior.
  Never depend on CI's umask, user, disk or `HOME`; always isolate
  `KLAB_WORK_DIR`.
- **Removing code is a change too.** After deleting tests, imports or helpers,
  run `ruff check .`, the touched tests and the fast suite. A red `main`
  blocks everyone.
- **No silent degradation.** A cache or proof that cannot be written says why.
  If a check refuses an ordinary local setup, fix the check rather than the test.

## 9. Report what people can inspect

When discussing training, give the dashboard URL, the run-store path, the model
name and the exact run ID. Use `/training?store=<URL-encoded-path>&run=<ID>` for
a stable link. Confirm the current run from fresh evidence; a stored `running`
status is not a heartbeat, and the newest record is not necessarily active.
Report status with its observation time, and say idle or completed when nothing
is verified to be training. Never restart training to make the dashboard look
active. Keep training progress separate from benchmark results for a particular
checkpoint.
