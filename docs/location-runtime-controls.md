# Location and runtime selection

For ordinary lab work, select one persistent root with global `--work-dir`
**before the command**, or set `KLAB_WORK_DIR`. Use `--runtime ID` when
executing in a registered interpreter. Neither setting moves explicit outputs,
changes a saved backend, or grants permission to spend on compute.

| Journey | Storage selection | Runtime selection |
| --- | --- | --- |
| `try` | Work root; optional declaration `lab_dir` for an explicit destination | `--runtime ID` or declaration `runtime_profile`; `--backend` is an explicit workload override |
| `report` / `compare` | Work root; explicit record path or `--lab-dir` for historical records; `report --kind run` uses `--runs-dir` | Read-only; no runtime selection |
| `probe` | Work root; declaration `lab_dir` / `runs_dir` for historical lookup | `--runtime ID` or declaration `runtime_profile`; inherits the saved backend unless `--backend` overrides it |
| `explore` | Work root; declaration `lab_dir` / `runs_dir` for lookup and cache | Mutually exclusive `--runtime ID` / `--runtime-profile PATH`; CPU by default |
| Saved weights (`eval`, `generate`, `chat`, `serve`) | Work root's `runs`; explicit `--runs-dir` retains a saved location | Mutually exclusive runtime ID / profile; backend selects the workload, not its interpreter |
| Explicit `train` / `stage` | Config output paths; `train --runs-dir` explicitly overrides the run root; stage has its own `--output` | Mutually exclusive runtime ID / profile; config/backend still determines the requested workload |
| Independent queue | `--store` selects controller state; use the same store for registration, submission, observation and control | Named worker, or single-config submission's runtime ID / profile to register a local worker |
| Release plan / Campaign | Declared workspace, operational binding and sealed receipt locations | Native runtime/worker/binding selection with existing authorization and accounting |
| `runtime env` | Host-local registry and separate runtime installation root | Environment management; never an experiment launch |

YAML lookup paths in TryConfig, ProbeConfig and ExploreConfig resolve relative
to their **source file**, with `~` expanded. They do not become relative to the
work root or the invoking directory. RunConfig's explicit tokenizer/cache/output
locations keep their existing loader semantics. Reauthor declarations when a
new destination is intended; selecting a work root is not relocation.

Before the project became Kybern Model Lab, the CLI was `sparselab`, its
variables were `SPARSELAB_*`, its default roots were `~/.local/share/sparselab`
and `~/.config/sparselab`, and its records used `sparselab-*` formats. Current
`klab` does not read or migrate those roots or records. Start a fresh work
root, and read older evidence from a pre-rename checkout.

An explicit record or checkpoint path is the safest way to read evidence from
another root. If a short ID matches multiple selected locations, native lookup
fails as `ambiguous_result`; it never chooses the newest or first match. Sealed
records select their recorded checkpoint even after `latest` advances.

## Retention decision (2026-10-11)

Audit base: `4deab09052fb799a3affd5d39b6c7c312f5bc8a3` (PR #85).
Current main has no `TODO.md`; the [historical CLI reduction plan](https://github.com/supernovae/model-lab/blob/5df1193198223bee50661e2db65a7f01cc8372c0/TODO.md)
and the current [roadmap](../ROADMAP.md) supply the intent. This decision covers
location/runtime controls only. It does not remove research launchers or add
the future parser surface budget gate.

Retain the audited controls. Their repeated spellings are not evidence of
interchangeable behavior:

- `--work-dir` chooses implicit persistent destinations and process scratch.
  Resolution is flag, environment, absolute nonempty `XDG_DATA_HOME/klab`,
  then `~/.local/share/klab`. `build_parser` resolves defaults without
  creating directories; execution authorizes the runtime before initializing
  the workspace. For `monitor` and `monitor-baseline`, select the outer CLI root
  with global `--work-dir` before the verb or per-command `KLAB_WORK_DIR`.
  `monitor --workspace` and the baseline positional root select the measured
  directory, not the CLI work root initialized before the handler. Keep an explicit
  worker lease namespace in the command environment when required.
- `--lab-dir` selects records and lab caches, whereas `--runs-dir` selects
  saved run manifests/checkpoints. A lab root has `tries`, `probes` and its
  own `runs`; it is not interchangeable with a run root. Report and dashboard
  can read independent historical lab and run roots. Removing these readers
  or searching more roots would lose access or introduce ambiguity.
- `--store` selects the controller database, workers, attempts and receipts,
  not just a directory of model checkpoints. `evaluation suite run` can need
  both `--runs-dir` for the checkpoint and `--store` for the executing worker.
  Study submission defaults to its derived experiment workspace, while study
  collection defaults beside its receipt. Making either follow a generic root
  would change existing locations. Renaming `--store` to `--runs-dir` would
  erase a useful distinction and conflict on suite evaluation.
- `--runtime ID` resolves a host-local registered interpreter/device descriptor.
  A file `--runtime-profile` supports an explicit descriptor without registering
  it globally; release bindings freeze that descriptor. They are exclusive
  sources for the same native loader, not two execution engines. Removing files
  would require mutating host registries for portable declarations. Removing
  IDs would lose provisioning's direct handoff to execution. Try/probe already
  put file selection in validated declarations instead of duplicate switches.
- `--backend` requests execution semantics or validates a worker's device.
  It does not select Python, install a framework or authorize acceleration.
  A runtime profile must agree with the effective engine/device request.
  Dropping backend overrides would prevent explicit CPU evaluation of saved
  accelerator weights; making a runtime override saved config would silently
  change execution. Explorer's CPU default is deliberately distinct from
  probe's saved-backend policy.

No public flags, output locations, defaults, identities or provisioning behavior
change in this decision. Further removal needs a concrete replacement for the
specific contract above and approval for any resulting behavior tradeoff.

## Audited callers

Parser-only traversal at the audit base gave these application command-local
occurrences (help excluded, repeated leaf options counted). The separate
environment route adds one `--backend`, on `register`, and none of the other
six audited switches:

| Control | Leaves | Callers |
| --- | ---: | --- |
| `--lab-dir` | 3 | report, compare, dashboard |
| `--runs-dir` | 20 | report; runtime status; facts evaluate/transfer-evaluate; engram export; train; model exercise; eval; capability evaluate/compare; study collect/report; serve; generate; chat; dashboard; corpus consumers; evaluation suite run and fixed-slices score/continuations |
| `--store` | 12 | runtime attention; study submit; worker register/status/relay check; experiment submit/list/cancel/resume/collect; controller run; evaluation suite run |
| `--runtime` | 19 | try; probe; explore; runtime attention; stage; train; eval; research snapshot; serve; generate; chat; campaign apply/resume; experiment submit/bind/run; evaluation suite run and fixed-slices score/continuations |
| `--runtime-profile` | 17 | Same as runtime ID except try/probe, whose file selection is in declarations |
| `--backend` | 24 | try; probe; facts evaluate/transfer-evaluate; engram export; train; model exercise; eval; capability evaluate/compare; study collect; research scaffold; research portability build; learn scaffold; serve; generate; chat; surface chat; worker register/serve-stdio/execute; evaluation suite run and fixed-slices score/continuations |

Global `--work-dir` is registered once on the application parser. Surface chat
also accepts a command-local work root: an explicit local flag overrides the
global prefix for both parent scratch and the chat app. Without a local flag,
the prefix is retained; without either flag, the environment and default rules
above apply. The resolved root is forwarded after Streamlit's `--` separator
to the app's chat mode. It is a subprocess storage handoff, not an extra search
root. Preliminary global parsing stops at the command, leaving command-local
flags and child command remainders untouched. The application parser does not
own `runtime env`.

Relevant implementations are `cli/main.py` (parser defaults and runtime gate),
`workdir.py` (persistent/scratch resolution), `lab_results.py` (unique lookup),
`try_config.py`, `probes/config.py`, `explore_config.py` (source-relative
declarations), `workers/cli.py` and `workers/controller.py` (controller store),
`evaluation/cli.py` (separate checkpoint/worker roots), and
`runtime_environments.py` / `runtime_profile.py` (native runtime descriptors).

## Torch-free environment route

`klab.cli.entry` dispatches `runtime env`, optionally preceded by global
`--work-dir PATH`, before importing the application CLI. Keep all eight paths:

| Path | Distinct operation |
| --- | --- |
| `list` | Observe registry and bounded interpreter inventory |
| `discover` | Observe bounded interpreter candidates |
| `show ID` | Observe one registered entry |
| `register ID` | Validate an absolute executable and publish a host-local entry |
| `unregister ID` | Remove registration without deleting its environment |
| `doctor ID` | Validate source, runtime and precision in the selected interpreter |
| `profile ID` | Emit the native executable profile |
| `provision ID` | Explicitly install a pinned recipe into a separate runtime root, retaining receipts/failures |

These operations are not aliases for training/runtime acceptance. The manager
does not import Torch; observation and doctor may run bounded children in the
selected interpreter. Provisioning intentionally installs packages, only on its
explicit path. The registry follows `XDG_CONFIG_HOME` (or home config), and the
installation root follows `KLAB_RUNTIME_DIR` or XDG data/home defaults,
not `KLAB_WORK_DIR`. The installation root must not equal the scientific
work root and must be outside Git. Nesting it under the work root is currently
allowed; the equality check is not a prohibition on nesting.

Global `--work-dir PATH` before `runtime env` selects the scientific work root
for native containment checks without creating it. It does not relocate the
registry or runtime installation root; use their host-local environment settings.
`klab runtime --help` points to the separate environment-management help.
On a provisioned vendor interpreter use `uv run --locked --no-sync` or the
executable directly; adding
`--extra cpu` can synchronize a CPU framework over the intended installation.
See [runtime environments](runtime.md#machine-local-runtime-environments).

## Regression evidence and limits

`tests/test_location_runtime_controls.py` checks source-relative declarations
across invoking directories at different depths and work roots, with absolute
and relative source paths. A mutation resolving against cwd instead of the
source directory is rejected for all three loaders in both cases.
It also checks framework-free help dispatch for all
eight operational paths without child execution or installation, including
fork/spawn audit events. Existing coverage retains the behavior behind this
decision:

- `test_workdir.py`: precedence, literal explicit destinations, read-only
  default resolution and storage receipts.
- `test_lab_results.py`: ID collisions, historical/exported paths, pinned
  checkpoints, missing/invalid evidence and saved-backend inheritance.
- `test_try_config.py`, `test_probe_config.py`, `test_explorer.py`:
  declaration validation, native dispatch and resource/runtime handling.
- `test_runtime_id_selection.py`, `test_runtime_authorization_required.py`:
  exclusive sources, early failures, interpreter handoff and acceleration gates.
- `test_runtime_environments.py`, `test_runtime_env_cli.py`,
  `test_runtime_env_provision.py`: separate roots/registry, no install on
  read-only commands, registration-before-READY publication and retained recovery
  evidence.

Fixture checks establish software contracts. They do not qualify hardware,
providers, throughput or scientific outcomes. Accelerator execution,
provisioning installs and optional heavyweight suites need their own authorized
validation. No such work is needed to retain these controls.

## Waiting for local publication locks

Campaign cache publication/cleanup, panel attempt journals, and runtime-registry
writes keep their normal blocking lock semantics. If their existing lock is
contended, they log one warning naming the lock before waiting; uncontended operations stay quiet.
The message does not identify an owner or authorize deleting/replacing the lock,
clearing state, or retrying an operation. There is no new timeout or stale-lock
reclamation. Inspect retained evidence and the operation holding the lock;
cancelling the waiting process does not release another process's lock.
