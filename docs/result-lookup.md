# Finding a lab result

Keep the record ID printed by `try` or `probe`. It works across the everyday
result commands:

```sh
klab report TRY_ID
klab compare TRY_ID --references
klab probe TRY_ID --vs BASELINE_RUN --backend cpu --config standard
klab explore TRY_ID
```

`report` reads the whole sealed record. `compare` reads its candidate's
measurements, preserving comparison groups and the explicit `--final` boundary.
Neither command loads a model. `probe` performs new scoring; `explore` performs
descriptive inspection. Both select the candidate checkpoint recorded by a try,
or the target checkpoint recorded by a probe. Supplying a record does not
implicitly select its baseline, tier, final split, backend or resource envelope.
`--vs OTHER_RECORD` selects the **other record's candidate**.

The headline probe example pins only the candidate. `--vs BASELINE_RUN` loads
the baseline run's current `latest`, which may have advanced since the try.
To use the same baseline the try scored, copy its `arms.baseline.run_dir` and
`arms.baseline.checkpoint` from `report TRY_ID --json` and combine them:

```sh
klab probe TRY_ID --vs /path/to/BASELINE_RUN/checkpoints/GENERATION --backend cpu --config standard
```

This pins both sides; there is no record selector for a try's baseline arm.

## IDs, paths and missing evidence

All four commands use the same lookup:

- A record ID is looked up under `LAB/tries` and `LAB/probes`, where `LAB`
  defaults to `WORK_DIR/lab`. `--lab-dir` selects another lab for `compare`
  and `report`; `try`, `probe` and `explore` use declaration `lab_dir`
  ([try](try-config.md), [probe](probe-config.md), [explore](explore-config.md)).
- Bare IDs also match existing entries in the current working directory,
  preserving the previous local-path lookup. If that is the only match, it is
  used; if a lab location also matches, lookup reports `ambiguous_result`.
  This includes unrelated same-named local directories. Use an absolute path
  to choose a location unambiguously; do not assume a bare ID is lab-only.
- A record file or directory can be passed explicitly, including an exported
  sealed JSON file with a different filename. Its digest is verified.
- `probe` and `explore` also accept run IDs under `LAB/runs` or an extra runs root
  (declaration `runs_dir`),
  run directories and checkpoint paths. A direct run still uses the native
  checkpoint loader's default (`latest`); an explicit checkpoint uses native
  checkpoint verification.
- A record selects its **exact recorded generation and digest**, even if the
  run has advanced. The native loader verifies it before model loading. If
  its saved location is gone, restore the run there or explicitly supply the
  relocated checkpoint path. No fallback to another run or `latest` occurs.
- If an ID matches multiple locations, pass an explicit path. A directory
  containing both record filenames requires an explicit file.

A run or checkpoint does not name a unique scientific record: it can have
several tiers, protocols and final verdicts. `report` and `compare` therefore
require a record ID/path, with an actionable error for a run/checkpoint input.
They do not guess the newest record or start scoring. Records remain readable
without the original checkpoint, including interrupted/resource-stopped records.
A record without a scored candidate cannot be used to load a checkpoint.

Lookup failures exit with status 1. With `--json`, the same text diagnostic is
returned as `status`, `result`, `reason` and `hint`. Status is `missing_evidence`,
`ambiguous_result` or `invalid_evidence`; successful record JSON and stored hashes
are unchanged. Probe cancellation, resource and OOM stops retain their existing
partial records and incomplete verdicts.

## Explicit devices

Standalone `probe` inherits each run's **saved backend** when `--backend` is
omitted. Installing `--extra cpu` only chooses dependencies; it cannot convert
a saved CUDA/ROCm run into CPU inference. A saved accelerator without runtime
authorization fails before loading with `runtime_required` and instructions
to select a compatible `--runtime` or declaration `runtime_profile` or explicitly choose
`--backend cpu` for a compatible PyTorch run. No unavailable device is silently
replaced. `explore` retains its documented CPU default; select an existing runtime
profile with `--runtime` or `--runtime-profile` to choose another backend. Backend overrides do not
rewrite saved configs or checkpoint identities. MLX cannot be converted into
PyTorch inference by a backend override.

## Scope of the everyday-command audit

The everyday commands now expose at most five executable options, excluding
help: try 5, probe 5, explore 5, compare 5, report 2. Scientific controls live
in validated declarations and runtime controls use native profiles/envelopes.
See [try declarations](try-config.md), [probe declarations](probe-config.md)
and [explorer declarations](explore-config.md). Removed flags are rejected,
not hidden aliases.

| Command | Retained distinct semantics |
| --- | --- |
| `try` | Derive a candidate, paired seed override, baseline reuse, held-out scoring, optional probe tier, checkpointed wall/resource stops. |
| `probe` | Explicit baseline, tier and final split; early-stop policy; benchmark tasks/limit change benchmark identity. |
| `explore` | Descriptive text diagnostics, refreshable cache, CPU default, resource stops. |
| `compare` | Read-only recorded metrics, comparison groups, references and explicit final-only reports. |
| `report` | Whole sealed record, including failed/interrupted/missing observations. |

## Baseline selector provenance

New `try` and standalone `probe` records retain the original `request.baseline`
selector, including relative spelling or an ID. The sealed
`baseline_provenance` records `selector`, the execution `working_directory`,
resolution `status` and, when available, `resolved` native selection. Try's
baseline is a config path (IDs are not supported by `try`). Probe uses native
unique lookup for run IDs, try/probe record IDs, run directories and checkpoint
paths. Its resolved selection records the matched path, kind, run identity,
checkpoint selector and expected digest; record inputs also retain the source
record digest. IDs are never converted into paths by joining them to the
execution directory. Paths in provenance describe locations, not scientific
identities or permission to compute.

For a run selector, `resolved.checkpoint: null` retains native `latest` selection
at load time. The scored `baseline` row (or `arms.baseline` in a try) records the
actual generation and digest. Record selectors keep their pinned generation and
expected digest; provenance neither bypasses verification nor changes baseline
reuse. Reading from another directory does not reinterpret the saved baseline
selector. Missing checkpoints still require restoration or an explicit new
checkpoint selection; provenance is not automatic relocation or replay.

Try seals provenance before input parsing. Probe seals a started record before
native lookup and retains failed/interrupted setup, with unresolved or explicit
lookup failure status when no baseline was selected. Such records contain no
scoring evidence and cannot select a checkpoint. Dashboard result history and
Home activity omit setup probes; `report` still reads their retained records.
Invalid declarations rejected before an execution starts do not create a record.

Text `try`, `probe` and `report` output shows the original selector and resolution
status. `--json` exposes the sealed fields directly. Historical records remain
readable under their existing format and digest: text reports explicitly mark
missing provenance and an unknown original working directory; JSON returns the
original sealed document without adding inferred fields. Readers never infer
that directory from their own cwd, reseal history, create stores, or score models.

For try, `resolved` means the config path was made absolute before parsing; for
probe it means native unique lookup selected a record, run or checkpoint.
Neither status proves that parsing, loading or scoring succeeded. Non-CLI callers
that omit `request` now receive a minimal sealed request containing `baseline`;
caller-supplied requests are preserved unchanged. Text says
`baseline: none requested` when sealed evidence explicitly records no baseline
request. Historical records without that request evidence retain uncertainty
about the original selector.
Historical records with a relative `run_dir` remain reportable, but
checkpoint consumers refuse to interpret that location against the reader's
working directory. Pass an explicit checkpoint path to start new work.
