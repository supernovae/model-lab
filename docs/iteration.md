# Validate an iteration with the lab

Commands below use the [activated, provisioned environment](../README.md#first-run).

Use Model Lab's declared inputs, native verifiers and retained evidence between
runs. A passing check establishes its stated scope; it does not approve model
quality or promote a child. Keep the baseline and parent generation immutable.
Record the intended scientific delta before compute, then read the native lab
record or resolved release plan rather than reconstructing it in a private script.

## Lab mode first

For a normal authorized local experiment, skip the sections below and run
[lab mode](lab-mode.md):

```sh
klab try DELTA.yaml --vs "$CONFIG"
klab report TRY_ID
```

It derives the candidate, trains both arms, scores both on the same held-out
split and writes one sealed record, with resource limits, data identity, safe
cancellation and held-out checks retained and no locks, approvals or
reconciliation. The rest of this guide is the **release path**: use it when a
result will be promoted, published or used as a release parent, or for paid
compute.

## Start a runner session

Use one external persistent work root and a fixed, tested source revision.
Examples below use shell variables `WORK`, `CONFIG`, `CAMPAIGN`, `GENERATION`
and `RUN_ID` for your actual paths and identifiers; they are not literal inputs.
Set `KLAB_WORK_DIR` to `WORK`, independently of checkout and branch.

Bash setup:

```sh
export KLAB_WORK_DIR="$HOME/.local/share/klab"
WORK="$KLAB_WORK_DIR"
```

PowerShell setup:

```powershell
$env:KLAB_WORK_DIR = Join-Path $env:USERPROFILE '.local/share/klab'
$WORK = $env:KLAB_WORK_DIR
klab readiness smoke --family dense --output "$WORK/experiments/session-check-001"
```

In PowerShell use `$env:KLAB_WORK_DIR` for the environment variable and
`$CONFIG`, `$CAMPAIGN`, `$GENERATION`, `$RUN_ID` for shell variables. Write
continued Bash commands on one line instead of using `\`. For a recipe copy, create a fresh
`$WORK/inputs` directory with `New-Item -ItemType Directory`, and use `Copy-Item`
for the recipe's declared files. Use forward slashes or quoted absolute paths
in YAML. Preserve reviewed inputs and their recorded identities.

For a dense-model session, run the bounded CPU wiring check once into a new,
task-owned output directory:

```sh
klab readiness smoke --family dense \
  --output "$WORK/experiments/session-check-001"
```

This check includes tiny training and child resume; it creates disposable runs
and evidence. Its `readiness.json` records commands, outcomes and identities.
It has no `--json` flag. Select other affected families when appropriate.
Success on CPU does not check whether the same workload executes in your
selected GPU environment. See [lab readiness](lab-readiness.md).
Do not repeat the entire test suite for each budget or config iteration on the
same tested revision. Code changes need nearest behavior tests and the relevant
broader gate. Full CPU CI jobs are currently manual-only on both Linux and
macOS (the small timed `lab-loop` job is the one automatic CPU model run); use [test-speed.md](test-speed.md) for the automatic checks and explicit
full-suite opt-in.

## Check the actual proposed configuration

```sh
klab inspect "$CONFIG"
klab workspace preflight "$CONFIG"
klab stage "$CONFIG" --through smoke \
  --output "$WORK/experiments/candidate-001-stage"
```

For agents, `inspect "$CONFIG" --json` gives machine-readable output.
`workspace preflight` and `stage` have no `--json` flag. Inspect estimates shape
and memory; preflight checks storage assumptions; staging verifies inputs and
executes a disposable pilot without advancing the eventual training state.
Use `--through warmup` for a measured full-shape runtime gate when required.
Provide the declared runtime profile using `--runtime-profile PROFILE` where
appropriate. Use a new stage output for changed intent.

On a provisioned vendor environment, use the documented
`klab ...` commands instead of synchronizing the
CPU extra into that environment. Follow [runtime acceptance](runtime.md).
Neither a memory estimate nor the session smoke proves that the actual model
fits or that its selected backend is stable.

## Let Campaign explain the next action

```sh
klab campaign status "$CAMPAIGN"
klab campaign next "$CAMPAIGN"
klab campaign explain "$CAMPAIGN"
klab campaign apply "$CAMPAIGN"
# Once state exists, reconcile or continue it:
klab campaign resume "$CAMPAIGN"
```

Every Campaign verb above accepts `--json`: agents should consume the native
states, reasons, deficits, identities and typed next actions instead of scraping
logs or recomputing them. Text mode exposes the same decision context to humans.
`status` reports last-committed outcomes plus fresh recoverability; it is not a
fresh rehash of every historical artifact. `next` and `explain` verify requisite
existing inputs without building or training.

`apply` and `resume` mutate orchestration state and can perform declared
deterministic stages, pilots, evaluation and generation. Without `--execute-runs`
they do not submit a new optimizer run; existing submissions can be reconciled.
Explicit submission still requires committed declarations, bound approval,
verified inputs and accepted runtime. Submission is an authorization boundary,
not proof that remote compute has completed. Read [Campaign](campaigns.md) for
worker/runtime binding and approval commands. Never infer permission to retry an
interrupted optimizer attempt from a missing output.

## Verify the parent and read outcomes

Use the finalized generation directory, not a moving `latest.json` or `best.json`
pointer, as the parent/evaluation binding:

```sh
klab checkpoint verify "$GENERATION" --json
klab report --kind run "$RUN_ID" --runs-dir "$WORK/runs" --json
```

These commands also have human-readable text mode: omit `--json`.
Full checkpoint verification checks supported native continuation state; do not
use `--weights-only` to claim optimizer-resume readiness. Evidence authenticates
retained checkpoint-bound heldout observations, with its own weights-only scope.
Triage reads the immutable advisory report without rerunning inference; missing
triage is `UNKNOWN`. Keep negative and interrupted results. Campaign collection,
evaluation, generation panels and model readiness provide declared result closure;
none automatically promotes a model. See [checkpointing](checkpointing.md),
[evidence](evidence.md) and [triage](research/post-train-triage.md).

## Repeat checks according to what changed

| Change | Repeat before the next run |
|---|---|
| Lab-mode idea (one YAML delta, local) | Nothing extra: `klab try` derives, trains, scores and records it; an identical completed baseline is verified and reused. |
| Only inspecting completed results | Native status/evidence/triage reads; do not train again to reconstruct evidence. |
| Training budget or other scientific config | Inspect the effective config and declared comparison; resolve a new immutable declaration/lock; check parent compatibility, storage, and candidate stage/runtime gates. Preserve unchanged identities. |
| Corpus, tokenizer or prepared bytes | Authenticate the changed artifact and affected dependency closure; prepare/resolve affected outputs and rerun config/runtime gates. Never relabel changed bytes as the baseline. |
| Lab implementation | Nearest contract tests, relevant broader checks and affected readiness families; refresh source-bound locks/compatibility and runtime acceptance as required. |
| Backend, environment, device or worker | Fresh acceptance on the actual runtime, full-shape pilot and storage check; CPU smoke cannot stand in for it. |
| New host/store, transfer, relocation or final archive | Cold authentication at the trust boundary; then only supported authenticated reuse. Check availability separately from historical identity. |

Verified reuse is already supported at selected native boundaries, not as a
universal skip-validation switch. Eligible unchanged artifacts in an owner-only
registered persistent store use signed host-local proofs; changed fingerprints,
closure/source identity, unsafe paths or missing/foreign proofs fall back to cold
verification. `stage --cold-verify` explicitly disables proof reuse; supported
ExperimentPlan commands also expose `--cold-verify`. Independent archive,
recovery and Family verification remain cold. Campaign
`plan|status|next|explain|apply|resume|approve` also accept `--cold-verify`. See [capacity-aware execution](capacity-aware-execution.md)
for exact support and trust limitations.

## Read-only iteration check

```sh
klab iteration check "$DECLARATION_OR_LOCK" \
  --parent "$GENERATION" --json
# At a new host or trust boundary:
klab iteration check "$DECLARATION_OR_LOCK" \
  --parent "$GENERATION" --json --cold-verify
```
For a single fresh phase, omit `--parent` to inspect its authored changes against
the expanded base configuration. This reports `parent.status: NOT_APPLICABLE`;
a continuation without an exact parent reports `PARENT_REQUIRED` and `BLOCKED`.


Pass an authored ExperimentPlan declaration, its immutable lock, or a Campaign
declaration. The parent must be one exact `step_*_gen_*` directory, not
`latest.json` or `best.json`. The command authenticates the parent full-state
checkpoint, compares its **saved** effective configuration with the selected
continuation cell, and reports actual changed fields and native transition
compatibility. It inspects source identity, available artifact bindings, storage
estimates and free capacity, runtime receipt evidence, and Campaign stage status
where declared. It does not run a runtime probe, pilot, prepare, download,
training, approval, or Campaign state transition. Cold fallbacks do not publish
proof receipts. Historical prepared/lock source mismatches require an
independently reviewed operational compatibility record. Full-state budget
continuation additionally requires the exact parent execution source: a lock
compatibility mapping does not authorize optimizer source drift. The command
never grants either authorization.

JSON and text expose the same versioned result
(`format: klab-iteration-check-v1`): `state`, `parent`, `declared_delta`, `delta`,
`artifacts`, `verification`, `storage`, `runtime`, `ingestion`, `campaign`,
`missing_gates` and `next_command`. `declared_delta` lists authored phase
patches where available; `delta` lists observed parent-to-child values, or
base-to-candidate values for a fresh phase without a parent, including derived
exposure and optimizer-horizon effects.
`verification.diagnostics` exposes proof hits/misses,
fallback reasons, artifact/kind/verifier authority, and measured bytes hashed
or avoided; unavailable measurements are `null`. Cold mode has no proof store,
so diagnostics are `null`, not zero. A retained runtime receipt alone is
`RECEIPT_ONLY`, not fresh runtime authorization. Campaign runtime acceptance
establishes the recorded runtime contract, **not** a full-shape staging pilot.
The separate pilot gate can be waived only for a verified, terminal full-state
parent already trained at the same model shape, on the exact execution source
and matching accepted runtime contract. Changed source or runtime still needs
a new pilot; a bare ExperimentPlan lock remains `NEEDS_PILOT` even with a
retained runtime binding receipt. Dispatch performs fresh runtime/source checks.

States are `SAFE_TO_PREPARE` (declaration needs immutable lock),
`NEEDS_PILOT` (unproven runtime or full-shape pilot), `NEEDS_APPROVAL`
(Campaign gate), `READY_TO_DISPATCH` (all inspected gates), `RUNNING`
(Campaign work in flight), and `BLOCKED` (failed, incompatible, missing,
ambiguous or unverifiable prerequisite). Exit status is 0 for non-blocked
states, 1 for `BLOCKED`, and 2 for CLI usage errors. A non-blocked state is
not an authorization to execute: the subsequent native command performs its
own fresh execution-time validation. Independent native `stage`, runtime
binding, `campaign explain`, and `checkpoint verify` remain authoritative;
no user-provided pointer or unchecked declaration is silently repaired.
