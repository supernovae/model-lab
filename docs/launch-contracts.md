# Launch contracts

Use `try DELTA.yaml --vs BASELINE.yaml` for a local comparison. Use `train`
for explicit checkpoint transitions, and `experiment submit` for independent
queue dispatch. Waiting is a submission option: `experiment submit CONFIG
--wait`. The queue, worker protocol and receipt ingestion are shared with
non-waiting submissions; [workers](workers.md) describes their operation.

The optional [bounded lab-job adapter](lab-jobs.md) binds an external handoff
to the existing native host runner. Its tiny CPU profile delegates tokenizer
training, paired model work and reporting to the native lab lifecycle. It accepts
no pytest or arbitrary-command workload and adds no worker/controller scheduler.

## Contract and caller audit

The audit covers the eight requested executable command paths at revision
`c6246c5211aac9bad6388c16b6bd22be04de1e69`. This change removes top-level `run`
from the parser, help groups and dispatch; no alias remains. The eight paths
become seven (nine become eight when counting `try`). Generic queue launch
choices decrease from two to one. These counts describe public command paths,
not training engines, worker transports or all executable commands in the CLI.

| Path | Inputs, outputs and active callers | Decision and remaining gap |
| --- | --- | --- |
| `train` | RunConfig plus explicit resume, extend-budget, promote or recover checkpoint; stage bundle, runtime authorization, spot policy and resource envelope. Native trainer emits checkpoint/manifest/evidence. Used by training/checkpoint guides, readiness subprocesses and supervised attempt commands. | Retain. Generic submission does not express all direct checkpoint transitions, run-ID selection, stop-after-step and spot-policy obligations. |
| `run` | RunConfig, optional registered worker or authorized local runtime, stage/promotion binding and resource limits; submission plus terminal/ingestion result. Active public examples were workers and hosted-environments guides; runtime-selection and retired-dataset tests exercised its parser/preflight. | Remove. All those inputs and waiting semantics are handled by `experiment submit`; active examples and tests use that route. |
| `stage` | RunConfig, explicit output, inspect/validate/smoke/warmup gate, optional already-prepared bundle, cold verification, pilot deadline and observation destination. Emits a reusable stage bundle without advancing eventual training state. Used by readiness, iteration/runtime/corpus guides and preparation phase commands. | Retain. Queue staging is part of eventual execution, not an independently requested pilot receipt with these output and verification controls. |
| `experiment submit` | Single config or matrix, worker requirements, stage/promotion binding; native queue submission identities. Matrix expansion preserves coordinates and atomic submission. Used by worker/matrix guides and now hosted wait examples. | Retain as the generic queue route. Single-config waiting and runtime/resource controls share one config submission adapter. Matrix waiting/runtime overrides are rejected before execution preflight. |
| `experiment run` | Immutable ExperimentPlan lock, cell/phase, runtime binding or worker; authenticates declaration provenance, evaluation declaration, artifacts and parent completion. Emits plan SHA and submissions carrying locked-cell/runtime identities. Used by experiment-program guides and iteration next-action recommendations. Campaign uses the underlying locked-cell request and controller APIs. | Retain. Generic submission cannot consume a lock, select dependency-ready cells, authenticate bound runtime receipts or emit the same release evidence. Removing it requires those exact inputs and provenance checks in the surviving route. |
| `study submit` | Architecture study, lifecycle diagnostics, optional receipt destination and worker/store; reserves a durable immutable intent before enqueue and atomically publishes an exclusive study receipt containing study/matrix/card digests and coordinate-to-submission identities. Used by architecture guides and generated research scaffold instructions; collect/report consume that receipt. | Retain. Raw matrix submission lacks study validation, card identity and the authenticated receipt required by collect/report. A replacement must preserve exclusive receipt publication and expose already-submitted IDs on publication failure; `study recover` reads completely bound native evidence into an unused output without dispatch. |
| `attempt run` | Immutable ledger, content identity, monitor policy, workspace baseline, explicit update/target/generation reservations, command and native/Campaign receipt bindings. Reserves persistently, supervises the owned process tree and reconciles charged work. Used by release attempt packets, preparation compiler output and contract tests. | Retain. Queue dispatch has no corresponding cumulative ledger, arbitrary native phase, receipt accounting or zero-survivor completion contract. It cannot replace this supervisor. |
| `attempt run-phase` | Canonical preparation plan, phase label, bindings and owned attempt root, plus ledger/policies/baseline. Compiles authenticated phase arguments and delegates through the native attempt parser/handler with zero model reservations. Used by preparation guides, phase-command tests and release preparation packets. | Retain. `attempt run` alone does not accept the canonical plan/phase inputs or enforce their compilation; replacing this adapter requires a typed phase mode with the same authenticated argument and supervision obligations. |

Historical research records, source pins, receipts and evidence remain unchanged.
The native controller still owns scheduling, cancellation, resume and ingestion;
this command consolidation grants no spending, execution or release approval.
`try` remains the beginner front door. Remaining consolidation must close the
specific contracts above before deleting additional commands. Everyday options
and result-identifier design are separate work.

## Verification boundary

Parser and mocked adapter checks cover removal, runtime conflicts, early rejection,
option forwarding, local versus registered worker selection, terminal failure,
controller lock ownership, reconnection/ingestion, Colab UNKNOWN handoff and
passive Ctrl-C. Focused native queue, attempt-ledger and preparation-command
checks cover durable cancellation, atomic submission, retained accounting and
owned-tree shutdown without model execution. They do not qualify a runtime,
train a model or establish scientific results.

## Remaining-contract assessment

Assessed against main at `5eeeb7e98fb9e841562061e78db61005c603ced3`, after
the everyday option reduction. **Retain all seven remaining launch paths
(eight counting `try`).** Keep
`try` as the ordinary comparison entry point and `experiment submit` as the
generic queue entry point. The earlier removal of `run` eliminated two ways
to request the same queue work. The remaining paths select different work or
evidence contracts; reducing their count alone would not simplify those jobs.
This assessment changes documentation only.

Choose by the input and intended outcome:

| Need | Path / guide |
| --- | --- |
| Compare one local change | `try` — [lab mode](lab-mode.md) |
| Direct training or a particular checkpoint transition | `train` — [checkpointing](checkpointing.md) |
| An independent readiness bundle or disposable pilot | `stage` — [runtime staging](runtime.md) |
| Queue a config or independent matrix, optionally wait for a single config | `experiment submit` — [workers](workers.md) |
| Dispatch selected immutable plan cells/phases | `experiment run` — [training programs](experiment-programs.md) |
| Queue an architecture study and retain its collection identity | `study submit` — [studies](experiments.md) |
| Supervise an explicitly reserved native phase | `attempt run` — [corpus preparation](corpus-preparation.md) |
| Compile and supervise one canonical preparation phase | `attempt run-phase` — [corpus preparation](corpus-preparation.md) |

### Consumer evidence and replacement costs

These are current guide, generated-instruction and executable source consumers,
not usage telemetry. Historical research packets are retained evidence, not a
reason by themselves to preserve a live command.

| Contract | Current consumers and implementation | Why consolidation is not recommended now |
| --- | --- | --- |
| Direct training | [readiness](../src/klab/readiness.py) generates stage, interrupted-train and resume subprocesses; [research scaffolding](../src/klab/research/scaffold.py) emits explicit run IDs and stop steps. [The train adapter](../src/klab/cli/main.py) forwards resume, extend-budget, promote, recover, run ID, stop step and separate spot policy to the native trainer. | Queue submission is not a complete replacement for these operator controls. Preserve all transitions and their checkpoint identities rather than translating them into an implicitly selected queue attempt. |
| Independent staging | [Training](training.md) and [iteration](iteration.md) request reusable bundles before training. [The stage adapter](../src/klab/cli/main.py) accepts an explicit output, gate, verified prepared inputs, cold verification, pilot policy and observations. | Staging during dispatch serves the queued run. An independently requested bundle/pilot has its own output and verification boundary; merging it into submit would require a separate mode anyway. |
| Generic submission | [Workers](workers.md) and [hosted environments](hosted-environments.md) use the shared queue route. [Historical launch regressions](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_launch_dispatch.py), retained at the pre-consolidation revision, covered config controls, waiting, registered workers and rejection of matrix-only conflicts before work. | Retain the single queue front door already established. Do not restore another waiting runner or silently interpret release declarations as raw configs. |
| Locked plan dispatch | [Iteration recommendations](../src/klab/iteration.py) generate `experiment run`; [Campaign](../src/klab/campaign/engine.py) directly shares `locked_cell_request`. [Plan dispatch](../src/klab/experiments/cli.py) opens the lock, verifies authored declaration hashes and evaluation presence, authenticates runtime bindings, and requires one complete, ingested parent for dependent cells. | `submit_many` is already shared downstream. Moving lock selection into submit would need lock/cell/phase/binding inputs, provenance checks and identical plan/scientific/parent/runtime identities, not merely a parser rename. |
| Study dispatch | [Research scaffolding](../src/klab/research/scaffold.py) emits `study submit`; [study collection](../src/klab/experiments/study.py) consumes the receipt. Submission pairs each coordinate/config digest with experiment, attempt and run IDs, plus study/matrix/card identities, then publishes exclusively. | Raw matrix submission omits study validation and the collection receipt. Replacing the wrapper must preserve those identities and publication failure behavior; there is no demonstrated simplification for the study author. |
| Attempt supervision | [The public adapter](../src/klab/training/attempt_contract_cli.py), [native phase classifier](../src/klab/training/attempt_commands.py) and [receipt reconciliation](../src/klab/training/attempt_receipts.py) feed [AttemptBudget](../src/klab/training/attempt_budget.py). The preparation compiler and release packet regressions consume this route. | A queue submission is not the cumulative reservation or owned-process-tree completion boundary. Preserve persistent nonrefundable reservations, native phase/receipt accounting and shutdown of detached descendants while leaving unrelated processes alone. |
| Compiled preparation | [The preparation guide](corpus-preparation.md) and [connected preparation regression](../tests/test_preparation_sequence.py) use the canonical plan. [Compilation](../src/klab/training/preparation.py) rejects missing/unused bindings, output divergence and model work; `run-phase` reparses its compiled command through native `attempt run`. | Supervision is already consolidated. Removing the adapter would make operators/agents copy compiled monitor argv, or add a typed plan mode to `run` with equivalent compilation and authentication. Neither has an evidenced benefit here. Compilation validates bindings and paths; native dispatch/verifiers authenticate the referenced artifacts. |

### Recovery limitation and bounded follow-up

Study queue submission and receipt publication are separate durable operations.
`submit_study` rejects an existing receipt before dispatch and uses exclusive
creation after `Controller.submit_many`. Only an `OSError` from
`_write_json_exclusive` is wrapped in an error naming the already-submitted
run IDs; other exception types are not wrapped with those IDs. Publication
failure does not roll back the queue or provide a dedicated receipt
reconstruction command. The exclusive `open("xb")` creates the file before
writing: a failed write, flush or `fsync` can leave a partial receipt or one
whose durability is unconfirmed. Later submission rejects that existing path,
and collection cannot parse a truncated JSON receipt. Preserve and inspect any
such receipt, quarantining it with its bytes intact if needed, along with the
failure output and queued identities. Moving or deleting the receipt does not
undo queued submissions or make resubmission safe. Collection still needs the
original coordinate/config/study/card identities; do not retry submission blindly.
This is a recovery limitation, not evidence that raw matrix submission can
replace the study contract.

The bounded recommendation is to retain the contracts and use the routing table
above to reduce discovery friction. If receipt recovery becomes the next scoped
change, first specify how reconstruction proves the original study and exact
coordinate-to-submission mapping, including concurrent publication and partial
write failures. That design needs owner review before behavior changes. No new
command schema, graph, reproducer or GUI is proposed for implementation here.

### Focused verification and limits

The inspected local selection passed 131 tests covering stage parser controls,
launch dispatch, spot policy, preparation
argv, attempt CLI/bindings/native receipts, parent identities, queue persistence,
cell-scoped runtime receipt revalidation, study submission, cumulative reservation
accounting and harmless owned-tree/timeout subprocesses. It uses synthetic data,
fixture tokenizers and mocked runtime/counter evidence; it executes no model,
optimizer, accelerator or remote worker.

Additional model-free checks parse all four train transitions with explicit run
ID/stop step/spot policy, inject study receipt publication failure and an existing
receipt, and intercept `run-phase` delegation to confirm compiled argv, zero
reservations and the same ledger/policies/baseline. These are assessment checks,
not newly committed regression tests. Their outcomes and ordinary exact-head CI
are reported in the accompanying PR.

Full training/checkpoint equivalence, real smoke/warmup pilots, the complete
connected preparation sequence, provider/runtime qualification and scientific
results were not rerun. The existing connected tests remain coverage to preserve
in any future implementation; mocked checks do not replace them. Repository
consumer evidence cannot establish external user frequency or prove that a
different interface would be easier. No command or useful regression is deleted.
