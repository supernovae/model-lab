# Read a result

`report` is the result front door. The default reads a sealed try/probe record;
explicit `--kind` selects a different native reader without guessing from an ID.

```sh
klab report TRY_OR_PROBE_ID --json
klab report RUN_ID --kind run --runs-dir /path/to/runs --json
klab report --kind research --config research-report.yaml --json
```

Advanced read selectors live in strict YAML, keeping the public report command
at five options. For the research example, create `research-report.yaml`:

```yaml
report_config_version: 1
lifecycle: /path/to/lifecycle.json
evidence_root: /path/to/evidence
# Optional: study: /path/to/study.yaml
# Or: protocol_sha256: EXACT_64_CHARACTER_DIGEST
```

For a run overlay, use `report RUN --kind run --config run-report.yaml` with:

```yaml
report_config_version: 1
surface_dir: /path/to/surface-review-bundles
```

Explicit relative declaration paths resolve beside the YAML; omitted
`evidence_root` retains the current-directory default. Unknown keys, a wrong
version, invalid protocol digests, conflicting diagnostic scopes and selectors
for the wrong report kind are rejected before the readers run. Declarations
contain only read selectors, never output paths or execution controls.

The default record text and JSON are unchanged. `--lab-dir` selects its existing
record location, and exported record files remain readable without local weights.
Run IDs remain distinct from sealed record IDs; see [result lookup](result-lookup.md).

A run report uses `experiment_evidence` for integrity-bound checkpoint and
held-out observations, and `read_triage` for saved advisory diagnostics. Its
`klab-run-report-v1` JSON envelope keeps `evidence` and `triage` separate;
A declaration's `surface_dir` adds the existing independent `surface_review` overlay. Negative
observations, rejected checkpoint/report digests and missing reports remain in
the native evidence payload. Missing or invalid triage stays `UNKNOWN` with its
reason; it never becomes a pass. Malformed artifact shapes raise `ValueError`
at the reader boundary; unexpected reader `TypeError`/`KeyError` bugs propagate
instead of becoming `UNKNOWN`. The triage boundary checks manifest container
shapes before delegating to native manifest verification, leaving other
`read_manifest` callers and their error contracts unchanged. Checkpoint PASS
bindings require nonnegative integer step/target counters. If the run evidence
reader cannot read its
inputs, the envelope reports `missing_evidence` or `invalid_evidence` and exits
1 after rendering. A rejected individual observation remains visible in the
native `rejected_reports` list. Triage advice is not a release decision.

A research report uses `research_status` and `next_experiments`. Its
`klab-research-report-v1` envelope includes both `status` and `next`, with
native evidence availability and diagnostics intact. Declaration `study` or
`protocol_sha256` scopes prior-design diagnostics as before. Next tests are
reviewed declarations, not instructions authorizing execution. `research validate`
and baseline inspection retain their distinct verification and identity roles.

Only record and run reports are attempt-dispatchable inspection commands.
`report --kind research` is a direct read-only CLI view outside the reviewed
attempt grammar, as the former research status/next views were.

All report kinds read existing artifacts, create no run store and start no
training, model loading, probe evaluation or collection. Interrupted sealed
records retain their original stop evidence. Reads do not reseal or rewrite
historical records. A passing report does not establish scientific qualification.

## Surface audit and boundaries

| Audited surface | Current native path | Why |
| --- | --- | --- |
| `report` | `report RECORD` | Sealed try/probe reader and existing renderers. |
| Run `evidence` and `triage` | `report RUN --kind run` | Two saved views of one result, combined without mixing their judgments. |
| `research status` and `research next` | `report --kind research` | Declared lifecycle status and next-test views, together. |
| `study report` | Retained | Validates supplied study/receipt/evidence and **writes** a content-addressed static bundle; publication is distinct from browsing. |
| Campaign `status`/`next`/`explain` | Retained | Native Campaign engine reports reconciliation, dependencies, approvals and release readiness; `explain` adds stage declarations. |
| `experiment explain` | Retained | Describes verified locked-plan cells and identities, rather than an observed result. |
| `monitor-baseline` | Retained | Measures live persistent storage for a time interval and writes a monitoring baseline. |

The four redundant executable paths are removed. Parser-only inventory against
main `f33b9f7` changes 186 executable leaves to 182 and 727 leaf option actions
to 717 (excluding help; the separate eight-path runtime-env route is unchanged). Active documentation and the
native attempt inspection grammar use `report`; there are no compatibility
commands. Historical research records, receipts, hashes and schemas remain
unchanged. This change does not remove distinct advanced workflow contracts or
introduce a shared experiment ledger.

New checkpoints expose [prepared-data exposure and timing](checkpoint-exposure.md)
through `evidence.checkpoints[].exposure` in run-report JSON. Historical absence is
reported as null, without reconstructing or modifying old evidence.

For separate checkpoint-bound development losses, see [named heldout panels](named-heldout-panels.md).

For finite local recovery and scientific checkpoint keep-sets, see
[checkpoint milestone retention](checkpoint-milestones.md).
