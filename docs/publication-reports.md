# Evidence-only publication drafts

Publication reports project a reopened native ExperimentPlan lock and an explicitly
selected evidence index into versioned JSON or a Markdown experiment report/model-card
draft. They do not collect evidence, run missing work, publish externally, or promote
a model. This is currently a callable API for integrations; public CLI wiring is a
separate follow-up. There is no publication-report command yet.

## Integration contract

The module is `klab.experiments.publication_report`:

- `build_publication_report(lock_path, *, workspace=None, index_path=None)` returns
  the allowlisted report dictionary. The lock is always reopened with native
  `open_lock`; arbitrary dictionaries and in-memory plans are not accepted.
- Supply both the experiment workspace and its content-addressed index path to
  consume collected evidence. Supply neither for a planned-only draft. There is
  no implicit latest-index selection and no collection fallback.
- `write_publication_report(lock_path, output, *, workspace=None, index_path=None,
  format="json")` verifies the inputs and writes one file. The only formats are
  `json` and `markdown`. It returns the output path.

Evidence verification uses the native
[retrospective views](research/experiment-dsl-retrospectives.md), including native
`read_evidence` verification of lock, controller, receipt, run and checkpoint
bindings. Changed evidence, mismatched identities, missing referenced bytes and
lock reopening failures propagate as errors. An index that already records an
invalid attempt remains a valid record of that invalid attempt; a formerly valid
snapshot whose supporting bytes change is refused.

Rendering reads native evidence, including locally available inputs required by
native lock verification. It cannot render an unavailable historical lock merely
from its hash. It creates no training, evaluation, acquisition or controller work.

## Interpretation and privacy

The machine format is `klab-publication-report`. Both formats retain the
original plan, scientific, source and evidence-index hashes, per-cell configuration
and input hashes, planned seeds/budgets, lineage, selected runs, attempt statuses,
checkpoint digests/counters and verification scopes, and integrity-bound validation
metrics with their recorded protocol and data/tokenizer hashes. Missing observations
stay empty or absent, never zero-filled. All attempts are retained, including failed
or censored attempts preceding a completed one.

`complete` describes recorded execution coverage. `pending`, `failed`, `censored`,
`invalid`, `ingestion_error` and `unavailable` retain their native meanings. An
explicit skipped observation would remain skipped; the current native collector
does not produce skipped observations, and this renderer never infers skipping
from absence. Planned-only cells are pending. Engineering smoke success and
scientific results are separately labelled `not_assessed`.

Comparison `available` only says both executions completed. The report preserves
the planned comparison mode and control/confounder counts, while leaving scientific
comparability unassessed and the winner null. It adds no comparison or uncertainty
engine. A lower recorded loss is an observation, not a causal finding or evidence
of across-seed significance. A descriptive or incompatible contrast cannot become
a controlled scientific result through rendering.

A strict field allowlist excludes machine-local availability paths, worker names,
raw configurations, arbitrary metadata, free-form error messages, report filenames
and comparison difference values. Rejected/missing report counts remain visible.
Markdown identifiers are escaped as inert text. This is a limited projection, not
an anonymization guarantee: authored identifiers and exact hashes are retained for
provenance, so review the draft before sharing it. No original hash is recomputed
as though the sanitized projection were the source record.

Dataset license, access rights, redistributability, distributed weights and
full-state resume availability remain `unknown`: this evidence contract does not
establish them. Checkpoint `verification_scope` is preserved independently. Even
locally verified weights do not establish distributable weights or resumable
optimizer/RNG state. No files, datasets or weights are bundled.

## Determinism and retained outputs

JSON is sorted, finite, UTF-8 encoded with ASCII escapes and a trailing newline.
Neither format includes render timestamps or destination paths. Repeated rendering
of the same verified inputs under this version yields identical bytes.

Writes reuse `result_publication.write_bytes_exclusive`: complete file bytes and
directory entries are synced without replacing an existing destination. Matching
bytes replay successfully after the primitive's directory sync. Different bytes,
symlink destinations and non-file collisions fail. A directory-sync failure can
leave a complete file behind; retain it and replay only after diagnosing the
failure. JSON and Markdown are independent publications, not a multi-file
transaction. A failure writing the second file never removes the first.

The test fixtures distinguish projection-only synthetic positive/negative and
incomplete observations from native verifier tests. Native coverage includes
reopened locks, incomplete/invalid receipts, a one-update synthetic CPU checkpoint
and held-out observations, post-collection tampering, read-only rendering, immutable
collisions and replay. These fixtures establish software behavior only.

The current report shape always includes `diagnostic_status` and
`diagnostic_evaluations`; planned/interim reports explicitly record uncollected
diagnostics. JSON and Markdown share the same contract identifier and observations.
Obsolete development evidence contracts require fresh collection; stored records
are never rewritten or upgraded in place.
