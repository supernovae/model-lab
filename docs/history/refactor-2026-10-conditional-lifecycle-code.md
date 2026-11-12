# Six conditional lifecycle code needs

The branch starts from remote `main` at
`1c1fbc570c1c522a69d0c29a82a371bcb50b548f`, the merge of PR #56's reviewed
head `2b981988dc454820d296bf6c950683be3123c975`. It does not reuse the
consolidation branch. The exact first six conditional entries in that revision
of `TODO.md` define this code task:

| Item | Native implementation and evidence |
| --- | --- |
| Source-bound held-out-item CLI adapter | `evaluation freeze-items` delegates cold release, family, span, item/control and complete-denominator checks; publishes an exclusive directory and immutable digests. |
| Signed warm source-snapshot proof reuse | A source-snapshot-only, pinned transport re-export exclusion removes an irrelevant dynamic execution import; authority and input changes fall back cold. |
| Read-only tokenizer-artifact CLI verification | `tokenizer verify RUN_CONFIG` authenticates immutable origin and artifact, actual vocabulary and special IDs without fitting or creating a work store. |
| Acquisition receipt resource counters | Bounded acquisition reports consumed expansion bytes and task-owned staging peaks; failure/unavailable readings and historical absent fields are covered explicitly. |
| Declarative verified snapshot inheritance | Recovery corpus steps bind parent receipt and selected snapshot identities; native verified staging removes manual snapshot-copy setup. |
| Operational spot-safety policy | Separate `checkpoint plan-spot` / `train --spot-policy` configuration uses observed write, boundary, restart, notice and capacity readings; operational saves preserve scientific checkpoint watermarks. |

Public usage and limits are in [artifact adapters](../verification-adapters.md),
[proof reuse](../capacity-aware-execution.md), [datasets](../datasets.md),
[recovery](../research/lifecycle-recovery.md), and
[checkpointing](../checkpointing.md#optional-spot-safety-policy).
The [offline 100M assessment](../research/100m-readiness.md) is planning evidence,
not a changed preparation packet or runtime allocation.

## Validation boundaries

All executed tests use inspected explicit selections, offline tiny bytes,
hand-authored tokenizer vocabularies, mocks, static declarations and native
readers. No source acquisition, tokenizer fitting, model initialization,
forward, update, generation, GPU job or production corpus preparation ran.
The locked CPU/development environment was synchronized as authorized; its uv
cache was relocated to the writable `/workspace/.cache/uv` after the default
home cache was read-only. No lockfile changed.

The held-out adapter's successful fixture mocks release authentication while
exercising real family/span/chunk/item validators; a separate test uses the real
cold verifier to reject an invalid release. Tokenizer success authenticates
real hand-written snapshot bytes. These checks do not establish reviewer
correctness, production admission or model quality.

Spot policy is explicit direct-training configuration, not implicit worker
configuration. Live qualification still requires bounded measurements of durable
full-state checkpoint writes, boundary latency including evaluation, restart,
provider notice delivery, signal-to-commit, and verified child resume on the
actual runtime. No provider or model qualification is claimed here.

## Review and automatic CI

Independent review checked actual CLI/config callers and failure paths. It found
and corrected operational checkpoint saves shifting the scientific watermark,
stale elapsed time after evaluation, and fresh ancestry depending on manually
staged snapshots. Resource review also added empty interrupted-file inode and
missing/symlink staging-root coverage.

Only `.github/workflows/ci.yml` exists in the current workflow tree. Its feature
branch push does not trigger a workflow; opening/updating a PR selects only
`safe`. The stale registered KML workflow has no file in either base or branch.
Model/platform/full-suite jobs remain explicit manual dispatch choices and were
not dispatched. The new regression modules and exact acquisition nodes are
included in the explicit safe selection and its routing assertion. Fixtures and
transitive calls were inspected before execution. Contributor/test documentation
was corrected to describe this actual routing.

The first safe-suite invocation overlapped an unfinished CLI module write and
reported two import failures (17 passed); the stable implementation rerun passed
all 19 original selected tests. A mid-edit format check found four unfinished
files. Intermediate inheritance fixtures also needed frozen-model and sorted
source-reference corrections; all final selections below pass.

## Final offline results

Commands use `UV_CACHE_DIR=/workspace/.cache/uv`, `uv run --locked --no-sync`,
offline settings and disabled pytest plugin autoload. Test counts overlap and
must not be summed as distinct coverage.

- **102 passed in 16.58s:** the exact expanded selection in
  [the safe CI job](../../.github/workflows/ci.yml), including its routing assertion.
- **59 passed in 3.49s:** `tests/test_verification_adapters.py`,
  `tests/test_kml_card03_items.py`, and `tests/test_spot_safety.py` together.
- **24 passed:** `tests/test_snapshot_inheritance.py`,
  `tests/test_snapshot_warm_reuse.py`, and
  `tests/test_snapshot_authority.py::test_semantic_domains_retain_snapshot_verifier[source_snapshot]`.
- **24 passed in 1.95s:** selected mocked acquisition nodes listed below.
- Independent review separately passed 53 inspected zero-model cases across
  three runs, including caller and negative-path checks; no unresolved blocking
  finding remained.
- Whole-tree Ruff rule and format checks pass (**852 Python files**); Git diff
  whitespace checks pass. Research lint passes (**417 historical records**).

Acquisition node selection (all from `tests/test_corpus_acquisition.py`):

```text
test_hf_legacy_declaration_hash_and_receipt_reuse
test_bounded_git_blob_metadata_checksum_and_resume
test_hf_explicit_metadata_redirect_and_budget_receipt
test_hf_transport_interruption_resume_and_exhaustion
test_hf_transport_disk_decompression_and_deadline_are_separate
test_hf_transport_ledger_reopens_and_rejects_exhaustion_or_corruption
test_hf_bounded_replay_hash_and_row_provenance
test_hf_bounded_rejects_caps_bad_shards_and_bad_rows
test_hf_bounded_output_caps_and_config_split
test_bounded_resource_readings_exact_expansion_cap
test_bounded_overlong_line_retains_measured_prefix
test_required_staging_reading_failure_is_closed
test_parquet_required_expansion_reading_is_unavailable
test_staging_counters_include_empty_interrupted_input
test_wikimedia_bounded_xml_provenance_and_receipt
test_interrupted_wikimedia_acquisition_reuses_verified_snapshot
test_verified_snapshot_alias_reused_across_project_ids_without_transfer
test_wikimedia_rejects_checksum_caps_and_doctype
test_wikimedia_interruption_retains_distinct_resource_receipts
test_wikimedia_resource_journal_rejects_symlink
test_hf_without_transport_ledger_retains_interrupted_shard
```

The JSONL/gzip/Parquet replay and exact expansion cap nodes are parametrized.
Native replay/build execution, model/optimizer integration, hardware/provider
checks and the broad test suite remain unrun. The unrelated existing array
authority exclusion signatures were not changed or qualified. Snapshot
inheritance tests mock the producer dispatch boundary, while exercising actual
native staging, worker inheritance preflight and cold acquisition verification
on hand-written immutable snapshots. The generic ancestry change also removes
historical hard-coded source names/counts; exact identity and declaration checks
remain enforced.

Historical experiment files, declarations, receipts, archive identities and
spent allocations are unchanged. No other TODO implementation was activated.
