# Offline lifecycle consolidation — review record

Base: `fb5b1d826428a8a7dc0a44c388de919db78bd7ec`, refreshed from
`origin/codex/kernel-base-50m-proposal`. This newer head already included the B17
source/split filename fix; it was not reimplemented. Work is isolated on
`codex/corpus-interface-consolidation`; the original checkout remains clean at
`58ac08ee1289c1446491c8082851f5be196dfbcd`. Production is paused. No push, PR,
merge, production processing, external source acquisition, tokenizer fitting, model initialization,
forward, optimizer update, generation or cloud/device job was performed.

## Implemented surface

The [current public mapping and coverage](../corpus-preparation.md) documents
replacements and pruning. The [current packet](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/card05-base-50m/current/README.md)
owns one 32-phase argument/path/template declaration. `attempt phase-command`
expands it; `attempt run-phase` delegates one zero-counter phase to the existing
attempt dispatcher and monitors. It is not a scheduler or review engine.
The public renderer and initialization bind the verified embedded baseline
identity. Project source/split/transform/release references are generated from
the same leaf paths, and cannot be overridden through template values.

Removed old writable budget APIs/module commands and old ledger replay support,
unbound admission review-v1 acceptance, DevMind-specific committed-evidence trust
and its flags, historical surface-study import modes, experiment-specific
launchers and obsolete tests. Current cold verification still reads the truthful
release/normalizer/tokenizer identities needed by retained inputs. A discovered
schema-3 cold-verifier bug now consistently reads a build receipt's `identity`
or frozen manifest's `build_identity`; no old artifacts were rewritten.

## Historical preservation and deletions

[Lessons learned](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/LESSONS_LEARNED.md) summarizes
dense token-budget/scale/decoding, DevMind and TinyStories findings and limitations.
[The archive index](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/README.md) is explicitly
inactive. [The path map](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/path-map.json) records
original paths, new locations, exact hashes/sizes and the recoverable source
revision for removed helpers.

- 194 files / 6,043,943 bytes relocated intact: ten earlier research directories
  and 35 superseded KML packet files, both old launch documents and the retired
  Card 03 continuation test allowlist.
- 40 obsolete research/tool scripts and 15 old test modules deleted, plus the
  historical `surface_studies.py` implementation. Useful guarantees moved to or
  remain in focused native lifecycle tests; old scientific qualification
  assertions were retired without claiming zero-model equivalence.
- All 96 KML result/evidence files, including B11–B17, verified byte-for-byte
  against the base revision at their existing paths. B16's outer receipt-based
  shutdown remains unverified. B17 has no production ledger or allocation used.
- Pre-existing machine-local collected data, checkpoints, ledgers and raw receipts untouched.
  No archive replay framework remains. Hash-bound inert learning-reference
  catalog records remain evidence, with no runnable recipe.

AGENTS now states six actionable invariants. TODO contains missing code rather
than old execution milestones or optional platform/provider qualification.

## Verification

Commands used the existing locked environment, without dependency sync:

```sh
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_PROJECT_ENVIRONMENT=/workspace/model-lab/.venv
export PYTHONPATH=/workspace/tiny-sparse-refactor/src
# All pytest commands below use: uv run --locked --no-sync pytest -q
```

Some independent checks used `/tmp/uv-cache` or `/tmp/klab-uv-cache`; the same existing environment and
source checkout were used. Test bodies and their transitive effects were
inspected before selecting these bounded checks. No broad pytest was run.

| Final selection | Result |
| --- | --- |
| `tests/test_attempt_contract.py tests/test_attempt_contract_cli.py tests/test_attempt_contract_binding.py tests/test_preparation_commands.py tests/test_ci_routing.py` | 33 passed, one integrated invocation; the 3 compiler cases also passed after preserving the original 8 MiB production batch default |
| `tests/test_research_lint.py` | 98 passed, includes actual staged records and archive/template rejection coverage |
| `tests/test_kml_50m_launch_packet.py` | 6 passed against current and explicitly archived inputs |
| `tests/test_preparation_sequence.py` | Final connected node passed (386.80s); the other 5 cases passed in the preceding module run |
| Eight token-authentication nodes listed below | 8 passed in one final integrated invocation (7.50s) |
| `tests/test_corpus_release_streaming.py::test_rehashed_release_rejects_tampered_evidence` | 3 passed (LM text, lineage, rights tampering) |
| Native sensor/storage/shutdown selection listed below | 19 passed |
| `tests/test_corpus_admission_inspection.py` | 3 passed in final invocation (3.42s); exact sample coverage, changed inputs/caps and obsolete-review refusal |
| Three archive fixture nodes, panel schema node, CI routing and native surface triage importer | 6 passed in targeted runs |
| `ruff check .`, `ruff format --check .`, `git diff --check` | Passed; 845 Python files formatted, 417 tracked research records valid |

Final connected command:

```sh
uv run --locked --no-sync pytest -q --tb=short tests/test_preparation_sequence.py::test_offline_preparation_sequence_through_native_supervision
```

This final run completed native prepared-bundle publication and cold verification,
including zero-update accounting and shutdown assertions. The other five cases
passed before the final connected-only optimizer fixture correction; they were
not reported as one six-test final invocation. A read-only public
`attempt phase-command --plan experiments/research/kernel-memory-lab/card05-base-50m/current/preparation.json --attempt-root /tmp/corpus-lifecycle-phase-command/attempt --label verify-snapshots`
check also passed and emitted the canonical `corpus acquire ... --offline` arguments
and distinct supervisor receipt/monitor paths, without executing acquisition.

Exact token nodes:

```text
tests/test_corpus_token_identity.py::test_normal_path_fully_authenticates_inputs
tests/test_corpus_token_identity.py::test_normal_operation_rejects_changed_manifest
tests/test_corpus_token_identity.py::test_unsafe_paths
tests/test_corpus_token_denominator.py::test_complete_receipt_reuse_and_mutation_refusal
tests/test_corpus_token_denominator.py::test_cross_release_origin_inventory_and_cli_reuse
tests/test_corpus_token_denominator.py::test_cross_release_rejects_wrong_origin_tokenizer_and_inventory
tests/test_corpus_token_denominator.py::test_unrelated_evidence_sha_never_bypasses_full_verification
tests/test_corpus_token_denominator.py::test_receipt_cannot_reactivate_committed_evidence_trust
```

Exact native sensor/storage/shutdown selection:

```text
tests/test_operational_monitor_sensors.py
tests/test_operational_monitor_safety.py::test_baseline_identity_is_immutable_and_live_files_counted
tests/test_operational_monitor_safety.py::test_transient_path_restarts_entire_sample_but_real_io_error_fails
tests/test_operational_monitor_safety.py::test_persistent_disappearance_exhausts_deadline
tests/test_operational_monitor_safety.py::test_missing_or_wrong_sensor_fails_before_launch
tests/test_operational_monitor_safety.py::test_late_sensor_failure_stops_owned_command
tests/test_operational_monitor_safety.py::test_early_parent_and_detached_term_ignoring_worker_are_gone_before_return
tests/test_attempt_contract.py::test_owned_runner_cleans_detached_worker_and_preserves_sentinel
tests/test_attempt_contract.py::test_runtime_deadline_keeps_charge_and_zero_survivors
```

Exact archive/panel/native-triage selection (first five together, final one separately;
`UV_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `-p no:cacheprovider`):

```text
tests/test_archive.py::test_partial_thin_is_honest_and_portable_refuses
tests/test_archive.py::test_unsafe_archive_members_and_mutation_are_rejected
tests/test_archive.py::test_declared_recovery_receipt_is_authenticated_in_thin_archive
tests/test_generation_panel.py::test_current_panel_declaration_is_supported
tests/test_ci_routing.py::test_ordinary_ci_selects_only_explicit_zero_model_nodes
tests/test_surface_triage_import.py::test_triage_import_matches_distinct_verified_sources_without_generating
```

Earlier failed/interrupted development checks are not counted as passes:
old budget tests exposed two stale fixture assumptions (output overlap and mock
initialization kwargs), corrected before the final 33-test invocation. Token
checks initially had two setup errors, then a shared fixture directory collision;
corrected affected nodes passed, then all eight passed together. The schema-3 receipt identity bug was separately
fixed and its three tamper cases passed. First actual research lint rejected
unrendered numeric template slots; explicit template validation fixed that while
ordinary configurations still require concrete types. Connected fixture revisions
exposed a helper argument collision, a list-versus-null source-effect override,
an optimizer warmup incompatible with the tiny step count, and obsolete command
removal during an in-flight run. Static config loading and cold mixture verification
passed after fixture corrections. A diagnostic publish against an older disposable
fixture stopped before data work because its temporary envelope had been removed
by pytest retention; it is not counted as verification. Disposable fixture runs
were stopped when template substitutions changed; no production attempt was
involved. The final connected run is the qualification for the final fixture.

## Remaining qualification

The connected fixture uses tiny retained local snapshots, explicit fixture
source/policy/selection/resource substitutions and a prebuilt WordLevel vocabulary,
without fitting. Actual production templates, native baseline rendering/init,
canonical phase arguments, real corpus verifiers and cold bundle verification
are the wiring gate. Synthetic review decisions are not corpus admission.

Production rights/sample review and reviewer independence, accepted protected
families, measured 50M supply, actual mixture/bundle evidence, live ROCm/hosted
shutdown and all model quality or performance remain unrun/unqualified. A new
production attempt needs separate authorization at the reviewed new commit.

## Follow-up review and repairs

Independent static review of `f1edf7ebfbe37ee51e920922911577999c5ab2bd` found
four concrete omissions: one live lifecycle evidence reference, two retained
Card 03 test modules importing deleted helpers, a moved tokenizer-probe fixture
reference and a source-effects test using the retired packet. Bounded review
checks reproduced 18 failures in the two Card 03 modules plus the 13 probe
validation cases (3.19s), and one source-effects declaration failure (2.67s).
These failures qualify the earlier bounded passing results; they were not
production failures.

The follow-up directly repairs the lifecycle location binding while preserving
the evidence digest/size, updates current fixture/declaration references, and
retires the two old script callers. Their allowlist is archived byte-for-byte.
The native Card 03 item tests retain source/family/chunk/control/output guarantees;
old context-screen and gold-blind review-receipt protocols are explicitly retired,
not represented as equivalent coverage. No deleted helper or replay framework
was restored. A new public lifecycle-reader regression verifies the retained
TinyStories evidence is available at its archived location.

Follow-up bounded checks use the environment above plus `UV_OFFLINE=1`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `--tb=short` and `-p no:cacheprovider`.

```text
tests/test_research_lifecycle.py::test_archived_tinystories_evidence_remains_available_without_rewriting_identity
tests/test_research_lifecycle.py::test_evidence_status_distinguishes_verified_missing_tampered_and_symlink
tests/test_corpus_tokenizer_probes.py::test_probe_suite_validation
tests/test_corpus_tokenizer_probes.py::test_probe_rejects_missing_tokenizer
tests/test_corpus_tokenizer_probes.py::test_probe_rejects_non_object_and_malformed_json
tests/test_kml_card03_items.py
tests/test_corpus_source_effects.py
```

**37 passed in 19.56s.** The scoring test that fits a tokenizer was deliberately
not run. All source-effects transports in this selection are fixture adapters
writing tiny local bytes; socket/HTTP/subprocess transport tripwires forbid real
source acquisition. The public CLI cases cover verified retained reuse, offline
and warm/cold readback, missing/corrupt/symlinked origins, declaration mismatch,
incomplete effect sets, stale proof/bad copies, changed lock origin and origin
mutation. Only the explicit fixture Wikimedia adapter can consume transfer
counters; retained origins consume none.

```text
tests/test_research_lint.py
tests/test_ci_routing.py
tests/test_preparation_commands.py
```

**102 passed in 7.51s.** The active source/test/workflow scan now finds no callers
of deleted modules or relocated fixtures. Remaining old TinyStories catalog paths
are hash-bound non-executing `recipe: null` provenance. CLI removal assertions
and migration documentation intentionally name obsolete interfaces. Workflow
selections remain explicit; no workflow references either retired Card 03 module.

A direct comparison with the original base verified all 194 archived files
(6,043,943 bytes) and all 96 KML result/evidence files unchanged.

The connected fixture follow-up replaces both `source_effects=None` substitutions
with complete sorted `reuse_only` declarations. It verifies the tiny prior
acquisition cold, copies its exact snapshots locally, binds the origin/copy
identities and verifies the resulting retained lock cold. A pre-ledger public
`corpus acquire --offline` refusal probe hides one fixture origin manifest while
keeping its copied snapshot intact; failure must preserve the acquisition lock,
create no transport budget and never reach the network tripwire. The fixture
manifest is restored byte-for-byte before the connected positive sequence.
All four fixture sources are retained; mixed production acquisition is covered
only by the separate mocked public-CLI source-effects tests above.

Independent follow-up static review found no new concrete defect in the inspected
changes and confirmed that the original four omissions were addressed. It did
not qualify production data, real acquisition, hardware or model behavior.

Final retained-origin connected result: **1 passed in 420.88s**, using
`uv run --locked --no-sync pytest -q --tb=short
tests/test_preparation_sequence.py::test_offline_preparation_sequence_through_native_supervision`.
The final run completed publication, substituted-release denial, cold prepared
bundle verification and zero-update/accounting/shutdown assertions with all four
retained-origin effects intact. This supersedes the earlier connected coverage
limitation without claiming real acquisition or corpus admission.

Final static checks: `ruff check .` passed; `ruff format --check .` passed
(843 Python files); `git diff --cached --check` passed; native research lint
reported 417 valid records and zero errors. No fitting, model work, production
processing, actual source acquisition, push or PR occurred. The original checkout
remains clean. No remaining concrete blocker was found in this bounded follow-up;
the production and scientific qualification limits above still apply.
