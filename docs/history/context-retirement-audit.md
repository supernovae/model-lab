# Context-family retirement audit

Later follow-up: [memory retirement](memory-retirement.md) records removal of the
four temporarily retained shared inputs; [the allocation boundary](path-allocation-boundary.md)
records neutral fixture coverage and remaining path-domain policy. The dated audit
rows and JSON below remain historical evidence, not current work requirements.

Current follow-on: [memory retirement](memory-retirement.md) completes the four
shared context inputs and eight memory-injection active copies after reviewed
Workshop ownership/native acceptance. The audit JSON and dated findings below
remain the original evidence; use the follow-on map for current file presence.

Follow-up: [the retirement record](context-retirement.json) retires 30 context
assets and redirects the guide. Four shared context inputs remain for the
memory-injection historical-identity reader. The pinned audit below records the
pre-retirement state; its original identities and ledger observations are unchanged.
Fresh basename/relative-path scanning found that the retained memory matrix uses
`base_config: context_study_dense_s17_b24k.yaml` and its study reads three context
cards. This shared closure prevents retiring all 34 context assets in isolation.

This is a disposition/coverage audit, not an asset retirement. Lab input pin:
`7e306a96bcc617468d2a06493dab4562bff6beb0`. Landed Workshop preservation pin:
`0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4`.
The [exact map](context-retirement-audit.json) records 60 current Lab files:
34 configuration declarations, 20 data/ledger/card files and six study artifacts.
All 60 match landed Workshop bytes, manifest SHA256/size and destination Git blob.
No source, declaration, test, catalog, frozen guide or historical artifact changes
in this audit. Generic runtime/smoke examples are excluded.

## Coherent families and smallest next slices

| Family | Files | Retained dependencies / next safe slice |
| --- | ---: | --- |
| Context Engram study | 34 | 25 configs, eight data/card/ledger files and one result artifact. Their frozen guide is separately preserved exactly. Reconcile its current integrity-map entry, original ledger and historical guide links together before retiring this family; no new training is needed for retirement. |
| Memory injection | 8 | Two declarations, two card/ledger files and four study artifacts. `test_study_reporting.py::test_historical_default_fields_preserve_collected_study_identity` still reads the original study to fence historical digests. Keep that guarantee through exact historical input fixtures or an explicitly scoped historical reader before removing declarations. |
| Path domain | 18 | Seven configs, ten data/card/ledger/audit files and one study artifact. `test_allocation_research.py::test_allocation_builder_binds_published_development_audit` still fits a tokenizer from this corpus and checks the builder. First replace generic coverage with a tiny neutral fixture; classify scientific constants in `data/allocation_tasks.py` separately. |

The context-only 34-file family is the smallest coherent asset-retirement
candidate from this audit. It is not deleted here: the frozen guide/ledger and
historical integrity references need a coordinated retirement record. The
memory-injection and path-domain families must not be swept into that change.
Archive membership is a preservation prerequisite, not sufficient deletion proof.

## Coverage already retained and coverage still required

Lab PR178 replaced generic personal-input dependencies with neutral fixtures.
`tests/context_fixtures.py` and
`test_study.py::test_neutral_memory_placement_plan_and_composition_card` retain
composition-card schema/scorer controls, tokenizer-relative context-window checks,
parameter inventories and memory placement planning without a scientific run.
`test_config.py::test_memory_injection_serialization_preserves_legacy_final_config`
retains the default serialization fence. Those guarantees do not replace frozen
study identity or allocation provenance/normalization checks.

Literal full-path scanning finds consumers in the frozen context guide, the
existing context-fixture disposition map, allocation source, allocation tests and
study-reporting tests. The JSON map records each literal match. It is not a complete
semantic dependency detector: split paths, generated names and basenames require
manual review. In particular the allocation test builds `data / path_domain_v1`,
and study-reporting preserves historical matrix/config digests. Keep generic
historical readers and exact identity checks; never substitute fresh results for
old evidence or add a historical execution adapter.

## Declared payloads versus preserved evidence

The original context input ledger declares 39 identities. Thirty-two files exist
and match their recorded SHA256/size in both pinned trees. Seven tokenizer/prepared
payloads are absent from both pinned tracked trees:

- `artifacts/tokenizer_context_study/tokenizer.json`
- `artifacts/tokenizer_context_study/tokenizer_manifest.json`
- Under `artifacts/tokenizer_context_study/prepared_data/700fe5e374dd8f13/`:
  `manifest.json`, `train.npy`, `validation.npy`, `train_supervision.npy`,
  `validation_supervision.npy`.

This says nothing about other hosts or external artifact stores. No recovery or
reconstruction was attempted, and their old recorded hashes were not changed.
The preserved guide/result may describe historical execution; this audit does
not establish current possession of those payloads or runtime readiness.

`docs/context-engram-study.md` remains byte-identical to its landed archive.
`docs/path-domain-corpus.md` is already a current navigation guide and differs from
its archived original; do not confuse a valid navigation edit with lost original
evidence. Future guide retirement must preserve the frozen origin and update
active navigation/integrity contracts deliberately.

## Validation and limits

All 60 destination comparisons, manifest hashes/sizes/blobs, the frozen guide and
32 present ledger entries were checked locally without network payload access.
No scientific suite, tokenizer fit, model work, historical replay or runtime probe
was run by this audit. Lab #157 remains partial. Further implementation should
preserve scientific choices in Workshop and generic functions/tests in Lab, with
new identities for any later reproduction.

Existing focused checks executed (no model or tokenizer fit):

```sh
python -m pytest -q \
  tests/test_study.py::test_neutral_memory_placement_plan_and_composition_card \
  tests/test_config.py::test_memory_injection_serialization_preserves_legacy_final_config \
  tests/test_study_reporting.py::test_historical_default_fields_preserve_collected_study_identity \
  tests/test_workspace_hygiene.py::test_retained_historical_evidence_keeps_exact_bytes
```

Four passed in 2.59 seconds. This validates retained metadata/history guards;
it does not establish that the still-specialized allocation test is replaceable.
