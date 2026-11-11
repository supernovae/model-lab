# Corpus Forge provenance and shape experiments

Corpus Forge stores acquisition receipts, normalized source documents, derived records, and frozen releases separately. A release's lineage ledger records three **independent**, versioned classifications per record:

- **Origin:** where the record came from (`primary_source`, `human_authored`, `deterministic_synthetic`, `source_transformed_synthetic`, `multi_source_synthetic`, `free_generated_synthetic`, or `model_self_generated`). The source snapshot and parent document IDs remain separate from this label. A prompt backed by a source is not the source itself.
  `sources/*.yaml` may explicitly declare `origin: human_authored` for independently authored input; unspecified source origin defaults to `primary_source`, not an inference of authorship from a local file path.
- **Verification:** what supports a particular answer or claim (`unverified`, `schema_validated`, `source_entailed`, `oracle_verified`, `cross_source_verified`, `human_reviewed`, or `rejected`). A verification receipt retains source passage and verifier rule, oracle inputs and comparison, or review protocol and decision, as appropriate. Malformed or rejected generated outputs remain inspectable; a model's own assertion is not independent verification.
- **Training shape:** how the example was constructed (`raw_document`, `direct_qa`, `troubleshooting_scenario`, `decision_record`, `tool_trace`, and other registered shapes). Normalized interaction, grounding, answer style, supervision, reasoning depth, task family, and source domains describe the rendering rather than inferred internal reasoning mechanisms. Rendered-text SHA-256 and shared parent document IDs let different shapes of one concept be queried together.

**Origin is not quality. Verification is not usefulness. Training shape is not truth.** Source-entailed QA may be poor training for tool use. An artificial but oracle-verified bounded scenario can be useful for a causal task. Fluent free-generated prose can remain unsupported. No scalar trust score, universal synthetic cap, or automatic recommendation follows from these fields.

Source declarations, normalized documents and lineage preserve `modality: text` / `modalities: [text]` explicitly. The current ingestion and rendering contracts are text-only; non-text data cannot be relabeled as text to pass validation. A future version must define a real acquisition, content identity and renderer before adding image, audio or multimodal records. This dimension is provenance for future assimilation comparisons, not evidence that such models have been trained.

For prospective acquisitions, [the v2 rights protocol](rights-policy.md) adds
file-level SPDX, source notice and explicit training-restriction evidence.
Rights, verification, origin and training shape remain distinct dimensions;
metadata-only publication does not republish the training export. Historical
Historical readiness decisions and release identities are not reinterpreted.

## Export storage and consumer authentication

`corpus export RELEASE --work-dir ROOT` stores immutable configs under
`ROOT/corpora/<corpus-id>/exports/<release-id>/<view>/<request-sha>`.
`ROOT` may differ from the verified release's workspace. The canonical suffix
must match the authenticated corpus, release and view; the request SHA binds
`release_id`, `view`, `base_config_sha256` and `vocab_size`, not the absolute
operational prefix. Do not relocate an accepted export to satisfy a consumer.

Consumers verify the full release, exact canonical export metadata, generated
config hashes, release/report/license-report hashes, split paths and hashes,
actual record/rendered-byte counts, license and applicable publication/training
policy. Both generated configs are model-validated and their complete dataset
declarations must equal the consuming dataset. Ceilings retain every record:
`max_documents = records`, `max_tokens = rendered_utf8_bytes + records + 1`.
Changing a consumer dataset field or rehashing an inconsistent generated config
does not bypass this contract. A selected external tokenizer path may differ;
tokenizer selection, vocabulary and provenance remain separately authenticated.

Operational paths inside configs and location-bound metadata can change file
digests between separately produced exports. That does not make the storage root
part of the canonical scientific request identity, nor permit editing an
existing immutable export. Typed artifact/evidence references still bind their
exact recorded file digests.

## Compare releases without reacquiring sources

One project may point to multiple checked-in release declarations while retaining the same `id`, source declarations, snapshots and acquisition lock. For example, keep `sources/*.yaml`, `splits.yaml`, and `transforms/*.yaml` fixed, then make project variants that differ only in the `release:` YAML reference. Declare `include_shapes` and/or `include_origins` in each release YAML. `corpus build ... --offline` verifies the shared acquisition lock, builds the selected view without mutating a frozen release, and `corpus freeze BUILD` publishes a new release ID. Inspect its `manifest.json` snapshots, `lineage.jsonl`, `audit.json`, and `report.json`; only then export the selected training view. A variant whose selected training split is empty fails freeze rather than silently changing a run.

A fraction-controlled variant may declare `fraction: {generated_share: 0.25, train_tokens: 2000, tokenizer_path: tokenizers/fixed.json, tokenizer_sha256: <full SHA-256>}`. The tokenizer must be a checked-in, regular project file matching its digest. Selection counts the **actual encoded training examples** (including chat rendering) and requires complete examples, the exact total and generated token counts, and representation of every candidate source family. An impossible fraction or budget fails; no bytes-per-token estimates or hidden replacement are used. Keep tokenizer, source families, splits, and total budget fixed between variants to isolate the attempted treatment. Equal token counts do **not** establish equal semantic information.

`corpus describe RELEASE` reports immutable record counts; token totals are `null` unless the release declares a pinned selection tokenizer. `corpus describe RELEASE --tokenizer PATH` verifies the release and recounts actual encoded view records by origin, verification, shape, generator and template, without changing the frozen report. The `training_mixture` denominator counts only selected **train** views; validation/test and unselected inventory remain separately visible. Watch both **record and token** proportions, duplicate and prefix concentration, lexical diversity, and shared source/world/template lineage across splits. Descriptive warnings are not corpus-quality judgments; holdout gates must be declared in the split policy.

`klab corpus near-duplicates RELEASE` verifies an immutable release and screens its normalized documents across train/validation/test with deterministic three-word shingles, 64 MinHashes in 16 bands and exact Jaccard scoring of proposed pairs. The JSON includes source locations, content hashes, pair similarity and bounded-work parameters; exceeding any cap fails instead of silently returning a partial result. This is a lexical **candidate review**, not an exhaustive match search or evidence of semantic independence. Resolve concerning pairs through a new reviewed split/source declaration and release; do not edit the frozen source evidence.

After training, attach each capability observation to its exact release ID, model/checkpoint, measured token budget and evaluation suite. A shape-by-capability matrix and transformation usefulness state (`HELPFUL_UNDER_TESTED_TASK`, `NO_MEASURED_EFFECT`, `MIXED`, `REGRESSION`, `UNKNOWN`) are meaningful only for those bound observations; do not add a single aggregate score or a global label to a shape. This release ledger also preserves lexical/semantic candidate parentage so future Engram-only, neural-only and combined assimilation studies can compare the same source knowledge without interpreting a candidate inventory as model training.

After a run has a verified checkpoint and a content-addressed capability card,
`klab corpus compare RELEASE --run-dir RUN` reports each card separately
alongside the selected training-shape mixture. Repeat `--with RELEASE RUN_DIR`
to compare at least two releases; the command requires identical source snapshot
IDs, model architecture, card protocols, and checkpoint token budgets. It
reports `UNKNOWN` usefulness per transform until paired evidence isolates the
variable. Its integrity checks bind results to a run/checkpoint and registered
card/scorer; they cannot establish an independent test population or causal
effect. A release with no capability results has no capability matrix.

## Tokenizer selection from a frozen release

Pilot tokenizer-bakeoff declarations (`schema_version: 2`) accept verified
rights-tracked release schemas 2 and 3. The release verifier remains the rights,
lineage and immutable-identity gate; accepting schema 3 does not waive its
training-use policy or admit legacy schema-1 releases.

The three candidates use one train-only fit receipt and independent validation
families. Reopening a bakeoff authenticates all candidate bytes/manifests,
fit/held-out samples, measured scores and the smallest-vocabulary winner under
the declaration's `near_best_ratio`. Consumer verification additionally requires
the selected candidate, not any tokenizer with a bakeoff marker. An incomplete
or hand-stamped marker is not a selection receipt.

Keep the winner at `BAKEOFF/candidates/VOCAB/tokenizer.json` with the original
`report.json`, samples and candidate manifests. Its manifest retains
`source: local_text`, the fit-sample revision and `corpus_forge_bakeoff` binding.
`data prepare` and authored tokenizer artifacts can verify that winner against
an LM export of the same release and vocabulary without changing provenance,
inventing a `corpus_export` sidecar or fitting again on the full export.
The artifact identifier is the candidate-directory name (for example `16384`);
the evidence-export identifier separately binds release ID and vocabulary.

Supply accounting streams distinct selected source documents and deduplicates
by normalized-content SHA; schema-2 pilots include unpaired/unclassified source
kinds, while legacy schema 1 retains its original nine-kind scope. Ordered LM
view totals count each emitted view record separately. General-education
membership uses lightweight document flags rather than a second full text map;
unclassified kind counts/bytes are streamed as counters. A RAM-censored bakeoff
without a complete authenticated report is not a selected tokenizer.

Worker preparation authenticates the original selection before sealing its
tokenizer/prepared arrays. Portable corpus-binding v2 carries the unchanged
selection-report bytes and fit/release binding, rather than labeling a bounded
fit as full-export training. Offline workers verify that metadata closure,
selected digest, vocabulary, release/export identities and sealed inventory;
they do not reacquire the corpus or refit the tokenizer. Existing full-export
tokenizers retain portable binding v1. Native runs preserve all verified metadata
members, including the selection report, after source locations become unavailable.
