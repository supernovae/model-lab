# Corpus Forge rights policy (prospective v2)

Rights are three independent decisions: **training eligibility**, **redistribution of source-backed material**, and **licensing of model weights**. No license name alone establishes legal permission for every use. In particular, copyleft source code is **not** an automatic ML-training prohibition and does **not** automatically set the license of trained weights. Source notices, license terms, contracts, collection restrictions, output risk, and any proposed distribution need their own review. A completed corpus build is evidence of implementation, not a legal opinion or a research result.

Existing v1 source declarations, acquisition receipts, frozen releases, exports, and historical readiness decisions retain their identities and behavior. Do not edit their declarations or reclassify old release evidence. New decisions use `schema_version: 2` on **every** source and the release declaration; mixed versions are rejected. The project index (`corpus.yaml`) remains schema v1. Build a new release ID rather than rewriting a v1 snapshot or receipt.

## Declare a source and release

Example source declaration (replace the example URI, revision, and notice with verified upstream values):

```yaml
schema_version: 2
id: upstream_docs
kind: git
canonical_uri: https://example.org/upstream.git
revision: 0123456789abcdef0123456789abcdef01234567
license: Apache-2.0
license_url: https://example.org/upstream/blob/0123456789abcdef0123456789abcdef01234567/LICENSE
rights:
  training_eligibility: eligible
  redistribution_mode: metadata_reconstruction_only
  weight_license_status: separate_analysis_required
  spdx_expression: Apache-2.0
  license_references:
    - https://example.org/upstream/blob/0123456789abcdef0123456789abcdef01234567/LICENSE
  notices:
    - Retain upstream copyright, Apache license and NOTICE when applicable
  allowed_boundary_paths: []
  nested_metadata_path: license-metadata.json
# Also supply required domains, document_kinds, source_family and acquisition.
```

`nested_metadata_path` is optional and valid only on Git sources; when declared, acquisition pins that `license-metadata.json` file even if it is outside the document include glob. The file and its SHA-256 are part of the immutable snapshot. Its REUSE-style directory/file/group rules are applied by specificity. SPDX headers within the first 30 lines and file copyright lines are recorded alongside source license references/notices. A file-specific SPDX license may override a root default; contradictory file header versus scoped metadata, unknown IDs/exceptions, malformed expressions, and unmatched external-material boundaries require review. `vendor/`, `third_party/`, `external/`, `generated/`, `bundled/`, and `fixtures/` are boundaries; only individually reviewed paths or prefixes should appear in `allowed_boundary_paths`. A blanket allowlist is not a substitute for inspecting included files. Only recognized expressions are classified automatically. Source-level explicit `training_restriction: {kind: prohibited, basis: ...}` requires `ineligible`; `conditional` requires `review_required` or `ineligible`. Record the exact basis, not a guessed SPDX implication.

Declare references for **all** applicable file-level license exceptions, not
only the root license; the resolver records declared URLs but cannot prove
that a URL contains the identified terms.

Training states are `eligible`, `eligible_with_obligations`, `review_required`, and `ineligible`. Only the first two enter normalized training documents; review-required and ineligible file paths remain in the pinned inventory and rejected evidence without training. Redistribution states are `redistributable_under_source_terms`, `metadata_reconstruction_only`, `derived_only`, `not_redistributable`, and `review_required`. A training-eligible file may still be nonredistributable. In particular, `eligible_with_obligations` preserves attribution/copyleft obligations without asserting that these govern trained model weights. The weight field remains `separate_analysis_required`; decide any model-weight license separately, with counsel where needed.

When identical normalized source text carries incompatible licenses, notices,
references, training restrictions or redistribution states, a v2 build fails
with an audit diagnostic rather than arbitrarily selecting one duplicate's
rights. Resolve attribution upstream and create a newly reviewed declaration.


A v2 release declaration must include one of:

```yaml
schema_version: 2
publication_mode: metadata_reconstruction_only
# Or: redistributable_under_source_terms, after reviewing each included source.
# Keep the existing mixture, view, generation and selection fields.
```

`metadata_reconstruction_only` is the conservative default **choice**, not an implicit default in the schema. Local `corpus export` remains an internal training artifact: it is **not** a publication command. `klab corpus publication RELEASE` verifies the frozen release and emits a JSON metadata manifest with source URIs and pinned revisions, acquisition settings, file paths/hashes and per-file rights decisions, project references and transform identifiers; it copies **no** raw snapshots, normalized source text, generated/derived records, tokenizer or weights. For a public package, include this metadata, actual license/NOTICE files and separately reviewed acquisition scripts/configs; obtain source-backed material from pinned upstream, reconstruct locally, and only attach synthetic/derived records reviewed for their own rights. Do not republish a local training export as a public dataset. `redistributable_under_source_terms` expresses a source-terms publication decision, not automatic bundling or relicensing.

`license-report.json` and `report.json` separately count file eligibility, redistribution modes, exact observed SPDX expressions, unresolved decisions and train-source documents. Eligible source token counts are `null` on the frozen report until a tokenizer is specified; `klab corpus describe RELEASE --tokenizer TOKENIZER_JSON` measures **unique retained train source passages**, separately from rendered LM/chat tokens or total generated examples. Counts exclude review-required/ineligible paths and are not proof of licensing adequacy. Snapshot file hashes, source URI/revision, license URLs, rights evidence and file decisions are verified at release read time. A pinned v2 Corpus Forge release can be used through the existing ExperimentPlan corpus-release reference and local export; a policy never overrides the ExperimentPlan artifact identity or automatically satisfies a readiness gate.

See [memorization diagnostic](memorization.md) for an optional bounded continuation/source overlap check. It records overlapping passages and edit/ngram scores for output review; it does **not** decide copyright, training eligibility or corpus construction.

## Prospective private-research selection

This selection policy uses `schema_version: 3` on **all** source declarations and the release; the project index remains schema v1. `explicit_training_restriction` on each source and `training_use_policy: allowed_unless_explicitly_prohibited` on the release record the selection rule. A public, attributable source without a documented incompatible ML-training restriction may be considered for private research even when its license is not recognized by the file-level SPDX classifier. This does **not** interpret a dataset's license as the license for every upstream page or paper; preserve source, file, page/paper and attribution evidence separately. Deny explicit incompatible bans, required ungranted authorization, bypassed gated access, unknown origin, secrets and private/confidential records. An absent person-by-person attestation is not an explicit no-training term.

V2 historical eligibility and incompatible-rights duplicate fail-gates remain unchanged. V3 resolves cross-source exact text to one local representative while retaining every attribution origin and rights decision in the audit/receipts. An identical train/held-out source passage belongs to the held-out pool; overlapping validation and test copies fail. Approximate overlap and different text from the same URL or paper still require contamination review. The v3 `publication` command emits reconstruction metadata only. A frozen private corpus, a local training export, permission to publish raw data, and permission to license/publish future model weights are four separate decisions. No final tokenizer or training follows from freezing the source inventory.

### Opt-in row attribution

Declared HF admission sources may select `row_use_policy.mode: record_attribution`
(see [corpus preparation](corpus-preparation.md#record-attribution-under-an-explicit-application-policy)).
This applies the attribution-first processing choice at original-row granularity:
retain labels, unknowns and notice conflicts for audit; enforce explicit restrictions
on intended use, independently of quality filters. It is consistent with the v3
`allowed_unless_explicitly_prohibited` direction, but does not change the source
declaration, grant permission or constitute legal clearance. The default code-label
gate remains when this opt-in policy is absent.
