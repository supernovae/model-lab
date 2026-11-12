# Fixed study identity acceptance

Use the existing read-only `study report` route with optional fixed expectations:

```sh
klab study report /absolute/study.yaml /absolute/receipt.json \
  --evidence /absolute/collected-report.json --output /absolute/new-report-root \
  --identity-expectations /absolute/expected.json \
  --candidate-commit FULL_LAB_CHECKOUT_COMMIT
```

The JSON/YAML expectation record has exactly `format` (value
`klab-study-identity-expectations-v1`), `matrix_sha256`, `study_sha256`,
`config_identity_mode` (`current` or `legacy_default_fields`), and ordered
`config_sha256` (one SHA256 per run, at most 1000). These are fixed authored
expectations, not values to regenerate from the candidate under test.

Both flags are required together and the expectation bytes are read, parsed and
hashed once before native report verification. The report still verifies study,
receipt and collected evidence, including all run identities; acceptance then
compares the verified identities and receipt-selection mode. Invalid expectations,
mismatches or unverifiable candidate source refuse before output creation.

The emitted `identity_acceptance` records the expectation bytes hash, candidate
commit and native package source hash/algorithm. Every file in `source_identity()`'s
inventory must be clean and committed at that candidate commit; the recorded
`candidate_package_status` covers this entire inventory. The commit belongs to
whichever repository contains the loaded package; it is not automatically an
upstream Lab commit. Compare `candidate_package_sha256` across checkouts. Dirty
files, untracked package copies, and wheels under unrelated enclosing repositories
refuse. The native inventory excludes its self-referential lifecycle record and
non-inventoried file types. This is source-byte provenance, not attestation of
dependencies, interpreter, in-memory modifications or installed-runtime
equivalence. Use a stable checkout without concurrent source edits during
verification.

This migration check asserts identities of already-collected receipts using the
existing reader; it does not execute historical workflows or add an identity
algorithm. `legacy_default_fields` preserves the current read-only verification
guarantee, without promising indefinite support. Retiring the in-Lab personal
assertion does not itself retire that mode or reader. Any later removal requires
an explicit reviewed replacement-guarantee mapping and coordinated coverage.

Lab CI uses neutral local fixtures offline. It does not fetch external collections
or silently skip personal acceptance. An external collection must explicitly run
this command against each candidate it intends to accept and retain the report.
Passing once at an old pin is preservation evidence, not ongoing regression
coverage. Workshop has retired its personal memory-study collection under the owner's
three-family scope; that collection no longer imposes a per-candidate merge gate.
Its fixed expectations remain historical evidence. Neutral Lab regressions and
ordinary review continue to protect the generic interface.
