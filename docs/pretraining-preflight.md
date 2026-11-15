# Pretraining preflight

Before a run of days, `klab preflight` proves that the release, tokenizer,
mixture, prepared data, model config and warmup agree with each other and with
the plan. It reads what earlier phases already produced and verified; it does
not rescan the corpus, so it takes minutes on fast storage. It writes one
sealed report with PASS, WARN or FAIL per check and the evidence for each, and
exits non-zero on any FAIL. `klab train` refuses a fresh pretraining run
without a report that is current for the exact config and artifacts.

## Run it

After `klab stage CONFIG --through warmup --output STAGE` (see [corpus
preparation](corpus-preparation.md#run-it-as-phases) for the phases before it):

```sh
klab corpus near-duplicates RELEASE > near-duplicates.json
klab data engram-sizing PREPARED --config CONFIG --target-tokens TOKENS --output engram-sizing.json
klab preflight CONFIG --output PREFLIGHT_DIR --stage-bundle STAGE \
  --near-duplicates near-duplicates.json \
  --near-duplicate-decisions near-duplicate-decisions.yaml \
  --engram-sizing engram-sizing.json --recipe RECIPE_DIR
klab train CONFIG --stage-bundle STAGE --require-prepared \
  --preflight PREFLIGHT_DIR/preflight-DIGEST.json
```

`CONFIG` is the composed config (`dataset.source: local_token_mixture`). The
release, tokenizer and mixture are found through it. The release is verified
with its signed receipt unless `--cold-verify` is given. The decision record is
described under [near-duplicates](corpus-preparation.md); the sizing
result under [engram memory](engram.md#size-the-tables).

## What each check protects against

| Check | Evidence it reads | Protects against |
| --- | --- | --- |
| `release` | the release verification (receipt) and its recorded build policy | training on an unverified or hand-edited release; a release built under a policy nobody can name |
| `sources` | the build report's per-source ledger | a rights rule silently removing 13 of 33 sources (all the code) while build, freeze and audit passed. FAIL if a declared source retained nothing; WARN if one lost more than `--max-set-aside-share` of its files or records |
| `quotas` | the mixture receipt (tokenizer-based counts per stratum) | a stratum that cannot fill its quota. FAIL if unique tokens times the stratum's allowed exposures fall short; WARN if the receipt shows an under-filled stratum |
| `validation` | the build report's per-source validation sizes, grouped by the mixture's strata | an empty or one-sided validation split. FAIL if empty; WARN naming strata with no validation documents; sizes are recorded |
| `near_duplicates` | the screen's report and the reviewer's decision record | validation text that has near-copies in training. FAIL unless the screen ran on this release, covers every document, is not truncated and every reported pair is decided |
| `tokenizer` | the tokenizer manifest and the mixture receipt | a tokenizer that differs from the one the mixture was built with, an unaudited tokenizer, round-trip failures |
| `mixture` | the same verification training uses (`verify_mixture_dataset`) | a mixture that does not match this release, tokenizer or config |
| `prepared` | the same cache lookup training uses (`--require-prepared`) | prepared arrays built from another mixture or tokenizer |
| `advisories` | `klab inspect` advisories and a bound engram-sizing result | Engram tables never measured on these tokens (WARN unless the sizing result names this prepared data and these table sizes and its clobbered share is within its criterion) |
| `warmup` | the stage bundle's warmup pilot | a config that does not fit the card. FAIL if the measured device peak exceeds the card's memory or the bundle holds other prepared data; WARN under 5% free. Records peak, targets/s and the device count |
| `revisions` | Git state of Lab and, with `--recipe`, the recipe | runs from uncommitted or unmerged code: WARN if either tree is dirty, not in `origin/main` (as last fetched), or the recipe is not recorded |

A check that cannot run (a missing input, an unreadable artifact) is a FAIL
that says what is missing; nothing is skipped.

## Reading a FAIL

Open the report. Each check has a one-line `summary` and an `evidence` object
with the numbers it compared. The failing check names the artifact to fix:
rebuild or re-materialize the stage that produced it, then run the later
phases and the preflight again. A changed config, Lab source, mixture receipt,
tokenizer, prepared data or stage bundle makes an older report stale; the
report's `subject` lists the digests it binds.

## The training gate

A fresh run on a token mixture needs `--preflight`:

- `--preflight REPORT` accepts a report that is sealed, not FAIL, and bound to
  this config, this Lab source, this mixture receipt, this tokenizer, and the
  prepared data and stage bundle the trainer opens. The run manifest records
  the report's digest and status. Warnings are allowed and recorded.
- `--preflight waive:REASON` runs without a report. The reason is recorded in
  the run manifest as `preflight.waived`.

Resumes, promotions, recoveries and budget extensions continue a run that was
admitted when it started and are not gated.

## Basis for the thresholds

| Threshold | Value | Basis |
| --- | --- | --- |
| `--max-set-aside-share` | 0.25 | Judgment, not measured: set-aside is routine (empty files, non-permissive licences in a repository), but a quarter of a source's files means its character may have changed. Lower it for sources that should be nearly whole. |
| Warmup headroom | 5% of the card | Judgment: allocator fragmentation and driver reserve; a sampled peak is a lower bound (`peak_method`). The FAIL line, a peak above the card's memory, is exact. |
| Quota exposures | the mixture's `max_exposures` | The mixture itself refuses a schedule beyond it; the check repeats that arithmetic on the receipt. |
| Engram clobbered share | the sizing result's own `--max-clobbered` (default 0.25) | See [engram memory](engram.md#size-the-tables). |
