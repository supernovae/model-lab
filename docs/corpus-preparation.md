# Corpus preparation

For a training corpus of billions of tokens drawn from several sources, Lab
prepares data in stages, each producing an immutable, verifiable artifact:

1. **Acquire** pinned sources into snapshots.
2. **Admit** rows or files: draft decisions, inspect a sample, review.
3. **Split** documents into train, validation and test by family, so related
   documents never cross splits.
4. **Build and freeze** a release: normalized documents, rights evidence and
   lineage.
5. **Export** a view of the release for a tokenizer and base config, fit the
   tokenizer and **materialize** a token mixture.
6. **Prepare** the mixture for training like any other dataset.

Every stage verifies its inputs, resumes after interruption from its last
committed point and refuses to publish a partial result. For a dataset that one
snapshot covers, [`klab data snapshot`](datasets.md#pin-a-dataset) is enough;
use this pipeline when you need admission review, family splits, rights
tracking or a mixture of sources.

`tests/fixtures/corpus-sample` is a small offline project that runs through
every stage.

## Run it as phases

A project's stages are listed once in a `preparation.json` phase map. Lab
expands each phase into its exact command and runs one supervised phase at a
time, with resource accounting:

```sh
klab attempt init --ledger "$LEDGER" --contract contract.json --contract-sha256 "$CONTRACT_SHA256"
klab attempt phase-command --plan preparation.json --attempt-root "$ATTEMPT" \
  --label acquire --bindings-json "$BINDINGS"
klab attempt run-phase --plan preparation.json --attempt-root "$ATTEMPT" \
  --label acquire --bindings-json "$BINDINGS" --ledger "$LEDGER" \
  --workspace "$WORK" --policy monitor.json --preparation-policy preparation-policy.json \
  --baseline baseline.json
klab attempt status --ledger "$LEDGER"
```

`attempt init` creates the budget ledger an attempt runs under.
`phase-command` prints a phase's exact command without running it; `run-phase`
runs it under the monitor policy and charges the ledger. `attempt
bind-artifact` records an artifact produced outside the attempt. The commands
below are what the phases run.

## Acquire

```sh
klab --work-dir WORK corpus acquire PROJECT
```

A project (`corpus.yaml`) lists source declarations under `sources/`. A source
is a pinned Git repository, Hugging Face dataset shard, Wikimedia dump, local
file, HTTP document or generator. Acquisition writes one snapshot per source and
then an acquisition lock binding them.

- **Transport budgets.** `corpus budget-init PROJECT` starts a durable ledger
  of bytes, retries, disk and wall time for bounded Hugging Face transfers;
  `corpus budget-status PROJECT` reads it. Failed and interrupted transfers keep
  their full charge.
- **Initial feasibility gate.** A budget may set `initial_body_bytes` below
  `max_total_body_bytes`. Acquisition stops at the initial cap until
  `corpus budget-release-gate PROJECT --report REPORT --review REVIEW` records
  a reviewed decision to continue.
- **One source at a time.** For a large project, run each source as its own step:

  ```sh
  klab --work-dir WORK corpus acquire PROJECT --source-id SOURCE_ID \
    --max-new-bytes REMAINING_BYTES --min-free-bytes FREE_FLOOR_BYTES --plan
  klab --work-dir WORK corpus acquire PROJECT --source-id SOURCE_ID \
    --max-new-bytes REMAINING_BYTES --min-free-bytes FREE_FLOOR_BYTES
  ```

  `--plan` verifies everything without network access or writes. A step returns
  `source_verified` and `complete: false`; once every source is present, plain
  `corpus acquire PROJECT` publishes the lock, reusing the snapshots.
- **Observed rather than enforced limits.** On a local workflow where the owner
  wants time, network or storage observed rather than capped,
  `corpus budget-observe-time`, `budget-observe-network` and
  `budget-observe-storage` switch that one policy on the existing ledger, with
  `--owner` and `--approval-reference`. History and charges are kept. Storage
  observation still enforces a `--min-free-bytes` floor.
- **Finalize without fetching.** `corpus acquire PROJECT --require-existing`
  publishes a lock from snapshots already on disk and refuses if any source is
  missing; it never downloads. `--offline` verifies an already published lock.
- **Retained subsets.** A Hugging Face source can declare
  `acquisition.retained_subset` to take a seeded 1% row sample of an existing
  snapshot (pinned by path and digests) without any network access.
- **Preview.** `corpus preview DECLARATION --input-root INPUTS --output NEW_DIR`
  shows seeded sample excerpts from acquired JSONL in place, without copying or
  admitting anything.

## Admit

```sh
klab corpus admission-draft PROJECT --template template.json \
  --policy-document policy.json --output draft.json
klab corpus inspect-admission --draft draft.json --policy-document policy.json \
  --selection selection.json --output inspection --max-input-bytes 1073741824 \
  --max-excerpt-bytes 2048 --max-output-bytes 16777216
```

`admission-draft` writes a conservative decision (qualify, exclude or
quarantine) for every row or file of the verified snapshots. `inspect-admission`
draws a deterministic sample for a reviewer; reissuing it after an interruption
resumes. A reviewed admission (v2) must be bound to that inspection.

### Qualifying Hugging Face rows

A bounded Hugging Face source can declare `acquisition.qualification` with a
profile and the original row fields to use:

```yaml
qualification:
  profile: code
  family_fields: [repo_name]
  language_field: language
  allowed_languages: [Python, JavaScript, TypeScript, Go, Rust, Shell, SQL, Markdown]
  repository_field: repo_name
  path_field: path
  license_field: license
  license_labels: {mit: MIT, apache-2.0: Apache-2.0, bsd-3-clause: BSD-3-Clause}
```

- **`code`** groups a whole repository into one family and quarantines rows
  with missing or unmapped license labels, disallowed languages, vendored or
  generated paths, detected credentials or conflicting notices.
- **`synthetic_text`** requires `family_fields` and
  `synthetic_provenance_fields`; only the acquisition `text_field` becomes text.
- **`web_text`** requires family fields and optionally an allowed language
  list. Dotted paths such as `metadata.url` reach nested fields.

An admission template may also set `row_use_policy.mode: record_attribution` for
such a source: rows keep their license labels and notice lines (unknowns
included) as attribution rather than passing a license allowlist, and rows with
a known explicit restriction are listed by original-row digest with the basis
and evidence, then excluded.

## Split

```sh
klab corpus split-inventory PROJECT --output inventory.jsonl
klab corpus freeze-splits PROJECT --inventory inventory.jsonl --clusters clusters.yaml --output splits.yaml
```

`split-inventory` records document IDs and family hints from verified
snapshots. A reviewed family-cluster declaration merges related families, maps
sources to strata and sets the seed; `freeze-splits` assigns whole families to
splits.

- **`split_weights`** sets family ratios, for example
  `{train: 98, validation: 1, test: 1}` (8/1/1 by default). They are ratios of
  families, not of bytes or tokens. A stratum needs at least as many eligible
  families as the weights' sum.
- **`family_fields_by_source`** groups a source's rows by original fields (for
  example `{cosmopedia_shard: [prompt]}`) without reacquiring it. Rows missing
  those values cannot enter training splits.
- **`prior_family_inventory`** adds a new release while keeping an earlier
  release's family assignments fixed.
- **Output.** The splits YAML points to a digest-named
  `assignments-DIGEST.jsonl` companion; keep the two together. Exact content
  that would cross splits, or a family spanning strata, is refused.
- **Interrupted?** Repeat the identical command with `--recover`. It
  revalidates inputs and every existing member, then publishes only what is
  missing; a conflicting file stops it.
- **`--reuse-completed-preparation`** freezes from an already completed local
  inventory and admission without repeating source verification. The receipt
  records which checks were deferred.

## Build and freeze

```sh
klab --work-dir WORK corpus build PROJECT
klab --work-dir WORK corpus freeze BUILD
```

`build` normalizes admitted documents, applies the frozen splits, removes exact
duplicates (refusing cross-split or incompatible-rights duplicates) and writes
documents, spans, lineage and language-model and chat views. `freeze` publishes
an immutable release with a content-derived ID.

`--reuse-completed-preparation` on both commands continues from completed local
preparation: `build` renders admitted shards without replaying the acquisition
audit, and `freeze` publishes the release in place without copying the payload.
Both record what was deferred. Keep the work root and its completion records
with the release. Completed phases are keyed by their inputs, configuration
and every corpus module they can run, so a change to the corpus code renders
them again rather than reusing old shards.

Each transform file in the project names one `kind`:

- `lm_text`: source documents as language-model text.
- `lexical_candidates`, `semantic_candidates`: candidate terms and key/value
  facts drawn from sources.
- `source_qa`: question-answer pairs over literal key/value spans.
- `manual_semantic`: hand-written records from a snapshotted file.
- `inference_qa`: recorded model responses, replayed, never generated live.
- `deterministic_scenarios`: worlds from a registered generator and its oracle.
- `chat_sft`, `tool_episode`: chat and tool-trace views of the records above.

`release.normalizer: normalizer-structure-v3` records retained line spans,
removed spans with reasons and document structure, and drops metadata-only front
matter from language-model views.

A Markdown file becomes one document per heading section. A source declaration
with `markdown_documents: files` keeps each file whole as one document instead,
for long-context training. The choice is part of the source declaration, so it
changes the build's identity; a declaration without it builds as before.

### Rights

Each source declaration records three independent decisions: training
eligibility, redistribution and the license of trained weights. No license name
settles all three.

```yaml
schema_version: 2
id: upstream_docs
kind: git
canonical_uri: https://example.org/upstream.git
revision: 0123456789abcdef0123456789abcdef01234567
license: Apache-2.0
rights:
  training_eligibility: eligible
  redistribution_mode: metadata_reconstruction_only
  weight_license_status: separate_analysis_required
  spdx_expression: Apache-2.0
  license_references: [https://example.org/upstream/blob/0123456789abcdef0123456789abcdef01234567/LICENSE]
  notices: [Retain upstream copyright, Apache license and NOTICE when applicable]
```

- **Training states:** `eligible`, `eligible_with_obligations`,
  `review_required` and `ineligible`. Only the first two produce training
  documents.
- **Redistribution modes:** `redistributable_under_source_terms`,
  `metadata_reconstruction_only`, `derived_only`, `not_redistributable`.
- **File-level evidence.** Git sources may pin a REUSE-style
  `nested_metadata_path`; SPDX headers in the first 30 lines of each file are
  recorded.
- **Release policy.** The release declares a `publication_mode`. With schema 3,
  sources record `explicit_training_restriction` and the release declares
  `training_use_policy: allowed_unless_explicitly_prohibited`; identical text
  from several sources keeps one copy and every origin's attribution.
- **Conflicts.** Identical text with incompatible rights fails the build.
- **Reports.** `license-report.json` counts eligibility, redistribution modes
  and SPDX expressions. `klab corpus publication RELEASE` emits a metadata
  manifest (source URIs, revisions, file paths and hashes) for sharing what was
  used without the text. `corpus export` is never a publication.

### Lineage

Every record carries three independent labels:

- **Origin:** `primary_source`, `human_authored`, `deterministic_synthetic`,
  `source_transformed_synthetic`, `multi_source_synthetic`,
  `free_generated_synthetic` or `model_self_generated`.
- **Verification:** `unverified`, `schema_validated`, `source_entailed`,
  `oracle_verified`, `cross_source_verified`, `human_reviewed` or `rejected`,
  with the evidence behind it.
- **Training shape:** `raw_document`, `direct_qa`, `troubleshooting_scenario`,
  `decision_record`, `tool_trace` and other registered shapes.

None of them is a quality score.

### Inspect a release

| Command | Shows |
| --- | --- |
| `klab corpus describe RELEASE [--tokenizer PATH]` | Record counts by origin, verification and shape; token counts with a tokenizer |
| `klab corpus sources RELEASE` | Sources and their rights |
| `klab corpus audit RELEASE` | The build audit |
| `klab corpus sample RELEASE --limit N` | Sample records, filterable by domain, kind and status |
| `klab corpus review RELEASE --kind chat` | Records to read before accepting a release; same filters as `sample` |
| `klab corpus lineage RELEASE RECORD_ID` | One record's lineage |
| `klab corpus near-duplicates RELEASE` | MinHash near-duplicate pairs across splits |
| `klab corpus consumers RELEASE --runs-dir RUNS` | Runs that trained on it |

`near-duplicates` compares every document, including dropped ones, against the
other splits by three-word shingles and reports pairs at Jaccard 0.65 or more,
plus identical text. It is a lexical screen: it can miss pairs and does not
catch paraphrases. `--max-documents`, `--max-input-text-bytes`,
`--max-total-shingles`, `--max-comparisons` and `--max-candidates` bound the
work; exceeding one stops the command with no partial report.

## Guard held-out data across releases

When a new release replaces one whose validation or test documents back a
fixed evaluation, check that none of those documents, their families or their
exact text moved into training:

```sh
klab corpus finalize-family-inventory RELEASE --splits splits.yaml --output family-inventory.jsonl
klab corpus audit-protected-lineage --prior-release PRIOR --candidate-release RELEASE \
  --prior-inventory prior-family-inventory.jsonl --candidate-inventory family-inventory.jsonl \
  --profile profile.json --suite suite.json --output lineage.json
```

- **`finalize-family-inventory`** writes one row per kept document of a
  verified release (`document_id`, `family_id`, `split`, `stratum`,
  `content_sha256`) from the frozen splits it was built with. It refuses a split
  file that differs from the release's and never overwrites an output.
- **`audit-protected-lineage`** verifies both releases and both inventories.
  The profile (`loss_slices`) and suite (`items`, `chunks`) name the protected
  validation and test documents and must be bound to the prior release and
  inventory. The receipt lists protected families or content now in the
  candidate's training split, and protected documents whose family, split or
  stratum changed. Any of these makes it `BLOCKED` and the command fails;
  otherwise `PASS`.
- **Acceptance.** A `klab-release-review-v1` record (decision `ACCEPTED`, the
  reviewer, the release, its family inventory and its lineage receipt, each
  with its SHA-256) is what an [attempt](#run-it-as-phases) binds as its
  `release_acceptance` artifact. It is refused unless the lineage receipt is
  `PASS` for this exact release and inventory.

## Export and tokenize

```sh
klab --work-dir WORK corpus export RELEASE --view lm --base-run-config base.yaml --vocab-size 32768
klab corpus verify-export RUN.yaml
```

An export writes a tokenizer config and a run config whose `dataset` binds the
release, view, base config and vocabulary. Consumers must keep that `dataset`
section exactly; `verify-export` checks it.

### Fitting a tokenizer on a byte sample

To fit on a weighted sample of strata, add `sampling` to the exported tokenizer
config:

```yaml
vocab_size: 32768
sampling:
  source_strata: {web_shard_000: real_english, synthetic_shard_000: synthetic, code_shard_000: code}
  byte_weights: {real_english: 60, synthetic: 20, code: 20}
  max_fit_bytes: 536870912
  audit_documents_per_stratum: 1000
  max_audit_bytes: 67108864
```

`klab tokenizer train` takes whole training documents in release order until
each stratum's byte cap is filled. This is a deterministic prefix, not a random
sample. It then audits different training documents for round-trip failures,
unknown tokens and bytes per token, and refuses to publish on a failure. No
validation or test document is used.

`klab corpus tokenizer-bakeoff DECLARATION --output DIR` instead fits several
vocabulary sizes on one training sample, scores them on held-out families and
selects the smallest within `near_best_ratio` of the best.

## Materialize a mixture

```sh
klab corpus materialize-mixture DECLARATION --output OUTPUT
klab corpus verify-mixture DECLARATION --output OUTPUT
```

A mixture declaration sets per-stratum token quotas, `max_exposures` (1 for
unique tokens only, or 2), the tokenizer and optional
`unique_train_content_floors`. Materialization encodes documents into a binary
token stream (`train.tokens.ids`) and a small record file; a run then uses it
through `dataset.source: local_token_mixture`. Nothing is published until every
quota and floor is met; a shortfall is reported per stratum.

Strata are written in name order and documents in seeded rank order within a
stratum. By default (`packing: contiguous`) the records follow each other and
training cuts the stream every `seq_len` targets, so a row holds several short
documents and a long one starts mid-row. Two optional settings change that:

```yaml
max_exposures: 1
packing: document_start
row_tokens: 2048            # must equal training.seq_len
strata:
  reference:
    max_exposures: 4        # repeat this stratum only
  books:
    min_document_tokens: 2048
```

- **`strata.NAME.max_exposures`** (an integer of 1 or more) repeats one
  stratum; strata without it use the mixture-wide `max_exposures`.
- **`packing: document_start`** with `row_tokens` and at least one
  `strata.NAME.min_document_tokens` starts every record of that many targets or
  more (its EOS included) on a row boundary. The space before it is masked
  filler ending in EOS, so its first token is read after an EOS as in a
  contiguous stream. Its tokens stay contiguous, so a long document fills
  consecutive rows in order; records after it fill the rest of its last row.
  Each training row is an independent example, so shuffling rows never mixes
  another document into one. Packing refuses a run whose `training.seq_len`
  differs from `row_tokens`.

A declaration that sets neither keeps its identity and its receipt. When
`strata` is set the receipt adds `stratum_max_exposures` and
`stratum_exposures` (records and target tokens per exposure, per stratum).
`document_start` adds `row_layout`: `row_tokens`, total `rows`,
`filler_positions` and, per stratum with a minimum, the number of row-start
`documents` and of `rows` whose every target belongs to one such document.
`materialize-mixture` and `verify-mixture` print both, and `verify-mixture`
recomputes them from `train.records.jsonl` and refuses a receipt that differs.
`training.matched_stories` still refuses a mixture, whose packed data always
carries a loss mask; `document_start` is the mixture's way to give long
documents their own rows.

- **Resume.** An interrupted run resumes from its last committed segment in
  `.OUTPUT.assembly`; keep that directory and the output parent's `cache`.
  Once the output is published, the segments and commit markers are removed;
  `progress.json` stays.
  `KLAB_MIXTURE_COMMIT_CANDIDATES=N` sets how often it commits (default 4096
  candidates, 64 MiB or 30 seconds).
- **Workers.** Encoding uses the shared [encoder pool](token-store.md#one-encoder)
  and reuses a [content token store](token-store.md#tokens-by-content), so
  rebuilding a mixture does not tokenize unchanged documents again.
  `KLAB_MIXTURE_WORKERS=N` caps the worker plan; by default it follows physical
  cores and memory.
- **Progress.** `.OUTPUT.assembly/progress.json` reports per-stratum counts,
  queue peaks, worker utilization and memory.
- **Verify.** `verify-mixture` checks the recorded completion and current
  inputs without rerunning; `--audit` reproduces the whole schedule without the
  token cache.
- **Several releases.** `additional_releases` adds one separately built release
  whose splits are bound to the baseline's families.

`tools/benchmark_mixture.py` runs a synthetic mixture cold and warm and writes
timings and output hashes, for comparing two Lab revisions on one host.

A mixture declaration's `tokenizer_observation` (`method:
post_run_observation`) reuses a tokenizer from a completed local job by binding
that job's records and the tokenizer files, instead of verifying the fit from
scratch.

## Resources

Admission, splitting, building and mixtures stream metadata through temporary
on-disk databases under the work root and spool encoded data to disk, so memory
is bounded by individual records rather than corpus size. Individual rows are
limited to 8 MiB in admission and splitting and 64 MiB in mixtures. Scratch
space is real: indexes, spools, committed segments and the output coexist until it is
published, so
watch free space on the corpus filesystem throughout. Reuse of completed work
depends on a private verification key under `XDG_CONFIG_HOME`; keep it, the work
root and the service user the same across restarts.
