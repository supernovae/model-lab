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
below are what the phases run. Before a long pretraining run, finish with
[`klab preflight`](pretraining-preflight.md), which checks that the phases'
outputs agree with each other.

## Acquire

```sh
klab --work-dir WORK corpus acquire PROJECT
```

A project (`corpus.yaml`) lists source declarations under `sources/`. A source
is a pinned Git repository, Hugging Face dataset shard, Wikimedia dump, tar or
zip archive, list of PDF files, MediaWiki XML dump, local file, HTTP document or
generator (see [source kinds](#source-kinds)). Acquisition writes one snapshot
per source and then an acquisition lock binding them.

- **Transport budgets.** `corpus budget-init PROJECT` starts a durable ledger
  of bytes, retries, disk and wall time for bounded Hugging Face and Git
  transfers and for archive, PDF and MediaWiki downloads;
  `corpus budget-status PROJECT` reads it. Failed and interrupted transfers keep
  their full charge. A Git source selected by `include` patterns is admitted
  under a budget but not metered by it: Git fetches the repository itself, and
  only the snapshot's own `max_bytes` bounds what is kept. List exact blobs
  (`bounded_blobs`) to have the ledger charge each file.
- **Transport cache.** `--transport-cache DIR` or `KLAB_TRANSPORT_CACHE` keeps
  raw downloads so that a change to an acquisition field that does not alter the
  request does not re-download; see [transport cache](#transport-cache).
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
- **Retry a bounded Hugging Face source.** Rerun the same acquisition after an
  interruption. With `bounded_shards`, completed selections are receipted
  individually. Lab skips those shards after verifying their selected bytes.
  A complete staged input body is reused only after checking its pinned
  SHA-256. Selection then restarts within that shard. Source-step plans list
  these cases under `hf_checkpoint_reuse` and reserve network transfers only
  for the remaining bodies. The plan also budgets the checkpoint metadata.
  Previous transport charges and deadlines still apply. If a checkpoint
  used for admission changes before acquisition, rerun the step for a fresh
  admission. A damaged part can use the transport cache or retry the pinned
  transfer within the existing ledger allowances. A partial source never
  appears as a snapshot or lock entry.

  Checkpoints live under the source's hidden `.hf-parts-<key>` directory
  beside the snapshots. They are bound to the acquisition declaration,
  adapter and ledger scope. Successful acquisition removes its checkpoint.
  Git, PDF and Hugging Face acquisition list checkpoints for other requests so
  you can find retained staging data. Remove abandoned checkpoints only
  when no attempt uses them. The published
  snapshot format is unchanged.
- **Retry a bounded Git source.** Rerun acquisition with the same
  `bounded_blobs` declaration and transport ledger. Lab retains downloaded
  blobs in a private `.git-parts-<key>` directory beside the snapshots. It
  recomputes each staged blob's Git object ID and verifies the declared byte
  count before reuse. A complete blob incurs no new transfer, and an
  incomplete or damaged blob needs a verified cache body or an allowed
  pinned retry. The staged tree becomes the snapshot on success without
  copying the completed blobs. Previous charges and retry limits still apply.
  If a stop interrupted the ledger update after the body was written, Lab
  reconciles the matching transfer from its verified bytes and existing charge.
- **Retry a multi-file PDF source.** Rerun the same acquisition to reuse
  complete pinned files and verified extraction receipts under
  `.pdf-parts-<key>` beside the snapshots. A valid original PDF with missing
  or damaged text needs extraction again. A damaged or partial original needs
  a verified cache body or a permitted retry. Source-step plans report
  `pdf_checkpoint_reuse` as `text` when both artifacts verify and `body` when
  only the original verifies. The existing ledger governs new requests. Cold
  snapshot verification still re-extracts the text to check fidelity.
  Successful publication moves the staged files into the snapshot and removes
  the checkpoint receipts. See [recovery by source adapter](#recovery-by-source-adapter)
  for the replay boundary of each source kind.
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

### Recovery by source adapter

A published immutable snapshot can be verified and reused even if a stop
prevented publication of the project lock. Inside an unfinished snapshot,
the source adapter determines what is reusable. A partially downloaded body
restarts at its first byte. Metered retries preserve existing charges and
need available allowance in the ledger.

| Source | Reusable work inside an unfinished source | Work replayed on retry |
| --- | --- | --- |
| Local files | The local inputs are still available. Acquire reads their current bytes. | Copy and hash the selected files again. |
| Pinned HTTP document | The optional transport cache can serve the verified body and recorded retrieval headers. | Copy and verify the body from cache, or download it. |
| Git with `bounded_blobs` | Complete staged blobs that match the declared sizes and object IDs. A verified cache body can supply a missing blob. | Transfer incomplete or damaged blobs as needed and verify the snapshot. |
| Git with `include` | The optional transport cache can reuse a verified bare repository at the pinned commit. | Recheck the repository and extract the selected files again. Without the transport cache, acquire calls Git fetch again. |
| Hugging Face with `bounded_shards` | Receipted selections and complete staged bodies that match their pins. | Transfer incomplete bodies as needed, then select rows for unfinished or damaged shards. |
| Hugging Face with `include` | The Hub client's download cache can supply files at the pinned revision. | Resolve the file set through the Hub client, validate selected rows and copy the selected files again. |
| Hugging Face retained subset | The verified local parent snapshot. | Replay the bounded selection from the parent. Cold verification also checks the derivative against its parent. |
| PDF documents | Verified originals and receipted extracted text. | Transfer or extract unfinished and damaged documents as needed. Cold verification also re-extracts the published text. |
| Tar, compressed tar or ZIP archive | The optional transport cache can supply the complete pinned body. | Copy or download the body and repeat selection and extraction, including for ZIP. |
| MediaWiki and Wikimedia dumps | The optional transport cache can supply the complete pinned body. | Copy or download the body and replay the XML stream from the start. A Wikimedia cache miss also fetches the checksum manifest. |
| Deterministic or recorded-response generator | Acquisition records the declared generator identity without producing documents. | Recreate the metadata-only snapshot if it was unfinished. Generation follows the build workflow. |

A cache hit avoids body transfer. It still re-hashes the cached input and
performs the local work listed above. A host-local verification receipt can
also avoid repeating a cold check on a published snapshot. See the
[measured replay cases](design/corpus-workarea.md#acquisition-replay-measurements)
for the distinction between transfer, extraction and publication costs.

### What identifies an acquisition

A snapshot's identity (`declaration_sha256` in its manifest, the acquisition
lock and the build identity's `lock`) depends only on what determines the
acquired bytes: `id`, `kind`, `canonical_uri`, `revision` and the whole
`acquisition` block. That block holds Git patterns and pinned blobs, Hugging
Face shards, row qualification and retained subsets, archive, PDF, MediaWiki and
Wikimedia pins and bounds, local file names and generator names. For a Git
source selected by `include` patterns, `rights.nested_metadata_path` is part of
the identity too, because the acquisition retains that file even when no
pattern selects it. For a local source, the file paths are not (only the
logical names), so moving a recipe does not orphan its snapshot.

Everything else is read only by builds and releases and never touches the
snapshot: `license`, `license_url`, `notes`, `rights` (including
`allowed_boundary_paths`, which classifies files after acquisition),
`explicit_training_restriction`, `domains`, `document_kinds`, `source_family`,
`origin` and `markdown_documents`. Editing them re-acquires nothing. They are
bound by the build identity instead, which records every source's full
declaration: a new rights statement is a new build, and release verification
refuses a `sources.json` row, license report or build identity that differs from
the declaration the build ran with.

### Transport cache

Editing an acquisition field that changes the identity (a larger cap, a wider
`include`) makes the snapshot stale, so acquisition downloads again. A transport
cache keeps the raw downloaded bodies, keyed by what was requested rather than by
the declaration, so a re-acquire after such an edit makes no request when the
request itself is unchanged.

It is a host setting, never recipe content: no recipe field names it. Choose the
directory per command or per host, in this order, and without one nothing is
cached:

```sh
klab --work-dir WORK corpus acquire PROJECT --transport-cache /pool/klab-transport
export KLAB_TRANSPORT_CACHE=/pool/klab-transport
```

It may live on a different disk from the work root (bodies are copied into
snapshot staging, which stays in the work root). Pre-fill it in the background,
with no snapshots written:

```sh
klab --work-dir WORK corpus cache-fill PROJECT --transport-cache /pool/klab-transport
klab corpus cache-fill PROJECT --source-id SOURCE_ID
klab corpus cache-list --verify
klab corpus cache-prune
```

`cache-fill` makes the same requests acquisition would and charges the
project's transport ledger the same way; a body already cached is re-hashed, not
fetched. `cache-list --verify` re-hashes every entry. `cache-prune` removes
entries that no longer verify, bodies no entry refers to (only after an hour, so
a concurrent writer is safe) and stale scratch. It never removes a valid entry;
delete cold storage you no longer want yourself. Copying a cache directory
between hosts seeds the other host, because nothing in it is trusted.

**What is stored.** Exactly the bytes acquisition downloads, and nothing it
derives: Hugging Face shard files (`bounded_shards`), archives, PDF files,
MediaWiki and Wikimedia dumps, an HTTP document, each GitHub blob of a
`bounded_blobs` Git source, and a bare Git repository for a Git source selected
by `include` patterns. Layout:

```text
bodies/<aa>/<sha256>      raw body, read-only, named by its own SHA-256
keys/<key-digest>.json    the request (kind, URL, revision, pinned SHA-256,
                          SHA-1 or Git blob id) and the body it resolved to
git/<key-digest>.git      bare repository for a pattern-selected Git source
tmp/                      private staging; bodies are published by atomic rename
```

The key digest covers the request only, so two requests that differ in kind,
URL, revision or pin never share an entry. Not cached, and why: a Hugging Face
source that selects files by `include` pattern (`snapshot_download` resolves the
file list over the network and has no pinned body), `retained_subset` (derived
offline from an existing snapshot), `local` files and generators (no network).

**Never trusted.** Every use re-hashes the body while copying it into snapshot
staging and checks the entry against the request and its pins (the declared
SHA-256, Git blob id or SHA-1). A truncated, altered or mismatched entry is
reported on stderr and downloaded again, replacing it. A Git repository is
reused only when `git fsck` re-hashes all of its objects and the pinned commit
is reachable, and each selected blob is checked against its object id. A body
is stored only after it matched its pin, so bytes from a server that miss the
pin are never cached. Writes are atomic and concurrent acquires and fills may
share one directory.

**Budget and receipts.** A hit reserves nothing and charges the ledger zero
body and zero metadata bytes. It also skips the pinned metadata cross-checks
(Hugging Face `paths-info` and splits, GitHub commit and tree), because the
cached bytes are verified against the pins in the declaration itself. A miss
downloads, charges as before and fills the cache. The acquisition lock records
each body under `transport_cache` for its source: `served_from` (`cache` or
`network`), `cache_key`, `sha256` and `bytes`. It sits outside the snapshot
identity, so the same bytes give the same snapshot whichever served them; an
HTTP document's recorded ETag and dates replay the original retrieval.

**Rights.** The cache is operator-local cold storage of exactly the bytes you
were entitled to fetch. Treat it with the source's license and access terms:
a hit does not recheck that access to a gated or private dataset still holds,
the files are as readable as the directory permissions make them, and it is
never published or committed. Declared rights, licenses and release policy are
applied to snapshots as always, not to the cache.

### Source kinds

Every downloaded body is pinned by SHA-256 and refused if a byte differs, and
every source carries the same [rights](#rights) fields. These three kinds reach
reference collections that have no Git mirror or Hugging Face copy. Each is
immutable, so a verified snapshot is reused rather than fetched again, and each
can run as a `--source-id` step under a transport budget.

**`archive`**: one tar, tar.gz or zip file, such as the IETF RFC collection.
Every regular member matching `include` (and no `exclude`) becomes one snapshot
file, so one document:

```yaml
id: ietf_rfcs
kind: archive
canonical_uri: https://www.rfc-editor.org/in-notes/tar/RFC-all.tar.gz
revision: "2026-11-01"
license: LicenseRef-IETF-Trust-Legal-Provisions
license_url: https://trustee.ietf.org/license-info
explicit_training_restriction: unknown
rights:
  training_eligibility: review_required
  redistribution_mode: metadata_reconstruction_only
  license_references: [https://trustee.ietf.org/license-info]
  notices: [Retain the IETF Trust copyright notice of each RFC]
domains: [standards]
document_kinds: [specification]
source_family: ietf
acquisition:
  expected_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  format: tar.gz            # tar, tar.gz or zip
  include: ["rfc*.txt"]
  exclude: ["rfc-index.txt"]
  max_archive_bytes: 700000000   # the downloaded body
  max_expanded_bytes: 2000000000 # bytes decompressed while scanning
  max_members: 12000
  max_member_bytes: 5000000
  max_bytes: 1500000000          # all selected members together
```

Members are streamed, never extracted by the archive library. Any member with an
absolute path or `..` refuses the whole archive; a selected member that is a
symbolic link, hard link, device, sparse or encrypted file is refused, and two
selected members may not share or nest inside one path. The snapshot records
each member's path, size and SHA-256 and the archive's digest and size.
`admission-draft` writes one decision per member, as for Git files, and
`split-inventory` gives each member the family hint `archive-member:SOURCE:PATH`.

**`pdf_documents`**: a list of pinned PDF files, such as NIST special
publications, NASA handbooks or NTSB and CSB reports. The extracted text of
each PDF is one document. It needs the optional `pdf` extra
(`uv sync --locked --extra pdf`), which installs pypdf:

```yaml
id: nist_sp
kind: pdf_documents
canonical_uri: https://csrc.nist.gov/publications/sp
revision: "2026-11-01"
license: LicenseRef-US-Government-Work
license_url: https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications
explicit_training_restriction: none_found
rights:
  training_eligibility: review_required
  redistribution_mode: metadata_reconstruction_only
  license_references: [https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications]
domains: [security]
document_kinds: [standard]
source_family: nist
acquisition:
  documents:                  # sorted by name
    - name: NIST.SP.800-53r5.txt  # the extracted text's snapshot path
      url: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf
      expected_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
      max_bytes: 12000000
  extraction: {engine: pypdf, version: 7.0.1, max_seconds: 120}
  max_pages: 1000             # per document
  max_text_bytes: 50000000    # all extracted text together
```

Text is extracted page by page in order, with LF newlines, trailing spaces
removed and pages joined by a blank line. Each PDF is extracted in a child
process that is killed after `max_seconds` (default 120), and a PDF that takes
longer is refused. A PDF without a text layer is refused; there is no OCR. The snapshot keeps each PDF's bytes under `pinned/` beside the
text and records both digests, the page count and the exact pypdf version.
Verifying the snapshot extracts the text again from the pinned bytes and
refuses it if the text differs or the installed pypdf is not the declared
version, so the text is always reproducible from the bytes. Admission and
family hints work per file as for archives, with the hint
`pdf-document:SOURCE:NAME`.

**`mediawiki_dump`**: a pinned XML export (`dumpBackup.php` or
`Special:Export`, saved to a file) of any MediaWiki site, such as the Cynefin
wiki. It reuses the Wikimedia page parser; each selected page is one JSONL row
with its title, page and revision IDs and a `site_url?curid=ID` link:

```yaml
id: cynefin_wiki
kind: mediawiki_dump
canonical_uri: https://example.org/dumps/cynefinwiki-20261101.xml.gz
revision: "20261101"
license: CC-BY-SA-4.0
license_url: https://creativecommons.org/licenses/by-sa/4.0/
explicit_training_restriction: none_found
rights:
  training_eligibility: eligible_with_obligations
  redistribution_mode: redistributable_under_source_terms
  spdx_expression: CC-BY-SA-4.0
  license_references: [https://creativecommons.org/licenses/by-sa/4.0/]
  notices: [Attribute page authors and share alike]
domains: [management]
document_kinds: [wiki_article]
source_family: cynefin
acquisition:
  expected_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  compression: gzip           # bz2, gzip or none, matching the file suffix
  dbname: cynefinwiki         # must match the dump's siteinfo
  language: en                # must match the dump's language
  site_url: https://example.org/index.php
  namespaces: [0]
  max_compressed_bytes: 50000000
  max_decompressed_bytes: 500000000
  max_scanned_pages: 100000
  max_selected_pages: 50000
  max_emitted_bytes: 400000000
```

The URL must name a dump file; a live API or query URL is refused. Redirects,
pages outside `namespaces` and pages whose title or text marks them as
non-free or not for training are skipped. For `language: en`, predominantly
non-Latin pages are skipped too. Documents carry the declared `language`; every
other source's documents are labelled `en`. Each page's family hint is
`mediawiki-page:DBNAME:PAGE_ID`. `admission-draft` does not yet cover
`mediawiki_dump` or `wikimedia_dump` rows, so they go through `corpus build`
without a reviewed admission.

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
resumes. A reviewed admission must be bound to that inspection (it needs
`rights.training_use: explicit_eligibility_required`).

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

`freeze` verifies the build (by its receipt if it has one), then publishes it in
place: the build directory is renamed to the release ID, `manifest.json` takes
the place of `build.json`, and no file is copied, so a release costs no extra
disk and no time proportional to its size. The build is consumed: its path keeps
only `build.json` and `consumed.json` naming the release, `freeze BUILD` again
returns that release, and `build` of the same project returns the consumed
build. A build file written while `freeze` was verifying is refused, so a
change after verification is never published. A `freeze` that was stopped part
way leaves `releases/.freezing-BUILD_ID`; run `freeze BUILD` again and it
verifies the staged files from their bytes before publishing them. The release
is not verified a second time: `freeze` records its receipt, and every later
command reuses it ([Verification cost, scratch and
receipts](#verification-cost-scratch-and-receipts)). Progress goes to
`--scratch-dir`.

`release.build.strategy` chooses the builder: `auto` (the default) uses the
disk-backed builder for a language-model-only release without a `fraction`, so
peak memory is bounded by individual records and no reviewed admission is
needed; `memory` holds the corpus in RAM and `disk` fails if the release needs
the in-memory builder. Both builders decide every duplicate group with the same code
(`dedup_policy.py`).

### Resume a failed build

The disk builder reuses verified outputs after a failure or a kill:

| Stage | Output under `WORK/corpora/<id>/builds` | Reused when |
| --- | --- | --- |
| normalize | `prepared/normalized-source-v2/<source>/<digest>`, one shard per source file | the shard's receipt and three output hashes verify |
| index | `stages/index/<key>/documents.sqlite`, `narrow.sqlite` | its receipt verifies; key below |
| dedup | `stages/dedup/<key>/dedup.sqlite`, `groups.jsonl` | its receipt verifies; key below |
| views | `../.views-<key>/build` with receipts beside the staged output | matching build and stage inputs, with a verified append boundary |
| publish | the build directory itself, renamed into place atomically | the build exists and verifies |

The key of a stage is the SHA-256 of exactly the inputs it depends on, recorded
in its `receipt.json` next to the SHA-256 and size of every output file:

- **index**: the receipt digest of every prepared shard in order (which bind the
  source bytes, rights decision, normalizer options and document policy), the
  split declaration (with the digest of its assignment file), the declared
  duplicate keys, every source's rank, and the digests of the implementation
  files `large_build.py`, `pipeline.py`, `split_assignments.py` and
  `dedup_policy.py`.
- **dedup**: the index key, the whole `release.policy.duplicates` section and
  the digests of `large_build.py` and `dedup_policy.py`.

A stage output is trusted only after its receipt verifies: the recorded key is
the digest of the recorded inputs, the directory holds exactly the receipted
files, and every size and hash matches. File names and modification times prove
nothing. An output that fails is deleted and rebuilt, incomplete
(`.incomplete-*`) directories are never inputs, and an output appears by one
atomic rename. Changing only a duplicate option (for example
`duplicates.rights_conflict` after a failure in deduplication) reuses the index
and reruns only deduplication; changing a source, split or normalizer option
changes the shard receipts and so reruns the index.

The checks that fail fast, such as the empty-source check and the admitted
inventory check, run on every start against the verified index, so a fix to the
recipe is judged in seconds, not after the stage that produced the data.

The native disk builder commits ordered span and view boundaries on reaching
25,000 rows or 64 MiB of input. A row can cross the byte threshold.
Each receipt binds the committed output offsets, range
digests, cursor and report counts. On retry the writer verifies the committed
prefix, truncates uncommitted or damaged suffixes and seeks to the last
verified row. It re-reads bytes to verify the prefix and restore the stage
digest. A stop near the end can still repeat report measurement or manifest
hashing. The staged view files become the published build, so checkpointing
adds receipt metadata and keeps one staged payload. Successful publication
removes the view checkpoint. Abandoned checkpoints from other build
identities can be removed between builds once they are no longer needed.
Changing the code revision can require new builder checkpoints, with acquired
snapshots still available for reuse.

**Disk cost.** The index holds every prepared record, so it is about as large
as the prepared shards plus their spans (`documents.sqlite`), with a small
narrow table beside it (`narrow.sqlite`: ID, split, rank, kind, a metadata-only
flag and a rights signature per document, computed once while indexing).
Deduplication reads and writes only that narrow table, never a payload, and
stores its decisions in `dedup.sqlite` (a few bytes per document) next to
`groups.jsonl`; it does not copy the index. Its time therefore does not grow
with document size, and it reports progress per pass (`dedup_groups`,
`dedup_compress`, `dedup_reasons`). The checkpoints are I/O-heavy, so point
`WORK/corpora/<id>/builds/stages` at fast storage (a symlink works). Cleanup rules:

- Publishing a stage output removes incomplete directories and every older
  output of that stage except the newest one, so at most two of each remain
  while a build is failing and being retried.
- A successful build removes all index and dedup outputs. `--keep-stages N`
  keeps the newest N of each, for example to try several duplicate policies
  against one index.
- Prepared shards are kept; they are the normalize checkpoint.

Run one build of a project at a time.

### When a build fails

A failure that stops a build is one of three kinds, and its message says which
by naming where it happened and what would change it:

1. **Fatal recipe error.** The declaration cannot be built whatever the data
   is: an unregistered transform, `release.policy.build.strategy: disk` with a
   source kind the disk builder cannot read, an admission file that no longer
   matches `release.record_admission.sha256`. The message names the field.
   Fix the recipe.
2. **Data-dependent condition, decided by a policy option.** The data meets a
   rule the recipe chose. The message names the source, file, row or
   duplicate group and the option:

   | Condition | Where it is named | Option |
   | --- | --- | --- |
   | an empty or non-text passage | source, file, row or line | `release.policy.documents.empty_text` (`error` or `skip`) |
   | a file without training eligibility | source and file, in `rejected.jsonl` and `license-report.json` | `release.policy.rights.training_use` and the source's `rights` |
   | a source with no document left | the source, an example file and its top reasons | the two options above |
   | text shared across splits | duplicate group (method, key, document IDs, splits) | `release.policy.duplicates.heldout_overlap` |
   | duplicate text with different rights evidence | duplicate group | `release.policy.duplicates.rights_conflict` |
   | validation and test sharing a page or paper, or discordant metadata-only cleaning, in one component | component (a document ID and its members) | no option allows it; `duplicates.keys` and the split assignments decide whether it arises |
   | a split unit in several splits | the unit and its splits | `splits.unit`, `splits.assignments` |

   `klab corpus build --canary` finds these on a sample first.
3. **Internal error.** A prepared shard, stage output or receipt that does not
   verify, or an invariant of the builder. These are never caused by a policy
   option; a stage output that fails its receipt is rebuilt, and an error that
   remains is a bug to report with the build ID.

### Try the build first: the canary

```sh
klab --work-dir WORK corpus build PROJECT --canary
```

runs the whole disk pipeline (rights decisions, normalize, index, deduplication,
views, the per-source ledger, the empty-source check and publishing) with the
full release policy on a deterministic sample of the acquired snapshots, in
minutes. It publishes into `WORK/scratch/canary/<id>/builds`, which each canary
run replaces, and never writes to `WORK/corpora/<id>/builds`. It prints the
sample and the source ledger; a failure stops with the same message the full
build would give.

The sample is, per source, the first 3 rights-eligible files in path order
(until 64 MiB of files are covered; the first file is always taken) and the
first 1000 rows of each JSONL file. Change it with `--canary-files`,
`--canary-rows` and `--canary-bytes`. Every file's rights decision is still
made, so a source whose files are all set aside fails the canary exactly as it
would fail the build.

What a canary can show: recipe and policy failures (a strict rights policy that
sets aside whole sources, an unknown domain or stratum, a source that yields no
documents), data-shape failures in the sampled files and rows
(`documents.empty_text: error` on an empty passage, unreadable rows), and
rights conflicts or held-out overlaps among duplicates that are both in the
sample. What it cannot show: a rare bad row or file past the sample, a
duplicate group whose members are in different, unsampled files, a rights
conflict between files outside the sample, the real size of any stage or the
disk it needs, and anything that depends on the whole corpus (token counts,
mixture proportions, sources that are only empty after deduplication). Its
counts are the sample's; do not read them as the release's. A canary refuses a
release with `release.record_admission` (every reviewed row must be present) and
a release the memory builder would build.

### Release policy

Behaviour is chosen by named options in `release.policy` and `release.normalizer`, never by a version
number. Every option is required, and the resolved policy is recorded in
`build.json`, the release manifest, `report.json` and `license-report.json`.

```yaml
policy:
  duplicates:
    heldout_overlap: error          # or drop_train_copy
    keys: [raw, normalized]         # also canonical_origin
    resolution: chain_smallest_id   # or component_ranked
    rights_conflict: error          # or allow
  documents:
    empty_text: error               # or skip
    exclusions: none                # or research_privacy
    markdown_kind: first_declared   # or prose_when_declared
    row_files: buffered             # or streaming
  rights:
    training_use: explicit_eligibility_required
    # or allowed_unless_explicitly_prohibited
  build:
    strategy: auto                  # or disk, memory
  mixture:
    total_tokens: 12000000000       # the budget the release's mixture weights divide
    max_exposures: 2                # uses of each unique token
    bytes_per_token: 3.0            # optional; the default is shown
```

- **`heldout_overlap`.** `error` fails the build on any text shared across
  splits. `drop_train_copy` keeps the held-out document and drops the train
  copies as `contaminated_heldout` when exactly one held-out split shares the
  text; text shared by two held-out splits still fails.
- **`resolution`.** `chain_smallest_id` links each duplicate group to its
  smallest document id. `component_ranked` resolves connected components of
  shared keys (including `canonical_origin` page and paper identities), keeping
  one document ranked by held-out split, source rank, non-paper kind and id; it
  requires `heldout_overlap: drop_train_copy` and `rights_conflict: allow`, and
  fails on validation/test overlap within a component.
- **`rights_conflict`.** `error` fails the build when identical normalized text
  carries differing rights evidence.
- **`documents`.** `empty_text: skip` records empty or non-text passages in
  `rejected.jsonl`. `exclusions: research_privacy` drops passages with secrets
  or private identifiers, disallowed platform mirrors and unknown page
  provenance. `row_files: streaming` reads JSONL row files row by row.
- **`rights.training_use`.** `explicit_eligibility_required` admits only
  sources with explicit eligibility evidence and allows a reviewed
  `record_admission`. `allowed_unless_explicitly_prohibited` allows
  private-research training unless a prohibition is recorded; every source then
  declares `explicit_training_restriction`.

- **`mixture`.** The release's `mixture` weights are shares of
  `total_tokens`, so a stratum is asked for `weight * total_tokens` tokens and
  can supply at most its unique tokens times `max_exposures`. Freeze refuses a
  build where a stratum's kept TRAIN bytes, divided by `bytes_per_token`, times
  `max_exposures`, is below its request. It names the stratum, the supply and
  the request; fix it by lowering the quota and weight together, giving the
  difference to a larger stratum, raising `max_exposures` or adding sources.
  Quotas are part of the release identity, so this is the cheapest point to
  learn it. `bytes_per_token` should be a lower bound so the check errs toward
  refusing: 3.0 is below English prose and code; a tokenizer audit of a 12B-token run
  measured 2.47 (Python) to 4.41 (web), and a recipe whose strata sit near the
  bound should set it from its own audit. The estimate and verdict are recorded
  in `report.json` as `mixture_supply`. Keep the mixture template's
  `target_quotas` and `max_exposures` equal to the same budget; the exact
  check against real tokens is `describe --mixture`, below.

### Normalizer options

`release.normalizer` declares how documents are cleaned. It is required, never a
version, and Lab recognises no source by name: the recipe lists the sources that
get a pass.

```yaml
normalizer:
  structure_cleaning: true          # false: only NFC and LF line endings
  front_matter_extra_keys:          # source ID -> keys removed with the built-in set
    docs_with_covers: [cover, hero, hero_alt_text, pdf, style]
  gutenberg_boilerplate:            # sources whose Project Gutenberg wrapper is stripped
    - books
```

- **`structure_cleaning`.** `false` applies only NFC normalisation and LF line
  endings, and the other two options must be empty. `true` removes recognised
  YAML front matter (keys `title`, `weight`, `categories`, `category`, `tags`,
  `date`, `draft`, `slug`, `description`; anything else leaves the header in
  place and flags it), records retained and removed line spans with reasons and
  the document structure, and drops metadata-only front matter from
  language-model views while keeping it in the release.
- **`front_matter_extra_keys`.** Additional lowercase YAML keys accepted as
  front matter for the named source only.
- **`gutenberg_boilerplate`.** Removes the Project Gutenberg header through the
  start marker (or a short production credit) and the footer from the end
  marker, for the named sources only.

Every source ID must be declared in the project. The options are recorded in
`build.json`, the release manifest and `report.json`, and their digest names the
normalizer (`normalizer-<16 hex>`) in each document identity, span and lineage
row, so any change builds a different release. Replaces the earlier
`normalizer-structure-v2`/`-v3` names, which hard-coded source IDs: rebuild
releases that used them. `normalizer-structure-v3` for a corpus with a
`project_gutenberg` source and a PagerDuty docs source is `structure_cleaning:
true` with those sources listed in `gutenberg_boilerplate` and the PagerDuty
source given `front_matter_extra_keys: [cover, hero, hero_alt_text, pdf,
style]`.

A record admission manifest has one format (`schema_version: 2`): it lists
every project source, with reviewed rows for Hugging Face sources and a
per-file decision for pinned Git sources. The earlier Hugging Face-only format
is no longer read; redraft such a manifest with `klab corpus admission-draft`.

The reviewed-admission and completed-preparation builds implement only the
strict policy (`heldout_overlap: error`, `keys: [raw, normalized]`,
`resolution: chain_smallest_id`, `rights_conflict: error`) and refuse any other.

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
  The generator ID carries its version (`platform_fault_v2`); a source
  declares `acquisition.generator` only, and the transform `version` must equal
  the ID's suffix.
- `chat_sft`, `tool_episode`: chat and tool-trace views of the records above.

`release.normalizer` is required and chooses the text cleaning by named options
(see Normalizer options below).

A Markdown file becomes one document per heading section. A source declaration
with `markdown_documents: files` keeps each file whole as one document instead,
for long-context training. The choice is part of the source declaration, so it
changes the build's identity; a declaration without it builds as before.

### Rights

Each source declaration records three independent decisions: training
eligibility, redistribution and the license of trained weights. No license name
settles all three.

```yaml
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
- **Release policy.** The release declares a `publication_mode` and
  `policy.rights.training_use` (see Release policy above). With
  `allowed_unless_explicitly_prohibited`, sources record
  `explicit_training_restriction` (a build input: editing it rebuilds but does
  not re-acquire); identical text from several sources keeps one
  copy and every origin's attribution.
- **Conflicts.** With `duplicates.rights_conflict: error`, identical text with
  incompatible rights fails the build.
- **Reports.** `license-report.json` counts eligibility, redistribution modes
  and SPDX expressions. `klab corpus publication RELEASE` emits a metadata
  manifest (source URIs, revisions, file paths and hashes) for sharing what was
  used without the text. `corpus export` is never a publication.
- **Per-source ledger.** `report.json` (`source_ledger`) and `report.md` list
  every declared source with the files seen, the files and records set aside
  with their most common reasons (rights decisions, path boundaries,
  unreadable or empty input), the documents the duplicate policy dropped by
  reason, and the documents retained. A source that retained nothing is named
  at the top of `report.md`.

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
| `klab corpus describe RELEASE [--tokenizer PATH [--mixture TEMPLATE]]` | Record counts by origin, verification and shape; token counts with a tokenizer. `--mixture` also checks the mixture template's `target_quotas` against the measured TRAIN tokens per stratum at its `max_exposures` (and `strata` overrides), failing with `required quota impossible for STRATUM`, as `materialize-mixture` does, for every short stratum. Runs on `--workers` processes and keeps each finished run under `WORK/cache/describe`, so a stopped or repeated describe measures only what is missing |
| `klab corpus sources RELEASE` | Sources and their rights |
| `klab corpus audit RELEASE` | The build audit |
| `klab corpus sample RELEASE --limit N` | Sample records, filterable by domain, kind and status |
| `klab corpus review RELEASE --kind chat` | Records to read before accepting a release; same filters as `sample` |
| `klab corpus lineage RELEASE RECORD_ID` | One record's lineage |
| `klab corpus near-duplicates RELEASE` | MinHash near-duplicate pairs across splits |
| `klab corpus consumers RELEASE --runs-dir RUNS` | Runs that trained on it |

### Verification cost, scratch and receipts

A release is verified from its bytes once. Every command that reads a release
then checks a signed receipt instead: `freeze` records one, and so does any
verification that had none. A verification checks each file's size and SHA-256
against the manifest, then every stage output, document, span, lineage and view
row against the pinned source snapshots. It runs on several processes and keeps
only identities and byte offsets in a disposable index, so memory stays bounded
and the index is a small fraction of the release, not a copy of it. A
language-model-only release with a reviewed admission or a `fraction`, and any
release with chat views, still uses the row-store verifier, whose index can be as
large as the release's documents.

```sh
klab --work-dir WORK corpus audit RELEASE --scratch-dir /fast/disk --workers 16
```

- **`--scratch-dir`** holds the disposable index and the progress journal.
  Default: `WORK/scratch`. Allow 2-3 times the release size for a release that
  uses the row-store verifier, and put it on fast storage (NVMe); it is deleted
  when the command ends.
- **`--workers`** sets the worker processes for every parallel pass over corpus
  files (default: half the CPU count; never more than the CPU count). Use `1` for
  a single-process run. A file smaller than a few hundred MB is read in-process.
  The same option is accepted by `klab tokenizer train`, `klab tokenizer verify`,
  `klab corpus export`, `klab corpus derive-family-inventory`, `klab corpus materialize-mixture` and `klab corpus verify-mixture`. Each parallel stage prints `STAGE: N shards, up to W workers` when it
  starts and `STAGE: S s wall, C s cpu, K cores used, reused R/N` when it ends,
  where cores used is CPU seconds over wall seconds.
- **Worker capacity.** The shard runner also caps large parallel passes by
  CPU affinity and available RAM. Its default planning estimate is 512 MiB
  per worker slot with 1 GiB left for the coordinator and other work. It
  refuses a parallel pass when RAM cannot be measured or cannot fit one
  slot. This estimate governs the worker window, while each operation owns
  the memory it needs to merge results. Spawned workers set common BLAS,
  OpenMP and Rayon thread counts to one. Cached results are read inside the
  same bounded window as newly finished results.
- **Shards.** A parallel stage cuts its file into line-aligned byte ranges
  (shards), runs them on the worker pool and merges results in file order, so the
  output never depends on `--workers`. With `--scratch-dir`, the stage also keeps
  each finished shard under `SCRATCH/shards`, so a killed run reuses them. Every
  stage is either sharded or states why one process is the limit:

  | Stage | Parallelism |
  | --- | --- |
  | `release verification` | shardable: byte ranges of the release files |
  | `tokenizer sampling: TRAIN lineage` | shardable: byte ranges of the lineage view |
  | `family inventory` | shardable: byte ranges of `documents.jsonl`, written in file order |
  | `export split statistics` | shardable: byte ranges of each exported view file |
  | `describe: lineage index` | shardable: byte ranges of `lineage.jsonl`, merged into one sorted index of row positions |
  | `describe: view lines` | shardable: byte ranges of each view file, counting rows |
  | `describe: view rows` | shardable: runs of consecutive view rows, merged by sums and set unions |
  | `describe: source rights` | shardable: byte ranges of `documents.jsonl`, merged in file order |
  | `tokenizer sampling: rank histogram` | shardable: byte ranges of `documents.jsonl`; bytes per stratum and hash bucket are summed |
  | `tokenizer sampling: candidates` | shardable: byte ranges of `documents.jsonl`; each keeps only the documents in the lowest hash buckets of their stratum |
  | `mixture index: documents` | shardable: byte ranges of `documents.jsonl`; kept documents merged in file order |
  | `mixture index: family inventory` | shardable: byte ranges of the family inventory, each row ranked; joined to the documents after the merge |
  | `mixture index: TRAIN lineage` | shardable: byte ranges of `lm/train.lineage.jsonl`; distinct record IDs, joined after the merge |
  | `tokenizer fit` | single process: the Hugging Face BPE trainer is multithreaded natively and selects merges sequentially |

- **Progress** goes to stderr and to `scratch/verify-<release>.progress.jsonl`
  in the builder's progress format: phase, documents and bytes processed,
  documents and MB per second and an estimated time remaining for the phase.
  Phases: `hash_files`, `stage_receipts`, `snapshots`, `spans`, `documents`,
  `raw_replay`, `lineage`, `stage:<id>`, `view:<split>`.
- **Receipts.** The receipt names the release ID, the verifier that accepted
  it and a digest of the stat index (device, inode, mode, size, mtime and ctime)
  of every file in the release and in the snapshots it cites. A command that
  finds a matching receipt makes one `lstat` call per file and reads no bytes:
  tens of thousands of files take seconds, however large the release. `audit`,
  `describe`, `sources`, `sample`, `review`, `lineage`, `publication`,
  `near-duplicates`, `export`, `derive-family-inventory`, `materialize-mixture`,
  `verify-mixture`, `preflight` and the training gate all use it, and each prints
  `release ID: verified by receipt` or `verified from the bytes in N s`. `build`
  and `export` have receipts of their own.
  The receipt is refused, and the release verified from its bytes and a new
  receipt recorded, when any file was written, replaced, added or removed (an
  edit that restores the modification time still changes ctime), when the
  verifier changed, when the receipt does not carry this host's key, or when a
  file has another hard link or is writable by others. The verifier is the code
  in `corpus.release` and what it imports within the corpus package (not the
  command line, the near-duplicate screen, the exporter, the measurement
  or the trainer), so editing those does not unverify a release.
  Receipts trust the filesystem's own metadata and the key under your
  configuration directory: they are not protection against an account that can
  write both. `--cold` (also on `acquire` and `build`) ignores receipts, verifies
  every byte and refreshes the receipt; use it on `audit` after copying or
  restoring a release, or when you want an independent deep check.
  Receipts need a work directory owned by you and not writable by others, and
  files with a single hard link; when they cannot be recorded the command says
  why and verifies every time.

  The completed-mixture, tokenizer, prepared-data and stage-bundle receipts are
  keyed the same way: on the data identities they bind and the source of only
  the modules that decide their result (the mixture, tokenizer, packing and
  staging code and what they import within the corpus and data packages), not
  the whole program. Editing the memory estimator, engines, command line or
  documentation does not make any of them unverified or force an audit replay;
  editing a module they run does. After upgrading Lab across this change, each
  such receipt is re-verified once.

`audit` verifies exactly what `freeze` did and nothing more; it prints the build
audit that was written with the release. Run it for that audit, not to verify.

`near-duplicates` compares every document, including dropped ones, against the
other splits by three-word shingles and reports pairs at Jaccard 0.65 or more,
plus identical text. Training documents are compared with validation and test
documents, and validation with test; training documents are not compared with
each other. It is a lexical screen: it does not catch paraphrases, and MinHash
banding (16 bands of 4 hashes) finds a pair at Jaccard 0.65 about 96% of the
time, at 0.7 about 99%, and above 0.8 almost always.

It indexes the validation and test documents, then streams the training split
past that index. Memory grows with the held-out side, not the corpus: about 8
bytes per held-out shingle plus about 4 KB per held-out document, so about 6 GB
at the default bounds. The training split adds only its longest document.
`documents.jsonl` is cut into line-aligned shards of about `--shard-bytes`
(default 256 MiB) and both reads run on `--workers` processes (default 8, at
most the CPU count). The workers are forked, so they share the held-out index
read-only rather than each building it; a worker's private memory is the
shard it reads, plus the index pages it touches. The shards merge in a fixed
order and the report is the same for any number of workers. Throughput is
about 2 million words per second per core, so the speedup is limited by the
read bandwidth of the disk, by the cores and by the held-out index.
A killed screen resumes: every `--checkpoint-every` training documents (default
200,000) a shard saves its accumulators and its byte offset in
`documents.jsonl` to `SCRATCH/near-duplicates/near-duplicates-<key>-shard-<n>.json`,
and a finished shard saves its result. The key names the release, the
algorithm, the implementation, every option and the shard plan (not
`--workers`), and each file carries its own digest; a checkpoint that does not
match is ignored and that shard starts over. A resumed run indexes the held-out
side again (one parallel read of the file), takes finished shards from their
checkpoints and screens only the rest; its report equals an uninterrupted
run's. The checkpoints are removed when the report is complete.
`--max-heldout-documents` (default 1,000,000) and `--max-heldout-shingles`
(default 200,000,000) bound the index; passing one stops the command with no
report, because every document is screened or none is.

The report's `coverage` lists documents, text bytes and documents without
words (matched by identical text only) for each split and source.
`candidate_count` counts every pair found. `--max-candidates` (default 20,000)
limits only how many are listed, in document-ID order, and
`candidates_truncated` says when the list is shorter than the count.

#### Template-aware screening

Threads, issues and pull requests of well-run projects repeat issue templates,
bot comments and CI notices, so a plain screen lists thousands of pairs that
share a template and nothing else, which hides the few that share content.
`--template-aware` scores each pair a second time over its content, and drops
nothing from the corpus:

```sh
klab corpus near-duplicates RELEASE --template-aware > near-duplicates.json
```

A shingle is boilerplate in a source when at least
`--boilerplate-heldout-documents` (default 20) held-out documents of that
source contain it, or when at least `--boilerplate-sample-documents` (default 4)
of the training documents sampled one in `--boilerplate-sample-stride` (default
256, chosen by document ID) do. The held-out counts are exact. The training
sample is counted in a fixed 256 MB sketch, and the held-out side is read
before the training split is screened, so no extra pass over the corpus is
needed. A pair whose raw Jaccard reaches the threshold is classed:

| Class | Meaning | Needs a decision |
| --- | --- | --- |
| `leak` | content Jaccard (over the non-boilerplate shingles) reaches the threshold, or the text is identical | yes, listed in `candidates` |
| `content_too_short` | no non-boilerplate shingle on either side, so content cannot tell the documents apart | yes, listed in `candidates` |
| `partly_template` | raw Jaccard reaches the threshold, content Jaccard does not, but boilerplate is under `--boilerplate-min-share` (default 0.5) of what the two share | yes, listed in `candidates`: the match is mostly content that falls just under the threshold |
| `template_only` | raw Jaccard reaches the threshold, content Jaccard does not, and boilerplate is at least that share | no: counted, summarized per held-out document in `template_only` |

Each candidate carries `jaccard` (raw), `content_jaccard` and
`boilerplate_share` (the share of its shared shingles that are boilerplate).
Each `template_only` entry gives the held-out document, its number of pairs, the
highest raw Jaccard, the lowest boilerplate share, and its closest partner by
content Jaccard, so a reviewer can spot-check the pairs nearest the line.
`class_counts`, `template_only_count` and `template_only_document_count` count
everything, listed or not, and [`klab preflight`](pretraining-preflight.md)
prints them. Because a `template_only` pair is not a candidate, no decision
record is needed for it; the strict screen (the default) lists every pair.
LSH also proposes pairs from the content shingles, so the same text inside two
different templates is found. This doubles the hashing of documents that
contain boilerplate.

It can miss in one direction that matters: text that many documents repeat is
taken for template, so a held-out document whose content is also repeated by 20
or more held-out documents of its source (or by roughly a thousand training
documents) can be called `template_only` against a training copy. A template
used by only a few documents is not recognised, and its pairs stay candidates.
Raise `--boilerplate-heldout-documents` to be stricter.

Save the report to a file and decide every listed pair in a decision record
that names the report by the SHA-256 of that file, so a record cannot be reused
for another screen:

```yaml
schema_version: 1
release_id: RELEASE_ID
report_sha256: SHA256_OF_THE_REPORT_FILE
accepted_by: NAME
accepted_on: 2026-01-31
decisions:
  - documents: [DOCUMENT_ID, DOCUMENT_ID]
    decision: merged   # merged, dropped, or kept (kept needs a reason:)
```

When a report is dominated by one kind of match (for example thousands of
GitHub threads that share issue templates and bot comments), accept that class
once and list only the exceptions. A `default` decides every listed pair that has
no entry of its own; it can only be `kept` and needs a reason, and the record's
`accepted_by` and `accepted_on` still name who accepted it:

```yaml
default:
  decision: kept
  reason: Matches are templated or boilerplate text, not shared content.
decisions:
  - documents: [DOCUMENT_ID, DOCUMENT_ID]
    decision: dropped   # an exception to the default
```

A record is complete when every listed pair has a decision (its own or the
default), `kept` pairs give a reason, and the report is not truncated. [`klab preflight`](pretraining-preflight.md)
checks all three, and that the screen covered every document of this release.

## Long steps and what a failure costs

| Step | Resumable after a kill? | A failure at the end costs |
| --- | --- | --- |
| `acquire` | yes: verified snapshots, transport cache, bounded HF selections, bounded Git blobs and pinned PDFs with extracted text receipts | incomplete HF shard, Git blob or PDF transfer or extraction, plus final verification. Other adapters replay the unfinished source. |
| `build` normalize | yes: per-file receipted shards | files being written |
| `build` index, dedup | yes: receipted stage outputs ([resume](#resume-a-failed-build)) | the stage that was running |
| `build` views and publish (native disk builder) | yes: verified append boundaries and atomic publication | uncommitted rows, prefix verification and any unfinished report or manifest pass |
| `freeze` | yes: rerun it; a stopped freeze is finished after verifying the staged files | the rename, or a verification from the bytes after a stop |
| `audit`, `describe`, `sources`, release readers | no, but a successful verification is a receipt that later commands reuse | one verification from the bytes |
| `export` | no: staged, then renamed | the export |
| `derive-family-inventory` | yes with `--scratch-dir`: verify and reuse committed segments | unfinished ranges and a sequential assembly pass |
| `near-duplicates` | yes, per shard by training-document offset | the documents since each shard's last checkpoint |
| `materialize-mixture` | yes: committed segments | the segment in progress |
| `verify-mixture` | no | one verification pass |
| `tokenizer` fit and `corpus tokenizer-bakeoff` | no | the fit |
| `klab data prepare` | yes: receipted chunks | the chunk in progress |
| `klab stage ... --through warmup` | no, but short | the pilot |

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
  `content_sha256`) from the reviewed, frozen splits it was built with. For a
  release built with declared splits use `derive-family-inventory` (see
  [Materialize a mixture](#materialize-a-mixture)); `finalize-family-inventory`
  refuses it. It refuses a split
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
  seed: 20260101
  audit_documents_per_stratum: 1000
  max_audit_bytes: 67108864
```

`klab tokenizer train` orders each stratum's training documents by a hash of
`sampling.seed` and the document's content digest, and takes whole documents in
that order until the stratum's byte quota is full (it stops at the first document
that no longer fits). The fit is a uniform random subset of each stratum: it does
not depend on where a document sits in `documents.jsonl`, so no source or region
of the release is favoured, and a smaller quota selects a prefix of a larger
quota's sample. Repeated content counts once within a stratum. The seed and the
order are recorded in the sampling block of the tokenizer manifest and in its
selection identity, so `klab tokenizer verify` binds them; a different seed
or release is a different tokenizer. There is no release-order option: a prefix
of the release is biased toward whatever comes first, so a tokenizer fitted that
way must be fitted again (the manifest of an older one has no `seed` and does not
verify).

The scan runs on `--workers` processes over line-aligned byte ranges in bounded
memory. A histogram pass sums bytes per stratum and hash bucket; a candidate
pass keeps only each stratum's documents in the lowest buckets, about twice its
quota; the coordinator orders them, drops repeated content and fills the quota.
This equals sorting every document and filling the quota whenever the walk stops
inside the candidates, because every document ranked before that point is among
them; when it does not (repeated content made the cut too tight) the stratum's
cut doubles and the candidate pass runs again. The selected documents and the
sampling digest therefore do not depend on the worker count. Pass
`--scratch-dir` so the fit, the replay and the audit share the stored shards.
The audit then takes the documents that follow the fit in the same order,
different from the fit, and checks round-trip failures, unknown tokens and bytes
per token; it refuses to publish on a failure. No validation or test document is
used.

`klab corpus tokenizer-bakeoff DECLARATION --output DIR` instead fits several
vocabulary sizes on one training sample, scores them on held-out families and
selects the smallest within `near_best_ratio` of the best.

A declaration has no schema version. Its fields are the options, all required
and recorded in the report's `identity.declaration`, so a selection states what
produced it: `groups` (the ordered measured groups; a group with both train and
validation documents may not be left out), `eval_split`, `max_fit_bytes`,
`eval_max_docs_per_group`, `near_best_ratio`, and `train_token_scope`
(`grouped_documents` counts train tokens only in the measured groups;
`all_selected_sources` counts every selected normalized source, including kinds
outside the nine groups). The report also lists unmeasured groups and
unclassified source kinds.

## Materialize a mixture

```sh
klab corpus materialize-mixture DECLARATION --output OUTPUT [--workers N]
klab corpus verify-mixture DECLARATION --output OUTPUT [--audit | --reseal]
```

Run `klab corpus describe RELEASE --tokenizer TOKENIZER --mixture TEMPLATE`
first: it needs only the release and the fitted tokenizer, not the family
inventory, and is the gate before near-duplicates, the mixture export and the
family inventory. It counts tokens before cross-document deduplication, so it
can pass what `materialize-mixture` still refuses by a little, never the
reverse by much. A quota failure there means the release must be rebuilt with
corrected `mixture` weights and `policy.mixture`, so check it as soon as a
tokenizer exists.

A mixture declaration sets per-stratum token quotas, `max_exposures` (1 for
unique tokens only, or 2), the tokenizer and optional
`unique_train_content_floors`. Materialization encodes documents into a binary
token stream (`train.tokens.ids`) and a small record file; a run then uses it
through `dataset.source: local_token_mixture`. Nothing is published until every
quota and floor is met; a shortfall is reported per stratum.

**Selection index.** Before encoding, `materialize-mixture` builds an index of
the train documents it may draw from: sorted by stratum and rank, with each
document's byte offset in `documents.jsonl`, as flat memory-mapped columns
(about 200 bytes per document, no database). Three sharded passes build it
(`mixture index: documents`, `mixture index: family inventory` and
`mixture index: TRAIN lineage`, see [Verification cost, scratch and
receipts](#verification-cost-scratch-and-receipts)) and check the family
inventory against the verified release. They take `--workers` and, with
`--scratch-dir`, keep each finished shard, so a killed run resumes; the index
is the same on any worker count. Document IDs and content digests longer than
255 bytes are refused. The finished index is kept under `WORK/cache/mixture-preflight`
and reused by any later mixture over the same documents, family inventory, TRAIN
lineage, `seed`, `order_policy` and `source_strata`, whatever its quotas,
tokenizer or output. The digests of `documents.jsonl` and the TRAIN lineage come
from the release manifest, not from rereading the files; the run refuses if an
input file's inode, size or times change while it runs.

**Family inventory.** The declaration's `family_inventory` is a JSONL file with
one row per kept document (`document_id`, `family_id`, `split`, `stratum`,
`content_sha256`). Materialization checks it against the verified release:
it must cover every kept document exactly once, agree with the release on
split and content, put each source in its mapped stratum (which must be one of
the document's `domains`), and keep every family and every identical content
inside one split. Which command writes it depends on how the release was split.

- **Reviewed splits** (`freeze-splits`): `finalize-family-inventory` exports the
  reviewed families.
- **Declared splits** (`unit: source_document_family` or `source_repository`
  with inline `assignments`, as a project's `splits.yaml` writes them): there is
  no reviewed freeze, so
  `klab corpus derive-family-inventory DECLARATION` reads the declaration's
  `release_path` and `source_strata` and writes the declaration's
  `family_inventory`. A document's family is its `source_family`, the unit the
  release assigned to a split, so isolation is whatever the source assignment
  says; the mixture re-checks that no family spans splits. The command refuses
  a release split by reviewed assignment files, a document whose source is
  unmapped, or one whose `domains` omit its stratum, and never overwrites. It
  streams the verified documents into receipted segments, then assembles
  them in release order. Several declarations over one release can share
  the inventory. Pass `--scratch-dir` and reuse that directory on retry to
  retain completed segments. Use `--cold` to read and verify the release
  from its bytes. Allow roughly the size of the resulting inventory for
  the segments, plus the same amount for the assembly beside the output.
  Without explicit scratch, both use temporary space beside the output.
  Explicit scratch keeps the segments after success. Remove them when no
  inventory attempt is using the scratch directory. Existing published
  inventories keep the same format and need no rebuild. Older disposable
  range checkpoints with a different input fingerprint are ignored.

The family does not choose what a quota takes: documents of a stratum are
ranked individually by a seeded hash and taken in rank order, so a stratum of
several sources is drawn across all of them in proportion to their documents,
never source by source. The family only salts that hash and is the unit of the
isolation check, and the receipt's `eligible_train_families` counts families,
so with declared splits it counts sources. A declared split does not review
repositories or near-copies; screen those separately.

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
  differs from `row_tokens`, and the training storage preflight sizes the
  prepared data from the receipt's packed rows, filler included.

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
  `.OUTPUT.assembly-ID`, where `ID` names the declaration, release manifest,
  family inventory, tokenizer and Lab source the checkpoint was written for;
  keep that directory and the output parent's `cache`. Another declaration,
  release, inventory, tokenizer or Lab revision beside the same output starts
  its own checkpoint, and the command lists the others at once (delete the ones
  you will not resume). A checkpoint whose `request.json` disagrees with its
  name is refused before any verification, sampling or indexing, naming the
  directory to delete. Once the output is published, the segments and commit
  markers are removed; `progress.json` stays.
  `KLAB_MIXTURE_COMMIT_CANDIDATES=N` sets how often it commits (default 4096
  candidates, 64 MiB or 30 seconds).
- **Workers.** `--workers N` (default: half the CPU count) sets the number of
  tokenizer processes. The coordinating process keeps the order-dependent work:
  it walks each stratum's candidates in rank order, drops duplicate content,
  counts quotas and writes the output. The workers read each document's row from
  `documents.jsonl` themselves and tokenize it on the shared
  [encoder pool](token-store.md#one-encoder), with up to three batches per
  worker requested ahead of the document being applied. The output, receipt and
  checkpoints are the same for every worker count, and a resumed run may use a
  different count. Tokens are kept in a [content token
  store](token-store.md#tokens-by-content), so rebuilding a mixture does not
  tokenize unchanged documents again. Allow about 512 MiB per worker; the
  count is lowered to what the host's memory allows.
- **Progress.** `.OUTPUT.assembly-ID/progress.json`, refreshed every few
  seconds, reports per-stratum counts, time spent waiting for the encoder,
  worker utilization and memory.
- **Verify.** `verify-mixture` checks the recorded completion and current
  inputs without rerunning; `--audit` reproduces the whole schedule without the
  token cache. Both commands authenticate the tokenizer's export and byte sample
  through the same signed receipts as the release:
  the export is read from its bytes and the sample scanned once, by the first
  command that needs them, and later commands check file metadata only.
- **Reseal.** When the recorded completion is gone or no longer matches the Lab
  code (`native mixture completion unavailable or changed`), `verify-mixture
  --reseal` records it again without replaying the schedule: it hashes
  `train.tokens.ids` and `train.records.jsonl` (the two files in parallel, large
  reads) against the receipt, authenticates the release and tokenizer through
  their receipts, checks that the receipt's declaration, release, tokenizer,
  inventory, seed, quotas and exposure limit are this declaration's, and checks
  the receipt's counts and sizes agree with each other. It takes about the time
  to read the two files (minutes for 12B tokens on NVMe, dominated by one SHA-256 pass over the token file)
  instead of the replay's hour or more, and its output says the schedule was not
  replayed. It proves the output is byte-identical to what a builder that
  verified its inputs sealed. It does not prove the receipt follows from the
  inputs: only `--audit`, an independent replay, shows that. `--audit` and
  `--reseal` are exclusive.
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
