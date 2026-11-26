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
per source and verifies the candidate acquisition lock before atomically
publishing it. A failed check leaves the previous lock intact. Lab checks
declared retained origins before starting acquisition work and checks the
candidate lock against both the origins and the copies before publication.

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
- **Restart interrupted acquisition.** Lab keeps disposable staging per
  source for work that has no verified checkpoints, including retained
  snapshot copies. Each attempt writes to a unique directory. Before it
  starts, Lab reclaims the previous attempt under the source's staging lock
  after checking the staging area's ownership marker. The lock must coordinate
  all writers. On filesystems that provide only host-local locks, use a
  separate work root per host. If Lab refuses an unsafe or unmarked staging
  area, stop acquisitions for that source, then inspect and move the indicated
  staging directory aside before retrying. An executing
  source step reclaims before measuring disk headroom. Planning, offline
  verification and existing-snapshot finalization leave disposable staging
  alone. Legacy `.acquire-*` and `.reuse-*` directories and checkpoints for
  other requests are retained for inspection. Published snapshots and the
  acquisition lock keep their existing formats.
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
  Acquisition for Git, PDF, Hugging Face, archives and wiki dumps reports
  checkpoints for other requests so you can find retained staging data. Remove
  abandoned checkpoints only when no attempt uses them. The published
  snapshot format is unchanged.
  Use [staging-prune](#reclaim-a-superseded-staging-generation) to preview or
  remove a superseded generation under the source's lock.
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
- **Retry an archive or wiki dump.** Use the same source declaration and
  any existing transport ledger to resume from the private checkpoint beside
  the snapshots. A finished derivation is verified and published from staging.
  If only the body verifies, extraction starts again from that local body.
  Archive and MediaWiki source-step plans report `pinned_checkpoint_reuse` as
  `derivation`, `body`, or null. If the current generation lacks a verified
  body, Lab can reuse a committed body from an earlier generation for the
  same source and adapter. It checks the newest eligible candidate against
  the current byte cap and pins, then moves the verified file under the
  source's adapter lock. Extraction uses the current generation's rules.
  When opening a staging generation or adopting its body, acquisition uses
  the lock to remove older staging directories that contain only the empty
  directory layout. Older generations with bodies, derived output, or other
  entries stay available for inspection.
  A standalone plan verifies bytes without moving them. Within the same
  command, the locked retry and publication reuse admission verification
  when the filesystem fingerprint still matches. Publishing
  removes the downloaded body after the snapshot rename is durable. A retry
  that verifies an already-published snapshot reclaims the matching staging
  directory if the snapshot was moved out of it. Wikimedia commits its
  body after the official checksum and declared pins verify, then rechecks
  SHA-1 and any declared SHA-256 pin together on resume. A complete
  download left before that commit can also be promoted after rechecking
  the pins. An incomplete or damaged body needs a verified cache hit or
  a fresh download.
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
  snapshot (pinned by path and digests) without any network access. Keep both
  snapshots for full build and release verification. Warm receipts track the
  retained parent alongside the subset. Parent changes force verification
  from bytes, and a missing parent blocks verification. Existing receipts
  for retained subsets refresh the first time they are used with this check.
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
| Tar, compressed tar or ZIP archive | A complete staged body or a verified staged derivation. The optional transport cache can supply a missing body. | An unfinished derivation repeats selection and extraction. A finished derivation is verified and published. |
| MediaWiki dump | A complete staged body or a verified staged derivation. The optional transport cache can supply a missing body. | An unfinished derivation replays the XML stream from the start. A finished derivation is verified and published. |
| Wikimedia dump | A verified staged body or derivation. With `multistream_index`, verified range groups, the complete range plan and member outputs also survive a stop. | An indexed retry rechecks the pins and decodes unfinished range groups or extraction members as needed. Other unfinished derivations replay the XML stream from the start. Without a verified staged body or cache entry, acquisition fetches the checksum manifest and dump. |
| Deterministic or recorded-response generator | Acquisition records the declared generator identity without producing documents. | Recreate the metadata-only snapshot if it was unfinished. Generation follows the build workflow. |

A hard kill can leave a transient `.acquire-*` directory containing an
entire staged payload. On the next staging attempt for that source, Lab
reports existing directories and retains their contents because another
acquisition may still be writing there. To reclaim the space, confirm all
acquisitions using that source's work directory have stopped before
removing the reported transient directories.

A cache hit avoids body transfer. It still re-hashes the cached input and
performs the local work listed above. A host-local verification receipt can
also avoid repeating a cold check on a published snapshot. See the
[measured replay cases](design/corpus-workarea.md#acquisition-replay-measurements)
for the distinction between transfer, extraction and publication costs.

### Reclaim a superseded staging generation

Use the generation digest from an `older acquisition staging found` message
or from the suffix of the source's `.ADAPTER-parts-<generation>` directory.
Pass the current project recipe to preview that exact generation.

```sh
klab --work-dir WORK corpus staging-prune PROJECT --source-id SOURCE \
  --generation GENERATION
```

The JSON report includes the path, file count and logical bytes. The preview
reads filesystem metadata. The selected directory can contain reusable work
for an earlier recipe. After deciding it can be discarded, repeat the same
command with `--apply`.

```sh
klab --work-dir WORK corpus staging-prune PROJECT --source-id SOURCE \
  --generation GENERATION --apply
```

Removal holds the source's adapter lock and checks the directory again. Lab
refuses the current recipe's generation, an active adapter writer, and unsafe
directory trees, including symlinks, hard-linked files, and mounts. The
command applies to checkpointed archive, MediaWiki, Wikimedia, Hugging Face,
Git and PDF acquisition. Use the current ledger for a budgeted project. The
generation's logical bytes can differ from the space released by the
filesystem.

### Verify Wikimedia multistream ranges

Use local copies of a declared Wikimedia dump and its bzip2 index, with
the index's known SHA-256 from the same release. For example, for a source
whose index fits these limits, run the following command.

```sh
klab corpus plan-wikimedia-ranges PROJECT --source-id SOURCE --dump DUMP.bz2 \
  --index INDEX.txt.bz2 --index-sha256 INDEX_SHA256 --max-index-bytes 8388608 \
  --max-index-expanded-bytes 67108864 --max-members 10000
```

The planner verifies the dump pins and index SHA-256, then reads the
bzip2 members in order. It checks the XML namespace, site and language,
refuses truncated members or pages that cross a member boundary, and
compares each member's ordered page IDs with the index. It prints JSON
with the input digests, the verified XML context, and each member's
compressed start and exclusive end, expanded size, page count and
ordered page-ID digest. Each digest hashes the decimal IDs in source
order, with a newline after each ID. Header and footer members appear
even when they contain no pages. Index titles are descriptive. Page IDs
bind the index to the content. [Wikimedia multistream format](https://meta.wikimedia.org/wiki/Data_dumps/Dump_format#Multistream_dumps)

Set the declaration's compressed and decompressed byte bounds for the
whole dump. The planner verifies every member, including pages past
the acquisition selection limits. It also caps index lines at 64 KiB
and XML nesting at 64 levels. The command reads local files and prints
a plan.

To use member checkpoints during `corpus acquire`, add `multistream_index`
inside the Wikimedia source's `acquisition` options.

```yaml
multistream_index:
  uri: https://dumps.wikimedia.org/enwiki/20261101/enwiki-20261101-pages-articles-multistream-index1.txt-p1p256.bz2
  expected_sha256: INDEX_SHA256
  max_bytes: 8388608
  max_expanded_bytes: 67108864
  max_members: 10000
  max_member_expanded_bytes: 67108864
```

Use the matching dated index and its known SHA-256. Acquisition verifies
the entire dump at the declared decompressed-byte bound before publishing
a snapshot. The XML header must fit `max_header_bytes` (default 1 MiB).
Each visited member must fit the declared expanded-member bound.

While building the range plan, acquisition commits verification metadata
after each group of up to 64 physical bzip2 members. A retry checks the
dump pins once and passes the verified file fingerprint to the planner.
The planner reads the pinned index, verifies saved receipts, and resumes
the XML scan from verified member boundaries. It may reread the bounded
XML header to restore the parser context. A stop during planning loses at most
one uncommitted group of verification work. A damaged receipt makes its
group eligible for rebuilding. The final plan still covers every member,
including those beyond the selected prefix. The standalone
`corpus plan-wikimedia-ranges` command verifies both input digests itself
and writes no checkpoint files.

Selection runs in source order so the scan, selection, and output limits
apply even inside a member. Acquisition can start extracting the selected
prefix from verified groups while the rest of the range plan is pending.
The selected JSONL is kept in member checkpoints and in the staged
snapshot during assembly. A retry verifies and reuses completed member
outputs, including those committed before the whole plan finished. A
failure anywhere in the range plan prevents snapshot publication.
Acquisition removes its checkpoints after publication.

Use `--wikimedia-checkpoint-members` to commit up to 64 consecutive
extraction members in one receipt. The default is one. Extraction groups
stay within a verified range group and share one staged JSONL file. A
restart checks and copies a finished group's JSONL in a single read. If
verification fails, Lab discards that copy and rebuilds the group. A hard
stop during extraction can require redoing the active group, up to the
configured number of members. Use the same group size on retry to reuse
those checkpoints. Changing the group size can leave the earlier receipts
in staging until publication.

```sh
klab corpus acquire PROJECT --wikimedia-checkpoint-members 8
```

Indexed acquisition can overlap bzip2 decoding with ordered selection.
Set an explicit budget for expanded bytes held ahead of the current
member. The largest member eligible in the current batch sets the per-slot
allowance. Available RAM, CPU affinity and `--workers` can lower
concurrency. The default budget is zero. Near a selection limit, workers
may finish decoding unused members in memory. Only selected rows are
staged, and the output and its identity are independent of worker count.

```sh
klab corpus acquire PROJECT --workers 3 --wikimedia-prefetch-bytes 67108864
```

To try process-based XML parsing, pass `--wikimedia-parse-bytes` along with
the decode budget. When there is room for multiple worker slots, Lab
divides the candidate JSONL budget across them. Each worker decodes and
parses its member. The coordinator cuts the candidates in source order
at the exact remaining scan, selection, and output limits. A worker that
cannot prepare a valid candidate within its bound leaves that member for
ordered serial extraction if needed. Candidate results stay in memory.
The selected member checkpoints are reusable across worker settings.

```sh
klab corpus acquire PROJECT --workers 3 --wikimedia-prefetch-bytes 67108864 \
  --wikimedia-parse-bytes 67108864
```

`--wikimedia-parse-bytes` defaults to zero. The byte budget covers candidate
JSONL payloads across the active window. The host memory estimate also
accounts for parser state, row positions, and process serialization.

A partial indexed acquisition is bound to the adapter implementation. After
an adapter update, the next attempt uses a new staging directory, verifies a
new range plan and extracts the selected prefix again. Lab checks at most
one eligible older committed dump and moves it into the current generation
if its SHA-1 and any declared SHA-256 match. The transport cache or a fresh
download supplies the dump if the candidate fails verification. The pinned
index is fetched if its cache entry is missing or invalid. Lab reports and
retains old `.wikimedia_dump-parts-<key>` directories with derived checkpoints,
unmatched or partial inputs, or other entries. Once all acquisitions for
that source have stopped, inspect those directories before removing abandoned
ones. Published snapshots remain reusable after verification.

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
  release's family assignments fixed. In the ordinary route, the inventory
  streams alongside the verified documents and checks its exact file digest
  during parsing. A reordered suffix uses a compact disk index. The completed
  preparation route uses the authenticated retained inventory member.
- **Output.** The splits YAML points to a digest-named
  `assignments-DIGEST.jsonl` companion; keep the two together. Exact content
  that would cross splits, or a family spanning strata, is refused.
- **Interrupted?** Repeat the identical command with `--recover`. It
  revalidates inputs and every existing member, then publishes only what is
  missing; a conflicting file stops it.
- **`--reuse-completed-preparation`** freezes from an already completed local
  inventory and admission without repeating source verification. The receipt
  records which checks were deferred.

On a synthetic Linux tmpfs workload with 100,000 ordered prior documents
and about 1 KiB of text per document, the ordinary prior join took 1.659 s
after release verification, down from 3.545 s. Logical process reads fell
from 747.47 MB to 138.72 MB and the operation database from 512.11 MB to
36.54 MB. Peak RSS stayed near 61–62 MiB. These are medians of three runs
against #652. The family, stratum and content binding digests matched.
This comparison excludes cold verification, reordered inventories and
production storage.

## Build and freeze

```sh
klab --work-dir WORK corpus plan PROJECT --json
klab --work-dir WORK corpus build PROJECT
klab --work-dir WORK corpus freeze BUILD
```

`klab corpus freeze BUILD --share-stage-artifacts` publishes verified
transform outputs by digest in `WORK/artifacts/corpus/release-stages`.
`klab corpus freeze BUILD --share-document-artifacts` publishes the release's
`documents.jsonl` and `spans.jsonl` by digest in
`WORK/artifacts/corpus/release-documents`.
`klab corpus freeze BUILD --share-lm-views` publishes the LM train,
validation and test text files in `WORK/artifacts/corpus/release-lm-views`.
The options can be combined. They use same-filesystem moves and bind the
physical placement into the release ID. When another release has the same
output digest, freeze authenticates the stored file before consuming the
duplicate staging copy. The receipts track the shared file generation.
When a verified file of at least 64 MiB moves into a digest store, freeze
records its new generation in a host-local signed placement receipt.
Matching receipts allow subsequent freezes in verified-reuse mode to
authenticate the same payload through metadata. Explicit cold verification
checks the bytes. Smaller members use the existing file verification path.
`corpus plan --build BUILD --inspect-verification` reports matching retained
placements as receipt-backed candidates and counts their bytes once per
digest. The plan inspects metadata and the signed records.

Retry interrupted publication with the same placement options. For a
shared document or span authenticated during recovery, staged verification
reuses that hash check while the file generation matches. Semantic replay
still reads the rows it validates. Referenced files are retained release
artifacts. Keep them with the release and its source snapshots when moving
a work area. Use Corpus commands or the verified release reader to resolve
shared members. Completed local preparation uses its retained publication.

To preview LM view placement, run
`klab corpus plan --build BUILD --share-lm-views --json`. LM lineage files
stay with each release. `klab corpus export` resolves shared view paths
for the tokenizer and training configs. When composing a token-mixture
config, use the exported dataset's `validation_path`. Keep the shared view
store with the release manifests and source snapshots when moving the work
area. See the [shared LM view measurements](#shared-lm-view-measurements).

Sharing a released view, document or span file requires exact bytes. Build still
renders its project-scoped files. A change to selection, duplicate policy
or row order can produce different release payloads. Source-level normalized
partitions continue to be reused separately during preparation.

Ordinary native disk builds can skip writing the flat evidence span file:

```sh
klab --work-dir WORK corpus plan PROJECT --reference-normalized-spans --json
klab --work-dir WORK corpus build PROJECT --reference-normalized-spans
klab --work-dir WORK corpus freeze BUILD
```

The option records a distinct build identity. Build authenticates the
ordered normalized partitions and seals their logical `spans.jsonl`
digest. If view emission is interrupted after that seal, a retry can
reuse it while the inputs still match. Freeze inherits the references
and verifies them as build inputs. `--share-document-artifacts` can
still share `documents.jsonl` by digest.

For an existing flat native build with a normalized span inventory, use
`klab --work-dir WORK corpus freeze BUILD --reference-normalized-spans`.
Freeze verifies that the normalized partitions reproduce the exact flat
digest before removing the flat file. Older builds can use inline or
digest-based publication.
Keep the referenced span files and their receipts under
`WORK/cache/corpus/normalized` when moving or retaining referenced builds
and releases.
The [explicit prune commands](#prune-shared-artifacts) account for these
references.

Three synthetic Linux tmpfs trials per size compared the span-emission
phase on fixed-width 1 KiB JSONL records. Reference sealing read the span
payload once. At 512 MiB it wrote about 7.9 KB of seal metadata in place of a
512 MiB flat span file. Restoring the flat checkpoint reread 512 MiB. Reusing
the completed projection read about 15.6 KB of metadata. These measurements
exclude normalization, indexing, other view emission and verification
replay on production storage.

| Span input | Flat emission | Reference seal | Flat checkpoint retry | Reference seal reuse |
| ---: | ---: | ---: | ---: | ---: |
| 16 MiB | 19.5 ms | 15.7 ms | 8.4 ms | 0.16 ms |
| 128 MiB | 192 ms | 116 ms | 61.9 ms | 0.36 ms |
| 512 MiB | 600 ms | 451 ms | 232 ms | 0.91 ms |

In a synthetic Linux tmpfs benchmark, verifying 256 normalized span
partitions (1 GiB total) took a median 0.854 s over three runs and read
1 GiB of payload plus about 81 KiB of process metadata and receipts.
The same flat payload took 0.421 s to hash once. Projection verification
computes the partition and logical release hashes in one read. These
measurements cover byte verification of sparse fixtures, excluding build,
semantic replay and production storage.

In a post-verification Linux tmpfs test with a 268.57 MB stage output,
the first move took 0.35 ms and preserved the inode. First reuse took
269 ms with a cold hash, and the median of three receipt-backed reuses took
72 ms. Two releases sharing those bytes retained one 268.57 MB stage file.
These measurements cover the stage publication step. Build, full release
verification and production storage were outside the benchmark.

Three Linux tmpfs runs per size measured the same handoff for two synthetic
document and span files. For 256 MiB combined, the first move read roughly
100 bytes of process metadata, first reuse read 268.86 MB including the
payload hash, and receipt-backed reuse read 0.22 MB. Each handoff left one
stored copy of each payload. The figures exclude build and semantic replay.

| Combined payload | First move | First reuse | Receipt-backed reuse |
| ---: | ---: | ---: | ---: |
| 4 MiB | 0.40 ms | 6.31 ms | 2.17 ms |
| 64 MiB | 0.37 ms | 34.51 ms | 2.11 ms |
| 256 MiB | 0.36 ms | 125.20 ms | 2.02 ms |

Inspect an existing ordinary build's freeze plan with:

```sh
klab --work-dir WORK corpus plan --build BUILD --share-stage-artifacts --json
klab --work-dir WORK corpus plan --build BUILD --share-document-artifacts --json
klab --work-dir WORK corpus plan --build BUILD --reference-normalized-spans --json
```

The freeze plan reports the release ID, the dependency order and whether
the build is ready to freeze, was interrupted during publication, or already
has a published release. It checks member locations and sizes. For shared
members, it gives the destination store and counts new store bytes, existing
candidate bytes and duplicate staging bytes by their planned actions. The
normalized span plan reports its partition count and logical bytes. Omit
the placement flags to plan inline publication. A retry must match the
interrupted placement.

Publication uses same-filesystem moves with zero payload-copy bytes. The
plan opens metadata only. This plan covers ordinary builds. Completed local
preparation uses its dedicated retained-release command option.

Inspect verification work and current scratch capacity with:

```sh
klab --work-dir WORK corpus plan --build BUILD --inspect-verification --json
klab --work-dir WORK corpus plan --build BUILD --scratch-dir /fast/disk --json
klab --work-dir WORK corpus plan --build BUILD --cold --json
```

Inspection reports each build, staged release or published release check as
`receipt`, `verify_bytes` or `after_placement`. A matching complete receipt
forecasts zero verification scratch. A check without that receipt can reuse
lower-level proofs during verification. At an explicitly selected scratch
location, the plan reports matching narrow row index candidates and their
logical bytes. The span index is only used if the document index is rebuilt.
With `--cold`, the plan describes an independent check from bytes.

The capacity is the filesystem's currently available bytes. It uses the
nearest existing parent of a scratch directory that has yet to be created.
The plan leaves peak scratch unestimated when byte verification is required.
When the selected scratch directory has a recent measurement for the
artifact, inspection includes `prior_scratch_observation` with the attempt's
mode, completion status and sampled peaks. The previous observation and the
current receipt forecast can both appear in the same step. Inspection reads
up to 64 KiB from the progress journal tail.
These observations use metadata and perform no payload authentication or
scratch creation. Inspecting an existing signed receipt checks the source
snapshot and member file generations, so the default placement plan skips
that walk.

Three Linux tmpfs plans per size used synthetic receipts and sparse 4 MiB
source files. The matching-receipt case read about 1.95 KB of metadata at
16,384 files. The placement and missing-receipt plans read about 0.68 KB
at that size. These measurements cover planning with small build manifests.

| Source files | Placement only | Missing receipt | Matching receipt |
| ---: | ---: | ---: | ---: |
| 16 | 0.21 ms | 0.41 ms | 0.62 ms |
| 2,048 | 0.20 ms | 0.26 ms | 13.97 ms |
| 16,384 | 0.20 ms | 0.27 ms | 129.39 ms |

Three synthetic Linux tmpfs plans per size inspected 4 MiB sparse stage
files, with half of the digests already present in the shared store. The
largest plan accounted for 4.29 GB of logical build payloads.

| Stage members | Median wall time | Process reads |
| ---: | ---: | ---: |
| 4 | 0.53 ms | 1.16 KB |
| 128 | 7.31 ms | 28.24 KB |
| 1,024 | 54.96 ms | 225.41 KB |

Run `plan PROJECT` to see the native build ID, acquisition pins and stage
dependencies. If the build already exists, the plan shows whether its
payloads are still in the build directory or have moved into a release.
Otherwise it lists eligible source partitions, candidate normalized cache
hits and source bytes that need normalization. A missing lock reports
`acquisition_required`. An inconsistent inventory reports `blocked` with
a reason. In-memory and reviewed-row-admission builds report `unsupported`.

When every eligible partition has a published receipt, the plan computes
the index key. A matching published index receipt or sealed assembly also
provides the deduplication key. Retained stages report `verify_candidate`
and their payload bytes, which the build will hash before reuse.
`resume_candidate` identifies a retained checkpoint or assembly. If input
identities are still unknown, the plan leaves the dependent key unresolved.
`corpus build PROJECT --keep-stages 2` retains recent index and deduplication
results for subsequent builds. A view-only policy change can preserve both
keys. A duplicate-resolution policy change can preserve the index and
change the deduplication key. With both upstream receipts, the plan can
also resolve the exact view checkpoint key, which binds the build ID and
the current normalized input generations.

A view `resume_candidate` reports committed document rows and checkpoint
records, the candidate prefix bytes to verify, the stage prefix bytes to
replay for its ordered digest, and the observed uncommitted output tail.
The plan checks the receipt filenames for gaps and reads only the latest
receipt and file metadata. Build authenticates the entire chain and can
resume from an earlier verified record if a checkpoint is damaged. A gap
or a short output leaves the byte forecast unresolved. Output byte counts
cover the checkpointed files.

Inspect native build storage with:

```sh
klab --work-dir WORK corpus plan PROJECT --inspect-storage --json
```

The inspection reports current filesystem capacity for the build and
normalized stores, with device IDs for paths on the same filesystem. It
totals published normalized candidate files from metadata. An incomplete
candidate inventory leaves the total logical span size unknown. When all
normalized inputs are present, the plan reports the expected flat span
output. A known view frontier also gives the candidate bytes still to append.
A build that references normalized spans has zero flat span bytes to write.
Build still authenticates its inputs and journal before reusing the prefix.

The last 64 KiB of the same build's progress journal can supply observed
logical output bytes for the sealed index, deduplication and checkpointed
view files, with their computation keys and the attempt's completion status.
Those footprints include reused outputs and exclude temporary scratch and
physical write amplification. Other fresh-stage writes and peak build
scratch remain unestimated.

Three synthetic Linux tmpfs runs per size measured the added inspection
step over sparse normalized files and a populated progress journal. The
4,096-partition case accounted for 17 GiB of candidate files and 512 MiB of
candidate flat span appends. These timings cover storage inspection apart
from the existing build plan.

| Normalized partitions | Added inspection median | Process reads |
| ---: | ---: | ---: |
| 4 | 0.14 ms | 1.39 KB |
| 128 | 1.16 ms | 1.39 KB |
| 4,096 | 38.3 ms | 1.40 KB |

Planning writes no artifacts or verification receipts. It streams the
acquisition lock and snapshot manifests, discards row-level inventories and
reads bounded file rights evidence where needed. Each lock or snapshot
manifest has a 4 MiB retained metadata limit. Large acquisition receipts
still cost a sequential metadata scan. The plan hashes normalized receipts
and reads stage receipts up to 4 MiB. It checks candidate file inventories
and sizes. Build authenticates snapshots and candidate outputs before
reuse. Candidate source byte counts describe the original inputs associated
with normalized cache locations. Stage candidate bytes cover the payloads
named by the stage receipt, with additional reads and write costs resolved
during execution.

Planning scans compact Hugging Face selected-row receipts in bounded batches.
Other valid formatting uses the same strict JSON parser used for the inventory.
Two warm scans per size on a local synthetic workload measured:

| Selected rows | Metadata read | Per-row parser | Batched scan |
| --- | ---: | ---: | ---: |
| 50,000 | 9.89 MB | 0.85–0.87 s | 0.015–0.016 s |
| 250,000 | 49.64 MB | 3.97–4.52 s | 0.077 s |
| 1,250,000 | 248.89 MB | 21.98–22.03 s | 0.39–0.47 s |

Peak process RSS was 22.0 MiB with the per-row parser and 22.2 MiB with
batching. These scans read each metadata file once and use no payload or
scratch I/O. Measure end-to-end planning separately on a production
inventory, where storage and the fraction of rows in compact form matter.

The downstream plan was measured independently on Linux tmpfs with sparse
4 MiB source files, published normalized receipts and synthetic index and
deduplication receipts. Each row is the median of three in-process plans.
The 1,024-partition case accounted for 4.29 GB of source bytes and 5.37 GB
of candidate stage payloads. Its process RSS peaked at 63.2 MiB, including
the Python process and fixture setup. These are planning measurements.

| Partitions | Median wall time | Process reads |
| ---: | ---: | ---: |
| 4 | 2.0 ms | 0.69 MB |
| 128 | 15.2 ms | 0.88 MB |
| 1,024 | 115.0 ms | 2.28 MB |

The view recovery forecast was measured separately on Linux tmpfs with 128
prepared partitions and synthetic 64 MiB checkpoints. The plan checked the
checkpoint directory, the last receipt, and file sizes. Three in-process
measurements per row left the sparse view payloads unread.

| Checkpoints | Candidate prefix | Median wall time | Process reads |
| ---: | ---: | ---: | ---: |
| 16 | 1 GiB | 1.75 ms | 29.6 KB |
| 256 | 16 GiB | 1.74 ms | 29.7 KB |
| 4,096 | 256 GiB | 3.58 ms | 29.7 KB |

Inspect shared normalized storage with:

```sh
klab --work-dir WORK corpus artifacts --json
klab --work-dir WORK corpus artifacts --source-id SOURCE --limit 20
klab --work-dir WORK corpus artifacts --release-references --source-id SOURCE --json
klab --work-dir WORK corpus artifacts --build-references --source-id SOURCE --json
```

The inventory reports published partition keys and the logical bytes of their
files, including receipts. It counts visible staging directories separately.
The default page has up to 100 rows. Use `--after` with the returned
`next_after` cursor to continue in source and key order. Totals describe the
whole selected namespace on each request. Each request scans directory
metadata and retains only one page. Concurrent changes can affect the results.
The command performs no payload reads or deletion. Its byte counts describe
retention, with no determination of which artifacts can be reclaimed.
On a local synthetic store with four sparse files per partition, a 20-row
page took 0.74–0.85 ms with 100 partitions and 42–44 ms with 5,000 partitions.
The larger scan counted 5.35 GB of logical file sizes. Both sizes had 37.9 MiB
peak process RSS.

Add `--release-references` to scan sealed published and staged release
manifests for normalized span references to the partitions on the current
page. Each row reports the number of published and staged releases. JSON
includes an example work-relative path. If a manifest is missing, invalid
or changes during the scan, the inventory reports `scan_status: incomplete`
with unknown reference counts. This scan reads manifests and directory metadata.
A zero release count only describes release manifests. Active builds,
unfinished preparation and future builds can also use a partition.

Use `--build-references` to include unconsumed builds and staged native
builds that have written `build.json` in the same scan. The inventory reports
build and release counts separately for each partition. A consumed build
passes its reference to a matching staged or published release. If the
release is missing or a native view has yet to seal its build manifest,
the combined scan reports incomplete with unknown counts. These counts
describe manifest references. Earlier preparation stages can also use
normalized partitions.

Three synthetic Linux tmpfs trials per size compared the same 20-row page
with and without the reference scan. The largest fixture had 64 span
references per release and 27.88 GB of logical normalized partition files.
The reference scan read only release manifests and process metadata.

| Partitions | Release manifests | Inventory median | With references | Scan process reads | Scan peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 20 | 1.3 ms | 7.4 ms | 0.05 MB | 44 MiB |
| 5,000 | 200 | 46 ms | 108 ms | 1.55 MB | 44 MiB |
| 25,000 | 2,000 | 223 ms | 1.29 s | 30.29 MB | 49 MiB |

Three Linux tmpfs trials per size also compared release-only and combined
inspection for the same 20-row page. Each fixture used equal numbers of
release manifests and build directories. Ten percent of the builds were
sealed native staging roots and ten percent were consumed tombstones.
The largest fixture declared 64 span references per manifest and 27.88 GB
of normalized files. The combined scan read control files and process
metadata.

| Partitions | Release manifests and build roots, each | Release-only median | Combined median | Combined process reads | Combined peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 20 | 6.4 ms | 7.7 ms | 0.10 MB | 44 MiB |
| 5,000 | 200 | 89.9 ms | 130 ms | 2.94 MB | 45 MiB |
| 25,000 | 1,000 | 576 ms | 826 ms | 28.77 MB | 50 MiB |

Inspect shared transform outputs, released document evidence and LM views, or preview
unreferenced payloads, with:

```sh
klab --work-dir WORK corpus artifacts --kind release-stages --json
klab --work-dir WORK corpus artifacts --kind release-stages --unreferenced --json
klab --work-dir WORK corpus artifacts --kind release-documents --unreferenced --json
klab --work-dir WORK corpus artifacts --kind release-lm-views --unreferenced --json
```

This view counts retained bytes once per digest. Each row gives the number
of published and in-progress release manifests in the work area that name
that digest and shows an example. Pagination uses the returned digest as
the `--after` cursor. Totals cover the whole store on each page.

The preview reads manifest and filesystem metadata. It checks the observed
release namespace and manifest generations for changes. When a scan is
incomplete, the JSON reports `scan_status: incomplete`, bounded examples of
problems and null unreferenced totals. The preview page is empty in that
case. Use the inventory as advisory retention evidence.

Three Linux tmpfs scans per size returned up to 20 unreferenced rows from
synthetic stores with ten distinct referenced payloads per release. The
largest scan counted 1.64 GB of logical payload sizes from metadata.

| Shared payloads | Release manifests | Median wall time | Process reads | Median peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 40 | 15 ms | 0.10 MB | 44 MiB |
| 5,000 | 400 | 128 ms | 1.04 MB | 45 MiB |
| 25,000 | 2,000 | 658 ms | 5.18 MB | 54 MiB |

### Prune shared artifacts

Preview pruning an exact digest and apply the same selection with:

```sh
klab --work-dir WORK corpus release-prune --kind release-documents --digest SHA256 --json
klab --work-dir WORK corpus release-prune --kind release-documents --digest SHA256 --apply --json
```

The command also accepts `--kind release-stages` and `--kind release-lm-views`.
Repeat `--digest` to select multiple files. The preview reports protected,
missing and eligible files.
Apply checks the published and staged release manifests in the work area
again while holding the publication lease and the selected digest locks.
Prune refuses the batch when a selected digest is referenced, a required lock
is held, or the release-root scan is incomplete. An already absent digest
can be retried safely. JSON reports each removed digest and logical bytes
removed. A refusal or partial apply exits nonzero.

Exports, prepared bundles and runs that use the digest stores reach the
files through their release manifests. Retain those manifests in the work
area for the lifetime of their consumers. Archives of a release need the
seal, shared payloads and source snapshots together.

Three Linux tmpfs trials per size measured explicit prune batches against
synthetic manifests with ten shared stage references each. The largest
applied selection removed 4 MiB of logical payloads. These timings include
the root scan and publisher locks. Reads cover manifests and metadata.

| Release manifests | Selected digests | Median preview | Median apply | Process reads |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 8 | 14.96 ms | 15.30 ms | 0.10 MB |
| 400 | 32 | 111.43 ms | 111.38 ms | 1.04 MB |
| 2,000 | 64 | 483.26 ms | 482.36 ms | 5.18 MB |

Select a normalized partition using its `artifact_id` from `klab corpus artifacts`:

```sh
klab --work-dir WORK corpus release-prune --kind normalized --partition SOURCE/KEY --json
klab --work-dir WORK corpus release-prune --kind normalized --partition SOURCE/KEY --apply --json
```

Repeat `--partition` to select multiple partitions. Preview reports build,
release and resumable-build references for each selection. A native build
pins its declared source namespaces before normalization, narrows the pin
to exact partition keys once resolved, and clears it after publication.
Published builds and releases protect normalized dependencies recorded in
their manifests. An inline build's optional span catalog is a reuse
candidate. If its partitions have been pruned, start a fresh build with
`--reference-normalized-spans` to create a referenced release.

Apply takes the normalized build lease, the release publication lease and
the selected partition writer locks before checking the roots again. Run
Corpus build and freeze commands from this revision or later while using
normalized prune on the same work area. Finish interrupted older native
checkpoints on their original revision when the scan reports them as
unregistered. Apply requires an owned store and refuses a batch with a
protected selection, an incomplete scan or a held lease. The payload bytes
stay unread. Apply removes the exact published partition directory and its
receipt. If removal is interrupted after the atomic rename, preview shows
`interrupted_prune` and the same selection can finish the removal.

Three synthetic Linux tmpfs trials per size measured normalized prune
with matching counts of release manifests and build roots, plus up to ten
resumable declarations. The 64-partition batch removed 71.34 MB of logical
files. The measurements include root and receipt checks, locking and
retirement. An exact 25,000-partition declaration took a median 21.6 ms to
write and occupied 1.88 MB. These figures cover the metadata operations.

| Release manifests and build roots, each | Selected partitions | Median preview | Median apply | Apply process reads |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 4 | 4.7 ms | 5.4 ms | 0.11 MB |
| 200 | 16 | 74.5 ms | 77.0 ms | 2.98 MB |
| 1,000 | 64 | 608 ms | 620 ms | 28.85 MB |

### Build publication and recovery

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
way leaves `releases/.freezing-BUILD_ID`. Run `freeze BUILD` again to verify
the staging directory and publish it. Once that check completes, a signed
receipt covers the staged files and pinned snapshots. A further retry can
reuse it at the same staging path, and `--cold` verifies the bytes again.
Before the publishing rename, `freeze` refuses changes since the staged
check. It writes the release receipt at the final path for later commands.
See [verification cost, scratch and receipts](#verification-cost-scratch-and-receipts).
Progress goes to `--scratch-dir`.

`release.build.strategy` chooses the builder: `auto` (the default) uses the
disk-backed builder for a language-model-only release without a `fraction`, so
peak memory is bounded by individual records and no reviewed admission is
needed; `memory` holds the corpus in RAM and `disk` fails if the release needs
the in-memory builder. Both builders decide every duplicate group with the same code
(`dedup_policy.py`).

### Resume a failed build

The disk builder reuses verified outputs after a failure or a kill:

| Stage | Output | Reused when |
| --- | --- | --- |
| normalize | `WORK/cache/corpus/normalized/<source>/<key>`, one shard per source file | the typed shard receipt and three output hashes verify |
| index | `WORK/corpora/<id>/builds/stages/.index-<key>.groups` for parts and assembly, then `stages/index/<key>/index.sqlite` under the build root | a part receipt, exact retained-row comparison or completed index receipt verifies |
| dedup | `WORK/corpora/<id>/builds/stages/.dedup-<key>.groups` for incomplete work, then `stages/dedup/<key>/dedup.sqlite` and `groups.jsonl` under the build root | the ordered group prefix verifies or the completed stage receipt verifies |
| views | `../.views-<key>/build` with receipts beside the staged output | matching build and stage inputs, with a verified append boundary |
| publish | the build directory itself, renamed into place atomically | the build exists and verifies |

The key of a stage is the SHA-256 of exactly the inputs it depends on, recorded
in its `receipt.json` next to the SHA-256 and size of every output file:

- **normalize**: the pinned source and file, resolved rights, applicable
  admission decisions, normalizer and document options, row limit and source
  digests for the parser, shard writer and their normalization dependencies.
  The completed-preparation build passes the same document policy into its
  renderer and also binds the execution and admission adapter.

- **index**: the receipt digest of every prepared shard in order (which bind the
  source bytes, rights decision, normalizer options and document policy), the
  split declaration (with the digest of its assignment file), the declared
  duplicate keys, source ranks, reviewed row-level files and the digests of
  `native_index.py`, `duplicate_evidence.py` and `split_assignments.py`.
- **dedup**: the verified index receipt digest, the whole
  `release.policy.duplicates` section and the digests of `native_dedup.py`,
  `dedup_policy.py` and `view_checkpoints.py`. The shared checkpoint
  implementation is part of the dependency because dedup uses it for its
  group and decision journals.

Native LM builds in the same work root share normalized source partitions.
Each project resolves rights and admission from its own declarations before
the lookup. Matching pins and resolved options reuse the same computation
key. The `normalized_source_partition` receipt records that key and the
digests of the document, span and rejection payloads. Document IDs keep their
pinned snapshot and source locations. Appending a source reuses the
unaffected partitions. Replacing a source snapshot can change document IDs
and rebuilds partitions bound to that snapshot.

Each key has one writer and one private staging directory. An interrupted
writer can promote a complete verified partition on retry. The native cache
is outside project-specific `--keep-stages` retention. Canary builds keep
their own scratch. After upgrading, the first native build materializes the
new shared generation from verified snapshots. Completed-preparation
renders retain their authenticated project stage placement.

A stage output is trusted only after its receipt verifies: the recorded key is
the digest of the recorded inputs, the directory holds exactly the receipted
files, and every size and hash matches. File names and modification times prove
nothing. An index or dedup stage with a bad receipt is rebuilt. A damaged
shared normalized partition stops the build. Shards become visible only
after a complete receipt and an atomic rename. In the ordinary disk build, changes confined to view emission
reuse both retained stages. A resolver change or an option such as
`duplicates.rights_conflict` reuses the retained index and reruns deduplication.
A split or index change reuses the normalized shards and reruns both stages.
Changing the normalizer, its implementation or the document policy produces
new shard receipts. A successful build removes these stage outputs by default.
Use `--keep-stages` to retain them for another iteration.

The separated stage implementations change older index and dedup keys once.
Shards with normalization-specific identities can supply that rebuild.

In one NVMe pair, the normalization phase used 200,000 synthetic rows,
425 MB of pinned input and 767 MB of prepared payloads. Fresh runs
took 19.65 s with project-local storage and 19.77 s with the shared
cache. The second project took 19.59 s to produce its own partition
and 1.73 s to verify the shared partition. Both produced identical
document and span digests. The shared hit wrote zero payload bytes
and read 767 MB from storage to verify the prepared files. The
project-local miss read 425 MB of input and wrote 767 MB. Acquisition
and downstream build stages were outside the timed interval.
Ordinary published releases remain readable. The completed-preparation
route still checks its own execution producer scope.

The checks that fail fast, such as the empty-source check and the admitted
inventory check, run on every start against the verified index, so a fix to the
recipe is judged in seconds, not after the stage that produced the data.

The index stores file and byte offsets into the verified prepared shards,
alongside split, duplicate and admission metadata. The builder indexes groups
of prepared files up to about 64 MiB. A larger prepared file is indexed in
paired document and span ranges of about 64 MiB. A document and its span stay
together and can cross the byte threshold. Retry verifies each completed
part and derives the next two byte offsets from its last indexed row.
Parts hold compact row metadata. The assembler builds the document-ID
uniqueness indexes once across the merged tables. A single small group uses
the direct index writer.

The compact merge commits each part and its cursor in a single SQLite
transaction. In receipt-backed mode an authenticated local receipt covers
each unchanged committed database generation. If SQLite rolls back an
unfinished transaction and the receipt no longer matches, Lab compares the
retained prefix with the verified parts. A cold retry always makes that
comparison. A damaged prefix is rebuilt from the parts. The prepared text
and span payloads are outside the comparison. An authenticated receipt can
also retain completed SQL indexes. An interruption inside index creation
can repeat that transaction. A sealed index can be published by rename on retry.
After publication the build removes the part files. A changed input or
producer starts a new index generation.

Spans retain acquisition order and documents and views retain document-ID
order. The emitter reads the original normalized payloads by reference. When
the operating system supports cache probes, a complete view with a working
set below 1 GiB and one eighth of available memory can read cold source files
sequentially before its indexed access. Other reads use batches of up to
8 MiB or 2,048 requested rows with at most 64 open payload files. An individual
row may exceed the batch byte limit.

The native disk builder commits ordered span and view boundaries on reaching
25,000 rows or 64 MiB of input. A row can cross the byte threshold.
Each receipt binds the committed output offsets, prefix SHA-256 digests,
cursor and report counts. On retry the writer verifies the committed
prefix, truncates uncommitted or damaged suffixes and seeks to the last
verified row. The writer extends each prefix checksum during writes and
validates each checkpoint by reading only its new range. The final prefix
checksum is the whole-file SHA-256. The manifest reuses those checksums
after confirming that the file generations still match. The stage and file
inventory use the same checksum for a stage's JSONL file. Resume still reads
the committed prefix and reconstructs the framed stage output identity.
Report measurement uses the existing lineage and view range
checkpoints. The finished census has its own receipt bound to the verified
view frontier and release options. A retry with the same frontier reuses the
finished census. The `writing_manifest` phase hashes ancillary outputs,
such as reports and assignment files. The staged
view files become the published build. The census scratch and receipts are
removed after successful publication. Abandoned checkpoints from other build
identities can be removed between builds once they are no longer needed.
Changing the code revision can require new builder checkpoints, with acquired
snapshots still available for reuse.

**Disk cost.** The index holds compact references and metadata per
record. Completed index groups and the assembled index overlap
briefly. Deduplication reads the index metadata and writes its
decisions into `dedup.sqlite` beside `groups.jsonl`. Its passes
report progress as `dedup_groups`, `dedup_compress` and `dedup_reasons`.
Use fast storage for `WORK/corpora/<id>/builds/stages`. A symlink works.

Native dedup checkpoints the ordered group report and a binary file of
document links. A window covers up to 100,000 origin rows, including unique
origins, and can close after 64 MiB of group and link output. Groups through
4,096 members are resolved in memory. Larger groups stream sorted member
IDs to the audit and use four bytes of ordinal spool per member. SQLite's
sort may also use scratch. A group stays atomic and can cross either window
bound, so a stop inside a large group can repeat that group. Retry verifies
the append prefixes and restores links in bounded batches directly from the
binary journal. It scans the index and calls the duplicate policy only for
the unfinished suffix. The final report is linked into the published stage
on the same filesystem, and the link file and journal are reclaimed after
publication.
The binary link layout is part of the dedup producer identity. After upgrading,
an unfinished build starts a new dedup checkpoint generation from its verified
normalization and index stages.
The chain resolver checkpoints compression and final drop decisions in
windows of 100,000 document numbers. For transitive links it resolves roots
through two file-backed ordinal arrays and records only changed final
representatives. The arrays use eight bytes of scratch per document. Groups
with only direct links need a single detection scan and no ordinal arrays.
The receipts bind each window to the complete group report. A partial
compression retry replays group links and committed representatives,
reconstructs the ordinal arrays when more work remains, and continues from
the saved document number. A completed compression receipt skips that
reconstruction. After the last decision window commits, retry verifies the
report hash and applies the final decisions directly to a fresh database.
The ranked resolver keeps union parents, component winners and flags in
file-backed ordinal arrays. Singletons select themselves. It checkpoints
component reduction and final decisions in 100,000-document windows and
retains only component state changes and nontrivial decisions. A partial
retry still replays group links to reconstruct components. A complete
decision journal skips group replay and reduction. Conflict checks still
run before final decisions, and conflict diagnostics report the total
membership with up to five example IDs. The disposable decisions database
rebuilds after a stop.

- Publishing a stage output removes incomplete directories and every older
  output of that stage except the newest one, so at most two of each remain
  while a build is failing and being retried.
- A same-generation index retry clears abandoned staging directories and
  retains its verified groups until the final index is durable. Abandoned
  group directories from other generations can be removed between builds.
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
with the release. Admission materialization binds the reviewed manifest, source
declarations and its producer. Each source render binds that source's
fingerprint, resolved normalizer and document policy, reviewed decision
digests and renderer. A downstream edit can reuse authenticated input phases
without reading their source bodies again. A changed input generation or
renderer invalidates the affected source receipt. The shared decision-table
adapter participates in both input phases.

The completed row phases reuse authenticated sort, split join, family join,
duplicate-key, deduplication and final-family outputs by their inputs and
producer digests. With an external assignment file, the reference sort
leaves split labels open. A new assignment file then reuses that sort and
reruns the join and its dependent phases. A resolver change can reuse the
duplicate-key sort. A change confined to view emission can reuse all six
row phases. Publication keeps the complete execution producer check and
binds the shared artifacts used by retained mixtures to the signed build.

The narrower keys require a one-time run of phases recorded only under the
older build-wide keys. Completed admission and renders with the scoped input
keys can feed that run.

The completed-preparation builder uses the same view checkpoints as the
native disk builder. The checkpoint records the input shard and byte
position for spans and the ordinal and byte position in the sorted
document references. Retry validates the written prefix and seeks to those
positions. Earlier documents do not need to be read from their source shards
again. A signed local receipt binds the finished staging files to their
generations before rename. Retry can publish those intact files or finish
recording completion after a stop between rename and the completion record.
A replacement directory with the same bytes cannot supply that receipt.
Keep the work root and its verification key available across retries.

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
  used without the text. It takes the acquisition declarations and rights
  report digest from the verified release. A warm publication need not read
  the snapshots' selected-row manifests. `corpus export` is never a publication.
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

`sample`, `review`, `lineage`, `sources`, `audit` and `describe` resolve declared
members of one verified release generation. Sampling streams lineage until it
has the requested matches. For rows that need supplementary parent domains,
it builds a temporary index of document IDs and domains under `--scratch-dir`.
A lineage lookup streams only the tables needed for the requested record and
keeps its matching parents. Late lookups may still scan a whole table.

`describe` stores the completed measurement under `WORK/cache/describe` or
the designated scratch directory. The result binds the verified release,
measurement implementation and tokenizer digest when supplied. A repeat
loads the verified result directly. Mixture quota checks use that result and
read the requested template on each call. Interrupted descriptions retain
their finished shards and rebuild the final result on retry. Existing
description caches from before the reader change are repopulated on use.

On synthetic Linux tmpfs, a tokenized description of 100,000 roughly
1 KiB documents was measured after release verification. The table shows
medians of three processes per case with one worker. Fresh measurements
started with empty measurement caches. Complete report digests matched.

| Measurement cache | Wall time before → after | Logical reads before → after | Peak RSS before → after |
| --- | ---: | ---: | ---: |
| Empty | 8.870 s → 8.952 s | 371.44 MB → 373.11 MB | 136 MiB → 138 MiB |
| Warm | 199.8 ms → 16.8 ms | 26.28 MB → 1.65 MB | 116 MiB → 69 MiB |

The warm run verifies and parses the completed measurement in one read.
Cold release verification and production storage were outside this comparison.

The following local measurements isolate inspection after verification.
The synthetic 100,000-row corpus had 1 KiB of text per document. Each query
ran in a fresh Python process with warm files. Reads use Linux process
`rchar` counters.

| Query | Logical reads before / after | Wall time before / after | Peak RSS before / after |
| --- | --- | --- | --- |
| Sample one early match | 125.77 / 0.13 MB | 587 / <1 ms | 308 / 43 MiB |
| Trace first document | 128.06 / 0.39 MB | 683 / <1 ms | 341 / 43 MiB |
| Trace last document | 128.06 / 128.06 MB | 687 / 428 ms | 342 / 44 MiB |

### Verification cost, scratch and receipts

A release is verified from its bytes once. Every command that reads a release
then checks a signed receipt instead: `freeze` records one, and so does any
verification that had none. A verification checks each file's size and SHA-256
against the manifest. For stage outputs, the ordered replay computes the
inventory checksum alongside the stage identity. It checks stage outputs,
documents, spans, lineage and view rows against the pinned source snapshots.
The verifier can run on several processes. Its disposable index keeps
identities, digests and source offsets in place of document and span payloads.
The verifier for reviewed admission, fractional selection and chat indexes
keys and byte ranges into the verified release members. It holds compact
document metadata for split and selected-view checks. Source replay sorts
range references and reads the cited document and span together. Lineage,
transform and chat checks read the full record where required. The indexed
file generations must still match when verification completes.

On the streaming LM path, receipt-backed verification with an owned
`--scratch-dir` commits the narrow span and document indexes separately.
The span checkpoint binds the pinned source declarations and file rights.
The document checkpoint carries the checked facts for raw replay, including
source file numbers, byte ranges and content and structure digests.
An interrupted document scan can reuse the completed span index. Once the
document index commits, restart can skip both input scans and use the
verified index for raw replay, lineage and view checks. An interruption
inside an unfinished index still repeats that index. The independent
`--cold` pass rebuilds both indexes.

The streaming replay uses one file-backed bitmap for lineage, each transform
and the LM views in turn. Workers validate a record before returning its
document row ID. The coordinator checks duplicates in the bitmap and matches
the number of unique rows against the eligible document count. Bitmap size
is one bit per indexed document, rounded up to a byte.

```sh
klab --work-dir WORK corpus audit RELEASE --scratch-dir /fast/disk --workers 16
```

- **`--scratch-dir`** selects the verification index and progress journal
  location. The default is the host temporary directory. Use fast storage
  for indexes and sorts. Inspect previous scratch observations with
  `corpus plan --build BUILD --inspect-verification`. Disposable indexes
  are removed after verification. A progress journal and streaming LM
  checkpoints can survive a stop. Reviewed, fractional and chat checks
  rebuild their disposable indexes after an interruption.
- **Scratch observations** in the progress journal report logical bytes and
  allocated bytes for named files in the row verification workspace. Samples
  occur at existing progress boundaries and once before cleanup. The final
  record keeps the observed peaks. Retained index files already in this
  workspace count toward its footprint. Separate processes, anonymous SQLite
  sort files and other verification phases can add disk demand.
- **`--workers`** sets the worker processes for every parallel pass over corpus
  files (default: half the CPU count; never more than the CPU count). Use `1` for
  a single-process run. A file smaller than a few hundred MB is read in-process.
  The same option is accepted by `klab tokenizer train`, `klab tokenizer verify`,
  `klab corpus export`, `klab corpus derive-family-inventory`, `klab corpus materialize-mixture` and `klab corpus verify-mixture`. Each parallel stage prints `STAGE: N shards, up to W workers` when it
  starts and `STAGE: S s wall, C s cpu, K cores used` when it ends. Cores used
  is CPU seconds over wall seconds. Stages whose receipts are managed by the
  shard runner also report `reused R/N`.
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
  | `wikimedia member decode` | opt-in threaded decoding of verified compressed members, consumed and checkpointed in source order |
  | `wikimedia member parse` | opt-in process decoding and XML parsing of verified members, with bounded candidates consumed in source order |
  | `release verification` | shardable: byte ranges of the release files |
  | `tokenizer sampling: TRAIN lineage` | shardable: byte ranges of the lineage view |
  | `family inventory` | shardable: byte ranges of `documents.jsonl`, written in file order |
  | `near-duplicate evidence` | shardable: byte ranges of `documents.jsonl`, with ordered compact shingle evidence |
  | `export document licenses` | shardable: byte ranges of `documents.jsonl`, merged as a set |
  | `export split statistics` | shardable: byte ranges of LM and chat splits, with chat bytes counted by the training reader's renderer |
  | `describe: lineage index` | shardable: byte ranges of `lineage.jsonl`, checkpointed as compact arrays and assembled into one sorted index |
  | `describe: view links` | one sequential scan per split link file, with reusable row counts and offsets |
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
  During inventory hashing, `input_bytes_reused` records bytes covered by
  signed file proofs. `input_bytes_processed` records bytes actually hashed
  in that phase.
- **Receipts.** The receipt names the release ID, the verifier that accepted
  it and a digest of the stat index (device, inode, mode, size, mtime and ctime)
  of every file in the release and in the snapshots it cites. A command that
  finds a matching receipt makes one `lstat` call per file and reads no bytes:
  tens of thousands of files take seconds, however large the release. `audit`,
  `describe`, `sources`, `sample`, `review`, `lineage`, `publication`,
  `near-duplicates`, `export`, `derive-family-inventory`, `materialize-mixture`,
  `verify-mixture`, `audit-protected-lineage`, `preflight` and the training
  gate all use it, and each prints `release ID: verified by receipt` or
  `verified from the bytes in N s`. `build` and `export` have receipts of
  their own.
  The complete receipt is refused when any file was written, replaced, added or removed (an
  edit that restores the modification time still changes ctime), when the
  verifier changed, when the receipt does not carry this host's key, or when a
  file has another hard link or is writable by others. The verifier digest
  includes `corpus.release`, its scoped Corpus imports and shared verification
  and rendering helpers. A change to any of those inputs refreshes the
  affected receipts on next use.
  Receipts trust the filesystem's own metadata and the key under your
  configuration directory: they are not protection against an account that can
  write both. `--cold` (also on `acquire` and `build`) ignores receipts, verifies
  every byte and refreshes the receipt; use it on `audit` after copying or
  restoring a release, or when you want an independent deep check.
  Receipts need a work directory owned by you and not writable by others, and
  files with a single hard link; when they cannot be recorded the command says
  why and verifies every time.

  For published source snapshots that verify files by SHA-256, a receipt-backed
  retry reuses completed file checks of at least 64 MiB at the same path and
  file generation. A complete snapshot receipt still requires the snapshot's
  declared inventory and adapter checks. Wikimedia verification uses its own
  replay checks, and an explicit cold check reads the evidence again.

  When a complete proof is missing, receipt-backed build and release checks
  can reuse independently signed SHA-256 checks of files at least 64 MiB.
  Each proof binds the file's path, filesystem identity and expected digest.
  A retry hashes incomplete or changed files, including any file moved to a
  new path. The verifier still runs the remaining source, row and lineage
  checks before it signs a complete proof.
  Stage outputs of at least 64 MiB get a signed checkpoint after the checksum,
  canonical rows and ordered stage identity pass validation. The proof also
  binds the exact stage receipt and inventory entry. A retry at the same path
  can skip both stage scans, which are recorded as reused bytes in progress.
  An independent cold check reads the stage again.

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
(default 256 MiB). The shared runner bounds the first pass's worker
admission by CPU and memory. Training-pass workers fork and share the
held-out index read-only. Both passes use `--workers` and merge shards in
file order, so changing worker count preserves the report.

The first pass saves verified ranges of held-out metadata and binary
shingles, with the sampled training shingles for template analysis.
The training pass commits its accumulators and byte offset every
`--checkpoint-every` training documents (default 200,000) and at each
shard's end. Both use `SCRATCH/near-duplicates`. The key binds the
release, local file generation, implementation, screen options and
shard plan. Retry reads healthy evidence shards, regenerates missing or
damaged ranges, rebuilds the in-memory index and template tables, and
resumes unfinished training shards. Checkpoint data is reclaimed on
success. Interrupted generations remain available at the same key.
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

The recovery inventory starts from Lab `fa567164` and incorporates the later
recovery changes noted below. A committed unit is work a retry can authenticate
and reuse. A completed verification receipt can
avoid repeating a check on unchanged inputs in normal receipt-backed mode.
An independent cold audit reads the evidence again. Timings describe the named
workloads and revisions. The cost of interrupting today's large runs still
needs measurement where the table gives historical or fixture evidence.

### Acquire, build, and publish a release

| Step | Committed unit | Work repeated after a stop | Measured evidence |
| --- | --- | --- | --- |
| Acquisition and transport cache | Published source snapshot, verified transport body or adapter-specific checkpoint. See [recovery by source adapter](#recovery-by-source-adapter). | Adapter-specific work and verification. Indexed Wikimedia retains committed planning groups and selected member groups. A partial pinned transfer restarts its body. | On a 287 MiB Wikimedia shard, a stop after 128 planned members avoided 539.88 MiB of retry decompression. [#592](https://github.com/supernovae/model-lab/pull/592) |
| Snapshot verification | Signed byte checks of eligible files in published snapshots, then a complete snapshot proof. | Reuse completed eligible file hashes at the same path; restart an incomplete file. An explicit cold check reads the bytes again. | Historical Workshop cold pass, about 15 min for 33 snapshots and 108 GB. See [snapshot hash recovery measurements](#snapshot-hash-recovery-measurements). |
| Native disk build, normalize | Receipted shard per source file with normalization-specific producer identity. | Verify completed shards and re-normalize unfinished files or shards with changed normalization inputs. | See [native normalization reuse measurements](#native-normalization-reuse-measurements). Historical Workshop normalize took about 25 min for the 6.7M-document workload. [#491](https://github.com/supernovae/model-lab/pull/491) |
| Native disk build, index and dedup | Verified prepared-file groups and ranges, transactional compact index parts, duplicate-group windows, chain compression and ranked component windows, and final decision journals. Reviewed row admission uses indexed file and row numbers. | Repeat the unfinished index group or range, verify the merged prefix, and append missing parts. Final SQL index creation can repeat. Restore verified binary group links and repeat the unfinished window, which can include an oversized group. Apply a complete decision journal directly. Partial chain compression and ranked reduction can still reconstruct ordinal state. | See [native compact assembly recovery measurements](#native-compact-assembly-recovery-measurements), [chain compression](#native-chain-compression-measurements), [ranked reduction](#native-ranked-reduction-measurements), and [large duplicate groups](#native-large-duplicate-group-measurements). Historical Workshop indexing took about 30 min for roughly 13M rows. [#485](https://github.com/supernovae/model-lab/issues/485), [#450](https://github.com/supernovae/model-lab/issues/450) |
| Native disk build, views and publication | Verified span and view append boundaries at 25,000 input rows or 64 MiB, lineage and view measurement checkpoints, and a receipted census. Final publication by rename. | Verify the committed view prefix, reconstruct its digests and render the uncommitted tail. Reuse finished census work for matching inputs. The manifest hashes ancillary outputs and reuses the view checksums. A stop in the final lineage sort or measurement merge may repeat that step. | A 128 MiB tmpfs append fixture took 0.082 s with two commits and 0.071 s for prefix verification. [Native build census measurements](#native-build-census-measurements) cover the report phase. [#579](https://github.com/supernovae/model-lab/pull/579) |
| Completed-preparation build | Authenticated admission and source renders, reusable row phases, view range checkpoints and a signed receipt before publication. | Reuse authenticated upstream phases after downstream changes. Repeat an unfinished phase. View recovery verifies the committed prefix and resumes from stored span and document cursors. Intact staged views can publish directly, and a renamed build can recover completion from the signed receipt. A partial emission retry can recopy the frozen control inputs. | See [completed-preparation input reuse measurements](#completed-preparation-input-reuse-measurements), [row phase measurements](#completed-preparation-row-phase-measurements) and [view recovery measurements](#completed-preparation-view-recovery-measurements). Production restart cost unmeasured. |
| Freeze | Native staging rename and build tombstone, signed verification after a staged replay, then the public release. Completed preparation seals a retained release against its existing build. | Reuse the complete staged receipt at the same path, or resume an unfinished check from eligible file and replay proofs. Retained freeze can repeat its final verification and seal. | Native freeze took 72.3 s on a 13.2 GB synthetic release, including build verification. The historical 374 GB copy took 92 min before in-place freeze. See [staged freeze recovery measurements](#staged-freeze-recovery-measurements). [#501](https://github.com/supernovae/model-lab/pull/501) |
| Native release verification row indexes | Verified span and document indexes at completion, with explicitly chosen owned scratch in receipt-backed mode. | Reuse the finished document index, or reuse the span index and rebuild the unfinished document pass. Raw replay and downstream tally checks retain their own recovery boundaries. | See [native release row index measurements](#native-release-row-index-measurements). |
| Release verification, `audit`, and `sources` | Signed checks of eligible inventory files, stage outputs and streaming raw source replay, then the complete release proof. `audit` and `sources` read the corresponding published report after verification. | A receipt-backed retry skips eligible completed checks and continues the remaining work. Cold mode starts the checks from bytes. | See [inventory hash recovery measurements](#inventory-hash-recovery-measurements), [stage output verification measurements](#stage-output-verification-measurements) and [raw source replay recovery](#raw-source-replay-recovery). An earlier cold audit took 67.3 s on 13.2 GB synthetic data, and the first receipt-backed audit took 0.94 s. [#500](https://github.com/supernovae/model-lab/pull/500), [#450](https://github.com/supernovae/model-lab/issues/450) |
| `describe` | Cached measurement runs, per-split link offsets and verified lineage index ranges when a proof store or persistent scratch is available. Release verification has its own receipt. | Resume missing index ranges, link files and measurement ranges. An unfinished index sort or cold release check can repeat. Each chat view is one measurement run. | See [describe lineage index recovery measurements](#describe-lineage-index-recovery-measurements) and [describe link planning recovery measurements](#describe-link-planning-recovery-measurements). An earlier full measurement took 55.1 s for 5M synthetic rows (23 GB), 12 workers, without a tokenizer. [#502](https://github.com/supernovae/model-lab/pull/502) |
| Corpus `export` | Document license labels and LM or chat split statistics by range with persistent scratch. Final export directory and verification receipt. | Reuse valid license and split statistics. Repeat unfinished ranges and final publication. Later consumers can use the complete receipt. | See [export verification measurements](#export-verification-measurements), [export license scan measurements](#export-license-scan-measurements) and [chat export split measurements](#chat-export-split-measurements). |

#### Native release row index measurements

On Linux tmpfs, the row verifier reported the following logical and allocated
peaks for named files while building the streaming span and document indexes.
The measured peaks matched the sum of the completed SQLite files. The
synthetic inputs used unique documents with about 2 KiB of text and one
normalized duplicate key each. These figures cover index construction with
one worker. Full verification also runs source replay and view checks.

| Documents | Document and span input | Observed named index scratch |
| ---: | ---: | ---: |
| 10,000 | 25.02 MB | 8.86 MB |
| 60,000 | 150.17 MB | 53.09 MB |
| 180,000 | 450.61 MB | 159.64 MB |

One cold NVMe pair used 300,000 synthetic documents, 875 MB of
document JSONL and 41 MB of span rows. The measured interval covered
index construction or restore through planning raw-source replay.
Release inventory hashing, raw replay, lineage and view checks were
outside the interval. Retry followed SIGKILL after both indexes were
complete. Ordered index rows matched across all cases.

| NVMe run | Wall | Logical reads | Device reads | Device writes | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior fresh | 7.431 s | 5.93 GB | 916 MB | 342 MB | 194 MiB |
| Checkpointed fresh | 8.125 s | 6.22 GB | 916 MB | 342 MB | 201 MiB |
| Prior retry | 7.803 s | 5.93 GB | 916 MB | 342 MB | 194 MiB |
| Checkpointed retry | 0.955 s | 326 MB | 287 MB | 16 KiB | 98 MiB |

The two indexes totaled 342 MB. Sealing added a read of the new
index files. With a same-generation retry, the completed document
index is checked by digest and opened read-only. The verifier keeps
one generation per index for the selected release path, reclaims its
abandoned replay scratch when restarting and clears the indexes after
successful row verification.

With source files on NVMe and scratch on a rotational ZFS pool, one
pair took 14.473 s before and 18.623 s with checkpoints when fresh.
Retry took 7.881 s before and 0.558 s with checkpoints. The retry
used the pool's warm cache, so it measures saved work with that cache
state. Both backends produced the same index rows.

#### Streaming replay coverage measurements

Three synthetic Linux tmpfs runs per size compared the lineage, stage and
view coverage reducers over the same immutable document index. The largest
case read 1.63 GB through the previous SQLite reducer and 45.58 MB through
the bitmap reducer. Peak process RSS was 78.9 MiB and 63.3 MiB respectively.
These figures cover membership reduction, excluding payload replay and
index construction.

| Indexed documents | SQLite time | Bitmap time | SQLite logical scratch | Bitmap logical scratch |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3.0 ms | 1.3 ms | 430 KB | 125 bytes |
| 25,000 | 77.1 ms | 29.0 ms | 9.95 MB | 3.13 KB |
| 180,000 | 715 ms | 234 ms | 71.97 MB | 22.50 KB |

Separate cold audits used local prose sources, one worker and the same
release bytes on both revisions. These are medians of three synthetic Linux
tmpfs runs. Scratch is the sampled peak allocated space for named files,
including the span and document indexes.

| Documents | Prior audit | Bitmap audit | Prior named scratch | Bitmap named scratch |
| ---: | ---: | ---: | ---: | ---: |
| 136 | 103 ms | 103 ms | 303 KB | 205 KB |
| 2,056 | 725 ms | 725 ms | 3.30 MB | 2.35 MB |

#### Streaming duplicate verification measurements

The document scan records each declared duplicate key with its representative.
The `document_duplicates` phase groups that compact evidence in one file-backed
sort. It verifies that every key resolves to one representative. Raw and
normalized hashes use binary encoding and canonical origins keep their exact
text. Each key method has its own namespace. The intermediate evidence is
discarded once checked, before the document index is sealed. A retry can
reuse the checked index and only repeats grouping if that index did not seal.
SQLite can spill its sort to a temporary file. Include that file in storage
provisioning alongside the named scratch files.

Three paired single-process trials per size used NVMe-backed scratch and
warm database pages from the preceding insert. The synthetic inputs had
two digest keys per document and a canonical origin for every second
document. A quarter of the documents were in five-member groups. These times
include inserting the synthetic rows and building the common ID index,
plus duplicate-key validation. They exclude reading real release payloads,
raw replay, and the remaining release checks.

| Documents | Prior time | Grouped check time | Prior sampled peak scratch | Grouped sampled peak scratch |
| ---: | ---: | ---: | ---: | ---: |
| 25,000 | 0.167 s | 0.135 s | 23.83 MB | 14.64 MB |
| 300,000 | 3.209 s | 1.707 s | 286.04 MB | 229.95 MB |
| 1,000,000 | 12.026 s | 6.351 s | 1.08 GB | 775.17 MB |

For one million documents, duplicate validation alone fell from 8.626 s to
1.563 s and from 23.95 GB to 557 MB of logical process reads. The resulting
document index occupied 390 MB, down from 953 MB. Sampled peaks include
SQLite's open temporary sort files. Peak process RSS was 153 MiB before
and 162 MiB after. The logical read counters include cache hits.

A separate full cold audit used the same 40,002-document local Markdown
release on both revisions, with one worker and input-cache eviction
requested before each trial. Across three pairs, median time was 2.681 s
before and 2.625 s after. Storage reads stayed at 253 MB, and storage writes
fell from 45.0 MB to 32.2 MB. Sampled peak allocated space for named scratch
fell from 44.85 MB to 32.06 MB, and peak RSS fell from 134 MiB to 130 MiB.
Both revisions returned the same release ID. Whole-audit performance
on the production release and slower storage remains to be measured.
[#450](https://github.com/supernovae/model-lab/issues/450)

#### Staged freeze recovery measurements

A synthetic NVMe fixture contained 896 local documents, 401 MB of build
artifacts and 131 MB of pinned source files. An initial SIGKILL stopped
freeze with a sealed staging directory. Three trials measured the retry
up to the publishing rename, with cache eviction requested between
attempts. Only the staged receipt was removed for the receipt-backed
control. Earlier eligible file and stage proofs could still be used.

| Staged freeze attempt | Median wall time | Storage reads |
| --- | ---: | ---: |
| Complete staged check required | 2.234 s | 534 MB |
| Completed staged receipt available | 0.234 s | 72 KiB |
| Explicit cold check | 2.606 s | 534 MB |

The final retry published in 0.247 s and preserved the staged manifest
and release ID. Reuse requires a complete signed check at that staging
path. A first interruption before the staged check is recorded can still
require replay. The timings cover this fixture on NVMe.
[#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Snapshot hash recovery measurements

A local source snapshot contained four synthetic files totaling 1.22 GB on
NVMe. Fresh receipt-backed verification took medians of 1.525 s before the
change and 1.527 s after it, with two workers and three fresh processes per
condition. Before each run the fixture requested page-cache eviction.
Both versions read the same 1.22 GB from storage.

Three SIGKILL trials stopped a serial verifier after it signed the largest
file's check. The two-worker retries took a median of 1.128 s. Each reused
537 MB of verification and read 688 MB from storage. An explicit cold
pass read all 1.22 GB. The signed file proof was 1,642 bytes on the fixture.
These results cover snapshot verification. The historical Workshop workload
and replay within a single interrupted file are tracked in [#485](https://github.com/supernovae/model-lab/issues/485).

#### Inventory hash recovery measurements

Six synthetic files totaling 2.38 GB were hashed with two workers on
NVMe. The cold trials requested removal of the fixture from the page cache
before each pass. Three runs gave median times of 1.815 s before the change
and 1.801 s after it, with the same 2.38 GB fetched from storage.
In an interleaved follow-up, fresh receipt-backed runs took 1.70–1.96 s
and fetched the same bytes.

SIGKILL stopped a serial attempt after the two largest file proofs were
durable. The retry with two workers took 0.742 s, reusing 1.54 GB and
hashing 840 MB. After all four eligible proofs existed, the inventory
pass took a median of 41 ms and hashed the remaining 35 MB of small
files. The four signed proofs contained 6,196 bytes. An explicit cold
pass still hashed every file. These times cover inventory hashing.
An unfinished file repeats as a whole, so production restart cost
still depends on the largest in-flight file. [#450](https://github.com/supernovae/model-lab/issues/450)

#### Stage output verification measurements

Stage verification reads line-aligned ranges for canonical JSON validation,
then streams those rows in order to check the stage identity and the file
checksum. Both passes are counted in verification progress. The verifier
checks that the file generation stays stable across the two passes.

A synthetic stage had 600,000 records and occupied 644 MB on NVMe. The table
uses medians of three runs. Cold trials requested eviction of the fixture from
the page cache. These times cover inventory and stage-output verification.

| Cache and workers | Previous time | Combined checksum and identity pass |
| --- | ---: | ---: |
| Warm, four workers | 2.049 s | 1.771 s |
| Cold, four workers | 2.359 s | 1.757 s |
| Cold, one worker | 4.092 s | 3.548 s |

For the four-worker cold trial, process reads fell from 1.96 GB to 1.31 GB.
Storage reads were 644 MB in both cases. The fixture fit in the page cache,
so the measured savings came from avoiding repeated processing of cached
bytes. Production and HDD results remain to be measured. [#450](https://github.com/supernovae/model-lab/issues/450)

On the same 644 MB fixture, a separate SIGKILL trial stopped just after the
stage proof was durable. The first retry in a new process took 214 ms and
skipped both stage scans, accounting for 1.287 GB of reused verification
work. A full pass took a median of 1.757 s. With checkpointing enabled and
no existing proof, the median was 1.777 s. An explicit cold pass after the
stop took a median of 1.750 s and read the stage again. The signed proof
contained 464 bytes. The fixture measures stage and inventory verification.
It leaves interruption inside a stage and end-to-end production recovery
for the long-stage audit. [#485](https://github.com/supernovae/model-lab/issues/485)

#### Frozen document assignment verification

Build and release verification compare frozen assignments with the IDs and
splits that document validation already decoded. The streaming validator
checks the projected document batches as they arrive. The validator used
for reviewed releases checks the same fields in its existing document loop.
The assignment lookup validates its pinned digest, row order and count as
it builds the memory-mapped index. Extra assignments for documents outside
the release are permitted.

On NVMe, an isolated check of 150,000 shuffled document IDs used a 720 MB
document file and a 15.1 MB assignment file. Each time is the median of
three runs. The new path consumes previously decoded document IDs and
splits. The timings exclude the document validation common to both paths.

| Page cache | Previous check | Reused document rows | Storage reads before and after |
| --- | ---: | ---: | ---: |
| Warm | 1.386 s | 0.797 s | 0 MB in both |
| Cold | 1.430 s | 0.800 s | 735 MB to 15.1 MB |

Process reads fell from 751 MB to 15.1 MB in both trials. End-to-end release
time, peak memory and performance on slower storage remain to be measured.
[#450](https://github.com/supernovae/model-lab/issues/450)

#### Raw source replay measurements

During native build and release verification, row replay checks raw files
against the digest pinned by the verified snapshot. The streaming verifier
hashes bytes as it reads the cited records and completes the hash over any
unread tail. Rewinding for overlapping spans does not hash the same prefix
twice. The reviewed verifier uses the pinned digest in its JSONL cursor or
checks the in-memory bytes it already loaded. Snapshot verification and
the source, rights and document checks still run independently.

A synthetic NVMe run replayed 85,714 selected documents from a 120,000-row,
516 MB JSONL source against 423 MB of documents and 44 MB of spans. The
table shows medians of three trials, with cache eviction requested before
each cold trial. The timings cover the source replay phase.

| Page cache | Separate checksum pass | Checksum during replay | Storage reads |
| --- | ---: | ---: | ---: |
| Warm | 1.363 s | 1.362 s | 0 MB in both |
| Cold | 1.652 s | 1.421 s | 1.00 GB in both |

Process reads fell from 1.432 GB to 1.002 GB in both cache conditions.
Peak RSS across individual cold runs ranged from 46 to 52 MiB before and
48 to 49 MiB after. End-to-end release time and HDD behavior remain
unmeasured. An interruption can still repeat an entire raw file.
[#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Raw source replay recovery

In receipt-backed mode, the streaming LM verifier records a signed check
after replaying every document citing an eligible raw file of at least
64 MiB. The check binds the build identity, snapshot, source file,
documents, spans and rights evidence to their verified location and
filesystem generation. A retry at that location resolves the checked
row index and reuses matching source checks. The progress journal counts
the size of each skipped raw file as reused bytes. Reviewed and chat
releases verify source evidence as part of their indexed audit. An explicit
cold audit replays all source files, and a stop within an uncommitted file
repeats that file.

On a synthetic NVMe fixture, one worker replayed 180,000 documents from
two 352 MB JSONL source files with 820 MB of document rows and 93 MB of
spans. In each of three trials, SIGKILL stopped the verifier after
the first source check was signed and committed. The retry replayed only
the second file. All trials requested cache eviction before timing.

| Source replay phase | Median wall time | Storage reads |
| --- | ---: | ---: |
| Fresh, receipt-backed mode | 2.857 s | 1.66 GB |
| Retry after SIGKILL | 1.520 s | 829 MB |
| Explicit cold check with both receipts present | 2.645 s | 1.66 GB |

The first receipt occupied 994 bytes. The isolated receipt-backed runs
included about 0.25 s to compute the verifier identity, which is shared
with other checks in the same operation. Index construction, snapshot
verification and end-to-end release recovery were outside this benchmark.
[#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Raw replay from checked index facts

The streaming verifier checks source attribution, file rights, span
identity and source locators while building the row indexes. Raw replay
uses those facts and streams the pinned source bytes, including bytes
outside selected ranges. It compares the cited byte ranges or row text
and replays the declared structure cleaning where applicable. The
raw-replay phase no longer seeks back into the release's document and
span files. A matching retry receipt still binds their generations.

Three paired one-worker audits used the same 100,002-document synthetic
Markdown release on NVMe. Input-cache eviction was requested before each
full audit. Earlier phases read the files before raw replay, so the
replay-phase read counters mainly measure userspace I/O from cache.
Both revisions returned the same release ID.

| Measure | #673 | Raw replay from checked index facts |
| --- | ---: | ---: |
| Raw replay elapsed | 1.116 s | 0.272 s |
| Raw replay logical reads | 739.5 MB | 476.6 MB |
| Full cold audit elapsed | 5.962 s | 5.836 s |
| Full cold audit logical reads | 2.60 GB | 2.35 GB |
| Full cold audit storage reads | 633.5 MB | 633.5 MB |
| Full cold audit storage writes | 94.33 MB | 95.79 MB |
| Peak allocated space for sampled named scratch | 80.29 MB | 81.75 MB |
| Peak process RSS | 180 MiB | 177 MiB |

The full audit still hashes release members and reads them to construct the
row indexes. Production-sized and slower-storage audits remain open.
[#450](https://github.com/supernovae/model-lab/issues/450)

#### Shared LM view measurements

Two synthetic native builds from different projects each rendered 24.23 MB
of identical LM views across 24,002 documents. Three paired one-worker NVMe
trials compared full cold freeze with inline and shared views. The view
digests matched in every trial.

| Operation or output | Inline views | Shared views |
| --- | ---: | ---: |
| First freeze | 1.574 s | 1.579 s |
| Second freeze | 1.563 s | 1.596 s |
| Second freeze storage reads | 176.12 MB | 200.36 MB |
| LM view bytes retained across both releases | 48.46 MB | 24.23 MB |

The extra second-freeze read authenticated the existing shared files.
Storage writes during both full freezes were about 21.18 MB per release
for either placement. These measurements include verification and exclude
the earlier build and repeated rendering.

An isolated placement benchmark on the same NVMe used three actual JSONL
files totaling 256 MiB, with the staged files already verified and synced.
Median inline placement took 1.32 ms. The first shared placement took
5.76 ms. Authenticating the cold retained files took 294 ms and fetched
256 MiB from storage. Warm placement with existing receipts took 13.0 ms
with no storage reads. Shared placement wrote metadata and moved or
discarded the staged view files. Production-sized freeze and slower
storage remain to be measured. [#567](https://github.com/supernovae/model-lab/issues/567)

#### Shared payload placement receipts

Three paired NVMe trials measured placement of already verified LM views.
The train view held seven eighths of the bytes. Each of the
other files was below the 64 MiB placement-receipt threshold.
Input-cache eviction was requested before the first reuse. Both
implementations retained exactly one file for each digest.

| View bytes | First placement before / after | First reuse before / after | First-reuse storage reads before / after |
| ---: | ---: | ---: | ---: |
| 256 MiB | 4.74 / 7.15 ms | 288.24 / 44.89 ms | 256 / 32 MiB |
| 512 MiB | 4.74 / 7.41 ms | 476.53 / 78.96 ms | 512 / 64 MiB |

First placement wrote a small signed record for the moved train file.
An unchanged generation matched that receipt at first reuse. Both
versions used receipts on later warm reuse and fetched no payload
bytes from storage in those trials.

The full NVMe verification and freeze comparison used two native
72,002-document builds with 72.71 MB of identical LM views. Across
three paired trials, median second freeze read 601.16 MB before and
528.45 MB after. Median time moved from 5.049 s to 5.001 s. First freeze
was near 5.0 s on both revisions. The timing includes byte verification
of the second build. Production-sized and slower-storage freezes remain
to be measured.

#### Source-ordered replay index measurements

The checked document index stores narrow replay facts in source file and
position order. A binary document ID breaks ties. The storage order lets
raw replay scan a file's facts without following a secondary index into
document rows. The `raw_replay_order` phase sorts temporary facts on disk
after duplicate validation has released its evidence. The sorted table
seals with the document index. A stop before that seal can repeat the
document pass and sort. A verified retry reuses the sorted table.

The isolated NVMe comparison generated 16 source files' worth of facts
with source positions permuted across document rows. A third of the
facts included a structure digest. Each size had three paired trials
with identical ordered replay evidence. Cache eviction was requested
before each cold index scan. The warm scan followed immediately.

| Facts | Cold scan before / after | Warm scan before / after | Cold storage reads before / after |
| ---: | ---: | ---: | ---: |
| 100,000 | 0.535 / 0.071 s | 0.135 / 0.069 s | 37.36 / 12.51 MB |
| 1,000,000 | 2.592 / 0.711 s | 1.500 / 0.682 s | 373.81 / 128.27 MB |

At one million facts, logical scan reads dropped from 4.18 GB to 128.26 MB.
Index construction took 3.143 s before and 3.970 s after. The sealed index
shrank from 448.27 MB to 418.95 MB. Sampled peak allocated scratch,
including open SQLite sort files, grew from 464.93 MB to 585.26 MB.
An immediately repeated scan of the 100,000-fact index on rotational ZFS
took 0.199 s before and 0.071 s after. Those ZFS trials measure cached
scans on that filesystem.

Three paired full cold audits on the same 100,002-document NVMe Markdown
release took 5.816 s before and 5.831 s after. Replay alone took
0.270 s before and 0.200 s after, with logical reads falling from
476.56 MB to 93.74 MB. The complete audit's logical reads fell from
2.35 GB to 1.96 GB. Storage reads stayed at 633.5 MB and writes rose
from 95.79 MB to 102.19 MB. Sampled peak allocated space for named
scratch fell from 81.75 MB to 78.00 MB. Peak RSS rose from about 176 MiB
to 190 MiB. Both revisions verified the same release ID. Production-size
and media-cold rotational audits remain open.
[#450](https://github.com/supernovae/model-lab/issues/450)

#### Reviewed release range index measurements

Three paired one-worker trials audited the same two synthetic file-admitted,
frozen-split Markdown releases on NVMe. Cold trials requested input-cache
eviction. Warm trials immediately repeated full verification from bytes.
Each revision verified the same release ID. Scratch columns are sampled
peak allocated bytes for named files.

| Reviewed release | Cold audit before / after | Warm byte audit before / after | Named scratch before / after |
| --- | ---: | ---: | ---: |
| 25,000 documents, 160 MB of release members | 6.713 / 4.592 s | 6.721 / 4.521 s | 378.47 / 25.74 MB |
| 10,000 wider documents, 277 MB of release members | 6.688 / 5.201 s | 6.425 / 5.072 s | 272.19 / 8.18 MB |

For 25,000 documents, logical process reads fell from 2.35 GB to 788 MB
and storage writes fell from 436 MB to 25.8 MB. For wider documents,
logical reads fell from 2.31 GB to 1.16 GB and storage writes fell from
370 MB to 8.26 MB. Cold storage reads stayed near 184 MB and 357 MB
respectively. Median peak RSS across cold trials was about 88 MiB before
and 87 MiB after for the 25,000-document release, and 83 MiB before and
79 MiB after for the wider release.

With the 25,000-document release and snapshots on NVMe and scratch on
rotational ZFS, three paired cold-input trials took a median 7.441 s
before and 4.922 s after. This measures slower scratch with NVMe
inputs. Media-cold rotational input scans remain unmeasured.

The indexed verifier still reads full evidence where the check requires
it, including cited text and generated payloads. Raw replay retains
its existing per-adapter reads. Production-sized and out-of-cache audits
remain to be measured. [#450](https://github.com/supernovae/model-lab/issues/450)

#### Describe link planning recovery measurements

When measurement caching is enabled, `describe` stores the row count
and one offset per 4,096 physical lines for each nonempty view's link file.
The checkpoint binds the file generation, stride and implementation.
An interrupted scan can reuse completed link files. The in-flight file
repeats.

A synthetic NVMe fixture had 3 million LM rows across two splits, with
294 MB of links and 222 MB of payloads. The table shows medians of three
trials with requested cache eviction. Times cover view planning and
exclude release verification, lineage index construction and view
measurement. Every run produced the same plan.

| Planning attempt | Previous time | With link checkpoints | Storage reads before and after |
| --- | ---: | ---: | ---: |
| Fresh | 0.455 s | 0.446 s | 516 MB in both |
| Repeat with prior checkpoints | 0.283 s | 0.002 s | 294 MB to 512 KiB |
| Retry after SIGKILL following the first link checkpoint | | 0.213 s | 273 MB on retry |

Each killed run committed the 243 MB train link file. The retry replayed
the validation link file and the payload counts. Production-scale and
slower-storage results remain open.
[#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Describe lineage index recovery measurements

Persistent measurement caches checkpoint each line-aligned lineage
range as a compact array of hashes, offsets and lengths. A retry checks
the range receipts and scans missing or damaged ranges. The assembly
maps one range at a time into the arrays for the final stable sort.
Ranges are removed after the sorted lookup is durably published.
An interruption during the final sort can still repeat that sort, and the
sort still needs memory proportional to the number of indexed records.

One worker indexed 2.4 million rows (1.11 GB) on a synthetic NVMe
fixture. The table reports medians of three runs with an eviction
request before each attempt. All completed index files matched byte for byte.

| Index build | Wall time | Storage reads | Peak verifier RSS |
| --- | ---: | ---: | ---: |
| Previous fresh build | 2.434 s | 1.11 GB | 170 MiB |
| Fresh with checkpoints | 2.591 s | 1.11 GB | 141 MiB |
| Retry after SIGKILL | 1.452 s | 572 MB | 133 MiB |

Each interrupted run reused eight verified ranges totaling 537 MB of
lineage input. Fresh writes rose from 48 MB to 96 MB while the range
files and sorted index overlapped. A completed-index retry took about
1 ms and read no lineage payload. One three-worker fresh comparison
took 1.172 s before and 1.242 s with checkpoints. These timings
cover index construction. Production-scale and slower-storage results
remain open. [#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Native admission coverage measurements

The disk builder records file numbers and source row ordinals in the
narrow index when a file has reviewed row-level admission. The
`checking_admissions` phase checks exact coverage using those rows once
the index is verified. A file-level review requires no row scan. The
index key binds the ordered reviewed file list and the prepared shard
receipts. The updated producer key causes older in-progress native build
checkpoints to rebuild on first use.

One worker indexed 150,000 synthetic normalized documents (798 MB),
including 100,000 reviewed rows, on NVMe. The table shows median times
over three runs. Cold coverage trials requested eviction of the index
files before checking.

| Phase | Previous time | Narrow admission evidence | Storage reads before and after |
| --- | ---: | ---: | ---: |
| Index construction | 2.957 s | 3.009 s | 814 MB in both |
| Warm coverage | 1.081 s | 0.346 s | 0 MB in both |
| Cold coverage | 1.219 s | 0.404 s | 961 MB to 1.18 MB |

The narrow index grew by 1.16 MB. The separate document index matched
byte for byte. These timings cover indexing and the coverage check.
This fixture used the prior document index layout. Build-wide restart
cost and slower-storage performance remain open.
[#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Native index and payload read measurements

The synthetic NVMe fixture contained 144,000 documents in 24 prepared
shards, totaling 700 MB with their spans. Three trials per condition
compared index construction and the ordered payload read used by view
emission. Cold trials requested eviction of the relevant inputs. Every
condition emitted the same ordered payloads and decisions.

| Operation | Previous time | References and index checkpoints |
| --- | ---: | ---: |
| Fresh index construction | 2.724 s | 2.641 s |
| Index retry after SIGKILL following six committed groups | | 1.578 s |
| Cold ordered span and document reads | 1.669 s | 1.531 s |
| Warm ordered span and document reads | 0.799 s | 0.767 s |

The index shrank from 833 MB to 113 MB. The group files occupied 86 MB
while they overlapped the new index, and fresh index writes fell from
857 MB to 223 MB. Each retry reused groups covering 350 MB of prepared
input. Cold ordered reads fetched 809 MB before and 767 MB after.
The buffered sequential cache assist was eligible for this fixture.
An initial batch-only reference reader took 5.588 s for the same cold
ordered reads.

A separate 1.70 GB fixture with 96,000 rows exceeded the cache budget.
In one paired run, index construction including the final receipt took
4.979 s before and 2.557 s with references. The index shrank from 1.83 GB
to 74 MB. Cold ordered reads took 2.390 s before and 3.275 s with
references. Warm ordered reads took 1.216 s before and 1.149 s after.
The sum of index and cold-read times fell from 7.369 s to 5.832 s.
These were separate timed phases with byte-identical ordered payloads.
Production-scale restart and slower storage remain open.
[#450](https://github.com/supernovae/model-lab/issues/450), [#485](https://github.com/supernovae/model-lab/issues/485)

#### Native build census measurements

The census uses the measured training origins to skip replay when there are
no generated training records. For mixed releases it uses the existing
lineage lookup for held-out rows and scans training lineage in file order.
SQLite holds only unique selected IDs, so there is no second copy of the
lineage payload. The held-out overlap and generated-record rules stay the same.

One-worker synthetic trials on NVMe used cold file pages. The 608 MB native
fixture included 304,000 lineage rows and LM view rows. The mixed fixture
included 92,000 rows and was measured with disposable scratch on both runs.
The measurements cover the census after view emission and exclude final
manifest hashing. Logical reads and device I/O are Linux process
counters. Report digests matched for every run of each fixture.

| Census run | Wall time | Logical reads | Device reads | Device writes | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native, prior implementation | 11.209 s | 6.78 GB | 601 MB | 495 MB | 205 MiB |
| Native, new with checkpointing | 6.975 s | 1.24 GB | 609 MB | 76 MB | 207 MiB |
| Native, retry after SIGKILL following one view range | 5.178 s | 488 MB | 487 MB | 50 MB | 206 MiB |
| Native, retry with all measurement ranges and no final census receipt | 0.482 s | 65 MB | 65 MB | 4 KiB | 197 MiB |
| Mixed, prior implementation | 3.636 s | 1.46 GB | 173 MB | 149 MB | 113 MiB |
| Mixed, new with disposable scratch | 2.755 s | 782 MB | 184 MB | 8.8 MB | 115 MiB |

The native SIGKILL retry reused one of seven view ranges, the assembled
lineage lookup, and every view link and line-count result. The completed
measurement cache held about 70 MB before publication cleanup.
A connected build fixture covers reuse of the finished census after a
publication stop. Larger mixed releases, production NVMe and slower storage
remain unmeasured.

#### Native build manifest measurements

An NVMe fixture rendered 70,000 documents and spans into 1.49 GB of
inventory files. The renderer ran without the census so the measurements
cover view emission, recovery and manifest creation. Three runs per
implementation used the same prepared input. Each ran through manifest
creation, stopped with SIGKILL before publication, and retried with cold
staged file pages. Every build manifest matched byte for byte. The append
journal held about 17 kB in both implementations.

| Run | Prior median manifest time | Inline checksum median | Prior total logical reads | Inline checksum total logical reads |
| --- | ---: | ---: | ---: | ---: |
| Fresh emission and manifest | 653 ms | 3.3 ms | 2.41 GB | 468 MB |
| Retry through publication | 646 ms | 3.8 ms | 3.89 GB | 1.95 GB |

Device reads were about 465 MB for fresh emission and 1.49 GB for a retry
with either implementation. The older extra reads came from the page cache
on this fixture. Whole-phase wall times varied with host load. Larger
working sets and slower storage remain unmeasured.

#### Completed-preparation input reuse measurements

A synthetic NVMe fixture used one reviewed source with 150,000 rows,
147 MB of source data and a 20 MB admission manifest. The first run
materialized admission and rendered the source in 16.273 s, reading
168 MB and writing 465 MB at the device. A second checkout changed
the downstream build module's digest. It reused both authenticated
input phases in 0.029 s and reported zero source bytes rendered and
zero device bytes read or written in those phases. The stage identity,
decisions and all prepared output digests matched.

Each elapsed measurement is a single run. The timing excludes acquisition,
split work, final build processing and publication. Production-scale and
slower-storage evidence is still open.

#### Completed-preparation row phase measurements

One cold NVMe run per case used 150,000 distinct documents and 401 MB of
prepared input, with an external assignment file and frozen family records.
Timing includes the six row phases and signed links into the build journal.
Admission, source rendering, view emission and final publication are outside
the measured phase. The changed digests simulate edits to the view producer
or duplicate resolver.

| Run | Elapsed | Device reads | Device writes |
| --- | ---: | ---: | ---: |
| Fresh with build-wide keys | 8.179 s | 458 MB | 421 MB |
| Fresh with scoped row phases | 8.125 s | 458 MB | 421 MB |
| Changed view producer with build-wide keys | 9.629 s | 458 MB | 421 MB |
| Changed view producer with scoped row phases | 0.528 s | 0 MB | 139 kB |
| Changed resolver with scoped row phases | 2.384 s | 128 MB | 44 MB |

The view edit reused all six phases. The resolver edit reused four and
rebuilt the decisions and final family inventory. Reference, duplicate-key,
decision and family inventory digests matched. This fixture had no duplicate
groups. Dense duplicate workloads, production memory and slower storage
remain unmeasured for this reuse comparison.

#### Completed-preparation view recovery measurements

An 80,000-document NVMe fixture emitted 657 MB of inventory files. Three
fresh and three interrupted runs per implementation measured the renderer
through publication from prebuilt references. The injected SIGKILL followed
25,000 rendered documents. It followed a committed boundary with the new
writer. All manifest bytes matched with the fixture build identity held fixed.
These timings exclude source rendering, sorting, deduplication, and copying
reviewed control inputs.

| Run | Median wall time | Documents rendered | Logical reads | Device reads | Device writes | Median peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior, fresh | 4.505 s | 80,000 | 902 MB | 248 MB | 658 MB | 78 MiB |
| Prefix checkpoints, fresh | 4.582 s | 80,000 | 245 MB | 248 MB | 658 MB | 78 MiB |
| Prior, retry | 4.440 s | 80,000 | 902 MB | 248 MB | 658 MB | 78 MiB |
| Prefix checkpoints, retry | 3.371 s | 55,000 | 426 MB | 376 MB | 448 MB | 76 MiB |

The retry reads the written prefix from storage and writes 209 MB less.
An earlier checkpoint prototype hashed each byte twice and took 4.940 s
fresh. The retained checkpoint checksum now doubles as the manifest checksum
at the final boundary. On a separate stop immediately after rename,
the retry recorded completion in 76 ms, read 8 KiB from storage, and rendered
zero documents. Production and slower-storage timings remain open.

#### Native dedup recovery measurements

Three single-process cold trials per case on NVMe used 400,000 documents
and a 221 MB index per fixture. Both fixtures used the chain resolver and
two duplicate methods, with 8,000 groups in the sparse fixture and 320,000
in the dense fixture. An early SIGKILL followed two committed windows.
The late dense SIGKILL followed all nine windows and preceded
representative compression. The timings include opening the decisions
database and running dedup. They exclude input-index and final stage receipt
verification. All group report and ordered decision digests matched within
each fixture.

| Run | Median wall time | Logical reads | Device reads | Device writes |
| --- | ---: | ---: | ---: | ---: |
| Sparse, prior full pass | 0.846 s | 419 MB | 153 MB | 119 MB |
| Sparse, fresh with checkpoints | 0.929 s | 471 MB | 153 MB | 119 MB |
| Sparse, early retry | 0.836 s | 439 MB | 137 MB | 119 MB |
| Dense, prior full pass | 4.189 s | 579 MB | 153 MB | 227 MB |
| Dense, fresh with checkpoints | 4.512 s | 634 MB | 153 MB | 232 MB |
| Dense, early retry | 4.051 s | 658 MB | 165 MB | 204 MB |
| Dense, late retry | 2.518 s | 727 MB | 199 MB | 116 MB |

The dense link journal used 5.12 MB beyond the report that becomes the final
audit input. Median peak RSS in the dense runs stayed near 196 MiB. The
logical read overhead includes bounded index look-ahead and prefix
verification. The late retry still replays links and recomputes final
decisions. Production-scale, slower-storage and ranked resolver performance
remain unmeasured.

#### Native chain decision recovery measurements

Three cold trials per case reused the 400,000-document NVMe fixtures above.
In the dense fixture, one stop followed two decision windows covering
200,000 documents, and another followed the last decision receipt.
The comparison phase excludes the final stage receipt and view emission.
All group and ordered decision digests matched.

| Dense fixture | Median wall time | Logical reads | Device reads | Device writes | Median peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh with group checkpoints | 4.522 s | 634 MB | 153 MB | 232 MB | 196 MiB |
| Fresh with chain decision checkpoints | 4.539 s | 620 MB | 153 MB | 234 MB | 125 MiB |
| Retry from the last group checkpoint | 2.456 s | 727 MB | 199 MB | 116 MB | 196 MiB |
| Retry from two decision windows | 2.366 s | 703 MB | 200 MB | 117 MB | 125 MiB |
| Retry from the complete decision log | 0.779 s | 265 MB | 195 MB | 116 MB | 121 MiB |

The dense decision log held 1.44 MB for 160,000 changed decisions. The
fresh sparse median went from 0.918 s to 0.849 s, and its log held 36 kB.
The reason pass streams its query rows and holds only the updates and
checkpoint records for the current window. A complete log skips group-link
replay and representative compression on retry.

#### Native chain compression measurements

The 400,000-document NVMe comparison used the sparse and dense pair
fixtures above and a third fixture with 6,250 chains of 64 documents.
Its descending insertion order made SQL pointer jumping take six
update passes plus a final detection scan. Each measurement is from
one cold run of the native dedup phase, including the disposable
decisions database. Final stage receipt verification is outside the
timing. The ordered decisions and group report matched in each fixture.

| Run | Elapsed | Logical reads | Device reads | Device writes |
| --- | ---: | ---: | ---: | ---: |
| Sparse pairs, prior fresh | 1.055 s | 388 MB | 153 MB | 119 MB |
| Sparse pairs, checkpointed fresh | 1.064 s | 388 MB | 153 MB | 119 MB |
| Dense pairs, prior fresh | 4.900 s | 620 MB | 153 MB | 234 MB |
| Dense pairs, checkpointed fresh | 4.901 s | 620 MB | 153 MB | 234 MB |
| Deep chains, prior fresh | 7.871 s | 1.64 GB | 153 MB | 266 MB |
| Deep chains, checkpointed fresh | 5.845 s | 683 MB | 153 MB | 269 MB |
| Deep chains, prior retry after compression | 5.262 s | 1.78 GB | 226 MB | 123 MB |
| Deep chains, retry after 100,000 committed ordinals | 3.185 s | 806 MB | 226 MB | 125 MB |
| Deep chains, checkpointed retry after compression | 2.760 s | 693 MB | 229 MB | 120 MB |

The deep-chain run recorded 387,500 changed final representatives in
3.10 MB. A stop after the first committed window left three windows
to resolve. A stop after the final compression receipt skipped root
resolution on retry. Group link replay and initial construction of the
disposable database still ran. Peak RSS on the fresh deep-chain runs
rounded to 134 MiB for both implementations. Production scale and slower
storage remain open.

#### Native ranked reduction measurements

One cold NVMe run per case reused the three 400,000-document fixtures
above with the ranked resolver. The timings include opening the temporary
decision database and deduplication. Final stage hashing and view emission
are outside the interval. Ordered decision and group report digests matched.

| Run | Elapsed | Logical reads | Peak RSS |
| --- | ---: | ---: | ---: |
| Sparse pairs, prior fresh | 1.934 s | 258 MB | 322 MiB |
| Sparse pairs, checkpointed fresh | 1.382 s | 284 MB | 124 MiB |
| Dense pairs, prior fresh | 4.452 s | 253 MB | 235 MiB |
| Dense pairs, checkpointed fresh | 4.256 s | 283 MB | 146 MiB |
| Deep chains, prior fresh | 4.790 s | 263 MB | 128 MiB |
| Deep chains, checkpointed fresh | 4.652 s | 279 MB | 128 MiB |
| Dense pairs, prior retry after group scan | 1.794 s | 350 MB | 235 MiB |
| Dense pairs, checkpointed retry after group scan | 1.607 s | 376 MB | 147 MiB |
| Dense pairs, retry after 200,000 reduction rows | 1.367 s | 375 MB | 146 MiB |
| Dense pairs, retry after final decisions | 0.834 s | 265 MB | 122 MiB |

The dense fixture wrote 1.44 MB each for the component and decision
journals. Fresh device reads stayed near 153 MB, and writes rose from
232 MB to 238 MB. The final retry read 195 MB from storage and applied
the finished decisions directly. Stops before the final decision receipt
still replay group links. Production and slower-storage measurements
remain open.

#### Native large duplicate group measurements

Cold NVMe trials covered 400,000 synthetic documents in three layouts.
The single-group fixture put one validation document and 399,999 training
copies in the same raw group. The dense and sparse fixtures had 320,000
and 8,000 groups across the raw and normalized passes. The timed interval
included opening the decisions database and resolving duplicates. Retry
followed SIGKILL after the last group receipt and before any later pass.
Group reports and ordered decisions had identical digests.

| Layout and resolver | Phase | Elapsed, prior → streamed | CPU, prior → streamed | Peak RSS, prior → streamed | Logical reads, prior → streamed |
| --- | --- | ---: | ---: | ---: | ---: |
| Single group, chain | Fresh | 2.814 → 2.787 s | 2.298 → 2.332 s | 342 → 176 MiB | 478 → 425 MB |
| Single group, chain | Retry | 2.288 → 2.055 s | 1.708 → 1.636 s | 236 → 134 MiB | 360 → 330 MB |
| Single group, ranked | Fresh | 2.775 → 5.323 s | 2.331 → 2.286 s | 344 → 177 MiB | 307 → 308 MB |
| Single group, ranked | Retry | 2.002 → 2.131 s | 1.593 → 1.642 s | 175 → 125 MiB | 216 → 189 MB |
| Dense pairs, chain | Retry | 2.785 → 3.431 s | 2.557 → 2.625 s | 125 → 125 MiB | 712 → 604 MB |
| Dense pairs, ranked | Retry | 1.870 → 1.805 s | 1.663 → 1.705 s | 146 → 145 MiB | 376 → 268 MB |
| Sparse pairs, chain | Retry | 0.833 → 0.780 s | 0.808 → 0.757 s | 119 → 120 MiB | 254 → 252 MB |

The single large group used 1.6 MB of ordinal spool. The dense journal
grew from 5.12 MB to 6.40 MB to store representative ordinals. Fresh
device reads remained near 167 MB for the single-group fixture and
153 MB for the dense one. Dense retries still read about 199–200 MB
from storage. Removing the second report scan primarily saved logical
reads on these fixtures. Chain replay also looks up the representative
ordinal in SQLite. Fresh ranked elapsed time varied with similar CPU
times and device reads. In a reversed-order pair the single group took
2.860 → 5.028 s and dense pairs took 10.960 → 4.838 s. A third single-group
pair with phase timing took 3.317 → 2.637 s and recorded a 567 ms report
sync in the prior run. Treat the elapsed results as local observations.
Production-scale group sizes, a stop within a group, and slower storage
still need measurement.

#### Native normalization reuse measurements

One cold NVMe run per case used a 200,000-row, 399 MB synthetic JSONL
source. Each implementation first normalized the source, then ran the
same phase with a changed digest for `large_build.py`. The old key caused
re-normalization in 7.539 s with 399 MB of device reads and 591 MB of
writes. The new key reused the shard in 0.441 s with 591 MB of reads to
verify all prepared files and zero device writes. Fresh times were
7.736 s and 7.567 s respectively. The three prepared output digests
matched across all runs. These timings exclude acquisition verification
and downstream work. Production and slower-storage results remain open.

#### Native index range recovery measurements

Three cold NVMe trials per case indexed one prepared file pair with
400,000 rows and 843 MB of input. The fixture also exercised reviewed
row admission metadata. Each SIGKILL followed seven committed ranges,
covering 223,034 rows and 470 MB of paired input. All ordered table
digests matched, including document and span offsets, origin keys,
decisions inputs and admitted source row numbers.

| Large file run | Median wall time | Logical reads | Device reads | Device writes | Median peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior direct writer | 8.398 s | 2.98 GB | 843 MB | 457 MB | 86 MiB |
| Fresh with range checkpoints and bulk indexes | 7.819 s | 1.65 GB | 843 MB | 710 MB | 100 MiB |
| Retry after seven committed ranges | 4.210 s | 1.18 GB | 486 MB | 597 MB | 95 MiB |

The completed parts held 202 MB of compact index data before cleanup.
The final index was 364 MB, compared with 371 MB in the prior writer.
The 38 MB, 18,000-row control kept the direct path and ran in 0.333 s
before and 0.329 s after. Timings cover indexing through final SQL index
creation. Verification of the prepared shards and the final stage receipt
are outside the timing. This range-only implementation rebuilt compact
assembly and SQL indexes from the saved parts on retry.

#### Native index and dedup reuse measurements

One cold NVMe run per case used the 843 MB prepared input above, with
400,000 documents and 32,101 duplicate groups. After building each baseline,
the run changed only the recorded view or resolver source digest. Timings
include index and dedup receipt verification and any work triggered by the
changed key. Acquisition, prepared-shard verification and view emission are
outside the timed phase.

| Stage inputs | Elapsed before / after | Device reads before / after | Device writes before / after |
| --- | ---: | ---: | ---: |
| Fresh | 10.974 / 10.440 s | 843 / 843 MB | 870 / 870 MB |
| Changed view producer | 10.784 / 0.410 s | 843 / 486 MB | 870 / 0 MB |
| Changed duplicate resolver | 10.830 / 2.735 s | 843 / 364 MB | 870 / 158 MB |

The view change reused both retained stages. The resolver change reused the
index and rebuilt the duplicate decisions. Ordered index rows, the decision
database and the duplicate-group report matched across runs. Cold
verification still reads retained stage files. Production and slower-storage
wall time and I/O remain unmeasured.

#### Native compact assembly recovery measurements

The same 843 MB, 400,000-document synthetic NVMe fixture evaluated
transactional assembly. Each row below is a median of three runs unless
specified otherwise. The runs requested eviction of the input and
checkpoint files from the page cache. The timed phase covers indexing
through SQL index creation and excludes the outer prepared-file checks
and final stage hashing and publication. All ordered table digests matched.

| Checkpoint path | Elapsed | CPU time | Device reads | Device writes |
| --- | ---: | ---: | ---: | ---: |
| Range-only fresh | 7.914 s | 7.66 s | 843 MB | 710 MB |
| Durable fresh, cold verification | 8.128 s | 7.64 s | 843 MB | 712 MB |
| Durable fresh, with local receipts | 8.833 s | 7.52 s | 843 MB | 712 MB |
| Range-only retry after seven parts | 4.277 s | 4.14 s | 486 MB | 597 MB |
| Durable cold retry after seven parts | 5.647 s | 4.10 s | 599 MB | 485 MB |
| Durable receipt-backed retry after seven parts | 4.369 s | 3.92 s | 598 MB | 485 MB |
| Range-only retry after stop inside SQL index creation | 1.479 s | 1.38 s | 202 MB | 508 MB |
| Durable cold retry after stop inside SQL index creation | 2.747 s | 1.37 s | 406 MB | 304 MB |
| Durable receipt-backed retry after SQL index commit (one run) | 0.185 s | 0.10 s | 202 MB | 0 MB |

After seven parts, the durable retry retains the compact rows for 223,034
documents. The SQL indexes still read the retained database pages when
they are built. A cold retry also compares the retained rows with the
parts. The cold retry inside SQL index creation rolls back the interrupted
transaction and redoes indexing over the complete merged rows. After the
signed SQL index commit, a retry verifies all 13 parts and reuses the
completed database. The receipt-backed halfway trials ranged from
4.288 s to 5.469 s. Production and slower-storage wall time, scratch and
I/O remain unmeasured.

### Select families, screen duplicates, and build a schedule

| Step | Committed unit | Work repeated after a stop | Measured evidence |
| --- | --- | --- | --- |
| Source split inventory | Receipted append windows at verified source-file boundaries, then the native completion evidence. | Verify committed inventory bytes. Resume at the next file in the ordered snapshot inventory. An unfinished window can include up to 64 MiB or 128 files, plus the file that crosses the byte bound. | The connected preparation fixture interrupts after one committed file and matches the clean inventory while skipping that file on retry. |
| `freeze-splits` | Verified final split bundle. Completed preparation also commits external-sort and join phases plus export chunks. | The ordinary path repeats unfinished assignment work. The completed-preparation route resumes at the last authenticated phase and reuses intact export chunks. | Production restart cost unmeasured. |
| `derive-family-inventory` | Verified file-backed range segments and an ordered assembly receipt per appended range. | Verify the assembled prefix and continue at the next range. Validate cached segments for the unfinished suffix, and regenerate missing or damaged segments. | See [declared family assembly measurements](#declared-family-assembly-measurements). |
| `finalize-family-inventory` | Receipted native merge windows at 64 MiB of combined input, with document and candidate offsets. Completed preparation can reuse the release's authenticated family inventory. | Verify the frozen candidate bundle and committed output prefix. Resume the ordered merge from the saved offsets, or republish the authenticated retained inventory. | See [native family finalization measurements](#native-family-finalization-measurements). |
| Near-duplicate screen | Verified held-out and sample shingle evidence per range, plus training offsets and accumulators every 200,000 training records by default. | Reuse evidence and rebuild the in-memory tables. Read source ranges with missing or damaged evidence and screen the uncommitted training suffixes. | See [near-duplicate evidence measurements](#near-duplicate-evidence-measurements). The historical 6.7M-document Workshop screen took 1–2 h on one core. [#491](https://github.com/supernovae/model-lab/pull/491) |
| Seeded tokenizer sample and verification | TRAIN lineage, rank histogram, and candidate range receipts with `--scratch-dir` on `tokenizer train` and `tokenizer verify`. | Reuse completed ranges and scan missing or damaged ranges. Reassemble the sample and read its selected payloads. A run without scratch does its own selection scans. | 13 s at 12 workers on a warm-page-cache synthetic 11.9 GB, 2M-document release with a 32 MiB fit quota. This measures selection. [#499](https://github.com/supernovae/model-lab/pull/499) |
| Tokenizer fit and bakeoff | Complete staged tokenizer and provenance, published together, or completed bakeoff artifacts. | Verify and publish a matching staged pair without fitting again. A stop before both staged files are complete repeats the native fit. Concurrent fits coordinate on the destination and publication preserves an existing destination. | A 16 MiB synthetic fit took 1.04 s, and its retry after a stop inside BPE spent 1.03 s in the trainer call. See [the fitting measurements](#tokenizer-fit-timings). |
| `materialize-mixture` | Authenticated output prefixes, index event segments and a reusable content token cache. Default commit thresholds are 4,096 candidates, 64 MiB or 30 s. | Verify committed bytes in place, trim the uncommitted tail, restore index state, and continue writing the same files. Reuse unchanged token encodings. | 24.2 s cold and 12.9 s with warm token cache on a synthetic 210M-token mixture. [Build and checkpoint measurements](token-store.md#mixture-speed) |
| `verify-mixture`, `--reseal`, and `--audit` | Complete verification or completion receipt. | Routine verification and reseal may repeat output checks and hashes. `--audit` reruns an independent schedule in fresh scratch with token reuse disabled. | A real full audit took 94 min. The synthetic 6,000-document audit took 1.1 s with a mocked tokenizer. [#521](https://github.com/supernovae/model-lab/pull/521) |

Retry `split-inventory`, `derive-family-inventory` or native
`finalize-family-inventory` with the same output path to resume a private
checkpoint beside it. The ordered source scan commits at file boundaries
to keep snapshot validation and document IDs intact. Derivation appends
verified worker ranges in input order and commits each range. By default
it removes the disposable worker segment after the append commits.
An explicit `--scratch-dir` retains worker segments for reuse in other
attempts. The native finalizer joins the release rows and frozen candidates
in document-ID order and checks each pair's split, content digest and
stratum, including dropped documents. The join runs in one ordered pass
and keeps one input pair in memory. These operations bind receipts to the
input generations and producer code. Each completed inventory is linked
into place atomically. Its stage directory is removed after publication.

#### Native family finalization measurements

One cold NVMe trial per case used 200,000 synthetic rows with 514 MB of
document JSONL and 46 MB of frozen candidates. Each trial included real
frozen bundle validation and excluded release verification. The streamed
retry followed SIGKILL after 71,925 pairs, with 185 MB of document bytes
and 16 MB of output already committed. All three inventories had the
same digest and 44 MB of output.

| Run | Wall time | CPU time | Peak RSS | Logical reads | Device reads | Device writes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Row-store join, fresh | 7.208 s | 6.721 s | 67 MiB | 3.21 GB | 560 MB | 1.68 GB |
| Ordered merge, fresh | 1.888 s | 1.461 s | 59 MiB | 626 MB | 560 MB | 44 MB |
| Ordered merge, retry | 1.076 s | 0.942 s | 60 MiB | 441 MB | 391 MB | 28 MB |

The retry still hashed the frozen candidates and verified the output
prefix. Production and slower-storage recovery measurements remain open.

#### Declared family assembly measurements

Two cold fresh pairs, a cold retry pair and a warm cached-segment pair
used 400,000 synthetic documents and 1.08 GB of JSONL on NVMe. The cold
trials used two workers and excluded release verification. Both retry
attempts followed a SIGKILL after six assembled ranges. The prior path
retained worker segments through `--scratch-dir`. The new path also
retained the assembled output prefix. Every run produced the same digest
for 378,947 kept documents and 88 MB of output.

| Run | Wall time, prior → checkpointed | Logical reads, prior → checkpointed | Device writes, prior → checkpointed |
| --- | ---: | ---: | ---: |
| Fresh cold | 5.49–5.79 → 5.20–5.68 s | 1.50 → 1.32 GB | 264 → 176 MB |
| Cold retry | 4.303 → 4.154 s | 1.09 → 0.88 GB | 231 → 111 MB |
| Warm segments, new output | 1.111 → 1.013 s | 354 → 178 MB | 176 → 88 MB |

Cold device reads stayed near 1.08 GB fresh and 709 MB on retry. The
writer hashes bytes during append and links the completed file into place.
It verifies the committed prefix on retry and takes the remaining segment
inputs from the existing runner. Producer identity changes start a new
assembly generation. Published inventories keep their existing format.

#### Near-duplicate evidence measurements

Two cold NVMe pairs used 400,000 synthetic documents, including 8,000
held-out documents, across 478 MB of JSONL. Setup covered the evidence
scan, template tables when enabled, held-out comparisons and LSH
construction. Each trial used one worker, 32 MiB shards and a 22-bit
template sketch. Release verification and the training pass were outside
the interval. A SIGKILL immediately after setup left all evidence ranges
available for the retry.

| Setup | Elapsed, prior → checkpointed | Device reads, prior → checkpointed |
| --- | ---: | ---: |
| Lexical, fresh | 1.95–2.10 → 2.16–2.25 s | 478 → 478 MB |
| Lexical, retry | 1.86–2.01 → 0.32–0.33 s | 478 → 11.5 MB |
| Template-aware, fresh | 2.69–2.71 → 3.13–3.26 s | 478 → 478 MB |
| Template-aware, retry | 2.62–2.87 → 0.90–0.91 s | 478 → 12.5 MB |

The lexical and template-aware runs retained 6.8 MB and 7.8 MB of
evidence respectively. The template-aware setup peaked at about
98–99 MiB before and 102–104 MiB with the checkpoints. Restored evidence
still goes through the held-out caps, in-memory template reduction and
LSH construction. The existing outcome test checks report parity and
rebuilds one corrupted evidence range after an interrupted screen.
A separate cold end-to-end template-aware pair used two workers and the
same inputs. The full screen took 30.340 s before and 31.548 s with
checkpointing, including the training comparison pass. Both processes
read 478 MB from storage and produced identical report digests.

### Prepare data, train, and score

| Step | Committed unit | Work repeated after a stop | Measured evidence |
| --- | --- | --- | --- |
| Streaming `data prepare` | Sealed chunks, configured by [preparation limits](runtime.md#interrupted-preparation). A finished split remains reusable. | Recheck sealed chunks, replay in-flight records, and repeat unfinished final array assembly or hashing. | Current interrupted-preparation cost unmeasured. |
| Token-store and mixture packing paths | Reusable completed token-store or mixture input and completed prepared bundle. | Reuse verified input encodings and rebuild unpublished arrays and masks where applicable. | Mixture packing achieved 126M tokens/s on the synthetic 210M-token workload. [Mixture speed](token-store.md#mixture-speed) |
| Stage validation and warmup | Complete verified stage bundle. Prepared inputs can be reused separately. | Restart the interrupted disposable pilot using the same declared inputs. | Runtime-specific interruption cost unmeasured. |
| Training and post-training objectives | Full-state checkpoint at a committed optimizer update, including data cursor and random state. | Resume from the newest valid checkpoint and repeat updates since that checkpoint. Supported device-layout changes require explicit runtime options and record numerical drift. | Validate cadence and preemption costs with [#516](https://github.com/supernovae/model-lab/issues/516). See [Checkpoints](checkpointing.md#resume) and [Post-training](post-training.md). |
| Evaluation and fixed panels | Complete result for one checkpoint and panel. A panel can retain incomplete accounting. | Repeat the interrupted pass over the same panel and checkpoint. | Per-panel recovery cost unmeasured. |
| Probe battery | Sealed result per invocation, including evidence from a graceful interruption when available. | Replay the interrupted battery on the same authorized split, subject to the final-split rules. | Current tier-specific recovery cost unmeasured. See [Probes](probes.md). |
| Post-training weight export and quantization | Completed exported artifact or quantization result. | Restart the interrupted export, or repeat conversion and its evaluation from the source checkpoint. | Large-checkpoint replay cost unmeasured. |

The recovery checklist tracks phase and scale gaps. [#485](https://github.com/supernovae/model-lab/issues/485)
Related work covers release verification [#450](https://github.com/supernovae/model-lab/issues/450),
reference releases [#567](https://github.com/supernovae/model-lab/issues/567),
token schedules [#569](https://github.com/supernovae/model-lab/issues/569),
and training recovery [#516](https://github.com/supernovae/model-lab/issues/516).

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
  [Materialize a mixture](#materialize-a-mixture)). Finalization resolves its
  inputs through the verified release and checks their generation before
  publishing. It reuses the family inventory from an authenticated completed
  build and refuses mismatched split evidence or an existing output.
  Upgrading across a split producer change restarts unfinished merge
  checkpoints. Completed-preparation releases that require the current
  producer must be rebuilt. The retained-release route can authenticate
  unchanged historical execution through its existing evidence.
- **`audit-protected-lineage`** verifies both releases and both inventories.
  The profile (`loss_slices`) and suite (`items`, `chunks`) name the protected
  validation and test documents and must be bound to the prior release and
  inventory. The receipt lists protected families or content now in the
  candidate's training split, and protected documents whose family, split or
  stratum changed. Any of these makes it `BLOCKED` and the command fails;
  otherwise `PASS`.
  Inventories in release order stream alongside the documents, with only
  protected IDs, families and content assignments held for comparison.
  Reordered rows use a disposable disk index for the unmatched suffix. Use
  `--scratch-dir` for cold verification and the index, and `--cold` to
  refresh release verification receipts from the bytes.
- **Acceptance.** A `klab-release-review-v1` record (decision `ACCEPTED`, the
  reviewer, the release, its family inventory and its lineage receipt, each
  with its SHA-256) is what an [attempt](#run-it-as-phases) binds as its
  `release_acceptance` artifact. It is refused unless the lineage receipt is
  `PASS` for this exact release and inventory.

### Publication and protected-lineage measurements

Single-process synthetic measurements on Linux tmpfs compare processing
after release verification with #651. The publication fixture used one
snapshot with 750,000 selected rows in a 117.64 MB manifest. The lineage
fixture used 100,000 documents per release with roughly 1 KiB of text per
document and one protected document moved to training. The reordered case
reversed the candidate inventory. Median of three processes per case.

| Operation | Elapsed before → after | Logical reads before → after | Peak RSS before → after |
| --- | ---: | ---: | ---: |
| Publication metadata | 402 ms → <1 ms | 117.66 MB → 0.02 MB | 497 MiB → 44 MiB |
| Lineage audit, ordered | 1.169 s → 0.738 s | 331.12 MB → 313.22 MB | 426 MiB → 46 MiB |
| Lineage audit, reordered | 1.268 s → 1.488 s | 331.12 MB → 352.95 MB | 426 MiB → 49 MiB |

The output digests matched in all cases. The slower reordered path trades
local index I/O for bounded memory. Cold release verification and the cost
of reordered inventories on slower devices were outside this benchmark.

## Export and tokenize

```sh
klab --work-dir WORK corpus export RELEASE --view lm --base-run-config base.yaml --vocab-size 32768
klab corpus verify-export RUN.yaml
```

An export writes a tokenizer config and a run config whose `dataset` binds the
release, view, base config and vocabulary. Consumers must keep that `dataset`
section exactly; `verify-export` checks it.

When a host-local proof store is available, export creation records a signed
verification receipt for `export.json`, both configs, the release and its
pinned inputs. The first consumer can reuse the export's verified counts and
license labels. Tokenizer output under `tokenizer/` and prepared data under
`prepared/` have their own verification and can be added without invalidating
the export receipt. Use `klab corpus verify-export RUN.yaml --cold` to replay
the release and export checks from bytes.

Export creation and cold verification resolve split and evidence files through
the verified release reader. Repeating the same export with a valid receipt
reuses the exact counts and license labels. It derives both configs from the
requested base again, compares their bytes and checks the export generation
before returning. A missing or stale receipt repeats the scans.

Export combines declared source licenses with the exact license labels on
normalized documents. When document-level labels are present, the scan uses
line-aligned ranges under the Corpus worker limit. Pass the same
`--scratch-dir` on local storage you control to resume completed license
ranges and LM or chat split counts. Chat range results bind the renderer and
report errors at the original file's line numbers. Release verification checks
the artifact and its provenance before these results are used.

### Export verification measurements

A synthetic NVMe fixture had 896 documents. The documents and the train
and validation views totaled 224 MB. A release verification receipt was
already present. The table reports medians of three trials, with page-cache
eviction requested before each timed operation. The last row adds files at
the paths used for tokenizer output and prepared data before verification.

| Operation | Previous time | Export receipt at publication | Storage reads before and after |
| --- | ---: | ---: | ---: |
| Export creation | 0.500 s | 0.508 s | 224 MB in both |
| First export verification | 0.476 s | 0.228 s | 224 MB to 0 MB |
| Export verification after adding derived outputs | 0.474 s | 0.229 s | 224 MB to 0 MB |

Process reads for verification fell from 229 MB to 1.4 MB. These runs
measure export creation and verification. Production-scale and slower
storage measurements remain open.
[#450](https://github.com/supernovae/model-lab/issues/450)

For repeated export creation, the following local measurements used built
and frozen synthetic LM corpora, one worker and warm signed receipts.
Wall times and process reads are medians of three repeats per size. The
payload sizes include documents and the train and validation views.

| Documents | Payload | Wall time before / after | Logical reads before / after |
| --- | ---: | ---: | ---: |
| 128 | 12.97 MB | 19.9 / 10.9 ms | 13.30 / 0.19 MB |
| 512 | 51.89 MB | 57.4 / 20.6 ms | 52.95 / 0.58 MB |

### Export license scan measurements

The scan below used 1 million synthetic documents (775 MB) and three distinct
license labels on NVMe. Times for uninterrupted scans are medians of three
runs with Python 3.14. The cold runs requested page-cache eviction of the
fixture. These measurements cover the license scan.

| Operation | Workers | Warm | Cold |
| --- | ---: | ---: | ---: |
| Serial baseline | 1 | 1.184 s | 1.631 s |
| Range scan | 1 | 1.200 s | |
| Range scan | 4 | 0.470 s | 0.519 s |

Linux reported 775 MB fetched from storage for both cold scans. Process
reads were 775 MB for the serial baseline and 800 MB for the four-worker
scan, including worker startup.

A SIGKILL after four of twelve commits let one retry reuse those four
ranges. The retry took 0.434 s and read 531 MB at the process boundary,
compared with 800 MB for a fresh four-worker pass. With all ranges
complete, a warm scan took 1.2 ms and logical reads fell to 1.5 MB,
including range-boundary probes. Twelve receipts occupied 1,428 bytes of
file contents. Every run produced the same labels. [#485](https://github.com/supernovae/model-lab/issues/485)

### Chat export split measurements

A synthetic 821 MB split contained 600,000 v2 chat records with non-ASCII
text. Three runs per condition on NVMe produced the same 742,628,890
rendered UTF-8 bytes. Cold runs requested eviction of the fixture from
the page cache.

| Chat split counting | Workers | Warm | Cold |
| --- | ---: | ---: | ---: |
| Serial baseline | 1 | 3.962 s | 4.011 s |
| Range scan | 1 | 3.754 s | |
| Range scan | 4 | 1.167 s | 1.185 s |

Both cold scans fetched about 821 MB from storage. After SIGKILL at the
fourth of thirteen commits, the four-worker retry reused four ranges and
took 0.921 s. A warm pass with all receipts took 1.5 ms and read 1.6 MB,
including the range-boundary probes. The thirteen receipts contained
1,404 bytes. These timings cover split counting. Release verification,
total export time and slower storage remain separate measurements.

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
sampling digest therefore do not depend on the worker count. Give both commands
the same `--scratch-dir` on storage you control to reuse completed ranges
within a fit, across retries, and during verification.

```sh
klab tokenizer train TOKENIZER.yaml --scratch-dir SCRATCH
klab tokenizer train TOKENIZER.yaml --cold --scratch-dir SCRATCH
klab tokenizer verify RUN.yaml --scratch-dir SCRATCH
```

Training reuses a valid host-local Corpus export receipt. Use `--cold` to
verify the export and release from bytes before fitting. For seeded sampling,
the authenticated export supplies the declared document, TRAIN lineage and
source rights paths. The sampler checks that generation when it finishes
selection and retains the resolved document path and input identities for
both the fit and audit.

Receipts bind the input files, range options and sampling implementation. An
interrupted run skips intact ranges and rebuilds missing ones. The scratch
directory holds disposable sampling and LM export counting work. Within
each attempt, the byte check, fit and sample audit share one selection of
document offsets. Lab checks the input file identities when using that
selection. The native BPE fit restarts if interrupted.
For verification with independent range scans, omit `--scratch-dir`.

The audit then takes the documents that follow the fit in the same order,
different from the fit, and checks round-trip failures, unknown tokens and bytes
per token; it refuses to publish on a failure. No validation or test document is
used.

The fit writes the tokenizer and manifest in a private sibling workspace and
publishes them together. A retry at the same output verifies the training
contract and file digest, and reruns the sample audit on the reused selection.
A complete staged pair skips BPE. If the stop left only partial staging, the
retry rebuilds that fit. Keep the sibling workspace while retrying. The command
preserves an occupied destination and refuses an unverifiable published pair,
including partial outputs left by older versions.

#### Tokenizer fit timings

The following CLI measurements cover startup, export verification, selection,
and selected input replay up to the BPE call. They use actual built and frozen
synthetic releases with their export receipts already present, one worker and
warm files. The table reports medians of three repeated runs per size. The fit
digests matched.

| Documents | Time before / after | Logical reads before / after |
| --- | ---: | ---: |
| 128 | 187.9 / 33.7 ms | 107.64 / 13.45 MB |
| 512 | 642.5 / 72.4 ms | 428.95 / 52.91 MB |

The fixture used 8,192 generated documents, a 16 MiB fit quota, a 2,048-token
vocabulary and Tokenizers 0.22.2 on CPU with one selection worker. It supplied
the export binding from the fixture so the timings isolate sampling, the
native fit and the audit. Each row is one local run on NVMe. The fit column
includes feeding the selected text into the trainer. Earlier runs include
repeated selection in the fit and audit columns. These runs preceded the
indexed-read change below.

| Operation | Selection calls | Native fit | Sample audit | Total |
| --- | ---: | ---: | ---: | ---: |
| Baseline `e35a2fce` | 3 | 1.158 s | 0.345 s | 1.656 s |
| One selection per attempt | 1 | 1.040 s | 0.130 s | 1.320 s |
| Retry after a stop inside the fit, with saved ranges | 1 | 1.035 s | 0.130 s | 1.217 s |

SIGKILL stopped the trainer after it consumed the input iterator's documents
and before it returned. The retry fed all 16,766,888 selected bytes again.
The sample and tokenizer digests matched across runs. Observed peak RSS was
379 MiB on the baseline and 376 MiB with one selection per attempt. Production
fit, audit and recovery costs remain unmeasured. [#485](https://github.com/supernovae/model-lab/issues/485)

In a separate publication test on NVMe, SIGKILL stopped fits on an
8,192-document synthetic release with a 16 MiB quota and a 2,048-token
vocabulary. The fixture supplied the export binding and the retries reused
sampling receipts. The destination was absent after both stops. Each retry
published the same tokenizer and manifest bytes as an uninterrupted fit.

| Stop point | Staged artifact bytes | Retry time in BPE | Total retry time |
| --- | ---: | ---: | ---: |
| After writing the tokenizer, before its manifest | 120,508 | 1.787 s | 3.276 s |
| After syncing both files, before publishing | 124,141 | 0 s | 1.372 s |

#### Indexed reads and sequential scans

Tokenizer fit and audit replay the selected rows in seeded order. They keep
the document file open for each pass and read exactly the byte range of each
selected row. Where the OS accepts prefetch hints, the fit stream looks
ahead at up to 256 rows and 1 MiB of requested byte ranges at a time.
Release source replay, disk-backed build assembly, view lineage measurement,
and admission inspection also use exact byte ranges for indexed lookups.
Keep sequential scans buffered, including candidate selection and mixture
exposure replay.

On a 143 MiB synthetic JSONL file on NVMe, replaying an existing selection of
8,192 rows requested 17.8 MiB. Five runs per condition on Linux and Python
3.14.8 gave these medians. The cold runs requested eviction of the fixture
from the page cache before replay. All readers produced the same sample
digest.

| Tokenizer sample replay | Bytes returned to the process | Warm time | Cold time | Cold storage bytes |
| --- | ---: | ---: | ---: | ---: |
| Buffered open and seek per row | 1,023.4 MiB | 119 ms | 337 ms | 142.6 MiB |
| Exact positional reads | 17.8 MiB | 44 ms | 536 ms | 50.0 MiB |
| Exact positional reads with bounded prefetch | 17.8 MiB | 46 ms | 84 ms | 44.5 MiB |

On the same fixture, keeping one buffered file open for shuffled lookups
still returned 57× the requested bytes. A buffered scan of 8,192 consecutive
rows took 8.5 ms versus 9.7 ms for individual positional reads. Linux `rchar`
reports bytes returned by read calls, and `read_bytes` tracks bytes fetched
from storage by the process. Production stage throughput and slower storage
remain to be measured. [#485](https://github.com/supernovae/model-lab/issues/485)

### Tokenizer bakeoff

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

Bakeoff resolves train and validation members through the verified release
reader. Selection reads the document table once for both splits, and the
report binds the view digests from the release manifest. The reader checks
the release generation before publication. Text for grouped documents is
still retained while selecting and scoring the sample.

The table uses medians of three warm synthetic runs and measures selection
and binding after release verification. Linux `rchar` reports logical
process reads. The binding digests matched at both sizes.

| Documents | Wall time before / after | Logical reads before / after |
| --- | ---: | ---: |
| 128 | 11.7 / 4.4 ms | 18.49 / 6.17 MB |
| 512 | 49.5 / 16.2 ms | 73.97 / 24.70 MB |

### Measure source token supply

```sh
klab corpus measure-tokens RELEASE --tokenizer TOKENIZER --policy POLICY --output RECEIPT
```

The measurement counts distinct kept training content per requested domain
using the supplied tokenizer and the token floors in the policy. Use
`--tokenizer-origin-release` with `--family-inventory` when the tokenizer
comes from an earlier release. The inventory must cover every kept
document and respect family, content and measured-stratum isolation.

Counting hashes the verified release's document member in one pass. When an
inventory is supplied, the same pass verifies its rows and digest. Scratch
SQLite indexes store family, content and per-domain deduplication keys. If
the inventory is reordered, the join indexes its unmatched suffix. The
receipt is published only after the input generation checks pass. An
existing receipt checks the current input bindings with one document scan
and skips tokenization. Regenerate source-token receipts made with the
previous measurement implementation using a fresh output path.

The following medians compare three Linux tmpfs runs per version after
release and tokenizer authentication. The fixture had 100,000 ordered
documents of roughly 1 KiB each and a separate tokenizer origin of the same
size. Fresh runs used a single-threaded WordLevel tokenizer. Counts and
input digests matched. Cold verification, production storage and reordered
inventories were outside this comparison.

| Operation | Elapsed before → after | Logical reads before → after | Peak RSS before → after |
| --- | ---: | ---: | ---: |
| Fresh measurement | 12.732 s → 11.847 s | 693.59 MB → 156.72 MB | 269 MiB → 194 MiB |
| Existing receipt | 1.174 s → 0.668 s | 434.04 MB → 138.76 MB | 247 MiB → 176 MiB |

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

- **Resume.** An interrupted run resumes from its last committed frontier in
  `.OUTPUT.assembly-ID`, where `ID` names the declaration, release manifest,
  family inventory, tokenizer and Lab source the checkpoint was written for;
  keep that directory and the output parent's `cache`. Another declaration,
  release, inventory, tokenizer or Lab revision beside the same output starts
  its own checkpoint, and the command lists the others at once (delete the ones
  you will not resume). A checkpoint whose `request.json` disagrees with its
  name is refused before any verification, sampling or indexing, naming the
  directory to delete. The commit markers bind the lengths and checksums of
  the staged token and record files. On retry, the writer checks the committed
  bytes in place, refuses damage to the committed prefix, discards the
  uncommitted tail and rebuilds disposable SQL state from the index events.
  In reference mode, retry derives the range file from the committed record
  prefix and the sealed segments before publishing.
  Publication renames the staged files into the output. It removes the index
  segments and commit markers and keeps `progress.json`. Checkpoints from
  earlier Lab revisions start a separate assembly. Published mixtures can
  refresh their completion proof with `verify-mixture --reseal` if required.
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
- **Token cache.** Add `token_cache: sealed` to the mixture declaration to
  write bounded [sealed content segments](token-store.md#sealed-content-segments)
  as the existing encoder runs. `token_cache: append` is the default.
- **Token output.** `token_output: references` uses the sealed cache and
  publishes a [reference schedule](token-store.md#reference-schedules).
  Configure `dataset.train_path` from the command's `train_schedule_path`
  field. The default `token_output: contiguous` reports
  `train_tokens_path`.
- **Progress.** `.OUTPUT.assembly-ID/progress.json`, refreshed every few
  seconds, reports per-stratum counts, time spent waiting for the encoder,
  worker utilization and memory.
- **Verify.** `verify-mixture` checks the recorded completion and current
  inputs without rerunning; `--audit` reproduces the whole schedule without the
  token cache. Both commands authenticate the tokenizer's export and byte sample
  through the same signed receipts as the release:
  the export is read from its bytes and the sample scanned once, by the first
  command that needs them, and later commands check file metadata only.
  Mixture input paths, including the held-out view used by training, resolve
  from the verified release manifest. When the receipt specifies exposure or
  row-layout checks, verification hashes the records during the same pass that
  recomputes those checks.
- **Reseal.** When the recorded completion is gone or no longer matches the Lab
  code (`native mixture completion unavailable or changed`), `verify-mixture
  --reseal` records it again without replaying the schedule: it hashes
  the contiguous ID and record files in parallel, or checks the range file,
  records and sealed segment dependencies for a reference schedule. It
  authenticates the release and tokenizer through
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
Set `--token-cache sealed` to compare immutable cache segments with the
default append cache. With that cache, `--token-output references`
separates schedule time, dependency verification and logical target scans.

The record-file check below used three warm runs per size with exposure and
row-layout verification. It measures record verification alone. Logical
reads come from the Linux process counters.

| Records | File size | Logical reads before / after | Median time before / after |
| --- | ---: | ---: | ---: |
| 100,000 | 21.30 MB | 63.90 / 21.30 MB | 199.6 / 184.9 ms |
| 1,000,000 | 213.00 MB | 639.00 / 213.00 MB | 1.681 / 1.556 s |

A mixture declaration's `tokenizer_observation` (`method:
post_run_observation`) reuses a tokenizer from a completed local job by binding
that job's records and the tokenizer files, instead of verifying the fit from
scratch.

## Resources

Admission, splitting, building and mixtures stream metadata through temporary
on-disk databases under the work root and spool encoded data to disk, so memory
is bounded by individual records rather than corpus size. Individual rows are
limited to 8 MiB in admission and splitting and 64 MiB in mixtures. Scratch
space includes indexes, spools, index events and the staged output. Mixture
checkpoints share the staged token and record files through authenticated
prefixes. Watch free space on the corpus filesystem throughout. Reuse of completed work
depends on a private verification key under `XDG_CONFIG_HOME`; keep it, the work
root and the service user the same across restarts.
