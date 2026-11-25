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

| Stage | Output under `WORK/corpora/<id>/builds` | Reused when |
| --- | --- | --- |
| normalize | `prepared/normalized-source-v2/<source>/<digest>`, one shard per source file | the shard's receipt and three output hashes verify |
| index | `stages/.index-<key>.groups` for incomplete work, then `stages/index/<key>/index.sqlite` | each group or the completed index has a matching verified receipt |
| dedup | `stages/.dedup-<key>.groups` for incomplete work, then `stages/dedup/<key>/dedup.sqlite` and `groups.jsonl` | the ordered group prefix verifies or the completed stage receipt verifies |
| views | `../.views-<key>/build` with receipts beside the staged output | matching build and stage inputs, with a verified append boundary |
| publish | the build directory itself, renamed into place atomically | the build exists and verifies |

The key of a stage is the SHA-256 of exactly the inputs it depends on, recorded
in its `receipt.json` next to the SHA-256 and size of every output file:

- **index**: the receipt digest of every prepared shard in order (which bind the
  source bytes, rights decision, normalizer options and document policy), the
  split declaration (with the digest of its assignment file), the declared
  duplicate keys, source ranks, reviewed row-level files and the implementation
  files `large_build.py`, `pipeline.py`, `split_assignments.py` and
  `dedup_policy.py`.
- **dedup**: the verified index receipt digest, the whole
  `release.policy.duplicates` section and the digests of `large_build.py`,
  `dedup_policy.py` and `view_checkpoints.py`.

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

The index stores file and byte offsets into the verified prepared shards,
alongside split, duplicate and admission metadata. The builder indexes groups
of prepared files up to about 64 MiB. A larger indivisible prepared file gets
its own group. Each completed group is verified on retry before its metadata
is assembled into the index. A stop during assembly may repeat the merge and
SQL index creation. After the final index is durable, the build removes the
group files. A changed input or producer starts a new index generation.

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
origins, and can close after 64 MiB of group and link output. An entire
duplicate group stays together and can cross either bound. Retry verifies
the append prefix and replays its links in a fresh decisions database.
It scans the index and calls the duplicate policy only for the unfinished
suffix. The final report is linked into the published stage on the same
filesystem, and the link file and journal are reclaimed after publication.
The representative compression, component and drop-reason passes can
still repeat. The scratch database from an interrupted pass is discarded.

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
with the release. Completed phases are keyed by their inputs, configuration
and every corpus module they can run, so a change to the corpus code renders
them again rather than reusing old shards.

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
against the manifest. For stage outputs, the ordered replay computes the
inventory checksum alongside the stage identity. It checks stage outputs,
documents, spans, lineage and view rows against the pinned source snapshots.
It runs on several processes and keeps
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
  `verify-mixture`, `preflight` and the training gate all use it, and each prints
  `release ID: verified by receipt` or `verified from the bytes in N s`. `build`
  and `export` have receipts of their own.
  The complete receipt is refused when any file was written, replaced, added or removed (an
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
| Native disk build, normalize | Receipted shard per source file. | Verify completed shards and re-normalize unfinished files. | Historical Workshop normalize, about 25 min for the 6.7M-document workload. [#491](https://github.com/supernovae/model-lab/pull/491) |
| Native disk build, index and dedup | Verified groups of prepared file references, the assembled index, verified duplicate-group windows, and the complete dedup stage. Reviewed row admission uses indexed file and row numbers. | Repeat the unfinished index group or compact index assembly. Verify and replay committed dedup group links, scan unfinished windows, and repeat the final representative and drop-reason passes as needed. | See [native index and payload read measurements](#native-index-and-payload-read-measurements) and [native dedup recovery measurements](#native-dedup-recovery-measurements). Historical Workshop indexing took about 30 min for roughly 13M rows. [#485](https://github.com/supernovae/model-lab/issues/485), [#450](https://github.com/supernovae/model-lab/issues/450) |
| Native disk build, views and publication | Verified span and view append boundaries at 25,000 input rows or 64 MiB, lineage and view measurement checkpoints, and a receipted census. Final publication by rename. | Verify the committed view prefix, reconstruct its digests and render the uncommitted tail. Reuse finished census work for matching inputs. The manifest hashes ancillary outputs and reuses the view checksums. A stop in the final lineage sort or measurement merge may repeat that step. | A 128 MiB tmpfs append fixture took 0.082 s with two commits and 0.071 s for prefix verification. [Native build census measurements](#native-build-census-measurements) cover the report phase. [#579](https://github.com/supernovae/model-lab/pull/579) |
| Completed-preparation build | Authenticated per-source render, completed sort and dedup phases, view range checkpoints and a signed receipt before publication. | Repeat an unfinished upstream phase. During view recovery, verify the committed prefix and resume from the stored span and document cursors. Intact staged views can publish directly, and a renamed build can recover completion from the signed receipt. A partial emission retry can recopy the frozen control inputs. | See [completed-preparation view recovery measurements](#completed-preparation-view-recovery-measurements). Production restart cost unmeasured. |
| Freeze | Native staging rename and build tombstone, signed verification after a staged replay, then the public release. Completed preparation seals a retained release against its existing build. | Reuse the complete staged receipt at the same path, or resume an unfinished check from eligible file and replay proofs. Retained freeze can repeat its final verification and seal. | Native freeze took 72.3 s on a 13.2 GB synthetic release, including build verification. The historical 374 GB copy took 92 min before in-place freeze. See [staged freeze recovery measurements](#staged-freeze-recovery-measurements). [#501](https://github.com/supernovae/model-lab/pull/501) |
| Release verification, `audit`, and `sources` | Signed checks of eligible inventory files, stage outputs and streaming raw source replay, then the complete release proof. `audit` and `sources` read the corresponding published report after verification. | A receipt-backed retry skips eligible completed checks and continues the remaining work. Cold mode starts the checks from bytes. | See [inventory hash recovery measurements](#inventory-hash-recovery-measurements), [stage output verification measurements](#stage-output-verification-measurements) and [raw source replay recovery](#raw-source-replay-recovery). An earlier cold audit took 67.3 s on 13.2 GB synthetic data, and the first receipt-backed audit took 0.94 s. [#500](https://github.com/supernovae/model-lab/pull/500), [#450](https://github.com/supernovae/model-lab/issues/450) |
| `describe` | Cached measurement runs, per-split link offsets and verified lineage index ranges when a proof store or persistent scratch is available. Release verification has its own receipt. | Resume missing index ranges, link files and measurement ranges. An unfinished index sort or cold release check can repeat. Each chat view is one measurement run. | See [describe lineage index recovery measurements](#describe-lineage-index-recovery-measurements) and [describe link planning recovery measurements](#describe-link-planning-recovery-measurements). An earlier full measurement took 55.1 s for 5M synthetic rows (23 GB), 12 workers, without a tokenizer. [#502](https://github.com/supernovae/model-lab/pull/502) |
| Corpus `export` | Document license labels and LM or chat split statistics by range with persistent scratch. Final export directory and verification receipt. | Reuse valid license and split statistics. Repeat unfinished ranges and final publication. Later consumers can use the complete receipt. | See [export verification measurements](#export-verification-measurements), [export license scan measurements](#export-license-scan-measurements) and [chat export split measurements](#chat-export-split-measurements). |

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
filesystem generation. A retry at that location reuses matching checks
and still reconstructs the document index. The progress journal counts
the size of each skipped raw file as reused bytes. Reviewed and chat
releases still use the row-store verifier. An explicit cold audit replays
all source files, and a stop within an uncommitted file repeats that file.

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

### Select families, screen duplicates, and build a schedule

| Step | Committed unit | Work repeated after a stop | Measured evidence |
| --- | --- | --- | --- |
| Source split inventory | Complete output with native completion evidence. | Rescan the admitted source files and write the inventory again if output was unpublished. | Production restart cost unmeasured. |
| `freeze-splits` | Verified final split bundle. Completed preparation also commits external-sort and join phases plus export chunks. | The ordinary path repeats unfinished assignment work. The completed-preparation route resumes at the last authenticated phase and reuses intact export chunks. | Production restart cost unmeasured. |
| `derive-family-inventory` | Receipted file-backed range segments with persistent `--scratch-dir`. | Verify existing segments, regenerate missing ones, and copy the segments again during final assembly. | Byte-identical fixture restart. Production assembly and verification cost unmeasured. [#576](https://github.com/supernovae/model-lab/pull/576) |
| `finalize-family-inventory` | Final inventory. Completed preparation can reuse the release's authenticated family inventory. | Rebuild the native candidate-to-document join and output, or republish the authenticated retained inventory. | Production restart cost unmeasured. |
| Near-duplicate screen | Training-pass offset and accumulators per range, every 200,000 training documents by default. | Rebuild the held-out index and template evidence when enabled. Replay training records after each range's commit. | Historical 6.7M-document Workshop screen, 1–2 h on one core. Held-out rebuild cost unmeasured. [#491](https://github.com/supernovae/model-lab/pull/491) |
| Seeded tokenizer sample and verification | TRAIN lineage, rank histogram, and candidate range receipts with `--scratch-dir` on `tokenizer train` and `tokenizer verify`. | Reuse completed ranges and scan missing or damaged ranges. Reassemble the sample and read its selected payloads. A run without scratch does its own selection scans. | 13 s at 12 workers on a warm-page-cache synthetic 11.9 GB, 2M-document release with a 32 MiB fit quota. This measures selection. [#499](https://github.com/supernovae/model-lab/pull/499) |
| Tokenizer fit and bakeoff | Complete staged tokenizer and provenance, published together, or completed bakeoff artifacts. | Verify and publish a matching staged pair without fitting again. A stop before both staged files are complete repeats the native fit. Concurrent fits coordinate on the destination and publication preserves an existing destination. | A 16 MiB synthetic fit took 1.04 s, and its retry after a stop inside BPE spent 1.03 s in the trainer call. See [the fitting measurements](#tokenizer-fit-timings). |
| `materialize-mixture` | Candidate and output segments and a reusable content token cache. Default commit thresholds are 4,096 candidates, 64 MiB or 30 s. | Verify committed segments, rebuild or verify indexes as needed, replay the uncommitted segment, and finish assembly. Reuse unchanged token encodings. | 24.2 s cold and 12.9 s with warm token cache on a synthetic 210M-token mixture. These are build timings. [Mixture speed](token-store.md#mixture-speed) |
| `verify-mixture`, `--reseal`, and `--audit` | Complete verification or completion receipt. | Routine verification and reseal may repeat output checks and hashes. `--audit` reruns an independent schedule in fresh scratch with token reuse disabled. | A real full audit took 94 min. The synthetic 6,000-document audit took 1.1 s with a mocked tokenizer. [#521](https://github.com/supernovae/model-lab/pull/521) |

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

When a host-local proof store is available, export creation records a signed
verification receipt for `export.json`, both configs, the release and its
pinned inputs. The first consumer can reuse the export's verified counts and
license labels. Tokenizer output under `tokenizer/` and prepared data under
`prepared/` have their own verification and can be added without invalidating
the export receipt. Use `klab corpus verify-export RUN.yaml --cold` to replay
the release and export checks from bytes.

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
klab tokenizer verify RUN.yaml --scratch-dir SCRATCH
```

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
