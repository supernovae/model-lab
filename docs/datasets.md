# Datasets

A run's `dataset` section says where its text comes from. Lab reads it, prepares
token arrays once and reuses them for every run whose data, tokenizer and
packing match, whatever the model size or optimizer.

## Source kinds

`dataset.source` names how data is read, never which dataset it is, so a new
dataset needs no Lab change.

| `dataset.source` | Reads | Required fields |
| --- | --- | --- |
| `local_text` | JSONL files, one `{"text": ...}` document per line | `train_path`, `validation_path`, `license` |
| `local_chat` | JSONL conversations, rendered by a [chat template](post-training.md#chat-templates) | `train_path`, `validation_path`, `license` |
| `local_preference` | JSONL [chosen and rejected reply pairs](post-training.md#preference-optimization) | `train_path`, `validation_path`, `license` |
| `local_token_mixture` | a [token mixture](token-store.md#mixture-output) | `mixture_declaration_path`, `mixture_output_path` |
| `snapshot` | a pinned, retained [snapshot](#pin-a-dataset) | `revision`, `license`, `train_path`, `validation_path`, `source_manifest_path` |
| `huggingface` | a Hugging Face dataset streamed at a pinned revision | `dataset_id`, `revision`, `license` |
| `synthetic` | a built-in offline generator for smoke tests | none |

Paths are relative to the config file. Train and validation must be different
files, and local sources refuse a conversation or document that appears in both.

Every source also takes budgets: `train_max_documents`,
`validation_max_documents`, `train_max_tokens`, `validation_max_tokens`, and a
`cache_dir` for prepared arrays.

### Hugging Face streams

```yaml
dataset:
  source: huggingface
  dataset_id: HuggingFaceFW/fineweb-edu
  dataset_config: sample-10BT
  revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
  license: ODC-By-1.0; Common Crawl Terms of Use
  cache_dir: work/data
  train_max_documents: 500000
  validation_max_documents: 5000
  train_max_tokens: 400000000
  validation_max_tokens: 4000000
```

`text_field` (default `text`), `train_split` (default `train`) and
`validation_split` are optional. Without a validation split, validation is the
documents after the first `train_max_documents` of the training split. A stream
can cut its last document at the token budget; use a snapshot when you need
whole documents, deduplication and a local copy.

### Synthetic fixtures

`synthetic_task` picks `stories` (the default), `instruction_reference`,
`chat_recall`, `engram_recall` or `withheld_facts`. They exist for smoke tests
and the probe battery, not for training useful models.

## Pin a dataset

For a Hugging Face dataset you want to keep, write a source declaration, lock it
and take a snapshot:

```sh
klab data lock source.yaml --output source.lock.json --json
klab data snapshot source.lock.json --output snapshot --json
```

- **The declaration** names the repository, an exact commit revision (moving
  names such as `main` are refused), the configuration, the split mapping and
  the text field. It runs no repository code; unsupported formats are refused.
- **Selection:** `selection.mode: bounded` keeps a number of documents per
  split; `exhaustion` reads splits to the end. Both have resource limits. An
  exact-text deduplication policy can remove repeats within a split and drop
  training documents that also appear in validation.
- **The snapshot** keeps the lock, the selected records, their source positions,
  what was excluded and why, and content digests. Never edit one; a different
  selection is a new lock and snapshot.
- **Interrupted?** Repeat `klab data snapshot` with `--resume`, the same lock
  and the same output. Lab verifies what was already written and continues.
  Publication is atomic, so a partial snapshot never looks complete.

Point a run at the snapshot with `dataset.source: snapshot`, giving its
`train_path`, `validation_path`, `source_manifest_path`, revision and license.
Lab verifies those against the snapshot before fitting a tokenizer or preparing.

For files that need explicit control, a source declaration with
`kind: forge_files` selects verified members of a [corpus release](corpus-preparation.md).

## Prepare

```sh
klab tokenizer train tokenizer.yaml
klab tokenizer verify run.yaml --json
PREPARED=$(klab data prepare run.yaml)
klab data coverage --prepared-root "$PREPARED" --config run.yaml --json
klab data budget --prepared-root "$PREPARED" --config run.yaml \
  --passes 1 --no-require-full --output one-pass.yaml --json
```

- **`tokenizer train`** fits on training text only.
- **`tokenizer verify`** checks a run's tokenizer against its manifest and
  origin, and that `<pad>`, `<unk>`, `<bos>` and `<eos>` are IDs 0 to 3. It
  writes nothing.
- **`data prepare`** packs documents end to end with EOS, writes token arrays
  (plus loss masks for assistant-only chat and byte addresses for byte memory)
  and prints the prepared directory. It refuses budgets that would cut a
  selected document instead of truncating silently.
- **`data coverage`** reports what was used: documents kept and excluded,
  truncation, packed tokens, complete blocks and supervised targets.
- **`data budget`** writes a new config whose step and token limits make the
  given number of passes over the prepared blocks. `--require-full` (the
  default) refuses when the source was not fully covered. Review the schedule
  it implies before training.

For large corpora, the [token store](token-store.md) encodes each document once
and builds prepared arrays by copying.

### What makes prepared data reusable

A prepared directory is identified by the dataset declaration and the digests of
its inputs, the complete tokenizer, the packing settings and the version of
Lab's preparation code. Model shape, optimizer, batch sizes, worker counts and
file locations are not part of it, so models of different sizes share prepared
data. Changing any input, the tokenizer or Lab's preparation code produces a new
directory.

To copy prepared data to another disk, copy the whole directory, manifest
included, and point the config's `cache_dir` at the new location. Lab verifies
the copy's contents on load.

### Bounded windows and appended data

From Python, `prepare_data` can reuse an already prepared `local_text` cache
instead of encoding again:

- `prefix_from=PreparedPrefix(root, manifest_sha256)` takes a whole-document
  prefix of the parent, for a smaller run. Set both `train_max_documents` and
  `train_max_tokens` to the boundary.
- `append_from=PreparedAppend(root, manifest_sha256)` encodes only documents
  appended to the end of the parent's training file. The new file must contain
  the old one byte for byte.

Both verify the parent and require it to come from the same preparation code.
To continue a trained run on appended data, see
[appending data](document-batching.md#append-data-to-a-run).

### Several runs, one tokenizer

To compare runs on different snapshots with the same tokenizer, give each run's
`tokenizer` an `origin` that names the tokenizer's digests and the snapshot it
was fitted on:

```yaml
tokenizer:
  path: ../shared-tokenizer/tokenizer.json
  origin:
    sha256: <tokenizer.json digest>
    manifest_sha256: <tokenizer_manifest.json digest>
    dataset:
      # The complete dataset section the tokenizer was fitted with.
      source: snapshot
      revision: <revision>
      license: <license>
      source_manifest_path: ../origin-snapshot/manifest.json
      train_path: ../origin-snapshot/train.jsonl
      validation_path: ../origin-snapshot/validation.jsonl
      cache_dir: ../origin-cache
      train_max_documents: 500000
      validation_max_documents: 10000
      train_max_tokens: 1000000000
      validation_max_tokens: 10000000
```

`klab tokenizer verify ORIGINAL_RUN.yaml --json` prints the two digests. The
origin snapshot must stay available. Sharing a tokenizer does not make the two
datasets' splits disjoint; check that separately.

A tokenizer fitted on a mixture names `mixture_declaration_path` and
`mixture_declaration_sha256` in its `origin` instead of `dataset`. Lab checks the
mixture's release and tokenizer-fitting closure; it does not replay the mixture.
`klab data prepare` runs the same origin check.

## Corpus releases

A [corpus release](corpus-preparation.md) can be exported once per view, base
config and vocabulary and used by many runs. Each run keeps the export's
`dataset` and `tokenizer` sections exactly. `klab corpus verify-export RUN.yaml`
checks a run's release and export without preparing anything.

Releases may set `release.normalizer: normalizer-structure-v3`, which records
retained line spans, removed spans with reasons and document structure, and
drops metadata-only front matter from language-model views while keeping it in
the release.

## Hugging Face credentials

Lab uses your Hugging Face login for dataset streams, snapshots, corpus
acquisition and reference models. Create a read token at [Hugging Face token
settings](https://huggingface.co/settings/tokens), and accept a gated
repository's terms with the same account. Then either log in:

```sh
hf auth login
hf auth whoami
```

or set `HF_TOKEN` before starting `klab`, or pass a token file for one command:

```sh
klab --hf-token-file "$HOME/.config/klab/hf-token" corpus acquire corpus.yaml
```

Precedence is `--hf-token-file`, then `HF_TOKEN`, then the saved login. There is
deliberately no option that takes the token itself, since it would show in
process lists and shell history. Tokens never appear in declarations, manifests
or run records, and are sent only to `huggingface.co`. A missing or empty token
file fails rather than falling back to anonymous access. Remote workers and
containers need their own credentials.
