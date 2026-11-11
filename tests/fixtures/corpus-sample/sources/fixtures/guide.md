# Corpus Forge local guide

A corpus release is a frozen, verified collection of derived views. Its source snapshots live in a private workspace, never in a model checkpoint.

## Preparing an input

Read the project declarations before acquiring a source. Use `klab corpus acquire` to copy declared bytes into an immutable snapshot.

- A declaration names a source revision and license.
- The acquisition lock records the selected blob hashes.
- A released view is verified again before tokenizer fitting.

```sh
klab corpus build corpora/devmind-sample-v0/corpus.yaml --offline
```

## Inspecting lineage

Use `klab corpus lineage` with a record identifier to inspect its parent documents, split and evidence spans. This command does not execute a tool or approve a generated answer.
