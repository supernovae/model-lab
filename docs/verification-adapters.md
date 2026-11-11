# Offline artifact adapters

These native commands perform no tokenizer fitting, source acquisition, model
initialization, scoring, or generation.

```sh
klab tokenizer verify RUN_CONFIG.yaml --json
klab evaluation freeze-items --release RELEASE_DIR \
  --family-inventory families.jsonl --draft reviewed-items.json \
  --output NEW_ITEMS_DIR --json
```

`tokenizer verify` reads a v2 RunConfig and binds its tokenizer path, model
vocabulary and dataset to the existing cold artifact verifier. It authenticates
the manifest and immutable origin, checks actual vocabulary size and special IDs
(`<pad>`, `<unk>`, `<bos>`, `<eos>` must be 0–3), and reports tokenizer/manifest
digests and origin. It creates no work directory or cache. The origin must be a
frozen corpus export, verified token mixture, or saved snapshot. Unbound mutable
or generated sources fail closed: their historical manifests cannot establish
unchanged source identity without replay. No historical identity is rewritten.

`evaluation freeze-items` calls the existing Card 03 source-bound item verifier
and writes its unchanged manifest as `NEW_ITEMS_DIR/items.json`. The release is
cold verified; the inventory must cover all retained documents without family
leakage. Drafts must bind source offsets, chunk digests, reviewed items and valid
controls. Complete denominators (200 closed-book and 400 open-book items, including
both missing-evidence outcomes) are mandatory; there is no partial CLI mode.
The command reports the native content and file digests, item/chunk digests,
and the digest of the category and missing-evidence denominator maps. Existing
output paths, including symlinks and empty directories, are rejected.

Already frozen items remain reusable through their existing identities. Freezing
checks structural review declarations, not semantic correctness of answers or
production evaluation qualification. Tiny fixtures establish adapter behavior;
actual retained release and item review remain separate research gates.
