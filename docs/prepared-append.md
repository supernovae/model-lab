# Reuse a verified parent when preparing a fresh append

`prepare_data` can preserve a pinned, same-producer local-text parent and encode
only newly appended documents. This opt-in producer operation does not change
training defaults, model math, optimizer state or ordinary preparation behavior.
It is separate from `PreparedPrefix`, which derives a smaller benchmark selection.

```python
from klab.data.packing import PreparedAppend, prepare_data

prepared = prepare_data(
    next_config,
    tokenizer,
    append_from=PreparedAppend(parent_root, parent_manifest_sha256),
)
```

The caller supplies an independently retained parent manifest pin. The parent may
be a prepared cache or the immutable `data/` directory of a run. The new raw train
file must contain the entire old source **byte for byte**, followed by the fresh
JSONL tail. Every record must be complete, newline-terminated, and selected; a
parent whose raw source had unselected trailing records is deliberately rejected.
The existing local-text reader owns record validation. Both document count and
exact token count must increase. Every selected document is retained whole.

The implementation cold-verifies the parent arrays, validates its producer,
tokenizer, all-token objective and unchanged validation source, hashes the exact
historical raw prefix and new tail, then uses the native streaming collector for
only the new documents. Native train/validation disjointness is checked for the
new tail; the unchanged validation identity preserves the parent's prior check.
Parent IDs are copied through bounded array chunks without per-token Python loops.
Validation IDs are copied without re-encoding. Embedded EOS, changed historical
bytes, incomplete parents, changed tokenizers/validation, wrong pins and partial
or overlarge token caps are rejected.

The output is an ordinary native cache with a complete new source identity and an
`append_derivation` recording the parent manifest/cache/array identities, raw
prefix length/digest, tail digest/array digest, and old/new document/token counts.
Cold loading and signed proof reuse follow the existing native verifier. Failed
staging is preserved; atomic publication never replaces an existing cache. An
existing child cache does not waive verification of the requested historical raw
prefix. Sources and parent fingerprints are checked again before publication.

## Training continuation remains a separate native operation

For matched-story data, the target configuration must freeze the parent token
endpoint in `training.matched_stories.frozen_update_ends`. Preparation does not
invent an optimizer cursor or regroup consumed examples. The existing native
`append_data=True` budget-extension validation must additionally check the
parent's full group-offset prefix, grouping settings, validation, checkpoint
cursor and schedule. Continue from the full-state parent; do not reset warmup,
Adam steps or RNG. The fixture covers Engram memory gradients/state, frozen groups,
and cold interrupted resume with exact model/Adam/RNG/schedule/cursor agreement.

## Producer and operational limits

Only plain `local_text`/`all_tokens` is supported initially. Byte/portable memory
sidecars, allocation/release datasets, supervision masks and source-compatibility
overrides are rejected. The same preparation fingerprint is mandatory. A manifest
pin proves bytes, not cross-producer encoding equivalence.

Adding this producer API changes the conservative packing fingerprint. Caches from older
producers are **not** automatically compatible. First adoption needs one full
native preparation under the new producer; subsequent same-producer appends can
reuse that cache. Avoiding this one-time preparation would require a separately
reviewed explicit compatibility mechanism. No alias or override is provided here.

This removes historical tokenization, not all historical I/O: parent hashes,
raw-prefix authentication and complete output arrays still require streaming
reads/writes. Storage preflight reserves the full output/spool, tail array/spool,
validation and conservative raw-tail staging. Resource envelopes remain active
through encoding and bounded array copying. There is no measured GPU or real-corpus
speed claim, and this feature does not choose data, screen fresh families, alter
the mixture, manage services or authorize production adoption.

## Focused CPU validation

Fixtures establish exact fresh-preparation IDs, tail-only encoder calls,
cold/signed reuse, stale/corrupt/mutated historical inputs, cached wrong-parent
refusal, frozen boundaries, publication failure and exact cold full-state
continuation. Use the existing locked development environment from the checkout:

```sh
uv run --locked --no-sync pytest -n 2 -q \
  tests/test_prepared_append.py tests/test_prepared_prefix.py \
  tests/test_verifier_authority_current.py \
  tests/test_document_budget_extension.py
```

These tiny CPU fixtures verify implementation behavior, not real-corpus quality,
accelerator performance or permission to continue a production run.
