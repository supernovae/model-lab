# Raw UTF-8 byte addressing

Commands below use the [activated, provisioned environment](../README.md#first-run).

Optional `model.memory: byte` uses one causal address per input token in `train_byte_addresses.npy` and `validation_byte_addresses.npy`. Each address hashes the raw-byte suffix ending at that token's exact ByteLevel byte boundary; the manifest records `raw-utf8-suffix-v1`, table size, n-gram size, and array hashes. Packing version `contiguous-eos-v3` introduced the corrected boundary semantics. Current `contiguous-eos-v4` caches additionally bind source/tokenizer/settings identity, a manifest digest, and each array's dtype, shape, token-stream length, and file hash. Byte-address arrays must align with the corresponding token arrays. Older cache directories remain untouched.

No Unicode normalization, case folding, whitespace normalization, or escaping occurs. Equal raw byte spans yield equal pre-reduction hashes regardless of text segmentation; visually similar strings with distinct UTF-8 bytes do not. EOS receives address zero because it is a structural model token, not source text.

Packing and generation share the same reversible ByteLevel token-byte conversion. A multibyte character may span several tokens: each partial UTF-8 byte sequence contributes only bytes actually consumed so far, never replacement characters or future bytes from a character-end offset. The next model call receives aligned cropped ID/address suffixes. This applies to greedy and sampled generation, including chat.

The byte-memory table is local trainable state, checkpointed with the model. Byte addressing is not proof of retrieval quality, withheld-fact learning, or cross-tokenizer semantic transfer. Prepared-data caches bind tokenizer serialization, source-code identity, packing version, source contents where local, and array digests; changing a generator cannot silently reuse old packed data.

The [`byte-engram` lesson](research/lesson-paths.md) makes the prepared UTF-8 address path inspectable with an initialized probe. It does not train data or provide a second hashing scheme: training still requires explicit tokenizer and data preparation. A verified exported table can instead be attached through the [`portable-engram` lesson](research/lesson-paths.md), whose table dimensions come from the supplied package.

Current byte addressing uses `mix31-terminal-v2` when the table size shares a
factor with 257, including a 257-entry table; coprime sizes retain
`poly257-terminal-v1` addresses. The lesson still uses 263 entries, and its
recorded smoke observations below remain unchanged. Results trained with the
old degenerate addressing need a fresh baseline before comparison; see
[the hash-affected baseline instructions](first-model.md#re-baseline-the-hash-affected-configs).

## Verified smoke execution

The `byte-engram` smoke lesson was scaffolded and run with offline data. After explicitly training its tokenizer, preparing data, and probing an initialized model, `café` mapped to `[189, 119, 246, 37, 94]` in the 263-entry table: five distinct nonzero slots. The prepared training and validation caches also used table size 263; their address arrays contained 111 and 104 distinct slots, with 26,779 and 799 nonzero positions, respectively. Training completed two optimizer steps with `--stop-after-step 2`. This verifies address diversity, cache preparation, and an actual update—not retrieval quality.

```sh
klab data prepare configs/smoke_byte_memory_cpu.yaml
klab train --runs-dir klab-work/runs configs/smoke_byte_memory_cpu.yaml --run-id byte-memory
klab eval byte-memory --runs-dir klab-work/runs
klab generate byte-memory --prompt "Once upon a time" --max-new-tokens 24 --runs-dir klab-work/runs
```
