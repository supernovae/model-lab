# Latent Attention

The MLA path compresses the joint key/value source into a smaller latent vector, then projects keys and values for attention. Its request-local decoding cache retains latent K/V state and explicit positional state. This is a readable reference implementation, not a fused bandwidth-optimized kernel or a claim of DeepSeek-equivalent inference performance.

Set `attention.implementation: sdpa` to use fused scaled-dot-product attention, which avoids a `[T, T]` weight tensor where the backend has a fused kernel. To combine latent keys and values with sparse selection at long context, use [`local_selected` with `latent_dim`](sparse-attention.md#local-plus-selected-blocks).

Use `configs/smoke_mla_cpu.yaml` for a bounded CPU smoke. Compare dense, MLA, sparse selection, and combined variants at matching data, tokenizer, and token budget.

For the standalone configuration and shape probe, see the [`mla` lesson](research/lesson-paths.md). The current cache retains expanded keys and latent values; cache claims must be calculated from that layout, not inferred as a fused compressed-cache implementation. The [MLA/Engram study](research/engram-mla-compression.md) reports declared endpoint contrasts only, not paper-equivalent compression or speed.
