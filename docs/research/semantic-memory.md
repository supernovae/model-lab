# Verified semantic memory and exact retrieval

The `semantic-retrieval` lesson teaches a bounded, verified retrieval interface. It is not a language-understanding lesson: a caller supplies **externally encoded canonical vectors**, already in the declared key space. Model Lab does not turn a prompt into those vectors, parse natural language, or infer that a vector match is semantically correct.

Scaffold and verify the tutorial pack through the lab:

```sh
klab learn scaffold semantic-retrieval --output experiments/semantic-retrieval
klab engram pack inspect experiments/semantic-retrieval/semantic-pack
klab engram pack verify experiments/semantic-retrieval/semantic-pack
```

The scaffold materializes the tutorial pack, semantic assets, query fixtures,
an API demonstration, a ready-to-run `probe.yaml`, and a hash-bound `lesson.json`.
Run the ordinary supplied-vector diagnostic with:

```sh
klab semantic probe experiments/semantic-retrieval/probe.yaml --json
```

The probe authenticates the pack before one CPU inference-only forward. Its lesson
declaration exercises the atlas hit, zero-vector unknown, tied fork conflict,
time-bounded harbor hit, vault temporal miss, and a masked position. It uses the
declared canonical key map; do not invent an encoder identity or treat its fixed
vectors as natural-language embeddings. A records-only EngramPack is valid as an
artifact, but is not executable semantic retrieval.

## Verified pack boundary

An [EngramPack](../portable-engram.md#engram-packs) can carry a semantic component only when it inventories all of:

- canonical `records.jsonl` records;
- `semantic_keys.safetensors` with finite FP32 keys `[M,K]`;
- `semantic_values.safetensors` with finite FP32 values `[M,V]`; and
- `semantic.json`, which orders the `M` record IDs and declares key/value encoder identities and whether scoring is raw dot product (`none`) or cosine (`l2`).

`SemanticRetriever.from_pack` first verifies the pack and its expected identity, then enforces entry, asset-byte, exact-comparison, and worst-case result-byte bounds before using arrays. `top_k` is capped at 16. The pack binds `K` (key width) and `V` (value width); neither is the model hidden width `D`. A verified pack ID establishes byte identity and integrity, not source ownership, licensing sufficiency, factual truth, encoder quality, retrieval quality, answer quality, or performance.

```sh
klab engram pack inspect experiments/semantic-retrieval/semantic-pack
klab engram pack verify experiments/semantic-retrieval/semantic-pack
```

`inspect` reads the manifest and reports that payloads are not yet verified. `verify` checks identity and payloads before loading. See `src/klab/engram/packs.py:EngramPack`, `verify_pack`, and `src/klab/engram/semantic.py:SemanticRetriever.from_pack`.

## Query, retrieval, and trace shapes

A semantic query is a `SemanticQueryBatch`: an encoder identity, precomputed vectors, and an optional mask. Its identity and width must exactly match the attached pack's key encoder. DenseLM accepts either one query batch or, for independently encoded multipack attachments, a mapping keyed by the key encoder's SHA-256 identity.

| Item | Shape | Meaning |
| --- | --- | --- |
| Semantic keys | `[M,K]` | One externally produced canonical key per represented record. |
| Semantic values | `[M,V]` | Retrieved value representation, independent of key and backbone widths. |
| Batch query | `[B,K]` | One query per batch item, broadcast across sequence positions. |
| Position query | `[B,T,K]` | One query at each batch/sequence position. |
| Mask | `[B]` or `[B,T]` | Matching batch or batch/position validity layout. |
| Hidden state | `[B,T,D]` | Backbone state at the adapter's owned injection site. |
| Retrieved value | `[B,T,V]` | Value selected or broadcast by retrieval before projection. |
| Adapter residual | `[B,T,D]` | Projected value mixed through a sigmoid gate. |

`SemanticRetriever.retrieve` compares a query exactly: raw dot product for `key_normalization: "none"`, cosine for `"l2"`. Scores sort descending; equal scores sort by ascending `record_id`. `top_k` is bounded to `[1,16]`. The trace exposes comparison/candidate counts, best score/record, full tie count, and at most 16 ordered tied IDs; returned hits contain only the requested top-k rows.

- **`hit`**: at least one valid candidate meets the optional minimum score;
- **`unknown`**: no valid candidate reaches that threshold;
- **`temporal_miss`**: a matching record reaches the threshold but is excluded by the inclusive `as_of` validity filter; and
- **`conflict`**: multiple valid records share the best score. Their top-k hits and bounded tie trace remain deterministic.

`last_traces` on the adapter retains at most the most recent 64 query traces. Inspect retrieval status, ordered hit IDs/scores, and tie metadata per query. These are mechanism observations, not a score or answer-quality claim. Sources: `src/klab/engram/semantic.py:SemanticQueryBatch`, `SemanticRetriever.retrieve`, and `SemanticRetrievalOutcome`.

## Native supplied-vector probe

`klab semantic probe DECLARATION.yaml [--json]` is a read-only CPU
mechanism diagnostic. The declaration has format
`"klab-semantic-probe-v1"` and names an initialized model or an
authenticated native checkpoint, rectangular token IDs, explicitly supplied
vectors, and one or more verified pack attachments. It never encodes text,
creates a run, trains a model, or chooses a retrieval threshold on the
caller’s behalf.

Every attachment names its pack ID, full encoder identity, query, injection
site, and either explicitly seeded initialized adapter weights or an already
restored checkpoint adapter. The report binds canonical FP32 vectors, masks,
tokens, time metadata, packs, model state, and adapter-weight origins. It
reports native per-position retrieval traces, while masked positions are
explicitly null—not inferred misses. These traces remain descriptive mechanism
observations, not a quality, factuality, or policy verdict.

## Frozen assets, trainable adapter, and explicit ownership

`SemanticMemoryAdapter` keeps the verified retriever outside the model `state_dict`. Retrieved keys and values are fixed, non-trainable assets; the adapter has a trainable `V → D` output projection and sigmoid gate. `replace_retriever()` accepts only a verified pack with the same key/value identities, dimensions, and representation-space contract, preserving adapter parameters. Frozen attachment comparisons freeze all model parameters and use inference mode.

Each attachment has a unique name and an explicitly owned site:

- `embedding`, at the embedding residual;
- `after_block`, at one zero-based decoder block index; or
- `final`, after final normalization.

These are semantic adapter sites, not aliases for lexical-memory placement. Multiple adapters can share a site or independently own hybrid multipack sites. A multipack request uses a SHA-256-keyed query mapping when key spaces differ. Missing/mismatched queries fail closed. Full-prefix and cached forwards agree when the aligned query trajectories are equivalent.

This is a direct PyTorch `DenseLM` API. The standard `klab train` and generation CLI paths do not construct semantic query vectors or load a text encoder. Position-shaped query batches follow context truncation and repeat their final vector, mask, and `as_of` value across generated continuation tokens. Trainable adapters can be optimized through ordinary `DenseLM` forward calls, but the standard trainer does not currently construct semantic batches. Observe attachment name/site/block index, per-attachment traces, gate, and lookup diagnostics. Sources: `src/klab/model/transformer.py:DenseLM.add_semantic_memory` and `src/klab/engram/semantic.py:SemanticMemoryAdapter`.

## Toy worlds are partitioned retrieval evidence

Toy worlds use deterministic structured-key and value encoders with declared identities. Their vectors are canonical synthetic keys, not text embeddings. Producer facts are separate from query/answer artifacts. Each training and held-out world has an isolated pack containing only that world's producers. Worlds share a canonical start key but change intermediate and terminal assignment values; a held-out query is also evaluated against the other held-out world's pack to check that retrieval follows the attached assignment. Wording templates are partitioned, so held-out wording checks a split boundary only—it is **not evidence of natural-language understanding**.

A multi-hop case stores one-edge records and derives each next key from the prior retrieved value; there is no direct start-to-final shortcut. Evaluation includes conflict, time-bounded, and unknown queries. Compare correct, disabled, random, conflicting, and incomplete verified packs; no parameter gradient is computed by exact retrieval. The standalone demo freezes its attached DenseLM and adapter before inference. Keep per-hop records, statuses, scores, pack identity, candidate/tie information, and adapter diagnostics. These observations establish only bounded synthetic retrieval behavior, not open-world facts, language understanding, learned transfer, or production quality. See `src/klab/data/toy_worlds.py`.

### Knowledge-swap ratio

Report knowledge-swap ratio separately for every task:

$$
\text{knowledge-swap ratio} =
\frac{\#\{\text{changed-assignment queries whose retrieval follows the attached pack}\}}
{\#\{\text{eligible changed-assignment queries}\}}.
$$

The benchmark swaps each held-out query onto the other held-out world's correct pack and compares the retrieved task result with that attachment's own assignment. It is not pooled across tasks, worlds, or controls. If no changed-assignment query is eligible, the denominator is zero: report `null` with its explicit reason, never `0`, `1`, or an inferred success/failure rate.

## What this lesson does and does not support

The lesson supports verified pack loading, exact bounded score ordering, deterministic ties, temporal/threshold statuses, tensor compatibility, frozen attachment state, and traceable adapter placement. It does **not** ship a natural-language text encoder, natural-language retrieval benchmark, language-understanding test, approximate-nearest-neighbor index, general-purpose semantic pack producer, standard-trainer query pipeline, or speed/quality claim.

For artifact provenance and semantic component layout, see [Portable Engram](../portable-engram.md). For the existing trainable token/byte tables, which are distinct from external semantic retrieval, see [Engram](../engram.md).

The gated teacher-representation compiler and its measured-cost requirements are described in the [research roadmap](roadmap.md).

## Primary context

- Karpukhin et al., [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906). It motivates vector retrieval research, but this lesson neither supplies DPR's text encoders nor reproduces its open-domain QA protocol or results.
- Weston, Chopra, and Bordes, [*Memory Networks*](https://arxiv.org/abs/1410.3916). It motivates explicit memory interfaces, but this lesson only exposes its declared verified-pack retrieval and toy-world controls; it does not reproduce Memory Networks results.
