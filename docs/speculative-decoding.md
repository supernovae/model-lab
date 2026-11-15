# Speculative decoding

Greedy generation normally runs the model once per token. Speculative decoding
lets something cheap guess several tokens ahead, then has the model score the
whole guess in one cached call and keep the longest prefix it agrees with. The
completion is the model's own greedy completion; only the number of model calls
changes.

Lab ships three drafters:

- **Prompt lookup.** When the last N tokens also appeared earlier in the
  context, it proposes the tokens that followed that earlier copy. It needs no
  second model and no table, and it pays off on text that repeats its context:
  code edits, structured records, quoting and retrieval-style prompts.
- **Corpus n-gram table.** A bounded next-token table built from a run's
  training split proposes the continuation that most often followed the last N
  tokens in the corpus. It needs no second model, and it pays off when the
  model reproduces its training text. See [the table](#corpus-n-gram-table).
- **Draft model.** A smaller run trained with the same tokenizer proposes its
  own greedy continuation. It costs a forward of the small model per proposed
  token, so it pays off when the draft is much cheaper than the target and
  agrees with it often.

```sh
klab generate RUN_ID --prompt "$PROMPT" --max-new-tokens 256 \
  --prompt-lookup 3 --draft-tokens 8 --json
klab data ngram-table RUN_ID --output TABLE_DIR --order 3 --table-size 1048576
klab generate RUN_ID --prompt "$PROMPT" --max-new-tokens 256 \
  --ngram-table TABLE_DIR --draft-tokens 8 --json
klab generate RUN_ID --prompt "$PROMPT" --max-new-tokens 256 \
  --draft-run DRAFT_RUN_ID --draft-tokens 4 --json
```

`--prompt-lookup N` sets the n-gram length to match. `--draft-run` names the
draft run, loaded from the same `--runs-dir` and backend (`--draft-checkpoint`
picks a checkpoint; the default is its latest), and is refused if its tokenizer
differs from the target's. Use one drafter at a time. `--draft-tokens` is the
most tokens proposed per model call. The JSON result gains a `speculation`
object (without `--json` it is printed to stderr):

| Field | Meaning |
|---|---|
| `model_calls` | Cached forwards that fed a token together with a draft to verify. Encoding the prompt and re-encoding a full context window are not counted. Plain greedy decoding uses one forward per generated token. |
| `drafted_tokens` | Tokens proposed by the drafter. |
| `accepted_tokens` | Proposed tokens the model confirmed. Each one is a model call saved. |
| `wall_seconds` | Wall time of the decoding call, prompt encoding included. |
| `table_identity` | Corpus table only: the table's manifest digest, order, size and source. |
| `draft_model_calls` | Draft model only: forwards of the draft model. |
| `draft_identity` | Draft model only: the draft run, checkpoint and tokenizer identity. |

While the prompt and completion fit in the model's context, compare
`model_calls` with the completion length to see the effect on a prompt. A
speedup is the ratio of `wall_seconds` to the top-level `wall_seconds` of the
same command without a drafter, on the same device: a call that verifies a
draft processes more positions than a single-token call, and a draft model's
own forwards are not free. Time a few prompts and discard the first, which
includes warm-up.

Drafting stops once the context window is full. From then on each token
re-encodes the whole window in one uncounted forward and no draft is proposed,
so a prompt that already fills the context reports `model_calls: 0` and
`drafted_tokens: 0` however long the completion is. The tokens are still the
model's greedy tokens.

## Corpus n-gram table

`klab data ngram-table RUN_ID --output DIR` reads the run's prepared training
split (verifying it first), hashes every `--order` token context into
`--table-size` slots and keeps, per slot, the continuation seen most often and
its count (ties go to the smaller token id). Contexts that contain the
end-of-sequence token span two documents and are skipped. The table is two
fixed-size arrays, so its size is `8 * table-size` bytes however large the
corpus is; colliding contexts share a slot, which costs acceptance and never
correctness. `DIR` must not exist, and it is separate from the run so the run's
artifact inventory is unchanged.

The table directory has its own `manifest.json`: format, hash scheme, order,
size, end-of-sequence id, source (run, prepared-data manifest digest and
tokenizer digest), file digests and a digest of the manifest itself.
`--ngram-table DIR` verifies the signature, the exact file inventory, file
digests, array shapes and token ranges, and refuses a table built with a
different tokenizer than the target's. Changing the corpus, order, size or
tokenizer is a new table; nothing upgrades an old one. Drafting looks the last
`order` tokens up, appends the guess and looks up again, and stops at an empty
slot or after an end-of-sequence token. Building holds one count per distinct
(context slot, continuation) pair of the split in memory.

## Train a draft model

A draft model is an ordinary run trained on the same tokenizer and data as the
target. The `draft` architecture preset is a shallow, narrow dense decoder; scale
it to a fraction of the target and derive its config from the target's, so the
vocabulary and context length stay the target's:

```sh
klab architecture derive draft --params 15M \
  --base TARGET_CONFIG.yaml --output draft.yaml
klab train draft.yaml --run-id DRAFT_RUN_ID
klab generate TARGET_RUN_ID --prompt "$PROMPT" --draft-run DRAFT_RUN_ID --json
```

Only `model` and `attention` differ from the target's config, so the draft sees
the same data and tokenizer; `--draft-run` refuses a run whose tokenizer differs.
Pick `--params` from the cost of a draft forward against the target's: drafting
pays off when the draft costs a small fraction of a target forward and agrees
with it often. Measure acceptance and wall time rather than assuming either.

## Measure drafters

`klab speculation` decodes a fixed set of prompts greedily with and without each
drafter and publishes a sealed record, in the same lab store as probes and
benchmarks:

```sh
klab speculation TARGET_RUN_ID --backend cpu \
  --prompt-lookup 3 --ngram-table TABLE_DIR --draft-run DRAFT_RUN_ID \
  --draft-tokens 4 --max-new-tokens 64 --repeats 3
klab report SPECULATION_ID
```

Each option may be repeated to compare several drafters (prompt-lookup lengths,
tables, draft runs) in one record. The prompts are evenly spaced windows of the
run's held-out validation split (`--prompt-count`, `--prompt-tokens`), which a
table built from the training split has not seen, or a JSON list of strings
given with `--prompts FILE`. The record stores a digest of the prompts.

The first check is the failure criterion: every drafter's completion must equal
plain greedy decoding's on every prompt and repeat. A drafter that differs makes
the record `mismatch` and the command exit 1. Then, per drafter:

| Field | Meaning |
|---|---|
| `acceptance_rate` | `accepted_tokens / drafted_tokens` over all prompts. |
| `model_calls` | Cached forwards that verified a draft; see the table above. |
| `tokens_per_second` | Generated tokens over summed wall seconds. Each prompt's time is the median of `--repeats` decodes, after one untimed warm-up decode per decoder. |
| `speedup_vs_plain` | `tokens_per_second` over plain decoding's, on the same prompts, host and backend. A value below 1 is a slowdown. |

The record names the backend, device, precision, torch version, CPU and thread
count. Wall times are properties of that host, model size and prompt set; they
do not transfer to another device, so run the same command with
`--backend cuda`, `--runtime ID` or the backend you will serve on. Acceptance
rate and `model_calls` do not depend on the device. Prompts that fit the context
only: decoding past the model's context window re-encodes it and proposes no
drafts (see above).

## Scope

- Drafts run only while the prompt and completion fit in `max_seq_len`; see
  the note under the table.
- Greedy decoding only (`--temperature 0`), on the PyTorch KV-cache path: dense,
  grouped-query, sliding-window and MLA attention, with or without token n-gram
  memory. Block-sparse attention, byte-addressed memory, sampling and the MLX engine are refused.
- Tokens match plain greedy decoding. Verifying several positions in one call can
  differ from single-token calls in the last floating-point bits, the same as the
  KV cache against full-prefix decoding.
- A draft model runs on the PyTorch KV-cache path with its own cache, which
  it rewinds to where its previous call's context ended; its context is its
  own `max_seq_len`, and it stops drafting when that is full.
- From Python, pass `speculation=PromptLookupSpeculation(ngram=3)` or
  `speculation=DraftModelSpeculation(model, device)` to
  `klab.evaluation.generation.generate`. A drafter is any object with the
  `begin`/`draft` methods and counters of `klab.evaluation.speculation.Drafter`,
  which is the extension point for other drafters. From Python the corpus
  table is `CorpusNgramSpeculation(load_ngram_table(DIR, tokenizer_sha256=...))`
  from `klab.evaluation.ngram_table`.
- The loop keeps at most the tokens it asked for from any drafter and refuses a
  token outside the vocabulary, so a misbehaving drafter can only be slow.
  Drafts that run past an end-of-sequence token or the context limit are
  discarded the same way as wrong ones.
