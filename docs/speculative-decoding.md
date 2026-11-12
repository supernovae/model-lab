# Speculative decoding with n-gram drafts

Greedy generation normally runs the model once per token. Speculative decoding
lets something cheap guess several tokens ahead, then has the model score the
whole guess in one cached call and keep the longest prefix it agrees with. The
completion is the model's own greedy completion; only the number of model calls
changes.

Lab ships one drafter, **prompt lookup**: when the last N tokens also appeared
earlier in the context, it proposes the tokens that followed that earlier copy.
It needs no second model and no table, and it pays off on text that repeats its
context: code edits, structured records, quoting and retrieval-style prompts.

```sh
klab generate RUN_ID --prompt "$PROMPT" --max-new-tokens 256 \
  --prompt-lookup 3 --draft-tokens 8 --json
```

`--prompt-lookup N` sets the n-gram length to match and `--draft-tokens` the most
tokens proposed per model call. The JSON result gains a `speculation` object
(without `--json` it is printed to stderr):

| Field | Meaning |
|---|---|
| `model_calls` | Cached forwards that fed a token together with a draft to verify. Encoding the prompt and re-encoding a full context window are not counted. Plain greedy decoding uses one forward per generated token. |
| `drafted_tokens` | Tokens proposed by the drafter. |
| `accepted_tokens` | Proposed tokens the model confirmed. Each one is a model call saved. |

While the prompt and completion fit in the model's context, compare
`model_calls` with the completion length to see the effect on a prompt. Measure
wall time on the target device before claiming a speedup, because a call that
verifies a draft processes more positions than a single-token call.

Drafting stops once the context window is full. From then on each token
re-encodes the whole window in one uncounted forward and no draft is proposed,
so a prompt that already fills the context reports `model_calls: 0` and
`drafted_tokens: 0` however long the completion is. The tokens are still the
model's greedy tokens.

## Scope

- Drafts run only while the prompt and completion fit in `max_seq_len`; see
  the note under the table.
- Greedy decoding only (`--temperature 0`), on the PyTorch KV-cache path: dense,
  grouped-query, sliding-window and MLA attention, with or without token n-gram
  memory. Block-sparse attention, byte-addressed or portable memory, semantic
  queries, sampling and the MLX engine are refused.
- Tokens match plain greedy decoding. Verifying several positions in one call can
  differ from single-token calls in the last floating-point bits, the same as the
  KV cache against full-prefix decoding.
- From Python, pass `speculation=PromptLookupSpeculation(ngram=3)` to
  `klab.evaluation.generation.generate`. A drafter is any object with the same
  `begin`/`draft` methods and counters, which is the extension point for a
  corpus n-gram table or a small draft model.
