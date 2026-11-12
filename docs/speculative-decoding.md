# Speculative decoding

Greedy generation normally runs the model once per token. Speculative decoding
lets something cheap guess several tokens ahead, then has the model score the
whole guess in one cached call and keep the longest prefix it agrees with. The
completion is the model's own greedy completion; only the number of model calls
changes.

Lab ships two drafters:

- **Prompt lookup.** When the last N tokens also appeared earlier in the
  context, it proposes the tokens that followed that earlier copy. It needs no
  second model and no table, and it pays off on text that repeats its context:
  code edits, structured records, quoting and retrieval-style prompts.
- **Draft model.** A smaller run trained with the same tokenizer proposes its
  own greedy continuation. It costs a forward of the small model per proposed
  token, so it pays off when the draft is much cheaper than the target and
  agrees with it often.

```sh
klab generate RUN_ID --prompt "$PROMPT" --max-new-tokens 256 \
  --prompt-lookup 3 --draft-tokens 8 --json
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
- A draft model runs on the PyTorch KV-cache path with its own cache, which
  it rewinds to where its previous call's context ended; its context is its
  own `max_seq_len`, and it stops drafting when that is full.
- From Python, pass `speculation=PromptLookupSpeculation(ngram=3)` or
  `speculation=DraftModelSpeculation(model, device)` to
  `klab.evaluation.generation.generate`. A drafter is any object with the
  `begin`/`draft` methods and counters of `klab.evaluation.speculation.Drafter`,
  which is the extension point for other drafters such as a corpus n-gram
  table.
