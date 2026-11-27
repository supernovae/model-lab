# Evaluation

Lab scores a checkpoint by its loss on held-out data it never trained on. This
page covers scoring a saved checkpoint, and fixed evaluation panels during and
after training. For behavioral checks and benchmarks see [probes](probes.md).

## Score a checkpoint

```sh
klab eval RUN_ID [--runs-dir RUNS] [--checkpoint GENERATION]
```

`eval` loads the run's verified checkpoint (latest by default) and reports
held-out loss and perplexity on its validation split, using the run's
evaluation settings. Scoring runs at the run's own `runtime.precision`, as its
in-training validation does: a BF16 run is scored under BF16 autocast and the
loss is reduced in FP32. At a long window this decides whether scoring fits at
all, because dense attention over 16,384 tokens holds `heads × 16,384²` scores:
24 GiB in FP32 for 24 heads, 12 GiB in BF16. `eval`, panels, `try` and `probe`
all score this way. Panel results record `scoring_precision`.

### Score declared answer choices

Use a JSON ranking declaration with a name and one or more items. Each item
has a unique ID, a category, a prefix, two to sixteen distinct candidates,
and the exact answer from those candidates.

```json
{
  "schema_version": 1,
  "name": "routes-development",
  "items": [
    {
      "id": "two-hop-1",
      "category": "composition",
      "prefix": "The inlet feeds the pool. The pool feeds the drain. Two links after the inlet is the",
      "candidates": ["drain", "pool", "inlet"],
      "answer": "drain"
    }
  ]
}
```

```sh
klab eval RUN_ID --ranking ranking.json
```

Lab uses its PyTorch probe scorer to average continuation log probability
per token, with one space between the prefix and each candidate. Tied top
choices split credit. Items must fit the checkpoint context and preserve
the prefix's token boundary. Results include each choice score, chance
accuracy, overall and category accuracy, the declaration's SHA-256, and
the checkpoint and tokenizer identity. Compare results from the same
declaration and tokenizer. The result is saved alongside the run's
ordinary evaluation results. Use separate development and final
declarations and keep final items outside model selection. The study owns
the answer keys and split assignments.

To test whether changing the context changes the answer, optionally add
`pairs` to the same declaration:

```json
"pairs": [{"id": "route-switch", "left": "left-rule", "right": "right-rule", "question": "Which route is open?"}]
```

`left` and `right` name two unused items. They must share a category, the same
ordered choices, and the exact `question` appended after a newline in each
prefix. The context before that newline must differ, and the correct choices
must differ. The study decides the change to make in the context and supplies
the answer keys.

The `ranking.pairs` result reports joint accuracy and mean answer margin overall
and by category. Joint credit multiplies the two item credits, so a fixed
unique answer earns zero and uniform binary choices earn 0.25. The answer
margin averages how much each context favors its correct choice over the
other context's correct choice, in mean log probability per token. It is zero
when the relative scores are unchanged. Lab computes both measures from the
existing item scores without running the model again.

### Score generated answers

Use a JSON declaration when a simulator, reference dataset, or formal system
supplies an exact answer. Each item supplies its own prompt and a unique ID.

```json
{
  "schema_version": 1,
  "name": "transfer-development",
  "max_new_tokens": 8,
  "items": [
    {
      "id": "remaining-1",
      "category": "mechanism",
      "prompt": "Tank A starts with 8 units and sends 3. How many units remain? Answer with an integer:",
      "kind": "integer",
      "accepted": ["5"]
    }
  ]
}
```

```sh
klab eval RUN_ID --answers answers.json
```

Lab greedily decodes each prompt using the selected checkpoint and stops at
the first line feed or end-of-sequence token. Prompts and the full completion
budget must fit the checkpoint context. The declaration supports up to 2,000
items and 1–32 new tokens per item. For a completed answer line, Lab trims
surrounding whitespace. `integer` accepts ASCII digits with an optional leading
sign and ignores leading zeros. `yes_no` accepts either word in any case with
an optional final period. `text` matches a whole line case-sensitively against
the supplied aliases. A response that reaches the token cap without a stop,
an empty response, or an invalid answer form earns zero credit.

The run's evaluation artifact includes individual responses, counts of valid
and correct answers, token and forward-position counts, and breakdowns by
category and answer kind. It pins the declaration, checkpoint, and tokenizer.
Lab verifies the weights and tokenizer and checks the sizes of the unused
training and validation payloads. Use a fixed prompt and decoding budget when
comparing checkpoints. This score measures the requested answer format on the
declared tasks. Keep separate development and final families.

### Bits per byte

Loss per token depends on the tokenizer: a larger vocabulary makes tokens
longer and the loss per token higher, without the model being any worse. Bits
per byte does not, so it compares models with different tokenizers. Set
`evaluation.text_documents` to score that many held-out documents this way:

```yaml
evaluation:
  text_documents: 1000
```

`eval` then also reports `text`: the documents, their UTF-8 bytes, the targets
scored, the total loss in nats, `bits_per_byte` (nats ÷ (ln 2 × bytes)) and
`text_sha256`. The documents are the first ones of the run's prepared
validation split, cut at each `<eos>` and decoded; Lab's ByteLevel BPE has no
normalizer, so decoding gives back the source text exactly, and two runs with
different tokenizers that report the same `text_sha256` scored the same text.
Each document is scored on its own after one `<eos>`, every token once, longer
documents in windows of the model's `max_seq_len`; the end of the document is
not a target. Corpus mixtures can use this score when all validation tokens
are supervised. Masked chat and preference validation is refused.
Prepare enough validation documents for every tokenizer you compare: a token
cap that cuts the split at a different document changes the digest.

### Gradient bottleneck

The output head maps a hidden state to the vocabulary, so the gradient the rest
of the network receives keeps only the part of the logit gradient that lies in
the head's column space: at most as many directions as the head's rank, out of
the vocabulary's. Set `evaluation.gradient_bottleneck: true` to measure how
much is lost on the same held-out targets the loss is scored on (Godey and
Artzi, [Lost in Backpropagation](https://arxiv.org/abs/2603.10145), section
3.3). `eval` then also reports `gradient_bottleneck`:

| Field | Meaning |
| --- | --- |
| `kernel_norm_fraction` | Share of the logit gradient's Frobenius norm in the kernel of the head's transpose: lost before it reaches the network |
| `kept_cosine` | Cosine between the logit gradient and the part that is kept |
| `kept_squared_fraction` | `kept_cosine` squared |
| `head_rank`, `vocab_size` | Rank of `output.weight` (the head's last factor with [`output_head_rank`](architecture.md#output-head-rank)) and the vocabulary |
| `random_kept_squared_fraction` | `head_rank / vocab_size`: what a gradient pointing in a random direction would keep |

The logit gradient of each target is `softmax(logits) - onehot(target)`. A
`try` whose baseline sets the field measures both arms, and a sweep reports the
candidates' mean `kernel_norm_fraction` per point.

A training run with the field set also measures it at each evaluation, on the
same held-out targets as the loss, and records `kernel_norm_fraction`,
`kept_cosine` and `kept_squared_fraction` as
`validation/gradient_bottleneck/...` beside `validation/loss`. The dashboard's
Evaluation tab draws them against the step, marks saved checkpoints and shows
each metric's change since the previous evaluation, so a bend in training shows
as a spike. The measurement needs the full logits: it refuses a run that uses
fixed validation panels, `runtime.output_head_chunk_tokens`, a custom objective
or the MLX engine.

It costs a short SVD of the head plus a softmax over the whole vocabulary for
every held-out target. On one Arc Pro B70 (a 42M model, 32,768-token vocabulary,
1,048,576 targets per evaluation) the SVD took 0.35 s and the softmax pass 5.3 s,
against a few seconds for the evaluation itself. Set
`evaluation.gradient_bottleneck_every: N` to measure on every Nth evaluation
(the first included), or leave `gradient_bottleneck` off during training and
`eval` the kept checkpoints afterwards, on any idle device.

## Panels

A panel is a fixed, named slice of prepared held-out data: for example one panel
of English prose and one of code, scored separately. A panel declaration pins
the prepared data, the tokenizer and exactly which blocks are scored:

```json
{
  "schema_version": 1,
  "name": "english",
  "purpose": "development",
  "prepared_path": "../prepared/english",
  "prepared_manifest_sha256": "<manifest_sha256 field of the prepared manifest>",
  "tokenizer_file_sha256": "<tokenizer.json file digest>",
  "tokenizer_canonical_sha256": "<prepared tokenizer identity>",
  "checkpoint_sha256": "<checkpoint manifest identity>",
  "split": "validation",
  "sampling": "first_eligible_blocks",
  "sequence_length": 2048,
  "blocks": 128,
  "supervised_targets": 262144,
  "batch_size": 2,
  "max_forward_positions": 262144
}
```

- **Data.** `prepared_path` (relative to the declaration) must already be
  prepared; Lab verifies it but never prepares it here. Only its validation
  member is scored, and `purpose` must be `development`.
- **Selection.** The first `blocks` full blocks with at least one supervised
  target, in stored order. The scored target count must equal
  `supervised_targets` exactly; a short or empty panel is incomplete, never a
  smaller success.
- **Bounds.** `blocks × sequence_length` must fit `max_forward_positions`.

### After training

Score one checkpoint on each panel:

```sh
klab eval RUN_ID --runs-dir "$RUNS" --checkpoint step_00013042_gen_000001 \
  --panel english.json --output "$WORK/results/english-step13042.json"
```

The declaration's `checkpoint_sha256` must match the checkpoint. The output must
be a new file in an existing directory, outside the panel and run directories.
It records the panel and checkpoint identities, committed training tokens, the
loss and its parts, and timing. The command exits zero only for complete
coverage; an interrupted evaluation keeps what it scored and leaves `loss`
null.

Two results are comparable when both are complete and their `comparison_group`
values match. The group covers the panel protocol and runtime but not the
checkpoint, so one panel can follow a run's trajectory.

### Gradient alignment between panels

Pin both development panels to the same checkpoint, then compare the
gradients of their mean next-token losses:

```sh
klab eval RUN_ID --checkpoint GENERATION --panel software.json \
  --gradient-against astronomy.json --output alignment.json
```

Lab loads the checkpoint once and makes one forward and backward pass per
panel batch on CPU in FP32. Each panel gradient is normalized by the actual
supervised target count. The result records panel coverage, gradient norms,
dot product and cosine. A zero norm leaves the cosine undefined. A negative
dot product means a small step down one raw gradient would raise the other
panel's loss to first order at this checkpoint. Adam updates and changes over
a training trajectory are separate measurements.

The diagnostic supports dense next-token PyTorch checkpoints with no memory
or token n-gram memory. The output is a new file outside the input roots.
The command exits nonzero for incomplete measurements and retains the status
and reason. The default gradient storage cap is 64 MiB for the two gradients.
Set `--gradient-max-bytes` to choose another cap. Model, activation and logits
storage are additional. Each panel retains its own batch size and forward
position bound.

### Loss by position

A complete panel result also reports `position_buckets`: the same scored
targets grouped by their position in the window, each bucket with its `start`
and `end` positions, `targets` and mean `loss`. A target at position `t` is
predicted from `t + 1` tokens of context. Buckets double in width up to the
panel's `sequence_length`. The first covers 0 up to 1,024, or up to the largest
power of two below `sequence_length` for a shorter panel, so a 32,768-token
panel reports `0-1024, 1024-2048, 2048-4096, 4096-8192, 8192-16384,
16384-32768`. Bucket target counts add up to `supervised_targets`, and their
target-weighted mean loss is the panel loss.

This is the standard context-extension curve: with a working extension,
late-position loss stays at or below early-position loss, while a model used
past the context it learned shows loss rising in the late buckets. Buckets are
reporting only. They do not change the panel declaration, its identity or
which targets are scored, so earlier panel results stay comparable. Panels
scored during training report them too, unless the run's training objective
scores whole batches rather than single targets; then `position_buckets` is
null and `position_buckets_unavailable` says why.

### During training

`evaluation.panels` replaces the ordinary validation sample with one or more
panels, scored at every evaluation point:

```yaml
evaluation:
  every_steps: 256
  panels:
    - name: english
      prepared_path: panels/english
      prepared_manifest_sha256: <digest>
      tokenizer_file_sha256: <digest>
      tokenizer_canonical_sha256: <digest>
      split: validation
      sampling: first_eligible_blocks
      sequence_length: 2048
      blocks: 128
      supervised_targets: 262144
      batch_size: 2
      max_forward_positions: 262144
```

Entries use the same fields without `checkpoint_sha256`, and `sequence_length`
must match training. Each panel logs `validation/<name>/loss` and perplexity, and
checkpoints and the best checkpoint use the combined loss: total loss over total
targets. `evaluation.max_batches` applies only to the ordinary sample.

A run copies its panels under `panels/<name>/`, so resuming does not need the
originals. Panels must have distinct data, and one that is the training array
itself is refused; Lab cannot detect partial overlap, so build panels from the
held-out split.

Panels work with dense PyTorch models without memory or with token n-gram
memory.

## Comparing context extensions

A long-document panel scores held-out text at the extended window. Prepare it
like any run's data, with the extension run's tokenizer and
`training.seq_len` set to the panel's `sequence_length` (for example 32768).
Use held-out documents at least that many tokens long as `validation_path` of
a `local_text` dataset, so late positions read context from the same
document. `data prepare` packs documents end to end, so shorter ones would put
document boundaries inside a window:

```sh
PREPARED=$(klab data prepare long-panel.yaml)
klab data coverage --prepared-root "$PREPARED" --config long-panel.yaml --json
```

Coverage reports the complete validation blocks and supervised targets; put
them in the panel declaration as `blocks` and `supervised_targets`, and point
`prepared_path` at the prepared directory.

To choose an extension method, pretrain one parent at short rows, then run two
`klab try` arms from the same `training.parent_checkpoint` that differ only in
their RoPE setting, with the same long-document data and row length (for
example `klab try yarn.yaml --vs abf.yaml`). Score each arm's checkpoint on the
long panel with `klab eval --panel` and compare their `position_buckets`. Then
run `klab probe TRY_ID --vs BASELINE --config full`, which adds the
[`long_needle` and `kv_recall`](probes.md#long-context-needle) retrieval
surfaces, by length and needle depth, to the short-context probes that catch
regressions at the pretraining window.

Choose on retrieval and on loss by position. Aggregate held-out loss is a
guardrail here: require that it stays within a margin you state beforehand,
and do not pick the arm with the lower loss for that reason alone. Loss
averages over every position and is dominated by local prediction, so an arm
can have the lower loss and still fail to use distant context (Gao et al.
2025, "How to Train Long-Context Language Models (Effectively)", report the
same of perplexity). Two arms that differ by a fraction of a percent in loss
and by chance against perfect in a retrieval cell are separated by the cell.
