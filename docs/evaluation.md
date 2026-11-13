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
evaluation settings. Scoring runs in FP32 even for a model trained in BF16.

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
not a target. Chat and preference data, whose targets are masked, are refused.
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
candidates' mean `kernel_norm_fraction` per point. To follow it over training,
keep periodic checkpoints and `eval` each one.

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
run `klab probe TRY_ID --vs BASELINE --config full`, which adds
[`long_needle`](probes.md#long-context-needle) retrieval at each length to the
short-context probes that catch regressions at the pretraining window.
