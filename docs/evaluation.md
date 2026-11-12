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
