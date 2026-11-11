# Fixed panels during training

`evaluation.panels` optionally declares one or more distinct named development
panels. When present, these panels replace the ordinary validation sample.
The trainer evaluates them using its already initialized engine at the existing
baseline, `evaluation.every_steps`, and terminal/stop evaluation boundaries.
It does not create additional training stops, model reloads or checkpoint
boundaries. Existing checkpoint and best-checkpoint decisions use the combined
loss.

Each entry uses the data-selection fields of the existing held-out panel
declaration, without a checkpoint selector:

```yaml
evaluation:
  every_steps: 256
  panels:
    - name: english
      prepared_path: panels/english
      prepared_manifest_sha256: <native prepared manifest digest>
      tokenizer_file_sha256: <training tokenizer file digest>
      tokenizer_canonical_sha256: <native canonical tokenizer digest>
      split: validation
      sampling: first_eligible_blocks
      sequence_length: 2048
      blocks: 128
      supervised_targets: 262144
      batch_size: 2
      max_forward_positions: 262144
    # Add separately authenticated panels with the same fields as needed.
```

Use native prepared bundles containing the fixed held-out material; the example
digests are placeholders, not runnable identities. The manifest authenticates
the arrays, source identity, supervision and tokenizer. Selection uses the first
declared number of eligible blocks and must score exactly `supervised_targets`.
Loss masks can make that count smaller than forward positions. Sequence length
must match training; the current hook supports dense PyTorch with no memory or
token n-gram memory, including context-key Engram. Byte/portable memory and
allocation sidecars are refused. `evaluation.max_batches` applies only to ordinary
validation. Fixed panels use their own complete block and batch declarations.
All panels must have distinct prepared manifests and validation arrays. An exact
alias of the authenticated training array is refused. These cheap identity
checks do not detect shared subsets, overlapping windows or common seed data;
construct all panels from the verified held-out split. A legitimate ordinary
validation-array alias is not itself training leakage.

The existing explicitly budgeted Lab workflow counts the sum of all panels'
batches through `evaluation.batch_limit`; no new campaign limit is added.
Separate probe/try scorers retain their own ordinary validation protocol.

Runs and materialized prepared-input stages own copies under `panels/<name>/`.
Their existing inventories bind all required native prepared files. Resume reads
those owned copies and the owned tokenizer, so caller-local originals may be
removed. Moving a path preserves identity; changing panel selection, content,
tokenizer or names changes the scientific continuation identity and is refused.
Prepare small panel bundles: copying a general large training cache also copies
its required training array, even though panel scoring reads validation only.

Every completed evaluation event includes named `panels` reports, exact scored
target counts, NLL sums, prepared-data identities and path-independent protocol
digests. Metrics include `validation/<name>/loss` and perplexity. The existing
checkpoint-bound evaluation report includes those same named results whenever
the trainer saves that evaluation boundary. Its top-level loss is
`sum(panel NLL) / sum(panel scored targets)`, and perplexity is the exponential of
that loss. The aggregate requires no additional forward pass. Aggregate `batch_size`
and `max_batches` are null because each panel carries its own batching protocol.
The existing held-out evidence reader checks named coverage, artifact identities
and weighting before accepting the report.

This authenticates fixed inputs and repeatable selection. It does not establish
scientific independence of underlying source seeds or replace corpus split and
provenance checks. CPU fixture tests do not qualify real GPU performance.
