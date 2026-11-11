# Hosted TinyStories T4 sample

This is a copyable operational acceptance sample, not research evidence or a
quality benchmark. Copy `source.yaml`, `tokenizer.yaml`, and `run-t4.yaml` into
an external task workspace's `inputs/` directory. Their relative paths then
place snapshots, tokenizer, data cache, and runs below that task workspace;
do not run them in this checkout.

`source.yaml` and `tokenizer.yaml` are byte-identical to the TinyStories
microlab declarations. They retain the pinned snapshot revision, bounded
selection, deduplication policy, and 2,048-token tokenizer identity. Prepare
those inputs on the local CPU before attaching a hosted worker.

For gcloud authorization, create user ADC once with
`gcloud auth application-default login`, then pass `colab --auth=adc` to the
public CLI and `--colab-auth adc` to Model Lab hosted/registration commands.
Later jobs refresh credentials noninteractively; do not export or log tokens.
Drive relay authorization is separate; no local Drive mount is required.

For Colab, generate `hosted notebook-cell --root /content/klab --output
ABSOLUTE_PATH --json` and run its fixed enrollment cell manually in a dedicated
idle notebook before inspection or setup. The sample does not create a session,
allocate a GPU, or establish that a requested accelerator is the observed one.

`run-t4.yaml` is the fixed acceptance declaration: six 256-wide dense layers,
4 heads, 704-wide FFN, sequence length 128, microbatch/accumulation `(8, 1)`,
20 updates / 20,480 targets, CUDA FP16, and opt-in SDPA. It explicitly sets
`training.deterministic: false` for CUDA performance; it is therefore not a
bitwise-reproducibility claim. Checkpoint and evaluation cadence are five steps.

The two matrices are separate one-factor pilots:

- `attention-matrix.yaml` compares the shipped SDPA declaration with reference
  attention, changing only `attention.implementation`.
- `batch-matrix.yaml` compares `(8, 1)` with `(4, 2)`, changing only the
  divisible microbatch/accumulation pair. Keep the checked-in sample at `(8, 1)`
  regardless of a pilot result.

Use only successful measured candidates at no more than 90% observed VRAM;
record targets/second, post-initialization step distribution, observed memory,
and every OOM or kernel-unavailable result. Neither matrix changes the sample
into an automatic tuning policy.

`relay-drive.example.yaml` and `relay-s3.example.yaml` are operational profiles.
Copy one outside the checkout and replace only its absolute host-local rclone
configuration references. Do not add credentials to a profile, declaration,
Git, logs, manifests, or a notebook.

See [`docs/hosted-environments.md`](../../../docs/hosted-environments.md) for
CPU preparation, provisioning, worker registration, collection, hardware gates,
and provider selection. The deliberately minimal
[`colab-fallback.ipynb`](colab-fallback.ipynb) invokes only the native prepared
attempt entrypoint when manual Colab execution is necessary.
