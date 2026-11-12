# Hosted CUDA example

Declarations for running on a hosted NVIDIA GPU, used by the
[hosted environments guide](../../docs/hosted-environments.md). Copy
`source.yaml`, `tokenizer.yaml` and `run-t4.yaml` into an `inputs/` directory
inside a workspace outside this checkout; their relative paths then put the
snapshot, tokenizer, data cache and runs beside it.

- `source.yaml` pins a bounded TinyStories snapshot, and `tokenizer.yaml`
  trains a 2,048-token tokenizer on its train split.
- `run-t4.yaml` is six 256-wide dense layers with 4 heads and a 704-wide FFN,
  sequence length 128, microbatch 8 with no accumulation, 20 updates and 20,480
  targets, CUDA FP16 and SDPA attention, checkpointing and evaluating every five
  steps. It sets `training.deterministic: false`, so runs are not bitwise
  repeatable.
- `attention-matrix.yaml` compares SDPA with reference attention, and
  `batch-matrix.yaml` compares `(8, 1)` with `(4, 2)`. Each changes one thing.
  Accept a candidate only if it succeeds under 90% of observed VRAM, and keep
  every out-of-memory or unavailable-kernel result.
- `relay-drive.example.yaml` and `relay-s3.example.yaml` are relay profiles.
  Copy one outside the checkout and fill in the absolute rclone config paths.
  Credentials never go in a profile, declaration, log or notebook.
- `colab-fallback.ipynb` runs only the prepared attempt entry point, for when a
  Colab session must be driven by hand.
