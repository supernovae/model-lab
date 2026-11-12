# Build a model from your own text

Commands below use the [activated, provisioned environment](../README.md#install).

This is the shortest path from a text corpus to a model you can sample from.
It uses five commands and no experiment, study or comparison vocabulary.
To test whether a change helps, continue with [lab mode](lab-mode.md).

## 1. Start from the example

Copy the maintained example out of the checkout, so outputs stay outside Git:

```sh
cp -r examples/build-local-text "$KLAB_WORK_DIR/my-model"
cd "$KLAB_WORK_DIR/my-model"
```

It holds two declarations and a tiny corpus:

| File | Purpose |
| --- | --- |
| `corpus/train.jsonl`, `corpus/validation.jsonl` | one `{"text": "..."}` document per line; the two files share no document |
| `tokenizer.yaml` | vocabulary size and the corpus to fit it on |
| `model.yaml` | model shape, data budget, training budget and backend |

Paths inside a declaration are relative to that file. Everything the commands
write goes under `work/` beside them.

## 2. Build

```sh
klab tokenizer train tokenizer.yaml
klab inspect model.yaml
klab data prepare model.yaml
klab train model.yaml --run-id first
klab generate first --runs-dir work/runs --prompt "The baker counts"
```

| Command | What it does |
| --- | --- |
| `tokenizer train` | fits a BPE vocabulary on the training file only |
| `inspect` | prints the parameter count and a memory estimate; trains nothing |
| `data prepare` | tokenizes both files once into a reusable [token store](token-store.md) and packs them |
| `train` | trains, evaluates on the validation file and writes checkpoints to `work/runs/first` |
| `generate` | samples a continuation from the latest verified checkpoint |

The example trains a 2-layer model for 200 steps in a few seconds on a CPU.
It shows that the path works; it is far too small to say anything about quality.

## 3. Use your own corpus

Replace the two corpus files, or point `train_path` and `validation_path` at
yours, then change these fields in both declarations:

- **`dataset.license`**: the terms your text is under. It is recorded with the
  prepared data and every run.
- **`train_max_documents`, `train_max_tokens`** and the validation pair: upper
  bounds on what is read. Raise them to cover your corpus.
- **`vocab_size`**: must match between `tokenizer.yaml` and `model.vocab_size`.
  The tokenizer refuses a size the corpus cannot fill.
- **`model.*` and `training.*`**: the shape and budget. Run `klab inspect`
  after each change; it reports whether the model fits before you train.
- **`runtime.backend`**: `cpu`, `cuda`, `rocm`, `xpu` or `mps`. See the
  [runtime guide](runtime.md) for accelerator setup.

Other inputs use the same five commands. `dataset.source` selects how data is
read: a [pinned snapshot](datasets.md#pin-a-dataset), a
[Hugging Face dataset](datasets.md#source-kinds), chat transcripts or a
[token mixture](token-store.md#mixture-output).

For a long accelerator run, start from `configs/accelerator_scale.yaml`
([keep accelerators busy](training.md#keep-accelerators-busy)). Other starting shapes are in `configs/`: `dense_7m.yaml` to `dense_50m.yaml`,
`first_model_20m_cpu.yaml` and `scale_dense_160m_cuda.yaml`. Copy one and
replace its `dataset` and `tokenizer` sections with yours.

To pick an architecture by name and size instead, scale a preset onto your
config: `klab architecture derive moe --params 300M --base model.yaml --output
moe-300m.yaml`. [Model components](components.md) lists the presets and covers
per-layer schedules and kinds supplied by another package.

## 4. Keep the model in its own repository

Lab is the engine; a model's declarations do not have to live in it. A project
is a directory of declarations (and, if you like, a corpus and notes) in any Git
repository. [Kybern Workshop](https://github.com/supernovae/kybern-workshop) is
one such repository; yours works the same way.

```sh
git clone https://github.com/you/your-models.git
cd your-models/story-model
klab tokenizer train tokenizer.yaml
klab train model.yaml --run-id first
```

- **Run from anywhere.** `klab` is a command of the installed Lab, not of a
  checkout. Paths inside a declaration resolve from that declaration's file, so
  a project keeps working after it is cloned, moved or copied.
- **Keep outputs out of Git.** Point `cache_dir`, the tokenizer `output_dir` and
  `logging.root_dir` at a `work/` directory you ignore, or at absolute paths
  under `$KLAB_WORK_DIR`. Lab-mode records go under `$KLAB_WORK_DIR/lab`.
- **Check before you run.** `klab config validate FILE --kind run` (or
  `tokenizer`, `model`, `optimizer`) checks a declaration against the current
  schema without reading data or probing hardware, and `klab inspect FILE`
  reports whether the model fits.
- **Share with ordinary Git.** Commit the declarations and push. Nothing in Lab
  names a repository, an organization or a project.
- **Bring your own parts.** A project that needs its own attention,
  feed-forward, memory or training objective ships a small Python package that
  registers it; see [model components](components.md) and
  [post-training](post-training.md#write-your-own-objective).

To validate, compose and derive declarations, see [configs](configs.md).
