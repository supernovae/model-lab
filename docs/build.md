# Build a model from your own text

Commands below use the [activated, provisioned environment](../README.md#first-run).

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
read: a [pinned snapshot](datasets.md#source-declaration-and-lock), a
[Hugging Face dataset](datasets.md#source-kinds), chat transcripts or a
[token mixture](token-store.md#mixture-output).

For a long accelerator run, start from `configs/accelerator_scale.yaml`
([keep accelerators busy](training.md#keep-accelerators-busy)). Other starting shapes are in `configs/`: `dense_7m.yaml` to `dense_50m.yaml`,
`first_model_20m_cpu.yaml` and `scale_dense_160m_cuda.yaml`. Copy one and
replace its `dataset` and `tokenizer` sections with yours.
