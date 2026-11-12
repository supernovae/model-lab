# Configs

Lab reads YAML or JSON declarations: run configs, tokenizer configs, and model
and optimizer sections. These commands validate, assemble and vary them without
loading a model, data or runtime, so a model repository (for example in
[Kybern Workshop](https://github.com/supernovae/kybern-workshop)) needs no
scripts of its own.

## Validate

Validate one section, or every declaration under a directory, without loading
a model, data or runtime:

```sh
klab config validate model.json --kind model --json
klab config validate optimizer.yaml --kind optimizer --json
klab config validate run.yaml --kind run --json
klab config validate tokenizer.yaml --kind tokenizer --json
klab config validate path/to/project
```

Unknown fields, malformed YAML or JSON and duplicate keys are refused. For a
directory, every complete run, tokenizer, model and optimizer declaration is
checked, other documents are counted as skipped, and the exit status is nonzero
if any fails. This is the check a recipe repository runs by hand or in CI. A
valid section says nothing about artifacts, hardware or whether sections fit
together; `klab inspect CONFIG` checks a full run config.

## Compose

Assemble a run config from parts:

```sh
klab config compose run.yaml model=model.json optimizer=optimizer.json \
  local-paths.yaml --output "$WORK/run.yaml"
```

Parts merge in order, later parts winning, and `FIELD=PATH` places a file at a
dotted field. Keep a model section, an optimizer section and a file of local
paths (tokenizer, cache, run store) outside Git and compose them. Relative paths
resolve from the part that declares them. The result must be a complete run
config and is written beside a `.derivation.json` receipt with each part's
digest.

## Derive

Change single fields of a complete config:

```sh
klab config derive configs/smoke_cpu.yaml \
  --set optimizer.peak=0.001 --set training.micro_batch_size=2 \
  --output "$WORK/smoke-peak.yaml" --json
```

Each `--set` is strict JSON, so strings are quoted. Paths in assignments resolve
from the source config's directory. The output and its `.derivation.json` (the
requested assignments and the observed change) must not exist. Deriving never
adjusts other settings: changing a batch size does not change the token budget.
