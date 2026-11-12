# Build a model from local text

A tokenizer declaration, a model declaration and a tiny CC0 corpus written for
this repository. The [build guide](../../docs/build.md) walks through it.

Copy the directory out of the checkout before running it, so `work/` is created
outside Git:

```sh
cp -r examples/build-local-text "$KLAB_WORK_DIR/my-model"
cd "$KLAB_WORK_DIR/my-model"
klab tokenizer train tokenizer.yaml
klab inspect model.yaml
klab data prepare model.yaml
klab train model.yaml --run-id first
klab generate first --runs-dir work/runs --prompt "The baker counts"
```

The corpus is 600 training and 80 held-out sentences built from a small word
grid. It proves the path works and says nothing about model quality.
