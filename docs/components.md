# Model components, per-layer schedules and architecture presets

A decoder is built from three kinds of part: an **attention** and a
**feed-forward** in every block, and an optional **memory** on the model. Each
is looked up by kind in one registry (`klab.components`), so a new kind is a
module and a registration, not a change to the block or the model.

| Slot | Config field | Kinds Lab ships |
| --- | --- | --- |
| attention | `attention.kind` | `dense`, `sliding_window`, `mla`, `block_sparse`, `local_selected` |
| feed-forward | `model.ffn` | `dense`, `moe` |
| memory | `model.memory` | `none`, `ngram`, `byte`, `portable` |

## Choose an architecture by name and size

```sh
klab architecture list
klab architecture derive moe --params 300M \
  --base configs/accelerator_scale.yaml --output my-model.yaml
klab inspect my-model.yaml
```

A preset holds only `model` and `attention` at a reference size. `derive`
scales it to the total parameter count closest to `--params` and writes a run
config in which only `model` and `attention` differ from `--base`. The
vocabulary and context length come from the base unless `--vocab-size` or
`--max-seq-len` is given; `--layers` fixes the depth.

| Preset | What it is |
| --- | --- |
| `dense` | full attention, SwiGLU feed-forward |
| `dense_gqa` | dense with grouped-query attention, three query heads per key/value head |
| `moe` | dense attention, eight routed experts with two per token, plus a shared expert |
| `moe_ngram` | `moe` with a hashed token n-gram memory at the embedding |
| `sparse` | long-context attention over a local window plus selected earlier blocks |

Scaling keeps the preset's head width, feed-forward ratio, key/value head ratio
and width-to-depth ratio, and searches width and depth. It reports the count it
reached and refuses a target it cannot get within 10% of: for example, the
`moe_ngram` tables alone hold about 33M parameters. A preset is a starting
shape, not a tuned recipe. `klab architecture derive PATH.yaml` scales a preset
file of your own, which is how a Workshop project keeps its own shapes.

## Vary the block by layer

`model.layers` replaces the attention or the feed-forward of some blocks. Every
block not named keeps `attention` and the model's own feed-forward settings.

```yaml
model:
  num_layers: 12
  layers:
    - blocks: "1::2"          # a Python slice: every second block from 1
      attention: {kind: sliding_window, window_size: 512}
    - blocks: [10, 11]        # or explicit zero-based indices
      ffn: {kind: moe, num_experts: 8, experts_per_token: 2}
```

An `ffn` entry may also set `ffn_dim`, `shared_expert`,
`router_aux_loss_coefficient` and `params`. Two entries may not set the same
slot of one block. Parameter names stay `blocks.N.attention.*` and
`blocks.N.ffn.*`, so inspection, memory estimates, checkpoints and cached
generation follow the schedule. The MLX engine and the tools that require a
uniformly dense model (weight import, the reference exercise, runtime
continuation, fixed panels) refuse a schedule.

## Add a kind from another package

A package registers a kind with an entry point named `SLOT.KIND` in the
`klab.components` group. Lab itself is not edited.

```toml
# pyproject.toml of the other package
[project.entry-points."klab.components"]
"attention.cosine" = "my_package.cosine:COMPONENT"
```

```python
# my_package/cosine.py
from pydantic import BaseModel, ConfigDict
from klab.components import Component


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 10.0


def build(model, attention, params):
    from my_package.modules import CosineAttention  # imports Torch here

    return CosineAttention(model.hidden_dim, model.num_heads, params.temperature)


COMPONENT = Component("attention", "cosine", build, params=Params)
```

```yaml
attention:
  kind: cosine
  params: {temperature: 5}
```

`build(model, attention, params)` receives the block's `ModelConfig` and
`AttentionConfig` and the validated `params` instance (`None` when the kind
declares no params model). The settings live in `attention.params`,
`model.ffn_params` or `model.memory_params`; they are validated by the
package's model and stored with its defaults filled in, so a later change of
default in the package does not silently change an existing config. Built-in
kinds keep their typed fields and accept no `params`.

Keep Torch imports inside `build`. The entry point is loaded only when a config
names a kind Lab does not ship, and configuration commands should not need
Torch to validate it.

### What a module may offer

Only `forward` is required. Everything else is optional and detected on the
module, never by its class.

| Slot | Required | Optional |
| --- | --- | --- |
| attention | `forward(x) -> Tensor` | `forward_with_aux`, `metric_tensors`, incremental cache |
| feed-forward | `forward(x) -> Tensor` | `forward_with_aux`, `metric_tensors` |
| memory | `forward(hidden, token_ids, *, position_mask=None) -> Tensor` | `forward_last`, `metric_tensors` |

- **Auxiliary loss:** `forward_with_aux(x, *, valid_target_mask, diagnostics)`
  returns `(output, loss or None)`. The loss is added to the training loss.
- **Metrics:** `metric_tensors()` returns named scalar tensors from the last
  forward. They are logged as `attention/layer_N/NAME` or `ffn/layer_N/NAME`
  (a module may set `metric_namespace`), and `memory/NAME`.
- **Incremental decoding (attention):** define `cache_backend` (a short name),
  `create_cache(batch, capacity, *, device, dtype)` and
  `forward_cached(x, cache)`. `cache_layout` (key/value heads, key width,
  value width) lets the model check a cache it is handed. Without these,
  generation recomputes the prefix, which is exact but slower.
- **Incremental decoding (memory):** `forward_last(hidden, token_ids, history)`
  returns `(updated, history)`. Without it, generation recomputes the prefix.

A memory from another package is token-addressed and has one injection site
(`memory_injection`, with `memory_block_index` for `after_block`).

### What Lab does for a kind it does not know

- **Parameter inventory:** the module is built once on the CPU per distinct
  setting and its parameters are listed, so `klab inspect`, checkpoints and
  optimizer accounting are exact.
- **Memory estimate:** an attention is charged a full `[T, T]` score allowance
  and a feed-forward as one dense SwiGLU. The estimate says so; confirm it with
  a measured warmup.
- **Provenance:** the run records an `external_components` event naming the
  package and version behind each kind. The config identity covers the kind and
  its params, not the package's code, so pin the package version for a run you
  intend to reproduce.
- **Engines:** kinds from other packages run on the PyTorch engine only.
