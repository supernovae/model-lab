# Post-training

Everything after pretraining is the same loop: start from a checkpoint, choose
what an update minimizes, choose the data, train, compare, repeat. In Lab each
turn of that loop is an ordinary run config with two extra settings.

```yaml
training:
  parent_checkpoint: work/runs/instruct   # start from these weights
  objective:
    kind: preference                      # what an update minimizes
    params: {beta: 1.0}
```

A **stage** is a run with a parent, an objective and a dataset. Stages chain by
naming the previous run as the parent. Every stage is trained, checkpointed,
resumed, compared with `klab try`, probed, chatted with and exported by the same
commands as any other run.

| Objective | What it minimizes | Needs |
| --- | --- | --- |
| `next_token` (default) | cross-entropy on the supervised targets | nothing extra |
| `distill` | cross-entropy blended with KL toward a frozen teacher | `teacher_checkpoint` |
| `preference` | DPO: prefer the chosen reply over the rejected one, relative to the frozen parent | `parent_checkpoint`, `local_preference` data |

The runnable example is [`examples/post-training`](../examples/post-training/README.md):
three stages and an export on a laptop CPU in about a minute.

## Start from a parent

`training.parent_checkpoint` loads a checkpoint's weights and starts a fresh
optimizer, schedule and step count. It accepts a run directory (its latest
checkpoint), a checkpoint generation directory or a `latest.json` pointer, and
is the config form of `klab train --promote`. The parent must have the same
architecture and tokenizer. The run's manifest records the parent run and
checkpoint digest.

Resuming a stage (`--resume`, `--recover`, `--extend-budget`) continues the
stage's own state; the parent is not loaded again as the model.

## Conversation data

`dataset.source: local_chat` reads JSONL conversations, one per line. Paths are
relative to the config; train and validation must be separate files, and a
conversation that appears in both is refused. Use the same `dataset` section in
the tokenizer config and the run config.

```yaml
dataset:
  source: local_chat
  chat_template: chatml
  train_path: data/train.jsonl
  validation_path: data/validation.jsonl
  license: CC0-1.0
  cache_dir: work/data
  train_max_documents: 10000
  validation_max_documents: 1000
  train_max_tokens: 1000000
  validation_max_tokens: 100000
```

A record is either a plain conversation, trained on every token:

```json
{"messages":[{"role":"system","content":"Answer from the workshop handbook."},{"role":"user","content":"Which drawer holds the torque wrench?"},{"role":"assistant","content":"The labeled blue drawer."}]}
```

or a version 2 record that chooses its loss and may include tool calls:

```json
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"user","content":"Look up the drawer."},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","name":"lookup_drawer","arguments":{"item":"torque wrench"}}]},{"role":"tool","tool_call_id":"call-1","content":"blue"},{"role":"assistant","content":"The labeled blue drawer."}]}
```

- **`loss_mode`.** `all_tokens` trains on the whole transcript. With
  `assistant_only`, system, user and tool-result text are context only, and
  assistant replies, serialized tool calls and the final EOS are supervised. A
  token that straddles a role boundary is not supervised, and EOS is supervised
  only when the whole record fits.
- **Structure.** An optional system message comes first; user turns and
  assistant answers alternate. Tool-call IDs are unique, and their results come
  in call order before the final answer. Arguments and results must be finite
  JSON. Tool calls are rendered as text; nothing is executed.
- **One mode per dataset.** Train and validation must use the same loss mode.

`synthetic_task: instruction_reference` generates a small offline curriculum
(arithmetic, word reversal, lookups, definitions) for checking the chat path, and
`configs/instruction_starter.yaml` and `configs/instruction_100m.yaml` train on
it. Those models learn the format, not general conversation; the
[from-toy-to-useful](from-toy-to-useful.md) guide walks from there to a narrow
useful task.

## Chat templates

A chat template says how a conversation becomes text. One declaration is used
to render training data, to build the prompt in `klab chat` and the
[local API](local-api.md), to stop a reply, and in the exported tokenizer.

```yaml
dataset:
  source: local_chat
  chat_template: chatml        # plain (default) | chatml | llama3
```

| Template | Format | Added tokens |
| --- | --- | --- |
| `plain` | `System: ...`, `User: ...`, `Assistant: ...` separated by blank lines | none |
| `chatml` | `<\|im_start\|>role\n...<\|im_end\|>`, as used by Qwen and SmolLM | `<\|im_start\|>`, `<\|im_end\|>` |
| `llama3` | `<\|start_header_id\|>role<\|end_header_id\|>\n\n...<\|eot_id\|>` | the three header and end-of-turn markers |

Train the tokenizer with the same template so its markers are single tokens.
A tokenizer config takes `chat_template` from its `dataset`, or as a top-level
field when the tokenizer corpus is plain text and chat comes in a later stage.
Preparation refuses a template whose markers the tokenizer lacks.

Declare a template of your own as a mapping:

```yaml
dataset:
  chat_template:
    name: bracketed
    system: "[SYS] {content}\n"
    user: "[USER] {content}\n"
    assistant: "[BOT] {content}\n"
    generation_prompt: "[BOT]"
    stop: ["\n[USER]"]
    special_tokens: []
```

Each role is a pattern with `{content}` once. The assistant pattern starts with
`generation_prompt`. With `loss_mode: assistant_only`, everything after the
generation prompt is supervised, the closing text included, so the model learns
to end its turn.

## Distill from a teacher

```yaml
training:
  objective:
    kind: distill
    teacher_checkpoint: work/runs/instruct
    params: {weight: 0.7, temperature: 2.0}
```

The teacher runs on the same batch without gradients. The loss is
`(1 - weight) * cross-entropy + weight * temperature² * KL(teacher ‖ student)`
over the supervised targets, with both distributions softened by `temperature`.
The teacher may be any size and architecture Lab can load; it must share the
tokenizer and read at least `training.seq_len` tokens. The student may start
fresh or from its own `parent_checkpoint`.

Logged per update: `objective/next_token_loss`, `objective/teacher_kl` and
`objective/teacher_agreement` (share of targets where both pick the same
token). Held-out loss stays plain cross-entropy, so a distilled run compares
directly with one trained without a teacher.

## Preference optimization

Preference data is one JSON object per line: the conversation so far, the reply
to prefer and the reply to move away from.

```json
{"messages": [{"role": "user", "content": "The team is split over the new plan. What should I do?"}], "chosen": "Who sees it differently? Gather their stories before you decide.", "rejected": "Pick a side and announce it."}
```

`messages` is an optional system message, then alternating user and assistant
turns ending with a user turn.

```yaml
dataset:
  source: local_preference
  chat_template: chatml
  train_path: data/pairs-train.jsonl
  validation_path: data/pairs-validation.jsonl
  license: CC0-1.0
training:
  seq_len: 96                             # each reply, prompt included, must fit
  parent_checkpoint: work/runs/instruct
  objective:
    kind: preference
    params: {beta: 1.0}
```

The objective is direct preference optimization (Rafailov et al., 2023). The
parent checkpoint is loaded a second time as the frozen reference. Each update
minimizes `-log σ(beta * margin)`, where `margin` is how much more the policy
raised the chosen reply's log-likelihood than the rejected reply's, both
measured against the reference. Rollout-based methods (PPO, GRPO) are not
implemented.

- **`beta`** sets how far the policy may move. A small value with a high
  learning rate can satisfy every pair by damaging fluency: in the example,
  `beta: 0.1` at `peak: 0.0005` reached a margin above 80 nats and produced
  garbage replies, while `beta: 1.0` at `peak: 0.0003` kept them fluent. Watch
  `objective/preference_margin` and chat with the result.
- **Logged per update:** `objective/preference_loss`, `preference_margin`,
  `preference_accuracy` (the policy alone prefers the chosen reply) and
  `reward_accuracy` (it moved toward the chosen reply relative to the
  reference).
- **Held-out loss during training** is the pair loss on the validation pairs.
  It starts at `log 2 ≈ 0.693`.
- **A pair that does not fit** in `training.seq_len` tokens is refused, not
  truncated.

The same file also works with the default objective. Its stored loss mask
covers the chosen replies only, so `objective: next_token` on a
`local_preference` dataset is instruction tuning on the chosen replies. That is
the natural baseline for a preference run, and it is why `klab try`, probes and
`klab eval` score a preference run by the held-out loss of the chosen replies.

## Compare settings in lab mode

Both arms of a try start from the parent the baseline names:

```yaml
# gentler.yaml
question: Does a tighter hold on the parent keep more of the chosen replies' likelihood?
set:
  training.objective.params.beta: 3.0
```

```sh
klab try gentler.yaml --vs 3-prefer.yaml
```

The parent and teacher checkpoints are part of what a reusable baseline is
keyed on. One seed is within-run noise only; confirm a difference with paired
seeds before calling it a result.

## Export for Hugging Face

```sh
klab weights export work/runs/prefer exported/prefer
```

writes a directory `transformers.LlamaForCausalLM` reads: `config.json`,
`model.safetensors`, the tokenizer with the run's chat template as its
`chat_template`, generation settings that stop at the template's end-of-turn
token, and `klab_export.json` with the source checkpoint digest and file
hashes. It is the inverse of
`klab weights import --format hf_llama_safetensors`, and a test imports an
exported directory back to bit-identical weights.

Only what Llama can express exports: dense attention (grouped key/value heads
included), dense SwiGLU feed-forward, no memory. Other architectures are
refused with the reason. The export holds float weights; a quantized artifact
(below) is refused, so export its source checkpoint. GGUF conversion is not in
Lab; run it on the exported directory with the tools built for that format.

## Quantize weights

```sh
klab weights quantize prefer --runs-dir work/runs --bits 8 --output work/quantized/prefer-int8
klab weights quantize prefer --runs-dir work/runs --bits 4 --group-size 128 --output work/quantized/prefer-int4
klab generate prefer --runs-dir work/runs --quantized work/quantized/prefer-int8 --prompt "Hello"
klab chat prefer --runs-dir work/runs --quantized work/quantized/prefer-int8 --message "Hello"
klab serve prefer --runs-dir work/runs --quantized work/quantized/prefer-int8
```

Weight-only, symmetric quantization of a verified checkpoint (activations stay
fp32):

| `--bits` | Scheme | Scale |
| --- | --- | --- |
| `8` | int8, codes in [-127, 127] | one per output channel |
| `4` | int4, codes in [-7, 7], two per byte | one per `--group-size` input columns (default 128; must divide every projection's input width) |

Only the `nn.Linear` projections inside decoder blocks are quantized
(attention and feed-forward, MoE experts included). Token embeddings, the output
head (often tied to them), norms, MoE routers and memory modules keep their
checkpoint weights. A quantized layer stores codes and scales and dequantizes on
the fly, so resident weight memory drops but decoding is not faster than fp32.
No quantization library is in the locked environment (torchao, bitsandbytes,
optimum-quanto, llm-compressor), and `torch.ao` dynamic quantization also
quantizes activations and has no 4-bit path, so Lab implements this minimal
scheme itself.

`quantize` scores the source checkpoint and the quantized model on the run's
validation split with the same evaluation settings (`evaluation.batch_size`,
`evaluation.max_batches`, `training.seq_len`, on `--backend`, default `cpu`),
and prints and records both losses, their difference (`loss_delta`, quantized
minus source) and the resident weight bytes. The output directory holds
`quantized.safetensors` and a sealed `klab_quantized.json`: the source run,
checkpoint and digest, the scheme, the quantized modules, the weight file's
SHA-256 and the `heldout` block. On a 300-step CPU smoke run (2 layers, hidden
width 32, 4 validation batches) the source loss was 1.8039; int8 scored 1.8042
(+0.0003) and int4 with group size 32 scored 1.7932 (-0.0106, within that tiny
evaluation's noise).

`--quantized DIR` in `generate`, `chat` and `serve` loads the run's checkpoint
through the usual verified path, then swaps in the artifact's layers. The
artifact must come from exactly that checkpoint (otherwise it is refused with
the `--checkpoint` to pass), its manifest digest and weight file must verify,
and its modules must match the model. The loaded identity carries a
`quantization` block. MLX runs and attached semantic packs are refused.

## Write your own objective

An objective is a small class. It receives a `forward` for the model being
trained, the batch and any frozen models, and returns a loss.

```python
# my_package/humble.py
from pydantic import BaseModel, ConfigDict

from klab.objectives import Objective
from klab.training.objectives import Loss, TrainingObjective, cross_entropy_sum


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence_penalty: float = 0.1


class Humble(TrainingObjective):
    """Cross-entropy, minus a reward for keeping the next-token distribution open."""

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def loss(self, forward, batch, frozen):
        logits = forward(batch.inputs).float()
        log_probs = logits[batch.targets != -100].log_softmax(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return Loss(
            cross_entropy_sum(logits, batch.targets) - self.penalty * entropy.sum(),
            {"entropy": entropy.detach().mean()},
        )


OBJECTIVE = Objective(
    "humble", lambda params: Humble(params.confidence_penalty), Params
)
```

```toml
[project.entry-points."klab.objectives"]
"humble" = "my_package.humble:OBJECTIVE"
```

```yaml
training:
  objective: {kind: humble, params: {confidence_penalty: 0.2}}
```

- **`Loss.total`** is summed over the microbatch's supervised targets; the
  engine divides the update's total by its target count. `Loss.metrics` are
  microbatch means, logged as `objective/<name>`.
- **`Objective(..., teacher=True)`** makes `teacher_checkpoint` required and
  passes the frozen model as `frozen.teacher`. `reference=True` does the same
  with the parent as `frozen.reference`. `pairs=True` delivers
  `[pairs, 2, tokens]` batches from a `local_preference` dataset, chosen first.
- **`evaluation_loss`** may be overridden; the default is cross-entropy.
- **Recorded:** params are stored with defaults filled in, and the run logs a
  `training_objective` event with the package name and version and the
  identities of the frozen checkpoints.

This is the place to try training signals that are not "the fastest right
answer": preference pairs that reward a clarifying question in an ambiguous
situation and a direct answer in a clear one (the shipped example does this),
a teacher that behaves the way you want, or a loss term of your own.

## Limits

- **Engine:** objectives other than `next_token`, and `local_preference` data,
  run on the PyTorch engine on one device. They are refused with byte or
  portable memory, allocation or portability sidecars, matched stories, the
  compiled output head and secondary devices. MLX is refused.
- **Memory:** a teacher or reference is a second model on the device.
  `klab inspect` does not count it.
- **Not implemented:** low-rank adapters (LoRA), rollout-based reinforcement
  learning, activation or quantization-aware training, GGUF. Weight-only int8
  and int4 inference is [above](#quantize-weights). Partial freezing through
  `training.trainable_parameters` works with every objective.
- **Export check:** the exported directory was verified by importing it back.
  It has not been loaded with `transformers` in Lab's test environment, which
  does not install it.
