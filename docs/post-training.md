# Post-training: instruction tuning, distillation, preference and export

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
to end its turn. Record format and loss modes are in
[instruction training](instruction-training.md).

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
refused with the reason. Quantization and GGUF conversion are not in Lab; run
them on the exported directory with the tools built for that format.

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
  learning, quantization, GGUF. Partial freezing through
  `training.trainable_parameters` works with every objective.
- **Export check:** the exported directory was verified by importing it back.
  It has not been loaded with `transformers` in Lab's test environment, which
  does not install it.
