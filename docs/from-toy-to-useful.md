# From a toy model to a useful one

You do not need to understand transformer internals to start. You need a task,
examples of good answers, held-out questions to test on, and a record of what
changed. This guide goes from a small instruction-following model to a narrow
model that does one real job.

## The ideas you need

| Term | Meaning |
| --- | --- |
| Token | A piece of text represented by a number. A word can be several tokens; spaces and punctuation count too. |
| Tokenizer | The fixed translator between text and token IDs. It is not the model's knowledge. |
| Parameter | A learned number in the model. More parameters add capacity, not knowledge or reliability by themselves. |
| Training | Show text, predict each next token, compare with the real one, adjust the weights. Repeating the same examples adds no new information. |
| Checkpoint | Saved weights and training state. A config describing a 100M model is not a trained model. |
| Prompt | Text given at inference time. It can ask for behavior the model learned; it does not change the weights. |
| Context window | How much text one prediction can see. Independent of parameter count and training budget. |
| Validation and test | Validation guides decisions while you develop; a separate, untouched test measures the final claim. Tuning on a test turns it into validation. |

A model trained on stories continues stories. A model trained on well-formed
question and answer conversations can learn to answer questions. A chat window
shows both the same way; it does not give them the same abilities.

Chat is serialized as text through a [chat template](post-training.md#chat-templates).
With the default `plain` template a prompt looks like:

```text
System: You are a concise local assistant.

User: Explain causal attention in one sentence.

Assistant:
```

The model continues after `Assistant:`. Role labels and system instructions are
learned conventions, not enforced rules; there is no hidden search, tool use or
memory behind them.

## 1. Train a small instruction learner

`configs/instruction_starter.yaml` trains a 3.3M-parameter model on a small
offline instruction curriculum (arithmetic, reversing words, lookups,
definitions, greetings) on CPU:

```sh
RUNS="$KLAB_WORK_DIR/runs"
klab tokenizer train configs/tokenizer_instruction_8k.yaml
klab data prepare configs/instruction_starter.yaml
klab train configs/instruction_starter.yaml --runs-dir "$RUNS" --run-id instruction-starter-pilot --stop-after-step 2
klab checkpoint verify "$RUNS/instruction-starter-pilot/checkpoints/latest.json" --json
klab train configs/instruction_starter.yaml --runs-dir "$RUNS" --run-id instruction-starter
klab eval instruction-starter
klab chat instruction-starter --system "You are a concise local assistant." --max-new-tokens 32
```

`eval`, `chat` and the other inspection commands read `$KLAB_WORK_DIR/runs` by
default; pass `--runs-dir` to `train` so they find the run.

The two-step pilot checks that everything runs before the full budget. Expect a
model that has learned the curriculum's format and some of its tasks, and that
answers anything outside it badly. `configs/instruction_100m.yaml` is a
104,843,648-parameter version of the same curriculum with a 20M-target budget;
a bigger model on the same synthetic data is still a curriculum learner.

## 2. Pick one job and define it

A good first job is narrow and checkable, for example routing support requests:
given a short request, reply with `access`, `billing` or `delivery`. Before
collecting data, write down:

- who will use it and what it decides;
- what each label means, and what to do with a request that fits two labels or
  none (if "unknown" or a clarifying question is right, train that too);
- an acceptance target on held-out requests, and the baseline to beat (a few
  keyword rules may already do well);
- which mistakes are costly and who reviews them.

Avoid medical, legal, financial or safety-critical advice as a first job.

## 3. Collect examples

Write conversations as JSONL, one per line, with the same system text you will
use in practice:

```json
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"system","content":"Classify the support request. Reply with exactly one label: access, billing, or delivery."},{"role":"user","content":"I forgot my password and cannot log in."},{"role":"assistant","content":"access"}]}
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"system","content":"Classify the support request. Reply with exactly one label: access, billing, or delivery."},{"role":"user","content":"I was charged twice for the same order."},{"role":"assistant","content":"billing"}]}
```

Two lines show the format; a useful set needs varied phrasing, misspellings,
short and long requests and real ambiguity, with private information removed.
Split train, validation and test by customer, conversation or source, not by
row: a light paraphrase of a training request is not an independent test. Lab
refuses exact duplicates across splits but cannot detect paraphrases.

`assistant_only` trains on the replies and uses the rest as context. See
[conversation data](post-training.md#conversation-data) for the full format.

## 4. Adapt the model

Copy `instruction_starter.yaml` to `support.yaml`, give it a new name and point
its dataset at your files:

```yaml
dataset:
  source: local_chat
  train_path: data/support-train.jsonl
  validation_path: data/support-validation.jsonl
  license: Proprietary, used with permission
  cache_dir: work/data
  train_max_documents: 10000
  validation_max_documents: 1000
  train_max_tokens: 1000000
  validation_max_tokens: 65536
training:
  parent_checkpoint: /data/klab/runs/instruction-starter   # the run from step 1
```

`parent_checkpoint` starts from the instruction learner's weights with a fresh
optimizer. Keep its model shape and tokenizer; a different tokenizer means
training from scratch.

```sh
klab data prepare support.yaml
klab train support.yaml --runs-dir "$RUNS" --run-id support
klab chat support --message "My parcel has not arrived." --max-new-tokens 4
```

To compare two settings fairly, use [lab mode](lab-mode.md): both arms start
from the same parent and are scored on the same held-out conversations.

## 5. Measure the job

Score the job, not the model's fluency:

- Held-out loss on your validation conversations (`klab eval support`) tells you
  whether training generalizes, but not whether answers are right.
- Run every test request through `klab chat --message ... --json` (or the
  [local API](local-api.md)), compare the reply with the expected label, and
  count. Review every error.
- Compare against your rules baseline on the same test set.
- Keep the test set untouched until you have chosen a model.

If the model is close but prefers the wrong style of answer in some situations,
[preference tuning](post-training.md#preference-optimization) on chosen and
rejected replies can shift that.

## 6. When it fails

| Symptom | Likely cause | Change one thing |
| --- | --- | --- |
| Good on training requests, poor on new ones | Too few or too similar examples | More varied data; check for leakage between splits |
| Confident wrong labels on ambiguous requests | No examples of ambiguity | Add an `unknown` or clarifying answer and train it |
| Fluent but wrong | Learned the format, not the task | More examples per label; check label definitions |
| Lost earlier abilities | Adaptation overwrote them | Mix some earlier data back in, or lower the learning rate |
| Loss falls but answers do not improve | The loss is dominated by easy tokens | Use `assistant_only`; check the system text matches |

## 7. Do more parameters help?

Sometimes, when capacity is the bottleneck. There is no parameter count at which
a model becomes a chat assistant. Four things scale independently:

1. **Capacity:** width, depth, feed-forward size and optional mechanisms.
2. **Data coverage:** distinct, correct, representative examples, not more
   repetitions.
3. **Training budget:** updates and supervised targets; larger models usually
   need more.
4. **Context:** the window, and training on dependencies that long.

Keep a dense baseline and add [memory tables or experts](architecture.md) only
to test a specific idea.

Training memory starts at about 16 bytes per parameter with FP32 weights and
AdamW (weights 4, gradients 4, two optimizer moments 8), before activations and
workspace: about 1.6 GiB at 100M parameters and 15 GiB at 1B. `klab inspect`
estimates the rest; see [memory](memory.md).

For broad assistant quality, adapting a properly licensed pretrained model
([`klab weights import`](checkpointing.md#import-weights)) is usually faster than
training one from scratch.
