# From flashcards to a useful local assistant

Commands below use the [activated, provisioned environment](../README.md#first-run).

**You do not need to understand transformer internals to start.** You need a task, examples of good answers, a way to test new questions, and a record of what changed.

This guide explains the strange `amber → lumen` example, gives a normal starting chat prompt, and walks from today's small experiments toward a useful narrow assistant. It distinguishes **what runs today**, **what still needs training**, and **framework extensions that are not implemented**.

For a first experiment, use [the TinyStories lab loop](tinystories-microlab.md).
This guide is the advanced instruction-learning route and includes retained
observations from earlier runs.

Jump to: [normal chat prompts](#a-normal-starting-prompt) · [runnable small learner](#5-a-smaller-instruction-learner-before-the-100m-run) · [a useful domain task](#6-turn-the-exercise-into-a-useful-job) · [parameter and memory scaling](#8-do-more-parameters-make-this-a-better-assistant).

## 1. What does “amber means lumen” mean?

Nothing about the real world. We deliberately invented a table:

| Question key, called an “alias” | Answer, called its “value” |
|---|---|
| amber | lumen |
| birch | orbit |
| coral | quartz |

Think of flashcards: the front says `amber`; the back says `lumen`. The model repeatedly sees questions and correct answers during training. Later we ask a question without showing the answer. A correct reply suggests it learned the association.

`lumen` is a word, but its dictionary meaning is irrelevant here. The task is **not** asking about colors, light, or gemstones. Inference does not call a Python dictionary: the trained model predicts the answer's tokens. A dictionary would nevertheless be the better application if all we needed were these 24 fixed lookups!

The exercise separates three abilities:

1. **Retention:** can it answer a question it practiced?
2. **New wording:** can it answer the same fact asked differently?
3. **Following context:** if the conversation says “for this question, amber means orbit,” can it use the new instruction rather than repeat the memorized answer?

Our [recorded experiment](history/project-review.md) obtained:

| Test | Dense model | Engram model |
|---|---:|---:|
| Retention before training | 0/24 | 0/24 |
| Retention after training | 24/24 | 24/24 |
| New wording | 8/12 | 6/12 |
| Conversation-local overrides | 0/8 | 0/8 |

That is evidence of learning, partial wording generalization, and a clear failure to follow changed context. **It is not evidence of a useful chatbot or an Engram advantage.** These are exploratory, single-seed results. We retain the failed tests rather than rename memorization “reasoning.”

## 2. The minimum mental model

| Term | What it means in this project |
|---|---|
| Token | A piece of text represented by a number. One word can require several tokens; punctuation and spaces count too. |
| Tokenizer | The fixed translator between text and token IDs. It is not the model's knowledge. |
| Parameter / weight | A learned number inside the model. More parameters provide capacity, not automatically more knowledge or reliability. |
| Training | Show text, predict the next token, compare with the actual next token, and update weights. Repetition of the same examples is not new information. |
| Checkpoint | Saved weights and associated training state. A configuration describing 100M parameters is not itself a trained checkpoint. |
| Prompt | Text supplied at inference time. It can request behavior the model has learned, but does not update its weights. |
| Context window | The amount of tokenized conversation available to a prediction. This is separate from parameter count and the training-token budget. |
| Validation / test | Validation helps make development decisions; an independent frozen test measures the final claim. Repeatedly tuning on a test turns it into development data. |
| Capability card | Versioned questions, expected answers, scoring rules, and generation settings. It makes an experiment repeatable. |

A model trained on stories learns to continue stories. A model trained on well-formed question/answer conversations can learn to answer questions. A model trained on 24 associations can learn those associations. The terminal can display all three in a chat interface; the interface does not give them the same capabilities.

Model Lab currently serializes chat as ordinary text:

```text
System: You are a concise local assistant.

User: Explain causal attention in one sentence.

Assistant:
```

The model continues after `Assistant:`. The role labels and system instructions are learned text conventions, not privileged programmatic rules. No search, tools, database access, persistent personal memory, or instruction-security guarantee is hidden behind them. Chat history is supplied as context; `/reset` clears it, not the trained weights.

## 3. Capability goals and current evidence

These are evidence questions, not parameter-count stages. A runnable path does not show that a model learned the target behavior.

| Capability goal | Evidence today | What is still needed |
|---|---|---|
| Execute and reproduce a run | CPU, MPS, and MLX paths have training/evaluation/checkpoint smoke evidence. | Larger-workload fit and performance must be measured on the actual backend; smoke execution is not a quality result. |
| Retain taught associations | The dense and Engram alias models reached 24/24 training-format retention; held-out wording was 8/12 dense and 6/12 Engram. | Independent facts/tasks and repeat seeds; the aliases are a tiny synthetic curriculum. |
| Use changed conversation context | The alias override card scored 0/8 for both models; the domain-adaptation study also found unreliable held-out behavior and severe retention loss. | Robust corrections, conflicting/missing facts, abstention, and forgetting checks on independent tasks. |
| Learn text patterns | Pinned-corpus model training and held-out next-token loss have been measured, including a multi-seed FineWeb-Edu micro study. | Human-reviewed continuations and broader held-out task evidence; lower language-model loss alone does not establish useful behavior. |
| Follow instructions | The measured instruction starter learned a recognizable curriculum template. Its ordinary arithmetic, color, and unrelated responses were wrong. | Diverse licensed conversations, natural paraphrases, independent cases, and explicit unknown/clarification behavior. |
| Complete one useful job | Task builders, custom cards, and blinded review tools are available. | No useful real-domain model has been established; compare a trained checkpoint against rules/search on permissioned, independently held-out requests. |
| Use external memory reliably | Lexical/byte tables and portable adapters run; semantic packs support bounded retrieval from supplied vectors. The two-case portable result was negative. | Broader transfer tests; semantic text encoding and automatic retrieval are not on the standard chat/training path. |

Choose the next capability that matters to a real user and define how it can fail. Open questions and remaining experiments are tracked in the [research roadmap](research/roadmap.md) and versioned lifecycle; report missing implementation through the [contribution workflow](../CONTRIBUTING.md#planning-and-contributions).

## 4. Which larger model do we actually have?

There are two different larger-model paths:

- **`micro_dense.yaml` through `dense_50m.yaml`:** approximately 3.3M–50.3M dense parameters, using TinyStories. The supplied scale runs have a deliberately small 204,800-target training ceiling. They are architecture/continuation experiments, not pretrained general assistants.
- **`instruction_100m.yaml`:** **104,843,648 parameters**, an 8,192-token vocabulary, a 20-million-target training ceiling, and a synthetic instruction corpus. This is a runnable training preset, not a demonstrated 100M chat result. Twenty million repeated/synthetic targets do not establish broad language competence.

Historical checkpoint conversion is now available through the validated weights-only import path. It preserves source architecture/tokenizer semantics and records provenance; it does not restore a missing optimizer history or turn an old model into a useful assistant. See [compatible weight import](checkpointing.md#import-compatible-external-weights).

The 100M curriculum includes arithmetic, reversing a few words, simple color questions, a definition, sentence templates, and greetings. It even prefixes requests with `Reference N:`. Different reference numbers do not imply different semantic tasks. Removing that artificial prefix is a useful development probe, but success on the repetitive curriculum is not a broad instruction-following benchmark.

### A normal starting prompt

For a run trained on the existing instruction curriculum, start with its actual training prefix:

```text
You are a concise local assistant.
```

After training the `instruction-100m` run below:

```sh
klab chat instruction-100m --runs-dir klab-work/runs \
  --system "You are a concise local assistant." \
  --max-new-tokens 32
```

Try natural questions such as `Hello!`, `What is 2 plus 3?`, and `Explain causal attention in one sentence.` These are **probes, not promised model outputs**. Compare natural wording with the curriculum-shaped request `Reference 5: reply with a friendly greeting.` Its training target is `Hello! How can I help?`; that is an example answer in the dataset, not a measured response from a trained 100M run.

You can also use this shorter-to-longer progression when building your own conversational corpus:

| Purpose | System prompt |
|---|---|
| Existing synthetic curriculum | `You are a concise local assistant.` |
| New, more natural assistant curriculum | `You are a helpful local assistant. Answer briefly. Ask one question if the request is unclear. Say when you do not know.` |
| New context-grounded curriculum | `Answer briefly using only the facts provided in this conversation. If the answer is missing, say you do not know.` |

The latter two are **training targets for a new curriculum**, not upgrades you can apply to the alias checkpoint. Include demonstrations of clarification and “I don't know” in that curriculum. A system prompt alone cannot enforce factuality, reliable abstention, or resistance to malicious instructions.

The current instruction preset has a **256-token maximum context** and trains on **128-token blocks**. System text, history, the current question, role labels, and the reserved answer all consume that context. A 32-token answer allowance leaves at most 224 tokens for the prompt, not 224 words. Chat drops old complete turns when necessary. A very long system prompt can make even the first question fail to fit. Increasing maximum context does not teach long-context behavior without suitable training examples.

## 5. A smaller instruction learner before the 100M run

Run these commands from the repository root. Choose new run names for reruns; saved runs and tokenizer artifacts are not scratch files.

First create the existing instruction tokenizer:

```sh
klab tokenizer train configs/tokenizer_instruction_8k.yaml
```

Use the included **[`configs/instruction_starter.yaml`](../configs/instruction_starter.yaml)**; there is no configuration to assemble first. It keeps the approximately 3.3M backbone but switches to the instruction dataset and its tokenizer.

| Setting | Starter value | Meaning |
|---|---:|---|
| Hidden width / blocks / heads | 192 / 4 / 4 | The size of this dense model. |
| Vocabulary | 8,192 | Must match the trained instruction tokenizer. |
| Training block / maximum context | 128 / 256 tokens | What training sees versus the inference ceiling. |
| Microbatch / accumulation | 2 / 1 | Two examples per nominal optimizer update. |
| Update / target ceilings | 1,000 / 256,000 | Training stops at the first limit reached. |
| Backend / precision | CPU / FP32 | A reproducible local starting point. |

The modest budget is for studying learning, not a quality guarantee. Copy the config to a new name before changing settings for your own experiment.

Then exercise the real path before a full run:

```sh
klab inspect configs/instruction_starter.yaml --json
klab data prepare configs/instruction_starter.yaml
klab train --runs-dir klab-work/runs configs/instruction_starter.yaml --run-id instruction-starter-pilot --stop-after-step 2
klab checkpoint verify klab-work/runs/instruction-starter-pilot/checkpoints/latest.json --json
klab eval instruction-starter-pilot --runs-dir klab-work/runs
```

The two-step run should be interrupted at a safe boundary with a verifiable checkpoint. It verifies execution, **not learning quality**. It uses the configured training schedule; it is not a hidden warmup for the next run. Start the learning run fresh:

```sh
klab train --runs-dir klab-work/runs configs/instruction_starter.yaml --run-id instruction-starter
klab eval instruction-starter --runs-dir klab-work/runs
klab chat instruction-starter --runs-dir klab-work/runs --system "You are a concise local assistant." --max-new-tokens 32
klab report --kind run instruction-starter --runs-dir klab-work/runs --json
```

Keep a short development questionnaire: greetings, arithmetic, a definition, rewordings, an unrelated question, and a request to correct an earlier answer. Record mistakes as well as good answers. Use a separate frozen set for a final claim; the synthetic validation split alone is not semantic generalization evidence.

### Measured starter example: what improved, what did not

While writing this guide, we trained the exact starter config on CPU as `guide-instruction-starter`: **3,344,064 parameters, 1,000 updates, 256,000 prediction targets**. The paired validation loss fell from **8.9680 at step zero to 1.0138 at the final checkpoint**, over the same 2,048 validation targets. Eleven checkpoint/validation pairs verified. This repetitive synthetic validation corpus is not an independent test of general chat ability.

The real interactive CLI produced the following with `You are a concise local assistant.`, greedy decoding, a 32-token answer allowance, and `/reset` between questions:

| User message | Actual model reply |
|---|---|
| `Hello!` | `This model studies sparse patterns.` |
| `Reference 5: reply with a friendly greeting.` | `Hello! How can I help?` |
| `What is 2 plus 3?` | `101` |
| `Explain causal attention in one sentence.` | `Causal attention lets each position use earlier positions but not future ones.` |
| `I have forgotten my password. Can you help me?` | `explain causal attention in one sentence plus 9?` |
| `What is the capital of France?` | `explain causal attention in one sentence.` |
| `Reference 2: Ava has a red book. What color is the book?` | `The map is blue.` |

This model learned recognizable sentence templates and one curriculum definition; it did **not** become a reliable assistant or even a reliable arithmetic/color solver. Low overall next-token loss can be dominated by predictable template text while the answer that matters is wrong. The next change should address data coverage and task evaluation before assuming that more parameters solve the problem.

The run and `evaluations/guide-chat.json` transcript are retained locally, not committed as model downloads. On the machine containing that run, try it without retraining:

```sh
klab chat guide-instruction-starter --system "You are a concise local assistant." --max-new-tokens 32
```

This is a verified learning/example run, not a recommended application model. No full 100M instruction run or domain-support quality evaluation was performed for this guide.

### Then try the larger reference deliberately

```sh
klab inspect configs/instruction_100m.yaml --json
klab data prepare configs/instruction_100m.yaml
klab train --runs-dir klab-work/runs configs/instruction_100m.yaml --run-id instruction-100m-pilot --stop-after-step 2
klab checkpoint verify klab-work/runs/instruction-100m-pilot/checkpoints/latest.json --json
klab train --runs-dir klab-work/runs configs/instruction_100m.yaml --run-id instruction-100m
klab eval instruction-100m --runs-dir klab-work/runs
klab chat instruction-100m --system "You are a concise local assistant." --max-new-tokens 32 --runs-dir klab-work/runs
```

Do not launch the full run solely because an estimate says `LIKELY_TO_FIT`. Inspect the actual configured backend and resource cost; use a short real run first. The `inspect` command uses tensor shapes without allocating the model, and `backend: auto` uses the same selector as training. Missing physical-capacity readings cannot certify fit. `stage --through warmup` runs an isolated, disposable pilot and records its measured report; it is still not a substitute for the configured full run or a large-model fit guarantee.

The starter and 100M recipes differ in size, optimizer settings and budget. Their scores are useful development observations, **not a controlled size comparison**. To isolate model size, copy one recipe, change only the backbone dimensions, and compare at the same observed steps/targets with `--vary scale`. Also study separate score-versus-training-budget curves; equal token budgets may undertrain a larger model.

## 6. Turn the exercise into a useful job

A good first target is **support-request triage**: given a short request, emit `access`, `billing`, or `delivery`. These labels have an application meaning, unlike amber/lumen. It still needs evidence: a rules-based baseline may be cheaper and better.

### Define the contract before collecting examples

Write down:

- The intended users and exactly what the classifier will and will not decide.
- Label definitions, including how to handle a request with two issues or no matching issue. Add and train an `unknown`/clarification behavior if it belongs in the contract.
- A held-out acceptance target chosen for the application, plus a baseline. Example development goal: at least 90% accuracy on 100 independently collected requests, with every error reviewed. This is an illustrative target, **not a measured result or a production guarantee**.
- Higher-risk failure categories and a human-review route. Do not start with medical, legal, financial, or safety-critical advice.

For a three-label curriculum, use the same system text in training and evaluation:

```text
Classify the support request. Reply with exactly one label: access, billing, or delivery.
```

### Prepare real examples, not just a better prompt

Use licensed UTF-8 JSONL. Every line is a complete conversation. For example:

```jsonl
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"system","content":"Classify the support request. Reply with exactly one label: access, billing, or delivery."},{"role":"user","content":"I forgot my password and cannot log in."},{"role":"assistant","content":"access"}]}
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"system","content":"Classify the support request. Reply with exactly one label: access, billing, or delivery."},{"role":"user","content":"I was charged twice for the same order."},{"role":"assistant","content":"billing"}]}
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"system","content":"Classify the support request. Reply with exactly one label: access, billing, or delivery."},{"role":"user","content":"My parcel has not arrived."},{"role":"assistant","content":"delivery"}]}
```

These three lines illustrate the format; they are **not an adequate training corpus**. Collect varied phrasing, spelling errors, short and long requests, and realistic ambiguity. Remove private information. Split by customer/conversation/source and relevant templates before training; changing row IDs or lightly paraphrasing the same request is not an independent test.

Copy `instruction_starter.yaml` to **`configs/support_triage.yaml`**, change its name, and replace the dataset section with:

```yaml
dataset:
  source: local_chat
  train_path: ../data/support-train.jsonl
  validation_path: ../data/support-validation.jsonl
  license: Proprietary - used with permission
  cache_dir: ../artifacts/data
  train_max_documents: 10000
  validation_max_documents: 1000
  train_max_tokens: 1000000
  validation_max_tokens: 65536
```

Create those files from your corpus and replace the license with your actual rights. Keep the starter's model dimensions and tokenizer for this continuation. The existing tokenizer can encode new text through its byte vocabulary, though its efficiency on a new domain may be poor. If starting a new model from scratch, you may train a domain tokenizer on training data only; changing the tokenizer is **not** a compatible weight continuation.

Model Lab rejects exact cross-split duplicates, but it cannot certify semantic independence. The v2 examples above explicitly use **assistant-only** loss: system/user text supplies context rather than supervised targets. V2 also supports `all_tokens`; historical unversioned records retain whole-transcript loss. Training/evaluation counters count valid supervised prediction targets. Long conversations are packed into fixed blocks, so the facts needed to predict an answer must still fit the training window. See [objective and inert tool-transcript semantics](instruction-training.md#local-conversation-corpora).

### Adapt a compatible checkpoint, or start fresh

After the starter learning run exists:

```sh
klab data prepare configs/support_triage.yaml
klab train --runs-dir klab-work/runs configs/support_triage.yaml --run-id support-adapted \
  --promote klab-work/runs/instruction-starter/checkpoints/best.json
klab chat support-adapted --runs-dir klab-work/runs \
  --system "Classify the support request. Reply with exactly one label: access, billing, or delivery." \
  --max-new-tokens 8
```

**Promotion** retains compatible model weights but starts a new optimizer, schedule, cursor and target counter on the new data. It is the route for this domain adaptation. **Resume** restores the same experiment after interruption; it is not a way to change the corpus or budget. Training without `--promote` starts from random weights. Compare both approaches rather than assuming the synthetic starter supplies useful pretraining.

Promotion does not grow a 3M model into a 100M model, attach new Engram parameters, or import an arbitrary downloaded model. Architecture and tokenizer compatibility are required. A larger backbone starts as a separate fresh experiment unless an explicit weight-transfer method is implemented.

### Score the job, not the model's confidence

Export a card template with `klab capability describe chat-alias-recall-v1`. Save the JSON as `data/support-test-v1.json`, remove `digest`, and replace the name, hypothesis, limitations and cases. For each test request use the same system text and this prompt structure:

```text
System: Classify the support request. Reply with exactly one label: access, billing, or delivery.

User: The invoice includes a service I never ordered.

Assistant:
```

The case has a unique `identifier`, this `prompt`, `expected: "billing"`, and `kind: "custom"`. Use `normalized_full_answer_exact_v1`, the existing normalization and greedy protocol, and an 8-token answer allowance. Declare chance as `1/3` for three possible labels; majority-answer baseline is the largest actual label fraction in the test set. Expand the labels and recompute controls if you add an unknown class. See the [full card contract](capabilities.md#bring-a-new-task-without-changing-python).

Once independently held-out cases have been frozen:

```sh
klab capability describe data/support-test-v1.json
klab capability evaluate support-adapted data/support-test-v1.json --runs-dir klab-work/runs
klab report --kind run support-adapted --json --runs-dir klab-work/runs
```

Review the saved per-case replies: wrong label, extra prose, or a truncated answer should not quietly become a pass. Exact-answer cards suit labels and fixed extraction. They do **not** grade open-ended explanations fairly; those need a declared human rubric or a separately implemented evaluator. `best.json` selects lowest validation loss, not the best test score.

A handbook assistant is a useful next test: supply facts in context, train answers and missing-information responses, then test on new documents. Claim **reading supplied context**, not memorizing unseen facts. Model Lab has bounded semantic pack retrieval in a direct PyTorch `DenseLM` API, but the standard chat/generation path does not encode text queries or retrieve automatically. Manual short context is supported; a complete text-to-query and retrieval application path remains open.

## 7. Decide what to change after a failure

| Observation | Investigate first | Do not assume |
|---|---|---|
| Cannot answer training examples | Data formatting, labels, tokenizer, context truncation, loss, learning rate, and sufficient updates | “We need a billion parameters.” |
| Training answers good, new requests poor | Diversity, duplicates/leakage, realistic splits, missing task coverage | More repetition equals more knowledge. |
| Repeats an old fact after a correction | Explicit override/correction examples and causal context availability | More memory automatically improves instruction following. |
| Explains well but invents facts | Grounding, missing-information training, abstention tests, external verification | A longer safety prompt enforces truth. |
| Answers are cut off | Completion budget, EOS/role stops, total context | Increasing temperature makes answers more complete. |
| Forgets old tasks after adaptation | Retention tests, data mixture/rehearsal, schedule and tradeoff measurements | Fine-tuning only adds knowledge. |
| Out of memory | Microbatch, sequence length, activation recomputation, model/optimizer storage | Sparse active weights mean sparse resident state. |

Change one factor at a time. Record the data/tokenizer/card versions, source identity, actual step/target counts, checkpoint, seed, generation settings, scores and failures. Preserve a previous good checkpoint and its evaluations.

## 8. Do more parameters make this a better assistant?

**Sometimes, when capacity is the bottleneck and the training supports the task. There is no chat-capable parameter threshold.** Friendly wording can be learned by a small model; broad knowledge, reliable multi-turn reasoning and robustness are much harder. None follows from a parameter count alone.

There are at least four independent scaling choices:

1. **Model capacity:** width, number of blocks, feed-forward size, and optional mechanisms.
2. **Data coverage:** distinct, correct examples and representative language/tasks—not just more repetitions.
3. **Optimization budget:** successful updates and actual prediction targets; larger models may need substantially more training.
4. **Context:** both the allowed window and training on dependencies of the relevant length.

Keep a dense baseline. Add Engram or MoE only for an explicit hypothesis, not as a synonym for intelligence. Engram adds addressed storage; MoE adds conditional computation and resident expert state. Compare quality, total parameters, active-use accounting, actual resources and training budgets separately. Neither adds a web search engine or a verified knowledge base.

### Memory, before any billion-parameter experiment

For ordinary trainable FP32 weights with AdamW, a useful lower-level accounting is:

- weights: approximately `4 × parameter_count` bytes;
- gradients: approximately another `4 × parameter_count`;
- two optimizer moments: approximately another `8 × parameter_count`.

That is approximately **16 bytes per parameter before activations, attention, logits, temporary buffers and runtime overhead**. At 104,843,648 parameters these three categories already total about **1.56 GiB**; at one billion they total about **14.9 GiB**. Actual peak use is higher. This approximation is not a fit certificate, and frozen weights/other optimizers change it.

Smaller microbatches with accumulation reduce the per-forward activation load while preserving the configured effective example batch. Block recomputation trades extra compute for lower activation retention. Neither eliminates weights or AdamW state. Longer sequences can increase dense attention working memory quadratically. CPU and Apple GPU unified memory are not two independent pools to add together.

Shape-only inspection estimates parameter, optimizer, activation, and working-memory costs without allocating the model. It is not a physical-fit result; use isolated staging and actual measurements before increasing the workload.

## 9. What remains before stronger capability claims?

| Available and exercised | Not yet established |
|---|---|
| Validated legacy and compatible Llama safetensor weight import | General external architectures, tokenizers, quantization, and chat-template conversion |
| Explicit matrices, multi-seed comparisons, paired deltas, factorial analysis, and per-case reports | Adequately powered general model selection, learning-to-threshold evidence, and open-ended response grading |
| Trainable token/byte Engram plus verified portable-table export and adapters | Generalized lexical/portable transfer or a consistent Engram benefit |
| Verified semantic retrieval of supplied vectors through the direct PyTorch API | Natural-language query encoding, standard trainer/chat integration, or semantic quality claims |
| Context passed in the conversation | Tool execution, long-lived user memory, and a secure application permission boundary |
| Verified bounded PyTorch KV caches and measured native MLX sparse components | Compatibility checks in additional execution environments, fused kernels, and production-serving guarantees |
| Shape-only CLI estimates, isolated smoke/warmup pilots, and backend-specific measurements | Measured large-model fit or a general hardware-capacity/performance guarantee |
| One host/device per experiment; independent local/SSH queues, leases, cancellation, recovery | Actual ROCm/XPU and cross-host acceptance; distributed training remains out of scope |
| MLX checkpoints, continuation, promotion, generation/chat, and held-out evidence for FP32 dense/native-sparse models | Blanket PyTorch feature or training-trajectory equivalence |

The [context/Engram](history/context-engram-study.md#execution-results--2026-09-22) and [domain-adaptation](history/path-domain-corpus.md#2026-09-22-execution-record) results show why these boundaries matter: untouched override scores stayed 0/8, while adaptation improved acquisition without reliable held-out behavior and sharply damaged retention. The [research roadmap](research/roadmap.md) separates exercised paths from open transfer, useful-task, and hardware evidence; the [feature roadmap](../ROADMAP.md) summarizes implementation direction.

Build these in response to measured bottlenecks. For broad assistant quality sooner, adapting a properly licensed pretrained instruction model is a different route from learning every capability from scratch. The current framework has no general Hugging Face weight/tokenizer/chat-template importer; do not point `--promote` at arbitrary downloaded weights and assume compatibility. Use an appropriate existing stack for that route, or implement and validate the exact architecture/tokenizer/checkpoint mapping here.

A narrow application additionally needs privacy/security controls, input/output validation, predictable failure handling, human escalation, latency measurements and rollback. Learned prompts are not access controls. The research CLI is not a production assistant service.

## 10. A concrete definition of “useful”

Prefer: **“This checkpoint routes these three kinds of support request better than our baseline, on this held-out population, within this latency budget, with a human reviewing ambiguous cases.”**

Avoid: “It has 100M parameters, uses Engram, and said hello, so it is an assistant.”

Your loop is:

**Define the job → collect examples → freeze an independent test → train small → inspect failures → change one thing → compare → scale only when justified → deploy with a bounded contract.**

The alias exercise is the first flashcard in that process, not the destination.
