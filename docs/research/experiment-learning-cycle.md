# Experiment learning cycle

Model Lab is meant to help turn model work into progressively better-understood systems, not a stream of disconnected PASS/FAIL runs. A model can learn, remain limited, expose a regression, or answer a question negatively without becoming useless evidence.

This guide is the default loop for moving from a known-good reference to a controlled follow-up.

Automatic [post-train triage](post-train-triage.md) reads retained evidence and
performs only bounded diagnostics. Its post-hoc recommendations are not frozen
acceptance gates, research decisions, or automatically dispatched experiments.

## The loop

```text
known-good reference
        |
        v
state one question
        |
        v
choose the cheapest discriminating experiment
        |
        v
freeze the contract
        |
        v
run and observe
        |
        v
separate integrity / learning / behavior / generalization / efficiency
        |
        v
record a Finding
        |
        +--> fix a hard failure
        +--> stop: question answered negatively
        +--> replicate
        +--> scale one dimension
        +--> promote a bounded reference
        +--> branch to the next question
```

The important distinction is between **an experiment that failed to answer its question** and **an experiment that answered its question negatively**. A corrupt checkpoint is broken evidence. A valid run showing that memory did not recover a narrower FFN is a useful negative result.

## 1. Start from something known

Before changing an architecture, establish the parent you are trying to understand or preserve.

A known-good baseline is deliberately bounded. It may be an integration reference, a learning reference, a behavior reference, or eventually a useful-task/efficiency reference. It does not need to be a great chatbot.

The current worked example is [`dense-lm-v1`](dense-lm-v1.md), a promoted TinyStories **learning reference**. Its three declared seeds each reached 4,096 updates / 4,194,304 targets and reduced held-out loss throughout the frozen schedule. The promotion does not claim convergence, chat quality, architecture superiority, or a useful real-world task.

Use:

```sh
uv run --locked klab report --kind research --json
uv run --locked klab research baseline describe dense-lm-v1
uv run --locked klab research validate --baseline dense-lm-v1
```

A historical or unrelated registry diagnostic can remain visible without changing the candidate-specific evidence state.

## 2. Ask one question, not "make it better"

Prefer questions that distinguish explanations.

Good:

- Does the current dense model still improve when only the token budget increases?
- At a matched training budget, what does a larger dense backbone buy?
- Can lexical memory preserve a capability when FFN width is reduced?
- Does earlier memory placement help composition?

Too broad:

- Can Engram make the model better?
- What is the best architecture?
- Can I optimize everything at once?

Write down what stays fixed, what changes, and what observation would weaken the hypothesis before starting the expensive run.

## 3. Choose the cheapest discriminating experiment

Do not jump directly to a larger model or a combined architecture.

Examples:

| Observation | Cheapest useful follow-up |
|---|---|
| Loss is still falling at the endpoint | Measure independent behavior first; consider an explicit target-budget extension only if comparable target behavior also improves |
| Loss plateaus while behavior remains weak | Review capacity, data and task hypotheses before proposing a larger model |
| Training is healthy but greedy text repeats | Run a separate decoding/behavior study |
| One seed looks unusual | Replicate the frozen configuration |
| An architecture candidate regresses the parent | Isolate the changed mechanism before adding more |
| A smoke cannot learn the task at all | Fix the task/optimization before architecture claims |

The goal is not to prove your preferred explanation. It is to eliminate alternatives cheaply.

## 4. Freeze the contract

Before execution, bind the things that must not move after results become visible:

- question and primary observation;
- parent/reference;
- model/config identity;
- tokenizer and data identity;
- seed policy;
- optimizer, sequence length and effective batch;
- target/step endpoint;
- checkpoint/evaluation milestones;
- applicable capability cards;
- fixed generation panel where used;
- hard stop conditions;
- promotion/replication rule.

Do not turn a post-hoc observation into a retroactive gate. The first `dense-lm-v1` seed exposed visible repetition, but no repetition threshold or human-review protocol had been frozen. The repetition was therefore retained as a limitation rather than used to rewrite the promotion contract.

## 5. Keep one experiment in one workspace

Mutable state belongs under one experiment root:

```text
klab-work/
  experiments/
    <experiment-id>/
      receipt.json
      runs/
      staging/
      exercises/
      captures/
      local-reports/
```

Seeds, model widths, memory settings, and other matrix cells are coordinates of one experiment, not peer directories in the repository root.

Project source and declarations stay in Git. Compact durable evidence may be promoted under `artifacts/`. Large mutable training state stays in the ignored workspace unless there is an explicit retention reason.

## 6. Observe multiple dimensions

Do not reduce a model to one number.

### Integrity

Did the intended experiment actually run?

Examples: source identity, tokenizer/data hashes, finite tensors, valid checkpoint, matched comparison identity, no leakage.

An integrity failure can make the scientific result uninterpretable.

### Learning

Did optimization move the model in the intended direction?

Examples: held-out loss trajectory, task acquisition, milestone behavior.

### Behavior

What does the checkpoint actually do?

Examples: fixed prompt outputs, repetition, context use, retrieval, composition.

A lower LM loss is not the same thing as a correct behavior.

### Generalization

Does the observation survive held-out wording, seeds, worlds, tasks, data or scale?

### Efficiency

How much did it cost?

Keep target tokens, parameters/active work, estimated FLOPs, device memory, optimizer/update time and end-to-end wall time distinct.

### Reproducibility

Does the declared observation recur when the frozen experiment is repeated?

## 7. Classify what happened without forcing good/bad

Useful outcome language:

- **BROKEN** — an integrity/runtime/data problem prevents interpretation.
- **DID_NOT_LEARN** — the valid baseline never acquired the phenomenon needed for the question.
- **LEARNED_BUT_LIMITED** — clear learning exists, but one or more behaviors remain weak.
- **MIXED** — seeds/tasks/conditions disagree.
- **REGRESSION** — a descendant loses a capability required by its parent contract.
- **REPLICATED** — the declared phenomenon recurs under the repetition policy.
- **PROMOTABLE** — the candidate meets the bounded purpose of a reference.

These are explanatory labels, not an automatic leaderboard.

A genuine negative hypothesis result stays negative. "We learned something" does not mean "the architecture worked."

## 8. Turn limitations into the next question

For a limitation, list plausible explanations without pretending to know which one is true.

Example:

```text
observation:
  repetitive story continuation

possible causes:
  undertraining
  model capacity
  narrow corpus behavior
  decoding policy

cheapest checks:
  inspect existing milestones
  compare declared seeds
  extend training budget only
  test decoding separately
  scale the model only after those
```

This is the core of the learning cycle: an observation becomes a narrower question.

## 9. Preserve failures and stop repeating known dead ends

Every completed study should leave behind enough durable evidence to answer:

- what was tested;
- under which identities/conditions;
- what was observed;
- what claims are supported;
- what is still unknown;
- whether the next action is reject, learning, replicate, scale or promote;
- what would legitimately reopen a negative result.

A prior negative finding should warn against rerunning the same design unchanged. It should not become a universal law about another model scale, task or mechanism.

## 10. Promote bounded references, not winners

Promotion means:

> we know what this model/configuration does well enough to use it as a parent.

It does **not** mean:

> this is the best model.

The first attempted dense synthetic alias reference was retained unpromoted because its held-out validation gate failed. The later `dense-lm-v1` TinyStories campaign replicated its bounded learning behavior across three seeds and was promoted as a `learning_reference`.

That gives later experiments a known parent:

```text
dense-lm-v1
    |
    +-- token-budget study
    +-- scale study
    +-- MLA descendant
    +-- Engram descendant
    +-- MoE descendant
    +-- sparse-attention descendant
```

Each branch must say which parent properties it intends to preserve and which resource or capability it is trying to change.

## Beginner checklist

Before a run:

- [ ] What is my parent/reference?
- [ ] What single question am I asking?
- [ ] What stays fixed?
- [ ] What changes?
- [ ] What would make the result uninterpretable?
- [ ] What observation would weaken my hypothesis?
- [ ] What is the cheapest useful scale?
- [ ] Are my evaluation tasks applicable to this data/model?
- [ ] Is the run in one experiment workspace?

After a run:

- [ ] Did integrity pass?
- [ ] Did the model learn?
- [ ] What behavior changed?
- [ ] What regressed relative to the parent?
- [ ] What remains unknown?
- [ ] Is the result broken, negative, limited, mixed or replicated?
- [ ] Should I fix, stop, replicate, scale, branch or promote?
- [ ] Did I preserve the Finding and evidence before cleaning mutable run state?

## Worked progression: dense-lm-v1

The dense reference is useful because it shows the loop without an architecture novelty:

1. A synthetic dense candidate exercised the lifecycle but failed its held-out validation gate. It stayed unpromoted.
2. A plain dense TinyStories candidate was trained under a frozen schedule.
3. Seed 42 learned strongly but exposed repetitive text. A post-hoc subjective rejection was audited and removed because it was not a frozen gate.
4. Seeds 17 and 73 repeated the bounded learning result while retaining their own generation limitations.
5. The three-seed candidate was formally promoted as a bounded learning reference.
6. Because all endpoints were reached while held-out loss was still decreasing, the next question became **training maturity**, not "add a clever architecture immediately."

See [known-good baselines](known-good-baselines.md), [evidence and reporting](evidence-and-reporting.md), [research roadmap](roadmap.md), and [experiment methods](../experiments.md) for the underlying contracts.

The point of the loop is simple: **turn unknowns into smaller knowns, and only spend more compute when the previous result justifies the next question.**
