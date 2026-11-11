# TinyText model guide: find, inspect and try a checkpoint

Commands below use the [activated, provisioned environment](../README.md#first-run).

Use this guide for human exploration of small text models: inspect their identity,
try a literal continuation, compare retained checkpoints, and only then try chat.
No notebook, Gradio installation or Open WebUI instance is required. This is the
inference part of the native [experiment learning cycle](research/experiment-learning-cycle.md):
retain the observation, then choose the next declared question. For interpreting
model size and recorded comparisons, use [model scaling and accounting](model-scaling.md);
for task-oriented learning, use [From flashcards to a local assistant](from-toy-to-useful.md).

**Naming boundary:** this checkout has no registered model, corpus declaration or
historical guide named `TinyText`. Do not substitute TinyStories or DevMind
MODEL-0 for an unidentified TinyText artifact. If TinyText is your local model's
name, use its actual run directory and checkpoint below; a display name is not
provenance. [TinyStories](tinystories-microlab.md) is the separate story-data
teaching workflow. [MODEL-0](model-0.md) is a completed DevMind base checkpoint.

## Try the complete path with a tiny offline fixture

If you have no retained model yet, this creates a tiny synthetic CPU fixture,
including a short training/resume check. It downloads no corpus and establishes
interface wiring, not text quality. Use a fresh `SMOKE` directory each time.
These commands were exercised against the fixture; its repetitive output is
expected at this training budget.

```sh
uv sync --locked --extra cpu --dev
export KLAB_WORK_DIR="${KLAB_WORK_DIR:-$HOME/.local/share/klab}"
SMOKE="$KLAB_WORK_DIR/text-interface-smoke-1"
klab readiness smoke --family dense --output "$SMOKE"
RUNS="$SMOKE/runs"
RUN_ID=readiness-dense
CHECKPOINT=step_00000002_gen_000002
GENERATION="$RUNS/$RUN_ID/checkpoints/$CHECKPOINT"
klab checkpoint inspect "$GENERATION" --json
klab checkpoint verify "$GENERATION" --json
klab generate "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --prompt 'hello' --max-new-tokens 4 --strict-context --show-prompt --json
klab chat "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --message 'Hi' --max-new-tokens 4 --show-prompt --json
```

The fixture has a 64-token context. Keep its chat questions and completion
budgets short; the longer examples below are for a selected checkpoint with
sufficient context. You can now use these variables in the [local API guide](local-api.md),
or compare its initial and step-2 generations:

```sh
klab surface chat \
  --cell initial="$RUNS/$RUN_ID/checkpoints/step_00000000_gen_000001" \
  --cell step2="$GENERATION" --backend cpu --seed 42 --port 8502
```

For a real teaching model, continue with the separately named
[TinyStories microlab](tinystories-microlab.md). For a retained research model,
follow the provenance and availability checks below.

## Locate the bytes before choosing an interface

Run commands from the repository root. These CPU examples use the locked CPU
extra; a provisioned accelerator environment must instead use the documented
[`--no-sync` and runtime authorization procedure](runtime.md#machine-local-runtime-environments).
Never sync the CPU extra into a vendor environment. Inference may be slow on CPU,
but does not train, resume training, promote a model or modify checkpoint files.

```sh
export KLAB_WORK_DIR="${KLAB_WORK_DIR:-$HOME/.local/share/klab}"
# List candidate retained manifests; this discovers paths, not verified models.
find "$KLAB_WORK_DIR" -type f -path '*/checkpoints/*/manifest.json' -print
```

Campaign ingestion can put a run under a Campaign's `controller/` directory,
not directly under `$KLAB_WORK_DIR/runs`. Use the run's actual parent for
`--runs-dir`. Read the experiment's checked-in Family, checkpoint reference and
result record when available; historical absolute paths are location hints,
not proof the files exist on this machine. See [MODEL-0's recorded location](model-0.md).

Set these three values to an existing run and immutable generation discovered
above. The values shown here are placeholders, not downloadable model names:

```sh
RUNS=/absolute/path/to/runs
RUN_ID=your-run-id
CHECKPOINT=step_00000020_gen_000001
GENERATION="$RUNS/$RUN_ID/checkpoints/$CHECKPOINT"
klab checkpoint inspect "$GENERATION" --json
klab checkpoint verify "$GENERATION" --json
```

`inspect` reports inventory and lineage; `verify` checks integrity and exits
nonzero for an invalid checkpoint. Verify the digest against the recorded
experiment identity. `latest.json` and `best.json` are lookup pointers; pin the
resolved generation when comparing results over time. Loading also validates
the run configuration, checkpoint runtime and run-owned tokenizer/prepared
assets. Keep the whole validated run closure, including `tokenizer.json`, data
manifest and required arrays/sidecars; a lone `.safetensors` file is insufficient.
Missing bytes are a recovery problem, not permission to refit the tokenizer or
retrain the model. [Checkpointing](checkpointing.md) and
[recovery](research/lifecycle-recovery.md) explain those boundaries.

## Saved inputs and historical runs

New dataset acquisition follows the canonical [dataset workflow](datasets.md):
a source declaration → `data lock` → `data snapshot` → `dataset.source: snapshot`
in the tokenizer/run configurations → tokenizer fitting and preparation. The
[TinyStories sample](tinystories-microlab.md) supplies a complete declaration-driven
example. Acquisition can download substantial bytes; it is separate from trying
an already retained checkpoint. Do not repeat acquisition or preparation merely
to open chat or start the server.

The retired `tinystories` and `local_stories` authoring forms remain readable for
historical verification and inference. New execution must use the supported
snapshot flow. `data migrate` writes a new identity; it does not repair an old
run in place. To import an existing legacy snapshot, use its authenticated
manifest and a fresh output directory:

```sh
klab data migrate /path/to/legacy/snapshot/manifest.json \
  --kind snapshot --output "$KLAB_WORK_DIR/imported-snapshot-1" --json
```

This preserves the original evidence and writes a generic snapshot plus a new
dataset binding. Direct-prefix migration from an old run config instead requires
an explicit resource policy and `--accept-policy-change`; its output remains
pending until the new source is locked and acquired. Follow
[legacy migration](datasets.md#legacy-input-migration), review its receipt and
rebuild/verify downstream inputs for new execution. Neither migration route
proves that old and new training inputs or results are interchangeable. A
verified historical checkpoint keeps using its original run-owned assets.

These three locations serve different purposes:

| Location | What it provides |
| --- | --- |
| Validated run directory | Checkpoints, resolved config, manifest, tokenizer and run-owned data/sidecars needed by inference |
| Array cache printed by `data prepare` | Prepared arrays for the declared inputs; not a standalone model or staging bundle |
| Sealed `PRIOR_STAGE/prepared` bundle | Authenticated inventory accepted by `stage --prepared-inputs`; not interchangeable with the array-cache path |

Missing run assets block inference even if unrelated prepared arrays exist.
Preserve manifests and location bindings; do not copy an arbitrary tokenizer or
array cache into a run to make validation pass.

## Begin with raw completion

A base model predicts continuations. Start with a short prefix resembling its
training data; asking a code base model to act as an assistant tests a different
behavior. This example sends the literal prompt without role labels:

```sh
klab generate "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --prompt 'def add(a, b):' --max-new-tokens 32 \
  --temperature 0 --top-k 0 --seed 42 --strict-context --show-prompt --json
```

Greedy decoding (`--temperature 0`) is a useful first diagnostic. For a second,
clearly labelled exploratory sample, try `--temperature 0.7 --top-k 20 --seed 42`.
Record the prompt, checkpoint digest, tokenizer, context limit, backend and
sampling settings together. The same seed does not promise identical output
across devices, kernels or tokenizers. `--stop 'END'` adds a literal stop string;
repeat it for more stops. `--no-cache` uses full-prefix decoding for comparison.
`--context-length N` caps context below the native limit. Prefer
`--strict-context` for raw debugging: it rejects an oversized prompt plus output
budget instead of silently using the legacy sliding-context behavior.

## Inspect the chat serialization

```sh
klab chat "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --message 'Write a function that adds two numbers.' \
  --system 'Answer briefly.' --max-new-tokens 32 \
  --temperature 0 --top-k 0 --seed 42 --show-prompt --json

# Omit --message for a terminal conversation; choose a fresh transcript path.
klab chat "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --max-new-tokens 32 --show-prompt \
  --transcript "$KLAB_WORK_DIR/chat-debug-session-1.json"
```

Chat uses this inspectable plain-text format, not an inferred tokenizer template:

```text
System: Answer briefly.

User: Write a function that adds two numbers.

Assistant:
```

`--show-prompt` prints the actual prompt as JSON to stderr (newlines are escaped);
response JSON remains on stdout. Raw `generate --json` exposes `text` (prompt
plus continuation) and `response` (continuation only).
Interactive `/reset` clears history and `/exit` or `/quit` finishes. The transcript
retains identity, prompts, settings and dropped-history counts and refuses to
overwrite an existing file. Context reserves the requested completion budget,
drops only whole oldest user/assistant pairs, and never crops the current user
message or system instruction. If those cannot fit, shorten them or lower the
output budget. EOS, role boundaries and explicit stops can end a turn early.

API/CLI support does not teach instruction following. Early base checkpoints
may repeat text, imitate role labels or ignore a question. Use raw continuations
for base behavior; use chat-trained/instruction-tuned checkpoints when assessing
assistant behavior, checking their actual training format and held-out results.
There is no parameter-count threshold that turns a base model into an assistant.

## Compare two to four retained checkpoints

```sh
# Replace SECOND_GENERATION with another real immutable generation path.
SECOND_GENERATION=/absolute/path/to/another-run/checkpoints/step_00000040_gen_000002
klab surface chat \
  --cell early="$GENERATION" --cell later="$SECOND_GENERATION" \
  --backend cpu --seed 42 --port 8502
```

Open the printed local Streamlit URL. This surface compares **raw prompt
continuations**, even though the command is named `surface chat`. Enter the same
prefix once, choose greedy or sampled decoding, output/context budgets, literal
stops and cache mode. It verifies every model's native tokenizer and context;
a prompt must fit all selected models. Response cards remain anonymous until
you finish and reveal identities. Diagnostic settings and exact input remain
inspectable without revealing model identities during the comparison.

Use a fixed prompt panel and record failed/empty/repetitive outputs as well as
interesting ones. Different tokenizers make token counts and equal token budgets
non-equivalent; different training data/budgets make causal claims unsafe.
Exploratory votes are not a sealed evaluation, population preference or a
promotion decision. For reviewing already generated, verified outputs, follow
[Surface Review](research/surface-review-v1.md). For client applications, continue
to the [local API guide](local-api.md).
