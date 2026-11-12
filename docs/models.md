# Inspecting models

How to find a trained checkpoint, try it, chat with it and look inside it.
None of these commands train or change a checkpoint.

## Find and verify a checkpoint

```sh
find "$KLAB_WORK_DIR" -type f -path '*/checkpoints/*/manifest.json'
RUNS=/data/klab/runs
RUN_ID=my-run
CHECKPOINT=step_00000020_gen_000001
klab checkpoint inspect "$RUNS/$RUN_ID/checkpoints/$CHECKPOINT" --json
klab checkpoint verify "$RUNS/$RUN_ID/checkpoints/$CHECKPOINT" --json
```

Loading a model also checks the run's config, runtime, tokenizer and prepared
data, so keep the whole run directory; a weights file on its own is not enough.
`latest.json` and `best.json` move as training continues, so name a generation
when you want to come back to the same model.

To try every command here without a trained model, `klab readiness smoke`
builds a tiny one:

```sh
klab readiness smoke --family dense --output "$KLAB_WORK_DIR/smoke-1"
```

Its run is `readiness-dense` under `$KLAB_WORK_DIR/smoke-1/runs`, with a
64-token context and repetitive output.

## Continue a prompt

A base model continues text. Start with a prefix like its training data:

```sh
klab generate "$RUN_ID" --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" \
  --prompt 'def add(a, b):' --max-new-tokens 32 \
  --temperature 0 --seed 42 --strict-context --show-prompt --json
```

- **Decoding.** `--temperature 0` is greedy, a good first look. For sampling
  try `--temperature 0.7 --top-k 20 --seed 42`. The same seed can still differ
  across devices.
- **Stops.** `--stop TEXT` (repeatable) ends at a literal string.
- **Cache.** `--no-cache` recomputes the full prefix at each step instead of
  using the generation cache.
- **Context.** `--context-length N` caps the context. `--strict-context`
  refuses a prompt that would not fit with its completion, instead of sliding.
- **Output.** `--json` returns `text` (prompt and continuation) and `response`
  (continuation only). `--show-prompt` prints the exact prompt to stderr.

## Chat

```sh
klab chat "$RUN_ID" --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" \
  --message 'Write a function that adds two numbers.' \
  --system 'Answer briefly.' --max-new-tokens 32 --show-prompt --json
```

Without `--message`, `chat` opens a conversation in the terminal: `/reset`
clears the history and `/exit` ends it. `--transcript FILE` saves the session
(identity, prompts, settings) to a new file. The prompt uses the run's
[chat template](post-training.md#chat-templates). When the history no longer
fits, whole oldest exchanges are dropped; the system text and the current
message are never cut. A reply ends at EOS, the template's end of turn or a
`--stop` string.

A chat window does not teach a model to follow instructions. A base model may
repeat itself or imitate role labels; judge assistant behavior on a model
trained on conversations.

For programs, [serve the model over HTTP](local-api.md).

## Look inside

```sh
klab explore TRY_ID
klab explore RUN --config explore.yaml --json
```

`explore` reports a small checkpoint's architecture and parameter counts,
per-token loss and top predictions for a short text, attention maps, weight
statistics, expert routing and memory lookups. It runs on CPU unless you pass
`--runtime ID` or `--runtime-profile FILE`, and refuses checkpoints over 60
million parameters. Results are cached by checkpoint and text under
`LAB/explorer`; the dashboard's Explorer page shows them.

```yaml
explore_config_version: 1
text: Once upon a time, a little dog found a ball.
refresh: true
# lab_dir: ../lab
# runs_dir: ../runs
```

`text` is at most 128 tokens. `refresh: true` recomputes instead of reading the
cache, which is shared across devices. `--resource-envelope` applies resource
limits.

## Compare checkpoints

The dashboard's Behaviors page shows each result's probe generations side by
side with its baseline, along with recall hits and misses. Its Generate page
continues a prompt from a selected checkpoint. See the [dashboard](dashboard.md).
