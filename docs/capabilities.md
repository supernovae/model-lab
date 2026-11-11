# Capability experiments

Commands below use the [activated, provisioned environment](../README.md#first-run).

SparseLab is a small-model workbench: train a narrow capability, exercise it through the same chat interface a person uses, and measure whether an architectural change helped. A trained checkpoint is reusable experimental material, not a disposable demo. Keep positive, negative, and inconclusive results.

The aliases below are invented flashcards, not useful world knowledge. For their plain-language meaning and the progression to ordinary chat and domain tasks, read [From flashcards to a useful local assistant](from-toy-to-useful.md).

## A runnable chat-native experiment

```sh
sparselab tokenizer train configs/tokenizer_chat_recall.yaml
sparselab inspect configs/chat_recall_dense_cpu.yaml --json
sparselab train --runs-dir sparselab-work/runs configs/chat_recall_dense_cpu.yaml --run-id chat-dense
sparselab train --runs-dir sparselab-work/runs configs/chat_recall_engram_cpu.yaml --run-id chat-engram
sparselab capability list
sparselab capability describe chat-alias-recall-v1
sparselab capability compare chat-dense chat-engram chat-alias-retention-v1 --runs-dir sparselab-work/runs
sparselab capability evaluate chat-dense chat-alias-recall-v1 --runs-dir sparselab-work/runs
sparselab capability compare chat-dense chat-engram chat-alias-recall-v1 --vary memory --runs-dir sparselab-work/runs
sparselab capability compare chat-dense chat-engram chat-context-override-v1 --vary memory --runs-dir sparselab-work/runs
sparselab chat chat-engram --system "Answer the requested alias with only its value." --max-new-tokens 12 --transcript sparselab-work/chat-engram.json --runs-dir sparselab-work/runs
```

These configurations deliberately train small models on a bounded task rather than promise general conversation. The dataset and evaluator share the ordinary `User:` / `Assistant:` formatter. There is no lookup-tool shortcut at inference: responses come from trained model logits. The tokenizer is trained on the training split only.

### What the cards measure

- **`chat-alias-retention-v1`:** training-seen query prompts for every learned alias. This is an acquisition sanity check, explicitly not generalization. Compare step zero to the trained checkpoint before interpreting the harder controls.
- **`chat-alias-recall-v1`:** learned fixed aliases under held-out query wording. The associations are intentionally in training; test wording is not. This measures access to learned information, not discovery of unseen facts.
- **`chat-context-override-v1`:** a conversation-local assignment overrides a learned alias. This control distinguishes static memorization from using supplied conversational context. Failure matters even when static recall succeeds.
- **`engram-recall-v1`:** historical wiring diagnostic. Its prompt contains the answer, and separate record IDs do not establish independent held-out wording. Its first-word scorer accepts extra text. Do not cite it as proof of recall generalization, chat ability, or Engram benefit. Use the newer full-answer chat cards for those narrow questions.

New cards contain frozen prompts, exact normalization/scoring, generation settings, limitations, and a digest. New chat cards use full normalized-answer equality: a correct first word followed by junk does not pass. Prompt plus completion budget must fit; evaluation refuses overflow instead of silently deleting evidence from the context. Balanced fixed-answer controls are reported beside model scores.

The bundled pair trains for up to 1,536 steps/393,216 targets. Epoch-end short batches can make the actual token count smaller when the step limit is reached; comparisons use checkpoint counters. The [recorded review experiment](project-review.md) reached 24/24 retained associations for both models but found no Engram advantage on the held-out or override cards. This is a useful negative architecture result, not a failed framework.

## Evidence and controlled comparisons

Each result retains every prompt, expected answer, generated response, score, and generation settings. Identity includes the exact resolved checkpoint digest, actual step and trained target tokens, run configuration, run-owned tokenizer and data hashes, code identity, runtime, and parameter inventory. Content-addressed filenames preserve earlier checkpoints and reruns instead of overwriting the latest score. The comparison itself is saved too.

```sh
sparselab capability evaluate chat-engram chat-alias-recall-v1 --checkpoint best.json --runs-dir sparselab-work/runs
sparselab report --kind run chat-engram --json --runs-dir sparselab-work/runs
```

`best.json` means best **validation loss**, not best capability score. Select a checkpoint on validation, not by searching test scores. For a learning curve, evaluate explicit generation directories at preregistered token budgets; preserve the step-zero result as the untrained control. Compare architectures at the same actual budget, not just the same configured maximum.

`capability compare` rejects mismatched card/protocol, seed, actual step/token budget, tokenizer contents, train/validation contents, source-code identity, and runtime. Paths and logging/cadence differences are not scientific differences. Select one permitted experimental axis:

| `--vary` | Permitted change | Interpretation |
|---|---|---|
| `memory` (default) | Engram configuration | Added memory may add total parameters; not a parameter-efficiency result. |
| `attention` | Attention configuration | Attention ablation, not a kernel speedup claim. |
| `ffn` | Dense/MoE feed-forward settings | Report total and active parameters; routing is not free. |
| `scale` | Backbone size | A declared scale comparison, not a mechanism-isolated result. |
| `none` | No scientific configuration changes | Repeatability control. |

Outputs show exact changed fields, paired gains/losses, score delta, and scope-limited interpretation. One seed, a few synthetic cases, or a positive delta is **not** statistical confirmation. Repeat a preregistered seed set, compare each seed to its matched counterpart, retain failures, and report the distribution of deltas. No automatic significance or universal superiority label is issued.

## Bring a new task without changing Python

Export a built-in card as a declarative starting point:

```sh
sparselab capability describe chat-alias-recall-v1
```

Save that JSON, remove its `digest` when editing, and choose a new versioned `name`, hypothesis, limitations, cases, expected responses, and response-token budget. Keep the declared normalization and greedy protocol; update the categorical-answer controls to reflect the cases. Pass the JSON path wherever a card name is accepted. The strict schema rejects unknown fields; custom executable scorers are not loaded. Changing prompts, scorer, or decoding creates a different digest. Never compare it as if it were the old card.

A useful task needs a training source and a separately held-out card. Use [local conversation JSONL](instruction-training.md#local-conversation-corpora) for your own data. Ensure the task's answers are not exposed in test prompts unless testing in-context use. Keep semantic/template overlap audits: distinct row IDs or file hashes alone do not prove generalization.

## Scale without losing the experiment

1. Start with a CPU pair and verify that at least one trained model improves over its step-zero score. If neither learns, fix corpus, context, or optimization before adding mechanisms.
2. Freeze the task/card, source split, tokenizer, optimizer, seed set, and target-token budget. Create a matched dense/Engram pair at the next backbone size; `inspect` the actual parameter counts.
3. Repeat each pair at declared budgets. Log both absolute task scores and paired deltas, held-out loss, memory, and synchronized training throughput. Do not tune on the held-out test set.
4. A parameter-efficiency question requires a separate matched-total-parameter design. A data-efficiency question requires fixed-model score-versus-token curves. Neither follows automatically from adding a table.
5. Add capabilities one at a time: collision stress, longer-distance context, compositional lookup, and realistic domain conversations need their own frozen cards and controls.

KV-cached PyTorch decoding and native MLX sparse attention are available within their documented boundaries. Neither implies arbitrary-model compatibility or a universal performance advantage. Capability comparisons must retain optimizer semantics, engine/backend, precision, checkpoint and tokenizer identities, actual token budgets, and the complete declared case set; execution support is not task competence.

## Additional task and review workflows

Set `SPARSELAB_WORK_DIR` to your external persistent root before these commands.
Here `RUN_ID`, `base` and `variant` stand for existing runs in
`$SPARSELAB_WORK_DIR/runs`; pass their actual store if it differs. The chat pair
above explicitly uses `sparselab-work/runs` instead. Use new output destinations
for each task bundle.

Inspect the outcome groups with `sparselab capability suite`. Held-out language-model loss remains separate from exact-answer behavior-card scores. The suite has distinct cards for canonical lexical recall, paraphrase, multi-turn follow-up, Python API behavior, novel-operand math, application, composition, long-context retrieval, conversation override, instruction-over-memory, and stale/conflicting/missing evidence. Lexical, paraphrase, and conversation cards share the synthetic alias domain; treat them as task conditions, not independent knowledge samples. Wikidata factual and paraphrase cards are generated from the explicitly built local source bundle.

Build the deterministic synthetic task splits without network access:

```sh
sparselab research tasks build math-identities --output "$SPARSELAB_WORK_DIR/experiments/capability-tasks/math"
sparselab research tasks build python-stdlib --output "$SPARSELAB_WORK_DIR/experiments/capability-tasks/python"
```

Each bundle contains `train.jsonl`, `validation.jsonl`, a held-out test card under `cards/`, test cases, and a hash-bound manifest. The JSONL files use the existing `local_chat` format; configure `dataset.source: local_chat`, point `train_path` and `validation_path` at the respective files, and declare the manifest's `MIT` license. Math identities use disjoint operand pools across train, validation, and test. Python tasks compute outputs through a finite allowlist of trusted standard-library calls; they never evaluate generated source, execute user code, or execute a benchmark project. No benchmark solutions are imported or extracted.

The optional factual miniature downloads data only when explicitly requested:

```sh
sparselab research tasks build wikidata-mini --output "$SPARSELAB_WORK_DIR/experiments/capability-tasks/wikidata"
sparselab capability evaluate RUN_ID "$SPARSELAB_WORK_DIR/experiments/capability-tasks/wikidata/cards/wikidata-mini-factual-recall-v1.json" --runs-dir "$SPARSELAB_WORK_DIR/runs"
```

The answer-free packaged source manifest pins Q42 and Q937 to training, Q7259 to validation, and Q7186 to test, each at a fixed Wikidata revision. The builder makes four serial requests to the revision-specific `Special:EntityData` endpoint, sends a descriptive User-Agent, caps each response at 8 MiB, and does not retry a rate-limited request. It downloads each complete entity JSON, extracts an English label or (when absent) a language-neutral `mul` label recorded as `label_language`, and extracts P569/P570 dates at day precision. It verifies each entity ID and revision, hashes the canonical response, and retains no raw entity JSON. Wikidata structured data is [CC0](https://www.wikidata.org/wiki/Wikidata:Reuse); the [data-access guidance](https://www.wikidata.org/wiki/Wikidata:Data_access) recommends specific revisions and considerate request rates. The manifest labels source facts `CC0-1.0` and generated prompts/format `MIT`; set the combined local-chat license to `MIT prompts/format; CC0-1.0 Wikidata facts`. Train/validation/test membership is entity-disjoint, and test answers are absent from training. This four-entity fixture is not a broad factual-knowledge benchmark.

Mine lexical statistics using only the explicitly supplied training corpus:

```sh
sparselab research corpus mine \
  --train-jsonl "$SPARSELAB_WORK_DIR/experiments/capability-tasks/math/train.jsonl" \
  --tokenizer "$SPARSELAB_WORK_DIR/runs/RUN_ID/tokenizer/tokenizer.json" \
  --table-size 65536 --memory-dim 64 --ngram-orders 2 4 --hash-heads 1 \
  --output "$SPARSELAB_WORK_DIR/experiments/capability-tasks/math-lexical-analysis.json"
```

The report binds the train-file and tokenizer hashes and contains token/document frequencies, empirical unigram entropy, exact per-order/head address occupancy, and storage estimates. Collisions count distinct zero-padded token n-gram keys that share an address (`collisions / distinct_ngram_count`); repeat lookups are reported separately. Exact key tracking is bounded to one million distinct keys and eight million key components. The estimator opens one explicit training JSONL and no validation/test path. It tokenizes each rendered train conversation independently, without packing, EOS insertion, or trainer truncation; address statistics are a source-document estimate, not the exact packed training run. Its table-byte estimate excludes the output projection and gate.

For addresses as seen by the trainer, use the native prepared-data directory:

```sh
sparselab research corpus mine \
  --prepared-train "$PREPARED_DATA" \
  --sequence-length 256 --max-blocks 128 \
  --table-size 65536 --memory-dim 64 --ngram-orders 2 3 --hash-heads 2 \
  --output "$SPARSELAB_WORK_DIR/packed-train-lexical-analysis.json"
```

This mode cold-verifies the complete prepared closure (including held-out file
integrity), then analyzes only the first eligible TRAIN input blocks. EOS inside
blocks is preserved; hash context resets at each block, exactly as in a batched
training forward. Loss-masked input positions still perform lookups, while blocks
with no supervised targets are excluded by the native dataset. Use the actual
training sequence length. The report binds the prepared manifest and tokenizer,
records selected block indices and reports block frequencies instead of document
frequencies. A bounded prefix is a sample, not the shuffled training schedule;
even a complete pass does not account for repeated epochs. Byte-addressed and
allocation-sidecar data are refused. No gate behavior or useful knowledge transfer
is inferred from these address counts. The canonical tokenizer identity comes from
the verified prepared manifest; token IDs are checked as nonnegative int32, but
this analysis does not verify their upper bound against a model vocabulary.

Create a blinded comparison directly from content-addressed capability results:

```sh
sparselab review bundle \
  --base-result "$SPARSELAB_WORK_DIR/runs/base/evaluations/BASE_RESULT.json" \
  --variant-result "$SPARSELAB_WORK_DIR/runs/variant/evaluations/VARIANT_RESULT.json" \
  --criteria rubric.json --seed 17 \
  --bundle artifacts/review/rater-bundle.json \
  --reveal-map private/reveal-map.json
sparselab review validate \
  --bundle artifacts/review/rater-bundle.json \
  --judgments artifacts/review/judgments.json \
  --output artifacts/review/validated-judgments.json
```

The bundle builder verifies both result digests and requires a shared card, evaluator, case set, prompts, and expected answers. It randomizes case order and A/B assignment, but keeps condition names, result identities, and case identifiers only in the separate reveal map. The rater bundle binds its seed, criteria digest, and reveal-map digest; give raters only that bundle. `--records` is also available for paired responses that do not come from capability results. Judgments bind to the bundle and criteria, cover every blind case exactly once, and contain no reveal fields. Validation only normalizes judgments; it does not score, rank, select, or tune models. Do not use held-out test scores or unblinded judgments in a hidden optimization loop.

Packed mining also refuses more than one million candidate blocks before native
block enumeration, regardless of `--max-blocks`. Use a bounded prepared subset for
larger corpora. Full closure verification still reads the declared artifact files.

Token-memory mechanisms, tokenizer-bound maps, multiple ports and explicit dense-checkpoint migration are documented in [Context-gated n-gram memory](ngram-memory.md).


### Optional Python code-work scoring

The optional Inspect/Docker code-work adapter keeps generated Python inside its
sandbox and refuses missing dependencies or images without installing or pulling
them. Its completion marker is supplied through stdin, not argv or environment,
so echoing a process command line cannot forge successful test completion.

This is **non-adversarial functional scoring**, not an adversarially qualified
judge. Generated code and tests share a Python interpreter: a solution can inspect
or alter harness state, and tests are visible inside the container. Container
isolation protects the host but does not establish score integrity. Results carry
this assumption and limitation; do not use these scores as trusted evidence for
adversarial or harness-aware generated code. A separately qualified trusted-test
execution protocol is needed for that use case.
