# Chat-oriented training

Commands below use the [activated, provisioned environment](../README.md#first-run).

Start with [From flashcards to a useful local assistant](from-toy-to-useful.md) for the beginner explanation, the runnable 3.3M `instruction_starter.yaml` recipe, and the distinction between a familiar chat prompt and trained conversational ability.

The optional `instruction_reference` source is a deterministic, offline synthetic curriculum for exercising the local chat transcript path. Each document uses the same plain-text form consumed by `klab chat`:

```text
System: You are a concise local assistant.

User: Explain causal attention in one sentence.

Assistant: Causal attention lets each position use earlier positions but not future ones.
```

The curriculum contains bounded arithmetic, word reversal, attribute lookup, concise definitions, sentence completion, and greetings. Its train and validation index ranges differ, but finite task instances/templates can repeat; this is a format/curriculum diagnostic, not a generalization benchmark. For a smaller, measured starting point, use the [chat recall capability pair](capabilities.md) with genuinely held-out query combinations.

## Reference run

```sh
klab tokenizer train configs/tokenizer_instruction_8k.yaml
klab data prepare configs/instruction_100m.yaml
klab inspect configs/instruction_100m.yaml --json
klab train --runs-dir klab-work/runs configs/instruction_100m.yaml --run-id instruction-100m-pilot --stop-after-step 2
klab checkpoint verify klab-work/runs/instruction-100m-pilot/checkpoints/latest.json --json
klab train --runs-dir klab-work/runs configs/instruction_100m.yaml --run-id instruction-100m
klab chat instruction-100m --system "You are a concise local assistant." --max-new-tokens 32 --runs-dir klab-work/runs
```

`instruction_100m.yaml` is the initial larger reference: 104,843,648 dense parameters, a 20-million-target-token curriculum budget, effective batch size 16, and block activation checkpointing. It is intentionally a 100M—not 200M—configuration: establish that the corpus, transcript, checkpoint, and held-out behavior are useful before paying the substantially higher memory and training cost of a 200M experiment.

The direct two-update pilot is a real execution/checkpoint check, not a quality result or an implicit warmup for the fresh run. `stage --through warmup` now runs a separate disposable measured pilot; it does not reuse that pilot's weights, optimizer, schedule, counters, or RNG for this reference run. The 100M preset is not a pretrained assistant: its corpus contains repetitive `Reference N:` tasks, and the full training/evaluation must still be performed before claiming useful chat. Its 256-token context includes the system prompt, history, current question, role labels and reserved answer.

## Evaluation boundary

Use chat to inspect behavior, but do not cite a fluent-looking reply as an instruction-following result. Compare the trained run with a matched dense base run where feasible and retain prompts, exact transcript formatting, generation settings, tokenizer, model configuration, token budget, and checkpoint identity. A future conversational corpus must have explicit license/provenance and a held-out multi-turn evaluation before it replaces or extends this synthetic reference.

## Local conversation corpora

Use `dataset.source: local_chat` for a corpus you own or are licensed to use. Supply distinct UTF-8 JSONL files, an explicit license declaration, and acquisition budgets. Paths are relative to the configuration file. Use the same dataset section in your tokenizer-training and model-training configs:

```yaml
dataset:
  source: local_chat
  train_path: ../data/domain-train.jsonl
  validation_path: ../data/domain-validation.jsonl
  license: CC0-1.0
  cache_dir: ../artifacts/data
  train_max_documents: 10000
  validation_max_documents: 1000
  train_max_tokens: 1000000
  validation_max_tokens: 100000
```

Each line is a complete conversation. Historical unversioned records remain valid and
keep their exact plain-text rendering and whole-transcript loss:

```json
{"messages":[{"role":"system","content":"Answer from the workshop handbook."},{"role":"user","content":"Which drawer holds the torque wrench?"},{"role":"assistant","content":"The labeled blue drawer."}]}
```

New corpora must use format version 2 and choose `all_tokens` or
`assistant_only` explicitly. In `assistant_only`, user/system/tool-result text and
role delimiters are context only; assistant text, serialized tool calls, and the
terminal EOS receive loss.

```json
{"format_version":2,"loss_mode":"assistant_only","messages":[{"role":"user","content":"Look up the drawer."},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","name":"lookup_drawer","arguments":{"item":"torque wrench"}}]},{"role":"tool","tool_call_id":"call-1","content":"blue"},{"role":"assistant","content":"The labeled blue drawer."}]}
```

V2 accepts only the fields shown for each role. A system message is first; user
turns and complete final assistant answers alternate. Tool-call IDs are unique;
their results occur exactly once and in call order before a final assistant answer.
Tool-call arguments and tool-result records use finite canonical JSON. Their
explicitly versioned `ToolCalls[v1]`/`ToolResult[v1]` text protocol escapes embedded
newlines and restores the assistant role after the final result; nothing is executed.
Packing persists boolean supervision masks alongside token IDs. Only tokens wholly
inside assistant spans are supervised: a token crossing a role boundary is excluded.
EOS is supervised for assistant-only records only when the complete record fits;
a truncated user prompt never becomes an EOS training target.

Prepared-data identity includes file contents, not just filenames. Changed data cannot reuse stale token arrays. Exact duplicate conversations across train and validation are rejected; this does not detect paraphrases, overlapping facts, or near-duplicates. Split by the unit relevant to your hypothesis (document, customer, task family, or template) before ingestion. Keep separate test cases in a capability card; never put validation examples into the test card.

Tokenizers are immutable artifacts. Training into an existing output directory succeeds only if source-content/training provenance and the tokenizer digest match. Otherwise choose a new tokenizer output directory. Referencing an already-trained tokenizer in a model config remains valid, including a pretrained tokenizer from another corpus.

V2 local-chat records support either whole-transcript or assistant-only next-token
loss, including inert tool-call transcripts; they do not execute functions, implement
preference optimization, or import a hidden commercial chat template. Start with short,
verifiable responses; train and evaluate the same system prefix. For a new domain, copy
the small chat configuration, change dataset/tokenizer paths and budgets, inspect
resource usage, and evaluate a separately versioned task card before increasing model size.

## Verified supervised runtime path

The acceptance run exercised actual CPU and MLX CLI training, evaluation, and evidence
commands on a disposable v2 fixture. Each committed exactly 13 supervised
training targets in one update. Evaluation scored 33 targets from a 398-token
validation stream while excluding 19 wholly masked blocks; both produced
`checkpointed_held_out` evidence bound to the supervision-mask digest.
[The acceptance record](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/assistant_only_2026_09_22.json)
is an objective/packing/runtime check, not a domain-quality result.
