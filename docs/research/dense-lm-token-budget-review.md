# Did more training help the small story model?

**Short answer:** On a fixed reading test, yes. On a handful of story-writing prompts, not reliably. We did not replace the current reference model or claim that the longer-trained model is generally better.

## What we compared

The starting point was a small model trained on TinyStories text. We took three independently initialized versions (called *seeds* 42, 17 and 73), each already trained for 4.19 million **prediction targets**. A target is one next-token prediction during training, not a new word or a new story. We continued each version to 8.39 million and then 16.78 million targets.

The training stories, tokenizer, model shape, batch size, precision, hardware and validation text stayed the same. We restored each model's weights **and** training state rather than starting over. Its learning rate had already tapered down; all extra updates used the existing low rate. The same prepared stories were revisited across training passes. Thus this asks what *more practice on the same material at that low rate* does, not what more unique data, a bigger model or a new training recipe would do.

Before running the longer training, we recorded the comparison rules and exact input fingerprints in the [protocol](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-token-budget-v1/protocol.md) and [preregistration](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-token-budget-v1/preregistration.md). The result from seed 42 met the predeclared threshold for repeating the study, so we also measured seeds 17 and 73 at both longer budgets.

## What we saw

On the same held-out text, the model's next-token prediction error (*validation loss*) fell for every seed. Lower is better on **this specific test**:

| Starting seed | After 4.19M targets | After 8.39M targets | After 16.78M targets |
|---|---:|---:|---:|
| 42 | 2.482 | 2.369 | 2.243 |
| 17 | 2.485 | 2.370 | 2.244 |
| 73 | 2.470 | 2.360 | 2.234 |

We also gave each saved model the same six unfinished story prompts and asked it to continue without randomness. Some passages improved. For example, the original seed-42 boat story introduced *Anna* and *Ben* after establishing Mia and Leo; the longer continuations kept Mia and Leo. But the 16.78M seed-42 and seed-17 continuations called a meadow **blue** even though the prompt described it as **green**. Seed 73's rabbit continuation became more repetitive as training continued. The test's automatic “eligible” flag checks basic output mechanics, not whether a story is sensible; the contradictory and repetitive outputs remain in the record.

All three loss curves were still improving between the two longer checkpoints under our predeclared definition. That does **not** establish that unlimited training will keep helping, or that a lower loss reliably produces a better story. The writing check has only six fixed prompts; it is not a survey of general language ability or chat quality.

## What the work cost and what went wrong

The first additional 4.19M targets took roughly **509–513 seconds per training command** on one AMD Radeon RX 7900 XTX; the following 8.39M took roughly **1,013–1,058 seconds**. That includes loading, validation, saving and reporting—not just computing weight updates. Peak recorded GPU allocation in the successful runs was about **1.06 GB**, with about **1.12 GB reserved** by the framework. These are measurements on this device, not forecasts for other hardware.

One initial seed-42 attempt reached the expected update count but missed 640 targets at a training-data pass boundary. The evaluation command rejected it. We preserved that failed attempt, corrected how training assembles a complete update across the boundary, and reran from the unchanged original checkpoint. Only the corrected run appears in the loss table. This matters because an apparently finished command is not enough to prove a fair comparison.

## Review decision

**Keep the original reference and the longer runs as separate evidence.** More practice at the fixed low learning rate consistently reduced held-out prediction error on these three seeds, while some individual story continuations worsened. There is no automatic promotion of a longer checkpoint. A future claim about better writing needs an independently planned, broader evaluation; a claim about new information needs a different experiment with new training data.

For exact outputs, all six prompt-by-prompt repetition measurements, checkpoint identities, runtime phases, and the preserved failed run, read the [technical results](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-token-budget-v1/results.md) and [machine-readable observations](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/dense_lm_token_budget_v1.json). Full checkpoints and prepared data stay in the ignored local workspace, not in GitHub.
