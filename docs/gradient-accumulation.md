# Gradient accumulation

Gradient accumulation lets a run use several small microbatches for one optimizer update. It is a memory-management and optimization-boundary choice, not a disk checkpoint, a distributed collective, or a way to make several hosts share gradients.

## Configuration and meaning

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation: 4
```

The nominal effective example batch is `micro_batch_size × gradient_accumulation`. Model Lab builds an update window from deterministic epoch order, up to that many examples, stopping at an epoch boundary or token-budget boundary. A short final window is real: its actual valid next-token targets, not the nominal product, determine its loss denominator and token counters.

For each microbatch, the trainer computes summed next-token cross-entropy, ignores labels marked `-100`, and weights the auxiliary routing loss by that microbatch's share of valid targets. It divides the accumulated loss by the update window's total valid targets. This makes ordinary dense no-dropout accumulation comparable to a corresponding large batch while preserving correct masking at a partial final block.

## One successful update

For each window, Model Lab:

1. clears gradients once;
2. runs backward for every nonempty microbatch;
3. unscales once for FP16, clips global gradients once, sets the scheduled learning rate once, and steps the optimizer once;
4. advances cursor, update counter, and target counter only after a successful step.

FP16 overflow restores the window RNG/cursor state, reduces the scaler, and retries identical data. Three consecutive skipped attempts fail while retaining the last good checkpoint. Checkpoint cadence and interruption handling act only after a completed update window.

The metrics record `batch/micro_batch_size`, `batch/accumulation_steps`, actual `batch/effective_batch_size`, and `batch/effective_tokens_per_update`, alongside the successful update's loss, learning rate, and throughput. Read actual counts when comparing a short final window.

## What it does not mean

Accumulation does not reduce resident parameters, gradients, or optimizer state. It usually reduces peak activation size because each forward/backward sees a smaller microbatch, but it can increase update wall time because more microbatches are executed serially. It does not provide distributed data parallelism, expert exchange, model sharding, remote workers, or a KV decode cache.

MoE routing balance remains microbatch-local. The implementation masks invalid targets before routing summaries/auxiliary reductions, but it does not claim that accumulation recreates a whole-window global routing objective.

For the other resource mechanisms, see [memory accounting](memory.md), [activation recomputation](activation-checkpointing.md), and [disk checkpoints](checkpointing.md).
