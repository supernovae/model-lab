# Multi-device training on two Arc Pro B70s

Measured 2026-11-11 on one host with two Intel Arc Pro B70 cards (`xpu:0` at PCI
`0000:23:00.0`, `xpu:1` at `0000:e3:00.0`), `torch 2.15.0+xpu`, starting from main
`e552e4a`. These are short throughput runs on synthetic data: they say how busy
the devices are, not how well a model trains.

## Workload

- Dense decoder, 12 layers, width 1024, 16 heads, FFN 2816, vocabulary 16,384:
  170.9M parameters.
- Context 4,096, microbatch 4, accumulation 8: 131,072 targets per update.
- bf16, SDPA attention, AdamW. "Compiled" adds `compile_transformer_blocks` and
  `compile_output_head_loss`.
- 14 updates per run; throughput is the median of updates 5 to 14 and
  "end to end" divides targets by wall time between those updates.

Device busy time was read from the driver with `benchmarks/device_busy.py`.

## Results

| Run | Engine tokens/s | End to end tokens/s | Peak device memory |
|---|---:|---:|---:|
| 1 card, eager | 22,204 | 22,051 | 17.1 GiB |
| 1 card, compiled | 33,006 | 32,919 | 10.2 GiB |
| 2 cards, eager, before | 43,766 | 43,173 | 17.1 GiB |
| 2 cards, eager, after | 44,488 | 44,075 | 17.1 GiB |
| 2 cards, compiled, before | 33,523 | 33,190 | 10.2 GiB |
| 2 cards, compiled, after | 65,499 | 65,293 | 10.2 GiB |

Two compiled cards now deliver 1.98 times one compiled card end to end; before
they delivered 1.01 times.

## What was wrong

**Compiled replicas ran one device at a time.** With compiled kernels the engine
issued every replica's slice from one thread and synchronized once, assuming
kernels queue without blocking. On this XPU build, embedding backward reads a
count back from the device once per microbatch (`aten::item` under
`copy_if_xpu` in `embedding_dense_backward`), so each slice ran to completion
before the next began. The driver timeline showed the two cards strictly
alternating. Each replica now runs on its own thread, as eager replicas did.

**The exchange sat on the primary's critical path.** After every update the
primary copied all weights to each replica and pulled all gradients back, then
clipped and stepped, while the replicas waited. Each replica thread now copies
the weights when its slice starts and moves its gradients to the primary when it
ends, both while the primary is still computing. At this model size the old flat
copies cost about 18 ms out and 20 ms back per replica; they scale with
parameter count.

**Reading the next batch was on the critical path.** Assembling an update's
blocks from the memory-mapped token file took about 20 ms per update with cold
pages, with every device idle. The trainer now asks the kernel to read the next
update's blocks while the current update computes. Assembly measured 3.7 ms per
update afterwards; part of that difference may be a warmer page cache.

## What is left, per update of two seconds

| Item | Time |
|---|---:|
| The faster card waits for the slower one | 72 ms |
| Gradient clipping and optimizer step on the primary | about 40 ms |
| Merging collected gradients | 14 ms |
| Metrics and progress writes | about 22 ms |

- The two cards are not equally fast: `xpu:0` took 77.8 ms and `xpu:1` 74.2 ms
  for the same compute probe. Equal shares therefore leave the faster card
  waiting. `performance/device_wait_seconds` reports this in every run.
- These fixed costs matter more as updates get shorter. With four cards and the
  same accumulation, an update is about one second, so use enough microbatches
  per update that each device's share is long compared with them.

## The single card is not host-bound

One core is busy during training because PyTorch's autograd thread spin-waits
in the Level Zero driver for device results. The driver timeline shows the card
continuously busy in both eager and compiled single-card runs, so that core is
not holding the card back and more host threads would not help.

## Reproduce

```sh
klab data prepare CONFIG.yaml
klab train --runtime RUNTIME_ID CONFIG.yaml --run-id RUN --require-prepared &
python benchmarks/device_busy.py record TRAINING_PID trace.json
python benchmarks/device_busy.py show trace.json
```

Use the PID of the Python process that holds the device, not the launcher.
Add `runtime.secondary_device_indices: [1]` for the second card.
