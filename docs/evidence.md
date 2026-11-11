# Experiment evidence

Commands below use the [activated, provisioned environment](../README.md#first-run).

Model Lab is intended to make a small experiment inspectable, not to convert a successful demo into a broad capability claim. The lightweight evidence path connects three local facts: an immutable run configuration and artifacts, a verified checkpoint, and a held-out validation observation recorded against that checkpoint.

## Surface workflow

```sh
klab train --runs-dir klab-work/runs configs/smoke_cpu.yaml --run-id evidence-smoke --stop-after-step 4
klab checkpoint verify klab-work/runs/evidence-smoke/checkpoints/latest.json --json
klab report --kind run evidence-smoke --json --runs-dir klab-work/runs
```

Both engines evaluate at step 0, every `evaluation.every_steps`, and the terminal boundary. Each evaluation forces a checkpoint and writes an immutable report under `klab-work/runs/<id>/evaluations/`. `klab report --kind run` verifies the run manifest, run-owned artifacts, every checkpoint generation, and each report's digest, checkpoint binding, counters, finite loss, validation protocol, tokenizer, and evaluation-input identities.

Assistant-only evidence counts the retained supervised targets, not every raw
prompt token. Wholly masked blocks are excluded before the configured batch
limit is applied. Reports explicitly bind supervision-mask and byte-address
arrays when present; rehashed reports with a different mask identity are
rejected. Older reports missing required bindings are not silently rewritten.

`checkpointed_held_out` means every validation-bearing verified generation has an accepted report. Missing or rejected reports produce `partial_held_out` when some valid observations remain, or `artifact_only` when none does; inspect `missing_reports` and `rejected_reports`. Historical unbound reports are not silently upgraded. None of these labels establishes arbitrary language usefulness, safety, production quality, or semantic train/test separation.

Quality evidence verifies checkpoint file integrity, weight inventories, and evaluation bindings; it does not certify optimizer-resume readiness. Rows expose `verification_scope: weights_only`. For format-2 generations, use `checkpoint verify` without `--weights-only` to check native continuation separately; legacy `.pt` inspection/verification remains promotion-only. With `checkpoint.keep_periodic: false`, pruned generations leave their old reports unverifiable; retain periodic generations when complete checkpoint-backed learning curves are required.

Task evidence is separate: [capability cards](capabilities.md) retain responses and exact-match scores, actual training budgets, parameter counts, and training/evaluation source identities. A model can reduce loss without learning a named task. Chat transcripts show particular behaviors; controlled cards test whether those behaviors persist across a declared set.

## What each level validates

| Level | Evidence | Does not establish |
|---|---|---|
| Behavior | Focused tests and a real smoke path | Generalization or hardware performance. |
| Artifact | Checkpoint hashes, tensor inventory, and run-owned configuration/data artifacts | Model quality. |
| Local experiment | Verified checkpoint paired with held-out loss/perplexity at recorded budgets | Superiority over an unmatched run or task-level ability. |
| Controlled comparison | Matched source, tokenizer, architecture boundary, optimizer, sequence length, token budget, device, and seed | A universal scaling law or production performance. |
| Hardware implementation | Defined numerical tolerance against reference vectors, one update or state transition where applicable, and measured end-to-end behavior | Correctness on untested operations, shapes, or numerical formats. |

## Hardware and numerical-format experiments

A future RISC, ASIC, or MXFP4 accelerator experiment should use Model Lab as a **reference harness**, not as proof from an isolated kernel demo. Freeze a small input/token batch, model weights, expected outputs, and tolerance policy; run the same operation through the PyTorch reference and the candidate path; then retain both artifact identities, output error statistics, and measured throughput/energy conditions. For training-capable hardware, also compare one optimizer update or an explicitly bounded state transition.

MXFP4 will need its own explicit quantization, rounding, accumulation, overflow, and dequantization contract before such a claim is meaningful. A successful dense FP32 run is useful baseline evidence, but it is not MXFP4 validation. Keep the candidate implementation behind an explicit configuration/backend selection; never silently substitute a reference result for hardware output.

## Experiment discipline

State the hypothesis before running: for example, “block selection retains comparable held-out loss at a fixed selection budget,” or “candidate MXFP4 matmul remains within the declared tolerance on this frozen corpus.” Preserve negative and inconclusive outcomes. Change one boundary per comparison where possible, and report the conditions that would falsify the hypothesis.
