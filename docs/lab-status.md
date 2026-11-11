# Recorded software integration checks

These records check Model Lab behavior: training-state continuity, checkpoint
integrity, worker orchestration and evaluation binding. The CPU/GPU column names
where each check ran. It is execution context, not the research question or a
certification of the processor. Feature availability is in the
[README matrix](../README.md#feature-matrix); model findings are in the
[experiment ledger](research/experiment-ledger.md).

| Record | Software behavior checked | Execution context |
| --- | --- | --- |
| [Host CLI checks](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/host_cli_2026_09_22.json) | Full-state continuation with FP32/BF16 and Adafactor; checkpoint/resume behavior. | PyTorch on CPU; separate native MLX execution on Apple Silicon. |
| [Single-host checks](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/single_host_gate_2026_09_22.json) | Interruption/recovery, promotion, offline installed-wheel operation, staging and populated dashboard views. | CPU and Apple Silicon GPU execution through PyTorch MPS or MLX. |
| [Independent-worker checks](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/independent_workers_2026_09_23.json) | Controller loss, replay, cancellation, child resume, executor loss, matrices and source/capacity rejection. | Concurrent logical CPU workers and a separate MLX worker. |
| [Registered-runtime checks](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/runtime-contract-rocm-20261001T134938Z.json) | Runtime binding, completed ingestion, checkpoint verification and evaluation preserving training identity. | Training used PyTorch ROCm/BF16 on an AMD RX 7900 XTX; evaluation ran explicitly on that GPU and on CPU. |
| [Research report checks](research/sample-report.md) | Catalog scaffolding, collection and report surfaces. Scientific observations retain their separate qualifications. | Each linked report records its own execution environment. |

The worker UI record includes rendered curve pixels and runtime-table checks
through the app harness; conventional browser screenshot capture was incomplete.
Keep that limitation with the observation. Source versions, commands and workload
bounds remain in the original records.

See [runtime support](runtime.md) for setup and feature restrictions,
[workers](workers.md) for operation, and [feature roadmap](../ROADMAP.md) for pending directions.
Use local discovery and workload warmup when selecting an execution environment.
