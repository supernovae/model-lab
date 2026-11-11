# Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| `train/loss` | Mean next-token negative log-probability over valid supervised targets, in nats. | Lower only under comparable data/tokenizer/objective/budget conditions; excludes auxiliary routing loss. |
| `validation/loss` | Held-out target-weighted mean negative log-probability. | Compare alongside train loss for overfit signals; assistant-only masks also apply to evaluation. |
| `validation/perplexity` | `exp(validation/loss)`. | Not fairly comparable across tokenizers. |
| `optimizer/learning_rate` | Scheduled optimizer learning-rate value. | AdamW uses an update learning rate; Adafactor uses a relative step-size cap, not a numerically equivalent AdamW rate. |
| `optimizer/grad_norm` | Global L2 gradient norm before clipping. | Spikes can signal instability. |
| `performance/tokens_per_second` | Valid targets divided by synchronized optimizer-update duration, including forward/backward and transfers. | Excludes validation/checkpoint/logging; not end-to-end job throughput. |
| `moe/router_auxiliary_loss` | Valid-target-weighted auxiliary loss summed over layers, including its configured coefficient. | Diagnose optimization contribution; it is separate from language-model loss. |
| `moe/layer_*/router_entropy` | Mean entropy of pre-selection softmax probabilities. | A routing diagnostic, not a balance-loss objective. |
| `moe/layer_*/maximum_expert_fraction` | Largest per-expert assignment share for a forward. | High concentration can indicate collapse; no automatic remediation exists. |
| `moe/layer_*/mean_topk_probability` | Mean total routing probability assigned to selected experts. | Shows how much probability mass survives Top-K selection. |
| `engram/*` | Lookups, distinct buckets, collisions, reuse/utilization, and gate mean. | Addressing and mixing diagnostics; not retrieval accuracy. |
| `attention/layer_*/available_tokens` | Causal key positions available across the forward. | Sequence-length dependent baseline for sparse selection. |
| `attention/layer_*/selected_tokens` | Sparse key positions admitted after block selection. | Compare with available positions for the same run. |
| `attention/layer_*/selection_ratio` | `selected_tokens / available_tokens`. | Lower means more aggressive key filtering; it does not establish quality. |
| `attention/layer_*/estimated_flops` | Attention score-product estimate for selected sparse keys. | Relative implementation estimate, not a device benchmark. |
| `attention/layer_*/dense_teacher_mass` | Dense-attention probability mass over the sparse layer's selected keys; requires full diagnostics. | Higher means the selected set retains more of the frozen forward's dense attention distribution. |
| `attention/layer_*/dense_teacher_topk_recall` | Recall of dense score Top-K keys using the sparse selected-key count; requires full diagnostics. | Retrieval-overlap diagnostic, not output equivalence or language-model quality. |
| `allocation/raw_tokens` | Packed input-token positions consumed by an optimizer update, including prompt/context positions without a valid next-token label. | Runtime input count; separate from supervised targets and weighted neural mass. |
| `allocation/valid_targets` | Count of targets whose label is not `-100` in the update. | Counts all valid owner types once; not raw input tokens. |
| `allocation/weighted_neural_supervision_mass` | `neural_loss_weight × (neural-owned + hybrid-owned valid targets)`. | Weighted target mass, not optimizer updates, compute, or FLOPs. |
| `allocation/owner/*_raw_tokens` | Raw positions by owner sidecar code. | In the path-domain builder, unsupervised/context positions use the neural code; do not read this as neural supervised mass. |
| `allocation/owner/*_targets` | Valid target count by owner sidecar code. | The owner-specific partition of `allocation/valid_targets`. |

The memory-allocation report adds static parameter counts and a rough
`estimated_parameter_proxy_flops` estimate, computed as
`6 * active_parameters_per_token * raw_input_tokens`. This proxy is not an
executed-operation count: exact attention work, memory lookup/retrieval,
optimizer work, and device effects are excluded. It is not measured latency
or a FLOP-saving claim.

## Runtime, memory, and update measurements

| Metric or family | Interpretation |
|---|---|
| `performance/step_seconds` | Synchronized successful-update duration; includes forward/backward, recomputation and transfers, but excludes staging, validation and checkpointing. |
| `performance/primary_compute_seconds`, `performance/replica_compute_seconds`, `performance/device_wait_seconds`, `performance/gradient_merge_seconds` | Reported only for updates that ran on [several devices](gradient-accumulation.md#several-devices-on-one-host): each side's compute time, how long finished devices waited for the slowest, and the time to merge collected gradients. |
| `batch/micro_batch_size`, `batch/accumulation_steps`, `batch/effective_batch_size` | Configured microbatch size and actual microbatches/examples in the committed update. Epoch boundaries and final budgets can shorten a window. |
| `batch/tokens_per_micro_batch`, `batch/effective_tokens_per_update` | Valid supervised prediction targets, not raw prompt length or a nominal batch-size product. |
| `memory/device_allocated_bytes`, `memory/device_reserved_bytes` | Native allocation/reservation where exposed; MLX active allocation is distinct from `memory/device_cache_bytes`. |
| `memory/device_peak_allocated_bytes`, `memory/device_peak_reserved_bytes` | Native per-update high-water counters where available. |
| `memory/device_sampled_peak_bytes`, `memory/driver_allocated_bytes` | Sampled allocation maximum and driver reading. MPS sampled peaks are lower bounds, not allocator high-water guarantees. |
| `memory/process_rss_bytes`, `memory/process_peak_rss_bytes`, `memory/system_available_bytes` | Host resident memory, sampled update maximum, and available system RAM. CPU does not invent GPU measurements. |
| `recompute/block_call_ratio` | Recomputed block calls divided by original calls; one means every block was recomputed. It is not a measured time/memory saving. |
| `offload/bytes_to_cpu`, `offload/bytes_to_device`, `offload/peak_host_bytes` | Actual saved-activation transfers and peak live host offload storage. |
| `offload/transfer_seconds`, `offload/transfer_fraction` | Synchronized transfer cost, included in total update time rather than subtracted from throughput. |

Unavailable readings are omitted with explanatory events, not written as zero. Engine/backend/worker/precision/optimizer names are manifest metadata rather than numeric metrics. Apple unified memory is one physical pool: do not add device recommendations and system RAM, or claim extra capacity from MPS host offload. See [memory accounting](memory.md), [runtime policy](runtime.md), and [offload](offload.md).

## Runtime forecast and progress records

| Record | Interpretation |
|---|---|
| `runtime_forecast.planning` | Optimizer-only estimate from the most recent exact-compatible historical observations; missing matches produce unavailable fields, not a theoretical time. |
| `runtime_forecast.warmup_calibrated` | Separate estimate from measured disposable pilot updates; it is used only when the pilot and run signatures match. |
| `runtime_forecast.live.optimizer_only_eta` | Raw low/high seconds and status for remaining optimizer work, based on completed supervised targets and robust recent/long windows. It excludes future non-optimizer phases. |
| `runtime_final_observation.phases.*` | Actual seconds, availability, and observation count for preparation, planning, batch assembly (host-side selection and stacking of each update's blocks, while the device waits), optimizer updates, validation, checkpointing, and reporting. Attempts are separate; evaluation and generation are `not_observed` unless measured by their own operation. |
| `runtime_final_observation.end_to_end_seconds` | Observed whole-run wall time. `unclassified_overhead` is the residual after measured phase components, not an independently timed phase. |
| `runtime_progress_snapshots` | One replaceable latest operational snapshot per run, outside the controller replication outbox. `runtime status` reads it and the separate final-observation event without modifying the database. |

All durations and rates are raw numeric JSON values. A completed run has zero
remaining optimizer ETA; unavailable or unobserved telemetry remains `null`
with an availability/status reason. The dashboard presents the same runtime
records as the status command. See [runtime policy](runtime.md#runtime-forecasting-and-progress)
for matching rules, commands, and forecast limitations.

## Diagnostics and durable aggregation

Architecture diagnostics (`engram/*`, per-layer MoE and attention values) are persisted from the **last nonempty training microbatch before validation**, not averaged over an accumulation window. They expose routing/address use; they do not prove successful retrieval. Scalar diagnostics do not archive full router tensors.

Parameter counts use the direct per-token convention documented in [model scaling](model-scaling.md). Routed experts, shared experts and routers have disjoint inventory categories; any readable expert subtotal must not be added again. Engram totals include every table and adapter, while active accounting includes one retrieved row per stream plus the adapter. These are parameter-use conventions, not measured FLOPs or sparse optimizer storage savings.

Every worker's SQLite database/WAL remains local. The controller imports contiguous, checksum-bound `(origin_id, sequence)` records idempotently, without re-emitting them into its own outbox. A controller disconnect can leave its view stale while the worker continues; terminal execution and verified artifact ingestion are separate states.

The read-only dashboard uses this local projection and retains actual step/target coordinates. Runtime, Memory, Checkpoints, Stages and Learn expose execution conditions and their limits alongside the loss curves. Different backends, optimizers, precision, tokenizers, objectives or observed budgets prevent an unqualified controlled comparison. See [worker acceptance](workers.md#observed-acceptance--2026-09-23).
