# Reproducibility and artifact identity

Reproducibility in SparseLab is an artifact contract, not a promise that every accelerator produces bitwise-identical floating-point results. A local run records requested/effective config, runtime identity, source identity, model architecture identity, tokenizer/data/package identities, and verified checkpoint lineage so later readers can determine what was actually run.

## Start from explicit inputs

Use schema-v2 configs. Old run configs are not silently reinterpreted by `train`; migrate them to a new path and inspect the printed field changes:

```sh
uv run --locked --extra cpu sparselab config migrate old-run.yaml --output migrated-run.yaml
```

Migration converts the historical device/batch/checkpoint cadence fields to explicit runtime, micro-batch/accumulation, and checkpoint sections. It refuses to overwrite its output. Tokenizer-training configs remain their own v1 schema.

For a CPU baseline, set `runtime.backend: cpu` and `training.deterministic: true`. The repository's observed CPU interruption proof establishes equality for its stated reference configuration; it does not establish bitwise equivalence on MPS or other accelerators. Runtime discovery/probing records what was available and tested at the time of a run.

## Run-owned inputs and immutable output

A PyTorch training run copies its tokenizer, prepared data, and optional portable package into the run directory. Checkpoint and continuation checks use those run-owned artifacts rather than requiring the original external paths. The manifest binds content hashes while excluding machine-local locations from identity decisions. Source identity hashes executable package/runtime resources, so same-version source edits are not silently treated as the same executable.

Checkpoint generations are immutable and verified before publication. `latest.json`/`best.json` are lookup pointers, not the primary durable evidence. Validation reports bind to the selected checkpoint and identities. Keep the complete run directory when retaining an experiment: copying only model weights discards the information needed for a full resume.

## Continuation choices

Use `--resume` for the same compatible PyTorch experiment. It restores model, optimizer, scheduler, scaler when present, cursor, counters, cadence, and RNG state into a child run. Full resume requires matching scientific state and engine/backend. If source/runtime identity has changed but the same-backend scientific contract remains compatible, `--allow-runtime-drift` is explicit and records the continuation as best effort.

For supported dense PyTorch full-state resume or recovery, `--allow-runtime-optimization` separately admits the supported execution changes while preserving scientific settings, effective training batch and validation coverage. It records a best-effort transition, not bitwise equivalence; see [explicit runtime optimization on resume](checkpointing.md#explicit-runtime-optimization-on-resume).

Use `--promote` for compatible **weights-only** reuse with fresh optimizer/schedule/scaler/RNG/cursor/counters. Promotion does not allow an architecture semantic or tokenizer/package mismatch, does not resize the model, and is not a general external checkpoint importer. Historical/incomplete state without required optimizer/RNG/provenance is promotion-only rather than pretending to be full-resumable.

`--recover RUN_DIR` is the explicit recovery path for finalized verified generations. It never turns a corrupt explicitly selected `--resume` pointer into a silent fallback.

## Compare honestly

A matched comparison names the data/tokenizer identities, architecture/attention choice, seed, token budget, effective batch, sequence length, optimizer/schedule, engine/backend/precision, and observed resource budget. Different values may still be useful experiments, but their results answer a different question. Keep negative findings: a zero delta, a failure to fit, or an unavailable hardware capability is data, not a result to hide.

Disk snapshots, activation recomputation, gradient accumulation, and activation offload serve different roles; see [checkpointing](checkpointing.md), [activation recomputation](activation-checkpointing.md), [gradient accumulation](gradient-accumulation.md), and [offload](offload.md). None replaces a verified artifact identity, and none turns independent local runs into distributed backward.
