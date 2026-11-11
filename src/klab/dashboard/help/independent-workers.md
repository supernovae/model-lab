# Independent workers and local projections

The CLI registers local or user-provisioned SSH workers and queues whole experiments. Each worker owns its optimizer, RNG, local database/outbox, immutable inputs, and checkpoints. A controller imports contiguous, deduplicated records into its own local database; it never shares SQLite WAL across hosts or performs distributed backward.

This dashboard reads that local projection. Runtime identity is the recorded engine/backend/device, not the worker's display name: a logical CPU worker named for an AMD or Intel role is still CPU. Separate loss curves and visible optimizer/precision/data/budget differences are required before interpreting comparisons.

`CANCEL_REQUESTED` is durable intent, not proof that a worker stopped. Acknowledged interruption retains the last committed full checkpoint. A transport disconnect does not restart training. Lost-executor evidence is finalized only after its claim and inherited resource holders are gone; the attempt then remains `UNKNOWN`, and resuming it requires an explicit new child. Remote terminal status and controller artifact ingestion are separate facts. Only verified, locally ingested full state can authorize a child resume; completed budgets cannot resume.

The dashboard remains read-only. Use `worker`, `experiment`, and `controller` CLI commands for operations. Real ROCm/XPU and multi-host concurrency require those hosts' own evidence; local logical workers do not close those gates.
