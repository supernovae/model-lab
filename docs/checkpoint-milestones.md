# Scientific checkpoint retention

Workshop can declare a finite local keep-set in the existing RunConfig before a
new run starts. This does not approve running the configuration. Without
`checkpoint.retention`, serialization and keep-all/latest-two-plus-best behavior
remain unchanged. The policy is opt-in, requires `keep_periodic: false`, and cannot
be added to an existing checkpoint directory or changed in place. A new child can
declare future boundaries while keeping already reached parent declarations compatible.

For example, this **unexecuted illustrative fragment** assumes training limits of
10 updates and 80 committed supervised targets. Replace every boundary and byte
limit from the actual approved model, whole-update accounting and qualified
checkpoint sizes before use:

```yaml
checkpoint:
  every_steps: 5
  keep_periodic: false
  retention:
    milestones:
      - {name: early, step: 2, committed_targets: 16}
      - {name: technical-check, step: 4, committed_targets: 25}
      - {name: final, step: 10, committed_targets: 80}
    terminal_step: 10
    terminal_targets: 80
    max_generations: 7
    max_generation_bytes: 100000
    max_retained_bytes: 700000
    max_peak_bytes: 900000
    reserve_free_bytes: 0
```

A milestone is the first verified full-state checkpoint whose update **and**
committed-target counters meet or exceed its boundary. Crossing either boundary
alone is insufficient. The existing trainer checkpoint cadence requests a save
when both become true; there is no background scheduler. The report records the
actual counters, including overshoot, rather than calling a crossed threshold an
exact observation. Unreached thresholds remain explicit in `pending_milestones`;
`milestones_complete` is false even if an update limit ends training first.
Milestone roles can coincide on the same immutable generation.
The native trainer resumes into a **child run**. For already reached boundaries it
verifies and records the parent's original milestone locations and digests without
copying payloads or relabeling the child's first checkpoint. Boundaries must match
the parent declaration, and a selection newer than the chosen recovery checkpoint
is refused. Missing or corrupt inherited files refuse continuation. Reopening a
manager also preserves its saved selections. Future boundaries are selected locally.

The keep-set also contains latest, previous, validation-best and terminal. Terminal
means the first checkpoint reaching either declared training limit, which must
match the RunConfig. An interrupted run's latest/previous are recovery points; an
interruption is not a completed terminal budget. Roles deduplicate by generation.
Parent files remain under their original retention policy. Inherited milestone
bytes count in the child keep-set and peak accounting; unrelated parent artifacts
remain a separate storage budget. The conservative plan reserves `number of milestones + 4` retained generations and
one additional generation for staging. Thus coincident roles save actual space,
but the declaration does not rely on coincidence to fit its future budget.

## Storage and evidence

`max_generation_bytes` includes all regular files in one generation, including its
manifest. Calibrate it from actual full-state checkpoint sizes with optimizer
state, not weights-only size or the smaller initial checkpoint. The declaration
must accommodate the conservative count times this bound in `max_retained_bytes`
and the extra staged generation in `max_peak_bytes`. Allow room for checkpoint
metadata and any existing interrupted stage too. `reserve_free_bytes` preserves
additional free-space headroom; the example's zero is only a tiny-fixture value.

The existing startup storage check includes the declared peak budget and reserve
before runtime/model initialization. Before every save, the manager observes current checkpoint-root bytes and free
space and forecasts the next write using the declared per-generation bound. Free
space must cover the remaining declared peak budget plus the free-space reserve.
After serialization, it checks actual generation and root sizes before publishing.
These are admission and post-write checks, **not a filesystem quota**: a serializer
or another process can temporarily exceed the estimate; the manager refuses an
oversized generation before publication and removes only its own staging files.
Other run outputs, archives, datasets and unrelated processes need their own
storage budget. No cgroup, mount, quota or host setting changes are made.

The existing verified-generation publication and writer lease remain authoritative.
Before pruning, the complete required keep-set must verify and fit. The manager
writes `checkpoints/retention.json` before removing any unselected generation. It
records checkpoint digest, full-state resume level, actual update/target counters,
roles, exact package source identity digest, verified local location, measured
bytes and write-admission evidence. The trainer also records this document as a
`checkpoint_retention` event in its existing output tracking. The exact package
digest identifies the checkpoint's implementation; it is not a guessed Git SHA.
The immutable checkpoint and run manifests retain their existing source provenance.

`checkpoints/retention-policy.json` binds the policy across reopening;
`retention-inherited.json` records any child-run references. Preserve those source
locations, or restore the explicitly declared recovery closure, for further resume. A missing,
corrupt or unverified previously retained generation stops reconciliation; it is
not replaced silently with a later candidate. Storage refusals write the latest
`retention-failure.json` without deleting generations. The successful retention
report's `prune` list is the intended deletion set, not a claim that every deletion
finished. A crash after publication may leave extra files. An interrupted deletion
that leaves an invalid generation stops reconciliation, later saves, and automatic
recovery for operator diagnosis; explicit resume from a verified generation into
a new child remains available.
There is no automatic historical cleanup or stale-state bypass.

Use the existing public readback commands on a reported generation:

```sh
klab checkpoint inspect /absolute/run/checkpoints/step_00000004_gen_000005 --json
klab checkpoint verify /absolute/run/checkpoints/step_00000004_gen_000005 --json
```

## Optional archive boundary

Existing `archive create`/`archive verify` operates on explicit committed recovery
references. Its exclusive, streamed publication is suitable for an independently
budgeted archive after selection, but cannot prevent routine trainer pruning by
itself: it is not called before each checkpoint projection. This small policy
therefore preserves the selected full-state generations locally and reuses the
existing manager, rather than introducing a transfer worker or copying payloads.
Workshop may declare their exact paths/digests in the existing recovery recipe and
publish/verify an archive under separate authorization and storage headroom.
Archive failure never changes this local keep-set. Thin metadata archives and
weights-only exports are not full-state recovery substitutes. No archive is
created automatically and no hosted-transfer recovery behavior is changed.

Supported here: the existing POSIX checkpoint manager and native full-state codecs;
tested with synthetic CPU tensors on Linux/WSL. Native Windows checkpoint writing
is not introduced. No accelerator execution or hardware-capacity qualification is
implied by these software tests.
