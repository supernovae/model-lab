# Explorer declarations

`explore` performs descriptive inspection, with five options: `--config`,
`--runtime`, `--runtime-profile`, `--resource-envelope` and `--json`.
The runtime selectors are mutually exclusive. Without either selector it uses
CPU, including for compatible saved PyTorch accelerator runs. An explicit
profile selects its backend through native runtime authorization; its engine
and device index must match the saved run. There is no device fallback.

```sh
klab explore TRY_ID
klab explore TRY_ID --config explore.yaml --json
klab explore TRY_ID --runtime MY_RUNTIME --resource-envelope bounds.yaml
```

Save a reusable descriptive request as `explore.yaml`:

```yaml
explore_config_version: 1
text: Once upon a time, a little dog found a ball.
refresh: true
# Optional lookup roots, relative to this file:
# lab_dir: ../lab
# runs_dir: ../runs
```

The declaration accepts only these fields. An empty file, a comment-only file,
YAML `null` or `{}` uses the same defaults as omitting `--config`. Other scalar
values and lists are rejected. Version defaults to 1, text defaults
to the existing short story, refresh defaults to false, and roots default to
native work-directory lookup. Unknown fields, unsupported versions, non-string
or whitespace-only text and non-boolean refresh values fail before runtime
preparation or model loading. Text remains subject to the explorer's existing 128-token limit;
checkpoints over 60 million parameters are refused before weight loading.
Meaningful text is preserved exactly, including leading/trailing whitespace.
Lookup roots expand `~` before anchoring relative paths to the declaration file.

Migration: `--text`, `--refresh`, `--lab-dir` and `--runs-dir` become `text`,
`refresh`, `lab_dir` and `runs_dir` fields. These flags are removed, including
from executable parsing. `--backend` is removed: CPU needs no option and other
devices use an existing [runtime profile](runtime.md) or registered runtime.
Resource bounds remain in the existing ResourceEnvelope, not in the descriptive
request. No new runtime or resource schema is introduced.

Record inputs still pin the recorded checkpoint generation and digest. Refresh
recomputes diagnostics over that same checkpoint and declared text. The sealed
cache remains keyed by checkpoint digest and text; configuration version and
lookup locations do not change scientific identities. The descriptive cache is
backend-independent: selecting another runtime can reuse diagnostics previously
computed on another backend. Set `refresh: true` to recompute with the selected
runtime; cached diagnostics are not evidence of that device's execution.
The native loader, cancellation sentinel, per-stage resource checks and OOM
cleanup are unchanged.

Everyday counts excluding help are try 5, probe 5, explore 5, compare 5,
report 2. See [try declarations](try-config.md) and [probe declarations](probe-config.md).
