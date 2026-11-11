# Controlled benchmarks

Workshop owns the experiment's YAML declaration. Invoke the existing public
`klab probe CHECKPOINT --config /absolute/workshop/probe.yaml --json` under
its separately approved resource envelope. This remains the full probe entrypoint,
including other full-tier probes; it is not a benchmark-only runner. Public tasks
do not consume the private final split unless `final: true` is selected separately.

`lm_eval_protocol` is optional. Legacy full-tier smoke and task/limit overrides
remain unchanged, but cannot be mixed with this controlled protocol. Supported
standard tasks are lambada_openai, piqa, arc_easy, hellaswag and sciq. Task choice,
primary metrics, count and seeds belong to the experiment. This slice supports
finite deterministic first-N selection, including counts above the legacy 50.

## Unexecuted declaration shape

Replace every placeholder with observed, reviewed pins from the already installed
environment and cached data. This example is deliberately not runnable or approval.

```yaml
tier: full
fast_fail: false
lm_eval_protocol:
  harness_version: '0.4.13'
  dependency_versions:
    datasets: REQUIRED_EXACT_VERSION
    tokenizers: REQUIRED_EXACT_VERSION
    torch: REQUIRED_EXACT_VERSION
    numpy: REQUIRED_EXACT_VERSION
  tasks:
    piqa:
      config_sha256: REQUIRED_64_HEX_DIGEST
      dataset_fingerprints:
        train: REQUIRED_CACHED_FINGERPRINT
        validation: REQUIRED_CACHED_FINGERPRINT
        test: REQUIRED_CACHED_FINGERPRINT
      evaluation_split: REQUIRED_ACTUAL_SPLIT
      primary_metric: acc,none
      context: standard
  limit: 200
  selection: first_n
  num_fewshot: 0
  seed: 7
  fewshot_seed: 7
  bootstrap_iters: 1000
  cache_policy: offline
```

The split map must equal every actual split in `task.dataset`, including few-shot
sources. `config_sha256` uses `benchmark_protocol.task_config_digest(task.dump_config())`
before evaluation overrides. This effective-config digest binds prompts, task
metadata, preprocessing and dataset settings. Only the `metadata.config_source`
field (the machine-local task YAML path) is excluded; moving an identical installed
task does not change its pin. Other metadata, including `metadata.version`, remains bound. It is not a YAML-file
hash. This location comes from upstream
[`TaskFactory._load_full_config`](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/_factory.py).
Dataset fingerprints bind cached content, not independently authenticated source
revisions, rights or contamination claims. Workshop must retain that source and
acquisition provenance alongside the machine-checked declaration. Package versions
are verified, not a complete binary/environment attestation.

The implementation follows the pinned upstream
[0.4.13 evaluator](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/evaluator.py)
and [task API](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/api/task.py).
It loads tasks under Hugging Face offline flags (including imported configuration)
and verifies pins before scoring. No cache/download fallback, custom task loading
or unsafe-code opt-in is provided. This is library offline policy, not an arbitrary
code sandbox. Tasks fixing zero-shot refuse nonzero overrides. Nonzero few-shot requests also
refuse when the explicit or fallback example source is the evaluation split.
The standard `lambada_openai` task therefore supports zero-shot only. Existing windowed
continuation scoring is unchanged, without chat templates. Actual context length,
parameter dtype and backend enter comparison identity.

[SciQ](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/sciq/sciq.yaml)
includes its support passage. Require `context: support_passage`; never call that
closed-book science knowledge. Its dataset and installed harness remain unqualified
by these fake-harness implementation tests.

## Existing receipts

The sealed probe row's `details.controlled` retains all task metrics, including
raw/normalized accuracy and upstream stderr; original/effective counts; logged
item counts and per-item result digests; protocol and actual
checkpoint/tokenizer identities. `subject.tokens_seen` is the checkpoint's count,
not independently audited exposure; unknown values stay unknown. Evaluation wall
time includes loading and scoring. Training time, compute and cost remain null
because this adapter has no measured source for them. Bootstrap settings and
reported uncertainty are preserved, never synthesized.

Completion requires each primary metric, exact requested count, unique first-N
item IDs, logged item identity and observed shot count. Partial results retain
metrics, mark evidence incomplete and withhold a comparison group. Nonfinite
metrics and sample values are encoded as `NaN`, `+Infinity` or `-Infinity` strings
for hashing and JSON retention, with `nonfinite_evidence: true`; finite metrics
remain available. No digest of the unretained, timing-dependent raw response is
advertised. Complete
controlled results are informational, with science `not_assessed`; they do not
infer a winner or average tasks into a scientific outcome. Legacy aggregate views
are not controlled score-versus-target curves; Workshop can consume the per-task
receipt fields for those future curves.

Comparison identity binds protocol, scored inputs and scoring semantics. Different
checkpoints remain distinct subjects within a compatible group. Task/data, shots,
seeds, sample scope, context, dtype or backend differences change compatibility.
No parallel output index is introduced: retain the native receipt path/digest.
Fresh invocations create new probe IDs without overwriting old receipts.

Actual protocol review, cache acquisition and model evaluation remain separately
approved work. No reference model, benchmark data or GPU was exercised by the
implementation tests. Training exposure accounting remains LAB-43's measured gap.
