# Experiment ledger

This is a curated index of retained project findings, not a live run database or
a model leaderboard. Each row links to the underlying record, states the bounded
lesson, and separates a possible follow-up from an approved protocol. Exact
configurations, checkpoint identities, metrics and reviewed decisions remain in
their original evidence records. No historical run is reclassified here.

For declared lifecycle state use `uv run --locked klab report --kind research`;
for an executing workflow use `campaign status`, `next` and
`explain`. Missing external weights or data can make a recorded result unavailable
to rerun without invalidating its historical identity.

## Learning and model behavior

| Question | Retained observation and evidence | What it teaches / limits | Candidate next question |
| --- | --- | --- | --- |
| Does lookup improve synthetic recall? | [Chat comparison](../history/project-review.md): both dense and Engram models retained all 24 taught associations; held-out wording and context override did not establish an Engram advantage. | Acquisition is distinct from generalization and context use. This was exploratory evidence. | Can a separately frozen task distinguish successful acquisition from robust retrieval? |
| Can a model use a new conversational assignment over a learned fact? | [Context study](../history/context-engram-study.md#execution-results--2026-09-22): untouched override remained 0/8 at every planned endpoint. | A failed control remains negative even when other metrics improve. | Which new curriculum teaches override on independent bindings without weakening static retention? |
| Does narrow domain adaptation preserve prior behavior? | [Path-domain study](../history/path-domain-corpus.md#2026-09-22-execution-record): acquisition improved, held-out reliability remained poor and static retention worsened sharply. | Adaptation needs both domain and retention evaluation. | Can a separately declared adaptation method improve held-out behavior with retention controls? |
| Can lookup replace dense FFN capacity? | [Offline FFN/lookup comparisons](sample-report.md), run on CPU, and the [FineWeb FFN/lookup study](development-evidence.md#completed-fineweb-edu-micro-study), run on Apple Silicon through PyTorch MPS: mixed or negative same-width lookup effects; narrower FFNs increased loss in the FineWeb comparison. | Lookup adds parameters; these short, bounded studies do not establish parameter efficiency or a scaling law. Out-of-domain alias scores are not web-text quality. | Replicate a declared contrast with adequate exposure and a task-relevant evaluator. |
| Can the dense baseline learn from repeated training exposure? | [Dense learning reference](dense-lm-v1.md): three seeds reduced held-out loss through the declared schedule; a bounded learning reference was explicitly promoted. | The endpoint was still learning; this is not a converged or generally useful assistant. | Compare exposure, data diversity and size under separate controls. |
| Does more exposure improve the same model? | [Token-budget results](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-token-budget-v1/results.md): three full-state continuations lowered held-out loss; some fixed continuations worsened. | Loss improvement does not guarantee better generation. The rejected partial child remains preserved. | Test continuity and repetition under a newly declared evaluation protocol. |
| Does a larger dense backbone help at the same exposure? | [Scale results](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-scale-v1/results.md): the larger model lowered like-seeded held-out loss with more update time and memory; fixed generation remained mixed. | This is a bounded size comparison, not an architecture-superiority result. | Measure task benefit alongside the resource cost at matched conditions. |
| Does sampling improve retained model outputs? | [Decoding results](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-decoding-v1/results.md): a development-selected sampled policy reduced mechanical repetition but showed drift; independent subjective quality remains unreviewed. | Decoder selection and test evaluation need separate prompts. Mechanical scores are not human preference. | Complete independent review of sealed outputs; do not tune on the opened test set. |

## Data and operational lessons

| Question | Evidence and bounded lesson | Remaining boundary |
| --- | --- | --- |
| Can a corpus support the intended exposure and source mix? | [Source expansion](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v3/protocol.md) and [tokenizer-readiness decision](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v4/protocol.md) show why distinct source supply, held-out independence and mixture exposure need separate measurement. | A corpus-readiness decision does not authorize model training, establish model quality or grant publication rights. See the [roadmap](roadmap.md) for subsequent declarations. |
| Can a successor preserve source history while establishing a runnable control? | [Reproducible successor protocol](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/protocol.md) preserves unavailable historical identities separately; its [dense control result](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model0-result.json) retains an accepted terminal full-state checkpoint. | The [equal-exposure continuation protocol](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model1-protocol.md) is declared, not a completed result. Source authorization, a child lock/Family pin and a current-source pilot remain gates in the retained roadmap. Check local Campaign state before acting. |
| Can input reuse reduce orchestration work without changing science? | [Capacity and verification measurements](../capacity-aware-execution.md) retain cold/warm observations, corrections and unavailable counters. | Same-store authenticated reuse is conditional; transfers, final archives and changed identities require their documented verification. Component savings do not establish full-run speedup. |
| Can training resume and workers recover safely? | [Software integration records](../history/lab-status.md) covers retained single-host, independent-worker and runtime-contract checks. | These checks concern orchestration and training-state behavior in the recorded execution environment; smoke does not measure model quality. |

## Maintaining the ledger

Add a row when a durable reviewed result adds a distinct lesson. Link the
protocol/results or immutable evidence; retain failed, negative, censored,
interrupted and unavailable observations. State changed and controlled settings
in the linked record. Update a conclusion only when a new identified record
supports it, and keep the prior observation discoverable. Proposed tests belong
in the [roadmap](roadmap.md), not an implicit execution queue.

The [dashboard](dashboard.md) already browses lifecycle findings and verified
reports. A shared CLI/dashboard ledger projection is a longer-term direction in
[ROADMAP.md](../../ROADMAP.md#longer-term-directions); it is not a shipped command
or a new evidence authority.
