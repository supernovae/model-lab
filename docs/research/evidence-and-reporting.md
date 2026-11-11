# Evidence and static reports

A report consumes one explicit collected report; it does not choose the latest/best result, train, load a model, generate, or evaluate cards.

```sh
klab study report experiments/ffn-memory/study.yaml RECEIPT.json \
  --evidence COLLECTED.json --research experiments/ffn-memory/research.json \
  --output artifacts/research-reports
```

The bundle retains supplied evidence and presents declared cards, endpoints, pair comparisons, calculated architecture/cache quantities, and limitations. It makes no winner, statistical-significance, threshold, or monetary-cost claim. Direction-aware nondominance sets are descriptive, per matched evidence group; exact ties remain together and incompatible groups are never ranked.

Without `--runs-dir`, the verification scope is `checksummed_report_only`: report values are checksummed supplied evidence, not a new experiment attestation. Supplying `--runs-dir RUNS` additionally performs local checkpoint/manifest validation and reads endpoint-bounded telemetry read-only; it does not re-collect cards. Missing local runs are unavailable, while unsafe/corrupt/mismatched local evidence is rejected rather than replaced with an inferred zero.

Historical v1 receipts created before `model.semantic_memory_dim` and `training.neural_loss_weight` were added remain renderable when those fields still have their current defaults. The report builder recomputes the original study, matrix, and per-run identities; non-default field values do not use this compatibility path.


For research-scaffolded studies, the configured endpoint policy is the first configured limit. An endpoint is complete only when it reaches either configured `max_steps` or `max_tokens` without exceeding either cap and has complete identities. Earlier interruption is partial. Pair deltas require complete matched endpoints; partial, unavailable, invalid, and inconclusive contexts stay visible and are not converted to a score or zero. Raw card cases, including zero and negative deltas, are retained.

Recipe-declared `klab-factorial-design` version 1 records extend the recipe, not the study v1 schema. A complete 2×2 context reports `y00`, `y10`, `y01`, and `y11`, with factor-A effect `((y10-y00)+(y11-y01))/2`, factor-B effect `((y01-y00)+(y11-y10))/2`, and `interaction_delta=y11-y10-y01+y00` on each catalog metric's raw scale. Each outcome requires all four committed cells to match at endpoint step/tokens, tokenizer, data, source, runtime, and training-runtime identity. Missing, partial, invalid, duplicated, or mismatched cells are inconclusive; no missing value is replaced with zero and no effect is emitted.

Packaged factorials cover FFN width (`4x` versus `1x`) × memory, MLA attention (`dense` versus `mla-half`) × memory, sparse attention (`dense` versus `sparse-2`) × memory, and MoE capacity (`dense-2x` versus `moe`) × memory. The `latent-sweep` and `budget-sweep` recipes also retain their other declared axis levels for boundary inspection; the factorial effect uses only its declared two-level subset.

Nondominance uses catalog-declared lower/higher directions for held-out validation loss and each capability-card score. It compares only runs sharing seed, endpoint, data/tokenizer/source, and runtime identities; candidates missing any objective remain inconclusive. Identical objective vectors stay as ties. This is not a statistical test or a winner claim.

Allocation heatmaps use configuration-derived parameter inventories (total, active-per-token, non-memory, table, and adapter counts); these are architectural estimates, not observed memory or runtime cost. Boundary-sweep tables and SVGs show only configured FFN, latent-width, sparse-budget, expert-capacity, or other recipe-axis levels. They do not interpolate or infer a failure threshold.

Bundles include `report.json`, original receipt/collected bytes, manifest, escaped static HTML/Markdown, deterministic SVG charts, and available inert inputs. They exclude model weights, prepared arrays, downloaded corpora, credentials, and database copies. Hashes establish bundle byte identity, not authenticity, license clearance, or removal of machine-local paths from copied evidence.

## Lifecycle declarations versus verification

The lifecycle sidecar records reviewed Findings, baseline declarations, and
promotion decisions; it does not recalculate report results. `report --kind research`
and `research validate` show every referenced item as verified, unavailable, or
invalid with its evidence basis. A matching document or source citation is not
measured experimental evidence. An intact static bundle can preserve historical
baseline establishment when optional local checkpoint weights are unavailable;
that does not claim present runtime compatibility or runnable weights. The
[known-good baseline guide](known-good-baselines.md) defines the complete
identity-bound evidence roles for a baseline.
