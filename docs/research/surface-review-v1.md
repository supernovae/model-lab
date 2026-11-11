# Surface Review: local self-blind comparisons

Surface Review separates **exploratory checkpoint chat** from a **sealed review of existing, verified generations**. Both are local single-reviewer tools, not an automatic model selector. A sealed import reads already generated text; it does not load weights, rerun a cell, replace missing output or alter the original study. Work from the repository root. Put new bundles and votes under the task-owned `$KLAB_WORK_DIR/experiments/surface-review/bundles/` (or another explicit `--output`); do not edit the original studies' review, test or checkpoint files.

## Import existing generations

Each import requires a new, nonexistent output directory and verifies indexed
source bytes before selection. The current importer consumes native triage
reports. Experiment-specific decoding and data-rich import adapters are retired;
their findings remain in the [inactive research history](https://github.com/supernovae/model-lab/tree/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history).

To compare compatible Tier-1 outputs from at least two verified, immutable post-train triage reports, provide the runs explicitly; no matching report/prompt/settings cells means no bundle:

```sh
uv run --locked klab surface import triage \
  --triage-run RUN_ID_1 /path/to/runs --triage-run RUN_ID_2 /path/to/runs \
  --sample standard --selection-seed 2026 --presentation-seed 2027 \
  --output "$KLAB_WORK_DIR/experiments/surface-review/bundles/triage-standard-2026"
```

`--sample quick` selects up to 12 eligible pairs, `standard` up to 32, and `full` all eligible pairs. No vote affects the sample size. Candidate IDs are sorted canonically by prompt, decoder, RNG and unordered source pair; a seeded, deterministic greedy rule favors underrepresented category, source pairing, decoder, RNG and prompt in that order, then hashes the selection seed and candidate ID to break ties. A separate presentation seed deterministically shuffles the selected cases, makes opaque case IDs and assigns A/B orientation. Both seeds are signed decimal integers in private provenance; changing either requires a new output directory. No quality judgment enters selection. These are balanced *descriptive* subsets, not a random population estimate.

## Answer and reveal

The review page is hosted by Streamlit on `127.0.0.1` and shows only the verified blind cases: exact prompt and full response A/B, category, applicable dimensions, issue tags and durable progress. No checkpoint alias, run identity, decoder/seed metadata or source filename is part of a blind case. Supply exactly one choice per displayed dimension: `A`, `B`, `tie`, `neither` (neither response wins), or `cannot_tell` (insufficient confidence). Optional tags are `entity_changed`, `attribute_changed`, `object_changed`, `causal_contradiction`, `repetition`, `nonsensical_drift`, `premature_ending`, `prompt_ignored`, `other`; an optional note is separate. A submitted answer is write-once: another tab cannot overwrite it. Progress is read from verified disk answers after restart, not merely browser session state. Finish all cases and select **Complete** to freeze the blind review; **Reveal** is a later explicit action. Before that action, do not open private `provenance.json` or try to infer identities from response content. After reveal, results map the recorded A/B choices to actual sources and show per-pair/per-dimension raw counts (including ties, neither and cannot-tell); they do not compute Elo or a probability of population preference.

| Applicability | Dimension IDs |
|---|---|
| Every triage case | `prompt_adherence`, `repetition`, `readability_coherence`, `overall_preference` |
| `entity_continuity`, `named_character_continuity` categories | additionally `entity_continuity` |
| `object_continuity`, `color_attribute_continuity` categories | additionally `attribute_consistency` |
| `cause_effect`, `temporal_ordering`, `location_permanence` categories | additionally `causal_temporal_coherence` |

These seven IDs are the available canonical dimensions; only the declared applicable subset is voted for a given case.

The v1 integrity chain is:

1. `manifest.json` (`klab_surface_manifest_v1`) records eligible/selected counts and SHA-256 of the exact canonical `blind.json` and `provenance.json` bytes, without identities. Blind format is `klab_surface_blind_v1`; private provenance format is `klab_surface_provenance_v1` and records indexed source file SHA-256/size, prompt-set digest, selected/eligible keys, both seeds, decoder coordinates, original and displayed A/B source mapping and checkpoint/run/tokenizer identities where available. Import verifies source files; later opening verifies the sealed files without needing those source files mounted.
2. Each exclusive `answers/<opaque-case-id>.json` is a `klab_surface_judgment_v1` record bound to the blind digest and its own canonical content digest, with choices, optional issue tags/note and UTC timestamp. Missing, changed, unexpected or malformed answers invalidate completion rather than being silently repaired.
3. `review.json` is a write-once `klab_surface_review_v1` record with a digest of its contents, case-ordered judgments, `single_reviewer_self_blind`, blind digest and `reveal_state: false`. Only after completion can a separate write-once `reveal.json` bind the review digest and private provenance digest to the exact source mapping with `reveal_state: true`.

Do not use a revealed bundle for another supposedly blind review: make a new bundle and predeclare its selection instead.

## Exploratory chat is not sealed review

For two to four *available* Model Lab generation paths, use separate `--cell ALIAS=GENERATION_PATH` arguments; the alias is a local selection aid, not a trusted model identity:

```sh
uv run --locked klab surface chat \
  --cell old=/path/to/runs/OLD_RUN/checkpoints/GENERATION_PATH \
  --cell new=/path/to/runs/NEW_RUN/checkpoints/GENERATION_PATH \
  --backend cpu --seed 11 --port 8502
```

Replace the illustrative generation paths with paths to real checkpoint generations accepted by `load_run`. The chat verifies checkpoint/run/tokenizer identity, checks that a prompt fits **every** native context before generating, uses the same selected greedy or sampled policy and session seed across models, and presents anonymously permuted response cards. It may save separately marked `EXPLORATORY` local responses/votes; these never become sealed import evidence. An empty prompt/session finish ends the chat and permits identity reveal. This mode runs inference and needs installed, accessible weights and a compatible backend; by contrast `surface review` needs only the sealed bundle. On a vendor-provisioned accelerator, protect that environment from a CPU wheel replacement by using the project's documented `--no-sync` procedure.

## Interpretation and comparison boundaries

`single_reviewer_self_blind` means one person's judgments about a prespecified local batch. Self-blinding does not establish inter-rater agreement, population preference, a general model-quality ranking or a causal effect. Response style can unblind a reviewer informally. TinyStories study prompt train-disjointness is not established merely by a held-out prompt label; investigate overlap before making generalization claims. The data-rich 30M and older 50M differ in source/data variety, tokenizer, context, model depth/width, target budget and precision; their difference cannot isolate a data-versus-depth cause. Tier-1 triage prose and mechanical repetition are diagnostics; Tier-2 recommendations are post-hoc hypotheses, not subjective votes or promotion gates. Immutable triage reports and their promotion criteria remain unchanged; the [optional verified overlay](post-train-triage.md#optional-surface-review-overlay) may report review availability or a completed single-reviewer observation only.

For larger standardized task evaluation, design an independent protocol using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) or [LightEval](https://github.com/huggingface/lighteval) with pinned model/tokenizer, tasks and evaluation settings; neither is silently executed by Surface Review. [Open WebUI Arena](https://docs.openwebui.com/features/evaluation/) is an optional future interface for broader local human comparisons, not the sealed v1 importer and not a requirement for this workflow. The separate [`klab serve` local API](../local-api.md) now provides a bounded, nonstreaming integration for one pinned checkpoint; it does not replace sealed review or add multi-model routing. The [model guide](../tinytext-model-guide.md) covers raw-prompt comparison, prompt visibility, stop/context/cache controls and reproducibility.
