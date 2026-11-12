# Corpus-layer teaching fixture

An original MIT-licensed, tiny offline fixture for the corpus commands. Three authored source families provide local Markdown, code and YAML; a fourth declaration names the deterministic `pathlib.PurePosixPath` suffix oracle. `technical_docs`, `code`, `configs`, and `systems_scenarios` are illustrative domains. The requested mixture is descriptive, not an optimized sampling policy. All generated text and snapshots live beneath the task-owned `$WORK/corpora/devmind-sample-v0/` directory and may be deleted when no run depends on them. No public corpus is downloaded.

Run from the repository root in the [locked CPU environment](../../../README.md#install)
(no model training). The optional `jq` utility below selects a record ID from
native JSON output; you can instead copy that ID from `corpus sample`. Choose a
fresh persistent workspace for a new copy of the sample:

```sh
export KLAB_WORK_DIR="$HOME/.local/share/klab"
WORK="$KLAB_WORK_DIR/experiments/corpus-teaching"
uv run --locked --extra cpu klab --work-dir "$WORK" corpus acquire tests/fixtures/corpus-sample/corpus.yaml
BUILD=$(uv run --locked --extra cpu klab --work-dir "$WORK" corpus build tests/fixtures/corpus-sample/corpus.yaml --offline)
RELEASE=$(uv run --locked --extra cpu klab --work-dir "$WORK" corpus freeze "$BUILD")
uv run --locked --extra cpu klab --work-dir "$WORK" corpus describe "$RELEASE"
uv run --locked --extra cpu klab --work-dir "$WORK" corpus sources "$RELEASE"
uv run --locked --extra cpu klab --work-dir "$WORK" corpus audit "$RELEASE"
SAMPLE_ID=$(uv run --locked --extra cpu klab --work-dir "$WORK" corpus sample "$RELEASE" --domain technical_docs | jq -er '.records[0].record_id')
uv run --locked --extra cpu klab --work-dir "$WORK" corpus lineage "$RELEASE" "$SAMPLE_ID"
uv run --locked --extra cpu klab --work-dir "$WORK" corpus review "$RELEASE" --kind chat
EXPORT=$(uv run --locked --extra cpu klab --work-dir "$WORK" corpus export "$RELEASE" --view lm --base-run-config configs/runtime_smoke_cpu.yaml --vocab-size 300)
uv run --locked --extra cpu klab --work-dir "$WORK" tokenizer train "$EXPORT/tokenizer.yaml"
uv run --locked --extra cpu klab --work-dir "$WORK" data prepare "$EXPORT/run.yaml"
uv run --locked --extra cpu klab --work-dir "$WORK" corpus describe "$RELEASE" --tokenizer "$EXPORT/tokenizer/tokenizer.json"
uv run --locked --extra cpu klab --work-dir "$WORK" corpus export "$RELEASE" --view chat --base-run-config configs/runtime_smoke_cpu.yaml --vocab-size 300
```

Frozen test rows are held out of tokenizer fitting and training exports. Assistant/tool messages are inert transcript examples; corpus derivation does not execute tool calls. A successful smoke proves local execution and provenance wiring, not source rights outside this MIT fixture, model usefulness or scientific performance.

## Shared-source shape demonstration

The same config declaration in `sources/fixtures/settings.yaml` is retained as a
`raw_document` and deterministically rendered as a `direct_qa`, a bounded
`troubleshooting_scenario`, and a `decision_record`. All four records point to
the same snapshotted document and passage; the rendered text, template family,
and shape identities differ. The authored config is not itself generated; its
three derived examples have source-transformed synthetic origin and an explicit
source-entailment rule. The independent path-world examples instead carry an
oracle receipt and deterministic synthetic origin.

Inspect their parent IDs and evidence with `corpus sample` and `corpus lineage`.
The default report counts each training view by origin, verification, and shape.
`corpus describe RELEASE --tokenizer TOKENIZER_JSON` recounts exact tokens; no
token count or capability improvement is inferred from the static bytes.
See [corpus preparation](../../../docs/corpus-preparation.md).
