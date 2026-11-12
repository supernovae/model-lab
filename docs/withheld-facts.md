# Withheld-fact diagnostic

Commands below use the [activated, provisioned environment](../README.md#first-run).

The offline deterministic fixture, immutable split manifest, verifier, and compact verified audit make fact-identity separation testable before transfer experiments. The native manifest command records six training facts and two held-out facts. The split is by complete `(subject, relation)` identity; held-out values are absent from training documents. Evaluation cases expose a prompt and expected value, with the expected value excluded from the prompt.

This is not a training source, benchmark score, or evidence of byte-memory transfer. It deliberately does not add held-out facts to backbone training, memory-adapter supervision, or router objectives. Its purpose is to make accidental leakage detectable before a future trained diagnostic is designed.

Before running a diagnostic, write and retain a manifest:

```sh
klab facts manifest --seed 0 --output "$KLAB_WORK_DIR/withheld-facts-seed-0.json"
```

Verify a retained artifact before using or citing it:

```sh
klab facts verify "$KLAB_WORK_DIR/withheld-facts-seed-0.json"
```

For automation, emit a compact verified audit:

```sh
klab facts audit "$KLAB_WORK_DIR/withheld-facts-seed-0.json"
```

The JSON report states the fixture seed, manifest digest, statement/case counts, and whether held-out values occur in the training statements. It is fixture evidence only; it contains neither model outputs nor an experimental score.

To measure a byte-memory adapter transfer between two compatible saved runs:

```sh
klab facts transfer-evaluate SOURCE_RUN TARGET_RUN "$KLAB_WORK_DIR/withheld-facts-seed-0.json"
```

Only `memory.table`, `memory.output`, and `memory.gate` transfer. The target keeps its own tokenizer, embedding, attention, output head, and decoder blocks. The result is retained under the target run's `evaluations/` directory with both run IDs and the verified manifest digest. An exact-match count remains an observation, not proof of general cross-tokenizer transfer.

Verification recomputes its digest and requires every field to match the current fixture for the recorded seed. A self-consistent substituted split is rejected, not merely a damaged JSON file.

The manifest records the ordered training statements, held-out prompt/expected-value cases, seed, format version, and a SHA-256 digest of its canonical payload. Re-running with the same seed accepts byte-identical evidence; a different seed cannot overwrite the existing manifest. It contains no model outputs, training examples, or score.

Record the verified manifest alongside any future experiment result. It proves the fixture split only; data-provenance evidence is still required to prove a training pipeline obeyed that boundary.
