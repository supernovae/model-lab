# Continuation/source overlap diagnostic

`src/klab/corpus/memorization.py` exposes a **read-only, descriptive** comparison of one generated continuation to explicitly supplied source passages. It does not decide whether text is memorized, licensed, infringing, eligible for training, or releasable; there is no copyright threshold and no corpus-construction gate.

## Native interface

Run the native adapter with:

```sh
klab memorization analyze INPUT.yaml --json
```

The input is a strict YAML or JSON mapping with `format: klab-memorization-input-v1`. Its `continuation` is exactly one of:

```yaml
continuation:
  kind: file
  path: continuation.txt
  file_sha256: <lowercase SHA-256 of the exact UTF-8 file bytes>
```

or:

```yaml
continuation:
  kind: generation_panel
  result: retained-panel-result.json
  row: 0
```

A file continuation must be a regular, non-symlink UTF-8 file whose raw bytes match the declared digest. A panel continuation is accepted only after the native retained-panel verifier authenticates it; `row` is an explicit zero-based row and must be `COMPLETED` with a string completion. The adapter never selects a latest, first successful, failed, or interrupted row. Paths are anchored at the input declaration, may be absolute, and may not traverse `..` or symlinks.

`sources` is a nonempty list of unique nonempty IDs and explicitly supplied text, for example:

```yaml
format: klab-memorization-input-v1
continuation:
  kind: file
  path: continuation.txt
  file_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
sources:
  - id: example
    text: The supplied comparison passage.
ngram_size: 3        # optional; 1 through 10
max_edit_chars: 512  # optional; 1 through 2048
```

Supplied passages are comparison material, not authenticated claims about external authorship. The report is `klab-memorization-report-v1` with role `descriptive_not_quality_gate`; it retains declaration binding, file or authenticated-panel provenance, the raw continuation digest, raw source digests, normalized overlap diagnostics, and the edit bound. JSON input or verifier failures return `{ "status": "error", "error": "..." }` and exit 2. Text output presents the same provenance and measures without policy interpretation.

Each `SourceMatch` contains the caller's source ID, SHA-256 of the **original UTF-8 passage bytes**, longest exact contiguous normalized substring (text and character count), and overlap fractions between 0 and 1. `MemorizationDiagnostic.sources` includes every source, sorted by ID. Best-source selection is deterministic: longest exact substring, then n-gram, word, character and available edit similarity, with lexicographically smallest ID breaking ties. Duplicate or empty IDs and an empty passage list are rejected.

Normalization is `NFC+casefold+whitespace-collapse-v1`: Unicode NFC, Unicode casefold, then Unicode whitespace splitting and joining with single spaces (trims edges). Punctuation remains; the returned exact excerpt is **normalized**, not a byte-offset into the source. Character and word overlap are multiset recall relative to continuation characters and Unicode `\w+` words. Configurable `ngram_size` (1–10; default 3) computes the fraction of **distinct continuation word n-grams** appearing in the passage; if no n-grams exist the fraction is zero. Empty continuation character/word denominators yield zero. An empty/short continuation can therefore produce zero overlap without implying non-memorization.

Edit similarity is `1 - Levenshtein_distance / max(normalized_character_lengths)` for the whole continuation and whole passage, not the best matching window. It is `null` when **either** normalized string exceeds `max_edit_chars` (default 512, allowed 1–2048) to bound quadratic work. Longest exact substring and other overlap metrics remain available for longer passages. Scores from passages of different lengths or sampling policies are not a calibrated probability; retain passage IDs, raw digests, model/checkpoint identity and sampling context when reporting observations.
