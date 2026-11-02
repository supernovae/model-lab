# Matched story packing

Opt in through `training.matched_stories`; omit it to preserve default fixed-block
packing, shuffling and config identity. This experiment uses the native trainer,
checkpoint cursor and summed cross-entropy / total valid targets normalization.
No attention API changes are required. Only plain all-token EOS-delimited data
and models without external memory are supported. EOS must be reserved for
story boundaries; source text containing literal EOS special tokens is unsuitable.
Prepared bytes and validation packing remain unchanged.

Both arms walk source-order documents, repeating that order each pass. A group
closes at the first document end reaching `targets_per_update` (4096 by default),
or at the corpus end. A long document can therefore make a larger group. Each
update uses the same group in both arms; `micro_batch_size` only partitions rows
within the update. `gradient_accumulation` does not size matched groups. Do not
interpret the requested group target as an exact fixed effective batch.

`layout: story` starts independent right-padded rows for each story. Long stories
continue in ordered chunks of at most `seq_len` targets with the preceding token
as context. `layout: contiguous` fills rows across story boundaries within the
same group. Both include all stored tokens (first token and every EOS included),
using a virtual preceding EOS for the first stored token. Padded labels are
`-100`; a clipped final update retains exactly the same target prefix in each
arm. Padding and context tokens do not count toward the budget. This matched
control is a new protocol, not a replay of the default shuffled block loader.

## Freeze the common plan before launch

Use the already prepared training array and matching tokenizer on the runner.
This only indexes tokens; it does not train or fetch data:

```python
import json
import numpy as np
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import sha256_file

tokens_path = prepared_root / "train.npy"  # Path to verified prepared inputs
ids = np.load(tokens_path, mmap_mode="r")
eos = load_tokenizer(tokenizer_path).token_to_id("<eos>")
plan = MatchedStoryDataset(ids, 512, eos, layout="contiguous", targets_per_update=4096)
manifest = plan.group_manifest(100_003_840, train_tokens_sha256=sha256_file(tokens_path))
print(json.dumps(manifest, indent=2))
```

Retain this manifest with the experiment declaration, compare it across arms,
and set **both** `training.max_steps` to its `max_steps`. The trainer rejects an
incorrect horizon; do not assume 24,415 updates. Set `optimizer.decay_steps` to
that same common horizon and freeze a shared `optimizer.warmup_steps` before
calibration. Public configuration overlay (replace `HORIZON`/`WARMUP` with integers):

```yaml
training:
  seq_len: 512
  micro_batch_size: 1
  gradient_accumulation: 1
  max_tokens: 100003840
  max_steps: HORIZON
  matched_stories:
    layout: contiguous  # story in the other arm; sole experimental delta
    targets_per_update: 4096
optimizer:
  name: adamw
  peak: 0.00006
  floor: 0.000006
  warmup_steps: WARMUP
  decay_steps: HORIZON
```

Apply these fields to the full runner-owned configuration. Use the existing
native weight-promotion path for each fresh arm, not full-state resume from the
parent: fresh matched optimizer, RNG and counters; identical seed and all other
settings. Parent checkpoint SHA:
`085adc01f1512dc41570be869d471b184d84decb3a80e453bb83d3bef43ccc89`;
weights SHA:
`e667a81daa8df8f3feaf72fd5a7cff3b7bce95473cf3518515600fef3d66db4d`.
Keep the 50,556,800-parameter model, BF16 reference attention and runner-frozen
evaluation unchanged. This guide does not launch training or alter evaluation.

Within an arm, ordinary full-state resume restores `(epoch, next_group)` in the
existing cursor. Config identity binds layout and target grouping; prepared
input identity binds token bytes. Changed layout/group size/horizon is rejected.
Exposure records identify this protocol and its group-offset digest separately
from default packed-block exposure. The legacy exposure `eligible_blocks` and
cursor field `next_block` count **groups** under this opt-in protocol.

## Validation and limits

Synthetic tests cover target coverage, long-story endings, clipping across
passes, matched per-update counts, padding loss/gradient and native interrupted
resume. They establish wiring, not quality or accelerator throughput. The local
NumPy-only check covered 249 budgets. On a 200,000-token synthetic array (1,000
stories of 200 tokens), both layouts used 48 groups with 8,392 bytes of NumPy
indices; story layout emitted 1,000 rows in 0.0069 s, contiguous 429 in 0.0027 s.
These are illustrative indexing measurements, not model-throughput estimates.
Index construction also temporarily holds document ends in a Python list.
Rows for one update are materialized; exceptionally long documents increase
host memory and padding can substantially increase model compute. The ROCm
runner must calibrate the frozen shapes before the full pair.
