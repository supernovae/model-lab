# Derive a bounded window from an existing encoded cache


For an all-token `local_text` corpus, the Python preparation API can derive a
whole-document prefix without encoding the corpus again. Set **both**
`dataset.train_max_documents` and `dataset.train_max_tokens` to the exact desired
boundary. A token cap alone does not limit the selected document count; native
preparation correctly rejects the next document if it would cross that cap.
Keep validation selection unchanged. Both splits must still contain a full
next-token block for the configured sequence length. Match the training budget and group endpoint
separately using the supported [continuation rules](training.md).

```python
from pathlib import Path
from klab.config.loading import load_config
from klab.data.packing import PreparedPrefix, prepare_data
from klab.data.tokenizer import load_tokenizer

config = load_config(Path("bounded-baseline.yaml"))
data = prepare_data(
    config,
    load_tokenizer(config.tokenizer.path),
    prefix_from=PreparedPrefix(
        root=Path("/absolute/path/to/verified-parent-cache"),
        manifest_sha256="<manifest digest from the prior verified preparation record>",
    ),
)
print(data.root, data.manifest["manifest_sha256"])
```

The parent pin is an explicit choice of previously prepared data, including its
recorded producer. The API cold-verifies the parent's arrays and manifest,
verifies current raw-source hashes and tokenizer identity, rejects changed
selection/packing settings, and checks the complete parent and prefix EOS counts.
It accepts only complete, unskipped documents with no embedded EOS tokens, masks,
allocation, byte sidecars or release-export bindings. The parent must have the
**same preparation source fingerprint**; a pinned hash authenticates old bytes but
does not establish equivalence between different encoders or preparation versions.
After a preparation-code upgrade, first perform one native preparation under that
version. Subsequent bounded windows can reuse it. Parent identity and manifest are
retained in `prefix_derivation`; ordinary preparation recognizes the result under
the same identity as a freshly encoded selection.

The operation reads/hashes source files and parent arrays, scans EOS counts in
bounded chunks, then copies the prefix and unchanged validation with native atomic
writers and sealed verification receipts. It needs space for the complete output
copy; it does not save I/O by retokenizing, linking mutable arrays or rewriting
existing manifests. Failed streaming chunks and failed derivation staging are
preserved. Pass the existing `resource_envelope` parameter to apply host/storage
limits. Measure preparation separately from subsequent training or evaluation.
