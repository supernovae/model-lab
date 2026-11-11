# dense-lm-v1 preprocessing observation

## Stall and storage

The first tokenizer build and data-preparation commands were launched concurrently. The user observed approximately 25 minutes for tokenizer training and 24 minutes for data preparation; the subprocess records subsequently reported 3,589.41 seconds per job and exit status 143. Both emitted their expected artifact paths, but process inspection found Python children sleeping in `futex_do_wait` with 0.2%/0.4% CPU, and the user observed no active disk or network I/O. The jobs were stopped by PID (`44454`, `44443`, `44816`, `44813`) after confirming those exact processes. The outputs were retained and verified; no unrelated process or `runs-rocm/` user data was touched.

The repository, `artifacts/data`, tokenizer artifact, and observed Hugging Face Hub cache under `/home/byron/.cache/huggingface/hub` are on `/dev/sdd`, ext4, under `/home`; none is under `/mnt/*`. The configured TinyStories streaming cache is `artifacts/data`, also on that filesystem.

## Serialized artifact verification and instrumented preparation

Tokenizer artifact verification succeeded before the serialized data preparation call:

- source/revision: `roneneldan/TinyStories` / `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`;
- selected tokenizer documents: 5,954; UTF-8 input budget: 5,000,000 bytes; vocabulary: 8,192;
- tokenizer file SHA-256: `6bd2b7c7086c85bcac66ba5d17a2f822805fab9f1c6fe7973d8569f4fb22b44e`;
- tokenizer content SHA-256 (canonical tokenizer JSON): `db49cb0dc20c6e160f7ca0b583d217e73bc5d808627bdeb7cbd83e5d9892d585`.

A final serialized rebuild after the stderr telemetry change completed in 16.40s from the warm local dataset cache. Train: 20,000 documents, 17,915,943 UTF-8 bytes, 4,382,585 tokenizer IDs before EOS, 4,402,585 packed IDs; dataset initialization 1.6s, source iteration 6.189s, encoding 4.661s, Python/list construction estimate 0.999s. Validation: 310 documents, 265,346 bytes, 65,317 pre-EOS IDs, 65,536 packed IDs; initialization 0.8s, iteration 1.862s, encoding 0.067s, construction estimate 0.009s. Source-content SHA-256 remained train `21d41f621ada5b54ee68ec6df5ad14714ffdb8bad6ecf2805c25179703de1849`, validation `f8eea7d1cb36f1d6e94bad41031fcca05ecadc8326d1e043d08a8993b954da82`. Train/validation array SHA-256 remained `1e9eb838ee865c5cbb9bc869e3c3dc4d00c1b908f432f9e84dd605d974339a46` / `3dac074c832502f92cf00d516c4fb83b5732b8e651e84b658fa7dfe69f4a56fd`; supervision-array SHA-256 also remained identical. Final artifact `artifacts/data/24a1c0e80617b06c` binds source identity `927dca9db1746ce2909013763980c7403b0b165ac59ef559a1f78b0eac4d6650`; manifest file/content SHA-256 `62dea7279347fc4a220ce3c0692997dc3725cc589e94118f912d3c6f4f3ba964` / `a3031a9aec99f104a465a64c198aaf0e5467b19d6c134771dd8315fba00a37de`. Earlier cache directories were retained, not rewritten.

Train rates from this final run: 1,688 documents/s, 1.51 MB source bytes/s, 940k pre-EOS tokenizer IDs/s, and 372k packed IDs/s. Validation: 160 documents/s, 137k source bytes/s, 971k pre-EOS IDs/s, and 33.8k packed IDs/s. Write/fsync and each array-hash stage rounded to 0.0s in emitted telemetry; treat these as below timer resolution, not zero-cost operations.

These are operational timings only. No preprocessing optimization has been accepted. Per the requested sequencing, comparative performance profiling and A–E optimization experiments remain deferred until the reference reaches a completed/failed lifecycle state. Progress JSON-lines and 30-second heartbeats are emitted to stderr so command stdout remains available for machine-readable result payloads. `klab data prepare` now requires a matching, complete tokenizer manifest before it initializes the dataset stream.

## Deferred profiling

The first tokenizer/data builds were reported by the user as approximately 25/24 minutes. Their wrappers each later returned after 3,589.41 seconds in `futex_do_wait`; those wrapper waits do not establish that tokenizer or data computation consumed that whole time. No active I/O or network activity was observed. No first-build stage breakdown was captured.

Detailed tokenizer stage profiling and comparative optimization experiments A–E remain deferred until the reference reaches a completed/failed lifecycle state. The new JSON-line instrumentation separates dataset initialization/iteration, tokenization, list/array construction, writes/fsync, hashes, and cache validation; no wall-time checks were added to CI. Preserve the pinned tokenizer/data/source identities when later comparing candidate preprocessing changes.
