# Source, definitions and runtime data

Lab owns source, tests and small maintained interface examples. [Workshop](https://github.com/supernovae/kybern-workshop) owns editable model experiment definitions. No implicit copy of Workshop is stored here. Bind execution paths explicitly when rendering a Workshop plan.

Runtime corpora, snapshots, token caches, prepared bundles, logs and checkpoints belong in the configured external work directory. Use `--work-dir` or `KLAB_WORK_DIR`, otherwise `$XDG_DATA_HOME/klab` or `~/.local/share/klab`. Experiment outputs use `WORK/experiments/<experiment>/<attempt>`. Existing saved outputs are not moved or deleted by this source cleanup.

The old top-level `corpora/devmind-sample-v0` was a 16.8 KB synthetic teaching fixture, not a downloaded corpus. It is now `tests/fixtures/corpus-sample`; tests and maintained examples use that location. Historical corpus pointers, generated artifacts and personal research records are available in [Git history](https://github.com/supernovae/model-lab/tree/9d4c840ef1f102becf27532c2548c505c2ed35a6). This is not a claim that Workshop contains byte-identical archives.

Keep definitions separate from output destinations. `experiments/research/README.md` is only a navigation pointer. No new storage framework, automatic downloads or execution permission is introduced.
