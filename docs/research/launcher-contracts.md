# Research launcher contracts

For an ordinary comparison use `try`, then `report` and `probe`. For mechanism
shapes use the [lesson path](lesson-paths.md). For a matched architecture matrix
use `research scaffold`, then native `study plan`/`submit`/`collect`/`report`.
Scaffolding writes declarations; execution is a separate explicit action.

This audit covers the one-off launcher and teaching overlap requested by the
[historical simplification plan](https://github.com/supernovae/model-lab/blob/5df1193198223bee50661e2db65a7f01cc8372c0/TODO.md),
against base `6762454de0337a0ea3b909e0487cdaebe02fb182`. The
[current roadmap](../../ROADMAP.md) owns direction. The requested set has 18
executable leaves before this change and 17 after it (including all four `learn`
and four `study` verbs). This is a bounded removal and justified retention,
not completion of the wider CLI reduction target.

## Consumer and semantic audit

| Path | Actual consumers and contract | Decision |
| --- | --- | --- |
| `research portability build` | [Compiled-world guide](engram-portability.md) and [learned protocol](learned-engram-portability.md); materializes disjoint worlds, assets, immutable protocol, allocation and coordinate identities. Learned mode additionally binds source-learning and recipient preparation. | Retain. Generic `research scaffold` is a config/recipe sketch; it rejects the learned protocol. `campaign` has no stage for these world/asset builders. |
| `research portability plan` | Those guides; compiled mode reports per-coordinate estimates and receipt-based calibration, learned mode reads sealed resource selection. | Retain. Generic Campaign dependency planning does not express these resource projections. No training or calibration is triggered by plan. |
| `research portability run` | Those guides; compiled mode selects one explicit arm, validates recipient/preparation identities, audits update membership and frozen assets, evaluates checkpoint and A/B/A replacements, and records failures. Learned mode seals source/preparation gates, resource allowance and immutable observations. | Retain. Native `train` supplies optimizer/checkpoint work, but does not supply the surrounding experimental protocol. Replacing this with a matrix or arbitrary shell stage would lose contracts. |
| `research portability continue` | Those guides; compiled mode visits pending arms in declared order, keeping immutable failure attempts; learned mode resumes under the already sealed allowance and checkpoint/asset identities. | Retain. Generic resume cannot infer this protocol's pending coordinates or permit a new allowance. Existing retry behavior is unchanged; failed attempts remain visible. |
| `research portability report` | Those guides and `study report --portability-evidence`; builds a strict evidence snapshot and, for learned mode, a static verified report bundle. | Retain. It writes derived evidence/report artifacts without training. General `report` reads evidence; it does not generate this specialized envelope. Partial, failed, censored and unavailable outcomes must remain separate. |
| `research portability probe-byte` | Only the compiled-world guide; wrapper over `byte_scheme` and `table_address` for arbitrary raw UTF-8 text. No internal runner, test or historical-record consumer calls it. | Remove handler and parser outright. The guide already requires the builder's `assets/address_observations.json`, which binds actual experiment addresses. Hash behavior remains in native byte hashing and its tests. There is no replacement arbitrary-text CLI or compatibility alias. |
| `reference pythia trajectory` | [Model scaling](../model-scaling.md); official step0/10000/143000 revisions on one pinned FineWeb-Edu validation slice, exclusive identity report, safe snapshot allowlist. | Retain. `probe ref:pythia-70m-deduped` scores text-level fact/lm-eval probes of one registered revision; it neither selects a trajectory nor measures this pinned validation population. Shared snapshot security is already native. |
| `research tasks build` | [Capabilities](../capabilities.md), [allocation](../history/path-domain-corpus.md), research README and capability missing-input guidance; math/Python disjoint splits and cards, pinned Wikidata acquisition, or allocation ownership/provenance audits. Exclusive publication prevents overwrite. | Retain. Corpus preparation and synthetic generation do not produce these card/test/source and allocation contracts. Wikidata can acquire source data: this is not a read-only discovery command. |
| `research corpus mine` | [Capabilities](../capabilities.md); explicit local training split/tokenizer, bounded exact token n-gram/address/entropy statistics, content identity and exclusive output. | Retain. `corpus describe` and `measure-tokens` report corpus/build and token-budget evidence; it does not measure configured tokenizer n-grams, table collision/reuse or memory dimensions. No held-out data is discovered. |
| `learn list` / `describe` / `scaffold` | [Lessons](lesson-paths.md), dashboard lesson command generator, semantic-retrieval and portable-Engram guides; metadata or a standalone config/artifact lesson. Portable lessons verify and copy a supplied package; semantic lessons supply structured vectors. | Retain. A single mechanism or artifact lesson is distinct from a paired/factorial research matrix. Neither route trains or implicitly downloads. |
| `learn probe` | Lesson guide, dashboard and generated standalone model lessons; bounded freshly initialized CPU FP32 reference forward and shapes/lookup diagnostics. | Retain. `probe`/`explore` consume saved checkpoints; `model exercise` requires a saved run/checkpoint. None replaces an initialized mechanism lesson, and its observations are not learned behavior or hardware qualification. |
| `research scaffold` | Research recipe guides, `papers.md`, samples and contributor guide; writes config/matrix/study definitions and source digests with lifecycle diagnostics. | Retain. It generates declarations consumed by native study dispatch; it is not another trainer. Remove its duplicated standalone lesson workflow and the six recipe guides' redundant lesson scaffold commands. |
| `study plan` / `submit` / `collect` / `report` | Generated research instructions, architecture guides and historical report verification; matched pair/card identities, exclusive submission receipts, collected observations and verified static report bundles. | Retain. Generic matrix submission lacks study pairing/card contracts; general `report` does not collect or publish study bundles. See [launch contracts](../launch-contracts.md). |

The comparison inspected the actual CLI adapters, `research/{scaffold,pythia,
portability_campaign,portability_runner,learned_portability_campaign}.py`, native
Campaign stage schemas/dispatch, reference probes, task/allocation builders and
lexical mining. Historical records are consumers of artifact formats and source
identities; they are not instructions to restart paused experiments.

## Validation and retained guarantees

A `tests/test_research_launch_surface.py` module, since retired, validated the removed command through
both entry routes before runtime preparation, parsed every retained leaf and all
generated study examples without calling handlers, and checked a single
acquisition → tokenizer → preparation sequence. Snapshot scaffold tests keep
source-relative paths and acquisition order. Documentation validation needs no
model, tokenizer fitting, dataset acquisition or provider.

No correctness tests are deleted. The retained native coverage includes:

- `test_byte_hash.py`, `test_memory_hash.py`: UTF-8 address identity, positive
  table bounds, nonconstant/coprime and changed-scheme behavior.
- `test_portability_campaign.py`: real asset/address compilation, collision
  rejection/no-clobber and exact replacement-path reproducibility.
- `test_learned_portability_campaign.py`: shared control provenance and
  checkpoint-owned asset paths without changed settings.
- `test_pythia_reference.py`, `test_hf_auth.py`: pinned revisions, safe snapshots,
  no remote code/pickle, exclusive report writes and sanitized acquisition errors.
- `test_phase_e_tasks.py`, `test_lexical_mining.py`: split disjointness, pinned
  source identity, tamper rejection, bounded exact statistics and train-only reads.
- `test_research_workbench.py`, `test_scaffold_snapshot_sources.py`,
  `test_study.py`, `test_study_reporting.py`: no-clobber scaffolds, bound configs,
  matched study identities and historical report/tamper verification.
- `test_research_lint.py`: original record bytes, exact pinned legacy paths and
  history map; additional or edited legacy artifacts remain rejected.

The implementation leaves `experiments/research/`, `artifacts/`, `data/`,
`configs/`, packaged research resources, verifiers and attempt/Campaign contracts
unchanged. Historical allocations grant no new compute or promotion authority.
Normal safe/fast/lab-loop CI is wiring evidence, not scientific qualification.

Fast validation exposed an existing Forge fixture isolation defect, reproduced
on the unchanged base: native preparation redirected `tempfile.tempdir` and
`TMPDIR`/`TEMP`/`TMP` into the subsequently renamed source workspace. The fixture
now restores those process settings through pytest's existing monkeypatch
teardown. The reproducing Forge → snapshot-verification sequence keeps both
identity checks intact; no runtime behavior or gate is weakened.

## Follow-up boundaries

Further launcher removal needs an explicit choice to retire the distinct
operation, or native declarations that preserve its contracts. In particular,
portability needs authenticated world/asset preparation, source/preparation
gates, update audits, A/B/A observations and cumulative continuation accounting;
Pythia needs pinned multi-revision validation observations. These are
consolidation gaps, not authorization to implement a new runner or execute a
protocol. Bring a concrete active use case and regression map to that decision.
Keep these choices separate from CLI path budgets, the later usability assessment,
reference resealing and the original-host evidence pending in issues #68/#69.
