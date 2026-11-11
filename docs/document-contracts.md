# Document loading and portable experiment boundaries

LAB-72 foundation, audited against main
`e58ca181221c330f3c1d59c89c84628f5853e11a` (post-PR118).
This is the first implementation slice of
[LAB-72](https://app.notion.com/p/3f83b68cca22815bb508c75be4c158c1),
not completion of the portable experiment milestone. No open PRs overlapped at
branch creation. The owner assignment supersedes the cards' planning-only status.

## Shipped boundary

`documents.read_document(Path, exact_decimals=False)` is the existing strict
experiment decoder extracted into a dependency-neutral module. It returns a
string-keyed mapping; JSON suffix selection remains case insensitive and all
other suffixes use safe YAML. Nested duplicate keys, non-finite numbers (including
JSON float overflow), unsafe tags and non-mapping roots fail. YAML merge keys
remain unsupported on this strict route. Decimal mode preserves authored numeric
precision for existing policy consumers. Repeated aliases remain valid; cyclic
sequences now receive a source/field error instead of recursive traversal.
Duplicate YAML diagnostics include line/column; non-finite values include the
field path; invalid key types now include the source path. The existing
`experiment document` diagnostic prefix and root/key TypeError category remain.

The decoder owns no schema, defaults, filesystem path interpretation, hashing,
artifact verification or runtime. A future plugin field named `path` stays data.
`experiments.plan.read_document` refers to the same function for existing callers;
there is no second parser. Internal parser-only consumers now import `documents`
directly: plans/Campaigns, lab candidate/seed preflight, derivation, source and
family declarations, evaluation suites/panels/readiness, corpus policy and
memorization declarations, preparation benchmarks, iteration, research lint and
recovery manifests/provenance/snapshots. Existing raw-byte/hash readers keep their read/verification
sequence and only move the parser import. `recovery.evidence.export_evidence`
retains its import through the plan re-export because its whole function body
is bound by reviewed verifier-exclusion signatures; those signatures are unchanged.

The native permissive YAML loaders are deliberately not migrated in this slice.
In particular RunConfig supports YAML merge defaults and explicit override
precedence, which the strict experiment reader never supported. Replacing it
blindly would break accepted inputs. Existing native schema validators remain
authoritative for unknown fields, types, versions and cross-field rules.

## Actual entry routes and ownership

| Route | Parsing / validation authority | Binding and effects |
| --- | --- | --- |
| Installed console / `python -m klab` | `cli.entry.main`: global work/token prefixes; `runtime env` → `runtime_env_cli`, `lab-job` → immutable-manifest CLI, remainder → `cli.main` | Prefix routing preserved. `runtime env --help` stays framework/credential free. Full application CLI currently imports Torch; this slice does not claim to fix that preexisting coupling. |
| `train`, `stage`, `inspect`, data/tokenizer operations | `config.loading.load_config`, `_load` / `RunConfig`, `TokenizerTrainConfig`; `runtime_profile` also uses `_load` | Existing `_resolve_paths` and model defaults; explicit source-relative fields, no new binding policy. Execution stays in native handlers. |
| `config migrate`, `config derive --set FIELD=JSON` | Native v1 migration; `derivation.parse_assignments` strict JSON plus shared compiler patch validation; derive source uses strict document read then native config | Duplicate/overlapping assignments fail in their existing authorities. Derive is an exclusive publication handoff, not an execution plan. |
| `try` | `TryConfig` presets/YAML, strict override; `lab_mode.read_candidate` strict delta/full-config classification; native baseline RunConfig | Source-relative try selectors; `_config` anchors typed defaults for derivation. `run_try` directly prepares/trains/scores/probes and seals records today. |
| `probe`, `explore`, `report`, `compare` | `probes.config`, `explore_config`, `report_config`; sealed readers/result resolver | Each retains its own empty-file/preset/default/path policy. Probe inherits saved backend; explore defaults CPU. Reports/comparisons remain evidence readers. |
| Experiment plans / matrices / studies | Plan shared decoder + `ExperimentPlan`; matrix `_UniqueKeyLoader`; study `_StudyLoader`; compiler patch/axis/comparison validation | Plan parsing does not open `base_run`; consumption resolves references. Lock/prepare/dispatch remain distinct actions. Matrix/study decoder unification is future work. |
| Campaign / recovery / workers | Campaign schema + shared reader; recovery schemas + shared reader; existing canonical JSON/wire readers | Existing DAG, approvals, reconciliation, checkpoint binding, replay and receipt authority retained. Not a new DAG or executor. |
| Runtime registry / resource envelope / corpus and historical readers | Registry has its own unique loader; envelope uses native safe YAML; several corpus loaders use safe YAML; historical receipt/manifest JSON readers are specialized | Migration requires individual compatibility fixtures. Do not replace canonical/signature or legacy readers with a generic config loader. |

The authoritative command inventory remains [launch contracts](launch-contracts.md)
and [location/runtime controls](location-runtime-controls.md). This table inventories
parser families rather than promising that all commands already share one schema.

## Computation versus workflow coupling

`model/` owns dense attention, FFN, MoE, memory and tensor behavior. A static import
fence forbids direct workflow/research/CLI consumers there. The shared decoder has
a fresh-interpreter import fence against schemas, lab modules and optional heavy
packages. Neither fence proves the entire repository is decoupled.

Remaining concrete seams: `lab_mode.run_try` chains derivation, preparation,
training, evaluation and probes; study collection directly invokes capability
and inference evaluation; `CampaignEngine` dispatches typed stages and verifies
dependencies. `training.trainer` and `data.tokenizer` import experiment artifact
verification; trainer and PyTorch engine know research portability manifests;
packing/coverage use experiment source-compatibility helpers; attempt budgets
and receipts consult CampaignEngine. These require typed operation/artifact
adapters, not moving computation into YAML. Future resolved plans must own
connections/order and the existing executor must follow them. None of those
workflow chains is replaced or advertised as portable by this extraction.

## Compatibility decisions

[LAB-67](https://app.notion.com/p/3f73b68cca228175b73dfa9772daa79e): RunConfig v2
continues to coerce `true` → `1`, string `"7"` → `7`, and integral float `7.0` → `7`.
Characterization covers programmatic validation, YAML/JSON loading, derivation
and resulting config identity. `TryConfig.seed` and direct try overrides remain
strict and reject these values before resource setup. Existing inclusive bounds
`0..2**32-1` and integer hash goldens remain. Callers also include inline plan
RunConfig, compiler patches, saved-config/checkpoint reconstruction and CLI native
loaders; they must not acquire a stricter type by accident.

Proposed next contract: an explicitly versioned portable request should validate
exact integer seeds before adapting to the native RunConfig API, while native v2
and historical read-only routes retain their current acceptance. This is a design
route, not an implemented new schema/version or a duplicate runner. A future
portable schema must reject unsupported versions explicitly. Tightening existing
RunConfig v2 or removing merge acceptance needs an owner compatibility decision
with external-caller impact; no such decision is needed for this extraction.

[LAB-66](https://app.notion.com/p/3f73b68cca2281c98f0bdbe641816503): mixture paths
still contribute to historical config identity as before. Same bytes in different
roots are not newly declared identical. No exclusions, hash algorithms, receipt
bytes, fixtures or historical records change. Content-versus-location identity
for future portable contracts remains a separate versioned decision.

Verifier-authority closure digests do change when their source modules change,
including the addition of `documents` to the affected closures. Existing warm
proofs bound to the old authority fail closed and require cold re-verification;
successful verification can issue a new proof for reuse under the current
closure. This is expected operational cache invalidation, not a change to
historical artifact/config identities or a rewrite of old proofs. Hash algorithms
and reviewed exclusion signatures remain unchanged. `file` and `prepared_array`
closures do not include `documents`.

## Change-impact test map

| Layer / capability | Regression evidence and when to expand |
| --- | --- |
| Decoder component | `test_documents`: YAML/JSON parity, opaque future fields/paths, duplicates, finite/exact values, aliases/cycles, source diagnostics, no schema/runtime imports or output. Any decoder change runs this plus its actual consumers. |
| Interface / legacy | `test_seed_bounds`, `test_config`, `test_derivation`, `test_try_config`: coercion versus strict overrides, bounds/hash goldens, defaults, merges, unknown fields, exclusive outputs and paths. Future schema changes expand every relevant loader/CLI route. |
| Planning | `test_experiment_plan`, `test_experiment_derivation`, Campaign plan/recovery tests; new parse-only test proves missing artifact is not opened and schema validation remains separate. Graph/port changes require full affected graph/executor tests. |
| Component direction | Static `model/` import fence; existing dense/MoE/memory tests remain in normal fast CI. Tensor/FFN internals require focused numerical tests, interfaces and tiny integration; shape/token/state changes expand downstream coverage. |
| Identity / history | `test_manifest`, contract fixtures, seed hash goldens, research lint's exact retained historical records. No rehash or rewritten evidence. |
| Runtime / CLI | `test_cli_entry`, `test_location_runtime_controls`, workdir/runtime selection suites retain prefixes, credentials, anchoring and pre-dispatch validation. PyTorch/MLX and backend restrictions remain existing schemas/native capability checks. CPU CI is not GPU or MLX execution evidence. |
| Executor / recovery / science | Existing safe/fast tests cover cancellation, recovery, replay, receipt and comparison controls; lab-loop provides the tiny synthetic tokenizer→try→probe→report integration plus explorer/dashboard. No CI gates are removed. |

Validation uses the pre-provisioned locked environment (`uv run --locked --no-sync`),
offline flags, hidden accelerator devices and disposable external test roots.
Required automatic CI remains safe, fast and lab-loop. Full/nightly, optional
accelerator suites, downloads, actual-host runs and scientific portability proofs
are outside this slice. `tests/known_failures.txt` retains existing issue #68/#69
skips unchanged; report actual skips separately from passes.

Remaining LAB-72 work: native-loader policy consolidation with explicit legacy
compatibility, typed request/version contracts, full CLI import/effect isolation,
expanded component/adaptor boundaries and consumer parity. Remaining milestone
work: portable identity/contracts, plan-owned composition/preflight, evidence
rendering and independently admitted external TinyStories CPU/GPU proof. This
slice changes no runner assignments/holds, experiment retirement, workshop,
website, release or tag.

The first strict portable descriptor and explicit declaration context are documented
in [portable recipe inspection](portable-recipes.md), including the native compiler
adapter, identity boundaries, unsupported features and its regression map.
