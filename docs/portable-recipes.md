# Portable recipe inspection

`klab experiment recipe inspect RECIPE.yaml --json` inspects a directory
outside the installed lab checkout using the existing native configuration matrix
compiler. It reads bounded, inventoried declaration bytes and prints a preview.
It never executes the recipe or produces a resolved lock. The result explicitly
says `status: incomplete` and `executable: false`.

The matrix profile validates declarations and previews configuration variations.
It does not execute training or replace a prepared-input workflow.

A separate [prepared-input workflow profile](workflow-recipes.md) now connects
explicit artifact bindings to native verification and lock publication. It does
not change this v1 matrix profile or grant execution.

## Descriptor and declaration context

A descriptor has these fields (replace the illustrative digests with exact
lowercase SHA-256 values before inspection):

```yaml
portable_recipe_version: 1
id: width-study
entrypoint: plans/plan.yaml
contracts:
  experiment_plan: 1
  run_config: 2
files:
  - path: plans/plan.yaml
    role: experiment_plan
    sha256: <sha256 of exact plan bytes>
  - path: configs/run.json
    role: run_config
    sha256: <sha256 of exact config bytes>
  - path: notes.yaml
    role: metadata
    sha256: <sha256 of exact notes bytes>
artifacts:
  tokenizer.path: <expected sha256 of tokenizer payload>
```

The descriptor's directory is the explicit recipe root. Its sole inventoried
`experiment_plan` entrypoint uses native `plan_version: 1`, with
`base_run: configs/run.json` relative to **that root**, regardless of the plan's
subdirectory, working directory, or enclosing Git repository. Every declaration
reference must be inventoried. All inventoried members must parse as YAML/JSON
mappings, use native duplicate/non-finite/cycle checks, and match exact bytes.
Inspection snapshots those bytes before schema validation and reference binding.
Limits are 64 members, 1 MiB per document (including descriptor), and 8 MiB total.

Native typed config paths remain relative to the **config's directory**. For
example, `tokenizer.path: artifacts/tokenizer.json` in `configs/run.json` binds to
`configs/artifacts/tokenizer.json`. Paths must use canonical relative POSIX
spellings: absolute paths, `..`, `./`, repeated separators, Windows paths and
symlinks are rejected. Input payloads need not exist; their expected identities
must be listed under `artifacts`. Output/cache locations are bound but never
created. Arbitrary metadata keys named `path` remain opaque and are not followed.

Supported typed inputs are `tokenizer.path`, `model.memory_package_path`,
`dataset.{train_path,validation_path,source_manifest_path,allocation_manifest_path,
corpus_release_path,corpus_export_path}`, and `training.portability_manifest_path`.
Supported output locations are `dataset.cache_dir` and `logging.root_dir`.
Mixture path fields and path-changing axes are unavailable in this profile.
Artifacts are requirements, not authenticated content: missing inputs report
`unavailable`, existing inputs `unverified`, and runtime capability `unknown`.
Neither payload hashes nor runtime probes are calculated.

The native matrix supports axes, derived fields, and comparisons with one cell
per arm. `--max-runs` retains the compiler's expansion limit (default 1000).
Inline base configs, non-checkpoint artifact/input/corpus-variant/evaluation composition,
evaluation suites, and attempt contracts are explicitly unavailable here. Use
existing native planning for these declarations; the inspector will not silently
omit their work or claim to have compiled a full workflow.

## Identity and compatibility

The new recipe fingerprint is SHA-256 of `klab-portable-recipe-v1\0` followed
by native canonical JSON of the validated descriptor, with `files` sorted by
path. Member byte digests and expected artifact digests participate. Descriptor
formatting alone does not change this fingerprint; its separate
`descriptor_sha256` retains exact descriptor bytes. Directory relocation changes
only the displayed local binding. Recipe names and declared relative paths are
part of this v1 recipe identity; it does not claim scientific equivalence.

`installed_lab` uses the existing source identity algorithm, independently of
recipe bytes. Cell `requested_config_sha256` uses the existing native config hash.
No historical record is rehashed. Existing backend/engine/precision/source
identity distinctions and scientific comparison rules remain intact.

This strict new descriptor does **not** tighten native RunConfig seed acceptance:
`true`, `"7"`, and `7.0` retain their native coercions, while
native strict try overrides remain strict. Descriptor versions and contract
versions reject booleans. Tightening legacy seeds would need a separate versioned
contract and owner decision. No such decision is needed for this inspector.

When no explicit `RecipeContext` is passed, `declaration_reference` retains its
existing source-relative then repository-root fallback. Existing commands,
defaults, execution, cancellation, resume, replay and history readers keep their
native routes. CLI dispatch makes this inspection/help path lightweight and
read-only even with global work-directory or token-file options; those options
are not consumed by inspection.

## Author and relocate declarations with installed Lab

Recipes are YAML/JSON and optional documentation in any user directory or Git
repository. Lab owns parsing, planning and execution. Neither Workshop, our
GitHub organization, Notion nor an internal runner is needed.

Validate an authored native section without probing a runtime or opening model
or corpus payloads:

```sh
klab config validate model.json --kind model --json
klab config validate optimizer.yaml --kind optimizer --json
klab config validate run.yaml --kind run --json
klab config validate tokenizer.yaml --kind tokenizer --json
```

These commands use the existing ModelConfig, optimizer union, RunConfig and
TokenizerTrainConfig schemas. They reject unknown fields and malformed YAML/JSON,
including duplicate keys. A valid individual section does not prove its compatibility
with other sections, artifact availability, hardware capacity or scientific readiness.
They do not allocate a model, train, download or evaluate anything. Unlike
`klab inspect CONFIG`, which inspects a full run configuration through its
runtime-aware path, `config validate` only checks the selected native schema.

Check a whole project at once by naming its directory:

```sh
klab config validate path/to/project
```

Every complete run, tokenizer, model and optimizer declaration under it is
validated against its schema, and the command exits non-zero if any fails.
Partial sections, plans and other documents are counted as skipped. This is
the check a recipe repository runs, by hand or in CI, to learn whether its
declarations still load with the installed Lab; it replaces per-project test
scripts.

Assemble one run config from several files with `config compose`:

```sh
klab config compose run.yaml model=model.json optimizer=optimizer.json \
  local-paths.yaml --output "$WORK/run.yaml"
```

Parts are YAML/JSON mappings merged in order, with later parts winning.
`FIELD=PATH` places a file at that dotted field, so a model section, an
optimizer section and a file of local paths (tokenizer, cache, run store) kept
outside Git compose without a script. Relative paths resolve from the part
that declares them. The result must be a complete valid run config; it is
written beside a `.derivation.json` receipt with each part's digest. Nothing
is read from the named paths or executed. To change single fields of a
complete config, use `config derive --set`.

For an existing inventoried matrix, preparation or workflow descriptor:

```sh
klab experiment recipe export /source/recipe.yaml --output /my/new-recipe
klab experiment recipe inspect /my/new-recipe/recipe.yaml --json
```

Export copies exactly the authenticated descriptor and its declared metadata
members. It refuses existing destinations, symlinks, missing files and digest
mismatches. The parent directory must exist. The descriptor is written last;
an I/O failure can leave a partial destination, which is not silently overwritten
on retry. Export does not copy undeclared payloads or credentials, create approval,
remap external artifact bindings, or publish to Git. Never inventory secrets.
Preparation/workflow exports use their respective native inspect commands with
new explicit local bindings; inventory validation alone is not a workflow plan.
Ordinary Git can then commit/push the authored directory to a user-selected remote.

Existing `research scaffold --resource-root ROOT` composes native studies and cards
from explicit current inputs; `config derive` authors a typed variant. Export reuses
an existing descriptor. `recipe create` below authors an inventory from explicitly
selected native declarations, without requiring a descriptor first. Automatic
scientific design, broader execution profiles and accelerator/evaluation composition
remain outside that command. A missing future
checkpoint or tokenizer digest must remain missing, never a fabricated placeholder
claimed as verified. Use existing native artifact verification after real outputs
exist. Full reproducibility of a model recipe cannot be inferred from schema dry run.

## Explicit content admission outside Git

Preparation and workflow execution accept exactly one source selection:

- `reviewed_commit`: existing clean committed inventory at the exact current recipe
  HEAD; unchanged semantics.
- `reviewed_descriptor_sha256`: SHA-256 of the exact descriptor bytes, for explicitly
  selected content in a plain directory or arbitrary repository. Omit
  `reviewed_commit` entirely in this mode.

Both still require the real `recipe_sha256` and `binding_sha256` returned by native
inspection, the current CPU interpreter and every existing resource/operation limit.
For example, obtain the content selection with `sha256sum /my/recipe.yaml`, then
put that digest in `reviewed_descriptor_sha256` of the normal admission document.
No command fills approval or admits execution automatically.

The context authenticates the descriptor and every inventoried member before
execution and on refresh. Receipts report `CONTENT_BOUND` and `source_commit: null`;
this is byte identity, **not** Git provenance, author authentication or new compute
authority. Changing bytes, bindings, source-selection mode or admission invalidates
resume. Existing native artifact checks, approvals, output ownership and resource
ceilings still apply. Content mode does not permit the Git-only recovery-publication
mutation exception. Execution outputs still require an explicit root outside Git.
This makes the currently supported native CPU preparation/workflow contracts usable
without a recipe Git checkout; it does not expand their supported operations.

## Create an inventory from authored native declarations

Start with your current native plan/campaign and RunConfig/tokenizer declarations.
Use `config derive` for an existing RunConfig variant or `research scaffold
--resource-root ROOT` to generate a native study and its selected capability cards.
Lab owns these authoring functions; the resulting DSL/config/cards belong in your
experiment repository. No organization-specific checkout or Python launcher is needed.

For a matrix plan whose native `base_run` names `configs/run.json` relative to the
recipe root, explicitly select the metadata to publish:

```sh
klab experiment recipe create /my/authored-inputs --profile matrix \
  --id new-study --entrypoint plans/plan.yaml \
  --file experiment_plan:plans/plan.yaml --file run_config:configs/run.json \
  --file metadata:notes.json --artifacts /my/expected-input-identities.json \
  --output /my/new-recipe
klab experiment recipe inspect /my/new-recipe/recipe.json --json
```

`expected-input-identities.json` uses the existing matrix artifact mapping (native
input field to declared SHA256). Supply actual known identities; never invent
future hashes. Inspection still labels payloads unavailable or unverified.
Creation is inventory validation, not a claim that the plan's operations or
payloads have been verified. The second command performs native planning.

The same command supports the existing `preparation` and `workflow` profiles:

| Profile | Entrypoint role | Other roles/options | Next native operation |
| --- | --- | --- | --- |
| matrix | `experiment_plan` | `run_config`, `metadata`; `--artifacts` native field-to-digest mapping | `recipe inspect RECIPE --json` |
| preparation | `campaign` | `run_config`, `tokenizer_config`, `metadata`, `source_lock`; artifact connections live in the campaign | `recipe preparation inspect RECIPE --bindings LOCAL.json --json` |
| workflow | `experiment_plan` | `run_config`, `metadata`; `--artifacts` existing tokenizer/prepared-data identity mapping; optional `--campaign` inventoried metadata path | `recipe workflow inspect RECIPE --bindings LOCAL.json --json` |

Each `--file ROLE:PATH` selects a bounded YAML/JSON mapping under the supplied
root. No directory scan, arbitrary file inclusion or secret discovery occurs.
Do not select credentials. Creation hashes the actual selected bytes and emits
`recipe.json` using the existing profile schema. The destination must be new and
its parent must exist. Unsafe paths, symlinks, duplicate members, malformed
metadata and invalid roles refuse before output. Publication writes the descriptor
last; an interrupted partial directory is retained and cannot be overwritten.

You can move the directory and inspect from an unrelated cwd. Preparation and
workflow local bindings stay separate from the portable inventory; bind actual
external inputs and output locations explicitly. After independent inspection,
ordinary Git can commit the directory to your own repository. Git publication is
not performed by Lab. Follow [preparation execution](preparation-recipes.md) or
[workflow execution](workflow-recipes.md) for the existing admission, apply,
approval, resume and report commands. Do not use matrix preview as an execution
admission.

For generated inputs, preserve the existing ordering: native preparation produces
and verifies outputs, then `experiment bind-inputs` binds verified tokenizer and
prepared-data identities; workflow resolve/publish verifies and locks those
inputs before execution. A future checkpoint uses the native `from_phase` and
selector declaration with no invented path/digest. This does not reset accounting
or add an internal runner dependency.

## Derivative declaration dry run

Matrix inspection also validates the native phase graph for checkpoint-only
artifact declarations. An external checkpoint artifact must declare its immutable
identity, full SHA256, state (`full` or `weights`) and a canonical path relative to the entrypoint plan's directory.
Portable inspection refuses absolute paths and symlinks; native execution also
accepts absolute artifact paths. A derivative phase names it through `checkpoint`, with the native
`resume`, `extend_budget` or `promote` transition. Alternatively, a phase may name
an earlier producer through `parent` with its deterministic selector. The shared
native graph validator rejects cycles, missing checkpoint names and incompatible
state declarations (resume/extend require full state).

The result distinguishes `supplied_checkpoint`, `produced_checkpoint` and
`fresh_training`. Missing supplied paths report `unavailable`; existing paths
report `unverified`, even if bytes are present. Planned outputs have no claimed
SHA256. Inspection never opens checkpoint payloads, allocates models, writes
receipts or probes runtime. It validates each requested phase patch against the
native RunConfig schema; it does **not** resolve inherited parent settings,
checkpoint/config/tokenizer compatibility, budgets or evaluation results.
Phase comparisons and path-changing patches refuse rather than being omitted.
Actual native lock/execution retains all checkpoint and provenance verification.
A declared derivative is not retraining evidence or an executed evaluation.

## Supported limits

The authoring and inspection commands validate metadata for the documented
profiles. Accelerator execution, composed evaluation and derivative runtime
qualification require actual inputs and the corresponding native verification,
lock and execution paths. A metadata preview does not establish those results.
