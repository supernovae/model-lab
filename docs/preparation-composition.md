# Compiled native preparation bindings

The first LAB-74 slice compiles `tokenizer_train` and `data_prepare` bindings from
the existing `CampaignPlan`. `compile_preparation(plan)` is a pure projection,
not a scheduler: `CampaignPlan` still validates producer kinds, dependencies and
cycles and supplies the stable topological order. Other stage kinds retain their
native routes. No new CLI, external recipe workflow profile or execution grant
is introduced.

Each immutable operation records its stage ID, operation version, declared config
reference, dependencies, typed input ports and output contract. Tokenizer and
snapshot inputs point to producer stages' artifact outputs. Future output bytes
have no digest or availability claim at compile time. Compiling opens no configs,
artifacts, stores or runtime metadata and does not import optional computation.
It does not certify content, resource availability or admission.

At dispatch or receipt verification, CampaignEngine selects the compiled
operation, loads the native config and uses its existing `_upstream` verification
to authenticate each selected producer. It copies those outputs and locations
into an immutable `PreparationRequest`. Request validation retains the native
config-versus-snapshot/tokenizer path checks. The adapter receives neither an
engine nor the entire stage/row map, and cannot choose another producer or
invoke a downstream workflow operation. It calls native tokenizer training or
packing and native artifact verification. The engine translates its typed result
back to the existing receipt shape.

The preparation operation consumes an already authorized execution context;
compilation and a well-typed request do not authorize execution. Existing apply,
claims, resource, provenance, runtime and attempt gates remain in force. The
snapshot acquisition branch is unchanged in behavior and remains in the native
orchestrator, including interrupted-journal resume and verification of previously
published snapshots. No acquisition path is added to compilation.

Two supported graphs exercise the same data-preparation port:

- `tokenizer_train → data_prepare`: use a newly produced native tokenizer.
- `artifact_reference(tokenizer) → data_prepare`: reuse a declared tokenizer.

The choice belongs to the authored graph and compiled binding, not an adapter
branch or a dispatcher patch. Optional snapshot edges remain explicit. Extra
control dependencies remain in the native graph and are not mistaken for data
ports.

## Compatibility and dependency boundary

The projection and requests are internal immutable values, not additions to v1
serialized campaign plans, receipts, configs or experiment locks. Native stage
IDs, declaration digests, result shapes and scientific hash algorithms remain
unchanged. Golden declaration digests were captured using CampaignPlan from
`b264f41ae560af5767cbcbd3fdc861edce43c914`. Installed source and verifier closure
digests naturally change with implementation; prior proofs still require the
existing authority check. No pinned exclusions or historical evidence is edited.

Readiness-policy computation imports are now local to its existing operations so
CampaignPlan parsing and preparation compilation do not load corpus/data code.
The schema and readiness computation are unchanged. Its token-only no-scan test
now fences the ancillary scanner itself instead of a former eager import alias.
The archive crash injector now intercepts only publication to the final archive
path, allowing newly deferred dependency imports to perform their own link probes;
all three publication crash assertions remain intact.

This is a narrow dependency-direction guarantee for the migrated adapters.
Native artifact/source-compatibility checks remain intact. The broader trainer,
packing and tokenizer dependency inventory in [document contracts](document-contracts.md)
remains open; this slice does not claim repository-wide component separation.

## Change-impact test map

| Boundary | Evidence |
| --- | --- |
| Native graph → typed projection | Two distinct graphs, native ordering, missing/wrong edges, cycles, duplicate producers, version and port refusals; campaigns with no preparation stages |
| Compiled binding → engine request | Dispatch spies reject decoy stage config/producer fields; immutable copies prevent row mutation changing a request; zero/two upstream output refusals |
| Request → native operation | Config/path/type checks; exact native computation and artifact-verification spies; native tiny synthetic train/reuse parity |
| Metadata-only compilation | Fresh-process import/write/network/process fences, including forbidden config/artifact reads |
| Legacy formats and lifecycle | Pre-change declaration hash goldens, unchanged result shape, native replay, recorded versus reverified identity mismatch, prepared-output location refusal before artifact access, parent binding, snapshot interruption and verifier-authority checks |
| Readiness / runtime / recovery | Existing focused policy, runtime, campaign recovery tests; normal safe/fast/tiny CPU lab-loop CI |

Focused mutation checks remove each of the replay identity, upstream cardinality
and prepared-output location guards in isolation. The new tests detect every
removal (two, two and one failing cases respectively); source is restored before
normal validation. These checks exercise refusal boundaries without model work.

LAB-74 is not complete: typed bindings for other operations, full compiled-plan
coverage, broader dependency-direction gates and recorded runtime-capability
preflight remain. The first [LAB-73/PR B workflow slice](workflow-recipes.md) adds explicit external
prepared-input bindings and native lock publication. Preparation-graph context,
evaluation closures, admission and evidence linkage remain open. Actual external CPU/GPU portability proof remains separately
admitted later work. Historical seed and mixture identity decisions are unchanged.

[Portable preparation planning](preparation-recipes.md) now supplies inventoried
external graphs/configs to this same compiler. It adds no engine admission or
execution route; the remaining execution/provenance boundaries still apply.

For a smaller whole-document window of an already encoded all-token local corpus,
use [verified prepared prefixes](prepared-prefix.md) to avoid encoding it again.
