# Portable CPU investigation: partial LAB-73/74 verification

The historical PR126 investigation snapshot was revision
`61269a2a8c7ce55b6b1a2027325f99be73b56048`; the final regression correction was
`d5ab0b4816572364870d12f9f5ab3da3a036c7d8`, merged as
`fdf1a09edc0661e107b30e9941ff7b954aa2738c`. That PR supplied partial LAB-73/74 verification. Neither card is resolved, and end-to-end
portable execution has not been proved. The retained contract coverage is:

| Card / boundary | Retained verification | Still missing |
| --- | --- | --- |
| LAB-73 inventory and bindings | Existing workflow test exercises a syntactically valid unsupported suite reference before optional imports or work | Admitted evaluation references through the same explicit recipe context |
| LAB-74 preparation admission | Changed descriptor, bindings or inventoried config after context creation refuse before dispatch or store creation | Portable model execution, native attempt/replay identity and real dispatch-to-report proof |
| LAB-75 | No change; the stopped test's planned-only report check was removed | This investigation supplies no completed execution evidence |

The existing native workflow execution-refusal tests remain in place. This adds
no new execution route, public schema, hash/seed policy or CI workload selection.
The large stopped integration and numerical pilot assertions were removed:
current missing capabilities and pilot defaults are not permanent requirements.

## Investigation finding and retained evidence

`workers/execution.py` calls `stage(..., through="warmup")` before training.
`staging.pilot_config` derives each pilot's budget from its staging setting,
independently of the main training budget. Runtime validation also performs toy
optimizer probes. A one-update main request therefore does not bound total work.

One development attempt exceeded the assigned envelope because this transitive
work was missed before dispatch: smoke 2, warmup 5, main 1 model-training updates.
The driver was interrupted. Worker receipt was COMPLETE, controller ingestion
PENDING; suite evaluation, collection and completed reporting were not reached.
No subsequent model proof had run at that revision. Exact run IDs, timings, resource readings,
retained paths and cleanup state are in the [PR126 evidence addendum](https://github.com/supernovae/model-lab/pull/126#issuecomment-6157874827).
These are development observations, not acceptance thresholds or scientific claims.

At that revision, the open decision was a separately bounded pilot-inclusive CPU
proof or a native total-budget admission change that refuses before dispatch when
all requested work cannot fit. Neither was implemented by that revision. Portable execution
also needs shared committed-inventory provenance and bounded evaluation/runtime
admission in the existing planner and request path.
TinyStories qualification, LAB-75 CLI, Workshop, website and release remain outside
this slice; historical hashes, seed acceptance and runner holds are unchanged.


A subsequent, separately authorized native synthetic CPU proof at `d5ab0b4`
completed in 153.2 seconds, including ingestion, diagnostic suite evaluation,
collection, publication-report rendering and replay with one retained attempt.
Its envelope included smoke and warmup pilots plus runtime optimizer probes;
measured peak process-tree RSS was 1,558,810,624 bytes and workspace growth was
3,752,253 bytes. The native cleanup receipt recorded no living descendants.
Artifact transfer/verification took 62.2 seconds, consistent with the earlier
interrupted ingestion. This used an explicitly authored native Campaign;
it did not establish portable workflow execution admission or scientific quality.
The [prepared-input bridge](../workflow-recipes.md#prepared-input-cpu-execution)
carries the external declaration context through that existing native path.
