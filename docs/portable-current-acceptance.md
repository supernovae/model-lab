# Current portable declaration acceptance

Checked 2026-10-18 against Lab `7e306a96bcc617468d2a06493dab4562bff6beb0` and
Workshop inputs `0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4`. This record reconciles
Lab #158's current bounded authoring/inspection path, not full issue completion.
The public Workshop update and unchanged-input ownership audit merged in
[Workshop PR35](https://github.com/supernovae/kybern-workshop/pull/35) at
`b74dff53259343aafd336cd00e3480246d70133b`, from reviewed head
`2cc1965555c9a4363de49eda396c63c605f38172`. References to PR35 below describe
that merged state; the original input pin above remains unchanged. That update
is not a prerequisite for these existing native capabilities.

## Environment and effects

Used the already installed Python 3.14.4 interpreter with the pinned Lab `src`
explicitly selected through `PYTHONPATH`, offline Hub flags and one thread per
CPU library. No installation or qualification was performed. This validates the
current source CLI; it is not evidence that a particular installed package/image
matches that source. Copies of the actual declarations lived outside both Git
checkouts, and every command ran from an unrelated directory.

| Inputs | Executed native command | Evidence / limitation |
| --- | --- | --- |
| DevMind model and optimizer sections | `config validate FILE --kind model/optimizer --json` | Valid schemas, executable false; no full RunConfig, generated inputs or runtime claim |
| Microlab run and tokenizer | `config validate FILE --kind run/tokenizer --json` | Valid schemas, executable false; no payloads loaded |
| Microlab existing width delta | `config derive RUN --set model.ffn_dim=384 --output CANDIDATE --json` | Only derived metadata and sidecar written; no scientific choice added |
| Synthetic preparation's campaign/run/tokenizer/monitor | `experiment recipe create ROOT --profile preparation ...` | New inventory copies four actual members unchanged; no descriptor prerequisite |
| Generated synthetic descriptor and current bindings | `experiment recipe preparation inspect ... --bindings LOCAL --json` | Native operation graph and future local paths; executable false |
| Same synthetic recipe moved to another directory | Repeat native inspect, then `recipe export` | Recipe fingerprint unchanged; five authenticated metadata files exported |
| DevMind preparation transport-init | `attempt phase-command --plan PLAN --attempt-root ROOT --label transport-init --bindings-json JSON` | Exact native budget-init argv; no attempt directory |
| DevMind pack with missing generated inputs | Same compiler, `--label pack` | Expected nonzero: missing RESOURCE_ENVELOPE and RUN_CONFIG |
| Actual microlab advanced plan | Matrix inventory creation then `recipe inspect` | Creation succeeds; preview refuses evaluation_suite explicitly |

For the synthetic creation above, `INPUT` is an absolute copy of Workshop's
`examples/synthetic-cpu` at the input pin, and `DEST` is a new absolute directory
whose parent exists. The exact recipe arguments are:

```sh
klab experiment recipe create "$INPUT" --profile preparation \
  --id workshop-synthetic-cpu-preparation --entrypoint preparation-campaign.json \
  --file campaign:preparation-campaign.json --file run_config:run.json \
  --file tokenizer_config:tokenizer-config.json --file metadata:monitor.json \
  --output "$DEST"
```

Synthetic recipe fingerprint:
`bb075a8e08db14b78990a28ae14a76a7489ee868d57a35073d86a922acf7dbe0`.
The current preparation bindings have `artifacts` and exact `config_paths`, not
an obsolete `preparation_bindings_version` field. Workshop's
`examples/synthetic-cpu/README.md` preserves the historical Lab `b155cdcc` procedure,
which requires that field. Current Lab rejects it, as well as
`preparation_admission_version` and `workflow_admission_version`; current workflow
admission also requires explicit `limits`. Workshop PR35 labels that guide as
historical and directs current non-executing use to `docs/native-workflow.md`.
The historical procedure and measured results remain intact. This is a scope
clarification, not runtime qualification or a compatibility adapter.

The CLI argv compiler takes a
literal JSON object for `--bindings-json`; an initial filename invocation failed
parsing and was retained before the corrected invocation passed.

No output payload or attempt directory was created. All nonzero outcomes were
retained. No future input hashes were invented, no unsupported advanced plan was
silently simplified, and no approvals, lock publication, native preparation,
training, evaluation or generation ran.

## Reusable coverage and next boundary

Executed existing neutral regressions:

```sh
python -m pytest -q tests/test_portable_authoring.py tests/test_recipe_authoring.py
```

55 passed in 2.44 seconds. These cover native schema validation, publication
no-clobber/integrity, relocation, authoring refusals and derivative graph guards.
No per-Workshop-experiment Python tests were added. The manual checks above use
actual migrated declarations; they complement generic regressions rather than
turning those declarations into test dependencies.

This exercise reproduced documented profile/input refusals, not a missing generic
executor. The next connection to establish is a reviewed concrete declaration
within a supported profile. DevMind needs actual configuration/input decisions;
its complete advanced execution profile is not implied by valid sections. The
microlab's simple `try` route is distinct from its advanced Campaign/phase plan.
Derivative inheritance, compatibility and budget resolution still require native
verification with actual inputs. Extend only demonstrated reusable gaps with
neutral tests, never with a Workshop launcher or historical compatibility adapter.

Scientific decisions, acquisition, runtime qualification, model work and useful
story evidence remain separately gated. No broader issue is closed by this record.

## Documentation follow-up recheck — 2026-10-18

Using the same source CLI and installed interpreter, creation with the exact
arguments above and inspection with the versionless bindings temporarily shown
in Workshop README head `97618a4e6703785e28ba700343f89b02a4a41acc` both
returned exit 0. Generated and committed synthetic descriptors had the fingerprint
recorded above and `executable: false`. Adding only the obsolete version field
returned exit 1 with the recorded refusal. All four copied members were byte
identical, and inspection created no bound work directory. The same 55 bounded
authoring tests passed again in 2.25 seconds. No preparation or model work ran.

That binding-only check did not validate the README's pinned historical procedure
or later admissions. The subsequent review identified that removing its required
historical field made the guide inconsistent. PR35 restores the historical
procedure; current metadata examples remain in `docs/native-workflow.md`.
