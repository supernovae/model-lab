# Path-domain allocation: neutral coverage and remaining ownership boundary

Assessed at Lab `0af1683188edacecf0856baa3520354363358f0d` and Workshop
`fd259e2cad4e9856d6d659e8df34728a98b32502`. This change replaces one test's
personal corpus/tokenizer-config dependency. No production behavior, scientific
constant, personal asset or archived byte changes.

## Classification

| Responsibility | Current location | Ownership and next action |
| --- | --- | --- |
| Canonical manifests, input hashes, nonsymlink validation, atomic no-clobber publication | `data/allocation.py`, packing/pack helpers, `allocation_tasks.py` | Reusable Lab verification and publication behavior; retain in Lab. |
| Token-aligned owner/query sidecars, causal target-to-input shift, supervised/raw accounting | Allocation and packing helpers | Reusable Lab mechanics; neutral tests retain accounting and mismatch coverage. |
| Resource-regime and owner-code vocabulary | `data/allocation.py` | Existing public engine contract; changing enums/accepted profile names needs caller review, not asset deletion. |
| Path operations, exact semantic-key fields, audit/provenance formats, 24/8/12 splits and record-ID templates | `data/allocation_tasks.py` | Path-domain task policy mixed into the builder, not a generic arbitrary-corpus contract. Keep unchanged pending explicit declaration design. |
| Five neural fractions and resulting 25-profile sweep, seeded assignment schedule, caps | `data/allocation_tasks.py` | Allocation experiment choices: fractions, seed base 20260923, 24/8 document caps and 16384/4096 token caps remain fixed. Separate declared choices from generic quota/shuffle mechanics before generalizing. |
| Dataset/namespace/version/license, three card filenames, pack name and creation timestamp | `data/allocation_tasks.py` | Personal source metadata/policy. It cannot truthfully describe arbitrary external corpora merely because paths are explicit. |
| Original corpus, audit, cards, configs, ledger and result | Eighteen `path-domain` rows in [the prior audit](context-retirement-audit.json) | Personal experiment definitions/evidence belong in Workshop; original bytes are preserved there, but active retirement is deferred. |

The frozen structured-key/value encoders are synthetic retrieval machinery, not
natural-language semantic generalization. They remain unchanged. The native
allocation CLI already requires external paths; that removes implicit discovery,
not its path-domain-specific semantics.

## Neutral replacement and retained guarantees

`tests/allocation_fixtures.py` generates new, short path literals and answers in
scratch space. It reads no personal corpus, audit, card, provenance or tokenizer
configuration. The fixed 256-byte alphabet plus four special tokens requires no
tokenizer training or external artifact. Existing schema/card names and 24/8/12
records are the minimum current builder contract, not copied scientific examples.
Those historical labels do not turn these temporary outputs into personal-study
evidence. Only train/development records are packed; the third split supplies
synthetic audit/card wiring.

The replacement test preserves all earlier checks: 25 profiles, owner totals,
raw versus supervised counts, query-bound cards, and prepared-query dimensions.
It additionally checks all four corpus/audit/provenance file hashes, rounded neural
quotas, balanced residual owners, semantic/hybrid query counts, and the causal
one-token shift of query masks. Literal card scoring preserves case and Unicode
composition while trimming outer whitespace; original generated answers survive
query binding unchanged. The existing analysis tests retain raw/nonmonotonic
observations, distinct resource regimes and explicitly estimated FLOP accounting.

Negative controls change audited answers, structured paths, provenance split
paths and card answers. Each must refuse with its specific error and leave no
published bundle or staging directory. Existing focused routing tests retain
corpus-digest mismatch, audited-prompt/answer refusal, and owner/query/mask tamper
checks. No model training or evaluation is needed for these guarantees.

## Specific blocker to retirement

Do not retire the eighteen active assets in this slice. The builder still declares
personal policy inside Python, including source license/version/namespace and
fixed scientific budgets/selection. Replacing test data alone does not externalize
that policy or establish a truthful generic corpus contract.

A subsequent scoped design must define which task metadata, audit/card mappings,
split sizes, budgets and assignment choices are explicit validated declarations,
which generic operations stay in Lab, and how current CLI callers consume them.
It also needs native acceptance of the unchanged archived personal declarations
and a reviewed consumer/guarantee map before active files are removed. Do not move
personal Python builders/validators into Workshop or add a second execution path.
No such schema/API expansion is implemented here.

Preservation check: all eighteen original files match the pinned Workshop
archive byte-for-byte, with prior-audit size, SHA256 and destination Git blobs.
All active originals and original audit JSON remain unchanged. This is a
preservation check, not retirement or runtime/scientific qualification. Lab
#157/#158 remain partial; historical results and missing payload status are not
rewritten.
