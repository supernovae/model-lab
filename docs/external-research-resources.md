# External research authoring resources

`research scaffold` composes the existing native matrix and study declarations;
it does not acquire data, train a tokenizer, run a model, or authorize execution.
Use a version-controlled external resource directory when authoring a study
outside the Lab repository:

```console
klab research scaffold neutral-study --resource-root /path/to/resources --scale smoke --data neutral --output /path/to/new-study
```

The selected directory uses the existing research schemas and layout:

- `catalog/<id>.json`: `klab-research-entry` version 1.
- The entry's relative `recipe` path: `klab-research-recipe` version 1.
- `base.yaml`: native run configuration, overlaid by the selected scale and recipe.
- `profiles.json`: native research profiles for smoke, nano, micro, and tiny.
- `datasets.json`: named native dataset profiles. Names need not match Lab's bundled catalog.
- Any dataset profile's `source_declaration`: native pinned dataset source JSON.
- Any path-based capability cards referenced by the entry, relative to this directory.
- `lifecycle.json`: native lifecycle registry used for CLI diagnostics. An explicit
  `--lifecycle PATH` selects another registry; `--evidence-root PATH` retains its
  existing meaning for lifecycle evidence references.

A resource root selects one closure. Missing files fail instead of falling back
to packaged resources. Resource paths must remain inside the root and cannot
traverse symlinks, including ancestors of the root. Supply a fully resolved root
path (for example, `/private/tmp/resources` rather than a symlinked `/tmp/resources`
on macOS). This restriction applies to the supplied resource root, not the output
directory. An explicit local JSON entry is also supported; with a resource
root it must be inside that root, and its recipe remains relative to the entry's
parent as in the existing local-entry contract. IDs resolve under `catalog/`,
with recipe paths relative to the resource root.

Every current recipe must declare a `seed` axis using the existing matrix option
format, with every option explicitly setting the native `seed` field. Native run
configuration validation still checks seed values. No implicit matched seed selection is injected, with or without a resource root.
Scale profiles retain their current four-scale schema and native overlay behavior,
including scale-dependent optimizer defaults; `base_set` can explicitly override
those defaults. Backend selection remains independent. This interface adds no
new scheduler, training engine, or arbitrary scale semantics.

Generated studies include the selected original base, entry, recipe, profiles,
and datasets under `research_sources/`. Selected source declarations and local
capability cards are copied, and their SHA256 values appear in the native
`research.json` input inventory. The native matrix expander and study planner
validate the generated configuration identities. The destination must be new;
a failed authoring attempt does not replace an existing directory.

`learn scaffold ID --resource-root ROOT ...` selects lesson declarations under
`lessons/` and the same authoring resources. Its existing mechanism behavior
remains: artifact walkthroughs and the optional semantic-retrieval `demo.py`
are teaching examples, not portable experiment execution declarations. Model
lessons preserve their source lesson/base/profiles/datasets in their input
inventory. This option does not convert a lesson into a new generic protocol.

The personal packaged recipes and their catalog entries have been retired together.
Use an explicit current resource closure; unknown retired IDs refuse. Packaged
mechanism lessons and auxiliary base/profile/dataset resources remain during their
separate migration. Unrelated command help does not require the dataset catalog.

The default packaged lifecycle registry is empty: no findings, baselines,
promotions, or prior-evidence warnings are selected. This is a new current default,
not a reset of an experiment registry or runner state. Empty-registry validity is
structural validation, never scientific clearance. Explicit `--lifecycle PATH`
selection remains strict; missing or invalid files refuse. An external resource
root still requires its own `lifecycle.json` unless overridden. `--evidence-root`
defaults to the working directory and controls relative evidence references.

Two personal no-recipe catalog cards remain pending separate consumer/protocol
retirement: `dense-lm-v1` and `learned-engram-portability-v1`. They appear
unassessed in the default research view; they are not generic examples or current
guarantees. The eleven `learn`
lessons are independent mechanism tutorials (nine model lessons and two artifact
walkthroughs); they do not require the personal lifecycle registry.

The personal TinyStories dense30M data-rich config and no-recipe catalog card
are retired from the packaged defaults. Their [original config][tinystories-config]
and [catalog card][tinystories-card] remain immutable Workshop evidence, not a
current runnable recipe. Historical protocol and results remain unchanged.

The [retirement inventory](history/research-resource-retirement.json) records original
hashes and landed Workshop destinations. The archived populated registry is
historical evidence only. The [preservation supplement][supplement] records five
references absent from the pinned tracked trees; they were not recovered.
Reproduction through current interfaces belongs in the Workshop backlog as new
experiments with new identities. Preserved recipes alone are not runnable ports.

[supplement]: https://github.com/supernovae/kybern-workshop/blob/1aa84218a95fabafc96f2cc7d667c0c969eb8d5e/provenance/lab-lifecycle-supplement.json

## Migration boundaries

Historical files belong in a provenance archive with their original bytes and
hashes. Adapted runnable native declarations belong alongside that archive and
must identify actual missing prerequisites. A website catalog can point to the
native entry, runnable recipe, provenance map, and evidence; it need not duplicate
the native scientific schema or become another execution framework.

This change does not externalize specialized learned-portability or Pythia
protocol selection, historical lint path exemptions, benchmark defaults, or
runtime assets. Those require separate coordinated migrations. It does not make
an unbound model recipe runnable or supply corpus/tokenizer/checkpoint evidence.

Legacy allocation cards such as `../../artifacts/allocation/cards/*.json` are not
contained authoring resources and refuse this copy path. Their native adaptation
must use safe relative card paths within the selected root. Likewise, local JSON
forks using path-based cards should supply `--resource-root`; the old no-root
interface retains packaged auxiliary-resource selection. Generated filenames and
card paths share a namespace; collisions fail without publishing the destination.
These limitations do not justify changing archived card paths or hashes.

[tinystories-config]: https://github.com/supernovae/kybern-workshop/blob/1aa84218a95fabafc96f2cc7d667c0c969eb8d5e/provenance/lab/configs/tinystories_dense_30m_data_rich_v1.yaml
[tinystories-card]: https://github.com/supernovae/kybern-workshop/blob/1aa84218a95fabafc96f2cc7d667c0c969eb8d5e/provenance/lab/src/klab/research/resources/catalog/tinystories-dense-30m-data-rich-v1.json
