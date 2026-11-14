# Dashboard tour

Commands below use the [activated, provisioned environment](../README.md#install).

`klab dashboard` opens a local, read-only Streamlit app. The **Lab** pages
come first. Each answers one question, opens with the verdict, explains its
numbers in plain words and, when it has nothing to show, names the command that
would fill it. Run telemetry (training curves, runtime, memory, checkpoints) and
the Learn page follow in their own sections.

```sh
klab dashboard               # WORK_DIR/runs + WORK_DIR/lab
klab dashboard --lab-dir DIR --runs-dir DIR
```

The default bind address is `127.0.0.1`. A container viewer may explicitly use
`--address 0.0.0.0` with a localhost-only host port mapping; see
[container usage](containers.md). Starting the dashboard replaces the CLI process
with Streamlit so stop signals and the server's exit status are preserved.

| Page | Question | What you see |
|---|---|---|
| **Home** | What happened, and what should I do next? | latest verdict, next-step cards with commands, counts, recent tries and probes |
| **Experiments** | Did my change help, and is the evidence good enough to act on? | verdict banner, per-probe table and charts, per-probe trends, history, side-by-side compare |
| **Models** | Where does each checkpoint sit on cost vs quality, next to known models? | checkpoint catalog, Pareto frontier, reference models, runs nobody has scored yet |
| **Behaviors** | What does the model actually do, and where does it fail? | greedy generations side by side, fact-recall hits and misses, needle by length, reliability diagram |
| **Explorer** | What is inside this model, and how does it process a sentence? | architecture diagram, per-token loss and top-k, attention maps, weight stats, residual activation RMS, expert routing, memory lookups, paired checkpoint comparison |

Every page reads lab records through the same verified reader as
`klab report` (`klab.lab_records`). A record that fails its seal is
listed as rejected, never shown as a result. Numbers are only put next to each
other inside one comparison group (`eval_group` for held-out loss,
`benchmark_group` for lm-eval, `item_group` for fact recall). A value from
another group is labelled **not comparable** and never silently mixed in.

Colors follow one palette (`klab.dashboard.ui`). Status colors pass WCAG
AA contrast against white text, and they always come with an icon (✔ ▲ ✖ ≠ ⊘),
so color is never the only signal. Charts use colorblind-safe series colors and
a perceptually uniform heatmap scale (Viridis). The layout is wide and its
columns stack on narrow screens.

The screenshots below come from a demo lab: a 2-layer, 32-dimension baseline
trained for 60 steps on synthetic text, plus four one-variable tries (wider
FFN, a third layer, 4 experts with 1 active, an n-gram memory table) at the
standard tier, one full-tier probe and three explorations. The models are tiny
on purpose. The point is how the pages read, not the scores.

## Home

![Home: latest verdict, next steps, recent activity](assets/dashboard-home.png)

The newest probe verdict leads (the same banner as `klab probe`). Next to
it, **Next steps** turns the lab's state into commands, each with a reason:

- the latest verdict, when it names a command (escalate, compare, rerun);
- candidate checkpoints that have held-out loss but no fact-recall or needle
  evidence yet (`klab probe RUN --config standard`);
- no lab checkpoint has lm-eval accuracy yet, so none sits on the reference
  curve (`--config full`);
- plain `klab train` runs that no lab record has scored
  (`klab probe DIR/RUN`);
- runs nobody has looked inside yet (`klab explore RUN`).

At most four steps are shown, in this order.

Recent activity lists tries and probe batteries, newest first, verdict first.
A try that never reached a verdict says so (for example `failed`). An empty lab
shows the two commands that get you started.

## Experiments

![Experiments: verdict, probe table and comparison charts](assets/dashboard-experiments.png)

Pick a result (a try's arms or a `klab probe` record). The page shows:

- the banner (verdict, next action, held-out guard, tiers run);
- four headline metrics;
- the probe table, with a meter centred on the baseline whose full half-bar is
  the fail threshold;
- baseline vs candidate bars;
- needle accuracy by length.

**History & trends** plots any probe across every result, colored by verdict.
A trend only joins results measured the same way: one probe-suite digest and
one comparison group (the eval group for held-out probes, the benchmark group
for lm-eval, the item group for recall and needle). Pick the group in the
selector (default: the newest result's); the other groups are hidden and
counted, and older results that recorded no group are listed as such. It also
lists tries that never ran a probe battery, each with the command to probe it.

![Compare: paired deltas, refused across groups](assets/dashboard-compare.png)

When the verdict suggests a rerun or escalation, the page shows a read-only
follow-up plan built from the sealed request: benchmark selectors, runtime and
resource values, checkpoint digests and baseline provenance. Fields the record
does not hold are listed as missing. The plan is not an executable declaration;
review it and write a new probe config yourself. For a `NOT_COMPARABLE` try the
plan lists the failed comparison checks instead: fix those inputs before a new
`klab try`, since an unchanged rerun cannot fix comparability.

**Compare** runs the same engine as `klab compare`: paired deltas ± SE for
each metric against other results and, optionally, the reference models. Cells
read **better**, **worse** or **within noise** (|Δ| ≤ 2 SE). **Not comparable**
gives the reason (for example, held-out loss against a reference with another
tokenizer). **Not run** means no result carries that measurement, as for the probes
above the fast tier; the cell names the command that measures it.

## Models

![Models: checkpoint catalog](assets/dashboard-models.png)

The **Catalog** has one row per verified checkpoint: resident and active
parameters, training tokens, and the newest held-out loss, fact recall and
lm-eval accuracy, with the records they came from. The caption reminds you
that a column is only comparable inside one group.

![Pareto frontier: lab runs on the reference curve](assets/dashboard-models-pareto.png)

**Pareto frontier** plots quality (held-out loss, lm-eval accuracy or fact
recall) against a cost (resident or active parameters, weight bytes, training
tokens, scoring latency) within one comparison group. Reference models appear
as purple diamonds wherever they share the group (lm-eval and fact recall,
never our held-out loss). Details: [probes](probes.md#dashboard).

**Reference models** lists the pinned public checkpoints and their packaged
scores, and explains what they can and cannot tell you. **Not yet scored**
lists plain `klab train` runs in the run directories that no try or probe
has measured, each with the `klab probe DIR/RUN` command that
places it in the catalog and on the frontier.

## Behaviors

![Behaviors: generations side by side](assets/dashboard-behaviors.png)

![Behaviors: fact recall hits and misses](assets/dashboard-behaviors-recall.png)

- **Generations.** The degeneration probe's greedy continuations, candidate
  next to baseline, with seq-rep-4 and distinct-n. A loop warning appears when
  more than half of the 4-grams repeat.
- **Fact recall.** Every held-out item with what was expected and what each
  arm picked (fractional credit for ties), misses first. In-context recall (the
  fact is stated in the prompt) and closed-book recall (the run's own
  withheld-facts training facts, from its `dataset.synthetic_seed`, asked with
  no context) sit side by side; for a run not trained on `withheld_facts` the
  closed-book block shows why it was not scored instead. The
  closed-book block also shows the never-trained control facts, which should
  stay at chance. The demo above shows what this is for: every lab model picks
  the same candidate for every question of a relation (`red`, `panda`), which
  no aggregate accuracy would tell you. It is the near-uniform prior of an
  untrained model, not candidate order (shuffling the candidates changes
  nothing; see [probes](probes.md#probes)); the page flags it as a constant
  pick.
- **Needle in a haystack.** Retrieval accuracy against context length, one line
  per checkpoint (its newest result) within one item group. Needle lengths are
  fixed character counts, so lab runs and references share a group; results
  with other items or lengths form another group. Pick the group in the
  selector (default: the newest result's; the others are counted as hidden).
  The chance line comes from that group's own results and is omitted when it is
  unknown or they disagree.
- **Calibration.** A reliability diagram (confidence of the top guess against
  how often it was right; marker size = share of tokens), candidate against
  baseline, with ECE.

![Behaviors: reliability diagram](assets/dashboard-behaviors-calibration.png)

## Explorer

The explorer looks inside a small checkpoint: up to
`klab.explorer.MAX_PARAMETERS` (60M) parameters and `MAX_TOKENS` (128) tokens
of sample text. Bigger models show only the architecture diagram, which comes
from the config alone, so nothing is loaded. On the page, pick an immutable
`step_*_gen_*` generation or a pinned try or probe record and enter a sentence.
Generations under `LAB/runs` and the configured run stores are listed with
their step; `latest.json` pointers and reference models are not. The page
always explores on CPU. From the CLI (see [looking inside](models.md#look-inside)):

```sh
klab explore RUN --config explore.yaml
klab explore RUN --json        # the full payload
```

Results are sealed records under `LAB/explorer/<checkpoint sha>/<key>.json`,
keyed by checkpoint digest, sample text, explorer format and the runtime that
produced them (engine, backend, device index, precision). ROCm and CUDA are
distinct even though Torch calls both `cuda`. The checkpoint is loaded and
authorized before every cache lookup, and an edited or mismatched cache file is
recomputed. Output reports the producing runtime and whether the cache was
reused.

Exploration uses the same runtime preparation as `klab probe`: CPU by default,
or `--runtime`/`--runtime-profile` authorized by the run's runtime policy.
`--resource-envelope FILE` is checked before the command and before every
stage. An out-of-memory error becomes a clear message, memory is released and
nothing is cached. Each invocation prints its own
`LAB/explorer/operations/<invocation-id>/CANCEL` path to stderr; touch it to
stop that invocation at the next stage. MLX checkpoints and attached semantic
packs are refused with a reason.

![Explorer: architecture and parameters (MoE run)](assets/dashboard-explorer.png)

- **Architecture.** Embedding → each layer (attention heads, dense FFN or
  experts with the active ones highlighted) → memory table → output, with
  parameter counts per component taken from the shape-only tensor inventory.
- **Tokens.** The sample text with each token shaded by its loss. Hover a token
  for its loss, rank and the model's next guesses. Below it: a per-token loss
  chart and the top-k guesses after any position, from the shared scoring path
  the probes use.

![Explorer: per-token loss and next-token guesses](assets/dashboard-explorer-tokens.png)

![Explorer: attention map of one head](assets/dashboard-explorer-attention.png)

- **Attention.** One heatmap per layer and head, computed by the attention
  module's own softmax (`DenseAttention.attention_probabilities`, which the
  forward pass also uses). Four plain numbers describe the head: how far back
  it looks, its share on the previous token, its share on the first token
  (attention sink) and its spread (entropy). Latent and block-sparse layers
  select or compress keys and expose no per-key weights; the page says so
  rather than showing something made up.
- **Activations.** Token-aligned residual-stream RMS, calculated in
  FP32 before reduction: `residual_pre` is the input to the attention normalization,
  `residual_mid` follows the attention residual, and `residual_post` follows the
  FFN residual (the next block's normalization input, or final normalization input).
  These are descriptive magnitudes, not attribution or a quality verdict. Only
  per-token scalars are retained, not full residual activation tensors.
- **Weights.** RMS per stored tensor (hover for norm, max |w| and shape) and a
  histogram of any tensor (sampled above 200k values, which the title says).

![Explorer: expert routing](assets/dashboard-explorer-routing.png)

- **Expert routing** (MoE runs). A token × expert heatmap of gate weights per
  layer, router entropy and each expert's share of tokens, so you can see
  collapse at a glance.

![Explorer: memory lookups](assets/dashboard-explorer-memory.png)

- **Memory lookups** (memory runs). The rows each token read in each table
  (↺ marks a row an earlier position already read: the same n-gram or a hash
  collision), each table's reuse rate and the per-token memory gate. When every
  address of an n-gram table is just one earlier token's id modulo the table
  size, a warning says the table is acting as a single-token table.

### Compare checkpoints

Choose two pinned selectors and a shared sentence, then press **Compare**. Both
explorations run on CPU. The two records must be sealed and verified, name
distinct checkpoints, and share the tokenizer digest, exact text, token IDs
(including the initial EOS) and runtime. Truncated inputs and missing activation
diagnostics are refused.

The paired RMS heatmaps share a scale; signed deltas are **right minus left**,
centered on zero. Different architectures show their layer overlap and individual
values but hide deltas (`architecture_matches=False`). Expert routing is shown
separately for each checkpoint; dense FFNs have no MoE router. Changing a selector
or sentence hides the previous result until you run again. A failed new comparison
clears the prior comparison, so old charts cannot appear under a new selection.

## Run telemetry

Run telemetry includes **Engram memory** (separate from device and host memory)
and **Snapshot inspector**. Training opens with a compact memory summary and three
curves: training loss, validation loss and throughput, when recorded. Use the
metric selector to add other curves. The quick navigation links retain the run
selection across telemetry pages.

Engram memory shows aggregate and recorded per-port gates, value/hidden norms,
address reuse and table utilization. Every value retains its observation step;
missing diagnostics say unavailable. These are target-weighted forward diagnostics,
so counts can be fractional. Address reuse includes repeated inputs and is not
proof of hash collisions or retrieval quality. The page does not infer gradients,
parameter updates or quality gains.

Snapshot inspector filters current recorded metric values and their changes from
the preceding observation, displays the saved configuration, shows recent events,
and inspects an explicitly selected checkpoint generation (newest by default).
It reads projected metadata only; no weights are loaded and no inference runs.
Save-time verification is distinct from a current local file check. Existing
**Checkpoints** verification and **Explorer** remain available through the quick
links.

Automatic refresh defaults to ten seconds and offers two seconds, thirty seconds,
or paused. **Refresh now** refreshes the telemetry fragment without a full-page
rerun. Only the selected runs are read. A failed read keeps that selection's
last successful snapshot with a visible stale-data warning; it never shows
another selection's data.

### Browse separate run stores

`--runs-dir` is the primary store; add named directories alongside it:

```sh
klab dashboard --runs-dir /data/klab/runs \
  --run-store 'Pilot=/data/pilot/runs' \
  --lab-dir /data/klab/lab
```

`--run-store LABEL=DIRECTORY` is repeatable. The **Run store** sidebar selector
appears only when extra stores are configured. It switches the entire telemetry
view without merging or copying databases. Labels and directories must be
distinct. A missing store is shown as unavailable, never created or replaced by
another store. Run selections are remembered per directory;
checkpoint and metric selectors reset on a store change, and stale snapshots are
also scoped to that directory. Identical run IDs in different stores stay separate.

`--lab-dir` selects the Lab records for Home, Experiments, Models and
Behaviors. Explorer sees the selected run store plus `LAB/runs`. Changing the
run store does not change the Lab records directory.

### Focus and evidence

The sidebar shows model names, exact run IDs, recorded status and update timestamps.
**Focus recorded running runs** selects running records in the chosen store; stored
status does not verify process liveness. **Open this run** links to Training with the
configured store path and run ID. Missing linked stores stop rendering; missing or
hidden runs select nothing instead of silently substituting another model. Store
links only select configured/discovered stores, never arbitrary filesystem paths.

Training opens with observed training and validation loss, change since the previous
observation, and their evidence limits. Capabilities require verified checkpoint
probes; memory activity alone does not demonstrate an improvement. Explorer's memory
lookup plot maps token positions to recorded table/row addresses. It does not label
hashed rows as concepts or claim neuron connections. It reads existing exploration
payloads and adds no model execution.

## Generate

**Generate** uses native dashboard controls, with prompt and continuation side by
side. It lists complete eligible local snapshots across the selected run store,
newest UTC save time first, independently of telemetry run filters. The **LATEST**
marker identifies the newest published verified checkpoint, including one from
an unfinished training run. It does not mean the latest completed training run.

An unpinned session defaults to that newest save. **Refresh snapshots** reads the
catalog without loading weights. Explicit choices and deep links stay pinned;
**Use latest** clears the pin and refreshes the list. Stable links accept `store`,
`run`, and `checkpoint`. Missing links fail without substitution. A new save never
changes an in-flight request. Missing and incomplete file inventories are excluded; every load is verified again.

### Checkpoint counts

The Overview and Checkpoints summaries distinguish the store-wide available count,
the selected runs' available count, and the newest saved step. The retained-save
table and latest recorded retention event show which files remain and why a new
checkpoint may replace an older one without increasing the count. Save sequence
numbers (Gen 2 or checkpoint_id 2) identify one run's second save, not the total
number of checkpoints. Lab Home's **Scored checkpoints** counts sealed lab scores,
including references, and is separate from training snapshots.

### CPU sampler

The dashboard sends bounded JSON requests to a separate loopback CPU sampler.
`KLAB_SNAPSHOT_SERVICE_CONFIG` points to a JSON file containing the allowed
`store` and `url`, for example:

```json
{"store": "/data/experiment/runs", "url": "http://127.0.0.1:8502"}
```

Set that environment variable for both processes, and run
`python src/klab/dashboard/snapshot_sampler.py` in the provisioned environment
(not through Streamlit).

Keep the sampler's 10 GiB memory cap, two CPU threads and 200% CPU quota,
separate from the dashboard's resource envelope. Checkpoints exceeding 5 GiB of
inventoried files are refused. The service only binds the configured 127.0.0.1
port, rejects browser origins and unexpected Host values, and does not log prompts
or outputs. The dashboard UI never loads a model.

**Generate completion** verifies and loads the exact selected digest through the
native loader. Cold verification checks finalized contents, manifest, model
configuration, tokenizer and Engram state. Recorded save-time verification alone
never replaces it. Partial or staging directories are never loaded.

One shared lock spans verification, eviction, loading and generation. Busy requests
are refused rather than queued; the old model is released before another loads.
Failed loading leaves no fallback model. The UI distinguishes the selected snapshot,
sampler status at observation time and output identity. Cold verification can take
minutes; generation is token-bounded, not forcibly cancelled at a wall-clock limit.

### Prompts and sampling

Prompts start blank; example prompts are added only when you pick one. Sampling defaults are temperature **0.7**,
top-k **40**, seed **42** and **32** new tokens. Temperature 0–2, top-k 0–200
(0 disables), a seed and 8–512 new tokens are supported. The combined prompt/output
limit is 1,024 tokens and is checked before generation. Longer requests take longer;
EOS may end a completion before its cap.

Temperature zero is greedy. A fixed seed helps repeat a sample; changing it explores
variation, not quality. Exact output can vary across runtimes. **Reset sampling
defaults** restores the controls, keeps the prompt and snapshot, and clears the
prior result without inference. Results retain the exact input, controls and
verified identity, and the continuation can be downloaded.

Prompts and outputs are never added to training data. Generate refreshes
manually so it does not interrupt a request.

## For developers

- `klab.dashboard.lab_data`: everything the lab pages show, with no
  Streamlit import (tested in `tests/test_dashboard_lab.py`).
- `klab.dashboard.lab_pages`: the five pages. `probes.py` keeps the
  shared probe components (banner, table, charts, Pareto). `explorer_view.py`
  turns an exploration into figures and HTML. `ui.py` holds the palette, chips,
  page headers and empty states.
- `klab.explorer`: the explorer data layer. It reads architecture from
  the config, and weights and activations from the verified loader. Attention
  comes from the module's own softmax, memory reads from forward hooks on the
  tables (the hash is never re-implemented), and routing from
  `diagnostics="full"`.
- Page renders are tested with `streamlit.testing.v1.AppTest` (empty and
  populated labs, the real `app.py` entry point). CI's `lab-loop` job runs
  them together with the probe and explorer tests.
