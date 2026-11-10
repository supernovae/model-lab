# Dashboard tour

Commands below use the [activated, provisioned environment](../README.md#first-run).

`sparselab dashboard` opens a local, read-only Streamlit app. The **Lab** pages
come first. Each one answers a single researcher question, opens with the
verdict, explains its numbers in plain words and, when it has nothing to show,
names the command that would fill it. Run telemetry (training curves, runtime,
memory, checkpoints) and the Learn/Research pages follow in their own sections.

```sh
sparselab dashboard               # WORK_DIR/runs + WORK_DIR/lab
sparselab dashboard --lab-dir DIR --runs-dir DIR
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
`sparselab report` (`sparselab.lab_records`). A record that fails its seal is
listed as rejected, never shown as a result. Numbers are only put next to each
other inside one comparison group (`eval_group` for held-out loss,
`benchmark_group` for lm-eval, `item_group` for fact recall). A value from
another group is labelled **not comparable** and never silently mixed in.

Colors follow one palette (`sparselab.dashboard.ui`). Status colors pass WCAG
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

The newest probe verdict leads (the same banner as `sparselab probe`). Next to
it, **Next steps** turns the lab's state into commands, each with a reason:

- the latest verdict, when it names a command (escalate, compare, rerun);
- candidate checkpoints that have held-out loss but no fact-recall or needle
  evidence yet (`sparselab probe RUN --config standard`);
- no lab checkpoint has lm-eval accuracy yet, so none sits on the reference
  curve (`--config full`);
- plain `sparselab train` runs that no lab record has scored
  (`sparselab probe DIR/RUN`);
- runs nobody has looked inside yet (`sparselab explore RUN`).

At most four steps are shown, in this order.

Recent activity lists tries and probe batteries, newest first, verdict first.
A try that never reached a verdict says so (for example `failed`). An empty lab
shows the two commands that get you started.

## Experiments

![Experiments: verdict, probe table and comparison charts](assets/dashboard-experiments.png)

Pick a result (a try's arms or a `sparselab probe` record). The page shows:

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

**Compare** runs the same engine as `sparselab compare`: paired deltas ± SE for
each metric against other results and, optionally, the reference models. Cells
read **better**, **worse** or **within noise** (|Δ| ≤ 2 SE). **Not comparable**
gives the reason (for example, held-out loss against a reference with another
tokenizer). **Missing evidence** names the command that measures it.

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
never our held-out loss). Details: [probe battery](probe-battery.md#dashboard).

**Reference models** lists the pinned public checkpoints and their packaged
scores, and explains what they can and cannot tell you. **Not yet scored**
lists plain `sparselab train` runs in the run directories that no try or probe
has measured, each with the `sparselab probe DIR/RUN` command that
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
  nothing; see the constant-picks note in [probe-battery.md](probe-battery.md)); the page flags it
  as a constant pick.
- **Needle in a haystack.** Retrieval accuracy against context length, one line
  per checkpoint (its newest result) within one item group. Needle lengths are
  fixed character counts from suite v4 on, so lab runs and re-scored
  references share a group; older results (sized in each model's own tokens)
  and results with other items or lengths are another group: pick it in the selector (default: the newest
  result's group; the others are counted as hidden). The chance line comes from
  that group's own results and is omitted when it is unknown or they disagree.
- **Calibration.** A reliability diagram (confidence of the top guess against
  how often it was right; marker size = share of tokens), candidate against
  baseline, with ECE.

![Behaviors: reliability diagram](assets/dashboard-behaviors-calibration.png)

## Explorer

The explorer looks inside a small checkpoint: up to
`sparselab.explorer.MAX_PARAMETERS` (60M) parameters and `MAX_TOKENS` (128)
tokens of sample text. Bigger models get a clear message and only the
architecture diagram, which comes from the config alone, so nothing is loaded.
Explore from the page by selecting an immutable `step_*_gen_*` checkpoint
generation or a pinned try/probe record and entering a sentence. Generations under
`LAB/runs` and configured run roots appear with their step; mutable `latest.json`
pointers and reference-model records are excluded. Duplicate selector paths are
listed once. A record and its generation can still name the same checkpoint;
comparison refuses their identical checkpoint digest. You can also use the CLI:

For a custom sentence, save `explore.yaml` (see [explorer declarations](explore-config.md)):

```yaml
explore_config_version: 1
text: Once upon a time
```

```sh
sparselab explore RUN --config explore.yaml
sparselab explore RUN --json        # the full payload
```

Results are cached as sealed records under
`LAB/explorer/<checkpoint sha>/<key>.json`, keyed by checkpoint digest, sample
text, explorer format and the native loader's actual runtime: engine, backend,
device index and precision. ROCm is distinct from CUDA even when Torch calls
both devices `cuda`. Requested backends (including `auto`) are not identities.
The full producer runtime (including framework version, OS and device name) is
recorded in sealed diagnostics; only the four canonical fields bind reuse.
This is runtime provenance, not hardware qualification or a science identity.
Loading and authorization still happen before every cache lookup.
An edited cache file is ignored and recomputed. Missing or malformed runtime
provenance and sealed payload/key mismatches cannot yield a verified hit.
Text/JSON output reports the producer runtime and invocation `cache_reuse`
status/reason (a sealed overlay, not persisted into diagnostic cache bytes).
New v2 cache records include activation diagnostics. Historical v1 records
remain readable in place, labeled "historical: re-explore for activations", and
are excluded from paired comparison. New format-bound paths do not overwrite
legacy cache bytes. The dashboard can display sealed legacy diagnostics, explicitly
labeling missing/malformed provenance or key mismatches as unverified.
Exploration goes through the same runtime preparation as `sparselab probe` and
`sparselab try`: the run is resolved, the CPU default or
`--runtime`/`--runtime-profile` is applied and authorized by the run's runtime
policy (an accelerator needs a runtime profile, exactly as for probe), and the
checkpoint loads through the verified loader. `--resource-envelope FILE` is
checked before the command and again before every explorer stage; a violation
stops it with the reason. The dashboard page always explores on CPU. Loading the
checkpoint is a stage like the others: the cancel sentinel and envelope are
checked before it, an out-of-memory error while loading or exploring becomes a
clear message, memory is released either way and nothing is cached. It stops
at the next stage if its invocation-specific cancel file appears. Before loading,
the explorer prints `LAB/explorer/operations/<invocation-id>/CANCEL` to stderr
(also with `--json`, whose stdout remains JSON). Touch that exact path to stop
only that invocation. Each call gets a fresh operation directory, including cache
hits; old operation directories and cancel files are retained. The old shared
`LAB/explorer/CANCEL` is left untouched and no longer controls new invocations.
Cache keys bind the explorer format as well as checkpoint, text and runtime. MLX checkpoints and
attached semantic packs are refused with a reason.

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
- **Activations** (v2 records). Token-aligned residual-stream RMS, calculated in
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
  size, a warning says the table is acting as a single-token table. In the demo
  this caught a real problem: with `memory_table_size: 257` the legacy n-gram
  hash multiplier (257) was a multiple of the row count, so the order-2 and
  order-3 tables read the previous token and the one before it, not n-grams.
  The hash now switches to a mixed scheme at such sizes (`sparselab.address_hash`;
  see [re-baselining guidance](first-model.md#re-baseline-the-hash-affected-configs));
  the screenshot predates the fix.

## For developers

- `sparselab.dashboard.lab_data`: everything the lab pages show, with no
  Streamlit import (tested in `tests/test_dashboard_lab.py`).
- `sparselab.dashboard.lab_pages`: the five pages. `probes.py` keeps the
  shared probe components (banner, table, charts, Pareto). `explorer_view.py`
  turns an exploration into figures and HTML. `ui.py` holds the palette, chips,
  page headers and empty states.
- `sparselab.explorer`: the explorer data layer. It reads architecture from
  the config, and weights and activations from the verified loader. Attention
  comes from the module's own softmax, memory reads from forward hooks on the
  tables (the hash is never re-implemented), and routing from
  `diagnostics="full"`.
- Page renders are tested with `streamlit.testing.v1.AppTest` (empty and
  populated labs, the real `app.py` entry point). CI's `lab-loop` job runs
  them together with the probe and explorer tests.

Rerun and escalation hints show a read-only follow-up plan from the sealed
request, including custom benchmark selectors, runtime/resource values,
checkpoint digests and baseline provenance. Rerun changes no policy; escalation
names its intended tier change. Historical missing fields remain explicit.
These plans are not executable declarations and do not promise reproduction.
Authoring a new ProbeConfig and copying saved runtime/resource descriptors into
native files is a separate explicit action. Verify checkpoint, suite and
benchmark identities and review relocated paths before executing; source files
and current defaults may have changed. Opening the dashboard never materializes
these files or launches work.

For a `NOT_COMPARABLE` try, the plan instead carries its failed comparison
checks and sealed try inputs. Diagnose and explicitly repair those inputs
before a new `sparselab try`; an unchanged rerun cannot fix comparability.
Probe-only settings absent from TryConfig are labelled separately from
historical missing fields. Recorded verdict tier advice is labelled as such;
a bare named tier does not retain the sealed effective policy.

### Compare checkpoints

Choose two pinned selectors and a shared sentence, then press **Compare**. Both
explorations run on CPU. Paired v2 records must be sealed and verified, identify
distinct checkpoints, and have the same tokenizer digest, exact text, token IDs
(including initial EOS), and canonical runtime (engine, backend, device index,
precision). Truncated inputs and missing or malformed activation diagnostics are
refused. Historical v1 records must be re-explored before comparison.

The paired RMS heatmaps share a scale; signed deltas are **right minus left**,
centered on zero. Different architectures show their layer overlap and individual
values but hide deltas (`architecture_matches=False`). Expert routing is shown
separately for each checkpoint; dense FFNs have no MoE router. Changing a selector
or sentence hides the previous result until you run again. A failed new comparison
clears the prior comparison, so old charts cannot appear under a new selection.

### Training, memory and snapshot inspection

Run telemetry includes **Engram memory** (separate from device/RAM memory) and
**Snapshot inspector**. Training opens with a compact memory summary and three
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
links. Explorer's native size/runtime restrictions still apply.

Automatic refresh defaults to ten seconds and offers two seconds, thirty seconds,
or paused. **Refresh now** refreshes the telemetry fragment without a full-page
rerun. The query reads telemetry for selected runs only. A failed read retains
that selection's last successful snapshot with a visible stale-data warning;
it never substitutes another selection's data. Charts retain their existing
interaction-preserving component.

### Browse separate run stores

Keep the existing `--runs-dir` as the primary store and add named directories:

```sh
sparselab dashboard --runs-dir /data/history/runs \
  --run-store 'Devmind pilot=/data/devmind/pilot/runs' \
  --lab-dir /data/history/lab
```

`--run-store LABEL=DIRECTORY` is repeatable. The **Run store** sidebar selector
appears only when extra stores are configured. It switches the entire telemetry
view without merging, copying or migrating databases. Labels and directories
must be distinct. Missing stores remain unavailable rather than being created or
falling back to another store. Run selections are remembered per directory;
checkpoint and metric selectors reset on a store change, and stale snapshots are
also scoped to that directory. Identical run IDs in different stores stay separate.

The existing `--lab-dir` still selects the Lab records for Home, Experiments,
Models and Behaviors. Explorer sees the selected run store plus `LAB/runs`, as
before. Changing the run store does not silently change the Lab records directory.
With no extra stores, existing single-store commands behave as before.


### Focus and evidence

The sidebar shows model names, exact run IDs, recorded status and update timestamps.
“Focus recorded running runs” selects running records in the chosen store; stored
status does not verify process liveness. “Open this run” links to Training with the
configured store path and run ID. Missing linked stores stop rendering; missing or
hidden runs select nothing instead of silently substituting another model. Store
links only select configured/discovered stores, never arbitrary filesystem paths.

Training opens with observed training and validation loss, change since the previous
observation, and their evidence limits. Capabilities require verified checkpoint
probes; memory activity alone does not demonstrate an improvement. Explorer's memory
lookup plot maps token positions to recorded table/row addresses. It does not label
hashed rows as concepts or claim neuron connections. It reads existing exploration
payloads and adds no model execution.

### Generate from the selected saved snapshot

**Generate** shares the telemetry store/run selection and offers checkpoint metadata
without opening weights. Stable links accept `store`, `run`, and `checkpoint`.
The loopback CPU sampler offers **Try latest complete saved checkpoint**: the newest
published verified checkpoint, including a checkpoint from an unfinished training
run. This is distinct from the latest completed training run. Select a recorded
checkpoint and press **Generate completion** to verify and load it. Browsing does
not load weights. Links pin exact store/run/checkpoint; missing or unverified links
fail without substitution. The sampler freezes its catalog for the browser session;
**Refresh available snapshots** explicitly discovers newer saved generations.
The list is ordered by UTC save time descending across runs, with a **NEWEST**
marker and date, step, generation and run labels. An unpinned session defaults to
the newest eligible saved checkpoint; an explicit choice or deep link stays pinned.
**Use newest available** clears that pin and refreshes the list. These actions do
not load weights. Missing files, incomplete inventories and unsupported archived
generations are excluded; native cold verification remains mandatory on Generate.

`SPARSELAB_SNAPSHOT_SERVICE_CONFIG` identifies the allowed `store` and loopback `url`.
The existing `run`, `checkpoint`, and `checkpoint_sha256` fields are retained as
legacy configuration metadata, not restrictions or automatic loading instructions.
Run `dashboard/snapshot_sampler.py` with the compatible provisioned runtime/source.
Native cold verification checks finalized generation contents, manifest, model
configuration, tokenizer and Engram state before loading. The catalog's recorded
verification alone never replaces this check. No partial/staging directory loads.

One shared lock spans verification, eviction, loading, and generation. Requests
while busy are rejected rather than queued; the old model is released before a
new model loads. Failed loading leaves no fallback model. Outputs retain the exact
identity and controls used. The UI distinguishes selected snapshot, sampler status
at observation time, and output identity. Checkpoints exceeding 5 GiB inventoried
size are refused; retain the service's 10 GiB memory cap, two CPU threads and 200%
CPU quota. Cold verification can take minutes; generation is limited by tokens,
not a hard wall-clock cancellation that could leave a worker running.

Historical mappings are consulted before generation reads. Mapped archived
checkpoints currently report their archive location as unavailable: the native
loader requires a generation inside its original run and does not support the
relocated archive layout. No copy, symlink workaround, or stale-source fallback is
used. Other stores and incompatible model schemas fail explicitly.

Prompts default to blank. Technical, code and story examples require visible choice;
no TinyStories text is silently appended. Temperature 0–2, top-k 0–200 (0 disables),
seed, and 8–512 new tokens are supported. Combined prompt/output is limited to 1024
tokens for this CPU preview. Temperature zero is greedy. Requests retain the exact
input, settings and verified identity with their output. No prompt/response is added
to training data. Generate has manual refresh to avoid interrupting an embedded
request; Explorer remains a distinct ≤60M internal-diagnostics tool and uses blank
sample text by default. No probe-suite prompts or scientific settings are changed.

Sampling defaults are temperature **0.7**, top-k **40**, seed **42**, and **32**
new tokens. Reset sampling defaults restores those controls, keeps the prompt and
snapshot selection, and clears the prior result without inference. Seed controls
repeatability, not quality; vary it to explore alternative completions. Exact output
can vary across runtimes. Temperature zero remains available for greedy decoding.
The sampler service uses Streamlit's light base and the dashboard's existing
`#0969da` primary color (STREAMLIT_THEME_BASE / STREAMLIT_THEME_PRIMARY_COLOR),
so embedded and standalone views use the same native widgets and status palettes.

Output length now supports 8–512 new tokens, in steps of eight; default stays 32.
The combined 1,024-token CPU context limit still applies and is checked before
generation. Longer requests take longer; EOS may end a completion before its cap.


### Native snapshot completion

The Generate page uses native dashboard controls, with prompt and continuation side
by side. It lists complete eligible local snapshots across the selected store, newest
UTC save time first, independently of telemetry run filters. Refresh snapshots reads
the catalog without loading weights. Explicit choices and deep links stay pinned;
Use latest selects the newest available save. A new save never changes an in-flight
request. Missing links are refused rather than substituted.

The Overview and Checkpoints summaries distinguish the store-wide available count,
the selected runs' available count, and the newest saved step. The retained-save
table and latest recorded retention event show which files remain and why a new
checkpoint may replace an older one without increasing the count. Save sequence numbers
(Gen 2 or checkpoint_id 2) identify one run's second save, not the total number of
checkpoints. Lab Home's Scored checkpoints counts sealed lab scores, including
references, and is separate from training snapshots.

The dashboard sends bounded JSON requests to the configured loopback CPU sampler.
Run `python src/sparselab/dashboard/snapshot_sampler.py` in the existing pinned
inference environment with `SPARSELAB_SNAPSHOT_SERVICE_CONFIG` set, rather than
launching that file with Streamlit. Keep the existing 2-thread, 200% CPU, 10 GiB
sampler service envelope and the dashboard's separate envelope. The sampler owns
one model, serializes verification/loading/generation, refuses concurrent work,
and verifies the exact digest on load. It only binds the configured 127.0.0.1 port,
rejects browser origins and unexpected Host values, and does not log prompts or
outputs. The UI never loads a model. Sampling defaults remain 0.7 / top-k 40 /
seed 42 / 32 new tokens, with the same 512-output and 1,024-context limits.
