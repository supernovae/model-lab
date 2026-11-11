# Read-only Learn and Research pages

Start the dashboard with optional report, lifecycle, and evidence roots:

```sh
klab dashboard --runs-dir klab-work/runs --reports-dir artifacts/research-reports \
  --evidence-root . --lifecycle PATH
```

Learn and Research are browseable before a run database exists. Learn shows the packaged lesson cards, scale/data choices, shape/source walkthroughs, and copyable standalone commands; its controls do not write a scaffold or submit training. Research retains the declarative catalog and validated report browsing, and adds known-good references, the declared six-stage research funnel (with unassessed entries outside it), findings including negatives/mixed/inconclusive outcomes, scale-ready declarations, blocked work, and scoped prior-evidence warnings.

The Research page discovers at most 200 **direct** children of `--reports-dir`. A child must be a non-symlink content-addressed directory with a safe `manifest.json`, and the shared bundle loader validates it before rendering. Invalid manifests, hash mismatches, unsafe children, and malformed bundles appear as rejected rows and their charts are not rendered. The page does not open the writer-side experiment store for report browsing, execute bundled Markdown/HTML, or dereference report-controlled paths.

A valid bundle presents raw outcomes, endpoint status, observed pair deltas, calculated architecture/cache quantities, available local telemetry, costs/unavailable values, limitations, and missing/rejected evidence. Entry-linked learned-portability bundles also render campaign outcome counts, source/preparation gates, matched per-seed contrasts, and costs; their raw observations remain in the static bundle. An unlinked historical bundle is labeled as evidence without recorded hypothesis metadata. “Observed delta” is not a winner label: partial/inconclusive data and unavailable values remain explicit.

Lifecycle rows show evidence basis, verification availability, and limitations.
Missing roots keep declarations visible with unavailable reasons; malformed
lifecycle metadata produces a visible error without hiding report browsing. The
page uses text and tables only for lifecycle/report values: it never renders
artifact HTML/Markdown as trusted content, opens a run store, creates a writer,
ranks models, adds a leaderboard, or exposes a mutation/run control. See the
[known-good baseline guide](known-good-baselines.md) for the review and evidence
rules.

The [curated experiment ledger](experiment-ledger.md) connects project questions
to retained lessons across studies. A shared CLI/dashboard ledger projection is
[a longer-term direction](../../ROADMAP.md#longer-term-directions); current Research pages
continue to read their existing lifecycle and verified report sources.
