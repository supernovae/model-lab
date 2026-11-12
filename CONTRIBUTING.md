# Contributing

Use Python 3.14 (selected by `.python-version`) and
`uv sync --locked --extra cpu --dev`. For the Linux CPU checks:

```sh
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --extra cpu pytest -q -n auto --maxprocesses=4 \
  --dist worksteal --durations=25 \
  -m "not mps and not mlx and not cuda and not rocm and not xpu and not network"
```

Automatic CI runs the explicit zero-model Linux `safe` job and the CPU
`lab-loop` job on pull requests and pushes to `main`. The latter trains tiny
models and exercises probes/explorer/dashboard. Manual `safe` dispatch runs
both too; additional integration, platform and full-suite choices require a
separate reviewed allocation. Do not dispatch those to validate documentation.
New commits cancel superseded runs on the same ref.
The manually selected full CPU suite runs on Linux x64 and macOS arm64, with up
to four pytest workers and one BLAS/OpenMP thread per worker.
Work stealing redistributes pending tests when one worker falls behind; module
fixtures can be instantiated on more than one worker and must use isolated paths.
The slowest 25 tests are printed in each job log. Full-suite jobs have a 30-minute
limit so a hung test cannot occupy a runner for six hours. Hardware and network
tests retain their separate opt-in gates. PR checkout uses GitHub's synthetic
merge commit to test integration with the base branch; detached HEAD is expected.

Base installation is Torch-free: the lightweight `klab runtime env` CLI
works without Torch, but the full CLI currently imports a backend framework.
For a provisioned vendor interpreter use it directly or set
`UV_PROJECT_ENVIRONMENT` to its environment prefix and run with
`uv run --locked --no-sync`; never sync the CPU extra into it. Do not commit
downloaded corpora, prepared arrays, run directories, or checkpoints. Reports
should include the resolved configuration and relevant run evidence.

For a local CPU example, run `klab runtime env discover --json`, then
`klab runtime env register cpu-py314 --python "$PWD/.venv/bin/python" --backend cpu --json`
and `klab runtime env doctor cpu-py314 --json`. Select it with
`klab stage configs/runtime_smoke_cpu.yaml --through inspect --runtime cpu-py314 --output /absolute/disposable/stage`.
The XDG runtime root and host-local registry are separate from
`KLAB_WORK_DIR` and scientific locks; never copy runtime IDs as capability
claims to another host. See [machine-local runtime environments](docs/runtime.md#machine-local-runtime-environments).

Work on an ordinary branch in the existing checkout, not an automatically created
worktree:

```sh
git switch main
git pull --ff-only
git switch -c feat/<task>
export KLAB_WORK_DIR=/data/klab
```

Choose an external persistent filesystem with enough space for actual campaigns.
For ordinary local experiments, follow [lab mode](docs/lab-mode.md); the sealed
record captures source state and effective inputs. For **release work**, commit
corpus/source-rights pins, ExperimentPlan/base run, evaluation suite and runtime
requirement **before** acquisition, tokenizer fitting, preparation or training.
Inspect `klab research snapshot <plan-or-campaign> --json` before
expensive execution. Retain a compact verified evidence reference in Git instead
of copying datasets/checkpoints into the repository. `scratch/` and optional
`cache/` under the persistent root are disposable; neither is proof. A reviewed
decision about model quality requires an explicit human receipt. Read the
[recovery and lineage operating protocol](docs/research/lifecycle-recovery.md)
before changing paths or archiving rights-bound corpus content.

## Pre-production migration policy

Until a production release is tagged, adapt experiment plans, recipes and models
to current supported Lab behavior. Do not change Lab to preserve old execution
behavior or add backward compatibility or version adapters for legacy recipes.
A production tag permits reassessing this policy; it does not automatically
authorize legacy adapters.

Preserve historical evidence bytes and identities unchanged. Keep historical
claims in Workshop, labeled as historical rather than guarantees of current Lab
behavior. Read-only interpretation, hash verification and audits of that evidence
are allowed; they do not make old artifacts executable or establish current
compatibility.

Track reproduction and regeneration in the backlog against current interfaces.
Any later reproduction or regeneration is a new experiment with a new identity;
retain the original result and its limitations. This policy grants no execution,
recovery, acquisition or training authority.

A change to a command, a config field or a default updates the page that
describes it in the same pull request. Each subject has one page, listed in the
[documentation index](docs/README.md); other pages link to it. A page that only
records what happened on a date belongs in `docs/history/`.

During PR review, check that a migration adapts declarations to current behavior,
preserves archived evidence and separates historical claims from current
validation. Reject changes whose purpose is restoring old execution behavior
through compatibility code. Record current-interface gaps separately from a
proposed reproduction, in a GitHub issue.

## Architecture experiments

The [research contribution workflow](docs/research/contributing.md) explains validated local catalog/recipe forks, declared controls and variations, train-only data boundaries, retained negative outcomes, and evidence identities. Do not commit downloaded corpora, prepared arrays, run directories, or checkpoints; a static report bundle is small evidence, not a replacement for its stated limitations.

## Planning and contributions

[ROADMAP.md](ROADMAP.md) summarizes feature direction. Use
[GitHub issues](https://github.com/supernovae/model-lab/issues) for feature
requests, bug reports and proposed code work. Search existing issues first;
include the use case, expected behavior and relevant reproduction details.
Link related issues in pull requests, where implementation, review and CI checks
are tracked. Keep the roadmap directional rather than duplicating issue status.

## Documentation

Write for someone running their own experiment. Use native `klab` commands
and YAML/JSON declarations, through the locked environment; keep Python snippets
out of user workflows. If a required operation is API-only, describe the missing
CLI/DSL input, output and failure criteria in a GitHub issue instead of inventing a command or teaching a private script. Label placeholders and prerequisites, and
use fresh task directories under the persistent work root. A config with an
explicit in-checkout destination keeps it; setting the root does not relocate it.

Keep the beginner route small: `try`, `report`, `probe`, `compare`, `explore`.
Link advanced operations separately. Remove superseded instructions rather than
adding deprecation labels, alias tables or compatibility walkthroughs. Preserve
original evidence and research data; documentation cleanup never authorizes
pruning run artifacts. Track proposed command removals in a GitHub issue before changing runtime behavior.

Keep feature availability in the [README matrix](README.md#feature-matrix),
runtime restrictions in [runtime](docs/runtime.md), and scientific lessons in the
[experiment ledger](docs/research/experiment-ledger.md). Link to the owner of a
topic instead of copying status summaries. Describe features by behavior rather
than the branch, release phase or author who introduced them. Retain schema and
protocol versions, source pins, checkpoint IDs, license attribution and historical
paths where they identify evidence. Never rewrite a prior result to modernize a
tutorial, or erase a failed result as obsolete documentation.

Describe research by its question and changed variables, then state where it
ran: “FFN-width comparison, run with PyTorch on an AMD GPU,” for example.
CPU, CUDA, ROCm, XPU and MPS describe PyTorch execution capabilities; MLX is a
separate engine using Metal on Apple Silicon. Reserve “runtime validation” and
“acceptance” for checks of Model Lab software behavior in an environment. Do not
present the processor as the subject of a model experiment unless the protocol
explicitly compares execution backends.

Keep general guides independent of the experiments used to develop the lab.
Explain the DSL, its value and supported behavior; put named campaigns, run
budgets and measured outcomes in research records. Dataset-specific walkthroughs
are optional teaching references. Preserve required schema fields such as
`campaign_version: 1`, but call the feature “Campaigns,” not “Campaign v1.”

For documentation-only changes, check local links and heading anchors, code-fence
balance, CLI parser support and explicit path semantics. Do not launch a research
run to validate prose. Record unavailable hardware or execution checks at handoff.
