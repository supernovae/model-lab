# Fast test feedback

From the repository root, install the locked development environment once:

```sh
uv sync --locked --extra cpu --dev
```

Run the tests nearest the files you changed first. Examples:

```sh
# Corpus acquisition, normalization, generation, exports, and provenance
uv run --locked --extra cpu pytest -q tests/test_corpus_acquisition.py tests/test_corpus_pipeline.py tests/test_corpus_generators.py tests/test_corpus_exports.py tests/test_corpus_provenance.py
# Dataset/config/tokenizer/packing integration
uv run --locked --extra cpu pytest -q tests/test_config.py tests/test_datasets.py tests/test_conversations.py tests/test_local_chat.py tests/test_tokenizer.py tests/test_packing_manifest.py
# Hub credentials (without a public endpoint)
uv run --locked --extra cpu pytest -q tests/test_hf_auth.py
```

Pushes to `main` and all PRs run four non-overlapping Linux jobs in parallel:
`safe` (Ruff and fast config and authority checks), `lab-loop` (model
construction, training, checkpointing, resume, `try` + probe, explorer,
references and generation), `fast` (the remaining non-slow CPU suite) and
`staging` (three supervised staging regressions from `tests/test_staging.py`,
which launch real pilot children, with a 12-minute limit and the same
four-worker cap).
See the [workflow](../.github/workflows/ci.yml) for exact nodes and budgets.

Draft PRs receive the same checks. Opening, synchronizing (new commits), or
reopening a PR runs CI; changing only draft/ready state does not rerun unchanged
code. Pushes to `main` still validate the merged tree. A draft/ready transition
also does not retry a failed or cancelled run: inspect the failure or cancellation
first, then explicitly rerun the applicable check if warranted, or push a fix.
Do not use blind retries to mask a failure.

The `lab-loop` job uses work stealing with the same four-worker cap so an
expensive file does not hold one worker while others finish. Its 25 slowest
test phases are printed in the job log, as in `fast`. Module fixtures can run
once per worker; their temporary directories remain worker-local.

Manual `workflow_dispatch` with `safe` selects `safe` and `lab-loop`. Additional
`integration-linux`, `platform-macos` and `release-candidate` choices need a
separately reviewed allocation. Do not dispatch them for prose changes. For
documentation, check links, anchors, parser arguments and configuration schemas
without calling command handlers. Ordinary PR success does not certify all
training paths, accelerator behavior or macOS behavior.

Plain `pytest` is parallel by default (`-n auto --dist worksteal` in
`pyproject.toml`). `tests/conftest.py` sizes `auto` to one worker per usable
CPU, capped at one per 2 GiB of available RAM, and pins each worker to one
math thread. A large workstation uses most of its cores, and a 4-vCPU runner or
small laptop gets 4 or fewer. Pass `-n N` to choose, or `-n 0` to debug
serially. Pytest's `tmp_path` and `tmp_path_factory` keep artifacts in
worker-specific directories. Do not share a mutable work directory between
independently launched suites.

```sh
uv run --locked --extra cpu pytest -q -m 'not mps and not mlx and not cuda and not rocm and not xpu and not network'
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
```

## Fast and full suites

The fast suite skips tests marked `slow`. It is the default local check before
a push and what every PR runs in CI. The full suite runs nightly and is the
release gate.

```sh
# Fast: duration depends on the selected tests, revision, and CPU
uv run --locked --extra cpu pytest -q -m 'not slow and not mps and not mlx and not cuda and not rocm and not xpu and not network'
# Full: includes slow integration tests; leave routine full runs to nightly
uv run --locked --extra cpu pytest -q -m 'not mps and not mlx and not cuda and not rocm and not xpu and not network'
# Only the slow tests
uv run --locked --extra cpu pytest -q -m slow
```

`tests/conftest.py` applies the `slow` marker to the base node ids listed in
[`tests/slow_tests.txt`](../tests/slow_tests.txt), so every parametrized case
of a listed test follows. The slow tests are fewer than a tenth of the suite
but take more than 90% of its test time: end-to-end campaign, corpus and
lifecycle runs, tiny training, and native subprocess monitors.
`tests/test_slow_list.py` fails when an entry no longer names a real test.

### Where each suite runs

- **While iterating (agents included):** run the fast suite, plus the slow
  tests for the areas you touched (`pytest -q path/to/test_file.py::test_name`,
  or `-m slow` restricted to the relevant files). Don't run the full suite
  locally; nightly covers it.
- **Every PR and push to main:** `safe`, `lab-loop`, `fast` and `staging` on
  Linux.
- **Nightly** ([`.github/workflows/nightly.yml`](../.github/workflows/nightly.yml),
  08:17 UTC, also runnable manually): the full CPU suite, slow tests included,
  on `main`. A failure opens the issue "Nightly full suite failing on main",
  or comments on it if it is already open.
- **Manual:** the `workflow_dispatch` choice `release-candidate` runs the full
  suite on Linux and macOS.

### Profiling and the slow list

Profile a full run, then list any test whose call time is at least 1 s:

```sh
uv run --locked --extra cpu pytest -q -p no:randomly --durations=50 \
  --junitxml=/tmp/suite.xml -o junit_duration_report=call
```

Add new slow tests to `tests/slow_tests.txt` in sorted order. A test whose
fixture setup is heavy, rather than its call, belongs there too.

### Known failures

[`tests/known_failures.txt`](../tests/known_failures.txt) lists tests that
fail for a root cause tracked in an open issue. The conftest skips each one
with its reason and issue number, so `-rs` shows exactly what is not running.
Delete an entry in the PR that fixes its issue.

### Host RAM in tests

Host-work planning (`klab.host_capacity`) refuses work when measured
available RAM minus a reserve cannot fit one worker. Under xdist, or on a box
shared with other jobs, that made unrelated tests fail with "inadequate
measured RAM after reserve". Every Model Lab reader of host RAM, including
reserves computed from total RAM, goes through
`host_capacity.measure_memory()`. An autouse fixture in `tests/conftest.py`
pins it to a fixed host (64 GiB total, 48 GiB available), whatever the
physical machine. For child interpreters launched by tests (`python -c`,
`python -m klab`), the fixture also sets
`KLAB_TEST_HOST_RAM=total:available` and puts
[`tests/child_bootstrap/`](../tests/child_bootstrap/sitecustomize.py) first on
`PYTHONPATH`. That directory's `sitecustomize.py` applies the same pin at
interpreter startup. The shipped `measure_memory()` has no override, so the
variable does nothing outside the test bootstrap: it cannot change a real
measurement or get past a low-memory limit. `tests/test_host_capacity.py`
covers both directions. Tests that
monkeypatch `psutil.virtual_memory` themselves still see their own values,
which is how `tests/test_host_capacity.py` covers the reserve logic.

## Runtime probe reuse in tests

Every `train` and `try` validates its runtime with `klab.runtime_probe` in a
fresh interpreter that imports the framework. That costs about 1.5 s on a
workstation and much more on a loaded 4-vCPU runner, and tests start hundreds
of tiny runs: the probe was most of the suite's wall time. The root
`conftest.py` therefore runs the real probe once per distinct request in each
test process and reuses the successful result. Failures are never reused,
`tests/test_runtime.py` always probes for real, a test that replaces
`subprocess.run` gets its own result, and children launched by a test still
probe. Shipped code is unchanged: every real run probes.

When a test is slow, profile it before shrinking it
(`python -m cProfile -m pytest -n 0 TEST`): repeated subprocess startup is the
usual cause, not model size.

## CPU worker limits and stall diagnosis

The root test `conftest.py` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1` during pytest
startup. This matches the CI environment and happens before collection and
xdist worker creation, so workers and subprocesses they launch inherit one
thread per CPU library. If a pytest plugin imported Torch before the
conftest, the conftest also caps Torch's intra-op pool without importing Torch
for Torch-free tests. Like CI, it leaves the inter-op pool unchanged. These limits
apply only to test processes; they do not change lab runtime defaults.

Campaign status projection verifies each completed dependency-DAG stage once
per traversal, including shared upstream nodes. Later projections and checks
before dispatch verify again; no completed-stage cache survives the traversal.
This is not generic snapshot proof reuse across projections or commands.

Verifier authority walks deduplicate module scheduling and file discovery only
within one closure walk. Each later authority lookup still reads and hashes the
verifier source bytes. Proof lookup diagnostics use the same authority that was
compared with the receipt, rather than walking the closure again for telemetry.

The CPU gate includes native Campaign integrations with real worker startup,
staging, training, checkpoint transfer, continuation, and evaluation. These can
take several minutes. A whole-command deadline must allow all separately
bounded runs and the subsequent verification; a faulthandler stack report is
not a test failure or a run deadline.

For a suspected CPU stall, use a small fixed worker count and ask
pytest's built-in faulthandler plugin for thread stacks after a bounded interval:

```sh
uv run --locked --extra cpu pytest -vv -n 2 \
  -o faulthandler_timeout=120 \
  -m 'not mps and not mlx and not cuda and not rocm and not xpu and not network'
```

`faulthandler_timeout` reports stacks; it does not skip, kill, or otherwise
turn a slow test into a passing result. Use the reported test name to rerun it
serially with the same option when isolating the blockage.

On an Apple Silicon machine with the optional MLX runtime installed (`uv sync --locked --dev --extra cpu --extra mlx`), run hardware tests in **one serial process**, after CPU workers finish:

```sh
uv run --locked --extra cpu --extra mlx pytest -q -n 0 -m 'mlx and not mps and not network'
# Opt-in MPS tests when the device is available; the current CI gate excludes these.
uv run --locked --extra cpu pytest -q -n 0 -m 'mps and not network'
```

Always pass `-n 0` for MLX/MPS or run another accelerator test/training process against the same device at the same time. Network-marked tests require explicit separate opt-in.

## Why the gate is structured this way

The full CPU jobs on Linux and macOS keep their hardware exclusions and
30-minute limits and run only on manual dispatch. The automatic `safe`, `lab-loop`, `fast` and `staging` jobs have 8-, 12-, 12- and 12-minute limits; the nightly full suite has 25 minutes. The manual hosted
CPU jobs use up to four workers with work-stealing, as local runs do by default.

Locked uv downloads are cached through `setup-uv`. Accelerator and network
checks are separate explicit opt-ins.
