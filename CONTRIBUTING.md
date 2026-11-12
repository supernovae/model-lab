# Contributing

## Set up

Lab uses Python 3.14 (pinned in `.python-version`) and uv:

```sh
uv sync --locked --extra cpu --dev
```

Work on a branch of your checkout and keep run outputs in a work directory
outside it:

```sh
git switch main && git pull --ff-only
git switch -c feat/my-change
export KLAB_WORK_DIR="$HOME/.local/share/klab"
```

Never commit corpora, prepared arrays, run directories or checkpoints.

## Check a change

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest tests -q -m 'not slow and not mps and not mlx and not cuda and not rocm and not xpu and not network'
```

The last command is the fast suite. Also run the slow tests for the area you
touched; [test speed](docs/test-speed.md) explains the split and how to find
them.

Every pull request runs four jobs: `safe` (configuration and verifier checks),
`lab-loop` (tiny models through `try`, probes, the explorer and the dashboard),
`fast` (the rest of the fast suite) and `staging`. The container workflow builds
the CPU image and smoke-tests it. Integration, macOS and full-suite runs are
started by hand from the Actions tab. Hardware and network tests run only on
machines that have them.

## Change formats forward

Lab has one current interface for each operation. When a config field, a
command or a file format changes, update every caller in the same pull request
and say in the description what users must rebuild or re-prepare. Do not add
readers for old formats, version adapters or aliases. A rerun with new code is
a new experiment with a new identity.

## Documentation

Each subject has one page, listed in the [documentation index](docs/README.md).
A change to a command, config field or default updates its page in the same pull
request, and other pages link to it rather than repeating it.

Write for someone building their own model:

- Describe how Lab behaves now. Leave out how it used to behave, which pull
  request introduced something and why it changed; Git keeps that history.
- Show `klab` commands and YAML. If an operation has no command, open an issue
  for one rather than documenting a script.
- Mark placeholders, and put outputs under the work directory.
- Describe an experiment by its question and what changed, then where it ran:
  "feed-forward width, PyTorch on an AMD GPU". Name the hardware as the subject
  only when the experiment compares hardware.
- Named datasets, model projects and their results belong in [Kybern
  Workshop](https://github.com/supernovae/kybern-workshop) or your own
  repository.

`tests/test_docs.py` checks links, the index and every `klab` command a page
shows. Run it after a docs change; do not train a model to check prose.

## Issues and planning

Use [GitHub issues](https://github.com/supernovae/model-lab/issues) for bugs,
feature requests and proposed work. Search first, and describe the use case and
the expected behavior. [ROADMAP.md](ROADMAP.md) is a short summary of direction,
not a task list.

Agents follow [AGENTS.md](AGENTS.md) as well.
