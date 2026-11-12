# Native MLX on Apple Silicon

Run Lab directly on an Apple Silicon Mac with macOS 14 or newer, an installed
stable CPython 3.14 interpreter, and uv. Use an arm64 terminal and interpreter,
not an x86_64 Rosetta environment. Linux containers on a Mac do not expose Metal,
so there is no MLX image.

The lock selects **MLX 0.32.4**, **mlx-metal 0.32.4**, and **PyTorch 2.15.0**.
Published stable wheels cover CPython 3.14 on macOS arm64; mlx-metal supplies the
Metal runtime. PyTorch is installed because the full Lab CLI currently imports
it even when a declaration selects MLX. Installing the `cpu` extra on macOS
selects the native PyPI Torch wheel, not the Linux CPU wheel index. It does not
change a declaration's engine or select a CPU fallback.

## Install once and reuse

These commands install locked dependencies into this checkout's persistent
`.venv`. They require network access for the first installation, but do not
fetch models or datasets. Keep each active experiment's checkout pinned until
its run finishes; use a separate checkout when trying a new Lab revision.

```sh
git clone https://github.com/supernovae/model-lab.git
cd model-lab
export UV_PYTHON_DOWNLOADS=never
uv sync --locked --python python3.14 --no-dev --extra cpu --extra mlx \
  --no-build-package mlx --no-build-package mlx-metal --no-build-package torch
. .venv/bin/activate
export KLAB_WORK_DIR="$HOME/.local/share/klab"
klab --help
klab runtime env discover --json
```

The wheel-only restrictions on the three framework packages fail if a compatible
published wheel is unavailable. No nightly index, Python downgrade, source build
of a framework, or automatic interpreter download is used. Discovery reports what
is available; it does not run a workload.

Lab is installed editable by `uv sync`. For a later reviewed source revision,
update an idle checkout with Git and repeat the same locked sync command. uv
keeps already satisfied dependency versions; application-only changes do not
rebuild or redownload MLX, Metal, or Torch. Keep `.venv` and uv's download cache
on the Mac. If `pyproject.toml`, `uv.lock`, Python, architecture, or macOS changes,
sync again and inspect the resulting runtime before executing work. Do not copy
a virtual environment between Macs or publish it as an OS-independent image.
Direct commands use the activated environment without checking or repairing
dependency drift; synchronize the selected revision explicitly first.

## Use the existing Lab commands

Set `engine: mlx` and `backend: metal` in the config's `runtime` section and
use the ordinary [lab mode](lab-mode.md) commands. MLX supports FP32 dense and
native block-sparse models; other features and mixed precision are refused. See
[runtime](runtime.md#choose-where-a-run-executes).

Open the native viewer in a separate terminal using the same work root:

```sh
. .venv/bin/activate
export KLAB_WORK_DIR="$HOME/.local/share/klab"
klab dashboard
```

The viewer binds to localhost and reads the local work directory. Each Mac keeps
its own results.

## Validate on the Mac

Linux CI does not exercise Metal. After installing or upgrading, run the MLX
test selection from [testing](test-speed.md) on the Mac before relying on
results. Package availability is listed in the
[MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html)
and on PyPI for [MLX 0.32.4](https://pypi.org/project/mlx/0.32.4/#files),
[mlx-metal 0.32.4](https://pypi.org/project/mlx-metal/0.32.4/#files) and
[Torch 2.15.0](https://pypi.org/project/torch/2.15.0/#files).
