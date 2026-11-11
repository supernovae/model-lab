# CUDA container

This Linux amd64 image packages the existing PyTorch CUDA backend. It does not
install a host driver or change Lab's execution engine. The CPU image and native
uv installation remain separate options.

## Pinned runtime

Verified against official package indexes on 2026-10-15:

- CPython 3.14.4, Debian Bookworm (immutable base digest in the dependency recipe).
- PyTorch **2.14.1+cu132**, CPython 3.14 Linux x86_64 wheel.
- CUDA runtime **13.2.86** and Triton **3.8.0**; the complete transitive Python
  and CUDA library closure is hash-pinned in
  [requirements.lock](../container/backends/cuda/requirements.lock).
- Source constraints and the reproducible resolver input are in
  [requirements.in](../container/backends/cuda/requirements.in); the official
  Torch wheel URL/hash are retained in
  [wheel-evidence.json](../container/backends/cuda/wheel-evidence.json).

The [official CUDA 13.2 wheel index](https://download.pytorch.org/whl/cu132/torch/)
contains this stable wheel. The CUDA 13.4 index currently has no CPython 3.14
wheel; using the newer toolkit release number would not produce a supported
stable PyTorch combination. This recipe uses neither nightly wheels nor a
Python downgrade. It does not alter the older native `cuda-cu126-v1` recipe or
its recorded runtime identities.

## Run on an already provisioned worker

Prerequisites: Linux amd64, a supported NVIDIA GPU, a compatible existing driver,
and an already configured container engine/device integration. CUDA 13.2's
corresponding driver branch is R595; CUDA 13.x minor-version compatibility has
additional restrictions. Check the actual device/driver against NVIDIA's
[release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
and [container CDI guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).
This guide does not provision or modify them. WSL support also depends on
[NVIDIA's supported Windows host driver path](https://docs.nvidia.com/cuda/wsl-user-guide/).

After the reviewed workflow has published a CUDA image, select its exact
`sha256` digest from the successful workflow summary. The following digest and
job ID are placeholders; replace both before use. The named volume persists
outputs under `/state`. This example uses an existing Podman NVIDIA CDI device
named `nvidia.com/gpu=0`; choose the already configured device on your worker.

```sh
IMAGE=ghcr.io/supernovae/model-lab@sha256:REPLACE_WITH_PUBLISHED_CUDA_DIGEST
podman run --rm --device nvidia.com/gpu=0 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 8g --cpus 2 --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  -v lab-cuda-JOB_ID:/state "$IMAGE" klab --help
```

This command is an installed CLI check, not a GPU execution test. Choose limits
for the actual approved attempt, select `runtime.backend: cuda` in its native
configuration, and use the existing [lab workflow](lab-mode.md). An unavailable
explicit CUDA backend is an error; it must not silently select CPU. Mount only
that job's inputs and persistent work directory. See [container storage and the
separate web viewer](containers.md) for read-only observation of active evidence.

## Dependency reuse and verification

[Container build delivery](container-builds.md) describes dependency image keys,
registry reuse by immutable digest, and separate thin application images. Source
updates do not resolve or reinstall the heavy dependency layer. Reviewed lock,
base image or dependency-recipe changes produce a new dependency identity.

Regenerate the lock only for an intentional dependency update using the installed
stable uv resolver (the initial lock used uv 0.12.19):

```sh
uv export --locked --extra cpu --no-dev --no-emit-project --no-emit-package torch \
  --no-hashes --no-header --no-annotate -o /tmp/lab-application-constraints.txt
uv pip compile pyproject.toml container/backends/cuda/requirements.in --no-sources \
  -c /tmp/lab-application-constraints.txt \
  --python-version 3.14 --python-platform x86_64-manylinux_2_36 \
  --only-binary :all: --generate-hashes --no-header --no-annotate \
  --emit-index-url --emit-find-links \
  -o container/backends/cuda/requirements.lock
```

Application requirements come directly from `pyproject.toml`; the constraints
retain their tested `uv.lock` versions. Only backend-specific packages change.

The resolver may omit the hash for a `--find-links` Torch wheel. Require the
matching CPython/platform wheel's SHA256 from the official index and retain it
in the lock and wheel evidence before review; never remove hash enforcement to
make an incomplete lock install. The pipeline validates that every package is
pinned and hashed before enabling a build.

Validation for this change is package-index/metadata resolution and lightweight
recipe checks only. No CUDA image has been built or published yet, and no NVIDIA
hardware, model, training or GPU correctness test was run. The workflow's CPU
import/CLI smoke will establish packaging only; actual CUDA execution remains a
separate worker check and approved attempt.