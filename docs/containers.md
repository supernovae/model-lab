# Containers

Lab ships as OCI images in `ghcr.io/supernovae/model-lab`: a CPU image with the
CLI and dashboard, and separate CUDA, ROCm and XPU images. Each installs `klab`
directly, so no host Python, uv or checkout is needed. The
[uv installation](../README.md#install) remains the alternative.

## CPU image

```sh
IMAGE=ghcr.io/supernovae/model-lab:cpu
podman pull "$IMAGE"

# A fresh job volume; this only prints help.
podman run --name lab-help --network=none --memory=4g --cpus=2 --pids-limit=256 \
  -v lab-job:/state "$IMAGE" --help

# A viewer over that job, with the job mounted read-only.
podman run -d --name lab-web --memory=2g --cpus=1 --pids-limit=256 \
  -v lab-web:/state -v lab-job:/observe:ro \
  -p 127.0.0.1:8501:8501 \
  --health-cmd="python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2).read()\"" \
  --health-interval=10s --health-timeout=3s --health-start-period=30s \
  "$IMAGE" dashboard --address 0.0.0.0 \
  --runs-dir /observe/work/runs --lab-dir /observe/work/lab
```

Open <http://127.0.0.1:8501>. Docker accepts the same commands. The image
targets Linux amd64.

- **Arguments.** With none, the image prints `klab --help`. Bare words go to
  Lab (`"$IMAGE" report TRY_ID`); `compare` and `eval` always mean Lab
  commands. `klab`, `python`, `python3`, `sh`, `bash` and `streamlit` run as
  themselves; give a full path for anything else. Nothing is evaluated as
  shell text.
- **State.** `/state/work` is the work directory; home, caches and temporary
  files stay under `/state`. Mount configs and data at the paths your
  declarations name.
- **Network.** The image starts offline. To download a source, run with a
  network and override the relevant `HF_*_OFFLINE` variables.
- **Health.** CLI jobs have no healthcheck; the viewer opts into one for its
  port. A healthy viewer means the server answers, not that a run succeeded.
  The dashboard replaces its launcher process, so stop signals reach Streamlit.
- **Ownership.** For bind mounts on rootless Podman add
  `--userns=keep-id:uid=10001,gid=10001` so host paths match the image user; on
  Docker use `--user "$(id -u):$(id -g)"`.
- **Viewer.** Keep the localhost port mapping. Through a read-only mount the
  viewer cannot write job evidence, including Explorer's cache; it still reads
  live SQLite WAL files.

Pin an execution by digest (`ghcr.io/supernovae/model-lab@sha256:...`). Main
builds publish `:cpu` and `:cpu-sha-COMMIT` and record the digest.

### Build locally

```sh
podman build --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t klab-cpu:local .
python3 container/smoke.py --engine podman --image klab-cpu:local \
  --revision "$(git rev-parse HEAD)" --output /tmp/klab-cpu-smoke
```

Base images are pinned by digest, and the dependency layer comes before the
application layer. The image holds the installed package and its dependencies,
not Git history or credentials. The smoke runs tiny CPU fixtures in bounded
containers.

## Accelerator images

Each backend is enabled by `container/backends/NAME/config.json` and a complete
hashed wheel lock, and builds independently:

- **Dependency images** are tagged `deps-BACKEND-SHA256`. The identity covers the
  wheel lock, backend config and recipe files, the dependency Dockerfile with
  its pinned base images, the build lock, the validation program and the build
  context allowlist. An existing tag is reused by digest and never overwritten.
  Only a reviewed change to those inputs builds a new one.
- **Application images** (`BACKEND-sha-COMMIT`) start from the exact dependency
  digest and install only Lab, offline. Shared packages keep their `uv.lock`
  versions; only the framework stack differs. A new project dependency must
  also update each backend lock, or publication is refused.
- **Checks.** Pull requests validate the plan without downloading wheels. On
  main, each image is checked for its framework build, a CPU tensor and CLI help
  in a network-disabled, read-only, non-root container, then published with its
  recorded digest. No GPU runs in CI.

```sh
python3 -m unittest discover -s container -p 'test_image_plan.py'
python3 container/image_plan.py plan cuda
```

Builds need disk: the workflow requires 12 GiB free (32 GiB for ROCm) and
refuses otherwise, without deleting anything. A backend may supply its own
`dependencies.Dockerfile` when its userspace needs a different OS.

An image does not prove the host driver or device supports every operation.
Select the backend in the run config (`runtime.backend`); an unavailable
explicit backend is an error, never a silent CPU run. Use a CPU viewer over the
job's output mounted read-only.

### Regenerate a lock

Export the common constraints, then resolve the backend's inputs against them:

```sh
uv export --locked --extra cpu --no-dev --no-emit-project --no-emit-package torch \
  --no-hashes --no-header --no-annotate -o /tmp/lab-common-constraints.txt
uv pip compile pyproject.toml container/backends/cuda/requirements.in --no-sources \
  -c /tmp/lab-common-constraints.txt \
  --python-version 3.14 --python-platform x86_64-manylinux_2_36 \
  --only-binary :all: --generate-hashes --no-header --no-annotate \
  --emit-index-url --emit-find-links \
  -o container/backends/cuda/requirements.lock
```

The resolver can omit the hash of a `--find-links` wheel. Take the SHA-256 from
the official index and add it; never drop hash enforcement. `image_plan.py plan`
fails on any unhashed entry.

## CUDA

| Component | Pin |
| --- | --- |
| Python, OS | CPython 3.14.4, Debian Bookworm |
| PyTorch | `2.14.1+cu132` |
| CUDA runtime, Triton | 13.2.86, 3.8.0 |

The full closure is in [requirements.lock](../container/backends/cuda/requirements.lock),
resolved from [requirements.in](../container/backends/cuda/requirements.in),
with the wheel's URL and hash in
[wheel-evidence.json](../container/backends/cuda/wheel-evidence.json). CUDA 13.2
pairs with the R595 driver branch; check the device and driver against NVIDIA's
[release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
and [CDI guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

```sh
IMAGE=ghcr.io/supernovae/model-lab@sha256:PUBLISHED_CUDA_DIGEST
podman run --rm --device nvidia.com/gpu=0 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 8g --cpus 2 --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  -v lab-cuda-JOB_ID:/state "$IMAGE" klab --help
```

## ROCm

Targets AMD gfx1100 (Radeon RX 7900 XTX) on Linux x86_64.

| Component | Pin |
| --- | --- |
| Python, OS | CPython 3.14.4, Debian Bookworm |
| PyTorch | `2.15.0+rocm10.0` (HIP 7.15.26333) |
| ROCm SDK | `rocm[device-all,libraries]==10.0.0` |
| Triton | `triton-rocm==3.9.0` |

The native `rocm-gfx1100-v1` [provisioning recipe](runtime.md#provision-a-vendor-environment)
installs from the same binary and source locks. The `rocm` metapackage is a
hash-pinned source package built with pinned setuptools and
`ROCM_SDK_TARGET_FAMILY=gfx1100`; everything else is a binary wheel. The SDK
wheels alone are about 6.5 GiB compressed. To regenerate, resolve
`container/backends/rocm/requirements.in` for `x86_64-manylinux_2_28`, exclude
`rocm`, and keep the separate source lock. Check host support against AMD's
[compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

```sh
IMAGE=ghcr.io/supernovae/model-lab@sha256:PUBLISHED_ROCM_DIGEST
docker run --rm --network=none --cap-drop=ALL \
  --security-opt=no-new-privileges --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v klab-rocm-state:/state "$IMAGE" klab --help
```

Use the render node of the intended GPU. Never weaken host security or change
device ownership to make access work.

## Intel XPU

| Component | Pin |
| --- | --- |
| OS | Ubuntu 26.04 |
| Python | CPython 3.14.8 (python-build-standalone 20261013) |
| PyTorch, Triton XPU | `2.15.0+xpu`, 3.9.0 |
| Intel SYCL runtimes | 2026.1.2 |
| Level Zero loader, GPU driver | 1.34.0, 26.35.39758.10-0 |
| Graphics compiler, gmmlib | 2.41.5, 22.10.0 |

The driver needs glibc 2.38 or newer, hence Ubuntu 26.04. The
[dependency recipe](../container/backends/xpu/dependencies.Dockerfile) pins the
base, interpreter and packages; [userspace pins](../container/backends/xpu/userspace.lock.json)
fix nine archive URLs and hashes from the Ubuntu snapshot service, and the
[wheel lock](../container/backends/xpu/requirements.lock) pins all packages.
The container supplies userspace only; the host needs the Intel kernel driver
(see [Intel Arc](intel-xpu.md)). To regenerate the lock, resolve
`requirements/xpu-arc.txt` and `container/build-requirements.txt` for
`x86_64-manylinux_2_28`.

```sh
IMAGE='ghcr.io/supernovae/model-lab@sha256:PUBLISHED_XPU_DIGEST'
RENDER_NODE='/dev/dri/INTEL_RENDER_NODE'
docker run --rm --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --device "$RENDER_NODE:$RENDER_NODE" --group-add "$(stat -c %g "$RENDER_NODE")" \
  --mount type=volume,src=lab-xpu-job,dst=/state \
  "$IMAGE" klab runtime env discover --json
```

Never use privileged mode or world-writable devices.
