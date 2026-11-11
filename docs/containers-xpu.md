# Intel XPU container

This recipe packages upstream PyTorch XPU for Linux amd64 on stable Python 3.14.
It uses the existing Lab backend, CLI and native web viewer. No IPEX or separate
trainer is introduced. **The recipe has not yet been built or tested on Intel
hardware.** Publication requires its dependency build/import checks; a successful
image build still does not establish GPU availability, precision or performance.

## Pinned environment

| Component | Pin |
| --- | --- |
| OS | Ubuntu 26.04, immutable amd64 image digest |
| Python | CPython 3.14.8, Astral standalone release 20261013, SHA256 |
| PyTorch | 2.15.0+xpu, stable CPython 3.14 manylinux_2_28 wheel |
| Triton XPU | 3.9.0, stable CPython 3.14 wheel |
| Intel SYCL runtimes | 2026.1.2 and exact transitive wheel pins |
| Level Zero loader | 1.34.0, official Ubuntu 24.04 binary |
| Intel Level Zero GPU driver | 26.35.39758.10-0 |
| Intel graphics compiler | 2.41.5, release build 22716 |
| Intel graphics memory management | 22.10.0 |
| Build backend | uv-build 0.12.17 |

The [dependency recipe](../container/backends/xpu/dependencies.Dockerfile)
contains immutable base, interpreter and package checksums.
[Userspace package pins](../container/backends/xpu/userspace.lock.json) record
nine exact Intel/Ubuntu archive URLs and SHA256 values. Ubuntu packages use the
[Ubuntu snapshot service](https://snapshot.ubuntu.com/) at `20261015T160000Z`,
matching the original recipe date, because superseded versions can disappear
from the live archive. The snapshot URLs retain the same package versions and
SHA256 checks; a different response still refuses the build. The
[wheel lock](../container/backends/xpu/requirements.lock) pins all 93 packages,
including public Lab dependencies. Nothing installs or modifies a host driver.

The driver needs glibc >=2.38 and libstdc++ >=13.1, so this backend uses Ubuntu
26.04 rather than the CPU image's Debian Bookworm base. Metadata for the loader,
driver, compiler and gmmlib was inspected; their declared OS dependencies are
included as pinned Ubuntu packages. Loader 1.34.0's Ubuntu 24.04 binary declares
minimum ABI versions satisfied by Ubuntu 26.04. Build checks must load the
loader, GPU driver and compiler libraries before checking the installed XPU
framework. All package URLs were verified, but no package was installed during
recipe preparation.

Official references: [stable PyTorch XPU](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html),
[Intel compute runtime](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10),
[Level Zero loader](https://github.com/oneapi-src/level-zero/releases/tag/v1.34.0),
[Python standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20261013).

## Build and reuse

The shared accelerator image workflow selects `xpu`. Its dependency identity
includes this backend's Dockerfile, wheel lock, package pins and config, plus the
shared pipeline inputs. A missing dependency image bootstraps once; subsequent
application changes reuse its exact registry digest and add only the public Lab
application layer. An explicit dependency-input update produces a new identity.
XPU dependencies are separate from CUDA and ROCm; no combined accelerator image
or per-application heavyweight dependency rebuild is needed.

For manual local builds after the shared accelerator pipeline is present:

```sh
docker build --platform linux/amd64 \
  -f container/backends/xpu/dependencies.Dockerfile -t lab-xpu-deps:local .
docker build --platform linux/amd64 -f container/application.Dockerfile \
  --build-arg DEPENDENCY_IMAGE=lab-xpu-deps:local \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t lab-xpu:local .
```

These commands download packages and build an image; they were not executed
while preparing this recipe. Use the published immutable image digest after a
successful publication, rather than assuming the `xpu` tag already exists.

## Device access and native usage

Use an already provisioned supported Intel kernel driver and render-device
access on native Linux. PyTorch's driver matrix and the existing
[Intel acceleration guide](intel-xpu.md) define host prerequisites. The container
supplies userspace libraries but cannot create hardware or replace a missing
kernel driver. Pick the actual render node for the intended Intel card, and use
its existing numeric group permissions. For Docker, after replacing the two
placeholders:

```sh
# IMAGE is a published immutable image reference; RENDER_NODE is the existing Intel node.
IMAGE='ghcr.io/supernovae/model-lab@sha256:REPLACE_WITH_PUBLISHED_DIGEST'
RENDER_NODE='/dev/dri/REPLACE_WITH_INTEL_RENDER_NODE'
docker run --rm --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --device "$RENDER_NODE:$RENDER_NODE" --group-add "$(stat -c %g "$RENDER_NODE")" \
  --mount type=volume,src=lab-xpu-job,dst=/state \
  "$IMAGE" klab runtime env discover --json
```

Discovery is a passive availability check, not a model workload. Do not use
privileged mode, unconfined profiles or world-writable devices. For rootless
engines use existing device/group access supported by that engine; failure to
access the device is a prerequisite failure, not a reason to relax confinement.
Windows/WSL GPU forwarding is not validated by this Linux recipe.

For authorized experiments, use the native `try` and `report` instructions in
[the Intel guide](intel-xpu.md), selecting `backend: xpu` in the config. Keep each
job's persistent output separate. Run the CPU web viewer without GPU access over
the job output mounted read-only, following [the container guide](containers.md).
The existing native provision/register/doctor path remains available if a
container is unsuitable; never sync the CPU extra into an XPU environment.

## Lock reproduction and validation

```sh
uv export --locked --extra cpu --no-dev --no-emit-project --no-emit-package torch \
  --no-hashes -o /tmp/lab-common-constraints.txt
uv pip compile pyproject.toml requirements/xpu-arc.txt container/build-requirements.txt \
  -c /tmp/lab-common-constraints.txt \
  --no-sources --python-version 3.14 --python-platform x86_64-manylinux_2_28 \
  --only-binary :all: --generate-hashes --no-annotate --no-header \
  --emit-index-url --emit-find-links -o /tmp/xpu-requirements.lock
```

The export preserves the repository-locked common application dependencies;
only the accelerator stack differs. This resolves wheel metadata without
creating an environment. uv 0.12.19 omitted
the two find-links hashes; the checked-in lock adds the official CPython 3.14
SHA256 fragments from the [Torch index](https://download.pytorch.org/whl/xpu/torch/)
and [Triton XPU index](https://download.pytorch.org/whl/xpu/triton-xpu/).
Installation requires binary wheels and hashes. Do not replace the checked-in
lock with unhashed resolver output.

Validation during preparation covers metadata resolution, exact package/hash
syntax, JSON, local documentation links and repository formatting. No XPU image
build, GPU workload, model acquisition or host configuration change was performed.
