# ROCm image for Radeon RX 7900 XTX

This recipe targets stable CPython 3.14 on Linux x86-64 with AMD gfx1100
(RX 7900 XTX). It uses the existing Lab ROCm backend. It is not an all-AMD image
or a claim of Windows/WSL qualification. A compatible host driver and device
access are prerequisites; see AMD's
[compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

## Exact runtime pins

| Component | Pin |
| --- | --- |
| Python/base OS | Python 3.14.4, Debian Bookworm, immutable base digest in the Dockerfile |
| Official PyTorch | `2.15.0+rocm10.0` |
| ROCm SDK | `10.0.0`, including the complete `device-all,libraries` closure |
| Embedded HIP build | `7.15.26333` (checked separately from SDK version) |
| Triton ROCm | `triton-rocm==3.9.0` (imports as `triton`) |
| Metapackage builder | `setuptools==84.0.0` |
| Lab build backend | `uv_build==0.12.17` |

The [official PyTorch index](https://download.pytorch.org/whl/rocm10.0/torch/)
publishes the CPython 3.14 Linux x86_64 wheel. Its metadata requires
`rocm[device-all,libraries]==10.0.*`, not AMD's former Torch device extras.
The local provisioner and container now use the same exact framework/SDK pins.
Existing runtimes remain intact; provision a new runtime ID and retain the prior
checkout/environment for rollback. Do not combine old `amd-torch-device-*` or
AMD `triton` packages with this distribution.

## Dependency reuse

The shared accelerator workflow builds this backend independently. Its immutable
dependency identity includes the backend recipe, exact wheel/source locks,
configuration, Python/OS base, and shared build inputs. Ordinary application
updates reuse the resolved dependency digest and add only the Lab application
layer. Missing dependency images bootstrap once; heavyweight dependencies are
not rebuilt just because the application source changed. See the
[accelerator image guide](container-builds.md) for publication and refresh.

`requirements.lock` contains the binary closure for Lab, Torch, Triton and all
SDK device packages requested by Torch. Package-specific official PyTorch
`--find-links` entries retain PyPI as the source for common application packages;
SDK links resolve to AMD's official package server. Exact versions and SHA256
hashes are recorded, while shared application versions remain constrained by
`uv.lock`. The SDK wheels alone total approximately 6.54 GiB compressed; plan
additional storage for Torch, Triton, unpacked packages, build layers and cache.
Independent review measured about 18 GB for the native installed environment.
The image workflow requires at least 32 GiB free before a ROCm build to leave
room for downloads, layers and the base image; it refuses
an undersized runner without deleting host data. This floor is not a capacity
guarantee for every Docker storage configuration.
The image's validation target remains gfx1100 even though Torch declares broader
package dependencies. Those dependencies do not establish other GPUs' support.

`rocm==10.0.0` is AMD's 24,780-byte source metapackage. Its inspected archive is
hash-pinned in `source-requirements.lock`. Only that package is excluded from
binary resolution; all of its required core, libraries and device-all dependencies
are explicitly retained. The build uses pinned setuptools without build isolation
or dependency resolution. `ROCM_SDK_TARGET_FAMILY=gfx1100` prevents build-time
hardware discovery in both the container and native provisioner.
The Dockerfile checks dependencies, exact Torch/HIP versions, and a tiny CPU tensor.
The image planner additionally rejects missing/mismatched recipe pins.
Native provisioning reuses the same binary and source locks with `--require-hashes`
and `--no-deps`; only the metapackage permits a source build, using the already
installed pinned setuptools without build isolation. Provision receipts record both
lock digests. The un-hashed `requirements/rocm-gfx1100.txt` is resolution input,
not the native installation path.

To regenerate, export the common constraints as in the [XPU guide](intel-xpu.md#reproducing-the-dependency-lock),
then resolve `pyproject.toml` with `container/backends/rocm/requirements.in` for
Python 3.14 / `x86_64-manylinux_2_28`, using the same hash and index emission
flags. Pass `--excludes` a requirements file containing only `rocm`, and set
`ROCM_SDK_TARGET_FAMILY=gfx1100`. Preserve the separately audited source lock.
Confirm any missing find-links hashes against downloaded bytes or the publisher's
wheel hash; an unhashed entry must fail `container/image_plan.py plan rocm`.

## Running Lab

After reviewed publication, use the published immutable ROCm application digest
with the [container state and CLI conventions](containers.md). On an already
configured Linux Docker host, a minimal CLI check is:

```sh
# Replace this placeholder with the published ROCm application digest.
IMAGE=ghcr.io/supernovae/model-lab@sha256:REPLACE_WITH_ROCM_DIGEST
docker run --rm --network=none --cap-drop=ALL \
  --security-opt=no-new-privileges --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v klab-rocm-state:/state "$IMAGE" klab --help
```

Use the render node belonging to the intended GPU if it differs from
`renderD128`; the example assumes an existing Docker host and existing device
permissions. Rootless Podman needs the host's supported supplemental-group/device
mapping. Do not weaken host security or change device ownership to make it work.
The help command checks CLI wiring only. Actual `try`/`probe` work still needs
your own inputs and resource envelope, with `backend: rocm` in the declaration.
The separate CPU viewer can observe the same persisted evidence read-only.

Lab records HIP-backed PyTorch execution as `rocm`; an unavailable explicit
backend refuses rather than silently retrying on CPU. See [runtime selection](runtime.md).
The `rocm-gfx1100-v1` provisioning recipe shares the image's framework pins;
this image does not mutate any already provisioned environment.

## Validation boundary

Package metadata and the wheel's `torch/version.py` establish the Torch/SDK/HIP
versions above. The source metapackage was inspected and hashed. CPU tests and
packaging checks do not exercise GPU kernels or certify the resulting ROCm image.

The coordinating task reported a separate host smoke on 2026-11-03 (task
`01a1a25a-19cb-75d8-9901-dd312eeb602a`): RX 7900 XTX/gfx1100, Python 3.14.4,
Torch 2.15.0+rocm10.0, Triton 3.9.0, three BF16 AdamW updates of a 50.6M model,
finite gradients, inference and exact checkpoint tensor recovery. The original
2.13 environment was retained. This packaging task received that summary but
could not retrieve the underlying task's raw evidence or installed dependency
manifest. It therefore does not claim an exact environment reproduction or full
Lab continuation, compiled kernels, performance, container, or full-run acceptance.
Native Mac/MPS/MLX and XPU validation are also separate outstanding checks.
No active installations, driver settings or training were modified here.
