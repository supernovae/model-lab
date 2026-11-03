# Intel Arc acceleration

Commands below use the [activated, provisioned environment](../README.md#first-run).

SparseLab uses upstream PyTorch's `torch.xpu` backend for Intel GPUs. The
`xpu-arc-v1` recipe provisions an isolated Linux x86_64 / Python 3.14 environment
with `torch==2.15.0+xpu` and the Intel runtime pins in
[`requirements/xpu-arc.txt`](../requirements/xpu-arc.txt). It follows the same
native provision/register/doctor path as ROCm. No Intel Extension for PyTorch
(IPEX), alternate trainer, CPU extra in the vendor environment, or full oneAPI
build toolkit is needed for this binary-wheel path.

PyTorch's [Intel GPU guide](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)
lists Arc A-series and B-series on supported Ubuntu and WSL2 environments.
The A380 is an A-series device. Arc Pro B60/B70 use the same XPU execution path,
but support for each SKU, host and precision must be checked against the current
vendor driver matrix and a fresh SparseLab doctor/stage; this guide does not
record hardware acceptance for those cards. The recipe selects device index 0
and requires a discrete Intel Arc A-series or B-series SKU in the device name;
integrated `Intel(R) Arc(TM) Graphics` does not match. Check
`torch.xpu.get_device_name(0)` in the XPU interpreter before provisioning. On
multi-GPU machines, register a separate explicit profile with `--device-index`
for the desired GPU.

## Reproducing the dependency lock

Use the repository's uv version and Python 3.14. Export common application
constraints from `uv.lock`, excluding Torch, then resolve the vendor recipe for
Linux x86_64. This does not install a runtime or exercise a device:

```sh
uv export --locked --extra cpu --no-dev --no-hashes --no-emit-project \
  --no-emit-package torch -o /tmp/lab-common.txt
uv pip compile pyproject.toml requirements/xpu-arc.txt container/build-requirements.txt \
  --constraint /tmp/lab-common.txt --python-version 3.14 \
  --python-platform x86_64-manylinux_2_28 --generate-hashes \
  --no-header --no-annotate --emit-index-url --emit-find-links \
  -o container/backends/xpu/requirements.lock
```

For find-links entries where uv omits hashes, retain the SHA-256 published by
the official package index for the selected CPython 3.14 Linux x86_64 wheel;
never leave an unhashed pin. `container/image_plan.py plan xpu` validates all
lock entries and their shared application versions. The lock includes Torch
2.15.0+xpu, Triton XPU 3.9.0, Intel compiler/SYCL runtime 2026.1.2, oneCCL
2022.1.2, PTI 1.1.0, and pyzes 0.1.3. These are Python dependency changes;
the separately pinned driver userspace is unchanged and requires native testing.

## Host prerequisites

Start with the [Intel GPU driver installation guide](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)
linked by PyTorch. Install the appropriate kernel driver, firmware and Intel
Level Zero user-space driver for the actual distro/card. Follow the vendor's
WSL-specific instructions when applicable. SparseLab provisions Python packages;
it does not modify kernel drivers, boot settings, permissions or the OS stack.
Do not replace an existing AMD/NVIDIA stack to install Intel support.

Check the host before downloading framework packages:

```sh
lspci -Dnnk
ls -l /dev/dri
id
sparselab runtime env discover --json
```

An Intel PCI display device is only a passive observation. Executing on native
Linux also needs an accessible DRM render node (typically
`/dev/dri/renderD128`) and a working Level Zero driver. Use the distro's `render`
(and, where needed, `video`) group policy, then start a new login session after
membership changes. In a container or restricted execution environment, expose
the intended render device and the corresponding group permissions. Do not use
world-writable device permissions. PCI visibility without `/dev/dri` access
cannot establish an executable GPU. Discovery reports hardware separately from
Python capability and never installs a framework or authorizes training.

## Provision and check

Run the manager from the ordinary CPU development environment. It creates a
separate vendor environment, preserves the checkout's CPU Torch, records
package/device/source identity and registers only after an FP32 optimizer pilot
succeeds. Existing ROCm runtimes and Colab/CUDA bootstrap continue to use their
own recipes and runtime IDs.

```sh
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
export SPARSELAB_RUNTIME_DIR="$HOME/.local/share/sparselab/runtimes"
sparselab runtime env provision intel-arc \
  --recipe xpu-arc-v1 --json
sparselab runtime env doctor intel-arc \
  --precision fp32 --json
sparselab runtime env show intel-arc --json
```

Provision uses an existing Python 3.14 executable; supply
`--python /absolute/path/to/python3.14` if necessary. The standard provisioner
requires 20 GiB free and 100,000 free inodes on a writable local filesystem,
outside Git and distinct from the scientific work root. It preserves failures
and logs without registering a failed environment; choose a fresh ID after
fixing a failed installation. See [runtime environments](runtime.md#machine-local-runtime-environments)
for receipts, roots and source repair.

To use an already provisioned XPU interpreter instead:

```sh
sparselab runtime env register intel-existing \
  --python /absolute/xpu-env/bin/python --backend xpu \
  --device-index 0 --device-name-regex '(?i)Intel.*Arc.*\b(?:A\d{3}|B\d{2,3})\b' --json
```

Use FP32 first on the A380. BF16 is optional and needs its own check:

```sh
sparselab runtime env doctor intel-arc \
  --precision bf16 --json
```

PyTorch documents that GradScaler training requires FP64 support and flags its
absence on Arc A-series. SparseLab's FP16 training requires an enabled scaler;
it does not silently disable scaling. Prefer FP32 or validated BF16 on the A380.
Neither discovery's precision declarations nor a different Intel card establish
that FP16 works. A failed precision pilot is an error, never CPU fallback.

## Run the native lab loop

Prepare the [TinyStories microlab inputs](tinystories-microlab.md), then author a
new baseline with this runtime section. Retain explicit tokenizer/data paths
and scientific settings; changing precision should be a visible declaration.

```yaml
runtime:
  engine: pytorch
  backend: xpu
  device_index: 0
  precision: fp32
```

Use the registered interpreter for the baseline and candidate:

```sh
sparselab try DELTA.yaml \
  --vs /absolute/intel-baseline.yaml --runtime intel-arc
sparselab report TRY_ID
sparselab probe TRY_ID \
  --backend xpu --runtime intel-arc --config standard --json
```

The CPU environment here launches the native handoff into `intel-arc`; its CPU
setup does not install into the selected vendor environment. When invoking that
vendor interpreter directly, use its `bin/sparselab`, or activate that environment
and run `sparselab`. Never synchronize `--extra cpu` into it.
Explicit `xpu` requests fail if unavailable. CPU evaluation of a saved XPU run
remains explicit via `--backend cpu` and a CPU runtime.

Start with micro-batch 1 and short sequences on the A380. Increase accumulation
explicitly when a larger effective batch is intended. Use the native resource
envelope and full-shape warmup before larger workloads. Free/total memory comes
from PyTorch XPU APIs when available; unknown telemetry stays unknown. Native
attention, training, checkpoint/resume, probes and the explorer share the existing
PyTorch engine; CUDA/HIP-specific sparse kernels remain device-specific and are
not Intel kernels.

## Software and hardware acceptance

[`runtime_smoke_xpu.yaml`](../configs/runtime_smoke_xpu.yaml) and
[`runtime_smoke_xpu_bf16.yaml`](../configs/runtime_smoke_xpu_bf16.yaml) retain the
ROCm smoke workload: 20 steps, 640 supervised targets, block recomputation,
checkpointing and held-out evaluation. Follow the
[runtime acceptance commands](runtime.md#runtime-acceptance-on-a-provisioned-host)
with `RUNTIME=intel-arc` and `CONFIG=configs/runtime_smoke_xpu.yaml` (BF16 only
after its doctor passes). These configs retain the existing explicit smoke
input paths; author external paths when persistent inputs are desired.

After CPU tests finish, run Intel hardware tests serially in the provisioned
interpreter with pytest available. `-o addopts=` drops the repository's
parallel default, which needs pytest-xdist, so one process owns the device:

```sh
/absolute/xpu-env/bin/python -m pytest -q -o addopts= -m 'xpu and not network'
```

The opt-in training test exercises native XPU forward/backward, recomputation,
checkpoint/resume state equality and checkpoint-bound held-out evidence. CPU
contract tests cover isolated provisioning, failure retention, PCI filtering,
selected-device memory queries and diagnostic RNG preservation. CPU tests do
not establish GPU throughput, driver compatibility or numerical acceptance.
Real Arc execution and exact full-versus-resumed checkpoint equality remain
unverified until the FP32 doctor JSON and this hardware test output are retained
from a provisioned card. The equality check requires deterministic XPU kernels
for the repeated training sequence.
