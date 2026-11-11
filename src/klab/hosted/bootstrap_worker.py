"""Fixed stdlib-only bootstrap, executed in an isolated remote namespace.

The caller injects the existing native run_bounded implementation. This module
never loads provider credentials or installs into the notebook interpreter.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

run_bounded: Callable[..., subprocess.CompletedProcess] | None = None

_MAX_JSON = 1024 * 1024
_UV_URL = "https://files.pythonhosted.org/packages/e5/83/85a6c63c24905af4924fddb11a499b934913f59a134248367a1ef1a4716f/uv-0.12.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
_UV_SHA = "565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff"
_deadline = 0.0


def remaining() -> float:
    value = _deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("hosted bootstrap deadline exhausted")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def strict_json(raw: bytes):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate bootstrap JSON field")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite bootstrap JSON number")

    if len(raw) > _MAX_JSON:
        raise ValueError("bootstrap JSON exceeds metadata bound")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON:
        raise ValueError("unsafe bootstrap receipt")
    return strict_json(path.read_bytes())


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(command: list[str], *, env=None, cwd: Path | None = None):
    # run_bounded is injected from Model Lab's versioned stdlib-only module.
    timeout = remaining()
    if run_bounded is None:
        raise RuntimeError("versioned bounded subprocess helper was not injected")
    result = run_bounded(
        command, timeout=timeout, output_limit=_MAX_JSON, env=env, cwd=cwd
    )
    if result.returncode:
        raise RuntimeError(
            (
                result.stdout.decode("utf-8", "replace")[:2048]
                + result.stderr.decode("utf-8", "replace")[-2048:]
            )
            or "bootstrap command failed"
        )
    return result.stdout


def closest(path: Path) -> Path:
    while not path.exists():
        if path.parent == path:
            raise ValueError("output has no existing filesystem ancestor")
        path = path.parent
    return path


def local_storage(path: Path) -> dict:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError("hosted output must be an absolute nonsymlink path")
    if path.exists() and (
        path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077
    ):
        raise ValueError("existing task roots must be private and owned")
    ancestor = closest(path)
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise ValueError("hosted filesystem is not locally writable")
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields = before.split()
        mount = fields[4]
        for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            mount = mount.replace(encoded, decoded)
        mount_path = Path(mount)
        if ancestor.is_relative_to(mount_path):
            mounts.append(
                (len(mount_path.parts), mount_path, after.split()[0], fields[5])
            )
    if not mounts:
        raise ValueError("hosted output filesystem cannot be established")
    _, mount, kind, options = max(mounts)
    if kind not in {
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "tmpfs",
        "overlay",
    } or "rw" not in options.split(","):
        raise ValueError(
            "hosted work/runtime roots require local writable storage, not network/Drive/FUSE"
        )
    observation = os.statvfs(ancestor)
    if observation.f_flag & os.ST_RDONLY:
        raise ValueError("hosted filesystem is read-only")
    return {
        "path": str(path),
        "observed_at": str(ancestor),
        "mount": str(mount),
        "filesystem": kind,
        "local_writable": True,
        "device": ancestor.stat().st_dev,
        "free_bytes": observation.f_bavail * observation.f_frsize,
        "free_inodes": observation.f_favail,
        "durability": "VM_LOCAL_EPHEMERAL",
    }


def diagnostic(p: dict) -> dict:
    root = Path(os.path.expanduser(p["root"]))
    answer = {
        "root": str(root),
        "host_os": platform.system(),
        "host_architecture": platform.machine(),
        "python": sys.version,
        "torch": None,
        "cpu": os.cpu_count(),
        "ram_bytes": None,
        "gpu_name": None,
        "compute_capability": None,
        "vram_bytes": None,
        "driver": None,
        "gpu_uuid": None,
        "cuda": None,
        "hip": None,
        "cuda_available": None,
        "nvidia_available": False,
        "free_bytes": None,
        "free_inodes": None,
        "requested_accelerator": None,
        "session_accelerator": None,
        "accelerator_metadata_reason": "Endpoint selection does not declare an accelerator; provider metadata is not actual device evidence",
        "tpu": next(
            (
                os.environ[key]
                for key in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_ACCELERATOR_TYPE")
                if os.environ.get(key)
            ),
            None,
        ),
        "reasons": {},
    }
    try:
        answer["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError) as error:
        answer["reasons"]["ram_bytes"] = str(error)
    try:
        torch = importlib.import_module("torch")
        answer.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            hip=torch.version.hip,
            cuda_available=torch.cuda.is_available(),
        )
        if answer["cuda_available"] and not answer["hip"]:
            properties = torch.cuda.get_device_properties(0)
            answer.update(
                gpu_name=properties.name,
                compute_capability=[properties.major, properties.minor],
                vram_bytes=properties.total_memory,
                nvidia_available=True,
            )
    except Exception as error:  # noqa: BLE001 - opaque vendor imports remain negative evidence
        answer["reasons"]["framework"] = f"{type(error).__name__}: {error}"
    try:
        raw = run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,driver_version,memory.total,compute_cap,uuid",
                "--format=csv,noheader,nounits",
            ]
        )
        fields = raw.decode().strip().split(",")
        if len(fields) != 5:
            raise ValueError("unexpected NVIDIA observation fields")
        capability = [int(value) for value in fields[3].strip().split(".")]
        answer.update(driver=fields[1].strip(), nvidia_available=True)
        if answer["gpu_name"] is None:
            answer.update(
                gpu_name=fields[0].strip(),
                vram_bytes=int(float(fields[2])) * 1024**2,
                compute_capability=capability,
                gpu_uuid=fields[4].strip(),
            )
        else:
            selected_uuid = getattr(properties, "uuid", None)
            answer["gpu_uuid"] = (
                str(selected_uuid) if selected_uuid is not None else None
            )
            if selected_uuid is None:
                answer["reasons"]["gpu_uuid"] = (
                    "Framework-selected device UUID is unavailable; physical index zero was not substituted"
                )
    except (OSError, ValueError, RuntimeError) as error:
        answer["reasons"]["driver_and_uuid"] = f"{type(error).__name__}: {error}"
    try:
        storage = os.statvfs(closest(root))
        answer.update(
            free_bytes=storage.f_bavail * storage.f_frsize, free_inodes=storage.f_favail
        )
    except (OSError, ValueError) as error:
        answer["reasons"]["storage"] = str(error)
    return answer


def binding_request(p: dict) -> dict:
    return {
        key: p[key]
        for key in ("root", "runtime_root", "source_commit", "source_sha256", "recipe")
    }


def enrollment(p: dict) -> dict:
    root = Path(p["root"])
    if platform.system() != "Linux":
        raise ValueError("Colab enrollment requires Linux")
    local_storage(root)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    worker = root / "work" / "worker"
    marker = root / "hosted-enrollment.json"
    binding = worker / "hosted-instance.json"
    if binding.exists():
        saved = read_json(binding)
        if set(saved) != {"instance_id", "boot_id"}:
            raise ValueError("invalid existing hosted instance binding")
    elif marker.exists():
        saved = read_json(marker)
        if (
            set(saved) != {"enrollment_version", "instance_id", "boot_id"}
            or type(saved["enrollment_version"]) is not int
            or saved["enrollment_version"] != 1
        ):
            raise ValueError("invalid existing Colab enrollment")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("enrollment requires a fresh private task root")
        saved = {
            "enrollment_version": 1,
            "instance_id": os.urandom(32).hex(),
            "boot_id": boot,
        }
    if (
        saved["boot_id"] != boot
        or not isinstance(saved["instance_id"], str)
        or len(saved["instance_id"]) != 64
        or any(char not in "0123456789abcdef" for char in saved["instance_id"])
    ):
        raise ValueError("Colab enrollment belongs to another VM or invalid nonce")
    for path in (root, root / "work", worker):
        if path.exists() and (
            path.is_symlink()
            or path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077
        ):
            raise ValueError("Colab enrollment paths must be private and owned")
    for path in (root, root / "work", worker):
        path.mkdir(mode=0o700, exist_ok=True)
    if not marker.exists() and not binding.exists():
        atomic_json(marker, saved)
    return {
        "worker_root": str(worker),
        "instance_id": saved["instance_id"],
        "boot_id": boot,
    }


def fresh_root(root: Path, p: dict) -> None:
    if not root.exists() or not any(root.iterdir()):
        return
    saved = read_json(root / "hosted-enrollment.json")
    if (
        set(saved) != {"enrollment_version", "instance_id", "boot_id"}
        or type(saved["enrollment_version"]) is not int
        or saved["enrollment_version"] != 1
        or saved["instance_id"] != p.get("instance_id")
        or saved["boot_id"]
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise ValueError("existing root is not the verified idle Colab enrollment")
    allowed = {
        "hosted-enrollment.json",
        "work",
        "work/worker",
        "work/worker/kernel-status.json",
    }
    pending = [root]
    count = 0
    while pending:
        for item in pending.pop().iterdir():
            count += 1
            if (
                count > len(allowed)
                or item.is_symlink()
                or item.relative_to(root).as_posix() not in allowed
                or item.stat().st_uid != os.getuid()
                or item.stat().st_mode & 0o077
            ):
                raise ValueError(
                    "partial hosted setup cannot be overwritten; choose a fresh root"
                )
            if item.is_dir():
                pending.append(item)


def verified_existing(p: dict):
    root = Path(p["root"])
    marker = root / "hosted-setup.json"
    if not marker.exists():
        fresh_root(root, p)
        return None
    saved = read_json(marker)
    if (
        saved.get("request") != binding_request(p)
        or saved.get("boot_id")
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise ValueError("existing hosted root conflicts with declaration or live VM")
    result = saved["result"]
    if (
        p.get("instance_id") is not None
        and result.get("instance_id") != p["instance_id"]
    ):
        raise ValueError("idle observer differs from completed hosted instance")
    if result.get("status") != "READY":
        raise ValueError("existing hosted setup is incomplete")
    binding = read_json(Path(result["worker_root"]) / "hosted-instance.json")
    if set(binding) != {"boot_id", "instance_id"} or binding != {
        "boot_id": saved["boot_id"],
        "instance_id": result["instance_id"],
    }:
        raise ValueError("existing hosted instance binding conflicts")
    source = Path(result["source_root"])
    if run(["git", "-C", str(source), "rev-parse", "HEAD"]).decode().strip() != p[
        "source_commit"
    ] or run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"]
    ):
        raise ValueError("existing source checkout is not the declared clean commit")
    receipt = read_json(Path(result["environment_path"]) / "provision-receipt.json")
    if receipt != result["receipt"] or receipt.get("recipe_sha256") != digest(
        source / "requirements" / "cuda-cu126.txt"
    ):
        raise ValueError("existing provision receipt or recipe content conflicts")
    env = os.environ.copy()
    env["KLAB_RUNTIME_DIR"] = p["runtime_root"]
    doctor = strict_json(
        run(
            [
                str(Path(result["environment_path"]) / "bin" / "klab"),
                "runtime",
                "env",
                "doctor",
                result["runtime_id"],
                "--precision",
                "fp16",
                "--json",
            ],
            env=env,
        )
    )
    old, new = receipt["tested_runtime"], doctor.get("tested_runtime", {})
    identity = (
        "physical_device_id",
        "device_name",
        "framework_version",
        "runtime_version",
        "driver_version",
    )
    if (
        doctor.get("status") != "READY"
        or old.get("physical_device_id") is None
        or any(old.get(key) != new.get(key) for key in identity)
    ):
        raise ValueError(
            "existing CUDA environment does not verify against its live device/build"
        )
    if doctor.get("probe", {}).get("source_sha256") != receipt.get(
        "source_sha256"
    ) or doctor.get("profile") != receipt.get("profile"):
        raise ValueError(
            "existing runtime source or profile differs from its provision receipt"
        )
    return result


def preflight(p: dict) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("hosted CUDA bootstrap requires Linux x86_64")
    root, runtime = Path(p["root"]), Path(p["runtime_root"])
    observations = [local_storage(path) for path in (root, runtime)]
    saved = verified_existing(p)
    if saved is not None:
        return {"reused": True, "result": saved, "storage": observations}
    if runtime.exists() and any(runtime.iterdir()):
        raise ValueError("existing runtime root lacks matching complete hosted receipt")
    for path in (root, runtime):
        if path.exists() and (
            path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077
        ):
            raise ValueError("existing task root must be private and owned")
    groups = {}
    for observation, size, inodes in zip(
        observations, (p["source_bytes"] * 3 + 4 * 1024**3, 16 * 1024**3), (5000, 95000)
    ):
        group = groups.setdefault(
            observation["device"], {"bytes": 0, "inodes": 0, "observation": observation}
        )
        group["bytes"] += size
        group["inodes"] += inodes
    for group in groups.values():
        if (
            group["observation"]["free_bytes"] < group["bytes"]
            or group["observation"]["free_inodes"] < group["inodes"]
        ):
            raise ValueError(
                "insufficient local capacity for source, transfer, runtime and checkpoints"
            )
    # All refusal gates run before creating even a directory.
    for path in (
        root,
        runtime,
        root / "source",
        root / "work",
        root / "work" / "worker",
        root / "work" / "worker" / ".colab-control",
        root / "work" / "worker" / ".colab-control" / "rpc",
        root / "tools",
        root / "scratch",
    ):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {"reused": False, "storage": observations}


def setup(p: dict) -> dict:
    root, runtime = Path(p["root"]), Path(p["runtime_root"])
    bundle = Path(p["source_bundle"])
    if (
        bundle.is_symlink()
        or not bundle.is_file()
        or digest(bundle) != p["source_sha256"]
    ):
        raise ValueError("source bundle digest verification failed")
    source = root / "source" / p["source_commit"]
    if source.exists() or (root / "work" / "worker" / "hosted-instance.json").exists():
        raise ValueError(
            "partial hosted setup cannot be overwritten; choose a fresh root"
        )
    verifier = root / "scratch" / "bundle-verify.git"
    run(["git", "init", "--bare", str(verifier)])
    run(["git", "-C", str(verifier), "bundle", "verify", str(bundle)])
    run(["git", "clone", str(bundle), str(source)])
    run(["git", "-C", str(source), "checkout", "--detach", p["source_commit"]])
    if (
        run(["git", "-C", str(source), "rev-parse", "HEAD"]).decode().strip()
        != p["source_commit"]
    ):
        raise ValueError("source commit verification failed")
    uv = shutil.which("uv")
    if uv is None:
        wheel = root / "scratch" / "uv-0.12.23.whl"
        with (
            urllib.request.urlopen(_UV_URL, timeout=min(60, remaining())) as response,
            wheel.open("xb") as output,
        ):
            total = 0
            while chunk := response.read(1024 * 1024):
                remaining()
                total += len(chunk)
                if total > 64 * 1024**2:
                    raise ValueError("uv download exceeds bound")
                output.write(chunk)
        if digest(wheel) != _UV_SHA:
            raise ValueError("downloaded uv wheel digest mismatch")
        uv_path = root / "tools" / "bin" / "uv"
        uv_path.parent.mkdir(mode=0o700)
        with zipfile.ZipFile(wheel) as archive:
            member = archive.getinfo("uv/uv")
            if (
                member.is_dir()
                or member.file_size > 64 * 1024**2
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("unsafe uv executable member")
            with archive.open(member) as incoming, uv_path.open("xb") as output:
                shutil.copyfileobj(incoming, output, 1024 * 1024)
        uv_path.chmod(0o700)
        uv = str(uv_path)
    env = os.environ.copy()
    env.update(
        PATH=str(Path(uv).parent) + os.pathsep + env.get("PATH", ""),
        KLAB_RUNTIME_DIR=str(runtime),
        UV_PROJECT_ENVIRONMENT=str(root / "tools" / "bootstrap-env"),
        UV_PYTHON_INSTALL_DIR=str(root / "tools" / "python"),
        UV_PYTHON_BIN_DIR=str(root / "tools" / "bin"),
        UV_CACHE_DIR=str(root / "scratch" / "uv-cache"),
    )
    run([uv, "python", "install", "3.14"], env=env)
    managed = (
        run([uv, "python", "find", "--managed-python", "3.14"], env=env)
        .decode()
        .strip()
    )
    if not Path(managed).is_absolute() or not Path(managed).resolve().is_relative_to(
        root / "tools" / "python"
    ):
        raise ValueError("uv did not select a task-owned absolute managed Python")
    native = strict_json(
        run(
            [
                uv,
                "run",
                "--python",
                managed,
                "--locked",
                "--no-default-groups",
                "klab",
                "runtime",
                "env",
                "provision",
                "hosted-cuda",
                "--recipe",
                p["recipe"],
                "--python",
                managed,
                "--json",
            ],
            env=env,
            cwd=source,
        )
    )
    if native.get("status") != "READY":
        raise ValueError("native CUDA provision failed: " + str(native.get("reason")))
    receipt = read_json(Path(native["receipt_path"]))
    if (
        receipt.get("status") != "READY"
        or receipt.get("source_commit") != p["source_commit"]
        or receipt.get("recipe") != p["recipe"]
        or receipt.get("recipe_sha256")
        != digest(source / "requirements" / "cuda-cu126.txt")
    ):
        raise ValueError("native provision receipt conflicts with fixed declaration")
    worker = root / "work" / "worker"
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    nonce = p["instance_id"]
    atomic_json(
        worker / "hosted-instance.json", {"instance_id": nonce, "boot_id": boot}
    )
    result = {
        "hosted_setup_version": 1,
        "status": "READY",
        "source_root": str(source),
        "worker_root": str(worker),
        "runtime_id": receipt["id"],
        "python": receipt["profile"]["python"],
        "environment_path": receipt["environment_path"],
        "instance_id": nonce,
        "receipt": receipt,
    }
    atomic_json(
        root / "hosted-setup.json",
        {"request": binding_request(p), "boot_id": boot, "result": result},
    )
    bundle.unlink()
    return result


def dispatch(action: str, p: dict):
    global _deadline
    _deadline = time.monotonic() + p.pop("timeout")
    return {
        "inspect": diagnostic,
        "preflight": preflight,
        "setup": setup,
        "enroll": enrollment,
    }[action](p)
