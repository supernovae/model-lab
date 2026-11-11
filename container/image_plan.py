"""Content-addressed accelerator dependencies; no model or GPU execution."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import tomllib
import uuid
from pathlib import Path

BACKENDS = ("cuda", "rocm", "xpu")
INPUTS = (
    "container/dependencies.Dockerfile",
    "container/image_plan.py",
    "container/build-requirements.txt",
    ".dockerignore",
)


def config(path):
    value = json.loads(path.read_text())
    if set(value) != {"backend", "torch_version", "runtime_version", "platform"}:
        raise ValueError("Unexpected backend config fields")
    if value["backend"] not in BACKENDS or value["platform"] != "linux/amd64":
        raise ValueError("Unsupported backend/platform")
    for name in ("torch_version", "runtime_version"):
        if not isinstance(value[name], str) or not re.fullmatch(
            r"[0-9][a-zA-Z0-9.+_-]*", value[name]
        ):
            raise ValueError(f"Invalid {name}")
    return value


def validate_lock(path):
    """Accept only exact hashed pins and public index declarations, not pip directives."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Regular lock required: {path}")
    text = path.read_text().replace("\\\n", " ")
    count = 0
    allowed_indexes = {
        "--index-url https://pypi.org/simple",
        "--find-links https://download.pytorch.org/whl/cu132/torch/",
        "--find-links https://download.pytorch.org/whl/xpu/torch/",
        "--find-links https://download.pytorch.org/whl/xpu/triton-xpu/",
    }
    allowed_indexes.update(
        "--find-links https://download.pytorch.org/whl/rocm10.0/" + name + "/"
        for name in (
            "torch",
            "triton-rocm",
            "rocm",
            "rocm-sdk-core",
            "rocm-sdk-device-gfx1010",
            "rocm-sdk-device-gfx1011",
            "rocm-sdk-device-gfx1012",
            "rocm-sdk-device-gfx1030",
            "rocm-sdk-device-gfx1031",
            "rocm-sdk-device-gfx1032",
            "rocm-sdk-device-gfx1033",
            "rocm-sdk-device-gfx1034",
            "rocm-sdk-device-gfx1035",
            "rocm-sdk-device-gfx1036",
            "rocm-sdk-device-gfx1100",
            "rocm-sdk-device-gfx1101",
            "rocm-sdk-device-gfx1102",
            "rocm-sdk-device-gfx1103",
            "rocm-sdk-device-gfx1150",
            "rocm-sdk-device-gfx1151",
            "rocm-sdk-device-gfx1152",
            "rocm-sdk-device-gfx1153",
            "rocm-sdk-device-gfx1200",
            "rocm-sdk-device-gfx1201",
            "rocm-sdk-device-gfx1250",
            "rocm-sdk-device-gfx908",
            "rocm-sdk-device-gfx90a",
            "rocm-sdk-device-gfx942",
            "rocm-sdk-device-gfx950",
            "rocm-sdk-libraries",
        )
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line in allowed_indexes:
            continue
        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+(?:\s+--hash=sha256:[0-9a-f]{64})+", line
        ):
            raise ValueError(f"Unpinned or unsupported lock entry: {line[:100]}")
        count += 1
    if not count:
        raise ValueError("Empty dependency lock")


def validate_application_pins(root, path):
    """Keep shared packages on tested uv.lock versions; replace only the backend."""
    locked = tomllib.loads((root / "uv.lock").read_text())
    common = {}
    for package in locked["package"]:
        name = package["name"].lower().replace("_", "-")
        common.setdefault(name, set()).add(package["version"])
    for name, version in re.findall(
        r"^([A-Za-z0-9_.-]+)==([^\s]+)", path.read_text(), re.MULTILINE
    ):
        name = name.lower().replace("_", "-")
        if name in {"torch", "triton", "triton-xpu"}:
            continue
        if name in common and version not in common[name]:
            raise ValueError(
                f"Application dependency differs from uv.lock: {name}=={version}"
            )


def validate_rocm_pins(root):
    """Require the full declared SDK closure in the hashed binary/source locks."""
    recipe = root / "requirements/rocm-gfx1100.txt"
    binary = root / "container/backends/rocm/requirements.lock"
    source = root / "container/backends/rocm/source-requirements.lock"
    pinned = dict(
        re.findall(
            r"^([A-Za-z0-9_.-]+)==([^\s]+)",
            binary.read_text() + source.read_text(),
            re.MULTILINE,
        )
    )
    for name, version in re.findall(
        r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s]+)",
        recipe.read_text(),
        re.MULTILINE,
    ):
        if pinned.get(name) != version:
            raise ValueError(
                f"ROCm recipe dependency missing or mismatched: {name}=={version}"
            )
    expected = config(root / "container/backends/rocm/config.json")["torch_version"]
    if pinned.get("torch") != expected:
        raise ValueError("ROCm framework lock differs from image config")


def dependency_recipe(root, backend):
    custom = f"container/backends/{backend}/dependencies.Dockerfile"
    return custom if (root / custom).exists() else "container/dependencies.Dockerfile"


def identity(root, backend):
    if backend not in BACKENDS:
        raise ValueError("Unsupported backend")
    directory = f"container/backends/{backend}"
    value = config(root / directory / "config.json")
    if value["backend"] != backend:
        raise ValueError("Backend mismatch")
    digest = hashlib.sha256()
    backend_inputs = []
    for path in sorted((root / directory).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink input refused: {path}")
        if path.is_file():
            backend_inputs.append(path.relative_to(root).as_posix())
    if backend == "rocm":
        backend_inputs.append("requirements/rocm-gfx1100.txt")
    for name in (*INPUTS, *backend_inputs):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Regular input required: {name}")
        data = path.read_bytes()
        digest.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    validate_lock(root / directory / "requirements.lock")
    validate_application_pins(root, root / directory / "requirements.lock")
    source_lock = root / directory / "source-requirements.lock"
    if backend == "rocm" or source_lock.exists():
        validate_lock(source_lock)
    if backend == "rocm":
        validate_rocm_pins(root)
    validate_lock(root / "container/build-requirements.txt")
    return digest.hexdigest()


def run(argv, **kwargs):
    kwargs.setdefault(
        "timeout", 3000 if argv[:3] == ["docker", "buildx", "build"] else 600
    )
    return subprocess.run(argv, check=True, text=True, **kwargs)


def registry_digest(reference):
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            reference,
            "--format",
            "{{json .Manifest}}",
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        # Authentication, transport and permissions failures must stop publication.
        missing_reference = re.search(
            r"(?:^|\n)(?:ERROR:\s*)?"
            + re.escape(reference)
            + r": not found\s*(?:\n|$)",
            result.stderr,
        )
        manifest_unknown = re.search(
            r"\bmanifest unknown\b", result.stderr, re.IGNORECASE
        )
        access_failure = re.search(
            r"unauthorized|denied|forbidden|authentication|timeout|connection|TLS",
            result.stderr,
            re.IGNORECASE,
        )
        if (missing_reference or manifest_unknown) and not access_failure:
            return None
        raise RuntimeError(f"Registry lookup failed: {result.stderr}")
    digest = json.loads(result.stdout)["digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Registry did not return a content digest")
    return digest


def verify(path):
    import torch

    value = config(path)
    if importlib.metadata.version("torch") != value["torch_version"]:
        raise ValueError("Torch version mismatch")
    backend = value["backend"]
    if backend == "cuda":
        valid = (
            torch.version.cuda == value["runtime_version"] and torch.version.hip is None
        )
    elif backend == "rocm":
        valid = (
            torch.version.hip == value["runtime_version"] and torch.version.cuda is None
        )
    else:
        valid = (
            torch.xpu._is_compiled()
            and torch.version.cuda is None
            and torch.version.hip is None
        )
    if not valid:
        raise ValueError("Wrong framework build")
    assert (torch.ones(2, device="cpu") + 1).tolist() == [2, 2]
    print(
        json.dumps(
            {"config": value, "torch": torch.__version__, "hardware_tested": False}
        )
    )


def publish(root, backend, repository, revision):
    if not re.fullmatch(r"ghcr.io/supernovae/model-lab", repository):
        raise ValueError("Only the public Lab package is supported")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An exact source commit is required")
    key = identity(root, backend)
    tag = f"{repository}:deps-{backend}-{key}"
    digest = registry_digest(tag)
    if digest is None:
        run(
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/amd64",
                "--load",
                "-f",
                dependency_recipe(root, backend),
                "--build-arg",
                f"BACKEND={backend}",
                "-t",
                tag,
                ".",
            ],
            cwd=root,
        )
        # Dockerfile verifies exact framework build before this immutable tag is published.
        run(["docker", "push", tag])
        digest = registry_digest(tag)
        if digest is None:
            raise RuntimeError("Published dependency digest unavailable")
    base = f"{repository}@{digest}"
    app = f"{repository}:{backend}-sha-{revision}"
    run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--load",
            "-f",
            "container/application.Dockerfile",
            "--build-arg",
            f"DEPENDENCY_IMAGE={base}",
            "--build-arg",
            f"SOURCE_REVISION={revision}",
            "-t",
            app,
            ".",
        ],
        cwd=root,
    )
    for command in (
        ["python", "/opt/image_plan.py", "verify", "/opt/backend.json"],
        ["klab", "--help"],
    ):
        name = "lab-image-smoke-" + uuid.uuid4().hex
        try:
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    name,
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--memory=4g",
                    "--cpus=2",
                    "--pids-limit=256",
                    "--tmpfs=/state:rw,uid=10001,gid=10001,size=67108864",
                    app,
                    *command,
                ],
                timeout=120,
            )
        finally:
            subprocess.run(
                ["docker", "rm", "--force", name],
                check=False,
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    # Push the same locally tested image, never a second build.
    run(["docker", "push", app])
    app_digest = registry_digest(app)
    if app_digest is None:
        raise RuntimeError("Published application digest unavailable")
    result = {
        "backend": backend,
        "dependency_key": key,
        "dependency": base,
        "application": f"{repository}@{app_digest}",
        "revision": revision,
    }
    Path(f"{backend}-image.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(
                f"{backend}: `{result['application']}`\n\nDependencies: `{base}`\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify")
    p.add_argument("config", type=Path)
    p = sub.add_parser("plan")
    p.add_argument("backend", choices=BACKENDS)
    p = sub.add_parser("publish")
    p.add_argument("backend", choices=BACKENDS)
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "verify":
        verify(args.config)
    elif args.command == "plan":
        print(identity(root, args.backend))
    else:
        publish(
            root,
            args.backend,
            "ghcr.io/supernovae/model-lab",
            os.environ["GITHUB_SHA"],
        )


if __name__ == "__main__":
    main()
