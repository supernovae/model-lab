"""Torch-free operational environment CLI, separate from scientific execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from klab.runtime_env_doctor import doctor, source_repair_command
from klab.runtime_env_inventory import inspect_python, inventory, observed_entry
from klab.runtime_environments import (
    RuntimeEntry,
    profile_for_id,
    read_registry,
    register_runtime,
    unregister_runtime,
    validate_id,
)
from klab.runtime_profile import RuntimeProfile

BACKENDS = ("cpu", "rocm", "cuda", "mps", "xpu", "metal")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="klab runtime env",
        description="Machine-local Python environments; hardware is not execution authorization.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in (
        "list",
        "discover",
        "show",
        "register",
        "unregister",
        "doctor",
        "profile",
        "provision",
    ):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true")
        if name not in {"list", "discover"}:
            command.add_argument("id")
        if name == "register":
            command.add_argument("--python", type=Path, required=True)
            command.add_argument("--backend", choices=BACKENDS)
            command.add_argument("--device-index", type=int, default=0)
            command.add_argument("--device-name-regex")
            command.add_argument("--require-bf16", action="store_true")
        if name == "doctor":
            command.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
        if name == "provision":
            command.add_argument(
                "--recipe",
                choices=["rocm-gfx1100-v1", "cuda-cu126-v1", "xpu-arc-v1"],
                required=True,
            )
            command.add_argument("--python", type=Path)
    return root


def _register(args: argparse.Namespace) -> dict:
    validate_id(args.id)
    if args.id in read_registry().runtimes:
        raise ValueError(f"runtime id already registered: {args.id}")
    if not args.python.is_absolute():
        raise ValueError("python must be an absolute executable path")
    observation = inspect_python(args.python)
    if observation["status"] != "READY":
        repair = (
            f"; repair: {source_repair_command(args.python)}"
            if observation["status"] == "SOURCE_MISMATCH"
            else ""
        )
        raise ValueError(
            f"{observation['status']}: {observation.get('reason')}{repair}"
        )
    backend = args.backend
    available = observation["backends"]
    if backend is None:
        accelerators = [item for item in available if item != "cpu"]
        if len(accelerators) == 1:
            backend = accelerators[0]
        elif available == ["cpu"]:
            backend = "cpu"
        else:
            raise ValueError("ambiguous executable backends; specify --backend")
    if backend not in available:
        raise ValueError(
            f"selected backend {backend} is not executable under this interpreter"
        )
    requirements = {}
    if backend == "rocm":
        requirements["torch_hip"] = True
    if args.require_bf16:
        requirements["bf16"] = True
    if args.device_name_regex is not None:
        requirements["device_name_regex"] = args.device_name_regex
    profile = RuntimeProfile.model_validate(
        {
            "runtime_profile_version": 1,
            "id": args.id,
            "python": args.python,
            "engine": "mlx" if backend == "metal" else "pytorch",
            "backend": backend,
            "device_index": args.device_index,
            "requirements": requirements,
        }
    )
    result = doctor(profile)
    if result["status"] != "READY":
        return result
    entry = RuntimeEntry.model_validate(
        profile.model_dump(exclude={"id", "runtime_profile_version"})
    )
    register_runtime(args.id, entry)
    return result


def execute(args: argparse.Namespace) -> dict:
    if args.command == "discover":
        return inventory()
    registry = read_registry()
    if args.command == "list":
        discovered = inventory(registry=registry)
        return {
            "runtime_registry_version": 1,
            "runtimes": [
                observed_entry(identifier, entry, discovered["environments"])
                for identifier, entry in sorted(registry.runtimes.items())
            ],
            "truncated": discovered["truncated"],
        }
    validate_id(args.id)
    if args.command == "register":
        return _register(args)
    if args.command == "provision":
        from klab.runtime_env_provision import provision

        return provision(args.id, recipe=args.recipe, python=args.python)
    if args.command == "unregister":
        unregister_runtime(args.id)
        return {"runtime_registry_version": 1, "id": args.id, "status": "UNREGISTERED"}
    if args.id not in registry.runtimes:
        raise ValueError(f"unknown runtime id: {args.id}")
    if args.command == "show":
        entry = registry.runtimes[args.id]
        # A specific show is not hidden by inventory's registry candidate cap.
        return observed_entry(args.id, entry, [inspect_python(entry.python)])
    if args.command == "doctor" and not registry.runtimes[args.id].python.is_file():
        return {
            "runtime_env_doctor_version": 1,
            "id": args.id,
            "status": "NOT_PROVISIONED",
            "profile": None,
            "probe": None,
            "tested_runtime": None,
            "reason": "registered interpreter does not exist",
        }
    profile = profile_for_id(args.id)
    if args.command == "profile":
        return profile.model_dump(mode="json")
    precision = getattr(args, "precision", None)
    return (
        doctor(profile) if precision is None else doctor(profile, precision=precision)
    )


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except (ValueError, TypeError, OSError) as error:
        result = {
            "runtime_env_cli_version": 1,
            "id": getattr(args, "id", None),
            "status": "ERROR",
            "reason": str(error),
        }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(yaml.safe_dump(result, sort_keys=False), end="")
    if result.get("status") in {
        "ERROR",
        "UNAVAILABLE",
        "SOURCE_MISMATCH",
        "NOT_PROVISIONED",
    }:
        sys.exit(1)
