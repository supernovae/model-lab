"""Bounded ROCm packaging fixtures: no wheels, network, framework or GPU."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rocm_image_plan", ROOT / "image_plan.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)
REPO = ROOT.parent
BACKEND = REPO / "container/backends/rocm"


class CpuTensor:
    def __add__(self, value):
        assert value == 1
        return self

    def tolist(self):
        return [2, 2]


def cpu_ones(size, device):
    assert (size, device) == (2, "cpu")
    return CpuTensor()


class RocmRecipeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in (*PLAN.INPUTS, "uv.lock", "requirements/rocm-gfx1100.txt"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        self.backend = self.root / "container/backends/rocm"
        shutil.copytree(
            BACKEND, self.backend, ignore=shutil.ignore_patterns("__pycache__")
        )

    def test_required_source_lock_missing_refuses(self):
        (self.backend / "source-requirements.lock").unlink()
        with self.assertRaises(ValueError):
            PLAN.identity(self.root, "rocm")

    def test_source_hash_missing_refuses(self):
        (self.backend / "source-requirements.lock").write_text("rocm==10.1.0\n")
        with self.assertRaises(ValueError):
            PLAN.identity(self.root, "rocm")

    def test_unlisted_amd_source_refuses(self):
        path = self.backend / "requirements.lock"
        path.write_text(
            "--find-links https://stable.repo.amd.com/rocm/whl-next/unlisted/\n"
            + path.read_text()
        )
        with self.assertRaises(ValueError):
            PLAN.identity(self.root, "rocm")

    def test_approved_lock_and_recipe_validate(self):
        self.assertEqual(len(PLAN.identity(self.root, "rocm")), 64)
        recipe = (self.backend / "dependencies.Dockerfile").read_text()
        for required in (
            "ROCM_SDK_TARGET_FAMILY=gfx1100",
            "UV_LINK_MODE=hardlink",
            "--no-build-isolation --require-hashes -r /opt/source-requirements.lock",
            "uv pip check --python /opt/venv/bin/python",
            "python /opt/image_plan.py verify /opt/backend.json",
            "uv cache clean",
        ):
            self.assertIn(required, recipe)

    def test_missing_device_all_dependency_refuses(self):
        path = self.backend / "requirements.lock"
        lines = path.read_text().splitlines(keepends=True)
        start = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("rocm-sdk-device-gfx950==")
        )
        end = start + 1
        while end < len(lines) and lines[end].startswith(" "):
            end += 1
        path.write_text("".join(lines[:start] + lines[end:]))
        with self.assertRaisesRegex(ValueError, "dependency missing or mismatched"):
            PLAN.identity(self.root, "rocm")

    def test_framework_config_mismatch_refuses(self):
        path = self.backend / "config.json"
        value = json.loads(path.read_text())
        value["torch_version"] = "2.13.0+rocm10.0.0"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "framework lock differs"):
            PLAN.identity(self.root, "rocm")

    def test_verify_requires_exact_framework_build(self):
        path = self.backend / "config.json"
        expected = json.loads(path.read_text())
        for hip, cuda, version, valid in (
            ("7.15.26333", None, expected["torch_version"], True),
            ("10.1.0", None, expected["torch_version"], False),
            ("7.15.26333", "13.2", expected["torch_version"], False),
            (None, None, expected["torch_version"], False),
            ("7.15.26333", None, "2.15.0+cpu", False),
        ):
            fake = SimpleNamespace(
                __version__=version,
                version=SimpleNamespace(hip=hip, cuda=cuda),
                ones=cpu_ones,
            )
            with (
                self.subTest(hip=hip, cuda=cuda, version=version),
                patch.dict(sys.modules, {"torch": fake}),
                patch.object(PLAN.importlib.metadata, "version", return_value=version),
            ):
                if valid:
                    PLAN.verify(path)
                else:
                    with self.assertRaises(ValueError):
                        PLAN.verify(path)
