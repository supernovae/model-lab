"""Lightweight pipeline contracts; no registry/network, wheels or containers."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import image_plan


class ImagePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in image_plan.INPUTS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                name
                if not name.endswith("build-requirements.txt")
                else "uv-build==0.12.17 --hash=sha256:" + "a" * 64
            )
        (self.root / "uv.lock").write_text(
            '[[package]]\nname="huggingface-hub"\nversion="0.36.2"\n'
            '[[package]]\nname="torch"\nversion="2.14.0+cpu"\n'
        )
        self.backend = self.root / "container/backends/cuda"
        self.backend.mkdir(parents=True)
        (self.backend / "config.json").write_text(
            json.dumps(
                {
                    "backend": "cuda",
                    "torch_version": "2.14.1+cu132",
                    "runtime_version": "13.2",
                    "platform": "linux/amd64",
                }
            )
        )
        (self.backend / "requirements.lock").write_text(
            "torch==2.14.1+cu132 --hash=sha256:" + "a" * 64
        )

    def test_publication_rejects_other_repositories_before_commands(self):
        for repository in (
            "ghcr.io/supernovae/lab-harness",
            "ghcr.io/someone/model-lab",
            "ghcr.io/supernovae/model-lab-extra",
        ):
            with (
                self.subTest(repository=repository),
                patch(
                    "image_plan.registry_digest", return_value="sha256:" + "a" * 64
                ) as lookup,
                patch("image_plan.run") as run,
                patch("image_plan.subprocess.run"),
                patch("image_plan.Path.write_text"),
            ):
                with self.assertRaises(ValueError):
                    image_plan.publish(self.root, "cuda", repository, "a" * 40)
                lookup.assert_not_called()
                run.assert_not_called()

    def test_publication_rejects_nonexact_revisions_before_commands(self):
        for revision in ("a" * 39, "a" * 41, "g" * 40, "A" * 40):
            with (
                self.subTest(revision=revision),
                patch(
                    "image_plan.registry_digest", return_value="sha256:" + "a" * 64
                ) as lookup,
                patch("image_plan.run") as run,
                patch("image_plan.subprocess.run"),
                patch("image_plan.Path.write_text"),
            ):
                with self.assertRaises(ValueError):
                    image_plan.publish(
                        self.root,
                        "cuda",
                        "ghcr.io/supernovae/model-lab",
                        revision,
                    )
                lookup.assert_not_called()
                run.assert_not_called()

    def test_registry_rejects_malformed_digest(self):
        for digest in (
            "",
            "sha256:" + "a" * 63,
            "sha256:" + "g" * 64,
            "sha512:" + "a" * 64,
        ):
            with (
                self.subTest(digest=digest),
                patch(
                    "image_plan.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, json.dumps({"digest": digest}), ""
                    ),
                ),
                self.assertRaises(ValueError),
            ):
                image_plan.registry_digest("example")

    def test_config_rejects_wrong_platform(self):
        path = self.backend / "config.json"
        value = json.loads(path.read_text())
        value["platform"] = "linux/arm64"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_identity_rejects_mismatched_backend(self):
        path = self.backend / "config.json"
        value = json.loads(path.read_text())
        value["backend"] = "xpu"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_config_requires_exact_fields(self):
        path = self.backend / "config.json"
        original = json.loads(path.read_text())
        for value in (
            dict(original, unexpected=True),
            {k: v for k, v in original.items() if k != "runtime_version"},
        ):
            with self.subTest(fields=list(value)):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    image_plan.identity(self.root, "cuda")

    def test_empty_lock_refuses(self):
        (self.backend / "requirements.lock").write_text("\n# No packages\n")
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_backend_directory_symlink_refuses(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "pin.txt").write_text("outside input")
        (self.backend / "linked-inputs").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink input refused"):
            image_plan.identity(self.root, "cuda")

    def test_optional_source_lock_is_validated(self):
        path = self.backend / "source-requirements.lock"
        path.write_text("rocm==10.1.0 --hash=sha256:" + "a" * 64)
        image_plan.identity(self.root, "cuda")
        path.write_text("rocm==10.1.0")
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_missing_postpush_app_digest_prevents_evidence_record(self):
        with (
            patch(
                "image_plan.registry_digest", side_effect=["sha256:" + "a" * 64, None]
            ),
            patch("image_plan.run") as run,
            patch("image_plan.subprocess.run"),
            patch("image_plan.Path.write_text") as write,
        ):
            with self.assertRaisesRegex(RuntimeError, "application digest unavailable"):
                image_plan.publish(
                    self.root, "cuda", "ghcr.io/supernovae/model-lab", "c" * 40
                )
            self.assertEqual(run.call_args_list[-1].args[0][:2], ["docker", "push"])
            write.assert_not_called()

    def test_buildx_missing_exact_reference_allows_bootstrap(self):
        reference = "ghcr.io/supernovae/model-lab:deps-cuda-123"
        with patch(
            "image_plan.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 1, "", "ERROR: " + reference + ": not found\n"
            ),
        ):
            self.assertIsNone(image_plan.registry_digest(reference))
        for message in (
            "not found",
            "ERROR: other-reference: not found",
            "unauthorized\nERROR: " + reference + ": not found",
            "connection reset\nmanifest unknown",
        ):
            with (
                self.subTest(message=message),
                patch(
                    "image_plan.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 1, "", message),
                ),
                self.assertRaises(RuntimeError),
            ):
                image_plan.registry_digest(reference)

    def test_application_pins_match_tested_lock(self):
        path = self.backend / "requirements.lock"
        original = path.read_text()
        path.write_text(
            original + "\nhuggingface-hub==0.36.2 --hash=sha256:" + "b" * 64
        )
        image_plan.identity(self.root, "cuda")
        path.write_text(
            path.read_text().replace(
                "huggingface-hub==0.36.2", "huggingface-hub==1.0.0"
            )
        )
        with self.assertRaisesRegex(ValueError, "Application dependency differs"):
            image_plan.identity(self.root, "cuda")

    def test_unlisted_links_and_range_pins_refuse(self):
        for value in (
            "torch>=2 --hash=sha256:" + "a" * 64,
            "--find-links https://untrusted.example/wheels/\ntorch==2 --hash=sha256:"
            + "a" * 64,
        ):
            with self.subTest(value=value):
                (self.backend / "requirements.lock").write_text(value)
                with self.assertRaises(ValueError):
                    image_plan.identity(self.root, "cuda")

    def test_build_backend_lock_must_be_hashed(self):
        (self.root / "container/build-requirements.txt").write_text("uv-build==0.12.17")
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_shared_input_symlink_refuses(self):
        target = self.root / "outside-script.py"
        target.write_text("outside")
        shared = self.root / "container/image_plan.py"
        shared.unlink()
        shared.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Regular input required"):
            image_plan.identity(self.root, "cuda")

    def test_digest_framing_has_stable_identity(self):
        self.assertEqual(
            "12028731e229c59666162826e1ec46b816c21750a84852c614a5a46f7c1d4f01",
            image_plan.identity(self.root, "cuda"),
        )

    def test_config_rejects_unsafe_versions(self):
        path = self.backend / "config.json"
        original = json.loads(path.read_text())
        for field in ("torch_version", "runtime_version"):
            for bad in ("", "nightly", "2.14 --index-url bad", 214):
                with self.subTest(field=field, value=bad):
                    path.write_text(json.dumps(dict(original, **{field: bad})))
                    with self.assertRaises(ValueError):
                        image_plan.identity(self.root, "cuda")

    def test_bootstrap_uses_custom_recipe_and_bounded_smokes(self):
        (self.backend / "dependencies.Dockerfile").write_text("FROM pinned-vendor-os")
        digest = "sha256:" + "a" * 64
        with (
            patch("image_plan.registry_digest", side_effect=[None, digest, digest]),
            patch("image_plan.run") as run,
            patch("image_plan.subprocess.run"),
            patch("image_plan.Path.write_text"),
        ):
            image_plan.publish(
                self.root, "cuda", "ghcr.io/supernovae/model-lab", "c" * 40
            )
        calls = [call.args[0] for call in run.call_args_list]
        builds = [argv for argv in calls if argv[:3] == ["docker", "buildx", "build"]]
        self.assertEqual(
            builds[0][builds[0].index("-f") + 1],
            "container/backends/cuda/dependencies.Dockerfile",
        )
        smokes = [argv for argv in calls if argv[:2] == ["docker", "run"]]
        self.assertEqual(len(smokes), 2)
        self.assertEqual(
            smokes[0][-4:],
            ["python", "/opt/image_plan.py", "verify", "/opt/backend.json"],
        )
        self.assertEqual(smokes[1][-2:], ["klab", "--help"])
        for argv in smokes:
            for flag in (
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--memory=4g",
                "--cpus=2",
                "--pids-limit=256",
                "--tmpfs=/state:rw,uid=10001,gid=10001,size=67108864",
            ):
                self.assertIn(flag, argv)

    def test_missing_dependency_digest_prevents_application_build(self):
        with (
            patch("image_plan.registry_digest", side_effect=[None, None]),
            patch("image_plan.run") as run,
            self.assertRaisesRegex(RuntimeError, "dependency digest unavailable"),
        ):
            image_plan.publish(
                self.root, "cuda", "ghcr.io/supernovae/model-lab", "c" * 40
            )
        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            len([cmd for cmd in calls if cmd[:3] == ["docker", "buildx", "build"]]), 1
        )
        self.assertEqual(calls[-1][:2], ["docker", "push"])

    def test_every_dependency_input_changes_identity(self):
        before = image_plan.identity(self.root, "cuda")
        for name in (*image_plan.INPUTS, "container/backends/cuda/requirements.lock"):
            path = self.root / name
            original = path.read_text()
            path.write_text(original + "\n# changed")
            self.assertNotEqual(before, image_plan.identity(self.root, "cuda"), name)
            path.write_text(original)
        cfg = self.backend / "config.json"
        cfg.write_text(cfg.read_text().replace("13.2", "13.4"))
        self.assertNotEqual(before, image_plan.identity(self.root, "cuda"))

    def test_custom_os_recipe_and_pins_change_identity(self):
        before = image_plan.identity(self.root, "cuda")
        recipe = self.backend / "dependencies.Dockerfile"
        recipe.write_text("FROM pinned-vendor-os")
        self.assertEqual(
            image_plan.dependency_recipe(self.root, "cuda"),
            "container/backends/cuda/dependencies.Dockerfile",
        )
        self.assertNotEqual(before, image_plan.identity(self.root, "cuda"))
        before = image_plan.identity(self.root, "cuda")
        (self.backend / "os-packages.lock").write_text("verified-package-pin")
        self.assertNotEqual(before, image_plan.identity(self.root, "cuda"))

    def test_source_and_application_recipe_reuse_dependencies(self):
        before = image_plan.identity(self.root, "cuda")
        (self.root / "src").mkdir()
        (self.root / "src/app.py").write_text("changed")
        (self.root / "pyproject.toml").write_text("application version changed")
        (self.root / "container/application.Dockerfile").write_text("changed")
        self.assertEqual(before, image_plan.identity(self.root, "cuda"))

    def test_invalid_config_and_missing_input_refuse(self):
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "../cuda")
        (self.backend / "requirements.lock").unlink()
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_registry_errors_are_not_bootstrap(self):
        for error in ("unauthorized", "denied", "connection reset", "not found"):
            with (
                patch(
                    "image_plan.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 1, "", error),
                ),
                self.assertRaises(RuntimeError),
            ):
                image_plan.registry_digest("example")
        with patch(
            "image_plan.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "", "manifest unknown"),
        ):
            self.assertIsNone(image_plan.registry_digest("example"))

    def test_unhashed_or_unpinned_lock_refuses(self):
        for value in (
            "torch>=2",
            "torch==2",
            "--extra-index-url https://untrusted.example",
        ):
            (self.backend / "requirements.lock").write_text(value)
            with self.assertRaises(ValueError):
                image_plan.identity(self.root, "cuda")

    def test_registry_digest_is_exact(self):
        digest = "sha256:" + "a" * 64
        with patch(
            "image_plan.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 0, json.dumps({"digest": digest}), ""
            ),
        ):
            self.assertEqual(digest, image_plan.registry_digest("example"))

    def test_existing_dependencies_are_never_rebuilt(self):
        digest = "sha256:" + "b" * 64
        with (
            patch("image_plan.registry_digest", return_value=digest),
            patch("image_plan.run") as run,
            patch("image_plan.subprocess.run"),
            patch("image_plan.Path.write_text"),
        ):
            image_plan.publish(
                self.root, "cuda", "ghcr.io/supernovae/model-lab", "c" * 40
            )
        calls = [call.args[0] for call in run.call_args_list]
        builds = [cmd for cmd in calls if cmd[:3] == ["docker", "buildx", "build"]]
        self.assertEqual(len(builds), 1)
        self.assertIn(
            "DEPENDENCY_IMAGE=ghcr.io/supernovae/model-lab@" + digest, builds[0]
        )
        self.assertEqual(calls[-1][:2], ["docker", "push"])

    def test_failed_smoke_prevents_application_push(self):
        def fake_run(argv, **kwargs):
            if argv[:2] == ["docker", "run"]:
                raise RuntimeError("smoke failed")

        with (
            patch("image_plan.registry_digest", return_value="sha256:" + "a" * 64),
            patch("image_plan.run", side_effect=fake_run) as run,
            patch("image_plan.subprocess.run"),
            self.assertRaises(RuntimeError),
        ):
            image_plan.publish(
                self.root, "cuda", "ghcr.io/supernovae/model-lab", "c" * 40
            )
        self.assertFalse(
            any(c.args[0][:2] == ["docker", "push"] for c in run.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
