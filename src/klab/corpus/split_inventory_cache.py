"""Authenticated completion of native inventory production, never a hash shortcut."""

from __future__ import annotations

import tempfile
from pathlib import Path

from klab.corpus.admission_inspection_cache import (
    InspectionCache,
    capture,
    unchanged,
)
from klab.corpus.project import source_declaration_payload
from klab.training.manifest import sha256_file


class InventoryCache(InspectionCache):
    """Reuse only completed producers with unchanged code, inputs and output."""

    def __init__(self, root):
        super().__init__(root)
        self.directory = self.root / "cache" / "split-inventory"
        if self.proofs is not None:
            try:
                self.authority = {
                    "corpus": self.authority["corpus"],
                    "inventory": {
                        name: sha256_file(Path(__file__).with_name(name))
                        for name in (
                            "split_inventory.py",
                            "split_inventory_cache.py",
                            "split_inventory_adoption.py",
                        )
                    },
                }
            except OSError:
                self.proofs = None
                self._unavailable("inventory_authority_unavailable")

    def request(self, project, path):
        splits = getattr(project, "splits", None)
        return {
            "path": str(Path(path).absolute()),
            "project_id": project.config.id,
            "sources": [
                source_declaration_payload(source) for source in project.sources
            ],
            "normalizer": getattr(getattr(project, "release", None), "normalizer", None)
            or "normalizer-nfc-markdown-v1",
            "family_fields_by_source": getattr(splits, "family_fields_by_source", {}),
        }

    def completed(self, project, path):
        result = self.load("inventory", self.request(project, path), limit=4 * 1024**2)
        if result is None:
            return None
        payload, dependencies = result
        if (
            not isinstance(payload, dict)
            or set(payload) != {"report", "completion"}
            or not isinstance(payload["report"], dict)
            or not isinstance(payload["completion"], dict)
            or payload["completion"].get("mode")
            not in {"native_production", "adopted_execution"}
        ):
            return None
        return payload["report"], dependencies, payload["completion"]

    def record_completed(self, project, path, report, dependencies, *, completion=None):
        path = Path(path).absolute()
        if any(not Path(p).is_relative_to(self.root) for p in dependencies):
            return  # External inputs remain cold; never omit them from a proof.
        self.publish(
            "inventory",
            self.request(project, path),
            {
                "report": report,
                "completion": completion
                or {
                    "mode": "native_production",
                    "producer_authority": self.authority,
                },
            },
            dependencies,
            limit=4 * 1024**2,
        )


def inventory_inputs(project, root, lock):
    """Capture metadata identity before the native producer reads source bodies."""
    paths = [Path(root) / "corpora" / project.config.id / "acquisition.json"]
    paths.extend(
        Path(entry["snapshot_path"])
        for entry in lock["sources"].values()
        if entry["snapshot_path"] is not None
    )
    return capture(paths)


def verified_inventory(
    project, root, path, expected_sha256, regenerate, *, adopted=None, work=None
):
    """A native completion proof avoids regeneration; a bare checksum does not."""
    path = Path(path).absolute()
    cache = InventoryCache(root)

    def result(report, reused, mode, adoption=None):
        if work is not None:
            work.update(completion_mode=mode, adopted=adoption)
        return report, reused

    if adopted is not None:
        from klab.corpus.split_inventory_adoption import _AdoptedInventory

        if not isinstance(adopted, _AdoptedInventory):
            raise TypeError(
                "inventory adoption requires process-owned checked execution evidence"
            )
        adopted.check(cache, project, path, expected_sha256)
        return result(adopted.report, True, "adopted_execution", adopted)
    completed = cache.completed(project, path)
    if completed is not None:
        report, _, completion = completed
        if completion["mode"] == "adopted_execution":
            from klab.corpus.split_inventory_adoption import (
                adopt_completed_inventory,
            )

            adopted = adopt_completed_inventory(
                project, root, path, completion["declaration"]
            )
            adopted.check(cache, project, path, expected_sha256)
            return result(adopted.report, True, "adopted_execution", adopted)
        if report.get("sha256") == expected_sha256 and report.get("path") == str(path):
            return result(report, True, "native_production")

    observed = capture([path])
    if sha256_file(path) != expected_sha256:
        raise ValueError(
            "invalid reviewed family-cluster declaration: "
            "split inventory differs from reviewed inventory digest"
        )
    unchanged(observed)
    # A fresh private child avoids inherited group-writable scratch parents;
    # existing directories and their permissions remain untouched.
    with tempfile.TemporaryDirectory(prefix=".inventory-check-", dir=root) as folder:
        temporary = Path(folder) / "inventory.jsonl"
        # Keep the authenticated temporary completion and its signed receipt
        # inside this operation's scratch lifetime, including on refusal/error.
        temporary_cache = InventoryCache(root)
        temporary_cache.directory = Path(folder) / "completion"
        if temporary_cache.proofs is not None:
            temporary_cache.proofs.directory = Path(folder) / "proofs"
        report = regenerate(project, root, temporary, completion_cache=temporary_cache)
        if report["sha256"] != expected_sha256:
            raise ValueError("split inventory differs from verified snapshots")
        unchanged(observed)
        report = {**report, "path": str(path)}
        produced = temporary_cache.completed(project, temporary)
        if produced is not None:
            native_report, dependencies, completion = produced
            if completion["mode"] != "native_production":
                raise ValueError(
                    "cold inventory verification lacks native producer completion"
                )
            if native_report["sha256"] != expected_sha256:
                raise ValueError("native inventory completion disagrees with output")
            dependencies.pop(str(temporary.absolute()), None)
            dependencies.update(observed)
            report = {**native_report, "path": str(path)}
            cache.record_completed(project, path, report, dependencies)
    return result(report, False, "cold_verification")
