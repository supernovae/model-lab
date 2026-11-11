"""Cold binding of an explicit reviewed release decision to its lineage evidence."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from klab.corpus.release import verify_release
from klab.training.manifest import sha256_file


def verify_accepted_prepared_config(path: Path, review: dict[str, object]) -> None:
    """Tie a prepared-input or training config to the exact accepted mixture."""
    from klab.config.loading import load_config

    dataset = load_config(path).dataset
    if (
        dataset.source != "local_token_mixture"
        or dataset.revision != review["release_id"]
        or dataset.mixture_declaration_path is None
        or dataset.mixture_output_path is None
        or dataset.train_path is None
        or dataset.validation_path is None
        or dataset.train_path.resolve()
        != (dataset.mixture_output_path / "train.tokens.jsonl").resolve()
        or dataset.validation_path.resolve()
        != (Path(str(review["release_path"])) / "lm" / "validation.jsonl").resolve()
    ):
        raise ValueError("prepared config differs from accepted release")
    mixture = yaml.safe_load(dataset.mixture_declaration_path.read_text())
    if (
        not isinstance(mixture, dict)
        or Path(mixture.get("release_path", "")).resolve()
        != Path(str(review["release_path"])).resolve()
        or Path(mixture.get("family_inventory", "")).resolve()
        != Path(str(review["family_inventory_path"])).resolve()
    ):
        raise ValueError("prepared config mixture differs from accepted release")


def verify_admission_review(admission: Path, work_root: Path) -> dict[str, object]:
    """Require an explicit reviewed decision over the new draft and exact manifest."""
    root = work_root.resolve(strict=True)
    receipt_path = admission.with_name(admission.name + ".review.json")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("admission review receipt is missing or symlinked")
    review = json.loads(receipt_path.read_text(encoding="utf-8"))
    if isinstance(review, dict) and review.get("format") == "klab-admission-review-v2":
        from klab.corpus.admission_inspection import verify_admission_inspection

        required = {
            "format",
            "decision",
            "reviewer",
            "reviewed_on",
            "draft_path",
            "draft_sha256",
            "admission_sha256",
            "inspection_path",
            "inspection_sha256",
            "item_decisions",
        }
        if (
            set(review) != required
            or review["decision"] != "ACCEPTED"
            or not isinstance(review["reviewer"], str)
            or not review["reviewer"].strip()
            or not isinstance(review["reviewed_on"], str)
            or not review["reviewed_on"].strip()
            or not isinstance(review["item_decisions"], list)
        ):
            raise ValueError("invalid v2 admission review")
        draft = Path(review["draft_path"])
        inspection_path = Path(review["inspection_path"])
        if (
            not draft.is_absolute()
            or draft.is_symlink()
            or not draft.resolve().is_relative_to(root)
            or not inspection_path.is_absolute()
            or inspection_path.is_symlink()
            or not inspection_path.resolve().is_relative_to(root)
            or sha256_file(draft) != review["draft_sha256"]
            or sha256_file(admission) != review["admission_sha256"]
            or review["draft_sha256"] != review["admission_sha256"]
            or sha256_file(inspection_path) != review["inspection_sha256"]
        ):
            raise ValueError("v2 admission review identity mismatch")
        inspection = verify_admission_inspection(inspection_path, root)
        if inspection["draft_sha256"] != review["draft_sha256"]:
            raise ValueError("v2 admission inspection draft mismatch")
        expected_ids = [item["item_id"] for item in inspection["items"]]
        decisions = review["item_decisions"]
        if (
            len(set(expected_ids)) != len(expected_ids)
            or len(decisions) != len(expected_ids)
            or any(
                not isinstance(item, dict)
                or set(item) != {"item_id", "outcome", "note"}
                or item.get("item_id") != expected
                or item.get("outcome") != "pass"
                or not isinstance(item.get("note"), str)
                or not item["note"].strip()
                for item, expected in zip(decisions, expected_ids, strict=True)
            )
        ):
            raise ValueError(
                "v2 admission review has incomplete, substituted or blocking decisions"
            )
        return review
    raise ValueError("admission review requires an inspection-bound decision")


def verify_release_review(
    path: Path, work_root: Path, *, build_project: Path | None = None
) -> dict[str, object]:
    root = work_root.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("release review is missing or symlinked")
    review = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format",
        "decision",
        "reviewer",
        "reviewed_on",
        "release_path",
        "release_id",
        "release_manifest_sha256",
        "family_inventory_path",
        "family_inventory_sha256",
        "protected_lineage_path",
        "protected_lineage_sha256",
    }
    if (
        not isinstance(review, dict)
        or set(review) != required
        or review["format"] != "klab-release-review-v1"
        or review["decision"] != "ACCEPTED"
        or not isinstance(review["reviewer"], str)
        or not review["reviewer"].strip()
        or not isinstance(review["reviewed_on"], str)
        or not review["reviewed_on"].strip()
    ):
        raise ValueError("release review has no explicit accepted decision")
    locations = {
        key: Path(review[key])
        for key in ("release_path", "family_inventory_path", "protected_lineage_path")
    }
    if any(
        not location.is_absolute()
        or location.is_symlink()
        or not location.is_relative_to(root)
        or not location.resolve().is_relative_to(root)
        for location in locations.values()
    ):
        raise ValueError("release review has an unsafe evidence path")
    release = locations["release_path"]
    manifest = verify_release(release)
    if build_project is not None:
        from klab.corpus.project import load_project, release_declaration_payload

        project = load_project(build_project)
        identity = manifest["build_identity"]
        if (
            identity["project_id"] != project.config.id
            or identity["project"] != project.config.model_dump(mode="json")
            or identity["split"] != project.splits.model_dump(mode="json")
            or identity["release"] != release_declaration_payload(project.release)
        ):
            raise ValueError("release review differs from bound build project")
    if (
        manifest["release_id"] != review["release_id"]
        or sha256_file(release / "manifest.json") != review["release_manifest_sha256"]
        or sha256_file(locations["family_inventory_path"])
        != review["family_inventory_sha256"]
        or sha256_file(locations["protected_lineage_path"])
        != review["protected_lineage_sha256"]
    ):
        raise ValueError("release review evidence identity mismatch")
    lineage = json.loads(locations["protected_lineage_path"].read_text())
    if (
        lineage.get("status") != "PASS"
        or lineage.get("candidate_release_id") != review["release_id"]
        or lineage.get("candidate_manifest_sha256") != review["release_manifest_sha256"]
        or lineage.get("candidate_inventory_sha256")
        != review["family_inventory_sha256"]
    ):
        raise ValueError("release review lacks a matching passing lineage audit")
    return review
