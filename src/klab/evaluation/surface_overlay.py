"""Read-only, identity-bound Surface Review status beside immutable triage reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from klab.evaluation.post_train_triage import read_triage
from klab.evaluation.surface_review import (
    _review,
    open_surface_bundle,
    read_surface_answers,
    surface_results,
)


def _matches(
    source: object,
    run_id: str,
    identity: dict[str, Any],
    inputs: dict[str, Any],
    generation: str,
) -> bool:
    if not isinstance(source, dict) or source.get("kind") != "klab_checkpoint":
        return False
    if source.get("run_id") != run_id:
        return False
    digest = identity.get("generation_manifest_sha256")
    if (
        not isinstance(digest, str)
        or source.get("checkpoint_digest", source.get("checkpoint_sha256")) != digest
    ):
        return False
    # Triage imports carry the verified run manifest. Study imports carry
    # tokenizer/source identities; check every overlapping field they provide.
    for source_field, report_field in (
        ("run_manifest_sha256", "manifest_sha256"),
        ("manifest_sha256", "manifest_sha256"),
        ("tokenizer_sha256", "tokenizer_sha256"),
        ("source_identity_sha256", "source_identity_sha256"),
    ):
        if (
            source_field in source
            and report_field in inputs
            and source[source_field] != inputs[report_field]
        ):
            return False
    return "generation" not in source or source["generation"] == generation


def surface_review_status(
    run_id: str, runs_dir: str | Path, surface_dir: str | Path
) -> dict[str, Any]:
    """List all verified bundles involving this run's verified endpoint.

    Invalid, incomplete or unrelated bundles never confer observed status. An
    invalid triage report is unknown, not a license to trust a bundle's claim.
    """
    result: dict[str, Any] = {
        "independent_subjective_quality": "UNKNOWN",
        "bundles": [],
    }
    try:
        report = read_triage(run_id, Path(runs_dir))
    except OSError, ValueError, TypeError, KeyError:
        return result
    if report is None or report["core"]["integrity"]["status"] != "PASS":
        return result
    identity = report["identity"]
    inputs = report["inputs"]
    root = Path(surface_dir)
    if root.is_symlink() or not root.is_dir():
        return result
    try:
        candidates = sorted(root.iterdir())
    except OSError:
        return result
    for bundle in candidates:
        if bundle.is_symlink() or not bundle.is_dir():
            continue
        try:
            view = open_surface_bundle(bundle, private=True)
            cases = view["provenance"]["cases"]
            sources = [
                source
                for case in cases
                for source in case["sides"].values()
                if isinstance(source, dict) and source.get("run_id") == run_id
            ]
            if not sources or not any(
                _matches(
                    source,
                    run_id,
                    identity,
                    inputs,
                    report["core"]["integrity"]["generation"],
                )
                for source in sources
            ):
                continue
            # One conflicting claim of the requested run invalidates this bundle.
            if any(
                not _matches(
                    source,
                    run_id,
                    identity,
                    inputs,
                    report["core"]["integrity"]["generation"],
                )
                for source in sources
            ):
                continue
            read_surface_answers(bundle)
            review_path = bundle / "review.json"
            reveal_path = bundle / "reveal.json"
            if reveal_path.exists() or reveal_path.is_symlink():
                if not (review_path.exists() or review_path.is_symlink()):
                    continue
                surface_results(
                    bundle
                )  # Verifies both immutable artifacts and their digest chain.
            if review_path.exists() or review_path.is_symlink():
                review = _review(bundle)
                status = "OBSERVED_SINGLE_REVIEWER"
            else:
                review = None
                status = "REVIEW_AVAILABLE"
            reference: dict[str, Any] = {"path": str(bundle), "status": status}
            if review is not None:
                reference["review_digest"] = review["digest"]
            result["bundles"].append(reference)
        except OSError, ValueError, TypeError, KeyError, AttributeError:
            continue
    if any(
        bundle["status"] == "OBSERVED_SINGLE_REVIEWER" for bundle in result["bundles"]
    ):
        result["independent_subjective_quality"] = "OBSERVED_SINGLE_REVIEWER"
    elif result["bundles"]:
        result["independent_subjective_quality"] = "REVIEW_AVAILABLE"
    return result
