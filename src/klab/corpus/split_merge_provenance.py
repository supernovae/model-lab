"""Reviewed exact-content components; early diagnostics, never release proof."""

from __future__ import annotations

import hashlib
from pathlib import Path

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import unchanged
from klab.corpus.bounded_metadata import read_metadata
from klab.corpus.project import project_path
from klab.corpus.split_inventory_adoption import _reference


def exact_content_family_id(stratum, members):
    members = sorted(members)
    if (
        not isinstance(stratum, str)
        or not stratum
        or len(members) < 2
        or len(set(members)) != len(members)
        or any(not isinstance(x, str) or not x for x in members)
    ):
        raise ValueError("invalid exact-content family component")
    payload = {"stratum": stratum, "member_hints": members}
    return "exact-content-family:" + canonical_digest(payload)


def reviewed_component_merges(components):
    result = []
    seen = set()
    for component in components:
        members = sorted(component["families"])
        if seen.intersection(members):
            raise ValueError("family appears in multiple reviewed components")
        seen.update(members)
        result.append(
            {
                "family_id": exact_content_family_id(component["stratum"], members),
                "member_hints": members,
            }
        )
    return sorted(result, key=lambda row: row["family_id"])


def cluster_semantics_sha256(clusters):
    """Exclude operational references and review prose to avoid a hash cycle."""
    fields = (
        "inventory_sha256",
        "seed",
        "source_strata",
        "merges",
        "split_weights",
        "prior_family_inventory",
    )
    return hashlib.sha256(
        canonical_json({k: clusters[k] for k in fields if k in clusters})
    ).hexdigest()


def check_reviewed_merge_provenance(project, root, inventory, clusters, reference):
    root = Path(root).resolve()
    path, dependencies = _reference(root, reference)
    provenance = read_metadata(path)
    if (
        set(provenance)
        != {
            "kind",
            "cluster_semantics_sha256",
            "diagnostic_summary",
            "diagnostic_components",
        }
        or provenance["kind"] != "reviewed_exact_content_components"
        or provenance["cluster_semantics_sha256"] != cluster_semantics_sha256(clusters)
    ):
        raise ValueError(
            "reviewed merge provenance differs from current split semantics"
        )
    summary_path, observed = _reference(root, provenance["diagnostic_summary"])
    dependencies.update(observed)
    components_path, observed = _reference(root, provenance["diagnostic_components"])
    dependencies.update(observed)
    summary, data = read_metadata(summary_path), read_metadata(components_path)
    admission = project.release.record_admission
    inventory_name = Path(inventory).relative_to(root).as_posix()
    admission_name = (
        project_path(project.root, admission.path).relative_to(root).as_posix()
    )
    if (
        summary.get("inputs", {}).get(inventory_name) != clusters["inventory_sha256"]
        or summary["inputs"].get(admission_name) != admission.sha256
    ):
        raise ValueError(
            "reviewed content diagnostic has stale inventory or admission inputs"
        )
    groups = data.get("groups", [])
    if (
        summary.get("cross_family_groups") != len(groups)
        or data.get("cross_family_groups") != len(groups)
        or summary.get("cross_stratum_groups") != 0
    ):
        raise ValueError(
            "reviewed content diagnostic has inconsistent group accounting"
        )
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in groups:
        families = {(r["stratum"], r["family_hint"]) for r in group["members"]}
        declared = {(r["stratum"], r["family_hint"]) for r in group["families"]}
        if (
            families != declared
            or len(families) < 2
            or len({s for s, _ in families}) != 1
        ):
            raise ValueError(
                "reviewed exact-content group crosses strata or omits members"
            )
        for member in group["members"]:
            if (
                clusters["source_strata"].get(member["source_id"]) != member["stratum"]
                or member["content_sha256"] != group["content_sha256"]
            ):
                raise ValueError(
                    "reviewed exact-content member differs from source stratum"
                )
        first, *rest = sorted(families)
        for family in rest:
            a, b = find(first), find(family)
            if a != b:
                parent[max(a, b)] = min(a, b)
    actual_components = {}
    for family in parent:
        actual_components.setdefault(find(family), []).append(family)
    actual = {frozenset(items) for items in actual_components.values()}
    declared = {
        frozenset((c["stratum"], f) for f in c["families"]) for c in data["components"]
    }
    if actual != declared or len(actual) != len(data["components"]):
        raise ValueError("reviewed components differ from exact-content connectivity")
    if clusters["merges"] != reviewed_component_merges(data["components"]):
        missing = next(
            (
                component
                for component in data["components"]
                if reviewed_component_merges([component])[0] not in clusters["merges"]
            ),
            None,
        )
        if missing is None:
            raise ValueError("reviewed merges differ from exact-content components")
        families = {(missing["stratum"], hint) for hint in missing["families"]}
        witness = next(
            group
            for group in groups
            if any(
                (member["stratum"], member["family_hint"]) in families
                for member in group["members"]
            )
        )
        locations = ", ".join(
            f"{r['source_id']} row {r['source_row_index']}"
            for r in witness["members"][:3]
        )
        raise ValueError(
            f"reviewed exact-content component is unresolved: {witness['content_sha256']} ({locations})"
        )
    unchanged(dependencies)
    return dependencies
