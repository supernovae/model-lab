"""Disk-bounded prior-release indexes for incremental split preparation.

The retained native release authenticates its previously executed family
inventory.  Index only that compact metadata; never reread the document body.
"""

from __future__ import annotations

import json
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.split_runs import SplitStages, external_sort, rows
from klab.training.manifest import sha256_file


def _index(directory, inventory, expected_sha256):
    """Sort old family and content bindings, checking the input once per sort."""
    fields = {"document_id", "family_id", "split", "stratum", "content_sha256"}

    def items(kind):
        for record in records_from_path(
            inventory, max_record_bytes=8 * 1024**2, expected_sha256=expected_sha256
        ):
            row = record.value
            if (
                not isinstance(row, dict)
                or set(row) != fields
                or row["split"] not in {"train", "validation", "test"}
                or any(not isinstance(row[k], str) or not row[k] for k in fields)
            ):
                raise ValueError("invalid retained family inventory row")
            if kind == "family":
                yield [row["family_id"], row["stratum"], row["split"]]
            else:
                yield [
                    row["content_sha256"],
                    row["family_id"],
                    row["stratum"],
                    row["split"],
                ]

    families = external_sort(items("family"), directory / "families")
    content = external_sort(items("content"), directory / "content")
    prior_counts = {}
    for path, description in (
        (families["path"], "family"),
        (content["path"], "content"),
    ):
        previous = None
        for row in rows(path):
            if previous is not None and row[0] == previous[0] and row != previous:
                raise ValueError(
                    f"retained {description} spans families, strata or splits"
                )
            if description == "family" and (previous is None or row[0] != previous[0]):
                counts = prior_counts.setdefault(row[1], {})
                counts[row[2]] = counts.get(row[2], 0) + 1
            previous = row
    return {
        "files": [families, content],
        "families": families["path"],
        "content": content["path"],
        "prior_counts": prior_counts,
    }


def prior_indexes(root, binding):
    """Authenticate retained execution, then reuse or build its metadata indexes."""
    from klab.corpus.admission_inspection_cache import capture
    from klab.corpus.completed_release import verify_retained_release

    root = Path(root).resolve(strict=True)
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "release_path",
        "family_clusters",
    }:
        raise ValueError("invalid prior family-inventory binding")
    raw_release, raw_inventory = Path(binding["release_path"]), Path(binding["path"])
    if any(
        p.is_symlink()
        for path in (raw_release, raw_inventory)
        for p in (path, *path.parents)
    ):
        raise ValueError("unsafe prior release or inventory")
    release, inventory = (
        raw_release.resolve(strict=True),
        raw_inventory.resolve(strict=True),
    )
    if (
        not release.is_relative_to(root)
        or inventory != release / "family-inventory.jsonl"
    ):
        raise ValueError("prior inventory must be the retained release output")
    manifest = verify_retained_release(release)
    if manifest["files"]["family-inventory.jsonl"]["sha256"] != binding["sha256"]:
        raise ValueError("prior inventory digest differs from retained execution")
    alias_binding = binding["family_clusters"]
    if not isinstance(alias_binding, dict) or set(alias_binding) != {"path", "sha256"}:
        raise ValueError("prior family clusters must be explicitly bound")
    aliases_path = Path(alias_binding["path"])
    if (
        not aliases_path.is_absolute()
        or any(p.is_symlink() for p in (aliases_path, *aliases_path.parents))
        or not aliases_path.resolve(strict=True).is_relative_to(root)
        or not aliases_path.is_file()
        or aliases_path.stat().st_size > 4 * 1024**2
        or sha256_file(aliases_path) != alias_binding["sha256"]
    ):
        raise ValueError("invalid prior family clusters")
    request_inputs = manifest["build_identity"]["completed_preparation"]["request"][
        "inputs"
    ]
    receipts = []
    for name in request_inputs:
        path = Path(name)
        if (
            name.endswith(".receipt.json")
            and path.is_file()
            and path.stat().st_size <= 4 * 1024**2
        ):
            if capture([path]).get(name) != request_inputs[name]:
                raise ValueError("prior split receipt fingerprint differs")
            value = json.loads(path.read_bytes())
            if capture([path]).get(name) != request_inputs[name]:
                raise ValueError("prior split receipt changed during read")
            if value.get("cluster_sha256"):
                receipts.append(value)
    if len(receipts) != 1 or receipts[0]["cluster_sha256"] != alias_binding["sha256"]:
        raise ValueError("prior clusters differ from the authenticated split receipt")
    clusters = json.loads(aliases_path.read_bytes())
    aliases = {}
    for merge in clusters.get("merges", []):
        if set(merge) != {"family_id", "member_hints"} or not merge["family_id"]:
            raise ValueError("invalid prior family merge")
        for hint in merge["member_hints"]:
            if not hint or hint in aliases:
                raise ValueError("ambiguous prior family merge")
            aliases[hint] = merge["family_id"]
    producer = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("streaming_prior.py", "split_runs.py", "jsonl_records.py")
    }
    request = {
        "producer": producer,
        "release_id": manifest["release_id"],
        "inventory_sha256": binding["sha256"],
        "inputs": capture(
            [
                inventory,
                release / "manifest.json",
                release / "release-publication.json",
                aliases_path,
            ]
        ),
    }
    stages = SplitStages(
        root / "corpora" / manifest["corpus_id"] / "prior-family-indexes",
        request,
        work_root=root,
    )
    if stages.cache.proofs is None or not stages.cache.writable:
        raise ValueError("prior family indexing requires native completion storage")
    result = stages.phase(
        "prior-family-indexes", lambda d: _index(d, inventory, binding["sha256"])
    )
    result["aliases"] = aliases
    return result, {
        "release_id": manifest["release_id"],
        "inventory_sha256": binding["sha256"],
        "cluster_sha256": alias_binding["sha256"],
        "request_sha256": canonical_digest(request),
    }


def check_merges(directory, prior, merges):
    """Never replace a historical family identity with a new component."""
    from klab.corpus.split_runs import write_rows

    members = sorted(merges)
    old = iter(rows(prior["families"]))
    current = next(old, None)
    for member in members:
        canonical = prior.get("aliases", {}).get(member, member)
        if canonical != member and merges[member] != canonical:
            raise ValueError("new merge changes a prior reviewed family")
    for member in members:
        while current is not None and current[0] < member:
            current = next(old, None)
        if current is not None and current[0] == member and merges[member] != member:
            raise ValueError("new merge changes a prior reviewed family")
    item = write_rows(directory / "checked.jsonl", [{"checked": True}])
    return {"files": [item], "checked": True}


def check_content(directory, joined, prior):
    """Anchor new families containing retained content to the retained split."""
    candidates = external_sort(
        ([row[4], row[1], row[2]] for row in rows(joined["qualified"]) if row[0]),
        directory / "candidates",
    )

    def anchors():
        old = iter(rows(prior["content"]))
        current = next(old, None)
        for content, family, stratum in rows(candidates["path"]):
            while current is not None and current[0] < content:
                current = next(old, None)
            if current is not None and current[0] == content:
                if current[2] != stratum:
                    raise ValueError("prior content changed token stratum")
                yield [family, stratum, current[3]]

    anchored = external_sort(anchors(), directory / "content-anchors")
    previous = None
    for row in rows(anchored["path"]):
        if previous is not None and previous[0] == row[0] and previous != row:
            raise ValueError("new family links to conflicting retained splits")
        previous = row
    return {
        "files": [candidates, anchored],
        "checked": True,
        "anchors": anchored["path"],
    }


def pin_families(directory, families, prior, content=None):
    """Merge old split anchors with new families without a corpus-sized map."""
    by_family = external_sort(
        (
            [family, stratum, rank]
            for stratum, rank, family in rows(families["families"])
        ),
        directory / "by-family",
    )
    old = iter(rows(prior["families"]))
    current = next(old, None)
    pinned, overlapping, extra = {}, {}, {}
    content_rows = iter(rows(content["anchors"])) if content else iter(())
    content_row = next(content_rows, None)

    def annotated():
        nonlocal current, content_row
        for family, stratum, rank in rows(by_family["path"]):
            while current is not None and current[0] < family:
                current = next(old, None)
            split = None
            if current is not None and current[0] == family:
                if current[1] != stratum:
                    raise ValueError("prior family changed token stratum")
                split = current[2]
                counts = overlapping.setdefault(stratum, {})
                counts[split] = counts.get(split, 0) + 1
            while content_row is not None and content_row[0] < family:
                content_row = next(content_rows, None)
            if content_row is not None and content_row[0] == family:
                if content_row[1] != stratum or (
                    split is not None and split != content_row[2]
                ):
                    raise ValueError("new family conflicts with retained split")
                if split is None:
                    split = content_row[2]
                    counts = extra.setdefault(stratum, {})
                    counts[split] = counts.get(split, 0) + 1
            if split is not None:
                counts = pinned.setdefault(stratum, {})
                counts[split] = counts.get(split, 0) + 1
            yield [stratum, rank, family, split]

    ranked = external_sort(annotated(), directory / "ranked")
    return {
        "files": [by_family, ranked],
        "families": ranked["path"],
        "counts": families["counts"],
        "pinned": pinned,
        "overlapping": overlapping,
        "extra_pinned": extra,
        "prior_counts": prior["prior_counts"],
    }
