"""Reuse completed preparation through immutable streaming split phases.

The explicit caller policy reuses reviewed admission decisions. It does not
reconstruct every source body/provenance envelope again, and records that limit.
Scientific family assignment and exact-content isolation remain mandatory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from pathlib import Path

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.bounded_metadata import _HashedInput, _Reader
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.project import project_path, source_declaration_payload
from klab.corpus.rights import RightsPolicy, resolve_file_rights
from klab.corpus.split_runs import (
    BUFFER_BYTES,
    SplitStages,
    _stamp,
    external_sort,
    rows,
    write_rows,
)
from klab.training.manifest import sha256_file

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_STATES = {"qualify", "exclude", "quarantine"}


class _Spool(list):
    def __init__(self, path):
        self.path, self.count = path, 0
        self.stream = path.open("xb")

    def __len__(self):
        return self.count


class _AdmissionSpools:
    """Use the existing bounded JSON reader, without SQLite materialization."""

    def __init__(self, directory):
        self.directory, self.spools = directory, []

    def new_rows(self):
        spool = _Spool(self.directory / f"decisions-{len(self.spools):04d}.jsonl")
        self.spools.append(spool)
        return spool

    def append(self, spool, item):
        if "source_row_index" in item:
            index, state = item["source_row_index"], item["decision"]
            if type(index) is not int or index < 0 or state not in _STATES:
                raise ValueError("invalid reviewed row decision")
            item = [index, state]
        spool.stream.write(canonical_json(item) + b"\n")
        spool.count += 1

    def close(self, complete):
        for spool in self.spools:
            if complete:
                spool.stream.flush()
                os.fsync(spool.stream.fileno())
            spool.stream.close()


def _admission_phase(directory, path, expected, declarations):
    store = _AdmissionSpools(directory)
    before = _stamp(path)
    try:
        with path.open("rb") as stream:
            hashed = _HashedInput(stream)
            reader = _Reader(
                hashed, store, path.stat().st_size + 1, 4 * 1024**2, 8 * 1024**2
            )
            manifest = reader.read()
        if hashed.digest.hexdigest() != expected or _stamp(path) != before:
            raise ValueError("reviewed admission changed or has wrong digest")
        if (
            set(manifest) != {"schema_version", "policy_id", "policy_sha256", "sources"}
            or manifest["schema_version"] != 2
            or not manifest["policy_id"]
            or not _HEX.fullmatch(manifest["policy_sha256"])
        ):
            raise ValueError("complete reviewed v2 admission required")
        headers = {}
        for item in manifest["sources"]:
            source_id = item["source_id"]
            if source_id in headers or source_id not in declarations:
                raise ValueError("admission source inventory mismatch")
            source = declarations[source_id]
            if item["source_revision"] != source["revision"]:
                raise ValueError("admission source revision changed")
            policy = RightsPolicy.model_validate(item["rights"])
            if (
                policy.training_eligibility
                not in {"eligible", "eligible_with_obligations"}
                or policy.redistribution_mode != "metadata_reconstruction_only"
                or not policy.license_references
                or not item["license_label"]
            ):
                raise ValueError("reviewed rights do not permit this local training")
            if not _HEX.fullmatch(item["snapshot_sha256"]):
                raise ValueError("invalid reviewed snapshot identity")
            if source["kind"] == "huggingface_dataset":
                spool = item["records"]
                sample = item["sample_path"]
                checked = resolve_file_rights(
                    policy, sample, b"", prospective_private_research=True
                )
                if checked.training_eligibility not in {
                    "eligible",
                    "eligible_with_obligations",
                }:
                    raise ValueError("reviewed rights do not qualify source shard")
                headers[source_id] = {
                    "snapshot": item["snapshot_sha256"],
                    "path": sample,
                    "decisions": str(spool.path),
                    "kind": "rows",
                    "count": len(spool),
                }
            elif source["kind"] == "git":
                if (
                    not policy.spdx_expression
                    or policy.training_restriction is not None
                ):
                    raise ValueError("reviewed Git rights are not eligible")
                spool = item["files"]
                headers[source_id] = {
                    "snapshot": item["snapshot_sha256"],
                    "decisions": str(spool.path),
                    "kind": "files",
                    "count": len(spool),
                }
            else:
                raise ValueError("unsupported completed admission source kind")
        if set(headers) != set(declarations):
            raise ValueError("admission source inventory mismatch")
        store.close(True)
    except BaseException:
        store.close(False)
        raise

    def decisions():
        for source_id, header in headers.items():
            for item in rows(header["decisions"]):
                if header["kind"] == "rows":
                    index, state = item
                    yield [source_id, header["path"], index, state]
                else:
                    if item["decision"] not in _STATES:
                        raise ValueError("invalid reviewed file decision")
                    yield [source_id, item["path"], -1, item["decision"]]

    sorted_file = external_sort(
        decisions(), directory / "sorted", key=lambda row: row[:3]
    )
    previous = None
    for record in rows(sorted_file["path"]):
        if record[:3] == previous:
            raise ValueError("duplicate reviewed admission decision")
        previous = record[:3]
    for header in headers.values():
        header.pop("decisions")
    return {
        "files": [sorted_file],
        "decisions": sorted_file["path"],
        "sources": headers,
        "policy_id": manifest["policy_id"],
        "policy_sha256": manifest["policy_sha256"],
    }


def _inventory_phase(directory, path, expected, headers, source_strata):
    count = 0
    before = _stamp(path)

    def records():
        nonlocal count
        for record in records_from_path(
            path, max_record_bytes=8 * 1024**2, expected_sha256=expected
        ):
            row = record.value
            source = row["source_id"]
            header = headers[source]
            if row["snapshot_sha256"] != header["snapshot"]:
                raise ValueError("inventory and admission snapshots disagree")
            if not _HEX.fullmatch(row["document_id"]) or not _HEX.fullmatch(
                row["content_sha256"]
            ):
                raise ValueError("invalid inventory document/content identity")
            raw_path = row["source_location"].split("#", 1)[0]
            index = row["source_row_index"] if header["kind"] == "rows" else -1
            if type(index) is not int or index < (
                -1 if header["kind"] == "files" else 0
            ):
                raise ValueError("invalid inventory source row index")
            count += 1
            yield [
                source,
                raw_path,
                index,
                row["document_id"],
                row["content_sha256"],
                row["family_hint"],
                source_strata[source],
            ]
        if _stamp(path) != before:
            raise ValueError("completed inventory changed or has wrong digest")

    sorted_file = external_sort(
        records(), directory / "sorted", key=lambda row: row[:3]
    )
    return {
        "files": [sorted_file],
        "inventory": sorted_file["path"],
        "documents": count,
    }


def _join_phase(directory, inventory, admission, merges, prior_aliases=None):
    decisions = iter(rows(admission["decisions"]))
    decision = next(decisions, None)
    seen_hints, eligible = set(), 0

    def qualified():
        nonlocal decision, eligible
        for row in rows(inventory["inventory"]):
            key = row[:3]
            while decision is not None and decision[:3] < key:
                decision = next(decisions, None)
            if decision is None or decision[:3] != key:
                raise ValueError("inventory lacks reviewed admission decision")
            _source, _path, _index, document, content, hint, stratum = row
            if hint in merges:
                seen_hints.add(hint)
            if decision[3] != "qualify":
                yield [0, "", "", document, content]
                continue
            if (
                not isinstance(hint, str)
                or not hint
                or hint.startswith("unresolved-family:")
            ):
                raise ValueError("eligible document has unresolved family")
            eligible += 1
            canonical = (prior_aliases or {}).get(hint, hint)
            yield [1, merges.get(hint, canonical), stratum, document, content]
        if seen_hints != set(merges):
            raise ValueError("reviewed merge contains absent family hint")

    sorted_file = external_sort(qualified(), directory / "sorted")
    return {
        "files": [sorted_file],
        "qualified": sorted_file["path"],
        "eligible": eligible,
    }


def _family_phase(directory, joined, seed):
    counts = {}

    def families():
        qualified = (row for row in rows(joined["qualified"]) if row[0])
        for family, members in itertools.groupby(qualified, key=lambda row: row[1]):
            stratum = None
            for row in members:
                if stratum is not None and row[2] != stratum:
                    raise ValueError("reviewed family spans multiple token strata")
                stratum = row[2]
            counts[stratum] = counts.get(stratum, 0) + 1
            rank = canonical_digest([seed, stratum, family])
            yield [stratum, rank, family]

    sorted_file = external_sort(families(), directory / "sorted")
    return {"files": [sorted_file], "families": sorted_file["path"], "counts": counts}


def _allocation_phase(directory, family_phase, weights, strata):
    from klab.corpus.split_freeze import _SPLITS, _counts

    realized, remaining = {}, {}
    for stratum in sorted(strata):
        delta_count = family_phase["counts"].get(stratum, 0)
        pinned = family_phase.get("pinned", {}).get(stratum, {})
        prior = family_phase.get("prior_counts", {}).get(stratum, {})
        overlapping = family_phase.get("overlapping", {}).get(stratum, {})
        extra = family_phase.get("extra_pinned", {}).get(stratum, {})
        count = delta_count + sum(prior.values()) - sum(overlapping.values())
        if count < sum(weights):
            raise ValueError(f"too few eligible families for {stratum} split")
        counts = _counts(count, weights)
        realized[stratum] = {
            "families": count,
            **dict(zip(_SPLITS, counts, strict=True)),
            "fractions": {
                s: f"{n}/{count}" for s, n in zip(_SPLITS, counts, strict=True)
            },
        }
        remaining[stratum] = {
            split: quota
            - (
                prior.get(split, 0) + extra.get(split, 0)
                if "prior_counts" in family_phase
                else pinned.get(split, 0)
            )
            for split, quota in zip(_SPLITS, counts, strict=True)
        }
        if any(count < 0 for count in remaining[stratum].values()):
            raise ValueError("prior family assignments exceed new split quota")
        if sum(remaining[stratum].values()) != delta_count - sum(pinned.values()):
            raise ValueError("prior family counts do not match pinned assignments")

    def allocations():
        previous, index = None, 0
        for row in rows(family_phase["families"]):
            stratum, _rank, family = row[:3]
            pinned = row[3] if len(row) == 4 else None
            if stratum != previous:
                previous, index = stratum, 0
            count = remaining[stratum]
            if pinned is not None:
                split = pinned
            else:
                split = (
                    "train"
                    if index < count["train"]
                    else (
                        "validation"
                        if index < count["train"] + count["validation"]
                        else "test"
                    )
                )
                index += 1
            yield [family, stratum, split]

    sorted_file = external_sort(allocations(), directory / "sorted")
    return {
        "files": [sorted_file],
        "allocations": sorted_file["path"],
        "family_counts": realized,
    }


def _assigned_phase(directory, joined, allocated):
    allocations = iter(rows(allocated["allocations"]))
    allocation = next(allocations, None)

    def assigned():
        nonlocal allocation
        for qualified, family, stratum, document, content in rows(joined["qualified"]):
            if not qualified:
                yield [document, "test", None, None, content]
                continue
            while allocation is not None and allocation[0] < family:
                allocation = next(allocations, None)
            if allocation is None or allocation[:2] != [family, stratum]:
                raise ValueError("family allocation coverage mismatch")
            yield [document, allocation[2], family, stratum, content]

    sorted_file = external_sort(
        assigned(), directory / "sorted", key=lambda row: row[0]
    )
    previous, count = None, 0
    for row in rows(sorted_file["path"]):
        if row[0] == previous:
            raise ValueError("split inventory has duplicate document IDs")
        previous, count = row[0], count + 1
    return {"files": [sorted_file], "assigned": sorted_file["path"], "documents": count}


def _content_phase(directory, assigned):
    sorted_file = external_sort(
        ([row[4], row[1]] for row in rows(assigned["assigned"]) if row[2] is not None),
        directory / "sorted",
    )
    previous = None
    for row in rows(sorted_file["path"]):
        if previous is not None and row[0] == previous[0] and row[1] != previous[1]:
            raise ValueError("cross-split exact content duplicate")
        previous = row
    return {"files": [sorted_file], "checked": True}


def _export(stages, kind, assigned, expected_rows):
    """Commit export chunks with source byte offsets; resume at the next chunk."""
    chunks, offset, finished, index = [], 0, False, 0
    while not finished:
        start = offset

        def chunk(directory, start=start):
            next_offset, size, eof = start, 0, False

            def records():
                nonlocal next_offset, size, eof
                with Path(assigned["assigned"]).open("rb") as stream:
                    stream.seek(start)
                    while raw := stream.readline():
                        row = json.loads(raw)
                        next_offset = stream.tell()
                        if kind == "families" and row[2] is None:
                            continue
                        value = (
                            {"document_id": row[0], "split": row[1]}
                            if kind == "assignments"
                            else {
                                "document_id": row[0],
                                "family_id": row[2],
                                "split": row[1],
                                "stratum": row[3],
                                "content_sha256": row[4],
                            }
                        )
                        size += len(canonical_json(value)) + 1
                        yield value
                        if size >= BUFFER_BYTES:
                            break
                    else:
                        eof = True

            item = write_rows(directory / "chunk.jsonl", records())
            return {
                "files": [item],
                "next_offset": next_offset,
                "eof": eof,
                "rows": item["rows"],
            }

        result = stages.phase(f"{kind}-export-{index:06d}", chunk)
        chunks.extend(result["files"])
        offset, finished, index = result["next_offset"], result["eof"], index + 1
    if sum(item["rows"] for item in chunks) != expected_rows:
        raise ValueError("split export coverage mismatch")

    def assemble(directory):
        output, digest = directory / (kind + ".jsonl"), hashlib.sha256()
        with output.open("xb") as target:
            for item in chunks:
                with Path(item["path"]).open("rb") as stream:
                    while raw := stream.read(1024**2):
                        target.write(raw)
                        digest.update(raw)
            target.flush()
            os.fsync(target.fileno())
        return {
            "files": [
                {
                    "path": str(output),
                    "sha256": digest.hexdigest(),
                    "rows": expected_rows,
                    "bytes": output.stat().st_size,
                }
            ]
        }

    return stages.phase(kind + "-assembled", assemble)["files"][0]


def freeze_completed_preparation(
    project, work_root, inventory_path, clusters_path, output, *, recover=False
):
    from klab.corpus.admission_inspection_cache import capture, unchanged
    from klab.corpus.split_freeze import (
        _FREEZE_MARKER,
        _publish_new,
        _split_inputs,
        verify_split_bundle,
    )

    # Prior release assignments remain immutable anchors for an extension.
    with Path(clusters_path).open("rb") as stream:
        raw_clusters = stream.read(4 * 1024**2 + 1)
    if len(raw_clusters) > 4 * 1024**2:
        raise ValueError("reviewed cluster header byte cap exceeded")
    json.loads(raw_clusters)
    (
        root,
        configuration,
        clusters,
        weights,
        cluster_sha,
        _prior_families,
        _prior_strata,
        _prior_content,
        adopted,
    ) = _split_inputs(
        project,
        work_root,
        inventory_path,
        clusters_path,
        output,
        recover=recover,
        load_prior=False,
    )
    if adopted is None:
        raise ValueError(
            "streaming reuse requires completed inventory execution evidence"
        )
    prior, prior_evidence = None, None
    if "prior_family_inventory" in clusters:
        from klab.corpus.streaming_prior import prior_indexes

        prior, prior_evidence = prior_indexes(root, clusters["prior_family_inventory"])
    reference = project.release.record_admission
    if reference is None:
        raise ValueError("streaming reuse requires reviewed admission")
    admission_path = project_path(project.root, reference.path)
    declarations = {
        source.id: source_declaration_payload(source) for source in project.sources
    }
    request = {
        "producer": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "streaming_split_freeze.py",
                "split_runs.py",
                "split_freeze.py",
                "bounded_metadata.py",
                "rights.py",
            )
        },
        "inventory_sha256": clusters["inventory_sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": cluster_sha,
        "sources": declarations,
        "inputs": capture([inventory_path, admission_path, clusters_path]),
        "inventory_execution": adopted.provenance,
    }
    if prior is not None:
        request["producer"]["streaming_prior.py"] = sha256_file(
            Path(__file__).with_name("streaming_prior.py")
        )
        request["prior_family_inventory"] = prior_evidence
        request["inputs"].update(capture([prior["families"], prior["content"]]))
    stages = SplitStages(
        root / "corpora" / project.config.id / "split-stages", request, work_root=root
    )
    admission = stages.phase(
        "admission",
        lambda d: _admission_phase(d, admission_path, reference.sha256, declarations),
    )
    inventory = stages.phase(
        "inventory",
        lambda d: _inventory_phase(
            d,
            inventory_path,
            clusters["inventory_sha256"],
            admission["sources"],
            clusters["source_strata"],
        ),
    )
    if inventory["documents"] != adopted.report["documents"]:
        raise ValueError("completed inventory document count mismatch")
    merges = {}
    for merge in clusters["merges"]:
        if (
            set(merge) != {"family_id", "member_hints"}
            or not merge["family_id"]
            or len(merge["member_hints"]) < 2
            or len(set(merge["member_hints"])) != len(merge["member_hints"])
        ):
            raise ValueError("invalid reviewed family merge")
        for hint in merge["member_hints"]:
            if hint in merges:
                raise ValueError("overlapping reviewed family merges")
            merges[hint] = merge["family_id"]
    if prior is not None:
        from klab.corpus.streaming_prior import (
            check_content,
            check_merges,
            pin_families,
        )

        stages.phase("prior-merge-isolation", lambda d: check_merges(d, prior, merges))
    joined = stages.phase(
        "join-admission",
        lambda d: _join_phase(
            d, inventory, admission, merges, prior.get("aliases") if prior else None
        ),
    )
    if prior is not None:
        content_anchors = stages.phase(
            "prior-content-isolation", lambda d: check_content(d, joined, prior)
        )
    families = stages.phase(
        "family-rank", lambda d: _family_phase(d, joined, clusters["seed"])
    )
    if prior is not None:
        families = stages.phase(
            "prior-family-anchors",
            lambda d: pin_families(d, families, prior, content_anchors),
        )
    allocated = stages.phase(
        "family-assign",
        lambda d: _allocation_phase(
            d, families, weights, set(clusters["source_strata"].values())
        ),
    )
    assigned = stages.phase(
        "document-assign", lambda d: _assigned_phase(d, joined, allocated)
    )
    if assigned["documents"] != inventory["documents"]:
        raise ValueError("split assignments do not cover inventory")
    stages.phase("content-isolation", lambda d: _content_phase(d, assigned))
    assignment = _export(stages, "assignments", assigned, inventory["documents"])
    candidates = _export(stages, "families", assigned, joined["eligible"])
    declaration = {
        "schema_version": 1,
        "unit": "document",
        "family_key": None,
        "assignments": {},
        "assignment_file": {
            "sha256": assignment["sha256"],
            "rows": inventory["documents"],
        },
    }
    overrides = getattr(getattr(project, "splits", None), "family_fields_by_source", {})
    if overrides:
        declaration["family_fields_by_source"] = {
            key: list(value) for key, value in sorted(overrides.items())
        }
    payload = _FREEZE_MARKER + yaml.safe_dump(declaration, sort_keys=False).encode()
    reuse_evidence = {
        **adopted.provenance,
        "source_validation": "This invocation reuses completed producer evidence; acquisition/snapshot verification and detailed row/file admission verification were not rerun.",
    }
    receipt = {
        "schema_version": 1,
        "split_sha256": hashlib.sha256(payload).hexdigest(),
        "family_candidate_sha256": candidates["sha256"],
        "inventory_sha256": clusters["inventory_sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": cluster_sha,
        "family_counts": allocated["family_counts"],
        "preparation_reuse": reuse_evidence,
        "deferred_verification": [
            "acquisition-lock and snapshot-manifest verification",
            "detailed row/file admission verification (coverage, source hashes, per-file rights exceptions and manual-review requirements)",
            "source-body hash reread",
            "per-record provenance-envelope reconstruction",
        ],
        "verification_scope": "completed inventory execution evidence, reviewed admission digest and decisions, source revision, inventory-to-admission snapshot bindings, basic source-level rights eligibility, family and exact-content split isolation",
        "checkpoint_chain": stages.previous,
        "checkpoint_cache": {
            "available": stages.cache.proofs is not None and stages.cache.writable,
            "unavailable_reason": stages.cache.unavailable_reason,
        },
    }
    if prior_evidence is not None:
        receipt["prior_family_inventory"] = prior_evidence
    if "split_weights" in clusters:
        receipt["split_weights"] = clusters["split_weights"]
    candidate_path = output.with_name(output.stem + ".family-candidates.jsonl")
    receipt_path = output.with_name(output.stem + ".receipt.json")
    members = [
        (candidate_path, Path(candidates["path"])),
        (
            output.parent / ("assignments-" + assignment["sha256"] + ".jsonl"),
            Path(assignment["path"]),
        ),
        (output, payload),
        (receipt_path, canonical_json(receipt) + b"\n"),
    ]
    for target, value in members:
        if target.is_symlink() or (
            target.exists()
            and (
                not target.is_file()
                or sha256_file(target)
                != (
                    sha256_file(value)
                    if isinstance(value, Path)
                    else hashlib.sha256(value).hexdigest()
                )
            )
        ):
            raise ValueError("conflicting frozen split publication")
    unchanged(configuration)
    unchanged(request["inputs"])
    from klab.corpus.split_inventory_cache import InventoryCache

    adopted.check(
        InventoryCache(root), project, inventory_path, clusters["inventory_sha256"]
    )
    for target, value in members:
        if not target.exists():
            _publish_new(target, value, work_root=root)
    verify_split_bundle(output)
    return {
        "schema_version": 1,
        "output": str(output),
        "sha256": sha256_file(output),
        "inventory_sha256": clusters["inventory_sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": cluster_sha,
        "family_candidates": str(candidate_path),
        "family_candidate_sha256": candidates["sha256"],
        "receipt": str(receipt_path),
        "eligible_documents": joined["eligible"],
        "placeholder_documents": inventory["documents"] - joined["eligible"],
        "family_counts": allocated["family_counts"],
        "inventory_reused": True,
        "inventory_completion": "adopted_execution",
        "reused_phases": stages.reused,
        "deferred_verification": receipt["deferred_verification"],
    }
