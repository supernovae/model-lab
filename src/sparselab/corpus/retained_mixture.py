"""Read authenticated native preparation offsets for additive mixtures.

This consumes executed split/build checkpoints, rather than reconstructing an
index by reading every old document body. It never creates a historical proof.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import mmap
import os
import sqlite3
import struct
import threading
from collections import OrderedDict
from pathlib import Path

from sparselab.canonical import canonical_digest
from sparselab.corpus.admission_inspection_cache import (
    InspectionCache,
    capture,
    unchanged,
)
from sparselab.corpus.completed_preparation import _producer
from sparselab.corpus.completed_release import verify_retained_release
from sparselab.corpus.mixture_index import MAX_ROW_BYTES, MixtureIndex
from sparselab.corpus.retained_render_adoption import _small
from sparselab.corpus.split_runs import (
    SplitStages,
    _stamp,
    external_sort,
    rows,
    write_rows,
)
from sparselab.training.manifest import sha256_file


def _phase(
    root,
    stage_root,
    name,
    *,
    expected_producer=None,
    allow_native=False,
    include_cache_hash=False,
):
    """Authenticate the latest live phase and its original signed binding."""
    expected_producer = _producer() if expected_producer is None else expected_producer
    root, stage_root = Path(root), Path(stage_root)
    if not stage_root.is_relative_to(root) or not stage_root.is_dir():
        raise ValueError("retained mixture stage is outside the work root")
    previous, active = None, {}
    for path in sorted(stage_root.glob("completion-*.json")):
        event = json.loads(_small(path, root))
        body = event["record"]
        if (
            event["sha256"] != canonical_digest(body)
            or body["previous"] != previous
            or body["request"] != stage_root.name
        ):
            raise ValueError("retained mixture completion chain differs")
        phase = body["phase"]
        if phase in active:
            for key in list(active)[list(active).index(phase) :]:
                del active[key]
        active[phase] = (event, path)
        previous = event["sha256"]
    if name not in active:
        raise ValueError(f"retained mixture phase unavailable: {name}")
    event, completion_path = active[name]
    payload = event["record"]["result"]
    request = {"request": stage_root.name, "completion": event["sha256"]}
    cache = InspectionCache(root)
    if cache.proofs is None:
        raise ValueError("retained mixture proof store unavailable")
    cache.directory = stage_root / "sealed"
    cache.authority = {**cache.authority, "streaming_producer": _producer()}
    cache_path = cache.entry_path("split-phase", request)
    raw = _small(cache_path, root)
    value = json.loads(raw)
    if set(value) != {"dependencies", "payload"} or value["payload"] != payload:
        raise ValueError("retained mixture phase cache differs")
    paths = value["dependencies"]
    if not isinstance(paths, list) or any(
        not isinstance(p, str)
        or not Path(p).is_absolute()
        or ".." in Path(p).parts
        or not Path(p).is_relative_to(root)
        for p in paths
    ):
        raise ValueError("unsafe retained mixture dependencies")
    dependencies = capture(paths)
    binding = cache.entry_binding(cache_path, raw, request, dependencies)
    proof_path = cache.proofs.receipt_path(binding)
    if allow_native and not proof_path.exists() and not proof_path.is_symlink():
        # Use the already-supported owned native completion only when the
        # optional receipt is truly absent. A present but rejected proof never
        # falls back to this path.
        cache.authority = {**cache.authority, "streaming_producer": expected_producer}
        completed = cache.load("split-phase", request, limit=4 * 1024**2)
        events = cache.proofs.diagnostics()["events"]
        if (
            completed is None
            or completed[0] != payload
            or not events
            or events[-1]["reason"] != "owned_native_completion"
        ):
            raise ValueError("retained mixture native completion unavailable")
        return payload, {
            "method": "owned_native_completion",
            "completion_sha256": event["sha256"],
            "completion_path": str(completion_path),
            "cache_path": str(cache_path),
            "cache_sha256": hashlib.sha256(raw).hexdigest(),
        }
    old_raw = _small(proof_path, root, private=True)
    saved = json.loads(old_raw)["proof"]["binding"]
    if not cache.proofs.lookup_recorded_execution(saved):
        raise ValueError("retained mixture historical proof invalid")
    before, after = saved["upstream_identities"], binding["upstream_identities"]
    if (
        before["authority"].get("streaming_producer") != expected_producer
        or before.get("mutable_dependencies", {})
        or after.get("mutable_dependencies", {})
    ):
        raise ValueError("retained mixture producer changed")
    normalized = {
        **binding,
        "upstream_identities": {**after, "authority": before["authority"]},
    }
    if "mutable_dependencies" not in before:
        normalized["upstream_identities"].pop("mutable_dependencies", None)
    normalized["manifest_inventory_sha256"] = canonical_digest(
        normalized["upstream_identities"]
    )
    if normalized != saved:
        raise ValueError("retained mixture semantic binding differs")
    for item in payload["files"]:
        path = Path(item["path"])
        if not path.is_relative_to(stage_root) or _stamp(path) != item["stamp"]:
            raise ValueError("retained mixture stage output changed")
    unchanged(dependencies)
    evidence = {
        "completion_sha256": event["sha256"],
        "historical_proof_sha256": hashlib.sha256(old_raw).hexdigest(),
        "completion_path": str(completion_path),
        "cache_path": str(cache_path),
        "proof_path": str(cache.proofs.receipt_path(binding)),
    }
    if include_cache_hash:
        evidence["cache_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload, evidence


def collect_mixture_stage_evidence(root, workspace, identity):
    """Inspect original stage evidence after the caller verifies its release.

    Global phases may use the existing owner-private native fallback; source
    renders still require their original historical proofs. No proof is minted.
    """
    root, workspace = Path(root), Path(workspace)
    request = identity["completed_preparation"]["request"]
    stage_root = workspace / "build-stages" / canonical_digest(request)
    refs, refs_proof = _phase(
        root,
        stage_root,
        "build-frozen-family-bindings",
        expected_producer=request["producer"],
        allow_native=True,
        include_cache_hash=True,
    )
    dedupe, dedupe_proof = _phase(
        root,
        stage_root,
        "build-deduplicate",
        expected_producer=request["producer"],
        allow_native=True,
        include_cache_hash=True,
    )
    origins, origins_proof = _phase(
        root,
        stage_root,
        "build-duplicate-keys",
        expected_producer=request["producer"],
        allow_native=True,
        include_cache_hash=True,
    )
    files = {}

    def bind_file(path, records):
        path = Path(path)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to(root)
        ):
            raise ValueError("unsafe retained mixture stage file")
        matches = [item for item in records if item.get("path") == str(path)]
        if len(matches) != 1:
            raise ValueError("retained mixture stage file not uniquely recorded")
        item = matches[0]
        if (
            not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or item.get("bytes") != path.stat().st_size
        ):
            raise ValueError("invalid retained mixture stage file identity")
        relative = path.relative_to(root).as_posix()
        value = {"sha256": item["sha256"], "bytes": item["bytes"]}
        if relative in files and files[relative] != value:
            raise ValueError("conflicting retained mixture stage file identities")
        files[relative] = value

    def bind_evidence(evidence):
        for key in ("completion_path", "cache_path", "proof_path"):
            if not evidence.get(key):
                continue
            path = Path(evidence[key])
            if (
                not path.is_absolute()
                or not path.is_relative_to(root)
                or path.is_symlink()
            ):
                raise ValueError("unsafe retained mixture evidence path")
            digest = sha256_file(path)
            if key == "cache_path" and digest != evidence["cache_sha256"]:
                raise ValueError("retained mixture stage cache changed")
            if key == "proof_path" and digest != evidence["historical_proof_sha256"]:
                raise ValueError("retained mixture stage proof changed")
            if key == "completion_path":
                event = json.loads(_small(path, root))
                if (
                    event.get("sha256") != evidence["completion_sha256"]
                    or canonical_digest(event["record"]) != event["sha256"]
                ):
                    raise ValueError("retained mixture stage completion changed")
            files[path.relative_to(root).as_posix()] = {
                "sha256": digest,
                "bytes": path.stat().st_size,
            }

    for value, field in (
        (refs, "references"),
        (dedupe, "parents"),
        (origins, "origins"),
    ):
        bind_file(value[field], value["files"])
    for evidence in (refs_proof, dedupe_proof, origins_proof):
        bind_evidence(evidence)
    sources, source_proofs = [], {}
    stage_ids = identity["completed_preparation"].get("source_stage_ids")
    for source_id in sorted(identity["lock"]):
        source_root = workspace / "source-build-stages" / source_id
        events = (
            list(source_root.glob("*/completion-*.json"))
            if stage_ids is None
            else list((source_root / stage_ids[source_id]).glob("completion-*.json"))
        )
        if len(events) != 1:
            raise ValueError("retained source render is missing or ambiguous")
        payload, proof = _phase(
            root,
            events[0].parent,
            "native-shard-render",
            expected_producer=request["producer"],
            include_cache_hash=True,
        )
        prepared = Path(payload["prepared"])
        receipt_path = prepared / "receipt.json"
        receipt = json.loads(_small(receipt_path, root))
        row = receipt["identity"]
        provenance = row if stage_ids is None else receipt["provenance"]
        if (
            stage_ids is None
            and row["admission_sha256"]
            != identity["release"]["record_admission"]["sha256"]
        ):
            raise ValueError("retained source admission identity differs")
        if (
            row["source_id"] != source_id
            or provenance["build_id"] != events[0].parent.name
            or row["snapshot_sha256"] != identity["lock"][source_id]["snapshot_sha256"]
        ):
            raise ValueError("retained source render identity differs")
        docs = prepared / "docs.jsonl"
        bind_file(docs, payload["files"])
        bind_file(receipt_path, payload["files"])
        bind_evidence(proof)
        sources.append(str(docs))
        source_proofs[source_id] = proof
    return {
        "documents": refs["documents"],
        "references": refs["references"],
        "parents": dedupe["parents"],
        "origins": origins["origins"],
        "sources": sources,
        "source_ids": sorted(identity["lock"]),
        "evidence": {
            "references": refs_proof,
            "deduplication": dedupe_proof,
            "origins": origins_proof,
            "sources": source_proofs,
        },
        "files": files,
    }


def prepared_release(root, release):
    """Resolve executed reference and source offsets for one retained release."""
    root, release = Path(root).resolve(strict=True), Path(release).resolve(strict=True)
    if not release.is_relative_to(root):
        raise ValueError("retained mixture release escapes work root")
    manifest = verify_retained_release(release)
    migration = root / "release-migrations" / (manifest["release_id"] + ".target.json")
    if migration.exists() or migration.is_symlink():
        from sparselab.corpus.release_migration import verify_mixture_stage_migration

        return verify_mixture_stage_migration(release, manifest)
    identity = manifest["build_identity"]
    request = identity["completed_preparation"]["request"]
    workspace = release.parent.parent
    stage_root = workspace / "build-stages" / canonical_digest(request)
    refs, refs_proof = _phase(
        root,
        stage_root,
        "build-frozen-family-bindings",
        expected_producer=request["producer"],
    )
    dedupe, dedupe_proof = _phase(
        root, stage_root, "build-deduplicate", expected_producer=request["producer"]
    )
    origins, origins_proof = _phase(
        root, stage_root, "build-duplicate-keys", expected_producer=request["producer"]
    )
    sources, source_proofs = [], {}
    for source_id in sorted(identity["lock"]):
        source_root = workspace / "source-build-stages" / source_id
        stage_ids = identity["completed_preparation"].get("source_stage_ids")
        if stage_ids is None:
            # Historical receipts precede exact stage selectors. Retain their
            # original unambiguous selection and global admission binding.
            events = list(source_root.glob("*/completion-*.json"))
        else:
            events = list(
                (source_root / stage_ids[source_id]).glob("completion-*.json")
            )
        if len(events) != 1:
            raise ValueError("retained source render is missing or ambiguous")
        payload, proof = _phase(
            root,
            events[0].parent,
            "native-shard-render",
            expected_producer=request["producer"],
        )
        prepared = Path(payload["prepared"])
        receipt = json.loads(_small(prepared / "receipt.json", root))
        row = receipt["identity"]
        provenance = row if stage_ids is None else receipt["provenance"]
        if (
            stage_ids is None
            and row["admission_sha256"]
            != identity["release"]["record_admission"]["sha256"]
        ):
            raise ValueError("retained source admission identity differs")
        if (
            row["source_id"] != source_id
            or provenance["build_id"] != events[0].parent.name
            or row["snapshot_sha256"] != identity["lock"][source_id]["snapshot_sha256"]
        ):
            raise ValueError("retained source render identity differs")
        docs = prepared / "docs.jsonl"
        if not any(item["path"] == str(docs) for item in payload["files"]):
            raise ValueError("retained source document output unavailable")
        sources.append(str(docs))
        source_proofs[source_id] = proof
    return {
        "release_id": manifest["release_id"],
        "release_path": str(release),
        "manifest": manifest,
        "references": refs["references"],
        "documents": refs["documents"],
        "parents": dedupe["parents"],
        "origins": origins["origins"],
        "sources": sources,
        "source_ids": sorted(identity["lock"]),
        "evidence": {
            "references": refs_proof,
            "deduplication": dedupe_proof,
            "origins": origins_proof,
            "sources": source_proofs,
        },
    }


def verify_additive_anchor(base, delta):
    """Require the delta split to have been built against this prior release."""
    if base["release_id"] == delta["release_id"]:
        raise ValueError("additive release duplicates its base")
    inputs = delta["manifest"]["build_identity"]["completed_preparation"]["request"][
        "inputs"
    ]
    matches = []
    migrated = None
    if delta.get("stage_migration_receipt"):
        from sparselab.corpus.release_migration import migrated_bound_inputs

        migrated = migrated_bound_inputs(Path(delta["release_path"]), delta["manifest"])
    for name in inputs:
        if not name.endswith(".receipt.json"):
            continue
        entry = migrated.get(name) if migrated is not None else None
        if migrated is not None and entry is None:
            raise ValueError("additive split receipt missing from migration evidence")
        path = Path(entry["path"] if entry is not None else name)
        if path.is_file() and path.stat().st_size <= 4 * 1024**2:
            before = capture([path])
            if entry is None and before.get(name) != inputs[name]:
                raise ValueError("additive split receipt fingerprint differs")
            raw = path.read_bytes()
            if entry is not None and hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("additive split receipt digest differs")
            receipt = json.loads(raw)
            unchanged(before)
            if receipt.get("prior_family_inventory"):
                matches.append((path, receipt["prior_family_inventory"]))
    if len(matches) != 1:
        raise ValueError("additive release lacks a unique prior split binding")
    path, prior = matches[0]
    if (
        prior.get("release_id") != base["release_id"]
        or prior.get("inventory_sha256")
        != base["manifest"]["files"]["family-inventory.jsonl"]["sha256"]
    ):
        raise ValueError("additive release prior split differs from base")
    return {
        "release_id": delta["release_id"],
        "split_receipt": str(path),
        "prior": prior,
    }


def _cross_release_isolation(directory, cohorts):
    """Merge signed duplicate-key indexes; refuse cross-release conflicts."""

    def tagged(index, path):
        for row in rows(path):
            if len(row) != 6 or row[0] not in (0, 1):
                raise ValueError("invalid retained duplicate-key row")
            yield (row[0], row[1], index, row[4], row[5])

    streams = [tagged(i, cohort["origins"]) for i, cohort in enumerate(cohorts)]
    merged = heapq.merge(*streams, key=lambda row: row[:2])
    overlaps = 0
    for (method, _key), values in itertools.groupby(merged, key=lambda row: row[:2]):
        owners, splits = set(), set()
        first_signature, incompatible = None, False
        for _, _, owner, split, signature in values:
            if split not in {"train", "validation", "test"}:
                raise ValueError("invalid retained duplicate-key split")
            owners.add(owner)
            splits.add(split)
            if method == 1:
                if first_signature is None:
                    first_signature = signature
                elif signature != first_signature:
                    incompatible = True
        if len(owners) < 2:
            continue
        if len(splits) != 1:
            raise ValueError("cross-release exact text overlaps different splits")
        if incompatible:
            raise ValueError("cross-release duplicate text has incompatible rights")
        overlaps += 1
    item = write_rows(
        directory / "cross-release-isolation.jsonl", [{"overlaps": overlaps}]
    )
    return {"files": [item], "overlaps": overlaps}


def _ranked(directory, cohorts, seed, strata, order_policy=None):
    """Sort kept TRAIN offsets from prior native metadata, not document bodies."""
    from sparselab.corpus.mixture import _rank

    def candidates():
        for cohort_index, cohort in enumerate(cohorts):
            count = cohort["documents"]
            with Path(cohort["parents"]).open("rb") as stream:
                if Path(cohort["parents"]).stat().st_size != count * 8:
                    raise ValueError("retained deduplication index size differs")
                if not count:
                    continue
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as parents:
                    seen = 0
                    for ordinal, row in enumerate(rows(cohort["references"])):
                        if len(row) != 13 or ordinal >= count:
                            raise ValueError("invalid retained mixture reference")
                        seen += 1
                        current = ordinal
                        while (
                            parent := struct.unpack_from("<Q", parents, current * 8)[0]
                        ) != current:
                            if parent >= current:
                                raise ValueError(
                                    "invalid retained deduplication parent"
                                )
                            current = parent
                        if current != ordinal or row[8] or row[4] != "train":
                            continue
                        shard = row[1]
                        if type(shard) is not int or not 0 <= shard < len(
                            cohort["sources"]
                        ):
                            raise ValueError("retained source index differs")
                        stratum, family, doc_id = row[12], row[11], row[0]
                        source_id = cohort["source_ids"][shard]
                        if strata.get(source_id) != stratum or stratum not in row[10]:
                            raise ValueError("retained source stratum differs")
                        yield [
                            stratum,
                            _rank(seed, stratum, family, doc_id, order_policy),
                            doc_id,
                            cohort_index,
                            shard,
                            row[2],
                            row[3],
                            row[6],
                            family,
                            source_id,
                        ]
                    if seen != count:
                        raise ValueError("retained reference count differs")

    ordered = external_sort(candidates(), directory / "ranked")
    return {"files": [ordered], "ranked": ordered["path"], "rows": ordered["rows"]}


class RetainedMixtureIndex(MixtureIndex):
    """Use signed source offsets and a bounded file-handle cache."""

    def cached_document(self, doc_id, source, content):
        # Ranked references do not carry per-record rights. A content-cache hit
        # from another document cannot authorize this record: read and check it.
        return None

    def candidate_bytes(self, source_bytes, content):
        return source_bytes

    def record_family(self, stratum, family):
        """Count an eligible family without retaining a second token copy."""
        self.record_event(["family", [stratum, family]])
        self.db.execute(
            "INSERT INTO encoded(offset,family,stratum) VALUES(NULL,?,?)",
            (family, stratum),
        )

    def __init__(self, scratch, work_root, cohorts, seed, strata, order_policy=None):
        if not 1 <= len(cohorts) <= 2:
            raise ValueError("at most one additional release is supported")
        self.work_root = Path(work_root)
        self.cohorts = cohorts
        self.handles = OrderedDict()
        self.handle_lock = threading.Lock()
        total_sources = sum(len(cohort["sources"]) for cohort in cohorts)
        if not total_sources:
            raise ValueError("retained mixture has no source renders")
        try:
            import resource

            limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            available = (
                total_sources if limit == resource.RLIM_INFINITY else max(1, limit - 64)
            )
        except ImportError, OSError, ValueError:
            available = 64
        self.handle_limit = min(total_sources, available, 64)
        paths = []
        for cohort in cohorts:
            if cohort.get("stage_migration_receipt"):
                paths.append(cohort["stage_migration_receipt"])
            paths.extend(
                [
                    cohort["references"],
                    cohort["parents"],
                    cohort["origins"],
                    *cohort["sources"],
                ]
            )
            if "release_path" in cohort:
                release = Path(cohort["release_path"])
                paths.extend(
                    [
                        release / "manifest.json",
                        release / "release-publication.json",
                        release / "sources.json",
                        release / "report.json",
                    ]
                )
            if cohort.get("anchor"):
                paths.append(cohort["anchor"]["split_receipt"])
            proofs = cohort["evidence"]
            for proof in [
                proofs["references"],
                proofs["deduplication"],
                proofs["origins"],
                *proofs["sources"].values(),
            ]:
                paths.extend([proof["completion_path"], proof["cache_path"]])
                if proof.get("proof_path"):
                    paths.append(proof["proof_path"])
        self.dependencies = capture(paths)
        request = {
            "producer": {
                "retained_mixture.py": sha256_file(Path(__file__)),
                "split_runs.py": sha256_file(Path(__file__).with_name("split_runs.py")),
                "mixture.py": sha256_file(Path(__file__).with_name("mixture.py")),
            },
            "releases": [cohort["release_id"] for cohort in cohorts],
            "evidence": [cohort["evidence"] for cohort in cohorts],
            "seed": seed,
            "order_policy": order_policy,
            "source_strata": strata,
            "inputs": self.dependencies,
        }
        stages = SplitStages(
            self.work_root
            / "corpora"
            / cohorts[0]["manifest"]["corpus_id"]
            / "mixture-reference-stages",
            request,
            work_root=self.work_root,
        )
        if stages.cache.proofs is None or not stages.cache.writable:
            raise ValueError(
                "retained mixture references require native completion storage"
            )
        isolation = stages.phase(
            "cross-release-isolation", lambda d: _cross_release_isolation(d, cohorts)
        )
        result = stages.phase(
            "ranked-train-references",
            lambda d: _ranked(d, cohorts, seed, strata, order_policy),
        )
        self.ranked = result["ranked"]
        self.records = iter(rows(self.ranked))
        self.current = next(self.records, None)
        self.reference_evidence = {
            "request_sha256": stages.identity,
            "reused": "ranked-train-references" in stages.reused,
            "cross_release_overlaps": isolation["overlaps"],
        }
        self.tokens = (Path(scratch) / "encoded.jsonl").open("xb+")
        try:
            self.db = sqlite3.connect(Path(scratch) / "mixture.sqlite")
            self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA temp_store=FILE;
            PRAGMA journal_mode=OFF;
            CREATE TABLE encoded(ordinal INTEGER PRIMARY KEY, offset INTEGER,
                family TEXT, stratum TEXT);
            CREATE TABLE seen(content TEXT PRIMARY KEY);
            CREATE INDEX encoded_order ON encoded(stratum,ordinal);
            CREATE INDEX encoded_families ON encoded(stratum,family);
            """)
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.tokens.close()
            self.records.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.records.close()
        for handle in self.handles.values():
            handle.close()
        self.tokens.close()
        self.db.close()

    def _read_document(self, row):
        stratum, _, doc_id, cohort, shard, offset, size, content, _family, source = row
        if (
            type(offset) is not int
            or type(size) is not int
            or offset < 0
            or not 0 < size <= MAX_ROW_BYTES
        ):
            raise ValueError("invalid retained document offset")
        # Duplicate under the cache lock so eviction cannot close an active
        # reader's descriptor. pread keeps parallel readers' offsets independent.
        key = (cohort, shard)
        with self.handle_lock:
            stream = self.handles.pop(key, None)
            if stream is None:
                if len(self.handles) >= self.handle_limit:
                    self.handles.popitem(last=False)[1].close()
                stream = Path(self.cohorts[cohort]["sources"][shard]).open("rb")  # noqa: SIM115 - owned by the bounded index cache
            self.handles[key] = stream
            descriptor = os.dup(stream.fileno())
        try:
            raw = os.pread(descriptor, size, offset)
        finally:
            os.close(descriptor)
        if len(raw) != size:
            raise ValueError("retained source document changed")
        doc = json.loads(raw)
        if (
            doc.get("document_id") != doc_id
            or doc.get("source_id") != source
            or doc.get("content_sha256") != content
            or stratum not in doc.get("domains", [])
            or doc.get("rights", {}).get("training_eligibility")
            not in {"eligible", "eligible_with_obligations"}
        ):
            raise ValueError(
                "retained source document disagrees with executed metadata"
            )
        doc["split"] = "train"
        doc["drop_reason"] = None
        return doc

    def candidates(self, stratum, frontier=None, *, workers=1):
        from sparselab.corpus.mixture_pipeline import ordered_fetch

        if frontier is None:
            self.db.execute("DELETE FROM seen")

        # Byte cursors refer to the authenticated ranked reference file, not
        # source bodies. Resume never parses the committed reference prefix.
        def references():
            with Path(self.ranked).open("rb") as stream:
                stream.seek(frontier or getattr(self, "frontier", 0))
                while raw := stream.readline(MAX_ROW_BYTES + 1):
                    if len(raw) > MAX_ROW_BYTES:
                        raise ValueError("mixture reference-byte cap exceeded")
                    row = json.loads(raw)
                    if row[0] != stratum:
                        break
                    yield row, stream.tell()

        for item, doc in ordered_fetch(
            references(),
            lambda item: self._read_document(item[0]),
            workers=workers,
            size=lambda item: self.candidate_bytes(item[0][6], item[0][7]),
            metrics=getattr(self, "pipeline_metrics", None),
            cached=lambda item: self.cached_document(
                item[0][2], item[0][9], item[0][7]
            ),
        ):
            self.frontier = item[1]
            yield doc, item[0][8]

    def verify_inputs(self):
        unchanged(self.dependencies)
