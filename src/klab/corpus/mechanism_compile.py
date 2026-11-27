"""Compile verified procedural worlds and bridge mechanisms to Corpus packing."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import ExitStack, closing
from typing import Annotated, Literal

from pydantic import Field, model_validator

from klab.canonical import canonical_json
from klab.config.models import StrictModel
from klab.corpus.build_stages import implementation_digests
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.mechanisms import (
    WORLD_CATALOG,
    WORLD_STAGE,
    ByteLimit,
    configure_private_database,
    mechanism_store,
    observed_worlds,
)
from klab.corpus.relation_compile import COMPILED as RELATION_SEQUENCES
from klab.corpus.relation_sources import Digest
from klab.corpus.relations import (
    SPLITS,
    open_relations,
    publish_relation_stage,
    relation_store,
)
from klab.corpus.release import open_release
from klab.corpus.shards import Single, declare_stage
from klab.corpus.stage_inspection import open_verified_stage
from klab.progress import progress_phase
from klab.workspace_preflight import check_storage, require_storage

COMPILED_STAGE = "procedural-sequences"
MEMBERSHIP_STAGE = "mechanism-memberships"
COMPILE = declare_stage(
    "procedural curriculum compilation",
    Single("stream ordered observations into split files"),
)
MEMBERSHIPS = declare_stage(
    "mechanism membership join",
    Single("one pass over verified release family inventory"),
)

Target = Literal[
    "prose", "trajectories", "derivations", "constraints", "counterfactuals"
]
_MIN_DB_BYTES = 1024**2


class MechanismCompilation(StrictModel):
    targets: Annotated[tuple[Target, ...], Field(min_length=1, max_length=5)]
    max_sequences: Annotated[int, Field(strict=True, gt=0)]
    max_bytes: ByteLimit
    relation_keys: Annotated[tuple[Digest, ...], Field(max_length=4)] = ()

    @model_validator(mode="after")
    def unique_targets(self):
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("procedural curriculum targets must be distinct")
        if len(set(self.relation_keys)) != len(self.relation_keys):
            raise ValueError("procedural curriculum relation keys must be distinct")
        return self


def _structured(observed, target):
    world = observed.world
    events = []
    for outcome in observed.observations:
        event = {"step": outcome.step, "source": outcome.source}
        if target == "trajectories":
            event.update(outcome.actual.model_dump(mode="json"))
        else:
            event["before"] = outcome.before
            if target == "constraints":
                event["legal_maximum"] = outcome.legal_maximum
            elif target == "derivations":
                event.update(
                    available=outcome.available,
                    headroom=outcome.headroom,
                    legal_maximum=outcome.legal_maximum,
                    **outcome.actual.model_dump(mode="json"),
                )
            else:
                event["actual"] = outcome.actual.model_dump(mode="json")
                event["counterfactual"] = outcome.counterfactual.model_dump(mode="json")
        events.append(event)
    return canonical_json(
        {
            "target": target,
            "world": world.id,
            "rule": world.rule.model_dump(mode="json"),
            "initial": world.initial,
            "events": events,
        }
    ).decode()


def _verify_relational_scope(db, catalog):
    uri = f"{catalog.as_uri()}?mode=ro&immutable=1"
    db.execute("ATTACH DATABASE ? AS procedural", (uri,))
    try:
        split = "CASE w.split WHEN 'train' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END"
        for statement in (
            f"SELECT f.id FROM families f JOIN procedural.families w USING(id) WHERE f.split<>{split} LIMIT 1",
            f"""SELECT e.name FROM entities e JOIN procedural.worlds w ON e.name=w.id
            WHERE e.family<>w.family OR e.split<>{split} LIMIT 1""",
        ):
            if mismatch := db.execute(statement).fetchone():
                raise ValueError(
                    f"relational curriculum has a conflicting family or split for {mismatch[0]}"
                )
    finally:
        db.execute("DETACH DATABASE procedural")


def _open_relation_arms(verified, root, options):
    arms = []
    for key in options.relation_keys:
        path, receipt, _ = verified.enter_context(
            open_verified_stage(relation_store(root), RELATION_SEQUENCES, key)
        )
        index = receipt["inputs"]["index"]
        target = receipt["inputs"]["options"]["target"]
        if target not in {"hierarchy", "factorized"}:
            raise ValueError(
                "curriculum relation keys must refer to hierarchy or factorized outputs"
            )
        if arms and arms[0][1]["inputs"]["index"] != index:
            raise ValueError(
                "curriculum relation arms must use the same canonical index"
            )
        arms.append((path, receipt))
    return arms


def _curriculum_rows(store, receipt, options, relation_arms):
    for observed in observed_worlds(store, receipt):
        world = observed.world
        source = {
            "world": world.id,
            "family": world.family,
            "source": observed.source,
            "line": observed.line,
        }
        for target in options.targets:
            value = world.prose if target == "prose" else _structured(observed, target)
            evidence = {
                **source,
                "target": target,
                "mechanism": observed.mechanism,
                "origin": "source_prose" if target == "prose" else "deterministic_rule",
            }
            yield world.split, target, value, evidence
    for directory, parent in relation_arms:
        target = parent["inputs"]["options"]["target"]
        for split in SPLITS:
            rows = records_from_path(
                directory / f"{split}.jsonl", max_record_bytes=64 * 1024**2
            )
            lineage = records_from_path(
                directory / f"{split}.lineage.jsonl", max_record_bytes=1024**2
            )
            for text, evidence in zip(rows, lineage, strict=True):
                yield (
                    split,
                    target,
                    text.value["text"],
                    {
                        "target": target,
                        "origin": "compiled_relations",
                        "relation_key": parent["key"],
                        "line": text.physical_line,
                        "assertions": evidence.value["assertions"],
                    },
                )


def _write_curriculum(store, receipt, staging, options, result, relation_arms):
    targets = (
        *options.targets,
        *(arm["inputs"]["options"]["target"] for _, arm in relation_arms),
    )
    counts = {split: dict.fromkeys(targets, 0) for split in SPLITS}
    sequences = payload_bytes = omitted = 0
    with (
        ExitStack() as files,
        progress_phase("corpus_procedural_compile", unit="records") as progress,
    ):
        text = {
            split: files.enter_context((staging / f"{split}.jsonl").open("wb"))
            for split in SPLITS
        }
        lineage = {
            split: files.enter_context((staging / f"{split}.lineage.jsonl").open("wb"))
            for split in SPLITS
        }
        omissions = files.enter_context((staging / "omissions.jsonl").open("wb"))
        rows = _curriculum_rows(store, receipt, options, relation_arms)
        completed = 0
        for completed, (split, target, value, evidence) in enumerate(rows, 1):
            if value is None:
                skipped = (
                    canonical_json(
                        {**evidence, "split": split, "reason": "no_source_prose"}
                    )
                    + b"\n"
                )
                payload_bytes += len(skipped)
                if payload_bytes > options.max_bytes:
                    raise ValueError("procedural curriculum exceeds its byte budget")
                omissions.write(skipped)
                omitted += 1
                continue
            row = canonical_json({"text": value}) + b"\n"
            evidence = (
                canonical_json(
                    {
                        **evidence,
                        "content_sha256": hashlib.sha256(value.encode()).hexdigest(),
                    }
                )
                + b"\n"
            )
            sequences += 1
            payload_bytes += len(row) + len(evidence)
            if sequences > options.max_sequences or payload_bytes > options.max_bytes:
                raise ValueError(
                    "procedural curriculum exceeds its sequence or byte budget"
                )
            text[split].write(row)
            lineage[split].write(evidence)
            counts[split][target] += 1
            if completed % 8192 == 0:
                progress.update(completed_work=completed, emit=False)
        progress.update(completed_work=completed)
    result.update(
        counts=counts,
        sequences=sequences,
        payload_bytes=payload_bytes,
        missing_prose=omitted,
    )


def compile_mechanisms(root, key, options, *, plan_only=False):
    store = mechanism_store(root)
    with ExitStack() as verified:
        directory, receipt, _ = verified.enter_context(
            open_verified_stage(store, WORLD_STAGE, key)
        )
        relation_arms = _open_relation_arms(verified, root, options)
        graph = None
        if relation_arms:
            index = relation_arms[0][1]["inputs"]["index"]
            graph, graph_receipt, _ = verified.enter_context(
                open_relations(root, index["key"])
            )
            if graph_receipt["files"] != index["files"]:
                raise ValueError(
                    "curriculum relation index differs from its compiled parent"
                )
        inputs = {
            "implementation": implementation_digests("mechanism_compile.py"),
            "worlds": {
                "key": key,
                "files": receipt["files"],
                "shards": receipt["result"]["shards"],
            },
            "relations": [
                {"key": parent["key"], "files": parent["files"]}
                for _, parent in relation_arms
            ],
            "options": options.model_dump(mode="json"),
        }
        candidate = store.directory(COMPILED_STAGE, store.key(inputs))
        if plan_only:
            return {
                "key": candidate.name,
                "candidate": candidate.is_dir(),
                "inputs": inputs,
                "stages": [COMPILE],
            }

        def produce(staging, result):
            require_storage([check_storage(staging, projected_bytes=options.max_bytes)])
            if graph is not None:
                _verify_relational_scope(graph, directory / WORLD_CATALOG)
            _write_curriculum(store, receipt, staging, options, result, relation_arms)
            verified.close()

        return publish_relation_stage(store, COMPILED_STAGE, inputs, produce)


def _memberships(staging, train_lineage, inventory, member, options, result):
    index_path = staging / "members.sqlite"
    path = staging / "entities.jsonl"
    matches = entries = size = 0
    try:
        with closing(sqlite3.connect(index_path)) as db, db, path.open("wb") as output:
            configure_private_database(db, options["scratch"])
            db.execute(
                "CREATE TABLE labels (content BLOB, mechanism BLOB, PRIMARY KEY(content,mechanism)) WITHOUT ROWID"
            )
            db.executemany(
                "INSERT OR IGNORE INTO labels VALUES (?,?)",
                (
                    (
                        bytes.fromhex(row.value["content_sha256"]),
                        bytes.fromhex(row.value["mechanism"]),
                    )
                    for row in records_from_path(
                        train_lineage, max_record_bytes=1024**2
                    )
                    if "mechanism" in row.value
                ),
            )
            for record in records_from_path(
                inventory,
                max_record_bytes=8192
                if member["name"] == "family-inventory.jsonl"
                else 64 * 1024**2,
                expected_sha256=member["sha256"],
            ):
                row = record.value
                if row["split"] != "train" or row.get("drop_reason") is not None:
                    continue
                labels = db.execute(
                    "SELECT mechanism FROM labels WHERE content=? ORDER BY mechanism",
                    (bytes.fromhex(row["content_sha256"]),),
                )
                matched = False
                for (mechanism,) in labels:
                    raw = (
                        canonical_json(
                            {
                                "entity_id": f"mechanism:{mechanism.hex()}",
                                "document_id": row["document_id"],
                            }
                        )
                        + b"\n"
                    )
                    size += len(raw)
                    if size > options["output"]:
                        raise ValueError(
                            "mechanism membership output exceeds its byte budget"
                        )
                    output.write(raw)
                    entries += 1
                    matched = True
                matches += matched
            result.update(matched_documents=matches, memberships=entries, bytes=size)
    finally:
        index_path.unlink(missing_ok=True)


def mechanism_memberships(
    root, key, release_path, *, max_inventory_bytes, max_scratch_bytes, max_output_bytes
):
    if (
        max_inventory_bytes < 1
        or max_scratch_bytes < _MIN_DB_BYTES
        or max_output_bytes < 1
    ):
        raise ValueError(
            "membership needs positive inventory and output limits and at least 1 MiB of scratch"
        )
    options = {
        "inventory": max_inventory_bytes,
        "scratch": max_scratch_bytes,
        "output": max_output_bytes,
    }
    store = mechanism_store(root)
    with ExitStack() as verified:
        directory, compiled, _ = verified.enter_context(
            open_verified_stage(store, COMPILED_STAGE, key)
        )
        release = verified.enter_context(open_release(release_path))
        inventory_name = (
            "family-inventory.jsonl"
            if "family-inventory.jsonl" in release.manifest["files"]
            else "documents.jsonl"
        )
        inventory_meta = {
            "name": inventory_name,
            **release.manifest["files"][inventory_name],
        }
        inventory = release.member(inventory_name)
        if inventory.stat().st_size > max_inventory_bytes:
            raise ValueError(
                "release family inventory exceeds the mechanism join byte limit"
            )
        inputs = {
            "implementation": implementation_digests("mechanism_compile.py"),
            "compiled": {
                "key": key,
                "lineage": compiled["files"]["train.lineage.jsonl"],
            },
            "release": {
                "id": release.manifest["release_id"],
                "inventory": inventory_meta,
            },
            "options": options,
        }

        def produce(staging, result):
            require_storage(
                [
                    check_storage(
                        staging, projected_bytes=max_scratch_bytes + max_output_bytes
                    )
                ]
            )
            _memberships(
                staging,
                directory / "train.lineage.jsonl",
                inventory,
                inventory_meta,
                options,
                result,
            )
            release.require_unchanged()
            verified.close()

        return publish_relation_stage(store, MEMBERSHIP_STAGE, inputs, produce)
