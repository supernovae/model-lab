"""Compile verified split-scoped relations into existing local_text inputs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import ExitStack, nullcontext
from typing import Annotated, Literal

from pydantic import Field, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.corpus.build_stages import implementation_digests
from klab.corpus.relation_sources import Identifier
from klab.corpus.relation_structure import (
    encode_structure,
    expand_structure,
    structural_groups,
    structural_scratch,
)
from klab.corpus.relations import (
    ASSERTION_COLUMNS,
    ASSERTION_JOINS,
    INDEX,
    SPLITS,
    assertion_record,
    claim_classes,
    open_relations,
    publish_relation_stage,
    relation_store,
)
from klab.corpus.shards import Single, declare_stage
from klab.progress import progress_phase
from klab.workspace_preflight import check_storage, require_storage

COMPILED = "relation-sequences"
MAX_RELATION_SEQUENCE_BYTES = 8 * 1024**2
declare_stage(
    "relational sequence compilation",
    Single(
        "one ordered writer streams split partitions from the read-only adjacency index"
    ),
)


class CompileRelations(StrictModel):
    target: Literal["triples", "walks", "hierarchy", "factorized"]
    qualifiers: Literal["include", "omit"] = "include"
    seed: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)] | None = None
    hops: Annotated[int, Field(ge=1, le=64, strict=True)] | None = None
    walks_per_entity: Annotated[int, Field(ge=1, le=64, strict=True)] | None = None
    max_sequences: Annotated[int, Field(gt=0, strict=True)]
    max_bytes: Annotated[int, Field(gt=0, le=2**40, strict=True)]

    group_size: Annotated[int, Field(ge=2, le=1024, strict=True)] | None = None
    max_scratch_bytes: (
        Annotated[int, Field(ge=1024**2, le=2**40, strict=True)] | None
    ) = None
    hierarchy_predicate: Identifier | None = None
    max_depth: Annotated[int, Field(ge=1, le=4096, strict=True)] | None = None

    @property
    def structural(self):
        return self.target in {"hierarchy", "factorized"}

    @model_validator(mode="after")
    def target_options(self):
        walk_parameters = (self.seed, self.hops, self.walks_per_entity)
        if self.target == "walks" and None in walk_parameters:
            raise ValueError("walks need seed, hops and walks_per_entity")
        if self.target != "walks" and any(
            value is not None for value in walk_parameters
        ):
            raise ValueError("seed, hops and walks_per_entity apply only to walks")
        structure = (self.group_size, self.max_scratch_bytes)
        hierarchy = (self.hierarchy_predicate, self.max_depth)
        for enabled, values, names in (
            (self.structural, structure, "group_size and max_scratch_bytes"),
            (
                self.target == "hierarchy",
                hierarchy,
                "hierarchy_predicate and max_depth",
            ),
        ):
            if enabled and None in values:
                raise ValueError(f"{self.target} requires {names}")
            if not enabled and any(value is not None for value in values):
                raise ValueError(f"{self.target} does not support {names}")
        return self


def _triples(db, split):
    for row in db.execute(
        f"""SELECT {ASSERTION_COLUMNS} FROM assertions a {ASSERTION_JOINS}
        WHERE a.split=? ORDER BY a.id""",
        (split,),
    ):
        yield [row]


def _walks(db, split, options):
    starts = db.execute(
        """SELECT e.id,e.name FROM entities e JOIN degrees d ON d.subject=e.id
        WHERE e.split=? ORDER BY e.id""",
        (split,),
    )
    for start in starts:
        for walk in range(options.walks_per_entity):
            yield _follow(db, split, options, start, walk)


def _follow(db, split, options, start, walk):
    current = start["id"]
    for hop in range(options.hops):
        degree = db.execute(
            "SELECT degree FROM degrees WHERE subject=?", (current,)
        ).fetchone()
        if degree is None:
            return
        slot = (
            int(canonical_digest([options.seed, start["name"], walk, hop]), 16)
            % degree["degree"]
        )
        row = db.execute(
            f"""SELECT {ASSERTION_COLUMNS},a.object AS next
            FROM neighbors n JOIN assertions a ON n.assertion=a.id {ASSERTION_JOINS}
            WHERE n.subject=? AND n.slot=? AND a.split=?""",
            (current, slot, split),
        ).fetchone()
        if row is None:
            raise ValueError("compiled walk left the verified split index")
        yield row
        current = row["next"]


def _serialize(rows, options, tag):
    triples = []
    assertion_ids = []
    omissions = []
    size = 0
    for row in rows:
        record = assertion_record(row)
        triple = [record["subject"], record["predicate"], record["value"]]
        if record["qualifiers"]:
            if options.qualifiers == "include":
                triple.append(record["qualifiers"])
            else:
                omissions.append(
                    {
                        "assertion": row["assertion_id"],
                        "qualifiers": len(record["qualifiers"]),
                    }
                )
        classes = claim_classes(row)
        if classes != ["explicit"]:
            triple.append({"claim_classes": classes})
        raw = canonical_json(triple)
        size += len(raw) + 1
        if size > min(options.max_bytes, MAX_RELATION_SEQUENCE_BYTES):
            raise ValueError("a compiled sequence exceeds the per-sequence byte limit")
        triples.append(raw)
        assertion_ids.append(row["assertion_id"])
    if tag:
        expected = [json.loads(triple) for triple in triples]
        representation = canonical_json(encode_structure(tag, expected))
        if list(expand_structure(json.loads(representation))) != expected:
            raise ValueError("structural compilation failed round-trip verification")
    elif options.target == "walks":
        representation = b"[" + b",".join(triples) + b"]"
    else:
        representation = triples[0]
    return representation.decode(), assertion_ids, omissions


def _write_sequences(db, staging, result, options, coverage, ledger):
    sequences = payload_bytes = 0
    partitions = {}
    for index, split in enumerate(SPLITS):
        count = steps = omitted = 0
        if options.structural:
            selected = structural_groups(
                db, index, options, MAX_RELATION_SEQUENCE_BYTES
            )
        else:
            groups = (
                _triples(db, index)
                if options.target == "triples"
                else _walks(db, index, options)
            )
            selected = ((rows, ()) for rows in groups)
        with (
            progress_phase(
                "corpus_relation_compile", unit="sequences", derived={"split": split}
            ) as progress,
            (staging / f"{split}.jsonl").open("wb") as text,
            (staging / f"{split}.lineage.jsonl").open("wb") as lineage,
        ):
            for rows, tag in selected:
                value, assertion_ids, omissions = _serialize(rows, options, tag)
                data = canonical_json({"text": value}) + b"\n"
                evidence = canonical_json({"assertions": assertion_ids}) + b"\n"
                omission = b""
                if omissions:
                    omission = (
                        canonical_json(
                            {
                                "split": split,
                                "sequence": count + 1,
                                "reason": "compiler_qualifiers_omit",
                                "assertions": omissions,
                            }
                        )
                        + b"\n"
                    )
                payload_bytes += len(data) + len(evidence) + len(omission)
                sequences += 1
                if (
                    payload_bytes > options.max_bytes
                    or sequences > options.max_sequences
                ):
                    raise ValueError(
                        "compiled sequences exceed the declared byte or sequence limit"
                    )
                text.write(data)
                lineage.write(evidence)
                if omission:
                    ledger.write(omission)
                count += 1
                steps += len(assertion_ids)
                omitted += sum(item["qualifiers"] for item in omissions)
                if count % 8192 == 0:
                    progress.update(completed_work=count, emit=False)
            progress.update(completed_work=count)
        if options.structural and steps != coverage[split]["assertions"]:
            raise ValueError(
                f"structural compilation changed assertion coverage in {split}"
            )
        partitions[split] = {
            "sequences": count,
            "assertion_visits": steps,
            "qualifier_visits_omitted": omitted,
            "literal_assertions_ineligible": coverage[split]["assertions"]
            - coverage[split]["edges"]
            if options.target == "walks"
            else 0,
        }
    result.update(
        partitions=partitions,
        sequences=sequences,
        payload_bytes=payload_bytes,
        fidelity=(
            "qualifier_projection"
            if any(row["qualifier_visits_omitted"] for row in partitions.values())
            else "assertion_round_trip"
        )
        if options.structural
        else "assertion_visits",
        scratch_bytes=result.get("scratch_bytes", 0),
        evidence_fields_omitted=[
            "source",
            "line",
            "location",
            "claim_class",
            "source_id",
            "rank",
            "references",
        ],
    )


def compile_relations(root, key, options, *, plan_only=False):
    store = relation_store(root)
    with ExitStack() as verified:
        db, receipt, _ = verified.enter_context(open_relations(root, key))
        implementation = ["relation_compile.py"]
        if options.structural:
            implementation.append("relation_structure.py")
        inputs = {
            "implementation": implementation_digests(*implementation),
            "index": {"stage": INDEX, "key": key, "files": receipt["files"]},
            "options": options.model_dump(mode="json"),
        }
        if plan_only:
            candidate = store.directory(COMPILED, store.key(inputs))
            return {
                "key": store.key(inputs),
                "path": str(candidate),
                "inputs": inputs,
                "candidate": candidate.is_dir(),
                "stages": ["relational sequence compilation"],
            }

        def produce(staging, result):
            scratch_bytes = options.max_scratch_bytes or 0
            require_storage(
                [
                    check_storage(
                        staging, projected_bytes=options.max_bytes + scratch_bytes
                    )
                ]
            )
            scratch = (
                structural_scratch(db, staging, scratch_bytes, result)
                if options.structural
                else nullcontext()
            )
            ledger = (
                (staging / "omissions.jsonl").open("wb")
                if options.qualifiers == "omit"
                else nullcontext()
            )
            try:
                with ledger as omissions, scratch:
                    _write_sequences(
                        db,
                        staging,
                        result,
                        options,
                        receipt["result"]["coverage"],
                        omissions,
                    )
            except sqlite3.Error as error:
                raise ValueError(
                    f"cannot compile the relation structure: {error}"
                ) from error
            # Check the graph generation before the compiled result can be published.
            verified.close()

        return publish_relation_stage(store, COMPILED, inputs, produce)
