"""Compile verified split-scoped relations into existing local_text inputs."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Annotated, Literal

from pydantic import Field, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.corpus.build_stages import implementation_digests
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
_MAX_SEQUENCE_BYTES = 8 * 1024**2
declare_stage(
    "relational sequence compilation",
    Single(
        "one ordered writer streams split partitions from the read-only adjacency index"
    ),
)


class CompileRelations(StrictModel):
    target: Literal["triples", "walks"]
    qualifiers: Literal["include", "omit"] = "include"
    seed: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)] | None = None
    hops: Annotated[int, Field(ge=1, le=64, strict=True)] | None = None
    walks_per_entity: Annotated[int, Field(ge=1, le=64, strict=True)] | None = None
    max_sequences: Annotated[int, Field(gt=0, strict=True)]
    max_bytes: Annotated[int, Field(gt=0, le=2**40, strict=True)]

    @model_validator(mode="after")
    def walk_options(self):
        walk_parameters = (self.seed, self.hops, self.walks_per_entity)
        if self.target == "walks" and None in walk_parameters:
            raise ValueError("walks need seed, hops and walks_per_entity")
        if self.target == "triples" and any(
            value is not None for value in walk_parameters
        ):
            raise ValueError("seed, hops and walks_per_entity apply only to walks")
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


def _serialize(rows, options):
    triples = []
    assertion_ids = []
    omitted = size = 0
    for row in rows:
        record = assertion_record(row)
        triple = [record["subject"], record["predicate"], record["value"]]
        if options.qualifiers == "include" and record["qualifiers"]:
            triple.append(record["qualifiers"])
        else:
            omitted += len(record["qualifiers"])
        classes = claim_classes(row)
        if classes != ["explicit"]:
            triple.append({"claim_classes": classes})
        raw = canonical_json(triple)
        size += len(raw) + 1
        if size > min(options.max_bytes, _MAX_SEQUENCE_BYTES):
            raise ValueError("a compiled sequence exceeds the per-sequence byte limit")
        triples.append(raw)
        assertion_ids.append(row["assertion_id"])
    representation = (
        triples[0] if options.target == "triples" else b"[" + b",".join(triples) + b"]"
    )
    return representation.decode(), assertion_ids, omitted


def _write_sequences(db, staging, result, options, coverage):
    sequences = payload_bytes = 0
    partitions = {}
    for index, split in enumerate(SPLITS):
        count = steps = omitted = 0
        selected = (
            _triples(db, index)
            if options.target == "triples"
            else _walks(db, index, options)
        )
        with (
            progress_phase(
                "corpus_relation_compile", unit="sequences", derived={"split": split}
            ) as progress,
            (staging / f"{split}.jsonl").open("wb") as text,
            (staging / f"{split}.lineage.jsonl").open("wb") as lineage,
        ):
            for rows in selected:
                value, assertion_ids, omissions = _serialize(rows, options)
                data = canonical_json({"text": value}) + b"\n"
                evidence = canonical_json({"assertions": assertion_ids}) + b"\n"
                payload_bytes += len(data) + len(evidence)
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
                count += 1
                steps += len(assertion_ids)
                omitted += omissions
                if count % 8192 == 0:
                    progress.update(completed_work=count, emit=False)
            progress.update(completed_work=count)
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
        inputs = {
            "implementation": implementation_digests("relation_compile.py"),
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
            require_storage([check_storage(staging, projected_bytes=options.max_bytes)])
            _write_sequences(
                db, staging, result, options, receipt["result"]["coverage"]
            )
            # Check the graph generation before the compiled result can be published.
            verified.close()

        return publish_relation_stage(store, COMPILED, inputs, produce)
