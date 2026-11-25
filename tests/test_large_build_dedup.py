from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from klab.corpus.native_dedup import deduplicate
from klab.corpus.native_index import INDEX_SCHEMA, insert_indexed
from klab.corpus.project import DuplicatePolicy
from klab.corpus.view_checkpoints import ViewCheckpoints


class _Progress:
    def update(self, **_: object) -> None:
        pass

    def record(self, _: str) -> None:
        pass


def _database(documents: list[tuple[str, str, str, str]]) -> sqlite3.Connection:
    """(document_id, split, raw_sha, normalized_sha) rows."""
    connection = sqlite3.connect(":memory:")
    connection.executescript(INDEX_SCHEMA)
    rights = {
        "training_eligibility": "eligible",
        "license_references": [],
        "notices": [],
        "training_restriction": None,
    }
    for identifier, split, raw, normalized in documents:
        doc = {
            "document_id": identifier,
            "source_id": "s",
            "source_family": "f",
            "document_kind": "prose",
            "license": "MIT",
            "redistribution": "open",
            "rights": rights,
        }
        insert_indexed(
            connection,
            doc,
            (0, 0, 0, 0, 0),
            split,
            2,
            [("raw", raw), ("normalized", normalized)],
        )
    return connection


CHAIN = DuplicatePolicy(
    heldout_overlap="drop_train_copy",
    keys=("raw", "normalized"),
    resolution="chain_smallest_id",
    rights_conflict="error",
)
RANKED = DuplicatePolicy(
    heldout_overlap="drop_train_copy",
    keys=("raw", "normalized"),
    resolution="component_ranked",
    rights_conflict="allow",
)


def _run(
    connection: sqlite3.Connection, tmp_path: Path, policy=CHAIN, *, checkpoints=None
):
    return deduplicate(
        connection,
        tmp_path / "groups.jsonl",
        policy=policy,
        diagnostics=tmp_path / "diagnostics.json",
        progress=_Progress(),
        checkpoints=checkpoints,
    )


def _outcome(connection: sqlite3.Connection) -> dict[str, tuple[str, str | None]]:
    return {
        identifier: (representative, reason)
        for identifier, representative, reason in connection.execute(
            "SELECT document_id,representative_id,drop_reason FROM dedup"
        )
    }


def test_train_copies_of_held_out_text_are_dropped_not_the_held_out_document(
    tmp_path: Path,
) -> None:
    # The train id sorts before the held-out id: it must still be the one dropped.
    connection = _database(
        [
            ("a-train", "train", "r1", "n1"),
            ("b-valid", "validation", "r1", "n1"),
            ("c-train", "train", "r2", "n2"),
            ("d-train", "train", "r2", "n2"),
        ]
    )
    groups, dropped = _run(connection, tmp_path)
    assert _outcome(connection) == {
        "a-train": ("b-valid", "contaminated_heldout"),
        "b-valid": ("b-valid", None),
        "c-train": ("c-train", None),
        "d-train": ("c-train", "duplicate"),
    }
    assert (groups, dropped) == (4, 2)
    record = json.loads((tmp_path / "groups.jsonl").read_text().splitlines()[0])
    assert record["splits"] == ["train", "validation"]
    assert record["representative"] == "b-valid"


def test_text_shared_by_two_held_out_splits_still_fails(tmp_path: Path) -> None:
    connection = _database(
        [
            ("a-train", "train", "r1", "n1"),
            ("b-valid", "validation", "r1", "n1"),
            ("c-test", "test", "r1", "n1"),
        ]
    )
    with pytest.raises(ValueError, match="cross-split exact text overlap"):
        _run(connection, tmp_path)


def test_error_policy_rejects_any_cross_split_overlap(tmp_path: Path) -> None:
    connection = _database(
        [("a-train", "train", "r1", "n1"), ("b-valid", "validation", "r1", "n1")]
    )
    strict = CHAIN.model_copy(update={"heldout_overlap": "error"})
    with pytest.raises(ValueError, match="cross-split exact text overlap") as failure:
        _run(connection, tmp_path, strict)
    # The failure names the group's documents and the option that changes it.
    assert "a-train, b-valid" in str(failure.value)
    assert "release.policy.duplicates.heldout_overlap" in str(failure.value)


def test_chain_and_component_resolution_agree_on_a_simple_corpus(
    tmp_path: Path,
) -> None:
    rows = [
        ("a-train", "train", "r1", "n1"),
        ("b-valid", "validation", "r1", "n1"),
        ("c-train", "train", "r2", "n2"),
        ("d-train", "train", "r2", "n2"),
        ("e-train", "train", "r3", "n3"),
    ]
    outcomes = []
    checkpoint_root = tmp_path / "ranked-checkpoint"
    for index, policy in enumerate((CHAIN, RANKED)):
        directory = tmp_path / str(index)
        directory.mkdir()
        connection = _database(rows)
        checkpoints = (
            ViewCheckpoints(checkpoint_root, "ranked") if policy is RANKED else None
        )
        _run(connection, directory, policy, checkpoints=checkpoints)
        outcomes.append(_outcome(connection))
    assert outcomes[0] == outcomes[1]

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    connection = _database(rows)
    _run(
        connection,
        resumed,
        RANKED,
        checkpoints=ViewCheckpoints(checkpoint_root, "ranked"),
    )
    assert _outcome(connection) == outcomes[1]
    assert (resumed / "groups.jsonl").read_bytes() == (
        tmp_path / "1/groups.jsonl"
    ).read_bytes()


# --- parity: the in-memory and disk-backed builders must decide identically ---

import itertools
import random

from klab.corpus.dedup_policy import (
    DuplicateError,
    deduplicate_documents,
    drop_reason,
)
from klab.corpus.duplicate_evidence import is_metadata_only

_RIGHTS = {
    "training_eligibility": "eligible",
    "license_references": [],
    "notices": [],
    "training_restriction": None,
}


def _corpus(seed: int) -> list[dict]:
    rng = random.Random(seed)
    spread = 20 + seed * 15  # later seeds collide less, so every outcome occurs
    docs = []
    for number in range(40):
        splits = ["train"] * 8 + ["validation", "test"]
        docs.append(
            {
                "document_id": f"d{number:03}",
                "split": rng.choice(splits),
                "source_id": rng.choice(["s0", "s1"]),
                "document_kind": rng.choice(["prose", "paper"]),
                "raw_content_sha256": f"r{rng.randrange(spread)}",
                "content_sha256": f"n{rng.randrange(spread)}",
                "origin": f"o{rng.randrange(spread)}",
                "license": "MIT",
                "redistribution": rng.choice(["open"] * 9 + ["closed"]),
                "rights": _RIGHTS,
                "structure": {
                    "decision": rng.choice(["keep"] * 12 + ["exclude_lm_metadata_only"])
                },
            }
        )
    return docs


def _disk_database(docs: list[dict], keys: tuple[str, ...]) -> sqlite3.Connection:
    connection = _database([])
    for doc in docs:
        record = {
            **doc,
            "source_family": "f",
        }
        origins = {
            "raw": doc["raw_content_sha256"],
            "normalized": doc["content_sha256"],
            "canonical_origin": doc["origin"],
        }
        insert_indexed(
            connection,
            record,
            (0, 0, 0, 0, 0),
            doc["split"],
            {"s0": 0, "s1": 2}[doc["source_id"]],
            [(method, origins[method]) for method in keys],
        )
    return connection


_POLICIES = [
    DuplicatePolicy(
        heldout_overlap=overlap,
        keys=keys,
        resolution="chain_smallest_id",
        rights_conflict=conflict,
    )
    for overlap, keys, conflict in itertools.product(
        ("error", "drop_train_copy"),
        (("raw",), ("raw", "normalized")),
        ("error", "allow"),
    )
] + [
    DuplicatePolicy(
        heldout_overlap="drop_train_copy",
        keys=keys,
        resolution="component_ranked",
        rights_conflict="allow",
    )
    for keys in (("raw", "normalized"), ("raw", "normalized", "canonical_origin"))
]


@pytest.mark.parametrize(
    "policy",
    _POLICIES,
    ids=lambda p: (
        f"{p.resolution}-{p.heldout_overlap}-{'+'.join(p.keys)}-{p.rights_conflict}"
    ),
)
def test_memory_and_disk_builders_decide_identically(
    policy: DuplicatePolicy, tmp_path: Path
) -> None:
    ranks = {"s0": 0, "s1": 2}
    compared = errors = 0
    for seed in range(40):
        docs = _corpus(seed)
        directory = tmp_path / str(seed)
        directory.mkdir()
        try:
            groups, representative = deduplicate_documents(
                docs, ranks, policy, lambda doc: (doc["origin"],)
            )
            memory = {
                doc["document_id"]: (
                    representative[doc["document_id"]],
                    drop_reason(
                        metadata_only=is_metadata_only(doc),
                        split=doc["split"],
                        is_representative=representative[doc["document_id"]]
                        == doc["document_id"],
                        representative_split=next(
                            other["split"]
                            for other in docs
                            if other["document_id"]
                            == representative[doc["document_id"]]
                        ),
                    ),
                )
                for doc in docs
            }
            memory_error = None
        except DuplicateError as error:
            memory, memory_error = None, str(error)
        connection = _disk_database(docs, policy.keys)
        try:
            _run(connection, directory, policy)
            disk, disk_error = _outcome(connection), None
        except ValueError as error:
            disk, disk_error = None, str(error)
        assert memory_error == disk_error, seed
        if memory_error is None:
            assert memory == disk, seed
            on_disk = [
                json.loads(line)
                for line in (directory / "groups.jsonl").read_text().splitlines()
            ]
            assert on_disk == groups, seed
            compared += 1
        else:
            errors += 1
    # The corpora exercise both outcomes under every policy.
    assert compared + errors == 40
    assert compared >= 1


def test_a_declared_source_without_documents_fails_the_build() -> None:
    from types import SimpleNamespace

    from klab.corpus.large_build import _require_every_source_has_documents

    connection = _database(
        [("a-train", "train", "r1", "n1"), ("b-train", "train", "r2", "n2")]
    )
    # _database records every document under the source "s".
    declared = SimpleNamespace(
        sources=[SimpleNamespace(id="s"), SimpleNamespace(id="code_python")]
    )
    events = [
        {
            "source_id": "code_python",
            "path": "a.py",
            "reason": "rights unknown: no license evidence",
        }
    ]
    with pytest.raises(
        ValueError, match="produced no documents: code_python"
    ) as failure:
        _require_every_source_has_documents(connection, declared, events)
    message = str(failure.value)
    assert "file a.py" in message
    assert "rights unknown: no license evidence" in message
    assert "release.policy.rights.training_use" in message
    complete = SimpleNamespace(sources=[SimpleNamespace(id="s")])
    _require_every_source_has_documents(connection, complete, events)


def test_deduplication_never_reads_a_payload(
    tmp_path: Path,
) -> None:
    connection = _database(
        [
            ("a-train", "train", "r1", "n1"),
            ("b-valid", "validation", "r1", "n1"),
            ("c-train", "train", "r2", "n2"),
            ("d-train", "train", "r2", "n2"),
        ]
    )
    read: list[str] = []

    def authorizer(action, table, column, *_):
        if action == sqlite3.SQLITE_READ and table == "documents":
            read.append(column)
        return sqlite3.SQLITE_OK

    class Phases:
        def __init__(self) -> None:
            self.phases: list[str] = []

        def update(self, **values: object) -> None:
            if "phase" in values:
                self.phases.append(str(values["phase"]))

    progress = Phases()
    connection.set_authorizer(authorizer)
    deduplicate(
        connection,
        tmp_path / "groups.jsonl",
        policy=CHAIN,
        diagnostics=tmp_path / "diagnostics.json",
        progress=progress,
    )
    assert "payload" not in read
    assert progress.phases == ["dedup_groups", "dedup_compress", "dedup_reasons"]
