"""The mixture selection index: same rows as the specification, on any worker count."""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import pytest

from klab.canonical import canonical_json
from klab.corpus import mixture_index, shards
from klab.corpus.mixture_index import MixtureIndex, document_rank
from klab.corpus.release_verify import verification_settings

SEED = 42
STRATA = {"s0": "alpha", "s1": "alpha", "s2": "beta", "s3": "beta", "s4": "gamma"}
FAMILY_SPLITS = {"validation": "f5", "test": "f6"}
POLICIES = [None, "kml-card03-v1"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))


def _release(tmp_path: Path) -> dict:
    """A release with dropped rows, a repeated ID, shared content and three splits."""
    rng = random.Random(7)
    documents = []
    for number in range(900):
        source = f"s{rng.randrange(5)}"
        family = f"{source}-f{rng.randrange(7)}"
        split = next(
            (name for name, tail in FAMILY_SPLITS.items() if family.endswith(tail)),
            "train",
        )
        documents.append(
            {
                "document_id": _sha(f"document-{number}")[: 12 + number % 3],
                "source_id": source,
                "source_family": family,
                "split": split,
                "domains": [STRATA[source]],
                "content_sha256": _sha(f"content-{family}-{number % 11}"),
                "drop_reason": "duplicate" if number % 13 == 0 else None,
                "text": "x" * rng.randrange(50),
                "rights": {
                    "training_eligibility": "eligible"
                    if split == "train"
                    else "ineligible"
                },
            }
        )
    # A later kept row repeats an ID: the last row of the file is the document.
    repeat = {**documents[3], "text": "the later row"}
    documents.append(repeat)
    kept = {}
    for doc in documents:
        if doc["drop_reason"] is None:
            kept[doc["document_id"]] = doc
    inventory = [
        {
            "document_id": doc["document_id"],
            "family_id": doc["source_family"],
            "split": doc["split"],
            "stratum": STRATA[doc["source_id"]],
            "content_sha256": doc["content_sha256"],
        }
        for doc in kept.values()
    ]
    train = [doc["document_id"] for doc in kept.values() if doc["split"] == "train"]
    lineage = [{"record_id": record, "split": "train"} for record in train]
    lineage.append(lineage[0])  # a repeated lineage row is harmless
    paths = {
        "documents": tmp_path / "documents.jsonl",
        "inventory": tmp_path / "family-inventory.jsonl",
        "lineage": tmp_path / "train.lineage.jsonl",
    }
    _write(paths["documents"], documents)
    _write(paths["inventory"], inventory)
    _write(paths["lineage"], lineage)
    return {"paths": paths, "documents": documents, "kept": kept}


def _expected(release: dict, order_policy) -> dict[str, list[tuple[str, str, str]]]:
    """The selection by its definition: train documents by (rank, id) within a stratum."""
    ranked: dict[str, list] = {name: [] for name in set(STRATA.values())}
    for doc in release["kept"].values():
        if doc["split"] != "train":
            continue
        stratum = STRATA[doc["source_id"]]
        rank = document_rank(
            SEED, stratum, doc["source_family"], doc["document_id"], order_policy
        )
        ranked[stratum].append((rank, doc["document_id"], doc["source_family"]))
    return {stratum: sorted(rows) for stratum, rows in ranked.items()}


def _build(release: dict, scratch: Path, order_policy=None) -> MixtureIndex:
    scratch.mkdir()
    index = MixtureIndex(scratch, release["paths"]["documents"])
    index.inventory(release["paths"]["inventory"], STRATA, SEED, order_policy)
    index.select(release["paths"]["lineage"])
    return index


def _shard_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shards, "CHUNK_BYTES", 5000)
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)


def _files(index: MixtureIndex) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes() for path in index.selection.directory.iterdir()
    }


@pytest.mark.parametrize("order_policy", POLICIES)
def test_selection_is_the_specification_for_any_workers_and_resumes_after_a_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order_policy
) -> None:
    release = _release(tmp_path)
    expected = _expected(release, order_policy)
    _shard_small(monkeypatch)
    with verification_settings(workers=1), _build(
        release, tmp_path / "serial", order_policy
    ) as serial:
        reference = _files(serial)
        for stratum, rows in expected.items():
            taken = [
                (doc["document_id"], family)
                for doc, family in serial.candidates(stratum)
            ]
            assert taken == [(doc_id, family) for _, doc_id, family in rows]
            # A resumed schedule continues after its frontier, in the same order.
            middle = len(rows) // 2
            rank, doc_id, _ = rows[middle]
            resumed = [
                doc["document_id"] for doc, _ in serial.candidates(stratum, [rank, doc_id])
            ]
            assert resumed == [row[1] for row in rows[middle + 1 :]]
        repeated = next(
            doc
            for doc, _ in serial.candidates(
                STRATA[release["documents"][3]["source_id"]]
            )
            if doc["document_id"] == release["documents"][3]["document_id"]
        )
        assert repeated["text"] == "the later row"

    scratch = tmp_path / "scratch"
    monkeypatch.setenv("KLAB_FAULT", f"{mixture_index.INVENTORY_STAGE}:2")
    with (
        verification_settings(workers=3, scratch_dir=scratch),
        pytest.raises(RuntimeError, match="KLAB_FAULT"),
    ):
        _build(release, tmp_path / "killed", order_policy)
    monkeypatch.delenv("KLAB_FAULT")
    echo = io.StringIO()
    with (
        verification_settings(workers=3, scratch_dir=scratch, echo=echo),
        _build(release, tmp_path / "resumed", order_policy) as resumed,
    ):
        assert _files(resumed) == reference
    reused = [line for line in echo.getvalue().splitlines() if "reused" in line]
    assert any("family inventory" in line and "reused 2/" in line for line in reused)
    assert any("documents" in line and "reused 0/" not in line for line in reused)


def _drop_document(release: dict) -> None:
    rows = [json.loads(line) for line in release["paths"]["documents"].read_text().splitlines()]
    _write(release["paths"]["documents"], rows[1:])


def _edit(path_name: str, change):
    def edit(release: dict) -> None:
        path = release["paths"][path_name]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        _write(path, rows)

    return edit


def _family_leak(rows: list[dict]) -> None:
    first = next(row for row in rows if row["split"] == "train")
    other = next(
        row
        for row in rows
        if row["split"] == "validation" and row["family_id"] != first["family_id"]
    )
    other["family_id"] = first["family_id"]


def _content_leak(rows: list[dict]) -> None:
    train = next(row for row in rows if row["split"] == "train")
    held = next(row for row in rows if row["split"] == "validation")
    held["content_sha256"] = train["content_sha256"]


REFUSALS = {
    "a document the release does not have": (
        _edit("inventory", lambda rows: rows.append({**rows[0], "document_id": "x"})),
        "unknown or duplicate document",
    ),
    "a document listed twice": (
        _edit("inventory", lambda rows: rows.append(dict(rows[0]))),
        "unknown or duplicate document",
    ),
    "a split the release disagrees with": (
        _edit("inventory", lambda rows: rows[0].update(split="test")),
        "disagrees with verified release",
    ),
    "a family on two splits": (
        _edit("inventory", _family_leak),
        "family leaks across splits",
    ),
    "content on two splits": (
        _edit("inventory", _content_leak),
        "identical content leaks across splits",
    ),
    "a kept document without a family": (
        _edit("inventory", lambda rows: rows.pop()),
        "does not cover every kept document",
    ),
    "a train document missing from the lineage": (
        _edit("lineage", lambda rows: rows.pop(0) and rows.pop(0)),
        "does not cover exactly eligible train documents",
    ),
    "a held-out document in the lineage": (
        _edit("lineage", lambda rows: rows.append({"record_id": "no-such-record"})),
        "does not cover exactly eligible train documents",
    ),
    "a train document without training rights": (
        _edit(
            "documents",
            lambda rows: next(r for r in rows if r["split"] == "train").update(
                rights={"training_eligibility": "ineligible"}
            ),
        ),
        "lacks admitted file or record rights",
    ),
}


@pytest.mark.parametrize("damage", REFUSALS)
def test_an_inventory_or_lineage_that_disagrees_with_the_release_is_refused(
    tmp_path: Path, damage: str
) -> None:
    release = _release(tmp_path)
    change, message = REFUSALS[damage]
    change(release)
    with verification_settings(workers=1), pytest.raises(ValueError, match=message):
        _build(release, tmp_path / "index")
