"""Pinned relations keep evidence and family boundaries through training."""

import json
from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from lab_fixtures import write_smoke_inputs
from recipe_fixtures import Lab

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.jsonl_records import read_rows
from klab.corpus.relation_sources import wikipedia_entity
from klab.training.manifest import sha256_file


def pin(root, name, rows, **options):
    path = root / name
    path.write_bytes(b"\n" + b"".join(canonical_json(row) + b"\n" for row in rows))
    return {"path": name, "sha256": sha256_file(path), "max_rows": len(rows), **options}


def entity_value(name):
    return {"kind": "entity", "id": name}


def triple(subject, predicate, target):
    return {"subject": subject, "predicate": predicate, "value": entity_value(target)}


def item_snak(predicate, target):
    return {
        "snaktype": "value",
        "property": predicate,
        "datavalue": {
            "type": "wikibase-entityid",
            "value": {"entity-type": "item", "id": target},
        },
    }


def relation_inputs(root):
    article = wikipedia_entity("enwiki", 0, "Mercury (planet)")
    category = wikipedia_entity("enwiki", 14, "Category:Solar System")
    names = [
        "world:a",
        "world:b",
        "world:c",
        article,
        category,
        "wikidata:Q1",
        "wikidata:Q2",
    ]
    scope = (
        [{"id": name, "family": "world:train", "split": "train"} for name in names]
        + [
            {
                "id": f"world:heldout/{i}",
                "family": "world:validation",
                "split": "validation",
            }
            for i in range(2)
        ]
        + [
            {"id": "world:test", "family": "world:test", "split": "test"},
        ]
    )
    claims = [
        triple("world:a", "world:points-to", "world:b"),
        {**triple("world:b", "world:points-to", "world:c"), "claim_class": "derived"},
        {**triple("world:b", "world:points-to", "world:a"), "claim_class": "candidate"},
        triple("world:c", "world:points-to", "world:heldout/0"),
        triple("world:heldout/0", "world:points-to", "world:heldout/1"),
        triple("world:heldout/1", "world:points-to", "world:heldout/0"),
        triple("world:test", "world:points-to", "world:test"),
    ]

    def time_snak(prop):
        return {
            "snaktype": "value",
            "property": prop,
            "datavalue": {
                "type": "time",
                "value": {
                    "time": "+1919-01-01T00:00:00Z",
                    "precision": 9,
                    "timezone": 0,
                    "before": 0,
                    "after": 0,
                    "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
                },
            },
        }

    valid = {
        "id": "Q1$first",
        "type": "statement",
        "rank": "preferred",
        "mainsnak": item_snak("P1", "Q2"),
        "qualifiers": {"P2": [time_snak("P2")]},
        "references": [
            {"hash": "source-group", "snaks": {"P3": [item_snak("P3", "Q2")]}}
        ],
    }
    wiki = [
        {
            "pageid": 18417,
            "ns": 0,
            "title": "Mercury (planet)",
            "links": [{"ns": 0, "title": "Mercury (planet)"}],
            "categories": [{"ns": 14, "title": "Category:Solar System"}],
        }
    ]
    wikidata = [
        {
            "type": "item",
            "id": "Q1",
            "claims": {
                "P1": [
                    valid,
                    {**valid, "id": "Q1$second", "rank": "normal"},
                    {
                        **valid,
                        "id": "Q1$contradiction",
                        "mainsnak": item_snak("P1", "Q1"),
                        "rank": "deprecated",
                    },
                    {
                        **valid,
                        "id": "Q1$unresolved",
                        "qualifiers": {
                            "P2": [{"property": "P2", "snaktype": "novalue"}]
                        },
                    },
                ]
            },
        }
    ]
    spec = {
        "scope": pin(root, "scope.jsonl", scope),
        "sources": [
            pin(
                root,
                f"{name}.jsonl",
                rows,
                id=name,
                format=name,
                license="CC0-1.0",
                uri=f"https://example.com/{name}",
                origin="structured",
                **options,
            )
            for name, rows, options in [
                ("records", claims, {}),
                ("wikipedia", wiki, {"site": "enwiki"}),
                ("wikidata", wikidata, {}),
            ]
        ],
        "max_index_bytes": 4 * 1024**2,
        "max_shard_output_bytes": 1024**2,
    }
    declaration = root / "relation.json"
    declaration.write_bytes(canonical_json(spec))
    return declaration, spec, scope, claims, article, category


def test_relational_sources_compile_and_train(tmp_path, monkeypatch, capsys):
    lab = Lab(tmp_path, monkeypatch, capsys)
    declaration, _, _, claims, article, category = relation_inputs(tmp_path)
    extract = ("corpus", "relations", "extract", str(declaration))
    plan = lab.json(*extract, "--plan")
    assert plan["candidate"] is False

    with monkeypatch.context() as patch:
        patch.setenv("KLAB_FAULT", "relation source extraction:1")
        with pytest.raises(RuntimeError, match="after shard 1"):
            lab.json(*extract, "--workers", "1")
    assert not Path(plan["path"]).exists()
    index = lab.json(*extract, "--workers", "2")
    assert "reused 1/3" in lab.stderr
    assert index["key"] == plan["key"]
    assert index["coverage"]["train"]["assertions"] == 7
    inspect = ("corpus", "relations", "inspect", index["key"])
    page = lab.json(*inspect, "--entity", "wikidata:Q1", "--limit", "1")
    (qualified,) = page["records"]
    assert qualified["value"] == entity_value("wikidata:Q2")
    assert qualified["qualifiers"][0]["value"]["precision"] == 9
    after = lab.json(*inspect, "--entity", "wikidata:Q1", "--after", str(page["next"]))
    assert after["records"][0]["value"] == entity_value("wikidata:Q1")
    evidence = lab.json(*inspect, "--assertion", qualified["id"])["records"]
    assert [(item["line"], item["location"], item["rank"]) for item in evidence] == [
        (2, "/claims/P1/0", "preferred"),
        (2, "/claims/P1/1", "normal"),
    ]
    assert evidence[0]["references"] == [
        {
            "id": "source-group",
            "snaks": [
                {"predicate": "wikidata:P3", "value": entity_value("wikidata:Q2")}
            ],
        }
    ]
    assert [
        (row["source"], row["line"], row["location"], row["reason"])
        for row in lab.json(*inspect, "--unresolved")["records"]
    ] == [
        ("records", 5, "/", "cross_split_assertion"),
        ("wikidata", 2, "/claims/P1/3", "unsupported_statement"),
    ]
    wikipedia = lab.json(*inspect, "--entity", article)["records"]
    assert [(row["predicate"], row["value"]["id"]) for row in wikipedia] == [
        ("wikipedia:links", article),
        ("wikipedia:categories", category),
    ]
    assert (
        lab.json(*inspect, "--assertion", wikipedia[0]["id"])["records"][0]["source_id"]
        == "18417"
    )

    options = tmp_path / "compile.json"
    compile_command = ("corpus", "relations", "compile", index["key"], str(options))
    limits = {"max_sequences": 500, "max_bytes": 1024**2}
    options.write_bytes(canonical_json({"target": "triples", **limits}))
    compiled = lab.json(*compile_command)
    path = Path(compiled["path"])
    first = claims[0]
    expected = [first["subject"], first["predicate"], first["value"]]
    assert read_rows(path / "train.jsonl")[0] == {
        "text": canonical_json(expected).decode()
    }
    assert read_rows(path / "train.lineage.jsonl")[0] == {
        "assertions": [canonical_digest({**first, "qualifiers": []})],
    }

    options.write_bytes(
        canonical_json(
            {
                "target": "walks",
                "seed": 7,
                "hops": 3,
                "walks_per_entity": 2,
                "qualifiers": "omit",
                **limits,
            }
        )
    )
    walks = lab.json(*compile_command)
    walk_path = Path(walks["path"])
    train_walks = [
        json.loads(row["text"]) for row in read_rows(walk_path / "train.jsonl")
    ]
    assert train_walks[0] == [
        expected,
        [
            "world:b",
            "world:points-to",
            entity_value("world:c"),
            {"claim_classes": ["derived"]},
        ],
    ]
    assert all(
        step[2]["id"] == next_step[0]
        for chain in train_walks
        for step, next_step in pairwise(chain)
    )
    assert "heldout" not in (walk_path / "train.jsonl").read_text()
    assert "world:heldout" in (walk_path / "validation.jsonl").read_text()
    assert lab.json(*compile_command)["files"] == walks["files"]
    assert lab.json(*extract)["files"] == index["files"]

    config = write_smoke_inputs(tmp_path, max_steps=1, train_tokenizer_now=False)
    run = yaml.safe_load(config.read_text())
    run["dataset"].pop("synthetic_seed", None)
    run["optimizer"]["warmup_steps"] = 0
    run["model"]["vocab_size"] = 320
    run["training"]["micro_batch_size"] = 1
    run["training"]["max_tokens"] = 32
    run["dataset"].update(
        source="local_text",
        train_path=str(path / "train.jsonl"),
        validation_path=str(path / "validation.jsonl"),
        revision=compiled["key"],
        train_max_documents=compiled["partitions"]["train"]["sequences"],
        validation_max_documents=compiled["partitions"]["validation"]["sequences"],
        license="CC0-1.0",
    )
    config.write_text(yaml.safe_dump(run))
    tokenizer_path = tmp_path / "tokenizer.yaml"
    tokenizer = yaml.safe_load(tokenizer_path.read_text())
    tokenizer["dataset"] = run["dataset"]
    tokenizer["vocab_size"] = run["model"]["vocab_size"]
    tokenizer_path.write_text(yaml.safe_dump(tokenizer))
    lab.run("tokenizer", "train", str(tokenizer_path))
    prepared = Path(lab.run("data", "prepare", str(config)))
    manifest = json.loads((prepared / "manifest.json").read_bytes())
    assert manifest["supervision"] == {"kind": "all_tokens"}
    lab.run("train", str(config), "--run-id", "compiled-smoke")
    assert (
        Path(run["logging"]["root_dir"]) / "compiled-smoke" / "manifest.json"
    ).is_file()

    database = Path(index["path"]) / "relations.sqlite"
    with database.open("r+b") as raw:
        raw.write(b"X")
    with pytest.raises(ValueError, match="output changed"):
        lab.json(*inspect)


def test_corpus_relations_refuse_a_family_shared_by_train_and_validation(
    tmp_path, monkeypatch, capsys
):
    lab = Lab(tmp_path, monkeypatch, capsys)
    declaration, spec, scope, *_ = relation_inputs(tmp_path)
    scope[-2]["family"] = scope[0]["family"]
    spec["scope"] = pin(tmp_path, "scope.jsonl", scope)
    declaration.write_bytes(canonical_json(spec))
    extract = ("corpus", "relations", "extract", str(declaration))
    with pytest.raises(ValueError, match="spans multiple splits"):
        lab.json(*extract)
    assert lab.json(*extract, "--plan")["candidate"] is False
