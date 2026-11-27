"""The procedural workflow keeps ground truth and family scope through preparation."""

import json
from pathlib import Path

import pytest
import yaml
from lab_fixtures import write_smoke_inputs
from recipe_fixtures import Lab, write_recipe

from klab.canonical import canonical_json
from klab.corpus.jsonl_records import read_rows
from klab.corpus.release import open_release
from klab.training.manifest import sha256_file

_RULE = {
    "kind": "bounded_transfer",
    "capacity": [10, 12],
    "minimum": [2, 3],
    "request_limit": 7,
}


def _world(name, split, prose=None):
    return {
        "id": f"world:{name}",
        "family": f"family:{name}",
        "split": split,
        "rule": _RULE,
        "initial": [2, 11],
        "steps": 5,
        "prose": prose,
    }


def _sources(root, *shards):
    sources = []
    for number, rows in enumerate(shards):
        path = root / f"source-{number}.jsonl"
        path.write_bytes(b"\n" + b"".join(canonical_json(row) + b"\n" for row in rows))
        sources.append(
            {
                "path": path.name,
                "id": f"source:{number}",
                "sha256": sha256_file(path),
                "max_rows": len(rows),
                "uri": f"urn:klab:mechanisms:{number}",
                "license": "CC0-1.0",
                "origin": "procedural",
            }
        )
    declaration = root / "worlds.json"
    declaration.write_bytes(
        canonical_json(
            {
                "sources": sources,
                "seed": 11,
                "max_catalog_bytes": 1024**2,
                "max_shard_bytes": 1024**2,
            }
        )
    )
    return declaration


def _factorized_worlds(root, lab, cases):
    scope, claims = [], []
    for case in cases:
        world = case["id"]
        placement = {"family": case["family"], "split": case["split"]}
        scope.append({"id": world, **placement})
        for index in range(2):
            child = f"{world}/node{index}"
            scope.append({"id": child, **placement})
            claims.append(
                {
                    "subject": child,
                    "predicate": "world:part-of",
                    "value": {"kind": "entity", "id": world},
                }
            )
    pins = {}
    for name, rows in (("scope", scope), ("claims", claims)):
        path = root / f"{name}.jsonl"
        path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
        pins[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "max_rows": len(rows),
        }
    spec = root / "relations.json"
    spec.write_bytes(
        canonical_json(
            {
                "scope": pins["scope"],
                "sources": [
                    {
                        **pins["claims"],
                        "id": "world-claims",
                        "format": "records",
                        "uri": "urn:klab:case-claims",
                        "license": "CC0-1.0",
                        "origin": "structured",
                    }
                ],
                "max_index_bytes": 4 * 1024**2,
                "max_shard_output_bytes": 1024**2,
            }
        )
    )
    index = lab.json("corpus", "relations", "extract", str(spec))
    options = root / "relations-options.json"
    options.write_bytes(
        canonical_json(
            {
                "target": "factorized",
                "max_bytes": 1024**2,
                "max_sequences": 30,
                "max_scratch_bytes": 1024**2,
                "group_size": 8,
            }
        )
    )
    return lab.json("corpus", "relations", "compile", index["key"], str(options))["key"]


def test_procedural_replay_curriculum_and_packing_evidence(
    tmp_path, monkeypatch, capsys
):
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    build = lab.run("corpus", "build", project, "--offline")
    release_path = lab.run("corpus", "freeze", build)
    examples = {}
    with open_release(Path(release_path)) as release:
        for doc in release.rows("documents.jsonl"):
            if doc["drop_reason"] is None:
                examples.setdefault(doc["split"], doc)

    train = _world("train", "train", examples["train"]["text"])
    heldout = _world("heldout", "validation", examples["validation"]["text"])
    test = _world("test", "test")
    declaration = _sources(tmp_path, [train], [heldout, test])
    simulation = ("corpus", "mechanisms", "simulate", str(declaration))
    plan = lab.json(*simulation, "--plan")
    with monkeypatch.context() as fault:
        fault.setenv("KLAB_FAULT", "procedural world simulation:1")
        with pytest.raises(RuntimeError, match="after shard 1"):
            lab.json(*simulation, "--workers", "1")
    worlds = lab.json(*simulation, "--workers", "2")
    assert "reused 1/2" in lab.stderr
    assert worlds["key"] == plan["key"]
    assert worlds["coverage"] == {"train": 1, "validation": 1, "test": 1}
    assert worlds["failed_derivations"] > 0
    audit = lab.json("corpus", "mechanisms", "audit", worlds["key"], "--samples", "3")
    assert audit["replayed"] == [
        {"world": "world:train", "split": "train", "steps": 5},
        {"world": "world:heldout", "split": "validation", "steps": 5},
        {"world": "world:test", "split": "test", "steps": 5},
    ]

    relation_key = _factorized_worlds(tmp_path, lab, (train, heldout, test))
    options = tmp_path / "compile.json"
    targets = ["trajectories", "derivations", "constraints", "counterfactuals", "prose"]
    compiler = ("corpus", "mechanisms", "compile", worlds["key"], str(options))
    options.write_bytes(
        canonical_json({"targets": targets, "max_sequences": 1, "max_bytes": 1024**2})
    )
    with pytest.raises(ValueError, match="exceeds its sequence or byte budget"):
        lab.json(*compiler)
    assert not lab.json(*compiler, "--plan")["candidate"]
    options.write_bytes(
        canonical_json(
            {
                "targets": targets,
                "max_sequences": 20,
                "max_bytes": 1024**2,
                "relation_keys": [relation_key],
            }
        )
    )
    mixed = lab.json(*compiler)
    assert mixed["sequences"] == 17
    path = Path(mixed["path"])
    assert "world:heldout" not in (path / "train.jsonl").read_text()
    [trajectory, derivations, constraints, alternatives, prose, structure] = [
        row["text"] for row in read_rows(path / "train.jsonl")
    ]
    assert prose == train["prose"]
    assert json.loads(structure) == [
        "factorized",
        "world:part-of",
        "world:train",
        ["world:train/node0", "world:train/node1"],
    ]
    assert read_rows(path / "train.lineage.jsonl")[-1]["relation_key"] == relation_key

    trajectory, derivations, constraints, alternatives = map(
        json.loads, (trajectory, derivations, constraints, alternatives)
    )
    for fact, proof, decision, alternative in zip(
        trajectory["events"],
        derivations["events"],
        constraints["events"],
        alternatives["events"],
        strict=True,
    ):
        source = proof["source"]
        before = proof["before"]
        expected = min(
            before[source] - _RULE["minimum"][source],
            _RULE["capacity"][1 - source] - before[1 - source],
            _RULE["request_limit"],
        )
        assert decision["legal_maximum"] == proof["legal_maximum"] == expected
        assert bool(fact["violations"]) == (fact["amount"] > expected)
        assert sum(fact["after"]) == sum(before) == 13
        assert bool(alternative["counterfactual"]["violations"]) != bool(
            fact["violations"]
        )
    assert read_rows(path / "omissions.jsonl")[0]["world"] == "world:test"
    assert lab.json(*compiler)["files"] == mixed["files"]

    memberships = lab.json(
        "corpus",
        "mechanisms",
        "memberships",
        mixed["key"],
        release_path,
        "--max-inventory-bytes",
        str(8 * 1024**2),
        "--max-scratch-bytes",
        str(1024**2),
        "--max-output-bytes",
        str(1024**2),
    )
    assert memberships["matched_documents"] == 1
    [member] = read_rows(Path(memberships["path"]) / "entities.jsonl")
    assert member["document_id"] == examples["train"]["document_id"]
    assert member["entity_id"].startswith("mechanism:")

    config = write_smoke_inputs(tmp_path, max_steps=1, train_tokenizer_now=False)
    run = yaml.safe_load(config.read_text())
    run["dataset"].pop("synthetic_seed")
    run["dataset"].update(
        source="local_text",
        train_path=str(path / "train.jsonl"),
        validation_path=str(path / "validation.jsonl"),
        train_max_documents=6,
        validation_max_documents=6,
        license="CC0-1.0",
        revision=mixed["key"],
    )
    run["model"]["vocab_size"] = 320
    run["optimizer"]["warmup_steps"] = 0
    run["training"].update(micro_batch_size=1, max_tokens=32)
    config.write_text(yaml.safe_dump(run))
    tokenizer_path = tmp_path / "tokenizer.yaml"
    tokenizer = yaml.safe_load(tokenizer_path.read_text())
    tokenizer.update(dataset=run["dataset"], vocab_size=320)
    tokenizer_path.write_text(yaml.safe_dump(tokenizer))
    lab.run("tokenizer", "train", str(tokenizer_path))
    prepared = Path(lab.run("data", "prepare", str(config)))
    assert (prepared / "train.npy").is_file()
    lab.run("train", str(config), "--run-id", "procedural-smoke")
    assert (
        Path(run["logging"]["root_dir"]) / "procedural-smoke/manifest.json"
    ).is_file()


def test_procedural_scope_rejects_invalid_initial_state_and_cross_shard_family(
    tmp_path, monkeypatch, capsys
):
    lab = Lab(tmp_path, monkeypatch, capsys)
    train = _world("train", "train")
    heldout = _world("heldout", "test")
    heldout["family"] = train["family"]
    declaration = _sources(tmp_path, [train], [heldout])
    simulate = ("corpus", "mechanisms", "simulate", str(declaration))
    with pytest.raises(ValueError, match="semantic family spans multiple splits"):
        lab.json(*simulate)
    assert not lab.json(*simulate, "--plan")["candidate"]
    heldout.update(initial=[0, 12], family="family:heldout")
    _sources(tmp_path, [train], [heldout])
    with pytest.raises(ValueError, match="initial state is outside"):
        lab.json(*simulate)
    heldout["initial"] = [2, 11]
    _sources(tmp_path, [train], [heldout])
    worlds = lab.json(*simulate)
    relation = _factorized_worlds(
        tmp_path, lab, ({**train, "split": "validation"}, heldout)
    )
    options = tmp_path / "conflicting-compile.json"
    options.write_bytes(
        canonical_json(
            {
                "targets": ["constraints"],
                "relation_keys": [relation],
                "max_sequences": 20,
                "max_bytes": 1024**2,
            }
        )
    )
    with pytest.raises(ValueError, match="conflicting family or split"):
        lab.json("corpus", "mechanisms", "compile", worlds["key"], str(options))
