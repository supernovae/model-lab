"""Oracle and recorded-response derivations are checked against source evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import (
    _path_scenario,
    _records_for_file,
    _scenario,
    _scenario_messages,
    _scenario_v2,
    _sections,
    build,
)
from klab.corpus.project import load_project
from klab.corpus.release import freeze
from klab.data.conversations import iter_rendered_conversations

PROJECT = Path(__file__).resolve().parents[1] / "tests/fixtures/corpus-sample"


def _replay_project(tmp_path: Path, *, bad_prompt: bool = False):
    root = tmp_path / "recipe"
    shutil.copytree(PROJECT, root)
    baseline = load_project(root / "corpus.yaml")
    acquire(baseline, tmp_path / "baseline")
    built = build(baseline, tmp_path / "baseline", offline=True)
    docs = [
        json.loads(line)
        for line in (built / "documents.jsonl").read_text().splitlines()
    ]
    parent = next(doc for doc in docs if doc["source_id"] == "sample_docs")
    template = "Answer using only the cited passage:\n{passages}"
    prompt = template.format(
        passages=f"[{parent['document_id']}] {parent['text'][:2000]}"
    )
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    provider = {
        "provider": "offline_fixture",
        "model": "recorded",
        "model_revision": "pinned-v1",
    }
    responses = []
    for number, answer in enumerate(
        (parent["text"].splitlines()[0], "invented unsupported answer", None)
    ):
        raw = (
            "not-json"
            if answer is None
            else json.dumps({"answer": answer, "citation_id": parent["document_id"]})
        )
        responses.append(
            {
                "request_id": f"request-{number}",
                "source_document_ids": [parent["document_id"]],
                "prompt_sha256": "0" * 64 if bad_prompt else prompt_hash,
                "raw_output": raw,
                **provider,
                "seed": number,
            }
        )
    (root / "sources/fixtures/prompt.txt").write_text(template)
    (root / "sources/fixtures/responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses)
    )
    (root / "sources/generator-inputs.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "recordings",
                "kind": "local",
                "canonical_uri": "fixture:responses",
                "revision": "v1",
                "license": "MIT",
                "redistribution": "redistributable",
                "domains": ["technical_docs"],
                "document_kinds": ["data"],
                "source_family": "sample_docs_train",
                "acquisition": {
                    "files": [
                        {"path": "sources/fixtures/prompt.txt", "name": "prompt.txt"},
                        {
                            "path": "sources/fixtures/responses.jsonl",
                            "name": "responses.jsonl",
                        },
                    ],
                    "max_bytes": 100000,
                },
            }
        )
    )
    (root / "transforms/inference.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "sample_inference",
                "version": "1",
                "kind": "inference_qa",
                "inputs": ["sample_docs", "recordings"],
                "parameters": {
                    "generator_source_id": "recordings",
                    "backend": "recorded_responses",
                    **provider,
                    "prompt_template_path": "prompt.txt",
                    "responses_path": "responses.jsonl",
                    "max_input_chars": 2000,
                    "generation_config": {"temperature": 0},
                },
            }
        )
    )
    corpus = yaml.safe_load((root / "corpus.yaml").read_text())
    corpus["sources"] = sorted([*corpus["sources"], "sources/generator-inputs.yaml"])
    corpus["transforms"] = sorted([*corpus["transforms"], "transforms/inference.yaml"])
    (root / "corpus.yaml").write_text(yaml.safe_dump(corpus))
    return load_project(root / "corpus.yaml"), parent


def test_scenario_oracle_is_deterministic_without_filesystem_access() -> None:
    first = _path_scenario(23, "world-23", "stage")
    assert first == _path_scenario(23, "world-23", "stage")
    assert first["oracle_answer"] == Path(first["world_state"]["path"]).suffix
    assert first["world_state"]["operation"] == "suffix"


def test_grounded_unverified_and_rejected_replies_replay_offline(
    tmp_path: Path,
) -> None:
    project, parent = _replay_project(tmp_path)
    work = tmp_path / "work"
    acquire(project, work)
    built = build(project, work, offline=True)
    assert build(project, work, offline=True) == built
    records = [
        json.loads(line)
        for line in (built / "generations.jsonl").read_text().splitlines()
    ]
    assert [row["validation_status"] for row in records] == [
        "source_entailed",
        "unverified",
        "rejected",
    ]
    assert all(
        row["generator_identity"]
        and row["seed"] is not None
        and row["source_document_ids"] == [parent["document_id"]]
        for row in records
    )
    evidence = records[0]["evidence"]
    assert (
        parent["text"][slice(*evidence["span"])]
        == evidence["passage"]
        == records[0]["parsed_output"]["answer"]
    )
    assert records[1]["evidence"] is None
    assert (
        records[-1]["raw_output"] == "not-json" and records[-1]["parsed_output"] is None
    )
    lineage_rows = [
        json.loads(line) for line in (built / "lineage.jsonl").read_text().splitlines()
    ]
    generated_lineage = [
        row for row in lineage_rows if row["record_kind"] == "generation"
    ]
    assert len(generated_lineage) == len(records)
    assert {row["origin"] for row in generated_lineage} == {
        "source_transformed_synthetic"
    }
    assert {row["verification"]["status"] for row in generated_lineage} == {
        "source_entailed",
        "unverified",
        "rejected",
    }
    records_by_id = {row["record_id"]: row for row in records}
    assert all(
        row["generator"] == records_by_id[row["record_id"]]["generator"]
        for row in generated_lineage
    )
    released = freeze(built, work)
    assert any(
        json.loads(line)["record_id"] == records[-1]["record_id"]
        for line in (released / "lineage.jsonl").read_text().splitlines()
    )
    for split in ("train", "validation", "test"):
        assert list(iter_rendered_conversations(released / "chat" / f"{split}.jsonl"))


def test_changed_prompt_digest_rejects_recorded_request(tmp_path: Path) -> None:
    project, _ = _replay_project(tmp_path, bad_prompt=True)
    work = tmp_path / "work"
    acquire(project, work)
    with pytest.raises(ValueError, match="prompt mismatch"):
        build(project, work, offline=True)
    release_root = work / "corpora" / project.config.id / "releases"
    assert not release_root.exists() or not any(release_root.iterdir())
    diagnostics = list(
        (work / "corpora" / project.config.id / "builds/diagnostics").glob("*.json")
    )
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["request_id"] == "request-0"


@pytest.mark.parametrize(
    "seed,expected",
    [
        (0, "stop"),
        (1, "ask"),
        (2, "inspect"),
        (3, "inspect"),
        (4, "stop"),
        (5, "proceed"),
        (6, "proceed"),
        (7, "inspect"),
        (8, "inspect"),
    ],
)
def test_filesystem_oracle_priority_and_inert_tools(seed: int, expected: str) -> None:
    scenario = _scenario(
        "filesystem_judgment_v1",
        seed,
        "filesystem_train",
        "filesystem_template_train",
        "scenario",
    )
    assert scenario == _scenario(
        "filesystem_judgment_v1",
        seed,
        "filesystem_train",
        "filesystem_template_train",
        "scenario",
    )
    assert scenario["oracle_receipt"]["judgment"] == expected
    assert scenario["oracle_answer"].startswith(expected + ":")
    assert scenario["oracle_receipt"]["world_facts"] == scenario["world_state"]
    messages = _scenario_messages(scenario, True)
    assert messages[1]["tool_calls"][0]["name"] == "declared_world_inspection_v1"
    assert (
        messages[1]["tool_calls"][0]["arguments"]["world_id"]
        == scenario["generator_world_id"]
    )
    assert messages[2]["content"] == scenario["oracle_receipt"]["evidence"]
    assert messages[-1]["role"] == "assistant"


def test_generator_worlds_and_templates_are_split_distinct_and_never_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import subprocess

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scenario data attempted to execute a command")

    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for generator in (
        "filesystem_judgment_v1",
        "platform_fault_v1",
        "deployment_change_v1",
        "code_test_workflow_v1",
    ):
        rows = [
            _scenario(
                generator,
                seed,
                f"{generator}_{split}",
                f"{generator}_template_{split}",
                "scenario",
            )
            for seed, split in ((23, "train"), (823, "validation"), (923, "test"))
        ]
        assert len({row["generator_world_id"] for row in rows}) == 3
        assert len({row["world_state"]["fixture_id"] for row in rows}) == 3
        assert len({row["template_family_id"] for row in rows}) == 3
        assert (
            len({row["rendered_example"]["question"].split(" ", 1)[0] for row in rows})
            == 3
        )
        for row in rows:
            assert _scenario_messages(row, True)[2]["role"] == "tool"


@pytest.mark.parametrize(
    "generator,expected",
    [
        ("platform_fault_v1", "image pull"),
        ("deployment_change_v1", "healthy rollout"),
        ("code_test_workflow_v1", "broad regression fails"),
    ],
)
def test_causal_oracle_receipt(generator: str, expected: str) -> None:
    seed = {
        "platform_fault_v1": 3,
        "deployment_change_v1": 3,
        "code_test_workflow_v1": 3,
    }[generator]
    row = _scenario(generator, seed, "family_train", "template_train", "stage")
    assert (
        row["world_state"]["cause" if generator == "platform_fault_v1" else "condition"]
        == expected
    )
    assert expected in row["oracle_answer"]
    assert row["oracle_receipt"]["verification"]
    assert row["oracle_receipt"]["rollback"]
    assert row["oracle_receipt"]["world_id"] == row["generator_world_id"]


def test_markdown_navigation_does_not_leak_across_source_families() -> None:
    document = (
        '## {{% heading "whatsnext" %}}\n\n'
        "* [Shared cross-family link](/docs/related/)\n"
        "## Troubleshooting\n\n"
        "## Diagnosis\n\n"
        "A failed probe needs its configured endpoint inspected.\n"
    )
    sections = _sections(document, True)
    assert len(sections) == 1
    assert "Shared cross-family link" not in sections[0][0]
    assert "configured endpoint" in sections[0][0]


def test_cnxml_extracts_prose_skips_local_media_and_rejects_entities_external_media() -> (
    None
):
    from types import SimpleNamespace

    source = SimpleNamespace(
        schema_version=1,
        id="physics",
        kind="git",
        modality="text",
        revision="a" * 40,
        license="CC-BY-4.0",
        redistribution="redistributable",
        domains=("general_education",),
        document_kinds=("prose",),
        source_family="general_train",
    )
    raw = b'<document xmlns="http://cnx.rice.edu/cnxml"><content><section><title>Motion</title><para>Speed is distance divided by time.</para></section></content></document>'
    rows = _records_for_file(raw, "modules/m54057/index.cnxml", source, "b" * 64)
    doc, span = rows[0]
    assert doc["text"] == "Motion\nSpeed is distance divided by time."
    assert "<para>" not in doc["text"]
    assert doc["source_location"].startswith("modules/m54057/index.cnxml#")
    assert doc["metadata"]["normalizer"] == span["normalizer"] == "cnxml-text-v1"
    assert (
        doc["document_id"]
        != _records_for_file(raw, "modules/m54057/index.cnxml", source, "c" * 64)[0][0][
            "document_id"
        ]
    )
    local_media = raw.replace(
        b"</section>",
        b'<figure><media><image src="../../media/photo.jpg"/></media><caption>Third-party caption.</caption></figure></section>',
    )
    extracted = _records_for_file(
        local_media, "modules/m54057/index.cnxml", source, "b" * 64
    )
    assert extracted[0][0]["text"] == doc["text"]
    safe_escape = raw.replace(b"Speed", b"Distance &amp; speed")
    assert (
        "Distance & speed"
        in _records_for_file(
            safe_escape, "modules/m54057/index.cnxml", source, "b" * 64
        )[0][0]["text"]
    )
    for invalid in (
        raw.replace(b"Speed", b"&untrusted; Speed"),
        raw.replace(b"</para>", b'<media src="https://example.org/photo.jpg"/></para>'),
    ):
        with pytest.raises(ValueError, match="entities|media"):
            _records_for_file(invalid, "modules/m54057/index.cnxml", source, "b" * 64)


def test_source_qa_export_binds_exact_parent_span_and_keeps_sample(
    tmp_path: Path,
) -> None:
    root = tmp_path / "recipe"
    shutil.copytree(PROJECT, root)
    qa_path = root / "transforms/source-qa.yaml"
    qa_path.write_text(
        yaml.safe_dump(
            {
                "id": "source_qa",
                "version": "1",
                "kind": "source_qa",
                "inputs": ["sample_config"],
                "parameters": {"extractor": "literal_span_v1"},
            }
        )
    )
    corpus = yaml.safe_load((root / "corpus.yaml").read_text())
    corpus["transforms"].append("transforms/source-qa.yaml")
    (root / "corpus.yaml").write_text(yaml.safe_dump(corpus))
    chat_path = root / "transforms/chat.yaml"
    chat = yaml.safe_load(chat_path.read_text())
    chat["inputs"].append("source_qa")
    chat_path.write_text(yaml.safe_dump(chat))
    project = load_project(root / "corpus.yaml")
    work = tmp_path / "work"
    acquire(project, work)
    built = build(project, work, offline=True)
    docs = {
        row["document_id"]: row
        for row in map(json.loads, (built / "documents.jsonl").read_text().splitlines())
    }
    generations = list(
        map(json.loads, (built / "generations.jsonl").read_text().splitlines())
    )
    assert generations
    assert {"source_grounded_qa", "paraphrased_qa"} == {
        row["parsed_output"]["shape"] for row in generations
    }
    for record in generations:
        evidence = record["evidence"]
        assert (
            docs[evidence["document_id"]]["text"][slice(*evidence["span"])]
            == evidence["passage"]
        )
        assert docs[evidence["document_id"]]["text"].encode("utf-8")[
            slice(*evidence["byte_span"])
        ] == evidence["passage"].encode("utf-8")
        assert record["parsed_output"]["answer"] in evidence["passage"]
    release = freeze(built, work)
    assert any(
        (release / "lm" / f"{split}.jsonl").read_text().strip()
        for split in ("train", "validation", "test")
    )
    assert any(
        row["messages"][0]["content"].startswith("According to this source")
        for split in ("train", "validation", "test")
        for row in map(
            json.loads, (release / "chat" / f"{split}.jsonl").read_text().splitlines()
        )
    )


def test_compact_generator_partition_and_chat_tool_lineage(tmp_path: Path) -> None:
    root = tmp_path / "recipe"
    shutil.copytree(PROJECT, root)
    source_path = root / "sources/scenarios.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["acquisition"]["generator"] = "filesystem_judgment_v1"
    source["canonical_uri"] = "klab://generators/filesystem_judgment_v1"
    source_path.write_text(yaml.safe_dump(source))
    path = root / "transforms/scenarios.yaml"
    transform = yaml.safe_load(path.read_text())
    transform["parameters"] = {
        "generator": "filesystem_judgment_v1",
        "seed_ranges": [
            {
                "start": start,
                "end": end,
                "scenario_family": family,
                "template_family": f"filesystem_template_{split}",
            }
            for start, end, family, split in (
                (0, 799, "sample_world_train", "train"),
                (800, 899, "sample_world_validation", "validation"),
                (900, 999, "sample_world_test", "test"),
            )
        ],
    }
    path.write_text(yaml.safe_dump(transform))
    project = load_project(root / "corpus.yaml")
    work = tmp_path / "work"
    acquire(project, work)
    built = build(project, work, offline=True)
    rows = list(map(json.loads, (built / "scenarios.jsonl").read_text().splitlines()))
    assert len(rows) == 1000
    assert [
        sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    ] == [800, 100, 100]
    assert len({row["generator_world_id"] for row in rows}) == 1000
    lineage = {
        row["record_id"]: row
        for row in map(json.loads, (built / "lineage.jsonl").read_text().splitlines())
    }
    for row in rows:
        info = lineage[row["scenario_id"]]
        assert info["template_family_id"] == row["template_family_id"]
        assert (
            info["verification"]["evidence"]["receipt"]["world_facts"]
            == row["world_state"]
        )
        assert info["verification"]["status"] == "oracle_verified"
    episodes = list(
        map(json.loads, (built / "tool_episodes.jsonl").read_text().splitlines())
    )
    assert len(episodes) == len(rows)
    assert all(
        episode["tool_call"]["name"] == "declared_world_inspection_v1"
        for episode in episodes
    )
    release = freeze(built, work)
    assert all(
        (release / "chat" / f"{split}.jsonl").read_text().strip()
        for split in ("train", "validation", "test")
    )


@pytest.mark.parametrize(
    "generator",
    (
        "filesystem_judgment_v2",
        "platform_fault_v2",
        "deployment_change_v2",
        "code_test_workflow_v2",
    ),
)
def test_v2_world_facts_are_independent_and_inert(
    generator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import subprocess

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scenario generator attempted command execution")

    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    rows = [
        _scenario_v2(
            generator, seed, f"family_{seed // 100}", f"template_{seed // 100}", "stage"
        )
        for seed in range(1000)
    ]
    hashes = [
        hashlib.sha256(
            json.dumps(row["world_state"], sort_keys=True).encode()
        ).hexdigest()
        for row in rows
    ]
    assert len(set(hashes)) == 1000
    assert len(set(hashes[900:])) == 100
    assert not set(hashes[900:]) & set(hashes[:900])
    assert all("fixture_id" not in row["world_state"] for row in rows)
    for row in rows[900:]:
        assert row == _scenario_v2(
            generator,
            row["world_seed"],
            row["scenario_family_id"],
            row["template_family_id"],
            "stage",
        )
        assert row["oracle_receipt"]["world_facts"] == row["world_state"]
        assert row["oracle_receipt"]["generator_version"] == "2"
        assert (
            _scenario_messages(row, True)[1]["tool_calls"][0]["arguments"][
                "generator_version"
            ]
            == "2"
        )


def test_v2_filesystem_oracle_precedence() -> None:
    rows = [
        _scenario_v2("filesystem_judgment_v2", seed, "family", "template", "stage")
        for seed in range(1000)
    ]
    assert {"stop", "ask", "inspect", "proceed"} <= {
        row["oracle_receipt"]["judgment"] for row in rows
    }
    for row in rows:
        facts = row["world_state"]
        expected = (
            "stop"
            if not facts["in_workspace"]
            or not facts["reversible"]
            or facts["operation"] in ("modify source", "reset tracked files")
            or facts["target"].startswith("tracked")
            else "ask"
            if not facts["authorized"]
            else "inspect"
            if facts["target"] == "empty target"
            or facts["evidence"]
            not in (
                "exact path and owner confirmed",
                "owner confirmed by manifest",
                "target confirmed by dry-run",
                "build manifest confirms target",
            )
            else "proceed"
        )
        assert row["oracle_receipt"]["judgment"] == expected


@pytest.mark.parametrize("generator", ("platform_fault_v2", "filesystem_judgment_v2"))
def test_v2_compact_build_binds_generator_version_and_world_facts(
    tmp_path: Path, generator: str
) -> None:
    root = tmp_path / "recipe"
    shutil.copytree(PROJECT, root)
    source_path = root / "sources/scenarios.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["acquisition"]["generator"] = generator
    source["acquisition"]["generator_version"] = "2"
    source["canonical_uri"] = f"klab://generators/{generator}"
    source_path.write_text(yaml.safe_dump(source))
    transform_path = root / "transforms/scenarios.yaml"
    transform = yaml.safe_load(transform_path.read_text())
    transform["version"] = "2"
    transform["parameters"] = {
        "generator": generator,
        "seed_ranges": [
            {
                "start": start,
                "end": end,
                "scenario_family": f"sample_world_{split}",
                "template_family": f"platform_template_{split}",
            }
            for start, end, split in (
                (0, 799, "train"),
                (800, 899, "validation"),
                (900, 999, "test"),
            )
        ],
    }
    transform_path.write_text(yaml.safe_dump(transform))
    project = load_project(root / "corpus.yaml")
    work = tmp_path / "work"
    acquire(project, work)
    built = build(project, work, offline=True)
    scenarios = list(
        map(json.loads, (built / "scenarios.jsonl").read_text().splitlines())
    )
    assert len(scenarios) == 1000
    assert (
        len({json.dumps(row["world_state"], sort_keys=True) for row in scenarios})
        == 1000
    )
    lineage = {
        row["record_id"]: row
        for row in map(json.loads, (built / "lineage.jsonl").read_text().splitlines())
    }
    for row in scenarios:
        assert row["oracle_receipt"]["generator_version"] == "2"
        assert (
            lineage[row["scenario_id"]]["verification"]["evidence"]["receipt"][
                "world_facts"
            ]
            == row["world_state"]
        )
    freeze(built, work)
