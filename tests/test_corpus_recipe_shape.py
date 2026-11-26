"""A miniature of a real recipe, end to end, through the public commands.

Unit fixtures use a source or two; a real recipe has a dozen of several kinds,
a held-out source per stratum, a release policy and a training section composed
from files. Failures that only appear with that shape (a class of sources set
aside by a rights rule, a verifier assuming a duplicate key the policy never
declared, rights edits invalidating acquired snapshots) passed every unit test.
This module builds `tests/recipe_fixtures.py`'s recipe and checks, at each
stage, the properties those failures violated.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from corpus_fixtures import edit_declared_source, forge_release
from recipe_fixtures import (
    HELDOUT,
    SOURCE_STRATA,
    STRATA,
    Lab,
    Recipe,
    commit_git_fixture,
    write_recipe,
    write_training_files,
)

from klab.canonical import canonical_json
from klab.corpus import prepared_shards
from klab.corpus import release as release_module
from klab.corpus.release_verify import verification_settings
from klab.training.manifest import sha256_file

VOCAB_SIZE = 300
DECLARED = sorted(SOURCE_STRATA)
CODE_SOURCES = {
    "code_python",
    "code_python_heldout",
    "code_go",
    "code_shell",
    "code_systems_heldout",
}


@dataclass
class Prepared:
    """The corpus half of the pipeline: acquired, built twice, frozen."""

    lab: Lab
    recipe: Recipe
    lock: dict
    build_before_edit: Path
    build: Path
    release: Path

    @property
    def project(self) -> str:
        return str(self.recipe.project)


def edit_excluded_fields(recipe: Recipe) -> None:
    """Rights text, notes and domains: build inputs, not acquisition inputs."""

    def reword_mdn(source: dict) -> None:
        source["rights"]["notices"].append("Attribute MDN contributors.")
        source["notes"] = "Reviewed against the upstream licence page."
        source["domains"].append("documentation")

    recipe.edit("ref_mdn", reword_mdn)
    recipe.edit(
        "code_python",
        lambda source: source.update(license_url="https://example.org/license/v2"),
    )


def prepare(
    root: Path, monkeypatch, capsys, *, placement=(), **recipe_options
) -> Prepared:
    lab = Lab(root, monkeypatch, capsys)
    recipe = write_recipe(root / "recipe", **recipe_options)
    write_training_files(recipe, vocab_size=VOCAB_SIZE)
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    cache = str(root / "transport-cache")
    lab.run("corpus", "budget-init", project)
    lock = lab.json("corpus", "acquire", project, "--transport-cache", cache)
    assert sorted(lock["sources"]) == DECLARED
    # Every pinned shard was fetched exactly once.
    assert recipe.hub.bodies == len(recipe.hub.shards)
    build_before_edit = Path(lab.run("corpus", "build", project, "--offline"))

    # Editing rights text, notes and domains re-acquires nothing.
    requests = len(recipe.hub.requests)
    edit_excluded_fields(recipe)
    again = lab.json("corpus", "acquire", project, "--transport-cache", cache)
    assert len(recipe.hub.requests) == requests
    for name in DECLARED:
        for key in ("declaration_sha256", "snapshot_sha256"):
            assert again["sources"][name][key] == lock["sources"][name][key], name
    build = Path(lab.run("corpus", "build", project, "--offline"))
    release = Path(lab.run("corpus", "freeze", str(build), *placement))
    return Prepared(lab, recipe, lock, build_before_edit, build, release)


def documents(release: Path) -> list[dict]:
    control = "manifest.json" if (release / "manifest.json").exists() else "build.json"
    members = release_module.ReleaseMembers(
        release, json.loads((release / control).read_text())
    )
    return list(members.rows("documents.jsonl"))


def test_recipe_shaped_corpus_reaches_a_verified_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    prepared = prepare(tmp_path, monkeypatch, capsys)
    lab, recipe, release = prepared.lab, prepared.recipe, prepared.release
    docs = documents(release)

    scanned: Counter[str] = Counter()
    scan = release_module.iter_rows

    def count_rows(path):
        for row in scan(path):
            scanned[path.name] += 1
            yield row

    with monkeypatch.context() as lookup:
        lookup.setattr(release_module, "iter_rows", count_rows)
        sample = lab.json("corpus", "sample", str(release), "--limit", "1")
        assert len(sample["records"]) == 1
        assert scanned == {"lineage.jsonl": 1}
        trace = lab.json("corpus", "lineage", str(release), docs[0]["document_id"])
        assert trace["parents"][0]["document"] == docs[0]
        assert scanned["documents.jsonl"] == 1

    snapshots = release.parent.parent / "snapshots"
    acquisitions = {
        name: json.loads(
            (
                snapshots / name / source["snapshot_sha256"] / "manifest.json"
            ).read_bytes()
        )["declaration"]["acquisition"]
        for name, source in prepared.lock["sources"].items()
    }
    original_open = Path.open

    def no_snapshot_body(path, *args, **options):
        if path.name == "manifest.json" and path.is_relative_to(snapshots):
            pytest.fail("warm publication reread a raw snapshot manifest")
        return original_open(path, *args, **options)

    with monkeypatch.context() as warm:
        warm.setattr(Path, "open", no_snapshot_body)
        publication = lab.json("corpus", "publication", str(release))
    assert {
        row["id"]: row["acquisition"] for row in publication["sources"]
    } == acquisitions
    assert publication["rights_report_sha256"] == sha256_file(
        release / "license-report.json"
    )

    # An edit to rights text, notes or domains is a new build, not a new acquisition.
    assert prepared.build != prepared.build_before_edit

    # Every declared source kept at least one document, and as many as designed.
    kept = Counter(doc["source_id"] for doc in docs if doc["drop_reason"] is None)
    assert sorted(kept) == DECLARED
    assert dict(kept) == recipe.expected["kept"]
    report = json.loads((release / "report.json").read_text())
    assert sorted(report["source_counts"]) == DECLARED
    assert report["kept_document_count"] == sum(kept.values())
    # Every stratum of the requested mixture has documents behind it.
    populated = {
        name for name, row in report["actual_mixture"].items() if row["documents"]
    }
    assert set(STRATA) <= populated

    # A vendored path is set aside by name; no source disappears with it.
    rejected = [
        json.loads(line)
        for line in (release / "rejected.jsonl").read_text().splitlines()
    ]
    assert [(row["source_id"], row["path"]) for row in rejected] == [
        ("ref_tldr_pages", "pages/vendor/bundled.md")
    ]

    # Text held out of training is dropped from train, in both its raw and its
    # normalized form, and kept as validation.
    contaminated = [doc for doc in docs if doc["drop_reason"] == "contaminated_heldout"]
    assert len(contaminated) == recipe.expected["contaminated"]
    assert {(doc["source_id"], doc["split"]) for doc in contaminated} == {
        ("code_python", "train")
    }
    # Each stratum that declares a held-out source has validation documents.
    validation = {
        SOURCE_STRATA[doc["source_id"]]
        for doc in docs
        if doc["split"] == "validation" and doc["drop_reason"] is None
    }
    assert validation == {SOURCE_STRATA[name] for name in HELDOUT}
    lm_validation = (release / "lm/validation.jsonl").read_text().splitlines()
    assert len(lm_validation) == sum(recipe.expected["kept"][name] for name in HELDOUT)

    # The audit verifies once; a second call reuses the signed receipt.
    cold = []
    original = release_module._verify_release_cold
    monkeypatch.setattr(
        release_module,
        "_verify_release_cold",
        lambda *args, **kwargs: (cold.append(args[0]), original(*args, **kwargs))[1],
    )
    first = lab.run("corpus", "audit", str(release))
    reused = len(cold)
    assert lab.run("corpus", "audit", str(release)) == first
    assert len(cold) == reused <= 1

    scratch = tmp_path / "verification-scratch"
    expected_manifest = json.loads((release / "manifest.json").read_bytes())

    def verify_indexes():
        return release_module.verify_release(
            release, verification_mode="verified_reuse"
        )

    def interrupted(*args, **kwargs):
        raise InterruptedError("stop after durable verification index")

    def reran(*args, **kwargs):
        pytest.fail("a verified row index was rebuilt")

    with verification_settings(scratch_dir=scratch, workers=1):
        with monkeypatch.context() as stop:
            stop.setattr(release_module, "_document_rows", interrupted)
            with pytest.raises(InterruptedError):
                verify_indexes()
        assert list(scratch.glob("release-indexes/*/spans/*/spans.sqlite"))
        assert not list(scratch.glob("release-indexes/*/documents/*/documents.sqlite"))
        with monkeypatch.context() as stop:
            stop.setattr(release_module, "_span_rows", reran)
            stop.setattr(release_module, "_replay_sources", interrupted)
            with pytest.raises(InterruptedError):
                verify_indexes()
            with monkeypatch.context() as retry:
                retry.setattr(release_module, "_document_rows", reran)
                with pytest.raises(InterruptedError):
                    verify_indexes()
        (index,) = scratch.glob("release-indexes/*/documents/*/documents.sqlite")
        index.write_bytes(b"damaged verification index")
        with monkeypatch.context() as retry:
            retry.setattr(release_module, "_span_rows", reran)
            assert verify_indexes() == expected_manifest
    assert not list(scratch.glob("release-indexes/*/*/*/*.sqlite"))

    # Independently acquire the same pins under another project, then append a source.
    other = tmp_path / "second-corpus"
    shutil.copytree(recipe.corpus, other)
    config_path = other / "corpus.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["id"] = "second-recipe"
    config["transport_budget"]["attempt_id"] = "second-recipe-acquire-1"
    config_path.write_text(yaml.safe_dump(config))
    cache = str(tmp_path / "transport-cache")
    plan_args = ("corpus", "plan", str(config_path), "--json")
    assert lab.json(*plan_args)["status"] == "acquisition_required"
    lab.run("corpus", "budget-init", str(config_path))
    lab.run("corpus", "acquire", str(config_path), "--transport-cache", cache)
    normalized = []
    parse = prepared_shards.records_for_file

    def observe(raw, name, source, *args, **kwargs):
        normalized.append(source.id)
        return parse(raw, name, source, *args, **kwargs)

    monkeypatch.setattr(prepared_shards, "records_for_file", observe)
    plan = lab.json(*plan_args)
    assert plan["status"] == "planned", plan
    assert plan["normalization_source_bytes"] == 0
    assert plan["verification_candidate_source_bytes"] > 0

    # Formatting discarded HF evidence preserves the plan. A malformed array
    # still blocks planning, including after a run of compact selected rows.
    lock_path = lab.work / "corpora/second-recipe/acquisition.json"
    lock_bytes = lock_path.read_bytes()
    hf = json.loads(lock_bytes)["sources"]["math_finemath"]["receipt"]["retrieval"]
    rows = next(
        shard["selected_rows"] for shard in hf["shards"] if shard["selected_rows"]
    )
    compact = b'"selected_rows":' + canonical_json(rows)
    assert compact in lock_bytes
    pretty = b'"selected_rows":' + json.dumps(rows, indent=2).encode()
    lock_path.write_bytes(lock_bytes.replace(compact, pretty, 1))
    assert lab.json(*plan_args) == plan
    lock_path.write_bytes(lock_bytes.replace(compact, compact[:-1] + b",]", 1))
    assert lab.json(*plan_args)["status"] == "blocked"
    lock_path.write_bytes(lock_bytes)

    build_args = (
        "corpus",
        "build",
        str(config_path),
        "--offline",
        "--keep-stages",
        "2",
    )
    other_build = Path(lab.run(*build_args))
    assert plan["build_id"] == other_build.name
    assert normalized == []
    for name in ("documents.jsonl", "spans.jsonl"):
        assert (other_build / name).read_bytes() == (release / name).read_bytes()

    existing = lab.json(*plan_args)
    assert existing["build_state"] == "published"
    assert existing["build_id"] == other_build.name
    assert existing["sources"] == []

    # A view-only change reuses the exact index and deduplication stages.
    release_path = other / "release.yaml"
    original_release = release_path.read_bytes()
    view = yaml.safe_load(original_release)
    view["lm"]["training_splits"] = ["train"]
    release_path.write_text(yaml.safe_dump(view))
    real_open = Path.open

    def no_stage_body(path, *args, **kwargs):
        if path.suffix == ".sqlite" or (
            path.is_relative_to(lab.work / "cache/corpus/normalized")
            and path.suffix == ".jsonl"
        ):
            pytest.fail(f"build planning opened payload bytes: {path}")
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as forecast:
        forecast.setattr(Path, "open", no_stage_body)
        view_plan = lab.json(*plan_args)
    assert view_plan["build_id"] != other_build.name
    candidate_stages = {row["id"]: row for row in view_plan["stages"]}
    stage_store = other_build.parent / "stages"
    for label, directory in (("index", "index"), ("deduplicate", "dedup")):
        step = candidate_stages[label]
        assert step["action"] == "verify_candidate", view_plan
        (stored,) = (stage_store / directory).glob("*/receipt.json")
        assert step["computation_key"] == stored.parent.name
        assert step["verification_candidate_bytes"] == sum(
            item.stat().st_size for item in stored.parent.iterdir() if item != stored
        )
    view_build = Path(lab.run(*build_args))
    assert view_build.name == view_plan["build_id"]
    progress = (
        view_build.parent.parent / "progress" / f"{view_build.name}.jsonl"
    ).read_text()
    assert "reused_index_stage" in progress and "reused_dedup_stage" in progress

    # A dedup-only option changes its key, keeping the index verification candidate.
    view["policy"]["duplicates"]["heldout_overlap"] = "error"
    release_path.write_text(yaml.safe_dump(view))
    changed = {row["id"]: row for row in lab.json(*plan_args)["stages"]}
    assert changed["index"] == candidate_stages["index"]
    assert changed["deduplicate"]["action"] == "run"
    assert (
        changed["deduplicate"]["computation_key"]
        != candidate_stages["deduplicate"]["computation_key"]
    )
    release_path.write_bytes(original_release)

    other_release = lab.run("corpus", "freeze", str(other_build))
    consumed = lab.json(*plan_args)
    assert consumed["build_state"] == "consumed"
    assert consumed["release_id"] == Path(other_release).name

    appended = yaml.safe_load((other / "sources/ref_tldr_pages.yaml").read_text())
    repo = tmp_path / "appended-reference"
    name = "docs/append.md"
    appended.update(
        id="ref_append",
        source_family="ref_append",
        canonical_uri=str(repo),
        revision=commit_git_fixture(
            repo, {name: "# Extra\n\nAn independent reference.\n"}
        ),
        acquisition={"include": [name], "max_bytes": 1 << 20},
    )
    (other / "sources/ref_append.yaml").write_text(yaml.safe_dump(appended))
    config["sources"].append("sources/ref_append.yaml")
    config["sources"].sort()
    config_path.write_text(yaml.safe_dump(config))
    for relative, key, change in (
        (
            "splits.yaml",
            "assignments",
            lambda mapping: mapping.update(ref_append="train"),
        ),
        ("transforms/lm.yaml", "inputs", lambda inputs: inputs.append("ref_append")),
    ):
        path = other / relative
        payload = yaml.safe_load(path.read_text())
        change(payload[key])
        path.write_text(yaml.safe_dump(payload))
    assert lab.json(*plan_args)["status"] == "blocked"
    lab.run("corpus", "acquire", str(config_path), "--transport-cache", cache)
    plan = lab.json(*plan_args)
    assert plan["status"] == "planned", plan
    scheduled = [
        (source["source_id"], partition["path"], partition["source_bytes"])
        for source in plan["sources"]
        for partition in source["partitions"]
        if partition["action"] == "normalize"
    ]
    source_bytes = len(b"# Extra\n\nAn independent reference.\n")
    assert scheduled == [("ref_append", name, source_bytes)]
    assert plan["normalization_source_bytes"] == source_bytes
    assert (
        next(row for row in plan["stages"] if row["id"] == "index")["action"]
        == "await_normalization"
    )
    appended_build = Path(lab.run("corpus", "build", str(config_path), "--offline"))
    assert plan["build_id"] == appended_build.name
    assert normalized == ["ref_append"]
    assert {doc["source_id"] for doc in documents(appended_build)} == set(DECLARED) | {
        "ref_append"
    }
    (partition,) = (lab.work / "cache/corpus/normalized/ref_append").glob(
        "*/docs.jsonl"
    )
    staged = partition.parent.parent / (".incomplete-" + "f" * 64)
    staged.mkdir()
    (staged / "docs.jsonl").write_bytes(b"partial generation")
    inventory_args = ("corpus", "artifacts", "--json")
    inventory = lab.json(*inventory_args, "--source-id", "ref_append")
    assert inventory["published_partitions"] == 1
    assert inventory["staging_directories"] == 1
    assert inventory["retained_bytes"] == sum(
        member.stat().st_size for member in partition.parent.iterdir()
    )
    assert inventory["partitions"][0]["computation_key"] == partition.parent.name
    first = lab.json(*inventory_args, "--limit", "1")
    second = lab.json(*inventory_args, "--limit", "1", "--after", first["next_after"])
    assert (
        first["partitions"][0]["artifact_id"] < second["partitions"][0]["artifact_id"]
    )
    assert (
        second["retained_bytes"]
        == first["retained_bytes"]
        > inventory["retained_bytes"]
    )
    partition.write_bytes(b"damaged normalized partition")
    shutil.rmtree(appended_build)
    with pytest.raises(ValueError, match="prepared source shard changed"):
        lab.run("corpus", "build", str(config_path), "--offline")

    # A rights field edited after the build is refused, even with every digest resealed.
    forged = forge_release(
        release,
        edit_declared_source(
            lambda row: row["rights_policy"].update(training_eligibility="eligible"),
            source_id="code_python",
        ),
    )
    with pytest.raises(ValueError, match="differs from its build identity declaration"):
        lab.run("corpus", "audit", str(forged))

    # Widening an acquisition bound changes the acquisition identity, and the
    # transport cache serves the re-acquire without a request.
    requests = len(recipe.hub.requests)
    recipe.edit(
        "math_finemath", lambda source: source["acquisition"].update(max_rows=20_000)
    )
    wider = lab.json(
        "corpus",
        "acquire",
        prepared.project,
        "--transport-cache",
        str(tmp_path / "transport-cache"),
    )
    assert (
        wider["sources"]["math_finemath"]["declaration_sha256"]
        != prepared.lock["sources"]["math_finemath"]["declaration_sha256"]
    )
    assert len(recipe.hub.requests) == requests
    assert {
        item["served_from"]
        for item in wider["sources"]["math_finemath"]["transport_cache"]
    } == {"cache"}


def test_freeze_refuses_a_mixture_a_stratum_cannot_supply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # The budget asks systems_code for 330 tokens; its kept TRAIN bytes give
    # about 157 unique tokens at the conservative 3 bytes per token, 314 over
    # two exposures. No tokenizer is needed to know this.
    (tmp_path / "short").mkdir()
    (tmp_path / "fixed").mkdir()
    with pytest.raises(ValueError, match="cannot be supplied") as failure:
        prepare(tmp_path / "short", monkeypatch, capsys, total_tokens=2200)
    message = str(failure.value)
    assert "systems_code: kept TRAIN documents hold 472 bytes" in message
    assert "314 with max_exposures=2, but 330 are requested" in message
    assert "Lower the quota and weight together" in message
    assert message.count("kept TRAIN documents hold") == 1

    # The fix: a smaller budget. The verdict and its numbers are in the report.
    fixed = prepare(tmp_path / "fixed", monkeypatch, capsys, total_tokens=2000)
    supply = json.loads((fixed.release / "report.json").read_text())["mixture_supply"]
    assert supply["feasible"]
    assert supply["strata"]["systems_code"] == {
        "requested_tokens": 300,
        "train_utf8_bytes": 472,
        "estimated_unique_tokens": 157,
        "supply_tokens": 314,
    }


def test_strict_eligibility_fails_the_build_instead_of_dropping_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe", strict_rights=True)
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lock = lab.json("corpus", "acquire", project)
    assert sorted(lock["sources"]) == DECLARED
    # Shard files whose licenses are per-row carry no file-level SPDX
    # expression, so strict eligibility sets every one of them aside. The
    # build must say so rather than publish a corpus without those sources.
    with pytest.raises(ValueError, match="produced no documents") as failure:
        lab.run("corpus", "build", project, "--offline")
    for name in CODE_SOURCES:
        assert name in str(failure.value)
    assert not list((lab.work / "corpora").glob("*/builds/*/build.json"))


def write_family_inventory(release: Path, output: Path) -> None:
    """One row per kept document: its family (the source's), split and stratum.

    OBSTACLE: `klab corpus finalize-family-inventory` cannot produce this for a
    recipe like this one. It needs a frozen split bundle from `freeze-splits`,
    which requires a reviewed record admission, while this recipe's
    `splits.yaml` is a hand-written family map and its release builds without
    admission. The mixture requires the file, so the test writes the same rows
    from the verified release instead.
    """
    output.write_bytes(
        b"".join(
            canonical_json(
                {
                    "document_id": doc["document_id"],
                    "family_id": doc["source_family"],
                    "split": doc["split"],
                    "stratum": SOURCE_STRATA[doc["source_id"]],
                    "content_sha256": doc["content_sha256"],
                }
            )
            + b"\n"
            for doc in documents(release)
            if doc["drop_reason"] is None
        )
    )


def test_recipe_shaped_pipeline_trains_on_the_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    prepared = prepare(
        tmp_path,
        monkeypatch,
        capsys,
        placement=("--share-stage-artifacts", "--share-document-artifacts"),
    )
    lab, recipe, release = prepared.lab, prepared.recipe, prepared.release
    inventory = tmp_path / "family-inventory.jsonl"
    write_family_inventory(release, inventory)

    # Export, then fit the tokenizer on the stratified sampling plan.
    export = Path(
        lab.run(
            "corpus", "export", str(release), "--view", "lm",
            "--base-run-config", str(recipe.corpus / "export-base.yaml"),
            "--vocab-size", str(VOCAB_SIZE),
        )
    )  # fmt: skip
    tokenizer = yaml.safe_load((export / "tokenizer.yaml").read_text())
    tokenizer["chat_template"] = "chatml"
    tokenizer["sampling"] = yaml.safe_load(
        (recipe.corpus / "tokenizer-sampling.yaml").read_text()
    )
    tokenizer_config = tmp_path / "tokenizer.yaml"
    tokenizer_config.write_text(yaml.safe_dump(tokenizer))
    scratch = ("--scratch-dir", str(tmp_path / "tokenizer-scratch"))
    from klab.corpus import export as export_module

    def unexpected_verification(*args, **kwargs):
        pytest.fail("warm sampled fit replayed release or export verification")

    with monkeypatch.context() as fault:
        fault.setenv("KLAB_FAULT", "tokenizer sampling: candidates:1")
        fault.setattr(release_module, "_verify_release_cold", unexpected_verification)
        fault.setattr(
            export_module, "_verify_release_export_cold", unexpected_verification
        )
        with pytest.raises(RuntimeError, match="KLAB_FAULT"):
            lab.run("tokenizer", "train", str(tokenizer_config), *scratch)
    capsys.readouterr()
    from klab.data import tokenizer as tokenizer_module

    progress_phase = tokenizer_module.progress_phase

    for member in ("sources.json", "manifest.json"):
        control = release / member
        original = control.read_bytes()

        def change_control_before_fit(
            name, *, path=control, content=original, **options
        ):
            if name == "tokenizer_bpe_training":
                path.write_bytes(content + b"\n")
            return progress_phase(name, **options)

        try:
            with monkeypatch.context() as fault:
                fault.setattr(
                    tokenizer_module, "progress_phase", change_control_before_fit
                )
                with pytest.raises(
                    ValueError, match=f"tokenizer sampling input changed: .*{member}"
                ):
                    lab.run("tokenizer", "train", str(tokenizer_config), *scratch)
            assert not Path(tokenizer["output_dir"]).exists()
        finally:
            control.write_bytes(original)

    output = Path(tokenizer["output_dir"])

    def stop_before_manifest(path, value):
        raise RuntimeError("stopped before tokenizer manifest")

    with monkeypatch.context() as fault:
        fault.setattr(tokenizer_module, "_atomic_json", stop_before_manifest)
        with pytest.raises(RuntimeError, match="stopped before tokenizer manifest"):
            lab.run("tokenizer", "train", str(tokenizer_config), *scratch)
    assert not output.exists()

    rename = tokenizer_module.rename_noreplace

    def occupy_output_at_commit(source, destination):
        destination.mkdir()
        (destination / "keep").write_text("another publisher's data")
        rename(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(tokenizer_module, "rename_noreplace", occupy_output_at_commit)
        with pytest.raises(FileExistsError):
            lab.run("tokenizer", "train", str(tokenizer_config), *scratch)
    assert sorted(path.name for path in output.iterdir()) == ["keep"]
    assert (output / "keep").read_text() == "another publisher's data"
    (output / "keep").unlink()
    output.rmdir()

    def refuse_refit(*args, **kwargs):
        pytest.fail("retry refit an already complete staged tokenizer")

    capsys.readouterr()
    with monkeypatch.context() as fault:
        fault.setattr(tokenizer_module, "BpeTrainer", refuse_refit)
        lab.run("tokenizer", "train", str(tokenizer_config), *scratch)
    assert lab.stderr.count("tokenizer sampling: rank histogram:") == 2
    assert any(
        "tokenizer sampling: candidates:" in line and "reused 1/" in line
        for line in lab.stderr.splitlines()
    )
    clean_output = tmp_path / "clean-tokenizer"
    clean_config = tmp_path / "clean-tokenizer.yaml"
    clean_config.write_text(
        yaml.safe_dump({**tokenizer, "output_dir": str(clean_output)})
    )
    lab.run("tokenizer", "train", str(clean_config), *scratch, "--cold")
    for name in ("tokenizer.json", "tokenizer_manifest.json"):
        assert (output / name).read_bytes() == (clean_output / name).read_bytes()
    verify = ("tokenizer", "verify", str(export / "run.yaml"), "--json")
    cold = lab.json(*verify)
    assert lab.json(*verify, *scratch) == cold
    assert any(
        "tokenizer sampling: rank histogram:" in line and "reused 1/" in line
        for line in lab.stderr.splitlines()
    )
    tokenizer_json = Path(tokenizer["output_dir"]) / "tokenizer.json"
    audit = json.loads(tokenizer_json.with_name("tokenizer_manifest.json").read_text())[
        "sampling_audit"
    ]
    assert audit["complete"]
    assert set(audit["strata"]) == set(STRATA)
    for stratum, row in audit["strata"].items():
        assert row["documents"] and not row["unknown_tokens"], stratum
        assert row["roundtrip_failures"] == 0, stratum

    # Describe: every declared source, and tokens in every stratum, train and
    # (where the recipe holds a stratum out) validation.
    described = lab.json(
        "corpus", "describe", str(release), "--tokenizer", str(tokenizer_json)
    )
    assert sorted(described["report"]["source_counts"]) == DECLARED
    views = described["measurement"]["views"]["lm"]

    def tokens(split: str) -> dict[str, int]:
        return {
            stratum: row["actual_tokens"]
            for stratum, row in views[split]["dimensions"]["source_domain"].items()
        }

    assert all(tokens("train")[stratum] > 0 for stratum in STRATA)

    # The exact check runs here, before near-duplicates, export of the mixture
    # and the family inventory: the template's quotas against real tokens,
    # worded as materialize-mixture words it. The template may still hold
    # placeholders.
    template_path = recipe.corpus / "mixture.template.yaml"
    from klab.corpus import measurement

    def no_shard_merge(*args, **kwargs):
        pytest.fail("warm describe merged measurement shards")

    with monkeypatch.context() as warm:
        warm.setattr(measurement, "measure_views", no_shard_merge)
        assert lab.json(
            "corpus", "describe", str(release),
            "--tokenizer", str(tokenizer_json), "--mixture", str(template_path),
        ) == described  # fmt: skip
    [completed] = (lab.work / "cache" / "describe" / release.name).glob(
        "description/*/measurement.json"
    )
    data = completed.read_bytes()
    changed = data.replace(b'"records":', b'"recxrds":', 1)
    assert changed != data
    completed.write_bytes(changed)
    assert (
        lab.json("corpus", "describe", str(release), "--tokenizer", str(tokenizer_json))
        == described
    )
    inflated = yaml.safe_load(template_path.read_text())
    inflated["target_quotas"]["web"] = 10**6
    inflated_path = tmp_path / "inflated.template.yaml"
    inflated_path.write_text(yaml.safe_dump(inflated))
    with pytest.raises(ValueError, match="required quota impossible for web: ") as bad:
        lab.run(
            "corpus", "describe", str(release),
            "--tokenizer", str(tokenizer_json), "--mixture", str(inflated_path),
        )  # fmt: skip
    assert "1000000 required, max_exposures=2, shortfall=" in str(bad.value)
    assert set(tokens("validation")) == {SOURCE_STRATA[name] for name in HELDOUT}
    assert all(tokens("validation").values())

    # Materialize the mixture from the checked-in template and verify it.
    template = (recipe.corpus / "mixture.template.yaml").read_text()
    for key, value in {
        "RELEASE_PATH": release,
        "TOKENIZER_CONFIG": tokenizer_config,
        "FAMILY_INVENTORY": inventory,
        "RELEASE_ID": release.name,
    }.items():
        template = template.replace("${" + key + "}", str(value))
    declaration, mixture = tmp_path / "mixture.yaml", tmp_path / "mixture"
    declaration.write_text(template)
    quotas = yaml.safe_load(template)["target_quotas"]
    receipt = lab.json(
        "corpus", "materialize-mixture", str(declaration), "--output", str(mixture)
    )
    assert receipt["complete"]
    assert receipt["actual_target_tokens"] == quotas
    # The tokenizer fit has refreshed the export receipt. Mixture creation
    # authenticates the byte sample once, and the next command reuses it.
    assert lab.stderr.count("tokenizer sampling: rank histogram:") == 2
    assert "verified from the bytes" not in lab.stderr
    verified = lab.json(
        "corpus", "verify-mixture", str(declaration), "--output", str(mixture)
    )
    assert verified["status"] == "verified"
    assert "tokenizer sampling" not in lab.stderr
    assert "verified from the bytes" not in lab.stderr
    scheduled = [
        json.loads(line)
        for line in (mixture / "train.records.jsonl").read_text().splitlines()
    ]
    assert {row["stratum"] for row in scheduled} == set(STRATA)

    # Compose the pretraining config; the export's dataset section carries the
    # release revision, licence and validation counts the mixture binds to.
    bound = yaml.safe_load((export / "run.yaml").read_text())["dataset"]
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        yaml.safe_dump(
            {
                "tokenizer": {"path": str(tokenizer_json)},
                "dataset": {
                    "source": "local_token_mixture",
                    "revision": bound["revision"],
                    "license": bound["license"],
                    "mixture_declaration_path": str(declaration),
                    "mixture_output_path": str(mixture),
                    "train_path": str(mixture / "train.tokens.ids"),
                    "validation_path": str(release / "lm/validation.jsonl"),
                    "cache_dir": str(tmp_path / "prepared"),
                    "train_max_documents": receipt["records"],
                    "train_max_tokens": sum(receipt["actual_target_tokens"].values()),
                    "validation_max_documents": bound["validation_max_documents"],
                    "validation_max_tokens": bound["validation_max_tokens"],
                },
                "logging": {"root_dir": str(tmp_path / "runs")},
            }
        )
    )
    run_config = tmp_path / "pretrain.yaml"
    root = recipe.root
    lab.run(
        "config", "compose", str(root / "pretrain.yaml"),
        f"model={root / 'model.yaml'}",
        f"attention={root / 'attention.yaml'}",
        f"optimizer={root / 'optimizer.yaml'}",
        str(paths), "--output", str(run_config),
    )  # fmt: skip
    lab.run("inspect", str(run_config))
    prepared_dir = Path(lab.run("data", "prepare", str(run_config)))
    assert prepared_dir.is_dir()

    # Two CPU updates on the prepared data; nothing is tokenized or packed here.
    lab.run(
        "train",
        str(run_config),
        "--run-id",
        "mini",
        "--require-prepared",
        "--preflight",
        "waive:two-step CPU run in the recipe-shape test, not a pretraining run",
    )
    run = tmp_path / "runs/mini"
    assert json.loads((run / "progress.json").read_text())["step"] == 2
    assert (run / "checkpoints/latest.json").is_file()
