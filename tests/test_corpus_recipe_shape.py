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
import sys
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
    Recipe,
    write_recipe,
    write_training_files,
)

from klab.canonical import canonical_json
from klab.cli.main import main
from klab.corpus import release as release_module

VOCAB_SIZE = 300
DECLARED = sorted(SOURCE_STRATA)
CODE_SOURCES = {
    "code_python",
    "code_python_heldout",
    "code_go",
    "code_shell",
    "code_systems_heldout",
}


class Lab:
    """Runs `klab` commands in this process against one isolated work root."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        self.root = root
        self.work = root / "work"
        self.work.mkdir(mode=0o700)
        self.monkeypatch = monkeypatch
        self.capsys = capsys
        monkeypatch.setenv("KLAB_WORK_DIR", str(self.work))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "settings"))

    def run(self, *argv: str) -> str:
        """The command's last line of standard output."""
        self.monkeypatch.setattr(
            sys, "argv", ["klab", "--work-dir", str(self.work), *argv]
        )
        main()
        lines = self.capsys.readouterr().out.strip().splitlines()
        return lines[-1] if lines else ""

    def json(self, *argv: str) -> dict:
        return json.loads(self.run(*argv))


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


def prepare(root: Path, monkeypatch, capsys, **recipe_options) -> Prepared:
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
    release = Path(lab.run("corpus", "freeze", str(build)))
    return Prepared(lab, recipe, lock, build_before_edit, build, release)


def documents(release: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]


def test_recipe_shaped_corpus_reaches_a_verified_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    prepared = prepare(tmp_path, monkeypatch, capsys)
    lab, recipe, release = prepared.lab, prepared.recipe, prepared.release
    docs = documents(release)

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
    prepared = prepare(tmp_path, monkeypatch, capsys)
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
    lab.run("tokenizer", "train", str(tokenizer_config))
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
    verified = lab.json(
        "corpus", "verify-mixture", str(declaration), "--output", str(mixture)
    )
    assert verified["status"] == "verified"
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
    lab.run("train", str(run_config), "--run-id", "mini", "--require-prepared")
    run = tmp_path / "runs/mini"
    assert json.loads((run / "progress.json").read_text())["step"] == 2
    assert (run / "checkpoints/latest.json").is_file()
