from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from klab.campaign.plan import CampaignPlan, load_campaign


def declaration() -> dict:
    return {
        "campaign_version": 1,
        "id": "tiny",
        "stages": [
            {
                "id": "corpus",
                "kind": "corpus_release",
                "scope": "corpus",
                "project": "recipe",
            },
            {
                "id": "ready",
                "kind": "corpus_readiness",
                "scope": "corpus",
                "requires": ["corpus"],
                "corpus": "corpus",
                "policy": {"min_heldout_families": 1},
            },
            {
                "id": "gate",
                "kind": "approval",
                "scope": "release",
                "requires": ["ready"],
                "bind": ["corpus", "ready"],
            },
        ],
    }


@pytest.mark.parametrize(
    "suffix,document",
    [
        ("yaml", "campaign_version: 1\ncampaign_version: 1\n"),
        ("json", '{"campaign_version":1,"campaign_version":1}'),
        ("yaml", "campaign_version: !!python/object/apply:os.system ['true']"),
        ("json", '{"campaign_version":NaN}'),
        ("yaml", "campaign_version: .nan"),
    ],
)
def test_reject_ambiguous_documents(tmp_path: Path, suffix: str, document: str) -> None:
    path = tmp_path / f"campaign.{suffix}"
    path.write_text(document)
    with pytest.raises((ValueError, TypeError)):
        load_campaign(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "duplicate_requirement",
        "unknown_requirement",
        "self_dependency",
        "cycle",
        "missing_typed_requirement",
        "wrong_kind",
        "wrong_scope",
        "unknown_field",
        "absolute_path",
        "parent_path",
        "nan",
        "empty_policy",
    ],
)
def test_reject_invalid_dag(tmp_path: Path, mutation: str) -> None:
    value = copy.deepcopy(declaration())
    corpus, ready, gate = value["stages"]
    if mutation == "duplicate_id":
        gate["id"] = ready["id"]
    elif mutation == "duplicate_requirement":
        ready["requires"] = ["corpus", "corpus"]
    elif mutation == "unknown_requirement":
        ready["requires"] = ["missing"]
    elif mutation == "self_dependency":
        ready["requires"] = ["ready"]
    elif mutation == "cycle":
        corpus["requires"] = ["gate"]
    elif mutation == "missing_typed_requirement":
        ready["requires"] = []
    elif mutation == "wrong_kind":
        ready["corpus"] = "gate"
        ready["requires"] = ["gate"]
        gate["requires"] = ["corpus"]
        gate["bind"] = ["corpus"]
    elif mutation == "wrong_scope":
        corpus["scope"] = "model"
    elif mutation == "unknown_field":
        value["execute"] = "arbitrary code"
    elif mutation == "absolute_path":
        corpus["project"] = "/etc"
    elif mutation == "parent_path":
        corpus["project"] = "../recipe"
    elif mutation == "nan":
        ready["policy"]["passes"] = {
            "basis": "bytes",
            "requested_total": 1,
            "mixture": {"developer": float("nan")},
            "max_required": 1,
        }
    elif mutation == "empty_policy":
        ready["policy"] = {}
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises((ValueError, TypeError)):
        load_campaign(path)


def test_symlink_path_rejected(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "recipe").symlink_to(tmp_path / "real", target_is_directory=True)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(declaration()))
    with pytest.raises(ValueError, match="symlink"):
        load_campaign(path)


def test_operational_artifact_accepts_external_absolute_location(
    tmp_path: Path,
) -> None:
    from klab.campaign.engine import CampaignEngine

    artifact = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "fixture",
        "identifier": "selected",
        "sha256": "a" * 64,
        "path": str(tmp_path / "external-state/tokenizer/tokenizer.json"),
    }
    source = tmp_path / "campaign.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "external-input",
                "stages": [
                    {
                        "id": "tokenizer",
                        "kind": "tokenizer_reference",
                        "scope": "tokenizer",
                        "artifact": artifact,
                    }
                ],
            }
        )
    )
    assert load_campaign(source).stages[0].artifact.path == artifact["path"]
    work = tmp_path / "workspace"
    status = CampaignEngine(source, work).inspect("status")
    assert by_id(status)["tokenizer"]["state"] == "BLOCKED"
    assert artifact["path"] in by_id(status)["tokenizer"]["reason"]
    assert not work.exists()

    artifact["path"] = str(tmp_path / "external-state/../escape.json")
    source.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "external-input",
                "stages": [
                    {
                        "id": "tokenizer",
                        "kind": "tokenizer_reference",
                        "scope": "tokenizer",
                        "artifact": artifact,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe operational path"):
        load_campaign(source)


@pytest.mark.parametrize(
    "receipt,sha,policy",
    [
        ("budget.json", None, {"min_unique_train_tokens_by_domain": {"developer": 1}}),
        (None, "a" * 64, {"min_unique_train_tokens_by_domain": {"developer": 1}}),
        (
            "budget.json",
            "A" * 64,
            {"min_unique_train_tokens_by_domain": {"developer": 1}},
        ),
        (
            "budget.json",
            "a" * 63,
            {"min_unique_train_tokens_by_domain": {"developer": 1}},
        ),
        (
            "../budget.json",
            "a" * 64,
            {"min_unique_train_tokens_by_domain": {"developer": 1}},
        ),
        (
            "budget.json",
            "a" * 64,
            {"min_unique_train_bytes_by_domain": {"developer": 1}},
        ),
        ("budget.json", "a" * 64, {"min_heldout_families": 1}),
    ],
)
def test_readiness_receipt_requires_bound_token_only_policy(
    tmp_path: Path, receipt: str | None, sha: str | None, policy: dict
) -> None:
    value = declaration()
    value["stages"].insert(
        1,
        {
            "id": "tokenizer",
            "kind": "tokenizer_reference",
            "scope": "tokenizer",
            "artifact": {
                "kind": "tokenizer",
                "version": 1,
                "producer": "fixture",
                "identifier": "selected",
                "sha256": "a" * 64,
                "path": str(tmp_path / "tokenizer.json"),
            },
        },
    )
    ready = value["stages"][2]
    ready.update(
        policy=policy,
        tokenizer="tokenizer",
        requires=["corpus", "tokenizer"],
        measurement_receipt=receipt,
        measurement_sha256=sha,
    )
    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="measurement|unsafe operational"):
        load_campaign(source)


def test_author_order_and_schema(tmp_path: Path) -> None:
    value = declaration()
    value["stages"] = [value["stages"][2], value["stages"][1], value["stages"][0]]
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    plan = load_campaign(path)
    assert [stage.id for stage in plan.ordered_stages()] == ["corpus", "ready", "gate"]
    schema = (
        Path(__file__).resolve().parents[1] / "schemas/campaign-plan-v1.schema.json"
    )
    assert json.loads(schema.read_text()) == CampaignPlan.model_json_schema()


def make_full_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Real synthetic model inputs, independent of the descriptive corpus release."""
    import shutil

    from klab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer, train_tokenizer
    from klab.training.manifest import sha256_file
    from klab.workdir import ensure_work_dir

    root = Path(__file__).resolve().parents[1]
    recipe = tmp_path / "recipe"
    shutil.copytree(root / "examples/tiny-campaign", recipe)
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    ensure_work_dir(tmp_path / "work")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    dataset = DatasetConfig(
        source="synthetic",
        cache_dir=tmp_path / "data",
        synthetic_seed=7,
        train_max_documents=100,
        validation_max_documents=100,
        train_max_tokens=129,
        validation_max_tokens=81,
    )
    tokenizer = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            max_documents=100,
            output_dir=tmp_path / "tokenizer",
            dataset=dataset,
        )
    )
    raw = yaml.safe_load((root / "configs/runtime_smoke_cpu.yaml").read_text())
    raw["name"] = "campaign-fixture"
    raw["model"]["vocab_size"] = 260
    raw["tokenizer"]["path"] = str(tokenizer)
    raw["dataset"] = dataset.model_dump(mode="json")
    raw["training"].update(gradient_accumulation=1, max_steps=2, max_tokens=32)
    raw["optimizer"]["warmup_steps"] = 1
    raw["checkpoint"]["every_steps"] = 1
    raw["evaluation"]["every_steps"] = 1
    raw["logging"]["root_dir"] = str(tmp_path / "work/runs")
    config = RunConfig.model_validate(raw)
    prepared = prepare_data(config, load_tokenizer(tokenizer))
    manifest = json.loads((prepared.root / "manifest.json").read_text())
    tokenizer_artifact = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "fixture",
        "identifier": "tokenizer",
        "sha256": sha256_file(tokenizer),
        "path": str(tokenizer.relative_to(tmp_path)),
    }
    prepared_artifact = {
        "kind": "prepared_data",
        "version": 1,
        "producer": "fixture",
        "identifier": manifest["settings_sha256"],
        "sha256": manifest["manifest_sha256"],
        "path": str(prepared.root.relative_to(tmp_path)),
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    (tmp_path / "evaluation-suite.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluation_suite_version": 1,
                "id": "fixture-heldout",
                "evaluations": [
                    {"id": "heldout", "role": "gate", "kind": "heldout_lm"}
                ],
            }
        )
    )
    (tmp_path / "model-readiness.yaml").write_text(
        yaml.safe_dump(
            {
                "readiness_version": 1,
                "id": "fixture-readiness",
                "require_verified_checkpoint": True,
                "required_gate_ids": ["heldout"],
                "min_completed_evaluations": 1,
                "max_heldout_loss": {"evaluation_id": "heldout", "value": 100.0},
                "require_human_review": False,
            }
        )
    )
    experiment = {
        "plan_version": 1,
        "id": "fixture",
        "base_run": "run.yaml",
        "evaluation_suite": "evaluation-suite.yaml",
        "artifacts": {"tokenizer": tokenizer_artifact, "prepared": prepared_artifact},
        "inputs": {"tokenizer": "tokenizer", "prepared_data": "prepared"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    stages = declaration()["stages"][:2]
    stages[0]["project"] = "recipe/corpus.yaml"
    stages.extend(
        [
            {
                "id": "tokenizer",
                "kind": "tokenizer_reference",
                "scope": "tokenizer",
                "requires": ["ready"],
                "artifact": tokenizer_artifact,
            },
            {
                "id": "prepared",
                "kind": "artifact_reference",
                "scope": "model",
                "artifact": prepared_artifact,
            },
            {
                "id": "measurement",
                "kind": "token_measurement",
                "scope": "tokenizer",
                "requires": ["corpus", "tokenizer"],
                "corpus": "corpus",
                "tokenizer": "tokenizer",
            },
            {
                "id": "plan",
                "kind": "experiment_plan",
                "scope": "model",
                "requires": ["tokenizer", "prepared", "measurement"],
                "source": "experiment.yaml",
                "mode": "lock",
                "tokenizer": "tokenizer",
                "prepared": "prepared",
            },
            {
                "id": "runtime",
                "kind": "runtime_acceptance",
                "scope": "runtime",
                "requires": ["plan"],
                "plan": "plan",
            },
            {
                "id": "gate",
                "kind": "approval",
                "scope": "model",
                "requires": ["runtime"],
                "bind": [
                    "corpus",
                    "ready",
                    "tokenizer",
                    "prepared",
                    "measurement",
                    "plan",
                    "runtime",
                ],
            },
            {
                "id": "run",
                "kind": "experiment_run",
                "scope": "model",
                "requires": ["plan", "runtime", "gate"],
                "plan": "plan",
                "runtime": "runtime",
                "cell": "main:single",
            },
            {
                "id": "collect",
                "kind": "experiment_collect",
                "scope": "evaluation",
                "requires": ["plan", "run"],
                "plan": "plan",
                "run": "run",
            },
            {
                "id": "evaluation",
                "kind": "evaluation",
                "scope": "evaluation",
                "requires": ["collect"],
                "collect": "collect",
                "suite": "evaluation-suite.yaml",
            },
            {
                "id": "model",
                "kind": "model_readiness",
                "scope": "model",
                "requires": ["evaluation"],
                "evaluation": "evaluation",
                "policy": "model-readiness.yaml",
            },
        ]
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(
        yaml.safe_dump(
            {"campaign_version": 1, "id": "fixture", "stages": stages}, sort_keys=False
        )
    )
    return path


def frozen_corpus(tmp_path: Path) -> Path:
    import shutil

    from klab.corpus.acquisition import acquire
    from klab.corpus.pipeline import build
    from klab.corpus.project import load_project
    from klab.corpus.release import freeze

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    project = load_project(tmp_path / "recipe/corpus.yaml")
    acquire(project, tmp_path / "work", offline=False)
    return freeze(build(project, tmp_path / "work", offline=True), tmp_path / "work")


def test_verified_readiness_denominator_and_deficits(tmp_path: Path) -> None:
    from klab.campaign.policy import CorpusReadinessPolicy, measure_readiness

    release = frozen_corpus(tmp_path)
    documents = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    expected = {}
    for domain in ("technical_docs", "developer"):
        unique = {
            row["content_sha256"]: row["text"]
            for row in documents
            if row["split"] == "train"
            and row["drop_reason"] is None
            and domain in row["domains"]
        }
        expected[domain] = sum(len(text.encode()) for text in unique.values())
    result = measure_readiness(
        release,
        CorpusReadinessPolicy(
            min_unique_train_bytes_by_domain=expected,
            required_nonzero_languages=("en",),
            required_nonzero_shapes=("raw_document",),
            min_heldout_families=1,
        ),
    )
    assert result["state"] == "COMPLETE"
    assert result["measurements"]["unique_train_bytes_by_domain"] == expected
    assert result["measurements"]["heldout_families"] == 1
    blocked = measure_readiness(
        release,
        CorpusReadinessPolicy(
            min_unique_train_bytes_by_domain={"absent": 1},
            required_nonzero_languages=("xx",),
            min_heldout_families=2,
        ),
    )
    assert blocked["state"] == "BLOCKED"
    assert blocked["outcome"] == "EXPAND_MORE"
    assert blocked["deficits"] == [
        {"dimension": "heldout_families", "observed": 1, "required": 2},
        {"dimension": "train_language", "observed": 0, "required": 1, "domain": "xx"},
        {
            "dimension": "unique_train_bytes",
            "observed": None,
            "required": 1,
            "domain": "absent",
        },
    ]


def test_source_pass_exact_ceil_and_zero_availability(tmp_path: Path) -> None:
    from klab.campaign.policy import CorpusReadinessPolicy, measure_readiness

    release = frozen_corpus(tmp_path)
    facts = measure_readiness(release, CorpusReadinessPolicy(min_heldout_families=1))[
        "measurements"
    ]
    amount = facts["unique_train_bytes_by_domain"]["developer"]
    for requested, expected in ((amount * 2, 1), (amount * 2 + 1, 2)):
        result = measure_readiness(
            release,
            CorpusReadinessPolicy(
                passes={
                    "basis": "bytes",
                    "requested_total": requested,
                    "mixture": {"developer": 0.5, "technical_docs": 0.5},
                    "max_required": 1,
                }
            ),
        )
        assert (
            result["measurements"]["projected_source_passes_by_domain"]["developer"]
            == expected
        )
        assert any(d["domain"] == "developer" for d in result["deficits"]) == (
            expected > 1
        )
    blocked = measure_readiness(
        release,
        CorpusReadinessPolicy(
            passes={
                "basis": "bytes",
                "requested_total": 1,
                "mixture": {"missing": 1.0},
                "max_required": 1,
            }
        ),
    )
    assert blocked["deficits"] == [
        {"dimension": "passes", "observed": None, "required": 1, "domain": "missing"}
    ]


def test_readiness_actual_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.campaign.policy import CorpusReadinessPolicy, measure_readiness
    from klab.config.models import DatasetConfig, TokenizerTrainConfig
    from klab.data.tokenizer import load_tokenizer, train_tokenizer

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    release = frozen_corpus(tmp_path)
    path = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            max_documents=100,
            output_dir=tmp_path / "tokenizer",
            dataset=DatasetConfig(
                source="synthetic",
                cache_dir=tmp_path / "data",
                synthetic_seed=7,
                train_max_documents=100,
                validation_max_documents=100,
                train_max_tokens=129,
                validation_max_tokens=81,
            ),
        )
    )
    tokenizer = load_tokenizer(path)
    docs = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    expected = sum(
        len(tokenizer.encode(row["text"]).ids)
        for row in docs
        if row["split"] == "train"
        and row["drop_reason"] is None
        and "developer" in row["domains"]
    )
    policy = CorpusReadinessPolicy(
        min_unique_train_tokens_by_domain={"developer": expected}
    )
    assert measure_readiness(release, policy)["state"] == "BLOCKED"
    result = measure_readiness(release, policy, path)
    assert result["state"] == "COMPLETE"
    assert (
        result["measurements"]["unique_train_tokens_by_domain"]["developer"] == expected
    )


def by_id(result: dict) -> dict:
    return {stage["id"]: stage for stage in result["stages"]}


def invoke_cli(
    source: Path, work: Path, command: str, *arguments: str, expected_code: int = 0
) -> dict:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "klab",
            "--work-dir",
            str(work),
            "campaign",
            command,
            str(source),
            *arguments,
            *(
                ("--allow-uncommitted-declaration",)
                if command in {"apply", "resume", "reconstruct"}
                else ()
            ),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["format"] == "klab-campaign-command-v1"
    assert envelope["command"] == command
    return envelope


def test_receipt_tampering_never_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    receipt = next((engine.store.root / "receipts/corpus").glob("*.json"))
    receipt.write_text("{}\n")
    with pytest.raises((ValueError, TypeError, KeyError)):
        engine.apply(resume=True, allow_uncommitted_declaration=True)


def test_approval_replay_rejection_and_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    approved = by_id(engine.approve("gate", note="original authorization"))
    assert approved["gate"]["state"] == "COMPLETE"
    approval = next((engine.store.root / "approvals/gate").glob("*.json"))
    before = approval.read_bytes()
    engine.approve("gate", note="replayed note must not overwrite")
    assert approval.read_bytes() == before
    with pytest.raises(ValueError):
        engine.approve("gate", decision="reject")
    value = yaml.safe_load(path.read_text())
    value["stages"][1]["policy"]["min_heldout_families"] = 2
    path.write_text(yaml.safe_dump(value))
    changed = CampaignEngine(path, tmp_path / "work")
    assert changed.store.declaration_sha != engine.store.declaration_sha
    assert by_id(changed.inspect("status"))["gate"]["state"] != "COMPLETE"


def test_changed_reference_digest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    value = yaml.safe_load(path.read_text())
    value["stages"][2]["artifact"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(value))
    result = by_id(
        CampaignEngine(path, tmp_path / "work").apply(
            allow_uncommitted_declaration=True
        )
    )
    assert result["tokenizer"]["state"] == "FAILED"
    assert result["plan"]["state"] == "BLOCKED"


@pytest.mark.parametrize("command", ["plan", "status", "next", "explain"])
def test_engine_readonly_never_creates_workspace(tmp_path: Path, command: str) -> None:
    from klab.campaign.engine import CampaignEngine

    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign.yaml"
    work = tmp_path / "nonexistent"
    result = CampaignEngine(source, work).inspect(command)
    assert not work.exists()
    rows = by_id(result)
    assert rows["corpus"]["state"] in {"NOT_STARTED", "READY"}
    assert rows["readiness"]["state"] in {"NOT_STARTED", "BLOCKED"}
    assert rows["corpus-approval"]["state"] in {"NOT_STARTED", "BLOCKED"}
    assert result["next_action"]


def test_receipt_reconciles_crash_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from klab.campaign.engine import CampaignEngine

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    value = declaration()
    value["stages"][0]["project"] = "recipe/corpus.yaml"
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    engine = CampaignEngine(path, tmp_path / "work")
    original_save = engine.store.save

    def fail_after_receipt(state):
        if list((engine.store.root / "receipts/corpus").glob("*.json")):
            raise KeyboardInterrupt("crash between receipt and state index")
        return original_save(state)

    monkeypatch.setattr(engine.store, "save", fail_after_receipt)
    with pytest.raises(KeyboardInterrupt):
        engine.apply(allow_uncommitted_declaration=True)
    receipt = next((engine.store.root / "receipts/corpus").glob("*.json"))
    before = receipt.read_bytes()
    resumed = CampaignEngine(path, tmp_path / "work")
    result = by_id(resumed.apply(resume=True, allow_uncommitted_declaration=True))
    assert result["corpus"]["state"] == "COMPLETE"
    assert result["ready"]["state"] == "COMPLETE"
    assert result["gate"]["state"] == "AWAITING_APPROVAL"
    assert receipt.read_bytes() == before


def test_stale_running_records_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from klab.campaign.engine import CampaignEngine

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    path = tmp_path / "campaign.yaml"
    value = declaration()
    value["stages"][0]["project"] = "recipe/corpus.yaml"
    path.write_text(yaml.safe_dump(value))
    engine = CampaignEngine(path, tmp_path / "work")

    def interrupt(stage, rows, max_wait_seconds, *, execute_runs=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "dispatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.apply(allow_uncommitted_declaration=True)
    assert by_id(engine.inspect("status"))["corpus"]["state"] == "RUNNING"
    recovered = CampaignEngine(path, tmp_path / "work")
    rows = by_id(recovered.apply(resume=True, allow_uncommitted_declaration=True))
    assert rows["corpus"]["state"] == "COMPLETE"
    assert [a["state"] for a in rows["corpus"]["attempts"]] == [
        "INTERRUPTED",
        "COMPLETE",
    ]
    assert rows["corpus"]["observations"][0]["state"] == "INTERRUPTED"


def test_state_receipt_conflict_is_rejected(tmp_path: Path) -> None:
    from klab.campaign.engine import CampaignEngine

    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign.yaml"
    engine = CampaignEngine(source, tmp_path / "work")
    rows = by_id(engine.apply(allow_uncommitted_declaration=True))
    changed = copy.deepcopy(rows["corpus"])
    changed["outputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="conflicting committed"):
        engine.store.commit(engine.stages["corpus"], changed)


def test_uncommitted_campaign_preflight_blocks_before_store_creation(
    tmp_path: Path,
) -> None:
    from klab.campaign.engine import CampaignEngine

    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(declaration()))
    work = tmp_path / "persistent-state"
    engine = CampaignEngine(source, work)
    with pytest.raises(ValueError, match="UNKNOWN"):
        engine.apply()
    assert not work.exists()
    with pytest.raises(ValueError, match="linked recovery manifest"):
        engine.reconstruct(allow_uncommitted_declaration=True)
    assert not work.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"profile_id": "../bad"},
        {"worker": "/absolute/path"},
        {"profile_id": "profile", "worker": "worker"},
    ],
)
def test_runtime_acceptance_rejects_unsafe_or_ambiguous_source(changes):
    from klab.campaign.plan import RuntimeAcceptance

    with pytest.raises(ValueError):
        RuntimeAcceptance(
            id="runtime",
            kind="runtime_acceptance",
            scope="runtime",
            requires=("plan",),
            plan="plan",
            **changes,
        )


def test_cpu_evaluation_cannot_inherit_accelerator_authorization():
    from klab.campaign.plan import Evaluation

    with pytest.raises(ValueError, match="CPU"):
        Evaluation(
            id="evaluate",
            kind="evaluation",
            scope="evaluation",
            collect="collect",
            suite="suite.yaml",
            backend="cpu",
            runtime="runtime",
        )
