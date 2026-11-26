"""Offline deterministic source-mixture and fail-closed inventory checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from klab import token_encode_worker
from klab.canonical import canonical_json
from klab.config.loading import load_config
from klab.config.models import DatasetConfig
from klab.corpus import mixture
from klab.corpus.mixture_index import document_rank
from klab.corpus.release import ReleaseMembers
from klab.corpus.release_verify import verification_settings
from klab.data.packing import TokenBlockDataset, collect_token_mixture
from klab.training.manifest import sha256_file


def _manifest(release: Path) -> dict:
    """What a verified release says about the files the index reads."""
    return {
        "release_id": release.name,
        "files": {
            name: {"sha256": sha256_file(release / name)}
            for name in (
                "documents.jsonl",
                "lm/train.lineage.jsonl",
                "lm/validation.jsonl",
                "sources.json",
                "report.json",
            )
            if (release / name).is_file()
        },
    }


@pytest.fixture(autouse=True)
def two_workers():
    with verification_settings(workers=2):
        yield


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, actual_content=False
) -> tuple[Path, Path, Path]:
    release = tmp_path / ("a" * 64)
    (release / "lm").mkdir(parents=True)
    docs = [
        {
            "document_id": "g1",
            "source_id": "gutenberg",
            "split": "train",
            "drop_reason": None,
            "content_sha256": "g" * 64,
            "text": "abcdefgh",
            "rights": {"training_eligibility": "eligible_with_obligations"},
        },
        {
            "document_id": "w1",
            "source_id": "wikimedia",
            "split": "train",
            "drop_reason": None,
            "content_sha256": "w" * 64,
            "text": "abcdef",
            "rights": {"training_eligibility": "eligible_with_obligations"},
        },
        {
            "document_id": "p1",
            "source_id": "pagerduty",
            "split": "train",
            "drop_reason": None,
            "content_sha256": "p" * 64,
            "text": "abc",
            "rights": {"training_eligibility": "eligible"},
        },
        {
            "document_id": "g2",
            "source_id": "gutenberg",
            "split": "validation",
            "drop_reason": None,
            "content_sha256": "h" * 64,
            "text": "heldout",
            "rights": {"training_eligibility": "eligible_with_obligations"},
        },
    ]
    for row in docs:
        if actual_content:
            row["content_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        row["domains"] = {
            "gutenberg": ["general"],
            "wikimedia": ["explanatory"],
            "pagerduty": ["incident"],
        }[row["source_id"]]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs)
    )
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": row["document_id"]}) + b"\n"
            for row in docs
            if row["split"] == "train"
        )
    )
    (release / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": source,
                    "explicit_training_restriction": "none_found",
                    "rights_policy": {"training_eligibility": "review_required"},
                }
                for source in ("gutenberg", "wikimedia", "pagerduty")
            ]
        )
    )
    (release / "manifest.json").write_text("{}")
    (release / "report.json").write_text(
        json.dumps(
            {"requested_mixture": {"general": 0.5, "explanatory": 0.3, "incident": 0.2}}
        )
    )
    monkeypatch.setattr(mixture, "verify_release", lambda path: _manifest(release))
    monkeypatch.setattr(
        mixture,
        "_tokenizer",
        lambda *args: (
            SimpleNamespace(
                encode=lambda text, **kwargs: SimpleNamespace(ids=list(text.encode())),
                token_to_id=lambda token: 3 if token == "<eos>" else None,
            ),
            hashlib.sha256(b"{}").hexdigest(),
        ),
    )
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer/tokenizer.json").write_bytes(b"{}")
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda path: SimpleNamespace(output_dir=tmp_path / "tokenizer"),
    )
    inventory = tmp_path / "families.jsonl"
    strata = {
        "gutenberg": "general",
        "wikimedia": "explanatory",
        "pagerduty": "incident",
    }
    inventory.write_bytes(
        b"".join(
            canonical_json(
                {
                    "document_id": row["document_id"],
                    "family_id": "family-" + row["document_id"],
                    "split": row["split"],
                    "stratum": strata[row["source_id"]],
                    "content_sha256": row["content_sha256"],
                }
            )
            + b"\n"
            for row in docs
        )
    )
    declaration = tmp_path / "mixture.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(release),
                "tokenizer_config": str(tmp_path / "tokenizer.yaml"),
                "family_inventory": str(inventory),
                "source_strata": strata,
                "target_quotas": {"general": 10, "explanatory": 6, "incident": 4},
                "seed": 42,
                "max_exposures": 2,
            }
        )
    )
    return declaration, inventory, release


def _scheduled(output: Path) -> list[dict]:
    """Record rows joined with their target IDs from the binary export."""
    ids = np.fromfile(output / "train.tokens.ids", dtype="<i4").tolist()
    rows, position = [], 0
    for line in (output / "train.records.jsonl").read_text().splitlines():
        row = json.loads(line)
        row["token_ids"] = ids[position : position + row["tokens"]]
        position += row["tokens"]
        rows.append(row)
    assert position == len(ids)
    return rows


def test_card03_order_policy_uses_contract_document_hash() -> None:
    expected = hashlib.sha256(b"kml-card03-v1 | incident | document-1").hexdigest()
    assert (
        document_rank(17, "incident", "family-a", "document-1", "kml-card03-v1")
        == expected
    )
    assert (
        document_rank(17, "incident", "family-b", "document-1", "kml-card03-v1")
        == expected
    )


def test_exact_quotas_repeats_and_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    assert receipt["actual_target_tokens"] == {
        "explanatory": 6,
        "general": 10,
        "incident": 4,
    }
    assert receipt["unique_target_positions"] == {
        "explanatory": 6,
        "general": 9,
        "incident": 4,
    }
    assert receipt["quota_shortfalls"] == {
        "general": 0,
        "explanatory": 0,
        "incident": 0,
    }
    rows = _scheduled(output)
    assert {row["document_id"] for row in rows} == {"g1", "w1", "p1"}
    assert max(row["exposure"] for row in rows) == 2
    assert all(row["token_ids"][-1] == 3 for row in rows)
    assert all(len(row["token_ids"]) > row["content_token_end"] for row in rows)
    assert "token_ids" not in (output / "train.records.jsonl").read_text()
    assert mixture.verify_mixture(declaration, output) == receipt


def test_completion_survives_unrelated_lab_changes_not_relevant_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    package = Path(mixture.__file__).parents[1]
    read = Path.read_bytes

    def edited(*names: str) -> None:
        changed = {package / name for name in names}
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: read(self) + (b"\n# edited" if self in changed else b""),
        )

    edited("memory.py", "engines/base.py", "cli/main.py")
    assert mixture.verify_mixture(declaration, output) == receipt
    edited("corpus/mixture_index.py")
    with pytest.raises(ValueError, match="--reseal"):
        mixture.verify_mixture(declaration, output)


def test_reseal_rebuilds_lost_completion_without_replay_and_refuses_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    policy = yaml.safe_load(declaration.read_bytes())
    policy["strata"] = {"general": {"max_exposures": policy["max_exposures"]}}
    declaration.write_text(yaml.safe_dump(policy))
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    shutil.rmtree(tmp_path / "cache")
    with pytest.raises(ValueError, match="--reseal"):
        mixture.verify_mixture(declaration, output)
    monkeypatch.setattr(
        mixture,
        "_materialize_mixture",
        lambda *a, **k: pytest.fail("reseal must not replay"),
    )
    with pytest.raises(ValueError, match="never"):
        mixture.verify_mixture(declaration, output, audit=True, reseal=True)
    records = output / "train.records.jsonl"
    opened = []
    original_open = Path.open

    def count_records(path, mode="r", *args, **kwargs):
        if path == records and mode == "rb":
            opened.append(path)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as observe:
        observe.setattr(Path, "open", count_records)
        assert mixture.verify_mixture(declaration, output, reseal=True) == receipt
        assert mixture.verify_mixture(declaration, output) == receipt
    assert opened == [records, records]

    original = records.read_bytes()
    changed = original.replace(b'"document_id":"g1"', b'"document_id":"x1"', 1)
    assert changed != original and len(changed) == len(original)
    records.write_bytes(changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        mixture.verify_mixture(declaration, output)
    records.write_bytes(original)

    shutil.rmtree(tmp_path / "cache")
    tokens = output / "train.tokens.ids"
    data = bytearray(tokens.read_bytes())
    data[0] ^= 1
    tokens.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        mixture.verify_mixture(declaration, output, reseal=True)
    with pytest.raises(ValueError, match="digest mismatch"):
        mixture.verify_mixture(declaration, output)


def test_accepted_65_25_10_target_shares_are_realized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    spec = yaml.safe_load(declaration.read_text())
    spec["target_quotas"] = {"general": 13, "explanatory": 5, "incident": 2}
    declaration.write_text(yaml.safe_dump(spec))
    (release / "report.json").write_text(
        json.dumps(
            {
                "requested_mixture": {
                    "general": 0.65,
                    "explanatory": 0.25,
                    "incident": 0.10,
                }
            }
        )
    )
    receipt = mixture.materialize_mixture(declaration, tmp_path / "accepted")
    assert receipt["actual_target_tokens"] == spec["target_quotas"]
    assert receipt["max_exposures"] == 2
    assert receipt["unique_target_positions"]["general"] >= 7
    assert receipt["unique_target_positions"]["explanatory"] >= 3
    assert receipt["unique_target_positions"]["incident"] >= 1


def test_shortfall_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    spec = yaml.safe_load(declaration.read_text())
    spec["target_quotas"]["incident"] = 9
    declaration.write_text(yaml.safe_dump(spec))
    (tmp_path / ("a" * 64) / "report.json").write_text(
        json.dumps(
            {
                "requested_mixture": {
                    "general": 10 / 25,
                    "explanatory": 6 / 25,
                    "incident": 9 / 25,
                }
            }
        )
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="quota impossible"):
        mixture.materialize_mixture(declaration, output)
    assert not output.exists()


def test_family_leak_and_rights_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, inventory, release = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[-1]["family_id"] = rows[0]["family_id"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="family leaks"):
        mixture.materialize_mixture(declaration, tmp_path / "leak")
    rows[-1]["family_id"] = "other"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    docs = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    docs[0]["rights"]["training_eligibility"] = "review_required"
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs)
    )
    with pytest.raises(ValueError, match="lacks admitted"):
        mixture.materialize_mixture(declaration, tmp_path / "rights")


def test_changed_export_and_inventory_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, inventory, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    mixture.materialize_mixture(declaration, output)
    for name, extra in (
        ("train.tokens.ids", b"\0\0\0\0"),
        ("train.records.jsonl", b"\n"),
    ):
        original = (output / name).read_bytes()
        (output / name).write_bytes(original + extra)
        with pytest.raises(ValueError, match="digest mismatch"):
            mixture.verify_mixture(declaration, output)
        (output / name).write_bytes(original)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[0]["content_sha256"] = "z" * 64
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="disagrees"):
        mixture.materialize_mixture(declaration, tmp_path / "changed")


def test_heldout_row_in_train_view_and_wrong_share_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    with (release / "lm/train.lineage.jsonl").open("ab") as stream:
        stream.write(canonical_json({"record_id": "g2"}) + b"\n")
    with pytest.raises(ValueError, match="train LM view"):
        mixture.materialize_mixture(declaration, tmp_path / "leaked")
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": doc_id}) + b"\n"
            for doc_id in ("g1", "w1", "p1")
        )
    )
    report = {
        "requested_mixture": {"general": 0.65, "explanatory": 0.25, "incident": 0.1}
    }
    (release / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="target quotas disagree"):
        mixture.materialize_mixture(declaration, tmp_path / "wrong_share")


def test_native_packing_supervises_exact_exported_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=output / "train.tokens.ids"),
        training=SimpleNamespace(seq_len=8),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    (tmp_path / "pack").mkdir()
    ids, mask, stats = collect_token_mixture(
        config, tokenizer, receipt, tmp_path / "pack"
    )
    assert ids[0] == 2 and not mask[0]
    assert stats["scheduled_target_positions"] == 20
    assert stats["supervised_target_positions"] == 20
    assert stats["masked_padding_positions"] == 4
    blocks = TokenBlockDataset(ids, 8, supervision=mask)
    targets = [blocks.numpy_block(index)[1] for index in range(len(blocks))]
    assert sum(int((block != -100).sum()) for block in targets) == 20
    assert sum(int((block == -100).sum()) for block in targets) == 4
    assert [value for block in targets for value in block if value != -100] == [
        value for row in _scheduled(output) for value in row["token_ids"]
    ]


def test_yaml_mixture_consumer_and_storage_identity_are_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Release authentication and tokenizer encoding use the tiny synthetic fixture;
    # mixture materialization, cold replay, dataset binding and packing are real.
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    assets = tmp_path / "source" / "nested"
    assets.mkdir(parents=True)
    declaration, _, release = _fixture(assets, monkeypatch)
    output = assets / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    (release / "lm/validation.jsonl").write_text('{"text":"heldout"}\n')
    fit_dir = assets / "fit"
    fit_dir.mkdir()
    tokenizer = _byte_tokenizer()
    tokenizer.save(str(fit_dir / "tokenizer.json"))
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda *args: SimpleNamespace(output_dir=fit_dir, vocab_size=260),
    )
    from klab.corpus import export

    monkeypatch.setattr(
        export, "release_licenses", lambda sources, documents: "local research"
    )
    raw = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    ).model_dump(mode="json")
    raw["model"]["vocab_size"] = 260
    # The seven-character validation fixture must contain a full next-token block.
    raw["training"]["seq_len"] = 4
    raw["logging"]["root_dir"] = "runs"
    raw["tokenizer"]["path"] = "fit/tokenizer.json"
    raw["dataset"] = {
        "source": "local_token_mixture",
        "revision": release.name,
        "cache_dir": "cache",
        "train_path": "output/train.tokens.ids",
        "validation_path": f"{release.name}/lm/validation.jsonl",
        "train_max_documents": receipt["records"],
        "validation_max_documents": 1,
        "train_max_tokens": 20,
        "validation_max_tokens": 9,
        "license": "local research",
        "mixture_declaration_path": "mixture.yaml",
        "mixture_output_path": "output",
    }
    source = assets / "run.yaml"
    source.write_text(yaml.safe_dump(raw))
    from klab.data.packing import prepare_data
    from klab.workspace_preflight import training_storage_checks

    for cwd in (tmp_path / "cwd-a", tmp_path / "unrelated" / "cwd-b"):
        cwd.mkdir(parents=True)
        monkeypatch.chdir(cwd)
        config = load_config(source)
        verified = mixture.verify_mixture_dataset(
            config.dataset, config.tokenizer.path, config.model.vocab_size
        )
        assert verified == receipt
        (cwd / "pack").mkdir()
        ids, mask, stats = collect_token_mixture(
            config, tokenizer, verified, cwd / "pack"
        )
        assert stats["supervised_target_positions"] == 20
        assert sum(mask) == 20
        assert len(ids) == len(mask)
        prepared = prepare_data(config, tokenizer)
        checks = training_storage_checks(
            config, work_dir=tmp_path / "work", verified_prepared=prepared.receipt
        )
        relocated = config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={
                        "mixture_declaration_path": cwd / "relocated-mixture.yaml",
                        "mixture_output_path": cwd / "relocated-output",
                    }
                )
            }
        )
        relocated_checks = training_storage_checks(
            relocated, work_dir=tmp_path / "work", verified_prepared=prepared.receipt
        )
        assert [item.projected_bytes for item in relocated_checks] == [
            item.projected_bytes for item in checks
        ]
        changed = config.model_copy(
            update={
                "dataset": config.dataset.model_copy(update={"train_max_tokens": 21})
            }
        )
        with pytest.raises(ValueError, match="does not match run configuration"):
            training_storage_checks(
                changed, work_dir=tmp_path / "work", verified_prepared=prepared.receipt
            )


@pytest.mark.parametrize("damage", ["missing_eos", "out_of_vocabulary", "record"])
def test_native_packing_rejects_malformed_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = dict(mixture.materialize_mixture(declaration, output))
    tokens, records = output / "train.tokens.ids", output / "train.records.jsonl"
    ids = np.fromfile(tokens, dtype="<i4")
    if damage == "record":
        rows = [json.loads(line) for line in records.read_text().splitlines()]
        rows[0]["tokens"] -= 1
        records.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
        receipt["train_records_sha256"] = hashlib.sha256(
            records.read_bytes()
        ).hexdigest()
    else:
        ids[-1] = 5 if damage == "missing_eos" else 260
        ids.tofile(tokens)
        receipt["train_tokens_sha256"] = hashlib.sha256(tokens.read_bytes()).hexdigest()
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=tokens),
        training=SimpleNamespace(seq_len=8),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    (tmp_path / "pack").mkdir()
    with pytest.raises(ValueError, match="token mixture"):
        collect_token_mixture(config, tokenizer, receipt, tmp_path / "pack")


LONG_TEXT = "abcdefghijklmnopqrst"


def _document_start_fixture(tmp_path, monkeypatch, seq_len=8):
    """One 21-target document under 8-target rows, and a stratum repeated 3x."""
    declaration, inventory, release = _fixture(
        tmp_path, monkeypatch, actual_content=True
    )
    documents = release / "documents.jsonl"
    docs = [json.loads(line) for line in documents.read_text().splitlines()]
    docs[0]["text"] = LONG_TEXT
    docs[0]["content_sha256"] = hashlib.sha256(LONG_TEXT.encode()).hexdigest()
    documents.write_bytes(b"".join(canonical_json(row) + b"\n" for row in docs))
    families = [json.loads(line) for line in inventory.read_text().splitlines()]
    for row in families:
        if row["document_id"] == docs[0]["document_id"]:
            row["content_sha256"] = docs[0]["content_sha256"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in families))
    quotas = {"explanatory": 5, "general": len(LONG_TEXT) + 1, "incident": 12}
    total = sum(quotas.values())
    (release / "report.json").write_text(
        json.dumps(
            {"requested_mixture": {key: value / total for key, value in quotas.items()}}
        )
    )
    spec = yaml.safe_load(declaration.read_text())
    spec |= {
        "target_quotas": quotas,
        "max_exposures": 1,
        "packing": "document_start",
        "row_tokens": 8,
        "strata": {
            "general": {"min_document_tokens": 9},
            "incident": {"max_exposures": 3},
        },
    }
    declaration.write_text(yaml.safe_dump(spec))
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=tmp_path / "output" / "train.tokens.ids"),
        training=SimpleNamespace(seq_len=seq_len),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    (tmp_path / "pack").mkdir()
    return declaration, config, tokenizer


def test_document_start_rows_and_stratum_exposures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, config, tokenizer = _document_start_fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    assert mixture.verify_mixture(declaration, output) == receipt
    assert receipt["stratum_exposures"] == {
        "explanatory": {"records": [1], "target_tokens": [5]},
        "general": {"records": [1], "target_tokens": [21]},
        "incident": {"records": [1, 1, 1], "target_tokens": [4, 4, 4]},
    }
    # explanatory fills targets 0-4; the long document starts on row 1 after
    # three masked positions and fills rows 1 and 2 alone; incident shares row 3.
    assert receipt["row_layout"] == {
        "layout": "document_start",
        "row_tokens": 8,
        "rows": 6,
        "filler_positions": 3,
        "strata": {"general": {"min_document_tokens": 9, "documents": 1, "rows": 2}},
    }
    ids, mask, stats = collect_token_mixture(
        config, tokenizer, receipt, tmp_path / "pack"
    )
    assert stats["supervised_target_positions"] == 38
    assert stats["masked_padding_positions"] == 10
    from klab.workspace_preflight import projected_data_bytes

    # The storage allowance (staging and published copies, past its fixed
    # reserve) covers the packed rows, filler included, not just the targets.
    planned = SimpleNamespace(
        dataset=SimpleNamespace(
            source="local_token_mixture",
            mixture_output_path=output,
            train_max_tokens=38,
            validation_max_tokens=0,
        ),
        model=SimpleNamespace(memory="none"),
        training=SimpleNamespace(mixture_prepared_storage="contiguous"),
    )
    assert projected_data_bytes(planned) - 32 * 1024**2 >= 2 * (
        ids.nbytes + mask.nbytes
    )
    blocks = TokenBlockDataset(ids, 8, supervision=mask)
    long_ids = [*LONG_TEXT.encode(), 3]
    rows = [blocks.numpy_block(index)[1].tolist() for index in range(len(blocks))]
    # The long document fills consecutive rows in order, its first row read
    # after an EOS; a shuffle of whole training rows cannot mix in another.
    assert rows[0][5:] == [-100] * 3
    assert blocks.numpy_block(1)[0][0] == 3
    assert rows[1] == long_ids[:8] and rows[2] == long_ids[8:16]
    assert rows[3][:5] == long_ids[16:]

    from klab.corpus.mixture_schedule import verify_schedule
    from klab.data import mixture_batches
    from klab.probes import contamination

    reference_declaration = tmp_path / "reference.yaml"
    reference_declaration.write_text(
        declaration.read_text() + "\ntoken_cache: sealed\ntoken_output: references\n"
    )
    references = tmp_path / "references"
    reference_receipt = mixture.materialize_mixture(reference_declaration, references)
    inspected, transfer = mixture.inspect_mixture_transfer(
        reference_declaration, references
    )
    assert inspected == reference_receipt
    assert transfer["selected_ranges"] == {
        "unique_token_bytes": 4 * 27,
        "repeated_token_bytes": 4 * 6,
        "coalesced_ranges": 2,
        "segments_without_selected_content": 0,
    }
    assert transfer["whole_segments"]["unselected_token_bytes"] == 4 * 2
    schedule = verify_schedule(
        references, tmp_path / ".source-tokens", reference_receipt, 3
    )
    monkeypatch.setattr(mixture_batches, "_PAGE_RECORDS", 2)
    with mixture_batches.ReferencedTokenArray(
        schedule,
        references / "train.records.jsonl",
        reference_receipt,
        seq_len=8,
        bos=2,
        pad=0,
    ) as source:
        # The exact scan must also catch a window that crosses a slice boundary.
        monkeypatch.setattr(contamination, "_CHUNK", 8)
        planted = tuple(int(token) for token in ids[14:27])
        assert contamination.find_windows(source, {planted}) == {planted}
        reference_blocks = TokenBlockDataset(source, 8, supervision=mask)
        assert reference_blocks.exposure_supply() == blocks.exposure_supply()
        for index in reversed(range(len(blocks))):
            expected, target, _ = blocks.numpy_microblock(index)
            actual, reference_target, _ = reference_blocks.numpy_microblock(index)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(reference_target, target)


@pytest.mark.parametrize("damage", ["receipt_count", "seq_len"])
def test_document_start_refuses_mismatched_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    declaration, config, tokenizer = _document_start_fixture(
        tmp_path, monkeypatch, seq_len=16 if damage == "seq_len" else 8
    )
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    if damage == "seq_len":
        with pytest.raises(ValueError, match="rows hold 8 targets"):
            collect_token_mixture(config, tokenizer, receipt, tmp_path / "pack")
        return
    receipt["row_layout"]["strata"]["general"]["rows"] = 3
    (output / "receipt.json").write_bytes(canonical_json(receipt) + b"\n")
    with pytest.raises(ValueError, match="document rows differ"):
        mixture.verify_mixture(declaration, output)


def test_tokenizer_accepts_retained_release_path_but_not_a_different_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_id = "a" * 64
    release = tmp_path / ("b" * 64)
    dataset = SimpleNamespace(
        source="local_text", corpus_release_path=release, revision=release_id
    )
    config = SimpleNamespace(
        dataset=dataset, output_dir=tmp_path / "fit", vocab_size=260
    )
    spec = SimpleNamespace(
        tokenizer_config=tmp_path / "config",
        tokenizer_origin_release_id=release_id,
        segment_dtype="int32",
    )
    monkeypatch.setattr(mixture, "load_tokenizer_config", lambda path: config)
    monkeypatch.setattr(
        mixture,
        "verify_tokenizer_artifact",
        lambda *args, **kwargs: {"sha256": "t" * 64},
    )
    monkeypatch.setattr(mixture, "load_tokenizer", lambda path: "tokenizer")
    members = ReleaseMembers(release, {"release_id": release_id})
    assert mixture._tokenizer(spec, members) == ("tokenizer", "t" * 64)
    dataset.revision = "c" * 64
    with pytest.raises(ValueError, match="not bound to this release"):
        mixture._tokenizer(spec, members)
    dataset.revision = release_id
    dataset.corpus_release_path = tmp_path / ("d" * 64)
    with pytest.raises(ValueError, match="not bound to this release"):
        mixture._tokenizer(spec, members)
    dataset.corpus_release_path = release
    spec.segment_dtype = "uint16"
    vocab = {"<eos>": 3, "last": 65535}
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab)
    monkeypatch.setattr(mixture, "load_tokenizer", lambda path: tokenizer)
    assert mixture._tokenizer(spec, members) == (tokenizer, "t" * 64)
    vocab["outside"] = 65536
    with pytest.raises(ValueError, match="vocabulary IDs must fit uint16"):
        mixture._tokenizer(spec, members)


def test_tokenizer_verification_uses_the_operators_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise every mixture command re-verifies the export and re-derives the sample."""
    from klab.corpus.release_verify import verification_settings

    release_id = "a" * 64
    dataset = SimpleNamespace(
        source="local_text", corpus_release_path=tmp_path, revision=release_id
    )
    config = SimpleNamespace(dataset=dataset, output_dir=tmp_path, vocab_size=260)
    spec = SimpleNamespace(
        tokenizer_config=tmp_path / "config",
        tokenizer_origin_release_id=release_id,
        segment_dtype="int32",
    )
    seen = []
    monkeypatch.setattr(mixture, "load_tokenizer_config", lambda path: config)
    monkeypatch.setattr(mixture, "load_tokenizer", lambda path: "tokenizer")
    monkeypatch.setattr(
        mixture,
        "verify_tokenizer_artifact",
        lambda *args, **kwargs: seen.append(kwargs) or {"sha256": "t" * 64},
    )
    store = object()
    with verification_settings(proof_store=store, verification_mode="verified_reuse"):
        mixture._tokenizer(spec, ReleaseMembers(tmp_path, {"release_id": release_id}))
    assert (seen[0]["proof_store"], seen[0]["verification_mode"]) == (
        store,
        "verified_reuse",
    )


@pytest.mark.parametrize("change_during_run", [False, True])
def test_mixture_records_and_rechecks_opt_in_tokenizer_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change_during_run: bool
) -> None:
    original_tokenizer = mixture._tokenizer
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mixture, "_tokenizer", original_tokenizer)
    (release / "manifest.json").write_text(json.dumps({"release_id": release.name}))
    data = SimpleNamespace(
        source="local_text", corpus_release_path=release, revision=release.name
    )
    config = SimpleNamespace(dataset=data, output_dir=tmp_path / "fit", vocab_size=260)
    config.output_dir.mkdir()
    (config.output_dir / "tokenizer.json").write_bytes(b"{}")
    monkeypatch.setattr(mixture, "load_tokenizer_config", lambda path: config)
    monkeypatch.setattr(
        mixture,
        "verify_tokenizer_artifact",
        lambda *args, **kwargs: pytest.fail("cold tokenizer replay must not run"),
    )
    monkeypatch.setattr(
        mixture,
        "load_tokenizer",
        lambda path: SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=list(text.encode())),
            token_to_id=lambda token: 3 if token == "<eos>" else None,
        ),
    )
    from klab.corpus import tokenizer_observation

    calls = []

    def observe(*args):
        calls.append(1)
        if change_during_run and len(calls) == 2:
            raise ValueError("observed input changed")
        return {
            "method": "post_run_observation",
            "tokenizer_sha256": hashlib.sha256(b"{}").hexdigest(),
        }

    monkeypatch.setattr(tokenizer_observation, "verify_observed_tokenizer", observe)
    data = yaml.safe_load(declaration.read_text())
    names = (
        "job_spec",
        "session_spec",
        "job_report",
        "session_report",
        "container_report",
        "stdout",
        "stderr",
        "tokenizer_config",
        "tokenizer",
        "tokenizer_manifest",
        "export",
    )
    data["tokenizer_observation"] = {
        "method": "post_run_observation",
        "observed_at_utc": "2026-10-29T05:00:00Z",
        "files": {
            name: {"path": str(tmp_path / name), "sha256": "a" * 64} for name in names
        },
    }
    declaration.write_text(yaml.safe_dump(data))
    output = tmp_path / "observed-output"
    if change_during_run:
        with pytest.raises(ValueError, match="observed input changed"):
            mixture.materialize_mixture(declaration, output)
        assert not output.exists()
    else:
        receipt = mixture.materialize_mixture(declaration, output)
        assert receipt["tokenizer_observation"]["method"] == "post_run_observation"
    assert len(calls) == 2


def test_mixture_dataset_binds_train_validation_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    train = output / "train.tokens.ids"
    train.write_bytes(b"")
    validation = release / "lm/validation.jsonl"
    validation.write_text('{"text":"heldout"}\n')
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        mixture,
        "_verify_mixture",
        lambda *args, **kwargs: (
            {
                "release_id": release.name,
                "actual_target_tokens": {"x": 20},
                "records": 3,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda *args: SimpleNamespace(output_dir=fit_dir, vocab_size=260),
    )
    from klab.corpus import export

    monkeypatch.setattr(
        export, "release_licenses", lambda sources, documents: "local research"
    )
    dataset = DatasetConfig(
        source="local_token_mixture",
        revision=release.name,
        cache_dir=tmp_path / "cache",
        train_path=train,
        validation_path=validation,
        train_max_documents=3,
        validation_max_documents=1,
        train_max_tokens=20,
        validation_max_tokens=9,
        license="local research",
        mixture_declaration_path=declaration,
        mixture_output_path=output,
    )
    assert (
        mixture.verify_mixture_dataset(dataset, fit_dir / "tokenizer.json", 260)[
            "records"
        ]
        == 3
    )
    wrong = dataset.model_copy(update={"validation_path": tmp_path / "other.jsonl"})
    with pytest.raises(ValueError, match="identity mismatch"):
        mixture.verify_mixture_dataset(wrong, fit_dir / "tokenizer.json", 260)


def _byte_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    special = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(
        models.BPE(
            vocab={
                **special,
                **{item: index + 4 for index, item in enumerate(alphabet)},
            },
            merges=[],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tokenizer


def test_output_is_identical_for_any_worker_count(tmp_path, monkeypatch):
    """Workers read and tokenize rows themselves; the output does not depend on how many."""
    from klab.data.encoding import PreparationEncoder

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    tokenizer = _byte_tokenizer()
    stand_in = mixture._tokenizer
    monkeypatch.setattr(
        mixture, "_tokenizer", lambda *args: (tokenizer, stand_in(*args)[1])
    )
    operations = []
    submit = PreparationEncoder.submit

    def observe(self, request):
        operations.append(request[:1])
        return submit(self, request)

    monkeypatch.setattr(PreparationEncoder, "submit", observe)
    receipts = {}
    for workers in (1, 3):
        with verification_settings(workers=workers):
            receipts[workers] = mixture.materialize_mixture(
                declaration, tmp_path / f"workers-{workers}"
            )
    assert receipts[1] == receipts[3]
    assert set(operations) == {b"D"}
    # Killed after a committed segment with the source read ahead, resumed on
    # another worker count.
    from klab.corpus.mixture_pipeline import MixtureFrontier
    from klab.data import encoding

    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 1)
    monkeypatch.setenv("KLAB_MIXTURE_COMMIT_CANDIDATES", "1")
    commit = MixtureFrontier.commit

    def interrupt(self, *args):
        commit(self, *args)
        if self.sequence == 2:
            raise InterruptedError("after a durable commit")

    monkeypatch.setattr(MixtureFrontier, "commit", interrupt)
    with verification_settings(workers=2), pytest.raises(InterruptedError):
        mixture.materialize_mixture(declaration, tmp_path / "resumed")
    monkeypatch.setattr(MixtureFrontier, "commit", commit)
    with verification_settings(workers=1):
        assert (
            mixture.materialize_mixture(declaration, tmp_path / "resumed")
            == (receipts[1])
        )
    assert (tmp_path / "resumed" / "train.tokens.ids").read_bytes() == (
        tmp_path / "workers-1" / "train.tokens.ids"
    ).read_bytes()
    for name in ("train.tokens.ids", "train.records.jsonl"):
        assert (tmp_path / "workers-1" / name).read_bytes() == (
            tmp_path / "workers-3" / name
        ).read_bytes()


def test_real_batch_matches_single_document_output_and_warm_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import struct

    from klab.corpus.retained_mixture import RetainedMixtureIndex
    from klab.corpus.split_runs import write_rows
    from klab.data import encoding
    from klab.data.encoding import PreparationEncoder

    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    config_root = tmp_path / "private-config"
    config_root.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
    workspace = root / "corpora" / "tiny"
    release = workspace / "builds" / "build"
    release.mkdir(parents=True)
    (release / "manifest.json").write_text("{}\n")
    source = workspace / "render.jsonl"
    refs = workspace / "references.jsonl"
    parents = workspace / "parents.bin"
    origins = workspace / "origins.jsonl"
    evidence = workspace / "evidence.json"
    family_inventory = workspace / "family-inventory.jsonl"
    evidence.write_text("{}\n")
    origins.write_bytes(b"")
    family_inventory.write_bytes(b"fixture\n")
    refs_data = []
    documents = [
        ("one", "abcdefgh", "a" * 64),
        ("two", "abcd", "b" * 64),
        ("duplicate", "abcdefgh", "a" * 64),
        ("empty1", "", "c" * 64),
        ("large", "longer than eight", "d" * 64),
        ("three", "efgh", "e" * 64),
    ]
    with source.open("wb") as stream:
        for name, text, _content in documents:
            content = hashlib.sha256(text.encode()).hexdigest()
            row = {
                "document_id": name,
                "source_id": "source",
                "content_sha256": content,
                "text": text,
                "domains": ["general"],
                "rights": {"training_eligibility": "eligible"},
            }
            raw = canonical_json(row) + b"\n"
            offset = stream.tell()
            stream.write(raw)
            refs_data.append(
                [
                    name,
                    0,
                    offset,
                    len(raw),
                    "train",
                    None,
                    content,
                    None,
                    None,
                    None,
                    ["general"],
                    "family-" + name,
                    "general",
                ]
            )
    write_rows(refs, refs_data)
    parents.write_bytes(b"".join(struct.pack("<Q", i) for i in range(len(documents))))
    proof = {"completion_path": str(evidence), "cache_path": str(evidence)}
    cohort = {
        "release_id": "a" * 64,
        "manifest": {"corpus_id": "tiny"},
        "documents": len(documents),
        "references": str(refs),
        "parents": str(parents),
        "origins": str(origins),
        "sources": [str(source)],
        "source_ids": ["source"],
        "evidence": {
            "references": proof,
            "deduplication": proof,
            "origins": proof,
            "sources": {"source": proof},
        },
    }
    spec = SimpleNamespace(
        target_quotas={"general": 10},
        unique_train_content_floors={"general": 1},
        max_exposures=1,
        strata={},
        packing="contiguous",
        token_cache="append",
        token_output="contiguous",
        seed=42,
        order_policy=None,
        family_inventory=family_inventory,
        additional_releases=[],
    )
    tokenizer = _byte_tokenizer()
    calls = []
    original_submit = PreparationEncoder.submit

    def observe_batch(self, request):
        texts = token_encode_worker.unpack_texts(request[1:])
        calls.append((len(texts), sum(len(text.encode("utf-8")) for text in texts)))
        return original_submit(self, request)

    monkeypatch.setattr(PreparationEncoder, "submit", observe_batch)

    def run(name, *, reuse_tokens=True, checkpoint=False):
        import tempfile

        from klab.corpus.admission_inspection_cache import InspectionCache
        from klab.corpus.mixture_pipeline import MixtureFrontier, pipeline_request

        staging = root / name
        staging.mkdir(exist_ok=checkpoint)
        scratch = Path(tempfile.mkdtemp(dir=staging))
        with RetainedMixtureIndex(
            scratch, root, [cohort], 42, {"source": "general"}
        ) as index:
            frontier = None
            if checkpoint:
                frontier = MixtureFrontier(
                    root / f".{name}.assembly-fixture",
                    InspectionCache(root),
                    pipeline_request(
                        {"tokenizer": "a" * 64, "declaration": "fixture"}, index
                    ),
                )
            result = mixture._schedule(
                spec,
                b"declaration",
                ReleaseMembers(release, {"release_id": "a" * 64}),
                tokenizer,
                "a" * 64,
                3,
                index,
                staging,
                reuse_tokens=reuse_tokens,
                frontier=frontier,
            )
        return result, (
            (staging / "train.tokens.ids").read_bytes(),
            (staging / "train.records.jsonl").read_bytes(),
        )

    result, output = run("batched")
    assert any(size > 1 for size, _ in calls)
    assert [row["reason"] for row in result["excluded"]] == [
        "zero_encoded_targets",
        "duplicate_train_content",
    ]
    assert result["actual_target_tokens"] == {"general": 10}
    with monkeypatch.context() as worker_patch:
        import psutil

        worker_patch.setattr(psutil, "cpu_count", lambda **kwargs: 4)
        worker_patch.setattr(psutil.Process, "cpu_affinity", lambda self: [0, 1, 2, 3])
        for name, count in (("three-workers", 3), ("four-workers", 4)):
            with verification_settings(workers=count):
                assert run(name, reuse_tokens=False) == (result, output)
            progress = json.loads((root / name / "progress.json").read_bytes())
            assert (progress["read_workers"], progress["encoder_rayon_threads"]) == (
                1,
                count,
            )

    calls.clear()
    warm_result, warm_output = run("warm")
    original_source = source.read_bytes()
    source.write_bytes(
        original_source.replace(
            b'"training_eligibility":"eligible"',
            b'"training_eligibility":"blocked!"',
            1,
        )
    )
    with pytest.raises(ValueError, match="disagrees with executed metadata"):
        run("warm-ineligible")
    source.write_bytes(original_source)
    assert (warm_result, warm_output) == (result, output)
    assert calls == []

    # Small limits exercise the real batch boundaries with tiny files.
    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 2)
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 64)
    calls.clear()
    bounded_result, bounded_output = run("bounded", reuse_tokens=False)
    assert (bounded_result, bounded_output) == (result, output)
    assert calls and all(
        size <= 2 and source_bytes <= 64 for size, source_bytes in calls
    )

    # A document over the byte limit is a batch of its own.
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 1)
    calls.clear()
    serial_result, serial_output = run("serial", reuse_tokens=False)
    assert (serial_result, serial_output) == (result, output)
    assert calls and all(size == 1 for size, _ in calls)

    from klab.corpus.mixture_pipeline import MixtureFrontier

    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 1)
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 64)
    monkeypatch.setenv("KLAB_MIXTURE_COMMIT_CANDIDATES", "4096")
    assert run("grouped", checkpoint=True) == (result, output)
    assert len(list(_assembly(root, "grouped").glob("commit-*.json"))) == 1
    with monkeypatch.context() as cadence:
        cadence.setattr(mixture, "MIXTURE_COMMIT_SEGMENT_BYTES", 1)
        assert run("byte-commits", checkpoint=True) == (result, output)
        assert len(list(_assembly(root, "byte-commits").glob("commit-*.json"))) > 1
    with monkeypatch.context() as cadence:
        cadence.setattr(mixture, "MIXTURE_COMMIT_INTERVAL_SECONDS", 0)
        assert run("time-commits", checkpoint=True) == (result, output)
        assert len(list(_assembly(root, "time-commits").glob("commit-*.json"))) > 1
    monkeypatch.setenv("KLAB_MIXTURE_COMMIT_CANDIDATES", "1")
    assert run("eager", checkpoint=True) == (result, output)
    assert len(list(_assembly(root, "eager").glob("commit-*.json"))) > 1
    commit = MixtureFrontier.commit

    def interrupt(self, *args):
        commit(self, *args)
        if self.sequence == 3:
            raise InterruptedError("retained frontier committed")

    monkeypatch.setattr(MixtureFrontier, "commit", interrupt)
    with pytest.raises(InterruptedError):
        run("resumed", checkpoint=True)
    monkeypatch.setattr(MixtureFrontier, "commit", commit)
    # The resumed run has a different worker count than the interrupted one.
    with verification_settings(workers=3):
        assert run("resumed", checkpoint=True) == (result, output)

    # Batched padding changes IDs, so padded tokenizers retain serial semantics.
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    assert (
        tokenizer.encode_batch(["abcdefgh", "abcd"], add_special_tokens=False)[1].ids
        != tokenizer.encode("abcd", add_special_tokens=False).ids
    )
    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 256)
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 8 * 1024**2)
    calls.clear()
    padded_result, padded_output = run("padded", reuse_tokens=False)
    assert calls == []
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 1)
    padded_serial, padded_serial_output = run("padded-serial", reuse_tokens=False)
    assert (padded_result, padded_output) == (padded_serial, padded_serial_output)


def test_real_batch_cache_survives_interrupted_caller(tmp_path: Path) -> None:
    from klab.data.encoding import PreparationEncoder
    from klab.data.token_store import ContentTokens

    tokenizer = _byte_tokenizer()
    texts = ["plain", "café", "plain", "other"]
    expected = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]

    def listed(arrays):
        return [ids.tolist() for ids in arrays]

    with (
        pytest.raises(RuntimeError, match="simulated interruption"),
        PreparationEncoder(tokenizer, max_workers=4) as encoder,
        ContentTokens(tmp_path / "cache", "b" * 64) as cache,
    ):
        assert listed(cache.encode_many(texts, encoder.encode)) == expected
        raise RuntimeError("simulated interruption")

    def unexpected(_texts):
        raise AssertionError("completed encodings must be reused")

    with ContentTokens(tmp_path / "cache", "b" * 64) as cache:
        assert listed(cache.encode_many(texts, unexpected)) == expected
        mixed = ["plain", "fresh", "other", "novel"]
        with PreparationEncoder(tokenizer, max_workers=4) as encoder:
            assert listed(cache.encode_many(mixed, encoder.encode)) == [
                tokenizer.encode(text, add_special_tokens=False).ids for text in mixed
            ]
        stored = cache.path / "tokens.ids"
    damaged = bytearray(stored.read_bytes())
    damaged[0] ^= 1
    stored.write_bytes(damaged)
    with (
        ContentTokens(tmp_path / "cache", "b" * 64) as cache,
        pytest.raises(ValueError, match="cached token payload digest mismatch"),
    ):
        cache.encode_many(["plain"], unexpected)
    # An index record whose IDs never reached the disk is encoded again.
    stored.write_bytes(bytes(damaged[:4]))
    with (
        ContentTokens(tmp_path / "cache", "b" * 64) as cache,
        PreparationEncoder(tokenizer, max_workers=1) as encoder,
    ):
        assert len(cache) == 0
        assert listed(cache.encode_many(texts, encoder.encode)) == expected
        with pytest.raises(ValueError, match="in use by another build"):
            ContentTokens(tmp_path / "cache", "b" * 64)


def _assembly(parent, name):
    (found,) = parent.glob(f".{name}.assembly-*")
    return found


@pytest.mark.parametrize("after_commit", [False, True])
def test_mixture_frontier_resumes_without_committed_candidate_replay(
    tmp_path, monkeypatch, after_commit
):
    from klab.corpus.mixture_pipeline import MixtureFrontier
    from klab.data import encoding

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline"
    expected = mixture._materialize_mixture(declaration, baseline, reuse_tokens=False)
    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 1)
    monkeypatch.setenv("KLAB_MIXTURE_COMMIT_CANDIDATES", "1")
    output = tmp_path / "interrupted"
    original = MixtureFrontier.commit

    def interrupt(self, *args):
        if self.sequence == 1 and not after_commit:
            raise InterruptedError("before durable commit")
        original(self, *args)
        if self.sequence == 2 and after_commit:
            raise InterruptedError("after durable commit")

    monkeypatch.setattr(MixtureFrontier, "commit", interrupt)
    with pytest.raises(InterruptedError):
        mixture.materialize_mixture(declaration, output)
    assert not output.exists()
    checkpoints = _assembly(tmp_path, "interrupted")
    last = json.loads(max(checkpoints.glob("commit-*.json")).read_bytes())
    committed = sum(last["state"]["candidates"].values())
    assert {path.name for path in Path(last["directory"]).iterdir()} == {
        "index.jsonl",
        "excluded.jsonl",
    }
    abandoned_scratch = checkpoints / ".mixture-index-abandoned"
    abandoned_scratch.mkdir()
    staged_inodes = {
        name: (checkpoints / "payload" / name).stat().st_ino
        for name in ("train.tokens.ids", "train.records.jsonl")
    }
    visited = []
    original_cached = mixture.MixtureIndex.cached_document

    def observe(self, doc_id, *args):
        visited.append(doc_id)
        return original_cached(self, doc_id, *args)

    monkeypatch.setattr(MixtureFrontier, "commit", original)
    monkeypatch.setattr(mixture.MixtureIndex, "cached_document", observe)

    def unexpected_inventory_replay(*args):
        raise AssertionError("authenticated preflight must bypass inventory replay")

    monkeypatch.setattr(mixture.MixtureIndex, "inventory", unexpected_inventory_replay)
    actual = mixture.materialize_mixture(declaration, output)
    assert actual == expected
    for name in ("train.tokens.ids", "train.records.jsonl"):
        assert (output / name).read_bytes() == (baseline / name).read_bytes()
        assert (output / name).stat().st_ino == staged_inodes[name]
    assert len(visited) == 3 - committed
    # Segments kept for the resume are removed once the output is published;
    # the progress report stays.
    assert not list(checkpoints.glob("segment-*"))
    assert not list(checkpoints.glob("commit-*.json"))
    assert not abandoned_scratch.exists()
    assert (checkpoints / "progress.json").is_file()


@pytest.mark.parametrize("damage", ["committed bytes", "marker"])
def test_mixture_frontier_rejects_changed_checkpoint(tmp_path, monkeypatch, damage):
    from klab.corpus.mixture_pipeline import MixtureFrontier

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "interrupted"
    original = MixtureFrontier.commit

    def interrupt(self, *args):
        original(self, *args)
        if self.state["outputs"]["train.tokens.ids"]["end"]:
            raise InterruptedError("after durable commit")

    monkeypatch.setattr(MixtureFrontier, "commit", interrupt)
    with pytest.raises(InterruptedError):
        mixture.materialize_mixture(declaration, output)
    monkeypatch.setattr(MixtureFrontier, "commit", original)
    checkpoints = _assembly(tmp_path, "interrupted")
    marker = max(checkpoints.glob("commit-*.json"))
    entry = json.loads(marker.read_bytes())
    if damage == "committed bytes":
        path = checkpoints / "payload" / "train.tokens.ids"
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        entry["state"]["records"] += 1
        marker.write_bytes(canonical_json(entry))
    with pytest.raises(ValueError, match="checkpoint"):
        mixture.materialize_mixture(declaration, output)
    assert not output.exists()


def test_checkpoints_are_named_by_request_and_conflicts_refuse_before_verification(
    tmp_path, monkeypatch, capsys
):
    from klab.corpus.mixture_pipeline import MixtureFrontier

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "interrupted"
    original = MixtureFrontier.commit

    def interrupt(self, *args):
        original(self, *args)
        raise InterruptedError("after durable commit")

    monkeypatch.setattr(MixtureFrontier, "commit", interrupt)
    with pytest.raises(InterruptedError):
        mixture.materialize_mixture(declaration, output)
    monkeypatch.setattr(MixtureFrontier, "commit", original)
    first = _assembly(tmp_path, "interrupted")
    # Another declaration beside the same output never meets that checkpoint.
    unchanged = declaration.read_bytes()
    declaration.write_bytes(unchanged + b"\n# another declaration\n")
    capsys.readouterr()
    mixture.materialize_mixture(declaration, output)
    assert first.is_dir()
    assert first.name in capsys.readouterr().err
    shutil.rmtree(output)
    # A checkpoint that claims another request is refused before verification
    # or indexing, naming what to delete.
    declaration.write_bytes(unchanged)
    binding = first / "request.json"
    stored = json.loads(binding.read_bytes())
    binding.write_bytes(canonical_json({**stored, "declaration_sha256": "0" * 64}))

    def verification_ran(*args):
        raise AssertionError("verification ran before the checkpoint check")

    monkeypatch.setattr(mixture, "verify_release", verification_ran)
    with pytest.raises(ValueError, match="delete that directory") as refused:
        mixture.materialize_mixture(declaration, output)
    assert str(first) in str(refused.value)


def test_ordered_source_fetch_bounds_reorder_and_propagates_failure(monkeypatch):
    import threading

    from klab.corpus import mixture_pipeline

    assert list(
        mixture_pipeline.ordered_fetch(range(3), str, workers=0, size=lambda i: 1)
    ) == [(i, str(i)) for i in range(3)]
    monkeypatch.setattr(mixture_pipeline, "PREFETCH_BYTES", 3)
    later = threading.Event()
    metrics = {}

    def fetch(i):
        if i == 0:
            assert later.wait(5)
        if i == 1:
            later.set()
        if i == 4:
            raise OSError("source read failed")
        return str(i)

    iterator = mixture_pipeline.ordered_fetch(
        range(6), fetch, workers=4, size=lambda i: 1, metrics=metrics
    )
    assert [next(iterator) for _ in range(4)] == [(i, str(i)) for i in range(4)]
    with pytest.raises(OSError, match="source read failed"):
        next(iterator)
    assert metrics["read_queue_bytes_peak"] <= 3
    assert metrics["read_queue_records_peak"] <= 3
    assert metrics["read_active_peak"] >= 2


def test_mixture_writer_rejects_concurrent_owner(tmp_path):
    from klab.corpus.mixture_pipeline import mixture_writer

    with (
        mixture_writer(tmp_path / "output"),
        pytest.raises(ValueError, match="active writer"),
        mixture_writer(tmp_path / "output"),
    ):
        pytest.fail("two writers acquired the same output")


def test_admitted_ordinary_metadata_can_bypass_warm_source_fetch(tmp_path, monkeypatch):
    declaration, _, _ = _fixture(tmp_path, monkeypatch, actual_content=True)
    cold = mixture.materialize_mixture(declaration, tmp_path / "cold")
    warm = mixture.materialize_mixture(declaration, tmp_path / "warm")
    assert cold == warm
    progress = json.loads((_assembly(tmp_path, "warm") / "progress.json").read_bytes())
    assert progress["cache_hits"] == 3
    assert progress.get("read_worker_seconds", 0) == 0


def test_sealed_mixture_recovers_published_segments_and_refuses_corrupt_tokens(
    tmp_path, monkeypatch
):
    from klab.data import token_segments

    declaration, _, _ = _fixture(tmp_path, monkeypatch, actual_content=True)
    declaration.write_text(
        declaration.read_text()
        + "\ntoken_cache: sealed\nsegment_dtype: uint16\ntoken_output: references\n"
    )
    monkeypatch.setattr(token_segments, "SHARD_TOKENS", 8)
    publish = token_segments.rename_noreplace

    def stop_after_publish(source, destination):
        publish(source, destination)
        if destination.parent.name == "segments":
            raise InterruptedError("token segment published before locator commit")

    output = tmp_path / "interrupted"
    with monkeypatch.context() as interruption:
        interruption.setattr(token_segments, "rename_noreplace", stop_after_publish)
        with pytest.raises(InterruptedError):
            mixture.materialize_mixture(declaration, output)
    (committed,) = tmp_path.glob(".source-tokens/sealed/*/segments/*/tokens.ids")
    first_inode = committed.stat().st_ino
    assert not output.exists()

    def stop_before_schedule_verification(*args):
        raise InterruptedError("schedule derived before publication")

    with monkeypatch.context() as interruption:
        interruption.setattr(
            mixture, "verify_schedule", stop_before_schedule_verification
        )
        with pytest.raises(InterruptedError):
            mixture.materialize_mixture(declaration, output)
    checkpoints = _assembly(tmp_path, "interrupted")
    # A derived schedule can be rebuilt from the authenticated record prefix.
    (checkpoints / "payload" / mixture.TRAIN_SCHEDULE).write_bytes(b"uncommitted")

    receipt = mixture.materialize_mixture(declaration, output)
    assert not (output / mixture.TRAIN_TOKENS).exists()
    assert mixture.verify_mixture(declaration, output, audit=True) == receipt
    assert len(list(committed.parent.parent.iterdir())) == 3
    assert committed.stat().st_ino == first_inode

    warm = tmp_path / "warm"
    assert mixture.materialize_mixture(declaration, warm) == receipt
    progress = json.loads((_assembly(tmp_path, "warm") / "progress.json").read_bytes())
    assert progress["sealed_token_cache_hits"] == 3
    assert progress["published_token_segments"] == 0
    for name in (mixture.TRAIN_SCHEDULE, mixture.TRAIN_RECORDS):
        assert (warm / name).read_bytes() == (output / name).read_bytes()

    with committed.open("r+b") as corrupt:
        first = corrupt.read(1)
        corrupt.seek(0)
        corrupt.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="file digest mismatch"):
        mixture.inspect_mixture_transfer(declaration, output)


def test_uint16_segments_preserve_the_upper_boundary_and_refuse_wrapping(tmp_path):
    from klab.data.token_segments import SealedContentTokens

    content = hashlib.sha256(b"boundary")
    with SealedContentTokens(tmp_path, "a" * 64, storage="uint16") as store:
        with pytest.raises(ValueError, match="token IDs must fit uint16"):
            store.store(content.digest(), np.array([65536], dtype=np.int32))
        store.store(content.digest(), np.array([0, 65535], dtype=np.int32))
        store.finish()
    with SealedContentTokens(tmp_path, "a" * 64, storage="uint16") as store:
        ids = store.lookup(content.hexdigest())
        assert ids.dtype == np.dtype("<i4")
        assert ids.tolist() == [0, 65535]
        assert store.lookup(content.hexdigest(), decode=False) == b"\x00\x00\xff\xff"


def test_worker_telemetry_survives_a_vanished_worker():
    import subprocess
    import sys

    from klab.corpus.mixture_pipeline import process_tree_sample

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    sample = process_tree_sample([gone.pid])
    assert sample["process_observation_errors"] == ["NoSuchProcess"]
    assert not sample["child_sample_complete"]


def test_source_prefetch_continues_while_consumer_is_encoding():
    import threading

    from klab.corpus.mixture_pipeline import ordered_fetch

    ready = threading.Event()
    lock = threading.Lock()
    fetched = 0

    def fetch(item):
        nonlocal fetched
        with lock:
            fetched += 1
            if fetched == 8:
                ready.set()
        return item

    iterator = ordered_fetch(range(8), fetch, workers=2, size=lambda item: 1)
    assert next(iterator) == (0, 0)
    # Pause the coordinator as encoding does. Both readers must continue
    # fetching beyond the old four-record window without another next().
    assert ready.wait(5)
    assert list(iterator) == [(i, i) for i in range(1, 8)]


def test_retained_handle_eviction_does_not_invalidate_active_read(
    tmp_path, monkeypatch
):
    import os
    import threading
    from collections import OrderedDict
    from concurrent.futures import ThreadPoolExecutor

    from klab.corpus.retained_mixture import RetainedMixtureIndex

    index = RetainedMixtureIndex.__new__(RetainedMixtureIndex)
    index.handles = OrderedDict()
    index.handle_lock = threading.Lock()
    index.handle_limit = 1
    sources, references = [], []
    for i in range(2):
        path = tmp_path / f"source-{i}.jsonl"
        raw = (
            canonical_json(
                {
                    "document_id": str(i),
                    "source_id": "source",
                    "content_sha256": str(i),
                    "domains": ["general"],
                    "rights": {"training_eligibility": "eligible"},
                    "text": str(i),
                }
            )
            + b"\n"
        )
        path.write_bytes(raw)
        sources.append(path)
        references.append(
            ["general", 0, str(i), 0, i, 0, len(raw), str(i), "family", "source"]
        )
    index.cohorts = [{"sources": sources}]
    entered, release = threading.Event(), threading.Event()
    original = os.pread

    def pause_first(fd, size, offset):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(fd, size, offset)

    monkeypatch.setattr(os, "pread", pause_first)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(index._read_document, references[0])
            assert entered.wait(5)
            try:
                assert (
                    pool.submit(index._read_document, references[1]).result(5)["text"]
                    == "1"
                )
            finally:
                release.set()
            assert first.result(5)["text"] == "0"
        assert len(index.handles) == 1
        handle = next(iter(index.handles.values()))
        assert index._read_document(references[1])["text"] == "1"
        assert next(iter(index.handles.values())) is handle
    finally:
        release.set()
        for handle in index.handles.values():
            handle.close()


def test_ordinary_fetch_reads_records_by_position(tmp_path, monkeypatch):
    """Records come from positional reads of one descriptor, not a file per record."""
    import os

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline"
    expected = mixture._materialize_mixture(declaration, baseline, reuse_tokens=False)
    reads = []
    original = os.pread

    def counted(descriptor, size, offset):
        reads.append((size, offset))
        return original(descriptor, size, offset)

    monkeypatch.setattr(os, "pread", counted)
    actual = mixture._materialize_mixture(
        declaration, tmp_path / "positional", reuse_tokens=False
    )
    assert actual == expected
    assert reads
    for name in ("train.tokens.ids", "train.records.jsonl"):
        assert (tmp_path / "positional" / name).read_bytes() == (
            baseline / name
        ).read_bytes()
