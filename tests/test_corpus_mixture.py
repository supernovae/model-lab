"""Offline deterministic source-mixture and fail-closed inventory checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.canonical import canonical_json
from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig
from sparselab.corpus import mixture
from sparselab.data.packing import TokenBlockDataset, _collect_token_mixture


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
    monkeypatch.setattr(
        mixture, "verify_release", lambda path: {"release_id": release.name}
    )
    monkeypatch.setattr(
        mixture,
        "_tokenizer",
        lambda *args: (
            SimpleNamespace(
                encode=lambda text, **kwargs: SimpleNamespace(ids=list(text.encode())),
                token_to_id=lambda token: 3 if token == "<eos>" else None,
            ),
            "t" * 64,
        ),
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


def test_card03_order_policy_uses_contract_document_hash() -> None:
    expected = hashlib.sha256(b"kml-card03-v1 | incident | document-1").hexdigest()
    assert (
        mixture._rank(17, "incident", "family-a", "document-1", "kml-card03-v1")
        == expected
    )
    assert (
        mixture._rank(17, "incident", "family-b", "document-1", "kml-card03-v1")
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
    rows = [
        json.loads(line)
        for line in (output / "train.tokens.jsonl").read_text().splitlines()
    ]
    assert {row["document_id"] for row in rows} == {"g1", "w1", "p1"}
    assert max(row["exposure"] for row in rows) == 2
    assert all(row["token_ids"][-1] == 3 for row in rows)
    assert all(len(row["token_ids"]) > row["content_token_end"] for row in rows)
    assert mixture.verify_mixture(declaration, output) == receipt


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
    with (output / "train.tokens.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        mixture.verify_mixture(declaration, output)
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
        dataset=SimpleNamespace(train_path=output / "train.tokens.jsonl"),
        training=SimpleNamespace(seq_len=8),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    (tmp_path / "pack").mkdir()
    ids, mask, stats = _collect_token_mixture(
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
        value
        for row in mixture._rows(output / "train.tokens.jsonl")
        for value in row["token_ids"]
    ]


def test_yaml_mixture_consumer_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Release authentication and tokenizer encoding use the tiny synthetic fixture;
    # mixture materialization, cold replay, dataset binding and packing are real.
    assets = tmp_path / "source" / "nested"
    assets.mkdir(parents=True)
    declaration, _, release = _fixture(assets, monkeypatch)
    output = assets / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    (release / "lm/validation.jsonl").write_text('{"text":"heldout"}\n')
    fit_dir = assets / "fit"
    fit_dir.mkdir()
    (fit_dir / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda *args: SimpleNamespace(output_dir=fit_dir, vocab_size=260),
    )
    from sparselab.corpus import export

    monkeypatch.setattr(export, "_licenses", lambda path: "local research")
    raw = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    ).model_dump(mode="json")
    raw["model"]["vocab_size"] = 260
    raw["training"]["seq_len"] = 8
    raw["tokenizer"]["path"] = "fit/tokenizer.json"
    raw["dataset"] = {
        "source": "local_token_mixture",
        "revision": release.name,
        "cache_dir": "cache",
        "train_path": "output/train.tokens.jsonl",
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
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    for cwd in (tmp_path / "cwd-a", tmp_path / "unrelated" / "cwd-b"):
        cwd.mkdir(parents=True)
        monkeypatch.chdir(cwd)
        config = load_config(source)
        verified = mixture.verify_mixture_dataset(
            config.dataset, config.tokenizer.path, config.model.vocab_size
        )
        assert verified == receipt
        (cwd / "pack").mkdir()
        ids, mask, stats = _collect_token_mixture(
            config, tokenizer, verified, cwd / "pack"
        )
        assert stats["supervised_target_positions"] == 20
        assert sum(mask) == 20
        assert len(ids) == len(mask)


def test_native_packing_rejects_malformed_target_row(tmp_path: Path) -> None:
    path = tmp_path / "train.tokens.jsonl"
    path.write_text(json.dumps({"token_ids": [5, 5]}) + "\n")
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=path),
        training=SimpleNamespace(seq_len=4),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    with pytest.raises(ValueError, match="invalid token mixture"):
        _collect_token_mixture(
            config,
            tokenizer,
            {
                "actual_target_tokens": {"x": 2},
                "records": 1,
                "train_tokens_sha256": "0" * 64,
            },
            tmp_path,
        )


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
        tokenizer_config=tmp_path / "config", tokenizer_origin_release_id=release_id
    )
    monkeypatch.setattr(mixture, "load_tokenizer_config", lambda path: config)
    monkeypatch.setattr(
        mixture,
        "verify_tokenizer_artifact",
        lambda *args, **kwargs: {"sha256": "t" * 64},
    )
    monkeypatch.setattr(mixture, "load_tokenizer", lambda path: "tokenizer")
    assert mixture._tokenizer(spec, release, release_id) == ("tokenizer", "t" * 64)
    dataset.revision = "c" * 64
    with pytest.raises(ValueError, match="not bound to this release"):
        mixture._tokenizer(spec, release, release_id)
    dataset.revision = release_id
    dataset.corpus_release_path = tmp_path / ("d" * 64)
    with pytest.raises(ValueError, match="not bound to this release"):
        mixture._tokenizer(spec, release, release_id)


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
    from sparselab.corpus import tokenizer_observation

    calls = []

    def observe(*args):
        calls.append(1)
        if change_during_run and len(calls) == 2:
            raise ValueError("observed input changed")
        return {"method": "post_run_observation", "tokenizer_sha256": "t" * 64}

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
    train = output / "train.tokens.jsonl"
    train.write_text("{}\n")
    validation = release / "lm/validation.jsonl"
    validation.write_text('{"text":"heldout"}\n')
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        mixture,
        "verify_mixture",
        lambda *args: {
            "release_id": release.name,
            "actual_target_tokens": {"x": 20},
            "records": 3,
        },
    )
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda *args: SimpleNamespace(output_dir=fit_dir, vocab_size=260),
    )
    from sparselab.corpus import export

    monkeypatch.setattr(export, "_licenses", lambda path: "local research")
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


def test_real_batch_matches_single_document_output_and_warm_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import struct

    from sparselab.corpus.retained_mixture import RetainedMixtureIndex
    from sparselab.corpus.split_runs import write_rows
    from sparselab.data import encoding
    from sparselab.data.encoding import PreparationEncoder

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
        seed=42,
        order_policy=None,
        family_inventory=family_inventory,
        additional_releases=[],
    )
    tokenizer = _byte_tokenizer()
    calls = []
    original_encode = PreparationEncoder.encode

    def observe_batch(self, texts):
        calls.append((len(texts), sum(len(text.encode("utf-8")) for text in texts)))
        return original_encode(self, texts)

    monkeypatch.setattr(PreparationEncoder, "encode", observe_batch)

    def run(name, *, reuse_tokens=True, checkpoint=False):
        import tempfile

        from sparselab.corpus.admission_inspection_cache import InspectionCache
        from sparselab.corpus.mixture_pipeline import MixtureFrontier, pipeline_request

        staging = root / name
        staging.mkdir(exist_ok=checkpoint)
        scratch = Path(tempfile.mkdtemp(dir=staging))
        with RetainedMixtureIndex(
            scratch, root, [cohort], 42, {"source": "general"}
        ) as index:
            frontier = None
            if checkpoint:
                frontier = MixtureFrontier(
                    staging,
                    InspectionCache(root),
                    pipeline_request(
                        {"tokenizer": "a" * 64, "declaration": "fixture"}, index
                    ),
                )
            result = mixture._schedule(
                spec,
                b"declaration",
                release,
                {"release_id": "a" * 64},
                tokenizer,
                "a" * 64,
                3,
                index,
                staging,
                reuse_tokens=reuse_tokens,
                frontier=frontier,
            )
        return result, (staging / "train.tokens.jsonl").read_bytes()

    result, output = run("batched")
    assert any(size > 1 for size, _ in calls)
    assert [row["reason"] for row in result["excluded"]] == [
        "zero_encoded_targets",
        "duplicate_train_content",
    ]
    assert result["actual_target_tokens"] == {"general": 10}
    with monkeypatch.context() as worker_patch:
        import psutil

        worker_patch.setenv("SPARSELAB_MIXTURE_WORKERS", "3")
        worker_patch.setattr(psutil, "cpu_count", lambda **kwargs: 4)
        worker_patch.setattr(psutil.Process, "cpu_affinity", lambda self: [0, 1, 2, 3])
        assert run("three-workers", reuse_tokens=False) == (result, output)
        progress = json.loads((root / "three-workers/progress.json").read_bytes())
        assert (progress["read_workers"], progress["encoder_rayon_threads"]) == (1, 2)

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

    # A tiny limit exercises the same fallback used above the real 8 MiB bound.
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 1)
    calls.clear()
    serial_result, serial_output = run("serial", reuse_tokens=False)
    assert (serial_result, serial_output) == (result, output)
    assert calls == []

    from sparselab.corpus.mixture_pipeline import MixtureFrontier

    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 1)
    monkeypatch.setattr(mixture, "MIXTURE_BATCH_SOURCE_BYTES", 64)
    monkeypatch.setenv("SPARSELAB_MIXTURE_COMMIT_CANDIDATES", "4096")
    assert run("grouped", checkpoint=True) == (result, output)
    assert len(list((root / ".grouped.assembly").glob("commit-*.json"))) == 1
    with monkeypatch.context() as cadence:
        cadence.setattr(mixture, "MIXTURE_COMMIT_SEGMENT_BYTES", 1)
        assert run("byte-commits", checkpoint=True) == (result, output)
        assert len(list((root / ".byte-commits.assembly").glob("commit-*.json"))) > 1
    with monkeypatch.context() as cadence:
        cadence.setattr(mixture, "MIXTURE_COMMIT_INTERVAL_SECONDS", 0)
        assert run("time-commits", checkpoint=True) == (result, output)
        assert len(list((root / ".time-commits.assembly").glob("commit-*.json"))) > 1
    monkeypatch.setenv("SPARSELAB_MIXTURE_COMMIT_CANDIDATES", "1")
    assert run("eager", checkpoint=True) == (result, output)
    assert len(list((root / ".eager.assembly").glob("commit-*.json"))) > 1
    commit = MixtureFrontier.commit

    def interrupt(self, *args):
        commit(self, *args)
        if self.sequence == 3:
            raise InterruptedError("retained frontier committed")

    monkeypatch.setattr(MixtureFrontier, "commit", interrupt)
    with pytest.raises(InterruptedError):
        run("resumed", checkpoint=True)
    monkeypatch.setattr(MixtureFrontier, "commit", commit)
    monkeypatch.setenv("SPARSELAB_MIXTURE_WORKERS", "8")
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
    from sparselab.corpus.token_cache import TokenCache
    from sparselab.data.encoding import PreparationEncoder

    tokenizer = _byte_tokenizer()
    texts = ["plain", "café", "plain", "other"]
    expected = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    with (
        pytest.raises(RuntimeError, match="simulated interruption"),
        PreparationEncoder(tokenizer, tmp_path, max_workers=4) as encoder,
        TokenCache(tmp_path / "cache", "b" * 64) as cache,
    ):
        assert cache.encode_many(texts, encoder.encode) == expected
        raise RuntimeError("simulated interruption")

    def unexpected(_texts):
        raise AssertionError("completed encodings must be reused")

    with TokenCache(tmp_path / "cache", "b" * 64) as cache:
        assert cache.encode_many(texts, unexpected) == expected
        mixed = ["plain", "fresh", "other", "novel"]
        with PreparationEncoder(tokenizer, tmp_path, max_workers=4) as encoder:
            assert cache.encode_many(mixed, encoder.encode) == [
                tokenizer.encode(text, add_special_tokens=False).ids for text in mixed
            ]
        digest = hashlib.sha256(b"plain").hexdigest()
        cache.db.execute(
            "UPDATE tokens SET digest=? WHERE content=?", ("0" * 64, digest)
        )
        with pytest.raises(ValueError, match="cached token payload digest mismatch"):
            cache.encode_many(["plain"], unexpected)


@pytest.mark.parametrize("after_commit", [False, True])
def test_mixture_frontier_resumes_without_committed_candidate_replay(
    tmp_path, monkeypatch, after_commit
):
    from sparselab.corpus.mixture_pipeline import MixtureFrontier
    from sparselab.data import encoding

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline"
    expected = mixture._materialize_mixture(declaration, baseline, reuse_tokens=False)
    monkeypatch.setattr(encoding, "TOKENIZER_BATCH_DOCUMENTS", 1)
    monkeypatch.setenv("SPARSELAB_MIXTURE_WORKERS", "1")
    monkeypatch.setenv("SPARSELAB_MIXTURE_COMMIT_CANDIDATES", "1")
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
    checkpoints = tmp_path / ".interrupted.assembly"
    last = json.loads(max(checkpoints.glob("commit-*.json")).read_bytes())
    committed = sum(last["state"]["candidates"].values())
    visited = []
    original_cached = mixture.MixtureIndex.cached_document

    def observe(self, doc_id, *args):
        visited.append(doc_id)
        return original_cached(self, doc_id, *args)

    monkeypatch.setattr(MixtureFrontier, "commit", original)
    monkeypatch.setattr(mixture.MixtureIndex, "cached_document", observe)

    def unexpected_inventory_replay(*args):
        raise AssertionError("authenticated preflight must bypass inventory replay")

    monkeypatch.setattr(
        mixture.MixtureIndex, "_load_documents", unexpected_inventory_replay
    )
    monkeypatch.setenv("SPARSELAB_MIXTURE_WORKERS", "8")
    actual = mixture.materialize_mixture(declaration, output)
    assert actual == expected
    assert (output / "train.tokens.jsonl").read_bytes() == (
        baseline / "train.tokens.jsonl"
    ).read_bytes()
    assert len(visited) == 3 - committed


@pytest.mark.parametrize("damage", ["output", "index", "identity", "marker"])
def test_mixture_frontier_rejects_changed_checkpoint(tmp_path, monkeypatch, damage):
    from sparselab.corpus.mixture_pipeline import MixtureFrontier

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
    checkpoints = tmp_path / ".interrupted.assembly"
    marker = next(checkpoints.glob("commit-*.json"))
    entry = json.loads(marker.read_bytes())
    if damage in {"output", "index"}:
        path = Path(entry["directory"]) / f"{damage}.jsonl"
        path.write_bytes(path.read_bytes() + b"{}\n")
    elif damage == "identity":
        declaration.write_bytes(declaration.read_bytes() + b"\n# changed declaration\n")
    else:
        entry["state"]["records"] += 1
        marker.write_bytes(canonical_json(entry))
    with pytest.raises(ValueError, match="checkpoint"):
        mixture.materialize_mixture(declaration, output)
    assert not output.exists()


def test_ordered_source_fetch_bounds_reorder_and_propagates_failure(monkeypatch):
    import threading

    from sparselab.corpus import mixture_pipeline

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
    from sparselab.corpus.mixture_pipeline import mixture_writer

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
    progress = json.loads((tmp_path / ".warm.assembly/progress.json").read_bytes())
    assert progress["cache_hits"] == 3
    assert progress.get("read_worker_seconds", 0) == 0


@pytest.mark.parametrize(
    "failure", ["children", "cpu_times", "memory_info", "num_threads"]
)
def test_child_telemetry_races_do_not_abort_materialization(
    tmp_path, monkeypatch, failure
):
    import psutil

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    expected = mixture.materialize_mixture(declaration, tmp_path / "expected")

    def sample(name, value):
        if name == failure:
            raise psutil.NoSuchProcess(123456)
        return value

    child = SimpleNamespace(
        cpu_times=lambda: sample("cpu_times", SimpleNamespace(user=1.0, system=0.5)),
        memory_info=lambda: sample("memory_info", SimpleNamespace(rss=1024)),
        num_threads=lambda: sample("num_threads", 1),
    )
    monkeypatch.setattr(
        psutil.Process, "children", lambda self: sample("children", [child])
    )
    actual = mixture.materialize_mixture(declaration, tmp_path / "raced")
    assert actual == expected
    progress = json.loads((tmp_path / ".raced.assembly/progress.json").read_bytes())
    assert progress["process_observation_errors"] == ["NoSuchProcess"]
    assert not progress["child_sample_complete"]
    assert progress["encoder_utilization"] is None


def test_source_prefetch_continues_while_consumer_is_encoding():
    import threading

    from sparselab.corpus.mixture_pipeline import ordered_fetch

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

    from sparselab.corpus.retained_mixture import RetainedMixtureIndex

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
