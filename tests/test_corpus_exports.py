"""Frozen release exports exercise real ingestion, fitting and packing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from klab.canonical import canonical_json
from klab.config.loading import load_config, load_tokenizer_config
from klab.corpus.acquisition import acquire
from klab.corpus.export import export_release, verify_release_export
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import freeze
from klab.data.conversations import iter_rendered_conversations
from klab.data.datasets import iter_documents
from klab.data.packing import prepare_data
from klab.data.tokenizer import load_tokenizer, train_tokenizer
from klab.training.manifest import sha256_file


@contextmanager
def _private_corpus_files():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture(scope="module")
def frozen_sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("forge-export")
    with _private_corpus_files():
        project = load_project(Path("tests/fixtures/corpus-sample/corpus.yaml"))
        acquire(project, root)
        build_dir = build(project, root, offline=True)
        assert build(project, root, offline=True) == build_dir
        release_dir = freeze(build_dir, root)
        assert freeze(build_dir, root) == release_dir
        return release_dir, root


@pytest.fixture
def isolated_export(
    frozen_sample: tuple[Path, Path], tmp_path: Path
) -> tuple[Path, Path, Path]:
    """Never mutate the module-scoped frozen release or its original work root."""
    original, _ = frozen_sample
    root = tmp_path / "source-work"
    release = (
        root / "corpora" / original.parent.parent.name / "releases" / original.name
    )
    release.parent.mkdir(parents=True)
    shutil.copytree(original, release)
    shutil.copytree(
        original.parent.parent / "snapshots", release.parent.parent / "snapshots"
    )
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    return release, exported, root


def _sidecar(exported: Path) -> dict[str, Any]:
    return json.loads((exported / "export.json").read_text(encoding="utf-8"))


def _write_sidecar(exported: Path, sidecar: dict[str, Any]) -> None:
    (exported / "export.json").write_bytes(canonical_json(sidecar) + b"\n")


def _dataset(exported: Path):
    return load_config(exported / "run.yaml").dataset


def _reject(exported: Path, **changes: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        verify_release_export(_dataset(exported).model_copy(update=changes))


def test_export_verifies_from_same_and_independent_work_roots(
    isolated_export: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, adjacent, adjacent_root = isolated_export
    independent_root = tmp_path / "independent-work"
    independent = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, independent_root
    )
    assert adjacent != independent
    assert adjacent.name == independent.name
    for exported in (adjacent, independent):
        dataset = _dataset(exported)
        assert dataset.corpus_release_path == release
        assert dataset.corpus_export_path == exported
        assert verify_release_export(dataset)["release_id"] == release.name
        request = _sidecar(exported)
        root = adjacent_root if exported == adjacent else independent_root
        request_sha = hashlib.sha256(
            canonical_json(
                {
                    "release_id": release.name,
                    "view": "lm",
                    "base_config_sha256": request["base_config_sha256"],
                    "vocab_size": 300,
                }
            )
        ).hexdigest()
        assert exported.relative_to(root) == (
            Path("corpora")
            / release.parent.parent.name
            / "exports"
            / release.name
            / "lm"
            / request_sha
        )
    assert (
        _sidecar(adjacent)["base_config_sha256"]
        == _sidecar(independent)["base_config_sha256"]
    )


def test_export_seals_immutable_inputs_before_first_verification(
    frozen_sample: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import export
    from klab.verification_proofs import verification_options

    release, _ = frozen_sample
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    options = verification_options(root)
    checks = []
    verify = export._verify_release_export_cold

    def observe(*args, **kwargs):
        checks.append(1)
        return verify(*args, **kwargs)

    monkeypatch.setattr(export, "_verify_release_export_cold", observe)
    with _private_corpus_files():
        exported = export_release(
            release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root, **options
        )
        dataset = _dataset(exported)
        binding = verify_release_export(dataset, **options)
        for member in ("tokenizer/tokenizer.json", "prepared/shard.bin"):
            path = exported / member
            path.parent.mkdir()
            path.write_bytes(b"derived output")
        assert verify_release_export(dataset, **options) == binding
        assert len(checks) == 0
        assert (
            verify_release_export(dataset, **verification_options(root, cold=True))
            == binding
        )
        assert len(checks) == 1
        config = exported / "tokenizer.yaml"
        original = config.read_bytes()
        edited = original.replace(b"vocab_size: 300", b"vocab_size: 301")
        assert edited != original
        config.write_bytes(edited)
        with pytest.raises(ValueError, match="corpus export config changed"):
            verify_release_export(dataset, **options)
        assert len(checks) == 2


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        # One case per refusal branch of the cold export verifier.
        ("release_id", "0" * 64),
        ("view", "chat"),
        ("base_config_sha256", "not-a-sha"),
        ("vocab_size", 301),
        ("vocab_size", True),
        ("run_config_sha256", "0" * 64),
        ("report_sha256", "0" * 64),
        ("unexpected_metadata", "must not be silently accepted"),
    ],
)
def test_export_rejects_changed_request_or_metadata(
    isolated_export: tuple[Path, Path, Path], field: str, bad: Any
) -> None:
    _, exported, _ = isolated_export
    sidecar = _sidecar(exported)
    sidecar[field] = bad
    _write_sidecar(exported, sidecar)
    _reject(exported)


def test_export_rejects_changed_request_sha_suffix(
    isolated_export: tuple[Path, Path, Path],
) -> None:
    _, exported, _ = isolated_export
    moved = exported.with_name("0" * 64)
    shutil.copytree(exported, moved)
    _reject(exported, corpus_export_path=moved)


def test_export_rejects_wrong_corpus_layout_suffix(
    isolated_export: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, exported, _ = isolated_export
    root = tmp_path / "other-work"
    moved = (
        root / "corpora" / "wrong-corpus" / "exports" / release.name / "lm"
    ) / exported.name
    moved.parent.mkdir(parents=True)
    shutil.copytree(exported, moved)
    _reject(exported, corpus_export_path=moved)


def test_export_rejects_wrong_release_path_and_revision(
    isolated_export: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, exported, _ = isolated_export
    _reject(exported, revision="0" * 64)
    other = tmp_path / ("0" * 64)
    shutil.copytree(release, other)
    _reject(exported, corpus_release_path=other)


def test_export_rejects_changed_generated_config_hash(
    isolated_export: tuple[Path, Path, Path],
) -> None:
    _, exported, _ = isolated_export
    config = exported / "tokenizer.yaml"
    config.write_bytes(config.read_bytes() + b"\n# modified\n")
    _reject(exported)


@pytest.mark.parametrize("filename", ["run.yaml", "tokenizer.yaml"])
def test_export_rejects_rehashed_generated_dataset_mismatch(
    isolated_export: tuple[Path, Path, Path], filename: str
) -> None:
    _, exported, _ = isolated_export
    consumer = _dataset(exported)
    config = exported / filename
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["dataset"]["license"] = "MIT-tampered"
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    sidecar = _sidecar(exported)
    sidecar[
        "run_config_sha256" if filename == "run.yaml" else "tokenizer_config_sha256"
    ] = sha256_file(config)
    _write_sidecar(exported, sidecar)
    with pytest.raises((ValueError, TypeError)):
        verify_release_export(consumer)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("source", "local_chat"),
        ("license", "MIT-tampered"),
        ("train_max_documents", "increment"),
    ],
)
def test_export_rejects_changed_consumer_dataset(
    isolated_export: tuple[Path, Path, Path], field: str, bad: Any
) -> None:
    _, exported, _ = isolated_export
    original = getattr(_dataset(exported), field)
    _reject(exported, **{field: original + 1 if bad == "increment" else bad})


def test_export_requires_exact_source_paths_and_split_bytes(
    isolated_export: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, exported, _ = isolated_export
    split = "train"
    path = release / "lm" / f"{split}.jsonl"
    copied = tmp_path / f"{split}.jsonl"
    shutil.copyfile(path, copied)
    _reject(exported, **{f"{split}_path": copied})
    with path.open("ab") as stream:
        stream.write(b'{"text":"unexpected"}\n')
    _reject(exported)


@pytest.mark.parametrize("name", ["report.json", "manifest.json"])
def test_export_rejects_changed_release_evidence(
    isolated_export: tuple[Path, Path, Path], name: str
) -> None:
    release, exported, _ = isolated_export
    evidence = release / name
    evidence.write_bytes(evidence.read_bytes() + b" ")
    _reject(exported)


def test_export_rejects_inventory_counts_bytes_and_budgets(
    isolated_export: tuple[Path, Path, Path],
) -> None:
    _, exported, _ = isolated_export
    split = "train"
    dataset = _dataset(exported)
    sidecar = _sidecar(exported)
    item = sidecar["splits"][split]
    selected = dataset.train_path if split == "train" else dataset.validation_path
    assert selected is not None
    records = [json.loads(line)["text"] for line in selected.read_text().splitlines()]
    assert item["records"] == len(records)
    assert item["rendered_bytes"] == sum(len(text.encode("utf-8")) for text in records)
    assert getattr(dataset, f"{split}_max_documents") == item["records"]
    assert getattr(dataset, f"{split}_max_tokens") == (
        item["rendered_bytes"] + item["records"] + 1
    )
    for key in ("records", "rendered_bytes"):
        tampered = json.loads(json.dumps(sidecar))
        tampered["splits"][split][key] += 1
        _write_sidecar(exported, tampered)
        _reject(exported)
    _write_sidecar(exported, sidecar)
    _reject(
        exported,
        **{f"{split}_max_tokens": item["rendered_bytes"] + item["records"]},
    )
    _reject(
        exported,
        **{f"{split}_max_tokens": getattr(dataset, f"{split}_max_tokens") + 1},
    )


def test_export_rejects_malformed_split_inventory(
    isolated_export: tuple[Path, Path, Path],
) -> None:
    _, exported, _ = isolated_export
    sidecar = _sidecar(exported)
    sidecar["splits"]["train"]["records"] = True
    _write_sidecar(exported, sidecar)
    _reject(exported)


@pytest.fixture
def resume_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Stop a real export after one range and verify exactly what the retry reads."""
    from klab.corpus import export, shards
    from klab.corpus.release_verify import verification_settings

    root = tmp_path / "work"
    base = Path("configs/runtime_smoke_cpu.yaml")
    monkeypatch.setattr(shards, "CHUNK_BYTES", 256)

    def resume(release, view, stage, worker, inputs):
        expected_ranges = [
            (str(path), start, end)
            for path in inputs
            for start, end in shards.chunk_ranges(path)
        ]
        assert len(expected_ranges) > 1
        observed = []
        scan = getattr(export, worker)

        def record_range(path, start, end):
            observed.append((path, start, end))
            return scan(path, start, end)

        with monkeypatch.context() as probe:
            probe.setattr(export, worker, record_range)
            with verification_settings(
                workers=1, scratch_dir=tmp_path / "export-scratch"
            ):
                with monkeypatch.context() as fault:
                    fault.setenv("KLAB_FAULT", f"{stage}:1")
                    with pytest.raises(RuntimeError, match="KLAB_FAULT"):
                        export_release(release, view, base, 300, root)
                assert observed == expected_ranges[:1]
                assert not list(root.rglob("export.json"))
                observed.clear()
                exported = export_release(release, view, base, 300, root)
                assert observed == expected_ranges[1:]
        assert exported == export_release(release, view, base, 300, root)
        return exported

    return resume


def test_lm_export_fits_every_train_document(
    frozen_sample: tuple[Path, Path], resume_export
) -> None:
    from klab.corpus.export import LICENSE_SCAN_STAGE

    release, _ = frozen_sample
    exported = resume_export(
        release,
        "lm",
        LICENSE_SCAN_STAGE,
        "_license_range",
        [release / "documents.jsonl"],
    )
    tokenizer_config = load_tokenizer_config(exported / "tokenizer.yaml")
    run_config = load_config(exported / "run.yaml")
    dataset = run_config.dataset
    assert dataset.source == "local_text"
    binding = verify_release_export(dataset)
    assert binding["release_id"] == release.name
    assert list(iter_documents(dataset, "train")) == [
        json.loads(row)["text"]
        for row in (release / "lm/train.jsonl").read_text().splitlines()
    ]
    artifact = train_tokenizer(tokenizer_config)
    manifest = json.loads((artifact.parent / "tokenizer_manifest.json").read_text())
    assert manifest["corpus_export"] == binding
    assert manifest["selected_documents"] == binding_split_count(exported, "train")
    prepared = prepare_data(run_config, load_tokenizer(artifact))
    assert prepared.manifest["corpus_export"] == binding
    assert prepared.manifest["train"]["retained_documents"] == binding_split_count(
        exported, "train"
    )
    assert prepared.manifest["train"]["truncated_documents"] == 0
    assert prepare_data(run_config, load_tokenizer(artifact)).root == prepared.root


def test_changed_base_config_uses_distinct_export(
    frozen_sample: tuple[Path, Path], tmp_path: Path
) -> None:
    release, root = frozen_sample
    original = Path("configs/runtime_smoke_cpu.yaml")
    changed = tmp_path / "base.yaml"
    changed.write_text(
        original.read_text(encoding="utf-8").replace("seed: 7", "seed: 11"),
        encoding="utf-8",
    )
    first = export_release(release, "lm", original, 300, root)
    second = export_release(release, "lm", changed, 300, root)
    assert first != second
    assert (
        verify_release_export(load_config(second / "run.yaml").dataset)["release_id"]
        == release.name
    )


def binding_split_count(exported: Path, split: str) -> int:
    return json.loads((exported / "export.json").read_text())["splits"][split][
        "records"
    ]


def test_chat_export_preserves_v2_records(
    frozen_sample: tuple[Path, Path], resume_export
) -> None:
    from klab.corpus.export import SPLIT_STATS_STAGE

    release, _ = frozen_sample
    exported = resume_export(
        release,
        "chat",
        SPLIT_STATS_STAGE,
        "_chat_range_stats",
        [release / "chat" / f"{split}.jsonl" for split in ("train", "validation")],
    )
    dataset = load_config(exported / "run.yaml").dataset
    assert dataset.source == "local_chat"
    assert verify_release_export(dataset)["view"] == "chat"
    stats = _sidecar(exported)["splits"]
    for split in ("train", "validation"):
        path = dataset.train_path if split == "train" else dataset.validation_path
        assert path is not None
        texts = [row.text for row in iter_rendered_conversations(path)]
        assert list(iter_documents(dataset, split)) == texts
        item = stats[split]
        assert item["records"] == len(texts)
        assert item["rendered_bytes"] == sum(
            len(text.encode("utf-8")) for text in texts
        )
        assert all(
            set(json.loads(row)) == {"format_version", "loss_mode", "messages"}
            for row in path.read_text().splitlines()
        )


def test_local_text_rejects_malformed_rows(tmp_path: Path) -> None:
    config = load_config(Path("configs/runtime_smoke_cpu.yaml")).dataset.model_copy(
        update={
            "source": "local_text",
            "train_path": tmp_path / "train.jsonl",
            "validation_path": tmp_path / "validation.jsonl",
            "license": "MIT",
        }
    )
    for bad in (
        '{"text": ""}\n',
        '{"text": "x", "extra": 1}\n',
        '{"text": 2}\n',
        "not-json\n",
    ):
        config.train_path.write_text(bad)
        with pytest.raises(ValueError, match="local_text"):
            list(iter_documents(config, "train"))


def test_cli_export_authenticates_release_once(
    frozen_sample: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from klab.cli.main import build_parser
    from klab.corpus import release as publication
    from klab.corpus.cli import _handle

    release, _ = frozen_sample
    real_files = publication._files
    scans = 0

    def observe_files(root: Path, inventory: dict, *rest, **options) -> None:
        nonlocal scans
        if root == release:
            scans += 1
        real_files(root, inventory, *rest, **options)

    monkeypatch.setattr(publication, "_files", observe_files)
    _handle(
        build_parser().parse_args(
            [
                "--work-dir",
                str(tmp_path / "work"),
                "corpus",
                "export",
                str(release),
                "--view",
                "lm",
                "--base-run-config",
                "configs/runtime_smoke_cpu.yaml",
                "--vocab-size",
                "300",
            ]
        )
    )
    exported = Path(capsys.readouterr().out.strip())
    run = load_config(exported / "run.yaml")
    assert run.dataset.corpus_release_path == release
    assert (
        json.loads((exported / "export.json").read_bytes())["release_id"]
        == release.name
    )
    assert scans == 1


def test_sharded_split_statistics_match_a_single_pass_and_name_the_bad_line(
    tmp_path, monkeypatch
) -> None:
    from klab.corpus import export, shards
    from klab.corpus.release_verify import verification_settings

    rows = [json.dumps({"text": f"line {n} é"}).encode() for n in range(200)]
    path = tmp_path / "train.jsonl"
    path.write_bytes(b"\n".join(rows) + b"\n")
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setattr(shards, "CHUNK_BYTES", 500)
    with verification_settings(workers=3):
        assert export._split_stats(path, "lm") == (
            200,
            sum(len(f"line {n} é".encode()) for n in range(200)),
        )
        rows[137] = b'{"text": " "}'
        path.write_bytes(b"\n".join(rows) + b"\n")
        with pytest.raises(ValueError, match=r"invalid LM record at .*:138"):
            export._split_stats(path, "lm")

        chat = tmp_path / "chat.jsonl"
        good = canonical_json(
            {
                "messages": [
                    {"role": "user", "content": "¿Qué?"},
                    {"role": "assistant", "content": "Éste."},
                ]
            }
        )
        chat.write_bytes((good + b"\n") * 35 + b'{"messages":[],"messages":[]}\n')
        with pytest.raises(ValueError, match=r"chat\.jsonl:36: .*duplicate JSON key"):
            export._split_stats(chat, "chat")
