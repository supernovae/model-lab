"""Small real-file checks for relocated tokenizer evidence."""

import hashlib

import pytest

from klab.corpus.tokenizer_observation import _read, _relocation_mapper


def test_tokenizer_relocation_maps_both_roots_and_detects_changed_bytes(tmp_path):
    source_work = tmp_path / "source" / "work"
    source_receipts = tmp_path / "source" / "receipts"
    target_work = tmp_path / "target" / "work"
    target_receipts = tmp_path / "target" / "receipts"
    target_work.mkdir(parents=True)
    target_receipts.mkdir(parents=True)
    token = target_work / "tokenizer.json"
    report = target_receipts / "report.json"
    token.write_bytes(b"tokenizer")
    report.write_bytes(b"completed")
    map_path, map_values = _relocation_mapper(
        source_work, target_work, source_receipts, target_receipts
    )
    for old, new in (
        (source_work / "tokenizer.json", token),
        (source_receipts / "report.json", report),
    ):
        assert map_path(old) == new
        digest = hashlib.sha256(new.read_bytes()).hexdigest()
        assert _read(map_path(old), digest) == new.read_bytes()
    assert map_values(
        {"dataset": {"train_path": str(source_work / "train")}, "vocab_size": 32768}
    ) == {"dataset": {"train_path": str(target_work / "train")}, "vocab_size": 32768}
    with pytest.raises(ValueError, match="outside relocation roots"):
        map_path(tmp_path / "other" / "report.json")
    with pytest.raises(ValueError, match="invalid tokenizer relocation path"):
        map_path(source_work / ".." / "outside")
    report.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="input changed"):
        _read(
            map_path(source_receipts / "report.json"),
            hashlib.sha256(b"completed").hexdigest(),
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing_train",
        "missing_rendered_bytes",
        "null_splits",
        "list_splits",
        "null_train",
        "string_rendered_bytes",
        "wrong_digest",
        "wrong_token_limit",
    ],
)
def test_observed_tokenizer_refuses_malformed_split_metadata(tmp_path, case):
    from tokenizer_observation_fixtures import observed_tokenizer

    from klab.corpus.export import _export_binding
    from klab.corpus.tokenizer_observation import verify_observed_tokenizer

    observation, path, config, release, manifest, export, token_manifest, write = (
        observed_tokenizer(tmp_path)
    )
    assert (
        verify_observed_tokenizer(observation, path, config, release, manifest)[
            "method"
        ]
        == "post_run_observation"
    )
    if case == "missing_train":
        export["splits"].pop("train")
    elif case == "missing_rendered_bytes":
        export["splits"]["train"].pop("rendered_bytes")
    elif case == "null_splits":
        export["splits"] = None
    elif case == "list_splits":
        export["splits"] = []
    elif case == "null_train":
        export["splits"]["train"] = None
    elif case == "string_rendered_bytes":
        export["splits"]["train"]["rendered_bytes"] = "8"
    elif case == "wrong_digest":
        export["splits"]["train"]["sha256"] = "f" * 64
    else:
        export["splits"]["train"]["rendered_bytes"] = 9
    # Reseal the declared observation, so refusal concerns metadata rather than
    # the earlier observed-file hash. Preserve coherent downstream bindings when
    # the malformed shape still permits their projection.
    write("export", config.dataset.corpus_export_path / "export.json", export)
    if case in {
        "missing_rendered_bytes",
        "string_rendered_bytes",
        "wrong_digest",
        "wrong_token_limit",
    }:
        binding = _export_binding(export)
        token_manifest["corpus_export"] = binding
        token_manifest["training_contract"]["corpus_export"] = binding
        write(
            "tokenizer_manifest",
            config.output_dir / "tokenizer_manifest.json",
            token_manifest,
        )
    with pytest.raises(ValueError, match="^tokenizer observation split differs$"):
        verify_observed_tokenizer(observation, path, config, release, manifest)
