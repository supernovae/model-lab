from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from klab.canonical import canonical_digest, canonical_json
from klab.data.packing import load_prepared_data, prepare_data
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file


def _resign(manifest: dict[str, object]) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_digest(manifest)


def test_resigned_metadata_tamper_cannot_reuse_cache(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train"]["shape"] = [999]  # type: ignore[index]
    _resign(manifest)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="metadata mismatch"):
        prepare_data(configured, load_tokenizer(configured.tokenizer.path))


def test_resigned_dtype_change_cannot_reuse_cache(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    array_path = prepared.root / "train.npy"
    np.save(array_path, np.load(array_path).astype(np.int64), allow_pickle=False)
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = manifest["train"]
    assert isinstance(train, dict)
    train.update({"dtype": "int64", "sha256": sha256_file(array_path)})
    _resign(manifest)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="metadata mismatch"):
        prepare_data(configured, load_tokenizer(configured.tokenizer.path))


def test_resigned_numeric_metadata_types_are_rejected(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train"]["tokens"] = float(manifest["train"]["tokens"])
    _resign(manifest)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_prepared_data(prepared.root, byte_enabled=False)


def test_unexpected_mask_fails_even_with_unchanged_manifest(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    assert prepared.manifest["supervision"] == {"kind": "all_tokens"}
    assert prepared.train_supervision is None
    np.save(
        prepared.root / "train_supervision.npy",
        np.ones(len(prepared.train), dtype=bool),
    )
    with pytest.raises(ValueError, match="unexpected prepared file inventory"):
        load_prepared_data(prepared.root, byte_enabled=False)


def test_v6_assistant_only_requires_exact_masks(tmp_path: Path) -> None:
    configured = config(tmp_path)
    paths = [
        tmp_path / "assistant-train.jsonl",
        tmp_path / "assistant-validation.jsonl",
    ]
    for split, path in enumerate(paths):
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "format_version": 2,
                        "loss_mode": "assistant_only",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Question {split} {index} about the fox",
                            },
                            {
                                "role": "assistant",
                                "content": f"The fox answered {index} with a book",
                            },
                        ],
                    }
                )
                + "\n"
                for index in range(8)
            ),
            encoding="utf-8",
        )
    dataset = configured.dataset.model_copy(
        update={
            "source": "local_chat",
            "license": "CC0-1.0",
            "train_path": paths[0],
            "validation_path": paths[1],
        }
    )
    local = configured.model_copy(update={"dataset": dataset})
    prepared = prepare_data(local, load_tokenizer(local.tokenizer.path))
    assert prepared.manifest["supervision"]["kind"] == "token-loss-mask-v1"
    assert prepared.train_supervision is not None
    assert not prepared.train_supervision.all()
    assert (prepared.root / "train_supervision.npy").is_file()
    assert (prepared.root / "validation_supervision.npy").is_file()
    (prepared.root / "train_supervision.npy").unlink()
    with pytest.raises(FileNotFoundError):
        load_prepared_data(prepared.root, byte_enabled=False)


@pytest.mark.parametrize("truncated", [False, True])
def test_supervision_never_crosses_roles_or_teaches_truncated_prompt_eos(
    tmp_path, truncated
):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from klab.data.conversations import iter_rendered_conversations
    from klab.data.packing import _collect

    configured = config(tmp_path)
    path = tmp_path / "masked.jsonl"
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "loss_mode": "assistant_only",
                "messages": [
                    {"role": "user", "content": "question " * 20},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        )
        + "\n"
    )
    rendered = next(iter_rendered_conversations(path))
    tokenizer = Tokenizer(
        models.WordLevel({"<unk>": 0, "<eos>": 1, rendered.text: 2}, unk_token="<unk>")
    )
    if truncated:
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    dataset = configured.dataset.model_copy(
        update={
            "source": "local_chat",
            "train_path": path,
            "validation_path": path,
            "train_max_documents": 1,
            "train_max_tokens": 4 if truncated else 100,
        }
    )
    _, mask, _, _ = _collect(dataset, tokenizer, "train")
    if truncated:
        assert not mask.any()
    else:
        # A tokenizer token spanning user, role marker and answer cannot be an
        # assistant-only target. The completed document's EOS remains supervised.
        assert mask.tolist() == [False, True]
