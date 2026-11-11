"""Synthetic metadata for corpus engine tests; never acquired or published."""

from pathlib import Path

from klab.corpus.project import Project


def neutral_corpus(root: Path) -> Project:
    """Exercise existing source adapters without a personal scientific recipe."""

    def source(identifier: str, kind: str = "prose") -> dict:
        # Prefixes select existing row-screening behavior, not a real dataset pin.
        return {
            "schema_version": 3,
            "id": identifier,
            "kind": "huggingface_dataset",
            "canonical_uri": "https://huggingface.co/datasets/fixture/neutral",
            "revision": "a" * 40,
            "license": "MIT",
            "license_url": "https://example.org/license",
            "explicit_training_restriction": "none_found",
            "rights": {
                "training_eligibility": "eligible_with_obligations",
                "redistribution_mode": "metadata_reconstruction_only",
                "spdx_expression": "MIT",
                "license_references": ["https://example.org/license"],
                "notices": ["Synthetic fixture; no upstream rights claim."],
            },
            "domains": ["fixture"],
            "document_kinds": [kind],
            "source_family": identifier + "_family",
            "acquisition": {
                "config": "fixture",
                "split": "train",
                "text_field": "text",
                "max_rows": 200,
                "max_bytes": 1048576,
                "bounded_shards": [
                    {
                        "path": "fixture/train.jsonl",
                        "expected_sha256": "b" * 64,
                        "max_shard_bytes": 1048576,
                        "max_scanned_rows": 200,
                        "hash_modulus": 1,
                        "hash_remainders": [0],
                    }
                ],
            },
        }

    sources = [
        source("fineweb_fixture"),
        source("pes2o_fixture_abstract", "abstract"),
        source("pes2o_fixture_paper", "paper"),
        source("wiki_fixture"),
        source("validation_fixture"),
        source("test_fixture"),
    ]
    wiki = sources[3]
    wiki.update(
        kind="wikimedia_dump",
        canonical_uri="https://dumps.wikimedia.org/enwikibooks/20260101/enwikibooks-20260101-pages-articles-multistream.xml.bz2",
        revision="20260101",
    )
    wiki["acquisition"] = {
        "expected_sha1": "c" * 40,
        "checksum_uri": "https://dumps.wikimedia.org/enwikibooks/20260101/enwikibooks-20260101-sha1sums.txt",
        "max_compressed_bytes": 1048576,
        "max_decompressed_bytes": 1048576,
        "max_scanned_pages": 10,
        "max_selected_pages": 10,
        "max_emitted_bytes": 1048576,
    }
    ids = [item["id"] for item in sources]
    return Project.model_validate(
        {
            "root": root,
            "config": {
                "schema_version": 1,
                "id": "neutral-corpus",
                "sources": sorted(
                    "sources/" + identifier + ".yaml" for identifier in ids
                ),
                "transforms": ["transforms/lm.yaml"],
                "splits": "splits.yaml",
                "release": "release.yaml",
            },
            "sources": sources,
            "transforms": [
                {
                    "id": "fixture_lm",
                    "version": "1",
                    "kind": "lm_text",
                    "parameters": {},
                    "inputs": ids,
                }
            ],
            "splits": {
                "schema_version": 1,
                "unit": "source_document_family",
                "family_key": "source_family",
                "assignments": {
                    item["source_family"]: (
                        "validation"
                        if item["id"] == "validation_fixture"
                        else "test"
                        if item["id"] == "test_fixture"
                        else "train"
                    )
                    for item in sources
                },
            },
            "release": {
                "schema_version": 3,
                "publication_mode": "metadata_reconstruction_only",
                "training_use_policy": "allowed_unless_explicitly_prohibited",
                "mixture": {"fixture": 1.0},
                "lm": {"selected": True, "training_splits": ["train", "validation"]},
                "chat": {"selected": False, "training_splits": []},
                "keep_nonredistributable_local": True,
            },
        }
    )
