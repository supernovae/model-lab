"""Synthetic metadata for corpus engine tests; never acquired or published."""

import io
import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.project import NormalizerOptions, Project
from klab.training.manifest import sha256_file

# The explicit release policies tests build on. STRICT is the cautious profile:
# any cross-split overlap fails and a reviewed admission may be recorded.
PLAIN_NORMALIZER = {
    "structure_cleaning": False,
    "front_matter_extra_keys": {},
    "gutenberg_boilerplate": [],
}
STRUCTURED_NORMALIZER = {
    "structure_cleaning": True,
    "front_matter_extra_keys": {},
    "gutenberg_boilerplate": [],
}

PLAIN_OPTIONS = NormalizerOptions.model_validate(PLAIN_NORMALIZER)

STRICT_POLICY = {
    "duplicates": {
        "heldout_overlap": "error",
        "keys": ["raw", "normalized"],
        "resolution": "chain_smallest_id",
        "rights_conflict": "error",
    },
    "documents": {
        "empty_text": "error",
        "exclusions": "none",
        "markdown_kind": "first_declared",
        "row_files": "buffered",
    },
    "rights": {"training_use": "explicit_eligibility_required"},
    "build": {"strategy": "auto"},
    "mixture": {"total_tokens": 1, "max_exposures": 2},
}
# PRIVATE_RESEARCH drops train copies of held-out text and screens private data.
PRIVATE_RESEARCH_POLICY = {
    "duplicates": {
        "heldout_overlap": "drop_train_copy",
        "keys": ["raw", "normalized", "canonical_origin"],
        "resolution": "component_ranked",
        "rights_conflict": "allow",
    },
    "documents": {
        "empty_text": "skip",
        "exclusions": "research_privacy",
        "markdown_kind": "prose_when_declared",
        "row_files": "streaming",
    },
    "rights": {"training_use": "allowed_unless_explicitly_prohibited"},
    "build": {"strategy": "auto"},
    "mixture": {"total_tokens": 1, "max_exposures": 2},
}


def fixture_rights(reference: str = "https://example.org/license") -> dict:
    """Rights evidence for a synthetic, eligible source."""
    return {
        "training_eligibility": "eligible",
        "redistribution_mode": "redistributable_under_source_terms",
        "spdx_expression": "MIT",
        "license_references": [reference],
    }


def neutral_corpus(root: Path) -> Project:
    """Exercise existing source adapters without a personal scientific recipe."""

    def source(identifier: str, kind: str = "prose") -> dict:
        # Prefixes select existing row-screening behavior, not a real dataset pin.
        return {
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
                "publication_mode": "metadata_reconstruction_only",
                "policy": PRIVATE_RESEARCH_POLICY,
                "mixture": {"fixture": 1.0},
                "lm": {"selected": True, "training_splits": ["train", "validation"]},
                "chat": {"selected": False, "training_splits": []},
                "keep_nonredistributable_local": True,
                "normalizer": PLAIN_NORMALIZER,
            },
        }
    )


def write_prose_project(
    root: Path, texts: dict[str, list[str]], *, corpus_id: str = "prose-fixture"
) -> Path:
    """A language-model-only project: one local source per `prose_<split>` family.

    Each listed text is one file and so one document. Returns the project directory.
    """
    project = root / "project"
    (project / "sources").mkdir(parents=True)
    (project / "transforms").mkdir()
    for family, documents in texts.items():
        files = []
        for index, text in enumerate(documents):
            name = f"{family}-{index}.txt"
            (project / "sources" / name).write_text(text, encoding="utf-8")
            files.append({"path": f"sources/{name}", "name": name})
        source = {
            "id": family,
            "kind": "local",
            "origin": "human_authored",
            "canonical_uri": f"klab://fixture/{family}",
            "revision": "v1",
            "license": "MIT",
            "rights": {**fixture_rights(), "notices": ["Synthetic fixture."]},
            "domains": ["prose"],
            "document_kinds": ["prose"],
            "source_family": family,
            "acquisition": {"files": files, "max_bytes": 1 << 24},
        }
        (project / "sources" / f"{family}.yaml").write_text(yaml.safe_dump(source))
    documents = {
        "transforms/lm.yaml": {
            "id": "lm",
            "version": "1",
            "kind": "lm_text",
            "parameters": {},
            "inputs": list(texts),
        },
        "splits.yaml": {
            "schema_version": 1,
            "unit": "source_document_family",
            "family_key": "source_family",
            "assignments": {family: family.split("_")[1] for family in texts},
        },
        "release.yaml": {
            "policy": STRICT_POLICY,
            "mixture": {"prose": 1.0},
            "lm": {"selected": True, "training_splits": ["train", "validation"]},
            "chat": {"selected": False, "training_splits": []},
            "publication_mode": "redistributable_under_source_terms",
            "normalizer": PLAIN_NORMALIZER,
        },
        "corpus.yaml": {
            "schema_version": 1,
            "id": corpus_id,
            "sources": sorted(f"sources/{family}.yaml" for family in texts),
            "transforms": ["transforms/lm.yaml"],
            "splits": "splits.yaml",
            "release": "release.yaml",
        },
    }
    for name, value in documents.items():
        (project / name).write_text(yaml.safe_dump(value))
    return project


def forge_release(release: Path, edit: Callable[[Path], None]) -> Path:
    """Copy a release, edit its rows, and re-seal digests and identity consistently.

    A forger who can rewrite every digest still must not get past the row checks.
    """
    scratch = release.parent / "forged"
    scratch.mkdir(exist_ok=True)
    staged = scratch / "staging"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(release, staged)
    edit(staged)
    manifest = json.loads((staged / "manifest.json").read_text())
    for name in manifest["files"]:
        path = staged / name
        manifest["files"][name] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    manifest.pop("release_id")
    manifest["release_id"] = canonical_digest(manifest)
    (staged / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    target = release.parent / manifest["release_id"]
    shutil.rmtree(target, ignore_errors=True)
    staged.rename(target)
    return target


def edit_declared_source(
    edit: Callable[[dict], None],
    *,
    identity: bool = False,
    source_id: str | None = None,
) -> Callable[[Path], None]:
    """Edit a declared source in sources.json and the license report together.

    The first source is edited unless `source_id` names one. With `identity`,
    also edit the build identity the release carries.
    """

    def apply(root: Path) -> None:
        rows = json.loads((root / "sources.json").read_text())
        row = next(item for item in rows if source_id in (None, item["id"]))
        edit(row)
        (root / "sources.json").write_bytes(canonical_json(rows) + b"\n")
        report = json.loads((root / "license-report.json").read_text())
        report["sources"] = rows
        (root / "license-report.json").write_bytes(canonical_json(report) + b"\n")
        if identity:
            manifest = json.loads((root / "manifest.json").read_text())
            declared = manifest["build_identity"]["sources"][row["id"]]
            declared["license"] = row["license"]
            declared["license_url"] = row["license_url"]
            declared["rights"] = row["rights_policy"]
            (root / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")

    return apply


def _write_acquisition_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def write_acquisition_recipe(
    tmp_path: Path,
    *,
    kind: str = "local",
    acquisition: dict | None = None,
    revision: str = "v1",
    uri: str = "fixture://source",
    license: str = "MIT",
    rights: dict | None = None,
) -> Path:
    root = tmp_path / "recipe"
    root.mkdir(exist_ok=True)
    _write_acquisition_yaml(
        root / "corpus.yaml",
        {
            "schema_version": 1,
            "id": "example",
            "sources": ["sources/one.yaml"],
            "transforms": [],
            "splits": "splits.yaml",
            "release": "release.yaml",
        },
    )
    _write_acquisition_yaml(
        root / "splits.yaml",
        {"schema_version": 1, "unit": "document", "assignments": {"document": "train"}},
    )
    release = {
        "mixture": {"technical_docs": 1.0},
        "lm": {"selected": True, "training_splits": ["train", "validation"]},
        "chat": {"selected": False, "training_splits": []},
        "publication_mode": "metadata_reconstruction_only",
        "normalizer": PLAIN_NORMALIZER,
        "policy": STRICT_POLICY,
    }
    rights = rights or {**fixture_rights(), "spdx_expression": license}
    source = {
        "id": "one",
        "kind": kind,
        "canonical_uri": uri,
        "revision": revision,
        "license": license,
        "license_url": rights["license_references"][0],
        "rights": rights,
        "domains": ["technical_docs"],
        "document_kinds": ["markdown"],
        "source_family": "docs",
        "acquisition": acquisition
        or {
            "files": [{"path": "sources/data.md", "name": "data.md"}],
            "max_bytes": 1024,
        },
    }
    _write_acquisition_yaml(root / "release.yaml", release)
    _write_acquisition_yaml(root / "sources/one.yaml", source)
    return root / "corpus.yaml"


def serve_pinned_acquisition(
    monkeypatch: pytest.MonkeyPatch, bodies: dict[str, bytes]
) -> list[str]:
    """Answer pinned HTTPS downloads from memory; record every requested URL."""
    calls: list[str] = []

    class Response(io.BytesIO):
        status = 200

        def __init__(self, body: bytes, url: str):
            super().__init__(body)
            self.url = url
            self.headers = {"Content-Length": str(len(body))}

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float) -> Response:
            calls.append(request.full_url)
            return Response(bodies[request.full_url], request.full_url)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    return calls
