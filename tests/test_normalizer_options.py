"""Release normalizer options reproduce the retired v1, v2 and v3 behaviour.

`fixtures/normalizer/golden.json` holds the output of the version-string
normalizers (`normalizer-nfc-markdown-v1`, `-structure-v2`, `-structure-v3`)
captured before they were removed, for a spread of inputs and the source IDs
those versions special-cased. The options below must reproduce every output
byte for byte, apart from the normalizer name and the document identities
derived from it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from corpus_fixtures import STRICT_POLICY, fixture_rights

from klab.corpus.document_normalization import records_for_file
from klab.corpus.project import (
    DocumentPolicy,
    NormalizerOptions,
    SourceDeclaration,
    load_project,
)
from klab.corpus.structure_cleaning import clean_structure

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/normalizer/golden.json").read_text()
)
OLD_NAMES = {
    "normalizer-nfc-markdown-v1",
    "normalizer-structure-v2",
    "normalizer-structure-v3",
}
DOCS = DocumentPolicy.model_validate(STRICT_POLICY["documents"])
PAGERDUTY_KEYS = ["cover", "hero", "hero_alt_text", "pdf", "style"]
OPTIONS = {
    "v1": {
        "structure_cleaning": False,
        "front_matter_extra_keys": {},
        "gutenberg_boilerplate": [],
    },
    "v2": {
        "structure_cleaning": True,
        "front_matter_extra_keys": {},
        "gutenberg_boilerplate": ["project_gutenberg_en"],
    },
    "v3": {
        "structure_cleaning": True,
        "front_matter_extra_keys": {"kml_scale_pagerduty": PAGERDUTY_KEYS},
        "gutenberg_boilerplate": [
            "kml_scale_project_gutenberg",
            "project_gutenberg_en",
        ],
    },
}


def _masked(value: Any, name: str) -> Any:
    """Blank the normalizer name and the identities derived from it."""
    if isinstance(value, dict):
        return {
            key: "<ID>" if key in {"document_id", "record_id"} else _masked(item, name)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_masked(item, name) for item in value]
    if isinstance(value, str) and value in OLD_NAMES | {name}:
        return "<N>"
    return value


def _source(source_id: str) -> SourceDeclaration:
    return SourceDeclaration.model_validate(
        {
            "id": source_id,
            "kind": "local",
            "canonical_uri": f"fixture:{source_id}",
            "revision": "v1",
            "license": "MIT",
            "rights": fixture_rights(),
            "domains": ["technical_docs"],
            "document_kinds": ["prose"],
            "source_family": source_id,
            "acquisition": {
                "files": [{"path": "a.txt", "name": "a.txt"}],
                "max_bytes": 100000,
            },
        }
    )


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_structure_cleaning_options_reproduce_the_retired_versions(
    version: str,
) -> None:
    options = NormalizerOptions.model_validate(OPTIONS[version])
    checked = 0
    for key, expected in GOLDEN["clean"].items():
        label, source_id, case = key.split("/")
        if label != version:
            continue
        actual = clean_structure(
            GOLDEN["cases"][case], source_id=source_id, options=options
        )
        assert json.dumps(_masked(actual, options.identity), sort_keys=True) == (
            json.dumps(_masked(expected, options.identity), sort_keys=True)
        ), key
        assert actual["structure"]["normalizer"] == options.identity
        checked += 1
    assert checked == len(GOLDEN["sources"]) * len(GOLDEN["cases"])


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_options_reproduce_document_and_evidence_rows(version: str) -> None:
    options = NormalizerOptions.model_validate(OPTIONS[version])
    checked = 0
    for key, expected in GOLDEN["records"].items():
        label, source_id, case = key.split("/")
        if label != version:
            continue
        try:
            rows: Any = [
                list(row)
                for row in records_for_file(
                    GOLDEN["cases"][case].encode(),
                    "a.txt",
                    _source(source_id),
                    "a" * 64,
                    normalizer_options=options,
                    document_policy=DOCS,
                )
            ]
        except ValueError as error:
            rows = {"error": str(error)}
        assert json.dumps(_masked(rows, options.identity), sort_keys=True) == (
            json.dumps(_masked(expected, options.identity), sort_keys=True)
        ), key
        checked += 1
    assert checked == len(GOLDEN["sources"]) * len(GOLDEN["cases"])


def test_the_options_change_the_golden_output_where_they_differ() -> None:
    """The comparison is not vacuous: each option moves some output."""
    differing = {
        key
        for key, row in GOLDEN["clean"].items()
        if key.startswith("v3/")
        and json.dumps(_masked(row, "x"), sort_keys=True)
        != json.dumps(_masked(GOLDEN["clean"]["v2/" + key[3:]], "x"), sort_keys=True)
    }
    assert "v3/kml_scale_pagerduty/front_matter_pagerduty" in differing
    assert "v3/kml_scale_project_gutenberg/gutenberg_marked" in differing


def test_identity_names_the_options_and_document_ids_follow_it() -> None:
    names = {
        version: NormalizerOptions.model_validate(raw).identity
        for version, raw in OPTIONS.items()
    }
    assert len(set(names.values())) == 3
    assert all(name.startswith("normalizer-") for name in names.values())
    reordered = NormalizerOptions.model_validate(
        {**OPTIONS["v3"], "gutenberg_boilerplate": ["project_gutenberg_en"]}
    )
    assert reordered.identity not in names.values()

    def document_id(options: NormalizerOptions) -> str:
        [(document, _)] = records_for_file(
            b"Plain prose.\n",
            "a.txt",
            _source("plain_source"),
            "a" * 64,
            normalizer_options=options,
            document_policy=DOCS,
        )
        return document["document_id"]

    assert document_id(NormalizerOptions.model_validate(OPTIONS["v1"])) != document_id(
        NormalizerOptions.model_validate(OPTIONS["v2"])
    )


@pytest.mark.parametrize(
    "options",
    [
        {**OPTIONS["v1"], "gutenberg_boilerplate": ["project_gutenberg_en"]},
        {**OPTIONS["v1"], "front_matter_extra_keys": {"docs": ["cover"]}},
        {**OPTIONS["v3"], "front_matter_extra_keys": {"docs": ["Cover"]}},
        {**OPTIONS["v3"], "front_matter_extra_keys": {"docs": []}},
        {**OPTIONS["v3"], "front_matter_extra_keys": {"docs": ["cover", "cover"]}},
        {**OPTIONS["v3"], "gutenberg_boilerplate": ["a", "a"]},
        {"structure_cleaning": True, "gutenberg_boilerplate": []},
        {**OPTIONS["v3"], "unknown": 1},
    ],
)
def test_invalid_options_are_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        NormalizerOptions.model_validate(options)


def test_release_requires_explicit_options_naming_declared_sources(
    tmp_path: Path,
) -> None:
    sample = Path(__file__).parent / "fixtures/corpus-sample"
    project = tmp_path / "project"
    shutil.copytree(sample, project)
    release = project / "release.yaml"
    data = yaml.safe_load(release.read_text())
    data["normalizer"] = {
        **OPTIONS["v2"],
        "gutenberg_boilerplate": ["not_a_declared_source"],
    }
    release.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unknown sources"):
        load_project(project / "corpus.yaml")
    del data["normalizer"]
    release.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="normalizer"):
        load_project(project / "corpus.yaml")


def test_cleaning_code_names_no_source() -> None:
    text = (
        Path(__file__).parents[1] / "src/klab/corpus/structure_cleaning.py"
    ).read_text()
    for forbidden in ("kml_scale", "pagerduty", 'startswith("project'):
        assert forbidden not in text.lower().replace("'", '"')
