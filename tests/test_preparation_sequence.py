"""Tiny, offline public-path preparation sequence; no tokenizer or model work."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config, load_tokenizer_config
from klab.corpus.acquisition import (
    _adapter,
    _project_sha,
    declaration_sha256,
    verify_acquisition,
    verify_snapshot,
)
from klab.corpus.admission_draft import draft_admission_manifest
from klab.corpus.declaration_render import render_declaration
from klab.corpus.export import (
    _licenses,
    _split_stats,
    export_release,
    verify_release_export,
)
from klab.corpus.pipeline import build
from klab.corpus.project import load_project, source_declaration_payload
from klab.corpus.protected_lineage import audit_protected_lineage
from klab.corpus.release import freeze, verify_release
from klab.corpus.release_review import verify_admission_review
from klab.corpus.split_freeze import finalize_family_inventory, freeze_splits
from klab.corpus.split_inventory import write_split_inventory
from klab.training.attempt_budget import AttemptBudget, AttemptContract
from klab.training.attempt_commands import (
    classify_attempt_command,
    phase_output_paths,
)
from klab.training.manifest import sha256_file
from klab.training.preparation import compile_phase

_PACKET = Path(__file__).resolve().parent / "fixtures/preparation"
_PLAN = _PACKET / "preparation.json"
_PHASES = json.loads(_PLAN.read_text())["phases"]
_PHASE_MAP = {
    label: item["leaf"] for label, item in _PHASES.items() if item.get("leaf")
}


# Each of the 32 native phases has its own 45-second subprocess bound. The
# shared attempt also includes refusal probes and binding commands. Its total
# covers all command bounds plus 120 seconds for local declaration/receipt work
# between commands; it is not an unrelated ten-minute speed test.
_PHASE_TIMEOUT_SECONDS = 45
_REFUSAL_PROBES = 4
_BIND_CALLS = 3
_BIND_TIMEOUT_SECONDS = 20
_LOCAL_METADATA_ALLOWANCE_SECONDS = 120
_ATTEMPT_WALL_SECONDS = (
    (len(_PHASES) + _REFUSAL_PROBES) * _PHASE_TIMEOUT_SECONDS
    + _BIND_CALLS * _BIND_TIMEOUT_SECONDS
    + _LOCAL_METADATA_ALLOWANCE_SECONDS
)


def _fixture_template(recipe: Path, filename: str, **substitutions: object) -> Path:
    """Load the native fixture template; substitute only explicitly named fixture data.

    Paths in projects are deliberately never substituted: a broken fixture
    source/split/transform/release literal must fail this connected sequence.
    Synthetic rights/review decisions here apply only to generated fixture rows.
    """
    source = _PACKET / filename
    payload = yaml.safe_load(source.read_text())
    assert set(substitutions) <= set(payload)
    payload.update(substitutions)
    destination = recipe / filename
    if source.suffix == ".json":
        destination.write_bytes(canonical_json(payload) + b"\n")
    else:
        _write_yaml(destination, payload)
    return destination


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_render_declaration_substitutes_exactly_and_refuses_unsafe_outputs(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.json"
    template.write_text('{"identity":"${RELEASE_SHA256}"}\n')
    output = tmp_path / "rendered.json"
    rendered = render_declaration(
        template, json.dumps({"RELEASE_SHA256": "a" * 64}), output, tmp_path
    )
    assert rendered["sha256"] == sha256_file(output)
    assert json.loads(output.read_text()) == {"identity": "a" * 64}
    array_template = tmp_path / "array-template.json"
    array_template.write_text('{"reviewed":"${REVIEWED_JSON}"}\n')
    array_output = tmp_path / "array-rendered.json"
    render_declaration(
        array_template,
        json.dumps({"REVIEWED_JSON": '[{"source_id":"covered"}]'}),
        array_output,
        tmp_path,
    )
    assert json.loads(array_output.read_text()) == {
        "reviewed": [{"source_id": "covered"}]
    }
    with pytest.raises(TypeError, match="must be an array"):
        render_declaration(
            array_template,
            json.dumps({"REVIEWED_JSON": "{}"}),
            tmp_path / "not-an-array.json",
            tmp_path,
        )
    with pytest.raises(ValueError, match="exactly cover"):
        render_declaration(template, "{}", tmp_path / "missing.json", tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        render_declaration(template, "{}", tmp_path.parent / "outside.json", tmp_path)
    symlink_parent = tmp_path / "alias"
    symlink_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        render_declaration(
            template,
            json.dumps({"RELEASE_SHA256": "a" * 64}),
            symlink_parent / "via-alias.json",
            tmp_path,
        )
    assert (
        classify_attempt_command(
            [
                "klab",
                "corpus",
                "render-declaration",
                "--template",
                str(template),
                "--values-json",
                "{}",
                "--output",
                str(tmp_path / "declaration.json"),
            ]
        ).effect
        == "preparation"
    )


def test_unset_optional_contract_fields_keep_the_contract_identity() -> None:
    """Spent ledgers bind the contract digest; unset options must not change it."""
    minimal = {
        "contract_version": 1,
        "max_optimizer_updates": 0,
        "max_actual_target_positions": 0,
        "max_generation_calls": 0,
        "max_generated_tokens": 0,
        "max_wall_seconds": 30,
        "content_identity_sha256": "a" * 64,
        "monitor_policy_sha256": "b" * 64,
        "workspace_baseline_sha256": "c" * 64,
    }
    payload = AttemptContract.model_validate(minimal).model_dump(mode="json")
    assert payload == {**minimal, "max_wall_seconds": 30.0}
    with pytest.raises(ValueError, match="inspected admission requires"):
        AttemptContract.model_validate(
            {**payload, "require_admission_inspection_binding": True}
        )


def test_reviewed_admission_cannot_substitute_an_edited_draft(tmp_path: Path) -> None:
    draft = tmp_path / "draft.json"
    draft.write_text('{"decision":"quarantine"}\n', encoding="utf-8")
    admitted = tmp_path / "admission.json"
    admitted.write_text('{"decision":"admit"}\n', encoding="utf-8")
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}\n")
    (tmp_path / "admission.json.review.json").write_bytes(
        canonical_json(
            {
                "format": "klab-admission-review-v2",
                "decision": "ACCEPTED",
                "reviewer": "Fixture reviewer",
                "reviewed_on": "2026-10-09",
                "draft_path": str(draft),
                "draft_sha256": sha256_file(draft),
                "admission_sha256": sha256_file(admitted),
                "inspection_path": str(inspection),
                "inspection_sha256": sha256_file(inspection),
                "item_decisions": [],
            }
        )
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_admission_review(admitted, tmp_path)


def _sources(root: Path, project_id: str) -> tuple[Path, dict[str, str]]:
    recipe = root / project_id
    strata = {
        "books": "general_prose",
        "books_rows": "general_prose",
        "incident": "incident_response_docs",
        "wiki": "explanatory_prose",
    }
    files_by_source: dict[str, list[tuple[str, bytes]]] = {}
    retrieval_by_source: dict[str, dict] = {}
    for source_id, stratum in strata.items():
        if source_id == "books_rows":
            shard = "fixture-books.json.gz"
            sample = shard + ".sample.jsonl"
            selected = []
            rendered = []
            for index, license_label in enumerate(("Public Domain", "Copyrighted")):
                record = {
                    "id": str(100 + index),
                    "metadata": {
                        "license": license_label,
                        "url": f"https://www.gutenberg.org/ebooks/{100 + index}",
                        "title": f"Fixture book {index}",
                        "provenance": f"{shard}:{index + 1}",
                        "language": "en",
                    },
                    "text": (
                        f"Fixture book {index} explains a clear sequence of events. "
                        "Readers can follow the account and understand the result. "
                    )
                    * 3,
                }
                digest = canonical_digest(record)
                selected.append(
                    {"source_row_index": index, "source_row_sha256": digest}
                )
                rendered.append(
                    {
                        **record,
                        "_klab_source": {
                            "id": record["id"],
                            "dataset_revision": "a" * 40,
                            "source_shard_path": shard,
                            "source_shard_sha256": "b" * 64,
                            "source_row_index": index,
                            "source_row_sha256": digest,
                        },
                    }
                )
            files_by_source[source_id] = [
                (sample, b"".join(canonical_json(row) + b"\n" for row in rendered))
            ]
            retrieval_by_source[source_id] = {
                "shards": [
                    {
                        "output_path": sample,
                        "source_shard_path": shard,
                        "source_shard_sha256": "b" * 64,
                        "selected_rows": selected,
                    }
                ]
            }
            _write_yaml(
                recipe / "sources" / f"{source_id}.yaml",
                {
                    "schema_version": 2,
                    "id": source_id,
                    "kind": "huggingface_dataset",
                    "canonical_uri": "https://huggingface.co/datasets/common-pile/project_gutenberg_filtered",
                    "revision": "a" * 40,
                    "license": "Public Domain source claim",
                    "license_url": "https://www.gutenberg.org/policy/license",
                    "rights": {
                        "training_eligibility": "review_required",
                        "redistribution_mode": "review_required",
                    },
                    "domains": [stratum],
                    "document_kinds": ["prose"],
                    "source_family": source_id,
                    "acquisition": {
                        "config": "default",
                        "split": "train",
                        "text_field": "text",
                        "max_rows": 2,
                        "max_bytes": 65536,
                        "bounded_shards": [
                            {
                                "path": shard,
                                "expected_sha256": "b" * 64,
                                "max_shard_bytes": 65536,
                                "max_scanned_rows": 2,
                                "hash_modulus": 1,
                                "hash_remainders": [0],
                                "declared_config": "default",
                                "declared_split": "train",
                            }
                        ],
                    },
                },
            )
            continue
        files = []
        for index in range(12):
            number = 0 if index == 10 else index
            prose = (
                f"# {source_id} topic {number}\n"
                f"The {source_id} topic {number} explains a concrete practice. "
                "A reader can follow the steps and check the result.\n"
            )
            if index == 11:
                prose = "# Exception\nSPDX-License-Identifier: Proprietary\n"
            files.append((f"docs/{index:02}.md", prose.encode()))
        files_by_source[source_id] = files
        _write_yaml(
            recipe / "sources" / f"{source_id}.yaml",
            {
                "schema_version": 2,
                "id": source_id,
                "kind": "git",
                "canonical_uri": f"https://github.com/example/{source_id}",
                "revision": "a" * 40,
                "license": "Apache-2.0 source claim",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "rights": {
                    "training_eligibility": "review_required",
                    "redistribution_mode": "review_required",
                },
                "domains": [stratum],
                "document_kinds": ["markdown"],
                "source_family": source_id,
                "acquisition": {"include": ["docs/*.md"], "max_bytes": 65536},
            },
        )
    _write_yaml(
        recipe / "splits-placeholder.yaml",
        {
            "schema_version": 1,
            "unit": "source_document_family",
            "family_key": "source_family",
            "assignments": {source_id: "train" for source_id in strata},
        },
    )
    _write_yaml(
        recipe / "release-quarantine.yaml",
        {
            "schema_version": 2,
            "publication_mode": "metadata_reconstruction_only",
            "mixture": {
                "general_prose": 0.4,
                "incident_response_docs": 0.2,
                "explanatory_prose": 0.4,
            },
            "lm": {"selected": False, "training_splits": []},
            "chat": {"selected": False, "training_splits": []},
        },
    )
    _write_yaml(
        recipe / "transforms/lm.yaml",
        {
            "id": "fixture_lm",
            "version": "1",
            "kind": "lm_text",
            "parameters": {},
            "inputs": list(strata),
        },
    )
    _write_yaml(
        recipe / "acquire.yaml",
        {
            "schema_version": 1,
            "id": project_id,
            "sources": [f"sources/{source_id}.yaml" for source_id in strata],
            "transforms": [],
            "splits": "splits-placeholder.yaml",
            "release": "release-quarantine.yaml",
        },
    )
    project = load_project(recipe / "acquire.yaml")
    entries = {}
    for source in project.sources:
        files = [
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for name, raw in files_by_source[source.id]
        ]
        identity = canonical_digest(
            {
                "declaration_sha256": declaration_sha256(source),
                "adapter": _adapter(source),
                "files": files,
            }
        )
        snapshot = root / "corpora" / project_id / "snapshots" / source.id / identity
        for name, raw in files_by_source[source.id]:
            destination = snapshot / "files" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        manifest = {
            "schema_version": 1,
            "source_id": source.id,
            "declaration": source_declaration_payload(source),
            "declaration_sha256": declaration_sha256(source),
            "adapter": _adapter(source),
            "files": files,
            "retrieval": retrieval_by_source.get(
                source.id, {"fixture": "network-disabled immutable snapshot"}
            ),
            "snapshot_sha256": identity,
        }
        (snapshot / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        entries[source.id] = {
            "declaration_sha256": declaration_sha256(source),
            "snapshot_sha256": identity,
            "snapshot_path": str(snapshot),
            "receipt": {"status": "acquired", "retrieval": manifest["retrieval"]},
        }
    lock = {
        "schema_version": 1,
        "project_id": project_id,
        "project_sha256": _project_sha(project),
        "sources": entries,
    }
    (root / "corpora" / project_id / "acquisition.json").write_bytes(
        canonical_json(lock) + b"\n"
    )
    return recipe, strata


def _retained_fixture_sources(root: Path, recipe: Path, origin_id: str) -> list[dict]:
    """Construct a local retained-copy fixture, never invoke source acquisition.

    The prior fixture is the immutable origin. Copy its verified bytes into the
    candidate and bind both locations exactly as the native retained reader does.
    Scientific IDs/hashes below describe these generated fixtures only.
    """
    origin_lock = verify_acquisition(
        load_project(root / origin_id / "acquire.yaml"), root
    )
    lock_path = root / "corpora" / recipe.name / "acquisition.json"
    lock = json.loads(lock_path.read_text())
    effects = []
    for source_id, entry in sorted(lock["sources"].items()):
        retained = origin_lock["sources"][source_id]
        origin = Path(retained["snapshot_path"])
        copied = Path(entry["snapshot_path"])
        manifest = verify_snapshot(origin)
        assert entry["declaration_sha256"] == retained["declaration_sha256"]
        assert entry["snapshot_sha256"] == retained["snapshot_sha256"]
        shutil.copytree(origin, copied, dirs_exist_ok=True)
        assert verify_snapshot(copied) == manifest
        effects.append(
            {
                "source_id": source_id,
                "declaration_sha256": retained["declaration_sha256"],
                "effect": "reuse_only",
                "origin_project_id": origin_id,
                "snapshot_sha256": retained["snapshot_sha256"],
            }
        )
        entry["reuse_origin"] = {
            "project_id": origin_id,
            "declaration_sha256": retained["declaration_sha256"],
            "snapshot_sha256": retained["snapshot_sha256"],
            "path": str(origin),
        }
    acquisition = recipe / "acquire.yaml"
    _write_yaml(
        acquisition,
        {**yaml.safe_load(acquisition.read_text()), "source_effects": effects},
    )
    lock["project_sha256"] = _project_sha(load_project(acquisition))
    lock_path.write_bytes(canonical_json(lock) + b"\n")
    assert verify_acquisition(load_project(acquisition), root) == lock
    return effects


def _policy(recipe: Path, strata: dict[str, str]) -> tuple[Path, Path]:
    document = recipe / "policy.md"
    document.write_text("Reviewed fixture source policy, Apache-2.0 local use.\n")
    # Native policy shape, with fixture-only source decisions and exact local
    # project/lock/snapshot identities. No production decision is synthesized.
    lock_path = recipe.parent / "corpora" / recipe.name / "acquisition.json"
    lock = json.loads(lock_path.read_text())
    template = recipe / "application.json"
    template.write_bytes(
        canonical_json(
            {
                **json.loads((_PACKET / "application.template.json").read_text()),
                "application_binding": {
                    "project_sha256": lock["project_sha256"],
                    "acquisition_lock_sha256": sha256_file(lock_path),
                    "sources": {
                        source_id: {
                            "declaration_sha256": row["declaration_sha256"],
                            "snapshot_sha256": row["snapshot_sha256"],
                        }
                        for source_id, row in lock["sources"].items()
                    },
                },
                "policy_id": "fixture-policy-v1",
                "policy_sha256": sha256_file(document),
                "sources": [
                    {
                        "source_id": source_id,
                        "license_label": (
                            "US public-domain local research"
                            if source_id == "books_rows"
                            else "Apache-2.0 local research"
                        ),
                        "rights": {
                            "training_eligibility": "eligible_with_obligations",
                            "redistribution_mode": "metadata_reconstruction_only",
                            **(
                                {}
                                if source_id == "books_rows"
                                else {"spdx_expression": "Apache-2.0"}
                            ),
                            "license_references": [
                                "https://www.apache.org/licenses/LICENSE-2.0"
                            ],
                            "notices": ["Keep source attribution"],
                        },
                    }
                    for source_id in strata
                ],
            }
        )
        + b"\n"
    )
    return template, document


def _reviewed_release(recipe: Path, admission: Path) -> None:
    _write_yaml(
        recipe / "release-reviewed.yaml",
        {
            "schema_version": 2,
            "publication_mode": "metadata_reconstruction_only",
            "mixture": {
                "general_prose": 0.4,
                "incident_response_docs": 0.2,
                "explanatory_prose": 0.4,
            },
            "lm": {"selected": True, "training_splits": ["train", "validation"]},
            "chat": {"selected": False, "training_splits": []},
            "record_admission": {
                "path": admission.name,
                "sha256": sha256_file(admission),
            },
            "normalizer": "normalizer-structure-v3",
        },
    )
    _write_yaml(
        recipe / "pre-freeze.yaml",
        {
            **yaml.safe_load((recipe / "acquire.yaml").read_text()),
            "release": "release-reviewed.yaml",
        },
    )


def _clusters(
    path: Path,
    inventory: Path,
    strata: dict[str, str],
    prior: tuple[Path, Path] | None = None,
    *,
    seed: str = "fixture-seed",
) -> None:
    payload = {
        **json.loads((_PACKET / "family-clusters-reviewed.template.json").read_text()),
        "inventory_sha256": sha256_file(inventory),
        "seed": seed,
        "source_strata": strata,
        "merges": [
            {
                "family_id": f"git-file:{source_id}:docs/00.md",
                "member_hints": [
                    f"git-file:{source_id}:docs/00.md",
                    f"git-file:{source_id}:docs/10.md",
                ],
            }
            for source_id in strata
            if source_id != "books_rows"
        ],
        "reviewer": "Fixture independent reviewer",
        "reviewed_on": "2026-10-09",
    }
    # The initial tiny fixture has no prior release; the connected candidate
    # substitutes its authenticated tiny predecessor into the actual template.
    payload.pop("prior_family_inventory")
    if prior is not None:
        family, release = prior
        payload["prior_family_inventory"] = {
            "path": str(family),
            "sha256": sha256_file(family),
            "release_path": str(release),
        }
    path.write_bytes(canonical_json(payload) + b"\n")


def _prior(root: Path) -> tuple[Path, Path, dict[str, str]]:
    recipe, strata = _sources(root, "fixture-prior")
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    result = draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, admission
    )
    assert result["counts"] == {"qualify": 34, "exclude": 0, "quarantine": 4}
    _reviewed_release(recipe, admission)
    project = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    write_split_inventory(project, root, inventory)
    clusters = recipe / "clusters.json"
    _clusters(clusters, inventory, strata)
    frozen = recipe / "splits-reviewed.yaml"
    freeze_splits(project, root, inventory, clusters, frozen)
    _write_yaml(
        recipe / "build.yaml",
        {
            **yaml.safe_load((recipe / "pre-freeze.yaml").read_text()),
            "transforms": ["transforms/lm.yaml"],
            "splits": frozen.name,
        },
    )
    release = freeze(
        build(load_project(recipe / "build.yaml"), root, offline=True), root
    )
    assert verify_release(release)["release_id"] == release.name
    family = recipe / "family.jsonl"
    finalize_family_inventory(release, frozen, family)
    return release, family, strata


def _authenticated_fixture_tokenizer(
    root: Path, release: Path, *, vocab_size: int = 300
) -> tuple[Path, Path]:
    """Reuse the existing cross-release test pattern with a prebuilt WordLevel model."""
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), vocab_size, root
    )
    config_path = exported / "tokenizer.yaml"
    config = load_tokenizer_config(config_path)
    tokenizer_path = config.output_dir / "tokenizer.json"
    tokenizer_path.parent.mkdir(parents=True)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "[UNK]": 3}
    vocab.update({f"unused-{index}": index for index in range(4, vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_path))
    binding = verify_release_export(config.dataset)
    (tokenizer_path.parent / "tokenizer_manifest.json").write_bytes(
        canonical_json(
            {
                "source": "local_text",
                "revision": release.name,
                "vocab_size": vocab_size,
                "sha256": sha256_file(tokenizer_path),
                "corpus_export": binding,
                "training_contract": {"corpus_export": binding},
            }
        )
        + b"\n"
    )
    return config_path, tokenizer_path


def _native(root: Path, *words: str) -> list[str]:
    return [
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "klab",
        "--work-dir",
        str(root),
        *words,
    ]


def _out(paths: dict[str, Path | str], label: str) -> Path:
    attempt = paths["attempt"]
    assert isinstance(attempt, Path)
    return phase_output_paths(attempt, label, _PHASE_MAP[label])["leaf"]


def _phase(
    paths: dict[str, Path | str],
    label: str,
    *words: str,
    completion_override: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    root = paths["root"]
    assert isinstance(root, Path)
    attempt = paths["attempt"]
    assert isinstance(attempt, Path)
    phase_paths = phase_output_paths(attempt, label, _PHASE_MAP.get(label))
    leaf = _native(root, *words)
    classified = classify_attempt_command(leaf)
    if "leaf" in phase_paths and "--output" in classified.options:
        assert Path(str(classified.options["--output"])) == phase_paths["leaf"]
    guarded = _native(
        root,
        "monitor",
        "--policy",
        str(paths["prep"]),
        "--log-dir",
        str(phase_paths["inner_monitor"]),
        "--workspace",
        str(root),
        "--baseline",
        str(paths["baseline"]),
        "--",
        *leaf,
    )
    command = _native(
        root,
        "attempt",
        "run",
        "--ledger",
        str(paths["ledger"]),
        "--label",
        label,
        "--activity",
        "inspect",
        "--content-identity-sha256",
        str(paths["identity"]),
        "--policy",
        str(paths["whole"]),
        "--baseline",
        str(paths["baseline"]),
        "--workspace",
        str(root),
        "--completion",
        str(completion_override or phase_paths["completion"]),
        "--updates",
        "0",
        "--target-positions",
        "0",
        "--generation-calls",
        "0",
        "--generated-tokens",
        "0",
        "--receipt-kind",
        "none",
        "--",
        *guarded,
    )
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=_PHASE_TIMEOUT_SECONDS,
        check=False,
        env=paths["env"],
    )


def _run(
    paths: dict[str, Path | str], label: str, bindings: dict[str, str] | None = None
) -> None:
    phase = compile_phase(_PLAN, Path(paths["attempt"]), label, bindings or {})
    completed = subprocess.run(
        _native(
            Path(paths["root"]),
            "attempt",
            "run-phase",
            "--plan",
            str(_PLAN),
            "--attempt-root",
            str(paths["attempt"]),
            "--label",
            label,
            "--bindings-json",
            json.dumps(bindings or {}),
            "--ledger",
            str(paths["ledger"]),
            "--workspace",
            str(paths["root"]),
            "--policy",
            str(paths["whole"]),
            "--preparation-policy",
            str(paths["prep"]),
            "--baseline",
            str(paths["baseline"]),
            "--content-identity-sha256",
            str(paths["identity"]),
        ),
        capture_output=True,
        text=True,
        timeout=_PHASE_TIMEOUT_SECONDS,
        check=False,
        env=paths["env"],
    )
    assert completed.returncode == 0, (label, completed.stdout, completed.stderr)
    receipt = json.loads(Path(phase["completion"]).read_text())
    assert receipt["living_descendants"] == 0


def _render(
    paths: dict[str, Path | str],
    label: str,
    template: Path,
    output: Path,
    values: dict[str, str] | None = None,
) -> None:
    assert output == _out(paths, label)
    _run(
        paths,
        label,
        {
            "TEMPLATE": str(template),
            "VALUES_JSON": json.dumps(values or {}, sort_keys=True),
        },
    )


def _attempt(root: Path, acquisition: Path) -> dict[str, Path | str]:
    attempt = root / "attempt"
    attempt.mkdir()
    baseline = attempt / "workspace-baseline.json"
    capture = subprocess.run(
        _native(
            root,
            "monitor-baseline",
            str(root),
            "--output",
            str(baseline),
            "--seconds",
            "2",
            "--json",
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert capture.returncode == 0, (capture.stdout, capture.stderr)
    for name in ("prep/sources", "prep/transforms", "receipts", "logs", "policies"):
        (attempt / name).mkdir(parents=True)
    whole = attempt / "policies" / "monitor-whole.yaml"
    prep = attempt / "policies" / "monitor-preparation.yaml"
    policy = {
        "monitor_policy_version": 1,
        "interval_seconds": 0.1,
        "termination_grace_seconds": 0.2,
        "max_tree_rss_bytes": 4 * 1024**3,
        "max_added_workspace_bytes": 512 * 1024**2,
        "max_added_workspace_inodes": 5000,
        "max_wall_seconds": _ATTEMPT_WALL_SECONDS,
    }
    _write_yaml(whole, policy)
    _write_yaml(prep, policy)
    project = load_project(acquisition)
    identity = "a" * 64
    # Use the sealed native contract shape with explicit tiny-fixture identity
    # and resource substitutions. Baseline identity still comes from the public
    # renderer, never the baseline receipt's file hash.
    template = _fixture_template(
        acquisition.parent,
        "attempt-contract.template.json",
        max_wall_seconds=_ATTEMPT_WALL_SECONDS,
        content_identity_sha256=identity,
        monitor_policy_sha256=sha256_file(whole),
        preparation_monitor_policy_sha256=sha256_file(prep),
        acquisition_project_sha256=sha256_file(acquisition),
        preparation_acquisition_identity_sha256=_project_sha(project),
        admission_lock_sha256=sha256_file(
            root / "corpora" / project.config.id / "acquisition.json"
        ),
        admission_policy_sha256=sha256_file(acquisition.parent / "policy.md"),
        admission_selection_sha256=sha256_file(
            acquisition.parent / "inspection-selection.json"
        ),
        fixed_profile_sha256=None,
        fixed_family_inventory_sha256=None,
    )
    contract = attempt / "attempt-contract.json"
    rendered = subprocess.run(
        _native(
            root,
            "corpus",
            "render-declaration",
            "--template",
            str(template),
            "--values-json",
            "{}",
            "--workspace-baseline",
            str(baseline),
            "--output",
            str(contract),
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert rendered.returncode == 0, (rendered.stdout, rendered.stderr)
    ledger = attempt / "attempt-ledger.sqlite"
    initialized = subprocess.run(
        _native(
            root,
            "attempt",
            "init",
            "--ledger",
            str(ledger),
            "--contract",
            str(contract),
            "--contract-sha256",
            sha256_file(contract),
            "--policy",
            str(whole),
            "--baseline",
            str(baseline),
            "--workspace",
            str(root),
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert initialized.returncode == 0, (initialized.stdout, initialized.stderr)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    return {
        "root": root,
        "attempt": attempt,
        "ledger": ledger,
        "baseline": baseline,
        "whole": whole,
        "prep": prep,
        "identity": identity,
        "env": env,
    }


def _bind(paths: dict[str, Path | str], kind: str, path: Path) -> None:
    root = paths["root"]
    assert isinstance(root, Path)
    command = _native(
        root,
        "attempt",
        "bind-artifact",
        "--ledger",
        str(paths["ledger"]),
        "--kind",
        kind,
        "--path",
        str(path),
        "--sha256",
        sha256_file(path),
        "--content-identity-sha256",
        str(paths["identity"]),
        "--workspace",
        str(root),
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env=paths["env"],
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)


@pytest.mark.skipif(os.name != "posix", reason="owned-process fixture requires POSIX")
def test_actual_admission_draft_collision_stops_before_leaf_and_reservation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    recipe, strata = _sources(root, "fixture-candidate")
    template, policy = _policy(recipe, strata)
    (recipe / "inspection-selection.json").write_text("{}\n")
    paths = _attempt(root, recipe / "acquire.yaml")
    collision_output = recipe / "collision-draft.json"
    denied = _phase(
        paths,
        "draft-collision",
        "corpus",
        "admission-draft",
        str(recipe / "acquire.yaml"),
        "--template",
        str(template),
        "--policy-document",
        str(policy),
        "--output",
        str(collision_output),
        completion_override=collision_output,
    )
    assert denied.returncode != 0
    assert "attempt output path collision" in denied.stderr
    assert not collision_output.exists()
    assert not phase_output_paths(root, "draft-collision")["inner_monitor"].exists()
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


@pytest.mark.skipif(os.name != "posix", reason="owned-process fixture requires POSIX")
def test_offline_preparation_sequence_through_native_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Observe the real public calls, so a new phase/probe cannot silently exceed
    # the aggregate allowance even if this machine happens to execute it quickly.
    native_run = subprocess.run
    command_bounds = []

    def record_bound(argv, *args, **kwargs):
        if "attempt" in argv:
            operation = argv[argv.index("attempt") + 1]
            if operation in {"run-phase", "run", "bind-artifact"}:
                command_bounds.append((operation, kwargs["timeout"]))
        return native_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_bound)
    root = tmp_path / "work"
    root.mkdir()
    prior_release, prior_family, strata = _prior(root)
    tokenizer_config, tokenizer_path = _authenticated_fixture_tokenizer(
        root, prior_release
    )
    recipe, _ = _sources(root, "fixture-candidate")
    retained_effects = _retained_fixture_sources(root, recipe, "fixture-prior")
    template, policy = _policy(recipe, strata)
    (root / "sitecustomize.py").write_text(
        "import socket\n"
        "def blocked(*args, **kwargs):\n"
        "    raise RuntimeError('network-disabled fixture')\n"
        "socket.socket.connect = blocked\n"
        "socket.create_connection = blocked\n"
    )
    # A local copied snapshot is insufficient when its declared retained origin
    # disappears. This negative probe is pre-ledger and cannot reacquire bytes.
    retained_manifest = (
        root
        / "corpora"
        / "fixture-prior"
        / "snapshots"
        / retained_effects[0]["source_id"]
        / retained_effects[0]["snapshot_sha256"]
        / "manifest.json"
    )
    original_manifest = retained_manifest.read_bytes()
    held_manifest = retained_manifest.with_name("manifest.fixture-held")
    retained_manifest.rename(held_manifest)
    lock_path = root / "corpora" / "fixture-candidate" / "acquisition.json"
    lock_identity = sha256_file(lock_path)
    offline_env = os.environ.copy()
    offline_env["PYTHONPATH"] = (
        str(root) + os.pathsep + offline_env.get("PYTHONPATH", "")
    )
    offline_env["HF_HUB_OFFLINE"] = "1"
    try:
        denied_origin = subprocess.run(
            _native(
                root, "corpus", "acquire", str(recipe / "acquire.yaml"), "--offline"
            ),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=offline_env,
        )
        assert denied_origin.returncode != 0
        assert "snapshot" in denied_origin.stderr
        assert "network-disabled fixture" not in denied_origin.stderr
        assert sha256_file(lock_path) == lock_identity
        assert not (lock_path.parent / "transport-budget.sqlite").exists()
    finally:
        held_manifest.rename(retained_manifest)
    assert retained_manifest.read_bytes() == original_manifest
    release_template = _fixture_template(
        recipe,
        "release-reviewed.template.yaml",
        mixture=yaml.safe_load(
            (root / "fixture-prior/release-reviewed.yaml").read_text()
        )["mixture"],
    )
    inspection_selection = recipe / "inspection-selection.json"
    inspection_selection.write_bytes(
        canonical_json(
            {
                "format": "klab-admission-inspection-selection-v1",
                "seed": "fixture-pre-admission-review-v1",
                "normalizer": "normalizer-structure-v3",
                "sources": [
                    {
                        "source_id": source_id,
                        "exception_count": 1,
                        "strata": [
                            {
                                "id": "all",
                                "count": 1,
                                "path_prefix": None,
                                "length_band": None,
                                "issues_nonempty": None,
                            }
                        ],
                    }
                    for source_id in strata
                ],
            }
        )
        + b"\n"
    )
    admission_review_template = _PACKET / "admission-review.template.json"
    fixture_project = yaml.safe_load((recipe / "acquire.yaml").read_text())
    # Only corpus identities/effects and transport limits vary with tiny inputs.
    # The native fixture source filenames and split references remain intact.
    prefreeze_template = _fixture_template(
        recipe,
        "project-pre-freeze.template.yaml",
        id=fixture_project["id"],
        source_effects=retained_effects,
        transport_budget=None,
    )
    cluster_template = recipe / "clusters.template.json"
    _clusters(
        cluster_template,
        root / "fixture-prior/inventory.jsonl",
        strata,
        (prior_family, prior_release),
    )
    cluster_spec = json.loads(cluster_template.read_text())
    cluster_spec["inventory_sha256"] = "${INVENTORY_SHA256}"
    cluster_template.write_bytes(canonical_json(cluster_spec) + b"\n")
    build_template = _fixture_template(
        recipe,
        "project-build.template.yaml",
        id=fixture_project["id"],
        source_effects=retained_effects,
        transport_budget=None,
    )
    prior_rows = [json.loads(line) for line in prior_family.read_text().splitlines()]
    protected = next(row for row in prior_rows if row["split"] == "test")
    profile = recipe / "profile.json"
    profile.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "release_manifest_sha256": sha256_file(prior_release / "manifest.json"),
                "documents_sha256": sha256_file(prior_release / "documents.jsonl"),
                "loss_slices": [protected],
            }
        )
    )
    suite = recipe / "suite.json"
    suite.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "family_inventory_sha256": sha256_file(prior_family),
                "content_sha256": "b" * 64,
                "items": [
                    {
                        "split": "test",
                        "parent_document_ids": [protected["document_id"]],
                        "parent_family": protected["family_id"],
                    }
                ],
                "chunks": [{"document_id": protected["document_id"]}],
            }
        )
    )
    acceptance_template = _PACKET / "release-acceptance.template.json"
    floors = recipe / "token-floors.yaml"
    _write_yaml(
        floors,
        {"min_unique_train_tokens_by_domain": {name: 1 for name in strata.values()}},
    )
    mixture_template = _fixture_template(
        recipe,
        "mixture.template.yaml",
        tokenizer_config=str(tokenizer_config),
        source_strata=strata,
        target_quotas={
            "general_prose": 40,
            "incident_response_docs": 20,
            "explanatory_prose": 40,
        },
        tokenizer_origin_release_id=prior_release.name,
    )
    base_config = load_config(Path("configs/runtime_smoke_cpu.yaml")).model_dump(
        mode="json"
    )
    base_config["name"] = "fixture-prepared-inputs-only"
    base_config["model"]["vocab_size"] = 300
    base_config["tokenizer"]["path"] = str(tokenizer_path)
    operator_run = yaml.safe_load((_PACKET / "prepared-run.template.yaml").read_text())
    fixture_dataset = {
        **operator_run["dataset"],
        "train_max_tokens": 100,
        "license": _licenses(prior_release),
    }
    base_config["training"].update(seq_len=16, max_steps=7, max_tokens=100)
    base_config["checkpoint"]["every_steps"] = 1
    base_config["evaluation"]["every_steps"] = 1
    run_template = _fixture_template(
        recipe,
        "prepared-run.template.yaml",
        name=base_config["name"],
        runtime=base_config["runtime"],
        model=base_config["model"],
        tokenizer=base_config["tokenizer"],
        training=base_config["training"],
        optimizer=base_config["optimizer"],
        checkpoint=base_config["checkpoint"],
        evaluation=base_config["evaluation"],
        dataset=fixture_dataset,
    )
    envelope = recipe / "resource-envelope.yaml"
    _write_yaml(
        envelope,
        {
            "resource_envelope_version": 1,
            "max_rss_bytes": 4 * 1024**3,
            "min_disk_bytes": 0,
            "min_inodes": 0,
            "max_workers": 2,
            "spill_to_disk": True,
        },
    )
    paths = _attempt(root, recipe / "acquire.yaml")
    denied_model = _phase(
        paths,
        "forbidden-model-stage",
        "stage",
        str(recipe / "missing-config.yaml"),
        "--through",
        "validate",
        "--output",
        str(recipe / "forbidden-stage"),
    )
    assert denied_model.returncode != 0
    assert (
        "training or staging cannot be hidden in a nested monitor"
        in denied_model.stderr
    )
    with pytest.raises(Exception, match="preparation-only attempt forbids model work"):
        AttemptBudget(paths["ledger"]).run_contract(
            _native(
                root,
                "stage",
                str(recipe / "missing-config.yaml"),
                "--through",
                "validate",
                "--output",
                str(recipe / "forbidden-direct-stage"),
            ),
            activity="inspect",
            label="forbidden-direct-stage",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=phase_output_paths(paths["attempt"], "forbidden-direct-stage")[
                "completion"
            ],
        )
    # All-reuse materialization is now allowed by the offline contract. Keep
    # this negative probe genuinely live without changing the bound project.
    live_probe = yaml.safe_load((recipe / "acquire.yaml").read_text())
    effect = live_probe["source_effects"][0]
    effect.update(effect="acquire")
    effect.pop("origin_project_id")
    effect.pop("snapshot_sha256")
    live_probe_path = recipe / "live-acquisition-probe.yaml"
    _write_yaml(live_probe_path, live_probe)
    denied_live = _phase(
        paths,
        "forbidden-live-acquisition",
        "corpus",
        "acquire",
        str(live_probe_path),
    )
    assert denied_live.returncode != 0
    assert "verified offline reuse only" in denied_live.stderr
    _run(
        paths, "verify-snapshots", {"ACQUISITION_PROJECT": str(recipe / "acquire.yaml")}
    )
    application = _out(paths, "application-declaration")
    _render(paths, "application-declaration", template, application)
    draft_path = _out(paths, "admission-draft")
    _run(
        paths,
        "admission-draft",
        {
            "ACQUISITION_PROJECT": str(recipe / "acquire.yaml"),
            "POLICY_DOCUMENT": str(policy),
        },
    )
    draft = json.loads(draft_path.read_text())
    assert all(
        sum(
            row["decision"] == "quarantine"
            for row in item.get("files", item.get("records", []))
        )
        == 1
        for item in draft["sources"]
    )
    inspection_path = _out(paths, "admission-inspection")
    _run(
        paths,
        "admission-inspection",
        {
            "ACQUISITION_PROJECT": str(recipe / "acquire.yaml"),
            "POLICY_DOCUMENT": str(policy),
            "INSPECTION_SELECTION": str(inspection_selection),
            "MAX_INPUT_BYTES": "1048576",
            "MAX_EXCERPT_BYTES": "256",
            "MAX_OUTPUT_BYTES": "262144",
        },
    )
    inspected = json.loads(inspection_path.read_text())
    assert len(inspected["items"]) == len(strata)
    assert len(inspected["quarantine_exceptions"]) == len(strata)
    assert all(item["decision"] == "qualify" for item in inspected["items"])
    assert all(
        item["raw_excerpt"] and item["cleaned_excerpt"] for item in inspected["items"]
    )
    admission = _out(paths, "reviewed-admission")
    _render(paths, "reviewed-admission", draft_path, admission)
    _render(
        paths,
        "admission-review",
        admission_review_template,
        _out(paths, "admission-review"),
        {
            "REVIEWER": "Synthetic fixture reviewer",
            "REVIEWED_ON": "2026-10-10",
            "DRAFT_PATH": str(draft_path),
            "DRAFT_SHA256": sha256_file(draft_path),
            "ADMISSION_SHA256": sha256_file(admission),
            "INSPECTION_PATH": str(inspection_path),
            "INSPECTION_SHA256": sha256_file(inspection_path),
            "ITEM_DECISIONS_JSON": json.dumps(
                [
                    {
                        "item_id": item["item_id"],
                        "outcome": "pass",
                        "note": "Fixture source notice and selected content reviewed.",
                    }
                    for item in inspected["items"]
                ]
            ),
        },
    )
    _render(
        paths,
        "reviewed-release-declaration",
        release_template,
        _out(paths, "reviewed-release-declaration"),
        {
            "REVIEWED_ADMISSION_SHA256": sha256_file(admission),
        },
    )
    for label, source_id in (
        ("source-scoutflo", "books"),
        ("source-gutenberg", "books_rows"),
        ("source-pagerduty", "incident"),
        ("source-wikimedia", "wiki"),
    ):
        _render(
            paths, label, recipe / "sources" / f"{source_id}.yaml", _out(paths, label)
        )
    _render(
        paths,
        "transform-declaration",
        recipe / "transforms/lm.yaml",
        _out(paths, "transform-declaration"),
    )
    _render(
        paths,
        "placeholder-split-declaration",
        recipe / "splits-placeholder.yaml",
        _out(paths, "placeholder-split-declaration"),
    )
    prefreeze = _out(paths, "pre-freeze-declaration")
    _render(paths, "pre-freeze-declaration", prefreeze_template, prefreeze)
    _bind(paths, "pre_freeze_project", prefreeze)
    inventory = _out(paths, "split-inventory")
    _run(paths, "split-inventory")
    clusters = _out(paths, "reviewed-family-decisions")
    _render(
        paths,
        "reviewed-family-decisions",
        cluster_template,
        clusters,
        {"INVENTORY_SHA256": sha256_file(inventory)},
    )
    _run(paths, "family-freeze")
    build_project = _out(paths, "final-build-declaration")
    _render(
        paths,
        "final-build-declaration",
        build_template,
        build_project,
    )
    _bind(paths, "build_project", build_project)
    _run(paths, "offline-build")
    builds = [
        path
        for path in (root / "corpora" / "fixture-candidate" / "builds").iterdir()
        if (path / "build.json").is_file()
    ]
    assert len(builds) == 1
    _run(paths, "freeze-release", {"BUILD_PATH": str(builds[0])})
    releases = list((root / "corpora" / "fixture-candidate" / "releases").iterdir())
    assert len(releases) == 1
    release = releases[0]
    assert verify_release(release)["release_id"] == release.name
    _run(paths, "release-audit", {"RELEASE_PATH": str(release)})
    _run(paths, "near-duplicate-audit", {"RELEASE_PATH": str(release)})
    family = _out(paths, "final-family")
    _run(paths, "final-family", {"RELEASE_PATH": str(release)})
    lineage = _out(paths, "protected-lineage")
    _run(
        paths,
        "protected-lineage",
        {
            "RELEASE_PATH": str(release),
            "PRIOR_RELEASE": str(prior_release),
            "PRIOR_FAMILY": str(prior_family),
            "PROFILE": str(profile),
            "SUITE": str(suite),
        },
    )
    assert json.loads(lineage.read_text())["status"] == "PASS"
    denied = _phase(
        paths,
        "supply-before-acceptance",
        "corpus",
        "measure-tokens",
        str(release),
        "--tokenizer",
        str(recipe / "missing-tokenizer.json"),
        "--tokenizer-origin-release",
        str(prior_release),
        "--family-inventory",
        str(family),
        "--policy",
        str(recipe / "missing-policy.yaml"),
        "--output",
        str(recipe / "forbidden-supply.json"),
        "--batch-source-bytes",
        "1024",
        "--json",
    )
    assert denied.returncode != 0 and "not been accepted" in denied.stderr
    review = _out(paths, "release-acceptance")
    _render(
        paths,
        "release-acceptance",
        acceptance_template,
        review,
        {
            "REVIEWER": "Synthetic fixture reviewer",
            "REVIEWED_ON": "2026-10-10",
            "RELEASE_PATH": str(release),
            "RELEASE_ID": release.name,
            "RELEASE_MANIFEST_SHA256": sha256_file(release / "manifest.json"),
            "FAMILY_INVENTORY_PATH": str(family),
            "FAMILY_INVENTORY_SHA256": sha256_file(family),
            "PROTECTED_LINEAGE_PATH": str(lineage),
            "PROTECTED_LINEAGE_SHA256": sha256_file(lineage),
        },
    )
    _bind(paths, "release_acceptance", review)
    _run(
        paths,
        "measure-accepted-supply",
        {
            "RELEASE_PATH": str(release),
            "TOKENIZER_PATH": str(tokenizer_path),
            "PRIOR_RELEASE": str(prior_release),
            "TOKEN_FLOORS": str(floors),
            "BATCH_SOURCE_BYTES": "1024",
        },
    )
    mixture = _out(paths, "mixture-declaration")
    _render(
        paths,
        "mixture-declaration",
        mixture_template,
        mixture,
        {
            "VERIFIED_NEW_RELEASE_DIR": str(release),
            "FINALIZED_NEW_FAMILY_INVENTORY": str(family),
        },
    )
    mixture_output = _out(paths, "materialize-mixture")
    _run(paths, "materialize-mixture")
    _run(paths, "cold-verify-mixture")
    mixture_receipt = json.loads((mixture_output / "receipt.json").read_text())
    assert sum(mixture_receipt["actual_target_tokens"].values()) == 100
    validation_documents, validation_bytes = _split_stats(
        release / "lm/validation.jsonl", "lm"
    )
    run_config = _out(paths, "prepared-run-declaration")
    _render(
        paths,
        "prepared-run-declaration",
        run_template,
        run_config,
        {
            "NEW_RELEASE_ID": release.name,
            "NEW_CACHE_DIR": str(recipe / "cache"),
            "VERIFIED_MIXTURE_DIR": str(mixture_output),
            "VERIFIED_NEW_RELEASE_DIR": str(release),
            "VERIFIED_MIXTURE_TRAIN_DOCUMENTS": str(mixture_receipt["records"]),
            "VERIFIED_RELEASE_VALIDATION_DOCUMENTS": str(validation_documents),
            "VERIFIED_RELEASE_VALIDATION_TOKENS": str(
                validation_bytes + validation_documents + 1
            ),
            "SEALED_MIXTURE_DECLARATION": str(mixture),
            "NEW_ATTEMPT_ROOT": str(paths["attempt"]),
        },
    )
    substituted_config = recipe / "substituted-prepared-run.yaml"
    substituted = yaml.safe_load(run_config.read_text())
    substituted["dataset"]["revision"] = prior_release.name
    _write_yaml(substituted_config, substituted)
    rejected_bundle = _phase(
        paths,
        "reject-substituted-prepared-release",
        "data",
        "prepared-inputs",
        "publish",
        str(substituted_config),
        "--output",
        str(recipe / "forbidden-prepared-bundle"),
        "--resource-envelope",
        str(envelope),
        "--tokenizer-batch-source-bytes",
        "1024",
    )
    assert rejected_bundle.returncode != 0
    assert "prepared config differs from accepted release" in rejected_bundle.stderr
    assert not (recipe / "forbidden-prepared-bundle").exists()
    _run(
        paths,
        "publish-prepared-bundle",
        {"RESOURCE_ENVELOPE": str(envelope), "BATCH_SOURCE_BYTES": "1024"},
    )
    _run(paths, "cold-verify-prepared-bundle")
    status = AttemptBudget(paths["ledger"]).status()
    assert status["charged_updates"] == 0
    assert all(row["actual_updates"] == 0 for row in status["reservations"])
    assert status["resolved_artifacts"][-1]["kind"] == "release_acceptance"
    from collections import Counter

    assert Counter(operation for operation, _ in command_bounds) == {
        "run-phase": len(_PHASES),
        "run": _REFUSAL_PROBES,
        "bind-artifact": _BIND_CALLS,
    }
    assert sum(seconds for _, seconds in command_bounds) <= (
        _ATTEMPT_WALL_SECONDS - _LOCAL_METADATA_ALLOWANCE_SECONDS
    )


def test_freeze_rejects_missing_admission_and_mismatched_normalizer_or_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    recipe, strata = _sources(root, "fixture-rejections")
    acquisition = load_project(recipe / "acquire.yaml")
    wrong_inventory = recipe / "acquisition-inventory.jsonl"
    write_split_inventory(acquisition, root, wrong_inventory)
    wrong_clusters = recipe / "acquisition-clusters.json"
    _clusters(wrong_clusters, wrong_inventory, strata)
    with pytest.raises(ValueError, match="reviewed record admission"):
        freeze_splits(
            acquisition,
            root,
            wrong_inventory,
            wrong_clusters,
            recipe / "forbidden.yaml",
        )
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    draft_admission_manifest(acquisition, root, template, policy, admission)
    _reviewed_release(recipe, admission)
    prefreeze = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    write_split_inventory(prefreeze, root, inventory)
    clusters = recipe / "clusters.json"
    _clusters(clusters, inventory, strata)
    release_file = recipe / "release-reviewed.yaml"
    release_data = yaml.safe_load(release_file.read_text())
    release_data["normalizer"] = "normalizer-structure-v2"
    _write_yaml(release_file, release_data)
    with pytest.raises(ValueError, match="inventory differs"):
        freeze_splits(
            load_project(recipe / "pre-freeze.yaml"),
            root,
            inventory,
            clusters,
            recipe / "wrong-normalizer.yaml",
        )
    release_data["normalizer"] = "normalizer-structure-v3"
    _write_yaml(release_file, release_data)
    rows = inventory.read_bytes().splitlines()
    rows[0] = rows[0].replace(b'"books"', b'"other"', 1)
    inventory.write_bytes(b"\n".join(rows) + b"\n")
    with pytest.raises(ValueError, match="invalid reviewed family-cluster"):
        freeze_splits(
            load_project(recipe / "pre-freeze.yaml"),
            root,
            inventory,
            clusters,
            recipe / "wrong-inventory.yaml",
        )


def test_real_lineage_audit_blocks_protected_family_moved_to_train(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    prior_release, prior_family, strata = _prior(root)
    prior_rows = [json.loads(line) for line in prior_family.read_text().splitlines()]
    protected = next(row for row in prior_rows if row["split"] == "test")
    recipe, _ = _sources(root, "fixture-leakage")
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, admission
    )
    _reviewed_release(recipe, admission)
    project = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    write_split_inventory(project, root, inventory)
    selected: Path | None = None
    for index in range(10):
        clusters = recipe / f"clusters-{index}.json"
        _clusters(clusters, inventory, strata, seed=f"different-{index}")
        frozen = recipe / f"splits-{index}.yaml"
        freeze_splits(project, root, inventory, clusters, frozen)
        from klab.corpus.bounded_metadata import metadata_operation
        from klab.corpus.split_assignments import Assignments, assignment_path

        reference = yaml.safe_load(frozen.read_text())["assignment_file"]
        with metadata_operation(root / "test-split-lookup"):
            assignments = Assignments(
                assignment_path(frozen.parent, reference), reference
            )
            protected_split = assignments[protected["document_id"]]
        if protected_split == "train":
            selected = frozen
            break
    assert selected is not None
    _write_yaml(
        recipe / "build.yaml",
        {
            **yaml.safe_load((recipe / "pre-freeze.yaml").read_text()),
            "transforms": ["transforms/lm.yaml"],
            "splits": selected.name,
        },
    )
    release = freeze(
        build(load_project(recipe / "build.yaml"), root, offline=True), root
    )
    family = recipe / "family.jsonl"
    finalize_family_inventory(release, selected, family)
    profile = recipe / "profile.json"
    profile.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "release_manifest_sha256": sha256_file(prior_release / "manifest.json"),
                "documents_sha256": sha256_file(prior_release / "documents.jsonl"),
                "loss_slices": [protected],
            }
        )
    )
    suite = recipe / "suite.json"
    suite.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "family_inventory_sha256": sha256_file(prior_family),
                "content_sha256": "c" * 64,
                "items": [
                    {
                        "split": "test",
                        "parent_document_ids": [protected["document_id"]],
                        "parent_family": protected["family_id"],
                    }
                ],
                "chunks": [{"document_id": protected["document_id"]}],
            }
        )
    )
    output = recipe / "blocked-lineage.json"
    result = audit_protected_lineage(
        prior_release, release, prior_family, family, profile, suite, output
    )
    assert result["status"] == "BLOCKED"
    assert result["new_train_family_collisions"]
    assert output.exists()


def test_split_preparation_reuses_verified_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    recipe, strata = _sources(root, "fixture-split-proof-reuse")
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, admission
    )
    _reviewed_release(recipe, admission)
    project = load_project(recipe / "pre-freeze.yaml")

    with monkeypatch.context() as patch:
        patch.setattr(
            acquisition,
            "_verify_snapshot_cold",
            lambda *args, **kwargs: pytest.fail("verified snapshot was read again"),
        )
        inventory = recipe / "inventory.jsonl"
        write_split_inventory(project, root, inventory)
        clusters = recipe / "clusters.json"
        _clusters(clusters, inventory, strata)
        frozen = recipe / "splits-reviewed.yaml"
        freeze_splits(project, root, inventory, clusters, frozen)

    assert inventory.is_file()
    assert frozen.is_file()


def test_native_inventory_handoff_reuses_completed_proof_and_rejects_changed_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus.split_inventory_adoption import (
        _native_completion_without_lock_binding,
    )
    from klab.corpus.split_inventory_cache import InventoryCache

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    recipe, _ = _sources(root, "fixture-native-inventory-handoff")
    release = recipe / "release-quarantine.yaml"
    release_data = yaml.safe_load(release.read_text())
    release_data["normalizer"] = "normalizer-structure-v3"
    _write_yaml(release, release_data)
    producer = load_project(recipe / "acquire.yaml")
    inventory = recipe / "inventory.jsonl"
    report = write_split_inventory(producer, root, inventory)

    admission = recipe / "admission.json"
    admission.write_text("{}\n")
    _reviewed_release(recipe, admission)
    consumer = load_project(recipe / "pre-freeze.yaml")
    from klab.corpus.completed_preparation import _check_completed_producer

    # Inventory-only production has no admission; the later frozen split binds
    # the consumer's real reviewed admission. Existing admitted producers still
    # must match that admission exactly.
    assert producer.release.record_admission is None
    assert _check_completed_producer(consumer, producer)
    assert _check_completed_producer(consumer, consumer)
    altered = consumer.model_copy(
        update={
            "release": consumer.release.model_copy(
                update={
                    "record_admission": consumer.release.record_admission.model_copy(
                        update={"sha256": "0" * 64}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="source/admission differs"):
        _check_completed_producer(consumer, altered)
    changed_source = producer.sources[0].model_copy(update={"revision": "b" * 40})
    changed_producer = producer.model_copy(
        update={"sources": (changed_source, *producer.sources[1:])}
    )
    with pytest.raises(ValueError, match="source/admission differs"):
        _check_completed_producer(consumer, changed_producer)
    lock = root / "corpora" / producer.config.id / "acquisition.json"
    source = Path(__file__).resolve().parents[1]
    cache = InventoryCache(root)
    dependencies, proof = _native_completion_without_lock_binding(
        cache, consumer, inventory, report, lock, source
    )
    assert str(lock) in dependencies
    assert proof["mode"] == "historical_signed_native_completion"
    assert report["documents"] > 0

    # The signed execution predates the two consumer-side handoff changes.
    # Change only the current authority; the recorded proof and producer files
    # stay real and unchanged, just as they do when continuing R53.
    cache.authority = json.loads(json.dumps(cache.authority))
    for module in ("split_inventory_adoption", "completed_preparation"):
        cache.authority["corpus"]["modules"]["corpus." + module] = "f" * 64
    cache.authority["inventory"]["split_inventory_adoption.py"] = "f" * 64
    reused, transition = _native_completion_without_lock_binding(
        cache, consumer, inventory, report, lock, source
    )
    assert str(lock) in reused
    assert transition["mode"] == "historical_signed_native_completion"

    # A different producer checkout or a modified signed proof cannot be used.
    other_source = tmp_path / "changed-producer"
    for module in ("split_inventory_adoption", "completed_preparation"):
        relative = Path("src/klab/corpus") / (module + ".py")
        destination = other_source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    with (other_source / "src/klab/corpus/completed_preparation.py").open(
        "ab"
    ) as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="producer code changed"):
        _native_completion_without_lock_binding(
            cache, consumer, inventory, report, lock, other_source
        )
    receipt = Path(proof["proof_path"])
    receipt_bytes = receipt.read_bytes()
    modified = json.loads(receipt_bytes)
    modified["hmac"] = "0" * 64
    receipt.write_bytes(canonical_json(modified) + b"\n")
    with pytest.raises(ValueError, match="signature or inputs changed"):
        _native_completion_without_lock_binding(
            cache, consumer, inventory, report, lock, source
        )
    receipt.write_bytes(receipt_bytes)

    lock.write_bytes(lock.read_bytes() + b" ")
    with pytest.raises(ValueError, match="signature or inputs changed"):
        _native_completion_without_lock_binding(
            InventoryCache(root), consumer, inventory, report, lock, source
        )


@pytest.mark.parametrize("outcome", ["success", "digest_mismatch", "producer_error"])
def test_cold_inventory_completion_has_operation_owned_lifetime(
    tmp_path, monkeypatch, outcome
):
    from klab.corpus.split_inventory_cache import (
        InventoryCache,
        verified_inventory,
    )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    recipe, _ = _sources(root, "fixture-cold-inventory")
    project = load_project(recipe / "acquire.yaml")
    original = recipe / "original.jsonl"
    original_report = write_split_inventory(project, root, original)
    # A distinct owned path has identical reviewed bytes but no completion key.
    reviewed = recipe / "reviewed.jsonl"
    reviewed.write_bytes(original.read_bytes())
    persistent = InventoryCache(root)
    assert persistent.completed(project, reviewed) is None
    original_completion = persistent.completed(project, original)
    assert original_completion is not None
    original_entries = {p: p.read_bytes() for p in persistent.directory.glob("*.json")}
    generated = []

    def regenerate(project, root, output, *, completion_cache):
        report = write_split_inventory(
            project, root, output, completion_cache=completion_cache
        )
        completed = completion_cache.completed(project, output)
        assert completed is not None
        assert completed[2]["mode"] == "native_production"
        assert completed[0]["sha256"] == original_report["sha256"]
        assert completion_cache.directory.is_relative_to(output.parent)
        assert completion_cache.proofs.directory.is_relative_to(output.parent)
        assert list(completion_cache.proofs.directory.glob("*.json"))
        generated.append(output)
        if outcome == "producer_error":
            raise OSError("fixture producer failure after completion")
        if outcome == "digest_mismatch":
            return {**report, "sha256": "0" * 64}
        return report

    arguments = (project, root, reviewed, original_report["sha256"], regenerate)
    if outcome == "success":
        report, reused = verified_inventory(*arguments)
        assert reused is False
        assert report["path"] == str(reviewed)
        assert report["sha256"] == original_report["sha256"]
        completion = persistent.completed(project, reviewed)
        assert completion is not None
        assert completion[2]["mode"] == "native_production"
        report_again, reused_again = verified_inventory(*arguments)
        assert reused_again is True
        assert report_again == report
    else:
        error, message = (
            (OSError, "fixture producer failure after completion")
            if outcome == "producer_error"
            else (ValueError, "split inventory differs from verified snapshots")
        )
        with pytest.raises(error, match=message):
            verified_inventory(*arguments)
        assert persistent.completed(project, reviewed) is None
    assert len(generated) == 1
    assert not generated[0].parent.exists()
    assert not list(root.glob(".inventory-check-*"))
    # Existing durable completions survive byte-for-byte; neither completion nor
    # signed proof in the persistent stores may mention the dead scratch path.
    assert all(path.read_bytes() == raw for path, raw in original_entries.items())
    for directory in (persistent.directory, persistent.proofs.directory):
        assert all(
            str(generated[0]).encode() not in path.read_bytes()
            for path in directory.glob("*.json")
        )
