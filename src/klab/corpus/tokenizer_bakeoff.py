"""Frozen Corpus Forge tokenizer selection over disjoint source-document families."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import DatasetConfig, TokenizerTrainConfig
from klab.corpus.release import verify_release
from klab.data.tokenizer import (
    _verify_tokenizer_manifest,
    load_tokenizer,
    train_tokenizer,
    verify_tokenizer_artifact,
)
from klab.training.manifest import sha256_file

GROUPS = ("prose", "python", "go", "rust", "shell", "yaml", "json", "toml", "logs")
VOCABS = (16384, 24576, 32768)
_EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".go": "go",
    ".rs": "rust",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".log": "logs",
    ".txt": "prose",
    ".md": "prose",
    ".rst": "prose",
    ".cnxml": "prose",
    ".xml": "prose",
}


class Declaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2]
    release_path: str
    vocab_sizes: tuple[int, int, int]
    max_fit_bytes: int = Field(gt=0)
    eval_split: Literal["validation"]
    eval_max_docs_per_group: int = Field(gt=0)
    groups: tuple[str, ...]
    near_best_ratio: float = Field(gt=0, le=1)


def load_declaration(path: Path) -> Declaration:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or any(
            type(raw.get(key)) is not expected
            for key, expected in {
                "schema_version": int,
                "release_path": str,
                "vocab_sizes": list,
                "max_fit_bytes": int,
                "eval_split": str,
                "eval_max_docs_per_group": int,
                "groups": list,
                "near_best_ratio": float,
            }.items()
        )
        or any(type(v) is not int for v in raw["vocab_sizes"])
        or any(type(v) is not str for v in raw["groups"])
    ):
        raise ValueError("tokenizer bakeoff declaration has invalid field types")
    spec = Declaration.model_validate(raw)
    if (
        spec.vocab_sizes != VOCABS
        or spec.max_fit_bytes != 268435456
        or spec.eval_max_docs_per_group != 200
        or spec.near_best_ratio != 0.98
        or not spec.release_path.strip()
        or not spec.groups
        or len(set(spec.groups)) != len(spec.groups)
        or tuple(g for g in GROUPS if g in spec.groups) != spec.groups
        or (spec.schema_version == 1 and spec.groups != GROUPS)
    ):
        raise ValueError(
            "tokenizer bakeoff requires approved candidates, ordered groups and budgets"
        )
    return spec


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _group(doc: dict[str, Any]) -> str | None:
    kind = doc["document_kind"].lower()
    if kind in GROUPS:
        return kind
    if kind in {"log", "error", "errors"}:
        return "logs"
    location = doc["source_location"].split("#", 1)[0].lower()
    extension = Path(location).suffix
    if extension in _EXTENSIONS:
        return _EXTENSIONS[extension]
    if kind in {"documentation", "docs", "text", "exposition", "prose"}:
        return "prose"
    return None


def _source_documents(release: Path, split: str):
    selected_ids = {
        link["record_id"] for link in _rows(release / "lm" / f"{split}.lineage.jsonl")
    }
    for doc in _rows(release / "documents.jsonl"):
        if doc["document_id"] in selected_ids and doc["split"] == split:
            yield doc


def _documents(release: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in _source_documents(release, split):
        group = _group(doc)
        if group in GROUPS:
            result[group].append(doc)
    for group, documents in result.items():
        documents.sort(key=lambda doc: (doc["content_sha256"], doc["document_id"]))
    return result


def _selection(release: Path, spec: Declaration):
    train = _documents(release, "train")
    validation = _documents(release, "validation")
    if missing := [g for g in spec.groups if not train[g] or not validation[g]]:
        raise ValueError(
            f"missing required train/validation document groups: {missing}"
        )
    if spec.schema_version == 2:
        unreported = [
            g for g in GROUPS if g not in spec.groups and train[g] and validation[g]
        ]
        if unreported:
            raise ValueError(
                f"measurable source groups omitted from pilot bakeoff: {unreported}"
            )
    train_families = {doc["source_family"] for rows in train.values() for doc in rows}
    val_families = {
        doc["source_family"] for rows in validation.values() for doc in rows
    }
    if train_families & val_families:
        raise ValueError("train and validation source families overlap")
    selected: dict[str, list[dict[str, Any]]] = {}
    heldout: dict[str, list[dict[str, Any]]] = {}
    cap = spec.max_fit_bytes // len(spec.groups)
    if cap <= 0:
        raise ValueError("per-group fitting budget cannot hold a document")
    seen_train: set[str] = set()
    for group in spec.groups:
        chosen = []
        size = 0
        for doc in train[group]:
            digest = doc["content_sha256"]
            if digest in seen_train:
                continue
            byte_size = len(doc["text"].encode("utf-8"))
            if size + byte_size > cap:
                continue
            seen_train.add(digest)
            chosen.append(doc)
            size += byte_size
        if not chosen:
            raise ValueError(
                f"no whole train document fits per-group byte cap: {group}"
            )
        selected[group] = chosen
        heldout[group] = validation[group][: spec.eval_max_docs_per_group]
    train_hashes = {d["content_sha256"] for rows in train.values() for d in rows}
    heldout_hashes = {d["content_sha256"] for rows in heldout.values() for d in rows}
    if train_hashes & heldout_hashes:
        raise ValueError("train and held-out validation share normalized content")
    return selected, heldout, train


def _train_token_counts(
    release: Path,
    train: dict[str, list[dict[str, Any]]],
    schema_version: int,
    model: Any,
) -> dict[str, int]:
    """Count source and ordered-view tokens without retaining duplicate text maps."""
    # Historical v1 counted only its nine approved fit kinds. Pilot v2 must
    # count every selected normalized source, including an unpaired C/code kind.
    documents = (
        _source_documents(release, "train")
        if schema_version == 2
        else (doc for rows in train.values() for doc in rows)
    )
    seen: set[str] = set()
    distinct_tokens = general_tokens = 0
    for doc in documents:
        digest = doc["content_sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        tokens = len(model.encode(doc["text"]).ids)
        distinct_tokens += tokens
        if "general_education" in doc.get("domains", []):
            general_tokens += tokens
    general_by_document = {
        doc["document_id"]: "general_education" in doc.get("domains", [])
        for doc in _rows(release / "documents.jsonl")
    }
    train_view_tokens = general_view_tokens = 0
    for row, link in zip(
        _rows(release / "lm" / "train.jsonl"),
        _rows(release / "lm" / "train.lineage.jsonl"),
        strict=True,
    ):
        tokens = len(model.encode(row["text"]).ids)
        train_view_tokens += tokens
        if general_by_document[link["record_id"]]:
            general_view_tokens += tokens
    return {
        "distinct_normalized_train_tokens": distinct_tokens,
        "total_selected_train_view_tokens": train_view_tokens,
        "general_education_distinct_train_tokens": general_tokens,
        "general_education_train_view_tokens": general_view_tokens,
    }


def choose_candidate(candidates: list[dict[str, Any]], ratio: float = 0.98) -> int:
    best = max(float(item["weighted_bytes_per_token"]) for item in candidates)
    return min(
        int(item["vocab_size"])
        for item in candidates
        if float(item["weighted_bytes_per_token"]) >= ratio * best
    )


def _summary(
    model: Any, docs: list[dict[str, Any]]
) -> tuple[dict[str, Any], Counter[int]]:
    counts: Counter[int] = Counter()
    sizes = []
    total_bytes = 0
    for doc in docs:
        data = doc["text"]
        total_bytes += len(data.encode("utf-8"))
        ids = model.encode(data).ids
        sizes.append(len(ids))
        counts.update(ids)
    sizes.sort()
    total_tokens = sum(sizes)
    return {
        "documents": len(sizes),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
        "tokens_per_document": {
            f"p{int(q * 100)}": sizes[math.ceil(q * len(sizes)) - 1]
            for q in (0.5, 0.95, 0.99)
        },
        "tokens_per_document_max": sizes[-1],
        "unique_vocab_used": len(counts),
        "vocab_utilization": len(counts) / model.get_vocab_size(),
        "tokens_seen_fewer_than_five_times": sum(n < 5 for n in counts.values()),
    }, counts


def _receipt(
    selected: dict[str, list[dict[str, Any]]],
    heldout: dict[str, list[dict[str, Any]]],
    train: dict[str, list[dict[str, Any]]],
    spec: Declaration,
) -> dict[str, Any]:
    def listing(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "document_ids": [row["document_id"] for row in rows],
            "content_sha256": [row["content_sha256"] for row in rows],
            "bytes": sum(len(row["text"].encode("utf-8")) for row in rows),
            "source_family_ids": sorted({row["source_family"] for row in rows}),
        }

    return {
        "fit": {
            g: listing(selected[g])
            | {
                "underfilled": listing(selected[g])["bytes"]
                < spec.max_fit_bytes // len(spec.groups)
            }
            for g in spec.groups
        },
        "heldout": {g: listing(heldout[g]) for g in spec.groups},
        "train": {
            g: {"documents": len(train[g]), "bytes": listing(train[g])["bytes"]}
            for g in spec.groups
        },
    }


def _dataset(
    sample: Path, validation: Path, digest: str, byte_count: int, count: int
) -> DatasetConfig:
    # train_tokenizer only understands a sequential local_text/export iterator. A Corpus
    # Forge export cannot express a hash-ordered, per-kind capped fit: bind the actual
    # immutable sample bytes and release receipt here, rather than mislabel an export.
    return DatasetConfig(
        source="local_text",
        revision=digest,
        cache_dir=sample.parent,
        train_max_documents=count,
        validation_max_documents=1,
        train_max_tokens=max(byte_count, 1),
        validation_max_tokens=1,
        train_path=sample,
        validation_path=validation,
        license="Mixed rights: see bound Corpus Forge release sources.json",
    )


def _write_sample(path: Path, docs: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps({"text": doc["text"]}, ensure_ascii=False) + "\n")


def _binding(
    identity: dict[str, Any],
    receipt: dict[str, Any],
    release: Path,
    sample: Path,
    validation: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_id": identity["release_id"],
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "lm_train_sha256": sha256_file(release / "lm" / "train.jsonl"),
        "lm_validation_sha256": sha256_file(release / "lm" / "validation.jsonl"),
        "declaration_sha256": identity["declaration_sha256"],
        "sample_receipt_sha256": canonical_digest(receipt),
        "fit_sample_sha256": sha256_file(sample),
        "heldout_sample_sha256": sha256_file(validation),
    }


def _bind_manifest(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path.with_name("tokenizer_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpus_forge_bakeoff") not in (None, binding):
        raise ValueError("tokenizer manifest has incompatible Corpus Forge identity")
    manifest["corpus_forge_bakeoff"] = binding
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def _verify_existing(
    output: Path,
    identity: dict[str, Any],
    receipt: dict[str, Any],
    selected: dict[str, list[dict[str, Any]]],
    heldout: dict[str, list[dict[str, Any]]],
    release: Path,
) -> Path:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if (
        report.get("identity") != identity
        or report.get("sample_receipt") != receipt
        or report.get("release_binding")
        != _binding(
            identity,
            receipt,
            release,
            output / "fit.jsonl",
            output / "validation.jsonl",
        )
        or [item["vocab_size"] for item in report.get("candidates", [])] != list(VOCABS)
    ):
        raise ValueError(
            "existing tokenizer bakeoff has mismatched release, declaration or samples"
        )
    sample = output / "fit.jsonl"
    groups = tuple(identity["declaration"]["groups"])
    expected = [{"text": doc["text"]} for group in groups for doc in selected[group]]
    if (
        list(_rows(sample)) != expected
        or sha256_file(sample) != report["fit_sample_sha256"]
    ):
        raise ValueError("existing tokenizer fitting sample has changed")
    validation = output / "validation.jsonl"
    if list(_rows(validation)) != [
        {"text": doc["text"]} for group in groups for doc in heldout[group]
    ]:
        raise ValueError("existing held-out sample has changed")
    binding = _binding(identity, receipt, release, sample, validation)
    dataset = _dataset(
        sample,
        output / "validation.jsonl",
        sha256_file(sample),
        sum(
            len(doc["text"].encode("utf-8"))
            for group in groups
            for doc in selected[group]
        ),
        len(expected),
    )
    weights = {g: receipt["train"][g]["bytes"] for g in groups}
    for candidate in report["candidates"]:
        vocab = candidate["vocab_size"]
        token_path = output / "candidates" / str(vocab) / "tokenizer.json"
        manifest = _verify_tokenizer_manifest(
            token_path,
            source="local_text",
            revision=dataset.revision,
            vocab_size=vocab,
            dataset=dataset,
        )
        if (
            manifest != candidate["manifest"]
            or manifest.get("corpus_forge_bakeoff") != binding
        ):
            raise ValueError(
                "existing tokenizer manifest differs from bound release receipt"
            )
        measured = {
            g: _summary(load_tokenizer(token_path), heldout[g])[0] for g in groups
        }
        score = sum(weights.values()) / sum(
            weights[g] * measured[g]["tokens"] / measured[g]["bytes"] for g in groups
        )
        if (
            candidate["per_kind"] != measured
            or candidate["weighted_bytes_per_token"] != score
            or candidate["tied_embedding_parameters"] != vocab * 768
            or candidate["tokenizer_path"] != str(token_path)
            or candidate["tokenizer_sha256"] != sha256_file(token_path)
            or candidate["manifest_path"]
            != str(token_path.with_name("tokenizer_manifest.json"))
            or candidate["manifest_sha256"]
            != sha256_file(token_path.with_name("tokenizer_manifest.json"))
        ):
            raise ValueError("existing tokenizer measurement or artifact has changed")
    if report["selected_tokenizer"] != str(
        output
        / "candidates"
        / str(
            choose_candidate(
                report["candidates"], identity["declaration"]["near_best_ratio"]
            )
        )
        / "tokenizer.json"
    ):
        raise ValueError("existing tokenizer selection mismatch")
    return output / "report.json"


def bakeoff(declaration: Path, output: Path, *, work_root: Path | None = None) -> Path:
    """Fit candidates to identical bounded train documents, score disjoint validation."""
    declaration = Path(declaration).resolve()
    output = Path(output).resolve()
    spec = load_declaration(declaration)
    if "@" in spec.release_path and not Path(spec.release_path).exists():
        from klab.corpus.cli import _release_path
        from klab.workdir import resolve_work_dir

        release = _release_path(spec.release_path, work_root or resolve_work_dir(None))
    else:
        release = Path(spec.release_path)
        if not release.is_absolute():
            release = (declaration.parent / release).resolve()
    manifest = verify_release(release)
    if spec.schema_version == 2 and manifest["build_identity"]["release"][
        "schema_version"
    ] not in (2, 3):
        raise ValueError("pilot bakeoff requires a prospective rights-tracked release")
    selected, heldout, train = _selection(release, spec)
    receipt = _receipt(selected, heldout, train, spec)
    identity = {
        "schema_version": spec.schema_version,
        "release_path": str(release),
        "release_id": manifest["release_id"],
        "declaration_sha256": sha256_file(declaration),
        "declaration": spec.model_dump(mode="json"),
    }
    if output.exists():
        return _verify_existing(output, identity, receipt, selected, heldout, release)
    output.mkdir(parents=True)
    fit_docs = [doc for group in spec.groups for doc in selected[group]]
    sample = output / "fit.jsonl"
    _write_sample(sample, fit_docs)
    _write_sample(
        output / "validation.jsonl",
        [doc for group in spec.groups for doc in heldout[group]],
    )
    dataset = _dataset(
        sample,
        output / "validation.jsonl",
        sha256_file(sample),
        sum(len(doc["text"].encode("utf-8")) for doc in fit_docs),
        len(fit_docs),
    )
    binding = _binding(identity, receipt, release, sample, output / "validation.jsonl")
    candidates = []
    weights = {g: receipt["train"][g]["bytes"] for g in spec.groups}
    for vocab in spec.vocab_sizes:
        config = TokenizerTrainConfig(
            schema_version=1,
            vocab_size=vocab,
            max_documents=len(fit_docs),
            output_dir=output / "candidates" / str(vocab),
            dataset=dataset,
        )
        path = train_tokenizer(config)
        artifact = verify_tokenizer_artifact(
            path,
            source="local_text",
            revision=dataset.revision,
            vocab_size=vocab,
            dataset=dataset,
        )
        artifact = _bind_manifest(path, binding)
        model = load_tokenizer(path)
        measured = {g: _summary(model, heldout[g])[0] for g in spec.groups}
        tokens = sum(
            weights[g] * measured[g]["tokens"] / measured[g]["bytes"]
            for g in spec.groups
        )
        weighted_bytes_per_token = sum(weights.values()) / tokens
        candidates.append(
            {
                "vocab_size": vocab,
                "tokenizer_path": str(path),
                "tokenizer_sha256": sha256_file(path),
                "manifest_path": str(path.with_name("tokenizer_manifest.json")),
                "manifest_sha256": sha256_file(
                    path.with_name("tokenizer_manifest.json")
                ),
                "manifest": artifact,
                "per_kind": measured,
                "weighted_bytes_per_token": weighted_bytes_per_token,
                "tied_embedding_parameters": vocab * 768,
            }
        )
    chosen = choose_candidate(candidates, spec.near_best_ratio)
    selected_path = output / "candidates" / str(chosen) / "tokenizer.json"
    model = load_tokenizer(selected_path)
    token_counts = _train_token_counts(release, train, spec.schema_version, model)
    unclassified_documents: Counter[str] = Counter()
    unclassified_bytes: Counter[str] = Counter()
    if spec.schema_version == 2:
        for doc in _source_documents(release, "train"):
            if _group(doc) is None:
                kind = doc["document_kind"]
                unclassified_documents[kind] += 1
                unclassified_bytes[kind] += len(doc["text"].encode("utf-8"))
    report = {
        **(
            {
                "unmeasured_source_groups": {
                    g: {
                        "train_documents": len(train[g]),
                        "train_bytes": sum(
                            len(d["text"].encode("utf-8")) for d in train[g]
                        ),
                        "reason": "no paired independent train/validation family",
                    }
                    for g in GROUPS
                    if g not in spec.groups
                },
                "unclassified_source_kinds": {
                    kind: {
                        "train_documents": unclassified_documents[kind],
                        "train_bytes": unclassified_bytes[kind],
                        "reason": "not in the frozen nine tokenizer groups",
                    }
                    for kind in sorted(unclassified_documents)
                },
            }
            if spec.schema_version == 2
            else {}
        ),
        "identity": identity,
        "sample_receipt": receipt,
        "fit_sample_sha256": sha256_file(sample),
        "release_binding": binding,
        "candidates": candidates,
        "selected_tokenizer": str(selected_path),
        "selected_vocab_size": chosen,
        **token_counts,
        "general_education_distinct_train_token_share": token_counts[
            "general_education_distinct_train_tokens"
        ]
        / token_counts["distinct_normalized_train_tokens"],
        "general_education_train_view_token_share": token_counts[
            "general_education_train_view_tokens"
        ]
        / token_counts["total_selected_train_view_tokens"],
    }
    (output / "report.json").write_bytes(canonical_json(report) + b"\n")
    return output / "report.json"
