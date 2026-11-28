"""Read pinned Workshop-style study references and their local Lab evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, urlsplit

from pydantic import Field, StringConstraints, field_validator, model_validator

from klab.config.models import StrictModel
from klab.corpus.project import safe_name
from klab.lab_records import DIGEST_FIELD, read_lab_record
from klab.safe_io import symlinked
from klab.verification_proofs import SMALL_METADATA_LIMIT

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
]
Label = Annotated[str, StringConstraints(min_length=1, max_length=160)]


class StudyFile(StrictModel):
    path: str
    sha256: Digest

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        return safe_name(value)


class StudyResult(StudyFile):
    id: Identifier
    title: Label
    evaluation: Literal["ranking", "answers"]
    split: Literal["development"]
    store: Literal["lab", "runs"]


class StudyDeclaration(StrictModel):
    schema_version: Literal[1]
    name: Label
    repository: str
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    directory: str
    recipe: StudyFile
    results: Annotated[list[StudyResult], Field(min_length=1, max_length=32)]

    @field_validator("directory")
    @classmethod
    def relative_directory(cls, value):
        return safe_name(value)

    @field_validator("repository")
    @classmethod
    def repository_url(cls, value):
        link = urlsplit(value)
        if (
            link.scheme != "https"
            or not link.hostname
            or link.username
            or link.password
            or link.query
            or link.fragment
            or len(link.path.strip("/").split("/")) != 2
        ):
            raise ValueError(
                "repository must be an HTTPS GitHub-style owner/repository URL"
            )
        return value.rstrip("/")

    @model_validator(mode="after")
    def unique_results(self):
        if len({result.id for result in self.results}) != len(self.results):
            raise ValueError("study result IDs must be unique")
        return self

    def blob_url(self, source: StudyFile) -> str:
        return f"{self.repository}/blob/{self.revision}/{quote(self.directory + '/' + source.path)}"


class _RunReference(StrictModel):
    arm: Label
    seed: int
    run_id: Identifier
    checkpoint_sha256: Digest
    try_id: Identifier | None = None
    result_sha256: Digest
    max_new_tokens: Annotated[int, Field(ge=1, le=32)] | None = None


def _small_file(path: Path):
    if symlinked(path):
        raise ValueError("study evidence contains a symbolic link")
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if not path.is_file() or size > SMALL_METADATA_LIMIT:
        raise ValueError("study metadata must be a regular file within 4 MiB")
    return size


def _pinned_bytes(path: Path, expected=None) -> bytes:
    size = _small_file(path)
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size:
        raise ValueError("study metadata changed size while reading")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"study digest differs for {path.name}")
    return raw


def read_study(path: Path) -> StudyDeclaration:
    return StudyDeclaration.model_validate_json(_pinned_bytes(path))


def _references(report, kind):
    if (
        report.get("final_evaluated")
        or report.get("final_evaluation_run")
        or report.get("split", "development") != "development"
    ):
        raise ValueError("the study view accepts development evidence only")
    values = report["runs"]
    if not isinstance(values, list) or not 1 <= len(values) <= 2000:
        raise ValueError("study result must contain 1–2,000 run references")
    references = []
    for row in values:
        ref = _RunReference.model_validate(
            {
                **{
                    name: row[name]
                    for name in ("arm", "seed", "run_id", "checkpoint_sha256")
                },
                "try_id": row.get("try_id"),
                "result_sha256": row[
                    "ranking_result_sha256" if kind == "ranking" else "result_sha256"
                ],
                "max_new_tokens": row.get("max_new_tokens"),
            }
        )
        if (kind == "answers") != (ref.max_new_tokens is not None):
            raise ValueError("generated answers require a recorded token budget")
        references.append(ref)
    if len({(ref.run_id, ref.max_new_tokens) for ref in references}) != len(references):
        raise ValueError("study repeats the same run and evaluation budget")
    return references


def _evaluation(runs_dir, reference, kind, panel, tokenizer):
    path = (
        runs_dir
        / reference.run_id
        / "evaluations"
        / f"{kind}-{reference.result_sha256}.json"
    )
    result = json.loads(_pinned_bytes(path, reference.result_sha256))
    identity = result["identity"]
    evaluation = result[kind]
    if (
        result["source"] != f"standalone_{kind}"
        or identity["run_id"] != reference.run_id
        or identity["checkpoint_sha256"] != reference.checkpoint_sha256
        or identity["tokenizer_sha256"] != tokenizer
        or evaluation["sha256"] != panel
        or (
            kind == "answers"
            and evaluation["max_new_tokens"] != reference.max_new_tokens
        )
    ):
        raise ValueError(
            "evaluation does not match the pinned run, checkpoint, tokenizer and panel"
        )
    overall = evaluation["overall"]
    if kind == "answers":
        metrics = {
            name: overall[name]
            for name in ("items", "correct", "valid_form", "length_limited")
        }
    else:
        metrics = {name: overall[name] for name in ("items", "accuracy", "chance")}
        if evaluation.get("pairs"):
            metrics["joint_accuracy"] = evaluation["pairs"]["overall"]["joint_accuracy"]
    return {
        "protocol": evaluation["protocol"],
        "panel_sha256": panel,
        "tokenizer_sha256": tokenizer,
        **metrics,
    }


def _try_binding(lab_dir, reference, cache):
    if reference.try_id is None:
        return None
    try_id = reference.try_id
    if try_id not in cache:
        path = lab_dir / "tries" / try_id / "try.json"
        _small_file(path)
        _, record = read_lab_record(path, "try")
        if record["try_id"] != try_id:
            raise ValueError("Lab try identity differs from the study reference")
        candidates = record["arms"]
        cache[try_id] = (
            record[DIGEST_FIELD],
            {(arm["run_id"], arm["checkpoint_sha256"]) for arm in candidates.values()},
        )
    digest, arms = cache[try_id]
    if (reference.run_id, reference.checkpoint_sha256) not in arms:
        raise ValueError("Lab try does not bind this run and checkpoint")
    return digest


def study_page(
    path: Path,
    lab_dir: Path,
    runs_dir: Path,
    *,
    result_id=None,
    limit=20,
    offset=0,
    budget=None,
):
    """Verify only the selected page of a pinned development study."""
    if not 1 <= limit <= 20 or offset < 0:
        raise ValueError("study requires a nonnegative offset and a limit of 1–20")
    study = read_study(path)
    _pinned_bytes(path.parent / study.recipe.path, study.recipe.sha256)
    target = (
        next((item for item in study.results if item.id == result_id), None)
        if result_id
        else study.results[0]
    )
    if target is None:
        raise ValueError(f"unknown study result: {result_id}")
    report = json.loads(_pinned_bytes(path.parent / target.path, target.sha256))
    refs = _references(report, target.evaluation)
    tokenizer = report["tokenizer_sha256"]
    budgets = sorted(
        {row.max_new_tokens for row in refs if row.max_new_tokens is not None}
    )
    if budget is not None and budget not in budgets:
        raise ValueError("study result does not record this completion budget")
    if budget is not None:
        refs = [row for row in refs if row.max_new_tokens == budget]
    rows, cache = [], {}
    source = lab_dir / "runs" if target.store == "lab" else runs_dir
    for ref in refs[offset : offset + limit]:
        row = ref.model_dump()
        try:
            panel = (
                report["budgets"][str(ref.max_new_tokens)]["sha256"]
                if target.evaluation == "answers"
                else report["score_panel_sha256"]
            )
            metrics = _evaluation(source, ref, target.evaluation, panel, tokenizer)
            try_digest = _try_binding(lab_dir, ref, cache)
        except FileNotFoundError as error:
            row.update(status="unavailable", reason=str(error))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="rejected", reason=str(error))
        else:
            row.update(status="verified", metrics=metrics, try_record_sha256=try_digest)
        rows.append(row)
    return {
        "name": study.name,
        "recipe": {**study.recipe.model_dump(), "url": study.blob_url(study.recipe)},
        "result": {**target.model_dump(), "url": study.blob_url(target)},
        "budgets": budgets,
        "run_store": str(source),
        "rows": rows,
        "row_count": len(refs),
        "next_offset": offset + len(rows) if offset + len(rows) < len(refs) else None,
    }
