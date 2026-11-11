"""Native, descriptive continuation/source overlap diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from klab.config.models import StrictModel
from klab.corpus.memorization import SourcePassage, diagnose_memorization
from klab.documents import read_document
from klab.evaluation.panel import verify_panel_result

_SHA256_LENGTH = 64


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_regular_file(path: Path, *, label: str) -> bytes:
    """Read one non-symlink regular file after checking every lexical component."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink: {current}")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular nonsymlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read {label}: {path}") from error


def _resolve_locator(value: str, parent: Path, *, label: str) -> Path:
    candidate = Path(value)
    if not value:
        raise ValueError(f"{label} path must not be empty")
    if ".." in candidate.parts:
        raise ValueError(f"{label} must not traverse parent directories")
    path = candidate if candidate.is_absolute() else parent / candidate
    _checked_regular_file(path, label=label)
    return path.absolute()


class _FileContinuation(StrictModel):
    kind: Literal["file"]
    path: StrictStr
    file_sha256: StrictStr

    @model_validator(mode="after")
    def _validate_digest(self) -> _FileContinuation:
        if len(self.file_sha256) != _SHA256_LENGTH or any(
            char not in "0123456789abcdef" for char in self.file_sha256
        ):
            raise ValueError(
                "continuation.file_sha256 must be a lowercase SHA-256 digest"
            )
        return self


class _PanelContinuation(StrictModel):
    kind: Literal["generation_panel"]
    result: StrictStr
    row: StrictInt = Field(ge=0)


class _Source(StrictModel):
    id: StrictStr
    text: StrictStr

    @model_validator(mode="after")
    def _nonempty_id(self) -> _Source:
        if not self.id:
            raise ValueError("source id must be nonempty")
        return self


class _Declaration(StrictModel):
    format: Literal["klab-memorization-input-v1"]
    continuation: Annotated[
        _FileContinuation | _PanelContinuation, Field(discriminator="kind")
    ]
    sources: list[_Source] = Field(min_length=1)
    ngram_size: StrictInt = Field(default=3, ge=1, le=10)
    max_edit_chars: StrictInt = Field(default=512, ge=1, le=2048)

    @model_validator(mode="after")
    def _unique_source_ids(self) -> _Declaration:
        ids = [source.id for source in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source IDs must be unique")
        return self


def _load_declaration(source: Path) -> tuple[_Declaration, bytes, Path]:
    source = Path(source)
    raw = _checked_regular_file(source, label="memorization declaration")
    try:
        declaration = _Declaration.model_validate(read_document(source))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid memorization declaration: {error}") from error
    return declaration, raw, source.absolute()


def _file_continuation(
    value: _FileContinuation, parent: Path
) -> tuple[str, dict[str, Any]]:
    path = _resolve_locator(value.path, parent, label="continuation file")
    raw = _checked_regular_file(path, label="continuation file")
    actual = _sha256(raw)
    if actual != value.file_sha256:
        raise ValueError("continuation file SHA-256 does not match declaration")
    try:
        continuation = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("continuation file is not valid UTF-8") from error
    return continuation, {
        "kind": "file",
        "path": str(path),
        "file_sha256": actual,
    }


def _panel_continuation(
    value: _PanelContinuation, parent: Path
) -> tuple[str, dict[str, Any]]:
    result_path = _resolve_locator(
        value.result, parent, label="generation panel result"
    )
    record = verify_panel_result(result_path, verification_mode="cold")
    rows = record.get("rows")
    if not isinstance(rows, list) or value.row >= len(rows):
        raise ValueError("generation panel row is outside retained result coverage")
    selected = rows[value.row]
    if not isinstance(selected, dict) or selected.get("status") != "COMPLETED":
        raise ValueError("selected generation panel row is not completed")
    completion = selected.get("completion")
    if not isinstance(completion, str):
        raise TypeError("selected generation panel row has no string completion")
    identity = record.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("generation panel result has no authenticated identity")
    return completion, {
        "kind": "generation_panel",
        "result": str(result_path),
        "result_sha256": record["record_sha256"],
        "panel": record["panel"],
        "panel_sha256": record["panel_sha256"],
        "row": value.row,
        "run": record["run"],
        "run_id": record["run_id"],
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "evaluation_index": record["evaluation_index"],
        "evaluation_index_sha256": record["evaluation_index_sha256"],
        "evaluation_index_file_sha256": record["evaluation_index_file_sha256"],
        "binding_sha256": record["binding_sha256"],
        "identity": identity,
    }


def _recheck_declaration(source: Path, expected: bytes) -> None:
    if _checked_regular_file(source, label="memorization declaration") != expected:
        raise ValueError("memorization declaration changed during analysis")


def analyze_memorization(source: Path) -> dict[str, Any]:
    """Authenticate supplied continuation input and describe its source overlap."""
    declaration, declaration_bytes, declaration_path = _load_declaration(Path(source))
    continuation_spec = declaration.continuation
    if isinstance(continuation_spec, _FileContinuation):
        continuation, origin = _file_continuation(
            continuation_spec, declaration_path.parent
        )
    else:
        continuation, origin = _panel_continuation(
            continuation_spec, declaration_path.parent
        )

    diagnostic = diagnose_memorization(
        continuation,
        [SourcePassage(item.id, item.text) for item in declaration.sources],
        ngram_size=declaration.ngram_size,
        max_edit_chars=declaration.max_edit_chars,
    )

    _recheck_declaration(declaration_path, declaration_bytes)
    if isinstance(continuation_spec, _FileContinuation):
        repeated, repeated_origin = _file_continuation(
            continuation_spec, declaration_path.parent
        )
        if repeated != continuation or repeated_origin != origin:
            raise ValueError("continuation file changed during analysis")
    else:
        repeated, repeated_origin = _panel_continuation(
            continuation_spec, declaration_path.parent
        )
        if repeated != continuation or repeated_origin != origin:
            raise ValueError("generation panel result changed during analysis")

    return {
        "format": "klab-memorization-report-v1",
        "role": "descriptive_not_quality_gate",
        "declaration": {
            "path": str(declaration_path),
            "file_sha256": _sha256(declaration_bytes),
        },
        "origin": origin,
        "continuation_sha256": _sha256(continuation.encode("utf-8")),
        "max_edit_chars": declaration.max_edit_chars,
        **asdict(diagnostic),
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        "Continuation/source overlap (descriptive, not a quality gate)",
        f"declaration: {report['declaration']['path']} ({report['declaration']['file_sha256']})",
        f"origin: {json.dumps(report['origin'], sort_keys=True)}",
        f"continuation_sha256: {report['continuation_sha256']}",
        f"normalization: {report['normalization']}",
        f"ngram_size: {report['ngram_size']}",
        f"max_edit_chars: {report['max_edit_chars']}",
        f"best_source: {report['best_source']['source_id']}",
    ]
    for source in report["sources"]:
        lines.append(
            f"source {source['source_id']}: {json.dumps(source, sort_keys=True)}"
        )
    return "\n".join(lines)


def _handle(args: argparse.Namespace) -> None:
    try:
        report = analyze_memorization(Path(args.source))
    except (OSError, TypeError, ValueError, KeyError) as error:
        if args.json:
            print(json.dumps({"status": "error", "error": str(error)}, allow_nan=False))
        else:
            print(f"klab memorization analyze: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    else:
        print(_render_text(report))


def register_parser(commands: argparse._SubParsersAction) -> None:
    """Register the read-only continuation/source diagnostic."""
    memorization = commands.add_parser(
        "memorization",
        help="Describe continuation/source overlap without policy decisions",
    )
    actions = memorization.add_subparsers(dest="memorization_command", required=True)
    analyze = actions.add_parser("analyze", help="Analyze one supplied continuation")
    analyze.add_argument("source", metavar="INPUT.yaml")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=_handle)
