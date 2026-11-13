"""Convert pinned external rows into local_chat or local_text JSONL with a receipt.

Public API: load_declaration -> ConversionDeclaration, convert_dataset -> output
Path, verify_conversion -> receipt dict.

A declaration names a pinned Hugging Face revision (or local JSONL/Parquet
files), a license, a train/validation selection and one mapping from columns to
chat roles or to a text template. Conversion reads every selected row in source
order, records each excluded row with its reason, drops exact repeats within a
split and refuses a conversation that would appear in both splits. Verification
re-reads the pinned source and requires the same bytes, counts and exclusions.
Templates have a fixed grammar: ``{column}``, ``{unified_diff(old, new)}`` or
``{unified_diff(old, new, old_path, new_path)}``, and ``{{``/``}}`` for braces.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from klab.canonical import (
    canonical_digest,
    canonical_json,
    domain_digest,
    reject_duplicate_keys,
)
from klab.documents import read_document
from klab.engram.packs import _rename_noreplace
from klab.result_publication import read_canonical, sync_directory
from klab.training.manifest import sha256_file, source_identity

FORMAT = "klab-dataset-conversion-v1"
RECEIPT = "receipt.json"
OUTPUTS = ("train", "validation")
_SPLIT_DOMAIN = "klab-conversion-split-v1"
_COLUMN = r"[A-Za-z_][A-Za-z0-9_]*"
_POLICY = {
    "within_split_duplicate": "exclude",
    "cross_split_overlap": "refuse",
    "key": "whole output record, text whitespace-collapsed, roles kept",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HubSource(_Model):
    kind: Literal["huggingface"]
    repo_id: str
    revision: str
    config: str = "default"

    @model_validator(mode="after")
    def pinned(self) -> HubSource:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision):
            raise ValueError("revision must be an immutable lowercase commit hash")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo_id):
            raise ValueError("repo_id must be owner/repository")
        if not self.config.strip():
            raise ValueError("config must be nonblank")
        return self


class LocalSource(_Model):
    kind: Literal["local"]
    # Upstream split name -> a .jsonl or .parquet file, relative to the declaration.
    files: dict[str, Path] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def formats(cls, files: dict[str, Path]) -> dict[str, Path]:
        if any(path.suffix not in {".jsonl", ".parquet"} for path in files.values()):
            raise ValueError("local source files must be .jsonl or .parquet")
        return files


class SourceSplits(_Model):
    """Use the source's own splits as train and validation."""

    mode: Literal["source"]
    train: str
    validation: str


class HashSplits(_Model):
    """Split one source split by a digest of the declared key columns."""

    mode: Literal["hash"]
    split: str
    key: tuple[Annotated[str, Field(pattern=f"^{_COLUMN}$")], ...] = Field(min_length=1)
    validation_fraction: float = Field(gt=0, lt=1)


Template = Annotated[str, Field(min_length=1)]


class ChatMapping(_Model):
    """A column holding a list of messages, each with a role and content."""

    kind: Literal["chat"]
    messages_field: str
    role_field: str = "role"
    content_field: str = "content"
    role_map: dict[str, Literal["system", "user", "assistant"]] = Field(
        default_factory=lambda: {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }
    )
    loss_mode: Literal["assistant_only", "all_tokens"]


class InstructionMapping(_Model):
    """One prompt and one response, each rendered from a template."""

    kind: Literal["instruction"]
    prompt: Template
    response: Template
    system: Template | None = None
    loss_mode: Literal["assistant_only", "all_tokens"]

    @model_validator(mode="after")
    def templates(self) -> InstructionMapping:
        for value in (self.prompt, self.response, self.system):
            if value is not None:
                _parse(value)
        return self


class TextMapping(_Model):
    """One text document rendered from a template."""

    kind: Literal["text"]
    template: Template

    @model_validator(mode="after")
    def parsed(self) -> TextMapping:
        _parse(self.template)
        return self


class Limits(_Model):
    # Rows read per source split; reading stops there and the receipt says so.
    max_rows: int = Field(gt=0, strict=True)
    # An output JSONL line above this many bytes is excluded as oversized.
    max_record_bytes: int = Field(gt=0, strict=True)


class ConversionDeclaration(_Model):
    schema_version: Literal[1] = 1
    source: Annotated[HubSource | LocalSource, Field(discriminator="kind")]
    license: str
    attribution: str
    splits: Annotated[SourceSplits | HashSplits, Field(discriminator="mode")]
    mapping: Annotated[
        ChatMapping | InstructionMapping | TextMapping, Field(discriminator="kind")
    ]
    limits: Limits

    @model_validator(mode="after")
    def contract(self) -> ConversionDeclaration:
        if not self.license.strip() or not self.attribution.strip():
            raise ValueError("license and attribution must be nonblank")
        upstream = self.upstream_splits()
        if len(set(upstream)) != len(upstream) or any(
            not name.strip() for name in upstream
        ):
            raise ValueError("train and validation must name distinct source splits")
        if isinstance(self.source, LocalSource) and set(self.source.files) != set(
            upstream
        ):
            raise ValueError("local files must name exactly the selected splits")
        return self

    def upstream_splits(self) -> tuple[str, ...]:
        if isinstance(self.splits, SourceSplits):
            return (self.splits.train, self.splits.validation)
        return (self.splits.split,)

    @property
    def output_kind(self) -> str:
        return "local_text" if isinstance(self.mapping, TextMapping) else "local_chat"


def load_declaration(path: str | Path) -> tuple[ConversionDeclaration, dict | None]:
    """Read a strict declaration; local file locations resolve beside it."""
    path = Path(path)
    declaration = ConversionDeclaration.model_validate(read_document(path))
    locations = None
    if isinstance(declaration.source, LocalSource):
        locations = {
            split: str((path.parent / file).resolve())
            for split, file in declaration.source.files.items()
        }
    return declaration, locations


# --- Templates -------------------------------------------------------------


class _Excluded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[{}]")
_DIFF = re.compile(rf"unified_diff\(\s*({_COLUMN}(?:\s*,\s*{_COLUMN})*)\s*\)")


@cache
def _parse(template: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Literal and placeholder parts: ("text", (s,)), ("field", (c,)), ("diff", args)."""
    parts: list[tuple[str, tuple[str, ...]]] = []
    position = 0
    for match in _TOKEN.finditer(template):
        if match.start() > position:
            parts.append(("text", (template[position : match.start()],)))
        position = match.end()
        token = match.group(0)
        if token in {"{{", "}}"}:
            parts.append(("text", (token[0],)))
            continue
        inner = match.group(1)
        if inner is None:
            raise ValueError("unbalanced brace in template; write {{ or }}")
        inner = inner.strip()
        if re.fullmatch(_COLUMN, inner):
            parts.append(("field", (inner,)))
            continue
        diff = _DIFF.fullmatch(inner)
        if diff is None:
            raise ValueError(f"unsupported template placeholder: {{{inner}}}")
        args = tuple(arg.strip() for arg in diff.group(1).split(","))
        if len(args) not in {2, 4}:
            raise ValueError(
                "unified_diff takes (old, new) or (old, new, old_path, new_path)"
            )
        parts.append(("diff", args))
    if position < len(template):
        parts.append(("text", (template[position:],)))
    return tuple(parts)


def _string(row: dict, column: str) -> str:
    if column not in row:
        raise _Excluded("missing_field")
    value = row[column]
    if value is None:
        raise _Excluded("null_field")
    if not isinstance(value, str):
        raise _Excluded("malformed")
    return value


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def unified_diff(old: str, new: str, old_label: str = "a", new_label: str = "b") -> str:
    """A unified diff with three lines of context, split on newlines only."""
    rendered = []
    for line in difflib.unified_diff(_lines(old), _lines(new), old_label, new_label):
        rendered.append(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        )
    return "".join(rendered)


def _render(template: str, row: dict) -> str:
    out = []
    for kind, args in _parse(template):
        if kind == "text":
            out.append(args[0])
        elif kind == "field":
            out.append(_string(row, args[0]))
        else:
            values = [_string(row, arg) for arg in args]
            labels = ("a/" + values[2], "b/" + values[3]) if len(values) == 4 else ()
            diff = unified_diff(values[0], values[1], *labels)
            if not diff:
                raise _Excluded("empty_diff")
            out.append(diff)
    return "".join(out)


# --- Rows to records -------------------------------------------------------


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _chat_record(
    loss_mode: str, messages: list[dict[str, str]], origin: str
) -> tuple[dict, str]:
    from klab.data.conversations import render_conversation

    if any(not message["content"].strip() for message in messages):
        raise _Excluded("empty")
    record = {"format_version": 2, "loss_mode": loss_mode, "messages": messages}
    try:
        render_conversation(record, origin)
    except ValueError:
        raise _Excluded("malformed") from None
    key = canonical_json(
        [[message["role"], _normalized(message["content"])] for message in messages]
    ).decode()
    return record, key


def _record(mapping, row: object, origin: str) -> tuple[dict, str]:
    """The output record for one source row and its cross-split identity text."""
    if not isinstance(row, dict):
        raise _Excluded("malformed")
    if isinstance(mapping, TextMapping):
        text = _render(mapping.template, row)
        if not text.strip():
            raise _Excluded("empty")
        return {"text": text}, _normalized(text)
    if isinstance(mapping, InstructionMapping):
        messages = []
        if mapping.system is not None:
            system = _render(mapping.system, row).strip()
            if system:
                messages.append({"role": "system", "content": system})
        messages.append(
            {"role": "user", "content": _render(mapping.prompt, row).strip()}
        )
        messages.append(
            {"role": "assistant", "content": _render(mapping.response, row).strip()}
        )
        return _chat_record(mapping.loss_mode, messages, origin)
    if mapping.messages_field not in row:
        raise _Excluded("missing_field")
    source = row[mapping.messages_field]
    if source is None:
        raise _Excluded("null_field")
    if not isinstance(source, list) or not source:
        raise _Excluded("malformed")
    messages = []
    for message in source:
        if not isinstance(message, dict):
            raise _Excluded("malformed")
        role = message.get(mapping.role_field)
        content = message.get(mapping.content_field)
        if not isinstance(role, str) or not isinstance(content, str):
            raise _Excluded("malformed")
        if role not in mapping.role_map:
            raise _Excluded("unmapped_role")
        messages.append({"role": mapping.role_map[role], "content": content})
    return _chat_record(mapping.loss_mode, messages, origin)


def _hash_split(splits: HashSplits, row: object) -> str:
    if not isinstance(row, dict):
        raise _Excluded("malformed")
    if any(column not in row for column in splits.key):
        raise _Excluded("missing_field")
    try:
        digest = domain_digest(_SPLIT_DOMAIN, [row[column] for column in splits.key])
    except TypeError, ValueError:
        raise _Excluded("malformed") from None
    position = int(digest[:16], 16) / 2**64
    return "validation" if position < splits.validation_fraction else "train"


# --- Sources ---------------------------------------------------------------


def _local_rows(path: Path) -> Iterator[object]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
                yield from batch.to_pylist()
        return
    with path.open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                yield json.loads(
                    line,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"nonfinite JSON value: {value}")
                    ),
                )
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: invalid JSONL row") from error


def _local_identity(locations: dict) -> dict:
    files = {}
    for split, location in sorted(locations.items()):
        path = Path(location)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"local source file is missing or a link: {path}")
        files[split] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"kind": "local", "files": files}


def _open_source(
    declaration: ConversionDeclaration, locations: dict | None, cache_dir: Path
) -> tuple[dict, dict[str, Iterator[object]]]:
    if isinstance(declaration.source, LocalSource):
        if locations is None or set(locations) != set(declaration.upstream_splits()):
            raise ValueError("local conversion needs a location for every split")
        identity = _local_identity(locations)
        return identity, {
            split: _local_rows(Path(locations[split])) for split in locations
        }
    from klab.data.sources import hub_identity, stream_hub

    source = declaration.source
    upstream = declaration.upstream_splits()
    provenance = hub_identity(
        source.repo_id,
        source.revision,
        source.config,
        {split: split for split in upstream},
        cache_dir=cache_dir,
    )
    streams = {
        split: stream_hub(source.repo_id, source.revision, provenance, split, cache_dir)
        for split in upstream
    }
    return {"kind": "huggingface", "provenance": provenance}, streams


# --- Conversion ------------------------------------------------------------


def _write(handle, payload: dict) -> int:
    line = canonical_json(payload) + b"\n"
    handle.write(line)
    return len(line)


def _convert(
    declaration: ConversionDeclaration,
    locations: dict | None,
    cache_dir: Path,
    work: Path,
) -> dict:
    """Write the split and exclusion files into ``work``; return the receipt body."""
    identity, streams = _open_source(declaration, locations, cache_dir)
    seen: dict[bytes, str] = {}
    reads: dict[str, dict] = {}
    counts = {name: {"rows": 0, "bytes": 0} for name in OUTPUTS}
    excluded_rows = 0
    handles = {name: (work / f"{name}.jsonl").open("xb") for name in OUTPUTS}
    excluded = (work / "excluded.jsonl").open("xb")
    try:
        for index, upstream in enumerate(declaration.upstream_splits()):
            fixed = (
                OUTPUTS[index] if isinstance(declaration.splits, SourceSplits) else None
            )
            read = {"rows_read": 0, "source_exhausted": True, "excluded": {}}
            for ordinal, row in enumerate(streams[upstream]):
                if ordinal >= declaration.limits.max_rows:
                    read["source_exhausted"] = False
                    break
                read["rows_read"] += 1
                try:
                    split = fixed or _hash_split(declaration.splits, row)
                    record, key = _record(
                        declaration.mapping, row, f"{upstream}:{ordinal}"
                    )
                    line = canonical_json(record) + b"\n"
                    if len(line) > declaration.limits.max_record_bytes:
                        raise _Excluded("oversized")
                    digest = hashlib.sha256(key.encode()).digest()
                    prior = seen.get(digest)
                    if prior is not None and prior != split:
                        raise ValueError(
                            f"{upstream} row {ordinal} repeats a conversation already "
                            f"in {prior}; a record cannot be in both train and "
                            "validation"
                        )
                    if prior is not None:
                        raise _Excluded("duplicate")
                except _Excluded as exclusion:
                    reasons = read["excluded"]
                    reasons[exclusion.reason] = reasons.get(exclusion.reason, 0) + 1
                    _write(
                        excluded,
                        {
                            "source_split": upstream,
                            "ordinal": ordinal,
                            "reason": exclusion.reason,
                        },
                    )
                    excluded_rows += 1
                    continue
                seen[digest] = split
                handles[split].write(line)
                counts[split]["rows"] += 1
                counts[split]["bytes"] += len(line)
            close = getattr(streams[upstream], "close", None)
            if close is not None:
                close()
            read["excluded"] = dict(sorted(read["excluded"].items()))
            reads[upstream] = read
    finally:
        for handle in (*handles.values(), excluded):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    for name in OUTPUTS:
        if not counts[name]["rows"]:
            raise ValueError(f"conversion selected no {name} rows")
    if identity["kind"] == "local" and _local_identity(locations) != identity:
        raise ValueError("local source files changed while they were read")
    outputs = {
        name: {
            "path": f"{name}.jsonl",
            "rows": counts[name]["rows"],
            "bytes": counts[name]["bytes"],
            "sha256": sha256_file(work / f"{name}.jsonl"),
        }
        for name in OUTPUTS
    }
    outputs["excluded"] = {
        "path": "excluded.jsonl",
        "rows": excluded_rows,
        "sha256": sha256_file(work / "excluded.jsonl"),
    }
    payload = declaration.model_dump(mode="json")
    return {
        "format": FORMAT,
        "declaration": payload,
        "declaration_sha256": canonical_digest(payload),
        "mapping_sha256": canonical_digest(payload["mapping"]),
        "license": declaration.license,
        "output_kind": declaration.output_kind,
        "source": identity,
        "reads": reads,
        "outputs": outputs,
        "policy": _POLICY,
    }


def _seal(body: dict) -> dict:
    return {**body, "receipt_sha256": canonical_digest(body)}


def convert_dataset(
    declaration_path: str | Path, output: str | Path, cache_dir: str | Path
) -> Path:
    """Publish train.jsonl, validation.jsonl, excluded.jsonl and receipt.json."""
    declaration, locations = load_declaration(declaration_path)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(f".{output.name}.work")
    try:
        work.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            error.errno,
            "conversion work directory exists; another conversion may be running, "
            "or an earlier one stopped (inspect, then remove it)",
            error.filename,
        ) from error
    try:
        body = _convert(declaration, locations, Path(cache_dir), work)
        if locations is not None:
            body["locations"] = locations
        body["lab_source_sha256"] = source_identity()["sha256"]
        with (work / RECEIPT).open("xb") as handle:
            handle.write(canonical_json(_seal(body)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(work)
        _rename_noreplace(work, output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    sync_directory(output.parent)
    return output


def verify_conversion(directory: str | Path, cache_dir: str | Path) -> dict:
    """Check output digests, then re-read the pinned source and convert again."""
    directory = Path(directory)
    receipt = read_canonical(directory / RECEIPT)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("format") != FORMAT or receipt.get(
        "receipt_sha256"
    ) != canonical_digest(body):
        raise ValueError("unsupported or changed conversion receipt")
    outputs = receipt["outputs"]
    names = {RECEIPT, *(entry["path"] for entry in outputs.values())}
    if {path.name for path in directory.iterdir()} != names:
        raise ValueError("conversion output inventory differs from its receipt")
    for name, entry in outputs.items():
        path = directory / entry["path"]
        if path.is_symlink() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"conversion output changed: {name}")
    declaration = ConversionDeclaration.model_validate(receipt["declaration"])
    with tempfile.TemporaryDirectory(prefix="klab-conversion-verify-") as temporary:
        replayed = _convert(
            declaration, receipt.get("locations"), Path(cache_dir), Path(temporary)
        )
    differences = sorted(key for key in replayed if replayed[key] != receipt.get(key))
    if differences:
        raise ValueError(
            "re-reading the pinned source gives a different conversion: "
            + ", ".join(differences)
        )
    return receipt
