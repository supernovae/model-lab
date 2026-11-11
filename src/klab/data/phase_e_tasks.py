"""Deterministic, provenance-bound Phase E task fixtures.

The public builders intentionally create small educational fixtures.  They never
execute supplied Python, download data during import, or embed Wikidata answers
in package resources.
"""

from __future__ import annotations

import hashlib
import html
import importlib.resources
import json
import os
import posixpath
import random
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from klab.canonical import reject_duplicate_keys
from klab.engram.packs import _rename_noreplace

_TASK_KINDS: Final[frozenset[str]] = frozenset(
    {
        "math",
        "api",
        "factual_recall",
        "paraphrase",
        "composition",
        "long_context",
        "instruction",
        "conversation",
        "stale_source",
        "conflicting_source",
        "missing_source",
        "application",
    }
)
_SOURCE_FORMAT: Final = "klab-wikidata-mini-source"
_OUTPUT_FORMAT: Final = "klab-wikidata-mini"
_VERSION: Final = 1
_MAX_RESPONSE_BYTES: Final = 8 * 1024 * 1024
_HTTP_TIMEOUT_SECONDS: Final = 15
_USER_AGENT: Final = "Kybern-Model-Lab/0.1 Phase-E-Wikidata-Mini (+https://github.com/supernovae/model-lab)"
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,79}$")


@dataclass(frozen=True)
class TaskCase:
    """An immutable prompt/answer case with a narrowly declared task kind."""

    identifier: str
    prompt: str
    expected: str
    kind: str

    def __post_init__(self) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(self.identifier):
            raise ValueError(f"unsafe task identifier: {self.identifier!r}")
        if self.kind not in _TASK_KINDS:
            raise ValueError(f"unsupported task kind: {self.kind!r}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("task prompt must be a nonblank string")
        if not isinstance(self.expected, str) or not self.expected.strip():
            raise ValueError("task expected value must be a nonblank string")


def _shuffled(
    values: Iterable[tuple[int, int]], seed: int
) -> tuple[tuple[int, int], ...]:
    result = list(values)
    random.Random(seed).shuffle(result)
    return tuple(result)


def _math_examples(seed: int, *, split: str) -> tuple[TaskCase, ...]:
    # The pools deliberately do not even share individual operands.  This makes
    # the novel-operand boundary easy to audit and avoids accidental examples
    # that differ only by their pairing.
    values_by_split = {
        "train": range(2, 18),
        "validation": range(51, 67),
        "test": range(101, 117),
    }
    values = values_by_split[split]
    pairs = tuple((left, right) for left, right in zip(values[::2], values[1::2]))
    cases: list[TaskCase] = []
    for index, (left, right) in enumerate(_shuffled(pairs, seed), start=1):
        expected = left * left + 2 * left * right + right * right
        cases.append(
            TaskCase(
                identifier=f"math-{split}-{index}",
                prompt=(
                    f"Use (x + y)^2 = x^2 + 2xy + y^2 to calculate ({left} + {right})^2."
                ),
                expected=str(expected),
                kind="math",
            )
        )
    return tuple(cases)


def _render_documents(cases: Iterable[TaskCase]) -> tuple[str, ...]:
    return tuple(f"User: {case.prompt}\nAssistant: {case.expected}" for case in cases)


def math_identity_training_cases(seed: int = 17) -> tuple[TaskCase, ...]:
    """Return structured training-only algebra examples."""
    return _math_examples(seed, split="train")


def math_identity_training_documents(seed: int = 17) -> tuple[str, ...]:
    """Render the structured training algebra cases as plain documents."""
    return _render_documents(math_identity_training_cases(seed))


def math_identity_validation_cases(seed: int = 17) -> tuple[TaskCase, ...]:
    """Return validation algebra examples disjoint from training and test."""
    return _math_examples(seed, split="validation")


def math_identity_validation_documents(seed: int = 17) -> tuple[str, ...]:
    """Render the structured validation algebra cases as plain documents."""
    return _render_documents(math_identity_validation_cases(seed))


def math_identity_test_cases(seed: int = 17) -> tuple[TaskCase, ...]:
    """Return algebra cases whose operands never occur in training or validation."""
    return _math_examples(seed, split="test")


@dataclass(frozen=True)
class _SafeStdlibOperation:
    name: str
    render: Callable[[str], str]
    call: Callable[[str], str]


# This is a finite, explicit allowlist.  Expected values are produced by direct
# calls here—not source parsing, eval, exec, imports selected by user input, or
# project code execution.
_SAFE_STDLIB_OPERATIONS: Final[tuple[_SafeStdlibOperation, ...]] = (
    _SafeStdlibOperation(
        "html.escape",
        lambda value: f"What does html.escape({value!r}, quote=True) return?",
        lambda value: html.escape(value, quote=True),
    ),
    _SafeStdlibOperation(
        "urllib.parse.quote",
        lambda value: f"What does urllib.parse.quote({value!r}, safe='') return?",
        lambda value: urllib.parse.quote(value, safe=""),
    ),
    _SafeStdlibOperation(
        "posixpath.normpath",
        lambda value: f"What does posixpath.normpath({value!r}) return?",
        lambda value: posixpath.normpath(value),
    ),
)


def _python_examples(seed: int, *, split: str) -> tuple[TaskCase, ...]:
    # Values are fixed bounded literals and disjoint by split.  Expected values
    # are derived only via the direct calls in _SAFE_STDLIB_OPERATIONS.
    values_by_split = {
        "train": ("a & b", "folder/./child/..", "two words"),
        "validation": ('"quoted"', "one/./two/../three", "semi;colon"),
        "test": ("<tag 'x'>", "alpha/../beta//gamma", "slash/value"),
    }
    operations = list(_SAFE_STDLIB_OPERATIONS)
    random.Random(seed).shuffle(operations)
    cases: list[TaskCase] = []
    for index, (operation, value) in enumerate(
        zip(operations, values_by_split[split]), start=1
    ):
        cases.append(
            TaskCase(
                identifier=f"api-{split}-{index}",
                prompt=operation.render(value),
                expected=operation.call(value),
                kind="api",
            )
        )
    return tuple(cases)


def python_stdlib_training_cases(seed: int = 17) -> tuple[TaskCase, ...]:
    """Return structured examples made by the finite trusted stdlib allowlist."""
    return _python_examples(seed, split="train")


def python_stdlib_training_documents(seed: int = 17) -> tuple[str, ...]:
    """Render structured safe-API training cases as plain documents."""
    return _render_documents(python_stdlib_training_cases(seed))


def python_stdlib_validation_cases(seed: int = 17) -> tuple[TaskCase, ...]:
    """Return safe-API validation cases with values absent from other splits."""
    return _python_examples(seed, split="validation")


def python_stdlib_validation_documents(seed: int = 17) -> tuple[str, ...]:
    """Render structured safe-API validation cases as plain documents."""
    return _render_documents(python_stdlib_validation_cases(seed))


def python_stdlib_test_cases(seed: int = 17) -> tuple[TaskCase, ...]:
    """Return held-out safe-API cases without interpreting any source string."""
    return _python_examples(seed, split="test")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(content: bytes, description: str) -> object:
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} JSON") from error


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resource_manifest_path() -> Any:
    return importlib.resources.files("klab.data").joinpath(
        "resources/wikidata_mini_v1.json"
    )


def wikidata_mini_spec() -> dict[str, object]:
    """Load and strictly validate the answer-free packaged Wikidata source manifest."""
    raw = _resource_manifest_path().read_bytes()
    value = _strict_json_bytes(raw, "Wikidata source manifest")
    if not isinstance(value, dict) or set(value) != {
        "format",
        "version",
        "license",
        "source",
        "entities",
    }:
        raise ValueError("Wikidata source manifest has invalid fields")
    if (
        value["format"] != _SOURCE_FORMAT
        or type(value["version"]) is not int
        or value["version"] != _VERSION
    ):
        raise ValueError("unsupported Wikidata source manifest version")
    if (
        value["license"] != "CC0-1.0"
        or value["source"] != "Wikidata structured entity data"
    ):
        raise ValueError("Wikidata source manifest has invalid provenance")
    entities = value["entities"]
    if not isinstance(entities, list) or len(entities) != 4:
        raise ValueError("Wikidata source manifest must list exactly four entities")
    checked: list[dict[str, object]] = []
    ids: set[str] = set()
    splits: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {"id", "revision", "split"}:
            raise ValueError("Wikidata source entity has invalid fields")
        qid, revision, split = entity["id"], entity["revision"], entity["split"]
        if not isinstance(qid, str) or not re.fullmatch(r"Q[1-9][0-9]*", qid):
            raise ValueError("Wikidata source entity has invalid ID")
        if type(revision) is not int or revision <= 0:
            raise ValueError("Wikidata source entity has invalid revision")
        if split not in splits or qid in ids:
            raise ValueError(
                "Wikidata source entities have duplicate IDs or invalid splits"
            )
        ids.add(qid)
        splits[split] += 1
        checked.append({"id": qid, "revision": revision, "split": split})
    if splits != {"train": 2, "validation": 1, "test": 1}:
        raise ValueError("Wikidata source manifest has invalid split membership")
    return {
        "format": _SOURCE_FORMAT,
        "version": _VERSION,
        "license": "CC0-1.0",
        "source": "Wikidata structured entity data",
        "entities": checked,
    }


def _entity_url(qid: str, revision: int) -> str:
    return f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json?revision={revision}"


def _fetch_entity(qid: str, revision: int) -> dict[str, object]:
    """Fetch exactly one pinned entity with one bounded request and no retry loop."""
    request = urllib.request.Request(
        _entity_url(qid, revision),
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            if length is not None and (
                not length.isdigit() or int(length) > _MAX_RESPONSE_BYTES
            ):
                raise ValueError(
                    f"Wikidata response exceeds {_MAX_RESPONSE_BYTES} byte limit"
                )
            content = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code == 429:
            raise RuntimeError(
                "Wikidata rate limited request; no retry was attempted"
            ) from error
        raise RuntimeError(
            f"Wikidata request failed for {qid}: HTTP {error.code}"
        ) from error
    except (OSError, TimeoutError) as error:
        raise RuntimeError(f"Wikidata request failed for {qid}") from error
    if len(content) > _MAX_RESPONSE_BYTES:
        raise ValueError(f"Wikidata response exceeds {_MAX_RESPONSE_BYTES} byte limit")
    value = _strict_json_bytes(content, f"Wikidata entity {qid}")
    if not isinstance(value, dict):
        raise TypeError(f"Wikidata entity {qid} response must be an object")
    return value


def _source_label(entity: Mapping[str, object], qid: str) -> tuple[str, str]:
    labels = entity.get("labels")
    if not isinstance(labels, dict):
        raise TypeError(f"Wikidata entity {qid} labels must be an object")
    for language in ("en", "mul"):
        source_label = labels.get(language)
        if source_label is None:
            continue
        if (
            not isinstance(source_label, dict)
            or set(source_label) != {"language", "value"}
            or source_label["language"] != language
        ):
            raise ValueError(f"Wikidata entity {qid} has invalid {language} label")
        label = source_label["value"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Wikidata entity {qid} has no usable {language} label")
        return label.strip(), language
    raise ValueError(f"Wikidata entity {qid} has no English or language-neutral label")


def _date_claim(entity: Mapping[str, object], property_id: str, qid: str) -> str | None:
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        raise TypeError(f"Wikidata entity {qid} claims must be an object")
    candidates = claims.get(property_id, [])
    if not isinstance(candidates, list):
        raise TypeError(f"Wikidata entity {qid} {property_id} claims must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        mainsnak = candidate.get("mainsnak")
        if not isinstance(mainsnak, dict) or mainsnak.get("snaktype") != "value":
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, dict) or datavalue.get("type") != "time":
            continue
        time = datavalue.get("value")
        if not isinstance(time, dict):
            continue
        raw = time.get("time")
        precision = time.get("precision")
        if not isinstance(raw, str) or type(precision) is not int:
            continue
        match = re.fullmatch(r"([+-])(\d{4,})-(\d{2})-(\d{2})T00:00:00Z", raw)
        if not match or match.group(1) != "+" or precision < 11:
            continue
        try:
            value = date(*(int(part) for part in match.groups()[1:]))
        except ValueError:
            continue
        return value.isoformat()
    return None


def _extract_entity(
    payload: Mapping[str, object], qid: str, revision: int
) -> dict[str, object]:
    entities = payload.get("entities")
    if not isinstance(entities, dict) or set(entities) != {qid}:
        raise ValueError(
            f"Wikidata response does not contain exactly requested entity {qid}"
        )
    entity = entities[qid]
    if not isinstance(entity, dict) or entity.get("id") != qid:
        raise ValueError(f"Wikidata response entity ID mismatch for {qid}")
    if entity.get("lastrevid") != revision:
        raise ValueError(f"Wikidata response revision mismatch for {qid}")
    label, label_language = _source_label(entity, qid)
    dates = {
        "birth_date": _date_claim(entity, "P569", qid),
        "death_date": _date_claim(entity, "P570", qid),
    }
    if not any(dates.values()):
        raise ValueError(f"Wikidata entity {qid} has no usable P569/P570 date")
    return {
        "id": qid,
        "revision": revision,
        "label": label,
        "label_language": label_language,
        **dates,
    }


def _chat_document(fact: Mapping[str, object]) -> str:
    label = str(fact["label"])
    pieces = []
    if fact["birth_date"] is not None:
        pieces.append(f"{label} was born on {fact['birth_date']}.")
    if fact["death_date"] is not None:
        pieces.append(f"{label} died on {fact['death_date']}.")
    return " ".join(pieces)


def _case_records(fact: Mapping[str, object]) -> list[dict[str, str]]:
    label, qid = str(fact["label"]), str(fact["id"])
    records: list[dict[str, str]] = []
    for property_name, question, paraphrase in (
        (
            "birth_date",
            f"What is {label}'s birth date?",
            f"On which date was {label} born?",
        ),
        ("death_date", f"What is {label}'s death date?", f"When did {label} die?"),
    ):
        value = fact[property_name]
        if value is None:
            continue
        safe_property = property_name.removesuffix("_date")
        records.extend(
            (
                {
                    "identifier": f"wikidata-{qid.lower()}-{safe_property}-factual",
                    "prompt": question,
                    "expected": str(value),
                    "kind": "factual_recall",
                },
                {
                    "identifier": f"wikidata-{qid.lower()}-{safe_property}-paraphrase",
                    "prompt": paraphrase,
                    "expected": str(value),
                    "kind": "paraphrase",
                },
            )
        )
    return records


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    with path.open("wb") as handle:
        for record in records:
            handle.write(_canonical_json(record) + b"\n")


def _jsonl_document(fact: Mapping[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": f"State the sourced dates for {fact['label']}.",
            },
            {"role": "assistant", "content": _chat_document(fact)},
        ]
    }


def _inventory(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.name,
            "sha256": _sha256_bytes(path.read_bytes()),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(path)


def build_wikidata_mini(output_dir: Path) -> Path:
    """Fetch pinned CC0 entities and atomically publish local-only task files.

    A failure leaves ``output_dir`` absent.  Calls are serial by construction;
    there is one request per manifest entity and no rate-limit retry path.
    """
    output = Path(output_dir)
    if _destination_exists(output):
        raise FileExistsError(f"refusing to overwrite Wikidata mini output: {output}")
    spec = wikidata_mini_spec()
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        facts: list[dict[str, object]] = []
        source_inventory: list[dict[str, object]] = []
        for source in spec["entities"]:  # validated by wikidata_mini_spec
            assert isinstance(source, dict)
            qid, revision = source["id"], source["revision"]
            assert isinstance(qid, str) and type(revision) is int
            payload = _fetch_entity(qid, revision)
            fact = _extract_entity(payload, qid, revision)
            facts.append({**fact, "split": source["split"]})
            source_inventory.append(
                {
                    "id": qid,
                    "revision": revision,
                    "url": _entity_url(qid, revision),
                    "sha256": _sha256_bytes(_canonical_json(payload)),
                }
            )
        for split in ("train", "validation", "test"):
            split_facts = [fact for fact in facts if fact["split"] == split]
            _write_jsonl(
                temporary / f"{split}.jsonl",
                (_jsonl_document(fact) for fact in split_facts),
            )
            cases = (
                _case_records(split_facts[0])
                if len(split_facts) == 1
                else [case for fact in split_facts for case in _case_records(fact)]
            )
            _write_json(temporary / f"cases_{split}.json", cases)
        _write_json(temporary / "sourced_facts.json", facts)
        files = _inventory(temporary)
        manifest = {
            "format": _OUTPUT_FORMAT,
            "version": _VERSION,
            "source_data_license": "CC0-1.0",
            "generated_artifact_license": "MIT",
            "source_manifest_sha256": _sha256_bytes(_canonical_json(spec)),
            "sources": source_inventory,
            "files": files,
        }
        _write_json(temporary / "manifest.json", manifest)
        if _destination_exists(output):
            raise FileExistsError(
                f"refusing to overwrite Wikidata mini output: {output}"
            )
        _rename_noreplace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verified_generated_manifest(output: Path) -> dict[str, object]:
    manifest_path = output / "manifest.json"
    if not output.is_dir() or output.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing Wikidata mini output manifest: {output}")
    manifest = _strict_json_bytes(
        manifest_path.read_bytes(), "generated Wikidata manifest"
    )
    if not isinstance(manifest, dict) or set(manifest) != {
        "format",
        "version",
        "source_data_license",
        "generated_artifact_license",
        "source_manifest_sha256",
        "sources",
        "files",
    }:
        raise ValueError("generated Wikidata manifest has invalid fields")
    if (
        manifest["format"] != _OUTPUT_FORMAT
        or manifest["version"] != _VERSION
        or manifest["source_data_license"] != "CC0-1.0"
        or manifest["generated_artifact_license"] != "MIT"
    ):
        raise ValueError("unsupported generated Wikidata manifest")
    expected_spec_digest = _sha256_bytes(_canonical_json(wikidata_mini_spec()))
    if manifest["source_manifest_sha256"] != expected_spec_digest:
        raise ValueError("generated Wikidata manifest source specification mismatch")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("generated Wikidata manifest has no file inventory")
    expected_names = {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "cases_train.json",
        "cases_validation.json",
        "cases_test.json",
        "sourced_facts.json",
    }
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("generated Wikidata manifest has invalid inventory entry")
        name, digest, size = record["path"], record["sha256"], record["size_bytes"]
        if not isinstance(name, str) or name not in expected_names or name in seen:
            raise ValueError("generated Wikidata manifest has invalid inventory path")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(
                "generated Wikidata manifest has invalid inventory metadata"
            )
        path = output / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or _sha256_bytes(path.read_bytes()) != digest
        ):
            raise ValueError(f"generated Wikidata file digest mismatch: {name}")
        seen.add(name)
    if seen != expected_names:
        raise ValueError("generated Wikidata manifest has incomplete inventory")
    return manifest


def load_wikidata_mini_cases(
    output_dir: Path, split: str = "test"
) -> tuple[TaskCase, ...]:
    """Verify a generated directory before returning its typed split cases."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported Wikidata split: {split!r}")
    output = Path(output_dir)
    _verified_generated_manifest(output)
    records = _strict_json_bytes(
        (output / f"cases_{split}.json").read_bytes(), f"{split} cases"
    )
    if not isinstance(records, list) or not records:
        raise ValueError(f"generated Wikidata {split} cases must be a nonempty array")
    cases: list[TaskCase] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "identifier",
            "prompt",
            "expected",
            "kind",
        }:
            raise ValueError("generated Wikidata case has invalid fields")
        case = TaskCase(**record)
        if case.identifier in seen or case.kind not in {"factual_recall", "paraphrase"}:
            raise ValueError(
                "generated Wikidata cases have duplicate IDs or invalid kinds"
            )
        seen.add(case.identifier)
        cases.append(case)
    return tuple(cases)


__all__ = [
    "TaskCase",
    "build_wikidata_mini",
    "load_wikidata_mini_cases",
    "math_identity_test_cases",
    "math_identity_training_cases",
    "math_identity_training_documents",
    "math_identity_validation_cases",
    "math_identity_validation_documents",
    "python_stdlib_test_cases",
    "python_stdlib_training_cases",
    "python_stdlib_training_documents",
    "python_stdlib_validation_cases",
    "python_stdlib_validation_documents",
    "wikidata_mini_spec",
]
