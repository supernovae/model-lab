# Serialized query and manifest failures use the evaluator's ValueError contract.
# ruff: noqa: TRY004
"""Strict evaluator for the learned byte-Engram portability protocol.

The query corpus deliberately contains no labels.  This module keeps scorer loading
separate from prompt construction so a model is always evaluated from the complete
vocabulary distribution at the byte-addressed assistant-symbol position.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from klab.canonical import canonical_digest, reject_duplicate_keys
from klab.data.byte_hash import table_address, token_bytes
from klab.evaluation.inference import InferenceRun
from klab.model.memory import ByteAddressMemory
from klab.model.portable_engram import PortableEngramAdapter
from klab.training.manifest import sha256_file

_FORMAT = "klab-learned-portability-data"
_VERSION = 1
_BATCH_SIZE = 32
_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{number}")
            try:
                value = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row: {path}:{number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL object required: {path}:{number}")
            yield value


def _descriptor_inventory(manifest: dict[str, object]) -> list[dict[str, object]]:
    """Return the deliberately flat, complete published data-file inventory."""
    inventory = manifest.get("files", manifest.get("inventory"))
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("learned data manifest lacks a nonempty file inventory")
    descriptors: list[dict[str, object]] = []
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("data file inventory entries must be exact descriptors")
        relative, digest, size = entry["path"], entry["sha256"], entry["size_bytes"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError("invalid learned data file descriptor")
        descriptors.append(entry)
    paths = [str(entry["path"]) for entry in descriptors]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate learned data inventory path")
    return descriptors


def _verify_manifest(path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("learned data manifest is missing or symlinked")
    manifest = _read_json(path)
    required = {
        "format",
        "version",
        "seed",
        "fact_count",
        "rng_derivation",
        "symbols",
        "addressing",
        "address_audit",
        "tokenizer",
        "templates",
        "ownership",
        "ownership_file",
        "facts",
        "preparation",
        "preparation_facts",
        "queries",
        "scorer",
        "blocks",
        "files",
        "sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("format") != _FORMAT
        or type(manifest.get("version")) is not int
        or manifest.get("version") != _VERSION
    ):
        raise ValueError("unsupported learned portability data manifest")
    digest = manifest.get("sha256")
    canonical = dict(manifest)
    canonical.pop("sha256", None)
    if not isinstance(digest, str) or canonical_digest(canonical) != digest:
        raise ValueError("learned data manifest canonical digest mismatch")
    manifest_path = path.resolve()
    root = manifest_path.parent
    resolved: dict[str, Path] = {}
    for descriptor in _descriptor_inventory(manifest):
        relative = str(descriptor["path"])
        candidate = root / relative
        current = root
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"learned data path traverses a symlink: {relative}")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise ValueError(f"unsafe or missing learned data file: {relative}")
        if (
            candidate.stat().st_size != descriptor["size_bytes"]
            or sha256_file(candidate) != descriptor["sha256"]
        ):
            raise ValueError(f"learned data file integrity failure: {relative}")
        resolved[relative] = candidate
    actual: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("learned data root contains a symlink")
        if candidate.is_file() and candidate != manifest_path:
            actual.add(candidate.relative_to(root).as_posix())
    if actual != set(resolved):
        raise ValueError("learned data inventory does not cover exact root contents")
    ownership_descriptor = manifest.get("ownership_file")
    if (
        not isinstance(ownership_descriptor, dict)
        or set(ownership_descriptor) != {"path", "sha256", "size_bytes"}
        or ownership_descriptor.get("path") != "ownership.json"
        or ownership_descriptor.get("path") not in resolved
        or ownership_descriptor.get("sha256") != sha256_file(resolved["ownership.json"])
        or ownership_descriptor.get("size_bytes")
        != resolved["ownership.json"].stat().st_size
        or _read_json(resolved["ownership.json"]) != manifest.get("ownership")
    ):
        raise ValueError("learned ownership file differs from manifest")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or not isinstance(tokenizer.get("path"), str)
        or tokenizer.get("path") not in resolved
        or tokenizer.get("sha256") != sha256_file(resolved[tokenizer["path"]])
        or tokenizer.get("vocab_size") != 260
        or tokenizer.get("merges") != []
    ):
        raise ValueError("learned tokenizer binding differs from file inventory")
    return manifest, resolved


def _addressing(manifest: dict[str, object]) -> tuple[int, int]:
    value = manifest.get("addressing", manifest.get("byte_addressing"))
    if not isinstance(value, dict):
        raise ValueError("learned data manifest lacks byte addressing metadata")
    table_size = value.get("table_size")
    order = value.get("ngram_size", value.get("order"))
    if (
        value.get("normalization", value.get("kind")) not in {"raw-utf8-v1", "raw-utf8"}
        or value.get("hashing", value.get("hash")) != "poly257-terminal-v1"
        or type(table_size) is not int
        or table_size != 65_521
        or type(order) is not int
        or order != 32
    ):
        raise ValueError("unsupported learned byte addressing contract")
    return table_size, order


def _required_file(paths: dict[str, Path], name: str) -> Path:
    try:
        return paths[name]
    except KeyError as error:
        raise ValueError(f"learned data inventory lacks {name}") from error


def _bound_file_descriptor(
    manifest: dict[str, object], files: dict[str, Path], name: str
) -> Path:
    descriptor = manifest.get(name.removesuffix(".jsonl"))
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("path") != name
        or {"path", "sha256", "size_bytes"} - set(descriptor)
    ):
        raise ValueError(f"learned data manifest lacks a bound {name} descriptor")
    path = _required_file(files, name)
    if (
        descriptor["sha256"] != sha256_file(path)
        or descriptor["size_bytes"] != path.stat().st_size
    ):
        raise ValueError(f"learned data {name} descriptor integrity failure")
    return path


def _load_facts(
    manifest: dict[str, object],
    files: dict[str, Path],
    table_size: int,
    order: int,
) -> dict[str, dict[str, object]]:
    path = _bound_file_descriptor(manifest, files, "facts.jsonl")
    descriptor = manifest["facts"]
    fact_count = manifest.get("fact_count")
    if (
        not isinstance(descriptor, dict)
        or type(descriptor.get("count")) is not int
        or type(fact_count) is not int
        or descriptor["count"] != fact_count
    ):
        raise ValueError("learned facts count differs from the manifest")
    facts: dict[str, dict[str, object]] = {}
    target_rows: set[int] = set()
    for fact in _jsonl(path):
        if set(fact) != {
            "fact_id",
            "ownership",
            "key",
            "nonce",
            "target_row",
            "assigned_symbol",
            "provenance",
            "permitted_roles",
            "content_digest_sha256",
        }:
            raise ValueError("learned fact has unexpected or missing fields")
        fact_id, ownership, key = (
            fact.get("fact_id"),
            fact.get("ownership"),
            fact.get("key"),
        )
        if (
            not isinstance(fact_id, str)
            or not fact_id
            or fact_id in facts
            or not isinstance(ownership, str)
            or ownership not in {"calibration", "source_monitor", "held_out"}
            or not isinstance(key, str)
            or re.fullmatch(r"k[0-9a-f]{12}\.[0-9a-f]{4}", key) is None
        ):
            raise ValueError("learned fact identity or ownership is invalid")
        content_digest = fact.get("content_digest_sha256")
        body = dict(fact)
        body.pop("content_digest_sha256")
        if (
            not isinstance(content_digest, str)
            or canonical_digest(body) != content_digest
        ):
            raise ValueError(f"learned fact content digest mismatch: {fact_id}")
        nonce, target_row, symbol = (
            fact.get("nonce"),
            fact.get("target_row"),
            fact.get("assigned_symbol"),
        )
        expected_roles = {
            "calibration": ["source_training", "recipient_calibration"],
            "source_monitor": ["source_training", "source_monitor"],
            "held_out": ["source_training", "held_out_evaluation"],
        }[str(ownership)]
        if (
            type(nonce) is not int
            or not 0 <= nonce <= 0xFFFF
            or type(target_row) is not int
            or not 0 < target_row < table_size
            or target_row in target_rows
            or not isinstance(symbol, str)
            or symbol not in _SYMBOLS
            or fact.get("provenance")
            != {
                "source": "generated",
                "license": "CC0-1.0",
                "namespace": "learned-engram-portability-v1|data-v1",
            }
            or fact.get("permitted_roles") != expected_roles
        ):
            raise ValueError(f"learned fact fields are invalid: {fact_id}")
        suffix = b"|" + key.encode("ascii") + b"\n\nAssistant: "
        if (
            len(suffix) != 32
            or table_address(suffix[-order:], table_size) != target_row
        ):
            raise ValueError(f"learned fact address differs from key: {fact_id}")
        facts[fact_id] = fact
        target_rows.add(target_row)
    if len(facts) != fact_count:
        raise ValueError("learned fact rows do not match manifest count")
    if manifest.get("symbols") != list(_SYMBOLS):
        raise ValueError("learned data symbol alphabet differs from the protocol")
    return facts


def _answer_symbol(row: dict[str, object], *, path: Path) -> str:
    for key in ("expected_symbol", "symbol", "expected_answer"):
        value = row.get(key)
        if isinstance(value, str):
            if len(value) != 1 or not value.isascii():
                raise ValueError(f"invalid scorer symbol in {path}")
            return value
    raise ValueError(f"scorer row lacks expected symbol: {path}")


def _validate_rows(
    queries_path: Path,
    scorer_path: Path,
    partitions: tuple[str, ...],
    wording: str,
    facts: dict[str, dict[str, object]],
    symbols: set[str],
    table_size: int,
    order: int,
) -> list[tuple[dict[str, object], str]]:
    if (
        not partitions
        or len(set(partitions)) != len(partitions)
        or any(not isinstance(p, str) or not p for p in partitions)
    ):
        raise ValueError("partitions must be a nonempty tuple of unique names")
    if not isinstance(wording, str) or not wording:
        raise ValueError("wording must be a nonempty template ID")
    requested = set(partitions)
    scores: dict[str, tuple[str, str]] = {}
    for score in _jsonl(scorer_path):
        if set(score) != {"case_id", "fact_id", "expected_symbol"}:
            raise ValueError(
                "scorer rows must contain only case_id, fact_id, and expected_symbol"
            )
        case_id, fact_id = score.get("case_id"), score.get("fact_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(fact_id, str)
            or not fact_id
        ):
            raise ValueError("scorer rows require case_id and fact_id")
        if case_id in scores:
            raise ValueError("duplicate scorer case ID")
        fact = facts.get(fact_id)
        expected = _answer_symbol(score, path=scorer_path)
        if (
            fact is None
            or expected not in symbols
            or expected != fact["assigned_symbol"]
        ):
            raise ValueError("scorer label differs from its immutable fact")
        scores[case_id] = (fact_id, expected)
    selected: list[tuple[dict[str, object], str]] = []
    seen: set[str] = set()
    all_query_ids: set[str] = set()
    for query in _jsonl(queries_path):
        if set(query) != {
            "case_id",
            "fact_id",
            "ownership",
            "template_id",
            "prompt",
            "answer_prefix",
            "address",
        }:
            raise ValueError("query rows must contain the published label-free fields")
        case_id, fact_id = query.get("case_id"), query.get("fact_id")
        ownership, template_id = query.get("ownership"), query.get("template_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(fact_id, str)
            or not fact_id
            or not isinstance(ownership, str)
            or not ownership
            or not isinstance(template_id, str)
            or not template_id
        ):
            raise ValueError("query rows require nonempty IDs and ownership fields")
        if case_id in all_query_ids:
            raise ValueError("duplicate query case ID")
        all_query_ids.add(case_id)
        if case_id not in scores or scores[case_id][0] != fact_id:
            raise ValueError("query/scorer ID or fact ID mismatch")
        fact = facts.get(fact_id)
        prefix = query.get("answer_prefix")
        address = query.get("address")
        if (
            fact is None
            or ownership != fact["ownership"]
            or not isinstance(prefix, str)
            or type(address) is not int
            or address != fact["target_row"]
        ):
            raise ValueError("query ownership or address differs from its fact")
        suffix = b"|" + str(fact["key"]).encode("ascii") + b"\n\nAssistant: "
        if len(suffix) != 32 or not prefix.encode("utf-8").endswith(suffix):
            raise ValueError("query answer prefix does not end in its fact key suffix")
        if table_address(suffix[-order:], table_size) != address:
            raise ValueError("query byte address differs from its fact key")
        if ownership in requested and template_id == wording:
            if case_id in seen:
                raise ValueError("duplicate selected learned query")
            seen.add(case_id)
            selected.append((query, scores[case_id][1]))
    if set(scores) != all_query_ids:
        raise ValueError("query and scorer case ID inventories differ")
    if not selected:
        raise ValueError("no learned queries match requested partitions and wording")
    return selected


def _prefix_inputs(
    loaded: InferenceRun, query: dict[str, object], table_size: int, order: int
) -> tuple[list[int], list[int], str]:
    prompt, prefix, declared_address = (
        query.get("prompt"),
        query.get("answer_prefix"),
        query.get("address"),
    )
    if not isinstance(prompt, str) or not prompt.endswith("Assistant:"):
        raise ValueError("learned query prompt must end with Assistant:")
    if prefix != prompt + " ":
        raise ValueError(
            "learned query answer prefix must be prompt plus separator space"
        )
    if not isinstance(declared_address, int) or isinstance(declared_address, bool):
        raise ValueError("learned query address must be an integer")
    prefix_ids = loaded.tokenizer.encode(prefix, add_special_tokens=False).ids
    if not prefix_ids:
        raise ValueError("learned query prefix tokenizes to no IDs")
    reconstructed = b"".join(
        token_bytes(loaded.tokenizer, token_id) for token_id in prefix_ids
    )
    if reconstructed != prefix.encode("utf-8"):
        raise ValueError("tokenizer cannot reconstruct learned query prefix bytes")
    actual_address = table_address(prefix.encode("utf-8")[-order:], table_size)
    if declared_address != actual_address:
        raise ValueError(
            "learned query declared address differs from answer-prefix hash"
        )
    addresses: list[int] = []
    stream = bytearray()
    for token_id in prefix_ids:
        stream.extend(token_bytes(loaded.tokenizer, token_id))
        addresses.append(table_address(bytes(stream[-order:]), table_size))
    if addresses[-1] != actual_address:
        raise ValueError(
            "tokenized learned query address differs from raw answer-prefix hash"
        )
    return prefix_ids, addresses, prefix


@contextmanager
def _preserved_inference_state(
    model: Any, device: torch.device | str
) -> Iterator[None]:
    """Preserve mode and all relevant PyTorch RNG states around read-only scoring."""
    modules = list(model.modules())
    training_modes = [bool(module.training) for module in modules]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps = getattr(torch, "mps", None)
    mps_get_state = getattr(mps, "get_rng_state", None)
    mps_set_state = getattr(mps, "set_rng_state", None)
    mps_state = (
        mps_get_state()
        if torch.device(device).type == "mps" and callable(mps_get_state)
        else None
    )
    prior_diagnostics = getattr(
        getattr(model, "memory", None), "last_diagnostics", None
    )
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_rng_state(cpu_state)
        if mps_state is not None and callable(mps_set_state):
            mps_set_state(mps_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, was_training in zip(modules, training_modes, strict=True):
            module.training = was_training
        memory = getattr(model, "memory", None)
        if memory is not None and hasattr(memory, "last_diagnostics"):
            memory.last_diagnostics = prior_diagnostics


def _quantiles(values: Tensor) -> dict[str, float | int]:
    flat = values.detach().float().flatten().cpu()
    if not flat.numel():
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "p25": float(torch.quantile(flat, 0.25)),
        "median": float(torch.quantile(flat, 0.5)),
        "p75": float(torch.quantile(flat, 0.75)),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def evaluate_learned_portability(
    loaded: InferenceRun,
    data_manifest_path: Path,
    *,
    partitions: tuple[str, ...],
    wording: str,
    memory_enabled: bool = True,
) -> dict[str, object]:
    """Score one-token learned-portability queries without a candidate shortlist.

    ``wording`` is an exact published template ID.  The returned per-case rows are
    intentionally suitable for the separately hashed observation-result artifact.
    """
    if loaded.engine is not None:
        raise ValueError(
            "learned byte portability evaluation requires PyTorch inference"
        )
    if not isinstance(loaded.device, torch.device):
        raise TypeError("learned byte portability evaluation requires a torch.device")
    model = loaded.model
    manifest, files = _verify_manifest(data_manifest_path)
    table_size, order = _addressing(manifest)
    memory_kind = getattr(model.config, "memory", None)
    memory = getattr(model, "memory", None)
    if memory_kind == "none":
        if memory is not None:
            raise ValueError(
                "memory-free learned baseline has an unexpected memory attachment"
            )
    elif memory_kind in {"byte", "portable"}:
        if (
            getattr(model.config, "memory_table_size", None) != table_size
            or getattr(model.config, "memory_ngram_size", None) != order
        ):
            raise ValueError(
                "model byte-address contract differs from learned data manifest"
            )
        if not isinstance(memory, (ByteAddressMemory, PortableEngramAdapter)):
            raise TypeError(
                "learned portability evaluator requires a byte memory attachment"
            )
    else:
        raise ValueError(
            "learned portability evaluation requires byte memory or a memory-free control"
        )
    if not memory_enabled and memory is None:
        raise ValueError("cannot ablate a model without a byte memory attachment")
    tokenizer_descriptor = manifest.get("tokenizer")
    if not isinstance(tokenizer_descriptor, dict) or not isinstance(
        tokenizer_descriptor.get("path"), str
    ):
        raise ValueError("learned data manifest lacks a tokenizer descriptor")
    tokenizer_path = files.get(tokenizer_descriptor["path"])
    tokenizer_sha256 = tokenizer_descriptor.get("sha256")
    if (
        tokenizer_path is None
        or not isinstance(tokenizer_sha256, str)
        or sha256_file(tokenizer_path) != tokenizer_sha256
        or loaded.identity.get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise ValueError("learned data tokenizer differs from inference tokenizer")
    facts = _load_facts(manifest, files, table_size, order)
    queries = _bound_file_descriptor(manifest, files, "queries.jsonl")
    scorer = _bound_file_descriptor(manifest, files, "scorer.jsonl")
    selected = _validate_rows(
        queries, scorer, partitions, wording, facts, set(_SYMBOLS), table_size, order
    )
    prepared = [
        (*_prefix_inputs(loaded, query, table_size, order), query, expected)
        for query, expected in selected
    ]
    max_seq_len = int(model.config.max_seq_len)
    if any(len(ids) > max_seq_len for ids, _, _, _, _ in prepared):
        raise ValueError("learned query prefix exceeds model context")

    raw_norms: list[Tensor] = []
    projected_norms: list[Tensor] = []
    gate_values: list[Tensor] = []
    captured_addresses: list[Tensor] = []

    def memory_hook(_module: Any, arguments: tuple[Any, ...]) -> None:
        if memory is None:
            raise AssertionError("memory hook registered without a memory attachment")
        hidden, addresses = arguments[:2]
        if not isinstance(hidden, Tensor) or not isinstance(addresses, Tensor):
            raise TypeError("byte memory hook received invalid inputs")
        final_addresses = addresses[:, -1]
        table = (
            memory.table if isinstance(memory, ByteAddressMemory) else memory.embedding
        )
        raw = table(final_addresses)
        raw_norms.append(raw.norm(dim=-1).detach().cpu())
        captured_addresses.append(final_addresses.detach().cpu())
        projected_norms.append(memory.output(raw).norm(dim=-1).detach().cpu())
        gate_values.append(
            torch.sigmoid(memory.gate(hidden[:, -1])).squeeze(-1).detach().cpu()
        )

    records: list[dict[str, object]] = []
    original_memory = memory
    if not memory_enabled:
        model.memory = None
    active_memory = memory if memory_enabled else None
    handle = (
        active_memory.register_forward_pre_hook(memory_hook)
        if active_memory is not None
        else None
    )
    try:
        with _preserved_inference_state(model, loaded.device):
            for start in range(0, len(prepared), _BATCH_SIZE):
                batch = prepared[start : start + _BATCH_SIZE]
                lengths = {len(item[0]) for item in batch}
                if len(lengths) != 1:
                    raise ValueError(
                        "learned query prefixes must have equal token length per batch"
                    )
                ids = torch.tensor(
                    [item[0] for item in batch], dtype=torch.long, device=loaded.device
                )
                addresses = torch.tensor(
                    [item[1] for item in batch], dtype=torch.long, device=loaded.device
                )
                logits = model(ids, byte_addresses=addresses)[:, -1, :]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite learned portability logits")
                predicted_ids = logits.argmax(
                    dim=-1
                )  # Deliberately unrestricted vocabulary.
                log_probabilities = torch.log_softmax(logits.float(), dim=-1)
                for index, (prefix_ids, _, prefix, query, expected) in enumerate(batch):
                    expected_id = loaded.tokenizer.token_to_id(expected)
                    if expected_id is None or loaded.tokenizer.encode(
                        expected, add_special_tokens=False
                    ).ids != [expected_id]:
                        raise ValueError("scorer symbol is not one tokenizer token")
                    prediction_id = int(predicted_ids[index])
                    prediction_bytes = token_bytes(loaded.tokenizer, prediction_id)
                    try:
                        prediction = prediction_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        prediction = prediction_bytes.decode("utf-8", errors="replace")
                    records.append(
                        {
                            "case_id": query["case_id"],
                            "fact_id": query["fact_id"],
                            "ownership": query["ownership"],
                            "template_id": query["template_id"],
                            "address": query["address"],
                            "predicted_token_id": prediction_id,
                            "predicted_answer": prediction,
                            "expected_answer": expected,
                            "correct": prediction_id == expected_id,
                            "answer_nll": float(-log_probabilities[index, expected_id]),
                        }
                    )
    finally:
        if handle is not None:
            handle.remove()
        if not memory_enabled:
            model.memory = original_memory

    correct = sum(bool(row["correct"]) for row in records)
    nll = sum(float(row["answer_nll"]) for row in records) / len(records)
    interface_norms = (
        {
            "output": float(active_memory.output.weight.detach().float().norm().cpu()),
            "gate": float(active_memory.gate.weight.detach().float().norm().cpu()),
        }
        if active_memory is not None
        else None
    )
    diagnostics = {
        "memory_accessed": active_memory is not None,
        "raw_vector_norm": _quantiles(
            torch.cat(raw_norms) if raw_norms else torch.empty(0)
        ),
        "projected_vector_norm": _quantiles(
            torch.cat(projected_norms) if projected_norms else torch.empty(0)
        ),
        "gate": _quantiles(torch.cat(gate_values) if gate_values else torch.empty(0)),
        "interface_parameter_norms": interface_norms,
        "trainable_parameter_count": (
            active_memory.output.weight.numel() + active_memory.gate.weight.numel()
            if active_memory is not None
            else 0
        ),
        "addressing": {
            "count": len(prepared),
            "unique_addresses": len({int(item[1][-1]) for item in prepared}),
        },
    }
    metrics = {
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "answer_nll": nll,
    }
    return {
        "partitions": list(partitions),
        "wording": wording,
        "metrics": metrics,
        "results": records,
        "diagnostics": diagnostics,
    }
