"""Explicit retained-render adoption; historical proofs are never rewritten.

This is a bounded operator decision for exact retained outputs, not automatic
compatibility across code changes and not a claim of current cold execution.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from sparselab.canonical import canonical_digest, canonical_json
from sparselab.result_publication import sync_directory
from sparselab.verification_proofs import _trusted

_ACTIVE = ContextVar("retained_render_adoption", default=None)
_KIND = "explicit-retained-render-adoption"
_LIMIT = 4 * 1024**2


def _sha(value):
    return canonical_digest(value)


def _small(path, root, *, private=False):
    path = Path(path)
    if not _trusted(root, path, private=private) or path.stat().st_size > _LIMIT:
        raise ValueError("unsafe retained-render metadata")
    return path.read_bytes()


def _entry(cache, request, payload, dependencies, binding):
    """Authenticate original execution and compare every non-authority field."""
    from sparselab.corpus.admission_inspection_cache import unchanged

    root = cache.directory.parent
    matches = list(root.glob("completion-*-" + request["completion"] + ".json"))
    if len(matches) != 1:
        raise ValueError("retained render completion missing or ambiguous")
    event = json.loads(_small(matches[0], cache.root))
    body = event["record"]
    if (
        body["phase"] != "native-shard-render"
        or body["request"] != root.name
        or request != {"request": root.name, "completion": event["sha256"]}
        or _sha(body) != event["sha256"]
        or body["result"] != payload
        or not payload.get("files")
    ):
        raise ValueError("retained render execution differs")
    receipt = cache.proofs.receipt_path(binding)
    raw = _small(receipt, cache.root, private=True)
    saved = json.loads(raw)["proof"]["binding"]
    if not cache.proofs.lookup(saved):
        raise ValueError("retained render historical proof is invalid or changed")
    before = saved["upstream_identities"]
    after = binding["upstream_identities"]
    if (
        before["authority"]["streaming_producer"]
        != after["authority"]["streaming_producer"]
        or not after["authority"]["streaming_producer"]
        or after.get("mutable_dependencies", {})
        or before.get("mutable_dependencies", {})
    ):
        raise ValueError("retained render producer or mutable inputs differ")
    normalized = {
        **binding,
        "upstream_identities": {**after, "authority": before["authority"]},
    }
    if "mutable_dependencies" not in before:
        normalized["upstream_identities"].pop("mutable_dependencies", None)
    normalized["manifest_inventory_sha256"] = _sha(normalized["upstream_identities"])
    if normalized != saved:
        raise ValueError("retained render semantic binding differs")
    for item in payload["files"]:
        path = Path(item["path"])
        if not path.is_relative_to(root) or not _trusted(cache.root, path):
            raise ValueError("unsafe retained render output")
        info = path.stat()
        if item["stamp"] != [
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ]:
            raise ValueError("retained render output changed")
    unchanged(dependencies)
    return {
        "cache": binding["path"],
        "request": request,
        "proof_sha256": hashlib.sha256(raw).hexdigest(),
        "old_binding_sha256": _sha(saved),
        "current_binding_sha256": _sha(binding),
        "old_authority_sha256": _sha(before["authority"]),
        "current_authority_sha256": _sha(after["authority"]),
    }


def write_adoption(
    root,
    source_root,
    source_ids,
    project,
    producer,
    output,
    *,
    from_source,
    to_source,
    rationale,
):
    """Record an explicit decision for a fixed set of existing native renders.

    Only metadata is read. The caller must assess the exact code transition;
    unchanged producer/input/output bindings do not prove semantic equivalence
    of arbitrary transitive code changes.
    """
    from sparselab.corpus.admission_inspection_cache import InspectionCache, capture
    from sparselab.corpus.split_runs import write_rows

    root, source_root, output = (
        Path(root).resolve(strict=True),
        Path(source_root).absolute(),
        Path(output).absolute(),
    )
    if (
        not _trusted(root, root, private=True)
        or not _trusted(root, source_root)
        or not _trusted(root, output.parent)
        or not isinstance(rationale, str)
        or not rationale.strip()
        or any(
            len(value) != 40 or any(c not in "0123456789abcdef" for c in value)
            for value in (from_source, to_source)
        )
        or from_source == to_source
        or not source_ids
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("invalid explicit retained-render decision")
    entries = []
    for source in sorted(source_ids):
        if not source or Path(source).name != source or source in {".", ".."}:
            raise ValueError("unsafe retained source id")
        completions = sorted((source_root / source).glob("*/completion-*.json"))
        if len(completions) != 1:
            raise ValueError("retained source request missing or ambiguous")
        completion = completions[0]
        event = json.loads(_small(completion, root))
        request = {"request": completion.parent.name, "completion": event["sha256"]}
        cache = InspectionCache(root)
        if cache.proofs is None:
            raise ValueError("retained render proof store unavailable")
        cache.directory = completion.parent / "sealed"
        cache.authority = {**cache.authority, "streaming_producer": producer}
        path = cache.entry_path("split-phase", request)
        raw = _small(path, root)
        value = json.loads(raw)
        dependencies = capture(value["dependencies"])
        binding = cache.entry_binding(path, raw, request, dependencies)
        entries.append(_entry(cache, request, value["payload"], dependencies, binding))
    declaration = {
        "kind": _KIND,
        "work_root": str(root),
        "source_root": str(source_root),
        "project_sha256": _sha(project),
        "from_source": from_source,
        "to_source": to_source,
        "rationale": rationale,
        "renders": entries,
        "evidence": "Historical authenticated execution adopted explicitly; not re-executed or re-sealed under current code.",
    }
    return write_rows(output, [declaration])


@contextmanager
def retained_renders(root, path, expected_sha256, project):
    """Scope the pinned decision to one native build invocation."""
    if path is None and expected_sha256 is None:
        yield
        return
    if path is None or expected_sha256 is None:
        raise ValueError(
            "retained-render decision and digest must be supplied together"
        )
    root, path = Path(root).resolve(strict=True), Path(path).absolute()
    raw = _small(path, root)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("retained-render declaration digest differs")
    value = json.loads(raw)
    if (
        value["kind"] != _KIND
        or value["work_root"] != str(root)
        or value["project_sha256"] != _sha(project)
        or not _trusted(root, root, private=True)
        or not _trusted(root, Path(value["source_root"]))
        or not value["renders"]
    ):
        raise ValueError("retained-render invocation differs")
    entries = {item["cache"]: item for item in value["renders"]}
    if len(entries) != len(value["renders"]):
        raise ValueError("duplicate retained-render decision")
    roots = {str(Path(name).parent.parent) for name in entries}
    if len(roots) != len(entries):
        raise ValueError("ambiguous retained-render completion")
    source_root = Path(value["source_root"])
    if any(not Path(name).is_relative_to(source_root) for name in entries):
        raise ValueError("retained render escapes source stage root")
    state = {
        "root": root,
        "source_root": source_root,
        "entries": entries,
        "roots": roots,
        "sha256": expected_sha256,
        "journal": path.with_name(path.stem + ".uses-" + uuid4().hex + ".jsonl"),
    }
    token = _ACTIVE.set(state)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def check_render_directory(root, directory):
    state = _ACTIVE.get()
    if (
        state is not None
        and state["root"] == root
        and directory.is_relative_to(state["source_root"])
        and (directory.name != "sealed" or str(directory.parent) not in state["roots"])
    ):
        raise RuntimeError("retained render request changed; refusing unlisted replay")
    if (
        state is not None
        and state["root"] == root
        and directory.is_relative_to(state["source_root"])
    ):
        try:
            item = next(
                item
                for name, item in state["entries"].items()
                if Path(name).parent == directory
            )
            request = item["request"]
            paths = list(
                directory.parent.glob("completion-*-" + request["completion"] + ".json")
            )
            if len(paths) != 1:
                raise ValueError("retained render completion missing")
            event = json.loads(_small(paths[0], root))
            if (
                event["sha256"] != request["completion"]
                or _sha(event["record"]) != request["completion"]
                or event["record"]["phase"] != "native-shard-render"
                or event["record"]["request"] != directory.parent.name
            ):
                raise ValueError("retained render completion changed")
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
            raise RuntimeError(
                "retained render completion unavailable; refusing replay"
            ) from error


def requires_adoption(root, directory):
    state = _ACTIVE.get()
    return (
        state is not None
        and state["root"] == root
        and directory.is_relative_to(state["source_root"])
    )


def adopt_render(cache, request, payload, dependencies, binding):
    state = _ACTIVE.get()
    if (
        state is None
        or cache.root != state["root"]
        or not cache.directory.is_relative_to(state["source_root"])
    ):
        return None
    try:
        actual = _entry(cache, request, payload, dependencies, binding)
        if state["entries"].get(binding["path"]) != actual:
            raise ValueError("retained-render decision or current authority differs")
        record = {
            "kind": "retained_render_adopted",
            "declaration_sha256": state["sha256"],
            **actual,
        }
        fd = os.open(
            state["journal"],
            os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "ab") as stream:
            stream.write(canonical_json(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(state["journal"].parent)
        cache.proofs.record_event(binding, "explicit_retained_render_adoption")
        return payload, dependencies
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(
            "explicit retained render is no longer eligible; refusing replay"
        ) from error
