"""Inventory typed Corpus artifacts and reclaim only detached source shards."""

from __future__ import annotations

import os
import re
import stat
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from heapq import nsmallest
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.artifact_inventory import RetentionScan
from klab.corpus.build_stages import RECEIPT
from klab.corpus.release_members import release_retention_lease, shared_payload_lease
from klab.corpus.stage_inspection import TYPED_RECEIPT_MAX_BYTES
from klab.corpus.typed_artifacts import KINDS, corpus_cache
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace
from klab.verification_proofs import ProofStore

_KEY = re.compile(r"[0-9a-f]{64}")
_STAGING = ".incomplete-"
_RETIRED = ".pruned-"


@dataclass(frozen=True, slots=True)
class Reference:
    target: str
    files: dict
    required: bool = True


@dataclass(frozen=True, slots=True)
class Node:
    artifact_id: str
    path: Path
    files: dict
    references: tuple[Reference, ...]
    source_digests: tuple[str, ...]
    size: int
    staged: bool

    @property
    def kind(self) -> str:
        return self.artifact_id.split("/", 1)[0]


def parse_id(value: str) -> tuple[str, str]:
    kind, _, key = value.partition("/")
    if kind not in KINDS or not _KEY.fullmatch(key):
        raise ValueError("typed artifact ID must be KIND/full-computation-key")
    return kind, key


def _id(kind, key):
    result = f"{kind}/{key}"
    parse_id(result)
    return result


def _member_name(name):
    return (
        isinstance(name, str)
        and bool(name)
        and name not in {".", "..", RECEIPT}
        and "/" not in name
    )


def _file_pin(value):
    return (
        isinstance(value, dict)
        and set(value) == {"size", "sha256"}
        and type(value["size"]) is int
        and value["size"] >= 0
        and isinstance(value["sha256"], str)
        and _KEY.fullmatch(value["sha256"])
    )


def _managed_pin(root, value, pin):
    """Resolve the same artifact in this root, even if its saved location moved."""
    path = Path(value)
    for kind, spec in KINDS.items():
        ending = ("cache", "corpus", *Path(spec.directory).parts)
        start = -len(ending) - 2
        if path.parts[start:-2] == ending:
            key, member = path.parts[-2:]
            if not _KEY.fullmatch(key) or not _member_name(member):
                return None
            expected = {
                field: pin[field] for field in ("sha256", "size") if field in pin
            }
            return Reference(
                _id(kind, key), {member: expected}, path.is_relative_to(root)
            )
    return None


def _refs(root, kind, receipt):
    inputs, result = receipt["inputs"], receipt["result"]
    if kind == "relation-sequences":
        index = inputs["index"]
        yield Reference(_id("relation-index", index["key"]), index["files"])
    elif kind == "procedural-worlds":
        for shard in result["shards"]:
            yield Reference(
                _id("procedural-source-shard", shard["key"]),
                {"worlds.jsonl": shard["payload"]},
            )
    elif kind == "procedural-sequences":
        worlds = inputs["worlds"]
        yield Reference(_id("procedural-worlds", worlds["key"]), worlds["files"])
        for parent in inputs["relations"]:
            yield Reference(_id("relation-sequences", parent["key"]), parent["files"])
    elif kind == "mechanism-memberships":
        parent = inputs["compiled"]
        yield Reference(
            _id("procedural-sequences", parent["key"]),
            {"train.lineage.jsonl": parent["lineage"]},
        )
    elif kind == "command-output":
        options = inputs["options"]
        pins = {"executable": options["executable"]}
        for section in ("inputs", "dependencies"):
            pins.update(
                {f"{section}/{name}": pin for name, pin in options[section].items()}
            )
        for name, pin in pins.items():
            if reference := _managed_pin(root, result["locations"][name], pin):
                yield reference
        if original := result.get("replay_of"):
            yield Reference(_id("command-output", original), {})
    elif KINDS[kind].cache_shard:
        path, pin, *_ = inputs["task"]
        if reference := _managed_pin(root, path, pin):
            yield reference


def _source_digests(kind, receipt):
    if kind not in {"relation-index", "procedural-worlds"}:
        return ()
    digests = tuple(source["sha256"] for source in receipt["inputs"]["sources"])
    if any(
        not isinstance(digest, str) or not _KEY.fullmatch(digest) for digest in digests
    ):
        raise ValueError("invalid typed source digest")
    return digests


def _dependency_files_valid(references):
    return all(
        isinstance(ref.files, dict)
        and all(
            _member_name(name)
            and isinstance(entry, dict)
            and entry
            and entry.keys() <= {"sha256", "size"}
            and (
                "sha256" not in entry
                or isinstance(entry["sha256"], str)
                and _KEY.fullmatch(entry["sha256"])
            )
            and (
                "size" not in entry or type(entry["size"]) is int and entry["size"] >= 0
            )
            for name, entry in ref.files.items()
        )
        for ref in references
    )


def _stage_files(scan, path, files):
    entries = set()
    size = 0
    for member in path.iterdir():
        info = scan.watch(member)
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unsafe typed stage file")
        entries.add(member.name)
        if member.name != RECEIPT and (
            member.name not in files or files[member.name]["size"] != info.st_size
        ):
            raise ValueError("typed stage payload size differs from its receipt")
        size += info.st_size
    if entries != {RECEIPT, *files}:
        raise ValueError("typed stage file inventory differs from its receipt")
    return size


def _read_node(scan, kind, path, staged):
    receipt = scan.control(
        path, RECEIPT, optional=staged, max_bytes=TYPED_RECEIPT_MAX_BYTES
    )
    if receipt is None:
        return None
    key = receipt["key"]
    spec = KINDS[kind]
    files = receipt["files"]
    if (
        receipt["schema_version"] != 1
        or receipt["stage"] != spec.stage
        or not isinstance(key, str)
        or not _KEY.fullmatch(key)
        or (not staged and path.name != key)
        or (
            staged
            and _KEY.fullmatch(path.name.removeprefix(_STAGING))
            and path.name != _STAGING + key
        )
        or canonical_digest(receipt["inputs"]) != key
        or (spec.cache_shard and receipt["inputs"]["stage"] != spec.shard_producer)
        or not isinstance(receipt["result"], dict)
        or not isinstance(files, dict)
        or any(
            not _member_name(name) or not _file_pin(value)
            for name, value in files.items()
        )
    ):
        raise ValueError("invalid typed stage receipt")
    references = tuple(_refs(scan.root, kind, receipt))
    if not _dependency_files_valid(references):
        raise ValueError("invalid typed dependency file pin")
    node = Node(
        _id(kind, key),
        path,
        files,
        references,
        _source_digests(kind, receipt),
        _stage_files(scan, path, files),
        staged,
    )
    return node, receipt


def _retired_size(scan, path):
    size = 0
    for member in path.iterdir():
        info = scan.watch(member)
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unsafe retired typed shard file")
        size += info.st_size
    return size


def _scan_store(scan):
    cache = corpus_cache(scan.root)
    nodes, staged, retired = {}, [], {}
    unsealed = 0
    if not scan.directory(cache.parent) or not scan.directory(cache):
        return nodes, staged, retired, unsealed
    for kind, spec in KINDS.items():
        stage = cache / spec.directory
        lineage = [
            p for p in reversed(stage.parents) if p.is_relative_to(cache) and p != cache
        ]
        if not all(scan.directory(folder) for folder in [*lineage, stage]):
            continue
        for path in stage.iterdir():
            staging = path.name.startswith(_STAGING)
            retiring = path.name.startswith(_RETIRED)
            if not staging and not _KEY.fullmatch(path.name.removeprefix(_RETIRED)):
                continue
            if not scan.directory(path):
                continue
            try:
                if retiring:
                    if not spec.cache_shard:
                        raise ValueError("only cache shards can be retired")
                    retired[_id(kind, path.name.removeprefix(_RETIRED))] = (
                        _retired_size(scan, path)
                    )
                    continue
                entry = _read_node(scan, kind, path, staging)
                if entry is None:
                    unsealed += 1
                    continue
                node, _ = entry
                if staging:
                    staged.append(node)
                else:
                    nodes[node.artifact_id] = node
            except (
                OSError,
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                scan.issue(path, "invalid_stage_metadata")
    return nodes, staged, retired, unsealed


def _graph(scan, nodes, staged):
    def edges(node):
        children = set()
        problems = set()
        for reference in node.references:
            child = nodes.get(reference.target)
            if child is None:
                if reference.required:
                    problems.add("missing_typed_dependency")
                continue
            if any(
                member not in child.files
                or any(
                    child.files[member].get(key) != value
                    for key, value in entry.items()
                )
                for member, entry in reference.files.items()
            ):
                problems.add("typed_dependency_pin_mismatch")
            children.add(reference.target)
        return children, problems

    observed = {artifact_id: edges(node) for artifact_id, node in nodes.items()}
    dependencies = {
        artifact_id: children for artifact_id, (children, _) in observed.items()
    }
    roots = {
        artifact_id
        for artifact_id, node in nodes.items()
        if not KINDS[node.kind].cache_shard
    }
    protected = set(roots)
    pending = list(roots)
    references = Counter()

    def retain(child):
        if child not in protected:
            protected.add(child)
            pending.append(child)

    def check(node, children, problems):
        for reason in sorted(problems):
            scan.issue(node.path, reason)
        for child in children:
            references[child] += 1
            retain(child)

    # Digest-only source pins have no local path. Conservatively retain every
    # matching copy with one lookup per shard instead of expanding every pair.
    source_pins = Counter(
        digest
        for node in [*(nodes[key] for key in roots), *staged]
        for digest in set(node.source_digests)
    )
    for artifact_id, node in nodes.items():
        if KINDS[node.kind].cache_shard:
            count = sum(
                source_pins[digest]
                for digest in {entry["sha256"] for entry in node.files.values()}
            )
            if count:
                references[artifact_id] += count
                retain(artifact_id)
    for node in staged:
        check(node, *edges(node))
    while pending:
        current = pending.pop()
        check(nodes[current], *observed[current])
    return dependencies, roots, protected, references


@dataclass(slots=True)
class Inventory:
    scan: RetentionScan
    nodes: dict[str, Node]
    retired: dict[str, int]
    unsealed: int
    dependencies: dict[str, set[str]]
    roots: set[str]
    protected: set[str]
    references: Counter

    @classmethod
    def read(cls, root, *, owned=False):
        scan = RetentionScan(root, owned=owned)
        try:
            nodes, staged, retired, unsealed = _scan_store(scan)
            dependencies, roots, protected, references = _graph(scan, nodes, staged)
            scan.check_generations()
        except OSError as error:
            scan.interrupted(error)
            return cls(scan, {}, {}, 0, {}, set(), set(), Counter())
        scan.published_manifests = len(nodes)
        scan.staging_manifests = len(staged)
        return cls(
            scan, nodes, retired, unsealed, dependencies, roots, protected, references
        )

    def row(self, artifact_id):
        node = self.nodes.get(artifact_id)
        protected = artifact_id in self.protected
        exists = node is not None or artifact_id in self.retired
        return {
            "artifact_id": artifact_id,
            "kind": artifact_id.split("/", 1)[0],
            "bytes": (node.size if node is not None else 0)
            + self.retired.get(artifact_id, 0),
            "action": (
                "unknown"
                if self.scan.problem_count
                else "protected"
                if protected
                else "prune"
                if exists
                else "missing"
            ),
            "published_root": artifact_id in self.roots,
            "retained_references": self.references[artifact_id],
            "dependency_count": len(self.dependencies.get(artifact_id, ())),
            "interrupted_prune": artifact_id in self.retired,
        }


def typed_inventory(root, *, limit, after=None, unreferenced=False, artifact_kind=None):
    if not 1 <= limit <= 1000:
        raise ValueError("artifact limit must be between 1 and 1000")
    if after is not None:
        parse_id(after)
    if artifact_kind is not None and artifact_kind not in KINDS:
        raise ValueError(f"unknown typed artifact kind: {artifact_kind}")
    inventory = Inventory.read(root)
    ids = inventory.nodes.keys() | inventory.retired.keys()
    reclaimable = (
        ids - inventory.protected if not inventory.scan.problem_count else set()
    )
    visible = reclaimable if unreferenced else ids
    page = nsmallest(
        limit + 1,
        (
            key
            for key in visible
            if (after is None or key > after)
            and (artifact_kind is None or key.startswith(artifact_kind + "/"))
        ),
    )
    rows = []
    for key in page[:limit]:
        dependencies = inventory.dependencies.get(key, ())
        rows.append(
            {
                **inventory.row(key),
                "dependencies": nsmallest(20, dependencies),
            }
        )
    published_bytes = sum(node.size for node in inventory.nodes.values())
    retired_bytes = sum(inventory.retired.values())
    protected_bytes = sum(
        inventory.nodes[key].size + inventory.retired.get(key, 0)
        for key in inventory.protected
    )
    return {
        **inventory.scan.summary(),
        "verification": "advisory",
        "store": str(corpus_cache(root)),
        "published_bytes": published_bytes,
        "retired_bytes": retired_bytes,
        "retained_bytes": protected_bytes if not inventory.scan.problem_count else None,
        "unsealed_staging": inventory.unsealed,
        "reclaimable_bytes": published_bytes + retired_bytes - protected_bytes
        if not inventory.scan.problem_count
        else None,
        "artifacts": rows,
        "next_after": page[limit - 1] if len(page) > limit else None,
    }


def typed_artifact_detail(root, artifact_id, *, offset=0, limit=20):
    """Read one sealed stage's metadata, without hashing or opening its payload."""
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError(
            "artifact detail requires a nonnegative offset and a limit of 1–100"
        )
    kind, key = parse_id(artifact_id)
    cache = corpus_cache(root)
    path = cache / KINDS[kind].directory / key
    scan = RetentionScan(root)
    lineage = reversed([part for part in path.parents if part.is_relative_to(root)])
    if not all(scan.directory(part) for part in (*lineage, path)):
        raise ValueError(f"missing or unsafe typed artifact: {artifact_id}")
    node, receipt = _read_node(scan, kind, path, staged=False)
    scan.check_generations()
    if scan.problem_count:
        raise ValueError(f"typed artifact changed during inspection: {artifact_id}")
    inputs, result = receipt["inputs"], receipt["result"]
    sources = inputs.get("sources", [])
    dependencies = sorted({reference.target for reference in node.references})
    end = offset + limit
    summary_keys = (
        "coverage",
        "partitions",
        "counts",
        "sequences",
        "payload_bytes",
        "scratch_bytes",
        "fidelity",
        "failed_derivations",
        "unresolved",
        "assertions",
        "evidence",
        "matched_documents",
        "memberships",
    )
    return {
        "artifact_id": artifact_id,
        "verification": "advisory",
        "bytes": node.size,
        "file_count": len(node.files),
        "result": {name: result[name] for name in summary_keys if name in result},
        "options": {
            name: inputs["options"][name]
            for name in ("target", "targets", "seed", "qualifiers")
            if name in inputs.get("options", {})
        },
        "source_count": len(sources),
        "sources": sources[offset:end],
        "dependency_count": len(dependencies),
        "dependencies": dependencies[offset:end],
        "next_offset": end if end < max(len(sources), len(dependencies)) else None,
    }


def _remove_retired(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with os.scandir(fd) as members:
            for member in members:
                info = member.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("retired typed shard changed after validation")
                os.unlink(member.name, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    path.rmdir()
    sync_directory(path.parent)


def prune_typed_artifacts(root, *, artifacts, apply=False):
    selected = {key: parse_id(key) for key in sorted(set(artifacts))}
    if not selected:
        raise ValueError("typed prune requires explicit artifact IDs")
    cache = corpus_cache(root)
    result = {"store": str(cache), "payloads": [], "pruned": [], "pruned_bytes": 0}
    retired_any = False
    try:
        with ExitStack() as locks:
            if apply:
                trust = ProofStore(root, read_only=True)
                if not trust.trusted(root):
                    raise ValueError("typed prune requires an owned work root")
                lock = root / "artifacts/corpus/.release-retention.lock"
                if any(
                    (entry.exists() or entry.is_symlink()) and not trust.trusted(entry)
                    for entry in (lock, lock.parent, lock.parent.parent)
                ):
                    raise ValueError("unsafe Corpus retention lock scope")
                locks.enter_context(release_retention_lease(root, prune=True))
            leased = set()
            if apply:
                for key, (kind, digest) in selected.items():
                    if not KINDS[kind].cache_shard:
                        continue
                    path = cache / KINDS[kind].directory / digest
                    retired = path.with_name(_RETIRED + digest)
                    if not any(
                        entry.exists() or entry.is_symlink()
                        for entry in (path, retired)
                    ):
                        continue
                    lock = path.parents[1] / "locks" / digest
                    if any(
                        (entry.exists() or entry.is_symlink())
                        and not trust.trusted(entry)
                        for entry in (lock, lock.parent, lock.parent.parent)
                    ):
                        raise ValueError("unsafe typed shard writer lock")
                    locks.enter_context(shared_payload_lease(path, wait=False))
                    leased.add(key)
            inventory = Inventory.read(root, owned=True)
            result.update(inventory.scan.summary())
            result["payloads"] = rows = [inventory.row(key) for key in selected]
            if inventory.scan.problem_count:
                return {
                    **result,
                    "status": "blocked" if apply else "preview",
                    "reason": "typed reference scan is incomplete",
                }
            if not apply:
                return {**result, "status": "preview"}

            if any(
                row["action"] == "prune" and row["artifact_id"] not in leased
                for row in rows
            ):
                raise ValueError(
                    "typed prune candidate appeared without its writer lease"
                )
            inventory.scan.check_generations()
            if inventory.scan.problem_count:
                raise ValueError("typed reference scan changed before prune")

            for row in rows:
                if row["action"] != "prune":
                    continue
                key = row["artifact_id"]
                kind, digest = selected[key]
                path = cache / KINDS[kind].directory / digest
                retired = path.with_name(_RETIRED + digest)
                if key in inventory.retired:
                    retired_any = True
                    _remove_retired(retired)
                    result["pruned_bytes"] += inventory.retired[key]
                if key in inventory.nodes:
                    rename_noreplace(path, retired)
                    retired_any = True
                    sync_directory(path.parent)
                    _remove_retired(retired)
                    result["pruned_bytes"] += inventory.nodes[key].size
                result["pruned"].append(key)
            return {**result, "status": "pruned"}
    except BlockingIOError:
        reason = "Corpus publication or shard writer holds a retention lease"
    except (OSError, ValueError) as error:
        reason = str(error)
    return {
        **result,
        "status": "partial" if retired_any else "blocked",
        "reason": reason,
    }
