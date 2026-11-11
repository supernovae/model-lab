"""Portable explanations of declared corpus digests, never proofs of absent bytes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from klab.campaign.state import read_canonical
from klab.canonical import canonical_digest
from klab.corpus.project import Project, safe_name

_SCOPE = "declared_digest_payloads_only"


def explain_corpus_identity(release: Path, project: Project) -> dict[str, Any]:
    """Capture metadata from an already verified release without copying source data.

    Acquisition normalizes local input paths out of declaration identities. Preserve
    that exact payload, not new hashes that depend on the workspace's location.
    """
    from klab.corpus.acquisition import _identity_declaration

    manifest = read_canonical(release / "manifest.json")
    snapshots = {}
    for row in manifest["snapshots"]:
        snapshot = read_canonical(
            release.parent.parent
            / "snapshots"
            / row["source_id"]
            / row["sha256"]
            / "manifest.json"
        )
        payload = {
            key: snapshot[key] for key in ("declaration_sha256", "adapter", "files")
        }
        if snapshot["declaration"]["kind"] == "wikimedia_dump":
            payload["retrieval"] = snapshot["retrieval"]
        snapshots[row["source_id"]] = payload
    result = {
        "format": "corpus-identity-explanation-v1",
        "verification_scope": _SCOPE,
        "release": manifest,
        "source_declarations": {
            source.id: _identity_declaration(source) for source in project.sources
        },
        "snapshot_payloads": snapshots,
    }
    verify_corpus_identity(result)
    return result


def verify_corpus_identity(value: Any) -> dict[str, Any]:
    """Check original digest payloads and their bindings using metadata only.

    This cannot verify source/release file contents, reproduce a build, establish
    scientific validity, or authenticate a fully replaced set of identities. A
    trusted release SHA must come from an independent declaration when needed.
    """
    try:
        return _verify(value)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("malformed corpus identity explanation") from error


def _verify(value: Any) -> dict[str, Any]:
    import re

    def sha(value: Any) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid corpus identity SHA-256")

    def files(inventory: dict[str, Any]) -> None:
        if not isinstance(inventory, dict):
            raise TypeError("invalid corpus file provenance")
        for name, entry in inventory.items():
            safe_name(name)
            sha(entry["sha256"])
            if type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError("invalid corpus file size")

    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "format",
            "verification_scope",
            "release",
            "source_declarations",
            "snapshot_payloads",
        }
        or value["format"] != "corpus-identity-explanation-v1"
        or value["verification_scope"] != _SCOPE
    ):
        raise ValueError("invalid corpus identity explanation schema/scope")
    release = value["release"]
    sha(release["release_id"])
    if (
        release["schema_version"] != 1
        or canonical_digest(
            {key: item for key, item in release.items() if key != "release_id"}
        )
        != release["release_id"]
    ):
        raise ValueError("corpus release identity payload mismatch")
    build = release["build_identity"]
    if canonical_digest(build) != release["build_id"]:
        raise ValueError("corpus build identity payload mismatch")
    if release["corpus_id"] != build["project_id"] or (
        build["project"]["id"] != build["project_id"]
    ):
        raise ValueError("corpus project identity mismatch")
    files(release["files"])
    for key in (
        "implementation_sha256",
        "schema_implementation_sha256",
        "provenance_implementation_sha256",
    ):
        sha(build[key])
    for key, item in build.items():
        if key.endswith("_implementation_sha256"):
            sha(item)
    declarations = value["source_declarations"]
    if set(declarations) != set(build["lock"]):
        raise ValueError("corpus declaration inventory mismatch")
    pinned = {}
    for source_id, entry in build["lock"].items():
        safe_name(source_id)
        declaration = declarations[source_id]
        if (
            declaration["id"] != source_id
            or canonical_digest(declaration) != entry["declaration_sha256"]
        ):
            raise ValueError("corpus source declaration payload mismatch")
        if entry["snapshot_sha256"] is not None:
            sha(entry["snapshot_sha256"])
            pinned[source_id] = entry["snapshot_sha256"]
    rows = release["snapshots"]
    if (
        len(rows) != len(pinned)
        or {row["source_id"]: row["sha256"] for row in rows} != pinned
        or set(value["snapshot_payloads"]) != set(pinned)
    ):
        raise ValueError("corpus source snapshot mapping mismatch")
    for source_id, payload in value["snapshot_payloads"].items():
        required = {"declaration_sha256", "adapter", "files"}
        if declarations[source_id]["kind"] == "wikimedia_dump":
            required.add("retrieval")
        if set(payload) != required or canonical_digest(payload) != pinned[source_id]:
            raise ValueError("corpus snapshot identity payload mismatch")
        if (
            payload["declaration_sha256"]
            != build["lock"][source_id]["declaration_sha256"]
            or payload["adapter"]["id"] != declarations[source_id]["kind"]
        ):
            raise ValueError("corpus snapshot declaration binding mismatch")
        sha(payload["adapter"]["module_sha256"])
        if (
            not isinstance(payload["adapter"]["version"], str)
            or not payload["adapter"]["version"]
        ):
            raise ValueError("invalid corpus adapter version")
        inventory = {entry["path"]: entry for entry in payload["files"]}
        if len(inventory) != len(payload["files"]):
            raise ValueError("duplicate corpus snapshot file")
        files(inventory)
    return {
        "verification_scope": _SCOPE,
        "project_id": build["project_id"],
        "build_id": release["build_id"],
        "release_id": release["release_id"],
        "source_snapshots": pinned,
        "source_contents_verified": False,
        "release_contents_verified": False,
    }
