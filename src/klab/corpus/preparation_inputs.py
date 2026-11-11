"""Local completion input bindings without importing preparation execution.

Packed-byte artifact verification uses these filesystem dependencies only.
Build/release semantic verification separately authenticates executed completion.
"""

from pathlib import Path


def local_scope(manifest):
    identity = manifest.get("identity", manifest.get("build_identity", {}))
    scope = identity.get("completed_preparation")
    return (
        scope
        if isinstance(scope, dict)
        and scope.get("policy") == "completed_local_preparation"
        else None
    )


def corpus_work_root(path):
    from klab.verification_proofs import _trusted

    if (
        path.parent.name not in {"builds", "releases"}
        or path.parents[2].name != "corpora"
    ):
        raise ValueError(
            "completed preparation artifact is outside its corpus work root"
        )
    root = path.parents[3]
    if not _trusted(root, root):
        raise ValueError("completed preparation work root is not owned and trusted")
    return root


def local_dependencies(manifest, *, work_root):
    scope = local_scope(manifest)
    if scope is None:
        return None
    if "release_id" in manifest:
        from klab.corpus.release_migration import migrated_dependencies

        migrated = migrated_dependencies(manifest, work_root)
        if migrated is not None:
            return migrated
    result = tuple(
        Path(value)
        for value in {*scope["request"]["inputs"], *scope["source_bindings"]}
    )
    if any(
        not p.is_absolute()
        or ".." in p.parts
        or not p.is_relative_to(work_root)
        or any(q.is_symlink() for q in (p, *p.parents))
        for p in result
    ):
        raise ValueError("completed preparation dependency escapes its owned work root")
    return result
