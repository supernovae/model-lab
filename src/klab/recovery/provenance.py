"""Read-only declaration closure and commit-before-compute provenance.

Operational locations and unrelated dirty files do not define scientific intent.
Git HEAD, not the index or a branch name, is the committed-byte authority.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from klab.declaration_context import RecipeContext, safe_path
from klab.documents import read_document

DeclarationKind = Literal["campaign", "experiment", "corpus", "recovery"]


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def repository_root(path: Path) -> Path | None:
    ancestor = Path(path).absolute()
    while not ancestor.is_dir() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        return Path(
            _git(ancestor, "rev-parse", "--show-toplevel").decode().strip()
        ).resolve()
    except OSError, subprocess.CalledProcessError, UnicodeError:
        return None


def declaration_reference(
    source: Path, reference: str, *, context: RecipeContext | None = None
) -> Path:
    """Resolve a safe local or repository-relative declaration, without traversal."""
    if context is not None:
        return context.reference(source, reference)
    source = Path(source).absolute()
    root = repository_root(source) or source.parent.resolve()
    local = safe_path(source.parent, reference)
    target = local if local.exists() else safe_path(root, reference)
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError(f"declaration outside repository: {target}")
    return target


def declaration_paths(
    source: Path, kind: DeclarationKind, *, include_local_inputs: bool = True
) -> tuple[Path, ...]:
    """Close only authored scientific references; never inventory generated stores."""
    from klab.campaign.plan import load_campaign
    from klab.corpus.project import load_project
    from klab.experiments.plan import load_plan

    source = Path(source).absolute()
    root = repository_root(source) or source.parent.resolve()
    found: set[Path] = set()

    def add(path: Path) -> bool:
        path = Path(path).absolute()
        if not path.resolve(strict=False).is_relative_to(root):
            raise ValueError(f"declaration outside repository: {path}")
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ValueError(f"symlinked declaration path: {component}")
            if component == root:
                break
        if path in found:
            return False
        found.add(path)
        return True

    def reference(document: Path, value: str, nested: str | None = None) -> None:
        target = declaration_reference(document, value)
        if nested:
            walk(target, nested)
        else:
            if not add(target):
                return
            if target.is_file() and target.suffix in {".yaml", ".yml", ".json"}:
                raw = read_document(target)
                # Suite sources are declarations/evidence references, not datasets.
                for evaluation in (
                    raw.get("evaluations", [])
                    if "evaluation_suite_version" in raw
                    else []
                ):
                    pinned = evaluation.get("source")
                    if evaluation.get("kind") == "surface_review":
                        if isinstance(pinned, str):
                            declaration_reference(target, pinned)
                        continue
                    if (
                        evaluation.get("kind") == "evidence_reference"
                        and isinstance(pinned, str)
                        and not declaration_reference(target, pinned).is_file()
                    ):
                        continue
                    if isinstance(pinned, str):
                        reference(target, pinned)
                    elif isinstance(pinned, dict) and isinstance(
                        pinned.get("path"), str
                    ):
                        reference(target, pinned["path"])
                if raw.get("lock_version") == 1:
                    sidecar = safe_path(
                        target.parent, f"{target.stem}.availability.json"
                    )
                    add(sidecar)
                    if sidecar.is_file():
                        read_document(sidecar)
                if "family_version" in raw:
                    from klab.family.manifest import load_family

                    family = load_family(target)
                    for node in family.nodes:
                        if node.recovery_manifest:
                            reference(target, node.recovery_manifest, "recovery")
                        for receipt in node.lifecycle_receipts:
                            reference(target, receipt)

    def walk(document: Path, document_kind: str) -> None:
        if not add(document):
            return
        if not document.is_file():
            return
        if document_kind == "corpus":
            from klab.corpus.project import ProjectConfig, SourceDeclaration

            config = ProjectConfig.model_validate(read_document(document))
            declarations = [
                safe_path(document.parent, name)
                for name in (
                    *config.sources,
                    *config.transforms,
                    config.splits,
                    config.release,
                )
            ]
            for path in declarations:
                add(path)
                if path.is_file():
                    read_document(path)
            if all(path.is_file() for path in declarations):
                sources = load_project(
                    document, verify_inputs=include_local_inputs
                ).sources
            else:
                sources = tuple(
                    SourceDeclaration.model_validate(
                        read_document(safe_path(document.parent, name))
                    )
                    for name in config.sources
                    if safe_path(document.parent, name).is_file()
                )
            for declaration in sources:
                if include_local_inputs and declaration.kind == "local":
                    for entry in declaration.acquisition.files:
                        add(safe_path(document.parent, entry.path))
            return
        if document_kind == "experiment":
            plan = load_plan(document)
            if isinstance(plan.base_run, str):
                reference(document, plan.base_run)
            for variant in plan.corpus_variants:
                reference(document, variant.project, "corpus")
            suite = getattr(plan, "evaluation_suite", None)
            if suite:
                reference(document, suite)
            for evaluation in plan.evaluations:
                if evaluation.cases:
                    reference(document, evaluation.cases)
            for artifact in plan.artifacts.values():
                if artifact.kind in {"capability_card", "prompt_set"} and artifact.path:
                    reference(document, artifact.path)
            return
        if document_kind == "campaign":
            plan = load_campaign(document)
            recovery = getattr(plan, "recovery", None)
            if recovery:
                reference(document, recovery, "recovery")
            for stage in plan.stages:
                if stage.kind == "corpus_release":
                    reference(document, stage.project, "corpus")
                elif stage.kind == "experiment_plan":
                    reference(document, stage.source, "experiment")
                elif stage.kind == "dataset_snapshot":
                    # External immutable locks are operational bindings; Campaign
                    # adds their content digest to its retained closure directly.
                    if not Path(stage.lock).is_absolute():
                        reference(document, stage.lock)
                elif stage.kind in {"tokenizer_train", "data_prepare"}:
                    reference(document, stage.config)
                for name in ("suite", "policy", "panel"):
                    value = getattr(stage, name, None)
                    if isinstance(value, str):
                        reference(document, value)
            return
        if document_kind == "recovery":
            from klab.recovery.manifest import load_manifest

            manifest = load_manifest(document)
            for name in ("evaluation_suite", "readiness_policy", "family"):
                value = getattr(manifest, name, None)
                if value:
                    reference(document, value)
            for evidence in manifest.evidence:
                reference(document, evidence)
            requirement = manifest.runtime_requirement
            if requirement is not None:
                value = getattr(requirement, "profile", None) or getattr(
                    requirement, "runtime_profile", None
                )
                if value:
                    reference(document, value)
            for step in manifest.steps:
                for name, nested in (
                    ("project", "corpus"),
                    ("plan", "experiment"),
                    ("base_run", None),
                    ("config", None),
                    ("run_config", None),
                ):
                    value = getattr(step, name, None)
                    if isinstance(value, str):
                        reference(document, value, nested)
            return
        raise ValueError(f"unknown declaration kind: {document_kind}")

    walk(source, kind)
    return tuple(sorted(found, key=lambda path: path.relative_to(root).as_posix()))


def git_provenance(
    paths: tuple[Path, ...], repo_root: Path | None = None
) -> dict[str, Any]:
    """Authenticate exact closure bytes against HEAD without Git writes."""
    paths = tuple(Path(path).absolute() for path in paths)
    root = repository_root(repo_root or paths[0]) if paths else None
    result: dict[str, Any] = {
        "status": "UNKNOWN",
        "source_commit": None,
        "declarations": [],
    }
    if paths:
        anchor = root or Path(os.path.commonpath([path.parent for path in paths]))
        for path in paths:
            try:
                actual = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file()
                    else None
                )
            except OSError:
                actual = None
            result["declarations"].append(
                {
                    "path": path.relative_to(anchor).as_posix(),
                    "sha256": actual,
                    "head_sha256": None,
                    "status": "UNKNOWN",
                }
            )
    if root is None:
        return result
    try:
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        if len(commit) not in {40, 64}:
            return result
        result["source_commit"] = commit
        result["declarations"] = []
        statuses: set[str] = set()
        for path in sorted(paths, key=lambda item: item.as_posix()):
            if not path.resolve(strict=False).is_relative_to(root) or path.is_symlink():
                raise ValueError(f"unsafe declaration path: {path}")
            relative = path.relative_to(root).as_posix()
            actual = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None
            )
            try:
                _git(root, "ls-files", "--error-unmatch", "--", f":(literal){relative}")
                head_bytes = _git(root, "show", f"HEAD:{relative}")
            except subprocess.CalledProcessError:
                head_sha = None
                status = "UNTRACKED_DECLARATION"
            else:
                head_sha = hashlib.sha256(head_bytes).hexdigest()
                changed = _git(
                    root,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    f":(literal){relative}",
                )
                status = (
                    "DIRTY_DECLARATION"
                    if changed or actual != head_sha
                    else "CLEAN_AND_COMMITTED"
                )
            statuses.add(status)
            result["declarations"].append(
                {
                    "path": relative,
                    "sha256": actual,
                    "head_sha256": head_sha,
                    "status": status,
                }
            )
        result["status"] = next(
            (
                status
                for status in ("UNTRACKED_DECLARATION", "DIRTY_DECLARATION")
                if status in statuses
            ),
            "CLEAN_AND_COMMITTED",
        )
    except OSError, subprocess.CalledProcessError, UnicodeError, ValueError:
        result["status"] = "UNKNOWN"
    return result


def declaration_preflight(
    source: Path, kind: DeclarationKind, allow_uncommitted: bool = False
) -> dict[str, Any]:
    provenance = git_provenance(declaration_paths(source, kind))
    provenance["allow_uncommitted_declaration"] = bool(allow_uncommitted)
    if provenance["status"] != "CLEAN_AND_COMMITTED" and not allow_uncommitted:
        raise ValueError(
            f"{provenance['status']}: scientific declaration must be committed before compute"
        )
    return provenance


def verify_source_commit(
    source: Path, source_commit: str, *, missing_local: frozenset[Path] = frozenset()
) -> dict[str, Any]:
    """Pin earlier scientific inputs, not later explicitly inventoried output evidence."""
    from klab.family.manifest import load_family
    from klab.recovery.manifest import load_manifest

    source = Path(source).absolute()
    manifest = load_manifest(source)
    outputs = {
        declaration_reference(source, reference) for reference in manifest.evidence
    }
    if manifest.family:
        family_path = declaration_reference(source, manifest.family)
        family = load_family(family_path)
        outputs.update(
            declaration_reference(family_path, reference)
            for node in family.nodes
            for reference in node.lifecycle_receipts
        )
    output_formats = {
        "scientific-evidence-reference-v1",
        "model-lifecycle-receipt-v1",
        "model-review-receipt-v1",
        "evaluation-index-reference-v1",
        "model-readiness-reference-v1",
        "evaluation-index-v1",
        "model-readiness-result-v1",
        "klab-recovery-receipt-v1",
        "experiment-preparation-v1",
    }
    sidecars: set[Path] = set()
    for output in outputs:
        if output.is_file():
            record = read_document(output)
            if (
                record.get("format") not in output_formats
                and record.get("lock_version") != 1
            ):
                raise ValueError(f"INVALID_EVIDENCE_REFERENCE: {output}")
            if record.get("lock_version") == 1:
                sidecars.add(
                    safe_path(output.parent, f"{output.stem}.availability.json")
                )
    outputs.update(sidecars)
    paths = tuple(
        path
        for path in declaration_paths(source, "recovery")
        if path != source and path not in outputs
    )
    root = repository_root(source)
    if root is None:
        raise ValueError("UNKNOWN: cannot verify pinned scientific input commit")
    for path in paths:
        if path in missing_local and not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            expected = _git(root, "show", f"{source_commit}:{relative}")
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"SOURCE_COMMIT_MISMATCH: {relative} absent from {source_commit}"
            ) from error
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError(
                f"SOURCE_COMMIT_MISMATCH: {relative} actual={hashlib.sha256(actual).hexdigest()} expected={hashlib.sha256(expected).hexdigest()}"
            )
    return {
        "source_commit": source_commit,
        "verified_paths": [
            path.relative_to(root).as_posix()
            for path in paths
            if path not in missing_local or path.exists()
        ],
        "missing_paths": [
            path.relative_to(root).as_posix()
            for path in paths
            if path in missing_local and not path.exists()
        ],
    }
