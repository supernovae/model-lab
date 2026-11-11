"""Publish small references to already verified evidence, never copy payload bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.campaign.state import publish_immutable, read_canonical, utc_now
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


KINDS = (
    "corpus_release",
    "tokenizer_selection",
    "prepared_data",
    "runtime_probe",
    "experiment_lock",
    "checkpoint",
    "evaluation_index",
)


def _selection(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> tuple[str, str]:
    from klab.corpus.release import verify_release
    from klab.corpus.tokenizer_bakeoff import (
        Declaration,
        _receipt,
        _verify_existing,
        choose_candidate,
    )
    from klab.corpus.tokenizer_bakeoff import (
        _selection as select_documents,
    )

    if path.name != "report.json" or path.is_symlink():
        raise ValueError(
            "tokenizer selection requires the verified bakeoff report.json"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = report["identity"]
    release = Path(identity["release_path"])
    if (
        verify_release(
            release, proof_store=proof_store, verification_mode=verification_mode
        )["release_id"]
        != identity["release_id"]
    ):
        raise ValueError("selection release binding changed")
    spec = Declaration.model_validate(identity["declaration"])
    selected_docs, heldout_docs, train_docs = select_documents(release, spec)
    receipt = _receipt(selected_docs, heldout_docs, train_docs, spec)
    _verify_existing(
        path.parent, identity, receipt, selected_docs, heldout_docs, release
    )
    if not isinstance(report["candidates"], list) or not report["candidates"]:
        raise ValueError("selection has no verified candidates")
    chosen = choose_candidate(report["candidates"], spec.near_best_ratio)
    if report.get("selected_vocab_size") != chosen:
        raise ValueError("selection vocabulary changed")
    candidate = next(
        item for item in report["candidates"] if item["vocab_size"] == chosen
    )
    selected = (
        path.parent / "candidates" / str(candidate["vocab_size"]) / "tokenizer.json"
    )
    if (
        selected.is_symlink()
        or selected.parent.is_symlink()
        or selected.with_name("tokenizer_manifest.json").is_symlink()
        or report["selected_tokenizer"] != str(selected)
        or sha256_file(selected) != candidate["tokenizer_sha256"]
    ):
        raise ValueError("selection receipt does not bind the chosen tokenizer")
    manifest = json.loads(
        selected.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest != candidate["manifest"]
        or manifest.get("corpus_forge_bakeoff") != report["release_binding"]
    ):
        raise ValueError("chosen tokenizer is not bound to bakeoff")
    return candidate[
        "tokenizer_sha256"
    ], f"{identity['release_id']}:{candidate['vocab_size']}"


def export_evidence(
    kind: str,
    target: Path,
    output: Path,
    *,
    source_commit: str,
    declaration_hashes: list[dict[str, str]],
) -> dict[str, Any]:
    """Exclusively publish a verified compact reference; identical replay keeps its timestamp."""
    target = Path(target).absolute()
    if kind == "corpus_release":
        from klab.corpus.release import verify_release

        verified = verify_release(target)
        sha, identifier = (
            verified["release_id"],
            f"{verified['corpus_id']}@{verified['release_id']}",
        )
    elif kind == "tokenizer_selection":
        sha, identifier = _selection(target)
    elif kind == "prepared_data":
        from klab.data.packing import load_prepared_data

        root = target if target.is_dir() else target.parent
        manifest = json.loads((root / "manifest.json").read_text())
        load_prepared_data(
            root,
            byte_enabled=manifest.get("byte_addressing") is not None,
            verification_mode="cold",
        )
        sha = json.loads((root / "manifest.json").read_text())["manifest_sha256"]
        identifier = root.name
    elif kind == "experiment_lock":
        from klab.experiments.lock import open_lock

        sha = open_lock(target, verification_mode="cold").plan_sha256
        identifier = target.stem
    elif kind == "evaluation_index":
        from klab.evaluation.suite import verify_evaluation_index

        verified = verify_evaluation_index(target)
        sha, identifier = (
            verified["index_sha256"],
            str(verified.get("run_id", target.stem)),
        )
    elif kind == "checkpoint":
        from klab.training.checkpoints import CheckpointManager

        report = CheckpointManager(target.parent.parent).verify(target)
        if not report.valid:
            raise ValueError(f"checkpoint verification failed: {report.errors}")
        from klab.training.mlx_checkpoints import strict_json

        sha, identifier = strict_json(target / "manifest.json")["sha256"], target.name
    elif kind == "runtime_probe":
        from klab.experiments.plan import read_document

        observation = read_document(target)
        if (
            not observation
            or "format_version" not in observation
            and "status" not in observation
        ):
            raise ValueError("runtime probe observation has no recognized record shape")
        sha, identifier = sha256_file(target), target.name
    else:
        raise ValueError(f"unknown scientific evidence kind: {kind}")
    binding = {
        "format": "scientific-evidence-reference-v1",
        "kind": kind,
        "identifier": identifier,
        "sha256": sha,
        "source_commit": source_commit,
        "declaration_hashes": declaration_hashes,
        "external_location": str(target),
        "verification_scope": "observation"
        if kind == "runtime_probe"
        else "verified_artifact",
    }
    output = Path(output)
    if output.exists():
        previous = read_canonical(output)
        if {
            key: value for key, value in previous.items() if key != "verified_at"
        } != binding:
            raise ValueError(f"conflicting evidence publication: {output}")
        return previous
    return publish_immutable(output, {**binding, "verified_at": utc_now()})


def _handle(args: argparse.Namespace) -> None:
    from klab.recovery.provenance import (
        declaration_paths,
        git_provenance,
        repository_root,
    )

    repository = repository_root(Path.cwd())
    if repository is None:
        raise ValueError(
            "UNKNOWN: compact evidence publication requires a Git repository"
        )
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    provenance = (
        git_provenance(declaration_paths(args.declaration, args.declaration_kind))
        if args.declaration
        else {"source_commit": commit, "declarations": []}
    )
    if provenance["source_commit"] is None:
        raise ValueError("UNKNOWN: declaration commit unavailable")
    result = export_evidence(
        args.kind,
        args.target,
        args.output,
        source_commit=provenance["source_commit"],
        declaration_hashes=provenance["declarations"],
    )
    print(json.dumps(result, sort_keys=True) if args.json else args.output)


def register_research_evidence_parser(subparsers: argparse._SubParsersAction) -> None:
    evidence = subparsers.add_parser(
        "evidence", help="Publish verified compact scientific references"
    )
    commands = evidence.add_subparsers(dest="evidence_command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--kind", choices=KINDS, required=True)
    export.add_argument("target", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--declaration", type=Path)
    export.add_argument(
        "--declaration-kind",
        choices=("corpus", "experiment", "campaign", "recovery"),
        default="recovery",
    )
    export.add_argument("--json", action="store_true")
    export.set_defaults(handler=_handle)
