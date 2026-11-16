"""Command-line entry points for immutable corpus workspaces."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from klab.corpus.release_verify import add_workers_flag
from klab.verification_proofs import verification_options
from klab.workdir import resolve_work_dir

_CACHE_HELP = (
    "Operator-local cold storage for raw downloads (flag > KLAB_TRANSPORT_CACHE > none)"
)


def _release_path(value: str, work_root: Path) -> Path:
    if "@" in value and not Path(value).exists():
        corpus_id, digest = value.split("@", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", corpus_id) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("release reference must be <corpus-id>@<full-sha256>")
        return work_root / "corpora" / corpus_id / "releases" / digest
    return Path(value).resolve()


def _verification_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--scratch-dir",
        type=Path,
        help="Directory for disposable verification state; default is WORK/scratch. "
        "Allow 2-3x the release size on fast storage",
    )
    add_workers_flag(command)
    command.add_argument(
        "--cold",
        action="store_true",
        help="Verify from the bytes instead of reusing a signed receipt",
    )


def _canary_sample(args: argparse.Namespace):
    from klab.corpus.large_build import CanarySample

    sizes = {
        name: value
        for name, value in (
            ("files_per_source", args.canary_files),
            ("rows_per_file", args.canary_rows),
            ("bytes_per_source", args.canary_bytes),
        )
        if value is not None
    }
    if not args.canary:
        if sizes:
            raise ValueError(
                "--canary-files, --canary-rows and --canary-bytes need --canary"
            )
        return None
    return CanarySample(**sizes)


def _handle(args: argparse.Namespace) -> None:
    """Run one command with the operator's verification placement and receipts."""
    from klab.corpus.release_verify import verification_settings

    root = resolve_work_dir(args.work_dir)
    with verification_settings(
        scratch_dir=getattr(args, "scratch_dir", None),
        workers=getattr(args, "workers", None),
        echo=sys.stderr,
        **verification_options(root, cold=getattr(args, "cold", False)),
    ):
        _run(args)


def _run(args: argparse.Namespace) -> None:
    from klab.corpus import release as publication
    from klab.corpus.acquisition import acquire
    from klab.corpus.export import export_release
    from klab.corpus.pipeline import build
    from klab.corpus.project import load_project

    root = resolve_work_dir(args.work_dir)
    command = args.corpus_command
    if getattr(args, "transport_cache", None) is not None:
        from klab.corpus.transport_cache import (
            TRANSPORT_CACHE_ENV,
            resolve_transport_cache,
        )

        os.environ[TRANSPORT_CACHE_ENV] = str(
            resolve_transport_cache(args.transport_cache)
        )
    if command in {"cache-fill", "cache-list", "cache-prune"}:
        from klab.corpus.transport_cache import TransportCache, resolve_transport_cache

        cache_root = resolve_transport_cache()
        if cache_root is None:
            raise ValueError(
                "no transport cache: pass --transport-cache or set KLAB_TRANSPORT_CACHE"
            )
        if command == "cache-fill":
            from klab.corpus.acquisition import fill_transport_cache

            result = fill_transport_cache(
                load_project(Path(args.project)),
                root,
                transport_cache=cache_root,
                source_id=args.source_id,
            )
        elif command == "cache-list":
            result = TransportCache(cache_root).entries(verify=args.verify)
        else:
            result = TransportCache(cache_root).prune()
        print(json.dumps(result, sort_keys=True, default=str))
        return
    verification = (
        verification_options(root, cold=getattr(args, "cold", False))
        if command in {"acquire", "build", "freeze", "export"}
        and not (command == "acquire" and args.source_id is not None)
        else {}
    )
    if command == "preview":
        from klab.corpus.preview import preview

        print(
            json.dumps(
                preview(
                    Path(args.declaration), Path(args.input_root), Path(args.output)
                ),
                sort_keys=True,
            )
        )
        return
    if command == "tokenizer-bakeoff":
        from klab.corpus.tokenizer_bakeoff import bakeoff

        print(bakeoff(Path(args.declaration), Path(args.output), work_root=root))
        return
    if command == "budget-init":
        from klab.corpus.transport_budget import TransportBudget

        project = load_project(Path(args.project))
        ledger = TransportBudget.initialize(
            root / "corpora" / project.config.id / "transport-budget.sqlite", project
        )
        print(json.dumps(ledger.receipt(), sort_keys=True))
        return
    if command == "budget-release-gate":
        from klab.corpus.transport_budget import TransportBudget

        project = load_project(Path(args.project))
        ledger = TransportBudget(
            root / "corpora" / project.config.id / "transport-budget.sqlite", project
        )
        print(
            json.dumps(
                ledger.release_gate(Path(args.report), Path(args.review)),
                sort_keys=True,
            )
        )
        return
    if command == "budget-status":
        from klab.corpus.transport_budget import TransportBudget

        project = load_project(Path(args.project))
        ledger = TransportBudget(
            root / "corpora" / project.config.id / "transport-budget.sqlite", project
        )
        print(json.dumps(ledger.receipt(), sort_keys=True))
        return
    if command in (
        "budget-observe-time",
        "budget-observe-storage",
        "budget-observe-network",
    ):
        from klab.corpus.transport_budget import TransportBudget

        project = load_project(Path(args.project))
        ledger = TransportBudget(
            root / "corpora" / project.config.id / "transport-budget.sqlite", project
        )
        if command == "budget-observe-time":
            ledger.observe_time(owner=args.owner, reference=args.approval_reference)
        elif command == "budget-observe-network":
            ledger.observe_network(owner=args.owner, reference=args.approval_reference)
        else:
            ledger.observe_storage(
                owner=args.owner,
                reference=args.approval_reference,
                min_free_bytes=args.min_free_bytes,
            )
        print(json.dumps(ledger.receipt(), sort_keys=True))
        return
    if command == "render-declaration":
        from klab.corpus.declaration_render import render_declaration

        print(
            json.dumps(
                render_declaration(
                    Path(args.template),
                    args.values_json,
                    Path(args.output),
                    root,
                    workspace_baseline=(
                        Path(args.workspace_baseline)
                        if args.workspace_baseline is not None
                        else None
                    ),
                ),
                sort_keys=True,
            )
        )
        return
    if command == "alias-snapshot":
        from klab.corpus.acquisition import alias_verified_snapshot

        project = load_project(Path(args.project))
        print(
            json.dumps(
                alias_verified_snapshot(
                    project,
                    root,
                    source_id=args.source_id,
                    snapshot=Path(args.snapshot),
                ),
                sort_keys=True,
            )
        )
        return
    if command == "admission-draft":
        from klab.corpus.admission_draft import draft_admission_manifest

        project = load_project(Path(args.project))
        result = draft_admission_manifest(
            project,
            root,
            Path(args.template),
            Path(args.policy_document),
            Path(args.output),
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "inspect-admission":
        from klab.corpus.admission_inspection import inspect_admission

        project_path = Path(args.project)
        result = inspect_admission(
            load_project(project_path),
            project_path,
            root,
            Path(args.draft),
            Path(args.policy_document),
            Path(args.selection),
            Path(args.output),
            max_input_bytes=args.max_input_bytes,
            max_excerpt_bytes=args.max_excerpt_bytes,
            max_output_bytes=args.max_output_bytes,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "split-inventory":
        from klab.corpus.split_inventory import write_split_inventory

        project = load_project(Path(args.project))
        print(
            json.dumps(
                write_split_inventory(project, root, Path(args.output)), sort_keys=True
            )
        )
        return
    if command == "freeze-splits":
        from klab.corpus.split_freeze import freeze_splits

        project = load_project(Path(args.project))
        result = freeze_splits(
            project,
            root,
            Path(args.inventory),
            Path(args.clusters),
            Path(args.output),
            recover=args.recover,
            reuse_completed_preparation=getattr(
                args, "reuse_completed_preparation", False
            ),
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "finalize-family-inventory":
        from klab.corpus.split_freeze import finalize_family_inventory

        result = finalize_family_inventory(
            Path(args.release), Path(args.splits), Path(args.output)
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "derive-family-inventory":
        from klab.corpus.declared_family_inventory import derive_family_inventory

        print(json.dumps(derive_family_inventory(Path(args.mixture)), sort_keys=True))
        return
    if command == "audit-protected-lineage":
        from klab.corpus.protected_lineage import audit_protected_lineage

        result = audit_protected_lineage(
            Path(args.prior_release),
            Path(args.candidate_release),
            Path(args.prior_inventory),
            Path(args.candidate_inventory),
            Path(args.profile),
            Path(args.suite),
            Path(args.output),
        )
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "PASS":
            raise ValueError("protected evaluation lineage collision")
        return
    if command in {"materialize-mixture", "verify-mixture"}:
        from klab.corpus.mixture import materialize_mixture, verify_mixture

        result = (
            materialize_mixture(Path(args.declaration), Path(args.output))
            if command == "materialize-mixture"
            else verify_mixture(
                Path(args.declaration), Path(args.output), audit=args.audit
            )
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "complete": True,
                    "release_id": result["release_id"],
                    "receipt_path": str(Path(args.output).resolve() / "receipt.json"),
                    "train_tokens_path": str(
                        Path(args.output).resolve() / "train.tokens.ids"
                    ),
                    "records": result["records"],
                    "actual_target_tokens": result["actual_target_tokens"],
                    **{
                        key: result[key]
                        for key in ("stratum_exposures", "row_layout")
                        if key in result
                    },
                },
                sort_keys=True,
            )
        )
        return
    if command in {"acquire", "build"}:
        reuse = command == "build" and getattr(
            args, "reuse_completed_preparation", False
        )
        project = load_project(Path(args.project), verify_inputs=not reuse)
        if command == "build":
            canary = _canary_sample(args)
            built = build(
                project,
                root,
                offline=args.offline,
                reuse_completed_preparation=reuse,
                keep_stages=args.keep_stages,
                canary=canary,
                **verification,
            )
            if canary is None:
                print(built)
            else:
                report = json.loads((built / "report.json").read_text())
                print(
                    json.dumps(
                        {
                            "canary": "passed",
                            "build": str(built),
                            "sample": asdict(canary),
                            "source_ledger": report["source_ledger"],
                        },
                        sort_keys=True,
                    )
                )
            return
        from klab.corpus.bounded_metadata import metadata_operation

        # Keep disk-backed receipt rows alive through verification and output.
        # Per-source acquisition retains its existing compact-result path.
        operation = (
            metadata_operation(
                root / "corpora" / project.config.id / "metadata-scratch",
                max_database_bytes=None,
            )
            if args.source_id is None
            else nullcontext()
        )
        if args.summary and not args.require_existing:
            raise ValueError("compact acquisition summary requires --require-existing")
        with operation:
            result = acquire(
                project,
                root,
                offline=args.offline,
                source_id=args.source_id,
                max_new_bytes=args.max_new_bytes,
                min_free_bytes=args.min_free_bytes,
                plan_only=args.plan,
                require_existing=args.require_existing,
                **verification,
            )
            if args.require_existing:
                result = {
                    "status": "existing_snapshots_verified",
                    "complete": True,
                    "project_id": result["project_id"],
                    "source_count": len(result["sources"]),
                    "lock_path": str(
                        root / "corpora" / project.config.id / "acquisition.json"
                    ),
                }
            encoder = json.JSONEncoder(sort_keys=True, default=str)
            for part in encoder.iterencode(result):
                sys.stdout.write(part)
            sys.stdout.write("\n")
        return
    if command == "freeze":
        print(
            publication.freeze(
                Path(args.build),
                root,
                reuse_completed_preparation=args.reuse_completed_preparation,
                **verification,
            )
        )
        return
    if command == "verify-export":
        from klab.config.loading import load_config
        from klab.corpus.export import verify_release_export

        config = load_config(Path(args.config))
        print(json.dumps(verify_release_export(config.dataset), sort_keys=True))
        return
    release = _release_path(args.release, root)
    if command == "measure-tokens":
        from klab.corpus.release import _verification_operation
        from klab.corpus.token_denominator import measure_source_tokens

        # Explicit evidence reuse accepts only the reviewed, Git-bound v5 cold
        # record. It is not authentication supplied by an arbitrary caller SHA.
        raw_release = Path(args.release)
        is_reference = "@" in args.release and not (
            raw_release.exists() or raw_release.is_symlink()
        )
        with _verification_operation():
            result = measure_source_tokens(
                release if is_reference else raw_release,
                Path(args.tokenizer),
                Path(args.policy),
                Path(args.output),
                tokenizer_origin_release=(
                    _release_path(args.tokenizer_origin_release, root)
                    if args.tokenizer_origin_release is not None
                    else None
                ),
                family_inventory=(
                    Path(args.family_inventory)
                    if args.family_inventory is not None
                    else None
                ),
                batch_documents=args.batch_documents,
                batch_source_bytes=args.batch_source_bytes,
            )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "export":
        print(
            export_release(
                release,
                args.view,
                Path(args.base_run_config),
                args.vocab_size,
                root,
                min_frequency=args.min_frequency,
                **verification,
            )
        )
        return
    if command in {"sample", "review"}:
        result = getattr(publication, command)(
            release,
            domain=args.domain,
            kind=args.kind,
            status=args.status,
            limit=args.limit,
        )
    elif command == "lineage":
        result = publication.lineage(release, args.record_id)
    elif command == "describe":
        result = publication.describe(release, tokenizer=args.tokenizer)
    elif command == "publication":
        from klab.corpus.publication import publication_manifest

        result = publication_manifest(release)
    elif command == "sources":
        result = publication.sources(release)
    elif command == "audit":
        result = publication.audit(release)
    elif command == "near-duplicates":
        from klab.corpus.near_duplicates import audit_release

        caps = {
            name: value
            for name, value in (
                ("max_heldout_documents", args.max_heldout_documents),
                ("max_heldout_shingles", args.max_heldout_shingles),
                ("max_candidates", args.max_candidates),
                ("boilerplate_heldout_documents", args.boilerplate_heldout_documents),
                ("boilerplate_sample_documents", args.boilerplate_sample_documents),
                ("boilerplate_sample_stride", args.boilerplate_sample_stride),
                ("boilerplate_min_share", args.boilerplate_min_share),
            )
            if value is not None
        }
        result = audit_release(
            release,
            checkpoint_dir=(args.scratch_dir or root / "scratch") / "near-duplicates",
            checkpoint_every=args.checkpoint_every,
            workers=min(args.workers or 8, os.cpu_count() or 1),
            shard_bytes=args.shard_bytes,
            template_aware=args.template_aware,
            **caps,
        )
    else:
        result = publication.consumers(release, args.runs_dir)
    print(json.dumps(result, sort_keys=True))


def add_commands(commands: argparse._SubParsersAction) -> None:
    corpus = commands.add_parser(
        "corpus", help="Acquire, derive, freeze and inspect corpus releases"
    )
    sub = corpus.add_subparsers(dest="corpus_command", required=True)
    command = sub.add_parser(
        "preview", help="Preview pinned JSONL bytes without admission or copies"
    )
    command.add_argument("declaration")
    command.add_argument("--input-root", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_handle)
    for name in ("acquire", "build"):
        command = sub.add_parser(name)
        command.add_argument("project")
        command.add_argument("--offline", action="store_true")
        command.add_argument("--transport-cache", type=Path, help=_CACHE_HELP)
        command.add_argument(
            "--cold",
            action="store_true",
            help="Hash every snapshot from its bytes instead of reusing a signed receipt",
        )
        if name == "build":
            command.add_argument(
                "--reuse-completed-preparation",
                action="store_true",
                help="Reuse completed local preparation and record deferred source re-audit",
            )
            command.add_argument(
                "--canary",
                action="store_true",
                help="Run the whole pipeline on a small deterministic sample into "
                "WORK/scratch/canary, never into builds, to surface policy and data failures early",
            )
            for flag, meaning in (
                ("canary-files", "rights-eligible files per source"),
                ("canary-rows", "rows of each JSONL file"),
                ("canary-bytes", "bytes of files per source"),
            ):
                command.add_argument(
                    f"--{flag}",
                    type=int,
                    metavar="N",
                    help=f"With --canary: sample at most N {meaning}",
                )
            command.add_argument(
                "--keep-stages",
                type=int,
                default=0,
                metavar="N",
                help="Keep the newest N index and dedup checkpoints after a successful build (default 0)",
            )
        if name == "acquire":
            command.add_argument(
                "--summary",
                action="store_true",
                help="Print a compact result for existing-snapshot finalization",
            )
            command.add_argument(
                "--require-existing",
                action="store_true",
                help="Finalize only from already verified snapshots; never fetch or reconstruct missing sources",
            )
            command.add_argument(
                "--source-id",
                help="Acquire one exact pinned-transfer source declaration",
            )
            command.add_argument(
                "--max-new-bytes",
                type=int,
                help="Remaining native path-growth allowance for this step",
            )
            command.add_argument(
                "--min-free-bytes",
                type=int,
                help="Required filesystem free-space floor",
            )
            command.add_argument(
                "--plan",
                action="store_true",
                help="Verify source-step admission without writes or network",
            )
        command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "budget-init", help="Start one durable pinned-transfer transport attempt"
    )
    command.add_argument("project")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "budget-release-gate",
        help="Record separately authorized review of the initial transfer feasibility gate",
    )
    command.add_argument("project")
    command.add_argument("--report", required=True)
    command.add_argument("--review", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser("budget-status", help="Read the existing transport budget")
    command.add_argument("project")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "cache-fill",
        help="Download every declared body into the transport cache; make no snapshot",
    )
    command.add_argument("project")
    command.add_argument("--source-id", help="Fill one source only")
    command.add_argument("--transport-cache", type=Path, help=_CACHE_HELP)
    command.set_defaults(handler=_handle)
    command = sub.add_parser("cache-list", help="List transport cache entries")
    command.add_argument(
        "--verify", action="store_true", help="Re-hash every entry and report its state"
    )
    command.add_argument("--transport-cache", type=Path, help=_CACHE_HELP)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "cache-prune",
        help="Remove corrupt entries, unreferenced bodies and stale scratch; never a valid entry",
    )
    command.add_argument("--transport-cache", type=Path, help=_CACHE_HELP)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "budget-observe-time",
        help="Record elapsed transport time without an overall expiry for this corpus ledger",
    )
    command.add_argument("project")
    command.add_argument("--owner", required=True)
    command.add_argument("--approval-reference", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "budget-observe-network",
        help="Observe local network bytes and explicit retries without cumulative quotas",
    )
    command.add_argument("project")
    command.add_argument("--owner", required=True)
    command.add_argument("--approval-reference", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "budget-observe-storage",
        help="Observe corpus storage usage without a retained or cumulative quota",
    )
    command.add_argument("project")
    command.add_argument("--owner", required=True)
    command.add_argument("--approval-reference", required=True)
    command.add_argument("--min-free-bytes", type=int, required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "render-declaration", help="Render one bounded JSON/YAML declaration"
    )
    command.add_argument("--template", required=True)
    command.add_argument("--values-json", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--workspace-baseline")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "alias-snapshot", help="Cold-verify and alias one immutable retained snapshot"
    )
    command.add_argument("project")
    command.add_argument("--source-id", required=True)
    command.add_argument("--snapshot", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "admission-draft",
        help="Draft complete conservative decisions from verified acquired snapshots",
    )
    command.add_argument("project")
    command.add_argument("--template", required=True)
    command.add_argument("--policy-document", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "inspect-admission",
        help="Inspect a bounded deterministic sample before admission",
    )
    command.add_argument("project")
    for flag in ("draft", "policy-document", "selection", "output"):
        command.add_argument("--" + flag, required=True)
    command.add_argument("--max-input-bytes", type=int, required=True)
    command.add_argument("--max-excerpt-bytes", type=int, required=True)
    command.add_argument("--max-output-bytes", type=int, required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "split-inventory", help="Record pre-build IDs from verified source snapshots"
    )
    command.add_argument("project")
    command.add_argument("--output", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "finalize-family-inventory",
        help="Bind kept release documents to reviewed, frozen family assignments",
    )
    command.add_argument("release")
    command.add_argument("--splits", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "derive-family-inventory",
        help="Write a mixture's family inventory for a release with declared splits",
    )
    command.add_argument("mixture")
    command.add_argument("--json", action="store_true")
    add_workers_flag(command)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "audit-protected-lineage",
        help="Cold-check protected evaluation documents against a candidate release",
    )
    for field in (
        "prior-release",
        "candidate-release",
        "prior-inventory",
        "candidate-inventory",
        "profile",
        "suite",
        "output",
    ):
        command.add_argument(f"--{field}", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "freeze-splits", help="Freeze reviewed admission-aware family assignments"
    )
    command.add_argument("project")
    command.add_argument("--inventory", required=True)
    command.add_argument("--clusters", required=True)
    command.add_argument(
        "--recover",
        action="store_true",
        help="Validate identical evidence and publish missing split bundle members",
    )
    command.add_argument(
        "--reuse-completed-preparation",
        action="store_true",
        help="Stream completed inventory and reviewed admission; record deferred source-body reconstruction",
    )
    command.add_argument("--output", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser("freeze")
    command.add_argument("build")
    _verification_flags(command)
    command.add_argument(
        "--reuse-completed-preparation",
        action="store_true",
        help="Freeze authenticated completed local payload in place without a data copy",
    )
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "verify-export", help="Cold-verify one run configuration's shared corpus export"
    )
    command.add_argument("config")
    command.set_defaults(handler=_handle)
    command = sub.add_parser("tokenizer-bakeoff")
    command.add_argument("declaration")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_handle)
    for name in ("materialize-mixture", "verify-mixture"):
        command = sub.add_parser(
            name, help="Materialize or authenticate an exact token-ID source mixture"
        )
        command.add_argument("declaration")
        command.add_argument("--output", required=True)
        command.add_argument("--json", action="store_true")
        if name == "verify-mixture":
            command.add_argument(
                "--audit",
                action="store_true",
                help="Explicitly reproduce the mixture schedule",
            )
        command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "measure-tokens", help="Measure authenticated distinct source-domain tokens"
    )
    command.add_argument("release")
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--tokenizer-origin-release")
    command.add_argument("--family-inventory")
    command.add_argument("--policy", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--batch-documents", type=int, default=256)
    command.add_argument("--batch-source-bytes", type=int, default=1_048_576)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    for name in (
        "describe",
        "sources",
        "audit",
        "near-duplicates",
        "sample",
        "review",
        "lineage",
        "consumers",
        "export",
        "publication",
    ):
        command = sub.add_parser(name)
        command.add_argument("release")
        _verification_flags(command)
        if name in {"sample", "review"}:
            command.add_argument("--domain")
            command.add_argument("--kind")
            command.add_argument("--status")
            command.add_argument("--limit", type=int, default=10)
        if name == "lineage":
            command.add_argument("record_id")
        if name == "describe":
            command.add_argument("--tokenizer", type=Path)
        if name == "consumers":
            command.add_argument("--runs-dir", required=True, type=Path)
        if name == "export":
            command.add_argument("--view", choices=("lm", "chat"), required=True)
            command.add_argument("--base-run-config", required=True)
            command.add_argument("--vocab-size", required=True, type=int)
            command.add_argument("--min-frequency", type=int, default=1)
        if name == "near-duplicates":
            for flag in (
                "max-heldout-documents",
                "max-heldout-shingles",
                "max-candidates",
                "boilerplate-heldout-documents",
                "boilerplate-sample-documents",
                "boilerplate-sample-stride",
            ):
                command.add_argument("--" + flag, type=int)
            command.add_argument(
                "--boilerplate-min-share",
                type=float,
                help="Share of a pair's shared shingles that must be boilerplate to class it template_only",
            )
            command.add_argument(
                "--template-aware",
                action="store_true",
                help="Set aside boilerplate shingles and report pairs similar only through them as template_only",
            )
            command.add_argument(
                "--shard-bytes",
                type=int,
                default=256 << 20,
                help="Approximate size of a shard, the unit of parallel work and of resuming",
            )
            command.add_argument(
                "--checkpoint-every",
                type=int,
                default=200_000,
                metavar="N",
                help="Save progress every N training documents to SCRATCH/near-duplicates so a killed run resumes",
            )
        command.set_defaults(handler=_handle)
