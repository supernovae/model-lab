"""Checkpoint-bound, immutable evaluation suites and verified result indexes."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from klab.campaign.plan import safe_path
from klab.campaign.state import publish_immutable, read_canonical, utc_now
from klab.canonical import canonical_digest, canonical_json
from klab.documents import read_document
from klab.engines.base import EngineCapabilityError
from klab.evaluation.suite_models import (  # noqa: F401
    EvaluationSuite,
    SuiteEvaluation,
)
from klab.runtime_profile import RuntimeAuthorization, RuntimeAuthorizationRequired
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import read_manifest, sha256_file
from klab.verification_proofs import ProofStore, VerificationMode, validate_mode

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def load_suite(source: Path) -> EvaluationSuite:
    suite = EvaluationSuite.model_validate(read_document(source))
    for item in suite.evaluations:
        if item.source is not None:
            safe_path(source.parent, item.source)
    return suite


def _source(path: Path, reference: str) -> Path:
    return safe_path(path.parent, reference)


def _checked_file(path: Path, expected: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"missing or changed evaluation evidence: {path}")
    return read_document(path)


def _selected_checkpoint(
    run: Path,
    relative: str,
    expected: str,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    parts = Path(relative).parts
    if (
        len(parts) != 2
        or parts[0] != "checkpoints"
        or parts[1] in {".", ".."}
        or (run / "checkpoints").is_symlink()
        or (run / relative).is_symlink()
        or not (run / relative).is_dir()
        or not _SHA.fullmatch(expected)
    ):
        raise ValueError("unsafe or invalid selected checkpoint")
    candidate = run / relative
    manifest = read_manifest(run / "manifest.json")
    run_sha = canonical_digest(manifest)
    report = CheckpointManager(run, manifest_sha256=run_sha).verify(
        candidate,
        expected_manifest=run_sha,
        require_training_state=False,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    if (
        not report.valid
        or read_document(candidate / "manifest.json").get("sha256") != expected
    ):
        raise ValueError("selected checkpoint verification failed")


def _provenance(source: Path, suite: EvaluationSuite) -> dict[str, Any]:
    """Observe declared evaluator documents; generated human bundles are outputs."""
    try:
        from klab.recovery.provenance import git_provenance

        paths = (
            source,
            *(
                _source(source, item.source)
                for item in suite.evaluations
                if item.source is not None and item.kind != "surface_review"
            ),
        )
        return git_provenance(paths)
    except ImportError, OSError, ValueError:
        return {"status": "UNKNOWN", "source_commit": None, "declarations": []}


class UnsupportedEvidenceReference(ValueError):
    """A source has no supported checkpoint-binding verifier."""


def _index_binding(
    suite_sha256: str,
    checkpoint_sha256: str,
    rows: list[dict[str, Any]],
    evaluation_runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "suite_sha256": suite_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_runtime": evaluation_runtime,
        "evaluations": [
            {
                "id": row["id"],
                "status": row["status"],
                "sha256": row["sha256"],
                "source_sha256": None
                if row["kind"] == "evidence_reference"
                else row["source_sha256"],
            }
            for row in rows
        ],
    }


def _supplied_index(
    reference: Path,
    checkpoint_sha256: str,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> tuple[Path, dict[str, Any]]:
    """Authenticate a checked-in small reference through the existing index verifier."""
    raw = read_document(reference)
    if (
        raw.get("format") != "scientific-evidence-reference-v1"
        or raw.get("kind") != "evaluation_index"
    ):
        raise UnsupportedEvidenceReference("no schema-aware evaluation index reference")
    record = read_canonical(reference)
    target = Path(record["external_location"])
    if not target.exists():
        raise FileNotFoundError(f"missing referenced evaluation index: {target}")
    verified = verify_evaluation_index(
        target, proof_store=proof_store, verification_mode=verification_mode
    )
    if (
        verified["index_sha256"] != record["sha256"]
        or verified["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError(
            "supplied evaluation index does not bind the selected checkpoint and digest"
        )
    return target, verified


def _surface(
    bundle: Path, checkpoint_sha256: str, *, allow_pending: bool = False
) -> tuple[str, dict[str, Any]] | None:
    from klab.evaluation.surface_review import (
        _review,
        open_surface_bundle,
        surface_results,
    )

    view = open_surface_bundle(bundle, private=True)
    if not any(
        isinstance(item, dict) and item.get("checkpoint_sha256") == checkpoint_sha256
        for item in view["provenance"]["source_artifacts"]
    ):
        raise ValueError("sealed Surface Review lacks selected checkpoint binding")
    if allow_pending:
        review, reveal = bundle / "review.json", bundle / "reveal.json"
        if review.is_symlink() or reveal.is_symlink():
            raise ValueError("symlinked Surface Review seal")
        if review.exists():
            _review(bundle)
        elif reveal.exists():
            raise ValueError("Surface Review reveal lacks completed review")
        if not review.exists() or not reveal.exists():
            return None
    return sha256_file(bundle / "review.json"), surface_results(bundle)


def run_suite(
    source: Path,
    run_id: str,
    checkpoint: str,
    runs_dir: Path,
    backend: str | None = None,
    *,
    authorization: RuntimeAuthorization | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> Path:
    """Evaluate one verified generation; unsupported declared evaluators remain unavailable."""
    from klab.evaluation.capabilities import (
        evaluate_capability,
        load_capability_card,
        write_capability_result,
    )
    from klab.evaluation.inference import (
        evaluation_config,
        load_run,
        write_inference_result,
    )

    validate_mode(verification_mode)
    verification = {
        "proof_store": proof_store,
        "verification_mode": verification_mode,
    }
    source, runs_dir = Path(source).resolve(), Path(runs_dir).resolve()
    suite = load_suite(source)
    run = runs_dir / run_id
    loaded = None
    unavailable_backend = None
    try:
        loaded = load_run(
            run_id,
            runs_dir,
            checkpoint,
            backend,
            authorization=authorization,
            **verification,
        )
    except (EngineCapabilityError, NotImplementedError, ValueError) as error:
        if (
            not isinstance(error, (EngineCapabilityError, NotImplementedError))
            and not any(
                reason in str(error)
                for reason in (
                    "requested device unavailable:",
                    "unsupported inference engine:",
                    "MLX inference only supports the stored metal backend",
                )
            )
            and not isinstance(error, RuntimeAuthorizationRequired)
        ):
            raise
        unavailable_backend = str(error)
    if loaded is None:
        if checkpoint in {"latest.json", "best.json"}:
            pointer_path = run / "checkpoints" / checkpoint
            if pointer_path.is_symlink():
                raise ValueError("symlinked checkpoint pointer")
            pointer = read_document(pointer_path)
            candidate = pointer["relative_path"]
            if not isinstance(candidate, str) or Path(candidate).name != candidate:
                raise ValueError("unsafe checkpoint pointer generation")
            generation = f"checkpoints/{candidate}"
            expected_pointer_sha = pointer["manifest_sha256"]
        else:
            parts = Path(checkpoint).parts
            if len(parts) == 1:
                generation = f"checkpoints/{checkpoint}"
            elif len(parts) == 2 and parts[0] == "checkpoints":
                generation = checkpoint
            else:
                raise ValueError("invalid requested checkpoint generation")
            expected_pointer_sha = None
        selected = read_document(run / generation / "manifest.json")
        checkpoint_sha256 = selected["sha256"]
        if (
            expected_pointer_sha is not None
            and checkpoint_sha256 != expected_pointer_sha
        ):
            raise ValueError(
                "checkpoint pointer digest does not match selected generation"
            )
        _selected_checkpoint(run, generation, checkpoint_sha256, **verification)
        identity = {
            "checkpoint_relative_path": generation,
            "checkpoint_sha256": checkpoint_sha256,
        }
    else:
        run = loaded.run
        identity = loaded.identity
        generation = identity["checkpoint_relative_path"]
        checkpoint_sha256 = identity["checkpoint_sha256"]
        _selected_checkpoint(run, generation, checkpoint_sha256, **verification)
    if loaded is not None:
        requested_runtime = loaded.config.runtime
    else:
        requested_runtime = evaluation_config(
            run_id, runs_dir, generation, backend, **verification
        ).runtime
    evaluation_runtime = {
        "engine": requested_runtime.engine,
        "backend": backend or requested_runtime.backend,
        "precision": "fp32",
        "device_index": 0,
        "observed": None if loaded is None else identity["runtime"],
    }
    rows: list[dict[str, Any]] = []
    for item in suite.evaluations:
        started = time.monotonic()
        row: dict[str, Any] = {
            "id": item.id,
            "role": item.role,
            "kind": item.kind,
            "source_sha256": None,
            "status": "UNAVAILABLE",
            "path": None,
            "sha256": None,
            "result": None,
            "reason": None,
            "elapsed_seconds": None,
        }
        if item.source is not None:
            path = _source(source, item.source)
            if path.is_file() and not path.is_symlink():
                row["source_sha256"] = sha256_file(path)
        try:
            if item.kind == "heldout_lm":
                if loaded is None:
                    row["reason"] = unavailable_backend
                else:
                    result = {"identity": identity, "metrics": loaded.evaluate()}
                    result_path = write_inference_result(run, "suite-heldout", result)
                    row["result"] = result["metrics"]
                    row.update(
                        status="COMPLETED",
                        path=str(result_path.resolve()),
                        sha256=sha256_file(result_path),
                    )
            elif item.kind == "capability_card":
                card_path = _source(source, item.source or "")
                card = load_capability_card(card_path)
                if loaded is None:
                    row["reason"] = unavailable_backend
                else:
                    result = evaluate_capability(
                        card,
                        loaded.model,
                        loaded.tokenizer,
                        loaded.config.model.max_seq_len,
                        loaded.device,
                        engine=loaded.engine,
                    )
                    result["identity"] = identity
                    result_path = write_capability_result(run, result)
                    row["result"] = {"score": result["score"], "valid": result["valid"]}
                    if not result["valid"]:
                        row["reason"] = "capability card exceeds run context"
                    else:
                        row.update(
                            status="COMPLETED",
                            path=str(result_path.resolve()),
                            sha256=sha256_file(result_path),
                        )
            elif item.kind == "surface_review":
                bundle = _source(source, item.source or "")
                reviewed = (
                    _surface(bundle, checkpoint_sha256, allow_pending=True)
                    if bundle.is_dir()
                    else None
                )
                if reviewed is not None:
                    reviewed_sha, results = reviewed
                    row.update(
                        status="COMPLETED",
                        path=str(bundle),
                        sha256=reviewed_sha,
                        source_sha256=reviewed_sha,
                        result={"review": results},
                    )
                else:
                    row.update(
                        status="SKIPPED_REVIEW",
                        reason="human Surface Review has not been sealed",
                    )
            else:
                reference = _source(source, item.source or "")
                if reference.is_file():
                    try:
                        target, verified = _supplied_index(
                            reference, checkpoint_sha256, **verification
                        )
                    except (UnsupportedEvidenceReference, FileNotFoundError) as error:
                        row["reason"] = f"unavailable supplied evidence: {error}"
                    else:
                        row.update(
                            status="COMPLETED",
                            path=str(target),
                            sha256=verified["index_sha256"],
                            result={"index_sha256": verified["index_sha256"]},
                        )
                else:
                    row["reason"] = (
                        "no schema-aware checkpoint-bound evidence is available"
                    )
        except (NotImplementedError, RuntimeError, ValueError) as error:
            if not isinstance(error, NotImplementedError) and not (
                item.kind == "capability_card"
                and isinstance(error, ValueError)
                and "semantic capability queries require a PyTorch device" in str(error)
            ):
                raise
            row["reason"] = str(error)
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
        rows.append(row)
    payload = {
        "format": "evaluation-index-v1",
        "suite": str(source),
        "suite_sha256": sha256_file(source),
        "run": str(run),
        "run_id": run_id,
        "checkpoint": generation,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluations": rows,
        "evaluation_runtime": evaluation_runtime,
        "runtime_authorization": None
        if authorization is None
        else authorization.as_dict(),
        "provenance": _provenance(source, suite),
        "created_at_utc": utc_now(),
    }
    # Stable scientific binding, with timing and timestamp preserved on replay.
    binding = _index_binding(
        payload["suite_sha256"], payload["checkpoint_sha256"], rows, evaluation_runtime
    )
    index_sha = canonical_digest(binding)
    payload["index_sha256"] = index_sha
    payload["record_sha256"] = canonical_digest(payload)
    output = run / "evaluations" / f"suite-{index_sha}.json"
    if output.exists():
        verify_evaluation_index(output, **verification)
    else:
        publish_immutable(output, payload)
    return output


def verify_evaluation_index(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Reopen the exact generation, suite, card and each completed result."""
    validate_mode(verification_mode)
    verification = {
        "proof_store": proof_store,
        "verification_mode": verification_mode,
    }
    path = Path(path)
    record = read_canonical(path)
    if record.get("format") != "evaluation-index-v1":
        raise ValueError("invalid evaluation index format")
    suite_path = Path(record["suite"])
    suite = load_suite(suite_path)
    if suite_path.is_symlink() or sha256_file(suite_path) != record["suite_sha256"]:
        raise ValueError("evaluation suite bytes changed")
    run = Path(record["run"])
    if (
        path.parent.resolve() != (run / "evaluations").resolve()
        or run.name != record["run_id"]
    ):
        raise ValueError("evaluation index run mismatch")
    _selected_checkpoint(
        run, record["checkpoint"], record["checkpoint_sha256"], **verification
    )
    evaluation_runtime = record["evaluation_runtime"]
    if (
        not isinstance(evaluation_runtime, dict)
        or evaluation_runtime.get("precision") != "fp32"
        or evaluation_runtime.get("device_index") != 0
    ):
        raise ValueError("invalid evaluation runtime selection")
    rows = record["evaluations"]
    if not isinstance(rows, list) or len(rows) != len(suite.evaluations):
        raise ValueError("evaluation index coverage mismatch")
    for row, item in zip(rows, suite.evaluations, strict=True):
        if (row["id"], row["kind"], row["role"]) != (item.id, item.kind, item.role):
            raise ValueError("evaluation index declaration mismatch")
        source = _source(suite_path, item.source) if item.source else None
        if item.kind == "surface_review":
            if row["status"] == "COMPLETED":
                if source is None or Path(row["path"]) != source or source.is_symlink():
                    raise ValueError("Surface Review source changed")
                reviewed = _surface(source, record["checkpoint_sha256"])
                if reviewed is None:
                    raise ValueError("completed Surface Review became pending")
                reviewed_sha, results = reviewed
                if (row["sha256"], row["source_sha256"], row["result"]) != (
                    reviewed_sha,
                    reviewed_sha,
                    {"review": results},
                ):
                    raise ValueError("Surface Review result changed")
            elif (
                row["status"] != "SKIPPED_REVIEW"
                or row["path"] is not None
                or row["sha256"] is not None
            ):
                raise ValueError("invalid Surface Review availability")
            continue
        if (
            source is not None
            and row["source_sha256"] is not None
            and (source.is_symlink() or sha256_file(source) != row["source_sha256"])
        ):
            raise ValueError("evaluation source changed")
        if item.kind == "capability_card" and row["source_sha256"] is None:
            raise ValueError("capability card source unverified")
        if row["status"] == "COMPLETED":
            result_path = Path(row["path"])
            if item.kind == "evidence_reference":
                if source is None or row["source_sha256"] is None:
                    raise ValueError("missing pinned evidence reference")
                target, verified = _supplied_index(
                    source, record["checkpoint_sha256"], **verification
                )
                if (
                    result_path != target
                    or row["sha256"] != verified["index_sha256"]
                    or row["result"] != {"index_sha256": verified["index_sha256"]}
                ):
                    raise ValueError("supplied evidence binding changed")
                continue
            if (
                result_path.parent.resolve() != path.parent.resolve()
                or result_path.is_symlink()
            ):
                raise ValueError("evaluation result outside run")
            result = _checked_file(result_path, row["sha256"])
            if (
                result.get("identity", {}).get("checkpoint_sha256")
                != record["checkpoint_sha256"]
                or result["identity"].get("checkpoint_relative_path")
                != record["checkpoint"]
            ):
                raise ValueError("evaluation result checkpoint mismatch")
            if item.kind in {"heldout_lm", "capability_card"} and (
                result["identity"].get("runtime") != evaluation_runtime["observed"]
                or result["identity"]["runtime"].get("engine")
                != evaluation_runtime["engine"]
                or result["identity"]["runtime"].get("backend")
                != evaluation_runtime["backend"]
                or result["identity"]["runtime"].get("precision")
                != evaluation_runtime["precision"]
                or result["identity"]["runtime"].get("device_index")
                != evaluation_runtime["device_index"]
            ):
                raise ValueError("evaluation result runtime mismatch")
            if item.kind == "heldout_lm":
                if result.get("metrics") != row["result"]:
                    raise ValueError("heldout result mismatch")
            elif item.kind == "capability_card":
                from klab.evaluation.capabilities import load_capability_card

                card = load_capability_card(source)
                inner = {
                    key: val for key, val in result.items() if key != "result_digest"
                }
                if (
                    result.get("result_digest") != canonical_digest(inner)
                    or result.get("card_digest") != card.digest
                    or result.get("valid") is not True
                    or row["result"] != {"score": result["score"], "valid": True}
                ):
                    raise ValueError("capability result integrity failure")
            else:
                raise ValueError("unverified evaluation result kind")
        elif (
            row["status"] != "UNAVAILABLE"
            or row["path"] is not None
            or row["sha256"] is not None
        ):
            raise ValueError("invalid unavailable evaluation row")
    binding = _index_binding(
        record["suite_sha256"], record["checkpoint_sha256"], rows, evaluation_runtime
    )
    if (
        canonical_digest(binding) != record["index_sha256"]
        or path.name != f"suite-{record['index_sha256']}.json"
    ):
        raise ValueError("evaluation index identity mismatch")
    if (
        record.get("record_sha256")
        != hashlib.sha256(
            canonical_json(
                {key: val for key, val in record.items() if key != "record_sha256"}
            )
        ).hexdigest()
    ):
        raise ValueError("evaluation index record changed")
    return record
