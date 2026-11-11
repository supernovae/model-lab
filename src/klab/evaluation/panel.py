"""Immutable, descriptive generation observations bound to an evaluation index."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_serializer,
)

from klab.campaign.state import publish_immutable, read_canonical
from klab.canonical import canonical_digest
from klab.config.models import StrictModel
from klab.documents import read_document
from klab.evaluation.generation import generate_with_token_ids
from klab.evaluation.inference import load_run
from klab.evaluation.suite import verify_evaluation_index
from klab.runtime_profile import RuntimeAuthorization
from klab.training.manifest import sha256_file
from klab.verification_proofs import ProofStore, VerificationMode
from klab.workspace_cleanup import campaign_lock


class PanelDecoder(StrictModel):
    temperature: StrictFloat | StrictInt = Field(ge=0, allow_inf_nan=False)
    top_k: StrictInt = Field(ge=0)
    max_new_tokens: StrictInt = Field(ge=0)
    seed: StrictInt = Field(ge=0, lt=2**64)
    use_cache: StrictBool = True
    strict_context: StrictBool = False

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        for field in ("use_cache", "strict_context"):
            if field not in self.model_fields_set:
                data.pop(field, None)
        return data


class GenerationPanel(StrictModel):
    generation_panel_version: Literal[1]
    id: str
    role: Literal["descriptive_not_quality_gate"]
    prompts: tuple[str, ...] = Field(min_length=1)
    decoder: PanelDecoder
    checkpoint_selection: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError("panel ID must be safe and nonempty")
        return value


class PanelPair(StrictModel):
    id: str
    cached_index: int = Field(ge=0)
    uncached_index: int = Field(ge=0)
    item_content_sha256: str
    source_family_id: str
    gold_answer: str = Field(min_length=1)
    prompt_sha256: str

    @field_validator("id", "source_family_id")
    @classmethod
    def safe_identity(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError("pair and family IDs must be safe")
        return value

    @field_validator("item_content_sha256", "prompt_sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("pair digests must be SHA-256")
        return value


class PanelPairManifest(StrictModel):
    panel_pair_version: Literal[1]
    frozen_content_sha256: str
    pairs: tuple[PanelPair, ...] = Field(min_length=1)

    @field_validator("frozen_content_sha256")
    @classmethod
    def valid_frozen_digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("frozen item digest must be SHA-256")
        return value


def verify_panel_pair_parity(
    cached_result: Path, uncached_result: Path, pair_manifest: Path
) -> dict[str, Any]:
    """Compare exact token IDs; a match is diagnostic, never model readiness."""
    cached = verify_panel_result(cached_result)
    uncached = verify_panel_result(uncached_result)
    if (
        cached["evaluation_index_sha256"] != uncached["evaluation_index_sha256"]
        or cached["checkpoint_sha256"] != uncached["checkpoint_sha256"]
        or cached["identity"]["tokenizer_sha256"]
        != uncached["identity"]["tokenizer_sha256"]
    ):
        raise ValueError("paired panels identify different model inputs")
    cached_decoder = cached["declaration"]["decoder"]
    uncached_decoder = uncached["declaration"]["decoder"]
    if (
        cached_decoder.get("use_cache", True) is not True
        or uncached_decoder.get("use_cache", True) is not False
        or {key: value for key, value in cached_decoder.items() if key != "use_cache"}
        != {key: value for key, value in uncached_decoder.items() if key != "use_cache"}
    ):
        raise ValueError("paired panels differ beyond cache setting")
    manifest = PanelPairManifest.model_validate(read_document(pair_manifest))
    pairs = manifest.pairs
    if (
        len({pair.id for pair in pairs}) != len(pairs)
        or len({pair.cached_index for pair in pairs}) != len(pairs)
        or len({pair.uncached_index for pair in pairs}) != len(pairs)
    ):
        raise ValueError("duplicate paired prompt ID or ordinal")
    results = []
    for pair in pairs:
        if pair.cached_index >= len(cached["rows"]) or pair.uncached_index >= len(
            uncached["rows"]
        ):
            raise ValueError("paired prompt ordinal outside panel")
        left = cached["rows"][pair.cached_index]
        right = uncached["rows"][pair.uncached_index]
        if (
            left["prompt"] != right["prompt"]
            or hashlib.sha256(left["prompt"].encode()).hexdigest() != pair.prompt_sha256
        ):
            raise ValueError("paired prompt content changed")
        status = (
            "UNAVAILABLE"
            if left["status"] != "COMPLETED" or right["status"] != "COMPLETED"
            else "MATCH"
            if left["token_ids"] == right["token_ids"]
            else "DIVERGED"
        )
        results.append(
            {
                "id": pair.id,
                "cached_index": pair.cached_index,
                "uncached_index": pair.uncached_index,
                "source_family_id": pair.source_family_id,
                "item_content_sha256": pair.item_content_sha256,
                "status": status,
            }
        )
    return {
        "role": "descriptive_not_quality_gate",
        "cached_record_sha256": cached["record_sha256"],
        "uncached_record_sha256": uncached["record_sha256"],
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "frozen_content_sha256": manifest.frozen_content_sha256,
        "pairs": results,
    }


def load_panel(source: Path) -> GenerationPanel:
    """Read the complete declaration without defaults or decoder overrides."""
    return GenerationPanel.model_validate(read_document(source))


def _runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    """Separate stable execution identity from discovery timestamps and counters."""
    return {
        key: runtime.get(key)
        for key in (
            "engine",
            "backend",
            "device_index",
            "device_name",
            "framework_version",
            "os",
            "precision",
            "runtime_version",
            "physical_device_id",
        )
    }


def _binding(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "panel_sha256",
            "evaluation_index_sha256",
            "evaluation_index_file_sha256",
            "run_id",
            "checkpoint",
            "checkpoint_sha256",
            "runtime_binding",
        )
    } | {"runtime": _runtime_identity(record["identity"]["runtime"])}


def _verify_runtime(index: dict[str, Any], identity: dict[str, Any]) -> None:
    observed = index["evaluation_runtime"].get("observed")
    if observed is None or _runtime_identity(observed) != _runtime_identity(
        identity["runtime"]
    ):
        raise ValueError(
            "generation panel runtime differs from observed evaluation runtime"
        )


def run_panel(
    source: Path,
    evaluation_index: Path,
    *,
    backend: str | None = None,
    authorization: RuntimeAuthorization | None = None,
    runtime_binding: dict[str, Any] | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> Path:
    """Record exactly one attempt per prompt, including empty outputs and failures.

    The authenticated evaluation index selects the immutable generation. Existing
    observations are reopened, never rerolled. Loading and runtime-authorization
    errors propagate before any panel observations can be published.
    """
    source = Path(source).resolve()
    evaluation_index = Path(evaluation_index).resolve()
    panel = load_panel(source)
    panel_sha = sha256_file(source)
    verification = {"proof_store": proof_store, "verification_mode": verification_mode}
    index = verify_evaluation_index(evaluation_index, **verification)
    index_file_sha = sha256_file(evaluation_index)
    run = Path(index["run"])
    loaded = load_run(
        index["run_id"],
        run.parent,
        index["checkpoint"],
        backend,
        authorization=authorization,
        **verification,
    )
    if (
        loaded.run.resolve() != run.resolve()
        or loaded.identity["checkpoint_sha256"] != index["checkpoint_sha256"]
        or loaded.identity["checkpoint_relative_path"] != index["checkpoint"]
    ):
        raise ValueError("panel inference differs from evaluation checkpoint binding")
    _verify_runtime(index, loaded.identity)
    record: dict[str, Any] = {
        "format": "generation-panel-result-v1",
        "role": "descriptive_not_quality_gate",
        "panel": str(source),
        "panel_sha256": panel_sha,
        "declaration": panel.model_dump(mode="json"),
        "evaluation_index": str(evaluation_index),
        "evaluation_index_sha256": index["index_sha256"],
        "evaluation_index_file_sha256": index_file_sha,
        "run": str(run),
        "run_id": index["run_id"],
        "checkpoint": index["checkpoint"],
        "checkpoint_sha256": index["checkpoint_sha256"],
        "identity": loaded.identity,
        "runtime_binding": runtime_binding,
        "runtime_authorization": None
        if authorization is None
        else authorization.as_dict(),
        "rows": [],
    }
    record["binding_sha256"] = canonical_digest(_binding(record))
    output = run / "evaluations" / f"panel-{record['binding_sha256']}.json"
    journal = output.with_suffix(".attempts")
    with campaign_lock(journal):
        if output.exists():
            verify_panel_result(output, **verification)
            return output
        for ordinal, prompt in enumerate(panel.prompts):
            marker = _attempt_marker(record, ordinal, prompt)
            started = journal / f"{ordinal}.started.json"
            completed = journal / f"{ordinal}.result.json"
            if completed.exists():
                record["rows"].append(_read_attempt(journal, marker))
                continue
            if started.exists():
                if read_canonical(started) != marker:
                    raise ValueError("generation panel attempt marker changed")
                row = {
                    "index": ordinal,
                    "prompt": prompt,
                    "attempts": 1,
                    "status": "FAILED",
                    "text": None,
                    "completion": None,
                    "token_ids": None,
                    "error": {
                        "type": "InterruptedAttempt",
                        "message": "Attempt started but no durable output was recorded; not retried.",
                    },
                }
                _publish_attempt(completed, marker, row)
                record["rows"].append(row)
                continue
            publish_immutable(started, marker)
            row: dict[str, Any] = {"index": ordinal, "prompt": prompt, "attempts": 1}
            try:
                text, token_ids = generate_with_token_ids(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    loaded.config.training.seq_len,
                    panel.decoder.max_new_tokens,
                    loaded.device,
                    temperature=panel.decoder.temperature,
                    top_k=panel.decoder.top_k,
                    seed=panel.decoder.seed,
                    stop_sequences=(),
                    strict_context=panel.decoder.strict_context,
                    use_cache=panel.decoder.use_cache,
                    engine=loaded.engine,
                )
                row.update(
                    status="COMPLETED",
                    text=text,
                    completion=text[len(prompt) :],
                    token_ids=token_ids,
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001 - preserve every failed observation
                # The native generator exposes no partial trajectory on exceptions.
                # Null is unavailable evidence, not a fabricated empty completion.
                row.update(
                    status="FAILED",
                    text=None,
                    completion=None,
                    token_ids=None,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            _publish_attempt(completed, marker, row)
            record["rows"].append(row)
        # Do not publish an observation if its declarations changed during execution.
        if (
            sha256_file(source) != panel_sha
            or sha256_file(evaluation_index) != index_file_sha
        ):
            raise ValueError("panel or evaluation index changed during generation")
        record["record_sha256"] = canonical_digest(record)
        publish_immutable(output, record)
        verify_panel_result(output, **verification)
        return output


def _attempt_marker(
    record: dict[str, Any], ordinal: int, prompt: str
) -> dict[str, Any]:
    return {
        "format": "generation-panel-attempt-v1",
        "binding_sha256": record["binding_sha256"],
        "index": ordinal,
        "prompt": prompt,
        "attempts": 1,
    }


def _publish_attempt(path: Path, marker: dict[str, Any], row: dict[str, Any]) -> None:
    payload = {"marker_sha256": canonical_digest(marker), "row": row}
    payload["record_sha256"] = canonical_digest(payload)
    publish_immutable(path, payload)


def _read_attempt(journal: Path, marker: dict[str, Any]) -> dict[str, Any]:
    ordinal = marker["index"]
    if read_canonical(journal / f"{ordinal}.started.json") != marker:
        raise ValueError("generation panel attempt marker changed")
    receipt = read_canonical(journal / f"{ordinal}.result.json")
    if receipt.get("marker_sha256") != canonical_digest(marker) or receipt.get(
        "record_sha256"
    ) != canonical_digest({k: v for k, v in receipt.items() if k != "record_sha256"}):
        raise ValueError("generation panel attempt receipt changed")
    return receipt["row"]


def verify_panel_result(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Authenticate declarations, evaluation binding and complete ordered coverage."""
    path = Path(path)
    record = read_canonical(path)
    if (
        record.get("format") != "generation-panel-result-v1"
        or record.get("role") != "descriptive_not_quality_gate"
    ):
        raise ValueError("invalid descriptive generation panel result")
    if record.get("record_sha256") != canonical_digest(
        {k: v for k, v in record.items() if k != "record_sha256"}
    ):
        raise ValueError("generation panel record changed")
    source = Path(record["panel"])
    panel = load_panel(source)
    if (
        sha256_file(source) != record["panel_sha256"]
        or panel.model_dump(mode="json") != record["declaration"]
    ):
        raise ValueError("generation panel declaration changed")
    index_path = Path(record["evaluation_index"])
    index = verify_evaluation_index(
        index_path, proof_store=proof_store, verification_mode=verification_mode
    )
    if (
        sha256_file(index_path) != record["evaluation_index_file_sha256"]
        or index["index_sha256"] != record["evaluation_index_sha256"]
        or any(
            index[key] != record[key]
            for key in ("run", "run_id", "checkpoint", "checkpoint_sha256")
        )
    ):
        raise ValueError("generation panel evaluation binding changed")
    identity = record["identity"]
    _verify_runtime(index, identity)
    if (
        identity["run_id"] != record["run_id"]
        or identity["checkpoint_sha256"] != record["checkpoint_sha256"]
        or identity["checkpoint_relative_path"] != record["checkpoint"]
        or sha256_file(Path(record["run"]) / "tokenizer.json")
        != identity["tokenizer_sha256"]
    ):
        raise ValueError("generation panel inference identity changed")
    if (
        record["binding_sha256"] != canonical_digest(_binding(record))
        or path.name != f"panel-{record['binding_sha256']}.json"
        or path.parent.resolve() != (Path(record["run"]) / "evaluations").resolve()
    ):
        raise ValueError("generation panel result identity mismatch")
    rows = record["rows"]
    if not isinstance(rows, list) or len(rows) != len(panel.prompts):
        raise ValueError("generation panel prompt coverage mismatch")
    for ordinal, (row, prompt) in enumerate(zip(rows, panel.prompts, strict=True)):
        journal_row = _read_attempt(
            path.with_suffix(".attempts"), _attempt_marker(record, ordinal, prompt)
        )
        if journal_row != row:
            raise ValueError(
                "generation panel observation differs from durable attempt"
            )
        if (
            row.get("index") != ordinal
            or row.get("prompt") != prompt
            or row.get("attempts") != 1
        ):
            raise ValueError("generation panel prompt order or attempt count changed")
        if row.get("status") == "COMPLETED":
            ids = row.get("token_ids")
            if (
                not isinstance(row.get("text"), str)
                or not isinstance(row.get("completion"), str)
                or row["text"] != prompt + row["completion"]
                or not isinstance(ids, list)
                or len(ids) > panel.decoder.max_new_tokens
                or any(type(token) is not int or token < 0 for token in ids)
                or row.get("error") is not None
            ):
                raise ValueError("invalid completed generation panel observation")
        elif row.get("status") == "FAILED":
            error = row.get("error")
            if (
                any(
                    row.get(key) is not None
                    for key in ("text", "completion", "token_ids")
                )
                or not isinstance(error, dict)
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)
            ):
                raise ValueError("invalid failed generation panel observation")
        else:
            raise ValueError("invalid generation panel observation status")
    return record
