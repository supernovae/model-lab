"""Authenticated, immutable target-token mixture over a frozen corpus release.

The family inventory is JSONL with one row per kept document: document_id,
family_id, split, stratum and content_sha256. It is deliberately separate from
the release's source-family label, which identifies a source rather than a work.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from klab.canonical import canonical_json
from klab.config.loading import load_tokenizer_config
from klab.config.models import DatasetConfig
from klab.corpus.mixture_index import (
    MAX_RECEIPT_BYTES,
    MAX_ROW_BYTES,
    MixtureIndex,
    rows,
)
from klab.corpus.mixture_schedule import (
    TRAIN_SCHEDULE,
    token_cache_root,
    verify_schedule,
    write_schedule,
)
from klab.corpus.quota import shortfall_message
from klab.corpus.release import ReleaseMembers, _verification_operation, verify_release
from klab.corpus.release_verify import current_settings
from klab.data.row_layout import document_rows
from klab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked, write_canonical_json
from klab.training.manifest import sha256_file

MIXTURE_BATCH_SOURCE_BYTES = 2 * 1024**2
PROGRESS_INTERVAL_SECONDS = 5
# A mixture output is the scheduled targets as raw little-endian int32 plus one
# small JSON row per scheduled record; token IDs are never written as JSON.
TRAIN_TOKENS = "train.tokens.ids"
TRAIN_RECORDS = "train.records.jsonl"
_TOKEN_STORE_SOURCE = Path(__file__).parents[1] / "data" / "token_store.py"
MIXTURE_COMMIT_SEGMENT_BYTES = 64 * 1024**2
MIXTURE_COMMIT_INTERVAL_SECONDS = 30


class ObservationFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TokenizerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["post_run_observation"]
    observed_at_utc: datetime
    files: dict[str, ObservationFile]

    @model_validator(mode="after")
    def exact_inputs(self) -> TokenizerObservation:
        if self.observed_at_utc.utcoffset() != timedelta(0):
            raise ValueError("tokenizer observation requires UTC")
        if set(self.files) != {
            "job_spec",
            "session_spec",
            "job_report",
            "session_report",
            "container_report",
            "stdout",
            "stderr",
            "tokenizer_config",
            "tokenizer",
            "tokenizer_manifest",
            "export",
        }:
            raise ValueError(
                "tokenizer observation requires exact completed-job inputs"
            )
        return self


class AdditiveRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_path: Path


class StratumPolicy(BaseModel):
    """Settings for one stratum; an unset field keeps the mixture-wide rule."""

    model_config = ConfigDict(extra="forbid")

    max_exposures: int | None = Field(
        default=None, strict=True, ge=1, exclude_if=lambda value: value is None
    )
    # Under document_start packing, a record of at least this many targets
    # (its EOS included) starts at a row boundary.
    min_document_tokens: int | None = Field(
        default=None, strict=True, ge=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def nonempty(self) -> StratumPolicy:
        if self.max_exposures is None and self.min_document_tokens is None:
            raise ValueError("a stratum policy must set at least one field")
        return self


class MixtureDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    release_path: Path
    tokenizer_config: Path
    family_inventory: Path
    additional_releases: list[AdditiveRelease] = Field(
        default_factory=list, max_length=1, exclude_if=lambda value: not value
    )
    source_strata: dict[str, str]
    target_quotas: dict[str, int]
    seed: int
    max_exposures: Literal[1, 2]
    unique_train_content_floors: dict[str, int] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    order_policy: Literal["kml-card03-v1"] | None = None
    # Reuse an authenticated tokenizer fitted to an earlier frozen release.
    # Absent for legacy declarations, preserving their canonical identity.
    tokenizer_origin_release_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    tokenizer_observation: TokenizerObservation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Unset fields are excluded, so declarations without them keep their
    # identity and their receipts.
    strata: dict[str, StratumPolicy] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    packing: Literal["contiguous", "document_start"] = Field(
        default="contiguous", exclude_if=lambda value: value == "contiguous"
    )
    row_tokens: int | None = Field(
        default=None, strict=True, gt=0, exclude_if=lambda value: value is None
    )
    token_cache: Literal["append", "sealed"] = Field(
        default="append", exclude_if=lambda value: value == "append"
    )
    token_output: Literal["contiguous", "references"] = Field(
        default="contiguous", exclude_if=lambda value: value == "contiguous"
    )

    @field_validator("max_exposures", mode="before")
    @classmethod
    def strict_exposure_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("max_exposures must be integer 1 or 2")
        return value

    @field_validator("unique_train_content_floors", mode="before")
    @classmethod
    def strict_content_floors(cls, value):
        if value is not None and (
            not isinstance(value, dict)
            or any(type(item) is not int for item in value.values())
        ):
            raise ValueError("unique TRAIN content floors must be integers")
        return value

    @model_validator(mode="after")
    def valid(self) -> MixtureDeclaration:
        if self.token_output == "references" and self.token_cache != "sealed":
            raise ValueError("reference token output requires token_cache: sealed")
        if (
            not self.source_strata
            or not self.target_quotas
            or set(self.source_strata.values()) != set(self.target_quotas)
            or any(not key or not value for key, value in self.source_strata.items())
            or any(
                not key or type(value) is not int or value <= 0
                for key, value in self.target_quotas.items()
            )
        ):
            raise ValueError(
                "mixture requires positive quotas for exactly the mapped strata"
            )
        if self.unique_train_content_floors is not None and (
            set(self.unique_train_content_floors) != set(self.target_quotas)
            or any(
                type(value) is not int or value <= 0
                for value in self.unique_train_content_floors.values()
            )
        ):
            raise ValueError(
                "unique TRAIN content floors must cover the exact mixture strata"
            )
        if not set(self.strata) <= set(self.target_quotas):
            raise ValueError("stratum policies must name mixture strata")
        document_start = self.packing == "document_start"
        if document_start != (self.row_tokens is not None) or document_start != bool(
            minimum_document_tokens(self)
        ):
            raise ValueError(
                "document_start packing requires row_tokens and at least one "
                "stratum min_document_tokens, and they require document_start"
            )
        return self


def stratum_exposures(spec, stratum: str) -> int:
    """A stratum's own repeat limit, else the mixture-wide one."""
    policy = spec.strata.get(stratum)
    if policy is not None and policy.max_exposures is not None:
        return policy.max_exposures
    return spec.max_exposures


def minimum_document_tokens(spec) -> dict[str, int]:
    return {
        stratum: policy.min_document_tokens
        for stratum, policy in sorted(spec.strata.items())
        if policy.min_document_tokens is not None
    }


def record_summary(lines, policy: dict, *, digest=None) -> dict:
    """Per-stratum exposure counts and the document_start row layout.

    ``lines`` are the encoded record rows; ``policy`` holds the effective
    settings the receipt records. Materialization writes this summary and
    ``verify-mixture`` recomputes it from the records file.
    """
    exposures = policy.get("stratum_max_exposures")
    counts = (
        {
            stratum: {"records": [0] * limit, "target_tokens": [0] * limit}
            for stratum, limit in exposures.items()
        }
        if exposures is not None
        else None
    )
    layout = document_rows(policy["row_layout"]) if "row_layout" in policy else None
    for line in lines:
        if digest is not None:
            digest.update(line)
        row = json.loads(line)
        if counts is not None:
            entry = counts[row["stratum"]]
            entry["records"][row["exposure"] - 1] += 1
            entry["target_tokens"][row["exposure"] - 1] += row["tokens"]
        if layout is not None:
            layout.place(row["stratum"], row["tokens"])
    summary = {}
    if counts is not None:
        summary["stratum_exposures"] = counts
    if layout is not None:
        summary["row_layout"] = layout.summary()
    return summary


def receipt_policy(spec) -> dict:
    """The effective per-stratum settings a receipt records, if any are declared."""
    policy = {}
    if spec.strata:
        policy["stratum_max_exposures"] = {
            stratum: stratum_exposures(spec, stratum)
            for stratum in sorted(spec.target_quotas)
        }
    if spec.packing == "document_start":
        policy["row_layout"] = {
            "row_tokens": spec.row_tokens,
            "strata": {
                stratum: {"min_document_tokens": minimum}
                for stratum, minimum in minimum_document_tokens(spec).items()
            },
        }
    return policy


def _verify_records(
    output: Path, receipt: dict, *, require_digest: bool = True
) -> None:
    """Verify bytes and any recorded exposure or row layout in one pass."""
    policy = {
        key: receipt[key]
        for key in ("stratum_max_exposures", "row_layout")
        if key in receipt
    }
    path = output / TRAIN_RECORDS
    if policy:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            summary = record_summary(stream, policy, digest=digest)
        actual = digest.hexdigest()
    elif require_digest:
        summary = {}
        actual = sha256_file(path)
    else:
        return
    if actual != receipt["train_records_sha256"]:
        raise ValueError("mixture token export digest mismatch")
    if any(receipt.get(key) != value for key, value in summary.items()):
        raise ValueError(
            "mixture exposure counts or document rows differ from the receipt"
        )


_rows = rows


def validate_source_strata(sources_path: Path, source_strata: dict[str, str]) -> None:
    """Require complete stratum assignments and an eligible source rights path."""
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    source_map = {row["id"]: row for row in sources}
    if set(source_map) != set(source_strata):
        raise ValueError(
            "source strata must cover exactly the verified release sources"
        )
    for row in sources:
        rights = row.get("rights_policy")
        if (
            row.get("explicit_training_restriction") not in {None, "none_found"}
            or not isinstance(rights, dict)
            or rights.get("training_eligibility")
            not in {"eligible", "eligible_with_obligations", "review_required"}
        ):
            raise ValueError("source lacks a reviewed training rights path")


def _verified_members(path: Path) -> ReleaseMembers:
    return ReleaseMembers(path, verify_release(path))


def _tokenizer(spec: MixtureDeclaration, release: ReleaseMembers):
    config = load_tokenizer_config(spec.tokenizer_config)
    dataset = config.dataset
    release_id = release.manifest["release_id"]
    origin = spec.tokenizer_origin_release_id or release_id
    if (
        dataset.source != "local_text"
        or dataset.corpus_release_path is None
        or (
            dataset.corpus_release_path.resolve().name != origin
            and not (
                dataset.corpus_release_path.resolve() == release.root
                and origin == release_id
            )
        )
        or dataset.revision != origin
        or (
            spec.tokenizer_origin_release_id is None
            and dataset.corpus_release_path.resolve() != release.root
        )
    ):
        raise ValueError("tokenizer configuration is not bound to this release")
    path = config.output_dir / "tokenizer.json"
    if getattr(spec, "tokenizer_observation", None) is not None:
        from klab.corpus.tokenizer_observation import verify_observed_tokenizer

        observed = verify_observed_tokenizer(
            spec.tokenizer_observation, spec.tokenizer_config, config, release
        )
        digest = observed["tokenizer_sha256"]
    else:
        # The operator's receipts apply here as to the release: a tokenizer whose
        # export, files and release are unchanged is not re-derived.
        settings = current_settings()
        digest = verify_tokenizer_artifact(
            path,
            source=dataset.source,
            revision=dataset.revision,
            vocab_size=config.vocab_size,
            dataset=dataset,
            proof_store=settings.proof_store,
            verification_mode=settings.verification_mode,
        )["sha256"]
    return load_tokenizer(path), digest


def materialize_mixture(declaration: Path, output: Path) -> dict:
    """Publish exact token IDs only after all quotas and closure checks pass."""
    from klab.corpus.mixture_pipeline import mixture_writer

    with mixture_writer(output), _verification_operation():
        return _materialize_mixture(declaration, output)


def _materialize_mixture(
    declaration: Path, output: Path, *, reuse_tokens: bool = True
) -> dict:
    raw = declaration.read_bytes()
    spec = MixtureDeclaration.model_validate(yaml.safe_load(raw))
    release = spec.release_path.resolve()
    output = output.resolve()
    if output.exists():
        return verify_mixture(declaration, output)
    # Everything that can refuse this output for its own sake refuses now, before
    # the hours of verification and indexing below.
    assembly, early_tokenizer_sha = (
        _early_assembly(spec, raw, output, release)
        if reuse_tokens and not spec.additional_releases
        else (None, None)
    )
    members = _verified_members(release)
    manifest = members.manifest
    cohorts = None
    additive_evidence = []
    if spec.additional_releases:
        from klab.corpus.completed_preparation import corpus_work_root
        from klab.corpus.retained_mixture import (
            prepared_release,
            verify_additive_anchor,
        )

        root = corpus_work_root(release)
        paths = [
            release,
            *(item.release_path.resolve() for item in spec.additional_releases),
        ]
        if len(set(paths)) != len(paths) or any(
            not path.is_relative_to(root) for path in paths
        ):
            raise ValueError(
                "additive releases must be distinct and inside the same work root"
            )
        if spec.family_inventory.resolve() != members.member("family-inventory.jsonl"):
            raise ValueError(
                "additive mixture requires the base release family inventory"
            )
        cohorts = [prepared_release(root, path) for path in paths]
        for cohort in cohorts[1:]:
            anchor = verify_additive_anchor(cohorts[0], cohort)
            cohort["anchor"] = anchor
            additive_evidence.append(anchor)
        source_ids = [source for cohort in cohorts for source in cohort["source_ids"]]
        if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(
            spec.source_strata
        ):
            raise ValueError(
                "additive mixture sources must be disjoint and fully mapped"
            )
        for cohort in cohorts:
            mapped = {
                source: spec.source_strata[source] for source in cohort["source_ids"]
            }
            cohort_members = ReleaseMembers(
                Path(cohort["release_path"]), cohort["manifest"]
            )
            validate_source_strata(cohort_members.member("sources.json"), mapped)
            report_path = cohort_members.member("report.json")
            summary = json.loads(report_path.read_bytes())
            if (
                summary["view_counts"]["lm"]["train"]
                != summary["split_counts"]["train"]
            ):
                raise ValueError(
                    "additive release TRAIN view does not cover its kept TRAIN documents"
                )
    else:
        validate_source_strata(members.member("sources.json"), spec.source_strata)
    report = json.loads(members.member("report.json").read_bytes())
    requested = report.get("requested_mixture")
    total_quota = sum(spec.target_quotas.values())
    if (
        not isinstance(requested, dict)
        or set(requested) != set(spec.target_quotas)
        or any(
            abs(float(requested[stratum]) - quota / total_quota) > 1e-8
            for stratum, quota in spec.target_quotas.items()
        )
    ):
        raise ValueError(
            "target quotas disagree with verified release requested mixture"
        )
    tokenizer, tokenizer_sha = _tokenizer(spec, members)
    if early_tokenizer_sha not in {None, tokenizer_sha}:
        raise ValueError("tokenizer changed while the mixture was being checked")
    from klab.corpus.completed_preparation import local_scope

    completed = local_scope(manifest)
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("verified tokenizer lacks EOS token")
    if completed is not None:
        cache = _completed_mixture_cache(spec)
        bound = [
            output,
            spec.family_inventory.resolve(),
            spec.tokenizer_config.resolve(),
            load_tokenizer_config(spec.tokenizer_config).output_dir.resolve(),
            *(item.release_path.resolve() for item in spec.additional_releases),
        ]
        if (
            cache.proofs is None
            or not cache.writable
            or any(not p.is_relative_to(cache.root) for p in bound)
        ):
            raise ValueError(
                "completed mixture requires existing trusted work-root completion storage"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_cache, checkpoint_request = _native_mixture_completion(
        spec, raw, output, manifest, tokenizer_sha
    )
    if reuse_tokens:
        from klab.corpus.mixture_pipeline import claim_assembly

        assembly = assembly or claim_assembly(output, checkpoint_request)
        staging = assembly / "payload"
        if symlinked(staging):
            raise ValueError("unsafe mixture checkpoint payload")
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        if any(
            path.name
            not in {TRAIN_TOKENS, TRAIN_RECORDS, TRAIN_SCHEDULE, "receipt.json"}
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            for path in staging.iterdir()
        ):
            raise ValueError("unsafe mixture checkpoint payload inventory")
        # These directories only contain rebuildable SQL state. A killed writer
        # cannot run TemporaryDirectory cleanup, so reclaim it under the lease.
        for scratch in assembly.glob(".mixture-index-*"):
            if scratch.is_dir() and not scratch.is_symlink():
                shutil.rmtree(scratch)
    else:
        staging = Path(tempfile.mkdtemp(prefix=".mixture-", dir=output.parent))
    try:
        with tempfile.TemporaryDirectory(
            prefix=".mixture-index-", dir=assembly if reuse_tokens else staging
        ) as scratch:
            preflight = None
            if cohorts is None:
                from klab.corpus.mixture_pipeline import MixturePreflight

                # The release manifest already carries the documents and lineage
                # digests; only the family inventory is outside it.
                documents = members.member("documents.jsonl")
                lineage = members.member("lm/train.lineage.jsonl")
                digests = {
                    documents: manifest["files"]["documents.jsonl"]["sha256"],
                    lineage: manifest["files"]["lm/train.lineage.jsonl"]["sha256"],
                    spec.family_inventory: _inventory_sha256(spec.family_inventory),
                }

                if reuse_tokens:
                    preflight = MixturePreflight(
                        checkpoint_cache,
                        digests,
                        seed=spec.seed,
                        order_policy=spec.order_policy,
                        strata=spec.source_strata,
                    )
                index = MixtureIndex(
                    Path(scratch),
                    documents,
                    digests,
                    prepared=preflight.snapshot if preflight else None,
                )
            else:
                from klab.corpus.retained_mixture import RetainedMixtureIndex

                index = RetainedMixtureIndex(
                    Path(scratch),
                    root,
                    cohorts,
                    spec.seed,
                    spec.source_strata,
                    spec.order_policy,
                )
            with index:
                if cohorts is None and not index.prepared:
                    index.inventory(
                        spec.family_inventory,
                        spec.source_strata,
                        spec.seed,
                        spec.order_policy,
                    )
                    index.select(lineage)
                    if preflight is not None:
                        preflight.publish(index)
                frontier = None
                if reuse_tokens:
                    from klab.corpus.mixture_pipeline import (
                        MixtureFrontier,
                        pipeline_request,
                    )

                    assert assembly is not None
                    frontier = MixtureFrontier(
                        assembly,
                        checkpoint_cache,
                        pipeline_request(checkpoint_request, index),
                    )
                receipt = _schedule(
                    spec,
                    raw,
                    members,
                    tokenizer,
                    tokenizer_sha,
                    eos,
                    index,
                    staging,
                    reuse_tokens=reuse_tokens,
                    frontier=frontier,
                )
                index.verify_inputs()
                if cohorts is not None:
                    composite = {
                        "base_release_id": manifest["release_id"],
                        "additional_release_ids": [
                            item["release_id"] for item in additive_evidence
                        ],
                    }
                    receipt["release_id"] = hashlib.sha256(
                        canonical_json(composite)
                    ).hexdigest()
                    receipt["additive_release_binding"] = {
                        **composite,
                        "evidence": additive_evidence,
                    }
                    receipt["retained_reference_execution"] = index.reference_evidence
        if spec.tokenizer_observation is not None:
            from klab.corpus.tokenizer_observation import verify_observed_tokenizer

            observed = verify_observed_tokenizer(
                spec.tokenizer_observation,
                spec.tokenizer_config,
                load_tokenizer_config(spec.tokenizer_config),
                members,
            )
            if observed["tokenizer_sha256"] != tokenizer_sha:
                raise ValueError("observed tokenizer changed during mixture execution")
            receipt["tokenizer_observation"] = observed
        if completed is not None:
            receipt["completed_preparation"] = {
                "policy": completed["policy"],
                "verification_scope": "native executed mixture schedule and unique-content counts; no later schedule replay",
            }
        if frontier is not None:
            for name, digest in frontier.output_digests.items():
                digest.inventory_entry(staging / name)
        if "token_schedule" in receipt:
            schedule = verify_schedule(
                staging, token_cache_root(release, output.parent), receipt, eos
            )
            schedule.require_unchanged()
        receipt_path = staging / "receipt.json"
        if reuse_tokens:
            receipt_path.unlink(missing_ok=True)
        write_canonical_json(receipt_path, receipt, exclusive=True)
        sync_directory(staging)
        rename_noreplace(staging, output)
        sync_directory(output.parent)
        if staging.parent != output.parent:
            sync_directory(staging.parent)
        if completed is not None:
            _seal_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
        else:
            _seal_native_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
        # The published output and its sealed receipt are the record now. The
        # committed segments and their markers existed only to resume an
        # interrupted schedule; progress.json and request.json stay.
        if frontier is not None:
            assembly = frontier.root
            for marker in sorted(assembly.glob("commit-*.json")):
                marker.unlink()
            for segment in assembly.glob("segment-*"):
                if segment.is_dir() and not segment.is_symlink():
                    shutil.rmtree(segment)
        return receipt
    except BaseException:
        if not reuse_tokens and staging.exists():
            shutil.rmtree(staging)
        raise


def _schedule(
    spec,
    raw,
    release: ReleaseMembers,
    tokenizer,
    tokenizer_sha,
    eos,
    index,
    staging,
    *,
    reuse_tokens=True,
    frontier=None,
):
    manifest = release.manifest
    totals: Counter[str] = Counter()
    unique: Counter[str] = Counter()
    available_unique: Counter[str] = Counter()
    available_content: Counter[str] = Counter()
    eligible_families: Counter[str] = Counter()
    source_totals: Counter[str] = Counter()
    records = 0
    excluded: list[dict] = []
    excluded_bytes = 0

    def exclude(doc, reason):
        nonlocal excluded_bytes
        row = {"document_id": doc["document_id"], "reason": reason}
        excluded_bytes += len(canonical_json(row)) + 1
        if excluded_bytes > MAX_RECEIPT_BYTES // 2:
            raise ValueError("mixture exclusion receipt-byte cap exceeded")
        excluded.append(row)

    from klab.corpus.mixture_pipeline import encode_ahead
    from klab.corpus.view_checkpoints import ViewOutput
    from klab.data import encoding
    from klab.data.encoding import PreparationEncoder
    from klab.data.token_store import ContentTokens

    work_parent = frontier.root.parent if frontier is not None else staging.parent
    token_root = token_cache_root(release.root, work_parent)
    sealed_cache = reuse_tokens and spec.token_cache == "sealed"
    reference_rows = spec.token_output == "references"
    reference_output = reference_rows and reuse_tokens
    cache_type = ContentTokens
    if sealed_cache:
        from klab.data.token_segments import SealedContentTokens

        cache_type = SealedContentTokens
    mode = "ab+" if frontier is not None else "wb+"
    with (
        ExitStack() as encoders,
        (
            cache_type(token_root, tokenizer_sha) if reuse_tokens else nullcontext(None)
        ) as tokens,
        (staging / TRAIN_RECORDS).open(mode) as record_stream,
    ):
        outputs = {TRAIN_RECORDS: ViewOutput(record_stream)}
        if not reference_output:
            stream = encoders.enter_context((staging / TRAIN_TOKENS).open(mode))
            outputs[TRAIN_TOKENS] = ViewOutput(stream)
        from klab.corpus.release_verify import worker_count
        from klab.host_capacity import plan_host_workers

        workers = plan_host_workers(
            "mixture",
            worker_memory_bytes=512 * 1024**2,
            reserve_bytes=1024**3,
            operator_cap=worker_count(),
        ).workers
        # The encoder workers read and tokenize; only an index that reads
        # source rows itself (retained references) uses reader threads.
        read_workers = max(1, workers // 4) if workers > 1 else 0
        metrics = {"workers": workers, "read_workers": read_workers}
        index.pipeline_metrics = metrics
        index.token_lookup = tokens.lookup if tokens is not None else None
        started = time.monotonic()
        selected = Counter()
        done = []
        resumed = None
        candidate_cursor = None
        emission_exposure = emission_ordinal = 0
        candidates = Counter()
        if frontier is not None:
            encoders.callback(frontier.close, index)
            frontier.restore(index, outputs, excluded)
            if frontier.state is not None:
                resumed = frontier.state["state"]
                for counter, name in (
                    (totals, "totals"),
                    (unique, "unique"),
                    (available_unique, "available_unique"),
                    (available_content, "available_content"),
                    (eligible_families, "eligible_families"),
                    (source_totals, "source_totals"),
                    (candidates, "candidates"),
                    (selected, "selected"),
                ):
                    counter.update(resumed[name])
                records = resumed["records"]
                excluded_bytes = resumed["excluded_bytes"]
                done = list(resumed["done"])
            frontier.begin(index)

        commit_candidates = int(
            os.environ.get("KLAB_MIXTURE_COMMIT_CANDIDATES", "4096")
        )
        if commit_candidates < 1:
            raise ValueError("KLAB_MIXTURE_COMMIT_CANDIDATES must be positive")
        committed_at = time.monotonic()
        reported_at = float("-inf")

        def checkpoint(stratum, available, *, force=False):
            nonlocal committed_at, reported_at
            commit_started = time.monotonic()
            committed = (
                frontier.state["state"]
                if frontier is not None and frontier.state
                else {}
            )
            pending_bytes = (
                frontier.event_file.tell() + frontier.pending_bytes(outputs)
                if frontier is not None
                else 0
            )
            due = (
                force
                or sum(candidates.values())
                - sum(committed.get("candidates", {}).values())
                >= commit_candidates
                or records - committed.get("records", 0) >= commit_candidates
                or pending_bytes >= MIXTURE_COMMIT_SEGMENT_BYTES
                or commit_started - committed_at >= MIXTURE_COMMIT_INTERVAL_SECONDS
            )
            if frontier is not None and due:
                frontier.commit(
                    {
                        "stratum": stratum,
                        "cursor": candidate_cursor,
                        "emission_exposure": emission_exposure,
                        "emission_ordinal": emission_ordinal,
                        "available": available,
                        "done": list(done),
                        "totals": dict(totals),
                        "unique": dict(unique),
                        "available_unique": dict(available_unique),
                        "available_content": dict(available_content),
                        "eligible_families": dict(eligible_families),
                        "source_totals": dict(source_totals),
                        "candidates": dict(candidates),
                        "selected": dict(selected),
                        "records": records,
                        "excluded_bytes": excluded_bytes,
                    },
                    outputs,
                    excluded,
                    index,
                )
                committed_at = time.monotonic()
                committed = frontier.state["state"]
                pending_bytes = 0
            metrics["checkpoint_seconds"] = (
                metrics.get("checkpoint_seconds", 0) + time.monotonic() - commit_started
            )
            if not force and time.monotonic() - reported_at < PROGRESS_INTERVAL_SECONDS:
                return
            reported_at = time.monotonic()
            from klab.corpus.mixture_pipeline import process_tree_sample

            observation = process_tree_sample(
                encoder.pids() if encoder is not None else []
            )
            child_cpu_seconds = observation["child_cpu_seconds"]
            metrics.update(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "candidates": dict(candidates),
                    "selected": dict(selected),
                    "committed_candidates": committed.get("candidates", {}),
                    "committed_target_tokens": committed.get("totals", {}),
                    "pending_segment_bytes": pending_bytes,
                    "commit_candidate_limit": commit_candidates,
                    "commit_segment_bytes": MIXTURE_COMMIT_SEGMENT_BYTES,
                    "commit_interval_seconds": MIXTURE_COMMIT_INTERVAL_SECONDS,
                    "cache_hits": tokens.hits if tokens is not None else 0,
                    "cache_misses": tokens.misses if tokens is not None else 0,
                    "cache_lookup_seconds": tokens.lookup_seconds
                    if tokens is not None
                    else 0,
                    "cache_write_seconds": tokens.write_seconds
                    if tokens is not None
                    else 0,
                    **observation,
                    "encoder_rayon_threads": encoder.rayon_threads
                    if encoder is not None
                    else 0,
                    "encoder_cpu_seconds": child_cpu_seconds,
                    "encoder_utilization": (
                        child_cpu_seconds
                        / max(
                            1e-9,
                            (time.monotonic() - started)
                            * (encoder.rayon_threads if encoder is not None else 1),
                        )
                        if observation["child_sample_complete"]
                        else None
                    ),
                    "commits": frontier.sequence if frontier is not None else 0,
                }
            )
            metrics["read_utilization"] = metrics.get("read_worker_seconds", 0) / max(
                1e-9, metrics["elapsed_seconds"] * max(1, read_workers)
            )
            if sealed_cache:
                metrics.update(
                    sealed_token_cache_hits=tokens.sealed_hits,
                    published_token_segments=tokens.published_segments,
                )
            progress = (
                frontier.root if frontier is not None else staging
            ) / "progress.json"
            temporary = progress.with_suffix(".partial")
            temporary.write_bytes(canonical_json(metrics) + b"\n")
            os.replace(temporary, progress)

        def emit(doc, stratum, quota, exposure):
            nonlocal records
            remaining = quota - totals[stratum]
            if remaining <= 0:
                return
            if reference_output:
                take = min(doc["token_count"], remaining)
            else:
                ids = doc["token_ids"]
                take = min(len(ids), remaining)
                targets = np.asarray(ids[:take], dtype="<i4")
                if take != len(ids):
                    targets = targets.copy()
                    targets[-1] = eos
                outputs[TRAIN_TOKENS].write(targets.tobytes())
            row = {
                "document_id": doc["document_id"],
                "family_id": doc["family_id"],
                "source_id": doc["source_id"],
                "stratum": stratum,
                "exposure": exposure,
                "content_token_start": 0,
                "content_token_end": take - 1,
                "tokens": take,
            }
            if reference_rows:
                row["content_sha256"] = doc["content_sha256"]
            encoded_row = canonical_json(row) + b"\n"
            outputs[TRAIN_RECORDS].write(encoded_row)
            totals[stratum] += take
            source_totals[doc["source_id"]] += take
            if exposure == 1:
                unique[stratum] += take
            records += 1

        # A padded or stand-in tokenizer is encoded here, one text at a time.
        encoder = (
            encoders.enter_context(PreparationEncoder(tokenizer, max_workers=workers))
            if getattr(tokenizer, "padding", None) is None
            and callable(getattr(tokenizer, "to_str", None))
            else None
        )

        for stratum, quota in sorted(spec.target_quotas.items()):
            if stratum in done:
                if hasattr(index, "ranked") and resumed is not None:
                    index.frontier = resumed["cursor"]
                continue
            continuing = resumed is not None and resumed["stratum"] == stratum
            candidate_cursor = resumed["cursor"] if continuing else None
            available = resumed["available"] if continuing else 0
            emission_exposure = resumed["emission_exposure"] if continuing else 0
            emission_ordinal = resumed["emission_ordinal"] if continuing else 0
            if not continuing:
                index.record_event(["reset", None])
            exposures = stratum_exposures(spec, stratum)
            # A single exposure is emitted as it is encoded; only repeats need
            # the index to retain encoded documents.
            direct = exposures == 1

            def process(
                doc, family, ids, *, stratum=stratum, quota=quota, direct=direct
            ):
                nonlocal available
                if not len(ids):
                    exclude(doc, "zero_encoded_targets")
                    return
                selected[stratum] += 1
                count = len(ids) + 1
                if 4 * count > MAX_ROW_BYTES and (direct or reference_output):
                    raise ValueError("mixture encoded record-byte cap exceeded")
                encoded_doc = {
                    "document_id": doc["document_id"],
                    "source_id": doc["source_id"],
                    "family_id": family,
                    "stratum": stratum,
                }
                if reference_rows:
                    encoded_doc["content_sha256"] = doc["content_sha256"]
                if reference_output:
                    encoded_doc["token_count"] = count
                else:
                    targets = np.empty(count, dtype="<i4")
                    targets[:-1] = ids
                    targets[-1] = eos
                    encoded_doc["token_ids"] = targets if direct else targets.tolist()
                if direct:
                    index.record_family(stratum, family)
                    emit(encoded_doc, stratum, quota, 1)
                else:
                    index.append(encoded_doc)
                available += len(ids) + 1
                available_content[stratum] += len(ids)

            candidate_rows = (
                ()
                if emission_exposure
                else encode_ahead(
                    index.candidates(stratum, candidate_cursor, workers=read_workers),
                    position=lambda: index.frontier,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    tokens=tokens,
                    documents=encoding.TOKENIZER_BATCH_DOCUMENTS,
                    source_bytes=MIXTURE_BATCH_SOURCE_BYTES,
                    metrics=metrics,
                )
            )
            if hasattr(candidate_rows, "close"):
                encoders.callback(candidate_rows.close)
            applied = 0
            for doc, family, cursor, ids in candidate_rows:
                candidate_cursor = cursor
                candidates[stratum] += 1
                if index.seen(doc["content_sha256"]):
                    exclude(doc, "duplicate_train_content")
                else:
                    process(doc, family, ids)
                applied += 1
                if applied >= encoding.TOKENIZER_BATCH_DOCUMENTS:
                    applied = 0
                    checkpoint(stratum, available)
            checkpoint(stratum, available)
            if (
                spec.unique_train_content_floors is not None
                and available_content[stratum]
                < spec.unique_train_content_floors[stratum]
            ):
                raise ValueError(
                    f"unique TRAIN content supply below floor for {stratum}: "
                    f"{available_content[stratum]} available, "
                    f"{spec.unique_train_content_floors[stratum]} required (EOS excluded)"
                )
            if message := shortfall_message(stratum, available, quota, exposures):
                raise ValueError(message)
            available_unique[stratum] = available
            eligible_families[stratum] = index.family_count(stratum)
            if not direct:
                start_exposure = emission_exposure or 1
                for exposure in range(start_exposure, exposures + 1):
                    after = emission_ordinal if exposure == start_exposure else 0
                    emission_exposure = exposure
                    for doc in index.encoded(stratum, after=after):
                        if totals[stratum] >= quota:
                            break
                        emit(doc, stratum, quota, exposure)
                        emission_ordinal = index.encoded_frontier
                        if records % encoding.TOKENIZER_BATCH_DOCUMENTS == 0:
                            checkpoint(stratum, available)
                    if totals[stratum] == quota:
                        break
            if totals[stratum] != quota:
                raise ValueError(f"unfilled quota for {stratum}")
            done.append(stratum)
            if sealed_cache and len(done) == len(spec.target_quotas):
                tokens.finish()
            checkpoint(stratum, available, force=True)
        if frontier is None:
            for output_file in outputs.values():
                output_file.seal()
        else:
            frontier.output_digests = {
                name: output_file.file_digest() for name, output_file in outputs.items()
            }
        schedule = (
            write_schedule(
                staging / TRAIN_SCHEDULE, staging / TRAIN_RECORDS, tokens, eos
            )
            if reference_output
            else None
        )
    receipt = {
        "schema_version": 2,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(release.root / "manifest.json"),
        "tokenizer_sha256": tokenizer_sha,
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "family_inventory_sha256": (
            manifest["files"]["family-inventory.jsonl"]["sha256"]
            if spec.additional_releases
            else sha256_file(spec.family_inventory)
        ),
        "seed": spec.seed,
        **({"order_policy": spec.order_policy} if spec.order_policy else {}),
        "requested_target_quotas": spec.target_quotas,
        "actual_target_tokens": dict(totals),
        "unique_target_positions": dict(unique),
        "available_unique_positions": dict(available_unique),
        "eligible_train_families": dict(eligible_families),
        "source_target_tokens": dict(source_totals),
        "max_exposures": spec.max_exposures,
        "target_semantics": "ordered encoded content IDs plus segment EOS; counts are scheduled IDs before any trainer shift or block-edge loss",
        "training_supervision_verified": False,
        "records": records,
        "excluded": excluded,
        "quota_shortfalls": {
            stratum: spec.target_quotas[stratum] - totals[stratum]
            for stratum in spec.target_quotas
        },
        "train_tokens_format": "int32-le",
        "train_tokens_bytes": 4 * sum(totals.values()),
        "train_records_sha256": outputs[TRAIN_RECORDS].whole.hexdigest(),
        "train_records_bytes": (staging / TRAIN_RECORDS).stat().st_size,
    }
    if schedule is None:
        receipt["train_tokens_sha256"] = outputs[TRAIN_TOKENS].whole.hexdigest()
    else:
        receipt["token_schedule"] = schedule
    if spec.unique_train_content_floors is not None:
        receipt["available_unique_content_tokens"] = dict(available_content)
        receipt["unique_train_content_floors"] = spec.unique_train_content_floors
    policy = receipt_policy(spec)
    if policy:
        with (staging / TRAIN_RECORDS).open("rb") as lines:
            receipt.update(policy | record_summary(lines, policy))
    if len(canonical_json(receipt)) + 1 > MAX_RECEIPT_BYTES:
        raise ValueError("mixture receipt-byte cap exceeded")
    return receipt


def _completed_mixture_cache(spec):
    from klab.corpus.admission_inspection_cache import InspectionCache
    from klab.corpus.completed_preparation import corpus_work_root

    cache = InspectionCache(corpus_work_root(spec.release_path.resolve()))
    cache.authority = {
        **cache.authority,
        "mixture": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "mixture.py",
                "mixture_index.py",
                "mixture_pipeline.py",
                "mixture_schedule.py",
            )
        }
        | {"data/token_store.py": sha256_file(_TOKEN_STORE_SOURCE)},
    }
    if spec.additional_releases:
        cache.authority["mixture"]["retained_mixture.py"] = sha256_file(
            Path(__file__).with_name("retained_mixture.py")
        )
    return cache


def _completed_mixture_request(spec, raw, output, manifest, tokenizer_sha):
    return {
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "release_id": manifest["release_id"],
        "tokenizer_sha256": tokenizer_sha,
        "release_path": str(spec.release_path.resolve()),
        "output": str(output.resolve()),
        "tokenizer_config": str(spec.tokenizer_config.resolve()),
        "family_inventory": str(spec.family_inventory.resolve()),
    }


def _seal_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt):
    from klab.corpus.admission_inspection_cache import capture
    from klab.corpus.completed_preparation import (
        corpus_work_root,
        local_dependencies,
    )

    release = spec.release_path.resolve()
    cache = _completed_mixture_cache(spec)
    additive_dependencies = []
    for item in spec.additional_releases:
        path = item.release_path.resolve()
        additive = verify_release(path)
        additive_dependencies.extend(
            [path, *local_dependencies(additive, work_root=corpus_work_root(path))]
        )
    cache.publish(
        "completed-mixture",
        _completed_mixture_request(spec, raw, output, manifest, tokenizer_sha),
        receipt,
        capture(
            [
                output,
                spec.family_inventory,
                spec.tokenizer_config,
                load_tokenizer_config(spec.tokenizer_config).output_dir,
                release,
                *local_dependencies(manifest, work_root=corpus_work_root(release)),
                *additive_dependencies,
            ]
        ),
        limit=MAX_RECEIPT_BYTES,
    )


def _verify_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt):
    from klab.corpus.completed_preparation import local_scope

    if local_scope(manifest) is None:
        raise ValueError("completed mixture has no matching local preparation scope")
    cached = _completed_mixture_cache(spec).load(
        "completed-mixture",
        _completed_mixture_request(spec, raw, output, manifest, tokenizer_sha),
        limit=MAX_RECEIPT_BYTES,
    )
    if cached is None or cached[0] != receipt:
        raise ValueError(
            "completed native mixture evidence unavailable; no automatic schedule replay"
        )
    return receipt


@functools.cache
def _stat_keyed_sha256(path: Path, stamp: tuple) -> str:
    return sha256_file(path)


def _inventory_sha256(path: Path) -> str:
    """The family inventory's digest, read once per process while the file is unchanged."""
    info = path.stat()
    return _stat_keyed_sha256(
        path, (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    )


def _native_mixture_request(spec, raw, output, release_id, tokenizer_sha):
    return {
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "release_id": release_id,
        "release_manifest_sha256": sha256_file(spec.release_path / "manifest.json"),
        "family_inventory_sha256": _inventory_sha256(spec.family_inventory),
        "tokenizer_sha256": tokenizer_sha,
        "output": str(output),
    }


def _early_assembly(spec, raw, output, release):
    """Name this request's checkpoint directory and refuse a conflicting one now.

    Uses only what is cheap to read (declaration, manifest, inventory and
    tokenizer bytes). Verification later confirms the tokenizer digest it
    assumed here.
    """
    from klab.corpus.mixture_pipeline import claim_assembly

    tokenizer_sha = sha256_file(
        load_tokenizer_config(spec.tokenizer_config).output_dir / "tokenizer.json"
    )
    # The manifest digest already determines the release id.
    request = _native_mixture_request(spec, raw, output, None, tokenizer_sha)
    del request["release_id"]
    return claim_assembly(output, request), tokenizer_sha


def _native_mixture_completion(spec, raw, output, manifest, tokenizer_sha):
    from klab.corpus.admission_inspection_cache import InspectionCache
    from klab.verification_proofs import ProofStore
    from klab.verifier_authority import scoped_authority

    # Native artifact verification owns its key beside the work, without a
    # writable user configuration directory or any supervisor dependency.
    proofs = ProofStore(output.parent)
    proofs.config_root = output.parent / "cache" / "native-mixture-identity"
    proofs.key_path = proofs.config_root / "klab" / "verification-key-v1"
    cache = InspectionCache(output.parent, proof_store=proofs)
    # Keyed on the modules that tokenize, mix and write a mixture, not on the
    # whole corpus package: unrelated Lab changes keep a completed mixture valid.
    cache.authority = {**cache.authority, "corpus": scoped_authority("native_mixture")}
    request = _native_mixture_request(
        spec, raw, output, manifest["release_id"], tokenizer_sha
    )
    return cache, request


def _seal_native_mixture(spec, raw, output, manifest, tokenizer_sha, receipt):
    from klab.corpus.admission_inspection_cache import capture

    cache, request = _native_mixture_completion(
        spec, raw, output, manifest, tokenizer_sha
    )
    cache.publish(
        "native-mixture", request, receipt, capture([output]), limit=MAX_RECEIPT_BYTES
    )


def verify_mixture(
    declaration: Path, output: Path, *, audit: bool = False, reseal: bool = False
) -> dict:
    """Authenticate completed execution; replay only for an explicit audit."""
    if audit and reseal:
        raise ValueError("--audit replays the mixture; --reseal never does")
    # One operation, so the release and export are authenticated once, however
    # many of the checks below reach for them.
    with _verification_operation():
        if reseal:
            return _reseal_mixture(declaration, output.resolve())
        return _verify_mixture(declaration, output, audit=audit)[0]


def mixture_train_file(receipt: dict) -> str:
    return TRAIN_SCHEDULE if "token_schedule" in receipt else TRAIN_TOKENS


def _reference_schedule(spec, output, receipt, tokenizer, tokenizer_sha):
    reference = spec.token_output == "references"
    if (
        reference != ("token_schedule" in receipt)
        or reference == ("train_tokens_sha256" in receipt)
        or receipt["tokenizer_sha256"] != tokenizer_sha
    ):
        raise ValueError("mixture token storage differs from declaration")
    if not reference:
        return None
    return verify_schedule(
        output,
        token_cache_root(spec.release_path.resolve(), output.parent),
        receipt,
        tokenizer.token_to_id("<eos>"),
    )


_DIGEST_CHUNK = 16 * 1024 * 1024


def _reseal_mixture(declaration: Path, output: Path) -> dict:
    """Re-record completion for an output a builder already sealed, without replay.

    Proves the files are byte-identical to the receipt and that the receipt
    matches this declaration, release and tokenizer. It does not re-derive the
    schedule: only --audit shows the receipt follows from the inputs.
    """
    with (output / "receipt.json").open("rb") as stream:
        raw_receipt = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw_receipt) > MAX_RECEIPT_BYTES:
        raise ValueError("mixture receipt-byte cap exceeded")
    receipt = json.loads(raw_receipt)
    raw = declaration.read_bytes()
    spec = MixtureDeclaration.model_validate(yaml.safe_load(raw))
    release = _verified_members(spec.release_path.resolve())
    manifest = release.manifest
    tokenizer, tokenizer_sha = _tokenizer(spec, release)
    expected = {
        "schema_version": 2,
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "release_manifest_sha256": sha256_file(spec.release_path / "manifest.json"),
        "tokenizer_sha256": tokenizer_sha,
        "seed": spec.seed,
        "requested_target_quotas": spec.target_quotas,
        "max_exposures": spec.max_exposures,
        "train_tokens_format": "int32-le",
        "order_policy": spec.order_policy,
    }
    if not spec.additional_releases:
        expected |= {
            "release_id": manifest["release_id"],
            "family_inventory_sha256": _inventory_sha256(spec.family_inventory),
        }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("mixture receipt does not match this declaration and release")
    actual = receipt["actual_target_tokens"]
    if (
        set(actual) != set(spec.target_quotas)
        or receipt["quota_shortfalls"]
        != {s: spec.target_quotas[s] - actual[s] for s in spec.target_quotas}
        or sum(receipt["source_target_tokens"].values()) != sum(actual.values())
        or any(
            receipt["unique_target_positions"][s] > actual[s]
            for s in spec.target_quotas
        )
        or receipt["train_tokens_bytes"] != 4 * sum(actual.values())
    ):
        raise ValueError("mixture receipt counts are not self-consistent")
    from concurrent.futures import ThreadPoolExecutor

    if (output / TRAIN_RECORDS).stat().st_size != receipt["train_records_bytes"]:
        raise ValueError("mixture token export digest mismatch")
    schedule = _reference_schedule(spec, output, receipt, tokenizer, tokenizer_sha)
    if schedule is not None:
        _verify_records(output, receipt)
        schedule.validate_counts(row["tokens"] for row in rows(output / TRAIN_RECORDS))
        schedule.require_unchanged()
    else:
        if (output / TRAIN_TOKENS).stat().st_size != receipt["train_tokens_bytes"]:
            raise ValueError("mixture token export digest mismatch")
        with ThreadPoolExecutor(2) as pool:
            tokens = pool.submit(sha256_file, output / TRAIN_TOKENS, _DIGEST_CHUNK)
            _verify_records(output, receipt)
            if tokens.result() != receipt["train_tokens_sha256"]:
                raise ValueError("mixture token export digest mismatch")
    if (
        receipt.get("completed_preparation", {}).get("policy")
        == "completed_local_preparation"
    ):
        _seal_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
    else:
        _seal_native_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
    return receipt


def _verify_mixture(declaration: Path, output: Path, *, audit: bool):
    output = output.resolve()
    receipt_path = output / "receipt.json"
    with receipt_path.open("rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("mixture receipt-byte cap exceeded")
    receipt = json.loads(raw)
    reference = "token_schedule" in receipt
    completed = (
        receipt.get("completed_preparation", {}).get("policy")
        == "completed_local_preparation"
    )
    completed_inline = completed and not reference and not audit
    if not completed_inline and (
        receipt.get("schema_version") != 2
        or (
            not reference
            and sha256_file(output / TRAIN_TOKENS) != receipt["train_tokens_sha256"]
        )
        or (output / TRAIN_RECORDS).stat().st_size != receipt["train_records_bytes"]
    ):
        raise ValueError("mixture token export digest mismatch")
    _verify_records(output, receipt, require_digest=not completed_inline)
    raw = declaration.read_bytes()
    spec = MixtureDeclaration.model_validate(yaml.safe_load(raw))
    release = _verified_members(spec.release_path.resolve())
    manifest = release.manifest
    tokenizer, tokenizer_sha = _tokenizer(spec, release)
    if not audit:
        if completed:
            _verify_completed_mixture(
                spec, raw, output, manifest, tokenizer_sha, receipt
            )
        else:
            cache, request = _native_mixture_completion(
                spec, raw, output, manifest, tokenizer_sha
            )
            cached = cache.load("native-mixture", request, limit=MAX_RECEIPT_BYTES)
            if cached is None or cached[0] != receipt:
                raise ValueError(
                    "native mixture completion unavailable or changed; run verify-mixture --reseal (cheap) or --audit (full replay)"
                )
        return receipt, _reference_schedule(
            spec, output, receipt, tokenizer, tokenizer_sha
        )
    schedule = _reference_schedule(spec, output, receipt, tokenizer, tokenizer_sha)
    with tempfile.TemporaryDirectory(
        prefix=".verify-mixture-", dir=output.parent
    ) as temp:
        with _verification_operation():
            replay = _materialize_mixture(
                declaration, Path(temp) / "replay", reuse_tokens=False
            )
        comparison = receipt
        if schedule is not None:
            digest = hashlib.sha256()
            for block in schedule.blocks(
                _DIGEST_CHUNK,
                lengths=(row["tokens"] for row in rows(output / TRAIN_RECORDS)),
            ):
                digest.update(block)
            comparison = {
                key: value for key, value in receipt.items() if key != "token_schedule"
            }
            comparison["train_tokens_sha256"] = digest.hexdigest()
        if replay != comparison:
            raise ValueError("mixture receipt differs from authenticated replay")
    # Explicit audit can recover an unsigned/legacy output without replacing it.
    if completed:
        _seal_completed_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
    else:
        _seal_native_mixture(spec, raw, output, manifest, tokenizer_sha, receipt)
    return receipt, schedule


def verify_mixture_tokenizer_origin(
    declaration: Path, tokenizer_path: Path, vocab_size: int
) -> dict:
    """Verify only the fit-tokenizer closure, not a consumer's mixture schedule.

    An explicit tokenizer origin reuses the declaration's native fit/observation
    verification. It does not claim that the mixture itself was materialized or
    that a post-run observation is a historical output-digest attestation.
    """
    spec = MixtureDeclaration.model_validate(yaml.safe_load(declaration.read_bytes()))
    release = spec.release_path.resolve()
    fit = load_tokenizer_config(spec.tokenizer_config)
    if (
        tokenizer_path.resolve() != (fit.output_dir / "tokenizer.json").resolve()
        or vocab_size != fit.vocab_size
    ):
        raise ValueError("tokenizer origin path or vocabulary mismatch")
    with _verification_operation():
        _, digest = _tokenizer(spec, _verified_members(release))
    result = json.loads(
        tokenizer_path.with_name("tokenizer_manifest.json").read_bytes()
    )
    if result["sha256"] != digest:
        raise ValueError("tokenizer origin digest mismatch")
    return result


def verify_mixture_dataset(
    dataset: DatasetConfig, tokenizer_path: Path, vocab_size: int
) -> dict:
    """Bind a token-ID consumer to the replayed mix and its fit tokenizer."""
    return verified_mixture_dataset(dataset, tokenizer_path, vocab_size)[0]


def verified_mixture_dataset(
    dataset: DatasetConfig, tokenizer_path: Path, vocab_size: int
):
    """Return the receipt and, for reference outputs, the checked range reader."""
    if (
        dataset.source != "local_token_mixture"
        or dataset.mixture_declaration_path is None
        or dataset.mixture_output_path is None
        or dataset.train_path is None
        or dataset.validation_path is None
    ):
        raise ValueError("complete local token-mixture dataset required")
    with _verification_operation():
        return _verify_mixture_dataset_members(dataset, tokenizer_path, vocab_size)


def _verify_mixture_dataset_members(
    dataset: DatasetConfig, tokenizer_path: Path, vocab_size: int
):
    spec = MixtureDeclaration.model_validate(
        yaml.safe_load(dataset.mixture_declaration_path.read_bytes())
    )
    receipt, schedule = _verify_mixture(
        dataset.mixture_declaration_path, dataset.mixture_output_path, audit=False
    )
    fit = load_tokenizer_config(spec.tokenizer_config)
    release = _verified_members(spec.release_path.resolve())
    validation = release.member("lm/validation.jsonl")
    if (
        dataset.train_path.resolve()
        != (dataset.mixture_output_path / mixture_train_file(receipt)).resolve()
        or dataset.validation_path.resolve() != validation.resolve()
        or dataset.revision != receipt["release_id"]
        or dataset.train_max_tokens != sum(receipt["actual_target_tokens"].values())
        or dataset.train_max_documents != receipt["records"]
        or tokenizer_path.resolve() != (fit.output_dir / "tokenizer.json").resolve()
        or vocab_size != fit.vocab_size
    ):
        raise ValueError("token mixture dataset or tokenizer identity mismatch")
    if spec.tokenizer_observation is not None:
        # verify_mixture has just checked the observed export and its generated
        # dataset against the retained release. Reuse those recorded validation
        # bounds rather than scanning the same release and held-out view again.
        expected_license = fit.dataset.license
        records = fit.dataset.validation_max_documents
        token_budget = fit.dataset.validation_max_tokens
    else:
        from klab.corpus.export import release_licenses, split_stats

        records, rendered_bytes = split_stats(validation, "lm")
        token_budget = rendered_bytes + records + 1
        expected_license = release_licenses(
            release.member("sources.json"), release.member("documents.jsonl")
        )
    if (
        dataset.license != expected_license
        or records != dataset.validation_max_documents
        or dataset.validation_max_tokens != token_budget
    ):
        raise ValueError("token mixture validation split count or budget mismatch")
    return receipt, schedule
