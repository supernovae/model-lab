"""Declared corpus readiness thresholds over verified, unique source evidence."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, WithJsonSchema, field_validator, model_validator

from klab.config.models import StrictModel
from klab.training.manifest import sha256_file

NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveWeight = Annotated[
    Decimal,
    Field(gt=0, allow_inf_nan=False),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0}),
]


class SourcePassRequirement(StrictModel):
    """Maximum projected per-domain reuse of unique source material.

    Overlapping source documents across domains are counted once *per domain*;
    these bounds do not describe train order or actual ingestion passes.
    """

    basis: Literal["bytes", "tokens"]
    requested_total: PositiveInt
    mixture: dict[str, PositiveWeight]
    max_required: PositiveInt

    @field_validator("mixture", mode="before")
    @classmethod
    def require_numeric_weights(cls, values: Any) -> Any:
        if isinstance(values, dict) and any(
            type(weight) not in (float, int, Decimal) for weight in values.values()
        ):
            raise ValueError("source pass mixture weights must be numeric")
        return values

    @model_validator(mode="after")
    def validate_mixture(self) -> SourcePassRequirement:
        if not self.mixture or any(not domain.strip() for domain in self.mixture):
            raise ValueError("source passes require named mixture domains")
        total = sum(
            (Fraction(weight) for weight in self.mixture.values()),
            Fraction(0),
        )
        if total != 1:
            raise ValueError("source pass mixture weights must sum to 1")
        return self


class CorpusReadinessPolicy(StrictModel):
    """Authored minima, independent of the frozen release's mixture."""

    min_unique_train_bytes_by_domain: dict[str, NonnegativeInt] = Field(
        default_factory=dict
    )
    min_unique_train_tokens_by_domain: dict[str, NonnegativeInt] = Field(
        default_factory=dict
    )
    required_nonzero_languages: tuple[str, ...] = ()
    required_nonzero_shapes: tuple[str, ...] = ()
    min_heldout_families: NonnegativeInt = 0
    passes: SourcePassRequirement | None = None

    @model_validator(mode="after")
    def validate_requirements(self) -> CorpusReadinessPolicy:
        if not any(
            (
                self.min_unique_train_bytes_by_domain,
                self.min_unique_train_tokens_by_domain,
                self.required_nonzero_languages,
                self.required_nonzero_shapes,
                self.min_heldout_families,
                self.passes is not None,
            )
        ):
            raise ValueError("corpus readiness needs at least one requirement")
        for labels in (
            self.min_unique_train_bytes_by_domain,
            self.min_unique_train_tokens_by_domain,
            self.required_nonzero_languages,
            self.required_nonzero_shapes,
        ):
            if any(not label.strip() for label in labels):
                raise ValueError("readiness labels must not be empty")
        if len(set(self.required_nonzero_languages)) != len(
            self.required_nonzero_languages
        ):
            raise ValueError("duplicate required language")
        if len(set(self.required_nonzero_shapes)) != len(self.required_nonzero_shapes):
            raise ValueError("duplicate required shape")
        return self


def measure_readiness(
    release: Path,
    policy: CorpusReadinessPolicy,
    tokenizer: Path | None = None,
    *,
    measurement_receipt: Path | None = None,
    measurement_sha256: str | None = None,
    _scratch: Path | None = None,
) -> dict[str, Any]:
    """Check unique source coverage without retaining document or lineage ledgers.

    A zero-availability mixture domain has no finite projected number of passes.
    Mixture weights do not partition overlapping source documents.
    """
    from klab.corpus.release import _verification_operation, verify_release
    from klab.corpus.token_denominator import (
        _measure_source_domains,
        read_source_token_receipt,
    )

    if (measurement_receipt is None) != (measurement_sha256 is None):
        raise ValueError("measurement receipt and SHA-256 must be supplied together")
    need_tokens = bool(policy.min_unique_train_tokens_by_domain) or (
        policy.passes is not None and policy.passes.basis == "tokens"
    )
    token_only = need_tokens and not (
        policy.min_unique_train_bytes_by_domain
        or policy.required_nonzero_languages
        or policy.required_nonzero_shapes
        or policy.min_heldout_families
        or (policy.passes is not None and policy.passes.basis == "bytes")
    )
    if measurement_receipt is not None and (not token_only or tokenizer is None):
        raise ValueError(
            "source-token receipts require a token-only policy and tokenizer"
        )
    release = Path(release)
    with _verification_operation():
        manifest = verify_release(release)
        if measurement_receipt is not None:
            path = Path(measurement_receipt)
            if sha256_file(path) != measurement_sha256:
                raise ValueError("source-token receipt SHA-256 mismatch")
            # The receipt pins exact policy bytes as well as its normalized model.
            header = json.loads(path.read_bytes())
            policy_path = Path(header["policy_path"])
            receipt = read_source_token_receipt(
                path, release, Path(tokenizer), policy_path
            )
            if sha256_file(path) != measurement_sha256:
                raise ValueError("source-token receipt changed during authentication")
            if receipt["policy"] != policy.model_dump(mode="json"):
                raise ValueError("source-token receipt inline policy mismatch")
            measured = receipt["domains"]
        else:
            measured = _measure_source_domains(
                release,
                tokenizer if need_tokens else None,
                policy,
                scratch=_scratch,
                all_domains=not token_only,
                max_document_source_bytes=None if not token_only else 1_048_576,
            )["domains"]
        if token_only:
            languages = None
            shapes = None
            heldout_count = None
        else:
            languages, shapes, heldout_count = _ancillary_facts(
                release, manifest, scratch=_scratch
            )

    # Historically mixed readiness lists only domains present in eligible rows.
    included = {
        domain: counts
        for domain, counts in measured.items()
        if token_only or counts["eligible_documents"] > 0
    }
    unique_bytes = {
        domain: counts["source_bytes"] for domain, counts in included.items()
    }
    unique_tokens = (
        {domain: counts["source_tokens"] for domain, counts in included.items()}
        if need_tokens and tokenizer is not None
        else {}
    )

    deficits: list[dict[str, Any]] = []

    def deficient(
        dimension: str, observed: int | None, required: int, domain: str | None = None
    ) -> None:
        if observed is None or observed < required:
            deficit: dict[str, Any] = {
                "dimension": dimension,
                "observed": observed,
                "required": required,
            }
            if domain is not None:
                deficit["domain"] = domain
            deficits.append(deficit)

    for domain, threshold in policy.min_unique_train_bytes_by_domain.items():
        deficient("unique_train_bytes", unique_bytes.get(domain), threshold, domain)
    for domain, threshold in policy.min_unique_train_tokens_by_domain.items():
        deficient(
            "unique_train_tokens",
            unique_tokens.get(domain) if tokenizer is not None else None,
            threshold,
            domain,
        )
    for language in policy.required_nonzero_languages:
        assert languages is not None
        deficient("train_language", languages.get(language, 0), 1, language)
    for shape in policy.required_nonzero_shapes:
        assert shapes is not None
        deficient("train_shape", shapes.get(shape, 0), 1, shape)
    if heldout_count is not None:
        deficient("heldout_families", heldout_count, policy.min_heldout_families)
    projected_passes: dict[str, int | None] = {}
    if policy.passes is not None:
        requirement = policy.passes
        amounts = unique_bytes if requirement.basis == "bytes" else unique_tokens
        for domain, weight in requirement.mixture.items():
            amount = (
                amounts.get(domain, 0)
                if requirement.basis == "bytes" or tokenizer is not None
                else None
            )
            if amount is None or amount == 0:
                projected_passes[domain] = None
                deficient("passes", None, requirement.max_required, domain)
                continue
            numerator, denominator = weight.as_integer_ratio()
            target = requirement.requested_total * numerator
            available = amount * denominator
            passes = (target + available - 1) // available
            projected_passes[domain] = passes
            if passes > requirement.max_required:
                deficits.append(
                    {
                        "dimension": "passes",
                        "observed": passes,
                        "required": requirement.max_required,
                        "domain": domain,
                    }
                )
    deficits.sort(key=lambda item: (item["dimension"], item.get("domain", "")))
    measurements = {
        "release_id": manifest["release_id"],
        "tokenizer_sha256": sha256_file(Path(tokenizer))
        if need_tokens and tokenizer is not None
        else None,
        "unique_train_bytes_by_domain": dict(sorted(unique_bytes.items())),
        "unique_train_tokens_by_domain": dict(sorted(unique_tokens.items()))
        if need_tokens and tokenizer is not None
        else None,
        "train_languages": (
            dict(sorted(languages.items())) if languages is not None else None
        ),
        "selected_train_shapes": (
            dict(sorted(shapes.items())) if shapes is not None else None
        ),
        "heldout_families": heldout_count,
        "projected_source_passes_by_domain": dict(sorted(projected_passes.items())),
    }
    return {
        "state": "BLOCKED" if deficits else "COMPLETE",
        "outcome": "EXPAND_MORE" if deficits else "READY_FOR_TOKENIZER",
        "deficits": deficits,
        "measurements": measurements,
    }


def _ancillary_facts(
    release: Path, manifest: dict[str, Any], *, scratch: Path | None = None
) -> tuple[dict[str, int], dict[str, int], int]:
    """Disk-backed joins preserve selected-view shape and heldout semantics."""
    from klab.corpus.release import _iter_rows
    from klab.corpus.token_denominator import _scratch_dir

    scratch = scratch if scratch is not None else _scratch_dir()
    scratch.mkdir(parents=True, exist_ok=True)
    languages: dict[str, int] = defaultdict(int)
    shapes: dict[str, int] = defaultdict(int)
    with (
        tempfile.TemporaryDirectory(prefix="readiness-", dir=scratch) as temporary,
        closing(sqlite3.connect(str(Path(temporary) / "facts.sqlite"))) as db,
    ):
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("CREATE TABLE language (digest TEXT PRIMARY KEY) WITHOUT ROWID")
        db.execute(
            "CREATE TABLE selected (record_id TEXT PRIMARY KEY, "
            "matched INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID"
        )
        db.execute("CREATE TABLE heldout (family TEXT PRIMARY KEY) WITHOUT ROWID")
        for row in _iter_rows(release / "documents.jsonl"):
            if (
                row["split"] != "train"
                or row["drop_reason"] is not None
                or not row["domains"]
            ):
                continue
            if db.execute(
                "INSERT OR IGNORE INTO language VALUES (?)", (row["content_sha256"],)
            ).rowcount:
                languages[row["language"]] += 1
        release_views = manifest["build_identity"]["release"]
        for view in ("lm", "chat"):
            if (
                release_views[view]["selected"]
                and "train" in release_views[view]["training_splits"]
            ):
                for link in _iter_rows(release / view / "train.lineage.jsonl"):
                    db.execute(
                        "INSERT OR IGNORE INTO selected (record_id) VALUES (?)",
                        (link["record_id"],),
                    )
        for row in _iter_rows(release / "lineage.jsonl"):
            record_id = row["record_id"]
            if db.execute(
                "UPDATE selected SET matched=1 WHERE record_id=? AND matched=0",
                (record_id,),
            ).rowcount:
                shapes[row["shape"]["id"]] += 1
            if row["split"] == "test" and row.get("drop_reason") is None:
                for family in row["source_family_ids"]:
                    if family:
                        db.execute(
                            "INSERT OR IGNORE INTO heldout VALUES (?)", (family,)
                        )
        if db.execute("SELECT 1 FROM selected WHERE matched=0 LIMIT 1").fetchone():
            raise ValueError("selected training lineage missing from release ledger")
        heldout_count = db.execute("SELECT count(*) FROM heldout").fetchone()[0]
    return languages, shapes, heldout_count
