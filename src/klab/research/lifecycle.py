"""Strict, read-only research lifecycle declarations and evidence projections."""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from klab.canonical import reject_duplicate_keys
from klab.config.loading import load_config, load_tokenizer_config
from klab.config.models import StrictModel
from klab.experiments.reporting import load_report_bundle
from klab.experiments.study import plan_study
from klab.research.catalog import ResearchEntry, list_research, load_research
from klab.training.manifest import (
    ArtifactIdentity,
    config_sha256,
    read_manifest,
    sha256_file,
)

_RESOURCE = Path(str(files("klab.research").joinpath("resources"))).resolve()
_MAX_BYTES = 2 * 1024 * 1024
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STAGES = ("mechanism", "micro", "replicate", "scale", "confirm", "promote")
_COSTS = ("mechanism", "short", "campaign")
_REQUIRED_ROLES = (
    "preparation",
    "training",
    "checkpoint",
    "validation",
    "capability",
    "generation",
    "resources",
    "static_report",
)
_INTEGRATION_CHECKS = (
    "lifecycle",
    "budget",
    "validation",
    "acquisition",
    "capability_controls",
    "generation",
    "resources",
    "static_report",
)


class LifecycleValidation(TypedDict):
    valid: bool
    diagnostics: list[dict[str, object]]
    availability: dict[str, dict[str, object]]


class _Strict(StrictModel):
    pass


class FileIdentity(StrictModel):
    relative_path: StrictStr
    sha256: StrictStr
    size_bytes: StrictInt

    @model_validator(mode="after")
    def validate_identity(self) -> FileIdentity:
        _safe_relative(self.relative_path, "artifact path")
        if not _SHA256.fullmatch(self.sha256) or self.size_bytes < 0:
            raise ValueError(
                "artifact requires a lowercase SHA-256 and nonnegative size"
            )
        return self

    def as_identity(self) -> ArtifactIdentity:
        return ArtifactIdentity(self.relative_path, self.sha256, self.size_bytes)


class EvidenceReference(StrictModel):
    id: StrictStr
    kind: Literal["report_bundle", "acceptance", "document", "source"]
    artifact: FileIdentity
    identity: StrictStr | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> EvidenceReference:
        _valid_id(self.id, "evidence id")
        if self.identity is not None and not _SHA256.fullmatch(self.identity):
            raise ValueError("evidence identity must be a lowercase SHA-256")
        if (
            self.kind == "report_bundle"
            and Path(self.artifact.relative_path).name != "manifest.json"
        ):
            raise ValueError("report_bundle artifact must name manifest.json")
        return self


class Profile(StrictModel):
    scale: StrictStr
    data: StrictStr
    note: StrictStr

    @model_validator(mode="after")
    def catalog_values(self) -> Profile:
        if self.scale not in {"smoke", "nano", "micro", "tiny"}:
            raise ValueError("profile scale must name an existing catalog scale")
        if self.data not in {"offline", "tinystories", "fineweb_edu"}:
            raise ValueError("profile data must name an existing catalog dataset")
        if not self.note.strip():
            raise ValueError("profile note must be explicit")
        return self


class Budget(StrictModel):
    max_steps: StrictInt
    max_tokens: StrictInt

    @model_validator(mode="after")
    def positive(self) -> Budget:
        if self.max_steps <= 0 or self.max_tokens <= 0:
            raise ValueError("baseline budgets must be positive")
        return self


class SourceRevision(StrictModel):
    git_commit: StrictStr | None = None
    git_dirty: StrictBool | None = None
    source_identity_sha256: StrictStr

    @model_validator(mode="after")
    def valid_digest(self) -> SourceRevision:
        if not _SHA256.fullmatch(self.source_identity_sha256):
            raise ValueError("source_identity_sha256 must be a lowercase SHA-256")
        if (self.git_commit is None) != (self.git_dirty is None):
            raise ValueError("git_commit and git_dirty must both be captured or absent")
        if self.git_commit is not None and not self.git_commit.strip():
            raise ValueError("git_commit must be nonempty when captured")
        return self


class CapabilityExpectation(StrictModel):
    card: StrictStr
    role: Literal["acquisition", "held_out", "stress"]
    expectation: StrictStr

    @model_validator(mode="after")
    def nonempty(self) -> CapabilityExpectation:
        if not self.card.strip() or not self.expectation.strip():
            raise ValueError("capability expectation card and prose must be nonempty")
        return self


class Baseline(StrictModel):
    id: StrictStr
    title: StrictStr
    purpose: StrictStr
    purpose_classification: Literal["capability_baseline", "learning_reference"] = (
        "capability_baseline"
    )
    endpoint_state: Literal[
        "not_assessed", "endpoint_reached_while_learning", "plateau_observed"
    ] = "not_assessed"
    status: Literal["candidate", "known_good", "superseded"]
    configuration: FileIdentity
    tokenizer_configuration: FileIdentity
    profile: Profile
    seeds: list[StrictInt]
    budget: Budget
    source_revision: SourceRevision | None
    required_evidence: dict[StrictStr, list[StrictStr]] = Field(default_factory=dict)
    capability_expectations: list[CapabilityExpectation]
    limitations: list[StrictStr]
    integration_study: FileIdentity
    reproduction: StrictStr
    promoted_from: StrictStr | None = None
    supersedes: StrictStr | None = None

    @model_validator(mode="after")
    def baseline_constraints(self) -> Baseline:
        _valid_id(self.id, "baseline id")
        if not self.title.strip() or not self.purpose.strip():
            raise ValueError("baseline title and purpose must be nonempty")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("baseline seeds must be unique and nonempty")
        if self.status in {"known_good", "superseded"} and self.source_revision is None:
            raise ValueError("established baselines require captured source_revision")
        _safe_relative(self.reproduction, "reproduction path")
        if any(not ids for ids in self.required_evidence.values()) or (
            self.status in {"known_good", "superseded"} and not self.required_evidence
        ):
            raise ValueError(
                "required_evidence roles must have nonempty reference lists"
            )
        if not self.capability_expectations and (
            self.purpose_classification != "learning_reference"
        ):
            raise ValueError(
                "capability baselines require nonempty capability expectations"
            )
        if self.purpose_classification == "learning_reference" and (
            self.capability_expectations
        ):
            raise ValueError(
                "learning references must not declare capability acceptance gates"
            )
        _unique(
            [item.card for item in self.capability_expectations],
            "baseline capability cards",
        )
        for ids in self.required_evidence.values():
            _unique(ids, "required evidence IDs")
        return self


class MaturityAxis(StrictModel):
    state: StrictStr
    finding_ids: list[StrictStr]


class Maturity(StrictModel):
    implementation: MaturityAxis | None = None
    capability: MaturityAxis | None = None
    efficiency: MaturityAxis | None = None
    portability: MaturityAxis | None = None

    @model_validator(mode="after")
    def validate_states(self) -> Maturity:
        allowed = {
            "implementation": {"proposed", "implemented", "verified"},
            "capability": {
                "untested",
                "not_supported_locally",
                "mixed",
                "supported_locally",
                "replicated",
                "scale_validated",
            },
            "efficiency": {
                "unmeasured",
                "estimated",
                "component_measured",
                "end_to_end_measured",
            },
            "portability": {
                "untested",
                "artifact_verified",
                "adapter_verified",
                "behavior_verified",
                "representation_verified",
            },
        }
        positive = {
            "implementation": {"implemented", "verified"},
            "capability": {
                "mixed",
                "supported_locally",
                "replicated",
                "scale_validated",
            },
            "efficiency": {"estimated", "component_measured", "end_to_end_measured"},
            "portability": {
                "artifact_verified",
                "adapter_verified",
                "behavior_verified",
                "representation_verified",
            },
        }
        for name, states in allowed.items():
            axis = getattr(self, name)
            if axis is not None and (
                axis.state not in states
                or (axis.state in positive[name] and not axis.finding_ids)
            ):
                raise ValueError(
                    f"maturity {name} state is invalid or lacks supporting findings"
                )
            if axis is not None:
                _unique(axis.finding_ids, f"{name} finding IDs")
        return self


class Blocker(StrictModel):
    id: StrictStr
    description: StrictStr
    resolution: StrictStr
    evidence_ids: list[StrictStr]

    @model_validator(mode="after")
    def valid_blocker(self) -> Blocker:
        _valid_id(self.id, "blocker id")
        if not self.description.strip() or not self.resolution.strip():
            raise ValueError("blocker description and resolution must be nonempty")
        _unique(self.evidence_ids, "blocker evidence IDs")
        return self


class NextTest(StrictModel):
    action: StrictStr
    stage: Literal["mechanism", "micro", "replicate", "scale", "confirm", "promote"]
    cost_class: Literal["mechanism", "short", "campaign"]
    prerequisites: list[StrictStr]
    blocker_ids: list[StrictStr]
    eligible_scales: list[StrictStr]
    reopens: list[StrictStr]
    design_change: StrictStr | None

    @model_validator(mode="after")
    def valid_next_test(self) -> NextTest:
        if not self.action.strip():
            raise ValueError("next-test action must be nonempty")
        for name in ("prerequisites", "blocker_ids", "eligible_scales", "reopens"):
            _unique(getattr(self, name), f"next-test {name}")
        if self.reopens and (
            self.design_change is None or not self.design_change.strip()
        ):
            raise ValueError("reopened findings require a design_change condition")
        if any(
            scale not in {"smoke", "nano", "micro", "tiny"}
            for scale in self.eligible_scales
        ):
            raise ValueError("eligible_scales must name catalog runnable scales")
        return self


class ResearchLifecycle(_Strict):
    entry: StrictStr
    entry_sha256: StrictStr
    baseline_id: StrictStr | None = None
    stage: Literal["mechanism", "micro", "replicate", "scale", "confirm", "promote"]
    maturity: Maturity
    finding_ids: list[StrictStr]
    next_test: NextTest | None
    blockers: list[Blocker]

    @model_validator(mode="after")
    def valid_entry(self) -> ResearchLifecycle:
        _validate_entry_reference(self.entry)
        if not _SHA256.fullmatch(self.entry_sha256):
            raise ValueError("entry_sha256 must be a lowercase SHA-256")
        _unique(self.finding_ids, "entry finding IDs")
        blocker_ids = [item.id for item in self.blockers]
        _unique(blocker_ids, "entry blocker IDs")
        return self


class DesignIdentity(StrictModel):
    study_sha256: StrictStr | None = None
    protocol_sha256: StrictStr | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> DesignIdentity:
        values = [
            value
            for value in (self.study_sha256, self.protocol_sha256)
            if value is not None
        ]
        if len(values) != 1 or not _SHA256.fullmatch(values[0]):
            raise ValueError("design_identity requires exactly one SHA-256 identity")
        return self


class Finding(_Strict):
    id: StrictStr
    entry: StrictStr
    recorded_at: StrictStr
    evidence_ids: list[StrictStr]
    tested_conditions: list[StrictStr]
    observation: StrictStr
    supported_claims: list[StrictStr]
    unsupported_claims: list[StrictStr]
    disposition: Literal[
        "rejected", "inconclusive", "learning", "replicate", "scale", "promote"
    ]
    decided_by: StrictStr
    rationale: StrictStr
    next_action: StrictStr
    reopen_conditions: list[StrictStr]
    design_identity: DesignIdentity | None = None

    @model_validator(mode="after")
    def valid_finding(self) -> Finding:
        _valid_id(self.id, "finding id")
        _validate_entry_reference(self.entry)
        try:
            parsed = date.fromisoformat(self.recorded_at)
        except ValueError as error:
            raise ValueError("recorded_at must be an ISO date") from error
        if parsed.isoformat() != self.recorded_at:
            raise ValueError("recorded_at must be an ISO date")
        if not self.tested_conditions or any(
            not item.strip() for item in self.tested_conditions
        ):
            raise ValueError("tested_conditions must be nonempty prose")
        if (
            not self.observation.strip()
            or not self.decided_by.strip()
            or not self.rationale.strip()
            or not self.next_action.strip()
        ):
            raise ValueError(
                "finding observation, attribution, rationale and next_action are required"
            )
        if (
            not self.evidence_ids
            or not self.supported_claims
            or not self.unsupported_claims
            or any(
                not item.strip()
                for item in self.supported_claims
                + self.unsupported_claims
                + self.reopen_conditions
            )
        ):
            raise ValueError(
                "findings need evidence IDs and nonempty supported, unsupported, and reopen prose"
            )
        if (
            self.disposition in {"rejected", "inconclusive", "learning"}
            and not self.reopen_conditions
        ):
            raise ValueError(
                "negative, inconclusive, and learning findings must declare reopen conditions"
            )
        for name in (
            "evidence_ids",
            "supported_claims",
            "unsupported_claims",
            "reopen_conditions",
        ):
            _unique(getattr(self, name), f"finding {name}")
        return self


class Regression(StrictModel):
    check: StrictStr
    evidence_ids: list[StrictStr]
    outcome: Literal["passed", "failed", "untested"]

    @model_validator(mode="after")
    def valid_regression(self) -> Regression:
        if not self.check.strip():
            raise ValueError("regression check must be nonempty")
        _unique(self.evidence_ids, "regression evidence IDs")
        return self


class ResourceTradeoff(StrictModel):
    description: StrictStr = Field(max_length=500)
    evidence_ids: list[StrictStr]

    @model_validator(mode="after")
    def valid_tradeoff(self) -> ResourceTradeoff:
        if not self.description.strip():
            raise ValueError("resource tradeoff description must be nonempty")
        _unique(self.evidence_ids, "resource tradeoff evidence IDs")
        return self


class Promotion(StrictModel):
    id: StrictStr
    from_baseline: StrictStr | None
    to_baseline: StrictStr
    mode: Literal["bootstrap", "parallel", "replace"]
    finding_ids: list[StrictStr]
    mechanism_change: StrictStr
    evidence_ids: list[StrictStr]
    regressions: list[Regression]
    untested_capabilities: list[StrictStr]
    resource_tradeoff: ResourceTradeoff
    integration_evidence_ids: list[StrictStr]
    decision: Literal["proposed", "accepted", "rejected"]
    decided_by: StrictStr
    rationale: StrictStr

    @model_validator(mode="after")
    def valid_promotion(self) -> Promotion:
        _valid_id(self.id, "promotion id")
        _valid_id(self.to_baseline, "target baseline id")
        if self.from_baseline is not None:
            _valid_id(self.from_baseline, "source baseline id")
            if self.from_baseline == self.to_baseline:
                raise ValueError("promotion cannot target its source baseline")
        if (self.mode == "bootstrap") != (self.from_baseline is None):
            raise ValueError("only bootstrap promotions omit from_baseline")
        for name in (
            "finding_ids",
            "evidence_ids",
            "untested_capabilities",
            "integration_evidence_ids",
        ):
            _unique(getattr(self, name), f"promotion {name}")
        checks = [item.check for item in self.regressions]
        _unique(checks, "promotion regression checks")
        if (
            not self.mechanism_change.strip()
            or not self.decided_by.strip()
            or not self.rationale.strip()
        ):
            raise ValueError("promotion decision text fields must be nonempty")
        if any(not item.strip() for item in self.untested_capabilities):
            raise ValueError("untested capability declarations must be nonempty prose")
        return self


class LifecycleRegistry(StrictModel):
    format: Literal["klab-research-lifecycle"]
    version: Literal[1]
    evidence: list[EvidenceReference]
    baselines: list[Baseline]
    entries: list[ResearchLifecycle]
    findings: list[Finding]
    promotions: list[Promotion]

    @model_validator(mode="before")
    @classmethod
    def strict_root_version(cls, value: object) -> object:
        if isinstance(value, dict) and (
            type(value.get("version")) is not int or value["version"] != 1
        ):
            raise ValueError("version must be integer 1")
        return value

    @model_validator(mode="after")
    def unique_record_ids(self) -> LifecycleRegistry:
        _unique([item.id for item in self.evidence], "evidence IDs")
        _unique([item.id for item in self.baselines], "baseline IDs")
        _unique([item.entry for item in self.entries], "entry IDs")
        _unique([item.id for item in self.findings], "finding IDs")
        _unique([item.id for item in self.promotions], "promotion IDs")
        return self


def _valid_id(value: str, field: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{field} must be safe kebab-case")


def _unique(values: list[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")


def _safe_relative(value: str, field: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _validate_entry_reference(value: str) -> None:
    if value.endswith(".json"):
        _safe_relative(value, "local research entry")
    else:
        _valid_id(value, "research entry")


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def _strict_json(path: Path, *, limit: int | None = None) -> tuple[object, bytes]:
    raw = path.read_bytes()
    if limit is not None and len(raw) > limit:
        raise ValueError(f"JSON exceeds {limit} bytes")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {path}: {error}") from error
    _reject_nonfinite(value)
    return value, raw


def load_lifecycle(reference: Path | None = None) -> LifecycleRegistry:
    """Load the empty current default or one explicit registry JSON document."""
    if reference is None:
        path = _RESOURCE / "lifecycle.json"
    else:
        path = Path(reference).expanduser()
        if path.is_symlink():
            raise ValueError("lifecycle metadata must not be a symlink")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"lifecycle metadata is unavailable: {path}")
    raw, _ = _strict_json(path, limit=_MAX_BYTES)
    try:
        return LifecycleRegistry.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid lifecycle metadata: {error}") from error


def _diagnostic(
    severity: str, code: str, message: str, *refs: str
) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "reference_ids": list(refs),
    }


def _resolve_under(root: Path, relative: str, field: str) -> Path:
    relative_path = _safe_relative(relative, field)
    root_resolved = root.resolve(strict=False)
    candidate = root_resolved
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{field} path traverses symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"{field} escapes evidence root: {relative}")
    return resolved


def _entry_digest(reference: str, evidence_root: Path) -> tuple[ResearchEntry, str]:
    if reference.endswith(".json"):
        path = _resolve_under(evidence_root, reference, "local research entry")
        raw, _ = _strict_json(path, limit=_MAX_BYTES)
        entry = ResearchEntry.model_validate(raw)
        if entry.id != Path(reference).stem:
            raise ValueError("local research entry ID does not match path stem")
        return entry, sha256_file(path)
    entry = load_research(reference)
    path = _RESOURCE / "catalog" / f"{reference}.json"
    return entry, sha256_file(path)


def _verify_evidence(
    registry: LifecycleRegistry, evidence_root: Path
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    availability: dict[str, dict[str, object]] = {}
    diagnostics: list[dict[str, object]] = []
    root = evidence_root.expanduser()
    for reference in registry.evidence:
        item: dict[str, object] = {
            "status": "unavailable",
            "basis": reference.kind,
            "reason": "artifact not found",
        }
        try:
            path = _resolve_under(
                root, reference.artifact.relative_path, f"evidence {reference.id}"
            )
        except FileNotFoundError:
            availability[reference.id] = item
            continue
        except (OSError, ValueError) as error:
            item = {"status": "invalid", "basis": reference.kind, "reason": str(error)}
            availability[reference.id] = item
            diagnostics.append(
                _diagnostic("error", "evidence_path_invalid", str(error), reference.id)
            )
            continue
        try:
            if not path.is_file():
                raise ValueError("evidence artifact must be a regular file")
            if (
                path.stat().st_size != reference.artifact.size_bytes
                or sha256_file(path) != reference.artifact.sha256
            ):
                raise ValueError("artifact byte identity mismatch")
            payload: object | None = None
            if reference.kind == "report_bundle":
                if path.name != "manifest.json":
                    raise ValueError("report reference must identify manifest.json")
                bundle = load_report_bundle(path.parent)
                manifest = bundle["manifest"]
                if not isinstance(manifest, dict):
                    raise ValueError("report bundle manifest must be an object")
                if (
                    reference.identity is not None
                    and manifest.get("bundle_sha256") != reference.identity
                ):
                    raise ValueError("embedded report bundle identity mismatch")
                payload = bundle
            elif reference.kind == "acceptance":
                payload, _ = _strict_json(path, limit=_MAX_BYTES)
                if not isinstance(payload, dict):
                    raise ValueError("acceptance evidence must be a JSON object")
                if "manifest_version" in payload:
                    read_manifest(path)
            if reference.identity is not None and reference.kind != "report_bundle":
                embedded = _embedded_identity(payload)
                if embedded is not None and embedded != reference.identity:
                    raise ValueError("embedded evidence identity mismatch")
            item = {
                "status": "verified",
                "basis": reference.kind,
                "path": str(path),
                "reason": None,
            }
        except (OSError, ValueError, TypeError, KeyError) as error:
            item = {
                "status": "invalid",
                "basis": reference.kind,
                "path": str(path),
                "reason": str(error),
            }
            diagnostics.append(
                _diagnostic(
                    "error",
                    "evidence_invalid",
                    f"Evidence {reference.id}: {error}",
                    reference.id,
                )
            )
        availability[reference.id] = item
    return availability, diagnostics


def _embedded_identity(payload: object) -> str | None:
    if isinstance(payload, dict):
        for key in ("bundle_sha256", "evidence_sha256", "report_sha256", "sha256"):
            value = payload.get(key)
            if isinstance(value, str) and _SHA256.fullmatch(value):
                return value
    return None


def _add_cross_reference_diagnostics(
    registry: LifecycleRegistry,
    evidence_root: Path,
    availability: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    evidence_ids = {item.id for item in registry.evidence}
    baseline_ids = {item.id for item in registry.baselines}
    finding_ids = {item.id for item in registry.findings}
    entry_map = {item.entry: item for item in registry.entries}
    baseline_map = {item.id: item for item in registry.baselines}
    finding_map = {item.id: item for item in registry.findings}
    promotion_map = {item.id: item for item in registry.promotions}
    for entry in registry.entries:
        try:
            loaded, digest = _entry_digest(entry.entry, evidence_root)
            if loaded.id != (
                Path(entry.entry).stem if entry.entry.endswith(".json") else entry.entry
            ):
                raise ValueError("catalog entry reference resolves to another ID")
            if digest != entry.entry_sha256:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "entry_identity_mismatch",
                        "ResearchEntry bytes differ from lifecycle entry_sha256.",
                        entry.entry,
                    )
                )
        except FileNotFoundError:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "entry_unavailable",
                    "Lifecycle ResearchEntry is unavailable.",
                    entry.entry,
                )
            )
        except (OSError, ValueError, TypeError) as error:
            diagnostics.append(
                _diagnostic("error", "entry_invalid", str(error), entry.entry)
            )
        if entry.baseline_id is not None and entry.baseline_id not in baseline_ids:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_reference_missing",
                    f"Unknown baseline {entry.baseline_id}.",
                    entry.entry,
                    entry.baseline_id,
                )
            )
        for finding_id in entry.finding_ids:
            finding = finding_map.get(finding_id)
            if finding is None:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "finding_reference_missing",
                        f"Unknown finding {finding_id}.",
                        entry.entry,
                        finding_id,
                    )
                )
            elif finding.entry != entry.entry:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "finding_scope_mismatch",
                        "Entry annotation cites a finding from another entry.",
                        entry.entry,
                        finding_id,
                    )
                )
        blocker_ids = {item.id for item in entry.blockers}
        if entry.next_test:
            for blocker_id in entry.next_test.blocker_ids:
                if blocker_id not in blocker_ids:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "blocker_reference_missing",
                            f"Unknown applicable blocker {blocker_id}.",
                            entry.entry,
                            blocker_id,
                        )
                    )
            for finding_id in entry.next_test.prerequisites + entry.next_test.reopens:
                finding = finding_map.get(finding_id)
                if finding is None:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "finding_reference_missing",
                            f"Unknown next-test finding {finding_id}.",
                            entry.entry,
                            finding_id,
                        )
                    )
                elif finding.entry != entry.entry:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "finding_scope_mismatch",
                            "Next-test references must remain within the research entry.",
                            entry.entry,
                            finding_id,
                        )
                    )
        for blocker in entry.blockers:
            _reference_diagnostics(
                blocker.evidence_ids, evidence_ids, diagnostics, entry.entry
            )
        for axis_name in ("implementation", "capability", "efficiency", "portability"):
            axis = getattr(entry.maturity, axis_name)
            if axis:
                for finding_id in axis.finding_ids:
                    if finding_id not in finding_ids:
                        diagnostics.append(
                            _diagnostic(
                                "error",
                                "finding_reference_missing",
                                f"Unknown maturity finding {finding_id}.",
                                entry.entry,
                                finding_id,
                            )
                        )
    for finding in registry.findings:
        _reference_diagnostics(
            finding.evidence_ids, evidence_ids, diagnostics, finding.id
        )
        if finding.entry not in entry_map:
            try:
                _entry_digest(finding.entry, evidence_root)
            except (OSError, ValueError) as error:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "finding_entry_invalid",
                        str(error),
                        finding.id,
                        finding.entry,
                    )
                )
    successor_edges: dict[str, str] = {}
    for baseline in registry.baselines:
        if baseline.promoted_from is not None:
            promotion = promotion_map.get(baseline.promoted_from)
            if promotion is None:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_reference_missing",
                        f"Unknown promotion {baseline.promoted_from}.",
                        baseline.id,
                        baseline.promoted_from,
                    )
                )
            elif (
                promotion.to_baseline != baseline.id or promotion.decision != "accepted"
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_promotion_mismatch",
                        "Baseline promoted_from must identify its accepted incoming promotion.",
                        baseline.id,
                        promotion.id,
                    )
                )
        if baseline.supersedes is not None:
            if baseline.supersedes not in baseline_ids:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_reference_missing",
                        f"Unknown superseded baseline {baseline.supersedes}.",
                        baseline.id,
                        baseline.supersedes,
                    )
                )
            elif baseline.supersedes == baseline.id:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_cycle",
                        "A baseline cannot supersede itself.",
                        baseline.id,
                    )
                )
            else:
                if baseline.supersedes in successor_edges:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "conflicting_replacements",
                            "More than one baseline replaces the same parent.",
                            baseline.id,
                            baseline.supersedes,
                        )
                    )
                successor_edges[baseline.supersedes] = baseline.id
        _reference_diagnostics(
            [eid for ids in baseline.required_evidence.values() for eid in ids],
            evidence_ids,
            diagnostics,
            baseline.id,
        )
        unknown_roles = sorted(set(baseline.required_evidence) - set(_REQUIRED_ROLES))
        if unknown_roles:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_role_unknown",
                    f"Unknown required evidence roles: {', '.join(unknown_roles)}.",
                    baseline.id,
                )
            )
        accepted_to_baseline = [
            item
            for item in registry.promotions
            if item.to_baseline == baseline.id and item.decision == "accepted"
        ]
        if (
            baseline.status in {"known_good", "superseded"}
            and len(accepted_to_baseline) != 1
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_promotion_count",
                    "Established baseline must have exactly one accepted incoming promotion.",
                    baseline.id,
                )
            )
        if baseline.status in {"known_good", "superseded"}:
            _check_baseline_evidence(baseline, registry, availability, diagnostics)
    accepted_replacements: dict[str, str] = {}
    for promotion in registry.promotions:
        _reference_diagnostics(
            promotion.evidence_ids
            + promotion.resource_tradeoff.evidence_ids
            + promotion.integration_evidence_ids,
            evidence_ids,
            diagnostics,
            promotion.id,
        )
        for regression in promotion.regressions:
            _reference_diagnostics(
                regression.evidence_ids, evidence_ids, diagnostics, promotion.id
            )
        if promotion.to_baseline not in baseline_ids:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_reference_missing",
                    f"Unknown promotion target {promotion.to_baseline}.",
                    promotion.id,
                    promotion.to_baseline,
                )
            )
        if (
            promotion.from_baseline is not None
            and promotion.from_baseline not in baseline_ids
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_reference_missing",
                    f"Unknown promotion parent {promotion.from_baseline}.",
                    promotion.id,
                    promotion.from_baseline,
                )
            )
        for finding_id in promotion.finding_ids:
            finding = finding_map.get(finding_id)
            if finding is None:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "finding_reference_missing",
                        f"Unknown finding {finding_id}.",
                        promotion.id,
                        finding_id,
                    )
                )
            elif (
                promotion.decision == "accepted"
                and promotion.mode != "bootstrap"
                and finding.disposition != "promote"
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_finding_disposition",
                        "Accepted non-bootstrap promotion needs a Finding dispositioned promote.",
                        promotion.id,
                        finding_id,
                    )
                )
        if (
            promotion.decision == "accepted"
            and promotion.mode != "bootstrap"
            and not promotion.finding_ids
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_finding_missing",
                    "Accepted non-bootstrap promotion requires at least one promoting Finding.",
                    promotion.id,
                )
            )
        if promotion.decision == "accepted":
            target = baseline_map.get(promotion.to_baseline)
            if target is None or target.status not in {"known_good", "superseded"}:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_target_not_established",
                        "An accepted promotion must name its established known-good baseline.",
                        promotion.id,
                        promotion.to_baseline,
                    )
                )
            elif target.promoted_from != promotion.id:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_target_lineage_mismatch",
                        "Accepted promotion and target promoted_from links must agree.",
                        promotion.id,
                        target.id,
                    )
                )
            elif target.status == "superseded" and not any(
                item.decision == "accepted"
                and item.mode == "replace"
                and item.from_baseline == target.id
                for item in registry.promotions
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_superseded_target_invalid",
                        "A superseded accepted target must have an accepted replacement.",
                        promotion.id,
                        target.id,
                    )
                )
            if (
                promotion.mode == "bootstrap"
                and target is not None
                and target.supersedes is not None
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "bootstrap_lineage_invalid",
                        "Bootstrap promotion target cannot supersede another baseline.",
                        promotion.id,
                        target.id,
                    )
                )
            if promotion.mode == "replace" and promotion.from_baseline is not None:
                parent = baseline_map.get(promotion.from_baseline)
                child = baseline_map.get(promotion.to_baseline)
                if (
                    parent is None
                    or child is None
                    or child.supersedes != parent.id
                    or parent.status != "superseded"
                ):
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "replacement_lineage_invalid",
                            "Accepted replacement requires a superseded parent and child.supersedes link.",
                            promotion.id,
                            promotion.from_baseline,
                            promotion.to_baseline,
                        )
                    )
                if promotion.from_baseline in accepted_replacements:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "conflicting_replacements",
                            "Parent has multiple accepted replacements.",
                            promotion.id,
                            promotion.from_baseline,
                        )
                    )
                accepted_replacements[promotion.from_baseline] = promotion.id
            if promotion.mode == "parallel" and promotion.from_baseline is not None:
                parent = baseline_map.get(promotion.from_baseline)
                child = baseline_map.get(promotion.to_baseline)
                if (
                    parent is None
                    or parent.status != "known_good"
                    or child is None
                    or child.status != "known_good"
                    or child.supersedes is not None
                ):
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "parallel_lineage_invalid",
                            "Parallel promotion preserves its known-good parent and adds a known-good child.",
                            promotion.id,
                        )
                    )
            _check_accepted_promotion(promotion, registry, availability, diagnostics)
    return diagnostics


def _reference_diagnostics(
    ids: list[str], known: set[str], diagnostics: list[dict[str, object]], owner: str
) -> None:
    for reference_id in ids:
        if reference_id not in known:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "evidence_reference_missing",
                    f"Unknown evidence reference {reference_id}.",
                    owner,
                    reference_id,
                )
            )


def _check_supersedes_cycles(edges: dict[str, str]) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for origin in edges:
        visited: set[str] = set()
        current = origin
        while current in edges:
            if current in visited:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_cycle",
                        "Baseline replacement ancestry contains a cycle.",
                        origin,
                        current,
                    )
                )
                break
            visited.add(current)
            current = edges[current]
    return diagnostics


def _reference_status(
    ids: list[str], availability: dict[str, dict[str, object]]
) -> list[str]:
    return [
        reference_id
        for reference_id in ids
        if availability.get(reference_id, {}).get("status") != "verified"
    ]


def _check_baseline_evidence(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    missing_roles = sorted(set(_REQUIRED_ROLES) - set(baseline.required_evidence))
    if missing_roles:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_roles_incomplete",
                f"Required evidence roles are missing: {', '.join(missing_roles)}.",
                baseline.id,
            )
        )
    evidence_map = {item.id: item for item in registry.evidence}
    for role in _REQUIRED_ROLES:
        ids = baseline.required_evidence.get(role, [])
        if not ids:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_role_empty",
                    f"Required evidence role {role} has no evidence.",
                    baseline.id,
                    role,
                )
            )
            continue
        if _reference_status(ids, availability):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_evidence_unavailable",
                    f"Required evidence role {role} is not fully verified.",
                    baseline.id,
                    *ids,
                )
            )
        if role == "static_report" and not any(
            evidence_map.get(eid) and evidence_map[eid].kind == "report_bundle"
            for eid in ids
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_report_missing",
                    "static_report role must reference a validated report bundle.",
                    baseline.id,
                    *ids,
                )
            )
    report_refs = [
        evidence_map[evidence_id]
        for evidence_id in baseline.required_evidence.get("static_report", [])
        if evidence_id in evidence_map
        and evidence_map[evidence_id].kind == "report_bundle"
    ]
    if len(report_refs) != 1:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_report_count",
                "Established baseline must reference exactly one static report bundle.",
                baseline.id,
            )
        )
    required_ids = {
        eid
        for role in _REQUIRED_ROLES
        for eid in baseline.required_evidence.get(role, [])
    }
    central_acceptances: list[tuple[str, dict[str, object]]] = []
    for evidence_id in sorted(required_ids):
        reference = evidence_map.get(evidence_id)
        if (
            reference is None
            or reference.kind != "acceptance"
            or availability.get(evidence_id, {}).get("status") != "verified"
        ):
            continue
        raw, _ = _load_evidence_json(reference, availability)
        if isinstance(raw, dict) and isinstance(raw.get("checks"), dict):
            central_acceptances.append((evidence_id, raw))
    if len(central_acceptances) != 1:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_acceptance_count",
                "Established baseline requires exactly one captured acceptance record.",
                baseline.id,
            )
        )
    for _, raw in central_acceptances:
        _validate_acceptance_checks(raw, baseline, registry, availability, diagnostics)
    manifest_refs = [
        evidence_map.get(evidence_id)
        for evidence_id in baseline.required_evidence.get("training", [])
    ]
    copied_manifests = [
        reference
        for reference in manifest_refs
        if reference is not None
        and reference.kind == "acceptance"
        and isinstance(availability.get(reference.id, {}).get("path"), str)
    ]
    copied_manifest_count = sum(
        _is_run_manifest(reference, availability) for reference in copied_manifests
    )
    if copied_manifest_count != len(baseline.seeds):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_manifest_missing",
                "Training role must include one copied, byte-verified RunManifest per seed.",
                baseline.id,
            )
        )


def _load_evidence_json(
    reference: EvidenceReference, availability: dict[str, dict[str, object]]
) -> tuple[object, Path | None]:
    path = availability.get(reference.id, {}).get("path")
    if not isinstance(path, str):
        return None, None
    try:
        raw, _ = _strict_json(Path(path), limit=_MAX_BYTES)
        return raw, Path(path)
    except OSError, ValueError:
        return None, None


def _is_run_manifest(
    reference: EvidenceReference, availability: dict[str, dict[str, object]]
) -> bool:
    raw, _ = _load_evidence_json(reference, availability)
    return (
        isinstance(raw, dict)
        and type(raw.get("manifest_version")) is int
        and raw.get("manifest_version") == 1
        and isinstance(raw.get("effective_config"), dict)
        and isinstance(raw.get("source_identity"), dict)
    )


def _validate_copied_manifest(
    reference: EvidenceReference,
    availability: dict[str, dict[str, object]],
    *,
    run_id: str,
    report_identity: dict[str, object],
    backend: object,
    source_revision: SourceRevision,
    diagnostics: list[dict[str, object]],
    baseline_id: str,
) -> None:
    path = availability.get(reference.id, {}).get("path")
    if not isinstance(path, str):
        return
    try:
        manifest = read_manifest(Path(path))
    except (OSError, ValueError, TypeError) as error:
        diagnostics.append(
            _diagnostic(
                "error",
                "copied_manifest_invalid",
                str(error),
                baseline_id,
                reference.id,
            )
        )
        return
    if manifest.get("run_id") != run_id:
        diagnostics.append(
            _diagnostic(
                "error",
                "copied_manifest_run_mismatch",
                "Copied RunManifest belongs to another run.",
                baseline_id,
                reference.id,
            )
        )
    effective = manifest.get("effective_config")
    report_config = report_identity.get("config")
    if (
        not isinstance(effective, dict)
        or not isinstance(report_config, dict)
        or config_sha256(effective) != config_sha256(report_config)
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "copied_manifest_config_mismatch",
                "Copied RunManifest configuration differs from report.",
                baseline_id,
                reference.id,
            )
        )
    source = manifest.get("source_identity")
    source_sha = source.get("sha256") if isinstance(source, dict) else None
    if (
        source_sha != report_identity.get("source_identity_sha256")
        or manifest.get("git_commit") != source_revision.git_commit
        or manifest.get("git_dirty") is not source_revision.git_dirty
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "copied_manifest_source_mismatch",
                "Copied RunManifest source/Git identity differs from report and baseline.",
                baseline_id,
                reference.id,
            )
        )
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("backend") != backend:
        diagnostics.append(
            _diagnostic(
                "error",
                "copied_manifest_backend_mismatch",
                "Copied RunManifest backend differs from report.",
                baseline_id,
                reference.id,
            )
        )
    artifact_hashes = {
        item.get("relative_path"): item.get("sha256")
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    data = report_identity.get("data_sha256")
    expected = {
        "tokenizer.json": report_identity.get("tokenizer_sha256"),
        "data/train.npy": data.get("train") if isinstance(data, dict) else None,
        "data/validation.npy": data.get("validation")
        if isinstance(data, dict)
        else None,
    }
    for name, digest in expected.items():
        if artifact_hashes.get(name) != digest:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "copied_manifest_artifact_mismatch",
                    f"Copied RunManifest artifact {name} differs from report.",
                    baseline_id,
                    reference.id,
                )
            )


def _validate_acceptance_checks(
    raw: dict[str, object],
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    checks = raw.get("checks")
    if not isinstance(checks, dict):
        diagnostics.append(
            _diagnostic(
                "error",
                "acceptance_checks_missing",
                "Acceptance record has no checks object.",
                baseline.id,
            )
        )
        return
    evidence_map = {item.id: item for item in registry.evidence}
    for name in _INTEGRATION_CHECKS:
        item = checks.get(name)
        if not isinstance(item, dict) or item.get("outcome") != "passed":
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_failed",
                    f"Acceptance check {name} must have passed.",
                    baseline.id,
                    name,
                )
            )
            continue
        refs = item.get("evidence_ids")
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(ref, str) or ref not in evidence_map for ref in refs)
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_evidence_missing",
                    f"Acceptance check {name} needs identity-bound evidence IDs.",
                    baseline.id,
                    name,
                )
            )
        elif not any(
            evidence_map[ref].kind in {"report_bundle", "acceptance"}
            and availability.get(ref, {}).get("status") == "verified"
            for ref in refs
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_evidence_unverified",
                    f"Acceptance check {name} needs available non-document evidence.",
                    baseline.id,
                    name,
                )
            )
        if (
            name == "static_report"
            and isinstance(refs, list)
            and not any(
                evidence_map[ref].kind == "report_bundle"
                and availability.get(ref, {}).get("status") == "verified"
                for ref in refs
                if isinstance(ref, str) and ref in evidence_map
            )
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_report_evidence_missing",
                    "static_report acceptance check must cite a verified report bundle.",
                    baseline.id,
                )
            )
        bound = item.get("identity")
        if (
            not isinstance(bound, dict)
            or not isinstance(bound.get("run_id"), str)
            or not isinstance(bound.get("checkpoint_sha256"), str)
            or not _SHA256.fullmatch(bound.get("checkpoint_sha256", ""))
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_identity_missing",
                    f"Acceptance check {name} must bind run_id and checkpoint_sha256.",
                    baseline.id,
                    name,
                )
            )
    for extra, item in checks.items():
        if extra not in _INTEGRATION_CHECKS and not isinstance(item, dict):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_invalid",
                    f"Acceptance check {extra} must be an object.",
                    baseline.id,
                    str(extra),
                )
            )


def _check_accepted_promotion(
    promotion: Promotion,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    evidence_map = {item.id: item for item in registry.evidence}
    target = next(
        (item for item in registry.baselines if item.id == promotion.to_baseline), None
    )
    target_evidence = (
        {
            evidence_id
            for references in target.required_evidence.values()
            for evidence_id in references
        }
        if target is not None
        else set()
    )
    if not promotion.evidence_ids or _reference_status(
        promotion.evidence_ids, availability
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "promotion_evidence_missing",
                "Accepted promotion requires available supporting evidence.",
                promotion.id,
            )
        )
    elif not _measured_evidence(promotion.evidence_ids, evidence_map, availability):
        diagnostics.append(
            _diagnostic(
                "error",
                "promotion_evidence_unmeasured",
                "Accepted promotion must cite a verified report-backed finding, not documents alone.",
                promotion.id,
            )
        )
    integration_acceptances: list[tuple[str, dict[str, object]]] = []
    for evidence_id in promotion.integration_evidence_ids:
        reference = evidence_map.get(evidence_id)
        if evidence_id not in target_evidence:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_integration_scope_mismatch",
                    "Integration acceptance must also be included in target baseline required_evidence.",
                    promotion.id,
                    evidence_id,
                )
            )
        if (
            reference is None
            or reference.kind != "acceptance"
            or availability.get(evidence_id, {}).get("status") != "verified"
        ):
            continue
        raw, _ = _load_evidence_json(reference, availability)
        if isinstance(raw, dict):
            integration_acceptances.append((evidence_id, raw))
    if not integration_acceptances:
        diagnostics.append(
            _diagnostic(
                "error",
                "promotion_integration_missing",
                "Accepted promotion requires an available acceptance record in integration_evidence_ids.",
                promotion.id,
            )
        )
    for evidence_id, raw in integration_acceptances:
        checks = raw.get("checks")
        if not isinstance(checks, dict):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_integration_checks_missing",
                    "Integration acceptance has no checks object.",
                    promotion.id,
                    evidence_id,
                )
            )
            continue
        for name in _INTEGRATION_CHECKS:
            check = checks.get(name)
            if (
                not isinstance(check, dict)
                or check.get("outcome") != "passed"
                or not isinstance(check.get("evidence_ids"), list)
                or not check["evidence_ids"]
                or not isinstance(check.get("identity"), dict)
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_integration_incomplete",
                        f"Accepted promotion requires passed identity-bound integration check {name}.",
                        promotion.id,
                        evidence_id,
                        name,
                    )
                )
    supplied = {row.check: row for row in promotion.regressions}
    if not supplied:
        diagnostics.append(
            _diagnostic(
                "error",
                "promotion_regressions_empty",
                "Accepted decisions must preserve a nonempty required regression set.",
                promotion.id,
            )
        )
    for row in promotion.regressions:
        if (
            row.outcome != "passed"
            or not row.evidence_ids
            or _reference_status(row.evidence_ids, availability)
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_regression_incomplete",
                    f"Accepted regression {row.check} needs available passed evidence.",
                    promotion.id,
                    row.check,
                )
            )
        elif not _measured_evidence(row.evidence_ids, evidence_map, availability):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_regression_identity_missing",
                    f"Regression {row.check} is not bound to verified report-backed evidence.",
                    promotion.id,
                    row.check,
                )
            )
    if promotion.mode != "bootstrap" and promotion.from_baseline is not None:
        parent_promotions = [
            item
            for item in registry.promotions
            if item.to_baseline == promotion.from_baseline
            and item.decision == "accepted"
        ]
        required_checks = {
            row.check
            for parent_promotion in parent_promotions
            for row in parent_promotion.regressions
        }
        if not required_checks:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_parent_regressions_missing",
                    "Parent accepted decision records no required regression checks.",
                    promotion.id,
                    promotion.from_baseline,
                )
            )
        if set(supplied) != required_checks:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_regression_set_mismatch",
                    "Accepted promotion must preserve exactly the parent baseline's required regression set.",
                    promotion.id,
                    promotion.from_baseline,
                )
            )
        for check in sorted(required_checks):
            regression = supplied.get(check)
            if (
                regression is None
                or regression.outcome != "passed"
                or not regression.evidence_ids
                or _reference_status(regression.evidence_ids, availability)
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "promotion_regression_incomplete",
                        f"Required parent regression {check} lacks available passed evidence.",
                        promotion.id,
                        check,
                    )
                )
        if (
            not promotion.resource_tradeoff.evidence_ids
            or _reference_status(promotion.resource_tradeoff.evidence_ids, availability)
            or not _measured_evidence(
                promotion.resource_tradeoff.evidence_ids, evidence_map, availability
            )
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "promotion_resource_tradeoff_missing",
                    "Non-bootstrap promotion requires available report-backed resource tradeoff evidence.",
                    promotion.id,
                )
            )


def _report_for_baseline(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    refs = {item.id: item for item in registry.evidence}
    for evidence_id in baseline.required_evidence.get("static_report", []):
        reference = refs.get(evidence_id)
        path = availability.get(evidence_id, {}).get("path")
        if (
            reference is not None
            and reference.kind == "report_bundle"
            and isinstance(path, str)
        ):
            try:
                bundle = load_report_bundle(Path(path).parent)
            except OSError, ValueError:
                continue
            report = bundle.get("report")
            manifest = bundle.get("manifest")
            return (
                report if isinstance(report, dict) else None,
                manifest if isinstance(manifest, dict) else None,
            )
    return None, None


def _validate_learning_reference_capture(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    report, _ = _report_for_baseline(baseline, registry, availability)
    if not isinstance(report, dict):
        return
    inputs = report.get("inputs")
    runs = report.get("runs")
    local_validation = report.get("local_validation")
    if report.get("comparisons") != []:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_report_comparisons_invalid",
                "Learning reference report must contain no comparison pairs.",
                baseline.id,
            )
        )
        return
    if (
        not isinstance(inputs, dict)
        or inputs.get("verification_scope") != "report_plus_local_checkpoint_validation"
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_report_scope_invalid",
                "Learning reference report must include local checkpoint validation.",
                baseline.id,
            )
        )
        return
    if not isinstance(runs, list) or len(runs) != len(baseline.seeds):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_endpoint_invalid",
                "Learning reference report must contain one run per declared seed.",
                baseline.id,
            )
        )
        return
    configuration_path = availability.get("__baseline_configuration__", {}).get("path")
    study_path = availability.get("__baseline_integration_study__", {}).get("path")
    try:
        if not isinstance(configuration_path, str) or not isinstance(study_path, str):
            raise TypeError(
                "baseline configuration or integration study is unavailable"
            )
        configuration = load_config(Path(configuration_path))
        base_config = configuration.model_dump(mode="json")
        study = plan_study(Path(study_path))
        if (
            len(study.expanded) != len(baseline.seeds)
            or study.comparisons
            or study.pairs
        ):
            raise ValueError(
                "integration study must plan one unpaired run per declared seed"
            )
        if inputs.get("study_sha256") != study.study_sha256:
            raise ValueError(
                "static report study identity differs from baseline integration study"
            )
    except (OSError, ValueError, TypeError) as error:
        diagnostics.append(
            _diagnostic("error", "baseline_config_mismatch", str(error), baseline.id)
        )
        return

    run_by_seed: dict[int, dict[str, object]] = {}
    shared_tokenizer: str | None = None
    shared_data: dict[str, object] | None = None
    for run in runs:
        if not isinstance(run, dict):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_endpoint_invalid",
                    "Learning reference report contains a malformed run.",
                    baseline.id,
                )
            )
            continue
        run_id = run.get("run_id")
        identity = run.get("identity")
        endpoint = run.get("endpoint_status")
        config_identity = identity.get("config") if isinstance(identity, dict) else None
        seed = (
            config_identity.get("seed") if isinstance(config_identity, dict) else None
        )
        if (
            not isinstance(run_id, str)
            or not isinstance(identity, dict)
            or not isinstance(endpoint, dict)
            or endpoint.get("status") != "complete"
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed in run_by_seed
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_endpoint_invalid",
                    "Learning reference report has a duplicate, incomplete, or unbound seed.",
                    baseline.id,
                    str(run_id),
                )
            )
            continue
        run_by_seed[seed] = run
        if seed not in baseline.seeds:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_seed_mismatch",
                    "Report run seed is not declared by the baseline.",
                    baseline.id,
                    run_id,
                )
            )
            continue
        try:
            expected_config = copy.deepcopy(base_config)
            expected_config["seed"] = seed
            logging = expected_config.get("logging")
            captured_logging = config_identity.get("logging")
            if not isinstance(logging, dict) or not isinstance(captured_logging, dict):
                raise TypeError("configuration logging identity is malformed")
            logging["root_dir"] = captured_logging.get("root_dir")
            if config_sha256(expected_config) != config_sha256(config_identity):
                raise ValueError(
                    "run configuration differs from the frozen learning reference"
                )
        except (ValueError, TypeError) as error:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_config_mismatch",
                    str(error),
                    baseline.id,
                    run_id,
                )
            )
        training_config = config_identity.get("training")
        if not isinstance(training_config, dict) or (
            config_identity.get("seed") != seed
            or training_config.get("max_steps") != baseline.budget.max_steps
            or training_config.get("max_tokens") != baseline.budget.max_tokens
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_config_mismatch",
                    "Run configuration omits the declared seed or training budget.",
                    baseline.id,
                    run_id,
                )
            )
        if (
            endpoint.get("configured_max_steps") != baseline.budget.max_steps
            or endpoint.get("configured_max_tokens") != baseline.budget.max_tokens
            or endpoint.get("step") != baseline.budget.max_steps
            or endpoint.get("tokens_seen") != baseline.budget.max_tokens
            or identity.get("run_id") != run_id
            or identity.get("step") != endpoint.get("step")
            or identity.get("tokens_seen") != endpoint.get("tokens_seen")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_budget_mismatch",
                    "Run did not reach the exact declared step and token budgets.",
                    baseline.id,
                    run_id,
                )
            )
        runtime = identity.get("runtime")
        config_runtime = config_identity.get("runtime")
        if (
            not isinstance(runtime, dict)
            or not isinstance(config_runtime, dict)
            or runtime.get("backend") != config_runtime.get("backend")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_runtime_mismatch",
                    "Run backend differs from its effective configuration.",
                    baseline.id,
                    run_id,
                )
            )
        if (
            identity.get("source_identity_sha256")
            != baseline.source_revision.source_identity_sha256
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_source_identity_mismatch",
                    "Run source package identity differs from the baseline.",
                    baseline.id,
                    run_id,
                )
            )
        tokenizer = identity.get("tokenizer_sha256")
        data = identity.get("data_sha256")
        if not isinstance(tokenizer, str) or not _SHA256.fullmatch(tokenizer):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_tokenizer_identity_missing",
                    "Run tokenizer identity is missing or malformed.",
                    baseline.id,
                    run_id,
                )
            )
        elif shared_tokenizer is None:
            shared_tokenizer = tokenizer
        elif shared_tokenizer != tokenizer:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_tokenizer_identity_mismatch",
                    "Replicated runs do not share one tokenizer identity.",
                    baseline.id,
                    run_id,
                )
            )
        if not isinstance(data, dict) or any(
            not _SHA256.fullmatch(str(data.get(key, "")))
            for key in ("train", "validation")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_data_identity_missing",
                    "Run data identity is missing or malformed.",
                    baseline.id,
                    run_id,
                )
            )
        elif shared_data is None:
            shared_data = data
        elif shared_data != data:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_data_identity_mismatch",
                    "Replicated runs do not share one train/validation data identity.",
                    baseline.id,
                    run_id,
                )
            )
        validation_row = (
            local_validation.get(run_id) if isinstance(local_validation, dict) else None
        )
        experiment_evidence = (
            validation_row.get("experiment_evidence")
            if isinstance(validation_row, dict)
            else None
        )
        checkpoints = (
            experiment_evidence.get("checkpoints")
            if isinstance(experiment_evidence, dict)
            else None
        )
        checkpoint_rows = checkpoints if isinstance(checkpoints, list) else []
        initial = next(
            (
                row
                for row in checkpoint_rows
                if isinstance(row, dict)
                and row.get("step") == 0
                and row.get("verified") is True
            ),
            None,
        )
        terminal = next(
            (
                row
                for row in checkpoint_rows
                if isinstance(row, dict)
                and row.get("step") == endpoint.get("step")
                and row.get("digest") == identity.get("checkpoint_sha256")
                and row.get("verified") is True
            ),
            None,
        )
        if (
            not isinstance(validation_row, dict)
            or validation_row.get("status") != "validated"
            or not isinstance(experiment_evidence, dict)
            or experiment_evidence.get("verified_checkpoints") is not True
            or initial is None
            or terminal is None
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_checkpoint_validation_failed",
                    "Each seed needs verified initial and terminal checkpoint records.",
                    baseline.id,
                    run_id,
                )
            )
        else:
            initial_loss = initial.get("validation_loss")
            terminal_loss = terminal.get("validation_loss")
            reported_loss = (
                run.get("validation", {}).get("loss")
                if isinstance(run.get("validation"), dict)
                else None
            )
            if (
                not isinstance(initial_loss, (int, float))
                or isinstance(initial_loss, bool)
                or not math.isfinite(initial_loss)
                or not isinstance(terminal_loss, (int, float))
                or isinstance(terminal_loss, bool)
                or not math.isfinite(terminal_loss)
                or terminal_loss >= initial_loss
                or reported_loss != terminal_loss
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_validation_gate_failed",
                        "Each seed needs a finite terminal held-out loss below its initial loss.",
                        baseline.id,
                        run_id,
                    )
                )
    if list(run_by_seed) != baseline.seeds:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_seed_mismatch",
                "Report runs must appear once each in the declared seed order.",
                baseline.id,
            )
        )
        return

    _validate_learning_reference_manifests(
        baseline, registry, availability, run_by_seed, diagnostics
    )
    _validate_learning_reference_acceptance(
        baseline, registry, availability, report, run_by_seed, diagnostics
    )


def _validate_learning_reference_manifests(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    runs_by_seed: dict[int, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    evidence_map = {item.id: item for item in registry.evidence}
    for evidence_id in baseline.required_evidence.get("training", []):
        reference = evidence_map.get(evidence_id)
        if reference is None or not _is_run_manifest(reference, availability):
            continue
        path = availability.get(evidence_id, {}).get("path")
        if not isinstance(path, str):
            continue
        try:
            manifest = read_manifest(Path(path))
        except (OSError, ValueError, TypeError) as error:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "copied_manifest_invalid",
                    str(error),
                    baseline.id,
                    evidence_id,
                )
            )
            continue
        run_id = manifest.get("run_id")
        matches = [run for run in runs_by_seed.values() if run.get("run_id") == run_id]
        if len(matches) != 1:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "copied_manifest_run_mismatch",
                    "Copied RunManifest does not match exactly one reported seed.",
                    baseline.id,
                    evidence_id,
                )
            )
            continue
        run = matches[0]
        identity = run.get("identity")
        runtime = identity.get("runtime") if isinstance(identity, dict) else None
        backend = runtime.get("backend") if isinstance(runtime, dict) else None
        if not isinstance(identity, dict) or baseline.source_revision is None:
            continue
        _validate_copied_manifest(
            reference,
            availability,
            run_id=str(run_id),
            report_identity=identity,
            backend=backend,
            source_revision=baseline.source_revision,
            diagnostics=diagnostics,
            baseline_id=baseline.id,
        )


def _validate_learning_reference_acceptance(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    report: dict[str, object],
    runs_by_seed: dict[int, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    evidence_map = {item.id: item for item in registry.evidence}
    role_ids = {
        evidence_id
        for references in baseline.required_evidence.values()
        for evidence_id in references
    }
    acceptances: list[dict[str, object]] = []
    for evidence_id in role_ids:
        reference = evidence_map.get(evidence_id)
        if (
            reference is None
            or reference.kind != "acceptance"
            or availability.get(evidence_id, {}).get("status") != "verified"
        ):
            continue
        raw, _ = _load_evidence_json(reference, availability)
        if isinstance(raw, dict) and isinstance(raw.get("checks"), dict):
            acceptances.append(raw)
    if len(acceptances) != 1:
        return
    acceptance = acceptances[0]
    captured = acceptance.get("seed_runs")
    if not isinstance(captured, list) or len(captured) != len(runs_by_seed):
        diagnostics.append(
            _diagnostic(
                "error",
                "acceptance_seed_identity_missing",
                "Canonical acceptance must bind every replicated seed.",
                baseline.id,
            )
        )
        return
    captured_by_seed = {
        item.get("seed"): item for item in captured if isinstance(item, dict)
    }
    for seed, run in runs_by_seed.items():
        identity = run.get("identity")
        endpoint = run.get("endpoint_status")
        row = captured_by_seed.get(seed)
        if not isinstance(identity, dict) or not isinstance(endpoint, dict):
            continue
        expected = {
            "seed": seed,
            "run_id": run.get("run_id"),
            "config_sha256": run.get("config_sha256"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "source_identity_sha256": identity.get("source_identity_sha256"),
            "tokenizer_sha256": identity.get("tokenizer_sha256"),
            "data_sha256": identity.get("data_sha256"),
            "backend": identity.get("runtime", {}).get("backend")
            if isinstance(identity.get("runtime"), dict)
            else None,
            "step": endpoint.get("step"),
            "tokens_seen": endpoint.get("tokens_seen"),
        }
        if not isinstance(row, dict) or any(
            row.get(key) != value for key, value in expected.items()
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_seed_identity_mismatch",
                    "Canonical acceptance seed identity differs from its static report run.",
                    baseline.id,
                    str(seed),
                )
            )
    primary_seed = baseline.seeds[0]
    primary = captured_by_seed.get(primary_seed)
    if not isinstance(primary, dict):
        return
    primary_run = runs_by_seed.get(primary_seed)
    run_identity = (
        primary_run.get("identity") if isinstance(primary_run, dict) else None
    )
    endpoint = (
        primary_run.get("endpoint_status") if isinstance(primary_run, dict) else None
    )
    runtime = run_identity.get("runtime") if isinstance(run_identity, dict) else None
    primary_identity = acceptance.get("identity")
    expected_identity = {
        "run_id": primary.get("run_id"),
        "config_sha256": primary.get("config_sha256"),
        "tokenizer_sha256": run_identity.get("tokenizer_sha256")
        if isinstance(run_identity, dict)
        else None,
        "data_sha256": run_identity.get("data_sha256")
        if isinstance(run_identity, dict)
        else None,
        "source_identity_sha256": run_identity.get("source_identity_sha256")
        if isinstance(run_identity, dict)
        else None,
        "checkpoint_sha256": primary.get("checkpoint_sha256"),
        "backend": runtime.get("backend") if isinstance(runtime, dict) else None,
    }
    if not isinstance(primary_identity, dict) or any(
        primary_identity.get(key) != value for key, value in expected_identity.items()
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "acceptance_report_identity_mismatch",
                "Acceptance identity must bind the first declared seed and all package/data identities.",
                baseline.id,
            )
        )
    counters = acceptance.get("counters")
    if (
        not isinstance(endpoint, dict)
        or not isinstance(counters, dict)
        or counters.get("step") != endpoint.get("step")
        or counters.get("tokens_seen") != endpoint.get("tokens_seen")
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "acceptance_budget_counters_mismatch",
                "Acceptance counters differ from the primary static-report endpoint.",
                baseline.id,
            )
        )
    if acceptance.get("source_revision") != baseline.source_revision.model_dump(
        mode="json"
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_source_identity_mismatch",
                "Acceptance source revision differs from the promoted baseline.",
                baseline.id,
            )
        )
    checks = acceptance.get("checks")
    if not isinstance(checks, dict):
        return
    for name in _INTEGRATION_CHECKS:
        check = checks.get(name)
        identities = check.get("seed_identities") if isinstance(check, dict) else None
        if not isinstance(identities, list) or len(identities) != len(captured):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_seed_identity_missing",
                    f"Acceptance check {name} must bind every seed.",
                    baseline.id,
                    name,
                )
            )
            continue
        expected_rows = [
            {
                "seed": row.get("seed"),
                "run_id": row.get("run_id"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
            }
            for row in captured
            if isinstance(row, dict)
        ]
        if identities != expected_rows:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_check_seed_identity_mismatch",
                    f"Acceptance check {name} seed identities differ from the captured runs.",
                    baseline.id,
                    name,
                )
            )

    generation_refs = baseline.required_evidence.get("generation", [])
    generation_captures: dict[str, dict[str, object]] = {}
    for evidence_id in generation_refs:
        reference = evidence_map.get(evidence_id)
        if (
            reference is None
            or reference.kind != "acceptance"
            or availability.get(evidence_id, {}).get("status") != "verified"
        ):
            continue
        payload, _ = _load_evidence_json(reference, availability)
        if not isinstance(payload, dict):
            continue
        identity = payload.get("identity")
        run_id = identity.get("run_id") if isinstance(identity, dict) else None
        if not isinstance(run_id, str):
            continue
        if payload.get("format") == "klab-generation-capture":
            generation_captures[run_id] = payload
    expected_run_ids = {str(run.get("run_id")) for run in runs_by_seed.values()}
    if set(generation_captures) != expected_run_ids:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_capture_count",
                "Generation role must contain one bound fixed-prompt capture per seed.",
                baseline.id,
            )
        )
        return
    for run in runs_by_seed.values():
        run_id = str(run.get("run_id"))
        identity = run.get("identity")
        if not isinstance(identity, dict):
            continue
        _validate_generation_capture(
            generation_captures[run_id], baseline, identity, run_id, diagnostics
        )


def _validate_baseline_capture(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    if baseline.status not in {"known_good", "superseded"}:
        return
    if baseline.source_revision is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_source_revision_missing",
                "Established baseline lacks its captured source revision.",
                baseline.id,
            )
        )
        return
    if baseline.purpose_classification == "learning_reference":
        _validate_learning_reference_capture(
            baseline, registry, availability, diagnostics
        )
        return
    report, _ = _report_for_baseline(baseline, registry, availability)
    if not isinstance(report, dict):
        return
    if report.get("comparisons") != []:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_report_comparisons_invalid",
                "Reference baseline report must contain no comparison pairs.",
                baseline.id,
            )
        )
        return
    inputs = report.get("inputs")
    runs = report.get("runs")
    local_validation = report.get("local_validation")
    if (
        not isinstance(inputs, dict)
        or inputs.get("verification_scope") != "report_plus_local_checkpoint_validation"
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_report_scope_invalid",
                "Baseline report must include local checkpoint validation.",
                baseline.id,
            )
        )
        return
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_endpoint_invalid",
                "Reference baseline report must contain its one complete integration run.",
                baseline.id,
            )
        )
        return
    run = runs[0]
    run_id = run.get("run_id")
    identity = run.get("identity")
    endpoint = run.get("endpoint_status")
    if (
        not isinstance(run_id, str)
        or not isinstance(identity, dict)
        or not isinstance(endpoint, dict)
        or endpoint.get("status") != "complete"
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_endpoint_invalid",
                "Baseline report has no complete identity-bound endpoint.",
                baseline.id,
            )
        )
        return
    validation_row = (
        local_validation.get(run_id) if isinstance(local_validation, dict) else None
    )
    if (
        not isinstance(validation_row, dict)
        or validation_row.get("status") != "validated"
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_checkpoint_validation_failed",
                "Baseline report local_validation must be validated for the run.",
                baseline.id,
                run_id,
            )
        )
    evidence = (
        validation_row.get("experiment_evidence")
        if isinstance(validation_row, dict)
        else None
    )
    checkpoints = evidence.get("checkpoints") if isinstance(evidence, dict) else None
    if (
        not isinstance(checkpoints, list)
        or not isinstance(evidence, dict)
        or evidence.get("verified_checkpoints") is not True
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_checkpoint_records_missing",
                "Report does not retain successful checkpoint verification results.",
                baseline.id,
                run_id,
            )
        )
    else:
        terminal_digest = identity.get("checkpoint_sha256")
        terminal_step = endpoint.get("step")
        initial_ok = any(
            isinstance(row, dict)
            and row.get("step") == 0
            and row.get("verified") is True
            for row in checkpoints
        )
        terminal_ok = any(
            isinstance(row, dict)
            and row.get("step") == terminal_step
            and row.get("digest") == terminal_digest
            and row.get("verified") is True
            for row in checkpoints
        )
        if not initial_ok or not terminal_ok:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_checkpoint_verification_failed",
                    "Initial and terminal checkpoint verification must both be captured successfully.",
                    baseline.id,
                    run_id,
                )
            )
    config_identity = identity.get("config")
    if not isinstance(config_identity, dict):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_config_identity_missing",
                "Report run identity omits the complete configuration.",
                baseline.id,
                run_id,
            )
        )
        return
    config_path = availability.get("__baseline_configuration__", {}).get("path")
    try:
        if not isinstance(config_path, str):
            raise TypeError("configuration file is unavailable")
        configuration = load_config(Path(config_path))
        expected_config = configuration.model_dump(mode="json")
        if config_sha256(expected_config) != config_sha256(config_identity):
            raise ValueError("run configuration differs from baseline configuration")
        if (
            configuration.training.max_steps != baseline.budget.max_steps
            or configuration.training.max_tokens != baseline.budget.max_tokens
        ):
            raise ValueError("RunConfig budget differs from baseline budget")
        if baseline.seeds != [configuration.seed]:
            raise ValueError("baseline seeds differ from the single integration run")
        if (
            endpoint.get("configured_max_steps") != baseline.budget.max_steps
            or endpoint.get("configured_max_tokens") != baseline.budget.max_tokens
        ):
            raise ValueError("report endpoint budget differs from baseline budget")
        step = endpoint.get("step")
        tokens_seen = endpoint.get("tokens_seen")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step > baseline.budget.max_steps
        ):
            raise ValueError("observed endpoint step exceeds baseline budget")
        if (
            not isinstance(tokens_seen, int)
            or isinstance(tokens_seen, bool)
            or tokens_seen > baseline.budget.max_tokens
        ):
            raise ValueError("observed endpoint tokens exceed baseline budget")
        if (
            step != baseline.budget.max_steps
            and tokens_seen != baseline.budget.max_tokens
        ):
            raise ValueError("configured step or token budget was not reached")
        runtime = identity.get("runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("backend") != configuration.runtime.backend
        ):
            raise ValueError("captured backend identity differs from RunConfig")
        if (
            identity.get("source_identity_sha256")
            != baseline.source_revision.source_identity_sha256
        ):
            raise ValueError(
                "captured source identity differs from baseline source_revision"
            )
        if not _SHA256.fullmatch(str(identity.get("tokenizer_sha256", ""))):
            raise ValueError("captured tokenizer identity is missing or malformed")
        data_identity = identity.get("data_sha256")
        if not isinstance(data_identity, dict) or any(
            not _SHA256.fullmatch(str(data_identity.get(key, "")))
            for key in ("train", "validation")
        ):
            raise ValueError("captured training or validation data identity is missing")
        tokenizer_path = availability.get(
            "__baseline_tokenizer_configuration__", {}
        ).get("path")
        if not isinstance(tokenizer_path, str):
            raise TypeError("tokenizer training configuration is unavailable")
        tokenizer_config = load_tokenizer_config(Path(tokenizer_path))
        if tokenizer_config.vocab_size != configuration.model.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes differ")
        study_path = availability.get("__baseline_integration_study__", {}).get("path")
        if not isinstance(study_path, str):
            raise TypeError("integration study is unavailable")
        study = plan_study(Path(study_path))
        if len(study.expanded) != 1 or study.comparisons or study.pairs:
            raise ValueError(
                "baseline integration study must plan exactly one run and zero comparisons"
            )
        if inputs.get("study_sha256") != study.study_sha256:
            raise ValueError(
                "static report study identity differs from baseline integration study"
            )
    except (OSError, ValueError, TypeError) as error:
        diagnostics.append(
            _diagnostic("error", "baseline_config_mismatch", str(error), baseline.id)
        )
    _validate_acceptance_report_identity(
        baseline, registry, availability, report, diagnostics
    )
    report_limitations = report.get("limitations")
    if isinstance(report_limitations, list) and report_limitations:
        diagnostics.append(
            _diagnostic(
                "warning",
                "report_applicability_limitations",
                "Static report declares applicability/identity limitations; inspect the linked report before interpreting claims.",
                baseline.id,
            )
        )
    quantities = report.get("implementation_quantities")
    if isinstance(quantities, dict) and quantities.get("kind") != "measured":
        diagnostics.append(
            _diagnostic(
                "advisory",
                "resource_measurement_limited",
                "End-to-end implementation telemetry is unavailable or non-measured in the linked report.",
                baseline.id,
            )
        )


def _validate_generation_capture(
    payload: object,
    baseline: Baseline,
    report_identity: dict[str, object],
    run_id: str,
    diagnostics: list[dict[str, object]],
) -> None:
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "klab-generation-capture"
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_capture_invalid",
                "Generation evidence must be a structured klab-generation-capture record.",
                baseline.id,
            )
        )
        return
    if type(payload.get("version")) is not int or payload["version"] != 1:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_capture_invalid",
                "Generation capture version must be integer 1.",
                baseline.id,
            )
        )
        return
    runtime = report_identity.get("runtime")
    expected_identity = {
        "run_id": run_id,
        "checkpoint_sha256": report_identity.get("checkpoint_sha256"),
        "tokenizer_sha256": report_identity.get("tokenizer_sha256"),
        "backend": runtime.get("backend") if isinstance(runtime, dict) else None,
    }
    if payload.get("identity") != expected_identity:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_identity_mismatch",
                "Generation capture must bind the static report run, checkpoint, tokenizer, and backend.",
                baseline.id,
                run_id,
            )
        )
    options = payload.get("options")
    model = report_identity.get("config")
    model = model.get("model") if isinstance(model, dict) else None
    max_seq_len = model.get("max_seq_len") if isinstance(model, dict) else None
    expected_seed = (
        42042 if baseline.purpose_classification == "learning_reference" else 0
    )
    token_limit_valid = (
        isinstance(options, dict)
        and type(options.get("max_new_tokens")) is int
        and (
            options["max_new_tokens"] == 32
            if baseline.purpose_classification == "learning_reference"
            else 1 <= options["max_new_tokens"] <= 32
        )
    )
    if (
        not token_limit_valid
        or not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or options["max_new_tokens"] > max_seq_len
        or isinstance(options.get("temperature"), bool)
        or options.get("temperature") != 0.0
        or type(options.get("top_k")) is not int
        or options["top_k"] != 0
        or type(options.get("seed")) is not int
        or options["seed"] != expected_seed
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_options_invalid",
                "Generation must use the captured fixed greedy options and fit max_seq_len.",
                baseline.id,
                run_id,
            )
        )
    outputs = payload.get("outputs")
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("prompt"), str)
            or not item["prompt"].strip()
            or not isinstance(item.get("output"), str)
            or not item["output"].startswith(item["prompt"])
            for item in outputs
        )
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_outputs_invalid",
                "Generation capture must contain prompt-bound exact CLI outputs.",
                baseline.id,
                run_id,
            )
        )


def _validate_chat_transcript_capture(
    payload: object,
    report_identity: dict[str, object],
    run_id: str,
    options: object,
    baseline: Baseline,
    diagnostics: list[dict[str, object]],
) -> None:
    if not isinstance(payload, dict) or payload.get("format") != "chat_transcript_v1":
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_chat_transcript_invalid",
                "Generation role must retain the structured chat transcript.",
                baseline.id,
                run_id,
            )
        )
        return
    identity = payload.get("identity")
    report_runtime = report_identity.get("runtime")
    report_config = report_identity.get("config")
    model = report_config.get("model") if isinstance(report_config, dict) else None
    max_seq_len = model.get("max_seq_len") if isinstance(model, dict) else None
    training_runtime = (
        identity.get("training_runtime") if isinstance(identity, dict) else None
    )
    captured_config = identity.get("config") if isinstance(identity, dict) else None
    expected_identity = {
        "run_id": run_id,
        "checkpoint_sha256": report_identity.get("checkpoint_sha256"),
        "tokenizer_sha256": report_identity.get("tokenizer_sha256"),
        "source_identity_sha256": report_identity.get("source_identity_sha256"),
        "data_sha256": report_identity.get("data_sha256"),
    }
    if (
        not isinstance(identity, dict)
        or any(identity.get(key) != value for key, value in expected_identity.items())
        or not isinstance(report_runtime, dict)
        or not isinstance(training_runtime, dict)
        or training_runtime.get("backend") != report_runtime.get("backend")
        or not isinstance(captured_config, dict)
        or not isinstance(report_config, dict)
        or config_sha256(captured_config) != config_sha256(report_config)
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_chat_transcript_identity_mismatch",
                "Chat transcript must bind the static report run, checkpoint, tokenizer, source, data, configuration, and backend.",
                baseline.id,
                run_id,
            )
        )
    if payload.get("generation") != options:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_chat_transcript_options_mismatch",
                "Chat transcript generation settings must match the fixed generation capture.",
                baseline.id,
                run_id,
            )
        )
    turns = payload.get("turns")
    if (
        not isinstance(turns, list)
        or not turns
        or not isinstance(options, dict)
        or type(options.get("max_new_tokens")) is not int
        or not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or options["max_new_tokens"] > max_seq_len
        or any(
            not isinstance(turn, dict)
            or not isinstance(turn.get("user"), str)
            or not turn["user"].strip()
            or not isinstance(turn.get("prompt"), str)
            or not isinstance(turn.get("response"), str)
            or type(turn.get("prompt_tokens")) is not int
            or turn["prompt_tokens"] <= 0
            or turn["prompt_tokens"] + options["max_new_tokens"] > max_seq_len
            for turn in turns
        )
    ):
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_chat_transcript_turns_invalid",
                "Chat transcript turns must be complete and fit prompt plus completion budgets.",
                baseline.id,
                run_id,
            )
        )


def _validate_generation_evidence(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    report_identity: dict[str, object],
    run_id: str,
    diagnostics: list[dict[str, object]],
) -> None:
    evidence_map = {item.id: item for item in registry.evidence}
    captures: list[dict[str, object]] = []
    transcripts: list[dict[str, object]] = []
    for evidence_id in baseline.required_evidence.get("generation", []):
        reference = evidence_map.get(evidence_id)
        if (
            reference is None
            or reference.kind != "acceptance"
            or availability.get(evidence_id, {}).get("status") != "verified"
        ):
            continue
        payload, _ = _load_evidence_json(reference, availability)
        if isinstance(payload, dict):
            if payload.get("format") == "klab-generation-capture":
                captures.append(payload)
            elif payload.get("format") == "chat_transcript_v1":
                transcripts.append(payload)
    if len(captures) != 1:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_generation_capture_count",
                "Generation role must contain exactly one verified structured capture.",
                baseline.id,
            )
        )
        return
    payload = captures[0]
    _validate_generation_capture(
        payload, baseline, report_identity, run_id, diagnostics
    )
    if len(transcripts) != 1:
        diagnostics.append(
            _diagnostic(
                "error",
                "baseline_chat_transcript_count",
                "Generation role must contain exactly one verified chat transcript.",
                baseline.id,
            )
        )
    elif len(captures) == 1:
        _validate_chat_transcript_capture(
            transcripts[0],
            report_identity,
            run_id,
            captures[0].get("options"),
            baseline,
            diagnostics,
        )


def _validate_acceptance_report_identity(
    baseline: Baseline,
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    report: dict[str, object],
    diagnostics: list[dict[str, object]],
) -> None:
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs or not isinstance(runs[0], dict):
        return
    run = runs[0]
    run_id = run.get("run_id")
    identity = run.get("identity")
    endpoint = run.get("endpoint_status")
    if (
        not isinstance(run_id, str)
        or not isinstance(identity, dict)
        or not isinstance(endpoint, dict)
    ):
        return
    checkpoint = identity.get("checkpoint_sha256")
    report_config = identity.get("config")
    report_config_sha = (
        config_sha256(report_config) if isinstance(report_config, dict) else None
    )
    report_runtime = identity.get("runtime")
    backend = (
        report_runtime.get("backend") if isinstance(report_runtime, dict) else None
    )
    role_ids = [
        eid
        for role in _REQUIRED_ROLES
        for eid in baseline.required_evidence.get(role, [])
    ]
    evidence_map = {item.id: item for item in registry.evidence}
    for evidence_id in dict.fromkeys(role_ids):
        reference = evidence_map.get(evidence_id)
        path = availability.get(evidence_id, {}).get("path")
        if (
            reference is None
            or reference.kind != "acceptance"
            or not isinstance(path, str)
        ):
            continue
        try:
            raw, _ = _strict_json(Path(path))
        except OSError, ValueError:
            continue
        if isinstance(raw, dict) and raw.get("manifest_version") == 1:
            if evidence_id in baseline.required_evidence.get("training", []):
                if baseline.source_revision is None:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "baseline_source_revision_missing",
                            "Training manifest cannot be checked without source_revision.",
                            baseline.id,
                            evidence_id,
                        )
                    )
                else:
                    _validate_copied_manifest(
                        reference,
                        availability,
                        run_id=run_id,
                        report_identity=identity,
                        backend=backend,
                        diagnostics=diagnostics,
                        source_revision=baseline.source_revision,
                        baseline_id=baseline.id,
                    )
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("checks"), dict):
            continue
        if raw.get("run_id") != run_id:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_run_identity_mismatch",
                    "Acceptance run_id does not match static report.",
                    baseline.id,
                    evidence_id,
                )
            )
        captured = raw.get("identity")
        expected = {
            "run_id": run_id,
            "config_sha256": report_config_sha,
            "tokenizer_sha256": identity.get("tokenizer_sha256"),
            "data_sha256": identity.get("data_sha256"),
            "source_identity_sha256": identity.get("source_identity_sha256"),
            "checkpoint_sha256": checkpoint,
            "backend": backend,
        }
        if not isinstance(captured, dict) or any(
            captured.get(key) != value for key, value in expected.items()
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_report_identity_mismatch",
                    "Acceptance capture identities differ from the static report run.",
                    baseline.id,
                    evidence_id,
                )
            )
        source_revision = raw.get("source_revision")
        if baseline.source_revision is not None and (
            not isinstance(source_revision, dict)
            or source_revision.get("git_commit") != baseline.source_revision.git_commit
            or source_revision.get("git_dirty") != baseline.source_revision.git_dirty
            or baseline.source_revision.source_identity_sha256
            != identity.get("source_identity_sha256")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_source_identity_mismatch",
                    "Captured Git/source revision differs from baseline source_revision.",
                    baseline.id,
                    evidence_id,
                )
            )
        counters = raw.get("counters")
        if (
            not isinstance(counters, dict)
            or counters.get("step") != endpoint.get("step")
            or counters.get("tokens_seen") != endpoint.get("tokens_seen")
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "acceptance_budget_counters_mismatch",
                    "Acceptance counters differ from the report endpoint.",
                    baseline.id,
                    evidence_id,
                )
            )
        checks = raw["checks"]
        for check_name in _INTEGRATION_CHECKS:
            check = checks.get(check_name)
            if not isinstance(check, dict):
                continue
            bound = check.get("identity")
            if (
                not isinstance(bound, dict)
                or bound.get("run_id") != run_id
                or bound.get("checkpoint_sha256") != checkpoint
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "acceptance_checkpoint_identity_mismatch",
                        f"Acceptance check {check_name} differs from report run/checkpoint.",
                        baseline.id,
                        evidence_id,
                        check_name,
                    )
                )
            refs = check.get("evidence_ids")
            if isinstance(refs, list):
                for ref in refs:
                    if (
                        not isinstance(ref, str)
                        or ref not in evidence_map
                        or availability.get(ref, {}).get("status") != "verified"
                    ):
                        diagnostics.append(
                            _diagnostic(
                                "error",
                                "acceptance_check_evidence_unavailable",
                                f"Acceptance check {check_name} cites unavailable evidence.",
                                baseline.id,
                                evidence_id,
                                check_name,
                                str(ref),
                            )
                        )
        capabilities = run.get("capabilities")
        if not isinstance(capabilities, dict):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_capability_results_missing",
                    "Report has no capability result records.",
                    baseline.id,
                    run_id,
                )
            )
            continue
        expected_cards = {item.card for item in baseline.capability_expectations}
        if set(capabilities) != expected_cards:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_capability_expectations_mismatch",
                    "Report capability results must match the declared baseline expectations exactly.",
                    baseline.id,
                    run_id,
                )
            )
        retention = capabilities.get("chat-alias-retention-v1")
        retention_result = (
            retention.get("result") if isinstance(retention, dict) else None
        )
        retention_rows = (
            retention_result.get("results")
            if isinstance(retention_result, dict)
            else None
        )
        if (
            not isinstance(retention, dict)
            or retention.get("valid") is not True
            or retention.get("case_count") != 24
            or not isinstance(retention_rows, list)
            or len(retention_rows) != 24
            or any(
                not isinstance(row, dict) or not isinstance(row.get("passed"), bool)
                for row in retention_rows
            )
            or not isinstance(retention.get("passed"), int)
            or isinstance(retention.get("passed"), bool)
            or retention.get("passed")
            != sum(
                row["passed"]
                for row in retention_rows
                if isinstance(row, dict) and isinstance(row.get("passed"), bool)
            )
            or retention["passed"] < 12
        ):
            diagnostics.append(
                _diagnostic(
                    "error",
                    "baseline_acquisition_gate_failed",
                    "Acquisition result must retain all 24 cases and pass at least 12.",
                    baseline.id,
                    run_id,
                )
            )
        recall = capabilities.get("chat-alias-recall-v1")
        override = capabilities.get("chat-context-override-v1")
        for name, result in (
            ("chat-alias-recall-v1", recall),
            ("chat-context-override-v1", override),
        ):
            nested = result.get("result") if isinstance(result, dict) else None
            rows = nested.get("results") if isinstance(nested, dict) else None
            count = result.get("case_count") if isinstance(result, dict) else None
            if (
                not isinstance(result, dict)
                or result.get("valid") is not True
                or not isinstance(nested, dict)
                or nested.get("valid") is not True
                or not isinstance(rows, list)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or len(rows) != count
                or result.get("passed") != nested.get("passed")
                or any(
                    not isinstance(row, dict) or not isinstance(row.get("passed"), bool)
                    for row in rows
                )
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_capability_results_incomplete",
                        f"Held-out capability results are incomplete for {name}.",
                        baseline.id,
                        run_id,
                    )
                )
            else:
                nested_identity = nested.get("identity")
                if (
                    not isinstance(nested_identity, dict)
                    or nested_identity.get("checkpoint_sha256") != checkpoint
                ):
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "baseline_capability_identity_mismatch",
                            f"Capability result {name} uses another checkpoint.",
                            baseline.id,
                            run_id,
                        )
                    )
        evidence = report.get("local_validation", {})
        validation_row = evidence.get(run_id) if isinstance(evidence, dict) else None
        exp_evidence = (
            validation_row.get("experiment_evidence")
            if isinstance(validation_row, dict)
            else None
        )
        checkpoint_rows = (
            exp_evidence.get("checkpoints") if isinstance(exp_evidence, dict) else None
        )
        if isinstance(checkpoint_rows, list):
            initial = next(
                (
                    row
                    for row in checkpoint_rows
                    if isinstance(row, dict)
                    and row.get("step") == 0
                    and row.get("verified") is True
                ),
                None,
            )
            terminal = next(
                (
                    row
                    for row in checkpoint_rows
                    if isinstance(row, dict)
                    and row.get("step") == endpoint.get("step")
                    and row.get("digest") == checkpoint
                    and row.get("verified") is True
                ),
                None,
            )
            if initial is not None and terminal is not None:
                initial_loss, terminal_loss = (
                    initial.get("validation_loss"),
                    terminal.get("validation_loss"),
                )
                if (
                    not isinstance(initial_loss, (int, float))
                    or isinstance(initial_loss, bool)
                    or not math.isfinite(initial_loss)
                    or not isinstance(terminal_loss, (int, float))
                    or isinstance(terminal_loss, bool)
                    or not math.isfinite(terminal_loss)
                    or terminal_loss >= initial_loss
                ):
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "baseline_validation_gate_failed",
                            "Terminal held-out validation must be finite and below initial validation.",
                            baseline.id,
                            run_id,
                        )
                    )
    _validate_generation_evidence(
        baseline, registry, availability, identity, run_id, diagnostics
    )


def _validate_baseline_artifact_identities(
    registry: LifecycleRegistry,
    evidence_root: Path,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    for baseline in registry.baselines:
        for name in ("configuration", "tokenizer_configuration", "integration_study"):
            identity = getattr(baseline, name)
            key = f"__baseline_{name}__"
            try:
                path = _resolve_under(
                    evidence_root, identity.relative_path, f"baseline {name}"
                )
            except FileNotFoundError:
                diagnostics.append(
                    _diagnostic(
                        "warning" if baseline.status == "candidate" else "error",
                        "baseline_artifact_unavailable",
                        f"Baseline {name} file is unavailable.",
                        baseline.id,
                        name,
                    )
                )
                availability[key] = {
                    "status": "unavailable",
                    "basis": name,
                    "reason": "file not found",
                }
                continue
            except (OSError, ValueError) as error:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_artifact_invalid",
                        str(error),
                        baseline.id,
                        name,
                    )
                )
                availability[key] = {
                    "status": "invalid",
                    "basis": name,
                    "reason": str(error),
                }
                continue
            if (
                not path.is_file()
                or path.stat().st_size != identity.size_bytes
                or sha256_file(path) != identity.sha256
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_artifact_identity_mismatch",
                        f"Baseline {name} identity mismatch.",
                        baseline.id,
                        name,
                    )
                )
                availability[key] = {
                    "status": "invalid",
                    "basis": name,
                    "reason": "artifact byte identity mismatch",
                }
                continue
            availability[key] = {
                "status": "verified",
                "basis": name,
                "path": str(path),
                "reason": None,
            }
            try:
                if name == "configuration":
                    load_config(path)
                elif name == "tokenizer_configuration":
                    load_tokenizer_config(path)
                else:
                    study = plan_study(path)
                    expected_runs = (
                        len(baseline.seeds)
                        if baseline.purpose_classification == "learning_reference"
                        else 1
                    )
                    if (
                        len(study.expanded) != expected_runs
                        or study.comparisons
                        or study.pairs
                    ):
                        raise ValueError(
                            "baseline integration study must plan one unpaired run per seed"
                        )
            except (OSError, ValueError, TypeError) as error:
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "baseline_artifact_invalid",
                        f"Baseline {name}: {error}",
                        baseline.id,
                        name,
                    )
                )
                availability[key] = {
                    "status": "invalid",
                    "basis": name,
                    "path": str(path),
                    "reason": str(error),
                }
                continue


def _check_maturity_evidence(
    registry: LifecycleRegistry,
    availability: dict[str, dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    finding_map = {item.id: item for item in registry.findings}
    evidence_map = {item.id: item for item in registry.evidence}
    for entry in registry.entries:
        for axis_name in ("implementation", "capability", "efficiency", "portability"):
            axis = getattr(entry.maturity, axis_name)
            if axis is None:
                continue
            for finding_id in axis.finding_ids:
                finding = finding_map.get(finding_id)
                if finding is None:
                    continue
                if finding.entry != entry.entry:
                    diagnostics.append(
                        _diagnostic(
                            "error",
                            "maturity_finding_scope_mismatch",
                            f"{axis_name} maturity cites a finding from another entry.",
                            entry.entry,
                            finding_id,
                        )
                    )
            required_measurement_states = {
                "capability": {
                    "mixed",
                    "supported_locally",
                    "replicated",
                    "scale_validated",
                },
                "efficiency": {"end_to_end_measured"},
                "portability": {
                    "adapter_verified",
                    "behavior_verified",
                    "representation_verified",
                },
            }
            if axis.state in required_measurement_states.get(
                axis_name, set()
            ) and not any(
                _measured_finding(finding_map[finding_id], evidence_map, availability)
                for finding_id in axis.finding_ids
                if finding_id in finding_map
            ):
                diagnostics.append(
                    _diagnostic(
                        "error",
                        "maturity_measured_evidence_missing",
                        f"{axis_name} state {axis.state} requires available measured evidence.",
                        entry.entry,
                    )
                )


def validate_lifecycle(
    registry: LifecycleRegistry, *, evidence_root: Path
) -> LifecycleValidation:
    availability, evidence_diagnostics = _verify_evidence(registry, evidence_root)
    diagnostics = evidence_diagnostics
    _validate_baseline_artifact_identities(
        registry, evidence_root, availability, diagnostics
    )
    diagnostics.extend(
        _add_cross_reference_diagnostics(registry, evidence_root, availability)
    )
    diagnostics.extend(
        _check_supersedes_cycles(
            {
                item.supersedes: item.id
                for item in registry.baselines
                if item.supersedes is not None
            }
        )
    )
    _check_maturity_evidence(registry, availability, diagnostics)
    for baseline in registry.baselines:
        _validate_baseline_capture(baseline, registry, availability, diagnostics)
    required_by_accepted = {
        evidence_id
        for baseline in registry.baselines
        if baseline.status in {"known_good", "superseded"}
        for references in baseline.required_evidence.values()
        for evidence_id in references
    }
    required_by_accepted.update(
        evidence_id
        for promotion in registry.promotions
        if promotion.decision == "accepted"
        for evidence_id in promotion.integration_evidence_ids
        + promotion.resource_tradeoff.evidence_ids
    )
    public_availability = {
        evidence_id: state
        for evidence_id, state in availability.items()
        if not evidence_id.startswith("__")
    }
    for evidence_id, state in public_availability.items():
        if (
            state.get("status") != "verified"
            and evidence_id not in required_by_accepted
        ):
            diagnostics.append(
                _diagnostic(
                    "warning",
                    "evidence_unavailable",
                    f"Evidence {evidence_id} is {state.get('status')}: {state.get('reason')}.",
                    evidence_id,
                )
            )
    return {
        "valid": not any(item["severity"] == "error" for item in diagnostics),
        "diagnostics": _sort_diagnostics(diagnostics),
        "availability": public_availability,
    }


def _baseline_scoped_diagnostics(
    registry: LifecycleRegistry,
    baseline_id: str,
    diagnostics: list[dict[str, object]],
) -> tuple[Baseline, list[dict[str, object]]]:
    baseline = next(
        (item for item in registry.baselines if item.id == baseline_id), None
    )
    if baseline is None:
        raise ValueError(f"unknown baseline: {baseline_id}")
    entries = [item for item in registry.entries if item.baseline_id == baseline_id]
    entry_ids = {item.entry for item in entries}
    finding_ids = {finding_id for item in entries for finding_id in item.finding_ids}
    finding_ids.update(
        finding.id for finding in registry.findings if finding.entry in entry_ids
    )
    promotions = [
        item
        for item in registry.promotions
        if item.to_baseline == baseline_id or item.id == baseline.promoted_from
    ]
    finding_ids.update(
        finding_id for promotion in promotions for finding_id in promotion.finding_ids
    )
    evidence_ids = {
        evidence_id
        for references in baseline.required_evidence.values()
        for evidence_id in references
    }
    evidence_ids.update(
        evidence_id
        for finding in registry.findings
        if finding.id in finding_ids
        for evidence_id in finding.evidence_ids
    )
    evidence_ids.update(
        evidence_id
        for promotion in promotions
        for evidence_id in (
            promotion.evidence_ids
            + promotion.integration_evidence_ids
            + promotion.resource_tradeoff.evidence_ids
            + [
                reference
                for row in promotion.regressions
                for reference in row.evidence_ids
            ]
        )
    )
    scope = {
        baseline_id,
        *entry_ids,
        *finding_ids,
        *evidence_ids,
        *(item.id for item in promotions),
    }
    scoped = [
        item
        for item in diagnostics
        if any(
            reference in scope
            for reference in item.get("reference_ids", [])
            if isinstance(reference, str)
        )
    ]
    return baseline, _sort_diagnostics(scoped)


def validate_baseline(
    registry: LifecycleRegistry,
    baseline_id: str,
    *,
    evidence_root: Path,
) -> dict[str, object]:
    """Return candidate-scoped validity alongside unfiltered registry health."""
    global_result = validate_lifecycle(registry, evidence_root=evidence_root)
    baseline, diagnostics = _baseline_scoped_diagnostics(
        registry, baseline_id, global_result["diagnostics"]
    )
    established = baseline.status in {"known_good", "superseded"}
    return {
        "global_registry": {
            "valid": global_result["valid"],
            "diagnostics": global_result["diagnostics"],
        },
        "baseline": {
            "id": baseline.id,
            "status": baseline.status,
            "valid": established
            and not any(item.get("severity") == "error" for item in diagnostics),
            "diagnostics": diagnostics,
        },
        "availability": global_result["availability"],
    }


def _sort_diagnostics(diagnostics: list[dict[str, object]]) -> list[dict[str, object]]:
    ranks = {"error": 0, "warning": 1, "advisory": 2}

    def key(item: dict[str, object]) -> tuple[object, ...]:
        references = item.get("reference_ids")
        return (
            ranks.get(str(item.get("severity")), 3),
            str(item.get("code")),
            tuple(references) if isinstance(references, list) else (),
            str(item.get("message")),
        )

    return sorted(diagnostics, key=key)


def _plain(model: object) -> dict[str, object]:
    return model.model_dump(mode="json")  # type: ignore[attr-defined,no-any-return]


def _diagnostic_references(item: dict[str, object]) -> list[object]:
    references = item.get("reference_ids")
    return references if isinstance(references, list) else []


def research_status(
    registry: LifecycleRegistry, *, evidence_root: Path
) -> dict[str, object]:
    validation = validate_lifecycle(registry, evidence_root=evidence_root)
    annotations = {item.entry: item for item in registry.entries}
    findings = sorted(
        registry.findings,
        key=lambda item: (-date.fromisoformat(item.recorded_at).toordinal(), item.id),
    )
    entry_rows: list[dict[str, object]] = []
    for reference, annotation in annotations.items():
        title = reference
        digest: str | None = None
        try:
            entry, digest = _entry_digest(reference, evidence_root)
            title = entry.title
        except OSError, ValueError, TypeError:
            pass
        entry_rows.append(
            {
                "id": reference,
                "title": title,
                "assessed": True,
                "entry_sha256": digest,
                "annotation": _plain(annotation),
            }
        )
    for entry in list_research():
        if entry.id not in annotations:
            entry_rows.append(
                {
                    "id": entry.id,
                    "title": entry.title,
                    "assessed": False,
                    "annotation": None,
                }
            )
    baseline_rows = []
    for baseline in registry.baselines:
        row = _plain(baseline)
        row["evidence_availability"] = {
            role: {
                evidence_id: validation["availability"].get(
                    evidence_id,
                    {
                        "status": "invalid",
                        "basis": "missing",
                        "reason": "unknown evidence reference",
                    },
                )
                for evidence_id in evidence_ids
            }
            for role, evidence_ids in baseline.required_evidence.items()
        }
        row["superseded_by"] = sorted(
            item.id for item in registry.baselines if item.supersedes == baseline.id
        )
        row["checkpoint_files_available"] = "unknown"
        row["verification_status"] = (
            "not_captured"
            if baseline.status == "candidate"
            else (
                "invalid"
                if any(
                    diagnostic.get("severity") == "error"
                    and baseline.id in _diagnostic_references(diagnostic)
                    for diagnostic in validation["diagnostics"]
                )
                else "verified"
            )
        )
        baseline_rows.append(row)
    statuses = _finding_evidence_status(registry, validation["availability"])
    stage_counts = {stage: 0 for stage in _STAGES}
    for annotation in registry.entries:
        stage_counts[annotation.stage] += 1
    return {
        "diagnostics": validation["diagnostics"],
        "availability": validation["availability"],
        "baselines": sorted(baseline_rows, key=lambda row: str(row["id"])),
        "entries": sorted(entry_rows, key=lambda row: str(row["id"])),
        "funnel": {
            **stage_counts,
            "unassessed": sum(not bool(row["assessed"]) for row in entry_rows),
        },
        "findings": [
            {**_plain(item), "evidence_availability": statuses.get(item.id, [])}
            for item in findings
        ],
        "promotions": [
            _plain(item) for item in sorted(registry.promotions, key=lambda row: row.id)
        ],
    }


def _finding_evidence_status(
    registry: LifecycleRegistry, availability: dict[str, dict[str, object]]
) -> dict[str, list[dict[str, object]]]:
    evidence = {item.id: item for item in registry.evidence}
    result = {}
    for finding in registry.findings:
        result[finding.id] = [
            {
                "id": eid,
                "kind": evidence[eid].kind if eid in evidence else "missing",
                **availability.get(
                    eid,
                    {
                        "status": "invalid",
                        "basis": "missing",
                        "reason": "unknown evidence reference",
                    },
                ),
            }
            for eid in finding.evidence_ids
        ]
    return result


def prior_evidence_warnings(
    registry: LifecycleRegistry,
    *,
    study_sha256: str | None = None,
    protocol_sha256: str | None = None,
) -> list[dict[str, object]]:
    if study_sha256 is not None and protocol_sha256 is not None:
        raise ValueError("provide study_sha256 or protocol_sha256, not both")
    target = study_sha256 if study_sha256 is not None else protocol_sha256
    key = "study_sha256" if study_sha256 is not None else "protocol_sha256"
    if target is None:
        return []
    if not _SHA256.fullmatch(target):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return [
        {
            "severity": "warning",
            "code": "prior_design_evidence",
            "message": "Prior evidence already covers this design under equivalent declared conditions.",
            "reference_ids": [finding.id],
            "finding_id": finding.id,
            "scope": finding.tested_conditions,
            "disposition": finding.disposition,
            "reopen_conditions": finding.reopen_conditions,
        }
        for finding in sorted(
            registry.findings, key=lambda row: (row.recorded_at, row.id)
        )
        if getattr(finding.design_identity, key, None) == target
    ]


def _accepted_scientific_record(payload: object) -> bool:
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "astra-scientific-acceptance-v1"
        or payload.get("accepted") is not True
    ):
        return False
    source_identity = payload.get("source_identity_sha256")
    endpoints = payload.get("endpoints")
    reports = payload.get("card_reports")
    pairs = payload.get("paired_deltas_verified")
    if (
        not isinstance(source_identity, str)
        or not _SHA256.fullmatch(source_identity)
        or not isinstance(endpoints, list)
        or not endpoints
        or not isinstance(reports, list)
        or not reports
        or not isinstance(pairs, list)
        or not pairs
    ):
        return False
    run_ids: set[str] = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            return False
        run_id = endpoint.get("run_id")
        checkpoint = endpoint.get("checkpoint_sha256")
        step = endpoint.get("step")
        targets = endpoint.get("targets")
        files_verified = endpoint.get("files_verified")
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or not isinstance(checkpoint, str)
            or not _SHA256.fullmatch(checkpoint)
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(targets, int)
            or isinstance(targets, bool)
            or targets <= 0
            or not isinstance(files_verified, int)
            or isinstance(files_verified, bool)
            or files_verified <= 0
        ):
            return False
        run_ids.add(run_id)
    if len(run_ids) != len(endpoints):
        return False
    reported_run_ids: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            return False
        run_id = report.get("run_id")
        card = report.get("card")
        path = report.get("path")
        cases = report.get("cases")
        passed = report.get("passed")
        result_digest = report.get("result_digest")
        digest = report.get("sha256")
        if (
            not isinstance(run_id, str)
            or run_id not in run_ids
            or not isinstance(card, str)
            or not card.strip()
            or not isinstance(path, str)
            or not path.strip()
            or not isinstance(cases, int)
            or isinstance(cases, bool)
            or cases <= 0
            or not isinstance(passed, int)
            or isinstance(passed, bool)
            or not 0 <= passed <= cases
            or not isinstance(result_digest, str)
            or not _SHA256.fullmatch(result_digest)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            return False
        reported_run_ids.add(run_id)
    if reported_run_ids != run_ids:
        return False
    for pair in pairs:
        if not isinstance(pair, dict):
            return False
        left = pair.get("left")
        right = pair.get("right")
        if (
            not isinstance(left, str)
            or left not in run_ids
            or not isinstance(right, str)
            or right not in run_ids
            or pair.get("verified") is not True
        ):
            return False
    return True


def _measured_evidence(
    evidence_ids: list[str],
    evidence_map: dict[str, EvidenceReference],
    availability: dict[str, dict[str, object]],
) -> bool:
    for evidence_id in evidence_ids:
        reference = evidence_map.get(evidence_id)
        if (
            reference is None
            or availability.get(evidence_id, {}).get("status") != "verified"
        ):
            continue
        if reference.kind == "report_bundle":
            return True
        if reference.kind != "acceptance":
            continue
        raw, _ = _load_evidence_json(reference, availability)
        checks = raw.get("checks") if isinstance(raw, dict) else None
        if isinstance(checks, dict) and any(
            isinstance(check, dict)
            and check.get("outcome") == "passed"
            and isinstance(check.get("evidence_ids"), list)
            and any(
                isinstance(ref, str)
                and evidence_map.get(ref) is not None
                and evidence_map[ref].kind == "report_bundle"
                and availability.get(ref, {}).get("status") == "verified"
                for ref in check["evidence_ids"]
            )
            for check in checks.values()
        ):
            return True
        if _accepted_scientific_record(raw):
            return True

    return False


def _measured_finding(
    finding: Finding,
    evidence_map: dict[str, EvidenceReference],
    availability: dict[str, dict[str, object]],
) -> bool:
    return _measured_evidence(finding.evidence_ids, evidence_map, availability)


def next_experiments(
    registry: LifecycleRegistry, *, evidence_root: Path
) -> dict[str, object]:
    validation = validate_lifecycle(registry, evidence_root=evidence_root)
    evidence_map = {item.id: item for item in registry.evidence}
    finding_map = {item.id: item for item in registry.findings}
    items: list[dict[str, object]] = []
    for entry in registry.entries:
        proposal = entry.next_test
        if proposal is None:
            items.append(
                {
                    "entry": entry.entry,
                    "status": "not_assessed",
                    "reasons": ["No next test is declared."],
                    "action": None,
                }
            )
            continue
        reasons: list[str] = []
        for blocker_id in proposal.blocker_ids:
            blocker = next(
                (row for row in entry.blockers if row.id == blocker_id), None
            )
            if blocker is not None:
                reasons.append(
                    f"Blocked by {blocker.id}: {blocker.description}; resolution: {blocker.resolution}"
                )
        for finding_id in proposal.prerequisites:
            finding = finding_map.get(finding_id)
            if finding is None:
                reasons.append(f"Prerequisite finding {finding_id} is missing.")
            elif finding.entry != entry.entry:
                reasons.append(
                    f"Prerequisite finding {finding_id} is outside this entry scope."
                )
            elif not _measured_finding(
                finding, evidence_map, validation["availability"]
            ):
                reasons.append(
                    f"Prerequisite finding {finding_id} lacks available non-document measured evidence."
                )
        relevant = [row for row in registry.findings if row.entry == entry.entry]
        if proposal.stage == "replicate" and not any(
            _measured_finding(row, evidence_map, validation["availability"])
            for row in relevant
        ):
            reasons.append(
                "Replication requires an earlier available measured finding."
            )
        if proposal.stage == "scale":
            if not any(
                row.disposition == "replicate"
                and _measured_finding(row, evidence_map, validation["availability"])
                for row in relevant
            ):
                reasons.append(
                    "Scale requires an available report-backed replicate finding."
                )
            capability = entry.maturity.capability
            if capability is None or capability.state != "replicated":
                reasons.append(
                    "Scale requires an explicit capability=replicated declaration."
                )
            elif not any(
                _measured_finding(
                    finding_map[finding_id], evidence_map, validation["availability"]
                )
                for finding_id in capability.finding_ids
                if finding_id in finding_map
            ):
                reasons.append(
                    "Replicated capability declaration lacks available measured evidence."
                )
        if proposal.stage == "promote" and not any(
            row.disposition == "promote"
            and _measured_finding(row, evidence_map, validation["availability"])
            for row in relevant
        ):
            reasons.append(
                "Promotion requires an available report-backed Finding dispositioned promote."
            )
        reopening = [
            row
            for row in relevant
            if row.disposition in {"rejected", "inconclusive", "learning"}
        ]
        if proposal.stage in {"mechanism", "micro", "confirm"} and reopening:
            uncovered = [row.id for row in reopening if row.id not in proposal.reopens]
            if uncovered:
                reasons.append(
                    "Prior negative, inconclusive or learning findings need explicit reopen conditions: "
                    + ", ".join(sorted(uncovered))
                    + "."
                )
            if not proposal.design_change or not proposal.design_change.strip():
                reasons.append(
                    "A design_change must document the applicable reopen condition."
                )
        items.append(
            {
                "entry": entry.entry,
                "baseline_id": entry.baseline_id,
                "stage": proposal.stage,
                "action": proposal.action,
                "prerequisites": proposal.prerequisites,
                "blocker_ids": proposal.blocker_ids,
                "blockers": [
                    _plain(row)
                    for row in entry.blockers
                    if row.id in proposal.blocker_ids
                ],
                "eligible_scales": proposal.eligible_scales,
                "cost_class": proposal.cost_class,
                "reopens": proposal.reopens,
                "reopen_conditions": [
                    {
                        "finding_id": finding_id,
                        "conditions": finding_map[finding_id].reopen_conditions,
                    }
                    for finding_id in proposal.reopens
                    if finding_id in finding_map
                ],
                "design_change": proposal.design_change,
                "status": "blocked" if reasons else "ready",
                "reasons": reasons,
            }
        )
    stage_rank = {name: index for index, name in enumerate(_STAGES)}
    cost_rank = {name: index for index, name in enumerate(_COSTS)}
    items.sort(
        key=lambda row: (
            stage_rank.get(str(row.get("stage")), len(_STAGES)),
            cost_rank.get(str(row.get("cost_class")), len(_COSTS)),
            str(row.get("entry")),
        )
    )
    grouped = {
        stage: [item for item in items if item.get("stage") == stage]
        for stage in _STAGES
    }
    return {
        "diagnostics": validation["diagnostics"],
        "availability": validation["availability"],
        "items": items,
        "stages": grouped,
        "scale_ready": [
            item for item in grouped["scale"] if item.get("status") == "ready"
        ],
        "blocked": [item for item in items if item.get("status") == "blocked"],
        "not_assessed": [
            item for item in items if item.get("status") == "not_assessed"
        ],
        "planning_basis": "declared next-test view; not optimal planning",
    }
