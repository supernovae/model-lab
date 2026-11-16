"""Pretraining preflight: do the data, tokenizer, mixture, prepared arrays,
model config and warmup agree, and are they what the plan says?

`klab preflight CONFIG` reads artifacts the earlier phases already produced and
verified (the release's signed verification receipt, its build report, the
mixture receipt, the tokenizer manifest, the prepared-data cache, the stage
bundle) and writes one sealed report: PASS, WARN or FAIL per check with the
evidence behind it. Nothing here rescans the corpus; every count comes from an
artifact a verifier already authenticated. `klab train` refuses a fresh
pretraining run without a report that is current for the exact config and
artifact set (see `open_gate`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig
from klab.training.manifest import config_sha256, sha256_file, source_identity

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
KIND = "pretraining-preflight"
# A source that loses more than this share of its files (or records) to
# set-aside decisions is surfaced as a WARN with its reasons.
DEFAULT_MAX_SET_ASIDE_SHARE = 0.25
# A warmup peak within this share of the card's memory is a WARN.
PEAK_HEADROOM_SHARE = 0.05
WAIVER_PREFIX = "waive:"


@dataclass
class Check:
    id: str
    status: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or linked file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


class Evidence:
    """The artifacts one preflight reads, each loaded and verified once."""

    def __init__(
        self,
        config: RunConfig,
        *,
        proof_store: Any = None,
        verification_mode: str = "cold",
        stage_bundle: Path | None = None,
        recipe: Path | None = None,
        near_duplicates: Path | None = None,
        near_duplicate_decisions: Path | None = None,
        engram_sizing: Path | None = None,
        max_set_aside_share: float = DEFAULT_MAX_SET_ASIDE_SHARE,
    ) -> None:
        dataset = config.dataset
        if (
            dataset.source != "local_token_mixture"
            or dataset.mixture_declaration_path is None
            or dataset.mixture_output_path is None
        ):
            raise ValueError(
                "preflight checks a pretraining config: dataset.source must be "
                "local_token_mixture with its mixture declaration and output"
            )
        self.config = config
        self.proof_store = proof_store
        self.verification_mode = verification_mode
        self.stage_bundle = stage_bundle
        self.recipe = recipe
        self.near_duplicates = near_duplicates
        self.near_duplicate_decisions = near_duplicate_decisions
        self.engram_sizing = engram_sizing
        self.max_set_aside_share = max_set_aside_share
        self.subject: dict[str, Any] = {
            "config_sha256": config_sha256(config.model_dump(mode="json")),
            "lab_source_identity_sha256": source_identity()["sha256"],
            "release_id": None,
            "mixture_receipt_sha256": None,
            "tokenizer_sha256": None,
            "prepared_manifest_sha256": None,
            "stage_bundle_sha256": None,
        }

    @cached_property
    def spec(self) -> Any:
        from klab.corpus.mixture import MixtureDeclaration

        path = self.config.dataset.mixture_declaration_path
        assert path is not None
        return MixtureDeclaration.model_validate(yaml.safe_load(path.read_bytes()))

    @property
    def release_path(self) -> Path:
        return self.spec.release_path.resolve()

    @cached_property
    def manifest(self) -> dict[str, Any]:
        from klab.corpus.release import verify_release

        manifest = verify_release(
            self.release_path,
            proof_store=self.proof_store,
            verification_mode=self.verification_mode,
        )
        self.subject["release_id"] = manifest["release_id"]
        return manifest

    @cached_property
    def report(self) -> dict[str, Any]:
        self.manifest  # noqa: B018 - the build report is read only after verification
        return _load_json(self.release_path / "report.json")

    @cached_property
    def receipt(self) -> dict[str, Any]:
        path = self.config.dataset.mixture_output_path
        assert path is not None
        receipt = _load_json(path / "receipt.json")
        self.subject["mixture_receipt_sha256"] = sha256_file(path / "receipt.json")
        return receipt

    @cached_property
    def tokenizer_manifest(self) -> dict[str, Any]:
        path = self.config.tokenizer.path
        manifest = _load_json(path.with_name("tokenizer_manifest.json"))
        self.subject["tokenizer_sha256"] = sha256_file(path)
        return manifest

    @cached_property
    def prepared(self) -> Any:
        from klab.data.packing import prepare_data
        from klab.data.tokenizer import load_tokenizer

        data = prepare_data(
            self.config,
            load_tokenizer(self.config.tokenizer.path),
            require_prepared=True,
            proof_store=self.proof_store,
            verification_mode=self.verification_mode,
        )
        self.subject["prepared_manifest_sha256"] = data.manifest["manifest_sha256"]
        return data

    @cached_property
    def staged(self) -> dict[str, Any]:
        from klab.staging import verify_stage_bundle

        if self.stage_bundle is None:
            raise ValueError("no stage bundle given (--stage-bundle)")
        staged = verify_stage_bundle(
            self.stage_bundle,
            self.config,
            proof_store=self.proof_store,
            verification_mode=self.verification_mode,
        )
        self.subject["stage_bundle_sha256"] = staged["sha256"]
        return staged


def check_release(ev: Evidence) -> Check:
    manifest = ev.manifest
    policy = manifest["build_identity"].get("release", {}).get("policy")
    evidence = {
        "release_id": manifest["release_id"],
        "path": str(ev.release_path),
        "manifest_sha256": sha256_file(ev.release_path / "manifest.json"),
        "verification_mode": ev.verification_mode,
        "build_policy": policy,
    }
    if not policy:
        return Check(
            "release",
            FAIL,
            "the release verified but records no build policy; rebuild it",
            evidence,
        )
    others = [r.model_dump(mode="json") for r in ev.spec.additional_releases]
    if others:
        evidence["additional_releases_not_checked"] = others
        return Check(
            "release",
            WARN,
            "release verified; additional releases in the mixture are not checked here",
            evidence,
        )
    return Check(
        "release", PASS, "release verified and its build policy recorded", evidence
    )


def check_sources(ev: Evidence) -> Check:
    ledger = ev.report.get("source_ledger")
    if ledger is None:
        raise ValueError("the build report has no source_ledger; rebuild the release")
    rows = {row["source_id"]: row for row in ledger}
    sources = json.loads((ev.release_path / "sources.json").read_text("utf-8"))
    declared = [row["id"] for row in sources]
    missing = sorted(set(declared) - rows.keys())
    empty = sorted(
        s for s in declared if s in rows and not rows[s]["retained_documents"]
    )
    limit = ev.max_set_aside_share
    over = []
    for source in declared:
        row = rows.get(source)
        if row is None or not row["retained_documents"]:
            continue
        files = row["file_set_aside_share"] or 0.0
        seen_rows = row["documents_built"] + row["rows_set_aside"]
        records = row["rows_set_aside"] / seen_rows if seen_rows else 0.0
        if max(files, records) > limit:
            over.append(
                {
                    "source_id": source,
                    "file_set_aside_share": files,
                    "row_set_aside_share": records,
                    "reasons": row["set_aside_reasons"],
                }
            )
    evidence = {
        "declared_sources": len(declared),
        "max_set_aside_share": limit,
        "retained_documents": sum(r["retained_documents"] for r in rows.values()),
        "without_retained_documents": empty,
        "missing_from_ledger": missing,
        "over_threshold": over,
    }
    if missing or empty:
        names = ", ".join(empty + missing)
        return Check(
            "sources",
            FAIL,
            f"{len(empty) + len(missing)} of {len(declared)} declared sources "
            f"retained nothing: {names}",
            evidence,
        )
    if over:
        return Check(
            "sources",
            WARN,
            f"{len(over)} source(s) lost more than {limit:.0%} to set-aside "
            f"decisions: {', '.join(item['source_id'] for item in over)}",
            evidence,
        )
    return Check(
        "sources",
        PASS,
        f"all {len(declared)} declared sources retained documents; none lost "
        f"more than {limit:.0%}",
        evidence,
    )


def check_quotas(ev: Evidence) -> Check:
    from klab.corpus.mixture import stratum_exposures

    receipt, spec = ev.receipt, ev.spec
    strata: dict[str, Any] = {}
    cannot_fill, short = [], []
    if receipt["requested_target_quotas"] != spec.target_quotas:
        raise ValueError(
            "the mixture declaration's quotas differ from the receipt's; "
            "materialize the mixture again"
        )
    for stratum, quota in sorted(spec.target_quotas.items()):
        available = receipt["available_unique_positions"][stratum]
        exposures = stratum_exposures(spec, stratum)
        actual = receipt["actual_target_tokens"][stratum]
        strata[stratum] = {
            "quota": quota,
            "available_unique_tokens": available,
            "max_exposures": exposures,
            "scheduled_tokens": actual,
            "shortfall": receipt["quota_shortfalls"][stratum],
            "repeats_needed": quota > available,
        }
        if available * exposures < quota:
            cannot_fill.append(stratum)
        elif actual < quota:
            short.append(stratum)
    if cannot_fill:
        return Check(
            "quotas",
            FAIL,
            f"stratum(s) cannot fill their quota even with repeats: {', '.join(cannot_fill)}",
            {"strata": strata},
        )
    if short:
        return Check(
            "quotas",
            WARN,
            f"the mixture receipt shows under-filled stratum(s): {', '.join(short)}",
            {"strata": strata},
        )
    return Check(
        "quotas",
        PASS,
        f"all {len(strata)} strata fill their token quota",
        {"strata": strata},
    )


def check_validation(ev: Evidence) -> Check:
    scale = ev.report["source_scale"]
    strata: dict[str, dict[str, int]] = {
        stratum: {"documents": 0, "utf8_bytes": 0, "whitespace_words": 0}
        for stratum in sorted(set(ev.spec.source_strata.values()))
    }
    for source, stratum in ev.spec.source_strata.items():
        held = scale.get(source, {}).get("validation", {})
        for key in strata[stratum]:
            strata[stratum][key] += held.get(key, 0)
    total = sum(item["documents"] for item in strata.values())
    empty = [name for name, item in strata.items() if not item["documents"]]
    evidence = {
        "strata": strata,
        "total_documents": total,
        "config_validation_max_documents": ev.config.dataset.validation_max_documents,
        "config_validation_max_tokens": ev.config.dataset.validation_max_tokens,
    }
    if not total:
        return Check("validation", FAIL, "the validation split is empty", evidence)
    if empty:
        return Check(
            "validation",
            WARN,
            f"no validation documents for stratum(s): {', '.join(empty)}",
            evidence,
        )
    return Check(
        "validation",
        PASS,
        f"validation is non-empty in all {len(strata)} strata ({total} documents)",
        evidence,
    )


def check_near_duplicates(ev: Evidence) -> Check:
    from klab.corpus.near_duplicates import check_decisions

    if ev.near_duplicates is None or ev.near_duplicate_decisions is None:
        raise ValueError(
            "give --near-duplicates (the screen's report) and "
            "--near-duplicate-decisions (the reviewer's record)"
        )
    report = _load_json(ev.near_duplicates)
    report_sha = sha256_file(ev.near_duplicates)
    problems: list[str] = []
    release = ev.report
    if report.get("release_id") != ev.manifest["release_id"]:
        problems.append("the screen ran on a different release")
    if report.get("document_count") != release["document_count"]:
        problems.append(
            f"the screen covered {report.get('document_count')} documents; "
            f"the release has {release['document_count']}"
        )
    covered: dict[str, int] = {}
    for row in report.get("coverage", []):
        covered[row["split"]] = covered.get(row["split"], 0) + row["documents"]
    for split in ("validation", "test"):
        if covered.get(split, 0) < release["split_counts"].get(split, 0):
            problems.append(f"the screen does not cover every {split} document")
    if not covered.get("validation"):
        problems.append("the screen covers no validation document")
    if report.get("candidates_truncated") is not False:
        problems.append(
            "the candidate list is truncated; rerun with a larger --max-candidates"
        )
    decisions = yaml.safe_load(ev.near_duplicate_decisions.read_bytes())
    result = check_decisions(report, report_sha, decisions)
    problems += result.pop("problems")
    if result["undecided_count"]:
        problems.append(
            f"{result['undecided_count']} reported pair(s) have no decision"
        )
    evidence = {
        "report_sha256": report_sha,
        "decisions_sha256": sha256_file(ev.near_duplicate_decisions),
        "documents_screened": report.get("document_count"),
        "candidate_count": report.get("candidate_count"),
        "identical_text_count": report.get("identical_text_count"),
        # Pairs similar only through boilerplate need no decision, but are shown.
        "template_aware": report.get("parameters", {}).get("template_aware", False),
        "template_only_count": report.get("template_only_count", 0),
        "template_only_documents": report.get("template_only_document_count", 0),
        **result,
        "problems": problems,
    }
    if problems:
        return Check("near_duplicates", FAIL, "; ".join(problems[:3]), evidence)
    templated = (
        f"; {evidence['template_only_count']} template-only pair(s) on "
        f"{evidence['template_only_documents']} held-out document(s) counted, "
        "not decided"
        if evidence["template_only_count"]
        else ""
    )
    return Check(
        "near_duplicates",
        PASS,
        f"every document screened; all {result['reported_pairs']} reported pairs "
        f"decided{templated}",
        evidence,
    )


def check_tokenizer(ev: Evidence) -> Check:
    path = ev.config.tokenizer.path
    manifest = ev.tokenizer_manifest
    digest = ev.subject["tokenizer_sha256"]
    audit = manifest.get("sampling_audit")
    evidence = {
        "path": str(path),
        "sha256": digest,
        "vocab_size": manifest.get("vocab_size"),
        "mixture_receipt_tokenizer_sha256": ev.receipt["tokenizer_sha256"],
    }
    problems = []
    if manifest.get("sha256") != digest:
        problems.append("the tokenizer file differs from its manifest")
    if manifest.get("vocab_size") != ev.config.model.vocab_size:
        problems.append("the tokenizer vocabulary differs from model.vocab_size")
    if ev.receipt["tokenizer_sha256"] != manifest.get("sha256"):
        problems.append("the mixture was built with a different tokenizer")
    if not audit:
        problems.append("the tokenizer has no recorded audit")
    else:
        groups = audit["strata"].values()
        evidence["audit"] = {
            "complete": audit["complete"],
            "roundtrip_failures": sum(g["roundtrip_failures"] for g in groups),
            "unknown_tokens": sum(g["unknown_tokens"] for g in groups),
            "selected_bytes": audit["selected_bytes"],
        }
        if evidence["audit"]["roundtrip_failures"]:
            problems.append("the audit found documents that do not round-trip")
    if problems:
        return Check("tokenizer", FAIL, "; ".join(problems), evidence)
    notes = []
    if audit["complete"] is not True:
        notes.append("audit did not reach its document target in every stratum")
    if evidence["audit"]["unknown_tokens"]:
        notes.append(
            f"{evidence['audit']['unknown_tokens']} unknown tokens in the audit"
        )
    if notes:
        return Check("tokenizer", WARN, "; ".join(notes), evidence)
    return Check(
        "tokenizer",
        PASS,
        "tokenizer trained, audited and bound to the mixture",
        evidence,
    )


def check_mixture(ev: Evidence) -> Check:
    from klab.corpus.mixture import verify_mixture_dataset

    receipt = verify_mixture_dataset(
        ev.config.dataset, ev.config.tokenizer.path, ev.config.model.vocab_size
    )
    evidence = {
        "receipt_sha256": ev.subject["mixture_receipt_sha256"]
        or sha256_file(ev.config.dataset.mixture_output_path / "receipt.json"),
        "release_id": receipt["release_id"],
        "records": receipt["records"],
        "scheduled_tokens": sum(receipt["actual_target_tokens"].values()),
    }
    ev.receipt  # noqa: B018 - records the receipt digest in the subject
    if receipt["release_id"] != ev.manifest["release_id"]:
        return Check(
            "mixture", FAIL, "the mixture was built from another release", evidence
        )
    return Check(
        "mixture",
        PASS,
        "mixture materialized and verified against the release and tokenizer",
        evidence,
    )


def check_prepared(ev: Evidence) -> Check:
    data = ev.prepared
    manifest = data.manifest
    evidence = {
        "root": str(data.root),
        "manifest_sha256": manifest["manifest_sha256"],
        "train_tokens": len(data.train),
        "validation_tokens": len(data.validation),
    }
    if manifest.get("token_mixture") != ev.receipt:
        return Check(
            "prepared",
            FAIL,
            "the prepared data was built from a different mixture receipt",
            evidence,
        )
    return Check(
        "prepared",
        PASS,
        "prepared data present and bound to this mixture, tokenizer and config",
        evidence,
    )


def check_advisories(ev: Evidence) -> Check:
    from klab.model.inspection import memory_advisories

    notes = {note["code"]: note for note in memory_advisories(ev.config)}
    sizing = _engram_sizing(ev, notes)
    for code in sizing["resolves"]:
        notes.pop(code, None)
    evidence = {
        "advisories": [
            {"code": code, "message": note["message"]} for code, note in notes.items()
        ],
        "engram_sizing": sizing["evidence"],
    }
    if sizing["problems"]:
        return Check("advisories", WARN, "; ".join(sizing["problems"]), evidence)
    if notes:
        return Check(
            "advisories",
            WARN,
            "inspect advisories: " + ", ".join(sorted(notes)),
            evidence,
        )
    return Check("advisories", PASS, "no inspect advisories outstanding", evidence)


def _engram_sizing(ev: Evidence, notes: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"resolves": [], "problems": [], "evidence": None}
    if "memory_tables_never_measured" not in notes:
        return result
    if ev.engram_sizing is None:
        return result
    sizing = _load_json(ev.engram_sizing)
    body = {key: value for key, value in sizing.items() if key != "sha256"}
    model = ev.config.model
    problems = result["problems"]
    if sizing.get("sha256") != canonical_digest(body):
        problems.append("the engram-sizing result is not sealed or was edited")
    if (
        sizing["source"]["prepared_manifest_sha256"]
        != ev.prepared.manifest["manifest_sha256"]
    ):
        problems.append(
            "the engram-sizing result measured other tokens than this prepared data"
        )
    if sizing["setup"]["configured_sizes"] != list(model.memory_table_sizes):
        problems.append(
            "the engram-sizing result measured other table sizes than this config"
        )
    limit = sizing["criterion"]["max_clobbered_all_heads"]
    worst = max(
        order["configured"]["clobbered_all_heads"]
        for site in sizing["sites"]
        for order in site["orders"]
        if "configured" in order
    )
    result["evidence"] = {
        "sha256": sizing.get("sha256"),
        "sample_tokens": sizing["sample_tokens"],
        "worst_clobbered_all_heads": worst,
        "max_clobbered_all_heads": limit,
    }
    if not problems:
        result["resolves"].append("memory_tables_never_measured")
        if worst > limit:
            problems.append(
                f"engram tables clobber {worst:.0%} of occurrences on these tokens "
                f"(criterion {limit:.0%})"
            )
    return result


def check_warmup(ev: Evidence) -> Check:
    staged = ev.staged
    if staged.get("through") != "warmup":
        return Check(
            "warmup",
            FAIL,
            f"the stage bundle stops at {staged.get('through')!r}; stage with --through warmup",
            {"through": staged.get("through")},
        )
    pilot = next(
        item["report"]
        for item in staged["pilot_reports"]
        if item["purpose"] == "warmup"
    )
    updates = pilot["update_observations"]
    steady = updates[2:] if len(updates) > 4 else updates
    targets_per_second = sum(u["targets"] for u in steady) / sum(
        u["update_seconds"] for u in steady
    )
    peak = pilot.get("observed_peak_bytes")
    total = pilot["runtime"].get("device_total_bytes")
    devices = 1 + len(ev.config.runtime.secondary_device_indices)
    assets = ev.stage_bundle / "assets"  # type: ignore[operator]
    staged_data = _load_json(assets / "data" / "manifest.json")["manifest_sha256"]
    evidence = {
        "stage_bundle_sha256": staged["sha256"],
        "warmup_report_sha256": pilot["sha256"],
        "warmup_updates": len(updates),
        "targets_per_second": targets_per_second,
        "peak_bytes": peak,
        "peak_method": pilot.get("peak_method"),
        "device_total_bytes": total,
        "device": pilot["runtime"].get("device_name"),
        "devices": devices,
        "staged_prepared_manifest_sha256": staged_data,
    }
    if staged_data != ev.prepared.manifest["manifest_sha256"]:
        return Check(
            "warmup",
            FAIL,
            "the stage bundle holds different prepared data than this run will use",
            evidence,
        )
    if peak is None or total is None:
        return Check(
            "warmup", FAIL, "the warmup recorded no device peak or card size", evidence
        )
    if peak > total:
        return Check(
            "warmup",
            FAIL,
            f"warmup peak {peak / 2**30:.1f} GiB exceeds the {total / 2**30:.1f} GiB card",
            evidence,
        )
    if peak > total * (1 - PEAK_HEADROOM_SHARE):
        return Check(
            "warmup",
            WARN,
            f"warmup peak {peak / 2**30:.1f} GiB leaves under {PEAK_HEADROOM_SHARE:.0%} "
            f"of the {total / 2**30:.1f} GiB card free",
            evidence,
        )
    return Check(
        "warmup",
        PASS,
        f"warmup peak {peak / 2**30:.1f} of {total / 2**30:.1f} GiB, "
        f"{targets_per_second:,.0f} targets/s, {devices} device(s)",
        evidence,
    )


def check_revisions(ev: Evidence) -> Check:
    from klab.lab_mode import git_state

    lab = git_state(Path(__file__).parent, upstream="origin/main")
    recipe = git_state(ev.recipe, upstream="origin/main") if ev.recipe else None
    evidence = {
        "lab": {
            **lab,
            "source_identity_sha256": ev.subject["lab_source_identity_sha256"],
        },
        "recipe": recipe,
    }
    notes = []
    if lab["git_commit"] is None:
        notes.append("Lab is not running from a Git checkout")
    elif lab["git_dirty"]:
        notes.append("Lab's working tree has uncommitted changes")
    elif lab["in_upstream"] is False:
        notes.append("Lab's commit is not in origin/main")
    if ev.recipe is None:
        notes.append("no --recipe given; the recipe revision is not recorded")
    elif recipe["git_commit"] is None:
        notes.append("the recipe directory is not in a Git checkout")
    elif recipe["git_dirty"]:
        notes.append("the recipe's working tree has uncommitted changes")
    elif recipe["in_upstream"] is False:
        notes.append("the recipe's commit is not in origin/main")
    if notes:
        return Check("revisions", WARN, "; ".join(notes), evidence)
    return Check("revisions", PASS, "Lab and recipe are committed and merged", evidence)


CHECKS: tuple[Callable[[Evidence], Check], ...] = (
    check_release,
    check_sources,
    check_quotas,
    check_validation,
    check_near_duplicates,
    check_tokenizer,
    check_mixture,
    check_prepared,
    check_advisories,
    check_warmup,
    check_revisions,
)
_EXPECTED = (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration)


def run_preflight(ev: Evidence) -> dict[str, Any]:
    """Run every check; a check that cannot run is a FAIL, never a skip."""
    checks = []
    for function in CHECKS:
        name = function.__name__.removeprefix("check_")
        started = time.monotonic()
        try:
            check = function(ev)
        except _EXPECTED as error:
            check = Check(
                name, FAIL, f"not established: {type(error).__name__}: {error}"
            )
        check.evidence["seconds"] = round(time.monotonic() - started, 3)
        checks.append(check)
    statuses = {check.status for check in checks}
    body = {
        "format_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": FAIL if FAIL in statuses else WARN if WARN in statuses else PASS,
        "options": {"max_set_aside_share": ev.max_set_aside_share},
        "subject": ev.subject,
        "checks": [check.__dict__ for check in checks],
    }
    return {**body, "sha256": canonical_digest(body)}


def write_report(report: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"preflight-{report['sha256'][:16]}.json"
    path.write_bytes(canonical_json(report) + b"\n")
    return path


def render(report: dict[str, Any]) -> str:
    lines = [f"Preflight {report['status']}  ({report['sha256'][:16]})"]
    for check in report["checks"]:
        lines.append(f"  {check['status']:<4}  {check['id']:<15} {check['summary']}")
    return "\n".join(lines)


def read_report(path: Path) -> dict[str, Any]:
    report = _load_json(Path(path))
    body = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("kind") != KIND or report.get("sha256") != canonical_digest(body):
        raise ValueError(f"not a sealed preflight report: {path}")
    return report


@dataclass(frozen=True)
class Gate:
    """What `klab train` recorded about its preflight, and what it still binds."""

    record: dict[str, Any]
    subject: dict[str, Any] | None = None

    def bind(
        self, *, stage_bundle_sha256: str | None, prepared_manifest_sha256: str
    ) -> None:
        """Check the artifacts only the trainer holds against the report."""
        if self.subject is None:
            return
        if self.subject["stage_bundle_sha256"] != stage_bundle_sha256:
            raise ValueError(
                "preflight was run for a different stage bundle; run it again "
                "(or train with the --stage-bundle it checked)"
            )
        if self.subject["prepared_manifest_sha256"] != prepared_manifest_sha256:
            raise ValueError(
                "preflight was run for different prepared data; run it again"
            )


def requires_preflight(config: RunConfig, *, continuation: bool) -> bool:
    """A fresh run on a token mixture is a pretraining run."""
    return config.dataset.source == "local_token_mixture" and not continuation


def open_gate(value: str | None, config: RunConfig) -> Gate:
    """Admit a pretraining run: a current report, or an explicit recorded waiver.

    `value` is a report path, or `waive:REASON`. The report must be sealed, not
    FAIL, and name exactly this config, this Lab source, this mixture receipt
    and this tokenizer. The stage bundle and prepared data are compared by
    `Gate.bind` once the trainer has opened them.
    """
    if not value:
        raise ValueError(
            "pretraining needs a preflight: run `klab preflight CONFIG`, then "
            "`klab train --preflight REPORT` (or --preflight waive:REASON)"
        )
    if value.startswith(WAIVER_PREFIX):
        reason = value[len(WAIVER_PREFIX) :].strip()
        if not reason:
            raise ValueError("a preflight waiver needs a reason: waive:REASON")
        return Gate({"waived": reason})
    report = read_report(Path(value))
    if report["status"] == FAIL:
        failed = [c["id"] for c in report["checks"] if c["status"] == FAIL]
        raise ValueError(f"preflight failed: {', '.join(failed)}")
    subject = report["subject"]
    dataset = config.dataset
    assert dataset.mixture_output_path is not None
    current = {
        "config_sha256": config_sha256(config.model_dump(mode="json")),
        "lab_source_identity_sha256": source_identity()["sha256"],
        "mixture_receipt_sha256": sha256_file(
            dataset.mixture_output_path / "receipt.json"
        ),
        "tokenizer_sha256": sha256_file(config.tokenizer.path),
    }
    stale = [name for name, digest in current.items() if subject[name] != digest]
    if stale:
        raise ValueError(
            f"preflight is not current ({', '.join(stale)} changed); run it again"
        )
    return Gate(
        {"report_sha256": report["sha256"], "status": report["status"]}, subject
    )
