"""Opt-in fixed identity assertions for an already verified native study report."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictStr

from klab.config.models import StrictModel
from klab.documents import parse_document
from klab.recovery.provenance import git_provenance
from klab.training.manifest import source_identity


class IdentityExpectations(StrictModel):
    format: Literal["klab-study-identity-expectations-v1"]
    matrix_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    study_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    config_identity_mode: Literal["current", "legacy_default_fields"]
    config_sha256: list[Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        min_length=1, max_length=1000
    )


def read_expectations(path: Path) -> tuple[IdentityExpectations, str]:
    data = path.read_bytes()
    expected = IdentityExpectations.model_validate(
        parse_document(data.decode("utf-8"), source=path)
    )
    return expected, hashlib.sha256(data).hexdigest()


def accept_identity(
    report: dict, expectations: tuple[IdentityExpectations, str], candidate_commit: str
) -> dict:
    """Refuse drift before report output creation; never run or load a model."""
    expected, expectations_sha256 = expectations
    actual = {
        key: report["inputs"][key]
        for key in ("matrix_sha256", "study_sha256", "config_identity_mode")
    }
    actual["config_sha256"] = [row["config_sha256"] for row in report["runs"]]
    if actual != expected.model_dump(exclude={"format"}):
        raise ValueError("study identity expectations differ from verified report")
    package = source_identity()
    root = Path(__file__).resolve().parents[1]
    inventory = sorted(package["files"])
    provenance = git_provenance(tuple(root / name for name, _digest in inventory))
    if provenance["status"] != "CLEAN_AND_COMMITTED":
        raise ValueError("candidate package must be CLEAN_AND_COMMITTED")
    if [row["sha256"] for row in provenance["declarations"]] != [
        digest for _name, digest in inventory
    ]:
        raise ValueError("candidate package changed during source verification")
    if provenance["source_commit"] != candidate_commit or not candidate_commit:
        raise ValueError("candidate commit does not match verifier source checkout")
    return {
        "verified": True,
        "expectations_sha256": expectations_sha256,
        "expected": expected.model_dump(),
        "candidate_commit": provenance["source_commit"],
        "candidate_package_sha256": package["sha256"],
        "candidate_package_algorithm": package["algorithm"],
        "candidate_package_status": provenance["status"],
        "scope": "Fixed metadata identities only; no payload or scientific qualification. "
        "Package inventory bytes match the candidate commit; this does not attest "
        "dependencies, interpreter, in-memory modifications or installed-runtime equivalence.",
    }
