"""Explicit Git or exact-content admission for native portable contexts."""

import hashlib
from pathlib import Path

from pydantic import Field, StrictStr, model_validator

from klab.config.models import StrictModel
from klab.experiments.recipe import _metadata_bytes
from klab.recovery.provenance import git_provenance


class RecipeAdmission(StrictModel):
    # Omitted new fields preserve existing Git admission receipt identities.
    reviewed_commit: StrictStr | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$", exclude_if=lambda value: value is None
    )
    reviewed_descriptor_sha256: StrictStr | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def one_source_identity(self):
        if (self.reviewed_commit is None) == (self.reviewed_descriptor_sha256 is None):
            raise ValueError(
                "select exactly one reviewed_commit or reviewed_descriptor_sha256"
            )
        return self

    def provenance(self, source: Path, context, *, operation: str) -> dict:
        paths = (source, *(context.root / name for name in context.documents))
        if self.reviewed_commit is not None:
            provenance = git_provenance(paths)
            if (
                provenance["status"] != "CLEAN_AND_COMMITTED"
                or provenance["source_commit"] != self.reviewed_commit
            ):
                raise ValueError(
                    f"{operation} requires unchanged clean declarations at the selected reviewed commit"
                )
            return provenance
        # Recipe/binding fingerprints were checked by the native context before
        # this call. The additional digest binds exact descriptor bytes, including
        # formatting; no commit, author review or compute authority is implied.
        raw = _metadata_bytes(source)
        if hashlib.sha256(raw).hexdigest() != self.reviewed_descriptor_sha256:
            raise ValueError("CONTENT_CONTEXT_CHANGED: descriptor bytes differ")
        snapshots = {source.name: raw, **dict(context.documents)}
        declarations = []
        for name, data in sorted(snapshots.items()):
            if _metadata_bytes(context.root / name) != data:
                raise ValueError("CONTENT_CONTEXT_CHANGED: declaration bytes differ")
            declarations.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "head_sha256": None,
                    "status": "CONTENT_BOUND",
                }
            )
        return {
            "status": "CONTENT_BOUND",
            "source_commit": None,
            "declarations": declarations,
        }
