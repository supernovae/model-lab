"""Strict operational inputs for user-provisioned hosted machines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from klab.config.models import StrictModel

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class HostedTarget(StrictModel):
    """Exactly one explicit existing endpoint; this model never allocates one."""

    colab_session: str | None = None
    ssh: str | None = None
    root: Path | None = None
    colab_config: Path | None = None
    colab_auth: Literal["oauth2", "adc"] = "oauth2"

    @model_validator(mode="after")
    def valid_target(self) -> HostedTarget:
        if (self.colab_session is None) == (self.ssh is None):
            raise ValueError("provide exactly one of colab_session or ssh")
        if self.colab_session is not None and (
            not self.colab_session or len(self.colab_session) > 128
        ):
            raise ValueError("colab_session must contain 1..128 characters")
        if self.ssh is not None and (not self.ssh or len(self.ssh) > 1024):
            raise ValueError("ssh host must contain 1..1024 characters")
        if self.ssh is not None and (
            self.ssh.startswith("-")
            or not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", self.ssh)
        ):
            raise ValueError("ssh must be one non-option host or user@host")
        if self.colab_session is not None and (
            self.colab_session.startswith("-")
            or any(char.isspace() or char == "\0" for char in self.colab_session)
        ):
            raise ValueError("colab_session must be one non-option session name")
        for field in ("root", "colab_config"):
            value = getattr(self, field)
            if value is not None and (not value.is_absolute() or ".." in value.parts):
                raise ValueError(f"{field} must be absolute and non-traversing")
        return self


class SetupRequest(HostedTarget):
    source_bundle: Path
    source_commit: str
    runtime_root: Path
    recipe: Literal["cuda-cu126-v1"] = "cuda-cu126-v1"
    timeout: float = Field(default=1800, gt=0, le=7200)

    @model_validator(mode="after")
    def valid_setup(self) -> SetupRequest:
        if not self.source_bundle.is_absolute():
            raise ValueError("source_bundle must be absolute")
        if not _SHA.fullmatch(self.source_commit):
            raise ValueError("source_commit must be a lowercase full Git object ID")
        if not self.runtime_root.is_absolute() or ".." in self.runtime_root.parts:
            raise ValueError("runtime_root must be absolute and non-traversing")
        if self.root is None:
            raise ValueError("setup requires an absolute root")
        return self
