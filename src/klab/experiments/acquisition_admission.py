"""Explicit bounded HTTP authority, separate from scientific source declarations."""

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from klab.config.models import StrictModel


class AcquisitionAdmission(StrictModel):
    acquisition_version: StrictInt = Field(ge=1, le=1)
    provider: Literal["huggingface"]
    repo_id: StrictStr
    revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    allow_network: StrictBool
    decoder_profile: Literal["scalar-parquet-v1"]
    max_source_body_bytes: StrictInt = Field(gt=0)
    max_metadata_body_bytes: StrictInt = Field(gt=0)
    max_transfers: StrictInt = Field(gt=0)
    max_requests: StrictInt = Field(gt=0)
    max_retries_per_shard: StrictInt = Field(ge=0)
    max_wall_seconds: StrictInt = Field(gt=0)
    max_cache_bytes: StrictInt = Field(gt=0)
    max_cache_files: StrictInt = Field(gt=0)
    max_scanned_rows: StrictInt = Field(gt=0)
    max_verification_bytes: StrictInt = Field(gt=0)
    max_observed_owned_bytes: StrictInt = Field(gt=0)
    max_observed_owned_inodes: StrictInt = Field(gt=0)
    require_hard_disk_bytes: StrictInt | None = Field(default=None, gt=0)
    require_hard_decoder_memory_bytes: StrictInt | None = Field(default=None, gt=0)
    require_expanded_bytes: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def supported_guarantees(self):
        if not self.allow_network:
            raise ValueError("acquisition requires explicit network authority")
        if any(
            value is not None
            for value in (
                self.require_hard_disk_bytes,
                self.require_hard_decoder_memory_bytes,
                self.require_expanded_bytes,
            )
        ):
            raise ValueError(
                "requested hard disk/decoder/expanded-byte guarantee is unavailable"
            )
        return self
