"""Resolve Hugging Face credentials at request time, never in source identities."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path

from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import get_token
from huggingface_hub.errors import HfHubHTTPError


class HuggingFaceCredentialError(ValueError):
    """A configured credential cannot be read or is syntactically invalid."""


class HuggingFaceAccessError(RuntimeError):
    """The Hub rejected a request or the repository is inaccessible."""


HUB_ACCESS_ERRORS = (HfHubHTTPError, DatasetNotFoundError)

_token_file: ContextVar[Path | None] = ContextVar("hf_token_file", default=None)


def set_token_file(path: Path | None) -> None:
    """Set the CLI credential source for the current execution context."""
    _token_file.set(path)


def _checked_token(value: str, source: str) -> str:
    token = value.strip()
    if not token or any(
        character.isspace() or not character.isprintable() for character in token
    ):
        raise HuggingFaceCredentialError(
            f"{source} must contain one nonblank Hugging Face token"
        )
    return token


def hub_auth_kwargs() -> dict[str, str]:
    """Resolve credentials only when an actual Hub request is made."""
    path = _token_file.get()
    if path is not None:
        try:
            token = path.read_text(encoding="utf-8")
        except OSError, UnicodeError:
            raise HuggingFaceCredentialError(
                "Hugging Face token file is unreadable"
            ) from None
        return {"token": _checked_token(token, "Hugging Face token file")}
    if "HF_TOKEN" in os.environ:
        return {"token": _checked_token(os.environ["HF_TOKEN"], "HF_TOKEN")}
    token = get_token()
    if token:
        return {"token": _checked_token(token, "Stored Hugging Face token")}
    return {}


def raise_for_hub_auth(error: Exception, *, credential_supplied: bool) -> None:
    """Replace denied-access errors without exposing request headers or tokens."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    name = type(error).__name__
    if status not in (401, 403) and name not in (
        "GatedRepoError",
        "RepositoryNotFoundError",
        "DatasetNotFoundError",
    ):
        raise error
    if credential_supplied:
        raise HuggingFaceAccessError(
            "Hugging Face access denied (401/403 or private/gated resource). "
            "Check the token's validity, permissions, and dataset access; "
            "use --hf-token-file, HF_TOKEN, or a stored Hugging Face login."
        ) from None
    raise HuggingFaceAccessError(
        "Hugging Face access denied (401/403 or private/gated resource). "
        "Request access if gated, then supply --hf-token-file, HF_TOKEN, "
        "or a stored Hugging Face login."
    ) from None
