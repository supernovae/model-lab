"""Whether a declared mixture can be supplied by the documents a release keeps.

The mixture template asks each stratum for a token quota and allows each unique
token `max_exposures` times. `klab corpus materialize-mixture` checks that with
real token counts, but those need the tokenizer, which needs the release, and
the quotas are part of the release's identity. So the same arithmetic runs
twice earlier: at freeze, from kept TRAIN bytes and a conservative
`release.policy.mixture.bytes_per_token`, and at `klab corpus describe
--tokenizer`, from measured tokens. Both use `shortfall_message`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def shortfall_message(
    stratum: str, available: int, quota: int, exposures: int
) -> str | None:
    """The refusal for a stratum whose unique supply cannot reach its quota."""
    if available * exposures >= quota:
        return None
    return (
        f"required quota impossible for {stratum}: {available} unique, "
        f"{quota} required, max_exposures={exposures}, "
        f"shortfall={quota - available * exposures}"
    )


def require_supply(
    available: Mapping[str, int],
    quotas: Mapping[str, int],
    exposures: Mapping[str, int],
) -> None:
    """Raise one error naming every stratum that cannot supply its quota."""
    messages = [
        message
        for stratum, quota in sorted(quotas.items())
        if (
            message := shortfall_message(
                stratum, available.get(stratum, 0), quota, exposures[stratum]
            )
        )
    ]
    if messages:
        raise ValueError("; ".join(messages))


def template_requirements(
    template: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """A mixture template's quotas and per-stratum exposures.

    The template still holds `${...}` placeholders when the tokenizer is not yet
    fitted, so only these fields are read, not the whole declaration.
    """
    raw = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    quotas = {
        str(stratum): int(quota) for stratum, quota in raw["target_quotas"].items()
    }
    overrides = raw.get("strata") or {}
    return quotas, {
        stratum: (overrides.get(stratum) or {}).get("max_exposures")
        or raw["max_exposures"]
        for stratum in quotas
    }


def estimate_supply(
    policy: Any, shares: Mapping[str, float], train_bytes: Mapping[str, int]
) -> dict[str, Any]:
    """The tokenizer-free estimate recorded in `report.json` as `mixture_supply`."""
    strata = {}
    for stratum, share in sorted(shares.items()):
        unique = int(train_bytes.get(stratum, 0) / policy.bytes_per_token)
        strata[stratum] = {
            "requested_tokens": round(share * policy.total_tokens),
            "train_utf8_bytes": train_bytes.get(stratum, 0),
            "estimated_unique_tokens": unique,
            "supply_tokens": unique * policy.max_exposures,
        }
    return {
        "total_tokens": policy.total_tokens,
        "max_exposures": policy.max_exposures,
        "bytes_per_token": policy.bytes_per_token,
        "strata": strata,
        "feasible": all(
            row["supply_tokens"] >= row["requested_tokens"] for row in strata.values()
        ),
    }


def require_estimated_supply(report: Mapping[str, Any]) -> None:
    """Refuse to freeze a build whose declared mixture its documents cannot fill."""
    estimate = report["mixture_supply"]
    if estimate["feasible"]:
        return
    problems = [
        f"{stratum}: kept TRAIN documents hold {row['train_utf8_bytes']} bytes, "
        f"about {row['estimated_unique_tokens']} tokens at "
        f"{estimate['bytes_per_token']} bytes per token, so "
        f"{row['supply_tokens']} with max_exposures={estimate['max_exposures']}, "
        f"but {row['requested_tokens']} are requested"
        for stratum, row in estimate["strata"].items()
        if row["supply_tokens"] < row["requested_tokens"]
    ]
    raise ValueError(
        "declared mixture cannot be supplied (release.policy.mixture, "
        f"total_tokens={estimate['total_tokens']}): "
        + "; ".join(problems)
        + ". Lower the quota and weight together, give the difference to a "
        "larger stratum, raise max_exposures, or add sources."
    )
