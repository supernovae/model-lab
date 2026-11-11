"""Tiny CPU smoke inputs for lab-loop tests (`try`, probes, reports)."""

from pathlib import Path

import yaml

from klab.config.loading import load_tokenizer_config
from klab.data.tokenizer import train_tokenizer

ROOT = Path(__file__).resolve().parents[1]

_CACHED_SMOKE_TOKENIZER: tuple[bytes, bytes] | None = None


def write_smoke_inputs(
    root: Path, *, max_steps: int = 4, train_tokenizer_now: bool = True
) -> Path:
    """Write a smoke tokenizer and baseline config under ``root``; return the config.

    The trained tokenizer is cached per process, so later calls only copy bytes.
    """
    global _CACHED_SMOKE_TOKENIZER
    tokenizer = yaml.safe_load((ROOT / "configs/tokenizer_smoke.yaml").read_text())
    tok_dir = root / "tokenizer"
    tokenizer["output_dir"] = str(tok_dir)
    tokenizer["dataset"]["cache_dir"] = str(root / "data")
    (root / "tokenizer.yaml").write_text(yaml.safe_dump(tokenizer, sort_keys=False))
    config = yaml.safe_load((ROOT / "configs/smoke_cpu.yaml").read_text())
    config["tokenizer"]["path"] = str(tok_dir / "tokenizer.json")
    config["dataset"]["cache_dir"] = str(root / "data")
    config["logging"]["root_dir"] = str(root / "unused-runs")
    config["training"]["max_steps"] = max_steps
    config["training"]["max_tokens"] = 128 * max_steps
    config["optimizer"]["warmup_steps"] = 1
    config["checkpoint"]["every_steps"] = 4
    config["evaluation"]["every_steps"] = 4
    baseline = root / "baseline.yaml"
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    if train_tokenizer_now:
        if _CACHED_SMOKE_TOKENIZER is not None and not tok_dir.exists():
            tok_dir.mkdir(parents=True, exist_ok=True)
            (tok_dir / "tokenizer.json").write_bytes(_CACHED_SMOKE_TOKENIZER[0])
            (tok_dir / "tokenizer_manifest.json").write_bytes(
                _CACHED_SMOKE_TOKENIZER[1]
            )
        else:
            out = train_tokenizer(load_tokenizer_config(root / "tokenizer.yaml"))
            if _CACHED_SMOKE_TOKENIZER is None:
                _CACHED_SMOKE_TOKENIZER = (
                    out.read_bytes(),
                    (tok_dir / "tokenizer_manifest.json").read_bytes(),
                )
    return baseline
