"""Pinned external reference models (Pythia, SmolLM2), loaded safely.

Every reference is a fixed Hugging Face repository at an immutable commit. Only
static metadata and ``*.safetensors`` weights may enter a snapshot (no pickles,
no remote code, no quantization configs).

A loaded reference behaves like a loaded Model Lab run for the probe battery
(``model``, ``tokenizer``, ``config.model.max_seq_len``, ``identity``), so it
is scored by the same :mod:`klab.probes.scoring` path. It is only
compared on tokenizer-independent standard tasks (lm-eval); its held-out loss
on our validation data is never computed (different tokenizer and data).

Install with ``uv sync --extra reference --extra lmeval``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from klab.hf_auth import hub_access_errors, hub_auth_kwargs, raise_for_hub_auth

REFERENCE_PREFIX = "ref:"
# lm-eval prompts are short; cap the scoring window so CPU memory stays bounded.
REFERENCE_MAX_SEQ_LEN = 2048
INSTALL_HINT = "uv sync --extra reference --extra lmeval"

SAFE_SNAPSHOT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "README.md",
        "LICENSE",
        "LICENSE.md",
    }
)
SNAPSHOT_ALLOW_PATTERNS = (*sorted(SAFE_SNAPSHOT_FILES), "*.safetensors")


class ReferenceUnavailable(RuntimeError):
    """The optional reference dependencies are not installed."""


@dataclass(frozen=True)
class ReferenceModel:
    """One pinned public checkpoint and the facts its model card declares."""

    name: str
    family: str
    repo_id: str
    revision: str  # immutable commit
    model_type: str
    architecture: str
    license: str
    step: int | None
    training_tokens: int
    training_tokens_source: str


REFERENCES: Mapping[str, ReferenceModel] = {
    model.name: model
    for model in (
        ReferenceModel(
            "pythia-70m-deduped",
            "Pythia",
            "EleutherAI/pythia-70m-deduped",
            "9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb",  # step143000 (final)
            "gpt_neox",
            "GPTNeoXForCausalLM",
            "Apache-2.0",
            143000,
            299_892_736_000,
            "model card: 299,892,736,000 tokens seen during training",
        ),
        ReferenceModel(
            "pythia-160m-deduped",
            "Pythia",
            "EleutherAI/pythia-160m-deduped",
            "c54a0e0b28cc667b6f278803024438d57f847b5d",  # step143000 (final)
            "gpt_neox",
            "GPTNeoXForCausalLM",
            "Apache-2.0",
            143000,
            299_892_736_000,
            "model card: 299,892,736,000 tokens seen during training",
        ),
        ReferenceModel(
            "SmolLM2-135M",
            "SmolLM2",
            "HuggingFaceTB/SmolLM2-135M",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "llama",
            "LlamaForCausalLM",
            "Apache-2.0",
            None,
            2_000_000_000_000,
            "model card: 'Pretraining tokens: 2T' (rounded by the authors)",
        ),
        ReferenceModel(
            "SmolLM2-360M",
            "SmolLM2",
            "HuggingFaceTB/SmolLM2-360M",
            "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "llama",
            "LlamaForCausalLM",
            "Apache-2.0",
            None,
            4_000_000_000_000,
            "model card: 'Pretraining tokens: 4T' (rounded by the authors)",
        ),
    )
}


def is_reference(spec: str) -> bool:
    return spec.startswith(REFERENCE_PREFIX)


def reference_for(spec: str) -> ReferenceModel:
    """``ref:NAME`` (or NAME) -> the pinned reference; unknown names are refused."""
    name = spec.removeprefix(REFERENCE_PREFIX)
    try:
        return REFERENCES[name]
    except KeyError:
        known = ", ".join(f"{REFERENCE_PREFIX}{n}" for n in REFERENCES)
        raise ValueError(f"unknown reference {spec!r}; known: {known}") from None


# --- Safe snapshots (shared with research.pythia) ---------------------------


def read_json_object(path: Path, label: str = "reference") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} metadata JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} metadata must be an object: {path.name}")
    return value


def validate_snapshot(snapshot: Path, label: str = "reference") -> list[Path]:
    """Reject every downloaded file outside the safe static/safetensor allowlist."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError(f"{label} snapshot must be a regular directory")
    files: list[Path] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} snapshot contains unsafe path: {path}")
        relative = path.relative_to(snapshot).as_posix()
        if "/" in relative or (
            relative not in SAFE_SNAPSHOT_FILES
            and not relative.endswith(".safetensors")
        ):
            raise ValueError(f"{label} snapshot contains disallowed file: {relative}")
        files.append(path)
    names = {path.name for path in files}
    missing = sorted({"config.json", "tokenizer.json"} - names)
    if missing:
        raise ValueError(
            f"{label} snapshot lacks required files: " + ", ".join(missing)
        )
    safetensors = {path.name for path in files if path.suffix == ".safetensors"}
    if not safetensors:
        raise ValueError(f"{label} snapshot contains no safetensors weights")
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weights = read_json_object(index, label).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"{label} safetensors index has no weight map")
        shards = set(weights.values())
        if (
            any(not isinstance(shard, str) or "/" in shard for shard in shards)
            or not shards <= safetensors
        ):
            raise ValueError(
                f"{label} safetensors index references unsafe or missing shards"
            )
    return files


def validate_model_metadata(
    snapshot: Path, *, model_type: str, architecture: str, label: str = "reference"
) -> dict[str, Any]:
    config = read_json_object(snapshot / "config.json", label)
    if config.get("model_type") != model_type:
        raise ValueError(f"{label} config must declare model_type={model_type}")
    if config.get("architectures") != [architecture]:
        raise ValueError(f"{label} config must declare {architecture} only")
    for path in (snapshot / "config.json", snapshot / "tokenizer_config.json"):
        if path.exists():
            metadata = read_json_object(path, label)
            if "auto_map" in metadata or "quantization_config" in metadata:
                raise ValueError(
                    f"{label} metadata forbids remote code or quantization: {path.name}"
                )
    return config


def safe_snapshot(
    repo_id: str,
    revision: str,
    cache_dir: Path | None,
    *,
    model_type: str,
    architecture: str,
    label: str = "reference",
) -> tuple[Path, list[Path]]:
    """Download (or reuse the cache of) a pinned safe snapshot and validate it."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "allow_patterns": list(SNAPSHOT_ALLOW_PATTERNS),
    }
    kwargs.update(hub_auth_kwargs())
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        snapshot = Path(snapshot_download(**kwargs))
    except hub_access_errors() as error:
        raise_for_hub_auth(error, credential_supplied="token" in kwargs)
    files = validate_snapshot(snapshot, label)
    validate_model_metadata(
        snapshot, model_type=model_type, architecture=architecture, label=label
    )
    return snapshot, files


def _sha256(path: Path) -> str:
    from klab.training.manifest import sha256_file

    return sha256_file(path)


def weights_digest(files: Sequence[Path]) -> str:
    """One digest over every safetensors shard (name and content)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in files if p.suffix == ".safetensors"):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


# --- Loading ------------------------------------------------------------------


@dataclass
class _ModelConfig:
    max_seq_len: int
    memory: str = "none"


@dataclass
class _Config:
    model: _ModelConfig


class ReferenceRun:
    """A pinned public checkpoint shaped like a loaded Model Lab run."""

    engine = None

    def __init__(
        self,
        reference: ReferenceModel,
        model: Any,
        tokenizer: Any,
        *,
        max_seq_len: int,
        eot_token_id: int,
        identity: dict[str, Any],
        device: Any,
    ) -> None:
        self.reference = reference
        self.model = model
        self.tokenizer = tokenizer
        self.config = _Config(_ModelConfig(max_seq_len=max_seq_len))
        self.eot_token_id = eot_token_id
        self.identity = identity
        self.device = device
        # Not a filesystem path: references live in the Hugging Face cache.
        self.run = f"hf://{reference.repo_id}@{reference.revision}"

    def describe(self) -> dict[str, Any]:
        return {
            **asdict(self.reference),
            "url": f"https://huggingface.co/{self.reference.repo_id}",
        }


def _causal_lm(hf_model: Any) -> Any:
    import torch

    class CausalLM(torch.nn.Module):
        """``model(ids) -> logits``, like a Model Lab decoder."""

        def __init__(self) -> None:
            super().__init__()
            self.hf = hf_model

        def forward(self, x: Any, **_: Any) -> Any:
            return self.hf(input_ids=x).logits

    return CausalLM()


def active_parameters(total: int, embedding: int, hidden: int, tied: bool) -> int:
    """Same rule as Model Lab's inventory: an untied input table counts one row."""
    return total if tied else total - (embedding - hidden)


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """``dtype=`` on transformers >= 4.56, ``torch_dtype=`` before it."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def require_reference_extras() -> None:
    """Fail early, with the install hint, when the optional stack is missing."""
    import importlib.util

    missing = [
        name
        for name in ("transformers", "lm_eval")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ReferenceUnavailable(
            f"reference models need {', '.join(missing)}: {INSTALL_HINT}"
        )


def load_reference(
    spec: str, *, cache_dir: Path | None = None, device: str = "cpu"
) -> ReferenceRun:
    """Load ``ref:NAME`` from its pinned safe snapshot (fp32, eval mode)."""
    reference = reference_for(spec)
    try:
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise ReferenceUnavailable(
            f"reference models need the optional extras: {INSTALL_HINT}"
        ) from error
    snapshot, files = safe_snapshot(
        reference.repo_id,
        reference.revision,
        cache_dir,
        model_type=reference.model_type,
        architecture=reference.architecture,
        label=reference.name,
    )
    config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        **_dtype_kwargs(torch.float32),
    ).to(device)
    hf_model.eval()
    model = _causal_lm(hf_model)
    total = sum(p.numel() for p in model.parameters())
    tied = bool(getattr(config, "tie_word_embeddings", False))
    embedding = hf_model.get_input_embeddings().weight.numel()
    tokenizer_path = snapshot / "tokenizer.json"
    eot = config.eos_token_id
    if isinstance(eot, list):
        eot = eot[0]
    identity = {
        "run_id": f"{REFERENCE_PREFIX}{reference.name}",
        "checkpoint_relative_path": f"{reference.repo_id}@{reference.revision}",
        "checkpoint_sha256": weights_digest(files),
        "step": reference.step,
        "tokens_seen": reference.training_tokens,
        "tokenizer_sha256": _sha256(tokenizer_path),
        "data_sha256": {},
        "parameter_inventory": {
            "total": total,
            "active_per_token": active_parameters(
                total, embedding, int(config.hidden_size), tied
            ),
        },
    }
    return ReferenceRun(
        reference,
        model,
        Tokenizer.from_file(str(tokenizer_path)),
        max_seq_len=min(int(config.max_position_embeddings), REFERENCE_MAX_SEQ_LEN),
        eot_token_id=int(eot if eot is not None else 0),
        identity=identity,
        device=torch.device(device),
    )
