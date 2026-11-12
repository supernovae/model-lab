"""Model explorer: look inside a small checkpoint (data layer, no Streamlit).

What it shows and where each part comes from:

* **architecture** (layers, attention heads, MoE experts, memory tables) and
  parameter counts: the run's resolved config through the shape-only
  :mod:`klab.model.inspection` schema, so it is available before (and
  without) loading any weights;
* **weights**: per-tensor norms, spread and histograms of the checkpoint
  loaded by the verified loader (``klab.probes.cli.load_target``, the same
  one ``klab probe`` uses);
* **tokens**: per-token loss and top-k next-token predictions over a text,
  from the shared scoring path (:func:`klab.probes.scoring.log_probs`);
* **attention** patterns per layer and head
  (:meth:`DenseAttention.attention_probabilities`, the reference forward's
  own weights), **expert routing** per token (the MoE layer's ``full``
  diagnostics) and **memory lookups** per token (the addresses each memory
  table was read at, and the gate), captured with forward hooks during one
  forward pass.

Big checkpoints are refused before loading (:func:`size_check`). Results are
cached per (checkpoint digest, text, actual runtime) under ``LAB/explorer`` as sealed JSON; a
cancel sentinel and an optional resource envelope are honored between stages
through :class:`~klab.lab_context.LabContext`.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from klab.lab_context import LabContext, is_out_of_memory, release_memory
from klab.lab_records import is_sealed, jsonable, seal, write_json_atomic

EXPLORER_FORMAT = "klab-explorer-v2"
_LEGACY_EXPLORER_FORMAT = "klab-explorer-v1"
_READABLE_EXPLORER_FORMATS = frozenset({EXPLORER_FORMAT, _LEGACY_EXPLORER_FORMAT})
ACTIVATION_SITES = ("residual_pre", "residual_mid", "residual_post")
_ARCH_COMPARE_FIELDS = (
    "hidden_dim",
    "layers",
    "memory",
    "vocab_size",
    "tie_embeddings",
    "ffn_dim",
)
# Loading and hooking a model this size on a laptop CPU takes seconds; beyond
# it the explorer refuses and says so (attention maps grow with heads x T^2).
MAX_PARAMETERS = 60_000_000
MAX_TOKENS = 128
TOP_K = 5
HISTOGRAM_BINS = 40
HISTOGRAM_SAMPLE = 200_000
DEFAULT_TEXT = (
    "Once upon a time, a little dog found a ball in the park. "
    "The dog ran to the ball and picked it up."
)


class ExplorerUnavailable(ValueError):
    """This checkpoint cannot be explored (too big, unsupported engine...)."""


# --- Architecture (shape-only, before loading) --------------------------------


def run_config(run: Path) -> Any:
    from klab.config.models import RunConfig

    return RunConfig.model_validate_json(
        (run / "resolved_config.yaml").read_text(encoding="utf-8")
    )


def architecture(config: Any) -> dict[str, Any]:
    """Layers, heads, experts and memory tables with their parameter counts."""
    from klab.model.inspection import inspection_report, named_tensor_inventory

    m, attention = config.model, config.attention
    tensors = named_tensor_inventory(m, attention)

    def count(prefix: str) -> int:
        return sum(
            spec.numel
            for name, spec in tensors.items()
            if name.startswith(prefix) and spec.alias_of is None
        )

    layers = []
    base_model, base_attention = m, attention
    for index, (m, attention) in enumerate(base_model.layer_specs(base_attention)):
        block = f"blocks.{index}"
        layer: dict[str, Any] = {
            "index": index,
            "attention": {
                "kind": attention.kind,
                "heads": m.num_heads,
                "kv_heads": m.num_kv_heads or m.num_heads,
                "head_dim": m.hidden_dim // m.num_heads,
                "window": attention.window_size,
                "parameters": count(f"{block}.attention."),
            },
            "ffn": {
                "kind": m.ffn,
                "experts": m.num_experts if m.ffn == "moe" else 0,
                "experts_per_token": m.experts_per_token if m.ffn == "moe" else 0,
                "shared_expert": bool(m.ffn == "moe" and m.shared_expert),
                "parameters": count(f"{block}.ffn."),
                "per_expert": count(f"{block}.ffn.experts.0.") if m.ffn == "moe" else 0,
            },
            "norm_parameters": count(f"{block}.norm"),
        }
        layers.append(layer)
    m, attention = base_model, base_attention
    memory = None
    if m.external_memory:
        memory = {
            "kind": m.memory,
            "injection": m.memory_injection,
            "block_index": m.memory_block_index,
            "params": dict(m.memory_params),
            "parameters": count("memory."),
        }
    elif m.memory != "none":
        streams = (
            (len(m.memory_ngram_orders) or 1) * m.memory_hash_heads
            if m.memory == "ngram"
            else 1
        )
        memory = {
            "kind": m.memory,
            "injection": m.memory_injection,
            "variant": m.memory_variant,
            "conv_kernel_size": m.memory_conv_kernel_size,
            "block_index": m.memory_block_index,
            "tables": streams * max(1, len(m.memory_block_sites)),
            **(
                {"block_indices": list(m.memory_block_sites)}
                if m.memory_block_indices
                else {}
            ),
            "rows": m.memory_table_size,
            **(
                {
                    "rows_per_stream": list(m.memory_table_sizes),
                    "rows_scope": "first_table",
                }
                if m.memory_table_sizes
                else {}
            ),
            "dim": m.memory_dim,
            "ngram_orders": list(m.memory_ngram_orders or (m.memory_ngram_size,))
            if m.memory == "ngram"
            else None,
            "parameters": count("memory.") + count("extra_memories."),
        }
    return {
        "vocab_size": m.vocab_size,
        "hidden_dim": m.hidden_dim,
        "ffn_dim": m.ffn_dim,
        "max_seq_len": m.max_seq_len,
        "tie_embeddings": m.tie_embeddings,
        "embedding_parameters": count("embedding."),
        "output_parameters": 0 if m.tie_embeddings else count("output."),
        "layers": layers,
        "memory": memory,
        "inventory": inspection_report(config),
    }


def size_check(arch: Mapping[str, Any]) -> str | None:
    """Why this model is too big to explore here (None when it fits)."""
    total = int(arch["inventory"]["total"])
    if total > MAX_PARAMETERS:
        return (
            f"{total:,} parameters is above the explorer's {MAX_PARAMETERS:,} "
            "limit (attention maps and per-tensor histograms would not fit an "
            "interactive budget). The architecture and parameter counts below "
            "come from the config alone; probe it with `klab probe` instead."
        )
    return None


# --- Measurements on a loaded checkpoint --------------------------------------


def weight_stats(model: Any) -> list[dict[str, Any]]:
    """Per stored tensor: shape, norm, spread and a histogram of its values."""
    out = []
    for name, parameter in model.named_parameters():
        values = parameter.detach().float().flatten().cpu().numpy()
        step = max(1, values.size // HISTOGRAM_SAMPLE)
        sample = values[::step]
        counts, edges = np.histogram(sample, bins=HISTOGRAM_BINS)
        out.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(values.size),
                "norm": float(np.linalg.norm(values)),
                "rms": float(np.sqrt(np.mean(values**2))) if values.size else 0.0,
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std()) if values.size else 0.0,
                "abs_max": float(np.abs(values).max()) if values.size else 0.0,
                "trainable": bool(parameter.requires_grad),
                "histogram": {
                    "counts": counts.tolist(),
                    "edges": edges.tolist(),
                    "sampled_every": step,
                },
            }
        )
    return out


def _token_text(loaded: Any, token: int) -> str:
    text = str(loaded.tokenizer.decode([int(token)]))
    if text:
        return text
    name = loaded.tokenizer.id_to_token(int(token))
    return f"⟨{name}⟩" if name else f"⟨{int(token)}⟩"


def tokenize(loaded: Any, text: str) -> tuple[list[int], bool]:
    from klab.probes import scoring

    ids = [scoring.eot(loaded), *scoring.encode(loaded, text)]
    limit = min(MAX_TOKENS, int(loaded.config.model.max_seq_len))
    return ids[:limit], len(ids) > limit


def token_view(loaded: Any, ids: list[int]) -> list[dict[str, Any]]:
    """Per position: the token, its loss given the prefix, the top-k guesses
    for the next token (from the shared scoring path)."""
    from klab.probes import scoring

    rows = scoring.log_probs(loaded, ids)
    out = []
    for position, token in enumerate(ids):
        loss = None if position == 0 else float(-rows[position - 1, token])
        top = np.argsort(-rows[position])[:TOP_K]
        rank = (
            None
            if position == 0
            else int((rows[position - 1] > rows[position - 1, token]).sum()) + 1
        )
        out.append(
            {
                "position": position,
                "id": int(token),
                "text": _token_text(loaded, token),
                "loss": loss,
                "rank": rank,
                "next_top": [
                    {
                        "id": int(t),
                        "text": _token_text(loaded, t),
                        "p": float(np.exp(rows[position, t])),
                    }
                    for t in top
                ],
            }
        )
    return out


def _forward_with_hooks(loaded: Any, ids: list[int]) -> dict[str, Any]:
    """One forward pass; capture activations, attention inputs, routing and memory reads."""
    import torch

    from klab.model.attention.dense import DenseAttention
    from klab.model.moe import TopKMoE
    from klab.probes import scoring

    model = loaded.model
    captured: dict[str, Any] = {
        "attention": {},
        "memory": {},
        "activation": {str(index): {} for index in range(len(model.blocks))},
    }
    handles = []

    def keep_rms(layer_key: str, site: str) -> Callable[..., None]:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            rms = (
                args[0].detach().float().square().mean(dim=-1).sqrt()[0].cpu().tolist()
            )
            if not all(math.isfinite(v) for v in rms):
                raise ExplorerUnavailable(
                    f"non-finite activation RMS at layer {layer_key} {site}"
                )
            captured["activation"][layer_key][site] = [float(v) for v in rms]

        return hook

    def keep_input(key: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], _output: Any) -> None:
            captured["attention"][key] = args[0].detach()

        return hook

    def keep_lookup(name: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], _output: Any) -> None:
            captured["memory"][name] = args[0].detach()

        return hook

    num_blocks = len(model.blocks)
    for index, block in enumerate(model.blocks):
        layer_key = str(index)
        handles.append(
            block.norm1.register_forward_pre_hook(keep_rms(layer_key, "residual_pre"))
        )
        handles.append(
            block.norm2.register_forward_pre_hook(keep_rms(layer_key, "residual_mid"))
        )
        post_target = (
            model.blocks[index + 1].norm1 if index + 1 < num_blocks else model.norm
        )
        handles.append(
            post_target.register_forward_pre_hook(keep_rms(layer_key, "residual_post"))
        )
        if isinstance(block.attention, DenseAttention):
            handles.append(block.attention.register_forward_hook(keep_input(layer_key)))
    memory = getattr(model, "memory", None)
    ports = ([("memory", memory)] if memory is not None else []) + [
        (f"extra_memories.{site}", module)
        for site, module in model.extra_memories.items()
    ]
    captured["gates_by_port"] = {}

    def keep_gate(port: str) -> Callable[..., None]:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured["gates_by_port"][port] = output.detach()
            if port == "memory":
                captured["gate"] = output.detach()

        return hook

    for port, memory in ports:
        for name, module in memory.named_modules():
            if isinstance(module, torch.nn.Embedding):
                handles.append(
                    module.register_forward_hook(
                        keep_lookup(name if port == "memory" else f"{port}.{name}")
                    )
                )
        handles.append(memory.gate.register_forward_hook(keep_gate(port)))
    x = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    try:
        with torch.inference_mode():
            model(x, **scoring._byte_inputs(loaded, ids), diagnostics="full")
            attention = {}
            for index, block in enumerate(model.blocks):
                key = str(index)
                if key in captured["attention"]:
                    weights = block.attention.attention_probabilities(
                        captured["attention"][key]
                    )
                    attention[key] = weights[0].cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    routing = {}
    for index, block in enumerate(model.blocks):
        if isinstance(block.ffn, TopKMoE) and block.ffn.last_diagnostics is not None:
            d = block.ffn.last_diagnostics
            routing[str(index)] = {
                "selected": d.selected_experts.cpu().numpy().tolist(),
                "weights": d.selected_weights.float().cpu().numpy().tolist(),
                "counts": d.counts.cpu().numpy().tolist(),
                "entropy": float(d.entropy),
            }
    activation = {
        "metric": "rms",
        "sites": list(ACTIVATION_SITES),
        "layers": {
            str(index): {
                site: captured["activation"][str(index)][site]
                for site in ACTIVATION_SITES
            }
            for index in range(num_blocks)
        },
    }
    return {
        **captured,
        "activation": activation,
        "attention": attention,
        "routing": routing,
    }


def attention_view(
    arch: Mapping[str, Any], attention: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    layers = {}
    for key, weights in attention.items():
        # Rounded to keep the cached payload small; rows still sum to ~1.
        layers[key] = np.round(weights.astype(np.float64), 4).tolist()
    missing = [
        layer["index"]
        for layer in arch["layers"]
        if str(layer["index"]) not in attention
    ]
    note = None
    if missing:
        kinds = sorted({arch["layers"][i]["attention"]["kind"] for i in missing})
        note = (
            f"attention weights are shown for dense attention only; "
            f"{', '.join(kinds)} layer(s) {missing} select or compress keys "
            "and expose no per-key weights"
        )
    return {"layers": layers, "note": note}


def _single_token_offset(
    addresses: list[int], ids: list[int], loaded: Any
) -> int | None:
    """The offset k when every address is just ``token[p-k] mod rows``.

    An n-gram table whose rows depend on one token only is a unigram table in
    disguise (e.g. a hash multiplier that is a multiple of the table size).
    Checked from the observed reads; the hash itself is not re-implemented.
    """
    rows = int(getattr(loaded.config.model, "memory_table_size", 0) or 0)
    if (
        rows <= 0
        or len(addresses) < 4
        or loaded.config.model.memory != "ngram"
        or loaded.config.model.memory_addressing != "legacy"
        or loaded.config.model.memory_token_mapping is not None
    ):
        return None
    for offset in range(4):
        expected = [
            ids[p - offset] % rows if p >= offset else None
            for p in range(len(addresses))
        ]
        tail = [
            (a, e) for a, e in zip(addresses, expected, strict=True) if e is not None
        ]
        if len(tail) >= 3 and all(a == e for a, e in tail):
            return offset
    return None


def memory_view(
    loaded: Any, ids: list[int], captured: Mapping[str, Any]
) -> dict | None:
    lookups = captured.get("memory") or {}
    if not lookups:
        return None
    streams = []
    for name, addresses in sorted(lookups.items()):
        row = addresses[0].cpu().numpy().tolist()
        first: dict[int, int] = {}
        reuse = []
        for position, address in enumerate(row):
            seen = first.setdefault(address, position)
            reuse.append(None if seen == position else seen)
        streams.append(
            {
                "table": name,
                "addresses": row,
                "first_seen_at": reuse,
                "single_token_offset": _single_token_offset(row, ids, loaded),
            }
        )
    gate = captured.get("gate")
    gates = (
        None
        if gate is None
        else [float(v) for v in 1 / (1 + np.exp(-gate[0, :, 0].float().cpu().numpy()))]
    )
    return {
        "kind": loaded.config.model.memory,
        "streams": streams,
        "gate": gates,
        **(
            {
                "gates_by_port": {
                    name: value[0, :, 0].float().sigmoid().cpu().tolist()
                    for name, value in captured["gates_by_port"].items()
                }
            }
            if len(captured.get("gates_by_port", {})) > 1
            else {}
        ),
        "note": "addresses are the table rows each token read; a repeat points "
        "at the first position that read the same row (the same n-gram, or a "
        "hash collision between different n-grams)",
    }


# --- Orchestration and cache ---------------------------------------------------


def runtime_identity(runtime: Any) -> dict[str, Any]:
    """Canonical reuse subset of the native loader's *actual* runtime.

    Framework version, OS and device name remain provenance, not cache identity.
    In particular ROCm is a backend, even though its torch device type is cuda.
    """
    if runtime is None:
        raise ValueError("missing actual-runtime provenance")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("engine") != "pytorch"
        or not isinstance(runtime.get("backend"), str)
        or runtime.get("backend") not in {"cpu", "cuda", "rocm", "mps", "xpu"}
        or type(runtime.get("device_index")) is not int
        or runtime["device_index"] < 0
        or not isinstance(runtime.get("precision"), str)
        or runtime.get("precision") not in {"fp32", "fp16", "bf16"}
    ):
        raise ValueError("malformed or unsupported actual-runtime provenance")
    return {
        key: runtime[key] for key in ("engine", "backend", "device_index", "precision")
    }


def runtime_label(value: Mapping[str, Any]) -> str:
    """Human-readable producer identity, including explicit legacy uncertainty."""
    try:
        runtime = runtime_identity(value.get("runtime"))
    except ValueError as error:
        return str(error)
    return (
        f"{runtime['engine']} / {runtime['backend']}:{runtime['device_index']} / "
        f"{runtime['precision']}"
    )


def _cache_key_for_format(
    fmt: str,
    checkpoint_sha256: str,
    text: str,
    runtime: Mapping[str, Any] | None = None,
) -> str:
    identity: dict[str, Any] = {
        "format": fmt,
        "checkpoint": checkpoint_sha256,
        "text": text,
    }
    if runtime is not None:
        identity["runtime"] = runtime_identity(runtime)
    payload = json.dumps(
        identity,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path_for_format(
    fmt: str,
    lab_dir: Path,
    checkpoint_sha256: str,
    text: str,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    return (
        lab_dir
        / "explorer"
        / checkpoint_sha256[:16]
        / f"{_cache_key_for_format(fmt, checkpoint_sha256, text, runtime)[:24]}.json"
    )


def _v1_cache_path(
    lab_dir: Path,
    checkpoint_sha256: str,
    text: str,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    return _cache_path_for_format(
        _LEGACY_EXPLORER_FORMAT, lab_dir, checkpoint_sha256, text, runtime
    )


def cache_path(
    lab_dir: Path,
    checkpoint_sha256: str,
    text: str,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    return _cache_path_for_format(
        EXPLORER_FORMAT, lab_dir, checkpoint_sha256, text, runtime
    )


def read_cached(path: Path) -> dict[str, Any] | None:
    """A cached exploration, or None when absent, edited or another format."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not is_sealed(value) or value.get("format") not in _READABLE_EXPLORER_FORMATS:
        return None
    return value


def cache_issue(value: Mapping[str, Any], path: Path) -> str | None:
    """Why sealed bytes cannot verify a producer-matched cache hit."""
    fmt = value.get("format")
    if fmt not in _READABLE_EXPLORER_FORMATS:
        return "unsupported explorer cache format"
    try:
        runtime = runtime_identity(value.get("runtime"))
    except ValueError as error:
        return str(error)
    target = value.get("target")
    sha = target.get("checkpoint_sha256") if isinstance(target, Mapping) else None
    text = value.get("text")
    if not isinstance(sha, str) or not isinstance(text, str):
        return "malformed checkpoint/text cache identity"
    expected = _cache_path_for_format(str(fmt), path.parents[2], sha, text, runtime)
    if path != expected:
        return "sealed payload/cache key mismatch"
    return None


def cached_explorations(lab_dir: Path) -> list[dict[str, Any]]:
    """Every readable cached exploration, newest first."""
    out = []
    for path in sorted((lab_dir / "explorer").glob("*/*.json")):
        value = read_cached(path)
        if value is not None:
            out.append(
                {**value, "path": str(path), "cache_issue": cache_issue(value, path)}
            )
    return sorted(out, key=lambda v: v.get("created_at", ""), reverse=True)


def _require_supported(loaded: Any) -> None:
    if getattr(loaded, "engine", None) is not None:
        raise ExplorerUnavailable(
            "the explorer needs the PyTorch engine (MLX not supported)"
        )
    if getattr(loaded.model, "semantic_memories", None):
        raise ExplorerUnavailable(
            "the explorer does not support attached semantic packs"
        )


def explore_loaded(
    loaded: Any,
    text: str,
    *,
    context: LabContext | None = None,
    arch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every explorer view of a loaded Model Lab checkpoint."""
    _require_supported(loaded)
    runtime_identity(loaded.identity.get("runtime"))
    started = time.monotonic()
    arch = dict(arch or architecture(loaded.config))
    state: dict[str, Any] = {}

    def checkpoint(stage: str) -> None:
        if context is not None:
            context.enter("explorer", stage)
            context.checkpoint()

    ids, truncated = tokenize(loaded, text)
    if len(ids) < 2:
        raise ExplorerUnavailable("the text needs at least one token")
    stages: list[tuple[str, Callable[[], Any]]] = [
        ("weights", lambda: weight_stats(loaded.model)),
        ("tokens", lambda: token_view(loaded, ids)),
        ("forward", lambda: _forward_with_hooks(loaded, ids)),
    ]
    was_training = loaded.model.training
    loaded.model.eval()
    try:
        for stage, work in stages:
            checkpoint(stage)
            state[stage] = work()
    finally:
        loaded.model.train(was_training)
    forward = state["forward"]
    identity = loaded.identity
    return {
        "format": EXPLORER_FORMAT,
        "runtime": dict(identity["runtime"]),
        "created_at": datetime.now(UTC).isoformat(),
        "target": {
            "run_id": identity.get("run_id"),
            "checkpoint": identity.get("checkpoint_relative_path"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "tokenizer_sha256": identity.get("tokenizer_sha256"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
        },
        "text": text,
        "truncated": truncated,
        "architecture": arch,
        "weights": state["weights"],
        "tokens": state["tokens"],
        "activation": forward["activation"],
        "attention": attention_view(arch, forward["attention"]),
        "routing": forward["routing"] or None,
        "memory": memory_view(loaded, ids, forward),
        "seconds": round(time.monotonic() - started, 3),
    }


def explore(
    target: str,
    *,
    lab_dir: Path,
    runs_dir: Path | None = None,
    text: str = DEFAULT_TEXT,
    backend: str | None = "cpu",
    authorization: Any = None,
    resource_envelope: Any = None,
    refresh: bool = False,
    load: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Explore TARGET, cached per checkpoint+text+canonical actual runtime.

    Returns ``(exploration, cache path)``. A too-big model returns only its
    architecture with ``unavailable`` set (nothing is loaded).
    """
    from klab.lab_results import resolve_checkpoint
    from klab.probes.cli import load_target
    from klab.resource_envelope import require_envelope_applicability

    require_envelope_applicability(resource_envelope, operation="explore")
    resolved = resolve_checkpoint(target, lab_dir=lab_dir, runs_dir=runs_dir)
    arch = architecture(run_config(resolved.run))
    too_big = size_check(arch)
    if too_big is not None:
        return (
            {
                "format": EXPLORER_FORMAT,
                "target": {"run_id": resolved.run.name},
                "architecture": arch,
                "unavailable": too_big,
            },
            None,
        )
    folder = lab_dir / "explorer"
    operations = folder / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    operation_id = uuid.uuid4().hex
    operation = operations / operation_id
    operation.mkdir(mode=0o700)
    context = LabContext(
        operation / "CANCEL", resource_envelope=resource_envelope, workspace=lab_dir
    )
    print(
        f"explore {operation_id}: touch {context.cancel_path} to stop at the next stage",
        file=sys.stderr,
        flush=True,
    )
    loader = load or (
        lambda: load_target(
            target,
            lab_dir=lab_dir,
            runs_dir=runs_dir,
            backend=backend,
            authorization=authorization,
        )
    )
    # Loading is part of the protected lifecycle: a pre-load stage check
    # (cancel sentinel + resource envelope), the OOM handler and the cleanup
    # cover it exactly like the exploration stages.
    loaded = None
    try:
        context.enter("explorer", "load")
        context.checkpoint()
        try:
            loaded = loader()
            _require_supported(loaded)
            sha = str(loaded.identity.get("checkpoint_sha256"))
            runtime = runtime_identity(loaded.identity.get("runtime"))
            path = cache_path(lab_dir, sha, text, runtime)
            cached = None if refresh else read_cached(path)
            reason = "refresh requested" if refresh else "no cache for actual runtime"
            if cached is not None and cached.get("format") == EXPLORER_FORMAT:
                issue = cache_issue(cached, path)
                if issue is None and (
                    cached["target"]["checkpoint_sha256"] != sha
                    or cached["text"] != text
                    or runtime_identity(cached["runtime"]) != runtime
                ):
                    issue = "sealed payload/cache key mismatch"
                if issue is None:
                    return seal({**cached, "cache_reuse": {"status": "hit"}}), path
                reason = issue
            elif not refresh and path.exists():
                reason = "unreadable, unsealed or wrong-format cache"
            elif not refresh:
                v1_runtime = _v1_cache_path(lab_dir, sha, text, runtime)
                if v1_runtime.exists():
                    v1_value = read_cached(v1_runtime)
                    reason = (
                        (
                            cache_issue(v1_value, v1_runtime)
                            or "historical v1 cache has no activation diagnostics"
                        )
                        if v1_value is not None
                        and v1_value.get("format") == _LEGACY_EXPLORER_FORMAT
                        else "unreadable, unsealed or wrong-format legacy cache"
                    )
                else:
                    legacy = _v1_cache_path(lab_dir, sha, text)
                    if legacy.exists():
                        legacy_value = read_cached(legacy)
                        reason = (
                            cache_issue(legacy_value, legacy)
                            if legacy_value is not None
                            else "unreadable, unsealed or wrong-format legacy cache"
                        ) or "legacy cache has no verified actual-runtime binding"
            result = explore_loaded(loaded, text, context=context, arch=arch)
        except Exception as error:
            if not is_out_of_memory(error):
                raise
            stage = context.phase or "load"
            raise ExplorerUnavailable(
                f"out of memory while exploring ({stage}): {type(error).__name__}"
            ) from None
    finally:
        del loaded
        release_memory()
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal(jsonable(result))
    write_json_atomic(path, sealed)
    # Invocation status is a sealed overlay; stored diagnostic bytes describe
    # their producer and are unchanged by a hit.
    return seal({**sealed, "cache_reuse": {"status": "miss", "reason": reason}}), path


def _validated_activation_layers(
    side: Mapping[str, Any], token_count: int, side_name: str
) -> dict[int, dict[str, list[float]]]:
    arch = side.get("architecture")
    if not isinstance(arch, Mapping) or not isinstance(arch.get("layers"), list):
        raise ExplorerUnavailable(f"{side_name} exploration is missing architecture")
    expected_indices = []
    for layer in arch["layers"]:
        if not isinstance(layer, Mapping) or type(layer.get("index")) is not int:
            raise ExplorerUnavailable(
                f"{side_name} exploration has malformed architecture layers"
            )
        expected_indices.append(layer["index"])
    activation = side.get("activation")
    if (
        not isinstance(activation, Mapping)
        or activation.get("metric") != "rms"
        or list(activation.get("sites") or ()) != list(ACTIVATION_SITES)
        or not isinstance(activation.get("layers"), Mapping)
        or not activation["layers"]
    ):
        raise ExplorerUnavailable(
            f"{side_name} exploration is missing activation RMS diagnostics"
        )
    raw_layers = activation["layers"]
    if set(raw_layers.keys()) != {str(i) for i in expected_indices}:
        raise ExplorerUnavailable(
            f"{side_name} exploration activation layers do not match architecture"
        )
    validated: dict[int, dict[str, list[float]]] = {}
    for index in expected_indices:
        key = str(index)
        layer_sites = raw_layers.get(key)
        if not isinstance(layer_sites, Mapping):
            raise ExplorerUnavailable(
                f"{side_name} exploration is missing activation layer {key}"
            )
        site_map: dict[str, list[float]] = {}
        for site in ACTIVATION_SITES:
            values = layer_sites.get(site)
            if not isinstance(values, list) or len(values) != token_count:
                raise ExplorerUnavailable(
                    f"{side_name} exploration has invalid activation length at layer {key} {site}"
                )
            converted: list[float] = []
            for val in values:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ExplorerUnavailable(
                        f"{side_name} exploration has non-numeric activation at layer {key} {site}"
                    )
                fval = float(val)
                if not math.isfinite(fval):
                    raise ExplorerUnavailable(
                        f"{side_name} exploration has non-finite activation at layer {key} {site}"
                    )
                converted.append(fval)
            site_map[site] = converted
        validated[index] = site_map
    return validated


def compare_explorations(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two sealed v2 explorations on the exact same tokenized sentence."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ExplorerUnavailable("both explorations must be sealed v2 records")
    for side_name, side in (("left", left), ("right", right)):
        if side.get("unavailable"):
            raise ExplorerUnavailable(
                f"{side_name} exploration is unavailable: {side['unavailable']}"
            )
        if side.get("cache_issue"):
            raise ExplorerUnavailable(
                f"{side_name} exploration has unverified cache provenance: {side['cache_issue']}"
            )
        candidate = (
            {k: v for k, v in side.items() if k not in {"path", "cache_issue"}}
            if "path" in side and "cache_issue" in side and not is_sealed(dict(side))
            else dict(side)
        )
        if not is_sealed(candidate):
            raise ExplorerUnavailable(
                f"{side_name} exploration record is unsealed or unverified"
            )
        if isinstance(side.get("path"), str):
            issue = cache_issue(candidate, Path(side["path"]))
            if issue is not None:
                raise ExplorerUnavailable(
                    f"{side_name} exploration has unverified cache provenance: {issue}"
                )
        if side.get("format") != EXPLORER_FORMAT:
            raise ExplorerUnavailable(
                f"{side_name} exploration uses format {side.get('format')!r}; "
                f"paired comparison requires {EXPLORER_FORMAT} (historical v1 has no activations)"
            )
    try:
        left_runtime = runtime_identity(left.get("runtime"))
        right_runtime = runtime_identity(right.get("runtime"))
    except ValueError as error:
        raise ExplorerUnavailable(f"unverified runtime identity: {error}") from None
    if left_runtime != right_runtime:
        raise ExplorerUnavailable(
            "explorations were produced on different runtimes: "
            f"{runtime_label(left)} vs {runtime_label(right)}"
        )
    left_target = left.get("target") if isinstance(left.get("target"), Mapping) else {}
    right_target = (
        right.get("target") if isinstance(right.get("target"), Mapping) else {}
    )
    left_sha = left_target.get("checkpoint_sha256")
    right_sha = right_target.get("checkpoint_sha256")
    if (
        not isinstance(left_sha, str)
        or not left_sha
        or not isinstance(right_sha, str)
        or not right_sha
    ):
        raise ExplorerUnavailable("both explorations require target.checkpoint_sha256")
    if left_sha == right_sha:
        raise ExplorerUnavailable(
            "cannot compare identical checkpoints (same checkpoint_sha256)"
        )
    left_tok = left_target.get("tokenizer_sha256")
    right_tok = right_target.get("tokenizer_sha256")
    if (
        not isinstance(left_tok, str)
        or not left_tok
        or not isinstance(right_tok, str)
        or not right_tok
    ):
        raise ExplorerUnavailable("both explorations require target.tokenizer_sha256")
    if left_tok != right_tok:
        raise ExplorerUnavailable(
            "checkpoints have different tokenizers (tokenizer_sha256 mismatch)"
        )
    left_text = left.get("text")
    right_text = right.get("text")
    if (
        not isinstance(left_text, str)
        or not isinstance(right_text, str)
        or left_text != right_text
    ):
        raise ExplorerUnavailable("explorations must use the exact same text")
    if bool(left.get("truncated")) or bool(right.get("truncated")):
        raise ExplorerUnavailable(
            "cannot compare truncated inputs; shorten the sample text"
        )
    left_tokens = left.get("tokens")
    right_tokens = right.get("tokens")
    if (
        not isinstance(left_tokens, list)
        or not isinstance(right_tokens, list)
        or len(left_tokens) < 2
        or len(right_tokens) < 2
    ):
        raise ExplorerUnavailable("explorations are missing token sequences")
    try:
        left_ids = [int(token["id"]) for token in left_tokens]
        right_ids = [int(token["id"]) for token in right_tokens]
    except KeyError, TypeError, ValueError:
        raise ExplorerUnavailable(
            "explorations have malformed token sequences"
        ) from None
    if left_ids != right_ids:
        raise ExplorerUnavailable(
            "explorations produced different token-ID sequences (including EOS)"
        )
    token_count = len(left_ids)
    left_layers = _validated_activation_layers(left, token_count, "left")
    right_layers = _validated_activation_layers(right, token_count, "right")
    left_set = set(left_layers)
    right_set = set(right_layers)
    common_layers = sorted(left_set & right_set)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)
    left_arch = left["architecture"]
    right_arch = right["architecture"]
    architecture_matches = all(
        left_arch.get(field) == right_arch.get(field) for field in _ARCH_COMPARE_FIELDS
    )
    delta: dict[str, dict[str, list[float]]] | None = None
    if architecture_matches:
        delta = {
            str(index): {
                site: [
                    right_layers[index][site][pos] - left_layers[index][site][pos]
                    for pos in range(token_count)
                ]
                for site in ACTIVATION_SITES
            }
            for index in common_layers
        }
    return {
        "left": left,
        "right": right,
        "common_layers": common_layers,
        "only_left": only_left,
        "only_right": only_right,
        "architecture_matches": architecture_matches,
        "delta": delta,
    }
