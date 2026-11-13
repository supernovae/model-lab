"""Export any built-in Lab architecture as Hugging Face remote code.

The export carries Lab's own decoder modules, not a second implementation of
them. :func:`vendor_modules` starts from the wrapper modules, follows every
``klab`` import and copies each module it reaches into the export as one flat
file (``klab.model.cache`` becomes ``klab_model_cache.py``), with its imports
rewritten to relative ones. It refuses the export when that closure reaches
anything but the decoder's modules, the standard library, Torch, safetensors
and transformers. Three rewrites keep the closure small:

* imports under ``if TYPE_CHECKING:`` never run and become ``pass``;
* ``klab.components`` (the registry, which reads entry points through pydantic)
  is replaced by :mod:`klab.component_builders`, the builders it registers for
  Lab's own kinds; and
* a function-level import of a module in :data:`NOT_EXPORTED` raises
  ``ImportError`` instead, on paths the export refuses up front.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic_core import to_jsonable_python

from klab.components import is_builtin
from klab.data.tokenizer import SPECIAL_TOKENS

if TYPE_CHECKING:
    from pydantic import BaseModel

    from klab.config.models import RunConfig

EXPORT_FORMAT = "hf_remote_code"
ROOTS = ("klab.model.hf_modeling", "klab.model.export_loader")
MODELING_MODULE = "klab.model.hf_modeling"
SUBSTITUTES = {"klab.components": "klab.component_builders"}
VENDORED = ("klab.model", "klab.address_hash", "klab.component_builders")
NOT_EXPORTED = {
    "klab.engram.semantic": "semantic memory packs attach at run time",
    "klab.model.attention.hip_sparse": "the ROCm block-sparse kernel is not exported",
    "klab.model.portable_engram": "portable memory reads a package outside the model",
}
THIRD_PARTY = frozenset({"torch", "safetensors", "transformers"})
_UNSUPPORTED_MEMORY = {
    "byte": "byte memory reads addresses Lab prepares from the tokenizer's bytes "
    "outside the model",
    "portable": "portable memory reads an Engram package outside the checkpoint",
}


def flat_name(module: str) -> str:
    return module.replace(".", "_")


def _vendorable(module: str) -> bool:
    return any(module == root or module.startswith(root + ".") for root in VENDORED)


def _type_checking(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


def _rewrite(module: str, source: str) -> tuple[str, list[str]]:
    """Return ``module``'s source with relative imports and the modules it needs."""
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    lines = source.splitlines(keepends=True)
    needed: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        typing_only = nested = False
        child, parent = node, parents.get(node)
        while parent is not None:
            if isinstance(parent, ast.If) and _type_checking(parent.test):
                typing_only = typing_only or child in parent.body
            if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                nested = True
            child, parent = parent, parents.get(parent)
        if isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError(f"{module} uses a relative import; write absolute imports")
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        top = {name.split(".")[0] for name in names}
        if top - {"klab"}:
            unknown = sorted(
                name
                for name in top - {"klab"}
                if name not in sys.stdlib_module_names and name not in THIRD_PARTY
            )
            if unknown:
                raise ValueError(
                    f"{module} imports {', '.join(unknown)}, which an export "
                    "does not carry"
                )
            continue
        if typing_only:
            replacement = "pass"
        elif isinstance(node, ast.Import) or node.module == "klab":
            raise ValueError(
                f"{module} must import klab modules as 'from klab.x import y'"
            )
        else:
            target = SUBSTITUTES.get(node.module, node.module)
            if target in NOT_EXPORTED:
                if not nested:
                    raise ValueError(
                        f"{module} imports {target} at module level; "
                        f"{NOT_EXPORTED[target]}"
                    )
                reason = f"{target} is not part of a Lab export: {NOT_EXPORTED[target]}"
                replacement = f"raise ImportError({reason!r})"
            elif _vendorable(target):
                imported = ", ".join(
                    alias.name
                    if alias.asname is None
                    else f"{alias.name} as {alias.asname}"
                    for alias in node.names
                )
                replacement = f"from .{flat_name(target)} import {imported}"
                needed.append(target)
            else:
                raise ValueError(
                    f"{module} imports {target}, which is outside the model code an "
                    "export carries"
                )
        start, end = node.lineno - 1, node.end_lineno or node.lineno
        indent = lines[start][: node.col_offset]
        # Blank continuation lines keep the line structure of the original file.
        lines[start:end] = [f"{indent}{replacement}\n"] + ["\n"] * (end - start - 1)
    return "".join(lines), needed


def vendor_modules(source_sha256: str) -> dict[str, tuple[str, str]]:
    """Map each export file name to its module and rewritten source."""
    files: dict[str, tuple[str, str]] = {}
    pending = list(ROOTS)
    while pending:
        module = pending.pop()
        name = f"{flat_name(module)}.py"
        if name in files:
            continue
        spec = importlib.util.find_spec(module)
        origin = None if spec is None else spec.origin
        if (
            origin is None
            or not origin.endswith(".py")
            or spec.submodule_search_locations
        ):
            raise ValueError(f"{module} is not a plain module an export can carry")
        source, needed = _rewrite(module, Path(origin).read_text(encoding="utf-8"))
        files[name] = (
            module,
            f"# {module} from Model Lab source {source_sha256}, "
            "copied by klab weights export.\n" + source,
        )
        pending.extend(needed)
    # transformers copies only the files a local directory's module imports
    # directly (a Hub repository is followed recursively), so the wrapper names
    # every other file once.
    modeling = f"{flat_name(MODELING_MODULE)}.py"
    module, source = files[modeling]
    files[modeling] = (
        module,
        source
        + "\n\n# Every vendored file, for transformers' copy of a local directory.\n"
        + "".join(
            f"from .{name.removesuffix('.py')} import __name__ as _  # noqa: E402\n"
            for name in sorted(files)
            if name != modeling
        ),
    )
    return files


def remote_code_refusal(config: RunConfig) -> str | None:
    """Why ``config`` cannot export as remote code, or ``None``."""
    model = config.model
    if model.memory in _UNSUPPORTED_MEMORY:
        return _UNSUPPORTED_MEMORY[model.memory]
    external = sorted(
        f"{slot} {kind}"
        for slot, kinds in (
            ("attention", config.attention_kinds),
            ("ffn", config.ffn_kinds),
            ("memory", {model.memory} - {"none"}),
        )
        for kind in kinds
        if not is_builtin(slot, kind)  # type: ignore[arg-type]
    )
    if external:
        return (
            f"{', '.join(external)} comes from another package; only Lab's "
            "built-in kinds export"
        )
    return None


def _settings(value: BaseModel, *, skip: frozenset[str] = frozenset()) -> dict:
    return {
        name: to_jsonable_python(getattr(value, name))
        for name in type(value).model_fields
        if name not in skip
    }


def architecture(config: RunConfig) -> dict[str, Any]:
    """The plain-JSON settings :mod:`klab.model.export_loader` builds from."""

    def model_settings(model: Any) -> dict[str, Any]:
        return {
            **_settings(model, skip=frozenset({"layers"})),
            "memory_block_sites": list(model.memory_block_sites),
        }

    result: dict[str, Any] = {
        "model": model_settings(config.model),
        "attention": _settings(config.attention),
    }
    if config.model.layers:
        result["layers"] = [
            {"model": model_settings(model), "attention": _settings(attention)}
            for model, attention in config.model.layer_specs(config.attention)
        ]
    return result


def remote_code_tensors(
    config: RunConfig, weights: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Prefix the decoder's tensors; refuse a checkpoint the decoder cannot load."""
    from klab.model.transformer import DenseLM

    with torch.device("meta"):
        expected = set(DenseLM(config.model, config.attention).state_dict())
    if set(weights) != expected:
        raise ValueError(
            "checkpoint tensors do not match the exported decoder: "
            f"unexpected {sorted(set(weights) - expected)}, "
            f"missing {sorted(expected - set(weights))}"
        )
    return {
        f"model.{name}": torch.as_tensor(tensor)
        .detach()
        .to(torch.float32)
        .contiguous()
        .clone()
        for name, tensor in weights.items()
        if not (config.model.tie_embeddings and name == "output.weight")
    }


def remote_code_config(
    config: RunConfig,
    stop_ids: tuple[int, ...],
    identity: dict[str, str],
) -> dict[str, Any]:
    model = config.model
    eos = SPECIAL_TOKENS.index("<eos>")
    modeling = flat_name(MODELING_MODULE)
    return {
        "architectures": ["KlabForCausalLM"],
        "model_type": "klab",
        "auto_map": {
            "AutoConfig": f"{modeling}.KlabConfig",
            "AutoModelForCausalLM": f"{modeling}.KlabForCausalLM",
        },
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_dim,
        "num_hidden_layers": model.num_layers,
        "num_attention_heads": model.num_heads,
        "max_position_embeddings": model.max_seq_len,
        "tie_word_embeddings": model.tie_embeddings,
        "use_cache": True,
        "torch_dtype": "float32",
        "pad_token_id": SPECIAL_TOKENS.index("<pad>"),
        "bos_token_id": SPECIAL_TOKENS.index("<bos>"),
        "eos_token_id": [eos, *stop_ids] if stop_ids else eos,
        "klab_architecture": architecture(config),
        "klab_source": identity,
    }
