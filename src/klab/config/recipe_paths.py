"""Typed RunConfig path binding for explicitly rooted declarations."""

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from klab.config.models import RunConfig
from klab.declaration_context import RecipeContext, canonical_reference, safe_path

_INPUT_FIELDS = frozenset(
    {
        "tokenizer.path",
        "tokenizer.origin.mixture_declaration_path",
        "tokenizer.origin.dataset.train_path",
        "tokenizer.origin.dataset.validation_path",
        "tokenizer.origin.dataset.source_manifest_path",
        "tokenizer.origin.dataset.cache_dir",
        "model.memory_package_path",
        "dataset.train_path",
        "dataset.validation_path",
        "dataset.source_manifest_path",
        "dataset.allocation_manifest_path",
        "dataset.corpus_release_path",
        "dataset.corpus_export_path",
        "training.portability_manifest_path",
    }
)
_OUTPUT_FIELDS = frozenset({"dataset.cache_dir", "logging.root_dir"})


def _paths(value: BaseModel, prefix: str = "") -> dict[str, Path]:
    """Traverse native model-owned Path values, never arbitrary dict keys."""
    paths = {}
    for name in type(value).model_fields:
        item = getattr(value, name)
        field = f"{prefix}.{name}" if prefix else name
        if isinstance(item, Path):
            paths[field] = item
        elif isinstance(item, BaseModel):
            paths.update(_paths(item, field))
        elif name == "panels" and isinstance(item, tuple):
            for index, panel in enumerate(item):
                paths.update(_paths(panel, f"{field}.{index}"))
    return paths


def _bind_config(
    config: RunConfig,
    source: Path,
    context: RecipeContext,
    *,
    bindings: Mapping[str, Path] | None = None,
) -> RunConfig:
    raw = config.model_dump(mode="python")
    bindings = bindings or {}
    if set(bindings) - _paths(config).keys():
        raise ValueError("binding does not name a declared native config path")
    for field, value in _paths(config).items():
        panel_input = field.startswith("evaluation.panels.") and field.endswith(
            ".prepared_path"
        )
        if field not in _INPUT_FIELDS | _OUTPUT_FIELDS and not panel_input:
            raise ValueError(
                f"unsupported portable path field: {field}; native loaders are unchanged"
            )
        canonical_reference(value.as_posix())
        path = bindings.get(field, safe_path(source.parent, value.as_posix()))
        if field not in bindings and not path.is_relative_to(context.root):
            raise ValueError(f"config path escapes recipe: {field}")
        if panel_input:
            raw["evaluation"]["panels"][int(field.split(".")[2])]["prepared_path"] = (
                path
            )
        else:
            parts = field.split(".")
            target = raw
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = path
    return RunConfig.model_validate(raw)
