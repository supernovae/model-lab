"""Verified same-experiment continuation versus explicit weight promotion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from klab.canonical import canonical_digest
from klab.config.models import RunConfig
from klab.runtime import RuntimeInfo
from klab.training.checkpoints import (
    CheckpointManager,
    LineageBest,
    TrainingSnapshot,
)
from klab.training.manifest import (
    architecture_sha256,
    config_sha256,
    read_manifest,
    sha256_file,
)


@dataclass(frozen=True)
class Continuation:
    kind: Literal["FRESH", "RESUMED", "PROMOTED"] = "FRESH"
    snapshot: TrainingSnapshot | None = None
    source_run: Path | None = None
    parent_run_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    decisions: tuple[dict[str, object], ...] = ()
    lineage_best: LineageBest | None = None


def _resume_settings(config: RunConfig) -> str:
    settings = config.model_dump(mode="json")
    for field in ("name", "logging", "checkpoint"):
        settings.pop(field)
    settings["runtime"].pop("device_index")
    return config_sha256(settings)


def runtime_optimization_transition(
    source: RunConfig, target: RunConfig, checkpoint_sha256: str
) -> dict[str, object]:
    """Admit only explicitly requested execution changes with a fixed effective batch."""
    if source.runtime.engine != "pytorch" or target.runtime.engine != "pytorch":
        raise ValueError("runtime optimization transition requires PyTorch")
    if source.attention_kinds != {"dense"} or target.attention_kinds != {"dense"}:
        raise ValueError("runtime optimization transition requires dense attention")
    source_matched = source.training.matched_stories
    target_matched = target.training.matched_stories
    document_groups = (
        source_matched is not None
        and target_matched is not None
        and source_matched.unpadded_documents
        and target_matched.unpadded_documents
        and source_matched.layout == target_matched.layout == "story"
    )
    # Grouping/data/budget/cursor remain covered by the complete settings
    # comparison below and native checkpoint/data verification.
    if document_groups and (
        source.training.gradient_accumulation != target.training.gradient_accumulation
    ):
        raise ValueError("document batching must preserve declared accumulation")
    if not document_groups and (
        source.training.micro_batch_size * source.training.gradient_accumulation
        != target.training.micro_batch_size * target.training.gradient_accumulation
    ):
        raise ValueError(
            "runtime optimization transition must preserve effective batch"
        )
    if not source.evaluation.panels and (
        (source.evaluation.batch_size or source.training.micro_batch_size)
        != (target.evaluation.batch_size or target.training.micro_batch_size)
    ):
        raise ValueError(
            "runtime optimization transition must preserve ordinary validation batch size"
        )
    original = source.model_dump(mode="json")
    comparable = target.model_dump(mode="json")
    paths = (
        ("attention", "implementation"),
        ("training", "deterministic"),
        ("training", "micro_batch_size"),
        ("training", "gradient_accumulation"),
        ("evaluation", "batch_size"),
        ("runtime", "memory", "estimate_policy"),
        ("runtime", "memory", "activation_checkpointing", "enabled"),
        ("runtime", "compile_transformer_blocks"),
        ("runtime", "compile_dynamic_shapes"),
        ("runtime", "compile_output_head_loss"),
    )
    defaults = {
        ("attention", "implementation"): "reference",
        ("evaluation", "batch_size"): None,
        ("runtime", "memory", "estimate_policy"): "enforce",
        ("training", "matched_stories", "document_batching"): "separate",
        ("training", "matched_stories", "padding_multiple"): 128,
    }
    if document_groups:
        paths += (
            ("training", "matched_stories", "document_batching"),
            ("training", "matched_stories", "padding_multiple"),
        )
    changes: list[dict[str, object]] = []
    for path in paths:
        old: dict[str, Any] = original
        new: dict[str, Any] = comparable
        for part in path[:-1]:
            old, new = old[part], new[part]
        key = path[-1]
        # Default-valued fields are intentionally omitted from historical identities.
        default = defaults.get(path, False)
        previous, requested = old.get(key, default), new.get(key, default)
        if previous != requested:
            if path == ("runtime", "memory", "estimate_policy") and (
                previous,
                requested,
            ) != ("enforce", "observe"):
                raise ValueError(
                    "runtime estimate policy transition permits only enforce-to-observe"
                )
            changes.append(
                {"field": ".".join(path), "previous": previous, "requested": requested}
            )
        if key in old:
            new[key] = old[key]
        else:
            new.pop(key, None)
    if _resume_settings(source) != _resume_settings(
        RunConfig.model_validate(comparable)
    ):
        raise ValueError(
            "runtime optimization transition changes scientific or unsupported settings"
        )
    return {
        "kind": "runtime_optimization_transition",
        "resume_level": "best_effort",
        "parent_checkpoint_sha256": checkpoint_sha256,
        "changes": changes,
        "reason": "explicit --allow-runtime-optimization; numerical trajectory may differ",
        **(
            {
                "effective_batch_contract": "original-document-update-groups",
                "document_diagnostics_version": 2
                if target_matched.document_batching == "padded"
                else 1,
            }
            if document_groups
            else {}
        ),
    }


def target_group_transition(
    source: RunConfig, target: RunConfig, snapshot: TrainingSnapshot
) -> dict[str, object]:
    """Explicit scientific transition; retain Adam/RNG and the already reached LR floor."""
    old, new = source.training.matched_stories, target.training.matched_stories
    policy = new.target_group_transition if new else None
    if (
        old is None
        or new is None
        or policy is None
        or old.target_group_transition is not None
        or source.runtime.engine != "pytorch"
        or source.optimizer.name != "adamw"
        or source.optimizer.decay_steps is None
        or snapshot.step < source.optimizer.decay_steps
        or policy.after_tokens != snapshot.tokens_seen
        or policy.previous_targets_per_update != old.targets_per_update
        or policy.parent_checkpoint_sha256 != snapshot.checkpoint_sha256
        or snapshot.cursor != (0, snapshot.step)
        or new.targets_per_update <= old.targets_per_update
        or target.training.max_steps <= snapshot.step
        or target.training.max_steps >= source.training.max_steps
        or new.document_batching != "padded"
        or new.max_padded_slots is None
        or new.max_attention_cells is None
    ):
        raise ValueError(
            "regrouping requires a first-pass complete checkpoint at the existing LR floor and explicit bounded larger groups"
        )
    comparable = target.model_dump(mode="json")
    comparable["training"]["max_steps"] = source.training.max_steps
    settings = comparable["training"]["matched_stories"]
    settings["targets_per_update"] = old.targets_per_update
    settings.pop("target_group_transition", None)
    for field in ("max_padded_slots", "max_attention_cells"):
        settings.pop(field, None)
        previous = getattr(old, field)
        if previous is not None:
            settings[field] = previous
    if _resume_settings(source) != _resume_settings(
        RunConfig.model_validate(comparable)
    ):
        raise ValueError(
            "regrouping must preserve data, curriculum, optimizer, runtime and evaluation settings"
        )
    from klab.training.optimizer import schedule_payload

    if snapshot.schedule != schedule_payload(source, snapshot.step):
        raise ValueError(
            "regrouping checkpoint schedule differs from its source configuration"
        )
    return {
        "kind": "target_group_transition",
        **policy.model_dump(),
        "targets_per_update": new.targets_per_update,
        "old_max_steps": source.training.max_steps,
        "new_max_steps": target.training.max_steps,
        "optimizer_state": "preserved; one Adam step per new effective group",
        "schedule": "existing absolute-step floor; no warmup restart or LR scaling",
        "cursor": "preserve historical group prefix; verify against prepared stream",
        "resume_level": "changed_optimizer_trajectory",
    }


def verify_regroup_cursor(dataset, snapshot: TrainingSnapshot) -> None:
    """Bind unchanged group-index cursor to exactly the next unconsumed document."""
    import numpy as np

    from klab.data.story_packing import MatchedStoryDataset

    source = RunConfig.model_validate(snapshot.config)
    old = source.training.matched_stories
    previous = MatchedStoryDataset(
        dataset.ids,
        dataset.seq_len,
        dataset.eos,
        layout=old.layout,
        targets_per_update=old.targets_per_update,
        unpadded_documents=True,
        frozen_update_ends=old.frozen_update_ends,
    )
    epoch, group = snapshot.cursor
    if (
        epoch != 0
        or group != snapshot.step
        or group >= len(previous)
        or int(previous.group_offsets[group]) != snapshot.tokens_seen
        or not np.array_equal(
            previous.group_offsets[: group + 1], dataset.group_offsets[: group + 1]
        )
    ):
        raise ValueError(
            "regrouping changed the consumed prefix or next-document cursor"
        )


def _extension_config(
    source: RunConfig,
    target: RunConfig,
    *,
    parent_tokenizer: Path | None = None,
    append_data: bool = False,
) -> RunConfig:
    """Compare budget-independent settings; authenticate relocated tokenizer bytes."""
    comparable = target.model_dump(mode="json")
    comparable["training"]["max_steps"] = source.training.max_steps
    comparable["training"]["max_tokens"] = source.training.max_tokens
    # Retention terminal/milestone bounds must describe these comparison caps.
    # Checkpoint bookkeeping is excluded from scientific resume settings.
    comparable["checkpoint"] = source.checkpoint.model_dump(mode="json")
    comparable["optimizer"]["decay_steps"] = source.optimizer.decay_steps
    if append_data:
        comparable["dataset"] = source.dataset.model_dump(mode="json")
        comparable["training"]["matched_stories"]["frozen_update_ends"] = list(
            source.training.matched_stories.frozen_update_ends
        )
    if parent_tokenizer is not None:
        manifest = read_manifest(parent_tokenizer.parent / "manifest.json")
        record = next(
            (
                entry
                for entry in manifest.get("artifacts", [])
                if entry.get("relative_path") == parent_tokenizer.name
            ),
            None,
        )
        parent_sha256 = sha256_file(parent_tokenizer)
        if (
            record is None
            or parent_tokenizer.is_symlink()
            or parent_tokenizer.stat().st_size != record["size_bytes"]
            or parent_sha256 != record["sha256"]
        ):
            raise ValueError(
                "parent tokenizer differs from its bound run artifact inventory"
            )
        if parent_sha256 != sha256_file(target.tokenizer.path):
            raise ValueError("budget extension requires matching tokenizer identity")
        comparable["tokenizer"]["path"] = str(source.tokenizer.path)
    return RunConfig.model_validate(comparable)


def _extension_settings(
    source: RunConfig,
    target: RunConfig,
    *,
    parent_tokenizer: Path | None = None,
    append_data: bool = False,
) -> str:
    return _resume_settings(
        _extension_config(
            source, target, parent_tokenizer=parent_tokenizer, append_data=append_data
        )
    )


def validate_budget_extension(
    source: RunConfig,
    target: RunConfig,
    snapshot: TrainingSnapshot,
    *,
    parent_tokenizer: Path | None = None,
    parent_root: Path | None = None,
    append_data: bool = False,
    append_from_checkpoint: bool = False,
    allow_runtime_optimization: bool = False,
) -> dict[str, object]:
    """Validate the sole scientific difference allowed for full-state extension."""
    old = source.training
    new = target.training
    targets_per_step = old.seq_len * old.micro_batch_size * old.gradient_accumulation
    variable = old.matched_stories is not None
    decay_horizon = getattr(source.optimizer, "decay_steps", None) or old.max_steps
    if append_from_checkpoint and not append_data:
        raise ValueError("append-from-checkpoint requires append-data")
    prior_steps = snapshot.step if append_from_checkpoint else old.max_steps
    prior_tokens = snapshot.tokens_seen if append_from_checkpoint else old.max_tokens
    if (
        source.optimizer.name != "adamw"
        or target.optimizer.name != "adamw"
        or target.optimizer.decay_steps != decay_horizon
        or snapshot.schedule.get("kind")
        != (
            "warmup_cosine_floor_v1"
            if source.optimizer.decay_steps is not None
            else "warmup_cosine_v1"
        )
        or (not append_from_checkpoint and snapshot.step != old.max_steps)
        or (not append_from_checkpoint and snapshot.tokens_seen != old.max_tokens)
        or (append_from_checkpoint and not variable)
        or (not variable and old.max_tokens != old.max_steps * targets_per_step)
        or (not variable and new.max_tokens != new.max_steps * targets_per_step)
        or new.max_steps <= prior_steps
        or new.max_tokens <= prior_tokens
    ):
        raise ValueError(
            "budget extension requires a terminal, whole-update AdamW parent and strictly larger whole-update caps"
        )
    supply = {}
    if variable:
        if parent_root is None:
            raise ValueError("matched budget extension requires verified parent data")
        from klab.training.data_extension import verify_matched_extension

        supply = verify_matched_extension(
            source,
            target,
            snapshot,
            parent_root,
            append_data=append_data,
            append_from_checkpoint=append_from_checkpoint,
        )
    elif append_data:
        raise ValueError("append-data requires matched whole-document training")
    comparable = _extension_config(
        source,
        target,
        parent_tokenizer=parent_tokenizer,
        append_data=append_data,
    )
    optimization = None
    if allow_runtime_optimization:
        optimization = runtime_optimization_transition(
            source, comparable, snapshot.checkpoint_sha256
        )
    elif _resume_settings(comparable) != _resume_settings(source):
        raise ValueError(
            "budget extension changes settings other than the budget and decay horizon"
        )
    return {
        "kind": "budget_extension",
        "parent_checkpoint_sha256": snapshot.checkpoint_sha256,
        "old_max_steps": old.max_steps,
        "old_max_tokens": old.max_tokens,
        "new_max_steps": new.max_steps,
        "new_max_tokens": new.max_tokens,
        "decay_steps": target.optimizer.decay_steps,
        **({"runtime_optimization": optimization} if optimization is not None else {}),
        **supply,
    }


def _verify_artifacts(root: Path, manifest: dict[str, Any], *, resume: bool) -> None:
    records = manifest.get("artifacts", [])
    names: set[str] = set()
    required = {"tokenizer.json"}
    if resume:
        required.update({"data/manifest.json", "data/train.npy", "data/validation.npy"})
    for record in records:
        name = record["relative_path"]
        if name in names:
            raise ValueError(f"duplicate run artifact: {name}")
        names.add(name)
        if (
            not resume
            and name != "tokenizer.json"
            and not name.startswith("portable_package")
        ):
            continue
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe run artifact: {name}")
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"symlinked run artifact: {name}")
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"run artifact integrity failure: {name}")
    if not required <= names:
        raise ValueError(
            f"run lacks bound continuation artifacts: {sorted(required - names)}; "
            "use compatible weight promotion for historical incomplete state"
        )


def _package_identity(path: Path) -> dict[str, str]:
    if path.is_file():
        return {".": sha256_file(path)}
    if not path.is_dir():
        raise ValueError(f"portable package is missing: {path}")
    inventory = {}
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"symlinked portable package member: {member}")
        if member.is_file():
            inventory[str(member.relative_to(path))] = sha256_file(member)
    return inventory


def load_continuation(
    config: RunConfig,
    runtime: RuntimeInfo,
    current_source: dict[str, object],
    *,
    resume: Path | None,
    extend_budget: Path | None = None,
    promote: Path | None,
    recover: Path | None,
    allow_runtime_drift: bool,
    allow_runtime_optimization: bool = False,
    add_ngram_memory: bool = False,
    extend_context: bool = False,
    append_data: bool = False,
    append_from_checkpoint: bool = False,
) -> Continuation:
    if append_from_checkpoint and not append_data:
        raise ValueError("append-from-checkpoint requires explicit append-data")
    if append_data and extend_budget is None:
        raise ValueError("append-data requires explicit extend-budget")
    if extend_context and not add_ngram_memory:
        raise ValueError("extend-context requires explicit add-ngram-memory promotion")
    if add_ngram_memory:
        if promote is None or any(
            path is not None for path in (resume, recover, extend_budget)
        ):
            raise ValueError(
                "add-ngram-memory requires only --promote, not resume or recovery"
            )
        if (
            promote.is_symlink()
            or not promote.is_dir()
            or promote.parent.name != "checkpoints"
            or not promote.name.startswith("step_")
            or "_gen_" not in promote.name
            or promote.resolve() != promote.parent.resolve() / promote.name
        ):
            raise ValueError(
                "add-ngram-memory requires an immutable checkpoint generation directory"
            )
    if allow_runtime_optimization and (
        (resume is None and recover is None and extend_budget is None)
        or promote is not None
    ):
        raise ValueError(
            "runtime optimization requires an explicit full-state resume, recovery or budget extension"
        )
    selected = extend_budget or resume or promote
    if selected is None and recover is None:
        return Continuation()
    if extend_budget is not None:
        if any(path is not None for path in (resume, promote, recover)):
            raise ValueError(
                "extend-budget, resume, promote, and recover are mutually exclusive"
            )
        if (
            extend_budget.is_symlink()
            or not extend_budget.is_dir()
            or extend_budget.parent.name != "checkpoints"
            or not extend_budget.name.startswith("step_")
            or "_gen_" not in extend_budget.name
            or extend_budget.resolve()
            != (extend_budget.parent.resolve() / extend_budget.name)
        ):
            raise ValueError(
                "extend-budget requires an immutable full-state generation directory"
            )
    if recover is not None:
        root = recover.resolve()
    else:
        assert selected is not None
        root = selected.parent.parent.resolve()
    manifest = read_manifest(root / "manifest.json")
    manifest_digest = canonical_digest(manifest)
    manager = CheckpointManager(root, manifest_sha256=manifest_digest)
    decisions: list[dict[str, object]] = []
    with manager.writer_lease():
        if recover is not None:
            recovery = manager.reconcile()
            if recovery.record is None:
                raise ValueError(
                    f"no valid checkpoint for recovery: {recovery.rejected}"
                )
            selected = root / "checkpoints" / recovery.record.relative_path
            decisions.append(
                {
                    "kind": "recovery",
                    "checkpoint_sha256": recovery.record.manifest_sha256,
                    "rejected": [list(report.errors) for report in recovery.rejected],
                    "reason": "explicit recovery from verified finalized generations",
                }
            )
        assert selected is not None
        snapshot = manager.load(
            selected, "promote" if promote is not None else "resume"
        )
        assert snapshot.checkpoint_sha256 is not None
        # Promotion verifies the source binding without loading its optimizer state.
        binding = manager.verify(
            selected, expected_manifest=manifest_digest, require_training_state=False
        )
        if not binding.valid:
            raise ValueError(f"checkpoint run binding failed: {binding.errors}")
        _verify_artifacts(root, manifest, resume=promote is None)
        source_config = RunConfig.model_validate(manifest["effective_config"])
        if not (allow_runtime_optimization or add_ngram_memory) and architecture_sha256(
            source_config.model_dump(mode="json")
        ) != architecture_sha256(config.model_dump(mode="json")):
            raise ValueError(
                "continuation configuration differs in architecture semantics"
            )
        if promote is not None:
            if sha256_file(root / "tokenizer.json") != sha256_file(
                config.tokenizer.path
            ):
                raise ValueError("promotion requires matching tokenizer identity")
            if add_ngram_memory:
                from klab.training.memory_warmstart import migrate_dense_to_ngram
                from klab.training.mlx_checkpoints import strict_json

                source_binding = manager.verify(
                    selected,
                    expected_manifest=manifest_digest,
                    require_training_state=False,
                    expected_config=source_config,
                )
                if not source_binding.valid:
                    raise ValueError(
                        f"source checkpoint architecture binding failed: {source_binding.errors}"
                    )
                checkpoint_manifest = strict_json(selected / "manifest.json")
                if (
                    checkpoint_manifest.get("sha256") != snapshot.checkpoint_sha256
                    or canonical_digest(
                        {
                            key: value
                            for key, value in checkpoint_manifest.items()
                            if key != "sha256"
                        }
                    )
                    != snapshot.checkpoint_sha256
                    or checkpoint_manifest.get("config_sha256")
                    != config_sha256(source_config.model_dump(mode="json"))
                ):
                    raise ValueError(
                        "source checkpoint scientific identity differs from its run manifest"
                    )
                snapshot.model, migration = migrate_dense_to_ngram(
                    source_config,
                    config,
                    snapshot.model,
                    source_tokenizer_sha256=sha256_file(root / "tokenizer.json"),
                    target_tokenizer_sha256=sha256_file(config.tokenizer.path),
                    source_checkpoint_sha256=snapshot.checkpoint_sha256,
                    extend_context=extend_context,
                )
                migration.update(
                    source_run_manifest_sha256=manifest_digest,
                    source_step=checkpoint_manifest["step"],
                    source_tokens_seen=checkpoint_manifest["tokens_seen"],
                )
                decisions.append(migration)
            if config.model.memory_package_path is not None and _package_identity(
                root / "portable_package"
            ) != _package_identity(config.model.memory_package_path):
                raise ValueError(
                    "promotion requires matching portable package identity"
                )
            return Continuation(
                "PROMOTED",
                snapshot,
                root,
                manifest["run_id"],
                snapshot.checkpoint_sha256,
                tuple(decisions),
                None
                if add_ngram_memory
                else manager.lineage_best_for_child(
                    str(manifest["run_id"]), snapshot.checkpoint_sha256
                ),
            )
        saved_config = RunConfig.model_validate(snapshot.config)
        if config_sha256(snapshot.config) != config_sha256(
            source_config.model_dump(mode="json")
        ) or snapshot.source_identity_sha256 != manifest.get("source_identity", {}).get(
            "sha256"
        ):
            raise ValueError(
                "checkpoint scientific identity differs from its source run manifest"
            )
        if extend_budget is not None:
            extension = validate_budget_extension(
                saved_config,
                config,
                snapshot,
                parent_tokenizer=root / "tokenizer.json",
                parent_root=root,
                append_data=append_data,
                append_from_checkpoint=append_from_checkpoint,
                allow_runtime_optimization=allow_runtime_optimization,
            )
            optimization = extension.pop("runtime_optimization", None)
            if optimization is not None:
                decisions.append(optimization)
        elif (
            config.training.matched_stories is not None
            and config.training.matched_stories.target_group_transition is not None
            and saved_config.training.matched_stories is not None
            and saved_config.training.matched_stories.target_group_transition is None
        ):
            decisions.append(target_group_transition(saved_config, config, snapshot))
        elif allow_runtime_optimization:
            decisions.append(
                runtime_optimization_transition(
                    saved_config, config, snapshot.checkpoint_sha256
                )
            )
        else:
            if _resume_settings(saved_config) != _resume_settings(config):
                raise ValueError(
                    "resume configuration differs from checkpoint; use promotion for changed scientific settings"
                )
        if snapshot.engine != runtime.engine or snapshot.backend != runtime.backend:
            raise ValueError("full resume requires the same engine and backend")
        drift: list[dict[str, object]] = []
        old_source = manifest.get("source_identity", {}).get("sha256")
        if old_source != current_source["sha256"]:
            drift.append(
                {
                    "field": "source_identity_sha256",
                    "requested": old_source,
                    "effective": current_source["sha256"],
                }
            )
        old_runtime = manifest["runtime"]
        for field in (
            "framework_version",
            "runtime_version",
            "driver_version",
            "device_name",
            "device_index",
            "physical_device_id",
            "os",
        ):
            previous, actual = old_runtime.get(field), getattr(runtime, field)
            if previous != actual:
                drift.append(
                    {
                        "field": f"runtime.{field}",
                        "requested": previous,
                        "effective": actual,
                    }
                )
        if drift and not allow_runtime_drift:
            raise ValueError(
                "source or runtime identity differs; --allow-runtime-drift is required "
                f"for best-effort same-backend resume: {drift}"
            )
        if drift:
            decisions.append(
                {
                    "kind": "runtime_drift",
                    "resume_level": "best_effort",
                    "changes": drift,
                    "reason": "explicit --allow-runtime-drift",
                }
            )
        if extend_budget is not None:
            decisions.append({**extension, "source_drift": drift})
        return Continuation(
            "RESUMED",
            snapshot,
            root,
            manifest["run_id"],
            snapshot.checkpoint_sha256,
            tuple(decisions),
            manager.lineage_best_for_child(
                str(manifest["run_id"]), snapshot.checkpoint_sha256
            ),
        )
