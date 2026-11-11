"""Full-state extension of native document groups and verified append-only data."""

from pathlib import Path

import numpy as np

from klab.config.models import RunConfig
from klab.data.packing import prepare_data
from klab.data.story_packing import MatchedStoryDataset
from klab.data.tokenizer import load_tokenizer
from klab.training.checkpoints import TrainingSnapshot
from klab.training.manifest import sha256_file


def verify_matched_extension(
    source: RunConfig,
    target: RunConfig,
    snapshot: TrainingSnapshot,
    parent_root: Path,
    *,
    append_data: bool,
    append_from_checkpoint: bool = False,
) -> dict[str, object]:
    """Require complete update caps; preserve prior tokens, groups and heldout bytes."""
    old, new = source.training.matched_stories, target.training.matched_stories
    if old is None or new is None:
        raise ValueError("matched budget extension requires matched target settings")
    tokenizer = load_tokenizer(parent_root / "tokenizer.json")
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("matched extension requires EOS")
    ids = np.load(parent_root / "data/train.npy", mmap_mode="r", allow_pickle=False)
    old_plan = MatchedStoryDataset(
        ids,
        source.training.seq_len,
        eos,
        layout=old.layout,
        targets_per_update=old.targets_per_update,
        target_group_transition=(
            old.target_group_transition.model_dump()
            if old.target_group_transition
            else None
        ),
        unpadded_documents=old.unpadded_documents,
        frozen_update_ends=old.frozen_update_ends,
    )
    consumed_steps = (
        snapshot.step if append_from_checkpoint else source.training.max_steps
    )
    consumed_tokens = (
        snapshot.tokens_seen if append_from_checkpoint else source.training.max_tokens
    )
    if not (
        0 < consumed_steps <= source.training.max_steps
        and 0 < consumed_tokens <= source.training.max_tokens
    ):
        raise ValueError("checkpoint prefix lies outside the declared source budget")
    if old_plan.targets_for_updates(consumed_steps) != consumed_tokens:
        raise ValueError("parent budget ends within a variable update")
    passes, group = divmod(consumed_steps, len(old_plan))
    expected_cursor = (passes, group) if group else (passes - 1, len(old_plan))
    if snapshot.cursor != expected_cursor:
        raise ValueError("parent cursor does not match completed document updates")
    metadata: dict[str, object] = {
        "variable_document_groups": True,
        "append_data": append_data,
        **({"append_from_checkpoint": True} if append_from_checkpoint else {}),
    }
    plan = old_plan
    if append_data:
        if (
            source.dataset.source != "local_text"
            or target.dataset.source != "local_text"
            or not old.unpadded_documents
            or not new.unpadded_documents
            or source.training.max_tokens != len(ids)
            or snapshot.cursor
            != (0, consumed_steps if append_from_checkpoint else len(old_plan))
        ):
            raise ValueError(
                "append-data requires a terminal single-pass whole-document parent"
            )
        data = prepare_data(target, load_tokenizer(target.tokenizer.path))
        prefix_tokens = consumed_tokens if append_from_checkpoint else len(ids)
        prefix_groups = consumed_steps if append_from_checkpoint else len(old_plan)
        if len(data.train) <= prefix_tokens:
            raise ValueError("append-data requires a larger prepared stream")
        for offset in range(0, prefix_tokens, 1_048_576):
            if not np.array_equal(
                ids[offset : min(offset + 1_048_576, prefix_tokens)],
                data.train[offset : min(offset + 1_048_576, prefix_tokens)],
            ):
                raise ValueError("append-data changed the trained token prefix")
        validation = np.load(
            parent_root / "data/validation.npy", mmap_mode="r", allow_pickle=False
        )
        if not np.array_equal(validation, data.validation):
            raise ValueError("append-data changed heldout tokens")
        plan = MatchedStoryDataset(
            data.train,
            target.training.seq_len,
            eos,
            layout=new.layout,
            targets_per_update=new.targets_per_update,
            target_group_transition=(
                new.target_group_transition.model_dump()
                if new.target_group_transition
                else None
            ),
            unpadded_documents=True,
            frozen_update_ends=new.frozen_update_ends,
        )
        if not np.array_equal(
            plan.group_offsets[: prefix_groups + 1],
            old_plan.group_offsets[: prefix_groups + 1],
        ):
            raise ValueError(
                "append-data changed prior update boundaries; freeze the parent terminal boundary"
            )
        if target.training.max_tokens != len(data.train):
            raise ValueError(
                "append-data budget must cover exactly the new single-pass stream"
            )
        metadata.update(
            parent_prefix_tokens=prefix_tokens,
            additional_unique_stream_tokens=len(data.train) - prefix_tokens,
            old_train_sha256=sha256_file(parent_root / "data/train.npy"),
            new_train_sha256=sha256_file(data.root / "train.npy"),
            validation_sha256=sha256_file(data.root / "validation.npy"),
            frozen_update_ends=list(new.frozen_update_ends),
        )
        if append_from_checkpoint:
            metadata.update(
                parent_consumed_steps=consumed_steps,
                parent_consumed_tokens=consumed_tokens,
                unconsumed_source_tokens_deferred=len(ids) - prefix_tokens,
                unconsumed_source_preserved_at=str(parent_root / "data/train.npy"),
            )
    if (
        plan.targets_for_updates(target.training.max_steps)
        != target.training.max_tokens
    ):
        raise ValueError("target budget is not a complete variable-update boundary")
    return metadata
