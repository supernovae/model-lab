"""Local exploratory checkpoint comparison; never a sealed Surface Review source."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from klab.evaluation.inference import InferenceRun

_ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}\Z")
_LABELS = "ABCD"


def checkpoint_selection(
    cells: Sequence[str],
) -> tuple[tuple[str, str, Path, Path], ...]:
    """Parse 2–4 generation directories without deriving identity from aliases.

    The returned fields are (display-only alias, run ID, runs directory, generation
    directory). A pointer, parent run directory, or symlink is not a generation.
    """
    if not 2 <= len(cells) <= 4:
        raise ValueError("select two to four checkpoint generations")
    selected = []
    aliases: set[str] = set()
    generations: set[Path] = set()
    for cell in cells:
        alias, separator, raw_path = cell.partition("=")
        if not separator or not _ALIAS.fullmatch(alias) or not raw_path:
            raise ValueError("each --cell must be ALIAS=GENERATION_PATH")
        if alias in aliases:
            raise ValueError("checkpoint aliases must be distinct")
        path = Path(raw_path).expanduser().absolute()
        # Refuse symlinked directory components, not just the last component.
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("checkpoint generation path must not contain symlinks")
        if not path.is_dir() or path.parent.name != "checkpoints":
            raise ValueError(
                "each cell must name an existing checkpoint generation directory"
            )
        run = path.parent.parent
        if not run.name or not (path / "manifest.json").is_file():
            raise ValueError("checkpoint generation lacks a run or manifest")
        if path in generations:
            raise ValueError("a checkpoint generation cannot appear twice")
        selected.append((alias, run.name, run.parent, path))
        aliases.add(alias)
        generations.add(path)
    return tuple(selected)


def verified_checkpoints(
    cells: Sequence[str], backend: str | None = None
) -> tuple[tuple[str, InferenceRun], ...]:
    """Load and verify every requested checkpoint before offering generation."""
    from klab.evaluation.inference import load_run

    verified = []
    identities: set[tuple[str, str]] = set()
    for alias, run_id, runs_dir, generation in checkpoint_selection(cells):
        loaded = load_run(run_id, runs_dir, generation, backend)
        identity = (str(loaded.run), str(loaded.identity["checkpoint_sha256"]))
        if identity in identities or loaded.identity["checkpoint_relative_path"] != str(
            generation.relative_to(loaded.run)
        ):
            raise ValueError("duplicate or mismatched verified checkpoint generation")
        identities.add(identity)
        verified.append((alias, loaded))
    return tuple(verified)


def check_common_context(
    prompt: str,
    checkpoints: Sequence[tuple[str, InferenceRun]],
    max_new_tokens: int,
    context_length: int | None = None,
) -> None:
    """Preflight *all* native tokenizers before generating from *any* model."""
    if not prompt.strip():
        raise ValueError("empty prompt ends the exploratory session")
    if max_new_tokens < 1:
        raise ValueError("completion length must be positive")
    for _, loaded in checkpoints:
        native = loaded.config.model.max_seq_len
        if context_length is not None and not 1 <= context_length <= native:
            raise ValueError("context cap must fit every model native context")
        context = native if context_length is None else context_length
        count = len(loaded.tokenizer.encode(prompt, add_special_tokens=False).ids)
        # generate_with_token_ids inserts BOS when tokenization is empty.
        if max(count, 1) + max_new_tokens > context:
            raise ValueError(
                "prompt and completion do not fit every model's native context"
            )


def anonymous_order(count: int, seed: int, ordinal: int) -> tuple[int, ...]:
    """Deterministic local permutation, independent of model generation RNG."""
    if count not in (2, 3, 4) or ordinal < 0:
        raise ValueError("invalid checkpoint count or prompt ordinal")
    digest = hashlib.sha256(f"surface-chat-v1:{seed}:{ordinal}".encode()).digest()
    order = list(range(count))
    random.Random(int.from_bytes(digest, "big")).shuffle(order)
    return tuple(order)


def generate_comparison(
    prompt: str,
    checkpoints: Sequence[tuple[str, InferenceRun]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    ordinal: int,
    stop_sequences: Sequence[str] = (),
    use_cache: bool = True,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Generate sequentially with the same policy and RNG seed on every model."""
    from klab.evaluation.generation import generate_with_token_ids

    if not 2 <= len(checkpoints) <= 4:
        raise ValueError("select two to four verified checkpoints")
    if temperature < 0 or top_k < 0:
        raise ValueError("temperature and top_k must be nonnegative")
    if any(not isinstance(stop, str) or not stop for stop in stop_sequences):
        raise ValueError("stop sequences must be nonempty strings")
    check_common_context(prompt, checkpoints, max_new_tokens, context_length)
    replies = []
    for alias, loaded in checkpoints:
        text, token_ids = generate_with_token_ids(
            loaded.model,
            loaded.tokenizer,
            prompt,
            context_length or loaded.config.model.max_seq_len,
            max_new_tokens,
            loaded.device,
            temperature=temperature,
            top_k=top_k,
            seed=seed + ordinal,
            strict_context=True,
            stop_sequences=stop_sequences,
            use_cache=use_cache,
            engine=loaded.engine,
        )
        replies.append(
            {
                "alias": alias,
                "identity": loaded.identity,
                "response": text[len(prompt) :],
                "token_ids": token_ids,
                "prompt_tokens": len(
                    loaded.tokenizer.encode(prompt, add_special_tokens=False).ids
                ),
                "context_length": context_length or loaded.config.model.max_seq_len,
            }
        )
    order = anonymous_order(len(checkpoints), seed, ordinal)
    return {
        "prompt": prompt,
        "prompt_format": "raw",
        "ordinal": ordinal,
        "policy": {
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "seed": seed + ordinal,
            "stop_sequences": list(stop_sequences),
            "use_cache": use_cache,
            "context_length": context_length,
            "strict_context": True,
        },
        "cards": [
            {"label": _LABELS[index], **replies[source]}
            for index, source in enumerate(order)
        ],
        "vote": None,
    }


def save_exploratory(
    work_dir: Path, seed: int, turns: Sequence[dict[str, Any]]
) -> Path:
    """Exclusively publish a plainly marked, non-importable local session."""
    root = work_dir / "surface-review" / "exploratory"
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("exploratory output path must not contain symlinks")
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "klab_exploratory_chat_v1",
        "eligibility": "EXPLORATORY_NOT_SEALED",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "turns": list(turns),
    }
    data = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path = root / f"chat-{secrets.token_hex(16)}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def render_chat(args: Any) -> None:
    """Render anonymous exploratory comparisons until explicit finish/reveal."""
    import streamlit as st

    st.title("Exploratory checkpoint chat")
    st.caption(
        "Local, exploratory responses only — not sealed judgments or research evidence."
    )
    signature = (tuple(args.cell), args.backend, args.seed)
    if st.session_state.get("surface_chat_signature") != signature:
        try:
            checkpoints = verified_checkpoints(args.cell, args.backend)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(
                f"Checkpoint verification failed ({type(error).__name__}); no generation was started."
            )
            return
        st.session_state.surface_chat_signature = signature
        st.session_state.surface_chat_checkpoints = checkpoints
        st.session_state.surface_chat_turns = []
        st.session_state.surface_chat_finished = False
        st.session_state.surface_chat_saved = None
    checkpoints = st.session_state.surface_chat_checkpoints
    turns = st.session_state.surface_chat_turns
    finished = st.session_state.surface_chat_finished
    st.caption(
        f"{len(checkpoints)} verified local checkpoints · session seed {args.seed} · {'finished' if finished else 'anonymous'}"
    )

    for turn in turns:
        st.subheader(f"Prompt {turn['ordinal'] + 1}")
        st.caption(
            "Raw completion prompt — passed unchanged; no chat template or history added."
        )
        st.code(turn["prompt"], language=None)
        st.json(turn["policy"])
        for card in turn["cards"]:
            st.markdown(f"**Response {card['label']}**")
            st.text(card["response"] or "(empty completion)")
        if not finished:
            labels = [card["label"] for card in turn["cards"]]
            choices = ["No vote", *labels, "tie", "neither", "cannot_tell"]
            current = turn["vote"] or "No vote"
            choice = st.selectbox(
                "Optional preference",
                choices,
                index=choices.index(current),
                key=f"surface_chat_vote_{turn['ordinal']}",
            )
            turn["vote"] = None if choice == "No vote" else choice

    if finished:
        st.success(
            "Session finished. Identity mapping revealed below; exploratory votes are not sealed reviews."
        )
        for turn in turns:
            st.subheader(f"Prompt {turn['ordinal'] + 1} identities")
            for card in turn["cards"]:
                identity = card["identity"]
                st.write(
                    f"{card['label']}: {card['alias']} — run {identity['run_id']}, checkpoint {identity['checkpoint_relative_path']}"
                )
                st.json(
                    {
                        "identity": identity,
                        "prompt_tokens": card["prompt_tokens"],
                        "context_length": card["context_length"],
                        "completion_token_ids": card["token_ids"],
                    }
                )
        if (
            turns
            and st.session_state.surface_chat_saved is None
            and st.button("Save exploratory responses and votes locally")
        ):
            try:
                from klab.workdir import resolve_work_dir

                st.session_state.surface_chat_saved = str(
                    save_exploratory(resolve_work_dir(args.work_dir), args.seed, turns)
                )
            except (OSError, ValueError) as error:
                st.error(f"Could not save exploratory session: {error}")
        if st.session_state.surface_chat_saved:
            st.success(
                f"Saved EXPLORATORY session: {st.session_state.surface_chat_saved}"
            )
        return

    if min(loaded.config.model.max_seq_len for _, loaded in checkpoints) < 2:
        st.error("Selected checkpoint context cannot fit a prompt and completion.")
        return
    st.caption(
        "Base checkpoints continue text; a chat interface does not teach instruction following."
    )
    with st.form("surface_chat_prompt", clear_on_submit=True):
        prompt = st.text_area(
            "Prompt (leave empty to finish)", key="surface_chat_input"
        )
        limit = min(
            256, *(loaded.config.model.max_seq_len - 1 for _, loaded in checkpoints)
        )
        max_new_tokens = st.number_input(
            "Maximum completion tokens",
            min_value=1,
            max_value=limit,
            value=min(96, limit),
        )
        policy = st.radio(
            "Common decoding policy", ("Greedy", "Sampled"), horizontal=True
        )
        temperature = st.number_input(
            "Sampling temperature",
            min_value=0.01,
            max_value=2.0,
            value=0.8,
            disabled=policy == "Greedy",
        )
        top_k = st.number_input(
            "Sampling top-k (0 = full vocabulary)",
            min_value=0,
            value=40,
            disabled=policy == "Greedy",
        )
        context_length = st.number_input(
            "Common context cap (prompt + completion tokens)",
            min_value=2,
            max_value=min(loaded.config.model.max_seq_len for _, loaded in checkpoints),
            value=min(loaded.config.model.max_seq_len for _, loaded in checkpoints),
        )
        stops_json = st.text_input("Stop strings (JSON list)", value="[]")
        use_cache = st.checkbox("Use request-local KV cache when supported", value=True)
        submitted = st.form_submit_button("Generate anonymously")
    if submitted:
        if not prompt.strip():
            st.session_state.surface_chat_finished = True
            st.rerun()
        try:
            stops = json.loads(stops_json)
            if not isinstance(stops, list):
                raise TypeError("stop strings must be a JSON list")
            turn = generate_comparison(
                prompt,
                checkpoints,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature) if policy == "Sampled" else 0.0,
                top_k=int(top_k) if policy == "Sampled" else 0,
                seed=args.seed,
                ordinal=len(turns),
                stop_sequences=stops,
                use_cache=use_cache,
                context_length=int(context_length),
            )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            st.error(
                f"Generation failed ({type(error).__name__}); no partial comparison was recorded."
            )
        else:
            turns.append(turn)
            st.rerun()
    if st.button("Finish and reveal identities"):
        st.session_state.surface_chat_finished = True
        st.rerun()
