"""CPU-only sampling of selected verified saved checkpoints."""

import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
import torch

from sparselab.evaluation.generation import generate
from sparselab.evaluation.inference import load_run

# Streamlit uses the matching inference runtime; catalog code lives beside this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_catalog import SnapshotOwner, catalog, resolve_candidate

CONFIG = json.loads(Path(os.environ["SPARSELAB_SNAPSHOT_SERVICE_CONFIG"]).read_text())
RUNS = Path(CONFIG["store"])


def validate_controls(prompt, count, temperature, top_k, seed):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Enter a nonempty prompt of at most 4,000 characters.")
    for value, lower, upper, label in (
        (count, 8, 128, "Output length"),
        (top_k, 0, 200, "Top-k"),
        (seed, 0, 2147483647, "Seed"),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{label} is outside the supported range.")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("Temperature must be finite and between 0 and 2.")


@st.cache_resource
def checkpoint():
    torch.set_num_threads(2)
    return SnapshotOwner()


def load_selected(selected):
    if selected not in catalog(RUNS):
        raise ValueError("Selected checkpoint is no longer in the verified catalog")
    location = resolve_candidate(RUNS, selected)
    return load_run(
        selected.run,
        RUNS,
        str(location),
        "cpu",
        expected_checkpoint_sha256=selected.digest,
    )


def render():
    st.set_page_config(
        page_title="DevMind snapshot — text completion", page_icon="🔬", layout="wide"
    )
    st.title("Try a saved snapshot")
    if st.query_params.get("store", str(RUNS)) != str(RUNS):
        st.error("This sampler only serves its configured run store.")
        st.stop()
    if st.button("Refresh available snapshots"):
        st.session_state.pop("snapshot_choices", None)
    try:
        if "snapshot_choices" not in st.session_state:
            st.session_state["snapshot_choices"] = catalog(RUNS)
        choices = st.session_state["snapshot_choices"]
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Checkpoint catalog unavailable: {error}")
        st.stop()
    if not choices:
        st.info("No verified complete saved checkpoints are recorded.")
        st.stop()
    requested_run = st.query_params.get("run")
    requested_checkpoint = st.query_params.get("checkpoint")
    matches = [
        item
        for item in choices
        if item.run == requested_run and item.checkpoint == requested_checkpoint
    ]
    if (requested_run or requested_checkpoint) and not matches:
        st.error(
            "Linked checkpoint is missing or not recorded verified; no substitute selected."
        )
        st.stop()
    initial = matches[0] if matches else choices[0]
    selected = st.selectbox(
        "Snapshot to query",
        choices,
        index=choices.index(initial),
        format_func=lambda item: item.label,
    )
    expected = {
        "store": str(RUNS),
        "run": selected.run,
        "checkpoint": selected.checkpoint,
    }
    if st.session_state.get("selected_identity") != selected.key:
        st.session_state["selected_identity"] = selected.key
        st.session_state.pop("completion_result", None)
    st.caption(f"Selected: {selected.label}")
    st.caption("Newest recorded complete snapshot: " + choices[0].label)
    st.info(
        "This base model continues text, not reliable instruction-following chat. A complete saved checkpoint can belong to a still-running training run."
    )
    st.caption(
        "CPU only · 2 threads · one shared model · 10 GiB service limit. Verification/loading may take minutes; other requests receive busy. No automatic weight changes during a response."
    )
    owner = checkpoint()
    state, identity = owner.status
    status_display = st.empty()
    status_display.caption(f"Sampler status at page render: {state} · {identity}")
    st.markdown(
        "[Link to selected snapshot](http://127.0.0.1:8501/generate?"
        + urlencode(expected)
        + ")"
    )
    st.caption(
        "Generate verifies and loads this selection if necessary, replacing the previous CPU model. Browsing alone does not load weights."
    )
    preset = st.selectbox(
        "Prompt starting point",
        ["Blank / custom", "Technical explanation", "Code", "Story (explicit choice)"],
    )
    presets = {
        "Blank / custom": "",
        "Technical explanation": "The purpose of a computer program is",
        "Code": "def add(a, b):\n",
        "Story (explicit choice)": "The traveler arrived at the station",
    }
    if st.session_state.get("prompt_preset") != preset:
        st.session_state["prompt_preset"] = preset
        st.session_state["sample_prompt"] = presets[preset]
        st.session_state.pop("completion_result", None)
    with st.form("completion"):
        prompt = st.text_area(
            "Text to continue",
            key="sample_prompt",
            placeholder="Enter any text you want this checkpoint to continue…",
            max_chars=4000,
            height=130,
        )
        count = st.slider(
            "Maximum new tokens", min_value=8, max_value=128, value=32, step=8
        )
        columns = st.columns(3)
        temperature = columns[0].number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="0 is deterministic greedy decoding; higher values sample more widely.",
        )
        top_k = columns[1].number_input(
            "Top-k",
            min_value=0,
            max_value=200,
            value=0,
            step=1,
            help="0 disables top-k filtering. Ignored by greedy decoding.",
        )
        seed = columns[2].number_input(
            "Seed", min_value=0, max_value=2147483647, value=0, step=1
        )
        submitted = st.form_submit_button("Generate completion", type="primary")
    if submitted:
        st.session_state.pop("completion_result", None)
        if not prompt.strip():
            st.warning("Enter a short text prefix to continue.")
        else:
            try:
                with st.spinner(
                    "Verifying and loading the checkpoint on first use, then generating on CPU…"
                ):
                    validate_controls(prompt, count, temperature, top_k, seed)

                    def complete(loaded):
                        if (
                            len(
                                loaded.tokenizer.encode(
                                    prompt, add_special_tokens=False
                                ).ids
                            )
                            + count
                            > 1024
                        ):
                            raise ValueError(
                                "Prompt plus completion must fit the 1,024-token local preview limit."
                            )
                        result = generate(
                            loaded.model,
                            loaded.tokenizer,
                            prompt,
                            1024,
                            count,
                            loaded.device,
                            temperature=temperature,
                            top_k=top_k,
                            seed=seed,
                            strict_context=True,
                            engine=loaded.engine,
                        )
                        return (
                            prompt,
                            result[len(prompt) :],
                            loaded.identity["tokenizer_sha256"],
                            {
                                "temperature": temperature,
                                "top_k": top_k,
                                "seed": seed,
                                "max_new_tokens": count,
                                "run": selected.run,
                                "checkpoint": selected.checkpoint,
                                "checkpoint_sha256": selected.digest,
                            },
                        )

                    st.session_state.completion_result = owner.execute(
                        selected, load_selected, complete
                    )
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                st.error(f"Completion unavailable: {error}")
            state, identity = owner.status
            status_display.caption(
                f"Sampler status after request: {state} · {identity}"
            )
    if "completion_result" in st.session_state:
        prior_prompt, reply, tokenizer, settings = st.session_state.completion_result
        st.subheader("Generated continuation")
        st.caption(
            "Output may be incorrect, repetitive, or end mid-sentence at the token limit."
        )
        st.caption(f"Settings used: {settings}")
        st.text(reply or "(No visible continuation generated.)")
        with st.expander("Exact input and verified identity"):
            st.text(prior_prompt)
            st.write(
                {
                    "checkpoint": settings["checkpoint"],
                    "checkpoint_sha256": settings["checkpoint_sha256"],
                    "tokenizer_sha256": tokenizer,
                    "backend": "cpu",
                }
            )
    st.caption(
        "Prompts and replies stay in this browser session and are not saved to training data. The verified tokenizer, weights and Engram state load together; opening this page never changes them."
    )


if __name__ == "__main__":
    render()
