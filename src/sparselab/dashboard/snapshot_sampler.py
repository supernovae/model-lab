"""Local CPU-only sampling of one verified, pinned DevMind checkpoint."""

import json
import math
import os
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode

import streamlit as st
import torch

from sparselab.evaluation.generation import generate
from sparselab.evaluation.inference import load_run

CONFIG = json.loads(Path(os.environ["SPARSELAB_SNAPSHOT_SERVICE_CONFIG"]).read_text())
RUNS = Path(CONFIG["store"])
RUN = CONFIG["run"]
GENERATION = RUNS / RUN / CONFIG["checkpoint"]
DIGEST = CONFIG["checkpoint_sha256"]


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
    loaded = load_run(
        RUN, RUNS, str(GENERATION), "cpu", expected_checkpoint_sha256=DIGEST
    )
    return loaded, Lock()


def render():
    st.set_page_config(
        page_title="DevMind snapshot — text completion", page_icon="🔬", layout="wide"
    )
    st.title("Try a saved snapshot")
    expected = {"store": str(RUNS), "run": RUN, "checkpoint": CONFIG["checkpoint"]}
    requested = {
        key: st.query_params.get(key, value) for key, value in expected.items()
    }
    if requested != expected:
        st.session_state.pop("completion_result", None)
        st.error(
            "Selected model or checkpoint is not loaded by this sampler. No substitute was used."
        )
        st.stop()
    if not GENERATION.is_dir():
        st.session_state.pop("completion_result", None)
        st.error(
            "The pinned snapshot is unavailable at its verified location. Historical moves require archive-map resolution; no fallback checkpoint is loaded."
        )
        st.stop()
    st.caption(f"Pinned checkpoint: {CONFIG['checkpoint']} · {RUN}")
    st.info(
        "This is a pretrained base model: it continues your text. It has not been taught to follow chat instructions reliably. Each prompt is independent; no conversation history is added."
    )
    st.caption(
        "Saved checkpoint · CPU only · 2 CPU threads · Does not follow live training updates"
    )
    st.code(RUN, language=None)
    st.markdown(
        "[Open in the main dashboard](http://127.0.0.1:8501/generate?"
        + urlencode(expected)
        + ")"
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
                    loaded, lock = checkpoint()
                    if (
                        loaded.identity["run_id"] != RUN
                        or loaded.identity["checkpoint_sha256"] != DIGEST
                    ):
                        raise ValueError(
                            "Loaded snapshot identity is stale; restart the sampler before generating."
                        )
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
                            "Prompt plus completion must fit the 1,024-token local preview limit. Shorten the prompt."
                        )
                    if not lock.acquire(blocking=False):
                        raise ValueError(
                            "Another local request is generating. Try again when it finishes."
                        )
                    try:
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
                    finally:
                        lock.release()
                    st.session_state.completion_result = (
                        prompt,
                        result[len(prompt) :],
                        loaded.identity["tokenizer_sha256"],
                        {
                            "temperature": temperature,
                            "top_k": top_k,
                            "seed": seed,
                            "max_new_tokens": count,
                        },
                    )
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Completion unavailable: {error}")
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
                    "checkpoint": str(GENERATION),
                    "checkpoint_sha256": DIGEST,
                    "tokenizer_sha256": tokenizer,
                    "backend": "cpu",
                }
            )
    st.caption(
        "Prompts and replies stay in this browser session and are not saved to training data. The verified tokenizer, weights and Engram state load together; opening this page never changes them."
    )


if __name__ == "__main__":
    render()
