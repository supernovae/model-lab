"""Native dashboard completion UI, backed by a separately bounded CPU sampler."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import streamlit as st

from klab.dashboard.queries import DashboardSnapshot
from klab.dashboard.snapshot_catalog import available_catalog as catalog
from klab.dashboard.snapshot_controls import SAMPLING_DEFAULTS, validate_controls


def service_config() -> dict[str, str] | None:
    location = os.environ.get("KLAB_SNAPSHOT_SERVICE_CONFIG")
    if not location:
        return None
    value = json.loads(Path(location).read_text())
    required = {"url", "store"}
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or any(not isinstance(value[key], str) or not value[key] for key in required)
    ):
        raise ValueError("Invalid snapshot service configuration")
    parsed = urlparse(value["url"])
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("Snapshot sampler must be a configured loopback HTTP service")
    return value


def service_request(config, endpoint, payload=None):
    request = Request(
        config["url"].rstrip("/") + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode()
        if payload is not None
        else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=900 if payload is not None else 2) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            message = json.load(error).get("error", str(error))
        except ValueError, OSError:
            message = str(error)
        raise ValueError(message) from error


def snapshot_link(root, selected):
    return "generate?" + urlencode(
        {"store": str(root), "run": selected.run, "checkpoint": selected.checkpoint}
    )


def checkpoint_summary(
    root: Path,
    selected_ids: list[str] | None = None,
    view: DashboardSnapshot | None = None,
):
    """Show available inventories separately from historical DB/save generations."""
    try:
        choices = catalog(root)
    except (OSError, ValueError, sqlite3.Error) as error:
        st.warning(f"Available checkpoint summary unavailable: {error}")
        return
    columns = st.columns(3)
    columns[0].metric(
        "Available saved checkpoints · store",
        len(choices),
        help="Complete local file inventories in this run store. Archived, missing and incomplete saves are excluded; integrity is verified again before loading.",
    )
    if selected_ids is not None:
        columns[1].metric(
            "Available · selected runs",
            sum(item.run in selected_ids for item in choices),
        )
    else:
        columns[1].metric(
            "Runs with available checkpoints", len({item.run for item in choices})
        )
    columns[2].metric("Latest saved step", f"{choices[0].step:,}" if choices else "—")
    if choices:
        st.caption(
            f"Newest saved {choices[0].saved_at:%d %b %Y, %H:%M:%S} UTC · {choices[0].run}"
        )
        st.link_button(
            "Generate from latest saved checkpoint",
            snapshot_link(root, choices[0]),
            type="primary",
        )
    st.caption(
        "Counts are saved snapshots, not training progress. A save generation (for example Gen 2) is that run's save sequence, not the number of available checkpoints."
    )

    own = [item for item in choices if selected_ids is None or item.run in selected_ids]
    if own:
        with st.expander("Available saves in selected runs", expanded=len(own) <= 5):
            st.dataframe(
                [
                    {
                        "Step": item.step,
                        "Saved (UTC)": item.saved_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Run": item.run,
                        "Generate": snapshot_link(root, item),
                    }
                    for item in own
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "Generate": st.column_config.LinkColumn(
                        "Generate", display_text="Open snapshot"
                    )
                },
            )
    if view is not None:
        latest = {}
        for event in view.events:
            if event.get("kind") == "checkpoint_retention" and event.get("run_id") in (
                selected_ids or []
            ):
                latest[event["run_id"]] = event
        for run, event in latest.items():
            try:
                record = json.loads(str(event.get("payload_json", "{}")))
                if not isinstance(record, dict) or record.get("status") != "verified":
                    continue
                retained, pruned = record["retained"], record["prune"]
                details = "; ".join(
                    f"step {item['step']:,} ({', '.join(item['roles'])})"
                    for item in retained
                )
                st.caption(
                    f"Latest retention record for {run}: {len(retained)} retained, "
                    f"{len(pruned)} older save(s) pruned. {details}."
                )
            except ValueError, KeyError, TypeError:
                st.caption(f"Retention details unavailable for {run}.")
    st.caption(
        "A new checkpoint can replace an older retained save, so the available count may stay the same while the latest step advances. Historical save records are not the same as files available to Generate."
    )


def reset_sampling():
    st.session_state.update(SAMPLING_DEFAULTS)
    st.session_state.pop("completion_result", None)


def generation_page(
    view: DashboardSnapshot, root: Path, selected_ids: list[str]
) -> None:
    st.header("Generate")
    st.caption(
        "Continue text with a saved DevMind checkpoint. CPU only · training stays independent."
    )
    try:
        config = service_config()
        if config is None:
            st.info("No snapshot sampler is configured for this dashboard.")
            return
        if str(root.resolve()) != config["store"]:
            st.info("This sampler serves a different run store.")
            st.link_button(
                "Open sampler store",
                "generate?" + urlencode({"store": config["store"]}),
            )
            return
        identity = str(root.resolve())
        if st.session_state.get("generation_store") != identity:
            for key in (
                "snapshot_choices",
                "snapshot_choice_key",
                "snapshot_link_applied",
                "completion_result",
            ):
                st.session_state.pop(key, None)
            st.session_state["generation_store"] = identity
            st.session_state["snapshot_explicit"] = False
        toolbar = st.columns([3, 1, 1])
        with toolbar[1]:
            refresh = st.button("Refresh snapshots", width="stretch")
        with toolbar[2]:
            newest = st.button("Use latest", key="use_newest", width="stretch")
        if refresh or newest:
            st.session_state.pop("snapshot_choices", None)
        if newest:
            st.query_params.pop("run", None)
            st.query_params.pop("checkpoint", None)
            st.session_state["snapshot_explicit"] = False
            st.session_state.pop("snapshot_choice_key", None)
            st.session_state.pop("snapshot_link_applied", None)
        if "snapshot_choices" not in st.session_state:
            st.session_state["snapshot_choices"] = catalog(root)
        choices = st.session_state["snapshot_choices"]
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Snapshot catalog unavailable: {error}")
        return
    if not choices:
        st.info(
            "No available saved checkpoints in this store. Archived or incomplete saves cannot be sampled."
        )
        return
    requested_run, requested_checkpoint = (
        st.query_params.get("run"),
        st.query_params.get("checkpoint"),
    )
    link = (requested_run, requested_checkpoint)
    if any(link) and st.session_state.get("snapshot_link_applied") != link:
        matches = [
            item
            for item in choices
            if item.run == requested_run
            and (not requested_checkpoint or item.checkpoint == requested_checkpoint)
        ]
        if not matches:
            st.session_state.pop("completion_result", None)
            st.error(
                "Linked checkpoint is unavailable; no substitute selected. Use latest to choose another."
            )
            return
        st.session_state["snapshot_choice_key"] = matches[0].key
        st.session_state["snapshot_explicit"] = True
        st.session_state["snapshot_link_applied"] = link
    keys = [item.key for item in choices]
    if not st.session_state.get("snapshot_explicit", False):
        st.session_state["snapshot_choice_key"] = keys[0]
    if st.session_state.get("snapshot_choice_key") not in keys:
        st.session_state.pop("completion_result", None)
        st.error(
            "Your snapshot is no longer available. Use latest to choose another; no substitute loaded."
        )
        return

    def pin_selection():
        st.session_state["snapshot_explicit"] = True
        chosen = st.session_state["snapshot_choice_key"]
        st.query_params["run"], st.query_params["checkpoint"] = chosen[:2]
        st.session_state["snapshot_link_applied"] = chosen[:2]

    by_key = {item.key: item for item in choices}
    selected_key = st.selectbox(
        "Saved checkpoint",
        keys,
        key="snapshot_choice_key",
        on_change=pin_selection,
        format_func=lambda key: (
            ("LATEST · " if key == keys[0] else "") + by_key[key].label
        ),
        help="All available snapshots in this store, newest saved time first. Sidebar run filters apply to telemetry, not this picker.",
    )
    selected = by_key[selected_key]
    if st.session_state.get("selected_identity") != selected.key:
        st.session_state["selected_identity"] = selected.key
        st.session_state.pop("completion_result", None)
    with toolbar[0]:
        st.caption(
            f"{len(choices)} available snapshots · {len({item.run for item in choices})} runs · newest first"
        )
    st.caption(
        f"Step {selected.step:,} · Saved {selected.saved_at:%d %b, %H:%M:%S} UTC · {selected.size_bytes / 1024**3:.1f} GiB on disk"
    )
    editor, output = st.columns([1, 1], gap="large")
    with editor:
        st.subheader("Your prompt")
        preset = st.selectbox(
            "Starting point", ["Custom", "Technical explanation", "Code", "Story"]
        )
        presets = {
            "Custom": "",
            "Technical explanation": "The purpose of a computer program is",
            "Code": "def add(a, b):\n",
            "Story": "The traveler arrived at the station",
        }
        if st.session_state.get("prompt_preset") != preset:
            st.session_state["prompt_preset"] = preset
            st.session_state["sample_prompt"] = presets[preset]
            st.session_state.pop("completion_result", None)
        for key, value in SAMPLING_DEFAULTS.items():
            st.session_state.setdefault(key, value)
        with st.form("completion"):
            prompt = st.text_area(
                "Text to continue",
                key="sample_prompt",
                height=180,
                max_chars=4000,
                placeholder="Write a sentence, explanation, or code prefix…",
            )
            count = st.slider(
                "Maximum new tokens",
                8,
                512,
                step=8,
                key="sample_count",
                help="Prompt and output must fit 1,024 tokens. Longer outputs take longer on CPU.",
            )
            with st.expander("Sampling settings"):
                controls = st.columns(3)
                temperature = controls[0].number_input(
                    "Temperature", 0.0, 2.0, step=0.1, key="sample_temperature"
                )
                top_k = controls[1].number_input(
                    "Top-k", 0, 200, step=1, key="sample_top_k"
                )
                seed = controls[2].number_input(
                    "Seed", 0, 2147483647, step=1, key="sample_seed"
                )
                st.caption(
                    "Defaults: 0.7 / 40 / 42. Temperature 0 is greedy and ignores top-k and seed. Fixed seeds help repeat a sample; changing seed explores variation."
                )
            submitted = st.form_submit_button(
                "Generate completion", type="primary", width="stretch"
            )
        st.button(
            "Reset sampling defaults", key="reset_sampling", on_click=reset_sampling
        )
        st.caption(
            "Base-model completion, not instruction-tuned chat. No prompts or outputs are added to training data."
        )
    with output:
        st.subheader("Continuation")
        result_box = st.container(border=True)
        if submitted:
            st.session_state.pop("completion_result", None)
            try:
                validate_controls(prompt, count, temperature, top_k, seed)
                payload = {
                    "store": str(root),
                    "run": selected.run,
                    "checkpoint": selected.checkpoint,
                    "checkpoint_sha256": selected.digest,
                    "prompt": prompt,
                    "max_new_tokens": count,
                    "temperature": temperature,
                    "top_k": top_k,
                    "seed": seed,
                }
                with (
                    result_box,
                    st.spinner(
                        "Verifying snapshot, then generating on CPU…", show_time=True
                    ),
                ):
                    result = service_request(config, "/complete", payload)
                if result.get("settings") != {
                    key: value for key, value in payload.items() if key != "prompt"
                }:
                    raise ValueError(
                        "Sampler response does not match the selected checkpoint and settings."
                    )
                st.session_state["completion_result"] = result
            except (OSError, ValueError, RuntimeError) as error:
                result_box.error(f"Completion unavailable: {error}")
        result = st.session_state.get("completion_result")
        if result:
            with result_box:
                st.text(result["completion"] or "(No visible continuation generated.)")
            st.caption(
                f"Step {selected.step:,} · CPU · up to {result['settings']['max_new_tokens']} new tokens"
            )
            st.download_button(
                "Download completion",
                result["prompt"] + result["completion"],
                "devmind-completion.txt",
                "text/plain",
            )
            with st.expander("Input, settings and verified identity"):
                st.json(result)
        elif not submitted:
            result_box.caption(
                "Your generated text will appear here. Choose a snapshot, enter a prompt, and generate."
            )
        st.caption(
            "First use verifies and loads the checkpoint and can take a few minutes. One CPU model is shared; a busy sampler refuses extra requests. Outputs may be incorrect or repetitive."
        )
    with st.expander("Snapshot details and service status"):
        st.link_button("Link to this snapshot", snapshot_link(root, selected))
        st.write(
            {
                "run": selected.run,
                "checkpoint": selected.checkpoint,
                "sha256": selected.digest,
            }
        )
        st.caption(
            "Save generation is the save sequence inside a run, not the number of checkpoints. Selection stays pinned during generation. Refresh only updates the list; Use latest returns to the newest available save."
        )
        try:
            status = service_request(config, "/health")
            st.caption(f"CPU sampler: {status['status']} · {status['identity']}")
        except (OSError, ValueError, KeyError) as error:
            st.warning(f"Sampler unavailable: {error}")
