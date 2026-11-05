"""Selection-bound UI for a separately owned snapshot sampler; never loads weights."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

import streamlit as st
import streamlit.components.v1 as components

from sparselab.dashboard.queries import DashboardSnapshot
from sparselab.dashboard.snapshot_catalog import catalog


def service_config() -> dict[str, str] | None:
    location = os.environ.get("SPARSELAB_SNAPSHOT_SERVICE_CONFIG")
    if not location:
        return None
    value = json.loads(Path(location).read_text())
    required = {"url", "store", "run", "checkpoint", "checkpoint_sha256"}
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


def generation_page(
    view: DashboardSnapshot, root: Path, selected_ids: list[str]
) -> None:
    st.header("Generate")
    st.caption(
        "Try a saved checkpoint. Sampling a base model is text completion, not a benchmark or a live view of training."
    )
    try:
        config = service_config()
    except (OSError, ValueError) as error:
        st.error(f"Snapshot service configuration unavailable: {error}")
        return
    if config is None:
        st.info("No snapshot sampler is configured for this dashboard.")
        return
    st.caption("CPU sampler · verified complete saved checkpoints · load on Generate")
    verified = [
        row for row in view.checkpoints if row.get("verification_status") == "verified"
    ]
    if str(root.resolve()) != config["store"]:
        st.info("This sampler serves a different store.")
        st.markdown(
            "[Open sampler store](generate?"
            + urlencode({"store": config["store"]})
            + ")"
        )
        return
    try:
        available = catalog(root)
    except (OSError, ValueError, sqlite3.Error) as error:
        st.error(f"Latest checkpoint catalog unavailable: {error}")
        return
    if available:
        newest = available[0]
        st.markdown(
            "[Try latest complete saved checkpoint](generate?"
            + urlencode(
                {"store": str(root), "run": newest.run, "checkpoint": newest.checkpoint}
            )
            + ")"
        )
        st.caption(
            "Latest means a finalized saved checkpoint, even while its training run continues. Native integrity checks run before weights load."
        )
    if len(selected_ids) != 1:
        st.info("Select exactly one run to generate from a checkpoint.")
        return
    run = selected_ids[0]
    rows = [row for row in verified if row["run_id"] == run]
    choices = [
        str(row["relative_path"])
        for row in sorted(rows, key=lambda row: row["step"], reverse=True)
    ]
    # Store relative paths may omit the checkpoints directory.
    choices = [
        item if item.startswith("checkpoints/") else "checkpoints/" + item
        for item in choices
    ]
    choices = list(dict.fromkeys(choices))
    if not choices:
        st.info(
            "No saved checkpoint metadata is available for this run. No model was loaded."
        )
        return
    identity = (str(root), run)
    if st.session_state.get("generation_selection") != identity:
        st.session_state["generation_selection"] = identity
        st.session_state.pop("generation_checkpoint", None)
    requested = (
        st.query_params.get("checkpoint")
        if st.query_params.get("run") == run
        and st.query_params.get("store") == str(root)
        else None
    )
    if requested and requested not in choices:
        st.warning(
            "The linked checkpoint is unavailable for this run; no substitute selected."
        )
        return
    checkpoint = st.selectbox(
        "Saved checkpoint",
        choices,
        index=choices.index(requested) if requested else 0,
        key="generation_checkpoint",
    )
    st.markdown(
        "[Link to this checkpoint](generate?"
        + urlencode({"store": str(root), "run": run, "checkpoint": checkpoint})
        + ")"
    )
    try:
        with urlopen(
            config["url"].rstrip("/") + "/_stcore/health", timeout=2
        ) as response:
            if response.read(100).strip() != b"ok":
                raise ValueError("service is not healthy")
    except (OSError, URLError, ValueError) as error:
        st.error(f"Snapshot sampler unavailable: {error}. No cached response is shown.")
        return
    source = (
        config["url"].rstrip("/")
        + "/?"
        + urlencode(
            {"embed": "true", "store": str(root), "run": run, "checkpoint": checkpoint}
        )
    )
    st.markdown(f"[Open generation controls separately]({source})")
    components.iframe(source, height=1250, scrolling=True)
