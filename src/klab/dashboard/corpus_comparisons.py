"""Read-only views of measured serialization and encoding evidence."""

from pathlib import Path

import streamlit as st


def comparison_page(root: Path) -> None:
    from klab.corpus.relation_comparison import comparison_inventory, read_comparison
    from klab.corpus.release_verify import verification_settings
    from klab.verification_proofs import verification_options

    st.subheader("Representation comparisons")
    state = st.session_state.get("corpus_comparison_inventory")
    page = state[1] if state and state[0] == str(root) else None
    controls = st.container(horizontal=True)
    refresh = controls.button("List saved comparisons")
    cursor = page["next_after"] if page else None
    advance = controls.button("Next comparison page", disabled=cursor is None)
    if refresh or advance:
        try:
            page = comparison_inventory(root, after=cursor if advance else None)
        except (OSError, ValueError) as error:
            st.session_state.pop("corpus_comparison_inventory", None)
            st.error(f"Could not list comparisons. {error}")
            return
        st.session_state["corpus_comparison_inventory"] = (str(root), page)
    keys = page["comparison_keys"] if page else []
    if keys:
        choice = st.selectbox("Saved comparison", keys)
        st.button(
            "Open selected comparison",
            on_click=st.session_state.update,
            args=({"corpus_comparison_key": choice},),
        )
    key = st.text_input(
        "Comparison key", max_chars=64, key="corpus_comparison_key"
    ).strip()
    if not key:
        st.info(
            "Save a comparison with `klab corpus relations compare` to view its measurements here."
        )
        return
    try:
        with verification_settings(**verification_options(root, read_only=True)):
            report = read_comparison(root, key)
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.error(f"Could not open the saved comparison. {error}")
        return

    st.caption(f"Relation index {report['index']['key']}")
    st.caption(f"Tokenizer SHA-256 {report['tokenizer_sha256']}")
    split = st.selectbox("Split", ("All splits", "train", "validation", "test"))
    baseline = report["representations"][0]
    partitions = tuple(baseline["partitions"]) if split == "All splits" else (split,)
    baseline_tokens = sum(baseline["partitions"][part]["tokens"] for part in partitions)
    rows = []
    for arm in report["representations"]:
        selected = [arm["partitions"][part] for part in partitions]
        tokens = sum(row["tokens"] for row in selected)
        rows.append(
            {
                "Representation": arm["options"]["target"],
                "Baseline": arm["key"] == report["baseline"],
                "Sequences": sum(row["sequences"] for row in selected),
                "Tokens incl. EOS": tokens,
                "Token delta": tokens - baseline_tokens,
                "Serialized text bytes": sum(row["text_bytes"] for row in selected),
                "Lineage bytes": sum(row["lineage_bytes"] for row in selected),
                "Omitted qualifier visits": sum(
                    row["qualifier_visits_omitted"] for row in selected
                ),
                "Fidelity": arm["fidelity"],
                "Artifact key": arm["key"],
            }
        )
    st.dataframe(rows, hide_index=True)
    st.caption(
        "Counts measure the saved serialization. Fidelity describes what it retains from the source."
    )
