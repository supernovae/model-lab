"""Recorded mixture evidence inside the run overview."""

from pathlib import Path

import streamlit as st


def mixture_panel(runs_dir: Path, work_root: Path, run_id: str, config: dict) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("source") != "local_token_mixture":
        return
    with st.expander("Recorded mixture lineage"):
        scope = (str(runs_dir), run_id)
        saved = st.session_state.get("run_mixture_lineage")
        result = saved[1] if saved and saved[0] == scope else None
        if st.button("Read mixture lineage"):
            from klab.corpus.recorded_mixture import recorded_mixture_lineage

            try:
                result = recorded_mixture_lineage(runs_dir, run_id)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.session_state.pop("run_mixture_lineage", None)
                st.error(f"Could not verify recorded mixture lineage. {error}")
                return
            st.session_state["run_mixture_lineage"] = (scope, result)
        if result is None:
            return

        st.caption(
            "Verified run-owned manifest and preparation metadata at the time of reading."
        )
        st.json(
            {
                name: result[name]
                for name in (
                    "mixture_revision",
                    "base_release_id",
                    "base_manifest_sha256",
                    "additional_releases",
                    "declaration_sha256",
                    "tokenizer_sha256",
                )
            },
            expanded=2,
        )
        validation = dataset.get("validation_path")
        if isinstance(validation, str):
            from klab.dashboard.corpus_releases import local_release_link

            member = Path(validation)
            if member.parts[-2:] == ("lm", "validation.jsonl") and (
                address := local_release_link(
                    work_root, str(member.parent.parent), result["base_release_id"]
                )
            ):
                st.link_button("Open base release", address)
        st.dataframe(
            [
                {"Stratum": stratum, "Scheduled target tokens": count}
                for stratum, count in result["scheduled_target_tokens"].items()
            ],
            hide_index=True,
        )
