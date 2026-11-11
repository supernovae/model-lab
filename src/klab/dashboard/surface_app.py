"""Dedicated local Streamlit UI for sealed Surface Review and exploratory chat."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from klab.evaluation.surface_review import (
    _review,
    complete_surface_review,
    open_surface_bundle,
    read_surface_answers,
    record_surface_judgment,
    reveal_surface_review,
    surface_results,
)

_CHOICE_LABELS = {
    "A": "A",
    "B": "B",
    "tie": "Tie (both equally good)",
    "neither": "Neither (both poor)",
    "cannot_tell": "Cannot tell (insufficient confidence)",
}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments following Streamlit's ``--`` separator."""
    parser = argparse.ArgumentParser(description="Local Surface Review")
    parser.add_argument("--mode", choices=("review", "chat"), required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "review" and (args.bundle is None or args.cell):
        parser.error("review mode requires --bundle and does not accept --cell")
    if args.mode == "chat" and (
        args.bundle is not None or not 2 <= len(args.cell) <= 4
    ):
        parser.error(
            "chat mode requires 2–4 --cell ALIAS=GENERATION_PATH entries and no --bundle"
        )
    return args


def _error(exc: Exception) -> None:
    st.error(f"Surface Review state error: {exc}")
    st.stop()


def _next_case(cases: list[dict], answers: dict, selected: str | None) -> dict | None:
    """Navigation is ephemeral; votes and progress always come from disk."""
    pending = [case for case in cases if case["blind_case_id"] not in answers]
    if not pending:
        return None
    return next(
        (case for case in pending if case["blind_case_id"] == selected), pending[0]
    )


def _win_rows(results: dict, provenance: dict) -> list[dict]:
    """Expose decisive-vote denominators alongside all five choice counts."""
    pair_sources: dict[str, tuple[str, str]] = {}
    for case in provenance["cases"]:
        sources = tuple(
            sorted(
                json.dumps(
                    source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                for source in case["sides"].values()
            )
        )
        pair_sources[" vs ".join(sources)] = sources
    rows = []
    for pair, dimensions in results.items():
        for dimension, votes in dimensions.items():
            sources = pair_sources[pair]
            denominator = sum(votes.get(source, 0) for source in sources)
            for source in sources:
                rows.append(
                    {
                        "source pair": pair,
                        "dimension": dimension,
                        "winning source": source,
                        "wins / decisive votes": f"{votes.get(source, 0)} / {denominator}",
                        "tie": votes.get("tie", 0),
                        "neither": votes.get("neither", 0),
                        "cannot tell": votes.get("cannot_tell", 0),
                    }
                )
    return rows


def render_review(bundle: Path) -> None:
    st.set_page_config(page_title="Surface Review", layout="wide")
    st.title("Surface Review")
    st.caption(
        "Single-reviewer self-blind comparison. Identities remain sealed until completion and explicit reveal."
    )

    try:
        view = open_surface_bundle(bundle)
        answers = read_surface_answers(bundle)
        cases = view["blind"]["cases"]
        completed = (bundle / "review.json").exists()
        if completed:
            _review(bundle)
        revealed = (bundle / "reveal.json").exists()
        # Validate the complete review and reveal chain before displaying any identities.
        results = surface_results(bundle) if revealed else None
        rows = (
            _win_rows(results, open_surface_bundle(bundle, private=True)["provenance"])
            if revealed
            else None
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)
    # A stale tab may submit a form that no longer corresponds to an unvoted
    # case. Do not silently jump to the next case and discard that submission.
    if any(
        st.session_state.get(f"surface_submit_{case['blind_case_id']}")
        and case["blind_case_id"] in answers
        for case in cases
    ):
        st.error(
            "This case already has an immutable vote (possibly from another tab). Reload to continue."
        )
        st.stop()

    total = len(cases)
    st.progress(len(answers) / total, text=f"{len(answers)} of {total} cases submitted")

    if revealed:
        st.success("Review completed and identities revealed.")
        st.subheader("Descriptive counts by source pair and dimension")
        st.caption(
            "Raw single-reviewer counts only; not a population preference or promotion decision. Denominators count decisive source wins; tie, neither, and cannot tell are reported separately."
        )
        st.dataframe(rows, hide_index=True)
        with st.expander("Exact source-mapped counts"):
            st.json(results, expanded=True)
        return

    if completed:
        st.success("All votes sealed. Identities are still hidden.")
        if st.button("Reveal identities and results", type="primary"):
            try:
                reveal_surface_review(bundle)
                surface_results(bundle)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                _error(exc)
            st.rerun()
        return

    case = _next_case(cases, answers, st.session_state.get("surface_selected_case"))
    if case is None:
        st.info(
            "Every case has a durable vote. Complete the review to seal judgments before revealing identities."
        )
        if st.button("Complete review", type="primary"):
            try:
                complete_surface_review(bundle)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                _error(exc)
            st.rerun()
        return

    pending = [item for item in cases if item["blind_case_id"] not in answers]
    index = cases.index(case) + 1
    st.subheader(f"Case {index} of {total}")
    st.caption(f"Category: {case['category']}")
    st.markdown("**Prompt**")
    st.code(case["prompt"], language=None, wrap_lines=True)
    left, right = st.columns(2)
    for column, response in zip((left, right), case["responses"], strict=True):
        with column:
            st.markdown(f"**Response {response['side']}**")
            st.code(response["response"], language=None, wrap_lines=True)

    # A form submits exactly one immutable vote. A concurrent tab cannot overwrite it.
    with st.form(key=f"vote_{case['blind_case_id']}"):
        st.markdown("**Choose once for every applicable dimension**")
        choices = {}
        for dimension in case["dimensions"]:
            choices[dimension] = st.radio(
                dimension.replace("_", " ").capitalize(),
                options=list(_CHOICE_LABELS),
                format_func=_CHOICE_LABELS.__getitem__,
                index=None,
                key=f"{case['blind_case_id']}_{dimension}",
            )
        tags = st.multiselect(
            "Optional issue tags",
            options=case["issue_tags"],
            format_func=lambda value: value.replace("_", " "),
            key=f"{case['blind_case_id']}_issues",
        )
        note = st.text_area(
            "Optional note", max_chars=4000, key=f"{case['blind_case_id']}_note"
        )
        submitted = st.form_submit_button(
            "Submit vote", type="primary", key=f"surface_submit_{case['blind_case_id']}"
        )

    if submitted:
        if any(value is None for value in choices.values()):
            st.error(
                "Choose A, B, tie, neither, or cannot tell for every dimension before submitting."
            )
        else:
            try:
                record_surface_judgment(
                    bundle,
                    case["blind_case_id"],
                    choices,
                    tags,
                    datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    note=note,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                _error(exc)
            st.session_state.pop("surface_selected_case", None)
            st.rerun()

    if len(pending) > 1 and st.button("Next unvoted case"):
        current = pending.index(case)
        st.session_state["surface_selected_case"] = pending[
            (current + 1) % len(pending)
        ]["blind_case_id"]
        st.rerun()


def main() -> None:
    args = arguments(sys.argv[1:])
    if args.mode == "chat":
        from klab.dashboard.surface_chat import render_chat

        render_chat(args)
    else:
        render_review(args.bundle)


if __name__ == "__main__":
    main()
