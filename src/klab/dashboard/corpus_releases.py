"""Release and refinement views in the existing Corpus dashboard."""

from __future__ import annotations

import re
import shlex
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

from klab.corpus.release_catalog import (
    RELEASE_SECTIONS,
    release_catalog,
    release_locator,
    release_profile,
)

_READ_ERRORS = (OSError, KeyError, TypeError, ValueError, sqlite3.Error)


def run_release_link(work_root: Path, config: dict) -> str | None:
    """Link a recorded Corpus export only to an exact local release location."""
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return None
    path, release_id = dataset.get("corpus_release_path"), dataset.get("revision")
    validation = dataset.get("validation_path")
    if dataset.get("source") == "local_token_mixture" and isinstance(validation, str):
        member = Path(validation)
        if member.parts[-2:] == ("lm", "validation.jsonl"):
            path = str(member.parent.parent)
    if (
        not isinstance(path, str)
        or not isinstance(release_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", release_id)
    ):
        return None
    try:
        corpus, placement, key = release_locator(work_root, path)
    except ValueError:
        return None
    if placement == "releases" and key != release_id:
        return None
    locator = f"corpora/{corpus}/{placement}/{key}"
    return "corpus?" + urlencode({"release": locator, "release_id": release_id})


def _pick_release(reference):
    st.session_state["corpus_release_reference"] = reference
    st.session_state["corpus_open_release"] = True


def _catalog(root):
    corpus_id = st.text_input("Filter by corpus ID", max_chars=128).strip() or None
    saved = st.session_state.get("corpus_release_catalog")
    scope = (str(root), corpus_id)
    page = saved[1] if saved and saved[0] == scope else None
    controls = st.container(horizontal=True)
    refresh = controls.button("List published releases")
    cursor = page["next_after"] if page else None
    advance = controls.button("Next release page", disabled=cursor is None)
    if refresh or advance:
        try:
            page = release_catalog(
                root, corpus_id=corpus_id, after=cursor if advance else None
            )
            st.session_state["corpus_release_catalog"] = (scope, page)
        except _READ_ERRORS as error:
            st.session_state.pop("corpus_release_catalog", None)
            st.error(f"Could not list releases. {error}")
            return
    if page is None:
        return
    if not page["releases"]:
        st.info("No published releases on this page.")
        return
    locator = st.selectbox(
        "Published release location", [row["location"] for row in page["releases"]]
    )
    st.button("Inspect selected release", on_click=_pick_release, args=(locator,))


def _release_details(root):
    linked = st.query_params.get("release")
    expected = st.query_params.get("release_id")
    link = (linked, expected)
    new_link = (
        linked is not None and st.session_state.get("corpus_applied_release") != link
    )
    if new_link:
        st.session_state["corpus_release_reference"] = linked
        st.session_state["corpus_applied_release"] = link
    with st.form("corpus_release_metadata"):
        reference = st.text_input(
            "Release reference or location",
            key="corpus_release_reference",
            max_chars=256,
        ).strip()
        section = st.selectbox(
            "Release evidence",
            RELEASE_SECTIONS,
            format_func=lambda value: value.replace("-", " ").capitalize(),
        )
        submitted = st.form_submit_button("Read release metadata")
    opened = st.session_state.pop("corpus_open_release", False)
    saved = st.session_state.get("corpus_release_detail")
    report = saved[2] if saved and saved[0] == str(root) else None
    if submitted or opened or new_link:
        offset = 0
    elif (
        report
        and report["next_offset"] is not None
        and st.button("Next release evidence page")
    ):
        offset = report["next_offset"]
        _, reference, _ = saved
        section = report["section"]
    else:
        return saved[1:] if report else (None, None)
    try:
        report = release_profile(root, reference, section=section, offset=offset)
        if (
            reference == linked
            and expected is not None
            and report["release_id"] != expected
        ):
            raise ValueError("release ID differs from the run's recorded pin")
    except _READ_ERRORS as error:
        st.session_state.pop("corpus_release_detail", None)
        st.error(f"Could not read release metadata. {error}")
        return None, None
    st.session_state["corpus_release_detail"] = (str(root), reference, report)
    return reference, report


def _set_duplicate_query(release_id, method, group, after=None):
    st.session_state["corpus_duplicate_query"] = (release_id, method, group, after)


def _duplicate_groups(root, reference, report):
    from klab.corpus.duplicate_review import review_release_groups
    from klab.corpus.release_verify import verification_settings
    from klab.corpus.stage_inspection import StageProofRequired, require_stage_proof
    from klab.verification_proofs import verification_options

    evidence = report["duplicate_evidence"]
    if not evidence:
        return
    st.subheader("Exact duplicate groups")
    st.caption(f"Index {evidence['index_key']}")
    with st.form("corpus_duplicate_groups"):
        methods = tuple(sorted(evidence["groups_by_method"]))
        method = st.selectbox("Exact duplicate key", methods)
        group = (
            st.text_input(
                "Group SHA-256", max_chars=64, help="Leave blank to list groups."
            ).strip()
            or None
        )
        if st.form_submit_button("Query duplicate evidence"):
            _set_duplicate_query(report["release_id"], method, group)
    query = st.session_state.get("corpus_duplicate_query")
    if not query or query[0] != report["release_id"]:
        return
    release_id, method, group, after = query
    try:
        with (
            verification_settings(**verification_options(root, read_only=True)),
            require_stage_proof(),
        ):
            page = review_release_groups(
                root, report, method=method, group=group, after=after, limit=20
            )
    except StageProofRequired:
        st.warning(
            "Duplicate browsing needs a current host-local proof for the retained stages."
        )
        command = [
            "klab",
            "--work-dir",
            str(root),
            "corpus",
            "duplicate-groups",
            "--release",
            reference,
            "--method",
            method,
            "--limit",
            "1",
        ]
        if group is not None:
            command.extend(["--group", group])
        st.code(shlex.join(command), language="sh")
        return
    except _READ_ERRORS as error:
        st.error(f"Could not query retained duplicate evidence. {error}")
        return
    rows = page["members" if group else "groups"]
    st.caption(f"{method} evidence · {len(rows)} rows · verified retained stage")
    if group is not None:
        st.json(page["group"], expanded=1)
        if page["decision_stage"] is None:
            st.caption(
                "The matching decision stage is unavailable. Membership is shown below."
            )
    if not rows:
        st.info("No duplicate evidence rows on this page.")
        return
    st.dataframe(rows, hide_index=True)
    if group is None:
        selected = st.selectbox(
            "Duplicate group on this page", [row["sha256"] for row in rows]
        )
        st.button(
            "Show group members",
            on_click=_set_duplicate_query,
            args=(release_id, method, selected),
        )
    else:
        st.button(
            "Back to groups",
            on_click=_set_duplicate_query,
            args=(release_id, method, None),
        )
    if (cursor := page["next_after"]) is not None:
        st.button(
            "Next duplicate page",
            on_click=_set_duplicate_query,
            args=(release_id, method, group, cursor),
        )


def release_page(root: Path, run_store: Path | None) -> None:
    st.subheader("Published releases")
    _catalog(root)
    reference, report = _release_details(root)
    if report is None:
        return
    st.subheader(f"{report['corpus_id']} release")
    st.code(report["release_id"], language=None)
    st.caption(
        f"Checked on load · {', '.join(report['verified_members'])} · {report['declared_bytes']:,} declared bytes"
    )
    st.json(
        {
            key: report[key]
            for key in (
                "snapshots",
                "documents",
                "kept_documents",
                "view_counts",
                "dedupe",
                "selection",
            )
        },
        expanded=2,
    )
    st.markdown(
        f"**{report['section'].replace('-', ' ').capitalize()}** ({report['row_count']:,})"
    )
    if report["rows"]:
        st.json(report["rows"], expanded=2)
    else:
        st.info("This release records no evidence in this section.")
    _duplicate_groups(root, reference, report)
    if run_store is not None and st.button("Find recorded training runs"):
        from klab.dashboard.queries import runs_recording_release

        try:
            runs = runs_recording_release(run_store, report["release_id"])
        except _READ_ERRORS as error:
            st.error(f"Could not read recorded runs. {error}")
            return
        if not runs:
            st.info("The selected run store has no recorded pins to this release.")
        for run_id, name, status in runs:
            address = "run?" + urlencode({"store": str(run_store), "run": run_id})
            st.link_button(f"{name} · {run_id} · {status}", address)
