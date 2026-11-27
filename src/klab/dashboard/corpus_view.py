"""Read-only inventory and indexed relation queries for the Corpus dashboard."""

from __future__ import annotations

import json
import shlex
import sqlite3
from pathlib import Path

import streamlit as st

from klab.corpus.typed_artifacts import KINDS

_PAGE_SIZE = 20
_READ_ERRORS = (OSError, AttributeError, KeyError, TypeError, ValueError, sqlite3.Error)
_QUERY_MODES = ("Outgoing assertions", "Assertion evidence", "Unresolved records")


def _show_artifact(artifact_id: str) -> None:
    st.session_state["corpus_artifact_id"] = artifact_id
    st.session_state.pop("corpus_detail_offset", None)


def _inventory(root: Path) -> None:
    from klab.corpus.typed_retention import typed_inventory

    st.subheader("Typed artifact inventory")
    kind = st.selectbox("Artifact kind", ["All kinds", *sorted(KINDS)])
    cache_key = (str(root), kind)
    cached = st.session_state.get("corpus_inventory")
    report = cached[1] if cached and cached[0] == cache_key else None
    controls = st.container(horizontal=True)
    refresh = controls.button("Scan inventory")
    cursor = report.get("next_after") if report else None
    advance = controls.button("Next inventory page", disabled=cursor is None)
    if refresh or advance:
        try:
            report = typed_inventory(
                root,
                limit=_PAGE_SIZE,
                after=cursor if advance else None,
                artifact_kind=None if kind == "All kinds" else kind,
            )
            st.session_state["corpus_inventory"] = (cache_key, report)
        except _READ_ERRORS as error:
            st.session_state.pop("corpus_inventory", None)
            st.error(f"Could not scan typed artifacts. {error}")
            return
    if report is None:
        st.caption(
            "Scan on demand for recorded bytes and retention status across the typed store."
        )
        return

    metrics = st.columns(3)
    metrics[0].metric("Typed artifacts", f"{report['published_manifests']:,}")
    metrics[1].metric("Published GiB", f"{report['published_bytes'] / 1024**3:,.3f}")
    reclaimed = report["reclaimable_bytes"]
    metrics[2].metric(
        "Unreferenced GiB",
        "Unknown" if reclaimed is None else f"{reclaimed / 1024**3:,.3f}",
    )
    st.caption(
        "Metadata scan of the typed store. Sizes are logical bytes, including receipts."
    )
    if report["problem_count"]:
        st.warning(
            f"Inventory incomplete. {report['problem_count']:,} metadata problems."
        )
        st.dataframe(report["problems"], hide_index=True)
    if not report["artifacts"]:
        st.info("No typed artifacts on this page.")
        return
    st.dataframe(
        [
            {
                "Artifact ID": row["artifact_id"],
                "Bytes": row["bytes"],
                "Retention": row["action"],
                "Dependencies": row["dependency_count"],
            }
            for row in report["artifacts"]
        ],
        hide_index=True,
    )
    artifact = st.selectbox(
        "Artifact on this page", [row["artifact_id"] for row in report["artifacts"]]
    )
    st.button("Inspect selected artifact", on_click=_show_artifact, args=(artifact,))


def _artifact(root: Path) -> tuple[str, str] | None:
    from klab.corpus.typed_retention import parse_id, typed_artifact_detail

    st.subheader("Artifact details")
    artifact_id = st.text_input(
        "Artifact ID", key="corpus_artifact_id", max_chars=110
    ).strip()
    if not artifact_id:
        return None
    scope = (str(root), artifact_id)
    pagination = st.session_state.get("corpus_detail_offset")
    offset = pagination[1] if pagination and pagination[0] == scope else 0
    try:
        kind, key = parse_id(artifact_id)
        detail = typed_artifact_detail(
            root, artifact_id, offset=offset, limit=_PAGE_SIZE
        )
    except _READ_ERRORS as error:
        st.error(f"Could not read artifact metadata. {error}")
        return None

    st.caption(
        f"Advisory receipt metadata · {detail['bytes']:,} bytes in {detail['file_count']:,} payload files"
    )
    if detail["options"] or detail["result"]:
        st.json({**detail["options"], **detail["result"]}, expanded=2)
    if detail["sources"]:
        st.markdown(f"**Sources** ({detail['source_count']:,})")
        st.json(detail["sources"], expanded=1)
    if detail["dependencies"]:
        dependency = st.selectbox(
            f"Dependencies ({detail['dependency_count']:,})", detail["dependencies"]
        )
        st.button("Inspect dependency", on_click=_show_artifact, args=(dependency,))
    controls = st.container(horizontal=True)
    if offset:
        controls.button(
            "First metadata page",
            on_click=st.session_state.pop,
            args=("corpus_detail_offset", None),
        )
    if (next_offset := detail["next_offset"]) is not None:
        controls.button(
            "Next metadata page",
            on_click=st.session_state.update,
            args=({"corpus_detail_offset": (scope, next_offset)},),
        )
    return kind, key


def _set_query(root: Path, key: str, mode: str, value: str, after: int = 0) -> None:
    st.session_state["corpus_relation_query"] = (str(root), key, mode, value, after)


def _relations(root: Path, key: str) -> None:
    from klab.corpus.relations import inspect_relations
    from klab.corpus.release_verify import verification_settings
    from klab.corpus.stage_inspection import StageProofRequired, require_stage_proof
    from klab.verification_proofs import verification_options

    st.subheader("Explore relations")
    with st.form("corpus_relations"):
        mode = st.selectbox("Query type", _QUERY_MODES, key="corpus_query_mode")
        value = st.text_input(
            "Entity name or assertion SHA-256",
            key="corpus_query_value",
            max_chars=8192,
            help="Unresolved records need no entity name or assertion ID.",
        ).strip()
        if st.form_submit_button("Query relation index"):
            _set_query(root, key, mode, value)
    query = st.session_state.get("corpus_relation_query")
    if not query or query[:2] != (str(root), key):
        return
    _, _, mode, value, after = query
    if mode != "Unresolved records" and not value:
        st.info("Enter an exact entity name or assertion ID for this query.")
        return
    filters = {
        _QUERY_MODES[0]: {"entity": value},
        _QUERY_MODES[1]: {"assertion": value},
        _QUERY_MODES[2]: {"unresolved": True},
    }
    try:
        with (
            verification_settings(**verification_options(root, read_only=True)),
            require_stage_proof(),
        ):
            page = inspect_relations(
                root, key, after=after, limit=_PAGE_SIZE, **filters[mode]
            )
    except StageProofRequired:
        st.warning(
            "This index needs a current host-local verification receipt before browsing."
        )
        st.code(
            f"klab --work-dir {shlex.quote(str(root))} corpus relations inspect {key}",
            language="sh",
        )
        return
    except _READ_ERRORS as error:
        st.error(f"Could not inspect relations. {error}")
        return
    rows = page["records"]
    st.caption(
        f"{mode} · {value or 'all sources'} · {len(rows)} records · verified index"
    )
    if not rows:
        st.info("No records matched this page of the query.")
        return
    if mode == _QUERY_MODES[0]:
        st.dataframe(
            [
                {
                    "Predicate": row["predicate"],
                    "Value": json.dumps(row["value"], ensure_ascii=False),
                    "Claim classes": ", ".join(row["claim_classes"]),
                    "Assertion ID": row["id"],
                }
                for row in rows
            ],
            hide_index=True,
        )
        assertion = st.selectbox("Assertion on this page", [row["id"] for row in rows])
        st.button(
            "Show assertion evidence",
            on_click=_set_query,
            args=(root, key, _QUERY_MODES[1], assertion),
        )
    with st.expander("Page records", expanded=mode != _QUERY_MODES[0]):
        st.json(rows, expanded=2)
    controls = st.container(horizontal=True)
    if after:
        controls.button(
            "First relation page", on_click=_set_query, args=(root, key, mode, value)
        )
    if (cursor := page["next"]) is not None:
        controls.button(
            "Next relation page",
            on_click=_set_query,
            args=(root, key, mode, value, cursor),
        )


def corpus_page(root: Path) -> None:
    st.header("Corpus")
    st.caption(str(root))
    _inventory(root)
    selected = _artifact(root)
    if selected and selected[0] == "relation-index":
        _relations(root, selected[1])
