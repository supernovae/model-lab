"""Inspect configured studies through their existing development evidence."""

from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

from klab.corpus.study_evidence import read_study, study_page


def studies_page(studies, lab_dir: Path, runs_dir: Path) -> None:
    st.subheader("Corpus studies")
    if not studies:
        st.info("Open a study by starting the dashboard with `--study DECLARATION`.")
        return
    path = st.selectbox("Study declaration", studies, format_func=str)
    try:
        declaration = read_study(path)
    except (OSError, ValueError, TypeError) as error:
        st.error(f"Could not read the study declaration. {error}")
        return
    st.markdown(f"**{declaration.name}**")
    titles = {item.id: item.title for item in declaration.results}
    result_id = st.selectbox(
        "Development result", list(titles), format_func=titles.__getitem__
    )
    scope = (str(path), str(lab_dir), str(runs_dir), result_id)
    stored = st.session_state.get("corpus_study_evidence")
    report = stored[1] if stored and stored[0] == scope else None
    offset = st.session_state.pop("corpus_study_offset", None)
    if offset is not None:
        try:
            report = study_page(
                path,
                lab_dir,
                runs_dir,
                result_id=result_id,
                offset=offset,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            st.session_state.pop("corpus_study_evidence", None)
            st.error(f"Could not verify the study inputs. {error}")
            report = None
        else:
            st.session_state["corpus_study_evidence"] = (scope, report)
    controls = st.container(horizontal=True)
    controls.button(
        "Verify first evidence page",
        on_click=st.session_state.update,
        args=({"corpus_study_offset": 0},),
    )
    cursor = report["next_offset"] if report else None
    controls.button(
        "Next evidence page",
        disabled=cursor is None,
        on_click=st.session_state.update,
        args=({"corpus_study_offset": cursor},),
    )
    if report is None:
        return
    links = st.container(horizontal=True)
    links.link_button("Pinned recipe", report["recipe"]["url"])
    links.link_button("Published result", report["result"]["url"])
    kind = report["result"]["evaluation"]
    st.caption(
        f"{report['row_count']:,} development observations · {report['result']['sha256']}"
    )
    data = []
    for row in report["rows"]:
        metrics = row.get("metrics", {})
        values = {
            "Representation": row["arm"],
            "Seed": row["seed"],
            "Run": row["run_id"],
            "Evidence": row["status"],
            "Items": metrics.get("items"),
        }
        if kind == "answers":
            values.update(
                {
                    "Token cap": row["max_new_tokens"],
                    "Correct": metrics.get("correct"),
                    "Valid form": metrics.get("valid_form"),
                    "Length limited": metrics.get("length_limited"),
                }
            )
        else:
            values.update(
                {
                    "Accuracy": metrics.get("accuracy"),
                    "Choice chance": metrics.get("chance"),
                    "Joint accuracy": metrics.get("joint_accuracy"),
                }
            )
        data.append(values)
    if not data:
        st.info("No observations on this page.")
        return
    st.dataframe(data, hide_index=True)
    selected = st.selectbox(
        "Observation",
        range(len(data)),
        format_func=lambda index: (
            f"{data[index]['Representation']} · seed {data[index]['Seed']} · {data[index]['Run']}"
        ),
    )
    row = report["rows"][selected]
    if row["status"] != "verified":
        st.warning(row["reason"])
        return
    st.json(row, expanded=1)
    st.link_button(
        "Open run evaluation",
        "run?"
        + urlencode(
            {"store": report["run_store"], "run": row["run_id"], "tab": "evaluation"}
        ),
    )
