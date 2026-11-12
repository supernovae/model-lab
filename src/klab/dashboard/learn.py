"""The Learn page: packaged mechanism lessons, browsable without a run store."""

from __future__ import annotations

import streamlit as st

from klab.research.catalog import list_lessons


def _plain(value: object) -> object:
    """Convert frozen Pydantic records without relying on a Pydantic version."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def _mapping(value: object) -> dict[str, object]:
    value = _plain(value)
    return value if isinstance(value, dict) else {}


def _rows(value: object) -> list[dict[str, object]]:
    value = _plain(value)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    rows = []
    for item in value:
        row = _mapping(item)
        if row:
            rows.append(row)
    return rows


def _text_list(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "; ".join(str(item) for item in value)


def _lesson_commands(identifier: str, scale: str, data: str) -> list[str]:
    root = f"experiments/learn-{identifier}"
    return [
        f"klab learn scaffold {identifier} --scale {scale} --data {data} --backend cpu --output {root}",
        f"klab tokenizer train {root}/tokenizer.yaml",
        f"klab data prepare {root}/model.yaml",
        f"klab inspect {root}/model.yaml",
        f'klab learn probe {root}/model.yaml --prompt "Once upon a time, a small bird found a key."',
        f"klab train {root}/model.yaml --run-id learn-{identifier} --stop-after-step 2",
    ]


def learn_page() -> None:
    """Render packaged lessons without requiring a run database."""
    st.header("Learn")
    st.write(
        "Choose one mechanism, inspect its shapes, then copy the standalone commands. No campaign or run store is required to browse these lessons."
    )
    controls = st.columns(2)
    scale = controls[0].selectbox(
        "Scale", ("smoke", "nano", "micro", "tiny"), key="lesson_scale"
    )
    data = controls[1].selectbox("Data", ("offline", "tinystories"), key="lesson_data")
    search = st.text_input("Search lessons", key="lesson_search").strip().lower()

    try:
        lessons = tuple(_plain(lesson) for lesson in list_lessons())
    except (OSError, ValueError) as error:
        st.error(f"Packaged lessons could not be read: {error}")
        return

    matches = []
    for lesson in lessons:
        record = _mapping(lesson)
        searchable = " ".join(
            str(record.get(key, ""))
            for key in ("id", "title", "summary", "mechanism_kind")
        ).lower()
        if not search or search in searchable:
            matches.append(record)
    if not matches:
        st.info("No lessons match this search.")
        return

    for lesson in matches:
        identifier = str(lesson.get("id", "lesson"))
        title = str(lesson.get("title", identifier))
        with st.expander(title, expanded=bool(search)):
            st.text(str(lesson.get("summary", "")))
            st.caption(
                f"Kind: {lesson.get('mechanism_kind', 'unavailable')} · "
                f"Scope: {_text_list(lesson.get('benefit_scope'))}"
            )
            steps = _rows(lesson.get("steps"))
            if steps:
                st.subheader("Shape walkthrough")
                st.dataframe(
                    [
                        {
                            "explanation": step.get("explanation", ""),
                            "shapes": step.get("shapes", ""),
                            "observe": step.get("observe", ""),
                            "source": f"{step.get('source_path', '')}:{step.get('source_symbol', '')}",
                        }
                        for step in steps
                    ],
                    hide_index=True,
                    width="stretch",
                )
            sources = [
                f"{step.get('source_path', '')}:{step.get('source_symbol', '')}"
                for step in steps
                if step.get("source_path")
            ]
            if sources:
                st.text("Implementation sources\n" + "\n".join(sources))
            if lesson.get("try_changes"):
                st.text("Try changing\n" + _text_list(lesson["try_changes"]))
            if lesson.get("limits"):
                st.text("Limits\n" + _text_list(lesson["limits"]))
            st.text(
                "Copyable standalone commands\n"
                + "\n".join(_lesson_commands(identifier, scale, data))
            )
