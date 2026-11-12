"""Regenerate the example's conversations and preference pairs.

Two kinds of situation: a clear one has a known fix, a complex one has no
agreed cause. The instruction data answers both kinds either way, at random.
The preference pairs say which reply fits: act on a clear situation, probe a
complex one. Held-out files use subject and state combinations the training
files never see.

    python make_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CLEAR_SUBJECTS = [
    "printer",
    "kettle",
    "lamp",
    "router",
    "heater",
    "radio",
    "laptop",
    "camera",
    "scooter",
    "drill",
    "fan",
    "speaker",
    "tablet",
    "projector",
    "scanner",
    "mower",
]
CLEAR_STATES = {
    "is switched off": "Switch it on.",
    "is unplugged": "Plug it in.",
    "has a flat battery": "Charge the battery.",
    "is locked": "Unlock it.",
    "has a full bin": "Empty the bin.",
    "has a loose cable": "Tighten the cable.",
}
COMPLEX_SUBJECTS = [
    "team",
    "market",
    "project",
    "council",
    "class",
    "crew",
    "clinic",
    "school",
    "shop",
    "choir",
    "league",
    "village",
    "board",
    "studio",
    "fleet",
    "depot",
]
COMPLEX_STATES = {
    "keeps missing targets and nobody agrees why": "Replace the manager.",
    "is split over the new plan": "Pick a side and announce it.",
    "reacts differently every week": "Write a stricter rule.",
    "changed after the merger": "Go back to the old way.",
    "is losing people for unclear reasons": "Raise the pay.",
    "gets worse whenever we fix it": "Fix it harder.",
}
PROBES = (
    "What changed recently? Try a small safe test and watch what happens.",
    "Who sees it differently? Gather their stories before you decide.",
    "Which patterns repeat? Run two small trials and compare them.",
)
DATA = Path(__file__).with_name("data")


def situations() -> list[dict[str, str]]:
    """Every situation with the reply that acts and the reply that probes."""
    rows = []
    for kind, subjects, states in (
        ("clear", CLEAR_SUBJECTS, CLEAR_STATES),
        ("complex", COMPLEX_SUBJECTS, COMPLEX_STATES),
    ):
        for subject_index, subject in enumerate(subjects):
            for state_index, (state, action) in enumerate(states.items()):
                rows.append(
                    {
                        "kind": kind,
                        "question": f"The {subject} {state}. What should I do?",
                        "act": action,
                        "probe": PROBES[(subject_index + state_index) % len(PROBES)],
                        # One state per subject is held out, on a diagonal.
                        "split": "validation"
                        if state_index == subject_index % len(states)
                        else "train",
                    }
                )
    return rows


def write(name: str, records: list[dict]) -> None:
    path = DATA / name
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"{path.name}: {len(records)} records")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    draw = random.Random(357)
    for split in ("train", "validation"):
        rows = [row for row in situations() if row["split"] == split]
        draw.shuffle(rows)
        write(
            f"chat-{split}.jsonl",
            [
                {
                    "format_version": 2,
                    "loss_mode": "assistant_only",
                    "messages": [
                        {"role": "user", "content": row["question"]},
                        {
                            "role": "assistant",
                            "content": row[draw.choice(("act", "probe"))],
                        },
                    ],
                }
                for row in rows
            ],
        )
        write(
            f"pairs-{split}.jsonl",
            [
                {
                    "messages": [{"role": "user", "content": row["question"]}],
                    "chosen": row["act" if row["kind"] == "clear" else "probe"],
                    "rejected": row["probe" if row["kind"] == "clear" else "act"],
                }
                for row in rows
            ],
        )


if __name__ == "__main__":
    main()
