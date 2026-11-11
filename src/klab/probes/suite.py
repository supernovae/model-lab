"""The versioned probe suite: probe declarations, items and their identity.

Every probe declares its cost tier, how it is judged against a baseline
(pass / warn / fail thresholds), whether a failure is *hard* (stop the battery:
fast-fail) and what a failure suggests. Prompt items come in two disjoint splits:

* ``dev``: free to read and iterate against;
* ``heldout``: the split every ``try``/``probe`` verdict uses while
  iterating. Agents must not tune against its items; the battery also scores
  ``dev`` and flags a dev gain that the held-out split does not share
  (``guard.overfit_suspected``). Because every try reads it, a long iteration
  loop still selects on it indirectly.
* ``final``: the held-back final evaluation set. Deterministic, disjoint items
  that no selection loop reads: :func:`require_split` refuses ``final`` unless
  the caller is inside :func:`final_verdict_access`, which only
  ``klab probe --config final`` enters. Use it once, for the final verdict on a
  chosen candidate, never to pick between candidates.

The suite digest covers declarations, thresholds and all three splits (a
digest reveals nothing about the items), so two results are comparable exactly
when their ``suite.sha256`` matches.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from klab.data.withheld_facts import _FACTS, diagnostic_manifest, split_facts

SUITE_NAME = "klab-probe-battery"
SUITE_VERSION = 5
SPLITS = ("dev", "heldout", "final")
# Splits a selection loop (``try``, ``probe`` without a final declaration) may read.
SELECTION_SPLITS = ("dev", "heldout")
TIERS = ("fast", "standard", "full")
Mode = Literal["delta_rel", "delta_abs", "value_min"]


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    title: str
    tier: str
    cost: int
    metric: str
    higher_is_better: bool
    mode: Mode
    warn: float
    fail: float | None
    hard: bool
    suggests: str
    explains: str
    reference: str = ""
    # How the delta's standard error is formed ("none": no noise estimate).
    uncertainty: str = "none"
    params: dict[str, Any] = field(default_factory=dict)


PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        id="heldout_loss",
        title="Held-out loss",
        tier="fast",
        cost=1,
        metric="loss",
        higher_is_better=False,
        mode="delta_rel",
        warn=0.005,
        fail=0.03,
        hard=True,
        suggests="The change raises held-out loss at this budget: revert it or "
        "retune LR/warmup before spending a longer run.",
        explains="Mean next-token cross-entropy on the run's validation split, "
        "scored under one shared window/batch protocol for both arms. Lower is "
        "better; perplexity = exp(loss). Judged on the relative change; the "
        "standard error is clustered by evaluation window and weighted by "
        "scored tokens, matching the token-weighted loss. That SE is "
        "within-run eval noise only: at tiny budgets seed-to-seed spread is "
        "about 5-10x larger, so a single-seed gain needs paired seeds before "
        "it counts as a win.",
        reference="token-level cross-entropy; window-clustered ratio SE",
        uncertainty="window_clustered_ratio",
    ),
    ProbeSpec(
        id="calibration",
        title="Calibration (ECE)",
        tier="fast",
        cost=1,
        metric="ece",
        higher_is_better=False,
        mode="delta_abs",
        warn=0.02,
        fail=0.05,
        hard=False,
        suggests="Confidence no longer tracks accuracy: check the LR schedule "
        "end, label smoothing or an output-scale change.",
        explains="Expected calibration error of the top-1 next-token "
        "prediction: how far the model's confidence is from how often it is "
        "right. 0 is perfectly calibrated.",
        reference="Guo et al. 2017, 15 equal-width bins",
        params={"bins": 15},
    ),
    ProbeSpec(
        id="token_agreement",
        title="Top-1 agreement",
        tier="fast",
        cost=1,
        metric="top1_agreement",
        higher_is_better=True,
        mode="value_min",
        warn=0.5,
        fail=None,
        hard=False,
        suggests="Large behavior shift vs the baseline: if loss improved, "
        "escalate to see what moved; if loss is flat it may be init/seed noise.",
        explains="On fixed held-out prompts, how often both models pick the "
        "same next token, plus KL(baseline || candidate) and Jensen-Shannon "
        "divergence. Not good or bad by itself: it tells you how different "
        "the candidate's predictions are.",
        reference="argmax agreement; KL and JS in nats",
    ),
    ProbeSpec(
        id="repetition",
        title="Degeneration",
        tier="fast",
        cost=2,
        metric="seq_rep_4",
        higher_is_better=False,
        mode="delta_abs",
        warn=0.05,
        fail=0.2,
        hard=True,
        suggests="Greedy generations loop more than the baseline: look for "
        "duplicated data, a too-high LR or a positional change; read samples.",
        explains="Greedy continuations of fixed prompts, scored by seq-rep-4 "
        "(share of repeated 4-grams within a continuation; 1 = stuck in a "
        "loop) and distinct-1/2 (vocabulary variety across continuations).",
        reference="seq-rep-n: Welleck et al. 2019; distinct-n: Li et al. 2016",
        params={"max_new_tokens": 32},
    ),
    ProbeSpec(
        id="fact_recall",
        title="Fact recall (reworded)",
        tier="standard",
        cost=2,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.05,
        fail=0.15,
        hard=False,
        suggests="Worse at using a stated fact when asked in different words: "
        "check context handling (attention/positions) or memory wiring.",
        explains="A fact is stated, then asked with a reworded held-out "
        "question; the model ranks candidate answers by mean token "
        "log-probability. Ties share credit (1/k for k tied at the top), so "
        "chance is 1/candidates.",
        reference="candidate ranking by mean answer log-probability",
        uncertainty="paired_items",
        params={"credit": "fractional_ties"},
    ),
    ProbeSpec(
        id="parametric_recall",
        title="Fact recall (from weights)",
        tier="standard",
        cost=1,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.05,
        fail=0.15,
        hard=False,
        suggests="Recalls fewer trained facts without the fact in context: "
        "check memory wiring (tables, gates, addresses) or capacity changes.",
        explains="Closed-book recall: the fact is NOT stated; the model must "
        "answer from its weights (or memory tables). Items are the training "
        "facts of the run's own withheld-facts manifest (bound to its "
        "`dataset.synthetic_seed`); the manifest's never-trained facts are a "
        "control that should stay at chance (above chance means leakage or a "
        "guessable answer). Only applies to runs trained on the "
        "`withheld_facts` dataset: other runs and reference models are marked "
        "inapplicable, and a run whose facts provenance is missing or "
        "inconsistent is missing evidence. Ranking with fractional tie credit.",
        reference="the run's withheld-facts manifest (dataset.synthetic_seed), "
        "candidate ranking",
        uncertainty="paired_items",
        params={
            "manifest_seed": "dataset.synthetic_seed",
            "credit": "fractional_ties",
        },
    ),
    ProbeSpec(
        id="needle",
        title="Needle in context",
        tier="standard",
        cost=3,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.05,
        fail=0.15,
        hard=False,
        suggests="Retrieval from earlier context degraded: suspect attention "
        "span, positional encoding or sequence-length changes.",
        explains="A code word opens a context of filler text of a few fixed "
        "character lengths; at the end the model must recall it. Lengths are "
        "in characters, not tokens, so every tokenizer (lab runs and "
        "references) is asked the same strings; a context longer than the "
        "model's window pushes the code word out of view, which scores as "
        "chance. Scored by candidate ranking at each length (ties share "
        "credit).",
        reference="needle-in-a-haystack (text-level), candidate ranking",
        uncertainty="paired_items",
        params={"char_lengths": [96, 192, 384], "credit": "fractional_ties"},
    ),
    ProbeSpec(
        id="lm_eval",
        title="Standard tasks (lm-eval)",
        tier="full",
        cost=4,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.02,
        fail=0.05,
        hard=False,
        suggests="Lower on standard zero-shot tasks; at tiny scale most tasks "
        "sit near chance, so treat small moves as noise.",
        explains="A tiny zero-shot slice of public tasks through "
        "EleutherAI's lm-evaluation-harness (optional `lmeval` extra). "
        "Metric definitions are the harness's own.",
        reference="lm-evaluation-harness acc (zero-shot)",
        params={
            "tasks": ["lambada_openai", "hellaswag", "arc_easy", "piqa"],
            "limit": 50,
            "chance": {
                "lambada_openai": 0.0,
                "hellaswag": 0.25,
                "arc_easy": 0.25,
                "piqa": 0.5,
            },
        },
    ),
)
BY_ID = {spec.id: spec for spec in PROBES}

# --- Items -----------------------------------------------------------------

_DEV_PROMPTS = (
    "Once upon a time, a little dog found a ball in the park.",
    "The girl looked at the sky and saw a big red balloon.",
    "Tom and his mom went to the shop to buy some bread.",
    "The cat sat by the window and watched the rain fall.",
)
_HELDOUT_PROMPTS = (
    "One day, a small bird flew over the river and sang a song.",
    "Lily had a blue box. She opened it and found a tiny key.",
    "The boy wanted to play, but his friend was very sleepy.",
    "In the garden there was a tall tree with green leaves.",
    "Sam was happy because it was his birthday today.",
    "The old man walked slowly to the bench and sat down.",
    "A little fish swam in the pond and met a frog.",
    "Mia helped her dad wash the car in the sun.",
)

_DEV_FACTS = (
    ("Pip", "favorite fruit", "apple"),
    ("Rex", "favorite fruit", "pear"),
    ("Zoe", "favorite fruit", "plum"),
    ("Kai", "favorite fruit", "lime"),
    ("Ivy", "favorite toy", "ball"),
    ("Leo", "favorite toy", "doll"),
    ("Nia", "favorite toy", "yo-yo"),
    ("Oz", "favorite toy", "top"),
)
# Held-out facts reuse the withheld-facts diagnostic assets.
_HELDOUT_FACTS = tuple((f.subject, f.relation, f.value) for f in _FACTS)
_DEV_TEMPLATES = (
    "Question: what is {s}'s {r}? Answer:",
    "Tell me {s}'s {r}:",
)
_HELDOUT_TEMPLATES = (
    "Someone asked about the {r} of {s}. It is",
    "If you ask {s} for their {r}, they say",
)

# Held-back final split: new subjects, relations, templates, needles, filler
# and prompts, disjoint from dev and heldout. Never tune against these.
_FINAL_PROMPTS = (
    "The rabbit hopped to the fence and looked at the farm.",
    "Anna found a shiny stone near the lake after lunch.",
    "The baker made a round loaf and gave it to his neighbor.",
    "A tired horse rested under the bridge until the rain stopped.",
    "Max drew a picture of a ship with three white sails.",
    "The kitten chased a leaf all the way down the hill.",
    "Grandma knitted a warm hat for the cold winter morning.",
    "Two friends built a small house out of sticks and mud.",
)
_FINAL_FACTS = (
    ("Ines", "lucky number", "seven"),
    ("Joel", "lucky number", "nine"),
    ("Kira", "lucky number", "four"),
    ("Luca", "lucky number", "six"),
    ("Mona", "home town", "Paris"),
    ("Nico", "home town", "Lima"),
    ("Opal", "home town", "Oslo"),
    ("Pete", "home town", "Cairo"),
)
_FINAL_TEMPLATES = (
    "When asked about the {r} of {s}, the answer was",
    "Here is {s}'s {r}:",
)
_FINAL_PARAMETRIC_TEMPLATE = "As everyone knows, the {r} of {s} is"
_FINAL_NEEDLES = ("rope", "bell", "leaf", "nest", "coin", "fork")
_FINAL_FILLER = (
    "The wind was soft.",
    "A frog sat on a log.",
    "Clouds moved across the hill.",
    "The bus was late again.",
)
_FINAL_OPEN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "klab_final_split_open", default=False
)


class FinalSplitLocked(PermissionError):
    """The held-back ``final`` split was read outside a final verdict."""


@contextlib.contextmanager
def final_verdict_access() -> Iterator[None]:
    """Open the held-back split for one final verdict (``probe --config final``)."""
    token = _FINAL_OPEN.set(True)
    try:
        yield
    finally:
        _FINAL_OPEN.reset(token)


def require_split(split: str) -> str:
    """Validate SPLIT; ``final`` only inside :func:`final_verdict_access`."""
    if split not in SPLITS:
        raise ValueError(f"unknown probe split {split!r}; choose {', '.join(SPLITS)}")
    if split == "final" and not _FINAL_OPEN.get():
        raise FinalSplitLocked(
            "the held-back final split is only read by `klab probe --config final` "
            "(final verdicts); selection loops (try, probe) use dev/heldout"
        )
    return split


def verdict_split() -> str:
    """The split verdicts read now: ``final`` inside a final verdict."""
    return "final" if _FINAL_OPEN.get() else "heldout"


def _pick(split: str, dev: Any, heldout: Any, final: Any) -> Any:
    return {"dev": dev, "heldout": heldout, "final": final}[require_split(split)]


_DEV_NEEDLES = ("lamp", "door", "cup", "shoe")
_HELDOUT_NEEDLES = ("moon", "star", "boat", "cake", "drum", "kite")
_DEV_FILLER = (
    "The sun was warm.",
    "A dog ran by.",
    "The grass was green.",
)
_HELDOUT_FILLER = (
    "The day was calm.",
    "A cat slept on the mat.",
    "Birds sang in the tree.",
    "The road was long.",
)


def fact_items(split: str) -> list[dict[str, Any]]:
    facts = _pick(split, _DEV_FACTS, _HELDOUT_FACTS, _FINAL_FACTS)
    templates = _pick(split, _DEV_TEMPLATES, _HELDOUT_TEMPLATES, _FINAL_TEMPLATES)
    items = []
    for subject, relation, value in facts:
        candidates = sorted({v for s, r, v in facts if r == relation})
        for template in templates:
            items.append(
                {
                    "context": f"{subject}'s {relation} is {value}.",
                    "question": template.format(s=subject, r=relation),
                    "answer": value,
                    "candidates": candidates,
                }
            )
    return items


def parametric_items(
    split: str, *, seed: int, control: bool = False
) -> list[dict[str, Any]]:
    """Closed-book prompts for the withheld-facts manifest of ``seed``.

    ``seed`` is the run's ``dataset.synthetic_seed`` (the seed its
    ``withheld_facts`` training documents were built with; see
    :func:`klab.probes.runner.parametric_binding`). ``heldout`` asks the
    manifest's training facts with its canonical prompt, ``dev`` rewords them,
    ``final`` uses a held-back wording; ``control`` asks the never-trained
    facts instead.
    """
    require_split(split)
    trained, never = split_facts(seed)
    facts = never if control else trained
    return [_parametric_item(fact, split) for fact in facts]


def _parametric_item(fact: Any, split: str) -> dict[str, Any]:
    prompt = _pick(
        split,
        _DEV_TEMPLATES[0].format(s=fact.subject, r=fact.relation),
        fact.prompt(),
        _FINAL_PARAMETRIC_TEMPLATE.format(s=fact.subject, r=fact.relation),
    )
    return {
        "question": prompt,
        "answer": fact.value,
        "candidates": sorted({f.value for f in _FACTS if f.relation == fact.relation}),
    }


def parametric_manifest_sha256(seed: int) -> str:
    return str(diagnostic_manifest(seed)["sha256"])


def needle_split(split: str) -> dict[str, Any]:
    return {
        "needles": list(_pick(split, _DEV_NEEDLES, _HELDOUT_NEEDLES, _FINAL_NEEDLES)),
        "filler": list(_pick(split, _DEV_FILLER, _HELDOUT_FILLER, _FINAL_FILLER)),
        "statement": "Code: {w}.",
        "question": "Code:",
    }


def prompts(split: str) -> list[str]:
    return list(_pick(split, _DEV_PROMPTS, _HELDOUT_PROMPTS, _FINAL_PROMPTS))


def split_payload(split: str) -> dict[str, Any]:
    return {
        "prompts": prompts(split),
        "facts": fact_items(split),
        # The whole fact pool; each run's trained/control split is
        # split_facts(dataset.synthetic_seed), bound at probe time.
        "parametric_pool": [_parametric_item(fact, split) for fact in _FACTS],
        "needle": needle_split(split),
    }


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def suite_identity() -> dict[str, Any]:
    """Name, version and digests of declarations and all item splits.

    Digesting ``final`` opens it only to hash it; nothing about its items
    leaves this function.
    """
    with final_verdict_access():
        splits = {name: _sha(split_payload(name)) for name in SPLITS}
    declarations = [asdict(spec) for spec in PROBES]
    return {
        "name": SUITE_NAME,
        "version": SUITE_VERSION,
        "sha256": _sha(
            {
                "name": SUITE_NAME,
                "version": SUITE_VERSION,
                "probes": declarations,
                "splits": splits,
            }
        ),
        "splits": splits,
        "verdict_split": verdict_split(),
    }


def tiers_through(tier: str) -> tuple[str, ...]:
    if tier not in TIERS:
        raise ValueError(f"unknown probe tier {tier!r}; choose {', '.join(TIERS)}")
    return TIERS[: TIERS.index(tier) + 1]


def ordered(tier: str) -> list[ProbeSpec]:
    """Probes to run through TIER, cheapest tier and cost first."""
    allowed = tiers_through(tier)
    return sorted(
        (spec for spec in PROBES if spec.tier in allowed),
        key=lambda spec: (TIERS.index(spec.tier), spec.cost, PROBES.index(spec)),
    )
