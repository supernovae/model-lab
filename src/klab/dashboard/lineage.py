"""Runs grouped into lineages, each attempt with a verdict from its recorded facts.

A long run is rarely one process. Every recovery is a new run whose parent is the run
it resumed from, so one training job appears in the store as a chain of attempts. This
module groups them and says, for each attempt, how it ended and why, using only what
the trainer recorded: its status, its ``run_failed`` or ``run_interrupted`` reason,
whether it left a resumable checkpoint, the layout changes of its recovery, and its last
heartbeat. Nothing here reads a process or a file, and nothing is inferred from a
stored ``running`` status alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from klab.dashboard.queries import RunFacts, RunRecord

# A heartbeat is written about every 30 seconds while a run trains.
LIVE_SECONDS = 150
# A recovery this much slower than the run it resumed is worth a warning.
SLOWER_THAN_PARENT = 0.85
# Rates of very short attempts are dominated by start-up.
MIN_UPDATES_FOR_RATE = 5

GOOD, WARN, BAD, LIVE, NEUTRAL = "pass", "warn", "fail", "info", "skipped"

_REASONS = (
    (
        "device memory spill",
        "Memory spill",
        "Host memory fell past the spill guard's limit.",
    ),
    ("out of memory", "Out of memory", "A device ran out of memory."),
    (
        "nonfinite",
        "Non-finite loss",
        "The loss or the gradients became NaN or infinite.",
    ),
    (
        "non-finite",
        "Non-finite loss",
        "The loss or the gradients became NaN or infinite.",
    ),
    ("storage preflight", "Disk full", "The workspace did not have room for the run."),
    (
        "differs from the checkpoint",
        "Corrupt restore",
        "A restored shard did not match the checkpoint.",
    ),
    ("no progress", "Stalled", "The run stopped making progress."),
)


@dataclass(frozen=True)
class Verdict:
    level: str  # a key of klab.dashboard.ui.STATUS_COLOR
    label: str
    why: str


@dataclass(frozen=True)
class Attempt:
    record: RunRecord
    facts: RunFacts
    verdict: Verdict
    notes: tuple[str, ...] = ()
    on_main_line: bool = True

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def last_step(self) -> int | None:
        return final_count(self.facts, "completed_steps", self.facts.last_step)

    @property
    def last_tokens(self) -> int | None:
        return final_count(self.facts, "completed_targets", self.facts.last_tokens)

    @property
    def steps_lost(self) -> int | None:
        """Updates after the last verified checkpoint, which a recovery repeats."""
        if self.last_step is None or self.facts.last_checkpoint_step is None:
            return None
        return max(0, self.last_step - self.facts.last_checkpoint_step)


@dataclass(frozen=True)
class Lineage:
    root_id: str
    name: str
    attempts: tuple[Attempt, ...]  # oldest first
    target_tokens: int | None
    target_steps: int | None

    @property
    def tip(self) -> Attempt:
        """The attempt that got furthest; ties go to the most recently updated."""
        return max(
            self.attempts,
            key=lambda a: (a.last_step or -1, a.record.updated_at, a.run_id),
        )

    @property
    def main_line(self) -> tuple[Attempt, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.on_main_line)

    @property
    def tokens(self) -> int | None:
        return self.tip.last_tokens

    @property
    def best_rate(self) -> float | None:
        rates = [
            a.facts.tokens_per_second
            for a in self.attempts
            if a.facts.tokens_per_second and a.facts.updates >= MIN_UPDATES_FOR_RATE
        ]
        return max(rates) if rates else None

    @property
    def progress(self) -> float | None:
        if not self.target_tokens or self.tokens is None:
            return None
        return min(1.0, self.tokens / self.target_tokens)

    @property
    def live(self) -> bool:
        return any(attempt.verdict.level == LIVE for attempt in self.attempts)

    @property
    def last_activity(self) -> float:
        """When an attempt last wrote a heartbeat, as epoch seconds.

        Not the row's ``updated_at``: settling a stale run long after it died updates
        that, and would list a week-old run as the most recent.
        """
        times = []
        for attempt in self.attempts:
            observed = _parse(attempt.facts.heartbeat_at)
            if observed is not None:
                times.append(observed.timestamp())
        if (
            not times
        ):  # nothing ever wrote a heartbeat: fall back to the rows themselves
            for attempt in self.attempts:
                recorded = _parse(attempt.record.updated_at.replace(" ", "T"))
                if recorded is not None:
                    times.append(recorded.timestamp())
        return max(times, default=0.0)

    @property
    def trained_seconds(self) -> float:
        return sum(
            (a.facts.last_wall_time or 0.0) - (a.facts.first_wall_time or 0.0)
            for a in self.attempts
            if a.facts.first_wall_time is not None
        )


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def heartbeat_age(facts: RunFacts, now: datetime) -> float | None:
    observed = _parse(facts.heartbeat_at)
    return None if observed is None else (now - observed).total_seconds()


def final_count(facts: RunFacts, key: str, fallback: int | None) -> int | None:
    """The trainer's own final count; metrics are only logged every few updates."""
    final = _event(facts, "runtime_final_observation") or {}
    value = final.get(key)
    if isinstance(value, int) and (fallback is None or value >= fallback):
        return value
    return fallback


def _event(facts: RunFacts, kind: str) -> dict[str, object] | None:
    for name, _, payload in reversed(facts.events):
        if name == kind:
            return payload
    return None


def _reason(facts: RunFacts, kind: str) -> str:
    payload = _event(facts, kind) or {}
    reason = payload.get("reason")
    return str(reason) if isinstance(reason, str) else ""


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def verdict(record: RunRecord, facts: RunFacts, now: datetime) -> Verdict:
    """How this attempt ended, from its recorded status, events and heartbeat."""
    status = record.status.lower()
    last_step = final_count(facts, "completed_steps", facts.last_step)
    made_progress = (
        facts.first_step is not None
        and last_step is not None
        and last_step > facts.first_step
    )
    if status == "running":
        age = heartbeat_age(facts, now)
        if (
            age is not None
            and age <= LIVE_SECONDS
            and facts.heartbeat_event
            not in {
                "failed",
                "finished",
            }
        ):
            return Verdict(LIVE, "Training", f"Last heartbeat {_age(age)} ago.")
        seen = (
            f"its last heartbeat was {_age(age)} ago"
            if age is not None
            else "it never wrote a heartbeat"
        )
        return Verdict(
            WARN,
            "Not verified",
            f"Recorded as running, but {seen}. The process probably died without "
            "recording its end; `klab workspace settle` marks such runs interrupted.",
        )
    if status == "completed":
        return Verdict(GOOD, "Completed", "The run reached the end of its budget.")
    if status == "failed":
        reason = _reason(facts, "run_failed")
        for needle, label, meaning in _REASONS:
            if needle in reason.lower():
                return Verdict(BAD, label, f"{meaning} Recorded reason: {reason}")
        return Verdict(
            BAD, "Failed", reason or "The run failed without a recorded reason."
        )
    if status in {"interrupted", "cancelled"}:
        reason = _reason(facts, "run_interrupted")
        resumable = _event(facts, "resumable") is not None
        if not made_progress:
            return Verdict(
                NEUTRAL,
                "No progress",
                "Stopped before it completed an update"
                + (f" ({reason})." if reason else "."),
            )
        if resumable and reason in {"signal", "cancel"}:
            return Verdict(
                GOOD,
                "Paused cleanly",
                "Stopped on request and left a resumable checkpoint at its last step.",
            )
        return Verdict(
            WARN,
            "Interrupted",
            (
                f"Stopped ({reason})"
                if reason
                else "Its process ended without recording why"
            )
            + (
                "; a resumable checkpoint exists."
                if resumable
                else "; no resumable checkpoint was recorded."
            ),
        )
    return Verdict(
        NEUTRAL, record.status or "Unknown", "No outcome is recorded for this run."
    )


def _notes(
    record: RunRecord, facts: RunFacts, parent: RunFacts | None
) -> tuple[str, ...]:
    notes: list[str] = []
    transition = _event(facts, "runtime_optimization_transition") or {}
    changes = transition.get("changes")
    if isinstance(changes, list):
        for change in changes:
            if isinstance(change, dict):
                notes.append(
                    f"Recovered with {change.get('field')}: "
                    f"{change.get('previous')!r} → {change.get('requested')!r}."
                )
    if _event(facts, "runtime_drift") is not None:
        notes.append(
            "Recovered under a different Lab version or runtime (best-effort resume)."
        )
    if (
        parent is not None
        and parent.tokens_per_second
        and facts.tokens_per_second
        and facts.updates >= MIN_UPDATES_FOR_RATE
        and parent.updates >= MIN_UPDATES_FOR_RATE
        and facts.tokens_per_second < SLOWER_THAN_PARENT * parent.tokens_per_second
    ):
        notes.append(
            f"Slower than the run it resumed: {facts.tokens_per_second:,.0f} against "
            f"{parent.tokens_per_second:,.0f} tokens per second."
        )
    return tuple(notes)


def _target(record: RunRecord, key: str) -> int | None:
    training = record.config.get("training")
    value = training.get(key) if isinstance(training, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def lineages(
    records: list[RunRecord] | tuple[RunRecord, ...],
    facts: dict[str, RunFacts],
    now: datetime | None = None,
) -> list[Lineage]:
    """Group runs by the root of their parent chain; newest activity first."""
    now = now or datetime.now(UTC)
    by_id = {record.run_id: record for record in records}

    def root_of(record: RunRecord) -> str:
        seen = {record.run_id}
        while record.parent_run_id in by_id and record.parent_run_id not in seen:
            record = by_id[record.parent_run_id]
            seen.add(record.run_id)
        return record.run_id

    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        groups.setdefault(root_of(record), []).append(record)
    result: list[Lineage] = []
    for root_id, members in groups.items():
        members.sort(key=lambda r: (r.created_at or r.updated_at, r.run_id))
        tip = max(
            members,
            key=lambda r: (
                final_count(
                    facts.get(r.run_id, RunFacts()),
                    "completed_steps",
                    facts.get(r.run_id, RunFacts()).last_step,
                )
                or -1,
                r.updated_at,
                r.run_id,
            ),
        )
        main: set[str] = set()
        cursor: RunRecord | None = tip
        while cursor is not None and cursor.run_id not in main:
            main.add(cursor.run_id)
            cursor = by_id.get(cursor.parent_run_id or "")
        attempts = []
        for record in members:
            own = facts.get(record.run_id, RunFacts())
            parent = facts.get(record.parent_run_id) if record.parent_run_id else None
            outcome = verdict(record, own, now)
            notes = _notes(record, own, parent)
            if outcome.level == GOOD and any(
                note.startswith("Slower") for note in notes
            ):
                outcome = Verdict(WARN, outcome.label, outcome.why)
            attempts.append(
                Attempt(record, own, outcome, notes, on_main_line=record.run_id in main)
            )
        root = by_id[root_id]
        result.append(
            Lineage(
                root_id=root_id,
                name=root.name,
                attempts=tuple(attempts),
                target_tokens=_target(tip, "max_tokens"),
                target_steps=_target(tip, "max_steps"),
            )
        )
    result.sort(key=lambda item: (item.live, item.last_activity), reverse=True)
    return result


def stitch(
    rows: list[dict[str, object]], main_line: tuple[Attempt, ...]
) -> list[dict[str, object]]:
    """One series per metric along the main line, without the steps a recovery redid.

    A recovery starts from its parent's last checkpoint, so the parent's updates after
    that step were repeated. Each attempt therefore contributes its points up to the
    step where the next attempt on the main line began.
    """
    ends: dict[str, int | None] = {}
    for attempt, following in zip(main_line, (*main_line[1:], None), strict=True):
        ends[attempt.run_id] = following.facts.first_step if following else None
    kept = [
        row
        for row in rows
        if row["run_id"] in ends
        and (
            ends[str(row["run_id"])] is None
            or int(row["step"]) <= ends[str(row["run_id"])]
        )  # type: ignore[arg-type,operator]
    ]
    kept.sort(key=lambda row: (str(row["name"]), int(row["step"])))  # type: ignore[arg-type]
    return kept
