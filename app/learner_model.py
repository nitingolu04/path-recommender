"""
learner_model.py — Learning preferences and observed learning patterns.

The brief personalises on "needs, interests, learning patterns and goals", and
notes that learners "have different skill levels, interests, career aspirations
and learning preferences". Interests, skills and goals are handled by
``profiling_engine``; the other two live here.

The distinction that matters
----------------------------
**Preferences are stated. Patterns are observed.** They are kept apart because
they carry different weight and can disagree — someone may say they prefer
hands-on work and then skip every project offered. Recording both lets the
dashboard show the discrepancy rather than silently averaging it away, and lets
selection trust behaviour over intention once enough behaviour exists.

Preferences
    style           hands_on | balanced | theory     shifts the project/assessment mix
    hours_per_week  int                              drives the schedule
    pace            intensive | steady | relaxed     bounds how much path to plan

Patterns (derived from the progress and feedback tables)
    completion rate, hours completed, per-type and per-category completion,
    observed hours per week, and which resource types the learner favours or
    avoids in practice.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from app.config import RESOURCE_TYPES, load_catalog_map

# ── Preference vocabulary ─────────────────────────────────────────────────────

STYLES = ("hands_on", "balanced", "theory")
PACES = ("intensive", "steady", "relaxed")

DEFAULT_HOURS_PER_WEEK = 6
MIN_HOURS_PER_WEEK = 1
MAX_HOURS_PER_WEEK = 60

#: Rough hour budgets by pace, used to bound how much path is worth planning.
#: A relaxed learner given a 200-hour plan is being handed a list, not a path.
PACE_HOUR_BUDGET = {"intensive": 160, "steady": 90, "relaxed": 50}

_HANDS_ON_RE = re.compile(
    # "learn by doing" but also "learn best by doing" / "I learn much better by
    # doing", so a couple of intervening words are tolerated.
    r"\blearn\w*(?:\s+\w+){0,2}\s+by\s+doing\b"
    r"|\b(?:hands[\s\-]?on|by\s+building|practical|project[\s\-]?based"
    r"|build\s+(?:things|stuff|projects)|prefer\s+projects|apply\s+it"
    r"|real[\s\-]?world\s+practice)\b",
    re.IGNORECASE,
)

_THEORY_RE = re.compile(
    r"\b(?:theory|theoretical|fundamentals\s+first|from\s+first\s+principles|"
    r"understand\s+(?:deeply|the\s+basics)|prefer\s+reading|"
    r"conceptual|academic|rigorous)\b",
    re.IGNORECASE,
)

# "5 hours a week", "10 hrs/week", "about 3 hours per week"
_HOURS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:\+)?\s*(?:hours?|hrs?)\s*(?:a|per|each|/)\s*week\b",
    re.IGNORECASE,
)

# "2 hours a day" — converted to a weekly figure
_HOURS_PER_DAY_RE = re.compile(
    r"\b(\d{1,2})\s*(?:hours?|hrs?)\s*(?:a|per|each|/)\s*day\b",
    re.IGNORECASE,
)

_INTENSIVE_RE = re.compile(
    r"\b(?:intensive|full[\s\-]?time|as\s+fast\s+as\s+possible|asap|"
    r"quickly|fast[\s\-]?track|bootcamp|cram|urgent|in\s+a\s+hurry)\b",
    re.IGNORECASE,
)

_RELAXED_RE = re.compile(
    r"\b(?:relaxed|slowly|slow\s+and\s+steady|casual|no\s+rush|"
    r"in\s+my\s+spare\s+time|evenings\s+and\s+weekends|part[\s\-]?time|"
    r"alongside\s+(?:work|my\s+job))\b",
    re.IGNORECASE,
)


@dataclass
class LearnerPreferences:
    """How the learner says they want to learn."""

    style: str = "balanced"
    hours_per_week: int = DEFAULT_HOURS_PER_WEEK
    pace: str = "steady"

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            self.style = "balanced"
        if self.pace not in PACES:
            self.pace = "steady"
        # 0 and None both mean "not stated", so they fall back to the default
        # rather than being clamped to one hour a week.
        hours = self.hours_per_week or DEFAULT_HOURS_PER_WEEK
        self.hours_per_week = max(MIN_HOURS_PER_WEEK, min(int(hours), MAX_HOURS_PER_WEEK))

    @property
    def hour_budget(self) -> int:
        """Hours worth planning for, from the stated pace."""
        return PACE_HOUR_BUDGET.get(self.pace, PACE_HOUR_BUDGET["steady"])

    @property
    def reinforcement_order(self) -> tuple[str, ...]:
        """
        Which reinforcement type to offer first at a checkpoint.

        A hands-on learner meets the project first; a theory-minded learner meets
        the assessment first. This is the concrete effect of ``style`` — without
        it the preference would be recorded and never acted on.
        """
        if self.style == "hands_on":
            return ("project", "assessment")
        if self.style == "theory":
            return ("assessment", "project")
        return ("assessment", "project")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "LearnerPreferences":
        data = data or {}
        return cls(
            style=data.get("style", "balanced"),
            hours_per_week=data.get("hours_per_week", DEFAULT_HOURS_PER_WEEK),
            pace=data.get("pace", "steady"),
        )


def extract_preferences(text: str) -> dict:
    """
    Pull stated preferences out of free text.

    Returns only the keys actually found, so a caller can merge over stored
    values without a silent default overwriting something the learner set
    deliberately in the UI.

    >>> extract_preferences("I learn best by doing, about 5 hours a week")
    {'style': 'hands_on', 'hours_per_week': 5}
    """
    found: dict = {}
    if not text or not text.strip():
        return found

    hands_on = bool(_HANDS_ON_RE.search(text))
    theory = bool(_THEORY_RE.search(text))
    # Mentioning both is a stated preference for neither.
    if hands_on and not theory:
        found["style"] = "hands_on"
    elif theory and not hands_on:
        found["style"] = "theory"
    elif hands_on and theory:
        found["style"] = "balanced"

    weekly = _HOURS_RE.search(text)
    daily = _HOURS_PER_DAY_RE.search(text)
    if weekly:
        found["hours_per_week"] = int(weekly.group(1))
    elif daily:
        found["hours_per_week"] = min(int(daily.group(1)) * 7, MAX_HOURS_PER_WEEK)

    intensive = bool(_INTENSIVE_RE.search(text))
    relaxed = bool(_RELAXED_RE.search(text))
    if intensive and not relaxed:
        found["pace"] = "intensive"
    elif relaxed and not intensive:
        found["pace"] = "relaxed"

    return found


# ── Observed patterns ─────────────────────────────────────────────────────────

@dataclass
class LearnerPatterns:
    """
    What the learner actually does, derived from stored progress and feedback.

    Distinct from :class:`LearnerPreferences`, which is what they say they want.
    """

    completed_count: int = 0
    offered_count: int = 0
    hours_completed: int = 0
    completion_rate: float = 0.0
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    too_easy: int = 0
    not_interested: int = 0
    observed_hours_per_week: float | None = None
    active_days: float = 0.0

    @property
    def has_signal(self) -> bool:
        """Whether enough has happened to say anything about behaviour."""
        return self.completed_count > 0 or self.too_easy > 0 or self.not_interested > 0

    @property
    def favoured_types(self) -> list[str]:
        """Resource types the learner completes at an above-average rate."""
        rates = {
            kind: counts["completed"] / counts["offered"]
            for kind, counts in self.by_type.items()
            if counts["offered"]
        }
        if len(rates) < 2:
            return []
        average = sum(rates.values()) / len(rates)
        return sorted([k for k, r in rates.items() if r > average], key=lambda k: -rates[k])

    @property
    def avoided_types(self) -> list[str]:
        """Types offered repeatedly and never completed."""
        return sorted(
            kind
            for kind, counts in self.by_type.items()
            if counts["offered"] >= 2 and counts["completed"] == 0
        )

    @property
    def strongest_category(self) -> str | None:
        if not self.by_category:
            return None
        return max(self.by_category.items(), key=lambda kv: kv[1])[0]

    def summary_lines(self) -> list[str]:
        """Human-readable observations, for the dashboard."""
        lines: list[str] = []
        if not self.has_signal:
            return ["Not enough activity yet to spot a pattern."]

        if self.offered_count:
            lines.append(
                f"Completed {self.completed_count} of {self.offered_count} planned "
                f"resources ({self.completion_rate:.0%}), totalling "
                f"{self.hours_completed} hours."
            )
        if self.observed_hours_per_week is not None:
            lines.append(
                f"Working at roughly {self.observed_hours_per_week:.1f} hours per week "
                f"over {self.active_days:.0f} days."
            )
        if self.strongest_category:
            lines.append(f"Most active category so far: {self.strongest_category}.")
        if self.favoured_types:
            lines.append(
                f"Tends to finish {', '.join(self.favoured_types)} work."
            )
        if self.avoided_types:
            lines.append(
                f"Consistently skips {', '.join(self.avoided_types)} items — worth "
                f"rebalancing the mix."
            )
        if self.too_easy:
            lines.append(
                f"Reported {self.too_easy} resource(s) as too easy, which has raised "
                f"the difficulty being suggested."
            )
        if self.not_interested:
            lines.append(f"Passed on {self.not_interested} resource(s) as not relevant.")
        return lines


def derive_patterns(session_id: str) -> LearnerPatterns:
    """
    Build a :class:`LearnerPatterns` from stored behaviour.

    Reads the ``progress`` and ``feedback`` tables rather than anything the
    learner declared, which is the point: this is the observed half of the model.

    ``observed_hours_per_week`` is only reported once activity spans at least a
    day. Extrapolating a weekly rate from a few minutes of clicking would produce
    a confident and meaningless number.
    """
    from app.db import get_feedback_counts, get_progress

    catalog = load_catalog_map()
    rows = get_progress(session_id)

    patterns = LearnerPatterns()
    patterns.offered_count = len(rows)

    by_type = {kind: {"offered": 0, "completed": 0} for kind in RESOURCE_TYPES}
    timestamps: list[datetime] = []

    for row in rows:
        resource = catalog.get(row["course_id"])
        if not resource:
            continue
        kind = str(resource.get("resource_type", "course"))
        by_type.setdefault(kind, {"offered": 0, "completed": 0})
        by_type[kind]["offered"] += 1

        if row["status"] != "completed":
            continue

        patterns.completed_count += 1
        by_type[kind]["completed"] += 1
        patterns.hours_completed += int(resource.get("duration_hours") or 0)

        category = str(resource.get("category", "Unknown"))
        patterns.by_category[category] = patterns.by_category.get(category, 0) + 1

        try:
            timestamps.append(datetime.fromisoformat(row["updated_at"]))
        except (TypeError, ValueError):
            pass

    patterns.by_type = {k: v for k, v in by_type.items() if v["offered"]}
    patterns.completion_rate = (
        patterns.completed_count / patterns.offered_count if patterns.offered_count else 0.0
    )

    counts = get_feedback_counts(session_id)
    patterns.too_easy = counts.get("too_easy", 0)
    patterns.not_interested = counts.get("not_interested", 0)

    if len(timestamps) >= 2:
        span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        patterns.active_days = span_days
        if span_days >= 1.0:
            patterns.observed_hours_per_week = patterns.hours_completed / (span_days / 7)

    return patterns


# ── Scheduling ────────────────────────────────────────────────────────────────

def schedule_path(path: list[dict], hours_per_week: int) -> list[dict]:
    """
    Assign each step in a path to a week, given a weekly hour budget.

    Turns an ordered list into a plan with dates attached, which is what makes
    ``hours_per_week`` more than a stored number. A resource longer than a single
    week's budget simply occupies consecutive weeks rather than being split.

    Returns
    -------
    list[dict]
        ``{step_number, course_id, title, resource_type, hours, week, cumulative_hours}``
        for each resource step. Milestone-only entries are skipped.
    """
    hours_per_week = max(MIN_HOURS_PER_WEEK, int(hours_per_week or DEFAULT_HOURS_PER_WEEK))

    schedule: list[dict] = []
    cumulative = 0
    for step in path:
        resource = step.get("course")
        if not resource:
            continue
        hours = max(int(resource.get("duration_hours") or 0), 0)
        cumulative += hours
        # Week 1 covers hours 1..hours_per_week, so ceil on the running total.
        week = max(1, -(-cumulative // hours_per_week))
        schedule.append(
            {
                "step_number": step.get("step_number"),
                "course_id": resource.get("course_id"),
                "title": resource.get("title"),
                "resource_type": resource.get("resource_type", "course"),
                "hours": hours,
                "week": week,
                "cumulative_hours": cumulative,
            }
        )
    return schedule


def weeks_required(path: list[dict], hours_per_week: int) -> int:
    """Total weeks to finish a path at the given weekly budget."""
    schedule = schedule_path(path, hours_per_week)
    return max((row["week"] for row in schedule), default=0)


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from app.db import (
        _get_connection,
        init_db,
        log_feedback,
        mark_course_status,
        sync_path_progress,
    )

    # ── preference defaults and clamping ─────────────────────────────────────
    default = LearnerPreferences()
    assert default.style == "balanced" and default.pace == "steady"
    assert default.hours_per_week == DEFAULT_HOURS_PER_WEEK

    assert LearnerPreferences(style="nonsense").style == "balanced"
    assert LearnerPreferences(pace="whenever").pace == "steady"
    # 0 and None mean "not stated" and fall back to the default; a nonsensical
    # negative is clamped; an unrealistic maximum is capped.
    assert LearnerPreferences(hours_per_week=0).hours_per_week == DEFAULT_HOURS_PER_WEEK
    assert LearnerPreferences(hours_per_week=None).hours_per_week == DEFAULT_HOURS_PER_WEEK
    assert LearnerPreferences(hours_per_week=-5).hours_per_week == MIN_HOURS_PER_WEEK
    assert LearnerPreferences(hours_per_week=500).hours_per_week == MAX_HOURS_PER_WEEK
    assert LearnerPreferences.from_dict(None).style == "balanced"
    assert LearnerPreferences.from_dict({"style": "theory"}).style == "theory"

    # Style must actually change something, not just be stored.
    assert LearnerPreferences(style="hands_on").reinforcement_order[0] == "project"
    assert LearnerPreferences(style="theory").reinforcement_order[0] == "assessment"
    assert LearnerPreferences(pace="intensive").hour_budget > \
        LearnerPreferences(pace="relaxed").hour_budget

    # ── preference extraction from prose ────────────────────────────────────
    cases = [
        ("I learn best by doing, about 5 hours a week",
         {"style": "hands_on", "hours_per_week": 5}),
        ("I prefer theory and want to understand deeply",
         {"style": "theory"}),
        ("I want to fast-track this, 20 hours per week",
         {"hours_per_week": 20, "pace": "intensive"}),
        ("Just casually alongside my job, 3 hrs/week",
         {"hours_per_week": 3, "pace": "relaxed"}),
        ("I can do 2 hours a day", {"hours_per_week": 14}),
        ("I want to become a data analyst", {}),
        ("", {}),
    ]
    for text, expected in cases:
        got = extract_preferences(text)
        assert got == expected, f"{text!r} -> {got}, expected {expected}"

    # Claiming both styles resolves to balanced rather than picking one.
    assert extract_preferences("hands-on but I also like theory")["style"] == "balanced"
    print(f"preference extraction: {len(cases) + 1} cases OK")

    # ── scheduling ───────────────────────────────────────────────────────────
    fake_path = [
        {"step_number": 1, "course": {"course_id": "A", "title": "A",
                                      "resource_type": "course", "duration_hours": 6}},
        {"step_number": 1, "course": None, "milestone": "checkpoint"},
        {"step_number": 2, "course": {"course_id": "B", "title": "B",
                                      "resource_type": "course", "duration_hours": 6}},
        {"step_number": 3, "course": {"course_id": "C", "title": "C",
                                      "resource_type": "project", "duration_hours": 10}},
    ]
    schedule = schedule_path(fake_path, hours_per_week=6)
    assert len(schedule) == 3, "milestone entries must be skipped"
    assert [r["week"] for r in schedule] == [1, 2, 4], [r["week"] for r in schedule]
    assert schedule[-1]["cumulative_hours"] == 22
    assert weeks_required(fake_path, 6) == 4
    # A bigger weekly budget must finish sooner.
    assert weeks_required(fake_path, 22) == 1
    assert weeks_required(fake_path, 11) == 2
    assert weeks_required([], 6) == 0
    # A guard against division by zero rather than a crash.
    assert weeks_required(fake_path, 0) > 0
    print("scheduling: 8 assertions OK")

    # ── observed patterns ────────────────────────────────────────────────────
    SID = "__selftest_learner__"

    def _cleanup() -> None:
        conn = _get_connection()
        with conn:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (SID,))
        conn.close()

    init_db()
    _cleanup()

    empty = derive_patterns(SID)
    assert not empty.has_signal
    assert empty.summary_lines() == ["Not enough activity yet to spot a pattern."]

    # A path with two courses, one project and one assessment.
    sync_path_progress(SID, [
        {"step_number": 1, "course": {"course_id": "DS005"}},   # course
        {"step_number": 2, "course": {"course_id": "DS001"}},   # course
        {"step_number": 3, "course": {"course_id": "PR001"}},   # project
        {"step_number": 4, "course": {"course_id": "AS002"}},   # assessment
    ])
    mark_course_status(SID, "DS005", "completed")
    mark_course_status(SID, "DS001", "completed")
    log_feedback(SID, "PR001", "not_interested")

    patterns = derive_patterns(SID)
    print()
    for line in patterns.summary_lines():
        print(f"  - {line}")

    assert patterns.has_signal
    assert patterns.completed_count == 2, patterns.completed_count
    assert patterns.offered_count == 4, patterns.offered_count
    assert abs(patterns.completion_rate - 0.5) < 1e-9
    assert patterns.hours_completed == 12, patterns.hours_completed  # two 6h courses
    assert patterns.by_type["course"] == {"offered": 2, "completed": 2}
    assert patterns.by_type["project"] == {"offered": 1, "completed": 0}
    assert patterns.not_interested == 1
    assert patterns.strongest_category == "Data Science", patterns.strongest_category

    # Courses are finished, projects and assessments are not, so the behavioural
    # signal must reflect that.
    assert "course" in patterns.favoured_types, patterns.favoured_types
    assert "project" not in patterns.favoured_types

    # A single offered-and-skipped item is not yet enough to call it avoidance.
    assert patterns.avoided_types == [], patterns.avoided_types
    sync_path_progress(SID, [{"step_number": 5, "course": {"course_id": "PR002"}}])
    patterns2 = derive_patterns(SID)
    assert "project" in patterns2.avoided_types, patterns2.avoided_types
    print("\n  after a second skipped project -> avoided_types="
          f"{patterns2.avoided_types}")

    # Too little elapsed time to claim a weekly rate.
    assert patterns.observed_hours_per_week is None, (
        "a weekly rate from minutes of activity would be meaningless"
    )

    _cleanup()
    print("\nlearner_model.py self-test passed: all assertions OK")
