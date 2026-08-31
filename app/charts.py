"""
charts.py — Chart data and specs for the dashboard.

The brief asks for a dashboard "visualizing progress, skill development,
milestones and next recommended actions". Progress and next actions were already
readable as numbers and text; *skill development* is inherently a shape over time
and was previously just a flat list of skill pills, which shows what you have but
not that you are getting anywhere.

Why this is a separate module
-----------------------------
The frame builders here take a session ID and return DataFrames, with no
Streamlit involved. That means the data behind every chart can be asserted in a
self-test, which is not possible once it is entangled with ``st.*`` calls. The
dashboard stays presentational and imports the specs.

Altair is used because it ships with Streamlit, so this adds no dependency.

A note on the x-axis
--------------------
Skill accumulation is plotted against *completion sequence*, not wall-clock time.
In a demo everything is completed within a minute, so a time axis collapses into a
vertical line and shows nothing. Sequence is meaningful at any timescale, and the
tooltip carries the timestamp for anyone who wants it.
"""

from __future__ import annotations

from datetime import datetime

import altair as alt
import pandas as pd

from app.config import load_catalog_map, parse_skills

#: Shared visual language, matching the app's dark theme.
HELD_COLOUR = "#4caf50"
MISSING_COLOUR = "#ef5350"
PLANNED_COLOUR = "#5c6bc0"
COMPLETED_COLOUR = "#26a69a"

_TYPE_ORDER = ["course", "project", "assessment"]


# ── Frame builders (no Streamlit, so they can be tested) ──────────────────────

def skill_growth_frame(session_id: str) -> pd.DataFrame:
    """
    Return cumulative skill and hour growth across completed resources.

    Columns: ``sequence``, ``resource``, ``resource_type``, ``new_skills``,
    ``cumulative_skills``, ``cumulative_hours``, ``completed_at``.

    Only skills *new at that point* count toward the cumulative total. Two
    resources sharing a skill must not make the curve rise twice for one piece of
    knowledge, or the chart would overstate progress.
    """
    from app.db import get_progress

    catalog = load_catalog_map()
    rows = [r for r in get_progress(session_id) if r["status"] == "completed"]

    def _sort_key(row: dict):
        try:
            return (0, datetime.fromisoformat(row["updated_at"]))
        except (TypeError, ValueError):
            return (1, datetime.min)

    rows.sort(key=_sort_key)

    records: list[dict] = []
    seen: set[str] = set()
    total_hours = 0

    for index, row in enumerate(rows, start=1):
        resource = catalog.get(row["course_id"])
        if not resource:
            continue
        skills = parse_skills(resource.get("skills", ""))
        fresh = [s for s in skills if s not in seen]
        seen.update(fresh)
        total_hours += int(resource.get("duration_hours") or 0)

        records.append(
            {
                "sequence": index,
                "resource": str(resource.get("title", row["course_id"]))[:38],
                "resource_type": str(resource.get("resource_type", "course")),
                "new_skills": len(fresh),
                "cumulative_skills": len(seen),
                "cumulative_hours": total_hours,
                "completed_at": row["updated_at"],
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "sequence", "resource", "resource_type", "new_skills",
            "cumulative_skills", "cumulative_hours", "completed_at",
        ],
    )


def gap_composition_frame(gap) -> pd.DataFrame:
    """
    Return the split of target skills into held and still-missing.

    Columns: ``status``, ``count``.
    """
    return pd.DataFrame(
        [
            {"status": "Already have", "count": len(gap.satisfied_skills)},
            {"status": "Still to learn", "count": gap.gap_count},
        ]
    )


def skills_by_category_frame(session_id: str) -> pd.DataFrame:
    """
    Return the number of distinct skills gained per catalog category.

    Columns: ``category``, ``skills``. A skill is credited to the category of the
    first completed resource that taught it, so a skill covered twice does not
    inflate two categories.
    """
    from app.db import get_progress

    catalog = load_catalog_map()
    rows = [r for r in get_progress(session_id) if r["status"] == "completed"]

    seen: set[str] = set()
    counts: dict[str, int] = {}
    for row in rows:
        resource = catalog.get(row["course_id"])
        if not resource:
            continue
        category = str(resource.get("category", "Other"))
        for skill in parse_skills(resource.get("skills", "")):
            if skill in seen:
                continue
            seen.add(skill)
            counts[category] = counts.get(category, 0) + 1

    return pd.DataFrame(
        [{"category": k, "skills": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
        columns=["category", "skills"],
    )


def weekly_plan_frame(schedule: list[dict], completed_ids: set[str]) -> pd.DataFrame:
    """
    Return planned versus completed hours per week.

    Columns: ``week``, ``status``, ``hours``. Long format, because Altair stacks
    from a category column rather than from side-by-side numeric columns.
    """
    per_week: dict[int, dict[str, int]] = {}
    for row in schedule:
        bucket = per_week.setdefault(row["week"], {"Completed": 0, "Remaining": 0})
        key = "Completed" if row["course_id"] in completed_ids else "Remaining"
        bucket[key] += int(row["hours"] or 0)

    records = [
        {"week": week, "status": status, "hours": hours}
        for week, buckets in sorted(per_week.items())
        for status, hours in buckets.items()
        if hours > 0
    ]
    return pd.DataFrame.from_records(records, columns=["week", "status", "hours"])


# ── Chart specs ───────────────────────────────────────────────────────────────

def skill_growth_chart(frame: pd.DataFrame) -> alt.Chart:
    """Cumulative skills acquired across the completion sequence."""
    line = (
        alt.Chart(frame)
        .mark_line(color=COMPLETED_COLOUR, strokeWidth=3, point=True)
        .encode(
            x=alt.X("sequence:O", title="Resources completed, in order"),
            y=alt.Y("cumulative_skills:Q", title="Distinct skills acquired"),
            tooltip=[
                alt.Tooltip("resource:N", title="Resource"),
                alt.Tooltip("resource_type:N", title="Type"),
                alt.Tooltip("new_skills:Q", title="New skills here"),
                alt.Tooltip("cumulative_skills:Q", title="Total skills"),
                alt.Tooltip("cumulative_hours:Q", title="Hours invested"),
                alt.Tooltip("completed_at:N", title="Completed"),
            ],
        )
        .properties(height=240)
    )
    return line


def gap_composition_chart(frame: pd.DataFrame) -> alt.Chart:
    """Held versus missing target skills as a single stacked bar."""
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Skills", stack="normalize"),
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(
                    domain=["Already have", "Still to learn"],
                    range=[HELD_COLOUR, MISSING_COLOUR],
                ),
            ),
            tooltip=[
                alt.Tooltip("status:N", title=None),
                alt.Tooltip("count:Q", title="Skills"),
            ],
        )
        .properties(height=64)
    )


def skills_by_category_chart(frame: pd.DataFrame) -> alt.Chart:
    """Distinct skills gained per category."""
    return (
        alt.Chart(frame)
        .mark_bar(color=COMPLETED_COLOUR)
        .encode(
            y=alt.Y("category:N", title=None, sort="-x"),
            x=alt.X("skills:Q", title="Distinct skills gained"),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("skills:Q", title="Skills"),
            ],
        )
        .properties(height=alt.Step(28))
    )


def weekly_plan_chart(frame: pd.DataFrame) -> alt.Chart:
    """Planned versus completed hours per week."""
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("hours:Q", title="Hours"),
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(
                    domain=["Completed", "Remaining"],
                    range=[COMPLETED_COLOUR, PLANNED_COLOUR],
                ),
            ),
            tooltip=[
                alt.Tooltip("week:O", title="Week"),
                alt.Tooltip("status:N", title=None),
                alt.Tooltip("hours:Q", title="Hours"),
            ],
        )
        .properties(height=220)
    )


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from app.db import (
        _get_connection,
        init_db,
        mark_course_status,
        sync_path_progress,
    )
    from app.skill_gap import SkillGap

    SID = "__selftest_charts__"

    def _cleanup() -> None:
        conn = _get_connection()
        with conn:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (SID,))
        conn.close()

    init_db()
    _cleanup()

    # ── empty state must produce empty frames, not crash ─────────────────────
    empty = skill_growth_frame(SID)
    assert empty.empty and list(empty.columns), empty.columns
    assert skills_by_category_frame(SID).empty
    assert weekly_plan_frame([], set()).empty
    print("empty state: frames build with correct columns")

    # ── seed a path and complete some of it ──────────────────────────────────
    sync_path_progress(SID, [
        {"step_number": 1, "course": {"course_id": "DS005"}},   # SQL, 6h
        {"step_number": 2, "course": {"course_id": "DS001"}},   # Python, 6h
        {"step_number": 3, "course": {"course_id": "AS002"}},   # SQL assessment, 1h
        {"step_number": 4, "course": {"course_id": "PR001"}},   # project, 10h
    ])
    mark_course_status(SID, "DS005", "completed")
    mark_course_status(SID, "DS001", "completed")
    mark_course_status(SID, "AS002", "completed")

    growth = skill_growth_frame(SID)
    print()
    print(growth[["sequence", "resource", "new_skills", "cumulative_skills",
                  "cumulative_hours"]].to_string(index=False))

    assert len(growth) == 3, len(growth)
    assert list(growth["sequence"]) == [1, 2, 3]

    # Cumulative measures may never decrease.
    assert list(growth["cumulative_skills"]) == sorted(growth["cumulative_skills"])
    assert list(growth["cumulative_hours"]) == sorted(growth["cumulative_hours"])
    assert growth["cumulative_hours"].iloc[-1] == 13, growth["cumulative_hours"].iloc[-1]

    # REGRESSION: AS002 validates SQL skills that DS005 already taught, so it must
    # add no new skills. Counting them twice would overstate progress.
    as002_row = growth[growth["resource_type"] == "assessment"].iloc[0]
    assert as002_row["new_skills"] == 0, (
        f"an assessment covering already-taught skills added "
        f"{as002_row['new_skills']} new skills"
    )
    assert as002_row["cumulative_skills"] == growth["cumulative_skills"].iloc[-2], (
        "cumulative skills should not rise for a resource teaching nothing new"
    )

    # The total must equal the number of distinct skills across completions.
    from app.config import load_catalog_map as _cat

    catalog = _cat()
    distinct = set()
    for cid in ("DS005", "DS001", "AS002"):
        distinct.update(parse_skills(catalog[cid]["skills"]))
    assert growth["cumulative_skills"].iloc[-1] == len(distinct), (
        f"{growth['cumulative_skills'].iloc[-1]} != {len(distinct)}"
    )

    # ── category breakdown ───────────────────────────────────────────────────
    categories = skills_by_category_frame(SID)
    print()
    print(categories.to_string(index=False))
    assert not categories.empty
    assert categories["skills"].sum() == len(distinct), (
        "every distinct skill must be credited to exactly one category"
    )

    # ── gap composition ──────────────────────────────────────────────────────
    gap = SkillGap(
        target_skills={"a": 1.0, "b": 1.0, "c": 1.0},
        held_skills=["a"],
        gap_skills={"b": 1.0, "c": 1.0},
        satisfied_skills=["a"],
        confidence=0.7,
    )
    composition = gap_composition_frame(gap)
    assert composition["count"].sum() == 3, composition
    assert composition.loc[composition["status"] == "Already have", "count"].iloc[0] == 1

    # ── weekly plan ──────────────────────────────────────────────────────────
    schedule = [
        {"week": 1, "course_id": "DS005", "hours": 6, "title": "SQL",
         "resource_type": "course", "step_number": 1},
        {"week": 1, "course_id": "DS001", "hours": 6, "title": "Python",
         "resource_type": "course", "step_number": 2},
        {"week": 2, "course_id": "PR001", "hours": 10, "title": "Project",
         "resource_type": "project", "step_number": 3},
    ]
    weekly = weekly_plan_frame(schedule, {"DS005"})
    print()
    print(weekly.to_string(index=False))
    assert weekly["hours"].sum() == 22, weekly["hours"].sum()
    completed_week1 = weekly[(weekly["week"] == 1) & (weekly["status"] == "Completed")]
    assert completed_week1["hours"].iloc[0] == 6, completed_week1
    # Zero-hour buckets are dropped rather than cluttering the chart.
    assert (weekly["hours"] > 0).all()

    # ── every chart spec must build and serialise ───────────────────────────
    specs = {
        "skill_growth": skill_growth_chart(growth),
        "gap_composition": gap_composition_chart(composition),
        "skills_by_category": skills_by_category_chart(categories),
        "weekly_plan": weekly_plan_chart(weekly),
    }
    for name, chart in specs.items():
        rendered = chart.to_dict()
        assert "encoding" in rendered or "layer" in rendered, name
    print(f"\nchart specs: {len(specs)} built and serialised OK")

    _cleanup()
    print("charts.py self-test passed: all assertions OK")
