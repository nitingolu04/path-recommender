"""
dashboard.py — Streamlit progress dashboard.

Shows:
    - Completion percentage across the path
    - Skills accumulated from completed courses
    - The next recommended action
    - Per-course controls: mark done, "too easy", "not interested"

Feedback semantics
------------------
The two feedback buttons do different things, which was not previously the case:

    "Too Easy"        drops the course AND counts toward promoting the user's
                      difficulty level, so subsequent suggestions get harder.
    "Not Interested"  drops the course only, leaving difficulty untouched.

Both then trigger a re-rank through ``pipeline.build_learning_path()``, which
also refreshes the persisted progress rows.
"""

from __future__ import annotations

import streamlit as st

from app.config import parse_skills

_DIFFICULTY_ICON = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}


def render_dashboard() -> None:
    """Render the progress dashboard tab."""
    from app.db import (
        get_completed_course_ids,
        get_feedback_counts,
        init_db,
        log_feedback,
        mark_course_status,
    )
    from app.pipeline import build_learning_path, resolve_experience_level

    init_db()
    session_id = st.session_state.get("session_id")
    path = st.session_state.get("path", [])
    profile = st.session_state.get("profile")
    result = st.session_state.get("path_result")

    if not path or not profile:
        st.info(
            "Enter your learning goal in the **Chat** tab and your dashboard "
            "will appear here."
        )
        return

    course_steps = [s for s in path if s["course"]]
    completed_ids = get_completed_course_ids(session_id)
    num_completed = sum(1 for s in course_steps if s["course"]["course_id"] in completed_ids)
    total = len(course_steps)
    pct = int(num_completed / total * 100) if total else 0

    effective_level, escalated = resolve_experience_level(session_id, profile)
    feedback_counts = get_feedback_counts(session_id)

    hours_total = sum(int(s["course"].get("duration_hours") or 0) for s in course_steps)
    hours_done = sum(
        int(s["course"].get("duration_hours") or 0)
        for s in course_steps
        if s["course"]["course_id"] in completed_ids
    )

    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#0f3460,#533483);
                    border-radius:16px;padding:24px;margin-bottom:24px;'>
            <h2 style='color:#fff;margin:0 0 4px;'>📊 Your Learning Dashboard</h2>
            <p style='color:#c0c0e0;margin:0;'>Track progress and refine your path.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Completed", f"{num_completed} / {total}")
    col2.metric("📈 Progress", f"{pct}%")
    col3.metric("⏱️ Hours", f"{hours_done} / {hours_total}")
    col4.metric(
        "🎯 Level",
        effective_level.capitalize(),
        delta="raised by feedback" if escalated else None,
        delta_color="normal" if escalated else "off",
    )

    st.progress(pct / 100, text=f"{pct}% of your path completed")

    if escalated:
        st.caption(
            f"You reported {feedback_counts.get('too_easy', 0)} course(s) as too easy, "
            f"so your level was raised from **{profile.get('experience_level')}** to "
            f"**{effective_level}** and harder material is now being surfaced."
        )
    st.divider()

    # ── Skill gap ─────────────────────────────────────────────────────────────
    # The brief names skill-gap identification as a core mechanism, so it gets
    # shown rather than left implicit in the ranking.
    if result is not None and result.gap.target_count:
        gap = result.gap
        st.markdown("### 🎯 Skill Gap")

        gap_col1, gap_col2, gap_col3 = st.columns(3)
        gap_col1.metric("Skills this goal needs", gap.target_count)
        gap_col2.metric("You already have", len(gap.satisfied_skills))
        gap_col3.metric("Still to learn", gap.gap_count)

        from app.charts import gap_composition_chart, gap_composition_frame

        st.altair_chart(
            gap_composition_chart(gap_composition_frame(gap)),
            width="stretch",
        )
        st.caption(
            f"{gap.coverage_ratio:.0%} of the skills this goal needs are already "
            f"covered. This path closes "
            f"{result.coverage.get('covered_ratio', 0):.0%} of what's left, across "
            f"{result.total_hours} hours of study."
        )

        remaining = [s for s, _ in gap.ranked_gap()]
        if remaining:
            st.markdown("**Still missing, most important first:**")
            st.markdown(
                " ".join(
                    f"<span style='background:#3a1e1e;color:#ffb3b3;padding:4px 10px;"
                    f"border-radius:999px;margin:3px;display:inline-block;"
                    f"font-size:0.82rem;'>{skill}</span>"
                    for skill in remaining[:14]
                ),
                unsafe_allow_html=True,
            )
        uncovered = result.coverage.get("uncovered") or []
        if uncovered:
            st.caption(
                f"Not covered by this path: {', '.join(uncovered[:8])}"
                f"{'...' if len(uncovered) > 8 else ''}. "
                f"No course in the catalog teaches these alongside your other needs."
            )
        st.divider()

    # ── Schedule ──────────────────────────────────────────────────────────────
    # Turns the learner's stated weekly hours into a real plan, which is what
    # makes that preference more than a stored number.
    if result is not None and result.schedule:
        prefs = result.preferences
        st.markdown("### 🗓️ Your Schedule")
        st.caption(
            f"At **{prefs.hours_per_week} hours a week** this path takes about "
            f"**{result.weeks} weeks**. Change your hours in the Chat tab under "
            f"\"How you like to learn\"."
        )

        from app.charts import weekly_plan_chart, weekly_plan_frame

        weekly = weekly_plan_frame(result.schedule, completed_ids)
        if not weekly.empty:
            st.altair_chart(weekly_plan_chart(weekly), width="stretch")

        by_week: dict[int, list[dict]] = {}
        for row in result.schedule:
            by_week.setdefault(row["week"], []).append(row)

        for week in sorted(by_week)[:8]:
            rows = by_week[week]
            hours = sum(r["hours"] for r in rows)
            done = all(r["course_id"] in completed_ids for r in rows)
            titles = ", ".join(
                f"{'✅ ' if r['course_id'] in completed_ids else ''}{r['title']}"
                for r in rows
            )
            marker = "✅" if done else f"{hours}h"
            st.markdown(
                f"<div style='display:flex;gap:12px;padding:6px 0;"
                f"border-bottom:1px solid #23233a;'>"
                f"<span style='color:#7ec8e3;min-width:80px;font-weight:600;'>"
                f"Week {week}</span>"
                f"<span style='color:#8a8aa8;min-width:44px;'>{marker}</span>"
                f"<span style='color:#d0d0e8;'>{titles}</span></div>",
                unsafe_allow_html=True,
            )
        if len(by_week) > 8:
            st.caption(f"...and {len(by_week) - 8} more week(s).")
        st.divider()

    # ── Skills gained ─────────────────────────────────────────────────────────
    gained: set[str] = set()
    for step in course_steps:
        course = step["course"]
        if course["course_id"] in completed_ids:
            gained.update(parse_skills(course.get("skills", "")))

    st.markdown("### 🧠 Skill Development")
    if gained:
        from app.charts import (
            skill_growth_chart,
            skill_growth_frame,
            skills_by_category_chart,
            skills_by_category_frame,
        )

        growth = skill_growth_frame(session_id)
        categories = skills_by_category_frame(session_id)

        if len(growth) >= 2:
            # Two points minimum, or a "growth" curve is a single dot.
            col_growth, col_cat = st.columns([3, 2])
            with col_growth:
                st.caption("Distinct skills acquired as you work through the path")
                st.altair_chart(skill_growth_chart(growth), width="stretch")
            with col_cat:
                st.caption("Where those skills came from")
                st.altair_chart(
                    skills_by_category_chart(categories), width="stretch"
                )
        else:
            st.caption(
                f"**{len(gained)} skills** so far. Complete one more resource and a "
                f"growth curve appears here."
            )

        with st.expander(f"All {len(gained)} skills gained", expanded=False):
            st.markdown(
                " ".join(
                    f"<span style='background:#1e3a5f;color:#7ec8e3;padding:4px 10px;"
                    f"border-radius:999px;margin:4px;display:inline-block;"
                    f"font-size:0.85rem;'>{skill}</span>"
                    for skill in sorted(gained)
                ),
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "Mark a resource as done and your skill growth will be charted here."
        )
    st.divider()

    # ── Next action ───────────────────────────────────────────────────────────
    st.markdown("### 🚀 Next Recommended Action")
    next_step = next(
        (s for s in course_steps if s["course"]["course_id"] not in completed_ids), None
    )
    if next_step:
        course = next_step["course"]
        st.success(
            f"**Step {next_step['step_number']}: {course['title']}**  \n"
            f"{course['difficulty_level'].capitalize()} · {course['category']}\n\n"
            f"{course.get('description', '')[:180]}..."
        )
    else:
        st.balloons()
        st.success("You've completed the whole path. Set a new goal in the Chat tab.")
    st.divider()

    # ── Observed learning patterns ────────────────────────────────────────────
    # The brief personalises on "learning patterns". These are derived from what
    # the learner actually did, deliberately kept separate from what they said
    # they preferred, so a mismatch between the two is visible rather than averaged
    # away.
    if result is not None and result.patterns.has_signal:
        patterns = result.patterns
        st.markdown("### 🔍 Your Learning Patterns")
        for line in patterns.summary_lines():
            st.markdown(f"- {line}")

        stated = result.preferences.style
        if stated == "hands_on" and "project" in patterns.avoided_types:
            st.warning(
                "You said you prefer hands-on work but have been skipping the "
                "projects. Either the projects are landing too early, or the "
                "preference is worth changing."
            )
        elif patterns.by_type:
            mix = ", ".join(
                f"{kind}s {counts['completed']}/{counts['offered']}"
                for kind, counts in sorted(patterns.by_type.items())
            )
            st.caption(f"Completion by type: {mix}")
        st.divider()

    # ── Full path with controls ───────────────────────────────────────────────
    st.markdown("### 📋 Full Learning Path")
    next_id = next_step["course"]["course_id"] if next_step else None

    for step in path:
        if step["milestone"]:
            st.markdown(
                f"<div style='background:#1a2e1a;border-left:4px solid #4caf50;"
                f"padding:10px 16px;border-radius:8px;margin:12px 0;color:#a5d6a7;'>"
                f"{step['milestone']}</div>",
                unsafe_allow_html=True,
            )
            continue

        course = step["course"]
        course_id = course["course_id"]
        is_done = course_id in completed_ids
        level = str(course.get("difficulty_level", "beginner")).lower()

        label = (
            f"{'✅' if is_done else '⬜'} Step {step['step_number']} — "
            f"{course['title']} {_DIFFICULTY_ICON.get(level, '')}"
        )
        if step.get("is_prerequisite_filler"):
            label += " · prerequisite"

        with st.expander(label, expanded=(not is_done and course_id == next_id)):
            col_desc, col_actions = st.columns([3, 1])

            with col_desc:
                st.markdown(f"**Category:** {course.get('category', '—')}")
                st.markdown(f"**Skills:** {course.get('skills', '—')}")
                if course.get("prerequisites"):
                    st.markdown(f"**Prerequisites:** {course['prerequisites']}")
                st.markdown(f"_{course.get('description', '')}_")

                if step.get("is_prerequisite_filler"):
                    st.caption(
                        "Included as groundwork for a later course, so it has no "
                        "match score of its own."
                    )
                else:
                    st.caption(
                        f"Match {step['similarity_score']:.0%} "
                        f"(semantic {course.get('semantic_score', 0.0):.0%}, "
                        f"level fit {(course.get('level_fit') or 0.0):.0%})"
                    )

            with col_actions:
                if is_done:
                    if st.button("↩️ Undo", key=f"undo_{course_id}", width="stretch"):
                        mark_course_status(session_id, course_id, "not_started")
                        st.rerun()
                else:
                    if st.button("✅ Mark Done", key=f"done_{course_id}", width="stretch"):
                        mark_course_status(session_id, course_id, "completed")
                        st.rerun()

                if st.button(
                    "😓 Too Easy",
                    key=f"easy_{course_id}",
                    width="stretch",
                    help="Removes this course and raises your difficulty level.",
                ):
                    log_feedback(session_id, course_id, "too_easy")
                    st.session_state.path_result = build_learning_path(session_id, profile)
                    st.session_state.path = st.session_state.path_result.path
                    st.rerun()

                if st.button(
                    "🚫 Not Interested",
                    key=f"skip_{course_id}",
                    width="stretch",
                    help="Removes this course without changing your difficulty level.",
                ):
                    log_feedback(session_id, course_id, "not_interested")
                    st.session_state.path_result = build_learning_path(session_id, profile)
                    st.session_state.path = st.session_state.path_result.path
                    st.rerun()
