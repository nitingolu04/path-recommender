"""
chat_interface.py — Streamlit component for natural-language goal entry.

Responsibilities:
    - Render a chat-style input where the user describes their learning goal.
    - Keep conversation history in ``st.session_state``.
    - Hand the text to ``pipeline.run_full_pipeline()`` and stash the resulting
      profile and path in session state for the other tabs to read.
    - Echo back the structured profile so the user can see what was understood.

The heavy lifting lives in ``app.pipeline``; this module is presentation only.
"""

from __future__ import annotations

import uuid

import streamlit as st

EXAMPLE_GOALS = [
    "I want to become a data analyst. I know some Excel.",
    "I'm a complete beginner and want to learn web development.",
    "I have 3 years of Python experience and want to get into ML.",
    "I want to switch careers into UX design.",
    "I want to get AWS certified and learn DevOps.",
]


def ensure_session() -> str:
    """
    Initialise session state and return a session ID that survives a refresh.

    The ID is kept in the URL query string. Streamlit's ``session_state`` is
    cleared on reload, so a freshly generated UUID each time meant a refresh
    orphaned everything already stored against the old ID: progress rows sat
    intact in SQLite while the dashboard reported no path at all. Round-tripping
    the ID through the URL makes a reload resume the same session.

    Anything already persisted is then restored, and the path is *rebuilt* rather
    than stored. The path is derived from the profile, progress and feedback, so
    recomputing it guarantees it agrees with them; a cached copy could not.
    """
    if "session_id" not in st.session_state:
        existing = st.query_params.get("s")
        session_id = existing or str(uuid.uuid4())
        st.session_state.session_id = session_id
        if not existing:
            st.query_params["s"] = session_id

    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("path", [])
    st.session_state.setdefault("path_result", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("restored", False)

    if st.session_state.profile is None and not st.session_state.restored:
        _restore_session(st.session_state.session_id)

    return st.session_state.session_id


def _restore_session(session_id: str) -> None:
    """Reload a persisted profile and rebuild its path, if one exists."""
    from app.profiling_engine import load_profile

    # Marked before doing the work, so a failure to restore is not retried on
    # every rerun.
    st.session_state.restored = True

    stored = load_profile(session_id)
    if not stored:
        return

    from app.pipeline import build_learning_path

    with st.spinner("Picking up where you left off..."):
        result = build_learning_path(session_id, stored)

    st.session_state.profile = stored
    st.session_state.path_result = result
    st.session_state.path = result.path if result.servable else []

    completed = len(
        [
            s for s in result.course_steps
            if s["course"]["course_id"] in _completed_ids(session_id)
        ]
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": (
                f"Welcome back. I've restored your goal — _{stored['goal']}_ — and "
                f"rebuilt your path around what you've completed so far "
                f"({completed} of {len(result.course_steps)} done)."
            ),
        }
    )


def _completed_ids(session_id: str) -> set[str]:
    from app.db import get_completed_course_ids

    return get_completed_course_ids(session_id)


def _describe_unservable(profile: dict, result) -> str:
    """
    Compose an honest reply when the catalog cannot address the goal.

    Saying so is a deliberate feature. Ranking always produces *some* top result,
    so without this the app would answer a request about medieval manuscripts with
    five cloud-computing courses and a confident match percentage.
    """
    from app.profiling_engine import CATEGORY_LABELS, LABEL_TO_CATEGORY

    domains = ", ".join(f"**{LABEL_TO_CATEGORY[c]}**" for c in CATEGORY_LABELS)
    return (
        f"I couldn't find a credible path for that goal.\n\n"
        f"My catalog covers {domains}, and nothing in it matches **{profile['goal']}** "
        f"closely enough for me to recommend honestly (best match scored "
        f"{result.gap.confidence:.0%}).\n\n"
        f"Rather than show you loosely related courses, I'd rather tell you. Try "
        f"rephrasing toward one of those domains, or describe the underlying skill "
        f"you're after."
    )


def _describe_profile(profile: dict, result) -> str:
    """Compose the assistant's confirmation message from the profile and gap."""
    have = ", ".join(profile["current_skills"]) if profile["current_skills"] else "none mentioned"
    want = ", ".join(profile.get("target_skills") or []) or "open to suggestions"
    interests = ", ".join(profile["interests"]) if profile["interests"] else "general"

    gap = result.gap
    steps = result.course_steps
    counts: dict[str, int] = {}
    for step in steps:
        kind = step["course"].get("resource_type", "course")
        counts[kind] = counts.get(kind, 0) + 1
    mix = ", ".join(f"{n} {kind}{'s' if n != 1 else ''}" for kind, n in sorted(counts.items()))

    credited = ""
    if gap.satisfied_skills:
        credited = (
            f"\nYou already have {len(gap.satisfied_skills)} of the "
            f"{gap.target_count} skills this goal needs "
            f"({', '.join(gap.satisfied_skills[:4])}"
            f"{'...' if len(gap.satisfied_skills) > 4 else ''}), so I've skipped those.\n"
        )

    prior = profile.get("completed_resources") or []
    if prior:
        credited += (
            f"\nI've also routed around the {len(prior)} resource"
            f"{'s' if len(prior) != 1 else ''} you've already completed "
            f"({', '.join(prior[:5])}{'...' if len(prior) > 5 else ''}).\n"
        )

    return (
        f"✅ Here's what I understood:\n\n"
        f"- **Goal:** {profile['goal']}\n"
        f"- **Skills you already have:** {have}\n"
        f"- **Skills you want:** {want}\n"
        f"- **Experience level:** {profile['experience_level']}"
        f"{' (raised by your feedback)' if result.level_escalated else ''}\n"
        f"- **Domain focus:** {interests}\n\n"
        f"**Skill gap:** this goal needs about **{gap.target_count} skills** and you're "
        f"missing **{gap.gap_count}** of them.\n"
        f"{credited}\n"
        f"I've built a path of **{mix}** totalling **{result.total_hours} hours**, which "
        f"closes **{result.coverage.get('covered_ratio', 0):.0%}** of that gap. Open "
        f"**My Learning Path** to see it, or **Dashboard** to track progress."
    )


_TYPE_LABEL = {"course": "course", "project": "project", "assessment": "assessment"}

_STYLE_LABELS = {
    "hands_on": "Hands-on — put me on projects early",
    "balanced": "Balanced — mix theory and practice",
    "theory": "Theory first — I want to understand before building",
}
_PACE_LABELS = {
    "intensive": "Intensive — plan a lot, I'll move fast",
    "steady": "Steady — a normal amount of work",
    "relaxed": "Relaxed — keep the plan short and manageable",
}


def _rebuild_path(session_id: str, spinner: str) -> None:
    """Rebuild an on-screen path after something that changes the plan."""
    profile = st.session_state.get("profile")
    if not profile:
        return
    from app.pipeline import build_learning_path

    with st.spinner(spinner):
        result = build_learning_path(session_id, profile)
    st.session_state.path_result = result
    st.session_state.path = result.path if result.servable else []


def _render_preferences(session_id: str) -> None:
    """
    Capture how the learner wants to learn, not just what.

    The brief personalises on "learning patterns" and notes that learners have
    different "learning preferences". These controls are the stated half;
    ``learner_model.derive_patterns`` handles the observed half.

    Each control has a concrete effect rather than being recorded and ignored:
    style decides whether a checkpoint opens with a project or an assessment,
    weekly hours drive the schedule, and pace bounds how much path is planned.
    """
    from app.db import get_preferences, init_db, set_preferences
    from app.learner_model import (
        MAX_HOURS_PER_WEEK,
        MIN_HOURS_PER_WEEK,
        LearnerPreferences,
    )

    init_db()
    current = LearnerPreferences.from_dict(get_preferences(session_id))

    with st.expander(
        f"⚙️ How you like to learn ({current.hours_per_week}h/week, "
        f"{current.style.replace('_', '-')})",
        expanded=False,
    ):
        col_style, col_pace = st.columns(2)

        with col_style:
            style = st.radio(
                "Preferred format",
                options=list(_STYLE_LABELS.keys()),
                index=list(_STYLE_LABELS.keys()).index(current.style),
                format_func=lambda s: _STYLE_LABELS[s],
                key="pref_style",
            )
        with col_pace:
            pace = st.radio(
                "Pace",
                options=list(_PACE_LABELS.keys()),
                index=list(_PACE_LABELS.keys()).index(current.pace),
                format_func=lambda p: _PACE_LABELS[p],
                key="pref_pace",
            )

        hours = st.slider(
            "Hours available per week",
            min_value=MIN_HOURS_PER_WEEK,
            max_value=min(MAX_HOURS_PER_WEEK, 40),
            value=current.hours_per_week,
            key="pref_hours",
            help="Used to turn your path into a week-by-week schedule.",
        )

        st.caption(
            "You can also just say it in the chat — \"I learn best by doing, about "
            "5 hours a week\" sets all three."
        )

        updated = {"style": style, "pace": pace, "hours_per_week": hours}
        if updated != current.to_dict():
            set_preferences(session_id, updated)
            _rebuild_path(session_id, "Rebuilding your path around those preferences...")
            st.rerun()


def _render_prior_learning(session_id: str) -> None:
    """
    Let the learner declare what they have already completed.

    The brief requires the profile to capture completed courses and previous
    learning history. Prose mentions are picked up by
    ``profiling_engine.extract_completed_resources``, but that can only catch what
    someone happens to write, so this picker is the reliable route.

    Declared items feed straight into gap analysis and are excluded from the path,
    which is why changing the selection rebuilds an existing path immediately
    rather than waiting for the next message.
    """
    from app.config import load_catalog_df
    from app.db import get_prior_learning_ids, init_db, set_prior_learning

    init_db()
    df = load_catalog_df()

    labels = {
        row["course_id"]: (
            f"{row['course_id']} — {row['title']} "
            f"({_TYPE_LABEL.get(row['resource_type'], row['resource_type'])}, "
            f"{row['difficulty_level']})"
        )
        for _, row in df.iterrows()
    }
    already = get_prior_learning_ids(session_id)

    with st.expander(
        f"📚 Already studied some of this? ({len(already)} recorded)", expanded=False
    ):
        st.caption(
            "Anything you mark here is treated as done: it won't be recommended "
            "again, its skills count toward closing your skill gap, and it "
            "satisfies prerequisites for later steps."
        )
        chosen = st.multiselect(
            "Courses, projects or assessments you've already completed",
            options=list(labels.keys()),
            default=sorted(already),
            format_func=lambda cid: labels.get(cid, cid),
            key="prior_learning_select",
            placeholder="Search by name, e.g. SQL, Figma, Docker",
        )

        if set(chosen) != already:
            set_prior_learning(session_id, chosen)
            # A path already on screen is now stale, so rebuild it against the
            # newly reduced gap instead of showing a path full of finished work.
            _rebuild_path(session_id, "Updating your path around what you've already done...")
            st.rerun()


def _handle_message(text: str, session_id: str) -> None:
    """
    Route one chat message according to what it is trying to do.

    Previously every message ran the full pipeline from scratch, so "I also know
    SQL" replaced the learner's goal with the phrase "I also know SQL". Messages
    are now classified first, and only a genuinely new objective rebuilds the
    profile — everything else refines what is already there.
    """
    from app.conversation import answer_learner_question, apply_refinement, classify_message
    from app.pipeline import build_learning_path, run_full_pipeline

    st.session_state.chat_history.append({"role": "user", "content": text})
    profile = st.session_state.get("profile")
    intent, payload = classify_message(text, profile)

    # ── start the goal over, keeping facts about the learner ─────────────────
    if intent == "reset":
        st.session_state.profile = None
        st.session_state.path = []
        st.session_state.path_result = None
        _reply(
            "Cleared your goal and path. Your stated preferences and anything you've "
            "marked as already completed are kept, since those are still true. "
            "What would you like to learn?"
        )
        return

    # ── answer a question without touching the plan ──────────────────────────
    if intent == "question":
        if not profile:
            _reply(
                "I don't have a goal to reason about yet. Tell me what you'd like to "
                "learn and I'll explain every recommendation I make."
            )
            return
        _reply(answer_learner_question(text, profile, st.session_state.get("path", [])))
        return

    # ── a fresh objective rebuilds from scratch ──────────────────────────────
    if intent == "new_goal":
        with st.spinner("Analysing your goal, finding your skill gap, building your path..."):
            new_profile, result = run_full_pipeline(session_id, text)
        st.session_state.profile = new_profile
        st.session_state.path_result = result
        st.session_state.path = result.path if result.servable else []
        _reply(
            _describe_profile(new_profile, result)
            if result.servable
            else _describe_unservable(new_profile, result)
        )
        return

    # ── everything else refines the existing profile ─────────────────────────
    before = st.session_state.get("path_result")
    updated, note = apply_refinement(session_id, profile, intent, payload)
    st.session_state.profile = updated

    with st.spinner("Updating your path..."):
        result = build_learning_path(session_id, updated)
    st.session_state.path_result = result
    st.session_state.path = result.path if result.servable else []

    _reply(f"Got it — {note}.\n\n{_describe_change(before, result)}")


def _describe_change(before, after) -> str:
    """
    Report what a refinement actually did to the plan.

    Stating the delta is the point: a refinement that changed nothing should say
    so rather than implying it worked.
    """
    if after is None or not after.servable:
        return (
            "That leaves nothing I can recommend confidently. Try describing the "
            "goal differently."
        )

    lines = [
        f"Your path is now **{len(after.course_steps)} resources** "
        f"(**{after.total_hours}h**, about **{after.weeks} weeks** at "
        f"{after.preferences.hours_per_week}h/week), closing "
        f"**{after.coverage.get('covered_ratio', 0):.0%}** of your remaining skill gap."
    ]

    if before is not None and before.servable:
        d_res = len(after.course_steps) - len(before.course_steps)
        d_gap = after.gap.gap_count - before.gap.gap_count
        if d_res == 0 and d_gap == 0:
            lines.append("The plan itself didn't need to change.")
        else:
            if d_res:
                lines.append(
                    f"That's {abs(d_res)} {'more' if d_res > 0 else 'fewer'} "
                    f"resource{'s' if abs(d_res) != 1 else ''} than before."
                )
            if d_gap < 0:
                lines.append(f"{abs(d_gap)} fewer skills left to learn.")
            elif d_gap > 0:
                lines.append(f"{d_gap} more skills identified as needed.")
    return " ".join(lines)


def _reply(content: str) -> None:
    """Append an assistant turn to the transcript."""
    st.session_state.chat_history.append({"role": "assistant", "content": content})


def render_chat_interface() -> None:
    """Render the goal-entry tab."""
    session_id = ensure_session()

    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border-radius:16px;padding:24px 28px 16px;margin-bottom:24px;'>
            <h2 style='color:#e0e0ff;margin:0 0 6px;font-size:1.5rem;'>
                💬 What do you want to learn?
            </h2>
            <p style='color:#a0a0c0;margin:0;font-size:0.95rem;'>
                Describe your goal in plain English. Mention what you already know
                and roughly how much experience you have, and the path will adapt.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Try an example:**")
    for col, example in zip(st.columns(len(EXAMPLE_GOALS)), EXAMPLE_GOALS):
        with col:
            if st.button(
                example[:32] + "...",
                key=f"example_{example[:18]}",
                width="stretch",
                help=example,
            ):
                st.session_state["_pending_goal"] = example

    _render_preferences(session_id)
    _render_prior_learning(session_id)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.get("profile"):
        st.caption(
            "You can keep talking: try *\"I also know Python\"*, *\"make it shorter\"*, "
            "*\"why this course?\"*, *\"I've finished DS001\"*, or *\"start over\"*."
        )

    typed = st.chat_input(
        placeholder="e.g. I want to become a data scientist. I know basic Python."
    )
    message = st.session_state.pop("_pending_goal", None) or typed

    if message:
        _handle_message(message, session_id)
        # Rerun so the new exchange renders through the history loop above,
        # rather than being drawn twice in this pass.
        st.rerun()
