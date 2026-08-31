"""
main.py — Streamlit entrypoint for the AI-Powered Learning Path Recommender.

Tabs:
    💬 Chat           → goal input, profile building, pipeline trigger
    📚 My Learning Path → ordered course list with explanations
    📊 Dashboard       → progress tracking, skill view, feedback buttons

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on the path when Streamlit runs this file directly
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Learning Path Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark background */
    .stApp { background: #0d0d1a; color: #e0e0f0; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #12122a;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #9090b0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #5e35b1, #3949ab) !important;
        color: white !important;
    }

    /* Cards */
    .course-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 20px;
        margin: 12px 0;
        transition: border-color 0.2s ease;
    }
    .course-card:hover { border-color: #5e35b1; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #12122a;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #12122a;
        border-radius: 12px;
        border: 1px solid #1e1e3a;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #5e35b1, #3949ab);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #12122a;
        border: 1px solid #2a2a4a;
        color: #e0e0f0;
        border-radius: 10px;
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #5e35b1, #3949ab); }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #12122a;
        border-radius: 10px;
        color: #c0c0e0 !important;
        font-weight: 500;
    }

    /* Divider */
    hr { border-color: #2a2a4a; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0d1a; }
    ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center;padding:32px 0 16px;'>
        <div style='font-size:3rem;margin-bottom:8px;'>🎓</div>
        <h1 style='background:linear-gradient(135deg,#a78bfa,#60a5fa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-size:2.4rem;font-weight:700;margin:0;'>
            AI Learning Path Recommender
        </h1>
        <p style='color:#7070a0;margin:8px 0 0;font-size:1.05rem;'>
            Describe your learning goal → get a personalised, sequenced course path
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session ───────────────────────────────────────────────────────────────────
# Initialised above the tabs rather than inside the first one. Streamlit runs every
# tab body on each rerun, so putting this in the chat tab happened to work — but
# only because that tab is declared first. Tabs 2 and 3 read session state, so
# their correctness should not depend on declaration order.
#
# ensure_session() also restores a persisted session, so this is where a page
# refresh picks up an existing goal and path.
from app.chat_interface import ensure_session, render_chat_interface  # noqa: E402

ensure_session()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_path, tab_dash, tab_eval = st.tabs(
    ["💬 Chat", "📚 My Learning Path", "📊 Dashboard", "🔬 How It Performs"]
)

# ── Tab 1: Chat ────────────────────────────────────────────────────────────────
with tab_chat:
    render_chat_interface()

# ── Tab 2: Learning Path ───────────────────────────────────────────────────────
with tab_path:
    path = st.session_state.get("path", [])
    profile = st.session_state.get("profile")

    if not path:
        st.info("💡 Enter your learning goal in the **💬 Chat** tab to generate your personalised path.")
    else:
        from app.ai_tutor import answer as ask_about_step
        from app.db import get_completed_course_ids
        from app.explainer import explain_course

        session_id = st.session_state.get("session_id", "")
        completed_ids = get_completed_course_ids(session_id)

        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,#0f3460,#533483);
                        border-radius:16px;padding:20px 28px;margin-bottom:20px;'>
                <h2 style='color:#fff;margin:0 0 4px;'>📚 Your Personalised Learning Path</h2>
                <p style='color:#c0c0e0;margin:0;'>
                    Courses are ordered by prerequisites and difficulty. Milestones mark key checkpoints.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Profile summary chip row
        if profile:
            summary = (
                f"**Goal:** _{profile.get('goal', '')}_ &nbsp;|&nbsp; "
                f"**Level:** `{profile.get('experience_level', '')}` &nbsp;|&nbsp; "
                f"**Focus:** {', '.join(profile.get('interests', []))}"
            )
            if profile.get("current_skills"):
                summary += f" &nbsp;|&nbsp; **You already know:** {', '.join(profile['current_skills'])}"
            if profile.get("target_skills"):
                summary += f" &nbsp;|&nbsp; **You want to learn:** {', '.join(profile['target_skills'])}"
            st.markdown(summary)
            st.markdown("---")

        for step in path:
            # Milestone
            if step["milestone"]:
                st.markdown(
                    f"<div style='background:#1a2e1a;border-left:4px solid #4caf50;"
                    f"padding:12px 18px;border-radius:10px;margin:16px 0;color:#a5d6a7;"
                    f"font-weight:500;'>{step['milestone']}</div>",
                    unsafe_allow_html=True,
                )
                continue

            c = step["course"]
            cid = c["course_id"]
            is_done = cid in completed_ids
            score = step.get("similarity_score", 0.0)
            is_filler = step.get("is_prerequisite_filler", False)

            diff_level = c.get("difficulty_level", "beginner").lower()
            diff_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(diff_level, "⚪")
            done_text = " ✅" if is_done else ""
            filler_text = " &nbsp;·&nbsp; 🧱 PREREQUISITE" if is_filler else ""

            # The brief asks for courses, projects and assessments, so the type is
            # labelled rather than left for the user to infer from the title.
            kind = c.get("resource_type", "course")
            kind_badge = {
                "course": "📘 COURSE",
                "project": "🛠️ PROJECT",
                "assessment": "📝 ASSESSMENT",
            }.get(kind, "📘 COURSE")

            with st.container(border=True):
                col_info, col_score = st.columns([5, 1])

                with col_info:
                    st.markdown(
                        f"<span style='color:#7070a0;font-size:0.78rem;'>"
                        f"STEP {step['step_number']} &nbsp;·&nbsp; {kind_badge} &nbsp;·&nbsp; "
                        f"{c.get('category','')} &nbsp;·&nbsp; "
                        f"{diff_emoji} {diff_level.upper()} &nbsp;·&nbsp; "
                        f"⏱️ {c.get('duration_hours','?')}h{done_text}{filler_text}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### {c['title']}")
                    st.markdown(f"{c.get('description','')[:220]}…")
                    st.caption(f"🔧 **Skills:** {c.get('skills','—')}")

                    # What this step closes in the learner's gap. This is the
                    # visible payoff of gap analysis: each step states which
                    # missing skills it accounts for.
                    covers = c.get("covers") or []
                    if covers:
                        st.caption(
                            f"🎯 **Closes {len(covers)} gap skill"
                            f"{'s' if len(covers) != 1 else ''}:** {', '.join(covers)}"
                        )

                    prereqs = str(c.get("prerequisites", "")).strip()
                    if prereqs and prereqs != "nan":
                        st.caption(f"📋 **Prerequisites:** {prereqs}")

                with col_score:
                    # Prerequisite fillers were never scored by the recommender,
                    # so showing them a "0% match" would misrepresent why they
                    # are in the path.
                    if is_filler:
                        st.metric(label="Role", value="Prereq")
                    else:
                        share = c.get("coverage_share")
                        if share:
                            st.metric(label="Gap closed", value=f"{share:.0%}")
                            st.caption(f"match {score:.0%}")
                        else:
                            st.metric(label="Match", value=f"{score:.0%}")
                            semantic = c.get("semantic_score")
                            if semantic is not None:
                                st.caption(f"semantic {semantic:.0%}")

            # Explanation + inline Q&A
            explanation = explain_course(c, profile, score) if profile else ""
            with st.expander("💡 Why this course? / Ask a question", expanded=False):
                st.info(explanation)

                st.markdown("**Ask something about this step:**")
                user_q = st.text_input(
                    label="question",
                    label_visibility="collapsed",
                    placeholder=(
                        "Anything — e.g. Should I learn JavaScript first? "
                        "Will this help me get a job? How does this relate to step 3?"
                    ),
                    key=f"qa_{cid}",
                )
                if user_q:
                    # Answered by an LLM when one is configured, grounded in this
                    # resource's real catalog data and the learner's profile, with
                    # the templates as a guaranteed fallback.
                    with st.spinner("Thinking..."):
                        result = ask_about_step(
                            user_q,
                            c,
                            profile,
                            path=path,
                            similarity_score=score,
                        )
                    st.success(result.text)
                    if result.used_llm:
                        st.caption(f"🤖 answered by {result.provider}, using only this "
                                   f"resource's catalog data and your profile")
                    elif result.error == "no LLM configured":
                        st.caption(
                            "📋 answered from built-in templates. Add an LLM API key "
                            "to handle open-ended questions — see the README."
                        )
                    else:
                        # A key exists but the call failed. Telling the user to add a
                        # key they already added wastes their time, so show the real
                        # reason — an expired key or a wrong model name is otherwise
                        # invisible from the UI.
                        st.caption("📋 answered from built-in templates.")
                        st.warning(
                            f"An LLM is configured but the call failed, so this fell "
                            f"back to templates: {result.error}",
                            icon="⚠️",
                        )

# ── Tab 3: Dashboard ───────────────────────────────────────────────────────────
with tab_dash:
    from app.dashboard import render_dashboard
    render_dashboard()

# ── Tab 4: Evaluation ──────────────────────────────────────────────────────────
# The brief requires the solution to *visibly* demonstrate its AI/ML work. A claim
# in a README is not visible; a measured comparison against a baseline is.
with tab_eval:
    from app.eval_view import render_evaluation
    render_evaluation()
