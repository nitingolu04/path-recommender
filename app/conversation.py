"""
conversation.py — Turn the chat tab into an actual conversation.

The brief's first requirement is "a conversational interface where learners
describe their goals in natural language". The interface was styled as a chat but
behaved as a single-shot form: every message called ``build_profile()``, which
rebuilt the profile from that message alone. Saying "I also know SQL" therefore
discarded the original goal and produced a path for the phrase "I also know SQL".

This module adds the missing piece: decide what a message is *doing*, then apply
it to the profile the learner already has.

Supported intents, in the order they are tested
-----------------------------------------------
    reset             "start over", "forget that"
    question          "why this course?", "how long will this take?"
    mark_completed    "I've finished DS001"
    resize            "make it shorter", "give me more"
    set_preferences   "I only have 3 hours a week", "more hands-on"
    add_skills        "I also know SQL", "I want to learn Docker too"
    new_goal          anything that reads as a fresh objective

Order matters. "I also want to learn Docker" contains "want to learn", which in
isolation reads as a new goal; the additive cue has to be tested first. Equally
a question must be recognised before its content words are mined for skills.

Why intent classification is rule-based
---------------------------------------
These are short, formulaic utterances with strong lexical markers, which is
exactly where patterns beat embeddings. Embeddings are used where meaning is
genuinely fuzzy — matching goals to courses — and rules are used where the signal
is a literal word like "also" or "shorter". Using the model here would add
latency and unpredictability for no accuracy gain.
"""

from __future__ import annotations

import re

from app.learner_model import PACES, LearnerPreferences, extract_preferences

# ── Intent patterns ───────────────────────────────────────────────────────────

_RESET_RE = re.compile(
    r"\b(?:start\s+over|start\s+again|reset|forget\s+(?:that|it|everything)|"
    r"scrap\s+that|never\s+mind|clear\s+(?:that|everything)|new\s+plan)\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"^\s*(?:why|what|how|when|which|who|can|could|should|would|do|does|did|is|are|"
    r"will|tell\s+me)\b",
    re.IGNORECASE,
)

_COMPLETION_CUE_RE = re.compile(
    r"\b(?:completed|finished|done\s+with|just\s+did|already\s+did|"
    r"passed|took|i've\s+done)\b",
    re.IGNORECASE,
)

_SHORTEN_RE = re.compile(
    r"\b(?:shorter|too\s+(?:long|much|many)|trim|cut\s+it\s+down|condense|"
    r"fewer|reduce|less\s+work|slim)\b",
    re.IGNORECASE,
)

_LENGTHEN_RE = re.compile(
    r"\b(?:longer|more\s+(?:courses|depth|detail|content|work)|too\s+(?:short|few|little)|"
    r"expand|go\s+deeper|add\s+more|comprehensive)\b",
    re.IGNORECASE,
)

# Signals that this adds to what we already know rather than replacing it.
_ADDITIVE_RE = re.compile(
    r"\b(?:also|as\s+well|too|plus|additionally|on\s+top\s+of\s+that|"
    r"and\s+i\s+(?:know|have|want)|i\s+forgot|forgot\s+to\s+mention|"
    r"one\s+more\s+thing|by\s+the\s+way)\b",
    re.IGNORECASE,
)

# Reads as a fresh objective rather than an adjustment.
_GOAL_RE = re.compile(
    r"\b(?:i\s+want\s+to\s+(?:become|be|learn|get\s+into|move\s+into|switch)|"
    r"my\s+goal|help\s+me\s+(?:become|learn)|i'?d\s+like\s+to\s+(?:become|learn)|"
    r"i\s+need\s+to\s+learn|career\s+change|retrain|"
    r"aiming\s+to\s+(?:become|be))\b",
    re.IGNORECASE,
)

INTENTS = (
    "reset",
    "question",
    "mark_completed",
    "resize",
    "set_preferences",
    "add_skills",
    "new_goal",
)

#: A message shorter than this is very unlikely to be a full goal statement, so a
#: bare "sql too" is treated as an addition rather than a replacement.
_MIN_GOAL_WORDS = 5


def classify_message(text: str, profile: dict | None) -> tuple[str, dict]:
    """
    Decide what a chat message is trying to do.

    Parameters
    ----------
    text : str
        The learner's message.
    profile : dict | None
        The profile already in play. With no profile there is nothing to refine,
        so every message is a new goal.

    Returns
    -------
    tuple[str, dict]
        The intent name and a payload of anything extracted while deciding.
    """
    text = (text or "").strip()
    if not text:
        return "new_goal", {}

    # Nothing to refine yet.
    if not profile or not profile.get("goal"):
        return "new_goal", {}

    if _RESET_RE.search(text):
        return "reset", {}

    # Questions must be caught before their content words are mined for skills,
    # or "what if I already know Python?" would register Python as a new skill.
    if _QUESTION_RE.match(text) or text.rstrip().endswith("?"):
        return "question", {"question": text}

    from app.profiling_engine import extract_completed_resources, extract_skills_by_polarity

    completed = extract_completed_resources(text)
    if completed and _COMPLETION_CUE_RE.search(text):
        return "mark_completed", {"resources": completed}

    if _SHORTEN_RE.search(text):
        return "resize", {"direction": "shorter"}
    if _LENGTHEN_RE.search(text):
        return "resize", {"direction": "longer"}

    stated_prefs = extract_preferences(text)
    have, want = extract_skills_by_polarity(text)

    # A pure preference statement carries no new skills.
    if stated_prefs and not have and not want:
        return "set_preferences", {"preferences": stated_prefs}

    additive = bool(_ADDITIVE_RE.search(text))
    if additive and (have or want):
        return "add_skills", {"current_skills": have, "target_skills": want}

    # A recognisable goal statement of reasonable length replaces the objective.
    if _GOAL_RE.search(text) and len(text.split()) >= _MIN_GOAL_WORDS and not additive:
        return "new_goal", {}

    # Falling through with skills but no additive cue: still an addition. Replacing
    # the whole goal on the strength of "I know Docker" would throw away context
    # the learner never asked to discard.
    if have or want:
        return "add_skills", {"current_skills": have, "target_skills": want}

    if stated_prefs:
        return "set_preferences", {"preferences": stated_prefs}

    return "new_goal", {}


def apply_refinement(
    session_id: str,
    profile: dict,
    intent: str,
    payload: dict,
) -> tuple[dict, str]:
    """
    Apply a refinement to an existing profile and persist the change.

    Returns
    -------
    tuple[dict, str]
        The updated profile and a short confirmation of what changed.
    """
    from app.db import (
        get_preferences,
        get_prior_learning_ids,
        set_preferences,
        set_prior_learning,
        upsert_profile,
    )

    updated = dict(profile)
    notes: list[str] = []

    if intent == "add_skills":
        added_current = _merge(updated.setdefault("current_skills", []),
                               payload.get("current_skills") or [])
        added_target = _merge(updated.setdefault("target_skills", []),
                              payload.get("target_skills") or [])
        # A skill can be held or wanted, not both; holding it is the stronger claim.
        updated["target_skills"] = [
            s for s in updated["target_skills"] if s not in updated["current_skills"]
        ]
        if added_current:
            notes.append(f"added **{', '.join(added_current)}** to what you already know")
        if added_target:
            notes.append(f"added **{', '.join(added_target)}** to what you want to learn")
        if not notes:
            notes.append("nothing new there — those were already on your profile")

    elif intent == "set_preferences":
        merged = {**get_preferences(session_id), **(payload.get("preferences") or {})}
        set_preferences(session_id, merged)
        prefs = LearnerPreferences.from_dict(merged)
        updated["preferences"] = prefs.to_dict()
        notes.append(
            f"set your pace to **{prefs.pace}** at **{prefs.hours_per_week}h/week**, "
            f"format **{prefs.style.replace('_', '-')}**"
        )

    elif intent == "resize":
        prefs = LearnerPreferences.from_dict(get_preferences(session_id))
        direction = payload.get("direction", "shorter")
        new_pace = _shift_pace(prefs.pace, direction)
        if new_pace == prefs.pace:
            notes.append(
                f"your plan is already as {'short' if direction == 'shorter' else 'long'} "
                f"as I can make it"
            )
        else:
            merged = {**prefs.to_dict(), "pace": new_pace}
            set_preferences(session_id, merged)
            updated["preferences"] = LearnerPreferences.from_dict(merged).to_dict()
            notes.append(f"switched your pace to **{new_pace}** to make the path "
                         f"{'shorter' if direction == 'shorter' else 'longer'}")

    elif intent == "mark_completed":
        resources = payload.get("resources") or []
        combined = sorted(get_prior_learning_ids(session_id) | set(resources))
        set_prior_learning(session_id, combined)
        updated["completed_resources"] = combined
        notes.append(f"marked **{', '.join(resources)}** as done and routed around them")

    if intent in ("add_skills",):
        upsert_profile(
            session_id=session_id,
            goal=updated.get("goal", ""),
            current_skills=updated.get("current_skills", []),
            experience_level=updated.get("experience_level", "beginner"),
            interests=updated.get("interests", []),
            target_skills=updated.get("target_skills", []),
        )

    return updated, "; ".join(notes) if notes else "updated your profile"


def _merge(existing: list[str], additions: list[str]) -> list[str]:
    """Append genuinely new entries to ``existing`` in place; return what was added."""
    added: list[str] = []
    for item in additions:
        if item not in existing:
            existing.append(item)
            added.append(item)
    existing.sort()
    return added


def _shift_pace(pace: str, direction: str) -> str:
    """
    Move one step along ``PACES`` toward a shorter or longer plan.

    ``PACES`` runs intensive -> steady -> relaxed, i.e. from most work planned to
    least, so "shorter" moves forward through the tuple.
    """
    order = list(PACES)
    try:
        index = order.index(pace)
    except ValueError:
        index = order.index("steady")
    index += 1 if direction == "shorter" else -1
    return order[max(0, min(index, len(order) - 1))]


def answer_learner_question(question: str, profile: dict, path: list[dict]) -> str:
    """
    Answer a question in the chat, choosing sensible context automatically.

    Answers are about a specific resource, so a bare "why this one?" needs a
    subject. A resource named in the question wins; otherwise the next incomplete
    step is used, since that is what "this" almost always refers to
    mid-conversation.

    Routed through ``ai_tutor``, so an LLM handles it when one is configured and
    the templates answer when not. The whole path is passed as context, which lets
    the model field comparative questions like "why this before step 4?".
    """
    from app.ai_tutor import answer as ask

    course_steps = [s for s in path if s.get("course")]
    if not course_steps:
        return (
            "I don't have a path built yet, so there's nothing to explain. "
            "Tell me what you'd like to learn first."
        )

    subject = None
    upper = question.upper()
    for step in course_steps:
        resource = step["course"]
        if resource["course_id"].upper() in upper:
            subject = step
            break
    if subject is None:
        subject = course_steps[0]

    resource = subject["course"]
    result = ask(
        question,
        resource,
        profile,
        path=path,
        similarity_score=subject.get("similarity_score", 0.0),
    )
    attribution = f"\n\n_{result.provider}_" if result.used_llm else ""
    return (
        f"About **{resource['title']}** (step {subject['step_number']}):\n\n"
        f"{result.text}{attribution}"
    )


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = {
        "goal": "I want to become a data analyst",
        "current_skills": ["excel"],
        "target_skills": ["sql"],
        "experience_level": "intermediate",
        "interests": ["Data Science"],
        "completed_resources": [],
        "preferences": LearnerPreferences().to_dict(),
    }

    # ── with no profile, everything starts a goal ─────────────────────────────
    assert classify_message("I also know SQL", None)[0] == "new_goal"
    assert classify_message("anything", {})[0] == "new_goal"

    # ── intent routing ───────────────────────────────────────────────────────
    cases = [
        # (message, expected intent)
        ("start over", "reset"),
        ("forget that, new plan", "reset"),
        ("Why was this course recommended?", "question"),
        ("How long will this take?", "question"),
        ("What if I already know Python?", "question"),
        ("is this too hard?", "question"),
        ("I've finished DS001", "mark_completed"),
        ("I already completed DS005 and DS006", "mark_completed"),
        ("make it shorter", "resize"),
        ("this is too many courses", "resize"),
        ("can you add more depth", "question"),      # leading "can" reads as a question
        ("I only have 3 hours a week", "set_preferences"),
        ("I learn best by doing", "set_preferences"),
        ("I also know SQL", "add_skills"),
        ("I forgot to mention I know Docker", "add_skills"),
        ("and I want to learn Airflow too", "add_skills"),
        ("I want to become a UX designer instead", "new_goal"),
        ("my goal is to move into cloud engineering", "new_goal"),
    ]
    for message, expected in cases:
        got, _ = classify_message(message, base)
        assert got == expected, f"{message!r} -> {got!r}, expected {expected!r}"
    print(f"intent routing: {len(cases)} cases OK")

    # REGRESSION: the reason this module exists. "I also want to learn Docker"
    # contains "want to learn", which alone reads as a brand-new goal and would
    # have discarded the learner's actual objective.
    intent, payload = classify_message("I also want to learn Docker", base)
    assert intent == "add_skills", intent
    assert any("docker" in s for s in payload["target_skills"]), payload

    # A question must not be mined for skills.
    intent, payload = classify_message("What if I already know Python?", base)
    assert intent == "question"
    assert "question" in payload

    # ── refinement application ───────────────────────────────────────────────
    from app.db import _get_connection, init_db

    SID = "__selftest_conversation__"

    def _cleanup() -> None:
        conn = _get_connection()
        with conn:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (SID,))
        conn.close()

    init_db()
    _cleanup()

    # Adding a skill must preserve the goal — the whole point.
    intent, payload = classify_message("I also know Python and Tableau", base)
    updated, note = apply_refinement(SID, base, intent, payload)
    assert updated["goal"] == base["goal"], "refinement must not discard the goal"
    assert "excel" in updated["current_skills"], updated["current_skills"]
    assert "python" in updated["current_skills"], updated["current_skills"]
    print(f"add_skills -> {note}")

    # A skill claimed as held must leave the wanted list.
    held_now = dict(base, target_skills=["docker"])
    intent, payload = classify_message("I also know Docker", held_now)
    updated, _ = apply_refinement(SID, held_now, intent, payload)
    assert "docker" in updated["current_skills"]
    assert "docker" not in updated["target_skills"], (
        "a skill cannot be both held and wanted"
    )

    # Re-adding something known is reported, not silently duplicated.
    intent, payload = classify_message("I also know Excel", base)
    updated, note = apply_refinement(SID, base, intent, payload)
    assert updated["current_skills"].count("excel") == 1, updated["current_skills"]
    assert "already" in note.lower(), note

    # ── resize walks the pace ladder and stops at the ends ───────────────────
    assert _shift_pace("intensive", "shorter") == "steady"
    assert _shift_pace("steady", "shorter") == "relaxed"
    assert _shift_pace("relaxed", "shorter") == "relaxed", "must clamp at the end"
    assert _shift_pace("relaxed", "longer") == "steady"
    assert _shift_pace("intensive", "longer") == "intensive", "must clamp at the end"
    assert _shift_pace("nonsense", "shorter") == "relaxed"

    intent, payload = classify_message("make it shorter", base)
    updated, note = apply_refinement(SID, base, intent, payload)
    assert updated["preferences"]["pace"] == "relaxed", updated["preferences"]
    print(f"resize -> {note}")

    # Asking again at the end of the ladder says so rather than pretending.
    updated2, note2 = apply_refinement(SID, updated, "resize", {"direction": "shorter"})
    assert "already" in note2.lower(), note2

    # ── preferences from prose ───────────────────────────────────────────────
    intent, payload = classify_message("I only have 2 hours a week", base)
    updated, note = apply_refinement(SID, base, intent, payload)
    assert updated["preferences"]["hours_per_week"] == 2, updated["preferences"]
    print(f"set_preferences -> {note}")

    # ── completion from prose ────────────────────────────────────────────────
    intent, payload = classify_message("I've finished DS001 and DS005", base)
    assert intent == "mark_completed"
    updated, note = apply_refinement(SID, base, intent, payload)
    assert set(updated["completed_resources"]) >= {"DS001", "DS005"}, updated
    print(f"mark_completed -> {note}")

    # ── question answering picks sensible context ────────────────────────────
    fake_path = [
        {"step_number": 1, "similarity_score": 0.8,
         "course": {"course_id": "DS005", "title": "SQL for Data Analysts",
                    "skills": "sql,joins", "prerequisites": "",
                    "difficulty_level": "beginner", "category": "Data Science",
                    "resource_type": "course", "duration_hours": 6}},
        {"step_number": 2, "similarity_score": 0.7,
         "course": {"course_id": "DS007", "title": "Machine Learning with Scikit-Learn",
                    "skills": "machine learning", "prerequisites": "DS001",
                    "difficulty_level": "intermediate", "category": "Data Science",
                    "resource_type": "course", "duration_hours": 15}},
    ]
    answer = answer_learner_question("Why this course?", base, fake_path)
    assert "SQL for Data Analysts" in answer, answer

    # Naming a resource overrides the default subject.
    answer = answer_learner_question("what prerequisites does DS007 need?", base, fake_path)
    assert "Machine Learning" in answer, answer
    # The explainer renders prerequisites as titles rather than bare IDs, so the
    # answer names the course instead of echoing "DS001" back at the learner.
    assert "Introduction to Python for Data Science" in answer, answer
    assert "step 2" in answer, "the answer should say which step it is about"

    # No path yet is handled without crashing.
    assert "don't have a path" in answer_learner_question("why?", base, [])

    _cleanup()
    print("\nconversation.py self-test passed: all assertions OK")
