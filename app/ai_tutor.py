"""
ai_tutor.py — Answer arbitrary learner questions, grounded in real catalog facts.

The problem this solves
-----------------------
``explainer`` answers seven question shapes from templates: why, prerequisites,
too-hard, too-easy, already-know, duration, skills. Anything outside that set —
"will this get me a job?", "should I learn TypeScript before this?", "how does
this compare to the next step?" — hits the generic "I don't have a specific
answer" reply. That is the gap.

How it stays honest
-------------------
This is retrieval-augmented, not free generation. The model is handed a block of
**facts drawn from the catalog and the learner's own profile**, and instructed to
answer only from those facts. It therefore cannot invent a course, because it is
never asked to choose one — selection already happened in ``skill_gap`` and
``path_generator``, from real data with real prerequisites.

That distinction matters. An LLM asked to *build a learning path* will happily
produce course names that do not exist. An LLM asked to *explain a course it has
been given* has nothing to fabricate.

Guarantees
----------
- **Never breaks.** Any failure — no key, dead network, rate limit, bad model
  name — falls back to the templates. ``answer()`` cannot raise.
- **Always attributable.** The result records whether an LLM or a template
  produced it, so the UI can say so rather than blurring the two.
- **Cached.** Streamlit re-runs the whole script on every interaction, so an
  uncached call would re-bill and re-wait every time the user clicks anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import load_catalog_map, parse_skills

#: Bound on the answer cache. Small because a session asks few questions, and
#: entries are keyed on question plus learner state.
_CACHE_LIMIT = 128
_cache: dict[tuple, "Answer"] = {}

#: What the model may and may not do. Written as hard constraints because the
#: failure mode being guarded against — confidently inventing a course, a price
#: or a duration — is exactly what would discredit the whole app.
SYSTEM_PROMPT = """\
You are a learning advisor embedded in a course-recommendation app. You answer a \
learner's question about one specific resource that the app has already selected \
for them.

Rules, in order of importance:

1. Answer ONLY using the FACTS block provided. It is the complete truth available \
to you about this resource and this learner.
2. If the FACTS do not contain the answer, say so in one sentence and state what \
you can tell them instead. Never guess.
3. Never invent courses, resources, prices, ratings, instructors, platforms, URLs \
or durations. Never recommend anything not named in the FACTS.
4. Do not restate the FACTS block. Answer the question that was asked.
5. Be concrete and address the learner as "you". 2-4 sentences. Use Markdown bold \
sparingly for key terms.
6. If the learner asks something off-topic, briefly redirect to what this resource \
covers.

You are explaining a decision the app already made. Do not second-guess the \
selection or suggest they study something else instead."""


@dataclass
class Answer:
    """An answer plus where it came from."""

    text: str
    source: str          # "llm" or "templates"
    provider: str = ""   # e.g. "gemini:gemini-2.0-flash"; empty for templates
    error: str = ""      # why the LLM was not used, when it was not

    @property
    def used_llm(self) -> bool:
        return self.source == "llm"


def build_facts(
    course: dict,
    profile: dict | None = None,
    path: list[dict] | None = None,
    similarity_score: float | None = None,
) -> str:
    """
    Assemble the factual context the model is allowed to reason from.

    Everything here comes from the catalog or the learner's stored profile. The
    breadth is deliberate: the more genuine context is supplied, the less reason
    the model has to reach for anything it wasn't given.
    """
    profile = profile or {}
    catalog = load_catalog_map()
    lines: list[str] = []

    # ── the resource itself ──────────────────────────────────────────────────
    kind = str(course.get("resource_type", "course"))
    lines.append("THIS RESOURCE")
    lines.append(f"- Title: {course.get('title', 'unknown')}")
    lines.append(f"- Type: {kind} (a course teaches skills, a project applies "
                 f"them, an assessment checks them)")
    lines.append(f"- Category: {course.get('category', 'unknown')}")
    lines.append(f"- Difficulty: {course.get('difficulty_level', 'unknown')}")
    lines.append(f"- Estimated effort: {course.get('duration_hours', '?')} hours")
    lines.append(f"- Description: {course.get('description', 'none')}")
    skills = parse_skills(course.get("skills", ""))
    if skills:
        verb = {"course": "Teaches", "project": "Exercises",
                "assessment": "Validates"}.get(kind, "Covers")
        lines.append(f"- {verb}: {', '.join(skills)}")

    prereq_ids = [
        p.strip() for p in str(course.get("prerequisites", "")).split(",")
        if p.strip() and p.strip().lower() != "nan"
    ]
    if prereq_ids:
        named = [
            f"{catalog[pid]['title']}" if pid in catalog else pid for pid in prereq_ids
        ]
        lines.append(f"- Prerequisites: {', '.join(named)}")
    else:
        lines.append("- Prerequisites: none, it can be started immediately")

    # ── why the app chose it ─────────────────────────────────────────────────
    covers = course.get("covers") or []
    if covers:
        lines.append(f"- Skills it closes in this learner's gap: {', '.join(covers)}")
    share = course.get("coverage_share")
    if isinstance(share, (int, float)) and share > 0:
        lines.append(f"- Share of the learner's remaining gap it covers: {share:.0%}")
    if similarity_score:
        lines.append(f"- Relevance to the learner's stated goal: {similarity_score:.0%}")
    if course.get("is_reinforcement"):
        lines.append("- It was inserted as a checkpoint after earlier steps, not "
                     "ranked by relevance")
    if course.get("is_prerequisite_filler"):
        lines.append("- It was added because a later step depends on it, not as a "
                     "direct match")

    # ── the learner ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("THIS LEARNER")
    lines.append(f"- Their stated goal: {profile.get('goal', 'not stated')}")
    lines.append(f"- Experience level: {profile.get('experience_level', 'unknown')}")
    held = profile.get("current_skills") or []
    lines.append(f"- Skills they already have: {', '.join(held) if held else 'none stated'}")
    wanted = profile.get("target_skills") or []
    if wanted:
        lines.append(f"- Skills they said they want: {', '.join(wanted)}")
    interests = profile.get("interests") or []
    if interests:
        lines.append(f"- Domain focus: {', '.join(interests)}")
    prefs = profile.get("preferences") or {}
    if prefs:
        lines.append(
            f"- Preferences: {prefs.get('style', 'balanced')} style, "
            f"{prefs.get('hours_per_week', '?')} hours per week, "
            f"{prefs.get('pace', 'steady')} pace"
        )
    done = profile.get("completed_resources") or []
    if done:
        named_done = [catalog[d]["title"] if d in catalog else d for d in done[:6]]
        lines.append(f"- Already completed: {', '.join(named_done)}")

    # ── where it sits in the plan ────────────────────────────────────────────
    steps = [s for s in (path or []) if s.get("course")]
    if steps:
        lines.append("")
        lines.append("THEIR FULL PATH, IN ORDER")
        this_id = course.get("course_id")
        for step in steps:
            resource = step["course"]
            marker = "  <-- the resource being asked about" \
                if resource.get("course_id") == this_id else ""
            lines.append(
                f"- Step {step.get('step_number')}: {resource.get('title')} "
                f"({resource.get('resource_type')}, "
                f"{resource.get('duration_hours', '?')}h){marker}"
            )

    return "\n".join(lines)


def answer(
    question: str,
    course: dict,
    profile: dict | None = None,
    path: list[dict] | None = None,
    similarity_score: float | None = None,
    use_cache: bool = True,
) -> Answer:
    """
    Answer ``question`` about ``course``, preferring an LLM and falling back to
    templates.

    Cannot raise. When no LLM is configured this is exactly the previous
    behaviour, so the app is unchanged for anyone who has not opted in.
    """
    from app.explainer import answer_question
    from app import llm

    question = (question or "").strip()
    profile = profile or {}

    if not question:
        return Answer(
            text="Ask me anything about this step and I'll explain it.",
            source="templates",
        )

    def _template() -> Answer:
        return Answer(
            text=answer_question(question, course, profile, similarity_score),
            source="templates",
        )

    if not llm.is_configured():
        result = _template()
        result.error = "no LLM configured"
        return result

    key = (
        question.lower(),
        str(course.get("course_id")),
        str(profile.get("goal", ""))[:120],
        str(profile.get("experience_level", "")),
        tuple(sorted(profile.get("current_skills") or [])),
    )
    if use_cache and key in _cache:
        return _cache[key]

    facts = build_facts(course, profile, path, similarity_score)
    prompt = f"FACTS\n{facts}\n\nLEARNER'S QUESTION\n{question}"

    text, error = llm.complete(SYSTEM_PROMPT, prompt)

    if text:
        result = Answer(text=text, source="llm", provider=llm.provider_label())
    else:
        # The templates still answer the seven common shapes well, so a failed
        # call costs the learner very little.
        result = _template()
        result.error = error or "unknown LLM error"

    if use_cache:
        if len(_cache) >= _CACHE_LIMIT:
            _cache.clear()
        _cache[key] = result
    return result


def clear_cache() -> None:
    """Drop cached answers. Used by the self-test and after a config change."""
    _cache.clear()


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    from app.config import load_catalog_map as _catalog

    catalog = _catalog()
    course = dict(
        catalog["WD001"],
        covers=["css grid", "flexbox", "responsive design"],
        coverage_share=0.19,
        similarity_score=0.72,
    )
    profile = {
        "goal": "I want to become a front-end web developer",
        "current_skills": ["html"],
        "target_skills": ["react"],
        "experience_level": "beginner",
        "interests": ["Web Development"],
        "completed_resources": [],
        "preferences": {"style": "hands_on", "hours_per_week": 8, "pace": "steady"},
    }
    fake_path = [
        {"step_number": 1, "course": catalog["WD001"]},
        {"step_number": 2, "course": catalog["WD002"]},
        {"step_number": 3, "course": catalog["PR007"]},
    ]

    # ── the facts block must contain real data and no placeholders ────────────
    facts = build_facts(course, profile, fake_path, 0.72)
    print("-- FACTS block handed to the model " + "-" * 30)
    print(facts)
    print()

    assert "HTML and CSS Fundamentals" in facts
    assert "Web Development" in facts
    assert "beginner" in facts
    assert "css grid" in facts, "gap coverage must be in the facts"
    assert "19%" in facts, "coverage share must be in the facts"
    assert "front-end web developer" in facts, "the learner's goal must be in the facts"
    assert "hands_on" in facts, "preferences must be in the facts"
    assert "THEIR FULL PATH" in facts and "Step 2" in facts, "path context missing"
    assert "the resource being asked about" in facts, "the subject must be marked"

    # A resource with no prerequisites must say so rather than leaving it blank.
    assert "Prerequisites: none" in facts, facts

    # Prerequisites must appear as titles, never as raw IDs.
    ml = dict(catalog["DS007"])
    ml_facts = build_facts(ml, profile)
    assert "Introduction to Python for Data Science" in ml_facts, ml_facts
    assert "Prerequisites: DS001" not in ml_facts, "prereqs should be titles, not IDs"

    # Type-specific verbs, so the model is not told a project "teaches".
    assert "Exercises:" in build_facts(catalog["PR007"], profile)
    assert "Validates:" in build_facts(catalog["AS006"], profile)
    assert "Teaches:" in build_facts(catalog["WD001"], profile)
    print("facts block: 13 assertions OK")

    # ── with no LLM configured it must fall back silently and correctly ───────
    #
    # The config source is stubbed rather than the environment cleared. Clearing
    # env vars stopped simulating "no key" once a real secrets.toml existed, since
    # Streamlit secrets are read first — so this asserted against whatever the
    # machine was configured with instead of the case it meant to test.
    from app import llm as _llm_mod

    _real_secret = _llm_mod._secret
    _fake: dict[str, str] = {}
    _llm_mod._secret = lambda name: _fake.get(name) or None
    try:
        clear_cache()
        result = answer("What prerequisites do I need?", course, profile, fake_path, 0.72)
        assert result.source == "templates", result.source
        assert not result.used_llm
        assert result.error == "no LLM configured"
        assert "no prerequisites" in result.text, result.text
        print(f"\nno key -> template answer: {result.text[:80]}")

        # A question outside the seven templates still returns something usable
        # rather than an error.
        odd = answer("Will this help me get a job at a startup?", course, profile)
        assert odd.source == "templates"
        assert odd.text, "must always return some text"
        print(f"unmatched question -> {odd.text[:90]}")

        # Empty input is handled without a call or a crash.
        blank = answer("   ", course, profile)
        assert blank.source == "templates" and blank.text

        # ── a broken LLM config must still yield a usable answer ─────────────
        _fake.update({
            "LPR_LLM_API_KEY": "definitely-not-valid",
            "LPR_LLM_PROVIDER": "openai",
            "LPR_LLM_BASE_URL": "http://127.0.0.1:9/v1",
            "LPR_LLM_TIMEOUT": "2",
        })
        clear_cache()
        broken = answer("Why this course?", course, profile, fake_path, 0.72)
        assert broken.source == "templates", (
            "a dead endpoint must fall back, not surface an error to the learner"
        )
        assert broken.text and "Recommended because" in broken.text, broken.text
        assert broken.error and "definitely-not-valid" not in broken.error, broken.error
        print(f"dead endpoint -> fell back to templates, error recorded: "
              f"{broken.error[:60]}")

        # ── caching must prevent repeat calls ────────────────────────────────
        clear_cache()
        first = answer("Why this course?", course, profile, fake_path, 0.72)
        assert len(_cache) == 1
        second = answer("Why this course?", course, profile, fake_path, 0.72)
        assert second is first, "an identical question must be served from cache"
        # A different question is a different entry.
        answer("How long will this take?", course, profile, fake_path, 0.72)
        assert len(_cache) == 2, len(_cache)
        clear_cache()
        assert len(_cache) == 0
        print("cache: identical questions served once, distinct questions kept apart")
    finally:
        _llm_mod._secret = _real_secret

    # ── if a key IS configured, do one live call so it can be verified ───────
    from app import llm as _llm

    if _llm.is_configured():
        clear_cache()
        live = answer(
            "Should I learn JavaScript before this, and will it help me get a job?",
            course, profile, fake_path, 0.72,
        )
        print(f"\n-- LIVE call via {_llm.provider_label()} " + "-" * 24)
        print(f"source={live.source} error={live.error or 'none'}")
        print(live.text)
        assert live.text
    else:
        print("\nno LLM key configured, so the live call was skipped "
              "(set LPR_LLM_API_KEY to exercise it)")

    print("\nai_tutor.py self-test passed: all assertions OK")
