"""
explainer.py — Natural-language explanations and a self-contained Q&A system.

Two responsibilities:

1.  ``explain_course(course, profile, similarity_score)``
    One sentence on why a course was recommended, assembled from a template
    chosen by which profile signals are actually available.

2.  ``answer_question(question, course, profile, similarity_score)``
    Answers questions like "why this course?", "what if I already know Python?"
    or "what prerequisites do I need?" by matching the question against ordered
    intent patterns.  Entirely template-driven, so the app needs no LLM API.

Two things worth knowing about the implementation
-------------------------------------------------
Intent patterns are ordered **most specific first** and the first match wins.
Ordering is load-bearing, not cosmetic: "What prerequisites do I need?" contains
the phrase "do I need", which belongs to the already-know intent.  With the
already-know pattern tested first, that question was answered with advice about
skipping the course instead of a list of its prerequisites.

Explanations distinguish ``current_skills`` from ``target_skills``.  Only skills
the user said they *have* justify "builds on your X experience"; skills they said
they *want* are phrased as what the course will teach them.
"""

from __future__ import annotations

import re

from app.config import parse_skills

# ── Reason templates ───────────────────────────────────────────────────────────

_T_INTEREST_AND_SKILL = (
    "Recommended because it matches your interest in **{interest}** and builds on "
    "your existing **{skill}** experience. (Match: {score:.0%})"
)
_T_INTEREST_AND_TARGET = (
    "Recommended because it matches your interest in **{interest}** and teaches "
    "**{target}**, which you said you want to learn. (Match: {score:.0%})"
)
_T_INTEREST_ONLY = (
    "This course aligns with your goal in **{interest}** and is pitched at "
    "**{level}** level, a good fit for where you are now. (Match: {score:.0%})"
)
_T_SKILL_ONLY = (
    "Recommended because your existing **{skill}** knowledge lets you take on this "
    "**{level}** course and build upward quickly. (Match: {score:.0%})"
)
_T_GENERIC = (
    "This course ranked among the top results for your stated goal. It is a "
    "**{level}** course with a match score of {score:.0%}."
)
_T_PREREQUISITE = (
    "Included as groundwork rather than as a direct match: a later course in your "
    "path lists **{title}** as a prerequisite, so it comes first."
)

# ── Gap-coverage template ─────────────────────────────────────────────────────
# Takes priority over the similarity templates when coverage data exists, because
# "closes these three specific gaps in your skill set" is a stronger and more
# checkable reason than "scored 0.62 against your goal".
_T_COVERAGE = (
    "Recommended because it closes **{n} skill{plural}** you're missing: "
    "**{skills}**.{interest_clause}{share_clause}"
)

# ── Reinforcement templates ───────────────────────────────────────────────────
# Projects and assessments are not chosen for similarity at all. They are unlocked
# by what precedes them, so their explanation has to reference that instead of
# reporting a match score of zero.
_T_PROJECT = (
    "A project rather than a course: it puts the **{skills}** you'll have from "
    "{sources} into practice on something you can show. Roughly **{hours} hours**."
)
_T_ASSESSMENT = (
    "A checkpoint assessment: it confirms you've absorbed **{skills}** from "
    "{sources} before the path moves on. About **{hours} hour{hplural}**."
)
_T_REINFORCEMENT_FALLBACK = (
    "A **{kind}** placed here because everything it depends on comes earlier in "
    "your path. Roughly **{hours} hours**."
)


def explain_course(course: dict, profile: dict, similarity_score: float) -> str:
    """
    Return a short Markdown explanation for why ``course`` was recommended.

    Parameters
    ----------
    course : dict
        A course row.  ``is_prerequisite_filler`` is honoured when present.
    profile : dict
        Profile from ``profiling_engine.build_profile()``.
    similarity_score : float
        Composite match score in [0, 1].

    Returns
    -------
    str
        A single Markdown-formatted sentence.
    """
    profile = profile or {}
    level = str(course.get("difficulty_level", "beginner"))

    # Courses dragged in purely to satisfy a dependency were never scored, so
    # reporting a match percentage for them would be misleading.
    if course.get("is_prerequisite_filler"):
        return _T_PREREQUISITE.format(title=course.get("title", "this course"))

    # Projects and assessments are unlocked by what precedes them rather than
    # ranked, so they are explained by their dependencies, not by a score.
    if course.get("is_reinforcement"):
        return _explain_reinforcement(course)

    # Gap coverage, where available, is the strongest available reason.
    covered = course.get("covers") or []
    if covered:
        return _explain_coverage(course, profile, covered)

    interests = profile.get("interests") or []
    current_skills = [s.lower() for s in (profile.get("current_skills") or [])]
    target_skills = [s.lower() for s in (profile.get("target_skills") or [])]
    course_skills = parse_skills(course.get("skills", ""))
    category = str(course.get("category", ""))

    matched_interest = next(
        (i for i in interests if i.lower() in category.lower()), None
    ) or (interests[0] if interests else None)

    matched_skill = _first_overlap(current_skills, course_skills)
    matched_target = _first_overlap(target_skills, course_skills)

    if matched_interest and matched_skill:
        return _T_INTEREST_AND_SKILL.format(
            interest=matched_interest, skill=matched_skill, score=similarity_score
        )
    if matched_interest and matched_target:
        return _T_INTEREST_AND_TARGET.format(
            interest=matched_interest, target=matched_target, score=similarity_score
        )
    if matched_interest:
        return _T_INTEREST_ONLY.format(
            interest=matched_interest, level=level, score=similarity_score
        )
    if matched_skill:
        return _T_SKILL_ONLY.format(
            skill=matched_skill, level=level, score=similarity_score
        )
    return _T_GENERIC.format(level=level, score=similarity_score)


def _prerequisite_titles(resource: dict, limit: int = 3) -> str:
    """
    Render a resource's prerequisites as readable titles rather than bare IDs.

    "the SQL for Data Analysts course" reads considerably better than "DS005", and
    the whole point of a checkpoint explanation is naming the work it follows on
    from. Falls back to IDs for anything not in the catalog.
    """
    from app.config import load_catalog_map

    catalog = load_catalog_map()
    ids = [
        p.strip()
        for p in str(resource.get("prerequisites", "")).split(",")
        if p.strip() and p.strip().lower() != "nan"
    ]
    if not ids:
        return "your earlier steps"

    titles = [
        f"**{catalog[cid]['title']}**" if cid in catalog else f"**{cid}**"
        for cid in ids[:limit]
    ]
    rendered = ", ".join(titles[:-1]) + f" and {titles[-1]}" if len(titles) > 1 else titles[0]
    if len(ids) > limit:
        rendered += f" and {len(ids) - limit} more"
    return rendered


def _explain_reinforcement(resource: dict) -> str:
    """Explain a checkpoint project or assessment by what it builds on."""
    kind = str(resource.get("resource_type", "course"))
    skills = parse_skills(resource.get("skills", ""))
    hours = int(resource.get("duration_hours") or 0)
    sources = _prerequisite_titles(resource)

    if not skills:
        return _T_REINFORCEMENT_FALLBACK.format(kind=kind, hours=hours)

    listed = ", ".join(skills[:4])
    if kind == "project":
        return _T_PROJECT.format(skills=listed, sources=sources, hours=hours)
    if kind == "assessment":
        return _T_ASSESSMENT.format(
            skills=listed, sources=sources, hours=hours,
            hplural="" if hours == 1 else "s",
        )
    return _T_REINFORCEMENT_FALLBACK.format(kind=kind, hours=hours)


def _explain_coverage(course: dict, profile: dict, covered: list[str]) -> str:
    """Explain a course by the specific gap skills it closes."""
    interests = profile.get("interests") or []
    category = str(course.get("category", ""))
    matched_interest = next(
        (i for i in interests if i.lower() in category.lower()), None
    )

    interest_clause = (
        f" That's squarely in **{matched_interest}**, your stated focus."
        if matched_interest
        else ""
    )
    share = course.get("coverage_share")
    share_clause = (
        f" It accounts for **{share:.0%}** of what you still need."
        if isinstance(share, (int, float)) and share > 0
        else ""
    )

    return _T_COVERAGE.format(
        n=len(covered),
        plural="" if len(covered) == 1 else "s",
        skills=", ".join(covered[:5]) + ("..." if len(covered) > 5 else ""),
        interest_clause=interest_clause,
        share_clause=share_clause,
    )


def _first_overlap(user_skills: list[str], course_skills: list[str]) -> str | None:
    """
    Return the first user skill that overlaps a course skill, or ``None``.

    Overlap is substring-in-either-direction so that "sql" matches "sql joins"
    and "advanced sql" alike, which is the common shape of catalog skill tokens.
    """
    for skill in user_skills:
        if any(skill in cs or cs in skill for cs in course_skills):
            return skill
    return None


# ── Q&A intents ────────────────────────────────────────────────────────────────
# Ordered most specific first; the first pattern to match wins.

def _answer_why(question: str, course: dict, profile: dict, score: float) -> str:
    return explain_course(course, profile, score)


def _answer_prerequisites(question: str, course: dict, profile: dict, score: float) -> str:
    title = course.get("title", "this resource")
    kind = _kind_noun(course)
    parsed = [
        p.strip()
        for p in str(course.get("prerequisites", "")).split(",")
        if p.strip() and p.strip().lower() != "nan"
    ]
    if not parsed:
        return f"**{title}** has no prerequisites, so you can start it right away."

    # Named titles rather than bare IDs — "DS001, DS002" tells the learner nothing
    # about what they actually need to have covered.
    return (
        f"**{title}** expects you to complete {_prerequisite_titles(course, limit=5)} "
        f"first. Those appear earlier in your path, so following the sequence covers "
        f"them. ({len(parsed)} prerequisite{'' if len(parsed) == 1 else 's'} for this "
        f"{kind}.)"
    )


def _answer_too_hard(question: str, course: dict, profile: dict, score: float) -> str:
    title = course.get("title", "this course")
    level = str(course.get("difficulty_level", "intermediate"))
    easier = {"advanced": "intermediate", "intermediate": "beginner"}.get(level, "beginner")
    parsed = [
        p.strip()
        for p in str(course.get("prerequisites", "")).split(",")
        if p.strip() and p.strip().lower() != "nan"
    ]
    hint = f" Completing {', '.join(parsed)} first should close the gap." if parsed else ""
    return (
        f"**{title}** sits at **{level}** level. If it feels like a stretch, drop to a "
        f"**{easier}** course in the same category and come back to it.{hint}"
    )


def _answer_too_easy(question: str, course: dict, profile: dict, score: float) -> str:
    title = course.get("title", "this course")
    return (
        f"If **{title}** covers ground you already have, mark it 'Too Easy' on the "
        f"dashboard. That drops it from your path, and repeated reports raise your "
        f"level so later suggestions get harder."
    )


def _answer_already_know(question: str, course: dict, profile: dict, score: float) -> str:
    title = course.get("title", "this course")
    level = str(course.get("difficulty_level", "beginner"))
    harder = {"beginner": "intermediate", "intermediate": "advanced"}.get(level, "advanced")
    return (
        f"Then you can skip **{title}** and look for {_article_for(harder)} **{harder}** "
        f"course in the same category instead. Marking it 'Too Easy' re-ranks your path "
        f"to do exactly that."
    )


def _article_for(word: str) -> str:
    """Return "an" before a vowel sound and "a" otherwise."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _kind_noun(resource: dict) -> str:
    """
    Return the right noun for a resource.

    The Q&A used to call everything a "course", which reads plainly wrong once a
    path contains projects and assessments.
    """
    return {
        "course": "course",
        "project": "project",
        "assessment": "assessment",
    }.get(str(resource.get("resource_type", "course")), "resource")


def _answer_duration(question: str, course: dict, profile: dict, score: float) -> str:
    """
    Report duration from the catalog rather than a per-level guess.

    Every resource now carries ``duration_hours``, so the previous band estimates
    ("roughly 10 to 20 hours") were guessing at a number already on record — and
    guessing badly for assessments, which take an hour regardless of level.
    """
    kind = _kind_noun(course)
    title = course.get("title", "this one")
    hours = int(course.get("duration_hours") or 0)

    if hours:
        return (
            f"**{title}** is a {kind} of about **{hours} hour"
            f"{'' if hours == 1 else 's'}**, depending on your pace and background."
        )

    level = str(course.get("difficulty_level", "beginner"))
    estimates = {
        "beginner": "roughly 4 to 8 hours",
        "intermediate": "roughly 10 to 20 hours",
        "advanced": "roughly 20 to 40 hours",
    }
    return (
        f"{_article_for(level).capitalize()} **{level}** {kind} like **{title}** "
        f"typically takes {estimates.get(level, 'a few hours')}."
    )


def _answer_skills(question: str, course: dict, profile: dict, score: float) -> str:
    """
    Describe the skills involved, using the verb the resource type deserves.

    A course *teaches* skills, a project *exercises* them and an assessment
    *checks* them. Saying "completing this assessment builds SQL" would be
    misleading — it verifies SQL you were taught earlier.
    """
    title = course.get("title", "this resource")
    kind = str(course.get("resource_type", "course"))
    skills = parse_skills(course.get("skills", ""))

    if not skills:
        return (
            f"**{title}** covers skills relevant to "
            f"{course.get('category', 'your goal')}."
        )

    listed = ", ".join(f"**{s}**" for s in skills)
    verb = {
        "course": "teaches you",
        "project": "puts into practice",
        "assessment": "checks your grasp of",
    }.get(kind, "covers")

    suffix = ""
    if kind == "course":
        held = {s.lower() for s in (profile or {}).get("current_skills", [])}
        new_skills = [s for s in skills if s not in held]
        if held and new_skills and len(new_skills) < len(skills):
            suffix = f" Of those, {', '.join(new_skills)} would be new for you."
    else:
        suffix = f" It builds on {_prerequisite_titles(course)}."

    # When gap data is present, say which of these you actually still need.
    covered = course.get("covers") or []
    if covered:
        suffix += (
            f" **{len(covered)}** of them are currently missing from your skill "
            f"set: {', '.join(covered[:5])}."
        )

    return f"**{title}** {verb}: {listed}.{suffix}"


#: (pattern, handler) in priority order. Specific intents precede general ones.
_QA_INTENTS: list[tuple[re.Pattern[str], object]] = [
    # "why", "explain", "reason" — unambiguous, so it can lead.
    (re.compile(r"\b(?:why|reason|explain|justif\w+)\b", re.IGNORECASE), _answer_why),
    # Must precede already-know: "What prerequisites do I need?" contains "do I need".
    (
        re.compile(
            r"\b(?:prerequisite|prerequisites|prereq\w*|required?\s+first|"
            r"need\s+to\s+know\s+first|before\s+(?:this|starting)|come\s+first)\b",
            re.IGNORECASE,
        ),
        _answer_prerequisites,
    ),
    (
        re.compile(
            r"\b(?:too\s+hard|too\s+difficult|too\s+advanced|can\s+i\s+handle|"
            r"over\s+my\s+head|struggle)\b",
            re.IGNORECASE,
        ),
        _answer_too_hard,
    ),
    (
        re.compile(r"\b(?:too\s+easy|too\s+basic|boring|beneath|below\s+my)\b", re.IGNORECASE),
        _answer_too_easy,
    ),
    (
        re.compile(
            r"\b(?:already\s+know|already\s+done|already\s+use|i\s+know|skip|"
            r"do\s+i\s+need|must\s+i)\b",
            re.IGNORECASE,
        ),
        _answer_already_know,
    ),
    (
        re.compile(
            r"\b(?:how\s+long|duration|hours|weeks|months|time\s+(?:does|will|to))\b",
            re.IGNORECASE,
        ),
        _answer_duration,
    ),
    (
        re.compile(
            r"\b(?:skills?|learn|gain|outcome|get\s+out\s+of|teach|cover)\b", re.IGNORECASE
        ),
        _answer_skills,
    ),
]


def answer_question(
    question: str,
    course: dict,
    profile: dict,
    similarity_score: float | None = None,
) -> str:
    """
    Match ``question`` against the ordered intents and return an answer.

    Parameters
    ----------
    question : str
        The user's question.
    course : dict
        The course under discussion.
    profile : dict
        The user's structured profile.
    similarity_score : float | None
        The course's match score, used by the "why" intent.  Defaults to the
        score already carried on the course dict.  The previous version hardcoded
        0.0 here and then stripped "0%" back out of the rendered string, which
        threw away information the caller already had.

    Returns
    -------
    str
        A Markdown-formatted answer.
    """
    if similarity_score is None:
        similarity_score = float(course.get("similarity_score") or 0.0)

    for pattern, handler in _QA_INTENTS:
        if pattern.search(question):
            return handler(question, course, profile or {}, similarity_score)

    return (
        f"I don't have a specific answer for that. **{course.get('title', 'This course')}** "
        f"is a {course.get('difficulty_level', 'beginner')} course in "
        f"{course.get('category', 'this category')}. Try asking why it was recommended, "
        f"what prerequisites it has, what skills it covers, or how long it takes."
    )


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    course = {
        "course_id": "DS007",
        "title": "Machine Learning with Scikit-Learn",
        "description": "Implement and tune ML algorithms.",
        "skills": "machine learning,scikit-learn,regression,classification",
        "prerequisites": "DS001,DS002,DS003",
        "difficulty_level": "intermediate",
        "category": "Data Science",
        "similarity_score": 0.87,
    }
    profile = {
        "goal": "I want to become a data analyst, I know some Python and Excel",
        "current_skills": ["python", "excel"],
        "target_skills": ["machine learning"],
        "experience_level": "intermediate",
        "interests": ["Data Science"],
    }

    from app.config import load_catalog_map

    print("-- Explanation " + "-" * 46)
    explanation = explain_course(course, profile, 0.87)
    print(f"  {explanation}")
    assert "87%" in explanation, explanation
    assert "Data Science" in explanation, explanation

    # ── REGRESSION: each question must reach its own intent ──────────────────
    routing_cases = [
        ("Why was this course recommended?", "why"),
        ("What prerequisites do I need?", "prerequisites"),
        ("Do I need anything before this?", "prerequisites"),
        ("Is this too hard for me?", "too_hard"),
        ("This looks too easy", "too_easy"),
        ("What if I already know Python?", "already_know"),
        ("How long will this take?", "duration"),
        ("What skills will I gain?", "skills"),
    ]
    handler_names = {
        "why": _answer_why,
        "prerequisites": _answer_prerequisites,
        "too_hard": _answer_too_hard,
        "too_easy": _answer_too_easy,
        "already_know": _answer_already_know,
        "duration": _answer_duration,
        "skills": _answer_skills,
    }

    print("\n-- Intent routing " + "-" * 43)
    for question, expected in routing_cases:
        matched = next(
            (h for p, h in _QA_INTENTS if p.search(question)), None
        )
        assert matched is handler_names[expected], (
            f"{question!r} routed to {getattr(matched, '__name__', matched)}, "
            f"expected {handler_names[expected].__name__}"
        )
        print(f"  OK  {question:<40} -> {expected}")

    # The specific regression that was failing: this question previously hit the
    # already-know handler and returned skip advice instead of the prereq list.
    prereq_answer = answer_question(
        "What prerequisites do I need?", load_catalog_map()["DS007"], profile
    )
    assert "skip" not in prereq_answer.lower(), prereq_answer
    # Titles, not bare IDs.
    assert "Introduction to Python for Data Science" in prereq_answer, prereq_answer
    assert "3 prerequisites" in prereq_answer, prereq_answer

    no_prereqs = answer_question(
        "What prerequisites do I need?", load_catalog_map()["DS001"], profile
    )
    assert "no prerequisites" in no_prereqs, no_prereqs

    # ── the "why" intent must carry the real score, not a hardcoded 0% ────────
    why_answer = answer_question("Why this course?", course, profile, similarity_score=0.87)
    assert "87%" in why_answer, f"REGRESSION: real score not threaded through -> {why_answer}"
    why_default = answer_question("Why this course?", course, profile)
    assert "87%" in why_default, f"score should fall back to the course dict -> {why_default}"

    # ── prerequisite fillers are explained as groundwork, not as a 0% match ───
    filler = dict(course, is_prerequisite_filler=True, similarity_score=0.0)
    filler_text = explain_course(filler, profile, 0.0)
    assert "groundwork" in filler_text, filler_text
    assert "0%" not in filler_text, f"filler must not advertise a 0% match -> {filler_text}"

    # ── a beginner with no held skills must not be told a course "builds on" one ─
    beginner_profile = {
        "goal": "complete beginner, want to learn web development with React",
        "current_skills": [],
        "target_skills": ["react"],
        "experience_level": "beginner",
        "interests": ["Web Development"],
    }
    react_course = {
        "title": "React Fundamentals", "skills": "react,jsx,components",
        "difficulty_level": "beginner", "category": "Web Development",
        "prerequisites": "", "similarity_score": 0.8,
    }
    beginner_text = explain_course(react_course, beginner_profile, 0.8)
    print(f"\n-- Beginner explanation " + "-" * 37)
    print(f"  {beginner_text}")
    assert "builds on" not in beginner_text, (
        f"REGRESSION: claimed experience a beginner does not have -> {beginner_text}"
    )
    assert "want to learn" in beginner_text, beginner_text

    # ── gap coverage outranks the similarity templates ───────────────────────
    with_coverage = dict(
        course,
        covers=["machine learning", "scikit-learn", "regression"],
        coverage_share=0.34,
    )
    coverage_text = explain_course(with_coverage, profile, 0.62)
    print()
    print("-- coverage explanation " + "-" * 37)
    print(f"  {coverage_text}")
    assert "3 skills" in coverage_text, coverage_text
    assert "machine learning" in coverage_text, coverage_text
    assert "34%" in coverage_text, coverage_text
    assert "Data Science" in coverage_text, coverage_text
    # A single covered skill must not read "1 skills".
    single = explain_course(dict(course, covers=["regression"]), profile, 0.5)
    assert "**1 skill**" in single, single
    assert "1 skills" not in single, single

    # ── projects and assessments are explained by what unlocks them ──────────
    from app.config import load_catalog_map as _catalog

    catalog = _catalog()

    project = dict(catalog["PR001"], is_reinforcement=True, similarity_score=0.0)
    project_text = explain_course(project, profile, 0.0)
    print()
    print("-- project explanation " + "-" * 38)
    print(f"  {project_text}")
    assert "project rather than a course" in project_text, project_text
    # Prerequisites must be named as titles, not raw IDs.
    assert "SQL for Data Analysts" in project_text, project_text
    assert "DS005" not in project_text, "prerequisites should read as titles, not IDs"
    assert "0%" not in project_text, "a reinforcement item must not advertise a score"
    assert "10 hours" in project_text, project_text

    assessment = dict(catalog["AS002"], is_reinforcement=True, similarity_score=0.0)
    assessment_text = explain_course(assessment, profile, 0.0)
    print()
    print("-- assessment explanation " + "-" * 35)
    print(f"  {assessment_text}")
    assert "checkpoint assessment" in assessment_text, assessment_text
    assert "SQL for Data Analysts" in assessment_text, assessment_text
    assert "1 hour" in assessment_text and "1 hours" not in assessment_text, assessment_text

    # ── the Q&A uses the right noun and verb per type ────────────────────────
    assert _kind_noun(catalog["DS005"]) == "course"
    assert _kind_noun(catalog["PR001"]) == "project"
    assert _kind_noun(catalog["AS002"]) == "assessment"

    project_skills = answer_question("What skills will I gain?", project, profile)
    assert "puts into practice" in project_skills, project_skills
    assert "builds on" in project_skills, project_skills

    assessment_skills = answer_question("What skills does this cover?", assessment, profile)
    assert "checks your grasp of" in assessment_skills, assessment_skills

    course_skills = answer_question("What skills will I gain?", catalog["DS005"], profile)
    assert "teaches you" in course_skills, course_skills

    # ── duration comes from the catalog, not a per-level guess ───────────────
    duration = answer_question("How long will this take?", catalog["AS002"], profile)
    assert "1 hour" in duration, duration
    assert "10 to 20" not in duration, "should use the recorded duration, not a band"
    project_duration = answer_question("How long will this take?", catalog["PR001"], profile)
    assert "10 hours" in project_duration and "project" in project_duration, project_duration

    # ── unmatched questions fall back gracefully ─────────────────────────────
    fallback = answer_question("zzzz qqqq", course, profile)
    assert "don't have a specific answer" in fallback, fallback

    print("\n-- Sample answers " + "-" * 43)
    for question, _ in routing_cases:
        print(f"\n  Q: {question}")
        print(f"  A: {answer_question(question, course, profile)}")

    print("\nexplainer.py self-test passed: all assertions OK")
