"""
profiling_engine.py — Turn raw user text into a structured profile.

Hybrid approach (rules + embeddings), as the three stages below.

1.  Clause-level skill extraction with polarity
    The text is split into clauses and each clause is tagged as describing
    something the user *has* or something the user *wants*.  Skills from the
    catalog vocabulary are then routed to ``current_skills`` or
    ``target_skills`` accordingly.

    This replaced flat substring matching over the whole sentence, which could
    not tell the two apart.  "I'm a complete beginner and want to learn web
    development with React" previously yielded ``current_skills: ['react']``,
    so the explainer went on to tell a self-declared beginner that a course
    "builds on your React experience".

2.  Experience-level inference
    A priority ladder: an explicit self-declaration ("complete beginner", "no
    experience") wins outright, then an extracted years-of-experience figure,
    then seniority keywords, then weaker keyword hints.

    The years patterns tolerate words between the number and the noun.  The
    previous pattern required "years of experience" to be adjacent, so "5 years
    of Python experience" failed to match and fell through to the default of
    beginner — handing a five-year Python developer a beginner path.

3.  Embedding-based domain classification
    The goal text is embedded and compared by cosine similarity against
    embeddings of the category labels.  Labels are kept only if they score
    close to the best label, so a focused goal yields one or two domains rather
    than always exactly three.

4.  Prior-learning detection
    Resources the learner says they have already completed are pulled out of the
    text and recorded, so the path can route around them. The brief asks the
    profiling engine to capture "completed courses" and "previous learning
    history"; this is that requirement.

Profile schema
--------------
    {
        "goal": str,
        "current_skills": [str],       # skills the user says they already have
        "target_skills":  [str],       # skills the user says they want
        "experience_level": "beginner" | "intermediate" | "advanced",
        "interests": [str],            # catalog category names
        "completed_resources": [str],  # catalog IDs already finished
        "preferences": {...},          # see learner_model.LearnerPreferences
    }

The profile is persisted to SQLite via ``db.upsert_profile()``.
"""

from __future__ import annotations

import re

import numpy as np

from app.config import load_catalog_map, load_skill_vocabulary
from app.db import (
    get_preferences,
    get_prior_learning_ids,
    get_profile,
    set_preferences,
    set_prior_learning,
    upsert_profile,
)
from app.learner_model import LearnerPreferences, extract_preferences
from app.recommendation_engine import get_engine

# ── Domain classification ─────────────────────────────────────────────────────
# Embedded once and cached, then compared against the user's goal text.
CATEGORY_LABELS = [
    "data science",
    "web development",
    "UX UI design",
    "business marketing",
    "cloud devops",
]

LABEL_TO_CATEGORY = {
    "data science": "Data Science",
    "web development": "Web Development",
    "UX UI design": "UX/UI Design",
    "business marketing": "Business/Marketing",
    "cloud devops": "Cloud/DevOps",
}

#: Keep a category only if it scores at least this fraction of the top category.
#: Without a cutoff the classifier returned a fixed top-3, so a purely
#: data-analytics goal still listed "Web Development" as an interest and the
#: explainer would cite it as something the user cares about.
INTEREST_RATIO = 0.72

#: Absolute floor, so a goal that matches nothing well does not drag in noise.
INTEREST_MIN_SIM = 0.15

#: Never return more than this many interests.
INTEREST_MAX_K = 3

# ── Clause segmentation ───────────────────────────────────────────────────────
# Sentence enders, commas, and coordinating conjunctions all begin a new clause.
# "also" is deliberately NOT a split token. It is an adverb, not a coordinating
# conjunction, so splitting on it tore "I also know Python" into "I" and
# "know Python" — stripping the subject and defeating the possession cue, which
# then filed Python as a skill the learner *wanted*.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[.;!?\n]+|,|\s+\b(?:and|but|then|while|although|however|though|plus|so)\b\s+)",
    re.IGNORECASE,
)

# Cues that the clause describes something the user WANTS to acquire.
# Checked before possession cues, because "want to learn X" contains "learn".
_ASPIRATION_RE = re.compile(
    r"\b(?:want|wanna|wish|hope|aim|aspire|aspiring|goal|become|becoming|"
    r"get\s+into|getting\s+into|break\s+into|switch\s+to|switching|transition|"
    r"interested\s+in|looking\s+to|plan\s+to|planning|trying\s+to|need\s+to\s+learn|"
    r"would\s+like|curious\s+about|pick\s+up|pursue|explore|"
    r"learn|study|master|upskill|start)\b",
    re.IGNORECASE,
)

# An optional adverb between subject and verb. Without this, "I also know Python"
# failed to register as possession — the pattern required "i know" to be adjacent —
# and the skill was filed as something the learner *wanted* instead of had.
_ADV = r"(?:\s+(?:also|already|currently|actually|really|basically|definitely|" \
       r"sort\s+of|kind\s+of|still))?"

# Cues that the clause describes something the user ALREADY HAS.
_POSSESSION_RE = re.compile(
    r"\b(?:"
    rf"i{_ADV}\s+know|i've{_ADV}\s+used|i{_ADV}\s+have\s+used|i{_ADV}\s+use|"
    rf"i{_ADV}\s+can|i{_ADV}\s+did|i{_ADV}\s+have|i've|i{_ADV}\s+am\s+a|"
    r"familiar\s+with|experienced\s+(?:in|with)|experience\s+(?:in|with)|"
    r"proficient\s+(?:in|with)|comfortable\s+(?:in|with)|confident\s+(?:in|with)|"
    r"worked\s+with|working\s+with|background\s+in|skilled\s+(?:in|at)|"
    r"good\s+at|strong\s+(?:in|at)|hands[\s\-]?on\s+(?:in|with)|"
    r"studied|learnt|learned|trained\s+in|my\s+experience"
    r")\b"
    # Bare verbs, for clauses where splitting has removed the subject
    # ("I know Python, use Docker daily" -> "use Docker daily"). Safe because
    # aspiration is tested first, so "want to use Docker" still reads as a want.
    r"|^\s*(?:know|knows|knew|use|uses|used|using)\b",
    re.IGNORECASE,
)

# ── Experience-level detection ────────────────────────────────────────────────
# "5 years of Python experience", "3 years experience", "10+ yrs of dev work".
_YEARS_FORWARD_RE = re.compile(
    r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b"
    r"(?:\s+(?:of|in|as|with|doing|using|writing|building))?"
    r"(?:\s+[\w\+\#\./\-]+){0,4}?"
    r"\s+(?:experience|exp|background|working|work|professionally|career|industry|"
    r"development|dev|engineering)\b",
    re.IGNORECASE,
)

# "I've been coding in Python for 6 years", "worked with Java for 8 years".
_YEARS_BACKWARD_RE = re.compile(
    r"\b(?:experience|experienced|working|worked|been|using|used|developing|"
    r"coding|programming|writing|building|shipping)\b"
    r"[^.;!?]{0,40}?"
    r"\b(?:for|over|about|around|nearly|almost)?\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

# An unambiguous self-declaration of being a beginner outranks everything else,
# because it is the user stating their own level directly.
_EXPLICIT_BEGINNER_RE = re.compile(
    r"\b(?:complete|absolute|total|utter)\s+beginner\b"
    r"|\bno\s+(?:prior\s+|previous\s+|real\s+)?experience\b"
    r"|\bzero\s+experience\b"
    r"|\bnever\s+(?:used|coded|programmed|written|done|worked|touched|tried)\b"
    r"|\bbrand[\s\-]?new\s+to\b"
    r"|\bstarting\s+from\s+(?:scratch|zero|nothing)\b"
    r"|\bdon'?t\s+know\s+(?:anything|any|much)\b",
    re.IGNORECASE,
)

_ADVANCED_RE = re.compile(
    r"\b(?:advanced|expert|senior|staff|principal|lead\s+(?:engineer|developer)|"
    r"proficient|seasoned|veteran|extensive\s+experience|deep\s+experience|"
    r"years\s+of\s+experience)\b",
    re.IGNORECASE,
)

_INTERMEDIATE_RE = re.compile(
    r"\b(?:intermediate|some\s+experience|a\s+bit\s+of\s+experience|"
    r"familiar\s+with|worked\s+with|dabbled|"
    r"i\s+know\s+(?:a\s+bit|some|basic|the\s+basics)|basic\s+knowledge|"
    r"comfortable\s+with|fairly\s+comfortable|"
    r"i\s+can\s+(?:write|code|build|use|make))\b",
    re.IGNORECASE,
)

_BEGINNER_RE = re.compile(
    r"\b(?:beginner|newbie|novice|new\s+to|just\s+starting|start\s+learning|"
    r"want\s+to\s+learn|from\s+scratch|no\s+idea)\b",
    re.IGNORECASE,
)

#: Years at or above this count read as advanced; 1 to this-1 reads as intermediate.
_ADVANCED_YEARS = 4

# ── Prior learning detection ──────────────────────────────────────────────────
# Cues that a clause describes study the learner has already finished, as opposed
# to a skill they merely hold. "I know Python" is a skill; "I finished the Python
# intro course" is a completed resource, and the difference matters because the
# second one can be excluded from the path by ID.
_COMPLETION_RE = re.compile(
    r"\b(?:completed|finished|done|took|taken|passed|graduated|"
    r"went\s+through|worked\s+through|already\s+did|have\s+done)\b",
    re.IGNORECASE,
)

#: Catalog IDs mentioned verbatim, e.g. "I've done DS001 and DS003".
_RESOURCE_ID_RE = re.compile(r"\b((?:DS|WD|UX|BM|CD|PR|AS)\d{3})\b", re.IGNORECASE)

#: A title has to overlap this strongly with the clause before it counts as a
#: match. Set high because a false positive silently removes a resource the
#: learner may actually need.
_TITLE_MATCH_RATIO = 0.6

# Cache for the category-label embedding matrix.
_label_embeddings: np.ndarray | None = None


# ── Public API ─────────────────────────────────────────────────────────────────

def build_profile(session_id: str, raw_text: str) -> dict:
    """
    Parse ``raw_text`` into a structured profile, persist it, and return it.

    Parameters
    ----------
    session_id : str
        Opaque session identifier from Streamlit session state.
    raw_text : str
        The user's free-form goal text.

    Returns
    -------
    dict
        ``{goal, current_skills, target_skills, experience_level, interests}``
    """
    goal = raw_text.strip()

    current_skills, target_skills = extract_skills_by_polarity(goal)
    experience_level = classify_experience(goal)
    interests = classify_interests(goal)

    # Resources named in the text as already finished are merged with anything the
    # learner picked explicitly in the UI, so prose and picker agree rather than
    # one silently overwriting the other.
    mentioned = extract_completed_resources(goal)
    if mentioned:
        combined = sorted(get_prior_learning_ids(session_id) | set(mentioned))
        set_prior_learning(session_id, combined)
    completed_resources = sorted(get_prior_learning_ids(session_id))

    # Preferences stated in this message override stored ones, but anything not
    # mentioned is left alone. A learner who set "6 hours a week" in the UI and
    # then types a goal without mentioning time keeps their six hours.
    stated = extract_preferences(goal)
    if stated:
        merged = {**get_preferences(session_id), **stated}
        set_preferences(session_id, merged)
    preferences = LearnerPreferences.from_dict(get_preferences(session_id))

    profile = {
        "goal": goal,
        "current_skills": current_skills,
        "target_skills": target_skills,
        "experience_level": experience_level,
        "interests": interests,
        "completed_resources": completed_resources,
        "preferences": preferences.to_dict(),
    }

    upsert_profile(
        session_id=session_id,
        goal=profile["goal"],
        current_skills=profile["current_skills"],
        experience_level=profile["experience_level"],
        interests=profile["interests"],
        target_skills=profile["target_skills"],
    )
    return profile


def load_profile(session_id: str) -> dict | None:
    """
    Retrieve a previously persisted profile from SQLite, in the shape
    ``build_profile()`` returns.

    The raw database row carries bookkeeping columns (``session_id``,
    ``updated_at``) and lacks ``completed_resources``, so it is not
    interchangeable with a freshly built profile. Normalising here means callers
    restoring a session get exactly the same structure as callers who just parsed
    a goal, rather than a nearly-identical dict that fails in one place.

    Returns ``None`` when there is no profile, or when a row exists but carries no
    goal — which happens when preferences were saved before any goal was stated.
    """
    row = get_profile(session_id)
    if row is None or not str(row.get("goal") or "").strip():
        return None

    return {
        "goal": row["goal"],
        "current_skills": row.get("current_skills") or [],
        "target_skills": row.get("target_skills") or [],
        "experience_level": row.get("experience_level") or "beginner",
        "interests": row.get("interests") or [],
        "completed_resources": sorted(get_prior_learning_ids(session_id)),
        "preferences": LearnerPreferences.from_dict(row.get("preferences")).to_dict(),
    }


def extract_completed_resources(text: str) -> list[str]:
    """
    Detect resources the learner says they have already completed.

    The brief asks the profiling engine to capture completed courses and previous
    learning history. Two conservative signals are used:

    1.  **Verbatim catalog IDs** anywhere in the text ("I've done DS001, DS003").
        Unambiguous, so no completion cue is required.
    2.  **Course titles inside a clause carrying a completion cue** ("I finished
        the SQL for Data Analysts course"). A title must overlap the clause by
        ``_TITLE_MATCH_RATIO`` of its significant words before it counts.

    Both are deliberately strict. A false positive removes a resource from the
    learner's path silently, so the threshold favours missing a mention over
    inventing one. The UI offers an explicit picker for the reliable route; this
    function only catches what someone happens to mention in prose.

    Returns
    -------
    list[str]
        Catalog IDs, in catalog order, deduplicated.
    """
    if not text or not text.strip():
        return []

    catalog = load_catalog_map()
    found: set[str] = set()

    for match in _RESOURCE_ID_RE.findall(text):
        candidate = match.upper()
        if candidate in catalog:
            found.add(candidate)

    # Words too generic to carry evidence of a specific title.
    stop = {
        "the", "a", "an", "and", "or", "for", "with", "to", "of", "in", "on",
        "course", "project", "assessment", "introduction", "intro", "fundamentals",
        "basics", "advanced", "beginner", "essentials", "your", "my",
    }

    for clause in split_clauses(text):
        if not _COMPLETION_RE.search(clause):
            continue
        clause_words = {w.strip(".,!?:;()").lower() for w in clause.split()}
        for cid, row in catalog.items():
            if cid in found:
                continue
            title_words = {
                w.strip(".,!?:;()").lower() for w in str(row["title"]).split()
            } - stop
            if not title_words:
                continue
            overlap = len(title_words & clause_words) / len(title_words)
            if overlap >= _TITLE_MATCH_RATIO:
                found.add(cid)

    return [cid for cid in catalog if cid in found]


# ── Stage 1: skill extraction with polarity ────────────────────────────────────

def split_clauses(text: str) -> list[str]:
    """
    Split ``text`` into clauses on sentence enders, commas and conjunctions.

    Splitting on commas is deliberate: it lets "I know Python, JavaScript and
    SQL" distribute one possession cue across all three skills via the
    carry-forward rule in ``extract_skills_by_polarity``.
    """
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(text) if part and part.strip()]


def clause_polarity(clause: str) -> str | None:
    """
    Classify a clause as ``'have'``, ``'want'``, or ``None`` when it has no cue.

    Aspiration is tested first: "want to learn Django" contains the possession-
    adjacent verb "learn", but the sentence is plainly about a future goal.
    """
    if _ASPIRATION_RE.search(clause):
        return "want"
    if _POSSESSION_RE.search(clause):
        return "have"
    return None


def extract_skills_by_polarity(text: str) -> tuple[list[str], list[str]]:
    """
    Return ``(current_skills, target_skills)`` extracted from ``text``.

    Each clause inherits the polarity of the previous clause when it carries no
    cue of its own, so a trailing list item like the "cloud" in "...want to get
    into MLOps and cloud" is treated as aspirational rather than possessed.

    Clauses before any cue appears default to ``target``: claiming a skill the
    user never said they had is the more damaging error, because it feeds
    explanations that flatter the user with experience they lack.
    """
    vocab = load_skill_vocabulary()
    current: list[str] = []
    target: list[str] = []
    carried: str | None = None

    for clause in split_clauses(text):
        polarity = clause_polarity(clause) or carried
        if polarity:
            carried = polarity

        matched = _match_skills(clause.lower(), vocab)
        if not matched:
            continue

        bucket = current if polarity == "have" else target
        for skill in matched:
            if skill not in bucket:
                bucket.append(skill)

    # A skill the user both has and wants stays in current_skills only; having
    # it is the stronger, more actionable claim.
    target = [s for s in target if s not in current]
    return sorted(current), sorted(target)


def _match_skills(clause_lower: str, vocab: frozenset[str]) -> list[str]:
    """
    Return catalog skills appearing in ``clause_lower`` as whole words.

    Word boundaries prevent the single-letter skill "r" from matching inside
    "developer".  Token-level containment then collapses redundant hits: given
    "sql joins", both "sql" and "sql joins" match, and only the more specific
    one is kept.  The check is token-based rather than substring-based so that
    "sql" survives alongside "nosql", which merely contains it as a substring.
    """
    hits = [skill for skill in vocab if re.search(rf"\b{re.escape(skill)}\b", clause_lower)]

    specific: list[str] = []
    for skill in hits:
        tokens = set(skill.split())
        subsumed = any(
            other != skill and tokens < set(other.split())
            for other in hits
        )
        if not subsumed:
            specific.append(skill)
    return sorted(specific)


# ── Stage 2: experience level ──────────────────────────────────────────────────

def extract_years_of_experience(text: str) -> int | None:
    """
    Return the largest plausible years-of-experience figure in ``text``, or None.

    Both orderings are matched: "5 years of Python experience" (number first)
    and "been coding in Python for 6 years" (number last).  When several
    figures appear the largest wins, since users tend to cite their longest
    tenure when listing multiple technologies.
    """
    found = [
        int(m)
        for pattern in (_YEARS_FORWARD_RE, _YEARS_BACKWARD_RE)
        for m in pattern.findall(text)
    ]
    plausible = [y for y in found if 0 < y <= 50]
    return max(plausible) if plausible else None


def classify_experience(text: str) -> str:
    """
    Infer the user's experience level via a priority ladder.

    Order of precedence, highest first:

    1. Explicit self-declaration of being a beginner — the user's own direct
       statement about themselves.
    2. An extracted years figure: >= 4 years is advanced, 1-3 is intermediate.
    3. Seniority keywords ("senior", "expert", "proficient").
    4. Intermediate keywords ("some experience", "familiar with").
    5. Beginner keywords ("new to", "want to learn").
    6. Default: beginner, the safer failure mode for recommendations.
    """
    if _EXPLICIT_BEGINNER_RE.search(text):
        return "beginner"

    years = extract_years_of_experience(text)
    if years is not None:
        return "advanced" if years >= _ADVANCED_YEARS else "intermediate"

    if _ADVANCED_RE.search(text):
        return "advanced"
    if _INTERMEDIATE_RE.search(text):
        return "intermediate"
    if _BEGINNER_RE.search(text):
        return "beginner"
    return "beginner"


# ── Stage 3: embedding-based domain classification ─────────────────────────────

def _get_label_embeddings() -> np.ndarray:
    """Lazily embed the category labels once and cache the matrix."""
    global _label_embeddings
    if _label_embeddings is None:
        _label_embeddings = get_engine().embed_texts(CATEGORY_LABELS)
    return _label_embeddings


def classify_interests(text: str, max_k: int = INTEREST_MAX_K) -> list[str]:
    """
    Rank catalog categories against ``text`` by cosine similarity and return
    those close enough to the best match.

    A category is kept when its score clears both ``INTEREST_MIN_SIM`` and
    ``INTEREST_RATIO`` times the top score.  The single best category is always
    returned, so the caller never has to handle an empty list.
    """
    engine = get_engine()
    label_embs = _get_label_embeddings()
    query_emb = engine.embed_texts([text])

    sims = (label_embs @ query_emb.T).ravel()
    order = np.argsort(sims)[::-1]

    top_sim = float(sims[order[0]])
    cutoff = max(INTEREST_MIN_SIM, top_sim * INTEREST_RATIO)

    kept = [
        LABEL_TO_CATEGORY[CATEGORY_LABELS[i]]
        for i in order[:max_k]
        if float(sims[i]) >= cutoff
    ]
    # Always keep the best match even if it fell below the absolute floor.
    if not kept:
        kept = [LABEL_TO_CATEGORY[CATEGORY_LABELS[order[0]]]]
    return kept


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from app.db import _get_connection, init_db

    init_db()

    # ── Stage 2: experience classification, incl. the regression case ────────
    experience_cases = [
        # (text, expected_level)
        ("I have 5 years of Python experience and want to get into MLOps.", "advanced"),
        ("I've been coding in Java for 8 years.", "advanced"),
        ("10+ years of software engineering experience.", "advanced"),
        ("I have 2 years of experience in marketing.", "intermediate"),
        ("3 years experience with figma.", "intermediate"),
        ("I'm a complete beginner and want to learn web development.", "beginner"),
        ("I have no prior experience but want to learn data science.", "beginner"),
        ("I want to learn SQL.", "beginner"),
        ("I know some Excel and basic SQL.", "intermediate"),
        ("I'm a senior developer moving into design.", "advanced"),
        ("I want to finish this in 2 years.", "beginner"),  # a deadline, not experience
    ]
    for text, expected in experience_cases:
        got = classify_experience(text)
        assert got == expected, f"classify_experience({text!r}) -> {got!r}, expected {expected!r}"

    assert extract_years_of_experience("5 years of Python experience") == 5
    assert extract_years_of_experience("been using Go for 7 years") == 7
    assert extract_years_of_experience("I want to finish in 2 years") is None
    print(f"experience classification: {len(experience_cases)} cases OK")

    # ── Stage 1: skill polarity, incl. the regression case ───────────────────
    have, want = extract_skills_by_polarity(
        "I'm a complete beginner and want to learn web development with React."
    )
    assert "react" not in have, f"REGRESSION: 'react' claimed as a held skill -> {have}"
    assert "react" in want, f"'react' should be a target skill -> {want}"

    have, want = extract_skills_by_polarity(
        "I want to become a data analyst. I know some Excel and basic SQL."
    )
    assert "excel" in have, have
    assert "sql" in have, f"carry-forward should put SQL in current skills -> {have}"

    have, want = extract_skills_by_polarity(
        "I have 5 years of Python experience and want to get into MLOps and cloud."
    )
    assert "python" in have, have
    assert "python" not in want, want
    assert any("mlops" in s for s in want), want

    # REGRESSION: an adverb between subject and verb. "also" was a clause-split
    # token, which tore "I also know Python" into "I" / "know Python" and lost the
    # subject; and even unsplit, the pattern required "i know" to be adjacent. The
    # skill ended up filed as something wanted rather than held.
    have, want = extract_skills_by_polarity("I also know Python")
    assert "python" in have, f"'I also know Python' -> held={have} wanted={want}"
    assert "python" not in want

    for phrasing in (
        "I already know Docker",
        "I currently use Kubernetes",
        "I actually have experience with Terraform",
    ):
        have, want = extract_skills_by_polarity(phrasing)
        assert have, f"{phrasing!r} registered no held skills (wanted={want})"

    # The adverb slot must not turn a want into a have.
    have, want = extract_skills_by_polarity("I also want to learn Docker")
    assert "docker" in want, f"held={have} wanted={want}"
    assert "docker" not in have
    print("skill polarity: 8 cases OK")

    # ── Stage 4: prior-learning detection ────────────────────────────────────
    # Verbatim catalog IDs need no completion cue.
    assert extract_completed_resources("I've already done DS001 and DS003.") == [
        "DS001", "DS003"
    ]
    assert extract_completed_resources("i did ds005") == ["DS005"]

    # A title inside a clause carrying a completion cue.
    by_title = extract_completed_resources(
        "I finished SQL for Data Analysts last month and want to go further."
    )
    assert "DS005" in by_title, by_title

    # Without a completion cue, a title mention must NOT count as completed —
    # "I want to take SQL for Data Analysts" is a goal, not a history.
    assert extract_completed_resources(
        "I want to take SQL for Data Analysts."
    ) == [], "a wanted course must not be recorded as completed"

    # Nothing spurious from an ordinary goal.
    assert extract_completed_resources(
        "I want to become a data analyst. I know some Excel."
    ) == []
    assert extract_completed_resources("") == []
    assert extract_completed_resources("I finished ZZ999") == [], "unknown IDs are ignored"
    print("prior-learning detection: 7 cases OK")

    # ── Stage 3: interests are focused, not a fixed top-3 ────────────────────
    focused = classify_interests("I want to become a data analyst working with SQL and dashboards")
    assert focused[0] == "Data Science", focused
    assert "Web Development" not in focused, f"REGRESSION: noise category retained -> {focused}"
    assert 1 <= len(focused) <= INTEREST_MAX_K, focused
    print(f"interest classification: focused goal -> {focused}")

    # ── End-to-end profiles, persisted and read back ─────────────────────────
    cases = [
        ("__selftest_A__", "I want to become a data analyst. I know some Excel and basic SQL."),
        ("__selftest_B__", "I'm a complete beginner and want to learn web development with React."),
        ("__selftest_C__", "I have 5 years of Python experience and want to get into MLOps and cloud."),
    ]
    print()
    for sid, text in cases:
        profile = build_profile(sid, text)
        print(f"-- {sid} " + "-" * 40)
        print(f"   goal:             {profile['goal']}")
        print(f"   current_skills:   {profile['current_skills']}")
        print(f"   target_skills:    {profile['target_skills']}")
        print(f"   experience_level: {profile['experience_level']}")
        print(f"   interests:        {profile['interests']}")

        reloaded = load_profile(sid)
        assert reloaded is not None, f"{sid} did not persist"
        assert reloaded["current_skills"] == profile["current_skills"]
        assert reloaded["target_skills"] == profile["target_skills"]
        assert reloaded["experience_level"] == profile["experience_level"]

        # A restored profile must be structurally interchangeable with a freshly
        # built one, or code paths that work on a new session break on a resumed
        # one. Same keys, same values.
        assert set(reloaded) == set(profile), (
            f"restored keys {sorted(set(reloaded) ^ set(profile))} differ from built"
        )
        assert reloaded == profile, {
            k: (profile[k], reloaded[k]) for k in profile if profile[k] != reloaded[k]
        }

    # The headline regressions, asserted on the real end-to-end output.
    profile_b = load_profile("__selftest_B__")
    assert profile_b["current_skills"] == [], (
        f"a self-declared beginner must hold no skills -> {profile_b['current_skills']}"
    )
    profile_c = load_profile("__selftest_C__")
    assert profile_c["experience_level"] == "advanced", (
        f"REGRESSION: 5-year dev classified as {profile_c['experience_level']}"
    )

    # A session with no profile at all restores as None rather than an empty shell.
    assert load_profile("__no_such_session__") is None

    # Preferences saved before any goal must not masquerade as a restorable profile.
    from app.db import set_preferences as _set_prefs

    _set_prefs("__prefs_only__", {"hours_per_week": 5})
    assert load_profile("__prefs_only__") is None, (
        "a preferences-only row has no goal and must not restore as a profile"
    )

    # Clean up the throwaway sessions.
    conn = _get_connection()
    with conn:
        for table in ("progress", "feedback", "profiles", "users"):
            conn.execute(
                f"DELETE FROM {table} WHERE session_id = ?", ("__prefs_only__",)
            )
    conn.close()

    conn = _get_connection()
    with conn:
        for sid, _ in cases:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (sid,))
    conn.close()

    print("\nprofiling_engine.py self-test passed: all assertions OK")
