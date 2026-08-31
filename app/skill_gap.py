"""
skill_gap.py — Identify the skills a goal requires and which of them are missing.

The brief names four mechanisms the solution works by: understanding the
learner's profile, analyzing learning objectives, **identifying skill gaps**, and
generating a structured roadmap. This module is the third one.

Why this matters beyond ticking a requirement
---------------------------------------------
Without a gap, the system can only answer "which resources resemble your goal?"
It cannot answer "what do you still need?" or "are we there yet?", because it has
no representation of the destination. Ranking by similarity alone also has no way
to reject a resource that is topically adjacent but teaches nothing the learner
needs, which is why a data-analyst path picked up cloud-computing courses at 0.43
similarity — they matched the *domain* loosely while covering none of the skills.

How the target skill set is derived
-----------------------------------
There is no ground-truth "skills required to be a data analyst" table, so the
target set is inferred from the catalog itself:

1.  Skills the learner explicitly asked for (``profile["target_skills"]``) are
    taken at face value and given maximum relevance. If someone says "I want to
    learn React", React is a target regardless of what the embeddings think.

2.  The goal is embedded and matched against **courses only**, since only a
    course teaches a skill. Every skill taught by a strongly matching course
    becomes a target candidate, with relevance equal to the similarity of the
    best course teaching it. Tracing relevance to a single course keeps the
    number explainable: "SQL is a target because *SQL for Data Analysts* matches
    your goal at 0.79".

3.  Candidates are kept only if they clear ``RELEVANCE_RATIO`` of the strongest
    candidate's relevance. An absolute floor alone does not work here, because
    similarity magnitudes differ a lot between a narrow goal and a vague one.

The gap is then the target set minus what the learner already holds, compared
with token-subset matching so that holding "sql" satisfies a target of
"sql joins" while remaining distinct from "nosql".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import load_teachable_skills, parse_skills

#: How many courses to inspect when inferring the target skill set. Large enough
#: to cover a goal's skill surface, small enough that weakly related courses do
#: not contribute skills.
TARGET_COURSE_POOL = 18

#: Keep a candidate skill only if its relevance is at least this fraction of the
#: anchor relevance. Relative rather than absolute, because a narrow goal scores
#: much higher across the board than a vague one.
RELEVANCE_RATIO = 0.72

#: How many top *courses* to average when setting the relevance anchor.
#:
#: The anchor was originally the single highest skill relevance, which broke
#: whenever one resource matched near-verbatim. Mentioning a course by name — "I
#: already finished SQL for Data Analysts" — made that course score ~0.99, lifting
#: the cutoff so high that only *its* skills survived: a data-analyst goal
#: collapsed from 19 target skills to 5.
#:
#: Averaging must run over courses rather than skills. A single course contributes
#: all of its skills at the same relevance, so a five-skill course fills the whole
#: top-five of the skill list and the outlier survives the average untouched.
#: Averaging distinct course similarities dilutes it as intended.
RELEVANCE_ANCHOR_K = 5

#: Hard ceiling on the target set.
#:
#: The relative cutoff alone is not enough. When a goal's similarity scores are
#: flat — no course stands out — the cutoff sits low and lets almost everything
#: through: an MLOps goal produced 80 target skills, which is a curriculum, not a
#: gap. Capping keeps the target set legible and comparable across goals.
MAX_TARGET_SKILLS = 24

#: Ignore courses below this raw cosine when inferring targets. A course this
#: weakly related contributes noise skills rather than goal skills.
MIN_COURSE_SEMANTIC = 0.25

#: Below this best-match similarity the catalog does not credibly serve the goal,
#: and the honest answer is to say so rather than return confident percentages for
#: whatever ranked highest among poor options.
#:
#: Calibrated against the labelled sets in ``app/evaluation.py``, measured on the
#: learner's raw wording:
#:
#:     33 served goals     0.43 - 0.87  (mean 0.62)
#:     7 unserved goals    0.12 - 0.33  (cello, sourdough, marathon, gibberish)
#:
#: The threshold sits at the midpoint of that empty band, giving ~0.05 of margin on
#: each side. An earlier value of 0.42 was chosen from a smaller sample and left
#: only 0.01 of margin above the worst real goal — one unlucky phrasing away from
#: rejecting something the catalog covers perfectly well. The wider evaluation set
#: is what surfaced that; ``evaluation.py`` now asserts a minimum margin so the
#: same drift cannot recur silently.
MIN_GOAL_CONFIDENCE = 0.38

#: Relevance assigned to skills the learner named explicitly. Above 1.0 so an
#: explicit request always outranks an inferred skill.
EXPLICIT_RELEVANCE = 1.1

#: How many adjacent suggestions to offer when the gap is already closed.
GAP_CLOSED_FALLBACK_N = 3

#: Semantic floor for those adjacent suggestions, so the fallback cannot become a
#: back door for the off-topic padding coverage selection exists to prevent.
FALLBACK_MIN_SEMANTIC = 0.40


def normalise_skill(skill: str) -> str:
    """
    Normalise a skill label so trivially different spellings compare equal.

    Handles the two variations the catalog actually contains: inconsistent case
    and mixed British/American spelling ("data visualisation" in one row,
    "data visualization" in another).
    """
    text = " ".join(str(skill).strip().lower().split())
    text = text.replace("isation", "ization").replace("ising", "izing")
    text = text.replace("ise ", "ize ")
    if text.endswith("ise"):
        text = text[:-3] + "ize"
    return text


def skills_equivalent(a: str, b: str) -> bool:
    """
    Return ``True`` when two skill labels denote the same skill.

    Deliberately conservative: normalised exact match, or the same set of tokens
    in a different order.

    Why not treat a broader skill as covering a narrower one
    -------------------------------------------------------
    An earlier version used token-subset matching, so "sql" counted as covering
    "sql joins". That is defensible for that pair, but the rule cannot tell it
    apart from "sql" covering "sql injection" — identical token structure,
    completely different topic. Bag-of-words has no way to distinguish a
    sub-skill from an unrelated qualifier.

    Given the choice between occasionally over-teaching and occasionally telling a
    learner they already know something they do not, over-teaching is the safer
    error. So equivalence stays strict and the gap errs slightly wide.

    >>> skills_equivalent("Data Visualisation", "data visualization")
    True
    >>> skills_equivalent("sql joins", "joins sql")
    True
    >>> skills_equivalent("sql", "sql injection")
    False
    >>> skills_equivalent("sql", "nosql")
    False
    """
    na, nb = normalise_skill(a), normalise_skill(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return set(na.split()) == set(nb.split())


def is_held(skill: str, held: list[str] | set[str]) -> bool:
    """Return ``True`` if ``skill`` is covered by any skill in ``held``."""
    return any(skills_equivalent(skill, h) for h in held)


@dataclass
class SkillGap:
    """
    The outcome of gap analysis for one learner and one goal.

    Attributes
    ----------
    target_skills : dict[str, float]
        Skill -> relevance. What the goal appears to require.
    held_skills : list[str]
        What the learner already has, from the profile and completed resources.
    gap_skills : dict[str, float]
        Skill -> relevance. Target skills the learner does not yet hold. This is
        what a path needs to cover.
    satisfied_skills : list[str]
        Target skills the learner already holds. Kept so the UI can show credit
        for prior knowledge rather than silently dropping it.
    confidence : float
        Best raw cosine seen among candidate courses. Low means the catalog does
        not serve this goal.
    evidence : dict[str, tuple[str, float]]
        Skill -> (course_id, similarity) explaining why the skill is a target.
    """

    target_skills: dict[str, float] = field(default_factory=dict)
    held_skills: list[str] = field(default_factory=list)
    gap_skills: dict[str, float] = field(default_factory=dict)
    satisfied_skills: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: dict[str, tuple[str, float]] = field(default_factory=dict)

    @property
    def target_count(self) -> int:
        return len(self.target_skills)

    @property
    def gap_count(self) -> int:
        return len(self.gap_skills)

    @property
    def coverage_ratio(self) -> float:
        """Fraction of the target set the learner already holds, 0-1."""
        if not self.target_skills:
            return 0.0
        return len(self.satisfied_skills) / len(self.target_skills)

    @property
    def is_servable(self) -> bool:
        """
        Whether the catalog can credibly address this goal.

        ``False`` means the UI should say so plainly instead of presenting
        confident match percentages for resources that merely rank highest among
        poor options.
        """
        return self.confidence >= MIN_GOAL_CONFIDENCE and bool(self.target_skills)

    def ranked_gap(self) -> list[tuple[str, float]]:
        """Gap skills ordered by relevance, most important first."""
        return sorted(self.gap_skills.items(), key=lambda kv: (-kv[1], kv[0]))


def derive_target_skills(
    query: str,
    engine,
    pool_size: int = TARGET_COURSE_POOL,
    explicit_skills: list[str] | None = None,
) -> tuple[dict[str, float], float, dict[str, tuple[str, float]]]:
    """
    Infer the skills a goal requires.

    Parameters
    ----------
    query : str
        The retrieval query composed from the learner's goal.
    engine : RecommendationEngine
        Used for semantic matching. Passed in rather than imported so tests can
        substitute a stub and so this module holds no model state.
    pool_size : int
        How many courses to inspect.
    explicit_skills : list[str] | None
        Skills the learner named directly; taken at face value.

    Returns
    -------
    tuple[dict[str, float], float, dict[str, tuple[str, float]]]
        (skill -> relevance, confidence, skill -> (course_id, similarity))
    """
    teachable = load_teachable_skills()

    # Courses only: a project exercises skills and an assessment checks them,
    # but neither teaches, so neither can define what a goal requires.
    courses = engine.recommend(
        query,
        top_n=pool_size,
        resource_types=("course",),
        min_semantic=MIN_COURSE_SEMANTIC,
    )

    confidence = max((c["semantic_score"] for c in courses), default=0.0)

    # Relevance of a skill is the similarity of the best course teaching it.
    candidates: dict[str, float] = {}
    evidence: dict[str, tuple[str, float]] = {}
    for course in courses:
        sim = float(course["semantic_score"])
        for skill in parse_skills(course.get("skills", "")):
            if sim > candidates.get(skill, 0.0):
                candidates[skill] = sim
                evidence[skill] = (course["course_id"], sim)

    targets: dict[str, float] = {}
    if candidates:
        # Anchor on the mean similarity of the top few *courses*, not the top few
        # skills, so one near-verbatim title match cannot raise the bar above the
        # goal's real skill surface. See RELEVANCE_ANCHOR_K.
        course_sims = sorted(
            (float(c["semantic_score"]) for c in courses), reverse=True
        )[:RELEVANCE_ANCHOR_K]
        anchor = (
            sum(course_sims) / len(course_sims) if course_sims else max(candidates.values())
        )
        cutoff = anchor * RELEVANCE_RATIO

        kept = [(s, r) for s, r in candidates.items() if r >= cutoff]
        # Sort by relevance descending, then alphabetically so the cap is
        # deterministic rather than dependent on dict insertion order.
        kept.sort(key=lambda kv: (-kv[1], kv[0]))
        targets = dict(kept[:MAX_TARGET_SKILLS])

    # Explicit requests override inference and cannot be filtered out, provided
    # something in the catalog actually teaches them.
    #
    # ``teachable`` is a frozenset, so it has no order. An exact match must be
    # preferred explicitly rather than taking whatever equivalent turns up first,
    # otherwise a request for "sql" could attach itself to an arbitrary
    # equivalent label instead of "sql" proper.
    for raw in explicit_skills or []:
        skill = normalise_skill(raw)
        if not skill:
            continue
        exact = next((t for t in teachable if normalise_skill(t) == skill), None)
        match = exact or next(
            (t for t in sorted(teachable) if skills_equivalent(skill, t)), None
        )
        if match:
            targets[match] = EXPLICIT_RELEVANCE
            evidence[match] = ("stated by learner", 1.0)

    return targets, confidence, evidence


def goal_confidence(goal_text: str, engine) -> float:
    """
    Return the best course match for the learner's own wording.

    Measured on the raw goal rather than the enriched retrieval query, and this
    distinction is load-bearing. The retrieval query appends the classified
    domain and target skills to improve recall, but the domain classifier always
    returns *something* — including for a goal the catalog cannot serve. Appending
    "UX/UI Design" to a request about medieval manuscripts lifted its similarity
    from 0.22 to 0.31 and made an unservable goal look servable.

    Enriching helps retrieval. It corrupts the honesty check. So servability is
    judged on what the learner actually said.
    """
    text = (goal_text or "").strip()
    if not text:
        return 0.0
    hits = engine.recommend(text, top_n=1, resource_types=("course",))
    return float(hits[0]["semantic_score"]) if hits else 0.0


def compute_gap(
    profile: dict,
    engine,
    query: str | None = None,
    acquired_skills: list[str] | None = None,
) -> SkillGap:
    """
    Run gap analysis for a learner profile.

    Parameters
    ----------
    profile : dict
        Profile from ``profiling_engine.build_profile()``.
    engine : RecommendationEngine
        Used for semantic matching.
    query : str | None
        Retrieval query used to infer target skills. Defaults to composing one
        from the profile, but the pipeline passes its own so gap analysis and
        ranking see identical text.
    acquired_skills : list[str] | None
        Skills gained from resources completed inside the app. Counted as held
        alongside the profile's ``current_skills``, which is what makes progress
        shrink the gap.

    Returns
    -------
    SkillGap
    """
    profile = profile or {}
    goal_text = profile.get("goal", "")
    if query is None:
        parts = [goal_text]
        parts.extend(profile.get("interests") or [])
        parts.extend(profile.get("target_skills") or [])
        query = " ".join(p for p in parts if p).strip()

    held = [s.strip().lower() for s in (profile.get("current_skills") or []) if s.strip()]
    for skill in acquired_skills or []:
        cleaned = skill.strip().lower()
        if cleaned and not is_held(cleaned, held):
            held.append(cleaned)

    targets, _pool_confidence, evidence = derive_target_skills(
        query, engine, explicit_skills=profile.get("target_skills")
    )

    # Judged on the learner's own words, not the enriched query. See goal_confidence.
    confidence = goal_confidence(goal_text or query, engine)

    gap: dict[str, float] = {}
    satisfied: list[str] = []
    for skill, relevance in targets.items():
        if is_held(skill, held):
            satisfied.append(skill)
        else:
            gap[skill] = relevance

    return SkillGap(
        target_skills=targets,
        held_skills=sorted(held),
        gap_skills=gap,
        satisfied_skills=sorted(satisfied),
        confidence=confidence,
        evidence=evidence,
    )


def select_by_coverage(
    candidates: list[dict],
    gap: SkillGap,
    max_items: int = 12,
    coverage_target: float = 0.92,
) -> list[dict]:
    """
    Choose the smallest useful set of courses that covers the learner's gap.

    This replaces "take the top N by similarity" as the selection step. Ranking by
    similarity fills its quota regardless of whether a resource teaches anything
    the learner needs, which is how cloud-computing courses at 0.43 similarity
    ended up in a data-analyst path. Here a course that covers no remaining gap
    skill is simply never selected, so irrelevant results are excluded
    structurally rather than by tuning a threshold.

    Algorithm
    ---------
    Greedy weighted set cover. At each step, pick the course maximising

        (relevance-weighted newly-covered skills / duration) x relevance factor

    then remove the skills it covers and repeat. Greedy is the standard
    approximation for set cover, which is NP-hard; it is within a
    ``ln(n)+1`` factor of optimal, which is more than good enough here and is
    fast and explainable, both of which matter more than optimality for a
    learning path a human has to read.

    Dividing covered weight by duration makes the objective "close the most gap
    per hour of study", which is what the brief's framing — the *right sequence*
    to reach a goal — actually asks for. The relevance factor then breaks ties
    toward courses that better match the stated goal and difficulty level.

    Courses only
    ------------
    Projects and assessments are filtered out. Their ``skills`` column lists what
    they exercise or validate, not what they teach, so letting them close a gap
    would credit the learner with skills nothing taught them. Assessments are
    also short, which under a per-hour objective would make them look like the
    most efficient way to learn anything. They are woven back in by the path
    generator once the teaching sequence is settled.

    Parameters
    ----------
    candidates : list[dict]
        Scored resources from the recommendation engine.
    gap : SkillGap
        The learner's gap.
    max_items : int
        Ceiling on selected courses, so a wide gap cannot produce an unreadable path.
    coverage_target : float
        Stop once this fraction of total gap weight is covered. Chasing the last
        few percent tends to add long courses for one marginal skill each.

    Returns
    -------
    list[dict]
        Selected course dicts, in selection order, each annotated with
        ``covers`` (gap skills it closes), ``coverage_share`` (fraction of total
        gap weight) and ``cumulative_coverage``.
    """
    courses = [c for c in candidates if c.get("resource_type", "course") == "course"]

    # With no gap there is nothing to cover. This is a real state — the learner
    # already holds every target skill — so offer a few adjacent options rather
    # than a full path.
    #
    # The earlier version returned ``courses[:max_items]`` here, i.e. raw
    # similarity order with no filter at all, which quietly reintroduced exactly
    # the off-topic padding that coverage selection exists to prevent. A small
    # count and a semantic floor keep the fallback honest.
    if not gap.gap_skills or not courses:
        adjacent = [
            c for c in courses
            if float(c.get("semantic_score") or 0.0) >= FALLBACK_MIN_SEMANTIC
        ]
        return adjacent[:GAP_CLOSED_FALLBACK_N]

    remaining = dict(gap.gap_skills)
    total_weight = sum(gap.gap_skills.values())
    pool = {c["course_id"]: c for c in courses}

    selected: list[dict] = []
    covered_weight = 0.0

    while pool and len(selected) < max_items:
        best_id, best_score, best_covered = None, 0.0, []

        for cid, course in pool.items():
            course_skills = parse_skills(course.get("skills", ""))
            newly = [
                skill
                for skill in remaining
                if any(skills_equivalent(skill, cs) for cs in course_skills)
            ]
            if not newly:
                continue

            new_weight = sum(remaining[s] for s in newly)
            hours = max(int(course.get("duration_hours") or 1), 1)
            relevance = float(course.get("similarity_score") or 0.0)

            # Coverage per hour, nudged by how well the course matches the goal.
            score = (new_weight / hours) * (0.5 + 0.5 * relevance)

            if score > best_score:
                best_id, best_score, best_covered = cid, score, newly

        # Nothing left covers anything new: stop rather than pad the path.
        if best_id is None:
            break

        course = dict(pool.pop(best_id))
        new_weight = sum(remaining[s] for s in best_covered)
        covered_weight += new_weight

        course["covers"] = sorted(best_covered)
        course["coverage_share"] = new_weight / total_weight if total_weight else 0.0
        course["cumulative_coverage"] = covered_weight / total_weight if total_weight else 0.0
        selected.append(course)

        for skill in best_covered:
            remaining.pop(skill, None)

        if not remaining or course["cumulative_coverage"] >= coverage_target:
            break

    return selected


def coverage_summary(selected: list[dict], gap: SkillGap) -> dict:
    """
    Summarise what a selection achieves, for the UI and the explainer.

    Returns
    -------
    dict
        ``covered`` / ``uncovered`` skill lists, ``covered_ratio`` by weight,
        ``total_hours`` and ``resource_count``.
    """
    covered: set[str] = set()
    for resource in selected:
        covered.update(resource.get("covers") or [])

    total_weight = sum(gap.gap_skills.values())
    covered_weight = sum(gap.gap_skills[s] for s in covered if s in gap.gap_skills)

    return {
        "covered": sorted(covered),
        "uncovered": sorted(s for s in gap.gap_skills if s not in covered),
        "covered_ratio": (covered_weight / total_weight) if total_weight else 0.0,
        "total_hours": sum(int(r.get("duration_hours") or 0) for r in selected),
        "resource_count": len(selected),
    }


def resource_coverage(resource: dict, gap: SkillGap) -> tuple[list[str], float]:
    """
    Return which gap skills a resource covers, and its share of total gap weight.

    Weighted by relevance rather than counted, so covering two high-relevance
    skills beats covering three marginal ones. That weighting is what stops the
    selector from padding a path with resources that technically touch the gap.

    Returns
    -------
    tuple[list[str], float]
        (covered gap skills, fraction of total gap relevance covered)
    """
    if not gap.gap_skills:
        return [], 0.0

    resource_skills = parse_skills(resource.get("skills", ""))
    covered = [
        gap_skill
        for gap_skill in gap.gap_skills
        if any(skills_equivalent(gap_skill, rs) for rs in resource_skills)
    ]
    total_weight = sum(gap.gap_skills.values())
    if total_weight <= 0:
        return covered, 0.0
    covered_weight = sum(gap.gap_skills[s] for s in covered)
    return sorted(covered), covered_weight / total_weight


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── skill equivalence, before loading any model ──────────────────────────
    assert skills_equivalent("sql", "SQL")
    assert skills_equivalent("Data Visualisation", "data visualization")
    assert skills_equivalent("sql joins", "joins sql"), "token order must not matter"
    assert skills_equivalent("  Machine   Learning ", "machine learning")

    # REGRESSION: token-subset matching equated these, which let a web-security
    # skill absorb an explicit request for SQL.
    assert not skills_equivalent("sql", "sql injection"), (
        "a qualifier can change the topic entirely; these are not the same skill"
    )
    assert not skills_equivalent("sql", "nosql"), "substring matching would equate these"
    assert not skills_equivalent("react", "redux")
    assert not skills_equivalent("", "sql")

    assert is_held("sql", ["SQL", "excel"])
    assert not is_held("sql injection", ["sql", "excel"])
    assert not is_held("kubernetes", ["sql", "excel"])
    print("skill equivalence: 11 cases OK")

    from app.recommendation_engine import get_engine

    engine = get_engine()

    # ── a focused beginner goal ──────────────────────────────────────────────
    analyst = {
        "goal": "I want to become a data analyst. I know some Excel.",
        "current_skills": ["excel"],
        "target_skills": ["sql"],
        "experience_level": "beginner",
        "interests": ["Data Science"],
    }
    gap = compute_gap(analyst, engine)
    print()
    print(f"-- data analyst goal " + "-" * 40)
    print(f"   confidence      : {gap.confidence:.3f}  (servable={gap.is_servable})")
    print(f"   target skills   : {gap.target_count}")
    print(f"   already held    : {gap.satisfied_skills}")
    print(f"   coverage now    : {gap.coverage_ratio:.0%}")
    print(f"   top gap skills  : {[s for s, _ in gap.ranked_gap()[:8]]}")

    assert gap.is_servable, "a data-analyst goal must be servable by this catalog"
    assert gap.target_count > 0
    assert "excel" in gap.satisfied_skills, (
        f"a stated Excel skill should be credited -> {gap.satisfied_skills}"
    )
    assert "excel" not in gap.gap_skills, "a held skill must not appear in the gap"
    assert any("sql" in s for s in gap.gap_skills), (
        f"an explicitly requested skill must land in the gap -> {list(gap.gap_skills)}"
    )
    assert gap.gap_count + len(gap.satisfied_skills) == gap.target_count

    # An explicit request must land on the skill itself and outrank inferred ones.
    assert "sql" in gap.gap_skills, (
        f"explicit 'sql' must appear as itself, not an equivalent -> {list(gap.gap_skills)}"
    )
    assert gap.gap_skills["sql"] >= EXPLICIT_RELEVANCE - 1e-9, gap.gap_skills["sql"]
    assert gap.evidence["sql"][0] == "stated by learner", gap.evidence["sql"]
    assert "sql injection" not in gap.target_skills, (
        "a web-security skill must not be inferred from a data-analyst goal"
    )

    # ── coverage scoring picks the right resource ────────────────────────────
    from app.config import load_catalog_map

    catalog = load_catalog_map()
    sql_course = catalog["DS005"]           # SQL for Data Analysts
    covered, share = resource_coverage(sql_course, gap)
    print(f"   DS005 covers    : {covered} ({share:.0%} of gap weight)")
    assert covered, "the SQL course must cover at least one gap skill"

    kubernetes = catalog["CD006"]           # Kubernetes: Container Orchestration
    k8s_covered, k8s_share = resource_coverage(kubernetes, gap)
    print(f"   CD006 covers    : {k8s_covered} ({k8s_share:.0%} of gap weight)")
    assert k8s_share < share, (
        "an off-topic course must cover less gap weight than an on-topic one"
    )

    # ── progress shrinks the gap ─────────────────────────────────────────────
    before = gap.gap_count
    after_gap = compute_gap(analyst, engine, acquired_skills=[s for s, _ in gap.ranked_gap()[:3]])
    print(f"   gap after completing 3 skills: {before} -> {after_gap.gap_count}")
    assert after_gap.gap_count < before, "acquiring skills must shrink the gap"
    assert after_gap.coverage_ratio > gap.coverage_ratio

    # ── an advanced goal in another domain ───────────────────────────────────
    mlops = {
        "goal": "I have 5 years of Python experience and want to move into MLOps.",
        "current_skills": ["python"],
        "target_skills": ["mlops"],
        "experience_level": "advanced",
        "interests": ["Cloud/DevOps"],
    }
    mlops_gap = compute_gap(mlops, engine)
    print()
    print(f"-- MLOps goal " + "-" * 47)
    print(f"   confidence      : {mlops_gap.confidence:.3f}")
    print(f"   target skills   : {mlops_gap.target_count}")
    print(f"   top gap skills  : {[s for s, _ in mlops_gap.ranked_gap()[:8]]}")
    assert mlops_gap.is_servable
    # The two goals must produce genuinely different targets.
    assert set(mlops_gap.target_skills) != set(gap.target_skills)

    # The core invariant: a skill the learner holds must never be reported as
    # missing. It may legitimately be absent from the target set entirely — an
    # MLOps goal does not infer "python" as a target, because the courses that
    # teach Python are not what matches an MLOps goal — but it must never appear
    # as a gap.
    for label, g in (("analyst", gap), ("mlops", mlops_gap)):
        for held_skill in g.held_skills:
            assert not any(skills_equivalent(held_skill, missing) for missing in g.gap_skills), (
                f"{label}: held skill {held_skill!r} was reported as a gap"
            )
        assert set(g.satisfied_skills).isdisjoint(g.gap_skills), label
        assert g.gap_count + len(g.satisfied_skills) == g.target_count, label

    # REGRESSION: a flat score distribution once let the relative cutoff pass 80
    # skills through, which is a curriculum rather than a gap.
    for label, g in (("analyst", gap), ("mlops", mlops_gap)):
        assert g.target_count <= MAX_TARGET_SKILLS, (
            f"{label} goal produced {g.target_count} targets, cap is {MAX_TARGET_SKILLS}"
        )
    # The explicitly requested skill must survive the cap, since it outranks
    # every inferred skill.
    assert "mlops" in mlops_gap.gap_skills, list(mlops_gap.gap_skills)

    # ── an off-catalog goal must NOT be reported as servable ─────────────────
    offbeat = {
        "goal": "I want to learn medieval Latin palaeography and manuscript restoration.",
        "current_skills": [],
        "target_skills": [],
        "experience_level": "beginner",
        "interests": [],
    }
    off_gap = compute_gap(offbeat, engine)
    print()
    print(f"-- off-catalog goal " + "-" * 41)
    print(f"   confidence      : {off_gap.confidence:.3f}  (servable={off_gap.is_servable})")
    assert off_gap.confidence < gap.confidence, (
        "an off-catalog goal must score lower confidence than a served one"
    )
    assert not off_gap.is_servable, (
        f"off-catalog goal reported servable at confidence {off_gap.confidence:.3f}"
    )

    # ── the servability threshold must separate served from unserved goals ───
    served_goals = [
        "I want to become a data analyst. I know some Excel.",
        "I am a complete beginner and want to learn web development.",
        "I have 5 years of Python experience and want to move into MLOps.",
        "I want to switch careers into UX design.",
        "I want to get AWS certified and learn DevOps.",
        "I want to learn digital marketing and SEO.",
    ]
    unserved_goals = [
        "I want to learn medieval Latin palaeography and manuscript restoration.",
        "I want to become a professional cellist.",
        "teach me to bake sourdough bread",
        "how do I train for a marathon",
        "asdfgh qwerty zxcvb",
    ]
    for text in served_goals:
        score = goal_confidence(text, engine)
        assert score >= MIN_GOAL_CONFIDENCE, (
            f"served goal fell below the threshold at {score:.3f}: {text!r}"
        )
    for text in unserved_goals:
        score = goal_confidence(text, engine)
        assert score < MIN_GOAL_CONFIDENCE, (
            f"unserved goal cleared the threshold at {score:.3f}: {text!r}"
        )
    print(f"   servability gate: {len(served_goals)} served / "
          f"{len(unserved_goals)} unserved classified correctly")

    # ── a learner who already holds everything has no gap ────────────────────
    expert = dict(analyst, current_skills=list(gap.target_skills), target_skills=[])
    expert_gap = compute_gap(expert, engine)
    assert expert_gap.gap_count < gap.gap_count, (
        "holding the target skills must shrink the gap"
    )
    print()
    print(f"   learner holding all targets: gap {gap.gap_count} -> {expert_gap.gap_count}, "
          f"coverage {expert_gap.coverage_ratio:.0%}")

    # ── greedy set cover selection ───────────────────────────────────────────
    analyst_query = (
        "I want to become a data analyst. I know some Excel. Data Science sql"
    )
    candidates = engine.recommend(analyst_query, top_n=30, experience_level="beginner")
    selected = select_by_coverage(candidates, gap)
    summary = coverage_summary(selected, gap)

    print()
    print("-- greedy coverage selection (data analyst) " + "-" * 17)
    for i, resource in enumerate(selected, 1):
        print(
            f"  {i}. {resource['course_id']} {resource['title'][:40]:<40} "
            f"{resource['duration_hours']:>3}h  +{resource['coverage_share']:.0%} "
            f"-> {resource['cumulative_coverage']:.0%}  covers={resource['covers']}"
        )
    print(f"  => {summary['resource_count']} courses, {summary['total_hours']}h, "
          f"{summary['covered_ratio']:.0%} of gap weight covered")
    print(f"  => still uncovered: {summary['uncovered']}")

    assert selected, "selection must not be empty for a servable goal"

    # Only courses may close a gap: a project exercises skills and an assessment
    # checks them, so neither teaches anything.
    assert all(r["resource_type"] == "course" for r in selected), (
        "projects/assessments must not be selected as gap-closers"
    )

    # Every selected course must earn its place by covering something new.
    for resource in selected:
        assert resource["covers"], f"{resource['course_id']} covers nothing"
        assert resource["coverage_share"] > 0

    # Cumulative coverage must increase monotonically.
    cumulative = [r["cumulative_coverage"] for r in selected]
    assert cumulative == sorted(cumulative), cumulative
    assert cumulative[-1] <= 1.0 + 1e-9

    # No course may be selected twice, and no skill credited twice.
    ids = [r["course_id"] for r in selected]
    assert len(ids) == len(set(ids)), ids
    all_covered = [s for r in selected for s in r["covers"]]
    assert len(all_covered) == len(set(all_covered)), "a skill was covered twice"

    # REGRESSION: off-topic resources must be excluded structurally. Kubernetes
    # covers no data-analyst gap skill, so it can never be selected however high
    # it ranks on similarity.
    assert "CD006" not in ids, "an off-topic course was selected"
    off_topic = {"CD001", "CD004", "CD006", "CD012", "WD012"}
    assert off_topic.isdisjoint(ids), f"off-topic courses selected: {off_topic & set(ids)}"

    # The SQL course covers the most valuable part of this gap, so it must appear.
    assert "DS005" in ids, f"the SQL course should be selected -> {ids}"

    # Selection must be materially smaller than the candidate pool it came from.
    assert len(selected) < len(candidates), (len(selected), len(candidates))

    # ── an already-closed gap offers a few adjacent options, not a full path ──
    # REGRESSION: this branch used to return raw similarity order with no filter,
    # which reintroduced the off-topic padding coverage selection prevents.
    no_gap = SkillGap(target_skills={"excel": 1.0}, held_skills=["excel"],
                      gap_skills={}, satisfied_skills=["excel"], confidence=0.7)
    fallback = select_by_coverage(candidates, no_gap, max_items=10)
    assert len(fallback) <= GAP_CLOSED_FALLBACK_N, (
        f"a closed gap should offer at most {GAP_CLOSED_FALLBACK_N} adjacent options, "
        f"got {len(fallback)}"
    )
    for resource in fallback:
        assert resource["semantic_score"] >= FALLBACK_MIN_SEMANTIC, resource["semantic_score"]
    assert coverage_summary(fallback, no_gap)["covered_ratio"] == 0.0

    # ── quoting a course title must not collapse the target set ──────────────
    # REGRESSION: anchoring the cutoff on the single best match meant naming a
    # course made it score ~0.99 and filtered every other skill out, taking a
    # data-analyst goal from 19 targets down to 5.
    plain_targets, _, _ = derive_target_skills(
        "I want to become a data analyst. Data Science", engine
    )
    quoted_targets, _, _ = derive_target_skills(
        "I want to become a data analyst. I already finished SQL for Data Analysts. "
        "Data Science",
        engine,
    )
    print()
    print(f"   targets for plain goal          : {len(plain_targets)}")
    print(f"   targets when a title is quoted  : {len(quoted_targets)}")
    assert len(quoted_targets) >= 0.6 * len(plain_targets), (
        f"quoting a course title collapsed the target set from "
        f"{len(plain_targets)} to {len(quoted_targets)}"
    )

    # ── the same machinery works for a different domain ──────────────────────
    mlops_candidates = engine.recommend(
        "I have 5 years of Python experience and want to move into MLOps Cloud/DevOps mlops",
        top_n=30,
        experience_level="advanced",
    )
    mlops_selected = select_by_coverage(mlops_candidates, mlops_gap)
    mlops_summary = coverage_summary(mlops_selected, mlops_gap)
    print()
    print("-- greedy coverage selection (MLOps) " + "-" * 24)
    for i, resource in enumerate(mlops_selected, 1):
        print(
            f"  {i}. {resource['course_id']} {resource['title'][:40]:<40} "
            f"{resource['duration_hours']:>3}h  -> {resource['cumulative_coverage']:.0%}"
        )
    print(f"  => {mlops_summary['resource_count']} courses, "
          f"{mlops_summary['total_hours']}h, "
          f"{mlops_summary['covered_ratio']:.0%} of gap weight covered")
    assert mlops_selected
    assert all(r["covers"] for r in mlops_selected)
    assert set(r["course_id"] for r in mlops_selected) != set(ids), (
        "two different goals must select different courses"
    )

    print("\nskill_gap.py self-test passed: all assertions OK")
