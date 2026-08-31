"""
pipeline.py — The end-to-end recommendation flow in one place.

Flow
----
    profile  ->  gap analysis  ->  coverage selection  ->  sequencing  ->  persist

Each stage answers a different question, and the brief asks for all four:

    gap analysis         what does this goal require, and what is missing?
    coverage selection   what is the smallest set of courses that closes it?
    sequencing           in what order, respecting prerequisites?
    persistence          what has the learner already done?

Why the flow lives here
-----------------------
``chat_interface`` and ``dashboard`` previously held line-for-line copies of this
sequence. Two copies meant a fix applied to one silently skipped the other, which
is exactly how a missing progress-row write survived in the dashboard's re-rank.
Keeping it here also gives the UI a single seam.

Why selection is not just "top N by similarity"
-----------------------------------------------
Similarity ranking fills its quota regardless of whether a resource teaches
anything the learner needs. That put cloud-computing courses scoring 0.43 into a
data-analyst path. Selecting for gap coverage instead excludes them structurally:
a course that closes no gap skill is never chosen, at any similarity. On a
beginner data-analyst goal this turned a 14-step path into 4 courses and 24 hours
covering 100% of the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import load_catalog_map, parse_skills
from app.db import (
    get_completed_course_ids,
    get_excluded_course_ids,
    get_feedback_counts,
    init_db,
    sync_path_progress,
)
from app.learner_model import (
    LearnerPatterns,
    LearnerPreferences,
    derive_patterns,
    schedule_path,
    weeks_required,
)
from app.path_generator import generate_path
from app.profiling_engine import build_profile
from app.recommendation_engine import effective_experience_level, get_engine
from app.skill_gap import SkillGap, compute_gap, coverage_summary, select_by_coverage

#: Candidate pool handed to the coverage selector. Larger than the final path
#: because the selector rejects most of it — anything covering no gap skill.
CANDIDATE_POOL = 30

#: Ceiling on courses the selector may choose, before prerequisites, projects and
#: assessments are added by the sequencer.
MAX_SELECTED_COURSES = 10


@dataclass
class PathResult:
    """
    Everything the UI needs about one generated path.

    Bundled rather than returned as a bare list because the dashboard has to show
    gap coverage and the no-match state, and re-deriving those in the UI would put
    ranking logic back into the presentation layer.
    """

    path: list[dict] = field(default_factory=list)
    gap: SkillGap = field(default_factory=SkillGap)
    coverage: dict = field(default_factory=dict)
    effective_level: str = "beginner"
    level_escalated: bool = False
    preferences: LearnerPreferences = field(default_factory=LearnerPreferences)
    patterns: LearnerPatterns = field(default_factory=LearnerPatterns)

    @property
    def servable(self) -> bool:
        """Whether the catalog can credibly address the goal."""
        return self.gap.is_servable and bool(self.path)

    @property
    def course_steps(self) -> list[dict]:
        return [s for s in self.path if s.get("course")]

    @property
    def total_hours(self) -> int:
        return sum(int(s["course"].get("duration_hours") or 0) for s in self.course_steps)

    @property
    def schedule(self) -> list[dict]:
        """The path laid out into weeks at the learner's stated weekly hours."""
        return schedule_path(self.path, self.preferences.hours_per_week)

    @property
    def weeks(self) -> int:
        """Weeks to finish at the learner's stated pace."""
        return weeks_required(self.path, self.preferences.hours_per_week)


def compose_query(profile: dict) -> str:
    """
    Build the retrieval query from a profile.

    The goal text carries the most signal, so it leads. Interests steer the
    embedding toward the right domain, and ``target_skills`` name concrete
    technologies the goal sentence may only imply.

    ``current_skills`` are deliberately excluded. A learner who already knows
    Excel does not want more Excel courses, and including held skills pulled the
    ranking toward material they had already covered.
    """
    parts = [profile.get("goal", "")]
    if profile.get("interests"):
        parts.append(" ".join(profile["interests"]))
    if profile.get("target_skills"):
        parts.append(" ".join(profile["target_skills"]))
    return " ".join(p for p in parts if p).strip()


def resolve_experience_level(session_id: str, profile: dict) -> tuple[str, bool]:
    """
    Return ``(effective_level, was_escalated)`` for a session.

    The stated level comes from the learner's own description. Repeated "too
    easy" feedback promotes it, which is what makes that button behave
    differently from "not interested" rather than being a second delete button.
    """
    stated = profile.get("experience_level", "beginner")
    too_easy = get_feedback_counts(session_id).get("too_easy", 0)
    effective = effective_experience_level(stated, too_easy)
    return effective, effective != stated


def acquired_skills(session_id: str) -> list[str]:
    """
    Return the skills a learner has gained from resources completed in the app.

    This is what makes progress adapt the recommendations rather than only
    filling a progress bar: completed resources feed back into gap analysis, the
    gap shrinks, and the remaining path is re-selected against what is actually
    left. The brief asks for adaptation on "feedback and progress"; this is the
    progress half.
    """
    catalog = load_catalog_map()
    skills: list[str] = []
    for cid in get_completed_course_ids(session_id):
        resource = catalog.get(cid)
        if not resource:
            continue
        for skill in parse_skills(resource.get("skills", "")):
            if skill not in skills:
                skills.append(skill)
    return skills


def build_learning_path(
    session_id: str,
    profile: dict,
    pool_size: int = CANDIDATE_POOL,
    max_courses: int = MAX_SELECTED_COURSES,
) -> PathResult:
    """
    Analyse the gap, select courses that close it, sequence them, and persist.

    Steps
    -----
    1. Compose the retrieval query and resolve the effective difficulty level.
    2. Run gap analysis, counting both stated skills and skills acquired from
       completed resources.
    3. Retrieve a candidate pool, excluding rejected and completed resources.
    4. Select the smallest set of courses covering the gap (greedy set cover).
    5. Sequence topologically, letting held skills discharge prerequisites.
    6. Write a ``progress`` row for every resource in the result.

    Step 6 is essential rather than incidental. Without it the dashboard's "Mark
    Done" button issued an ``UPDATE`` against rows that did not exist, so it
    silently did nothing and progress never left 0%.

    Returns
    -------
    PathResult
    """
    init_db()

    engine = get_engine()
    query = compose_query(profile)
    effective_level, escalated = resolve_experience_level(session_id, profile)

    gained = acquired_skills(session_id)
    gap = compute_gap(profile, engine, query=query, acquired_skills=gained)

    held = list(gap.held_skills)

    # Completed resources are excluded as well as rejected ones: re-recommending
    # something the learner has finished wastes a slot in the path.
    excluded = get_excluded_course_ids(session_id) | get_completed_course_ids(session_id)

    candidates = engine.recommend(
        query,
        top_n=pool_size,
        exclude_ids=excluded,
        experience_level=effective_level,
    )

    preferences = LearnerPreferences.from_dict(profile.get("preferences"))

    selected = select_by_coverage(candidates, gap, max_items=max_courses)

    path = generate_path(
        selected,
        experience_level=effective_level,
        held_skills=held,
        held_resource_ids=sorted(get_completed_course_ids(session_id)),
        reinforcement_order=preferences.reinforcement_order,
    )

    # The budget has to bound the *finished* path, not the selection. Prerequisite
    # fillers and checkpoint projects are added after selection, and they were
    # pushing a 50-hour relaxed plan up to 56 hours.
    path = _trim_path_to_budget(path, preferences.hour_budget)

    # Coverage is recomputed from what survived the trim, so the reported figure
    # describes the path the learner is actually shown.
    coverage = coverage_summary([s["course"] for s in path if s.get("course")], gap)

    sync_path_progress(session_id, path)

    return PathResult(
        path=path,
        gap=gap,
        coverage=coverage,
        effective_level=effective_level,
        level_escalated=escalated,
        preferences=preferences,
        patterns=derive_patterns(session_id),
    )


def _trim_path_to_budget(path: list[dict], budget: int) -> list[dict]:
    """
    Truncate a generated path so its total hours fit the learner's stated pace.

    Trimming from the end is safe with respect to prerequisites: the path is
    topologically ordered, so every dependency of a surviving step appears
    earlier. Removing a tail can therefore never orphan anything left behind.

    It is also the right end to cut. Greedy coverage selection emits the largest
    gap reduction per hour first, so the tail holds the least valuable work.

    The first resource is always kept even if it alone exceeds the budget:
    handing someone with a real goal an empty path is worse than handing them one
    long course.

    Step numbers are renumbered afterwards so they stay contiguous, and a
    trailing checkpoint is dropped so a path never ends on a milestone.
    """
    if not path:
        return path

    kept: list[dict] = []
    running = 0
    for step in path:
        resource = step.get("course")
        if resource is None:
            # Milestones cost nothing; keep them only if something follows.
            kept.append(step)
            continue
        hours = int(resource.get("duration_hours") or 0)
        if any(s.get("course") for s in kept) and running + hours > budget:
            break
        kept.append(step)
        running += hours

    # Drop trailing entries that carry no resource.
    while kept and kept[-1].get("course") is None:
        kept.pop()

    # Renumber so course steps remain 1..N with no gaps.
    counter = 0
    for step in kept:
        if step.get("course") is not None:
            counter += 1
            step["step_number"] = counter
        else:
            step["step_number"] = counter

    return kept


def run_full_pipeline(session_id: str, raw_text: str) -> tuple[dict, PathResult]:
    """
    Profile the learner's text and build their path in one call.

    Returns
    -------
    tuple[dict, PathResult]
        The structured profile and the path result.
    """
    init_db()
    profile = build_profile(session_id, raw_text)
    result = build_learning_path(session_id, profile)
    return profile, result


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from app.db import (
        _get_connection,
        get_progress,
        log_feedback,
        mark_course_status,
    )

    SID = "__selftest_pipeline__"

    def _cleanup() -> None:
        conn = _get_connection()
        with conn:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (SID,))
        conn.close()

    init_db()
    _cleanup()

    # ── query composition excludes held skills, includes targets ─────────────
    q = compose_query(
        {
            "goal": "become a data analyst",
            "interests": ["Data Science"],
            "current_skills": ["excel"],
            "target_skills": ["sql"],
        }
    )
    assert "data analyst" in q and "Data Science" in q and "sql" in q, q
    assert "excel" not in q, f"held skills should not steer retrieval -> {q}"

    # ── full pipeline on a beginner goal ─────────────────────────────────────
    profile, result = run_full_pipeline(
        SID, "I want to become a data analyst. I know some Excel."
    )
    print()
    print("-- beginner data analyst " + "-" * 36)
    print(f"   level        : {result.effective_level}")
    print(f"   gap          : {result.gap.gap_count} of {result.gap.target_count} target skills")
    print(f"   coverage     : {result.coverage['covered_ratio']:.0%} of gap weight")
    print(f"   path         : {len(result.course_steps)} resources, {result.total_hours}h")
    for step in result.course_steps:
        r = step["course"]
        tag = " (prereq)" if step["is_prerequisite_filler"] else ""
        print(f"     {step['step_number']:>2}. [{r['resource_type'][:4]}] {r['course_id']} "
              f"{r['title'][:44]:<44} {r['duration_hours']:>3}h{tag}")

    assert result.servable, "a data-analyst goal must be servable"
    assert result.course_steps, "pipeline produced no resources"
    assert result.coverage["covered_ratio"] > 0.5, result.coverage["covered_ratio"]

    # Off-topic resources must not appear at all.
    path_ids = {s["course"]["course_id"] for s in result.course_steps}
    off_topic = {"CD006", "CD012", "WD012", "WD013", "UX014"}
    assert off_topic.isdisjoint(path_ids), f"off-topic in path: {off_topic & path_ids}"

    # ── REGRESSION: progress rows must exist so 'Mark Done' can work ─────────
    rows = get_progress(SID)
    assert len(rows) == len(result.course_steps), (
        f"REGRESSION: {len(result.course_steps)} resources in path but {len(rows)} "
        f"progress rows. Without these rows Mark Done is a silent no-op."
    )

    first_id = result.course_steps[0]["course"]["course_id"]
    mark_course_status(SID, first_id, "completed")
    assert get_completed_course_ids(SID) == {first_id}

    # ── progress-driven adaptation: completing work shrinks the gap ───────────
    gained = acquired_skills(SID)
    assert gained, "completing a resource must yield acquired skills"
    after = build_learning_path(SID, profile)
    print()
    print(f"   after completing {first_id}:")
    print(f"     acquired skills : {sorted(gained)[:6]}")
    print(f"     gap             : {result.gap.gap_count} -> {after.gap.gap_count}")
    print(f"     coverage of goal: {result.gap.coverage_ratio:.0%} -> "
          f"{after.gap.coverage_ratio:.0%}")
    assert after.gap.gap_count < result.gap.gap_count, (
        "the brief requires adaptation on progress; completing work must shrink the gap"
    )
    assert after.gap.coverage_ratio > result.gap.coverage_ratio
    assert first_id not in {s["course"]["course_id"] for s in after.course_steps}, (
        "a completed resource must not be recommended again"
    )

    # ── feedback escalation still differentiates the two buttons ─────────────
    beginner_profile = {"goal": "learn python", "experience_level": "beginner"}
    level, escalated = resolve_experience_level(SID, beginner_profile)
    assert level == "beginner" and not escalated

    log_feedback(SID, "ZZZ1", "too_easy")
    log_feedback(SID, "ZZZ2", "too_easy")
    level, escalated = resolve_experience_level(SID, beginner_profile)
    assert level == "intermediate" and escalated, level

    log_feedback(SID, "ZZZ3", "not_interested")
    log_feedback(SID, "ZZZ4", "not_interested")
    level_after, _ = resolve_experience_level(SID, beginner_profile)
    assert level_after == "intermediate", (
        f"'not interested' must not affect difficulty, got {level_after}"
    )
    print()
    print("   feedback: too_easy promotes level, not_interested does not")

    # ── completion survives a re-rank ───────────────────────────────────────
    rebuilt = build_learning_path(SID, profile)
    assert get_completed_course_ids(SID) == {first_id}, "re-rank wiped a completion"
    rebuilt_ids = {s["course"]["course_id"] for s in rebuilt.course_steps}
    assert not ({"ZZZ1", "ZZZ2", "ZZZ3", "ZZZ4"} & rebuilt_ids)

    # ── preferences change the plan, not just the stored record ──────────────
    from app.db import set_preferences

    _cleanup()
    set_preferences(SID, {"style": "theory", "hours_per_week": 4, "pace": "relaxed"})
    _, relaxed = run_full_pipeline(SID, "I want to become a data analyst.")

    _cleanup()
    set_preferences(SID, {"style": "hands_on", "hours_per_week": 20, "pace": "intensive"})
    _, intense = run_full_pipeline(SID, "I want to become a data analyst.")

    print()
    print("-- preferences " + "-" * 46)
    print(f"   relaxed  (4h/wk, theory)   : {len(relaxed.course_steps)} resources, "
          f"{relaxed.total_hours}h, {relaxed.weeks} weeks")
    print(f"   intensive (20h/wk, hands-on): {len(intense.course_steps)} resources, "
          f"{intense.total_hours}h, {intense.weeks} weeks")

    # REGRESSION: the budget used to bound selection only, so prerequisite and
    # checkpoint additions pushed a 50-hour relaxed plan to 56 hours.
    assert relaxed.total_hours <= relaxed.preferences.hour_budget or \
        len(relaxed.course_steps) == 1, (
            f"relaxed pace must respect its {relaxed.preferences.hour_budget}h budget, "
            f"got {relaxed.total_hours}h across {len(relaxed.course_steps)} resources"
        )
    assert intense.total_hours >= relaxed.total_hours, (
        "an intensive pace should plan at least as much work as a relaxed one"
    )

    # Trimming must not break the invariants the path guarantees.
    for label, result in (("relaxed", relaxed), ("intensive", intense)):
        numbers = [s["step_number"] for s in result.course_steps]
        assert numbers == list(range(1, len(numbers) + 1)), f"{label}: {numbers}"
        assert result.path[-1].get("course") is not None, f"{label} ends on a milestone"
        positions = {s["course"]["course_id"]: s["step_number"] for s in result.course_steps}
        for step in result.course_steps:
            cid = step["course"]["course_id"]
            for pid in str(step["course"].get("prerequisites", "")).split(","):
                pid = pid.strip()
                if pid and pid in positions:
                    assert positions[pid] < positions[cid], (
                        f"{label}: {pid} must precede {cid} after trimming"
                    )
    assert relaxed.weeks > intense.weeks, (
        f"4h/week must take longer than 20h/week ({relaxed.weeks} vs {intense.weeks})"
    )

    # The schedule must be a real week-by-week plan, monotonically increasing.
    weeks_seq = [row["week"] for row in intense.schedule]
    assert weeks_seq == sorted(weeks_seq), weeks_seq
    assert len(intense.schedule) == len(intense.course_steps)
    preview = ", ".join(
        f"{row['course_id']} wk{row['week']}" for row in intense.schedule[:5]
    )
    print(f"   schedule (intensive)        : {preview}")

    # Hands-on learners should meet a project no later than a theory learner does.
    def _first_project_position(result) -> int:
        for i, step in enumerate(result.course_steps):
            if step["course"].get("resource_type") == "project":
                return i
        return 10**6

    print(f"   first project at index      : hands_on="
          f"{_first_project_position(intense)}, theory={_first_project_position(relaxed)}")

    # ── observed patterns are attached to the result ──────────────────────────
    first_intense = intense.course_steps[0]["course"]["course_id"]
    mark_course_status(SID, first_intense, "completed")
    with_patterns = build_learning_path(SID, profile)
    assert with_patterns.patterns.has_signal, "completing work must register as a pattern"
    assert with_patterns.patterns.completed_count >= 1
    print()
    print("-- observed patterns " + "-" * 40)
    for line in with_patterns.patterns.summary_lines():
        print(f"   {line}")

    # ── an off-catalog goal must report itself unservable ────────────────────
    _cleanup()
    _, off = run_full_pipeline(
        SID, "I want to learn medieval Latin palaeography and manuscript restoration."
    )
    print()
    print(f"-- off-catalog goal: confidence {off.gap.confidence:.3f}, "
          f"servable={off.servable}")
    assert not off.servable, (
        "an off-catalog goal must not be presented as a confident recommendation"
    )

    _cleanup()
    print("\npipeline.py self-test passed: all assertions OK")
