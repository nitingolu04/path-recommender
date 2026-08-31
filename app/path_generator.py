"""
path_generator.py — Turn ranked candidate courses into an ordered learning path.

Algorithm
---------
1.  Expand the candidate set with any prerequisite courses the recommender did
    not itself return, so the sequence is actually completable.
2.  Topologically sort the result with Kahn's algorithm over the prerequisite
    graph, using (difficulty, -score) as the tiebreaker among courses that are
    simultaneously available.  Difficulty first means beginner material leads
    even where no explicit prerequisite forces it.
3.  Insert a milestone checkpoint after every ``MILESTONE_EVERY`` courses.

Output shape
------------
    [
      {"step_number": 1, "course": {...}, "similarity_score": 0.78,
       "is_prerequisite_filler": False, "milestone": None},
      {"step_number": 3, "course": None, "similarity_score": None,
       "is_prerequisite_filler": False, "milestone": "Milestone: ..."},
      ...
    ]

Notes on two deliberate choices
-------------------------------
``step_number`` counts courses only and is contiguous (1, 2, 3, ...).  Milestone
entries carry the number of the course they follow rather than consuming a step
of their own.  Previously milestones incremented the counter, so the fourth
course in a path was labelled "Step 5" and the numbering looked broken.

Courses pulled in purely as prerequisites are flagged with
``is_prerequisite_filler``.  They were never scored by the recommender, so their
score is 0.0; the flag lets the UI label them as groundwork instead of rendering
a misleading "0% match".
"""

from __future__ import annotations

from collections import defaultdict, deque

from app.config import (
    DIFFICULTY_ORDER,
    load_catalog_map,
    parse_prerequisites,
    parse_skills,
)

#: Number of courses between milestone checkpoints.
MILESTONE_EVERY = 3

#: Reinforcement items to add at each checkpoint: one assessment to validate what
#: was just taught, then one project to apply it. Capped so a long path does not
#: become mostly checkpoints.
MAX_REINFORCEMENT_PER_CHECKPOINT = 2

#: How many levels of prerequisites to pull in beyond the recommended courses.
#:
#: The walk used to be unbounded, which made advanced goals unusable: because
#: advanced courses sit at the end of long dependency chains, five recommended
#: courses expanded into a twenty-course path whose first entries were absolute
#: basics.  Two levels keeps a path completable without reconstructing an entire
#: curriculum from first principles.
MAX_PREREQUISITE_DEPTH = 2

#: Fallback checkpoint text, used only when the catalog offers no project or
#: assessment whose prerequisites are satisfied at that point.
#:
#: Previously these five strings *were* the milestone feature — inserted by
#: position, referring to project work that did not exist as anything the learner
#: could open. Real project and assessment resources are now inserted instead, and
#: these remain only as a graceful fallback.
MILESTONE_MESSAGES = [
    "Checkpoint: solid start. Review what you've covered before moving on.",
    "Checkpoint: halfway. Share your work with someone and act on their feedback.",
    "Checkpoint: foundations in place. Try applying these skills to your own data.",
    "Checkpoint: advanced territory. Consider contributing to an open-source project.",
    "Checkpoint: final stretch. Write up what you've built.",
]

#: Headline shown with an inserted reinforcement resource, by type.
REINFORCEMENT_INTRO = {
    "assessment": "Checkpoint: confirm what you've learned before going further.",
    "project": "Checkpoint: put the last few courses to work on something real.",
}


def generate_path(
    candidates: list[dict],
    experience_level: str | None = None,
    max_prerequisite_depth: int = MAX_PREREQUISITE_DEPTH,
    held_skills: list[str] | None = None,
    held_resource_ids: list[str] | None = None,
    reinforcement_order: tuple[str, ...] = ("assessment", "project"),
) -> list[dict]:
    """
    Order ``candidates`` into a sequenced learning path and insert milestones.

    Parameters
    ----------
    candidates : list[dict]
        Resource dicts from the recommendation engine or the coverage selector.
        Each needs at least ``course_id``, ``difficulty_level``,
        ``prerequisites`` and ``similarity_score``.
    experience_level : str | None
        When given, prerequisite resources pitched *below* this level are
        omitted, on the reasoning that someone at intermediate level has already
        covered beginner groundwork.  Selected resources are never dropped this
        way — only the extra dependencies pulled in automatically.
    max_prerequisite_depth : int
        How many levels of the dependency chain to follow.
    held_skills : list[str] | None
        Skills the learner already has.  A prerequisite whose skills are *all*
        held is dropped, because the learner has the knowledge it exists to
        supply.

        This addresses a modelling limitation in the catalog: prerequisites are
        expressed as resource IDs, so nothing but ``DS001`` itself can satisfy a
        dependency on ``DS001``. A learner with five years of Python would still
        have an introductory Python course inserted ahead of the material they
        actually came for. Checking skills rather than IDs lets prior knowledge
        discharge a dependency. The test is deliberately strict — *every* skill
        must be held — so partial overlap does not skip real groundwork.
    held_resource_ids : list[str] | None
        Resources the learner has already completed.  Counted as satisfied when
        deciding whether a project or assessment is unlockable, so prior work
        counts toward reaching a checkpoint.
    reinforcement_order : tuple[str, ...]
        Which reinforcement type to try first at a checkpoint.  Comes from the
        learner's stated style: hands-on learners meet the project first, others
        the assessment.  This is where that preference has an observable effect.

    Returns
    -------
    list[dict]
        Ordered step dicts with keys ``step_number``, ``course``,
        ``similarity_score``, ``is_prerequisite_filler`` and ``milestone``.
    """
    if not candidates:
        return []

    catalog = load_catalog_map()
    candidate_ids = {c["course_id"] for c in candidates}
    course_map: dict[str, dict] = {c["course_id"]: dict(c) for c in candidates}

    user_rank = (
        DIFFICULTY_ORDER.get(str(experience_level).lower()) if experience_level else None
    )
    held = [s.strip().lower() for s in (held_skills or []) if s and s.strip()]

    # Pull in prerequisites the recommender did not surface, otherwise the path
    # would tell the user to start a resource they cannot yet take.
    extra_ids = _expand_prerequisites(candidate_ids, catalog, max_prerequisite_depth)
    for cid in extra_ids - candidate_ids:
        if cid not in catalog:
            continue

        filler = dict(catalog[cid])

        # Skip groundwork the user is already past by level.
        if user_rank is not None:
            filler_rank = DIFFICULTY_ORDER.get(str(filler.get("difficulty_level", "")).lower(), 0)
            if filler_rank < user_rank:
                continue

        # Skip groundwork whose content the user demonstrably already holds.
        if held and _all_skills_held(filler, held):
            continue

        filler["similarity_score"] = 0.0
        filler["semantic_score"] = 0.0
        filler["level_fit"] = None
        filler["is_prerequisite_filler"] = True
        course_map[cid] = filler

    for cid, course in course_map.items():
        course.setdefault("is_prerequisite_filler", False)

    ordered_ids = _topological_sort(list(course_map.keys()), course_map)

    path: list[dict] = []
    step_number = 0
    milestone_idx = 0

    # Tracks what the learner will have completed by this point in the sequence,
    # which is what decides whether a project or assessment is yet unlockable.
    scheduled_ids: set[str] = set(held_resource_ids or [])
    taught_skills: set[str] = set(held)
    used_ids: set[str] = set(course_map.keys()) | set(held_resource_ids or [])

    def _append(resource: dict, *, filler: bool = False, intro: str | None = None) -> None:
        nonlocal step_number
        step_number += 1
        path.append(
            {
                "step_number": step_number,
                "course": resource,
                "similarity_score": resource.get("similarity_score", 0.0),
                "is_prerequisite_filler": filler,
                "milestone": None,
                "checkpoint_note": intro,
            }
        )

    for position, cid in enumerate(ordered_ids, start=1):
        course = course_map[cid]
        _append(course, filler=course.get("is_prerequisite_filler", False))

        scheduled_ids.add(cid)
        taught_skills.update(parse_skills(course.get("skills", "")))

        is_last = position == len(ordered_ids)
        if position % MILESTONE_EVERY != 0 and not is_last:
            continue

        # Checkpoint reached. Prefer real resources over a canned message:
        # validate first, then apply.
        inserted = 0
        for kind in reinforcement_order:
            if inserted >= MAX_REINFORCEMENT_PER_CHECKPOINT:
                break
            pick = _pick_reinforcement(kind, scheduled_ids, taught_skills, used_ids, catalog)
            if pick is None:
                continue
            enriched = dict(pick)
            enriched["similarity_score"] = 0.0
            enriched["semantic_score"] = 0.0
            enriched["level_fit"] = None
            enriched["is_reinforcement"] = True
            _append(enriched, intro=REINFORCEMENT_INTRO.get(kind))
            used_ids.add(pick["course_id"])
            scheduled_ids.add(pick["course_id"])
            inserted += 1

        # Only fall back to prose when the catalog offered nothing, and never
        # trailing at the very end of the path.
        if inserted == 0 and not is_last:
            path.append(
                {
                    "step_number": step_number,
                    "course": None,
                    "similarity_score": None,
                    "is_prerequisite_filler": False,
                    "milestone": MILESTONE_MESSAGES[
                        min(milestone_idx, len(MILESTONE_MESSAGES) - 1)
                    ],
                    "checkpoint_note": None,
                }
            )
            milestone_idx += 1

    return path


def _pick_reinforcement(
    resource_type: str,
    scheduled_ids: set[str],
    taught_skills: set[str],
    used_ids: set[str],
    catalog: dict[str, dict],
) -> dict | None:
    """
    Choose the best unlockable project or assessment for a checkpoint.

    A candidate qualifies only when **every** prerequisite is already scheduled
    earlier in the path, which is what makes the insertion safe: the resource can
    never appear before the material it depends on, so topological validity is
    preserved by construction rather than by re-sorting afterwards.

    Among the qualifying candidates, the one sharing the most skills with what has
    been taught so far wins. That keeps the checkpoint about the work just
    completed rather than being a generic detour, and it is why a project can only
    appear once the courses feeding it are in place.

    Returns ``None`` when nothing is unlockable yet, in which case the caller
    falls back to a short prose checkpoint.
    """
    best: dict | None = None
    best_score = 0

    for cid, row in catalog.items():
        if row.get("resource_type") != resource_type or cid in used_ids:
            continue

        prereqs = parse_prerequisites(row.get("prerequisites", ""))
        # Ungated resources are excluded: without a prerequisite there is nothing
        # to establish that the learner is ready for it.
        if not prereqs or not set(prereqs) <= scheduled_ids:
            continue

        overlap = len(set(parse_skills(row.get("skills", ""))) & taught_skills)
        if overlap > best_score:
            best, best_score = row, overlap

    return best


# ── Internals ──────────────────────────────────────────────────────────────────

def _expand_prerequisites(
    candidate_ids: set[str],
    catalog: dict[str, dict],
    max_depth: int = MAX_PREREQUISITE_DEPTH,
) -> set[str]:
    """
    Walk up to ``max_depth`` levels of the prerequisite graph from
    ``candidate_ids`` and return every course ID reached, candidates included.

    Breadth-first with a visited set and an explicit depth per node, so a
    diamond dependency (two courses sharing a prerequisite) resolves once and a
    cyclic edge in the data cannot cause infinite traversal.
    """
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque((cid, 0) for cid in candidate_ids)
    while queue:
        cid, depth = queue.popleft()
        if cid in visited:
            continue
        visited.add(cid)
        if depth >= max_depth:
            continue
        for pid in parse_prerequisites(catalog.get(cid, {}).get("prerequisites", "")):
            if pid not in visited:
                queue.append((pid, depth + 1))
    return visited


def _all_skills_held(resource: dict, held: list[str]) -> bool:
    """
    Return ``True`` if every skill a resource teaches is already held.

    Uses the same conservative equivalence as gap analysis, so "sql" does not
    count as covering "sql injection". A resource with no listed skills returns
    ``False``: absence of data is not evidence the learner knows it.
    """
    from app.skill_gap import is_held

    skills = parse_skills(resource.get("skills", ""))
    if not skills:
        return False
    return all(is_held(skill, held) for skill in skills)


def _sort_key(cid: str, course_map: dict[str, dict]) -> tuple[int, float]:
    """Order by difficulty ascending, then by score descending."""
    course = course_map[cid]
    rank = DIFFICULTY_ORDER.get(str(course.get("difficulty_level", "beginner")).lower(), 0)
    return (rank, -float(course.get("similarity_score") or 0.0))


def _topological_sort(course_ids: list[str], course_map: dict[str, dict]) -> list[str]:
    """
    Kahn's algorithm over the prerequisite graph, restricted to ``course_ids``.

    Rather than a plain FIFO queue, the frontier of currently-available courses
    is re-sorted on each iteration by (difficulty, -score).  A FIFO would emit a
    valid order but an arbitrary one among courses whose prerequisites are all
    satisfied; re-sorting means the easiest, best-matching available course is
    always chosen next.

    A cycle in the data leaves nodes with a permanently non-zero in-degree.
    Those are appended in difficulty order rather than dropped, so a malformed
    catalog degrades the ordering instead of losing courses.
    """
    id_set = set(course_ids)
    in_degree: dict[str, int] = {cid: 0 for cid in course_ids}
    dependents: dict[str, list[str]] = defaultdict(list)

    for cid in course_ids:
        for pid in parse_prerequisites(course_map[cid].get("prerequisites", "")):
            if pid in id_set and pid != cid:
                dependents[pid].append(cid)
                in_degree[cid] += 1

    available = [cid for cid in course_ids if in_degree[cid] == 0]
    result: list[str] = []

    while available:
        available.sort(key=lambda c: _sort_key(c, course_map))
        cid = available.pop(0)
        result.append(cid)
        for dependent in dependents[cid]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                available.append(dependent)

    if len(result) < len(course_ids):
        remaining = [cid for cid in course_ids if cid not in set(result)]
        print(
            f"[PathGenerator] Warning: prerequisite cycle detected; "
            f"appending {len(remaining)} course(s) in difficulty order."
        )
        result.extend(sorted(remaining, key=lambda c: _sort_key(c, course_map)))

    return result


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from app.config import load_catalog_df

    df = load_catalog_df()
    sample = df[df["category"] == "Data Science"].head(6).to_dict(orient="records")
    for i, row in enumerate(sample):
        row["similarity_score"] = round(0.90 - i * 0.05, 2)

    path = generate_path(sample)
    course_steps = [s for s in path if s["course"]]

    print("-- Generated learning path " + "-" * 34)
    for step in path:
        if step["milestone"]:
            print(f"      | {step['milestone']}")
            continue
        course = step["course"]
        tag = " (prerequisite)" if step["is_prerequisite_filler"] else ""
        if course.get("is_reinforcement"):
            tag = " (reinforcement)"
        if step.get("checkpoint_note"):
            print(f"      | {step['checkpoint_note']}")
        print(
            f"  Step {step['step_number']:>2}  [{step['similarity_score']:.2f}]  "
            f"[{course['resource_type'][:4]}]  {course['course_id']}  {course['title'][:44]} "
            f"({course['difficulty_level']}){tag}"
        )

    # ── step numbers must be contiguous starting at 1 ────────────────────────
    numbers = [s["step_number"] for s in course_steps]
    assert numbers == list(range(1, len(course_steps) + 1)), (
        f"REGRESSION: course step numbers must be contiguous, got {numbers}"
    )

    # ── every prerequisite present in the path must come before its dependent ─
    position = {s["course"]["course_id"]: s["step_number"] for s in course_steps}
    for step in course_steps:
        cid = step["course"]["course_id"]
        for pid in parse_prerequisites(step["course"].get("prerequisites", "")):
            if pid in position:
                assert position[pid] < position[cid], (
                    f"prerequisite {pid} (step {position[pid]}) must precede "
                    f"{cid} (step {position[cid]})"
                )
    print(f"\ntopological order valid across {len(course_steps)} courses")

    # ── no duplicates, and every input candidate survived ────────────────────
    ids = [s["course"]["course_id"] for s in course_steps]
    assert len(ids) == len(set(ids)), f"duplicate courses in path: {ids}"
    for row in sample:
        assert row["course_id"] in position, f"candidate {row['course_id']} dropped from path"

    # ── each step is exactly one of: selected, prerequisite, reinforcement ────
    sample_ids = {r["course_id"] for r in sample}
    for step in course_steps:
        resource = step["course"]
        is_filler = step["is_prerequisite_filler"]
        is_reinforcement = bool(resource.get("is_reinforcement"))
        if is_filler or is_reinforcement:
            assert step["similarity_score"] == 0.0, step
            assert not (is_filler and is_reinforcement), "a step cannot be both"
        else:
            assert resource["course_id"] in sample_ids, resource["course_id"]

    # ── checkpoints insert real projects and assessments ─────────────────────
    # These used to be five fixed strings inserted by position, referring to
    # project work the learner could not actually open.
    reinforcement = [s for s in course_steps if s["course"].get("is_reinforcement")]
    types_inserted = {s["course"]["resource_type"] for s in reinforcement}
    print(f"reinforcement inserted: {len(reinforcement)} item(s) {sorted(types_inserted)}")
    assert reinforcement, "a 6-course path should reach a checkpoint and unlock something"
    assert types_inserted <= {"project", "assessment"}, types_inserted

    for step in reinforcement:
        assert step["checkpoint_note"], "a reinforcement step needs its checkpoint framing"
        # Its prerequisites must all appear strictly earlier in the path.
        cid = step["course"]["course_id"]
        prereqs = parse_prerequisites(step["course"].get("prerequisites", ""))
        assert prereqs, f"{cid} was inserted without prerequisites"
        for pid in prereqs:
            assert pid in position, f"{cid} inserted before its prerequisite {pid} was scheduled"
            assert position[pid] < position[cid], (
                f"{cid} (step {position[cid]}) precedes prerequisite {pid} "
                f"(step {position[pid]})"
            )

    # A path must never end on a bare prose milestone.
    assert path[-1]["course"] is not None, "path must not end on a milestone"

    # ── empty input is handled ───────────────────────────────────────────────
    assert generate_path([]) == []

    # ── held skills discharge a prerequisite ─────────────────────────────────
    # DS003 lists DS001 as a prerequisite. A learner who already holds
    # everything DS001 teaches should not have DS001 inserted ahead of it.
    ds001_skills = parse_skills(df[df["course_id"] == "DS001"].iloc[0]["skills"])
    ds003 = df[df["course_id"] == "DS003"].iloc[0].to_dict()
    ds003["similarity_score"] = 0.9

    without_prior = generate_path([dict(ds003)])
    ids_without = {s["course"]["course_id"] for s in without_prior if s["course"]}
    assert "DS001" in ids_without, (
        f"DS001 should be pulled in as a prerequisite -> {ids_without}"
    )

    with_prior = generate_path([dict(ds003)], held_skills=ds001_skills)
    ids_with = {s["course"]["course_id"] for s in with_prior if s["course"]}
    assert "DS001" not in ids_with, (
        f"holding DS001's skills should discharge the prerequisite -> {ids_with}"
    )
    assert "DS003" in ids_with, "the requested course must survive"

    # Partial overlap must NOT discharge it — that would skip real groundwork.
    partial = generate_path([dict(ds003)], held_skills=ds001_skills[:1])
    ids_partial = {s["course"]["course_id"] for s in partial if s["course"]}
    assert "DS001" in ids_partial, (
        f"partial skill overlap must not skip a prerequisite -> {ids_partial}"
    )
    print(f"prerequisite discharge: DS001 present={('DS001' in ids_without)} without prior, "
          f"{('DS001' in ids_with)} with prior, {('DS001' in ids_partial)} with partial")

    # ── a synthetic cycle must not lose courses ──────────────────────────────
    cyclic = [
        {"course_id": "X1", "prerequisites": "X2", "difficulty_level": "beginner",
         "similarity_score": 0.9, "title": "X1", "skills": "", "category": "T",
         "description": ""},
        {"course_id": "X2", "prerequisites": "X1", "difficulty_level": "beginner",
         "similarity_score": 0.8, "title": "X2", "skills": "", "category": "T",
         "description": ""},
    ]
    cyclic_path = [s for s in generate_path(cyclic) if s["course"]]
    assert len(cyclic_path) == 2, f"cycle handling dropped courses: {cyclic_path}"

    print("path_generator.py self-test passed: all assertions OK")
