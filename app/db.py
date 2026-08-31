"""
db.py — SQLite persistence layer for the Learning Path Recommender.

Schema
------
    users       one row per session
    profiles    structured profile derived from the natural-language goal
    progress    per-course completion status for a session
    feedback    per-course feedback events (too_easy / not_interested)

Tables are created on demand via ``init_db()`` (CREATE TABLE IF NOT EXISTS), and
``_migrate()`` adds columns introduced after a database file already existed, so
a prototype DB from an earlier run keeps working without manual intervention.

Design note on connections
--------------------------
Each helper opens and closes its own short-lived connection.  For a local
single-user Streamlit prototype the overhead is negligible and it keeps the
functions safe to call from Streamlit's re-run model, where module state is
recreated unpredictably.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from app.config import DB_PATH

__all__ = [
    "DB_PATH",
    "init_db",
    "create_user",
    "upsert_profile",
    "get_profile",
    "set_preferences",
    "get_preferences",
    "upsert_progress",
    "sync_path_progress",
    "get_progress",
    "mark_course_status",
    "get_completed_course_ids",
    "set_prior_learning",
    "get_prior_learning_ids",
    "log_feedback",
    "get_excluded_course_ids",
    "get_feedback_counts",
    "get_feedback_course_ids",
]


def _get_connection() -> sqlite3.Connection:
    """Return a connection with ``row_factory`` set so rows behave like dicts."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create all tables if they do not already exist, then apply migrations.

    Safe to call repeatedly — every statement is idempotent.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_connection()
    with conn:
        conn.executescript("""
            -- One row per anonymous user/session
            CREATE TABLE IF NOT EXISTS users (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL
            );

            -- Structured profile extracted from the user's natural-language goal.
            -- current_skills, target_skills and interests are stored as JSON arrays.
            CREATE TABLE IF NOT EXISTS profiles (
                session_id        TEXT PRIMARY KEY,
                goal              TEXT NOT NULL,
                current_skills    TEXT NOT NULL DEFAULT '[]',   -- JSON array
                experience_level  TEXT NOT NULL DEFAULT 'beginner',
                interests         TEXT NOT NULL DEFAULT '[]',   -- JSON array
                updated_at        TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES users (session_id)
            );

            -- Progress tracking: one row per (session, course).
            -- status: 'not_started' | 'in_progress' | 'completed'
            CREATE TABLE IF NOT EXISTS progress (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                course_id   TEXT NOT NULL,
                step_number INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'not_started',
                -- 'in_app'   completed while using this app
                -- 'declared' prior learning the user told us about
                source      TEXT NOT NULL DEFAULT 'in_app',
                updated_at  TEXT NOT NULL,
                UNIQUE (session_id, course_id),
                FOREIGN KEY (session_id) REFERENCES users (session_id)
            );

            -- Feedback events logged when the user reacts to a recommended course.
            CREATE TABLE IF NOT EXISTS feedback (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT NOT NULL,
                course_id      TEXT NOT NULL,
                feedback_type  TEXT NOT NULL,   -- 'too_easy' | 'not_interested'
                created_at     TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES users (session_id)
            );

            -- Lookups are always scoped by session_id, so index it.
            CREATE INDEX IF NOT EXISTS idx_progress_session
                ON progress (session_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_session
                ON feedback (session_id);
        """)
    conn.close()
    _migrate()


def _migrate() -> None:
    """
    Apply additive schema migrations to an existing database file.

    Each column below was introduced after databases already existed in the wild,
    and ``ALTER TABLE`` is the cheapest fix that preserves existing rows:

    ``profiles.target_skills``
        Added when the profiling engine learned to tell "skills I have" apart
        from "skills I want to learn".
    ``progress.source``
        Added to distinguish prior learning the user declared from resources they
        completed inside the app. Both count as completed for recommendation
        purposes, but only declared rows should disappear when the user edits
        their prior-learning list.
    ``profiles.preferences``
        Added for learning preferences (style, hours per week, pace). Stored as a
        JSON blob rather than one column per field, so adding another preference
        later needs no further migration.
    """
    additions = (
        ("profiles", "target_skills", "TEXT NOT NULL DEFAULT '[]'"),
        ("progress", "source", "TEXT NOT NULL DEFAULT 'in_app'"),
        ("profiles", "preferences", "TEXT NOT NULL DEFAULT '{}'"),
    )
    conn = _get_connection()
    try:
        for table, column, spec in additions:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                with conn:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")
    finally:
        conn.close()


# ── User helpers ───────────────────────────────────────────────────────────────

def create_user(session_id: str) -> None:
    """Insert a new user row; silently ignored if ``session_id`` already exists."""
    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (session_id, created_at) VALUES (?, ?)",
            (session_id, _now()),
        )
    conn.close()


# ── Profile helpers ────────────────────────────────────────────────────────────

def upsert_profile(
    session_id: str,
    goal: str,
    current_skills: list,
    experience_level: str,
    interests: list,
    target_skills: list | None = None,
) -> None:
    """
    Insert or replace the profile for a session.  List fields are JSON-encoded.

    Parameters
    ----------
    current_skills : list
        Skills the user stated they already have.
    target_skills : list | None
        Skills the user stated they want to acquire.  Kept separate from
        ``current_skills`` so the explainer never claims a course "builds on"
        experience the user does not have.
    """
    create_user(session_id)
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO profiles
                (session_id, goal, current_skills, experience_level,
                 interests, target_skills, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                goal             = excluded.goal,
                current_skills   = excluded.current_skills,
                experience_level = excluded.experience_level,
                interests        = excluded.interests,
                target_skills    = excluded.target_skills,
                updated_at       = excluded.updated_at
            """,
            (
                session_id,
                goal,
                json.dumps(current_skills),
                experience_level,
                json.dumps(interests),
                json.dumps(target_skills or []),
                _now(),
            ),
        )
    conn.close()


def get_profile(session_id: str) -> dict | None:
    """Return the profile dict for a session, or ``None`` if not found."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM profiles WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    profile = dict(row)
    profile["current_skills"] = json.loads(profile.get("current_skills") or "[]")
    profile["interests"] = json.loads(profile.get("interests") or "[]")
    profile["target_skills"] = json.loads(profile.get("target_skills") or "[]")
    profile["preferences"] = json.loads(profile.get("preferences") or "{}")
    return profile


def set_preferences(session_id: str, preferences: dict) -> None:
    """
    Store the learner's stated learning preferences.

    Written separately from ``upsert_profile`` because preferences outlive any one
    goal: re-describing what you want to learn should not reset the fact that you
    have six hours a week and prefer hands-on work.
    """
    create_user(session_id)
    conn = _get_connection()
    with conn:
        # A profile row may not exist yet if preferences are set before a goal.
        conn.execute(
            """
            INSERT INTO profiles
                (session_id, goal, current_skills, experience_level,
                 interests, target_skills, preferences, updated_at)
            VALUES (?, '', '[]', 'beginner', '[]', '[]', ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                preferences = excluded.preferences,
                updated_at  = excluded.updated_at
            """,
            (session_id, json.dumps(preferences or {}), _now()),
        )
    conn.close()


def get_preferences(session_id: str) -> dict:
    """Return stored preferences, or an empty dict when none are set."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT preferences FROM profiles WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return {}
    return json.loads(row["preferences"] or "{}")


# ── Progress helpers ───────────────────────────────────────────────────────────

def upsert_progress(
    session_id: str,
    course_id: str,
    step_number: int,
    status: str = "not_started",
) -> None:
    """Insert or update a progress row for a (session, course) pair."""
    create_user(session_id)
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO progress (session_id, course_id, step_number, status, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, course_id) DO UPDATE SET
                step_number = excluded.step_number,
                status      = excluded.status,
                updated_at  = excluded.updated_at
            """,
            (session_id, course_id, step_number, status, _now()),
        )
    conn.close()


def sync_path_progress(session_id: str, path: list[dict]) -> int:
    """
    Ensure a ``progress`` row exists for every course in a generated path.

    This is the fix for a bug that silently disabled the whole dashboard: the
    UI's "Mark Done" button issued an ``UPDATE`` against a row that had never
    been inserted, so it matched zero rows and the click did nothing.  Progress
    stayed at 0%, "Skills Gained" stayed empty, and "Next Action" never advanced.

    Existing ``status`` values are deliberately preserved — only ``step_number``
    is refreshed — so re-ranking after feedback reorders the path without
    discarding what the user has already completed.

    Parameters
    ----------
    session_id : str
        Session the path belongs to.
    path : list[dict]
        Output of ``path_generator.generate_path()``.  Milestone entries (where
        ``course`` is ``None``) are skipped.

    Returns
    -------
    int
        Number of course rows synced.
    """
    course_steps = [s for s in path if s.get("course")]
    if not course_steps:
        return 0

    create_user(session_id)
    rows = [
        (session_id, s["course"]["course_id"], int(s.get("step_number", 0)), "not_started", _now())
        for s in course_steps
    ]

    conn = _get_connection()
    with conn:
        conn.executemany(
            """
            INSERT INTO progress (session_id, course_id, step_number, status, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, course_id) DO UPDATE SET
                step_number = excluded.step_number,
                updated_at  = excluded.updated_at
            """,
            rows,
        )
    conn.close()
    return len(rows)


def get_progress(session_id: str) -> list[dict]:
    """Return all progress rows for a session, ordered by step number."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM progress WHERE session_id = ? ORDER BY step_number",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_course_status(session_id: str, course_id: str, status: str) -> None:
    """
    Set the status of a single course for a session.

    Implemented as an upsert rather than a bare ``UPDATE``.  The previous
    UPDATE-only version was a no-op whenever the progress row did not already
    exist, which was always the case in the running app.  Writing it as an
    upsert makes the operation correct on its own, independent of whether
    ``sync_path_progress()`` ran first.
    """
    create_user(session_id)
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO progress (session_id, course_id, step_number, status, updated_at)
            VALUES (?, ?, 0, ?, ?)
            ON CONFLICT(session_id, course_id) DO UPDATE SET
                status     = excluded.status,
                updated_at = excluded.updated_at
            """,
            (session_id, course_id, status, _now()),
        )
    conn.close()


def get_completed_course_ids(session_id: str) -> set[str]:
    """
    Return every resource ID counted as completed, from either source.

    Prior learning the user declared is included deliberately: as far as
    recommendation, gap analysis and prerequisites are concerned, a course
    finished last year and one finished in the app are the same fact.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT course_id FROM progress WHERE session_id = ? AND status = 'completed'",
        (session_id,),
    ).fetchall()
    conn.close()
    return {r["course_id"] for r in rows}


# ── Prior learning ─────────────────────────────────────────────────────────────

def set_prior_learning(session_id: str, resource_ids: list[str]) -> int:
    """
    Record the resources a user says they have already completed.

    The brief requires the profiling engine to capture completed courses and
    previous learning history. Rather than adding a parallel store, prior
    learning is written as ``progress`` rows with ``status='completed'`` and
    ``source='declared'``. Everything downstream then picks it up for free:
    ``get_completed_course_ids()`` excludes it from recommendations,
    ``pipeline.acquired_skills()`` counts its skills as held so the gap shrinks,
    and the path generator treats it as satisfying prerequisites.

    Replaces the whole declared set rather than appending, so unticking an item
    in the UI actually removes it. Rows completed *inside* the app are never
    touched — losing real progress because someone edited a checkbox list would
    be a bad trade.

    Returns
    -------
    int
        Number of declared resources recorded.
    """
    create_user(session_id)
    conn = _get_connection()
    with conn:
        conn.execute(
            "DELETE FROM progress WHERE session_id = ? AND source = 'declared'",
            (session_id,),
        )
        if resource_ids:
            conn.executemany(
                """
                INSERT INTO progress
                    (session_id, course_id, step_number, status, source, updated_at)
                VALUES (?, ?, 0, 'completed', 'declared', ?)
                ON CONFLICT(session_id, course_id) DO UPDATE SET
                    status     = 'completed',
                    source     = 'declared',
                    updated_at = excluded.updated_at
                """,
                [(session_id, rid, _now()) for rid in dict.fromkeys(resource_ids)],
            )
    conn.close()
    return len(set(resource_ids))


def get_prior_learning_ids(session_id: str) -> set[str]:
    """Return only the resource IDs the user declared as prior learning."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT course_id FROM progress WHERE session_id = ? AND source = 'declared'",
        (session_id,),
    ).fetchall()
    conn.close()
    return {r["course_id"] for r in rows}


# ── Feedback helpers ───────────────────────────────────────────────────────────

def log_feedback(session_id: str, course_id: str, feedback_type: str) -> None:
    """
    Append a feedback event for a (session, course) pair.

    This records the event only.  Acting on it is the caller's job:
    ``get_excluded_course_ids()`` drops the course from future candidate sets,
    and ``get_feedback_counts()`` lets the recommender escalate difficulty when
    the user repeatedly reports courses as too easy.

    Parameters
    ----------
    feedback_type : str
        ``'too_easy'`` or ``'not_interested'``.
    """
    create_user(session_id)
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO feedback (session_id, course_id, feedback_type, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, course_id, feedback_type, _now()),
        )
    conn.close()


def get_excluded_course_ids(session_id: str) -> set[str]:
    """
    Return every course ID the user gave negative feedback on, of any type.

    Both feedback types remove the specific course from future candidate sets —
    "too easy" and "not interested" both mean "don't show me this one again".
    What differs is the *secondary* effect: see ``get_feedback_counts()``, which
    drives difficulty escalation for ``too_easy``.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT DISTINCT course_id FROM feedback WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    conn.close()
    return {r["course_id"] for r in rows}


def get_feedback_counts(session_id: str) -> dict[str, int]:
    """
    Return ``{feedback_type: count}`` for a session.

    Lets callers distinguish the two feedback buttons, which previously behaved
    identically because only the union of course IDs was ever read back.
    """
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT feedback_type, COUNT(*) AS n
        FROM feedback WHERE session_id = ?
        GROUP BY feedback_type
        """,
        (session_id,),
    ).fetchall()
    conn.close()
    return {r["feedback_type"]: r["n"] for r in rows}


def get_feedback_course_ids(session_id: str, feedback_type: str) -> set[str]:
    """Return course IDs the user flagged with one specific ``feedback_type``."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT DISTINCT course_id FROM feedback
        WHERE session_id = ? AND feedback_type = ?
        """,
        (session_id, feedback_type),
    ).fetchall()
    conn.close()
    return {r["course_id"] for r in rows}


# ── Utility ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Uses a throwaway session ID and deletes its rows at the end so running
    # this never pollutes the real database.
    SID = "__selftest_db__"

    def _cleanup() -> None:
        conn = _get_connection()
        with conn:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (SID,))
        conn.close()

    init_db()
    _cleanup()

    # ── profile round-trip, including the new target_skills field ─────────────
    upsert_profile(
        SID,
        goal="I want to become a data analyst",
        current_skills=["excel"],
        experience_level="beginner",
        interests=["Data Science"],
        target_skills=["sql", "python"],
    )
    profile = get_profile(SID)
    assert profile is not None, "profile should exist after upsert"
    assert profile["current_skills"] == ["excel"], profile["current_skills"]
    assert profile["target_skills"] == ["sql", "python"], profile["target_skills"]
    assert profile["interests"] == ["Data Science"], profile["interests"]

    # ── REGRESSION: mark_course_status must work with no pre-existing row ─────
    # The old UPDATE-only version silently matched zero rows here, which is what
    # left the dashboard permanently stuck at 0% complete.
    mark_course_status(SID, "DS001", "completed")
    assert get_completed_course_ids(SID) == {"DS001"}, (
        "mark_course_status must create the row when absent"
    )

    # ── sync_path_progress seeds rows for a generated path ────────────────────
    fake_path = [
        {"step_number": 1, "course": {"course_id": "DS001"}, "milestone": None},
        {"step_number": 2, "course": {"course_id": "DS002"}, "milestone": None},
        {"step_number": 3, "course": None, "milestone": "checkpoint"},
        {"step_number": 4, "course": {"course_id": "DS003"}, "milestone": None},
    ]
    synced = sync_path_progress(SID, fake_path)
    assert synced == 3, f"expected 3 course rows synced, got {synced}"
    assert len(get_progress(SID)) == 3, get_progress(SID)

    # Syncing must NOT wipe an existing completion.
    assert get_completed_course_ids(SID) == {"DS001"}, (
        "sync_path_progress must preserve existing status values"
    )

    # ── preferences round-trip ────────────────────────────────────────────────
    assert get_preferences(SID) == {}, "no preferences should be set initially"
    set_preferences(SID, {"style": "hands_on", "hours_per_week": 8, "pace": "steady"})
    assert get_preferences(SID)["style"] == "hands_on"
    assert get_preferences(SID)["hours_per_week"] == 8

    # Preferences must outlive a change of goal: re-describing what you want to
    # learn should not reset how you want to learn it.
    upsert_profile(
        SID,
        goal="a completely different goal",
        current_skills=[],
        experience_level="beginner",
        interests=[],
        target_skills=[],
    )
    assert get_preferences(SID)["style"] == "hands_on", (
        "upsert_profile must not clobber stored preferences"
    )
    assert get_profile(SID)["preferences"]["hours_per_week"] == 8

    # Restore the profile the later assertions expect.
    upsert_profile(
        SID,
        goal="I want to become a data analyst",
        current_skills=["excel"],
        experience_level="beginner",
        interests=["Data Science"],
        target_skills=["sql", "python"],
    )

    # ── prior learning ────────────────────────────────────────────────────────
    n = set_prior_learning(SID, ["DS004", "DS005"])
    assert n == 2, n
    assert get_prior_learning_ids(SID) == {"DS004", "DS005"}
    # Declared prior learning must count as completed everywhere downstream.
    assert {"DS004", "DS005"} <= get_completed_course_ids(SID), get_completed_course_ids(SID)
    # DS001 was completed in the app earlier and must still be there.
    assert "DS001" in get_completed_course_ids(SID)

    # Replacing the declared set removes what was unticked...
    set_prior_learning(SID, ["DS005"])
    assert get_prior_learning_ids(SID) == {"DS005"}
    assert "DS004" not in get_completed_course_ids(SID)
    # ...but must never touch progress earned inside the app.
    assert "DS001" in get_completed_course_ids(SID), (
        "editing prior learning must not erase real in-app progress"
    )

    # Clearing it entirely leaves in-app progress intact.
    set_prior_learning(SID, [])
    assert get_prior_learning_ids(SID) == set()
    assert "DS001" in get_completed_course_ids(SID)

    # Re-syncing a path must not resurrect or overwrite declared completions.
    set_prior_learning(SID, ["DS002"])
    sync_path_progress(SID, [{"step_number": 1, "course": {"course_id": "DS002"}}])
    assert "DS002" in get_completed_course_ids(SID), (
        "sync_path_progress must preserve a declared completion"
    )

    # ── typed feedback ────────────────────────────────────────────────────────
    log_feedback(SID, "DS002", "too_easy")
    log_feedback(SID, "DS003", "not_interested")
    log_feedback(SID, "DS004", "too_easy")
    assert get_excluded_course_ids(SID) == {"DS002", "DS003", "DS004"}
    assert get_feedback_counts(SID) == {"too_easy": 2, "not_interested": 1}
    assert get_feedback_course_ids(SID, "too_easy") == {"DS002", "DS004"}
    assert get_feedback_course_ids(SID, "not_interested") == {"DS003"}

    _cleanup()
    assert get_profile(SID) is None, "cleanup should have removed the test profile"
    assert get_prior_learning_ids(SID) == set()
    print("db.py self-test passed: all assertions OK")
