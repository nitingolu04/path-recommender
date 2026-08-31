"""
config.py — Single source of truth for filesystem paths and catalog loading.

Why this module exists
---------------------
``recommendation_engine``, ``profiling_engine`` and ``path_generator`` each used
to rebuild the catalog path by hand *and* each called ``pd.read_csv()``
independently. That caused two concrete problems:

1.  The path string was duplicated in three places, so the on-disk filename
    casing had to be fixed in three places. Case-insensitive filesystems (NTFS)
    hid a mismatch on Windows that would raise ``FileNotFoundError`` on
    Linux / macOS / Docker.
2.  A single pipeline run parsed the same CSV three separate times.

Both are solved here: one path constant, one cached loader.

The catalog holds three resource types
-------------------------------------
The brief asks for a roadmap of "courses, projects and assessments", so
``data/catalog.csv`` carries a ``resource_type`` discriminator rather than being
courses only:

    course      teaches skills
    project     applies skills taught elsewhere
    assessment  validates a cluster of skills

One table means one embedding index and one prerequisite graph, so the path
generator can interleave all three types under the same topological rules.

Note on the identifier column
-----------------------------
The primary key is still named ``course_id`` even though it now addresses
projects and assessments. Renaming it would ripple through ``progress.course_id``
and ``feedback.course_id`` in SQLite plus every module, for no functional gain.
``resource_type`` is the authoritative type marker.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# config.py lives in app/, so the project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CATALOG_CSV = os.path.join(DATA_DIR, "catalog.csv")
DB_PATH = os.path.join(DATA_DIR, "recommender.db")

# Columns every downstream module relies on.
REQUIRED_COLUMNS = (
    "course_id",
    "title",
    "description",
    "skills",
    "prerequisites",
    "difficulty_level",
    "category",
    "resource_type",
    "duration_hours",
)

# Canonical difficulty ordering, shared by the recommender and the path builder
# so the two can never drift out of sync.
DIFFICULTY_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}
DIFFICULTY_LEVELS = ("beginner", "intermediate", "advanced")

# Resource types, in the order they naturally occur in a learning cycle:
# learn it, apply it, prove it.
RESOURCE_TYPES = ("course", "project", "assessment")

#: Fallback duration when a row is missing one, by difficulty.
DEFAULT_HOURS = {"beginner": 6, "intermediate": 15, "advanced": 30}


def resolve_case_insensitive(path: str) -> str:
    """
    Return ``path`` if it exists, otherwise look for a case-insensitive match in
    the same directory and return that instead.

    A safety net rather than a licence to be sloppy: the repository ships
    ``data/catalog.csv`` in lowercase. This exists so a catalog restored from a
    case-mangling source (some zip tools, some cloud syncs) still loads rather
    than crashing the app at startup.
    """
    if os.path.exists(path):
        return path

    directory, filename = os.path.split(path)
    if not os.path.isdir(directory):
        return path  # let the caller raise a clear FileNotFoundError

    target = filename.lower()
    for candidate in os.listdir(directory):
        if candidate.lower() == target:
            return os.path.join(directory, candidate)
    return path


@lru_cache(maxsize=1)
def load_catalog_df() -> pd.DataFrame:
    """
    Load and validate the learning catalog, cached for the process lifetime.

    Returns
    -------
    pd.DataFrame
        The catalog with ``prerequisites`` normalised to ``str`` ("" when the
        resource has none) and ``duration_hours`` coerced to ``int``, so
        downstream code never has to juggle NaN.

    Raises
    ------
    FileNotFoundError
        If the catalog is missing entirely.
    ValueError
        If a required column is absent — failing loudly at startup beats a
        confusing ``KeyError`` deep inside the ranking code.
    """
    path = resolve_case_insensitive(CATALOG_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Learning catalog not found at '{CATALOG_CSV}'. "
            f"Run 'python generate_catalog.py' to create it."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Catalog '{path}' is missing required column(s): {missing}. "
            f"Expected: {list(REQUIRED_COLUMNS)}. "
            f"If this file predates the projects/assessments change, regenerate "
            f"it with 'python generate_catalog.py'."
        )

    # Normalise the free-text list columns so callers can rely on str ops.
    df["prerequisites"] = (
        df["prerequisites"].fillna("").astype(str).replace("nan", "", regex=False).str.strip()
    )
    df["skills"] = df["skills"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["difficulty_level"] = (
        df["difficulty_level"].fillna("beginner").astype(str).str.strip().str.lower()
    )
    df["resource_type"] = (
        df["resource_type"].fillna("course").astype(str).str.strip().str.lower()
    )

    # Durations drive the schedule view, so a missing value must not become NaN.
    fallback = df["difficulty_level"].map(DEFAULT_HOURS).fillna(10)
    df["duration_hours"] = (
        pd.to_numeric(df["duration_hours"], errors="coerce").fillna(fallback).astype(int)
    )

    unknown = set(df["resource_type"]) - set(RESOURCE_TYPES)
    if unknown:
        raise ValueError(
            f"Catalog contains unknown resource_type value(s): {sorted(unknown)}. "
            f"Expected one of {list(RESOURCE_TYPES)}."
        )

    return df


@lru_cache(maxsize=1)
def load_catalog_map() -> dict[str, dict]:
    """
    Return the catalog as ``{course_id: row_dict}``, cached.

    Used by the path generator for prerequisite lookups, where dict access is the
    natural shape rather than DataFrame filtering.
    """
    df = load_catalog_df()
    return {row["course_id"]: row.to_dict() for _, row in df.iterrows()}


def load_resources_of_type(resource_type: str) -> pd.DataFrame:
    """Return only the catalog rows of one ``resource_type``."""
    if resource_type not in RESOURCE_TYPES:
        raise ValueError(
            f"Unknown resource_type {resource_type!r}; expected one of {list(RESOURCE_TYPES)}"
        )
    df = load_catalog_df()
    return df[df["resource_type"] == resource_type]


@lru_cache(maxsize=1)
def load_skill_vocabulary() -> frozenset[str]:
    """
    Return every distinct lower-cased skill token in the catalog, cached.

    The profiling engine matches user text against this vocabulary. Returned as a
    ``frozenset`` because ``lru_cache`` requires a hashable result and callers
    must not mutate the shared value.
    """
    df = load_catalog_df()
    skills: set[str] = set()
    for raw in df["skills"]:
        for token in str(raw).split(","):
            cleaned = token.strip().lower()
            if cleaned:
                skills.add(cleaned)
    return frozenset(skills)


@lru_cache(maxsize=1)
def load_teachable_skills() -> frozenset[str]:
    """
    Return the skills that some *course* teaches, cached.

    Distinct from ``load_skill_vocabulary()``, which includes skills that only
    appear on projects and assessments. Gap analysis needs this narrower set:
    a gap is only closeable if something actually teaches the skill, and a
    project that merely exercises it does not.
    """
    df = load_resources_of_type("course")
    skills: set[str] = set()
    for raw in df["skills"]:
        for token in str(raw).split(","):
            cleaned = token.strip().lower()
            if cleaned:
                skills.add(cleaned)
    return frozenset(skills)


def parse_prerequisites(raw: object) -> list[str]:
    """
    Parse a ``prerequisites`` cell into a clean list of resource IDs.

    Handles every shape the column has been observed to hold: NaN, the literal
    string "nan", empty string, and comma-separated IDs with stray whitespace.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_skills(raw: object) -> list[str]:
    """Parse a ``skills`` cell into a clean list of lower-cased skill tokens."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip().lower() for part in text.split(",") if part.strip()]


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_catalog_df()

    for column in REQUIRED_COLUMNS:
        assert column in df.columns, f"missing column {column}"

    assert df["prerequisites"].isna().sum() == 0, "prerequisites must be normalised to str"
    assert df["duration_hours"].dtype.kind in "iu", df["duration_hours"].dtype
    assert (df["duration_hours"] > 0).all(), "every resource needs a positive duration"
    assert set(df["resource_type"]) <= set(RESOURCE_TYPES), set(df["resource_type"])

    counts = df["resource_type"].value_counts().to_dict()
    print(f"catalog: {len(df)} resources -> {counts}")

    # The brief requires all three artifact types to be present.
    for resource_type in RESOURCE_TYPES:
        assert counts.get(resource_type, 0) > 0, (
            f"catalog has no {resource_type} rows; the brief asks for courses, "
            f"projects and assessments"
        )

    # Referential integrity across the whole prerequisite graph.
    ids = set(df["course_id"])
    assert len(ids) == len(df), "duplicate course_id values"
    for _, row in df.iterrows():
        for pid in parse_prerequisites(row["prerequisites"]):
            assert pid in ids, f"{row['course_id']} references unknown prerequisite {pid}"

    # Projects and assessments must be gated, or they could be scheduled before
    # anything that teaches their skills.
    gated = df[df["resource_type"].isin(["project", "assessment"])]
    ungated = gated[gated["prerequisites"] == ""]["course_id"].tolist()
    assert not ungated, f"projects/assessments without prerequisites: {ungated}"

    # Caching must return the identical object, not a copy.
    assert load_catalog_df() is load_catalog_df()
    assert load_catalog_map() is load_catalog_map()

    vocab = load_skill_vocabulary()
    teachable = load_teachable_skills()
    assert teachable <= vocab, "teachable skills must be a subset of the full vocabulary"
    print(f"skills: {len(vocab)} total, {len(teachable)} taught by a course")

    assert load_resources_of_type("project")["resource_type"].eq("project").all()
    try:
        load_resources_of_type("nonsense")
        raise AssertionError("expected ValueError for an unknown resource_type")
    except ValueError:
        pass

    assert parse_prerequisites("DS001, DS002") == ["DS001", "DS002"]
    assert parse_prerequisites("nan") == []
    assert parse_prerequisites(None) == []
    assert parse_skills("Python, SQL") == ["python", "sql"]

    print("config.py self-test passed: all assertions OK")
