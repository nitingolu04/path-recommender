"""
recommendation_engine.py — Core ML component: semantic retrieval + scoring.

Pipeline
--------
1.  At instantiation the engine loads the course catalog and embeds every
    course exactly once with ``all-MiniLM-L6-v2``.  The resulting (N x 384)
    matrix is L2-normalised and cached in memory as ``self._course_embeddings``.

2.  ``recommend()`` embeds the caller's query, then scores all N courses in a
    single numpy matrix multiply — no Python loop, no per-request model call on
    the catalog side.

3.  Courses the user rejected are removed before ranking.

Why all-MiniLM-L6-v2?
    - Fast on CPU (the full 84-course catalog embeds in well under a second)
    - 384-dimensional output: a good quality/speed trade-off for short text
    - Trained with a cosine-similarity objective, so cosine is the metric the
      embedding space was actually optimised for
    - Fully local, no API key

Why cosine similarity?
    - Scale-invariant, so a long course description is not penalised against a
      short one purely for having a larger vector norm
    - The standard metric for sentence-embedding spaces
    - On L2-normalised vectors it reduces to a dot product, which lets the whole
      catalog be scored with one ``@`` operation

Scoring model
-------------
The returned ``similarity_score`` is a **bounded composite**, not raw cosine:

    score = (1 - LEVEL_WEIGHT) * semantic + LEVEL_WEIGHT * level_fit

where both ``semantic`` and ``level_fit`` lie in [0, 1], so ``score`` is
guaranteed to lie in [0, 1] as well.

This replaced an earlier approach that multiplied cosine similarity by 1.15 for
courses at or below the user's level.  That was wrong in two ways: it pushed
scores above 1.0 (the UI rendered a "115% match"), and multiplying a similarity
metric by a magic constant has no principled interpretation.  A convex blend of
two normalised signals is bounded by construction and each term can be shown to
the user separately, which is why ``semantic_score`` and ``level_fit`` are
returned alongside the composite.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import DIFFICULTY_LEVELS, DIFFICULTY_ORDER, load_catalog_df

# ── Singleton cache ───────────────────────────────────────────────────────────
_ENGINE_INSTANCE: "RecommendationEngine | None" = None


def get_engine() -> "RecommendationEngine":
    """
    Return the shared ``RecommendationEngine`` singleton, creating it on first
    call.  A singleton matters here because Streamlit re-executes the script on
    every interaction; without it the model would reload and the catalog would
    be re-embedded on every click.
    """
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = RecommendationEngine()
    return _ENGINE_INSTANCE


def effective_experience_level(base_level: str, too_easy_count: int, step: int = 2) -> str:
    """
    Escalate the user's effective level based on repeated "too easy" feedback.

    Every ``step`` "too easy" reports promote the user one difficulty tier,
    capped at ``advanced``.  Without this, the two feedback buttons were
    indistinguishable: both merely deleted a course, and the user's level never
    moved no matter how many times they said the material was beneath them.

    Parameters
    ----------
    base_level : str
        Level inferred from the user's own description of themselves.
    too_easy_count : int
        Number of ``too_easy`` feedback events logged for the session.
    step : int
        How many reports constitute one promotion.

    Returns
    -------
    str
        One of ``'beginner' | 'intermediate' | 'advanced'``.

    Examples
    --------
    >>> effective_experience_level("beginner", 0)
    'beginner'
    >>> effective_experience_level("beginner", 2)
    'intermediate'
    >>> effective_experience_level("beginner", 4)
    'advanced'
    >>> effective_experience_level("intermediate", 2)
    'advanced'
    """
    base_rank = DIFFICULTY_ORDER.get(str(base_level).lower(), 0)
    promotions = max(0, too_easy_count) // max(1, step)
    new_rank = min(base_rank + promotions, len(DIFFICULTY_LEVELS) - 1)
    return DIFFICULTY_LEVELS[new_rank]


class RecommendationEngine:
    """
    Semantic course recommendation engine backed by sentence-transformers.

    Attributes
    ----------
    courses : pd.DataFrame
        The full course catalog.
    model : SentenceTransformer
        The embedding model (all-MiniLM-L6-v2).
    _course_embeddings : np.ndarray, shape (N, 384)
        L2-normalised course embeddings, computed once at construction.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    #: Weight given to difficulty fit in the composite score.  The remaining
    #: 0.75 goes to semantic similarity, keeping meaning the dominant signal.
    #:
    #: This started at 0.15 and was raised after testing: a five-year developer
    #: searching for MLOps still had "Introduction to Cloud Computing" ranked
    #: first, because a strong semantic match swamped the level term entirely.
    LEVEL_WEIGHT = 0.25

    #: level_fit as a function of (course_rank - user_rank).
    #:
    #: Courses at the user's level score best.  One tier below retains real
    #: value as revision, two tiers below is mostly redundant.  Upward, one tier
    #: is a reasonable stretch worth surfacing, two tiers is likely unusable
    #: without the intervening material.
    _LEVEL_FIT = {0: 1.00, -1: 0.60, -2: 0.20, 1: 0.55, 2: 0.20}
    _LEVEL_FIT_DEFAULT = 0.20

    def __init__(self) -> None:
        print(f"[RecommendationEngine] Loading model '{self.MODEL_NAME}' ...")
        self.model = SentenceTransformer(self.MODEL_NAME)

        self.courses = load_catalog_df()

        # Embed title + description together so both contribute to the match.
        # Skills are appended because they carry high-signal keywords ("pandas",
        # "figma") that a prose description sometimes only implies.
        texts = (
            self.courses["title"]
            + ". "
            + self.courses["description"]
            + " Skills covered: "
            + self.courses["skills"]
        ).tolist()

        raw = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self._course_embeddings = self._l2_normalise(raw)

        # Pre-compute difficulty ranks once so scoring stays vectorised.
        self._course_level_ranks = np.array(
            [DIFFICULTY_ORDER.get(str(lv).lower(), 0) for lv in self.courses["difficulty_level"]],
            dtype=np.int8,
        )

        print(
            f"[RecommendationEngine] Cached {len(texts)} course embeddings "
            f"(shape {self._course_embeddings.shape})."
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        query: str,
        top_n: int = 20,
        exclude_ids: set[str] | None = None,
        experience_level: str | None = None,
        resource_types: tuple[str, ...] | None = None,
        min_semantic: float = 0.0,
    ) -> list[dict]:
        """
        Return the top-N best-matching resources for ``query``.

        Parameters
        ----------
        query : str
            Free-text goal, e.g. "I want to become a data analyst".
        top_n : int
            Maximum number of resources to return.
        exclude_ids : set[str] | None
            Resource IDs to drop before ranking (rejected or already completed).
        experience_level : str | None
            When given, difficulty fit is blended into the score.  When
            ``None``, the score is pure semantic similarity.
        resource_types : tuple[str, ...] | None
            Restrict results to these ``resource_type`` values.  Gap analysis
            passes ``("course",)`` because only courses teach skills — a project
            that merely exercises a skill cannot close a gap.
        min_semantic : float
            Drop resources whose raw cosine falls below this floor.  Without one,
            ``top_n`` is filled regardless of quality, which is how cloud courses
            at 0.43 ended up padding a data-analyst path.

        Returns
        -------
        list[dict]
            Resource rows, each with three added keys: ``similarity_score``
            (composite, 0-1), ``semantic_score`` (raw cosine clamped to 0-1) and
            ``level_fit`` (0-1, or ``None`` when no experience level was given).
        """
        exclude_ids = exclude_ids or set()

        query_vec = self._l2_normalise(self.model.encode([query], convert_to_numpy=True))

        # Cosine similarity == dot product, since both sides are L2-normalised.
        semantic = (self._course_embeddings @ query_vec.T).ravel()

        # Cosine is mathematically in [-1, 1]; clamp to [0, 1] because a
        # negative match is not meaningfully "worse than unrelated" here, and a
        # negative term would break the bounds of the composite below.
        semantic = np.clip(semantic, 0.0, 1.0)

        if experience_level:
            level_fit = self._level_fit_vector(experience_level)
            composite = (1.0 - self.LEVEL_WEIGHT) * semantic + self.LEVEL_WEIGHT * level_fit
        else:
            level_fit = None
            composite = semantic

        # Defensive clamp: guarantees the invariant the UI depends on even if
        # the weights above are ever edited inconsistently.
        composite = np.clip(composite, 0.0, 1.0)

        df = self.courses.copy()
        df["similarity_score"] = composite
        df["semantic_score"] = semantic
        df["level_fit"] = level_fit if level_fit is not None else None

        if exclude_ids:
            df = df[~df["course_id"].isin(exclude_ids)]
        if resource_types:
            df = df[df["resource_type"].isin(resource_types)]
        if min_semantic > 0.0:
            df = df[df["semantic_score"] >= min_semantic]

        df = df.sort_values("similarity_score", ascending=False, kind="mergesort")
        return df.head(top_n).to_dict(orient="records")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed arbitrary strings and return L2-normalised vectors, shape
        (len(texts), 384).  Used by the profiling engine to classify the user's
        domain against category-label embeddings.
        """
        raw = self.model.encode(list(texts), convert_to_numpy=True)
        return self._l2_normalise(raw)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalise(vecs: np.ndarray) -> np.ndarray:
        """Divide each row by its L2 norm, guarding against zero vectors."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def _level_fit_vector(self, experience_level: str) -> np.ndarray:
        """
        Return a per-course ``level_fit`` array in [0, 1] for the given user level.

        Vectorised over the pre-computed difficulty ranks, so this costs one
        array subtraction plus a lookup rather than a Python loop over courses.
        """
        user_rank = DIFFICULTY_ORDER.get(str(experience_level).lower(), 0)
        deltas = self._course_level_ranks.astype(np.int16) - user_rank
        return np.array(
            [self._LEVEL_FIT.get(int(d), self._LEVEL_FIT_DEFAULT) for d in deltas],
            dtype=np.float64,
        )


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── level escalation is pure logic; check it before loading the model ─────
    assert effective_experience_level("beginner", 0) == "beginner"
    assert effective_experience_level("beginner", 1) == "beginner"
    assert effective_experience_level("beginner", 2) == "intermediate"
    assert effective_experience_level("beginner", 4) == "advanced"
    assert effective_experience_level("beginner", 99) == "advanced", "must cap at advanced"
    assert effective_experience_level("intermediate", 2) == "advanced"
    assert effective_experience_level("advanced", 10) == "advanced"

    engine = RecommendationEngine()

    results = engine.recommend(
        "I want to become a data analyst, I know some Excel",
        top_n=5,
        experience_level="beginner",
    )
    assert len(results) == 5, f"expected 5 results, got {len(results)}"

    print("\n-- Top 5 for 'become a data analyst, know some Excel' (beginner) --")
    for r in results:
        print(
            f"  score={r['similarity_score']:.3f}  "
            f"(semantic={r['semantic_score']:.3f} level_fit={r['level_fit']:.2f})  "
            f"{r['course_id']}  {r['title']}  [{r['difficulty_level']}]"
        )

    # Results must be sorted descending.
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "results must be ranked descending"

    # ── REGRESSION: the composite score must never exceed 1.0 ────────────────
    # The old x1.15 multiplier produced 1.15 here, which the UI rendered as
    # "Match 115%".  Feeding a course's own text back in is the worst case,
    # because cosine against itself is ~1.0.
    df = load_catalog_df()
    row = df[df["course_id"] == "DS006"].iloc[0]
    self_query = f"{row['title']}. {row['description']} Skills covered: {row['skills']}"

    for level in (None, "beginner", "intermediate", "advanced"):
        hits = engine.recommend(self_query, top_n=10, experience_level=level)
        worst = max(h["similarity_score"] for h in hits)
        assert 0.0 <= worst <= 1.0, f"score {worst} out of bounds for level={level}"
        for h in hits:
            assert 0.0 <= h["semantic_score"] <= 1.0, h["semantic_score"]

    top_self = engine.recommend(self_query, top_n=1, experience_level="beginner")[0]
    print(
        f"\n  near-identical-text query -> score {top_self['similarity_score']:.4f} "
        f"(displays as {top_self['similarity_score']:.0%}), semantic "
        f"{top_self['semantic_score']:.4f}"
    )
    assert top_self["course_id"] == "DS006", "a course's own text should rank it first"
    assert top_self["similarity_score"] <= 1.0, "REGRESSION: score exceeded 100%"

    # ── exclusions are honoured ───────────────────────────────────────────────
    excluded = {"DS006"}
    hits = engine.recommend(self_query, top_n=5, exclude_ids=excluded, experience_level="beginner")
    assert all(h["course_id"] not in excluded for h in hits), "exclude_ids was ignored"

    # ── level fit actually shifts the ranking ─────────────────────────────────
    beginner_hits = engine.recommend("learn cloud computing", top_n=15, experience_level="beginner")
    advanced_hits = engine.recommend("learn cloud computing", top_n=15, experience_level="advanced")
    n_beginner_in_beginner = sum(1 for h in beginner_hits[:5] if h["difficulty_level"] == "beginner")
    n_beginner_in_advanced = sum(1 for h in advanced_hits[:5] if h["difficulty_level"] == "beginner")
    assert n_beginner_in_beginner >= n_beginner_in_advanced, (
        "a beginner should not see fewer beginner courses than an advanced user"
    )

    print("\nrecommendation_engine.py self-test passed: all assertions OK")
