"""
evaluation.py — Quantitative evaluation of the retrieval and gap machinery.

Why this exists
---------------
Every claim about the ML in this project was previously unfalsifiable: the README
asserted that sentence embeddings suit the task, and nothing measured whether they
beat a far cheaper alternative. This module answers that with numbers, against a
labelled set, with a baseline to compare to.

The baseline is TF-IDF over exactly the same text the embedder sees. That is the
honest comparison — if lexical matching scores the same, the 90 MB model and its
load time are not earning their place, and saying so is more useful than assuming
otherwise.

Ground truth
------------
Two levels, deliberately:

1.  **Category relevance** (all goals). A retrieved resource counts as relevant if
    its catalog category matches the goal's known domain. Derived automatically
    from the catalog, so it needs no hand-labelling and cannot drift out of sync
    with the data.

2.  **Must-have resources** (a subset). For some goals a handful of specific
    resource IDs *ought* to appear — a data-analyst goal that never surfaces
    ``DS005`` (SQL for Data Analysts) has failed regardless of how many Data
    Science courses it returned. This is hand-labelled and stricter, and it is
    what catches a ranker that is directionally right but specifically wrong.

Metrics
-------
    P@k            share of the top k in the correct domain
    MRR            1 / rank of the first correct-domain hit
    must-have      recall of the hand-labelled required resources within top 10
    gate accuracy  correctness of the servable / not-servable decision
    coverage       mean share of the skill gap the produced path closes

Fairness note
-------------
The embedding ranker is evaluated with the difficulty blend switched off, so the
comparison isolates text matching. Blending level fit into one side and not the
other would flatter the embedder for a reason that has nothing to do with
embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import load_catalog_df

# ── Labelled evaluation set ───────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalGoal:
    """A goal with its known domain and, optionally, resources that must surface."""

    text: str
    category: str
    expected_ids: tuple[str, ...] = field(default=())


EVAL_GOALS: tuple[EvalGoal, ...] = (
    # ── Data Science ─────────────────────────────────────────────────────────
    EvalGoal("I want to become a data analyst", "Data Science", ("DS005", "DS006")),
    EvalGoal("I want to learn machine learning with Python", "Data Science", ("DS007",)),
    EvalGoal("teach me how to clean and prepare messy datasets", "Data Science", ("DS003",)),
    EvalGoal("I need to learn SQL for querying databases", "Data Science", ("DS005",)),
    EvalGoal("how do I forecast future sales from historical data", "Data Science", ("DS010",)),
    EvalGoal("I want to build and deploy machine learning models in production",
             "Data Science", ("DS015",)),
    EvalGoal("I want to work with neural networks and deep learning", "Data Science", ("DS014",)),
    EvalGoal("help me understand statistics and hypothesis testing", "Data Science", ("DS002",)),

    # ── Web Development ──────────────────────────────────────────────────────
    EvalGoal("I want to become a front-end web developer", "Web Development", ("WD001", "WD002")),
    EvalGoal("I want to learn React and build single page apps",
             "Web Development", ("WD004",)),
    EvalGoal("how do I build a REST API with Node and Express",
             "Web Development", ("WD005",)),
    EvalGoal("I want to learn HTML and CSS from scratch", "Web Development", ("WD001",)),
    EvalGoal("teach me TypeScript for my JavaScript projects",
             "Web Development", ("WD007",)),
    EvalGoal("I want to make my website faster and score well on Lighthouse",
             "Web Development", ("WD011",)),
    EvalGoal("I need to secure my web app against XSS and CSRF",
             "Web Development", ("WD016",)),

    # ── UX/UI Design ─────────────────────────────────────────────────────────
    EvalGoal("I want to switch careers into UX design", "UX/UI Design", ("UX001",)),
    EvalGoal("I want to learn Figma for interface design", "UX/UI Design", ("UX002",)),
    EvalGoal("how do I run usability tests and user interviews",
             "UX/UI Design", ("UX004",)),
    EvalGoal("I need to make my product accessible and WCAG compliant",
             "UX/UI Design", ("UX009",)),
    EvalGoal("teach me typography colour theory and visual hierarchy",
             "UX/UI Design", ("UX003",)),
    EvalGoal("I want to build a design system with reusable components",
             "UX/UI Design", ("UX007",)),

    # ── Business/Marketing ───────────────────────────────────────────────────
    EvalGoal("I want to learn digital marketing and SEO", "Business/Marketing",
             ("BM001", "BM006")),
    EvalGoal("how do I build a financial model for my startup",
             "Business/Marketing", ("BM009",)),
    EvalGoal("I want to become a product manager", "Business/Marketing", ("BM010",)),
    EvalGoal("teach me to run paid ad campaigns on Google and Meta",
             "Business/Marketing", ("BM012",)),
    EvalGoal("I want to improve conversion rates on my landing pages",
             "Business/Marketing", ("BM008",)),
    EvalGoal("help me start and run an online store", "Business/Marketing", ("BM015",)),

    # ── Cloud/DevOps ─────────────────────────────────────────────────────────
    EvalGoal("I want to get AWS certified and learn cloud architecture",
             "Cloud/DevOps", ("CD005", "CD009")),
    EvalGoal("I want to learn Docker and Kubernetes", "Cloud/DevOps", ("CD003", "CD006")),
    EvalGoal("how do I manage infrastructure as code with Terraform",
             "Cloud/DevOps", ("CD007",)),
    EvalGoal("I need to set up CI/CD pipelines for automated deployment",
             "Cloud/DevOps", ("CD008",)),
    EvalGoal("teach me monitoring logging and observability",
             "Cloud/DevOps", ("CD010",)),
    EvalGoal("I want to learn the Linux command line", "Cloud/DevOps", ("CD002",)),
)

#: Goals this catalog genuinely cannot serve. Used to measure the servability
#: gate, which is only meaningful if it rejects things as well as accepting them.
OFF_CATALOG_GOALS: tuple[str, ...] = (
    "I want to learn medieval Latin palaeography and manuscript restoration",
    "I want to become a professional cellist",
    "teach me to bake sourdough bread",
    "how do I train for a marathon",
    "I want to learn to scuba dive",
    "help me restore a vintage motorcycle engine",
    "asdfgh qwerty zxcvb",
)

#: Goals used for the slower end-to-end pass, one per domain.
END_TO_END_GOALS: tuple[EvalGoal, ...] = (
    EVAL_GOALS[0],   # data analyst
    EVAL_GOALS[8],   # front-end developer
    EVAL_GOALS[15],  # UX career switch
    EVAL_GOALS[21],  # digital marketing
    EVAL_GOALS[27],  # AWS / cloud architecture
)


# ── Rankers ───────────────────────────────────────────────────────────────────

def catalog_text() -> tuple[list[str], list[str], list[str]]:
    """
    Return ``(ids, categories, texts)`` for the whole catalog.

    The text is assembled exactly as ``RecommendationEngine`` assembles it, so the
    baseline is scoring the same input rather than a convenient variant of it.
    """
    df = load_catalog_df()
    texts = (
        df["title"] + ". " + df["description"] + " Skills covered: " + df["skills"]
    ).tolist()
    return df["course_id"].tolist(), df["category"].tolist(), texts


class TfidfRanker:
    """
    Lexical baseline: TF-IDF vectors with cosine similarity.

    Represents what you get without a neural model — bag-of-words with inverse
    document frequency. It should do respectably on goals that reuse catalog
    vocabulary ("learn Docker and Kubernetes") and poorly on goals phrased in
    words the catalog never uses ("I want to become a data analyst" against a
    course titled "SQL for Data Analysts"). Quantifying that difference is the
    point of having it.
    """

    name = "TF-IDF (lexical baseline)"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.ids, self.categories, texts = catalog_text()
        # Unigrams and bigrams, English stopwords removed. sublinear_tf dampens
        # the effect of a term repeating within one long description.
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def rank(self, query: str, top_n: int) -> list[tuple[str, str]]:
        """Return ``(course_id, category)`` for the top matches, best first."""
        from sklearn.metrics.pairwise import linear_kernel

        vector = self._vectorizer.transform([query])
        scores = linear_kernel(vector, self._matrix).ravel()
        order = scores.argsort()[::-1][:top_n]
        return [(self.ids[i], self.categories[i]) for i in order]


class EmbeddingRanker:
    """
    The system's own ranker: ``all-MiniLM-L6-v2`` with cosine similarity.

    Evaluated with the difficulty blend disabled so the comparison measures text
    matching only.
    """

    name = "MiniLM embeddings"

    def __init__(self) -> None:
        from app.recommendation_engine import get_engine

        self._engine = get_engine()

    def rank(self, query: str, top_n: int) -> list[tuple[str, str]]:
        hits = self._engine.recommend(query, top_n=top_n, experience_level=None)
        return [(h["course_id"], h["category"]) for h in hits]


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(ranked: list[tuple[str, str]], category: str, k: int) -> float:
    """Share of the top ``k`` results in the expected domain."""
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for _, cat in top if cat == category) / len(top)


def reciprocal_rank(ranked: list[tuple[str, str]], category: str) -> float:
    """``1 / rank`` of the first correct-domain result, or 0 if none appear."""
    for index, (_, cat) in enumerate(ranked, start=1):
        if cat == category:
            return 1.0 / index
    return 0.0


def must_have_recall(ranked: list[tuple[str, str]], expected_ids: tuple[str, ...]) -> float | None:
    """
    Share of hand-labelled required resources that appear in ``ranked``.

    ``None`` when a goal has no required resources, so goals without labels are
    excluded from the average rather than counted as perfect scores.
    """
    if not expected_ids:
        return None
    found = {cid for cid, _ in ranked}
    return sum(1 for cid in expected_ids if cid in found) / len(expected_ids)


def evaluate_retrieval(
    ranker, goals: tuple[EvalGoal, ...] = EVAL_GOALS, recall_k: int = 10
) -> dict:
    """Run a ranker over the labelled set and return averaged metrics."""
    p1: list[float] = []
    p3: list[float] = []
    p5: list[float] = []
    mrr: list[float] = []
    recalls: list[float] = []
    misses: list[tuple[str, list[str]]] = []

    for goal in goals:
        ranked = ranker.rank(goal.text, max(recall_k, 5))
        p1.append(precision_at_k(ranked, goal.category, 1))
        p3.append(precision_at_k(ranked, goal.category, 3))
        p5.append(precision_at_k(ranked, goal.category, 5))
        mrr.append(reciprocal_rank(ranked, goal.category))

        recall = must_have_recall(ranked[:recall_k], goal.expected_ids)
        if recall is not None:
            recalls.append(recall)
            if recall < 1.0:
                found = {cid for cid, _ in ranked[:recall_k]}
                misses.append((goal.text, [c for c in goal.expected_ids if c not in found]))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "ranker": ranker.name,
        "goals": len(goals),
        "p@1": mean(p1),
        "p@3": mean(p3),
        "p@5": mean(p5),
        "mrr": mean(mrr),
        f"must_have_recall@{recall_k}": mean(recalls),
        "misses": misses,
    }


def evaluate_servability() -> dict:
    """
    Measure the accuracy of the servable / not-servable decision.

    Reported as both halves rather than one accuracy figure, because the two
    failure modes are not equivalent: wrongly rejecting a real goal is a broken
    feature, while wrongly accepting an impossible one produces confident
    nonsense. Which matters more is a product decision, so both are shown.
    """
    from app.recommendation_engine import get_engine
    from app.skill_gap import MIN_GOAL_CONFIDENCE, goal_confidence

    engine = get_engine()

    served_scores = [(g.text, goal_confidence(g.text, engine)) for g in EVAL_GOALS]
    unserved_scores = [(text, goal_confidence(text, engine)) for text in OFF_CATALOG_GOALS]

    accepted = [t for t, s in served_scores if s >= MIN_GOAL_CONFIDENCE]
    rejected = [t for t, s in unserved_scores if s < MIN_GOAL_CONFIDENCE]

    return {
        "threshold": MIN_GOAL_CONFIDENCE,
        "served_total": len(served_scores),
        "served_accepted": len(accepted),
        "unserved_total": len(unserved_scores),
        "unserved_rejected": len(rejected),
        "served_min": min((s for _, s in served_scores), default=0.0),
        "served_mean": (
            sum(s for _, s in served_scores) / len(served_scores) if served_scores else 0.0
        ),
        "unserved_max": max((s for _, s in unserved_scores), default=0.0),
        "false_rejections": [t for t, s in served_scores if s < MIN_GOAL_CONFIDENCE],
        "false_acceptances": [t for t, s in unserved_scores if s >= MIN_GOAL_CONFIDENCE],
    }


def evaluate_end_to_end(goals: tuple[EvalGoal, ...] = END_TO_END_GOALS) -> dict:
    """
    Run the whole pipeline per goal and measure the paths it produces.

    Retrieval metrics say whether the right resources can be found; these say
    whether the assembled path is actually usable — does it close the gap, is it a
    sane length, does it contain all three resource types the brief asks for, and
    is it prerequisite-valid.
    """
    from app.db import _get_connection, init_db
    from app.pipeline import run_full_pipeline

    init_db()
    session = "__evaluation__"

    def _clean() -> None:
        conn = _get_connection()
        with conn:
            for table in ("progress", "feedback", "profiles", "users"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session,))
        conn.close()

    rows: list[dict] = []
    for goal in goals:
        _clean()
        _, result = run_full_pipeline(session, goal.text)
        steps = result.course_steps
        types = {s["course"].get("resource_type") for s in steps}

        positions = {s["course"]["course_id"]: s["step_number"] for s in steps}
        valid = True
        for step in steps:
            cid = step["course"]["course_id"]
            for raw in str(step["course"].get("prerequisites", "")).split(","):
                pid = raw.strip()
                if pid and pid in positions and positions[pid] >= positions[cid]:
                    valid = False

        rows.append(
            {
                "goal": goal.text,
                "servable": result.servable,
                "resources": len(steps),
                "hours": result.total_hours,
                "coverage": result.coverage.get("covered_ratio", 0.0),
                "gap": result.gap.gap_count,
                "targets": result.gap.target_count,
                "types": sorted(types),
                "prereq_valid": valid,
            }
        )
    _clean()

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "rows": rows,
        "mean_coverage": mean("coverage"),
        "mean_resources": mean("resources"),
        "mean_hours": mean("hours"),
        "all_servable": all(r["servable"] for r in rows),
        "all_prereq_valid": all(r["prereq_valid"] for r in rows),
    }


def run_evaluation(include_end_to_end: bool = True) -> dict:
    """Run every evaluation and return one report dict."""
    embedding = evaluate_retrieval(EmbeddingRanker())
    tfidf = evaluate_retrieval(TfidfRanker())
    report = {
        "retrieval": [embedding, tfidf],
        "servability": evaluate_servability(),
    }
    if include_end_to_end:
        report["end_to_end"] = evaluate_end_to_end()
    return report


def format_report(report: dict) -> str:
    """Render a report as plain text, for the CLI and the docs."""
    lines: list[str] = []
    retrieval = report["retrieval"]
    recall_key = next(k for k in retrieval[0] if k.startswith("must_have_recall"))

    lines.append(f"Retrieval quality over {retrieval[0]['goals']} labelled goals")
    lines.append("")
    header = f"  {'ranker':<28}{'P@1':>7}{'P@3':>8}{'P@5':>8}{'MRR':>8}{'must-have':>11}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for result in retrieval:
        lines.append(
            f"  {result['ranker']:<28}{result['p@1']:>7.2f}{result['p@3']:>8.2f}"
            f"{result['p@5']:>8.2f}{result['mrr']:>8.2f}{result[recall_key]:>11.2f}"
        )

    best, other = retrieval[0], retrieval[1]
    delta = best[recall_key] - other[recall_key]
    lines.append("")
    lines.append(
        f"  {best['ranker']} finds {delta:+.0%} more of the hand-labelled required "
        f"resources than the lexical baseline."
    )

    gate = report["servability"]
    lines.append("")
    lines.append(f"Servability gate (threshold {gate['threshold']:.2f})")
    lines.append(
        f"  accepted {gate['served_accepted']}/{gate['served_total']} real goals, "
        f"rejected {gate['unserved_rejected']}/{gate['unserved_total']} impossible ones"
    )
    lines.append(
        f"  real goals scored {gate['served_min']:.2f} at worst "
        f"(mean {gate['served_mean']:.2f}); impossible goals peaked at "
        f"{gate['unserved_max']:.2f}"
    )
    if gate["false_rejections"]:
        lines.append(f"  wrongly rejected: {gate['false_rejections']}")
    if gate["false_acceptances"]:
        lines.append(f"  wrongly accepted: {gate['false_acceptances']}")

    if "end_to_end" in report:
        e2e = report["end_to_end"]
        lines.append("")
        lines.append(f"End-to-end paths over {len(e2e['rows'])} goals")
        lines.append(
            f"  {'goal':<44}{'res':>5}{'hrs':>6}{'cover':>8}{'types':>26}"
        )
        lines.append("  " + "-" * 87)
        for row in e2e["rows"]:
            lines.append(
                f"  {row['goal'][:42]:<44}{row['resources']:>5}{row['hours']:>6}"
                f"{row['coverage']:>7.0%}{','.join(t[:4] for t in row['types']):>26}"
            )
        lines.append(
            f"  mean: {e2e['mean_resources']:.1f} resources, {e2e['mean_hours']:.0f}h, "
            f"{e2e['mean_coverage']:.0%} of gap closed"
        )
        lines.append(
            f"  all servable: {e2e['all_servable']}   "
            f"all prerequisite-valid: {e2e['all_prereq_valid']}"
        )

    return "\n".join(lines)


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── the labelled set must be internally consistent ───────────────────────
    df = load_catalog_df()
    valid_ids = set(df["course_id"])
    valid_categories = set(df["category"])

    for goal in EVAL_GOALS:
        assert goal.category in valid_categories, f"unknown category: {goal.category}"
        for cid in goal.expected_ids:
            assert cid in valid_ids, f"{goal.text!r} expects unknown resource {cid}"
            actual = df.loc[df["course_id"] == cid, "category"].iloc[0]
            assert actual == goal.category, (
                f"{goal.text!r} expects {cid} but that sits in {actual}, "
                f"not {goal.category}"
            )

    per_category: dict[str, int] = {}
    for goal in EVAL_GOALS:
        per_category[goal.category] = per_category.get(goal.category, 0) + 1
    assert len(per_category) == 5, per_category
    assert min(per_category.values()) >= 5, f"thin coverage: {per_category}"
    print(f"labelled set: {len(EVAL_GOALS)} goals across {len(per_category)} domains "
          f"-> {per_category}")
    print(f"off-catalog set: {len(OFF_CATALOG_GOALS)} goals")

    # ── metric helpers behave on hand-built input ────────────────────────────
    sample = [("A", "Data Science"), ("B", "Web Development"), ("C", "Data Science")]
    assert precision_at_k(sample, "Data Science", 1) == 1.0
    assert precision_at_k(sample, "Data Science", 2) == 0.5
    assert abs(precision_at_k(sample, "Data Science", 3) - 2 / 3) < 1e-9
    assert precision_at_k([], "Data Science", 3) == 0.0
    assert reciprocal_rank(sample, "Web Development") == 0.5
    assert reciprocal_rank(sample, "Data Science") == 1.0
    assert reciprocal_rank(sample, "UX/UI Design") == 0.0
    assert must_have_recall(sample, ("A", "C")) == 1.0
    assert must_have_recall(sample, ("A", "Z")) == 0.5
    assert must_have_recall(sample, ()) is None, (
        "unlabelled goals must be excluded, not scored as perfect"
    )
    print("metric helpers: 11 assertions OK")

    # ── the comparison itself ────────────────────────────────────────────────
    print()
    report = run_evaluation()
    print(format_report(report))

    embedding, tfidf = report["retrieval"]
    recall_key = next(k for k in embedding if k.startswith("must_have_recall"))

    # Both rankers must beat chance. With 5 roughly equal domains, chance P@1 ~ 0.2.
    for result in (embedding, tfidf):
        assert result["p@1"] > 0.2, f"{result['ranker']} is at or below chance"
        assert result["mrr"] > 0.2, result

    # The headline claim: embeddings must actually earn their cost here.
    assert embedding[recall_key] >= tfidf[recall_key], (
        f"embeddings ({embedding[recall_key]:.2f}) did not beat TF-IDF "
        f"({tfidf[recall_key]:.2f}) on required-resource recall. The model is not "
        f"paying for itself and the README claim would be false."
    )

    # ── the servability gate must work in both directions ────────────────────
    gate = report["servability"]
    assert gate["served_accepted"] == gate["served_total"], (
        f"wrongly rejected real goals: {gate['false_rejections']}"
    )
    assert gate["unserved_rejected"] == gate["unserved_total"], (
        f"wrongly accepted impossible goals: {gate['false_acceptances']}"
    )
    assert gate["unserved_max"] < gate["served_min"], (
        f"served and unserved score ranges overlap "
        f"({gate['unserved_max']:.3f} >= {gate['served_min']:.3f}), so no single "
        f"threshold can separate them"
    )

    # The threshold must sit clear of both sides, not merely on the correct side of
    # the boundary. At 0.42 it cleared the worst real goal by 0.01, which is one
    # unlucky phrasing away from rejecting a goal the catalog covers well.
    lower_margin = gate["served_min"] - gate["threshold"]
    upper_margin = gate["threshold"] - gate["unserved_max"]
    print(
        f"\ngate margins: {lower_margin:+.3f} below the worst real goal, "
        f"{upper_margin:+.3f} above the best impossible one"
    )
    assert lower_margin >= 0.03, (
        f"only {lower_margin:.3f} of margin above the worst real goal "
        f"({gate['served_min']:.3f} vs threshold {gate['threshold']:.3f}) — too tight"
    )
    assert upper_margin >= 0.03, (
        f"only {upper_margin:.3f} of margin below the best impossible goal "
        f"({gate['unserved_max']:.3f} vs threshold {gate['threshold']:.3f}) — too tight"
    )

    # ── end-to-end paths must be usable, not merely produced ────────────────
    e2e = report["end_to_end"]
    assert e2e["all_servable"], [r["goal"] for r in e2e["rows"] if not r["servable"]]
    assert e2e["all_prereq_valid"], "a produced path violated prerequisite order"
    assert e2e["mean_coverage"] > 0.6, e2e["mean_coverage"]
    for row in e2e["rows"]:
        assert 1 <= row["resources"] <= 20, row
        assert row["hours"] > 0, row

    # The brief asks for a roadmap of courses, projects and assessments, so a path
    # should generally contain more than one type.
    typed = sum(1 for r in e2e["rows"] if len(r["types"]) > 1)
    assert typed >= len(e2e["rows"]) // 2, (
        f"only {typed}/{len(e2e['rows'])} paths mixed resource types"
    )
    print(f"\n{typed}/{len(e2e['rows'])} paths contained more than one resource type")

    print("\nevaluation.py self-test passed: all assertions OK")
