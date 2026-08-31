"""
eval_view.py — Streamlit view for the evaluation results.

The brief states the product must *visibly* demonstrate all six judged areas. A
README claiming that sentence embeddings suit the task is not visible, and is not
evidence. This tab runs the real evaluation and shows the numbers, including the
baseline that could have beaten it.

Presentation only: every figure comes from ``app.evaluation``, which has no
Streamlit dependency and is separately testable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _cached_report() -> dict:
    """
    Run the evaluation once per process and cache it.

    It embeds ~40 queries and runs five full pipelines, so a few seconds. Caching
    keeps tab switching instant without hiding the cost on first view.
    """
    from app.evaluation import run_evaluation

    return run_evaluation(include_end_to_end=True)


def render_evaluation() -> None:
    """Render the evaluation tab."""
    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#1b2a4a,#2d1b4a);
                    border-radius:16px;padding:24px 28px 16px;margin-bottom:24px;'>
            <h2 style='color:#e0e0ff;margin:0 0 6px;font-size:1.5rem;'>
                🔬 How It Performs
            </h2>
            <p style='color:#a0a0c0;margin:0;font-size:0.95rem;'>
                Measured, not asserted. Every number below is computed live against
                a labelled goal set, with a cheaper baseline for comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Running the evaluation..."):
        report = _cached_report()

    embedding, tfidf = report["retrieval"]
    recall_key = next(k for k in embedding if k.startswith("must_have_recall"))

    # ── Retrieval comparison ─────────────────────────────────────────────────
    st.markdown("### Does the embedding model earn its place?")
    st.caption(
        f"Both rankers score the same {embedding['goals']} labelled goals over the "
        f"same catalog text. A result counts as relevant when it sits in the goal's "
        f"known domain. **Must-have** is stricter: the share of hand-labelled "
        f"resources that a goal genuinely ought to surface, found within the top 10."
    )

    table = pd.DataFrame(
        [
            {
                "Ranker": result["ranker"],
                "P@1": round(result["p@1"], 2),
                "P@3": round(result["p@3"], 2),
                "P@5": round(result["p@5"], 2),
                "MRR": round(result["mrr"], 2),
                "Must-have recall": round(result[recall_key], 2),
            }
            for result in (embedding, tfidf)
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    delta_recall = embedding[recall_key] - tfidf[recall_key]
    delta_p5 = embedding["p@5"] - tfidf["p@5"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Top result in right domain", f"{embedding['p@1']:.0%}")
    col2.metric(
        "Required resources found",
        f"{embedding[recall_key]:.0%}",
        delta=f"{delta_recall:+.0%} vs TF-IDF",
    )
    col3.metric(
        "Precision@5",
        f"{embedding['p@5']:.0%}",
        # Expressed in percentage points, so a 0.13 difference reads as "+13 pts"
        # rather than being rounded to "+0".
        delta=f"{delta_p5 * 100:+.0f} pts vs TF-IDF" if delta_p5 else None,
    )

    if delta_recall <= 0.02 and delta_p5 <= 0.05:
        st.warning(
            "The lexical baseline is close. This catalog phrases its titles and "
            "descriptions much like the goals people type, which is exactly the "
            "situation where TF-IDF competes well. The embedding model's advantage "
            "would widen on a catalog whose wording diverges from how learners "
            "describe their goals."
        )
    else:
        st.success(
            f"Embeddings beat the lexical baseline by {delta_recall:+.0%} on "
            f"required-resource recall and {delta_p5:+.2f} on precision@5. The gap "
            f"comes from goals phrased in words the catalog never uses — "
            f"\"become a data analyst\" has no lexical overlap with \"SQL for Data "
            f"Analysts\", but the two are close in embedding space."
        )
    st.divider()

    # ── Servability gate ─────────────────────────────────────────────────────
    gate = report["servability"]
    st.markdown("### Does it know what it can't do?")
    st.caption(
        "Ranking always returns *something*, so a recommender with no floor answers "
        "a request about medieval manuscripts with five cloud courses and a "
        "confident percentage. This measures the check that prevents that."
    )

    gcol1, gcol2, gcol3 = st.columns(3)
    gcol1.metric(
        "Real goals accepted",
        f"{gate['served_accepted']}/{gate['served_total']}",
    )
    gcol2.metric(
        "Impossible goals rejected",
        f"{gate['unserved_rejected']}/{gate['unserved_total']}",
    )
    gcol3.metric("Threshold", f"{gate['threshold']:.2f}")

    lower = gate["served_min"] - gate["threshold"]
    upper = gate["threshold"] - gate["unserved_max"]
    st.markdown(
        f"- Real goals score **{gate['served_min']:.2f}** at worst, "
        f"**{gate['served_mean']:.2f}** on average.\n"
        f"- Goals this catalog cannot serve peak at **{gate['unserved_max']:.2f}**.\n"
        f"- The threshold sits **{lower:+.2f}** below the worst real goal and "
        f"**{upper:+.2f}** above the best impossible one, so neither side is "
        f"decided by a hair."
    )
    if gate["false_rejections"]:
        st.error(f"Wrongly rejected: {gate['false_rejections']}")
    if gate["false_acceptances"]:
        st.error(f"Wrongly accepted: {gate['false_acceptances']}")
    st.divider()

    # ── End-to-end path quality ──────────────────────────────────────────────
    if "end_to_end" in report:
        e2e = report["end_to_end"]
        st.markdown("### Are the paths it builds actually usable?")
        st.caption(
            "Retrieval metrics only say the right resources can be found. These run "
            "the full pipeline and check the assembled path: does it close the skill "
            "gap, is it a sane length, does it mix courses with projects and "
            "assessments, and is every prerequisite ordered correctly."
        )

        ecol1, ecol2, ecol3 = st.columns(3)
        ecol1.metric("Mean gap closed", f"{e2e['mean_coverage']:.0%}")
        ecol2.metric("Mean path length", f"{e2e['mean_resources']:.1f} resources")
        ecol3.metric("Mean study time", f"{e2e['mean_hours']:.0f}h")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Goal": row["goal"],
                        "Resources": row["resources"],
                        "Hours": row["hours"],
                        "Gap closed": f"{row['coverage']:.0%}",
                        "Skills still missing": f"{row['gap']}/{row['targets']}",
                        "Types included": ", ".join(row["types"]),
                        "Prereqs valid": "yes" if row["prereq_valid"] else "NO",
                    }
                    for row in e2e["rows"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        if e2e["all_prereq_valid"]:
            st.success(
                "Every path respects prerequisite order: no resource is ever "
                "scheduled before something it depends on."
            )
        else:
            st.error("A generated path violated prerequisite order.")
    st.divider()

    # ── Honest limitations ───────────────────────────────────────────────────
    st.markdown("### What these numbers don't tell you")
    st.markdown(
        """
        - **Domain relevance is a proxy.** "In the right category" is automatic to
          compute but generous: a Data Science goal that returns five unhelpful Data
          Science courses still scores 1.00 on P@5. The hand-labelled must-have
          recall exists to counter that, and it covers only part of the set.
        - **The labelled set is ours.** 33 goals written alongside the catalog they
          are tested against. Real learners phrase things in ways nobody anticipated,
          and no offline set captures that.
        - **The catalog is synthetic.** 129 generated resources with clean, complete
          prerequisite data. Real catalogs have gaps, duplicates and inconsistent
          skill labels, all of which would hurt these figures.
        - **Nothing here measures learning.** Whether a path actually gets someone to
          their goal needs longitudinal data on real learners, which a prototype
          cannot produce.
        """
    )

    with st.expander("Full text report"):
        from app.evaluation import format_report

        st.code(format_report(report), language="text")
