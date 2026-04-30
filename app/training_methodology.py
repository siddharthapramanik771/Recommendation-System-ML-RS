import json

import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig


class TrainingMethodologyRenderer:
    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config

    def render(self) -> None:
        st.subheader("Training Methodology")
        metrics = self.load_metrics()
        if metrics is None:
            st.info("Training metrics are not available yet.")
            st.code("python -m src.train", language="bash")
            return

        self.render_metric_strip(metrics)
        st.markdown("#### Matrix Factorization")
        st.write(
            "The model learns a low-dimensional vector for each user and movie, "
            "plus user and movie bias terms. Ratings are observed sparsely, so "
            "training only iterates over known user-movie interactions."
        )
        st.code(
            "predicted_rating = global_mean + user_bias + movie_bias "
            "+ dot(user_factors, movie_factors)",
            language="text",
        )

        st.markdown("#### Precision@K Evaluation")
        st.write(
            "Evaluation uses a leave-one-relevant-rating-out split per eligible "
            "user. The held-out relevant movie is hidden during training, then "
            "the recommender ranks unseen candidates. Precision@K measures the "
            "fraction of top-K slots that recover held-out relevant movies."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("#### Training Settings")
            st.json(metrics.get("settings", {}))
        with right:
            st.markdown("#### Catalog Stats")
            st.json(metrics.get("catalog", {}))

    def render_metric_strip(self, metrics: dict) -> None:
        ranking = metrics.get("ranking", {})
        precision = ranking.get("precision_at_k", {})
        recall = ranking.get("recall_at_k", {})
        hit_rate = ranking.get("hit_rate_at_k", {})
        coverage = ranking.get("catalog_coverage_at_k", {})
        preferred_k = "10" if "10" in precision else next(iter(precision), "n/a")

        left, middle, right, extra = st.columns(4)
        left.metric(f"Precision@{preferred_k}", _format_metric(precision, preferred_k))
        middle.metric(f"Recall@{preferred_k}", _format_metric(recall, preferred_k))
        right.metric(f"Hit Rate@{preferred_k}", _format_metric(hit_rate, preferred_k))
        extra.metric(f"Coverage@{preferred_k}", _format_metric(coverage, preferred_k))
        st.caption(f"Evaluated users: {ranking.get('evaluated_users', 0):,}")

    def load_metrics(self) -> dict | None:
        if not self.config.metrics_path.exists():
            return None
        try:
            return json.loads(self.config.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def _format_metric(payload: dict, key: str) -> str:
    value = payload.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"

