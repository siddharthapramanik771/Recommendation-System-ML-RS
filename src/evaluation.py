from dataclasses import dataclass
from math import log2

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.matrix_factorization import MatrixFactorizationRecommender


@dataclass(frozen=True)
class RankingEvaluation:
    evaluated_users: int
    precision_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    hit_rate_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]
    catalog_coverage_at_k: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "evaluated_users": self.evaluated_users,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "catalog_coverage_at_k": self.catalog_coverage_at_k,
        }


def evaluate_ranking(
    recommender: MatrixFactorizationRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_movie_ids: list[int],
    k_values: tuple[int, ...],
    config: RuntimeConfig = RUNTIME_CONFIG,
) -> RankingEvaluation:
    if test.empty:
        empty_scores = {str(k): 0.0 for k in k_values}
        return RankingEvaluation(0, empty_scores, empty_scores, empty_scores, empty_scores, empty_scores)

    train_seen = _user_movie_sets(train, config)
    test_relevant = _user_movie_sets(
        test[test[config.rating_column] >= config.relevance_threshold], config
    )
    max_k = max(k_values)

    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    hit_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    coverage_sets = {k: set() for k in k_values}
    evaluated_users = 0

    for user_id, relevant_items in test_relevant.items():
        if not relevant_items:
            continue
        seen_items = train_seen.get(user_id, set())
        recommendations = recommender.recommend_for_user(
            user_id=user_id,
            candidate_movie_ids=all_movie_ids,
            seen_movie_ids=seen_items,
            k=max_k,
        )
        recommended_movie_ids = [movie_id for movie_id, _ in recommendations]
        if not recommended_movie_ids:
            continue

        evaluated_users += 1
        for k in k_values:
            top_k = recommended_movie_ids[:k]
            hits = [movie_id for movie_id in top_k if movie_id in relevant_items]
            precision_sums[k] += len(hits) / k
            recall_sums[k] += len(hits) / len(relevant_items)
            hit_sums[k] += float(bool(hits))
            ndcg_sums[k] += _ndcg(top_k, relevant_items, k)
            coverage_sets[k].update(top_k)

    if evaluated_users == 0:
        empty_scores = {str(k): 0.0 for k in k_values}
        return RankingEvaluation(0, empty_scores, empty_scores, empty_scores, empty_scores, empty_scores)

    catalog_size = max(len(set(all_movie_ids)), 1)
    return RankingEvaluation(
        evaluated_users=evaluated_users,
        precision_at_k={
            str(k): precision_sums[k] / evaluated_users for k in k_values
        },
        recall_at_k={str(k): recall_sums[k] / evaluated_users for k in k_values},
        hit_rate_at_k={str(k): hit_sums[k] / evaluated_users for k in k_values},
        ndcg_at_k={str(k): ndcg_sums[k] / evaluated_users for k in k_values},
        catalog_coverage_at_k={
            str(k): len(coverage_sets[k]) / catalog_size for k in k_values
        },
    )


def _user_movie_sets(
    frame: pd.DataFrame, config: RuntimeConfig
) -> dict[int, set[int]]:
    return {
        int(user_id): set(group[config.item_column].astype(int).tolist())
        for user_id, group in frame.groupby(config.user_column)
    }


def _ndcg(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg

