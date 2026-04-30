from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.training_settings import TrainingSettings


@dataclass
class MatrixFactorizationRecommender:
    """Biased matrix factorization trained with stochastic gradient descent."""

    user_id_to_index: dict[int, int]
    movie_id_to_index: dict[int, int]
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    global_mean: float
    movie_means: dict[int, float]
    movie_counts: dict[int, int]
    rating_min: float
    rating_max: float

    @classmethod
    def fit(
        cls,
        ratings: pd.DataFrame,
        settings: TrainingSettings,
        config: RuntimeConfig = RUNTIME_CONFIG,
    ) -> "MatrixFactorizationRecommender":
        if ratings.empty:
            raise ValueError("Cannot train matrix factorization on an empty ratings set.")

        user_ids = sorted(ratings[config.user_column].astype(int).unique().tolist())
        movie_ids = sorted(ratings[config.item_column].astype(int).unique().tolist())
        user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

        user_indices = ratings[config.user_column].map(user_id_to_index).to_numpy()
        movie_indices = ratings[config.item_column].map(movie_id_to_index).to_numpy()
        rating_values = ratings[config.rating_column].astype(float).to_numpy()

        rng = np.random.default_rng(settings.random_state)
        user_factors = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(len(user_ids), settings.latent_factors),
        )
        item_factors = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(len(movie_ids), settings.latent_factors),
        )
        user_bias = np.zeros(len(user_ids), dtype=float)
        item_bias = np.zeros(len(movie_ids), dtype=float)
        global_mean = float(np.mean(rating_values))

        order = np.arange(len(rating_values))
        for _ in range(settings.epochs):
            rng.shuffle(order)
            for position in order:
                user_index = user_indices[position]
                movie_index = movie_indices[position]
                rating = rating_values[position]

                prediction = (
                    global_mean
                    + user_bias[user_index]
                    + item_bias[movie_index]
                    + float(user_factors[user_index] @ item_factors[movie_index])
                )
                error = rating - prediction

                current_user_factors = user_factors[user_index].copy()
                current_item_factors = item_factors[movie_index].copy()

                user_bias[user_index] += settings.learning_rate * (
                    error - settings.regularization * user_bias[user_index]
                )
                item_bias[movie_index] += settings.learning_rate * (
                    error - settings.regularization * item_bias[movie_index]
                )
                user_factors[user_index] += settings.learning_rate * (
                    error * current_item_factors
                    - settings.regularization * current_user_factors
                )
                item_factors[movie_index] += settings.learning_rate * (
                    error * current_user_factors
                    - settings.regularization * current_item_factors
                )

        movie_means = (
            ratings.groupby(config.item_column)[config.rating_column]
            .mean()
            .astype(float)
            .to_dict()
        )
        movie_counts = (
            ratings.groupby(config.item_column)[config.rating_column]
            .count()
            .astype(int)
            .to_dict()
        )

        return cls(
            user_id_to_index=user_id_to_index,
            movie_id_to_index=movie_id_to_index,
            user_factors=user_factors,
            item_factors=item_factors,
            user_bias=user_bias,
            item_bias=item_bias,
            global_mean=global_mean,
            movie_means={int(key): float(value) for key, value in movie_means.items()},
            movie_counts={int(key): int(value) for key, value in movie_counts.items()},
            rating_min=config.rating_min,
            rating_max=config.rating_max,
        )

    @property
    def known_user_ids(self) -> list[int]:
        return sorted(self.user_id_to_index)

    @property
    def known_movie_ids(self) -> list[int]:
        return sorted(self.movie_id_to_index)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        score = self.global_mean
        user_index = self.user_id_to_index.get(int(user_id))
        movie_index = self.movie_id_to_index.get(int(movie_id))

        if user_index is not None:
            score += float(self.user_bias[user_index])
        if movie_index is not None:
            score += float(self.item_bias[movie_index])
        if user_index is not None and movie_index is not None:
            score += float(self.user_factors[user_index] @ self.item_factors[movie_index])
        if user_index is None and movie_index is not None:
            score = self._popularity_score(int(movie_id))

        return float(np.clip(score, self.rating_min, self.rating_max))

    def score_items_for_user(
        self, user_id: int, movie_ids: Iterable[int]
    ) -> list[tuple[int, float]]:
        movie_ids = [int(movie_id) for movie_id in movie_ids]
        user_index = self.user_id_to_index.get(int(user_id))
        scores: list[tuple[int, float]] = []

        if user_index is None:
            return [
                (movie_id, self._popularity_score(movie_id)) for movie_id in movie_ids
            ]

        base = self.global_mean + float(self.user_bias[user_index])
        user_vector = self.user_factors[user_index]
        for movie_id in movie_ids:
            movie_index = self.movie_id_to_index.get(movie_id)
            if movie_index is None:
                score = self._popularity_score(movie_id)
            else:
                score = (
                    base
                    + float(self.item_bias[movie_index])
                    + float(user_vector @ self.item_factors[movie_index])
                )
            scores.append((movie_id, float(np.clip(score, self.rating_min, self.rating_max))))
        return scores

    def recommend_for_user(
        self,
        user_id: int,
        candidate_movie_ids: Iterable[int],
        seen_movie_ids: Iterable[int] | None = None,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        seen = set(int(movie_id) for movie_id in (seen_movie_ids or []))
        candidates = [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) not in seen]
        scores = self.score_items_for_user(user_id, candidates)
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def similar_items(
        self, movie_id: int, candidate_movie_ids: Iterable[int], k: int = 10
    ) -> list[tuple[int, float]]:
        movie_index = self.movie_id_to_index.get(int(movie_id))
        if movie_index is None:
            return []

        source_vector = self.item_factors[movie_index]
        source_norm = np.linalg.norm(source_vector)
        if source_norm == 0:
            return []

        similarities: list[tuple[int, float]] = []
        for candidate_id in candidate_movie_ids:
            candidate_id = int(candidate_id)
            if candidate_id == int(movie_id):
                continue
            candidate_index = self.movie_id_to_index.get(candidate_id)
            if candidate_index is None:
                continue
            candidate_vector = self.item_factors[candidate_index]
            denominator = source_norm * np.linalg.norm(candidate_vector)
            if denominator == 0:
                continue
            similarity = float(source_vector @ candidate_vector / denominator)
            similarities.append((candidate_id, similarity))

        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:k]

    def _popularity_score(self, movie_id: int) -> float:
        mean_rating = self.movie_means.get(int(movie_id), self.global_mean)
        rating_count = self.movie_counts.get(int(movie_id), 0)
        shrinkage = 10
        score = (
            (rating_count / (rating_count + shrinkage)) * mean_rating
            + (shrinkage / (rating_count + shrinkage)) * self.global_mean
        )
        return float(np.clip(score, self.rating_min, self.rating_max))

