from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifact, ModelArtifactRepository


@dataclass(frozen=True)
class MovieRecommendation:
    rank: int
    movie_id: int
    title: str
    genres: str
    predicted_rating: float
    mean_rating: float | None
    rating_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "predicted_rating": self.predicted_rating,
            "mean_rating": self.mean_rating,
            "rating_count": self.rating_count,
        }


@dataclass(frozen=True)
class RecommendationResult:
    user_id: int
    model_name: str
    recommendations: list[MovieRecommendation]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [recommendation.to_dict() for recommendation in self.recommendations]
        )


class MovieRecommendationService:
    """Artifact-backed recommendation service used by the Streamlit app."""

    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        artifact_repository: ModelArtifactRepository | None = None,
    ) -> None:
        self.config = config
        self.artifact_repository = artifact_repository or ModelArtifactRepository(
            config.model_path
        )
        self._artifact: ModelArtifact | None = None

    def known_user_ids(self) -> list[int]:
        return self.load_artifact().recommender.known_user_ids

    def recommend_for_user(self, user_id: int, k: int = 10) -> RecommendationResult:
        artifact = self.load_artifact()
        candidate_movie_ids = artifact.movies[self.config.item_column].astype(int).tolist()
        seen_movie_ids = artifact.seen_movie_ids_by_user.get(int(user_id), [])
        ranked = artifact.recommender.recommend_for_user(
            user_id=int(user_id),
            candidate_movie_ids=candidate_movie_ids,
            seen_movie_ids=seen_movie_ids,
            k=k,
        )
        return RecommendationResult(
            user_id=int(user_id),
            model_name=artifact.model_name,
            recommendations=self._enrich_ranked_movies(artifact, ranked),
        )

    def similar_movies(self, movie_id: int, k: int = 10) -> list[MovieRecommendation]:
        artifact = self.load_artifact()
        candidate_movie_ids = artifact.movies[self.config.item_column].astype(int).tolist()
        ranked = artifact.recommender.similar_items(
            movie_id=int(movie_id),
            candidate_movie_ids=candidate_movie_ids,
            k=k,
        )
        return self._enrich_ranked_movies(artifact, ranked)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        artifact = self.load_artifact()
        return artifact.recommender.predict_rating(int(user_id), int(movie_id))

    def load_artifact(self) -> ModelArtifact:
        if self._artifact is None:
            self._artifact = self.artifact_repository.load()
        return self._artifact

    def _enrich_ranked_movies(
        self, artifact: ModelArtifact, ranked: list[tuple[int, float]]
    ) -> list[MovieRecommendation]:
        movie_lookup = artifact.movies.set_index(self.config.item_column)
        recommendations: list[MovieRecommendation] = []
        for rank, (movie_id, score) in enumerate(ranked, start=1):
            if movie_id in movie_lookup.index:
                row = movie_lookup.loc[movie_id]
                title = str(row.get(self.config.title_column, f"Movie {movie_id}"))
                genres = str(row.get(self.config.genres_column, "Unknown"))
            else:
                title = f"Movie {movie_id}"
                genres = "Unknown"
            recommendations.append(
                MovieRecommendation(
                    rank=rank,
                    movie_id=int(movie_id),
                    title=title,
                    genres=genres,
                    predicted_rating=round(float(score), 3),
                    mean_rating=_round_or_none(
                        artifact.recommender.movie_means.get(int(movie_id))
                    ),
                    rating_count=int(
                        artifact.recommender.movie_counts.get(int(movie_id), 0)
                    ),
                )
            )
        return recommendations


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)

