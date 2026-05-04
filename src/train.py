from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.evaluation import RankingEvaluation, evaluate_ranking
from src.matrix_factorization import MatrixFactorizationRecommender
from src.model_bundle import ModelArtifact, ModelArtifactRepository
from src.preprocessing import MovieLensPreprocessor
from src.tmdb_data_source import (
    TmdbDatasetClient,
    TmdbDatasetSettings,
    TmdbDatasetSnapshot,
)
from src.training_settings import TrainingSettings


CSV_SOURCE = "csv"
TMDB_SOURCE = "tmdb"
SAMPLE_SOURCE = "sample"


class RecommenderTrainer:
    """Offline training workflow for the MovieLens recommender."""

    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        settings: TrainingSettings = TrainingSettings(),
        preprocessor: MovieLensPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.preprocessor = preprocessor or MovieLensPreprocessor(config)

    def run(
        self,
        ratings_path: Path | str | None = None,
        movies_path: Path | str | None = None,
        use_sample: bool = False,
        data_source_summary: dict | None = None,
    ) -> dict:
        self.config.ensure_runtime_dirs()
        resolved_ratings_path, resolved_movies_path = self.config.resolve_dataset_paths(
            ratings_path=ratings_path,
            movies_path=movies_path,
            use_sample=use_sample,
        )

        raw_ratings = self.config.load_ratings(resolved_ratings_path)
        raw_movies = self.config.load_movies(resolved_movies_path)
        ratings = self.preprocessor.clean_ratings(raw_ratings)
        movies = self.preprocessor.clean_movies(raw_movies)
        ratings = ratings[
            ratings[self.config.item_column].isin(movies[self.config.item_column])
        ].reset_index(drop=True)
        filtered_ratings = self.preprocessor.filter_interactions(
            ratings, self.settings
        )
        if filtered_ratings.empty:
            raise ValueError(
                "No ratings remain after filtering. Lower min-user or min-movie "
                "settings, or provide a larger MovieLens dataset."
            )

        split = self.preprocessor.leave_one_relevant_out_split(
            filtered_ratings, self.settings
        )
        evaluation_model = MatrixFactorizationRecommender.fit(
            split.train, self.settings, self.config
        )
        all_movie_ids = movies[self.config.item_column].astype(int).tolist()
        evaluation = evaluate_ranking(
            recommender=evaluation_model,
            train=split.train,
            test=split.test,
            all_movie_ids=all_movie_ids,
            k_values=self.settings.k_values,
            config=self.config,
        )

        final_model = MatrixFactorizationRecommender.fit(
            filtered_ratings, self.settings, self.config
        )
        artifact = ModelArtifact(
            recommender=final_model,
            movies=movies,
            seen_movie_ids_by_user=self._seen_movie_ids_by_user(filtered_ratings),
            training_summary={
                "ratings_path": str(resolved_ratings_path),
                "movies_path": str(resolved_movies_path),
                "data_source": data_source_summary or {"type": "csv"},
                "settings": self.settings.to_dict(),
            },
            relevance_threshold=self.config.relevance_threshold,
        )
        ModelArtifactRepository(self.config.model_path).save(artifact)

        metrics_payload = self._metrics_payload(
            ratings=filtered_ratings,
            movies=movies,
            split_train=split.train,
            split_test=split.test,
            evaluation=evaluation,
            ratings_path=resolved_ratings_path,
            movies_path=resolved_movies_path,
            data_source_summary=data_source_summary,
        )
        self.config.metrics_path.write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        return metrics_payload

    def _metrics_payload(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        split_train: pd.DataFrame,
        split_test: pd.DataFrame,
        evaluation: RankingEvaluation,
        ratings_path: Path,
        movies_path: Path,
        data_source_summary: dict | None = None,
    ) -> dict:
        return {
            "model_name": "Biased Matrix Factorization",
            "data_source": data_source_summary or {"type": "csv"},
            "ratings_path": str(ratings_path),
            "movies_path": str(movies_path),
            "artifact_path": str(self.config.model_path),
            "metrics_path": str(self.config.metrics_path),
            "settings": self.settings.to_dict(),
            "catalog": self.preprocessor.catalog_stats(ratings, movies),
            "split": {
                "train_ratings": int(len(split_train)),
                "test_ratings": int(len(split_test)),
                "test_strategy": "leave-one-relevant-rating-out per eligible user",
                "relevance_threshold": self.config.relevance_threshold,
            },
            "ranking": evaluation.to_dict(),
        }

    def _seen_movie_ids_by_user(self, ratings: pd.DataFrame) -> dict[int, list[int]]:
        return {
            int(user_id): group[self.config.item_column].astype(int).tolist()
            for user_id, group in ratings.groupby(self.config.user_column)
        }


def settings_from_env() -> TrainingSettings:
    return TrainingSettings(
        random_state=_int_env("RECOMMENDER_RANDOM_STATE", TrainingSettings.random_state),
        latent_factors=_int_env(
            "RECOMMENDER_LATENT_FACTORS",
            TrainingSettings.latent_factors,
        ),
        epochs=_int_env("RECOMMENDER_EPOCHS", TrainingSettings.epochs),
        learning_rate=_float_env(
            "RECOMMENDER_LEARNING_RATE",
            TrainingSettings.learning_rate,
        ),
        regularization=_float_env(
            "RECOMMENDER_REGULARIZATION",
            TrainingSettings.regularization,
        ),
        min_user_ratings=_int_env(
            "RECOMMENDER_MIN_USER_RATINGS",
            TrainingSettings.min_user_ratings,
        ),
        min_movie_ratings=_int_env(
            "RECOMMENDER_MIN_MOVIE_RATINGS",
            TrainingSettings.min_movie_ratings,
        ),
        k_values=parse_k_values(
            _first_non_empty(os.getenv("RECOMMENDER_K_VALUES"), "5,10,20")
        ),
        recommendation_count=_int_env(
            "RECOMMENDER_RECOMMENDATION_COUNT",
            TrainingSettings.recommendation_count,
        ),
    )


def parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(
        sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    )
    if not values:
        raise ValueError("At least one K value is required.")
    return values


def resolve_training_inputs() -> tuple[
    Path | None,
    Path | None,
    bool,
    TmdbDatasetSnapshot | None,
]:
    source = training_data_source()
    tmdb_snapshot: TmdbDatasetSnapshot | None = None

    if source == TMDB_SOURCE:
        tmdb_settings = TmdbDatasetSettings.from_env()
        tmdb_snapshot = TmdbDatasetClient(tmdb_settings).fetch_and_cache()
        return (
            tmdb_snapshot.ratings_path,
            tmdb_snapshot.movies_path,
            False,
            tmdb_snapshot,
        )

    if source == SAMPLE_SOURCE:
        return None, None, True, None

    if source != CSV_SOURCE:
        valid_sources = ", ".join([CSV_SOURCE, TMDB_SOURCE, SAMPLE_SOURCE])
        raise ValueError(
            f"Unsupported RECOMMENDER_TRAINING_SOURCE={source!r}. "
            f"Use one of: {valid_sources}."
        )

    ratings_path = _optional_path("RECOMMENDER_RATINGS_PATH", "RATINGS_PATH")
    movies_path = _optional_path("RECOMMENDER_MOVIES_PATH", "MOVIES_PATH")
    if bool(ratings_path) != bool(movies_path):
        raise ValueError(
            "Provide both RECOMMENDER_RATINGS_PATH and RECOMMENDER_MOVIES_PATH."
        )
    return ratings_path, movies_path, False, None


def training_data_source() -> str:
    source = _first_non_empty(
        os.getenv("RECOMMENDER_TRAINING_SOURCE"),
        os.getenv("TRAINING_DATA_SOURCE"),
    )
    if source:
        return source.casefold()
    if _tmdb_credentials_available():
        return TMDB_SOURCE
    if _bool_env("RECOMMENDER_USE_SAMPLE", False):
        return SAMPLE_SOURCE
    return CSV_SOURCE


def main() -> None:
    ratings_path, movies_path, use_sample, tmdb_snapshot = resolve_training_inputs()

    trainer = RecommenderTrainer(settings=settings_from_env())
    metrics = trainer.run(
        ratings_path=ratings_path,
        movies_path=movies_path,
        use_sample=use_sample,
        data_source_summary=tmdb_snapshot.to_dict() if tmdb_snapshot else None,
    )
    ranking = metrics["ranking"]
    k = str(trainer.settings.recommendation_count)
    precision = ranking["precision_at_k"].get(k)
    if precision is None:
        k = next(iter(ranking["precision_at_k"]), "n/a")
        precision = ranking["precision_at_k"].get(k, 0.0)
    print(
        "Training complete. "
        f"Evaluated users: {ranking['evaluated_users']}. "
        f"Precision@{k}: {precision:.4f}. "
        f"Artifact: {metrics['artifact_path']}"
    )
    if tmdb_snapshot:
        print(
            "TMDB snapshot saved. "
            f"Ratings: {tmdb_snapshot.ratings_count}. "
            f"Movies: {tmdb_snapshot.movies_count}. "
            f"Directory: {tmdb_snapshot.ratings_path.parent}"
        )


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _optional_path(*names: str) -> Path | None:
    value = _first_non_empty(*(os.getenv(name) for name in names))
    return Path(value) if value else None


def _tmdb_credentials_available() -> bool:
    return bool(
        _first_non_empty(
            os.getenv("TMDB_API_KEY"),
            os.getenv("TMDB_READ_ACCESS_TOKEN"),
            os.getenv("TMDB_BEARER_TOKEN"),
        )
    )


def _int_env(name: str, default: int) -> int:
    value = _first_non_empty(os.getenv(name))
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = _first_non_empty(os.getenv(name))
    return float(value) if value else default


def _bool_env(name: str, default: bool) -> bool:
    value = _first_non_empty(os.getenv(name))
    if not value:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
