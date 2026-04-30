from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.evaluation import RankingEvaluation, evaluate_ranking
from src.matrix_factorization import MatrixFactorizationRecommender
from src.model_bundle import ModelArtifact, ModelArtifactRepository
from src.preprocessing import MovieLensPreprocessor
from src.training_settings import TrainingSettings


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
    ) -> dict:
        return {
            "model_name": "Biased Matrix Factorization",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MovieLens recommender.")
    parser.add_argument("--ratings-path", type=Path, default=None)
    parser.add_argument("--movies-path", type=Path, default=None)
    parser.add_argument("--sample", action="store_true", help="Use sample CSV files.")
    parser.add_argument("--factors", type=int, default=TrainingSettings.latent_factors)
    parser.add_argument("--epochs", type=int, default=TrainingSettings.epochs)
    parser.add_argument(
        "--learning-rate", type=float, default=TrainingSettings.learning_rate
    )
    parser.add_argument(
        "--regularization", type=float, default=TrainingSettings.regularization
    )
    parser.add_argument(
        "--min-user-ratings", type=int, default=TrainingSettings.min_user_ratings
    )
    parser.add_argument(
        "--min-movie-ratings", type=int, default=TrainingSettings.min_movie_ratings
    )
    parser.add_argument("--k-values", default="5,10,20")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> TrainingSettings:
    return TrainingSettings(
        latent_factors=args.factors,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
        k_values=parse_k_values(args.k_values),
    )


def parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values:
        raise ValueError("At least one K value is required.")
    return values


def main() -> None:
    args = parse_args()
    trainer = RecommenderTrainer(settings=settings_from_args(args))
    metrics = trainer.run(
        ratings_path=args.ratings_path,
        movies_path=args.movies_path,
        use_sample=args.sample,
    )
    ranking = metrics["ranking"]
    k = str(TrainingSettings().recommendation_count)
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


if __name__ == "__main__":
    main()

