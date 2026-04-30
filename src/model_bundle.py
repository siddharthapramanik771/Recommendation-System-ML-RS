from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import RUNTIME_CONFIG
from src.matrix_factorization import MatrixFactorizationRecommender


ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable recommender bundle used by training, app, and services."""

    recommender: MatrixFactorizationRecommender
    movies: pd.DataFrame
    seen_movie_ids_by_user: dict[int, list[int]] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    relevance_threshold: float = RUNTIME_CONFIG.relevance_threshold
    model_name: str = "Biased Matrix Factorization"
    artifact_version: int = ARTIFACT_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelArtifact":
        if "recommender" not in payload:
            raise ValueError("Invalid model artifact: expected a 'recommender' key.")
        return cls(
            recommender=payload["recommender"],
            movies=payload.get("movies", pd.DataFrame()),
            seen_movie_ids_by_user={
                int(user_id): [int(movie_id) for movie_id in movie_ids]
                for user_id, movie_ids in payload.get(
                    "seen_movie_ids_by_user", {}
                ).items()
            },
            training_summary=payload.get("training_summary", {}),
            relevance_threshold=payload.get(
                "relevance_threshold", RUNTIME_CONFIG.relevance_threshold
            ),
            model_name=payload.get("model_name", "Biased Matrix Factorization"),
            artifact_version=payload.get("artifact_version", ARTIFACT_VERSION),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "model_name": self.model_name,
            "recommender": self.recommender,
            "movies": self.movies,
            "seen_movie_ids_by_user": self.seen_movie_ids_by_user,
            "training_summary": self.training_summary,
            "relevance_threshold": self.relevance_threshold,
        }


class ModelArtifactRepository:
    """Persists artifacts without leaking joblib details to app services."""

    def __init__(self, model_path: Path = RUNTIME_CONFIG.model_path) -> None:
        self.model_path = model_path

    def load(self) -> ModelArtifact:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Train the recommender first with `python -m src.train`."
            )

        payload = joblib.load(self.model_path)
        if isinstance(payload, ModelArtifact):
            return payload
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unsupported model artifact format at {self.model_path}: "
                f"{type(payload).__name__}"
            )
        return ModelArtifact.from_payload(payload)

    def save(self, artifact: ModelArtifact) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact.to_payload(), self.model_path)

