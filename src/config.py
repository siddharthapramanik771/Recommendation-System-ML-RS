from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RuntimeConfig:
    """Central paths and data contract for training and app runtime."""

    project_root: Path
    ratings_path: Path
    movies_path: Path
    sample_ratings_path: Path
    sample_movies_path: Path
    model_path: Path
    metrics_path: Path
    user_column: str = "userId"
    item_column: str = "movieId"
    rating_column: str = "rating"
    timestamp_column: str = "timestamp"
    title_column: str = "title"
    genres_column: str = "genres"
    relevance_threshold: float = 4.0
    rating_min: float = 0.5
    rating_max: float = 5.0

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> "RuntimeConfig":
        root = project_root or Path(__file__).resolve().parents[1]
        return cls(
            project_root=root,
            ratings_path=root / "data" / "ml-latest-small" / "ratings.csv",
            movies_path=root / "data" / "ml-latest-small" / "movies.csv",
            sample_ratings_path=root / "data" / "sample" / "ratings.csv",
            sample_movies_path=root / "data" / "sample" / "movies.csv",
            model_path=root / "models" / "model.joblib",
            metrics_path=root / "models" / "training_metrics.json",
        )

    def resolve_dataset_paths(
        self,
        ratings_path: Path | str | None = None,
        movies_path: Path | str | None = None,
        use_sample: bool = False,
    ) -> tuple[Path, Path]:
        if ratings_path and movies_path:
            return Path(ratings_path), Path(movies_path)
        if use_sample:
            return self.sample_ratings_path, self.sample_movies_path
        if self.ratings_path.exists() and self.movies_path.exists():
            return self.ratings_path, self.movies_path
        if self.sample_ratings_path.exists() and self.sample_movies_path.exists():
            return self.sample_ratings_path, self.sample_movies_path
        return self.ratings_path, self.movies_path

    def load_ratings(self, path: Path | str | None = None) -> pd.DataFrame:
        ratings_path = Path(path) if path else self.resolve_dataset_paths()[0]
        return pd.read_csv(ratings_path)

    def load_movies(self, path: Path | str | None = None) -> pd.DataFrame:
        movies_path = Path(path) if path else self.resolve_dataset_paths()[1]
        return pd.read_csv(movies_path)

    def ensure_runtime_dirs(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


RUNTIME_CONFIG = RuntimeConfig.from_project_root()

