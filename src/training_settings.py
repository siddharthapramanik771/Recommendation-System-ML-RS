from dataclasses import asdict
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSettings:
    """Model and evaluation settings for matrix factorization training."""

    random_state: int = 42
    latent_factors: int = 32
    epochs: int = 30
    learning_rate: float = 0.015
    regularization: float = 0.08
    min_user_ratings: int = 4
    min_movie_ratings: int = 1
    k_values: tuple[int, ...] = (5, 10, 20)
    recommendation_count: int = 10

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["k_values"] = list(self.k_values)
        return payload

