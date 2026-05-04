from dataclasses import dataclass

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.training_settings import TrainingSettings


NO_GENRES = "(no genres listed)"
UNKNOWN_GENRE = "Unknown"


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


class MovieLensPreprocessor:
    """Cleans MovieLens CSV files and creates ranking-evaluation splits."""

    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config

    def clean_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        df = ratings.copy()
        df.columns = df.columns.str.strip()
        self._require_columns(
            df,
            {
                self.config.user_column,
                self.config.item_column,
                self.config.rating_column,
            },
            "ratings",
        )

        keep_columns = [
            self.config.user_column,
            self.config.item_column,
            self.config.rating_column,
        ]
        if self.config.timestamp_column in df.columns:
            keep_columns.append(self.config.timestamp_column)

        df = df[keep_columns].copy()
        for column in (self.config.user_column, self.config.item_column):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df[self.config.rating_column] = pd.to_numeric(
            df[self.config.rating_column], errors="coerce"
        )
        if self.config.timestamp_column in df.columns:
            numeric_timestamp = pd.to_numeric(
                df[self.config.timestamp_column], errors="coerce"
            )
            parsed_timestamp = pd.to_datetime(
                df[self.config.timestamp_column], errors="coerce", utc=True
            )
            timestamp_seconds = parsed_timestamp.map(
                lambda value: value.timestamp() if pd.notna(value) else pd.NA
            )
            df[self.config.timestamp_column] = numeric_timestamp.fillna(
                timestamp_seconds.astype("float64")
            ).fillna(0)
        else:
            df[self.config.timestamp_column] = range(len(df))

        df = df.dropna(
            subset=[
                self.config.user_column,
                self.config.item_column,
                self.config.rating_column,
            ]
        )
        df[self.config.user_column] = df[self.config.user_column].astype(int)
        df[self.config.item_column] = df[self.config.item_column].astype(int)
        df[self.config.rating_column] = df[self.config.rating_column].astype(float)
        df = df[
            df[self.config.rating_column].between(
                self.config.rating_min, self.config.rating_max
            )
        ]
        df = df.sort_values(self.config.timestamp_column)
        df = df.drop_duplicates(
            subset=[self.config.user_column, self.config.item_column],
            keep="last",
        )
        return df.reset_index(drop=True)

    def clean_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        df = movies.copy()
        df.columns = df.columns.str.strip()
        self._require_columns(
            df,
            {
                self.config.item_column,
                self.config.title_column,
                self.config.genres_column,
            },
            "movies",
        )
        df = df[
            [
                self.config.item_column,
                self.config.title_column,
                self.config.genres_column,
            ]
        ].copy()
        df[self.config.item_column] = pd.to_numeric(
            df[self.config.item_column], errors="coerce"
        )
        df = df.dropna(subset=[self.config.item_column])
        df[self.config.item_column] = df[self.config.item_column].astype(int)
        df[self.config.title_column] = (
            df[self.config.title_column].fillna("Untitled").astype(str).str.strip()
        )
        df[self.config.genres_column] = (
            df[self.config.genres_column]
            .fillna(UNKNOWN_GENRE)
            .astype(str)
            .str.strip()
            .replace({"": UNKNOWN_GENRE, NO_GENRES: UNKNOWN_GENRE})
        )
        return df.drop_duplicates(subset=[self.config.item_column]).reset_index(
            drop=True
        )

    def clean_links(self, links: pd.DataFrame) -> pd.DataFrame:
        df = links.copy()
        df.columns = df.columns.str.strip()
        self._require_columns(df, {self.config.item_column}, "links")

        keep_columns = [self.config.item_column]
        for column in (self.config.imdb_column, self.config.tmdb_column):
            if column in df.columns:
                keep_columns.append(column)
        df = df[keep_columns].copy()

        df[self.config.item_column] = pd.to_numeric(
            df[self.config.item_column], errors="coerce"
        )
        df = df.dropna(subset=[self.config.item_column])
        df[self.config.item_column] = df[self.config.item_column].astype(int)

        if self.config.imdb_column in df.columns:
            df[self.config.imdb_column] = df[self.config.imdb_column].map(
                _format_imdb_id
            )
        if self.config.tmdb_column in df.columns:
            df[self.config.tmdb_column] = pd.to_numeric(
                df[self.config.tmdb_column], errors="coerce"
            ).astype("Int64")

        return df.drop_duplicates(subset=[self.config.item_column]).reset_index(
            drop=True
        )

    def clean_tags(self, tags: pd.DataFrame) -> pd.DataFrame:
        df = tags.copy()
        df.columns = df.columns.str.strip()
        self._require_columns(
            df,
            {self.config.item_column, self.config.tag_column},
            "tags",
        )

        keep_columns = [self.config.item_column, self.config.tag_column]
        if self.config.user_column in df.columns:
            keep_columns.insert(0, self.config.user_column)
        if self.config.timestamp_column in df.columns:
            keep_columns.append(self.config.timestamp_column)
        df = df[keep_columns].copy()

        df[self.config.item_column] = pd.to_numeric(
            df[self.config.item_column], errors="coerce"
        )
        df[self.config.tag_column] = (
            df[self.config.tag_column].fillna("").astype(str).str.strip()
        )
        df = df.dropna(subset=[self.config.item_column])
        df = df[df[self.config.tag_column] != ""]
        df[self.config.item_column] = df[self.config.item_column].astype(int)

        if self.config.user_column in df.columns:
            df[self.config.user_column] = pd.to_numeric(
                df[self.config.user_column], errors="coerce"
            ).astype("Int64")
        if self.config.timestamp_column in df.columns:
            df[self.config.timestamp_column] = pd.to_numeric(
                df[self.config.timestamp_column], errors="coerce"
            ).fillna(0)

        return df.reset_index(drop=True)

    def filter_interactions(
        self, ratings: pd.DataFrame, settings: TrainingSettings
    ) -> pd.DataFrame:
        df = ratings.copy()
        previous_shape: tuple[int, int] | None = None
        while previous_shape != df.shape and not df.empty:
            previous_shape = df.shape
            user_counts = df[self.config.user_column].value_counts()
            active_users = user_counts[
                user_counts >= settings.min_user_ratings
            ].index
            df = df[df[self.config.user_column].isin(active_users)]

            movie_counts = df[self.config.item_column].value_counts()
            active_movies = movie_counts[
                movie_counts >= settings.min_movie_ratings
            ].index
            df = df[df[self.config.item_column].isin(active_movies)]

        return df.reset_index(drop=True)

    def leave_one_relevant_out_split(
        self, ratings: pd.DataFrame, settings: TrainingSettings
    ) -> DatasetSplit:
        test_indices: list[int] = []
        sorted_ratings = ratings.sort_values(self.config.timestamp_column)
        for _, user_frame in sorted_ratings.groupby(self.config.user_column):
            if len(user_frame) < settings.min_user_ratings:
                continue
            relevant = user_frame[
                user_frame[self.config.rating_column] >= self.config.relevance_threshold
            ]
            if relevant.empty:
                continue
            test_indices.append(int(relevant.index[-1]))

        test = ratings.loc[test_indices].copy() if test_indices else ratings.iloc[0:0]
        train = ratings.drop(index=test_indices).copy() if test_indices else ratings
        return DatasetSplit(
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
        )

    def enrich_ratings(
        self, ratings: pd.DataFrame, movies: pd.DataFrame
    ) -> pd.DataFrame:
        return ratings.merge(movies, on=self.config.item_column, how="left")

    def catalog_stats(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
        users = ratings[self.config.user_column].nunique()
        rated_movies = ratings[self.config.item_column].nunique()
        catalog_movies = len(movies)
        possible_interactions = max(users * catalog_movies, 1)
        density = len(ratings) / possible_interactions
        return {
            "ratings": int(len(ratings)),
            "users": int(users),
            "rated_movies": int(rated_movies),
            "catalog_movies": int(catalog_movies),
            "matrix_density": float(density),
            "matrix_sparsity": float(1 - density),
            "mean_rating": float(ratings[self.config.rating_column].mean()),
        }

    @staticmethod
    def split_genres(genres: str) -> list[str]:
        if not genres or genres == UNKNOWN_GENRE:
            return [UNKNOWN_GENRE]
        return [genre.strip() for genre in str(genres).split("|") if genre.strip()]

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, required_columns: set[str], label: str
    ) -> None:
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(
                f"Missing required {label} column(s): {', '.join(missing)}"
            )


def display_name(column_name: str) -> str:
    return column_name.replace("_", " ").replace("Id", " ID").title()


def _format_imdb_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    imdb_id = str(value).strip()
    if not imdb_id:
        return None
    if imdb_id.endswith(".0"):
        imdb_id = imdb_id[:-2]
    return imdb_id.zfill(7)
