from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig


TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
PAGE_AUDIENCE_USER_OFFSET = 10_000
GENRE_AUDIENCE_USER_OFFSET = 100_000


@dataclass(frozen=True)
class TmdbDatasetSnapshot:
    """Local CSV snapshot written from TMDB before training."""

    ratings_path: Path
    movies_path: Path
    ratings_count: int
    movies_count: int
    source: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.source,
            "ratings_path": str(self.ratings_path),
            "movies_path": str(self.movies_path),
            "ratings_count": self.ratings_count,
            "movies_count": self.movies_count,
        }


@dataclass(frozen=True)
class TmdbDatasetSettings:
    """Configuration for fetching a TMDB catalog snapshot."""

    api_key: str | None = None
    bearer_token: str | None = None
    pages: int = 10
    language: str = "en-US"
    region: str | None = None
    sort_by: str = "popularity.desc"
    vote_count_min: int = 50
    include_adult: bool = False
    output_dir: Path | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        bearer_token: str | None = None,
        pages: int | None = None,
        language: str | None = None,
        region: str | None = None,
        sort_by: str | None = None,
        vote_count_min: int | None = None,
        include_adult: bool | None = None,
        output_dir: Path | str | None = None,
        timeout_seconds: float | None = None,
    ) -> "TmdbDatasetSettings":
        resolved_api_key = _first_non_empty(api_key, os.getenv("TMDB_API_KEY"))
        resolved_bearer_token = _first_non_empty(
            bearer_token,
            os.getenv("TMDB_READ_ACCESS_TOKEN"),
            os.getenv("TMDB_BEARER_TOKEN"),
        )
        if not resolved_api_key and not resolved_bearer_token:
            raise ValueError(
                "TMDB training requires TMDB_API_KEY, TMDB_READ_ACCESS_TOKEN, "
                "or TMDB_BEARER_TOKEN."
            )

        resolved_output_dir = _first_non_empty(
            str(output_dir) if output_dir else None,
            os.getenv("TMDB_OUTPUT_DIR"),
        )

        return cls(
            api_key=resolved_api_key,
            bearer_token=resolved_bearer_token,
            pages=pages if pages is not None else _int_env("TMDB_PAGES", 10),
            language=_first_non_empty(language, os.getenv("TMDB_LANGUAGE"), "en-US"),
            region=_first_non_empty(region, os.getenv("TMDB_REGION")),
            sort_by=_first_non_empty(
                sort_by,
                os.getenv("TMDB_SORT_BY"),
                "popularity.desc",
            ),
            vote_count_min=(
                vote_count_min
                if vote_count_min is not None
                else _int_env("TMDB_VOTE_COUNT_MIN", 50)
            ),
            include_adult=(
                include_adult
                if include_adult is not None
                else _bool_env("TMDB_INCLUDE_ADULT", False)
            ),
            output_dir=Path(resolved_output_dir) if resolved_output_dir else None,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else _float_env("TMDB_TIMEOUT_SECONDS", 30.0)
            ),
        )

    def to_source_summary(self, links_path: Path) -> dict[str, object]:
        return {
            "type": "tmdb_aggregate_votes",
            "base_url": TMDB_API_BASE_URL,
            "pages": self.pages,
            "language": self.language,
            "region": self.region,
            "sort_by": self.sort_by,
            "vote_count_min": self.vote_count_min,
            "include_adult": self.include_adult,
            "links_path": str(links_path),
            "auth": "bearer" if self.bearer_token else "api_key",
            "note": (
                "TMDB provides catalog metadata and aggregate public votes, not "
                "per-user collaborative filtering interactions."
            ),
        }


class TmdbDatasetClient:
    """Builds a MovieLens-style training snapshot from TMDB aggregate data."""

    def __init__(
        self,
        settings: TmdbDatasetSettings,
        config: RuntimeConfig = RUNTIME_CONFIG,
    ) -> None:
        self.settings = settings
        self.config = config

    def fetch_and_cache(self) -> TmdbDatasetSnapshot:
        genres_by_id = self._fetch_genres()
        movies = self._fetch_movies()
        if not movies:
            raise ValueError("TMDB returned no movies for the configured filters.")

        movies_frame = self._movies_frame(movies, genres_by_id)
        ratings_frame = self._ratings_frame(movies)
        links_frame = self._links_frame(movies)

        output_dir = self.settings.output_dir or self.config.project_root / "data" / "tmdb"
        output_dir.mkdir(parents=True, exist_ok=True)
        ratings_path = output_dir / "ratings.csv"
        movies_path = output_dir / "movies.csv"
        links_path = output_dir / "links.csv"

        ratings_frame.to_csv(ratings_path, index=False)
        movies_frame.to_csv(movies_path, index=False)
        links_frame.to_csv(links_path, index=False)

        return TmdbDatasetSnapshot(
            ratings_path=ratings_path,
            movies_path=movies_path,
            ratings_count=int(len(ratings_frame)),
            movies_count=int(len(movies_frame)),
            source=self.settings.to_source_summary(links_path),
        )

    def _fetch_genres(self) -> dict[int, str]:
        payload = self._get_json(
            "genre/movie/list",
            {"language": self.settings.language},
        )
        genres = payload.get("genres", [])
        if not isinstance(genres, list):
            return {}
        return {
            int(genre["id"]): str(genre["name"])
            for genre in genres
            if isinstance(genre, dict) and genre.get("id") and genre.get("name")
        }

    def _fetch_movies(self) -> list[dict[str, Any]]:
        movies_by_id: dict[int, dict[str, Any]] = {}
        for page in range(1, self.settings.pages + 1):
            payload = self._get_json(
                "discover/movie",
                {
                    "include_adult": str(self.settings.include_adult).lower(),
                    "include_video": "false",
                    "language": self.settings.language,
                    "page": page,
                    "region": self.settings.region,
                    "sort_by": self.settings.sort_by,
                    "vote_count.gte": self.settings.vote_count_min,
                },
            )
            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                break

            for movie in results:
                if not isinstance(movie, dict) or not movie.get("id"):
                    continue
                movie_id = int(movie["id"])
                movies_by_id.setdefault(movie_id, {**movie, "_tmdb_page": page})

            total_pages = int(payload.get("total_pages") or page)
            if page >= total_pages:
                break

        return list(movies_by_id.values())

    def _movies_frame(
        self, movies: list[dict[str, Any]], genres_by_id: dict[int, str]
    ) -> pd.DataFrame:
        rows = []
        for movie in movies:
            movie_id = int(movie["id"])
            genre_names = [
                genres_by_id.get(int(genre_id), "Unknown")
                for genre_id in movie.get("genre_ids", [])
            ]
            rows.append(
                {
                    self.config.item_column: movie_id,
                    self.config.title_column: _title_with_year(movie),
                    self.config.genres_column: "|".join(genre_names) or "Unknown",
                }
            )
        return pd.DataFrame(rows)

    def _ratings_frame(self, movies: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        now = int(datetime.now(tz=timezone.utc).timestamp())
        for movie in movies:
            movie_id = int(movie["id"])
            rating = _tmdb_vote_to_rating(movie.get("vote_average"), self.config)
            timestamp = _release_timestamp(movie.get("release_date"), now)

            page = int(movie.get("_tmdb_page") or 1)
            rows.append(
                {
                    self.config.user_column: PAGE_AUDIENCE_USER_OFFSET + page,
                    self.config.item_column: movie_id,
                    self.config.rating_column: rating,
                    self.config.timestamp_column: timestamp,
                }
            )

            for genre_id in movie.get("genre_ids", []):
                rows.append(
                    {
                        self.config.user_column: GENRE_AUDIENCE_USER_OFFSET
                        + int(genre_id),
                        self.config.item_column: movie_id,
                        self.config.rating_column: rating,
                        self.config.timestamp_column: timestamp,
                    }
                )

        return pd.DataFrame(rows).drop_duplicates(
            subset=[self.config.user_column, self.config.item_column],
            keep="last",
        )

    def _links_frame(self, movies: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.config.item_column: [int(movie["id"]) for movie in movies],
                self.config.imdb_column: [None for _ in movies],
                self.config.tmdb_column: [int(movie["id"]) for movie in movies],
            }
        )

    def _get_json(self, endpoint: str, query: dict[str, object | None]) -> dict[str, Any]:
        query_params = {
            key: value for key, value in query.items() if value is not None
        }
        if self.settings.api_key:
            query_params["api_key"] = self.settings.api_key

        url = f"{TMDB_API_BASE_URL}/{endpoint.lstrip('/')}"
        if query_params:
            url = f"{url}?{urlencode(query_params)}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "recommendation-system-tmdb-trainer/1.0",
        }
        if self.settings.bearer_token:
            headers["Authorization"] = f"Bearer {self.settings.bearer_token}"

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"TMDB request failed with HTTP {exc.code}: {message}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"TMDB request failed: {exc}") from exc


def _tmdb_vote_to_rating(value: object, config: RuntimeConfig) -> float:
    try:
        rating = float(value) / 2.0
    except (TypeError, ValueError):
        rating = config.rating_min
    rating = max(config.rating_min, min(config.rating_max, rating))
    return round(rating, 2)


def _release_timestamp(value: object, fallback: int) -> int:
    if not value:
        return fallback
    try:
        return int(datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return fallback


def _title_with_year(movie: dict[str, Any]) -> str:
    title = str(movie.get("title") or movie.get("name") or f"TMDB {movie['id']}").strip()
    release_date = str(movie.get("release_date") or "").strip()
    if len(release_date) >= 4 and release_date[:4].isdigit():
        return f"{title} ({release_date[:4]})"
    return title


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _float_env(name: str, default: float) -> float:
    value = _first_non_empty(os.getenv(name))
    return float(value) if value is not None else default


def _int_env(name: str, default: int) -> int:
    value = _first_non_empty(os.getenv(name))
    return int(value) if value is not None else default


def _bool_env(name: str, default: bool) -> bool:
    value = _first_non_empty(os.getenv(name))
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}
