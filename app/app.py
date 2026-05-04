from dataclasses import dataclass
import hashlib
from html import escape
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from app.data_analysis import DataAnalysisRenderer
from app.styles import GITHUB_REPOSITORY_URL, apply_page_styles
from app.training_methodology import TrainingMethodologyRenderer
from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifactRepository
from src.predict import MovieRecommendationService
from src.preprocessing import MovieLensPreprocessor


TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"


@dataclass(frozen=True)
class ReferenceDataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    links: pd.DataFrame
    tags: pd.DataFrame
    ratings_path: Path
    movies_path: Path
    links_path: Path | None
    tags_path: Path | None


class ReferenceDataService:
    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: MovieLensPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or MovieLensPreprocessor(config)

    def load(self) -> ReferenceDataset | None:
        ratings_path, movies_path = self._resolve_dataset_paths()
        if not ratings_path.exists() or not movies_path.exists():
            return None
        ratings = self.preprocessor.clean_ratings(self.config.load_ratings(ratings_path))
        movies = self.preprocessor.clean_movies(self.config.load_movies(movies_path))
        ratings = ratings[
            ratings[self.config.item_column].isin(movies[self.config.item_column])
        ].reset_index(drop=True)
        links_path = movies_path.with_name("links.csv")
        tags_path = movies_path.with_name("tags.csv")
        links = self._load_optional_links(links_path)
        tags = self._load_optional_tags(tags_path)
        return ReferenceDataset(
            ratings=ratings,
            movies=movies,
            links=links,
            tags=tags,
            ratings_path=ratings_path,
            movies_path=movies_path,
            links_path=links_path if links_path.exists() else None,
            tags_path=tags_path if tags_path.exists() else None,
        )

    def _resolve_dataset_paths(self) -> tuple[Path, Path]:
        artifact_paths = self._artifact_training_paths()
        if artifact_paths is not None:
            ratings_path, movies_path = artifact_paths
            if ratings_path.exists() and movies_path.exists():
                return ratings_path, movies_path
        return self.config.resolve_dataset_paths()

    def _artifact_training_paths(self) -> tuple[Path, Path] | None:
        if not self.config.model_path.exists():
            return None
        try:
            artifact = ModelArtifactRepository(self.config.model_path).load()
        except (OSError, ValueError, ImportError):
            return None

        summary = artifact.training_summary or {}
        data_source = summary.get("data_source", {})
        path_pairs = [
            (summary.get("ratings_path"), summary.get("movies_path")),
        ]
        if isinstance(data_source, dict):
            path_pairs.append(
                (data_source.get("ratings_path"), data_source.get("movies_path"))
            )

        for ratings_path, movies_path in path_pairs:
            if ratings_path and movies_path:
                return Path(str(ratings_path)), Path(str(movies_path))
        return None

    def _load_optional_links(self, links_path: Path) -> pd.DataFrame:
        if not links_path.exists():
            return pd.DataFrame()
        try:
            return self.preprocessor.clean_links(self.config.load_links(links_path))
        except (OSError, ValueError, pd.errors.ParserError):
            return pd.DataFrame()

    def _load_optional_tags(self, tags_path: Path) -> pd.DataFrame:
        if not tags_path.exists():
            return pd.DataFrame()
        try:
            return self.preprocessor.clean_tags(self.config.load_tags(tags_path))
        except (OSError, ValueError, pd.errors.ParserError):
            return pd.DataFrame()


class DashboardRenderer:
    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        reference_data_service: ReferenceDataService | None = None,
        recommendation_service: MovieRecommendationService | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = MovieLensPreprocessor(config)
        self.reference_data_service = reference_data_service or ReferenceDataService(
            config=config,
            preprocessor=self.preprocessor,
        )
        self.recommendation_service = recommendation_service or MovieRecommendationService(
            config=config
        )
        self.data_analysis_renderer = DataAnalysisRenderer(config)
        self.training_methodology_renderer = TrainingMethodologyRenderer(config)

    def render(self) -> None:
        st.set_page_config(
            page_title="MovieLens Recommender",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        apply_page_styles()
        self.render_header()
        reference = self.reference_data_service.load()
        self.render_sidebar(reference)

        if reference is None:
            st.error(
                f"MovieLens files were not found at {self.config.ratings_path.parent}. "
                "Add ratings.csv and movies.csv, or keep the included sample files."
            )
            st.stop()

        self.render_status_strip(reference)
        movie_tab, recommendation_tab, analysis_tab, methodology_tab = st.tabs(
            [
                "Movie Suggestions",
                "User Recommendations",
                "Data Analysis",
                "Training Methodology",
            ]
        )
        with movie_tab:
            self.render_movie_suggestion_tab(reference)
        with recommendation_tab:
            self.render_recommendation_tab(reference)
        with analysis_tab:
            link_lookup = self.movie_link_lookup(reference.links)
            self.data_analysis_renderer.render(
                reference.ratings,
                reference.movies,
                poster_url_by_movie_id=lambda movie_id: self.poster_url_for_movie(
                    int(movie_id), link_lookup
                ),
                tags=reference.tags,
            )
        with methodology_tab:
            self.training_methodology_renderer.render()

    @staticmethod
    def render_header() -> None:
        st.markdown(
            """
<section class="app-header">
    <h1>MovieLens Recommendation System</h1>
    <p>
        Collaborative filtering with biased matrix factorization, sparse rating
        data, and ranking evaluation through precision@K.
    </p>
</section>
""",
            unsafe_allow_html=True,
        )

    def render_sidebar(self, reference: ReferenceDataset | None) -> None:
        with st.sidebar:
            st.markdown("### Recommender ML")
            st.link_button("GitHub repository", GITHUB_REPOSITORY_URL)
            st.divider()
            if reference is not None:
                st.caption(f"Ratings: `{reference.ratings_path.as_posix()}`")
                st.caption(f"Movies: `{reference.movies_path.as_posix()}`")
                if reference.links_path is not None:
                    st.caption(f"Links: `{reference.links_path.as_posix()}`")
                if reference.tags_path is not None:
                    st.caption(f"Tags: `{reference.tags_path.as_posix()}`")
            self.render_poster_status()
            st.markdown("### Model Artifact")
            if not self.config.model_path.exists():
                st.info("Train the model to enable recommendations.")
                st.code("python -m src.train", language="bash")
                return
            artifact = ModelArtifactRepository(self.config.model_path).load()
            st.caption(f"Loaded model: {artifact.model_name}")
            st.caption(f"Known users: {len(artifact.recommender.known_user_ids):,}")
            st.caption(f"Known movies: {len(artifact.recommender.known_movie_ids):,}")
            st.download_button(
                "Download model",
                data=self.config.model_path.read_bytes(),
                file_name=self.config.model_path.name,
                mime="application/octet-stream",
                width="stretch",
            )
            if self.config.metrics_path.exists():
                st.download_button(
                    "Download metrics",
                    data=self.config.metrics_path.read_bytes(),
                    file_name=self.config.metrics_path.name,
                    mime="application/json",
                    width="stretch",
                )

    @staticmethod
    def render_poster_status() -> None:
        st.markdown("### Poster Images")
        source = tmdb_credentials_source()
        if source:
            st.success(f"TMDB credential detected from {source}.")
            return
        st.info(
            "No TMDB credential is visible to Streamlit. GitHub repository "
            "secrets are only available to GitHub Actions; add `TMDB_API_KEY` "
            "or `TMDB_READ_ACCESS_TOKEN` in Streamlit Cloud secrets."
        )

    def render_status_strip(self, reference: ReferenceDataset) -> None:
        stats = self.preprocessor.catalog_stats(reference.ratings, reference.movies)
        model_status = "Not trained"
        if self.config.model_path.exists():
            model_status = "Artifact ready"
        st.markdown(
            f"""
<div class="status-strip">
    <div class="status-tile">
        <span>Users</span>
        <strong>{stats['users']:,}</strong>
        <small>With rating history</small>
    </div>
    <div class="status-tile">
        <span>Ratings</span>
        <strong>{stats['ratings']:,}</strong>
        <small>Observed interactions</small>
    </div>
    <div class="status-tile">
        <span>Sparsity</span>
        <strong>{stats['matrix_sparsity']:.2%}</strong>
        <small>User-item matrix empty cells</small>
    </div>
    <div class="status-tile">
        <span>Model</span>
        <strong>{model_status}</strong>
        <small>Train offline, score locally</small>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    def render_recommendation_tab(self, reference: ReferenceDataset) -> None:
        st.subheader("Recommend Movies for a User")
        if not self.config.model_path.exists():
            st.warning(
                "Prediction requires a trained artifact. Run `python -m src.train` "
                "from the project root, then refresh the dashboard."
            )
            return

        known_users = self.recommendation_service.known_user_ids()
        if not known_users:
            st.error("The model artifact does not contain trained users.")
            return

        controls = st.columns([2, 1, 1])
        user_id = controls[0].selectbox(
            "User ID",
            known_users,
            key="user_recommendation_user_id",
        )
        k = controls[1].slider(
            "Top K",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key="user_recommendation_top_k",
        )
        show_history = controls[2].toggle(
            "Show history",
            value=True,
            key="user_recommendation_show_history",
        )

        result = self.recommendation_service.recommend_for_user(int(user_id), k=int(k))
        link_lookup = self.movie_link_lookup(reference.links)
        recommendation_frame = self.with_poster_column(result.to_frame(), link_lookup)
        display_columns = [
            "rank",
            "title",
            "genres",
            "predicted_rating",
            "mean_rating",
            "rating_count",
        ]
        if self.should_show_poster_column(recommendation_frame):
            display_columns.insert(0, "poster")
        st.dataframe(
            recommendation_frame[display_columns],
            column_config=self.movie_table_column_config(),
            hide_index=True,
            use_container_width=True,
        )

        if show_history:
            self.render_user_history(reference, int(user_id))

    def render_movie_suggestion_tab(self, reference: ReferenceDataset) -> None:
        st.subheader("Get Suggestions from a Movie")
        if not self.config.model_path.exists():
            st.warning(
                "Prediction requires a trained artifact. Run `python -m src.train` "
                "from the project root, then refresh the dashboard."
            )
            return

        artifact = self.recommendation_service.load_artifact()
        known_movie_ids = set(artifact.recommender.known_movie_ids)
        movie_options = (
            reference.movies[
                reference.movies[self.config.item_column].astype(int).isin(known_movie_ids)
            ]
            .sort_values(self.config.title_column)
            .reset_index(drop=True)
        )
        if movie_options.empty:
            st.error("The model artifact does not contain trained movies.")
            return

        tag_lookup = self.movie_tag_lookup(reference.tags)
        link_lookup = self.movie_link_lookup(reference.links)
        movie_label_by_id = dict(
            zip(
                movie_options[self.config.item_column].astype(int),
                movie_options[self.config.title_column].astype(str),
                strict=False,
            )
        )

        controls = st.columns([3, 1])
        selected_movie_id = controls[0].selectbox(
            "Movie",
            movie_options[self.config.item_column].astype(int).tolist(),
            format_func=lambda movie_id: movie_label_by_id.get(
                int(movie_id), f"Movie {movie_id}"
            ),
            key="movie_suggestion_movie_title",
        )
        k = controls[1].slider(
            "Top K",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key="movie_suggestion_top_k",
        )

        selected_movie = movie_options[
            movie_options[self.config.item_column] == int(selected_movie_id)
        ].iloc[0]
        st.caption(str(selected_movie[self.config.genres_column]))
        self.render_movie_context(
            movie_id=int(selected_movie_id),
            tag_lookup=tag_lookup,
            link_lookup=link_lookup,
        )

        candidate_count = max(int(k) * 5, 40)
        similar = self.recommendation_service.similar_movies(
            int(selected_movie_id), k=candidate_count
        )
        if not similar:
            st.info("Similar movies are unavailable for this title.")
            return

        similar_frame = self.enriched_similar_movie_frame(
            similar=similar,
            source_movie_id=int(selected_movie_id),
            tag_lookup=tag_lookup,
            link_lookup=link_lookup,
        ).head(int(k))
        similar_frame["rank"] = range(1, len(similar_frame) + 1)
        similar_frame = self.with_poster_column(similar_frame, link_lookup)

        display_columns = [
            "title",
            "genres",
            "hybrid_score",
            "similarity_score",
            "matching_tags",
            "community_tags",
            "mean_rating",
            "rating_count",
        ]
        if similar_frame["poster"].notna().any():
            display_columns.insert(0, "poster")
        display_columns.insert(1 if "poster" in display_columns else 0, "rank")
        if "imdb" in similar_frame and similar_frame["imdb"].notna().any():
            display_columns.append("imdb")
        if "tmdb" in similar_frame and similar_frame["tmdb"].notna().any():
            display_columns.append("tmdb")

        st.dataframe(
            similar_frame[display_columns],
            column_config=self.movie_table_column_config(),
            hide_index=True,
            use_container_width=True,
        )

    def with_poster_column(
        self, frame: pd.DataFrame, link_lookup: dict[int, dict[str, str]]
    ) -> pd.DataFrame:
        if frame.empty or "movie_id" not in frame.columns:
            return frame
        enriched = frame.copy()
        enriched["poster"] = enriched["movie_id"].map(
            lambda movie_id: self.poster_url_for_movie(int(movie_id), link_lookup)
        )
        return enriched

    @staticmethod
    def should_show_poster_column(frame: pd.DataFrame) -> bool:
        return "poster" in frame.columns and frame["poster"].notna().any()

    @staticmethod
    def movie_table_column_config() -> dict:
        return {
            "poster": st.column_config.ImageColumn("Poster", width="small"),
            "hybrid_score": st.column_config.NumberColumn(
                "hybrid_score", format="%.3f"
            ),
            "similarity_score": st.column_config.NumberColumn(
                "similarity_score", format="%.3f"
            ),
            "predicted_rating": st.column_config.NumberColumn(
                "predicted_rating", format="%.3f"
            ),
            "mean_rating": st.column_config.NumberColumn("mean_rating", format="%.3f"),
            "imdb": st.column_config.LinkColumn("IMDb", display_text="IMDb"),
            "tmdb": st.column_config.LinkColumn("TMDB", display_text="TMDB"),
        }

    def render_movie_context(
        self,
        movie_id: int,
        tag_lookup: dict[int, list[str]],
        link_lookup: dict[int, dict[str, str]],
    ) -> None:
        poster_url = self.poster_url_for_movie(movie_id, link_lookup)
        if poster_url:
            metadata_columns = st.columns([1, 3, 1])
            with metadata_columns[0]:
                st.image(poster_url, width=120)
            tag_column = metadata_columns[1]
            link_column = metadata_columns[2]
        else:
            metadata_columns = st.columns([3, 1])
            tag_column = metadata_columns[0]
            link_column = metadata_columns[1]

        with tag_column:
            self.render_tag_chips("Community tags", tag_lookup.get(int(movie_id), []))
            if not tmdb_credentials_available():
                st.caption(
                    "Add TMDB_API_KEY or TMDB_READ_ACCESS_TOKEN to show posters."
                )
        with link_column:
            self.render_external_links(link_lookup.get(int(movie_id), {}))

    @staticmethod
    def render_tag_chips(label: str, tags: list[str]) -> None:
        if not tags:
            st.caption(f"{label}: none available")
            return
        chips = "".join(
            f"<span class=\"tag-chip\">{escape(tag)}</span>" for tag in tags[:8]
        )
        st.markdown(
            f"<div class=\"tag-row\"><strong>{escape(label)}</strong>{chips}</div>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_external_links(links: dict[str, str]) -> None:
        if not links:
            return
        buttons = st.columns(2)
        if "imdb" in links:
            buttons[0].link_button("IMDb", links["imdb"], width="stretch")
        if "tmdb" in links:
            buttons[1].link_button("TMDB", links["tmdb"], width="stretch")

    def enriched_similar_movie_frame(
        self,
        similar: list,
        source_movie_id: int,
        tag_lookup: dict[int, list[str]],
        link_lookup: dict[int, dict[str, str]],
    ) -> pd.DataFrame:
        source_tag_set = self.normalized_tag_set(tag_lookup.get(source_movie_id, []))
        rows = []
        for recommendation in similar:
            row = recommendation.to_dict()
            movie_id = int(row["movie_id"])
            similarity_score = float(row.pop("predicted_rating"))
            candidate_tags = tag_lookup.get(movie_id, [])
            candidate_tag_set = self.normalized_tag_set(candidate_tags)
            matched_tag_keys = source_tag_set & candidate_tag_set
            matching_tags = [
                tag for tag in candidate_tags if tag.casefold() in matched_tag_keys
            ][:5]
            tag_match = (
                len(matched_tag_keys) / len(source_tag_set) if source_tag_set else 0.0
            )
            normalized_similarity = (similarity_score + 1.0) / 2.0
            hybrid_score = (
                0.85 * normalized_similarity + 0.15 * tag_match
                if source_tag_set
                else normalized_similarity
            )
            links = link_lookup.get(movie_id, {})
            rows.append(
                {
                    **row,
                    "hybrid_score": round(float(hybrid_score), 3),
                    "similarity_score": round(similarity_score, 3),
                    "tag_match": round(float(tag_match), 3),
                    "matching_tags": self.join_tags(matching_tags),
                    "community_tags": self.join_tags(candidate_tags[:5]),
                    "imdb": links.get("imdb"),
                    "tmdb": links.get("tmdb"),
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(
            ["hybrid_score", "similarity_score"], ascending=[False, False]
        ).reset_index(drop=True)

    def movie_tag_lookup(self, tags: pd.DataFrame) -> dict[int, list[str]]:
        if tags.empty or self.config.tag_column not in tags.columns:
            return {}

        lookup: dict[int, list[str]] = {}
        for movie_id, group in tags.groupby(self.config.item_column):
            tag_counts = (
                group[self.config.tag_column]
                .astype(str)
                .str.strip()
                .str.casefold()
                .value_counts()
            )
            lookup[int(movie_id)] = tag_counts.index[:8].tolist()
        return lookup

    def movie_link_lookup(self, links: pd.DataFrame) -> dict[int, dict[str, str]]:
        if links.empty:
            return {}

        lookup: dict[int, dict[str, str]] = {}
        for _, row in links.iterrows():
            movie_links: dict[str, str] = {}
            imdb_id = row.get(self.config.imdb_column)
            if pd.notna(imdb_id):
                movie_links["imdb"] = f"https://www.imdb.com/title/tt{imdb_id}/"
            tmdb_id = row.get(self.config.tmdb_column)
            if pd.notna(tmdb_id):
                movie_links["tmdb"] = f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
            if movie_links:
                lookup[int(row[self.config.item_column])] = movie_links
        return lookup

    @staticmethod
    def poster_url_for_movie(
        movie_id: int, link_lookup: dict[int, dict[str, str]]
    ) -> str | None:
        links = link_lookup.get(int(movie_id), {})
        tmdb_url = links.get("tmdb")
        if not tmdb_url:
            return None
        tmdb_id = tmdb_url.rstrip("/").split("/")[-1]
        return fetch_tmdb_poster_url(tmdb_id, tmdb_credentials_cache_key())

    @staticmethod
    def normalized_tag_set(tags: list[str]) -> set[str]:
        return {tag.casefold() for tag in tags if tag.strip()}

    @staticmethod
    def join_tags(tags: list[str]) -> str:
        return ", ".join(dict.fromkeys(tag for tag in tags if tag))

    def render_user_history(self, reference: ReferenceDataset, user_id: int) -> None:
        history = reference.ratings[
            reference.ratings[self.config.user_column] == int(user_id)
        ].merge(reference.movies, on=self.config.item_column, how="left")
        if history.empty:
            st.info("This user has no visible ratings in the reference dataset.")
            return
        history = history.sort_values(
            [self.config.rating_column, self.config.timestamp_column],
            ascending=[False, False],
        )
        history = history.rename(columns={self.config.item_column: "movie_id"}).head(10)
        link_lookup = self.movie_link_lookup(reference.links)
        history = self.with_poster_column(history, link_lookup)
        display_columns = [
            self.config.title_column,
            self.config.genres_column,
            self.config.rating_column,
        ]
        if self.should_show_poster_column(history):
            display_columns.insert(0, "poster")
        st.markdown("#### Highest Rated by This User")
        st.dataframe(
            history[display_columns],
            column_config=self.movie_table_column_config(),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    DashboardRenderer().render()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_tmdb_poster_url(
    tmdb_id: str, credentials_cache_key: str | None
) -> str | None:
    token = config_value("TMDB_READ_ACCESS_TOKEN") or config_value(
        "TMDB_BEARER_TOKEN"
    )
    api_key = config_value("TMDB_API_KEY")
    if not token and not api_key:
        return None

    url = TMDB_API_URL.format(tmdb_id=tmdb_id)
    headers = {
        "Accept": "application/json",
        "User-Agent": "MovieLens-Recommender/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        url = f"{url}?{urlencode({'api_key': api_key})}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    poster_path = payload.get("poster_path")
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"


def tmdb_credentials_available() -> bool:
    return bool(tmdb_credentials_source())


def tmdb_credentials_cache_key() -> str | None:
    for name in (
        "TMDB_READ_ACCESS_TOKEN",
        "TMDB_BEARER_TOKEN",
        "TMDB_API_KEY",
    ):
        value = config_value(name)
        if value:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            return f"{name}:{digest}"
    return None


def tmdb_credentials_source() -> str | None:
    for name in (
        "TMDB_READ_ACCESS_TOKEN",
        "TMDB_BEARER_TOKEN",
        "TMDB_API_KEY",
    ):
        source = config_value_source(name)
        if source:
            return f"{source} `{name}`"
    return None


def config_value(name: str) -> str | None:
    value, _ = config_value_with_source(name)
    return value


def config_value_source(name: str) -> str | None:
    _, source = config_value_with_source(name)
    return source


def config_value_with_source(name: str) -> tuple[str | None, str | None]:
    value = os.environ.get(name)
    if value:
        return value, "environment variable"
    try:
        secret_value = st.secrets.get(name)
    except (AttributeError, FileNotFoundError, KeyError, RuntimeError):
        return None, None
    if secret_value:
        return str(secret_value), "Streamlit secret"
    return None, None
