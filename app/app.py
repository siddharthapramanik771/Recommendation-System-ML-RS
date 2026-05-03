from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from app.data_analysis import DataAnalysisRenderer
from app.styles import GITHUB_REPOSITORY_URL, apply_page_styles
from app.training_methodology import TrainingMethodologyRenderer
from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifactRepository
from src.predict import MovieRecommendationService
from src.preprocessing import MovieLensPreprocessor


@dataclass(frozen=True)
class ReferenceDataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    ratings_path: Path
    movies_path: Path


class ReferenceDataService:
    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: MovieLensPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or MovieLensPreprocessor(config)

    def load(self) -> ReferenceDataset | None:
        ratings_path, movies_path = self.config.resolve_dataset_paths()
        if not ratings_path.exists() or not movies_path.exists():
            return None
        ratings = self.preprocessor.clean_ratings(self.config.load_ratings(ratings_path))
        movies = self.preprocessor.clean_movies(self.config.load_movies(movies_path))
        ratings = ratings[
            ratings[self.config.item_column].isin(movies[self.config.item_column])
        ].reset_index(drop=True)
        return ReferenceDataset(ratings, movies, ratings_path, movies_path)


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
            self.data_analysis_renderer.render(reference.ratings, reference.movies)
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
        user_id = controls[0].selectbox("User ID", known_users)
        k = controls[1].slider("Top K", min_value=5, max_value=20, value=10, step=1)
        show_history = controls[2].toggle("Show history", value=True)

        result = self.recommendation_service.recommend_for_user(int(user_id), k=int(k))
        recommendation_frame = result.to_frame()
        st.dataframe(
            recommendation_frame[
                [
                    "rank",
                    "title",
                    "genres",
                    "predicted_rating",
                    "mean_rating",
                    "rating_count",
                ]
            ],
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

        controls = st.columns([3, 1])
        selected_title = controls[0].selectbox(
            "Movie",
            movie_options[self.config.title_column].tolist(),
        )
        k = controls[1].slider("Top K", min_value=5, max_value=20, value=10, step=1)

        selected_movie = movie_options[
            movie_options[self.config.title_column] == selected_title
        ].iloc[0]
        st.caption(str(selected_movie[self.config.genres_column]))

        similar = self.recommendation_service.similar_movies(
            int(selected_movie[self.config.item_column]), k=int(k)
        )
        if not similar:
            st.info("Similar movies are unavailable for this title.")
            return

        similar_frame = pd.DataFrame(
            [recommendation.to_dict() for recommendation in similar]
        ).rename(columns={"predicted_rating": "similarity_score"})
        st.dataframe(
            similar_frame[
                [
                    "rank",
                    "title",
                    "genres",
                    "similarity_score",
                    "mean_rating",
                    "rating_count",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

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
        st.markdown("#### Highest Rated by This User")
        st.dataframe(
            history[
                [
                    self.config.title_column,
                    self.config.genres_column,
                    self.config.rating_column,
                ]
            ].head(10),
            hide_index=True,
            use_container_width=True,
        )

def main() -> None:
    DashboardRenderer().render()
