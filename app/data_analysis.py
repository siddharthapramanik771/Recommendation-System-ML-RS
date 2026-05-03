from collections import Counter
from collections.abc import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.preprocessing import MovieLensPreprocessor


class DataAnalysisRenderer:
    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config
        self.preprocessor = MovieLensPreprocessor(config)

    def render(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        poster_url_by_movie_id: Callable[[int], str | None] | None = None,
    ) -> None:
        st.subheader("MovieLens Data Analysis")
        stats = self.preprocessor.catalog_stats(ratings, movies)
        left, middle, right, extra = st.columns(4)
        left.metric("Ratings", f"{stats['ratings']:,}")
        middle.metric("Users", f"{stats['users']:,}")
        right.metric("Catalog movies", f"{stats['catalog_movies']:,}")
        extra.metric("Sparsity", f"{stats['matrix_sparsity']:.2%}")

        rating_tab, catalog_tab, genre_tab = st.tabs(
            ["Ratings", "Catalog", "Genres"]
        )
        with rating_tab:
            self.render_rating_distribution(ratings)
        with catalog_tab:
            self.render_catalog_tables(ratings, movies, poster_url_by_movie_id)
        with genre_tab:
            self.render_genre_summary(movies)

    def render_rating_distribution(self, ratings: pd.DataFrame) -> None:
        fig = px.histogram(
            ratings,
            x=self.config.rating_column,
            nbins=10,
            title="Rating Distribution",
            labels={self.config.rating_column: "Rating"},
        )
        fig.update_layout(bargap=0.08)
        st.plotly_chart(fig, use_container_width=True)

        if self.config.timestamp_column in ratings.columns:
            timeline = ratings.copy()
            timeline["rated_at"] = pd.to_datetime(
                timeline[self.config.timestamp_column], unit="s", errors="coerce"
            )
            timeline = (
                timeline.dropna(subset=["rated_at"])
                .set_index("rated_at")
                .resample("YE")[self.config.rating_column]
                .count()
                .reset_index(name="ratings")
            )
            if not timeline.empty:
                fig = px.line(
                    timeline,
                    x="rated_at",
                    y="ratings",
                    markers=True,
                    title="Ratings Over Time",
                    labels={"rated_at": "Year", "ratings": "Rating count"},
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_catalog_tables(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        poster_url_by_movie_id: Callable[[int], str | None] | None = None,
    ) -> None:
        movie_stats = (
            ratings.groupby(self.config.item_column)[self.config.rating_column]
            .agg(rating_count="count", mean_rating="mean")
            .reset_index()
            .merge(movies, on=self.config.item_column, how="left")
            .sort_values(["rating_count", "mean_rating"], ascending=False)
            .head(25)
        )
        movie_stats = movie_stats.rename(columns={self.config.item_column: "movie_id"})
        if poster_url_by_movie_id is not None:
            movie_stats["poster"] = movie_stats["movie_id"].map(
                lambda movie_id: poster_url_by_movie_id(int(movie_id))
            )

        display_columns = [
            self.config.title_column,
            self.config.genres_column,
            "rating_count",
            "mean_rating",
        ]
        if "poster" in movie_stats.columns and movie_stats["poster"].notna().any():
            display_columns.insert(0, "poster")

        st.dataframe(
            movie_stats[display_columns],
            column_config={
                "poster": st.column_config.ImageColumn("Poster", width="small"),
                "mean_rating": st.column_config.NumberColumn(
                    "mean_rating", format="%.3f"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    def render_genre_summary(self, movies: pd.DataFrame) -> None:
        genre_counter: Counter[str] = Counter()
        for genres in movies[self.config.genres_column].dropna():
            genre_counter.update(MovieLensPreprocessor.split_genres(str(genres)))
        genre_frame = pd.DataFrame(
            [{"genre": genre, "movies": count} for genre, count in genre_counter.items()]
        ).sort_values("movies", ascending=False)
        if genre_frame.empty:
            st.info("No genre data is available.")
            return
        fig = px.bar(
            genre_frame.head(20),
            x="movies",
            y="genre",
            orientation="h",
            title="Top Genres by Catalog Count",
            labels={"movies": "Movies", "genre": "Genre"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
