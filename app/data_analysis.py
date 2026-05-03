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
        tags: pd.DataFrame | None = None,
    ) -> None:
        st.subheader("MovieLens Data Analysis")
        stats = self.preprocessor.catalog_stats(ratings, movies)
        left, middle, right, extra = st.columns(4)
        left.metric("Ratings", f"{stats['ratings']:,}")
        middle.metric("Users", f"{stats['users']:,}")
        right.metric("Catalog movies", f"{stats['catalog_movies']:,}")
        extra.metric("Sparsity", f"{stats['matrix_sparsity']:.2%}")

        rating_tab, activity_tab, catalog_tab, genre_tab, tag_tab = st.tabs(
            ["Ratings", "User Activity", "Catalog", "Genres", "Tags"]
        )
        with rating_tab:
            self.render_rating_distribution(ratings)
            self.render_rating_time_analysis(ratings)
        with activity_tab:
            self.render_user_activity(ratings)
        with catalog_tab:
            self.render_catalog_tables(ratings, movies, poster_url_by_movie_id)
            self.render_movie_quality_analysis(ratings, movies)
        with genre_tab:
            self.render_genre_summary(ratings, movies)
        with tag_tab:
            self.render_tag_summary(tags, movies, poster_url_by_movie_id)

    def render_rating_distribution(self, ratings: pd.DataFrame) -> None:
        rating_counts = (
            ratings[self.config.rating_column]
            .value_counts()
            .sort_index()
            .rename_axis(self.config.rating_column)
            .reset_index(name="ratings")
        )
        fig = px.bar(
            rating_counts,
            x=self.config.rating_column,
            y="ratings",
            title="Rating Distribution",
            labels={self.config.rating_column: "Rating", "ratings": "Rating count"},
        )
        fig.update_layout(bargap=0.12)
        st.plotly_chart(fig, use_container_width=True)

        summary_columns = st.columns(4)
        summary_columns[0].metric(
            "Mean rating", f"{ratings[self.config.rating_column].mean():.2f}"
        )
        summary_columns[1].metric(
            "Median rating", f"{ratings[self.config.rating_column].median():.2f}"
        )
        high_rating_share = (
            ratings[self.config.rating_column] >= self.config.relevance_threshold
        ).mean()
        summary_columns[2].metric(
            f"Ratings >= {self.config.relevance_threshold:g}",
            f"{high_rating_share:.1%}",
        )
        summary_columns[3].metric(
            "Distinct values", f"{ratings[self.config.rating_column].nunique():,}"
        )

    def render_rating_time_analysis(self, ratings: pd.DataFrame) -> None:
        timeline = self.ratings_with_datetime(ratings)
        if timeline.empty:
            st.info("Timestamp data is not available for time-based charts.")
            return

        yearly = (
            timeline.assign(year=timeline["rated_at"].dt.year)
            .groupby("year")[self.config.rating_column]
            .agg(ratings="count", mean_rating="mean")
            .reset_index()
        )
        if yearly.empty:
            return

        left, right = st.columns(2)
        with left:
            fig = px.line(
                yearly,
                x="year",
                y="ratings",
                markers=True,
                title="Ratings by Year",
                labels={"year": "Year", "ratings": "Rating count"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.line(
                yearly,
                x="year",
                y="mean_rating",
                markers=True,
                title="Average Rating by Year",
                labels={"year": "Year", "mean_rating": "Mean rating"},
            )
            fig.update_yaxes(range=[self.config.rating_min, self.config.rating_max])
            st.plotly_chart(fig, use_container_width=True)

        self.render_time_heatmap(timeline)

    def render_time_heatmap(self, ratings: pd.DataFrame) -> None:
        if ratings.empty:
            return
        heatmap_frame = ratings.copy()
        heatmap_frame["weekday"] = heatmap_frame["rated_at"].dt.day_name()
        heatmap_frame["hour"] = heatmap_frame["rated_at"].dt.hour
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heatmap = (
            heatmap_frame.groupby(["weekday", "hour"], observed=False)
            .size()
            .reset_index(name="ratings")
        )
        heatmap["weekday"] = pd.Categorical(
            heatmap["weekday"], categories=weekday_order, ordered=True
        )
        heatmap = heatmap.sort_values(["weekday", "hour"])
        fig = px.density_heatmap(
            heatmap,
            x="hour",
            y="weekday",
            z="ratings",
            histfunc="sum",
            title="Rating Activity by Weekday and Hour",
            labels={"hour": "Hour of day", "weekday": "Weekday", "ratings": "Ratings"},
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_user_activity(self, ratings: pd.DataFrame) -> None:
        user_stats = (
            ratings.groupby(self.config.user_column)[self.config.rating_column]
            .agg(rating_count="count", mean_rating="mean")
            .reset_index()
            .sort_values("rating_count", ascending=False)
        )
        if user_stats.empty:
            st.info("No user activity data is available.")
            return

        activity_columns = st.columns(4)
        activity_columns[0].metric(
            "Median ratings/user", f"{user_stats['rating_count'].median():.0f}"
        )
        activity_columns[1].metric(
            "90th percentile", f"{user_stats['rating_count'].quantile(0.9):.0f}"
        )
        activity_columns[2].metric(
            "Most active user", f"{int(user_stats.iloc[0][self.config.user_column])}"
        )
        activity_columns[3].metric(
            "Ratings by top 10 users",
            f"{user_stats.head(10)['rating_count'].sum() / len(ratings):.1%}",
        )

        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                user_stats,
                x="rating_count",
                nbins=50,
                log_y=True,
                title="User Activity Distribution",
                labels={"rating_count": "Ratings per user"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            top_users = user_stats.head(20).copy()
            top_users[self.config.user_column] = top_users[
                self.config.user_column
            ].astype(str)
            fig = px.bar(
                top_users,
                x="rating_count",
                y=self.config.user_column,
                orientation="h",
                title="Most Active Users",
                labels={
                    "rating_count": "Rating count",
                    self.config.user_column: "User ID",
                },
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            user_stats,
            x="rating_count",
            y="mean_rating",
            log_x=True,
            title="User Leniency vs Activity",
            labels={
                "rating_count": "Ratings per user",
                "mean_rating": "Mean rating",
            },
            hover_data=[self.config.user_column],
        )
        fig.update_yaxes(range=[self.config.rating_min, self.config.rating_max])
        st.plotly_chart(fig, use_container_width=True)

    def render_movie_quality_analysis(
        self, ratings: pd.DataFrame, movies: pd.DataFrame
    ) -> None:
        movie_stats = self.movie_stats(ratings, movies)
        if movie_stats.empty:
            return

        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                movie_stats,
                x="rating_count",
                nbins=60,
                log_y=True,
                title="Movie Popularity Distribution",
                labels={"rating_count": "Ratings per movie"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                movie_stats,
                x="rating_count",
                y="mean_rating",
                color="primary_genre",
                log_x=True,
                title="Popularity vs Average Rating",
                labels={
                    "rating_count": "Ratings per movie",
                    "mean_rating": "Mean rating",
                    "primary_genre": "Primary genre",
                },
                hover_data=[self.config.title_column],
            )
            fig.update_yaxes(range=[self.config.rating_min, self.config.rating_max])
            st.plotly_chart(fig, use_container_width=True)

        reliable_min_count = max(25, int(movie_stats["rating_count"].quantile(0.75)))
        reliable_movies = movie_stats[
            movie_stats["rating_count"] >= reliable_min_count
        ].sort_values("mean_rating", ascending=False)
        if reliable_movies.empty:
            return
        fig = px.bar(
            reliable_movies.head(15).sort_values("mean_rating"),
            x="mean_rating",
            y=self.config.title_column,
            color="rating_count",
            orientation="h",
            title=f"Highest Rated Movies with at Least {reliable_min_count} Ratings",
            labels={
                "mean_rating": "Mean rating",
                self.config.title_column: "Movie",
                "rating_count": "Ratings",
            },
        )
        fig.update_xaxes(range=[self.config.rating_min, self.config.rating_max])
        st.plotly_chart(fig, use_container_width=True)

    def render_tag_summary(
        self,
        tags: pd.DataFrame | None,
        movies: pd.DataFrame,
        poster_url_by_movie_id: Callable[[int], str | None] | None = None,
    ) -> None:
        if tags is None or tags.empty or self.config.tag_column not in tags.columns:
            st.info("No tag data is available for this dataset.")
            return

        tag_frame = tags.copy()
        tag_frame["tag_normalized"] = (
            tag_frame[self.config.tag_column].astype(str).str.strip().str.casefold()
        )
        tag_frame = tag_frame[tag_frame["tag_normalized"] != ""]
        if tag_frame.empty:
            st.info("No usable tag data is available.")
            return

        tag_columns = st.columns(4)
        tag_columns[0].metric("Tags", f"{len(tag_frame):,}")
        tag_columns[1].metric(
            "Unique tags", f"{tag_frame['tag_normalized'].nunique():,}"
        )
        tag_columns[2].metric(
            "Tagged movies", f"{tag_frame[self.config.item_column].nunique():,}"
        )
        if self.config.user_column in tag_frame.columns:
            tag_columns[3].metric(
                "Tagging users", f"{tag_frame[self.config.user_column].nunique():,}"
            )
        else:
            tag_columns[3].metric("Tagging users", "n/a")

        top_tags = (
            tag_frame["tag_normalized"]
            .value_counts()
            .head(25)
            .rename_axis("tag")
            .reset_index(name="uses")
        )
        fig = px.bar(
            top_tags.sort_values("uses"),
            x="uses",
            y="tag",
            orientation="h",
            title="Most Common Community Tags",
            labels={"uses": "Tag uses", "tag": "Tag"},
        )
        st.plotly_chart(fig, use_container_width=True)

        movie_tag_stats = (
            tag_frame.groupby(self.config.item_column)
            .agg(
                tag_count=("tag_normalized", "count"),
                unique_tags=("tag_normalized", "nunique"),
            )
            .reset_index()
            .merge(movies, on=self.config.item_column, how="left")
            .sort_values(["tag_count", "unique_tags"], ascending=False)
            .head(20)
            .rename(columns={self.config.item_column: "movie_id"})
        )
        if poster_url_by_movie_id is not None:
            movie_tag_stats["poster"] = movie_tag_stats["movie_id"].map(
                lambda movie_id: poster_url_by_movie_id(int(movie_id))
            )
        display_columns = [
            self.config.title_column,
            self.config.genres_column,
            "tag_count",
            "unique_tags",
        ]
        if (
            "poster" in movie_tag_stats.columns
            and movie_tag_stats["poster"].notna().any()
        ):
            display_columns.insert(0, "poster")
        st.markdown("#### Most Tagged Movies")
        st.dataframe(
            movie_tag_stats[display_columns],
            column_config={
                "poster": st.column_config.ImageColumn("Poster", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )

        tag_timeline = self.tags_with_datetime(tag_frame)
        if tag_timeline.empty:
            return
        tag_timeline = (
            tag_timeline.assign(year=tag_timeline["tagged_at"].dt.year)
            .groupby("year")
            .size()
            .reset_index(name="tags")
        )
        fig = px.line(
            tag_timeline,
            x="year",
            y="tags",
            markers=True,
            title="Tags Over Time",
            labels={"year": "Year", "tags": "Tag count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    def movie_stats(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
        stats = (
            ratings.groupby(self.config.item_column)[self.config.rating_column]
            .agg(rating_count="count", mean_rating="mean")
            .reset_index()
            .merge(movies, on=self.config.item_column, how="left")
        )
        if stats.empty:
            return stats
        stats["primary_genre"] = stats[self.config.genres_column].map(
            lambda genres: MovieLensPreprocessor.split_genres(str(genres))[0]
        )
        return stats

    def ratings_with_datetime(self, ratings: pd.DataFrame) -> pd.DataFrame:
        if self.config.timestamp_column in ratings.columns:
            dated = ratings.copy()
            dated["rated_at"] = pd.to_datetime(
                dated[self.config.timestamp_column], unit="s", errors="coerce"
            )
            return dated.dropna(subset=["rated_at"])
        return ratings.iloc[0:0].copy()

    def tags_with_datetime(self, tags: pd.DataFrame) -> pd.DataFrame:
        if self.config.timestamp_column in tags.columns:
            dated = tags.copy()
            dated["tagged_at"] = pd.to_datetime(
                dated[self.config.timestamp_column], unit="s", errors="coerce"
            )
            return dated.dropna(subset=["tagged_at"])
        return tags.iloc[0:0].copy()

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

    def render_genre_summary(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
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
        self.format_horizontal_bar_chart(fig, len(genre_frame.head(20)))
        st.plotly_chart(fig, use_container_width=True)

        genre_movies = movies[[self.config.item_column, self.config.genres_column]].copy()
        genre_movies["genre"] = genre_movies[self.config.genres_column].map(
            lambda genres: MovieLensPreprocessor.split_genres(str(genres))
        )
        genre_movies = genre_movies.explode("genre")
        rated_genres = ratings[
            [self.config.item_column, self.config.rating_column]
        ].merge(genre_movies[[self.config.item_column, "genre"]], on=self.config.item_column)
        if rated_genres.empty:
            return

        genre_stats = (
            rated_genres.groupby("genre")[self.config.rating_column]
            .agg(ratings="count", mean_rating="mean")
            .reset_index()
            .merge(genre_frame, on="genre", how="left")
        )
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                genre_stats.sort_values("ratings").tail(20),
                x="ratings",
                y="genre",
                orientation="h",
                title="Top Genres by Rating Volume",
                labels={"ratings": "Ratings", "genre": "Genre"},
            )
            self.format_horizontal_bar_chart(
                fig, len(genre_stats.sort_values("ratings").tail(20))
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                genre_stats,
                x="ratings",
                y="mean_rating",
                size="movies",
                hover_name="genre",
                log_x=True,
                title="Genre Volume vs Average Rating",
                labels={
                    "ratings": "Ratings",
                    "mean_rating": "Mean rating",
                    "movies": "Catalog movies",
                },
            )
            fig.update_yaxes(range=[self.config.rating_min, self.config.rating_max])
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def format_horizontal_bar_chart(fig, row_count: int) -> None:
        fig.update_layout(
            height=max(460, 30 * row_count + 120),
            margin={"l": 120, "r": 24, "t": 80, "b": 48},
            yaxis={"automargin": True, "categoryorder": "total ascending"},
        )
