# Recommendation System Code Flow

This document explains how the project works end to end: data contracts, module
responsibilities, training logic, recommendation logic, evaluation, saved
artifacts, Streamlit runtime behavior, poster enrichment, and GitHub workflows.

## System Goal

The project is a MovieLens-style recommendation system. It trains a collaborative
filtering model from explicit user ratings and uses the trained artifact to
recommend movies in a Streamlit dashboard.

The core problem is top-N recommendation:

```text
Given a user and a catalog of movies,
rank movies the user has not already rated,
then return the top K movies.
```

The project does not train on TMDB data. TMDB is used only at dashboard runtime
to fetch poster images when MovieLens `links.csv` contains a TMDB ID and a TMDB
credential is available.

## Runtime Inputs

Default training data:

```text
data/ml-latest-small/ratings.csv
data/ml-latest-small/movies.csv
```

Optional dashboard enrichment:

```text
data/ml-latest-small/links.csv
data/ml-latest-small/tags.csv
```

Smoke-test fallback data:

```text
data/sample/ratings.csv
data/sample/movies.csv
```

Saved runtime outputs:

```text
models/model.joblib
models/training_metrics.json
```

## Data Contract

`ratings.csv` must contain:

```text
userId, movieId, rating, timestamp
```

Required semantics:

- `userId`: integer user identifier.
- `movieId`: integer movie/item identifier.
- `rating`: explicit user rating. The default accepted range is `0.5` to `5.0`.
- `timestamp`: rating time. Numeric timestamps are accepted; ISO-like datetime
  strings are also cleaned into seconds.

`movies.csv` must contain:

```text
movieId, title, genres
```

Required semantics:

- `movieId`: integer movie identifier matching ratings.
- `title`: display title.
- `genres`: pipe-separated genre string, for example `Action|Comedy`.

`links.csv` is optional and can contain:

```text
movieId, imdbId, tmdbId
```

This file is used by the dashboard for IMDb/TMDB links and poster lookup. It is
not used by the training algorithm.

`tags.csv` is optional and can contain:

```text
userId, movieId, tag, timestamp
```

Tags are used for dashboard context and movie-to-movie suggestion enrichment.
They are not used directly by matrix factorization training.

## Module Map

`src/config.py`

- Defines `RuntimeConfig`, the central path and schema contract.
- Knows where data, model artifacts, and metrics are located.
- Caches the selected default ratings/movies path pair through
  `default_dataset_paths` so the same path resolution is not repeated during a
  run.
- Provides simple CSV loading helpers.
- Creates `RUNTIME_CONFIG`, the default runtime configuration.

`src/training_settings.py`

- Defines `TrainingSettings`.
- Stores model hyperparameters and evaluation settings.
- Converts settings to a JSON-safe dictionary for saved metrics.

`src/preprocessing.py`

- Cleans ratings, movies, links, and tags.
- Filters sparse interactions.
- Builds the leave-one-relevant-out train/test split.
- Computes catalog statistics used in metrics and the dashboard.

`src/matrix_factorization.py`

- Implements biased matrix factorization trained with stochastic gradient
  descent.
- Scores known and unknown users/items.
- Produces user recommendations and item-to-item similarity.

`src/evaluation.py`

- Computes ranking metrics: precision@K, recall@K, hit rate@K, NDCG@K, and
  catalog coverage@K.

`src/model_bundle.py`

- Defines the persisted model artifact contract.
- Saves and loads model artifacts with `joblib`.

`src/train.py`

- Orchestrates the offline training workflow.
- Loads data, cleans it, filters it, evaluates it, trains the final model, saves
  the model, and writes metrics.

`src/predict.py`

- Loads the persisted artifact.
- Provides recommendation and prediction services for the Streamlit app.

`app/app.py`

- Streamlit dashboard entrypoint logic.
- Loads reference data and model artifacts.
- Renders recommendations, movie-to-movie suggestions, poster images, and
  external links.

`app/data_analysis.py`

- Renders exploratory data analysis views.

`app/training_methodology.py`

- Reads saved training metrics and explains model/evaluation results in the UI.

`streamlit_app.py`

- Small Streamlit entrypoint that calls the app package.

## Configuration Flow

The default configuration is created by:

```python
RUNTIME_CONFIG = RuntimeConfig.from_project_root()
```

This sets paths relative to the repository root:

```text
ratings_path -> data/ml-latest-small/ratings.csv
movies_path  -> data/ml-latest-small/movies.csv
links_path   -> data/ml-latest-small/links.csv
tags_path    -> data/ml-latest-small/tags.csv
model_path   -> models/model.joblib
metrics_path -> models/training_metrics.json
```

Training can be run with no arguments:

```powershell
python -m src.train
```

By default, this trains on `data/ml-latest-small`. Optional environment
variables can override settings:

```text
RECOMMENDER_RATINGS_PATH
RECOMMENDER_MOVIES_PATH
RECOMMENDER_USE_SAMPLE
RECOMMENDER_RANDOM_STATE
RECOMMENDER_LATENT_FACTORS
RECOMMENDER_EPOCHS
RECOMMENDER_LEARNING_RATE
RECOMMENDER_REGULARIZATION
RECOMMENDER_MIN_USER_RATINGS
RECOMMENDER_MIN_MOVIE_RATINGS
RECOMMENDER_K_VALUES
RECOMMENDER_RECOMMENDATION_COUNT
```

`RECOMMENDER_USE_SAMPLE=true` makes training use `data/sample`. This exists for
smoke tests only.

## Training Flow

Training starts in `src/train.py`:

```text
main()
  -> resolve_training_inputs()
  -> settings_from_env()
  -> RecommenderTrainer.run()
```

`resolve_training_inputs()` reads optional data path overrides. If no overrides
are provided, the trainer uses `RuntimeConfig.default_dataset_paths`, which is
cached per config object. `RuntimeConfig.resolve_dataset_paths()` is kept for
explicit custom path pairs and `RECOMMENDER_USE_SAMPLE=true`.

`settings_from_env()` creates `TrainingSettings` from defaults plus optional
environment overrides.

`RecommenderTrainer.run()` does the full offline model lifecycle:

```text
1. Create runtime output directories.
2. Resolve ratings/movies paths.
3. Load raw CSV files.
4. Clean ratings and movies.
5. Remove ratings for movies that are not in the movie catalog.
6. Filter users/movies with too few interactions.
7. Split data for ranking evaluation.
8. Train an evaluation model on the split train set.
9. Evaluate ranking metrics on the split test set.
10. Train the final model on all filtered ratings.
11. Save the final model artifact.
12. Save training metrics JSON.
13. Print a concise training summary.
```

## Cleaning Logic

Ratings cleaning:

```text
raw ratings
  -> trim column names
  -> require userId/movieId/rating
  -> keep only userId/movieId/rating/timestamp
  -> coerce userId and movieId to numeric IDs
  -> coerce rating to float
  -> coerce timestamp to numeric seconds
  -> drop rows with missing required values
  -> cast userId/movieId to int
  -> keep ratings between rating_min and rating_max
  -> sort by timestamp
  -> keep the latest user/movie rating when duplicates exist
```

Movie cleaning:

```text
raw movies
  -> trim column names
  -> require movieId/title/genres
  -> coerce movieId to integer
  -> replace missing title with Untitled
  -> replace missing/no genres with Unknown
  -> drop duplicate movieId rows
```

Links cleaning:

```text
raw links
  -> require movieId
  -> keep movieId plus optional imdbId/tmdbId
  -> normalize IMDb IDs to 7-digit strings
  -> coerce TMDB IDs to nullable integers
```

Tags cleaning:

```text
raw tags
  -> require movieId and tag
  -> keep optional userId/timestamp when present
  -> drop blank tags
  -> coerce movieId/userId/timestamp where applicable
```

## Interaction Filtering

Collaborative filtering performs poorly when many users or items have almost no
interactions. The project applies an iterative filter:

```text
while rows are still being removed:
  keep users with at least min_user_ratings interactions
  keep movies with at least min_movie_ratings interactions
```

This is iterative because removing unpopular movies can make a user fall below
the user threshold, and removing sparse users can make a movie fall below the
movie threshold.

Defaults:

```text
min_user_ratings = 4
min_movie_ratings = 1
```

## Evaluation Split

The project uses a leave-one-relevant-rating-out split:

```text
For each eligible user:
  find ratings where rating >= relevance_threshold
  hold out the latest relevant rating as test
  keep all other ratings as train
```

Default relevance threshold:

```text
rating >= 4.0
```

Why this split is used:

- It evaluates recommendation ranking, not just rating prediction.
- It simulates hiding one item the user liked.
- It asks whether the model can recover that hidden item in the top-K list.

## Matrix Factorization Concept

The model is biased matrix factorization.

Each user receives a latent vector:

```text
user_factors[user]
```

Each movie receives a latent vector:

```text
item_factors[movie]
```

The model also learns:

```text
global_mean
user_bias[user]
item_bias[movie]
```

Prediction formula:

```text
predicted_rating =
  global_mean
  + user_bias[user]
  + item_bias[movie]
  + dot(user_factors[user], item_factors[movie])
```

The dot product captures user/movie compatibility in a compact latent space.
The bias terms capture systematic user and item tendencies.

Examples:

- A generous user may have a positive user bias.
- A broadly liked movie may have a positive item bias.
- A user vector close to an item vector means the model expects affinity.

## Training Algorithm

Training uses stochastic gradient descent over observed ratings only. The dense
user-item matrix is never materialized.

For each epoch:

```text
shuffle observed ratings
for each observed rating:
  predict rating
  error = actual_rating - predicted_rating
  update user bias
  update item bias
  update user latent vector
  update item latent vector
```

Regularization is applied to reduce overfitting:

```text
updated_parameter += learning_rate * (error_signal - regularization * parameter)
```

Defaults:

```text
latent_factors = 32
epochs = 30
learning_rate = 0.015
regularization = 0.08
random_state = 42
```

## Cold-Start Behavior

When the user is unknown:

```text
score each candidate movie by popularity-adjusted mean rating
```

When a movie is unknown:

```text
fall back toward global_mean
```

Popularity score uses shrinkage:

```text
score =
  (count / (count + shrinkage)) * movie_mean
  + (shrinkage / (count + shrinkage)) * global_mean
```

This prevents a movie with very few ratings from ranking too high only because
of a small number of strong ratings.

## Recommendation Flow

For a known user:

```text
candidate_movie_ids = all catalog movies
seen_movie_ids = movies the user has already rated
unseen_candidates = candidate_movie_ids - seen_movie_ids
score every unseen candidate
sort by predicted rating descending
return top K
```

For a movie-to-movie suggestion:

```text
get source movie latent vector
compute cosine similarity against candidate movie vectors
sort by similarity descending
return top K
```

Cosine similarity measures whether two movie vectors point in a similar
direction in latent space.

## Evaluation Metrics

Evaluation ranks candidate movies for each evaluated user and compares the top-K
recommendations against held-out relevant movies.

Precision@K:

```text
hits_in_top_k / k
```

Interpretation:

```text
Of the K recommended slots, how many were relevant?
```

Recall@K:

```text
hits_in_top_k / number_of_relevant_test_items
```

Interpretation:

```text
Of the hidden relevant items, how many did the model recover?
```

Hit Rate@K:

```text
1 if at least one relevant item appears in top K else 0
```

Interpretation:

```text
Did the recommender produce at least one useful hit?
```

NDCG@K:

```text
discounted gain / ideal discounted gain
```

Interpretation:

```text
Hits near the top count more than hits near the bottom.
```

Catalog Coverage@K:

```text
unique recommended movies across users / total catalog movies
```

Interpretation:

```text
How much of the catalog does the recommender surface?
```

## Model Artifact

Training saves a `ModelArtifact` through `ModelArtifactRepository`.

Artifact contents:

```text
artifact_version
model_name
recommender
movies
seen_movie_ids_by_user
training_summary
relevance_threshold
```

`recommender` contains:

```text
user_id_to_index
movie_id_to_index
user_factors
item_factors
user_bias
item_bias
global_mean
movie_means
movie_counts
rating_min
rating_max
```

`movies` is the cleaned movie catalog. It is stored with the model so prediction
does not need to reload the raw movie CSV to enrich recommendations with titles
and genres.

`seen_movie_ids_by_user` lets the app exclude already-rated movies from user
recommendations.

## Metrics JSON

Training writes:

```text
models/training_metrics.json
```

Top-level fields:

```text
model_name
ratings_path
movies_path
artifact_path
metrics_path
settings
catalog
split
ranking
```

`settings` records training hyperparameters.

`catalog` records counts, density, sparsity, and mean rating.

`split` records train/test sizes and relevance threshold.

`ranking` records precision, recall, hit rate, NDCG, coverage, and evaluated
user count.

## Dashboard Flow

The Streamlit app starts from:

```text
streamlit_app.py
  -> DashboardRenderer().render()
```

The dashboard does this:

```text
1. Set Streamlit page configuration.
2. Apply CSS styling.
3. Load reference MovieLens data.
4. Render sidebar status and artifact controls.
5. Load model artifact when recommendations are requested.
6. Render movie suggestions, user recommendations, data analysis, and methodology tabs.
```

Reference data loading:

```text
ReferenceDataService.load()
  -> resolve MovieLens ratings/movies paths
  -> clean ratings
  -> clean movies
  -> keep ratings whose movieId exists in movies
  -> optionally clean links.csv
  -> optionally clean tags.csv
```

User recommendation tab:

```text
select user
select K
load artifact
recommend unseen movies
enrich rows with title/genre/rating stats
optionally add poster URLs
display dataframe
optionally show user rating history
```

Movie suggestion tab:

```text
select source movie
load artifact
compute similar movies from item factors
blend factor similarity with tag overlap for display sorting
optionally add poster URLs
display dataframe and external links
```

Data analysis tab:

```text
render catalog/rating/user/genre/tag summaries
use poster URL callback when posters are available
```

Training methodology tab:

```text
read models/training_metrics.json
display model formula
display precision@K explanation
display saved settings, catalog stats, and metrics
```

## TMDB Poster Flow

TMDB is not part of training. It is only a poster enrichment service in the app.

Poster lookup requires:

```text
links.csv has tmdbId
and
one of these values is set:
  TMDB_READ_ACCESS_TOKEN
  TMDB_BEARER_TOKEN
  TMDB_API_KEY
```

Flow:

```text
movieId
  -> links.csv lookup
  -> tmdbId
  -> GET https://api.themoviedb.org/3/movie/{tmdb_id}
  -> poster_path
  -> https://image.tmdb.org/t/p/w342/{poster_path}
```

The app caches poster lookups with Streamlit cache data for one day.

GitHub repository secrets are not automatically visible to Streamlit Cloud. For
posters in a deployed Streamlit app, add the TMDB credential to Streamlit secrets
or the runtime environment.

## GitHub Workflow Flow

CI workflow:

```text
.github/workflows/ci.yml
```

Runs on:

```text
push to main
pull_request
manual workflow_dispatch
```

Steps:

```text
checkout
set up Python 3.12
install requirements-train.txt
compile app/src
train MovieLens recommender
upload model artifacts
commit refreshed model on non-PR runs
```

The CI workflow ignores pushes that only change:

```text
models/model.joblib
models/training_metrics.json
```

This prevents generated model commits from causing an endless workflow loop.

Monthly workflow:

```text
.github/workflows/monthly-movielens-training.yml
```

Runs at:

```text
03:00 UTC on the first day of each month
```

Steps:

```text
checkout
set up Python 3.12
install training dependencies
compile source
train from MovieLens data
upload model and metrics
commit refreshed model and metrics
```

## Local Development Commands

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Train:

```powershell
python -m src.train
```

Train with sample data:

```powershell
$env:RECOMMENDER_USE_SAMPLE="true"
python -m src.train
```

Run dashboard:

```powershell
streamlit run streamlit_app.py
```

Run compile check:

```powershell
python -m compileall app src streamlit_app.py
```

## Design Notes

The model is intentionally compact:

- No external recommendation library is required.
- The artifact is portable through `joblib`.
- The app can run without retraining.
- TMDB remains optional and presentation-only.
- The default dataset is MovieLens, which has the real user-item interactions
  needed for collaborative filtering.

The most important boundary in the project is:

```text
training uses MovieLens ratings
dashboard poster enrichment optionally uses TMDB
```

Keeping that boundary clear prevents aggregate catalog metadata from being
mistaken for real personalization data.
