# Recommendation System (ML/RS)

End-to-end collaborative filtering project for MovieLens recommendations. The
structure follows the same portfolio pattern as the referenced ML projects:
offline training saves a reusable artifact, and Streamlit loads that artifact for
interactive movie suggestions and model evaluation.

## Overview

```text
MovieLens ratings and movie metadata
  -> data cleaning
  -> sparse user-item interaction handling
  -> leave-one-relevant-rating-out split
  -> biased matrix factorization training
  -> precision@K ranking evaluation
  -> artifact persistence
  -> Streamlit movie recommendation demo
```

## Problem Definition

The goal is top-N recommendation: for a selected user, rank movies they have not
already rated and return the most relevant suggestions.

The project treats ratings greater than or equal to `4.0` as relevant for ranking
evaluation.

```text
rating >= 4.0 -> relevant
rating < 4.0  -> not relevant for held-out ranking metrics
```

## Data Contract

Default MovieLens location:

```text
data/ml-latest-small/ratings.csv
data/ml-latest-small/movies.csv
```

Required columns:

- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`
- `movies.csv`: `movieId`, `title`, `genres`

Optional app enrichment:

- `links.csv`: `movieId`, `imdbId`, `tmdbId` for IMDb/TMDB links
- `tags.csv`: `userId`, `movieId`, `tag`, `timestamp` for community tag context

Movie posters are loaded from TMDB when one of these values is available as an
environment variable or Streamlit secret:

- `TMDB_READ_ACCESS_TOKEN` or `TMDB_BEARER_TOKEN`
- `TMDB_API_KEY`

For Streamlit Community Cloud, add the key in the app's **Settings -> Secrets**,
for example:

```toml
TMDB_API_KEY = "your_key_here"
```

GitHub repository secrets are only exposed to GitHub Actions workflows; they are
not automatically available to the running Streamlit app.

Training uses `data/ml-latest-small/` by default. TMDB is used only by the
Streamlit app to fetch poster images when `links.csv` provides TMDB IDs.

## Repository Layout

```text
.
|-- .github/
|   `-- workflows/
|       |-- ci.yml                    # Compile and MovieLens training workflow
|       `-- monthly-movielens-training.yml # Scheduled MovieLens retraining
|-- app/
|   |-- app.py                        # Streamlit UI and app services
|   |-- data_analysis.py              # MovieLens EDA views
|   |-- styles.py                     # Streamlit styling
|   `-- training_methodology.py       # Saved metrics and methodology view
|-- data/
|   |-- sample/                       # Tiny local smoke-test dataset
|   `-- README.md                     # MovieLens placement note
|-- models/
|   `-- README.md                     # Artifact output note
|-- notebooks/
|   `-- README.md
|-- src/
|   |-- config.py                     # Runtime paths and MovieLens schema
|   |-- evaluation.py                 # Precision@K, recall@K, hit rate, NDCG
|   |-- matrix_factorization.py       # Biased matrix factorization model
|   |-- model_bundle.py               # Artifact contract and joblib persistence
|   |-- predict.py                    # Artifact-backed recommendation service
|   |-- preprocessing.py              # Cleaning and ranking split
|   |-- train.py                      # Offline training workflow
|   `-- training_settings.py          # Model and split settings
|-- Dockerfile.streamlit
|-- docker-compose.yml
|-- requirements.txt
|-- requirements-train.txt
`-- streamlit_app.py                  # Streamlit entrypoint
```

## Run Locally

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Train the recommender:

```powershell
python -m src.train
```

Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Training

Training is implemented in `src/train.py`.

The workflow:

1. Load ratings and movie metadata.
2. Clean IDs, ratings, timestamps, titles, and genres.
3. Filter users and movies with too few interactions.
4. Hold out each eligible user's latest relevant rating.
5. Train matrix factorization on the remaining sparse interactions.
6. Evaluate ranking quality with precision@K and related metrics.
7. Retrain on all filtered ratings.
8. Save `models/model.joblib` and `models/training_metrics.json`.

### Model Structure

The model is biased matrix factorization trained with stochastic gradient
descent:

```text
predicted_rating =
  global_mean
  + user_bias
  + movie_bias
  + dot(user_factors, movie_factors)
```

This learns compact latent representations for users and movies without
materializing the dense user-item matrix.

Default settings are stored in `src/training_settings.py`:

- latent factors: `32`
- epochs: `30`
- learning rate: `0.015`
- L2 regularization: `0.08`
- ranking metrics: `precision@5`, `precision@10`, `precision@20`

Change settings with environment variables:

```powershell
$env:RECOMMENDER_LATENT_FACTORS="64"
$env:RECOMMENDER_EPOCHS="40"
$env:RECOMMENDER_LEARNING_RATE="0.01"
$env:RECOMMENDER_K_VALUES="5,10,20"
python -m src.train
```

Use explicit MovieLens paths:

```powershell
$env:RECOMMENDER_RATINGS_PATH="data/ml-latest-small/ratings.csv"
$env:RECOMMENDER_MOVIES_PATH="data/ml-latest-small/movies.csv"
python -m src.train
```

The monthly retraining workflow is defined in
`.github/workflows/monthly-movielens-training.yml`. It runs at `03:00 UTC` on
the first day of each month, uploads the trained model as a workflow artifact,
and commits refreshed `models/model.joblib` and `models/training_metrics.json`
back to the repository.

## Evaluation

The project reports ranking metrics instead of only rating-prediction error:

- `precision@K`: fraction of recommendation slots that contain held-out relevant movies
- `recall@K`: fraction of held-out relevant movies recovered
- `hit_rate@K`: fraction of users with at least one relevant hit
- `ndcg@K`: ranking quality with higher credit for earlier hits
- `catalog_coverage@K`: portion of the movie catalog surfaced across users

Metrics are generated dynamically during training and saved to:

```text
models/training_metrics.json
```

## Dashboard

The Streamlit app includes:

- user-based movie recommendations
- top-K controls
- user's highest-rated history
- movie-to-movie suggestions from learned item factors, community tags, and IMDb/TMDB links
- optional TMDB poster thumbnails for selected movies and suggestion rows
- sparse matrix density and catalog statistics
- rating distributions, time trends, user activity, catalog quality, genre, and tag analysis
- saved precision@K methodology and metrics
- model and metrics downloads

## Docker

```powershell
docker-compose up --build
```

Then open:

```text
http://localhost:8501
```

## Notes

This is a portfolio and learning project. Recommendations are generated from the
available MovieLens-style interactions and are not a production personalization
system by default.
