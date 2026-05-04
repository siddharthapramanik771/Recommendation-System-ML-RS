# Data

The training code expects MovieLens CSV files from GroupLens.

Recommended placement:

```text
data/ml-latest-small/ratings.csv
data/ml-latest-small/movies.csv
```

Required columns:

- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`
- `movies.csv`: `movieId`, `title`, `genres`

Optional enrichment files:

- `links.csv`: `movieId`, `imdbId`, `tmdbId`
- `tags.csv`: `userId`, `movieId`, `tag`, `timestamp`

The app can use `links.csv` plus a TMDB secret (`TMDB_READ_ACCESS_TOKEN`,
`TMDB_BEARER_TOKEN`, or `TMDB_API_KEY`) to show poster thumbnails.

The repository includes a tiny `data/sample/` dataset so the pipeline and app can
be smoke-tested without downloading anything. For portfolio metrics, use the real
MovieLens dataset.

## TMDB snapshots

If only TMDB credentials are available, the trainer can fetch a public movie
catalog and genre map from TMDB:

```powershell
$env:TMDB_API_KEY="your_key_here"
python -m src.train
```

By default, TMDB snapshots are written to:

```text
data/tmdb/ratings.csv
data/tmdb/movies.csv
data/tmdb/links.csv
```

TMDB provides aggregate public votes, not individual user histories. The generated
`ratings.csv` is a pseudo-interaction dataset for demos and scheduled refreshes;
it is not a real personalized collaborative-filtering dataset.
