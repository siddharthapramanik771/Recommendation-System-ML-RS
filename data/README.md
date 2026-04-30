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

The repository includes a tiny `data/sample/` dataset so the pipeline and app can
be smoke-tested without downloading anything. For portfolio metrics, use the real
MovieLens dataset.

