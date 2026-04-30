# Models

Training writes the recommender artifact and ranking metrics here:

```text
models/model.joblib
models/training_metrics.json
```

Run:

```powershell
python -m src.train
```

If the real MovieLens files are not present, the trainer falls back to the tiny
sample dataset included in `data/sample/`.

