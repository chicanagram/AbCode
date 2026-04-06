# Model settings repository

Drop one JSON file per model, named `<model_name>.json`.

Supported shapes:

```json
{
  "n_estimators": 300,
  "random_state": 42
}
```

or task-specific:

```json
{
  "classification": {"n_estimators": 300, "random_state": 42},
  "regression": {"n_estimators": 300, "random_state": 42}
}
```

Current model keys:
`rf`, `random_forest`, `ridge`, `mlp_sklearn`, `mlp_pytorch`, `xgboost`, `lightgbm`.
