from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from abcode.tools.ml.metrics import compute_metrics
from abcode.tools.ml.model_registry import (
    adapt_tuned_params_for_task,
    build_model,
    get_tuning_space,
)


TUNING_RANDOM_SEED = 42
TUNING_TIMEOUT_SEC = 3600


_MAXIMIZE_METRICS = {
    "accuracy",
    "auc",
    "f1",
    "f1_weighted",
    "mcc",
    "pearson",
    "r2",
    "spearman",
}


def resolve_tuning_metric(*, task_type: str, tuning_metric: str) -> str:
    metric = str(tuning_metric or "").strip().lower()
    if metric:
        return metric
    return "spearman" if str(task_type).strip().lower() == "regression" else "mcc"


def _resolve_direction(metric_name: str) -> str:
    return "maximize" if metric_name.lower() in _MAXIMIZE_METRICS else "minimize"


def _suggest_from_spec(trial: Any, name: str, spec: Dict[str, Any]) -> Any:
    t = str(spec.get("type", "")).strip().lower()
    if t == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if t == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step)
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported tuning spec type '{t}' for '{name}'.")


def _split_train_validation(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y_train)
    if n < 10:
        raise ValueError("Insufficient training samples for robust tuning (n_train < 10).")

    stratify = None
    if str(task_type).strip().lower() == "classification":
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) > 1 and int(np.min(counts)) >= 2:
            stratify = y_train

    return train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )


def tune_model_hyperparameters(
    *,
    model_name: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_params: Dict[str, Any],
    tuning_metric: str,
    n_trials: int,
    show_progress: bool = False,
) -> Dict[str, Any]:
    try:
        import optuna
    except Exception as exc:
        raise ImportError("Optuna is required for hyperparameter tuning but is not installed.") from exc

    metric_name = resolve_tuning_metric(task_type=task_type, tuning_metric=tuning_metric)
    direction = _resolve_direction(metric_name)
    n_trials = max(int(n_trials), 1)

    X_subtrain, X_val, y_subtrain, y_val = _split_train_validation(
        X_train=X_train,
        y_train=y_train,
        task_type=task_type,
        seed=TUNING_RANDOM_SEED,
    )
    tuning_space = get_tuning_space(model_name=model_name, task_type=task_type)
    if not tuning_space:
        return {
            "best_params": dict(base_params),
            "metric_name": metric_name,
            "direction": direction,
            "n_trials": 0,
            "best_value": np.nan,
            "used_tuning": False,
            "reason": "no_tuning_space",
        }

    def objective(trial: Any) -> float:
        tuned_raw = {k: _suggest_from_spec(trial, k, spec) for k, spec in tuning_space.items()}
        tuned_effective = adapt_tuned_params_for_task(
            model_name=model_name,
            task_type=task_type,
            tuned_params=tuned_raw,
        )
        params = dict(base_params)
        params.update(tuned_effective)

        model = build_model(model_name=model_name, task_type=task_type, params=params)
        model.fit(X_subtrain, y_subtrain)
        y_pred = model.predict(X_val)

        y_score = None
        if str(task_type).strip().lower() == "classification" and hasattr(model, "predict_proba"):
            try:
                y_score = model.predict_proba(X_val)
            except Exception:
                y_score = None
        scores = compute_metrics(
            y_true=y_val,
            y_pred=y_pred,
            task_type=task_type,
            y_score=y_score,
        )
        if metric_name not in scores:
            raise KeyError(
                f"Tuning metric '{metric_name}' is unavailable. Available metrics: {sorted(scores.keys())}"
            )
        trial.set_user_attr("effective_params", params)
        return float(scores[metric_name])

    sampler = optuna.samplers.TPESampler(seed=TUNING_RANDOM_SEED)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def _trial_callback(study_obj: Any, trial_obj: Any) -> None:
        if not show_progress:
            return
        print(
            "[tuning-trial] "
            f"model={model_name} | metric={metric_name} | trial={trial_obj.number + 1}/{n_trials} | "
            f"value={trial_obj.value:.6f} | best={study_obj.best_value:.6f}"
        )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=TUNING_TIMEOUT_SEC,
        show_progress_bar=False,
        callbacks=[_trial_callback],
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs.get("effective_params", dict(base_params)))

    if show_progress:
        print(
            "[tuning-complete] "
            f"model={model_name} | metric={metric_name} | direction={direction} | "
            f"best_value={best_trial.value:.6f} | n_trials={len(study.trials)}"
        )
        print(
            "[tuning-selected] "
            f"model={model_name} | selected_hyperparameters={best_params}"
        )

    return {
        "best_params": best_params,
        "metric_name": metric_name,
        "direction": direction,
        "n_trials": int(len(study.trials)),
        "best_value": float(best_trial.value),
        "used_tuning": True,
    }


def select_best_params_from_candidates(
    *,
    model_name: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_params: Dict[str, Any],
    candidate_params_list: list[Dict[str, Any]],
    tuning_metric: str,
    show_progress: bool = False,
) -> Dict[str, Any]:
    metric_name = resolve_tuning_metric(task_type=task_type, tuning_metric=tuning_metric)
    direction = _resolve_direction(metric_name)

    X_subtrain, X_val, y_subtrain, y_val = _split_train_validation(
        X_train=X_train,
        y_train=y_train,
        task_type=task_type,
        seed=TUNING_RANDOM_SEED,
    )

    if not candidate_params_list:
        return {
            "best_params": dict(base_params),
            "metric_name": metric_name,
            "direction": direction,
            "n_trials": 0,
            "best_value": np.nan,
            "used_tuning": False,
            "reason": "no_candidate_params",
        }

    best_params = dict(base_params)
    best_value = -np.inf if direction == "maximize" else np.inf

    for i, cand in enumerate(candidate_params_list, start=1):
        params = dict(base_params)
        params.update(dict(cand))
        model = build_model(model_name=model_name, task_type=task_type, params=params)
        model.fit(X_subtrain, y_subtrain)
        y_pred = model.predict(X_val)

        y_score = None
        if str(task_type).strip().lower() == "classification" and hasattr(model, "predict_proba"):
            try:
                y_score = model.predict_proba(X_val)
            except Exception:
                y_score = None
        scores = compute_metrics(
            y_true=y_val,
            y_pred=y_pred,
            task_type=task_type,
            y_score=y_score,
        )
        if metric_name not in scores:
            raise KeyError(
                f"Tuning metric '{metric_name}' is unavailable. Available metrics: {sorted(scores.keys())}"
            )
        value = float(scores[metric_name])
        improved = (value > best_value) if direction == "maximize" else (value < best_value)
        if improved:
            best_value = value
            best_params = params

        if show_progress:
            print(
                "[preset-search] "
                f"model={model_name} | metric={metric_name} | candidate={i}/{len(candidate_params_list)} | "
                f"value={value:.6f} | best={best_value:.6f}"
            )

    if show_progress:
        print(
            "[preset-search-selected] "
            f"model={model_name} | metric={metric_name} | selected_hyperparameters={best_params}"
        )

    return {
        "best_params": best_params,
        "metric_name": metric_name,
        "direction": direction,
        "n_trials": int(len(candidate_params_list)),
        "best_value": float(best_value),
        "used_tuning": True,
    }
