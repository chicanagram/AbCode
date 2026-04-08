from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from abcode.tools.ml.metrics import compute_metrics
from abcode.tools.ml.model_registry import (
    build_model,
    get_feature_param_preset_candidates,
    get_feature_ntrain_param_preset,
    load_model_params,
)
from abcode.tools.ml.feature_matrix import (
    build_feature_matrix,
    load_dataset_table,
    prepare_dataset_bundle,
)
from abcode.tools.ml.feature_coefficients import (
    extract_coefficients,
    normalize_feature_model_pairs,
    resolve_sequence_base,
    save_coefficients_csv,
    should_extract_coefficients,
)
from abcode.tools.ml.hyperparameter_tuning import (
    resolve_tuning_metric,
    select_best_params_from_candidates,
    tune_model_hyperparameters,
)
from abcode.tools.ml.fit_svd import (
    fit_transform_svd_train_test,
    resolve_svd_feature_config,
)
from abcode.tools.ml.splits import generate_splits, generate_progressive_splits


PROGRESSIVE_SPLIT_KEYS = {"random", "mutres-modulo", "mutres_modulo", "custom", "retrospective", "retrospective-segment", "retrospective_segment"}
CUSTOM_LIKE_KEYS = {"custom", "retrospective", "retrospective-segment", "retrospective_segment"}


def _utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _clean_optional_str(value: Any) -> str:
    s = str(value or "").strip()
    if s.lower() in {"none", "null", "nan"}:
        return ""
    return s


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _append_metrics_row(metrics_csv_path: Path, row: Dict[str, Any]) -> None:
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if metrics_csv_path.exists():
        existing = pd.read_csv(metrics_csv_path)
        out = pd.concat([existing, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(metrics_csv_path, index=False)


def _build_metrics_csv_name(task_type: str, suffix: str) -> str:
    clean_suffix = str(suffix or "").strip()
    return f"{task_type}_metrics{clean_suffix}.csv"


def _build_summary_csv_name(task_type: str, suffix: str) -> str:
    clean_suffix = str(suffix or "").strip()
    return f"{task_type}_metrics_summary{clean_suffix}.csv"



def _parse_segment_index_range(value: Any) -> tuple[int | None, int | None] | None:
    if value is None:
        return None
    if isinstance(value, str) and str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("segment_index_range must be a list/tuple like [start_idx, end_idx].")
    if len(value) == 0:
        return None
    if len(value) != 2:
        raise ValueError("segment_index_range must contain exactly two elements: [start_idx, end_idx].")

    def _to_optional_int(v: Any) -> int | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"", "none", "null", "nan"}:
            return None
        out = int(v)
        if out < 0:
            raise ValueError("segment_index_range values must be >= 0 when provided.")
        return out

    start_idx = _to_optional_int(value[0])
    end_idx = _to_optional_int(value[1])
    if start_idx is None and end_idx is None:
        return None
    if start_idx is not None and end_idx is not None and end_idx < start_idx:
        raise ValueError("segment_index_range end_idx must be >= start_idx.")
    return start_idx, end_idx


def _normalize_hyperparameter_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"", "none", "null", "false", "default"}:
        return "default"
    if mode in {"ntrain_preset", "preset_search", "optuna_search"}:
        return mode
    raise ValueError(
        "hyperparameter_mode must be one of: default, ntrain_preset, preset_search, optuna_search"
    )


def _normalize_summary_source_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"", "run_cache", "cache", "current_run"}:
        return "run_cache"
    if mode in {"all_saved_rows", "saved_rows", "all_rows", "metrics_csv"}:
        return "all_saved_rows"
    raise ValueError("summary_source_mode must be one of: run_cache, all_saved_rows")


def _normalize_summary_metric_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"", "average", "avg", "mean"}:
        return "average"
    if mode in {"pooled"}:
        return "pooled"
    raise ValueError("summary_metric_mode must be one of: average, pooled")


def _build_summary_from_fold_rows(
    *,
    rows_df: pd.DataFrame,
) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    group_cols = ["target_col", "feature_label", "split_type", "model_name", "task_type", "eval_group"]
    for c in group_cols:
        if c not in rows_df.columns:
            rows_df[c] = ""
    metric_cols = [
        c
        for c in rows_df.columns
        if c.startswith("test_")
        and c not in {"test_segment"}
        and pd.api.types.is_numeric_dtype(rows_df[c])
    ]
    out_rows: List[Dict[str, Any]] = []
    grouped = rows_df.groupby(group_cols, dropna=False, sort=False)
    for keys, grp in grouped:
        row = dict(zip(group_cols, keys))
        row["n_test_pooled"] = int(pd.to_numeric(grp.get("n_test"), errors="coerce").fillna(0).sum())
        row["n_train"] = float(pd.to_numeric(grp.get("n_train"), errors="coerce").mean())
        row["n_folds"] = int(len(grp))
        row["selected_hyperparameters"] = _collapse_param_strings(
            [str(x) for x in grp.get("selected_hyperparameters", pd.Series(dtype=object)).tolist()]
        )
        row["hyperparameter_tuning_enabled"] = bool(
            pd.to_numeric(
                grp.get("hyperparameter_tuning_enabled", pd.Series([0] * len(grp), index=grp.index)),
                errors="coerce",
            ).fillna(0).astype(int).max()
        )
        row["hyperparameter_tuning_metric"] = _collapse_param_strings(
            [str(x) for x in grp.get("hyperparameter_tuning_metric", pd.Series(dtype=object)).tolist()]
        )
        for mc in metric_cols:
            row[mc] = round(float(pd.to_numeric(grp[mc], errors="coerce").mean()), 4)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _build_summary_from_run_cache_pooled(
    *,
    pooled_test_cache: Dict[Tuple[str, str, str, str, str], Dict[str, List[Any]]],
    task_type: str,
    hyperparameter_mode: str,
    tuning_metric: str,
) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    for (target_col, feature_label, split_type, model_name, eval_group), cached in pooled_test_cache.items():
        y_true_all = np.concatenate(cached["y_true"], axis=0)
        y_pred_all = np.concatenate(cached["y_pred"], axis=0)
        pooled_metrics = _round_metric_dict(compute_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            task_type=task_type,
            y_score=None,
        ))
        summary_row = {
            "target_col": target_col,
            "feature_label": feature_label,
            "split_type": split_type,
            "model_name": model_name,
            "task_type": task_type,
            "eval_group": eval_group,
            "n_test_pooled": int(len(y_true_all)),
            "n_train": float(np.mean(cached.get("n_train", [np.nan]))),
            "n_folds": int(len(cached["y_true"])),
            "selected_hyperparameters": _collapse_param_strings(cached.get("selected_hyperparameters", [])),
            "hyperparameter_tuning_enabled": bool(hyperparameter_mode != "default"),
            "hyperparameter_tuning_metric": tuning_metric,
        }
        summary_row.update({f"test_{k}": v for k, v in pooled_metrics.items()})
        summary_rows.append(summary_row)
    return pd.DataFrame(summary_rows)


def _build_summary_dataframe(
    *,
    summary_source_mode: str,
    summary_metric_mode: str,
    run_df: pd.DataFrame,
    pooled_test_cache: Dict[Tuple[str, str, str, str, str], Dict[str, List[Any]]],
    metrics_csv_path: Path,
    task_type: str,
    hyperparameter_mode: str,
    tuning_metric: str,
) -> pd.DataFrame:
    if summary_metric_mode == "pooled":
        if summary_source_mode == "all_saved_rows":
            raise ValueError(
                "summary_metric_mode='pooled' is only available with summary_source_mode='run_cache' "
                "because pooled y_true/y_pred arrays are not stored in metrics CSV rows."
            )
        return _build_summary_from_run_cache_pooled(
            pooled_test_cache=pooled_test_cache,
            task_type=task_type,
            hyperparameter_mode=hyperparameter_mode,
            tuning_metric=tuning_metric,
        )

    if summary_source_mode == "run_cache":
        rows_df = run_df.copy()
    else:
        if metrics_csv_path.exists():
            rows_df = pd.read_csv(metrics_csv_path)
        else:
            rows_df = run_df.copy()
    return _build_summary_from_fold_rows(rows_df=rows_df)


def _compose_pooled_pred_name(
    run_label: str,
    feature_label: str,
    split_type: str,
    model_name: str,
    suffix: str,
) -> str:
    if str(suffix or "").strip():
        return f"{run_label}__{feature_label}__{split_type}__{model_name}__all_folds__{suffix}.csv"
    return f"{run_label}__{feature_label}__{split_type}__{model_name}__all_folds.csv"


def _append_fold_predictions(
    *,
    pooled_prediction_rows: Dict[Tuple[str, str, str, str], List[pd.DataFrame]],
    target_col: str,
    feature_label: str,
    split_type: str,
    model_name: str,
    split_id: int,
    eval_group: str,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    dataset_df: pd.DataFrame,
    sample_id_col: str,
) -> None:
    pred_df = pd.DataFrame(
        {
            "row_index": test_idx,
            "split_id": split_id,
            "eval_group": eval_group,
            "y_true": y_test,
            "y_pred": yhat_test,
        }
    )
    if sample_id_col and sample_id_col in dataset_df.columns:
        pred_df.insert(
            1,
            "sample_id",
            dataset_df.iloc[test_idx][sample_id_col].to_numpy(),
        )
    pred_key = (
        str(target_col),
        str(feature_label),
        str(split_type),
        str(model_name),
    )
    pooled_prediction_rows.setdefault(pred_key, []).append(pred_df)


def _write_pooled_predictions(
    *,
    pooled_prediction_rows: Dict[Tuple[str, str, str, str], List[pd.DataFrame]],
    pred_dir: Path,
    run_label: str,
    csv_suffix: str,
    show_progress: bool,
) -> List[str]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    pooled_prediction_paths: List[str] = []
    for (target_col, feature_label, split_type, model_name), fold_pred_dfs in pooled_prediction_rows.items():
        pooled_pred_df = pd.concat(fold_pred_dfs, axis=0, ignore_index=True)
        pooled_pred_df = pooled_pred_df.sort_values(["row_index", "split_id"], kind="stable").reset_index(drop=True)
        pred_path = pred_dir / _compose_pooled_pred_name(
            run_label=run_label,
            feature_label=f"{feature_label}__{target_col}",
            split_type=split_type,
            model_name=model_name,
            suffix=csv_suffix,
        )
        pooled_pred_df.to_csv(pred_path, index=False)
        pooled_prediction_paths.append(str(pred_path))
        if show_progress:
            print(
                "[predictions-saved] "
                f"target_col={target_col} | split_type={split_type} | "
                f"feature_combi_name={feature_label} | model_name={model_name} | path={pred_path}"
            )
    return pooled_prediction_paths


def _round_metric_dict(metrics: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            out[k] = round(float(v), ndigits)
        except Exception:
            out[k] = v
    return out


def _params_to_json(params: Dict[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True, default=str)


def _collapse_param_strings(values: List[str]) -> str:
    uniq = sorted({str(v) for v in values if str(v).strip()})
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]
    return json.dumps(uniq)


def _progress_row(task_type: str, test_metrics: Dict[str, Any], train_metrics: Dict[str, Any]) -> pd.DataFrame:
    if task_type == "regression":
        cols = [
            "test_spearman",
            "test_pearson",
            "test_r2",
            "test_rmse",
            "train_spearman",
            "train_pearson",
            "train_r2",
            "train_rmse",
        ]
    else:
        cols = [
            "test_mcc",
            "test_accuracy",
            "test_f1_weighted",
            "train_mcc",
            "train_accuracy",
            "train_f1_weighted",
        ]
    row: Dict[str, Any] = {}
    for c in cols:
        if c.startswith("test_"):
            row[c] = test_metrics.get(c.replace("test_", ""), np.nan)
        else:
            row[c] = train_metrics.get(c.replace("train_", ""), np.nan)
    return pd.DataFrame([row])


def _drop_invalid_rows_from_xy(
    *,
    X: np.ndarray,
    y: np.ndarray,
    dataset_df: pd.DataFrame,
    show_progress: bool = True,
    context: str = "",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[0] != len(dataset_df):
        raise ValueError("X, y, and dataset_df must have the same number of rows before invalid-row filtering.")

    x_nan_mask = pd.isna(pd.DataFrame(X_arr)).any(axis=1).to_numpy()
    x_inf_mask = np.zeros(X_arr.shape[0], dtype=bool)
    if np.issubdtype(X_arr.dtype, np.number):
        x_inf_mask = ~np.isfinite(X_arr).all(axis=1)

    if y_arr.ndim == 1:
        y_series = pd.Series(y_arr)
        y_nan_mask = y_series.isna().to_numpy()
        y_inf_mask = np.zeros(y_arr.shape[0], dtype=bool)
        if np.issubdtype(y_arr.dtype, np.number):
            y_inf_mask = ~np.isfinite(y_arr)
    else:
        y_nan_mask = pd.isna(pd.DataFrame(y_arr)).any(axis=1).to_numpy()
        y_inf_mask = np.zeros(y_arr.shape[0], dtype=bool)
        if np.issubdtype(y_arr.dtype, np.number):
            y_inf_mask = ~np.isfinite(y_arr).all(axis=1)

    invalid_mask = np.asarray(x_nan_mask | x_inf_mask | y_nan_mask | y_inf_mask, dtype=bool)
    keep_mask = ~invalid_mask
    if keep_mask.all():
        return X_arr, y_arr, dataset_df.reset_index(drop=True), keep_mask

    X_clean = X_arr[keep_mask]
    y_clean = y_arr[keep_mask]
    dataset_clean = dataset_df.loc[keep_mask].reset_index(drop=True)
    if len(y_clean) == 0:
        raise ValueError("All rows were dropped due to invalid values in X or y.")
    if show_progress:
        print(
            "[row-filter] "
            f"context={context or 'workflow'} | dropped_rows={int(invalid_mask.sum())} | remaining_rows={int(len(y_clean))}"
        )
    return X_clean, y_clean, dataset_clean, keep_mask


def _summary_progress_row(task_type: str, test_metrics: Dict[str, Any]) -> pd.DataFrame:
    if task_type == "regression":
        cols = ["test_spearman", "test_pearson", "test_r2", "test_rmse", "n_test_pooled", "n_train", "n_folds"]
    else:
        cols = ["test_mcc", "test_accuracy", "test_f1_weighted", "n_test_pooled", "n_train", "n_folds"]
    row = {k: test_metrics.get(k, np.nan) for k in cols}
    return pd.DataFrame([row])


def _build_standard_bundle(
    *,
    dataset_path: Path,
    enc_dir: Path,
    feature_files: List[str],
    target_col: str,
    input_filename_prefix: str = "",
):
    dataset_df = load_dataset_table(dataset_path)
    X = build_feature_matrix(
        encodings_dir=enc_dir,
        feature_files=feature_files,
        feature_prefix=input_filename_prefix or dataset_path.stem,
    )
    bundle = prepare_dataset_bundle(dataset_df=dataset_df, X=X, target_col=target_col)
    return bundle, dataset_df


def _build_custom_external_bundle(
    *,
    train_dataset_path: Path,
    test_dataset_path: Path,
    train_enc_dir: Path,
    test_enc_dir: Path,
    feature_files: List[str],
    target_col: str,
    train_input_filename_prefix: str = "",
    test_input_filename_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    train_df = load_dataset_table(train_dataset_path)
    test_df = load_dataset_table(test_dataset_path)

    X_train = build_feature_matrix(
        encodings_dir=train_enc_dir,
        feature_files=feature_files,
        feature_prefix=train_input_filename_prefix or train_dataset_path.stem,
    )
    X_test = build_feature_matrix(
        encodings_dir=test_enc_dir,
        feature_files=feature_files,
        feature_prefix=test_input_filename_prefix or test_dataset_path.stem,
    )

    train_bundle = prepare_dataset_bundle(dataset_df=train_df, X=X_train, target_col=target_col)
    test_bundle = prepare_dataset_bundle(dataset_df=test_df, X=X_test, target_col=target_col)

    X_all = np.concatenate([train_bundle.X, test_bundle.X], axis=0)
    y_all = np.concatenate([train_bundle.y, test_bundle.y], axis=0)

    combined_df = pd.concat([train_bundle.dataset_df, test_bundle.dataset_df], axis=0, ignore_index=True)
    train_idx = np.arange(len(train_bundle.y), dtype=int)
    test_idx = np.arange(len(train_bundle.y), len(train_bundle.y) + len(test_bundle.y), dtype=int)
    return X_all, y_all, combined_df, train_idx, test_idx


def _build_tune_key(
    *,
    target_col: str,
    split_type: str,
    feature_label: str,
    model_name: str,
    eval_group: str,
) -> Tuple[str, str, str, str, str]:
    return (
        str(target_col),
        str(split_type),
        str(feature_label),
        str(model_name),
        str(eval_group),
    )


def _apply_ntrain_preset_to_params(
    *,
    params: Dict[str, Any],
    hyperparameter_mode: str,
    model_name: str,
    task_type: str,
    feature_label: str,
    n_train: int,
    feature_ntrain_param_presets: Dict[str, Any] | None,
) -> Dict[str, Any]:
    out = dict(params)
    if hyperparameter_mode != "ntrain_preset":
        return out
    preset_params = get_feature_ntrain_param_preset(
        model_name=model_name,
        task_type=task_type,
        feature_label=feature_label,
        n_train=int(n_train),
        presets_override=feature_ntrain_param_presets,
    )
    if preset_params:
        out.update(preset_params)
    return out


def _select_params_for_split(
    *,
    base_params: Dict[str, Any],
    hyperparameter_mode: str,
    model_name: str,
    task_type: str,
    feature_label: str,
    n_train: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_key: Tuple[str, str, str, str, str],
    tuning_cache: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
    preset_search_cache: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
    feature_ntrain_param_presets: Dict[str, Any] | None,
    tuning_metric: str,
    tuning_n_trials: int,
    show_progress: bool,
    progress_context: str,
) -> Dict[str, Any]:
    params = _apply_ntrain_preset_to_params(
        params=base_params,
        hyperparameter_mode=hyperparameter_mode,
        model_name=model_name,
        task_type=task_type,
        feature_label=feature_label,
        n_train=n_train,
        feature_ntrain_param_presets=feature_ntrain_param_presets,
    )

    if hyperparameter_mode == "optuna_search":
        if tune_key not in tuning_cache:
            try:
                tuning_result = tune_model_hyperparameters(
                    model_name=model_name,
                    task_type=task_type,
                    X_train=X_train,
                    y_train=y_train,
                    base_params=base_params,
                    tuning_metric=tuning_metric,
                    n_trials=tuning_n_trials,
                    show_progress=show_progress,
                )
                tuning_cache[tune_key] = dict(tuning_result)
            except Exception as exc:
                tuning_cache[tune_key] = {
                    "best_params": dict(base_params),
                    "metric_name": tuning_metric,
                    "n_trials": 0,
                    "best_value": np.nan,
                    "used_tuning": False,
                    "reason": str(exc),
                }
                if show_progress:
                    print(f"[tuning-skipped] {progress_context} | reason={exc}")
        return dict(tuning_cache[tune_key].get("best_params", base_params))

    if hyperparameter_mode == "preset_search":
        if tune_key not in preset_search_cache:
            try:
                candidate_params_list = get_feature_param_preset_candidates(
                    model_name=model_name,
                    task_type=task_type,
                    feature_label=feature_label,
                    presets_override=feature_ntrain_param_presets,
                )
                search_result = select_best_params_from_candidates(
                    model_name=model_name,
                    task_type=task_type,
                    X_train=X_train,
                    y_train=y_train,
                    base_params=base_params,
                    candidate_params_list=candidate_params_list,
                    tuning_metric=tuning_metric,
                    show_progress=show_progress,
                )
                preset_search_cache[tune_key] = dict(search_result)
            except Exception as exc:
                preset_search_cache[tune_key] = {
                    "best_params": dict(base_params),
                    "metric_name": tuning_metric,
                    "n_trials": 0,
                    "best_value": np.nan,
                    "used_tuning": False,
                    "reason": str(exc),
                }
                if show_progress:
                    print(f"[preset-search-skipped] {progress_context} | reason={exc}")
        return dict(preset_search_cache[tune_key].get("best_params", base_params))

    return params


def run_supervised_ml_workflow(inputs: Dict[str, Any]) -> Dict[str, Any]:
    data_root = Path(str(inputs["data_root_dir"]).strip()).expanduser().resolve()
    data_subfolder = str(inputs.get("data_subfolder", "") or "").strip().strip("/")
    dataset_fname = str(inputs["dataset_fname"]).strip()
    target_col_input = inputs.get("target_col", "y")
    if isinstance(target_col_input, (list, tuple)):
        target_col_list = [str(x).strip() for x in target_col_input if str(x).strip()]
    else:
        target_col_list = [str(target_col_input).strip()]
    if not target_col_list:
        raise ValueError("target_col must be a non-empty string or list of non-empty strings.")
    task_type = str(inputs.get("classification_or_regression", "regression")).strip().lower()

    split_type_list = [str(x).strip() for x in inputs.get("split_type_list", ["random"]) if str(x).strip()]
    model_list = [str(x).strip() for x in inputs.get("model_list", ["rf"]) if str(x).strip()]
    feature_combinations_dict = dict(inputs.get("feature_combinations_dict", {}))
    if not feature_combinations_dict:
        raise ValueError("feature_combinations_dict cannot be empty.")

    sample_id_col = str(inputs.get("sample_id_col", "")).strip()
    split_seed = int(inputs.get("split_seed", 42))
    k_folds = int(inputs.get("k_folds", 5))
    kfold_repeats = max(int(inputs.get("random_kfold_repeats", 1)), 1)

    mutres_col = str(inputs.get("mutres_col", "mutres_idx")).strip()
    random_split_col = str(inputs.get("random_split_col", f"fold_random_{k_folds}")).strip()
    mutres_split_col = str(inputs.get("mutres_split_col", f"fold_mutres-modulo_{k_folds}")).strip()
    segment_col = str(inputs.get("segment_col", "segment_index_0")).strip()
    segment_index_range = _parse_segment_index_range(inputs.get("segment_index_range"))
    contiguous_split_col = str(inputs.get("contiguous_split_col", f"fold_contiguous_{k_folds}")).strip()

    custom_split_col = str(inputs.get("custom_split_col", "split")).strip()
    custom_test_value = inputs.get("custom_test_value", "test")
    custom_split_indices = inputs.get("custom_split_indices")
    custom_test_dataset_fname = _clean_optional_str(inputs.get("custom_test_dataset_fname", ""))
    custom_test_data_subfolder_raw = _clean_optional_str(inputs.get("custom_test_data_subfolder", "")).strip().strip("/")
    custom_test_data_subfolder = custom_test_data_subfolder_raw or data_subfolder
    input_filename_prefix = _clean_optional_str(inputs.get("input_filename_prefix", ""))
    custom_input_filename_prefix = _clean_optional_str(inputs.get("custom_input_filename_prefix", ""))
    sequence_base_input = inputs.get("sequence_base")

    settings_repo_dir = Path(
        str(inputs.get("model_settings_repo_dir", "src/abcode/tools/ml/model_settings"))
    ).expanduser().resolve()
    model_params_override = inputs.get("model_params")
    hyperparameter_mode = _normalize_hyperparameter_mode(inputs.get("hyperparameter_mode", "default"))
    summary_source_mode = _normalize_summary_source_mode(inputs.get("summary_source_mode", "run_cache"))
    summary_metric_mode = _normalize_summary_metric_mode(inputs.get("summary_metric_mode", "average"))
    feature_ntrain_param_presets = inputs.get("feature_ntrain_param_presets")
    tuning_metric = resolve_tuning_metric(
        task_type=task_type,
        tuning_metric=str(inputs.get("tuning_metric", "spearman")),
    )
    tuning_n_trials = int(inputs.get("tuning_n_trials", 30))

    save_trained_models = bool(inputs.get("save_trained_models", True))
    train_full_data_model = bool(inputs.get("train_full_data_model", False))
    save_predictions = bool(inputs.get("save_predictions", True))
    show_progress = bool(inputs.get("show_progress", True))
    feature_model_pairs = normalize_feature_model_pairs(
        inputs.get("featurecombi_model_pair_to_extract_coefficients_for", [])
    )
    extract_feature_coefficients = bool(feature_model_pairs)

    run_label = str(inputs.get("run_label", _utc_now_label())).strip() or _utc_now_label()
    csv_suffix = str(inputs.get("csv_suffix", "")).strip()

    train_enc_dir = data_root / "encodings"
    train_dataset_dir = data_root / "expdata"
    model_dir = data_root / "ml" / "trained_models"
    pred_dir = data_root / "ml" / "predictions"
    out_dir = data_root / "ml" / "output"

    if data_subfolder:
        train_enc_dir = train_enc_dir / data_subfolder
        train_dataset_dir = train_dataset_dir / data_subfolder
        model_dir = model_dir / data_subfolder
        pred_dir = pred_dir / data_subfolder
        out_dir = out_dir / data_subfolder

    train_dataset_path = train_dataset_dir / dataset_fname

    metrics_csv_path = out_dir / _build_metrics_csv_name(task_type=task_type, suffix=csv_suffix)
    summary_csv_path = out_dir / _build_summary_csv_name(task_type=task_type, suffix=csv_suffix)
    run_rows: List[Dict[str, Any]] = []
    pooled_prediction_rows: Dict[Tuple[str, str, str, str], List[pd.DataFrame]] = {}
    pooled_prediction_paths: List[str] = []
    pooled_test_cache: Dict[Tuple[str, str, str, str, str], Dict[str, List[Any]]] = {}
    matched_coeff_pairs: set[tuple[str, str]] = set()
    tuning_cache: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    preset_search_cache: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}

    for target_col in target_col_list:
        for split_type in split_type_list:
            for feature_label, feature_files_raw in feature_combinations_dict.items():
                feature_files = [str(x).strip() for x in list(feature_files_raw) if str(x).strip()]
                if not feature_files:
                    raise ValueError(f"feature_combinations_dict['{feature_label}'] must contain at least one feature file name.")
                feature_files, svd_n_components = resolve_svd_feature_config(
                    feature_label=feature_label,
                    feature_files=feature_files,
                )
                split_key = str(split_type).strip().lower()

                if split_key == "custom" and custom_test_dataset_fname:
                    test_enc_dir = data_root / "encodings"
                    test_dataset_dir = data_root / "expdata"
                    if custom_test_data_subfolder:
                        test_enc_dir = test_enc_dir / custom_test_data_subfolder
                        test_dataset_dir = test_dataset_dir / custom_test_data_subfolder
                    test_dataset_path = test_dataset_dir / custom_test_dataset_fname

                    X_all, y_all, dataset_df, train_idx, test_idx = _build_custom_external_bundle(
                        train_dataset_path=train_dataset_path,
                        test_dataset_path=test_dataset_path,
                        train_enc_dir=train_enc_dir,
                        test_enc_dir=test_enc_dir,
                        feature_files=feature_files,
                        target_col=target_col,
                        train_input_filename_prefix=input_filename_prefix,
                        test_input_filename_prefix=custom_input_filename_prefix,
                    )
                    is_test_row = np.concatenate(
                        [
                            np.zeros(len(train_idx), dtype=bool),
                            np.ones(len(test_idx), dtype=bool),
                        ],
                        axis=0,
                    )
                    X_all, y_all, dataset_df, keep_mask = _drop_invalid_rows_from_xy(
                        X=X_all,
                        y=y_all,
                        dataset_df=dataset_df,
                        show_progress=show_progress,
                        context=f"target_col={target_col} | split_type=custom_external | feature_combi_name={feature_label}",
                    )
                    is_test_row = is_test_row[keep_mask]
                    train_idx = np.flatnonzero(~is_test_row)
                    test_idx = np.flatnonzero(is_test_row)
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        raise ValueError(
                            "After invalid-row filtering, custom external split has empty train or test partition."
                        )
                    split_defs = [
                        {
                            "split_type": "custom",
                            "split_id": "custom_external_0",
                            "train_idx": train_idx,
                            "test_idx": test_idx,
                        }
                    ]
                else:
                    bundle, dataset_df = _build_standard_bundle(
                        dataset_path=train_dataset_path,
                        enc_dir=train_enc_dir,
                        feature_files=feature_files,
                        target_col=target_col,
                        input_filename_prefix=input_filename_prefix,
                    )
                    X_all, y_all, dataset_df, _ = _drop_invalid_rows_from_xy(
                        X=bundle.X,
                        y=bundle.y,
                        dataset_df=dataset_df,
                        show_progress=show_progress,
                        context=f"target_col={target_col} | split_type={split_type} | feature_combi_name={feature_label}",
                    )
                    can_use_progressive = split_key in PROGRESSIVE_SPLIT_KEYS
                    has_segment_data = (
                        segment_col in dataset_df.columns
                        and dataset_df[segment_col].dropna().nunique() >= 2
                    )
                    use_progressive = can_use_progressive and has_segment_data
                    if split_key in CUSTOM_LIKE_KEYS and not use_progressive:
                        if show_progress:
                            print(
                                "[split-skip] custom (retrospective-style) requires segment frontiers "
                                f"(segment_col='{segment_col}', "
                                f"segment_index_range={inputs.get('segment_index_range', None)})."
                            )
                        continue

                    if use_progressive:
                        split_defs = generate_progressive_splits(
                            dataset_df=dataset_df,
                            split_type=split_type,
                            segment_col=segment_col,
                            random_split_col=random_split_col,
                            mutres_split_col=mutres_split_col,
                        )
                        if segment_index_range is not None:
                            start_idx, end_idx = segment_index_range
                            split_defs = [
                                s
                                for s in split_defs
                                if (start_idx is None or int(s.get("frontier_idx", 0)) >= start_idx)
                                and (end_idx is None or int(s.get("frontier_idx", 0)) <= end_idx)
                            ]
                        if not split_defs:
                            raise ValueError(
                                "No progressive splits remain after applying segment_index_range filter."
                            )
                    else:
                        split_defs = generate_splits(
                            dataset_df=dataset_df,
                            n_rows=len(dataset_df),
                            split_type=split_type,
                            k_folds=k_folds,
                            kfold_repeats=kfold_repeats,
                            seed=split_seed,
                            mutres_col=mutres_col,
                            random_split_col=random_split_col,
                            mutres_split_col=mutres_split_col,
                            contiguous_split_col=contiguous_split_col,
                            custom_split_col=custom_split_col,
                            custom_test_value=custom_test_value,
                            custom_split_indices=custom_split_indices,
                        )

                for model_name in model_list:
                    base_params = load_model_params(
                        model_name=model_name,
                        task_type=task_type,
                        settings_repo_dir=settings_repo_dir,
                        model_params_override=model_params_override,
                    )
                    for split_idx, split_info in enumerate(split_defs):
                        split_id = int(split_idx)
                        eval_group = str(split_info.get("eval_group", "all_data"))
                        train_idx = np.asarray(split_info["train_idx"], dtype=int)
                        test_idx = np.asarray(split_info["test_idx"], dtype=int)

                        X_train, X_test = X_all[train_idx], X_all[test_idx]
                        y_train, y_test = y_all[train_idx], y_all[test_idx]
                        if svd_n_components is not None:
                            X_train, X_test, _ = fit_transform_svd_train_test(
                                X_train=X_train,
                                X_test=X_test,
                                n_components=svd_n_components,
                                random_state=split_seed,
                            )
                        if show_progress:
                            print(
                                "[eval-start] "
                                f"target_col={target_col} | split_type={split_info['split_type']} | "
                                f"feature_combi_name={feature_label} | model_name={model_name}"
                            )

                        tune_key = _build_tune_key(
                            target_col=target_col,
                            split_type=str(split_info["split_type"]),
                            feature_label=feature_label,
                            model_name=str(model_name),
                            eval_group=eval_group,
                        )
                        progress_context = (
                            f"target_col={target_col} | split_type={split_info['split_type']} | "
                            f"feature_combi_name={feature_label} | model_name={model_name}"
                        )
                        params = _select_params_for_split(
                            base_params=base_params,
                            hyperparameter_mode=hyperparameter_mode,
                            model_name=model_name,
                            task_type=task_type,
                            feature_label=feature_label,
                            n_train=int(len(train_idx)),
                            X_train=X_train,
                            y_train=y_train,
                            tune_key=tune_key,
                            tuning_cache=tuning_cache,
                            preset_search_cache=preset_search_cache,
                            feature_ntrain_param_presets=feature_ntrain_param_presets,
                            tuning_metric=tuning_metric,
                            tuning_n_trials=tuning_n_trials,
                            show_progress=show_progress,
                            progress_context=progress_context,
                        )

                        model = build_model(model_name=model_name, task_type=task_type, params=params)
                        model.fit(X_train, y_train)

                        yhat_train = model.predict(X_train)
                        yhat_test = model.predict(X_test)

                        y_score_train = None
                        y_score_test = None
                        if task_type == "classification" and hasattr(model, "predict_proba"):
                            try:
                                y_score_train = model.predict_proba(X_train)
                                y_score_test = model.predict_proba(X_test)
                            except Exception:
                                y_score_train = None
                                y_score_test = None

                        train_metrics = _round_metric_dict(compute_metrics(
                            y_true=y_train,
                            y_pred=yhat_train,
                            task_type=task_type,
                            y_score=y_score_train,
                        ))
                        test_metrics = _round_metric_dict(compute_metrics(
                            y_true=y_test,
                            y_pred=yhat_test,
                            task_type=task_type,
                            y_score=y_score_test,
                        ))

                        row = {
                            "target_col": target_col,
                            "feature_label": feature_label,
                            "split_type": split_info["split_type"],
                            "split_id": split_id,
                            "model_name": model_name,
                            "task_type": task_type,
                            "n": int(X_all.shape[0]),
                            "p": int(X_all.shape[1]),
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            "selected_hyperparameters": _params_to_json(params),
                            "hyperparameter_tuning_enabled": bool(hyperparameter_mode != "default"),
                            "hyperparameter_tuning_metric": tuning_metric,
                            "eval_group": eval_group,
                            "frontier_idx": split_info.get("frontier_idx", np.nan),
                            "segment_min": split_info.get("segment_min", np.nan),
                            "segment_max": split_info.get("segment_max", np.nan),
                            "segments_included": str(split_info.get("segments_included", "")),
                            "data_size_n": split_info.get("data_size_n", int(len(train_idx) + len(test_idx))),
                            "test_segment": split_info.get("test_segment", np.nan),
                        }
                        row.update({f"test_{k}": v for k, v in test_metrics.items()})
                        row.update({f"train_{k}": v for k, v in train_metrics.items()})
                        run_rows.append(row)
                        _append_metrics_row(metrics_csv_path, row)
                        if show_progress:
                            fold_progress = _progress_row(task_type, test_metrics, train_metrics)
                            split_name = split_info.get("split_id", split_id)
                            print(
                                "[fold-result] "
                                f"split_id={split_id} | split_name={split_name} | split_type={split_info['split_type']} | "
                                f"eval_group={eval_group} | data_size_n={row['data_size_n']} | "
                                f"segments_included={row['segments_included']} | n_train={row['n_train']} | n_test={row['n_test']}"
                            )
                            print(fold_progress.to_string(index=False))

                        pool_key = (
                            target_col,
                            feature_label,
                            str(split_info["split_type"]),
                            model_name,
                            eval_group,
                        )
                        cache = pooled_test_cache.setdefault(pool_key, {"y_true": [], "y_pred": [], "n_train": []})
                        cache["y_true"].append(np.asarray(y_test))
                        cache["y_pred"].append(np.asarray(yhat_test))
                        cache["n_train"].append(int(len(train_idx)))
                        cache.setdefault("selected_hyperparameters", []).append(_params_to_json(params))

                        if save_trained_models:
                            model_path = model_dir / f"{run_label}__{feature_label}__{target_col}__{split_id}__{model_name}.pkl"
                            payload = {
                                "model": model,
                                "metadata": {
                                    "feature_label": feature_label,
                                    "feature_files": list(feature_files),
                                    "split_type": split_info["split_type"],
                                    "split_id": split_id,
                                    "model_name": model_name,
                                    "task_type": task_type,
                                    "params": params,
                                    "target_col": target_col,
                                },
                            }
                            _write_pickle(model_path, payload)

                        if save_predictions:
                            _append_fold_predictions(
                                pooled_prediction_rows=pooled_prediction_rows,
                                target_col=target_col,
                                feature_label=feature_label,
                                split_type=str(split_info["split_type"]),
                                model_name=model_name,
                                split_id=split_id,
                                eval_group=eval_group,
                                test_idx=test_idx,
                                y_test=y_test,
                                yhat_test=yhat_test,
                                dataset_df=dataset_df,
                                sample_id_col=sample_id_col,
                            )

                if train_full_data_model or extract_feature_coefficients:
                    resolved_sequence_base = resolve_sequence_base(
                        sequence_base_input=sequence_base_input,
                        dataset_df=dataset_df,
                        data_root=data_root,
                        data_subfolder=data_subfolder,
                    )
                    for model_name in model_list:
                        want_coefficients = should_extract_coefficients(
                            feature_label=feature_label,
                            model_name=model_name,
                            feature_files=feature_files,
                            feature_model_pairs=feature_model_pairs,
                        )
                        if not train_full_data_model and not want_coefficients:
                            continue

                        params = load_model_params(
                            model_name=model_name,
                            task_type=task_type,
                            settings_repo_dir=settings_repo_dir,
                            model_params_override=model_params_override,
                        )
                        params = _apply_ntrain_preset_to_params(
                            params=params,
                            hyperparameter_mode=hyperparameter_mode,
                            model_name=model_name,
                            task_type=task_type,
                            feature_label=feature_label,
                            n_train=int(len(y_all)),
                            feature_ntrain_param_presets=feature_ntrain_param_presets,
                        )
                        X_full = X_all
                        if svd_n_components is not None:
                            X_full, _, _ = fit_transform_svd_train_test(
                                X_train=X_all,
                                X_test=None,
                                n_components=svd_n_components,
                                random_state=split_seed,
                            )
                        model = build_model(model_name=model_name, task_type=task_type, params=params)
                        model.fit(X_full, y_all)

                        if train_full_data_model and save_trained_models:
                            model_path = model_dir / f"{run_label}__{feature_label}__{target_col}__full__{model_name}.pkl"
                            payload = {
                                "model": model,
                                "metadata": {
                                    "feature_label": feature_label,
                                    "feature_files": list(feature_files),
                                    "split_type": "full",
                                    "split_id": "full",
                                    "model_name": model_name,
                                    "task_type": task_type,
                                    "params": params,
                                    "target_col": target_col,
                                },
                            }
                            _write_pickle(model_path, payload)

                        if want_coefficients:
                            matched_coeff_pairs.add((str(feature_label).strip().lower(), str(model_name).strip().lower()))
                            if len(feature_files) == 1:
                                matched_coeff_pairs.add((str(feature_files[0]).strip().lower(), str(model_name).strip().lower()))
                            try:
                                coefficients = extract_coefficients(model_name=model_name, model_obj=model)
                                coeff_path = save_coefficients_csv(
                                    coefficients=coefficients,
                                    out_dir=out_dir,
                                    feature_label=feature_label,
                                    model_name=model_name,
                                    feature_files=feature_files,
                                    sequence_base=resolved_sequence_base,
                                )
                                if show_progress:
                                    print(f"[coefficients-saved] {coeff_path}")
                            except Exception as exc:
                                if show_progress:
                                    print(
                                        "[coefficients-skipped] "
                                        f"target_col={target_col} | feature_combi_name={feature_label} | "
                                        f"model_name={model_name} | reason={exc}"
                                    )

    if feature_model_pairs and not matched_coeff_pairs and show_progress:
        print(
            "[coefficients-warning] No requested feature/model pairs matched this run. "
            f"requested={sorted(feature_model_pairs)}"
        )

    if save_predictions:
        pooled_prediction_paths = _write_pooled_predictions(
            pooled_prediction_rows=pooled_prediction_rows,
            pred_dir=pred_dir,
            run_label=run_label,
            csv_suffix=csv_suffix,
            show_progress=show_progress,
        )

    run_df = pd.DataFrame(run_rows)
    summary_df = _build_summary_dataframe(
        summary_source_mode=summary_source_mode,
        summary_metric_mode=summary_metric_mode,
        run_df=run_df,
        pooled_test_cache=pooled_test_cache,
        metrics_csv_path=metrics_csv_path,
        task_type=task_type,
        hyperparameter_mode=hyperparameter_mode,
        tuning_metric=tuning_metric,
    )
    if show_progress and not summary_df.empty:
        for _, srow in summary_df.iterrows():
            summary_row = dict(srow)
            print(
                "[summary] "
                f"target_col={summary_row.get('target_col')} | split_type={summary_row.get('split_type')} | "
                f"feature_combi_name={summary_row.get('feature_label')} | model_name={summary_row.get('model_name')}"
            )
            summary_progress = _summary_progress_row(task_type, summary_row)
            print(summary_progress.to_string(index=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary_df.empty:
        if summary_csv_path.exists():
            existing_summary = pd.read_csv(summary_csv_path)
            summary_out = pd.concat([existing_summary, summary_df], ignore_index=True)
        else:
            summary_out = summary_df
        summary_out.to_csv(summary_csv_path, index=False)
    return {
        "status": "ok",
        "run_label": run_label,
        "dataset_path": str(train_dataset_path),
        "metrics_csv_path": str(metrics_csv_path),
        "summary_csv_path": str(summary_csv_path),
        "data_efficiency_csv_path": "",
        "n_results": int(len(run_df)),
        "n_summary_rows": int(len(summary_df)),
        "model_dir": str(model_dir),
        "prediction_dir": str(pred_dir),
        "prediction_csv_paths": pooled_prediction_paths,
        "output_dir": str(out_dir),
    }
