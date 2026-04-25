from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_split_key(split_type: str) -> str:
    key = str(split_type).strip().lower()
    alias_map = {
        "mutres_modulo": "mutres-modulo",
    }
    return alias_map.get(key, key)


def _build_custom_split_from_columns(
    dataset_df: pd.DataFrame,
    split_col: str,
    custom_test_value: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    if split_col not in dataset_df.columns:
        raise KeyError(f"custom split column '{split_col}' not found in dataset.")

    raw = dataset_df[split_col]
    split_values = raw.astype(str).str.lower().str.strip()

    train_idx = np.where(split_values == "train")[0]
    test_idx = np.where(split_values == "test")[0]
    if len(train_idx) > 0 and len(test_idx) > 0:
        return train_idx, test_idx

    # Fallback: treat one fold value as test and all other rows as train.
    value = str(custom_test_value).strip().lower()
    test_idx = np.where(split_values == value)[0]
    train_idx = np.where(split_values != value)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"custom split column '{split_col}' must contain train/test labels or test value '{custom_test_value}'."
        )
    return train_idx, test_idx


def _build_custom_split_from_indices(
    custom_split_indices: Dict[str, Iterable[int]],
    n_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.array(list(custom_split_indices.get("train", [])), dtype=int)
    test_idx = np.array(list(custom_split_indices.get("test", [])), dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("custom_split_indices must include non-empty 'train' and 'test' lists.")
    if train_idx.min() < 0 or test_idx.min() < 0 or train_idx.max() >= n_rows or test_idx.max() >= n_rows:
        raise ValueError("custom_split_indices contains out-of-range row indices.")
    return train_idx, test_idx


def _build_leave_one_group_out_splits(
    values: pd.Series,
    *,
    split_type: str,
    split_id_prefix: str,
    extra_key: str | None = None,
) -> List[Dict[str, Any]]:
    group_ids = list(pd.unique(values.dropna()))
    idx = np.arange(len(values))
    splits: List[Dict[str, Any]] = []
    for group_id in group_ids:
        test_mask = values == group_id
        test_idx = idx[test_mask.to_numpy()]
        train_idx = idx[~test_mask.to_numpy()]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        split_info = {
            "split_type": split_type,
            "split_id": f"{split_id_prefix}_{group_id}",
            "train_idx": np.sort(train_idx),
            "test_idx": np.sort(test_idx),
        }
        if extra_key:
            split_info[extra_key] = str(group_id)
        splits.append(split_info)
    return splits


def _splits_from_fold_column(dataset_df: pd.DataFrame, split_col: str, split_type: str) -> List[Dict[str, Any]]:
    values = dataset_df[split_col]
    splits = _build_leave_one_group_out_splits(
        values,
        split_type=split_type,
        split_id_prefix=f"{split_type}_fold",
    )
    if not splits:
        raise ValueError(f"Fold column '{split_col}' did not produce valid train/test splits.")
    return splits


def _splits_from_asset_column(dataset_df: pd.DataFrame, asset_col: str) -> List[Dict[str, Any]]:
    if asset_col not in dataset_df.columns:
        raise KeyError(f"asset split column '{asset_col}' not found in dataset.")

    values = dataset_df[asset_col]
    asset_ids = [asset_id for asset_id in pd.unique(values.dropna())]
    if len(asset_ids) < 2:
        raise ValueError(
            f"asset split column '{asset_col}' must contain at least 2 unique non-null values for LOO splitting."
        )

    splits = _build_leave_one_group_out_splits(
        values,
        split_type="asset",
        split_id_prefix="asset",
        extra_key="test_asset",
    )
    if not splits:
        raise ValueError(f"asset split column '{asset_col}' did not produce valid leave-one-asset-out splits.")
    return splits


def get_random_split_indices(
    *,
    n_rows: int,
    k_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    k = max(int(k_folds), 2)
    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    idx = np.arange(n_rows)
    return [(np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)) for train_idx, test_idx in kf.split(idx)]


def get_repeated_random_split_indices(
    *,
    n_rows: int,
    k_folds: int,
    n_repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    from sklearn.model_selection import RepeatedKFold

    k = max(int(k_folds), 2)
    r = max(int(n_repeats), 1)
    rkf = RepeatedKFold(n_splits=k, n_repeats=r, random_state=int(seed))
    idx = np.arange(n_rows)

    out: List[Dict[str, Any]] = []
    for i, (train_idx, test_idx) in enumerate(rkf.split(idx)):
        repeat_id = int(i // k)
        fold_id = int(i % k)
        out.append(
            {
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "train_idx": np.asarray(train_idx, dtype=int),
                "test_idx": np.asarray(test_idx, dtype=int),
            }
        )
    return out


def get_mutres_modulo_split_indices(
    *,
    dataset_df: pd.DataFrame,
    n_rows: int,
    k_folds: int,
    mutres_col: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Placeholder implementation: bucket by modulo group and leave-one-group-out.
    idx = np.arange(n_rows)
    k = max(int(k_folds), 2)
    if mutres_col in dataset_df.columns:
        source = dataset_df[mutres_col].to_numpy(dtype=int)
    else:
        source = idx

    groups = source % k
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(k):
        test_mask = groups == fold_id
        test_idx = idx[test_mask]
        train_idx = idx[~test_mask]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((np.sort(train_idx), np.sort(test_idx)))
    if not splits:
        raise ValueError("mutres-modulo placeholder split could not produce valid folds.")
    return splits


def get_contiguous_split_indices(
    *,
    n_rows: int,
    k_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Placeholder implementation: contiguous chunks as test folds.
    idx = np.arange(n_rows)
    k = max(int(k_folds), 2)
    fold_sizes = np.full(k, n_rows // k, dtype=int)
    fold_sizes[: (n_rows % k)] += 1

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        if len(train_idx) == 0 or len(test_idx) == 0:
            start = stop
            continue
        splits.append((np.sort(train_idx), np.sort(test_idx)))
        start = stop

    if not splits:
        raise ValueError("contiguous placeholder split could not produce valid folds.")
    return splits


def generate_splits(
    *,
    dataset_df: pd.DataFrame,
    n_rows: int,
    split_type: str,
    k_folds: int,
    kfold_repeats: int,
    seed: int,
    mutres_col: str,
    random_split_col: str,
    mutres_split_col: str,
    contiguous_split_col: str,
    asset_split_col: str,
    custom_split_col: str,
    custom_test_value: Any,
    custom_split_indices: Optional[Dict[str, Iterable[int]]] = None,
) -> List[Dict[str, Any]]:
    split_key = _normalize_split_key(split_type)

    if split_key == "random":
        repeats = max(int(kfold_repeats), 1)
        if repeats == 1 and random_split_col and random_split_col in dataset_df.columns:
            return _splits_from_fold_column(dataset_df, split_col=random_split_col, split_type="random")

        if repeats > 1:
            repeated = get_repeated_random_split_indices(
                n_rows=n_rows,
                k_folds=k_folds,
                n_repeats=repeats,
                seed=seed,
            )
            return [
                {
                    "split_type": "random",
                    "split_id": f"random_repeat_{d['repeat_id']}_fold_{d['fold_id']}",
                    "repeat_id": int(d["repeat_id"]),
                    "fold_id": int(d["fold_id"]),
                    "train_idx": np.sort(np.asarray(d["train_idx"], dtype=int)),
                    "test_idx": np.sort(np.asarray(d["test_idx"], dtype=int)),
                }
                for d in repeated
            ]

        pairs = get_random_split_indices(n_rows=n_rows, k_folds=k_folds, seed=seed)
        return [
            {
                "split_type": "random",
                "split_id": f"random_fold_{i}",
                "repeat_id": 0,
                "fold_id": int(i),
                "train_idx": tr,
                "test_idx": te,
            }
            for i, (tr, te) in enumerate(pairs)
        ]

    if split_key == "mutres-modulo":
        if mutres_split_col and mutres_split_col in dataset_df.columns:
            return _splits_from_fold_column(dataset_df, split_col=mutres_split_col, split_type="mutres-modulo")

        pairs = get_mutres_modulo_split_indices(
            dataset_df=dataset_df,
            n_rows=n_rows,
            k_folds=k_folds,
            mutres_col=mutres_col,
        )
        return [
            {
                "split_type": "mutres-modulo",
                "split_id": f"mutres_modulo_fold_{i}",
                "train_idx": tr,
                "test_idx": te,
            }
            for i, (tr, te) in enumerate(pairs)
        ]

    if split_key == "contiguous":
        if contiguous_split_col and contiguous_split_col in dataset_df.columns:
            return _splits_from_fold_column(dataset_df, split_col=contiguous_split_col, split_type="contiguous")

        pairs = get_contiguous_split_indices(n_rows=n_rows, k_folds=k_folds)
        return [
            {
                "split_type": "contiguous",
                "split_id": f"contiguous_fold_{i}",
                "train_idx": tr,
                "test_idx": te,
            }
            for i, (tr, te) in enumerate(pairs)
        ]

    if split_key == "asset":
        return _splits_from_asset_column(dataset_df, asset_col=asset_split_col)

    if split_key == "custom":
        if custom_split_indices:
            train_idx, test_idx = _build_custom_split_from_indices(custom_split_indices, n_rows=n_rows)
        else:
            train_idx, test_idx = _build_custom_split_from_columns(
                dataset_df,
                split_col=custom_split_col,
                custom_test_value=custom_test_value,
            )
        return [
            {
                "split_type": "custom",
                "split_id": "custom_0",
                "train_idx": np.sort(train_idx),
                "test_idx": np.sort(test_idx),
            }
        ]

    raise ValueError(
        f"Unsupported split_type '{split_type}'. Supported: random, mutres-modulo, contiguous, asset, custom"
    )
