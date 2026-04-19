from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    dataset_df: pd.DataFrame


@dataclass
class FeatureBlock:
    feature_name: str
    start_col: int
    end_col: int


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        # Prefer scipy sparse loading when files were written with scipy.sparse.save_npz.
        try:
            from scipy import sparse as sp

            sparse_arr = sp.load_npz(path)
            arr = sparse_arr.toarray()
        except Exception:
            with np.load(path) as npz:
                if not npz.files:
                    raise ValueError(f"NPZ has no arrays: {path}")
                arr = npz[npz.files[0]]
    else:
        raise ValueError(f"Unsupported feature file extension '{suffix}' for {path}")

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _resolve_feature_path(encodings_dir: Path, feature_name: str) -> Path:
    """Resolve a feature file by exact name or by trying .npy/.npz suffixes."""
    raw = str(feature_name).strip()
    if not raw:
        raise ValueError("Feature name cannot be empty.")

    direct = (encodings_dir / raw).resolve()
    if direct.exists():
        return direct

    if Path(raw).suffix:
        raise FileNotFoundError(f"Feature file not found: {direct}")

    candidates = [direct.with_suffix(".npy"), direct.with_suffix(".npz")]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            f"Ambiguous feature name '{raw}': both {existing[0].name} and {existing[1].name} exist. "
            "Specify the extension explicitly."
        )

    # Fallback: allow dataset-prefixed filenames like "<dataset>_<feature>.npy|npz".
    prefixed = sorted(encodings_dir.glob(f"*_{raw}.npy")) + sorted(encodings_dir.glob(f"*_{raw}.npz"))
    if len(prefixed) == 1:
        return prefixed[0].resolve()
    if len(prefixed) > 1:
        names = ", ".join(p.name for p in prefixed[:3])
        raise ValueError(
            f"Ambiguous feature name '{raw}': multiple prefixed matches found ({names}...). "
            "Use an explicit filename to disambiguate."
        )

    raise FileNotFoundError(
        f"Feature '{raw}' not found in {encodings_dir}. Tried: {raw}, {raw}.npy, {raw}.npz"
    )


def _resolve_feature_path_with_prefix(encodings_dir: Path, feature_name: str, feature_prefix: str) -> Path:
    raw = str(feature_name).strip()
    pref = str(feature_prefix).strip()
    if not raw or not pref:
        raise FileNotFoundError("feature_name and feature_prefix are required.")

    prefix_candidates = [pref]
    if not pref.endswith("_"):
        prefix_candidates.append(f"{pref}_")
    else:
        prefix_candidates.append(pref[:-1])

    for pfx in prefix_candidates:
        for ext in ("", ".npy", ".npz"):
            p = (encodings_dir / f"{pfx}{raw}{ext}").resolve()
            if p.exists():
                return p
    raise FileNotFoundError


def build_feature_matrix(
    encodings_dir: Path,
    feature_files: Sequence[str],
    feature_prefix: str = "",
) -> np.ndarray:
    X, _ = build_feature_matrix_with_blocks(
        encodings_dir=encodings_dir,
        feature_files=feature_files,
        feature_prefix=feature_prefix,
    )
    return X


def build_feature_matrix_with_blocks(
    encodings_dir: Path,
    feature_files: Sequence[str],
    feature_prefix: str = "",
) -> tuple[np.ndarray, list[FeatureBlock]]:
    arrays: List[np.ndarray] = []
    feature_blocks: List[FeatureBlock] = []
    expected_n: Optional[int] = None
    start_col = 0

    for feature_file in feature_files:
        feature_path: Path
        if str(feature_prefix or "").strip():
            try:
                feature_path = _resolve_feature_path_with_prefix(
                    encodings_dir, str(feature_file), str(feature_prefix)
                )
            except FileNotFoundError:
                feature_path = _resolve_feature_path(encodings_dir, str(feature_file))
        else:
            feature_path = _resolve_feature_path(encodings_dir, str(feature_file))
        arr = _load_array(feature_path)
        if expected_n is None:
            expected_n = int(arr.shape[0])
        elif arr.shape[0] != expected_n:
            raise ValueError(
                "Feature row count mismatch while concatenating arrays: "
                f"expected {expected_n}, got {arr.shape[0]} from {feature_path.name}"
            )
        arrays.append(arr)
        end_col = start_col + int(arr.shape[1])
        feature_blocks.append(
            FeatureBlock(
                feature_name=str(feature_file),
                start_col=start_col,
                end_col=end_col,
            )
        )
        start_col = end_col

    if not arrays:
        raise ValueError("No feature files were provided for this feature combination.")
    return np.concatenate(arrays, axis=1), feature_blocks


def load_dataset_table(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"Dataset CSV is empty: {dataset_path}")
    return df


def prepare_dataset_bundle(
    *,
    dataset_df: pd.DataFrame,
    X: np.ndarray,
    target_col: str,
) -> DatasetBundle:
    if target_col not in dataset_df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataset columns.")

    y = dataset_df[target_col].to_numpy()
    if X.shape[0] != len(y):
        raise ValueError(
            "Mismatch between feature rows and target rows: "
            f"X has {X.shape[0]}, y has {len(y)}"
        )
    return DatasetBundle(X=X, y=y, dataset_df=dataset_df)
