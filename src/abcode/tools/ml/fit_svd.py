from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD


_SVD_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_svd(?P<n_components>\d+)$", re.IGNORECASE)


def parse_svd_signature(feature_name: str) -> tuple[str, Optional[int]]:
    """
    Parse an optional trailing SVD signature from a feature name like '<feature_name>_svd{p}'.
    Returns (base_feature_name, n_components). If no signature is present, n_components is None.
    """
    raw_name = str(feature_name).strip()
    if not raw_name:
        return "", None
    match = _SVD_SUFFIX_RE.match(raw_name)
    if not match:
        return raw_name, None
    base_name = str(match.group("base")).strip()
    n_components = int(match.group("n_components"))
    return base_name, n_components


def fit_transform_svd_train_test(
    *,
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_components: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], TruncatedSVD]:
    """
    Fit TruncatedSVD on X_train and transform X_train/X_test with the same fitted model.
    """
    if int(n_components) <= 0:
        raise ValueError("n_components must be > 0 for SVD.")
    X_train_arr = np.asarray(X_train)
    X_test_arr = None if X_test is None else np.asarray(X_test)
    if X_train_arr.ndim != 2:
        raise ValueError(f"X_train must be 2D; got shape {X_train_arr.shape}.")
    if X_test_arr is not None and X_test_arr.ndim != 2:
        raise ValueError(f"X_test must be 2D; got shape {X_test_arr.shape}.")

    n_features = int(X_train_arr.shape[1])
    if int(n_components) > n_features:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed n_features ({n_features}) in X_train."
        )

    svd = TruncatedSVD(n_components=int(n_components), random_state=int(random_state))
    X_train_out = svd.fit_transform(X_train_arr)
    X_test_out = None if X_test_arr is None else svd.transform(X_test_arr)
    return X_train_out, X_test_out, svd


def resolve_feature_svd_specs(
    feature_files: Sequence[str],
) -> list[tuple[str, Optional[int]]]:
    """
    Parse per-feature SVD directives from entries like '<feature_name>_svd{p}'.
    Returns a list aligned to feature_files as (base_feature_name, n_components_or_none).
    """
    specs: list[tuple[str, Optional[int]]] = []
    for feature_file in feature_files:
        base_feature_file, svd_n_components = parse_svd_signature(str(feature_file).strip())
        specs.append((base_feature_file, svd_n_components))
    return specs
