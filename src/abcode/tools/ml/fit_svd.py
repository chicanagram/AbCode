from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD


_SVD_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_svd(?P<n_components>\d+)$", re.IGNORECASE)


def parse_svd_signature(feature_label: str) -> tuple[str, Optional[int]]:
    """
    Parse feature labels with optional trailing SVD signature: '<label>_svd{p}'.
    Returns (base_label, n_components). If no signature is present, n_components is None.
    """
    label = str(feature_label).strip()
    if not label:
        return "", None
    match = _SVD_SUFFIX_RE.match(label)
    if not match:
        return label, None
    base_label = str(match.group("base")).strip()
    n_components = int(match.group("n_components"))
    return base_label, n_components


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


def resolve_svd_feature_config(
    *,
    feature_label: str,
    feature_files: Sequence[str],
) -> tuple[list[str], Optional[int]]:
    """
    Resolve SVD config from feature label and normalize feature file names.
    If feature_label uses '<label>_svd{p}', matching feature file suffixes '_svd{p}'
    are stripped before loading raw feature arrays.
    """
    _, svd_n_components = parse_svd_signature(feature_label)
    normalized_feature_files = [str(x).strip() for x in feature_files if str(x).strip()]
    if svd_n_components is None:
        inferred_values = set()
        for feature_file in normalized_feature_files:
            _, file_svd_n_components = parse_svd_signature(feature_file)
            if file_svd_n_components is not None:
                inferred_values.add(int(file_svd_n_components))
        if len(inferred_values) > 1:
            raise ValueError(
                "Multiple SVD signatures found across feature files without a label-level signature: "
                f"feature_label='{feature_label}', svd_values={sorted(inferred_values)}."
            )
        if len(inferred_values) == 1:
            svd_n_components = int(next(iter(inferred_values)))
        else:
            return normalized_feature_files, None

    out: list[str] = []
    for feature_file in normalized_feature_files:
        base_feature_file, file_svd_n_components = parse_svd_signature(feature_file)
        if file_svd_n_components is None:
            out.append(feature_file)
            continue
        if file_svd_n_components != svd_n_components:
            raise ValueError(
                "SVD signature mismatch between feature label and feature file: "
                f"feature_label='{feature_label}', feature_file='{feature_file}'."
            )
        out.append(base_feature_file)
    return out, svd_n_components
