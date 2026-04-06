from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

from project_config.variables import aaList
from abcode.tools.utils.seq_utils import fetch_sequences_from_fasta

FASTA_SUFFIXES = {".fasta", ".fa", ".faa", ".fna"}
AA_SET = set(aaList)


def normalize_feature_model_pairs(pairs: Iterable[tuple[str, str]] | None) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if not pairs:
        return out
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        feature_label = str(item[0]).strip().lower()
        model_name = str(item[1]).strip().lower()
        if feature_label and model_name:
            out.add((feature_label, model_name))
    return out


def should_extract_coefficients(
    *,
    feature_label: str,
    model_name: str,
    feature_files: Iterable[str] | None,
    feature_model_pairs: Set[Tuple[str, str]],
) -> bool:
    if not feature_model_pairs:
        return False
    model_key = str(model_name).strip().lower()
    label_key = (str(feature_label).strip().lower(), model_key)
    if label_key in feature_model_pairs:
        return True

    files = [str(x).strip().lower() for x in (feature_files or []) if str(x).strip()]
    # Allow matching by feature-set name only for single-feature combinations.
    if len(files) == 1:
        file_key = (files[0], model_key)
        if file_key in feature_model_pairs:
            return True
    return False


def _flatten_coefficients(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        # For multi-output/multiclass, reduce to one score per input feature.
        return arr.mean(axis=0).astype(float)
    return arr.reshape(-1).astype(float)


def _clean_sequence(seq: str) -> str:
    return "".join(str(seq or "").upper().split())


def _resolve_fasta_path(path_value: str, data_root: Path, data_subfolder: str) -> Optional[Path]:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                p,
                data_root / p,
                data_root / "sequences" / str(data_subfolder).strip().strip("/") / p,
            ]
        )
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def resolve_sequence_base(
    *,
    sequence_base_input: Any,
    dataset_df: pd.DataFrame,
    data_root: Path,
    data_subfolder: str,
) -> Optional[str]:
    # Priority 1: explicit user input (sequence string or fasta file path/name).
    if sequence_base_input is not None and str(sequence_base_input).strip():
        raw = str(sequence_base_input).strip()
        suffix = Path(raw).suffix.lower()
        if suffix in FASTA_SUFFIXES:
            fasta_path = _resolve_fasta_path(raw, data_root=data_root, data_subfolder=data_subfolder)
            if fasta_path is None:
                raise FileNotFoundError(f"sequence_base FASTA not found: {raw}")
            seqs, _, _ = fetch_sequences_from_fasta(str(fasta_path))
            if not seqs:
                raise ValueError(f"No sequences found in FASTA: {fasta_path}")
            return _clean_sequence(seqs[0])
        return _clean_sequence(raw)

    # Priority 2: infer from dataset sequence_base column if consistent.
    if "sequence_base" in dataset_df.columns:
        vals = dataset_df["sequence_base"].astype(str).map(_clean_sequence)
        vals = [v for v in vals.tolist() if v]
        unique_vals = sorted(set(vals))
        if len(unique_vals) == 1:
            return unique_vals[0]
    return None


def _is_one_hot_single_feature(feature_files: Iterable[str] | None) -> bool:
    files = [Path(str(x)).stem.lower() for x in (feature_files or []) if str(x).strip()]
    if len(files) != 1:
        return False
    name = files[0]
    return name in {"one_hot", "onehot"} or "one_hot" in name or name.endswith("onehot")


def build_mutation_labels(
    *,
    n_features: int,
    feature_files: Iterable[str] | None,
    sequence_base: Optional[str],
) -> list[str]:
    # One-hot-specific mutation labels: index -> WTposAA.
    if _is_one_hot_single_feature(feature_files) and n_features % len(aaList) == 0:
        n_pos = n_features // len(aaList)
        seq = _clean_sequence(sequence_base or "")
        labels: list[str] = []
        for pos_idx in range(n_pos):
            wt = seq[pos_idx] if pos_idx < len(seq) and seq[pos_idx] in AA_SET else "X"
            for aa in aaList:
                labels.append(f"{wt}{pos_idx + 1}{aa}")
        if len(labels) == n_features:
            return labels
    return [f"feature_{i}" for i in range(n_features)]


def _mutation_position(mut_label: str) -> float:
    digits = "".join(ch for ch in str(mut_label) if ch.isdigit())
    return int(digits) if digits else None


def _is_non_wt_mutation(mut_label: str) -> int:
    m = str(mut_label)
    if len(m) >= 2 and m[0].isalpha() and m[-1].isalpha():
        return 1 if m[0] != m[-1] else 0
    return 0


def extract_coefficients(model_name: str, model_obj: Any) -> np.ndarray:
    key = str(model_name).strip().lower()

    if key == "ridge":
        if not hasattr(model_obj, "coef_"):
            raise ValueError("Ridge model has no coef_ attribute.")
        return _flatten_coefficients(np.asarray(model_obj.coef_))

    if key == "xgboost":
        if not hasattr(model_obj, "feature_importances_"):
            raise ValueError("XGBoost model has no feature_importances_ attribute.")
        return _flatten_coefficients(np.asarray(model_obj.feature_importances_))

    raise ValueError(f"Coefficient extraction is not implemented for model '{model_name}'.")


def save_coefficients_csv(
    *,
    coefficients: np.ndarray,
    out_dir: Path,
    feature_label: str,
    model_name: str,
    feature_files: Iterable[str] | None = None,
    sequence_base: Optional[str] = None,
) -> Path:
    coeffs = np.asarray(coefficients).reshape(-1)
    mutation_labels = build_mutation_labels(
        n_features=len(coeffs),
        feature_files=feature_files,
        sequence_base=sequence_base,
    )
    positions = [_mutation_position(mut) for mut in mutation_labels]
    is_mut = [_is_non_wt_mutation(mut) for mut in mutation_labels]

    df = pd.DataFrame(
        {
            "mutation": mutation_labels,
            "position": positions,
            "is_mut": is_mut,
            "fitted coefficient": coeffs,
        }
    )

    filename = f"coefficients_{feature_label}_{model_name}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path
