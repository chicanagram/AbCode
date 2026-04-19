#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Optional, Sequence, Tuple, Union

from project_config.variables import aaList, aa2idx
from abcode.tools.utils.seq_utils import fetch_sequences_from_fasta
from abcode.tools.encodings.common import _sanitize_name, save_layerwise_embeddings


# This section of code is copied from ProFET (Ofer & Linial, DOI: 10.1093/bioinformatics/btv345)
# Original comment by the ProFET authors: 'Acquired from georgiev's paper of
# AAscales using helper script "GetTextData.py". + RegEx cleaning DOI: 10.1089/cmb.2008.0173'
gg_1 = {'Q': -2.54, 'L': 2.72, 'T': -0.65, 'C': 2.66, 'I': 3.1, 'G': 0.15, 'V': 2.64, 'K': -3.89, 'M': 1.89,
        'F': 3.12, 'N': -2.02, 'R': -2.8, 'H': -0.39, 'E': -3.08, 'W': 1.89, 'A': 0.57, 'D': -2.46, 'Y': 0.79,
        'S': -1.1, 'P': -0.58}
gg_2 = {'Q': 1.82, 'L': 1.88, 'T': -1.6, 'C': -1.52, 'I': 0.37, 'G': -3.49, 'V': 0.03, 'K': 1.47, 'M': 3.88,
        'F': 0.68, 'N': -1.92, 'R': 0.31, 'H': 1, 'E': 3.45, 'W': -0.09, 'A': 3.37, 'D': -0.66, 'Y': -2.62,
        'S': -2.05, 'P': -4.33}
gg_3 = {'Q': -0.82, 'L': 1.92, 'T': -1.39, 'C': -3.29, 'I': 0.26, 'G': -2.97, 'V': -0.67, 'K': 1.95, 'M': -1.57,
        'F': 2.4, 'N': 0.04, 'R': 2.84, 'H': -0.63, 'E': 0.05, 'W': 4.21, 'A': -3.66, 'D': -0.57, 'Y': 4.11,
        'S': -2.19, 'P': -0.02}
gg_4 = {'Q': -1.85, 'L': 5.33, 'T': 0.63, 'C': -3.77, 'I': 1.04, 'G': 2.06, 'V': 2.34, 'K': 1.17, 'M': -3.58,
        'F': -0.35, 'N': -0.65, 'R': 0.25, 'H': -3.49, 'E': 0.62, 'W': -2.77, 'A': 2.34, 'D': 0.14, 'Y': -0.63,
        'S': 1.36, 'P': -0.21}
gg_5 = {'Q': 0.09, 'L': 0.08, 'T': 1.35, 'C': 2.96, 'I': -0.05, 'G': 0.7, 'V': 0.64, 'K': 0.53, 'M': -2.55,
        'F': -0.88, 'N': 1.61, 'R': 0.2, 'H': 0.05, 'E': -0.49, 'W': 0.72, 'A': -1.07, 'D': 0.75, 'Y': 1.89,
        'S': 1.78, 'P': -8.31}
gg_6 = {'Q': 0.6, 'L': 0.09, 'T': -2.45, 'C': -2.23, 'I': -1.18, 'G': 7.47, 'V': -2.01, 'K': 0.1, 'M': 2.07,
        'F': 1.62, 'N': 2.08, 'R': -0.37, 'H': 0.41, 'E': 0, 'W': 0.86, 'A': -0.4, 'D': 0.24, 'Y': -0.53,
        'S': -3.36, 'P': -1.82}
gg_7 = {'Q': 0.25, 'L': 0.27, 'T': -0.65, 'C': 0.44, 'I': -0.21, 'G': 0.41, 'V': -0.33, 'K': 4.01, 'M': 0.84,
        'F': -0.15, 'N': 0.4, 'R': 3.81, 'H': 1.61, 'E': -5.66, 'W': -1.07, 'A': 1.23, 'D': -5.15, 'Y': -1.3,
        'S': 1.39, 'P': -0.12}
gg_8 = {'Q': 2.11, 'L': -4.06, 'T': 3.43, 'C': -3.49, 'I': 3.45, 'G': 1.62, 'V': 3.93, 'K': -0.01, 'M': 1.85,
        'F': -0.41, 'N': -2.47, 'R': 0.98, 'H': -0.6, 'E': -0.11, 'W': -1.66, 'A': -2.32, 'D': -1.17, 'Y': 1.31,
        'S': -1.21, 'P': -1.18}
gg_9 = {'Q': -1.92, 'L': 0.43, 'T': 0.34, 'C': 2.22, 'I': 0.86, 'G': -0.47, 'V': -0.21, 'K': -0.26, 'M': -2.05,
        'F': 4.2, 'N': -0.07, 'R': 2.43, 'H': 3.55, 'E': 1.49, 'W': -5.87, 'A': -2.01, 'D': 0.73, 'Y': -0.56,
        'S': -2.83, 'P': 0}
gg_10 = {'Q': -1.67, 'L': -1.2, 'T': 0.24, 'C': -3.78, 'I': 1.98, 'G': -2.9, 'V': 1.27, 'K': -1.66, 'M': 0.78,
         'F': 0.73, 'N': 7.02, 'R': -0.99, 'H': 1.52, 'E': -2.26, 'W': -0.66, 'A': 1.31, 'D': 1.5, 'Y': -0.95,
         'S': 0.39, 'P': -0.66}
gg_11 = {'Q': 0.7, 'L': 0.67, 'T': -0.53, 'C': 1.98, 'I': 0.89, 'G': -0.98, 'V': 0.43, 'K': 5.86, 'M': 1.53,
         'F': -0.56, 'N': 1.32, 'R': -4.9, 'H': -2.28, 'E': -1.62, 'W': -2.49, 'A': -1.14, 'D': 1.51, 'Y': 1.91,
         'S': -2.92, 'P': 0.64}
gg_12 = {'Q': -0.27, 'L': -0.29, 'T': 1.91, 'C': -0.43, 'I': -1.67, 'G': -0.62, 'V': -1.71, 'K': -0.06, 'M': 2.44,
         'F': 3.54, 'N': -2.44, 'R': 2.09, 'H': -3.12, 'E': -3.97, 'W': -0.3, 'A': 0.19, 'D': 5.61, 'Y': -1.26,
         'S': 1.27, 'P': -0.92}
gg_13 = {'Q': -0.99, 'L': -2.47, 'T': 2.66, 'C': -1.03, 'I': -1.02, 'G': -0.11, 'V': -2.93, 'K': 1.38, 'M': -0.26,
         'F': 5.25, 'N': 0.37, 'R': -3.08, 'H': -1.45, 'E': 2.3, 'W': -0.5, 'A': 1.66, 'D': -3.85, 'Y': 1.57,
         'S': 2.86, 'P': -0.37}
gg_14 = {'Q': -1.56, 'L': -4.79, 'T': -3.07, 'C': 0.93, 'I': -1.21, 'G': 0.15, 'V': 4.22, 'K': 1.78, 'M': -3.09,
         'F': 1.73, 'N': -0.89, 'R': 0.82, 'H': -0.77, 'E': -0.06, 'W': 1.64, 'A': 4.39, 'D': 1.28, 'Y': 0.2,
         'S': -1.88, 'P': 0.17}
gg_15 = {'Q': 6.22, 'L': 0.8, 'T': 0.2, 'C': 1.43, 'I': -1.78, 'G': -0.53, 'V': 1.06, 'K': -2.71, 'M': -1.39,
         'F': 2.14, 'N': 3.13, 'R': 1.32, 'H': -4.18, 'E': -0.35, 'W': -0.72, 'A': 0.18, 'D': -1.98, 'Y': -0.76,
         'S': -2.42, 'P': 0.36}
gg_16 = {'Q': -0.18, 'L': -1.43, 'T': -2.2, 'C': 1.45, 'I': 5.71, 'G': 0.35, 'V': -1.31, 'K': 1.62, 'M': -1.02,
         'F': 1.1, 'N': 0.79, 'R': 0.69, 'H': -2.91, 'E': 1.51, 'W': 1.75, 'A': -2.6, 'D': 0.05, 'Y': -5.19,
         'S': 1.75, 'P': 0.08}
gg_17 = {'Q': 2.72, 'L': 0.63, 'T': 3.73, 'C': -1.15, 'I': 1.54, 'G': 0.3, 'V': -1.97, 'K': 0.96, 'M': -4.32,
         'F': 0.68, 'N': -1.54, 'R': -2.62, 'H': 3.37, 'E': -2.29, 'W': 2.73, 'A': 1.49, 'D': 0.9, 'Y': -2.56,
         'S': -2.77, 'P': 0.16}
gg_18 = {'Q': 4.35, 'L': -0.24, 'T': -5.46, 'C': -1.64, 'I': 2.11, 'G': 0.32, 'V': -1.21, 'K': -1.09, 'M': -1.34,
         'F': 1.46, 'N': -1.71, 'R': -1.49, 'H': 1.87, 'E': -1.47, 'W': -2.2, 'A': 0.46, 'D': 1.38, 'Y': 2.87,
         'S': 3.36, 'P': -0.34}
gg_19 = {'Q': 0.92, 'L': 1.01, 'T': -0.73, 'C': -1.05, 'I': -4.18, 'G': 0.05, 'V': 4.77, 'K': 1.36, 'M': 0.09,
         'F': 2.33, 'N': -0.25, 'R': -2.57, 'H': 2.17, 'E': 0.15, 'W': 0.9, 'A': -4.22, 'D': -0.03, 'Y': -3.43,
         'S': 2.67, 'P': 0.04}
GEORGIEV_PARAMETERS = [gg_1, gg_2, gg_3, gg_4, gg_5, gg_6, gg_7, gg_8, gg_9,
                       gg_10, gg_11, gg_12, gg_13, gg_14, gg_15, gg_16, gg_17,
                       gg_18, gg_19]
GEORGIEV_AA_TO_VECTOR = {
    aa: np.array([param[aa] for param in GEORGIEV_PARAMETERS], dtype=np.float32)
    for aa in aaList
}


def one_hot_encode_sequence(sequence, max_length):
    one_hot = np.zeros((max_length, len(aaList)), dtype=np.float32)
    sequence = sequence.upper()
    for i, aa in enumerate(sequence[:max_length]):
        if aa in aa2idx:
            one_hot[i, aa2idx[aa]] = 1.0
    return one_hot


def georgiev_encode_sequence(sequence, max_length):
    georgiev = np.zeros((max_length, len(GEORGIEV_PARAMETERS)), dtype=np.float32)
    sequence = sequence.upper()
    for i, aa in enumerate(sequence[:max_length]):
        if aa in GEORGIEV_AA_TO_VECTOR:
            georgiev[i, :] = GEORGIEV_AA_TO_VECTOR[aa]
    return georgiev


CLASSICAL_ENCODER_REGISTRY = {
    'one_hot': one_hot_encode_sequence,
    'georgiev': georgiev_encode_sequence,
}

def _parse_classical_feature_name(feature_name: str) -> Tuple[str, str]:
    """Parse feature-set name into (encoder_name, mode)."""
    name = str(feature_name).strip()
    if name in ("one_hot", "georgiev"):
        return name, "per_residue"
    if name.endswith("_per_residue"):
        encoder = name[: -len("_per_residue")]
        if encoder in CLASSICAL_ENCODER_REGISTRY:
            return encoder, "per_residue"
    if name.endswith("_mean_pooled"):
        encoder = name[: -len("_mean_pooled")]
        if encoder in CLASSICAL_ENCODER_REGISTRY:
            return encoder, "mean_pooled"
    raise ValueError(
        f"Unknown classical feature '{feature_name}'. "
        "Expected one of: one_hot, georgiev, one_hot_per_residue, georgiev_per_residue, "
        "one_hot_mean_pooled, georgiev_mean_pooled."
    )


def _encode_sequence_matrices(
    sequences: Sequence[str],
    encoder_name: str,
    max_length: Optional[int] = None,
) -> List[np.ndarray]:
    """Encode each sequence to a per-residue 2D matrix (L, F)."""
    if encoder_name not in CLASSICAL_ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{encoder_name}'. Available: {sorted(CLASSICAL_ENCODER_REGISTRY)}")
    encode_fn = CLASSICAL_ENCODER_REGISTRY[encoder_name]
    mats: List[np.ndarray] = []
    for seq in sequences:
        target_len = int(max_length) if max_length is not None else len(str(seq))
        mats.append(np.asarray(encode_fn(seq, target_len), dtype=np.float32))
    return mats


def _flatten_per_residue_matrices(
    matrices: Sequence[np.ndarray],
) -> Tuple[Union[np.ndarray, List[np.ndarray]], bool]:
    """Flatten (L,F) per sequence into 1D vectors; return dense or ragged payload."""
    rows = [np.asarray(m).reshape(-1).astype(np.float32, copy=False) for m in matrices]
    if not rows:
        return np.asarray([], dtype=np.float32), False
    lengths = {int(r.shape[0]) for r in rows}
    if len(lengths) == 1:
        return np.stack(rows, axis=0), False
    return rows, True


def _mean_pool_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pool residue axis for each (L, F) matrix -> (N, F)."""
    pooled = [np.asarray(m).mean(axis=0) for m in matrices]
    return np.asarray(pooled, dtype=np.float32)


def _save_output_matrix(array: np.ndarray, output_path: Path, use_sparse: bool) -> str:
    """Save a feature matrix as dense `.npy` or sparse `.npz`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if use_sparse:
        sp.save_npz(str(output_path), sp.csr_matrix(array))
    else:
        np.save(str(output_path), np.asarray(array))
    return str(output_path)


def get_classical_encodings(
    classical_feature_sets,
    sequence_list,
    sequence_base_list=None,
    *,
    encodings_dir: str,
    filename_prefix: str = "",
    get_embeddings_for_seq_base: bool = False,
    max_length=None,
):
    """
    Generate classical sequence encodings and save outputs under `encodings_dir`.

    Supported feature sets:
    - per-residue(flattened): `one_hot`, `georgiev`, `one_hot_per_residue`, `georgiev_per_residue`
    - mean pooled: `one_hot_mean_pooled`, `georgiev_mean_pooled`
    """
    out_dir = Path(encodings_dir)
    file_prefix = str(filename_prefix or "")
    results = {}
    # Section 1: compute each classical feature and save dense/sparse/ragged outputs.
    for feature_name in classical_feature_sets:
        print(f'Obtaining {feature_name} encodings...')
        encoder_name, mode = _parse_classical_feature_name(str(feature_name).strip())
        canonical_feature_name = encoder_name if mode == "per_residue" else f"{encoder_name}_mean_pooled"
        stem = f"{file_prefix}{_sanitize_name(canonical_feature_name)}"
        use_sparse = bool(mode == "per_residue" and encoder_name == "one_hot")

        seq_mats = _encode_sequence_matrices(sequence_list, encoder_name=encoder_name, max_length=max_length)
        if mode == "mean_pooled":
            seq_payload = _mean_pool_matrices(seq_mats)
            seq_encoding_path = out_dir / f"{stem}.npy"
            seq_encoding_saved = _save_output_matrix(seq_payload, seq_encoding_path, use_sparse=False)
            seq_shape = list(seq_payload.shape)
            is_ragged = False
        else:
            flat_payload, is_ragged = _flatten_per_residue_matrices(seq_mats)
            if is_ragged:
                # Save ragged flattened rows under indexed keys ("0","1",...).
                ragged_paths = save_layerwise_embeddings(
                    {0: flat_payload},
                    output_stem=(out_dir / stem),
                    suffix="per_residue",
                    log_tag="get_classical_encodings",
                )
                seq_encoding_saved = str(next(iter(ragged_paths.values())))
                seq_shape = [len(flat_payload), -1]  # type: ignore[arg-type]
            else:
                seq_encoding_path = out_dir / f"{stem}{'.npz' if use_sparse else '.npy'}"
                seq_encoding_saved = _save_output_matrix(flat_payload, seq_encoding_path, use_sparse=use_sparse)  # type: ignore[arg-type]
                seq_shape = list(np.asarray(flat_payload).shape)

        # Section 2: optional base-sequence outputs.
        seq_base_encoding_saved = None
        seq_base_shape = None
        if get_embeddings_for_seq_base and sequence_base_list is not None:
            base_mats = _encode_sequence_matrices(sequence_base_list, encoder_name=encoder_name, max_length=max_length)
            if mode == "mean_pooled":
                base_payload = _mean_pool_matrices(base_mats)
                seq_base_encoding_path = out_dir / f"{stem}_base.npy"
                seq_base_encoding_saved = _save_output_matrix(base_payload, seq_base_encoding_path, use_sparse=False)
                seq_base_shape = list(base_payload.shape)
            else:
                base_flat_payload, base_is_ragged = _flatten_per_residue_matrices(base_mats)
                if base_is_ragged:
                    base_paths = save_layerwise_embeddings(
                        {0: base_flat_payload},
                        output_stem=(out_dir / f"{stem}_base"),
                        suffix="per_residue",
                        log_tag="get_classical_encodings",
                    )
                    seq_base_encoding_saved = str(next(iter(base_paths.values())))
                    seq_base_shape = [len(base_flat_payload), -1]  # type: ignore[arg-type]
                else:
                    seq_base_encoding_path = out_dir / f"{stem}_base{'.npz' if use_sparse else '.npy'}"
                    seq_base_encoding_saved = _save_output_matrix(
                        base_flat_payload,  # type: ignore[arg-type]
                        seq_base_encoding_path,
                        use_sparse=use_sparse,
                    )
                    seq_base_shape = list(np.asarray(base_flat_payload).shape)

        results[str(feature_name).strip()] = {
            "feature_name": str(feature_name).strip(),
            "canonical_feature_name": canonical_feature_name,
            "encoder_name": encoder_name,
            "mode": mode,
            "seq_encoding_path": seq_encoding_saved,
            "seq_encoding_shape": seq_shape,
            "seq_base_encoding_path": seq_base_encoding_saved,
            "seq_base_encoding_shape": seq_base_shape,
            "sparse": use_sparse,
            "ragged": bool(is_ragged),
        }
    return results
