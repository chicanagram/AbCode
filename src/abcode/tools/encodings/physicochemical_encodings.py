#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from protlearn.features import aac, aaindex1, ctd, ctdc, ctdt, ctdd, apaac
from typing import List, Optional, Sequence

from project_config.variables import aaList, aa2idx
from abcode.tools.utils.seq_utils import fetch_sequences_from_fasta
from abcode.tools.encodings.common import _sanitize_name, save_layerwise_embeddings


def encode_length(sequences):
    return np.array([len(seq) for seq in sequences]).reshape(-1,1)


def encode_aac(sequences):
    # amino acid composition
    comp, aa = aac(sequences, remove_zero_cols=False)
    return comp


def encode_aaindex1(sequences):
    # aaindex
    aaind, inds = aaindex1(sequences, standardize='none') # standardize: 'none', 'zscore', 'minmax'
    return aaind


def encode_ctdc(sequences):
    c, desc = ctdc(sequences)
    return c


def encode_ctdt(sequences):
    t, desc = ctdt(sequences)
    return t


PHYSICOCHEMICAL_ENCODER_REGISTRY = {
    'length': encode_length,
    'aac': encode_aac,
    'aaindex1': encode_aaindex1,
    'ctdc': encode_ctdc,
    'ctdt': encode_ctdt,
}


def _encode_sequence_matrices(
    sequences: Sequence[str],
    encoder_name: str,
) -> List[np.ndarray]:
    """Encode each sequence to a per-residue 2D matrix (L, F)."""
    if encoder_name not in PHYSICOCHEMICAL_ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{encoder_name}'. Available: {sorted(PHYSICOCHEMICAL_ENCODER_REGISTRY)}")
    encode_fn = PHYSICOCHEMICAL_ENCODER_REGISTRY[encoder_name]
    mat = []
    for sequence in sequences:
        if sequence!='X':
            vect = encode_fn(sequence)[0,:]
        else:
            vect = [np.nan]*len(vect)
        mat.append(list(vect))
    mat = np.array(mat)
    return mat


def _save_output_matrix(array: np.ndarray, output_path: Path, use_sparse: bool) -> str:
    """Save a feature matrix as dense `.npy` or sparse `.npz`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if use_sparse:
        sp.save_npz(str(output_path), sp.csr_matrix(array))
    else:
        np.save(str(output_path), np.asarray(array))
    return str(output_path)


def get_physicochemical_encodings(
    physicochemical_feature_sets,
    sequence_list,
    sequence_base_list=None,
    *,
    encodings_dir: str,
    filename_prefix: str = "",
    get_embeddings_for_seq_base: bool = False,
    max_length=None,
):
    """
    Generate physicochemical sequence encodings and save outputs under `encodings_dir`.
    """
    out_dir = Path(encodings_dir)
    file_prefix = str(filename_prefix or "")
    results = {}
    # Section 1: compute each classical feature and save dense/sparse/ragged outputs.
    for encoder_name in physicochemical_feature_sets:
        print(f'Obtaining {encoder_name} encodings...')
        mat = _encode_sequence_matrices(sequence_list, encoder_name=encoder_name)
        stem = f"{file_prefix}{encoder_name}"
        seq_encoding_path = out_dir / f"{stem}.npy"
        seq_encoding_saved = _save_output_matrix(mat, seq_encoding_path, use_sparse=False)

        results[str(encoder_name).strip()] = {
            "feature_name": str(encoder_name).strip(),
            "canonical_feature_name": encoder_name,
            "encoder_name": encoder_name,
            "seq_encoding_path": seq_encoding_saved,
            "seq_encoding_shape": mat.shape,
        }
    return results
