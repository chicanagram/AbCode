from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[3]
    _src_root = _repo_root / "src"
    for _path in (str(_repo_root), str(_src_root)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from abcode.core.paths import resolve_project_root
from project_config.feature_registry import (
    CLASSICAL_ENCODING_FEATURE_SETS,
    PHYSICOCHEMICAL_ENCODING_FEATURE_SETS,
    FEATURE_SETS_DEFAULT,
    PLM_MODELS_DICT,
    PLM_ENCODING_FEATURE_SETS,
)
from project_config.variables import address_dict
from abcode.tools.encodings.classical_encodings import get_classical_encodings
from abcode.tools.encodings.physicochemical_encodings import get_physicochemical_encodings
from abcode.tools.encodings.common import _sanitize_name
from abcode.tools.encodings.plm_encodings import get_plm_encodings
from abcode.tools.utils.seq_utils import fetch_sequences_from_fasta, get_mutated_sequence, normalize_sequences


FASTA_SUFFIXES = {".fasta", ".fa", ".faa", ".fna"}
PLM_FEATURE_SUFFIXES = ("_mean_pooled", "_mut_pooled", "_svd_pooled", "_per_residue", "_PLL", "_LLR")
PLM_CHUNKABLE_SUFFIXES = ("_mean_pooled", "_mut_pooled", "_svd_pooled")
EMBEDDING_TRACE_SUFFIXES = ("_per_residue", "_mean_pooled", "_mut_pooled")


def default_user_inputs() -> Dict[str, Any]:
    """Return editable defaults for sequence-encoding runs."""
    return {
        "root_key": "examples",
        "data_subfolder": "",
        "encodings_subfolder": "encodings/",
        "filename_prefix": "",
        "sequence_input": "",
        "sequence_col": "sequence",
        "sequence_base_col": "sequence_base",
        "mutation_col": "mutations",
        "allow_null_sequences": False,
        "null_sequence_placeholder": "X",
        "sequence_base": None,
        "feature_sets": list(FEATURE_SETS_DEFAULT),
        "get_embeddings_for_seq_base": False,
        "classical_max_length": None,
        "marginal_type": "wt",
        "llr_cache_vect_filename_prefix": "",
        "resave_llr_cache_if_found": False,
        "mutations_sep": "+",
        "layers": {k: list(v) for k, v in PLM_MODELS_DICT.items()},
        "n_components": 256,
        "sample_mutants_for_svd": False,
        "svd_data_reduction": None,
        "chunk_size": 4000,
        "cleanup_chunk_files": True,
        "batch_size": 4,
        "device": None,
        "save_per_residue_embeddings": True,
    }


def split_feature_sets(
    feature_sets: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Split requested feature-set names into classical vs PLM groups.

    Unknown feature-set names raise an error.
    """
    # Section 1: normalize requested feature-set names.
    if feature_sets is None:
        requested = list(FEATURE_SETS_DEFAULT)
    elif isinstance(feature_sets, str):
        requested = [x.strip() for x in feature_sets.split(",") if x.strip()]
    else:
        requested = [str(x).strip() for x in feature_sets if str(x).strip()]
    if not requested:
        requested = list(FEATURE_SETS_DEFAULT)

    # Section 2: preserve order while de-duplicating names.
    deduped: List[str] = []
    seen = set()
    for name in requested:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    requested = deduped

    # Section 2: split by backend ownership.
    classical = [f for f in requested if f in CLASSICAL_ENCODING_FEATURE_SETS]
    physicochemical =  [f for f in requested if f in PHYSICOCHEMICAL_ENCODING_FEATURE_SETS]
    plm = [f for f in requested if f in PLM_ENCODING_FEATURE_SETS]

    # Section 3: reject unknown feature names early.
    unknown = [
        f
        for f in requested
        if f not in CLASSICAL_ENCODING_FEATURE_SETS + PHYSICOCHEMICAL_ENCODING_FEATURE_SETS + PLM_ENCODING_FEATURE_SETS
    ]
    if unknown:
        raise ValueError(
            "Unknown feature sets: "
            f"{unknown}. Available feature sets are defined in project_config/feature_registry.py"
        )

    return {
        "requested": requested,
        "classical": classical,
        "physicochemical": physicochemical,
        "plm": plm,
    }


def _normalize_required_sequence_value(value: str, *, field_name: str, row_idx: int) -> str:
    normalized = normalize_sequences([str(value).strip()])
    if len(normalized) != 1:
        raise ValueError(f"Could not normalize {field_name} at row {row_idx} into a non-empty sequence.")
    return normalized[0]


def _infer_sequence_for_row(
    *,
    seq_base: str,
    mut_str: str,
    mutations_sep: str,
    row_idx: int,
) -> str:
    curr_seq = _normalize_required_sequence_value(seq_base, field_name="sequence_base", row_idx=row_idx)
    raw_mut = str(mut_str or "").strip()
    if not raw_mut:
        return curr_seq
    mut_tokens = [m.strip() for m in raw_mut.split(str(mutations_sep or "+")) if m.strip()]
    for mut in mut_tokens:
        _, _, seq_out, _ = get_mutated_sequence(curr_seq, [mut], seq_name="inferred", write_to_fasta=None)
        if not seq_out:
            raise ValueError(f"Failed to infer sequence at row {row_idx} for mutation '{mut}'.")
        curr_seq = str(seq_out[0]).strip()
    return _normalize_required_sequence_value(curr_seq, field_name="inferred sequence", row_idx=row_idx)


def _infer_sequences_from_base_and_mutations(
    *,
    sequence_base_list: Sequence[str],
    mutations_raw: Sequence[str],
    mutations_sep: str = "+",
) -> List[str]:
    """
    Infer mutated sequences row-wise from sequence_base + mutation strings.

    Mutation strings may contain multi-mutations separated by `mutations_sep`.
    """
    # Section 1: validate row alignment.
    if len(sequence_base_list) != len(mutations_raw):
        raise ValueError(
            f"Length mismatch: sequence_base_list ({len(sequence_base_list)}) vs "
            f"mutations ({len(mutations_raw)})."
        )

    # Section 2: apply mutations sequentially for each row.
    inferred_sequences: List[str] = []
    for row_idx, (seq_base, mut_str) in enumerate(zip(sequence_base_list, mutations_raw)):
        curr_seq = str(seq_base).strip()
        if not curr_seq:
            raise ValueError(f"Empty sequence_base at row {row_idx}; cannot infer sequence.")
        raw_mut = str(mut_str or "").strip()
        if not raw_mut:
            inferred_sequences.append(curr_seq)
            continue
        mut_tokens = [m.strip() for m in raw_mut.split(str(mutations_sep or "+")) if m.strip()]
        for mut in mut_tokens:
            _, _, seq_out, _ = get_mutated_sequence(curr_seq, [mut], seq_name="inferred", write_to_fasta=None)
            if not seq_out:
                raise ValueError(f"Failed to infer sequence at row {row_idx} for mutation '{mut}'.")
            curr_seq = str(seq_out[0]).strip()
        inferred_sequences.append(curr_seq)

    # Section 3: normalize and enforce non-empty inferred sequences.
    normalized = normalize_sequences(inferred_sequences)
    if len(normalized) != len(inferred_sequences):
        raise ValueError("Could not normalize all inferred sequences into non-empty strings.")
    return normalized


def _normalize_sequence_base_input(
    sequence_base: Optional[Union[str, Sequence[Optional[str]]]],
    n_rows: int = 1,
) -> Optional[List[str]]:
    """Normalize optional sequence_base into a list of non-empty sequences."""
    # Section 1: no sequence_base provided.
    if sequence_base is None:
        return None

    # Section 2: single-sequence input.
    if isinstance(sequence_base, str):
        normalized = normalize_sequences(sequence_base)
        if not normalized:
            return None
        return [normalized[0]] * max(int(n_rows), 1)

    # Section 3: list input.
    raw = [str(x).strip() if x is not None else "" for x in sequence_base]
    if not any(raw):
        return None
    if any(not x for x in raw):
        raise ValueError("sequence_base list contains empty rows; provide non-empty sequences only.")
    normalized = normalize_sequences(raw)
    if len(normalized) == 1 and n_rows > 1:
        return [normalized[0]] * n_rows
    if len(normalized) != n_rows:
        raise ValueError(
            f"sequence_base list length ({len(normalized)}) must be 1 or match n_rows ({n_rows})."
        )
    return normalized


def _resolve_sequence_base_user_input(
    sequence_base: Optional[Union[str, Sequence[Optional[str]]]],
    data_fbase: Optional[Path] = None,
) -> Optional[Union[str, Sequence[Optional[str]]]]:
    """
    Resolve `sequence_base` when provided as a FASTA filepath.

    Supported inputs:
    - `None`: unchanged.
    - explicit amino-acid string: unchanged.
    - FASTA path string (absolute or relative to `data_fbase`): replaced with first sequence in FASTA.
    - list input: unchanged.
    """
    # Section 1: no input or non-string types pass through unchanged.
    if sequence_base is None or not isinstance(sequence_base, str):
        return sequence_base

    token = str(sequence_base).strip()
    if not token:
        return None

    # Section 2: treat FASTA-like strings as file paths and load first sequence.
    p = Path(token).expanduser()
    if p.suffix.lower() in FASTA_SUFFIXES:
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            if data_fbase is not None:
                candidates.append((Path(data_fbase) / p).resolve())
            candidates.append(p.resolve())

        fasta_path = next((c for c in candidates if c.exists()), None)
        if fasta_path is None:
            raise FileNotFoundError(
                f"sequence_base FASTA file not found: '{token}'. "
                f"Checked: {[str(c) for c in candidates]}"
            )

        sequences, _, _ = fetch_sequences_from_fasta(str(fasta_path))
        if not sequences:
            raise ValueError(f"No sequences found in sequence_base FASTA: {fasta_path}")
        return str(sequences[0]).strip()

    # Section 3: explicit sequence string path.
    return sequence_base


def parse_sequence_input(
    *,
    sequence_input: Union[str, Path],
    sequence_col: str = "sequence",
    sequence_base_col: str = "sequence_base",
    mutation_col: str = "mutations",
    mutations_sep: str = "+",
    allow_null_sequences: bool = False,
    null_sequence_placeholder: Optional[str] = None,
    sequence_base: Optional[Union[str, Sequence[Optional[str]]]] = None,
) -> Dict[str, Any]:
    """
    Parse CSV or FASTA input into sequence lists used by encoding pipelines.

    Returns keys: `sequence_list`, `sequence_base_list`, `mutations_list`.
    """
    # Section 1: resolve and validate input file path.
    input_path = Path(sequence_input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Section 2: route parser logic by file extension.
    suffix = input_path.suffix.lower()

    # CSV parsing pathway.
    if suffix == ".csv":
        # Section 2A: parse CSV sequence/base/mutation columns.
        df = pd.read_csv(input_path)
        n_rows = len(df)
        n_seq = n_rows

        # Section 2A.1: resolve sequence_base list from CSV column or fallback input.
        sequence_base_list: Optional[List[str]] = None
        if sequence_base_col in df.columns:
            raw_base = [str(x).strip() if pd.notna(x) else "" for x in df[sequence_base_col].tolist()]
            if any(raw_base):
                if any(not x for x in raw_base):
                    raise ValueError(
                        f"Column '{sequence_base_col}' contains empty rows; fill all rows or remove the column."
                    )
                sequence_base_list = normalize_sequences(raw_base)
                if len(sequence_base_list) != n_seq:
                    raise ValueError(
                        f"Could not normalize all values in '{sequence_base_col}' into non-empty sequences."
                    )
            else:
                sequence_base_list = _normalize_sequence_base_input(sequence_base, n_rows=n_seq)
        else:
            sequence_base_list = _normalize_sequence_base_input(sequence_base, n_rows=n_seq)

        mutations_list: Optional[List[str]] = None
        raw_mutations: Optional[List[str]] = None
        if mutation_col in df.columns:
            raw_mutations = [str(x).strip() for x in df[mutation_col].fillna("").tolist()]
            if any(raw_mutations):
                mutations_list = raw_mutations

        sequence_list: Optional[List[str]] = None
        null_sequence_row_indices: List[int] = []
        placeholder_filled_row_indices: List[int] = []

        # Section 2A.2: resolve sequence column (with optional null-row preservation).
        if sequence_col in df.columns:
            raw_sequence_vals = df[sequence_col].tolist()
            if null_sequence_placeholder is not None and str(null_sequence_placeholder).strip():
                placeholder_seq = _normalize_required_sequence_value(
                    str(null_sequence_placeholder),
                    field_name="null_sequence_placeholder",
                    row_idx=-1,
                )
            else:
                placeholder_seq = "X"
            resolved_sequences: List[str] = []
            for row_idx, raw_val in enumerate(raw_sequence_vals):
                has_value = pd.notna(raw_val) and str(raw_val).strip() != ""
                if has_value:
                    resolved_sequences.append(
                        _normalize_required_sequence_value(
                            str(raw_val),
                            field_name=sequence_col,
                            row_idx=row_idx,
                        )
                    )
                    continue

                null_sequence_row_indices.append(row_idx)
                if not allow_null_sequences:
                    # Keep historical strict behavior unless explicitly enabled.
                    raise ValueError(
                        f"Found null/empty values in '{sequence_col}' at rows "
                        f"{null_sequence_row_indices[:5]}."
                    )

                if sequence_base_list is None:
                    # No per-row base available: preserve row count with placeholder sequence.
                    resolved_sequences.append(placeholder_seq)
                    placeholder_filled_row_indices.append(row_idx)
                    continue

                base_row = sequence_base_list[row_idx]
                mut_row = raw_mutations[row_idx] if raw_mutations is not None else ""
                inferred = _infer_sequence_for_row(
                    seq_base=base_row,
                    mut_str=mut_row,
                    mutations_sep=mutations_sep,
                    row_idx=row_idx,
                )
                resolved_sequences.append(inferred)
            sequence_list = resolved_sequences
        else:
            if sequence_base_list is None:
                raise ValueError(
                    f"Column '{sequence_col}' not found in CSV: {input_path}. "
                    f"To infer sequences, provide '{sequence_base_col}' column or user_inputs['sequence_base']."
                )
            if raw_mutations is None:
                raise ValueError(
                    f"Column '{sequence_col}' not found in CSV: {input_path}. "
                    f"To infer sequences, provide '{mutation_col}' column."
                )
            sequence_list = _infer_sequences_from_base_and_mutations(
                sequence_base_list=sequence_base_list,
                mutations_raw=raw_mutations,
                mutations_sep=mutations_sep,
            )
            # Section 2A.3: persist inferred sequence column back to source CSV.
            df[sequence_col] = sequence_list
            df.to_csv(input_path, index=False)

        return {
            "input_path": str(input_path),
            "input_type": "csv",
            "n_sequences": n_seq,
            "sequence_list": sequence_list,
            "sequence_base_list": sequence_base_list,
            "mutations_list": mutations_list,
            "null_sequence_row_indices": null_sequence_row_indices,
            "placeholder_filled_row_indices": placeholder_filled_row_indices,
        }

    # FASTA parsing pathway.
    if suffix in FASTA_SUFFIXES:
        # Section 2B: parse FASTA sequences and optional user-provided base sequence.
        sequence_list, sequence_names, _ = fetch_sequences_from_fasta(str(input_path))
        if not sequence_list:
            raise ValueError(f"No sequences found in FASTA: {input_path}")
        raw_sequence_list = [str(seq).strip() for seq in sequence_list]
        sequence_list = normalize_sequences(raw_sequence_list)
        if len(sequence_list) != len(raw_sequence_list):
            raise ValueError(f"Could not normalize all FASTA entries into non-empty sequences: {input_path}")
        n_seq = len(sequence_list)
        sequence_base_list = _normalize_sequence_base_input(sequence_base, n_rows=n_seq)

        return {
            "input_path": str(input_path),
            "input_type": "fasta",
            "n_sequences": n_seq,
            "sequence_names": sequence_names,
            "sequence_list": sequence_list,
            "sequence_base_list": sequence_base_list,
            "mutations_list": None,
            "null_sequence_row_indices": [],
            "placeholder_filled_row_indices": [],
        }

    raise ValueError(
        f"Unsupported input format '{suffix}'. Provide a CSV or FASTA file ({sorted(FASTA_SUFFIXES)})."
    )


def _slice_or_none(values: Optional[Sequence[Any]], start: int, end: int):
    if values is None:
        return None
    return list(values[start:end])


def _get_model_prefix(feature_name: str) -> str:
    name = str(feature_name).strip()
    for suffix in PLM_FEATURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _split_chunked_plm_features(plm_features: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split PLM features for chunk mode into:
    - per_residue_features (unsupported in chunk mode)
    - chunked_features (pooled embeddings)
    - passthrough_features (LLR/PLL)
    """
    per_residue_features = [f for f in plm_features if str(f).endswith("_per_residue")]
    chunked_features = [f for f in plm_features if str(f).endswith(PLM_CHUNKABLE_SUFFIXES)]
    passthrough_features = [f for f in plm_features if f not in chunked_features]
    return per_residue_features, chunked_features, passthrough_features


def _collect_embedding_shape_trace(plm_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Collect compact embedding shape trace payload from PLM results."""
    embedding_shape_trace: Dict[str, Any] = {}
    for feature_name, meta in plm_results.items():
        if not str(feature_name).endswith(EMBEDDING_TRACE_SUFFIXES):
            continue
        shape_by_layer = meta.get("shape_by_layer")
        base_shape_by_layer = meta.get("base_shape_by_layer")
        if shape_by_layer is None and base_shape_by_layer is None:
            continue
        embedding_shape_trace[feature_name] = {
            "shape_by_layer": shape_by_layer,
            "base_shape_by_layer": base_shape_by_layer,
        }
    return embedding_shape_trace


def _apply_null_mask_to_npy(path: Path, null_rows: Sequence[int]) -> bool:
    arr = np.load(str(path), allow_pickle=False)
    if arr.ndim < 1 or not null_rows:
        return False
    valid_rows = [i for i in null_rows if 0 <= int(i) < int(arr.shape[0])]
    if not valid_rows:
        return False
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    arr[valid_rows] = np.nan
    np.save(str(path), arr)
    return True


def _apply_null_mask_to_npz(path: Path, null_rows: Sequence[int]) -> str:
    # Section 1: sparse matrices (.npz from scipy.sparse) cannot encode NaN efficiently.
    try:
        from scipy import sparse as sp  # type: ignore

        mat = sp.load_npz(str(path))
        valid_rows = [i for i in null_rows if 0 <= int(i) < int(mat.shape[0])]
        if valid_rows:
            mat = mat.tolil()
            mat[valid_rows, :] = 0.0
            sp.save_npz(str(path), mat.tocsr())
            return "sparse_zeroed"
    except Exception:
        pass

    # Section 2: ragged npz (keyed arrays) -> replace selected rows with NaN arrays.
    with np.load(str(path), allow_pickle=False) as npz:
        payload = {k: np.asarray(npz[k]) for k in npz.files}
    if not payload:
        return "noop"
    for row_idx in null_rows:
        key = str(int(row_idx))
        if key in payload:
            arr = payload[key]
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            payload[key] = np.full(arr.shape, np.nan, dtype=arr.dtype)
    np.savez_compressed(str(path), **payload)
    return "ragged_nan"


def _apply_null_mask_to_path(path: Path, null_rows: Sequence[int]) -> str:
    """
    Apply null-row masking to a single artifact path.

    Returns one of:
    - "noop"
    - "nan_masked"
    - "sparse_zeroed"
    """
    if not path.exists():
        return "noop"
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "nan_masked" if _apply_null_mask_to_npy(path, null_rows) else "noop"
    if suffix == ".npz":
        mode = _apply_null_mask_to_npz(path, null_rows)
        if mode == "sparse_zeroed":
            return "sparse_zeroed"
        if mode == "ragged_nan":
            return "nan_masked"
    return "noop"


def _apply_null_sequence_mask_to_results(
    *,
    null_rows: Sequence[int],
    classical_results: Dict[str, Dict[str, Any]],
    plm_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Write null-sequence row masks into encoded artifacts while preserving row count."""
    if not null_rows:
        return {"applied": False, "null_rows": []}

    applied_to: List[str] = []
    sparse_zeroed: List[str] = []

    # Section 1: classical outputs.
    for feature_name, meta in (classical_results or {}).items():
        seq_path_val = meta.get("seq_encoding_path")
        if not seq_path_val:
            continue
        mode = _apply_null_mask_to_path(Path(str(seq_path_val)), null_rows)
        if mode == "sparse_zeroed":
            sparse_zeroed.append(f"classical:{feature_name}")
        elif mode == "nan_masked":
            applied_to.append(f"classical:{feature_name}")

    # Section 2: PLM score/artifact outputs.
    for feature_name, meta in (plm_results or {}).items():
        output_base = meta.get("output_path")
        if output_base:
            npy_path = Path(f"{output_base}.npy")
            if _apply_null_mask_to_path(npy_path, null_rows) == "nan_masked":
                applied_to.append(f"plm-score:{feature_name}")

        artifacts = meta.get("artifacts", {}) or {}
        for key in ("pooled_paths", "per_residue_paths", "base_pooled_paths", "base_per_residue_paths"):
            layer_paths = artifacts.get(key, {})
            if not isinstance(layer_paths, dict):
                continue
            for _, path_val in layer_paths.items():
                mode = _apply_null_mask_to_path(Path(str(path_val)), null_rows)
                if mode == "sparse_zeroed":
                    sparse_zeroed.append(f"plm-artifact:{feature_name}:{key}")
                elif mode == "nan_masked":
                    applied_to.append(f"plm-artifact:{feature_name}:{key}")

    return {
        "applied": bool(applied_to or sparse_zeroed),
        "null_rows": [int(i) for i in null_rows],
        "nan_masked_outputs": sorted(set(applied_to)),
        "sparse_zeroed_outputs": sorted(set(sparse_zeroed)),
    }


def _build_plm_call_kwargs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build shared `get_plm_encodings` kwargs from user inputs."""
    return {
        "marginal_type": str(inputs.get("marginal_type", "wt")),
        "llr_cache_vect_filename_prefix": str(inputs.get("llr_cache_vect_filename_prefix", "") or "").strip(),
        "resave_llr_cache_if_found": bool(inputs.get("resave_llr_cache_if_found", False)),
        "sep": str(inputs.get("mutations_sep", "+")),
        "layers": inputs.get("layers"),
        "n_components": int(inputs.get("n_components", 256)),
        "sample_mutants_for_svd": bool(inputs.get("sample_mutants_for_svd", False)),
        "svd_data_reduction": inputs.get("svd_data_reduction", None),
        "batch_size": int(inputs.get("batch_size", 4)),
        "device": inputs.get("device"),
    }


def _run_chunked_pooled_plm_features(
    *,
    plm_feature_sets: Sequence[str],
    sequence_list: Sequence[str],
    sequence_base_list: Optional[Sequence[str]],
    mutations_list: Optional[Sequence[str]],
    encodings_dir: str,
    filename_prefix: str,
    chunk_size: int,
    cleanup_chunk_files: bool,
    plm_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Run pooled PLM features in chunks and merge chunk outputs."""
    # Preserve requested order while grouping by model.
    model_to_features: Dict[str, List[str]] = {}
    ordered_models: List[str] = []
    for feat in plm_feature_sets:
        model = _get_model_prefix(feat)
        if model not in model_to_features:
            model_to_features[model] = []
            ordered_models.append(model)
        model_to_features[model].append(feat)

    n = len(sequence_list)
    merged_all: Dict[str, Dict[str, Any]] = {}
    n_chunks = int(np.ceil(n / chunk_size))

    # Run one model across all chunks before moving to the next model.
    for model in ordered_models:
        model_features = model_to_features[model]
        chunk_results: List[Dict[str, Dict[str, Any]]] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = (start // chunk_size) + 1
            print(f"[chunk:{model}] Processing {start}:{end} ({chunk_idx}/{n_chunks})")
            chunk_res = get_plm_encodings(
                plm_feature_sets=model_features,
                sequence_list=sequence_list[start:end],
                sequence_base_list=_slice_or_none(sequence_base_list, start, end),
                encodings_dir=encodings_dir,
                filename_prefix=f"{filename_prefix}chunk{chunk_idx:03d}_",
                mutations=_slice_or_none(mutations_list, start, end),
                get_embeddings_for_seq_base=False,
                **plm_kwargs,
            )
            chunk_results.append(chunk_res)

        merged_model = _merge_chunked_plm_results(
            chunk_results=chunk_results,
            requested_features=model_features,
            encodings_dir=encodings_dir,
            filename_prefix=filename_prefix,
            cleanup_chunk_files=cleanup_chunk_files,
        )
        merged_all.update(merged_model)

    return merged_all


def _merge_chunked_plm_results(
    *,
    chunk_results: List[Dict[str, Dict[str, Any]]],
    requested_features: Sequence[str],
    encodings_dir: Union[str, Path],
    filename_prefix: str,
    cleanup_chunk_files: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Merge chunk-level pooled embedding outputs into final full-dataset arrays.
    """
    out_dir = Path(encodings_dir)
    merged: Dict[str, Dict[str, Any]] = {}
    for feature_name in requested_features:
        chunk_feature_meta = [cr[feature_name] for cr in chunk_results if feature_name in cr]
        if not chunk_feature_meta:
            continue
        artifacts0 = chunk_feature_meta[0].get("artifacts", {})
        pooled_key = "pooled_paths" if "pooled_paths" in artifacts0 else "per_residue_paths"
        if pooled_key not in artifacts0:
            # Non-artifact features (e.g., scalar score outputs) are not chunk-merged in this path.
            continue
        layer_keys = sorted(list(artifacts0[pooled_key].keys()), key=lambda x: int(x))
        final_paths: Dict[str, str] = {}
        final_shapes: Dict[int, List[int]] = {}
        for layer in layer_keys:
            arrays = [
                np.load(str(meta["artifacts"][pooled_key][layer]), allow_pickle=False)
                for meta in chunk_feature_meta
            ]
            concat = np.concatenate(arrays, axis=0)
            final_stem = out_dir / f"{str(filename_prefix)}{_sanitize_name(feature_name)}"
            final_path = final_stem.parent / f"{final_stem.name}-{int(layer)}.npy"
            np.save(str(final_path), concat)
            final_paths[str(int(layer))] = str(final_path)
            final_shapes[int(layer)] = list(concat.shape)

            if cleanup_chunk_files:
                for meta in chunk_feature_meta:
                    try:
                        Path(meta["artifacts"][pooled_key][layer]).unlink(missing_ok=True)
                    except TypeError:
                        p = Path(meta["artifacts"][pooled_key][layer])
                        if p.exists():
                            p.unlink()

        merged[feature_name] = {
            "feature_name": feature_name,
            "model_prefix": chunk_feature_meta[0].get("model_prefix"),
            "plm_name": chunk_feature_meta[0].get("plm_name"),
            "artifacts": {
                pooled_key: final_paths,
                "base_pooled_paths": None,
            },
            "shape_by_layer": final_shapes,
        }
    return merged


def get_sequence_encodings(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse sequence input, split feature sets, and dispatch encoding generation.
    """
    # Section 1: resolve high-level inputs.
    sequence_input = str(inputs.get("sequence_input", "") or "").strip()
    # Section 2: split requested feature sets by encoding backend.
    split_sets = split_feature_sets(inputs.get("feature_sets", FEATURE_SETS_DEFAULT))

    # Section 3: resolve output root and preprocess sequence_base filepath inputs.
    root_key = str(inputs.get("root_key", "examples") or "").strip() or "examples"
    if root_key not in address_dict:
        raise KeyError(f"Unknown root_key: {root_key}. Available keys: {sorted(address_dict.keys())}")
    project_root = resolve_project_root()
    data_fbase = (project_root / address_dict[root_key]).resolve()
    sequence_base_input = _resolve_sequence_base_user_input(inputs.get("sequence_base"), data_fbase)

    # Section 4: parse input sequences and optional metadata (or base-only mode).
    if sequence_input:
        parsed = parse_sequence_input(
            sequence_input=sequence_input,
            sequence_col=str(inputs.get("sequence_col", "sequence")),
            sequence_base_col=str(inputs.get("sequence_base_col", "sequence_base")),
            mutation_col=str(inputs.get("mutation_col", "mutations")),
            mutations_sep=str(inputs.get("mutations_sep", "+")),
            allow_null_sequences=bool(inputs.get("allow_null_sequences", False)),
            null_sequence_placeholder=inputs.get("null_sequence_placeholder"),
            sequence_base=sequence_base_input,
        )
    else:
        sequence_base_list = _normalize_sequence_base_input(sequence_base_input, n_rows=1)
        if not sequence_base_list:
            raise ValueError("Provide either 'sequence_input' (CSV/FASTA) or a non-empty 'sequence_base'.")
        parsed = {
            "input_path": None,
            "input_type": "base_only",
            "n_sequences": 0,
            "sequence_list": None,
            "sequence_base_list": sequence_base_list,
            "mutations_list": None,
        }

    # Section 5: validate feature/input compatibility.
    if split_sets["classical"] and parsed["sequence_list"] is None:
        raise ValueError("Classical encodings require sequence inputs from CSV/FASTA; base-only mode is unsupported.")
    if split_sets["physicochemical"] and parsed["sequence_list"] is None:
        raise ValueError("Physicochemical encodings require sequence inputs from CSV/FASTA; base-only mode is unsupported.")
    if split_sets["plm"] and parsed["sequence_list"] is None:
        non_llr = [f for f in split_sets["plm"] if not str(f).endswith("_LLR")]
        if non_llr:
            raise ValueError(
                "Base-only mode supports only PLM *_LLR feature sets. "
                f"Unsupported in this mode: {non_llr}"
            )

    # Section 6: resolve output directory from root_key + subfolder settings.
    encodings_subfolder = str(inputs.get("encodings_subfolder", "encodings/") or "").strip().strip("/")
    data_subfolder = str(inputs.get("data_subfolder", "") or "").strip().strip("/")
    encodings_path = data_fbase / encodings_subfolder
    if data_subfolder:
        encodings_path = encodings_path / data_subfolder
    encodings_path.mkdir(parents=True, exist_ok=True)
    encodings_dir = str(encodings_path.resolve())

    # Section 7: run encoding generation backend(s) as requested.
    classical_results = {}
    physicochemical_results = {}
    plm_results = {}
    filename_prefix = str(inputs.get("filename_prefix", "") or "")

    if split_sets["classical"]:
        classical_results = get_classical_encodings(
            classical_feature_sets=split_sets["classical"],
            sequence_list=parsed["sequence_list"],
            sequence_base_list=parsed["sequence_base_list"],
            encodings_dir=encodings_dir,
            filename_prefix=filename_prefix,
            get_embeddings_for_seq_base=bool(inputs.get("get_embeddings_for_seq_base", False)),
            max_length=inputs.get("classical_max_length"),
        )

    if split_sets["physicochemical"]:
        physicochemical_results = get_physicochemical_encodings(
            physicochemical_feature_sets=split_sets["physicochemical"],
            sequence_list=parsed["sequence_list"],
            sequence_base_list=parsed["sequence_base_list"],
            encodings_dir=encodings_dir,
            filename_prefix=filename_prefix,
            get_embeddings_for_seq_base=bool(inputs.get("get_embeddings_for_seq_base", False)),
            max_length=inputs.get("classical_max_length"),
        )

    if split_sets["plm"]:
        plm_kwargs = _build_plm_call_kwargs(inputs)
        chunk_size = int(inputs.get("chunk_size", 4000))
        cleanup_chunk_files = bool(inputs.get("cleanup_chunk_files", True))
        sequence_list = parsed["sequence_list"]
        use_chunking = bool(sequence_list is not None and len(sequence_list) > chunk_size)

        # For large datasets, chunk only pooled embedding features.
        # Non-embedding PLM features (e.g. LLR/PLL) run as regular full calls.
        if use_chunking:
            per_residue_features, chunked_features, passthrough_features = _split_chunked_plm_features(
                split_sets["plm"]
            )
            if per_residue_features:
                raise ValueError(
                    "Chunked mode does not support PLM per-residue features. "
                    f"Please remove: {per_residue_features}"
                )

            # Section 7A: run pooled embedding features in chunks and merge outputs.
            chunked_results: Dict[str, Dict[str, Any]] = {}
            if chunked_features:
                chunked_results = _run_chunked_pooled_plm_features(
                    plm_feature_sets=chunked_features,
                    sequence_list=sequence_list,
                    sequence_base_list=parsed["sequence_base_list"],
                    mutations_list=parsed["mutations_list"],
                    encodings_dir=encodings_dir,
                    filename_prefix=filename_prefix,
                    chunk_size=chunk_size,
                    cleanup_chunk_files=cleanup_chunk_files,
                    plm_kwargs=plm_kwargs,
                )

            # Section 7B: run non-chunked PLM features (e.g. LLR/PLL) normally.
            passthrough_results: Dict[str, Dict[str, Any]] = {}
            if passthrough_features:
                passthrough_results = get_plm_encodings(
                    plm_feature_sets=passthrough_features,
                    sequence_list=parsed["sequence_list"],
                    sequence_base_list=parsed["sequence_base_list"],
                    encodings_dir=encodings_dir,
                    filename_prefix=filename_prefix,
                    mutations=parsed["mutations_list"],
                    get_embeddings_for_seq_base=bool(inputs.get("get_embeddings_for_seq_base", False)),
                    **plm_kwargs,
                )
            plm_results = {**passthrough_results, **chunked_results}
        else:
            plm_results = get_plm_encodings(
                plm_feature_sets=split_sets["plm"],
                sequence_list=parsed["sequence_list"],
                sequence_base_list=parsed["sequence_base_list"],
                encodings_dir=encodings_dir,
                filename_prefix=filename_prefix,
                mutations=parsed["mutations_list"],
                get_embeddings_for_seq_base=bool(inputs.get("get_embeddings_for_seq_base", False)),
                **plm_kwargs,
            )

    # Section 8: collect compact shape trace for embedding artifacts.
    embedding_shape_trace = _collect_embedding_shape_trace(plm_results)
    null_mask_trace = _apply_null_sequence_mask_to_results(
        null_rows=parsed.get("null_sequence_row_indices", []),
        classical_results=classical_results,
        plm_results=plm_results,
    )

    # Section 9: return consolidated execution metadata and outputs.
    return {
        "status": "ok",
        "input": parsed,
        "feature_sets": split_sets,
        "encodings_dir": encodings_dir,
        "classical_results": classical_results,
        "plm_results": plm_results,
        "trace": {
            "embedding_shapes": embedding_shape_trace,
            "null_sequence_mask": null_mask_trace,
        },
    }


if __name__ == "__main__":
    # Section 1: mirror notebook-style user inputs for IDE/script execution.
    user_inputs = default_user_inputs()

    # Section 1A: Configure output root and subfolders.
    user_inputs["root_key"] = 'PIPS' # "MUTAGENESIS-DATA-BENCHMARKS"
    user_inputs["data_subfolder"] = 'GOh1052_mutagenesis' # "D7PM05_CLYGR_Somermeyer_2022"
    user_inputs["encodings_subfolder"] = "encodings/"
    filename = f'{user_inputs["data_subfolder"]}.csv'
    user_inputs["filename_prefix"] = filename.split(".")[0] + "_" if filename is not None else ""

    # Section 1B: Configure sequence input source.
    project_root = resolve_project_root()
    data_root = (project_root / Path(address_dict[user_inputs["root_key"]])).resolve()
    user_inputs["sequence_input"] = str(
        data_root / "expdata" / user_inputs["data_subfolder"] / filename
    )

    # Section 1C: Configure columns and optional base-sequence fallback.
    user_inputs["sequence_col"] = "sequence"
    user_inputs["sequence_base_col"] = "sequence_base"
    user_inputs["mutation_col"] = "mutations"
    # Leave as None by default so CSV `sequence_base_col` is used when present.
    # Set this to a FASTA path string (e.g. "sequences/<name>.fasta") if desired.
    user_inputs["sequence_base"] = None
    # user_inputs["sequence_base"] = f'sequences/{user_inputs["data_subfolder"]}.fasta'  # None


    # Section 1D: Configure feature sets.
    user_inputs["feature_sets"] = [
        # "one_hot",
        # "georgiev",
        # "esm2-650m_LLR",
        # "esm2-650m_PLL",
        "esm2-650m_PLL_mean_pooled",
        # "esm2-650m_per_residue",
        # "esm2-650m_mean_pooled",
        # "esmc-600m_LLR",
        # "esm2-650m_PLL",
        # "esmc-600m_per_residue",
        # "esmc-600m_mean_pooled",
        # "poet2_LLR",
        # "poet2_PLL",
        # "poet2_per_residue",
        # "poet2_mean_pooled",
        # "poet2_svd_pooled",
    ]

    # Section 1E: Configure runtime controls.
    user_inputs["get_embeddings_for_seq_base"] = False
    user_inputs["classical_max_length"] = None
    user_inputs["marginal_type"] = "masked"
    user_inputs["llr_cache_vect_filename_prefix"] = user_inputs["data_subfolder"] + "_"
    user_inputs["resave_llr_cache_if_found"] = False
    user_inputs["mutations_sep"] = "+"
    user_inputs["layers"] = {"esm2-650m": [33], "esmc-600m": [36], "poet2": [12]}
    user_inputs["n_components"] = 1024
    user_inputs["sample_mutants_for_svd"] = False
    user_inputs["svd_data_reduction"] = None
    user_inputs["chunk_size"] = 200
    user_inputs["cleanup_chunk_files"] = True
    user_inputs["batch_size"] = 4
    user_inputs["device"] = None

    # Section 2: Execute and print concise run summary.
    result = get_sequence_encodings(user_inputs)
    print("status:", result.get("status", ""))
    print("encodings_dir:", result.get("encodings_dir", ""))
    print("n_classical_features:", len(result.get("feature_sets", {}).get("classical", [])))
    print("n_plm_features:", len(result.get("feature_sets", {}).get("plm", [])))
    print("classical result keys:", sorted(result.get("classical_results", {}).keys()))
    print("plm result keys:", sorted(result.get("plm_results", {}).keys()))
