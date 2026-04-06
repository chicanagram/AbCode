from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from project_config.variables import aaList, aaList_with_X
from abcode.tools.utils.plot_utils import plot_variant_heatmap
from abcode.tools.utils.general_utils import flatten_2D_arr

def extract_sequence_token_spans(attention_mask: np.ndarray) -> List[slice]:
    """
    Compute per-sequence residue slices excluding BOS/EOS and padding tokens.
    """
    spans: List[slice] = []
    lengths = attention_mask.sum(axis=1).tolist()
    for length in lengths:
        usable = max(int(length) - 2, 0)
        spans.append(slice(1, 1 + usable))
    return spans


def _sanitize_name(value: str) -> str:
    """Build a filesystem-safe stem for output files."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")


def _resolve_model_name(
        model_name: str,
        model_name_aliases: Dict[str, str],
        default_model_name: str
) -> str:
    """Resolve a shorthand alias or pass through a raw Hugging Face model id."""
    key = str(model_name or default_model_name).strip().lower()
    return model_name_aliases.get(key, str(model_name or default_model_name).strip())


def save_layerwise_embeddings(
    value: Dict[int, Union[np.ndarray, Sequence[np.ndarray]]],
    output_stem: Union[str, Path],
    suffix: str,
    file_suffix: str = "",
    log_tag: str = "get_embeddings",
) -> Dict[str, str]:
    """
    Save one embedding payload per layer as `.npy` (dense) or `.npz` (ragged).

    Output filenames follow:
    - default: `{output_stem}{file_suffix}-{suffix}-{layer}.npy`
    - when `single_layer_flat=True` and one layer: `{output_stem}{file_suffix}.npy`
    """
    # normalize output stem and iterate layer payloads.
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # default layerwise filename pattern with duplicate-suffix guard.
    for layer, layer_value in value.items():
        stem_name = stem.name
        normalized_suffix = str(suffix).strip()
        if normalized_suffix and (
            stem_name.endswith(f"_{normalized_suffix}") or stem_name.endswith(f"-{normalized_suffix}")
        ):
            base = stem.parent / f"{stem_name}{str(file_suffix)}-{int(layer)}"
        else:
            base = stem.parent / f"{stem_name}{str(file_suffix)}-{normalized_suffix}-{int(layer)}"

        is_ragged = isinstance(layer_value, (list, tuple))
        if not is_ragged:
            arr_probe = np.asarray(layer_value)
            is_ragged = bool(arr_probe.dtype == object and arr_probe.ndim == 1)
        if is_ragged:
            ragged = [np.asarray(x) for x in list(layer_value)]
            fpath = base.with_suffix(".npz")
            np.savez_compressed(str(fpath), **{str(i): arr for i, arr in enumerate(ragged)})
            paths[str(int(layer))] = str(fpath)
            print(f"[{log_tag}] Saved {suffix} layer {int(layer)}: {fpath} (n_sequences={len(ragged)})")
            continue

        arr = np.asarray(layer_value)
        fpath = base.with_suffix(".npy")
        np.save(str(fpath), arr.astype(arr.dtype, copy=False))
        paths[str(int(layer))] = str(fpath)
        print(f"[{log_tag}] Saved {suffix} layer {int(layer)}: {fpath} (shape={arr.shape})")
    return paths


def _resolve_layers(requested_layers: Optional[Sequence[int]], model: Any) -> List[int]:
    """Resolve requested hidden-state layers into positive layer indices."""
    max_layer = int(getattr(model.config, "num_hidden_layers", 0))
    if requested_layers is None:
        return [max_layer]
    resolved: List[int] = []
    for layer in requested_layers:
        value = int(layer)
        if value < 0:
            value = max_layer + 1 + value
        if value < 0 or value > max_layer:
            raise ValueError(f"Requested layer {layer} is outside valid range [0, {max_layer}].")
        if value not in resolved:
            resolved.append(value)
    return resolved


def compute_pooled_embeddings(
    embeddings_per_residue: Dict[int, Union[np.ndarray, Sequence[np.ndarray]]],
    pool_method: str = "mean",
    mutations: Optional[Sequence[str]] = None,
    sep: Optional[str] = "+",
) -> Dict[int, np.ndarray]:
    """
    Pool per-residue embedding matrices by layer.

    `mean`: mean over all residues.
    `mut`: mean over mutated residue positions parsed from mutation strings.
    """
    # Section 1: normalize pooling mode and define mutation parsing helper.
    method = str(pool_method or "mean").strip().lower()

    def _parse_mutation_positions(mut: str, token_sep: str) -> List[int]:
        positions: List[int] = []
        for token in str(mut or "").split(token_sep):
            tok = token.strip()
            if len(tok) < 3:
                continue
            if not tok[0].isalpha() or not tok[-1].isalpha():
                continue
            pos_txt = tok[1:-1]
            if pos_txt.isdigit():
                positions.append(int(pos_txt) - 1)  # convert to 0-based
        return positions

    # Section 2: pool each layer matrix to shape (n_sequences, hidden_dim).
    out: Dict[int, np.ndarray] = {}
    for layer, layer_matrix in embeddings_per_residue.items():
        if isinstance(layer_matrix, np.ndarray) and layer_matrix.ndim == 3:
            rows = [np.asarray(layer_matrix[i]) for i in range(layer_matrix.shape[0])]
        elif isinstance(layer_matrix, (list, tuple)):
            rows = [np.asarray(x) for x in layer_matrix]
        elif isinstance(layer_matrix, np.ndarray) and layer_matrix.dtype == object and layer_matrix.ndim == 1:
            rows = [np.asarray(x) for x in layer_matrix.tolist()]
        else:
            raise ValueError(
                "Expected per-residue embeddings as a dense 3D array or a ragged list of 2D arrays."
            )
        if not rows:
            out[int(layer)] = np.asarray([], dtype=np.float32)
            continue
        if method == "mean":
            out[int(layer)] = np.asarray([emb_arr.mean(axis=0) for emb_arr in rows])
            continue
        if method != "mut":
            raise ValueError("pool_method must be 'mean' or 'mut'.")
        if mutations is None or sep is None:
            raise ValueError("mut pooling requires mutations and sep.")
        if len(mutations) != len(rows):
            raise ValueError("mutations must have same length as number of sequences.")
        pooled_rows: List[np.ndarray] = []
        for emb_arr, mut in zip(rows, mutations):
            pos = _parse_mutation_positions(str(mut), str(sep))
            pos = [p for p in pos if 0 <= p < emb_arr.shape[0]]
            if not pos:
                raise ValueError(f"No valid mutated positions parsed for mutation '{mut}'.")
            pooled_rows.append(emb_arr[pos, :].mean(axis=0))
        out[int(layer)] = np.asarray(pooled_rows)
    return out


def save_llr_vect_and_heatmap(
    llr_all: np.ndarray,
    seq_base: str,
    *,
    vect_cache_path: Union[str, Path],
    map_png_cache_path: Union[str, Path],
) -> None:
    """Save flattened LLR vector CSV and heatmap PNG for a single base sequence."""
    # Section 1: prepare output paths and remove X-row before export.
    vect_path = Path(vect_cache_path)
    png_path = Path(map_png_cache_path)
    vect_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    llr_no_x = np.asarray(llr_all)[:-1, :]

    # Section 2: write flattened mutation-level LLR vector CSV.
    llr_flattened, mutations_all = flatten_2D_arr(llr_no_x, seq_base, MT_aa=aaList)
    llr_flattened_df = pd.DataFrame({"mutations": mutations_all, "LLR": llr_flattened})
    llr_flattened_df["is_WT"] = [(mut[0] == mut[-1]) * 1 for mut in mutations_all]
    llr_flattened_df = (
        llr_flattened_df
        .loc[llr_flattened_df["is_WT"] == 0, ["mutations", "LLR"]]
        .reset_index(drop=True)
        .round(4)
    )
    llr_flattened_df.to_csv(vect_path)

    # Section 3: write mutation heatmap PNG.
    plot_variant_heatmap(
        -llr_no_x,
        seq_base,
        N_res_per_heatmap_row=100,
        aa_list=aaList,
        savefig=png_path,
        figtitle="Predicted Effects of Mutations on Protein Sequence (LLR)",
    )


def resolve_llr_cache_paths(
    output_path: Union[str, Path],
    *,
    llr_cache_vect_filename_prefix: str = "",
) -> Dict[str, Union[Path, bool]]:
    """Resolve vect-cache and heatmap paths under `encodings/LLR` for a given output stem."""
    # Section 1: locate cache directory parallel to data subfolder under encodings.
    output_stem = Path(str(output_path))
    llr_cache_dir = None
    for parent in [output_stem.parent, *output_stem.parents]:
        if parent.name == "encodings":
            llr_cache_dir = parent / "LLR"
            break
    if llr_cache_dir is None:
        llr_cache_dir = output_stem.parent / "LLR"
    llr_cache_dir.mkdir(parents=True, exist_ok=True)

    # Section 2: derive vect and map-png cache paths.
    vect_prefix = str(llr_cache_vect_filename_prefix or "").strip()
    vect_filename = f"{vect_prefix}{output_stem.name}_vect.csv"
    vect_cache_path = llr_cache_dir / vect_filename
    map_base = vect_filename[:-4] if vect_filename.lower().endswith(".csv") else vect_filename
    if map_base.endswith("_vect"):
        map_base = map_base[:-5]
    map_png_cache_path = llr_cache_dir / f"{map_base}_map.png"
    vect_exists = bool(vect_cache_path.exists())
    status = "FOUND" if vect_exists else "NOT FOUND"
    print(f"[resolve_llr_cache_paths] vect cache {status}: {vect_cache_path}")
    return {
        "llr_cache_dir": llr_cache_dir,
        "vect_cache_path": vect_cache_path,
        "map_png_cache_path": map_png_cache_path,
        "vect_exists": vect_exists,
    }


def score_mutants_from_llr_map(
    llr_vect_csv_path: Union[str, Path],
    sequences_base: Sequence[str],
    sequences: Optional[Sequence[str]] = None,
) -> Optional[np.ndarray]:
    """
    Compute sequence-level zero-shot scores by summing single-mutation LLR values from a cached vect CSV.

    The CSV is expected to contain at least `mutations` (or `mutation`) and `LLR` columns.
    """
    # Section 1: no sequence list means no sequence-level scores to compute.
    if sequences is None:
        return None
    if len(sequences) != len(sequences_base):
        raise ValueError("sequences and sequences_base must have the same length.")

    # Section 2: load mutation->LLR lookup from cached vect CSV.
    vect_path = Path(llr_vect_csv_path)
    if not vect_path.exists():
        raise FileNotFoundError(f"LLR vect CSV not found: {vect_path}")
    df = pd.read_csv(vect_path)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    mut_col = cols_lower.get("mutations", cols_lower.get("mutation"))
    llr_col = cols_lower.get("llr")
    if mut_col is None or llr_col is None:
        raise ValueError(
            f"Expected columns ['mutations' or 'mutation', 'LLR'] in {vect_path}. "
            f"Found: {list(df.columns)}"
        )
    llr_lookup = {
        str(mut).strip(): float(score)
        for mut, score in zip(df[mut_col].tolist(), df[llr_col].tolist())
    }

    # Section 3: sum mutation LLRs per sequence by diffing against each base sequence.
    llr_sum_seq: List[float] = []
    for seq, seq_base in zip(sequences, sequences_base):
        if len(seq) != len(seq_base):
            raise ValueError("Each sequence must have same length as its sequence_base.")
        total = 0.0
        for pos_idx, (mut_aa, wt_aa) in enumerate(zip(seq, seq_base)):
            if mut_aa == wt_aa:
                continue
            mut_label = f"{wt_aa}{pos_idx + 1}{mut_aa}"
            if mut_label not in llr_lookup:
                raise KeyError(
                    f"Mutation '{mut_label}' not found in cached LLR map: {vect_path}"
                )
            total += float(llr_lookup[mut_label])
        llr_sum_seq.append(total)
    return np.asarray(llr_sum_seq, dtype=np.float32)


def _print_progress(progress_tag: str, done: int, total: int) -> None:
    """Render a minimal in-place progress bar."""
    # Section 1: compute bounded progress ratio.
    total_safe = max(int(total), 1)
    done_safe = min(max(int(done), 0), total_safe)
    ratio = done_safe / total_safe
    # Section 2: draw compact bar and flush in-place.
    bar_len = 20
    filled = int(round(ratio * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{progress_tag}] [{bar}] {done_safe}/{total_safe}", end="", flush=True)
    if done_safe >= total_safe:
        print("")


def _coerce_sequence_list(
    value: Optional[Union[str, Sequence[str]]],
    *,
    name: str,
    required: bool = False,
) -> Optional[List[str]]:
    """Normalize an optional sequence input into a list of strings."""
    # Section 1: enforce required inputs.
    if value is None:
        if required:
            raise ValueError(f"{name} requires non-empty sequences.")
        return None
    # Section 2: coerce str/list-like into list[str].
    if isinstance(value, str):
        out = [value]
    else:
        out = [str(x) for x in value]
    # Section 3: validate non-empty list when required.
    if required and not out:
        raise ValueError(f"{name} requires non-empty sequences.")
    return out
