from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[4]
    _src_root = _repo_root / "src"
    for _path in (str(_repo_root), str(_src_root)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

import argparse
import json
import re
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from abcode.core.paths import resolve_project_root
from project_config.variables import address_dict


def default_user_inputs() -> Dict[str, Any]:
    """Return editable defaults for CDR feature parsing."""
    return {
        "data_csv_path": "",
        "encodings_dir": "",
        "output_subdir": "cdr_features",
        "column_mappings": [
            {"sequence_vh": ["seq_cdrh1", "seq_cdrh2", "seq_cdrh3"]},
            {"sequence_vl": ["seq_cdrl1", "seq_cdrl2", "seq_cdrl3"]},
        ],
        "embedding_per_residue_paths": {},
        "pll_per_residue_paths": {},
        "on_match_error": "nan",  # "nan" or "raise"
    }


def _normalize_seq(value: Any) -> str:
    return "".join(str(value or "").upper().split())


def _infer_region_tag(parent_col: str) -> str:
    txt = str(parent_col or "").strip()
    if txt.startswith("sequence_"):
        return txt[len("sequence_") :]
    return txt


def _normalize_cdr_tag(cdr_col: str) -> str:
    txt = str(cdr_col or "").strip()
    if txt.startswith("seq_"):
        return txt[len("seq_") :]
    return txt


def _normalize_embedding_feature_stem(source_stem: str, region_tag: str) -> str:
    """
    Convert source stem like `vh_esm2-650m_per_residue-33` to `esm2-650m-33`.
    """
    s = str(source_stem or "").strip()
    prefix = f"{region_tag}_"
    if s.startswith(prefix):
        s = s[len(prefix) :]
    s = s.replace("_per_residue-", "-")
    s = s.replace("_per_residue", "")
    return s


def _embedding_output_stem(source_stem: str, region_tag: str) -> str:
    """
    Build embedding output stem with layer suffix at the end.

    Example:
    - input stem: `vh_esm2-650m_per_residue-33`
    - output stem: `esm2-650m_mean_pooled-33`
    """
    s = str(source_stem or "").strip()
    prefix = f"{region_tag}_"
    if s.startswith(prefix):
        s = s[len(prefix) :]

    m = re.match(r"^(?P<base>.+)_per_residue-(?P<layer>\d+)$", s)
    if m:
        return f"{m.group('base')}_mean_pooled-{m.group('layer')}"

    # Fallback for unexpected stems.
    normalized = _normalize_embedding_feature_stem(source_stem, region_tag)
    return f"{normalized}_mean_pooled"


def _normalize_pll_feature_stem(source_stem: str, region_tag: str) -> str:
    """
    Convert source stem like `vh_esm2-650m_PLL-wt_per_residue` to `esm2-650m_PLL-wt`.
    """
    s = str(source_stem or "").strip()
    prefix = f"{region_tag}_"
    if s.startswith(prefix):
        s = s[len(prefix) :]
    s = s.replace("_per_residue", "")
    return s


def _validate_column_mappings(column_mappings: Any) -> List[Dict[str, List[str]]]:
    """Validate and normalize list-of-dicts sequence/CDR mappings."""
    if not isinstance(column_mappings, list):
        raise TypeError("column_mappings must be a list of dictionaries.")
    out: List[Dict[str, List[str]]] = []
    for idx, item in enumerate(column_mappings):
        if not isinstance(item, dict):
            raise TypeError(f"column_mappings[{idx}] must be a dict.")
        normalized_item: Dict[str, List[str]] = {}
        for parent_col, cdr_cols in item.items():
            if not isinstance(cdr_cols, (list, tuple)):
                raise TypeError(f"column_mappings[{idx}]['{parent_col}'] must be a list.")
            normalized_item[str(parent_col)] = [str(x) for x in cdr_cols]
        out.append(normalized_item)
    return out


def _find_subseq_span(parent_seq: str, subseq: str) -> Optional[Tuple[int, int]]:
    p = _normalize_seq(parent_seq)
    s = _normalize_seq(subseq)
    if not p or not s:
        return None
    start = p.find(s)
    if start < 0:
        return None
    return start, start + len(s)


def _build_spans(
    df: pd.DataFrame,
    parent_col: str,
    cdr_col: str,
    *,
    on_match_error: str = "nan",
) -> List[Optional[Tuple[int, int]]]:
    spans: List[Optional[Tuple[int, int]]] = []
    for row_idx, (parent_seq, cdr_seq) in enumerate(zip(df[parent_col].tolist(), df[cdr_col].tolist())):
        span = _find_subseq_span(parent_seq, cdr_seq)
        if span is None and on_match_error == "raise":
            raise ValueError(
                f"Could not locate {cdr_col} subsequence in {parent_col} at row {row_idx}."
            )
        spans.append(span)
    return spans


def _discover_embedding_paths(encodings_dir: Path, region_tag: str) -> List[Path]:
    out = []
    out.extend(sorted(encodings_dir.glob(f"{region_tag}_*_per_residue-*.npy")))
    out.extend(sorted(encodings_dir.glob(f"{region_tag}_*_per_residue-*.npz")))
    return [p for p in out if p.is_file()]


def _discover_pll_paths(encodings_dir: Path, region_tag: str) -> List[Path]:
    out = sorted(encodings_dir.glob(f"{region_tag}_*_PLL*_per_residue.csv"))
    return [p for p in out if p.is_file()]


def _align_rows(rows: List[np.ndarray], n_rows: int) -> List[Optional[np.ndarray]]:
    aligned: List[Optional[np.ndarray]] = [None] * n_rows
    for i in range(min(n_rows, len(rows))):
        aligned[i] = rows[i]
    return aligned


def _load_embedding_rows(path: Path, n_rows: int) -> List[Optional[np.ndarray]]:
    """Load row-aligned per-residue embedding arrays from .npy/.npz."""
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path), allow_pickle=False)
        if arr.ndim == 3:
            rows = [np.asarray(arr[i]) for i in range(arr.shape[0])]
            return _align_rows(rows, n_rows)
        if arr.dtype == object and arr.ndim == 1:
            rows = [np.asarray(x) for x in arr.tolist()]
            return _align_rows(rows, n_rows)
        raise ValueError(f"Unsupported npy shape for per-residue embeddings: {path} shape={arr.shape}")

    if path.suffix.lower() == ".npz":
        # Prefer ragged keyed format (keys are row indices).
        with np.load(str(path), allow_pickle=False) as npz:
            keys = sorted(npz.files, key=lambda x: int(str(x)))
            rows = [np.asarray(npz[k]) for k in keys]
        return _align_rows(rows, n_rows)

    raise ValueError(f"Unsupported embedding format for per-residue file: {path}")


def _parse_pos_col(col: Any) -> Optional[int]:
    if isinstance(col, (int, np.integer)):
        return int(col)
    c = str(col)
    if c.isdigit():
        return int(c)
    if c.startswith("pll_pos_"):
        tail = c[len("pll_pos_") :]
        if tail.isdigit():
            return int(tail) - 1
    return None


def _load_pll_matrix(path: Path, n_rows: int) -> np.ndarray:
    """Load per-residue PLL table into dense row x position matrix (NaN padded)."""
    df = pd.read_csv(path)
    pos_cols = []
    for col in df.columns:
        pos = _parse_pos_col(col)
        if pos is not None:
            pos_cols.append((pos, col))
    if not pos_cols:
        raise ValueError(f"No positional PLL columns found in {path}")
    pos_cols = sorted(pos_cols, key=lambda x: x[0])
    max_pos = pos_cols[-1][0]

    mat = np.full((n_rows, max_pos + 1), np.nan, dtype=np.float32)
    copy_rows = min(n_rows, len(df))
    for pos, col in pos_cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)[:copy_rows]
        mat[:copy_rows, pos] = vals
    return mat


def _infer_embedding_dim(rows: Sequence[Optional[np.ndarray]]) -> int:
    for row in rows:
        if row is None:
            continue
        arr = np.asarray(row)
        if arr.ndim == 2 and arr.shape[1] > 0:
            return int(arr.shape[1])
    return 0


def _subseq_mean_pooled_embeddings(
    rows: Sequence[Optional[np.ndarray]],
    spans: Sequence[Optional[Tuple[int, int]]],
) -> np.ndarray:
    dim = _infer_embedding_dim(rows)
    out = np.full((len(spans), dim), np.nan, dtype=np.float32)
    if dim == 0:
        return out
    for i, span in enumerate(spans):
        if span is None or rows[i] is None:
            continue
        start, end = span
        arr = np.asarray(rows[i], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != dim or start < 0 or end > arr.shape[0] or start >= end:
            continue
        out[i] = np.nanmean(arr[start:end, :], axis=0)
    return out


def _subseq_mean_pooled_pll(
    pll_matrix: np.ndarray,
    spans: Sequence[Optional[Tuple[int, int]]],
) -> np.ndarray:
    out = np.full((len(spans),), np.nan, dtype=np.float32)
    for i, span in enumerate(spans):
        if span is None:
            continue
        start, end = span
        if start < 0 or end > pll_matrix.shape[1] or start >= end:
            continue
        vals = pll_matrix[i, start:end]
        if np.isfinite(vals).any():
            out[i] = float(np.nanmean(vals))
    return out


def _save_embedding_mean_pooled(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), values)
    return str(path)


def _save_pll_mean_pooled(
    npy_path: Path,
    csv_path: Path,
    df: pd.DataFrame,
    cdr_col: str,
    values: np.ndarray,
) -> Dict[str, str]:
    """Save pooled PLL vector as .npy and .csv."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), values)
    out_df = pd.DataFrame(
        {
            "row_index": np.arange(len(values), dtype=int),
            cdr_col: df[cdr_col].tolist(),
            "PLL_mean_pooled": values,
        }
    )
    out_df.to_csv(csv_path, index=False)
    return {"npy": str(npy_path), "csv": str(csv_path)}


def parse_cdr_features(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and pool CDR features from parent variable-region per-residue features.

    For each parent sequence column and selected CDR subsequence column:
    - find row-wise [start:end) spans in parent sequence
    - slice per-residue embeddings / PLL vectors
    - save pooled CDR outputs
    """
    payload = dict(default_user_inputs())
    payload.update(dict(inputs or {}))

    data_csv_path = Path(str(payload.get("data_csv_path", "")).strip()).expanduser().resolve()
    if not data_csv_path.exists():
        raise FileNotFoundError(f"data_csv_path not found: {data_csv_path}")

    encodings_dir = Path(str(payload.get("encodings_dir", "")).strip()).expanduser().resolve()
    if not encodings_dir.exists():
        raise FileNotFoundError(f"encodings_dir not found: {encodings_dir}")

    out_dir = (encodings_dir / str(payload.get("output_subdir", "cdr_features")).strip()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    on_match_error = str(payload.get("on_match_error", "nan")).strip().lower()
    if on_match_error not in {"nan", "raise"}:
        raise ValueError("on_match_error must be 'nan' or 'raise'.")

    column_mappings = _validate_column_mappings(payload.get("column_mappings"))
    df = pd.read_csv(data_csv_path)
    n_rows = len(df)

    explicit_emb = payload.get("embedding_per_residue_paths", {}) or {}
    explicit_pll = payload.get("pll_per_residue_paths", {}) or {}

    outputs: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for mapping in column_mappings:
        for parent_col, cdr_cols in mapping.items():
            if parent_col not in df.columns:
                raise ValueError(f"Parent sequence column not found in CSV: {parent_col}")
            region_tag = _infer_region_tag(parent_col)

            emb_paths = [Path(p).expanduser().resolve() for p in explicit_emb.get(parent_col, [])]
            pll_paths = [Path(p).expanduser().resolve() for p in explicit_pll.get(parent_col, [])]
            if not emb_paths:
                emb_paths = _discover_embedding_paths(encodings_dir, region_tag)
            if not pll_paths:
                pll_paths = _discover_pll_paths(encodings_dir, region_tag)

            for cdr_col in cdr_cols:
                if cdr_col not in df.columns:
                    raise ValueError(f"CDR column not found in CSV: {cdr_col}")
                cdr_tag = _normalize_cdr_tag(cdr_col)
                spans = _build_spans(df, parent_col, cdr_col, on_match_error=on_match_error)
                n_unmatched = sum(1 for x in spans if x is None)
                if n_unmatched:
                    warnings.append(
                        f"{parent_col}->{cdr_col}: {n_unmatched}/{n_rows} rows had no valid subsequence span."
                    )

                # Section 1: embedding per-residue -> CDR mean pooled embedding (.npy).
                for emb_path in emb_paths:
                    feature_stem = _embedding_output_stem(emb_path.stem, region_tag)
                    rows = _load_embedding_rows(emb_path, n_rows=n_rows)
                    pooled = _subseq_mean_pooled_embeddings(rows, spans)
                    out_name = f"{cdr_tag}_{feature_stem}.npy"
                    out_path = out_dir / out_name
                    saved = _save_embedding_mean_pooled(out_path, pooled)
                    outputs.append(
                        {
                            "type": "embedding_mean_pooled",
                            "parent_col": parent_col,
                            "cdr_col": cdr_col,
                            "source_path": str(emb_path),
                            "output_npy": saved,
                            "shape": list(pooled.shape),
                        }
                    )

                # Section 2: PLL per-residue -> CDR PLL_mean_pooled (.npy + .csv).
                for pll_path in pll_paths:
                    feature_stem = _normalize_pll_feature_stem(pll_path.stem, region_tag)
                    pll_matrix = _load_pll_matrix(pll_path, n_rows=n_rows)
                    pooled = _subseq_mean_pooled_pll(pll_matrix, spans)
                    out_npy = out_dir / f"{cdr_tag}_{feature_stem}.npy"
                    out_csv = out_dir / f"{cdr_tag}_{feature_stem}.csv"
                    saved = _save_pll_mean_pooled(out_npy, out_csv, df, cdr_col, pooled)
                    outputs.append(
                        {
                            "type": "pll_mean_pooled",
                            "parent_col": parent_col,
                            "cdr_col": cdr_col,
                            "source_path": str(pll_path),
                            "output_npy": saved["npy"],
                            "output_csv": saved["csv"],
                            "shape": list(pooled.shape),
                        }
                    )

    return {
        "status": "ok",
        "data_csv_path": str(data_csv_path),
        "encodings_dir": str(encodings_dir),
        "output_dir": str(out_dir),
        "n_rows": n_rows,
        "outputs": outputs,
        "warnings": warnings,
    }


def _load_inputs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.inputs_json:
        loaded = json.loads(args.inputs_json)
        if not isinstance(loaded, dict):
            raise ValueError("--inputs-json must decode to a JSON object.")
        return loaded
    if args.inputs_file:
        p = Path(args.inputs_file).expanduser().resolve()
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--inputs-file must contain a JSON object.")
        return loaded
    return default_user_inputs()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CDR subsequence features from variable-region encodings.")
    parser.add_argument("--inputs-json", type=str, default="", help="Inline JSON string of inputs.")
    parser.add_argument("--inputs-file", type=str, default="", help="Path to JSON file of inputs.")
    args = parser.parse_args()

    user_inputs = _load_inputs_from_args(args)
    result = parse_cdr_features(user_inputs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import sys

    # Auto-mode: CLI when args are passed, IDE when run with no args.
    RUN_MODE = "cli" if len(sys.argv) > 1 else "ide"

    if RUN_MODE == "ide":
        # Section 1: editable defaults for local IDE execution.
        user_inputs = default_user_inputs()

        # Section 1A: configure output root and subfolders (same style as notebook).
        user_inputs["root_key"] = "biostream-developability-data"
        user_inputs["data_subfolder"] = "opensource"
        user_inputs["encodings_subfolder"] = "encodings/"
        filename = f'{user_inputs["data_subfolder"]}.csv'

        # Section 1B: configure sequence input source from root_key + subfolders.
        repo_root = resolve_project_root()
        data_root = (repo_root / Path(address_dict[user_inputs["root_key"]])).resolve()
        user_inputs["data_csv_path"] = str(
            data_root / "expdata" / user_inputs["data_subfolder"] / filename
        )
        user_inputs["encodings_dir"] = str(
            (data_root / user_inputs["encodings_subfolder"] / user_inputs["data_subfolder"]).resolve()
        )
        user_inputs["output_subdir"] = "cdr_features"

        # Section 1C: set sequence->CDR column mappings.
        # user_inputs["column_mappings"] = [
        #     {"sequence_vh": ["seq_cdrh1", "seq_cdrh2", "seq_cdrh3"]},
        #     {"sequence_vl": ["seq_cdrl1", "seq_cdrl2", "seq_cdrl3"]},
        # ]
        user_inputs["column_mappings"] = [
            {"sequence_vh": ["seq_cdrh3"]},
        ]

        # Section 1D: optional explicit feature-path overrides.
        # If left empty, files are auto-discovered from encodings_dir.
        user_inputs["embedding_per_residue_paths"] = {}
        user_inputs["pll_per_residue_paths"] = {}

        # Section 1E: behavior on CDR/parent match failures.
        user_inputs["on_match_error"] = "nan"  # "nan" or "raise"

        # Section 2: execute and print concise result.
        if not str(user_inputs["data_csv_path"]).strip() or not str(user_inputs["encodings_dir"]).strip():
            raise ValueError(
                "Set user_inputs['data_csv_path'] and user_inputs['encodings_dir'] in the __main__ IDE block."
            )
        result = parse_cdr_features(user_inputs)
        pprint(
            {
                "status": result.get("status"),
                "output_dir": result.get("output_dir"),
                "n_rows": result.get("n_rows"),
                "n_outputs": len(result.get("outputs", [])),
                "warnings": result.get("warnings", []),
            }
        )
    else:
        main()
