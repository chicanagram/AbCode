from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForMaskedLM, AutoTokenizer

from abcode.tools.encodings.common import (
    _resolve_model_name,
    _resolve_layers,
    _coerce_sequence_list,
    _print_progress,
    compute_pooled_embeddings,
    extract_sequence_token_spans,
    resolve_llr_cache_paths,
    save_layerwise_embeddings,
    save_llr_vect_and_heatmap,
    score_mutants_from_llr_map,
)
from abcode.tools.utils.model_utils import get_device, iter_batches
from project_config.variables import aaList_with_X

# Common shorthand names mapped to Hugging Face model ids.
MODEL_NAME_ALIASES: Dict[str, str] = {
    "esm2-8m": "facebook/esm2_t6_8M_UR50D",
    "esm2-35m": "facebook/esm2_t12_35M_UR50D",
    "esm2-150m": "facebook/esm2_t30_150M_UR50D",
    "esm2-650m": "facebook/esm2_t33_650M_UR50D",
    "esm2-3b": "facebook/esm2_t36_3B_UR50D",
    "esm2-15b": "facebook/esm2_t48_15B_UR50D",
    "esm1v-1": "facebook/esm1v_t33_650M_UR90S_1",
    "esm1v-2": "facebook/esm1v_t33_650M_UR90S_2",
    "esm1v-3": "facebook/esm1v_t33_650M_UR90S_3",
    "esm1v-4": "facebook/esm1v_t33_650M_UR90S_4",
    "esm1v-5": "facebook/esm1v_t33_650M_UR90S_5",
}
DEFAULT_MODEL_NAME = "esm2-650M"
_MODEL_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
DEFAULT_MARGINAL_TYPE = "wt"


def _prepare_tokenizer_inputs(sequences: Sequence[str]) -> List[str]:
    """Convert sequences into the spaced token format expected by ESM tokenizers."""
    return [" ".join(seq) for seq in sequences]


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and cache an ESM masked-LM model plus tokenizer.

    Args:
        model_name: ESM alias (for example `esm2_650m`, `esm1v_1`) or raw HF id.
        device: Optional device override. Defaults to `get_device()`.

    Returns:
        Dict containing tokenizer, model, resolved model name, and device.
    """
    resolved_name = _resolve_model_name(model_name, MODEL_NAME_ALIASES, DEFAULT_MODEL_NAME)
    resolved_device = str(device or get_device())
    cache_key = (resolved_name, resolved_device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(resolved_name)
    model = AutoModelForMaskedLM.from_pretrained(resolved_name)
    model.eval()
    model.to(resolved_device)

    bundle = {
        "model_name": resolved_name,
        "device": resolved_device,
        "tokenizer": tokenizer,
        "model": model,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def tokenize_sequences(
    sequences: Union[str, Sequence[str]],
    *,
    tokenizer: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tokenize one or more protein sequences for ESM-family models.

    Args:
        sequences: Single sequence or iterable of sequences.
        tokenizer: Optional preloaded tokenizer.
        model_name: ESM alias or raw HF model id when tokenizer is not supplied.
        device: Optional device override.
        max_length: Optional truncation length including special tokens.

    Returns:
        Dict with normalized sequences plus the tokenized batch tensors.
    """
    bundle = None
    tok = tokenizer
    if tok is None:
        bundle = load_model(model_name=model_name, device=device)
        tok = bundle["tokenizer"]
    resolved_device = str(device or (bundle["device"] if bundle else get_device()))

    tokenizer_inputs = _prepare_tokenizer_inputs(sequences)
    batch = tok(
        tokenizer_inputs,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        add_special_tokens=True,
    )
    batch = {k: v.to(resolved_device) for k, v in batch.items()}
    return {"sequences": sequences, "tokenizer_inputs": tokenizer_inputs, "batch": batch}


def forward_pass(
    seq_tokens: Dict[str, Any],
    *,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    output_hidden_states: bool = True,
) -> Any:
    """
    Run an ESM forward pass on a tokenized batch.

    Args:
        seq_tokens: Output of `tokenize_sequences(...)`.
        model: Optional preloaded model.
        model_name: ESM alias or raw HF id when model is not supplied.
        device: Optional device override.
        output_hidden_states: Whether to request hidden states.

    Returns:
        Hugging Face model output object.
    """
    mdl = model
    if mdl is None:
        mdl = load_model(model_name=model_name, device=device)["model"]
    with torch.no_grad():
        return mdl(**seq_tokens["batch"], output_hidden_states=output_hidden_states)


def get_LLR_scores(
    sequences_base: Optional[Union[str, Sequence[str]]] = None,
    sequences: Optional[Union[str, Sequence[str]]] = None,
    mutations: Optional[Union[str, Sequence[str]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    llr_cache_vect_filename_prefix: str = "",
    resave_llr_cache_if_found: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    **kwargs,
):
    """Compute sequence-level LLR scores using WT or masked marginal scoring."""
    # get kwargs
    marginal_type = kwargs.get('marginal_type', DEFAULT_MARGINAL_TYPE)
    batch_size = kwargs.get('batch_size', 4)
    device = kwargs.get('device', 'cpu')

    # Section 1: normalize sequence inputs and scoring backend.
    base_list = _coerce_sequence_list(sequences_base, name="get_LLR_scores(sequences_base)", required=True)
    seq_list = _coerce_sequence_list(sequences, name="get_LLR_scores(sequences)", required=False)
    unique_bases = list(dict.fromkeys(base_list))
    if len(unique_bases) != 1:
        raise ValueError(
            "LLR cache/vector scoring currently supports exactly one unique sequence_base."
        )
    if marginal_type == "wt":
        map_fn = _compute_wt_marginal_llr_map
    elif marginal_type == "masked":
        map_fn = _compute_masked_marginal_llr_map
    else:
        raise ValueError("marginal_type must be either 'wt' or 'masked'.")

    # Section 2: resolve output/cache paths.
    output_path = str(output_path) if output_path is not None else f"{model_name}_LLR"
    output_path = output_path.replace('LLR', f'LLR-{marginal_type}')
    output_stem = Path(str(output_path))
    cache_stem = output_stem.with_name(f"{str(model_name)}_LLR-{marginal_type}")
    cache_info = resolve_llr_cache_paths(
        cache_stem,
        llr_cache_vect_filename_prefix=llr_cache_vect_filename_prefix,
    )
    vect_cache_path = Path(cache_info["vect_cache_path"])
    map_png_cache_path = Path(cache_info["map_png_cache_path"])

    # Cache resolution for vect CSV.
    cache_hit = bool(cache_info["vect_exists"])
    if cache_hit:
        print(f"[zeroshot/{marginal_type}-marginal] Loaded LLR cache: {vect_cache_path}")

    # Section 3: compute and persist full marginal map only on cache miss (or explicit refresh).
    if (not cache_hit) or bool(resave_llr_cache_if_found):
        seq_base = unique_bases[0]
        map_fn(
            [seq_base],
            model_name,
            batch_size,
            device,
            save_artifacts=True,
            vect_cache_path=vect_cache_path,
            map_png_cache_path=map_png_cache_path,
        )
        action = "Refreshed" if cache_hit else "Saved"
        print(f"[zeroshot/{marginal_type}-marginal] {action} LLR cache: {vect_cache_path}")

    # Section 4: score provided sequences by summing cached mutation LLRs.
    LLRsum_seq = score_mutants_from_llr_map(
        llr_vect_csv_path=vect_cache_path,
        sequences_base=base_list,
        sequences=seq_list,
    )

    # Section 5: save LLR for provided sequences.
    if LLRsum_seq is not None:
        np.save(f'{output_path}.npy', np.array(LLRsum_seq).astype(np.float32))
        pd.DataFrame({'mutations': mutations, 'LLR': LLRsum_seq}).round(4).to_csv(f'{output_path}.csv')
        print(f"[zeroshot/{marginal_type}-marginal] Saved raw scores: {output_path}.csv")
    return LLRsum_seq


def get_mean_PLL_scores(
    sequences: Optional[Union[str, Sequence[str]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    **kwargs,
) -> np.ndarray:
    """Compute mean pseudo-log-likelihood (mean PLL) scores for sequences without a shared base."""
    # get kwargs
    marginal_type = kwargs.get('marginal_type', DEFAULT_MARGINAL_TYPE)
    batch_size = kwargs.get('batch_size', 4)
    device = kwargs.get('device', 'cpu')

    # Section 1: validate/normalize sequence input and select marginal backend.
    seq_list = _coerce_sequence_list(sequences, name="get_mean_PLL_scores", required=True)

    if marginal_type == "wt":
        map_fn = _compute_wt_marginal_llr_map
    elif marginal_type == "masked":
        map_fn = _compute_masked_marginal_llr_map
    else:
        raise ValueError("marginal_type must be either 'wt' or 'masked'.")

    # Section 2: resolve output naming.
    output_str: Optional[str] = None
    if output_path is not None:
        output_str = str(output_path)
        if "LLR" in output_str:
            output_str = output_str.replace("LLR", f"PLL-{marginal_type}")
        elif "PLL-" not in output_str and output_str.endswith("PLL"):
            output_str = f"{output_str}-{marginal_type}"
        elif "PLL" not in output_str:
            output_str = f"{output_str}_PLL-{marginal_type}"

    # Section 3: compute per-position WT log-probabilities.
    _, wt_logp_by_base = map_fn(
        seq_list,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        return_wt_logp=True,
    )

    mean_pll_scores = np.asarray(
        [float(np.mean(wt_logp_by_base[seq])) for seq in seq_list],
        dtype=np.float32,
    )

    # Section 4: save outputs to .npy/.csv.
    if output_str is not None:
        np.save(f"{output_str}.npy", mean_pll_scores)
        pd.DataFrame({"sequence": seq_list, "PLL": mean_pll_scores}).round(6).to_csv(
            f"{output_str}.csv",
            index=False,
        )
        print(f"[PLL/{marginal_type}-marginal] Saved scores: {output_str}.csv")

        # Save per-residue PLL scores with NaN padding to the longest sequence.
        per_row = [np.asarray(wt_logp_by_base[seq], dtype=np.float32).reshape(-1) for seq in seq_list]
        max_len = max((arr.shape[0] for arr in per_row), default=0)
        padded = np.full((len(seq_list), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(per_row):
            padded[i, : arr.shape[0]] = arr
        pos_cols = [i for i in range(max_len)]
        pll_per_residue = pd.DataFrame(padded, columns=pos_cols)
        pll_per_residue.insert(0, "sequence", seq_list)
        pll_per_residue.round(6).reset_index(drop=False).to_csv(f"{output_str}_per_residue.csv", index=False)
        print(f"[PLL_per_residue/{marginal_type}-marginal] Saved scores: {output_str}_per_residue.csv")

    return mean_pll_scores


def _compute_wt_marginal_llr_map(
    sequences_base: Sequence[str],
    model_name: Optional[str] = DEFAULT_MODEL_NAME,
    batch_size: Optional[int] = 4,
    device: Optional[str] = "cpu",
    return_wt_logp: bool = False,
    save_artifacts: bool = False,
    vect_cache_path: Optional[Union[str, Path]] = None,
    map_png_cache_path: Optional[Union[str, Path]] = None,
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """
    Compute WT-marginal LLR matrices from one WT forward pass per unique base sequence.

    Returns:
      - llr_by_base: dict {base_seq: matrix shape (len(aaList_with_X), len(base_seq))}
    """
    bundle = load_model(model_name=model_name, device=device)
    llr_by_base: Dict[str, np.ndarray] = {}
    wt_logp_by_base: Dict[str, np.ndarray] = {}

    # Section 1: one forward pass per unique WT base sequence.
    unique_bases = list(dict.fromkeys(sequences_base))
    for i, seq_base in enumerate(unique_bases):
        print(i, end=' ')
        tokenized = tokenize_sequences(
            [seq_base],
            tokenizer=bundle["tokenizer"],
            device=bundle["device"],
        )
        outputs = forward_pass(
            tokenized,
            model=bundle["model"],
            output_hidden_states=False,
        )
        # logits: [1, T, V]
        logits = outputs.logits[0]  # [T, V]
        probs = softmax(logits, axis=-1)

        n = len(seq_base)
        llr_mat = np.zeros((len(aaList_with_X), n), dtype=np.float32)
        wt_logp_vec = np.zeros((n,), dtype=np.float32)

        for pos_idx, wt_aa in enumerate(seq_base):
            token_pos = pos_idx + 1  # ESM BOS offset
            wt_id = bundle["tokenizer"].convert_tokens_to_ids(wt_aa)
            wt_probs = probs[token_pos, wt_id]
            wt_logp_vec[pos_idx] = np.log(wt_probs)

            for i, aa in enumerate(aaList_with_X):
                aa_id = bundle["tokenizer"].convert_tokens_to_ids(aa)
                aa_probs = probs[token_pos, aa_id]
                llr_mat[i, pos_idx] = np.round(np.log(aa_probs) - np.log(wt_probs), 4)

        llr_by_base[seq_base] = llr_mat
        wt_logp_by_base[seq_base] = wt_logp_vec
        if save_artifacts:
            if vect_cache_path is None or map_png_cache_path is None:
                raise ValueError("save_artifacts=True requires vect_cache_path and map_png_cache_path.")
            save_llr_vect_and_heatmap(
                llr_mat,
                seq_base,
                vect_cache_path=vect_cache_path,
                map_png_cache_path=map_png_cache_path,
            )

    if return_wt_logp:
        return llr_by_base, wt_logp_by_base
    return llr_by_base


def _compute_masked_marginal_llr_map(
    sequences_base: Sequence[str],
    model_name: Optional[str] = DEFAULT_MODEL_NAME,
    batch_size: Optional[int] = 4,  # kept for interface parity
    device: Optional[str] = "cpu",
    return_wt_logp: bool = False,
    save_artifacts: bool = False,
    vect_cache_path: Optional[Union[str, Path]] = None,
    map_png_cache_path: Optional[Union[str, Path]] = None,
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """
    Compute masked-marginal LLR matrices from one masked sweep per unique base sequence.

    Returns:
      - llr_by_base: dict {base_seq: matrix shape (len(aaList_with_X), len(base_seq))}
    """
    bundle = load_model(model_name=model_name, device=device)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    mask_id = getattr(tokenizer, "mask_token_id", None)

    # Section 1: one masked sweep per unique WT base sequence.
    llr_by_base: Dict[str, np.ndarray] = {}
    wt_logp_by_base: Dict[str, np.ndarray] = {}
    unique_bases = list(dict.fromkeys(sequences_base))
    masked_batch_size = max(int(batch_size or 4), 1)
    for i, seq_base in enumerate(unique_bases):
        print(i, end=' ')
        n = len(seq_base)

        # Tokenize WT sequence
        inp = tokenizer(" ".join(seq_base), return_tensors="pt")
        input_ids = inp["input_ids"][0].to(bundle["device"])       # [T]
        attn = inp["attention_mask"][0].to(bundle["device"])       # [T]

        # ESM residue token positions (BOS at 0)
        res_pos = torch.arange(1, 1 + n, device=input_ids.device)  # [n]
        wt_ids = input_ids[res_pos]                                # [n]

        # Build masked variants in chunks to control memory and improve throughput.
        pos_token_llr_chunks: List[torch.Tensor] = []
        wt_logp_chunks: List[torch.Tensor] = []
        start_time = time.time()
        print(
            f"[masked-marginal] Processing base sequence (len={n}) "
            f"with masked_batch_size={masked_batch_size}..."
        )
        for start in range(0, n, masked_batch_size):
            end = min(start + masked_batch_size, n)
            chunk_len = end - start

            chunk_masked_ids = input_ids.unsqueeze(0).repeat(chunk_len, 1)   # [chunk, T]
            chunk_masked_attn = attn.unsqueeze(0).repeat(chunk_len, 1)       # [chunk, T]
            chunk_rows = torch.arange(chunk_len, device=input_ids.device)
            chunk_res_pos = res_pos[start:end]
            chunk_wt_ids = wt_ids[start:end]
            chunk_masked_ids[chunk_rows, chunk_res_pos] = mask_id

            with torch.inference_mode():
                out = model(input_ids=chunk_masked_ids, attention_mask=chunk_masked_attn)
            log_probs = torch.log_softmax(out.logits, dim=-1)                # [chunk, T, V]

            # Position-specific WT log-probability and full residue-position table for chunk.
            chunk_wt_logp = log_probs[chunk_rows, chunk_res_pos, chunk_wt_ids]           # [chunk]
            chunk_pos_token_llr = log_probs[chunk_rows, chunk_res_pos, :] - chunk_wt_logp[:, None]  # [chunk, V]
            pos_token_llr_chunks.append(chunk_pos_token_llr)
            wt_logp_chunks.append(chunk_wt_logp)

            done = end
            elapsed = max(time.time() - start_time, 1e-9)
            rate = done / elapsed
            eta = (n - done) / rate if rate > 0 else float("inf")
            print(
                f"[masked-marginal] {done}/{n} ({100.0 * done / n:.1f}%) "
                f"| {rate:.2f} pos/s | ETA {eta:.1f}s"
            )

        pos_token_llr = torch.cat(pos_token_llr_chunks, dim=0)  # [n, V]
        wt_logp_vec = torch.cat(wt_logp_chunks, dim=0).detach().cpu().numpy().astype(np.float32)

        # Convert to shape [aa, pos]
        llr_mat = np.zeros((len(aaList_with_X), n), dtype=np.float32)
        for i, aa in enumerate(aaList_with_X):
            aa_id = tokenizer.convert_tokens_to_ids(aa)
            llr_mat[i, :] = pos_token_llr[:, aa_id].detach().cpu().numpy()

        llr_by_base[seq_base] = llr_mat
        wt_logp_by_base[seq_base] = wt_logp_vec
        if save_artifacts:
            if vect_cache_path is None or map_png_cache_path is None:
                raise ValueError("save_artifacts=True requires vect_cache_path and map_png_cache_path.")
            save_llr_vect_and_heatmap(
                llr_mat,
                seq_base,
                vect_cache_path=vect_cache_path,
                map_png_cache_path=map_png_cache_path,
            )

    if return_wt_logp:
        return llr_by_base, wt_logp_by_base
    return llr_by_base


def get_embeddings(
    sequences: Union[str, Sequence[str]],
    sequences_base: Optional[Union[str, Sequence[str]]] = None,
    save_per_residue_embeddings: bool = False,
    get_embeddings_for_seq_base: bool = False,
    pool_method: Optional[str] = "mean",
    mutations: Optional[Sequence[str]] = None,
    sep: Optional[str] = '+',
    output_path: Optional[Union[str, Path]] = None,
    layers: Optional[Sequence[int]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    **kwargs,
):
    """
    Compute and save layer-wise embeddings for sequences (and optionally WT/base).

    Per-residue embeddings are saved as `*.npy` arrays with suffix `-{layer_number}`.
    If `pool_method` is provided, pooled embeddings are also saved layer-wise.
    """
    # get kwargs
    batch_size = kwargs.get('batch_size', 4)
    device = kwargs.get('device', 'cpu')

    # Section 1: prepare output stem and compute per-residue embeddings.
    out_base = Path(output_path) if output_path else Path("embeddings")
    result: Dict[str, Any] = {}
    per_residue = _compute_per_residue_embeddings(
        sequences,
        model_name=model_name,
        layers=layers,
        batch_size=batch_size,
        device=device,
        progress_tag="esm2 embeddings",
    )

    # Section 2: optionally persist per-residue embeddings for input sequences.
    if save_per_residue_embeddings:
        result["per_residue_paths"] = save_layerwise_embeddings(
            per_residue,
            output_stem=out_base,
            suffix="per_residue",
            log_tag="get_embeddings",
        )

    # Section 3: optionally pool embeddings and persist pooled matrices.
    if pool_method is not None:
        pooled = compute_pooled_embeddings(
            per_residue,
            pool_method=str(pool_method),
            mutations=mutations,
            sep=sep,
        )
        result["pooled_paths"] = save_layerwise_embeddings(
            pooled,
            output_stem=out_base,
            suffix=f"{str(pool_method)}_pooled",
            log_tag="get_embeddings",
        )

    # Section 4: optionally compute/save separate embeddings for base sequences.
    if get_embeddings_for_seq_base:
        base_list = sequences_base if sequences_base is not None else []
        if not base_list:
            raise ValueError("get_embeddings_for_seq_base=True requires non-empty sequences_base.")
        base_per_residue = _compute_per_residue_embeddings(
            base_list,
            model_name=model_name,
            layers=layers,
            batch_size=batch_size,
            device=device,
            progress_tag="esm2 base embeddings",
        )
        if save_per_residue_embeddings:
            result["base_per_residue_paths"] = save_layerwise_embeddings(
                base_per_residue,
                output_stem=out_base,
                suffix="per_residue",
                file_suffix="_base",
                log_tag="get_embeddings",
            )
        if pool_method is not None:
            base_pooled = compute_pooled_embeddings(
                base_per_residue,
                pool_method="mean",
                mutations=None,
                sep=sep,
            )
            result["base_pooled_paths"] = save_layerwise_embeddings(
                base_pooled,
                output_stem=out_base,
                suffix=f"{str(pool_method)}_pooled",
                file_suffix="_base",
                log_tag="get_embeddings",
            )

    return result


def _compute_per_residue_embeddings(
    sequences: Union[str, Sequence[str]],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    layers: Optional[Sequence[int]] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    progress_tag: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract per-residue hidden-state embeddings for one or more ESM layers.

    Args:
        sequences: Single sequence or iterable of sequences.
        model_name: ESM alias or raw HF model id.
        layers: Hidden-state layers to return. Defaults to the final layer.
        batch_size: Batch size for inference.
        device: Optional device override.

    Returns:
        Dict `{layer_index: embedding_payload}` where each value is either:
        - dense array `(n_sequences, seq_len, hidden_dim)` when all lengths match
        - ragged list of arrays `(seq_len_i, hidden_dim)` when lengths differ
    """
    # Section 1: Load model
    bundle = load_model(model_name=model_name, device=device)
    resolved_layers = _resolve_layers(layers, bundle["model"])
    collected: Dict[int, List[np.ndarray]] = {layer: [] for layer in resolved_layers}
    seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
    n_total = len(seq_list)
    n_done = 0

    # Section 2: run batched forward passes and collect residue spans per layer.
    for batch_sequences in iter_batches(seq_list, batch_size):
        batch_list = list(batch_sequences)
        tokenized = tokenize_sequences(
            batch_list,
            tokenizer=bundle["tokenizer"],
            device=bundle["device"],
        )
        outputs = forward_pass(
            tokenized,
            model=bundle["model"],
            output_hidden_states=True,
        )
        attention_mask = tokenized["batch"]["attention_mask"].detach().cpu().numpy()
        token_spans = extract_sequence_token_spans(attention_mask)

        for layer in resolved_layers:
            hidden = outputs.hidden_states[layer].detach().cpu().numpy()
            for row_idx, span in enumerate(token_spans):
                collected[layer].append(np.asarray(hidden[row_idx, span, :]))
        n_done += len(batch_list)
        if progress_tag:
            _print_progress(progress_tag, n_done, n_total)

    # Section 3: convert per-layer list outputs to dense matrices.
    out: Dict[int, np.ndarray] = {}
    for layer, arrays in collected.items():
        if not arrays:
            out[int(layer)] = np.asarray([], dtype=np.float32)
            continue
        lengths = {int(np.asarray(a).shape[0]) for a in arrays}
        if len(lengths) != 1:
            print(
                f"Layer {int(layer)} embeddings have variable sequence lengths {sorted(lengths)}; "
                "returning ragged list for this layer."
            )
            out[int(layer)] = [np.asarray(a) for a in arrays]
        else:
            out[int(layer)] = np.stack([np.asarray(a) for a in arrays], axis=0)

    return out
