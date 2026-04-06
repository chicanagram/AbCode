from __future__ import annotations

import os

from sympy.codegen.ast import continue_

os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from abcode.tools.encodings.common import (
    _coerce_sequence_list,
    resolve_llr_cache_paths,
    save_layerwise_embeddings,
    score_mutants_from_llr_map,
)
from abcode.tools.openprotein.openprotein_utils import connect_openprotein_session, create_prompt
from abcode.tools.openprotein.align_msa_openprotein import create_openprotein_msa, upload_openprotein_msa
from abcode.tools.utils.plot_utils import plot_variant_heatmap
from project_config.variables import aaList

DEFAULT_MODEL_NAME = 'poet2' # 'poet'


def get_poet_prompt(
        sequence_seed: Union[str, Sequence[str]],
        msa_fpath: Optional[Union[str, Sequence[str]]] = None,
        num_prompts: Optional[int] = 3,
        session: Optional[Any] = None,
):
    # create session
    sess = session or connect_openprotein_session()
    # get prompt for each MSA
    processed_prompts = {}

    # create msa from scratch
    if msa_fpath is None:
        msa = create_openprotein_msa(seed_sequence=sequence_seed, session=sess, seq_fasta_path=None)
        prompt = create_prompt(msa, num_prompts=num_prompts, show_prompt=False)
        processed_prompts[sequence_seed] = prompt
    # upload msa
    else:
        if msa_fpath not in processed_prompts:
            msa = upload_openprotein_msa(msa_fpath, session=session)
            prompt = create_prompt(msa, num_prompts=num_prompts, show_prompt=False)
            processed_prompts[msa_fpath] = prompt
        else:
            prompt = processed_prompts[msa_fpath]
    return prompt


def _parse_openprotein_output_scores(results):
    results_avg = {
        item.mut_code: round(float(item.score.mean()), 4)
        for item in results
    }
    return results_avg


def process_single_site_analysis(ssp_results, sequence_base, vect_path, png_path):

    # parse results
    ssp_results_avg = _parse_openprotein_output_scores(ssp_results)
    wt_score = ssp_results_avg['WT']
    llr_vect = []
    llr_2D = pd.DataFrame(np.zeros((len(aaList), len(sequence_base))), columns=[f'{i+1}{wt_aa}' for i, wt_aa in enumerate(sequence_base)], index=aaList)
    for mut, score in ssp_results_avg.items():
        if mut != 'WT':
            mt_aa = mut[-1]
            pos = int(mut[1:-1])
            wt_aa = sequence_base[pos-1]
            pos_wtaa = f'{pos}{wt_aa}'
            score_vs_wt = score - wt_score
            llr_vect.append({'mutations': mut, 'LLR': score_vs_wt})
            llr_2D.loc[mt_aa, pos_wtaa] = score_vs_wt
    llr_vect = pd.DataFrame(llr_vect).round(4)
    llr_vect.to_csv(vect_path)

    # plot variant heatmap
    plot_variant_heatmap(
        -llr_2D.to_numpy(),
        sequence_base,
        N_res_per_heatmap_row=100,
        aa_list=aaList,
        savefig=png_path,
        figtitle="Predicted Effects of Mutations on Protein Sequence (LLR)",
    )
    return llr_vect


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
    # get kwargs
    session =  kwargs.get('session', connect_openprotein_session())
    prompt_id = kwargs.get('prompt_id', None)
    num_prompts = kwargs.get('num_prompts', 3)
    msa_fpath = kwargs.get('msa_fpath', None)

    # Section 1: normalize sequence inputs and scoring backend.
    base_list = _coerce_sequence_list(sequences_base, name="get_LLR_scores(sequences_base)", required=True)
    seq_list = _coerce_sequence_list(sequences, name="get_LLR_scores(sequences)", required=False)
    unique_bases = list(dict.fromkeys(base_list))
    if len(unique_bases) != 1:
        raise ValueError(
            "LLR cache/vector scoring currently supports exactly one unique sequence_base."
        )

    # Section 2: resolve output/cache paths.
    output_path = str(output_path) if output_path is not None else f"{model_name}_LLR"
    output_stem = Path(str(output_path))
    cache_stem = output_stem.with_name(f"{str(model_name)}_LLR")
    cache_info = resolve_llr_cache_paths(
        cache_stem,
        llr_cache_vect_filename_prefix=llr_cache_vect_filename_prefix,
    )
    vect_cache_path = Path(cache_info["vect_cache_path"])
    map_png_cache_path = Path(cache_info["map_png_cache_path"])

    # Cache resolution for vect CSV.
    cache_hit = bool(cache_info["vect_exists"])
    if cache_hit:
        print(f"[zeroshot] Loaded LLR cache: {vect_cache_path}")

    # Section 3: compute and persist full marginal map only on cache miss (or explicit refresh).
    if (not cache_hit) or bool(resave_llr_cache_if_found):

        poet = session.embedding.get_model(model_name)

        # get prompts
        if prompt_id is None:
            prompt = get_poet_prompt(unique_bases[0], msa_fpath, num_prompts, session)
            prompt_id = prompt.id

        # get single site prediction jobs
        sequence_base = unique_bases[0]
        sspjob = poet.single_site(prompt=prompt_id, sequence=sequence_base.encode())
        ssp_results = sspjob.wait()
        llr_vect = process_single_site_analysis(ssp_results, sequence_base, vect_cache_path, map_png_cache_path)

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
        print(f"[zeroshot] Saved raw scores: {output_path}.csv")

    return LLRsum_seq, prompt_id


def get_mean_PLL_scores(
    sequences: Optional[Union[str, Sequence[str]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    **kwargs,
) -> np.ndarray:
    """Compute mean pseudo-log-likelihood (mean PLL) scores for sequences without a shared base."""
    # get kwargs
    session =  kwargs.get('session', connect_openprotein_session())
    prompt_id = kwargs.get('prompt_id', None)
    num_prompts = kwargs.get('num_prompts', 3)
    msa_fpaths = kwargs.get('msa_fpaths', None)
    sequence_seed = kwargs.get('sequence_seed', None)

    # Section 1: validate/normalize sequence input and select marginal backend.
    base_list = _coerce_sequence_list(sequence_seed, name="get_LLR_scores(sequences_base)", required=True)
    seq_list = _coerce_sequence_list(sequences, name="get_mean_PLL_scores", required=True)
    unique_bases = list(dict.fromkeys(base_list))

    # Section 2: resolve output naming.
    output_str: Optional[str] = None
    if output_path is not None:
        output_str = str(output_path)
        if "LLR" in output_str:
            output_str = output_str.replace("LLR", "PLL")
        elif "PLL" not in output_str:
            output_str = f"{output_str}_PLL"

    # Section 3: compute per-position WT log-probabilities.
    # standardize msa_fpath
    if msa_fpaths is None or isinstance(msa_fpaths, str):
        msa_fpaths = [msa_fpaths] * len(unique_bases)
    poet = session.embedding.get_model(model_name)

    # Section 4: score provided sequences by summing cached mutation LLRs.
    # get prompts
    if prompt_id is None:
        prompt = get_poet_prompt(unique_bases[0], msa_fpaths, num_prompts, session)
        prompt_id = prompt.id
    scorejob = poet.score(prompt=prompt_id, sequences=sequences)
    score_results = scorejob.wait()
    score_results_avg = _parse_openprotein_output_scores(score_results)
    mean_pll_scores = np.array(list(score_results_avg.values()), dtype=np.float32)

    # Section 4: save outputs to .npy/.csv.
    if output_str is not None:
        np.save(f"{output_str}.npy", mean_pll_scores)
        pd.DataFrame({"sequence": seq_list, "PLL": mean_pll_scores}).round(6).to_csv(
            f"{output_str}.csv",
            index=False,
        )
        print(f"[PLL] Saved scores: {output_str}.csv")
    return mean_pll_scores, prompt_id


def get_embeddings(
        sequences: Union[str, Sequence[str]],
        sequences_base: Optional[Union[str, Sequence[str]]] = None,
        save_per_residue_embeddings: bool = False,
        get_embeddings_for_seq_base: bool = False,
        pool_method: Optional[str] = "mean",
        mutations: Optional[Sequence[str]] = None,
        sep: Optional[str] = '+',
        output_path: Optional[Union[str, Path]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        **kwargs,
):
    session = kwargs.get("session", connect_openprotein_session())
    prompt_id = kwargs.get("prompt_id", None)
    num_prompts = kwargs.get("num_prompts", 3)
    msa_fpath = kwargs.get("msa_fpath", None)
    sequence_seed = kwargs.get("sequence_seed", None)
    n_components = kwargs.get("n_components", 1024)
    sample_mutants_for_svd = kwargs.get("sample_mutants_for_svd", False)
    svd_data_reduction = kwargs.get("svd_data_reduction", None)

    # Section 1: reject unsupported requests for POET2 embedding helper.
    if get_embeddings_for_seq_base:
        raise ValueError("poet2.get_embeddings does not support base-sequence embedding outputs.")
    if pool_method is not None and str(pool_method).strip().lower() == "mut":
        raise ValueError("poet2.get_embeddings does not support mutation-pooled embeddings.")

    # Section 2: normalize inputs and resolve prompt.
    out_base = Path(output_path) if output_path else Path("embeddings")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {}
    seq_list = _coerce_sequence_list(sequences, name="get_embeddings(sequences)", required=True)
    seq_encoded = [s.encode("utf-8") for s in seq_list]

    base_list = _coerce_sequence_list(sequences_base, name="get_embeddings(sequences_base)", required=False)
    if prompt_id is None:
        if sequence_seed is None:
            if base_list:
                sequence_seed = list(dict.fromkeys(base_list))[0]
            else:
                sequence_seed = seq_list[0]
        prompt = get_poet_prompt(sequence_seed, msa_fpath, num_prompts, session)
        prompt_id = prompt.id

    poet = session.embedding.get_model(model_name)
    futures: Dict[str, Any] = {}

    # Section 3: submit embedding jobs for requested outputs.
    if save_per_residue_embeddings:
        futures["per_residue"] = poet.embed(seq_encoded, reduction=None, prompt=prompt_id)
    if pool_method is not None and str(pool_method).strip().lower() == "mean":
        futures["mean_pooled"] = poet.embed(seq_encoded, reduction="MEAN", prompt=prompt_id)
    if pool_method is not None and str(pool_method).strip().lower() == "svd":
        futures["svd_pooled"] = compute_svd_pooling(
            sequences, sequence_seed,
            n_components, sample_mutants_for_svd,
            svd_data_reduction, model_name, session, prompt_id
        )

    # Section 4: wait for all futures.
    for _, future in futures.items():
        future.wait_until_done(verbose=True)

    # Section 5: parse and save sequence embeddings with consistent naming.
    # Force float32 so saved matrices match the expected precision across PLMs.
    for embedding_type, future in futures.items():
        emb_rows = future.get()  # expected: List[(seq_bytes, np.ndarray)]
        res_key = 'pooled' if embedding_type.find('pooled')>-1 else 'per_residue'
        arr_list = [
            _collapse_prompt_ensemble(np.asarray(row[1], dtype=np.float32), embedding_type, num_prompts)
            for row in emb_rows
        ]
        lengths = {int(arr.shape[0]) for arr in arr_list}
        emb_layer = {12: np.stack(arr_list, axis=0).astype(np.float32, copy=False)}

        if embedding_type == "per_residue" and len(lengths) != 1:
            raise ValueError(f"POET2 per_residue embeddings have variable sequence lengths {sorted(lengths)}; cannot stack into one dense matrix.")

        result[f"{res_key}_paths"] = save_layerwise_embeddings(
            emb_layer,
            output_stem=out_base,
            suffix=embedding_type,
            log_tag="get_embeddings/poet2",
        )

    return result, prompt_id


def compute_svd_pooling(
        sequences,
        sequence_seed,
        n_components=1024,
        sample_mutants_for_svd=False,
        svd_data_reduction=None,
        model_name: str = DEFAULT_MODEL_NAME,
        session: Optional[Any] = None,
        prompt_id: Optional[str] = None,
):

    # sample variants for constructing SVD
    if sample_mutants_for_svd:
        sequences_to_compute_svd_on = []
        sequence_mut = list(sequence_seed)
        for i, wt_aa in enumerate(sequence_seed):
            for mt_aa in aaList:
                if mt_aa != wt_aa:
                    sequence_mut[i] = mt_aa
                    sequences_to_compute_svd_on.append(''.join(sequence_mut))
    else:
        sequences_to_compute_svd_on = sequences

    # get SVD model
    model = session.embedding.get_model(model_name)
    fit_kwargs = {"n_components": n_components, "prompt": prompt_id}
    if svd_data_reduction is not None:
        fit_kwargs["reduction"] = svd_data_reduction
    svd = model.fit_svd(sequences_to_compute_svd_on, **fit_kwargs)
    svd.wait_until_done(verbose=True)
    svd_embed_future = svd.embed(sequences)

    return svd_embed_future

def _collapse_prompt_ensemble(
        arr: np.ndarray,
        embedding_type: str,
        num_prompts: int,
) -> np.ndarray:
    # OpenProtein/POET can return one embedding per prompt (e.g., num_prompts=3).
    # We collapse that axis by averaging so downstream artifacts remain model-consistent.
    n_prompts = int(num_prompts) if num_prompts is not None else None
    out = np.asarray(arr, dtype=np.float32)
    if n_prompts is None or n_prompts <= 1:
        return out

    # Mean-pooled expected target shape per sequence: (hidden_dim,).
    if embedding_type == "mean_pooled" and out.ndim == 2:
        if out.shape[0] == n_prompts:
            return out.mean(axis=0, dtype=np.float32)
        if out.shape[1] == n_prompts:
            return out.mean(axis=1, dtype=np.float32)
        return out

    # Per-residue expected target shape per sequence: (seq_len, hidden_dim).
    if embedding_type == "per_residue" and out.ndim == 3:
        # Prefer collapsing explicit prompt dimension when detected.
        prompt_axes = [ax for ax, dim in enumerate(out.shape) if dim == n_prompts]
        if prompt_axes:
            return out.mean(axis=prompt_axes[0], dtype=np.float32)
        # Fallback: keep tensor unchanged if prompt axis is ambiguous.
        return out
    return out
