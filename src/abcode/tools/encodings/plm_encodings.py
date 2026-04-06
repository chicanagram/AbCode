from __future__ import annotations

import importlib
import gc
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from abcode.tools.encodings.common import _sanitize_name, compute_pooled_embeddings, save_layerwise_embeddings
from abcode.tools.openprotein.openprotein_utils import connect_openprotein_session


def _parse_plm_feature_name(feature_name: str) -> Tuple[str, str]:
    """
    Parse a PLM feature string into (model_prefix, feature_suffix).

    Supported suffixes:
    - LLR
    - PLL
    - per_residue
    - mean_pooled
    - mut_pooled
    """
    name = str(feature_name).strip()
    suffixes = ("mean_pooled", "mut_pooled", "per_residue", "svd_pooled", "PLL", "LLR")
    for suffix in suffixes:
        marker = f"_{suffix}"
        if name.endswith(marker):
            return name[: -len(marker)], suffix
    raise ValueError(
        f"Unsupported PLM feature name '{feature_name}'. "
        "Expected '<model>_(LLR|PLL|per_residue|mean_pooled|mut_pooled|svd_pooled)'."
    )


def _resolve_plm_name(model_prefix: str) -> str:
    """Route a model prefix to an available plm_name module."""
    lowered = str(model_prefix).strip().lower()
    if lowered.startswith("esmc"):
        return "esmc"
    if lowered.startswith(("esm2", "esm1", "esm")):
        return "esm2"
    if lowered.startswith("poet2"):
        return "poet2"
    raise ValueError(f"Could not infer PLM plm_name from model prefix '{model_prefix}'.")


def _load_plm_module(plm_name: str):
    """Import a PLM plm_name module and return it."""
    module_name = f"abcode.tools.encodings.{plm_name}"
    return importlib.import_module(module_name)


def _load_layerwise_arrays(paths_by_layer: Dict[str, str]) -> Dict[int, np.ndarray]:
    # Section 1: load saved layer-wise `.npy`/`.npz` payloads into memory.
    out: Dict[int, np.ndarray] = {}
    for layer, path in paths_by_layer.items():
        p = Path(path)
        if p.suffix.lower() == ".npz":
            with np.load(str(p), allow_pickle=False) as npz:
                keys = sorted(npz.files, key=lambda x: int(str(x)))
                out[int(layer)] = [np.asarray(npz[k]) for k in keys]
        else:
            out[int(layer)] = np.load(str(path), allow_pickle=False)
    return out


def _load_layerwise_shapes(paths_by_layer: Dict[str, str]) -> Dict[int, List[int]]:
    # Section 1: load only matrix shapes for trace metadata.
    shapes: Dict[int, List[int]] = {}
    for layer, path in paths_by_layer.items():
        p = Path(path)
        if p.suffix.lower() == ".npz":
            with np.load(str(p), allow_pickle=False) as npz:
                keys = sorted(npz.files, key=lambda x: int(str(x)))
                lengths = [int(np.asarray(npz[k]).shape[0]) for k in keys]
                hidden_dim = int(np.asarray(npz[keys[0]]).shape[1]) if keys else 0
                if lengths and len(set(lengths)) == 1:
                    shapes[int(layer)] = [len(keys), lengths[0], hidden_dim]
                else:
                    # Ragged marker shape: [n_sequences, -1, hidden_dim]
                    shapes[int(layer)] = [len(keys), -1, hidden_dim]
        else:
            arr = np.load(str(path), allow_pickle=False)
            shapes[int(layer)] = list(arr.shape)
    return shapes


def _layer_payload_shape(payload: Union[np.ndarray, Sequence[np.ndarray]]) -> List[int]:
    """
    Compute a stable shape summary for a layer payload.

    Dense payloads return their exact shape.
    Ragged payloads return `[n_sequences, -1, hidden_dim]`.
    Uniform list payloads return `[n_sequences, seq_len, hidden_dim]`.
    """
    if isinstance(payload, np.ndarray) and payload.dtype != object:
        return list(payload.shape)

    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.ndim == 1:
        rows = [np.asarray(x) for x in payload.tolist()]
    elif isinstance(payload, (list, tuple)):
        rows = [np.asarray(x) for x in payload]
    else:
        arr = np.asarray(payload)
        return list(arr.shape)

    if not rows:
        return [0, 0, 0]

    lengths = [int(r.shape[0]) for r in rows]
    hidden_dim = int(rows[0].shape[1]) if rows[0].ndim > 1 else 0
    if len(set(lengths)) == 1:
        return [len(rows), lengths[0], hidden_dim]
    return [len(rows), -1, hidden_dim]


def _unpack_optional_prompt(out: Any, prompt_id: Optional[str]) -> Tuple[Any, Optional[str]]:
    # Section 1: support both `value` and `(value, prompt_id)` return signatures.
    if isinstance(out, tuple):
        if len(out) == 0:
            return None, prompt_id
        if len(out) == 1:
            return out[0], prompt_id
        return out[0], out[1]
    return out, prompt_id


def _delete_artifact_paths(paths_by_layer: Dict[str, str]) -> None:
    # Section 1: remove temporary files used only for intermediate pooling.
    for _, path in paths_by_layer.items():
        p = Path(path)
        if p.exists():
            p.unlink()


def _rename_layerwise_paths(
    paths_by_layer: Dict[str, str],
    target_stem: Union[str, Path],
) -> Dict[str, str]:
    # Section 1: rename saved layerwise files to a target stem preserving layer suffix.
    stem = Path(target_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    renamed: Dict[str, str] = {}
    for layer, old_path in paths_by_layer.items():
        old = Path(old_path)
        suffix = old.suffix if old.suffix else ".npy"
        new = stem.parent / f"{stem.name}-{int(layer)}{suffix}"
        if old.resolve() != new.resolve():
            old.replace(new)
        renamed[str(int(layer))] = str(new)
    return renamed


def _resolve_layers_for_model(
    model_prefix: str,
    layers: Optional[Union[Sequence[int], Mapping[str, Sequence[int]]]],
) -> Optional[Sequence[int]]:
    """
    Resolve layer selection for a specific model.

    Supported inputs:
    - `None`: plm_name default behavior
    - `List[int]`: shared layer list for all PLMs
    - `Dict[str, List[int]]`: per-model layer list keyed by PLM name
    """
    # Section 1: pass through None or shared sequence input.
    if layers is None:
        return None
    if isinstance(layers, (list, tuple)):
        return [int(x) for x in layers]

    # Section 2: resolve per-model dictionary keys.
    if isinstance(layers, Mapping):
        exact = layers.get(model_prefix)
        lowered_map = {str(k).strip().lower(): v for k, v in layers.items()}
        lowered = lowered_map.get(str(model_prefix).strip().lower())
        value = exact if exact is not None else lowered
        if value is None:
            return None
        return [int(x) for x in value]

    raise TypeError(
        "layers must be None, List[int], or Dict[str, List[int]] keyed by PLM model name."
    )


def _feature_output_path(out_dir: Path, file_prefix: str, feature_name: str) -> str:
    """Build a normalized output stem path for a feature."""
    return str(out_dir / f"{file_prefix}{_sanitize_name(feature_name)}")


def _score_result_record(
    *,
    feature_name: str,
    model_prefix: str,
    plm_name: str,
    output_path: str,
    scores: Any,
) -> Dict[str, Any]:
    """Build a standardized scalar-score result record."""
    return {
        "feature_name": feature_name,
        "model_prefix": model_prefix,
        "plm_name": plm_name,
        "output_path": output_path,
        "shape": list(scores.shape) if isinstance(scores, np.ndarray) else None,
    }


def _artifact_result_record(
    *,
    feature_name: str,
    model_prefix: str,
    plm_name: str,
    artifacts: Dict[str, Any],
    shape_by_layer: Dict[int, List[int]],
    base_shape_by_layer: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Any]:
    """Build a standardized embedding-artifact result record."""
    record: Dict[str, Any] = {
        "feature_name": feature_name,
        "model_prefix": model_prefix,
        "plm_name": plm_name,
        "artifacts": artifacts,
        "shape_by_layer": shape_by_layer,
    }
    if base_shape_by_layer is not None:
        record["base_shape_by_layer"] = base_shape_by_layer
    return record


def get_plm_encodings(
    plm_feature_sets: Sequence[str],
    sequence_list: Optional[Sequence[str]],
    sequence_base_list: Optional[Sequence[str]] = None,
    *,
    encodings_dir: str,
    filename_prefix: str = "",
    marginal_type: str = "wt",
    llr_cache_vect_filename_prefix: str = "",
    resave_llr_cache_if_found: bool = False,
    mutations: Optional[Sequence[str]] = None,
    sep: str = "+",
    layers: Optional[Union[Sequence[int], Mapping[str, Sequence[int]]]] = None,
    n_components: int = 256,
    sample_mutants_for_svd: bool = False,
    svd_data_reduction: Optional[str] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
    get_embeddings_for_seq_base: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate PLM-derived encodings by parsing feature-set names and dispatching by plm_name.

    All artifact paths are rooted under `encodings_dir`.
    """
    # Section 1: validate inputs and create output directory.
    if not plm_feature_sets:
        return {}
    out_dir = Path(encodings_dir)
    file_prefix = str(filename_prefix or "")

    # Section 2: group requested features by model so each model is processed once.
    grouped: Dict[str, Dict[str, Any]] = {}
    ordered_features: List[str] = []
    for feature_name in plm_feature_sets:
        model_prefix, suffix = _parse_plm_feature_name(feature_name)
        plm_name = _resolve_plm_name(model_prefix)
        if model_prefix not in grouped:
            grouped[model_prefix] = {
                "plm_name": plm_name,
                "suffixes": set(),
            }
        grouped[model_prefix]["suffixes"].add(suffix)
        ordered_features.append(feature_name)

    # Section 3: process each model group and build per-feature result records.
    results: Dict[str, Dict[str, Any]] = {}
    for model_prefix, group_info in grouped.items():
        # Section 3.0: resolve model-specific layer selection.
        model_layers = _resolve_layers_for_model(model_prefix, layers)

        plm_name = str(group_info["plm_name"])
        module = _load_plm_module(plm_name)
        requested_suffixes = set(group_info["suffixes"])

        session = None
        prompt_id = None
        if plm_name in ['poet2']:
            session = connect_openprotein_session()

        # Section 3A: run LLR scoring once per model if requested.
        if "LLR" in requested_suffixes:
            if sequence_base_list is None:
                raise ValueError(
                    f"Feature '{model_prefix}_LLR' requires sequence_base. "
                    f"Use '{model_prefix}_PLL' when no base sequence is provided."
                )
            if not hasattr(module, "get_LLR_scores"):
                raise NotImplementedError(
                    f"Backend '{plm_name}' does not implement get_LLR_scores."
                )
            llr_feature_name = f"{model_prefix}_LLR"
            output_path = _feature_output_path(out_dir, file_prefix, llr_feature_name)

            print(f'Obtaining LLR scores for {model_prefix}...')
            llr_out = module.get_LLR_scores(
                sequences_base=sequence_base_list,
                sequences=sequence_list,
                mutations=mutations,
                output_path=output_path,
                llr_cache_vect_filename_prefix=llr_cache_vect_filename_prefix,
                resave_llr_cache_if_found=resave_llr_cache_if_found,
                model_name=model_prefix,
                marginal_type=marginal_type,
                batch_size=batch_size,
                device=device,
                session=session,
                prompt_id=prompt_id,
            )
            scores, prompt_id = _unpack_optional_prompt(llr_out, prompt_id)
            results[llr_feature_name] = _score_result_record(
                feature_name=llr_feature_name,
                model_prefix=model_prefix,
                plm_name=plm_name,
                output_path=output_path,
                scores=scores,
            )

        # Section 3A.1: run mean PLL scoring once per model if requested.
        if "PLL" in requested_suffixes:
            if sequence_list is None:
                raise ValueError(
                    f"Feature '{model_prefix}_PLL' requires non-empty sequence inputs."
                )
            if not hasattr(module, "get_mean_PLL_scores"):
                raise NotImplementedError(
                    f"Backend '{plm_name}' does not implement get_mean_PLL_scores."
                )
            pll_feature_name = f"{model_prefix}_PLL"
            output_path = _feature_output_path(out_dir, file_prefix, pll_feature_name)

            print(f'Obtaining PLL scores for {model_prefix}...')
            pll_out = module.get_mean_PLL_scores(
                sequences=sequence_list,
                output_path=output_path,
                model_name=model_prefix,
                marginal_type=marginal_type,
                batch_size=batch_size,
                device=device,
                session=session,
                prompt_id=prompt_id,
            )
            scores, prompt_id = _unpack_optional_prompt(pll_out, prompt_id)
            results[pll_feature_name] = _score_result_record(
                feature_name=pll_feature_name,
                model_prefix=model_prefix,
                plm_name=plm_name,
                output_path=output_path,
                scores=scores,
            )


        # Section 3B: run embeddings once per model, then derive requested artifacts.
        embedding_suffixes = {"per_residue", "mean_pooled", "mut_pooled", "svd_pooled"} & requested_suffixes
        if not embedding_suffixes:
            continue
        if sequence_list is None:
            raise ValueError("Embedding features require a non-empty sequence_list.")
        if "mut_pooled" in embedding_suffixes and mutations is None:
            raise ValueError("mut_pooled feature requested but mutations is None.")
        if not hasattr(module, "get_embeddings"):
            raise NotImplementedError(f"Backend '{plm_name}' does not implement get_embeddings.")

        # Section 3B.1: POET2 branch.
        if plm_name == "poet2":
            if "mut_pooled" in embedding_suffixes:
                raise ValueError("poet2 does not support mut_pooled embeddings.")
            if get_embeddings_for_seq_base:
                raise ValueError("poet2 does not support base-sequence embedding outputs.")

            save_per_residue_embeddings = "per_residue" in embedding_suffixes
            if "mean_pooled" in embedding_suffixes:
                pool_method = "mean"
            elif "svd_pooled" in embedding_suffixes:
                pool_method = "svd"
            else:
                pool_method = None

            print(f"Obtaining embeddings for {model_prefix}...")
            shared_stem = Path(_feature_output_path(out_dir, file_prefix, model_prefix))
            embed_out = module.get_embeddings(
                sequences=sequence_list,
                sequences_base=None,
                save_per_residue_embeddings=save_per_residue_embeddings,
                get_embeddings_for_seq_base=False,
                pool_method=pool_method,
                mutations=None,
                sep=sep,
                output_path=shared_stem,
                layers=model_layers,
                model_name=model_prefix,
                n_components=int(n_components),
                sample_mutants_for_svd=bool(sample_mutants_for_svd),
                svd_data_reduction=svd_data_reduction,
                batch_size=batch_size,
                device=device,
                session=session,
                prompt_id=prompt_id,
            )
            embed_result, prompt_id = _unpack_optional_prompt(embed_out, prompt_id)

            for embedding_type in embedding_suffixes:
                feature_name = f"{model_prefix}_{embedding_type}"
                res_path_key = 'pooled_paths' if embedding_type.find('pooled')>-1 else 'per_residue_paths'
                embedding_paths = embed_result.get(res_path_key, {})
                if not embedding_paths:
                    raise RuntimeError(f"Backend '{plm_name}' did not return {res_path_key} ({embedding_type}).")

                target_stem = Path(_feature_output_path(out_dir, file_prefix, feature_name))
                embedding_paths = _rename_layerwise_paths(embedding_paths, target_stem)
                results[feature_name] = _artifact_result_record(
                    feature_name=feature_name,
                    model_prefix=model_prefix,
                    plm_name=plm_name,
                    artifacts={
                        res_path_key: embedding_paths,
                        f"base_{res_path_key}": None,
                    },
                    shape_by_layer=_load_layerwise_shapes(embedding_paths),
                )

        else:
            # Section 3B.2: non-POET branch (force per_residue first; derive pooled downstream).
            pooled_only = set(embedding_suffixes) - {"per_residue"}
            direct_pool_method: Optional[str] = None
            if "per_residue" not in embedding_suffixes and pooled_only == {"mean_pooled"}:
                direct_pool_method = "mean"
            elif "per_residue" not in embedding_suffixes and pooled_only == {"mut_pooled"}:
                direct_pool_method = "mut"

            # Fast path: for a single pooled feature, ask backend to pool directly and avoid per-residue artifacts.
            if direct_pool_method is not None:
                print(f'Obtaining embeddings for {model_prefix}...')
                feature_name = f"{model_prefix}_{'mean_pooled' if direct_pool_method == 'mean' else 'mut_pooled'}"
                feature_base_path = Path(_feature_output_path(out_dir, file_prefix, feature_name))
                embed_out = module.get_embeddings(
                    sequences=sequence_list,
                    sequences_base=sequence_base_list,
                    save_per_residue_embeddings=False,
                    get_embeddings_for_seq_base=bool(get_embeddings_for_seq_base and sequence_base_list is not None),
                    pool_method=direct_pool_method,
                    mutations=mutations if direct_pool_method == "mut" else None,
                    sep=sep,
                    output_path=feature_base_path,
                    layers=model_layers,
                    model_name=model_prefix,
                    n_components=int(n_components),
                    sample_mutants_for_svd=bool(sample_mutants_for_svd),
                    svd_data_reduction=svd_data_reduction,
                    batch_size=batch_size,
                    device=device,
                    session=session,
                    prompt_id=prompt_id,
                )
                embed_result, prompt_id = _unpack_optional_prompt(embed_out, prompt_id)
                pooled_paths = embed_result.get("pooled_paths", {})
                if not pooled_paths:
                    raise RuntimeError(f"Backend '{plm_name}' did not return pooled_paths ({direct_pool_method}).")
                base_pooled_paths = embed_result.get("base_pooled_paths", None)
                results[feature_name] = _artifact_result_record(
                    feature_name=feature_name,
                    model_prefix=model_prefix,
                    plm_name=plm_name,
                    artifacts={
                        "pooled_paths": pooled_paths,
                        "base_pooled_paths": base_pooled_paths,
                    },
                    shape_by_layer=_load_layerwise_shapes(pooled_paths),
                    base_shape_by_layer=(
                        _load_layerwise_shapes(base_pooled_paths) if base_pooled_paths else None
                    ),
                )
                gc.collect()
                continue

            print(f'Obtaining embeddings for {model_prefix}...')
            per_res_feature_name = f"{model_prefix}_per_residue"
            per_res_base_path = Path(_feature_output_path(out_dir, file_prefix, per_res_feature_name))

            embed_out = module.get_embeddings(
                sequences=sequence_list,
                sequences_base=sequence_base_list,
                save_per_residue_embeddings=True,
                get_embeddings_for_seq_base=bool(get_embeddings_for_seq_base and sequence_base_list is not None),
                pool_method=None,
                mutations=None,
                sep=sep,
                output_path=per_res_base_path,
                layers=model_layers,
                model_name=model_prefix,
                n_components=int(n_components),
                sample_mutants_for_svd=bool(sample_mutants_for_svd),
                svd_data_reduction=svd_data_reduction,
                batch_size=batch_size,
                device=device,
                session=session,
                prompt_id=prompt_id,
            )
            embed_result, prompt_id = _unpack_optional_prompt(embed_out, prompt_id)
            per_res_paths = embed_result.get("per_residue_paths", {})
            if not per_res_paths:
                raise RuntimeError(f"Backend '{plm_name}' did not return per_residue_paths.")
            per_res_arrays = _load_layerwise_arrays(per_res_paths)

            base_per_res_paths = embed_result.get("base_per_residue_paths", {})
            base_per_res_arrays: Optional[Dict[int, np.ndarray]] = None
            if base_per_res_paths:
                base_per_res_arrays = _load_layerwise_arrays(base_per_res_paths)

            # Section 3C: persist per-residue feature if requested.
            if "per_residue" in embedding_suffixes:
                per_res_shape_by_layer = {
                    int(layer): _layer_payload_shape(arr) for layer, arr in per_res_arrays.items()
                }
                base_per_res_shape_by_layer = (
                    {int(layer): _layer_payload_shape(arr) for layer, arr in base_per_res_arrays.items()}
                    if base_per_res_arrays is not None
                    else None
                )
                results[per_res_feature_name] = _artifact_result_record(
                    feature_name=per_res_feature_name,
                    model_prefix=model_prefix,
                    plm_name=plm_name,
                    artifacts={
                        "per_residue_paths": per_res_paths,
                        "base_per_residue_paths": base_per_res_paths or None,
                    },
                    shape_by_layer=per_res_shape_by_layer,
                    base_shape_by_layer=base_per_res_shape_by_layer,
                )

            # Section 3D: derive and persist mean-pooled feature from cached per-residue arrays.
            if "mean_pooled" in embedding_suffixes:
                mean_feature_name = f"{model_prefix}_mean_pooled"
                mean_base_path = Path(_feature_output_path(out_dir, file_prefix, mean_feature_name))
                mean_pooled = compute_pooled_embeddings(per_res_arrays, pool_method="mean", mutations=None, sep=sep)
                mean_paths = save_layerwise_embeddings(
                    mean_pooled,
                    output_stem=mean_base_path,
                    suffix="mean_pooled",
                    log_tag="get_plm_encodings",
                )
                base_mean_paths = None
                if base_per_res_arrays is not None and get_embeddings_for_seq_base:
                    base_mean = compute_pooled_embeddings(base_per_res_arrays, pool_method="mean", mutations=None, sep=sep)
                    base_mean_paths = save_layerwise_embeddings(
                        base_mean,
                        output_stem=mean_base_path,
                        suffix="mean_pooled",
                        file_suffix="_base",
                        log_tag="get_plm_encodings",
                    )
                mean_shape_by_layer = {int(layer): list(arr.shape) for layer, arr in mean_pooled.items()}
                base_mean_shape_by_layer = (
                    {int(layer): list(arr.shape) for layer, arr in base_mean.items()}
                    if base_per_res_arrays is not None and get_embeddings_for_seq_base
                    else None
                )
                results[mean_feature_name] = _artifact_result_record(
                    feature_name=mean_feature_name,
                    model_prefix=model_prefix,
                    plm_name=plm_name,
                    artifacts={
                        "pooled_paths": mean_paths,
                        "base_pooled_paths": base_mean_paths,
                    },
                    shape_by_layer=mean_shape_by_layer,
                    base_shape_by_layer=base_mean_shape_by_layer,
                )

            # Section 3E: derive and persist mutation-pooled feature from cached per-residue arrays.
            if "mut_pooled" in embedding_suffixes:
                mut_feature_name = f"{model_prefix}_mut_pooled"
                mut_base_path = Path(_feature_output_path(out_dir, file_prefix, mut_feature_name))
                mut_pooled = compute_pooled_embeddings(
                    per_res_arrays,
                    pool_method="mut",
                    mutations=mutations,
                    sep=sep,
                )
                mut_paths = save_layerwise_embeddings(
                    mut_pooled,
                    output_stem=mut_base_path,
                    suffix="mut_pooled",
                    log_tag="get_plm_encodings",
                )
                base_mut_paths = None
                if base_per_res_arrays is not None and get_embeddings_for_seq_base:
                    base_mut = compute_pooled_embeddings(base_per_res_arrays, pool_method="mut", mutations=mutations, sep=sep)
                    base_mut_paths = save_layerwise_embeddings(
                        base_mut,
                        output_stem=mut_base_path,
                        suffix="mut_pooled",
                        file_suffix="_base",
                        log_tag="get_plm_encodings",
                    )
                mut_shape_by_layer = {int(layer): list(arr.shape) for layer, arr in mut_pooled.items()}
                base_mut_shape_by_layer = (
                    {int(layer): list(arr.shape) for layer, arr in base_mut.items()}
                    if base_per_res_arrays is not None and get_embeddings_for_seq_base
                    else None
                )
                results[mut_feature_name] = _artifact_result_record(
                    feature_name=mut_feature_name,
                    model_prefix=model_prefix,
                    plm_name=plm_name,
                    artifacts={
                        "pooled_paths": mut_paths,
                        "base_pooled_paths": base_mut_paths,
                    },
                    shape_by_layer=mut_shape_by_layer,
                    base_shape_by_layer=base_mut_shape_by_layer,
                )

            # Remove temporary per-residue artifacts when they were used only as intermediates.
            keep_per_residue = "per_residue" in embedding_suffixes
            if not keep_per_residue:
                _delete_artifact_paths(per_res_paths)
                if base_per_res_paths:
                    _delete_artifact_paths(base_per_res_paths)
            gc.collect()

    # Section 4: return only requested features, preserving the original request order.
    ordered_results: Dict[str, Dict[str, Any]] = {}
    for feature_name in ordered_features:
        model_prefix, suffix = _parse_plm_feature_name(feature_name)
        normalized_feature_name = f"{model_prefix}_{suffix}"
        if normalized_feature_name in results:
            ordered_results[feature_name] = results[normalized_feature_name]
        elif feature_name in results:
            ordered_results[feature_name] = results[feature_name]
        else:
            raise RuntimeError(f"Missing result record for requested PLM feature '{feature_name}'.")

    return ordered_results
