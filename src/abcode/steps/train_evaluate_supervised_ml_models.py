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
import json
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

from abcode.core.paths import resolve_project_root
from project_config.variables import address_dict
from abcode.tools.ml.workflow import run_supervised_ml_workflow


def default_user_inputs() -> Dict[str, Any]:
    return {
        "data_fbase": "examples",
        "data_subfolder": "ET096_R1-2",
        "dataset_fname": "ET096_R1-2.csv",
        "input_filename_prefix": "ET096_R1-2_",
        "sequence_base": None,
        "target_col": "foldchange_NBD_activity_25C",
        "classification_or_regression": "regression",
        "split_type_list": ["random", "mutres-modulo", "contiguous", "custom"],
        "feature_combinations_dict": {
            "onehot": ["one_hot"],
            "georgiev": ["georgiev"],
            "onehot_georgiev": ["one_hot", "georgiev"],
        },
        "model_list": ["rf", "ridge", "mlp_sklearn"],
        "hyperparameter_mode": "default",
        "tuning_metric": "spearman",
        "tuning_n_trials": 30,
        "summary_source_mode": "run_cache",
        "summary_metric_mode": "average",
        "k_folds": 5,
        "random_kfold_repeats": 1,
        "split_seed": 42,
        "mutres_col": "mutres_idx",
        "random_split_col": "fold_random_5",
        "mutres_split_col": "fold_mutres-modulo_5",
        "segment_col": "segment_index_0",
        "segment_index_range": None,
        "contiguous_split_col": "",
        "custom_split_col": "fold_modulo_5",
        "custom_test_value": 0,
        "custom_test_dataset_fname": "",
        "custom_test_data_subfolder": "",
        "custom_input_filename_prefix": "",
        "csv_suffix": "",
        "save_trained_models": True,
        "save_predictions": True,
        "train_full_data_model": False,
        "featurecombi_model_pair_to_extract_coefficients_for": [],
        "show_progress": True,
        "model_settings_repo_dir": "src/abcode/tools/ml/model_settings",
        "run_label": "",
        "sample_id_col": "",
    }


def _resolve_data_root_dir(data_fbase: str) -> Path:
    project_root = resolve_project_root()
    value = str(data_fbase or "").strip()
    if not value:
        raise ValueError("data_fbase cannot be empty.")

    if value in address_dict:
        return (project_root / address_dict[value]).resolve()

    as_path = Path(value).expanduser()
    if as_path.is_absolute():
        return as_path.resolve()
    return (project_root / as_path).resolve()


def train_evaluate_supervised_ml_models(inputs: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(default_user_inputs())
    payload.update(inputs or {})

    project_root = resolve_project_root()
    data_root_dir = _resolve_data_root_dir(str(payload.get("data_fbase", "examples")))
    settings_dir = Path(
        str(payload.get("model_settings_repo_dir", "src/abcode/tools/ml/model_settings"))
    ).expanduser()
    if not settings_dir.is_absolute():
        settings_dir = (project_root / settings_dir).resolve()

    payload["data_root_dir"] = str(data_root_dir)
    payload["model_settings_repo_dir"] = str(settings_dir)

    result = run_supervised_ml_workflow(payload)
    return {
        "status": result.get("status", "ok"),
        "step": "train_evaluate_supervised_ml_models",
        "data_root_dir": str(data_root_dir),
        **result,
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
    parser = argparse.ArgumentParser(description="Train/evaluate supervised ML models from precomputed encodings.")
    parser.add_argument("--inputs-json", type=str, default="", help="Inline JSON string of step inputs.")
    parser.add_argument("--inputs-file", type=str, default="", help="Path to JSON file of step inputs.")
    args = parser.parse_args()

    user_inputs = _load_inputs_from_args(args)
    result = train_evaluate_supervised_ml_models(user_inputs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    user_inputs = default_user_inputs()

    # === Dataset Inputs ===
    user_inputs["root_key"] = 'ECOHARVEST' # "MUTAGENESIS-DATA-BENCHMARKS"  # e.g. "examples"
    user_inputs["data_fbase"] = user_inputs["root_key"]
    user_inputs["data_subfolder"] = 'RML_R1' # "D7PM05_CLYGR_Somermeyer_2022"
    user_inputs["csv_suffix"] = ""
    default_dataset_fname = user_inputs["data_subfolder"] + user_inputs["csv_suffix"] + ".csv"
    user_inputs["dataset_fname"] = default_dataset_fname
    user_inputs["input_filename_prefix"] = user_inputs["data_subfolder"] + "_"
    user_inputs["sequence_base"] = "sequences/RML/RML-propeptide-mature.fasta" # "sequences/D7PM05_CLYGR_Somermeyer_2022.fasta"
    user_inputs["target_col"] = ['foldchange_TSO'] # ["DMS_score"]
    user_inputs["classification_or_regression"] = "regression"

    # === Split Settings ===
    user_inputs["split_type_list"] = ['random'] # ["custom", "mutres-modulo", "random"]
    user_inputs["k_folds"] = 5
    user_inputs["random_kfold_repeats"] = 5
    user_inputs["random_split_col"] = f'fold_random_{user_inputs["k_folds"]}'
    user_inputs["mutres_split_col"] = f'fold_mutres-modulo_{user_inputs["k_folds"]}'
    user_inputs["contiguous_split_col"] = f'fold_contiguous_{user_inputs["k_folds"]}'
    user_inputs["custom_split_col"] = "fold_custom"
    user_inputs["custom_test_value"] = 1
    user_inputs["segment_col"] = "segment_index_0"
    user_inputs["segment_index_range"] = None  # e.g. [0, 26] or [24, 26]

    # === Custom Test Dataset (Optional) ===
    user_inputs["custom_test_dataset_fname"] = None
    user_inputs["custom_test_data_subfolder"] = user_inputs["data_subfolder"]
    user_inputs["custom_input_filename_prefix"] = (
        None
        if user_inputs["custom_test_dataset_fname"] is None
        else str(user_inputs["custom_test_dataset_fname"]).split(".")[0]
    )

    # === Feature Combinations ===
    user_inputs["feature_combinations_dict"] = {
        "one_hot": ["one_hot"],
        "onehot_esm2_LLR-masked": ["one_hot", "esm2-650m_LLR-masked"],
        "onehot_esm2_LLR-wt": ["one_hot", "esm2-650m_LLR-wt"],
        "georgiev": ["georgiev"],
        "georgiev_esm2_LLR-masked": ["georgiev", "esm2-650m_LLR-masked"],
        "georgiev_esm2_LLR-wt": ["georgiev", "esm2-650m_LLR-wt"],
        # "esm2_seq_embeddings_LLR": ["esm2-650m_mean_pooled-33", "esm2-650m_LLR-masked"],
    }

    # === Models ===
    user_inputs["model_list"] = ["ridge"] # ["ridge", "xgboost"]

    # === Hyperparameter Mode ===
    # options: "default", "ntrain_preset", "preset_search", "optuna_search"
    user_inputs["hyperparameter_mode"] = "optuna_search"
    user_inputs["tuning_metric"] = "spearman"
    user_inputs["tuning_n_trials"] = 20
    user_inputs["summary_source_mode"] = "run_cache"  # "run_cache" or "all_saved_rows"
    user_inputs["summary_metric_mode"] = "average"  # "average" or "pooled"

    # === Save / Output Controls ===
    user_inputs["save_trained_models"] = False
    user_inputs["save_predictions"] = False
    user_inputs["train_full_data_model"] = False
    user_inputs["featurecombi_model_pair_to_extract_coefficients_for"] = [] # [('one_hot', 'ridge')]
    user_inputs["show_progress"] = True

    pprint(user_inputs)
    result = train_evaluate_supervised_ml_models(user_inputs)
    pprint(result)
