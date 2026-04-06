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
from typing import Any, Dict

from abcode.core.paths import resolve_project_root
from project_config.variables import address_dict
from abcode.tools.ml.visualize_ml_results import run_visualize_ml_results


def default_user_inputs() -> Dict[str, Any]:
    return {
        "data_fbase": "examples",
        "data_subfolder": "ET096_R1-2",
        "user_outputs": {},  # from train_evaluate_supervised_ml_models(...)
        "metrics_summary_csv_path": "",
        "metrics_summary_fname": "",
        "model_name_list": ["best"],
        "metric_col": "test_spearman",
        "target_col_list": [],
        "split_type_list": [],
        "feature_label_list": [],
        "higher_is_better": None,
        "save_figure": True,
        "show_figure": True,
        "figure_output_dir": "",
        "figure_fname": "",
        "figure_dpi": 200,
        "figsize": (8, 5),
        "x_limits": None,
        "y_limits": None,
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


def visualize_ml_results(inputs: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(default_user_inputs())
    payload.update(dict(inputs or {}))
    payload["data_root_dir"] = str(_resolve_data_root_dir(str(payload.get("data_fbase", "examples"))))
    return run_visualize_ml_results(payload)


def _load_inputs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.inputs_json:
        return dict(json.loads(args.inputs_json))
    if args.inputs_file:
        p = Path(args.inputs_file).expanduser().resolve()
        return dict(json.loads(p.read_text(encoding="utf-8")))
    return default_user_inputs()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize supervised ML metrics summary outputs.")
    parser.add_argument("--inputs-json", type=str, default="", help="Inline JSON string of step inputs.")
    parser.add_argument("--inputs-file", type=str, default="", help="Path to JSON file of step inputs.")
    args = parser.parse_args()

    user_inputs = _load_inputs_from_args(args)
    result = visualize_ml_results(user_inputs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
