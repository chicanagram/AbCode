from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _canonical_model_key(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key == "random_forest":
        return "rf"
    return key


def _import_sklearn_models():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Lasso, LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    return {
        "rf": (RandomForestClassifier, RandomForestRegressor),
        "random_forest": (RandomForestClassifier, RandomForestRegressor),
        "ridge": (LogisticRegression, Ridge),
        "lasso": (LogisticRegression, Lasso),
        "mlp_sklearn": (MLPClassifier, MLPRegressor),
    }


def _import_xgboost_models():
    from xgboost import XGBClassifier, XGBRegressor

    return XGBClassifier, XGBRegressor


def _import_lightgbm_models():
    from lightgbm import LGBMClassifier, LGBMRegressor

    return LGBMClassifier, LGBMRegressor


def _default_params(model_name: str, task_type: str) -> Dict[str, Any]:
    key = _canonical_model_key(model_name)
    task = str(task_type).strip().lower()
    defaults = {
        "rf": {"shared": {"n_estimators": 300, "random_state": 42, "n_jobs": -1}},
        "ridge": {
            "shared": {"reg_strength": 1.0, "max_iter": 1000},
        },
        "lasso": {
            "shared": {"reg_strength": 1.0, "penalty": "l1", "solver": "liblinear", "max_iter": 1000},
        },
        "mlp_sklearn": {
            "shared": {
                "hidden_layer_sizes": (64, 64),
                "solver": "adam",
                "learning_rate_init": 1e-3,
                "batch_size": 32,
                "max_iter": 400,
                "early_stopping": True,
                "n_iter_no_change": 15,
                "tol": 1e-5,
                "random_state": 42,
            }
        },
        "mlp_pytorch": {
            "shared": {
                "n_layers": 2,
                "hidden_size": 32,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "max_epochs": 400,
                "patience": 15,
                "min_delta": 1e-5,
                "validation_split": 0.2,
                "random_state": 42,
                "device": "cpu",
            }
        },
        "xgboost": {"shared": {"n_estimators": 600, "learning_rate": 0.08, "max_depth": 5, "random_state": 42}}, # OLD DEFAULTS
        "lightgbm": {"shared": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": -1, "random_state": 42}},
    }
    record = defaults.get(key, {})
    merged = dict(record.get("shared", {}))
    merged.update(dict(record.get(task, {})))
    return adapt_tuned_params_for_task(
        model_name=key,
        task_type=task,
        tuned_params=merged,
    )


def load_model_params(
    *,
    model_name: str,
    task_type: str,
    settings_repo_dir: Path,
    model_params_override: Dict[str, Any] | None,
) -> Dict[str, Any]:
    key = _canonical_model_key(model_name)
    task = str(task_type).strip().lower()
    resolved: Dict[str, Any] | None = None
    if model_params_override:
        if model_name in model_params_override:
            resolved = dict(model_params_override[model_name])
        if key in model_params_override:
            resolved = dict(model_params_override[key])

    if resolved is None:
        settings_path = settings_repo_dir / f"{key}.json"
        if settings_path.exists():
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                scoped = raw.get(task)
                if isinstance(scoped, dict):
                    resolved = dict(scoped)
                else:
                    resolved = dict(raw)

    if resolved is None:
        resolved = _default_params(model_name=key, task_type=task)

    return adapt_tuned_params_for_task(
        model_name=key,
        task_type=task,
        tuned_params=resolved,
    )


# Empirical presets from D7PM05 summary trends.
# Each rule is applied by model + feature_label, with optional n_train window.
FEATURE_NTRAIN_PARAM_PRESETS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "ridge": {
        "esm2_seq_embeddings_LLR": [
            {"n_train_min": 0, "n_train_max": None, "params": {"reg_strength": 0.01}},
        ],
        "onehot_esm2_LLR": [
            {"n_train_min": 0, "n_train_max": 2500, "params": {"reg_strength": 0.05}},
            {"n_train_min": 2500, "n_train_max": None, "params": {"reg_strength": 0.02}},
        ],
        "georgiev_esm2_LLR": [
            {"n_train_min": 0, "n_train_max": 2500, "params": {"reg_strength": 0.1}},
            {"n_train_min": 2500, "n_train_max": None, "params": {"reg_strength": 2.0}},
        ],
    },
    "xgboost": {
        "esm2_seq_embeddings_LLR": [
            {
                "n_train_min": 0,
                "n_train_max": 5000,
                "params": {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 500},
            },
            {
                "n_train_min": 5000,
                "n_train_max": None,
                "params": {"learning_rate": 0.09, "max_depth": 5, "n_estimators": 500},
            },
        ],
        "onehot_esm2_LLR": [
            {
                "n_train_min": 0,
                "n_train_max": None,
                "params": {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 600},
            },
        ],
        "georgiev_esm2_LLR": [
            {
                "n_train_min": 0,
                "n_train_max": None,
                "params": {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 600},
            },
        ],
    },
}


def get_feature_ntrain_param_preset(
    *,
    model_name: str,
    task_type: str,
    feature_label: str,
    n_train: int,
    presets_override: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
) -> Dict[str, Any]:
    key = _canonical_model_key(model_name)
    label = str(feature_label).strip()
    table = presets_override if presets_override is not None else FEATURE_NTRAIN_PARAM_PRESETS
    model_rules = table.get(key, {})
    rules = model_rules.get(label, [])
    n_train_val = int(n_train)
    for rule in rules:
        lo = int(rule.get("n_train_min", 0))
        hi_raw = rule.get("n_train_max", None)
        hi = None if hi_raw is None else int(hi_raw)
        if n_train_val < lo:
            continue
        if hi is not None and n_train_val >= hi:
            continue
        params = dict(rule.get("params", {}))
        return adapt_tuned_params_for_task(
            model_name=key,
            task_type=task_type,
            tuned_params=params,
        )
    return {}


def get_feature_param_preset_candidates(
    *,
    model_name: str,
    task_type: str,
    feature_label: str,
    presets_override: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
) -> List[Dict[str, Any]]:
    key = _canonical_model_key(model_name)
    label = str(feature_label).strip()
    table = presets_override if presets_override is not None else FEATURE_NTRAIN_PARAM_PRESETS
    model_rules = table.get(key, {})
    rules = model_rules.get(label, [])

    out: List[Dict[str, Any]] = []
    seen = set()
    for rule in rules:
        params = dict(rule.get("params", {}))
        if not params:
            continue
        effective = adapt_tuned_params_for_task(
            model_name=key,
            task_type=task_type,
            tuned_params=params,
        )
        signature = json.dumps(effective, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(effective)
    return out


HYPERPARAMETER_TUNING_SPACE: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "ridge": {
        "shared": {
            # Canonical regularization knob. Mapped to alpha (regression) or 1/C (classification).
            # "reg_strength": {"type": "float", "low": 1e-3, "high": 1e3, "log": True}
            "reg_strength": {"type": "float", "low": 1e-2, "high": 1e3, "log": True}
        }
    },
    "lasso": {
        "shared": {
            # Canonical regularization knob. Mapped to alpha (regression) or 1/C (classification).
            "reg_strength": {"type": "float", "low": 1e-4, "high": 1e2, "log": True}
        }
    },
    "rf": {
        "shared": {
            "n_estimators": {"type": "int", "low": 200, "high": 1000, "step": 100},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.3, 0.5]},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        }
    },
    "xgboost": {
        "shared": {
            # "learning_rate": {"type": "float", "low": 5e-3, "high": 2e-1, "log": True},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 0.1, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 6},
            "n_estimators": {"type": "int", "low": 300, "high": 600, "step": 100},
        }
    },
    "lightgbm": {
        "shared": {
            "learning_rate": {"type": "float", "low": 1e-3, "high": 2e-1, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 255},
            "n_estimators": {"type": "int", "low": 200, "high": 1500, "step": 100},
        }
    },
    "mlp_sklearn": {
        "shared": {
            "hidden_layer_sizes": {"type": "categorical", "choices": [(32,), (64,), (128,), (64, 64)]},
            "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
        }
    },
    "mlp_pytorch": {
        "shared": {
            "hidden_size": {"type": "int", "low": 16, "high": 256, "step": 16},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
        }
    },
}


def get_tuning_space(*, model_name: str, task_type: str) -> Dict[str, Dict[str, Any]]:
    key = _canonical_model_key(model_name)
    record = HYPERPARAMETER_TUNING_SPACE.get(key, {})
    shared = dict(record.get("shared", {}))
    task = str(task_type).strip().lower()
    task_specific = dict(record.get(task, {}))
    shared.update(task_specific)
    return shared


def adapt_tuned_params_for_task(
    *,
    model_name: str,
    task_type: str,
    tuned_params: Dict[str, Any],
) -> Dict[str, Any]:
    key = _canonical_model_key(model_name)
    task = str(task_type).strip().lower()
    out = dict(tuned_params)

    if key in {"ridge", "lasso"} and "reg_strength" in out:
        reg_strength = float(out.pop("reg_strength"))
        reg_strength = max(reg_strength, 1e-12)
        if task == "classification":
            out["C"] = 1.0 / reg_strength
        else:
            out["alpha"] = reg_strength

    return out


def build_model(
    *,
    model_name: str,
    task_type: str,
    params: Dict[str, Any],
):
    key = _canonical_model_key(model_name)
    task = str(task_type).strip().lower()
    if task not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    sklearn_models = _import_sklearn_models()
    if key in sklearn_models:
        cls_cls, reg_cls = sklearn_models[key]
        cls = cls_cls if task == "classification" else reg_cls
        return cls(**params)

    if key == "xgboost":
        try:
            cls_cls, reg_cls = _import_xgboost_models()
        except Exception as exc:
            raise ImportError("xgboost is not installed but model 'xgboost' was requested.") from exc
        cls = cls_cls if task == "classification" else reg_cls
        return cls(**params)

    if key == "lightgbm":
        try:
            cls_cls, reg_cls = _import_lightgbm_models()
        except Exception as exc:
            raise ImportError("lightgbm is not installed but model 'lightgbm' was requested.") from exc
        cls = cls_cls if task == "classification" else reg_cls
        return cls(**params)

    if key == "mlp_pytorch":
        try:
            from abcode.tools.ml.torch_mlp import TorchMLPClassifier, TorchMLPRegressor
        except Exception as exc:
            raise ImportError("PyTorch MLP requested but torch is not installed.") from exc
        cls = TorchMLPClassifier if task == "classification" else TorchMLPRegressor
        return cls(**params)

    raise ValueError(
        "Unsupported model "
        f"'{model_name}'. Supported: rf, random_forest, ridge, lasso, mlp_sklearn, mlp_pytorch, xgboost, lightgbm"
    )
