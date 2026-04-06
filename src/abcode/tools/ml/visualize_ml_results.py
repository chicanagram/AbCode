from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


MODEL_TO_MARKER_DICT: Dict[str, str] = {
    "ridge": "o",
    "xgboost": "x",
    "mlp_pytorch": "*",
    "mlp_sklearn": "s",
    "rf": "^",
    "lightgbm": "D",
    "lasso": "v",
    "best": "P",
}


def default_visualization_inputs() -> Dict[str, Any]:
    return {
        "user_outputs": {},  # train/eval step outputs dict containing summary_csv_path
        "metrics_summary_csv_path": "",  # optional full path override
        "metrics_summary_fname": "",  # optional filename override (used with data_root_dir)
        "data_root_dir": "",
        "data_subfolder": "",
        "model_name_list": ["best"],  # e.g. ["xgboost", "ridge", "best"]
        "metric_col": "test_spearman",  # str or list[str]
        "target_col_list": [],  # if empty, iterate all target_col values in data
        "split_type_list": [],  # if empty, iterate all split types in data
        "feature_label_list": [],  # optional filters
        "show_model_type": False,  # if True, marker style follows MODEL_TO_MARKER_DICT
        "higher_is_better": None,  # auto if None
        "save_figure": True,
        "show_figure": True,
        "figure_output_dir": "",  # default: metrics csv parent dir
        "figure_fname": "",  # default auto-generated
        "figure_dpi": 200,
        "figsize": (8, 5),
        "x_limits": None,  # e.g. [0, 10000]; None/[] -> matplotlib default
        "y_limits": None,  # e.g. [0.0, 1.0]; None/[] -> matplotlib default
    }


def _as_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        s = values.strip()
        return [s] if s else []
    return [str(v).strip() for v in values if str(v).strip()]


def _marker_for_model(model_name: str, show_model_type: bool) -> str:
    if not show_model_type:
        return "o"
    return MODEL_TO_MARKER_DICT.get(str(model_name).strip().lower(), "o")


def _metric_higher_is_better(metric_col: str, override: Any) -> bool:
    if override is not None:
        return bool(override)
    m = str(metric_col).strip().lower()
    lower_is_better_tokens = ("loss", "error", "mae", "mse", "rmse", "logloss")
    return not any(tok in m for tok in lower_is_better_tokens)


def _parse_axis_limits(value: Any, axis_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        raise ValueError(
            f"{axis_name} string input is not supported. Use list/tuple [min, max] or None."
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) != 2:
            raise ValueError(f"{axis_name} must have exactly two values: [min, max].")
        vmin, vmax = value
        if vmin is None or vmax is None:
            return None
        return (float(vmin), float(vmax))
    raise ValueError(f"{axis_name} must be None, [] or a list/tuple with two numeric values.")


def _resolve_metrics_summary_path(inputs: Dict[str, Any]) -> Path:
    direct = str(inputs.get("metrics_summary_csv_path", "")).strip()
    if direct:
        p = Path(direct).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"metrics summary CSV not found: {p}")
        return p

    user_outputs = dict(inputs.get("user_outputs", {}) or {})
    from_outputs = str(user_outputs.get("summary_csv_path", "")).strip()
    if from_outputs:
        p = Path(from_outputs).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"summary_csv_path in user_outputs not found: {p}")
        return p

    fname = str(inputs.get("metrics_summary_fname", "")).strip()
    data_root_dir = str(inputs.get("data_root_dir", "")).strip()
    if fname and data_root_dir:
        data_subfolder = str(inputs.get("data_subfolder", "")).strip().strip("/")
        p = Path(data_root_dir).expanduser().resolve() / "ml" / "output"
        if data_subfolder:
            p = p / data_subfolder
        p = p / fname
        if not p.exists():
            raise FileNotFoundError(f"metrics summary CSV not found: {p}")
        return p

    raise ValueError(
        "Unable to resolve metrics summary CSV path. Provide one of: "
        "metrics_summary_csv_path, user_outputs['summary_csv_path'], or "
        "metrics_summary_fname with data_root_dir."
    )


def _apply_optional_filters(df: pd.DataFrame, inputs: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    feature_labels = _as_list(inputs.get("feature_label_list", []))

    if feature_labels:
        out = out[out["feature_label"].astype(str).isin(feature_labels)]
    return out


def _select_best_rows(df: pd.DataFrame, metric_col: str, higher_is_better: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if metric_col not in df.columns:
        raise KeyError(f"Metric column '{metric_col}' not found in summary dataframe.")

    group_cols = ["split_type", "feature_label", "eval_group", "n_test_pooled", "n_train"]
    if "target_col" in df.columns:
        group_cols = ["target_col"] + group_cols

    ranked = df.sort_values(by=metric_col, ascending=not higher_is_better)
    best = ranked.groupby(group_cols, as_index=False).head(1).copy()
    best["winner_model_name"] = best["model_name"].astype(str)
    best["plot_model_name"] = "best"
    return best


def _build_plot_df(
    df: pd.DataFrame,
    model_name_list: Iterable[str],
    metric_col: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    selected = _as_list(model_name_list)
    if not selected:
        raise ValueError("model_name_list cannot be empty.")

    frames: List[pd.DataFrame] = []
    for name in selected:
        key = str(name).strip().lower()
        if key == "best":
            best_df = _select_best_rows(df, metric_col=metric_col, higher_is_better=higher_is_better)
            if not best_df.empty:
                frames.append(best_df)
            continue
        part = df[df["model_name"].astype(str).str.lower() == key].copy()
        if part.empty:
            continue
        part["winner_model_name"] = part["model_name"].astype(str)
        part["plot_model_name"] = str(name)
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["plot_model_name"])
    return pd.concat(frames, axis=0, ignore_index=True)


def _default_figure_name(metric_col: str, model_name_list: Iterable[str]) -> str:
    model_tag = "_".join(_as_list(model_name_list)).replace(" ", "")
    return f"data_efficiency_{metric_col}_{model_tag}.png"


def _build_figure_name(
    *,
    metric_col: str,
    target_col: str,
    split_type: str,
    model_name_list: Iterable[str],
    figure_fname: str,
    n_total: int,
) -> str:
    if not figure_fname:
        return _default_figure_name(
            metric_col=f"{metric_col}_{target_col}_{split_type}",
            model_name_list=model_name_list,
        )
    if n_total <= 1:
        return figure_fname
    p = Path(figure_fname)
    return f"{p.stem}__{metric_col}__{target_col}__{split_type}{p.suffix or '.png'}"


def _plot_data_efficiency(
    df_plot: pd.DataFrame,
    metric_col: str,
    target_col: str,
    split_type: str,
    out_path: Path | None,
    show_figure: bool,
    figsize: tuple[float, float],
    dpi: int,
    feature_label_order: list[str] | None,
    show_model_type: bool,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
) -> None:
    if df_plot.empty:
        raise ValueError("No rows available after filtering/model selection.")
    if metric_col not in df_plot.columns:
        raise KeyError(f"Metric column '{metric_col}' not found in plotting dataframe.")

    plot_df = df_plot.copy()
    plot_df["feature_label_str"] = plot_df["feature_label"].astype(str)
    plot_df["plot_model_name_str"] = plot_df["plot_model_name"].astype(str)
    plot_df["winner_model_name_str"] = plot_df.get("winner_model_name", plot_df["plot_model_name"]).astype(str)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    model_order = [str(x) for x in pd.unique(plot_df["plot_model_name_str"])]
    feature_seen = [str(x) for x in pd.unique(plot_df["feature_label_str"])]
    if feature_label_order:
        desired = [str(x) for x in feature_label_order]
        feature_order = [f for f in desired if f in feature_seen] + [f for f in feature_seen if f not in desired]
    else:
        feature_order = feature_seen

    grouped = {
        (feature_label, model_name): grp
        for (feature_label, model_name), grp in plot_df.groupby(["feature_label_str", "plot_model_name_str"], sort=False)
    }

    for feature_label in feature_order:
        for model_name in model_order:
            grp = grouped.get((feature_label, model_name))
            if grp is None:
                continue
            if grp.empty:
                continue
            # Aggregate same n_train values to a single point for a cleaner data-efficiency line.
            agg = grp.groupby("n_train", as_index=False)[metric_col].mean().sort_values("n_train")
            label = f"{feature_label} | {model_name}"
            if show_model_type and str(model_name).strip().lower() == "best":
                # Draw one line for the "best" curve, and marker per-point based on winning base model.
                ax.plot(agg["n_train"], agg[metric_col], marker=None, label=label, alpha=0.8)
                point_model = (
                    grp.groupby("n_train")["winner_model_name_str"]
                    .agg(lambda s: str(s.mode().iloc[0]) if not s.mode().empty else str(s.iloc[0]))
                    .reindex(agg["n_train"])
                    .to_numpy()
                )
                for x, y, mname in zip(agg["n_train"], agg[metric_col], point_model):
                    marker = _marker_for_model(str(mname), show_model_type=True)
                    ax.plot([x], [y], marker=marker, linestyle="None", color=ax.lines[-1].get_color(), alpha=0.8)
            else:
                marker = _marker_for_model(str(model_name), show_model_type=show_model_type)
                ax.plot(agg["n_train"], agg[metric_col], marker=marker, label=label, alpha=0.8)

    ax.set_xlabel("n_train")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} vs n_train | target_col={target_col} | split_type={split_type}")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.3)
    ax.legend(title="feature | model", frameon=False)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def run_visualize_ml_results(inputs: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(default_visualization_inputs())
    cfg.update(dict(inputs or {}))

    summary_path = _resolve_metrics_summary_path(cfg)
    summary_df = pd.read_csv(summary_path)
    filtered_df = _apply_optional_filters(summary_df, cfg)
    if filtered_df.empty:
        raise ValueError("No rows remain after feature-label filtering.")

    metric_list = _as_list(cfg.get("metric_col", "test_spearman"))
    if not metric_list:
        raise ValueError("metric_col must be a non-empty string or list of strings.")
    for m in metric_list:
        if m not in filtered_df.columns:
            raise KeyError(f"Metric column '{m}' not found in summary dataframe.")

    split_type_list = _as_list(cfg.get("split_type_list", []))
    if split_type_list:
        split_values = split_type_list
    else:
        split_values = sorted(filtered_df["split_type"].astype(str).dropna().unique().tolist())
    if not split_values:
        raise ValueError("No split_type values available to plot.")
    target_col_list = _as_list(cfg.get("target_col_list", []))
    if target_col_list:
        target_values = target_col_list
    else:
        if "target_col" in filtered_df.columns:
            target_values = sorted(filtered_df["target_col"].astype(str).dropna().unique().tolist())
        else:
            target_values = ["all_targets"]
    if not target_values:
        raise ValueError("No target_col values available to plot.")

    save_figure = bool(cfg.get("save_figure", True))
    show_figure = bool(cfg.get("show_figure", True))
    fig_dir_raw = str(cfg.get("figure_output_dir", "")).strip()
    fig_name_base = str(cfg.get("figure_fname", "")).strip()
    fig_dir = Path(fig_dir_raw).expanduser().resolve() if fig_dir_raw else summary_path.parent
    x_limits = _parse_axis_limits(cfg.get("x_limits", None), "x_limits")
    y_limits = _parse_axis_limits(cfg.get("y_limits", None), "y_limits")

    planned = [
        (metric_col, split_type, target_col)
        for metric_col in metric_list
        for split_type in split_values
        for target_col in target_values
    ]
    figure_paths: List[str] = []
    plotted_pairs: List[Dict[str, Any]] = []
    skipped_pairs: List[Dict[str, Any]] = []
    total_plot_rows = 0

    for metric_col, split_type, target_col in planned:
        split_df = filtered_df[filtered_df["split_type"].astype(str) == str(split_type)].copy()
        if "target_col" in split_df.columns and target_col != "all_targets":
            split_df = split_df[split_df["target_col"].astype(str) == str(target_col)].copy()
        if split_df.empty:
            skipped_pairs.append(
                {
                    "metric_col": metric_col,
                    "split_type": str(split_type),
                    "target_col": str(target_col),
                    "reason": "no_rows_for_split_or_target",
                }
            )
            continue

        higher_is_better = _metric_higher_is_better(metric_col=metric_col, override=cfg.get("higher_is_better"))
        plot_df = _build_plot_df(
            split_df,
            model_name_list=cfg.get("model_name_list", ["best"]),
            metric_col=metric_col,
            higher_is_better=higher_is_better,
        )
        if plot_df.empty:
            skipped_pairs.append(
                {
                    "metric_col": metric_col,
                    "split_type": str(split_type),
                    "target_col": str(target_col),
                    "reason": "no_rows_for_model_selection",
                }
            )
            continue

        total_plot_rows += int(len(plot_df))
        output_name = _build_figure_name(
            metric_col=metric_col,
            target_col=str(target_col),
            split_type=str(split_type),
            model_name_list=cfg.get("model_name_list", ["best"]),
            figure_fname=fig_name_base,
            n_total=len(planned),
        )
        out_path = (fig_dir / output_name) if save_figure else None
        _plot_data_efficiency(
            df_plot=plot_df,
            metric_col=metric_col,
            target_col=str(target_col),
            split_type=str(split_type),
            out_path=out_path,
            show_figure=show_figure,
            figsize=tuple(cfg.get("figsize", (8, 5))),
            dpi=int(cfg.get("figure_dpi", 200)),
            feature_label_order=_as_list(cfg.get("feature_label_list", [])),
            show_model_type=bool(cfg.get("show_model_type", False)),
            x_limits=x_limits,
            y_limits=y_limits,
        )
        if out_path is not None:
            figure_paths.append(str(out_path))
        plotted_pairs.append(
            {
                "metric_col": metric_col,
                "split_type": str(split_type),
                "target_col": str(target_col),
                "n_rows": int(len(plot_df)),
                "n_lines": int(plot_df[["feature_label", "plot_model_name"]].drop_duplicates().shape[0]),
            }
        )

    return {
        "status": "ok",
        "summary_csv_path": str(summary_path),
        "figure_paths": figure_paths,
        "metric_col_list": metric_list,
        "target_col_list": target_values,
        "split_type_list": split_values,
        "n_summary_rows": int(len(summary_df)),
        "n_filtered_rows": int(len(filtered_df)),
        "n_plot_rows_total": int(total_plot_rows),
        "plotted_pairs": plotted_pairs,
        "skipped_pairs": skipped_pairs,
    }
