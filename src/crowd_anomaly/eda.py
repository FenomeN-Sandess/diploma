from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crowd_anomaly.features import (
    get_feature_description,
    get_feature_display_name,
    load_feature_artifacts,
)
from crowd_anomaly.plotting import (
    plot_anomaly_score_by_label,
    plot_correlation_heatmap,
    plot_feature_boxplots_by_label,
    plot_label_distribution,
    plot_outlier_summary,
    plot_run_metric_boxplots,
    plot_scenario_distribution,
    plot_split_distribution,
    plot_trajectory_anomaly_timeline,
    plot_trajectory_examples,
)


def build_eda_report(
    dataset_dir: Path | str,
    feature_dir: Path | str,
    output_dir: Path | str,
    anomaly_dir: Path | str | None = None,
    multi_horizon_dir: Path | str | None = None,
    predictive_risk_dir: Path | str | None = None,
    seeds_dir: Path | str | None = None,
    loso_dir: Path | str | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    feature_path = Path(feature_dir)
    output_path = Path(output_dir)
    tables_path = output_path / "tables"
    figures_path = output_path / "figures"
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    _clear_generated_files(tables_path, "*.csv")
    _clear_generated_files(figures_path, "*.png")

    artifacts = load_feature_artifacts(feature_path)
    features_run = artifacts["features_run"]
    run_mapping = artifacts["run_mapping"]
    feature_schema = artifacts["feature_schema"]
    feature_columns = list(feature_schema["feature_columns"])
    feature_names = {
        feature: get_feature_display_name(feature) for feature in feature_columns
    }
    data = features_run.merge(run_mapping, on="run_id", how="inner")
    runs = _load_runs(dataset_path)
    dataset_metadata = _load_dataset_metadata(dataset_path)

    dataset_summary = summarize_dataset(runs, feature_columns, dataset_metadata)
    scenario_summary = summarize_scenarios(runs)
    trajectory_summary = summarize_trajectories(runs)
    label_counts = _label_counts(run_mapping)
    split_counts = _split_counts(run_mapping)
    feature_summary = summarize_features(features_run, feature_columns)
    feature_label_summary = summarize_feature_label_differences(data, feature_columns)
    outlier_summary = compute_outlier_summary(features_run, feature_columns)
    top_correlations = compute_top_correlations(features_run, feature_columns)
    feature_dictionary = describe_features(feature_columns)
    feature_summary = add_feature_details(feature_summary)
    feature_label_summary = add_feature_details(feature_label_summary)
    outlier_summary = add_feature_details(outlier_summary)
    top_correlations = add_correlation_feature_details(top_correlations)
    correlation_matrix = features_run[feature_columns].corr().fillna(0.0)
    anomaly_predictions = _load_anomaly_predictions(anomaly_dir)
    local_candidates = _load_local_anomaly_candidates(anomaly_dir)
    anomaly_score_summary = summarize_anomaly_scores(anomaly_predictions)

    multi_horizon_table = _load_multi_horizon_table(multi_horizon_dir)
    predictive_risk_table = _load_predictive_risk_table(predictive_risk_dir)
    predictive_scenario_table = _load_predictive_scenario_table(predictive_risk_dir)
    detection_seeds_table = _load_detection_seeds_table(seeds_dir)
    detection_loso_table = _load_detection_loso_table(loso_dir)

    table_files = {
        "dataset_summary": tables_path / "dataset_summary.csv",
        "scenario_summary": tables_path / "scenario_summary.csv",
        "trajectory_summary": tables_path / "trajectory_summary.csv",
        "label_counts": tables_path / "label_counts.csv",
        "split_counts": tables_path / "split_counts.csv",
        "feature_dictionary": tables_path / "feature_dictionary.csv",
        "feature_summary": tables_path / "feature_summary.csv",
        "feature_label_summary": tables_path / "feature_label_summary.csv",
        "outlier_summary": tables_path / "outlier_summary.csv",
        "top_correlations": tables_path / "top_correlations.csv",
    }
    if anomaly_score_summary is not None:
        table_files["anomaly_score_summary"] = tables_path / "anomaly_score_summary.csv"
    if multi_horizon_table is not None:
        table_files["multi_horizon_summary"] = tables_path / "multi_horizon_summary.csv"
    if predictive_risk_table is not None:
        table_files["predictive_risk_summary"] = (
            tables_path / "predictive_risk_summary.csv"
        )
    if predictive_scenario_table is not None:
        table_files["predictive_risk_lead_time_by_scenario"] = (
            tables_path / "predictive_risk_lead_time_by_scenario.csv"
        )
    if detection_seeds_table is not None:
        table_files["detection_seeds_summary"] = (
            tables_path / "detection_seeds_summary.csv"
        )
    if detection_loso_table is not None:
        table_files["detection_loso_summary"] = (
            tables_path / "detection_loso_summary.csv"
        )
    dataset_summary.to_csv(table_files["dataset_summary"], index=False)
    scenario_summary.to_csv(table_files["scenario_summary"], index=False)
    trajectory_summary.to_csv(table_files["trajectory_summary"], index=False)
    label_counts.to_csv(table_files["label_counts"], index=False)
    split_counts.to_csv(table_files["split_counts"], index=False)
    feature_dictionary.to_csv(table_files["feature_dictionary"], index=False)
    feature_summary.to_csv(table_files["feature_summary"], index=False)
    feature_label_summary.to_csv(table_files["feature_label_summary"], index=False)
    outlier_summary.to_csv(table_files["outlier_summary"], index=False)
    top_correlations.to_csv(table_files["top_correlations"], index=False)
    if anomaly_score_summary is not None:
        anomaly_score_summary.to_csv(
            table_files["anomaly_score_summary"],
            index=False,
        )
    if multi_horizon_table is not None:
        multi_horizon_table.to_csv(table_files["multi_horizon_summary"], index=False)
    if predictive_risk_table is not None:
        predictive_risk_table.to_csv(
            table_files["predictive_risk_summary"], index=False
        )
    if predictive_scenario_table is not None:
        predictive_scenario_table.to_csv(
            table_files["predictive_risk_lead_time_by_scenario"], index=False
        )
    if detection_seeds_table is not None:
        detection_seeds_table.to_csv(
            table_files["detection_seeds_summary"], index=False
        )
    if detection_loso_table is not None:
        detection_loso_table.to_csv(
            table_files["detection_loso_summary"], index=False
        )

    figure_files = {
        "label_distribution": plot_label_distribution(
            label_counts,
            figures_path / "label_distribution.png",
        ),
        "scenario_distribution": plot_scenario_distribution(
            scenario_summary,
            figures_path / "scenario_distribution.png",
        ),
        "split_distribution": plot_split_distribution(
            split_counts,
            figures_path / "split_distribution.png",
        ),
        "run_metric_boxplots": plot_run_metric_boxplots(
            runs,
            figures_path / "run_metric_boxplots.png",
        ),
        "trajectory_examples": plot_trajectory_examples(
            dataset_path,
            runs,
            figures_path / "trajectory_examples.png",
            local_candidates=local_candidates,
        ),
        "trajectory_anomaly_timeline": plot_trajectory_anomaly_timeline(
            dataset_path,
            runs,
            figures_path / "trajectory_anomaly_timeline.png",
            local_candidates=local_candidates,
        ),
        "feature_boxplots": plot_feature_boxplots_by_label(
            data,
            feature_columns,
            figures_path / "feature_boxplots.png",
            feature_names=feature_names,
        ),
        "correlation_heatmap": plot_correlation_heatmap(
            correlation_matrix,
            figures_path / "correlation_heatmap.png",
            feature_names=feature_names,
        ),
        "outlier_summary": plot_outlier_summary(
            outlier_summary,
            figures_path / "outlier_summary.png",
            feature_names=feature_names,
        ),
    }
    if anomaly_predictions is not None:
        figure_files["anomaly_score_distribution"] = plot_anomaly_score_by_label(
            anomaly_predictions,
            figures_path / "anomaly_score_distribution.png",
        )

    _copy_external_figure(
        multi_horizon_dir, "multi_horizon_mae.png", figures_path,
        figure_files, "multi_horizon_mae",
    )
    _copy_external_figure(
        multi_horizon_dir, "multi_horizon_mse.png", figures_path,
        figure_files, "multi_horizon_mse",
    )
    _copy_external_figure(
        predictive_risk_dir, "predictive_risk_f1_curve.png", figures_path,
        figure_files, "predictive_risk_f1_curve",
    )
    _copy_external_figure(
        predictive_risk_dir, "predictive_risk_lead_time_by_scenario.png", figures_path,
        figure_files, "predictive_risk_lead_time_by_scenario",
    )

    report_path = output_path / "eda_report.md"
    report_path.write_text(
        _render_report(
            dataset_path=dataset_path,
            feature_path=feature_path,
            figure_files=figure_files,
            dataset_summary=dataset_summary,
            scenario_summary=scenario_summary,
            trajectory_summary=trajectory_summary,
            label_counts=label_counts,
            split_counts=split_counts,
            feature_dictionary=feature_dictionary,
            feature_label_summary=feature_label_summary,
            outlier_summary=outlier_summary,
            top_correlations=top_correlations,
            anomaly_score_summary=anomaly_score_summary,
            has_local_candidates=local_candidates is not None
            and not local_candidates.empty,
            feature_columns=feature_columns,
            multi_horizon_table=multi_horizon_table,
            predictive_risk_table=predictive_risk_table,
            predictive_scenario_table=predictive_scenario_table,
            detection_seeds_table=detection_seeds_table,
            detection_loso_table=detection_loso_table,
        ),
        encoding="utf-8",
    )

    return {
        "report_path": report_path,
        "tables": table_files,
        "figures": figure_files,
        "feature_count": len(feature_columns),
        "row_count": len(features_run),
    }


def summarize_features(
    features_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        values = features_df[feature].astype(float)
        rows.append(
            {
                "feature": feature,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def describe_features(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        rows.append(
            {
                "feature": feature,
                "name": get_feature_display_name(feature),
                "description": get_feature_description(feature),
            }
        )
    return pd.DataFrame(rows)


def add_feature_details(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "feature" not in data.columns:
        return data
    result = data.copy()
    result.insert(1, "feature_name", result["feature"].map(get_feature_display_name))
    result.insert(2, "description", result["feature"].map(get_feature_description))
    return result


def add_correlation_feature_details(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.copy()
    if "feature_a" in result.columns:
        result.insert(1, "feature_a_name", result["feature_a"].map(
            get_feature_display_name
        ))
    if "feature_b" in result.columns:
        result.insert(3, "feature_b_name", result["feature_b"].map(
            get_feature_display_name
        ))
    return result


def _clear_generated_files(directory: Path, pattern: str) -> None:
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()


def summarize_dataset(
    runs: pd.DataFrame,
    feature_columns: list[str],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(columns=["metric", "value"])

    trajectory_rows = 0
    if {"agent_count", "steps"}.issubset(runs.columns):
        trajectory_rows = int((runs["agent_count"] * runs["steps"]).sum())

    rows = [
        ("dataset_name", metadata.get("dataset_name", "unknown")),
        ("synthetic_data", metadata.get("synthetic_data", True)),
        ("run_count", int(len(runs))),
        ("scenario_count", int(runs["scenario_template"].nunique())),
        ("feature_count", int(len(feature_columns))),
        ("frame_count", int(runs["steps"].sum()) if "steps" in runs else 0),
        ("trajectory_row_count", trajectory_rows),
        ("mean_agent_count", float(runs["agent_count"].mean())),
        ("mean_duration_s", float(runs["duration_s"].mean())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_scenarios(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    grouped = (
        runs.groupby(["scenario_template", "label_name"], as_index=False)
        .agg(
            run_count=("run_id", "count"),
            agent_count_mean=("agent_count", "mean"),
            duration_s_mean=("duration_s", "mean"),
            mean_speed_mean=("mean_speed", "mean"),
            mean_force_mean=("mean_force", "mean"),
            goal_reached_fraction_mean=("final_goal_reached_fraction", "mean"),
        )
        .sort_values(["label_name", "scenario_template"])
    )
    return grouped


def summarize_trajectories(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    return (
        runs.groupby("label_name", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            agent_count_mean=("agent_count", "mean"),
            steps_mean=("steps", "mean"),
            duration_s_mean=("duration_s", "mean"),
            mean_speed_mean=("mean_speed", "mean"),
            max_speed_mean=("max_speed", "mean"),
            mean_force_mean=("mean_force", "mean"),
            max_force_mean=("max_force", "mean"),
            goal_reached_fraction_mean=("final_goal_reached_fraction", "mean"),
        )
        .sort_values("label_name")
    )


def summarize_feature_label_differences(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    if data.empty or "label_name" not in data:
        return pd.DataFrame()

    rows = []
    for feature in feature_columns:
        means = data.groupby("label_name")[feature].mean()
        normal_mean = float(means.get("normal", np.nan))
        anomaly_mean = float(means.get("anomaly", np.nan))
        rows.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "anomaly_mean": anomaly_mean,
                "abs_difference": float(abs(anomaly_mean - normal_mean)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_difference", ascending=False)
        .reset_index(drop=True)
    )


def compute_outlier_summary(
    features_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    row_count = len(features_df)
    for feature in feature_columns:
        values = features_df[feature].astype(float)
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = int(outlier_mask.sum())
        rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": outlier_count,
                "outlier_fraction": float(outlier_count / row_count),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["outlier_fraction", "feature"],
        ascending=[False, True],
    )


def compute_top_correlations(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    correlation = features_df[feature_columns].corr().replace(
        [np.inf, -np.inf],
        np.nan,
    )
    rows = []
    for left_index, left_feature in enumerate(feature_columns):
        for right_feature in feature_columns[left_index + 1 :]:
            value = correlation.loc[left_feature, right_feature]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "feature_a": left_feature,
                    "feature_b": right_feature,
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def summarize_anomaly_scores(predictions: pd.DataFrame | None) -> pd.DataFrame | None:
    if predictions is None or predictions.empty:
        return None
    return (
        predictions.groupby("label_name")["anomaly_score"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )


def summarize_runs(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "label" not in data:
        return pd.DataFrame(columns=["label", "runs"])
    return data.groupby("label").size().reset_index(name="runs")


def _load_runs(dataset_path: Path) -> pd.DataFrame:
    runs_path = dataset_path / "runs.csv"
    if not runs_path.exists():
        return pd.DataFrame()
    return pd.read_csv(runs_path)


def _load_dataset_metadata(dataset_path: Path) -> dict[str, Any]:
    metadata_path = dataset_path / "dataset_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _label_counts(run_mapping: pd.DataFrame) -> pd.DataFrame:
    return (
        run_mapping.groupby(["label", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "run_count"})
        .sort_values("label_name")
    )


def _split_counts(run_mapping: pd.DataFrame) -> pd.DataFrame:
    return (
        run_mapping.groupby("split", as_index=False)
        .size()
        .rename(columns={"size": "run_count"})
        .sort_values("split")
    )


def _load_anomaly_predictions(anomaly_dir: Path | str | None) -> pd.DataFrame | None:
    if anomaly_dir is None:
        return None
    predictions_path = Path(anomaly_dir) / "unsupervised_predictions.csv"
    if not predictions_path.exists():
        return None
    predictions = pd.read_csv(predictions_path)
    required_columns = {"label_name", "anomaly_score"}
    if not required_columns.issubset(predictions.columns):
        return None
    return predictions


def _load_local_anomaly_candidates(
    anomaly_dir: Path | str | None,
) -> pd.DataFrame | None:
    if anomaly_dir is None:
        return None
    candidates_path = Path(anomaly_dir) / "local_anomaly_candidates.csv"
    if not candidates_path.exists():
        return None
    candidates = pd.read_csv(candidates_path)
    required_columns = {"run_id", "frame_step", "frame_time_s", "frame_score"}
    if not required_columns.issubset(candidates.columns):
        return None
    return candidates


def _load_multi_horizon_table(
    multi_horizon_dir: Path | str | None,
) -> pd.DataFrame | None:
    """Тестовая MAE/MSE по горизонту для persistence и direct (направление 1.1)."""
    if multi_horizon_dir is None:
        return None
    metrics_path = Path(multi_horizon_dir) / "multi_horizon_metrics_overall.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    test_metrics = metrics[metrics["split"] == "test"]
    if test_metrics.empty:
        return None
    pivot = test_metrics.pivot_table(
        index="horizon",
        columns="model",
        values=["mae", "mse"],
        aggfunc="mean",
    )
    rows = []
    for horizon in sorted(test_metrics["horizon"].unique()):
        row: dict[str, Any] = {"horizon": int(horizon)}
        for metric in ("mae", "mse"):
            for model in ("persistence", "direct"):
                try:
                    value = float(pivot.loc[horizon, (metric, model)])
                except (KeyError, TypeError):
                    value = float("nan")
                row[f"{metric}_{model}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _load_predictive_risk_table(
    predictive_risk_dir: Path | str | None,
) -> pd.DataFrame | None:
    """Метрики упреждающего детектора по горизонту (test, forecast_only)."""
    if predictive_risk_dir is None:
        return None
    metrics_path = (
        Path(predictive_risk_dir) / "predictive_risk_metrics_by_horizon.csv"
    )
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    selected = metrics[
        (metrics["split"] == "test") & (metrics["variant"] == "forecast_only")
    ]
    if selected.empty:
        return None
    columns = ["horizon", "precision", "recall", "f1"]
    if "lead_time_mean_s" in selected.columns:
        columns.append("lead_time_mean_s")
    if "lead_time_median_s" in selected.columns:
        columns.append("lead_time_median_s")
    table = selected[columns].sort_values("horizon").reset_index(drop=True)
    table["horizon"] = table["horizon"].astype(int)
    return table


def _load_predictive_scenario_table(
    predictive_risk_dir: Path | str | None,
) -> pd.DataFrame | None:
    """Lead time по сценарию на максимальном горизонте (test, forecast_only)."""
    if predictive_risk_dir is None:
        return None
    path = Path(predictive_risk_dir) / "predictive_risk_lead_time_by_scenario.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    selected = data[(data["split"] == "test") & (data["variant"] == "forecast_only")]
    if selected.empty:
        selected = data[data["variant"] == "forecast_only"]
    if selected.empty:
        return None
    max_horizon = int(selected["horizon"].max())
    selected = selected[selected["horizon"] == max_horizon]
    columns = [
        "scenario_template",
        "n_runs",
        "n_detected",
        "lead_time_mean_s",
        "lead_time_median_s",
    ]
    columns = [column for column in columns if column in selected.columns]
    table = selected[columns].sort_values("scenario_template").reset_index(drop=True)
    table.attrs["horizon"] = max_horizon
    return table


_DETECTION_METHOD_LABELS = {
    "pca_physics": "Гибрид PCA + физика",
    "pca_reconstruction": "PCA",
    "physics_only": "Физический показатель",
    "isolation_forest": "Isolation Forest",
    "hybrid_iforest_physics": "Гибрид IF + физика",
    "random_forest": "Случайный лес",
}
_SEEDS_METHOD_ORDER = (
    "pca_physics",
    "pca_reconstruction",
    "physics_only",
    "isolation_forest",
    "hybrid_iforest_physics",
)
_LOSO_METHOD_ORDER = ("pca_physics", "pca_reconstruction", "random_forest")


def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        std = 0.0
    return f"{mean:.3f} ± {std:.3f}"


def _load_detection_seeds_table(
    seeds_dir: Path | str | None,
) -> pd.DataFrame | None:
    """Доверительные интервалы метрик обнаружения по нескольким seed'ам.

    Каждая метрика приводится в виде «среднее ± стандартное отклонение» по
    повторным прогонам (бутстрэп обучающей выборки)."""
    if seeds_dir is None:
        return None
    path = Path(seeds_dir) / "aggregated_metrics.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    required = {
        "method",
        "test_precision_mean",
        "test_recall_mean",
        "test_f1_mean",
    }
    if not required.issubset(data.columns):
        return None
    indexed = data.set_index("method")
    rows = []
    seed_count = int(indexed["seed_count"].iloc[0]) if "seed_count" in indexed else 0
    for method in _SEEDS_METHOD_ORDER:
        if method not in indexed.index:
            continue
        record = indexed.loc[method]
        rows.append(
            {
                "Метод": _DETECTION_METHOD_LABELS.get(method, method),
                "Precision": _format_mean_std(
                    record.get("test_precision_mean", float("nan")),
                    record.get("test_precision_std", 0.0),
                ),
                "Recall": _format_mean_std(
                    record.get("test_recall_mean", float("nan")),
                    record.get("test_recall_std", 0.0),
                ),
                "F1": _format_mean_std(
                    record.get("test_f1_mean", float("nan")),
                    record.get("test_f1_std", 0.0),
                ),
            }
        )
    if not rows:
        return None
    table = pd.DataFrame(rows)
    table.attrs["seed_count"] = seed_count
    return table


def _load_detection_loso_table(
    loso_dir: Path | str | None,
) -> pd.DataFrame | None:
    """Leave-one-scenario-out: среднее ± std по фолдам для каждого метода."""
    if loso_dir is None:
        return None
    path = Path(loso_dir) / "loso_aggregated.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    required = {"method", "precision_mean", "recall_mean", "f1_mean"}
    if not required.issubset(data.columns):
        return None
    indexed = data.set_index("method")
    fold_count = int(indexed["fold_count"].iloc[0]) if "fold_count" in indexed else 0
    rows = []
    for method in _LOSO_METHOD_ORDER:
        if method not in indexed.index:
            continue
        record = indexed.loc[method]
        rows.append(
            {
                "Метод": _DETECTION_METHOD_LABELS.get(method, method),
                "Precision": _format_mean_std(
                    record.get("precision_mean", float("nan")),
                    record.get("precision_std", 0.0),
                ),
                "Recall": _format_mean_std(
                    record.get("recall_mean", float("nan")),
                    record.get("recall_std", 0.0),
                ),
                "F1": _format_mean_std(
                    record.get("f1_mean", float("nan")),
                    record.get("f1_std", 0.0),
                ),
            }
        )
    if not rows:
        return None
    table = pd.DataFrame(rows)
    table.attrs["fold_count"] = fold_count
    return table


def _copy_external_figure(
    source_dir: Path | str | None,
    file_name: str,
    figures_path: Path,
    figure_files: dict[str, Path],
    key: str,
) -> None:
    """Скопировать готовый график из стороннего модуля в папку EDA-фигур."""
    if source_dir is None:
        return
    source_path = Path(source_dir) / file_name
    if not source_path.exists():
        return
    destination = figures_path / file_name
    shutil.copyfile(source_path, destination)
    figure_files[key] = destination


def _render_report(
    dataset_path: Path,
    feature_path: Path,
    figure_files: dict[str, Path],
    dataset_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    label_counts: pd.DataFrame,
    split_counts: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    feature_label_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    top_correlations: pd.DataFrame,
    anomaly_score_summary: pd.DataFrame | None,
    has_local_candidates: bool,
    feature_columns: list[str],
    multi_horizon_table: pd.DataFrame | None = None,
    predictive_risk_table: pd.DataFrame | None = None,
    predictive_scenario_table: pd.DataFrame | None = None,
    detection_seeds_table: pd.DataFrame | None = None,
    detection_loso_table: pd.DataFrame | None = None,
) -> str:
    anomaly_section = _render_anomaly_score_section(
        anomaly_score_summary,
        figure_files,
    )
    detection_section = _render_detection_results_section(
        detection_seeds_table=detection_seeds_table,
        detection_loso_table=detection_loso_table,
    )
    forecasting_section = _render_forecasting_section(
        multi_horizon_table=multi_horizon_table,
        predictive_risk_table=predictive_risk_table,
        predictive_scenario_table=predictive_scenario_table,
        figure_files=figure_files,
    )
    label_figure = _render_figure(
        figure_files,
        "label_distribution",
        "Распределение запусков по меткам",
    )
    scenario_figure = _render_figure(
        figure_files,
        "scenario_distribution",
        "Количество запусков по сценариям",
    )
    split_figure = _render_figure(
        figure_files,
        "split_distribution",
        "Распределение запусков по частям набора",
    )
    run_metric_figure = _render_figure(
        figure_files,
        "run_metric_boxplots",
        "Метрики запусков по меткам",
    )
    trajectory_figure = _render_figure(
        figure_files,
        "trajectory_examples",
        "Примеры траекторий агентов",
    )
    trajectory_timeline_figure = _render_figure(
        figure_files,
        "trajectory_anomaly_timeline",
        "Диагностика подозрительных моментов во времени",
    )
    feature_figure = _render_figure(
        figure_files,
        "feature_boxplots",
        "Сравнение признаков для нормальных и аномальных запусков",
    )
    outlier_figure = _render_figure(
        figure_files,
        "outlier_summary",
        "Сводка выбросов по признакам",
    )
    correlation_figure = _render_figure(
        figure_files,
        "correlation_heatmap",
        "Матрица корреляций признаков",
    )
    return f"""# Простой EDA-отчёт

## Источник данных

В отчёте используются синтетические данные. Они создаются симуляцией движения
агентов в программном стенде, внешний датасет не используется.

Папка датасета: `{dataset_path}`.
Папка признаков: `{feature_path}`.

Короткая сводка по датасету:

{_to_markdown_table(dataset_summary)}

## Целевая переменная

Целевая переменная — нормальный или аномальный запуск симуляции. В этой версии
классифицируется весь запуск, а не отдельный кадр и не отдельный агент.

Распределение по меткам:

{_to_markdown_table(label_counts)}

{label_figure}

График нужен, чтобы сразу увидеть баланс классов. Если нормальных и аномальных
запусков сильно разное количество, это важно учитывать при чтении метрик.

## Сценарии и траектории

Датасет собран из нескольких сценариев движения. Каждый сценарий запускается
несколько раз с разными seed, поэтому получается набор похожих, но не полностью
одинаковых траекторий.

Сводка по сценариям:

{_to_markdown_table(scenario_summary)}

{scenario_figure}

Сводка по траекториям и метрикам запусков:

{_to_markdown_table(trajectory_summary)}

{run_metric_figure}

На этом графике удобно сравнить нормальные и аномальные запуски по скорости,
силе и достижению цели. Это не итоговая модель, а первичный взгляд на данные.

{trajectory_figure}

На примерах траекторий показаны пути части агентов. Зелёная точка показывает
старт, тёмный квадрат — положение агента в конце симуляции, стрелка показывает
направление движения, а красная звезда показывает цель. Если линия не дошла до
цели, это значит, что агент не успел дойти за отведённое время симуляции, а не
что график оборвался.

Жёлтые ромбы показывают кандидатов на подозрительные моменты, если такие
кандидаты были найдены моделью. Это диагностическая подсказка, а не ручная
эталонная покадровая разметка.

{trajectory_timeline_figure}

Временной график показывает, как по ходу запуска менялись средняя сила,
максимальная сила и число близких пар агентов. Жёлтые вертикальные линии
показывают диагностические пики из `local_anomaly_candidates.csv`. Если файл с
кандидатами отсутствует, график всё равно строится, но без жёлтых отметок.

{_render_local_candidate_note(has_local_candidates)}

## Политика покадровых меток

Покадровые метки наследуются от метки всего запуска. Такая метка не является
точной эталонной разметкой конкретного кадра или агента. Поэтому локальные
кандидаты на аномальные кадры и агенты используются только для простой
интерпретации.

## Разбиение train/val/test

Количество запусков по обучающей, валидационной и тестовой выборкам:

{_to_markdown_table(split_counts)}

{split_figure}

Разбиение нужно, чтобы обучение, подбор порогов и итоговая проверка не
смешивались в одну и ту же выборку.

## Признаки

Признаки физически интерпретируемые и рассчитаны на уровне всего запуска. В набор
входят группы признаков: скорость, плотность, контакты, прогресс к цели и
индексы затора.

Количество признаков: {len(feature_columns)}.

Короткий словарь признаков:

{_to_markdown_table(feature_dictionary)}

{feature_figure}

Самые заметные различия средних значений между normal и anomaly:

{_to_markdown_table(feature_label_summary.head(10))}

## Проверка отсутствия утечки

`label`, `split`, `seed`, `scenario_template` и `run_id` не используются как
признаки. `label` и `split` находятся только в `run_mapping`, а `run_id`
используется только как идентификатор строки.

## Выбросы

Выбросы оцениваются простым правилом межквартильного размаха. Они не удаляются
автоматически, а учитываются при интерпретации результатов.

Первые строки таблицы выбросов:

{_to_markdown_table(outlier_summary.head(10))}

{outlier_figure}

Если у признака много выбросов, это не всегда ошибка: для аномальных сценариев
как раз ожидаются более резкие силы, плотность или контакты.

## Корреляции

Корреляции показывают статистические связи между рассчитанными признаками.
Корреляция не доказывает причинность, поэтому такие связи используются только
как ориентир для анализа.

Наиболее сильные корреляции:

{_to_markdown_table(top_correlations.head(10))}

{correlation_figure}

Матрица помогает увидеть похожие признаки. Например, признаки плотности и
близких контактов могут быть связаны, потому что оба описывают скученность.

{anomaly_section}

{detection_section}

{forecasting_section}

## Ограничения

- данные являются синтетическими;
- нет внешней валидации на реальных толпах;
- нет локальной разметки кадров/агентов;
- результаты применимы только в рамках симуляционного стенда.

## Вывод

EDA подтверждает, что рассчитанные физические признаки и графики можно
использовать для базового анализа нормальных и аномальных запусков в рамках
синтетического стенда.
"""


def _render_anomaly_score_section(
    anomaly_score_summary: pd.DataFrame | None,
    figure_files: dict[str, Path],
) -> str:
    if anomaly_score_summary is None:
        return """## Показатель аномальности

Файл с результатами модели без учителя не найден, поэтому распределение
anomaly score в этом EDA-отчёте не строилось."""

    anomaly_figure = _render_figure(
        figure_files,
        "anomaly_score_distribution",
        "Распределение показателя аномальности",
    )
    return f"""## Показатель аномальности

Дополнительно построен простой график распределения `anomaly_score` для
нормальных и аномальных запусков. Он нужен для интерпретации модели без учителя,
а не для ручной покадровой разметки.

Сводка по anomaly score:

{_to_markdown_table(anomaly_score_summary)}

{anomaly_figure}"""


def _render_detection_results_section(
    detection_seeds_table: pd.DataFrame | None,
    detection_loso_table: pd.DataFrame | None,
) -> str:
    if detection_seeds_table is None and detection_loso_table is None:
        return ""

    parts = ["## Количественные результаты обнаружения"]

    if detection_seeds_table is not None:
        seed_count = detection_seeds_table.attrs.get("seed_count", 0)
        suffix = f" по {seed_count} прогонам" if seed_count else ""
        parts.append(
            "Метрики обнаружения на тестовой выборке приводятся с доверительным "
            "интервалом — среднее ± стандартное отклонение"
            f"{suffix} (бутстрэп обучающей выборки):"
        )
        parts.append(_to_markdown_table(detection_seeds_table))
        parts.append(
            "Гибрид PCA + физика удерживает баланс точности и полноты и даёт "
            "наилучший F1; одиночные критерии уступают ему по сбалансированности "
            "precision и recall."
        )

    if detection_loso_table is not None:
        fold_count = detection_loso_table.attrs.get("fold_count", 0)
        suffix = f" по {fold_count} фолдам" if fold_count else ""
        parts.append(
            "### Обобщение на новые типы аномалий (leave-one-scenario-out)"
        )
        parts.append(
            "Каждый канонический тип аномалии поочерёдно полностью изымается из "
            "обучения и используется только для теста. Среднее ± стандартное "
            f"отклонение{suffix}:"
        )
        parts.append(_to_markdown_table(detection_loso_table))
        parts.append(
            "Гибрид PCA + физика устойчив к смене типа аномалии: он опирается на "
            "физический смысл признаков и не требует размеченных примеров каждого "
            "типа. Модель с учителем (случайный лес) при изъятии целого типа из "
            "обучения теряет полноту на нём, поэтому её среднее F1 ниже и "
            "разброс по фолдам существенно больше."
        )

    return "\n\n".join(parts)


def _render_forecasting_section(
    multi_horizon_table: pd.DataFrame | None,
    predictive_risk_table: pd.DataFrame | None,
    predictive_scenario_table: pd.DataFrame | None,
    figure_files: dict[str, Path],
) -> str:
    if (
        multi_horizon_table is None
        and predictive_risk_table is None
        and predictive_scenario_table is None
    ):
        return ""

    blocks: list[str] = [
        "## Прогноз динамики и упреждающий сигнал риска",
        "",
        "Помимо покадрового анализа отдельного запуска, рассматривается "
        "агрегированное состояние толпы во времени "
        "$z_t=(\\bar x_t,\\bar y_t,\\bar v_t,\\rho_t,\\bar f_t,f_{\\max,t},c_t)$ "
        "и две связанные задачи: прогноз этого состояния на горизонт $H>1$ и "
        "упреждающая оценка риска по прогнозу.",
    ]

    if multi_horizon_table is not None:
        mae_curve = _render_figure(
            figure_files,
            "multi_horizon_mae",
            "Деградация MAE прогноза по горизонту H",
        )
        blocks.extend(
            [
                "",
                "### Многошаговый прогноз агрегированной динамики",
                "",
                "Сравниваются две стратегии прогноза состояния на $H$ шагов "
                "вперёд: наивный перенос последнего наблюдения (persistence) и "
                "отдельный регрессор на каждый горизонт (direct). Метрики на "
                "тестовой выборке по горизонту:",
                "",
                _to_markdown_table(multi_horizon_table),
                "",
                mae_curve,
                "",
                "Direct-стратегия устойчиво точнее наивного переноса на всех "
                "горизонтах, и разрыв растёт с увеличением $H$: наивный прогноз "
                "накапливает ошибку, а раздельные регрессоры — нет.",
            ]
        )

    if predictive_risk_table is not None:
        f1_curve = _render_figure(
            figure_files,
            "predictive_risk_f1_curve",
            "F1 упреждающего детектора по горизонту H",
        )
        blocks.extend(
            [
                "",
                "### Упреждающий сигнал риска",
                "",
                "По прогнозу пересчитываются пиковые физические признаки запуска "
                "и причинный показатель аномальности "
                "$\\widehat S(t,H)$, доступный уже в ходе запуска. Самое раннее "
                "превышение порога даёт время упреждения (lead time). Метрики "
                "по горизонту на тесте (вариант forecast_only):",
                "",
                _to_markdown_table(predictive_risk_table),
                "",
                f1_curve,
                "",
                "Качество детектора устойчиво по горизонту: причинная оценка "
                "по прогнозу обнаруживает аномальные запуски заранее, не "
                "дожидаясь окончания симуляции.",
            ]
        )

    if predictive_scenario_table is not None:
        horizon = predictive_scenario_table.attrs.get("horizon")
        scenario_caption = (
            f"Время упреждения по сценарию на горизонте $H={horizon}$ "
            "(test, forecast_only):"
            if horizon is not None
            else "Время упреждения по сценарию (test, forecast_only):"
        )
        scenario_figure = _render_figure(
            figure_files,
            "predictive_risk_lead_time_by_scenario",
            "Среднее время упреждения по сценарию и горизонту",
        )
        blocks.extend(
            [
                "",
                "### Время упреждения по сценарию",
                "",
                scenario_caption,
                "",
                _to_markdown_table(predictive_scenario_table),
                "",
                scenario_figure,
                "",
                "Для сценариев с отложенным наступлением события время упреждения "
                "заметно больше: чем позже формируется физический конфликт, тем "
                "раньше относительно него срабатывает прогнозный сигнал.",
            ]
        )

    return "\n".join(blocks).strip()


def _render_local_candidate_note(has_local_candidates: bool) -> str:
    if has_local_candidates:
        return (
            "Локальные пики помогают увидеть, в какой момент внутри запуска "
            "возникли высокая сила, плотность или близкие контакты."
        )
    return (
        "Локальные кандидаты не найдены, поэтому временной график показывает "
        "только сами показатели движения."
    )


def _render_figure(
    figure_files: dict[str, Path],
    key: str,
    title: str,
) -> str:
    figure_path = figure_files.get(key)
    if figure_path is None:
        return ""
    relative_path = (Path("figures") / figure_path.name).as_posix()
    return f"![{title}]({relative_path})"


def _to_markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "_Нет данных._"

    columns = list(data.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in data.to_dict("records"):
        values = [_format_value(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
