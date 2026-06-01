"""Ablation study: feature-space configurations × Direct forecasting strategy.

Сравнивает 4 конфигурации признакового пространства расширенного состояния
для модулей multi_horizon и predictive_risk. Стратегия прогнозирования —
Direct (отдельный регрессор на каждый горизонт H). Цель: минимизировать
MAE/MSE многошагового прогноза, сохранив или улучшив метрики
упреждающего детектора (F1, Precision, Recall, Lead time).

Конфигурации:
  A_baseline       — текущий 7-мерный вектор;
  B_no_close_pair  — исключён целочисленный close_pair_count;
  C_smoothed_peaks — rolling-mean (окно 3) на max_force, density_proxy,
                     close_pair_count перед регрессией;
  D_force_p90      — max_force заменён на покадровый 90-й перцентиль силы.

Артефакты на каждый конфиг сохраняются в outputs/feature_ablation/demo/<name>/.
Сводная таблица — outputs/feature_ablation/demo/ablation_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crowd_anomaly import multi_horizon as mh  # noqa: E402
from crowd_anomaly import predictive_risk as pr  # noqa: E402
from crowd_anomaly.anomaly import _binary_metrics  # noqa: E402
from crowd_anomaly.features import (  # noqa: E402
    FEATURE_COLUMNS,
    load_feature_artifacts,
)

ORIGINAL_BUILD_SERIES = mh.build_extended_run_time_series


# ---------------------------------------------------------------------------
# Builders покадрового состояния для каждой конфигурации
# ---------------------------------------------------------------------------

def build_baseline_series(trajectories: pd.DataFrame) -> pd.DataFrame:
    return ORIGINAL_BUILD_SERIES(trajectories)


def build_no_close_pair_series(trajectories: pd.DataFrame) -> pd.DataFrame:
    full = ORIGINAL_BUILD_SERIES(trajectories)
    return full.drop(columns=["close_pair_count"])


def _apply_rolling(df: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = (
                out[column].rolling(window=window, min_periods=1, center=False).mean()
            )
    return out


def build_smoothed_series(trajectories: pd.DataFrame) -> pd.DataFrame:
    full = ORIGINAL_BUILD_SERIES(trajectories)
    return _apply_rolling(
        full,
        ["max_force", "density_proxy", "close_pair_count"],
        window=3,
    )


def build_force_p90_series(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, frame in trajectories.groupby("step", sort=True):
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        forces = frame.get("force_norm", pd.Series([0.0] * len(frame)))
        force_values = forces.astype(float).to_numpy()
        nearest = mh._nearest_distances(coordinates)
        density = float(np.mean(1.0 / (nearest + mh.EPSILON))) if nearest.size else 0.0
        rows.append(
            {
                "step": int(step),
                "mean_x": float(frame["x"].mean()),
                "mean_y": float(frame["y"].mean()),
                "mean_speed": float(frame["speed"].mean()),
                "density_proxy": density,
                "mean_force": (
                    float(np.mean(force_values)) if force_values.size else 0.0
                ),
                "force_p90": (
                    float(np.percentile(force_values, 90))
                    if force_values.size
                    else 0.0
                ),
                "close_pair_count": float(mh._count_close_pairs(coordinates)),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Конфигурации
# ---------------------------------------------------------------------------

CONFIGS: list[dict[str, Any]] = [
    {
        "name": "A_baseline",
        "description": (
            "Текущий 7-мерный вектор: mean_x, mean_y, mean_speed, density_proxy, "
            "mean_force, max_force, close_pair_count."
        ),
        "series_columns": [
            "mean_x", "mean_y", "mean_speed", "density_proxy",
            "mean_force", "max_force", "close_pair_count",
        ],
        "build_series": build_baseline_series,
        "force_field": "max_force",
    },
    {
        "name": "B_no_close_pair",
        "description": (
            "Исключён целочисленный close_pair_count с резкими переходами. "
            "close_pair_count_max берётся из оракульных признаков запуска."
        ),
        "series_columns": [
            "mean_x", "mean_y", "mean_speed", "density_proxy",
            "mean_force", "max_force",
        ],
        "build_series": build_no_close_pair_series,
        "force_field": "max_force",
    },
    {
        "name": "C_smoothed_peaks",
        "description": (
            "Скользящее среднее (окно=3, причинное) по max_force, density_proxy, "
            "close_pair_count до подачи в регрессор."
        ),
        "series_columns": [
            "mean_x", "mean_y", "mean_speed", "density_proxy",
            "mean_force", "max_force", "close_pair_count",
        ],
        "build_series": build_smoothed_series,
        "force_field": "max_force",
    },
    {
        "name": "D_force_p90",
        "description": (
            "max_force заменён на покадровый 90-й перцентиль силы (force_p90) — "
            "более устойчивая пиковая метрика. Используется на месте max_force "
            "при подстановке в физический штраф."
        ),
        "series_columns": [
            "mean_x", "mean_y", "mean_speed", "density_proxy",
            "mean_force", "force_p90", "close_pair_count",
        ],
        "build_series": build_force_p90_series,
        "force_field": "force_p90",
    },
]


# ---------------------------------------------------------------------------
# Кастомный синтезатор признакового вектора по прогнозу
# ---------------------------------------------------------------------------

def make_synthesize_features(
    series_columns: list[str],
    force_field: str,
) -> Callable:
    column_idx = {name: i for i, name in enumerate(series_columns)}

    def synthesize(
        history_peaks: dict[str, float],
        future_states: np.ndarray,
        base_features: pd.DataFrame,
        run_id: str,
        use_history_peaks: bool = False,
    ) -> dict[str, float]:
        base_row = (
            base_features.loc[base_features["run_id"] == run_id]
            .iloc[0]
            .to_dict()
        )
        synthesized = {col: float(base_row[col]) for col in FEATURE_COLUMNS}

        future_force_max = float(np.max(future_states[:, column_idx[force_field]]))
        if "mean_force" in column_idx:
            future_force_mean = float(
                np.mean(future_states[:, column_idx["mean_force"]])
            )
        else:
            future_force_mean = synthesized["force_mean"]

        if "density_proxy" in column_idx:
            future_density_max = float(
                np.max(future_states[:, column_idx["density_proxy"]])
            )
        else:
            future_density_max = synthesized["density_proxy_max"]

        if "close_pair_count" in column_idx:
            future_close_max = float(
                np.max(future_states[:, column_idx["close_pair_count"]])
            )
        else:
            future_close_max = synthesized["close_pair_count_max"]

        if use_history_peaks:
            if "max_force" in history_peaks:
                future_force_max = max(future_force_max, history_peaks["max_force"])
            if "force_p90" in history_peaks:
                future_force_max = max(future_force_max, history_peaks["force_p90"])
            if "density_proxy" in history_peaks:
                future_density_max = max(
                    future_density_max, history_peaks["density_proxy"]
                )
            if "close_pair_count" in history_peaks:
                future_close_max = max(
                    future_close_max, history_peaks["close_pair_count"]
                )

        synthesized["force_max"] = future_force_max
        synthesized["density_proxy_max"] = future_density_max
        synthesized["close_pair_count_max"] = future_close_max
        synthesized["force_mean"] = future_force_mean
        return synthesized

    return synthesize


# ---------------------------------------------------------------------------
# Префиксные пики покадрового состояния (для history_plus_forecast)
# ---------------------------------------------------------------------------

def _prefix_peaks(values: np.ndarray, columns: list[str]) -> list[dict[str, float]]:
    candidate_keys = ("max_force", "density_proxy", "close_pair_count", "force_p90")
    keys = [k for k in candidate_keys if k in columns]
    col_idx = {name: i for i, name in enumerate(columns)}
    peaks: list[dict[str, float]] = []
    running = {key: -np.inf for key in keys}
    for state in values:
        for key in keys:
            current = float(state[col_idx[key]])
            if current > running[key]:
                running[key] = current
        peaks.append(dict(running))
    return peaks


# ---------------------------------------------------------------------------
# Основной прогон конфигурации
# ---------------------------------------------------------------------------

def run_one_config(
    config: dict[str, Any],
    dataset_dir: Path,
    feature_dir: Path,
    output_root: Path,
    event_times: dict[str, dict[str, float]],
    horizons: tuple[int, ...],
    window_size: int,
) -> dict[str, Any]:
    name = config["name"]
    series_columns = config["series_columns"]
    build_series = config["build_series"]
    force_field = config["force_field"]

    output_path = output_root / name
    output_path.mkdir(parents=True, exist_ok=True)

    saved_mh_series = mh.EXTENDED_SERIES_COLUMNS
    saved_mh_target = mh.EXTENDED_TARGET_COLUMNS
    saved_mh_build = mh.build_extended_run_time_series
    saved_pr_series = pr.EXTENDED_SERIES_COLUMNS

    mh.EXTENDED_SERIES_COLUMNS = series_columns
    mh.EXTENDED_TARGET_COLUMNS = series_columns
    mh.build_extended_run_time_series = build_series
    pr.EXTENDED_SERIES_COLUMNS = series_columns

    try:
        runs = pd.read_csv(dataset_dir / "runs.csv")
        series_by_run = mh._load_extended_series(dataset_dir, runs)
        feature_artifacts = load_feature_artifacts(feature_dir)
        features_run = feature_artifacts["features_run"]
        run_mapping = feature_artifacts["run_mapping"]

        train_runs = set(runs.loc[runs["split"] == "train", "run_id"].astype(str))
        max_horizon = max(horizons)

        direct_models: dict[int, dict[str, Any]] = {}
        for h in range(1, max_horizon + 1):
            sequences, targets = mh._build_training_pairs(
                series_by_run=series_by_run,
                run_ids=train_runs,
                window_size=window_size,
                horizon=h,
            )
            if len(sequences) == 0:
                continue
            direct_models[h] = mh._fit_recurrent_regressor(
                train_sequences=sequences,
                train_targets=targets,
            )

        # -----------------------------------------------------------------
        # Раздел 1: оценка MAE/MSE для persistence и direct
        # -----------------------------------------------------------------
        prediction_rows: list[dict[str, Any]] = []
        for run in runs.to_dict("records"):
            run_id = str(run["run_id"])
            series = series_by_run.get(run_id)
            if series is None or len(series) < window_size + 1:
                continue
            values = series[series_columns].to_numpy(dtype=float)
            steps = series["step"].to_numpy(dtype=int)
            max_start = len(values) - window_size - max_horizon + 1
            if max_start <= 0:
                continue
            for start in range(max_start):
                window = values[start: start + window_size]
                hist_step = int(steps[start + window_size - 1])
                persistence_state = window[-1].copy()
                for horizon in horizons:
                    target_index = start + window_size + horizon - 1
                    if target_index >= len(values):
                        continue
                    actual = values[target_index]
                    step_target = int(steps[target_index])
                    mh._append_prediction_rows(
                        prediction_rows=prediction_rows,
                        run=run, run_id=run_id,
                        history_step=hist_step, target_step=step_target,
                        horizon=horizon, actual=actual, predicted=persistence_state,
                        model_name="persistence",
                    )
                    direct_model = direct_models.get(horizon)
                    if direct_model is None:
                        continue
                    predicted = mh._predict_with_model(
                        direct_model, window[np.newaxis]
                    )[0]
                    mh._append_prediction_rows(
                        prediction_rows=prediction_rows,
                        run=run, run_id=run_id,
                        history_step=hist_step, target_step=step_target,
                        horizon=horizon, actual=actual, predicted=predicted,
                        model_name="direct",
                    )

        predictions = pd.DataFrame(prediction_rows)
        metrics_overall = mh._summarize_overall(predictions)
        metrics_by_split = mh._summarize_metrics(predictions)
        metrics_overall.to_csv(
            output_path / "multi_horizon_metrics_overall.csv", index=False
        )
        metrics_by_split.to_csv(
            output_path / "multi_horizon_metrics_by_split.csv", index=False
        )
        predictions.to_csv(output_path / "multi_horizon_predictions.csv", index=False)

        # -----------------------------------------------------------------
        # Раздел 2: predictive_risk через Direct-rollout
        # -----------------------------------------------------------------
        physics_calibration = pr._calibrate_physics_score(features_run, run_mapping)
        pca_calibration = pr._calibrate_pca_reconstruction(features_run, run_mapping)
        synthesize_fn = make_synthesize_features(series_columns, force_field)

        causal_records: list[dict[str, Any]] = []
        for run in runs.to_dict("records"):
            run_id = str(run["run_id"])
            series = series_by_run.get(run_id)
            if series is None or len(series) < window_size + max_horizon:
                continue
            values = series[series_columns].to_numpy(dtype=float)
            steps = series["step"].to_numpy(dtype=int)
            time_seconds = pr._time_per_step(run, len(values))
            prefix_peaks = _prefix_peaks(values, series_columns)

            for start in range(len(values) - window_size - max_horizon + 1):
                window = values[start: start + window_size]
                history_end = start + window_size - 1
                rollout = mh.direct_rollout(direct_models, window, max_horizon)
                for horizon in horizons:
                    future = rollout[:horizon]
                    for variant_name, use_history in (
                        ("forecast_only", False),
                        ("history_plus_forecast", True),
                    ):
                        synthetic_features = synthesize_fn(
                            history_peaks=prefix_peaks[history_end],
                            future_states=future,
                            base_features=features_run,
                            run_id=run_id,
                            use_history_peaks=use_history,
                        )
                        scores = pr._evaluate_anomaly_score(
                            synthetic_features=synthetic_features,
                            physics_calibration=physics_calibration,
                            pca_calibration=pca_calibration,
                        )
                        causal_records.append({
                            "run_id": run_id,
                            "split": run["split"],
                            "label": int(run["label"]),
                            "label_name": run["label_name"],
                            "scenario_template": run["scenario_template"],
                            "history_step": int(steps[history_end]),
                            "history_time_s": float(time_seconds[history_end]),
                            "horizon": int(horizon),
                            "variant": variant_name,
                            "target_step": int(steps[history_end + horizon]),
                            "target_time_s": float(time_seconds[history_end + horizon]),
                            "anomaly_score": float(scores["anomaly_score"]),
                        })

        causal_predictions = pd.DataFrame(causal_records)
        causal_predictions.to_csv(
            output_path / "predictive_risk_scores.csv", index=False
        )

        # -----------------------------------------------------------------
        # Метрики детектора по горизонту/варианту/split
        # -----------------------------------------------------------------
        horizon_summaries: list[dict[str, Any]] = []
        for horizon in horizons:
            for variant in ["forecast_only", "history_plus_forecast"]:
                hp = causal_predictions[
                    (causal_predictions["horizon"] == horizon)
                    & (causal_predictions["variant"] == variant)
                ]
                if hp.empty:
                    continue
                run_summary = pr._runwise_summary(hp)
                threshold, _ = pr._select_threshold(run_summary)
                run_summary["predicted_label"] = (
                    run_summary["max_score"] > threshold
                ).astype(int)
                run_summary["event_step"] = run_summary["run_id"].map(
                    {k: v["event_step"] for k, v in event_times.items()}
                )
                run_summary["event_time_s"] = run_summary["run_id"].map(
                    {k: v["event_time_s"] for k, v in event_times.items()}
                )
                run_summary["alarm_step"] = run_summary["run_id"].map(
                    lambda rid, hp=hp, t=threshold: pr._alarm_step(hp, rid, t)
                )
                run_summary["alarm_time_s"] = run_summary["run_id"].map(
                    lambda rid, hp=hp, t=threshold: pr._alarm_time(hp, rid, t)
                )
                run_summary["lead_time_s"] = (
                    run_summary["event_time_s"] - run_summary["alarm_time_s"]
                )

                for split_name, split_data in run_summary.groupby("split"):
                    metrics = _binary_metrics(
                        y_true=split_data["label"].to_numpy(),
                        y_pred=split_data["predicted_label"].to_numpy(),
                    )
                    anomaly_split = split_data[split_data["label"] == 1]
                    detected = anomaly_split[
                        (anomaly_split["predicted_label"] == 1)
                        & anomaly_split["lead_time_s"].notna()
                    ]
                    detected_pos = detected[detected["lead_time_s"] >= 0.0]
                    horizon_summaries.append({
                        "config": name,
                        "horizon": int(horizon),
                        "variant": variant,
                        "split": split_name,
                        **metrics,
                        "threshold": float(threshold),
                        "lead_time_mean_s": (
                            float(detected_pos["lead_time_s"].mean())
                            if not detected_pos.empty else float("nan")
                        ),
                        "lead_time_median_s": (
                            float(detected_pos["lead_time_s"].median())
                            if not detected_pos.empty else float("nan")
                        ),
                        "detected_with_lead": int(len(detected_pos)),
                        "run_count": int(len(split_data)),
                    })

        horizon_metrics = pd.DataFrame(horizon_summaries)
        horizon_metrics.to_csv(
            output_path / "predictive_risk_metrics_by_horizon.csv", index=False
        )

        forecast_h1_test = horizon_metrics[
            (horizon_metrics["split"] == "test")
            & (horizon_metrics["variant"] == "forecast_only")
            & (horizon_metrics["horizon"] == 1)
        ]
        mh_h1_test = metrics_overall[
            (metrics_overall["split"] == "test")
            & (metrics_overall["horizon"] == 1)
            & (metrics_overall["model"] == "direct")
        ]
        summary_h1 = {
            "config": name,
            "description": config["description"],
            "n_series_columns": len(series_columns),
            "mae_direct_h1_test": (
                float(mh_h1_test["mae"].iloc[0])
                if not mh_h1_test.empty
                else float("nan")
            ),
            "mse_direct_h1_test": (
                float(mh_h1_test["mse"].iloc[0])
                if not mh_h1_test.empty
                else float("nan")
            ),
            "f1_h1_test": (
                float(forecast_h1_test["f1"].iloc[0])
                if not forecast_h1_test.empty
                else float("nan")
            ),
            "precision_h1_test": (
                float(forecast_h1_test["precision"].iloc[0])
                if not forecast_h1_test.empty
                else float("nan")
            ),
            "recall_h1_test": (
                float(forecast_h1_test["recall"].iloc[0])
                if not forecast_h1_test.empty
                else float("nan")
            ),
            "lead_time_mean_s_h1_test": (
                float(forecast_h1_test["lead_time_mean_s"].iloc[0])
                if not forecast_h1_test.empty else float("nan")
            ),
            "lead_time_median_s_h1_test": (
                float(forecast_h1_test["lead_time_median_s"].iloc[0])
                if not forecast_h1_test.empty else float("nan")
            ),
        }

        return {
            "name": name,
            "summary_h1": summary_h1,
            "metrics_overall": metrics_overall,
            "horizon_metrics": horizon_metrics,
            "output_dir": output_path,
        }
    finally:
        mh.EXTENDED_SERIES_COLUMNS = saved_mh_series
        mh.EXTENDED_TARGET_COLUMNS = saved_mh_target
        mh.build_extended_run_time_series = saved_mh_build
        pr.EXTENDED_SERIES_COLUMNS = saved_pr_series


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    dataset_dir = PROJECT_ROOT / "outputs" / "datasets" / "demo"
    feature_dir = PROJECT_ROOT / "outputs" / "features" / "demo"
    output_root = PROJECT_ROOT / "outputs" / "feature_ablation" / "demo"
    output_root.mkdir(parents=True, exist_ok=True)

    horizons = (1, 3, 5)
    window_size = 5

    print(f"Ablation: dataset={dataset_dir.name}, horizons={horizons}, strategy=Direct")

    # Эталонные event_times считаем один раз на baseline-серии,
    # чтобы lead_time всех конфигов сравнивался по одной шкале.
    runs = pd.read_csv(dataset_dir / "runs.csv")
    baseline_series_by_run = mh._load_extended_series(dataset_dir, runs)
    event_times = pr._event_times_per_run(baseline_series_by_run, runs)

    all_summaries: list[dict[str, Any]] = []
    all_horizon_metrics: list[pd.DataFrame] = []
    for config in CONFIGS:
        print(f"\n[{config['name']}] {config['description']}")
        result = run_one_config(
            config=config,
            dataset_dir=dataset_dir,
            feature_dir=feature_dir,
            output_root=output_root,
            event_times=event_times,
            horizons=horizons,
            window_size=window_size,
        )
        all_summaries.append(result["summary_h1"])
        all_horizon_metrics.append(result["horizon_metrics"])
        s = result["summary_h1"]
        print(
            f"  MAE_h1={s['mae_direct_h1_test']:.4f}  "
            f"MSE_h1={s['mse_direct_h1_test']:.4f}  "
            f"F1_h1={s['f1_h1_test']:.3f}  "
            f"P_h1={s['precision_h1_test']:.3f}  "
            f"R_h1={s['recall_h1_test']:.3f}  "
            f"Lead={s['lead_time_mean_s_h1_test']:.3f}s"
        )

    summary_df = pd.DataFrame(all_summaries)
    summary_path = output_root / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    combined_horizon = pd.concat(all_horizon_metrics, ignore_index=True)
    combined_path = output_root / "ablation_all_horizons.csv"
    combined_horizon.to_csv(combined_path, index=False)

    print(f"\nSummary CSV: {summary_path}")
    print(f"All-horizon CSV: {combined_path}")
    print("\n=== ABLATION SUMMARY (H=1, test, forecast_only) ===")
    display_cols = [
        "config", "n_series_columns",
        "mae_direct_h1_test", "mse_direct_h1_test",
        "f1_h1_test", "precision_h1_test", "recall_h1_test",
        "lead_time_mean_s_h1_test",
    ]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
