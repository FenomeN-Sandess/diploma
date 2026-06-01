"""Упреждающий сигнал риска по прогнозируемым признакам аномальности.

Реализует направление 1.2 из `Разбор/перспективы_развития.md`:

* строится покадровый ряд расширенного агрегированного состояния
  (см. EXTENDED_SERIES_COLUMNS в `multi_horizon.py`);
* для каждого горизонта h ∈ {1..H} обучается отдельный direct-регрессор,
  его выход — состояние сразу на h шагов вперёд из текущего окна
  (стратегия Direct: ошибка не накапливается по шагам);
* по прогнозу считаются предсказанные пиковые признаки запуска
  (force_max, density_proxy_max, close_pair_count_max) и затем
  функционал аномальности

      Ŝ(t, H) = α · E_PCA(x̂(R | t, H)) + (1 - α) · P(x̂(R | t, H));

* для каждого запуска фиксируется самое раннее t_alarm, при котором
  Ŝ(t, H) превышает порог θ, выбранный на валидации по максимуму F1;
* для аномальных запусков считается lead time =
  t_event − t_alarm, где t_event — кадр с пиком frame_score
  (тот же показатель используется в `anomaly.build_local_anomaly_candidates`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crowd_anomaly.anomaly import (
    MODEL_RANDOM_STATE,
    PCA_COMPONENTS,
    PCA_FEATURE_FRAGMENTS,
    PCA_SCORE_WEIGHT,
    PHYSICS_ANOMALY_COLUMNS,
    _binary_metrics,
    _candidate_thresholds,
)
from crowd_anomaly.features import FEATURE_COLUMNS, load_feature_artifacts
from crowd_anomaly.multi_horizon import (
    DEFAULT_HORIZONS,
    DEFAULT_WINDOW,
    EXTENDED_SERIES_COLUMNS,
    _build_training_pairs,
    _fit_recurrent_regressor,
    _load_extended_series,
    direct_rollout,
)

PREDICTIVE_FORECAST_COLUMNS = ("mean_force", "max_force", "close_pair_count")
PREDICTIVE_TO_FEATURE_NAME = {
    "mean_force": "force_mean",
    "max_force": "force_max",
    "close_pair_count": "close_pair_count_max",
    "density_proxy": "density_proxy_max",
}
EPSILON = 1e-9


def run_predictive_risk(
    dataset_dir: Path | str,
    feature_dir: Path | str,
    output_dir: Path | str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    window_size: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Главный сценарий: прогноз → пиковые признаки → S(t, H) → lead time."""
    dataset_path = Path(dataset_dir)
    feature_path = Path(feature_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in sorted(set(horizons)))

    runs = pd.read_csv(dataset_path / "runs.csv")
    series_by_run = _load_extended_series(dataset_path, runs)
    feature_artifacts = load_feature_artifacts(feature_path)
    features_run = feature_artifacts["features_run"]
    run_mapping = feature_artifacts["run_mapping"]

    train_runs = set(runs.loc[runs["split"] == "train", "run_id"].astype(str))
    max_horizon = max(horizons)
    direct_models: dict[int, dict[str, Any]] = {}
    for h in range(1, max_horizon + 1):
        sequences, targets = _build_training_pairs(
            series_by_run=series_by_run,
            run_ids=train_runs,
            window_size=window_size,
            horizon=h,
        )
        if len(sequences) == 0:
            continue
        direct_models[h] = _fit_recurrent_regressor(
            train_sequences=sequences,
            train_targets=targets,
        )
    if not direct_models:
        raise ValueError("Нет обучающих окон для прогноза.")

    physics_calibration = _calibrate_physics_score(features_run, run_mapping)
    pca_calibration = _calibrate_pca_reconstruction(features_run, run_mapping)

    causal_records: list[dict[str, Any]] = []
    for run in runs.to_dict("records"):
        run_id = str(run["run_id"])
        series = series_by_run.get(run_id)
        if series is None or len(series) < window_size + max_horizon:
            continue
        values = series[EXTENDED_SERIES_COLUMNS].to_numpy(dtype=float)
        steps = series["step"].to_numpy(dtype=int)
        time_seconds = _time_per_step(run, len(values))
        prefix_peaks = _prefix_peaks(values)
        for start in range(len(values) - window_size - max_horizon + 1):
            window = values[start : start + window_size]
            history_end = start + window_size - 1
            rollout = direct_rollout(
                direct_models=direct_models,
                window=window,
                max_horizon=max_horizon,
            )
            for horizon in horizons:
                future = rollout[:horizon]
                for variant_name, use_history in (
                    ("forecast_only", False),
                    ("history_plus_forecast", True),
                ):
                    synthetic_features = _synthesize_features(
                        history_peaks=prefix_peaks[history_end],
                        future_states=future,
                        base_features=features_run,
                        run_id=run_id,
                        use_history_peaks=use_history,
                    )
                    scores = _evaluate_anomaly_score(
                        synthetic_features=synthetic_features,
                        physics_calibration=physics_calibration,
                        pca_calibration=pca_calibration,
                    )
                    causal_records.append(
                        {
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
                            "predicted_force_max": synthetic_features["force_max"],
                            "predicted_density_proxy_max": synthetic_features[
                                "density_proxy_max"
                            ],
                            "predicted_close_pair_count_max": synthetic_features[
                                "close_pair_count_max"
                            ],
                            "physics_score": float(scores["physics_score"]),
                            "pca_score": float(scores["pca_score"]),
                            "anomaly_score": float(scores["anomaly_score"]),
                        }
                    )

    causal_predictions = pd.DataFrame(causal_records)
    event_times = _event_times_per_run(series_by_run, runs)
    oracle_scores = _oracle_run_scores(
        features_run=features_run,
        run_mapping=run_mapping,
        physics_calibration=physics_calibration,
        pca_calibration=pca_calibration,
    )

    horizon_summaries: list[dict[str, Any]] = []
    lead_time_rows: list[dict[str, Any]] = []
    per_combo_alarms: list[pd.DataFrame] = []
    variants = sorted(causal_predictions["variant"].unique())
    for horizon in horizons:
        for variant in variants:
            horizon_predictions = causal_predictions[
                (causal_predictions["horizon"] == horizon)
                & (causal_predictions["variant"] == variant)
            ]
            run_summary = _runwise_summary(horizon_predictions)
            threshold, threshold_rule = _select_threshold(run_summary)
            run_summary["predicted_label"] = (
                run_summary["max_score"] > threshold
            ).astype(int)
            run_summary["threshold"] = float(threshold)
            run_summary["horizon"] = int(horizon)
            run_summary["variant"] = variant
            run_summary["event_step"] = run_summary["run_id"].map(
                {key: value["event_step"] for key, value in event_times.items()}
            )
            run_summary["event_time_s"] = run_summary["run_id"].map(
                {key: value["event_time_s"] for key, value in event_times.items()}
            )
            run_summary["alarm_step"] = run_summary["run_id"].map(
                lambda rid, hp=horizon_predictions, t=threshold: _alarm_step(hp, rid, t)
            )
            run_summary["alarm_time_s"] = run_summary["run_id"].map(
                lambda rid, hp=horizon_predictions, t=threshold: _alarm_time(hp, rid, t)
            )
            run_summary["lead_time_s"] = (
                run_summary["event_time_s"] - run_summary["alarm_time_s"]
            )
            per_combo_alarms.append(run_summary)

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
                detected_positive = detected[detected["lead_time_s"] >= 0.0]
                horizon_summaries.append(
                    {
                        "horizon": int(horizon),
                        "variant": variant,
                        "split": split_name,
                        **metrics,
                        "threshold": float(threshold),
                        "threshold_rule": threshold_rule,
                        "lead_time_mean_s": (
                            float(detected_positive["lead_time_s"].mean())
                            if not detected_positive.empty
                            else float("nan")
                        ),
                        "lead_time_median_s": (
                            float(detected_positive["lead_time_s"].median())
                            if not detected_positive.empty
                            else float("nan")
                        ),
                        "lead_time_p10_s": (
                            float(detected_positive["lead_time_s"].quantile(0.1))
                            if not detected_positive.empty
                            else float("nan")
                        ),
                        "lead_time_p90_s": (
                            float(detected_positive["lead_time_s"].quantile(0.9))
                            if not detected_positive.empty
                            else float("nan")
                        ),
                        "detected_with_lead": int(len(detected_positive)),
                        "detected_anomalies": int(len(detected)),
                        "run_count": int(len(split_data)),
                    }
                )
                for record in detected_positive.to_dict("records"):
                    lead_time_rows.append(
                        {
                            "horizon": int(horizon),
                            "variant": variant,
                            "split": split_name,
                            "run_id": record["run_id"],
                            "scenario_template": record["scenario_template"],
                            "lead_time_s": float(record["lead_time_s"]),
                            "event_time_s": float(record["event_time_s"]),
                            "alarm_time_s": float(record["alarm_time_s"]),
                        }
                    )

    horizon_metrics = pd.DataFrame(horizon_summaries)
    lead_time_table = pd.DataFrame(lead_time_rows)
    causal_predictions.to_csv(
        output_path / "predictive_risk_scores.csv", index=False
    )
    if per_combo_alarms:
        combined_alarms = pd.concat(per_combo_alarms, ignore_index=True)
        combined_alarms.to_csv(
            output_path / "predictive_risk_alarms.csv", index=False
        )
    horizon_metrics.to_csv(
        output_path / "predictive_risk_metrics_by_horizon.csv", index=False
    )
    lead_time_table.to_csv(
        output_path / "predictive_risk_lead_times.csv", index=False
    )
    oracle_scores.to_csv(
        output_path / "predictive_risk_oracle_scores.csv", index=False
    )

    scenario_lead_time = _per_scenario_lead_time(per_combo_alarms, horizons)
    scenario_lead_time.to_csv(
        output_path / "predictive_risk_lead_time_by_scenario.csv", index=False
    )
    plots = {
        "f1_curve": _plot_metric_curve(
            metrics=horizon_metrics,
            metric="f1",
            output_path=output_path / "predictive_risk_f1_curve.png",
            ylabel="F1 (по запуску)",
            title="F1 упреждающего детектора по горизонту H",
        ),
        "recall_curve": _plot_metric_curve(
            metrics=horizon_metrics,
            metric="recall",
            output_path=output_path / "predictive_risk_recall_curve.png",
            ylabel="Recall",
            title="Recall упреждающего детектора по горизонту H",
        ),
        "lead_time": _plot_lead_time_distribution(
            lead_time_table=lead_time_table,
            output_path=output_path / "predictive_risk_lead_time.png",
        ),
        "score_timeline": _plot_score_timeline(
            causal_predictions=causal_predictions,
            event_times=event_times,
            output_path=output_path / "predictive_risk_score_timeline.png",
            horizon=horizons[0] if horizons else 1,
        ),
        "scenario_bars": _plot_scenario_lead_time(
            scenario_lead_time=scenario_lead_time,
            output_path=output_path / "predictive_risk_lead_time_by_scenario.png",
        ),
    }

    metrics_payload = {
        "task": "predictive_risk_signal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_path),
        "feature_dir": str(feature_path),
        "window_size": int(window_size),
        "horizons": list(horizons),
        "anomaly_score_formula": (
            f"S_hat(t, H) = {PCA_SCORE_WEIGHT} * E_PCA(x_hat) + "
            f"{1 - PCA_SCORE_WEIGHT} * P(x_hat); "
            "x_hat — вектор признаков запуска, в котором пиковые "
            "force_max/density_proxy_max/close_pair_count_max заменены "
            "максимумом по прогнозу на H шагов вперёд."
        ),
        "predictive_forecast_columns": list(PREDICTIVE_FORECAST_COLUMNS),
        "threshold_rule": "val_best_f1",
        "lead_time_definition": (
            "event_time_s − alarm_time_s. event_time_s — первый кадр после "
            "warm-up SIMULATOR_WARMUP_STEPS, где close_pair_count превышает "
            "max(q95 нормы train, 1). alarm_time_s — самое раннее t, "
            "когда S_hat(t, H) > θ."
        ),
        "limitations": [
            "Прогнозируемые пиковые признаки берутся как максимум по горизонту H, "
            "а не по полному будущему запуска.",
            "PCA и нормировка обучаются на полных признаках train-выборки; "
            "ошибка прогноза будущих кадров переносится в признаковое пространство.",
            "Для нормальных запусков event_time_s используется только для отчёта, "
            "решающая метрика — false alarm flag.",
        ],
        "files": {
            "scores": "predictive_risk_scores.csv",
            "alarms": "predictive_risk_alarms.csv",
            "metrics": "predictive_risk_metrics_by_horizon.csv",
            "lead_times": "predictive_risk_lead_times.csv",
            "lead_time_by_scenario": "predictive_risk_lead_time_by_scenario.csv",
            "oracle_scores": "predictive_risk_oracle_scores.csv",
            "f1_curve": "predictive_risk_f1_curve.png",
            "recall_curve": "predictive_risk_recall_curve.png",
            "lead_time_plot": "predictive_risk_lead_time.png",
            "lead_time_by_scenario_plot": "predictive_risk_lead_time_by_scenario.png",
            "score_timeline_plot": "predictive_risk_score_timeline.png",
        },
    }
    (output_path / "predictive_risk_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": output_path,
        "causal_predictions": causal_predictions,
        "horizon_metrics": horizon_metrics,
        "lead_times": lead_time_table,
        "oracle_scores": oracle_scores,
        "metrics": metrics_payload,
        "plots": plots,
    }


def _time_per_step(run: dict[str, Any], length: int) -> np.ndarray:
    duration_s = float(run.get("duration_s", 0.0))
    if length <= 1 or duration_s <= 0.0:
        return np.zeros(length, dtype=float)
    dt = duration_s / float(length - 1)
    return np.arange(length, dtype=float) * dt


def _prefix_peaks(values: np.ndarray) -> list[dict[str, float]]:
    columns = {name: index for index, name in enumerate(EXTENDED_SERIES_COLUMNS)}
    peaks: list[dict[str, float]] = []
    running = {key: -np.inf for key in PREDICTIVE_TO_FEATURE_NAME}
    for state in values:
        for key in PREDICTIVE_TO_FEATURE_NAME:
            current = float(state[columns[key]])
            if current > running[key]:
                running[key] = current
        peaks.append(dict(running))
    return peaks


def _synthesize_features(
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
    columns = {name: index for index, name in enumerate(EXTENDED_SERIES_COLUMNS)}

    synthesized = {column: float(base_row[column]) for column in FEATURE_COLUMNS}
    future_force_max = float(np.max(future_states[:, columns["max_force"]]))
    future_density_max = float(np.max(future_states[:, columns["density_proxy"]]))
    future_close_max = float(np.max(future_states[:, columns["close_pair_count"]]))
    future_force_mean = float(np.mean(future_states[:, columns["mean_force"]]))

    if use_history_peaks:
        future_force_max = max(future_force_max, history_peaks["max_force"])
        future_density_max = max(future_density_max, history_peaks["density_proxy"])
        future_close_max = max(future_close_max, history_peaks["close_pair_count"])

    synthesized["force_max"] = future_force_max
    synthesized["density_proxy_max"] = future_density_max
    synthesized["close_pair_count_max"] = future_close_max
    synthesized["force_mean"] = future_force_mean
    return synthesized


def _calibrate_physics_score(
    features_run: pd.DataFrame,
    run_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    merged = features_run.merge(run_mapping[["run_id", "split"]], on="run_id")
    train_data = merged[merged["split"] == "train"]
    calibration: dict[str, dict[str, float]] = {}
    for column in PHYSICS_ANOMALY_COLUMNS:
        values = train_data[column].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = mad if mad > EPSILON else float(np.std(values))
        if scale <= EPSILON:
            scale = 1.0
        calibration[column] = {"median": median, "scale": scale}
    return calibration


def _calibrate_pca_reconstruction(
    features_run: pd.DataFrame,
    run_mapping: pd.DataFrame,
) -> dict[str, Any]:
    merged = features_run.merge(run_mapping[["run_id", "split"]], on="run_id")
    train_data = merged[merged["split"] == "train"]
    pca_columns = [
        column
        for column in FEATURE_COLUMNS
        if any(fragment in column for fragment in PCA_FEATURE_FRAGMENTS)
    ]
    if not pca_columns:
        pca_columns = FEATURE_COLUMNS

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[pca_columns])
    component_count = max(
        1,
        min(PCA_COMPONENTS, train_scaled.shape[1], len(train_scaled) - 1),
    )
    pca_model = PCA(n_components=component_count, random_state=MODEL_RANDOM_STATE)
    pca_model.fit(train_scaled)
    train_restored = pca_model.inverse_transform(pca_model.transform(train_scaled))
    train_error = np.mean((train_scaled - train_restored) ** 2, axis=1)
    median = float(np.median(train_error))
    mad = float(np.median(np.abs(train_error - median)))
    scale = mad if mad > EPSILON else float(np.std(train_error))
    if scale <= EPSILON:
        scale = 1.0
    return {
        "pca_columns": pca_columns,
        "scaler": scaler,
        "pca_model": pca_model,
        "median": median,
        "scale": scale,
        "component_count": int(component_count),
    }


def _evaluate_anomaly_score(
    synthetic_features: dict[str, float],
    physics_calibration: dict[str, dict[str, float]],
    pca_calibration: dict[str, Any],
) -> dict[str, float]:
    physics_components = []
    for column in PHYSICS_ANOMALY_COLUMNS:
        calibration = physics_calibration[column]
        value = float(synthetic_features[column])
        component = max((value - calibration["median"]) / calibration["scale"], 0.0)
        physics_components.append(component)
    physics_score = float(np.mean(physics_components))

    pca_columns = pca_calibration["pca_columns"]
    feature_row = pd.DataFrame(
        [[float(synthetic_features[column]) for column in pca_columns]],
        columns=pca_columns,
    )
    scaled = pca_calibration["scaler"].transform(feature_row)
    restored = pca_calibration["pca_model"].inverse_transform(
        pca_calibration["pca_model"].transform(scaled)
    )
    reconstruction_error = float(np.mean((scaled - restored) ** 2))
    pca_score = max(
        (reconstruction_error - pca_calibration["median"]) / pca_calibration["scale"],
        0.0,
    )

    anomaly_score = (
        PCA_SCORE_WEIGHT * pca_score + (1.0 - PCA_SCORE_WEIGHT) * physics_score
    )
    return {
        "physics_score": physics_score,
        "pca_score": pca_score,
        "anomaly_score": anomaly_score,
    }


SIMULATOR_WARMUP_STEPS = 10


def _event_times_per_run(
    series_by_run: dict[str, pd.DataFrame],
    runs: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Момент «опасного события» в запуске.

    Событие — первый кадр **после переходного процесса симулятора**
    (`SIMULATOR_WARMUP_STEPS` шагов), в котором close_pair_count превышает
    порог, выбранный по нормальным запускам train-выборки
    (95-й перцентиль, рассчитанный с пропуском первых SIMULATOR_WARMUP_STEPS
    шагов; не менее 1.0, чтобы не срабатывать на единичных случайных
    парах в разреженной норме).

    Раньше событие определялось по max_force, но max_force даёт
    артефактный пик на t ≈ 1 с во всех сценариях из-за того, что в
    модели Хелбинга goal_force на первом шаге резко разгоняет агентов
    из v = 0. Этот пик не связан с реальным скоплением и сдвигает
    t_event к началу запуска, обнуляя физический смысл lead_time на
    отложенных сценариях. close_pair_count свободен от этого артефакта
    (геометрический индикатор скопления), и его пик соответствует
    моменту физического сжатия толпы.
    """
    columns = {name: index for index, name in enumerate(EXTENDED_SERIES_COLUMNS)}
    event_signal = "close_pair_count"
    normal_train_signal: list[float] = []
    for run in runs.to_dict("records"):
        if run.get("split") != "train" or int(run.get("label", 0)) != 0:
            continue
        run_id = str(run["run_id"])
        series = series_by_run.get(run_id)
        if series is None or len(series) <= SIMULATOR_WARMUP_STEPS:
            continue
        if event_signal not in columns:
            continue
        values = series[EXTENDED_SERIES_COLUMNS].to_numpy(dtype=float)
        normal_train_signal.extend(
            values[SIMULATOR_WARMUP_STEPS:, columns[event_signal]].tolist()
        )
    if normal_train_signal:
        raw_threshold = float(np.quantile(normal_train_signal, 0.95))
    else:
        raw_threshold = 0.0
    # Защита от вырожденного случая: q95 на разреженной норме может
    # быть равен 0, тогда условие `> 0` сработало бы на первом же
    # кадре с любой парой агентов. Поднимаем порог как минимум до 1.
    threshold = max(raw_threshold, 1.0)

    event_times: dict[str, dict[str, float]] = {}
    for run in runs.to_dict("records"):
        run_id = str(run["run_id"])
        series = series_by_run.get(run_id)
        if series is None or series.empty:
            event_times[run_id] = {
                "event_step": float("nan"),
                "event_time_s": float("nan"),
            }
            continue
        values = series[EXTENDED_SERIES_COLUMNS].to_numpy(dtype=float)
        if event_signal in columns:
            signal_series = values[:, columns[event_signal]]
        else:
            # fallback на max_force, если в текущей конфигурации
            # close_pair_count отсутствует (например, ablation B).
            signal_series = values[:, columns["max_force"]]
        time_array = _time_per_step(run, len(values))
        warmup = min(SIMULATOR_WARMUP_STEPS, len(values) - 1)
        crossing = np.where(signal_series[warmup:] > threshold)[0]
        if crossing.size > 0:
            event_index = int(warmup + crossing[0])
        else:
            event_index = int(np.argmax(signal_series[warmup:])) + warmup
        event_times[run_id] = {
            "event_step": float(series["step"].iloc[event_index]),
            "event_time_s": float(time_array[event_index]),
            "event_threshold": float(threshold),
        }
    return event_times


def _oracle_run_scores(
    features_run: pd.DataFrame,
    run_mapping: pd.DataFrame,
    physics_calibration: dict[str, dict[str, float]],
    pca_calibration: dict[str, Any],
) -> pd.DataFrame:
    merged = features_run.merge(
        run_mapping[["run_id", "split", "label", "label_name"]], on="run_id"
    )
    rows = []
    for record in merged.to_dict("records"):
        synthesized = {column: float(record[column]) for column in FEATURE_COLUMNS}
        scores = _evaluate_anomaly_score(
            synthetic_features=synthesized,
            physics_calibration=physics_calibration,
            pca_calibration=pca_calibration,
        )
        rows.append(
            {
                "run_id": record["run_id"],
                "split": record["split"],
                "label": int(record["label"]),
                "label_name": record["label_name"],
                "physics_score": float(scores["physics_score"]),
                "pca_score": float(scores["pca_score"]),
                "anomaly_score": float(scores["anomaly_score"]),
            }
        )
    return pd.DataFrame(rows)


def _runwise_summary(horizon_predictions: pd.DataFrame) -> pd.DataFrame:
    if horizon_predictions.empty:
        return pd.DataFrame(
            columns=[
                "run_id", "split", "label", "label_name",
                "scenario_template", "max_score",
            ]
        )
    aggregated = horizon_predictions.groupby(
        ["run_id", "split", "label", "label_name", "scenario_template"],
        sort=True,
    ).agg(max_score=("anomaly_score", "max")).reset_index()
    return aggregated


def _select_threshold(run_summary: pd.DataFrame) -> tuple[float, str]:
    validation = run_summary[run_summary["split"] == "val"]
    if validation.empty or validation["label"].nunique() < 2:
        train_data = run_summary[run_summary["split"] == "train"]
        if train_data.empty:
            return 0.0, "fallback_zero"
        threshold = float(np.quantile(train_data["max_score"], 0.7))
        return threshold, "train_quantile_fallback"
    best_threshold = 0.0
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for threshold in _candidate_thresholds(validation["max_score"].to_numpy()):
        predicted = (validation["max_score"] > threshold).astype(int).to_numpy()
        metrics = _binary_metrics(validation["label"].to_numpy(), predicted)
        key = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["accuracy"],
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, "val_best_f1"


def _alarm_step(
    horizon_predictions: pd.DataFrame,
    run_id: str,
    threshold: float,
) -> float:
    rows = horizon_predictions[
        (horizon_predictions["run_id"] == run_id)
        & (horizon_predictions["anomaly_score"] > threshold)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.sort_values("history_step")["history_step"].iloc[0])


def _alarm_time(
    horizon_predictions: pd.DataFrame,
    run_id: str,
    threshold: float,
) -> float:
    rows = horizon_predictions[
        (horizon_predictions["run_id"] == run_id)
        & (horizon_predictions["anomaly_score"] > threshold)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.sort_values("history_time_s")["history_time_s"].iloc[0])


def _per_scenario_lead_time(
    per_combo_alarms: list[pd.DataFrame],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if not per_combo_alarms:
        return pd.DataFrame(
            columns=[
                "scenario_template",
                "horizon",
                "variant",
                "split",
                "n_runs",
                "n_detected",
                "n_with_positive_lead",
                "lead_time_mean_s",
                "lead_time_median_s",
                "alarm_time_mean_s",
            ]
        )
    combined = pd.concat(per_combo_alarms, ignore_index=True)
    anomalous = combined[combined["label"] == 1]
    rows = []
    for (scenario, horizon, variant, split), group in anomalous.groupby(
        ["scenario_template", "horizon", "variant", "split"], sort=True
    ):
        detected = group[group["alarm_time_s"].notna()]
        positive = detected[detected["lead_time_s"] >= 0.0]
        rows.append(
            {
                "scenario_template": scenario,
                "horizon": int(horizon),
                "variant": variant,
                "split": split,
                "n_runs": int(len(group)),
                "n_detected": int(len(detected)),
                "n_with_positive_lead": int(len(positive)),
                "lead_time_mean_s": (
                    float(positive["lead_time_s"].mean())
                    if not positive.empty
                    else float("nan")
                ),
                "lead_time_median_s": (
                    float(positive["lead_time_s"].median())
                    if not positive.empty
                    else float("nan")
                ),
                "alarm_time_mean_s": (
                    float(detected["alarm_time_s"].mean())
                    if not detected.empty
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_scenario_lead_time(
    scenario_lead_time: pd.DataFrame,
    output_path: Path,
) -> Path:
    if scenario_lead_time.empty:
        return output_path
    target = scenario_lead_time[
        (scenario_lead_time["variant"] == "forecast_only")
        & (scenario_lead_time["split"] == "test")
    ]
    if target.empty:
        target = scenario_lead_time[scenario_lead_time["variant"] == "forecast_only"]
    if target.empty:
        return output_path
    pivot = target.pivot_table(
        index="scenario_template",
        columns="horizon",
        values="lead_time_mean_s",
        aggfunc="mean",
    )
    horizons = sorted(pivot.columns.tolist())
    scenarios = pivot.index.tolist()
    figure, axis = plt.subplots(figsize=(9, 5))
    bar_width = 0.8 / max(len(horizons), 1)
    indices = np.arange(len(scenarios))
    palette = plt.cm.viridis(np.linspace(0, 0.9, len(horizons)))
    for offset, (horizon, color) in enumerate(zip(horizons, palette)):
        if horizon not in pivot.columns:
            continue
        values = pivot[horizon].fillna(0.0).to_numpy()
        axis.bar(
            indices + offset * bar_width,
            values,
            width=bar_width,
            color=color,
            label=f"H = {horizon}",
        )
    axis.set_xticks(indices + bar_width * (len(horizons) - 1) / 2.0)
    axis.set_xticklabels(scenarios, rotation=20, ha="right")
    axis.set_ylabel("Средний lead time, c (test, forecast_only)")
    axis.set_title(
        "Средний lead time упреждающего детектора по сценарию и горизонту"
    )
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_metric_curve(
    metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
    title: str,
) -> Path:
    if metrics.empty:
        return output_path
    figure, axis = plt.subplots(figsize=(8, 5))
    palette = {"train": "#4c78a8", "val": "#f28e2b", "test": "#54a24b"}
    line_styles = {
        "forecast_only": "-",
        "history_plus_forecast": "--",
    }
    for split, color in palette.items():
        for variant, style in line_styles.items():
            rows = metrics[
                (metrics["split"] == split) & (metrics["variant"] == variant)
            ].sort_values("horizon")
            if rows.empty:
                continue
            axis.plot(
                rows["horizon"],
                rows[metric],
                marker="o",
                color=color,
                linestyle=style,
                linewidth=1.5,
                label=f"{split} · {variant}",
            )
    axis.set_xlabel("Горизонт прогноза H, шагов")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_lead_time_distribution(
    lead_time_table: pd.DataFrame,
    output_path: Path,
) -> Path:
    if lead_time_table.empty:
        return output_path
    test_records = lead_time_table[lead_time_table["split"] == "test"]
    if test_records.empty:
        test_records = lead_time_table
    test_records = test_records[test_records["variant"] == "forecast_only"]
    if test_records.empty:
        return output_path
    horizons = sorted(test_records["horizon"].unique())
    figure, axes = plt.subplots(
        1, len(horizons), figsize=(4 * len(horizons), 4), sharey=True
    )
    if len(horizons) == 1:
        axes = [axes]
    for axis, horizon in zip(axes, horizons):
        rows = test_records[test_records["horizon"] == horizon]
        axis.hist(rows["lead_time_s"], bins=12, color="#f28e2b", alpha=0.8)
        axis.set_title(f"H = {horizon}")
        axis.set_xlabel("Lead time, c")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Число запусков (test)")
    figure.suptitle(
        "Распределение lead time для аномальных запусков (вариант forecast_only)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_score_timeline(
    causal_predictions: pd.DataFrame,
    event_times: dict[str, dict[str, float]],
    output_path: Path,
    horizon: int,
) -> Path:
    if causal_predictions.empty:
        return output_path
    horizon_predictions = causal_predictions[
        (causal_predictions["horizon"] == horizon)
        & (causal_predictions["variant"] == "forecast_only")
    ]
    if horizon_predictions.empty:
        return output_path
    sample = (
        horizon_predictions.groupby("scenario_template")
        .first()
        .reset_index()[["scenario_template", "run_id", "label_name"]]
    )
    if sample.empty:
        return output_path
    figure, axis = plt.subplots(figsize=(9, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, len(sample)))
    for color, record in zip(palette, sample.to_dict("records")):
        rows = horizon_predictions[
            horizon_predictions["run_id"] == record["run_id"]
        ].sort_values("history_time_s")
        axis.plot(
            rows["history_time_s"],
            rows["anomaly_score"],
            color=color,
            linewidth=1.4,
            label=(
                f"{record['scenario_template']} · {record['label_name']}"
            ),
        )
        event = event_times.get(record["run_id"], {})
        if event and not np.isnan(event.get("event_time_s", float("nan"))):
            axis.axvline(
                event["event_time_s"],
                color=color,
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
            )
    axis.set_xlabel("Время наблюдения t, c")
    axis.set_ylabel(f"Ŝ(t, H = {horizon})")
    axis.set_title(
        "Динамика упреждающего скора аномальности по одному запуску каждого сценария"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
