"""Многошаговый прогноз агрегированной динамики толпы.

Реализует направление 1.1 из `Разбор/перспективы_развития.md`:

* строит оператор F_theta, переводящий окно прошлых состояний
  z_{t-k+1..t} в траекторию ẑ_{t+1..t+H};
* сравнивает две стратегии прогноза на горизонте H:
  1) persistence (повтор последнего состояния);
  2) direct — отдельные регрессоры на каждый горизонт H,
     каждый выдаёт состояние сразу на h шагов вперёд из исходного окна;
* считает MAE/MSE по каждому H и сохраняет таблицы + графики
  деградации точности.

Модель F_theta — расширение `forecasting.fit_simple_recurrent_model`
на полноразмерный вектор состояния (см. EXTENDED_SERIES_COLUMNS),
необходимый для направления 1.2: пиковые признаки $f_{\\max}, c, \\bar f$
входят в физический штраф P(R).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crowd_anomaly.forecasting import (
    _fixed_recurrent_weights,
    _recurrent_hidden,
)

EXTENDED_SERIES_COLUMNS = [
    "mean_x",
    "mean_y",
    "mean_speed",
    "density_proxy",
    "mean_force",
    "max_force",
    "close_pair_count",
]
EXTENDED_TARGET_COLUMNS = EXTENDED_SERIES_COLUMNS
DEFAULT_WINDOW = 5
DEFAULT_HORIZONS = (1, 2, 3, 5, 7, 10, 15)
HIDDEN_SIZE = 12
RIDGE_ALPHA = 1e-3
CLOSE_PAIR_DISTANCE = 0.75
EPSILON = 1e-9


def run_multi_horizon_forecast(
    dataset_dir: Path | str,
    output_dir: Path | str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    window_size: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Главный сценарий: обучение, оценка и сохранение результатов."""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in sorted(set(horizons)))

    runs = pd.read_csv(dataset_path / "runs.csv")
    series_by_run = _load_extended_series(dataset_path, runs)

    max_horizon = max(horizons)
    train_runs = set(runs.loc[runs["split"] == "train", "run_id"].astype(str))

    direct_models: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        train_sequences, train_targets = _build_training_pairs(
            series_by_run=series_by_run,
            run_ids=train_runs,
            window_size=window_size,
            horizon=horizon,
        )
        if len(train_sequences) == 0:
            continue
        direct_models[horizon] = _fit_recurrent_regressor(
            train_sequences=train_sequences,
            train_targets=train_targets,
        )
    if not direct_models:
        raise ValueError("Нет обучающих окон ни для одного горизонта.")

    prediction_rows: list[dict[str, Any]] = []
    for run in runs.to_dict("records"):
        run_id = str(run["run_id"])
        series = series_by_run.get(run_id)
        if series is None or len(series) < window_size + 1:
            continue
        values = series[EXTENDED_SERIES_COLUMNS].to_numpy(dtype=float)
        steps = series["step"].to_numpy(dtype=int)
        max_start = len(values) - window_size - max_horizon + 1
        if max_start <= 0:
            continue
        for start in range(max_start):
            window = values[start : start + window_size]
            history_step = int(steps[start + window_size - 1])
            persistence_state = window[-1].copy()
            for horizon in horizons:
                target_index = start + window_size + horizon - 1
                if target_index >= len(values):
                    continue
                actual = values[target_index]
                step_target = int(steps[target_index])
                _append_prediction_rows(
                    prediction_rows=prediction_rows,
                    run=run,
                    run_id=run_id,
                    history_step=history_step,
                    target_step=step_target,
                    horizon=horizon,
                    actual=actual,
                    predicted=persistence_state,
                    model_name="persistence",
                )
                direct_model = direct_models.get(horizon)
                if direct_model is None:
                    continue
                predicted = _predict_with_model(direct_model, window[np.newaxis])[0]
                _append_prediction_rows(
                    prediction_rows=prediction_rows,
                    run=run,
                    run_id=run_id,
                    history_step=history_step,
                    target_step=step_target,
                    horizon=horizon,
                    actual=actual,
                    predicted=predicted,
                    model_name="direct",
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics_by_split = _summarize_metrics(predictions)
    metrics_overall = _summarize_overall(predictions)

    predictions.to_csv(output_path / "multi_horizon_predictions.csv", index=False)
    metrics_by_split.to_csv(
        output_path / "multi_horizon_metrics_by_split.csv",
        index=False,
    )
    metrics_overall.to_csv(
        output_path / "multi_horizon_metrics_overall.csv",
        index=False,
    )

    plots = {
        "mae": _plot_degradation_curve(
            metrics_by_split=metrics_by_split,
            metric="mae",
            output_path=output_path / "multi_horizon_mae.png",
        ),
        "mse": _plot_degradation_curve(
            metrics_by_split=metrics_by_split,
            metric="mse",
            output_path=output_path / "multi_horizon_mse.png",
        ),
        "per_target": _plot_per_target_curve(
            metrics_by_split=metrics_by_split,
            metric="mae",
            output_path=output_path / "multi_horizon_mae_per_target.png",
        ),
    }

    metrics_payload = {
        "task": "multi_horizon_forecasting",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_path),
        "window_size": int(window_size),
        "horizons": list(horizons),
        "series_columns": EXTENDED_SERIES_COLUMNS,
        "target_columns": EXTENDED_TARGET_COLUMNS,
        "strategies": {
            "persistence": (
                "Базовый прогноз: повторение последнего наблюдаемого состояния "
                "на каждом горизонте H."
            ),
            "direct": (
                "Отдельный рекуррентный регрессор обучается для каждого H, "
                "выход — одно состояние на H шагов вперёд."
            ),
        },
        "models_count": {
            "direct": len(direct_models),
        },
        "row_count": int(len(predictions) / max(len(EXTENDED_TARGET_COLUMNS), 1)),
        "limitations": [
            "Состояние z_t — глобально-агрегированное по всему запуску.",
            "Direct-стратегия требует обучения отдельной модели на каждый H.",
        ],
        "files": {
            "predictions": "multi_horizon_predictions.csv",
            "metrics_by_split": "multi_horizon_metrics_by_split.csv",
            "metrics_overall": "multi_horizon_metrics_overall.csv",
            "mae_curve": "multi_horizon_mae.png",
            "mse_curve": "multi_horizon_mse.png",
            "per_target_curve": "multi_horizon_mae_per_target.png",
        },
    }
    (output_path / "multi_horizon_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": output_path,
        "predictions": predictions,
        "metrics_by_split": metrics_by_split,
        "metrics_overall": metrics_overall,
        "metrics": metrics_payload,
        "direct_models": direct_models,
        "series_by_run": series_by_run,
        "plots": plots,
    }


def build_extended_run_time_series(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Покадровый ряд расширенного состояния агрегированной динамики."""
    rows = []
    for step, frame in trajectories.groupby("step", sort=True):
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        forces = frame.get("force_norm", pd.Series([0.0] * len(frame)))
        force_values = forces.astype(float).to_numpy()
        nearest = _nearest_distances(coordinates)
        density_proxy = (
            float(np.mean(1.0 / (nearest + EPSILON))) if nearest.size else 0.0
        )
        close_pair_count = _count_close_pairs(coordinates)
        rows.append(
            {
                "step": int(step),
                "mean_x": float(frame["x"].mean()),
                "mean_y": float(frame["y"].mean()),
                "mean_speed": float(frame["speed"].mean()),
                "density_proxy": density_proxy,
                "mean_force": (
                    float(np.mean(force_values)) if force_values.size else 0.0
                ),
                "max_force": (
                    float(np.max(force_values)) if force_values.size else 0.0
                ),
                "close_pair_count": float(close_pair_count),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_extended_series(
    dataset_path: Path,
    runs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    series_by_run: dict[str, pd.DataFrame] = {}
    for run_id in runs["run_id"].astype(str):
        path = dataset_path / "runs" / run_id / "trajectories.csv"
        if not path.exists():
            continue
        trajectories = pd.read_csv(path)
        series_by_run[run_id] = build_extended_run_time_series(trajectories)
    return series_by_run


def _build_training_pairs(
    series_by_run: dict[str, pd.DataFrame],
    run_ids: set[str],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for run_id, series in series_by_run.items():
        if run_id not in run_ids:
            continue
        values = series[EXTENDED_SERIES_COLUMNS].to_numpy(dtype=float)
        max_start = len(values) - window_size - horizon + 1
        for start in range(max_start):
            window = values[start : start + window_size]
            target = values[start + window_size + horizon - 1]
            sequences.append(window)
            targets.append(target)
    return (
        np.array(sequences, dtype=float),
        np.array(targets, dtype=float),
    )


def _fit_recurrent_regressor(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, Any]:
    feature_count = train_sequences.shape[2]
    output_count = train_targets.shape[1]
    mean = train_sequences.mean(axis=(0, 1))
    std = train_sequences.std(axis=(0, 1))
    std = np.where(std < 1e-9, 1.0, std)
    normalized = (train_sequences - mean) / std

    weights = _fixed_recurrent_weights(
        input_size=feature_count,
        hidden_size=HIDDEN_SIZE,
    )
    hidden = _recurrent_hidden(normalized, weights)
    design = _design_matrix(hidden=hidden, sequences=train_sequences)
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    output_weights = (
        np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_targets
    )

    return {
        "mean": mean,
        "std": std,
        "weights": weights,
        "output_weights": output_weights,
        "output_count": int(output_count),
        "feature_count": int(feature_count),
    }


def _predict_with_model(
    model: dict[str, Any],
    sequences: np.ndarray,
) -> np.ndarray:
    normalized = (sequences - model["mean"]) / model["std"]
    hidden = _recurrent_hidden(normalized, model["weights"])
    design = _design_matrix(hidden=hidden, sequences=sequences)
    return design @ model["output_weights"]


def direct_rollout(
    direct_models: dict[int, dict[str, Any]],
    window: np.ndarray,
    max_horizon: int,
) -> np.ndarray:
    """Развёртка прогноза по H direct-моделям: на каждый h своя модель.

    Используется и в `multi_horizon`, и в `predictive_risk` для построения
    траектории ẑ_{t+1..t+H} без накопления ошибки.
    """
    n_features = window.shape[1]
    path = np.zeros((max_horizon, n_features), dtype=float)
    last = window[-1].copy()
    for h in range(1, max_horizon + 1):
        model = direct_models.get(h)
        if model is not None:
            last = _predict_with_model(model, window[np.newaxis])[0]
        path[h - 1] = last
    return path


def _design_matrix(hidden: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    last_values = sequences[:, -1, :]
    return np.column_stack([np.ones(len(hidden)), hidden, last_values])


def _append_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    run: dict[str, Any],
    run_id: str,
    history_step: int,
    target_step: int,
    horizon: int,
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
) -> None:
    for column_index, target_name in enumerate(EXTENDED_TARGET_COLUMNS):
        actual_value = float(actual[column_index])
        predicted_value = float(predicted[column_index])
        error = actual_value - predicted_value
        prediction_rows.append(
            {
                "run_id": run_id,
                "split": run["split"],
                "label": int(run["label"]),
                "label_name": run["label_name"],
                "scenario_template": run["scenario_template"],
                "horizon": int(horizon),
                "history_step": int(history_step),
                "target_step": int(target_step),
                "target": target_name,
                "actual": actual_value,
                "predicted": predicted_value,
                "absolute_error": abs(error),
                "squared_error": error**2,
                "model": model_name,
            }
        )


def _summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "model", "split", "horizon", "target", "mae", "mse", "sample_count",
            ]
        )
    grouped = predictions.groupby(
        ["model", "split", "horizon", "target"], sort=True
    )
    summary = grouped.agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        sample_count=("absolute_error", "size"),
    ).reset_index()
    return summary


def _summarize_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=["model", "split", "horizon", "mae", "mse", "sample_count"]
        )
    grouped = predictions.groupby(["model", "split", "horizon"], sort=True)
    summary = grouped.agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        sample_count=("absolute_error", "size"),
    ).reset_index()
    return summary


def _plot_degradation_curve(
    metrics_by_split: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> Path:
    overall = metrics_by_split.groupby(
        ["model", "split", "horizon"], as_index=False
    ).agg({metric: "mean"})
    splits = ["train", "val", "test"]
    palette = {
        "persistence": "#9e9e9e",
        "direct": "#4c78a8",
    }
    model_labels = {"persistence": "перенос", "direct": "прогноз"}
    split_labels = {"train": "обучение", "val": "валидация", "test": "тест"}
    line_styles = {"train": "-", "val": "--", "test": "-."}
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, color in palette.items():
        for split in splits:
            mask = (overall["model"] == model_name) & (overall["split"] == split)
            if not mask.any():
                continue
            data = overall[mask].sort_values("horizon")
            axis.plot(
                data["horizon"],
                data[metric],
                marker="o",
                color=color,
                linestyle=line_styles[split],
                linewidth=1.6,
                label=f"{model_labels[model_name]} · {split_labels[split]}",
            )
    axis.set_xlabel("Горизонт прогноза H, шагов")
    metric_title = "MAE" if metric == "mae" else "MSE"
    axis.set_ylabel(metric_title)
    axis.set_title(
        f"Деградация точности прогноза по горизонту: {metric_title}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_per_target_curve(
    metrics_by_split: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> Path:
    test_metrics = metrics_by_split[metrics_by_split["split"] == "test"]
    if test_metrics.empty:
        return output_path
    figure, axis = plt.subplots(figsize=(9, 5))
    palette_targets = plt.cm.tab10(
        np.linspace(0, 1, len(EXTENDED_TARGET_COLUMNS))
    )
    for target_name, color in zip(EXTENDED_TARGET_COLUMNS, palette_targets):
        mask = (test_metrics["target"] == target_name) & (
            test_metrics["model"] == "direct"
        )
        if not mask.any():
            continue
        data = test_metrics[mask].sort_values("horizon")
        axis.plot(
            data["horizon"],
            data[metric],
            marker="o",
            color=color,
            linewidth=1.4,
            label=target_name,
        )
    axis.set_xlabel("Горизонт прогноза H, шагов")
    axis.set_ylabel("MAE по тестовой выборке")
    axis.set_title(
        "Поканальная MAE direct-прогноза по горизонту (test)"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _nearest_distances(coordinates: np.ndarray) -> np.ndarray:
    agent_count = len(coordinates)
    if agent_count < 2:
        return np.array([0.0])
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(offsets**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def _count_close_pairs(coordinates: np.ndarray) -> int:
    agent_count = len(coordinates)
    if agent_count < 2:
        return 0
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(offsets**2, axis=2))
    pair_indices = np.triu_indices(agent_count, k=1)
    pair_distances = distances[pair_indices]
    return int(np.sum(pair_distances < CLOSE_PAIR_DISTANCE))
