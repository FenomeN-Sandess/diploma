from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SERIES_COLUMNS = ["mean_x", "mean_y", "mean_speed", "density_proxy"]
TARGET_COLUMNS = ["mean_x", "mean_y", "mean_speed"]
MODEL_RANDOM_STATE = 42


def train_forecast_models(
    dataset_dir: Path | str,
    output_dir: Path | str,
    window_size: int = 5,
    horizon: int = 1,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = build_forecast_samples(dataset_path, window_size, horizon)
    sequences = samples["sequences"]
    targets = samples["targets"]
    metadata = samples["metadata"]

    train_mask = metadata["split"] == "train"
    if not bool(train_mask.any()):
        raise ValueError("Нет обучающих окон для прогноза.")

    train_sequences = sequences[train_mask.to_numpy()]
    train_targets = targets[train_mask.to_numpy()]

    recurrent_model = fit_simple_recurrent_model(train_sequences, train_targets)
    predictions = pd.concat(
        [
            predict_persistence(sequences, targets, metadata),
            predict_simple_recurrent(recurrent_model, sequences, targets, metadata),
        ],
        ignore_index=True,
    )
    metrics_by_split = summarize_forecast_metrics(predictions)
    metrics_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "forecasting",
        "description": (
            "Прогнозирование будущего состояния толпы по предыдущим агрегированным "
            "состояниям."
        ),
        "dataset_dir": str(dataset_path),
        "window_size": int(window_size),
        "horizon": int(horizon),
        "series_columns": SERIES_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "models": {
            "persistence": (
                "Простое сравнение: следующее состояние равно последнему наблюдаемому."
            ),
            "simple_recurrent": (
                "Учебная рекуррентная модель: tanh-скрытое состояние по окну "
                "наблюдений и линейный выход."
            ),
        },
        "row_count": int(len(metadata)),
        "limitations": [
            "Модель прогнозирует агрегированные величины по всему запуску.",
            "Это учебная компактная модель, а не большая нейросетевая система.",
            "Локализация аномального агента не выполняется.",
        ],
    }

    predictions.to_csv(output_path / "forecast_predictions.csv", index=False)
    metrics_by_split.to_csv(output_path / "forecast_metrics_by_split.csv", index=False)
    (output_path / "forecast_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": output_path,
        "predictions": predictions,
        "metrics_by_split": metrics_by_split,
        "metrics": metrics_payload,
    }


def build_forecast_samples(
    dataset_dir: Path | str,
    window_size: int,
    horizon: int,
) -> dict[str, Any]:
    if window_size < 2:
        raise ValueError("window_size должен быть не меньше 2.")
    if horizon < 1:
        raise ValueError("horizon должен быть не меньше 1.")

    dataset_path = Path(dataset_dir)
    runs = pd.read_csv(dataset_path / "runs.csv")
    sequence_rows = []
    target_rows = []
    metadata_rows = []

    for run in runs.to_dict("records"):
        run_id = str(run["run_id"])
        trajectories = pd.read_csv(dataset_path / "runs" / run_id / "trajectories.csv")
        series = build_run_time_series(trajectories)
        max_start = len(series) - window_size - horizon + 1
        for start in range(max_start):
            end = start + window_size
            target_index = end + horizon - 1
            sequence_rows.append(series.loc[start : end - 1, SERIES_COLUMNS].to_numpy())
            target_rows.append(series.loc[target_index, TARGET_COLUMNS].to_numpy())
            metadata_rows.append(
                {
                    "sample_id": f"{run_id}__w{start:03d}",
                    "run_id": run_id,
                    "split": run["split"],
                    "label": run["label"],
                    "label_name": run["label_name"],
                    "start_step": int(series.loc[start, "step"]),
                    "target_step": int(series.loc[target_index, "step"]),
                }
            )

    return {
        "sequences": np.array(sequence_rows, dtype=float),
        "targets": np.array(target_rows, dtype=float),
        "metadata": pd.DataFrame(metadata_rows),
    }


def build_run_time_series(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, frame in trajectories.groupby("step", sort=True):
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        rows.append(
            {
                "step": int(step),
                "mean_x": float(frame["x"].mean()),
                "mean_y": float(frame["y"].mean()),
                "mean_speed": float(frame["speed"].mean()),
                "density_proxy": _density_proxy(coordinates),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_simple_recurrent_model(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    hidden_size: int = 8,
    alpha: float = 1e-3,
) -> dict[str, Any]:
    mean = train_sequences.mean(axis=(0, 1))
    std = train_sequences.std(axis=(0, 1))
    std = np.where(std < 1e-9, 1.0, std)
    normalized = (train_sequences - mean) / std

    weights = _fixed_recurrent_weights(
        input_size=train_sequences.shape[2],
        hidden_size=hidden_size,
    )
    hidden = _recurrent_hidden(normalized, weights)
    design = _forecast_design(hidden, train_sequences)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    output_weights = (
        np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_targets
    )

    return {
        "mean": mean,
        "std": std,
        "weights": weights,
        "output_weights": output_weights,
    }


def predict_persistence(
    sequences: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    last_state = sequences[:, -1, : len(TARGET_COLUMNS)]
    return _prediction_frame(
        metadata=metadata,
        targets=targets,
        predictions=last_state,
        model_name="persistence",
    )


def predict_simple_recurrent(
    model: dict[str, Any],
    sequences: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    normalized = (sequences - model["mean"]) / model["std"]
    hidden = _recurrent_hidden(normalized, model["weights"])
    design = _forecast_design(hidden, sequences)
    predictions = design @ model["output_weights"]
    return _prediction_frame(
        metadata=metadata,
        targets=targets,
        predictions=predictions,
        model_name="simple_recurrent",
    )


def summarize_forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), group in predictions.groupby(["model", "split"], sort=True):
        rows.append(
            {
                "model": model,
                "split": split,
                "mae": float(group["absolute_error"].mean()),
                "mse": float(group["squared_error"].mean()),
                "sample_count": int(group["sample_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _fixed_recurrent_weights(
    input_size: int,
    hidden_size: int,
) -> dict[str, np.ndarray]:
    random = np.random.default_rng(MODEL_RANDOM_STATE)
    return {
        "input": random.normal(
            0.0,
            0.6 / np.sqrt(input_size),
            (input_size, hidden_size),
        ),
        "hidden": random.normal(
            0.0,
            0.4 / np.sqrt(hidden_size),
            (hidden_size, hidden_size),
        ),
        "bias": np.zeros(hidden_size, dtype=float),
    }


def _recurrent_hidden(
    sequences: np.ndarray,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    hidden = np.zeros((len(sequences), weights["hidden"].shape[0]), dtype=float)
    for step_index in range(sequences.shape[1]):
        current = sequences[:, step_index, :]
        hidden = np.tanh(
            current @ weights["input"]
            + hidden @ weights["hidden"]
            + weights["bias"]
        )
    return hidden


def _forecast_design(hidden: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    last_values = sequences[:, -1, : len(TARGET_COLUMNS)]
    return np.column_stack([np.ones(len(hidden)), hidden, last_values])


def _prediction_frame(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for row_index, meta in metadata.reset_index(drop=True).iterrows():
        for column_index, target_name in enumerate(TARGET_COLUMNS):
            actual = float(targets[row_index, column_index])
            predicted = float(predictions[row_index, column_index])
            error = actual - predicted
            rows.append(
                {
                    **meta.to_dict(),
                    "model": model_name,
                    "target": target_name,
                    "actual": actual,
                    "predicted": predicted,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                }
            )
    return pd.DataFrame(rows)


def _density_proxy(coordinates: np.ndarray) -> float:
    agent_count = len(coordinates)
    if agent_count < 2:
        return 0.0
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(offsets**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.mean(1.0 / (nearest + 1e-9)))
