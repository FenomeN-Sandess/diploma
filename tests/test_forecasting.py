import json
from pathlib import Path

import pandas as pd

from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.forecasting import train_forecast_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_forecast_models_write_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    forecast_dir = tmp_path / "forecast"
    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=dataset_dir,
        seed=42,
        runs_per_template=1,
    )

    train_forecast_models(dataset_dir=dataset_dir, output_dir=forecast_dir)

    expected_files = [
        "forecast_metrics.json",
        "forecast_metrics_by_split.csv",
        "forecast_predictions.csv",
    ]
    for file_name in expected_files:
        assert (forecast_dir / file_name).exists()

    metrics = json.loads(
        (forecast_dir / "forecast_metrics.json").read_text(encoding="utf-8")
    )
    metrics_by_split = pd.read_csv(forecast_dir / "forecast_metrics_by_split.csv")
    predictions = pd.read_csv(forecast_dir / "forecast_predictions.csv")

    assert metrics["task"] == "forecasting"
    assert {"persistence", "simple_recurrent"} <= set(metrics["models"])
    assert {"mae", "mse"} <= set(metrics_by_split.columns)
    assert {"actual", "predicted", "target"} <= set(predictions.columns)
    assert not predictions.empty
