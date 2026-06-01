import json
from pathlib import Path

import pandas as pd

from crowd_anomaly.baselines import train_baselines
from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.features import FORBIDDEN_FEATURE_NAMES, build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_baselines_train_and_write_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    model_dir = tmp_path / "models"

    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=dataset_dir,
        seed=42,
        runs_per_template=2,
    )
    build_features(dataset_dir=dataset_dir, output_dir=feature_dir)
    train_baselines(feature_dir=feature_dir, output_dir=model_dir)

    expected_files = [
        "metrics.json",
        "metrics_by_split.csv",
        "predictions.csv",
        "feature_importance.csv",
        "score_distribution.png",
    ]
    for file_name in expected_files:
        assert (model_dir / file_name).exists()

    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(model_dir / "predictions.csv")
    metrics_by_split = pd.read_csv(model_dir / "metrics_by_split.csv")
    feature_importance = pd.read_csv(model_dir / "feature_importance.csv")
    feature_schema = json.loads(
        (feature_dir / "feature_schema.json").read_text(encoding="utf-8")
    )

    assert metrics["synthetic_data"] is True
    assert {"logistic_regression", "random_forest"} <= set(metrics["models"])
    assert not predictions.empty
    assert not metrics_by_split.empty
    assert not feature_importance.empty
    assert FORBIDDEN_FEATURE_NAMES.isdisjoint(
        set(feature_schema["feature_columns"])
    )
