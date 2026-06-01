import json
from pathlib import Path

import pandas as pd

from crowd_anomaly.anomaly import train_unsupervised_anomaly
from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.features import FORBIDDEN_FEATURE_NAMES, build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unsupervised_anomaly_writes_artifacts_without_label_features(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    anomaly_dir = tmp_path / "anomaly"

    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=dataset_dir,
        seed=42,
        runs_per_template=1,
    )
    build_features(dataset_dir=dataset_dir, output_dir=feature_dir)
    train_unsupervised_anomaly(feature_dir=feature_dir, output_dir=anomaly_dir)

    expected_files = [
        "unsupervised_metrics.json",
        "unsupervised_metrics_by_split.csv",
        "unsupervised_predictions.csv",
        "unsupervised_method_comparison.csv",
        "local_anomaly_candidates.csv",
        "unsupervised_score_distribution.png",
    ]
    for file_name in expected_files:
        assert (anomaly_dir / file_name).exists()

    metrics = json.loads(
        (anomaly_dir / "unsupervised_metrics.json").read_text(encoding="utf-8")
    )
    metrics_by_split = pd.read_csv(anomaly_dir / "unsupervised_metrics_by_split.csv")
    method_comparison = pd.read_csv(anomaly_dir / "unsupervised_method_comparison.csv")
    predictions = pd.read_csv(anomaly_dir / "unsupervised_predictions.csv")
    local_candidates = pd.read_csv(anomaly_dir / "local_anomaly_candidates.csv")

    assert metrics["task"] == "unsupervised_anomaly_detection"
    assert metrics["training_uses_label_as_feature"] is False
    assert metrics["training_uses_labels_for_fit"] is False
    assert metrics["training_uses_label_to_select_reference"] is False
    assert metrics["training_reference"] == "all_train_runs"
    assert metrics["method_comparison_file"] == "unsupervised_method_comparison.csv"
    assert FORBIDDEN_FEATURE_NAMES.isdisjoint(set(metrics["feature_columns"]))
    assert {"anomaly_score", "predicted_label"} <= set(predictions.columns)
    assert metrics["model"] == (
        "Метод главных компонент + физический показатель аномальности"
    )
    assert {
        "pca_physics",
        "hybrid_iforest_physics",
        "physics_only",
    } <= set(method_comparison["method"])
    assert {"tp", "tn", "fp", "fn"} <= set(metrics_by_split.columns)
    assert {"frame_step", "frame_score", "top_agent_ids"} <= set(
        local_candidates.columns
    )
    assert set(metrics_by_split["class_count"]) == {2}
