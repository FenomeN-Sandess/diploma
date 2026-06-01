import json
from pathlib import Path

import numpy as np
import pandas as pd

from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.features import (
    FORBIDDEN_FEATURE_NAMES,
    build_features,
    load_feature_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_features_creates_artifacts_without_leakage(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=dataset_dir,
        seed=321,
        runs_per_template=1,
    )

    build_features(dataset_dir=dataset_dir, output_dir=feature_dir)

    expected_files = [
        "features_run.csv",
        "run_mapping.csv",
        "feature_schema.json",
        "features_metadata.json",
    ]
    for file_name in expected_files:
        assert (feature_dir / file_name).exists()

    artifacts = load_feature_artifacts(feature_dir)
    features_run = artifacts["features_run"]
    run_mapping = artifacts["run_mapping"]
    feature_schema = artifacts["feature_schema"]
    metadata = artifacts["metadata"]
    feature_columns = feature_schema["feature_columns"]

    assert "run_id" in features_run.columns
    assert "run_id" not in feature_columns
    assert FORBIDDEN_FEATURE_NAMES.isdisjoint(set(feature_columns))
    assert {"label", "split"} <= set(run_mapping.columns)
    assert metadata["leakage_checked"] is True

    feature_matrix = features_run[feature_columns]
    assert all(
        pd.api.types.is_numeric_dtype(feature_matrix[column])
        for column in feature_columns
    )
    assert np.isfinite(feature_matrix.to_numpy(dtype=float)).all()


def test_feature_schema_json_contains_leakage_policy(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=dataset_dir,
        seed=123,
        runs_per_template=1,
    )
    build_features(dataset_dir=dataset_dir, output_dir=feature_dir)

    schema = json.loads(
        (feature_dir / "feature_schema.json").read_text(encoding="utf-8")
    )

    assert "run_id" in schema["index_columns"]
    assert "run_id" not in schema["feature_columns"]
    assert "label" in schema["excluded_columns"]

    feature_details = {
        item["feature"]: item
        for item in schema["feature_details"]
    }
    assert feature_details["close_pair_count_max"]["name"] == "Максимум близких пар"
    assert "близких пар" in feature_details["close_pair_count_max"]["description"]
