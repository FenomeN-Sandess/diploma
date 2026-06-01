import json
from pathlib import Path

import pandas as pd

from crowd_anomaly.dataset import FRAME_LABEL_SOURCE, build_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_dataset_creates_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=output_dir,
        seed=123,
        runs_per_template=5,
    )

    metadata_path = output_dir / "dataset_metadata.json"
    runs_path = output_dir / "runs.csv"
    frames_path = output_dir / "frames.csv"

    assert metadata_path.exists()
    assert runs_path.exists()
    assert frames_path.exists()

    runs = pd.read_csv(runs_path)
    frames = pd.read_csv(frames_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert runs["run_id"].is_unique
    assert {"normal", "anomaly"} <= set(runs["label_name"])
    assert not runs["split"].isna().any()
    assert "" not in set(runs["split"].astype(str))
    for split_name, split_frame in runs.groupby("split"):
        assert {"normal", "anomaly"} <= set(split_frame["label_name"]), split_name
    assert set(frames["frame_label_source"]) == {FRAME_LABEL_SOURCE}
    assert metadata["synthetic_data"] is True
    assert metadata["local_anomaly_ground_truth"] == "not_available"

    for run_id in runs["run_id"]:
        run_dir = output_dir / "runs" / run_id
        assert run_dir.is_dir()
        assert (run_dir / "trajectories.csv").exists()
        assert (run_dir / "run_metrics.json").exists()
