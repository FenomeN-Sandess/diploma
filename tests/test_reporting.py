import json
from pathlib import Path

import pandas as pd

from crowd_anomaly.reporting import build_project_report


def test_project_report_is_created_from_pipeline_outputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "demo"
    feature_dir = tmp_path / "features" / "demo"
    forecast_dir = tmp_path / "forecast" / "demo"
    anomaly_dir = tmp_path / "anomaly" / "demo"
    eda_dir = tmp_path / "eda" / "demo"
    models_dir = tmp_path / "models" / "demo"
    report_dir = tmp_path / "final_report" / "demo"

    for directory in [
        dataset_dir,
        feature_dir,
        forecast_dir,
        anomaly_dir,
        eda_dir / "tables",
        eda_dir / "figures",
        models_dir,
    ]:
        directory.mkdir(parents=True)

    _write_demo_inputs(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
        forecast_dir=forecast_dir,
        anomaly_dir=anomaly_dir,
        eda_dir=eda_dir,
        models_dir=models_dir,
    )

    result = build_project_report(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
        forecast_dir=forecast_dir,
        anomaly_dir=anomaly_dir,
        eda_dir=eda_dir,
        models_dir=models_dir,
        output_dir=report_dir,
    )

    report_path = result["report_path"]
    assert report_path.exists()
    assert (report_dir / "project_report_metadata.json").exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "Итоговый отчёт по pipeline" in report_text
    assert "EDA, то есть разведывательный анализ данных" in report_text
    assert "Что создаётся" in report_text
    assert "Кто использует" in report_text
    assert "Короткий словарь признаков" in report_text
    assert "Средняя скорость" in report_text
    assert "F1 anomaly test" in report_text


def _write_demo_inputs(
    dataset_dir: Path,
    feature_dir: Path,
    forecast_dir: Path,
    anomaly_dir: Path,
    eda_dir: Path,
    models_dir: Path,
) -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "normal__run_000",
                "scenario_template": "normal",
                "split": "train",
                "label": 0,
                "label_name": "normal",
                "agent_count": 10,
                "steps": 5,
                "duration_s": 0.5,
                "mean_speed": 1.0,
                "max_speed": 1.4,
                "mean_force": 0.2,
                "max_force": 1.0,
                "final_goal_reached_fraction": 0.8,
            },
            {
                "run_id": "anomaly__run_000",
                "scenario_template": "anomaly",
                "split": "test",
                "label": 1,
                "label_name": "anomaly",
                "agent_count": 10,
                "steps": 5,
                "duration_s": 0.5,
                "mean_speed": 0.8,
                "max_speed": 1.6,
                "mean_force": 0.7,
                "max_force": 3.0,
                "final_goal_reached_fraction": 0.2,
            },
        ]
    )
    runs.to_csv(dataset_dir / "runs.csv", index=False)
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": "demo",
                "synthetic_data": True,
                "model": "social_force_model",
                "run_count": 2,
                "runs_per_template": 1,
                "seed": 42,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (feature_dir / "feature_schema.json").write_text(
        json.dumps({"feature_columns": ["mean_speed", "mean_force"]}),
        encoding="utf-8",
    )

    _write_csv(eda_dir / "tables" / "label_counts.csv", ["label_name", "run_count"])
    _write_csv(eda_dir / "tables" / "split_counts.csv", ["split", "run_count"])
    _write_csv(
        eda_dir / "tables" / "scenario_summary.csv",
        ["scenario_template", "label_name", "run_count"],
    )
    _write_csv(
        eda_dir / "tables" / "trajectory_summary.csv",
        ["label_name", "run_count", "mean_speed_mean"],
    )
    _write_csv(
        eda_dir / "tables" / "feature_label_summary.csv",
        ["feature", "normal_mean", "anomaly_mean", "abs_difference"],
    )
    _write_csv(
        eda_dir / "tables" / "feature_summary.csv",
        ["feature", "mean", "std"],
    )
    pd.DataFrame(
        [
            {
                "feature": "mean_speed",
                "name": "Средняя скорость",
                "description": "средняя скорость агентов",
            },
            {
                "feature": "mean_force",
                "name": "Средняя сила",
                "description": "средняя сила взаимодействия",
            },
        ]
    ).to_csv(eda_dir / "tables" / "feature_dictionary.csv", index=False)
    _write_csv(
        eda_dir / "tables" / "outlier_summary.csv",
        ["feature", "outlier_fraction"],
    )
    _write_csv(
        eda_dir / "tables" / "top_correlations.csv",
        ["feature_a", "feature_b", "correlation"],
    )

    _write_csv(
        forecast_dir / "forecast_metrics_by_split.csv",
        ["model", "split", "mae"],
    )
    (forecast_dir / "forecast_metrics.json").write_text(
        json.dumps(
            {
                "window_size": 5,
                "horizon": 1,
                "target_columns": ["mean_x", "mean_y"],
                "row_count": 10,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(anomaly_dir / "unsupervised_metrics_by_split.csv", ["split", "f1"])
    _write_csv(
        anomaly_dir / "unsupervised_method_comparison.csv",
        ["method", "split", "f1"],
    )
    (anomaly_dir / "unsupervised_metrics.json").write_text(
        json.dumps({"anomaly_score_formula": "0.13 * pca + 0.87 * physics"}),
        encoding="utf-8",
    )
    _write_csv(models_dir / "metrics_by_split.csv", ["model", "split", "f1"])
    _write_csv(models_dir / "feature_importance.csv", ["feature", "importance"])
    (models_dir / "metrics.json").write_text(
        json.dumps({"best_model_by_val_f1": "logistic_regression"}),
        encoding="utf-8",
    )

def _write_csv(path: Path, columns: list[str]) -> None:
    data = {column: [f"{column}_value"] for column in columns}
    pd.DataFrame(data).to_csv(path, index=False)
