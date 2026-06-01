from pathlib import Path

from crowd_anomaly.anomaly import train_unsupervised_anomaly
from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.eda import build_eda_report
from crowd_anomaly.features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_eda_report_tables_and_figures_are_created(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    feature_dir = tmp_path / "features"
    anomaly_dir = tmp_path / "anomaly"
    eda_dir = tmp_path / "eda"

    build_dataset(
        config_path=PROJECT_ROOT / "configs" / "dataset.yaml",
        scenarios_path=PROJECT_ROOT / "configs" / "scenarios.yaml",
        output_dir=dataset_dir,
        seed=456,
        runs_per_template=1,
    )
    build_features(dataset_dir=dataset_dir, output_dir=feature_dir)
    train_unsupervised_anomaly(feature_dir=feature_dir, output_dir=anomaly_dir)
    build_eda_report(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
        output_dir=eda_dir,
        anomaly_dir=anomaly_dir,
    )

    report_path = eda_dir / "eda_report.md"
    assert report_path.exists()

    table_files = [
        "dataset_summary.csv",
        "scenario_summary.csv",
        "trajectory_summary.csv",
        "label_counts.csv",
        "split_counts.csv",
        "feature_dictionary.csv",
        "feature_summary.csv",
        "feature_label_summary.csv",
        "outlier_summary.csv",
        "top_correlations.csv",
        "anomaly_score_summary.csv",
    ]
    for file_name in table_files:
        assert (eda_dir / "tables" / file_name).exists()

    figure_files = list((eda_dir / "figures").glob("*.png"))
    assert len(figure_files) >= 3

    report_text = report_path.read_text(encoding="utf-8")
    assert "синтетические данные" in report_text
    assert "Короткий словарь признаков" in report_text
    assert "Максимум близких пар" in report_text
    assert "на уровне всего запуска" in report_text
    assert "точной эталонной разметкой" in report_text
    assert "Корреляция не доказывает причинность" in report_text
    assert "Показатель аномальности" in report_text
    assert (
        "![Распределение запусков по меткам](figures/label_distribution.png)"
        in report_text
    )
    assert (
        "![Распределение запусков по частям набора]"
        "(figures/split_distribution.png)" in report_text
    )
    assert (
        "![Распределение показателя аномальности]"
        "(figures/anomaly_score_distribution.png)" in report_text
    )
    assert (
        "![Примеры траекторий агентов](figures/trajectory_examples.png)"
        in report_text
    )
    assert (
        "![Диагностика подозрительных моментов во времени]"
        "(figures/trajectory_anomaly_timeline.png)" in report_text
    )
    assert "кандидатов на подозрительные моменты" in report_text
    assert "эталонная покадровая разметка" in report_text
    assert (eda_dir / "figures" / "trajectory_examples.png").exists()
    assert (eda_dir / "figures" / "trajectory_anomaly_timeline.png").exists()
    assert (eda_dir / "figures" / "run_metric_boxplots.png").exists()
    assert (eda_dir / "figures" / "anomaly_score_distribution.png").exists()

    (anomaly_dir / "local_anomaly_candidates.csv").unlink()
    eda_without_local_dir = tmp_path / "eda_without_local_candidates"
    build_eda_report(
        dataset_dir=dataset_dir,
        feature_dir=feature_dir,
        output_dir=eda_without_local_dir,
        anomaly_dir=anomaly_dir,
    )
    assert (
        eda_without_local_dir / "figures" / "trajectory_anomaly_timeline.png"
    ).exists()
