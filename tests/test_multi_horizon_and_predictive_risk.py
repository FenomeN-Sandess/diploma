"""Smoke-тесты модулей multi_horizon и predictive_risk.

Проверяют, что код не падает на маленьком синтетическом наборе и
выдаёт ожидаемые артефакты (CSV-файлы и графики).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "outputs" / "datasets" / "demo"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features" / "demo"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_multi_horizon_smoke(tmp_path: Path) -> None:
    if not DATASET_DIR.exists():
        return
    from crowd_anomaly.multi_horizon import run_multi_horizon_forecast

    output_dir = tmp_path / "multi_horizon"
    run_multi_horizon_forecast(
        dataset_dir=DATASET_DIR,
        output_dir=output_dir,
        horizons=(1, 3, 5),
        window_size=5,
    )
    assert (output_dir / "multi_horizon_metrics.json").exists()
    assert (output_dir / "multi_horizon_mae.png").exists()
    summary = pd.read_csv(output_dir / "multi_horizon_metrics_overall.csv")
    expected_models = {"persistence", "direct"}
    assert set(summary["model"].unique()) == expected_models
    assert {1, 3, 5}.issubset(set(summary["horizon"].unique()))
    # direct должен побеждать persistence хотя бы на H=1
    test_one_step = summary[
        (summary["split"] == "test") & (summary["horizon"] == 1)
    ].set_index("model")
    assert test_one_step.loc["direct", "mae"] < test_one_step.loc[
        "persistence", "mae"
    ]


def test_predictive_risk_smoke(tmp_path: Path) -> None:
    if not DATASET_DIR.exists() or not FEATURE_DIR.exists():
        return
    from crowd_anomaly.predictive_risk import run_predictive_risk

    output_dir = tmp_path / "predictive_risk"
    run_predictive_risk(
        dataset_dir=DATASET_DIR,
        feature_dir=FEATURE_DIR,
        output_dir=output_dir,
        horizons=(1, 3),
        window_size=5,
    )
    assert (output_dir / "predictive_risk_metrics.json").exists()
    assert (output_dir / "predictive_risk_f1_curve.png").exists()
    metrics = pd.read_csv(output_dir / "predictive_risk_metrics_by_horizon.csv")
    assert {"forecast_only", "history_plus_forecast"}.issubset(
        set(metrics["variant"].unique())
    )
    # Должен быть хоть один TP с положительным lead time на тесте
    test_forecast = metrics[
        (metrics["split"] == "test") & (metrics["variant"] == "forecast_only")
    ]
    assert (test_forecast["detected_with_lead"].sum() > 0)
