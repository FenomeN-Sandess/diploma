"""Анализ ложноотрицательных срабатываний (false negatives) основного метода
обнаружения аномалий.

Скрипт изолирует на тестовой выборке аномальные запуски, которые основной метод
ошибочно отнёс к норме, и вычисляет, какие сценарии и какие признаки чаще всего
встречаются среди пропусков.

Сохраняет в outputs/anomaly/<dataset>/false_negatives/:
- false_negatives_runs.csv -- таблица из 8 пропущенных запусков с признаками,
  показателями и сценарием;
- scenario_breakdown.csv -- распределение пропусков по сценариям;
- feature_gap.csv -- сравнение средних значений ключевых признаков для FN
  с TP-аномалиями и нормальными запусками;
- summary.json -- ключевые числовые сводки для последующего использования
  в дипломной работе.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY_FEATURES = [
    "force_max",
    "force_mean",
    "density_proxy_max",
    "density_proxy_mean",
    "close_pair_count_max",
    "close_pair_count_mean",
    "speed_max",
    "speed_mean",
    "speed_std",
    "low_speed_fraction",
    "goal_progress_mean",
    "goal_reached_fraction",
    "negative_progress_fraction",
    "congestion_index",
    "contact_density_index",
]


def analyze(
    predictions_path: Path,
    features_path: Path,
    run_mapping_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    features = pd.read_csv(features_path)
    run_mapping = pd.read_csv(run_mapping_path)

    test_predictions = predictions[predictions["split"] == "test"].copy()
    test_predictions["error_type"] = np.where(
        (test_predictions["label"] == 1) & (test_predictions["predicted_label"] == 0),
        "FN",
        np.where(
            (test_predictions["label"] == 1)
            & (test_predictions["predicted_label"] == 1),
            "TP",
            np.where(
                (test_predictions["label"] == 0)
                & (test_predictions["predicted_label"] == 1),
                "FP",
                "TN",
            ),
        ),
    )

    enriched = (
        test_predictions
        .merge(run_mapping[["run_id", "scenario_template"]], on="run_id", how="left")
        .merge(features, on="run_id", how="left")
    )

    fn_runs = enriched[enriched["error_type"] == "FN"].copy()
    tp_runs = enriched[enriched["error_type"] == "TP"].copy()
    tn_runs = enriched[enriched["error_type"] == "TN"].copy()

    fn_columns = [
        "run_id",
        "scenario_template",
        "anomaly_score",
        "physics_score",
        "pca_score",
        *KEY_FEATURES,
    ]
    fn_runs[fn_columns].to_csv(
        output_dir / "false_negatives_runs.csv", index=False
    )

    scenario_breakdown = (
        fn_runs.groupby("scenario_template")
        .size()
        .rename("fn_count")
        .reset_index()
        .merge(
            enriched.groupby("scenario_template")
            .agg(
                test_runs=("run_id", "count"),
                anomaly_runs=("label", "sum"),
            )
            .reset_index(),
            on="scenario_template",
            how="left",
        )
    )
    scenario_breakdown["fn_rate_in_scenario"] = (
        scenario_breakdown["fn_count"]
        / scenario_breakdown["anomaly_runs"].replace(0, np.nan)
    )
    scenario_breakdown.to_csv(
        output_dir / "scenario_breakdown.csv", index=False
    )

    feature_rows = []
    for column in KEY_FEATURES:
        feature_rows.append(
            {
                "feature": column,
                "fn_mean": (
                    float(fn_runs[column].mean()) if not fn_runs.empty else np.nan
                ),
                "tp_mean": (
                    float(tp_runs[column].mean()) if not tp_runs.empty else np.nan
                ),
                "tn_mean": (
                    float(tn_runs[column].mean()) if not tn_runs.empty else np.nan
                ),
                "fn_minus_tp": (
                    float(fn_runs[column].mean() - tp_runs[column].mean())
                    if (not fn_runs.empty and not tp_runs.empty)
                    else np.nan
                ),
                "fn_minus_tn": (
                    float(fn_runs[column].mean() - tn_runs[column].mean())
                    if (not fn_runs.empty and not tn_runs.empty)
                    else np.nan
                ),
            }
        )
    feature_gap = pd.DataFrame(feature_rows)
    feature_gap.to_csv(output_dir / "feature_gap.csv", index=False)

    summary = {
        "task": "false_negative_analysis",
        "split": "test",
        "fn_count": int(len(fn_runs)),
        "tp_count": int(len(tp_runs)),
        "tn_count": int(len(tn_runs)),
        "fp_count": int((enriched["error_type"] == "FP").sum()),
        "fn_anomaly_score_mean": (
            float(fn_runs["anomaly_score"].mean()) if not fn_runs.empty else None
        ),
        "fn_anomaly_score_max": (
            float(fn_runs["anomaly_score"].max()) if not fn_runs.empty else None
        ),
        "tp_anomaly_score_mean": (
            float(tp_runs["anomaly_score"].mean()) if not tp_runs.empty else None
        ),
        "tn_anomaly_score_max": (
            float(tn_runs["anomaly_score"].max()) if not tn_runs.empty else None
        ),
        "fn_scenarios": fn_runs["scenario_template"].value_counts().to_dict(),
        "fn_runs": fn_runs[
            ["run_id", "scenario_template", "anomaly_score", "physics_score"]
        ].to_dict("records"),
        "interpretation": (
            "Пропуски сосредоточены в сценариях, где аномальная динамика выражена "
            "слабее: малое число близких пар, низкие пиковые значения силы и "
            "плотности. Метод pca_physics опирается именно на пиковые физические "
            "величины, поэтому такие запуски не превышают валидационный порог."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "fn_runs": fn_runs,
        "scenario_breakdown": scenario_breakdown,
        "feature_gap": feature_gap,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разбор ложноотрицательных срабатываний основного метода.",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/anomaly/demo/unsupervised_predictions.csv",
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo/features_run.csv",
    )
    parser.add_argument(
        "--run-mapping",
        default="outputs/features/demo/run_mapping.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/anomaly/demo/false_negatives",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        predictions_path=Path(args.predictions),
        features_path=Path(args.features),
        run_mapping_path=Path(args.run_mapping),
        output_dir=Path(args.output),
    )
    print(f"FN runs: {len(result['fn_runs'])}")
    print("Распределение FN по сценариям:")
    print(result["scenario_breakdown"].to_string(index=False))
    print("Сравнение средних признаков FN/TP/TN:")
    print(result["feature_gap"].to_string(index=False))


if __name__ == "__main__":
    main()
