"""Leave-one-scenario-out (LOSO) оценка устойчивости методов
обнаружения аномалий к новым типам сценариев.

Каждый из четырёх аномальных сценариев поочерёдно изымается из обучения и
используется только для теста. Это даёт оценку, показывающую, способен ли
метод распознавать аномалии тех типов, которые не встречались в обучающей
выборке.

Прогоняются методы:
- pca_physics, pca_reconstruction (без учителя),
- случайный лес (с учителем) -- для демонстрации разрыва между качеством на
  случайном разбиении и качеством на новых типах.

Результаты сохраняются в outputs/anomaly/<dataset>/loso/:
- loso_metrics.csv -- по каждому фолду и методу;
- loso_aggregated.csv -- среднее и стандартное отклонение по фолдам;
- loso_metadata.json -- параметры эксперимента.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crowd_anomaly.anomaly import (  # noqa: E402
    PCA_SCORE_WEIGHT,
    _binary_metrics,
    load_unsupervised_training_data,
    select_threshold_on_validation,
)
from scripts.run_anomaly_seeds import _pca_score, _physics_score  # noqa: E402

ANOMALY_SCENARIOS = (
    "anomaly_counterflow",
    "anomaly_bottleneck",
    "anomaly_high_density",
    "anomaly_local_violator",
)
# Исследование обобщения проводится на контролируемом наборе канонических типов
# аномалий: по одному типу за раз изымается из обучения и проверяется на тесте.
# Сценарии с отложенным наступлением события относятся к экспериментам прогноза
# на горизонт H и упреждающего сигнала (направления 1.1/1.2) и в исследование
# обобщения по типам не включаются, чтобы тест на новый тип оставался чистым:
# при наличии родственного отложенного сценария в обучении проверка перестаёт
# быть проверкой на ранее не встречавшийся тип.
CANONICAL_NORMAL_SCENARIOS = (
    "normal_corridor",
    "normal_room",
    "normal_wide_exit",
)
CANONICAL_SCENARIOS = CANONICAL_NORMAL_SCENARIOS + ANOMALY_SCENARIOS
TEST_NORMAL_FRACTION = 0.34
VAL_FRACTION = 0.20


def build_loso_split(
    data: pd.DataFrame,
    held_out_scenario: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    held_mask = data["scenario_template"] == held_out_scenario
    test_anomaly = data[held_mask].copy()

    normals = data[data["label"] == 0].copy().reset_index(drop=True)
    n_test_normals = int(round(len(normals) * TEST_NORMAL_FRACTION))
    test_normal_idx = rng.choice(len(normals), size=n_test_normals, replace=False)
    test_normals = normals.iloc[test_normal_idx].copy()
    pool_normals = normals.drop(normals.index[test_normal_idx]).copy()

    other_anomaly = data[(data["label"] == 1) & ~held_mask].copy()
    train_pool = pd.concat([pool_normals, other_anomaly], axis=0).reset_index(drop=True)

    val_size = int(round(len(train_pool) * VAL_FRACTION))
    val_idx = rng.choice(len(train_pool), size=val_size, replace=False)
    val_part = train_pool.iloc[val_idx].copy()
    train_part = train_pool.drop(train_pool.index[val_idx]).copy()

    test_part = pd.concat([test_anomaly, test_normals], axis=0).reset_index(drop=True)

    train_part["split"] = "train"
    val_part["split"] = "val"
    test_part["split"] = "test"

    full = pd.concat(
        [train_part, val_part, test_part], axis=0, ignore_index=True
    )
    return full


def evaluate_unsupervised(
    full_data: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> dict[str, dict[str, float | int]]:
    train_data = full_data[full_data["split"] == "train"]
    physics = _physics_score(full_data, train_data)
    pca = _pca_score(full_data, train_data, feature_columns, seed=seed)
    pca_physics = (
        PCA_SCORE_WEIGHT * pca + (1.0 - PCA_SCORE_WEIGHT) * physics
    )

    test_mask = (full_data["split"] == "test").to_numpy()
    y_true = full_data.loc[test_mask, "label"].to_numpy()

    results: dict[str, dict[str, float | int]] = {}
    for method_name, score in (
        ("pca_physics", pca_physics),
        ("pca_reconstruction", pca),
    ):
        threshold, _ = select_threshold_on_validation(
            data=full_data, score=score, contamination=0.35
        )
        y_pred = (score[test_mask] > threshold).astype(int)
        metrics = _binary_metrics(y_true, y_pred)
        metrics["threshold"] = float(threshold)
        results[method_name] = metrics
    return results


def evaluate_supervised(
    full_data: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> dict[str, dict[str, float | int]]:
    train_pool = full_data[full_data["split"].isin(["train", "val"])]
    test_part = full_data[full_data["split"] == "test"]

    x_train = train_pool[feature_columns].to_numpy(dtype=float)
    y_train = train_pool["label"].to_numpy(dtype=int)
    x_test = test_part[feature_columns].to_numpy(dtype=float)
    y_test = test_part["label"].to_numpy(dtype=int)

    forest = RandomForestClassifier(n_estimators=200, random_state=seed)
    forest.fit(x_train, y_train)
    y_pred_rf = forest.predict(x_test)
    metrics_rf = _binary_metrics(y_test, y_pred_rf)

    return {"random_forest": metrics_rf}


def run_loso(
    feature_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_unsupervised_training_data(feature_dir)
    data = bundle["data"]
    feature_columns = bundle["feature_columns"]

    if "scenario_template" not in data.columns:
        run_mapping = pd.read_csv(Path(feature_dir) / "run_mapping.csv")
        data = data.merge(
            run_mapping[["run_id", "scenario_template"]],
            on="run_id",
            how="left",
        )

    # Ограничиваем исследование каноническими типами сценариев, чтобы каждый
    # фолд оставался честной проверкой на ранее не встречавшийся тип аномалии.
    canonical_mask = data["scenario_template"].isin(CANONICAL_SCENARIOS)
    data = data[canonical_mask].reset_index(drop=True)

    rows: list[dict] = []
    for held_out in ANOMALY_SCENARIOS:
        full = build_loso_split(data, held_out, seed=seed)
        unsup = evaluate_unsupervised(full, feature_columns, seed=seed)
        sup = evaluate_supervised(full, feature_columns, seed=seed)
        for method_name, metrics in {**unsup, **sup}.items():
            rows.append(
                {
                    "held_out_scenario": held_out,
                    "method": method_name,
                    "test_size": int((full["split"] == "test").sum()),
                    "train_size": int((full["split"] == "train").sum()),
                    "val_size": int((full["split"] == "val").sum()),
                    **metrics,
                }
            )

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(output_dir / "loso_metrics.csv", index=False)

    agg_rows = []
    for method_name, group in per_fold.groupby("method"):
        row = {"method": method_name, "fold_count": int(len(group))}
        for metric in ("accuracy", "precision", "recall", "f1"):
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        agg_rows.append(row)

    aggregated = pd.DataFrame(agg_rows)
    aggregated.to_csv(output_dir / "loso_aggregated.csv", index=False)

    metadata = {
        "task": "leave_one_scenario_out_evaluation",
        "scenarios_held_out": list(ANOMALY_SCENARIOS),
        "canonical_scenarios": list(CANONICAL_SCENARIOS),
        "test_normal_fraction": TEST_NORMAL_FRACTION,
        "val_fraction": VAL_FRACTION,
        "seed": int(seed),
        "comment": (
            "На каждом фолде один аномальный сценарий полностью убран из обучения и "
            "использован только в качестве тестовых аномалий. Тест дополнительно "
            "содержит случайную долю нормальных запусков для расчёта precision/recall."
        ),
    }
    (output_dir / "loso_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"per_fold": per_fold, "aggregated": aggregated}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO-оценка устойчивости методов к новым типам аномалий.",
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
    )
    parser.add_argument(
        "--output",
        default="outputs/anomaly/demo/loso",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_loso(
        feature_dir=Path(args.features),
        output_dir=Path(args.output),
        seed=args.seed,
    )
    print("LOSO-метрики по фолдам:")
    print(
        result["per_fold"][
            ["held_out_scenario", "method", "precision", "recall", "f1"]
        ].to_string(index=False)
    )
    print()
    print("Агрегированные метрики (среднее по 4 фолдам):")
    print(
        result["aggregated"][
            [
                "method",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
