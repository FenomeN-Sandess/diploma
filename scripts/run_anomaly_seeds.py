"""Многократный запуск моделей обнаружения аномалий для оценки доверительных интервалов.

Для каждого seed:
- Делается бутстрэп-выборка обучающей части (with replacement) того же размера.
- Перевычисляются физический показатель, ошибка восстановления PCA, оценка
  Isolation Forest, гибридные методы.
- Порог подбирается на валидации, метрики считаются на тесте.

Результат сохраняется в outputs/anomaly/<dataset>/seeds/:
- per_seed_metrics.csv -- метрики каждого прогона по всем методам и сплитам;
- aggregated_metrics.csv -- среднее и стандартное отклонение по тестовым метрикам;
- seeds_metadata.json -- набор seed'ов и параметры эксперимента.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from crowd_anomaly.anomaly import (
    OLD_ISOLATION_SCORE_WEIGHT,
    PCA_COMPONENTS,
    PCA_FEATURE_FRAGMENTS,
    PCA_SCORE_WEIGHT,
    PHYSICS_ANOMALY_COLUMNS,
    _binary_metrics,
    _robust_positive_score,
    load_unsupervised_training_data,
    select_threshold_on_validation,
)

DEFAULT_SEEDS = (42, 17, 23, 71, 109)

# Сравнение качества методов обнаружения проводится на каноническом наборе типов
# (3 нормальных + 4 мгновенные аномалии). Сценарии с отложенным наступлением
# события относятся к экспериментам прогноза на горизонт H и упреждающего сигнала
# (направления 1.1/1.2). На полном наборе доля аномалий поднимается до ~0,73, и
# тривиальная разметка «всё аномально» сама по себе даёт высокий F1; на
# каноническом наборе доля сбалансированнее, и пороги методов не вырождаются.
CANONICAL_SCENARIOS = (
    "normal_corridor",
    "normal_room",
    "normal_wide_exit",
    "anomaly_counterflow",
    "anomaly_bottleneck",
    "anomaly_high_density",
    "anomaly_local_violator",
)


def _restrict_to_canonical(
    data: pd.DataFrame,
    feature_dir: Path,
) -> pd.DataFrame:
    """Оставить только канонические типы сценариев для сравнения методов."""
    if "scenario_template" not in data.columns:
        run_mapping = pd.read_csv(Path(feature_dir) / "run_mapping.csv")
        data = data.merge(
            run_mapping[["run_id", "scenario_template"]],
            on="run_id",
            how="left",
        )
    mask = data["scenario_template"].isin(CANONICAL_SCENARIOS)
    return data[mask].reset_index(drop=True)


def _physics_score(data: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(data), dtype=float)
    for column in PHYSICS_ANOMALY_COLUMNS:
        score += _robust_positive_score(
            values=data[column].to_numpy(dtype=float),
            reference_values=train[column].to_numpy(dtype=float),
        )
    return score / len(PHYSICS_ANOMALY_COLUMNS)


def _pca_score(
    data: pd.DataFrame,
    train: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> np.ndarray:
    pca_columns = [
        column
        for column in feature_columns
        if any(fragment in column for fragment in PCA_FEATURE_FRAGMENTS)
    ]
    if not pca_columns:
        pca_columns = feature_columns

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[pca_columns])
    all_scaled = scaler.transform(data[pca_columns])

    component_count = min(PCA_COMPONENTS, train_scaled.shape[1], len(train_scaled) - 1)
    component_count = max(component_count, 1)
    model = PCA(n_components=component_count, random_state=seed)
    model.fit(train_scaled)

    train_restored = model.inverse_transform(model.transform(train_scaled))
    all_restored = model.inverse_transform(model.transform(all_scaled))
    train_error = np.mean((train_scaled - train_restored) ** 2, axis=1)
    all_error = np.mean((all_scaled - all_restored) ** 2, axis=1)
    return _robust_positive_score(all_error, train_error)


def _iforest_score(
    data: pd.DataFrame,
    train: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> np.ndarray:
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
    )
    model.fit(train[feature_columns])
    raw_score = -model.decision_function(data[feature_columns])
    train_raw_score = -model.decision_function(train[feature_columns])
    return _robust_positive_score(raw_score, train_raw_score)


def _evaluate_method(
    data: pd.DataFrame,
    score: np.ndarray,
    contamination: float,
) -> dict[str, float | int]:
    threshold, threshold_rule = select_threshold_on_validation(
        data=data,
        score=score,
        contamination=contamination,
    )
    rows = {}
    for split_name in ("train", "val", "test"):
        mask = (data["split"] == split_name).to_numpy()
        if mask.sum() == 0:
            continue
        y_true = data.loc[mask, "label"].to_numpy()
        y_pred = (score[mask] > threshold).astype(int)
        metrics = _binary_metrics(y_true, y_pred)
        for metric_name, value in metrics.items():
            rows[f"{split_name}_{metric_name}"] = value
    rows["threshold"] = float(threshold)
    rows["threshold_rule"] = threshold_rule
    return rows


def run_seeds(
    feature_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    contamination: float = 0.35,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = load_unsupervised_training_data(feature_dir)
    data = training_data["data"].reset_index(drop=True)
    feature_columns = training_data["feature_columns"]
    data = _restrict_to_canonical(data, Path(feature_dir))
    train_full = data[data["split"] == "train"].reset_index(drop=True)
    if train_full.empty:
        raise ValueError("Нет train-строк для оценки доверительных интервалов.")

    per_seed_rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(train_full), size=len(train_full), replace=True)
        train_sample = train_full.iloc[sample_idx].reset_index(drop=True)

        physics = _physics_score(data, train_sample)
        pca = _pca_score(data, train_sample, feature_columns, seed=seed)
        iforest = _iforest_score(data, train_sample, feature_columns, seed=seed)

        pca_physics = (
            PCA_SCORE_WEIGHT * pca + (1.0 - PCA_SCORE_WEIGHT) * physics
        )
        iforest_physics = (
            OLD_ISOLATION_SCORE_WEIGHT * iforest
            + (1.0 - OLD_ISOLATION_SCORE_WEIGHT) * physics
        )

        method_scores = {
            "pca_physics": pca_physics,
            "pca_reconstruction": pca,
            "physics_only": physics,
            "hybrid_iforest_physics": iforest_physics,
            "isolation_forest": iforest,
        }
        for method_name, score in method_scores.items():
            metrics = _evaluate_method(data, score, contamination=contamination)
            per_seed_rows.append({"seed": int(seed), "method": method_name, **metrics})

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)

    test_metric_cols = [
        column
        for column in per_seed.columns
        if column.startswith("test_") and column != "test_class_count"
    ]
    aggregated_rows = []
    for method_name, group in per_seed.groupby("method"):
        row = {"method": method_name, "seed_count": int(len(group))}
        for column in test_metric_cols:
            values = group[column].astype(float).to_numpy()
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{column}_min"] = float(np.min(values))
            row[f"{column}_max"] = float(np.max(values))
        aggregated_rows.append(row)

    aggregated = pd.DataFrame(aggregated_rows)
    aggregated.to_csv(output_dir / "aggregated_metrics.csv", index=False)

    metadata = {
        "task": "anomaly_seeds_confidence_interval",
        "seeds": list(int(seed) for seed in seeds),
        "contamination": float(contamination),
        "feature_dir": str(feature_dir),
        "train_resampling": "bootstrap_with_replacement",
        "train_resampling_size": int(len(train_full)),
        "canonical_scenarios": list(CANONICAL_SCENARIOS),
        "methods": list(method_scores.keys()),
        "comment": (
            "Метрики получены при бутстрэп-перевыборке обучающей части. "
            "Тестовая выборка фиксирована."
        ),
    }
    (output_dir / "seeds_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"per_seed": per_seed, "aggregated": aggregated}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка доверительных интервалов методов поиска аномалий.",
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
        help="Папка с рассчитанными признаками на уровне запуска.",
    )
    parser.add_argument(
        "--output",
        default="outputs/anomaly/demo/seeds",
        help="Папка для сохранения результатов многократного прогона.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.35,
        help="Ожидаемая доля аномальных запусков (для альтернативного отбора порога).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Список seed'ов для запусков (по умолчанию 42 17 23 71 109).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_seeds(
        feature_dir=Path(args.features),
        output_dir=Path(args.output),
        seeds=tuple(args.seeds),
        contamination=args.contamination,
    )
    aggregated = result["aggregated"]
    print("Доверительные интервалы по тестовой выборке (среднее ± std):")
    cols_to_print = [
        c
        for c in aggregated.columns
        if c == "method"
        or c.startswith("test_f1")
        or c.startswith("test_precision")
        or c.startswith("test_recall")
        or c.startswith("test_accuracy")
    ]
    print(aggregated[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
