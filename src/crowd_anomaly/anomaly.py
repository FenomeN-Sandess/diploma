from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from crowd_anomaly.features import (
    get_feature_description,
    get_feature_display_name,
    load_feature_artifacts,
    validate_feature_columns,
)

MODEL_RANDOM_STATE = 42
PCA_SCORE_WEIGHT = 0.13
PCA_COMPONENTS = 5
OLD_ISOLATION_SCORE_WEIGHT = 0.25
PHYSICS_ANOMALY_COLUMNS = [
    "force_max",
    "density_proxy_max",
    "close_pair_count_max",
]
PCA_FEATURE_FRAGMENTS = ("force", "density", "close", "contact")
LOCAL_CANDIDATE_COLUMNS = [
    "run_id",
    "split",
    "label",
    "label_name",
    "anomaly_score",
    "frame_step",
    "frame_time_s",
    "frame_score",
    "mean_force",
    "max_force",
    "density_proxy",
    "close_pair_count",
    "top_agent_ids",
]
LOCAL_CLOSE_DISTANCE = 0.75
EPSILON = 1e-9


def train_unsupervised_anomaly(
    feature_dir: Path | str,
    output_dir: Path | str,
    contamination: float = 0.35,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_data = load_unsupervised_training_data(feature_dir)
    data = training_data["data"]
    feature_columns = training_data["feature_columns"]
    train_data = data[data["split"] == "train"]
    if train_data.empty:
        raise ValueError("Нет train-строк для обучения модели без учителя.")

    physics_score = build_physics_anomaly_score(data, train_data)
    pca_score, pca_columns, pca_components = build_pca_reconstruction_score(
        data=data,
        train_data=train_data,
        feature_columns=feature_columns,
    )
    anomaly_score = (
        PCA_SCORE_WEIGHT * pca_score
        + (1.0 - PCA_SCORE_WEIGHT) * physics_score
    )

    old_isolation_score = build_isolation_forest_score(
        data=data,
        train_data=train_data,
        feature_columns=feature_columns,
    )
    old_hybrid_score = (
        OLD_ISOLATION_SCORE_WEIGHT * old_isolation_score
        + (1.0 - OLD_ISOLATION_SCORE_WEIGHT) * physics_score
    )

    threshold, threshold_rule = select_threshold_on_validation(
        data=data,
        score=anomaly_score,
        contamination=contamination,
    )
    predictions = predict_unsupervised(
        data=data,
        pca_score=pca_score,
        physics_score=physics_score,
        old_isolation_score=old_isolation_score,
        anomaly_score=anomaly_score,
        threshold=threshold,
    )
    metrics_by_split = summarize_anomaly_metrics(predictions)
    local_candidates = build_local_anomaly_candidates(
        training_data=training_data,
        predictions=predictions,
    )
    method_comparison = build_method_comparison(
        data=data,
        physics_score=physics_score,
        pca_score=pca_score,
        pca_hybrid_score=anomaly_score,
        old_isolation_score=old_isolation_score,
        old_hybrid_score=old_hybrid_score,
        contamination=contamination,
    )
    metrics_payload = {
        "task": "unsupervised_anomaly_detection",
        "description": (
            "Обнаружение аномальных запусков без использования ручной разметки "
            "при обучении. Метки применяются только для итоговой проверки метрик."
        ),
        "model": "Метод главных компонент + физический показатель аномальности",
        "contamination": float(contamination),
        "target_level": "run",
        "training_uses_label_as_feature": False,
        "training_uses_labels_for_fit": False,
        "training_uses_label_to_select_reference": False,
        "training_reference": "all_train_runs",
        "training_run_count": int(len(train_data)),
        "feature_columns": feature_columns,
        "feature_details": _feature_details(feature_columns),
        "pca_feature_columns": pca_columns,
        "pca_components": pca_components,
        "physics_anomaly_columns": PHYSICS_ANOMALY_COLUMNS,
        "physics_anomaly_feature_details": _feature_details(PHYSICS_ANOMALY_COLUMNS),
        "anomaly_score_formula": (
            "0.13 * нормированная ошибка восстановления по методу главных компонент + "
            "0.87 * средний нормированный физический показатель "
            "(force_max, density_proxy_max, close_pair_count_max)"
        ),
        "threshold": threshold,
        "threshold_rule": threshold_rule,
        "local_candidate_file": "local_anomaly_candidates.csv",
        "method_comparison_file": "unsupervised_method_comparison.csv",
        "comparison_methods": [
            "pca_physics",
            "physics_only",
            "hybrid_iforest_physics",
        ],
        "limitations": [
            "Метка label не передаётся в модель и не входит в матрицу признаков.",
            "Train-выборка используется целиком, без отбора строк по label.",
            "Оценка относится ко всему запуску, а не к отдельному кадру или агенту.",
            (
                "Локальные кандидаты являются диагностикой, "
                "а не ручной эталонной разметкой."
            ),
            "Данные синтетические и требуют внешней проверки для реальных толп.",
        ],
    }

    predictions.to_csv(output_path / "unsupervised_predictions.csv", index=False)
    local_candidates.to_csv(
        output_path / "local_anomaly_candidates.csv",
        index=False,
    )
    metrics_by_split.to_csv(
        output_path / "unsupervised_metrics_by_split.csv",
        index=False,
    )
    method_comparison.to_csv(
        output_path / "unsupervised_method_comparison.csv",
        index=False,
    )
    (output_path / "unsupervised_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_score_distribution(
        predictions,
        output_path / "unsupervised_score_distribution.png",
        threshold=threshold,
    )
    _plot_roc_pr_curves(
        predictions,
        output_path / "roc_pr_curves.png",
    )

    return {
        "output_dir": output_path,
        "predictions": predictions,
        "local_candidates": local_candidates,
        "metrics_by_split": metrics_by_split,
        "method_comparison": method_comparison,
        "metrics": metrics_payload,
    }


def load_unsupervised_training_data(feature_dir: Path | str) -> dict[str, Any]:
    artifacts = load_feature_artifacts(feature_dir)
    features_run = artifacts["features_run"]
    run_mapping = artifacts["run_mapping"]
    feature_columns = list(artifacts["feature_schema"]["feature_columns"])
    validate_feature_columns(features_run, feature_columns)

    data = features_run.merge(run_mapping, on="run_id", how="inner")
    return {
        "data": data,
        "feature_columns": feature_columns,
        "metadata": artifacts["metadata"],
    }


def _feature_details(feature_columns: list[str]) -> list[dict[str, str]]:
    return [
        {
            "feature": feature,
            "name": get_feature_display_name(feature),
            "description": get_feature_description(feature),
        }
        for feature in feature_columns
    ]


def build_physics_anomaly_score(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
) -> np.ndarray:
    score = np.zeros(len(data), dtype=float)
    for column in PHYSICS_ANOMALY_COLUMNS:
        score += _robust_positive_score(
            values=data[column].to_numpy(dtype=float),
            reference_values=train_data[column].to_numpy(dtype=float),
        )
    return score / len(PHYSICS_ANOMALY_COLUMNS)


def build_pca_reconstruction_score(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[str], int]:
    pca_columns = [
        column
        for column in feature_columns
        if any(fragment in column for fragment in PCA_FEATURE_FRAGMENTS)
    ]
    if not pca_columns:
        pca_columns = feature_columns

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[pca_columns])
    all_scaled = scaler.transform(data[pca_columns])

    component_count = min(PCA_COMPONENTS, train_scaled.shape[1], len(train_scaled) - 1)
    component_count = max(component_count, 1)
    model = PCA(n_components=component_count, random_state=MODEL_RANDOM_STATE)
    model.fit(train_scaled)

    train_restored = model.inverse_transform(model.transform(train_scaled))
    all_restored = model.inverse_transform(model.transform(all_scaled))
    train_error = np.mean((train_scaled - train_restored) ** 2, axis=1)
    all_error = np.mean((all_scaled - all_restored) ** 2, axis=1)

    return (
        _robust_positive_score(all_error, train_error),
        pca_columns,
        int(component_count),
    )


def build_isolation_forest_score(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(train_data[feature_columns])
    raw_score = -model.decision_function(data[feature_columns])
    train_raw_score = -model.decision_function(train_data[feature_columns])
    return _robust_positive_score(raw_score, train_raw_score)


def select_threshold_on_validation(
    data: pd.DataFrame,
    score: np.ndarray,
    contamination: float,
) -> tuple[float, str]:
    validation_mask = data["split"] == "val"
    validation_data = data[validation_mask]
    validation_score = score[validation_mask.to_numpy()]

    if validation_data.empty or validation_data["label"].nunique() < 2:
        train_mask = data["split"] == "train"
        threshold = float(
            np.quantile(score[train_mask.to_numpy()], 1.0 - contamination)
        )
        return threshold, "train_quantile_fallback"

    best_threshold = 0.0
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for threshold in _candidate_thresholds(validation_score):
        predicted = np.where(validation_score > threshold, 1, 0)
        metrics = _binary_metrics(validation_data["label"].to_numpy(), predicted)
        key = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["accuracy"],
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, "val_best_f1"


def predict_unsupervised(
    data: pd.DataFrame,
    pca_score: np.ndarray,
    physics_score: np.ndarray,
    old_isolation_score: np.ndarray,
    anomaly_score: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    result = data[["run_id", "split", "label", "label_name"]].copy()
    result["pca_score"] = pca_score
    result["physics_score"] = physics_score
    result["old_isolation_score"] = old_isolation_score
    result["anomaly_score"] = anomaly_score
    result["predicted_label"] = np.where(result["anomaly_score"] > threshold, 1, 0)
    result["predicted_label_name"] = np.where(
        result["predicted_label"] == 1,
        "anomaly",
        "normal",
    )
    return result


def summarize_anomaly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split_data in predictions.groupby("split", sort=True):
        y_true = split_data["label"].to_numpy()
        y_pred = split_data["predicted_label"].to_numpy()
        metrics = _binary_metrics(y_true, y_pred)
        rows.append(
            {
                "split": split_name,
                **metrics,
                "run_count": int(len(split_data)),
                "class_count": int(pd.Series(y_true).nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_method_comparison(
    data: pd.DataFrame,
    physics_score: np.ndarray,
    pca_score: np.ndarray,
    pca_hybrid_score: np.ndarray,
    old_isolation_score: np.ndarray,
    old_hybrid_score: np.ndarray,
    contamination: float,
) -> pd.DataFrame:
    method_scores = {
        "pca_physics": pca_hybrid_score,
        "pca_reconstruction": pca_score,
        "physics_only": physics_score,
        "hybrid_iforest_physics": old_hybrid_score,
        "isolation_forest": old_isolation_score,
    }

    rows = []
    for method_name, score in method_scores.items():
        threshold, threshold_rule = select_threshold_on_validation(
            data=data,
            score=score,
            contamination=contamination,
        )
        method_predictions = data[["run_id", "split", "label"]].copy()
        method_predictions["predicted_label"] = np.where(score > threshold, 1, 0)
        method_metrics = summarize_anomaly_metrics(method_predictions)
        for row in method_metrics.to_dict("records"):
            rows.append(
                {
                    "method": method_name,
                    **row,
                    "threshold": threshold,
                    "threshold_rule": threshold_rule,
                }
            )

    return pd.DataFrame(rows)


def _robust_positive_score(
    values: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    median = float(np.median(reference_values))
    mad = float(np.median(np.abs(reference_values - median)))
    scale = mad if mad > EPSILON else float(np.std(reference_values))
    if scale <= EPSILON:
        scale = 1.0
    return np.maximum((values - median) / scale, 0.0)


def _candidate_thresholds(score: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(score, dtype=float))
    if len(unique) == 1:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.concatenate(([unique[0] - EPSILON], unique, midpoints)))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_local_anomaly_candidates(
    training_data: dict[str, Any],
    predictions: pd.DataFrame,
    rows_per_run: int = 3,
    max_runs: int = 10,
) -> pd.DataFrame:
    metadata = training_data.get("metadata", {})
    dataset_dir = Path(str(metadata.get("dataset_dir", "")))
    if not dataset_dir.exists():
        return pd.DataFrame(columns=LOCAL_CANDIDATE_COLUMNS)

    selected_runs = predictions[predictions["predicted_label"] == 1]
    if selected_runs.empty:
        selected_runs = predictions
    selected_runs = selected_runs.sort_values("anomaly_score", ascending=False).head(
        max_runs
    )

    rows = []
    for prediction in selected_runs.to_dict("records"):
        trajectory_path = dataset_dir / "runs" / str(prediction["run_id"])
        trajectory_path = trajectory_path / "trajectories.csv"
        if not trajectory_path.exists():
            continue

        trajectories = pd.read_csv(trajectory_path)
        frame_scores = _score_local_frames(trajectories)
        for frame_row in frame_scores.head(rows_per_run).to_dict("records"):
            rows.append(
                {
                    "run_id": prediction["run_id"],
                    "split": prediction["split"],
                    "label": prediction["label"],
                    "label_name": prediction["label_name"],
                    "anomaly_score": prediction["anomaly_score"],
                    **frame_row,
                }
            )

    return pd.DataFrame(rows, columns=LOCAL_CANDIDATE_COLUMNS)


def _score_local_frames(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step, frame in trajectories.groupby("step", sort=True):
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        distance_stats = _frame_distance_stats(coordinates)
        force_values = frame.get("force_norm", pd.Series([0.0] * len(frame)))
        mean_force = float(force_values.mean())
        max_force = float(force_values.max())
        frame_score = (
            mean_force
            + max_force
            + distance_stats["density_proxy"]
            + distance_stats["close_pair_count"] / max(len(frame), 1)
        )
        rank_column = "force_norm" if "force_norm" in frame.columns else "speed"
        top_agent_ids = ";".join(
            frame.sort_values(rank_column, ascending=False)["agent_id"]
            .astype(str)
            .head(3)
        )
        rows.append(
            {
                "frame_step": int(step),
                "frame_time_s": float(frame["time_s"].iloc[0]),
                "frame_score": float(frame_score),
                "mean_force": mean_force,
                "max_force": max_force,
                "density_proxy": distance_stats["density_proxy"],
                "close_pair_count": distance_stats["close_pair_count"],
                "top_agent_ids": top_agent_ids,
            }
        )

    return pd.DataFrame(rows).sort_values("frame_score", ascending=False)


def _frame_distance_stats(coordinates: np.ndarray) -> dict[str, float | int]:
    agent_count = len(coordinates)
    if agent_count < 2:
        return {"density_proxy": 0.0, "close_pair_count": 0}

    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(offsets**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    pair_indices = np.triu_indices(agent_count, k=1)
    pair_distances = distances[pair_indices]
    close_pair_count = int(np.sum(pair_distances < LOCAL_CLOSE_DISTANCE))
    return {
        "density_proxy": float(np.mean(1.0 / (nearest + EPSILON))),
        "close_pair_count": close_pair_count,
    }


def _plot_score_distribution(
    predictions: pd.DataFrame,
    output_path: Path,
    threshold: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    for label_name, label_text, color in [
        ("normal", "норма", "#4c78a8"),
        ("anomaly", "аномалия", "#f28e2b"),
    ]:
        scores = predictions.loc[
            predictions["label_name"] == label_name,
            "anomaly_score",
        ]
        axis.hist(scores, bins=10, alpha=0.65, label=label_text, color=color)
    axis.axvline(
        threshold,
        color="#333333",
        linestyle="--",
        linewidth=1,
        label="порог",
    )
    axis.set_title("Распределение показателя аномальности без учителя")
    axis.set_xlabel("Показатель аномальности")
    axis.set_ylabel("Количество запусков")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_roc_pr_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Построение ROC- и PR-кривых для основного метода на тестовой выборке."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_data = predictions[predictions["split"] == "test"]
    if test_data.empty or test_data["label"].nunique() < 2:
        return output_path

    y_true = test_data["label"].to_numpy(dtype=int)
    scores = test_data["anomaly_score"].to_numpy(dtype=float)

    # ROC: сортируем пороги по убыванию
    thresholds_roc = np.sort(np.unique(scores))[::-1]
    tpr_list = [0.0]
    fpr_list = [0.0]
    for t in thresholds_roc:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    # Сортировка по FPR для корректного AUC
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]
    roc_auc = float(np.trapezoid(tpr_arr, fpr_arr))

    # PR: сортируем пороги по убыванию
    precision_list = []
    recall_list = []
    for t in thresholds_roc:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        precision_list.append(prec)
        recall_list.append(rec)
    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)
    # Сортируем по recall для корректного AUC
    order_pr = np.argsort(recall_arr)
    recall_sorted = recall_arr[order_pr]
    precision_sorted = precision_arr[order_pr]
    # Добавляем крайние точки
    recall_sorted = np.concatenate(([0.0], recall_sorted, [1.0]))
    precision_sorted = np.concatenate(
        ([precision_sorted[0] if len(precision_sorted) > 0 else 1.0],
         precision_sorted,
         [float(y_true.sum()) / max(len(y_true), 1)])
    )
    pr_auc = float(np.trapezoid(precision_sorted, recall_sorted))

    figure, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax_roc.plot(
        fpr_arr, tpr_arr,
        color="#4c78a8", linewidth=1.5,
        label=f"ROC (AUC = {roc_auc:.3f})",
    )
    ax_roc.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8, label="случайный")
    ax_roc.set_xlabel("Доля ложноположительных (FPR)")
    ax_roc.set_ylabel("Доля истинноположительных (TPR)")
    ax_roc.set_title("ROC-кривая основного метода (тест)")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])

    # PR
    ax_pr.plot(
        recall_sorted, precision_sorted,
        color="#6f8daa", linewidth=1.5,
        label=f"PR (AUC = {pr_auc:.3f})",
    )
    ax_pr.set_xlabel("Полнота (Recall)")
    ax_pr.set_ylabel("Точность (Precision)")
    ax_pr.set_title("PR-кривая основного метода (тест)")
    ax_pr.legend(loc="lower left")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
