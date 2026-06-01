from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crowd_anomaly.features import (
    FORBIDDEN_FEATURE_NAMES,
    get_feature_description,
    get_feature_display_name,
    load_feature_artifacts,
    validate_feature_columns,
)

MODEL_RANDOM_STATE = 42


def train_baselines(feature_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_data = load_features_for_training(feature_dir)
    data = training_data["data"]
    feature_columns = training_data["feature_columns"]
    warnings = list(training_data["warnings"])

    train_data = data[data["split"] == "train"]
    if train_data.empty or train_data["label"].nunique() < 2:
        warnings.append(
            "Train split is too small or has one class; using all rows for training."
        )
        train_data = data

    models = _build_models()
    metrics_rows = []
    prediction_frames = []
    importance_rows = []

    for model_name, model in models.items():
        model.fit(train_data[feature_columns], train_data["label"])

        for split_name in ["train", "val", "test"]:
            split_data = data[data["split"] == split_name]
            split_predictions, split_warnings = _predict_split(
                model=model,
                model_name=model_name,
                split_name=split_name,
                split_data=split_data,
                feature_columns=feature_columns,
            )
            warnings.extend(split_warnings)
            prediction_frames.append(split_predictions)

            metrics = evaluate_model(
                y_true=split_predictions["label"],
                scores=split_predictions["score"],
                predicted_labels=split_predictions["predicted_label"],
            )
            metrics_rows.append({"model": model_name, "split": split_name, **metrics})

        importance_rows.extend(
            _extract_feature_importance(model_name, model, feature_columns)
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_by_split = pd.DataFrame(metrics_rows)
    feature_importance = pd.DataFrame(importance_rows)
    best_model = _best_model_by_val_f1(metrics_by_split)
    test_result = _test_result_for_best_model(metrics_by_split, best_model)

    metrics_payload = {
        "synthetic_data": True,
        "target_level": "run",
        "models": {
            model_name: _model_metrics(metrics_by_split, model_name)
            for model_name in models
        },
        "best_model_by_val_f1": best_model,
        "test_result_for_best_model": test_result,
        "warnings": sorted(set(warnings)),
        "limitations": [
            "данные только синтетические",
            "нет проверки на реальных данных",
            "нет точной покадровой локализации",
            "целевая метка задана на уровне всего запуска",
        ],
    }

    write_metrics(
        output_dir=output_path,
        metrics_payload=metrics_payload,
        metrics_by_split=metrics_by_split,
        predictions=predictions,
        feature_importance=feature_importance,
    )
    _plot_score_distribution(predictions, output_path / "score_distribution.png")

    return {
        "output_dir": output_path,
        "metrics": metrics_payload,
        "metrics_by_split": metrics_by_split,
        "predictions": predictions,
        "feature_importance": feature_importance,
    }


def load_features_for_training(feature_dir: Path | str) -> dict[str, Any]:
    artifacts = load_feature_artifacts(feature_dir)
    features_run = artifacts["features_run"]
    run_mapping = artifacts["run_mapping"]
    schema = artifacts["feature_schema"]
    feature_columns = list(schema["feature_columns"])
    validate_feature_columns(features_run, feature_columns)

    forbidden = set(feature_columns) & FORBIDDEN_FEATURE_NAMES
    if forbidden:
        raise ValueError(f"Forbidden leakage columns found: {sorted(forbidden)}")

    data = features_run.merge(run_mapping, on="run_id", how="inner")
    warnings = []
    if len(data) != len(features_run):
        warnings.append("Some feature rows were not matched with run_mapping.")

    for split_name, split_data in data.groupby("split"):
        if len(split_data) < 2:
            warnings.append(f"Split '{split_name}' has fewer than 2 rows.")
        if split_data["label"].nunique() < 2:
            warnings.append(f"Split '{split_name}' has fewer than 2 classes.")

    return {
        "data": data,
        "feature_columns": feature_columns,
        "schema": schema,
        "warnings": warnings,
    }


def evaluate_model(
    y_true: pd.Series,
    scores: pd.Series,
    predicted_labels: pd.Series,
) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    result = {
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "precision": float(
            precision_score(y_true, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }

    if y_true.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["pr_auc"] = float(average_precision_score(y_true, scores))

    return result


def write_metrics(
    output_dir: Path | str,
    metrics_payload: dict[str, Any],
    metrics_by_split: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_by_split.to_csv(output_path / "metrics_by_split.csv", index=False)
    predictions.to_csv(output_path / "predictions.csv", index=False)
    feature_importance.to_csv(output_path / "feature_importance.csv", index=False)


def _build_models() -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=MODEL_RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=1,
            random_state=MODEL_RANDOM_STATE,
        ),
    }


def _predict_split(
    model: Any,
    model_name: str,
    split_name: str,
    split_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    if split_data.empty:
        warnings.append(f"Split '{split_name}' is empty for {model_name}.")
        return (
            pd.DataFrame(
                columns=[
                    "run_id",
                    "split",
                    "label",
                    "label_name",
                    "model",
                    "score",
                    "predicted_label",
                    "predicted_label_name",
                ]
            ),
            warnings,
        )

    scores = _positive_class_scores(model, split_data[feature_columns])
    predicted_label = (scores >= 0.5).astype(int)

    predictions = split_data[
        ["run_id", "split", "label", "label_name"]
    ].copy()
    predictions["model"] = model_name
    predictions["score"] = scores
    predictions["predicted_label"] = predicted_label
    predictions["predicted_label_name"] = np.where(
        predictions["predicted_label"] == 1,
        "anomaly",
        "normal",
    )
    return predictions, warnings


def _positive_class_scores(model: Any, data: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(data)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(data), dtype=float)
    return probabilities[:, classes.index(1)]


def _extract_feature_importance(
    model_name: str,
    model: Any,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if model_name == "logistic_regression":
        estimator = model.named_steps["logisticregression"]
        values = np.abs(estimator.coef_[0])
        importance_type = "absolute_coefficient"
    elif model_name == "random_forest":
        values = model.feature_importances_
        importance_type = "gini_importance"
    else:
        values = np.zeros(len(feature_columns), dtype=float)
        importance_type = "not_available"

    return [
        {
            "model": model_name,
            "feature": feature,
            "feature_name": get_feature_display_name(feature),
            "description": get_feature_description(feature),
            "importance": float(value),
            "importance_type": importance_type,
        }
        for feature, value in zip(feature_columns, values, strict=True)
    ]


def _best_model_by_val_f1(metrics_by_split: pd.DataFrame) -> str | None:
    val_metrics = metrics_by_split[
        (metrics_by_split["split"] == "val") & metrics_by_split["f1"].notna()
    ]
    if val_metrics.empty:
        return None
    best_row = val_metrics.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    return str(best_row["model"])


def _test_result_for_best_model(
    metrics_by_split: pd.DataFrame,
    best_model: str | None,
) -> dict[str, Any] | None:
    if best_model is None:
        return None

    test_rows = metrics_by_split[
        (metrics_by_split["model"] == best_model)
        & (metrics_by_split["split"] == "test")
    ]
    if test_rows.empty:
        return None
    return _clean_metric_record(test_rows.iloc[0].to_dict())


def _model_metrics(metrics_by_split: pd.DataFrame, model_name: str) -> dict[str, Any]:
    rows = metrics_by_split[metrics_by_split["model"] == model_name]
    return {
        str(row["split"]): _clean_metric_record(row.to_dict())
        for _, row in rows.iterrows()
    }


def _clean_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    clean_record = {}
    for key, value in record.items():
        if pd.isna(value):
            clean_record[key] = None
        elif isinstance(value, np.generic):
            clean_record[key] = value.item()
        else:
            clean_record[key] = value
    return clean_record


def _plot_score_distribution(predictions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(predictions["model"].unique())
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(model_names),
        figsize=(6 * len(model_names), 4),
        squeeze=False,
    )

    for axis, model_name in zip(axes.ravel(), model_names, strict=True):
        model_predictions = predictions[predictions["model"] == model_name]
        for label_name, label_text, color in [
            ("normal", "норма", "#4c78a8"),
            ("anomaly", "аномалия", "#f28e2b"),
        ]:
            scores = model_predictions.loc[
                model_predictions["label_name"] == label_name,
                "score",
            ]
            axis.hist(
                scores,
                bins=8,
                alpha=0.65,
                label=label_text,
                color=color,
                range=(0.0, 1.0),
            )
        axis.set_title(_model_name_ru(model_name))
        axis.set_xlabel("Показатель модели")
        axis.set_ylabel("Количество запусков")
        axis.legend()

    figure.suptitle("Распределение показателя модели по меткам")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _model_name_ru(model_name: str) -> str:
    return {
        "logistic_regression": "Логистическая регрессия",
        "random_forest": "Случайный лес",
    }.get(model_name, model_name)
