from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN_FEATURE_NAMES = {
    "label",
    "label_name",
    "split",
    "seed",
    "run_id",
    "dataset_id",
    "scenario_template",
    "scenario_geometry",
    "anomaly_type",
    "frame_label",
    "frame_label_source",
}
SUSPICIOUS_FEATURE_FRAGMENTS = (
    "label",
    "split",
    "seed",
    "template",
    "scenario",
    "anomaly",
    "path",
    "dir",
)
FEATURE_COLUMNS = [
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_p95",
    "force_mean",
    "force_max",
    "low_speed_fraction",
    "nearest_distance_mean",
    "nearest_distance_min",
    "density_proxy_mean",
    "density_proxy_max",
    "goal_progress_mean",
    "goal_progress_std",
    "goal_reached_fraction",
    "negative_progress_fraction",
    "close_pair_count_mean",
    "close_pair_count_max",
    "close_pair_fraction_mean",
    "congestion_index",
    "contact_density_index",
]
FEATURE_INFO = {
    "speed_mean": {
        "name": "Средняя скорость",
        "description": "средняя скорость агентов за весь запуск",
    },
    "speed_std": {
        "name": "Разброс скорости",
        "description": "насколько сильно скорость менялась внутри запуска",
    },
    "speed_max": {
        "name": "Максимальная скорость",
        "description": "самая большая скорость агента за запуск",
    },
    "speed_p95": {
        "name": "Скорость 95%",
        "description": "высокая, но не единичная пиковая скорость",
    },
    "force_mean": {
        "name": "Средняя сила",
        "description": "средняя сила взаимодействия агентов",
    },
    "force_max": {
        "name": "Максимальная сила",
        "description": "самый сильный толчок или контакт в запуске",
    },
    "low_speed_fraction": {
        "name": "Доля медленного движения",
        "description": "какая часть наблюдений была почти без движения",
    },
    "nearest_distance_mean": {
        "name": "Среднее расстояние до соседа",
        "description": "средняя дистанция до ближайшего агента",
    },
    "nearest_distance_min": {
        "name": "Минимальное расстояние до соседа",
        "description": "самое тесное сближение агентов",
    },
    "density_proxy_mean": {
        "name": "Средняя плотность",
        "description": "условная плотность по расстоянию до ближайших соседей",
    },
    "density_proxy_max": {
        "name": "Максимальная плотность",
        "description": "самый плотный момент внутри запуска",
    },
    "goal_progress_mean": {
        "name": "Средний прогресс к цели",
        "description": "насколько агенты в среднем приблизились к цели",
    },
    "goal_progress_std": {
        "name": "Разброс прогресса к цели",
        "description": "насколько по-разному агенты продвигались к цели",
    },
    "goal_reached_fraction": {
        "name": "Доля дошедших до цели",
        "description": "какая часть агентов дошла до цели",
    },
    "negative_progress_fraction": {
        "name": "Доля движения назад",
        "description": "какая часть агентов стала дальше от цели",
    },
    "close_pair_count_mean": {
        "name": "Среднее число близких пар",
        "description": "среднее число пар агентов ближе 0.75 м",
    },
    "close_pair_count_max": {
        "name": "Максимум близких пар",
        "description": "самый тесный момент по числу близких пар",
    },
    "close_pair_fraction_mean": {
        "name": "Средняя доля близких пар",
        "description": "доля близких пар среди всех пар агентов",
    },
    "congestion_index": {
        "name": "Индекс затора",
        "description": "сочетание медленного движения и плотности",
    },
    "contact_density_index": {
        "name": "Индекс контактов и плотности",
        "description": "сочетание близких контактов и плотности",
    },
}
INDEX_COLUMNS = ["run_id"]
LOW_SPEED_THRESHOLD = 0.2
CLOSE_PAIR_DISTANCE = 0.75
GOAL_REACHED_DISTANCE = 0.75
EPSILON = 1e-9


def build_features(dataset_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(dataset_path / "runs.csv")
    feature_rows = []
    for run_row in runs.to_dict("records"):
        run_id = str(run_row["run_id"])
        run_dir = dataset_path / "runs" / run_id
        features = extract_features_for_run(run_dir, run_row)
        feature_rows.append({"run_id": run_id, **features})

    features_run = pd.DataFrame(feature_rows)[["run_id", *FEATURE_COLUMNS]]
    validate_feature_columns(features_run, FEATURE_COLUMNS)

    run_mapping = runs[
        ["run_id", "split", "label", "label_name", "scenario_template"]
    ].copy()

    features_run.to_csv(output_path / "features_run.csv", index=False)
    run_mapping.to_csv(output_path / "run_mapping.csv", index=False)

    feature_schema = _build_feature_schema()
    (output_path / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "dataset_dir": str(dataset_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(features_run)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "synthetic_data": True,
        "target_level": "run",
        "leakage_checked": True,
    }
    (output_path / "features_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": output_path,
        "features_run": features_run,
        "run_mapping": run_mapping,
        "feature_schema": feature_schema,
        "metadata": metadata,
    }


def extract_features_for_run(
    run_dir: Path | str,
    run_row: dict[str, Any] | pd.Series,
) -> dict[str, float]:
    run_path = Path(run_dir)
    trajectories = pd.read_csv(run_path / "trajectories.csv")
    speeds = trajectories["speed"].astype(float).to_numpy()
    forces = trajectories.get("force_norm", pd.Series([0.0] * len(trajectories)))
    force_values = forces.astype(float).to_numpy()
    distance_stats = _extract_distance_features(trajectories)
    progress_stats = _extract_progress_features(trajectories, run_row)

    low_speed_fraction = float(np.mean(speeds < LOW_SPEED_THRESHOLD))
    density_proxy_mean = distance_stats["density_proxy_mean"]
    close_pair_fraction_mean = distance_stats["close_pair_fraction_mean"]

    features = {
        "speed_mean": float(np.mean(speeds)),
        "speed_std": float(np.std(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "force_mean": float(np.mean(force_values)),
        "force_max": float(np.max(force_values)),
        "low_speed_fraction": low_speed_fraction,
        **distance_stats,
        **progress_stats,
        "congestion_index": low_speed_fraction * density_proxy_mean,
        "contact_density_index": close_pair_fraction_mean * density_proxy_mean,
    }
    return {name: float(features[name]) for name in FEATURE_COLUMNS}


def validate_feature_columns(
    features_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    normalized_names = {column.lower() for column in feature_columns}
    forbidden = normalized_names & FORBIDDEN_FEATURE_NAMES
    if forbidden:
        raise ValueError(f"Forbidden feature columns found: {sorted(forbidden)}")

    suspicious = [
        column
        for column in feature_columns
        for fragment in SUSPICIOUS_FEATURE_FRAGMENTS
        if fragment in column.lower()
    ]
    if suspicious:
        raise ValueError(f"Suspicious feature columns found: {sorted(suspicious)}")

    if "run_id" in feature_columns:
        raise ValueError("run_id may identify rows but must not be a feature.")

    missing = [
        column for column in feature_columns if column not in features_df.columns
    ]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(features_df[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")

    values = features_df[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")


def load_feature_artifacts(feature_dir: Path | str) -> dict[str, Any]:
    feature_path = Path(feature_dir)
    return {
        "features_run": pd.read_csv(feature_path / "features_run.csv"),
        "run_mapping": pd.read_csv(feature_path / "run_mapping.csv"),
        "feature_schema": json.loads(
            (feature_path / "feature_schema.json").read_text(encoding="utf-8")
        ),
        "metadata": json.loads(
            (feature_path / "features_metadata.json").read_text(encoding="utf-8")
        ),
    }


def get_feature_display_name(feature: str) -> str:
    return FEATURE_INFO.get(feature, {}).get("name", feature)


def get_feature_description(feature: str) -> str:
    return FEATURE_INFO.get(feature, {}).get("description", "")


def empty_feature_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["run_id", *FEATURE_COLUMNS])


def _extract_distance_features(trajectories: pd.DataFrame) -> dict[str, float]:
    nearest_distances = []
    density_proxies = []
    close_pair_counts = []
    close_pair_fractions = []

    for _, frame in trajectories.groupby("step", sort=True):
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        frame_stats = _frame_distance_stats(coordinates)
        nearest_distances.extend(frame_stats["nearest_distances"])
        density_proxies.append(frame_stats["density_proxy"])
        close_pair_counts.append(frame_stats["close_pair_count"])
        close_pair_fractions.append(frame_stats["close_pair_fraction"])

    nearest = np.array(nearest_distances, dtype=float)
    density = np.array(density_proxies, dtype=float)
    close_counts = np.array(close_pair_counts, dtype=float)
    close_fractions = np.array(close_pair_fractions, dtype=float)

    return {
        "nearest_distance_mean": float(np.mean(nearest)),
        "nearest_distance_min": float(np.min(nearest)),
        "density_proxy_mean": float(np.mean(density)),
        "density_proxy_max": float(np.max(density)),
        "close_pair_count_mean": float(np.mean(close_counts)),
        "close_pair_count_max": float(np.max(close_counts)),
        "close_pair_fraction_mean": float(np.mean(close_fractions)),
    }


def _frame_distance_stats(coordinates: np.ndarray) -> dict[str, Any]:
    agent_count = len(coordinates)
    if agent_count < 2:
        return {
            "nearest_distances": [0.0],
            "density_proxy": 0.0,
            "close_pair_count": 0,
            "close_pair_fraction": 0.0,
        }

    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(offsets**2, axis=2))
    np.fill_diagonal(distances, np.inf)

    nearest = np.min(distances, axis=1)
    pair_indices = np.triu_indices(agent_count, k=1)
    pair_distances = distances[pair_indices]
    close_pair_count = int(np.sum(pair_distances < CLOSE_PAIR_DISTANCE))
    total_pairs = agent_count * (agent_count - 1) / 2

    return {
        "nearest_distances": nearest.tolist(),
        "density_proxy": float(np.mean(1.0 / (nearest + EPSILON))),
        "close_pair_count": close_pair_count,
        "close_pair_fraction": float(close_pair_count / total_pairs),
    }


def _extract_progress_features(
    trajectories: pd.DataFrame,
    run_row: dict[str, Any] | pd.Series,
) -> dict[str, float]:
    progress_values = []
    reached_values = []

    for _, agent_track in trajectories.groupby("agent_id", sort=True):
        ordered_track = agent_track.sort_values("step")
        first = ordered_track.iloc[0]
        last = ordered_track.iloc[-1]

        initial_distance = _distance_to_goal(first)
        final_distance = _distance_to_goal(last)
        if initial_distance <= EPSILON:
            progress = 1.0
        else:
            progress = (initial_distance - final_distance) / initial_distance

        progress_values.append(progress)
        reached_values.append(final_distance <= GOAL_REACHED_DISTANCE)

    progress = np.array(progress_values, dtype=float)
    goal_reached_fraction = _goal_reached_fraction(run_row, reached_values)

    return {
        "goal_progress_mean": float(np.mean(progress)),
        "goal_progress_std": float(np.std(progress)),
        "goal_reached_fraction": float(goal_reached_fraction),
        "negative_progress_fraction": float(np.mean(progress < 0)),
    }


def _goal_reached_fraction(
    run_row: dict[str, Any] | pd.Series,
    reached_values: list[bool],
) -> float:
    if "final_goal_reached_fraction" in run_row:
        return float(run_row["final_goal_reached_fraction"])
    return float(np.mean(reached_values))


def _distance_to_goal(row: pd.Series) -> float:
    return float(np.hypot(row["goal_x"] - row["x"], row["goal_y"] - row["y"]))


def _build_feature_schema() -> dict[str, Any]:
    return {
        "feature_columns": FEATURE_COLUMNS,
        "feature_details": [
            {
                "feature": feature,
                "name": get_feature_display_name(feature),
                "description": get_feature_description(feature),
            }
            for feature in FEATURE_COLUMNS
        ],
        "index_columns": INDEX_COLUMNS,
        "leakage_policy": {
            "target_level": "run",
            "run_id_policy": (
                "run_id хранится как идентификатор строки, но не входит "
                "в feature_columns."
            ),
            "forbidden_feature_names": sorted(FORBIDDEN_FEATURE_NAMES),
            "suspicious_feature_fragments": list(SUSPICIOUS_FEATURE_FRAGMENTS),
        },
        "excluded_columns": sorted(FORBIDDEN_FEATURE_NAMES),
    }
