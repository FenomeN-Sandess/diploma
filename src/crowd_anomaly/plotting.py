from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def plot_label_distribution(label_counts: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = label_counts.copy()
    plot_data["_label_order"] = plot_data["label_name"].map(_label_order)
    plot_data = plot_data.sort_values("_label_order")
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = [_label_name_ru(label) for label in plot_data["label_name"]]
    axis.bar(labels, plot_data["run_count"], color="#4c78a8")
    axis.set_title("Распределение запусков по меткам")
    axis.set_xlabel("Метка")
    axis.set_ylabel("Количество запусков")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_split_distribution(split_counts: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(split_counts["split"], split_counts["run_count"], color="#59a14f")
    axis.set_title("Распределение запусков по выборкам")
    axis.set_xlabel("Выборка")
    axis.set_ylabel("Количество запусков")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_scenario_distribution(
    scenario_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if scenario_summary.empty:
        return save_empty_plot(output_path)

    plot_data = scenario_summary.sort_values("run_count", ascending=True)
    colors = plot_data["label_name"].map(
        {"normal": "#4c78a8", "anomaly": "#e15759"}
    ).fillna("#bab0ac")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(plot_data["scenario_template"], plot_data["run_count"], color=colors)
    axis.set_title("Количество запусков по сценариям")
    axis.set_xlabel("Количество запусков")
    axis.set_ylabel("Сценарий")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_run_metric_boxplots(runs: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required_columns = {"label_name", "mean_speed", "mean_force", "max_force"}
    if runs.empty or not required_columns.issubset(runs.columns):
        return save_empty_plot(output_path)

    metric_titles = [
        ("mean_speed", "Средняя скорость"),
        ("mean_force", "Средняя сила"),
        ("max_force", "Максимальная сила"),
        ("final_goal_reached_fraction", "Доля дошедших до цели"),
    ]
    labels = _ordered_label_names(runs["label_name"].unique())

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 7), squeeze=False)
    for axis, (column, title) in zip(axes.ravel(), metric_titles, strict=True):
        if column not in runs.columns:
            axis.axis("off")
            continue
        values_by_label = [
            runs.loc[runs["label_name"] == label, column].to_numpy(dtype=float)
            for label in labels
        ]
        axis.boxplot(
            values_by_label,
            tick_labels=[_label_name_ru(label) for label in labels],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)

    figure.suptitle("Метрики запусков по нормальным и аномальным сценариям")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_trajectory_examples(
    dataset_dir: Path,
    runs: pd.DataFrame,
    output_path: Path,
    local_candidates: pd.DataFrame | None = None,
    max_agents: int = 10,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if runs.empty or "label_name" not in runs.columns:
        return save_empty_plot(output_path)

    sample_runs = _select_demo_runs(runs, local_candidates)
    if sample_runs.empty:
        return save_empty_plot(output_path)
    has_candidates = False

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(sample_runs),
        figsize=(7 * len(sample_runs), 5.2),
        squeeze=False,
    )

    for axis, run in zip(axes.ravel(), sample_runs.to_dict("records"), strict=False):
        run_id = str(run["run_id"])
        trajectories_path = dataset_dir / "runs" / run_id / "trajectories.csv"
        label_name = _label_name_ru(str(run["label_name"]))
        axis.set_title(f"{label_name}: {run['scenario_template']}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        if not trajectories_path.exists():
            axis.text(0.5, 0.5, "Нет файла траекторий", ha="center", va="center")
            continue

        trajectories = pd.read_csv(trajectories_path)
        run_candidates = _run_candidates(local_candidates, run_id)
        has_candidates = has_candidates or not run_candidates.empty
        agent_ids = _select_agent_ids(trajectories, run_candidates, max_agents)
        line_color = "#e15759" if run["label_name"] == "anomaly" else "#4c78a8"

        for agent_id in agent_ids:
            agent_path = trajectories[trajectories["agent_id"] == agent_id]
            axis.plot(
                agent_path["x"],
                agent_path["y"],
                color=line_color,
                linewidth=1.0,
                alpha=0.45,
            )
            _draw_direction_arrow(axis, agent_path, line_color)

            start = agent_path.iloc[0]
            end = agent_path.iloc[-1]
            axis.scatter(
                [start["x"]],
                [start["y"]],
                marker="o",
                s=24,
                color="#59a14f",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            axis.scatter(
                [end["x"]],
                [end["y"]],
                marker="s",
                s=24,
                color="#222222",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

        if {"goal_x", "goal_y"}.issubset(trajectories.columns):
            goal = trajectories[["goal_x", "goal_y"]].iloc[0]
            axis.scatter(
                [goal["goal_x"]],
                [goal["goal_y"]],
                marker="*",
                s=150,
                color="#e15759",
                edgecolor="#8b0000",
                linewidth=0.8,
                zorder=5,
            )
            axis.text(goal["goal_x"], goal["goal_y"], " цель", va="center")

        _draw_candidate_points(axis, trajectories, run_candidates)
        axis.grid(alpha=0.2)
        axis.set_aspect("auto")

    figure.suptitle("Примеры траекторий агентов")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18, wspace=0.12)
    figure.legend(
        handles=_trajectory_legend_handles(has_candidates),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        fontsize=8,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_trajectory_anomaly_timeline(
    dataset_dir: Path,
    runs: pd.DataFrame,
    output_path: Path,
    local_candidates: pd.DataFrame | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_runs = _select_demo_runs(runs, local_candidates)
    if sample_runs.empty:
        return save_empty_plot(output_path)

    metrics = [
        ("mean_force", "Средняя сила"),
        ("max_force", "Максимальная сила"),
        ("close_pair_count", "Близкие пары"),
    ]
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(sample_runs),
        figsize=(7 * len(sample_runs), 8),
        squeeze=False,
        constrained_layout=True,
    )

    for column_index, run in enumerate(sample_runs.to_dict("records")):
        run_id = str(run["run_id"])
        trajectories_path = dataset_dir / "runs" / run_id / "trajectories.csv"
        if not trajectories_path.exists():
            continue

        trajectories = pd.read_csv(trajectories_path)
        timeline = _build_frame_timeline(trajectories)
        run_candidates = _run_candidates(local_candidates, run_id)
        line_color = "#e15759" if run["label_name"] == "anomaly" else "#4c78a8"

        for row_index, (metric, metric_title) in enumerate(metrics):
            axis = axes[row_index][column_index]
            axis.plot(
                timeline["time_s"],
                timeline[metric],
                color=line_color,
                linewidth=1.5,
            )
            _draw_candidate_steps(axis, run_candidates)
            label_name = _label_name_ru(str(run["label_name"]))
            axis.set_title(f"{label_name}: {metric_title}")
            axis.set_xlabel("Время, с")
            axis.set_ylabel(metric_title)
            axis.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right")
    figure.suptitle("Диагностика подозрительных моментов во времени")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _select_demo_runs(
    runs: pd.DataFrame,
    local_candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    selected_rows = []

    normal_runs = runs[runs["label_name"] == "normal"].sort_values(
        ["scenario_template", "run_id"]
    )
    if not normal_runs.empty:
        selected_rows.append(normal_runs.iloc[0])

    anomaly_run = _select_anomaly_run(runs, local_candidates)
    if anomaly_run is not None:
        selected_rows.append(anomaly_run)

    if not selected_rows:
        selected_rows.append(runs.sort_values("run_id").iloc[0])

    return pd.DataFrame(selected_rows).drop_duplicates("run_id")


def _select_anomaly_run(
    runs: pd.DataFrame,
    local_candidates: pd.DataFrame | None,
) -> pd.Series | None:
    anomaly_runs = runs[runs["label_name"] == "anomaly"]
    if anomaly_runs.empty:
        return None

    if local_candidates is not None and not local_candidates.empty:
        candidate_runs = (
            local_candidates.sort_values(
                ["anomaly_score", "frame_score"],
                ascending=False,
            )["run_id"]
            .astype(str)
            .drop_duplicates()
        )
        for run_id in candidate_runs:
            matched = anomaly_runs[anomaly_runs["run_id"] == run_id]
            if not matched.empty:
                return matched.iloc[0]

    return anomaly_runs.sort_values(["scenario_template", "run_id"]).iloc[0]


def _run_candidates(
    local_candidates: pd.DataFrame | None,
    run_id: str,
) -> pd.DataFrame:
    if local_candidates is None or local_candidates.empty:
        return pd.DataFrame()
    if "run_id" not in local_candidates.columns:
        return pd.DataFrame()
    return local_candidates[local_candidates["run_id"].astype(str) == run_id]


def _select_agent_ids(
    trajectories: pd.DataFrame,
    run_candidates: pd.DataFrame,
    max_agents: int,
) -> list[int]:
    selected_ids = []
    if not run_candidates.empty and "top_agent_ids" in run_candidates.columns:
        top_rows = run_candidates.sort_values("frame_score", ascending=False).head(3)
        for value in top_rows["top_agent_ids"]:
            selected_ids.extend(_parse_agent_ids(value))

    all_ids = sorted(int(agent_id) for agent_id in trajectories["agent_id"].unique())
    for agent_id in all_ids:
        if agent_id not in selected_ids:
            selected_ids.append(agent_id)
        if len(selected_ids) >= max_agents:
            break
    return selected_ids[:max_agents]


def _parse_agent_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    result = []
    for item in str(value).split(";"):
        item = item.strip()
        if item.isdigit():
            result.append(int(item))
    return result


def _draw_direction_arrow(axis: plt.Axes, agent_path: pd.DataFrame, color: str) -> None:
    if len(agent_path) < 3:
        return
    first_index = max(0, int(len(agent_path) * 0.55) - 1)
    second_index = min(len(agent_path) - 1, first_index + 2)
    start = agent_path.iloc[first_index]
    end = agent_path.iloc[second_index]
    axis.annotate(
        "",
        xy=(end["x"], end["y"]),
        xytext=(start["x"], start["y"]),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 1.0,
            "alpha": 0.8,
        },
    )


def _draw_candidate_points(
    axis: plt.Axes,
    trajectories: pd.DataFrame,
    run_candidates: pd.DataFrame,
) -> None:
    if run_candidates.empty:
        return

    label_was_added = False
    top_candidates = run_candidates.sort_values("frame_score", ascending=False).head(3)
    for candidate in top_candidates.to_dict("records"):
        step = candidate.get("frame_step")
        agent_ids = _parse_agent_ids(candidate.get("top_agent_ids"))
        if step is None or not agent_ids:
            continue
        points = trajectories[
            (trajectories["step"] == step) & (trajectories["agent_id"].isin(agent_ids))
        ]
        if points.empty:
            continue
        axis.scatter(
            points["x"],
            points["y"],
            marker="D",
            s=44,
            color="#f2c94c",
            edgecolor="#222222",
            linewidth=0.5,
            label="подозрительный момент" if not label_was_added else None,
            zorder=6,
        )
        label_was_added = True


def _draw_candidate_steps(axis: plt.Axes, run_candidates: pd.DataFrame) -> None:
    if run_candidates.empty or "frame_time_s" not in run_candidates.columns:
        return

    top_candidates = run_candidates.sort_values("frame_score", ascending=False).head(5)
    label_was_added = False
    for time_s in sorted(top_candidates["frame_time_s"].astype(float).unique()):
        axis.axvline(
            time_s,
            color="#f2c94c",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="кандидат" if not label_was_added else None,
        )
        label_was_added = True


def _trajectory_legend_handles(has_candidates: bool) -> list[plt.Line2D]:
    handles = [
        plt.Line2D([0], [0], color="#4c78a8", lw=1.5, label="траектория нормы"),
        plt.Line2D([0], [0], color="#e15759", lw=1.5, label="траектория аномалии"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#59a14f",
            label="старт",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#222222",
            label="конец симуляции",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#e15759",
            label="цель",
        ),
    ]
    if has_candidates:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor="#f2c94c",
                markeredgecolor="#222222",
                label="подозрительный момент",
            )
        )
    return handles


def _build_frame_timeline(
    trajectories: pd.DataFrame,
    close_distance: float = 0.75,
) -> pd.DataFrame:
    rows = []
    for step, frame in trajectories.groupby("step", sort=True):
        coordinates = frame[["x", "y"]].to_numpy(dtype=float)
        rows.append(
            {
                "step": int(step),
                "time_s": float(frame["time_s"].iloc[0]),
                "mean_force": float(frame["force_norm"].mean()),
                "max_force": float(frame["force_norm"].max()),
                "mean_speed": float(frame["speed"].mean()),
                "close_pair_count": _count_close_pairs(coordinates, close_distance),
            }
        )
    return pd.DataFrame(rows)


def _count_close_pairs(coordinates: np.ndarray, close_distance: float) -> int:
    count = 0
    for left_index in range(len(coordinates)):
        for right_index in range(left_index + 1, len(coordinates)):
            distance = float(
                np.linalg.norm(coordinates[left_index] - coordinates[right_index])
            )
            if distance < close_distance:
                count += 1
    return count


def plot_feature_boxplots_by_label(
    data: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    feature_names: dict[str, str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_features = _select_features_for_plot(feature_columns)
    labels = _ordered_label_names(data["label_name"].unique())

    figure, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 7),
        squeeze=False,
    )
    for axis, feature in zip(axes.ravel(), selected_features, strict=False):
        values_by_label = [
            data.loc[data["label_name"] == label, feature].to_numpy(dtype=float)
            for label in labels
        ]
        axis.boxplot(
            values_by_label,
            tick_labels=[_label_name_ru(label) for label in labels],
        )
        axis.set_title(_feature_name(feature, feature_names))
        axis.tick_params(axis="x", rotation=20)

    for axis in axes.ravel()[len(selected_features) :]:
        axis.axis("off")

    figure.suptitle("Сравнение понятных признаков по меткам")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
    feature_names: dict[str, str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(
        correlation_matrix.to_numpy(dtype=float),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axis.set_xticks(np.arange(len(correlation_matrix.columns)))
    axis.set_yticks(np.arange(len(correlation_matrix.index)))
    x_labels = [_feature_name(column, feature_names) for column in correlation_matrix]
    y_labels = [
        _feature_name(column, feature_names)
        for column in correlation_matrix.index
    ]
    axis.set_xticklabels(x_labels, rotation=90, fontsize=7)
    axis.set_yticklabels(y_labels, fontsize=7)
    axis.set_title("Корреляции между признаками")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_outlier_summary(
    outlier_summary: pd.DataFrame,
    output_path: Path,
    feature_names: dict[str, str] | None = None,
    top_n: int = 10,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = outlier_summary.sort_values(
        "outlier_fraction",
        ascending=False,
    ).head(top_n)

    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [
        _feature_name(feature, feature_names)
        for feature in plot_data["feature"]
    ]
    axis.barh(labels, plot_data["outlier_fraction"], color="#f28e2b")
    axis.invert_yaxis()
    axis.set_title("Доля выбросов по признакам")
    axis.set_xlabel("Доля выбросов")
    axis.set_ylabel("Признак")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_anomaly_score_by_label(
    predictions: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = _ordered_label_names(predictions["label_name"].unique())
    values_by_label = [
        predictions.loc[
            predictions["label_name"] == label,
            "anomaly_score",
        ].to_numpy(dtype=float)
        for label in labels
    ]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.boxplot(
        values_by_label,
        tick_labels=[_label_name_ru(label) for label in labels],
    )
    axis.set_title("Показатель аномальности по меткам")
    axis.set_xlabel("Метка запуска")
    axis.set_ylabel("Оценка аномальности")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _select_features_for_plot(feature_columns: list[str]) -> list[str]:
    preferred_features = [
        "speed_mean",
        "force_max",
        "density_proxy_max",
        "close_pair_count_max",
        "low_speed_fraction",
        "goal_progress_mean",
        "congestion_index",
        "contact_density_index",
    ]
    selected = [feature for feature in preferred_features if feature in feature_columns]
    for feature in feature_columns:
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= 8:
            break
    return selected[:8]


def _feature_name(feature: str, feature_names: dict[str, str] | None) -> str:
    if feature_names is None:
        return feature
    return feature_names.get(feature, feature)


def _label_name_ru(label_name: str) -> str:
    return {
        "normal": "норма",
        "anomaly": "аномалия",
    }.get(label_name, label_name)


def _ordered_label_names(values: list[str] | np.ndarray) -> list[str]:
    return sorted((str(value) for value in values), key=_label_order)


def _label_order(label_name: str) -> tuple[int, str]:
    order = {
        "normal": 0,
        "anomaly": 1,
    }
    return (order.get(str(label_name), 99), str(label_name))


def save_empty_plot(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.set_title("Демонстрационный график")
    axis.set_xlabel("Запуск")
    axis.set_ylabel("Показатель")
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def count_rows(data: pd.DataFrame) -> int:
    return len(data)
