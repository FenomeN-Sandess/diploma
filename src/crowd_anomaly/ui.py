from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from crowd_anomaly.core.export import compute_run_metrics, trajectories_to_frame
from crowd_anomaly.core.simulation import SimulationResult, run_simulation
from crowd_anomaly.scenarios import ScenarioTemplate, build_scenario, load_scenarios

DEFAULT_SCENARIOS_PATH = Path("configs/scenarios.yaml")
DEFAULT_OUTPUTS_PATH = Path("outputs")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EDA_FIGURE_ORDER = [
    "label_distribution.png",
    "split_distribution.png",
    "feature_boxplots.png",
    "anomaly_score_distribution.png",
    "correlation_heatmap.png",
    "outlier_summary.png",
]


def render_home() -> None:
    st.set_page_config(
        page_title="Синтетический стенд анализа аномалий",
        layout="wide",
    )

    st.title("Синтетический стенд анализа аномальных сценариев движения толпы")
    st.write(
        "Данные в проекте синтетические: они создаются симуляцией на основе "
        "модели социальных сил. Основная задача - определить, является ли "
        "запуск симуляции нормальным или аномальным. Интерфейс предназначен только для "
        "демонстрации идеи проекта."
    )

    scenarios = _load_scenarios_for_ui(DEFAULT_SCENARIOS_PATH)
    outputs_path = Path(
        st.sidebar.text_input(
            "Папка с готовыми outputs",
            value=str(DEFAULT_OUTPUTS_PATH),
        )
    )
    selected_template = _scenario_selector(scenarios)
    seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
    )
    run_clicked = st.sidebar.button("Запустить симуляцию")

    (
        simulation_tab,
        forecast_tab,
        anomaly_tab,
        eda_tab,
        baseline_tab,
        limitations_tab,
    ) = st.tabs(
        [
            "Симуляция",
            "Прогноз",
            "Аномалии без учителя",
            "Результаты EDA",
            "Контрольные модели",
            "Ограничения",
        ]
    )

    with simulation_tab:
        _render_simulation_block(
            template=selected_template,
            seed=int(seed),
            run_clicked=run_clicked,
        )

    with forecast_tab:
        _render_forecast_block(outputs_path)

    with anomaly_tab:
        _render_unsupervised_anomaly_block(outputs_path)

    with eda_tab:
        _render_eda_block(outputs_path)

    with baseline_tab:
        _render_baseline_block(outputs_path)

    with limitations_tab:
        _render_limitations_block()


def build_simulation_figure(result: SimulationResult) -> plt.Figure:
    trajectories = trajectories_to_frame(result)
    figure, axis = plt.subplots(figsize=(9, 5))

    for _, agent_track in trajectories.groupby("agent_id"):
        ordered_track = agent_track.sort_values("step")
        axis.plot(
            ordered_track["x"],
            ordered_track["y"],
            linewidth=0.9,
            alpha=0.55,
        )
        axis.scatter(
            ordered_track["x"].iloc[0],
            ordered_track["y"].iloc[0],
            s=8,
            color="#4c78a8",
            alpha=0.5,
        )
        axis.scatter(
            ordered_track["x"].iloc[-1],
            ordered_track["y"].iloc[-1],
            s=10,
            color="#f28e2b",
            alpha=0.6,
        )

    for goal in result.scenario.goals:
        axis.scatter(
            goal.position.x,
            goal.position.y,
            marker="x",
            s=90,
            color="#d62728",
            linewidths=2,
            label="Цель",
        )

    width, height = result.scenario.world_size
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(result.scenario.name)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.25)

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles[:1], labels[:1], loc="upper right")

    figure.tight_layout()
    return figure


def _load_scenarios_for_ui(path: Path) -> list[ScenarioTemplate]:
    if not path.exists():
        st.error(f"Файл сценариев не найден: {path}")
        return []
    return load_scenarios(path)


def _scenario_selector(scenarios: list[ScenarioTemplate]) -> ScenarioTemplate | None:
    if not scenarios:
        return None

    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    selected_name = st.sidebar.selectbox(
        "Сценарий",
        options=list(scenario_by_name),
        format_func=lambda name: f"{name} ({scenario_by_name[name].label})",
    )
    selected_template = scenario_by_name[selected_name]
    st.sidebar.caption(selected_template.description)
    return selected_template


def _render_simulation_block(
    template: ScenarioTemplate | None,
    seed: int,
    run_clicked: bool,
) -> None:
    st.subheader("Симуляция")
    if template is None:
        st.warning("Нет доступных сценариев для запуска симуляции.")
        return

    st.write(
        "Выберите сценарий и seed в боковой панели, затем запустите симуляцию. "
        "График показывает траектории агентов в синтетическом пространстве."
    )

    if not run_clicked:
        st.info("Нажмите кнопку \"Запустить симуляцию\", чтобы построить график.")
        return

    scenario = build_scenario(template, seed=seed)
    result = run_simulation(scenario)
    metrics = compute_run_metrics(result)

    st.pyplot(build_simulation_figure(result))
    st.dataframe(
        _simulation_metrics_frame(metrics),
        width="stretch",
        hide_index=True,
    )


def _simulation_metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    selected_metrics = [
        "agent_count",
        "mean_speed",
        "max_speed",
        "mean_force",
        "max_force",
        "final_goal_reached_fraction",
        "duration_s",
    ]
    return pd.DataFrame(
        {
            "Метрика": selected_metrics,
            "Значение": [metrics[name] for name in selected_metrics],
        }
    )


def _render_eda_block(outputs_path: Path) -> None:
    st.subheader("Результаты EDA")
    dataset_dir = outputs_path / "datasets" / "demo"
    feature_dir = outputs_path / "features" / "demo"
    anomaly_dir = outputs_path / "anomaly" / "demo"
    eda_dir = outputs_path / "eda" / "demo"
    report_path = eda_dir / "eda_report.md"
    figures_dir = eda_dir / "figures"
    command = _build_eda_command(dataset_dir, feature_dir, anomaly_dir, eda_dir)
    missing_inputs = _missing_paths([dataset_dir, feature_dir])

    if not report_path.exists():
        st.info(
            "Готовый EDA-отчёт не найден. Его можно создать командой "
            "`python scripts/run_final_eda.py --dataset outputs/datasets/demo "
            "--features outputs/features/demo --anomaly outputs/anomaly/demo "
            "--output outputs/eda/demo`."
        )
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
            return
        if not _run_command_button(
            label="Сформировать EDA-отчёт",
            command=command,
            success_message="EDA-отчёт сформирован.",
            key="run_eda_report",
        ):
            return
        if not report_path.exists():
            st.error(f"Команда выполнена, но отчёт не найден: {report_path}")
            return
    else:
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
        else:
            _run_command_button(
                label="Обновить EDA-отчёт",
                command=command,
                success_message="EDA-отчёт обновлён.",
                key="refresh_eda_report",
            )

    st.caption(f"Отчёт: {report_path}")
    report_text = report_path.read_text(encoding="utf-8")
    st.markdown(_short_report_excerpt(report_text))

    figure_paths = _eda_figure_paths(figures_dir)
    if not figure_paths:
        st.warning(f"Графики EDA не найдены в папке {figures_dir}.")
        return

    for figure_path in figure_paths:
        st.image(
            str(figure_path),
            caption=figure_path.name,
            width="stretch",
        )


def _render_forecast_block(outputs_path: Path) -> None:
    st.subheader("Прогнозирование динамики")
    dataset_dir = outputs_path / "datasets" / "demo"
    forecast_dir = outputs_path / "forecast" / "demo"
    metrics_path = forecast_dir / "forecast_metrics.json"
    metrics_by_split_path = forecast_dir / "forecast_metrics_by_split.csv"
    predictions_path = forecast_dir / "forecast_predictions.csv"
    command = _build_forecast_command(dataset_dir, forecast_dir)
    missing_inputs = _missing_paths([dataset_dir])

    if not metrics_path.exists():
        st.info(
            "Результаты прогноза не найдены. Их можно создать командой "
            "`python scripts/train_forecast.py --dataset outputs/datasets/demo "
            "--output outputs/forecast/demo`."
        )
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
            return
        if not _run_command_button(
            label="Построить прогноз",
            command=command,
            success_message="Прогнозирование выполнено.",
            key="run_forecast",
        ):
            return
        if not metrics_path.exists():
            st.error(
                "Команда выполнена, но forecast_metrics.json не найден: "
                f"{metrics_path}"
            )
            return
    else:
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
        else:
            _run_command_button(
                label="Обновить прогноз",
                command=command,
                success_message="Прогнозирование обновлено.",
                key="refresh_forecast",
            )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.write(
        "Прогноз строится по агрегированным временным рядам: среднее положение, "
        "средняя скорость и proxy плотности."
    )
    st.json(
        {
            "window_size": metrics.get("window_size"),
            "horizon": metrics.get("horizon"),
            "models": list((metrics.get("models") or {}).keys()),
        }
    )
    if metrics_by_split_path.exists():
        st.write("Метрики прогноза:")
        st.dataframe(
            pd.read_csv(metrics_by_split_path),
            width="stretch",
            hide_index=True,
        )
    if predictions_path.exists():
        st.write("Первые строки прогнозов:")
        st.dataframe(
            pd.read_csv(predictions_path).head(100),
            width="stretch",
            hide_index=True,
        )


def _render_unsupervised_anomaly_block(outputs_path: Path) -> None:
    st.subheader("Обнаружение аномалий без учителя")
    feature_dir = outputs_path / "features" / "demo"
    anomaly_dir = outputs_path / "anomaly" / "demo"
    metrics_path = anomaly_dir / "unsupervised_metrics.json"
    metrics_by_split_path = anomaly_dir / "unsupervised_metrics_by_split.csv"
    method_comparison_path = anomaly_dir / "unsupervised_method_comparison.csv"
    predictions_path = anomaly_dir / "unsupervised_predictions.csv"
    local_candidates_path = anomaly_dir / "local_anomaly_candidates.csv"
    score_distribution_path = anomaly_dir / "unsupervised_score_distribution.png"
    command = _build_unsupervised_anomaly_command(feature_dir, anomaly_dir)
    missing_inputs = _missing_paths([feature_dir])

    if not metrics_path.exists():
        st.info(
            "Результаты модели без учителя не найдены. Их можно создать командой "
            "`python scripts/run_unsupervised_anomaly.py "
            "--features outputs/features/demo "
            "--output outputs/anomaly/demo`."
        )
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
            return
        if not _run_command_button(
            label="Запустить модель без учителя",
            command=command,
            success_message="Модель без учителя выполнена.",
            key="run_unsupervised_anomaly",
        ):
            return
        if not metrics_path.exists():
            st.error(
                "Команда выполнена, но unsupervised_metrics.json не найден: "
                f"{metrics_path}"
            )
            return
    else:
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
        else:
            _run_command_button(
                label="Обновить модель без учителя",
                command=command,
                success_message="Модель без учителя обновлена.",
                key="refresh_unsupervised_anomaly",
            )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.write(
        "Модель без учителя обучается на train-строках без использования "
        "`label` как признака и без отбора строк по метке. Затем для каждого "
        "запуска считается показатель аномальности на основе ошибки восстановления "
        "по методу главных компонент, силы, плотности и близких контактов."
    )
    st.json(
        {
            "model": metrics.get("model"),
            "training_uses_label_as_feature": metrics.get(
                "training_uses_label_as_feature"
            ),
            "training_reference": metrics.get("training_reference"),
        }
    )
    if metrics_by_split_path.exists():
        st.write("Метрики по split:")
        st.dataframe(
            pd.read_csv(metrics_by_split_path),
            width="stretch",
            hide_index=True,
        )
    if method_comparison_path.exists():
        st.write(
            "Сравнение простых способов поиска аномалий. Основной вариант проекта - "
            "`hybrid_iforest_physics`; остальные нужны только для контроля."
        )
        st.dataframe(
            pd.read_csv(method_comparison_path),
            width="stretch",
            hide_index=True,
        )
    if predictions_path.exists():
        st.write("Оценки запусков:")
        st.dataframe(
            pd.read_csv(predictions_path),
            width="stretch",
            hide_index=True,
        )
    if local_candidates_path.exists():
        st.write(
            "Кандидаты на аномальные кадры и агентов. Это диагностическая подсказка, "
            "а не точная ручная разметка."
        )
        st.dataframe(
            pd.read_csv(local_candidates_path),
            width="stretch",
            hide_index=True,
        )
    if score_distribution_path.exists():
        st.image(
            str(score_distribution_path),
            caption="Распределение показателя аномальности без учителя",
            width="stretch",
        )


def _render_baseline_block(outputs_path: Path) -> None:
    st.subheader("Контрольные модели с учителем")
    feature_dir = outputs_path / "features" / "demo"
    model_dir = outputs_path / "models" / "demo"
    metrics_path = model_dir / "metrics.json"
    metrics_by_split_path = model_dir / "metrics_by_split.csv"
    feature_importance_path = model_dir / "feature_importance.csv"
    score_distribution_path = model_dir / "score_distribution.png"
    command = _build_baseline_command(feature_dir, model_dir)
    missing_inputs = _missing_paths([feature_dir])

    if not metrics_path.exists():
        st.info(
            "Результаты контрольных моделей не найдены. Их можно создать командой "
            "`python scripts/train_baseline.py --features outputs/features/demo "
            "--output outputs/models/demo`."
        )
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
            return
        if not _run_command_button(
            label="Обучить контрольные модели",
            command=command,
            success_message="Baseline ML обучен.",
            key="run_baseline_ml",
        ):
            return
        if not metrics_path.exists():
            st.error(f"Команда выполнена, но metrics.json не найден: {metrics_path}")
            return
    else:
        if missing_inputs:
            _show_missing_inputs(missing_inputs)
        else:
            _run_command_button(
                label="Обновить контрольные модели",
                command=command,
                success_message="Baseline ML обновлён.",
                key="refresh_baseline_ml",
            )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_model = metrics.get("best_model_by_val_f1")
    test_result = metrics.get("test_result_for_best_model") or {}

    st.metric("Лучшая модель по val F1", best_model or "нет данных")
    if test_result:
        st.write("Показатели качества на тестовой части:")
        st.dataframe(
            pd.DataFrame([test_result]),
            width="stretch",
            hide_index=True,
        )

    warnings = metrics.get("warnings") or []
    if warnings:
        st.warning("Предупреждения: " + "; ".join(warnings))

    if metrics_by_split_path.exists():
        st.write("Метрики по train/val/test:")
        st.dataframe(
            pd.read_csv(metrics_by_split_path),
            width="stretch",
            hide_index=True,
        )

    if feature_importance_path.exists():
        st.write("Важность признаков:")
        st.dataframe(
            pd.read_csv(feature_importance_path),
            width="stretch",
            hide_index=True,
        )

    if score_distribution_path.exists():
        st.image(
            str(score_distribution_path),
            caption="Распределение показателя аномальности",
            width="stretch",
        )
    else:
        st.warning(
            "График распределения показателя не найден: "
            f"{score_distribution_path}"
        )


def _render_limitations_block() -> None:
    st.subheader("Ограничения")
    st.markdown(
        """
- данные синтетические;
- внешний датасет не используется;
- покадровые метки наследуются от метки всего запуска;
- локальные кадры/агенты выводятся только как кандидаты для интерпретации;
- прогноз строится по агрегированному состоянию всей толпы;
- модель без учителя проверяется на синтетических метках только после обучения;
- корреляции не доказывают причинность;
- перенос на реальные толпы требует внешней валидации.
"""
    )


def _short_report_excerpt(report_text: str, max_chars: int = 1800) -> str:
    if len(report_text) <= max_chars:
        return report_text
    return (
        report_text[:max_chars].rstrip()
        + "\n\n_Отчёт сокращён для просмотра в интерфейсе._"
    )


def _build_eda_command(
    dataset_dir: Path,
    feature_dir: Path,
    anomaly_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "python",
        "scripts/run_final_eda.py",
        "--dataset",
        str(dataset_dir),
        "--features",
        str(feature_dir),
        "--anomaly",
        str(anomaly_dir),
        "--output",
        str(output_dir),
    ]


def _build_forecast_command(dataset_dir: Path, output_dir: Path) -> list[str]:
    return [
        "python",
        "scripts/train_forecast.py",
        "--dataset",
        str(dataset_dir),
        "--output",
        str(output_dir),
    ]


def _build_unsupervised_anomaly_command(
    feature_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "python",
        "scripts/run_unsupervised_anomaly.py",
        "--features",
        str(feature_dir),
        "--output",
        str(output_dir),
    ]


def _build_baseline_command(feature_dir: Path, output_dir: Path) -> list[str]:
    return [
        "python",
        "scripts/train_baseline.py",
        "--features",
        str(feature_dir),
        "--output",
        str(output_dir),
    ]


def _run_command_button(
    label: str,
    command: list[str],
    success_message: str,
    key: str,
) -> bool:
    st.caption(f"Команда: `{subprocess.list2cmdline(command)}`")
    if len(command) > 1 and command[1].startswith("scripts/"):
        script_path = PROJECT_ROOT / command[1]
        if not script_path.exists():
            st.error(f"Файл команды не найден: {script_path}")
            return False

    if not st.button(label, key=key):
        return False

    with st.spinner("Выполняю команду..."):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    command_output = "\n".join(
        output for output in [result.stdout.strip(), result.stderr.strip()] if output
    )
    if result.returncode != 0:
        st.error(f"Команда завершилась с кодом {result.returncode}.")
        if command_output:
            st.code(command_output, language="text")
        return False

    st.success(success_message)
    if command_output:
        with st.expander("Вывод команды"):
            st.code(command_output, language="text")
    return True


def _eda_figure_paths(figures_dir: Path) -> list[Path]:
    if not figures_dir.exists():
        return []

    existing_figures = {path.name: path for path in figures_dir.glob("*.png")}
    ordered_figures = [
        existing_figures.pop(figure_name)
        for figure_name in EDA_FIGURE_ORDER
        if figure_name in existing_figures
    ]
    return ordered_figures + [
        existing_figures[name] for name in sorted(existing_figures)
    ]


def _missing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _show_missing_inputs(paths: list[Path]) -> None:
    formatted_paths = ", ".join(f"`{path}`" for path in paths)
    st.error(f"Не найдены входные данные: {formatted_paths}.")
