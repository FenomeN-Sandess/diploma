from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from crowd_anomaly.features import get_feature_description, get_feature_display_name


def build_project_report(
    dataset_dir: Path | str,
    feature_dir: Path | str,
    forecast_dir: Path | str,
    anomaly_dir: Path | str,
    eda_dir: Path | str,
    models_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    feature_path = Path(feature_dir)
    forecast_path = Path(forecast_dir)
    anomaly_path = Path(anomaly_dir)
    eda_path = Path(eda_dir)
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "project_report.md"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": dataset_path.as_posix(),
        "feature_dir": feature_path.as_posix(),
        "forecast_dir": forecast_path.as_posix(),
        "anomaly_dir": anomaly_path.as_posix(),
        "eda_dir": eda_path.as_posix(),
        "models_dir": models_path.as_posix(),
        "report_path": report_path.as_posix(),
    }

    context = _load_report_context(
        dataset_path=dataset_path,
        feature_path=feature_path,
        forecast_path=forecast_path,
        anomaly_path=anomaly_path,
        eda_path=eda_path,
        models_path=models_path,
    )
    report_path.write_text(
        _render_project_report(report_path, payload, context),
        encoding="utf-8",
    )
    (output_path / "project_report_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": output_path,
        "report_path": report_path,
        "metadata": payload,
    }


def _load_report_context(
    dataset_path: Path,
    feature_path: Path,
    forecast_path: Path,
    anomaly_path: Path,
    eda_path: Path,
    models_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_metadata": _read_json(dataset_path / "dataset_metadata.json"),
        "runs": _read_csv(dataset_path / "runs.csv"),
        "feature_schema": _read_json(feature_path / "feature_schema.json"),
        "feature_dictionary": _read_csv(
            eda_path / "tables" / "feature_dictionary.csv"
        ),
        "feature_summary": _read_csv(eda_path / "tables" / "feature_summary.csv"),
        "feature_label_summary": _read_csv(
            eda_path / "tables" / "feature_label_summary.csv"
        ),
        "label_counts": _read_csv(eda_path / "tables" / "label_counts.csv"),
        "split_counts": _read_csv(eda_path / "tables" / "split_counts.csv"),
        "scenario_summary": _read_csv(eda_path / "tables" / "scenario_summary.csv"),
        "trajectory_summary": _read_csv(
            eda_path / "tables" / "trajectory_summary.csv"
        ),
        "outlier_summary": _read_csv(eda_path / "tables" / "outlier_summary.csv"),
        "top_correlations": _read_csv(eda_path / "tables" / "top_correlations.csv"),
        "forecast_metrics": _read_csv(
            forecast_path / "forecast_metrics_by_split.csv"
        ),
        "forecast_metadata": _read_json(forecast_path / "forecast_metrics.json"),
        "anomaly_metrics": _read_csv(
            anomaly_path / "unsupervised_metrics_by_split.csv"
        ),
        "anomaly_metadata": _read_json(
            anomaly_path / "unsupervised_metrics.json"
        ),
        "method_comparison": _read_csv(
            anomaly_path / "unsupervised_method_comparison.csv"
        ),
        "baseline_metrics": _read_csv(models_path / "metrics_by_split.csv"),
        "baseline_metadata": _read_json(models_path / "metrics.json"),
        "feature_importance": _read_csv(models_path / "feature_importance.csv"),
    }


def _render_project_report(
    report_path: Path,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> str:
    runs = context["runs"]
    dataset_metadata = context["dataset_metadata"]
    feature_schema = context["feature_schema"]
    anomaly_metadata = context["anomaly_metadata"]
    baseline_metadata = context["baseline_metadata"]
    forecast_metadata = context["forecast_metadata"]

    quick_summary = _build_quick_summary(context)
    pipeline_table = _build_pipeline_table(payload)
    artifact_table = _build_artifact_table(payload, report_path)
    graph_section = _build_graph_section(payload, report_path)

    return f"""# Итоговый отчёт по pipeline

Этот отчёт собирается автоматически после полного запуска программы. Он нужен,
чтобы в одном месте посмотреть основные результаты: какие синтетические данные
были созданы, какие признаки посчитаны, как сработал прогноз, как найдены
аномалии и какие метрики получились.

Важно разделять два отчёта:

- `outputs/eda/demo/eda_report.md` — EDA, то есть разведывательный анализ данных.
- `outputs/final_report/demo/project_report.md` — полный отчёт по работе pipeline.

## Короткий итог

{_to_markdown_table(quick_summary)}

Данные в проекте синтетические. Они создаются симуляцией движения агентов по
модели социальных сил. Внешний датасет реальных толп здесь не используется.

## Как устроен pipeline

{_to_markdown_table(pipeline_table)}

Главная логика простая: каждый шаг создаёт файлы, а следующий шаг читает эти
файлы как вход. Поэтому результаты можно пересоздать полным запуском команды
`python scripts/run_all.py`.

## Датасет и траектории

Папка датасета: `{payload["dataset_dir"]}`.

Сводка из `dataset_metadata.json`:

{_metadata_summary(dataset_metadata, runs)}

Распределение normal/anomaly:

{_to_markdown_table(context["label_counts"])}

Распределение train/val/test:

{_to_markdown_table(context["split_counts"])}

Сценарии:

{_to_markdown_table(context["scenario_summary"])}

Сводка по траекториям:

{_to_markdown_table(context["trajectory_summary"])}

## Признаки

Признаки считаются по траекториям и описывают весь запуск целиком. В них входят
скорости, силы, расстояния, плотность, близкие контакты, прогресс к цели и
простые показатели затора.

Количество признаков: {_feature_count(feature_schema)}.

Короткий словарь признаков:

{_feature_dictionary_table(context)}

Самые заметные различия средних значений между normal и anomaly:

{_to_markdown_table(context["feature_label_summary"].head(10))}

Первые строки общей статистики по признакам:

{_to_markdown_table(context["feature_summary"].head(10))}

## Прогноз динамики

Прогноз здесь нужен как отдельная проверка динамики системы: модель смотрит на
несколько прошлых агрегированных состояний толпы и предсказывает следующее
состояние. Это не прогноз отдельной траектории каждого агента, а простой прогноз
состояния всей группы.

Параметры прогноза:

{_forecast_summary(forecast_metadata)}

Метрики прогноза:

{_to_markdown_table(context["forecast_metrics"])}

## Поиск аномалий без учителя

Основной результат по аномалиям лежит в `outputs/anomaly/demo`. Модель не
использует `label` как входной признак. Метки нужны только после обучения,
чтобы проверить качество на train/val/test.

Формула итогового показателя:

```text
{anomaly_metadata.get("anomaly_score_formula", "нет данных")}
```

Метрики:

{_to_markdown_table(context["anomaly_metrics"])}

Сравнение простых способов поиска аномалий:

{_to_markdown_table(context["method_comparison"])}

## Контрольные модели с учителем

Эти модели нужны для сравнения. Они показывают, насколько хорошо обычные модели
с учителем решают ту же задачу на тех же признаках. Главным результатом проекта
остаётся блок без учителя.

Лучшая модель по validation F1:
`{baseline_metadata.get("best_model_by_val_f1", "нет данных")}`.

Метрики моделей:

{_to_markdown_table(context["baseline_metrics"])}

Самые важные признаки по контрольным моделям:

{_to_markdown_table(_add_feature_names(context["feature_importance"]).head(10))}

## Основные графики

{graph_section}

## Где лежат результаты

{_to_markdown_table(artifact_table)}

## Итоговый вывод

Pipeline создаёт синтетические траектории, превращает их в признаки, строит
простой прогноз динамики, ищет аномальные запуски без учителя и сохраняет
метрики с графиками. EDA отвечает за понимание данных, а этот отчёт собирает
общую картину всей выполненной программной части.
"""


def _build_quick_summary(context: dict[str, Any]) -> pd.DataFrame:
    runs = context["runs"]
    anomaly_test = _pick_row(context["anomaly_metrics"], split="test")
    forecast_test = _pick_row(
        context["forecast_metrics"],
        split="test",
        model="simple_recurrent",
    )
    best_model = context["baseline_metadata"].get("best_model_by_val_f1")
    baseline_test = _pick_row(
        context["baseline_metrics"],
        split="test",
        model=best_model,
    )

    rows = [
        ("Количество запусков", len(runs)),
        ("Количество сценариев", runs["scenario_template"].nunique()),
        ("Уровень целевой метки", "весь запуск"),
        ("F1 anomaly test", anomaly_test.get("f1", "нет данных")),
        ("MAE forecast test", forecast_test.get("mae", "нет данных")),
        ("Лучшая baseline-модель", best_model or "нет данных"),
        ("F1 baseline test", baseline_test.get("f1", "нет данных")),
    ]
    return pd.DataFrame(rows, columns=["Показатель", "Значение"])


def _build_pipeline_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows = [
        (
            "1. Симуляция и датасет",
            "`outputs/datasets/demo`",
            "Используют признаки, прогноз и EDA.",
        ),
        (
            "2. Признаки",
            "`outputs/features/demo`",
            "Используют anomaly-модель, baseline-модели и EDA.",
        ),
        (
            "3. Прогноз динамики",
            "`outputs/forecast/demo`",
            "Используется в итоговом отчёте как проверка динамики.",
        ),
        (
            "4. Аномалии без учителя",
            "`outputs/anomaly/demo`",
            "Используют EDA и итоговый отчёт.",
        ),
        (
            "5. EDA",
            "`outputs/eda/demo/eda_report.md`",
            "Используется для анализа данных и графиков.",
        ),
        (
            "6. Контрольные модели",
            "`outputs/models/demo`",
            "Используются для сравнения с методом без учителя.",
        ),
        (
            "7. Итоговый отчёт",
            f"`{payload['report_path']}`",
            "Собирает результаты всех предыдущих шагов.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Шаг", "Что создаётся", "Кто использует"])


def _build_artifact_table(payload: dict[str, Any], report_path: Path) -> pd.DataFrame:
    rows = [
        ("Датасет", Path(payload["dataset_dir"]) / "runs.csv"),
        ("Траектории", Path(payload["dataset_dir"]) / "runs"),
        ("Признаки", Path(payload["feature_dir"]) / "features_run.csv"),
        ("Прогноз", Path(payload["forecast_dir"]) / "forecast_metrics_by_split.csv"),
        (
            "Аномалии",
            Path(payload["anomaly_dir"]) / "unsupervised_metrics_by_split.csv",
        ),
        ("EDA", Path(payload["eda_dir"]) / "eda_report.md"),
        ("Контрольные модели", Path(payload["models_dir"]) / "metrics_by_split.csv"),
    ]
    return pd.DataFrame(
        [(name, _markdown_link(path, report_path)) for name, path in rows],
        columns=["Раздел", "Файл или папка"],
    )


def _build_graph_section(payload: dict[str, Any], report_path: Path) -> str:
    figures = [
        (
            "Распределение normal/anomaly",
            Path(payload["eda_dir"]) / "figures" / "label_distribution.png",
            "Показывает баланс классов в синтетическом датасете.",
        ),
        (
            "Примеры траекторий",
            Path(payload["eda_dir"]) / "figures" / "trajectory_examples.png",
            (
                "Показывает старт, конец симуляции, направление движения, цель "
                "и кандидатов на подозрительные моменты."
            ),
        ),
        (
            "Диагностика во времени",
            Path(payload["eda_dir"]) / "figures" / "trajectory_anomaly_timeline.png",
            (
                "Показывает пики силы и близких контактов по времени. "
                "Жёлтые линии являются диагностикой, а не ручной разметкой."
            ),
        ),
        (
            "Метрики запусков",
            Path(payload["eda_dir"]) / "figures" / "run_metric_boxplots.png",
            "Сравнивает скорость, силу и достижение цели для normal/anomaly.",
        ),
        (
            "Признаки",
            Path(payload["eda_dir"]) / "figures" / "feature_boxplots.png",
            "Помогает увидеть различия признаков между классами.",
        ),
        (
            "Корреляции",
            Path(payload["eda_dir"]) / "figures" / "correlation_heatmap.png",
            "Показывает связи между признаками.",
        ),
        (
            "Anomaly score",
            Path(payload["anomaly_dir"]) / "unsupervised_score_distribution.png",
            "Показывает распределение итогового показателя аномальности.",
        ),
        (
            "Baseline score",
            Path(payload["models_dir"]) / "score_distribution.png",
            "Показывает оценки контрольных моделей с учителем.",
        ),
    ]

    blocks = []
    for title, path, note in figures:
        if not path.exists():
            continue
        blocks.append(
            f"### {title}\n\n![{title}]({_relative_path(path, report_path)})\n\n{note}"
        )
    if not blocks:
        return "_Графики не найдены. Сначала запустите `python scripts/run_all.py`._"
    return "\n\n".join(blocks)


def _metadata_summary(metadata: dict[str, Any], runs: pd.DataFrame) -> str:
    rows = [
        ("Название датасета", metadata.get("dataset_name", "нет данных")),
        ("Синтетические данные", metadata.get("synthetic_data", "нет данных")),
        ("Модель", metadata.get("model", "нет данных")),
        ("Количество запусков", metadata.get("run_count", len(runs))),
        ("Запусков на сценарий", metadata.get("runs_per_template", "нет данных")),
        ("Seed", metadata.get("seed", "нет данных")),
    ]
    return _to_markdown_table(pd.DataFrame(rows, columns=["Поле", "Значение"]))


def _forecast_summary(metadata: dict[str, Any]) -> str:
    rows = [
        ("Размер окна", metadata.get("window_size", "нет данных")),
        ("Горизонт", metadata.get("horizon", "нет данных")),
        ("Что прогнозируется", ", ".join(metadata.get("target_columns", []))),
        ("Количество окон", metadata.get("row_count", "нет данных")),
    ]
    return _to_markdown_table(pd.DataFrame(rows, columns=["Параметр", "Значение"]))


def _feature_dictionary_table(context: dict[str, Any]) -> str:
    feature_dictionary = context.get("feature_dictionary", pd.DataFrame())
    if not feature_dictionary.empty:
        return _to_markdown_table(feature_dictionary)

    schema = context.get("feature_schema", {})
    feature_columns = schema.get("feature_columns", [])
    rows = [
        {
            "feature": feature,
            "name": get_feature_display_name(feature),
            "description": get_feature_description(feature),
        }
        for feature in feature_columns
    ]
    return _to_markdown_table(pd.DataFrame(rows))


def _add_feature_names(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "feature" not in data.columns:
        return data
    if "feature_name" in data.columns:
        return data
    result = data.copy()
    result.insert(1, "feature_name", result["feature"].map(get_feature_display_name))
    result.insert(2, "description", result["feature"].map(get_feature_description))
    return result


def _feature_count(feature_schema: dict[str, Any]) -> int | str:
    feature_columns = feature_schema.get("feature_columns")
    if isinstance(feature_columns, list):
        return len(feature_columns)
    return "нет данных"


def _pick_row(
    table: pd.DataFrame,
    split: str,
    model: str | None = None,
) -> dict[str, Any]:
    if table.empty or "split" not in table:
        return {}
    rows = table[table["split"] == split]
    if model is not None and "model" in rows:
        rows = rows[rows["model"] == model]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_link(path: Path, report_path: Path) -> str:
    relative = _relative_path(path, report_path)
    return f"[`{relative}`]({relative})"


def _relative_path(path: Path, report_path: Path) -> str:
    return os.path.relpath(path, start=report_path.parent).replace("\\", "/")


def _to_markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "_Нет данных._"

    columns = list(data.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in data.to_dict("records"):
        values = [_format_value(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if pd.isna(value):
        return ""
    return str(value)
