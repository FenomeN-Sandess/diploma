# Program

`Program` — это программная часть ВКР. Здесь лежит код, который создаёт
синтетический датасет движения толпы, считает признаки, запускает модели,
строит графики и собирает отчёты.

Проект сделан как простой воспроизводимый pipeline: один шаг сохраняет файлы в
`outputs`, а следующий шаг читает эти файлы.

## Что реализовано

- симуляция движения агентов по учебной реализации модели Хелбинга;
- нормальные и аномальные сценарии движения толпы;
- создание синтетического датасета траекторий;
- расчёт признаков по траекториям;
- простой прогноз динамики толпы по прошлым состояниям;
- поиск аномальных запусков без учителя;
- контрольные модели с учителем для сравнения;
- EDA-отчёт по данным, траекториям и признакам;
- итоговый отчёт по всему pipeline;
- тесты основных частей проекта.

## Структура

```text
Program/
├── configs/
│   ├── dataset.yaml
│   └── scenarios.yaml
├── scripts/
│   ├── build_dataset.py
│   ├── build_features.py
│   ├── train_forecast.py
│   ├── run_unsupervised_anomaly.py
│   ├── run_final_eda.py
│   ├── train_baseline.py
│   ├── build_project_report.py
│   └── run_all.py
├── src/crowd_anomaly/
├── tests/
└── outputs/
```

Jupyter Notebook из проекта убран, чтобы не дублировать автоматический EDA.
Основной анализ теперь создаётся скриптами и хранится в `outputs`.

## Что здесь находится

| Файл/папка | Зачем нужен |
|---|---|
| `configs/` | Настройки датасета и сценариев движения. |
| `scripts/` | Команды для запуска отдельных шагов и полного pipeline. |
| `src/` | Исходный код Python-пакета `crowd_anomaly`. |
| `tests/` | Автоматические проверки проекта. |
| `outputs/` | Автоматически создаваемые результаты, таблицы, графики и отчёты. |
| `app.py` | Входной файл простой Streamlit-страницы. |
| `pyproject.toml` | Описание пакета и зависимостей. |
| `Dockerfile`, `docker-compose.yml` | Запуск проекта через Docker. |

## Главный запуск

```bash
cd Program
python -m pip install -e ".[dev]"
python scripts/run_all.py
```

После запуска создаются:

| Папка | Что внутри |
|---|---|
| `outputs/datasets/demo` | Синтетический датасет, траектории и метрики запусков. |
| `outputs/features/demo` | Признаки, рассчитанные по траекториям. |
| `outputs/forecast/demo` | Метрики и предсказания прогноза динамики. |
| `outputs/anomaly/demo` | Предсказания и метрики поиска аномалий без учителя. |
| `outputs/eda/demo` | EDA-отчёт, таблицы и графики по данным. |
| `outputs/models/demo` | Контрольные модели с учителем и их метрики. |
| `outputs/final_report/demo` | Итоговый отчёт по всей программной работе. |

## Методы поиска аномалий без учителя

В блоке `run_unsupervised_anomaly.py` сравниваются несколько способов расчёта
`anomaly_score`. Все они работают на уровне целого запуска и не получают
`label` как входной признак.

| Метод | Чем отличается | F1 на test |
|---|---|---:|
| `pca_physics` | Основной вариант: ошибка восстановления PCA + физический показатель по силе, плотности и близким парам. | 0.774 |
| `pca_reconstruction` | Только PCA: запуск подозрителен, если его признаки плохо восстанавливаются. | 0.759 |
| `isolation_forest` | Готовый алгоритм поиска выбросов в таблице признаков. | 0.727 |
| `physics_only` | Только физический показатель: `force_max`, `density_proxy_max`, `close_pair_count_max`. | 0.724 |
| `hybrid_iforest_physics` | Isolation Forest вместе с физическим показателем. | 0.667 |

Итоговым выбран `pca_physics`: он дал лучший F1 на тестовой части и проще
объясняется через физику движения толпы. Полная таблица лежит в
`outputs/anomaly/demo/unsupervised_method_comparison.csv`.

## Как шаги связаны между собой

| Шаг | Что создаётся | Следующий шаг использует |
|---|---|---|
| `build_dataset.py` | `outputs/datasets/demo` | `build_features.py`, `train_forecast.py`, `run_final_eda.py` |
| `build_features.py` | `outputs/features/demo` | `run_unsupervised_anomaly.py`, `train_baseline.py`, `run_final_eda.py` |
| `train_forecast.py` | `outputs/forecast/demo` | `build_project_report.py` |
| `run_unsupervised_anomaly.py` | `outputs/anomaly/demo` | `run_final_eda.py`, `build_project_report.py` |
| `run_final_eda.py` | `outputs/eda/demo` | чтение EDA и `build_project_report.py` |
| `train_baseline.py` | `outputs/models/demo` | `build_project_report.py` |
| `build_project_report.py` | `outputs/final_report/demo` | финальное чтение результатов |

## Отдельные команды

Создать датасет:

```bash
python scripts/build_dataset.py --config configs/dataset.yaml --scenarios configs/scenarios.yaml --output outputs/datasets/demo
```

Посчитать признаки:

```bash
python scripts/build_features.py --dataset outputs/datasets/demo --output outputs/features/demo
```

Построить прогноз динамики:

```bash
python scripts/train_forecast.py --dataset outputs/datasets/demo --output outputs/forecast/demo
```

Запустить поиск аномалий без учителя:

```bash
python scripts/run_unsupervised_anomaly.py --features outputs/features/demo --output outputs/anomaly/demo
```

Создать EDA-отчёт:

```bash
python scripts/run_final_eda.py --dataset outputs/datasets/demo --features outputs/features/demo --anomaly outputs/anomaly/demo --output outputs/eda/demo
```

Создать итоговый отчёт по всему pipeline:

```bash
python scripts/build_project_report.py --dataset outputs/datasets/demo --features outputs/features/demo --forecast outputs/forecast/demo --anomaly outputs/anomaly/demo --eda outputs/eda/demo --models outputs/models/demo --output outputs/final_report/demo
```

## Какие отчёты читать

| Отчёт | Для чего нужен |
|---|---|
| `outputs/eda/demo/eda_report.md` | Разведывательный анализ данных: датасет, траектории, временные пики, признаки, баланс классов, графики. |
| `outputs/final_report/demo/project_report.md` | Общая сводка всей работы pipeline: данные, прогноз, аномалии, baseline-модели, метрики и графики. |

EDA — это только анализ данных. Полный отчёт по выполненной работе находится в
`outputs/final_report/demo/project_report.md`.

## Ограничения

- данные синтетические и создаются внутри проекта;
- внешний датасет реальных толп не используется;
- целевая метка задана на уровне всего запуска;
- точной ручной разметки аномального кадра или агента нет;
- прогноз строится по агрегированному состоянию толпы, а не по каждой отдельной траектории.

## Проверка

```bash
python -m pytest
python -m ruff check .
```

Перед защитой удобно выполнить полный запуск и потом открыть два основных
отчёта: `eda_report.md` и `project_report.md`.
