# Program/scripts

В этой папке находятся командные скрипты. Они нужны, чтобы запускать отдельные
части проекта из терминала: генерацию данных, признаки, прогноз, поиск аномалий
и отчеты.

## Что здесь находится

| Файл/папка | Зачем нужен |
|---|---|
| `run_all.py` | Полный запуск всего pipeline от датасета до результатов. |
| `build_dataset.py` | Создает синтетический датасет по конфигам. |
| `build_features.py` | Считает признаки по готовому датасету. |
| `train_forecast.py` | Строит прогноз динамики толпы. |
| `run_unsupervised_anomaly.py` | Запускает поиск аномалий без учителя. |
| `run_final_eda.py` | Строит EDA-отчет, таблицы, графики и словарь признаков. |
| `train_baseline.py` | Обучает контрольные модели с учителем для сравнения. |
| `build_project_report.py` | Собирает итоговый отчет по всем результатам pipeline. |

## Как это связано с проектом

Скрипты являются внешним входом в программу. Они вызывают функции из
`src/crowd_anomaly`, сохраняют результаты в `outputs` и позволяют повторить
расчеты без ручного запуска кода по частям.

## Что смотреть в первую очередь

1. `run_all.py` - главное оглавление всего проекта.
2. `build_dataset.py` - с чего начинается создание данных.
3. `run_unsupervised_anomaly.py` - как запускается основной поиск аномалий.
4. `build_project_report.py` - где собирается полный отчет по результатам.

## Как запускать

Полный запуск:

```bash
python scripts/run_all.py
```

Отдельные шаги:

```bash
python scripts/build_dataset.py --config configs/dataset.yaml --scenarios configs/scenarios.yaml --output outputs/datasets/demo
python scripts/build_features.py --dataset outputs/datasets/demo --output outputs/features/demo
python scripts/run_unsupervised_anomaly.py --features outputs/features/demo --output outputs/anomaly/demo
python scripts/build_project_report.py --output outputs/final_report/demo
```

Запускать команды нужно из папки `Program`.
