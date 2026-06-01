import argparse
import sys
from pathlib import Path

from crowd_anomaly.anomaly import train_unsupervised_anomaly
from crowd_anomaly.baselines import train_baselines
from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.eda import build_eda_report
from crowd_anomaly.features import build_features
from crowd_anomaly.forecasting import train_forecast_models
from crowd_anomaly.multi_horizon import run_multi_horizon_forecast
from crowd_anomaly.predictive_risk import run_predictive_risk
from crowd_anomaly.reporting import build_project_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_anomaly_seeds import run_seeds  # noqa: E402
from scripts.run_loso_evaluation import run_loso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск полного минимального pipeline проекта."
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="Путь к YAML-конфигурации датасета.",
    )
    parser.add_argument(
        "--scenarios-config",
        default="configs/scenarios.yaml",
        help="Путь к YAML-конфигурации сценариев.",
    )
    parser.add_argument(
        "--dataset-output",
        default="outputs/datasets/demo",
        help="Папка для синтетического датасета.",
    )
    parser.add_argument(
        "--features-output",
        default="outputs/features/demo",
        help="Папка для признаков на уровне запуска.",
    )
    parser.add_argument(
        "--eda-output",
        default="outputs/eda/demo",
        help="Папка для EDA-отчёта.",
    )
    parser.add_argument(
        "--forecast-output",
        default="outputs/forecast/demo",
        help="Папка для результатов прогнозирования.",
    )
    parser.add_argument(
        "--anomaly-output",
        default="outputs/anomaly/demo",
        help="Папка для модели обнаружения аномалий без учителя.",
    )
    parser.add_argument(
        "--multi-horizon-output",
        default="outputs/multi_horizon/demo",
        help="Папка для многошагового прогноза (направление 1.1).",
    )
    parser.add_argument(
        "--predictive-risk-output",
        default="outputs/predictive_risk/demo",
        help="Папка для упреждающего сигнала риска (направление 1.2).",
    )
    parser.add_argument(
        "--models-output",
        default="outputs/models/demo",
        help="Папка для контрольных моделей с учителем.",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/final_report/demo",
        help="Папка для итогового отчёта по всему pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Переопределить seed из dataset.yaml.",
    )
    parser.add_argument(
        "--runs-per-template",
        type=int,
        default=None,
        help="Переопределить число запусков на сценарный шаблон.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_output = Path(args.dataset_output)
    features_output = Path(args.features_output)
    eda_output = Path(args.eda_output)
    forecast_output = Path(args.forecast_output)
    anomaly_output = Path(args.anomaly_output)
    multi_horizon_output = Path(args.multi_horizon_output)
    predictive_risk_output = Path(args.predictive_risk_output)
    models_output = Path(args.models_output)
    report_output = Path(args.report_output)

    seeds_output = anomaly_output / "seeds"
    loso_output = anomaly_output / "loso"

    print("1/11 Сборка синтетического датасета...")
    dataset_result = build_dataset(
        config_path=args.dataset_config,
        scenarios_path=args.scenarios_config,
        output_dir=dataset_output,
        seed=args.seed,
        runs_per_template=args.runs_per_template,
    )
    print(f"   runs: {len(dataset_result['runs'])}")
    print(f"   output: {dataset_output}")

    print("2/11 Расчёт признаков на уровне запуска...")
    feature_result = build_features(
        dataset_dir=dataset_output,
        output_dir=features_output,
    )
    feature_count = len(feature_result["feature_schema"]["feature_columns"])
    print(f"   features: {feature_count}")
    print(f"   output: {features_output}")

    print("3/11 Прогнозирование динамики...")
    forecast_result = train_forecast_models(
        dataset_dir=dataset_output,
        output_dir=forecast_output,
    )
    print(f"   forecast: {forecast_result['output_dir']}")

    print("4/11 Обнаружение аномалий без учителя...")
    anomaly_result = train_unsupervised_anomaly(
        feature_dir=features_output,
        output_dir=anomaly_output,
    )
    print(f"   anomaly: {anomaly_result['output_dir']}")

    print("5/11 Доверительные интервалы обнаружения (повторные seed'ы)...")
    seeds_result = run_seeds(
        feature_dir=features_output,
        output_dir=seeds_output,
    )
    print(f"   seeds: {seeds_output} ({len(seeds_result['per_seed'])} строк)")

    print("6/11 Обобщение на новые типы аномалий (leave-one-scenario-out)...")
    loso_result = run_loso(
        feature_dir=features_output,
        output_dir=loso_output,
    )
    print(f"   loso: {loso_output} ({len(loso_result['per_fold'])} строк)")

    print("7/11 Многошаговый прогноз агрегированной динамики (направление 1.1)...")
    multi_horizon_result = run_multi_horizon_forecast(
        dataset_dir=dataset_output,
        output_dir=multi_horizon_output,
    )
    print(f"   multi_horizon: {multi_horizon_result['output_dir']}")

    print("8/11 Упреждающий сигнал риска по прогнозу (направление 1.2)...")
    predictive_risk_result = run_predictive_risk(
        dataset_dir=dataset_output,
        feature_dir=features_output,
        output_dir=predictive_risk_output,
    )
    print(f"   predictive_risk: {predictive_risk_result['output_dir']}")

    print("9/11 Финальный EDA...")
    eda_result = build_eda_report(
        dataset_dir=dataset_output,
        feature_dir=features_output,
        output_dir=eda_output,
        anomaly_dir=anomaly_output,
        multi_horizon_dir=multi_horizon_output,
        predictive_risk_dir=predictive_risk_output,
        seeds_dir=seeds_output,
        loso_dir=loso_output,
    )
    print(f"   report: {eda_result['report_path']}")

    print("10/11 Обучение контрольных моделей с учителем...")
    model_result = train_baselines(
        feature_dir=features_output,
        output_dir=models_output,
    )
    best_model = model_result["metrics"]["best_model_by_val_f1"]
    print(f"   best_model_by_val_f1: {best_model}")
    print(f"   output: {models_output}")

    print("11/11 Итоговый отчёт по всему pipeline...")
    report_result = build_project_report(
        dataset_dir=dataset_output,
        feature_dir=features_output,
        forecast_dir=forecast_output,
        anomaly_dir=anomaly_output,
        eda_dir=eda_output,
        models_dir=models_output,
        output_dir=report_output,
    )
    print(f"   report: {report_result['report_path']}")

    print("Готово. Полный простой pipeline выполнен.")


if __name__ == "__main__":
    main()
