import argparse
from pathlib import Path

from crowd_anomaly.reporting import build_project_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собрать итоговый Markdown-отчёт по всему pipeline."
    )
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/demo",
        help="Папка с синтетическим датасетом.",
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
        help="Папка с рассчитанными признаками.",
    )
    parser.add_argument(
        "--forecast",
        default="outputs/forecast/demo",
        help="Папка с результатами прогноза.",
    )
    parser.add_argument(
        "--anomaly",
        default="outputs/anomaly/demo",
        help="Папка с результатами поиска аномалий.",
    )
    parser.add_argument(
        "--eda",
        default="outputs/eda/demo",
        help="Папка с EDA-отчётом.",
    )
    parser.add_argument(
        "--models",
        default="outputs/models/demo",
        help="Папка с контрольными моделями.",
    )
    parser.add_argument(
        "--output",
        default="outputs/final_report/demo",
        help="Папка для итогового отчёта.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_project_report(
        dataset_dir=Path(args.dataset),
        feature_dir=Path(args.features),
        forecast_dir=Path(args.forecast),
        anomaly_dir=Path(args.anomaly),
        eda_dir=Path(args.eda),
        models_dir=Path(args.models),
        output_dir=Path(args.output),
    )

    print("Итоговый отчёт создан.")
    print(f"Markdown-отчёт: {result['report_path']}")


if __name__ == "__main__":
    main()
