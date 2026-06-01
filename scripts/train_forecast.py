import argparse
from pathlib import Path

from crowd_anomaly.forecasting import train_forecast_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Простое прогнозирование динамики толпы по временным окнам."
    )
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/demo",
        help="Папка с синтетическим датасетом.",
    )
    parser.add_argument(
        "--output",
        default="outputs/forecast/demo",
        help="Папка для результатов прогноза.",
    )
    parser.add_argument("--window-size", type=int, default=5, help="Длина окна.")
    parser.add_argument("--horizon", type=int, default=1, help="Горизонт прогноза.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_forecast_models(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        window_size=args.window_size,
        horizon=args.horizon,
    )
    print(f"Прогнозирование выполнено: {result['output_dir']}")


if __name__ == "__main__":
    main()
