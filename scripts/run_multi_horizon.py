import argparse
from pathlib import Path

from crowd_anomaly.multi_horizon import (
    DEFAULT_HORIZONS,
    DEFAULT_WINDOW,
    run_multi_horizon_forecast,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Многошаговый прогноз агрегированной динамики (направление 1.1). "
            "Сравнивает persistence и direct стратегии по MAE/MSE на горизонте H."
        )
    )
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/demo",
        help="Папка с синтетическим датасетом.",
    )
    parser.add_argument(
        "--output",
        default="outputs/multi_horizon/demo",
        help="Папка для метрик и графиков прогноза по горизонту H.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW,
        help="Длина окна наблюдений k.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Список горизонтов H для оценки.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_multi_horizon_forecast(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        horizons=tuple(args.horizons),
        window_size=args.window_size,
    )
    print(f"Многошаговый прогноз выполнен: {result['output_dir']}")


if __name__ == "__main__":
    main()
