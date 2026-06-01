import argparse
from pathlib import Path

from crowd_anomaly.multi_horizon import DEFAULT_HORIZONS, DEFAULT_WINDOW
from crowd_anomaly.predictive_risk import run_predictive_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Упреждающий сигнал риска по прогнозу (направление 1.2). "
            "Считает Ŝ(t, H) и lead time на синтетическом датасете."
        )
    )
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/demo",
        help="Папка с синтетическим датасетом.",
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
        help="Папка с признаками запуска (для PCA и физического штрафа).",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictive_risk/demo",
        help="Папка для упреждающих скоров, метрик и графиков.",
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
    result = run_predictive_risk(
        dataset_dir=Path(args.dataset),
        feature_dir=Path(args.features),
        output_dir=Path(args.output),
        horizons=tuple(args.horizons),
        window_size=args.window_size,
    )
    print(f"Упреждающий сигнал риска посчитан: {result['output_dir']}")


if __name__ == "__main__":
    main()
