import argparse
from pathlib import Path

from crowd_anomaly.baselines import train_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение простых контрольных моделей на уровне запуска."
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
        help="Папка с рассчитанными признаками на уровне запуска.",
    )
    parser.add_argument(
        "--output",
        default="outputs/models/demo",
        help="Папка для сохранения результатов контрольных моделей.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_baselines(feature_dir=args.features, output_dir=args.output)
    metrics = result["metrics"]
    best_model = metrics["best_model_by_val_f1"]

    print("Контрольные модели обучены.")
    print(f"Папка признаков: {Path(args.features)}")
    print(f"Папка результатов: {Path(args.output)}")
    print("Модели: logistic_regression, random_forest")
    print(f"Лучшая модель по val F1: {best_model}")
    print(f"metrics.json: {Path(args.output) / 'metrics.json'}")


if __name__ == "__main__":
    main()
