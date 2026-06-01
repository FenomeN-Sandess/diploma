import argparse
from pathlib import Path

from crowd_anomaly.anomaly import train_unsupervised_anomaly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обнаружение аномалий без учителя по признакам запуска."
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
        help="Папка с рассчитанными признаками на уровне запуска.",
    )
    parser.add_argument(
        "--output",
        default="outputs/anomaly/demo",
        help="Папка для результатов модели без учителя.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.35,
        help=(
            "Ожидаемая доля аномальных запусков для дополнительного способа "
            "сравнения."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_unsupervised_anomaly(
        feature_dir=Path(args.features),
        output_dir=Path(args.output),
        contamination=args.contamination,
    )
    print(f"Обнаружение аномалий без учителя выполнено: {result['output_dir']}")
    print(f"Кандидатов на локальные аномалии: {len(result['local_candidates'])}")


if __name__ == "__main__":
    main()
