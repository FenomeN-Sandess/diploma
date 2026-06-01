import argparse
from pathlib import Path

from crowd_anomaly.dataset import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка синтетического датасета."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Путь к YAML-конфигурации датасета.",
    )
    parser.add_argument(
        "--scenarios",
        default="configs/scenarios.yaml",
        help="Путь к YAML-конфигурации сценариев.",
    )
    parser.add_argument(
        "--output",
        default="outputs/datasets/demo",
        help="Папка для сохранения датасета.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Переопределить seed из конфигурации.",
    )
    parser.add_argument(
        "--runs-per-template",
        type=int,
        default=None,
        help="Переопределить число запусков на один шаблон.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(
        config_path=args.config,
        scenarios_path=args.scenarios,
        output_dir=args.output,
        seed=args.seed,
        runs_per_template=args.runs_per_template,
    )
    runs = result["runs"]

    print("Синтетический датасет создан.")
    print(f"Папка: {Path(args.output)}")
    print(f"Количество запусков: {len(runs)}")
    print("Распределение по split:")
    print(runs["split"].value_counts().sort_index().to_string())
    print("Распределение по меткам:")
    print(runs["label_name"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
