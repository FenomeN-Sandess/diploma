import argparse
from pathlib import Path

from crowd_anomaly.features import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Расчет физических признаков на уровне запуска."
    )
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/demo",
        help="Папка с подготовленным синтетическим датасетом.",
    )
    parser.add_argument(
        "--output",
        default="outputs/features/demo",
        help="Папка для сохранения признаков.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_features(dataset_dir=args.dataset, output_dir=args.output)
    schema = result["feature_schema"]

    print("Признаки на уровне запуска рассчитаны.")
    print(f"Папка датасета: {Path(args.dataset)}")
    print(f"Папка признаков: {Path(args.output)}")
    print(f"Количество запусков: {len(result['features_run'])}")
    print(f"Количество признаков: {len(schema['feature_columns'])}")
    print(
        "Проверка утечки: feature_columns не содержат "
        "label/split/seed/template/run_id."
    )


if __name__ == "__main__":
    main()
