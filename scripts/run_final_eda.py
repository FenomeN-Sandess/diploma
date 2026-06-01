import argparse
from pathlib import Path

from crowd_anomaly.eda import build_eda_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Финальный разведочный анализ синтетических запусков."
    )
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/demo",
        help="Папка с подготовленным датасетом.",
    )
    parser.add_argument(
        "--features",
        default="outputs/features/demo",
        help="Папка с рассчитанными признаками.",
    )
    parser.add_argument(
        "--output",
        default="outputs/eda/demo",
        help="Папка для сохранения графиков, таблиц и отчёта.",
    )
    parser.add_argument(
        "--anomaly",
        default="outputs/anomaly/demo",
        help="Папка с результатами модели без учителя для графика anomaly score.",
    )
    parser.add_argument(
        "--multi-horizon",
        default="outputs/multi_horizon/demo",
        help="Папка с результатами многошагового прогноза (направление 1.1).",
    )
    parser.add_argument(
        "--predictive-risk",
        default="outputs/predictive_risk/demo",
        help="Папка с результатами упреждающего сигнала риска (направление 1.2).",
    )
    parser.add_argument(
        "--seeds",
        default="outputs/anomaly/demo/seeds",
        help="Папка с доверительными интервалами обнаружения (повторные seed'ы).",
    )
    parser.add_argument(
        "--loso",
        default="outputs/anomaly/demo/loso",
        help="Папка с результатами leave-one-scenario-out оценки обобщения.",
    )
    return parser.parse_args()


def _existing_dir(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def main() -> None:
    args = parse_args()
    multi_horizon_dir = _existing_dir(args.multi_horizon)
    predictive_risk_dir = _existing_dir(args.predictive_risk)
    seeds_dir = _existing_dir(args.seeds)
    loso_dir = _existing_dir(args.loso)
    result = build_eda_report(
        dataset_dir=args.dataset,
        feature_dir=args.features,
        output_dir=args.output,
        anomaly_dir=args.anomaly,
        multi_horizon_dir=multi_horizon_dir,
        predictive_risk_dir=predictive_risk_dir,
        seeds_dir=seeds_dir,
        loso_dir=loso_dir,
    )

    print("Финальный EDA-отчёт создан.")
    print(f"Папка датасета: {Path(args.dataset)}")
    print(f"Папка признаков: {Path(args.features)}")
    print(f"Папка anomaly: {Path(args.anomaly)}")
    print(f"Папка multi_horizon: {multi_horizon_dir}")
    print(f"Папка predictive_risk: {predictive_risk_dir}")
    print(f"Папка seeds: {seeds_dir}")
    print(f"Папка loso: {loso_dir}")
    print(f"Папка EDA: {Path(args.output)}")
    print(f"Markdown-отчёт: {result['report_path']}")
    print(f"Таблиц: {len(result['tables'])}")
    print(f"Графиков: {len(result['figures'])}")


if __name__ == "__main__":
    main()
