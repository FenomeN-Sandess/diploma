"""Анализ чувствительности: проверка устойчивости F1 при изменении ключевых
параметров генерации данных.

Варьируемые параметры:
  * social_force_strength (A)  — амплитуда социального отталкивания, ±20 %
  * agent_count (N)            — число агентов в каждом сценарии, ±20 %
  * desired_speed_mean (v0)    — средняя желаемая скорость, ±20 %

Для каждого варианта полностью перезапускается мини-конвейер:
  build_dataset → build_features → train_unsupervised_anomaly.

Число запусков на шаблон снижено до 15, чтобы сократить время расчёта.
Результаты сохраняются в outputs/sensitivity/.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from crowd_anomaly.anomaly import train_unsupervised_anomaly
from crowd_anomaly.dataset import build_dataset
from crowd_anomaly.features import build_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mutate_scenarios_yaml(
    src_path: Path,
    dst_path: Path,
    *,
    force_strength_factor: float = 1.0,
    agent_count_factor: float = 1.0,
    speed_factor: float = 1.0,
) -> None:
    """Создаёт модифицированный YAML со сценариями."""
    with src_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for sc in data.get("scenarios", []):
        if agent_count_factor != 1.0:
            sc["agent_count"] = max(5, round(sc["agent_count"] * agent_count_factor))
            for region in sc.get("start_regions", []):
                region["count"] = max(1, round(region["count"] * agent_count_factor))

        if force_strength_factor != 1.0:
            speed = sc.get("speed", {})
            if "social_force_strength" in speed:
                speed["social_force_strength"] = round(
                    speed["social_force_strength"] * force_strength_factor, 4
                )

        if speed_factor != 1.0:
            speed = sc.get("speed", {})
            if "desired_speed_mean" in speed:
                speed["desired_speed_mean"] = round(
                    speed["desired_speed_mean"] * speed_factor, 4
                )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def _run_pipeline(
    scenarios_yaml: Path,
    dataset_yaml: Path,
    work_dir: Path,
    runs_per_template: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Запускает мини-конвейер и возвращает метрики тестовой выборки."""
    ds_out = work_dir / "dataset"
    feat_out = work_dir / "features"
    anom_out = work_dir / "anomaly"

    build_dataset(
        config_path=dataset_yaml,
        scenarios_path=scenarios_yaml,
        output_dir=ds_out,
        seed=seed,
        runs_per_template=runs_per_template,
    )
    build_features(dataset_dir=ds_out, output_dir=feat_out)
    result = train_unsupervised_anomaly(
        feature_dir=feat_out,
        output_dir=anom_out,
    )

    comp = result["method_comparison"]
    row = comp[(comp["method"] == "pca_physics") & (comp["split"] == "test")]
    if row.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    row = row.iloc[0]
    return {
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "accuracy": float(row["accuracy"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class VariationSpec:
    label: str
    param_name: str
    factor: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ чувствительности: A и N ±20 %%."
    )
    parser.add_argument(
        "--scenarios", default="configs/scenarios.yaml",
        help="Путь к базовому YAML сценариев."
    )
    parser.add_argument(
        "--dataset-config", default="configs/dataset.yaml",
        help="Путь к YAML конфигурации датасета."
    )
    parser.add_argument(
        "--output", default="outputs/sensitivity",
        help="Папка для результатов."
    )
    parser.add_argument(
        "--runs-per-template", type=int, default=15,
        help="Запусков на шаблон (умеренное значение для скорости)."
    )
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    dataset_yaml = Path(args.dataset_config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variations: list[VariationSpec] = [
        VariationSpec("A x 0.8", "force", 0.8),
        VariationSpec("A x 1.0", "force", 1.0),
        VariationSpec("A x 1.2", "force", 1.2),
        VariationSpec("N x 0.8", "count", 0.8),
        VariationSpec("N x 1.0", "count", 1.0),
        VariationSpec("N x 1.2", "count", 1.2),
        VariationSpec("v0 x 0.8", "speed", 0.8),
        VariationSpec("v0 x 1.0", "speed", 1.0),
        VariationSpec("v0 x 1.2", "speed", 1.2),
    ]

    results: list[dict[str, Any]] = []

    for spec in variations:
        print(f"\n=== Variation: {spec.label} ===")
        work_dir = output_dir / f"_work_{spec.label.replace(' ', '_')}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        sc_yaml = work_dir / "scenarios.yaml"
        force_factor = spec.factor if spec.param_name == "force" else 1.0
        count_factor = spec.factor if spec.param_name == "count" else 1.0
        spd_factor = spec.factor if spec.param_name == "speed" else 1.0

        _mutate_scenarios_yaml(
            scenarios_path, sc_yaml,
            force_strength_factor=force_factor,
            agent_count_factor=count_factor,
            speed_factor=spd_factor,
        )

        metrics = _run_pipeline(
            scenarios_yaml=sc_yaml,
            dataset_yaml=dataset_yaml,
            work_dir=work_dir,
            runs_per_template=args.runs_per_template,
        )

        row = {
            "variation": spec.label,
            "parameter": spec.param_name,
            "factor": spec.factor,
            **metrics,
        }
        results.append(row)
        print(
            f"   F1={metrics['f1']:.3f}  "
            f"Prec={metrics['precision']:.3f}  "
            f"Rec={metrics['recall']:.3f}"
        )

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "sensitivity_results.csv", index=False)
    print(f"\nРезультаты сохранены: {output_dir / 'sensitivity_results.csv'}")

    # ---- Построение графика ----
    _plot_sensitivity(df, output_dir / "sensitivity_f1.png")
    print(f"График: {output_dir / 'sensitivity_f1.png'}")

    # ---- Удаление рабочих директорий ----
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("_work_"):
            shutil.rmtree(d, ignore_errors=True)


def _plot_sensitivity(df: pd.DataFrame, output_path: Path) -> None:
    """Групповая столбчатая диаграмма F1 по вариациям."""
    param_list = ["force", "count", "speed"]
    titles = {
        "force": "$A$ (social force strength)",
        "count": "$N$ (agent count)",
        "speed": "$v_0$ (desired speed)",
    }
    present = [p for p in param_list if p in df["parameter"].values]
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, param in enumerate(present):
        ax = axes[idx]
        subset = df[df["parameter"] == param].sort_values("factor")
        bars = ax.bar(
            subset["variation"],
            subset["f1"],
            color=["#2f5f98", "#6f8daa", "#a7b6c8"],
            edgecolor="#1f2f43",
            linewidth=0.5,
        )
        for bar, val in zip(bars, subset["f1"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(titles[param])
        ax.set_ylabel("F1 (test)")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Sensitivity analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
