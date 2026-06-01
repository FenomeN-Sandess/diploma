from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from crowd_anomaly.core.export import compute_run_metrics, export_run
from crowd_anomaly.core.simulation import run_simulation
from crowd_anomaly.scenarios import ScenarioTemplate, build_scenario, load_scenarios

LABEL_TO_ID = {
    "normal": 0,
    "anomaly": 1,
}
FRAME_LABEL_SOURCE = "run_level_inherited"


def build_dataset(
    config_path: Path | str,
    scenarios_path: Path | str,
    output_dir: Path | str,
    seed: int | None = None,
    runs_per_template: int | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    templates = load_scenarios(scenarios_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs_path = output_path / "runs"
    runs_path.mkdir(parents=True, exist_ok=True)

    dataset_name = str(config.get("dataset_name", output_path.name))
    base_seed = int(seed if seed is not None else config.get("seed", 42))
    repeat_count = int(
        runs_per_template
        if runs_per_template is not None
        else config.get("runs_per_template", 1)
    )
    splits = config.get("splits", {"train": 0.6, "val": 0.2, "test": 0.2})

    run_plan = _build_run_plan(templates, base_seed, repeat_count)
    run_plan = assign_splits(run_plan, splits=splits, seed=base_seed)

    run_rows = []
    frame_rows = []

    for run_info in run_plan.to_dict("records"):
        template = templates[int(run_info["template_index"])]
        scenario = build_scenario(template, seed=int(run_info["seed"]))
        result = run_simulation(scenario)
        run_output_dir = runs_path / str(run_info["run_id"])
        export_run(result, run_output_dir)

        metrics = compute_run_metrics(result)
        run_rows.append(
            {
                "run_id": run_info["run_id"],
                "scenario_template": run_info["scenario_template"],
                "split": run_info["split"],
                "label": LABEL_TO_ID[str(run_info["label_name"])],
                "label_name": run_info["label_name"],
                "seed": run_info["seed"],
                "agent_count": metrics["agent_count"],
                "steps": metrics["steps"],
                "duration_s": metrics["duration_s"],
                "mean_speed": metrics["mean_speed"],
                "max_speed": metrics["max_speed"],
                "mean_force": metrics["mean_force"],
                "max_force": metrics["max_force"],
                "final_goal_reached_fraction": metrics[
                    "final_goal_reached_fraction"
                ],
            }
        )
        frame_rows.extend(_build_frame_rows(result, run_info))

    runs = pd.DataFrame(run_rows)
    frames = pd.DataFrame(frame_rows)
    runs.to_csv(output_path / "runs.csv", index=False)
    frames.to_csv(output_path / "frames.csv", index=False)

    metadata = write_dataset_metadata(
        output_dir=output_path,
        dataset_name=dataset_name,
        templates=templates,
        runs=runs,
        splits=splits,
        seed=base_seed,
        runs_per_template=repeat_count,
        split_policy=str(run_plan.attrs.get("split_policy", "unknown")),
    )

    return {
        "output_dir": output_path,
        "metadata": metadata,
        "runs": runs,
        "frames": frames,
    }


def assign_splits(
    runs: pd.DataFrame,
    splits: dict[str, float],
    seed: int,
    split_by: str = "label_name",
) -> pd.DataFrame:
    result = runs.copy()
    split_names = list(splits.keys())
    split_values = list(splits.values())
    random = np.random.default_rng(seed)
    group_column = split_by if split_by in result.columns else "run_id"
    result["split"] = ""

    for _, group_frame in result.groupby(group_column, sort=True):
        indices = group_frame.index.to_list()
        random.shuffle(indices)
        counts = _split_counts(len(indices), split_values)

        cursor = 0
        for split_name, count in zip(split_names, counts, strict=True):
            selected = indices[cursor : cursor + count]
            result.loc[selected, "split"] = split_name
            cursor += count

    result.attrs["split_policy"] = f"{group_column}_stratified_run_seeded"
    return result


def write_dataset_metadata(
    output_dir: Path | str,
    dataset_name: str,
    templates: list[ScenarioTemplate],
    runs: pd.DataFrame,
    splits: dict[str, float],
    seed: int,
    runs_per_template: int,
    split_policy: str,
) -> dict[str, Any]:
    metadata = {
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_data": True,
        "data_source": "simulation",
        "model": "social_force_model",
        "target_level": "run",
        "target_description": "нормальный или аномальный запуск",
        "frame_label_policy": FRAME_LABEL_SOURCE,
        "local_anomaly_ground_truth": "not_available",
        "real_world_validation": "not_performed",
        "seed": seed,
        "runs_per_template": runs_per_template,
        "run_count": int(len(runs)),
        "scenario_templates": [template.name for template in templates],
        "splits": splits,
        "split_policy": split_policy,
        "split_note": (
            "Split выполняется воспроизводимо. При split_policy="
            "label_name_stratified_run_seeded разбиение выполняется отдельно "
            "для normal и anomaly, чтобы train/val/test содержали оба класса "
            "при достаточном числе запусков."
        ),
        "limitations": [
            "Данные являются синтетическими и получены внутри симулятора.",
            "Целевая метка задается на уровне всего запуска симуляции.",
            "Покадровые метки наследуются от метки всего запуска.",
            "Локальная разметка аномальных кадров и агентов отсутствует.",
            "Валидация на реальных данных не выполнялась.",
        ],
    }
    metadata_path = Path(output_dir) / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _build_run_plan(
    templates: list[ScenarioTemplate],
    seed: int,
    runs_per_template: int,
) -> pd.DataFrame:
    rows = []
    for template_index, template in enumerate(templates):
        for run_index in range(runs_per_template):
            run_seed = seed + template_index * 10_000 + run_index
            rows.append(
                {
                    "run_id": f"{template.name}__run_{run_index:03d}",
                    "template_index": template_index,
                    "scenario_template": template.name,
                    "label_name": template.label,
                    "seed": run_seed,
                }
            )
    return pd.DataFrame(rows)


def _build_frame_rows(result: Any, run_info: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_info["run_id"],
            "scenario_template": run_info["scenario_template"],
            "split": run_info["split"],
            "label": LABEL_TO_ID[str(run_info["label_name"])],
            "label_name": run_info["label_name"],
            "step": state.step,
            "time_s": state.time_s,
            "frame_label_source": FRAME_LABEL_SOURCE,
        }
        for state in result.states
    ]


def _split_counts(total: int, split_values: list[float]) -> list[int]:
    weights = np.array(split_values, dtype=float)
    weights = weights / weights.sum()
    active_indices = [index for index, value in enumerate(split_values) if value > 0]

    counts = np.zeros(len(split_values), dtype=int)
    if total >= len(active_indices):
        for index in active_indices:
            counts[index] = 1

    remaining = total - int(counts.sum())
    raw_counts = weights * remaining
    extra_counts = np.floor(raw_counts).astype(int)
    counts += extra_counts
    remaining = total - int(counts.sum())

    order = np.argsort(-(raw_counts - extra_counts))
    for index in order[:remaining]:
        counts[index] += 1

    return counts.tolist()


def _load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def empty_dataset(output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame(columns=["run_id", "scenario", "label"])
