import json
from pathlib import Path

import pandas as pd

from crowd_anomaly.core.simulation import SimulationResult


def export_run(result: SimulationResult, output_dir: Path | str) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trajectories = trajectories_to_frame(result)
    trajectories_path = output_path / "trajectories.csv"
    trajectories.to_csv(trajectories_path, index=False)

    metrics = compute_run_metrics(result, trajectories)
    metrics_path = output_path / "run_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "trajectories": trajectories_path,
        "metrics": metrics_path,
    }


def trajectories_to_frame(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for state in result.states:
        for agent in state.agents:
            goal = result.scenario.goals[agent.goal_index]
            rows.append(
                {
                    "step": state.step,
                    "time_s": state.time_s,
                    "agent_id": agent.agent_id,
                    "x": agent.position.x,
                    "y": agent.position.y,
                    "vx": agent.velocity.x,
                    "vy": agent.velocity.y,
                    "speed": agent.velocity.norm(),
                    "fx": agent.force.x,
                    "fy": agent.force.y,
                    "force_norm": agent.force.norm(),
                    "goal_x": goal.position.x,
                    "goal_y": goal.position.y,
                }
            )

    return pd.DataFrame(rows)


def compute_run_metrics(
    result: SimulationResult,
    trajectories: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    if trajectories is None:
        trajectories = trajectories_to_frame(result)

    final_state = result.states[-1]
    reached_count = 0
    for agent in final_state.agents:
        goal = result.scenario.goals[agent.goal_index]
        if agent.position.distance_to(goal.position) <= goal.tolerance:
            reached_count += 1

    return {
        "scenario_name": result.scenario.name,
        "label": result.scenario.label,
        "seed": result.scenario.seed,
        "agent_count": result.scenario.agent_count,
        "steps": result.scenario.steps,
        "duration_s": result.scenario.steps * result.scenario.dt,
        "mean_speed": float(trajectories["speed"].mean()),
        "max_speed": float(trajectories["speed"].max()),
        "mean_force": float(trajectories["force_norm"].mean()),
        "max_force": float(trajectories["force_norm"].max()),
        "final_goal_reached_fraction": reached_count / result.scenario.agent_count,
    }


def export_table(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path
