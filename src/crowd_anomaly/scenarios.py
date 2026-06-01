from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crowd_anomaly.core.geometry import Vector2D
from crowd_anomaly.core.simulation import Scenario
from crowd_anomaly.core.social_force import Agent, Goal, SocialForceConfig


@dataclass(frozen=True)
class ScenarioTemplate:
    name: str
    label: str
    description: str
    agent_count: int
    world_size: tuple[float, float]
    steps: int
    dt: float
    start_regions: tuple[dict[str, Any], ...]
    goals: tuple[dict[str, Any], ...]
    speed: dict[str, float]


def load_scenarios(path: Path | str) -> list[ScenarioTemplate]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    return [_parse_template(item) for item in raw_config.get("scenarios", [])]


def build_scenario(template: ScenarioTemplate, seed: int) -> Scenario:
    random = np.random.default_rng(seed)
    goals = tuple(
        Goal(
            position=Vector2D(float(item["x"]), float(item["y"])),
            tolerance=float(item.get("tolerance", 0.5)),
        )
        for item in template.goals
    )
    config = SocialForceConfig(
        relaxation_time=float(template.speed.get("relaxation_time", 0.5)),
        social_force_strength=float(
            template.speed.get(
                "social_force_strength",
                template.speed.get("repulsion_strength", 2.1),
            )
        ),
        social_force_range=float(
            template.speed.get(
                "social_force_range",
                template.speed.get("repulsion_range", 0.3),
            )
        ),
        wall_force_strength=float(template.speed.get("wall_force_strength", 10.0)),
        wall_force_range=float(template.speed.get("wall_force_range", 0.2)),
        body_force_strength=float(template.speed.get("body_force_strength", 0.4)),
        friction_strength=float(template.speed.get("friction_strength", 0.1)),
        max_speed=float(template.speed.get("max_speed", 2.0)),
    )
    agents = _build_agents(template, random, config.max_speed)

    return Scenario(
        name=template.name,
        label=template.label,
        description=template.description,
        seed=seed,
        agent_count=template.agent_count,
        world_size=template.world_size,
        steps=template.steps,
        dt=template.dt,
        agents=agents,
        goals=goals,
        config=config,
    )


def _parse_template(item: dict[str, Any]) -> ScenarioTemplate:
    width, height = item["world_size"]
    return ScenarioTemplate(
        name=str(item["name"]),
        label=str(item["label"]),
        description=str(item["description"]),
        agent_count=int(item["agent_count"]),
        world_size=(float(width), float(height)),
        steps=int(item["steps"]),
        dt=float(item["dt"]),
        start_regions=tuple(item["start_regions"]),
        goals=tuple(item["goals"]),
        speed={key: float(value) for key, value in item["speed"].items()},
    )


def _build_agents(
    template: ScenarioTemplate,
    random: np.random.Generator,
    max_speed: float,
) -> tuple[Agent, ...]:
    agents = []
    remaining = template.agent_count

    for region_index, region in enumerate(template.start_regions):
        count = int(region.get("count", remaining))
        if region_index == len(template.start_regions) - 1:
            count = remaining

        for _ in range(count):
            agent_id = len(agents)
            position = Vector2D(
                x=float(random.uniform(*region["x"])),
                y=float(random.uniform(*region["y"])),
            )
            desired_speed = _sample_desired_speed(
                template=template,
                random=random,
                multiplier=float(region.get("desired_speed_multiplier", 1.0)),
                max_speed=max_speed,
            )
            agents.append(
                Agent(
                    agent_id=agent_id,
                    position=position,
                    velocity=Vector2D(0.0, 0.0),
                    desired_speed=desired_speed,
                    radius=float(template.speed.get("radius", 0.3)),
                    goal_index=int(region.get("goal_index", 0)),
                )
            )

        remaining -= count
        if remaining <= 0:
            break

    return tuple(agents)


def _sample_desired_speed(
    template: ScenarioTemplate,
    random: np.random.Generator,
    multiplier: float,
    max_speed: float,
) -> float:
    mean = template.speed.get("desired_speed_mean", 1.2) * multiplier
    std = template.speed.get("desired_speed_std", 0.1)
    speed = float(random.normal(mean, std))
    return min(max(speed, 0.2), max_speed)
