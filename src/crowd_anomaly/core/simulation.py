from __future__ import annotations

from dataclasses import dataclass

from crowd_anomaly.core.geometry import Vector2D
from crowd_anomaly.core.social_force import (
    Agent,
    Goal,
    SocialForceConfig,
    agent_repulsion,
    goal_force,
    limit_speed,
    wall_repulsion,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    description: str
    seed: int
    agent_count: int
    world_size: tuple[float, float]
    steps: int
    dt: float
    agents: tuple[Agent, ...]
    goals: tuple[Goal, ...]
    config: SocialForceConfig


@dataclass(frozen=True)
class SimulationState:
    step: int
    time_s: float
    agents: tuple[Agent, ...]


@dataclass(frozen=True)
class SimulationResult:
    scenario: Scenario
    states: tuple[SimulationState, ...]


def run_simulation(scenario: Scenario) -> SimulationResult:
    states = [
        SimulationState(step=0, time_s=0.0, agents=scenario.agents),
    ]
    agents = scenario.agents

    for step in range(1, scenario.steps + 1):
        agents = _next_agents(agents, scenario)
        states.append(
            SimulationState(
                step=step,
                time_s=round(step * scenario.dt, 10),
                agents=agents,
            )
        )

    return SimulationResult(scenario=scenario, states=tuple(states))


def _next_agents(agents: tuple[Agent, ...], scenario: Scenario) -> tuple[Agent, ...]:
    updated_agents = []
    for agent in agents:
        goal = scenario.goals[agent.goal_index]
        if agent.position.distance_to(goal.position) <= goal.tolerance:
            updated_agents.append(
                _copy_agent(
                    agent,
                    velocity=Vector2D(0.0, 0.0),
                    force=Vector2D(0.0, 0.0),
                )
            )
            continue

        total_force = goal_force(agent, goal, scenario.config)
        for other in agents:
            total_force += agent_repulsion(agent, other, scenario.config)
        total_force += wall_repulsion(agent, scenario.world_size, scenario.config)

        velocity = limit_speed(
            agent.velocity + total_force * scenario.dt,
            scenario.config.max_speed,
        )
        position = _clamp_position(
            agent.position + velocity * scenario.dt,
            scenario.world_size,
        )
        updated_agents.append(
            _copy_agent(
                agent,
                position=position,
                velocity=velocity,
                force=total_force,
            )
        )

    return tuple(updated_agents)


def _copy_agent(
    agent: Agent,
    position: Vector2D | None = None,
    velocity: Vector2D | None = None,
    force: Vector2D | None = None,
) -> Agent:
    return Agent(
        agent_id=agent.agent_id,
        position=position if position is not None else agent.position,
        velocity=velocity if velocity is not None else agent.velocity,
        desired_speed=agent.desired_speed,
        radius=agent.radius,
        goal_index=agent.goal_index,
        force=force if force is not None else agent.force,
    )


def _clamp_position(
    position: Vector2D,
    world_size: tuple[float, float],
) -> Vector2D:
    width, height = world_size
    return Vector2D(
        x=min(max(position.x, 0.0), width),
        y=min(max(position.y, 0.0), height),
    )
