from dataclasses import dataclass
from math import exp

from crowd_anomaly.core.geometry import Vector2D


@dataclass(frozen=True)
class Agent:
    agent_id: int
    position: Vector2D
    velocity: Vector2D
    desired_speed: float
    radius: float = 0.3
    goal_index: int = 0
    force: Vector2D = Vector2D(0.0, 0.0)


@dataclass(frozen=True)
class Goal:
    position: Vector2D
    tolerance: float = 0.5


@dataclass(frozen=True)
class SocialForceConfig:
    relaxation_time: float = 0.5
    social_force_strength: float = 2.1
    social_force_range: float = 0.3
    wall_force_strength: float = 10.0
    wall_force_range: float = 0.2
    body_force_strength: float = 0.4
    friction_strength: float = 0.1
    max_speed: float = 2.0


def goal_force(agent: Agent, goal: Goal, config: SocialForceConfig) -> Vector2D:
    direction = (goal.position - agent.position).normalized()
    desired_velocity = direction * agent.desired_speed
    relaxation_time = max(config.relaxation_time, 1e-6)
    return (desired_velocity - agent.velocity) * (1.0 / relaxation_time)


def agent_repulsion(
    agent: Agent,
    other: Agent,
    config: SocialForceConfig,
) -> Vector2D:
    if agent.agent_id == other.agent_id:
        return Vector2D(0.0, 0.0)

    offset = agent.position - other.position
    distance = max(offset.norm(), 1e-9)
    normal = offset.normalized() if offset.norm() > 0 else Vector2D(1.0, 0.0)
    tangent = normal.perpendicular()

    contact_distance = agent.radius + other.radius
    overlap = max(contact_distance - distance, 0.0)
    social_range = max(config.social_force_range, 1e-6)

    social_force = (
        config.social_force_strength
        * exp((contact_distance - distance) / social_range)
    )
    body_force = config.body_force_strength * overlap
    relative_tangent_speed = (other.velocity - agent.velocity).dot(tangent)
    friction_force = config.friction_strength * overlap * relative_tangent_speed

    return normal * (social_force + body_force) + tangent * friction_force


def wall_repulsion(
    agent: Agent,
    world_size: tuple[float, float],
    config: SocialForceConfig,
) -> Vector2D:
    width, height = world_size
    walls = [
        (agent.position.x, Vector2D(1.0, 0.0)),
        (width - agent.position.x, Vector2D(-1.0, 0.0)),
        (agent.position.y, Vector2D(0.0, 1.0)),
        (height - agent.position.y, Vector2D(0.0, -1.0)),
    ]

    total = Vector2D(0.0, 0.0)
    wall_range = max(config.wall_force_range, 1e-6)
    for distance, normal in walls:
        gap = agent.radius - max(distance, 0.0)
        social_force = config.wall_force_strength * exp(gap / wall_range)
        body_force = config.body_force_strength * max(gap, 0.0)
        total += normal * (social_force + body_force)
    return total


def limit_speed(velocity: Vector2D, max_speed: float) -> Vector2D:
    speed = velocity.norm()
    if speed <= max_speed or speed == 0:
        return velocity
    return velocity.normalized() * max_speed
