from math import exp

from crowd_anomaly.core.geometry import Vector2D
from crowd_anomaly.core.simulation import Scenario, run_simulation
from crowd_anomaly.core.social_force import (
    Agent,
    Goal,
    SocialForceConfig,
    agent_repulsion,
    limit_speed,
    wall_repulsion,
)
from crowd_anomaly.scenarios import build_scenario, load_scenarios


def test_vector2d_operations() -> None:
    first = Vector2D(3.0, 4.0)
    second = Vector2D(1.0, 2.0)

    assert (first + second).to_tuple() == (4.0, 6.0)
    assert (first - second).to_tuple() == (2.0, 2.0)
    assert (second * 3).to_tuple() == (3.0, 6.0)
    assert first.norm() == 5.0
    assert first.normalized().to_tuple() == (0.6, 0.8)
    assert first.distance_to(second) == (first - second).norm()
    assert first.dot(second) == 11.0
    assert first.perpendicular().to_tuple() == (-4.0, 3.0)


def test_agent_repulsion_uses_helbing_exponential_term() -> None:
    first = Agent(
        agent_id=1,
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        desired_speed=1.0,
        radius=0.3,
    )
    second = Agent(
        agent_id=2,
        position=Vector2D(1.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        desired_speed=1.0,
        radius=0.3,
    )
    config = SocialForceConfig(
        social_force_strength=2.1,
        social_force_range=0.3,
        body_force_strength=0.0,
        friction_strength=0.0,
    )

    force = agent_repulsion(first, second, config)
    expected = -2.1 * exp((0.6 - 1.0) / 0.3)

    assert abs(force.x - expected) < 1e-9
    assert abs(force.y) < 1e-9


def test_wall_repulsion_pushes_agent_inside_world() -> None:
    agent = Agent(
        agent_id=1,
        position=Vector2D(0.3, 2.0),
        velocity=Vector2D(0.0, 0.0),
        desired_speed=1.0,
        radius=0.3,
    )
    config = SocialForceConfig(
        wall_force_strength=10.0,
        wall_force_range=0.2,
        body_force_strength=0.0,
    )

    force = wall_repulsion(agent, world_size=(10.0, 5.0), config=config)

    assert force.x > 9.0
    assert abs(force.y) < 0.1


def test_agent_moves_towards_goal() -> None:
    scenario = Scenario(
        name="single_agent",
        label="normal",
        description="Проверочный сценарий с одним агентом.",
        seed=1,
        agent_count=1,
        world_size=(10.0, 5.0),
        steps=3,
        dt=0.1,
        agents=(
            Agent(
                agent_id=0,
                position=Vector2D(1.0, 2.5),
                velocity=Vector2D(0.0, 0.0),
                desired_speed=1.0,
            ),
        ),
        goals=(Goal(position=Vector2D(8.0, 2.5), tolerance=0.3),),
        config=SocialForceConfig(max_speed=1.5),
    )

    result = run_simulation(scenario)
    start_x = result.states[0].agents[0].position.x
    final_x = result.states[-1].agents[0].position.x

    assert final_x > start_x
    assert result.states[-1].agents[0].force.norm() > 0


def test_speed_is_limited() -> None:
    velocity = Vector2D(10.0, 0.0)
    limited = limit_speed(velocity, max_speed=2.0)

    assert limited.norm() == 2.0


def test_simulation_returns_expected_number_of_steps() -> None:
    template = load_scenarios("configs/scenarios.yaml")[0]
    scenario = build_scenario(template, seed=42)
    result = run_simulation(scenario)

    assert len(result.states) == scenario.steps + 1
    assert result.states[0].step == 0
    assert result.states[-1].step == scenario.steps


def test_simulation_is_reproducible_with_same_seed() -> None:
    template = load_scenarios("configs/scenarios.yaml")[0]
    first_result = run_simulation(build_scenario(template, seed=7))
    second_result = run_simulation(build_scenario(template, seed=7))

    first_final = first_result.states[-1].agents[0]
    second_final = second_result.states[-1].agents[0]

    assert first_final.position.to_tuple() == second_final.position.to_tuple()
    assert first_final.velocity.to_tuple() == second_final.velocity.to_tuple()
