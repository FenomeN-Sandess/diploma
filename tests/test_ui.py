from pathlib import Path

import matplotlib.pyplot as plt

from crowd_anomaly.core.simulation import run_simulation
from crowd_anomaly.scenarios import build_scenario, load_scenarios
from crowd_anomaly.ui import build_simulation_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ui_module_imports_and_scenarios_load() -> None:
    scenarios = load_scenarios(PROJECT_ROOT / "configs" / "scenarios.yaml")

    assert scenarios
    assert {scenario.label for scenario in scenarios} == {"normal", "anomaly"}


def test_build_simulation_figure_returns_matplotlib_figure() -> None:
    template = load_scenarios(PROJECT_ROOT / "configs" / "scenarios.yaml")[0]
    result = run_simulation(build_scenario(template, seed=42))

    figure = build_simulation_figure(result)

    assert isinstance(figure, plt.Figure)
    plt.close(figure)
