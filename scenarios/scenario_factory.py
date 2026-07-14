
from __future__ import annotations

from typing import Any

from scenarios.base_scenario import UrbanScenario
from scenarios.easy_city import create_easy_city_scenario


def create_scenario(config: dict[str, Any]) -> UrbanScenario:
    """
    Create a scenario object from the YAML configuration.

    Supported initially:
        easy_city
    """

    environment = config.get("environment", {})
    mission = config.get("mission", {})

    scenario_name = environment.get("scenario", "easy_city")

    common_kwargs = {
        "target_altitude": mission.get("target_altitude", 1.0),
        "start_x": mission.get("start_x", 0.0),
        "goal_x": mission.get("goal_x", 7.0),
        "landing_altitude": mission.get("landing_altitude", 0.08),
    }

    if scenario_name == "easy_city":
        return create_easy_city_scenario(**common_kwargs)

    raise ValueError(
        f"Unknown scenario: {scenario_name}. "
        "Currently supported: easy_city."
    )
