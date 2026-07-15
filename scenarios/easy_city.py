
from __future__ import annotations

import numpy as np

from envs.obstacle_manager import Obstacle
from scenarios.base_scenario import UrbanScenario


class EasyCityScenario(UrbanScenario):
    """
    Deterministic easy urban benchmark.

    Layout:
        - Open start area
        - One narrow passage formed by two building blocks
        - Open recovery area
        - Goal and landing area

    Expected formation sequence:
        triangle -> line -> triangle
    """

    def __init__(
        self,
        target_altitude: float = 1.0,
        start_x: float = 0.0,
        goal_x: float = 7.0,
        landing_altitude: float = 0.08,
    ) -> None:
        start_center = np.array(
            [start_x, 0.0, target_altitude],
            dtype=np.float32,
        )

        goal_center = np.array(
            [goal_x, 0.0, target_altitude],
            dtype=np.float32,
        )

        landing_center = np.array(
            [goal_x, 0.0, landing_altitude],
            dtype=np.float32,
        )

        obstacles = [
            Obstacle(
                name="easy_building_left",
                kind="building",
                center=np.array([3.5, -1.45, 1.0], dtype=np.float32),
                size=np.array([1.2, 1.5, 2.0], dtype=np.float32),
                safety_margin=0.12,
            ),
            Obstacle(
                name="easy_building_right",
                kind="building",
                center=np.array([3.5, 1.45, 1.0], dtype=np.float32),
                size=np.array([1.2, 1.5, 2.0], dtype=np.float32),
                safety_margin=0.12,
            ),
        ]

        super().__init__(
            name="easy_city",
            difficulty="easy",
            start_center=start_center,
            goal_center=goal_center,
            landing_center=landing_center,
            preferred_formation="triangle",
            obstacles=obstacles,
            switch_to_line_x=2.35,
            restore_triangle_x=4.65,
            metadata={
                "description": (
                    "Two building blocks form one central narrow passage."
                ),
                "expected_switches": 2,
                "randomized": False,
            },
        )


def create_easy_city_scenario(
    target_altitude: float = 1.0,
    start_x: float = 0.0,
    goal_x: float = 7.0,
    landing_altitude: float = 0.08,
) -> EasyCityScenario:
    return EasyCityScenario(
        target_altitude=target_altitude,
        start_x=start_x,
        goal_x=goal_x,
        landing_altitude=landing_altitude,
    )
