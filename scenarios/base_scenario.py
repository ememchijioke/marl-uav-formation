
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from envs.obstacle_manager import Obstacle


@dataclass
class UrbanScenario:
    """
    Reusable description of a known-obstacle UAV mission.
    """

    name: str
    difficulty: str
    start_center: np.ndarray
    goal_center: np.ndarray
    landing_center: np.ndarray
    preferred_formation: str
    obstacles: list[Obstacle] = field(default_factory=list)
    switch_to_line_x: float = 0.0
    restore_triangle_x: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start_center = np.asarray(self.start_center, dtype=np.float32)
        self.goal_center = np.asarray(self.goal_center, dtype=np.float32)
        self.landing_center = np.asarray(self.landing_center, dtype=np.float32)

        for name, value in (
            ("start_center", self.start_center),
            ("goal_center", self.goal_center),
            ("landing_center", self.landing_center),
        ):
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,).")

        if self.restore_triangle_x <= self.switch_to_line_x:
            raise ValueError(
                "restore_triangle_x must be greater than switch_to_line_x."
            )

    def requested_formation(self, centroid_x: float) -> str:
        """
        Return the desired formation for the current mission position.
        """

        if centroid_x < self.switch_to_line_x:
            return self.preferred_formation

        if centroid_x < self.restore_triangle_x:
            return "line"

        return self.preferred_formation

    def obstacle_count(self) -> int:
        return len(self.obstacles)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "start_center": self.start_center.copy(),
            "goal_center": self.goal_center.copy(),
            "landing_center": self.landing_center.copy(),
            "preferred_formation": self.preferred_formation,
            "switch_to_line_x": float(self.switch_to_line_x),
            "restore_triangle_x": float(self.restore_triangle_x),
            "obstacle_count": self.obstacle_count(),
            "metadata": dict(self.metadata),
        }
