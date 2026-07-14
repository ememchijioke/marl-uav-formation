from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class MissionPhase:
    """Definition of one mission phase."""

    name: str
    target_location: str
    requested_formation: str
    hold_steps: int


class ScenarioManager:
    """
    Owns the obstacle-free adaptive-formation mission definition.

    Current mission:
        takeoff triangle
        start hover triangle
        triangle -> line
        line flight to goal
        line -> triangle
        goal hover triangle
        landing triangle

    Obstacle layouts will be added later through separate scenario classes.
    """

    def __init__(
        self,
        start_x: float = 0.0,
        goal_x: float = 7.0,
        target_altitude: float = 1.0,
        landing_altitude: float = 0.08,
        mission_stage: int = 3,
    ) -> None:
        self.start_center = np.array(
            [float(start_x), 0.0, float(target_altitude)],
            dtype=np.float32,
        )
        self.goal_center = np.array(
            [float(goal_x), 0.0, float(target_altitude)],
            dtype=np.float32,
        )
        self.landing_center = np.array(
            [float(goal_x), 0.0, float(landing_altitude)],
            dtype=np.float32,
        )

        self.target_altitude = float(target_altitude)
        self.goal_x = float(goal_x)
        self.mission_stage = int(np.clip(mission_stage, 1, 3))

        self.phases: List[MissionPhase] = [
            MissionPhase("takeoff_triangle", "start", "triangle", 18),
            MissionPhase("start_hover_triangle", "start", "triangle", 35),
            MissionPhase("triangle_to_line", "start", "line", 12),
            MissionPhase("line_to_goal", "goal", "line", 12),
            MissionPhase("line_to_triangle", "goal", "triangle", 12),
            MissionPhase("goal_hover_triangle", "goal", "triangle", 35),
            MissionPhase("landing_triangle", "landing", "triangle", 8),
        ]

        self.max_phase_for_stage: Dict[int, int] = {
            1: 2,
            2: 5,
            3: 6,
        }

    @property
    def phase_names(self) -> List[str]:
        return [phase.name for phase in self.phases]

    @property
    def max_phase(self) -> int:
        return self.max_phase_for_stage[self.mission_stage]

    def get_phase(self, phase_index: int) -> MissionPhase:
        if not 0 <= phase_index < len(self.phases):
            raise IndexError(f"Invalid phase index: {phase_index}")
        return self.phases[phase_index]

    def get_target_center(self, phase_index: int) -> np.ndarray:
        location = self.get_phase(phase_index).target_location

        if location == "start":
            return self.start_center.copy()
        if location == "goal":
            return self.goal_center.copy()
        if location == "landing":
            return self.landing_center.copy()

        raise ValueError(f"Unknown target location: {location}")

    def get_requested_formation(self, phase_index: int) -> str:
        return self.get_phase(phase_index).requested_formation

    def get_hold_steps(self, phase_index: int) -> int:
        return self.get_phase(phase_index).hold_steps

    def is_phase_complete(
        self,
        phase_index: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        info: dict,
        current_formation: str,
        formation_transitioning: bool,
    ) -> bool:
        """Evaluate completion of the active mission phase."""

        centroid = np.mean(positions, axis=0)
        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))

        center_error = float(info.get("center_error", 999.0))
        formation_error = float(info.get("formation_error", 999.0))
        spacing_error = float(info.get("spacing_error", 999.0))
        collision_count = int(info.get("collision_count", 99))
        crashed = bool(info.get("crashed", False))

        stable_formation = bool(
            formation_error < 0.38
            and spacing_error < 0.30
            and collision_count == 0
            and not crashed
        )

        if phase_index == 0:
            mean_altitude = float(np.mean(positions[:, 2]))
            return bool(
                mean_altitude > 0.90
                and abs(mean_altitude - self.target_altitude) < 0.18
                and stable_formation
            )

        if phase_index == 1:
            return bool(stable_formation and mean_speed < 0.45)

        if phase_index == 2:
            return bool(
                not formation_transitioning
                and current_formation == "line"
                and stable_formation
                and mean_speed < 0.45
            )

        if phase_index == 3:
            return bool(
                centroid[0] >= self.goal_x - 0.35
                and stable_formation
            )

        if phase_index == 4:
            return bool(
                not formation_transitioning
                and current_formation == "triangle"
                and stable_formation
                and mean_speed < 0.45
            )

        if phase_index == 5:
            return bool(
                center_error < 0.45
                and stable_formation
                and mean_speed < 0.45
            )

        if phase_index == 6:
            mean_altitude = float(np.mean(positions[:, 2]))
            max_altitude = float(np.max(positions[:, 2]))
            return bool(
                mean_altitude < 0.16
                and max_altitude < 0.22
                and mean_speed < 0.35
                and collision_count == 0
                and not crashed
            )

        raise IndexError(f"Unsupported phase index: {phase_index}")
