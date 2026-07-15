import numpy as np

from envs.formations import (
    get_formation_offsets,
    interpolate_formations,
)


class FormationManager:
    """
    Manages triangle-line formation switching.

    The transition is gradual so the UAV target positions
    do not change suddenly.
    """

    def __init__(
        self,
        initial_formation: str = "triangle",
        spacing: float = 0.6,
        transition_steps: int = 100,
    ):
        if transition_steps <= 0:
            raise ValueError("transition_steps must be greater than zero.")

        self.spacing = spacing
        self.transition_steps = transition_steps

        self.current_formation = initial_formation
        self.target_formation = initial_formation

        self.current_offsets = get_formation_offsets(
            initial_formation,
            spacing,
        )

        self.start_offsets = self.current_offsets.copy()
        self.target_offsets = self.current_offsets.copy()

        self.transition_step = 0
        self.transitioning = False

    def reset(self, formation_name: str = "triangle") -> np.ndarray:
        self.current_formation = formation_name
        self.target_formation = formation_name

        self.current_offsets = get_formation_offsets(
            formation_name,
            self.spacing,
        )

        self.start_offsets = self.current_offsets.copy()
        self.target_offsets = self.current_offsets.copy()

        self.transition_step = 0
        self.transitioning = False

        return self.current_offsets.copy()

    def request_formation(self, formation_name: str) -> None:
        """
        Start a transition to another formation.
        """

        if formation_name == self.target_formation:
            return

        new_offsets = get_formation_offsets(
            formation_name,
            self.spacing,
        )

        self.start_offsets = self.current_offsets.copy()
        self.target_offsets = new_offsets

        self.target_formation = formation_name
        self.transition_step = 0
        self.transitioning = True

    def update(self) -> np.ndarray:
        """
        Advance the formation transition by one environment step.
        """

        if not self.transitioning:
            return self.current_offsets.copy()

        self.transition_step += 1

        progress = self.transition_step / self.transition_steps

        self.current_offsets = interpolate_formations(
            self.start_offsets,
            self.target_offsets,
            progress,
        )

        if progress >= 1.0:
            self.current_offsets = self.target_offsets.copy()
            self.current_formation = self.target_formation
            self.transitioning = False

        return self.current_offsets.copy()

    def get_transition_progress(self) -> float:
        if not self.transitioning:
            return 1.0

        return float(
            np.clip(
                self.transition_step / self.transition_steps,
                0.0,
                1.0,
            )
        )