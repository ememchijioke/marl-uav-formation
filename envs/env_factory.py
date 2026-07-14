from typing import Any

from envs.adaptive_formation_env import AdaptiveFormationEnv
from utils.config_loader import load_config


def create_adaptive_env(
    config: dict[str, Any],
    render_mode: str | None = None,
) -> AdaptiveFormationEnv:
    """Create an AdaptiveFormationEnv from a loaded YAML config."""

    environment = config.get("environment", {})
    mission = config.get("mission", {})
    formation = config.get("formation", {})
    control = config.get("control", {})
    communication = config.get("communication", {})

    selected_render_mode = (
        render_mode
        if render_mode is not None
        else environment.get("render_mode", "headless")
    )

    return AdaptiveFormationEnv(
        render_mode=selected_render_mode,
        num_agents=environment.get("num_uavs", 3),
        max_steps=environment.get("max_steps", 1500),
        ctrl_freq=environment.get("ctrl_freq", 48),
        sim_freq=environment.get("sim_freq", 240),
        mission_stage=environment.get("mission_stage", 3),
        target_altitude=mission.get("target_altitude", 1.0),
        start_x=mission.get("start_x", 0.0),
        goal_x=mission.get("goal_x", 7.0),
        triangle_side=formation.get("triangle_side", 1.20),
        formation_spacing=formation.get("spacing", None),
        transition_steps=formation.get("transition_steps", 100),
        reference_max_vx=control.get("reference_max_vx", 0.30),
        reference_max_vy=control.get("reference_max_vy", 0.22),
        reference_max_vz=control.get("reference_max_vz", 0.14),
        correction_scale=control.get("correction_scale", 0.06),
        velocity_lookahead=control.get("velocity_lookahead", 0.30),
        kp_center=control.get("kp_center", 0.38),
        kp_form=control.get("kp_form", 0.70),
        kp_alt=control.get("kp_alt", 0.45),
        use_neighbor_dropout=communication.get("use_neighbor_dropout", True),
        neighbor_dropout_prob=communication.get("neighbor_dropout_prob", 0.02),
    )


def create_adaptive_env_from_yaml(
    config_path: str,
    render_mode: str | None = None,
) -> AdaptiveFormationEnv:
    """Load YAML and create an AdaptiveFormationEnv."""

    config = load_config(config_path)
    return create_adaptive_env(config=config, render_mode=render_mode)
