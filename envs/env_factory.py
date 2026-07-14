from __future__ import annotations

from typing import Any

from envs.adaptive_formation_env import AdaptiveFormationEnv
from envs.easy_city_runtime import attach_obstacle_runtime
from envs.obstacle_manager import ObstacleManager
from scenarios.scenario_factory import create_scenario
from utils.config_loader import load_config


def create_adaptive_env(
    config: dict[str, Any],
    render_mode: str | None = None,
) -> AdaptiveFormationEnv:

    environment = config.get("environment", {})
    formation = config.get("formation", {})
    control = config.get("control", {})
    communication = config.get("communication", {})
    obstacle_config = config.get("obstacles", {})

    selected_render_mode = (
        render_mode
        if render_mode is not None
        else environment.get("render_mode", "headless")
    )

    scenario = create_scenario(config)

    env = AdaptiveFormationEnv(
        render_mode=selected_render_mode,
        num_agents=environment.get("num_uavs", 3),
        max_steps=environment.get("max_steps", 1500),
        ctrl_freq=environment.get("ctrl_freq", 48),
        sim_freq=environment.get("sim_freq", 240),
        mission_stage=environment.get("mission_stage", 3),

        target_altitude=float(scenario.start_center[2]),
        start_x=float(scenario.start_center[0]),
        goal_x=float(scenario.goal_center[0]),

        triangle_side=formation.get("triangle_side", 1.20),

        reference_max_vx=control.get("reference_max_vx", 0.30),
        reference_max_vy=control.get("reference_max_vy", 0.22),
        reference_max_vz=control.get("reference_max_vz", 0.14),
        correction_scale=control.get("correction_scale", 0.06),
        velocity_lookahead=control.get("velocity_lookahead", 0.30),
        kp_center=control.get("kp_center", 0.38),
        kp_form=control.get("kp_form", 0.70),
        kp_alt=control.get("kp_alt", 0.45),

        use_neighbor_dropout=communication.get(
            "use_neighbor_dropout",
            True,
        ),
        neighbor_dropout_prob=communication.get(
            "neighbor_dropout_prob",
            0.02,
        ),
    )

    # Apply YAML-controlled formation settings after construction.
    formation_spacing = float(formation.get("spacing", 0.60))
    transition_steps = int(formation.get("transition_steps", 100))

    env.formation_spacing = formation_spacing
    env.transition_steps = transition_steps

    env.formation_manager.spacing = formation_spacing
    env.formation_manager.transition_steps = transition_steps
    env.formation_manager.reset("triangle")

    env.scenario = scenario

    env.obstacle_manager = ObstacleManager(
        obstacles=scenario.obstacles,
        drone_radius=obstacle_config.get("drone_radius", 0.08),
    )

    env.start_center = scenario.start_center.copy()
    env.goal_center = scenario.goal_center.copy()
    env.landing_center = scenario.landing_center.copy()

    return attach_obstacle_runtime(env)


def create_adaptive_env_from_yaml(
    config_path: str,
    render_mode: str | None = None,
) -> AdaptiveFormationEnv:

    config = load_config(config_path)

    return create_adaptive_env(
        config=config,
        render_mode=render_mode,
    )
