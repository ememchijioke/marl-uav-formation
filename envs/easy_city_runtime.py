
from __future__ import annotations

from types import MethodType

import numpy as np
import pybullet as p


def create_obstacle_visuals(env) -> list[int]:
    """
    Create visual-only Easy City obstacle bodies.

    Visual-only bodies do not affect PyBullet physics yet. Obstacle
    collisions are evaluated geometrically by ObstacleManager.
    """

    if not hasattr(env, "scenario"):
        raise AttributeError("Environment has no attached scenario.")

    client_id = env.env.CLIENT
    body_ids: list[int] = []

    colors = {
        "building": [0.48, 0.50, 0.55, 1.0],
        "tree": [0.20, 0.55, 0.25, 1.0],
        "column": [0.80, 0.45, 0.20, 1.0],
    }

    for obstacle in env.scenario.obstacles:
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=(obstacle.size / 2.0).tolist(),
            rgbaColor=colors.get(
                obstacle.kind,
                [0.60, 0.60, 0.60, 1.0],
            ),
            physicsClientId=client_id,
        )

        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=obstacle.center.tolist(),
            physicsClientId=client_id,
        )

        body_ids.append(int(body_id))

    return body_ids


def attach_obstacle_runtime(env):
    """
    Add obstacle visualization and metrics to an existing environment.

    This deliberately does not:
    - change the observation size;
    - terminate on obstacle collision;
    - add obstacle penalties;
    - apply APF.
    """

    if not hasattr(env, "obstacle_manager"):
        raise AttributeError("Environment has no attached obstacle_manager.")

    env.obstacle_body_ids = create_obstacle_visuals(env)
    original_step = env.step

    def step_with_obstacle_metrics(self, action):
        obs, reward, terminated, truncated, info = original_step(action)

        sim_obs = self._get_current_sim_obs()
        positions = np.asarray(sim_obs[:, 0:3], dtype=np.float32)

        obstacle_collision_count, obstacle_events = (
            self.obstacle_manager.check_team_collisions(positions)
        )

        (
            min_obstacle_clearance,
            nearest_uav_index,
            nearest_obstacle_name,
        ) = self.obstacle_manager.team_min_clearance(positions)

        centroid_x = float(np.mean(positions[:, 0]))
        scenario_requested_formation = self.scenario.requested_formation(
            centroid_x
        )

        info["scenario_name"] = self.scenario.name
        info["scenario_difficulty"] = self.scenario.difficulty
        info["scenario_requested_formation"] = scenario_requested_formation
        info["obstacle_collision_count"] = int(obstacle_collision_count)
        info["obstacle_collision_events"] = obstacle_events
        info["min_obstacle_clearance"] = float(min_obstacle_clearance)
        info["nearest_obstacle_uav"] = nearest_uav_index
        info["nearest_obstacle_name"] = nearest_obstacle_name

        return obs, reward, terminated, truncated, info

    env.step = MethodType(step_with_obstacle_metrics, env)
    return env
