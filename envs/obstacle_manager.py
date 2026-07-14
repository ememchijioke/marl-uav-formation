
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """
    Axis-aligned obstacle description shared across PyBullet, Gazebo,
    evaluation, and later hardware layouts.

    Parameters
    ----------
    name:
        Unique obstacle identifier.
    kind:
        "building", "tree", "column", or another descriptive type.
    center:
        Obstacle center position [x, y, z].
    size:
        Full obstacle dimensions [length_x, width_y, height_z].
    safety_margin:
        Additional clearance added during collision and distance checks.
    """

    name: str
    kind: str
    center: np.ndarray
    size: np.ndarray
    safety_margin: float = 0.15

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float32)
        size = np.asarray(self.size, dtype=np.float32)

        if center.shape != (3,):
            raise ValueError("Obstacle center must have shape (3,).")

        if size.shape != (3,):
            raise ValueError("Obstacle size must have shape (3,).")

        if np.any(size <= 0.0):
            raise ValueError("Obstacle dimensions must be greater than zero.")

        if self.safety_margin < 0.0:
            raise ValueError("safety_margin cannot be negative.")

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)

    @property
    def half_size(self) -> np.ndarray:
        return self.size / 2.0

    @property
    def min_corner(self) -> np.ndarray:
        return self.center - self.half_size

    @property
    def max_corner(self) -> np.ndarray:
        return self.center + self.half_size


class ObstacleManager:
    """
    Stores known static obstacles and provides geometry queries.

    This class does not implement APF or policy logic. It only answers:
    - Is a UAV inside an obstacle safety volume?
    - What is the nearest obstacle?
    - What is the nearest point on an obstacle?
    - What is the signed clearance to an obstacle?
    """

    def __init__(
        self,
        obstacles: Iterable[Obstacle] | None = None,
        drone_radius: float = 0.08,
    ) -> None:
        if drone_radius <= 0.0:
            raise ValueError("drone_radius must be greater than zero.")

        self.drone_radius = float(drone_radius)
        self.obstacles: list[Obstacle] = list(obstacles or [])

    def reset(self, obstacles: Iterable[Obstacle] | None = None) -> None:
        if obstacles is not None:
            self.obstacles = list(obstacles)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        if any(existing.name == obstacle.name for existing in self.obstacles):
            raise ValueError(f"Obstacle name already exists: {obstacle.name}")

        self.obstacles.append(obstacle)

    def get_obstacle(self, name: str) -> Obstacle:
        for obstacle in self.obstacles:
            if obstacle.name == name:
                return obstacle

        raise KeyError(f"Unknown obstacle: {name}")

    def _inflated_bounds(self, obstacle: Obstacle) -> tuple[np.ndarray, np.ndarray]:
        inflation = obstacle.safety_margin + self.drone_radius
        inflation_vector = np.array(
            [inflation, inflation, inflation],
            dtype=np.float32,
        )

        return (
            obstacle.min_corner - inflation_vector,
            obstacle.max_corner + inflation_vector,
        )

    def nearest_point_on_obstacle(
        self,
        position: np.ndarray,
        obstacle: Obstacle,
        include_safety_margin: bool = True,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float32)

        if position.shape != (3,):
            raise ValueError("position must have shape (3,).")

        if include_safety_margin:
            min_corner, max_corner = self._inflated_bounds(obstacle)
        else:
            min_corner = obstacle.min_corner
            max_corner = obstacle.max_corner

        return np.clip(position, min_corner, max_corner).astype(np.float32)

    def signed_clearance(
        self,
        position: np.ndarray,
        obstacle: Obstacle,
    ) -> float:
        """
        Signed Euclidean clearance to the inflated obstacle box.

        Positive:
            UAV is outside the safety volume.
        Zero:
            UAV is on the safety boundary.
        Negative:
            UAV is inside the safety volume.
        """

        position = np.asarray(position, dtype=np.float32)
        min_corner, max_corner = self._inflated_bounds(obstacle)

        outside_delta = np.maximum(
            np.maximum(min_corner - position, position - max_corner),
            0.0,
        )

        outside_distance = float(np.linalg.norm(outside_delta))

        inside = bool(
            np.all(position >= min_corner)
            and np.all(position <= max_corner)
        )

        if not inside:
            return outside_distance

        distances_to_faces = np.array(
            [
                position[0] - min_corner[0],
                max_corner[0] - position[0],
                position[1] - min_corner[1],
                max_corner[1] - position[1],
                position[2] - min_corner[2],
                max_corner[2] - position[2],
            ],
            dtype=np.float32,
        )

        return -float(np.min(distances_to_faces))

    def check_collision(
        self,
        position: np.ndarray,
    ) -> tuple[bool, str | None]:
        for obstacle in self.obstacles:
            if self.signed_clearance(position, obstacle) <= 0.0:
                return True, obstacle.name

        return False, None

    def check_team_collisions(
        self,
        positions: np.ndarray,
    ) -> tuple[int, list[dict]]:
        positions = np.asarray(positions, dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (num_uavs, 3).")

        events: list[dict] = []

        for uav_index, position in enumerate(positions):
            collided, obstacle_name = self.check_collision(position)

            if collided:
                events.append(
                    {
                        "uav_index": int(uav_index),
                        "obstacle_name": obstacle_name,
                    }
                )

        return len(events), events

    def nearest_obstacle(
        self,
        position: np.ndarray,
    ) -> dict | None:
        position = np.asarray(position, dtype=np.float32)

        if not self.obstacles:
            return None

        best_result: dict | None = None

        for obstacle in self.obstacles:
            nearest_point = self.nearest_point_on_obstacle(
                position,
                obstacle,
                include_safety_margin=True,
            )

            relative_vector = nearest_point - position
            clearance = self.signed_clearance(position, obstacle)

            candidate = {
                "name": obstacle.name,
                "kind": obstacle.kind,
                "center": obstacle.center.copy(),
                "size": obstacle.size.copy(),
                "nearest_point": nearest_point,
                "relative_vector": relative_vector.astype(np.float32),
                "distance": float(abs(clearance)),
                "signed_clearance": float(clearance),
                "inside_safety_volume": bool(clearance <= 0.0),
            }

            if (
                best_result is None
                or candidate["signed_clearance"] < best_result["signed_clearance"]
            ):
                best_result = candidate

        return best_result

    def team_min_clearance(
        self,
        positions: np.ndarray,
    ) -> tuple[float, int | None, str | None]:
        positions = np.asarray(positions, dtype=np.float32)

        if not self.obstacles:
            return float("inf"), None, None

        best_clearance = float("inf")
        best_uav_index: int | None = None
        best_obstacle_name: str | None = None

        for uav_index, position in enumerate(positions):
            nearest = self.nearest_obstacle(position)

            if nearest is None:
                continue

            clearance = float(nearest["signed_clearance"])

            if clearance < best_clearance:
                best_clearance = clearance
                best_uav_index = int(uav_index)
                best_obstacle_name = str(nearest["name"])

        return best_clearance, best_uav_index, best_obstacle_name

    def obstacle_features(
        self,
        position: np.ndarray,
        position_scale: float = 5.0,
        size_scale: float = 5.0,
        clearance_scale: float = 3.0,
    ) -> np.ndarray:
        """
        Compact known-obstacle feature vector for one UAV.

        Returns 8 features:
        relative nearest point xyz
        obstacle size xyz
        signed clearance
        valid mask
        """

        nearest = self.nearest_obstacle(position)

        if nearest is None:
            return np.zeros(8, dtype=np.float32)

        relative = np.clip(
            nearest["relative_vector"] / max(position_scale, 1e-6),
            -1.0,
            1.0,
        )

        size = np.clip(
            nearest["size"] / max(size_scale, 1e-6),
            0.0,
            1.0,
        )

        clearance = np.array(
            [
                np.clip(
                    nearest["signed_clearance"] / max(clearance_scale, 1e-6),
                    -1.0,
                    1.0,
                )
            ],
            dtype=np.float32,
        )

        valid = np.array([1.0], dtype=np.float32)

        return np.concatenate(
            [
                relative.astype(np.float32),
                size.astype(np.float32),
                clearance,
                valid,
            ],
            axis=0,
        ).astype(np.float32)

    def create_pybullet_visuals(
        self,
        pybullet_client,
        rgba_by_kind: dict[str, list[float]] | None = None,
    ) -> list[int]:
        """
        Create static box visuals in the supplied PyBullet client.

        This is optional and only used by environments that want visible
        obstacles. Collision checks remain handled by this manager.
        """

        colors = rgba_by_kind or {
            "building": [0.45, 0.45, 0.50, 1.0],
            "tree": [0.20, 0.55, 0.25, 1.0],
            "column": [0.80, 0.45, 0.20, 1.0],
        }

        body_ids: list[int] = []

        for obstacle in self.obstacles:
            half_extents = (obstacle.size / 2.0).tolist()
            rgba = colors.get(obstacle.kind, [0.60, 0.60, 0.60, 1.0])

            collision_shape = pybullet_client.createCollisionShape(
                shapeType=pybullet_client.GEOM_BOX,
                halfExtents=half_extents,
            )

            visual_shape = pybullet_client.createVisualShape(
                shapeType=pybullet_client.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=rgba,
            )

            body_id = pybullet_client.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obstacle.center.tolist(),
            )

            body_ids.append(int(body_id))

        return body_ids
