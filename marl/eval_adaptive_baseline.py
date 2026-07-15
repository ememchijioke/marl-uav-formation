
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.env_factory import create_adaptive_env_from_yaml


def compute_geometry_error(
    positions: np.ndarray,
    formation_offsets: np.ndarray,
) -> float:
    """
    Pure formation geometry error.

    Compares actual pairwise UAV distances with the pairwise distances
    required by the active formation. This removes centroid and goal
    tracking error from the formation metric.
    """

    errors = []

    num_uavs = positions.shape[0]

    for i in range(num_uavs):
        for j in range(i + 1, num_uavs):
            actual_distance = float(
                np.linalg.norm(positions[i] - positions[j])
            )

            desired_distance = float(
                np.linalg.norm(
                    formation_offsets[i] - formation_offsets[j]
                )
            )

            errors.append(abs(actual_distance - desired_distance))

    if not errors:
        return 0.0

    return float(np.mean(errors))


def evaluate(
    config_path: str,
    episodes: int = 10,
) -> None:
    results = []

    for episode in range(episodes):
        env = create_adaptive_env_from_yaml(config_path)
        obs, _ = env.reset(seed=episode)

        action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

        episode_reward = 0.0

        min_pair_distance = float("inf")
        min_obstacle_clearance = float("inf")

        target_tracking_errors = []
        geometry_errors = []

        phase_steps = Counter()

        episode_uav_collision = False
        episode_obstacle_collision = False
        episode_crash = False

        final_info = {}
        step = 0

        for step in range(env.max_steps):
            obs, reward, terminated, truncated, info = env.step(action)

            final_info = info
            episode_reward += float(reward)

            min_pair_distance = min(
                min_pair_distance,
                float(
                    info.get(
                        "min_pair_dist",
                        float("inf"),
                    )
                ),
            )

            min_obstacle_clearance = min(
                min_obstacle_clearance,
                float(
                    info.get(
                        "min_obstacle_clearance",
                        float("inf"),
                    )
                ),
            )

            target_tracking_errors.append(
                float(info.get("formation_error", 0.0))
            )

            sim_obs = env._get_current_sim_obs()
            positions = np.asarray(
                sim_obs[:, 0:3],
                dtype=np.float32,
            )

            active_offsets = (
                env.formation_manager.current_offsets.copy()
            )

            geometry_errors.append(
                compute_geometry_error(
                    positions=positions,
                    formation_offsets=active_offsets,
                )
            )

            phase_name = info.get(
                "phase_name",
                "unknown",
            )

            phase_steps[phase_name] += 1

            if int(info.get("collision_count", 0)) > 0:
                episode_uav_collision = True

            if int(info.get("obstacle_collision_count", 0)) > 0:
                episode_obstacle_collision = True

            if bool(info.get("crashed", False)):
                episode_crash = True

            if terminated or truncated:
                break

        result = {
            "episode": episode + 1,
            "success": bool(
                final_info.get(
                    "mission_success",
                    False,
                )
            ),
            "crashed": bool(episode_crash),
            "uav_collision": bool(
                episode_uav_collision
            ),
            "obstacle_collision": bool(
                episode_obstacle_collision
            ),
            "reward": float(episode_reward),
            "steps": int(step + 1),
            "min_pair_distance": float(
                min_pair_distance
            ),
            "min_obstacle_clearance": float(
                min_obstacle_clearance
            ),
            "mean_target_tracking_error": float(
                np.mean(target_tracking_errors)
                if target_tracking_errors
                else 0.0
            ),
            "mean_geometry_error": float(
                np.mean(geometry_errors)
                if geometry_errors
                else 0.0
            ),
            "final_phase": final_info.get(
                "phase_name",
                "unknown",
            ),
            "phase_steps": dict(phase_steps),
        }

        results.append(result)

        print(
            f"Episode {episode + 1:02d} | "
            f"success={result['success']} | "
            f"crashed={result['crashed']} | "
            f"uav_collision={result['uav_collision']} | "
            f"obstacle_collision={result['obstacle_collision']} | "
            f"steps={result['steps']} | "
            f"pair={result['min_pair_distance']:.3f} | "
            f"clearance={result['min_obstacle_clearance']:.3f} | "
            f"geometry={result['mean_geometry_error']:.3f}"
        )

        env.close()

    success_rate = (
        np.mean([r["success"] for r in results])
        * 100.0
    )

    crash_rate = (
        np.mean([r["crashed"] for r in results])
        * 100.0
    )

    uav_collision_rate = (
        np.mean(
            [r["uav_collision"] for r in results]
        )
        * 100.0
    )

    obstacle_collision_rate = (
        np.mean(
            [r["obstacle_collision"] for r in results]
        )
        * 100.0
    )

    mean_pair_distance = float(
        np.mean(
            [r["min_pair_distance"] for r in results]
        )
    )

    mean_obstacle_clearance = float(
        np.mean(
            [
                r["min_obstacle_clearance"]
                for r in results
            ]
        )
    )

    mean_geometry_error = float(
        np.mean(
            [r["mean_geometry_error"] for r in results]
        )
    )

    mean_target_tracking_error = float(
        np.mean(
            [
                r["mean_target_tracking_error"]
                for r in results
            ]
        )
    )

    mean_steps = float(
        np.mean([r["steps"] for r in results])
    )

    print("\n===== ADAPTIVE BASELINE SUMMARY =====")
    print(f"Episodes: {episodes}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Crash rate: {crash_rate:.1f}%")
    print(
        f"UAV collision rate: "
        f"{uav_collision_rate:.1f}%"
    )
    print(
        f"Obstacle collision rate: "
        f"{obstacle_collision_rate:.1f}%"
    )
    print(
        "Mean minimum pair distance: "
        f"{mean_pair_distance:.3f} m"
    )
    print(
        "Mean minimum obstacle clearance: "
        f"{mean_obstacle_clearance:.3f} m"
    )
    print(
        "Mean pure geometry error: "
        f"{mean_geometry_error:.3f} m"
    )
    print(
        "Mean assigned-target tracking error: "
        f"{mean_target_tracking_error:.3f} m"
    )
    print(
        "Mean episode steps: "
        f"{mean_steps:.1f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        episodes=args.episodes,
    )
