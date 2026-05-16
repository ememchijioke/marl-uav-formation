import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pybullet as p
import cv2

from envs.quad_env import MultiUAVRealisticEnv
from marl.train_mappo import SharedPolicyNet


# ----------- CONFIG ----------------
NUM_EVAL_EPISODES = 10
MODEL_PATH = "marl/models/best_shared_policy_realistic.pth"

RENDER = "human"
SAVE_VIDEO = True
VIDEO_PATH = "eval_3uav_mission_best.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 20

CAMERA_MODE = "fixed"     # "fixed" or "follow"
SLOW_PLAYBACK = True
SLEEP_TIME = 0.03


# ----------- CAMERA / VISUALIZATION ----------------
def get_team_centroid(env):
    positions = []

    for i in range(env.num_agents):
        s = env.env._getDroneStateVector(i)
        positions.append(s[0:3])

    return np.mean(np.array(positions, dtype=np.float32), axis=0)


def capture_frame(env):
    if CAMERA_MODE == "follow":
        target = get_team_centroid(env).tolist()
        distance = 4.8
        yaw = 60
        pitch = -30
        fov = 65.0
    else:
        # Wide view for 3 UAV lane mission.
        target = [1.0, 0.0, 0.65]
        distance = 5.2
        yaw = 65
        pitch = -30
        fov = 70.0

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=WIDTH / HEIGHT,
        nearVal=0.1,
        farVal=100.0,
    )

    _, _, px, _, _ = p.getCameraImage(
        width=WIDTH,
        height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    frame = np.array(px, dtype=np.uint8)[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def draw_reference_path(env):
    """
    Draw clean reference guide lines only.
    No marker balls, no formation slots.
    """

    for i in range(env.num_agents):
        start = env.start_positions[i].tolist()
        hover_a = env.hover_A_targets[i].tolist()
        hover_b = env.hover_B_targets[i].tolist()
        land_b = env.land_B_targets[i].tolist()

        # Takeoff line
        p.addUserDebugLine(
            lineFromXYZ=start,
            lineToXYZ=hover_a,
            lineColorRGB=[0.0, 1.0, 0.0],
            lineWidth=2.0,
            lifeTime=0,
        )

        # A to B travel line
        p.addUserDebugLine(
            lineFromXYZ=hover_a,
            lineToXYZ=hover_b,
            lineColorRGB=[1.0, 1.0, 1.0],
            lineWidth=2.0,
            lifeTime=0,
        )

        # Landing line
        p.addUserDebugLine(
            lineFromXYZ=hover_b,
            lineToXYZ=land_b,
            lineColorRGB=[1.0, 0.0, 0.0],
            lineWidth=2.0,
            lifeTime=0,
        )

    p.addUserDebugText(
        text="3 UAV WAYPOINT + LANDING MISSION",
        textPosition=[0.75, -1.0, 1.35],
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.2,
        lifeTime=0,
    )


def eval_playback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train first or check the saved model path."
        )

    env = MultiUAVRealisticEnv(
        render_mode=RENDER,
        num_agents=3,
        max_steps=600,
    )

    n_agents = env.num_agents
    obs_dim_total = env.observation_space.shape[0]
    act_dim_total = env.action_space.shape[0]

    obs_dim = obs_dim_total // n_agents
    act_dim = act_dim_total // n_agents

    policy = SharedPolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    successes = 0

    all_rewards = []
    all_completed_agents = []
    all_mean_distances = []
    all_mean_xy_errors = []
    all_mean_speeds = []
    all_mean_attitudes = []
    all_min_pair_dists = []
    all_collision_counts = []

    best_video_frames = []
    best_episode_reward = -np.inf
    best_episode_index = -1

    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset()

        if RENDER == "human":
            draw_reference_path(env)

        obs_split = np.split(obs, n_agents)

        ep_reward = 0.0
        info = {}
        episode_frames = []

        for step in range(env.max_steps):
            actions = []

            with torch.no_grad():
                for i in range(n_agents):
                    obs_t = torch.tensor(
                        obs_split[i],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    action = policy(obs_t)
                    actions.append(action.squeeze(0).cpu().numpy())

            flat_action = np.concatenate(actions, axis=0)

            next_obs, reward, terminated, truncated, info = env.step(flat_action)
            done = bool(terminated or truncated)

            obs_split = np.split(next_obs, n_agents)
            ep_reward += reward

            if SAVE_VIDEO and RENDER == "human":
                frame = capture_frame(env)
                episode_frames.append(frame)

            if SLOW_PLAYBACK and RENDER == "human":
                time.sleep(SLEEP_TIME)

            if done:
                break

        success = bool(info.get("is_success", False))
        successes += int(success)

        if ep_reward > best_episode_reward:
            best_episode_reward = ep_reward
            best_video_frames = episode_frames.copy()
            best_episode_index = ep + 1

        all_rewards.append(float(ep_reward))
        all_completed_agents.append(float(info.get("completed_agents", 0)))
        all_mean_distances.append(float(info.get("mean_dist_to_target", 0.0)))
        all_mean_xy_errors.append(float(info.get("mean_xy_error", 0.0)))
        all_mean_speeds.append(float(info.get("mean_speed", 0.0)))
        all_mean_attitudes.append(float(info.get("mean_attitude_error", 0.0)))
        all_min_pair_dists.append(float(info.get("min_pair_dist", 0.0)))
        all_collision_counts.append(float(info.get("collision_count", 0.0)))

        print(
            f"Episode {ep + 1}/{NUM_EVAL_EPISODES} | "
            f"Reward: {ep_reward:.2f} | "
            f"Success: {success} | "
            f"Agents: {info.get('completed_agents', 0)}/3 | "
            f"MeanPhase: {info.get('mean_phase', 0.0):.2f} | "
            f"Phases: {info.get('phase_labels', [])} | "
            f"Dist: {info.get('mean_dist_to_target', 0.0):.3f} | "
            f"XYErr: {info.get('mean_xy_error', 0.0):.3f} | "
            f"Speed: {info.get('mean_speed', 0.0):.3f} | "
            f"MinSep: {info.get('min_pair_dist', 0.0):.3f} | "
            f"Collisions: {info.get('collision_count', 0)} | "
            f"Crashed: {info.get('crashed', False)}"
        )

    print("\n========== 3 UAV MISSION EVAL SUMMARY ==========")
    print(
        f"Success Rate: {successes}/{NUM_EVAL_EPISODES} "
        f"({100.0 * successes / NUM_EVAL_EPISODES:.1f}%)"
    )
    print(f"Mean Reward: {np.mean(all_rewards):.3f}")
    print(f"Mean Completed Agents: {np.mean(all_completed_agents):.3f}/3")
    print(f"Mean Distance To Target: {np.mean(all_mean_distances):.3f}")
    print(f"Mean XY Error: {np.mean(all_mean_xy_errors):.3f}")
    print(f"Mean Speed: {np.mean(all_mean_speeds):.3f}")
    print(f"Mean Attitude Error: {np.mean(all_mean_attitudes):.3f}")
    print(f"Mean Min Pair Distance: {np.mean(all_min_pair_dists):.3f}")
    print(f"Mean Collision Count: {np.mean(all_collision_counts):.3f}")

    if SAVE_VIDEO and len(best_video_frames) > 0:
        out = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT),
        )

        for frame in best_video_frames:
            out.write(frame)

        out.release()

        print(f"\nSaved BEST episode video: {VIDEO_PATH}")
        print(f"Best Episode: {best_episode_index}")
        print(f"Best Episode Reward: {best_episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    eval_playback()