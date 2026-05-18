import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pybullet as p
import cv2

from envs.quad_env import MultiUAVRealisticEnv


# ----------- CONFIG ----------------
NUM_EVAL_EPISODES = 10
MAX_STEPS = 700

# v0.4 trained model path. Keep this as the first/default path.
MODEL_PATH = "marl/models/v0_4_line_landing/best_shared_policy_v0_4.pth"
FALLBACK_MODEL_PATH = "checkpoints_v0_4_line_landing/shared_policy.pth"

RENDER = "human"          # "human" or "headless"
SAVE_VIDEO = True
VIDEO_PATH = "eval_v0_4_straight_line_best.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 20

CAMERA_MODE = "fixed"     # "fixed" or "follow"
SLOW_PLAYBACK = True
SLEEP_TIME = 0.03


# ----------- POLICY NETWORK ----------------
# Defined here instead of importing from train_mappo.py so evaluation does not
# accidentally trigger argparse/training-side code.
class SharedPolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


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
        # Wide view for v0.4 straight-line formation A -> B -> land.
        target = [1.1, 0.0, 0.65]
        distance = 5.6
        yaw = 65
        pitch = -30
        fov = 72.0

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
    Draw clean v0.4 reference guide lines.
    This keeps the same style as the old evaluator, but now uses the
    straight-line formation targets from the v0.4 environment.
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
        
    # Formation line at A and B for visual clarity.
    p.addUserDebugLine(
        lineFromXYZ=env.hover_A_targets[1].tolist(),
        lineToXYZ=env.hover_A_targets[2].tolist(),
        lineColorRGB=[0.0, 0.6, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    p.addUserDebugLine(
        lineFromXYZ=env.hover_B_targets[1].tolist(),
        lineToXYZ=env.hover_B_targets[2].tolist(),
        lineColorRGB=[0.0, 0.6, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="v0.4: 3 UAV STRAIGHT-LINE FORMATION + LANDING",
        textPosition=[0.35, -1.25, 1.45],
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.1,
        lifeTime=0,
    )


def get_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists(FALLBACK_MODEL_PATH):
        print(f"[WARN] Main model not found: {MODEL_PATH}")
        print(f"[WARN] Using fallback checkpoint: {FALLBACK_MODEL_PATH}")
        return FALLBACK_MODEL_PATH

    raise FileNotFoundError(
        f"Model not found. Tried:\n"
        f"  1. {MODEL_PATH}\n"
        f"  2. {FALLBACK_MODEL_PATH}\n"
        "Train first or update MODEL_PATH in marl/eval_playback.py."
    )


def metric(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


# ----------- EVALUATION ----------------
def eval_playback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = get_model_path()
    print(f"Loading model: {model_path}")

    env = MultiUAVRealisticEnv(
        render_mode=RENDER,
        num_agents=3,
        max_steps=MAX_STEPS,
    )

    n_agents = env.num_agents
    obs_dim_total = env.observation_space.shape[0]
    act_dim_total = env.action_space.shape[0]

    obs_dim = obs_dim_total // n_agents
    act_dim = act_dim_total // n_agents

    print(f"Obs dim per agent: {obs_dim}")
    print(f"Action dim per agent: {act_dim}")

    policy = SharedPolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    successes = 0

    all_rewards = []
    all_centroid_errors = []
    all_formation_errors = []
    all_max_formation_errors = []
    all_mean_distances = []
    all_mean_speeds = []
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

        for step in range(MAX_STEPS):
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
            ep_reward += float(reward)

            # Keep same behavior as your old evaluator: save GUI video only in human mode.
            if SAVE_VIDEO and RENDER == "human":
                frame = capture_frame(env)
                episode_frames.append(frame)

            if SLOW_PLAYBACK and RENDER == "human":
                time.sleep(SLEEP_TIME)

            if done:
                break

        success = bool(info.get("is_success", False))
        landed = bool(info.get("landed", False))
        successes += int(success)

        if ep_reward > best_episode_reward:
            best_episode_reward = ep_reward
            best_video_frames = episode_frames.copy()
            best_episode_index = ep + 1

        all_rewards.append(float(ep_reward))
        all_centroid_errors.append(metric(info, "centroid_error"))
        all_formation_errors.append(metric(info, "formation_error"))
        all_max_formation_errors.append(metric(info, "max_agent_formation_error"))
        all_mean_distances.append(metric(info, "mean_dist_to_target"))
        all_mean_speeds.append(metric(info, "mean_speed"))
        all_min_pair_dists.append(metric(info, "min_pair_dist"))
        all_collision_counts.append(metric(info, "collision_count"))

        print(
            f"Episode {ep + 1}/{NUM_EVAL_EPISODES} | "
            f"Reward: {ep_reward:.2f} | "
            f"Success: {success} | "
            f"Landed: {landed} | "
            f"Phase: {info.get('phase_label', '')} | "
            f"CentroidErr: {metric(info, 'centroid_error'):.3f} | "
            f"FormErr: {metric(info, 'formation_error'):.3f} | "
            f"MaxFormErr: {metric(info, 'max_agent_formation_error'):.3f} | "
            f"TargetDist: {metric(info, 'mean_dist_to_target'):.3f} | "
            f"Speed: {metric(info, 'mean_speed'):.3f} | "
            f"MinSep: {metric(info, 'min_pair_dist'):.3f} | "
            f"Collisions: {int(metric(info, 'collision_count'))} | "
            f"Crashed: {info.get('crashed', False)}"
        )

    print("\n========== v0.4 STRAIGHT-LINE FORMATION EVAL SUMMARY ==========")
    print(
        f"Success Rate: {successes}/{NUM_EVAL_EPISODES} "
        f"({100.0 * successes / NUM_EVAL_EPISODES:.1f}%)"
    )
    print(f"Mean Reward: {np.mean(all_rewards):.3f}")
    print(f"Mean Centroid Error: {np.mean(all_centroid_errors):.3f}")
    print(f"Mean Formation Error: {np.mean(all_formation_errors):.3f}")
    print(f"Mean Max Formation Error: {np.mean(all_max_formation_errors):.3f}")
    print(f"Mean Distance To Target: {np.mean(all_mean_distances):.3f}")
    print(f"Mean Speed: {np.mean(all_mean_speeds):.3f}")
    print(f"Mean Min Pair Distance: {np.mean(all_min_pair_dists):.3f}")
    print(f"Mean Collision Count: {np.mean(all_collision_counts):.3f}")

    if SAVE_VIDEO and RENDER == "human" and len(best_video_frames) > 0:
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