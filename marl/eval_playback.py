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


# ============================================================
# CONFIG
# ============================================================

NUM_EVAL_EPISODES = 1
MAX_STEPS = 700

# v0.5 trained model path
MODEL_PATH = "marl/models/v0_5_column_landing/best_shared_policy_v0_5.pth"
FALLBACK_MODEL_PATH = "checkpoints_v0_5_column_landing/shared_policy.pth"

# Rendering / saving
RENDER = "human"          # "human" or "headless"
SAVE_VIDEO = True
VIDEO_PATH = "eval_v0_5_column_formation.mp4"

# Video quality
WIDTH = 1280
HEIGHT = 720
FPS = 20

# Save raw PNG frames for high-quality ffmpeg encoding
SAVE_FRAMES = True
FRAME_DIR = "eval_frames_v0_5_column"

# Camera mode
CAMERA_MODE = "follow"    # "follow" or "fixed"

# Playback speed
SLOW_PLAYBACK = True
SLEEP_TIME = 0.0


# ============================================================
# POLICY NETWORK
# ============================================================

class SharedPolicyNet(nn.Module):
    """
    Evaluation-side policy definition.

    Kept here instead of importing from train_mappo.py so that evaluation
    does not accidentally trigger argparse/training-side code.
    """

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


# ============================================================
# CAMERA / VISUALIZATION
# ============================================================

def get_team_centroid(env):
    positions = []

    for i in range(env.num_agents):
        state = env.env._getDroneStateVector(i)
        positions.append(state[0:3])

    return np.mean(np.array(positions, dtype=np.float32), axis=0)


def get_camera_settings(env):
    if CAMERA_MODE == "follow":
        centroid = get_team_centroid(env)

        target = [
            float(centroid[0] + 0.10),
            float(centroid[1]),
            float(centroid[2] + 0.05),
        ]

        # Slightly farther than v0.4 so the full column is visible.
        distance = 2.3
        yaw = 45
        pitch = -15
        fov = 38.0

    else:
        target = [1.4, 0.0, 0.9]
        distance = 4.6
        yaw = 45
        pitch = -24
        fov = 55.0

    return target, distance, yaw, pitch, fov


def reset_live_camera(env):
    target, distance, yaw, pitch, _ = get_camera_settings(env)

    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
    )


def capture_frame(env):
    target, distance, yaw, pitch, fov = get_camera_settings(env)

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
    Draw clean v0.5 reference guide lines.

    The guide lines show:
    - each UAV's takeoff path
    - each UAV's A to B travel path
    - each UAV's landing path
    - the column shape at A and B
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

    # Column shape at A: rear -> middle -> front
    p.addUserDebugLine(
        lineFromXYZ=env.hover_A_targets[2].tolist(),
        lineToXYZ=env.hover_A_targets[1].tolist(),
        lineColorRGB=[0.0, 0.6, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    p.addUserDebugLine(
        lineFromXYZ=env.hover_A_targets[1].tolist(),
        lineToXYZ=env.hover_A_targets[0].tolist(),
        lineColorRGB=[0.0, 0.6, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    # Column shape at B: rear -> middle -> front
    p.addUserDebugLine(
        lineFromXYZ=env.hover_B_targets[2].tolist(),
        lineToXYZ=env.hover_B_targets[1].tolist(),
        lineColorRGB=[0.0, 0.6, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    p.addUserDebugLine(
        lineFromXYZ=env.hover_B_targets[1].tolist(),
        lineToXYZ=env.hover_B_targets[0].tolist(),
        lineColorRGB=[0.0, 0.6, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="v0.5: 3 UAV SINGLE-FILE COLUMN FORMATION + LANDING",
        textPosition=[0.15, -1.05, 1.45],
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.1,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="UAV0 FRONT",
        textPosition=env.hover_A_targets[0].tolist(),
        textColorRGB=[1.0, 1.0, 0.0],
        textSize=0.8,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="UAV1 MIDDLE",
        textPosition=env.hover_A_targets[1].tolist(),
        textColorRGB=[1.0, 1.0, 0.0],
        textSize=0.8,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="UAV2 REAR",
        textPosition=env.hover_A_targets[2].tolist(),
        textColorRGB=[1.0, 1.0, 0.0],
        textSize=0.8,
        lifeTime=0,
    )


# ============================================================
# HELPERS
# ============================================================

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


def save_video(frames, video_path):
    if len(frames) == 0:
        print("[WARN] No frames captured. Video was not saved.")
        return

    if SAVE_FRAMES:
        os.makedirs(FRAME_DIR, exist_ok=True)

        for idx, frame in enumerate(frames):
            frame_path = os.path.join(FRAME_DIR, f"frame_{idx:05d}.png")
            cv2.imwrite(frame_path, frame)

        print(f"\nSaved PNG frames to: {FRAME_DIR}")
        print(f"Frames saved: {len(frames)}")
        print("\nNow create a high-quality MP4 with:")
        print(
            f"ffmpeg -y -framerate {FPS} -i {FRAME_DIR}/frame_%05d.png "
            f"-c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p {video_path}"
        )

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    for frame in frames:
        out.write(frame)

    out.release()

    print(f"\nSaved fallback OpenCV video: {video_path}")
    print(f"Duration: {len(frames) / FPS:.2f} seconds")


# ============================================================
# EVALUATION
# ============================================================

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
    print(f"Camera mode: {CAMERA_MODE}")
    print(f"Saving video: {SAVE_VIDEO}")
    print(f"Video path: {VIDEO_PATH}")

    policy = SharedPolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    successes = 0

    all_rewards = []
    all_centroid_errors = []
    all_formation_errors = []
    all_column_spacing_errors = []
    all_mean_distances = []
    all_mean_speeds = []
    all_min_pair_dists = []
    all_collision_counts = []

    saved_frames = []

    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset()

        if RENDER == "human":
            reset_live_camera(env)
            draw_reference_path(env)

        obs_split = np.split(obs, n_agents)

        ep_reward = 0.0
        info = {}

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

            if RENDER == "human":
                reset_live_camera(env)

            if SAVE_VIDEO and RENDER == "human":
                frame = capture_frame(env)
                saved_frames.append(frame)

            if SLOW_PLAYBACK and RENDER == "human":
                time.sleep(SLEEP_TIME)

            if done:
                break

        success = bool(info.get("is_success", False))
        landed = bool(info.get("landed", False))
        successes += int(success)

        all_rewards.append(float(ep_reward))
        all_centroid_errors.append(metric(info, "centroid_error"))
        all_formation_errors.append(metric(info, "formation_error"))
        all_column_spacing_errors.append(metric(info, "column_spacing_error"))
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
            f"ColumnErr: {metric(info, 'column_spacing_error'):.3f} | "
            f"D01: {metric(info, 'spacing_uav0_uav1'):.3f} | "
            f"D12: {metric(info, 'spacing_uav1_uav2'):.3f} | "
            f"TargetDist: {metric(info, 'mean_dist_to_target'):.3f} | "
            f"Speed: {metric(info, 'mean_speed'):.3f} | "
            f"MinSep: {metric(info, 'min_pair_dist'):.3f} | "
            f"Collisions: {int(metric(info, 'collision_count'))} | "
            f"Crashed: {info.get('crashed', False)}"
        )

    print("\n========== v0.5 SINGLE-FILE COLUMN FORMATION EVAL SUMMARY ==========")
    print(
        f"Success Rate: {successes}/{NUM_EVAL_EPISODES} "
        f"({100.0 * successes / NUM_EVAL_EPISODES:.1f}%)"
    )
    print(f"Mean Reward: {np.mean(all_rewards):.3f}")
    print(f"Mean Centroid Error: {np.mean(all_centroid_errors):.3f}")
    print(f"Mean Formation Error: {np.mean(all_formation_errors):.3f}")
    print(f"Mean Column Spacing Error: {np.mean(all_column_spacing_errors):.3f}")
    print(f"Mean Distance To Target: {np.mean(all_mean_distances):.3f}")
    print(f"Mean Speed: {np.mean(all_mean_speeds):.3f}")
    print(f"Mean Min Pair Distance: {np.mean(all_min_pair_dists):.3f}")
    print(f"Mean Collision Count: {np.mean(all_collision_counts):.3f}")

    if SAVE_VIDEO and RENDER == "human":
        save_video(saved_frames, VIDEO_PATH)

    env.close()


if __name__ == "__main__":
    eval_playback()