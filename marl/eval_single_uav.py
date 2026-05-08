import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pybullet as p
import cv2

from envs.single_uav_env import SingleUAVHoverEnv
from marl.train_single_uav import PolicyNet


# ----------- CONFIG ----------------
NUM_EVAL_EPISODES = 10
MODEL_PATH = "marl/models/best_single_uav_policy.pth"

RENDER = "human"
SAVE_VIDEO = True
VIDEO_PATH = "eval_single_uav_best.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 20

CAMERA_MODE = "fixed"     # "fixed" or "follow"
SLOW_PLAYBACK = True
SLEEP_TIME = 0.02        # increase to 0.06 or 0.08 for slower playback


# ----------- CAMERA / VISUALIZATION ----------------
def get_drone_position(env):
    s = env.env._getDroneStateVector(0)
    return np.array(s[0:3], dtype=np.float32)


def capture_frame(env):
    if CAMERA_MODE == "follow":
        target = get_drone_position(env).tolist()
        distance = 3.5
        yaw = 60
        pitch = -30
        fov = 65.0
    else:
        midpoint = ((env.start_pos + env.hover_target) / 2.0).tolist()
        target = midpoint
        distance = 4.0
        yaw = 60
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


def draw_marker(position, color, radius=0.08):
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis_id,
        basePosition=position,
    )

    return body_id


def draw_reference_path(env):
    A = env.start_pos.tolist()
    B = env.hover_target.tolist()
    M = ((env.start_pos + env.hover_target) / 2.0).tolist()

    draw_marker(A, [0.0, 0.8, 0.0, 1.0], radius=0.10)
    draw_marker(M, [1.0, 1.0, 0.0, 1.0], radius=0.07)
    draw_marker(B, [1.0, 0.0, 0.0, 1.0], radius=0.10)

    p.addUserDebugLine(
        lineFromXYZ=A,
        lineToXYZ=B,
        lineColorRGB=[1.0, 1.0, 1.0],
        lineWidth=3.0,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="START",
        textPosition=[A[0], A[1], A[2] + 0.25],
        textColorRGB=[0.0, 1.0, 0.0],
        textSize=1.4,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="LIFT PATH",
        textPosition=[M[0], M[1], M[2] + 0.25],
        textColorRGB=[1.0, 1.0, 0.0],
        textSize=1.2,
        lifeTime=0,
    )

    p.addUserDebugText(
        text="HOVER TARGET",
        textPosition=[B[0], B[1], B[2] + 0.25],
        textColorRGB=[1.0, 0.0, 0.0],
        textSize=1.4,
        lifeTime=0,
    )


def eval_playback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train first or check the saved model path."
        )

    env = SingleUAVHoverEnv(
        render_mode=RENDER,
        max_steps=300,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    successes = 0

    all_rewards = []
    all_hover_dists = []
    all_altitudes = []
    all_altitude_errors = []
    all_xy_errors = []
    all_speeds = []
    all_attitude_errors = []
    all_hover_counters = []
    all_crashes = []

    best_video_frames = []
    best_episode_reward = -np.inf
    best_episode_index = -1

    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset()

        if RENDER == "human":
            draw_reference_path(env)

        ep_rew = 0.0
        info = {}
        episode_frames = []

        for step in range(env.max_steps):
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_t)

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            obs = next_obs
            ep_rew += reward

            if SAVE_VIDEO and RENDER == "human":
                frame = capture_frame(env)
                episode_frames.append(frame)

            if SLOW_PLAYBACK and RENDER == "human":
                time.sleep(SLEEP_TIME)

            if done:
                break

        success = bool(info.get("is_success", False))
        successes += int(success)

        if ep_rew > best_episode_reward:
            best_episode_reward = ep_rew
            best_video_frames = episode_frames.copy()
            best_episode_index = ep + 1

        all_rewards.append(float(ep_rew))
        all_hover_dists.append(float(info.get("dist_to_hover", 0.0)))
        all_altitudes.append(float(info.get("altitude", 0.0)))
        all_altitude_errors.append(float(info.get("altitude_error", 0.0)))
        all_xy_errors.append(float(info.get("xy_error", 0.0)))
        all_speeds.append(float(info.get("speed", 0.0)))
        all_attitude_errors.append(float(info.get("attitude_error", 0.0)))
        all_hover_counters.append(float(info.get("hover_counter", 0.0)))
        all_crashes.append(float(info.get("crashed", False)))

        print(
            f"Episode {ep + 1}/{NUM_EVAL_EPISODES} | "
            f"Reward: {ep_rew:.2f} | "
            f"Success: {success} | "
            f"StableHover: {info.get('stable_hover', False)} | "
            f"HoverCounter: {info.get('hover_counter', 0)} | "
            f"DistHover: {info.get('dist_to_hover', 0.0):.3f} | "
            f"XYErr: {info.get('xy_error', 0.0):.3f} | "
            f"Altitude: {info.get('altitude', 0.0):.3f} | "
            f"AltErr: {info.get('altitude_error', 0.0):.3f} | "
            f"Speed: {info.get('speed', 0.0):.3f} | "
            f"AttErr: {info.get('attitude_error', 0.0):.3f} | "
            f"Crashed: {info.get('crashed', False)}"
        )

    print("\n========== SINGLE UAV HOVER EVAL SUMMARY ==========")
    print(
        f"Success Rate: {successes}/{NUM_EVAL_EPISODES} "
        f"({100.0 * successes / NUM_EVAL_EPISODES:.1f}%)"
    )
    print(f"Mean Reward: {np.mean(all_rewards):.3f}")
    print(f"Mean Distance To Hover: {np.mean(all_hover_dists):.3f}")
    print(f"Mean XY Error: {np.mean(all_xy_errors):.3f}")
    print(f"Mean Altitude: {np.mean(all_altitudes):.3f}")
    print(f"Mean Altitude Error: {np.mean(all_altitude_errors):.3f}")
    print(f"Mean Speed: {np.mean(all_speeds):.3f}")
    print(f"Mean Attitude Error: {np.mean(all_attitude_errors):.3f}")
    print(f"Mean Hover Counter: {np.mean(all_hover_counters):.3f}")
    print(f"Crash Rate: {np.mean(all_crashes) * 100.0:.1f}%")

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