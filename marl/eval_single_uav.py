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


NUM_EVAL_EPISODES = 10
MODEL_PATH = "marl/models/best_single_uav_policy.pth"

RENDER = "human"
SAVE_VIDEO = True
VIDEO_PATH = "eval_single_uav_best.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 20

CAMERA_MODE = "fixed"
SLOW_PLAYBACK = True
SLEEP_TIME = 0.03


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
        target = [1.0, 0.0, 0.6]
        distance = 4.5
        yaw = 65
        pitch = -28
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
    A_ground = env.start_pos.tolist()
    A_hover = env.hover_A.tolist()
    B_hover = env.hover_B.tolist()
    B_land = env.land_B.tolist()

    #draw_marker(A_ground, [0.0, 0.8, 0.0, 1.0], radius=0.08)
    # draw_marker(A_hover, [0.0, 0.5, 1.0, 1.0], radius=0.08)
    # draw_marker(B_hover, [1.0, 0.5, 0.0, 1.0], radius=0.08)
    # draw_marker(B_land, [1.0, 0.0, 0.0, 1.0], radius=0.08)

    p.addUserDebugLine(A_ground, A_hover, [0.0, 1.0, 0.0], 3.0, 0)
    p.addUserDebugLine(A_hover, B_hover, [1.0, 1.0, 1.0], 3.0, 0)
    p.addUserDebugLine(B_hover, B_land, [1.0, 0.0, 0.0], 3.0, 0)

    p.addUserDebugText(
        "A TAKEOFF",
        [A_ground[0], A_ground[1], A_ground[2] + 0.20],
        [0.0, 1.0, 0.0],
        textSize=1.3,
        lifeTime=0,
    )

    p.addUserDebugText(
        "A HOVER",
        [A_hover[0], A_hover[1], A_hover[2] + 0.20],
        [0.0, 0.6, 1.0],
        textSize=1.3,
        lifeTime=0,
    )

    p.addUserDebugText(
        "B HOVER",
        [B_hover[0], B_hover[1], B_hover[2] + 0.20],
        [1.0, 0.6, 0.0],
        textSize=1.3,
        lifeTime=0,
    )

    p.addUserDebugText(
        "B LAND",
        [B_land[0], B_land[1], B_land[2] + 0.20],
        [1.0, 0.0, 0.0],
        textSize=1.3,
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
        max_steps=600,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    successes = 0

    all_rewards = []
    all_distances = []
    all_altitudes = []
    all_xy_errors = []
    all_speeds = []
    all_attitude_errors = []
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
        all_distances.append(float(info.get("dist_to_target", 0.0)))
        all_altitudes.append(float(info.get("altitude", 0.0)))
        all_xy_errors.append(float(info.get("xy_error", 0.0)))
        all_speeds.append(float(info.get("speed", 0.0)))
        all_attitude_errors.append(float(info.get("attitude_error", 0.0)))
        all_crashes.append(float(info.get("crashed", False)))

        print(
            f"Episode {ep + 1}/{NUM_EVAL_EPISODES} | "
            f"Reward: {ep_rew:.2f} | "
            f"Success: {success} | "
            f"Phase: {info.get('phase', '')} | "
            f"StableCounter: {info.get('stable_counter', 0)} | "
            f"MoveProgress: {info.get('move_progress', 0.0):.2f} | "
            f"Dist: {info.get('dist_to_target', 0.0):.3f} | "
            f"XYErr: {info.get('xy_error', 0.0):.3f} | "
            f"Altitude: {info.get('altitude', 0.0):.3f} | "
            f"Speed: {info.get('speed', 0.0):.3f} | "
            f"AttErr: {info.get('attitude_error', 0.0):.3f} | "
            f"Crashed: {info.get('crashed', False)} | "
            f"Landed: {info.get('landed_successfully', False)}"
        )

    print("\n========== SINGLE UAV MISSION EVAL SUMMARY ==========")
    print(
        f"Success Rate: {successes}/{NUM_EVAL_EPISODES} "
        f"({100.0 * successes / NUM_EVAL_EPISODES:.1f}%)"
    )
    print(f"Mean Reward: {np.mean(all_rewards):.3f}")
    print(f"Mean Distance To Target: {np.mean(all_distances):.3f}")
    print(f"Mean XY Error: {np.mean(all_xy_errors):.3f}")
    print(f"Mean Altitude: {np.mean(all_altitudes):.3f}")
    print(f"Mean Speed: {np.mean(all_speeds):.3f}")
    print(f"Mean Attitude Error: {np.mean(all_attitude_errors):.3f}")
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