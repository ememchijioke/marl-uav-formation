import sys
import os
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
RENDER = "human"          # "human" or "headless"
SAVE_VIDEO = False        # True only if rendering with GUI
VIDEO_PATH = "eval_run.mp4"
WIDTH = 1280
HEIGHT = 720
FPS = 20

# Camera mode:
# "fixed"  -> shows the full A to B path clearly
# "follow" -> follows the centroid of the UAV team
CAMERA_MODE = "fixed"


# ----------- CAMERA / VISUALIZATION ----------------
def get_team_centroid(env):
    positions = []
    for i in range(env.num_agents):
        s = env.env._getDroneStateVector(i)
        positions.append(s[0:3])
    return np.mean(np.array(positions, dtype=np.float32), axis=0)


def capture_frame(env):
    if CAMERA_MODE == "follow":
        centroid = get_team_centroid(env).tolist()
        target = centroid
        distance = 4.8
        yaw = 60
        pitch = -30
        fov = 65.0
    else:
        # Fixed wide view showing full A -> B task length
        midpoint = ((env.center_A + env.center_B) / 2.0).tolist()
        target = midpoint
        distance = 7.5
        yaw = 90
        pitch = -35
        fov = 70.0

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=WIDTH / HEIGHT,
        nearVal=0.1,
        farVal=100.0
    )

    _, _, px, _, _ = p.getCameraImage(
        width=WIDTH,
        height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    frame = np.array(px, dtype=np.uint8)[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def draw_marker(position, color, radius=0.08):
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis_id,
        basePosition=position
    )
    return body_id


def draw_reference_path(env):
    """
    Draw clear visual references between A and B:
    - A center
    - midpoint
    - B center
    - connecting line
    - labels
    - triangle slot markers for start and goal formations
    """
    marker_ids = []

    A = env.center_A.tolist()
    B = env.center_B.tolist()
    M = ((env.center_A + env.center_B) / 2.0).tolist()

    # Main center markers
    marker_ids.append(draw_marker(A, [0.0, 0.8, 0.0, 1.0], radius=0.10))   # green
    marker_ids.append(draw_marker(M, [1.0, 1.0, 0.0, 1.0], radius=0.08))   # yellow
    marker_ids.append(draw_marker(B, [1.0, 0.0, 0.0, 1.0], radius=0.10))   # red

    # Main route line
    p.addUserDebugLine(
        lineFromXYZ=A,
        lineToXYZ=B,
        lineColorRGB=[1.0, 1.0, 1.0],
        lineWidth=3.0,
        lifeTime=0
    )

    # Labels
    p.addUserDebugText(
        text="A START",
        textPosition=[A[0], A[1], A[2] + 0.25],
        textColorRGB=[0.0, 1.0, 0.0],
        textSize=1.4,
        lifeTime=0
    )
    p.addUserDebugText(
        text="MID",
        textPosition=[M[0], M[1], M[2] + 0.25],
        textColorRGB=[1.0, 1.0, 0.0],
        textSize=1.3,
        lifeTime=0
    )
    p.addUserDebugText(
        text="B GOAL",
        textPosition=[B[0], B[1], B[2] + 0.25],
        textColorRGB=[1.0, 0.0, 0.0],
        textSize=1.4,
        lifeTime=0
    )

    # Triangle slot markers for start and goal formation
    for i, off in enumerate(env.offsets):
        start_slot = (env.center_A + off).tolist()
        goal_slot = (env.center_B + off).tolist()

        marker_ids.append(draw_marker(start_slot, [0.0, 0.5, 1.0, 0.55], radius=0.05))
        marker_ids.append(draw_marker(goal_slot, [1.0, 0.3, 0.3, 0.55], radius=0.05))

        p.addUserDebugLine(
            lineFromXYZ=start_slot,
            lineToXYZ=goal_slot,
            lineColorRGB=[0.6, 0.6, 0.6],
            lineWidth=1.5,
            lifeTime=0
        )

        p.addUserDebugText(
            text=f"S{i+1}",
            textPosition=[start_slot[0], start_slot[1], start_slot[2] + 0.12],
            textColorRGB=[0.4, 0.8, 1.0],
            textSize=1.0,
            lifeTime=0
        )

        p.addUserDebugText(
            text=f"G{i+1}",
            textPosition=[goal_slot[0], goal_slot[1], goal_slot[2] + 0.12],
            textColorRGB=[1.0, 0.5, 0.5],
            textSize=1.0,
            lifeTime=0
        )

    return marker_ids


def eval_playback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiUAVRealisticEnv(
        render_mode=RENDER,
        num_agents=3,
        max_steps=300,
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
    all_max_errs = []
    all_mean_errs = []
    all_cx = []
    all_goal_dists = []
    all_collision_counts = []
    all_slot_errs = []
    video_frames = []

    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset()

        if RENDER == "human":
            draw_reference_path(env)

        obs_split = np.split(obs, n_agents)

        terminated = False
        truncated = False
        done = False
        ep_rew = 0.0
        info = {}

        for step in range(env.max_steps):
            actions = []

            with torch.no_grad():
                for i in range(n_agents):
                    ot = torch.tensor(
                        obs_split[i],
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)

                    mu = policy(ot)
                    actions.append(mu.squeeze(0).cpu().numpy())   # deterministic

            flat_action = np.concatenate(actions, axis=0)

            next_obs, r, terminated, truncated, info = env.step(flat_action)
            done = bool(terminated or truncated)

            obs_split = np.split(next_obs, n_agents)
            ep_rew += r

            if SAVE_VIDEO and RENDER == "human":
                frame = capture_frame(env)
                video_frames.append(frame)

            if done:
                break

        success = bool(info.get("is_success", False))
        successes += int(success)

        all_max_errs.append(float(info.get("max_formation_error", 0.0)))
        all_mean_errs.append(float(info.get("mean_formation_error", 0.0)))
        all_cx.append(float(info.get("centroid_x", 0.0)))
        all_goal_dists.append(float(info.get("dist_to_goal", 0.0)))
        all_collision_counts.append(float(info.get("collision_count", 0.0)))
        all_slot_errs.append(float(info.get("mean_slot_error", 0.0)))

        print(
            f"Episode {ep+1}/{NUM_EVAL_EPISODES} | "
            f"Reward: {ep_rew:.2f} | "
            f"Success: {success} | "
            f"GoalDist: {info.get('dist_to_goal', 0.0):.3f} | "
            f"MaxErr: {info.get('max_formation_error', 0.0):.3f} | "
            f"MeanErr: {info.get('mean_formation_error', 0.0):.3f} | "
            f"SlotErr: {info.get('mean_slot_error', 0.0):.3f} | "
            f"CentroidX: {info.get('centroid_x', 0.0):.2f} | "
            f"Collisions: {info.get('collision_count', 0)}"
        )

    print("\n========== EVAL SUMMARY ==========")
    print(f"Success Rate: {successes}/{NUM_EVAL_EPISODES} ({100.0 * successes / NUM_EVAL_EPISODES:.1f}%)")
    print(f"Mean Max Formation Error: {np.mean(all_max_errs):.3f}")
    print(f"Mean Formation Error: {np.mean(all_mean_errs):.3f}")
    print(f"Mean Slot Error: {np.mean(all_slot_errs):.3f}")
    print(f"Mean Centroid X: {np.mean(all_cx):.3f}")
    print(f"Mean Distance To Goal: {np.mean(all_goal_dists):.3f}")
    print(f"Mean Collision Count: {np.mean(all_collision_counts):.3f}")

    if SAVE_VIDEO and len(video_frames) > 0:
        out = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT)
        )
        for frame in video_frames:
            out.write(frame)
        out.release()
        print(f"Saved video: {VIDEO_PATH}")

    env.close()


if __name__ == "__main__":
    eval_playback()