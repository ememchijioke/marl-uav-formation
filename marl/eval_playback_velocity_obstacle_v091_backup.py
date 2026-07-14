import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env_velocity_obstacle import MultiUAVVelocityObstacleFormationEnv


parser = argparse.ArgumentParser()

parser.add_argument("--render", choices=["human", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=1200)
parser.add_argument("--sleep", type=float, default=0.02)

parser.add_argument(
    "--model-path",
    type=str,
    default="checkpoints/v0_9_1_obstacle_curriculum_velocity_3uav/shared_velocity_policy.pth",
    help="Path to the trained obstacle-aware shared velocity policy checkpoint.",
)

parser.add_argument(
    "--obstacle-layout",
    type=str,
    default="single_center",
    choices=["none", "easy_offset", "side_column", "single_center", "offset_gate"],
)

parser.add_argument("--save-video", action="store_true")
parser.add_argument("--video-path", type=str, default="eval_obstacle_curriculum_velocity_3uav.mp4")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=20)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedVelocityPolicyNet(nn.Module):
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


def split_obs(flat_obs, n_agents):
    return np.split(flat_obs, n_agents)


def safe_float(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def make_env(render_mode):
    use_obstacles = args.obstacle_layout != "none"

    return MultiUAVVelocityObstacleFormationEnv(
        render_mode=render_mode,
        num_agents=3,
        max_steps=args.max_steps,
        ctrl_freq=48,
        sim_freq=240,
        mission_stage=3,
        target_altitude=1.0,
        start_x=0.0,
        goal_x=7.0,
        triangle_side=1.20,
        reference_max_vx=0.25,
        reference_max_vy=0.24,
        reference_max_vz=0.14,
        correction_scale=0.070,
        velocity_lookahead=0.30,
        kp_center=0.38,
        kp_form=0.72,
        kp_alt=0.45,
        min_separation=0.30,
        min_obstacle_clearance=0.32,
        obstacle_influence_radius=1.20,
        obstacle_repulsion_gain=0.30,
        safety_filter_gain=0.30,
        neighbor_dropout_prob=0.0,
        use_neighbor_dropout=False,
        use_obstacles=use_obstacles,
        obstacle_layout=args.obstacle_layout,
    )


def get_drone_positions(env):
    positions = []

    for i in range(env.num_agents):
        state = env.env._getDroneStateVector(i)
        positions.append(state[0:3])

    return np.asarray(positions, dtype=np.float32)


def reset_camera(env):
    if args.render != "human":
        return

    try:
        import pybullet as p

        positions = get_drone_positions(env)
        centroid = np.mean(positions, axis=0)

        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[
                float(centroid[0]),
                float(centroid[1]),
                float(max(centroid[2], 0.8)),
            ],
        )

    except Exception:
        pass


def draw_demo_guides(env):
    if args.render != "human":
        return

    try:
        import pybullet as p

        start_center = env.start_center
        goal_center = env.goal_center
        landing_center = env.landing_center
        offsets = env.triangle_offsets

        start_targets = start_center[None, :] + offsets
        goal_targets = goal_center[None, :] + offsets
        landing_targets = landing_center[None, :] + offsets

        p.addUserDebugLine(
            start_center.tolist(),
            goal_center.tolist(),
            [1.0, 1.0, 1.0],
            lineWidth=3.0,
            lifeTime=0,
        )

        p.addUserDebugLine(
            goal_center.tolist(),
            landing_center.tolist(),
            [1.0, 0.0, 0.0],
            lineWidth=3.0,
            lifeTime=0,
        )

        def draw_triangle(points, color):
            p.addUserDebugLine(points[0].tolist(), points[1].tolist(), color, 2.0, 0)
            p.addUserDebugLine(points[0].tolist(), points[2].tolist(), color, 2.0, 0)
            p.addUserDebugLine(points[1].tolist(), points[2].tolist(), color, 2.0, 0)

        draw_triangle(start_targets, [0.0, 0.7, 1.0])
        draw_triangle(goal_targets, [0.0, 1.0, 0.4])
        draw_triangle(landing_targets, [1.0, 0.2, 0.2])

        if env.use_obstacles and len(env.obstacles) > 0:
            for obs in env.obstacles:
                center = obs["center"]
                influence = float(env.obstacle_influence_radius)

                p.addUserDebugText(
                    "OBSTACLE",
                    [float(center[0]), float(center[1]), 1.75],
                    [1.0, 0.25, 0.1],
                    textSize=1.1,
                    lifeTime=0,
                )

                circle_pts = []

                for k in range(48):
                    theta = 2.0 * np.pi * k / 48
                    x = center[0] + influence * np.cos(theta)
                    y = center[1] + influence * np.sin(theta)
                    z = 0.04
                    circle_pts.append([float(x), float(y), float(z)])

                for k in range(48):
                    p.addUserDebugLine(
                        circle_pts[k],
                        circle_pts[(k + 1) % 48],
                        [1.0, 0.6, 0.0],
                        lineWidth=1.0,
                        lifeTime=0,
                    )

        p.addUserDebugText(
            "v0.9.1: OBSTACLE-CURRICULUM TRIANGLE POLICY",
            [0.0, -1.8, 1.75],
            [1.0, 1.0, 1.0],
            textSize=1.1,
            lifeTime=0,
        )

    except Exception:
        pass


def capture_frame(env):
    import pybullet as p
    import cv2

    positions = get_drone_positions(env)
    centroid = np.mean(positions, axis=0)

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[
            float(centroid[0]),
            float(centroid[1]),
            float(max(centroid[2], 0.8)),
        ],
        distance=5.0,
        yaw=45,
        pitch=-25,
        roll=0,
        upAxisIndex=2,
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=args.width / args.height,
        nearVal=0.1,
        farVal=100.0,
    )

    _, _, px, _, _ = p.getCameraImage(
        width=args.width,
        height=args.height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    frame = np.asarray(px, dtype=np.uint8)[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def save_video(frames, path):
    if len(frames) == 0:
        print("[WARN] No frames captured. Video not saved.")
        return

    import cv2

    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )

    for frame in frames:
        out.write(frame)

    out.release()

    print(f"\nSaved video: {path}")
    print(f"Frames: {len(frames)}")
    print(f"Approx duration: {len(frames) / args.fps:.2f} seconds")


def main():
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Check your checkpoint path or train the obstacle-curriculum model first."
        )

    env = make_env(args.render)

    n_agents = env.num_agents
    obs_dim_total = env.observation_space.shape[0]
    act_dim_total = env.action_space.shape[0]

    obs_dim = obs_dim_total // n_agents
    act_dim = act_dim_total // n_agents

    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Render mode: {args.render}")
    print(f"Obstacle layout: {args.obstacle_layout}")
    print(f"Number of agents: {n_agents}")
    print(f"Observation dim per agent: {obs_dim}")
    print(f"Action dim per agent: {act_dim}")
    print("Action meaning: learned velocity correction [rx, ry, rz]")
    print("Mission: takeoff -> hover -> obstacle-aware triangle flight -> goal hover -> landing")
    print("Distance: start x=0.0, goal x=7.0")

    policy = SharedVelocityPolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    successes = []
    final_phases = []
    center_errors = []
    formation_errors = []
    spacing_errors = []
    collisions = []
    crashes = []
    min_seps = []
    rewards_total = []

    min_obstacle_margins = []
    obstacle_collisions = []
    obstacle_near_misses = []
    obstacle_clears = []

    all_frames = []

    for ep in range(1, args.episodes + 1):
        flat_obs, _ = env.reset()
        obs_split = split_obs(flat_obs, n_agents)

        ep_reward = 0.0
        last_info = {}

        if args.render == "human":
            reset_camera(env)
            draw_demo_guides(env)

        print(f"\n=== Evaluation Episode {ep} ===")

        episode_frames = []

        for step in range(1, args.max_steps + 1):
            actions = []

            for i in range(n_agents):
                obs_i = obs_split[i]

                obs_t = torch.tensor(
                    obs_i,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    action = policy(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                actions.append(action_np)

            flat_action = np.concatenate(actions, axis=0)

            flat_obs, reward, terminated, truncated, info = env.step(flat_action)
            obs_split = split_obs(flat_obs, n_agents)

            ep_reward += float(reward)
            last_info = info

            if args.render == "human":
                reset_camera(env)

                if args.save_video:
                    try:
                        frame = capture_frame(env)
                        episode_frames.append(frame)
                    except Exception as e:
                        print(f"[WARN] Frame capture failed: {e}")

                time.sleep(args.sleep)

            if step % 50 == 0 or bool(info.get("is_success", False)):
                print(
                    f"Step {step:4d} | "
                    f"Success: {bool(info.get('is_success', False))} | "
                    f"Phase: {int(info.get('phase', 0))}({info.get('phase_name', '')}) | "
                    f"CenterErr: {safe_float(info, 'center_error'):.3f} | "
                    f"FormErr: {safe_float(info, 'formation_error'):.3f} | "
                    f"SpacingErr: {safe_float(info, 'spacing_error'):.3f} | "
                    f"Speed: {safe_float(info, 'mean_speed'):.3f} | "
                    f"CmdSpeed: {safe_float(info, 'mean_cmd_speed'):.3f} | "
                    f"Correction: {safe_float(info, 'mean_correction'):.3f} | "
                    f"MinSep: {safe_float(info, 'min_pair_dist'):.3f} | "
                    f"ObsMargin: {safe_float(info, 'min_obstacle_margin'):.3f} | "
                    f"ObsColl: {int(safe_float(info, 'obstacle_collision_count'))} | "
                    f"NearMiss: {int(safe_float(info, 'obstacle_near_miss_count'))} | "
                    f"Collisions: {int(safe_float(info, 'collision_count'))} | "
                    f"Crashed: {bool(info.get('crashed', False))} | "
                    f"Safe: {bool(info.get('formation_safe', False))}"
                )

            if terminated or truncated:
                break

        if args.save_video and len(episode_frames) > len(all_frames):
            all_frames = episode_frames.copy()

        success = bool(last_info.get("is_success", False))

        successes.append(float(success))
        final_phases.append(float(last_info.get("phase", 0)))
        center_errors.append(safe_float(last_info, "center_error"))
        formation_errors.append(safe_float(last_info, "formation_error"))
        spacing_errors.append(safe_float(last_info, "spacing_error"))
        collisions.append(safe_float(last_info, "collision_count"))
        crashes.append(float(bool(last_info.get("crashed", False))))
        min_seps.append(safe_float(last_info, "min_pair_dist"))
        rewards_total.append(ep_reward)

        min_obstacle_margins.append(safe_float(last_info, "min_obstacle_margin"))
        obstacle_collisions.append(safe_float(last_info, "obstacle_collision_count"))
        obstacle_near_misses.append(safe_float(last_info, "obstacle_near_miss_count"))
        obstacle_clears.append(float(bool(last_info.get("obstacle_clear", False))))

        print(
            f"\nEpisode {ep} finished | "
            f"Reward: {ep_reward:.2f} | "
            f"Success: {success} | "
            f"FinalPhase: {int(last_info.get('phase', 0))}({last_info.get('phase_name', '')}) | "
            f"CenterErr: {safe_float(last_info, 'center_error'):.3f} | "
            f"FormErr: {safe_float(last_info, 'formation_error'):.3f} | "
            f"SpacingErr: {safe_float(last_info, 'spacing_error'):.3f} | "
            f"Speed: {safe_float(last_info, 'mean_speed'):.3f} | "
            f"MinSep: {safe_float(last_info, 'min_pair_dist'):.3f} | "
            f"ObsMargin: {safe_float(last_info, 'min_obstacle_margin'):.3f} | "
            f"ObsColl: {int(safe_float(last_info, 'obstacle_collision_count'))} | "
            f"NearMiss: {int(safe_float(last_info, 'obstacle_near_miss_count'))} | "
            f"Collisions: {int(safe_float(last_info, 'collision_count'))} | "
            f"Crashed: {bool(last_info.get('crashed', False))}"
        )

    env.close()

    if args.save_video:
        save_video(all_frames, args.video_path)

    print("\n========== OBSTACLE-CURRICULUM EVALUATION SUMMARY ==========")
    print(f"Model path: {args.model_path}")
    print(f"Obstacle layout: {args.obstacle_layout}")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {np.mean(successes) * 100:.1f}%")
    print(f"Mean reward: {np.mean(rewards_total):.2f}")
    print(f"Mean final phase: {np.mean(final_phases):.2f}")
    print(f"Mean center error: {np.mean(center_errors):.3f} m")
    print(f"Mean formation error: {np.mean(formation_errors):.3f} m")
    print(f"Mean spacing error: {np.mean(spacing_errors):.3f} m")
    print(f"Mean min separation: {np.mean(min_seps):.3f} m")
    print(f"Mean collisions: {np.mean(collisions):.3f}")
    print(f"Mean min obstacle margin: {np.mean(min_obstacle_margins):.3f} m")
    print(f"Mean obstacle collisions: {np.mean(obstacle_collisions):.3f}")
    print(f"Mean obstacle near misses: {np.mean(obstacle_near_misses):.3f}")
    print(f"Obstacle clear rate: {np.mean(obstacle_clears) * 100:.1f}%")
    print(f"Crash rate: {np.mean(crashes) * 100:.1f}%")


if __name__ == "__main__":
    main()