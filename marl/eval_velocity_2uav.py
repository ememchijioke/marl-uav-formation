import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env_velocity_2uav import MultiUAVVelocityFormation2UAVEnv


parser = argparse.ArgumentParser()

parser.add_argument("--render", choices=["human", "headless"], default="human")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--sleep", type=float, default=0.02)

parser.add_argument(
    "--model-path",
    type=str,
    default="marl/models/v0_8_3_residual_2uav/best_shared_residual_policy_2uav_v0_8_3.pth",
)

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


def main():
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Check the exact saved model path inside marl/models/."
        )

    env = MultiUAVVelocityFormation2UAVEnv(
        render_mode=args.render,
        num_agents=2,
        max_steps=args.max_steps,
        ctrl_freq=48,
        sim_freq=240,
        target_altitude=1.0,
        target_distance_x=2.4,
        formation_spacing=0.80,
        reference_max_vx=0.24,
        reference_max_vy=0.18,
        reference_max_vz=0.10,
        residual_scale=0.10,
        velocity_lookahead=0.30,
        kp_goal=0.38,
        kp_form=0.60,
        kp_alt=0.45,
        goal_bonus=1200.0,
        alive_reward=0.04,
        progress_weight=38.0,
        centroid_weight=3.3,
        own_goal_weight=1.7,
        formation_weight=24.0,
        spacing_weight=16.0,
        velocity_weight=0.28,
        command_weight=0.10,
        residual_weight=0.14,
        smoothness_weight=0.28,
        attitude_weight=0.90,
        collision_penalty=2600.0,
        crash_penalty=3000.0,
        boundary_penalty=1000.0,
        success_centroid_tol=0.30,
        success_formation_tol=0.20,
        success_spacing_tol=0.16,
        min_separation=0.35,
        neighbor_dropout_prob=0.0,
        use_neighbor_dropout=False,
    )

    n_agents = env.num_agents
    obs_dim_total = env.observation_space.shape[0]
    act_dim_total = env.action_space.shape[0]

    obs_dim = obs_dim_total // n_agents
    act_dim = act_dim_total // n_agents

    print(f"Using device: {device}")
    print(f"Number of agents: {n_agents}")
    print(f"Observation dim per agent: {obs_dim}")
    print(f"Action dim per agent: {act_dim}")
    print(f"Loading model: {args.model_path}")

    policy = SharedVelocityPolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    all_successes = []
    all_centroid_errors = []
    all_formation_errors = []
    all_spacing_errors = []
    all_collisions = []
    all_crashes = []

    for ep in range(1, args.episodes + 1):
        flat_obs, _ = env.reset()
        obs_split = split_obs(flat_obs, n_agents)

        ep_reward = 0.0
        last_info = {}
        success = False

        print(f"\n=== Evaluation Episode {ep} ===")

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

            if step % 25 == 0 or info.get("is_success", False):
                print(
                    f"Step {step:4d} | "
                    f"Reward: {ep_reward:9.2f} | "
                    f"Success: {bool(info.get('is_success', False))} | "
                    f"CentErr: {safe_float(info, 'centroid_error'):.3f} | "
                    f"FormErr: {safe_float(info, 'formation_error'):.3f} | "
                    f"SpacingErr: {safe_float(info, 'spacing_error'):.3f} | "
                    f"PairDist: {safe_float(info, 'pair_distance'):.3f} | "
                    f"Speed: {safe_float(info, 'mean_speed'):.3f} | "
                    f"CmdSpeed: {safe_float(info, 'mean_cmd_speed'):.3f} | "
                    f"Residual: {safe_float(info, 'mean_residual'):.3f} | "
                    f"Collisions: {int(safe_float(info, 'collision_count'))} | "
                    f"Crashed: {bool(info.get('crashed', False))} | "
                    f"Safe: {bool(info.get('formation_safe', False))}"
                )

            if args.render == "human":
                time.sleep(args.sleep)

            if terminated or truncated:
                success = bool(info.get("is_success", False))
                break

        all_successes.append(float(success))
        all_centroid_errors.append(safe_float(last_info, "centroid_error"))
        all_formation_errors.append(safe_float(last_info, "formation_error"))
        all_spacing_errors.append(safe_float(last_info, "spacing_error"))
        all_collisions.append(safe_float(last_info, "collision_count"))
        all_crashes.append(float(bool(last_info.get("crashed", False))))

        print(
            f"\nEpisode {ep} finished | "
            f"Reward: {ep_reward:.2f} | "
            f"Success: {success} | "
            f"CentErr: {safe_float(last_info, 'centroid_error'):.3f} | "
            f"FormErr: {safe_float(last_info, 'formation_error'):.3f} | "
            f"SpacingErr: {safe_float(last_info, 'spacing_error'):.3f} | "
            f"PairDist: {safe_float(last_info, 'pair_distance'):.3f} | "
            f"Collisions: {int(safe_float(last_info, 'collision_count'))} | "
            f"Crashed: {bool(last_info.get('crashed', False))}"
        )

    env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Success rate: {np.mean(all_successes) * 100:.1f}%")
    print(f"Mean centroid error: {np.mean(all_centroid_errors):.3f} m")
    print(f"Mean formation error: {np.mean(all_formation_errors):.3f} m")
    print(f"Mean spacing error: {np.mean(all_spacing_errors):.3f} m")
    print(f"Mean collisions: {np.mean(all_collisions):.3f}")
    print(f"Crash rate: {np.mean(all_crashes) * 100:.1f}%")


if __name__ == "__main__":
    main()
