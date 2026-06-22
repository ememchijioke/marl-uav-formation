import os
import sys
import csv
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env_velocity_obstacle import MultiUAVVelocityObstacleFormationEnv


parser = argparse.ArgumentParser()

parser.add_argument("--render", choices=["human", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=4000)
parser.add_argument("--max-steps", type=int, default=1200)

parser.add_argument("--lr-policy", type=float, default=2.0e-4)
parser.add_argument("--lr-value", type=float, default=8.0e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.008)
parser.add_argument("--vf-coef", type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--explore-std", type=float, default=0.02)
parser.add_argument("--update-epochs", type=int, default=8)

parser.add_argument("--resume", action="store_true")
parser.add_argument("--use-wandb", action="store_true")
parser.add_argument("--init-from-v085", action="store_true")

parser.add_argument(
    "--v085-policy-path",
    type=str,
    default="checkpoints/v0_8_5_triangle_velocity_3uav/shared_velocity_policy.pth",
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


class CentralValueNet(nn.Module):
    def __init__(self, obs_dim_total):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim_total),
            nn.Linear(obs_dim_total, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_obs(flat_obs, n_agents):
    return np.split(flat_obs, n_agents)


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = []
    gae = 0.0
    values_ext = values + [next_value]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def build_env(stage_cfg):
    return MultiUAVVelocityObstacleFormationEnv(
        render_mode=args.render,
        num_agents=3,
        max_steps=args.max_steps,
        ctrl_freq=48,
        sim_freq=240,
        mission_stage=stage_cfg["mission_stage"],
        target_altitude=1.0,
        start_x=0.0,
        goal_x=7.0,
        triangle_side=stage_cfg["triangle_side"],
        reference_max_vx=stage_cfg["reference_max_vx"],
        reference_max_vy=stage_cfg["reference_max_vy"],
        reference_max_vz=stage_cfg["reference_max_vz"],
        correction_scale=stage_cfg["correction_scale"],
        velocity_lookahead=stage_cfg["velocity_lookahead"],
        kp_center=stage_cfg["kp_center"],
        kp_form=stage_cfg["kp_form"],
        kp_alt=stage_cfg["kp_alt"],
        goal_bonus=stage_cfg["goal_bonus"],
        alive_reward=stage_cfg["alive_reward"],
        phase_bonus=stage_cfg["phase_bonus"],
        progress_weight=stage_cfg["progress_weight"],
        center_weight=stage_cfg["center_weight"],
        assigned_target_weight=stage_cfg["assigned_target_weight"],
        formation_weight=stage_cfg["formation_weight"],
        spacing_weight=stage_cfg["spacing_weight"],
        velocity_weight=stage_cfg["velocity_weight"],
        command_weight=stage_cfg["command_weight"],
        correction_weight=stage_cfg["correction_weight"],
        smoothness_weight=stage_cfg["smoothness_weight"],
        attitude_weight=stage_cfg["attitude_weight"],
        collision_penalty=stage_cfg["collision_penalty"],
        crash_penalty=stage_cfg["crash_penalty"],
        boundary_penalty=stage_cfg["boundary_penalty"],
        obstacle_collision_penalty=stage_cfg["obstacle_collision_penalty"],
        obstacle_near_penalty_weight=stage_cfg["obstacle_near_penalty_weight"],
        obstacle_clearance_bonus=stage_cfg["obstacle_clearance_bonus"],
        formation_near_obstacle_weight=stage_cfg["formation_near_obstacle_weight"],
        min_separation=stage_cfg["min_separation"],
        min_obstacle_clearance=stage_cfg["min_obstacle_clearance"],
        obstacle_influence_radius=stage_cfg["obstacle_influence_radius"],
        obstacle_repulsion_gain=stage_cfg["obstacle_repulsion_gain"],
        safety_filter_gain=stage_cfg["safety_filter_gain"],
        neighbor_dropout_prob=stage_cfg["neighbor_dropout_prob"],
        use_neighbor_dropout=True,
        use_obstacles=stage_cfg["use_obstacles"],
        obstacle_layout=stage_cfg["obstacle_layout"],
    )


def save_checkpoint(policy, value, opt_policy, opt_value, episode, stage_idx, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "shared_velocity_policy.pth"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "central_value.pth"))
    torch.save(opt_policy.state_dict(), os.path.join(ckpt_dir, "opt_policy.pth"))
    torch.save(opt_value.state_dict(), os.path.join(ckpt_dir, "opt_value.pth"))

    with open(os.path.join(ckpt_dir, "last_episode.txt"), "w") as f:
        f.write(str(episode))

    with open(os.path.join(ckpt_dir, "stage.txt"), "w") as f:
        f.write(str(stage_idx))


def load_checkpoint(policy, value, opt_policy, opt_value, ckpt_dir):
    start_episode = 1
    stage_idx = 0

    ep_path = os.path.join(ckpt_dir, "last_episode.txt")
    st_path = os.path.join(ckpt_dir, "stage.txt")

    if os.path.exists(ep_path):
        with open(ep_path, "r") as f:
            start_episode = int(f.read().strip()) + 1

    if os.path.exists(st_path):
        with open(st_path, "r") as f:
            stage_idx = int(f.read().strip())

    paths = {
        "policy": os.path.join(ckpt_dir, "shared_velocity_policy.pth"),
        "value": os.path.join(ckpt_dir, "central_value.pth"),
        "opt_policy": os.path.join(ckpt_dir, "opt_policy.pth"),
        "opt_value": os.path.join(ckpt_dir, "opt_value.pth"),
    }

    if os.path.exists(paths["policy"]):
        policy.load_state_dict(torch.load(paths["policy"], map_location=device))

    if os.path.exists(paths["value"]):
        value.load_state_dict(torch.load(paths["value"], map_location=device))

    if os.path.exists(paths["opt_policy"]):
        opt_policy.load_state_dict(torch.load(paths["opt_policy"], map_location=device))

    if os.path.exists(paths["opt_value"]):
        opt_value.load_state_dict(torch.load(paths["opt_value"], map_location=device))

    return start_episode, stage_idx


def partial_init_policy_from_v085(policy, old_path):
    if not os.path.exists(old_path):
        print(f"[WARN] v0.8.5 policy not found: {old_path}")
        return

    try:
        old_state = torch.load(old_path, map_location=device)
        new_state = policy.state_dict()

        if "net.1.weight" in old_state and "net.1.weight" in new_state:
            old_w = old_state["net.1.weight"]
            new_w = new_state["net.1.weight"]

            cols = min(old_w.shape[1], new_w.shape[1])
            new_w[:, :cols] = old_w[:, :cols]
            new_state["net.1.weight"] = new_w

        for key in ["net.1.bias", "net.3.weight", "net.3.bias", "net.5.weight", "net.5.bias"]:
            if key in old_state and key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]

        policy.load_state_dict(new_state)

        print(f"[INFO] Partially initialized obstacle policy from: {old_path}")
        print("[INFO] Copied compatible layers from v0.8.5. New obstacle observation weights remain trainable.")

    except Exception as e:
        print("[WARN] Partial initialization failed. Continuing from random initialization.")
        print(f"[WARN] {e}")


def safe_float(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def train():
    curriculum = [
        {
            "name": "no_obstacle_bootstrap",
            "mission_stage": 3,
            "use_obstacles": False,
            "obstacle_layout": "none",
            "triangle_side": 1.20,
            "reference_max_vx": 0.28,
            "reference_max_vy": 0.20,
            "reference_max_vz": 0.14,
            "correction_scale": 0.055,
            "velocity_lookahead": 0.30,
            "kp_center": 0.36,
            "kp_form": 0.72,
            "kp_alt": 0.45,
            "goal_bonus": 1500.0,
            "alive_reward": 0.04,
            "phase_bonus": 230.0,
            "progress_weight": 45.0,
            "center_weight": 3.5,
            "assigned_target_weight": 1.7,
            "formation_weight": 30.0,
            "spacing_weight": 18.0,
            "velocity_weight": 0.26,
            "command_weight": 0.09,
            "correction_weight": 0.13,
            "smoothness_weight": 0.26,
            "attitude_weight": 0.85,
            "collision_penalty": 3500.0,
            "crash_penalty": 4000.0,
            "boundary_penalty": 1500.0,
            "obstacle_collision_penalty": 2500.0,
            "obstacle_near_penalty_weight": 0.0,
            "obstacle_clearance_bonus": 0.0,
            "formation_near_obstacle_weight": 0.0,
            "min_separation": 0.30,
            "min_obstacle_clearance": 0.30,
            "obstacle_influence_radius": 1.05,
            "obstacle_repulsion_gain": 0.00,
            "safety_filter_gain": 0.00,
            "neighbor_dropout_prob": 0.01,
        },
        {
            "name": "easy_offset_obstacle",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "easy_offset",
            "triangle_side": 1.20,
            "reference_max_vx": 0.27,
            "reference_max_vy": 0.22,
            "reference_max_vz": 0.14,
            "correction_scale": 0.060,
            "velocity_lookahead": 0.30,
            "kp_center": 0.36,
            "kp_form": 0.72,
            "kp_alt": 0.45,
            "goal_bonus": 1650.0,
            "alive_reward": 0.04,
            "phase_bonus": 240.0,
            "progress_weight": 46.0,
            "center_weight": 3.5,
            "assigned_target_weight": 1.8,
            "formation_weight": 31.0,
            "spacing_weight": 18.0,
            "velocity_weight": 0.27,
            "command_weight": 0.09,
            "correction_weight": 0.13,
            "smoothness_weight": 0.27,
            "attitude_weight": 0.88,
            "collision_penalty": 3500.0,
            "crash_penalty": 4000.0,
            "boundary_penalty": 1500.0,
            "obstacle_collision_penalty": 3000.0,
            "obstacle_near_penalty_weight": 10.0,
            "obstacle_clearance_bonus": 8.0,
            "formation_near_obstacle_weight": 6.0,
            "min_separation": 0.30,
            "min_obstacle_clearance": 0.28,
            "obstacle_influence_radius": 1.00,
            "obstacle_repulsion_gain": 0.18,
            "safety_filter_gain": 0.20,
            "neighbor_dropout_prob": 0.012,
        },
        {
            "name": "side_column_obstacle",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "side_column",
            "triangle_side": 1.20,
            "reference_max_vx": 0.26,
            "reference_max_vy": 0.23,
            "reference_max_vz": 0.14,
            "correction_scale": 0.065,
            "velocity_lookahead": 0.30,
            "kp_center": 0.37,
            "kp_form": 0.72,
            "kp_alt": 0.45,
            "goal_bonus": 1750.0,
            "alive_reward": 0.04,
            "phase_bonus": 250.0,
            "progress_weight": 47.0,
            "center_weight": 3.5,
            "assigned_target_weight": 1.8,
            "formation_weight": 32.0,
            "spacing_weight": 19.0,
            "velocity_weight": 0.28,
            "command_weight": 0.10,
            "correction_weight": 0.14,
            "smoothness_weight": 0.28,
            "attitude_weight": 0.90,
            "collision_penalty": 3600.0,
            "crash_penalty": 4200.0,
            "boundary_penalty": 1500.0,
            "obstacle_collision_penalty": 3500.0,
            "obstacle_near_penalty_weight": 16.0,
            "obstacle_clearance_bonus": 10.0,
            "formation_near_obstacle_weight": 8.0,
            "min_separation": 0.30,
            "min_obstacle_clearance": 0.30,
            "obstacle_influence_radius": 1.10,
            "obstacle_repulsion_gain": 0.24,
            "safety_filter_gain": 0.25,
            "neighbor_dropout_prob": 0.015,
        },
        {
            "name": "single_center_obstacle",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "single_center",
            "triangle_side": 1.20,
            "reference_max_vx": 0.25,
            "reference_max_vy": 0.24,
            "reference_max_vz": 0.14,
            "correction_scale": 0.070,
            "velocity_lookahead": 0.30,
            "kp_center": 0.38,
            "kp_form": 0.72,
            "kp_alt": 0.45,
            "goal_bonus": 1900.0,
            "alive_reward": 0.04,
            "phase_bonus": 260.0,
            "progress_weight": 48.0,
            "center_weight": 3.5,
            "assigned_target_weight": 1.8,
            "formation_weight": 34.0,
            "spacing_weight": 20.0,
            "velocity_weight": 0.28,
            "command_weight": 0.10,
            "correction_weight": 0.14,
            "smoothness_weight": 0.28,
            "attitude_weight": 0.90,
            "collision_penalty": 3600.0,
            "crash_penalty": 4200.0,
            "boundary_penalty": 1500.0,
            "obstacle_collision_penalty": 4000.0,
            "obstacle_near_penalty_weight": 20.0,
            "obstacle_clearance_bonus": 12.0,
            "formation_near_obstacle_weight": 10.0,
            "min_separation": 0.30,
            "min_obstacle_clearance": 0.32,
            "obstacle_influence_radius": 1.20,
            "obstacle_repulsion_gain": 0.30,
            "safety_filter_gain": 0.30,
            "neighbor_dropout_prob": 0.018,
        },
        {
            "name": "offset_gate_obstacle",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "offset_gate",
            "triangle_side": 1.20,
            "reference_max_vx": 0.24,
            "reference_max_vy": 0.25,
            "reference_max_vz": 0.14,
            "correction_scale": 0.075,
            "velocity_lookahead": 0.30,
            "kp_center": 0.38,
            "kp_form": 0.72,
            "kp_alt": 0.45,
            "goal_bonus": 2100.0,
            "alive_reward": 0.04,
            "phase_bonus": 280.0,
            "progress_weight": 50.0,
            "center_weight": 3.5,
            "assigned_target_weight": 1.8,
            "formation_weight": 36.0,
            "spacing_weight": 21.0,
            "velocity_weight": 0.30,
            "command_weight": 0.11,
            "correction_weight": 0.15,
            "smoothness_weight": 0.30,
            "attitude_weight": 0.95,
            "collision_penalty": 3800.0,
            "crash_penalty": 4400.0,
            "boundary_penalty": 1600.0,
            "obstacle_collision_penalty": 4500.0,
            "obstacle_near_penalty_weight": 24.0,
            "obstacle_clearance_bonus": 14.0,
            "formation_near_obstacle_weight": 12.0,
            "min_separation": 0.30,
            "min_obstacle_clearance": 0.32,
            "obstacle_influence_radius": 1.25,
            "obstacle_repulsion_gain": 0.34,
            "safety_filter_gain": 0.34,
            "neighbor_dropout_prob": 0.02,
        },
    ]

    rolling_window = 30
    stage_success_thresholds = [0.55, 0.50, 0.45, 0.40, 0.35]
    recent_successes = []
    curr_stage = 0

    ckpt_dir = "checkpoints/v0_9_1_obstacle_curriculum_velocity_3uav"
    model_dir = "marl/models/v0_9_1_obstacle_curriculum_velocity_3uav"
    log_dir = "results/logs"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    wandb_enabled = False

    if args.use_wandb:
        try:
            import wandb

            wandb.init(
                project="marl-uav-v0-9-1-obstacle-curriculum-velocity-3uav",
                name=f"obstacle-curriculum-velocity-{args.render}",
                config=vars(args),
                resume="allow",
            )
            wandb_enabled = True
        except Exception as e:
            print("[WARN] W&B failed. Continuing without W&B.")
            print(f"[WARN] {e}")

    stage_cfg = curriculum[curr_stage]
    env = build_env(stage_cfg)

    n_agents = env.num_agents
    obs_dim_total = env.observation_space.shape[0]
    act_dim_total = env.action_space.shape[0]

    obs_dim = obs_dim_total // n_agents
    act_dim = act_dim_total // n_agents

    print(f"Using device: {device}")
    print(f"Number of agents: {n_agents}")
    print(f"Observation dim per agent: {obs_dim}")
    print(f"Action dim per agent: {act_dim}")
    print("Action meaning: learned velocity correction [rx, ry, rz]")
    print("Mission: takeoff -> hover -> obstacle-aware triangle flight -> goal hover -> landing")
    print("Curriculum: no obstacle -> easy offset -> side column -> center obstacle -> gate obstacle")
    print("Distance: start x=0.0, goal x=7.0")

    policy = SharedVelocityPolicyNet(obs_dim, act_dim).to(device)
    value = CentralValueNet(obs_dim_total).to(device)

    if args.init_from_v085 and not args.resume:
        partial_init_policy_from_v085(policy, args.v085_policy_path)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_value = torch.optim.Adam(value.parameters(), lr=args.lr_value)

    start_episode = 1
    best_score = -np.inf
    patience = 0
    patience_limit = 1800

    if args.resume:
        start_episode, curr_stage = load_checkpoint(
            policy=policy,
            value=value,
            opt_policy=opt_policy,
            opt_value=opt_value,
            ckpt_dir=ckpt_dir,
        )

        curr_stage = min(curr_stage, len(curriculum) - 1)
        stage_cfg = curriculum[curr_stage]

        env.close()
        env = build_env(stage_cfg)

        print(f"Resumed from episode {start_episode}, stage {curr_stage + 1}: {stage_cfg['name']}")

    csv_path = os.path.join(log_dir, "training_logs_v0_9_1_obstacle_curriculum_velocity_3uav.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "timestamp",
                    "episode",
                    "stage",
                    "stage_name",
                    "reward",
                    "success",
                    "phase",
                    "phase_name",
                    "center_error",
                    "mean_dist_to_target",
                    "formation_error",
                    "spacing_error",
                    "mean_speed",
                    "mean_ref_speed",
                    "mean_cmd_speed",
                    "mean_correction",
                    "action_smoothness",
                    "collision_count",
                    "min_pair_dist",
                    "min_obstacle_margin",
                    "mean_obstacle_margin",
                    "obstacle_collision_count",
                    "obstacle_near_miss_count",
                    "obstacle_clear",
                    "crashed",
                    "progress",
                    "formation_safe",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                ]
            )

    fixed_std = torch.full((act_dim,), args.explore_std, device=device)

    for ep in range(start_episode, args.episodes + 1):
        flat_obs, _ = env.reset()
        obs_split = split_obs(flat_obs, n_agents)

        ep_reward = 0.0
        success = False
        last_info = {}

        buf_global_obs = []
        buf_local_obs = []
        buf_actions = []
        buf_logps = []
        buf_values = []
        rewards = []
        dones = []

        entropy_coef = args.ent_coef * max(0.20, 1.0 - ep / (0.85 * args.episodes))

        for _ in range(args.max_steps):
            global_obs_t = torch.tensor(
                flat_obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                value_t = value(global_obs_t).item()

            local_obs_step = []
            actions_step = []
            logps_step = []

            for i in range(n_agents):
                obs_i = obs_split[i]

                obs_t = torch.tensor(
                    obs_i,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    mu = policy(obs_t)
                    dist = Normal(mu, fixed_std)
                    action = dist.sample()
                    action = torch.clamp(action, -1.0, 1.0)
                    logp = dist.log_prob(action).sum(dim=-1)

                local_obs_step.append(obs_i.copy())
                actions_step.append(action.squeeze(0).cpu().numpy())
                logps_step.append(float(logp.item()))

            flat_action = np.concatenate(actions_step, axis=0)

            next_flat_obs, reward, terminated, truncated, info = env.step(flat_action)
            done = bool(terminated or truncated)

            buf_global_obs.append(flat_obs.copy())
            buf_local_obs.append(np.array(local_obs_step, dtype=np.float32))
            buf_actions.append(np.array(actions_step, dtype=np.float32))
            buf_logps.append(np.array(logps_step, dtype=np.float32))
            buf_values.append(float(value_t))

            rewards.append(float(reward))
            dones.append(bool(done))

            ep_reward += float(reward)
            flat_obs = next_flat_obs
            obs_split = split_obs(flat_obs, n_agents)
            last_info = info

            if done:
                success = bool(info.get("is_success", False))
                break

        with torch.no_grad():
            next_global_obs_t = torch.tensor(
                flat_obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            next_value = value(next_global_obs_t).item()

        advantages_list, returns_list = compute_gae(
            rewards=rewards,
            dones=dones,
            values=buf_values,
            next_value=next_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)
        returns = torch.tensor(returns_list, dtype=torch.float32, device=device)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        global_obs_batch = torch.tensor(
            np.array(buf_global_obs),
            dtype=torch.float32,
            device=device,
        )

        local_obs_batch = torch.tensor(
            np.array(buf_local_obs),
            dtype=torch.float32,
            device=device,
        )

        action_batch = torch.tensor(
            np.array(buf_actions),
            dtype=torch.float32,
            device=device,
        )

        old_logp_batch = torch.tensor(
            np.array(buf_logps),
            dtype=torch.float32,
            device=device,
        )

        t_steps = local_obs_batch.shape[0]

        local_obs_actor = local_obs_batch.reshape(t_steps * n_agents, obs_dim)
        action_actor = action_batch.reshape(t_steps * n_agents, act_dim)
        old_logp_actor = old_logp_batch.reshape(t_steps * n_agents)
        adv_actor = advantages.unsqueeze(1).repeat(1, n_agents).reshape(t_steps * n_agents)

        policy_losses = []
        value_losses = []
        entropy_terms = []

        for _ in range(args.update_epochs):
            mu = policy(local_obs_actor)
            dist = Normal(mu, fixed_std)

            new_logp = dist.log_prob(action_actor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratio = torch.exp(new_logp - old_logp_actor)

            surr1 = ratio * adv_actor
            surr2 = torch.clamp(
                ratio,
                1.0 - args.clip_range,
                1.0 + args.clip_range,
            ) * adv_actor

            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = value(global_obs_batch)
            value_loss = (new_values - returns).pow(2).mean()

            entropy_term = entropy.mean()

            total_loss = (
                policy_loss
                + args.vf_coef * value_loss
                - entropy_coef * entropy_term
            )

            opt_policy.zero_grad()
            opt_value.zero_grad()

            total_loss.backward()

            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            nn.utils.clip_grad_norm_(value.parameters(), args.max_grad_norm)

            opt_policy.step()
            opt_value.step()

            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_terms.append(entropy_term.detach())

        mean_policy_loss = torch.stack(policy_losses).mean()
        mean_value_loss = torch.stack(value_losses).mean()
        mean_entropy = torch.stack(entropy_terms).mean()

        center_error = safe_float(last_info, "center_error")
        formation_error = safe_float(last_info, "formation_error")
        spacing_error = safe_float(last_info, "spacing_error")
        collision_count = safe_float(last_info, "collision_count")
        obstacle_collision_count = safe_float(last_info, "obstacle_collision_count")
        obstacle_near_miss_count = safe_float(last_info, "obstacle_near_miss_count")
        min_obstacle_margin = safe_float(last_info, "min_obstacle_margin", 999.0)
        crashed = bool(last_info.get("crashed", False))
        phase = int(last_info.get("phase", 0))

        score = (
            12.0 * float(success)
            + phase * 4.0
            - center_error
            - formation_error
            - spacing_error
            - 4.0 * collision_count
            - 5.0 * obstacle_collision_count
            - 1.0 * obstacle_near_miss_count
            + 0.10 * min_obstacle_margin
            - 4.0 * float(crashed)
        )

        if score > best_score:
            best_score = score

            torch.save(
                policy.state_dict(),
                os.path.join(model_dir, "best_shared_velocity_policy_v0_9_1_obstacle_curriculum.pth"),
            )

            torch.save(
                value.state_dict(),
                os.path.join(model_dir, "best_central_value_v0_9_1_obstacle_curriculum.pth"),
            )

        recent_successes.append(float(success))

        if len(recent_successes) > rolling_window:
            recent_successes.pop(0)

        if success:
            patience = 0
        else:
            patience += 1

        current_threshold = stage_success_thresholds[curr_stage]

        if (
            len(recent_successes) == rolling_window
            and np.mean(recent_successes) >= current_threshold
            and curr_stage < len(curriculum) - 1
        ):
            curr_stage += 1
            stage_cfg = curriculum[curr_stage]

            env.close()
            env = build_env(stage_cfg)

            recent_successes = []

            print(f"\nCurriculum advanced to stage {curr_stage + 1}: {stage_cfg['name']}")

        if wandb_enabled:
            import wandb

            wandb.log(
                {
                    "episode": ep,
                    "curriculum_stage": curr_stage + 1,
                    "stage_name": stage_cfg["name"],
                    "total_reward": ep_reward,
                    "success": float(success),
                    "phase": phase,
                    "center_error": center_error,
                    "formation_error": formation_error,
                    "spacing_error": spacing_error,
                    "mean_dist_to_target": safe_float(last_info, "mean_dist_to_target"),
                    "mean_speed": safe_float(last_info, "mean_speed"),
                    "mean_ref_speed": safe_float(last_info, "mean_ref_speed"),
                    "mean_cmd_speed": safe_float(last_info, "mean_cmd_speed"),
                    "mean_correction": safe_float(last_info, "mean_correction"),
                    "action_smoothness": safe_float(last_info, "action_smoothness"),
                    "collision_count": collision_count,
                    "min_pair_dist": safe_float(last_info, "min_pair_dist"),
                    "min_obstacle_margin": safe_float(last_info, "min_obstacle_margin"),
                    "mean_obstacle_margin": safe_float(last_info, "mean_obstacle_margin"),
                    "obstacle_collision_count": safe_float(last_info, "obstacle_collision_count"),
                    "obstacle_near_miss_count": safe_float(last_info, "obstacle_near_miss_count"),
                    "obstacle_clear": float(bool(last_info.get("obstacle_clear", False))),
                    "crashed": float(crashed),
                    "progress": safe_float(last_info, "progress"),
                    "formation_safe": float(bool(last_info.get("formation_safe", False))),
                    "policy_loss": float(mean_policy_loss.item()),
                    "value_loss": float(mean_value_loss.item()),
                    "entropy": float(mean_entropy.item()),
                    "episode_length": int(len(rewards)),
                }
            )

        timestamp = datetime.datetime.now().isoformat()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    timestamp,
                    ep,
                    curr_stage + 1,
                    stage_cfg["name"],
                    ep_reward,
                    int(success),
                    last_info.get("phase", 0),
                    last_info.get("phase_name", ""),
                    last_info.get("center_error", 0.0),
                    last_info.get("mean_dist_to_target", 0.0),
                    last_info.get("formation_error", 0.0),
                    last_info.get("spacing_error", 0.0),
                    last_info.get("mean_speed", 0.0),
                    last_info.get("mean_ref_speed", 0.0),
                    last_info.get("mean_cmd_speed", 0.0),
                    last_info.get("mean_correction", 0.0),
                    last_info.get("action_smoothness", 0.0),
                    last_info.get("collision_count", 0),
                    last_info.get("min_pair_dist", 0.0),
                    last_info.get("min_obstacle_margin", 0.0),
                    last_info.get("mean_obstacle_margin", 0.0),
                    last_info.get("obstacle_collision_count", 0),
                    last_info.get("obstacle_near_miss_count", 0),
                    int(last_info.get("obstacle_clear", False)),
                    int(last_info.get("crashed", False)),
                    last_info.get("progress", 0.0),
                    int(last_info.get("formation_safe", False)),
                    last_info.get("centroid_x", 0.0),
                    last_info.get("centroid_y", 0.0),
                    last_info.get("centroid_z", 0.0),
                ]
            )

        if ep % 50 == 0:
            save_checkpoint(
                policy=policy,
                value=value,
                opt_policy=opt_policy,
                opt_value=opt_value,
                episode=ep,
                stage_idx=curr_stage,
                ckpt_dir=ckpt_dir,
            )

        if ep % 10 == 0 or success:
            rolling_success_rate = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0

            print(
                f"[Ep {ep:4d}] "
                f"Stage: {curr_stage + 1}({stage_cfg['name']}) | "
                f"RollSR: {rolling_success_rate:.2f} | "
                f"Reward: {ep_reward:9.2f} | "
                f"Success: {success} | "
                f"Phase: {phase}({last_info.get('phase_name', '')}) | "
                f"CenterErr: {center_error:.3f} | "
                f"FormErr: {formation_error:.3f} | "
                f"SpacingErr: {spacing_error:.3f} | "
                f"Speed: {safe_float(last_info, 'mean_speed'):.3f} | "
                f"CmdSpeed: {safe_float(last_info, 'mean_cmd_speed'):.3f} | "
                f"Correction: {safe_float(last_info, 'mean_correction'):.3f} | "
                f"MinSep: {safe_float(last_info, 'min_pair_dist'):.3f} | "
                f"ObsMargin: {safe_float(last_info, 'min_obstacle_margin'):.3f} | "
                f"ObsColl: {int(safe_float(last_info, 'obstacle_collision_count'))} | "
                f"NearMiss: {int(safe_float(last_info, 'obstacle_near_miss_count'))} | "
                f"Collisions: {int(collision_count)} | "
                f"Crashed: {crashed} | "
                f"Safe: {bool(last_info.get('formation_safe', False))}"
            )

        if patience >= patience_limit:
            print("Early stopping: too many episodes without success.")
            break

    save_checkpoint(
        policy=policy,
        value=value,
        opt_policy=opt_policy,
        opt_value=opt_value,
        episode=ep,
        stage_idx=curr_stage,
        ckpt_dir=ckpt_dir,
    )

    env.close()

    if wandb_enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()