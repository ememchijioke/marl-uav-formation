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

from envs.quad_env_velocity_2uav import MultiUAVVelocityFormation2UAVEnv

parser = argparse.ArgumentParser()

parser.add_argument("--render", choices=["human", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--max-steps", type=int, default=500)

parser.add_argument("--lr-policy", type=float, default=2.0e-4)
parser.add_argument("--lr-value", type=float, default=8.0e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.010)
parser.add_argument("--vf-coef", type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--explore-std", type=float, default=0.08)
parser.add_argument("--update-epochs", type=int, default=8)

parser.add_argument("--resume", action="store_true")
parser.add_argument("--use-wandb", action="store_true")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedResidualPolicyNet(nn.Module):
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
    return MultiUAVVelocityFormation2UAVEnv(
        render_mode=args.render,
        num_agents=2,
        max_steps=args.max_steps,
        ctrl_freq=48,
        sim_freq=240,
        target_altitude=1.0,
        target_distance_x=stage_cfg["target_distance_x"],
        formation_spacing=stage_cfg["formation_spacing"],
        reference_max_vx=stage_cfg["reference_max_vx"],
        reference_max_vy=stage_cfg["reference_max_vy"],
        reference_max_vz=stage_cfg["reference_max_vz"],
        residual_scale=stage_cfg["residual_scale"],
        velocity_lookahead=stage_cfg["velocity_lookahead"],
        kp_goal=stage_cfg["kp_goal"],
        kp_form=stage_cfg["kp_form"],
        kp_alt=stage_cfg["kp_alt"],
        goal_bonus=stage_cfg["goal_bonus"],
        alive_reward=stage_cfg["alive_reward"],
        progress_weight=stage_cfg["progress_weight"],
        centroid_weight=stage_cfg["centroid_weight"],
        own_goal_weight=stage_cfg["own_goal_weight"],
        formation_weight=stage_cfg["formation_weight"],
        spacing_weight=stage_cfg["spacing_weight"],
        velocity_weight=stage_cfg["velocity_weight"],
        command_weight=stage_cfg["command_weight"],
        residual_weight=stage_cfg["residual_weight"],
        smoothness_weight=stage_cfg["smoothness_weight"],
        attitude_weight=stage_cfg["attitude_weight"],
        collision_penalty=stage_cfg["collision_penalty"],
        crash_penalty=stage_cfg["crash_penalty"],
        boundary_penalty=stage_cfg["boundary_penalty"],
        success_centroid_tol=stage_cfg["success_centroid_tol"],
        success_formation_tol=stage_cfg["success_formation_tol"],
        success_spacing_tol=stage_cfg["success_spacing_tol"],
        min_separation=stage_cfg["min_separation"],
        neighbor_dropout_prob=stage_cfg["neighbor_dropout_prob"],
        use_neighbor_dropout=True,
    )


def save_checkpoint(policy, value, opt_policy, opt_value, episode, stage_idx, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "shared_residual_policy_2uav.pth"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "central_value_2uav.pth"))
    torch.save(opt_policy.state_dict(), os.path.join(ckpt_dir, "opt_policy_2uav.pth"))
    torch.save(opt_value.state_dict(), os.path.join(ckpt_dir, "opt_value_2uav.pth"))

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
        "policy": os.path.join(ckpt_dir, "shared_residual_policy_2uav.pth"),
        "value": os.path.join(ckpt_dir, "central_value_2uav.pth"),
        "opt_policy": os.path.join(ckpt_dir, "opt_policy_2uav.pth"),
        "opt_value": os.path.join(ckpt_dir, "opt_value_2uav.pth"),
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


def safe_float(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def train():
    curriculum = [
        {
            "target_distance_x": 1.6,
            "formation_spacing": 0.80,
            "reference_max_vx": 0.18,
            "reference_max_vy": 0.14,
            "reference_max_vz": 0.08,
            "residual_scale": 0.06,
            "velocity_lookahead": 0.25,
            "kp_goal": 0.30,
            "kp_form": 0.55,
            "kp_alt": 0.40,
            "goal_bonus": 900.0,
            "alive_reward": 0.05,
            "progress_weight": 30.0,
            "centroid_weight": 2.5,
            "own_goal_weight": 1.2,
            "formation_weight": 22.0,
            "spacing_weight": 14.0,
            "velocity_weight": 0.20,
            "command_weight": 0.06,
            "residual_weight": 0.12,
            "smoothness_weight": 0.22,
            "attitude_weight": 0.70,
            "collision_penalty": 2200.0,
            "crash_penalty": 2600.0,
            "boundary_penalty": 800.0,
            "success_centroid_tol": 0.36,
            "success_formation_tol": 0.24,
            "success_spacing_tol": 0.20,
            "min_separation": 0.32,
            "neighbor_dropout_prob": 0.02,
        },
        {
            "target_distance_x": 2.0,
            "formation_spacing": 0.80,
            "reference_max_vx": 0.22,
            "reference_max_vy": 0.16,
            "reference_max_vz": 0.10,
            "residual_scale": 0.08,
            "velocity_lookahead": 0.28,
            "kp_goal": 0.35,
            "kp_form": 0.55,
            "kp_alt": 0.40,
            "goal_bonus": 1050.0,
            "alive_reward": 0.045,
            "progress_weight": 34.0,
            "centroid_weight": 3.0,
            "own_goal_weight": 1.5,
            "formation_weight": 22.0,
            "spacing_weight": 14.0,
            "velocity_weight": 0.25,
            "command_weight": 0.08,
            "residual_weight": 0.12,
            "smoothness_weight": 0.25,
            "attitude_weight": 0.80,
            "collision_penalty": 2400.0,
            "crash_penalty": 2800.0,
            "boundary_penalty": 900.0,
            "success_centroid_tol": 0.32,
            "success_formation_tol": 0.22,
            "success_spacing_tol": 0.18,
            "min_separation": 0.34,
            "neighbor_dropout_prob": 0.03,
        },
        {
            "target_distance_x": 2.4,
            "formation_spacing": 0.80,
            "reference_max_vx": 0.24,
            "reference_max_vy": 0.18,
            "reference_max_vz": 0.10,
            "residual_scale": 0.10,
            "velocity_lookahead": 0.30,
            "kp_goal": 0.38,
            "kp_form": 0.60,
            "kp_alt": 0.45,
            "goal_bonus": 1200.0,
            "alive_reward": 0.04,
            "progress_weight": 38.0,
            "centroid_weight": 3.3,
            "own_goal_weight": 1.7,
            "formation_weight": 24.0,
            "spacing_weight": 16.0,
            "velocity_weight": 0.28,
            "command_weight": 0.10,
            "residual_weight": 0.14,
            "smoothness_weight": 0.28,
            "attitude_weight": 0.90,
            "collision_penalty": 2600.0,
            "crash_penalty": 3000.0,
            "boundary_penalty": 1000.0,
            "success_centroid_tol": 0.30,
            "success_formation_tol": 0.20,
            "success_spacing_tol": 0.16,
            "min_separation": 0.35,
            "neighbor_dropout_prob": 0.05,
        },
    ]

    consecutive_success_needed = 6
    curr_stage = 0

    ckpt_dir = "checkpoints/v0_8_3_residual_2uav"
    model_dir = "marl/models/v0_8_3_residual_2uav"
    log_dir = "results/logs"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    wandb_enabled = False

    if args.use_wandb:
        try:
            import wandb

            wandb.init(
                project="marl-uav-v0-8-3-residual-2uav",
                name=f"residual-mappo-2uav-{args.render}",
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
    print("Action meaning: residual correction on reference velocity [rx, ry, rz]")
    print("v0.8.3 mode: 2-UAV residual velocity MAPPO")

    policy = SharedResidualPolicyNet(obs_dim, act_dim).to(device)
    value = CentralValueNet(obs_dim_total).to(device)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_value = torch.optim.Adam(value.parameters(), lr=args.lr_value)

    start_episode = 1
    best_score = -np.inf
    success_streak = 0
    patience = 0
    patience_limit = 1000

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

        print(f"Resumed from episode {start_episode}, stage {curr_stage + 1}")

    csv_path = os.path.join(log_dir, "training_logs_v0_8_3_residual_2uav.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "timestamp",
                    "episode",
                    "stage",
                    "reward",
                    "success",
                    "centroid_error",
                    "mean_dist_to_target",
                    "max_dist_to_target",
                    "formation_error",
                    "spacing_error",
                    "pair_distance",
                    "mean_speed",
                    "mean_ref_speed",
                    "mean_cmd_speed",
                    "mean_residual",
                    "action_smoothness",
                    "mean_attitude_error",
                    "mean_tracking_error",
                    "collision_count",
                    "min_pair_dist",
                    "crashed",
                    "out_of_bounds",
                    "unstable_attitude",
                    "progress",
                    "formation_safe",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                    "uav0_x",
                    "uav1_x",
                    "uav0_y",
                    "uav1_y",
                    "uav0_z",
                    "uav1_z",
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

        centroid_error = safe_float(last_info, "centroid_error")
        formation_error = safe_float(last_info, "formation_error")
        spacing_error = safe_float(last_info, "spacing_error")
        collision_count = safe_float(last_info, "collision_count")
        crashed = bool(last_info.get("crashed", False))

        score = (
            -centroid_error
            -formation_error
            -spacing_error
            -2.0 * collision_count
            -2.0 * float(crashed)
        )

        if score > best_score:
            best_score = score

            torch.save(
                policy.state_dict(),
                os.path.join(model_dir, "best_shared_residual_policy_2uav_v0_8_3.pth"),
            )

            torch.save(
                value.state_dict(),
                os.path.join(model_dir, "best_central_value_2uav_v0_8_3.pth"),
            )

        if success:
            success_streak += 1
            patience = 0
        else:
            success_streak = 0
            patience += 1

        if success_streak >= consecutive_success_needed and curr_stage < len(curriculum) - 1:
            curr_stage += 1
            stage_cfg = curriculum[curr_stage]

            env.close()
            env = build_env(stage_cfg)

            success_streak = 0

            print(f"\nCurriculum advanced to stage {curr_stage + 1}")

        if wandb_enabled:
            import wandb

            wandb.log(
                {
                    "episode": ep,
                    "curriculum_stage": curr_stage + 1,
                    "total_reward": ep_reward,
                    "success": float(success),
                    "centroid_error": centroid_error,
                    "formation_error": formation_error,
                    "spacing_error": spacing_error,
                    "pair_distance": safe_float(last_info, "pair_distance"),
                    "mean_dist_to_target": safe_float(last_info, "mean_dist_to_target"),
                    "mean_speed": safe_float(last_info, "mean_speed"),
                    "mean_ref_speed": safe_float(last_info, "mean_ref_speed"),
                    "mean_cmd_speed": safe_float(last_info, "mean_cmd_speed"),
                    "mean_residual": safe_float(last_info, "mean_residual"),
                    "action_smoothness": safe_float(last_info, "action_smoothness"),
                    "mean_tracking_error": safe_float(last_info, "mean_tracking_error"),
                    "collision_count": collision_count,
                    "min_pair_dist": safe_float(last_info, "min_pair_dist"),
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
                    ep_reward,
                    int(success),
                    last_info.get("centroid_error", 0.0),
                    last_info.get("mean_dist_to_target", 0.0),
                    last_info.get("max_dist_to_target", 0.0),
                    last_info.get("formation_error", 0.0),
                    last_info.get("spacing_error", 0.0),
                    last_info.get("pair_distance", 0.0),
                    last_info.get("mean_speed", 0.0),
                    last_info.get("mean_ref_speed", 0.0),
                    last_info.get("mean_cmd_speed", 0.0),
                    last_info.get("mean_residual", 0.0),
                    last_info.get("action_smoothness", 0.0),
                    last_info.get("mean_attitude_error", 0.0),
                    last_info.get("mean_tracking_error", 0.0),
                    last_info.get("collision_count", 0),
                    last_info.get("min_pair_dist", 0.0),
                    int(last_info.get("crashed", False)),
                    int(last_info.get("out_of_bounds", False)),
                    int(last_info.get("unstable_attitude", False)),
                    last_info.get("progress", 0.0),
                    int(last_info.get("formation_safe", False)),
                    last_info.get("centroid_x", 0.0),
                    last_info.get("centroid_y", 0.0),
                    last_info.get("centroid_z", 0.0),
                    last_info.get("uav0_x", 0.0),
                    last_info.get("uav1_x", 0.0),
                    last_info.get("uav0_y", 0.0),
                    last_info.get("uav1_y", 0.0),
                    last_info.get("uav0_z", 0.0),
                    last_info.get("uav1_z", 0.0),
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
            print(
                f"[Ep {ep:4d}] "
                f"Stage: {curr_stage + 1} | "
                f"Reward: {ep_reward:9.2f} | "
                f"Success: {success} | "
                f"CentErr: {centroid_error:.3f} | "
                f"FormErr: {formation_error:.3f} | "
                f"SpacingErr: {spacing_error:.3f} | "
                f"PairDist: {safe_float(last_info, 'pair_distance'):.3f} | "
                f"Dist: {safe_float(last_info, 'mean_dist_to_target'):.3f} | "
                f"Speed: {safe_float(last_info, 'mean_speed'):.3f} | "
                f"RefSpeed: {safe_float(last_info, 'mean_ref_speed'):.3f} | "
                f"CmdSpeed: {safe_float(last_info, 'mean_cmd_speed'):.3f} | "
                f"Residual: {safe_float(last_info, 'mean_residual'):.3f} | "
                f"MinSep: {safe_float(last_info, 'min_pair_dist'):.3f} | "
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