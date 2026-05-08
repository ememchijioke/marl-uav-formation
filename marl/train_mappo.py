import os
import sys
import csv
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVRealisticEnv


# =========================
# Arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--render", choices=["human", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=1500)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--lr-policy", type=float, default=3e-4)
parser.add_argument("--lr-value", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--vf-coef", type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--explore-std", type=float, default=0.15)
parser.add_argument("--update-epochs", type=int, default=8)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


# =========================
# Curriculum
# =========================
curriculum = [
    {
        "goal_tol": 1.2,
        "formation_tol": 2.0,
        "goal_bonus": 200.0,
        "progress_weight": 8.0,
        "alive_reward": 0.2,
        "collision_penalty": 30.0,
        "min_separation": 0.16,
    },
    {
        "goal_tol": 0.8,
        "formation_tol": 1.5,
        "goal_bonus": 350.0,
        "progress_weight": 12.0,
        "alive_reward": 0.15,
        "collision_penalty": 40.0,
        "min_separation": 0.18,
    },
    {
        "goal_tol": 0.6,
        "formation_tol": 1.2,
        "goal_bonus": 500.0,
        "progress_weight": 15.0,
        "alive_reward": 0.10,
        "collision_penalty": 50.0,
        "min_separation": 0.20,
    },
]

consecutive_success_needed = 15
curr_stage = 0


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Models
# =========================
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
            nn.Tanh(),  # actions normalized to [-1, 1]
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


# =========================
# Utilities
# =========================
def build_env(stage_cfg):
    return MultiUAVRealisticEnv(
        render_mode=args.render,
        num_agents=3,
        max_steps=args.max_steps,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.20, 0.15, 0.10),
        goal_tol=stage_cfg["goal_tol"],
        formation_tol=stage_cfg["formation_tol"],
        goal_bonus=stage_cfg["goal_bonus"],
        formation_penalty=2.0,
        mean_penalty=1.0,
        cohesion_bonus_weight=3.0,
        progress_weight=stage_cfg["progress_weight"],
        alive_reward=stage_cfg["alive_reward"],
        collision_penalty=stage_cfg["collision_penalty"],
        min_separation=stage_cfg["min_separation"],
    )


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


def save_checkpoint(policy, value, opt_pol, opt_val, episode, stage_idx, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "shared_policy.pth"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "central_value.pth"))
    torch.save(opt_pol.state_dict(), os.path.join(ckpt_dir, "opt_policy.pth"))
    torch.save(opt_val.state_dict(), os.path.join(ckpt_dir, "opt_value.pth"))
    with open(os.path.join(ckpt_dir, "last_episode.txt"), "w") as f:
        f.write(str(episode))
    with open(os.path.join(ckpt_dir, "stage.txt"), "w") as f:
        f.write(str(stage_idx))


# =========================
# Training
# =========================
def train():
    global curr_stage

    wandb.init(
        project="marl-uav-formation-realistic",
        name=f"mappo-realistic-{args.render}",
        config=vars(args),
        resume="allow",
    )
    wandb_url = wandb.run.url

    stage = curriculum[curr_stage]
    env = build_env(stage)

    n_agents = env.num_agents
    obs_dim_total = env.observation_space.shape[0]
    obs_dim = obs_dim_total // n_agents
    act_dim = env.action_space.shape[0] // n_agents

    policy = SharedPolicyNet(obs_dim, act_dim).to(device)
    value = CentralValueNet(obs_dim_total).to(device)

    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_val = torch.optim.Adam(value.parameters(), lr=args.lr_value)

    ckpt_dir = "checkpoints_realistic"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("marl/models", exist_ok=True)

    start_episode = 1
    best_reward = -np.inf
    success_streak = 0
    patience = 0
    patience_limit = 400

    if args.resume:
        ep_path = os.path.join(ckpt_dir, "last_episode.txt")
        st_path = os.path.join(ckpt_dir, "stage.txt")
        if os.path.exists(ep_path):
            with open(ep_path, "r") as f:
                start_episode = int(f.read().strip()) + 1
        if os.path.exists(st_path):
            with open(st_path, "r") as f:
                curr_stage = int(f.read().strip())

        stage = curriculum[curr_stage]
        env.close()
        env = build_env(stage)

        policy_path = os.path.join(ckpt_dir, "shared_policy.pth")
        value_path = os.path.join(ckpt_dir, "central_value.pth")
        opt_pol_path = os.path.join(ckpt_dir, "opt_policy.pth")
        opt_val_path = os.path.join(ckpt_dir, "opt_value.pth")

        if os.path.exists(policy_path):
            policy.load_state_dict(torch.load(policy_path, map_location=device))
        if os.path.exists(value_path):
            value.load_state_dict(torch.load(value_path, map_location=device))
        if os.path.exists(opt_pol_path):
            opt_pol.load_state_dict(torch.load(opt_pol_path, map_location=device))
        if os.path.exists(opt_val_path):
            opt_val.load_state_dict(torch.load(opt_val_path, map_location=device))

        print(f"Resumed from episode {start_episode}, curriculum stage {curr_stage + 1}")

    csv_file = "training_logs_realistic.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "episode",
                "stage",
                "reward",
                "success",
                "dist_to_goal",
                "max_err",
                "mean_err",
                "collision_count",
                "centroid_x",
                "wandb_url"
            ])

    fixed_std = torch.full((act_dim,), args.explore_std, device=device)

    for ep in range(start_episode, args.episodes + 1):
        flat_obs, _ = env.reset()
        obs_split = split_obs(flat_obs, n_agents)

        ep_reward = 0.0
        success = False
        last_info = {}

        # Rollout buffers
        buf_global_obs = []
        buf_local_obs = []
        buf_actions = []
        buf_logps = []
        buf_values = []
        rewards = []
        dones = []

        entropy_coef = args.ent_coef * max(0.25, 1.0 - ep / (0.8 * args.episodes))

        for step in range(args.max_steps):
            local_obs_step = []
            actions_step = []
            logps_step = []

            global_obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                value_t = value(global_obs_t).item()

            for i in range(n_agents):
                obs_t = torch.tensor(obs_split[i], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    mu = policy(obs_t)
                    dist = Normal(mu, fixed_std)
                    action = dist.sample()
                    action = torch.clamp(action, -1.0, 1.0)
                    logp = dist.log_prob(action).sum(dim=-1)

                local_obs_step.append(obs_split[i].copy())
                actions_step.append(action.squeeze(0).cpu().numpy())
                logps_step.append(logp.item())

            flat_action = np.concatenate(actions_step, axis=0)
            next_flat_obs, reward,terminated, truncated, info = env.step(flat_action)
            done = bool(terminated or truncated)

            buf_global_obs.append(flat_obs.copy())
            buf_local_obs.append(np.array(local_obs_step, dtype=np.float32))     # [n_agents, obs_dim]
            buf_actions.append(np.array(actions_step, dtype=np.float32))         # [n_agents, act_dim]
            buf_logps.append(np.array(logps_step, dtype=np.float32))             # [n_agents]
            buf_values.append(float(value_t))                                    # scalar team value

            rewards.append(float(reward))
            dones.append(bool(done))

            ep_reward += reward
            flat_obs = next_flat_obs
            obs_split = split_obs(flat_obs, n_agents)
            last_info = info

            if done:
                success = bool(info.get("is_success", False))
                break

        # Bootstrap centralized value
        with torch.no_grad():
            next_global_obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = value(next_global_obs_t).item()

        adv_list, ret_list = compute_gae(
            rewards=rewards,
            dones=dones,
            values=buf_values,
            next_value=next_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        advantages = torch.tensor(adv_list, dtype=torch.float32, device=device)
        returns = torch.tensor(ret_list, dtype=torch.float32, device=device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert rollout buffers to tensors
        global_obs_batch = torch.tensor(np.array(buf_global_obs), dtype=torch.float32, device=device)   # [T, obs_dim_total]
        local_obs_batch = torch.tensor(np.array(buf_local_obs), dtype=torch.float32, device=device)     # [T, n_agents, obs_dim]
        action_batch = torch.tensor(np.array(buf_actions), dtype=torch.float32, device=device)          # [T, n_agents, act_dim]
        old_logp_batch = torch.tensor(np.array(buf_logps), dtype=torch.float32, device=device)          # [T, n_agents]

        T = local_obs_batch.shape[0]

        # Flatten agents for shared actor update
        local_obs_actor = local_obs_batch.reshape(T * n_agents, obs_dim)
        action_actor = action_batch.reshape(T * n_agents, act_dim)
        old_logp_actor = old_logp_batch.reshape(T * n_agents)

        # Same team advantage repeated for each agent at each timestep
        adv_actor = advantages.unsqueeze(1).repeat(1, n_agents).reshape(T * n_agents)

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
            surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * adv_actor
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = value(global_obs_batch)
            value_loss = (new_values - returns).pow(2).mean()

            entropy_term = entropy.mean()

            total_loss = policy_loss + args.vf_coef * value_loss - entropy_coef * entropy_term

            opt_pol.zero_grad()
            opt_val.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            nn.utils.clip_grad_norm_(value.parameters(), args.max_grad_norm)
            opt_pol.step()
            opt_val.step()

            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_terms.append(entropy_term.detach())

        total_policy_loss = torch.stack(policy_losses).mean()
        total_value_loss = torch.stack(value_losses).mean()
        total_entropy = torch.stack(entropy_terms).mean()

        # Best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(policy.state_dict(), "marl/models/best_shared_policy_realistic.pth")
            torch.save(value.state_dict(), "marl/models/best_central_value_realistic.pth")

        # Curriculum
        if success:
            success_streak += 1
            patience = 0
        else:
            success_streak = 0
            patience += 1

        if success_streak >= consecutive_success_needed and curr_stage < len(curriculum) - 1:
            curr_stage += 1
            stage = curriculum[curr_stage]
            env.close()
            env = build_env(stage)
            success_streak = 0
            print(f"\nCurriculum advanced to stage {curr_stage + 1}: {stage}")

        # Logging
        wandb.log({
            "episode": ep,
            "curriculum_stage": curr_stage + 1,
            "total_reward": ep_reward,
            "success": float(success),
            "dist_to_goal": float(last_info.get("dist_to_goal", math.nan)),
            "centroid_x": float(last_info.get("centroid_x", math.nan)),
            "max_formation_error": float(last_info.get("max_formation_error", math.nan)),
            "mean_formation_error": float(last_info.get("mean_formation_error", math.nan)),
            "collision_count": float(last_info.get("collision_count", 0)),
            "min_pair_dist": float(last_info.get("min_pair_dist", math.nan)),
            "cohesion_bonus": float(last_info.get("cohesion_bonus", 0.0)),
            "mean_speed": float(last_info.get("mean_speed", 0.0)),
            "entropy_coef": float(entropy_coef),
            "policy_loss": float(total_policy_loss.item()),
            "value_loss": float(total_value_loss.item()),
            "entropy": float(total_entropy.item()),
            "episode_length": int(len(rewards)),
        })

        timestamp = datetime.datetime.now().isoformat()
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                ep,
                curr_stage + 1,
                ep_reward,
                int(success),
                last_info.get("dist_to_goal", 0.0),
                last_info.get("max_formation_error", 0.0),
                last_info.get("mean_formation_error", 0.0),
                last_info.get("collision_count", 0),
                last_info.get("centroid_x", 0.0),
                wandb_url,
            ])

        if ep % 50 == 0:
            save_checkpoint(policy, value, opt_pol, opt_val, ep, curr_stage, ckpt_dir)

        if ep % 25 == 0 or success:
            print(
                f"[Ep {ep:4d}] "
                f"Stage: {curr_stage + 1} | "
                f"Reward: {ep_reward:8.2f} | "
                f"Success: {success} | "
                f"GoalDist: {float(last_info.get('dist_to_goal', 0.0)):.2f} | "
                f"CentroidX: {float(last_info.get('centroid_x', 0.0)):.2f} | "
                f"MeanErr: {float(last_info.get('mean_formation_error', 0.0)):.2f} | "
                f"Collisions: {int(last_info.get('collision_count', 0))}"
            )

        if patience >= patience_limit:
            print("Early stopping: too many episodes without success.")
            break

    save_checkpoint(policy, value, opt_pol, opt_val, ep, curr_stage, ckpt_dir)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()