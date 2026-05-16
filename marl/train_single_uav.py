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

from envs.single_uav_env import SingleUAVHoverEnv


parser = argparse.ArgumentParser()
parser.add_argument("--render", choices=["human", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=1500)
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument("--lr-policy", type=float, default=3e-4)
parser.add_argument("--lr-value", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--vf-coef", type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--explore-std", type=float, default=0.04)
parser.add_argument("--update-epochs", type=int, default=8)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


curriculum = [
    {
        "hover_tol": 0.30,
        "landing_tol": 0.30,
        "goal_bonus": 300.0,
        "distance_weight": 3.0,
        "altitude_weight": 5.0,
        "velocity_weight": 1.0,
        "attitude_weight": 1.0,
        "alive_reward": 0.3,
    },
    {
        "hover_tol": 0.23,
        "landing_tol": 0.25,
        "goal_bonus": 450.0,
        "distance_weight": 5.0,
        "altitude_weight": 7.0,
        "velocity_weight": 1.3,
        "attitude_weight": 1.3,
        "alive_reward": 0.2,
    },
    {
        "hover_tol": 0.18,
        "landing_tol": 0.20,
        "goal_bonus": 600.0,
        "distance_weight": 6.0,
        "altitude_weight": 8.0,
        "velocity_weight": 1.5,
        "attitude_weight": 1.5,
        "alive_reward": 0.1,
    },
]

consecutive_success_needed = 12
curr_stage = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
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


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_env(stage_cfg):
    return SingleUAVHoverEnv(
        render_mode=args.render,
        max_steps=args.max_steps,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.04, 0.04, 0.05),
        hover_tol=stage_cfg["hover_tol"],
        landing_tol=stage_cfg["landing_tol"],
        goal_bonus=stage_cfg["goal_bonus"],
        distance_weight=stage_cfg["distance_weight"],
        altitude_weight=stage_cfg["altitude_weight"],
        velocity_weight=stage_cfg["velocity_weight"],
        attitude_weight=stage_cfg["attitude_weight"],
        alive_reward=stage_cfg["alive_reward"],
        crash_penalty=250.0,
    )


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

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "single_policy.pth"))
    torch.save(value.state_dict(), os.path.join(ckpt_dir, "single_value.pth"))
    torch.save(opt_pol.state_dict(), os.path.join(ckpt_dir, "opt_policy.pth"))
    torch.save(opt_val.state_dict(), os.path.join(ckpt_dir, "opt_value.pth"))

    with open(os.path.join(ckpt_dir, "last_episode.txt"), "w") as f:
        f.write(str(episode))

    with open(os.path.join(ckpt_dir, "stage.txt"), "w") as f:
        f.write(str(stage_idx))


def train():
    global curr_stage

    wandb.init(
        project="single-uav-rl-baseline",
        name=f"ppo-single-uav-mission-{args.render}",
        config=vars(args),
        resume="allow",
    )

    wandb_url = wandb.run.url

    stage = curriculum[curr_stage]
    env = build_env(stage)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNet(obs_dim, act_dim).to(device)
    value = ValueNet(obs_dim).to(device)

    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_val = torch.optim.Adam(value.parameters(), lr=args.lr_value)

    ckpt_dir = "checkpoints_single_uav"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("marl/models", exist_ok=True)

    start_episode = 1
    best_reward = -np.inf
    success_streak = 0
    patience = 0
    patience_limit = 700

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

        policy_path = os.path.join(ckpt_dir, "single_policy.pth")
        value_path = os.path.join(ckpt_dir, "single_value.pth")
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

    csv_file = "training_logs_single_uav.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "episode",
                "stage",
                "reward",
                "success",
                "phase",
                "phase_id",
                "stable_counter",
                "move_progress",
                "dist_to_target",
                "altitude",
                "xy_error",
                "speed",
                "attitude_error",
                "crashed",
                "landed_successfully",
                "pos_x",
                "pos_y",
                "pos_z",
                "wandb_url",
            ])

    fixed_std = torch.full((act_dim,), args.explore_std, device=device)

    for ep in range(start_episode, args.episodes + 1):
        obs, _ = env.reset()

        ep_reward = 0.0
        success = False
        last_info = {}

        buf_obs = []
        buf_actions = []
        buf_logps = []
        buf_values = []
        rewards = []
        dones = []

        entropy_coef = args.ent_coef * max(0.25, 1.0 - ep / (0.8 * args.episodes))

        for step in range(args.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                value_t = value(obs_t).item()
                mu = policy(obs_t)
                dist = Normal(mu, fixed_std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                logp = dist.log_prob(action).sum(dim=-1)

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            buf_obs.append(obs.copy())
            buf_actions.append(action_np.copy())
            buf_logps.append(float(logp.item()))
            buf_values.append(float(value_t))

            rewards.append(float(reward))
            dones.append(bool(done))

            ep_reward += reward
            obs = next_obs
            last_info = info

            if done:
                success = bool(info.get("is_success", False))
                break

        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = value(next_obs_t).item()

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

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.tensor(np.array(buf_obs), dtype=torch.float32, device=device)
        action_batch = torch.tensor(np.array(buf_actions), dtype=torch.float32, device=device)
        old_logp_batch = torch.tensor(np.array(buf_logps), dtype=torch.float32, device=device)

        policy_losses = []
        value_losses = []
        entropy_terms = []

        for _ in range(args.update_epochs):
            mu = policy(obs_batch)
            dist = Normal(mu, fixed_std)

            new_logp = dist.log_prob(action_batch).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratio = torch.exp(new_logp - old_logp_batch)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - args.clip_range,
                1.0 + args.clip_range,
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = value(obs_batch)
            value_loss = (new_values - returns).pow(2).mean()

            entropy_term = entropy.mean()

            total_loss = (
                policy_loss
                + args.vf_coef * value_loss
                - entropy_coef * entropy_term
            )

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

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(policy.state_dict(), "marl/models/best_single_uav_policy.pth")
            torch.save(value.state_dict(), "marl/models/best_single_uav_value.pth")

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

        wandb.log({
            "episode": ep,
            "curriculum_stage": curr_stage + 1,
            "total_reward": ep_reward,
            "success": float(success),
            "phase_id": float(last_info.get("phase_id", 0)),
            "stable_counter": float(last_info.get("stable_counter", 0)),
            "move_progress": float(last_info.get("move_progress", 0.0)),
            "dist_to_target": float(last_info.get("dist_to_target", math.nan)),
            "altitude": float(last_info.get("altitude", math.nan)),
            "xy_error": float(last_info.get("xy_error", math.nan)),
            "speed": float(last_info.get("speed", math.nan)),
            "attitude_error": float(last_info.get("attitude_error", math.nan)),
            "crashed": float(last_info.get("crashed", False)),
            "landed_successfully": float(last_info.get("landed_successfully", False)),
            "pos_x": float(last_info.get("pos_x", math.nan)),
            "pos_y": float(last_info.get("pos_y", math.nan)),
            "pos_z": float(last_info.get("pos_z", math.nan)),
            "target_x": float(last_info.get("target_x", math.nan)),
            "target_y": float(last_info.get("target_y", math.nan)),
            "target_z": float(last_info.get("target_z", math.nan)),
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
                last_info.get("phase", ""),
                last_info.get("phase_id", 0),
                last_info.get("stable_counter", 0),
                last_info.get("move_progress", 0.0),
                last_info.get("dist_to_target", 0.0),
                last_info.get("altitude", 0.0),
                last_info.get("xy_error", 0.0),
                last_info.get("speed", 0.0),
                last_info.get("attitude_error", 0.0),
                int(last_info.get("crashed", False)),
                int(last_info.get("landed_successfully", False)),
                last_info.get("pos_x", 0.0),
                last_info.get("pos_y", 0.0),
                last_info.get("pos_z", 0.0),
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
                f"Phase: {last_info.get('phase', '')} | "
                f"Dist: {float(last_info.get('dist_to_target', 0.0)):.2f} | "
                f"Alt: {float(last_info.get('altitude', 0.0)):.2f} | "
                f"XYErr: {float(last_info.get('xy_error', 0.0)):.2f} | "
                f"Speed: {float(last_info.get('speed', 0.0)):.2f} | "
                f"Crashed: {bool(last_info.get('crashed', False))}"
            )

        if patience >= patience_limit:
            print("Early stopping: too many episodes without success.")
            break

    save_checkpoint(policy, value, opt_pol, opt_val, ep, curr_stage, ckpt_dir)

    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()