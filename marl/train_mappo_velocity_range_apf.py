import os
import sys
import csv
import argparse
import copy
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env_velocity_range_apf import MultiUAVVelocityRangeAPFEnv


parser = argparse.ArgumentParser()

parser.add_argument("--render", choices=["human", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=4000)
parser.add_argument("--max-steps", type=int, default=1200)

parser.add_argument("--lr-policy", type=float, default=1.5e-4)
parser.add_argument("--lr-value", type=float, default=7.0e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.006)
parser.add_argument("--vf-coef", type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--explore-std", type=float, default=0.018)
parser.add_argument("--update-epochs", type=int, default=8)

parser.add_argument("--resume", action="store_true")
parser.add_argument("--init-from-v010", type=str, default=None,
                    help="Optional path to a v0.10.x checkpoint directory used to warm-start the policy/value networks.")
parser.add_argument("--start-stage", type=int, default=0,
                    help="Zero-based curriculum stage to start from when not resuming. Use 3 for soft_center. With this file, stage 4 is medium_center and stage 5 is single_center.")
parser.add_argument("--use-wandb", action="store_true")

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
    return MultiUAVVelocityRangeAPFEnv(
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
        velocity_lookahead=0.30,
        kp_center=stage_cfg["kp_center"],
        kp_form=stage_cfg["kp_form"],
        kp_alt=0.45,
        goal_bonus=stage_cfg["goal_bonus"],
        alive_reward=0.04,
        phase_bonus=stage_cfg["phase_bonus"],
        progress_weight=stage_cfg["progress_weight"],
        center_weight=3.5,
        assigned_target_weight=1.8,
        formation_weight=stage_cfg["formation_weight"],
        spacing_weight=stage_cfg["spacing_weight"],
        velocity_weight=0.28,
        command_weight=0.10,
        correction_weight=0.14,
        smoothness_weight=0.28,
        attitude_weight=0.90,
        collision_penalty=3800.0,
        crash_penalty=4400.0,
        boundary_penalty=1600.0,
        obstacle_collision_penalty=stage_cfg["obstacle_collision_penalty"],
        obstacle_near_penalty_weight=stage_cfg["obstacle_near_penalty_weight"],
        obstacle_clearance_bonus=stage_cfg["obstacle_clearance_bonus"],
        formation_near_obstacle_weight=stage_cfg["formation_near_obstacle_weight"],
        spacing_near_obstacle_weight=stage_cfg["spacing_near_obstacle_weight"],
        min_separation=0.30,
        min_obstacle_clearance=0.32,
        obstacle_influence_radius=1.20,
        obstacle_repulsion_gain=0.0,
        safety_filter_gain=stage_cfg["safety_filter_gain"],
        emergency_clearance=stage_cfg["range_emergency"],
        centroid_bypass_gain=0.0,
        centroid_bypass_window=1.45,
        max_shared_avoidance_speed=0.0,
        formation_lock_gain=stage_cfg["formation_lock_gain"],
        neighbor_dropout_prob=stage_cfg["neighbor_dropout_prob"],
        use_neighbor_dropout=True,
        use_obstacles=stage_cfg["use_obstacles"],
        obstacle_layout=stage_cfg["obstacle_layout"],
        range_max=stage_cfg["range_max"],
        range_safe=stage_cfg["range_safe"],
        range_emergency=stage_cfg["range_emergency"],
        range_apf_gain=stage_cfg["range_apf_gain"],
        formation_apf_gain=stage_cfg["formation_apf_gain"],
        max_apf_speed=stage_cfg["max_apf_speed"],
        tangential_apf_gain=stage_cfg["tangential_apf_gain"],
        formation_tangential_gain=stage_cfg["formation_tangential_gain"],
        tangential_clearance_bias=stage_cfg["tangential_clearance_bias"],
        bypass_memory_enabled=stage_cfg["bypass_memory_enabled"],
        bypass_memory_trigger=stage_cfg["bypass_memory_trigger"],
        bypass_memory_release=stage_cfg["bypass_memory_release"],
        bypass_hold_steps=stage_cfg["bypass_hold_steps"],
        bypass_preferred_side=stage_cfg["bypass_preferred_side"],
        bypass_lateral_gain=stage_cfg["bypass_lateral_gain"],
        recovery_enabled=stage_cfg["recovery_enabled"],
        recovery_steps=stage_cfg["recovery_steps"],
        recovery_forward_boost=stage_cfg["recovery_forward_boost"],
        recovery_lateral_gain=stage_cfg["recovery_lateral_gain"],
        recovery_apf_scale=stage_cfg["recovery_apf_scale"],
        recovery_clear_range=stage_cfg["recovery_clear_range"],
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


def warm_start_from_checkpoint(policy, value, checkpoint_dir):
    """Load policy/value weights from a previous compatible checkpoint directory."""
    policy_path = os.path.join(checkpoint_dir, "shared_velocity_policy.pth")
    value_path = os.path.join(checkpoint_dir, "central_value.pth")

    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"Warm-started policy from: {policy_path}")
    else:
        print(f"[WARN] Policy warm-start path not found: {policy_path}")

    if os.path.exists(value_path):
        value.load_state_dict(torch.load(value_path, map_location=device))
        print(f"Warm-started value network from: {value_path}")
    else:
        print(f"[WARN] Value warm-start path not found: {value_path}")


def safe_float(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def get_curriculum():
    curriculum = [
        {
            "name": "no_obstacle_range_bootstrap",
            "mission_stage": 3,
            "use_obstacles": False,
            "obstacle_layout": "none",
            "triangle_side": 1.20,
            "reference_max_vx": 0.27,
            "reference_max_vy": 0.22,
            "reference_max_vz": 0.14,
            "correction_scale": 0.060,
            "kp_center": 0.38,
            "kp_form": 0.72,
            "goal_bonus": 1600.0,
            "phase_bonus": 240.0,
            "progress_weight": 46.0,
            "formation_weight": 32.0,
            "spacing_weight": 20.0,
            "obstacle_collision_penalty": 3000.0,
            "obstacle_near_penalty_weight": 0.0,
            "obstacle_clearance_bonus": 0.0,
            "formation_near_obstacle_weight": 0.0,
            "spacing_near_obstacle_weight": 0.0,
            "formation_lock_gain": 0.0,
            "safety_filter_gain": 0.0,
            "neighbor_dropout_prob": 0.01,
            "range_max": 2.50,
            "range_safe": 0.65,
            "range_emergency": 0.25,
            "range_apf_gain": 0.0,
            "formation_apf_gain": 0.0,
            "max_apf_speed": 0.0,
            "tangential_apf_gain": 0.0,
            "formation_tangential_gain": 0.0,
            "tangential_clearance_bias": 0.0,
            "bypass_memory_enabled": False,
            "bypass_memory_trigger": 1.15,
            "bypass_memory_release": 1.85,
            "bypass_hold_steps": 180,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.0,
            "recovery_enabled": False,
            "recovery_steps": 0,
            "recovery_forward_boost": 0.0,
            "recovery_lateral_gain": 0.0,
            "recovery_apf_scale": 1.0,
            "recovery_clear_range": 1.60,
        },
        {
            "name": "easy_offset_range_apf",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "easy_offset",
            "triangle_side": 1.20,
            "reference_max_vx": 0.26,
            "reference_max_vy": 0.24,
            "reference_max_vz": 0.14,
            "correction_scale": 0.065,
            "kp_center": 0.38,
            "kp_form": 0.72,
            "goal_bonus": 1750.0,
            "phase_bonus": 250.0,
            "progress_weight": 47.0,
            "formation_weight": 34.0,
            "spacing_weight": 22.0,
            "obstacle_collision_penalty": 3500.0,
            "obstacle_near_penalty_weight": 12.0,
            "obstacle_clearance_bonus": 10.0,
            "formation_near_obstacle_weight": 18.0,
            "spacing_near_obstacle_weight": 16.0,
            "formation_lock_gain": 0.55,
            "safety_filter_gain": 0.14,
            "neighbor_dropout_prob": 0.012,
            "range_max": 2.50,
            "range_safe": 0.70,
            "range_emergency": 0.24,
            "range_apf_gain": 0.16,
            "formation_apf_gain": 0.30,
            "max_apf_speed": 0.20,
            "tangential_apf_gain": 0.08,
            "formation_tangential_gain": 0.12,
            "tangential_clearance_bias": 0.15,
            "bypass_memory_enabled": True,
            "bypass_memory_trigger": 1.10,
            "bypass_memory_release": 1.85,
            "bypass_hold_steps": 160,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.12,
            "recovery_enabled": True,
            "recovery_steps": 180,
            "recovery_forward_boost": 0.08,
            "recovery_lateral_gain": 0.45,
            "recovery_apf_scale": 0.45,
            "recovery_clear_range": 1.60,
        },
        {
            "name": "side_column_range_apf",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "side_column",
            "triangle_side": 1.20,
            "reference_max_vx": 0.24,
            "reference_max_vy": 0.25,
            "reference_max_vz": 0.14,
            "correction_scale": 0.065,
            "kp_center": 0.38,
            "kp_form": 0.72,
            "goal_bonus": 1850.0,
            "phase_bonus": 260.0,
            "progress_weight": 48.0,
            "formation_weight": 44.0,
            "spacing_weight": 34.0,
            "obstacle_collision_penalty": 3800.0,
            "obstacle_near_penalty_weight": 12.0,
            "obstacle_clearance_bonus": 12.0,
            "formation_near_obstacle_weight": 36.0,
            "spacing_near_obstacle_weight": 30.0,
            "formation_lock_gain": 1.20,
            "safety_filter_gain": 0.12,
            "neighbor_dropout_prob": 0.015,
            "range_max": 2.50,
            "range_safe": 0.80,
            "range_emergency": 0.25,
            "range_apf_gain": 0.12,
            "formation_apf_gain": 0.45,
            "max_apf_speed": 0.22,
            "tangential_apf_gain": 0.10,
            "formation_tangential_gain": 0.18,
            "tangential_clearance_bias": 0.20,
            "bypass_memory_enabled": True,
            "bypass_memory_trigger": 1.00,
            "bypass_memory_release": 1.70,
            "bypass_hold_steps": 120,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.08,
            "recovery_enabled": True,
            "recovery_steps": 180,
            "recovery_forward_boost": 0.08,
            "recovery_lateral_gain": 0.45,
            "recovery_apf_scale": 0.45,
            "recovery_clear_range": 1.60,
        },
        {
            "name": "soft_center_range_apf",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "soft_center",
            "triangle_side": 1.20,
            "reference_max_vx": 0.09,
            "reference_max_vy": 0.34,
            "reference_max_vz": 0.14,
            "correction_scale": 0.035,
            "kp_center": 0.36,
            "kp_form": 0.78,
            "goal_bonus": 2200.0,
            "phase_bonus": 320.0,
            "progress_weight": 54.0,
            "formation_weight": 62.0,
            "spacing_weight": 50.0,
            "obstacle_collision_penalty": 4000.0,
            "obstacle_near_penalty_weight": 2.0,
            "obstacle_clearance_bonus": 14.0,
            "formation_near_obstacle_weight": 68.0,
            "spacing_near_obstacle_weight": 58.0,
            "formation_lock_gain": 4.50,
            "safety_filter_gain": 0.04,
            "neighbor_dropout_prob": 0.012,
            "range_max": 2.50,
            "range_safe": 1.35,
            "range_emergency": 0.28,
            "range_apf_gain": 0.02,
            "formation_apf_gain": 1.10,
            "max_apf_speed": 0.40,
            "tangential_apf_gain": 0.10,
            "formation_tangential_gain": 1.15,
            "tangential_clearance_bias": 0.35,
            "bypass_memory_enabled": True,
            "bypass_memory_trigger": 1.35,
            "bypass_memory_release": 2.05,
            "bypass_hold_steps": 260,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.20,
            "recovery_enabled": True,
            "recovery_steps": 260,
            "recovery_forward_boost": 0.13,
            "recovery_lateral_gain": 0.75,
            "recovery_apf_scale": 0.25,
            "recovery_clear_range": 1.75,
        },
        {
            "name": "medium_center_range_apf",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "soft_center_mid",
            "triangle_side": 1.20,
            "reference_max_vx": 0.10,
            "reference_max_vy": 0.36,
            "reference_max_vz": 0.14,
            "correction_scale": 0.045,
            "kp_center": 0.35,
            "kp_form": 0.80,
            "goal_bonus": 2250.0,
            "phase_bonus": 320.0,
            "progress_weight": 56.0,
            "formation_weight": 70.0,
            "spacing_weight": 58.0,
            "obstacle_collision_penalty": 4300.0,
            "obstacle_near_penalty_weight": 8.0,
            "obstacle_clearance_bonus": 14.0,
            "formation_near_obstacle_weight": 76.0,
            "spacing_near_obstacle_weight": 66.0,
            "formation_lock_gain": 4.50,
            "safety_filter_gain": 0.04,
            "neighbor_dropout_prob": 0.012,
            "range_max": 2.50,
            "range_safe": 1.30,
            "range_emergency": 0.24,
            "range_apf_gain": 0.02,
            "formation_apf_gain": 1.10,
            "max_apf_speed": 0.40,
            "tangential_apf_gain": 0.10,
            "formation_tangential_gain": 1.15,
            "tangential_clearance_bias": 0.35,
            "bypass_memory_enabled": True,
            "bypass_memory_trigger": 1.35,
            "bypass_memory_release": 2.05,
            "bypass_hold_steps": 260,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.20,
            "recovery_enabled": True,
            "recovery_steps": 260,
            "recovery_forward_boost": 0.13,
            "recovery_lateral_gain": 0.75,
            "recovery_apf_scale": 0.25,
            "recovery_clear_range": 1.75,
        },
        {
            "name": "single_center_range_apf",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "single_center",
            "triangle_side": 1.20,
            "reference_max_vx": 0.10,
            "reference_max_vy": 0.20,
            "reference_max_vz": 0.14,
            "correction_scale": 0.040,
            "kp_center": 0.34,
            "kp_form": 0.82,
            "goal_bonus": 2100.0,
            "phase_bonus": 280.0,
            "progress_weight": 46.0,
            "formation_weight": 85.0,
            "spacing_weight": 70.0,
            "obstacle_collision_penalty": 4400.0,
            "obstacle_near_penalty_weight": 3.0,
            "obstacle_clearance_bonus": 16.0,
            "formation_near_obstacle_weight": 90.0,
            "spacing_near_obstacle_weight": 80.0,
            "formation_lock_gain": 4.00,
            "safety_filter_gain": 0.08,
            "neighbor_dropout_prob": 0.010,
            "range_max": 2.50,
            "range_safe": 1.20,
            "range_emergency": 0.30,
            "range_apf_gain": 0.04,
            "formation_apf_gain": 0.90,
            "max_apf_speed": 0.30,
            "tangential_apf_gain": 0.08,
            "formation_tangential_gain": 0.90,
            "tangential_clearance_bias": 0.35,
            "bypass_memory_enabled": True,
            "bypass_memory_trigger": 1.45,
            "bypass_memory_release": 2.10,
            "bypass_hold_steps": 300,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.22,
            "recovery_enabled": True,
            "recovery_steps": 300,
            "recovery_forward_boost": 0.12,
            "recovery_lateral_gain": 0.70,
            "recovery_apf_scale": 0.25,
            "recovery_clear_range": 1.60,
        },
        {
            "name": "offset_gate_range_apf",
            "mission_stage": 3,
            "use_obstacles": True,
            "obstacle_layout": "offset_gate",
            "triangle_side": 1.20,
            "reference_max_vx": 0.12,
            "reference_max_vy": 0.24,
            "reference_max_vz": 0.14,
            "correction_scale": 0.045,
            "kp_center": 0.34,
            "kp_form": 0.82,
            "goal_bonus": 2300.0,
            "phase_bonus": 300.0,
            "progress_weight": 48.0,
            "formation_weight": 88.0,
            "spacing_weight": 72.0,
            "obstacle_collision_penalty": 4800.0,
            "obstacle_near_penalty_weight": 4.0,
            "obstacle_clearance_bonus": 18.0,
            "formation_near_obstacle_weight": 95.0,
            "spacing_near_obstacle_weight": 84.0,
            "formation_lock_gain": 4.20,
            "safety_filter_gain": 0.08,
            "neighbor_dropout_prob": 0.010,
            "range_max": 2.50,
            "range_safe": 1.20,
            "range_emergency": 0.30,
            "range_apf_gain": 0.05,
            "formation_apf_gain": 0.95,
            "max_apf_speed": 0.32,
            "tangential_apf_gain": 0.08,
            "formation_tangential_gain": 0.95,
            "tangential_clearance_bias": 0.40,
            "bypass_memory_enabled": True,
            "bypass_memory_trigger": 1.45,
            "bypass_memory_release": 2.10,
            "bypass_hold_steps": 300,
            "bypass_preferred_side": 1.0,
            "bypass_lateral_gain": 0.24,
            "recovery_enabled": True,
            "recovery_steps": 300,
            "recovery_forward_boost": 0.12,
            "recovery_lateral_gain": 0.70,
            "recovery_apf_scale": 0.25,
            "recovery_clear_range": 1.60,
        },
    ]

    # v0.11.2: softer center-obstacle bridge curriculum.
    # Previous Stage 4 (soft_center) was solved, but jumping directly to the
    # center obstacle collapsed the formation. We therefore remove the old
    # single medium bridge and insert several small lateral shifts toward the
    # center obstacle. The policy still receives only local range readings.
    base_soft = copy.deepcopy(curriculum[3])

    def make_center_bridge(name, layout, vx, vy, correction_scale, form_weight, spacing_weight,
                           near_weight, clear_bonus, lock_gain):
        stage = copy.deepcopy(base_soft)
        stage["name"] = name
        stage["obstacle_layout"] = layout
        stage["reference_max_vx"] = vx
        stage["reference_max_vy"] = vy
        stage["correction_scale"] = correction_scale
        stage["formation_weight"] = form_weight
        stage["spacing_weight"] = spacing_weight
        stage["obstacle_near_penalty_weight"] = near_weight
        stage["obstacle_clearance_bonus"] = clear_bonus
        stage["formation_near_obstacle_weight"] = form_weight + 12.0
        stage["spacing_near_obstacle_weight"] = spacing_weight + 10.0
        stage["formation_lock_gain"] = lock_gain
        stage["progress_weight"] = 56.0
        stage["goal_bonus"] = 2250.0
        stage["phase_bonus"] = 320.0
        return stage

    # Original curriculum indices:
    #   0 no obstacle, 1 easy offset, 2 side column, 3 soft center,
    #   4 old medium center, 5 single center, 6 offset gate.
    # v0.11.3 change:
    #   The previous v0.11.2 jump from soft_center_plus (y=0.82) to
    #   center_bridge_1 (y=0.70) was still too large and caused crash-heavy
    #   training. We now add micro-bridges and make advancement stricter.
    soft_center_plus = make_center_bridge(
        "soft_center_plus_range_apf", "soft_center_plus",
        vx=0.10, vy=0.36, correction_scale=0.045,
        form_weight=70.0, spacing_weight=58.0,
        near_weight=8.0, clear_bonus=16.0, lock_gain=4.20,
    )

    bridge_specs = [
        ("center_bridge_0_range_apf", "center_bridge_0", 0.100, 0.365, 0.050, 72.0, 60.0, 10.0, 17.0, 4.05),
        ("center_bridge_1_range_apf", "center_bridge_1", 0.098, 0.370, 0.052, 74.0, 62.0, 11.0, 17.5, 4.00),
        ("center_bridge_2_range_apf", "center_bridge_2", 0.096, 0.375, 0.055, 76.0, 64.0, 12.0, 18.0, 3.90),
        ("center_bridge_3_range_apf", "center_bridge_3", 0.094, 0.382, 0.058, 78.0, 66.0, 13.0, 18.5, 3.80),
        ("center_bridge_4_range_apf", "center_bridge_4", 0.092, 0.390, 0.061, 80.0, 68.0, 14.0, 19.0, 3.70),
        ("center_bridge_5_range_apf", "center_bridge_5", 0.090, 0.398, 0.064, 84.0, 72.0, 15.0, 20.0, 3.55),
        ("center_bridge_6_range_apf", "center_bridge_6", 0.088, 0.405, 0.067, 88.0, 76.0, 16.0, 21.0, 3.40),
        ("center_bridge_7_range_apf", "center_bridge_7", 0.086, 0.412, 0.070, 92.0, 80.0, 18.0, 22.0, 3.25),
        ("center_bridge_8_range_apf", "center_bridge_8", 0.084, 0.420, 0.073, 96.0, 84.0, 20.0, 23.0, 3.10),
    ]

    bridge_stages = [
        make_center_bridge(name, layout, vx, vy, corr, form, spacing, near, clear, lock)
        for name, layout, vx, vy, corr, form, spacing, near, clear, lock in bridge_specs
    ]

    single_center = copy.deepcopy(curriculum[5])
    single_center["reference_max_vx"] = 0.085
    single_center["reference_max_vy"] = 0.42
    single_center["correction_scale"] = 0.075
    single_center["formation_lock_gain"] = 3.00
    single_center["obstacle_near_penalty_weight"] = 20.0
    single_center["obstacle_clearance_bonus"] = 24.0
    single_center["progress_weight"] = 58.0

    offset_gate = copy.deepcopy(curriculum[6])
    offset_gate["reference_max_vx"] = 0.10
    offset_gate["reference_max_vy"] = 0.40
    offset_gate["correction_scale"] = 0.075

    curriculum = (
        curriculum[:4]
        + [soft_center_plus]
        + bridge_stages
        + [single_center, offset_gate]
    )

    # v0.11.0: range-based MARL with weak emergency APF only.
    # The policy still receives local range readings, but active APF navigation
    # is removed from the reference command. The policy must learn the bypass.
    for stage in curriculum:
        stage["name"] = stage["name"].replace("range_apf", "range_marl")

        # Disable active APF navigation and all hand-coded bypass helpers.
        stage["range_apf_gain"] = 0.0
        stage["formation_apf_gain"] = 0.0
        stage["tangential_apf_gain"] = 0.0
        stage["formation_tangential_gain"] = 0.0
        stage["tangential_clearance_bias"] = 0.0
        stage["bypass_memory_enabled"] = False
        stage["bypass_lateral_gain"] = 0.0
        stage["recovery_enabled"] = False
        stage["recovery_forward_boost"] = 0.0
        stage["recovery_lateral_gain"] = 0.0
        stage["recovery_apf_scale"] = 1.0

        # Keep a weak APF only for extreme emergency safety.
        # This does not plan around obstacles; it only prevents very close hits.
        if stage["use_obstacles"]:
            stage["range_emergency"] = 0.18
            stage["safety_filter_gain"] = 0.045
            stage["max_apf_speed"] = 0.16
        else:
            stage["range_emergency"] = 0.18
            stage["safety_filter_gain"] = 0.0
            stage["max_apf_speed"] = 0.0

        # Give the policy enough lateral authority to learn avoidance.
        if (
            stage["obstacle_layout"] in ["soft_center", "soft_center_plus", "single_center", "offset_gate"]
            or str(stage["obstacle_layout"]).startswith("center_bridge_")
        ):
            stage["reference_max_vx"] = min(stage["reference_max_vx"] + 0.05, 0.22)
            stage["reference_max_vy"] = max(stage["reference_max_vy"], 0.34)
            stage["correction_scale"] = max(stage["correction_scale"], 0.095)
            stage["formation_lock_gain"] = min(stage["formation_lock_gain"], 2.0)
            stage["obstacle_near_penalty_weight"] = max(stage["obstacle_near_penalty_weight"], 16.0)
            stage["obstacle_clearance_bonus"] = max(stage["obstacle_clearance_bonus"], 22.0)
            stage["progress_weight"] = max(stage["progress_weight"], 56.0)
        elif stage["use_obstacles"]:
            stage["correction_scale"] = max(stage["correction_scale"], 0.075)
            stage["reference_max_vy"] = max(stage["reference_max_vy"], 0.28)
            stage["obstacle_near_penalty_weight"] = max(stage["obstacle_near_penalty_weight"], 12.0)

    return curriculum


def train():
    curriculum = get_curriculum()

    rolling_window = 80
    # Stricter thresholds for bridge stages. The previous run advanced
    # from Stage 5 while crashes were still present, then collapsed in Stage 6.
    # Zero-based stage indices after v0.11.3:
    # 0 no obstacle, 1 easy offset, 2 side column, 3 soft center,
    # 4 soft_center_plus, 5-13 micro center bridges,
    # 14 single center, 15 offset gate.
    stage_success_thresholds = [
        0.85, 0.80, 0.75, 0.60,
        0.86,
        0.84, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70,
        0.68, 0.65,
    ]
    recent_successes = []
    recent_crashes = []
    curr_stage = int(np.clip(args.start_stage, 0, len(curriculum) - 1))

    ckpt_dir = "checkpoints/v0_11_0_range_marl_velocity_3uav"
    model_dir = "marl/models/v0_11_0_range_marl_velocity_3uav"
    log_dir = "results/logs"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    wandb_enabled = False

    if args.use_wandb:
        try:
            import wandb

            wandb.init(
                project="marl-uav-v0-11-0-range-marl-velocity-3uav",
                name=f"range-apf-optionD-{args.render}",
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
    print("Mission: v0.11.3 range-based MARL with micro-bridges and crash-gated curriculum")
    print("Policy obstacle input: local range readings only, no global obstacle coordinates")
    print("Active APF navigation: disabled; weak APF retained only as emergency safety")
    print("Curriculum: v0.11.3 soft-center-plus -> micro center bridges -> single-center")
    print("Curriculum: active APF navigation disabled; policy learns bypass from range observations")

    policy = SharedVelocityPolicyNet(obs_dim, act_dim).to(device)
    value = CentralValueNet(obs_dim_total).to(device)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_value = torch.optim.Adam(value.parameters(), lr=args.lr_value)

    if (not args.resume) and args.init_from_v010:
        warm_start_from_checkpoint(policy, value, args.init_from_v010)
        print(f"Starting v0.11.3 from curriculum stage {curr_stage + 1}: {stage_cfg['name']}")

    start_episode = 1
    best_score = -np.inf

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

    csv_path = os.path.join(log_dir, "training_logs_v0_11_0_range_marl_velocity_3uav.csv")

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
                    "formation_error",
                    "spacing_error",
                    "mean_speed",
                    "mean_cmd_speed",
                    "mean_correction",
                    "collision_count",
                    "min_pair_dist",
                    "min_obstacle_margin",
                    "obstacle_collision_count",
                    "episode_obstacle_collision_count",
                    "obstacle_near_miss_count",
                    "episode_near_miss_count",
                    "max_spacing_during_danger",
                    "max_shape_error_during_danger",
                    "obstacle_clear",
                    "crashed",
                    "formation_safe",
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
            global_obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                value_t = value(global_obs_t).item()

            local_obs_step = []
            actions_step = []
            logps_step = []

            for i in range(n_agents):
                obs_i = obs_split[i]
                obs_t = torch.tensor(obs_i, dtype=torch.float32, device=device).unsqueeze(0)

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
            next_global_obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
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

        global_obs_batch = torch.tensor(np.array(buf_global_obs), dtype=torch.float32, device=device)
        local_obs_batch = torch.tensor(np.array(buf_local_obs), dtype=torch.float32, device=device)
        action_batch = torch.tensor(np.array(buf_actions), dtype=torch.float32, device=device)
        old_logp_batch = torch.tensor(np.array(buf_logps), dtype=torch.float32, device=device)

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
            surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * adv_actor

            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = value(global_obs_batch)
            value_loss = (new_values - returns).pow(2).mean()

            entropy_term = entropy.mean()
            total_loss = policy_loss + args.vf_coef * value_loss - entropy_coef * entropy_term

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
        episode_near_miss_count = safe_float(last_info, "episode_near_miss_count")
        max_spacing_danger = safe_float(last_info, "max_spacing_during_danger")
        max_shape_danger = safe_float(last_info, "max_shape_error_during_danger")
        crashed = bool(last_info.get("crashed", False))
        phase = int(last_info.get("phase", 0))

        score = (
            14.0 * float(success)
            + phase * 4.0
            - center_error
            - 2.0 * formation_error
            - 2.0 * spacing_error
            - 4.0 * collision_count
            - 6.0 * obstacle_collision_count
            - 2.0 * obstacle_near_miss_count
            - max_spacing_danger
            - max_shape_danger
            - 4.0 * float(crashed)
        )

        if score > best_score:
            best_score = score

            torch.save(
                policy.state_dict(),
                os.path.join(model_dir, "best_shared_velocity_policy_v0_11_0_range_marl.pth"),
            )

            torch.save(
                value.state_dict(),
                os.path.join(model_dir, "best_central_value_v0_11_0_range_marl.pth"),
            )

        recent_successes.append(float(success))
        recent_crashes.append(float(crashed))

        if len(recent_successes) > rolling_window:
            recent_successes.pop(0)
        if len(recent_crashes) > rolling_window:
            recent_crashes.pop(0)

        current_threshold = stage_success_thresholds[curr_stage]

        formation_good = formation_error < 0.45
        spacing_good = spacing_error < 0.45

        # Stage-aware near-miss gate and crash gate.
        # v0.11.3 adds crash-rate gating so the curriculum cannot advance
        # while a stage still has occasional obstacle crashes.
        if curr_stage == 3:
            obstacle_near_miss_limit = 90.0
        elif curr_stage == 4:
            obstacle_near_miss_limit = 140.0
        elif 5 <= curr_stage <= 8:
            obstacle_near_miss_limit = 120.0
        elif 9 <= curr_stage <= 13:
            obstacle_near_miss_limit = 100.0
        elif curr_stage == 14:
            obstacle_near_miss_limit = 70.0
        else:
            obstacle_near_miss_limit = 40.0

        obstacle_good = episode_near_miss_count <= obstacle_near_miss_limit

        crash_rate = float(np.mean(recent_crashes)) if len(recent_crashes) > 0 else 0.0
        if curr_stage <= 3:
            max_crash_rate = 0.10
        elif curr_stage == 4:
            max_crash_rate = 0.02
        elif 5 <= curr_stage <= 13:
            max_crash_rate = 0.01
        else:
            max_crash_rate = 0.01

        crash_good = crash_rate <= max_crash_rate

        if ep % 25 == 0:
            rolling_success_rate_dbg = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0
            print(
                f"[Curriculum Check] "
                f"Stage={curr_stage + 1} | "
                f"Len={len(recent_successes)}/{rolling_window} | "
                f"RollSR={rolling_success_rate_dbg:.2f}/{current_threshold:.2f} | "
                f"FormGood={formation_good} ({formation_error:.3f}) | "
                f"SpacingGood={spacing_good} ({spacing_error:.3f}) | "
                f"NearMissNow={obstacle_near_miss_count:.1f} | "
                f"EpisodeNearMiss={episode_near_miss_count:.1f} | "
                f"NearMissLimit={obstacle_near_miss_limit:.1f} | "
                f"ObstacleGood={obstacle_good} | "
                f"CrashRate={crash_rate:.2f}/{max_crash_rate:.2f} | "
                f"CrashGood={crash_good}"
            )

        if (
            len(recent_successes) == rolling_window
            and np.mean(recent_successes) >= current_threshold
            and formation_good
            and spacing_good
            and obstacle_good
            and crash_good
            and curr_stage < len(curriculum) - 1
        ):
            curr_stage += 1
            stage_cfg = curriculum[curr_stage]

            env.close()
            env = build_env(stage_cfg)

            recent_successes = []
            recent_crashes = []

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
                    "mean_speed": safe_float(last_info, "mean_speed"),
                    "mean_cmd_speed": safe_float(last_info, "mean_cmd_speed"),
                    "mean_correction": safe_float(last_info, "mean_correction"),
                    "collision_count": collision_count,
                    "min_pair_dist": safe_float(last_info, "min_pair_dist"),
                    "min_obstacle_margin": safe_float(last_info, "min_obstacle_margin"),
                    "obstacle_collision_count": obstacle_collision_count,
                    "episode_obstacle_collision_count": safe_float(last_info, "episode_obstacle_collision_count"),
                    "obstacle_near_miss_count": obstacle_near_miss_count,
                    "episode_near_miss_count": episode_near_miss_count,
                    "max_spacing_during_danger": max_spacing_danger,
                    "max_shape_error_during_danger": max_shape_danger,
                    "obstacle_clear": float(bool(last_info.get("obstacle_clear", False))),
                    "crashed": float(crashed),
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
                    last_info.get("formation_error", 0.0),
                    last_info.get("spacing_error", 0.0),
                    last_info.get("mean_speed", 0.0),
                    last_info.get("mean_cmd_speed", 0.0),
                    last_info.get("mean_correction", 0.0),
                    last_info.get("collision_count", 0),
                    last_info.get("min_pair_dist", 0.0),
                    last_info.get("min_obstacle_margin", 0.0),
                    last_info.get("obstacle_collision_count", 0),
                    last_info.get("episode_obstacle_collision_count", 0),
                    last_info.get("obstacle_near_miss_count", 0),
                    last_info.get("episode_near_miss_count", 0),
                    last_info.get("max_spacing_during_danger", 0.0),
                    last_info.get("max_shape_error_during_danger", 0.0),
                    int(last_info.get("obstacle_clear", False)),
                    int(last_info.get("crashed", False)),
                    int(last_info.get("formation_safe", False)),
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
                f"ObsMargin: {safe_float(last_info, 'min_obstacle_margin'):.3f} | "
                f"ObsColl: {int(obstacle_collision_count)} | "
                f"NearMiss: {int(obstacle_near_miss_count)} | "
                f"Crashed: {crashed}"
            )

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