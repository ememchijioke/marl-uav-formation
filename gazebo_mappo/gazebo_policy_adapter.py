#!/usr/bin/env python3

import csv
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Gazebo / Policy configuration
# ============================================================

WORLD_NAME = "industrial_three_crazyflie_world"

MODEL_PATH = "checkpoints/v0_8_5_triangle_velocity_3uav/shared_velocity_policy.pth"

RESULT_DIR = "results/gazebo_validation"
RESULT_CSV = os.path.join(RESULT_DIR, "gazebo_policy_adapter_validation.csv")

UAV_NAMES = ["uav1", "uav2", "uav3"]
N_AGENTS = 3

OBS_DIM_PER_AGENT = 36
ACT_DIM_PER_AGENT = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Mission mapping
# PyBullet policy was trained with start_x=0.0 and goal_x=7.0.
# Gazebo visual world uses approximately start_x=-2.4 and goal_x=2.4.
# ============================================================

PYBULLET_START_X = 0.0
PYBULLET_GOAL_X = 7.0

GAZEBO_START_X = -2.4
GAZEBO_GOAL_X = 2.4

CENTER_Y = 0.0

GROUND_Z = 0.35
TARGET_ALTITUDE = 1.0
LANDING_Z = 0.35

MAX_STEPS = 450
STEP_SLEEP = 0.02

CTRL_DT = 1.0 / 24.0

REFERENCE_MAX_VEL = 0.65
CORRECTION_SCALE = 0.10

KP_CENTER = 0.70
KP_FORM = 1.10
KP_LANDING = 0.45

MIN_SEPARATION_LIMIT = 0.50


# ============================================================
# Observation normalization constants
# These follow the structure in envs/quad_env_velocity.py::_build_obs
# ============================================================

GOAL_SCALE = 7.0
NEIGHBOR_POS_SCALE = 3.0
NEIGHBOR_VEL_SCALE = 2.0
REFERENCE_VEL_SCALE = 1.0
ALTITUDE_SCALE = 1.0
FORMATION_ERROR_SCALE = 1.5
SPACING_ERROR_SCALE = 1.0


# ============================================================
# Triangle formation geometry
# UAV 1 = front, UAV 2 = rear-left, UAV 3 = rear-right
# ============================================================

FORMATION_OFFSETS = np.array(
    [
        [0.60, 0.00, 0.0],
        [-0.35, 0.60, 0.0],
        [-0.35, -0.60, 0.0],
    ],
    dtype=np.float32,
)


# ============================================================
# Policy network: must match marl/eval_playback_velocity.py
# ============================================================

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


# ============================================================
# Gazebo service helpers
# ============================================================

def run_gz_service(
    service: str,
    reqtype: str,
    reptype: str,
    req: str,
    timeout_ms: int = 1500,
    silent: bool = True,
) -> bool:
    cmd = [
        "gz",
        "service",
        "-s",
        service,
        "--reqtype",
        reqtype,
        "--reptype",
        reptype,
        "--timeout",
        str(timeout_ms),
        "--req",
        req,
    ]

    if silent:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        result = subprocess.run(cmd)

    return result.returncode == 0


def set_gravity(enabled: bool) -> bool:
    gz_value = -9.8 if enabled else 0.0

    req = (
        f"gravity {{ x: 0.0 y: 0.0 z: {gz_value} }} "
        "max_step_size: 0.001 "
        "real_time_factor: 1.0"
    )

    service = f"/world/{WORLD_NAME}/set_physics"

    ok = run_gz_service(
        service=service,
        reqtype="gz.msgs.Physics",
        reptype="gz.msgs.Boolean",
        req=req,
    )

    print(f"[Gazebo] Gravity {'enabled' if enabled else 'disabled'}: {'OK' if ok else 'FAILED'}")
    return ok


def build_pose_vector_request(poses: Dict[str, Tuple[float, float, float]]) -> str:
    parts = []

    for name in UAV_NAMES:
        x, y, z = poses[name]
        parts.append(
            "pose { "
            f'name: "{name}" '
            f"position {{ x: {x:.6f} y: {y:.6f} z: {z:.6f} }} "
            "orientation { x: 0.0 y: 0.0 z: 0.0 w: 1.0 } "
            "}"
        )

    return " ".join(parts)


def set_uav_poses(poses: Dict[str, Tuple[float, float, float]]) -> bool:
    service = f"/world/{WORLD_NAME}/set_pose_vector"
    req = build_pose_vector_request(poses)

    return run_gz_service(
        service=service,
        reqtype="gz.msgs.Pose_V",
        reptype="gz.msgs.Boolean",
        req=req,
    )


# ============================================================
# Coordinate mapping
# ============================================================

def map_pybullet_x_to_gazebo_x(x_policy: float) -> float:
    alpha = (x_policy - PYBULLET_START_X) / (PYBULLET_GOAL_X - PYBULLET_START_X)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return GAZEBO_START_X + alpha * (GAZEBO_GOAL_X - GAZEBO_START_X)


def policy_positions_to_gazebo_poses(
    positions_policy: np.ndarray,
) -> Dict[str, Tuple[float, float, float]]:
    poses = {}

    for i, name in enumerate(UAV_NAMES):
        x_p, y_p, z_p = positions_policy[i]

        x_g = map_pybullet_x_to_gazebo_x(float(x_p))
        y_g = float(y_p)
        z_g = float(np.clip(z_p, 0.20, 1.35))

        poses[name] = (x_g, y_g, z_g)

    return poses


# ============================================================
# Mission / observation logic
# ============================================================

def target_center_for_phase(phase: int, positions: np.ndarray) -> np.ndarray:
    if phase == 0:
        return np.array([PYBULLET_START_X, CENTER_Y, TARGET_ALTITUDE], dtype=np.float32)

    if phase == 1:
        return np.array([PYBULLET_START_X, CENTER_Y, TARGET_ALTITUDE], dtype=np.float32)

    if phase == 2:
        center = np.mean(positions, axis=0)
        progress_x = min(center[0] + 0.55, PYBULLET_GOAL_X)
        return np.array([progress_x, CENTER_Y, TARGET_ALTITUDE], dtype=np.float32)

    if phase == 3:
        return np.array([PYBULLET_GOAL_X, CENTER_Y, TARGET_ALTITUDE], dtype=np.float32)

    return np.array([PYBULLET_GOAL_X, CENTER_Y, LANDING_Z], dtype=np.float32)


def desired_positions_for_phase(phase: int, positions: np.ndarray) -> np.ndarray:
    center = target_center_for_phase(phase, positions)
    return center[None, :] + FORMATION_OFFSETS


def desired_pair_distances() -> Dict[Tuple[int, int], float]:
    distances = {}

    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            distances[(i, j)] = float(
                np.linalg.norm(FORMATION_OFFSETS[i] - FORMATION_OFFSETS[j])
            )

    return distances


EXPECTED_PAIR_DISTANCES = desired_pair_distances()


def compute_spacing_error(positions: np.ndarray) -> float:
    errors = []

    for (i, j), expected in EXPECTED_PAIR_DISTANCES.items():
        d = float(np.linalg.norm(positions[i] - positions[j]))
        errors.append(abs(d - expected))

    return float(np.mean(errors))


def compute_formation_error(positions: np.ndarray, desired_positions: np.ndarray) -> float:
    per_agent = np.linalg.norm(desired_positions - positions, axis=1)
    return float(np.mean(per_agent))


def compute_min_separation(positions: np.ndarray) -> float:
    min_sep = float("inf")

    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            min_sep = min(min_sep, d)

    return float(min_sep)


def compute_reference_velocity(
    phase: int,
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    desired_positions = desired_positions_for_phase(phase, positions)
    formation_tracking_error = desired_positions - positions

    reference_velocity = np.zeros((N_AGENTS, 3), dtype=np.float32)

    for i in range(N_AGENTS):
        if phase in [0, 1, 2, 3]:
            cmd = KP_FORM * formation_tracking_error[i]
        else:
            cmd = KP_LANDING * formation_tracking_error[i]

        speed = float(np.linalg.norm(cmd))
        if speed > REFERENCE_MAX_VEL:
            cmd = cmd / (speed + 1e-8) * REFERENCE_MAX_VEL

        reference_velocity[i] = cmd.astype(np.float32)

    return reference_velocity, formation_tracking_error.astype(np.float32)


def build_obs(
    phase: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    prev_action: np.ndarray,
    reference_velocity: np.ndarray,
    formation_tracking_error: np.ndarray,
) -> np.ndarray:
    target_center = target_center_for_phase(phase, positions)
    desired_positions = desired_positions_for_phase(phase, positions)

    spacing_error = np.array([compute_spacing_error(positions)], dtype=np.float32)

    phase_onehot = np.zeros(5, dtype=np.float32)
    phase_onehot[phase] = 1.0

    obs_parts = []

    for i in range(N_AGENTS):
        own_pos = positions[i]
        own_vel = velocities[i]

        rel_target = desired_positions[i] - own_pos
        altitude_error = np.array([target_center[2] - own_pos[2]], dtype=np.float32)

        neighbor_parts = []

        for j in range(N_AGENTS):
            if j == i:
                continue

            rel_neighbor_pos = positions[j] - positions[i]
            rel_neighbor_vel = velocities[j] - velocities[i]
            neighbor_valid = np.array([1.0], dtype=np.float32)

            rel_neighbor_pos_norm = np.clip(
                rel_neighbor_pos / NEIGHBOR_POS_SCALE,
                -1.0,
                1.0,
            ).astype(np.float32)

            rel_neighbor_vel_norm = np.clip(
                rel_neighbor_vel / NEIGHBOR_VEL_SCALE,
                -1.0,
                1.0,
            ).astype(np.float32)

            neighbor_parts.extend(
                [
                    rel_neighbor_pos_norm,
                    rel_neighbor_vel_norm,
                    neighbor_valid,
                ]
            )

        rel_target_norm = np.clip(rel_target / GOAL_SCALE, -1.0, 1.0).astype(np.float32)
        own_vel_norm = np.clip(own_vel / REFERENCE_VEL_SCALE, -1.0, 1.0).astype(np.float32)
        altitude_error_norm = np.clip(
            altitude_error / ALTITUDE_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)

        reference_velocity_norm = np.clip(
            reference_velocity[i] / REFERENCE_VEL_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)

        formation_tracking_error_norm = np.clip(
            formation_tracking_error[i] / FORMATION_ERROR_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)

        spacing_error_norm = np.clip(
            spacing_error / SPACING_ERROR_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)

        obs_i = np.concatenate(
            [
                rel_target_norm,
                own_vel_norm,
                altitude_error_norm,
                phase_onehot.astype(np.float32),
                neighbor_parts[0],
                neighbor_parts[1],
                neighbor_parts[2],
                neighbor_parts[3],
                neighbor_parts[4],
                neighbor_parts[5],
                prev_action[i].astype(np.float32),
                reference_velocity_norm,
                formation_tracking_error_norm,
                spacing_error_norm,
            ],
            axis=0,
        ).astype(np.float32)

        if obs_i.shape[0] != OBS_DIM_PER_AGENT:
            raise RuntimeError(
                f"Bad obs dim for agent {i}: got {obs_i.shape[0]}, expected {OBS_DIM_PER_AGENT}"
            )

        obs_parts.append(obs_i)

    return np.concatenate(obs_parts, axis=0).astype(np.float32)


def update_phase(
    phase: int,
    phase_counter: int,
    positions: np.ndarray,
) -> Tuple[int, int]:
    center = np.mean(positions, axis=0)
    desired = desired_positions_for_phase(phase, positions)

    formation_error = compute_formation_error(positions, desired)
    spacing_error = compute_spacing_error(positions)

    stable_formation = formation_error < 0.45 and spacing_error < 0.30

    if phase == 0:
        if abs(center[2] - TARGET_ALTITUDE) < 0.15 and stable_formation:
            phase_counter += 1
        else:
            phase_counter = 0

        if phase_counter >= 12:
            return 1, 0

    elif phase == 1:
        if stable_formation:
            phase_counter += 1
        else:
            phase_counter = 0

        if phase_counter >= 10:
            return 2, 0

    elif phase == 2:
        if abs(center[0] - PYBULLET_GOAL_X) < 0.45 and stable_formation:
            phase_counter += 1
        else:
            phase_counter = 0

        if phase_counter >= 12:
            return 3, 0

    elif phase == 3:
        if stable_formation:
            phase_counter += 1
        else:
            phase_counter = 0

        if phase_counter >= 15:
            return 4, 0

    elif phase == 4:
        phase_counter += 1

    return phase, phase_counter


# ============================================================
# Policy loading / inference
# ============================================================

def load_policy() -> SharedVelocityPolicyNet:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Policy checkpoint not found: {MODEL_PATH}")

    policy = SharedVelocityPolicyNet(
        obs_dim=OBS_DIM_PER_AGENT,
        act_dim=ACT_DIM_PER_AGENT,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        clean_state[k] = v

    policy.load_state_dict(clean_state)
    policy.eval()

    return policy


def policy_action(policy: SharedVelocityPolicyNet, obs_flat: np.ndarray) -> np.ndarray:
    obs_agents = np.split(obs_flat, N_AGENTS)
    actions = []

    with torch.no_grad():
        for obs_i in obs_agents:
            obs_tensor = torch.tensor(
                obs_i,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            action_i = policy(obs_tensor).squeeze(0).cpu().numpy()
            actions.append(action_i.astype(np.float32))

    return np.stack(actions, axis=0).astype(np.float32)


# ============================================================
# Logging
# ============================================================

def phase_name(phase: int) -> str:
    names = {
        0: "TAKEOFF",
        1: "START_HOVER",
        2: "TRIANGLE_FLIGHT",
        3: "GOAL_HOVER",
        4: "LANDING",
    }
    return names.get(phase, "UNKNOWN")


def build_row(
    step: int,
    phase: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    success: bool = False,
) -> Dict[str, float]:
    center = np.mean(positions, axis=0)
    desired = desired_positions_for_phase(phase, positions)

    target_center = target_center_for_phase(phase, positions)
    goal_error = float(np.linalg.norm(center - target_center))
    formation_error = compute_formation_error(positions, desired)
    spacing_error = compute_spacing_error(positions)
    min_separation = compute_min_separation(positions)
    action_norm = float(np.mean(np.linalg.norm(actions, axis=1)))

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "world_name": WORLD_NAME,
        "step": step,
        "phase": phase,
        "phase_name": phase_name(phase),

        "center_x_policy": center[0],
        "center_y_policy": center[1],
        "center_z_policy": center[2],

        "target_x_policy": target_center[0],
        "target_y_policy": target_center[1],
        "target_z_policy": target_center[2],

        "goal_error": goal_error,
        "formation_error": formation_error,
        "spacing_error": spacing_error,
        "min_separation": min_separation,
        "action_norm": action_norm,
        "success": int(success),
    }

    for i, name in enumerate(UAV_NAMES):
        row[f"{name}_x_policy"] = positions[i, 0]
        row[f"{name}_y_policy"] = positions[i, 1]
        row[f"{name}_z_policy"] = positions[i, 2]

        row[f"{name}_vx_policy"] = velocities[i, 0]
        row[f"{name}_vy_policy"] = velocities[i, 1]
        row[f"{name}_vz_policy"] = velocities[i, 2]

        row[f"{name}_ax"] = actions[i, 0]
        row[f"{name}_ay"] = actions[i, 1]
        row[f"{name}_az"] = actions[i, 2]

    return row


def write_csv(rows: List[Dict[str, float]]) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)

    fieldnames = [
        "timestamp",
        "world_name",
        "step",
        "phase",
        "phase_name",

        "center_x_policy",
        "center_y_policy",
        "center_z_policy",

        "target_x_policy",
        "target_y_policy",
        "target_z_policy",

        "goal_error",
        "formation_error",
        "spacing_error",
        "min_separation",
        "action_norm",
        "success",
    ]

    for name in UAV_NAMES:
        fieldnames.extend(
            [
                f"{name}_x_policy",
                f"{name}_y_policy",
                f"{name}_z_policy",
                f"{name}_vx_policy",
                f"{name}_vy_policy",
                f"{name}_vz_policy",
                f"{name}_ax",
                f"{name}_ay",
                f"{name}_az",
            ]
        )

    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\n[Results] Saved CSV: {RESULT_CSV}")


def evaluate_success(rows: List[Dict[str, float]]) -> bool:
    if not rows:
        return False

    last = rows[-1]

    final_phase = int(last["phase"])
    goal_error = float(last["goal_error"])
    formation_error = float(last["formation_error"])
    spacing_error = float(last["spacing_error"])
    min_separation = float(last["min_separation"])
    center_z = float(last["center_z_policy"])

    return bool(
        final_phase == 4
        and goal_error < 0.30
        and formation_error < 0.45
        and spacing_error < 0.30
        and min_separation > MIN_SEPARATION_LIMIT
        and center_z <= 0.55
    )


# ============================================================
# Main
# ============================================================

def main():
    print("============================================================")
    print(" Gazebo Policy Adapter: PyBullet MAPPO Velocity Policy")
    print("============================================================")
    print(f"World: {WORLD_NAME}")
    print(f"Policy: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Obs dim per agent: {OBS_DIM_PER_AGENT}")
    print(f"Action dim per agent: {ACT_DIM_PER_AGENT}")
    print("Action meaning: learned velocity correction [rx, ry, rz]")
    print("")

    print("Make sure Gazebo is already running:")
    print(f"  gz sim -v 4 gazebo_mappo/worlds/{WORLD_NAME}.sdf")
    print("")

    policy = load_policy()
    print("[Policy] Loaded successfully.")

    print("[Gazebo] Disabling gravity for proxy validation...")
    set_gravity(enabled=False)

    # Initial PyBullet-policy coordinate positions
    positions = np.array(
        [
            [PYBULLET_START_X + FORMATION_OFFSETS[0, 0], FORMATION_OFFSETS[0, 1], GROUND_Z],
            [PYBULLET_START_X + FORMATION_OFFSETS[1, 0], FORMATION_OFFSETS[1, 1], GROUND_Z],
            [PYBULLET_START_X + FORMATION_OFFSETS[2, 0], FORMATION_OFFSETS[2, 1], GROUND_Z],
        ],
        dtype=np.float32,
    )

    velocities = np.zeros((N_AGENTS, 3), dtype=np.float32)
    prev_action = np.zeros((N_AGENTS, 3), dtype=np.float32)

    phase = 0
    phase_counter = 0

    rows = []

    print("\n[Mission] Starting Gazebo policy adapter playback...\n")

    for step in range(1, MAX_STEPS + 1):
        reference_velocity, formation_tracking_error = compute_reference_velocity(
            phase=phase,
            positions=positions,
        )

        obs_flat = build_obs(
            phase=phase,
            positions=positions,
            velocities=velocities,
            prev_action=prev_action,
            reference_velocity=reference_velocity,
            formation_tracking_error=formation_tracking_error,
        )

        actions = policy_action(policy, obs_flat)
        actions = np.clip(actions, -1.0, 1.0)

        correction_velocity = CORRECTION_SCALE * actions
        command_velocity = reference_velocity + correction_velocity

        speeds = np.linalg.norm(command_velocity, axis=1)
        for i in range(N_AGENTS):
            max_speed = REFERENCE_MAX_VEL + CORRECTION_SCALE
            if speeds[i] > max_speed:
                command_velocity[i] = command_velocity[i] / (speeds[i] + 1e-8) * max_speed

        velocities = command_velocity.astype(np.float32)
        positions = positions + velocities * CTRL_DT

        # Keep motion inside reasonable policy bounds
        positions[:, 0] = np.clip(positions[:, 0], -0.5, PYBULLET_GOAL_X + 0.6)
        positions[:, 1] = np.clip(positions[:, 1], -2.0, 2.0)

        if phase == 4:
            positions[:, 2] = np.clip(positions[:, 2], LANDING_Z, TARGET_ALTITUDE + 0.15)
        else:
            positions[:, 2] = np.clip(positions[:, 2], GROUND_Z, TARGET_ALTITUDE + 0.20)

        prev_action = actions.copy()

        gazebo_poses = policy_positions_to_gazebo_poses(positions)
        ok = set_uav_poses(gazebo_poses)

        row = build_row(
            step=step,
            phase=phase,
            positions=positions,
            velocities=velocities,
            actions=actions,
            success=False,
        )

        rows.append(row)

        print(
            f"[{step:03d}] "
            f"Phase={phase_name(phase):15s} | "
            f"Center=({row['center_x_policy']:.2f}, {row['center_y_policy']:.2f}, {row['center_z_policy']:.2f}) | "
            f"GoalErr={row['goal_error']:.3f} | "
            f"FormErr={row['formation_error']:.3f} | "
            f"SpacingErr={row['spacing_error']:.3f} | "
            f"MinSep={row['min_separation']:.3f} | "
            f"ActNorm={row['action_norm']:.3f} | "
            f"Gazebo={'OK' if ok else 'FAILED'}"
        )

        phase, phase_counter = update_phase(
            phase=phase,
            phase_counter=phase_counter,
            positions=positions,
        )

        if phase == 4 and row["center_z_policy"] <= 0.55 and step > 80:
            break

        if STEP_SLEEP > 0:
            time.sleep(STEP_SLEEP)

    success = evaluate_success(rows)

    if rows:
        rows[-1]["success"] = int(success)

    write_csv(rows)

    print("\n============================================================")
    print(" Gazebo Policy Adapter Summary")
    print("============================================================")
    print(f"Success: {success}")

    if rows:
        last = rows[-1]
        print(f"Final phase: {last['phase_name']}")
        print(
            f"Final policy center: "
            f"({last['center_x_policy']:.3f}, "
            f"{last['center_y_policy']:.3f}, "
            f"{last['center_z_policy']:.3f})"
        )
        print(f"Final goal error: {last['goal_error']:.3f}")
        print(f"Final formation error: {last['formation_error']:.3f}")
        print(f"Final spacing error: {last['spacing_error']:.3f}")
        print(f"Final min separation: {last['min_separation']:.3f}")
        print(f"Final action norm: {last['action_norm']:.3f}")

    print("\n[Cleanup] Re-enabling gravity...")
    set_gravity(enabled=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
