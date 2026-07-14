#!/usr/bin/env python3

import csv
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np


# ============================================================
# Official Gazebo validation world
# ============================================================

WORLD_NAME = "industrial_three_crazyflie_world"

RESULT_DIR = "results/gazebo_validation"
RESULT_CSV = os.path.join(RESULT_DIR, "gazebo_kinematic_validation.csv")


# ============================================================
# Fast stable demo settings
# ============================================================

TAKEOFF_STEPS = 5
START_HOVER_STEPS = 1
FORWARD_FLIGHT_STEPS = 10
GOAL_HOVER_STEPS = 1
LANDING_STEPS = 5
FINAL_HOLD_STEPS = 1

STEP_SLEEP = 0.0

START_CENTER_X = -2.4
GOAL_CENTER_X = 2.4
CENTER_Y = 0.0

GROUND_Z = 0.35
HOVER_Z = 1.35

TRIANGLE_OFFSETS = {
    "uav1": (0.60, 0.00),
    "uav2": (-0.35, 0.60),
    "uav3": (-0.35, -0.60),
}

UAV_NAMES = ["uav1", "uav2", "uav3"]

EXPECTED_SIDE_LENGTHS = {
    ("uav1", "uav2"): None,
    ("uav1", "uav3"): None,
    ("uav2", "uav3"): None,
}

SUCCESS_GOAL_ERROR_THRESHOLD = 0.20
SUCCESS_FORMATION_ERROR_THRESHOLD = 0.08
SUCCESS_MIN_SEPARATION_THRESHOLD = 0.50
SUCCESS_LANDING_Z_THRESHOLD = 0.45


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
    """
    Calls a Gazebo service using the gz CLI.
    Returns True if the command exits cleanly.
    """

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
    """
    Disables or enables gravity in the active Gazebo world.
    Gravity is disabled during the kinematic validation mission.
    """

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
        timeout_ms=1500,
    )

    state = "enabled" if enabled else "disabled"
    print(f"[Gazebo] Gravity {state}: {'OK' if ok else 'FAILED'}")

    return ok


def build_pose_vector_request(
    poses: Dict[str, Tuple[float, float, float]]
) -> str:
    """
    Builds gz.msgs.Pose_V text request for set_pose_vector.
    """

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


def set_uav_poses(
    poses: Dict[str, Tuple[float, float, float]]
) -> bool:
    """
    Moves all proxy UAVs at once using Gazebo set_pose_vector.
    """

    service = f"/world/{WORLD_NAME}/set_pose_vector"
    req = build_pose_vector_request(poses)

    return run_gz_service(
        service=service,
        reqtype="gz.msgs.Pose_V",
        reptype="gz.msgs.Boolean",
        req=req,
        timeout_ms=1500,
    )


# ============================================================
# Mission geometry helpers
# ============================================================

def initialize_expected_side_lengths() -> None:
    """
    Computes expected pairwise distances from the triangle offsets.
    """

    offset_vectors = {}

    for name, (ox, oy) in TRIANGLE_OFFSETS.items():
        offset_vectors[name] = np.array([ox, oy, 0.0], dtype=np.float64)

    for pair in EXPECTED_SIDE_LENGTHS.keys():
        a, b = pair
        EXPECTED_SIDE_LENGTHS[pair] = float(
            np.linalg.norm(offset_vectors[a] - offset_vectors[b])
        )


def formation_poses(
    center_x: float,
    center_y: float,
    center_z: float,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Returns UAV poses for a given formation center.
    """

    poses = {}

    for name, (ox, oy) in TRIANGLE_OFFSETS.items():
        poses[name] = (
            center_x + ox,
            center_y + oy,
            center_z,
        )

    return poses


def compute_center(
    poses: Dict[str, Tuple[float, float, float]]
) -> np.ndarray:
    arr = np.array([poses[name] for name in UAV_NAMES], dtype=np.float64)
    return np.mean(arr, axis=0)


def compute_goal_error(center: np.ndarray, phase: str) -> float:
    """
    Goal error is measured to the correct phase target.
    """

    if phase in ["LANDING", "FINAL_HOLD"]:
        target = np.array([GOAL_CENTER_X, CENTER_Y, GROUND_Z], dtype=np.float64)
    else:
        target = np.array([GOAL_CENTER_X, CENTER_Y, HOVER_Z], dtype=np.float64)

    return float(np.linalg.norm(center - target))


def compute_formation_error(
    poses: Dict[str, Tuple[float, float, float]]
) -> float:
    """
    Mean absolute pairwise spacing error relative to desired triangle geometry.
    """

    errors = []

    for pair, expected in EXPECTED_SIDE_LENGTHS.items():
        a, b = pair
        pa = np.array(poses[a], dtype=np.float64)
        pb = np.array(poses[b], dtype=np.float64)

        d = float(np.linalg.norm(pa - pb))
        errors.append(abs(d - expected))

    return float(np.mean(errors))


def compute_min_separation(
    poses: Dict[str, Tuple[float, float, float]]
) -> float:
    """
    Minimum pairwise separation between UAVs.
    """

    min_sep = float("inf")

    for i in range(len(UAV_NAMES)):
        for j in range(i + 1, len(UAV_NAMES)):
            a = UAV_NAMES[i]
            b = UAV_NAMES[j]

            pa = np.array(poses[a], dtype=np.float64)
            pb = np.array(poses[b], dtype=np.float64)

            d = float(np.linalg.norm(pa - pb))
            min_sep = min(min_sep, d)

    return float(min_sep)


def build_row(
    mission_step: int,
    phase: str,
    poses: Dict[str, Tuple[float, float, float]],
    success: bool = False,
) -> Dict[str, float]:
    center = compute_center(poses)
    goal_error = compute_goal_error(center, phase)
    formation_error = compute_formation_error(poses)
    min_separation = compute_min_separation(poses)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "world_name": WORLD_NAME,
        "mission_step": mission_step,
        "phase": phase,

        "uav1_x": poses["uav1"][0],
        "uav1_y": poses["uav1"][1],
        "uav1_z": poses["uav1"][2],

        "uav2_x": poses["uav2"][0],
        "uav2_y": poses["uav2"][1],
        "uav2_z": poses["uav2"][2],

        "uav3_x": poses["uav3"][0],
        "uav3_y": poses["uav3"][1],
        "uav3_z": poses["uav3"][2],

        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],

        "goal_error": goal_error,
        "formation_error": formation_error,
        "min_separation": min_separation,
        "success": int(success),
    }

    return row


def write_csv(rows: List[Dict[str, float]]) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)

    fieldnames = [
        "timestamp",
        "world_name",
        "mission_step",
        "phase",

        "uav1_x", "uav1_y", "uav1_z",
        "uav2_x", "uav2_y", "uav2_z",
        "uav3_x", "uav3_y", "uav3_z",

        "center_x",
        "center_y",
        "center_z",

        "goal_error",
        "formation_error",
        "min_separation",
        "success",
    ]

    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\n[Results] Saved validation CSV:")
    print(f"  {RESULT_CSV}")


# ============================================================
# Mission generator
# ============================================================

def interpolate(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def generate_mission_sequence() -> List[Tuple[str, Dict[str, Tuple[float, float, float]]]]:
    """
    Generates the kinematic validation mission sequence.

    Mission:
      1. takeoff
      2. start hover
      3. forward triangle flight
      4. goal hover
      5. landing
      6. final hold
    """

    sequence = []

    # Takeoff at start center.
    for i in range(TAKEOFF_STEPS):
        t = (i + 1) / TAKEOFF_STEPS
        z = interpolate(GROUND_Z, HOVER_Z, t)
        poses = formation_poses(START_CENTER_X, CENTER_Y, z)
        sequence.append(("TAKEOFF", poses))

    # Start hover.
    for _ in range(START_HOVER_STEPS):
        poses = formation_poses(START_CENTER_X, CENTER_Y, HOVER_Z)
        sequence.append(("START_HOVER", poses))

    # Forward flight.
    for i in range(FORWARD_FLIGHT_STEPS):
        t = (i + 1) / FORWARD_FLIGHT_STEPS
        x = interpolate(START_CENTER_X, GOAL_CENTER_X, t)
        poses = formation_poses(x, CENTER_Y, HOVER_Z)
        sequence.append(("FORWARD_FLIGHT", poses))

    # Goal hover.
    for _ in range(GOAL_HOVER_STEPS):
        poses = formation_poses(GOAL_CENTER_X, CENTER_Y, HOVER_Z)
        sequence.append(("GOAL_HOVER", poses))

    # Landing.
    for i in range(LANDING_STEPS):
        t = (i + 1) / LANDING_STEPS
        z = interpolate(HOVER_Z, GROUND_Z, t)
        poses = formation_poses(GOAL_CENTER_X, CENTER_Y, z)
        sequence.append(("LANDING", poses))

    # Final hold.
    for _ in range(FINAL_HOLD_STEPS):
        poses = formation_poses(GOAL_CENTER_X, CENTER_Y, GROUND_Z)
        sequence.append(("FINAL_HOLD", poses))

    return sequence


def evaluate_success(last_row: Dict[str, float]) -> bool:
    """
    Success condition for the Gazebo kinematic validation.
    """

    goal_error = float(last_row["goal_error"])
    formation_error = float(last_row["formation_error"])
    min_separation = float(last_row["min_separation"])
    center_z = float(last_row["center_z"])

    success = (
        goal_error <= SUCCESS_GOAL_ERROR_THRESHOLD
        and formation_error <= SUCCESS_FORMATION_ERROR_THRESHOLD
        and min_separation >= SUCCESS_MIN_SEPARATION_THRESHOLD
        and center_z <= SUCCESS_LANDING_Z_THRESHOLD
    )

    return bool(success)


# ============================================================
# Main validation run
# ============================================================

def main():
    initialize_expected_side_lengths()

    print("============================================================")
    print(" Gazebo Kinematic Validation: 3-UAV Triangle Formation")
    print("============================================================")
    print(f"World name: {WORLD_NAME}")
    print(f"Result CSV: {RESULT_CSV}")
    print("")
    print("Make sure Gazebo is already running:")
    print(f"  gz sim -v 4 gazebo_mappo/worlds/{WORLD_NAME}.sdf")
    print("")

    print("[Setup] Disabling gravity for kinematic proxy validation...")
    set_gravity(enabled=False)

    rows = []
    mission_sequence = generate_mission_sequence()

    print("\n[Mission] Starting validation sequence...\n")

    for mission_step, (phase, poses) in enumerate(mission_sequence, start=1):
        ok = set_uav_poses(poses)

        row = build_row(
            mission_step=mission_step,
            phase=phase,
            poses=poses,
            success=False,
        )

        rows.append(row)

        print(
            f"[{mission_step:03d}] "
            f"Phase={phase:15s} | "
            f"Center=({row['center_x']:.2f}, {row['center_y']:.2f}, {row['center_z']:.2f}) | "
            f"GoalErr={row['goal_error']:.3f} | "
            f"FormErr={row['formation_error']:.3f} | "
            f"MinSep={row['min_separation']:.3f} | "
            f"Gazebo={'OK' if ok else 'FAILED'}"
        )

        if STEP_SLEEP > 0:
            time.sleep(STEP_SLEEP)

    if len(rows) > 0:
        success = evaluate_success(rows[-1])
        rows[-1]["success"] = int(success)
    else:
        success = False

    write_csv(rows)

    print("\n============================================================")
    print(" Gazebo Validation Summary")
    print("============================================================")
    print(f"Success: {success}")

    if rows:
        print(f"Final phase: {rows[-1]['phase']}")
        print(
            f"Final center: "
            f"({rows[-1]['center_x']:.3f}, "
            f"{rows[-1]['center_y']:.3f}, "
            f"{rows[-1]['center_z']:.3f})"
        )
        print(f"Final goal error: {rows[-1]['goal_error']:.3f}")
        print(f"Final formation error: {rows[-1]['formation_error']:.3f}")
        print(f"Final minimum separation: {rows[-1]['min_separation']:.3f}")

    print("\n[Cleanup] Re-enabling gravity...")
    set_gravity(enabled=True)

    print("\nDone.")


if __name__ == "__main__":
    main()