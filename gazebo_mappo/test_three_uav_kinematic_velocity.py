#!/usr/bin/env python3
"""
test_three_uav_kinematic_velocity.py

Fast 3-UAV Gazebo kinematic formation validation.

Purpose:
    - Validate 3-UAV mission movement in Gazebo.
    - Show takeoff, hover, forward triangle movement, goal hover, and landing.
    - Use fewer Gazebo service calls so playback is much faster.

Important:
    This is a kinematic validation layer using Gazebo set_pose_vector.
    It is not final physical Crazyflie flight control.

Required world:
    gazebo_mappo/worlds/three_uav_kinematic_demo_world.sdf

Important world requirement:
    uav1, uav2, and uav3 must have:
        <static>false</static>

Run:
    Terminal 1:
        gz sim -v 4 gazebo_mappo/worlds/three_uav_kinematic_demo_world.sdf

    Terminal 2:
        python3 gazebo_mappo/test_three_uav_kinematic_velocity.py
"""

import subprocess
import time
import math
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================
# CONFIG
# ============================================================

WORLD_NAME = "three_uav_kinematic_demo_world"

SET_POSE_VECTOR_SERVICE = f"/world/{WORLD_NAME}/set_pose_vector"
SET_PHYSICS_SERVICE = f"/world/{WORLD_NAME}/set_physics"

UAV_NAMES = ["uav1", "uav2", "uav3"]

SERVICE_TIMEOUT = 1000

# ------------------------------------------------------------
# Fast step-based timing
# ------------------------------------------------------------
# This is intentionally step-based instead of time-based.
# Gazebo service calls through subprocess are slow, so fewer steps
# gives playback closer to the PyBullet visual speed.
TAKEOFF_STEPS = 3
START_HOVER_STEPS = 1
FORWARD_FLIGHT_STEPS = 6
GOAL_HOVER_STEPS = 1
LANDING_STEPS = 3
FINAL_HOLD_STEPS = 1

# Optional tiny pause between updates.
# Keep this at 0.0 for fastest playback.
STEP_SLEEP = 0.0

# Mission centers
START_CENTER_X = -2.4
GOAL_CENTER_X = 3.0
CENTER_Y = 0.0

GROUND_Z = 0.35
HOVER_Z = 1.35

# Triangle formation offsets
# uav1 = front
# uav2 = rear-left
# uav3 = rear-right
TRIANGLE_OFFSETS = {
    "uav1": (0.60, 0.00),
    "uav2": (-0.35, 0.60),
    "uav3": (-0.35, -0.60),
}


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float = 0.0


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def smootherstep(alpha: float) -> float:
    """
    Smooth interpolation curve.
    Keeps movement less jumpy while still using few steps.
    """
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)


def run_gz_service(service: str, reqtype: str, reptype: str, req: str) -> bool:
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
        str(SERVICE_TIMEOUT),
        "--req",
        req,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"[ERROR] Service call failed: {service}")
        print(result.stderr)
        return False

    return True


def set_gravity(enabled: bool) -> bool:
    """
    Disable gravity during kinematic validation so the drone proxies do not fall.
    Re-enable gravity at the end.
    """
    gz = -9.8 if enabled else 0.0
    label = "ENABLED" if enabled else "DISABLED"

    req = f"""
gravity {{
  x: 0
  y: 0
  z: {gz}
}}
"""

    ok = run_gz_service(
        service=SET_PHYSICS_SERVICE,
        reqtype="gz.msgs.Physics",
        reptype="gz.msgs.Boolean",
        req=req,
    )

    if ok:
        print(f"[PHYSICS] Gravity {label}")

    return ok


def set_team_poses(poses: Dict[str, Pose]) -> bool:
    """
    Update all UAV poses together using /set_pose_vector.
    """
    blocks = []

    for name in UAV_NAMES:
        pose = poses[name]
        qx, qy, qz, qw = yaw_to_quaternion(pose.yaw)

        block = f"""
pose {{
  name: "{name}"
  position {{
    x: {pose.x}
    y: {pose.y}
    z: {pose.z}
  }}
  orientation {{
    x: {qx}
    y: {qy}
    z: {qz}
    w: {qw}
  }}
}}
"""
        blocks.append(block)

    req = "\n".join(blocks)

    return run_gz_service(
        service=SET_POSE_VECTOR_SERVICE,
        reqtype="gz.msgs.Pose_V",
        reptype="gz.msgs.Boolean",
        req=req,
    )


def interpolate_pose(start: Pose, end: Pose, alpha: float) -> Pose:
    return Pose(
        x=(1.0 - alpha) * start.x + alpha * end.x,
        y=(1.0 - alpha) * start.y + alpha * end.y,
        z=(1.0 - alpha) * start.z + alpha * end.z,
        yaw=(1.0 - alpha) * start.yaw + alpha * end.yaw,
    )


def interpolate_team(
    start_poses: Dict[str, Pose],
    end_poses: Dict[str, Pose],
    alpha: float,
) -> Dict[str, Pose]:
    return {
        name: interpolate_pose(start_poses[name], end_poses[name], alpha)
        for name in UAV_NAMES
    }


def team_center(poses: Dict[str, Pose]) -> Tuple[float, float, float]:
    cx = sum(poses[name].x for name in UAV_NAMES) / len(UAV_NAMES)
    cy = sum(poses[name].y for name in UAV_NAMES) / len(UAV_NAMES)
    cz = sum(poses[name].z for name in UAV_NAMES) / len(UAV_NAMES)
    return cx, cy, cz


def build_triangle_poses(center_x: float, center_y: float, center_z: float) -> Dict[str, Pose]:
    poses = {}

    for name in UAV_NAMES:
        ox, oy = TRIANGLE_OFFSETS[name]

        poses[name] = Pose(
            x=center_x + ox,
            y=center_y + oy,
            z=center_z,
            yaw=0.0,
        )

    return poses


def run_phase_steps(
    phase_name: str,
    start_poses: Dict[str, Pose],
    end_poses: Dict[str, Pose],
    steps: int,
) -> Dict[str, Pose]:
    """
    Run one movement phase using a fixed number of pose updates.
    Fewer steps = faster playback.
    """
    print(f"\n[PHASE] {phase_name}")

    steps = max(1, steps)

    for k in range(steps + 1):
        raw_alpha = k / steps
        alpha = smootherstep(raw_alpha)

        poses = interpolate_team(start_poses, end_poses, alpha)
        ok = set_team_poses(poses)

        cx, cy, cz = team_center(poses)
        print(
            f"  step={k:02d}/{steps:02d} | "
            f"center=({cx: .2f}, {cy: .2f}, {cz: .2f}) | "
            f"ok={ok}"
        )

        if STEP_SLEEP > 0.0:
            time.sleep(STEP_SLEEP)

    return end_poses


def hold_phase_steps(
    phase_name: str,
    poses: Dict[str, Pose],
    steps: int,
) -> Dict[str, Pose]:
    """
    Hold current pose for a few service updates.
    """
    print(f"\n[PHASE] {phase_name}")

    steps = max(1, steps)

    for k in range(steps):
        ok = set_team_poses(poses)

        cx, cy, cz = team_center(poses)
        print(
            f"  hold={k + 1:02d}/{steps:02d} | "
            f"center=({cx: .2f}, {cy: .2f}, {cz: .2f}) | "
            f"ok={ok}"
        )

        if STEP_SLEEP > 0.0:
            time.sleep(STEP_SLEEP)

    return poses


def build_mission():
    ground_start = build_triangle_poses(
        center_x=START_CENTER_X,
        center_y=CENTER_Y,
        center_z=GROUND_Z,
    )

    hover_start = build_triangle_poses(
        center_x=START_CENTER_X,
        center_y=CENTER_Y,
        center_z=HOVER_Z,
    )

    hover_goal = build_triangle_poses(
        center_x=GOAL_CENTER_X,
        center_y=CENTER_Y,
        center_z=HOVER_Z,
    )

    ground_goal = build_triangle_poses(
        center_x=GOAL_CENTER_X,
        center_y=CENTER_Y,
        center_z=GROUND_Z,
    )

    return ground_start, hover_start, hover_goal, ground_goal


def main():
    print("======================================================")
    print("3-UAV Gazebo Fast Kinematic Formation Validation")
    print("======================================================")
    print(f"World: {WORLD_NAME}")
    print(f"Pose vector service: {SET_POSE_VECTOR_SERVICE}")
    print(f"Physics service: {SET_PHYSICS_SERVICE}")
    print(f"UAV names: {UAV_NAMES}")
    print("")
    print("Step configuration:")
    print(f"  TAKEOFF_STEPS:        {TAKEOFF_STEPS}")
    print(f"  START_HOVER_STEPS:    {START_HOVER_STEPS}")
    print(f"  FORWARD_FLIGHT_STEPS: {FORWARD_FLIGHT_STEPS}")
    print(f"  GOAL_HOVER_STEPS:     {GOAL_HOVER_STEPS}")
    print(f"  LANDING_STEPS:        {LANDING_STEPS}")
    print(f"  FINAL_HOLD_STEPS:     {FINAL_HOLD_STEPS}")
    print("")

    print("[SETUP] Disabling gravity for clean kinematic validation")
    set_gravity(enabled=False)
    time.sleep(0.2)

    ground_start, hover_start, hover_goal, ground_goal = build_mission()

    print("[INIT] Resetting UAVs to triangle start position")
    set_team_poses(ground_start)
    time.sleep(0.2)

    current = ground_start

    current = run_phase_steps(
        phase_name="TAKEOFF TO TRIANGLE HOVER",
        start_poses=current,
        end_poses=hover_start,
        steps=TAKEOFF_STEPS,
    )

    current = hold_phase_steps(
        phase_name="START HOVER",
        poses=current,
        steps=START_HOVER_STEPS,
    )

    current = run_phase_steps(
        phase_name="FORWARD TRIANGLE FORMATION FLIGHT",
        start_poses=current,
        end_poses=hover_goal,
        steps=FORWARD_FLIGHT_STEPS,
    )

    current = hold_phase_steps(
        phase_name="GOAL HOVER",
        poses=current,
        steps=GOAL_HOVER_STEPS,
    )

    current = run_phase_steps(
        phase_name="LANDING",
        start_poses=current,
        end_poses=ground_goal,
        steps=LANDING_STEPS,
    )

    current = hold_phase_steps(
        phase_name="FINAL GROUND HOLD",
        poses=current,
        steps=FINAL_HOLD_STEPS,
    )

    print("\n[TEARDOWN] Re-enabling gravity")
    set_gravity(enabled=True)

    print("\n======================================================")
    print("Validation complete.")
    print("Expected visible result:")
    print("  1. Three UAV proxies start in triangle formation.")
    print("  2. They rise quickly.")
    print("  3. They move forward together.")
    print("  4. They pause briefly at the goal.")
    print("  5. They land together.")
    print("======================================================")


if __name__ == "__main__":
    main()