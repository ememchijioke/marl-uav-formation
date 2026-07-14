#!/usr/bin/env python3

import subprocess
import time
import re


ODOM_TOPIC = "/model/cf1/odometry"
ENABLE_TOPIC = "/crazyflie/enable"
CMD_TOPIC = "/crazyflie/gazebo/command/twist"


def run_command(command):
    """Run a normal terminal command from Python."""
    subprocess.run(command, check=True)


def enable_drone():
    """Enable the Crazyflie controller."""
    run_command([
        "gz", "topic",
        "-t", ENABLE_TOPIC,
        "-m", "gz.msgs.Boolean",
        "-p", "data: true"
    ])


def send_velocity(vx, vy, vz, yaw_rate=0.0):
    """Send velocity command to the Crazyflie."""
    message = (
        f"linear: {{x: {vx}, y: {vy}, z: {vz}}}, "
        f"angular: {{z: {yaw_rate}}}"
    )

    run_command([
        "gz", "topic",
        "-t", CMD_TOPIC,
        "-m", "gz.msgs.Twist",
        "-p", message
    ])


def stop_drone():
    """Stop the drone movement."""
    send_velocity(0.0, 0.0, 0.0)


def get_current_position():
    """
    Read one odometry message from Gazebo and extract x, y, z position.
    This uses the working Gazebo topic directly.
    """

    result = subprocess.run(
        ["timeout", "1", "gz", "topic", "-e", "-t", ODOM_TOPIC],
        capture_output=True,
        text=True
    )

    text = result.stdout

    # Extract position block from the odometry output
    match = re.search(
        r"position\s*\{\s*"
        r"x:\s*([-+eE0-9\.]+)\s*"
        r"y:\s*([-+eE0-9\.]+)\s*"
        r"z:\s*([-+eE0-9\.]+)",
        text
    )

    if match is None:
        return None

    x = float(match.group(1))
    y = float(match.group(2))
    z = float(match.group(3))

    return x, y, z


def limit(value, minimum, maximum):
    """Keep velocity inside a safe range."""
    return max(minimum, min(value, maximum))


def main():
    print("Simple Crazyflie goal controller")
    print("Make sure Gazebo is already running.")

    print("Enabling drone...")
    enable_drone()
    time.sleep(1)

    # Target position inside the arena
    goal_x = 2.1
    goal_y = 0.0
    goal_z = 0.8

    print(f"Goal position: x={goal_x}, y={goal_y}, z={goal_z}")

    # Controller settings
    kp = 0.4
    max_speed = 0.30

    for step in range(150):
        position = get_current_position()

        if position is None:
            print("Could not read position.")
            time.sleep(0.2)
            continue

        x, y, z = position

        error_x = goal_x - x
        error_y = goal_y - y
        error_z = goal_z - z

        vx = limit(kp * error_x, -max_speed, max_speed)
        vy = limit(kp * error_y, -max_speed, max_speed)
        vz = limit(kp * error_z, -max_speed, max_speed)

        send_velocity(vx, vy, vz)

        distance = (error_x**2 + error_y**2 + error_z**2) ** 0.5

        print(
            f"Step {step:03d} | "
            f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f} | "
            f"Distance to goal: {distance:.2f}"
        )

        if distance < 0.20:
            print("Goal reached.")
            break

        time.sleep(0.2)

    print("Stopping drone...")
    stop_drone()
    print("Done.")


if __name__ == "__main__":
    main()