#!/usr/bin/env python3

import subprocess
import time
import re
import random


ODOM_TOPIC = "/model/cf1/odometry"
ENABLE_TOPIC = "/crazyflie/enable"
CMD_TOPIC = "/crazyflie/gazebo/command/twist"


class SingleCrazyflieGazeboEnv:
    def __init__(self):
        self.goal = (2.1, 0.0, 0.8)
        self.max_speed = 0.30
        self.step_delay = 0.2
        self.max_steps = 150
        self.current_step = 0

    def run_command(self, command):
        subprocess.run(command, check=True)

    def enable_drone(self):
        self.run_command([
            "gz", "topic",
            "-t", ENABLE_TOPIC,
            "-m", "gz.msgs.Boolean",
            "-p", "data: true"
        ])

    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        message = (
            f"linear: {{x: {vx}, y: {vy}, z: {vz}}}, "
            f"angular: {{z: {yaw_rate}}}"
        )

        self.run_command([
            "gz", "topic",
            "-t", CMD_TOPIC,
            "-m", "gz.msgs.Twist",
            "-p", message
        ])

    def stop_drone(self):
        self.send_velocity(0.0, 0.0, 0.0)

    def get_position(self):
        result = subprocess.run(
            ["timeout", "1", "gz", "topic", "-e", "-t", ODOM_TOPIC],
            capture_output=True,
            text=True
        )

        text = result.stdout

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

    def get_observation(self):
        position = self.get_position()

        if position is None:
            return None

        x, y, z = position
        goal_x, goal_y, goal_z = self.goal

        error_x = goal_x - x
        error_y = goal_y - y
        error_z = goal_z - z

        observation = [
            x, y, z,
            error_x, error_y, error_z
        ]

        return observation

    def compute_reward(self, observation):
        x, y, z, error_x, error_y, error_z = observation

        distance = (error_x**2 + error_y**2 + error_z**2) ** 0.5

        reward = -distance

        if distance < 0.20:
            reward += 10.0

        if z < 0.05:
            reward -= 2.0

        return reward, distance

    def limit_action(self, value):
        return max(-self.max_speed, min(value, self.max_speed))

    def reset(self):
        print("Resetting simple environment state.")
        print("For now, manually reset Gazebo from the GUI if needed.")

        self.current_step = 0
        self.enable_drone()
        time.sleep(0.5)

        observation = self.get_observation()
        return observation

    def step(self, action):
        vx, vy, vz = action

        vx = self.limit_action(vx)
        vy = self.limit_action(vy)
        vz = self.limit_action(vz)

        self.send_velocity(vx, vy, vz)
        time.sleep(self.step_delay)

        observation = self.get_observation()

        if observation is None:
            reward = -10.0
            done = True
            info = {"error": "No odometry received"}
            return None, reward, done, info

        reward, distance = self.compute_reward(observation)

        self.current_step += 1

        reached_goal = distance < 0.20
        timeout = self.current_step >= self.max_steps

        done = reached_goal or timeout

        info = {
            "distance": distance,
            "reached_goal": reached_goal,
            "step": self.current_step
        }

        if done:
            self.stop_drone()

        return observation, reward, done, info


def test_random_actions():
    env = SingleCrazyflieGazeboEnv()

    observation = env.reset()

    if observation is None:
        print("Could not read initial observation.")
        return

    print("Initial observation:", observation)

    for step in range(50):
        action = [
            random.uniform(-0.2, 0.3),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.2)
        ]

        observation, reward, done, info = env.step(action)

        print(
            f"Step {step:03d} | "
            f"Action: {action} | "
            f"Reward: {reward:.2f} | "
            f"Distance: {info.get('distance', 0):.2f}"
        )

        if done:
            print("Episode finished:", info)
            break

    env.stop_drone()


if __name__ == "__main__":
    test_random_actions()
