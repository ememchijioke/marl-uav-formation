#!/usr/bin/env python3

import re
import time
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces


ODOM_TOPIC = "/model/cf1/odometry"
ENABLE_TOPIC = "/crazyflie/enable"
CMD_TOPIC = "/crazyflie/gazebo/command/twist"
WORLD_NAME = "single_crazyflie_world"


class SingleCrazyflieGymEnv(gym.Env):
    """
    Simple Gymnasium environment for one Crazyflie in Gazebo.

    Observation:
        [x, y, z, error_x, error_y, error_z]

    Action:
        [vx, vy, vz]
    """

    def __init__(self):
        super().__init__()

        self.goal = np.array([2.1, 0.0, 0.8], dtype=np.float32)

        self.max_speed = 0.30
        self.max_steps = 150
        self.step_sleep = 0.20

        self.current_step = 0
        self.last_distance = None

        self.action_space = spaces.Box(
            low=np.array([-self.max_speed, -self.max_speed, -self.max_speed], dtype=np.float32),
            high=np.array([self.max_speed, self.max_speed, self.max_speed], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, -1.0, -10.0, -10.0, -10.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

    def run_command(self, command):
        subprocess.run(command, check=False, capture_output=True, text=True)

    def enable_drone(self):
        self.run_command([
            "gz", "topic",
            "-t", ENABLE_TOPIC,
            "-m", "gz.msgs.Boolean",
            "-p", "data: true",
        ])

    def send_velocity(self, vx, vy, vz):
        message = (
            f"linear: {{x: {vx:.4f}, y: {vy:.4f}, z: {vz:.4f}}}, "
            f"angular: {{z: 0.0000}}"
        )

        self.run_command([
            "gz", "topic",
            "-t", CMD_TOPIC,
            "-m", "gz.msgs.Twist",
            "-p", message,
        ])

    def stop_drone(self):
        self.send_velocity(0.0, 0.0, 0.0)

    def reset_world(self):
        """
        Reset the Gazebo world to the initial state.
        Gazebo must already be running.
        """

        self.stop_drone()
        time.sleep(0.2)

        self.run_command([
            "gz", "service",
            "-s", f"/world/{WORLD_NAME}/control",
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", "reset: {all: true}",
        ])

        time.sleep(0.8)

    def get_position(self):
        """
        Read one odometry message from Gazebo and extract x, y, z.
        """

        result = subprocess.run(
            ["timeout", "0.4", "gz", "topic", "-e", "-t", ODOM_TOPIC],
            capture_output=True,
            text=True,
        )

        text = result.stdout

        match = re.search(
            r"position\s*\{\s*"
            r"x:\s*([-+eE0-9\.]+)\s*"
            r"y:\s*([-+eE0-9\.]+)\s*"
            r"z:\s*([-+eE0-9\.]+)",
            text,
        )

        if match is None:
            return None

        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))

        return np.array([x, y, z], dtype=np.float32)

    def get_observation(self):
        position = self.get_position()

        if position is None:
            return None

        error = self.goal - position

        observation = np.array([
            position[0],
            position[1],
            position[2],
            error[0],
            error[1],
            error[2],
        ], dtype=np.float32)

        return observation

    def calculate_distance(self, observation):
        error_x = observation[3]
        error_y = observation[4]
        error_z = observation[5]

        distance = float(np.sqrt(error_x**2 + error_y**2 + error_z**2))
        return distance

    def calculate_reward(self, observation):
        distance = self.calculate_distance(observation)

        reward = -distance

        if self.last_distance is not None:
            progress = self.last_distance - distance
            reward += 5.0 * progress

        if distance < 0.25:
            reward += 20.0

        z = observation[2]

        if z < 0.05:
            reward -= 1.0

        if z > 2.0:
            reward -= 5.0

        self.last_distance = distance

        return reward, distance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.last_distance = None

        self.reset_world()
        self.enable_drone()
        time.sleep(0.5)

        observation = self.get_observation()

        if observation is None:
            observation = np.zeros(6, dtype=np.float32)

        distance = self.calculate_distance(observation)
        self.last_distance = distance

        info = {
            "distance": distance,
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "goal_z": float(self.goal[2]),
        }

        return observation, info

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -self.max_speed, self.max_speed)

        vx = float(action[0])
        vy = float(action[1])
        vz = float(action[2])

        self.send_velocity(vx, vy, vz)
        time.sleep(self.step_sleep)

        observation = self.get_observation()

        if observation is None:
            observation = np.zeros(6, dtype=np.float32)
            reward = -20.0
            terminated = True
            truncated = False
            info = {"error": "Could not read odometry"}
            return observation, reward, terminated, truncated, info

        reward, distance = self.calculate_reward(observation)

        self.current_step += 1

        reached_goal = bool(distance < 0.25)
        out_of_bounds = (
            abs(observation[0]) > 4.0
            or abs(observation[1]) > 3.0
            or observation[2] > 2.5
        )
        too_low = bool(observation[2] < -0.20)

        terminated = bool(reached_goal or out_of_bounds or too_low)
        truncated = bool(self.current_step >= self.max_steps)

        if terminated or truncated:
            self.stop_drone()

        info = {
            "step": self.current_step,
            "distance": distance,
            "reached_goal": reached_goal,
            "out_of_bounds": out_of_bounds,
            "too_low": too_low,
            "x": float(observation[0]),
            "y": float(observation[1]),
            "z": float(observation[2]),
        }

        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.stop_drone()


def test_environment():
    env = SingleCrazyflieGymEnv()

    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)

    for step in range(80):
        error_x = obs[3]
        error_y = obs[4]
        error_z = obs[5]

        action = np.array([
            0.4 * error_x,
            0.4 * error_y,
            0.4 * error_z,
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step:03d} | "
            f"x={info.get('x', 0):.2f}, "
            f"y={info.get('y', 0):.2f}, "
            f"z={info.get('z', 0):.2f}, "
            f"distance={info.get('distance', 0):.2f}, "
            f"reward={reward:.2f}"
        )

        if terminated or truncated:
            print("Episode finished:", info)
            break

    env.close()


if __name__ == "__main__":
    test_environment()