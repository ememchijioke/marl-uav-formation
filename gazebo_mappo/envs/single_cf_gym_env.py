#!/usr/bin/env python3

import re
import time
import subprocess
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces


ODOM_TOPIC = "/model/cf1/odometry"
ENABLE_TOPIC = "/crazyflie/enable"
CMD_TOPIC = "/crazyflie/gazebo/command/twist"
WORLD_NAME = "single_crazyflie_world"


class LiveOdomReader:
    """
    Continuously reads Crazyflie odometry from Gazebo.
    Faster than calling `gz topic` separately at every step.
    """

    def __init__(self, topic=ODOM_TOPIC):
        self.topic = topic
        self.process = None
        self.thread = None
        self.running = False
        self.latest_position = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return

        self.running = True

        self.process = subprocess.Popen(
            ["gz", "topic", "-e", "-t", self.topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def read_loop(self):
        block = []

        while self.running and self.process is not None:
            line = self.process.stdout.readline()

            if line == "":
                time.sleep(0.01)
                continue

            clean = line.strip()
            block.append(clean)

            if clean == "":
                self.parse_block(block)
                block = []

    def parse_block(self, lines):
        text = "\n".join(lines)

        match = re.search(
            r"position\s*\{\s*"
            r"x:\s*([-+eE0-9\.]+)\s*"
            r"y:\s*([-+eE0-9\.]+)\s*"
            r"z:\s*([-+eE0-9\.]+)",
            text,
        )

        if match is None:
            return

        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))

        with self.lock:
            self.latest_position = np.array([x, y, z], dtype=np.float32)

    def get_position(self):
        with self.lock:
            if self.latest_position is None:
                return None
            return self.latest_position.copy()

    def clear(self):
        with self.lock:
            self.latest_position = None

    def stop(self):
        self.running = False

        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()


class SingleCrazyflieGymEnv(gym.Env):
    """
    Single Crazyflie Gazebo environment.

    Observation:
        [x, y, z, error_x, error_y, error_z]

    Action:
        [ax, ay, az] in [-1, 1]

    The action is converted to velocity commands:
        vx = ax * max_xy_speed
        vy = ay * max_xy_speed
        vz = az * max_z_speed

    A small takeoff assist is used near the ground so PPO does not get stuck
    before learning meaningful horizontal motion.
    """

    def __init__(self):
        super().__init__()

        self.goal = np.array([2.1, 0.0, 0.8], dtype=np.float32)

        self.max_xy_speed = 0.35
        self.max_z_speed = 0.30

        self.max_steps = 120
        self.step_sleep = 0.10

        self.current_step = 0
        self.last_distance = None

        self.odom_reader = LiveOdomReader()
        self.odom_reader.start()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
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
        self.stop_drone()
        time.sleep(0.2)

        self.odom_reader.clear()

        self.run_command([
            "gz", "service",
            "-s", f"/world/{WORLD_NAME}/control",
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", "reset: {all: true}",
        ])

        time.sleep(0.8)

    def wait_for_position(self, timeout=3.0):
        start = time.time()

        while time.time() - start < timeout:
            position = self.odom_reader.get_position()

            if position is not None:
                return position

            time.sleep(0.05)

        return None

    def get_observation(self):
        position = self.odom_reader.get_position()

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

    def calculate_distance_values(self, observation):
        error_x = float(observation[3])
        error_y = float(observation[4])
        error_z = float(observation[5])

        xy_distance = float(np.sqrt(error_x**2 + error_y**2))
        z_error = abs(error_z)
        distance = float(np.sqrt(error_x**2 + error_y**2 + error_z**2))

        return distance, xy_distance, z_error

    def calculate_reward(self, observation):
        x = float(observation[0])
        y = float(observation[1])
        z = float(observation[2])

        distance, xy_distance, z_error = self.calculate_distance_values(observation)

        reward = 0.0

        # Main objective: get closer to the goal.
        reward -= distance

        # Progress reward: reward movement toward the goal.
        if self.last_distance is not None:
            progress = self.last_distance - distance
            reward += 15.0 * progress

        # Encourage useful flight altitude around 0.8 m.
        reward -= 2.0 * z_error

        # Encourage leaving the ground.
        if z < 0.25:
            reward -= 3.0

        # Penalize flying too high, but do not end too early.
        if z > 1.4:
            reward -= 8.0 * (z - 1.4)

        # Success bonus.
        if xy_distance < 0.30 and z_error < 0.25:
            reward += 40.0

        self.last_distance = distance

        return float(reward), distance, xy_distance, z_error

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.last_distance = None

        self.reset_world()
        self.enable_drone()

        position = self.wait_for_position(timeout=3.0)

        if position is None:
            observation = np.zeros(6, dtype=np.float32)
        else:
            error = self.goal - position
            observation = np.array([
                position[0],
                position[1],
                position[2],
                error[0],
                error[1],
                error[2],
            ], dtype=np.float32)

        distance, xy_distance, z_error = self.calculate_distance_values(observation)
        self.last_distance = distance

        info = {
            "distance": float(distance),
            "xy_distance": float(xy_distance),
            "z_error": float(z_error),
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "goal_z": float(self.goal[2]),
        }

        return observation, info

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        vx = float(action[0] * self.max_xy_speed)
        vy = float(action[1] * self.max_xy_speed)
        vz = float(action[2] * self.max_z_speed)

        current_obs = self.get_observation()

        # Takeoff assist:
        # If the drone is still very low, force enough upward velocity.
        # This prevents PPO from wasting episodes stuck on the floor.
        if current_obs is not None:
            current_z = float(current_obs[2])
            if current_z < 0.45:
                vz = max(vz, 0.28)

        self.send_velocity(vx, vy, vz)
        time.sleep(self.step_sleep)

        observation = self.get_observation()

        if observation is None:
            observation = np.zeros(6, dtype=np.float32)
            reward = -20.0
            terminated = True
            truncated = False
            info = {"error": "Could not read odometry"}
            return observation, float(reward), bool(terminated), bool(truncated), info

        reward, distance, xy_distance, z_error = self.calculate_reward(observation)

        self.current_step += 1

        z = float(observation[2])

        reached_goal = bool(xy_distance < 0.30 and z_error < 0.25)

        # Less aggressive termination so PPO gets longer learning episodes.
        out_of_bounds = bool(
            abs(float(observation[0])) > 4.5
            or abs(float(observation[1])) > 3.5
            or z > 2.4
        )

        too_low = bool(self.current_step > 20 and z < -0.20)

        terminated = bool(reached_goal or out_of_bounds or too_low)
        truncated = bool(self.current_step >= self.max_steps)

        if terminated or truncated:
            self.stop_drone()

        info = {
            "step": int(self.current_step),
            "distance": float(distance),
            "xy_distance": float(xy_distance),
            "z_error": float(z_error),
            "reached_goal": reached_goal,
            "out_of_bounds": out_of_bounds,
            "too_low": too_low,
            "x": float(observation[0]),
            "y": float(observation[1]),
            "z": float(observation[2]),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
        }

        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.stop_drone()
        self.odom_reader.stop()


def test_environment():
    env = SingleCrazyflieGymEnv()

    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)

    for step in range(100):
        error_x = obs[3]
        error_y = obs[4]
        error_z = obs[5]

        # Simple hand-made controller for testing the environment.
        action = np.array([
            0.8 * error_x,
            0.8 * error_y,
            0.8 * error_z,
        ], dtype=np.float32)

        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step:03d} | "
            f"x={info.get('x', 0):.2f}, "
            f"y={info.get('y', 0):.2f}, "
            f"z={info.get('z', 0):.2f}, "
            f"dist={info.get('distance', 0):.2f}, "
            f"xy={info.get('xy_distance', 0):.2f}, "
            f"z_err={info.get('z_error', 0):.2f}, "
            f"reward={reward:.2f}"
        )

        if terminated or truncated:
            print("Episode finished:", info)
            break

    env.close()


if __name__ == "__main__":
    test_environment()