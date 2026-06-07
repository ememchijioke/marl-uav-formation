#!/usr/bin/env python3

import os
import sys
import time

from stable_baselines3 import PPO

# Allow Python to find the environment file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "envs")))

from single_cf_gym_env import SingleCrazyflieGymEnv


MODEL_PATH = "gazebo_mappo/models/ppo_single_cf_gazebo"


def main():
    print("Evaluating trained PPO model for single Crazyflie in Gazebo")
    print("Make sure Gazebo is already running.")

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Model not found: {MODEL_PATH}.zip")
        return

    env = SingleCrazyflieGymEnv()

    print("Loading model...")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()

    print("Initial observation:", obs)
    print("Initial distance:", info.get("distance"))

    total_reward = 0.0

    for step in range(150):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"Step {step:03d} | "
            f"x={info.get('x', 0):.2f}, "
            f"y={info.get('y', 0):.2f}, "
            f"z={info.get('z', 0):.2f}, "
            f"distance={info.get('distance', 0):.2f}, "
            f"reward={reward:.2f}, "
            f"total_reward={total_reward:.2f}"
        )

        if terminated or truncated:
            print("Episode finished.")
            print("Info:", info)
            break

        time.sleep(0.05)

    env.close()

    print("Evaluation complete.")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
