#!/usr/bin/env python3

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Allow Python to find the environment file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "envs")))

from single_cf_gym_env import SingleCrazyflieGymEnv


def main():
    print("Starting PPO training for single Crazyflie in Gazebo")

    # Create environment
    env = SingleCrazyflieGymEnv()

    # Check if the environment follows Gymnasium rules
    print("Checking environment...")
    check_env(env, warn=True)

    # Create folder for saved models
    os.makedirs("gazebo_mappo/models", exist_ok=True)

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="gazebo_mappo/tensorboard_logs/",
    )

    # Train model
    print("Training started...")
    model.learn(total_timesteps=20_000)

    # Save model
    model_path = "gazebo_mappo/models/ppo_single_cf_gazebo"
    model.save(model_path)

    print(f"Model saved to: {model_path}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
