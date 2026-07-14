#!/usr/bin/env python3

import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "envs")))

from single_cf_gym_env import SingleCrazyflieGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("Starting PPO training for single Crazyflie in Gazebo")
    print(f"Training timesteps: {args.timesteps}")

    env = SingleCrazyflieGymEnv()

    os.makedirs("gazebo_mappo/models", exist_ok=True)
    os.makedirs("gazebo_mappo/checkpoints", exist_ok=True)
    os.makedirs("gazebo_mappo/tensorboard_logs", exist_ok=True)

    model_path = "gazebo_mappo/models/ppo_single_cf_gazebo"

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="gazebo_mappo/checkpoints/",
        name_prefix="ppo_single_cf_checkpoint",
    )

    if args.resume and os.path.exists(model_path + ".zip"):
        print("Loading existing model and continuing training...")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=64,
            batch_size=64,
            n_epochs=6,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            tensorboard_log="gazebo_mappo/tensorboard_logs/",
            device="cpu",
        )

    print("Training started...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    model.save(model_path)
    env.close()

    print(f"Model saved to: {model_path}.zip")
    print("Training finished.")


if __name__ == "__main__":
    main()