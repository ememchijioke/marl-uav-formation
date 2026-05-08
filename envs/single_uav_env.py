import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class SingleUAVHoverEnv(gym.Env):
    """
    Single UAV takeoff + hover environment.

    Goal:
        Lift from initial height and hover at fixed position:
        target_pos = [0.0, 0.0, 1.0]

    Action:
        Small residual correction around the fixed hover target.
        action = [dx, dy, dz] in [-1, 1]

    Observation:
        pos(3), vel(3), rpy(3), rel_target(3), target_pos(3) = 15
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        render_mode="headless",
        max_steps=300,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.05, 0.05, 0.08),
        hover_tol=0.15,
        hover_target_z=1.0,
        goal_bonus=300.0,
        distance_weight=8.0,
        altitude_weight=10.0,
        velocity_weight=2.0,
        attitude_weight=2.0,
        alive_reward=0.2,
        crash_penalty=200.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.ctrl_freq = ctrl_freq
        self.sim_freq = sim_freq

        self.action_scales = np.array(action_scales, dtype=np.float32)

        self.hover_tol = hover_tol
        self.hover_target_z = hover_target_z

        self.goal_bonus = goal_bonus
        self.distance_weight = distance_weight
        self.altitude_weight = altitude_weight
        self.velocity_weight = velocity_weight
        self.attitude_weight = attitude_weight
        self.alive_reward = alive_reward
        self.crash_penalty = crash_penalty

        self.start_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        self.hover_target = np.array([0.0, 0.0, self.hover_target_z], dtype=np.float32)

        self.initial_xyzs = self.start_pos.reshape(1, 3)
        self.initial_rpys = np.zeros((1, 3), dtype=np.float32)

        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=np.inf,
            initial_xyzs=self.initial_xyzs,
            initial_rpys=self.initial_rpys,
            physics=Physics.PYB,
            pyb_freq=self.sim_freq,
            ctrl_freq=self.ctrl_freq,
            gui=(self.render_mode == "human"),
            record=False,
            obstacles=False,
            user_debug_gui=(self.render_mode == "human"),
        )

        self.controller = DSLPIDControl(drone_model=DroneModel.CF2X)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.current_target = self.hover_target.copy()
        self.hover_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.hover_counter = 0
        self.current_target = self.hover_target.copy()

        sim_obs, _ = self.env.reset(seed=seed, options=options)

        obs = self._build_obs(sim_obs)
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # RL only adds tiny correction around safe hover target.
        residual = action * self.action_scales
        self.current_target = self.hover_target + residual

        # Very strict safe target box.
        self.current_target[0] = np.clip(self.current_target[0], -0.10, 0.10)
        self.current_target[1] = np.clip(self.current_target[1], -0.10, 0.10)
        self.current_target[2] = np.clip(self.current_target[2], 0.85, 1.15)

        s = self.env._getDroneStateVector(0)

        cur_pos = s[0:3]
        cur_quat = s[3:7]
        cur_vel = s[10:13]
        cur_ang_vel = s[13:16]

        rpm, _, _ = self.controller.computeControl(
            control_timestep=1.0 / self.ctrl_freq,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=self.current_target,
            target_rpy=np.zeros(3),
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3),
        )

        rpm_action = np.array([rpm], dtype=np.float32)

        sim_obs, _, terminated, truncated, _ = self.env.step(rpm_action)

        reward, done, info = self._compute_reward_done_info(sim_obs)

        terminated = bool(done or terminated)
        truncated = bool(truncated or self.step_count >= self.max_steps)

        obs = self._build_obs(sim_obs)

        return obs, reward, terminated, truncated, info

    def _build_obs(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)
        s = sim_obs[0]

        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]
        rel_target = self.hover_target - pos

        obs = np.concatenate(
            [
                pos,
                vel,
                rpy,
                rel_target,
                self.current_target,
            ]
        ).astype(np.float32)

        return obs

    def _compute_reward_done_info(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)
        s = sim_obs[0]

        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]

        dist_to_hover = float(np.linalg.norm(self.hover_target - pos))
        altitude_error = float(abs(self.hover_target_z - pos[2]))
        xy_error = float(np.linalg.norm(pos[0:2] - self.hover_target[0:2]))
        speed = float(np.linalg.norm(vel))
        attitude_error = float(np.linalg.norm(rpy[0:2]))

        crashed = bool(
            pos[2] < 0.05
            or abs(rpy[0]) > 1.0
            or abs(rpy[1]) > 1.0
        )

        stable_hover = bool(
            dist_to_hover <= self.hover_tol
            and speed <= 0.35
            and attitude_error <= 0.35
        )

        if stable_hover:
            self.hover_counter += 1
        else:
            self.hover_counter = 0

        is_success = self.hover_counter >= 40

        reward = 0.0
        reward += self.alive_reward
        reward += -self.distance_weight * dist_to_hover
        reward += -self.altitude_weight * altitude_error
        reward += -self.velocity_weight * speed
        reward += -self.attitude_weight * attitude_error

        # Extra reward for being close to hover region.
        if dist_to_hover <= 0.30:
            reward += 5.0

        if stable_hover:
            reward += 10.0

        if is_success:
            reward += self.goal_bonus

        if crashed:
            reward -= self.crash_penalty

        done = bool(
            is_success
            or crashed
            or self.step_count >= self.max_steps
        )

        info = {
            "is_success": is_success,
            "stable_hover": stable_hover,
            "hover_counter": self.hover_counter,
            "dist_to_hover": dist_to_hover,
            "altitude": float(pos[2]),
            "altitude_error": altitude_error,
            "xy_error": xy_error,
            "speed": speed,
            "attitude_error": attitude_error,
            "crashed": crashed,
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "pos_z": float(pos[2]),
            "target_x": float(self.current_target[0]),
            "target_y": float(self.current_target[1]),
            "target_z": float(self.current_target[2]),
        }

        return reward, done, info

    def _to_numpy_obs(self, sim_obs):
        if isinstance(sim_obs, tuple):
            sim_obs = sim_obs[0]

        if isinstance(sim_obs, dict):
            key = "0" if "0" in sim_obs else 0
            entry = sim_obs[key]

            if isinstance(entry, dict) and "state" in entry:
                sim_obs = np.asarray(entry["state"], dtype=np.float32).reshape(1, -1)
            else:
                sim_obs = np.asarray(entry, dtype=np.float32).reshape(1, -1)
        else:
            sim_obs = np.asarray(sim_obs, dtype=np.float32)
            if sim_obs.ndim == 1:
                sim_obs = sim_obs.reshape(1, -1)

        return sim_obs

    def render(self):
        return

    def close(self):
        self.env.close()