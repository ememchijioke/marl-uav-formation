import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class SingleUAVHoverEnv(gym.Env):
    """
    Single UAV mission environment.

    Mission:
        1. Take off at A
        2. Hover at A
        3. Move from A to B
        4. Hover at B
        5. Land at B

    Action:
        Small residual correction around a safe phase-based reference target.

    Observation:
        pos(3), vel(3), rpy(3), rel_target(3), current_target(3), phase_onehot(5) = 20
    """

    metadata = {"render_modes": ["human", "headless"]}

    PHASE_TAKEOFF_A = 0
    PHASE_HOVER_A = 1
    PHASE_MOVE_TO_B = 2
    PHASE_HOVER_B = 3
    PHASE_LAND_B = 4

    def __init__(
        self,
        render_mode="headless",
        max_steps=600,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.05, 0.05, 0.06),
        hover_tol=0.18,
        landing_tol=0.20,
        hover_target_z=1.0,
        landing_z=0.10,
        goal_bonus=500.0,
        distance_weight=6.0,
        altitude_weight=8.0,
        velocity_weight=1.5,
        attitude_weight=1.5,
        alive_reward=0.2,
        crash_penalty=250.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.ctrl_freq = ctrl_freq
        self.sim_freq = sim_freq

        self.action_scales = np.array(action_scales, dtype=np.float32)

        self.hover_tol = hover_tol
        self.landing_tol = landing_tol
        self.hover_target_z = hover_target_z
        self.landing_z = landing_z

        self.goal_bonus = goal_bonus
        self.distance_weight = distance_weight
        self.altitude_weight = altitude_weight
        self.velocity_weight = velocity_weight
        self.attitude_weight = attitude_weight
        self.alive_reward = alive_reward
        self.crash_penalty = crash_penalty

        self.start_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        self.hover_A = np.array([0.0, 0.0, self.hover_target_z], dtype=np.float32)
        self.hover_B = np.array([2.0, 0.0, self.hover_target_z], dtype=np.float32)
        self.land_B = np.array([2.0, 0.0, self.landing_z], dtype=np.float32)

        # Keep compatibility with old evaluator names.
        self.hover_target = self.hover_A.copy()

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
            shape=(20,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.current_phase = self.PHASE_TAKEOFF_A
        self.current_target = self.hover_A.copy()

        self.phase_counter = 0
        self.stable_counter = 0
        self.move_progress = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.current_phase = self.PHASE_TAKEOFF_A
        self.current_target = self.hover_A.copy()

        self.phase_counter = 0
        self.stable_counter = 0
        self.move_progress = 0.0

        sim_obs, _ = self.env.reset(seed=seed, options=options)

        obs = self._build_obs(sim_obs)
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1
        self.phase_counter += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        reference_target = self._get_reference_target()

        residual = action * self.action_scales
        self.current_target = reference_target + residual

        self.current_target[0] = np.clip(self.current_target[0], -0.20, 2.20)
        self.current_target[1] = np.clip(self.current_target[1], -0.15, 0.15)

        if self.current_phase == self.PHASE_LAND_B:
            self.current_target[2] = np.clip(self.current_target[2], 0.08, 1.10)
        else:
            self.current_target[2] = np.clip(self.current_target[2], 0.80, 1.15)

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

    def _get_reference_target(self):
        if self.current_phase == self.PHASE_TAKEOFF_A:
            return self.hover_A.copy()

        if self.current_phase == self.PHASE_HOVER_A:
            return self.hover_A.copy()

        if self.current_phase == self.PHASE_MOVE_TO_B:
            self.move_progress = min(1.0, self.move_progress + 0.004)
            return (1.0 - self.move_progress) * self.hover_A + self.move_progress * self.hover_B

        if self.current_phase == self.PHASE_HOVER_B:
            return self.hover_B.copy()

        if self.current_phase == self.PHASE_LAND_B:
            descent_progress = min(1.0, self.phase_counter / 140.0)
            return (1.0 - descent_progress) * self.hover_B + descent_progress * self.land_B

        return self.hover_A.copy()

    def _advance_phase_if_ready(self, pos, vel, rpy):
        speed = float(np.linalg.norm(vel))
        attitude_error = float(np.linalg.norm(rpy[0:2]))

        if self.current_phase == self.PHASE_TAKEOFF_A:
            stable = (
                np.linalg.norm(pos - self.hover_A) < 0.25
                and speed < 0.50
                and attitude_error < 0.50
            )
            if stable:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            if self.stable_counter >= 25:
                self._set_phase(self.PHASE_HOVER_A)

        elif self.current_phase == self.PHASE_HOVER_A:
            stable = (
                np.linalg.norm(pos - self.hover_A) < self.hover_tol
                and speed < 0.40
                and attitude_error < 0.45
            )
            if stable:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            if self.stable_counter >= 40:
                self._set_phase(self.PHASE_MOVE_TO_B)

        elif self.current_phase == self.PHASE_MOVE_TO_B:
            near_B = (
                np.linalg.norm(pos - self.hover_B) < 0.30
                and speed < 0.60
                and self.move_progress >= 1.0
            )
            if near_B:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            if self.stable_counter >= 25:
                self._set_phase(self.PHASE_HOVER_B)

        elif self.current_phase == self.PHASE_HOVER_B:
            stable = (
                np.linalg.norm(pos - self.hover_B) < self.hover_tol
                and speed < 0.40
                and attitude_error < 0.45
            )
            if stable:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            if self.stable_counter >= 40:
                self._set_phase(self.PHASE_LAND_B)

    def _set_phase(self, new_phase):
        self.current_phase = new_phase
        self.phase_counter = 0
        self.stable_counter = 0

    def _build_obs(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)
        s = sim_obs[0]

        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]

        reference_target = self._get_reference_target()
        rel_target = reference_target - pos

        phase_onehot = np.zeros(5, dtype=np.float32)
        phase_onehot[self.current_phase] = 1.0

        obs = np.concatenate(
            [
                pos,
                vel,
                rpy,
                rel_target,
                self.current_target,
                phase_onehot,
            ]
        ).astype(np.float32)

        return obs

    def _compute_reward_done_info(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)
        s = sim_obs[0]

        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]

        reference_target = self._get_reference_target()

        dist_to_target = float(np.linalg.norm(reference_target - pos))
        altitude_error = float(abs(reference_target[2] - pos[2]))
        xy_error = float(np.linalg.norm(pos[0:2] - reference_target[0:2]))
        speed = float(np.linalg.norm(vel))
        attitude_error = float(np.linalg.norm(rpy[0:2]))

        landed_successfully = bool(
            self.current_phase == self.PHASE_LAND_B
            and abs(pos[0] - self.land_B[0]) < self.landing_tol
            and abs(pos[1] - self.land_B[1]) < self.landing_tol
            and pos[2] < 0.18
            and speed < 0.35
            and attitude_error < 0.55
        )

        crashed = bool(
            not landed_successfully
            and (
                pos[2] < 0.035
                or abs(rpy[0]) > 1.20
                or abs(rpy[1]) > 1.20
                or abs(pos[1]) > 1.20
            )
        )

        reward = 0.0
        reward += self.alive_reward
        reward += -self.distance_weight * dist_to_target
        reward += -self.altitude_weight * altitude_error
        reward += -self.velocity_weight * speed
        reward += -self.attitude_weight * attitude_error

        # Encourage staying close to safe corridor y = 0
        reward += -2.0 * abs(float(pos[1]))

        # Phase completion rewards
        old_phase = self.current_phase
        self._advance_phase_if_ready(pos, vel, rpy)

        if self.current_phase != old_phase:
            reward += 80.0

        if dist_to_target < 0.25:
            reward += 5.0

        if dist_to_target < 0.18 and speed < 0.45:
            reward += 10.0

        if landed_successfully:
            reward += self.goal_bonus

        if crashed:
            reward -= self.crash_penalty

        done = bool(
            landed_successfully
            or crashed
            or self.step_count >= self.max_steps
        )

        phase_names = {
            self.PHASE_TAKEOFF_A: "TAKEOFF_A",
            self.PHASE_HOVER_A: "HOVER_A",
            self.PHASE_MOVE_TO_B: "MOVE_TO_B",
            self.PHASE_HOVER_B: "HOVER_B",
            self.PHASE_LAND_B: "LAND_B",
        }

        info = {
            "is_success": landed_successfully,
            "phase": phase_names[self.current_phase],
            "phase_id": self.current_phase,
            "stable_counter": self.stable_counter,
            "move_progress": float(self.move_progress),
            "dist_to_target": dist_to_target,
            "altitude": float(pos[2]),
            "altitude_error": altitude_error,
            "xy_error": xy_error,
            "speed": speed,
            "attitude_error": attitude_error,
            "crashed": crashed,
            "landed_successfully": landed_successfully,
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