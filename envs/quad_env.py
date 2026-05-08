import numpy as np
import gymnasium as gym
from gymnasium import spaces

# gym-pybullet-drones imports
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiUAVRealisticEnv(gym.Env):
    """
    Multi-agent quadrotor formation environment built on gym-pybullet-drones.

    High-level action:
        Per-agent delta target in xyz, normalized to [-1, 1].

    Low-level control:
        DSLPIDControl converts desired position targets into motor RPMs.

    Observation per drone:
        pos(3), vel(3), rpy(3), rel_goal(3), rel_centroid(3) = 15

    Joint observation:
        concatenation of all per-agent observations

    Main design:
        - shared cooperative team reward
        - decentralized per-agent observation for actor
        - full joint observation available to centralized critic through trainer
        - realistic drone dynamics via CtrlAviary + DSLPIDControl
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        render_mode="headless",
        num_agents=3,
        max_steps=300,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.35, 0.25, 0.20),
        goal_tol=0.6,
        formation_tol=1.2,
        goal_bonus=500.0,
        formation_penalty=2.0,
        mean_penalty=1.0,
        slot_penalty_weight=1.5,
        cohesion_bonus_weight=4.0,
        progress_weight=20.0,
        alive_reward=0.2,
        collision_penalty=50.0,
        min_separation=0.18,
        speed_penalty_weight=0.5,
        episode_len_sec=None,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.ctrl_freq = ctrl_freq
        self.sim_freq = sim_freq

        self.action_scales = np.array(action_scales, dtype=np.float32)

        self.goal_tol = goal_tol
        self.formation_tol = formation_tol
        self.goal_bonus = goal_bonus
        self.formation_penalty = formation_penalty
        self.mean_penalty = mean_penalty
        self.slot_penalty_weight = slot_penalty_weight
        self.cohesion_bonus_weight = cohesion_bonus_weight
        self.progress_weight = progress_weight
        self.alive_reward = alive_reward
        self.collision_penalty = collision_penalty
        self.min_separation = min_separation
        self.speed_penalty_weight = speed_penalty_weight

        self.episode_len_sec = (
            episode_len_sec if episode_len_sec is not None else max_steps / ctrl_freq
        )

        # Formation centers
        self.center_A = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.center_B = np.array([4.0, 0.0, 1.0], dtype=np.float32)

        # 3-agent triangle formation offsets
        self.offsets = np.array(
            [
                [-0.35, -0.20, 0.0],
                [ 0.35, -0.20, 0.0],
                [ 0.00,  0.35, 0.0],
            ],
            dtype=np.float32,
        )

        if self.num_agents != len(self.offsets):
            raise ValueError("Offsets must match num_agents for this formation.")

        self.initial_xyzs = self.center_A + self.offsets
        self.initial_rpys = np.zeros((self.num_agents, 3), dtype=np.float32)

        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_agents,
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

        self.controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X)
            for _ in range(self.num_agents)
        ]

        # Action = per-agent xyz delta target in normalized range [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 * self.num_agents,),
            dtype=np.float32,
        )

        # Obs per drone:
        # pos(3), vel(3), rpy(3), rel_goal(3), rel_centroid(3) = 15
        self.obs_dim_per_agent = 15
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim_per_agent * self.num_agents,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.current_targets = self.initial_xyzs.copy()
        self.prev_centroid_x = float(self.initial_xyzs[:, 0].mean())
        self.got_goal_bonus = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.got_goal_bonus = False

        sim_obs, sim_info = self.env.reset(seed=seed, options=options)

        self.current_targets = self.initial_xyzs.copy()
        self.prev_centroid_x = float(self.initial_xyzs[:, 0].mean())

        obs = self._build_obs(sim_obs)
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, 3)
        action = np.clip(action, -1.0, 1.0)

        # Move each drone's target slightly each control step
        delta_targets = action * self.action_scales[None, :]
        self.current_targets += delta_targets

        # Keep targets in a sane box
        self.current_targets[:, 0] = np.clip(
            self.current_targets[:, 0], -1.0, self.center_B[0] + 1.0
        )
        self.current_targets[:, 1] = np.clip(
            self.current_targets[:, 1], -2.0, 2.0
        )
        self.current_targets[:, 2] = np.clip(
            self.current_targets[:, 2], 0.6, 1.8
        )

        rpm_action = np.zeros((self.num_agents, 4), dtype=np.float32)

        for i in range(self.num_agents):
            s = self.env._getDroneStateVector(i)

            # Typical state layout in gym-pybullet-drones:
            # pos: 0:3, quat: 3:7, rpy: 7:10, vel: 10:13, ang_vel: 13:16
            cur_pos = s[0:3]
            cur_quat = s[3:7]
            cur_vel = s[10:13]
            cur_ang_vel = s[13:16]

            target_pos = self.current_targets[i]
            target_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            target_vel = np.zeros(3, dtype=np.float32)
            target_rpy_rates = np.zeros(3, dtype=np.float32)

            rpm, _, _ = self.controllers[i].computeControl(
                control_timestep=1.0 / self.ctrl_freq,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=target_rpy,
                target_vel=target_vel,
                target_rpy_rates=target_rpy_rates,
            )
            rpm_action[i] = rpm

        sim_obs, _, terminated, truncated, sim_info = self.env.step(rpm_action)

        reward, done, info = self._compute_reward_done_info(sim_obs)

        terminated = bool(done or terminated)
        truncated = bool(truncated or (self.step_count >= self.max_steps))

        obs = self._build_obs(sim_obs)

        return obs, reward, terminated, truncated, info

    def _build_obs(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)

        positions = sim_obs[:, 0:3]
        centroid = positions.mean(axis=0)

        obs = []
        for i in range(self.num_agents):
            s = sim_obs[i]

            pos = s[0:3]
            rpy = s[7:10]
            vel = s[10:13]

            rel_goal = self.center_B - pos
            rel_centroid = centroid - pos

            obs.extend(pos.tolist())
            obs.extend(vel.tolist())
            obs.extend(rpy.tolist())
            obs.extend(rel_goal.tolist())
            obs.extend(rel_centroid.tolist())

        return np.array(obs, dtype=np.float32)

    def _compute_reward_done_info(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)

        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]

        centroid = positions.mean(axis=0)
        centroid_x = float(centroid[0])

        # Shape maintenance around current centroid
        desired_shape_positions = centroid + self.offsets
        formation_errors = np.linalg.norm(positions - desired_shape_positions, axis=1)
        max_err = float(np.max(formation_errors))
        mean_err = float(np.mean(formation_errors))

        # Slot alignment around the final target center
        desired_goal_slots = self.center_B + self.offsets
        slot_errors = np.linalg.norm(positions - desired_goal_slots, axis=1)
        mean_slot_error = float(np.mean(slot_errors))

        dist_to_goal = float(np.linalg.norm(centroid - self.center_B))

        min_pair_dist = np.inf
        collision_count = 0
        cohesion_bonus = 0.0

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                min_pair_dist = min(min_pair_dist, d)

                if d < self.min_separation:
                    collision_count += 1

                # Light cohesion shaping in a reasonable band
                if 0.25 < d < 1.0:
                    cohesion_bonus += (1.0 - d) * self.cohesion_bonus_weight

        r_form = -(self.formation_penalty * max_err + self.mean_penalty * mean_err)
        r_slot = -self.slot_penalty_weight * mean_slot_error
        r_prog = self.progress_weight * (centroid_x - self.prev_centroid_x)
        r_alive = self.alive_reward

        r_goal_dense = 10.0 * (
            1.0 - np.clip(
                dist_to_goal / np.linalg.norm(self.center_B - self.center_A),
                0.0,
                1.0,
            )
        )

        r_collision = -self.collision_penalty * collision_count

        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))
        r_smooth = -self.speed_penalty_weight * mean_speed

        r_goal = 0.0
        is_success = False
        if (
            not self.got_goal_bonus
            and dist_to_goal <= self.goal_tol
            and max_err <= self.formation_tol
        ):
            r_goal = self.goal_bonus
            self.got_goal_bonus = True
            is_success = True

        reward = (
            r_form
            + r_slot
            + cohesion_bonus
            + r_prog
            + r_alive
            + r_goal_dense
            + r_collision
            + r_smooth
            + r_goal
        )

        self.prev_centroid_x = centroid_x

        done = bool(
            is_success
            or collision_count > 0
            or self.step_count >= self.max_steps
        )

        if min_pair_dist == np.inf:
            min_pair_dist = 0.0

        info = {
            "is_success": is_success,
            "centroid_x": centroid_x,
            "dist_to_goal": dist_to_goal,
            "max_formation_error": max_err,
            "mean_formation_error": mean_err,
            "mean_slot_error": mean_slot_error,
            "within_formation_tol": float(max_err <= self.formation_tol),
            "collision_count": collision_count,
            "min_pair_dist": float(min_pair_dist),
            "cohesion_bonus": float(cohesion_bonus),
            "mean_speed": mean_speed,
        }

        return reward, done, info

    def _to_numpy_obs(self, sim_obs):
        """
        Normalize observations from gym-pybullet-drones into ndarray shape:
            (num_agents, state_dim)
        """
        if isinstance(sim_obs, tuple):
            sim_obs = sim_obs[0]

        if isinstance(sim_obs, dict):
            rows = []
            for i in range(self.num_agents):
                key = str(i) if str(i) in sim_obs else i
                entry = sim_obs[key]

                if isinstance(entry, dict) and "state" in entry:
                    rows.append(np.asarray(entry["state"], dtype=np.float32))
                else:
                    rows.append(np.asarray(entry, dtype=np.float32))
            sim_obs = np.stack(rows, axis=0)
        else:
            sim_obs = np.asarray(sim_obs, dtype=np.float32)

        return sim_obs

    def render(self):
        return

    def close(self):
        self.env.close()