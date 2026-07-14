import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiUAVVelocityFormation2UAVEnv(gym.Env):
    """
    v0.8.3 2-UAV residual velocity MAPPO environment.

    Core idea:
        final_velocity = reference_velocity + residual_scale * policy_action

    The reference controller provides safe goal-directed formation motion.
    MAPPO learns residual corrections for formation quality, smoothness,
    robustness, and transfer-readiness.

    Action per UAV:
        normalized residual velocity correction [rx, ry, rz] in [-1, 1].

    Actor observation per UAV = 24:
        relative_goal_vector_normalized(3)
        own_velocity_normalized(3)
        altitude_error_normalized(1)
        relative_neighbor_position_normalized(3)
        relative_neighbor_velocity_normalized(3)
        neighbor_valid_mask(1)
        previous_action(3)
        reference_velocity_normalized(3)
        formation_tracking_error_normalized(3)
        spacing_error_normalized(1)

    Mission:
        - Start airborne at z = 1.0 m.
        - Maintain 2-UAV side-by-side formation.
        - Move formation center from A to B.
        - No takeoff, no landing, no obstacle yet.
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        render_mode="headless",
        num_agents=2,
        max_steps=500,
        ctrl_freq=48,
        sim_freq=240,
        target_altitude=1.0,
        target_distance_x=2.4,
        formation_spacing=0.80,
        reference_max_vx=0.22,
        reference_max_vy=0.16,
        reference_max_vz=0.10,
        residual_scale=0.10,
        velocity_lookahead=0.28,
        kp_goal=0.35,
        kp_form=0.45,
        kp_alt=0.40,
        goal_bonus=1200.0,
        alive_reward=0.05,
        progress_weight=35.0,
        centroid_weight=3.0,
        own_goal_weight=1.5,
        formation_weight=20.0,
        spacing_weight=12.0,
        velocity_weight=0.25,
        command_weight=0.08,
        residual_weight=0.10,
        smoothness_weight=0.20,
        attitude_weight=0.80,
        collision_penalty=2000.0,
        crash_penalty=2400.0,
        boundary_penalty=800.0,
        success_centroid_tol=0.30,
        success_formation_tol=0.22,
        success_spacing_tol=0.18,
        min_separation=0.32,
        neighbor_dropout_prob=0.03,
        use_neighbor_dropout=True,
    ):
        super().__init__()

        if num_agents != 2:
            raise ValueError("ResidualVelocityFormation2UAVEnv expects exactly 2 UAVs.")

        self.render_mode = render_mode
        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.ctrl_freq = int(ctrl_freq)
        self.sim_freq = int(sim_freq)

        self.target_altitude = float(target_altitude)
        self.target_distance_x = float(target_distance_x)
        self.formation_spacing = float(formation_spacing)

        self.reference_max_vel = np.array(
            [reference_max_vx, reference_max_vy, reference_max_vz],
            dtype=np.float32,
        )

        self.residual_scale = float(residual_scale)
        self.velocity_lookahead = float(velocity_lookahead)

        self.kp_goal = float(kp_goal)
        self.kp_form = float(kp_form)
        self.kp_alt = float(kp_alt)

        self.goal_bonus = float(goal_bonus)
        self.alive_reward = float(alive_reward)
        self.progress_weight = float(progress_weight)
        self.centroid_weight = float(centroid_weight)
        self.own_goal_weight = float(own_goal_weight)
        self.formation_weight = float(formation_weight)
        self.spacing_weight = float(spacing_weight)
        self.velocity_weight = float(velocity_weight)
        self.command_weight = float(command_weight)
        self.residual_weight = float(residual_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.attitude_weight = float(attitude_weight)

        self.collision_penalty = float(collision_penalty)
        self.crash_penalty = float(crash_penalty)
        self.boundary_penalty = float(boundary_penalty)

        self.success_centroid_tol = float(success_centroid_tol)
        self.success_formation_tol = float(success_formation_tol)
        self.success_spacing_tol = float(success_spacing_tol)
        self.min_separation = float(min_separation)

        self.neighbor_dropout_prob = float(neighbor_dropout_prob)
        self.use_neighbor_dropout = bool(use_neighbor_dropout)

        self.goal_scale = 3.0
        self.neighbor_pos_scale = 2.0
        self.neighbor_vel_scale = 1.0
        self.reference_vel_scale = np.maximum(self.reference_max_vel, 1e-6)
        self.form_error_scale = 1.0
        self.spacing_error_scale = 1.0
        self.altitude_scale = 1.0

        half_spacing = self.formation_spacing / 2.0

        self.formation_offsets = np.array(
            [
                [0.0, -half_spacing, 0.0],
                [0.0, half_spacing, 0.0],
            ],
            dtype=np.float32,
        )

        self.center_A = np.array([0.0, 0.0, self.target_altitude], dtype=np.float32)
        self.center_B = np.array(
            [self.target_distance_x, 0.0, self.target_altitude],
            dtype=np.float32,
        )

        self.start_positions = self.center_A[None, :] + self.formation_offsets
        self.goal_positions = self.center_B[None, :] + self.formation_offsets

        self.initial_xyzs = self.start_positions.copy()
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

        self.obs_dim_per_agent = 24
        self.act_dim_per_agent = 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim_per_agent * self.num_agents,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim_per_agent * self.num_agents,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.prev_centroid_dist = None

        self.prev_action = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_final_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        self.last_valid_neighbor_pos = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.last_valid_neighbor_vel = np.zeros((self.num_agents, 3), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.prev_action = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_final_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        sim_obs = self._safe_reset(seed=seed, options=options)
        sim_obs = self._to_numpy_obs(sim_obs)

        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]
        centroid = np.mean(positions, axis=0)

        self.prev_centroid_dist = float(np.linalg.norm(self.center_B - centroid))

        for i in range(self.num_agents):
            j = 1 - i
            self.last_valid_neighbor_pos[i] = positions[j] - positions[i]
            self.last_valid_neighbor_vel[i] = velocities[j] - velocities[i]

        reference_velocity, formation_tracking_error = self._compute_reference_velocity(sim_obs)
        self.prev_reference_velocity = reference_velocity.copy()

        obs = self._build_obs(
            sim_obs=sim_obs,
            reference_velocity=reference_velocity,
            formation_tracking_error=formation_tracking_error,
        )

        return obs, {}

    def step(self, action):
        self.step_count += 1

        residual_action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, 3)
        residual_action = np.clip(residual_action, -1.0, 1.0)

        current_sim_obs = self._get_current_sim_obs()
        reference_velocity, formation_tracking_error = self._compute_reference_velocity(current_sim_obs)

        residual_velocity = self.residual_scale * residual_action
        final_velocity = reference_velocity + residual_velocity

        max_final_velocity = self.reference_max_vel + self.residual_scale
        final_velocity = np.clip(
            final_velocity,
            -max_final_velocity[None, :],
            max_final_velocity[None, :],
        ).astype(np.float32)

        rpm_action = np.zeros((self.num_agents, 4), dtype=np.float32)

        for i in range(self.num_agents):
            state = self.env._getDroneStateVector(i)

            cur_pos = state[0:3]
            cur_quat = state[3:7]
            cur_vel = state[10:13]
            cur_ang_vel = state[13:16]

            cmd_vel = final_velocity[i].copy()

            if cur_pos[2] < 0.80 and cmd_vel[2] < 0.0:
                cmd_vel[2] = 0.0

            if cur_pos[2] > 1.20 and cmd_vel[2] > 0.0:
                cmd_vel[2] = 0.0

            target_pos = cur_pos + cmd_vel * self.velocity_lookahead

            target_pos[0] = np.clip(target_pos[0], -1.0, 3.4)
            target_pos[1] = np.clip(target_pos[1], -1.2, 1.2)
            target_pos[2] = np.clip(target_pos[2], 0.75, 1.25)

            rpm, _, _ = self.controllers[i].computeControl(
                control_timestep=1.0 / self.ctrl_freq,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=np.zeros(3),
                target_vel=cmd_vel,
                target_rpy_rates=np.zeros(3),
            )

            rpm_action[i] = rpm

        step_result = self.env.step(rpm_action)

        if len(step_result) == 5:
            sim_obs, _, terminated, truncated, _ = step_result
        else:
            sim_obs, _, done, _ = step_result
            terminated = bool(done)
            truncated = False

        sim_obs = self._to_numpy_obs(sim_obs)

        reward, done, info = self._compute_reward_done_info(
            sim_obs=sim_obs,
            residual_action=residual_action,
            reference_velocity=reference_velocity,
            final_velocity=final_velocity,
            formation_tracking_error=formation_tracking_error,
        )

        next_reference_velocity, next_formation_tracking_error = self._compute_reference_velocity(sim_obs)

        obs = self._build_obs(
            sim_obs=sim_obs,
            reference_velocity=next_reference_velocity,
            formation_tracking_error=next_formation_tracking_error,
        )

        self.prev_action = residual_action.copy()
        self.prev_reference_velocity = next_reference_velocity.copy()
        self.prev_final_velocity = final_velocity.copy()

        terminated = bool(done or terminated)
        truncated = bool(truncated or self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, info

    def _safe_reset(self, seed=None, options=None):
        try:
            sim_obs, _ = self.env.reset(seed=seed, options=options)
            return sim_obs
        except TypeError:
            try:
                sim_obs, _ = self.env.reset(seed=seed)
                return sim_obs
            except TypeError:
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    return reset_result[0]
                return reset_result

    def _get_current_sim_obs(self):
        rows = []
        for i in range(self.num_agents):
            rows.append(self.env._getDroneStateVector(i))
        return np.asarray(rows, dtype=np.float32)

    def _compute_reference_velocity(self, sim_obs):
        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]

        centroid = np.mean(positions, axis=0)
        centroid_error = self.center_B - centroid

        centroid_velocity_cmd = self.kp_goal * centroid_error
        centroid_velocity_cmd[2] = self.kp_alt * (self.target_altitude - centroid[2])

        desired_positions = centroid[None, :] + self.formation_offsets
        formation_tracking_error = desired_positions - positions

        reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        for i in range(self.num_agents):
            form_cmd = self.kp_form * formation_tracking_error[i]
            reference_velocity[i] = centroid_velocity_cmd + form_cmd

        reference_velocity = np.clip(
            reference_velocity,
            -self.reference_max_vel[None, :],
            self.reference_max_vel[None, :],
        )

        return reference_velocity.astype(np.float32), formation_tracking_error.astype(np.float32)

    def _build_obs(self, sim_obs, reference_velocity, formation_tracking_error):
        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]

        obs_parts = []

        for i in range(self.num_agents):
            j = 1 - i

            own_pos = positions[i]
            own_vel = velocities[i]

            rel_goal = self.goal_positions[i] - own_pos
            altitude_error = np.array(
                [self.target_altitude - own_pos[2]],
                dtype=np.float32,
            )

            true_rel_neighbor_pos = positions[j] - positions[i]
            true_rel_neighbor_vel = velocities[j] - velocities[i]

            neighbor_valid = 1.0

            if self.use_neighbor_dropout and self.render_mode == "headless":
                if np.random.rand() < self.neighbor_dropout_prob:
                    neighbor_valid = 0.0

            if neighbor_valid > 0.5:
                rel_neighbor_pos = true_rel_neighbor_pos
                rel_neighbor_vel = true_rel_neighbor_vel
                self.last_valid_neighbor_pos[i] = rel_neighbor_pos.copy()
                self.last_valid_neighbor_vel[i] = rel_neighbor_vel.copy()
            else:
                rel_neighbor_pos = self.last_valid_neighbor_pos[i].copy()
                rel_neighbor_vel = self.last_valid_neighbor_vel[i].copy()

            _, pair_distance, _ = self._compute_spacing_metrics(positions)
            spacing_error = np.array(
                [pair_distance - self.formation_spacing],
                dtype=np.float32,
            )

            rel_goal_norm = np.clip(rel_goal / self.goal_scale, -1.0, 1.0)
            own_vel_norm = np.clip(own_vel / self.reference_vel_scale, -1.0, 1.0)
            altitude_error_norm = np.clip(altitude_error / self.altitude_scale, -1.0, 1.0)

            rel_neighbor_pos_norm = np.clip(
                rel_neighbor_pos / self.neighbor_pos_scale,
                -1.0,
                1.0,
            )

            rel_neighbor_vel_norm = np.clip(
                rel_neighbor_vel / self.neighbor_vel_scale,
                -1.0,
                1.0,
            )

            reference_velocity_norm = np.clip(
                reference_velocity[i] / self.reference_vel_scale,
                -1.0,
                1.0,
            )

            formation_tracking_error_norm = np.clip(
                formation_tracking_error[i] / self.form_error_scale,
                -1.0,
                1.0,
            )

            spacing_error_norm = np.clip(
                spacing_error / self.spacing_error_scale,
                -1.0,
                1.0,
            )

            obs_i = np.concatenate(
                [
                    rel_goal_norm.astype(np.float32),
                    own_vel_norm.astype(np.float32),
                    altitude_error_norm.astype(np.float32),
                    rel_neighbor_pos_norm.astype(np.float32),
                    rel_neighbor_vel_norm.astype(np.float32),
                    np.array([neighbor_valid], dtype=np.float32),
                    self.prev_action[i].astype(np.float32),
                    reference_velocity_norm.astype(np.float32),
                    formation_tracking_error_norm.astype(np.float32),
                    spacing_error_norm.astype(np.float32),
                ],
                axis=0,
            ).astype(np.float32)

            obs_parts.append(obs_i)

        return np.concatenate(obs_parts, axis=0).astype(np.float32)

    def _compute_formation_error(self, positions):
        centroid = np.mean(positions, axis=0)
        desired_positions = centroid[None, :] + self.formation_offsets
        per_agent_error = np.linalg.norm(positions - desired_positions, axis=1)
        return float(np.mean(per_agent_error)), per_agent_error

    def _compute_spacing_metrics(self, positions):
        d01 = float(np.linalg.norm(positions[0] - positions[1]))
        spacing_error = abs(d01 - self.formation_spacing)
        return float(spacing_error), d01, self.formation_spacing

    def _compute_min_pair_distance(self, positions):
        d01 = float(np.linalg.norm(positions[0] - positions[1]))
        collision_count = int(d01 < self.min_separation)
        return d01, collision_count

    def _compute_reward_done_info(
        self,
        sim_obs,
        residual_action,
        reference_velocity,
        final_velocity,
        formation_tracking_error,
    ):
        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]
        rpys = sim_obs[:, 7:10]

        centroid = np.mean(positions, axis=0)
        centroid_dist = float(np.linalg.norm(self.center_B - centroid))

        if self.prev_centroid_dist is None:
            raw_progress = 0.0
        else:
            raw_progress = self.prev_centroid_dist - centroid_dist

        self.prev_centroid_dist = centroid_dist

        own_goal_distances = np.linalg.norm(self.goal_positions - positions, axis=1)
        mean_own_goal_dist = float(np.mean(own_goal_distances))
        max_own_goal_dist = float(np.max(own_goal_distances))

        formation_error, per_agent_formation_error = self._compute_formation_error(positions)
        spacing_error, pair_distance, desired_spacing = self._compute_spacing_metrics(positions)
        min_pair_dist, collision_count = self._compute_min_pair_distance(positions)

        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))
        max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        mean_ref_speed = float(np.mean(np.linalg.norm(reference_velocity, axis=1)))
        mean_cmd_speed = float(np.mean(np.linalg.norm(final_velocity, axis=1)))
        mean_residual = float(np.mean(np.linalg.norm(residual_action, axis=1)))
        action_smoothness = float(np.mean(np.linalg.norm(residual_action - self.prev_action, axis=1)))
        mean_attitude_error = float(np.mean(np.linalg.norm(rpys[:, 0:2], axis=1)))
        mean_tracking_error = float(np.mean(np.linalg.norm(formation_tracking_error, axis=1)))

        out_of_bounds = bool(
            np.any(positions[:, 0] < -1.2)
            or np.any(positions[:, 0] > 3.6)
            or np.any(np.abs(positions[:, 1]) > 1.4)
            or np.any(positions[:, 2] < 0.65)
            or np.any(positions[:, 2] > 1.35)
        )

        unstable_attitude = bool(
            np.any(np.abs(rpys[:, 0]) > 1.20)
            or np.any(np.abs(rpys[:, 1]) > 1.20)
        )

        crashed = bool(
            collision_count > 0
            or out_of_bounds
            or unstable_attitude
        )

        formation_safe = bool(
            formation_error < 0.30
            and spacing_error < 0.25
            and collision_count == 0
            and not out_of_bounds
            and not unstable_attitude
        )

        gated_progress = raw_progress if formation_safe else min(raw_progress, 0.0)

        success = bool(
            centroid_dist < self.success_centroid_tol
            and formation_error < self.success_formation_tol
            and spacing_error < self.success_spacing_tol
            and mean_speed < 0.65
            and collision_count == 0
            and not crashed
        )

        reward = 0.0

        reward += self.alive_reward * self.num_agents
        reward += self.progress_weight * gated_progress
        reward += -self.centroid_weight * centroid_dist
        reward += -self.own_goal_weight * mean_own_goal_dist
        reward += -self.formation_weight * formation_error
        reward += -self.spacing_weight * spacing_error
        reward += -self.velocity_weight * mean_speed
        reward += -self.command_weight * mean_cmd_speed
        reward += -self.residual_weight * mean_residual
        reward += -self.smoothness_weight * action_smoothness
        reward += -self.attitude_weight * mean_attitude_error

        if formation_error < 0.20 and spacing_error < 0.18 and collision_count == 0:
            reward += 10.0

        if formation_error < 0.12 and spacing_error < 0.10 and collision_count == 0:
            reward += 20.0

        if centroid_dist < 1.50 and formation_safe:
            reward += 15.0

        if centroid_dist < 0.75 and formation_safe:
            reward += 30.0

        if collision_count > 0:
            reward -= self.collision_penalty * collision_count

        if out_of_bounds:
            reward -= self.boundary_penalty

        if unstable_attitude:
            reward -= self.crash_penalty

        if crashed:
            reward -= self.crash_penalty

        if success:
            reward += self.goal_bonus

        done = bool(
            success
            or crashed
            or self.step_count >= self.max_steps
        )

        info = {
            "is_success": success,
            "centroid_error": centroid_dist,
            "mean_dist_to_target": mean_own_goal_dist,
            "max_dist_to_target": max_own_goal_dist,
            "formation_error": float(formation_error),
            "max_agent_formation_error": float(np.max(per_agent_formation_error)),
            "spacing_error": float(spacing_error),
            "pair_distance": float(pair_distance),
            "desired_spacing": float(desired_spacing),
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "mean_ref_speed": mean_ref_speed,
            "mean_cmd_speed": mean_cmd_speed,
            "mean_residual": mean_residual,
            "action_smoothness": action_smoothness,
            "mean_attitude_error": mean_attitude_error,
            "mean_tracking_error": mean_tracking_error,
            "collision_count": int(collision_count),
            "min_pair_dist": float(min_pair_dist),
            "crashed": crashed,
            "out_of_bounds": out_of_bounds,
            "unstable_attitude": unstable_attitude,
            "raw_progress": float(raw_progress),
            "progress": float(gated_progress),
            "formation_safe": formation_safe,
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "target_center_x": float(self.center_B[0]),
            "target_center_y": float(self.center_B[1]),
            "target_center_z": float(self.center_B[2]),
            "uav0_x": float(positions[0, 0]),
            "uav1_x": float(positions[1, 0]),
            "uav0_y": float(positions[0, 1]),
            "uav1_y": float(positions[1, 1]),
            "uav0_z": float(positions[0, 2]),
            "uav1_z": float(positions[1, 2]),
        }

        return float(reward), done, info

    def _to_numpy_obs(self, sim_obs):
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

            if sim_obs.ndim == 1:
                sim_obs = sim_obs.reshape(self.num_agents, -1)

        return sim_obs.astype(np.float32)

    def render(self):
        return

    def close(self):
        self.env.close()