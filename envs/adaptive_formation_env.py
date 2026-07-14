import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from envs.formation_manager import FormationManager


class AdaptiveFormationEnv(gym.Env):
    """
    Three-UAV reference-guided adaptive formation environment.

    Mission:
        Phase 0: Take off in triangle formation
        Phase 1: Hover at the start in triangle formation
        Phase 2: Transition from triangle to line
        Phase 3: Fly to the goal in line formation
        Phase 4: Transition from line back to triangle
        Phase 5: Hover at the goal in triangle formation
        Phase 6: Land in triangle formation

    Control:
        v_final = v_reference + learned_velocity_correction

    This is the non-APF foundation. No obstacle repulsive force is applied.

    Actor observation per UAV remains 36-dimensional:
        relative_assigned_target(3)
        own_velocity(3)
        altitude_error(1)
        mission_phase_group_onehot(5)
        neighbor_1_relative_position(3)
        neighbor_1_relative_velocity(3)
        neighbor_1_valid_mask(1)
        neighbor_2_relative_position(3)
        neighbor_2_relative_velocity(3)
        neighbor_2_valid_mask(1)
        previous_action(3)
        reference_velocity(3)
        formation_tracking_error(3)
        spacing_error(1)

    Action per UAV:
        learned velocity correction [rx, ry, rz] in [-1, 1]
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        render_mode="headless",
        num_agents=3,
        max_steps=1200,
        ctrl_freq=48,
        sim_freq=240,
        mission_stage=3,
        target_altitude=1.0,
        start_x=0.0,
        goal_x=7.0,
        triangle_side=1.20,
        formation_spacing=0.60,
        transition_steps=160,
        reference_max_vx=0.30,
        reference_max_vy=0.22,
        reference_max_vz=0.14,
        correction_scale=0.06,
        velocity_lookahead=0.30,
        kp_center=0.38,
        kp_form=0.70,
        kp_alt=0.45,
        goal_bonus=1500.0,
        alive_reward=0.04,
        phase_bonus=250.0,
        progress_weight=45.0,
        center_weight=3.5,
        assigned_target_weight=1.8,
        formation_weight=30.0,
        spacing_weight=18.0,
        velocity_weight=0.28,
        command_weight=0.10,
        correction_weight=0.14,
        smoothness_weight=0.28,
        attitude_weight=0.90,
        collision_penalty=3000.0,
        crash_penalty=3500.0,
        boundary_penalty=1500.0,
        min_separation=0.30,
        neighbor_dropout_prob=0.02,
        use_neighbor_dropout=True,
    ):
        super().__init__()

        if num_agents != 3:
            raise ValueError("This environment expects exactly 3 UAVs.")

        self.render_mode = render_mode
        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.ctrl_freq = int(ctrl_freq)
        self.sim_freq = int(sim_freq)

        self.mission_stage = int(np.clip(mission_stage, 1, 3))

        self.target_altitude = float(target_altitude)
        self.landing_altitude = 0.08

        self.start_x = float(start_x)
        self.goal_x = float(goal_x)

        self.triangle_side = float(triangle_side)
        self.formation_spacing = float(formation_spacing)
        self.transition_steps = int(transition_steps)

        if self.formation_spacing <= 0.0:
            raise ValueError("formation_spacing must be greater than zero.")

        if self.transition_steps <= 0:
            raise ValueError("transition_steps must be greater than zero.")

        self.reference_max_vel = np.array(
            [reference_max_vx, reference_max_vy, reference_max_vz],
            dtype=np.float32,
        )

        self.correction_scale = float(correction_scale)
        self.velocity_lookahead = float(velocity_lookahead)

        self.kp_center = float(kp_center)
        self.kp_form = float(kp_form)
        self.kp_alt = float(kp_alt)

        self.goal_bonus = float(goal_bonus)
        self.alive_reward = float(alive_reward)
        self.phase_bonus = float(phase_bonus)
        self.progress_weight = float(progress_weight)
        self.center_weight = float(center_weight)
        self.assigned_target_weight = float(assigned_target_weight)
        self.formation_weight = float(formation_weight)
        self.spacing_weight = float(spacing_weight)
        self.velocity_weight = float(velocity_weight)
        self.command_weight = float(command_weight)
        self.correction_weight = float(correction_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.attitude_weight = float(attitude_weight)

        self.collision_penalty = float(collision_penalty)
        self.crash_penalty = float(crash_penalty)
        self.boundary_penalty = float(boundary_penalty)
        self.min_separation = float(min_separation)

        self.neighbor_dropout_prob = float(neighbor_dropout_prob)
        self.use_neighbor_dropout = bool(use_neighbor_dropout)

        self.goal_scale = max(self.goal_x, 1.0)
        self.neighbor_pos_scale = 2.5
        self.neighbor_vel_scale = 1.5
        self.reference_vel_scale = np.maximum(self.reference_max_vel, 1e-6)
        self.formation_error_scale = 1.5
        self.spacing_error_scale = 1.5
        self.altitude_scale = 1.0

        self.phase_names = [
            "takeoff_triangle",
            "start_hover_triangle",
            "triangle_approach",
            "transition_to_line",
            "line_corridor",
            "transition_to_triangle",
            "triangle_to_goal",
            "goal_hover_triangle",
            "landing_triangle",
        ]

        self.max_phase_for_stage = {
            1: 3,
            2: 7,
            3: 8,
        }[self.mission_stage]

        self.start_center = np.array(
            [self.start_x, 0.0, self.target_altitude],
            dtype=np.float32,
        )

        self.goal_center = np.array(
            [self.goal_x, 0.0, self.target_altitude],
            dtype=np.float32,
        )

        self.landing_center = np.array(
            [self.goal_x, 0.0, self.landing_altitude],
            dtype=np.float32,
        )

        self.formation_manager = FormationManager(
            initial_formation="triangle",
            spacing=self.formation_spacing,
            transition_steps=self.transition_steps,
        )

        initial_offsets = self.formation_manager.current_offsets.copy()

        self.initial_xyzs = np.array(
            [
                self.start_center + initial_offsets[0] + np.array([0.0, 0.0, -0.95]),
                self.start_center + initial_offsets[1] + np.array([0.0, 0.0, -0.95]),
                self.start_center + initial_offsets[2] + np.array([0.0, 0.0, -0.95]),
            ],
            dtype=np.float32,
        )

        self.initial_xyzs[:, 2] = 0.05
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

        self.obs_dim_per_agent = 36
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
        self.phase = 0
        self.phase_hold_counter = 0
        self.last_phase_completed = False
        self.mission_success = False
        self.prev_center_dist = None

        self.prev_action = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_final_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        self.last_valid_neighbor_pos = np.zeros(
            (self.num_agents, self.num_agents, 3),
            dtype=np.float32,
        )

        self.last_valid_neighbor_vel = np.zeros(
            (self.num_agents, self.num_agents, 3),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.phase = 0
        self.phase_hold_counter = 0
        self.last_phase_completed = False
        self.mission_success = False
        self.prev_center_dist = None

        self.prev_action = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_final_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        self.formation_manager.reset("triangle")

        sim_obs = self._safe_reset(seed=seed, options=options)
        sim_obs = self._to_numpy_obs(sim_obs)

        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]

        target_center, _ = self._phase_target_and_offsets()
        centroid = np.mean(positions, axis=0)
        self.prev_center_dist = float(np.linalg.norm(target_center - centroid))

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                self.last_valid_neighbor_pos[i, j] = positions[j] - positions[i]
                self.last_valid_neighbor_vel[i, j] = velocities[j] - velocities[i]

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

        learned_action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, 3)
        learned_action = np.clip(learned_action, -1.0, 1.0)

        # Advance a requested formation transition exactly once per environment step.
        self.formation_manager.update()

        current_sim_obs = self._get_current_sim_obs()
        reference_velocity, formation_tracking_error = self._compute_reference_velocity(current_sim_obs)

        phase_correction_scale = self.correction_scale

        if self.phase == 7:
            phase_correction_scale = min(self.correction_scale, 0.04)

        elif self.phase == 8:
            phase_correction_scale = min(self.correction_scale, 0.03)

        correction_velocity = phase_correction_scale * learned_action

        final_velocity = reference_velocity + correction_velocity

        max_final_velocity = self.reference_max_vel + phase_correction_scale

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

            if self.phase != 6:
                if cur_pos[2] < 0.15 and cmd_vel[2] < 0.0:
                    cmd_vel[2] = 0.0

                if cur_pos[2] > 1.25 and cmd_vel[2] > 0.0:
                    cmd_vel[2] = 0.0
            else:
                if cur_pos[2] < 0.05 and cmd_vel[2] < 0.0:
                    cmd_vel[2] = 0.0

            target_pos = cur_pos + cmd_vel * self.velocity_lookahead

            target_pos[0] = np.clip(target_pos[0], -1.0, self.goal_x + 1.2)
            target_pos[1] = np.clip(target_pos[1], -2.5, 2.5)

            if self.phase == 8:
                target_pos[2] = np.clip(target_pos[2], 0.04, 1.20)
            else:
                target_pos[2] = np.clip(target_pos[2], 0.05, 1.35)

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
            learned_action=learned_action,
            reference_velocity=reference_velocity,
            final_velocity=final_velocity,
            formation_tracking_error=formation_tracking_error,
        )

        self._update_phase(sim_obs, info)

        next_reference_velocity, next_formation_tracking_error = self._compute_reference_velocity(sim_obs)

        obs = self._build_obs(
            sim_obs=sim_obs,
            reference_velocity=next_reference_velocity,
            formation_tracking_error=next_formation_tracking_error,
        )

        self.prev_action = learned_action.copy()
        self.prev_reference_velocity = next_reference_velocity.copy()
        self.prev_final_velocity = final_velocity.copy()

        if self.mission_success:
            done = True
            info["is_success"] = True
            info["mission_success"] = True
            reward += self.goal_bonus

        terminated = bool(done or terminated)
        truncated = bool(truncated or self.step_count >= self.max_steps)

        return obs, float(reward), terminated, truncated, info

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

    def _phase_target_and_offsets(self):
        offsets = self.formation_manager.current_offsets.copy()

        switch_to_line_x = float(
            getattr(getattr(self, "scenario", None), "switch_to_line_x", 2.35)
        )
        restore_triangle_x = float(
            getattr(getattr(self, "scenario", None), "restore_triangle_x", 4.65)
        )

        if self.phase in (0, 1):
            return self.start_center.copy(), offsets

        if self.phase in (2, 3):
            target = np.array(
                [switch_to_line_x, 0.0, self.target_altitude],
                dtype=np.float32,
            )
            return target, offsets

        if self.phase in (4, 5):
            target = np.array(
                [restore_triangle_x, 0.0, self.target_altitude],
                dtype=np.float32,
            )
            return target, offsets

        if self.phase in (6, 7):
            return self.goal_center.copy(), offsets

        return self.landing_center.copy(), offsets

    def _desired_pair_distances(self, formation_offsets):
        distances = {}

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distances[(i, j)] = float(np.linalg.norm(formation_offsets[i] - formation_offsets[j]))

        return distances

    def _compute_reference_velocity(self, sim_obs):
        positions = sim_obs[:, 0:3]

        target_center, formation_offsets = self._phase_target_and_offsets()
        centroid = np.mean(positions, axis=0)

        center_error = target_center - centroid

        center_velocity_cmd = self.kp_center * center_error
        center_velocity_cmd[2] = self.kp_alt * center_error[2]

        # During formation transitions, slow only the swarm centroid motion.
        # Do not suppress the per-UAV x-axis formation correction; the line
        # requires agents to separate longitudinally while converging laterally.
        if self.phase in (3, 5):
            center_velocity_cmd[0] *= 0.05
            center_velocity_cmd[1] *= 0.20

        desired_positions = target_center[None, :] + formation_offsets
        formation_tracking_error = desired_positions - positions

        reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        for i in range(self.num_agents):
            form_cmd = self.kp_form * formation_tracking_error[i]
            reference_velocity[i] = center_velocity_cmd + form_cmd

        if self.phase == 1:
            reference_velocity[:, 0] *= 0.30

        elif self.phase == 7:
            reference_velocity[:, 0] *= 0.20

        if self.phase == 8:
            reference_velocity[:, 0] *= 0.15
            reference_velocity[:, 1] *= 0.45
            reference_velocity[:, 2] = np.minimum(reference_velocity[:, 2], -0.05)

        reference_velocity = np.clip(
            reference_velocity,
            -self.reference_max_vel[None, :],
            self.reference_max_vel[None, :],
        )

        return reference_velocity.astype(np.float32), formation_tracking_error.astype(np.float32)

    def _build_obs(self, sim_obs, reference_velocity, formation_tracking_error):
        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]

        target_center, formation_offsets = self._phase_target_and_offsets()
        desired_positions = target_center[None, :] + formation_offsets

        spacing_error = np.array(
            [self._compute_spacing_error(positions, formation_offsets)],
            dtype=np.float32,
        )

        # Preserve the original 5-value phase representation while the
        # detailed Easy City mission uses nine internal phases.
        phase_group = {
            0: 0,  # takeoff
            1: 1,  # start hover
            2: 2,  # triangle approach
            3: 2,  # transition to line
            4: 2,  # line corridor
            5: 2,  # transition to triangle
            6: 2,  # triangle to goal
            7: 3,  # goal hover
            8: 4,  # landing
        }[self.phase]

        phase_onehot = np.zeros(5, dtype=np.float32)
        phase_onehot[phase_group] = 1.0

        obs_parts = []

        for i in range(self.num_agents):
            own_pos = positions[i]
            own_vel = velocities[i]

            rel_target = desired_positions[i] - own_pos

            altitude_error = np.array(
                [target_center[2] - own_pos[2]],
                dtype=np.float32,
            )

            neighbor_parts = []

            for j in range(self.num_agents):
                if j == i:
                    continue

                true_rel_neighbor_pos = positions[j] - positions[i]
                true_rel_neighbor_vel = velocities[j] - velocities[i]

                neighbor_valid = 1.0

                if self.use_neighbor_dropout and self.render_mode == "headless":
                    if np.random.rand() < self.neighbor_dropout_prob:
                        neighbor_valid = 0.0

                if neighbor_valid > 0.5:
                    rel_neighbor_pos = true_rel_neighbor_pos
                    rel_neighbor_vel = true_rel_neighbor_vel

                    self.last_valid_neighbor_pos[i, j] = rel_neighbor_pos.copy()
                    self.last_valid_neighbor_vel[i, j] = rel_neighbor_vel.copy()
                else:
                    rel_neighbor_pos = self.last_valid_neighbor_pos[i, j].copy()
                    rel_neighbor_vel = self.last_valid_neighbor_vel[i, j].copy()

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

                neighbor_parts.extend(
                    [
                        rel_neighbor_pos_norm.astype(np.float32),
                        rel_neighbor_vel_norm.astype(np.float32),
                        np.array([neighbor_valid], dtype=np.float32),
                    ]
                )

            rel_target_norm = np.clip(rel_target / self.goal_scale, -1.0, 1.0)
            own_vel_norm = np.clip(own_vel / self.reference_vel_scale, -1.0, 1.0)
            altitude_error_norm = np.clip(altitude_error / self.altitude_scale, -1.0, 1.0)

            reference_velocity_norm = np.clip(
                reference_velocity[i] / self.reference_vel_scale,
                -1.0,
                1.0,
            )

            formation_tracking_error_norm = np.clip(
                formation_tracking_error[i] / self.formation_error_scale,
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
                    rel_target_norm.astype(np.float32),
                    own_vel_norm.astype(np.float32),
                    altitude_error_norm.astype(np.float32),
                    phase_onehot.astype(np.float32),
                    neighbor_parts[0],
                    neighbor_parts[1],
                    neighbor_parts[2],
                    neighbor_parts[3],
                    neighbor_parts[4],
                    neighbor_parts[5],
                    self.prev_action[i].astype(np.float32),
                    reference_velocity_norm.astype(np.float32),
                    formation_tracking_error_norm.astype(np.float32),
                    spacing_error_norm.astype(np.float32),
                ],
                axis=0,
            ).astype(np.float32)

            obs_parts.append(obs_i)

        return np.concatenate(obs_parts, axis=0).astype(np.float32)

    def _compute_formation_error(self, positions, formation_offsets):
        target_center, _ = self._phase_target_and_offsets()
        desired_positions = target_center[None, :] + formation_offsets
        per_agent_error = np.linalg.norm(positions - desired_positions, axis=1)
        return float(np.mean(per_agent_error)), per_agent_error

    def _compute_spacing_error(self, positions, formation_offsets):
        desired = self._desired_pair_distances(formation_offsets)

        errors = []

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                actual = float(np.linalg.norm(positions[i] - positions[j]))
                errors.append(abs(actual - desired[(i, j)]))

        return float(np.mean(errors))

    def _compute_pair_metrics(self, positions):
        min_dist = np.inf
        collision_count = 0

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                min_dist = min(min_dist, dist)

                if dist < self.min_separation:
                    collision_count += 1

        return float(min_dist), int(collision_count)

    def _compute_reward_done_info(
        self,
        sim_obs,
        learned_action,
        reference_velocity,
        final_velocity,
        formation_tracking_error,
    ):
        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]
        rpys = sim_obs[:, 7:10]

        target_center, formation_offsets = self._phase_target_and_offsets()
        desired_positions = target_center[None, :] + formation_offsets

        centroid = np.mean(positions, axis=0)
        center_dist = float(np.linalg.norm(target_center - centroid))

        if self.prev_center_dist is None:
            raw_progress = 0.0
        else:
            raw_progress = self.prev_center_dist - center_dist

        self.prev_center_dist = center_dist

        assigned_target_distances = np.linalg.norm(desired_positions - positions, axis=1)
        mean_assigned_target_dist = float(np.mean(assigned_target_distances))
        max_assigned_target_dist = float(np.max(assigned_target_distances))

        formation_error, per_agent_formation_error = self._compute_formation_error(
            positions,
            formation_offsets,
        )

        spacing_error = self._compute_spacing_error(positions, formation_offsets)
        min_pair_dist, collision_count = self._compute_pair_metrics(positions)

        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))
        max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        mean_ref_speed = float(np.mean(np.linalg.norm(reference_velocity, axis=1)))
        mean_cmd_speed = float(np.mean(np.linalg.norm(final_velocity, axis=1)))
        mean_correction = float(np.mean(np.linalg.norm(learned_action, axis=1)))
        action_smoothness = float(np.mean(np.linalg.norm(learned_action - self.prev_action, axis=1)))
        mean_attitude_error = float(np.mean(np.linalg.norm(rpys[:, 0:2], axis=1)))
        mean_tracking_error = float(np.mean(np.linalg.norm(formation_tracking_error, axis=1)))

        out_of_bounds = bool(
            np.any(positions[:, 0] < -1.5)
            or np.any(positions[:, 0] > self.goal_x + 1.5)
            or np.any(np.abs(positions[:, 1]) > 2.8)
            or np.any(positions[:, 2] < 0.02)
            or np.any(positions[:, 2] > 1.65)
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
            formation_error < 0.38
            and spacing_error < 0.30
            and collision_count == 0
            and not out_of_bounds
            and not unstable_attitude
        )

        gated_progress = raw_progress if formation_safe else min(raw_progress, 0.0)

        reward = 0.0

        reward += self.alive_reward * self.num_agents
        reward += self.progress_weight * gated_progress
        reward += -self.center_weight * center_dist
        reward += -self.assigned_target_weight * mean_assigned_target_dist
        reward += -self.formation_weight * formation_error
        reward += -self.spacing_weight * spacing_error
        reward += -self.velocity_weight * mean_speed
        reward += -self.command_weight * mean_cmd_speed
        reward += -self.correction_weight * mean_correction
        reward += -self.smoothness_weight * action_smoothness
        reward += -self.attitude_weight * mean_attitude_error

        if formation_error < 0.25 and spacing_error < 0.22 and collision_count == 0:
            reward += 8.0

        if formation_error < 0.15 and spacing_error < 0.12 and collision_count == 0:
            reward += 18.0

        if center_dist < 0.75 and formation_safe:
            reward += 20.0

        if center_dist < 0.40 and formation_safe:
            reward += 35.0

        if self.last_phase_completed:
            reward += self.phase_bonus
            self.last_phase_completed = False

        if collision_count > 0:
            reward -= self.collision_penalty * collision_count

        if out_of_bounds:
            reward -= self.boundary_penalty

        if unstable_attitude:
            reward -= self.crash_penalty

        if crashed:
            reward -= self.crash_penalty

        done = bool(
            crashed
            or self.step_count >= self.max_steps
        )

        info = {
            "is_success": False,
            "mission_success": False,
            "phase": int(self.phase),
            "phase_name": self.phase_names[self.phase],
            "current_formation": self.formation_manager.current_formation,
            "target_formation": self.formation_manager.target_formation,
            "formation_transitioning": bool(self.formation_manager.transitioning),
            "formation_transition_progress": float(
                self.formation_manager.get_transition_progress()
            ),
            "center_error": center_dist,
            "mean_dist_to_target": mean_assigned_target_dist,
            "max_dist_to_target": max_assigned_target_dist,
            "formation_error": float(formation_error),
            "max_agent_formation_error": float(np.max(per_agent_formation_error)),
            "spacing_error": float(spacing_error),
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "mean_ref_speed": mean_ref_speed,
            "mean_cmd_speed": mean_cmd_speed,
            "mean_correction": mean_correction,
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
            "target_center_x": float(target_center[0]),
            "target_center_y": float(target_center[1]),
            "target_center_z": float(target_center[2]),
            "uav0_x": float(positions[0, 0]),
            "uav1_x": float(positions[1, 0]),
            "uav2_x": float(positions[2, 0]),
            "uav0_y": float(positions[0, 1]),
            "uav1_y": float(positions[1, 1]),
            "uav2_y": float(positions[2, 1]),
            "uav0_z": float(positions[0, 2]),
            "uav1_z": float(positions[1, 2]),
            "uav2_z": float(positions[2, 2]),
        }

        return float(reward), done, info

    def _update_phase(self, sim_obs, info):
        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]

        centroid = np.mean(positions, axis=0)
        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))

        phase = self.phase
        completed = False

        center_error = float(info.get("center_error", 999.0))
        formation_error = float(info.get("formation_error", 999.0))
        spacing_error = float(info.get("spacing_error", 999.0))
        collision_count = int(info.get("collision_count", 99))
        crashed = bool(info.get("crashed", False))

        stable_formation = (
            formation_error < 0.38
            and spacing_error < 0.30
            and collision_count == 0
            and not crashed
        )

        switch_to_line_x = float(
            getattr(getattr(self, "scenario", None), "switch_to_line_x", 2.35)
        )
        restore_triangle_x = float(
            getattr(getattr(self, "scenario", None), "restore_triangle_x", 4.65)
        )

        if phase == 0:
            completed = bool(
                np.mean(positions[:, 2]) > 0.90
                and abs(np.mean(positions[:, 2]) - self.target_altitude) < 0.18
                and stable_formation
            )

        elif phase == 1:
            completed = bool(stable_formation and mean_speed < 0.45)

        elif phase == 2:
            completed = bool(
                centroid[0] >= switch_to_line_x - 0.15
                and stable_formation
            )

        elif phase == 3:
            completed = bool(
                not self.formation_manager.transitioning
                and self.formation_manager.current_formation == "line"
                and spacing_error < 0.15
                and float(info.get("min_pair_dist", 0.0)) > self.min_separation + 0.10
                and mean_speed < 0.25
                and collision_count == 0
                and not crashed
            )

        elif phase == 4:
            completed = bool(
                centroid[0] >= restore_triangle_x - 0.15
                and stable_formation
            )

        elif phase == 5:
            completed = bool(
                not self.formation_manager.transitioning
                and self.formation_manager.current_formation == "triangle"
                and spacing_error < 0.15
                and float(info.get("min_pair_dist", 0.0)) > self.min_separation + 0.10
                and mean_speed < 0.25
                and collision_count == 0
                and not crashed
            )

        elif phase == 6:
            completed = bool(
                centroid[0] >= self.goal_x - 0.35
                and stable_formation
            )

        elif phase == 7:
            completed = bool(
                center_error < 0.45
                and stable_formation
                and mean_speed < 0.45
            )

        elif phase == 8:
            mean_altitude = float(np.mean(positions[:, 2]))
            max_altitude = float(np.max(positions[:, 2]))

            completed = bool(
                mean_altitude < 0.16
                and max_altitude < 0.22
                and mean_speed < 0.35
                and collision_count == 0
                and not crashed
            )

        if completed:
            self.phase_hold_counter += 1
        else:
            self.phase_hold_counter = 0

        hold_needed = {
            0: 18,
            1: 35,
            2: 12,
            3: 40,
            4: 12,
            5: 40,
            6: 12,
            7: 35,
            8: 8,
        }[phase]

        if self.phase_hold_counter >= hold_needed:
            self.phase_hold_counter = 0
            self.last_phase_completed = True

            if self.phase >= self.max_phase_for_stage:
                self.mission_success = True
                return

            self.phase += 1

            if self.phase == 3:
                self.formation_manager.request_formation("line")

            elif self.phase == 5:
                self.formation_manager.request_formation("triangle")

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

# Backward-compatible alias for scripts that still import the old class name.
MultiUAVVelocityFormationEnv = AdaptiveFormationEnv
