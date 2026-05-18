import numpy as np
import gymnasium as gym
from gymnasium import spaces

# gym-pybullet-drones imports
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiUAVRealisticEnv(gym.Env):
    """
    v0.4 Multi-UAV straight-line formation + landing environment.

    Platform:
        gym-pybullet-drones CtrlAviary with Crazyflie 2.x model.

    Mission:
        3 UAVs operate as one formation group.

        Phase 0: take off and establish straight-line formation at A.
        Phase 1: move the formation center from A to B while preserving line geometry.
        Phase 2: land the formation at B.

    Formation:
        The line is defined by fixed offsets around a moving formation center:
            UAV 0: center / leader reference
            UAV 1: left follower
            UAV 2: right follower

        Default offsets:
            [0.0,  0.0, 0.0]
            [0.0, -0.6, 0.0]
            [0.0,  0.6, 0.0]

    High-level action:
        Each agent outputs a small xyz residual correction around its assigned
        formation target. The residual is deliberately small so that learning
        improves tracking without destroying the formation objective.

    Low-level control:
        DSLPIDControl converts target positions into motor RPMs.

    Observation per UAV:
        pos(3), vel(3), rpy(3), rel_own_target(3), rel_centroid_target(3),
        formation_error(1), phase(1) = 17

    Joint observation:
        Concatenation of all per-agent observations.
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        render_mode="headless",
        num_agents=3,
        max_steps=700,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.035, 0.035, 0.04),
        # Mission rewards
        goal_bonus=700.0,
        phase_bonus=100.0,
        alive_reward=0.10,
        centroid_weight=4.0,
        distance_weight=2.0,
        formation_weight=8.0,
        velocity_weight=1.0,
        attitude_weight=1.0,
        smoothness_weight=0.05,
        # Safety rewards / penalties
        collision_penalty=350.0,
        min_separation=0.30,
        max_formation_error_for_success=0.18,
    ):
        super().__init__()

        if num_agents != 3:
            raise ValueError("v0.4 straight-line formation expects exactly 3 UAVs.")

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.ctrl_freq = ctrl_freq
        self.sim_freq = sim_freq

        self.action_scales = np.array(action_scales, dtype=np.float32)

        self.goal_bonus = goal_bonus
        self.phase_bonus = phase_bonus
        self.alive_reward = alive_reward
        self.centroid_weight = centroid_weight
        self.distance_weight = distance_weight
        self.formation_weight = formation_weight
        self.velocity_weight = velocity_weight
        self.attitude_weight = attitude_weight
        self.smoothness_weight = smoothness_weight

        self.collision_penalty = collision_penalty
        self.min_separation = min_separation
        self.max_formation_error_for_success = max_formation_error_for_success

        # -----------------------------
        # v0.4 formation geometry
        # -----------------------------
        self.formation_offsets = np.array(
            [
                [0.0,  0.0, 0.0],
                [0.0, -0.6, 0.0],
                [0.0,  0.6, 0.0],
            ],
            dtype=np.float32,
        )

        self.center_A_takeoff = np.array([0.0, 0.0, 1.00], dtype=np.float32)
        self.center_B_hover = np.array([2.2, 0.0, 1.00], dtype=np.float32)
        self.center_B_land = np.array([2.2, 0.0, 0.10], dtype=np.float32)

        self.start_center = np.array([0.0, 0.0, 0.20], dtype=np.float32)
        self.start_positions = self.start_center[None, :] + self.formation_offsets

        self.hover_A_targets = self.center_A_takeoff[None, :] + self.formation_offsets
        self.hover_B_targets = self.center_B_hover[None, :] + self.formation_offsets
        self.land_B_targets = self.center_B_land[None, :] + self.formation_offsets

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

        # Action = xyz residual per UAV
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 * self.num_agents,),
            dtype=np.float32,
        )

        # Observation per UAV:
        # pos(3), vel(3), rpy(3), rel_own_target(3), rel_centroid_target(3),
        # formation_error(1), phase(1) = 17
        self.obs_dim_per_agent = 17

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim_per_agent * self.num_agents,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.phase = 0
        self.phase_counter = 0
        self.current_targets = self.hover_A_targets.copy()
        self.prev_action = np.zeros((self.num_agents, 3), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.phase = 0
        self.phase_counter = 0
        self.current_targets = self.hover_A_targets.copy()
        self.prev_action = np.zeros((self.num_agents, 3), dtype=np.float32)

        sim_obs, _ = self.env.reset(seed=seed, options=options)
        obs = self._build_obs(sim_obs)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        self.phase_counter += 1

        action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, 3)
        action = np.clip(action, -1.0, 1.0)

        base_targets = self._get_base_targets()
        residual = action * self.action_scales[None, :]
        self.current_targets = base_targets + residual

        # Keep all targets inside a safe training volume while preserving the line corridor.
        for i in range(self.num_agents):
            self.current_targets[i, 0] = np.clip(self.current_targets[i, 0], -0.3, 2.5)
            self.current_targets[i, 1] = np.clip(self.current_targets[i, 1], -1.0, 1.0)

            if self.phase == 2:
                self.current_targets[i, 2] = np.clip(self.current_targets[i, 2], 0.08, 1.10)
            else:
                self.current_targets[i, 2] = np.clip(self.current_targets[i, 2], 0.80, 1.20)

        rpm_action = np.zeros((self.num_agents, 4), dtype=np.float32)

        for i in range(self.num_agents):
            s = self.env._getDroneStateVector(i)

            cur_pos = s[0:3]
            cur_quat = s[3:7]
            cur_vel = s[10:13]
            cur_ang_vel = s[13:16]

            rpm, _, _ = self.controllers[i].computeControl(
                control_timestep=1.0 / self.ctrl_freq,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=self.current_targets[i],
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3),
            )

            rpm_action[i] = rpm

        sim_obs, _, terminated, truncated, _ = self.env.step(rpm_action)

        reward, done, info = self._compute_reward_done_info(sim_obs, action)
        obs = self._build_obs(sim_obs)

        self.prev_action = action.copy()

        terminated = bool(done or terminated)
        truncated = bool(truncated or self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, info

    def _get_base_center_target(self):
        if self.phase == 0:
            return self.center_A_takeoff.copy()

        if self.phase == 1:
            progress = min(1.0, self.phase_counter / 260.0)
            return (
                (1.0 - progress) * self.center_A_takeoff
                + progress * self.center_B_hover
            ).astype(np.float32)

        progress = min(1.0, self.phase_counter / 160.0)
        return (
            (1.0 - progress) * self.center_B_hover
            + progress * self.center_B_land
        ).astype(np.float32)

    def _get_base_targets(self):
        center_target = self._get_base_center_target()
        return (center_target[None, :] + self.formation_offsets).astype(np.float32)

    def _build_obs(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)
        base_targets = self._get_base_targets()
        center_target = self._get_base_center_target()

        positions = sim_obs[:, 0:3]
        centroid = np.mean(positions, axis=0)
        formation_error, _ = self._compute_formation_error(positions)

        obs = []

        for i in range(self.num_agents):
            s = sim_obs[i]

            pos = s[0:3]
            rpy = s[7:10]
            vel = s[10:13]

            rel_own_target = base_targets[i] - pos
            rel_centroid_target = center_target - centroid
            formation_err_norm = np.array([formation_error], dtype=np.float32)
            phase_norm = np.array([self.phase / 2.0], dtype=np.float32)

            obs.extend(pos.tolist())
            obs.extend(vel.tolist())
            obs.extend(rpy.tolist())
            obs.extend(rel_own_target.tolist())
            obs.extend(rel_centroid_target.tolist())
            obs.extend(formation_err_norm.tolist())
            obs.extend(phase_norm.tolist())

        return np.array(obs, dtype=np.float32)

    def _compute_formation_error(self, positions):
        """
        Computes how far the current UAV positions are from the desired straight-line
        geometry after aligning the desired formation to the current centroid.

        This makes the metric independent of where the formation is in the arena.
        """
        centroid = np.mean(positions, axis=0)
        desired_positions = centroid[None, :] + self.formation_offsets
        per_agent_error = np.linalg.norm(positions - desired_positions, axis=1)
        mean_error = float(np.mean(per_agent_error))
        return mean_error, per_agent_error

    def _compute_reward_done_info(self, sim_obs, action):
        sim_obs = self._to_numpy_obs(sim_obs)

        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]
        rpys = sim_obs[:, 7:10]

        base_targets = self._get_base_targets()
        center_target = self._get_base_center_target()
        centroid = np.mean(positions, axis=0)

        formation_error, per_agent_formation_error = self._compute_formation_error(positions)
        centroid_error = float(np.linalg.norm(center_target - centroid))
        mean_target_dist = float(np.mean(np.linalg.norm(base_targets - positions, axis=1)))
        max_target_dist = float(np.max(np.linalg.norm(base_targets - positions, axis=1)))
        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))
        max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        mean_attitude_error = float(np.mean(np.linalg.norm(rpys[:, 0:2], axis=1)))
        smoothness_cost = float(np.mean(np.linalg.norm(action - self.prev_action, axis=1)))

        # Shared team reward: all agents receive the same mission-level feedback.
        total_reward = 0.0
        total_reward += self.alive_reward * self.num_agents
        total_reward += -self.centroid_weight * centroid_error
        total_reward += -self.distance_weight * mean_target_dist
        total_reward += -self.formation_weight * formation_error
        total_reward += -self.velocity_weight * mean_speed
        total_reward += -self.attitude_weight * mean_attitude_error
        total_reward += -self.smoothness_weight * smoothness_cost

        # Positive shaping when the team is stable and geometrically aligned.
        if formation_error < 0.25:
            total_reward += 8.0
        if formation_error < 0.15:
            total_reward += 15.0
        if centroid_error < 0.25:
            total_reward += 8.0
        if mean_target_dist < 0.25 and formation_error < 0.20:
            total_reward += 20.0

        phase_changes = 0

        # Phase 0 -> Phase 1: formation has taken off and stabilized at A.
        if self.phase == 0:
            if (
                centroid_error < 0.25
                and formation_error < 0.22
                and mean_speed < 0.65
                and self.phase_counter > 50
            ):
                self.phase = 1
                self.phase_counter = 0
                total_reward += self.phase_bonus
                phase_changes += 1

        # Phase 1 -> Phase 2: formation reached B while preserving line shape.
        elif self.phase == 1:
            if (
                centroid_error < 0.30
                and formation_error < 0.25
                and mean_speed < 0.80
                and self.phase_counter > 250
            ):
                self.phase = 2
                self.phase_counter = 0
                total_reward += self.phase_bonus
                phase_changes += 1

        # Collision check.
        collision_count = 0
        min_pair_dist = np.inf

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                min_pair_dist = min(min_pair_dist, d)

                if d < self.min_separation:
                    collision_count += 1

        if collision_count > 0:
            total_reward -= self.collision_penalty * collision_count

        if min_pair_dist == np.inf:
            min_pair_dist = 0.0

        crashed = bool(
            collision_count > 0
            or np.any(np.abs(rpys[:, 0]) > 1.30)
            or np.any(np.abs(rpys[:, 1]) > 1.30)
            or np.any(positions[:, 2] < 0.035)
            or np.any(positions[:, 2] > 1.60)
        )

        landed = bool(
            self.phase == 2
            and abs(centroid[0] - self.center_B_land[0]) < 0.25
            and abs(centroid[1] - self.center_B_land[1]) < 0.20
            and centroid[2] < 0.22
            and formation_error < self.max_formation_error_for_success
            and mean_speed < 0.55
            and mean_attitude_error < 0.70
            and not crashed
        )

        if landed:
            total_reward += self.goal_bonus

        is_success = landed

        done = bool(
            is_success
            or crashed
            or self.step_count >= self.max_steps
        )

        phase_label = self._phase_name(self.phase)

        info = {
            "is_success": is_success,
            "completed_agents": int(self.num_agents if landed else 0),
            "phase_changes": int(phase_changes),
            "collision_count": int(collision_count),
            "min_pair_dist": float(min_pair_dist),
            "mean_dist_to_target": float(mean_target_dist),
            "max_dist_to_target": float(max_target_dist),
            "centroid_error": float(centroid_error),
            "formation_error": float(formation_error),
            "max_agent_formation_error": float(np.max(per_agent_formation_error)),
            "mean_speed": float(mean_speed),
            "max_speed": float(max_speed),
            "mean_attitude_error": float(mean_attitude_error),
            "smoothness_cost": float(smoothness_cost),
            "mean_phase": float(self.phase),
            "phase": int(self.phase),
            "phase_label": phase_label,
            "phases": [int(self.phase)] * self.num_agents,
            "phase_labels": [phase_label] * self.num_agents,
            "crashed": crashed,
            "landed": landed,
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "target_center_x": float(center_target[0]),
            "target_center_y": float(center_target[1]),
            "target_center_z": float(center_target[2]),
            "uav0_phase": phase_label,
            "uav1_phase": phase_label,
            "uav2_phase": phase_label,
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

        return total_reward, done, info

    def _phase_name(self, phase):
        if phase == 0:
            return "TAKEOFF_FORMATION_A"
        if phase == 1:
            return "MOVE_FORMATION_TO_B"
        return "LAND_FORMATION_B"

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