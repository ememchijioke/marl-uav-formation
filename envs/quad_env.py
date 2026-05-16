import numpy as np
import gymnasium as gym
from gymnasium import spaces

# gym-pybullet-drones imports
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiUAVRealisticEnv(gym.Env):
    """
    Multi-UAV waypoint-and-landing environment built on gym-pybullet-drones.

    Mission:
        3 UAVs fly independently in parallel lanes.

        Phase 0: take off / hover at A
        Phase 1: move from A to B
        Phase 2: land at B

    Important:
        This version does NOT enforce formation control.
        It only tests whether multiple UAVs can execute a simple waypoint mission
        without colliding.

    High-level action:
        Per-agent xyz residual correction around a safe target.

    Low-level control:
        DSLPIDControl converts target positions into motor RPMs.

    Observation per UAV:
        pos(3), vel(3), rpy(3), rel_target(3), phase(1) = 13

    Joint observation:
        Concatenation of all per-agent observations.
    """

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        render_mode="headless",
        num_agents=3,
        max_steps=600,
        ctrl_freq=48,
        sim_freq=240,
        action_scales=(0.04, 0.04, 0.05),
        goal_bonus=500.0,
        phase_bonus=80.0,
        distance_weight=3.0,
        velocity_weight=1.0,
        attitude_weight=1.0,
        alive_reward=0.15,
        collision_penalty=250.0,
        min_separation=0.30,
    ):
        super().__init__()

        if num_agents != 3:
            raise ValueError("This simplified mission environment expects exactly 3 UAVs.")

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.ctrl_freq = ctrl_freq
        self.sim_freq = sim_freq

        self.action_scales = np.array(action_scales, dtype=np.float32)

        self.goal_bonus = goal_bonus
        self.phase_bonus = phase_bonus
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.attitude_weight = attitude_weight
        self.alive_reward = alive_reward
        self.collision_penalty = collision_penalty
        self.min_separation = min_separation

        # -----------------------------
        # Mission layout: parallel lanes
        # -----------------------------
        self.lane_y = np.array([-0.60, 0.00, 0.60], dtype=np.float32)

        self.start_positions = np.array(
            [
                [0.0, -0.60, 0.20],
                [0.0,  0.00, 0.20],
                [0.0,  0.60, 0.20],
            ],
            dtype=np.float32,
        )

        self.hover_A_targets = np.array(
            [
                [0.0, -0.60, 1.00],
                [0.0,  0.00, 1.00],
                [0.0,  0.60, 1.00],
            ],
            dtype=np.float32,
        )

        self.hover_B_targets = np.array(
            [
                [2.0, -0.60, 1.00],
                [2.0,  0.00, 1.00],
                [2.0,  0.60, 1.00],
            ],
            dtype=np.float32,
        )

        self.land_B_targets = np.array(
            [
                [2.0, -0.60, 0.10],
                [2.0,  0.00, 0.10],
                [2.0,  0.60, 0.10],
            ],
            dtype=np.float32,
        )

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
        # pos(3), vel(3), rpy(3), rel_target(3), phase(1) = 13
        self.obs_dim_per_agent = 13

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim_per_agent * self.num_agents,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.phases = np.zeros(self.num_agents, dtype=np.int32)
        self.phase_counters = np.zeros(self.num_agents, dtype=np.int32)
        self.current_targets = self.hover_A_targets.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.phases = np.zeros(self.num_agents, dtype=np.int32)
        self.phase_counters = np.zeros(self.num_agents, dtype=np.int32)
        self.current_targets = self.hover_A_targets.copy()

        sim_obs, _ = self.env.reset(seed=seed, options=options)

        obs = self._build_obs(sim_obs)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        self.phase_counters += 1

        action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, 3)
        action = np.clip(action, -1.0, 1.0)

        base_targets = self._get_base_targets()

        residual = action * self.action_scales[None, :]
        self.current_targets = base_targets + residual

        # Keep each target inside its own safe lane corridor.
        for i in range(self.num_agents):
            self.current_targets[i, 0] = np.clip(self.current_targets[i, 0], -0.2, 2.2)
            self.current_targets[i, 1] = np.clip(
                self.current_targets[i, 1],
                self.lane_y[i] - 0.15,
                self.lane_y[i] + 0.15,
            )

            if self.phases[i] == 2:
                self.current_targets[i, 2] = np.clip(self.current_targets[i, 2], 0.08, 1.10)
            else:
                self.current_targets[i, 2] = np.clip(self.current_targets[i, 2], 0.80, 1.15)

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

        reward, done, info = self._compute_reward_done_info(sim_obs)

        obs = self._build_obs(sim_obs)

        terminated = bool(done or terminated)
        truncated = bool(truncated or self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, info

    def _get_base_targets(self):
        base_targets = np.zeros((self.num_agents, 3), dtype=np.float32)

        for i in range(self.num_agents):
            if self.phases[i] == 0:
                base_targets[i] = self.hover_A_targets[i]

            elif self.phases[i] == 1:
                progress = min(1.0, self.phase_counters[i] / 220.0)
                base_targets[i] = (
                    (1.0 - progress) * self.hover_A_targets[i]
                    + progress * self.hover_B_targets[i]
                )

            else:
                progress = min(1.0, self.phase_counters[i] / 120.0)
                base_targets[i] = (
                    (1.0 - progress) * self.hover_B_targets[i]
                    + progress * self.land_B_targets[i]
                )

        return base_targets

    def _build_obs(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)
        base_targets = self._get_base_targets()

        obs = []

        for i in range(self.num_agents):
            s = sim_obs[i]

            pos = s[0:3]
            rpy = s[7:10]
            vel = s[10:13]

            rel_target = base_targets[i] - pos
            phase_norm = np.array([self.phases[i] / 2.0], dtype=np.float32)

            obs.extend(pos.tolist())
            obs.extend(vel.tolist())
            obs.extend(rpy.tolist())
            obs.extend(rel_target.tolist())
            obs.extend(phase_norm.tolist())

        return np.array(obs, dtype=np.float32)

    def _compute_reward_done_info(self, sim_obs):
        sim_obs = self._to_numpy_obs(sim_obs)

        positions = sim_obs[:, 0:3]
        velocities = sim_obs[:, 10:13]
        rpys = sim_obs[:, 7:10]

        base_targets = self._get_base_targets()

        total_reward = 0.0
        completed_agents = 0
        phase_changes = 0

        dists = []
        speeds = []
        attitudes = []
        xy_errors = []

        for i in range(self.num_agents):
            pos = positions[i]
            vel = velocities[i]
            rpy = rpys[i]
            target = base_targets[i]

            dist = float(np.linalg.norm(target - pos))
            speed = float(np.linalg.norm(vel))
            attitude_error = float(np.linalg.norm(rpy[0:2]))
            xy_error = float(np.linalg.norm(target[0:2] - pos[0:2]))

            reward_i = 0.0
            reward_i += self.alive_reward
            reward_i += -self.distance_weight * dist
            reward_i += -self.velocity_weight * speed
            reward_i += -self.attitude_weight * attitude_error

            # Reward stable closeness to current target.
            if dist < 0.25:
                reward_i += 5.0

            if dist < 0.18 and speed < 0.50:
                reward_i += 10.0

            # Phase 0 -> Phase 1: finished takeoff/hover A.
            if self.phases[i] == 0:
                if dist < 0.25 and speed < 0.55 and self.phase_counters[i] > 40:
                    self.phases[i] = 1
                    self.phase_counters[i] = 0
                    reward_i += self.phase_bonus
                    phase_changes += 1

            # Phase 1 -> Phase 2: arrived near B hover.
            elif self.phases[i] == 1:
                if dist < 0.30 and speed < 0.70 and self.phase_counters[i] > 220:
                    self.phases[i] = 2
                    self.phase_counters[i] = 0
                    reward_i += self.phase_bonus
                    phase_changes += 1

            # Phase 2: landing at B.
            else:
                landed = (
                    abs(pos[0] - self.land_B_targets[i, 0]) < 0.25
                    and abs(pos[1] - self.land_B_targets[i, 1]) < 0.25
                    and pos[2] < 0.20
                    and speed < 0.55
                    and attitude_error < 0.70
                )

                if landed:
                    completed_agents += 1
                    reward_i += self.goal_bonus

            total_reward += reward_i

            dists.append(dist)
            speeds.append(speed)
            attitudes.append(attitude_error)
            xy_errors.append(xy_error)

        # Collision check
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
        )

        is_success = completed_agents == self.num_agents

        done = bool(
            is_success
            or crashed
            or self.step_count >= self.max_steps
        )

        phase_labels = [self._phase_name(p) for p in self.phases]

        info = {
            "is_success": is_success,
            "completed_agents": int(completed_agents),
            "phase_changes": int(phase_changes),
            "collision_count": int(collision_count),
            "min_pair_dist": float(min_pair_dist),
            "mean_dist_to_target": float(np.mean(dists)),
            "max_dist_to_target": float(np.max(dists)),
            "mean_xy_error": float(np.mean(xy_errors)),
            "mean_speed": float(np.mean(speeds)),
            "mean_attitude_error": float(np.mean(attitudes)),
            "mean_phase": float(np.mean(self.phases)),
            "phases": self.phases.astype(int).tolist(),
            "phase_labels": phase_labels,
            "crashed": crashed,
            "uav0_phase": phase_labels[0],
            "uav1_phase": phase_labels[1],
            "uav2_phase": phase_labels[2],
            "uav0_x": float(positions[0, 0]),
            "uav1_x": float(positions[1, 0]),
            "uav2_x": float(positions[2, 0]),
            "uav0_z": float(positions[0, 2]),
            "uav1_z": float(positions[1, 2]),
            "uav2_z": float(positions[2, 2]),
        }

        return total_reward, done, info

    def _phase_name(self, phase):
        if phase == 0:
            return "TAKEOFF_A"
        if phase == 1:
            return "MOVE_TO_B"
        return "LAND_B"

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