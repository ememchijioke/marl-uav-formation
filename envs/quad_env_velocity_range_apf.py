import os
import sys
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env_velocity_obstacle import MultiUAVVelocityObstacleFormationEnv


class MultiUAVVelocityRangeAPFEnv(MultiUAVVelocityObstacleFormationEnv):
    """
    v0.11.2: range-based MARL with weak emergency APF only and softer center-obstacle bridge layouts.

    Difference from v0.9.2:
        - The policy is NOT given global obstacle coordinates.
        - The simulator uses obstacle geometry only to produce local range readings
          and safety metrics.
        - The policy receives local range-like observations and APF features.
        - No formation-level APF navigation is added to the reference command.
        - No tangential bypass, bypass-side memory, or recovery command is used
          as active navigation.
        - The policy receives local range readings and must learn obstacle
          avoidance through MARL rewards.
        - APF is retained only as a weak emergency safety filter when a UAV is
          extremely close to an obstacle.

    Actor observation per UAV = 45:
        Original v0.8.5 formation observation = 36
        + range distances = 5
        + minimum range = 1
        + range danger flag = 1
        + APF repulsive velocity xy = 2

    Range directions:
        front, front-left, front-right, left, right

    Final command idea:
        reference formation velocity + learned MARL correction + weak emergency APF only
    """

    def __init__(
        self,
        *args,
        range_max=2.50,
        range_safe=0.65,
        range_emergency=0.25,
        range_apf_gain=0.32,
        formation_apf_gain=0.42,
        max_apf_speed=0.28,
        tangential_apf_gain=0.0,
        formation_tangential_gain=0.0,
        tangential_clearance_bias=0.0,
        bypass_memory_enabled=True,
        bypass_memory_trigger=1.15,
        bypass_memory_release=1.85,
        bypass_hold_steps=180,
        bypass_preferred_side=1.0,
        bypass_lateral_gain=0.18,
        recovery_enabled=True,
        recovery_steps=220,
        recovery_forward_boost=0.10,
        recovery_lateral_gain=0.55,
        recovery_apf_scale=0.35,
        recovery_clear_range=1.60,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.range_max = float(range_max)
        self.range_safe = float(range_safe)
        self.range_emergency = float(range_emergency)
        self.range_apf_gain = float(range_apf_gain)
        self.formation_apf_gain = float(formation_apf_gain)
        self.max_apf_speed = float(max_apf_speed)
        self.tangential_apf_gain = float(tangential_apf_gain)
        self.formation_tangential_gain = float(formation_tangential_gain)
        self.tangential_clearance_bias = float(tangential_clearance_bias)

        # Option D: bypass-side memory.
        # The formation chooses one lateral bypass direction when an obstacle
        # appears in the forward sector, then holds that direction until the
        # obstacle is cleared or the hold counter expires.
        self.bypass_memory_enabled = bool(bypass_memory_enabled)
        self.bypass_memory_trigger = float(bypass_memory_trigger)
        self.bypass_memory_release = float(bypass_memory_release)
        self.bypass_hold_steps = int(bypass_hold_steps)
        self.bypass_preferred_side = 1.0 if float(bypass_preferred_side) >= 0.0 else -1.0
        self.bypass_lateral_gain = float(bypass_lateral_gain)

        # Option E: post-obstacle recovery. Once the bypass memory releases,
        # the formation receives a temporary goal-recovery command. This helps
        # the centroid return to the goal corridor and continue forward instead
        # of hovering near the obstacle after a successful side slip.
        self.recovery_enabled = bool(recovery_enabled)
        self.recovery_steps = int(recovery_steps)
        self.recovery_forward_boost = float(recovery_forward_boost)
        self.recovery_lateral_gain = float(recovery_lateral_gain)
        self.recovery_apf_scale = float(recovery_apf_scale)
        self.recovery_clear_range = float(recovery_clear_range)

        self._bypass_active = False
        self._bypass_side_sign = self.bypass_preferred_side
        self._bypass_hold_counter = 0
        self._recovery_active = False
        self._recovery_counter = 0
        self._last_bypass_side_sign = self.bypass_preferred_side

        self.range_ray_angles = np.array(
            [
                0.0,                 # front
                np.deg2rad(45.0),    # front-left
                np.deg2rad(-45.0),   # front-right
                np.deg2rad(90.0),    # left
                np.deg2rad(-90.0),   # right
            ],
            dtype=np.float32,
        )

        self.obs_dim_per_agent = 45

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim_per_agent * self.num_agents,),
            dtype=np.float32,
        )

    def _apply_v010_custom_obstacle_layout(self):
        """
        v0.10.0 custom obstacle layouts for range/APF curriculum.

        The policy does NOT receive these coordinates. The coordinates are used
        only inside the simulator to generate local range readings and metrics.
        """
        if not self.use_obstacles:
            return

        # Softer bridge sequence for v0.11.2.
        # These coordinates are internal simulator geometry only. The policy does
        # not receive global obstacle coordinates; it only receives local ranges.
        bridge_layouts = {
            "soft_center":      (0.95, 0.24),
            "soft_center_plus": (0.82, 0.24),
            # v0.11.3: much smaller center-shift steps after soft_center_plus.
            # Previous jump from y=0.82 to y=0.70 caused repeated crashes.
            "center_bridge_0":  (0.78, 0.24),
            "center_bridge_1":  (0.74, 0.24),
            "center_bridge_2":  (0.70, 0.24),
            "center_bridge_3":  (0.62, 0.24),
            "center_bridge_4":  (0.54, 0.24),
            "center_bridge_5":  (0.45, 0.25),
            "center_bridge_6":  (0.35, 0.25),
            "center_bridge_7":  (0.25, 0.25),
            "center_bridge_8":  (0.15, 0.25),
        }

        if self.obstacle_layout in bridge_layouts:
            y_offset, radius = bridge_layouts[self.obstacle_layout]
            self.obstacles = [
                {
                    "center": np.array([3.50, y_offset, 0.75], dtype=np.float32),
                    "radius": radius,
                    "height": 1.50,
                }
            ]

        elif self.obstacle_layout == "soft_center_left":
            self.obstacles = [
                {
                    "center": np.array([3.50, -0.95, 0.75], dtype=np.float32),
                    "radius": 0.24,
                    "height": 1.50,
                }
            ]

    def _get_current_sim_obs(self):
        """Return current PyBullet drone state vectors as a numpy array."""
        sim_obs = []
        for i in range(self.num_agents):
            sim_obs.append(self.env._getDroneStateVector(i))
        return np.asarray(sim_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super_obs, info = super().reset(seed=seed, options=options)

        self._bypass_active = False
        self._bypass_side_sign = self.bypass_preferred_side
        self._bypass_hold_counter = 0
        self._recovery_active = False
        self._recovery_counter = 0
        self._last_bypass_side_sign = self.bypass_preferred_side

        self._apply_v010_custom_obstacle_layout()

        # Rebuild the first observation after applying custom obstacle layout.
        # This avoids returning one stale first observation from the parent reset.
        try:
            sim_obs = self._get_current_sim_obs()
            reference_velocity, formation_tracking_error = self._compute_reference_velocity(sim_obs)
            obs = self._build_obs(sim_obs, reference_velocity, formation_tracking_error)
            return obs, info
        except Exception:
            return super_obs, info

    def _ray_circle_intersection_distance(self, ray_origin_xy, ray_dir_xy, circle_center_xy, radius):
        oc = ray_origin_xy - circle_center_xy

        a = float(np.dot(ray_dir_xy, ray_dir_xy))
        b = 2.0 * float(np.dot(oc, ray_dir_xy))
        c = float(np.dot(oc, oc)) - float(radius) * float(radius)

        discriminant = b * b - 4.0 * a * c

        if discriminant < 0.0:
            return self.range_max

        sqrt_disc = np.sqrt(discriminant)

        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        candidates = [t for t in [t1, t2] if t >= 0.0]

        if len(candidates) == 0:
            return self.range_max

        return float(min(candidates))

    def _range_distances_at_point(self, position, yaw=0.0):
        if not self.use_obstacles or len(self.obstacles) == 0:
            return np.ones(len(self.range_ray_angles), dtype=np.float32) * self.range_max

        pos_xy = np.asarray(position[0:2], dtype=np.float32)
        z = float(position[2])

        distances = []

        for angle in self.range_ray_angles:
            theta = float(yaw + angle)

            ray_dir = np.array(
                [np.cos(theta), np.sin(theta)],
                dtype=np.float32,
            )

            best_dist = self.range_max

            for obs in self.obstacles:
                center = np.asarray(obs["center"], dtype=np.float32)
                radius = float(obs["radius"])
                height = float(obs["height"])

                inside_height = bool(0.02 <= z <= height + 0.10)

                if not inside_height:
                    continue

                dist = self._ray_circle_intersection_distance(
                    ray_origin_xy=pos_xy,
                    ray_dir_xy=ray_dir,
                    circle_center_xy=center[0:2],
                    radius=radius,
                )

                best_dist = min(best_dist, dist)

            distances.append(np.clip(best_dist, 0.0, self.range_max))

        return np.asarray(distances, dtype=np.float32)

    def _repulsive_apf_from_ranges(self, ranges, yaw=0.0, gain=None):
        """
        Standard APF repulsion from range readings.

        This component pushes directly away from nearby obstacles. It is useful
        for emergency safety, but by itself it can create a local minimum when
        the obstacle is directly between the formation and the goal.
        """
        if gain is None:
            gain = self.range_apf_gain

        apf_xy = np.zeros(2, dtype=np.float32)

        for dist, angle in zip(ranges, self.range_ray_angles):
            dist = float(dist)

            if dist >= self.range_safe:
                continue

            theta = float(yaw + angle)

            ray_dir = np.array(
                [np.cos(theta), np.sin(theta)],
                dtype=np.float32,
            )

            repulsion_dir = -ray_dir

            strength = (self.range_safe - dist) / max(self.range_safe, 1e-6)
            strength = float(np.clip(strength, 0.0, 2.0))

            apf_xy += float(gain) * strength * repulsion_dir

        return apf_xy.astype(np.float32)

    def _choose_bypass_side_from_ranges(self, ranges):
        """
        Return lateral side sign from local ranges only.

        Convention for yaw=0:
            +1.0 = left / positive y
            -1.0 = right / negative y
        """
        ranges = np.asarray(ranges, dtype=np.float32)

        # Range order: front, front-left, front-right, left, right.
        left_clearance = float(max(ranges[1], ranges[3]))
        right_clearance = float(max(ranges[2], ranges[4]))
        clearance_delta = right_clearance - left_clearance

        if abs(clearance_delta) < 0.05:
            return self.bypass_preferred_side

        if clearance_delta > 0.0:
            return -1.0
        return 1.0

    def _update_bypass_memory(self, ranges):
        """
        Option D/E memory logic for centroid-level bypass.

        The memory is only based on range readings; it does not use global
        obstacle coordinates. It prevents APF dithering by holding the selected
        lateral side for a short period.

        Option E extension:
            when the bypass is released because the forward sector is clear,
            activate a temporary recovery mode. Recovery later pulls the
            centroid back toward the goal corridor and boosts forward motion.
        """
        if not self.bypass_memory_enabled:
            self._bypass_active = False
            self._bypass_hold_counter = 0
            return 0.0

        ranges = np.asarray(ranges, dtype=np.float32)

        front_sector = float(min(ranges[0], ranges[1], ranges[2]))
        min_range = float(np.min(ranges))

        if not self._bypass_active:
            if front_sector < self.bypass_memory_trigger:
                self._bypass_active = True
                self._recovery_active = False
                self._recovery_counter = 0
                self._bypass_side_sign = self._choose_bypass_side_from_ranges(ranges)
                self._last_bypass_side_sign = self._bypass_side_sign
                self._bypass_hold_counter = self.bypass_hold_steps
        else:
            self._bypass_hold_counter -= 1

            if front_sector < self.bypass_memory_trigger:
                self._bypass_hold_counter = max(self._bypass_hold_counter, self.bypass_hold_steps // 2)

            clear_enough = (
                (front_sector > self.bypass_memory_release)
                and (min_range > self.range_safe)
            )

            forced_timeout = self._bypass_hold_counter <= 0

            if clear_enough or forced_timeout:
                self._bypass_active = False
                self._bypass_hold_counter = 0

                # Only activate recovery if the formation truly cleared the
                # obstacle region. If timeout happens while still close, do not
                # trigger recovery because that would push it into danger.
                if self.recovery_enabled and clear_enough:
                    self._recovery_active = True
                    self._recovery_counter = self.recovery_steps

        if self._bypass_active:
            return float(self._bypass_side_sign)

        return 0.0

    def _update_recovery_state_from_ranges(self, ranges):
        """
        Maintain post-obstacle recovery using local range status only.

        Recovery is disabled if a close obstacle reappears in the forward
        sector. Otherwise it counts down and allows the centroid to recentre
        and continue toward the goal.
        """
        if not self._recovery_active:
            return False

        ranges = np.asarray(ranges, dtype=np.float32)
        front_sector = float(min(ranges[0], ranges[1], ranges[2]))
        min_range = float(np.min(ranges))

        if front_sector < self.bypass_memory_trigger or min_range < self.range_emergency:
            self._recovery_active = False
            self._recovery_counter = 0
            return False

        self._recovery_counter -= 1
        if self._recovery_counter <= 0:
            self._recovery_active = False
            self._recovery_counter = 0
            return False

        return True

    def _tangential_apf_from_ranges(self, ranges, yaw=0.0, gain=None, forced_side_sign=None):
        """
        Tangential APF bypass from range readings.

        This is the Option C fix. Instead of only pushing backward from an
        obstacle in front, this term creates a sideways velocity component.
        The side is selected from local range clearance:
            - if the right side is more open, move right
            - if the left side is more open, move left

        The policy still receives no global obstacle coordinate. The bypass is
        computed only from local range readings.
        """
        if gain is None:
            gain = self.tangential_apf_gain

        gain = float(gain)
        if gain <= 0.0:
            return np.zeros(2, dtype=np.float32)

        ranges = np.asarray(ranges, dtype=np.float32)
        min_range = float(np.min(ranges))

        if min_range >= self.range_safe:
            return np.zeros(2, dtype=np.float32)

        # Range order: front, front-left, front-right, left, right.
        left_clearance = float(max(ranges[1], ranges[3]))
        right_clearance = float(max(ranges[2], ranges[4]))

        if forced_side_sign is not None and abs(float(forced_side_sign)) > 0.5:
            side_sign = 1.0 if float(forced_side_sign) > 0.0 else -1.0
        else:
            # Choose the more open side. If both sides are similar, use a small
            # deterministic bias so the formation does not hesitate.
            clearance_delta = right_clearance - left_clearance

            if abs(clearance_delta) < 0.05:
                side_sign = self.bypass_preferred_side
            elif clearance_delta > 0.0:
                # right side in the world frame is negative y for yaw=0.
                side_sign = -1.0
            else:
                side_sign = 1.0

        lateral_dir = np.array(
            [-np.sin(float(yaw)), np.cos(float(yaw))],
            dtype=np.float32,
        )
        tangent_dir = side_sign * lateral_dir

        # Tangential strength becomes stronger when something is close in the
        # forward sector. This avoids sideways drift when only a side ray is
        # mildly active.
        forward_pressure = max(
            0.0,
            self.range_safe - float(min(ranges[0], ranges[1], ranges[2])),
        ) / max(self.range_safe, 1e-6)

        nearest_pressure = max(0.0, self.range_safe - min_range) / max(self.range_safe, 1e-6)

        strength = 0.70 * forward_pressure + 0.30 * nearest_pressure
        strength = float(np.clip(strength, 0.0, 2.0))

        return (gain * strength * tangent_dir).astype(np.float32)

    def _apf_from_ranges(self, ranges, yaw=0.0, gain=None, tangential_gain=None, forced_side_sign=None):
        """
        Combined APF = repulsive APF + tangential bypass APF.

        This keeps the original range-based safety behavior, but adds a sideways
        bypass term to escape the local-minimum problem around center obstacles.
        """
        if gain is None:
            gain = self.range_apf_gain

        if tangential_gain is None:
            tangential_gain = self.tangential_apf_gain

        repulsive_xy = self._repulsive_apf_from_ranges(ranges, yaw=yaw, gain=gain)
        tangent_xy = self._tangential_apf_from_ranges(
            ranges,
            yaw=yaw,
            gain=tangential_gain,
            forced_side_sign=forced_side_sign,
        )

        apf_xy = repulsive_xy + tangent_xy

        speed = float(np.linalg.norm(apf_xy))

        if speed > self.max_apf_speed and self.max_apf_speed > 0.0:
            apf_xy = apf_xy / speed * self.max_apf_speed

        return apf_xy.astype(np.float32)

    def _range_features_for_agent(self, position, yaw=0.0):
        ranges = self._range_distances_at_point(position, yaw=yaw)
        min_range = float(np.min(ranges))

        apf_xy = self._apf_from_ranges(
            ranges,
            yaw=yaw,
            gain=self.range_apf_gain,
            tangential_gain=self.tangential_apf_gain,
        )

        ranges_norm = np.clip(ranges / self.range_max, 0.0, 1.0).astype(np.float32)

        min_range_norm = np.array(
            [np.clip(min_range / self.range_max, 0.0, 1.0)],
            dtype=np.float32,
        )

        danger = np.array(
            [1.0 if min_range < self.range_safe else 0.0],
            dtype=np.float32,
        )

        if self.max_apf_speed > 0.0:
            apf_norm = np.clip(
                apf_xy / max(self.max_apf_speed, 1e-6),
                -1.0,
                1.0,
            ).astype(np.float32)
        else:
            apf_norm = np.zeros(2, dtype=np.float32)

        return ranges_norm, min_range_norm, danger, apf_norm

    def _formation_centroid_range_apf(self, centroid, yaw=0.0):
        ranges = self._range_distances_at_point(centroid, yaw=yaw)
        forced_side = self._update_bypass_memory(ranges)

        recovery_now = self._update_recovery_state_from_ranges(ranges)

        formation_apf_gain = self.formation_apf_gain
        formation_tangential_gain = self.formation_tangential_gain

        # In recovery mode, do not keep pushing the formation sideways as hard.
        # The obstacle has been cleared, so APF should decay and allow goal
        # tracking/re-centering to dominate.
        if recovery_now:
            formation_apf_gain *= self.recovery_apf_scale
            formation_tangential_gain *= self.recovery_apf_scale

        apf_xy = self._apf_from_ranges(
            ranges,
            yaw=yaw,
            gain=formation_apf_gain,
            tangential_gain=formation_tangential_gain,
            forced_side_sign=forced_side,
        )

        # Extra memory-guided lateral push. This term is deliberately applied
        # only to the formation centroid, not individual UAVs, so the group
        # moves around the obstacle as one body.
        if (not self._recovery_active) and abs(forced_side) > 0.5 and self.bypass_lateral_gain > 0.0:
            lateral_dir = np.array(
                [-np.sin(float(yaw)), np.cos(float(yaw))],
                dtype=np.float32,
            )
            apf_xy += float(forced_side) * self.bypass_lateral_gain * lateral_dir

        speed = float(np.linalg.norm(apf_xy))
        if speed > self.max_apf_speed and self.max_apf_speed > 0.0:
            apf_xy = apf_xy / speed * self.max_apf_speed

        apf = np.zeros(3, dtype=np.float32)
        apf[0:2] = apf_xy
        return apf

    def _range_danger_active(self, positions, yaws):
        for i in range(self.num_agents):
            ranges = self._range_distances_at_point(positions[i], yaw=float(yaws[i]))
            if float(np.min(ranges)) < self.range_safe:
                return True
        return False

    def _apply_emergency_safety_filter(self, sim_obs, final_velocity):
        positions = sim_obs[:, 0:3]
        rpys = sim_obs[:, 7:10]

        safe_velocity = final_velocity.copy()

        for i in range(self.num_agents):
            pos = positions[i]
            yaw = float(rpys[i, 2])

            ranges = self._range_distances_at_point(pos, yaw=yaw)

            if float(np.min(ranges)) < self.range_emergency:
                apf_xy = self._apf_from_ranges(
                    ranges,
                    yaw=yaw,
                    gain=self.safety_filter_gain,
                    tangential_gain=0.0,
                )

                safe_velocity[i, 0:2] += apf_xy

        return safe_velocity.astype(np.float32)

    def _compute_reference_velocity(self, sim_obs):
        """
        v0.11.0 reference velocity.

        Important difference from v0.10.x:
            - The reference controller only handles goal/formation tracking.
            - It does NOT add formation-level APF, tangential bypass, bypass
              memory, or recovery navigation.
            - Obstacle avoidance must be learned by the MARL policy from local
              range observations.
            - Only the emergency safety filter may add a small APF correction
              after the policy command, and only when the UAV is extremely close.
        """
        positions = sim_obs[:, 0:3]
        rpys = sim_obs[:, 7:10]
        yaws = rpys[:, 2]

        target_center, formation_offsets = self._phase_target_and_offsets()

        centroid = np.mean(positions, axis=0)
        center_error = target_center - centroid

        center_velocity_cmd = self.kp_center * center_error
        center_velocity_cmd[2] = self.kp_alt * center_error[2]

        danger = self._range_danger_active(positions, yaws)

        effective_kp_form = self.kp_form
        if danger:
            effective_kp_form = self.kp_form * (1.0 + self.formation_lock_gain)

        desired_positions = target_center[None, :] + formation_offsets
        formation_tracking_error = desired_positions - positions

        reference_velocity = np.zeros((self.num_agents, 3), dtype=np.float32)

        for i in range(self.num_agents):
            form_cmd = effective_kp_form * formation_tracking_error[i]
            reference_velocity[i] = center_velocity_cmd + form_cmd

        if self.phase == 1:
            reference_velocity[:, 0] *= 0.30

        elif self.phase == 3:
            reference_velocity[:, 0] *= 0.20

        if self.phase == 4:
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
        rpys = sim_obs[:, 7:10]

        target_center, formation_offsets = self._phase_target_and_offsets()
        desired_positions = target_center[None, :] + formation_offsets

        spacing_error = np.array(
            [self._compute_spacing_error(positions, formation_offsets)],
            dtype=np.float32,
        )

        phase_onehot = np.zeros(5, dtype=np.float32)
        phase_onehot[self.phase] = 1.0

        obs_parts = []

        for i in range(self.num_agents):
            own_pos = positions[i]
            own_vel = velocities[i]
            yaw = float(rpys[i, 2])

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

            ranges_norm, min_range_norm, danger, apf_norm = self._range_features_for_agent(
                own_pos,
                yaw=yaw,
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
                    ranges_norm.astype(np.float32),
                    min_range_norm.astype(np.float32),
                    danger.astype(np.float32),
                    apf_norm.astype(np.float32),
                ],
                axis=0,
            ).astype(np.float32)

            if obs_i.shape[0] != self.obs_dim_per_agent:
                raise RuntimeError(
                    f"Bad obs dim for agent {i}: got {obs_i.shape[0]}, expected {self.obs_dim_per_agent}"
                )

            obs_parts.append(obs_i)

        return np.concatenate(obs_parts, axis=0).astype(np.float32)