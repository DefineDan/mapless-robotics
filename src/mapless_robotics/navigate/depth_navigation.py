"""Depth-only local navigation with a short rolling occupancy window."""

from collections import deque

import numpy as np

from .planners import AStarPlanner

UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
OCCUPIED = np.uint8(255)


class DepthNavigation:
    """Generate planar velocity commands from a goal and recent depth frames.

    Poses are Habitat local-to-world transforms (X right, Y up, camera forward
    along local -Z). Occupancy is rebuilt from only the last ``history_size``
    frames, limiting the effect of noisy pose estimates. Depth returns are
    grouped into 3D voxels; voxels with fewer than ``min_voxel_points`` returns
    are discarded before obstacle and free-space evidence is added.
    """

    def __init__(
        self,
        history_size=5,
        positive_delta=0.4,
        negative_delta=1.2,
        occupancy_resolution=0.1,
        map_padding=0.4,
        unknown_cost=3.0,
        max_depth=2.0,
        voxel_size=0.15,
        min_voxel_points=10,
        linear_speed=0.25,
        angular_gain=1.5,
        max_angular_speed=1.0,
        goal_tolerance=0.1,
        heading_tolerance=0.1,
        angular_smoothing=1.0,
        vertical_axis="y",
        camera_intrinsics=None,
        planner=None,
        max_map_cells=4_000_000,
        floor_height=None,
        minimum_obstacle_height=None,
        maximum_obstacle_height=None,
        floor_clearance=0.05,
        require_goal_heading=True,
    ):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if vertical_axis not in {"x", "y", "z"}:
            raise ValueError("vertical_axis must be 'x', 'y', or 'z'")
        if not np.isfinite(map_padding) or map_padding < 0:
            raise ValueError("map_padding must be finite and non-negative")
        if not np.isfinite(unknown_cost) or unknown_cost < 1:
            raise ValueError("unknown_cost must be finite and at least 1")
        values = (positive_delta, negative_delta, occupancy_resolution, max_depth,
                  voxel_size,
                  linear_speed, angular_gain, max_angular_speed, goal_tolerance,
                  heading_tolerance, angular_smoothing)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("navigation distances and speeds must be finite and positive")
        if angular_smoothing > 1:
            raise ValueError("angular_smoothing must be in the interval (0, 1]")
        if (not np.isfinite(min_voxel_points) or min_voxel_points < 1
                or int(min_voxel_points) != min_voxel_points):
            raise ValueError("min_voxel_points must be a positive integer")
        if not isinstance(max_map_cells, int) or max_map_cells <= 0:
            raise ValueError("max_map_cells must be a positive integer")
        from ..floor_reference import validate_height_band
        validate_height_band(floor_height, minimum_obstacle_height, maximum_obstacle_height)
        if not np.isfinite(floor_clearance) or floor_clearance < 0:
            raise ValueError('Floor clearance must be finite and nonnegative')
        self.floor_clearance = float(floor_clearance)
        self.height_diagnostics = {}
        self.floor_height = floor_height
        self.minimum_obstacle_height = minimum_obstacle_height
        self.maximum_obstacle_height = maximum_obstacle_height
        self.max_map_cells = max_map_cells
        self.history_size = int(history_size)
        self.positive_delta = float(positive_delta)
        self.negative_delta = float(negative_delta)
        self.resolution = float(occupancy_resolution)
        self.map_padding = float(map_padding)
        self.unknown_cost = float(unknown_cost)
        self.max_depth = float(max_depth)
        self.voxel_size = float(voxel_size)
        self.min_voxel_points = int(min_voxel_points)
        self.linear_speed = float(linear_speed)
        self.angular_gain = float(angular_gain)
        self.max_angular_speed = float(max_angular_speed)
        self.goal_tolerance = float(goal_tolerance)
        self.heading_tolerance = float(heading_tolerance)
        self.require_goal_heading = bool(require_goal_heading)
        self.angular_smoothing = float(angular_smoothing)
        self._last_angular_velocity = 0.0
        self.vertical_index = {"x": 0, "y": 1, "z": 2}[vertical_axis]
        self.plane_indices = [i for i in range(3) if i != self.vertical_index]
        self.camera_intrinsics = camera_intrinsics
        self.planner = planner or AStarPlanner(
            self.resolution,
            self.plane_indices,
            inflation_radius=0.15,
            unknown_cost=self.unknown_cost,
        )
        self.trajectory = np.zeros((0, 2), dtype=float)
        self._trajectory_index = 0
        self._frames = deque(maxlen=self.history_size)
        self._point_frames = deque(maxlen=self.history_size)
        self._free_frames = deque(maxlen=self.history_size)
        self.goal = None
        self.angle_objective = None
        self.camera_pose = None
        self.occupancy = {}
        self.free_space = set()

    @staticmethod
    def _ray_cells(start, end):
        """Return integer grid cells on a 2D ray, excluding its endpoint."""
        x0, y0 = map(int, start)
        x1, y1 = map(int, end)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy
        cells = []
        while (x0, y0) != (x1, y1):
            cells.append((x0, y0))
            twice_error = 2 * error
            if twice_error > -dy:
                error -= dy
                x0 += sx
            if twice_error < dx:
                error += dx
                y0 += sy
        return cells

    def set_data(
        self, goal, depth_image, camera_pose, camera_intrinsics=None,
        angle_objective=None,
    ):
        """Add a depth observation and update the current goal and occupancy.

        ``goal`` is a world-frame XYZ point, ``depth_image`` is metric depth,
        and ``camera_pose`` is a Habitat camera-to-world 4x4 matrix.
        """
        goal = np.asarray(goal, dtype=float).reshape(-1)
        depth = np.asarray(depth_image, dtype=float)
        pose = np.asarray(camera_pose, dtype=float)
        if goal.size not in (2, 3) or not np.isfinite(goal).all():
            raise ValueError("goal must be a finite world-frame XY or XYZ point")
        if goal.size == 2:
            expanded = np.zeros(3, dtype=float)
            expanded[self.plane_indices] = goal
            goal = expanded
        if depth.ndim != 2:
            raise ValueError("depth_image must be a 2D metric depth image")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("camera_pose must be a finite 4x4 local-to-world matrix")
        if not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("camera_pose must have homogeneous last row [0, 0, 0, 1]")

        intrinsics = camera_intrinsics if camera_intrinsics is not None else self.camera_intrinsics
        height, width = depth.shape
        if intrinsics is None:
            focal = width / 2.0  # Habitat's default horizontal FOV is 90 degrees.
            intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        intrinsics = np.asarray(intrinsics, dtype=float)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("camera_intrinsics must be a finite 3x3 matrix")

        # Subsample for bounded computation, then back-project OpenCV-style
        # pixels and convert them into Habitat camera coordinates.
        stride = max(1, int(max(height, width) / 100))
        rows, cols = np.mgrid[0:height:stride, 0:width:stride]
        z = depth[::stride, ::stride]
        # Keep visibility evidence separate from obstacle filtering below:
        # height gating and voxel denoising should not erase rays that the
        # camera actually observed. Returns beyond max_depth still establish
        # free space up to the configured range, but are not obstacle points.
        observed = np.isfinite(z) & (z > 0)
        rows, cols, z = rows[observed], cols[observed], z[observed]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        ray_depth = np.minimum(z, self.max_depth)
        ray_local = np.column_stack(((cols - cx) * ray_depth / fx,
                                     -(rows - cy) * ray_depth / fy,
                                     -ray_depth))
        ray_world = ray_local @ pose[:3, :3].T + pose[:3, 3]
        local = np.column_stack(((cols - cx) * z / fx,
                                 -(rows - cy) * z / fy,
                                 -z))
        world = local @ pose[:3, :3].T + pose[:3, 3]
        if self.floor_height is not None:
            # Returns below the known floor cannot establish valid free space.
            above_floor = world[:, self.vertical_index] >= self.floor_height - self.floor_clearance
            world, ray_world, z = world[above_floor], ray_world[above_floor], z[above_floor]
        # Apply the depth mask while world and z still describe the same samples.
        # Visibility rays retain all observed returns independently of obstacles.
        world = world[z <= self.max_depth]

        if self.floor_height is not None:
            heights = world[:, self.vertical_index]-self.floor_height
            ground = np.abs(heights) <= self.floor_clearance
            minimum = max(self.floor_clearance, self.minimum_obstacle_height
                          if self.minimum_obstacle_height is not None else self.floor_clearance)
            obstacle = heights > minimum
            if self.maximum_obstacle_height is not None:
                obstacle &= heights <= self.maximum_obstacle_height
            elif self.minimum_obstacle_height is None:
                # Retain the previous ceiling cutoff unless the operator
                # supplies an explicit floor-relative obstacle band.
                obstacle &= world[:, self.vertical_index] <= pose[self.vertical_index, 3]+self.positive_delta
            world = world[ground | obstacle]
        else:
            # Without a known floor, preserve the original camera-relative band.
            height_delta = world[:, self.vertical_index] - pose[self.vertical_index, 3]
            within_height = ((height_delta <= self.positive_delta)
                             & (height_delta >= -self.negative_delta))
            world = world[within_height]

        if len(world) and self.min_voxel_points > 1:
            voxel_indices = np.floor(world / self.voxel_size).astype(np.int64)
            _, inverse, counts = np.unique(
                voxel_indices, axis=0, return_inverse=True, return_counts=True
            )
            world = world[counts[inverse] >= self.min_voxel_points]
        # Ground is a depth surface, not a vertical obstruction. Keep its
        # projected ray and endpoint as free-space evidence. Above-floor
        # obstacle endpoints remain occupied, with occupied evidence winning.
        ground_world = np.zeros((0, 3), dtype=float)
        if self.floor_height is not None:
            heights = world[:, self.vertical_index]-self.floor_height
            ground = np.abs(heights) <= self.floor_clearance
            ground_world, world = world[ground], world[~ground]
        self.height_diagnostics = dict(
            floor_height=self.floor_height, floor_clearance=self.floor_clearance,
            ground_samples=len(ground_world), obstacle_samples=len(world))
        points = world[:, self.plane_indices]

        camera_cell = np.floor(
            pose[:3, 3][self.plane_indices] / self.resolution
        ).astype(np.int64)
        endpoint_cells = np.unique(
            np.floor(ray_world[:, self.plane_indices] / self.resolution).astype(np.int64),
            axis=0,
        )
        free_cells = set()
        for endpoint in endpoint_cells:
            free_cells.update(self._ray_cells(camera_cell, endpoint))

        ground_cells = np.unique(
            np.floor(ground_world[:, self.plane_indices] / self.resolution).astype(np.int64), axis=0)
        for endpoint in ground_cells:
            free_cells.update(self._ray_cells(camera_cell, endpoint))
            free_cells.add(tuple(map(int, endpoint)))

        # A return beyond the planning range says the ray stayed clear through
        # the range boundary, so that boundary cell is observed free as well.
        far_ray_endpoints = np.unique(
            np.floor(ray_world[z > self.max_depth][:, self.plane_indices]
                     / self.resolution).astype(np.int64),
            axis=0,
        ) if np.any(z > self.max_depth) else np.empty((0, 2), dtype=np.int64)
        for endpoint in far_ray_endpoints:
            free_cells.add((int(endpoint[0]), int(endpoint[1])))

        self.goal = goal
        if angle_objective is not None:
            angle_objective = float(angle_objective)
            if not np.isfinite(angle_objective):
                raise ValueError("angle_objective must be finite when provided")
        self.angle_objective = angle_objective
        self.camera_pose = pose.copy()
        self._frames.append(points)
        self._point_frames.append(world.copy())
        self._free_frames.append(free_cells)
        self.occupancy = {}
        for frame in self._frames:
            cells = np.floor(frame / self.resolution).astype(np.int64)
            for cell in cells:
                key = (int(cell[0]), int(cell[1]))
                self.occupancy[key] = self.occupancy.get(key, 0) + 1
        self.free_space = set().union(*self._free_frames)
        # Obstacle evidence takes precedence if recent observations disagree.
        self.free_space.difference_update(self.occupancy)

        # Plan a new trajectory
        self.trajectory = self.planner.plan(
            self.camera_pose, self.goal, self.get_map(), resolution=self.resolution
        )
        if len(self.trajectory) == 0:
            print("DepthNavigation: no path found to goal")

        self._trajectory_index = 0
        return self.occupancy

    def get_size_trajectory(self):
        """Return the number of waypoints in the current trajectory."""
        return len(self.trajectory)

    def _smooth_angular_velocity(self, angular):
        """Low-pass filter angular commands across control updates."""
        self._last_angular_velocity += self.angular_smoothing * (
            float(angular) - self._last_angular_velocity
        )
        return self._last_angular_velocity

    def get_command(self, camera_pose=None):
        """Return ``(linear_velocity, angular_velocity)`` in m/s and rad/s.

        An optional latest camera-to-world pose updates the agent position and
        heading used to calculate the command.
        """
        if camera_pose is not None:
            pose = np.asarray(camera_pose, dtype=float)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1])):
                raise ValueError("camera_pose must be a finite 4x4 local-to-world matrix")
            self.camera_pose = pose.copy()

        if self.goal is None or self.camera_pose is None:
            raise RuntimeError("Call set_data before get_command")

        position = self.camera_pose[:3, 3][self.plane_indices]
        target = self.goal[self.plane_indices]
        goal_distance = float(np.linalg.norm(target - position))
        within_goal_tolerance = goal_distance <= self.goal_tolerance
        if within_goal_tolerance and not self.require_goal_heading:
            self._last_angular_velocity = 0.0
            return 0.0, 0.0
        if not len(self.trajectory) and not within_goal_tolerance:
            return 0.0, self._smooth_angular_velocity(0.0)

        # Advance past route cells already reached; the planner's waypoints
        # represent cell centers, so use the goal tolerance as the threshold.
        while self._trajectory_index < len(self.trajectory) - 1:
            delta = self.trajectory[self._trajectory_index] - position
            if np.linalg.norm(delta) > self.goal_tolerance:
                break
            self._trajectory_index += 1
        if within_goal_tolerance:
            if self.angle_objective is not None:
                # Yaw uses Habitat's convention: zero faces world -Z and
                # positive yaw turns toward world -X.
                desired = np.array([
                    -np.sin(self.angle_objective),
                    -np.cos(self.angle_objective),
                ])
            else:
                # Preserve the previous goal-facing behavior for callers that
                # do not provide a final objective angle.
                desired = target - position
                if np.linalg.norm(desired) < 1e-9 and len(self.trajectory) > 1:
                    desired = self.trajectory[-1] - self.trajectory[-2]
                if np.linalg.norm(desired) < 1e-9:
                    return 0.0, self._smooth_angular_velocity(0.0)
                desired /= np.linalg.norm(desired)
        else:
            desired = self.trajectory[self._trajectory_index] - position
            distance = float(np.linalg.norm(desired))
            if distance < 1e-9:
                return 0.0, self._smooth_angular_velocity(0.0)
            desired /= distance

        # Camera forward is local -Z, projected into the selected ground plane.
        forward = (self.camera_pose[:3, :3] @ np.array([0.0, 0.0, -1.0]))[self.plane_indices]
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-9:
            return 0.0, self._smooth_angular_velocity(0.0)
        forward /= forward_norm
        cross = forward[0] * desired[1] - forward[1] * desired[0]
        dot = float(np.clip(np.dot(forward, desired), -1.0, 1.0))
        # In Habitat's X-right/Z-forward ground plane, positive yaw about +Y
        # turns left, so negate the 2D cross product's sign convention.
        angle = float(np.arctan2(-cross, dot))
        if within_goal_tolerance:
            if abs(angle) <= self.heading_tolerance:
                self._last_angular_velocity = 0.0
                return 0.0, 0.0
            angular = float(np.clip(self.angular_gain * angle,
                                    -self.max_angular_speed, self.max_angular_speed))
            return 0.0, self._smooth_angular_velocity(angular)
        angular = float(np.clip(self.angular_gain * angle,
                                -self.max_angular_speed, self.max_angular_speed))
        linear = self.linear_speed * max(0.0, dot)
        return float(linear), self._smooth_angular_velocity(angular)

    def get_sim_action(self, linear, angular, integration_time=0.1):
        """Return a callable that applies velocities to a Habitat simulator."""
        def apply(sim):
            # Physical adapters consume suggestions and wait for measured motion.
            # They never import Habitat or synthesize a new robot pose.
            if hasattr(sim, 'apply_velocity'):
                return sim.apply_velocity(float(linear), float(angular), float(integration_time))
            import habitat_sim
            from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
            vel = habitat_sim.physics.VelocityControl()
            vel.controlling_lin_vel = vel.controlling_ang_vel = True
            vel.lin_vel_is_local = vel.ang_vel_is_local = True
            vel.linear_velocity = np.array([0.0, 0.0, -float(linear)])
            vel.angular_velocity = np.array([0.0, float(angular), 0.0])
            agent = sim.get_agent(0)
            state = agent.get_state()
            previous = habitat_sim.RigidState(quat_to_magnum(state.rotation), state.position)
            target = vel.integrate_transform(float(integration_time), previous)
            state.position = sim.step_filter(previous.translation, target.translation)
            state.rotation = quat_from_magnum(target.rotation)
            agent.set_state(state, reset_sensors=False)
        return apply

    def get_map(self):
        """Return ``(grid, origin)`` for the current planar obstacle map.

        ``grid`` is a uint8 raster with 0 for unknown, 1 for observed free, and
        255 for occupied cells. ``origin`` gives the world-plane coordinates
        of the grid's lower-left cell; each cell spans ``resolution`` metres.
        The first raster axis follows ``plane_indices[0]`` and the second
        follows ``plane_indices[1]``.
        """
        known_cells = set(self.occupancy) | self.free_space
        if not known_cells:
            return np.zeros((0, 0), dtype=np.uint8), np.zeros(2, dtype=float)

        known = np.asarray(list(known_cells), dtype=np.int64)
        minimum = known.min(axis=0)
        maximum = known.max(axis=0)
        # Keep the current pose and goal inside the raster, even when they lie
        # beyond the observed obstacle bounds. A* can then use their real cells
        # instead of clipping either one to an unrelated map edge.
        if self.camera_pose is not None:
            start = np.floor(
                self.camera_pose[:3, 3][self.plane_indices] / self.resolution
            ).astype(np.int64)
            minimum = np.minimum(minimum, start)
            maximum = np.maximum(maximum, start)
        if self.goal is not None:
            goal = np.floor(
                self.goal[self.plane_indices] / self.resolution
            ).astype(np.int64)
            minimum = np.minimum(minimum, goal)
            maximum = np.maximum(maximum, goal)
        padding_cells = int(np.ceil(self.map_padding / self.resolution))
        minimum -= padding_cells
        maximum += padding_cells
        shape = maximum - minimum + 1
        if int(shape[0])*int(shape[1]) > self.max_map_cells:
            raise ValueError(f'Local navigation map would require {tuple(shape)} cells '
                             f'(limit {self.max_map_cells:,}); check odometry and target coordinates')
        grid = np.full(tuple(shape), UNKNOWN, dtype=np.uint8)
        if self.free_space:
            free = np.asarray(list(self.free_space), dtype=np.int64) - minimum
            grid[free[:, 0], free[:, 1]] = FREE
        if self.occupancy:
            occupied = np.asarray(list(self.occupancy), dtype=np.int64) - minimum
            grid[occupied[:, 0], occupied[:, 1]] = OCCUPIED
        origin = minimum.astype(float) * self.resolution
        return grid, origin

    def get_point_cloud(self):
        """Return the measured world XYZ of recent retained obstacle samples.

        Planar occupancy uses ``_frames``. Keep this separate 3D history so the
        display cannot turn floor, door and wall samples into a horizontal
        sheet at the latest camera height.
        """
        if not self._point_frames:
            return np.zeros((0, 3), dtype=float)
        return np.concatenate(tuple(self._point_frames), axis=0).copy()
