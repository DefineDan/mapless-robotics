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
    frames, limiting the effect of noisy pose estimates.
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
        linear_speed=0.25,
        angular_gain=1.5,
        max_angular_speed=1.0,
        goal_tolerance=0.25,
        heading_tolerance=0.1,
        vertical_axis="y",
        camera_intrinsics=None,
        planner=None,
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
                  linear_speed, angular_gain, max_angular_speed, goal_tolerance,
                  heading_tolerance)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("navigation distances and speeds must be finite and positive")
        self.history_size = int(history_size)
        self.positive_delta = float(positive_delta)
        self.negative_delta = float(negative_delta)
        self.resolution = float(occupancy_resolution)
        self.map_padding = float(map_padding)
        self.unknown_cost = float(unknown_cost)
        self.max_depth = float(max_depth)
        self.linear_speed = float(linear_speed)
        self.angular_gain = float(angular_gain)
        self.max_angular_speed = float(max_angular_speed)
        self.goal_tolerance = float(goal_tolerance)
        self.heading_tolerance = float(heading_tolerance)
        self.vertical_index = {"x": 0, "y": 1, "z": 2}[vertical_axis]
        self.plane_indices = [i for i in range(3) if i != self.vertical_index]
        self.camera_intrinsics = camera_intrinsics
        self.planner = planner or AStarPlanner(
            self.resolution,
            self.plane_indices,
            inflation_radius=0.2,
            unknown_cost=self.unknown_cost,
        )
        self.trajectory = np.zeros((0, 2), dtype=float)
        self._trajectory_index = 0
        self._frames = deque(maxlen=self.history_size)
        self._free_frames = deque(maxlen=self.history_size)
        self.goal = None
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

    def set_data(self, goal, depth_image, camera_pose, camera_intrinsics=None):
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
        valid = np.isfinite(z) & (z > 0) & (z <= self.max_depth)
        rows, cols, z = rows[valid], cols[valid], z[valid]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        local = np.column_stack(((cols - cx) * z / fx,
                                 -(rows - cy) * z / fy,
                                 -z))
        world = local @ pose[:3, :3].T + pose[:3, 3]
        height_delta = world[:, self.vertical_index] - pose[self.vertical_index, 3]
        within_height = ((height_delta <= self.positive_delta)
                         & (height_delta >= -self.negative_delta))
        points = world[within_height][:, self.plane_indices]

        # A depth ray confirms free space up to (but not including) its surface
        # endpoint. Many pixels project into the same cell, so deduplicate the
        # endpoints before tracing them through the planar grid.
        camera_cell = np.floor(
            pose[:3, 3][self.plane_indices] / self.resolution
        ).astype(np.int64)
        endpoint_cells = np.unique(
            np.floor(world[:, self.plane_indices] / self.resolution).astype(np.int64),
            axis=0,
        )
        free_cells = set()
        for endpoint in endpoint_cells:
            free_cells.update(self._ray_cells(camera_cell, endpoint))

        self.goal = goal
        self.camera_pose = pose.copy()
        self._frames.append(points)
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
        if not len(self.trajectory):
            return 0.0, 0.0

        # Advance past route cells already reached; the planner's waypoints
        # represent cell centers, so use the goal tolerance as the threshold.
        while self._trajectory_index < len(self.trajectory) - 1:
            delta = self.trajectory[self._trajectory_index] - position
            if np.linalg.norm(delta) > self.goal_tolerance:
                break
            self._trajectory_index += 1
        if goal_distance <= self.goal_tolerance:
            # Face the goal while approaching it. At the goal itself, use the
            # last path segment as the desired final heading.
            desired = target - position
            if np.linalg.norm(desired) < 1e-9 and len(self.trajectory) > 1:
                desired = self.trajectory[-1] - self.trajectory[-2]
            if np.linalg.norm(desired) < 1e-9:
                return 0.0, 0.0
            desired /= np.linalg.norm(desired)
        else:
            desired = self.trajectory[self._trajectory_index] - position
            distance = float(np.linalg.norm(desired))
            if distance < 1e-9:
                return 0.0, 0.0
            desired /= distance

        # Camera forward is local -Z, projected into the selected ground plane.
        forward = (self.camera_pose[:3, :3] @ np.array([0.0, 0.0, -1.0]))[self.plane_indices]
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-9:
            return 0.0, 0.0
        forward /= forward_norm
        cross = forward[0] * desired[1] - forward[1] * desired[0]
        dot = float(np.clip(np.dot(forward, desired), -1.0, 1.0))
        # In Habitat's X-right/Z-forward ground plane, positive yaw about +Y
        # turns left, so negate the 2D cross product's sign convention.
        angle = float(np.arctan2(-cross, dot))
        if goal_distance <= self.goal_tolerance:
            if abs(angle) <= self.heading_tolerance:
                return 0.0, 0.0
            angular = float(np.clip(self.angular_gain * angle,
                                    -self.max_angular_speed, self.max_angular_speed))
            return 0.0, angular
        angular = float(np.clip(self.angular_gain * angle,
                                -self.max_angular_speed, self.max_angular_speed))
        linear = self.linear_speed * max(0.0, dot)
        return float(linear), angular

    def get_sim_action(self, linear, angular, integration_time=0.1):
        """Return a callable that applies velocities to a Habitat simulator."""
        import habitat_sim

        from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum

        vel = habitat_sim.physics.VelocityControl()
        vel.controlling_lin_vel = vel.controlling_ang_vel = True
        vel.lin_vel_is_local = vel.ang_vel_is_local = True
        vel.linear_velocity = np.array([0.0, 0.0, -float(linear)])
        vel.angular_velocity = np.array([0.0, float(angular), 0.0])

        def apply(sim):
            agent = sim.get_agent(0)
            state = agent.get_state()
            previous = habitat_sim.RigidState(
                quat_to_magnum(state.rotation), state.position
            )
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
        grid = np.full(tuple(maximum - minimum + 1), UNKNOWN, dtype=np.uint8)
        if self.free_space:
            free = np.asarray(list(self.free_space), dtype=np.int64) - minimum
            grid[free[:, 0], free[:, 1]] = FREE
        if self.occupancy:
            occupied = np.asarray(list(self.occupancy), dtype=np.int64) - minimum
            grid[occupied[:, 0], occupied[:, 1]] = OCCUPIED
        origin = minimum.astype(float) * self.resolution
        return grid, origin

    def get_point_cloud(self):
        """Return recent obstacle points in Habitat world XYZ coordinates.

        The planar coordinates are inserted on ``plane_indices`` and the
        latest camera position supplies the configured vertical coordinate.
        This maps the navigation frame's vertical axis onto Habitat's world
        axes (Y is up for the default configuration).
        """
        if self.camera_pose is None or not self._frames:
            return np.zeros((0, 3), dtype=float)
        points = np.concatenate(tuple(self._frames), axis=0)
        if not len(points):
            return np.zeros((0, 3), dtype=float)
        world_points = np.zeros((len(points), 3), dtype=float)
        world_points[:, self.plane_indices] = points
        world_points[:, self.vertical_index] = self.camera_pose[
            self.vertical_index, 3
        ]
        return world_points
