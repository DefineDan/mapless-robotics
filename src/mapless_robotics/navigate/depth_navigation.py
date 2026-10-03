"""Depth-only local navigation with a short rolling occupancy window."""

from collections import deque

import numpy as np


class DepthNavigation:
    """Generate planar velocity commands from a goal and recent depth frames.

    Poses are Habitat local-to-world transforms (X right, Y up, camera forward
    along local -Z). Occupancy is rebuilt from only the last ``history_size``
    frames, limiting the effect of noisy pose estimates.
    """

    def __init__(
        self,
        history_size=5,
        delta=0.4,
        occupancy_resolution=0.1,
        max_depth=5.0,
        obstacle_influence=0.8,
        linear_speed=0.25,
        angular_gain=1.5,
        max_angular_speed=1.0,
        goal_tolerance=0.25,
        vertical_axis="y",
        camera_intrinsics=None,
    ):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if vertical_axis not in {"x", "y", "z"}:
            raise ValueError("vertical_axis must be 'x', 'y', or 'z'")
        values = (delta, occupancy_resolution, max_depth, obstacle_influence,
                  linear_speed, angular_gain, max_angular_speed, goal_tolerance)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("navigation distances and speeds must be finite and positive")
        self.history_size = int(history_size)
        self.delta = float(delta)
        self.resolution = float(occupancy_resolution)
        self.max_depth = float(max_depth)
        self.obstacle_influence = float(obstacle_influence)
        self.linear_speed = float(linear_speed)
        self.angular_gain = float(angular_gain)
        self.max_angular_speed = float(max_angular_speed)
        self.goal_tolerance = float(goal_tolerance)
        self.vertical_index = {"x": 0, "y": 1, "z": 2}[vertical_axis]
        self.plane_indices = [i for i in range(3) if i != self.vertical_index]
        self.camera_intrinsics = camera_intrinsics
        self._frames = deque(maxlen=self.history_size)
        self.goal = None
        self.camera_pose = None
        self.occupancy = {}

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
        within_height = np.abs(world[:, self.vertical_index] - pose[self.vertical_index, 3]) <= self.delta
        points = world[within_height][:, self.plane_indices]

        self.goal = goal
        self.camera_pose = pose.copy()
        self._frames.append(points)
        self.occupancy = {}
        for frame in self._frames:
            cells = np.floor(frame / self.resolution).astype(np.int64)
            for cell in cells:
                key = (int(cell[0]), int(cell[1]))
                self.occupancy[key] = self.occupancy.get(key, 0) + 1
        return self.occupancy

    def get_command(self):
        """Return ``(linear_velocity, angular_velocity)`` in m/s and rad/s."""
        if self.goal is None or self.camera_pose is None:
            raise RuntimeError("Call set_data before get_command")
        position = self.camera_pose[:3, 3][self.plane_indices]
        target = self.goal[self.plane_indices]
        attractive = target - position
        distance = float(np.linalg.norm(attractive))
        if distance <= self.goal_tolerance:
            return 0.0, 0.0
        force = attractive / max(distance, self.goal_tolerance)

        # Repel from nearby occupied cell centers. Multiple observations in a
        # cell strengthen confidence that the obstacle is real.
        for cell, count in self.occupancy.items():
            obstacle = (np.asarray(cell, dtype=float) + 0.5) * self.resolution
            away = position - obstacle
            obstacle_distance = float(np.linalg.norm(away))
            if 1e-6 < obstacle_distance < self.obstacle_influence:
                strength = min(count, self.history_size) / self.history_size
                force += (strength * (1 / obstacle_distance - 1 / self.obstacle_influence)
                          / obstacle_distance**2) * (away / obstacle_distance)

        force_norm = float(np.linalg.norm(force))
        if force_norm < 1e-9:
            return 0.0, 0.0
        desired = force / force_norm
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
        angular = float(np.clip(self.angular_gain * angle,
                                -self.max_angular_speed, self.max_angular_speed))
        linear = self.linear_speed * max(0.0, dot)
        return float(linear), angular
