"""Detect when local navigation is stuck or cannot produce a path."""

from collections import deque

import numpy as np


class RecoveryBehaviour:
    """Track recent robot positions and planned path sizes for one objective."""

    def __init__(self, max_len_poses=10, max_len_size_array=4, min_distance=0.1,
                 calculate_every=5):
        if max_len_poses < 1 or max_len_size_array < 1:
            raise ValueError("history lengths must be positive")
        if calculate_every < 1:
            raise ValueError("calculate_every must be positive")
        if not np.isfinite(min_distance) or min_distance < 0:
            raise ValueError("min_distance must be finite and non-negative")
        self.max_len_poses = int(max_len_poses)
        self.max_len_size_array = int(max_len_size_array)
        self.min_distance = float(min_distance)
        self.calculate_every = int(calculate_every)
        self.reset()

    def reset(self):
        """Clear history when navigation starts toward a new objective."""
        self.poses = deque(maxlen=self.max_len_poses)
        self.trajectory_sizes = deque(maxlen=self.max_len_size_array)
        self._is_stuck_call_count = 0

    def is_stuck(self, observation, trajectory_size):
        """Record this observation and report an empty plan or no progress.

        ``observation.robot_pose`` is a robot-to-world homogeneous transform.
        Position progress is measured in 3D against the center of the full
        recent pose window.
        """
        self._is_stuck_call_count += 1
        if self._is_stuck_call_count % self.calculate_every != 0:
            return False 

        pose = np.asarray(observation.robot_pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("observation.robot_pose must be a finite 4x4 matrix")
        size = int(trajectory_size)
        if size < 0:
            raise ValueError("trajectory_size must be non-negative")

        self.poses.append(pose[:3, 3].copy())
        self.trajectory_sizes.append(size)

        no_path = (len(self.trajectory_sizes) == self.max_len_size_array
                   and all(path_size == 0 for path_size in self.trajectory_sizes))
        stationary = False
        if len(self.poses) == self.max_len_poses:
            center = np.mean(np.asarray(self.poses), axis=0)
            stationary = np.linalg.norm(self.poses[-1] - center) <= self.min_distance

        return no_path or stationary
