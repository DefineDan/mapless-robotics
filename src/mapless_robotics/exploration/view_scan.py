"""Bounded camera sweeps and reproducible Habitat start selection."""

import numpy as np


def choose_start(pathfinder, seed=None, position=None, heading=None):
    """Return a navigable XYZ, yaw in degrees, and the actual random seed."""
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
    if not 0 <= seed < 2**32:
        raise ValueError('--seed must be between 0 and 4294967295')
    pathfinder.seed(seed)
    if position is None:
        position = pathfinder.get_random_navigable_point()
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Start position must be a finite Habitat XYZ point')
    if not pathfinder.is_navigable(position):
        raise ValueError('Start position is not on the navmesh; choose another XYZ point')
    heading = float(np.random.default_rng(seed).uniform(-180, 180) if heading is None else heading)
    if not np.isfinite(heading):
        raise ValueError('--start-yaw must be finite')
    return position, heading, seed


class ViewScan:
    """Analyze each direction while stationary before choosing a frontier."""

    def __init__(self, start_yaw, views=6):
        if views < 2:
            raise ValueError('A scan needs at least two directions')
        self.start_yaw = start_yaw
        self.views = views
        self.completed_views = 0
        self.waiting_frame = None

    @property
    def complete(self):
        return self.completed_views == self.views

    def command(self, current_yaw):
        if self.complete or self.waiting_frame is not None:
            return 0.0, False
        target = self.start_yaw + self.completed_views * (2 * np.pi / self.views)
        error = float(np.arctan2(np.sin(target - current_yaw), np.cos(target - current_yaw)))
        ready = abs(error) <= np.deg2rad(5)
        return (0.0 if ready else float(np.clip(2 * error, -1., 1.))), ready

    def analyzed(self, frame_index):
        if self.waiting_frame is not None and frame_index == self.waiting_frame:
            self.completed_views += 1
            self.waiting_frame = None
            return True
        return False
