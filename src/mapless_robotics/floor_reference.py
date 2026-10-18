"""A fixed horizontal floor in the measured pipeline world frame (Y up)."""
from dataclasses import dataclass, asdict
import numpy as np


def validate_height_band(floor_height, minimum, maximum):
    values = [value for value in (floor_height, minimum, maximum) if value is not None]
    if not np.isfinite(values).all():
        raise ValueError('Floor reference and height limits must be finite')
    if floor_height is None and (minimum is not None or maximum is not None):
        raise ValueError('Height limits require a floor reference (--camera-height)')
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError('Minimum height cannot exceed maximum height')


@dataclass(frozen=True)
class FloorReference:
    floor_y: float
    initial_camera_height: float
    initial_camera_y: float

    @classmethod
    def from_camera_pose(cls, pose, camera_height):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Initial camera pose must be a finite 4x4 matrix')
        if not np.isfinite(camera_height) or camera_height <= 0:
            raise ValueError('Camera height must be finite and positive')
        y = float(pose[1, 3])
        return cls(y-float(camera_height), float(camera_height), y)

    def height_above_floor(self, position):
        position = np.asarray(position, dtype=float)
        return position[..., 1]-self.floor_y

    def to_dict(self):
        return dict(asdict(self), world_up_axis='y', assumption='fixed horizontal floor in this odometry session')
