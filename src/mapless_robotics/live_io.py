"""Camera geometry and atomic RGB-D snapshots shared across ROS and ML Python envs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROS_WORLD_TO_PIPELINE = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                                  [0., -1., 0., 0.], [0., 0., 0., 1.]])
PIPELINE_CAMERA_TO_OPTICAL = np.diag([1., -1., -1., 1.])


def pose_matrix(position, quaternion_xyzw):
    p = np.asarray(position, dtype=float)
    q = np.asarray(quaternion_xyzw, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
        raise ValueError('Pose requires finite XYZ and XYZW quaternion')
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError('Invalid zero quaternion')
    x, y, z, w = q / norm
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = p
    return result


def camera_pose_in_pipeline(world_from_imu, imu_from_optical):
    """ROS world Z-up and optical +Z-forward -> pipeline Y-up and -Z-forward.

    imu_from_optical is TF lookup_transform(imu_optical_frame, rgb_optical_frame).
    Odometry must describe the IMU pose in world, never the camera pose itself.
    """
    return (ROS_WORLD_TO_PIPELINE @ world_from_imu @ imu_from_optical
            @ PIPELINE_CAMERA_TO_OPTICAL)


class PoseBuffer:
    """Interpolate bracketed odometry; reject gaps and never extrapolate."""
    def __init__(self, max_gap=0.05, maxlen=1000):
        self.max_gap = max_gap
        self.samples = deque(maxlen=maxlen)

    def add(self, stamp, position, quaternion):
        pose_matrix(position, quaternion)  # validate before changing the buffer
        if not np.isfinite(stamp):
            raise ValueError('Nonfinite odometry timestamp')
        if self.samples and stamp < self.samples[-1][0]:
            self.samples.clear()
            raise ValueError('Odometry clock moved backwards; start a new session')
        sample = (stamp, np.array(position, dtype=float), np.array(quaternion, dtype=float))
        if self.samples and stamp == self.samples[-1][0]:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)

    def at(self, stamp):
        if not self.samples or stamp < self.samples[0][0] or stamp > self.samples[-1][0]:
            return None
        for t, p, q in self.samples:
            if stamp == t:
                return pose_matrix(p, q)
        for first, second in zip(self.samples, list(self.samples)[1:]):
            t0, p0, q0 = first
            t1, p1, q1 = second
            if t0 < stamp < t1:
                if t1 - t0 > self.max_gap:
                    return None
                alpha = (stamp-t0)/(t1-t0)
                q0 = q0 / np.linalg.norm(q0)
                q1 = q1 / np.linalg.norm(q1)
                dot = float(q0 @ q1)
                if dot < 0:
                    q1, dot = -q1, -dot
                if dot > .9995:
                    q = (1-alpha)*q0 + alpha*q1
                else:
                    angle = np.arccos(np.clip(dot, -1, 1))
                    q = (np.sin((1-alpha)*angle)*q0 + np.sin(alpha*angle)*q1)/np.sin(angle)
                return pose_matrix((1-alpha)*p0 + alpha*p1, q)
        return None


@dataclass(frozen=True)
class LiveFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_pose: np.ndarray
    timestamp: float
    written_at: float
    sequence: int
    session: str

    def validate(self):
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError('RGB must be uint8 HxWx3')
        if self.depth.shape != self.rgb.shape[:2] or self.depth.dtype.kind != 'f':
            raise ValueError('Depth must be metric floating point, aligned with RGB')
        k, pose = self.intrinsics, self.camera_pose
        if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError('Invalid pinhole intrinsics')
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1)):
            raise ValueError('Invalid rigid camera-to-world pose')
        if not np.isfinite([self.timestamp, self.written_at]).all() or self.sequence < 0 or not self.session:
            raise ValueError('Invalid frame metadata')
        return self

    @property
    def key(self):
        return self.session, self.sequence


def write_frame(directory, frame):
    frame.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.frame-', suffix='.npz', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as output:
            np.savez(output, rgb=frame.rgb, depth=frame.depth, intrinsics=frame.intrinsics,
                     camera_pose=frame.camera_pose, timestamp=frame.timestamp,
                     written_at=frame.written_at, sequence=frame.sequence, session=frame.session)
        os.replace(temporary, directory/'latest.npz')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_frame(directory):
    with np.load(Path(directory)/'latest.npz', allow_pickle=False) as data:
        frame = LiveFrame(data['rgb'], data['depth'], data['intrinsics'], data['camera_pose'],
                          float(data['timestamp']), float(data['written_at']),
                          int(data['sequence']), str(data['session']))
    return frame.validate()


def guidance_status(frame, advice_frame, *, paused=False, now=None, max_frame_age=.75,
                    max_advice_age=2.):
    now = time.time() if now is None else now
    if paused:
        return 'PAUSED'
    if frame is None or not 0 <= now-frame.written_at <= max_frame_age:
        return 'WAIT: fresh synchronized RGB-D and OpenVINS pose'
    if advice_frame is None or advice_frame.session != frame.session:
        return 'WAIT: guidance'
    if not 0 <= now-advice_frame.written_at <= max_advice_age:
        return 'STOP: guidance expired'
    return None


def atomic_json(path, data):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(data, output)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class OdometryFault(ValueError):
    """Input is inconsistent with the configured handheld operating envelope."""


class OdometryGuard:
    def __init__(self, max_distance=100., max_speed=3.):
        if not np.isfinite([max_distance, max_speed]).all() or min(max_distance, max_speed) <= 0:
            raise ValueError('Odometry operating limits must be finite and positive')
        self.max_distance, self.max_speed = max_distance, max_speed
        self.previous = None

    def check(self, frame):
        position = frame.camera_pose[:3, 3]
        distance = float(np.linalg.norm(position))
        if distance > self.max_distance:
            raise OdometryFault(f'OpenVINS odometry is unhealthy: position norm {distance:,.1f} m '
                                f'exceeds the {self.max_distance:g} m handheld operating radius. '
                                'Restart OpenVINS and the bridge; do not reuse this route.')
        if self.previous is not None:
            if frame.session != self.previous.session:
                raise OdometryFault('Camera/odometry session changed. Start a new route in a fresh output directory.')
            if frame.key != self.previous.key:
                dt = frame.timestamp-self.previous.timestamp
                if dt <= 0:
                    raise OdometryFault('Camera/odometry timestamps moved backwards or repeated')
                speed = float(np.linalg.norm(position-self.previous.camera_pose[:3, 3])/dt)
                if speed > self.max_speed:
                    raise OdometryFault(f'OpenVINS odometry is unhealthy: pose changed at {speed:,.1f} m/s '
                                        f'(handheld limit {self.max_speed:g} m/s). Restart OpenVINS and the bridge.')
        self.previous = frame
        return frame
