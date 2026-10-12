"""Noisy wheel odometry layered over a Habitat agent.

``OdometryAgent`` stands in for the agent wherever the robot *reads* its pose.
Motion still goes to the real agent (``sim.step``, ``step_velocity``,
``DepthNavigation.get_sim_action``); every pose read integrates the true motion
since the previous read and adds Gaussian noise, so states and captures come
back in a drifting odometry frame. With ``noise == 0`` it is a passthrough to
the ground truth.

Noise model, per integrated motion of planar distance ``d`` (m) and turn
``dtheta`` (rad), in the robot frame:

    translation std = noise * sqrt(d)                (x and z, each)
    yaw std         = noise * sqrt(|dtheta| + d)

Variance grows linearly with travelled distance and rotation (a random walk),
so the drift does not depend on how often the pose is read. The model measures
the motion the robot actually made, so it does not model wheel slip.
"""

from __future__ import annotations

import copy

import numpy as np
import quaternion


def _matrix(pose):
    result = np.eye(4)
    result[:3, :3] = quaternion.as_rotation_matrix(pose.rotation)
    result[:3, 3] = pose.position
    return result


def _set_matrix(pose, matrix):
    pose.position = np.asarray(matrix[:3, 3], dtype=np.float32)
    pose.rotation = quaternion.from_rotation_matrix(matrix[:3, :3])


def _yaw_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def transform_state(state, transform):
    """Copy of an AgentState with the body and every sensor moved by ``transform``."""
    state = copy.deepcopy(state)
    _set_matrix(state, transform @ _matrix(state))
    for sensor in state.sensor_states.values():
        _set_matrix(sensor, transform @ _matrix(sensor))
    return state


class OdometryAgent:
    """Agent proxy whose ``get_state`` reports noisy odometry instead of ground truth."""

    def __init__(self, agent, noise=0.0, seed=None):
        if not np.isfinite(noise) or noise < 0:
            raise ValueError('Odometry noise must be finite and non-negative')
        self.agent = agent
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def enabled(self):
        return self.noise > 0

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def reset(self):
        """Make odometry equal to the current ground-truth pose."""
        self._last_truth = _matrix(self.agent.get_state())
        self._odometry = self._last_truth.copy()

    def _noisy(self, delta):
        distance = float(np.linalg.norm(delta[[0, 2], 3]))
        turn = abs(float(np.arctan2(delta[0, 2], delta[2, 2])))
        if distance < 1e-9 and turn < 1e-9:  # stationary, up to float error
            return delta
        noisy = delta.copy()
        noisy[[0, 2], 3] += self.rng.normal(0.0, self.noise * np.sqrt(distance), 2)
        yaw_error = self.rng.normal(0.0, self.noise * np.sqrt(turn + distance))
        noisy[:3, :3] = _yaw_rotation(yaw_error) @ delta[:3, :3]
        return noisy

    def _correction(self, truth_state):
        """Integrate motion since the last read; return odometry @ inv(ground truth)."""
        truth = _matrix(truth_state)
        delta = np.linalg.inv(self._last_truth) @ truth
        self._odometry = self._odometry @ self._noisy(delta)
        self._last_truth = truth
        return self._odometry @ np.linalg.inv(truth)

    def get_state(self):
        state = self.agent.get_state()
        if not self.enabled:
            return state
        return transform_state(state, self._correction(state))

    def set_state(self, state, *args, **kwargs):
        """Move to an odometry-frame state as the relative motion the robot would execute."""
        if self.enabled:
            correction = self._correction(self.agent.get_state())
            state = transform_state(state, np.linalg.inv(correction))
        return self.agent.set_state(state, *args, **kwargs)

    def ground_truth_state(self):
        return self.agent.get_state()

    def teleport(self, state, *args, **kwargs):
        """Place the real agent at a ground-truth state and restart odometry there."""
        result = self.agent.set_state(state, *args, **kwargs)
        self.reset()
        return result
