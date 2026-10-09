"""Visibility-gated LoTIS steering for exploration returns.

This uses LoTIS image-space predictions, not a simulator shortest-path follower.
The upstream adapter drives blindly when no reference pose is visible; this
wrapper searches in place and reports failure instead.
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    linear_velocity: float
    angular_velocity: float


class LoTISRoutePolicy:
    def __init__(self, method, max_lost_steps=60, max_speed=None, lookahead_distance=0.35,
                 arrival_steps=3):
        self.method = method
        self.max_lost_steps = max_lost_steps
        self.max_speed = method.forward_velocity if max_speed is None else max_speed
        if not np.isfinite(self.max_speed) or self.max_speed <= 0:
            raise ValueError('LoTIS speed must be finite and positive')
        if not np.isfinite(lookahead_distance) or lookahead_distance <= 0:
            raise ValueError('LoTIS lookahead distance must be finite and positive')
        self.lookahead_distance = lookahead_distance
        # Arrived once the endpoint is the nearest visible route frame for arrival_steps
        # consecutive steps. LoTIS distances only rank frames; they are not metres.
        self.arrival_steps = arrival_steps
        self.arrival_count = 0
        self.arrived = False
        self.nearest_frame = None  # encoded-route index of the last localized frame
        self.route_lengths = None
        self.turning = False
        self.inference_ms = 0.
        self.lost_steps = 0
        self.navigation_failed = False
        self.diagnostics = {}
        self.last_turn = 0.35

    def __getattr__(self, name):
        return getattr(self.method, name)

    def reset_episode(self, episode):
        positions = episode.get('route_positions')
        self.route_lengths = None
        if positions is not None:
            positions = np.asarray(positions, dtype=float)
            if positions.shape != (len(episode['rgb_images']), 3) or not np.isfinite(positions).all():
                raise ValueError('Reference positions must match encoded RGB frames')
            self.route_lengths = np.r_[0., np.cumsum(np.linalg.norm(
                np.diff(positions[:, [0, 2]], axis=0), axis=1))]
        self.method.reset_episode(episode)
        self.arrival_count = 0
        self.arrived = False
        self.nearest_frame = None
        self.turning = False
        self.lost_steps = 0
        self.navigation_failed = False
        self.diagnostics = {}
        self.last_turn = 0.35

    def act(self, observation):
        started = time.perf_counter()
        result = self.method.localizer.localize(observation['rgb_pil'], self.method.encoding)
        self.inference_ms = (time.perf_counter() - started) * 1000
        self.method._last_result = result
        coords = np.asarray(result.coords, dtype=float)
        visibility = np.asarray(result.visibility, dtype=float).reshape(-1)
        distances = np.asarray(result.distances, dtype=float).reshape(-1)
        if coords.shape != (len(visibility), 2) or distances.shape != visibility.shape:
            raise ValueError('LoTIS must predict one coordinate, visibility and distance per route frame')
        valid = (np.isfinite(visibility) & (visibility > self.method.visibility_threshold)
                 & np.isfinite(distances) & (distances >= 0)
                 & np.isfinite(coords).all(axis=1) & (np.abs(coords) <= 1).all(axis=1))
        indices = np.flatnonzero(valid)
        if not len(indices):
            return self.search('searching', 0)
        nearest = int(indices[np.argmin(distances[indices])])
        backward = self.method.tracking_mode == 'backward'
        ahead = indices[indices < nearest] if backward else indices[indices > nearest]
        ahead = ahead[::-1] if backward else ahead
        # Avoid steering toward the robot's own projected pose or a sequence of
        # stationary camera captures. Never command motion from those alone.
        if self.route_lengths is not None:
            if len(self.route_lengths) != len(coords):
                raise ValueError('Reference positions do not match LoTIS predictions')
            travel_distance = np.abs(self.route_lengths[ahead] - self.route_lengths[nearest])
            far_enough = ahead[travel_distance >= self.lookahead_distance]
            # Near the end, the last visible route point is the best remaining
            # lookahead. Geometry belongs to the recorded route, not live pose.
            ahead = far_enough if len(far_enough) else ahead[-1:]
        else:
            ahead = ahead[distances[ahead] >= 0.25]
        endpoint = 0 if backward else len(coords) - 1
        self.nearest_frame = nearest
        self.arrival_count = self.arrival_count + 1 if nearest == endpoint else 0
        self.arrived = self.arrival_count >= self.arrival_steps
        if len(ahead):
            target = int(ahead[0])
        elif nearest == endpoint:
            target = nearest
        else:
            return self.search('no_visible_lookahead', len(indices), nearest)
        self.lost_steps = 0
        self.navigation_failed = False
        intrinsics = observation['intrinsics']
        angle = float(np.arctan(coords[target, 1] * intrinsics[0, 2] / intrinsics[0, 0]))
        angular = -float(np.clip(self.method.turn_gain * angle, -1., 1.))
        if abs(angular) > 0.05:
            self.last_turn = float(np.copysign(0.35, angular))
        measured_distance = observation.get('goal_distance_m')
        endpoint_approach = target == endpoint and (
            measured_distance < 0.75 if measured_distance is not None else distances[target] < 0.25)
        speed = min(self.max_speed, self.method.forward_velocity)
        if endpoint_approach:
            if measured_distance is None:
                speed = min(speed, 0.05)
            else:
                # Do not creep across a room because the model predicts a small
                # relative distance. Slow only inside the measured goal region.
                speed *= float(np.clip(measured_distance / 0.75, 0.2, 1.))
        # Hysteresis avoids forward/stop chatter when predictions fluctuate
        # near the turn threshold. Continue small, aligned steering corrections.
        if abs(angle) >= np.deg2rad(35):
            self.turning = True
        elif abs(angle) <= np.deg2rad(20):
            self.turning = False
        linear = 0. if self.turning else speed * max(0., np.cos(angle)) ** 2
        status = 'arrived' if self.arrived else (
            'endpoint_approach' if endpoint_approach else 'tracking')
        self.diagnostics = dict(status=status, visible_points=len(indices),
                                nearest_frame=nearest, target_frame=target,
                                target_distance=float(distances[target]),
                                endpoint_distance=float(distances[endpoint]),
                                inference_ms=round(self.inference_ms, 1),
                                linear_mps=float(linear), angular_rps=angular, turning=self.turning)
        return VelocityCommand(float(linear), angular)

    def search(self, status, visible_points, nearest=None):
        self.arrival_count = 0
        self.lost_steps += 1
        self.navigation_failed = self.lost_steps >= self.max_lost_steps
        self.diagnostics = dict(status='lost_stop' if self.navigation_failed else status,
                                visible_points=visible_points, lost_steps=self.lost_steps,
                                inference_ms=round(self.inference_ms, 1), linear_mps=0.,
                                angular_rps=0. if self.navigation_failed else self.last_turn)
        if nearest is not None:
            self.diagnostics['nearest_frame'] = nearest
        return VelocityCommand(0., 0. if self.navigation_failed else self.last_turn)


def compact_route_ids(frame_ids, states, spacing=0.15):
    """Keep translational reference poses and both endpoints, in original order."""
    if len(frame_ids) <= 2:
        return list(frame_ids)
    retained = [frame_ids[0]]
    for index in frame_ids[1:-1]:
        distance = np.linalg.norm((states[index].position - states[retained[-1]].position)[[0, 2]])
        if distance >= spacing:
            retained.append(index)
    retained.append(frame_ids[-1])
    return retained
