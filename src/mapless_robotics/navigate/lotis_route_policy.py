"""Visibility-gated LoTIS steering for exploration returns.

This uses LoTIS image-space predictions, not a simulator shortest-path follower.
The upstream adapter drives blindly when no reference pose is visible; this
wrapper searches in place and reports failure instead.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    linear_velocity: float
    angular_velocity: float


class LoTISRoutePolicy:
    def __init__(self, method, max_lost_steps=60, max_speed=None, lookahead_distance=0.35,
                 arrival_steps=3, tail_distance=1.0, tail_lost_steps=10,
                 endpoint_stall_steps=60, slowdown_distance=0.5):
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
        # In the last tail_distance of recorded route LoTIS rarely recovers once lost
        # (backward legs lose roughly the last metre), so give up quickly and count the
        # leg as arrived instead of searching for max_lost_steps.
        self.tail_distance = tail_distance
        self.tail_lost_steps = tail_lost_steps
        # While approaching a visible endpoint, accept arrival once LoTIS stops localizing
        # the robot closer to it for endpoint_stall_steps. Progress is the recorded path
        # remaining from the nearest frame, so odometry drift does not affect it.
        self.endpoint_stall_steps = endpoint_stall_steps
        # Speed tapers within this recorded path length of the endpoint.
        self.slowdown_distance = slowdown_distance
        self.endpoint_best = None
        self.endpoint_stalled_steps = 0
        self.in_tail = False
        self.arrival_count = 0
        self.arrived = False
        self.nearest_frame = None  # encoded-route index of the last localized frame
        self.route_lengths = None
        self.turning = False
        self.turn_sign = 0.  # turn direction latched while turning in place
        self.inference_ms = 0.
        self.lost_steps = 0
        self.navigation_failed = False
        self.diagnostics = {}
        self.last_turn = 0.35
        # Recovery state (see set_window): only frames within (back, ahead) metres of
        # recorded path around progress may be matched. Off during normal tracking.
        self.route_xz = None
        self.window = None
        self.progress = None
        self.progress_candidate = None
        self.history = deque(maxlen=80)  # recent nearest frames, for find_fold

    def __getattr__(self, name):
        return getattr(self.method, name)

    def reset_episode(self, episode):
        positions = episode.get('route_positions')
        self.route_lengths = None
        self.route_xz = None
        if positions is not None:
            positions = np.asarray(positions, dtype=float)
            if positions.shape != (len(episode['rgb_images']), 3) or not np.isfinite(positions).all():
                raise ValueError('Reference positions must match encoded RGB frames')
            self.route_xz = positions[:, [0, 2]]
            self.route_lengths = np.r_[0., np.cumsum(np.linalg.norm(
                np.diff(positions[:, [0, 2]], axis=0), axis=1))]
        self.method.reset_episode(episode)
        self.arrival_count = 0
        self.arrived = False
        self.nearest_frame = None
        self.endpoint_best = None
        self.endpoint_stalled_steps = 0
        # A route no longer than the tail starts in it, even before LoTIS localizes.
        self.in_tail = self.route_lengths is not None and self.route_lengths[-1] <= self.tail_distance
        self.turning = False
        self.turn_sign = 0.
        self.lost_steps = 0
        self.navigation_failed = False
        self.diagnostics = {}
        self.last_turn = 0.35
        self.window = None
        self.progress = None
        self.progress_candidate = None
        self.history.clear()

    def predict(self, observation):
        """Localize the view; return coords, visibility, distances and the usable-frame mask."""
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
        if self.window is not None:
            back, ahead = self.window
            offset = self.along_offset(np.arange(len(valid)))
            valid &= (offset >= -back) & (offset <= ahead)
        return coords, visibility, distances, valid

    def along_offset(self, frames, origin=None):
        """Recorded path length of frames ahead of origin (default progress) in travel direction."""
        origin = self.progress if origin is None else origin
        sign = -1. if self.method.tracking_mode == 'backward' else 1.
        return sign * (self.route_lengths[frames] - self.route_lengths[origin])

    def act(self, observation):
        coords, visibility, distances, valid = self.predict(observation)
        indices = np.flatnonzero(valid)
        if not len(indices):
            return self.search('searching', 0)
        nearest = int(indices[np.argmin(distances[indices])])
        self.history.append(nearest)
        if self.window is not None:
            # Slide the window once the robot is matched further along for two steps.
            if nearest == self.progress_candidate and self.along_offset(nearest) > 0:
                self.progress = nearest
            self.progress_candidate = nearest
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
        if self.route_lengths is not None:
            # Recorded path length left from where LoTIS localizes the robot.
            remaining = abs(self.route_lengths[endpoint] - self.route_lengths[nearest])
            self.in_tail = remaining <= self.tail_distance
        else:
            remaining = None
            self.in_tail = abs(endpoint - nearest) <= 2
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
        endpoint_approach = target == endpoint and (
            remaining < self.slowdown_distance if remaining is not None else distances[target] < 0.25)
        speed = min(self.max_speed, self.method.forward_velocity)
        if endpoint_approach:
            if remaining is None:
                speed = min(speed, 0.05)
            else:
                # Do not creep across a room because the model predicts a small
                # relative distance. Slow only inside the last recorded slowdown_distance.
                speed *= float(np.clip(remaining / self.slowdown_distance, 0.33, 1.))
        # Hysteresis avoids forward/stop chatter when predictions fluctuate
        # near the turn threshold. Continue small, aligned steering corrections.
        if abs(angle) >= np.deg2rad(35):
            if not self.turning:
                self.turn_sign = float(np.sign(angular))
            self.turning = True
        elif abs(angle) <= np.deg2rad(20):
            self.turning = False
        if self.window is not None and self.turning and np.sign(angular) != self.turn_sign:
            # Recovery only: keep turning in place the same way. A target that switches
            # sides between frames would otherwise reverse the turn every step.
            angular = self.turn_sign * abs(angular)
            self.last_turn = float(np.copysign(0.35, angular))
        linear = 0. if self.turning else speed * max(0., np.cos(angle)) ** 2
        endpoint_stalled = self.track_endpoint_progress(
            endpoint_approach, remaining if remaining is not None else abs(endpoint - nearest),
            driving=linear > 0)
        if endpoint_stalled:
            self.arrived = True
        if self.arrival_count >= self.arrival_steps:
            status = 'arrived'
        elif endpoint_stalled:
            status = 'endpoint_stalled'
        else:
            status = 'endpoint_approach' if endpoint_approach else 'tracking'
        self.diagnostics = dict(status=status, visible_points=len(indices),
                                nearest_frame=nearest, target_frame=target,
                                target_distance=float(distances[target]),
                                endpoint_distance=float(distances[endpoint]),
                                inference_ms=round(self.inference_ms, 1),
                                linear_mps=float(linear), angular_rps=angular, turning=self.turning)
        if remaining is not None:
            self.diagnostics['remaining_m'] = round(float(remaining), 2)
        if self.window is not None:
            self.diagnostics['progress_frame'] = self.progress
        return VelocityCommand(float(linear), angular)

    def set_window(self, centre, back, ahead):
        """Restrict matching to (back, ahead) metres of recorded path around centre."""
        if self.route_lengths is None:
            return False
        self.window = (back, ahead)
        self.progress = centre
        self.progress_candidate = None
        return True

    def resume(self):
        """Clear failure state for a retry; keep the encoding, window and progress."""
        self.arrival_count = 0
        self.arrived = False
        self.endpoint_best = None
        self.endpoint_stalled_steps = 0
        self.turning = False
        self.turn_sign = 0.
        self.lost_steps = 0
        self.navigation_failed = False
        self.progress_candidate = None
        self.history.clear()

    def find_fold(self, min_path=1.5, max_gap=1.5, min_jumps=2):
        """Detect matches oscillating between two legs of a U-turn.

        A fold jump is a change of nearest frame between consecutive matches that
        skips at least min_path of recorded route but stays within max_gap in space:
        both frames claim the robot's location, so they lie on parallel legs.
        Returns (earlier, later) frames of the last jump, or None.
        """
        if self.route_lengths is None:
            return None
        history = list(self.history)
        jumps = []
        for first, second in zip(history, history[1:]):
            skipped = self.along_offset(second, origin=first)
            gap = np.linalg.norm(self.route_xz[second] - self.route_xz[first])
            if abs(skipped) >= min_path and gap <= max_gap:
                jumps.append((first, second) if skipped > 0 else (second, first))
        return jumps[-1] if len(jumps) >= min_jumps else None

    def score_view(self, observation):
        """Visibility summed over usable frames ahead of progress, without steering."""
        _, visibility, _, valid = self.predict(observation)
        if self.route_lengths is not None and self.progress is not None:
            valid &= self.along_offset(np.arange(len(valid))) >= 0.15
        return float(visibility[valid].sum())

    def track_endpoint_progress(self, endpoint_approach, remaining, driving=False):
        """True once the endpoint approach has not advanced for endpoint_stall_steps.

        Only steps that drive forward count; turning in place is not a stall.
        """
        if not endpoint_approach:
            self.endpoint_best = None
            self.endpoint_stalled_steps = 0
            return False
        if self.endpoint_best is None or remaining < self.endpoint_best - 0.03:
            self.endpoint_best = remaining
            self.endpoint_stalled_steps = 0
        elif driving:
            self.endpoint_stalled_steps += 1
        return self.endpoint_stalled_steps >= self.endpoint_stall_steps

    def search(self, status, visible_points, nearest=None):
        self.arrival_count = 0
        self.track_endpoint_progress(False, None)
        self.lost_steps += 1
        if self.in_tail and self.lost_steps >= self.tail_lost_steps:
            self.arrived = True
            self.diagnostics = dict(status='end_lost', visible_points=visible_points,
                                    lost_steps=self.lost_steps,
                                    inference_ms=round(self.inference_ms, 1),
                                    linear_mps=0., angular_rps=0.)
            return VelocityCommand(0., 0.)
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
