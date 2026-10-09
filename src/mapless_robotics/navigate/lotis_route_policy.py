"""Visibility-gated LoTIS steering for exploration returns.

This uses LoTIS image-space predictions, not a simulator shortest-path follower.
The upstream adapter drives blindly when no reference pose is visible; this
wrapper spins in place to find the route again and reports failure instead.
"""

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    linear_velocity: float
    angular_velocity: float


class LoTISRoutePolicy:
    def __init__(self, method, max_speed=None, lookahead_distance=0.35,
                 arrival_steps=3, tail_distance=1.0, tail_lost_steps=10,
                 search_speed=0.5, control_dt=0.1, reacquire_steps=5,
                 rejoin_steps=20, rejoin_speed=0.15, max_recoveries=2):
        self.method = method
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
        # Mid-route, a lost robot spins one full revolution in a fixed direction, scoring each
        # view by visible route frames, turns back to the best view and creeps toward the
        # nearest visible frame. Headings are open loop: steps at a fixed rate and control_dt.
        self.search_speed = search_speed
        self.control_dt = control_dt
        self.spin_steps = math.ceil(2 * np.pi / (search_speed * control_dt))
        self.reacquire_steps = reacquire_steps
        self.rejoin_steps = rejoin_steps
        self.rejoin_speed = rejoin_speed
        self.max_recoveries = max_recoveries
        self.reset_search()
        self.in_tail = False
        self.arrival_count = 0
        self.arrived = False
        self.nearest_frame = None  # encoded-route index of the last localized frame
        self.route_lengths = None
        self.turning = False
        self.turn_sign = 1.
        self.turned_angle = 0.
        self.inference_ms = 0.
        self.lost_steps = 0
        self.navigation_failed = False
        self.diagnostics = {}
        self.last_turn = 0.35

    @property
    def searching(self):
        return self.search_phase is not None

    def reset_search(self):
        self.search_phase = None  # 'spin', 'to_best_view' or 'rejoin'
        self.search_step = 0
        self.search_sign = 1.
        self.best_score = (0, 0)  # (visible frames ahead, visible frames)
        self.best_step = 0
        self.rejoin_count = 0
        self.recoveries = 0
        self.tracking_streak = 0

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
        # A route no longer than the tail starts in it, even before LoTIS localizes.
        self.in_tail = self.route_lengths is not None and self.route_lengths[-1] <= self.tail_distance
        self.turning = False
        self.turned_angle = 0.
        self.lost_steps = 0
        self.navigation_failed = False
        self.diagnostics = {}
        self.last_turn = 0.35
        self.reset_search()

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
            return self.search('searching', (0, 0))
        nearest = int(indices[np.argmin(distances[indices])])
        backward = self.method.tracking_mode == 'backward'
        ahead = indices[indices < nearest] if backward else indices[indices > nearest]
        ahead = ahead[::-1] if backward else ahead
        score = (len(ahead), len(indices))
        intrinsics = observation['intrinsics']

        def bearing(frame):
            return float(np.arctan(coords[frame, 1] * intrinsics[0, 2] / intrinsics[0, 0]))
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
            remaining = abs(self.route_lengths[endpoint] - self.route_lengths[nearest])
            self.in_tail = remaining <= self.tail_distance
        else:
            self.in_tail = abs(endpoint - nearest) <= 2
        self.arrival_count = self.arrival_count + 1 if nearest == endpoint else 0
        self.arrived = self.arrival_count >= self.arrival_steps
        if len(ahead):
            target = int(ahead[0])
        elif nearest == endpoint:
            target = nearest
        else:
            return self.search('no_visible_lookahead', score, nearest, bearing(nearest))
        self.lost_steps = 0
        self.navigation_failed = False
        if self.searching:
            # One good frame while spinning is not enough; keep the search state until the
            # route stays in view, so a flicker does not restart the search.
            self.tracking_streak += 1
            if self.tracking_streak >= self.reacquire_steps:
                self.reset_search()
        angle = bearing(target)
        angular = -float(np.clip(self.method.turn_gain * angle, -1., 1.))
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
        if self.turning:
            self.turning = abs(angle) > np.deg2rad(20)
        elif abs(angle) >= np.deg2rad(35):
            self.turning = True
            self.turn_sign = float(np.sign(angular))
            self.turned_angle = 0.
        if self.turning:
            # Commit to the turn direction: during an in-place turn the image shifts and the
            # picked target can jump to the other side (e.g. at a U-turn), flipping the sign.
            angular = self.turn_sign
            self.turned_angle += abs(angular) * self.control_dt
            if self.turned_angle >= 2 * np.pi:
                self.turning = False
                return self.search('turn_exhausted', score, nearest, bearing(nearest))
        if abs(angular) > 0.05:
            self.last_turn = float(np.copysign(0.35, angular))
        linear = 0. if self.turning else speed * max(0., np.cos(angle)) ** 2
        status = 'arrived' if self.arrived else (
            'endpoint_approach' if endpoint_approach else 'tracking')
        self.diagnostics = dict(status=status, visible_points=len(indices),
                                nearest_frame=nearest, target_frame=target,
                                target_distance=float(distances[target]),
                                endpoint_distance=float(distances[endpoint]),
                                inference_ms=round(self.inference_ms, 1),
                                linear_mps=float(linear), angular_rps=angular, turning=self.turning)
        if self.searching:
            self.diagnostics['reacquiring'] = self.tracking_streak
        return VelocityCommand(float(linear), angular)

    def search(self, status, score, nearest=None, nearest_bearing=None):
        self.arrival_count = 0
        self.tracking_streak = 0
        self.lost_steps += 1
        if self.in_tail and self.lost_steps >= self.tail_lost_steps:
            self.arrived = True
            self.diagnostics = dict(status='end_lost', visible_points=score[1],
                                    lost_steps=self.lost_steps,
                                    inference_ms=round(self.inference_ms, 1),
                                    linear_mps=0., angular_rps=0.)
            return VelocityCommand(0., 0.)
        if self.search_phase is None:
            self.search_phase, self.search_step = 'spin', 0
            self.search_sign = float(np.sign(self.last_turn))
            self.best_score, self.best_step = (0, 0), 0
        linear, angular = 0., self.search_sign * self.search_speed
        if self.search_phase == 'spin':
            if score > self.best_score:
                self.best_score, self.best_step = score, self.search_step
            self.search_step += 1
            if self.search_step >= self.spin_steps:
                # Revolution done; keep turning the same way until back at the best view.
                self.search_phase = 'to_best_view' if self.best_score[1] else None
                self.navigation_failed = not self.best_score[1]
        elif self.search_phase == 'to_best_view':
            if self.search_step % self.spin_steps == self.best_step:
                self.search_phase, self.rejoin_count = 'rejoin', 0
            else:
                self.search_step += 1
        if self.search_phase == 'rejoin':
            if nearest_bearing is None or self.rejoin_count >= self.rejoin_steps:
                # Rejoining did not bring the route ahead into view: spin again or give up.
                self.recoveries += 1
                self.navigation_failed = self.recoveries >= self.max_recoveries
                self.search_phase, self.search_step = 'spin', 0
                self.best_score, self.best_step = (0, 0), 0
            else:
                # Creep toward the nearest visible route frame to get back onto the route.
                self.rejoin_count += 1
                angular = -float(np.clip(self.method.turn_gain * nearest_bearing, -1., 1.))
                linear = self.rejoin_speed * max(0., np.cos(nearest_bearing)) ** 2
        if self.navigation_failed:
            linear = angular = 0.
            self.search_phase = None
        self.diagnostics = dict(status='lost_stop' if self.navigation_failed else status,
                                search=self.search_phase, search_step=self.search_step,
                                best_step=self.best_step, best_score=self.best_score,
                                recoveries=self.recoveries, visible_points=score[1],
                                lost_steps=self.lost_steps,
                                inference_ms=round(self.inference_ms, 1),
                                linear_mps=linear, angular_rps=angular)
        if nearest is not None:
            self.diagnostics['nearest_frame'] = nearest
        return VelocityCommand(linear, angular)

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
