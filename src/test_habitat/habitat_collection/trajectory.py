"""Random path generation and pure-pursuit traversal for automatic collection.

Ports the navmesh sampling + velocity-control path following from
../habitat_data_gen/data_generation/habitat/trajectory_generator.py onto the
lighter-weight HabitatCollectionEnv, without the offline pipeline's camera
randomization / query splitting (those are canonical_v2 dataset-generation
concerns, not raw trajectory collection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .env import HabitatCollectionEnv, StepResult


@dataclass
class TrajectorySettings:
    min_path_distance: float = 3.0
    max_trajectory_length: float = 30.0
    control_frequency: float = 10.0
    frame_skip: int = 1
    look_ahead_distance: float = 0.4
    max_forward_speed: float = 0.7
    max_turn_speed: float = 0.8
    arrival_distance: float = 0.3
    save_every_n_frames: Tuple[int, int] = (2, 8)

    @classmethod
    def from_config(cls, config) -> "TrajectorySettings":
        settings = config.TRAJECTORY_SETTINGS
        return cls(
            min_path_distance=float(settings.MIN_PATH_DISTANCE),
            max_trajectory_length=float(settings.MAX_TRAJECTORY_LENGTH),
            control_frequency=float(settings.CONTROL_FREQUENCY),
            frame_skip=int(settings.FRAME_SKIP),
            look_ahead_distance=float(settings.LOOK_AHEAD_DISTANCE),
            max_forward_speed=float(settings.MAX_FORWARD_SPEED),
            max_turn_speed=float(settings.MAX_TURN_SPEED),
            arrival_distance=float(settings.ARRIVAL_DISTANCE),
            save_every_n_frames=tuple(settings.SAVE_EVERY_N_FRAMES),
        )


def sample_random_path(
    env: HabitatCollectionEnv, min_path_distance: float, max_attempts: int = 200
) -> Tuple[Optional[np.ndarray], float]:
    """Sample a navmesh path between two random points at least min_path_distance apart.

    Returns (path_points, geodesic_distance), or (None, 0.0) if no such path
    was found within max_attempts.
    """
    best_points = None
    best_distance = 0.0

    for _ in range(max_attempts):
        start = env.sample_navigable_point()
        island = env.island_of(start)
        end = env.sample_navigable_point(island_index=island)
        # if np.linalg.norm(end - start) < min_path_distance:
        #     continue
        points, geodesic_distance = env.compute_path(start, end)
        if points is None:
            continue
        if geodesic_distance > best_distance:
            best_distance = geodesic_distance
            best_points = points
        if geodesic_distance >= min_path_distance:
            return points, geodesic_distance
        
    print(
        f"Best path found after {max_attempts} attempts: "
        f"{best_distance:.2f}m "
        f"(required >= {min_path_distance:.2f}m)"
    )
    return None, 0.0


def traverse_path(
    env: HabitatCollectionEnv,
    path_points: np.ndarray,
    settings: Optional[TrajectorySettings] = None,
) -> Iterator[StepResult]:
    """Drive the agent along path_points with a pure-pursuit controller.

    Yields a StepResult for every saved frame; the pose dict is augmented
    with path_progress, distance_to_path, total_path_length, elapsed_time.
    """
    settings = settings or TrajectorySettings()
    path_points = np.asarray(path_points, dtype=np.float32)
    segments, total_length = _path_segments(path_points)
    _face_first_waypoint(env, path_points)

    dt = 1.0 / (settings.control_frequency * settings.frame_skip)
    save_every_n = np.arange(*settings.save_every_n_frames)
    next_saved_frame = 1
    frame_counter = 0
    time_elapsed = 0.0
    linear_velocity = angular_velocity = 0.0

    while time_elapsed < settings.max_trajectory_length:
        state = env.sim.get_agent(0).get_state()
        position = np.asarray(state.position, dtype=np.float32)

        if np.linalg.norm(position - path_points[-1]) < settings.arrival_distance:
            return

        _, progress, distance_to_path = _project_onto_path(position, segments)

        if frame_counter % settings.frame_skip == 0:
            look_ahead = _look_ahead_point(
                segments, total_length, progress, settings.look_ahead_distance
            )
            linear_velocity, angular_velocity = _pure_pursuit_control(
                position,
                state.rotation,
                look_ahead,
                distance_to_path,
                settings.max_forward_speed,
                settings.max_turn_speed,
            )

        env.step_velocity(linear_velocity, angular_velocity, dt)
        frame_counter += 1
        time_elapsed += dt

        if frame_counter == next_saved_frame:
            result = env.observe(action="trajectory")
            result.pose["path_progress"] = float(progress)
            result.pose["distance_to_path"] = float(distance_to_path)
            result.pose["total_path_length"] = float(total_length)
            result.pose["elapsed_time"] = float(time_elapsed)
            yield result
            next_saved_frame += int(np.random.choice(save_every_n))


def _face_first_waypoint(env: HabitatCollectionEnv, path_points: np.ndarray) -> None:
    from habitat_sim.utils.common import quat_from_two_vectors

    start = path_points[0]
    if len(path_points) > 1:
        direction = path_points[1] - path_points[0]
        direction[1] = 0.0
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), direction)
            env.set_pose(start, rotation)
            return
    env.set_pose(start, env.sim.get_agent(0).get_state().rotation)


def _path_segments(path_points: np.ndarray) -> Tuple[List[dict], float]:
    """Accumulated-length segments along path_points, ignoring height."""
    segments = []
    length = 0.0
    for i in range(1, len(path_points)):
        segment_vector = path_points[i] - path_points[i - 1]
        segment_vector[1] = 0.0
        segment_length = float(np.linalg.norm(segment_vector))
        segments.append(
            {
                "start_point": path_points[i - 1],
                "end_point": path_points[i],
                "length": segment_length,
                "accumulated_length": length,
            }
        )
        length += segment_length
    return segments, length


def _project_onto_path(position: np.ndarray, segments: List[dict]) -> Tuple[np.ndarray, float, float]:
    """Project position onto the closest point on the path.

    Returns (closest_point, path_progress, distance_to_path).
    """
    min_distance = float("inf")
    closest_point = position
    path_progress = 0.0

    for segment in segments:
        start = segment["start_point"]
        end = segment["end_point"]

        segment_vector = end - start
        segment_vector[1] = 0.0
        point_vector = position - start
        point_vector[1] = 0.0

        segment_length = segment["length"]
        if segment_length == 0:
            continue

        segment_dir = segment_vector / segment_length
        projection = np.dot(point_vector, segment_dir)
        projection = max(0.0, min(segment_length, projection))

        projected_point = start + segment_dir * projection
        projected_point[1] = position[1]

        distance = float(np.linalg.norm(position - projected_point))
        if distance < min_distance:
            min_distance = distance
            closest_point = projected_point
            path_progress = segment["accumulated_length"] + projection

    return closest_point, path_progress, min_distance


def _look_ahead_point(
    segments: List[dict], total_length: float, current_progress: float, look_ahead_distance: float
) -> np.ndarray:
    """Point a fixed distance ahead of current_progress along the path."""
    target_progress = current_progress + look_ahead_distance
    if target_progress >= total_length:
        return segments[-1]["end_point"]

    for segment in segments:
        segment_end_progress = segment["accumulated_length"] + segment["length"]
        if segment["accumulated_length"] <= target_progress < segment_end_progress:
            start = segment["start_point"]
            end = segment["end_point"]
            t = (target_progress - segment["accumulated_length"]) / segment["length"]
            return start + t * (end - start)

    return segments[-1]["end_point"]


def _pure_pursuit_control(
    position: np.ndarray,
    rotation,
    look_ahead_point: np.ndarray,
    distance_to_path: float,
    max_forward_speed: float,
    max_turn_speed: float,
) -> Tuple[float, float]:
    """Body-frame (linear, angular) velocity steering toward look_ahead_point."""
    from habitat_sim.utils.common import quat_rotate_vector

    forward = quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0]))

    direction = look_ahead_point - position
    direction[1] = 0.0
    distance = np.linalg.norm(direction)
    if distance < 0.05:
        return 0.0, 0.0
    direction = direction / distance

    forward_xz = np.array([forward[0], forward[2]])
    direction_xz = np.array([direction[0], direction[2]])
    alignment = float(np.dot(forward_xz, direction_xz))
    forward_norm = np.linalg.norm(forward_xz)
    if forward_norm > 0:
        alignment /= forward_norm  # direction_xz is already unit length

    # Cross product (Habitat's left-handed convention) for turn direction.
    cross = -(forward[0] * direction[2] - forward[2] * direction[0])

    forward_speed_factor = max(0.0, alignment)
    path_distance_factor = max(0.0, 1.0 - distance_to_path / 0.5)
    forward_velocity = max_forward_speed * forward_speed_factor * path_distance_factor

    turn_factor = min(1.0, 0.5 + 0.5 * (1 - abs(alignment)))
    path_turn_factor = min(1.0, 1.0 + distance_to_path)
    angular_velocity = max_turn_speed * np.sign(cross) * turn_factor * path_turn_factor

    if alignment < 0.5:  # More than ~60 degrees off target
        forward_velocity *= max(0.0, alignment)

    return forward_velocity, angular_velocity
