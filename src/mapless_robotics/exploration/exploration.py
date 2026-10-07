"""Capture snapshots, inference, and triggers without simulator dependencies."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import cv2

from .exploration_graph import pose_matrix, ExplorationGraph


def camera_intrinsics(width, height, hfov_degrees):
    focal = width / (2 * np.tan(np.deg2rad(hfov_degrees) / 2))
    return np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32
    )


def opencv_extrinsic(camera_pose):
    """Habitat camera-to-world -> OpenCV world-to-camera.

    Habitat cameras look along -Z with +Y up; OpenFrontier backprojects
    along +Z with +Y down. Sensor pose includes camera height and pitch.
    """
    return np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(pose_matrix(camera_pose))


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    robot_pose: np.ndarray
    camera_pose: np.ndarray
    timestamp: float
    depth: np.ndarray | None = None

    def __post_init__(self):
        # Copy before processing: movement and reused camera buffers cannot
        # change data while the worker is processing it.
        for name in ("rgb", "robot_pose", "camera_pose", "depth"):
            value = getattr(self, name)
            if value is not None:
                value = (
                    pose_matrix(value) if "pose" in name else np.array(value, copy=True)
                )
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image")
        if not np.isfinite(self.timestamp):
            raise ValueError("Capture timestamp must be finite")


@dataclass(frozen=True)
class AnalysisResult:
    observation: Observation
    frontiers: list
    marked_rgb: np.ndarray
    depth: np.ndarray | None = None
    frontier_analysis: bool = True


class ExplorationPipeline:
    def __init__(self, frontier_extractor, depth_estimator=None):
        self.frontier_extractor = frontier_extractor
        self.depth_estimator = depth_estimator

    def estimate_depth(self, observation):
        depth = (
            self.depth_estimator(observation.rgb)
            if self.depth_estimator is not None
            else observation.depth
        )
        if depth is None:
            raise ValueError("Supply captured depth or a depth estimator")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != observation.rgb.shape[:2]:
            raise ValueError("Depth and RGB resolutions must match")
        return np.where(np.isfinite(depth) & (depth > 0), depth, 0)

    def process_depth(self, observation):
        return AnalysisResult(observation, [], observation.rgb,
                              self.estimate_depth(observation), frontier_analysis=False)

    def process(self, observation):
        depth = self.estimate_depth(observation)
        # Captured HM3D depth uses zero for unobserved image patches. Keep
        # frontiers anchored in observed pixels only; estimated depth has no
        # corresponding ground-truth coverage mask.
        if observation.depth is not None:
            invalid = (depth == 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            invalid = cv2.dilate(invalid, kernel)
            mask = (invalid == 0).astype(np.uint8)
            # If any pixel in a column is invalid, exclude the whole column.
            mask[:, np.any(mask == 0, axis=0)] = 0
        else:
            mask = None
        result = self.frontier_extractor(
            rgb_input=observation.rgb,
            depth=depth,
            extrinsic_camera_pos=opencv_extrinsic(observation.camera_pose),
            mask=mask,
        )
        return ([], depth) if result is None else (result, depth)

    def process_with_visualization(self, observation)->AnalysisResult:
        frontiers, depth = self.process(observation)
        visualize = getattr(self.frontier_extractor, "visualize", None)
        marked = visualize(frontiers) if visualize is not None else observation.rgb
        # Own the result before another inference can mutate detector buffers.
        marked = np.array(marked, dtype=np.uint8, copy=True)
        marked.setflags(write=False)
        return AnalysisResult(observation, frontiers, marked, depth)


class AnalysisTrigger:
    def __init__(self, mode="manual", interval=1.0, distance=0.5, angle_degrees=15.0):
        if mode not in {"manual", "periodic", "movement"}:
            raise ValueError(f"Unknown trigger mode: {mode}")
        if not all(
            np.isfinite(v) and v > 0 for v in (interval, distance, angle_degrees)
        ):
            raise ValueError(
                "Trigger interval, distance, and angle must be finite and positive"
            )
        self.mode = mode
        self.interval = interval
        self.distance = distance
        self.angle = np.deg2rad(angle_degrees)
        self.last_time = None
        self.last_pose = None

    def due(self, observation, now, manual=False):
        if manual:
            return True
        if self.mode == "manual":
            return False
        if self.last_time is None:
            return True
        if now - self.last_time < self.interval:
            return False
        if self.mode == "periodic":
            return True
        current = observation.camera_pose
        distance = np.linalg.norm(current[:3, 3] - self.last_pose[:3, 3])
        relative = self.last_pose[:3, :3].T @ current[:3, :3]
        angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
        return distance >= self.distance or angle >= self.angle

    def mark_submitted(self, observation, now):
        self.last_time = now
        self.last_pose = observation.camera_pose.copy()


class ExplorationWorker:
    """One in-flight capture, no stale inference queue. Graph stays on UI thread."""

    def __init__(self, pipeline:ExplorationPipeline, graph: ExplorationGraph, trigger: AnalysisTrigger):
        self.pipeline, self.graph, self.trigger = pipeline, graph, trigger
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frontier")
        self.future = None
        self.observation = None
        self.frame_index = None
        self.last_result = None

    @property
    def busy(self):
        return self.future is not None

    def submit(self, observation, now, manual=False, frame_index=None, depth_only=False):
        if self.busy or not self.trigger.due(observation, now, manual):
            return False
        self.observation, self.frame_index = observation, frame_index
        self.future = self.executor.submit(
            self.pipeline.process_depth if depth_only else self.pipeline.process_with_visualization,
            observation
        )
        self.trigger.mark_submitted(observation, now)
        return True

    def poll(self)->tuple[int | None, AnalysisResult | None]:
        if self.future is None or not self.future.done():
            return None, None
        future, observation, frame_index = (
            self.future, self.observation, self.frame_index
        )
        self.future = self.observation = self.frame_index = None
        result: AnalysisResult = future.result()  # Caller reports errors; no partial pose.
        pose_id = self.graph.add_frontiers_graph(
            result.frontiers,
            observation.robot_pose,
            observation.timestamp,
            observation.camera_pose,
            frame_index,
            depth=result.depth,
        )
        self.last_result = result
        return pose_id, result

    def close(self):
        self.executor.shutdown(wait=True)
        return self.poll()
