"""Persistent ground-truth pose/frontier history, independent of perception models."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def pose_matrix(value):
    pose = np.array(value, dtype=float, copy=True)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be a finite 4x4 local-to-world matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Pose must have homogeneous last row [0, 0, 0, 1]")
    return pose


@dataclass
class PoseNode:
    id: int
    pose: list
    parent_id: int | None
    timestamp: float
    distance: float
    camera_pose: list | None = None
    frame_index: int | None = None  # recorded trajectory frame taken at this pose


@dataclass
class FrontierNode:
    id: int
    position: list
    observations: list = field(default_factory=list)


class ExplorationGraph:
    """Distances are metres in Habitat's world frame (Y up).

    Each processed capture adds a pose, even when it contains no frontiers.
    Frontier representatives stay at their first position; every subsequent
    measurement is retained, avoiding drift from repeated duplicate merges.
    """

    def __init__(self, merge_distance=0.5):
        if not np.isfinite(merge_distance) or merge_distance < 0:
            raise ValueError("merge_distance must be finite and nonnegative")
        self.merge_distance = float(merge_distance)
        self.poses: dict[int, PoseNode] = {}
        self.frontiers: dict[int, FrontierNode] = {}

    def add_pose(
        self, node_id, pose, parent_id, timestamp, camera_pose=None, frame_index=None
    ):
        if node_id in self.poses:
            raise ValueError(f"Pose ID {node_id} already exists")
        if parent_id is not None and parent_id not in self.poses:
            raise ValueError(f"Unknown parent pose {parent_id}")
        pose = pose_matrix(pose)
        camera_pose = None if camera_pose is None else pose_matrix(camera_pose).tolist()
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        distance = 0.0
        if parent_id is not None:
            parent = np.asarray(self.poses[parent_id].pose)
            distance = float(np.linalg.norm(pose[:3, 3] - parent[:3, 3]))
        node = PoseNode(
            node_id,
            pose.tolist(),
            parent_id,
            float(timestamp),
            distance,
            camera_pose,
            frame_index,
        )
        self.poses[node_id] = node
        return node

    def add_frontier(self, frontier, pose_id):
        """Consume OpenFrontier's world-frame ``pos3d``; return its graph ID."""
        if pose_id not in self.poses:
            raise ValueError(f"Unknown observation pose {pose_id}")
        position = np.asarray(frontier.pos3d, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Frontier position must be a finite xyz vector")
        nearest = None
        if self.frontiers:
            nearest = min(
                self.frontiers.values(),
                key=lambda n: np.linalg.norm(position - n.position),
            )
            if np.linalg.norm(position - nearest.position) > self.merge_distance:
                nearest = None
        if nearest is None:
            node_id = max(self.frontiers, default=-1) + 1
            nearest = FrontierNode(node_id, position.tolist())
            self.frontiers[node_id] = nearest
        gain = getattr(frontier, "gain", None)
        nearest.observations.append(
            {
                "pose_id": pose_id,
                "timestamp": self.poses[pose_id].timestamp,
                "position": position.tolist(),
                "gain": float(gain) if gain is not None and np.isfinite(gain) else None,
            }
        )
        return nearest.id

    def add_frontiers_graph(
        self, frontiers, robot_pose, timestamp, camera_pose=None, frame_index=None
    ):
        """Append a capture and link it to the last processed capture."""
        parent_id = next(reversed(self.poses), None)
        pose_id = max(self.poses, default=-1) + 1
        self.add_pose(
            pose_id, robot_pose, parent_id, timestamp, camera_pose, frame_index
        )
        for frontier in frontiers if frontiers is not None else []:
            # Invalid network predictions must not poison spatial matching/JSON.
            position = np.asarray(frontier.pos3d, dtype=float)
            if position.shape == (3,) and np.isfinite(position).all():
                self.add_frontier(frontier, pose_id)
        return pose_id

    # Names used in the architecture discussion.
    addPose = add_pose
    addFrontier = add_frontier
    AddFrontiersGraph = add_frontiers_graph

    def to_dict(self):
        return {
            "schema_version": 1,
            "world_frame": "Habitat: X right, Y up, camera forward -Z; metres",
            "merge_distance": self.merge_distance,
            "poses": [asdict(node) for node in self.poses.values()],
            "frontiers": [asdict(node) for node in self.frontiers.values()],
        }

    def save(self, path):
        """Atomically replace the snapshot, preserving all processed history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["schema_version"] != 1:
            raise ValueError("Unsupported exploration graph schema")
        graph = cls(data["merge_distance"])
        for node in data["poses"]:
            graph.add_pose(
                node["id"],
                node["pose"],
                node["parent_id"],
                node["timestamp"],
                node["camera_pose"],
                node.get("frame_index"),
            )
        graph.frontiers = {
            node["id"]: FrontierNode(**node) for node in data["frontiers"]
        }
        return graph
