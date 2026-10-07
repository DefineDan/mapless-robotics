"""Persistent ground-truth pose/frontier history, independent of perception models."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .exploration_map import ExplorationMap


def _json_value(value):
    """Recursively convert numpy values when exporting the graph as JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


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
    pose: np.ndarray
    parent_id: int | None
    timestamp: float
    distance: float
    camera_pose: np.ndarray | None = None
    frame_index: int | None = None  # recorded trajectory frame taken at this pose

    def __post_init__(self):
        self.pose = pose_matrix(self.pose)
        if self.camera_pose is not None:
            self.camera_pose = pose_matrix(self.camera_pose)


@dataclass
class FrontierNode:
    id: int
    position: np.ndarray
    observations: list = field(default_factory=list)
    gain: float | None = None
    label: str | None = None
    probability: float = 0.5
    justification: str = "Not analysed"
    direct_angle: float | np.ndarray | None = None
    pixel_pos: np.ndarray | None = None
    vd: np.ndarray | None = None
    valid_flag: bool = True
    u_gain: float | None = None
    utility: float | None = None
    parent_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    pose6d: np.ndarray | None = None
    is_objective: bool = False

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.pixel_pos is not None:
            self.pixel_pos = np.asarray(self.pixel_pos, dtype=float)
        if self.vd is not None:
            self.vd = np.asarray(self.vd, dtype=float)
        if self.pose6d is not None:
            self.pose6d = np.asarray(self.pose6d, dtype=float)
        self.parent_ids = np.asarray(self.parent_ids, dtype=int)
        if self.direct_angle is not None and not np.isscalar(self.direct_angle):
            self.direct_angle = np.asarray(self.direct_angle, dtype=float)


class ExplorationGraph:
    """Distances are metres in Habitat's world frame (Y up).

    Each processed capture adds a pose, even when it contains no frontiers.
    Fused frontier representatives use the mean detected position; every
    measurement is retained in observations.
    """

    def __init__(
        self,
        merge_distance=0.5,
        minimum_gain=0.5,
        maximum_parent_distance=7.0,
        max_vectical_vec=0.65,
        upward_pose="y",
        fusion_angle_degrees=30.0,
        visited_distance=0.5,
        u_gain_reduction_percentage=1.0,
        u_gain_threshold=0.1,
        enable_occupancy=True,
        occupancy_ini_depth=0.1,
        occupancy_end_depth=1.0,
        occupancy_resolution=0.05,
        camera_intrinsics=None,
        val_depth_width=10,
        depth_validity_threshold=0.5,
    ):
        if not np.isfinite(merge_distance) or merge_distance < 0:
            raise ValueError("merge_distance must be finite and nonnegative")
        if not 0 <= u_gain_reduction_percentage <= 1:
            raise ValueError("u_gain_reduction_percentage must be between 0 and 1")
        if min(
            minimum_gain,
            maximum_parent_distance,
            fusion_angle_degrees,
            visited_distance,
            u_gain_threshold,
        ) < 0:
            raise ValueError("frontier thresholds and distances must be nonnegative")
        self.merge_distance = float(merge_distance)
        self.minimum_gain = float(minimum_gain)
        self.maximum_parent_distance = float(maximum_parent_distance)
        axis_indices = {"x": 0, "y": 1, "z": 2}
        if upward_pose not in axis_indices:
            raise ValueError("upward_pose must be one of 'x', 'y', or 'z'")
        self.upward_pose = upward_pose
        self.idx_upward = axis_indices[upward_pose]
        self.plane_idx = np.ones(3, dtype=bool)
        self.plane_idx[self.idx_upward] = False
        self.max_vectical_vec = (
            None if max_vectical_vec is None else float(max_vectical_vec)
        )
        self.fusion_angle_degrees = float(fusion_angle_degrees)
        self.visited_distance = float(visited_distance)
        self.u_gain_reduction_percentage = float(u_gain_reduction_percentage)
        self.u_gain_threshold = float(u_gain_threshold)
        if val_depth_width < 1:
            raise ValueError("val_depth_width must be positive")
        if not np.isfinite(depth_validity_threshold) or depth_validity_threshold < 0:
            raise ValueError("depth_validity_threshold must be finite and nonnegative")
        self.val_depth_width = int(val_depth_width)
        self.depth_validity_threshold = float(depth_validity_threshold)
        self._depth_images: dict[int, np.ndarray] = {}
        self.enable_occupancy = bool(enable_occupancy)
        self.ena_map = (
            ExplorationMap(occupancy_ini_depth, occupancy_end_depth, occupancy_resolution)
            if self.enable_occupancy else None
        )
        # Runtime camera: 640x480 with Habitat's 90 degree horizontal FOV by default.
        self.camera_intrinsics = (
            np.asarray(camera_intrinsics, dtype=float)
            if camera_intrinsics is not None
            else np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
        )
        self.poses: dict[int, PoseNode] = {}
        self.frontiers: dict[int, FrontierNode] = {}
        self.objective_frontier: FrontierNode | None = None

    def add_pose(
        self, node_id, pose, parent_id, timestamp, camera_pose=None, frame_index=None
    ):
        if node_id in self.poses:
            raise ValueError(f"Pose ID {node_id} already exists")
        if parent_id is not None and parent_id not in self.poses:
            raise ValueError(f"Unknown parent pose {parent_id}")
        pose = pose_matrix(pose)
        camera_pose = None if camera_pose is None else pose_matrix(camera_pose)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        distance = 0.0
        if parent_id is not None:
            parent = np.asarray(self.poses[parent_id].pose)
            distance = float(np.linalg.norm(pose[:3, 3] - parent[:3, 3]))
        node = PoseNode(
            node_id,
            pose,
            parent_id,
            float(timestamp),
            distance,
            camera_pose,
            frame_index,
        )
        self.poses[node_id] = node
        if self.ena_map is not None:
            camera_view = camera_pose if camera_pose is not None else pose
            self.ena_map.add_pose(camera_view, self.camera_intrinsics)
        self._apply_pose_visit(node)
        return node

    def _apply_pose_visit(self, pose):
        """Reduce utility for frontiers approached by this new robot pose."""
        robot_position = np.asarray(pose.pose, dtype=float)[:3, 3]

        frontiers = list(self.frontiers.values())
        if not frontiers:
            return

        positions = np.asarray([f.position for f in frontiers], dtype=float)

        if self.ena_map is not None:
            visited = self.ena_map.is_occupied(positions[:, self.plane_idx])
        else:
            visited = self.test_visit(robot_position, None, positions)

        if not np.any(visited):
            return

        visited_frontiers = [f for f, v in zip(frontiers, visited) if v]

        # gain = u_gain if available, otherwise gain
        gains = np.asarray([
            f.u_gain if f.u_gain is not None else f.gain
            for f in visited_frontiers
        ], dtype=float)

        gains = np.nan_to_num(gains, nan=0.0)
        new_u_gains = gains * (1.0 - self.u_gain_reduction_percentage)

        # Update
        for frontier, u_gain in zip(visited_frontiers, new_u_gains):
            frontier.u_gain = float(u_gain)

        # Eliminate below threshold
        for frontier in visited_frontiers:
            if not frontier.is_objective and frontier.u_gain < self.u_gain_threshold:
                self.eliminate_frontier(frontier.id)

    def test_visit(
        self,
        ego_position,
        ego_orientation_vec,
        positions,
        orientations_vec=None,
    ):
        """Return a vector indicating which poses are within visit range.

        Positions have shape ``(N, 3)``. Optional pointing vectors have the
        same shape and additionally require the candidate and ego orientations
        to be within ``fusion_angle_degrees``. A missing orientation skips the
        orientation check.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)[:, self.plane_idx]
        ego_position = np.asarray(ego_position, dtype=float).reshape(3)[self.plane_idx]
        visited = np.linalg.norm(positions - ego_position, axis=1) <= self.visited_distance

        return visited

        #TODO, add something with the orientation
        # if ego_orientation_vec is None or orientations_vec is None:
        #     return visited

        # ego_direction = np.asarray(ego_orientation_vec, dtype=float).reshape(3)
        # directions = np.asarray(orientations_vec, dtype=float).reshape(-1, 3)
        # if len(directions) != len(positions):
        #     raise ValueError("positions and orientations_vec must have matching lengths")
        # ego_norm = np.linalg.norm(ego_direction)
        # direction_norms = np.linalg.norm(directions, axis=1)
        # valid = np.isfinite(ego_direction).all() & np.isfinite(directions).all(axis=1)
        # valid &= (ego_norm > 0) & (direction_norms > 0)
        # cosine = np.zeros(len(directions), dtype=float)
        # np.divide(
        #     directions @ ego_direction,
        #     direction_norms * ego_norm,
        #     out=cosine,
        #     where=valid,
        # )
        # angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        # return visited & valid & (angle <= self.fusion_angle_degrees)

    @staticmethod
    def _camera_forward(pose):
        """Return the camera's world-frame forward vector (local -Z)."""
        return np.asarray(pose, dtype=float)[:3, :3] @ np.array([0.0, 0.0, -1.0])

    def create_frontier(self, frontier, pose_id):
        """Convert one OpenFrontier detection into a graph-owned node."""
        node_id = max(self.frontiers, default=-1) + 1
        position = np.asarray(frontier.pos3d, dtype=float)
        gain = getattr(frontier, "gain", None)
        node = FrontierNode(
            id=node_id,
            position=position.copy(),
            observations=[
                {
                    "pose_id": pose_id,
                    "timestamp": self.poses[pose_id].timestamp,
                    "position": position.copy(),
                    "gain": _json_value(gain),
                }
            ],
            gain=_json_value(gain),
            label=getattr(frontier, "label", None),
            probability=_json_value(getattr(frontier, "probability", 0.5)),
            justification=getattr(frontier, "justification", "Not analysed"),
            direct_angle=getattr(frontier, "direct_angle", None),
            pixel_pos=getattr(frontier, "pixel_pos", None),
            vd=getattr(frontier, "view_direction", None),
            valid_flag=bool(getattr(frontier, "is_valid", True)),
            u_gain=_json_value(getattr(frontier, "u_gain", gain)),
            utility=_json_value(getattr(frontier, "utility", None)),
            parent_ids=getattr(frontier, "parent_ids", []),
            pose6d=getattr(frontier, "pose6d", None),
        )
        return node

    def is_frontier_valid(self, frontier, pose_id):
        """Apply gain, distance, depth-occlusion, and view-direction filters."""
        if not frontier.valid_flag:
            return False
        gain = frontier.u_gain if frontier.u_gain is not None else frontier.gain
        if gain is None or not np.isfinite(gain) or gain < self.minimum_gain:
            return False
        parent_position = np.asarray(self.poses[pose_id].pose, dtype=float)[:3, 3]

        dist_parent = np.linalg.norm(frontier.position - parent_position)
        if dist_parent > self.maximum_parent_distance:
            return False



        if self.max_vectical_vec is not None and (
            frontier.vd is None
            or frontier.vd.shape != (3,)
            or not np.isfinite(frontier.vd).all()
            or abs(float(frontier.vd[self.idx_upward])) > self.max_vectical_vec
        ):
            return False

        # Reject viewpoints that have already been seen (or are close to an
        # earlier camera pose when occupancy mapping is disabled).
        if frontier.pose6d is not None:
            if self.ena_map is not None:
                if self.ena_map.is_occupied(frontier.pose6d[[0, 2], 3])[0]:
                    return False
            else:
                camera_poses = [p.camera_pose for p in self.poses.values() if p.camera_pose is not None]
                if camera_poses:
                    camera_poses = np.asarray(camera_poses, dtype=float)
                    if np.any(self.test_visit(
                        frontier.pose6d[:3, 3],
                        self._camera_forward(frontier.pose6d),
                        camera_poses[:, :3, 3],
                        camera_poses[:, :3, :3] @ np.array([0.0, 0.0, -1.0]),
                    )):
                        return False

        # depth = self._depth_images.get(pose_id)
        # pixel = frontier.pixel_pos
        # camera_pose = self.poses[pose_id].camera_pose
        # if depth is not None and pixel is not None and camera_pose is not None:
        #     depth = np.asarray(depth, dtype=float)
        #     if depth.ndim == 2 and pixel.shape == (2,) and np.isfinite(pixel).all():
        #         height, width = depth.shape
        #         # OpenFrontier reports normalized (x, y) coordinates.
        #         x = int(np.clip(round(float(pixel[0]) * width), 0, width - 1))
        #         y = int(np.clip(round(float(pixel[1]) * height), 0, height - 1))
        #         half_width = self.val_depth_width // 2
        #         x0 = max(0, x - half_width)
        #         x1 = min(width, x + self.val_depth_width - half_width)
        #         nearby_depth = depth[y, x0:x1]
        #         nearby_depth = nearby_depth[np.isfinite(nearby_depth) & (nearby_depth > 0)]
        #         if nearby_depth.size:
        #             camera_position = np.asarray(camera_pose, dtype=float)[:3, 3]
        #             camera_rotation = np.asarray(camera_pose, dtype=float)[:3, :3]
        #             frontier_camera_position = camera_rotation.T @ (
        #                 frontier.position - camera_position
        #             )
        #             frontier_depth = -float(frontier_camera_position[2])
        #             if frontier_depth - float(np.min(nearby_depth)) > self.depth_validity_threshold:
        #                 print(f'It would be occluded, frontier depth: {frontier_depth}, min nearby depth: {np.min(nearby_depth)}')
        #                 # return False

        return True

    @staticmethod
    def calculate_metric_fusion(frontier_a, frontier_b):
        """Return spatial distance and view-direction angle in degrees."""
        distance = float(np.linalg.norm(frontier_a.position - frontier_b.position))
        if frontier_a.vd is None or frontier_b.vd is None:
            return distance, float("inf")
        a = np.asarray(frontier_a.vd, dtype=float)
        b = np.asarray(frontier_b.vd, dtype=float)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0 or not np.isfinite(denominator):
            return distance, float("inf")
        cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
        return distance, float(np.degrees(np.arccos(cosine)))

    def fuse_nodes(self, nodes):
        """Fuse compatible nodes into the lowest-ID representative."""
        nodes = list(nodes)
        if any(node.is_objective for node in nodes) and len(nodes) > 1:
            raise ValueError("Objective frontiers cannot be fused")
        representative = min(nodes, key=lambda node: node.id)
        representative.position = np.mean([node.position for node in nodes], axis=0)
        representative.observations = [
            observation for node in nodes for observation in node.observations
        ]
        for field_name in ("gain", "u_gain", "utility"):
            values = [getattr(node, field_name) for node in nodes]
            values = [value for value in values if value is not None]
            setattr(representative, field_name, max(values) if values else None)
        best = max(
            nodes,
            key=lambda node: node.gain if node.gain is not None else -float("inf"),
        )
        representative.label = best.label
        representative.justification = best.justification
        representative.pose6d = best.pose6d
        representative.probability = float(np.mean([n.probability for n in nodes]))
        representative.valid_flag = all(n.valid_flag for n in nodes)
        representative.parent_ids = np.unique(
            np.concatenate([n.parent_ids for n in nodes])
        ).astype(int)
        directions = [n.vd for n in nodes if n.vd is not None]
        if directions:
            direction = np.mean(directions, axis=0)
            norm = np.linalg.norm(direction)
            representative.vd = direction / norm if norm else None
        angles = [n.direct_angle for n in nodes if n.direct_angle is not None]
        if angles:
            representative.direct_angle = float(
                np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
            )
        pixels = [n.pixel_pos for n in nodes if n.pixel_pos is not None]
        if pixels:
            representative.pixel_pos = np.mean(pixels, axis=0)
        for node in nodes:
            if node is not representative:
                self.eliminate_frontier(node.id)
        return representative

    def eliminate_frontier(self, frontier_id):
        """Remove a frontier from the active graph, if present."""
        frontier = self.frontiers.pop(frontier_id, None)
        if frontier is self.objective_frontier:
            self.objective_frontier = None
        return frontier

    def set_objective_unreachable(self):
        """Clear and remove the currently selected objective frontier."""
        objective = self.objective_frontier
        if objective is None:
            return None
        objective.is_objective = False
        return self.eliminate_frontier(objective.id)

    def add_frontier(self, frontier, pose_id):
        """Consume OpenFrontier's world-frame ``pos3d``; return its graph ID."""
        if pose_id not in self.poses:
            raise ValueError(f"Unknown observation pose {pose_id}")
        position = np.asarray(frontier.pos3d, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Frontier position must be a finite xyz vector")
        node = self.create_frontier(frontier, pose_id)

        if not self.is_frontier_valid(node, pose_id):
            return None

        matches = []
        for existing in self.frontiers.values():
            # Keep the objective identity and target stable until it is evaluated.
            if existing.is_objective or node.is_objective:
                continue
            distance, angle = self.calculate_metric_fusion(node, existing)
            if distance <= self.merge_distance and angle <= self.fusion_angle_degrees:
                matches.append(existing)
        if matches:
            node = self.fuse_nodes([node, *matches])
        self.frontiers[node.id] = node
        return node.id

    @staticmethod
    def _position_from_pose(pose):
        if isinstance(pose, PoseNode):
            pose = pose.pose
        pose = np.asarray(pose, dtype=float)
        if pose.shape == (4, 4):
            return pose[:3, 3]
        if pose.shape == (3,):
            return pose
        raise ValueError("pose must be a PoseNode, 4x4 matrix, or xyz vector")

    def select_frontier(self, pose):
        """Mark and return the best frontier, balancing utility and travel distance."""
        if not self.frontiers:
            return None

        position = self._position_from_pose(pose)

        ids = list(self.frontiers.keys())
        frontiers = list(self.frontiers.values())

        # Stack frontier positions: (N, 3)
        positions = np.stack([f.position for f in frontiers])

        # u_gain takes precedence over gain
        gains = np.asarray([
            f.u_gain if f.u_gain is not None else f.gain
            for f in frontiers
        ], dtype=float)

        # Valid candidates
        valid = np.isfinite(gains) & (gains >= 0)

        if not np.any(valid):
            return None

        # Distances in the selected plane
        deltas = positions[:, self.plane_idx] - position[self.plane_idx]
        distances = np.linalg.norm(deltas, axis=1)

        # Score = gain / (1 + distance)
        scores = gains / (1.0 + distances)

        # Ignore invalid candidates
        scores[~valid] = -np.inf

        # np.argmax gives the first maximum, matching the deterministic behavior
        # of max() when scores and distances are tied.
        selected_idx = np.argmax(scores)
        selected = frontiers[selected_idx]
        selected.is_objective = True
        self.objective_frontier = selected

        return selected

    def eval_objective(self, pose, look_ahead_vector):
        """Return the active objective, replacing it when reached.

        Reaching requires being within ``visited_distance`` and facing the
        objective within ``fusion_angle_degrees``. On arrival the old target is
        removed, then the next target is selected; ``None`` means no target is
        available (or no valid look-ahead direction was provided).
        """
        objective = self.objective_frontier
        if objective is None:
            return self.select_frontier(pose)

        position = self._position_from_pose(pose)
        to_goal = (objective.position - position)[self.plane_idx]
        distance = float(np.linalg.norm(to_goal))
        direction = np.asarray(look_ahead_vector, dtype=float)
        if direction.shape == (4, 4):
            direction = self._camera_forward(direction)
        if direction.shape == (3,):
            direction = direction[self.plane_idx]
        elif direction.shape != (2,):
            raise ValueError("look_ahead_vector must have 2 or 3 components, or be a 4x4 pose")
        direction_norm = float(np.linalg.norm(direction))
        goal_norm = distance
        if direction_norm == 0 or goal_norm == 0 or not np.isfinite(direction).all():
            return objective
        cosine = float(np.clip(np.dot(direction, to_goal) / (direction_norm * goal_norm), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        print(f"Distance to objective: {distance:.3f}, angle to objective: {angle:.3f}")
        if distance <= self.visited_distance and angle <= self.fusion_angle_degrees:
            self.eliminate_frontier(objective.id)
            return self.select_frontier(position)
        return objective

    def add_frontiers_graph(
        self, frontiers, robot_pose, timestamp, camera_pose=None, frame_index=None,
        depth=None,
    ):
        """Append a capture and link it to the last processed capture."""
        parent_id = next(reversed(self.poses), None)
        pose_id = max(self.poses, default=-1) + 1
        self.add_pose(
            pose_id, robot_pose, parent_id, timestamp, camera_pose, frame_index
        )
        if depth is not None:
            depth_image = np.array(depth, dtype=np.float32, copy=True)
            if depth_image.ndim != 2:
                raise ValueError("depth must be a 2D image")
            self._depth_images[pose_id] = depth_image
        for frontier in frontiers if frontiers is not None else []:
            # Invalid network predictions must not poison spatial matching/JSON.
            position = np.asarray(frontier.pos3d, dtype=float)
            if position.shape == (3,) and np.isfinite(position).all():
                self.add_frontier(frontier, pose_id)
        self.eval_objective(
            robot_pose,
            camera_pose if camera_pose is not None else robot_pose,
        )
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
            "frontier_settings": {
                "minimum_gain": self.minimum_gain,
                "maximum_parent_distance": self.maximum_parent_distance,
                "max_vectical_vec": self.max_vectical_vec,
                "upward_pose": self.upward_pose,
                "fusion_angle_degrees": self.fusion_angle_degrees,
                "visited_distance": self.visited_distance,
                "u_gain_reduction_percentage": self.u_gain_reduction_percentage,
                "u_gain_threshold": self.u_gain_threshold,
            },
            "poses": [_json_value(asdict(node)) for node in self.poses.values()],
            "frontiers": [
                _json_value(asdict(node)) for node in self.frontiers.values()
            ],
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
        graph = cls(data["merge_distance"], **data.get("frontier_settings", {}))
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
        graph.objective_frontier = next(
            (
                frontier
                for frontier in graph.frontiers.values()
                if frontier.is_objective
            ),
            None,
        )
        return graph
