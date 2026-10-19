"""Persistent ground-truth pose/frontier history, independent of perception models."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .exploration_map import ExplorationMap

DIRECTIONAL_REWARD = 10.0


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
        visited_distance=0.1,
        u_gain_reduction_percentage=1.0,
        u_gain_threshold=0.1,
        enable_occupancy=True,
        occupancy_ini_depth=0.1,
        occupancy_end_depth=1.0,
        occupancy_resolution=0.05,
        camera_intrinsics=None,
        val_depth_width=10,
        depth_validity_threshold=0.5,
        auto_select_objective=True,
        compare_frontiers_2d=True,
        occupancy_report_every=0,
        enable_ground_truth_occupancy=False,
        ground_truth_occupancy_report_every=0,
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
        # Retired-frontier suppression can compare in the ground plane or in
        # full xyz. Keep the existing 3D behavior unless explicitly enabled.
        self.comp_axis_front = (
            np.array([0, 1, 2])[self.plane_idx]
            if compare_frontiers_2d else np.array([0, 1, 2])
        )
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
            ExplorationMap(occupancy_ini_depth, occupancy_end_depth, occupancy_resolution,
                           report_every=occupancy_report_every)
            if self.enable_occupancy else None
        )
        self.gt_ena_map = (
            ExplorationMap(
                occupancy_ini_depth,
                occupancy_end_depth,
                occupancy_resolution,
                report_every=ground_truth_occupancy_report_every,
                report_name="Ground-truth exploration map",
            )
            if self.enable_occupancy and enable_ground_truth_occupancy else None
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
        # The automatic launcher waits for a fresh capture before choosing a
        # goal; visitation, fusion, and arrival evaluation still use this graph.
        self.auto_select_objective = bool(auto_select_objective)
        self.retired_frontiers = []
        self.excluded_pose_ids = set()
        self._next_frontier_id = 0
        self.path_parent_id = None
        self.objective_check = {}

    def add_pose(
        self, node_id, pose, parent_id, timestamp, camera_pose=None, frame_index=None,
        gt_camera_pose=None,
    ):
        if node_id in self.poses:
            raise ValueError(f"Pose ID {node_id} already exists")
        if parent_id is not None and parent_id not in self.poses:
            raise ValueError(f"Unknown parent pose {parent_id}")
        pose = pose_matrix(pose)
        camera_pose = None if camera_pose is None else pose_matrix(camera_pose)
        gt_camera_pose = None if gt_camera_pose is None else pose_matrix(gt_camera_pose)
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
        self.path_parent_id = node_id
        if self.ena_map is not None:
            camera_view = camera_pose if camera_pose is not None else pose
            self.ena_map.add_pose(camera_view, self.camera_intrinsics)
        if self.gt_ena_map is not None and gt_camera_pose is not None:
            self.gt_ena_map.add_pose(gt_camera_pose, self.camera_intrinsics)
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
                self.retire(frontier, 'visited')

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
        node_id = self._next_frontier_id
        self._next_frontier_id += 1
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
                if self.ena_map.is_occupied(frontier.pose6d[[self.idx_upward], 3])[0]:
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
        self.retire(objective, 'unreachable')
        return objective

    def add_frontier(self, frontier, pose_id):
        """Consume OpenFrontier's world-frame ``pos3d``; return its graph ID."""
        if pose_id not in self.poses:
            raise ValueError(f"Unknown observation pose {pose_id}")
        position = np.asarray(frontier.pos3d, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Frontier position must be a finite xyz vector")
        if any(np.linalg.norm((position - np.asarray(item['frontier']['position']))
                              [self.comp_axis_front])
               <= max(self.merge_distance, self.visited_distance)
               for item in self.retired_frontiers):
            return None
        node = self.create_frontier(frontier, pose_id)
        node.observations[0]['valid'] = self.is_frontier_valid(node, pose_id)

        if not node.observations[0]['valid']:
            return None

        # Attach compatible reobservations to the stable goal. Keeping its
        # position and effective utility avoids moving the target or undoing
        # visit decay, and prevents a duplicate from reviving a retired goal.
        objective = self.objective_frontier
        if objective is not None:
            distance, angle = self.calculate_metric_fusion(node, objective)
            if distance <= self.merge_distance and angle <= self.fusion_angle_degrees:
                objective.observations.extend(node.observations)
                return objective.id

        matches = []
        for existing in self.frontiers.values():
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

    def score(self, frontier, pose):
        """Current effective gain divided by one plus ground-plane distance."""
        gain = frontier.u_gain if frontier.u_gain is not None else frontier.gain
        if not frontier.valid_flag or gain is None or not np.isfinite(gain) or gain < 0:
            return -np.inf
        position = self._position_from_pose(pose)
        distance = np.linalg.norm((frontier.position - position)[self.plane_idx])
        return float(gain / (1.0 + distance))

    def _best_frontier(self, pose, available_frames=None):
        frontiers = list(self.frontiers.values())
        if available_frames is not None:
            frontiers = [node for node in frontiers if any(
                self.poses[obs['pose_id']].frame_index in available_frames
                and obs.get('valid', True) for obs in node.observations)]
        if not frontiers:
            return None
        position_cam = self._position_from_pose(pose)
        # Stack frontier positions: (N, 3)
        positions = np.stack([f.position for f in frontiers])

        # u_gain takes precedence over gain
        gains = np.asarray([
            f.u_gain if f.u_gain is not None else f.gain
            for f in frontiers
        ], dtype=float)

        # Valid candidates
        valid = np.isfinite(gains) & (gains >= 0)
        valid &= np.asarray([f.valid_flag for f in frontiers], dtype=bool)
        if not np.any(valid):
            return None

        # Distances in the selected plane
        deltas = positions[:, self.plane_idx] - position_cam[self.plane_idx]
        distances = np.linalg.norm(deltas, axis=1)

        # Prefer frontiers in the direction the camera is facing. Project both
        # vectors onto the ground plane before taking their normalized dot
        # product. A position-only pose (or a vertical/degenerate heading)
        # has no usable direction, so retain distance-only scoring in that case.
        pose_matrix_value = None
        if isinstance(pose, PoseNode):
            pose_matrix_value = pose.camera_pose if pose.camera_pose is not None else pose.pose
        else:
            pose_array = np.asarray(pose, dtype=float)
            if pose_array.shape == (4, 4):
                pose_matrix_value = pose_array

        if pose_matrix_value is not None:
            heading = self._camera_forward(pose_matrix_value)[self.plane_idx]
            heading_norm = np.linalg.norm(heading)
            if np.isfinite(heading_norm) and heading_norm > 0:
                delta_norms = distances
                # A frontier at the robot position has no defined bearing;
                # leave its directional multiplier neutral.
                alignment = np.ones_like(distances)
                np.divide(
                    deltas @ (heading / heading_norm),
                    delta_norms,
                    out=alignment,
                    where=delta_norms > 0,
                )
                # Map cosine similarity from [-1, 1] to a [0, 1] multiplier.
                directional_reward = (1.0 + np.clip(alignment, -1.0, 1.0)) / 2.0
            else:
                directional_reward = 1.0
        else:
            directional_reward = 1.0

        # Score = gain / (1 + distance), weighted by facing alignment.
        scores = gains / (1.0 + distances)
        scores *= (directional_reward * DIRECTIONAL_REWARD) 
        scores[~valid] = -np.inf

        # np.argmax gives the first maximum for deterministic tie breaking.
        selected_idx = np.argmax(scores)
        return frontiers[selected_idx]

    def select_frontier(self, pose, available_frames=None):
        """Select by utility and distance, optionally requiring an archived RGB view."""
        selected = self._best_frontier(pose, available_frames)
        for node in self.frontiers.values():
            node.is_objective = node is selected
        self.objective_frontier = selected
        return selected

    def best_saved(self, available_frames, pose):
        """Find a route-backed candidate using ordinary graph utility scoring."""
        return self._best_frontier(pose, available_frames)

    def select_saved(self, available_frames, pose):
        return self.select_frontier(pose, available_frames)

    def eval_objective(self, pose, look_ahead_vector):
        """Return the active objective, replacing it when reached.

        Reaching requires being within ``visited_distance`` and facing the
        objective within ``fusion_angle_degrees``. On arrival the old target is
        retired, then the next target is selected. With ``auto_select_objective``
        disabled, arrival clears the objective so the launcher can refresh its
        view before selecting. An exact positional match also counts as arrival.
        """
        objective = self.objective_frontier
        if objective is None:
            return self.select_frontier(pose) if self.auto_select_objective else None

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
        # A frontier's view direction is the orientation it should be seen
        # from. Compare that with the robot camera's view direction; the
        # bearing to the frontier is unrelated to the frontier's orientation.
        frontier_direction = objective.vd
        if frontier_direction is None and objective.pose6d is not None:
            # OpenFrontier frontier poses use the CV convention (+Z forward).
            frontier_direction = np.asarray(objective.pose6d, dtype=float)[:3, 2]
        if frontier_direction is not None:
            frontier_direction = np.asarray(frontier_direction, dtype=float)
            if frontier_direction.shape == (3,):
                frontier_direction = frontier_direction[self.plane_idx]
        if distance == 0:
            self.retire(objective, 'reached')
            return self.select_frontier(pose) if self.auto_select_objective else None
        if (
            direction_norm == 0
            or not np.isfinite(direction).all()
            or frontier_direction is None
            or frontier_direction.shape != (2,)
            or not np.isfinite(frontier_direction).all()
        ):
            return objective
        frontier_norm = float(np.linalg.norm(frontier_direction))
        if frontier_norm == 0:
            return objective
        cosine = float(np.clip(
            np.dot(direction, frontier_direction) / (direction_norm * frontier_norm),
            -1.0,
            1.0,
        ))
        angle = float(np.degrees(np.arccos(cosine)))
        if distance <= self.visited_distance and angle <= self.fusion_angle_degrees:
            self.retire(objective, 'reached')
            return self.select_frontier(pose) if self.auto_select_objective else None
        return objective

    def add_frontiers_graph(
        self, frontiers, robot_pose, timestamp, camera_pose=None, frame_index=None,
        depth=None, gt_camera_pose=None,
    ):
        """Append a capture, linking to the latest capture or returned-to pose."""
        parent_id = self.path_parent_id
        pose_id = max(self.poses, default=-1) + 1
        self.add_pose(
            pose_id, robot_pose, parent_id, timestamp, camera_pose, frame_index,
            gt_camera_pose,
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

    def retire(self, frontier, reason):
        """Remove an active frontier while preserving its identity and history."""
        if frontier.id not in self.frontiers:
            return
        frontier.is_objective = False
        self.retired_frontiers.append({'reason': reason, 'frontier': asdict(frontier)})
        self.eliminate_frontier(frontier.id)

    def calculate_distance(self, node_a_id, node_b_id):
        """Return route distance in graph steps between two pose nodes.

        Every parent-to-child edge currently costs one step. The graph is a
        tree, so the distance is the number of edges through the common
        ancestor.
        """
        if node_a_id not in self.poses or node_b_id not in self.poses:
            raise ValueError("Both distance endpoints must be known pose nodes")

        def ancestors(node_id):
            result = {}
            distance = 0
            while node_id is not None:
                result[node_id] = distance
                node_id = self.poses[node_id].parent_id
                distance += 1
            return result

        a_ancestors = ancestors(node_a_id)
        b_ancestors = ancestors(node_b_id)
        common = a_ancestors.keys() & b_ancestors.keys()
        if not common:
            raise ValueError("Pose nodes are not connected")
        return min(a_ancestors[node_id] + b_ancestors[node_id] for node_id in common)

    def observer(self, frontier: FrontierNode, available_frames, current_position=None):
        """Find nearest archived viewpoint and its route distance from here."""
        # Retrieval pruning does not destroy the archived route to a viewpoint.
        poses = [self.poses[obs['pose_id']] for obs in frontier.observations
                 if self.poses[obs['pose_id']].frame_index in available_frames
                 and obs.get('valid', True)]
        if not poses:
            raise ValueError(f'No archived route to F{frontier.id}')
        reference = frontier.position if current_position is None else np.asarray(current_position)
        target = min(poses, key=lambda pose: (np.linalg.norm(
            pose.pose[:3, 3][self.plane_idx] - reference[self.plane_idx]), -pose.id))
        if current_position is None or not self.poses:
            distance_node_to_node = 0
        else:
            current_node = min(self.poses.values(), key=lambda pose: np.linalg.norm(
                pose.pose[:3, 3][self.plane_idx] - reference[self.plane_idx]))
            distance_node_to_node = self.calculate_distance(target.id, current_node.id)
        return target, distance_node_to_node

    def revalidate_objective(self, result, confirmations=3):
        """Reject repeated explicit bad matches, never a missing detection alone."""
        objective = self.objective_frontier
        if objective is None or not result.frontier_analysis:
            return None
        if np.linalg.norm((result.observation.robot_pose[:3, 3]
                           - objective.position)[self.plane_idx]) <= self.visited_distance:
            self.objective_check = dict(frontier_id=objective.id, status='at_goal', invalid_count=0)
            return None
        previous = self.objective_check
        count = previous.get('invalid_count', 0) if previous.get('frontier_id') == objective.id else 0
        matches = [frontier for frontier in result.frontiers
                   if np.shape(frontier.pos3d) == (3,)
                   and np.isfinite(frontier.pos3d).all()
                   and np.linalg.norm((np.asarray(frontier.pos3d) - objective.position)
                                      [self.comp_axis_front])
                   <= max(self.merge_distance, self.visited_distance)]
        match = min(matches, key=lambda frontier: np.linalg.norm(
            (np.asarray(frontier.pos3d) - objective.position)[self.comp_axis_front]),
            default=None)
        gain = getattr(match, 'gain', None)
        if match is None:
            status, count = 'not_observed', 0
        elif (not getattr(match, 'is_valid', True) or gain is None
              or not np.isfinite(gain) or gain < self.minimum_gain):
            status, count = 'low_quality', count + 1
        else:
            status, count = 'confirmed', 0
        self.objective_check = dict(frontier_id=objective.id, status=status,
                                    invalid_count=count, latest_gain=(float(gain)
                                    if gain is not None and np.isfinite(gain) else None),
                                    timestamp=result.observation.timestamp)
        if count >= confirmations:
            self.retire(objective, 'invalidated')
            return objective
        return None

    def prune_after_return(self, active_frames, target_pose):
        """Reconnect future captures without deleting archived observation poses."""
        retained = set(active_frames)
        self.excluded_pose_ids = {
            pose.id for pose in self.poses.values() if pose.frame_index not in retained}
        self.path_parent_id = (target_pose.id if target_pose is not None else next(
            (pose.id for pose in reversed(list(self.poses.values()))
             if pose.frame_index in retained), None))

    # Names used in the architecture discussion.
    addPose = add_pose
    addFrontier = add_frontier
    AddFrontiersGraph = add_frontiers_graph

    def to_dict(self):
        return {
            "schema_version": 1,
            "world_frame": "Habitat: X right, Y up, camera forward -Z; metres",
            "merge_distance": self.merge_distance,
            "auto_select_objective": self.auto_select_objective,
            "pipeline_memory": _json_value({
                "retired_frontiers": self.retired_frontiers,
                "excluded_pose_ids": sorted(self.excluded_pose_ids),
                "path_parent_id": self.path_parent_id,
                "next_frontier_id": self._next_frontier_id,
                "score": "effective gain / (1 + ground-plane distance)",
                "objective_check": self.objective_check,
            }),
            "frontier_settings": {
                "minimum_gain": self.minimum_gain,
                "maximum_parent_distance": self.maximum_parent_distance,
                "max_vectical_vec": self.max_vectical_vec,
                "upward_pose": self.upward_pose,
                "fusion_angle_degrees": self.fusion_angle_degrees,
                "visited_distance": self.visited_distance,
                "u_gain_reduction_percentage": self.u_gain_reduction_percentage,
                "u_gain_threshold": self.u_gain_threshold,
                "enable_occupancy": self.enable_occupancy,
                "occupancy_ini_depth": self.ena_map.ini_depth if self.ena_map else 0.1,
                "occupancy_end_depth": self.ena_map.end_depth if self.ena_map else 1.0,
                "occupancy_resolution": self.ena_map.resolution if self.ena_map else 0.05,
                "camera_intrinsics": _json_value(self.camera_intrinsics),
                "val_depth_width": self.val_depth_width,
                "depth_validity_threshold": self.depth_validity_threshold,
                "compare_frontiers_2d": len(self.comp_axis_front) == 2,
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
        graph = cls(data["merge_distance"], **data.get("frontier_settings", {}),
                    auto_select_objective=data.get("auto_select_objective", True))
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
        memory = data.get("pipeline_memory", {})
        graph.retired_frontiers = memory.get("retired_frontiers", [])
        graph.excluded_pose_ids = set(memory.get("excluded_pose_ids", []))
        graph.objective_check = memory.get("objective_check", {})
        graph.path_parent_id = memory.get("path_parent_id", graph.path_parent_id)
        known_ids = [*graph.frontiers, *(item['frontier']['id']
                     for item in graph.retired_frontiers)]
        graph._next_frontier_id = max(memory.get("next_frontier_id", 0),
                                      max(known_ids, default=-1) + 1)
        return graph
