"""Archived RGB views and a weighted pose graph for LoTIS routes."""

import heapq
from itertools import product

import numpy as np


def camera_forward(state):
    """Return the RGB camera's world-frame forward unit vector."""
    import quaternion

    sensor = getattr(state, 'sensor_states', {}).get('rgb')
    rotation = sensor.rotation if sensor is not None else state.rotation
    matrix = quaternion.as_rotation_matrix(rotation)
    forward = matrix @ np.array([0.0, 0.0, -1.0])
    length = np.linalg.norm(forward)
    return forward / length if length > 0 else forward


class RecordedRoute:
    def __init__(self, connect_distance=0.5, heading_weight=0.5):
        if not np.isfinite(connect_distance) or connect_distance <= 0:
            raise ValueError('connect_distance must be finite and positive')
        if not np.isfinite(heading_weight) or heading_weight < 0:
            raise ValueError('heading_weight must be finite and non-negative')
        self.frames = []
        self.states = []
        self.frame_metadata = []
        self.lotis_encodings = []
        # Simulator body poses per frame (4x4 or None), for evaluation only.
        self.gt_poses = []
        self.active = []
        self.parents = []
        self.edges = []
        self.adjacency = []
        self.connect_distance = float(connect_distance)
        self.heading_weight = float(heading_weight)
        self.spatial_index = {}
        self.removed_branches = []
        self.pending_return = False

    def _cell(self, position):
        return tuple(np.floor(np.asarray(position, dtype=float) / self.connect_distance).astype(int))

    def _nearby(self, position):
        cell = self._cell(position)
        for offset in product((-1, 0, 1), repeat=3):
            neighbor_cell = tuple(axis + delta for axis, delta in zip(cell, offset))
            yield from self.spatial_index.get(neighbor_cell, ())

    def _add_edge(self, first, second):
        if first == second or second in self.adjacency[first]:
            return
        first_position = np.asarray(self.states[first].position, dtype=float)
        second_position = np.asarray(self.states[second].position, dtype=float)
        distance = float(np.linalg.norm(second_position - first_position))
        first_forward = camera_forward(self.states[first])
        second_forward = camera_forward(self.states[second])
        cosine = float(np.clip(np.dot(first_forward, second_forward), -1.0, 1.0))
        view_angle = float(np.arccos(cosine))
        cost = distance + self.heading_weight * view_angle ** 2
        edge = dict(first=first, second=second, distance=distance,
                    view_angle=view_angle, cost=cost)
        self.edges.append(edge)
        self.adjacency[first][second] = edge
        self.adjacency[second][first] = edge

    def append(self, frame, state, gt_pose=None, *, timestamp=None, robot_pose=None,
               camera_pose=None, gt_camera_pose=None):
        import copy

        position = np.asarray(state.position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Route state position must be a finite 3D point')
        index = len(self.frames)
        self.frames.append(np.array(frame, copy=True))
        self.states.append(copy.deepcopy(state))
        self.gt_poses.append(None if gt_pose is None else np.array(gt_pose, copy=True))
        self.frame_metadata.append({
            'timestamp': None if timestamp is None else float(timestamp),
            'robot_pose': None if robot_pose is None else np.array(robot_pose, dtype=float, copy=True).tolist(),
            'camera_pose': None if camera_pose is None else np.array(camera_pose, dtype=float, copy=True).tolist(),
            'ground_truth_robot_pose': None if gt_pose is None else np.array(gt_pose, dtype=float, copy=True).tolist(),
            'ground_truth_camera_pose': (None if gt_camera_pose is None else
                                         np.array(gt_camera_pose, dtype=float, copy=True).tolist()),
        })
        self.parents.append(self.active[-1] if self.active else None)
        self.adjacency.append({})

        nearby = set(self._nearby(position))
        if self.active:
            # Consecutive captures record a traversed segment, even if the robot
            # moved farther than the loop-closure threshold between captures.
            self._add_edge(self.active[-1], index)
        for neighbor in nearby:
            if np.linalg.norm(position - np.asarray(self.states[neighbor].position)) <= self.connect_distance:
                self._add_edge(neighbor, index)
        self.spatial_index.setdefault(self._cell(position), []).append(index)
        self.active.append(index)
        return index

    @property
    def available_frames(self):
        return set(range(len(self.frames)))

    def path_to(self, frame_index):
        if not self.active:
            return [frame_index]
        return self.shortest_path(self.active[0], frame_index)

    def shortest_path(self, start, goal):
        """Return the minimum weighted route between archived frame IDs."""
        if not 0 <= start < len(self.frames) or not 0 <= goal < len(self.frames):
            raise ValueError('Route endpoints must be archived frame IDs')
        costs = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            cost, current = heapq.heappop(queue)
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return path[::-1]
            if cost != costs.get(current):
                continue
            for neighbor, edge in self.adjacency[current].items():
                candidate = cost + edge['cost']
                if candidate < costs.get(neighbor, np.inf):
                    costs[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        return []

    def connection(self, frame_index):
        """Return the current frame and shortest graph path to an archived view."""
        if not self.active:
            raise ValueError('Cannot connect a target before recording a route')
        path = self.shortest_path(self.active[-1], frame_index)
        if not path:
            raise ValueError(f'No route connects the current frame to {frame_index}')
        return path[0], path

    def backtrack(self, frame_index):
        start = self.active.index(frame_index)
        return [self.frames[index] for index in self.active[start:]]

    def close_loop(self, frame_index, reason):
        retained = self.path_to(frame_index)
        if not retained:
            raise ValueError(f'No route from the run start to frame {frame_index}')
        removed = [index for index in self.active if index not in retained]
        if removed:
            self.removed_branches.append({'frame_ids': removed, 'reason': reason,
                                          'retrieval_eligible': True})
        self.active = retained
        return removed

    def to_dict(self):
        return {'retrieval_ready': not self.pending_return,
                'active_frame_ids': self.active,
                'removed_branches': self.removed_branches,
                'archive_frame_count': len(self.frames),
                'archive_parent_ids': self.parents,
                'frame_metadata': self.frame_metadata,
                'lotis_encodings': self.lotis_encodings,
                'route_edges': self.edges,
                'route_connect_distance': self.connect_distance,
                'route_heading_weight': self.heading_weight}
