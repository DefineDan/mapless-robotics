"""Frontier memory and loop-pruned RGB routes for autonomous exploration.

Frame IDs are immutable archive indices. Only ``active`` is shortened after a
successful return; LoTIS can still use the abandoned branch to get home.
"""

from dataclasses import asdict

import numpy as np

from .exploration_graph import ExplorationGraph, _json_value


class FrontierMemoryGraph(ExplorationGraph):
    """Keep alternatives until explicitly reached or rejected by the operator.

    Selection uses peak observed gain (the reduced detector's available score),
    independent of distance. Ordinary graph utility decay would erase historical
    alternatives, so this launcher handles arrivals itself.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retired_frontiers = []
        self.excluded_pose_ids = set()
        self._next_frontier_id = 0
        self.path_parent_id = None
        self.objective_check = {}

    def _apply_pose_visit(self, pose):
        pass

    def eval_objective(self, pose, look_ahead_vector):
        return self.objective_frontier

    def create_frontier(self, frontier, pose_id):
        node = super().create_frontier(frontier, pose_id)
        node.id = self._next_frontier_id
        self._next_frontier_id += 1
        return node

    def add_frontier(self, frontier, pose_id):
        position = np.asarray(frontier.pos3d, dtype=float)
        if any(np.linalg.norm(position - item['frontier']['position'])
               <= max(self.merge_distance, self.visited_distance)
               for item in self.retired_frontiers):
            return None
        # An objective stays stable, but repeated observations should update its
        # score/history rather than create a second target at the same location.
        objective = self.objective_frontier
        if objective is not None and np.linalg.norm(position - objective.position) <= self.merge_distance:
            node = self.create_frontier(frontier, pose_id)
            node.observations[0]['valid'] = node.valid_flag
            objective.observations.extend(node.observations)
            objective.gain = node.gain
            objective.u_gain = node.u_gain
            return objective.id
        return super().add_frontier(frontier, pose_id)

    def add_frontiers_graph(self, *args, **kwargs):
        pose_id = super().add_frontiers_graph(*args, **kwargs)
        node = self.poses[pose_id]
        node.parent_id = self.path_parent_id
        node.distance = (0.0 if node.parent_id is None else float(np.linalg.norm(
            node.pose[:3, 3] - self.poses[node.parent_id].pose[:3, 3])))
        self.path_parent_id = pose_id
        return pose_id

    @staticmethod
    def score(frontier):
        values = [frontier.gain, frontier.u_gain]
        values.extend(obs.get('gain') for obs in frontier.observations)
        values = [float(value) for value in values
                  if value is not None and np.isfinite(value)]
        return max(values, default=-np.inf)

    def best_saved(self, available_frames, pose_id=None):
        candidates = [node for node in self.frontiers.values()
                      if node.valid_flag and np.isfinite(self.score(node)) and any(
                          self.poses[obs['pose_id']].frame_index in available_frames
                          and (pose_id is None or obs['pose_id'] == pose_id)
                          and obs.get('valid', True)
                          and (pose_id is None or (obs.get('gain') is not None
                               and np.isfinite(obs['gain']) and obs['gain'] >= self.minimum_gain))
                          for obs in node.observations)]
        def rank(node):
            gain = self.score(node) if pose_id is None else max(
                float(obs['gain']) for obs in node.observations
                if obs['pose_id'] == pose_id and obs.get('gain') is not None
                and np.isfinite(obs['gain']) and obs.get('valid', True))
            return gain, -node.id
        return max(candidates, key=rank, default=None)

    def select_saved(self, available_frames, preferred_pose_id=None):
        selected = (self.best_saved(available_frames, preferred_pose_id)
                    if preferred_pose_id is not None else None)
        selected = selected or self.best_saved(available_frames)
        for node in self.frontiers.values():
            node.is_objective = node is selected
        self.objective_frontier = selected
        return selected

    def retire(self, frontier, reason):
        frontier.is_objective = False
        self.retired_frontiers.append({'reason': reason, 'frontier': asdict(frontier)})
        self.eliminate_frontier(frontier.id)

    def observer(self, frontier, available_frames, current_position=None):
        # Retrieval pruning does not destroy the archived route to a viewpoint.
        poses = [self.poses[obs['pose_id']] for obs in frontier.observations
                 if self.poses[obs['pose_id']].frame_index in available_frames]
        if not poses:
            raise ValueError(f'No archived route to F{frontier.id}')
        reference = frontier.position if current_position is None else np.asarray(current_position)
        return min(poses, key=lambda pose: (np.linalg.norm(
            pose.pose[:3, 3][self.plane_idx] - reference[self.plane_idx]), -pose.id))

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
                   and np.linalg.norm(np.asarray(frontier.pos3d) - objective.position)
                   <= max(self.merge_distance, self.visited_distance)]
        match = min(matches, key=lambda frontier: np.linalg.norm(
            np.asarray(frontier.pos3d) - objective.position), default=None)
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
        retained = set(active_frames)
        self.excluded_pose_ids = {
            pose.id for pose in self.poses.values() if pose.frame_index not in retained}
        self.path_parent_id = (target_pose.id if target_pose is not None else next(
            (pose.id for pose in reversed(list(self.poses.values()))
             if pose.frame_index in retained), None))

    def to_dict(self):
        data = super().to_dict()
        data['pipeline_memory'] = _json_value({
            'retired_frontiers': self.retired_frontiers,
            'excluded_pose_ids': sorted(self.excluded_pose_ids),
            'score': 'peak observed gain',
            'objective_check': self.objective_check,
        })
        return data


class RecordedRoute:
    def __init__(self):
        self.frames = []
        self.states = []
        self.active = []
        self.parents = []
        self.removed_branches = []
        self.pending_return = False

    def append(self, frame, state):
        import copy

        index = len(self.frames)
        self.frames.append(np.array(frame, copy=True))
        self.states.append(copy.deepcopy(state))
        self.parents.append(self.active[-1] if self.active else None)
        self.active.append(index)
        return index

    @property
    def available_frames(self):
        return set(range(len(self.frames)))

    def path_to(self, frame_index):
        path = []
        while frame_index is not None:
            path.append(frame_index)
            frame_index = self.parents[frame_index]
        return path[::-1]

    def connection(self, frame_index):
        """Return the shared viewpoint and forward leg to an archived branch."""
        path = self.path_to(frame_index)
        common = next(index for index in reversed(path) if index in self.active)
        return common, path[path.index(common):]

    def backtrack(self, frame_index):
        start = self.active.index(frame_index)
        return [self.frames[index] for index in self.active[start:]]

    def close_loop(self, frame_index, reason):
        retained = self.path_to(frame_index)
        removed = [index for index in self.active if index not in retained]
        if removed:
            self.removed_branches.append({'frame_ids': removed, 'reason': reason,
                                          'retrieval_eligible': False})
        self.active = retained
        return removed

    def to_dict(self):
        return {'retrieval_ready': not self.pending_return,
                'active_frame_ids': self.active,
                'removed_branches': self.removed_branches,
                'archive_frame_count': len(self.frames),
                'archive_parent_ids': self.parents}
