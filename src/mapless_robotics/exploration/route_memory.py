"""Archived RGB routes and loop-pruned retrieval paths for LoTIS.

Frame IDs are immutable archive indices. Only ``active`` is shortened after a
successful return; LoTIS can still use the abandoned branch to get home.
Graph updates and route-viewpoint links belong to ``ExplorationGraph``.
"""

import numpy as np


class RecordedRoute:
    def __init__(self):
        self.frames = []
        self.states = []
        # Simulator body poses per frame (4x4 or None), for evaluation only.
        self.gt_poses = []
        self.active = []
        self.parents = []
        self.removed_branches = []
        self.pending_return = False

    def append(self, frame, state, gt_pose=None):
        import copy

        index = len(self.frames)
        self.frames.append(np.array(frame, copy=True))
        self.states.append(copy.deepcopy(state))
        self.gt_poses.append(None if gt_pose is None else np.array(gt_pose, copy=True))
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
