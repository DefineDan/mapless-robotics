"""Grid based global path planners for depth navigation."""

from heapq import heappop, heappush
from itertools import count
import cv2

import numpy as np


class AStarPlanner:
    """Plan an 8-connected path through free cells in an occupancy raster.

    The map is ``(grid, origin)`` as returned by ``DepthNavigation.get_map``;
    occupied cells have nonzero values and all other cells are traversable.
    ``inflation_radius`` expands obstacles by a Euclidean radius in metres.
    Poses are world-frame XYZ points or 4x4 local-to-world transforms.
    """

    def __init__(self, resolution=0.1, plane_indices=(0, 2), inflation_radius=0.0):
        self.resolution = float(resolution)
        self.plane_indices = tuple(plane_indices)
        self.inflation_radius = float(inflation_radius)
        if not np.isfinite(self.inflation_radius) or self.inflation_radius < 0:
            raise ValueError("inflation_radius must be finite and non-negative")
        self.path = np.zeros((0, 2), dtype=float)


    def _inflate(self, grid, cell_size):
        """Mark cells within ``inflation_radius`` of obstacles as occupied."""
        radius_cells = int(np.ceil(self.inflation_radius / cell_size))

        if radius_cells == 0 or not grid.size:
            return grid

        occupied = (grid != 0).astype(np.uint8)

        kernel_size = 2 * radius_cells + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cv2.circle(kernel, (radius_cells, radius_cells), radius_cells, 1, thickness=-1,)

        inflated = cv2.dilate(occupied, kernel)

        result = grid.copy()
        result[inflated != 0] = 255

        return result

    @staticmethod
    def _position(pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape == (4, 4):
            return pose[:3, 3]
        if pose.size in (2, 3):
            return pose.reshape(-1)
        raise ValueError("pose must be a 2D/3D point or a 4x4 transform")

    def _plane_position(self, pose):
        position = self._position(pose)
        if position.size == 2:
            return position
        return position[list(self.plane_indices)]

    def plan(self, pose, goal_pose, occupancy_map, origin=None, resolution=None):
        """Return world-plane waypoints from ``pose`` to ``goal_pose``.

        ``occupancy_map`` can be the ``(grid, origin)`` pair from
        ``get_map()``, or a grid with ``origin`` supplied separately. Before
        searching, occupied cells are inflated by ``inflation_radius``.
        """
        if origin is None:
            grid, origin = occupancy_map
        else:
            grid = occupancy_map
        grid = np.asarray(grid)
        origin = np.asarray(origin, dtype=float).reshape(2)
        cell_size = self.resolution if resolution is None else float(resolution)
        if cell_size <= 0 or not np.isfinite(cell_size):
            raise ValueError("resolution must be finite and positive")
        if grid.ndim != 2:
            raise ValueError("occupancy map must be a 2D grid")
        if not grid.size:
            self.path = self._plane_position(goal_pose).reshape(1, 2)
            return self.path.copy()

        grid = self._inflate(grid, cell_size)

        start = np.floor((self._plane_position(pose) - origin) / cell_size).astype(int)
        goal = np.floor((self._plane_position(goal_pose) - origin) / cell_size).astype(int)
        free = grid == 0

        def closest_free(cell):
            clipped = np.clip(cell, (0, 0), np.array(grid.shape) - 1)
            if free[tuple(clipped)]:
                return tuple(map(int, clipped))
            candidates = np.argwhere(free)
            if not len(candidates):
                return None
            index = int(np.argmin(np.sum((candidates - cell) ** 2, axis=1)))
            return tuple(map(int, candidates[index]))

        start_cell, goal_cell = closest_free(start), closest_free(goal)
        if start_cell is None or goal_cell is None:
            self.path = np.zeros((0, 2), dtype=float)
            return self.path.copy()

        frontier = []
        sequence = count()
        heappush(frontier, (0.0, next(sequence), start_cell))
        cost = {start_cell: 0.0}
        parent = {}
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))
        while frontier:
            _, _, current = heappop(frontier)
            if current == goal_cell:
                break
            for di, dj in directions:
                neighbor = (current[0] + di, current[1] + dj)
                if (neighbor[0] < 0 or neighbor[1] < 0
                        or neighbor[0] >= grid.shape[0] or neighbor[1] >= grid.shape[1]
                        or not free[neighbor]):
                    continue
                if di and dj and (not free[current[0] + di, current[1]]
                                  or not free[current[0], current[1] + dj]):
                    continue
                step = 1.41421356237 if di and dj else 1.0
                new_cost = cost[current] + step
                if new_cost >= cost.get(neighbor, float("inf")):
                    continue
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heuristic = np.linalg.norm(np.asarray(neighbor) - goal_cell)
                heappush(frontier, (new_cost + heuristic, next(sequence), neighbor))

        if goal_cell not in cost:
            self.path = np.zeros((0, 2), dtype=float)
            return self.path.copy()
        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        cells.reverse()
        self.path = origin + (np.asarray(cells, dtype=float) + 0.5) * cell_size
        return self.path.copy()
