"""2D record of ground-plane regions seen by a pinhole camera."""

import numpy as np


class ExplorationMap:
    """Occupancy/visibility grid in Habitat's X-Z ground plane.

    Cells covered by a camera's horizontal field of view are set to 255. The
    grid is centered at world origin initially and expands when a pose falls
    outside its current bounds.
    """

    def __init__(self, ini_depth=0.1, end_depth=5.0, resolution=0.05):
        if not (np.isfinite(ini_depth) and np.isfinite(end_depth)
                and 0 <= ini_depth < end_depth):
            raise ValueError("depth limits must satisfy 0 <= ini_depth < end_depth")
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        self.ini_depth = float(ini_depth)
        self.end_depth = float(end_depth)
        self.resolution = float(resolution)
        self.origin = np.array([-end_depth, -end_depth], dtype=float)
        size = int(np.ceil(2 * end_depth / resolution))
        self.map = np.zeros((size, size), dtype=np.uint8)

    @property
    def grid(self):
        return self.map

    def get_map(self):
        """Return the occupancy grid and its world pose ``(x, z, theta)``.

        The grid's first array axis is world X and its second is world Z;
        ``theta`` is zero because the grid is axis-aligned. The returned grid
        is a copy, so callers cannot modify the map accidentally.
        """
        return self.map.copy(), (float(self.origin[0]), float(self.origin[1]), 0.0)

    def _cell(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def _ensure_contains(self, points):
        cells = self._cell(points)
        lo = np.minimum(cells.min(axis=0), 0)
        hi = np.maximum(cells.max(axis=0), np.array([self.map.shape[1]-1, self.map.shape[0]-1]))
        if np.any(lo < 0) or np.any(hi >= np.array([self.map.shape[1], self.map.shape[0]])):
            # Grow in whole grid-sized blocks and preserve existing world coordinates.
            old_h, old_w = self.map.shape
            left, bottom = np.maximum(-lo, 0)
            right, top = np.maximum(hi - np.array([old_w-1, old_h-1]), 0)
            new = np.zeros((old_h + bottom + top, old_w + left + right), dtype=np.uint8)
            new[bottom:bottom+old_h, left:left+old_w] = self.map
            self.map = new
            self.origin -= np.array([left, bottom]) * self.resolution

    def add_pose(self, pose, camera):
        """Mark the ground-plane camera frustum.

        ``pose`` is a camera-to-world 4x4 matrix. ``camera`` is either a 3x3
        intrinsic matrix or ``(width, height, hfov_degrees)``.
        """
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 camera-to-world matrix")
        if np.shape(camera) == (3, 3):
            fx, cx = float(camera[0, 0]), float(camera[0, 2])
            width = 2 * cx
        else:
            width, _height, hfov = camera
            width = float(width)
            fx = width / (2 * np.tan(np.deg2rad(hfov) / 2))
            cx = width / 2
        if fx <= 0 or width <= 0:
            raise ValueError("camera intrinsics must have positive focal length and width")

        # Image edges define the horizontal angular limits of the pinhole view.
        angles = np.arctan2(np.array([0.0, width]) - cx, fx)
        forward = pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
        right = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
        center = pose[[0, 2], 3]
        corners = [center]
        for depth in (self.ini_depth, self.end_depth):
            for angle in angles:
                direction = forward * np.cos(angle) + right * np.sin(angle)
                horizontal = direction[[0, 2]]
                norm = np.linalg.norm(horizontal)
                if norm > 1e-8:
                    corners.append(center + depth * horizontal / norm)
        corners = np.asarray(corners)
        self._ensure_contains(corners)

        # Rasterize only the frustum bounding box, then use a ray-crossing test.
        lo = np.maximum(self._cell(corners).min(axis=0), 0)
        hi = np.minimum(self._cell(corners).max(axis=0),
                        np.array([self.map.shape[1]-1, self.map.shape[0]-1]))
        xs = self.origin[0] + (np.arange(lo[0], hi[0]+1) + 0.5) * self.resolution
        zs = self.origin[1] + (np.arange(lo[1], hi[1]+1) + 0.5) * self.resolution
        xx, zz = np.meshgrid(xs, zs)
        delta = np.stack((xx-center[0], zz-center[1]), axis=-1)
        dist = np.linalg.norm(delta, axis=-1)
        # Evaluate camera-relative horizontal angle, accounting for orientation.
        f = forward[[0, 2]]
        r = right[[0, 2]]
        f /= max(np.linalg.norm(f), 1e-12)
        r /= max(np.linalg.norm(r), 1e-12)
        forward_depth = delta @ f
        horizontal_angle = np.arctan2(delta @ r, forward_depth)
        visible = ((dist >= self.ini_depth) & (dist <= self.end_depth)
                   & (forward_depth > 0)
                   & (horizontal_angle >= angles[0])
                   & (horizontal_angle <= angles[1]))
        self.map[lo[1]:hi[1]+1, lo[0]:hi[0]+1][visible] = 255

    def is_occupied(self, positions):
        """Return a bool per world ``(x, z)`` position, true if previously seen."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, 2)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        cells = self._cell(positions)
        valid = (np.isfinite(positions).all(axis=1)
                 & (cells[:, 0] >= 0) & (cells[:, 0] < self.map.shape[1])
                 & (cells[:, 1] >= 0) & (cells[:, 1] < self.map.shape[0]))
        result = np.zeros(len(positions), dtype=bool)
        result[valid] = self.map[cells[valid, 1], cells[valid, 0]] == 255
        return result
