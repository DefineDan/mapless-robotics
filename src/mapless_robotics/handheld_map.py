"""Camera-centred top-down display of the actual local planner's map and path."""
from dataclasses import dataclass
import threading
import time
import numpy as np


@dataclass(frozen=True)
class NavigationSnapshot:
    grid: np.ndarray
    origin: np.ndarray
    resolution: float
    path: np.ndarray
    goal: np.ndarray | None
    updated_at: float


class TopDownMap:
    """X is right, world -Z is up; camera position remains at image centre."""
    def __init__(self, span=6., pixels=640):
        if not np.isfinite(span) or span <= 0:
            raise ValueError('Map span must be finite and positive')
        self.span = float(span)
        self.pixels = int(pixels)
        self.lock = threading.Lock()
        self.snapshot = None

    def update(self, navigator):
        grid, origin = navigator.get_map()
        index = getattr(navigator, '_trajectory_index', 0)
        snapshot = NavigationSnapshot(
            np.array(grid, copy=True), np.array(origin, copy=True), float(navigator.resolution),
            np.array(navigator.trajectory[index:], dtype=float, copy=True).reshape(-1, 2),
            None if navigator.goal is None else np.array(navigator.goal, copy=True), time.time())
        with self.lock:
            self.snapshot = snapshot

    def world_to_pixel(self, points, camera_position):
        points = np.asarray(points, dtype=float)
        centre = np.asarray(camera_position, dtype=float)[[0, 2]]
        if points.shape[-1] == 3:
            points = points[..., [0, 2]]
        return (points-centre)*(self.pixels/self.span)+self.pixels/2

    def background(self, camera_position, snapshot):
        """Nearest-neighbour crop with unknown outside the measured local grid."""
        image = np.full((self.pixels, self.pixels, 3), (45, 49, 56), dtype=np.uint8)
        if snapshot is None or not snapshot.grid.size:
            return image
        camera = np.asarray(camera_position)[[0, 2]]
        offsets = (np.arange(self.pixels)-self.pixels/2)*self.span/self.pixels
        x = np.floor((camera[0]+offsets-snapshot.origin[0])/snapshot.resolution).astype(int)
        z = np.floor((camera[1]+offsets-snapshot.origin[1])/snapshot.resolution).astype(int)
        valid_x = np.flatnonzero((x >= 0) & (x < snapshot.grid.shape[0]))
        valid_z = np.flatnonzero((z >= 0) & (z < snapshot.grid.shape[1]))
        values = snapshot.grid[x[valid_x][None, :], z[valid_z][:, None]]
        colors = np.full((*values.shape, 3), (45, 49, 56), dtype=np.uint8)
        colors[values == 1] = (204, 214, 218)
        colors[values == 255] = (225, 78, 67)
        image[valid_z[:, None], valid_x[None, :]] = colors
        return image

    def render(self, frame, goal=None, label='No selected goal', status='STOP',
               arrival_radius=None, reference_path=None, depth_source='realsense'):
        import cv2
        with self.lock:
            snapshot = self.snapshot
        pose = np.eye(4) if frame is None else frame.camera_pose
        position = pose[:3, 3]
        image = self.background(position, snapshot)
        scale = self.pixels/self.span
        centre = (self.pixels//2, self.pixels//2)
        # Metric grid, fixed in world coordinates while the viewport follows us.
        spacing = 10. ** np.floor(np.log10(self.span/6.))
        for axis in [0, 2]:
            lower = position[axis]-self.span/2
            upper = position[axis]+self.span/2
            for value in np.arange(np.ceil(lower/spacing)*spacing, upper, spacing):
                coordinate = int(round((value-position[axis])*scale+self.pixels/2))
                if axis == 0:
                    cv2.line(image, (coordinate, 0), (coordinate, self.pixels-1), (90, 99, 110), 1)
                else:
                    cv2.line(image, (0, coordinate), (self.pixels-1, coordinate), (90, 99, 110), 1)
        if arrival_radius is not None and np.isfinite(arrival_radius):
            cv2.circle(image, centre, int(round(min(arrival_radius, self.span)*scale)), (100, 155, 230), 1)
        matching_path = (snapshot is not None and snapshot.goal is not None and goal is not None
                         and np.allclose(snapshot.goal, goal, atol=1e-5))
        path = snapshot.path if matching_path else np.zeros((0, 2))
        kind = 'planned path'
        color = (35, 205, 240)
        if reference_path is not None:
            path = np.asarray(reference_path, dtype=float).reshape(-1, 3)[:, [0, 2]]
            kind, color = 'recorded LoTIS route', (255, 165, 55)
        if goal is not None and len(path) >= 2:
            pixels = np.rint(self.world_to_pixel(path, position)).astype(int)
            for a, b in zip(pixels[:-1], pixels[1:]):
                ok, start, end = cv2.clipLine((0, 0, self.pixels, self.pixels), tuple(map(int, a)), tuple(map(int, b)))
                if ok:
                    cv2.line(image, start, end, color, 4, cv2.LINE_AA)
        goal_visible = False
        distance = None
        if goal is not None:
            target = self.world_to_pixel(goal, position)
            distance = float(np.linalg.norm((np.asarray(goal)-position)[[0, 2]]))
            margin = 90
            vector = target-np.array(centre)
            limit = self.pixels/2-margin
            factor = min(1., limit/max(np.max(np.abs(vector)), 1e-9))
            marker = tuple(np.rint(np.array(centre)+factor*vector).astype(int))
            goal_visible = factor == 1.
            cv2.drawMarker(image, marker, (255, 195, 55), cv2.MARKER_DIAMOND, 22, 3)
            text = f'{label}: {distance:.2f} m'+('' if goal_visible else ' (off-screen)')
        else:
            text = label
        # The heading marker represents measured orientation, not a command.
        forward = -pose[:3, 2][[0, 2]]
        norm = np.linalg.norm(forward)
        if frame is not None and norm > 1e-8:
            endpoint = tuple(np.rint(np.array(centre)+forward/norm*32).astype(int))
            cv2.arrowedLine(image, centre, endpoint, (80, 145, 255), 4, tipLength=.3)
        if frame is not None:
            cv2.circle(image, centre, 8, (80, 145, 255), -1)
        if frame is None:
            text = 'Waiting for live OpenVINS position'
        age = None if snapshot is None else max(0., time.time()-snapshot.updated_at)
        path_text = kind if len(path) >= 2 and goal is not None else 'No local path available'
        cv2.rectangle(image, (0, 0), (self.pixels, 73), (22, 26, 32), -1)
        cv2.putText(image, f'LOCAL MAP | {status}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1)
        cv2.putText(image, text[:85], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 195, 55), 1)
        cv2.putText(image, path_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
        cv2.rectangle(image, (0, self.pixels-49), (self.pixels, self.pixels), (22, 26, 32), -1)
        caption = f'{self.span:g} m view | -Z up, +X right | depth: {depth_source}'
        if age is not None:
            caption += f' | update {age:.1f}s ago'
        cv2.putText(image, caption, (10, self.pixels-29), cv2.FONT_HERSHEY_SIMPLEX, .39, (240, 240, 240), 1)
        cv2.putText(image, 'Red obstacle | light free | grey unknown | blue you | gold goal',
                    (10, self.pixels-10), cv2.FONT_HERSHEY_SIMPLEX, .38, (240, 240, 240), 1)
        return image, dict(goal_visible=goal_visible, distance=distance, path_kind=path_text)
