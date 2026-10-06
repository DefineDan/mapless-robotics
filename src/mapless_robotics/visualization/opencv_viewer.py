"""OpenCV windows implementing the exploration visualization interface."""

from contextlib import nullcontext
from datetime import UTC, datetime

import cv2

from .graph_view import render_graph
from .keys import action_for_key, key_name


class OpenCVViewer:
    def __init__(self):
        self.status = ""

    def get_context(self):
        return nullcontext(self)

    def get_keys(self):
        return key_name(cv2.waitKeyEx(33))

    @staticmethod
    def get_action(key):
        return action_for_key(key)

    def start_step(self):
        pass

    def log_camera(self, observation, now, *, force=False, graph=None):
        frame = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            self.status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        cv2.imshow("Habitat", frame)
        if graph is not None:
            cv2.imshow("Exploration graph", render_graph(graph, observation.robot_pose))
            if graph.ena_map is not None:
                grid, _map_pose = graph.ena_map.get_map()
                cv2.imshow("Exploration map", render_map(grid, graph.ena_map.resolution))

    def log_navigation_map(self, navigator):
        grid, _origin = navigator.get_map()
        if grid.size:
            cv2.imshow("Depth navigation map", render_map(grid, navigator.resolution))

    def log_analysis(self, result, pose_id, graph):
        frame = cv2.cvtColor(result.marked_rgb, cv2.COLOR_RGB2BGR)
        frame = cv2.copyMakeBorder(
            frame, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
        )
        captured = datetime.fromtimestamp(
            result.observation.timestamp, tz=UTC
        ).strftime("%H:%M:%S UTC")
        lines = (
            f"Pose {pose_id} | {len(result.frontiers)} detections (local labels)",
            f"Captured {captured} | detector crop",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, 18 + index * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("Detected frontiers", frame)

    def set_status(self, status):
        self.status = status


def render_map(grid, resolution):
    """Render seen cells as a readable top-down grayscale OpenCV image."""
    image = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    # Scale small grids for convenient inspection while keeping large maps usable.
    scale = min(4.0, max(1.0, 600.0 / max(image.shape[:2])))
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.putText(image, f"{resolution:.3f} m/cell", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return image
