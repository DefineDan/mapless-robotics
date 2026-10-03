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
