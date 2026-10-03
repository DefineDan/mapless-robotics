"""Rerun camera, detector crop, and editable 3D graph visualization."""

from contextlib import contextmanager

import numpy as np

from .frontier_colors import gain_colors, latest_gain
from ..terminal_keyboard import TerminalKeyboard
from .keys import action_for_key, key_name


class RerunView:
    def __init__(self, *, spawn=True):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        self.step = 0
        self.last_camera_log = -float("inf")
        rr.init("frontier_exploration", spawn=spawn)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="Exploration graph", origin="world"),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="Live camera", origin="camera"),
                        rrb.Spatial2DView(
                            name="Detected frontiers", origin="detections"
                        ),
                    ),
                    column_shares=[2, 1],
                ),
                collapse_panels=True,
            )
        )
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    @contextmanager
    def get_context(self):
        try:
            with TerminalKeyboard() as keyboard:
                self._keyboard = keyboard
                yield self
        finally:
            self._keyboard = None

    def get_keys(self):
        """Read one key event as a shared symbolic key name."""
        return key_name(self._keyboard.poll())

    @staticmethod
    def get_action(key):
        return action_for_key(key)

    def set_status(self, status):
        self.status = status

    def start_step(self):
        self.rr.set_time("frame", sequence=self.step)
        self.step += 1

    def log_camera(self, observation, now, *, force=False, graph=None):
        if not force and now - self.last_camera_log < 0.1:
            return
        self.last_camera_log = now
        rr = self.rr
        rr.log("camera/rgb", rr.Image(observation.rgb).compress(jpeg_quality=85))
        position = observation.robot_pose[:3, 3]
        heading = -observation.robot_pose[:3, 2] * 0.5
        rr.log(
            "world/robot",
            rr.Arrows3D(origins=[position], vectors=[heading], colors=[(30, 180, 60)]),
        )

    def log_analysis(self, result, pose_id, graph):
        rr = self.rr
        rr.log(
            "detections/image", rr.Image(result.marked_rgb).compress(jpeg_quality=90)
        )
        rr.log(
            "detections/status",
            rr.TextDocument(
                f"Pose {pose_id}: {len(result.frontiers)} detections; "
                f"capture time {result.observation.timestamp:.3f}"
            ),
        )
        poses = {node.id: np.asarray(node.pose)[:3, 3] for node in graph.poses.values()}
        if poses:
            rr.log(
                "world/poses",
                rr.Points3D(list(poses.values()), colors=[(35, 100, 210)], radii=0.08),
            )
            edges = [
                [poses[node.parent_id], poses[node.id]]
                for node in graph.poses.values()
                if node.parent_id is not None
            ]
            if edges:
                rr.log(
                    "world/pose_connections",
                    rr.LineStrips3D(edges, colors=[(35, 100, 210)]),
                )
        if graph.frontiers:
            nodes = list(graph.frontiers.values())
            colors, _ = gain_colors(nodes)
            rr.log(
                "world/frontiers",
                rr.Points3D(
                    [node.position for node in nodes],
                    colors=colors,
                    radii=0.13,
                    labels=[
                        f"F{node.id} gain {gain:.2f}"
                        if (gain := latest_gain(node)) is not None
                        else f"F{node.id} gain ?"
                        for node in nodes
                    ],
                ),
            )
            observed_from = [
                [poses[pose_id], node.position]
                for node in nodes
                for pose_id in {obs["pose_id"] for obs in node.observations}
            ]
            if observed_from:
                rr.log(
                    "world/observations",
                    rr.LineStrips3D(observed_from, colors=[(170, 170, 170)]),
                )
