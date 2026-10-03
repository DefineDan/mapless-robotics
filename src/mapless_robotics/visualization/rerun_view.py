"""Rerun camera, detector crop, and editable 3D graph visualization."""

from contextlib import contextmanager

import numpy as np

from .frontier_colors import latest_u_gain, u_gain_colors
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
        if graph is not None and graph.ena_map is not None:
            grid, (map_x, map_z, map_theta) = graph.ena_map.get_map()
            # GridMap is rendered in 3D as a planar raster. ExplorationMap's
            # first axis is world X and its second is world Z, so rotate its
            # local XY image plane onto Habitat's XZ ground plane.
            height, width = grid.shape
            occupancy = np.full(grid.shape, -1, dtype=np.int8)
            occupancy[grid == 255] = 0  # observed/known area
            rr.log(
                "world/exploration_map",
                rr.GridMap(
                    data=occupancy.tobytes(),
                    format=rr.components.ImageFormat(
                        width=width,
                        height=height,
                        color_model="L",
                        channel_datatype="U8",
                    ),
                    cell_size=graph.ena_map.resolution,
                    translation=[map_x, 0.0, map_z],
                    rotation_axis_angle=rr.components.RotationAxisAngle(
                        [1.0, 0.0, 0.0], -np.pi / 2
                    ),
                    colormap=rr.components.Colormap.RvizMap,
                    opacity=0.75,
                ),
            )
            rr.log(
                "world/exploration_map_info",
                rr.TextDocument(
                    f"Map origin: x={map_x:.3f} m, z={map_z:.3f} m; "
                    f"theta={map_theta:.3f} rad; "
                    f"resolution={graph.ena_map.resolution:.3f} m/cell"
                ),
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
            pose_directions = [
                -np.asarray(node.pose, dtype=float)[:3, 2] * 0.4
                for node in graph.poses.values()
            ]
            rr.log(
                "world/pose_directions",
                rr.Arrows3D(
                    origins=list(poses.values()),
                    vectors=pose_directions,
                    colors=[(35, 100, 210)],
                ),
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
            for is_objective, entity, objective_color, radius in (
                (False, "world/frontiers", None, 0.13),
                (True, "world/objective_frontier", (255, 40, 40), 0.22),
            ):
                displayed = [node for node in nodes if node.is_objective == is_objective]
                if not displayed:
                    continue
                colors, _ = u_gain_colors(displayed)
                if objective_color is not None:
                    colors = [objective_color] * len(displayed)
                rr.log(
                    entity,
                    rr.Points3D(
                        [node.position for node in displayed],
                        colors=colors,
                        radii=radius,
                        labels=[
                            f"OBJECTIVE F{node.id} u_gain {gain:.2f}"
                            if is_objective and (gain := latest_u_gain(node)) is not None
                            else f"OBJECTIVE F{node.id} u_gain ?"
                            if is_objective
                            else f"F{node.id} u_gain {gain:.2f}"
                            if (gain := latest_u_gain(node)) is not None
                            else f"F{node.id} u_gain ?"
                            for node in displayed
                        ],
                    ),
                )
            frontier_arrows = [
                (node.position, np.asarray(node.vd, dtype=float))
                for node in nodes
                if node.vd is not None
                and np.asarray(node.vd, dtype=float).shape == (3,)
                and np.isfinite(node.vd).all()
                and np.linalg.norm(node.vd) > 0
            ]
            if frontier_arrows:
                rr.log(
                    "world/frontier_directions",
                    rr.Arrows3D(
                        origins=[position for position, _ in frontier_arrows],
                        vectors=[
                            direction / np.linalg.norm(direction) * 0.4
                            for _, direction in frontier_arrows
                        ],
                        colors=[(220, 80, 180)],
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
