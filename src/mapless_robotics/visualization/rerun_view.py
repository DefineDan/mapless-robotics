"""Rerun camera, detector crop, and editable 3D graph visualization."""

from contextlib import contextmanager

import numpy as np

from .frontier_colors import latest_u_gain, u_gain_colors
from ..terminal_keyboard import TerminalKeyboard
from .keys import action_for_key, key_name


def segment_capsules(rr, segments, *, radius, color):
    """Solid capsules along segments; unlike LineStrips3D they honour alpha."""
    starts = np.asarray([start for start, _ in segments], dtype=float)
    deltas = np.asarray([end for _, end in segments], dtype=float) - starts
    lengths = np.linalg.norm(deltas, axis=1)
    directions = deltas / np.maximum(lengths, 1e-9)[:, None]
    # Shortest-arc quaternion (xyzw) rotating the capsule's +Z onto each segment.
    quats = np.stack(
        (-directions[:, 1], directions[:, 0], np.zeros(len(directions)),
         1.0 + directions[:, 2]),
        axis=1,
    )
    quats[quats[:, 3] < 1e-6] = (1.0, 0.0, 0.0, 0.0)  # anti-parallel to +Z
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return rr.Capsules3D(
        lengths=lengths,
        radii=radius,
        translations=starts,
        quaternions=[rr.Quaternion(xyzw=q) for q in quats],
        colors=[color],
        fill_mode=rr.components.FillMode.Solid,
    )


class RerunView:
    def __init__(
        self, *, spawn=True, enable_edge_vis=False, enable_depth_vis=True,
        enable_navigation_points_vis=False,
    ):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        self.step = 0
        self.last_camera_log = -float("inf")
        self.enable_edge_vis = enable_edge_vis
        self.enable_depth_vis = enable_depth_vis
        self.enable_navigation_points_vis = enable_navigation_points_vis
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
        # Lift the robot marker above the graph so it is not hidden under it.
        position = observation.robot_pose[:3, 3] + np.array([0.0, 0.4, 0.0])
        heading = -observation.robot_pose[:3, 2] * 0.6
        rr.log(
            "world/robot",
            rr.Arrows3D(
                origins=[position], vectors=[heading], colors=[(40, 230, 70)],
                radii=0.04,
            ),
        )
        if graph is not None and graph.ena_map is not None:
            grid, (map_x, map_z, map_theta) = graph.ena_map.get_map()
            # ExplorationMap already stores rows along world Z and columns
            # along world X. GridMap uses top-row-first image data, while the
            # map's row zero is its low-Z edge, so flip the rows vertically.
            # GridMap's translation is the lower-left image corner, not the
            # center of the map or the first (top) row.
            occupancy = np.full(grid.shape, -1, dtype=np.int8)
            occupancy[grid == 255] = 0  # observed/known area
            occupancy = occupancy[::-1, :]
            height, width = occupancy.shape
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
                        [1.0, 0.0, 0.0], np.pi / 2
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

    def log_navigation_map(self, navigator):
        rr = self.rr
        if self.enable_navigation_points_vis:
            points = navigator.get_point_cloud()
            if len(points):
                self.rr.log(
                    "world/depth_navigation_points",
                    self.rr.Points3D(points, colors=[(255, 180, 40)], radii=0.025),
                )
        for entity in ("world/depth_navigation_trajectory", "world/depth_navigation_waypoints"):
            rr.log(entity, rr.Clear(recursive=True))
        trajectory = np.asarray(getattr(navigator, "trajectory", ()), dtype=float)
        if trajectory.ndim == 2 and trajectory.shape[1] == 2 and len(trajectory):
            path_points = np.zeros((len(trajectory), 3), dtype=float)
            path_points[:, navigator.plane_indices] = trajectory
            if navigator.camera_pose is not None:
                # path_points[:, navigator.vertical_index] = navigator.camera_pose[navigator.vertical_index, 3]
                path_points[:, navigator.vertical_index] = 0
            rr.log(
                "world/depth_navigation_trajectory",
                rr.LineStrips3D([path_points], colors=[(40, 220, 255)], radii=0.035),
            )
            rr.log(
                "world/depth_navigation_waypoints",
                rr.Points3D(path_points, colors=[(40, 220, 255)], radii=0.05),
            )

        grid, origin = navigator.get_map()
        if not grid.size:
            rr.log("world/depth_navigation_map", rr.Clear(recursive=True))
            return
        # get_map() stores planar coordinates as [axis 0, axis 1]. Rerun
        # expects a top-row-first image, with columns along local +X and rows
        # along local +Y. The X/-Z rotation below maps those axes onto Habitat
        # world X/Z, so transpose the grid and reverse its rows accordingly.
        occupancy = np.full(grid.shape, -1, dtype=np.int8)
        occupancy[grid == 1] = 0
        occupancy[grid == 255] = 100
        occupancy = occupancy.T[::-1, :]
        height, width = occupancy.shape
        plane = navigator.plane_indices
        translation = np.zeros(3, dtype=float)
        translation[plane[0]] = origin[0]
        # GridMap translation is the lower-left corner. Reversing image rows
        # makes row zero the high end of axis 1, but leaves the lower edge at
        # the map origin.
        translation[plane[1]] = origin[1]
        # The raster axes follow the configured planar axes. For the default
        # X/Z ground plane, rotating the image plane around X places it on the
        # Habitat floor, matching the exploration map visualization.
        rr.log(
            "world/depth_navigation_map",
            rr.GridMap(
                data=occupancy.tobytes(),
                format=rr.components.ImageFormat(
                    width=width,
                    height=height,
                    color_model="L",
                    channel_datatype="U8",
                ),
                cell_size=navigator.resolution,
                translation=translation,
                rotation_axis_angle=rr.components.RotationAxisAngle(
                    [1.0, 0.0, 0.0], np.pi / 2
                ),
                colormap=rr.components.Colormap.RvizMap,
                opacity=0.85,
            ),
        )

    def log_analysis(self, result, pose_id, graph):
        rr = self.rr
        if result.frontier_analysis:
            rr.log(
                "detections/image", rr.Image(result.marked_rgb).compress(jpeg_quality=90)
            )
        depth = result.depth
        intrinsics = getattr(graph, "camera_intrinsics", None)
        if self.enable_depth_vis and depth is not None and intrinsics is not None:
            depth = np.asarray(depth, dtype=np.float32)
            rgb = np.asarray(result.observation.rgb)
            stride = max(1, int(np.ceil(max(depth.shape) / 160)))
            height, width = depth.shape
            v, u = np.mgrid[0:height:stride, 0:width:stride]
            z = depth[::stride, ::stride]
            valid = np.isfinite(z) & (z > 0)
            fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
            cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
            # Depth is forward distance. Convert OpenCV (+Z forward, +Y down)
            # into Habitat camera coordinates (-Z forward, +Y up).
            camera_points = np.stack(
                ((u - cx) * z / fx, -(v - cy) * z / fy, -z), axis=-1
            )[valid]
            camera_pose = np.asarray(result.observation.camera_pose, dtype=float)
            world_points = camera_points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
            colors = rgb[::stride, ::stride][valid]
            rr.log(
                "world/depth_cloud",
                rr.Points3D(world_points, colors=colors, radii=0.012),
            )
        if result.frontier_analysis:
            rr.log(
                "detections/status",
                rr.TextDocument(
                    f"Pose {pose_id}: {len(result.frontiers)} detections; "
                    f"capture time {result.observation.timestamp:.3f}"
                ),
            )
        for entity in ("world/frontiers", "world/objective_frontier",
                       "world/frontier_directions", "world/observations",
                       "world/pose_connections", "world/poses", "world/pose_directions",
                       "world/excluded_branches", "world/frontier_history"):
            rr.log(entity, rr.Clear(recursive=True))
        history = getattr(graph, "retired_frontiers", [])
        if history:
            rr.log("world/frontier_history", rr.Points3D(
                [item["frontier"]["position"] for item in history],
                colors=[(160, 60, 60) if item["reason"] in {"unreachable", "invalidated"}
                        else (135, 135, 135) for item in history],
                radii=0.05,
                labels=[f"F{item['frontier']['id']} {item['reason']}" for item in history],
            ))
        excluded = getattr(graph, "excluded_pose_ids", set())
        visible_poses = [node for node in graph.poses.values() if node.id not in excluded]
        excluded_edges = [
            [graph.poses[node.parent_id].pose[:3, 3], node.pose[:3, 3]]
            for node in graph.poses.values()
            if node.id in excluded and node.parent_id in graph.poses
        ]
        if excluded_edges:
            rr.log("world/excluded_branches", rr.LineStrips3D(
                excluded_edges, colors=[(180, 65, 65)]))
        poses = {node.id: np.asarray(node.pose)[:3, 3] for node in visible_poses}
        if poses:
            rr.log(
                "world/poses",
                rr.Points3D(list(poses.values()), colors=[(35, 100, 210)], radii=0.08),
            )
            pose_directions = [
                -np.asarray(node.pose, dtype=float)[:3, 2] * 0.4
                for node in visible_poses
            ]
            rr.log(
                "world/pose_directions",
                rr.Arrows3D(
                    origins=list(poses.values()),
                    vectors=pose_directions,
                    colors=[(35, 100, 210)],
                ),
            )
            edges = (
                [
                    [poses[node.parent_id], poses[node.id]]
                    for node in visible_poses
                    if node.parent_id in poses
                ]
                if self.enable_edge_vis
                else []
            )
            if edges:
                rr.log(
                    "world/pose_connections",
                    rr.LineStrips3D(edges, colors=[(35, 100, 210)]),
                )
        if graph.frontiers:
            nodes = list(graph.frontiers.values())
            node_colors = {}
            for is_objective, entity, objective_color, radius in (
                (False, "world/frontiers", None, 0.065),
                (True, "world/objective_frontier", (255, 40, 40), 0.075),
            ):
                displayed = [node for node in nodes if node.is_objective == is_objective]
                if not displayed:
                    continue
                colors, _ = u_gain_colors(displayed)
                if objective_color is not None:
                    colors = [objective_color] * len(displayed)
                node_colors.update(zip((node.id for node in displayed), colors))
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
                (node.position, np.asarray(node.vd, dtype=float), node_colors[node.id])
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
                        origins=[position for position, _, _ in frontier_arrows],
                        vectors=[
                            direction / np.linalg.norm(direction) * 0.4
                            for _, direction, _ in frontier_arrows
                        ],
                        colors=[color for _, _, color in frontier_arrows],
                    ),
                )
            observed_from = [
                [poses[pose_id], node.position]
                for node in nodes
                for pose_id in {obs["pose_id"] for obs in node.observations}
                if pose_id in poses
            ]
            if observed_from and self.enable_edge_vis:
                rr.log(
                    "world/observations",
                    segment_capsules(
                        rr, observed_from, radius=0.005, color=(170, 170, 170, 20)
                    ),
                )

    def log_return_target(self, frontier, pose, reference_rgb, *, score=None,
                          status="Returning"):
        """Mark the frontier and the saved viewpoint LoTIS must reach first."""
        rr = self.rr
        self.clear_return_target()
        viewpoint = np.asarray(pose.pose)[:3, 3]
        gain_label = f" | peak gain {score:.3f}" if score is not None else ""
        rr.log(
            "world/return_target/frontier",
            rr.Points3D(
                [frontier.position], colors=[(255, 210, 40)], radii=0.25,
                labels=[f"RETURN FRONTIER F{frontier.id}{gain_label}"],
                show_labels=True,
            ),
        )
        rr.log(
            "world/return_target/viewpoint",
            rr.Points3D(
                [viewpoint], colors=[(40, 220, 255)], radii=0.18,
                labels=[f"{status}: P{pose.id} | frame {pose.frame_index}"],
                show_labels=True,
            ),
        )
        rr.log(
            "world/return_target/viewpoint_heading",
            rr.Arrows3D(
                origins=[viewpoint], vectors=[-np.asarray(pose.pose)[:3, 2] * 0.6],
                colors=[(40, 220, 255)],
            ),
        )
        rr.log(
            "world/return_target/observation",
            rr.LineStrips3D([[viewpoint, frontier.position]],
                            colors=[(40, 220, 255)], radii=0.025),
        )
        rr.log("return_target/image", rr.Image(reference_rgb).compress(jpeg_quality=90))
        rr.log(
            "return_target/status",
            rr.TextDocument(
                f"{status} to P{pose.id} (recorded frame {pose.frame_index}), "
                f"then depth navigation to F{frontier.id}{gain_label}.\n"
                "Yellow: selected frontier. Cyan: saved return viewpoint and heading.\n"
                "The connecting line shows which viewpoint observed the frontier."
            ),
        )

    def clear_return_target(self):
        """Remove the previous return markers and reference image."""
        for entity in ("world/return_target", "return_target"):
            self.rr.log(entity, self.rr.Clear(recursive=True))

    def clear_route_overlay(self):
        """Remove LoTIS's projected route from the live camera view."""
        for entity in ("camera/points", "camera/lines"):
            self.rr.log(entity, self.rr.Clear(recursive=False))
