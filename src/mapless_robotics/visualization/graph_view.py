"""Simple OpenCV top-down (world X/Z) graph view."""

import cv2
import numpy as np

from .frontier_colors import latest_u_gain, u_gain_colors


def render_graph(graph, robot_pose=None, size=700):
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    poses = {i: np.asarray(n.pose)[:3, 3] for i, n in graph.poses.items()}
    frontiers = {i: np.asarray(n.position) for i, n in graph.frontiers.items()}
    points = list(poses.values()) + list(frontiers.values())
    if robot_pose is not None:
        points.append(np.asarray(robot_pose)[:3, 3])
    if not points:
        points = [np.zeros(3)]
    xz = np.asarray(points)[:, [0, 2]]
    center = (xz.min(axis=0) + xz.max(axis=0)) / 2
    span = max(float(np.ptp(xz, axis=0).max()), 2.0)
    scale = (size - 120) / span

    def pixel(point):
        return tuple(
            np.rint((np.asarray(point)[[0, 2]] - center) * scale + size / 2).astype(int)
        )

    for node in graph.frontiers.values():
        for pose_id in {obs["pose_id"] for obs in node.observations}:
            cv2.line(
                image, pixel(poses[pose_id]), pixel(node.position), (215, 215, 215), 1
            )
    for node in graph.poses.values():
        if node.parent_id is not None:
            cv2.line(
                image,
                pixel(poses[node.parent_id]),
                pixel(poses[node.id]),
                (200, 90, 30),
                2,
            )
        cv2.circle(image, pixel(poses[node.id]), 4, (200, 90, 30), -1)
        direction = -np.asarray(node.pose, dtype=float)[:3, 2]
        cv2.arrowedLine(
            image,
            pixel(poses[node.id]),
            pixel(poses[node.id] + direction * 0.4),
            (210, 100, 35),
            1,
            tipLength=0.3,
        )
    frontier_nodes = list(graph.frontiers.values())
    colors, gain_range = u_gain_colors(frontier_nodes)
    for node, rgb_color in zip(frontier_nodes, colors):
        p = pixel(node.position)
        if node.is_objective:
            cv2.circle(image, p, 10, (0, 0, 255), 2)
            cv2.drawMarker(image, p, (0, 0, 255), cv2.MARKER_STAR, 16, 2)
        else:
            cv2.circle(image, p, 6, rgb_color[::-1], -1)
        if node.vd is not None:
            direction = np.asarray(node.vd, dtype=float)
            if direction.shape == (3,) and np.isfinite(direction).all():
                norm = np.linalg.norm(direction)
                if norm > 0:
                    cv2.arrowedLine(
                        image,
                        p,
                        pixel(node.position + direction / norm * 0.4),
                        (180, 70, 180),
                        1,
                        tipLength=0.3,
                    )
        gain = latest_u_gain(node)
        label = (
            f"OBJECTIVE F{node.id} u:{gain:.2f}"
            if node.is_objective and gain is not None
            else f"OBJECTIVE F{node.id} u:?"
            if node.is_objective
            else f"F{node.id} u:{gain:.2f}"
            if gain is not None
            else f"F{node.id} u:?"
        )
        cv2.putText(
            image,
            label,
            (p[0] + 8, p[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 200) if node.is_objective else (50, 50, 50),
            1,
        )
    if robot_pose is not None:
        position = robot_pose[:3, 3]
        cv2.circle(image, pixel(position), 7, (40, 160, 40), -1)
        cv2.arrowedLine(
            image,
            pixel(position),
            pixel(position - robot_pose[:3, 2] * 0.4),
            (40, 160, 40),
            2,
        )
    lines = [
        f"Poses: {len(poses)}  Frontiers: {len(frontiers)}",
        "Blue: poses  Green: robot  Red star: objective frontier",
        (
            f"u_gain: purple {gain_range[0]:.2f} -> yellow {gain_range[1]:.2f}"
            if gain_range is not None
            else "u_gain: no estimates"
        ),
        "Gray: observations | X right, Z down (Y up)",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, 22 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
        )
    return image
