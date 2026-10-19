"""Manually walk a Habitat scene and inspect observed exploration-map pixels.

Keys: arrows move, a/d turn, w/s look, g saves the pose graph, q/Escape quits.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import time

import numpy as np

from .exploration.exploration_graph import ExplorationGraph
from .terminal_keyboard import TerminalKeyboard
from .visualization.keys import action_for_key, key_name
from .visualization.rerun_view import RerunView
from .walk_frontiers import camera_pitch_degrees, capture, make_sim, resolve_scene


def pose_changed(current_pose, previous_pose):
    """Check whether the camera moved at least 10 cm or rotated 5 degrees."""
    if previous_pose is None:
        return True
    if np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3]) >= 0.1:
        return True
    relative_rotation = previous_pose[:3, :3].T @ current_pose[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)) >= 5.0


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=25, help="HM3D scene index")
    parser.add_argument("--scene", help="Path to a .glb scene (overrides --index)")
    parser.add_argument("--scenes-dir", type=Path,
                        help="HM3D split directory containing indexed scene folders")
    parser.add_argument("--camera-pitch", type=camera_pitch_degrees, default=20.0,
                        help="Initial camera tilt in degrees, positive down (default: 20)")
    parser.add_argument("--map-pixel-report-every", type=int, default=10,
                        help="Print observed pixels every N checked poses (default: 10; 0 disables)")
    parser.add_argument("--graph-output", type=Path,
                        help="Pose graph path (default: runs/manual_explore_navigation/<timestamp>/graph.json)")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.map_pixel_report_every < 0:
        parser.error("--map-pixel-report-every must be nonnegative")
    try:
        scene = resolve_scene(args.scene, args.index, args.scenes_dir)
    except FileNotFoundError as error:
        parser.error(str(error))

    output = args.graph_output or (
        Path("runs/manual_explore_navigation")
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        / "graph.json"
    )
    graph = ExplorationGraph(occupancy_report_every=args.map_pixel_report_every)
    sim = None
    try:
        sim = make_sim(str(scene), with_depth=False, camera_pitch=args.camera_pitch)
        if not sim.pathfinder.is_loaded:
            import habitat_sim

            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError("Could not build a walkable navmesh for this scene")

        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)
        view = RerunView()
        last_camera_pose = None
        print("Manual exploration | arrows move | a/d turn | w/s look | g save | q quit")
        print(f"Scene: {scene}")
        with TerminalKeyboard() as keyboard:
            while True:
                observation = capture(sim, agent)
                if pose_changed(observation.camera_pose, last_camera_pose):
                    pose_id = max(graph.poses, default=-1) + 1
                    graph.add_pose(
                        pose_id,
                        observation.robot_pose,
                        graph.path_parent_id,
                        observation.timestamp,
                        observation.camera_pose,
                    )
                    last_camera_pose = observation.camera_pose.copy()

                view.start_step()
                view.log_camera(observation, time.monotonic(), graph=graph)
                key = keyboard.poll()
                key_label = key_name(key)
                if key_label == "quit":
                    break
                if key_label == "g":
                    graph.save(output)
                    print(f"Saved {len(graph.poses)} poses to {output}")
                    continue
                action = action_for_key(key_label)
                if action is not None:
                    sim.step(action)
    finally:
        graph.save(output)
        if graph.poses:
            print(f"Saved {len(graph.poses)} poses to {output}")
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()