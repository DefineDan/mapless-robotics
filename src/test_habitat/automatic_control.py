#!/usr/bin/env python
"""Automatic trajectory generation.

Samples random navmesh paths, traverses them with the pure-pursuit
controller, logs RGB/depth frames, and automatically labels room
transitions using HM3D semantic region IDs.

Transition definition:
    start = fixed metric distance before semantic region change
    peak  = first frame in the new stable semantic region
    end   = fixed metric distance after semantic region change
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from habitat_collection.env import (
    HabitatCollectionEnv,
    discover_scenes,
    load_config,
)
from habitat_collection.logging_utils import (
    RerunLogger,
    TrajectoryLogger,
    unique_run_dir,
)
from habitat_collection.trajectory import (
    TrajectorySettings,
    sample_random_path,
    traverse_path,
)
from habitat_collection.room_transition_labels import (
    save_room_transition_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and traverse automatic Habitat trajectories "
            "with HM3D semantic room-transition labels."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "collection_config.yaml",
        help="Standalone Habitat collection config.",
    )

    parser.add_argument(
        "--scene-id",
        help="Specific discovered scene_id to load.",
    )

    parser.add_argument(
        "--scene-filter",
        help="Only list/select scenes whose id contains this text.",
    )

    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Print discovered scenes and exit.",
    )

    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help=(
            "Index into the selected scene list when "
            "--scene-id is not set."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "automatic",
        help=(
            "Directory that receives one numbered "
            "subdirectory per trajectory."
        ),
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save RGB/depth data.",
    )

    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth sensor even if the config enables it.",
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=0,
        help=(
            "Trajectories to generate for the scene. "
            "Defaults to "
            "TRAJECTORY_SETTINGS.TRAJECTORIES_PER_SCENE."
        ),
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help=(
            "Max start/end sampling attempts per trajectory "
            "before giving up."
        ),
    )

    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable live Rerun logging.",
    )

    parser.add_argument(
        "--rerun-addr",
        default="127.0.0.1:9876",
        help="Rerun server address for grpc proxy.",
    )

    parser.add_argument(
        "--rerun-spawn",
        action="store_true",
        help=(
            "Spawn a local Rerun viewer instead of "
            "connecting to --rerun-addr."
        ),
    )

    # ------------------------------------------------------------
    # Automatic room-transition labeling
    # ------------------------------------------------------------

    parser.add_argument(
        "--transition-stable-frames",
        type=int,
        default=3,
        help=(
            "Number of consecutive frames required before a "
            "new semantic region is accepted as a room change."
        ),
    )

    parser.add_argument(
        "--transition-half-width",
        type=float,
        default=0.75,
        help=(
            "Distance in meters before/after the semantic "
            "room boundary used for start/end labels."
        ),
    )

    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    args = parse_args()

    config_path = args.config

    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config = load_config(config_path)

    if args.no_depth:
        config.SIM_SETTINGS.DEPTH_SENSOR = False

    # We need the full scene list for explicit scene selection
    # and listing.
    if args.scene_id or args.scene_filter or args.list_scenes:
        config.DATASET.SCENE_SELECTION.MAX_SCENES = -1

    scenes = discover_scenes(config)

    if args.scene_filter:
        scenes = [
            scene
            for scene in scenes
            if args.scene_filter in scene.scene_id
        ]

    if not scenes:
        raise SystemExit(
            "No Habitat scenes found from the provided config."
        )

    if args.list_scenes:
        for idx, scene in enumerate(scenes):
            print(
                f"{idx:04d}  "
                f"{scene.dataset_name}  "
                f"{scene.scene_id}"
            )
        return

    # ------------------------------------------------------------
    # Select scene
    # ------------------------------------------------------------

    if args.scene_id:
        matches = [
            scene
            for scene in scenes
            if scene.scene_id == args.scene_id
        ]

        if not matches:
            raise SystemExit(
                f"Scene id not found: {args.scene_id}"
            )

        scene = matches[0]

    else:
        if (
            args.scene_index < 0
            or args.scene_index >= len(scenes)
        ):
            raise SystemExit(
                f"Scene index {args.scene_index} is out of "
                f"range for {len(scenes)} scenes."
            )

        scene = scenes[args.scene_index]

    num_trajectories = (
        args.num_trajectories
        or int(
            config.TRAJECTORY_SETTINGS.TRAJECTORIES_PER_SCENE
        )
    )

    settings = TrajectorySettings.from_config(config)

    # ------------------------------------------------------------
    # Create Habitat environment
    # ------------------------------------------------------------

    env = HabitatCollectionEnv(config, scene)
    env.estimate_max_path_distance(3000)

    # ------------------------------------------------------------
    # The automatic labels require HM3D semantic annotations.
    # ------------------------------------------------------------

    semantic_scene = env.sim.semantic_scene

    if len(semantic_scene.objects) == 0:
        env.close()

        raise SystemExit(
            "\nSelected scene has no HM3D semantic annotations.\n"
            f"Scene: {scene.scene_id}\n\n"
            "Use one of the HM3D-Sem annotated scenes."
        )

    if len(semantic_scene.regions) == 0:
        env.close()

        raise SystemExit(
            "\nSelected scene has semantic objects but no "
            "semantic regions were loaded.\n"
            f"Scene: {scene.scene_id}\n"
        )

    output_dir = unique_run_dir(
        args.output_dir
    )

    rerun_logger = None

    if not args.no_rerun:
        rerun_logger = RerunLogger(
            args.rerun_addr,
            spawn=args.rerun_spawn,
        )

    print()
    print(f"Loaded scene: {scene.scene_id}")
    print(f"Logging to: {output_dir}")

    print(
        "Semantic annotations: "
        f"{len(semantic_scene.objects)} objects, "
        f"{len(semantic_scene.regions)} regions"
    )

    print(
        f"Generating {num_trajectories} trajectories "
        f"(min path distance "
        f"{settings.min_path_distance}m)"
    )

    print(
        "Transition labels: "
        f"stable_frames="
        f"{args.transition_stable_frames}, "
        f"half_width="
        f"{args.transition_half_width:.2f}m"
    )

    print()

    completed = 0

    try:

        for idx in range(num_trajectories):

            # ====================================================
            # 1. Sample trajectory
            # ====================================================

            path_points, geodesic_distance = (
                sample_random_path(
                    env,
                    settings.min_path_distance,
                    max_attempts=args.max_attempts,
                )
            )

            if path_points is None:
                print(
                    f"[{idx:04d}] Failed to find a suitable "
                    f"path after {args.max_attempts} attempts, "
                    "skipping."
                )
                continue

            env.reset_frame_counter()

            trajectory_dir = (
                output_dir / f"{idx:04d}"
            )

            logger = TrajectoryLogger(
                trajectory_dir,
                scene,
                save_frames=args.save_frames,
            )

            if rerun_logger is not None:
                rerun_logger.new_trajectory(idx)

            print(
                f"[{idx:04d}] "
                f"path length "
                f"{geodesic_distance:.2f}m over "
                f"{len(path_points)} waypoints"
            )

            # ====================================================
            # 2. Traverse and collect per-frame GT region IDs
            # ====================================================

            frame_count = 0
            frame_records = []

            try:

                for result in traverse_path(
                    env,
                    path_points,
                    settings,
                ):

                    record = logger.log(
                        result,
                        "trajectory",
                    )

                    if (rerun_logger is not None and result.frame_index % 5 == 0):
                        rerun_logger.log(
                            result,
                            "trajectory",
                        )

                    frame_count += 1

                    position = record[
                        "pose"
                    ]["position"]

                    progress = record[
                        "pose"
                    ]["path_progress"]

                    # region_id is generated inside env.observe()
                    # from the downward semantic GT sensor.
                    region_id = result.pose.get(
                        "region_id"
                    )

                    frame_records.append(
                        {
                            "frame_index": int(
                                record["frame_index"]
                            ),

                            "path_progress": float(
                                progress
                            ),

                            "position": [
                                float(position[0]),
                                float(position[1]),
                                float(position[2]),
                            ],

                            "region_id": region_id,
                        }
                    )

                    print(
                        f"[{idx:04d}] "
                        f"frame "
                        f"{record['frame_index']:06d} "
                        f"progress="
                        f"{progress:.2f}/"
                        f"{geodesic_distance:.2f}m "
                        f"region={region_id} "
                        f"pos=("
                        f"{position[0]:.2f}, "
                        f"{position[1]:.2f}, "
                        f"{position[2]:.2f})"
                    )

            finally:
                logger.close()

            # ====================================================
            # 3. Automatic semantic room-transition labels
            # ====================================================

            transitions = []

            if frame_records:

                transitions = (
                    save_room_transition_labels(
                        trajectory_dir,
                        frame_records,
                        stable_frames=(
                            args.transition_stable_frames
                        ),
                        transition_half_width_m=(
                            args.transition_half_width
                        ),
                    )
                )

            # ====================================================
            # 4. Print transition summary
            # ====================================================

            if transitions:

                print(
                    f"[{idx:04d}] detected "
                    f"{len(transitions)} semantic "
                    f"room transition(s)"
                )

                for transition in transitions:

                    print(
                        f"    transition "
                        f"{transition['transition_id']}: "
                        f"region "
                        f"{transition['source_region']} "
                        f"-> "
                        f"{transition['target_region']} | "
                        f"start="
                        f"{transition['start_frame']:06d} "
                        f"peak="
                        f"{transition['peak_frame']:06d} "
                        f"end="
                        f"{transition['end_frame']:06d}"
                    )

            else:

                print(
                    f"[{idx:04d}] "
                    "no stable semantic room transition "
                    "detected"
                )

            print(
                f"[{idx:04d}] saved "
                f"{frame_count} frames to "
                f"{trajectory_dir}"
            )

            completed += 1

            print()

    finally:
        env.close()

    print(
        f"Completed "
        f"{completed}/"
        f"{num_trajectories} trajectories."
    )


if __name__ == "__main__":
    main()