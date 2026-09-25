#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from habitat_collection.controllers import KeyboardController, TcpKeyboardController
from habitat_collection.env import HabitatCollectionEnv, discover_scenes, load_config
from habitat_collection.logging_utils import RerunLogger, TrajectoryLogger, unique_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect keyboard Habitat trajectories.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "collection_config.yaml",
        help="Standalone Habitat collection config.",
    )
    parser.add_argument("--scene-id", help="Specific discovered scene_id to load.")
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
        help="Index into the selected scene list when --scene-id is not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "keyboard",
        help="Directory for trajectory.jsonl, metadata.json, and optional frames.",
    )
    parser.add_argument("--save-frames", action="store_true", help="Save RGB/depth data.")
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth sensor even if the config enables it.",
    )
    parser.add_argument(
        "--control",
        choices=("keyboard", "tcp"),
        default="keyboard",
        help="Use local terminal keyboard or a TCP keyboard client.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind in TCP mode. Use 0.0.0.0 for Tailscale reachability.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for TCP mode.")
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
        help="Spawn a local Rerun viewer instead of connecting to --rerun-addr.",
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
    if args.scene_id or args.scene_filter or args.list_scenes:
        config.DATASET.SCENE_SELECTION.MAX_SCENES = -1

    scenes = discover_scenes(config)
    if args.scene_filter:
        scenes = [scene for scene in scenes if args.scene_filter in scene.scene_id]

    if not scenes:
        raise SystemExit("No Habitat scenes found from the provided config.")

    if args.list_scenes:
        for idx, scene in enumerate(scenes):
            print(f"{idx:04d}  {scene.dataset_name}  {scene.scene_id}")
        return

    if args.scene_id:
        matches = [scene for scene in scenes if scene.scene_id == args.scene_id]
        if not matches:
            raise SystemExit(f"Scene id not found: {args.scene_id}")
        scene = matches[0]
    else:
        if args.scene_index < 0 or args.scene_index >= len(scenes):
            raise SystemExit(
                f"Scene index {args.scene_index} is out of range for {len(scenes)} scenes."
            )
        scene = scenes[args.scene_index]

    env = HabitatCollectionEnv(config, scene)
    output_dir = unique_run_dir(args.output_dir)
    logger = TrajectoryLogger(output_dir, scene, save_frames=args.save_frames)
    rerun_logger = None
    if not args.no_rerun:
        rerun_logger = RerunLogger(args.rerun_addr, spawn=args.rerun_spawn)

    print(f"Loaded scene: {scene.scene_id}")
    print(f"Logging to: {output_dir}")
    print("Controls: w forward, a/d turn, s or space record, r reset, q quit")

    if args.control == "tcp":
        controller_cm = TcpKeyboardController(args.host, args.port)
        print(f"Remote client: python remote_keyboard_client.py <tailscale-host> --port {args.port}")
    else:
        controller_cm = KeyboardController()

    try:
        with controller_cm as controller:
            while True:
                command = controller.next_command()
                if command.should_quit:
                    break
                if command.should_reset:
                    result = env.reset_random()
                else:
                    result = env.step(command.action)
                record = logger.log(result, command.label)
                if rerun_logger is not None:
                    rerun_logger.log(result, command.label)
                position = record["pose"]["position"]
                print(
                    f"{record['frame_index']:06d} {record['command']:>10} "
                    f"pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                )
                controller.send_result(record)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
