"""Walk through a Habitat scene and build a pose/frontier exploration graph.

    python -m mapless_robotics.walk_frontiers --index 25
    python -m mapless_robotics.walk_frontiers --scene path/to/scene.basis.glb

    python -m mapless_robotics.walk_frontiers --index 25 --exploration manual --depth-source habitat
    python -m mapless_robotics.walk_frontiers --index 25 --exploration periodic --interval 1

Keys: arrows = move (left/right strafe), a/d = turn, w/s = look up/down,
      f = analyze current capture, g = save graph, q/esc = save and quit.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .exploration.exploration import (
    AnalysisTrigger,
    ExplorationPipeline,
    ExplorationWorker,
    Observation,
    camera_intrinsics,
)
from .exploration.exploration_graph import ExplorationGraph
from .visualization import OpenCVViewer, RerunView

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = Path("data/versioned_data/hm3d-0.2/hm3d/train")


def resolve_scene(scene=None, index=25, scenes_dir=None):
    """Resolve an explicit scene or look up an HM3D numeric directory prefix."""
    if scene is not None:
        path = Path(scene).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scene file not found: {path}")
        return path

    roots = (
        [Path(scenes_dir).expanduser()]
        if scenes_dir is not None
        else [
            Path.cwd() / SCENES_DIR,
            PROJECT_ROOT / SCENES_DIR,
            PROJECT_ROOT / "scene_datasets/hm3d/train",
            PROJECT_ROOT / "external/habitat-sim" / SCENES_DIR,
        ]
    )
    roots = list(dict.fromkeys(root.resolve() for root in roots))
    pattern = f"{index:05d}-*/*.basis.glb"
    for root in roots:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if matches:
            return matches[0]

    searched = "\n".join(f"  {root / pattern}" for root in roots)
    message = (
        f"HM3D scene {index} not found. Searched:\n{searched}\n"
        "Use --scenes-dir /path/to/hm3d/train for an installed HM3D dataset, "
        "or --scene /path/to/scene.glb for another scene."
    )
    example = PROJECT_ROOT.parent / "openfrontier/examples/mv2HUxq3B53.glb"
    if example.is_file():
        message += (
            f"\nAn available example scene can be selected with --scene {example}"
        )
    raise FileNotFoundError(message)


def make_sim(scene_path: str, with_depth=False):
    import habitat_sim

    # 1. Simulator: which scene to load
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    # 2. Sensor: an RGB camera at eye height, attached to the agent
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]  # height, width
    rgb.position = [0.0, 1.5, 0.0]
    rgb.hfov = 90.0

    # 3. Agent: the body carrying the sensor, plus the discrete actions it can take
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    if with_depth:
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = rgb.resolution
        depth.position = rgb.position
        depth.hfov = rgb.hfov
        agent_cfg.sensor_specifications.append(depth)
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        # look_up/look_down tilt only the camera, not the agent body
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=10.0, constraint=80.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=10.0, constraint=80.0)
        ),
    }

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def capture(sim, agent):
    """Read pixels and ground-truth body/sensor poses before the next action."""
    from habitat_sim.utils.common import quat_to_magnum

    pixels = sim.get_sensor_observations()
    timestamp = time.time()
    state = agent.get_state()

    def matrix(sensor_state):
        result = np.eye(4)
        result[:3, :3] = np.asarray(quat_to_magnum(sensor_state.rotation).to_matrix())
        result[:3, 3] = sensor_state.position
        return result

    return Observation(
        pixels["rgb"][..., :3],
        matrix(state),
        matrix(state.sensor_states["rgb"]),
        timestamp,
        pixels.get("depth"),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=int, default=25, help="HM3D scene index")
    parser.add_argument(
        "--viewer",
        choices=["opencv", "rerun"],
        default="opencv",
        help="Visualization backend (default: opencv)",
    )
    parser.add_argument("--scene", help="Path to a .glb scene (overrides --index)")
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        help="HM3D split directory containing 00025-* scene folders (e.g. hm3d/train)",
    )
    parser.add_argument(
        "--exploration",
        choices=["off", "manual", "periodic", "movement"],
        default="manual",
        help="Analysis trigger (default: manual, press f)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Minimum automatic analysis interval in seconds",
    )
    parser.add_argument(
        "--movement-distance",
        type=float,
        default=0.5,
        help="Translation trigger in metres",
    )
    parser.add_argument(
        "--movement-angle", type=float, default=15.0, help="Rotation trigger in degrees"
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=0.5,
        help="Frontier deduplication radius in metres",
    )
    parser.add_argument(
        "--depth-source",
        choices=["estimated", "habitat"],
        default="estimated",
        help="UniK3D (document pipeline) or simulator depth",
    )
    parser.add_argument("--depth-model", default="lpiccinelli/unik3d-vitl")
    parser.add_argument(
        "--frontier-weights", type=Path, help="OpenFrontier RGB-D checkpoint"
    )
    parser.add_argument(
        "--frontier-config", type=Path, help="OpenFrontier YAML configuration"
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        help="JSON snapshot path (default: runs/exploration/<timestamp>/graph.json)",
    )
    return parser


def make_worker(args, graph, trigger):
    import yaml

    from .exploration.frontier import OPENFRONTIER_ROOT, DepthEstimator, FrontierExtractor

    config_path = args.frontier_config or OPENFRONTIER_ROOT / "config/navigation.yaml"
    config = yaml.safe_load(config_path.read_text())
    intrinsics = camera_intrinsics(640, 480, 90.0)
    weights = (
        args.frontier_weights or OPENFRONTIER_ROOT / "model_weights/rgbd_11cls.pth"
    )
    extractor = FrontierExtractor(config, intrinsics, weights)
    estimator = (
        DepthEstimator(intrinsics, args.depth_model)
        if args.depth_source == "estimated"
        else None
    )
    return ExplorationWorker(ExplorationPipeline(extractor, estimator), graph, trigger)


def detection_frame(result, pose_id):
    """Caption a completed capture, keeping labels outside the detector image."""
    frame = cv2.cvtColor(result.marked_rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.copyMakeBorder(
        frame, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
    )
    captured = datetime.fromtimestamp(result.observation.timestamp, tz=UTC).strftime(
        "%H:%M:%S UTC"
    )
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
    return frame


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        graph = ExplorationGraph(args.merge_distance)
        trigger = AnalysisTrigger(
            "manual" if args.exploration == "off" else args.exploration,
            args.interval,
            args.movement_distance,
            args.movement_angle,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        scene = resolve_scene(args.scene, args.index, args.scenes_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    output = (
        args.graph_output
        or Path("runs/exploration")
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        / "graph.json"
    )
    worker = None
    sim = None
    viewer = None
    try:
        if args.exploration != "off":
            print("Loading frontier perception models...")
            worker = make_worker(args, graph, trigger)
        print(f"Loading {scene}")
        sim = make_sim(
            str(scene), with_depth=worker is not None and args.depth_source == "habitat"
        )
        if not sim.pathfinder.is_loaded:
            import habitat_sim

            print("Building a navmesh for keyboard walking...")
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError("Could not build a walkable navmesh for this scene.")

        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)
        viewer = RerunView() if args.viewer == "rerun" else OpenCVViewer()
        status = "Press f to analyze" if worker else "Exploration off"

        print("Arrows move | a/d turn | w/s look | f analyze | g save | q quit")

        if worker:
            print(f"Graph snapshots: {output}")

        with viewer.get_context():
            while True:

                observation = capture(sim, agent)
                viewer.start_step()

                if worker:
                    try:
                        pose_id = worker.poll()
                        if pose_id is not None:
                            viewer.log_analysis(worker.last_result, pose_id, graph)
                            graph.save(output)
                            status = (
                                f"Pose {pose_id} | {len(graph.frontiers)} frontiers"
                            )
                            viewer.set_status(status)

                    except Exception:
                        # Avoid retrying a broken model on every timer tick.
                        trigger.mode = "manual"
                        status = "Analysis failed; see terminal. Press f to retry."
                        logger.exception("Analysis/save failed")

                    worker.submit(observation, time.monotonic())
                viewer.set_status(
                    "Analyzing captured frame..." if worker and worker.busy else status
                )
                viewer.log_camera(observation, time.monotonic(), graph=graph)
                key = viewer.get_keys()
                if key == "quit":
                    break
                if key == "f" and worker:
                    if not worker.submit(observation, time.monotonic(), manual=True):
                        status = "Analysis already running"

                elif key == "g" and worker:
                    graph.save(output)
                    print(f"Saved {output}")
                action = viewer.get_action(key)
                if action is not None:
                    sim.step(action)
    finally:
        try:
            if worker:
                print("Finishing pending analysis and saving graph...")
                try:
                    final_pose_id = worker.close()
                    if final_pose_id is not None and viewer:
                        viewer.start_step()
                        viewer.log_analysis(
                            worker.last_result, final_pose_id, graph
                        )
                except Exception:
                    logger.exception("Final analysis failed")
                graph.save(output)
                print(
                    f"Saved {len(graph.poses)} poses and {len(graph.frontiers)} frontiers to {output}"
                )
        finally:
            if sim is not None:
                sim.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
