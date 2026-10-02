"""Teach and repeat with LoTIS while building an OpenFrontier exploration graph.

Run from the repository root with ``python -m mapless_robotics.teach_repeat_frontiers``.
Keep the terminal focused: arrows move, a/d turn, w/s look, f analyzes a frame,
g saves the graph, Enter ends teaching and starts repeating, q quits.
The camera, LoTIS predictions, frontier detections, and 3D graph appear in Rerun.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .exploration import AnalysisTrigger, camera_intrinsics
from .exploration_graph import ExplorationGraph
from .rerun_view import RerunView
from .terminal_keyboard import TerminalKeyboard
from .walk_frontiers import (
    KEY_TO_ACTION,
    build_parser,
    capture,
    make_sim,
    make_worker,
    resolve_scene,
)

logger = logging.getLogger(__name__)
ENTER = {10, 13}
QUIT = {ord("q"), 27}


def make_parser():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(viewer="rerun")
    parser.add_argument("--lotis-config", type=Path, help="LoTIS method YAML")
    parser.add_argument("--checkpoint", type=Path, help="LoTIS checkpoint")
    parser.add_argument("--model-config", type=Path, help="LoTIS model architecture YAML")
    parser.add_argument("--dinov3-weights", type=Path, help="DINOv3 ViT-B/16 checkpoint")
    parser.add_argument("--run-dir", type=Path, help="Directory for frames and graph")
    parser.add_argument("--max-steps", type=int, default=1000, help="Repeat step limit")
    return parser


def configure_view(view):
    import rerun.blueprint as rrb

    view.rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Exploration graph", origin="world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Live camera and LoTIS", origin="camera"),
                    rrb.Spatial2DView(name="Detected frontiers", origin="detections"),
                    rrb.Spatial2DView(name="Last taught frame", origin="last_recorded_frame"),
                    rrb.Spatial2DView(name="Goal", origin="goal"),
                    rrb.Spatial2DView(
                        name="Closest trajectory frame", origin="closest_trajectory_frame"
                    ),
                ),
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        )
    )


def changed_enough(state, recorded, turned, distance=0.25, angle_degrees=15.0):
    if recorded is None:
        return True
    return bool(
        np.linalg.norm(state.position - recorded.position) >= distance
        or np.rad2deg(turned(state, recorded)) >= angle_degrees
    )


def save_frame(frame, frames, states, state, view):
    frames.append(np.array(frame, copy=True))
    states.append(copy.deepcopy(state))
    view.rr.log(
        "last_recorded_frame", view.rr.Image(frames[-1]).compress(jpeg_quality=90)
    )


def poll_analysis(worker, graph, view, graph_output):
    if worker is None:
        return
    try:
        pose_id = worker.poll()
        if pose_id is not None:
            view.log_analysis(worker.last_result, pose_id, graph)
            graph.save(graph_output)
            print(f"Pose {pose_id}: {len(graph.frontiers)} frontiers")
    except Exception:
        worker.trigger.mode = "manual"
        logger.exception("Frontier analysis failed; press f to retry")


def offer_capture(worker, observation, key, graph, graph_output):
    if worker is None:
        return
    if key == ord("g"):
        graph.save(graph_output)
        print(f"Saved {graph_output}")
    else:
        worker.submit(observation, time.monotonic(), manual=key == ord("f"))


def teach(sim, agent, worker, graph, graph_output, view, keyboard, turned):
    frames, states = [], []
    print("Teaching: arrows move | a/d turn | w/s look | f analyze | Enter repeat | q quit")
    while True:
        observation = capture(sim, agent)
        state = agent.get_state()
        view.start_step()
        view.log_camera(observation, time.monotonic())
        poll_analysis(worker, graph, view, graph_output)
        if changed_enough(state, states[-1] if states else None, turned):
            save_frame(observation.rgb, frames, states, state, view)

        key = keyboard.poll()
        if key in QUIT:
            return None
        if key in ENTER:
            # Always retain the precise end of the taught route as the goal.
            if (
                not np.array_equal(state.position, states[-1].position)
                or turned(state, states[-1]) > 1e-6
            ):
                save_frame(observation.rgb, frames, states, state, view)
            return frames, states
        offer_capture(worker, observation, key, graph, graph_output)
        action = KEY_TO_ACTION.get(key)
        if action is not None:
            sim.step(action)


def repeat(
    sim, agent, method, frames, states, worker, graph, graph_output,
    view, keyboard, max_steps, step_velocity,
):
    import rerun as rr

    agent.set_state(states[0])
    goal = states[-1]
    rr.log("goal", rr.Image(frames[-1]), static=True)
    intrinsics = camera_intrinsics(frames[-1].shape[1], frames[-1].shape[0], 90.0)
    print("Encoding taught trajectory...")
    method.reset_episode(
        {
            "rgb_images": [Image.fromarray(frame) for frame in frames],
            "tracking_mode": "forward",
            "goal_point_id": None,
        }
    )
    print("Repeating with LoTIS; press q to stop")
    for step in range(max_steps):
        observation = capture(sim, agent)
        rgb = observation.rgb
        policy_obs = {
            "rgb": rgb, "rgb_pil": Image.fromarray(rgb), "intrinsics": intrinsics
        }
        view.start_step()
        view.log_camera(observation, time.monotonic(), force=True)
        poll_analysis(worker, graph, view, graph_output)
        action = method.act(policy_obs)
        method.visualize(policy_obs)
        key = keyboard.poll(0)
        if key in QUIT:
            print("Repeat stopped")
            return
        offer_capture(worker, observation, key, graph, graph_output)
        step_velocity(sim, action.linear_velocity, action.angular_velocity)
        distance = float(np.linalg.norm(agent.get_state().position - goal.position))
        if distance < 0.5:
            print(f"Reached the goal in {step + 1} steps")
            return
    print(f"Stopped after {max_steps} steps, {distance:.2f} m from the goal")


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.viewer != "rerun":
        parser.error("This combined script uses Rerun; choose --viewer rerun")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    try:
        scene = resolve_scene(args.scene, args.index, args.scenes_dir)
        graph = ExplorationGraph(args.merge_distance)
        trigger = AnalysisTrigger(
            "manual" if args.exploration == "off" else args.exploration,
            args.interval,
            args.movement_distance,
            args.movement_angle,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    run_dir = args.run_dir or (
        Path("runs/teach_repeat_frontiers")
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    graph_output = args.graph_output or run_dir / "graph.json"
    sim = worker = method = view = None
    try:
        # Habitat's EGL/Mesa LLVM must load before LoTIS's Triton LLVM. Reversing
        # this order aborts the process before Python can catch the error.
        print(f"Loading {scene}")
        sim = make_sim(
            str(scene),
            with_depth=args.exploration != "off" and args.depth_source == "habitat",
        )
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

        if args.exploration != "off":
            print("Loading frontier perception models...")
            worker = make_worker(args, graph, trigger)

        # This import is deliberately after make_sim(): it imports Triton.
        from .teach_repeat import load_lotis_config, step_velocity, turned
        from nav_eval.methods.lotis import LoTISNavigationMethod

        config_kwargs = dict(
            checkpoint=args.checkpoint,
            model_config=args.model_config,
            dinov3_weights=args.dinov3_weights,
        )
        config = (
            load_lotis_config(args.lotis_config, **config_kwargs)
            if args.lotis_config is not None
            else load_lotis_config(**config_kwargs)
        )
        print("Loading LoTIS...")
        method = LoTISNavigationMethod(config)
        method.setup()

        view = RerunView()
        configure_view(view)
        with TerminalKeyboard() as keyboard:
            taught = teach(sim, agent, worker, graph, graph_output, view, keyboard, turned)
            if taught is None:
                return
            frames, states = taught
            if len(frames) < 2:
                print("Recorded fewer than two frames; move before pressing Enter")
                return
            frame_dir = run_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{index:03d}.png")
            print(f"Saved {len(frames)} taught frames to {frame_dir}")
            repeat(
                sim, agent, method, frames, states, worker, graph,
                graph_output, view, keyboard, args.max_steps, step_velocity,
            )
    finally:
        if worker is not None:
            try:
                pose_id = worker.close()
                if pose_id is not None and view is not None:
                    view.start_step()
                    view.log_analysis(worker.last_result, pose_id, graph)
            except Exception:
                logger.exception("Final frontier analysis failed")
            graph.save(graph_output)
            print(
                f"Saved {len(graph.poses)} poses and "
                f"{len(graph.frontiers)} frontiers to {graph_output}"
            )
        if method is not None:
            method.cleanup()
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
