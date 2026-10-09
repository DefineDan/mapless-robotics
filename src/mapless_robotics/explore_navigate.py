"""Explore with LoTIS and an OpenFrontier exploration graph, then navigate to the goal.

Run from the repository root with ``python -m mapless_robotics.explore_navigate``.
Keep the terminal focused: arrows move, a/d turn, w/s look, f analyzes a frame,
g saves the graph, n navigates back to a frontier (type its id), Enter ends
exploration and starts navigating to where it ended, q quits.
The camera, LoTIS predictions, frontier detections, and 3D graph appear in Rerun.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .exploration.exploration import (
    AnalysisTrigger,
    ExplorationWorker,
    Observation,
    camera_intrinsics,
)
from .exploration.exploration_graph import ExplorationGraph, PoseNode
from .visualization.rerun_view import RerunView
from .terminal_keyboard import TerminalKeyboard
from .visualization.keys import action_for_key, key_name
from .walk_frontiers import (
    build_parser,
    capture,
    make_sim,
    make_worker,
    resolve_scene,
)

if TYPE_CHECKING:
    import argparse

    from habitat_sim import Agent, AgentState, Simulator
    from nav_eval.methods.lotis import LoTISNavigationMethod

    Turned = Callable[[AgentState, AgentState], float]
    StepVelocity = Callable[[Simulator, float, float], None]

logger = logging.getLogger(__name__)
ENTER = {10, 13}
QUIT = {ord("q"), 27}
NAVIGATE = ord("n")
GOAL_RADIUS = 1.0  # metres
ALIGN_TOLERANCE = np.deg2rad(20.0)
TAIL_STALL_STEPS = 40  # steps without progress tolerated near the end of a route
ALIGN_GAIN = 2.0  # angular velocity per radian of heading error, like LoTIS's turn_gain


def make_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(viewer="rerun")
    parser.add_argument("--lotis-config", type=Path, help="LoTIS method YAML")
    parser.add_argument("--checkpoint", type=Path, help="LoTIS checkpoint")
    parser.add_argument("--model-config", type=Path, help="LoTIS model architecture YAML")
    parser.add_argument("--dinov3-weights", type=Path, help="DINOv3 ViT-B/16 checkpoint")
    parser.add_argument("--run-dir", type=Path, help="Directory for frames and graph")
    parser.add_argument("--max-steps", type=int, default=1000, help="Navigation step limit")
    return parser


def configure_view(view: RerunView) -> None:
    import rerun.blueprint as rrb

    view.rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Exploration graph", origin="world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Live camera and LoTIS", origin="camera"),
                    rrb.Spatial2DView(name="Detected frontiers", origin="detections"),
                    rrb.Spatial2DView(name="Last recorded frame", origin="last_recorded_frame"),
                    rrb.Spatial2DView(name="LoTIS goal", origin="goal"),
                    rrb.Spatial2DView(name="Return viewpoint", origin="return_target/image"),
                    rrb.TextDocumentView(name="Return target", origin="return_target/status"),
                    rrb.Spatial2DView(
                        name="Closest trajectory frame", origin="closest_trajectory_frame"
                    ),
                ),
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        )
    )


def changed_enough(
    state: AgentState,
    recorded: AgentState | None,
    turned: Turned,
    distance: float = 0.25,
    angle_degrees: float = 15.0,
) -> bool:
    if recorded is None:
        return True

    # print(np.linalg.norm(state.position - recorded.position))
    # print(np.rad2deg(turned(state, recorded)))
    return bool(
        np.linalg.norm(state.position - recorded.position) >= distance
        or np.rad2deg(turned(state, recorded)) >= angle_degrees
    )


def save_frame(
    frame: np.ndarray,
    frames: list[np.ndarray],
    states: list[AgentState],
    state: AgentState,
    view: RerunView,
) -> None:
    frames.append(np.array(frame, copy=True))
    states.append(copy.deepcopy(state))
    view.rr.log(
        "last_recorded_frame", view.rr.Image(frames[-1]).compress(jpeg_quality=90)
    )


def record_frame(
    observation: Observation,
    state: AgentState,
    frames: list[np.ndarray],
    states: list[AgentState],
    view: RerunView,
    turned: Turned,
) -> None:
    if changed_enough(state, states[-1] if states else None, turned):
        save_frame(observation.rgb, frames, states, state, view)


def poll_analysis(
    worker: ExplorationWorker | None,
    graph: ExplorationGraph,
    view: RerunView,
    graph_output: Path,
) -> None:
    if worker is None:
        return
    try:
        pose_id, _ = worker.poll()
        if pose_id is not None:
            view.log_analysis(worker.last_result, pose_id, graph)
            graph.save(graph_output)
            print(f"Pose {pose_id}: {len(graph.frontiers)} frontiers")
    except Exception:
        worker.trigger.mode = "manual"
        logger.exception("Frontier analysis failed; press f to retry")


def offer_capture(
    worker: ExplorationWorker | None,
    observation: Observation,
    key: int,
    graph: ExplorationGraph,
    graph_output: Path,
    frame_index: int | None,
) -> None:
    if worker is None:
        return
    if key == ord("g"):
        graph.save(graph_output)
        print(f"Saved {graph_output}")
    else:
        # The latest recorded frame was taken within a few cm/degrees of this
        # capture, so the pose node can point at it for later navigation.
        worker.submit(
            observation, time.monotonic(), manual=key == ord("f"), frame_index=frame_index
        )


def subsample(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    """Subsample like LoTIS's encoder, so result indices match the frames we keep."""
    if len(frames) <= max_frames:
        return frames
    return [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames, dtype=int)]


def encode_route(
    method: LoTISNavigationMethod, frames: list[np.ndarray], tracking_mode: str,
    positions: list[np.ndarray] | None = None,
) -> list[int]:
    """Encode the route for LoTIS; returns the subsampled indices into frames."""
    indices = np.linspace(0, len(frames) - 1, min(len(frames), method.localizer.max_seq_len), dtype=int)
    if positions is not None and len(positions) != len(frames):
        raise ValueError('Route images and positions must have the same length')
    frames = [frames[index] for index in indices]
    method.reset_episode(
        {
            "rgb_images": [Image.fromarray(frame) for frame in frames],
            "tracking_mode": tracking_mode,
            "goal_point_id": None,
            "route_positions": None if positions is None else [positions[index] for index in indices],
        }
    )
    return [int(index) for index in indices]


def closest_pose(graph: ExplorationGraph, frontier_id: int) -> PoseNode:
    """Of the poses that observed the frontier, the one nearest to it."""
    if frontier_id not in graph.frontiers:
        raise ValueError(f"Unknown frontier F{frontier_id}")
    frontier = graph.frontiers[frontier_id]
    pose_ids = {observation["pose_id"] for observation in frontier.observations}
    poses = [
        graph.poses[pose_id]
        for pose_id in pose_ids
        if graph.poses[pose_id].frame_index is not None
    ]
    if not poses:
        raise ValueError(f"No recorded frame for a pose that observed F{frontier_id}")
    return min(
        poses,
        key=lambda pose: np.linalg.norm(
            np.asarray(pose.pose)[:3, 3] - frontier.position
        ),
    )


def setup_frontier_navigation(
    method: LoTISNavigationMethod,
    graph: ExplorationGraph,
    frontier_id: int,
    frames: list[np.ndarray],
    view: RerunView,
) -> PoseNode | None:
    """Encode the route from the pose that saw the frontier up to now.

    Returns the target pose, or None if we are already there.
    """
    pose = closest_pose(graph, frontier_id)
    route = frames[pose.frame_index :]
    if len(route) < 2:
        print(f"Already at pose {pose.id}, which observed F{frontier_id}")
        return None
    print(
        f"Navigating to pose {pose.id} (frame {pose.frame_index}) for F{frontier_id}, "
        f"{len(route)} frames back"
    )
    view.log_return_target(graph.frontiers[frontier_id], pose, route[0])
    # Backward tracking heads for frame 0 of the route: the target pose.
    encode_route(method, route, "backward")
    view.rr.log("goal", view.rr.Image(route[0]).compress(jpeg_quality=90))
    return pose


def yaw(pose: np.ndarray) -> float:
    """Heading about Habitat's up axis (Y) of a local-to-world pose."""
    pose = np.asarray(pose)
    return float(np.arctan2(pose[0, 2], pose[2, 2]))


def drive(
    sim: Simulator,
    agent: Agent,
    method: LoTISNavigationMethod,
    goal_position: np.ndarray,
    worker: ExplorationWorker | None,
    graph: ExplorationGraph,
    graph_output: Path,
    view: RerunView,
    keyboard: TerminalKeyboard,
    max_steps: int,
    step_velocity: StepVelocity,
    turned: Turned,
    frames: list[np.ndarray] | None = None,
    states: list[AgentState] | None = None,
    goal_yaw: float | None = None,
    turn_around: bool = False,
    goal_radius: float | None = GOAL_RADIUS,
    initial_yaw: float | None = None,
) -> bool:
    """Let LoTIS drive to the goal. False means cancelled or step limit reached.

    With turn_around, first turn 180 degrees in place so the route back is in view.
    With goal_yaw, turn in place on arrival until facing that heading.
    With frames and states, keep recording the route as during manual exploration.
    """
    intrinsics = None
    distance = np.inf
    arrived = False
    target_yaw = None  # while set, turn in place towards it instead of using LoTIS
    progress_position = None
    stalled_steps = 0
    endpoint_best_distance = None
    endpoint_stalled_steps = 0
    for step in range(max_steps):
        iteration_started = time.perf_counter()
        observation = capture(sim, agent)
        rgb = observation.rgb
        if intrinsics is None:
            intrinsics = camera_intrinsics(rgb.shape[1], rgb.shape[0], 90.0)
        policy_obs = {
            "rgb": rgb, "rgb_pil": Image.fromarray(rgb), "intrinsics": intrinsics
        }
        view.start_step()
        view.log_camera(observation, time.monotonic(), force=True)
        poll_analysis(worker, graph, view, graph_output)
        if frames is not None:
            record_frame(observation, agent.get_state(), frames, states, view, turned)
        if step == 0:
            if initial_yaw is not None:
                target_yaw = initial_yaw
            elif turn_around:
                target_yaw = yaw(observation.robot_pose) + np.pi
        distance = float(np.linalg.norm(
            (agent.get_state().position - goal_position)[[0, 2]]
        ))
        if goal_radius is not None and distance < goal_radius and not arrived:
            if goal_yaw is None:
                return True
            arrived = True
            target_yaw = goal_yaw
        if target_yaw is None:
            policy_obs['goal_distance_m'] = distance
            action = method.act(policy_obs)
            method.visualize(policy_obs)
            diagnostics = getattr(method, "diagnostics", None)
            if isinstance(diagnostics, dict):
                diagnostics['cycle_ms'] = round((time.perf_counter() - iteration_started) * 1000, 1)
                diagnostics['control_dt_s'] = 0.1
                diagnostics['goal_distance_m'] = distance
                diagnostics['goal_radius_m'] = goal_radius
                if diagnostics.get('status') == 'endpoint_approach':
                    if endpoint_best_distance is None or distance < endpoint_best_distance - 0.03:
                        endpoint_best_distance = distance
                        endpoint_stalled_steps = 0
                    else:
                        endpoint_stalled_steps += 1
                    if endpoint_stalled_steps >= TAIL_STALL_STEPS:
                        # The endpoint is in view but not getting closer: accept the arrival.
                        print(f"Visible endpoint is not converging: {distance:.2f} m from goal; accepting")
                        method.arrived = True
                else:
                    endpoint_best_distance = None
                    endpoint_stalled_steps = 0
                view.rr.log("pipeline/return_status", view.rr.TextDocument(str(diagnostics)))
                if step % 20 == 0:
                    print(f"LoTIS return: {diagnostics}")
            if getattr(method, "navigation_failed", False) is True:
                print("LoTIS lost the reference route; stopped for retry")
                return False
            if getattr(method, "arrived", False) is True:
                status = diagnostics.get('status') if isinstance(diagnostics, dict) else None
                print(f"LoTIS reached the end of the route ({status}) in {step + 1} steps, "
                      f"{distance:.2f} m from the goal")
                if goal_yaw is None:
                    return True
                arrived = True
                target_yaw = goal_yaw
        key = keyboard.poll(0)
        if key in QUIT:
            print("Navigation stopped")
            return False
        frame_index = len(frames) - 1 if frames else None
        offer_capture(worker, observation, key, graph, graph_output, frame_index)
        if target_yaw is not None:
            error = target_yaw - yaw(observation.robot_pose)
            error = float(np.arctan2(np.sin(error), np.cos(error)))
            if abs(error) >= ALIGN_TOLERANCE:
                step_velocity(sim, 0.0, float(np.clip(ALIGN_GAIN * error, -1.0, 1.0)))
                continue
            target_yaw = None
            if arrived:
                print("Facing the original view")
                return True
            print("Aligned with return route; LoTIS takes over")
            continue
        step_velocity(sim, action.linear_velocity, action.angular_velocity)
        if isinstance(getattr(method, "diagnostics", None), dict):
            position = np.asarray(agent.get_state().position)
            if progress_position is None or np.linalg.norm((position - progress_position)[[0, 2]]) >= 0.1:
                progress_position = position.copy()
                stalled_steps = 0
            else:
                stalled_steps += 1
                if getattr(method, "in_tail", False) and stalled_steps >= TAIL_STALL_STEPS:
                    # Near the end of the route LoTIS rarely recovers; accept the arrival.
                    print(f"LoTIS made no progress near the end of the route; "
                          f"{distance:.2f} m from goal; accepting")
                    if goal_yaw is None:
                        return True
                    arrived = True
                    target_yaw = goal_yaw
                    continue
                if stalled_steps >= 80:
                    print(f"LoTIS return made no translational progress for 80 steps; "
                          f"{distance:.2f} m from goal; stopped for retry")
                    return False
        distance = float(np.linalg.norm((agent.get_state().position - goal_position)[[0, 2]]))
        if goal_radius is not None and distance < goal_radius:
            print(f"Reached the goal in {step + 1} steps")
            if goal_yaw is None:
                return True
            arrived = True
            target_yaw = goal_yaw
    print(f"Stopped after {max_steps} steps, {distance:.2f} m from the goal")
    return False


def navigate_to_frontier(
    sim: Simulator,
    agent: Agent,
    method: LoTISNavigationMethod,
    frames: list[np.ndarray],
    states: list[AgentState],
    worker: ExplorationWorker | None,
    graph: ExplorationGraph,
    graph_output: Path,
    view: RerunView,
    keyboard: TerminalKeyboard,
    max_steps: int,
    step_velocity: StepVelocity,
    turned: Turned,
) -> None:
    text = keyboard.read_line("Navigate to frontier id (Esc cancels): F")
    if text is None:
        return
    try:
        pose = setup_frontier_navigation(method, graph, int(text), frames, view)
    except ValueError as error:
        print(error)
        return
    if pose is None:
        return
    drive(
        sim, agent, method, np.asarray(pose.pose)[:3, 3], worker, graph,
        graph_output, view, keyboard, max_steps, step_velocity, turned,
        frames, states, goal_yaw=yaw(pose.pose), turn_around=True,
    )
    view.clear_return_target()
    print("Back to manual exploration")


def explore(
    sim: Simulator,
    agent: Agent,
    method: LoTISNavigationMethod,
    worker: ExplorationWorker | None,
    graph: ExplorationGraph,
    graph_output: Path,
    view: RerunView,
    keyboard: TerminalKeyboard,
    max_steps: int,
    step_velocity: StepVelocity,
    turned: Turned,
) -> tuple[list[np.ndarray], list[AgentState]] | None:
    frames: list[np.ndarray] = []
    states: list[AgentState] = []
    help_line = (
        "Exploring: arrows move | a/d turn | w/s look | f analyze | "
        "n navigate to frontier | Enter navigate to end | q quit"
    )
    print(help_line)
    while True:
        observation = capture(sim, agent)
        state = agent.get_state()
        view.start_step()
        view.log_camera(observation, time.monotonic())
        poll_analysis(worker, graph, view, graph_output)
        record_frame(observation, state, frames, states, view, turned)

        key = keyboard.poll()
        if key in QUIT:
            return None
        if key in ENTER:
            # Always retain the precise end of the explored route as the goal.
            if (
                not np.array_equal(state.position, states[-1].position)
                or turned(state, states[-1]) > 1e-6
            ):
                save_frame(observation.rgb, frames, states, state, view)
            return frames, states
        if key == NAVIGATE:
            navigate_to_frontier(
                sim, agent, method, frames, states, worker, graph, graph_output,
                view, keyboard, max_steps, step_velocity, turned,
            )
            print(help_line)
            continue
        offer_capture(worker, observation, key, graph, graph_output, len(frames) - 1)
        action = action_for_key(key_name(key))
        if action is not None:
            sim.step(action)


def navigate(
    sim: Simulator,
    agent: Agent,
    method: LoTISNavigationMethod,
    frames: list[np.ndarray],
    states: list[AgentState],
    worker: ExplorationWorker | None,
    graph: ExplorationGraph,
    graph_output: Path,
    view: RerunView,
    keyboard: TerminalKeyboard,
    max_steps: int,
    step_velocity: StepVelocity,
    turned: Turned,
) -> None:
    """Reset to the start of the explored route and let LoTIS drive to its end."""
    agent.set_state(states[0])
    view.rr.log("goal", view.rr.Image(frames[-1]).compress(jpeg_quality=90))
    print("Encoding explored route...")
    encode_route(method, frames, "forward", positions=[state.position for state in states])
    print("Navigating with LoTIS; press q to stop")
    drive(
        sim, agent, method, states[-1].position, worker, graph, graph_output,
        view, keyboard, max_steps, step_velocity, turned,
    )


def main() -> None:
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
        Path("runs/explore_navigate")
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
            camera_pitch=args.camera_pitch,
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
            explored = explore(
                sim, agent, method, worker, graph, graph_output, view, keyboard,
                args.max_steps, step_velocity, turned,
            )
            if explored is None:
                return
            frames, states = explored
            if len(frames) < 2:
                print("Recorded fewer than two frames; move before pressing Enter")
                return
            frame_dir = run_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{index:03d}.png")
            print(f"Saved {len(frames)} explored frames to {frame_dir}")
            navigate(
                sim, agent, method, frames, states, worker, graph, graph_output,
                view, keyboard, args.max_steps, step_velocity, turned,
            )
    finally:
        if worker is not None:
            try:
                pose_id, _ = worker.close()
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
