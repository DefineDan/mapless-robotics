"""Explore frontiers automatically, recover with LoTIS, then retrieve.

Controls: u rejects a frontier, o dismisses an object, Space pauses/resumes, f forces analysis, g saves,
r retries a stopped recovery, Enter marks the object found and starts retrieval,
q/Escape quits (during recovery it stops the return and pauses exploration).
While a recovery is stopped, arrows move, a/d turn and b turns around before r retries.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .exploration.exploration import AnalysisTrigger, camera_intrinsics
from .exploration.exploration_graph import ExplorationGraph
from .exploration.goal_query import MODEL as QUERY_MODEL, translate_goal
from .exploration.object_search import (
    DEFAULT_MODEL, ObjectSearch, ObjectSearchWorker, OWLv2Detector, approach_position,
)
from .exploration.route_memory import RecordedRoute
from .exploration.view_scan import ViewScan, choose_start
from .explore_navigate import (
    ENTER, QUIT, DriveOutcome, changed_enough, configure_view, drive, encode_retrieval,
    encode_route, make_parser as base_parser, sweep_heading, yaw,
)
from .navigate.depth_navigation import DepthNavigation
from .navigate.odometry import OdometryAgent
from .navigate.recovery_behaviour import RecoveryBehaviour, should_recover
from .navigate.lotis_route_policy import LoTISRoutePolicy, compact_route_ids
from .terminal_keyboard import TerminalKeyboard
from .visualization.keys import action_for_key, key_name
from .visualization.rerun_view import RerunView
from .walk_frontiers import capture, make_sim, make_worker, resolve_scene

if TYPE_CHECKING:
    import habitat_sim
    from .exploration.exploration import ExplorationWorker

logger = logging.getLogger(__name__)

# Consulted only after LoTIS stops a return leg without signalling arrival (endpoint is
# the nearest visible frame, see LoTISRoutePolicy): the leg is still accepted if the frame
# LoTIS last localized the robot at was recorded within this distance of the end frame.
ARRIVAL_BACKSTOP = 1.0  # metres
# A forced return closer than this to its target viewpoint only closes the loop, without driving.
SKIP_RETURN_RADIUS = 0.2  # metres
# Automatic unstick attempts per return leg before the operator takes over (see plan_retry).
AUTO_RETRIES = 2
# turn_left steps (10 degrees each, see make_sim) for b to turn a stopped return around.
TURN_AROUND_STEPS = 18



def plan_retry(outcome, method, previous):
    """Choose an automatic unstick behaviour for a LoTIS return leg stopped mid-route.

    previous lists (kind, fold) of earlier retries on this leg. Returns
    (kind, description, fold, (window centre, back, ahead)) for
    LoTISRoutePolicy.set_window; kind None leaves the leg to the operator and the
    description says why. Frames are indices into the encoded route, like the
    policy diagnostics.
    """
    if outcome not in (DriveOutcome.LOST, DriveOutcome.STALLED):
        return None, f'drive {outcome.value}', None, None
    nearest = getattr(method, 'nearest_frame', None)
    if nearest is None or getattr(method, 'route_lengths', None) is None:
        return None, 'LoTIS never localized the robot on this route', None, None
    if method.in_tail:
        return None, 'stopped in the last metre of the route', None, None
    fold = method.find_fold()
    if fold is None:
        # After a first U-turn retry the window hides the other leg, so the
        # oscillation is gone; reuse that fold while the robot is still matched in it.
        earlier = next((f for _, f in reversed(previous) if f is not None), None)
        if earlier is not None:
            a, b = earlier
            if -1.0 <= method.along_offset(nearest, origin=a) <= method.along_offset(b, origin=a) + 1.0:
                fold = earlier
    if fold is not None:
        a, b = fold
        if not any(kind == 'u_turn' for kind, _ in previous):
            # Hide the other leg: it starts more than half the fold's path length ahead.
            separation = method.along_offset(b, origin=a)
            return ('u_turn', f'U-turn, matches jump between frames {a} and {b}; following the U',
                    fold, (a, 0.5, min(2.0, separation / 2)))
        return ('u_turn', f'U-turn between frames {a} and {b}; cutting across to frame {b}',
                fold, (b, 0.5, 2.0))
    tracking = method.diagnostics.get('status') in ('tracking', 'endpoint_approach')
    if (outcome is DriveOutcome.STALLED and tracking
            and method.diagnostics.get('linear_mps', 0.) > 0):
        return None, ('route in view but the robot is not moving forward, likely an obstacle '
                      '(no automatic behaviour yet)'), None, None
    problem = 'stuck turning in place' if tracking else 'lost the route'
    if not any(kind == 'reorient' for kind, _ in previous):
        return ('reorient', f'{problem} near frame {nearest}; sweeping for the route',
                None, (nearest, 0.5, 2.0))
    return ('reorient', f'{problem} near frame {nearest}; sweeping with a wider window',
            None, (nearest, 1.0, 4.0))


def drive_with_retries(sim, agent, method, target_position, graph, graph_output, view, keyboard,
                       max_steps, step_velocity, turned, goal_yaw=None, turn_around=False,
                       initial_yaw=None):
    """drive() the encoded route, then up to AUTO_RETRIES automatic unstick attempts.

    Each retry turns a full circle to the best view (sweep_heading) and continues the
    same encoding inside a localization window chosen by plan_retry.
    """
    outcome = drive(
        sim, agent, method, target_position, None, graph, graph_output, view, keyboard,
        max_steps, step_velocity, turned, goal_yaw=goal_yaw, turn_around=turn_around,
        goal_radius=None, initial_yaw=initial_yaw,
    )
    previous = []
    for attempt in range(1, AUTO_RETRIES + 1):
        kind, description, fold, window = plan_retry(outcome, method, previous)
        if kind is None:
            if not outcome and outcome is not DriveOutcome.CANCELLED:
                print(f'No automatic retry: {description}')
            break
        previous.append((kind, fold))
        print(f'Auto-retry {attempt}/{AUTO_RETRIES}: {description}')
        view.rr.log('pipeline/return_status', view.rr.TextDocument(
            f'auto_retry {attempt}/{AUTO_RETRIES}: {description}'))
        method.set_window(*window)
        score = sweep_heading(sim, agent, method, view, keyboard)
        if score is None:
            return DriveOutcome.CANCELLED
        if score <= 0:
            print('Sweep found no usable route frame')
            continue
        # Continue the same encoded route; the window stays on until it ends.
        method.resume()
        outcome = drive(
            sim, agent, method, target_position, None, graph, graph_output, view, keyboard,
            max_steps, step_velocity, turned, goal_yaw=goal_yaw, goal_radius=None,
        )
    return outcome


def nudge(sim, key):
    """Move the robot for a key while a LoTIS drive is stopped; True if the key moved it.

    Arrows move, a/d turn, b turns around. Moves are not recorded, and the camera
    pitch stays fixed so LoTIS can still match.
    """
    if key == ord('b'):
        for _ in range(TURN_AROUND_STEPS):
            sim.step('turn_left')
        return True
    action = action_for_key(key_name(key))
    if action is None or action in ('look_up', 'look_down'):
        return False
    sim.step(action)
    return True


def retrieve(sim, agent, method, route, graph, graph_output, view, keyboard, args,
             step_velocity, turned, start_state):
    """Drive the explored route from the start to its end with LoTIS.

    A stopped drive pauses like a stopped return: nudge the robot, r retries LoTIS
    from here with a fresh encoding (and new automatic retries), q quits.
    Returns True on arrival.
    """
    frames = [route.frames[index] for index in route.active]
    states = [route.states[index] for index in route.active]
    # The follower starts at the true start with fresh odometry, which is
    # where the explorer's odometry frame began (route frame 0).
    agent.teleport(start_state)
    agent.set_state(states[0])
    view.set_paused(False)
    view.rr.log('goal', view.rr.Image(frames[-1]).compress(jpeg_quality=90))
    initial_yaw = encode_retrieval(method, frames, states)
    print('Retrieving with LoTIS; press q to stop')
    while True:
        try:
            outcome = drive_with_retries(
                sim, agent, method, states[-1].position, graph, graph_output, view, keyboard,
                args.max_steps, step_velocity, turned, initial_yaw=initial_yaw)
        finally:
            view.clear_route_overlay()
        if outcome:
            print('Retrieval complete')
            return True
        view.set_paused(True)
        print('Retrieval stopped. Arrows move, a/d turn, b turns around, '
              'r retries LoTIS from here, q quits')
        while True:
            view.start_step()
            view.log_camera(capture(sim, agent), time.monotonic(), graph=graph)
            key = keyboard.poll()
            if key in QUIT:
                return False
            if key == ord('r'):
                break
            nudge(sim, key)
        view.set_paused(False)
        encode_retrieval(method, frames, states)
        initial_yaw = None  # retry from the operator's heading


def make_parser():
    parser = base_parser()
    parser.description = __doc__
    parser.set_defaults(exploration='periodic', enable_auto=True, interval=0.5)
    for action in parser._actions:
        if action.dest == 'exploration':
            action.help = 'Analysis trigger (default: periodic)'
        elif action.dest == 'enable_auto':
            action.help = 'Compatibility flag; this launcher always navigates automatically'
    parser.add_argument('--lotis-speed', type=float, help='Maximum LoTIS speed in m/s (default: method config)')
    parser.add_argument('--lotis-cpu-threads', type=int, default=8,
                        help='PyTorch threads when LoTIS runs on CPU (default: 8)')
    parser.add_argument('--seed', type=int, help='Reproduce the random start position and heading')
    parser.add_argument('--start-position', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Start at this navigable Habitat world point')
    parser.add_argument('--start-yaw', type=float, help='Initial yaw in degrees; 0 faces -Z, 90 faces -X')
    parser.add_argument('--scan-views', type=int, default=6,
                        help='Directions analyzed at startup and when goals run out (default: 6; 0 disables)')
    parser.add_argument('--recovery-decision-mode', choices=('confidence', 'graph_distance'),
                        default='graph_distance', help='Rule used to decide when to recover')
    parser.add_argument('--confidence-dist-th', type=float, default=0.5,
                        help='Viewpoint distance threshold in confidence mode (metres)')
    parser.add_argument('--distance-node-to-node-th', type=float, default=-1,
                        help='Route step threshold in graph-distance mode')
    parser.add_argument('--object-query', help='Object or functional request, e.g. "a place to sit"')
    parser.add_argument('--no-qwen', action='store_true',
                        help='Skip request translation and pass --object-query directly to OWLv2')
    parser.add_argument('--query-device', choices=('cpu', 'cuda'), default='cpu',
                        help='Device for one-time Qwen request translation (default: cpu)')
    parser.add_argument('--object-model', default=DEFAULT_MODEL,
                        help='OWLv2 model ID or local model directory')
    parser.add_argument('--object-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--object-threshold', type=float, default=0.2,
                        help='Minimum OWLv2 detection score (default: 0.2)')
    parser.add_argument('--object-image-size', type=int,
                        help='OWLv2 input resolution; multiple of 16, e.g. 512 for CPU')
    parser.add_argument('--object-confirmations', type=int, default=3,
                        help='Consistent captures required before approaching (default: 3)')
    parser.add_argument('--object-consistency-radius', type=float, default=0.75,
                        help='Detection consistency radius in metres (default: 0.75)')
    parser.add_argument('--object-stand-off', type=float, default=1.0,
                        help='Approach distance from the object surface in metres (default: 1)')
    parser.add_argument('--odometry-noise', type=float, default=0.0,
                        help='Odometry noise scale; std is noise*sqrt(metres) in translation and '
                             'noise*sqrt(radians + metres) in yaw (default: 0, ground-truth pose)')
    parser.add_argument('--odometry-seed', type=int,
                        help='Seed for the odometry noise (default: the start seed)')
    parser.add_argument('--map-pixel-report-every', type=int, default=10,
                        help='Print observed exploration-map pixels every N added poses (default: 10; 0 disables)')
    return parser


def save_run(route, graph, run_dir, graph_output):
    graph.save(graph_output)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = route.to_dict()
    metadata['initialization'] = getattr(route, 'initialization', None)
    # Believed (odometry) and simulator positions of every archived frame, for drift plots.
    metadata['odometry_positions'] = [np.asarray(state.position).tolist() for state in route.states]
    metadata['ground_truth_positions'] = [None if pose is None else pose[:3, 3].tolist()
                                          for pose in route.gt_poses]
    if hasattr(route, 'object_search'):
        metadata['object_search'] = route.object_search
    metadata['archive_directory'] = 'archive'
    metadata['retrieval_directory'] = 'frames'
    (run_dir / 'route.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for directory, indices in [('archive', range(len(route.frames))), ('frames', route.active)]:
        destination = run_dir / directory
        destination.mkdir(parents=True, exist_ok=True)
        # Existing runs may have a longer route before a loop was removed.
        # Only generated numeric PNGs are owned by this exporter.
        for path in destination.glob('*.png'):
            if path.stem.isdecimal() and int(path.stem) >= len(indices):
                path.unlink()
        for output_index, frame_index in enumerate(indices):
            path = destination / f'{output_index:06d}.png'
            Image.fromarray(route.frames[frame_index]).save(path)


def explore(
    sim: habitat_sim.Simulator,
    agent: habitat_sim.Agent,
    method: LoTISRoutePolicy,
    worker: ExplorationWorker,
    graph: ExplorationGraph,
    graph_output: Path,
    view: RerunView,
    keyboard: TerminalKeyboard,
    args: argparse.Namespace,
    route: RecordedRoute,
    step_velocity: Callable[[habitat_sim.Simulator, float, float], None],
    turned: Callable[[habitat_sim.AgentState, habitat_sim.AgentState], float],
    object_search: ObjectSearch | None = None,
    object_worker: ObjectSearchWorker | None = None,
) -> bool:
    navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
    recovery_behaviour = RecoveryBehaviour()
    next_control = 0.0
    paused = False
    recovery = None
    forward_retry = None
    last_pose_id = None
    force_analysis = True
    prepared_goal_id = None
    scan = None
    scan_views = args.scan_views
    # Look all the way around once before choosing the first frontier.
    initial_scan = scan_views > 0 and graph.objective_frontier is None
    scan_requested = initial_scan
    discovery_frame = None
    discovery_done = False
    selection_pending = graph.objective_frontier is None
    object_goal = None
    object_arrival_time = None
    object_reached = False
    object_frame_index = None

    def object_metadata():
        if object_search is not None:
            route.object_search = object_search.to_dict()
            route.object_search.update(
                approach_position=None if object_goal is None else object_goal.tolist(),
                reached=object_reached, observing_frame_index=object_frame_index)

    def offer_analysis(observation, state, key):
        nonlocal force_analysis, discovery_frame
        now = time.monotonic()
        if force_analysis or key == ord('f') or worker.trigger.due(observation, now):
            if not worker.busy:
                index = route.append(observation.rgb, state, observation.gt_robot_pose)
                if worker.submit(observation, now, manual=force_analysis or key == ord('f'),
                                 frame_index=index, depth_only=False):
                    discovery_frame = index
                    force_analysis = False

    def collect(wait=False):
        nonlocal last_pose_id, paused, discovery_done, scan, selection_pending, force_analysis
        if wait and worker.future is not None:
            # Drain the one pending capture before altering route topology.
            # poll owns error handling and clears the failed future.
            try:
                worker.future.result()
            except Exception:
                pass
        try:
            previous_objective = graph.objective_frontier
            pose_id, result = worker.poll()
            if pose_id is not None:
                last_pose_id = pose_id
                if graph.poses[pose_id].frame_index == discovery_frame:
                    discovery_done = True
                if scan is not None:
                    scan.analyzed(graph.poses[pose_id].frame_index)
                    if (not scan.full and graph.best_saved(
                            route.available_frames, agent.get_state().position) is not None):
                        scan = None
                if previous_objective is not None and graph.objective_frontier is None:
                    selection_pending = True
                    discovery_done = False
                    force_analysis = True
                invalidated = graph.revalidate_objective(result)
                if invalidated is not None:
                    print(f'F{invalidated.id} failed three quality checks; selecting from a fresh view')
                    selection_pending = True
                    discovery_done = False
                    force_analysis = True

                view.log_analysis(result, pose_id, graph)
                if (object_worker is not None and recovery is None
                        and not route.pending_return and not object_reached):
                    object_worker.submit(graph.poses[pose_id].frame_index, result)
                graph.save(graph_output)
                objective = graph.objective_frontier
                if object_goal is not None:
                    navigator.set_data(object_goal, result.depth, result.observation.camera_pose)
                elif objective is not None:
                    frontier_angle = None
                    if objective.vd is not None:
                        view_direction = np.asarray(objective.vd, dtype=float)
                        if (view_direction.shape == (3,)
                                and np.isfinite(view_direction).all()
                                and np.linalg.norm(view_direction[[0, 2]]) > 0):
                            frontier_angle = float(np.arctan2(
                                -view_direction[0], -view_direction[2]))
                    navigator.set_data(
                        objective.position, result.depth, result.observation.camera_pose,
                        angle_objective=frontier_angle,
                    )
        except Exception:
            paused = True
            view.set_paused(paused)
            if scan is not None:
                scan.waiting_frame = None
            worker.trigger.mode = 'manual'
            logger.exception('Analysis failed; paused. Press f to retry, Space to resume')

    def collect_objects():
        nonlocal object_goal, navigator, scan, scan_requested, object_frame_index, paused
        if object_worker is None:
            return
        try:
            frame_index, result = object_worker.poll()
            if result is None:
                return
            intrinsics = camera_intrinsics(
                result.observation.rgb.shape[1], result.observation.rgb.shape[0], 90.0)
            if recovery is None and not route.pending_return:
                object_search.update(result, intrinsics)
                if object_search.target is not None and object_goal is None:
                    object_frame_index = frame_index
                    object_goal = approach_position(object_search.target.position,
                                                    capture(sim, agent).robot_pose,
                                                    args.object_stand_off)
                    navigator = DepthNavigation(goal_tolerance=graph.visited_distance, max_depth=5.0,
                                                camera_intrinsics=intrinsics)
                    latest = worker.last_result or result
                    navigator.set_data(object_goal, latest.depth, latest.observation.camera_pose)
                    scan = None
                    scan_requested = False
                    print(f'Confirmed {object_search.target.label}; approaching '
                          f'{object_goal.tolist()}. Press o to dismiss this object')
            view.log_object_search(result, object_search)
            object_metadata()
        except Exception:
            paused = True
            view.set_paused(paused)
            logger.exception('Object analysis failed; paused. Press f to retry, Space to resume')

    def restore_view(frame_index):
        state = agent.get_state()
        reference = route.states[frame_index]
        state.rotation = reference.rotation
        state.sensor_states = copy.deepcopy(reference.sensor_states)
        offset = state.position - reference.position
        for sensor in state.sensor_states.values():
            sensor.position = sensor.position + offset
        agent.set_state(state, reset_sensors=False)

    def recover(target_pose, reason, turn_around=True):
        nonlocal recovery, paused, navigator, force_analysis, next_control, forward_retry
        route.pending_return = True
        if object_search is not None:
            object_search.reset_candidate()
            object_worker.discard()
        view.log_navigation_map(DepthNavigation())
        objective = graph.objective_frontier
        if objective is not None:
            view.log_return_target(
                objective, target_pose, route.frames[target_pose.frame_index],
                score=graph.score(objective, agent.get_state().position),
            )
            print(f'Return target: F{objective.id} via P{target_pose.id} '
                  f'(recorded frame {target_pose.frame_index})')

        def leg(frame_ids, tracking, target_position, target_yaw, turn):
            """Drive one leg; returns (success, frame LoTIS last localized the robot at)."""
            endpoint = frame_ids[0] if tracking == 'backward' else frame_ids[-1]
            if len(frame_ids) < 2:
                # A one-frame leg: the robot is already at that recorded frame.
                return True, endpoint
            compact_ids = compact_route_ids(frame_ids, route.states)
            frames = [route.frames[index] for index in compact_ids]
            print(f'Return reference: {len(frame_ids)} recorded frames, {len(compact_ids)} travel frames')
            travel_ids = compact_ids[::-1] if tracking == 'backward' else compact_ids
            initial_yaw = None
            if turn:
                origin = route.states[travel_ids[0]].position
                for index in travel_ids[1:]:
                    delta = route.states[index].position - origin
                    if np.linalg.norm(delta[[0, 2]]) >= 0.15:
                        initial_yaw = float(np.arctan2(-delta[0], -delta[2]))
                        break
            encoded_ids = [compact_ids[index] for index in encode_route(
                method, frames, tracking, positions=[route.states[index].position for index in compact_ids])]
            goal = frames[0] if tracking == 'backward' else frames[-1]
            view.log('goal', view.rr.Image(goal).compress(jpeg_quality=90))
            print(f'LoTIS {tracking} to frame '
                  f'{frame_ids[0] if tracking == "backward" else frame_ids[-1]}; press q to stop')
            try:
                outcome = drive_with_retries(
                    sim, agent, method, target_position, graph, graph_output, view, keyboard,
                    args.max_steps, step_velocity, turned, goal_yaw=target_yaw,
                    turn_around=turn, initial_yaw=initial_yaw,
                )
            finally:
                view.clear_route_overlay()
            success = bool(outcome)
            # Close the loop where LoTIS last localized the robot, not at the nominal endpoint.
            # Every arrival signal implies that frame is near the end (see LoTISRoutePolicy).
            nearest = getattr(method, 'nearest_frame', None)
            reached = encoded_ids[nearest] if nearest is not None else endpoint
            if not success:
                # Recorded poses of both frames, so live odometry drift does not enter.
                gap = np.linalg.norm((route.states[reached].position
                                      - route.states[endpoint].position)[graph.plane_idx])
                if nearest is None or gap > ARRIVAL_BACKSTOP:
                    return False, None
                print(f'LoTIS stopped without arriving; last localized at frame {reached}, '
                      f'{gap:.2f} m from the end frame is within the backstop, accepting')
            return True, reached

        def pose_at(frame_index):
            return next((pose for pose in graph.poses.values()
                         if pose.frame_index == frame_index), None)

        # The return only matters for the route tree when the robot already stands at the
        # viewpoint (e.g. a goal rejected right next to it): close the loop, skip driving.
        distance = np.linalg.norm((agent.get_state().position
                                   - target_pose.pose[:3, 3])[graph.plane_idx])
        if distance <= SKIP_RETURN_RADIUS:
            print(f'Already {distance:.2f} m from P{target_pose.id}; closing the loop at '
                  f'frame {target_pose.frame_index} without driving')
            success, reached, forward, retry_forward = True, target_pose.frame_index, [], False
        else:
            common, forward = route.connection(target_pose.frame_index)
            common_pose = pose_at(common)
            backward = route.active[route.active.index(common):]
            view.log_return_legs([route.states[index].position for index in backward],
                                 [route.states[index].position for index in forward], common)
            retry_forward = forward_retry == target_pose.frame_index
            if retry_forward:
                success, reached = True, route.active[-1]
            else:
                success, reached = leg(
                    backward, 'backward', route.states[common].position,
                    yaw(common_pose.pose) if common_pose is not None else None, turn_around)
        if success:
            if not retry_forward:
                restore_view(reached)
            if len(forward) > 1:
                # Keep a valid route to where the backward leg ended if the forward leg is stopped.
                route.close_loop(reached, reason)
                graph.prune_after_return(route.active, pose_at(reached))
                success, reached = leg(forward, 'forward', target_pose.pose[:3, 3],
                                       yaw(target_pose.pose), False)
                forward_retry = None if success else target_pose.frame_index
        if not success:
            recovery = (target_pose, reason)
            paused = True
            view.set_paused(paused)
            if objective is not None:
                view.log_return_target(
                    objective, target_pose, route.frames[target_pose.frame_index],
                    score=graph.score(objective, agent.get_state().position), status='Return paused',
                )
            print('Return incomplete; archived routes have been retained. Arrows move, a/d turn, '
                  'b turns around, r retries LoTIS from here, q quits')
            return
        print(f'Closed loop at frame {reached} (target {target_pose.frame_index})')
        route.close_loop(reached, reason)
        route.pending_return = False
        graph.prune_after_return(route.active, pose_at(reached))
        restore_view(reached)
        # navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
        recovery = None
        forward_retry = None
        paused = False
        view.set_paused(paused)
        next_control = 0.0
        force_analysis = True
        # Start local navigation immediately with live Habitat depth. Estimated
        # depth is initialized by the forced capture, even in manual mode.
        current = capture(sim, agent)
        objective = graph.objective_frontier
        # if objective is not None and current.depth is not None:
        #     navigator.set_data(objective.position, current.depth, current.camera_pose)
        view.clear_return_target()
        graph.save(graph_output)
        if worker.last_result is not None and last_pose_id is not None:
            view.log_analysis(worker.last_result, last_pose_id, graph)
        print(f'Returned to P{target_pose.id}; continuing to '
              f'{f"F{objective.id}" if objective is not None else "the next frontier"}; '
              'local depth refresh requested')

    print('Auto exploration | u frontier unreachable | o dismiss object | Space pause | f analyze | g save | '
          'Enter object found/retrieve | q quit')
    if object_search is not None:
        print(f'OWLv2 searching for: {", ".join(object_search.labels)}; '
              'arrival pauses for Enter/retrieval')
    while True:
        observation = capture(sim, agent)
        state = agent.get_state()
        view.start_step()
        view.log_camera(observation, time.monotonic(), graph=graph)
        if recovery is None and changed_enough(
                state, route.states[route.active[-1]] if route.active else None, turned):
            index = route.append(observation.rgb, state, observation.gt_robot_pose)
            view.log('last_recorded_frame', view.rr.Image(route.frames[index]))
        collect()
        collect_objects()
        key = keyboard.poll()
        if key in QUIT:
            return False
        if key in ENTER:
            if recovery is not None or route.pending_return:
                print('Finish recovery before retrieval; r retries a stopped return, '
                      'or find an alternative with f and resume with Space')
                continue
            collect(wait=True)
            route.append(observation.rgb, state, observation.gt_robot_pose)
            if object_search is not None and not object_reached:
                object_search.status = 'Found (operator confirmed)'
                object_metadata()
            return True
        if key == ord('g'):
            save_run(route, graph, args.run_dir, graph_output)
        if key == ord('r') and recovery is not None:
            # Show the retried return; recover() pauses the view again on failure.
            view.set_paused(False)
            recover(*recovery, turn_around=False)
            continue
        if key == ord(' ') and recovery is None:
            paused = True if object_reached else not paused
            view.set_paused(paused)
            print('Paused' if paused else 'Resumed')
            if not paused and graph.objective_frontier is None:
                force_analysis = True
                discovery_done = False
        if recovery is not None:
            # A stopped return waits for the operator: nudge the robot, then r retries from here.
            nudge(sim, key)
            continue
        if key == ord('o') and object_goal is not None:
            collect(wait=True)
            object_worker.discard()
            object_search.reject()
            object_goal = None
            object_arrival_time = None
            object_reached = False
            object_frame_index = None
            paused = False
            view.set_paused(paused)
            navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
            recovery_behaviour.reset()
            view.clear_object_goal()
            force_analysis = True
            discovery_done = False
            selection_pending = graph.objective_frontier is None
            object_metadata()
            print('Object dismissed at this position; resuming frontier exploration')
            continue
        if key == ord('u') and graph.objective_frontier is not None:
            collect(wait=True)
            rejected = graph.objective_frontier
            if rejected is not None:
                graph.retire(rejected, 'unreachable')
            route.pending_return = True
            selection_pending = True
            force_analysis = True
            discovery_done = False
            # navigator = DepthNavigation()
            view.log_navigation_map(navigator)
            print('Goal rejected; refreshing frontiers here before considering a return')
            graph.save(graph_output)
            continue
        action = action_for_key(key_name(key))
        if action is not None:
            paused = True
            view.set_paused(paused)
            scan = None
            scan_requested = False
            initial_scan = False
            discovery_done = False
            force_analysis = True
            object_arrival_time = None
            object_reached = False
            if object_goal is not None:
                object_search.status = 'Confirmed; approaching'
            if object_search is not None and object_search.target is None:
                object_search.reset_candidate()
                object_worker.discard()
            sim.step(action)
            continue
        if object_goal is not None:
            now = time.monotonic()
            if not paused and not object_reached and now >= next_control:
                if args.depth_source == 'habitat' and observation.depth is not None:
                    navigator.set_data(object_goal, observation.depth, observation.camera_pose)
                distance = np.linalg.norm((object_goal - state.position)[[0, 2]])
                linear = angular = 0.0
                if distance <= navigator.goal_tolerance:
                    delta = object_search.target.position - state.position
                    heading = np.arctan2(-delta[0], -delta[2])
                    error = (heading - yaw(observation.robot_pose) + np.pi) % (2 * np.pi) - np.pi
                    angular = float(np.clip(navigator.angular_gain * error,
                                            -navigator.max_angular_speed, navigator.max_angular_speed))
                    if abs(error) <= np.deg2rad(15):
                        angular = 0.0
                        if object_arrival_time is None:
                            object_arrival_time = observation.timestamp
                            force_analysis = True
                            object_search.status = 'At approach point; verifying object'
                        elif object_search.last_seen >= object_arrival_time:
                            object_reached = True
                            paused = True
                            view.set_paused(paused)
                            object_search.status = 'Found; press Enter to retrieve or o to dismiss'
                            route.append(observation.rgb, state, observation.gt_robot_pose)
                            print(f'Found {object_search.target.label}; paused. '
                                  'Enter starts LoTIS retrieval, o dismisses the object')
                    else:
                        object_arrival_time = None
                else:
                    object_arrival_time = None
                    if (navigator.goal is not None and len(navigator.trajectory)
                            and np.linalg.norm(navigator.trajectory[-1] - object_goal[[0, 2]])
                            <= navigator.goal_tolerance):
                        linear, angular = navigator.get_command(observation.camera_pose)
                        object_search.status = 'Confirmed; approaching'
                    else:
                        object_search.status = 'No local path to object approach; o dismisses'
                if not paused:
                    navigator.get_sim_action(linear, angular, args.integration_time)(sim)
                view.log_navigation_map(navigator)
                next_control = now + args.integration_time
            offer_analysis(observation, state, key)
            view.log_object_goal(object_search, object_goal)
            object_metadata()
            view.rr.log('pipeline/status', view.rr.TextDocument(
                f'Object: {object_search.status} | {len(route.active)} retrieval frames'))
            continue
        if not paused and (scan_requested or scan is not None):
            if scan is None:
                collect(wait=True)
                scan = ViewScan(yaw(observation.robot_pose), scan_views, full=initial_scan)
                if not initial_scan:
                    # The current heading was just analyzed without candidates.
                    scan.completed_views = 1
                initial_scan = False
                scan_requested = False
                print(f'Scanning {scan_views} directions before selecting a frontier')
            if scan.complete:
                scan = None
                # Every direction is analyzed; select without another discovery frame.
                discovery_done = True
                if graph.best_saved(route.available_frames, agent.get_state().position) is None:
                    paused = True
                    view.set_paused(paused)
                    print('Scan finished with no available frontier; paused. '
                          'Move to another viewpoint, then Space to scan again')
            else:
                now = time.monotonic()
                angular, ready = scan.command(yaw(observation.robot_pose))
                if ready and not worker.busy:
                    index = route.append(observation.rgb, state, observation.gt_robot_pose)
                    if worker.submit(observation, now, manual=True, frame_index=index):
                        scan.waiting_frame = index
                elif angular and now >= next_control:
                    navigator.get_sim_action(0.0, angular, args.integration_time)(sim)
                    next_control = now + args.integration_time
                view.log('pipeline/status', view.rr.TextDocument(
                    f'Scanning: {scan.completed_views}/{scan.views} directions analyzed'))
                continue
        objective = graph.objective_frontier
        if objective is not None:
            graph.eval_objective(observation.robot_pose, observation.camera_pose)
            if graph.objective_frontier is None:
                collect(wait=True)
                objective = None
                # navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
                force_analysis = True
                discovery_done = False
                selection_pending = True
        if not paused:
            if objective is None and (not selection_pending or discovery_done):
                objective = graph.select_saved(route.available_frames, observation.camera_pose)
                selection_pending = False
            if objective is not None and objective.id != prepared_goal_id:
                prepared_goal_id = objective.id
                recovery_behaviour.reset()
                target, distance_node_to_node = graph.observer(
                    objective, route.available_frames, current_position=state.position)
                print(f'Selected F{objective.id}, utility score {graph.score(objective, agent.get_state().position):.3f}')
                if worker.last_result is not None and last_pose_id is not None:
                    view.log_analysis(worker.last_result, last_pose_id, graph)

                print(f'Distance to target viewpoint: {distance_node_to_node:.3f}; ')
                if (should_recover(
                        objective, target, state.position,
                        distance_node_to_node=distance_node_to_node,
                        mode=args.recovery_decision_mode,
                        confidence_dist_th=args.confidence_dist_th,
                        distance_node_to_node_th=args.distance_node_to_node_th,
                        plane_idx=graph.plane_idx)
                        or route.pending_return):
                    collect(wait=True)
                    recover(target, 'unreachable' if route.pending_return else 'loop_closure')
                    continue
                # Rebuild local occupancy in the selected viewpoint. Never use
                # the last scan's depth for a goal observed in another heading.
                restore_view(target.frame_index)
                observation = capture(sim, agent)
                state = agent.get_state()
                # navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
                # if observation.depth is not None:
                #     navigator.set_data(objective.position, observation.depth, observation.camera_pose)
                force_analysis = True
        if not paused and objective is None and discovery_done and not worker.busy and scan_views > 0:
            scan_requested = True
            continue
        now = time.monotonic()
        # Attach every capture to its exact archived image, including forced
        # analysis while stationary. The worker keeps just one capture in flight.
        if force_analysis or key == ord('f') or worker.trigger.due(observation, now):
            if not worker.busy:
                index = route.append(observation.rgb, state, observation.gt_robot_pose)
                if worker.submit(observation, now, manual=force_analysis or key == ord('f'),
                                 frame_index=index, depth_only=False):
                    discovery_frame = index
                    force_analysis = False
        if not paused and objective is not None and now >= next_control:
            # if observation.depth is not None:
            #     navigator.set_data(objective.position, observation.depth, observation.camera_pose)
            if navigator.goal is not None:

                trajectory_size = navigator.get_size_trajectory()
                if recovery_behaviour.is_stuck(observation, trajectory_size):
                    rejected = graph.objective_frontier
                    if rejected is not None:
                        graph.retire(rejected, 'unreachable')
                    route.pending_return = True
                    selection_pending = True
                    force_analysis = True
                    discovery_done = False
                    # navigator = DepthNavigation()
                    view.log_navigation_map(navigator)
                    print('Goal rejected; refreshing frontiers here before considering a return')

                linear, angular = navigator.get_command(observation.camera_pose)
                navigator.get_sim_action(linear, angular, args.integration_time)(sim)
                view.log_navigation_map(navigator)
            next_control = now + args.integration_time
        view.log('pipeline/status', view.rr.TextDocument(
            f"{'Paused' if paused else 'Exploring'} | "
            f"goal {f'F{objective.id}' if objective else 'none'} | "
            f'{len(route.active)} retrieval frames | '
            f"target check: {graph.objective_check.get('status', 'pending')}"))


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.viewer != 'rerun' or args.exploration == 'off':
        parser.error('This pipeline requires --viewer rerun and frontier analysis enabled')
    if args.lotis_cpu_threads <= 0:
        parser.error('--lotis-cpu-threads must be positive')
    if args.lotis_speed is not None and (not np.isfinite(args.lotis_speed) or args.lotis_speed <= 0):
        parser.error('--lotis-speed must be finite and positive')
    if args.max_steps <= 0 or not np.isfinite(args.integration_time) or args.integration_time <= 0:
        parser.error('--max-steps and --integration-time must be positive')
    if not np.isfinite(args.odometry_noise) or args.odometry_noise < 0:
        parser.error('--odometry-noise must be finite and non-negative')
    if args.map_pixel_report_every < 0:
        parser.error('--map-pixel-report-every must be nonnegative')
    try:
        scene = resolve_scene(args.scene, args.index, args.scenes_dir)
        graph = ExplorationGraph(
            args.merge_distance,
            auto_select_objective=False,
            enable_ground_truth_occupancy=True,
            ground_truth_occupancy_report_every=args.map_pixel_report_every,
        )
        trigger = AnalysisTrigger(args.exploration, args.interval,
                                  args.movement_distance, args.movement_angle)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.scan_views == 1 or args.scan_views < 0:
        parser.error('--scan-views must be 0 (disabled) or at least 2')
    object_search = None
    if args.object_query is not None:
        try:
            object_search = ObjectSearch(args.object_query, args.object_confirmations,
                                         args.object_consistency_radius)
        except ValueError as error:
            parser.error(str(error))
        if not np.isfinite(args.object_threshold) or not 0 < args.object_threshold < 1:
            parser.error('--object-threshold must be between 0 and 1')
        if args.object_image_size is not None and (args.object_image_size < 32 or args.object_image_size % 16):
            parser.error('--object-image-size must be a multiple of 16, at least 32')
        if not np.isfinite(args.object_stand_off) or args.object_stand_off <= 0:
            parser.error('--object-stand-off must be finite and positive')
    args.run_dir = args.run_dir or Path('runs/auto_explore_navigate') / datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')
    graph_output = args.graph_output or args.run_dir / 'graph.json'
    route = RecordedRoute()
    sim = worker = object_worker = method = view = None
    try:
        # Load Habitat before importing LoTIS/Triton (LLVM initialization order).

        sim = make_sim(str(scene), with_depth=True,
                       camera_pitch=args.camera_pitch)
        # sim = make_sim(str(scene), with_depth=args.depth_source == 'habitat',
        #                camera_pitch=args.camera_pitch)
        if not sim.pathfinder.is_loaded:
            import habitat_sim
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError('Could not build a walkable navmesh')
        agent = sim.initialize_agent(0)
        state = agent.get_state()
        
        try:
            position, heading, start_seed = choose_start(
                sim.pathfinder, args.seed, args.start_position, args.start_yaw)
        except ValueError as error:
            parser.error(str(error))
        import magnum as mn
        from habitat_sim.utils.common import quat_from_magnum
        state.position = position
        print(f'heading: {heading}')
        # exit()
        state.rotation = quat_from_magnum(mn.Quaternion.rotation(
            mn.Rad(np.deg2rad(heading)), mn.Vector3(0.0, 1.0, 0.0)))
        agent.set_state(state)
        start_state = agent.get_state()
        # From here on the system only sees odometry; noise 0 passes ground truth through.
        odometry_seed = start_seed if args.odometry_seed is None else args.odometry_seed
        agent = OdometryAgent(agent, args.odometry_noise, odometry_seed)
        route.initialization = dict(position=position.tolist(), yaw_degrees=heading,
                                    camera_pitch_degrees=args.camera_pitch, seed=start_seed,
                                    odometry_noise=args.odometry_noise, odometry_seed=odometry_seed)
        print(f'Start XYZ={position.tolist()}, yaw={heading:.1f} degrees, '
              f'camera pitch={args.camera_pitch:.1f} degrees (positive down), seed={start_seed}')
        print(f'Odometry noise {args.odometry_noise}' if agent.enabled
              else 'Odometry noise 0: using ground-truth pose')
        if object_search is not None:
            if args.no_qwen:
                object_search.labels = (object_search.query,)
                print(f'Qwen disabled; OWLv2 query: {object_search.query}')
            else:
                import torch
                if args.query_device == 'cpu':
                    torch.set_num_threads(args.lotis_cpu_threads)
                print(f'Translating "{object_search.query}" with {QUERY_MODEL} on {args.query_device}')
                try:
                    object_search.labels = translate_goal(object_search.query, args.query_device)
                except ValueError as error:
                    parser.error(str(error))
                print(f'OWLv2 labels: {", ".join(object_search.labels)}; Qwen unloaded')
            route.object_search = object_search.to_dict()
        worker = make_worker(args, graph, trigger)
        from .teach_repeat import load_lotis_config, step_velocity, turned
        from nav_eval.methods.lotis import LoTISNavigationMethod
        kwargs = dict(checkpoint=args.checkpoint, model_config=args.model_config,
                      dinov3_weights=args.dinov3_weights)
        config = (load_lotis_config(args.lotis_config, **kwargs)
                  if args.lotis_config is not None else load_lotis_config(**kwargs))
        import torch
        if str(config.get('device', 'cuda')).startswith('cpu'):
            torch.set_num_threads(args.lotis_cpu_threads)
        print(f"LoTIS device={config.get('device', 'cuda')}, CPU threads={torch.get_num_threads()}")
        method = LoTISNavigationMethod(config)
        method.setup()
        method = LoTISRoutePolicy(method, max_speed=args.lotis_speed)
        print(f'LoTIS speed limit: {min(method.max_speed, method.forward_velocity):.2f} m/s')
        if object_search is not None:
            if args.object_device == 'cpu' or (args.object_device == 'auto' and not torch.cuda.is_available()):
                torch.set_num_threads(args.lotis_cpu_threads)
            print(f'Loading OWLv2 {args.object_model}')
            detector = OWLv2Detector(
                object_search.labels, args.object_model, args.object_device,
                args.object_threshold, args.object_image_size)
            object_worker = ObjectSearchWorker(detector)
            print(f'OWLv2 device={detector.device}; independent object worker')
        view = RerunView(enable_edge_vis=True,
                         enable_navigation_points_vis=args.show_navigation_points)
        configure_view(view, object_search=object_search is not None)
        with TerminalKeyboard() as keyboard:
            found = explore(sim, agent, method, worker, graph, graph_output, view,
                            keyboard, args, route, step_velocity, turned, object_search, object_worker)

            if object_worker is not None:
                object_worker.close()
                object_worker = None
            save_run(route, graph, args.run_dir, graph_output)
            if found and len(route.active) >= 2:
                retrieve(sim, agent, method, route, graph, graph_output, view, keyboard,
                         args, step_velocity, turned, start_state)
    finally:
        try:
            if object_worker is not None:
                object_worker.close()
            if worker is not None:
                pose_id, result = worker.close()
                if pose_id is not None and view is not None:
                    view.log_analysis(result, pose_id, graph)
            if route.frames:
                save_run(route, graph, args.run_dir, graph_output)
                print(f'Saved exploration and retrieval route to {args.run_dir}')
        finally:
            if method is not None:
                method.cleanup()
            if sim is not None:
                sim.close()


if __name__ == '__main__':
    main()
