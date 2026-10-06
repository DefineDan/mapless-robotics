"""Explore frontiers automatically, recover with LoTIS, then retrieve.

Controls: u rejects a goal, Space pauses/resumes, f forces analysis, g saves,
r retries a stopped recovery, Enter marks the object found and starts retrieval,
q/Escape quits (during recovery it stops the return and pauses exploration).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .exploration.exploration import AnalysisTrigger, camera_intrinsics
from .exploration.route_memory import FrontierMemoryGraph, RecordedRoute
from .exploration.view_scan import ViewScan, choose_start
from .explore_navigate import (
    ENTER, QUIT, changed_enough, configure_view, drive, encode_route,
    make_parser as base_parser, navigate, yaw,
)
from .navigate.depth_navigation import DepthNavigation
from .navigate.lotis_route_policy import LoTISRoutePolicy, compact_route_ids
from .terminal_keyboard import TerminalKeyboard
from .visualization.keys import action_for_key, key_name
from .visualization.rerun_view import RerunView
from .walk_frontiers import capture, make_sim, make_worker, resolve_scene

logger = logging.getLogger(__name__)


def make_parser():
    parser = base_parser()
    parser.description = __doc__
    parser.set_defaults(exploration='periodic', enable_auto=True, interval=0.5, camera_pitch=10.0)
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
    return parser


def save_run(route, graph, run_dir, graph_output):
    graph.save(graph_output)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = route.to_dict()
    metadata['initialization'] = getattr(route, 'initialization', None)
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


def explore(sim, agent, method, worker, graph, graph_output, view, keyboard,
            args, route, step_velocity, turned):
    navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
    next_control = 0.0
    paused = False
    recovery = None
    forward_retry = None
    last_pose_id = None
    force_analysis = True
    prepared_goal_id = None
    scan = None
    scan_views = args.scan_views
    scan_requested = False
    discovery_frame = None
    discovery_done = False
    selection_pending = graph.objective_frontier is None

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
            pose_id, result = worker.poll()
            if pose_id is not None:
                last_pose_id = pose_id
                if graph.poses[pose_id].frame_index == discovery_frame:
                    discovery_done = True
                if scan is not None:
                    scan.analyzed(graph.poses[pose_id].frame_index)
                    if graph.best_saved(route.available_frames) is not None:
                        scan = None
                invalidated = graph.revalidate_objective(result)
                if invalidated is not None:
                    print(f'F{invalidated.id} failed three quality checks; selecting from a fresh view')
                    selection_pending = True
                    discovery_done = False
                    force_analysis = True
                view.log_analysis(result, pose_id, graph)
                graph.save(graph_output)
                objective = graph.objective_frontier
                if objective is not None:
                    navigator.set_data(objective.position, result.depth,
                                       result.observation.camera_pose)
        except Exception:
            paused = True
            if scan is not None:
                scan.waiting_frame = None
            worker.trigger.mode = 'manual'
            logger.exception('Analysis failed; paused. Press f to retry, Space to resume')

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
        view.log_navigation_map(DepthNavigation())
        objective = graph.objective_frontier
        if objective is not None:
            view.log_return_target(
                objective, target_pose, route.frames[target_pose.frame_index],
                score=graph.score(objective),
            )
            print(f'Return target: F{objective.id} via P{target_pose.id} '
                  f'(recorded frame {target_pose.frame_index})')

        def leg(frame_ids, tracking, target_position, target_yaw, turn):
            observation = capture(sim, agent)
            distance = np.linalg.norm((observation.robot_pose[:3, 3]
                                       - target_position)[graph.plane_idx])
            if distance <= 1.0:
                return True
            if len(frame_ids) < 2:
                print('No recorded route; exploration paused')
                return False
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
            encode_route(method, frames, tracking, positions=[route.states[index].position for index in compact_ids])
            goal = frames[0] if tracking == 'backward' else frames[-1]
            view.rr.log('goal', view.rr.Image(goal).compress(jpeg_quality=90))
            print(f'LoTIS {tracking} to frame '
                  f'{frame_ids[0] if tracking == "backward" else frame_ids[-1]}; press q to stop')
            return drive(
                sim, agent, method, target_position, None, graph,
                graph_output, view, keyboard, args.max_steps, step_velocity, turned,
                goal_yaw=target_yaw, turn_around=turn, goal_radius=1.0, initial_yaw=initial_yaw,
            )

        common, forward = route.connection(target_pose.frame_index)
        common_pose = next((pose for pose in graph.poses.values()
                            if pose.frame_index == common), None)
        backward = route.active[route.active.index(common):]
        retry_forward = forward_retry == target_pose.frame_index
        success = retry_forward or leg(
            backward, 'backward', route.states[common].position,
            yaw(common_pose.pose) if common_pose is not None else None, turn_around)
        if success:
            if not retry_forward:
                restore_view(common)
            if len(forward) > 1:
                # Keep a valid route to the junction if the forward leg is stopped.
                route.close_loop(common, reason)
                graph.prune_after_return(route.active, common_pose)
                success = leg(forward, 'forward', target_pose.pose[:3, 3],
                              yaw(target_pose.pose), False)
                forward_retry = None if success else target_pose.frame_index
        if not success:
            recovery = (target_pose, reason)
            paused = True
            if objective is not None:
                view.log_return_target(
                    objective, target_pose, route.frames[target_pose.frame_index],
                    score=graph.score(objective), status='Return paused',
                )
            print('Return incomplete. Press r to retry; archived routes have been retained')
            return
        route.close_loop(target_pose.frame_index, reason)
        route.pending_return = False
        graph.prune_after_return(route.active, target_pose)
        restore_view(target_pose.frame_index)
        navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
        recovery = None
        forward_retry = None
        paused = False
        next_control = 0.0
        force_analysis = True
        # Start local navigation immediately with live Habitat depth. Estimated
        # depth is initialized by the forced capture, even in manual mode.
        current = capture(sim, agent)
        objective = graph.objective_frontier
        if objective is not None and current.depth is not None:
            navigator.set_data(objective.position, current.depth, current.camera_pose)
        view.clear_return_target()
        graph.save(graph_output)
        if worker.last_result is not None and last_pose_id is not None:
            view.log_analysis(worker.last_result, last_pose_id, graph)
        print(f'Returned to P{target_pose.id}; continuing to '
              f'{f"F{objective.id}" if objective is not None else "the next frontier"}; '
              'local depth refresh requested')

    print('Auto exploration | u unreachable | Space pause | f analyze | g save | '
          'Enter object found/retrieve | q quit')
    while True:
        observation = capture(sim, agent)
        state = agent.get_state()
        view.start_step()
        view.log_camera(observation, time.monotonic(), graph=graph)
        if recovery is None and changed_enough(
                state, route.states[route.active[-1]] if route.active else None, turned):
            index = route.append(observation.rgb, state)
            view.rr.log('last_recorded_frame', view.rr.Image(route.frames[index]))
        collect()
        key = keyboard.poll()
        if key in QUIT:
            return False
        if key in ENTER:
            if recovery is not None or route.pending_return:
                print('Finish recovery before retrieval; r retries a stopped return, '
                      'or find an alternative with f and resume with Space')
                continue
            collect(wait=True)
            route.append(observation.rgb, state)
            return True
        if key == ord('g'):
            save_run(route, graph, args.run_dir, graph_output)
        if key == ord('r') and recovery is not None:
            recover(*recovery, turn_around=False)
            continue
        if key == ord(' ') and recovery is None:
            paused = not paused
            print('Paused' if paused else 'Resumed')
            if not paused and graph.objective_frontier is None:
                force_analysis = True
                discovery_done = False
        if recovery is not None:
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
            navigator = DepthNavigation()
            view.log_navigation_map(navigator)
            print('Goal rejected; refreshing frontiers here before considering a return')
            graph.save(graph_output)
            continue
        action = action_for_key(key_name(key))
        if action is not None:
            paused = True
            scan = None
            scan_requested = False
            discovery_done = False
            force_analysis = True
            sim.step(action)
            continue
        if not paused and (scan_requested or scan is not None):
            if scan is None:
                collect(wait=True)
                scan = ViewScan(yaw(observation.robot_pose), scan_views)
                # The current heading was just analyzed without candidates.
                scan.completed_views = 1
                scan_requested = False
                print(f'Scanning {scan_views} directions before selecting a frontier')
            if scan.complete:
                scan = None
                if graph.best_saved(route.available_frames) is None:
                    paused = True
                    print('Scan finished with no available frontier; paused. '
                          'Move to another viewpoint, then Space to scan again')
            else:
                now = time.monotonic()
                angular, ready = scan.command(yaw(observation.robot_pose))
                if ready and not worker.busy:
                    index = route.append(observation.rgb, state)
                    if worker.submit(observation, now, manual=True, frame_index=index):
                        scan.waiting_frame = index
                elif angular and now >= next_control:
                    navigator.get_sim_action(0.0, angular, args.integration_time)(sim)
                    next_control = now + args.integration_time
                view.rr.log('pipeline/status', view.rr.TextDocument(
                    f'Scanning: {scan.completed_views}/{scan.views} directions analyzed'))
                continue
        objective = graph.objective_frontier
        if objective is not None:
            distance = np.linalg.norm((objective.position - state.position)[graph.plane_idx])
            if distance <= graph.visited_distance:
                collect(wait=True)
                if objective.id in graph.frontiers:
                    graph.retire(objective, 'reached')
                objective = None
                navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
                force_analysis = True
                discovery_done = False
                selection_pending = True
        if not paused:
            if objective is None and (not selection_pending or discovery_done):
                objective = graph.select_saved(route.available_frames, preferred_pose_id=last_pose_id)
                selection_pending = False
            if objective is not None and objective.id != prepared_goal_id:
                prepared_goal_id = objective.id
                target = graph.observer(objective, route.available_frames, current_position=state.position)
                distance = np.linalg.norm((target.pose[:3, 3] - state.position)[graph.plane_idx])
                print(f'Selected F{objective.id}, peak gain {graph.score(objective):.3f}')
                if worker.last_result is not None and last_pose_id is not None:
                    view.log_analysis(worker.last_result, last_pose_id, graph)
                if distance > 0.5 or route.pending_return:
                    collect(wait=True)
                    recover(target, 'unreachable' if route.pending_return else 'loop_closure')
                    continue
                # Rebuild local occupancy in the selected viewpoint. Never use
                # the last scan's depth for a goal observed in another heading.
                restore_view(target.frame_index)
                observation = capture(sim, agent)
                state = agent.get_state()
                navigator = DepthNavigation(goal_tolerance=graph.visited_distance)
                if observation.depth is not None:
                    navigator.set_data(objective.position, observation.depth, observation.camera_pose)
                force_analysis = True
        if not paused and objective is None and discovery_done and not worker.busy and scan_views > 0:
            scan_requested = True
            continue
        now = time.monotonic()
        # Attach every capture to its exact archived image, including forced
        # analysis while stationary. The worker keeps just one capture in flight.
        if force_analysis or key == ord('f') or worker.trigger.due(observation, now):
            if not worker.busy:
                index = route.append(observation.rgb, state)
                if worker.submit(observation, now, manual=force_analysis or key == ord('f'),
                                 frame_index=index, depth_only=False):
                    discovery_frame = index
                    force_analysis = False
        if not paused and objective is not None and now >= next_control:
            if observation.depth is not None:
                navigator.set_data(objective.position, observation.depth, observation.camera_pose)
            if navigator.goal is not None:
                linear, angular = navigator.get_command(observation.camera_pose)
                navigator.get_sim_action(linear, angular, args.integration_time)(sim)
                view.log_navigation_map(navigator)
            next_control = now + args.integration_time
        view.rr.log('pipeline/status', view.rr.TextDocument(
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
    try:
        scene = resolve_scene(args.scene, args.index, args.scenes_dir)
        graph = FrontierMemoryGraph(args.merge_distance)
        trigger = AnalysisTrigger(args.exploration, args.interval,
                                  args.movement_distance, args.movement_angle)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.scan_views == 1 or args.scan_views < 0:
        parser.error('--scan-views must be 0 (disabled) or at least 2')
    args.run_dir = args.run_dir or Path('runs/auto_explore_navigate') / datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')
    graph_output = args.graph_output or args.run_dir / 'graph.json'
    route = RecordedRoute()
    sim = worker = method = view = None
    try:
        # Load Habitat before importing LoTIS/Triton (LLVM initialization order).
        sim = make_sim(str(scene), with_depth=args.depth_source == 'habitat',
                       camera_pitch=args.camera_pitch)
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
        state.rotation = quat_from_magnum(mn.Quaternion.rotation(
            mn.Rad(np.deg2rad(heading)), mn.Vector3(0.0, 1.0, 0.0)))
        agent.set_state(state)
        route.initialization = dict(position=position.tolist(), yaw_degrees=heading,
                                    camera_pitch_degrees=args.camera_pitch, seed=start_seed)
        print(f'Start XYZ={position.tolist()}, yaw={heading:.1f} degrees, '
              f'camera pitch={args.camera_pitch:.1f} degrees (positive down), seed={start_seed}')
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
        view = RerunView(enable_edge_vis=True,
                         enable_navigation_points_vis=args.show_navigation_points)
        configure_view(view)
        with TerminalKeyboard() as keyboard:
            found = explore(sim, agent, method, worker, graph, graph_output, view,
                            keyboard, args, route, step_velocity, turned)
            save_run(route, graph, args.run_dir, graph_output)
            if found and len(route.active) >= 2:
                navigate(sim, agent, method, [route.frames[i] for i in route.active],
                         [route.states[i] for i in route.active], None, graph,
                         graph_output, view, keyboard, args.max_steps, step_velocity, turned)
    finally:
        try:
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
