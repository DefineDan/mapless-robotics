"""Run auto_explore_navigate's workflow with a human carrying the real camera."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import UTC, datetime
import fcntl
import logging
from pathlib import Path
import sys

import numpy as np

from .auto_explore_navigate import explore, make_parser as pipeline_parser, save_run
from .exploration.exploration import AnalysisTrigger, ExplorationPipeline, ExplorationWorker
from .exploration.exploration_graph import ExplorationGraph
from .exploration.route_memory import RecordedRoute
from .handheld_runtime import HandheldRerunView, HandheldRuntime, step_velocity, turned
from .live_io import OdometryFault
from .lotis_config import load_lotis_config, LOTIS_CONFIG
from .navigate.lotis_route_policy import LoTISRoutePolicy
from .terminal_keyboard import TerminalKeyboard

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class NoKeyboard:
    def poll(self, timeout=0.):
        import time
        time.sleep(timeout)
        return -1


def make_parser():
    parser = pipeline_parser()
    parser.description = __doc__
    hidden = {'index', 'scene', 'scenes_dir', 'camera_pitch', 'seed', 'start_position',
              'start_yaw', 'odometry_noise', 'odometry_seed', 'enable_auto', 'viewer'}
    for action in parser._actions:
        if action.dest in hidden:
            action.help = argparse.SUPPRESS
        if action.dest == 'depth_source':
            action.choices = ['realsense', 'estimated']
            action.help = 'Depth for frontier detection, obstacle mapping and object localization: aligned D455 depth or UniK3D prediction (default: realsense)'
    parser.set_defaults(viewer='rerun', depth_source='realsense')
    parser.add_argument('--record-dir', type=Path, help='Alias for --run-dir; use a fresh directory')
    parser.add_argument('--stream-dir', type=Path, default=Path('/tmp/mapless-live'))
    parser.add_argument('--map-span', type=float, default=6., help='Width/height of the camera-centred local map, metres (default: 6)')
    parser.add_argument('--speed', type=float, default=.2, help='Suggested forward walking speed, m/s')
    parser.add_argument('--floor-clearance', type=float, default=.05, help='Ground rejection band around the defined floor, metres (default: 0.05; requires --camera-height to take effect)')
    parser.add_argument('--camera-height', type=float, help='RGB camera optical-centre height above the floor at the first live capture, metres')
    parser.add_argument('--frontier-min-height', type=float, help='Optional minimum frontier height above the defined floor, metres')
    parser.add_argument('--frontier-max-height', type=float, help='Optional maximum frontier height above the defined floor, metres')
    parser.add_argument('--obstacle-min-height', type=float, help='Optional minimum depth-obstacle height above the defined floor, metres')
    parser.add_argument('--obstacle-max-height', type=float, help='Optional maximum depth-obstacle height above the defined floor, metres')
    parser.add_argument('--max-frame-age', type=float, default=.75)
    parser.add_argument('--max-advice-age', type=float, default=2.)
    parser.add_argument('--max-odometry-distance', type=float, default=100., help='Reject odometry beyond this world-origin radius, metres')
    parser.add_argument('--max-odometry-speed', type=float, default=3., help='Reject pose jumps faster than this handheld speed, m/s')
    parser.add_argument('--skip-view-alignment', action='store_true', help='Skip recorded-view yaw/tilt matching at selected viewpoints, recovery endpoints and retrieval start; explicit turns and scans remain active')
    parser.add_argument('--retrieval-start-tolerance', type=float, default=.5)
    parser.add_argument('--lotis-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--display', choices=['rerun', 'terminal'], default='rerun')
    parser.add_argument('--no-spawn', action='store_true', help='Record Rerun data without opening a viewer')
    parser.add_argument('--rrd', type=Path, help='Rerun recording path; default run directory/session.rrd')
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    from .floor_reference import validate_height_band
    if not np.isfinite(args.floor_clearance) or args.floor_clearance < 0:
        parser.error('--floor-clearance must be finite and nonnegative')
    if args.camera_height is not None and (not np.isfinite(args.camera_height) or args.camera_height <= 0):
        parser.error('--camera-height must be finite and positive')
    try:
        floor = 0. if args.camera_height is not None else None
        validate_height_band(floor, args.frontier_min_height, args.frontier_max_height)
        validate_height_band(floor, args.obstacle_min_height, args.obstacle_max_height)
    except ValueError as error:
        parser.error(str(error))
    for name in ['map_span', 'speed', 'max_frame_age', 'max_advice_age', 'max_odometry_distance',
                 'max_odometry_speed', 'retrieval_start_tolerance', 'integration_time', 'interval']:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name.replace('_', '-')+' must be finite and positive')
    if args.lotis_speed is not None and (not np.isfinite(args.lotis_speed) or args.lotis_speed <= 0):
        parser.error('--lotis-speed must be finite and positive')
    if args.max_steps <= 0 or args.lotis_cpu_threads <= 0:
        parser.error('Step limit and CPU thread count must be positive')
    if args.scan_views == 1 or args.scan_views < 0:
        parser.error('--scan-views must be zero or at least two')
    if args.viewer != 'rerun' or args.exploration == 'off':
        parser.error('The shared handheld pipeline requires frontier analysis and Rerun')
    if args.odometry_noise or args.start_position is not None or args.start_yaw is not None or args.scene:
        parser.error('Handheld mode reads live OpenVINS poses; simulator pose options are unavailable')
    if args.run_dir and args.record_dir:
        parser.error('Choose either --run-dir or --record-dir')
    args.run_dir = args.run_dir or args.record_dir or Path('runs/handheld')/datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')
    if (args.run_dir/'route.json').exists():
        parser.error('Run already contains a route; choose a new output directory')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.stream_dir.mkdir(parents=True, exist_ok=True)
    graph_output = args.graph_output or args.run_dir/'graph.json'
    route = RecordedRoute()
    runtime = worker = object_worker = method = graph = None
    input_failed = False
    print('FULL HANDHELD PIPELINE: scan -> frontier exploration -> LoTIS recovery -> object confirmation -> retrieval', flush=True)
    print('Follow the Rerun local map and planned path physically. Terminal: Space pause, f analyze, u reject, o dismiss, r retry, Enter retrieve, q quit.', flush=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.stream_dir/'.guidance.lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another handheld session is using this stream directory; stop it first')
        keyboard = stack.enter_context(TerminalKeyboard()) if sys.stdin.isatty() else NoKeyboard()
        view = HandheldRerunView(spawn=not args.no_spawn and args.display=='rerun',
                                save_path=args.rrd or args.run_dir/'session.rrd', object_search=args.object_query is not None,
                                map_span=args.map_span, depth_source=args.depth_source)
        runtime = HandheldRuntime(args, view, keyboard).start()
        try:
            observation = runtime.capture_observation()
            intrinsics = observation.intrinsics
            route.initialization = dict(source='OpenVINS D455 handheld', session=runtime.control_frame.session,
                                        position=observation.robot_pose[:3, 3].tolist(), camera_intrinsics=intrinsics.tolist(),
                                        align_recorded_views=not args.skip_view_alignment,
                                        depth_source=args.depth_source,
                                        depth_model=args.depth_model if args.depth_source == 'estimated' else None)
            if runtime.floor_reference is not None:
                route.initialization['floor_reference'] = runtime.floor_reference.to_dict()
                route.initialization['navigation_height_filter'] = runtime.navigation_height_options
            graph = ExplorationGraph(
                args.merge_distance, auto_select_objective=False, camera_intrinsics=intrinsics,
                require_arrival_heading=False,
                floor_height=runtime.floor_reference.floor_y if runtime.floor_reference is not None else None,
                minimum_frontier_height=args.frontier_min_height,
                maximum_frontier_height=args.frontier_max_height)
            trigger = AnalysisTrigger(args.exploration, args.interval, args.movement_distance, args.movement_angle)
            import torch
            torch.set_num_threads(args.lotis_cpu_threads)
            object_search = None
            if args.object_query is not None:
                from .exploration.object_search import ObjectSearch
                from .exploration.goal_query import translate_goal
                object_search = ObjectSearch(args.object_query, args.object_confirmations, args.object_consistency_radius)
                runtime.phase = 'translating the object query with Qwen'
                object_search.labels = (object_search.query,) if args.no_qwen else translate_goal(object_search.query, args.query_device)
                route.object_search = object_search.to_dict()
            print(f'DEPTH | {args.depth_source}'+(f' ({args.depth_model})' if args.depth_source == 'estimated' else ''), flush=True)
            runtime.phase = 'loading OpenFrontier'
            from .exploration.frontier import OPENFRONTIER_ROOT, DepthEstimator, FrontierExtractor
            import yaml
            config = yaml.safe_load((args.frontier_config or OPENFRONTIER_ROOT/'config/navigation.yaml').read_text())
            extractor = FrontierExtractor(config, intrinsics, args.frontier_weights or OPENFRONTIER_ROOT/'model_weights/rgbd_11cls.pth')
            estimator = None
            if args.depth_source == 'estimated':
                runtime.phase = f'loading depth model {args.depth_model}'
                estimator = DepthEstimator(intrinsics, args.depth_model)
            worker = ExplorationWorker(ExplorationPipeline(extractor, estimator, mask_invalid_columns=False, invalid_depth_dilation=0), graph, trigger)
            runtime.phase = 'loading LoTIS and DINOv3 for recovery and retrieval'
            config = load_lotis_config(args.lotis_config or LOTIS_CONFIG, checkpoint=args.checkpoint,
                                       model_config=args.model_config, dinov3_weights=args.dinov3_weights)
            config['device'] = ('cuda' if torch.cuda.is_available() else 'cpu') if args.lotis_device=='auto' else args.lotis_device
            sys.path.insert(0, str(PROJECT_ROOT/"external"/"lotis"))
            from nav_eval.methods.lotis import LoTISNavigationMethod
            method = LoTISNavigationMethod(config)
            method.setup()
            method = LoTISRoutePolicy(method, max_speed=args.lotis_speed or args.speed)
            if object_search is not None:
                from .exploration.object_search import ObjectSearchWorker, OWLv2Detector
                runtime.phase = 'loading OWLv2 object detection'
                object_worker = ObjectSearchWorker(OWLv2Detector(object_search.labels, args.object_model, args.object_device,
                                                                 args.object_threshold, args.object_image_size))
            runtime.phase = 'waiting for the shared pipeline to analyze the current view'
            # This is the same state machine called by the simulator launcher.
            found = explore(runtime, runtime.agent, method, worker, graph, graph_output, view,
                            keyboard, args, route, step_velocity, turned, object_search, object_worker)
            if object_worker is not None:
                object_worker.close()
                object_worker = None
            save_run(route, graph, args.run_dir, graph_output)
            if found and len(route.active) >= 2:
                from .explore_navigate import navigate
                view.prepare_retrieval([route.states[i] for i in route.active])
                navigate(runtime, runtime.agent, method, [route.frames[i] for i in route.active],
                         [route.states[i] for i in route.active], None, graph, graph_output,
                         view, keyboard, args.max_steps, step_velocity, turned)
        except KeyboardInterrupt:
            pass
        except OdometryFault as error:
            input_failed = True
            print('STOP: '+str(error), file=sys.stderr, flush=True)
        finally:
            # Stop the instruction heartbeat before draining model workers.
            runtime.close()
            if object_worker is not None:
                object_worker.close()
            if worker is not None:
                pose_id, result = worker.close()
                if pose_id is not None and graph is not None:
                    view.log_analysis(result, pose_id, graph)
            if graph is not None:
                save_run(route, graph, args.run_dir, graph_output)
            if method is not None:
                method.cleanup()
            recording = view.rr.get_global_data_recording()
            if recording is not None:
                recording.flush(timeout_sec=5)
            print(f'Saved shared-pipeline run: {args.run_dir}', flush=True)
    if input_failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
