"""Handheld RealSense exploration, teaching, and LoTIS following with advisory arrows.

Consumes ros_rgbd_bridge snapshots. Never publishes cmd_vel or moves a simulator.
"""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
import json
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

from .handheld_guidance import Advice, RouteRecorder, draw_advice, load_route
from .live_io import atomic_json, guidance_status, read_frame, OdometryGuard, OdometryFault

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LivePolicy:
    """Stateful existing pipeline; called only by the single inference worker."""
    def __init__(self, args, intrinsics, output):
        self.args, self.intrinsics, self.output = args, intrinsics.copy(), output
        self.graph = self.pipeline = self.navigator = self.method = None
        self.finished = False
        if args.mode == 'explore':
            import yaml
            from .exploration.frontier import OPENFRONTIER_ROOT, DepthEstimator, FrontierExtractor
            from .exploration.exploration import ExplorationPipeline
            from .exploration.exploration_graph import ExplorationGraph
            from .navigate.depth_navigation import DepthNavigation
            config = yaml.safe_load((args.frontier_config or OPENFRONTIER_ROOT/'config/navigation.yaml').read_text())
            extractor = FrontierExtractor(config, intrinsics, args.frontier_weights or OPENFRONTIER_ROOT/'model_weights/rgbd_11cls.pth')
            estimator = DepthEstimator(intrinsics, args.depth_model) if args.depth_source == 'estimated' else None
            self.pipeline = ExplorationPipeline(extractor, estimator, mask_invalid_columns=False, invalid_depth_dilation=0)
            self.graph = ExplorationGraph(camera_intrinsics=intrinsics)
            self.navigator = DepthNavigation(camera_intrinsics=intrinsics, linear_speed=args.speed)
        elif args.mode == 'follow':
            from .lotis_config import load_lotis_config, LOTIS_CONFIG
            from .navigate.lotis_route_policy import LoTISRoutePolicy
            sys.path.insert(0, str(PROJECT_ROOT/'external'/'lotis'))
            from nav_eval.methods.lotis import LoTISNavigationMethod
            config = load_lotis_config(args.lotis_config or LOTIS_CONFIG, checkpoint=args.checkpoint,
                                       model_config=args.model_config, dinov3_weights=args.dinov3_weights)
            import torch
            config["device"] = ("cuda" if torch.cuda.is_available() else "cpu") if args.lotis_device == "auto" else args.lotis_device
            base = LoTISNavigationMethod(config)
            base.setup()
            self.method = LoTISRoutePolicy(base, max_speed=args.speed)
            images, positions = load_route(args.route, base.localizer.max_seq_len)
            self.method.reset_episode(dict(rgb_images=images, route_positions=positions,
                                           tracking_mode=args.direction, goal_point_id=None))

    def step(self, frame, frame_index):
        if not np.allclose(frame.intrinsics, self.intrinsics):
            raise ValueError('Camera intrinsics changed; restart the handheld session')
        if self.args.mode == 'explore':
            from .exploration.exploration import Observation
            observation = Observation(frame.rgb, frame.camera_pose, frame.camera_pose, frame.timestamp,
                                      frame.depth if self.args.depth_source == 'realsense' else None)
            result = self.pipeline.process_with_visualization(observation)
            self.graph.add_frontiers_graph(result.frontiers, observation.robot_pose, observation.timestamp,
                                           observation.camera_pose, frame_index, depth=result.depth)
            objective = self.graph.eval_objective(frame.camera_pose, frame.camera_pose)
            self.graph.save(self.output/'graph.json')
            if objective is None:
                return Advice(status='No reachable frontier selected; inspect the scene'), result.marked_rgb
            self.navigator.set_data(objective.position, result.depth, frame.camera_pose)
            linear, angular = self.navigator.get_command(frame.camera_pose)
            return Advice(linear, angular, f'Frontier F{objective.id} | {len(self.graph.frontiers)} frontiers'), result.marked_rgb
        if self.args.mode == 'follow':
            if self.finished:
                return Advice(status='Route following finished; stop and verify arrival'), None
            from PIL import Image
            # Recorded positions define reference arc length only. Live odometry
            # starts in a different world frame, so never compare endpoints directly.
            action = self.method.act(dict(rgb=frame.rgb, rgb_pil=Image.fromarray(frame.rgb), intrinsics=frame.intrinsics))
            status = str(self.method.diagnostics.get('status', 'tracking'))
            if self.method.arrived:
                self.finished = True
                return Advice(status='LoTIS endpoint detected; stop and verify arrival'), None
            if self.method.navigation_failed:
                return Advice(status='Reference route lost; stop and retry'), None
            return Advice(action.linear_velocity, action.angular_velocity, 'LoTIS '+status), None
        return Advice(status='RECORDING: walk the demonstration route' if self.args.mode == 'teach' else 'PREVIEW: synchronized camera and odometry'), None


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preview', 'teach', 'explore', 'follow'], default='explore')
    parser.add_argument('--stream-dir', type=Path, default=Path('/tmp/mapless-live'))
    parser.add_argument('--display', choices=['terminal', 'opencv', 'both'], default='both')
    parser.add_argument('--viewer-python', default='/usr/bin/python3', help='Python with GUI OpenCV; isolated from ML dependencies')
    parser.add_argument('--record-dir', type=Path, help='New route/output directory; never overwrite an existing route')
    parser.add_argument('--route', type=Path, help='Recorded directory containing route.json (follow mode)')
    parser.add_argument('--direction', choices=['forward', 'backward'], default='forward')
    parser.add_argument('--interval', type=float, default=.5, help='Minimum inference interval in seconds; one pending frame at most')
    parser.add_argument('--max-frame-age', type=float, default=.75)
    parser.add_argument('--max-advice-age', type=float, default=2.)
    parser.add_argument('--speed', type=float, default=.2, help='Suggested forward speed, m/s')
    parser.add_argument('--depth-source', choices=['realsense', 'estimated'], default='realsense')
    parser.add_argument('--depth-model', default='lpiccinelli/unik3d-vitl')
    parser.add_argument('--frontier-weights', type=Path)
    parser.add_argument('--frontier-config', type=Path)
    parser.add_argument('--lotis-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--lotis-config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-config', type=Path)
    parser.add_argument('--dinov3-weights', type=Path)
    parser.add_argument('--max-seconds', type=float, help='Optional bounded run for diagnostics')
    return parser


def main():
    dispatch = argparse.ArgumentParser(add_help=False)
    dispatch.add_argument('--mode', default='explore')
    mode, remaining = dispatch.parse_known_args()
    if mode.mode == 'explore':
        from .handheld_auto import main as full_pipeline
        return full_pipeline(remaining)
    parser = make_parser()
    args = parser.parse_args()
    for name in ['interval', 'max_frame_age', 'max_advice_age', 'speed']:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name.replace('_', '-')+' must be finite and positive')
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('--max-seconds must be finite and positive')
    if args.mode == 'follow' and args.route is None:
        parser.error('--mode follow requires --route')
    output = args.record_dir or Path('runs/handheld')/datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')
    recorder = RouteRecorder(output) if args.mode in ('teach', 'explore') else None
    output.mkdir(parents=True, exist_ok=True)
    print('HANDHELD GUIDANCE ONLY | space: pause | f: analyze | g: save | q: quit', flush=True)
    print(f'Mode: {args.mode}; output: {output}', flush=True)
    frame = advice_frame = None
    advice = Advice()
    future = None
    pending_frame = None
    last_recorded_frame = None
    policy = None
    paused = False
    generation = 0
    pending_generation = 0
    last_submit = last_print = 0.
    submitted_key = None
    session = None
    pose_guard = OdometryGuard()
    started = time.monotonic()
    viewer = None
    last_viewer_key = None
    with ExitStack() as stack:
        args.stream_dir.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((args.stream_dir/'.guidance.lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another handheld session is using this stream directory')
        guidance_log = stack.enter_context((output/'guidance.jsonl').open('a'))
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix='handheld-policy'))
        keyboard = None
        if args.display != 'opencv' and sys.stdin.isatty():
            from .terminal_keyboard import TerminalKeyboard
            keyboard = stack.enter_context(TerminalKeyboard())
        if args.display != 'terminal':
            (args.stream_dir/'viewer_key.json').unlink(missing_ok=True)
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(PROJECT_ROOT/'src')+os.pathsep+environment.get('PYTHONPATH', '')
            for name in ['QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR']:
                if 'site-packages/cv2' in environment.get(name, ''):
                    environment.pop(name)
            viewer = subprocess.Popen([args.viewer_python, '-m', 'mapless_robotics.handheld_viewer', '--stream-dir', str(args.stream_dir)], env=environment)
        try:
            while args.max_seconds is None or time.monotonic()-started < args.max_seconds:
                key = keyboard.poll(0) if keyboard else -1
                if viewer is not None:
                    if viewer.poll() is not None:
                        if viewer.returncode != 0:
                            raise RuntimeError('Arrow viewer exited; check its error above or use --display terminal')
                        break
                    try:
                        event = json.loads((args.stream_dir/'viewer_key.json').read_text())
                        if event['id'] != last_viewer_key and 0 <= time.time()-event['time'] < 1.:
                            key, last_viewer_key = event['key'], event['id']
                    except (FileNotFoundError, ValueError, OSError, KeyError):
                        pass
                if key in (ord('q'), 27):
                    break
                if key == ord(' '):
                    paused = not paused
                    generation += 1
                    advice_frame = None
                    submitted_key = None
                if key == ord('g') and recorder:
                    recorder.save()
                try:
                    latest = pose_guard.check(read_frame(args.stream_dir))
                    if session is not None and latest.session != session:
                        raise RuntimeError('Camera/odometry session restarted. Saved route; restart guidance to use the new frame.')
                    session, frame = latest.session, latest
                except OdometryFault:
                    raise
                except (FileNotFoundError, ValueError, OSError):
                    frame = None
                if future is not None and future.done():
                    try:
                        advice, marked = future.result()
                        advice_frame = pending_frame if not paused and pending_generation == generation else None
                    except Exception as error:
                        advice = Advice(status=f'Inference error: {error}')
                        advice_frame = pending_frame
                        paused = True
                        import traceback
                        traceback.print_exc()
                    future = None
                fresh = frame is not None and 0 <= time.time()-frame.written_at <= args.max_frame_age
                if fresh and recorder and not paused:
                    recorder.add(frame)
                    last_recorded_frame = frame
                if (fresh and not paused and future is None and frame.key != submitted_key
                        and (time.monotonic()-last_submit >= args.interval or key == ord('f'))):
                    if policy is None:
                        # Model loading/inference happens in this worker, leaving
                        # UI freshness/STOP indications responsive throughout.
                        def first_step(captured=frame, index=len(recorder.frames)-1 if recorder else None):
                            nonlocal policy
                            policy = LivePolicy(args, captured.intrinsics, output)
                            return policy.step(captured, index)
                        future = executor.submit(first_step)
                    else:
                        future = executor.submit(policy.step, frame, len(recorder.frames)-1 if recorder else None)
                    pending_frame = frame
                    pending_generation = generation
                    submitted_key = frame.key
                    last_submit = time.monotonic()
                status = guidance_status(frame, advice_frame, paused=paused,
                                         max_frame_age=args.max_frame_age, max_advice_age=args.max_advice_age)
                shown = Advice(status=status) if status else advice
                if time.monotonic()-last_print >= .5:
                    guidance_log.write(json.dumps(dict(time=time.time(), linear=shown.linear, angular=shown.angular, status=shown.status, instruction=shown.instruction, source_timestamp=advice_frame.timestamp if advice_frame else None))+'\n')
                    guidance_log.flush()
                    if args.display != 'opencv':
                        print(shown.terminal(), flush=True)
                    last_print = time.monotonic()
                if viewer is not None:
                    footer = f'{args.mode} | frames {len(recorder.frames) if recorder else "-"} | space pause, f analyze, g save, q quit'
                    atomic_json(args.stream_dir/'advice.json', dict(linear=shown.linear, angular=shown.angular,
                                status=shown.status, footer=footer, session=frame.session if frame else '',
                                max_frame_age=args.max_frame_age, valid_until=min(time.time()+.25,
                                advice_frame.written_at+args.max_advice_age) if not status else time.time()+.25))
                time.sleep(.02)
        except KeyboardInterrupt:
            pass
        finally:
            print(Advice(status='Session stopped').terminal(), flush=True)
            if recorder:
                if last_recorded_frame is not None:
                    recorder.add(last_recorded_frame, force=True)
                recorder.save()
                print(f'Saved route: {output}/route.json', flush=True)
            # ExitStack waits for one in-flight inference before releasing models.
            if viewer is not None and viewer.poll() is None:
                viewer.terminate()
                viewer.wait(timeout=5)


if __name__ == '__main__':
    main()
