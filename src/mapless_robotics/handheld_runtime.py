"""Physical input and advisory actuation for the shared automatic pipeline."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time

import numpy as np

from .exploration.exploration import Observation
from .floor_reference import FloorReference
from .handheld_guidance import Advice
from .handheld_map import TopDownMap
from .live_io import OdometryFault, OdometryGuard, read_frame
from .visualization.rerun_view import RerunView


@dataclass
class SensorState:
    position: np.ndarray
    rotation: object


@dataclass
class HandheldState(SensorState):
    sensor_states: dict = field(default_factory=dict)


def state_from_pose(pose):
    import quaternion
    position = pose[:3, 3].copy()
    rotation = quaternion.from_rotation_matrix(pose[:3, :3])
    return HandheldState(position, rotation, {'rgb': SensorState(position.copy(), rotation)})


def turned(first, second):
    import quaternion
    relative = quaternion.as_rotation_matrix(first.rotation).T @ quaternion.as_rotation_matrix(second.rotation)
    return float(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))


class HandheldAgent:
    def __init__(self, runtime):
        self.runtime = runtime

    def get_state(self):
        if self.runtime.control_frame is None:
            raise RuntimeError('No synchronized physical pose')
        return state_from_pose(self.runtime.control_frame.camera_pose)

    def set_state(self, *args, **kwargs):
        raise RuntimeError('Physical camera poses are read-only; walk or turn according to the guidance')


class HandheldRerunView(RerunView):
    def __init__(self, *, spawn=True, save_path=None, object_search=False, map_span=6., depth_source='realsense'):
        super().__init__(spawn=False, enable_edge_vis=True, enable_navigation_points_vis=True)
        import rerun.blueprint as rrb
        rr = self.rr
        # save() replaces the viewer connection. Configure both destinations in
        # one call so recording a session keeps the live viewer connected.
        sinks = []
        if spawn:
            rr.spawn(connect=False)
            sinks.append(rr.GrpcSink())
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            sinks.append(rr.FileSink(str(save_path)))
        if sinks:
            rr.set_sinks(*sinks)
        rr.log('/', rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        self.floor_reference = None
        self._navigation_summary = ''
        self._graph = None
        self.local_map = TopDownMap(map_span)
        self.depth_source = depth_source
        self._return_goal = self._object_goal = None
        self._return_path = None
        self._route_goal = None
        self._route_path = None
        self._retrieval_positions = None
        diagnostics = [
            rrb.Vertical(
                rrb.Spatial2DView(name='Frontier analysis', origin='detections'),
                rrb.TextDocumentView(name='Frontier detector status', origin='detections/status'),
                row_shares=[4, 1], name='Frontiers'),
            rrb.Spatial2DView(name='LoTIS projections', origin='camera'),
            rrb.Vertical(
                rrb.Spatial2DView(name='Return viewpoint', origin='return_target/image'),
                rrb.TextDocumentView(name='Return target', origin='return_target/status'),
                row_shares=[4, 1], name='Recovery'),
            rrb.Spatial2DView(name='LoTIS goal', origin='goal'),
            rrb.Spatial2DView(name='Closest reference', origin='closest_trajectory_frame'),
        ]
        if object_search:
            diagnostics.append(rrb.Vertical(
                rrb.Spatial2DView(name='Object detections', origin='object_search/image'),
                rrb.TextDocumentView(name='Object search', origin='object_search/status'),
                row_shares=[4, 1], name='Object search'))
        # Keep the following local map outside the diagnostic tabs. Additional
        # model panes must not shrink navigation guidance to a thin strip.
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name='Exploration and recovery graph', origin='world'),
                    rrb.Tabs(*diagnostics, active_tab=0, name='Analysis details'),
                    row_shares=[3, 2]),
                rrb.Vertical(
                    rrb.Spatial2DView(name='Local navigation map', origin='handheld/map'),
                    rrb.TextDocumentView(name='Walking instruction and input health', origin='handheld/status'),
                    rrb.TextDocumentView(name='Pipeline state', origin='pipeline/status'),
                    row_shares=[6, 3, 1]),
                column_shares=[3, 2]),
            # Rerun times are thread-local. Both the control loop and the
            # instruction heartbeat publish on this wall-clock timeline.
            rrb.TimePanel(timeline='live_time', play_state='Following', state='Collapsed'),
            auto_layout=False, auto_views=False, collapse_panels=True))

    def set_floor_reference(self, reference, camera_pose):
        self.floor_reference = reference
        x, z = camera_pose[0, 3], camera_pose[2, 3]
        y = reference.floor_y
        lines = []
        for delta in np.linspace(-1.5, 1.5, 7):
            lines.append([[x-1.5, y, z+delta], [x+1.5, y, z+delta]])
            lines.append([[x+delta, y, z-1.5], [x+delta, y, z+1.5]])
        self.rr.log('world/floor_reference', self.rr.LineStrips3D(
            lines, colors=[(110, 150, 170)], radii=.003), static=True)
        self.rr.log('world/floor_label', self.rr.Points3D(
            [[x, y, z]], labels=[f'Floor Y={y:.2f} m'], radii=.02), static=True)

    def start_step(self):
        self.rr.set_time('live_time', timestamp=time.time())
        super().start_step()

    def log_navigation_map(self, navigator):
        self._obstacle_diagnostics = getattr(navigator, 'height_diagnostics', {})
        self.local_map.update(navigator)
        goal = navigator.goal
        if goal is not None and navigator.camera_pose is not None:
            axes = navigator.plane_indices
            distance = float(np.linalg.norm(
                (np.asarray(goal)-navigator.camera_pose[:3, 3])[axes]))
            graph = getattr(self, '_graph', None)
            objective = graph.objective_frontier if graph is not None else None
            label = (f'frontier F{objective.id}' if objective is not None
                     and np.allclose(goal, objective.position) else 'navigation target')
            path_status = 'follow the cyan path' if len(navigator.trajectory) else 'no local path; stop'
            self._navigation_summary = f'To {label}: {distance:.2f} m | {path_status}'
        else:
            self._navigation_summary = 'Waiting for a selected frontier and local path'
        super().log_navigation_map(navigator)

    def log_analysis(self, result, pose_id, graph):
        self._graph = graph
        self.rr.set_time('live_time', timestamp=time.time())
        super().log_analysis(result, pose_id, graph)
        if not result.frontier_analysis:
            return
        diagnostics = result.frontier_diagnostics or {}
        raw = diagnostics.get('raw_anchors', len(result.frontiers))
        status = (f'P{pose_id}: {raw} raw anchors -> {len(result.frontiers)} depth-valid detections; '
                  f'{len(graph.frontiers)} graph frontiers')
        if diagnostics:
            status += f"\nDepth coverage: {diagnostics['valid_depth_fraction']:.1%}; frontier pixels: {diagnostics['frontier_pixels']}"
            coverage = diagnostics.get('mask_valid_fraction')
            if coverage is not None:
                status += f'; anchor-mask coverage: {coverage:.1%}'
        # Manual analysis remains visible even when walking guidance is paused.
        if self.paused:
            self.rr.log('detections/image', self.rr.Image(result.marked_rgb).compress(jpeg_quality=90))
        self.rr.log('detections/status', self.rr.TextDocument(status))
        print('FRONTIERS | '+status.replace('\n', ' | '), flush=True)

    def prepare_retrieval(self, states):
        self._retrieval_positions = np.stack([state.position for state in states]).copy()
        self._route_goal = (self._retrieval_positions[0], 'Recorded start')
        self._route_path = None

    def begin_retrieval(self):
        positions = getattr(self, '_retrieval_positions', None)
        if positions is not None:
            self._route_goal = (positions[-1], 'Retrieval goal')
            self._route_path = positions

    def log_return_target(self, frontier, pose, reference_rgb, **kwargs):
        super().log_return_target(frontier, pose, reference_rgb, **kwargs)
        self._return_goal = (np.array(pose.pose[:3, 3], copy=True), f'Return to P{pose.id}')

    def log_return_legs(self, backward, forward, junction_frame):
        super().log_return_legs(backward, forward, junction_frame)
        backward = np.asarray(backward, dtype=float).reshape(-1, 3)
        forward = np.asarray(forward, dtype=float).reshape(-1, 3)
        self._return_path = np.concatenate((backward[::-1], forward[1:]), axis=0)

    def clear_return_target(self):
        self._return_goal = None
        self._return_path = None
        super().clear_return_target()

    def log_object_goal(self, search, approach):
        self._object_goal = (np.array(approach, copy=True), f'Object: {search.target.label}')
        super().log_object_goal(search, approach)

    def clear_object_goal(self):
        self._object_goal = None
        super().clear_object_goal()

    def log_advice(self, frame, advice, *, print_terminal=False):
        # This heartbeat remains active while perception loads or inference blocks.
        rr = self.rr
        rr.set_time('live_time', timestamp=time.time())
        summary = getattr(self, '_navigation_summary', '')
        if advice.status == 'Follow the pipeline suggestion' and summary:
            advice = Advice(advice.linear, advice.angular, summary, advice.tilt)
        graph = getattr(self, '_graph', None)
        goal, label = None, 'No selected goal'
        reference_path = None
        route_goal = getattr(self, '_route_goal', None)
        return_goal = getattr(self, '_return_goal', None)
        object_goal = getattr(self, '_object_goal', None)
        objective = getattr(graph, 'objective_frontier', None)
        if route_goal is not None:
            goal, label = route_goal
            reference_path = self._route_path
        elif return_goal is not None:
            goal, label = return_goal
            reference_path = self._return_path
        elif object_goal is not None:
            goal, label = object_goal
        elif objective is not None:
            goal, label = objective.position, f'Frontier F{objective.id}'
        image, _ = self.local_map.render(
            frame, goal, label, advice.instruction,
            arrival_radius=getattr(graph, 'visited_distance', None),
            reference_path=reference_path, depth_source=self.depth_source)
        rr.log('handheld/map/image', rr.Image(image).compress(jpeg_quality=90))
        health = f'\nDepth source: {self.depth_source}'
        health += '' if frame is None else (f'\nCamera XYZ: {np.round(frame.camera_pose[:3, 3], 3).tolist()} m'
                  f'\nInput age: {max(0., time.time()-frame.written_at):.2f} s | session {frame.session[:8]}')
        reference = getattr(self, 'floor_reference', None)
        if reference is not None and frame is not None:
            position = frame.camera_pose[:3, 3]
            height = reference.height_above_floor(position)
            health += f'\nHeight above floor: {height:.2f} m | floor Y={reference.floor_y:.2f} m'
            foot = position.copy()
            foot[1] = reference.floor_y
            rr.log('world/camera_height', rr.LineStrips3D(
                [[position, foot]], colors=[(80, 220, 180)], radii=.008))
        diagnostics = getattr(self, '_obstacle_diagnostics', {})
        if diagnostics.get('floor_height') is not None:
            health += (f"\nGround excluded: {diagnostics['ground_samples']} samples | "
                       f"obstacles: {diagnostics['obstacle_samples']} | floor band: +/-{diagnostics['floor_clearance']:.2f} m")
        rr.log('handheld/status', rr.TextDocument(
            f'# {advice.instruction}\n\n{advice.status}\n\n'
            f'Suggested speed: {advice.linear:+.2f} m/s | yaw: {advice.angular:+.2f} rad/s'
            +health, media_type='text/markdown'))
        rr.log('handheld/linear_mps', rr.Scalars(advice.linear))
        rr.log('handheld/angular_rps', rr.Scalars(advice.angular))
        if print_terminal:
            print(advice.terminal(), flush=True)


class HandheldRuntime:
    """Same capture/control interface as the simulator, driven only by live poses."""
    def __init__(self, args, view, keyboard):
        self.args, self.view, self.keyboard = args, view, keyboard
        self.stream_dir = Path(args.stream_dir)
        self.agent = HandheldAgent(self)
        self.control_frame = self.latest = None
        self.camera_intrinsics = None
        self.floor_reference = None
        self.linear_speed = args.speed
        self.guard = OdometryGuard(args.max_odometry_distance, args.max_odometry_speed)
        self.lock = threading.RLock()
        self.fatal = None
        self.phase = 'waiting for initialized OpenVINS and synchronized RGB-D'
        self.warning = 'WAIT: fresh synchronized RGB-D and initialized OpenVINS'
        self.advice = Advice(status='Loading the shared exploration pipeline')
        self.advice_until = 0.
        self.manual_hint = 'stop walking; Space resumes the pipeline'
        self.stop_event = threading.Event()
        self.heartbeat = None
        self.log = (args.run_dir/'guidance.jsonl').open('a')

    def _sample(self):
        with self.lock:
            if self.fatal is not None:
                return None
            try:
                frame = read_frame(self.stream_dir)
                if not 0 <= time.time()-frame.written_at <= self.args.max_frame_age:
                    self.warning = 'STOP: live RGB-D or OpenVINS input expired'
                    return None
                self.guard.check(frame)
                if self.camera_intrinsics is not None and not np.allclose(frame.intrinsics, self.camera_intrinsics):
                    raise OdometryFault('Camera calibration changed; start a new handheld session')
                self.camera_intrinsics = frame.intrinsics.copy()
                self.latest = frame
                initial_height = getattr(self.args, 'camera_height', None)
                if self.floor_reference is None and initial_height is not None:
                    reference = FloorReference.from_camera_pose(frame.camera_pose, initial_height)
                    self.floor_reference = reference
                    self.view.set_floor_reference(reference, frame.camera_pose)
                    print(f'FLOOR | startup camera height {initial_height:.2f} m; fixed floor Y={reference.floor_y:.3f} m; ground band +/-{getattr(self.args, 'floor_clearance', .05):.2f} m', flush=True)
                self.warning = ''
                return frame
            except OdometryFault as error:
                self.fatal = error
                return None
            except (FileNotFoundError, ValueError, OSError):
                self.warning = 'WAIT: fresh synchronized RGB-D and initialized OpenVINS'
                try:
                    status = json.loads((self.stream_dir/'status.json').read_text())
                    if status.get('state') == 'fault':
                        self.fatal = OdometryFault(status['message'])
                except (FileNotFoundError, ValueError, OSError):
                    pass
                return None

    def _heartbeat(self):
        last_print = 0.
        while not self.stop_event.is_set():
            frame = self._sample()
            with self.lock:
                if self.fatal is not None:
                    advice = Advice(status=str(self.fatal))
                elif frame is None:
                    advice = Advice(status=self.warning)
                elif self.view.paused:
                    advice = Advice(status='PAUSED: '+self.manual_hint)
                elif time.time() > self.advice_until:
                    advice = Advice(status='STOP: '+self.phase)
                else:
                    advice = self.advice
            print_now = time.monotonic()-last_print >= .5
            self.view.log_advice(frame or self.latest, advice, print_terminal=print_now)
            if print_now:
                self.log.write(json.dumps(dict(time=time.time(), linear=advice.linear, angular=advice.angular,
                                               status=advice.status, instruction=advice.instruction,
                                               source_timestamp=frame.timestamp if frame else None))+'\n')
                self.log.flush()
                last_print = time.monotonic()
            self.stop_event.wait(.1)

    def start(self):
        self.heartbeat = threading.Thread(target=self._heartbeat, daemon=True, name='handheld-guidance')
        self.heartbeat.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.heartbeat is not None:
            self.heartbeat.join(timeout=5)
        self.view.log_advice(self.latest, Advice(status='Session stopped'), print_terminal=True)
        self.log.close()

    def capture_observation(self):
        while True:
            frame = self._sample()
            if self.fatal is not None:
                raise self.fatal
            if frame is not None:
                self.control_frame = frame
                return Observation(frame.rgb, frame.camera_pose, frame.camera_pose, frame.timestamp,
                                   frame.depth if self.args.depth_source == 'realsense' else None, intrinsics=frame.intrinsics)
            if self.keyboard.poll(.05) in (ord('q'), 27):
                raise KeyboardInterrupt

    @property
    def navigation_height_options(self):
        if self.floor_reference is None:
            return {}
        return dict(floor_height=self.floor_reference.floor_y,
                    minimum_obstacle_height=getattr(self.args, 'obstacle_min_height', None),
                    maximum_obstacle_height=getattr(self.args, 'obstacle_max_height', None),
                    floor_clearance=getattr(self.args, 'floor_clearance', .05))

    def command(self, linear, angular, duration=.1, *, status='Follow the pipeline suggestion', tilt=0.):
        source = self.control_frame
        frame = self._sample()
        if self.fatal is not None:
            raise self.fatal
        if (frame is not None and source is not None
                and time.time()-source.written_at <= self.args.max_advice_age):
            with self.lock:
                self.phase = 'waiting for a fresh pipeline instruction'
                self.advice = Advice(float(linear), float(angular), status, float(tilt))
                self.advice_until = min(time.time()+self.args.max_advice_age,
                                        source.written_at+self.args.max_advice_age)
        else:
            with self.lock:
                self.advice_until = 0.
        time.sleep(max(0., duration))

    def apply_velocity(self, linear, angular, duration):
        self.command(linear, angular, duration)

    def get_agent(self, index):
        if index != 0:
            raise ValueError('There is one handheld camera')
        return self.agent

    def restore_view(self, reference, *, force=False):
        if self.args.skip_view_alignment and not force:
            self.command(0, 0, status='Recorded-view alignment skipped; keeping current heading')
            return
        import quaternion
        target = quaternion.as_rotation_matrix(reference.sensor_states['rgb'].rotation)
        target_forward = -target[:, 2]
        target_yaw = np.arctan2(target[0, 2], target[2, 2])
        target_pitch = np.arctan2(target_forward[1], np.linalg.norm(target_forward[[0, 2]]))
        while True:
            observation = self.capture_observation()
            current = observation.camera_pose
            yaw = np.arctan2(current[0, 2], current[2, 2])
            forward = -current[:3, 2]
            pitch = np.arctan2(forward[1], np.linalg.norm(forward[[0, 2]]))
            yaw_error = float((target_yaw-yaw+np.pi)%(2*np.pi)-np.pi)
            pitch_error = float(target_pitch-pitch)
            key = self.keyboard.poll(0)
            if key in (ord('q'), 27):
                raise KeyboardInterrupt
            if key == ord(' '):
                self.view.set_paused(not self.view.paused)
            if self.view.paused:
                time.sleep(.05)
                continue
            if abs(yaw_error) <= np.deg2rad(5) and abs(pitch_error) <= np.deg2rad(5):
                self.command(0, 0, status='Recorded view aligned')
                return
            self.command(0, float(np.clip(2*yaw_error, -1, 1)),
                         status=f'Align with the recorded view: yaw {np.degrees(yaw_error):+.0f} deg, tilt {np.degrees(pitch_error):+.0f} deg',
                         tilt=pitch_error if abs(pitch_error) > np.deg2rad(5) else 0.)

    def step(self, action):
        # Keyboard nudges in the shared loop pause automatic guidance. Tell the
        # operator what to do, without integrating a synthetic pose.
        self.manual_hint = action.replace('_', ' ')+' physically; Space resumes, r retries recovery'
        print(self.manual_hint, flush=True)

    def turn_around(self):
        import quaternion
        reference = self.agent.get_state()
        rotation = quaternion.as_rotation_matrix(reference.rotation)
        target = np.eye(4)
        target[:3, :3] = np.diag([-1., 1., -1.])@rotation
        old_paused = self.view.paused
        self.view.set_paused(False)
        try:
            self.restore_view(state_from_pose(target), force=True)
        finally:
            self.view.set_paused(old_paused)

    def wait_for_retrieval_start(self, reference):
        print('Walk back to the recorded start physically. Enter starts retrieval once within the start tolerance.', flush=True)
        while True:
            observation = self.capture_observation()
            distance = float(np.linalg.norm((observation.robot_pose[:3, 3]-reference.position)[[0, 2]]))
            self.command(0, 0, status=f'Return to the start yourself: {distance:.2f} m away. Enter starts LoTIS retrieval.')
            key = self.keyboard.poll(0)
            if key in (ord('q'), 27):
                raise KeyboardInterrupt
            if key in (10, 13) and distance <= self.args.retrieval_start_tolerance:
                self.view.set_paused(False)
                self.restore_view(reference)
                self.view.begin_retrieval()
                return


def step_velocity(runtime, linear, angular):
    runtime.apply_velocity(linear, angular, runtime.args.integration_time)
