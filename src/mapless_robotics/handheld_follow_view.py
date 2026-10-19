"""Standalone LoTIS following in Rerun, with separate reference/live frames."""
from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np


@dataclass(frozen=True)
class FollowReference:
    images: tuple
    positions: np.ndarray
    direction: str


@dataclass(frozen=True)
class FollowPrediction:
    reference: FollowReference
    rgb: np.ndarray
    frame_key: tuple
    coords: np.ndarray
    usable: np.ndarray
    diagnostics: dict


def prediction_snapshot(method, reference, frame):
    """Copy model output on the inference worker; the UI never reads model state."""
    result = method.method._last_result
    coords = np.asarray(result.coords, dtype=float).copy()
    visibility = np.asarray(result.visibility, dtype=float).reshape(-1)
    distances = np.asarray(result.distances, dtype=float).reshape(-1)
    usable = (np.isfinite(visibility) & (visibility > method.visibility_threshold)
              & np.isfinite(distances) & (distances >= 0)
              & np.isfinite(coords).all(axis=1) & (np.abs(coords) <= 1).all(axis=1))
    return FollowPrediction(reference, frame.rgb.copy(), frame.key, coords, usable,
                            dict(method.diagnostics))


class FollowRerunView:
    def __init__(self, *, spawn=True, save_path=None):
        import rerun as rr

        self.rr = rr
        rr.init('mapless_handheld_follow', spawn=False)
        sinks = []
        if spawn:
            rr.spawn(connect=False)
            sinks.append(rr.GrpcSink())
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            sinks.append(rr.FileSink(str(save_path)))
        if sinks:
            rr.set_sinks(*sinks)
        self._send_blueprint()
        self.reference = None
        self.trail = []
        self.last_frame_key = None
        self.last_prediction = None
        rr.log('guidance/controls', rr.TextDocument(
            'Controls are in the launch terminal: Space pauses/resumes; f analyzes; q quits.\n'
            'Reference map: +X right, -Z up. Blue is the LoTIS matched waypoint, '
            'not a metric estimate of your position. Gold is the goal.\n'
            'The live odometry tab uses its own session origin.'), static=True)
        self.log(None, None, None, prediction_valid=False)

    def _send_blueprint(self, reference=None):
        import rerun.blueprint as rrb
        rr = self.rr
        bounds = None
        if reference is not None:
            positions = reference.positions[:, [0, 2]]
            lower, upper = positions.min(axis=0), positions.max(axis=0)
            padding = np.maximum((upper-lower)*.12, .5)
            bounds = rrb.VisualBounds2D(x_range=[lower[0]-padding[0], upper[0]+padding[0]],
                                       y_range=[lower[1]-padding[1], upper[1]+padding[1]])
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(name='Recorded LoTIS route', origin='reference/map', visual_bounds=bounds),
                    rrb.TextDocumentView(name='Walking instructions', origin='guidance/status'),
                    row_shares=[3, 2]),
                rrb.Vertical(
                    rrb.Tabs(
                        rrb.Spatial2DView(name='Live camera', origin='camera'),
                        rrb.Spatial2DView(name='LoTIS projections (analyzed frame)', origin='lotis'),
                        rrb.Spatial2DView(name='Live odometry (separate origin)', origin='live/map'),
                        active_tab=0),
                    rrb.Horizontal(
                        rrb.Spatial2DView(name='Matched reference', origin='reference/matched_image'),
                        rrb.Spatial2DView(name='Goal view', origin='reference/goal_image')),
                    row_shares=[3, 2])),
            rrb.TimePanel(timeline='live_time', play_state='Following', state='Collapsed'),
            auto_layout=False, auto_views=False, collapse_panels=True))

    def set_reference(self, reference):
        if self.reference is reference:
            return
        self.reference = reference
        self._send_blueprint(reference)
        rr = self.rr
        positions = reference.positions[:, [0, 2]]
        goal = 0 if reference.direction == 'backward' else len(positions) - 1
        rr.log('reference/map/path', rr.LineStrips2D(
            [positions], colors=[(85, 190, 225)], radii=rr.Radius.ui_points(2)))
        rr.log('reference/map/goal', rr.Points2D(
            [positions[goal]], colors=[(255, 195, 55)], radii=rr.Radius.ui_points(9),
            labels=['GOAL'], show_labels=True))
        rr.log('reference/goal_image', rr.Image(np.asarray(reference.images[goal])).compress(jpeg_quality=85))

    def log(self, frame, advice, prediction, *, prediction_valid):
        from .handheld_guidance import Advice
        rr = self.rr
        rr.set_time('live_time', timestamp=time.time())
        advice = advice or Advice(status='WAIT: synchronized camera and LoTIS guidance')
        if frame is not None and frame.key != self.last_frame_key:
            rr.log('camera/rgb', rr.Image(frame.rgb).compress(jpeg_quality=85))
            position = frame.camera_pose[[0, 2], 3]
            if not self.trail or np.linalg.norm(position - self.trail[-1]) >= .03:
                self.trail.append(position.copy())
            rr.log('live/map/trail', rr.LineStrips2D(
                [np.asarray(self.trail)], colors=[(85, 190, 225)], radii=rr.Radius.ui_points(2)))
            rr.log('live/map/camera', rr.Arrows2D(
                origins=[position], vectors=[-frame.camera_pose[[0, 2], 2] * .3],
                colors=[(90, 225, 100)]))
            self.last_frame_key = frame.key
        elif frame is None:
            rr.log('camera/rgb', rr.Image(np.zeros((240, 320, 3), np.uint8)).compress(jpeg_quality=85))
        if prediction is not None:
            self.set_reference(prediction.reference)
        if prediction is not None and prediction_valid:
            new_prediction = prediction is not self.last_prediction
            if new_prediction:
                rr.log('lotis/rgb', rr.Image(prediction.rgb).compress(jpeg_quality=85))
                indices = np.flatnonzero(prediction.usable)
                if len(indices):
                    height, width = prediction.rgb.shape[:2]
                    pixels = (prediction.coords[indices, ::-1] + 1.) * [width / 2., height / 2.]
                    rr.log('lotis/projections', rr.Points2D(
                        pixels, colors=[(85, 190, 225)], radii=rr.Radius.ui_points(4),
                        labels=[str(i) for i in indices]))
                else:
                    rr.log('lotis/projections', rr.Clear(recursive=True))
                self.last_prediction = prediction
            for entity, key, color in [('matched', 'nearest_frame', (70, 145, 255)),
                                        ('lookahead', 'target_frame', (245, 115, 65))]:
                index = prediction.diagnostics.get(key)
                valid = (index is not None and 0 <= index < len(prediction.reference.positions)
                         and prediction.usable[index])
                if valid:
                    rr.log('reference/map/' + entity, rr.Points2D(
                        [prediction.reference.positions[index, [0, 2]]], colors=[color],
                        radii=rr.Radius.ui_points(7), labels=[f'{entity} {index}'], show_labels=True))
                    if entity == 'matched' and new_prediction:
                        rr.log('reference/matched_image', rr.Image(
                            np.asarray(prediction.reference.images[index])).compress(jpeg_quality=85))
                else:
                    rr.log('reference/map/' + entity, rr.Clear(recursive=True))
            rr.log('guidance/model', rr.TextDocument(json.dumps(prediction.diagnostics, indent=2)))
        else:
            for entity in ['reference/map/matched', 'reference/map/lookahead',
                           'reference/matched_image', 'lotis/projections']:
                rr.log(entity, rr.Clear(recursive=True))
            # A resumed prediction must republish overlays cleared during a pause.
            self.last_prediction = None
        health = ('Waiting for camera input.' if frame is None else
                  f'Input age: {max(0., time.time()-frame.written_at):.2f} s | session {frame.session[:8]}')
        reference_note = ('Loading the reference route and LoTIS model.' if self.reference is None else
                          f'{len(self.reference.positions)} encoded reference frames | {self.reference.direction}')
        rr.log('guidance/status', rr.TextDocument(
            f'# {advice.instruction}\n\n{advice.status}\n\n'
            f'Suggested speed: {advice.linear:+.2f} m/s | yaw: {advice.angular:+.2f} rad/s\n\n'
            f'{health}\n\n{reference_note}\n\n'
            'Blue marker = matched reference waypoint, not measured current position.\n\n'
            'Terminal keys: Space pause/resume · f analyze · q quit', media_type='text/markdown'))

    def close(self):
        from .handheld_guidance import Advice
        self.log(None, Advice(status='Session stopped'), None, prediction_valid=False)
        recording = self.rr.get_global_data_recording()
        if recording is not None:
            recording.flush(timeout_sec=5)
