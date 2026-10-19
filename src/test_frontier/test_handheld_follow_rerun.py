"""Rerun follow integration, source-frame overlays, and independent odometry frames."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mapless_robotics import handheld
from mapless_robotics.handheld_follow_view import (
    FollowPrediction, FollowReference, FollowRerunView, prediction_snapshot,
)
from mapless_robotics.handheld_guidance import Advice
from mapless_robotics.live_io import LiveFrame


def frame(x=0.):
    pose = np.eye(4)
    pose[0, 3] = x
    return LiveFrame(np.full((24, 32, 3), 80, np.uint8), np.ones((24, 32), np.float32),
                     np.array([[30., 0, 16], [0, 30., 12], [0, 0, 1]]), pose,
                     1., time.time(), 4, 'follow-test')


def reference():
    return FollowReference(tuple(np.zeros((24, 32, 3), np.uint8) for _ in range(3)),
                           np.array([[100., 0, 0], [101., 0, -1], [102., 0, -2]]), 'forward')


def prediction():
    return FollowPrediction(reference(), np.full((24, 32, 3), 120, np.uint8),
                            ('follow-test', 3), np.zeros((3, 2)), np.ones(3, bool),
                            dict(status='tracking', nearest_frame=1, target_frame=2))


class FollowDisplayTests(unittest.TestCase):
    def mock_view(self):
        view = FollowRerunView.__new__(FollowRerunView)
        view.rr = MagicMock()
        view.reference = view.last_prediction = view.last_frame_key = None
        view.trail = []
        return view

    def test_reference_and_live_positions_are_not_overlaid_in_different_world_frames(self):
        view = self.mock_view()
        view.log(frame(-50.), Advice(.2, .1, 'LoTIS tracking'), prediction(), prediction_valid=True)
        points = view.rr.Points2D.call_args_list
        np.testing.assert_array_equal(points[0].args[0], [[102., -2.]])  # reference goal
        np.testing.assert_array_equal(points[2].args[0], [[101., -1.]])  # matched reference
        np.testing.assert_array_equal(view.trail, [[-50., 0.]])
        text = view.rr.TextDocument.call_args_list[-1].args[0]
        self.assertIn('not measured current position', text)

    def test_projection_rgb_comes_from_analyzed_frame_instead_of_new_live_frame(self):
        view = self.mock_view()
        view.log(frame(), Advice(.2), prediction(), prediction_valid=True)
        images = [call.args[0] for call in view.rr.Image.call_args_list]
        self.assertEqual(int(images[0][0, 0, 0]), 80)
        self.assertEqual(int(images[2][0, 0, 0]), 120)

    def test_pause_or_stale_input_clears_dynamic_matches_and_projections(self):
        view = self.mock_view()
        data = prediction()
        view.log(frame(), Advice(.2), data, prediction_valid=True)
        view.rr.log.reset_mock()
        view.log(frame(), Advice(status='PAUSED'), data, prediction_valid=False)
        paths = [call.args[0] for call in view.rr.log.call_args_list]
        self.assertIn('reference/map/matched', paths)
        self.assertIn('reference/map/lookahead', paths)
        self.assertIn('lotis/projections', paths)
        self.assertIsNone(view.last_prediction)
        view.log(frame(), Advice(.2), data, prediction_valid=True)
        self.assertIs(view.last_prediction, data)

    def test_worker_snapshot_copies_outputs_without_affecting_navigation_inputs(self):
        current = frame()
        raw = SimpleNamespace(coords=np.zeros((3, 2)), visibility=np.ones(3), distances=np.arange(3.))
        base = SimpleNamespace(_last_result=raw)
        model = SimpleNamespace(method=base, visibility_threshold=.5,
                                diagnostics={'status': 'tracking', 'nearest_frame': 1})
        snapshot = prediction_snapshot(model, reference(), current)
        raw.coords[:] = 99
        current.rgb[:] = 0
        model.diagnostics['nearest_frame'] = 2
        np.testing.assert_array_equal(snapshot.coords, np.zeros((3, 2)))
        self.assertEqual(int(snapshot.rgb[0, 0, 0]), 80)
        self.assertEqual(snapshot.diagnostics['nearest_frame'], 1)

    def test_nonfinite_and_invisible_projections_are_not_displayed(self):
        raw = SimpleNamespace(coords=np.array([[0., 0.], [np.nan, 0.], [0., 0.]]),
                              visibility=np.array([1., 1., .1]), distances=np.zeros(3))
        method = SimpleNamespace(method=SimpleNamespace(_last_result=raw), visibility_threshold=.5,
                                 diagnostics={})
        result = prediction_snapshot(method, reference(), frame())
        np.testing.assert_array_equal(result.usable, [True, False, False])

    def test_real_rerun_blueprint_and_entities_are_saved_headlessly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'follow.rrd'
            view = FollowRerunView(spawn=False, save_path=path)
            view.log(frame(), Advice(.2, .1, 'LoTIS tracking'), prediction(), prediction_valid=True)
            view.log(frame(), Advice(status='STOP: guidance expired'), prediction(), prediction_valid=False)
            view.close()
            command = Path(sys.executable).parent/'rerun'
            result = subprocess.run([str(command), 'rrd', 'print', str(path)],
                                    capture_output=True, text=True, check=True)
            for entity in ['reference/map/path', 'guidance/status', 'camera/rgb',
                           'lotis/projections', 'live/map/trail', 'reference/goal_image']:
                self.assertIn(entity, result.stdout)
            camera = subprocess.run([str(command), 'rrd', 'print', '--entity', 'camera/rgb', str(path)],
                                    capture_output=True, text=True, check=True)
            self.assertIn('EncodedImage:blob', camera.stdout)
            self.assertNotIn('Image:buffer', camera.stdout)


class FollowMainTests(unittest.TestCase):
    def run_main(self, *display_args):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = frame()
            fake = SimpleNamespace(step=MagicMock(return_value=(Advice(.2), prediction())))
            argv = ['handheld', '--mode', 'follow', '--route', str(path/'reference'),
                    '--record-dir', str(path/'output'), '--stream-dir', str(path/'stream'),
                    '--max-seconds', '.12', *display_args]
            with patch.object(sys, 'argv', argv), patch.object(sys.stdin, 'isatty', return_value=False), \
                    patch.object(handheld, 'read_frame', return_value=data), \
                    patch.object(handheld, 'LivePolicy', return_value=fake), \
                    patch('mapless_robotics.handheld_follow_view.FollowRerunView') as view, \
                    patch.object(handheld.subprocess, 'Popen') as opencv, redirect_stdout(io.StringIO()):
                handheld.main()
                return view, opencv

    def test_same_follow_command_defaults_to_rerun_without_opencv_process(self):
        view, opencv = self.run_main()
        view.assert_called_once()
        self.assertTrue(view.call_args.kwargs['spawn'])
        view.return_value.log.assert_called()
        view.return_value.close.assert_called_once()
        opencv.assert_not_called()

    def test_no_spawn_still_records_the_follow_session(self):
        view, opencv = self.run_main('--no-spawn')
        self.assertFalse(view.call_args.kwargs['spawn'])
        self.assertEqual(view.call_args.kwargs['save_path'].name, 'session.rrd')
        opencv.assert_not_called()

    def test_explicit_terminal_display_keeps_no_gui(self):
        view, opencv = self.run_main('--display', 'terminal')
        view.assert_not_called()
        opencv.assert_not_called()


if __name__ == '__main__':
    unittest.main()
