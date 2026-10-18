"""OpenCV viewer in the system Python; ML OpenCV may be headless."""
import argparse
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .handheld_guidance import Advice, draw_advice
from .live_io import atomic_json, read_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-dir', type=Path, default=Path('/tmp/mapless-live'))
    args = parser.parse_args()
    if 'GUI:                           NONE' in cv2.getBuildInformation():
        parser.error('Viewer Python needs OpenCV GUI support; use /usr/bin/python3 or --display terminal')
    try:
        while True:
            rgb = np.zeros((480, 640, 3), np.uint8)
            shown, footer = Advice(status='WAIT: live frame'), 'space pause | f analyze | g save | q quit'
            try:
                frame = read_frame(args.stream_dir)
                rgb = frame.rgb
                data = json.loads((args.stream_dir/'advice.json').read_text())
                if (frame.session == data['session'] and 0 <= time.time()-frame.written_at <= data['max_frame_age']
                        and time.time() <= data['valid_until']):
                    shown = Advice(data['linear'], data['angular'], data['status'])
                    footer = data['footer']
                else:
                    shown = Advice(status='STOP: camera or guidance expired')
            except (FileNotFoundError, ValueError, OSError, KeyError):
                pass
            cv2.imshow('Mapless handheld guidance', draw_advice(rgb, shown, footer))
            key = cv2.waitKeyEx(30)
            if key != -1:
                atomic_json(args.stream_dir/'viewer_key.json', dict(id=uuid.uuid4().hex, key=key, time=time.time()))
            if key in (ord('q'), 27) or cv2.getWindowProperty('Mapless handheld guidance', cv2.WND_PROP_VISIBLE) < 1:
                atomic_json(args.stream_dir/'viewer_key.json', dict(id=uuid.uuid4().hex, key=ord('q'), time=time.time()))
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
