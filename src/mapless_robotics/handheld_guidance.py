"""Human-readable guidance and image/pose route recording. No robot command output."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Advice:
    linear: float = 0.
    angular: float = 0.
    status: str = 'WAIT'
    tilt: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.linear, self.angular, self.tilt]).all():
            raise ValueError('Guidance velocities must be finite')

    @property
    def instruction(self):
        turn = 'LEFT' if self.angular > .05 else 'RIGHT' if self.angular < -.05 else ''
        motion = 'FORWARD' if self.linear > .01 else 'BACKWARD' if self.linear < -.01 else ''
        if motion:
            return motion + (' + '+turn if turn else '')
        if turn:
            return 'TURN '+turn
        if abs(self.tilt) > .01:
            return 'LOOK UP' if self.tilt > 0 else 'LOOK DOWN'
        return 'STOP'

    def terminal(self):
        return f'{self.instruction:18s} v={self.linear:+.2f} m/s  yaw={self.angular:+.2f} rad/s | {self.status}'


def draw_advice(rgb, advice, footer=''):
    import cv2
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    color = (80, 220, 80) if advice.instruction != 'STOP' else (0, 190, 255)
    cv2.rectangle(image, (0, 0), (width, 88), (25, 25, 25), -1)
    cv2.putText(image, advice.instruction, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    cv2.putText(image, f'v {advice.linear:+.2f} m/s | yaw {advice.angular:+.2f} rad/s',
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    cv2.putText(image, advice.status[:90], (12, 77), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
    if advice.instruction != 'STOP':
        start = (width//2, height-70)
        dx = -int(np.clip(advice.angular, -1, 1)*width*.3)
        if abs(advice.tilt) > .01:
            end = (start[0], start[1]-int(np.sign(advice.tilt)*height*.3))
        elif abs(advice.linear) > .01:
            end = (start[0]+dx, start[1]-int(np.sign(advice.linear)*height*.3))
        else:
            end = (start[0]+(-width//4 if advice.angular > 0 else width//4), start[1])
        cv2.arrowedLine(image, start, end, color, 8, tipLength=.25)
    cv2.putText(image, footer[:100], (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return image


class RouteRecorder:
    def __init__(self, directory, distance=.2, angle_degrees=15.):
        self.directory = Path(directory)
        if (self.directory/'route.json').exists():
            raise ValueError('Route already exists; choose a new --record-dir')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.last_pose = None
        self.session = None
        self.distance = distance
        self.angle = np.deg2rad(angle_degrees)

    def add(self, frame, force=False):
        import cv2
        if self.session is not None and self.session != frame.session:
            raise ValueError('Odometry session changed: finish this route and start a new recording')
        if self.frames and frame.sequence == self.frames[-1]['sequence']:
            return False
        pose = frame.camera_pose
        if self.last_pose is not None and not force:
            rotation = self.last_pose[:3, :3].T @ pose[:3, :3]
            angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))
            if np.linalg.norm(pose[:3, 3]-self.last_pose[:3, 3]) < self.distance and angle < self.angle:
                return False
        name = f'{len(self.frames):06d}.png'
        if not cv2.imwrite(str(self.directory/name), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
            raise OSError('Could not save route image')
        self.session = frame.session
        self.frames.append(dict(file=name, sequence=frame.sequence, timestamp=frame.timestamp,
                                camera_pose=pose.tolist(), intrinsics=frame.intrinsics.tolist()))
        self.last_pose = pose.copy()
        self.save()
        return True

    def save(self):
        metadata = dict(version=1, frame_convention='pipeline_y_up_camera_minus_z_forward',
                        session=self.session, frames=self.frames)
        temporary = self.directory/'route.json.tmp'
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(self.directory/'route.json')


def load_route(directory, max_frames):
    import cv2
    from PIL import Image
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory/'route.json').read_text())
    if 'active_frame_ids' in metadata and 'odometry_positions' in metadata:
        # Shared auto_explore_navigate export: immutable archive and loop-pruned
        # active retrieval path. Positions describe reference arc length only.
        converted = []
        for frame_id in metadata['active_frame_ids']:
            pose = np.eye(4)
            pose[:3, 3] = metadata['odometry_positions'][frame_id]
            converted.append(dict(file=f"{metadata.get('archive_directory', 'archive')}/{frame_id:06d}.png",
                                  camera_pose=pose.tolist()))
        metadata = dict(version=1, frame_convention='pipeline_y_up_camera_minus_z_forward', frames=converted)
    if metadata.get('version') != 1 or metadata.get('frame_convention') != 'pipeline_y_up_camera_minus_z_forward':
        raise ValueError('Unsupported route format/convention')
    frames = metadata['frames']
    if len(frames) < 2:
        raise ValueError('Record at least two distinct route viewpoints')
    indices = np.linspace(0, len(frames)-1, min(len(frames), max_frames), dtype=int)
    images, positions = [], []
    for index in indices:
        item = frames[index]
        path = (directory/item['file']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Route image path must stay inside the route directory')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Missing route image: {path}')
        pose = np.asarray(item['camera_pose'], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Invalid recorded camera pose')
        images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        positions.append(pose[:3, 3])
    return images, positions
