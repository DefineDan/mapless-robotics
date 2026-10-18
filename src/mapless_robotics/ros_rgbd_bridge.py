"""ROS-side RGB-D/OpenVINS adapter. Run with /usr/bin/python3 after sourcing ROS."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
from pathlib import Path
import time
import uuid

import numpy as np
from .live_io import LiveFrame, PoseBuffer, camera_pose_in_pipeline, pose_matrix, write_frame, atomic_json


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-dir', type=Path, default=Path('/tmp/mapless-live'))
    parser.add_argument('--rgb-topic', default='/camera/camera/color/image_raw')
    parser.add_argument('--depth-topic', default='/camera/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--info-topic', default='/camera/camera/color/camera_info')
    parser.add_argument('--odom-topic', default='/ov_msckf/odomimu')
    parser.add_argument('--imu-frame', default='camera_imu_optical_frame')
    parser.add_argument('--camera-time-offset', type=float, default=0., help='t_imu = t_rgb + offset (seconds)')
    parser.add_argument('--sync-slop', type=float, default=.03)
    parser.add_argument('--max-pose-gap', type=float, default=.05)
    parser.add_argument('--max-source-age', type=float, default=.75)
    parser.add_argument('--rate', type=float, default=5.)
    parser.add_argument('--max-odometry-distance', type=float, default=100., help='Fail closed beyond this world-origin radius, metres')
    parser.add_argument('--max-odometry-speed', type=float, default=3., help='Fail closed above this estimated speed, m/s')
    parser.add_argument('--image-qos', choices=['reliable', 'best_effort'], default='reliable')
    return parser


def main():
    parser = make_parser()
    args, ros_args = parser.parse_known_args()
    if (not np.isfinite([args.rate, args.sync_slop, args.max_pose_gap, args.max_source_age,
                         args.camera_time_offset, args.max_odometry_distance, args.max_odometry_speed]).all()
            or min(args.rate, args.sync_slop, args.max_pose_gap, args.max_source_age, args.max_odometry_distance, args.max_odometry_speed) <= 0):
        parser.error('Rates and time limits must be finite and positive')
    # Keep ROS's Python/native dependencies out of the ML virtual environment.
    import cv2
    import message_filters
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener, TransformException

    def stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec*1e-9

    class Bridge(Node):
        def __init__(self):
            super().__init__('mapless_rgbd_bridge')
            self.poses = PoseBuffer(args.max_pose_gap)
            self.pending = deque(maxlen=5)
            self.info = None
            self.session = uuid.uuid4().hex
            self.sequence = 0
            self.last_write = 0.
            self.last_notice = 0.
            self.world_frame = None
            self.odometry_fault = None
            self.cv = CvBridge()
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)
            self.sub_odom = self.create_subscription(Odometry, args.odom_topic, self.odometry, qos_profile_sensor_data)
            self.sub_info = self.create_subscription(CameraInfo, args.info_topic, self.camera_info, 10)
            image_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE if args.image_qos == 'reliable' else ReliabilityPolicy.BEST_EFFORT)
            self.rgb_sub = message_filters.Subscriber(self, Image, args.rgb_topic, qos_profile=image_qos)
            self.depth_sub = message_filters.Subscriber(self, Image, args.depth_topic, qos_profile=image_qos)
            self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, args.sync_slop)
            self.sync.registerCallback(self.images)
            self.timer = self.create_timer(.02, self.flush)
            self.get_logger().info('Waiting for RGB, aligned depth, RGB CameraInfo, static TF, and initialized OpenVINS odometry')

        def notice(self, text):
            if time.monotonic()-self.last_notice > 2:
                self.get_logger().info(text)
                self.last_notice = time.monotonic()

        def reset_session(self):
            self.poses.samples.clear()
            self.pending.clear()
            self.session = uuid.uuid4().hex
            (args.stream_dir/'latest.npz').unlink(missing_ok=True)

        def odometry(self, message):
            if self.odometry_fault is not None:
                return  # A fault is latched until the operator restarts the bridge.
            p, v = message.pose.pose.position, message.twist.twist.linear
            distance, speed = np.linalg.norm([p.x, p.y, p.z]), np.linalg.norm([v.x, v.y, v.z])
            if (not np.isfinite([distance, speed]).all() or distance > args.max_odometry_distance
                    or speed > args.max_odometry_speed):
                self.odometry_fault = (f'OpenVINS odometry unhealthy: position norm {distance:,.1f} m, '
                                       f'speed {speed:,.1f} m/s. Restart OpenVINS and the bridge before recording.')
                self.reset_session()
                atomic_json(args.stream_dir/'status.json', dict(state='fault', message=self.odometry_fault))
                self.get_logger().error(self.odometry_fault)
                return
            t = stamp(message)
            if (self.world_frame is not None and self.world_frame != message.header.frame_id
                    or self.poses.samples and (t < self.poses.samples[-1][0] or t-self.poses.samples[-1][0] > 1.)):
                self.reset_session()
                self.notice('Odometry restarted or clock/frame changed; new stream session')
            self.world_frame = message.header.frame_id
            p, q = message.pose.pose.position, message.pose.pose.orientation
            try:
                self.poses.add(t, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
            except ValueError as error:
                self.notice(str(error))

        def camera_info(self, message):
            self.info = message

        def images(self, rgb, depth):
            self.pending.append((rgb, depth))

        def flush(self):
            if self.odometry_fault is not None:
                return
            if not self.pending or self.info is None or time.monotonic()-self.last_write < 1/args.rate:
                return
            rgb, depth = self.pending[-1]
            source_age = self.get_clock().now().nanoseconds/1e9 - stamp(rgb)
            if source_age < -.1 or source_age > args.max_source_age:
                self.notice('Waiting: image timestamp is stale or outside the ROS clock')
                return
            world_imu = self.poses.at(stamp(rgb)+args.camera_time_offset)
            if world_imu is None:
                self.notice('Waiting: odometry must bracket RGB timestamp without a large gap')
                return
            if ((rgb.height, rgb.width) != (depth.height, depth.width)
                    or (rgb.height, rgb.width) != (self.info.height, self.info.width)
                    or rgb.header.frame_id != depth.header.frame_id
                    or rgb.header.frame_id != self.info.header.frame_id):
                self.notice('Waiting: RGB, aligned depth, and CameraInfo must share optical frame and dimensions')
                self.pending.clear()
                return
            try:
                transform = self.tf.lookup_transform(args.imu_frame, rgb.header.frame_id, Time()).transform
                p, q = transform.translation, transform.rotation
                imu_optical = pose_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
                image = self.cv.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
                if depth.encoding not in ('16UC1', 'mono16', '32FC1'):
                    raise ValueError('Unsupported metric depth encoding: '+depth.encoding)
                metric = self.cv.imgmsg_to_cv2(depth).astype(np.float32)
                if depth.encoding != '32FC1':
                    metric *= .001
                metric[~np.isfinite(metric) | (metric <= 0)] = 0
                k = np.asarray(self.info.k, dtype=float).reshape(3, 3)
                distortion = np.asarray(self.info.d, dtype=float)
                if np.any(distortion):
                    if self.info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                        raise ValueError('Unsupported RGB distortion model: '+self.info.distortion_model)
                    # Apply identical rectification maps to RGB and RGB-aligned depth.
                    maps = cv2.initUndistortRectifyMap(k, distortion, np.eye(3), k,
                                                     (rgb.width, rgb.height), cv2.CV_32FC1)
                    image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
                    metric = cv2.remap(metric, *maps, cv2.INTER_NEAREST)
                frame = LiveFrame(image, metric, k, camera_pose_in_pipeline(world_imu, imu_optical),
                                  stamp(rgb), time.time(), self.sequence, self.session)
                write_frame(args.stream_dir, frame)
            except (TransformException, ValueError, RuntimeError) as error:
                self.notice('Waiting: '+str(error))
                return
            self.pending.clear()
            self.sequence += 1
            self.last_write = time.monotonic()
            self.notice(f'Streaming {rgb.width}x{rgb.height} RGB-D with interpolated OpenVINS pose ({self.sequence} frames)')

    args.stream_dir.mkdir(parents=True, exist_ok=True)
    with (args.stream_dir/'.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another ROS bridge is writing this stream directory')
        (args.stream_dir/'latest.npz').unlink(missing_ok=True)
        (args.stream_dir/'status.json').unlink(missing_ok=True)
        rclpy.init(args=ros_args)
        node = Bridge()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            (args.stream_dir/'latest.npz').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
