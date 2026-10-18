#!/usr/bin/env bash
# D455 profile for RGB-D guidance + the existing stereo/IMU OpenVINS configuration.
set -e
D455_VIO_WS="${OPENVINS_WORKSPACE:-$HOME/openvins_ws}"
D455_DIAG_DIR="$D455_VIO_WS/diagnostics/d455_20261015"
if [[ ! -f "$D455_DIAG_DIR/driver_overlay/install/setup.bash" ]]; then
    echo "Patched RealSense overlay missing: $D455_DIAG_DIR/driver_overlay/install/setup.bash" >&2
    exit 1
fi
source /opt/ros/jazzy/setup.bash
source "$D455_DIAG_DIR/driver_overlay/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/mapless-ros-logs}"
# A parameter dump contains profile-dependent ROI/control values; start the new
# RGB-D profile from explicit launch settings instead of replaying those values.
exec ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera camera_name:=camera \
    enable_infra1:=true enable_infra2:=true enable_color:=true enable_depth:=true \
    enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
    gyro_fps:=200 accel_fps:=63 enable_sync:=true align_depth.enable:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 \
    depth_module.infra_profile:=640x480x30 "$@"
