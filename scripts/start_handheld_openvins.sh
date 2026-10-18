#!/usr/bin/env bash
set -e
D455_VIO_WS="${OPENVINS_WORKSPACE:-$HOME/openvins_ws}"
source /opt/ros/jazzy/setup.bash
source "$D455_VIO_WS/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/mapless-ros-logs}"
exec ros2 launch ov_msckf subscribe.launch.py \
    "config_path:=$D455_VIO_WS/config/d455_stereo/estimator_config.yaml" verbosity:=DEBUG "$@"
