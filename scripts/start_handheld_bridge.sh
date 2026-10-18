#!/usr/bin/env bash
set -e
MAPLESS_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/mapless-ros-logs}"
export PYTHONPATH="$MAPLESS_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -m mapless_robotics.ros_rgbd_bridge "$@"
