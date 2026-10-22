#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /home/nh/aerothon_ws/install/setup.bash 2>/dev/null || true

echo "=========================================================="
echo " Starting Camera Bridge (Gazebo -> ROS 2 /camera/image_raw)"
echo "=========================================================="
ros2 run ros_gz_bridge parameter_bridge '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' &
BRIDGE_PID=$!

trap "kill $BRIDGE_PID 2>/dev/null || true" EXIT

echo "=========================================================="
echo " Launching Camera Pipeline & Viewer"
echo "=========================================================="
python3 /home/nh/aerothon_ws/src/aerothon_sim/scripts/camera_pipeline_node.py
