#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /home/nh/aerothon_ws/install/setup.bash 2>/dev/null || true

# Start camera bridge in background if not already running
if ! pgrep -f "parameter_bridge" > /dev/null; then
    echo "[SETUP] Starting ros_gz_bridge for camera..."
    ros2 run ros_gz_bridge parameter_bridge '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' &
    BRIDGE_PID=$!
    trap "kill $BRIDGE_PID 2>/dev/null || true" EXIT
    sleep 2
fi

echo "=========================================================="
echo " Starting Aerothon Lawnmower Survey & QR Aligner"
echo "=========================================================="
python3 /home/nh/aerothon_ws/scripts/lawnmower_qr_mission.py "$@"
