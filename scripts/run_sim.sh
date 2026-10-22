#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/gz_ws/src/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/aerothon_ws/src/aerothon_sim/models:$HOME/aerothon_ws/src/aerothon_sim/worlds:$HOME/gz_ws/src/ardupilot_gazebo/models:$HOME/gz_ws/src/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}

echo "=========================================================="
echo " Starting Gazebo Sim: Aerothon 50x50m Green Patch World"
echo "=========================================================="
gz sim -v4 -r aerothon_50x50.sdf
