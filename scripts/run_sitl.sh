#!/bin/bash
set -e

cd /home/nh/ardupilot
echo "=========================================================="
echo " Starting ArduPilot SITL (with outputs on 14550 and 14551)"
echo "=========================================================="
./Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --map --out=127.0.0.1:14550 --out=127.0.0.1:14551
