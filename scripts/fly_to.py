#!/usr/bin/env python3
"""
Simple Flight Command Utility for ArduPilot in Local Meters
-----------------------------------------------------------
Translates local (X, Y) meter commands into MAVLink LOCAL_NED setpoints.

Usage:
  python3 fly_to.py --takeoff 4         # Arms and takes off to 4m
  python3 fly_to.py --x 12 --y 10       # Flies to X=12m, Y=10m (meters from takeoff)
  python3 fly_to.py --qr 1              # Flies directly over QR 1 (12, 10)
  python3 fly_to.py --qr 2              # Flies directly over QR 2 (-14, 15)
  python3 fly_to.py --qr 3              # Flies directly over QR 3 (16, -12)
  python3 fly_to.py --qr 4              # Flies directly over QR 4 (-10, -15)
  python3 fly_to.py --home              # Returns to (0, 0)
  python3 fly_to.py --land              # Lands
"""

import argparse
import sys
import time
from pymavlink import mavutil

QR_POSITIONS = {
    1: (10.0, 12.0),   # qr_tile_1 (North=10, East=12)
    2: (8.0, -12.0),   # qr_tile_2 [MATCH] (North=8, East=-12)
    3: (-8.0, 14.0),   # qr_tile_3 (North=-8, East=14)
    4: (-10.0, -10.0)  # qr_tile_4 (North=-10, East=-10)
}
TAKEOFF_PAD = (-25.0, 0.0)

def connect_drone(url=None, baud=921600):
    if url and url.startswith("/dev/") and ":" in url:
        parts = url.split(":")
        url = parts[0]
        try:
            baud = int(parts[1])
        except ValueError:
            pass

    if url:
        ports_to_try = [(url, baud)]
    else:
        import glob
        detected = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        ports_to_try = [(p, baud) for p in detected] + [
            ("udpin:0.0.0.0:14550", 115200),
            ("udpin:0.0.0.0:14551", 115200),
            ("tcp:127.0.0.1:5762", 115200),
            ("tcp:127.0.0.1:5763", 115200)
        ]
    for p, b in ports_to_try:
        if not p: continue
        try:
            print(f"[DRONE] Trying connection on {p} @ {b}...")
            m = mavutil.mavlink_connection(p, baud=b)
            try:
                m.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                )
            except Exception:
                pass
            hb = m.wait_heartbeat(timeout=3.5)
            if hb:
                print(f"[DRONE] Connected successfully to ArduPilot on {p} (System {m.target_system})")
                return m
        except Exception:
            pass
    raise RuntimeError("Could not connect to ArduPilot. Check physical connection (/dev/ttyACM0 or /dev/ttyUSB0).")

def send_local_ned(m, north, east, down):
    type_mask = 0b110111111000
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, 1,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        north, east, down,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

def set_mode(m, mode_name):
    mode_id = m.mode_mapping().get(mode_name)
    if mode_id is None:
        print(f"Unknown mode: {mode_name}")
        return
    m.set_mode(mode_id)
    print(f"Mode set to {mode_name}")

def arm_and_takeoff(m, target_alt=3.0):
    set_mode(m, 'GUIDED')
    time.sleep(0.5)
    print("Arming motors...")
    for _ in range(5):
        m.mav.command_long_send(
            m.target_system, 1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0
        )
        time.sleep(0.3)
    print(f"Taking off to {target_alt}m...")
    m.mav.command_long_send(
        m.target_system, 1,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_alt
    )

def main():
    parser = argparse.ArgumentParser(description="Fly ArduPilot in local meter coordinates")
    parser.add_argument("--takeoff", type=float, help="Arm and takeoff to specified altitude in meters")
    parser.add_argument("--x", "--target-n", dest="x", type=float, default=None, help="Target North (X) coordinate in meters")
    parser.add_argument("--y", "--target-e", dest="y", type=float, default=None, help="Target East (Y) coordinate in meters")
    parser.add_argument("--alt", type=float, default=3.5, help="Target altitude in meters")
    parser.add_argument("--target-d", type=float, default=None, help="Target Down coordinate in meters (negative is up)")
    parser.add_argument("--qr", type=int, choices=[1, 2, 3, 4], help="Fly directly to QR code index (1-4)")
    parser.add_argument("--home", action="store_true", help="Fly back to (0, 0)")
    parser.add_argument("--land", action="store_true", help="Land the drone")
    parser.add_argument("--url", default=None, help="MAVLink connection URL")
    
    args = parser.parse_args()
    if args.target_d is not None:
        args.alt = abs(args.target_d)
    
    m = connect_drone(args.url)
    
    if args.land:
        set_mode(m, 'LAND')
        print("Landing command sent.")
        return
        
    if args.takeoff is not None:
        arm_and_takeoff(m, args.takeoff)
        return
        
    target_x = None
    target_y = None
    
    if args.qr is not None:
        target_x, target_y = QR_POSITIONS[args.qr]
        print(f"Selected QR {args.qr} at ({target_x}m, {target_y}m)")
    elif args.home:
        target_x, target_y = TAKEOFF_PAD[0], TAKEOFF_PAD[1]
        print(f"Selected Takeoff Pad ({target_x:.1f}m, {target_y:.1f}m)")
    elif args.x is not None or args.y is not None:
        target_x = args.x if args.x is not None else 0.0
        target_y = args.y if args.y is not None else 0.0
        
    if target_x is not None and target_y is not None:
        set_mode(m, 'GUIDED')
        print(f"Streaming waypoint target: North(X)={target_x:.1f}m, East(Y)={target_y:.1f}m, Alt={args.alt:.1f}m")
        for _ in range(25):
            send_local_ned(m, north=target_x, east=target_y, down=-args.alt)
            time.sleep(0.1)
        print("Target setpoint sent! Drone is flying to position.")

if __name__ == '__main__':
    main()
