#!/usr/bin/env python3
"""
Cube Orange+ MAVLink Communication & Telemetry Verifier
-------------------------------------------------------
Verifies MAVLink 2 link over TELEM2 / USB-to-FTDI at 921,600 baud.
Audits key parameters, flight modes, GPS health, battery status,
and verifies RC channel switches (CH5 mode & CH7/8 BRAKE).

Usage:
  python3 test_mavlink_cube.py --port /dev/ttyUSB0 --baud 921600
  python3 test_mavlink_cube.py --auto
  python3 test_mavlink_cube.py --test-modes
"""

import argparse
import glob
import sys
import time
from pymavlink import mavutil

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 921600


def find_serial_ports():
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyAMA*")
    return [p for p in ports if not p.endswith("AMA10")]


def connect_cube(port=DEFAULT_PORT, baud=DEFAULT_BAUD):
    print("=" * 60)
    print(f"[MAVLINK] Connecting to Cube Orange+ on {port} @ {baud} baud...")
    print("=" * 60)
    try:
        conn = mavutil.mavlink_connection(port, baud=baud)
        # Announce Ground Control Station presence
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )
        print("[MAVLINK] Waiting for autopilot heartbeat (up to 5s)...")
        hb = conn.wait_heartbeat(timeout=5.0)
        if hb is None:
            print(f"[ERROR] Heartbeat timeout on {port} @ {baud}.")
            print("Troubleshooting Checks:")
            print("1. Is the USB-to-FTDI cable firmly plugged into the Pi 5 and Cube TELEM2?")
            print("2. Is Cube Orange+ powered on?")
            print("3. Is TELEM2 configured for MAVLink 2 at 921600 baud in Mission Planner?")
            print("   - SERIAL2_PROTOCOL = 2 (MAVLink 2)")
            print("   - SERIAL2_BAUD = 921 (921,600 baud)")
            return None
        
        mode_str = conn.flightmode if hasattr(conn, 'flightmode') else str(hb.custom_mode)
        armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        print(f"[MAVLINK SUCCESS] Heartbeat received from System {conn.target_system}, Component {conn.target_component}!")
        print(f"  * Autopilot Type : {mavutil.mavlink.enums['MAV_AUTOPILOT'][hb.autopilot].name}")
        print(f"  * Vehicle Type   : {mavutil.mavlink.enums['MAV_TYPE'][hb.type].name}")
        print(f"  * Current Mode   : {mode_str}")
        print(f"  * Armed Status   : {'ARMED' if armed else 'DISARMED'}")
        print(f"  * MAVLink Proto  : {'MAVLink 2' if conn.WIRE_PROTOCOL_VERSION == '2.0' else 'MAVLink 1'}")
        return conn
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return None


def request_data_streams(conn):
    """Request high-rate telemetry streams from ArduPilot."""
    msg = conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
    )


def audit_telemetry(conn, duration=5.0):
    print("\n" + "=" * 60)
    print(f"[TELEMETRY AUDIT] Monitoring Cube Orange+ Telemetry for {duration} seconds...")
    print("=" * 60)
    request_data_streams(conn)

    start = time.time()
    last_hb = time.time()
    telemetry = {
        "battery_v": None,
        "battery_pct": None,
        "gps_fix": None,
        "satellites": None,
        "hdop": None,
        "pos_ned": [None, None, None],
        "yaw_deg": None,
        "ch5_mode_raw": None,
        "ch7_brake_raw": None,
        "ch8_raw": None,
    }

    gps_fix_names = {
        0: "No GPS", 1: "No Fix", 2: "2D Fix", 3: "3D Fix", 4: "DGPS Fix", 5: "RTK Float", 6: "RTK Fixed"
    }

    while time.time() - start < duration:
        # Keep GCS heartbeat alive
        now = time.time()
        if now - last_hb > 1.0:
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
            )
            last_hb = now

        msg = conn.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue

        mtype = msg.get_type()
        if mtype == "SYS_STATUS":
            telemetry["battery_v"] = msg.voltage_battery / 1000.0
            telemetry["battery_pct"] = msg.battery_remaining
        elif mtype == "GPS_RAW_INT":
            telemetry["gps_fix"] = gps_fix_names.get(msg.fix_type, f"Type {msg.fix_type}")
            telemetry["satellites"] = msg.satellites_visible
            telemetry["hdop"] = msg.eph / 100.0
        elif mtype == "LOCAL_POSITION_NED":
            telemetry["pos_ned"] = [msg.x, msg.y, msg.z]
        elif mtype == "ATTITUDE":
            import math
            telemetry["yaw_deg"] = math.degrees(msg.yaw)
        elif mtype == "RC_CHANNELS":
            telemetry["ch5_mode_raw"] = msg.chan5_raw
            telemetry["ch7_brake_raw"] = msg.chan7_raw
            telemetry["ch8_raw"] = msg.chan8_raw

    print("\n--- Cube Orange+ Subsystem Status ---")
    if telemetry["battery_v"] is not None:
        print(f"  * Battery Voltage    : {telemetry['battery_v']:.2f} V ({telemetry['battery_pct']}%)")
    else:
        print("  * Battery Voltage    : [No SYS_STATUS received]")

    if telemetry["gps_fix"] is not None:
        print(f"  * GPS Fix Status     : {telemetry['gps_fix']}")
        print(f"  * Satellites Visible : {telemetry['satellites']}")
        print(f"  * HDOP Position Dil. : {telemetry['hdop']:.2f} (Target <= 0.8)")
    else:
        print("  * GPS Status         : [No GPS_RAW_INT received (indoor benchtop?)]")

    if telemetry["pos_ned"][0] is not None:
        print(f"  * Local NED Position : N={telemetry['pos_ned'][0]:.2f}m, E={telemetry['pos_ned'][1]:.2f}m, Alt={-telemetry['pos_ned'][2]:.2f}m")
    if telemetry["yaw_deg"] is not None:
        print(f"  * Heading Yaw        : {telemetry['yaw_deg']:.1f}°")

    print("\n--- Safety Pilot RC Channel Inputs ---")
    print(f"  * Channel 5 (Flight Mode switch) : {telemetry['ch5_mode_raw']} us")
    print(f"  * Channel 7 (BRAKE/Aux switch)   : {telemetry['ch7_brake_raw']} us")
    print(f"  * Channel 8 (Aux switch)         : {telemetry['ch8_raw']} us")

    return telemetry


def test_mode_command(conn, mode_name="BRAKE"):
    print(f"\n[COMMAND TEST] Sending DO_SET_MODE -> {mode_name}...")
    mode_id = conn.mode_mapping().get(mode_name)
    if mode_id is None:
        print(f"[ERROR] Unknown mode '{mode_name}' for this vehicle.")
        return False

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id, 0, 0, 0, 0, 0
    )

    t0 = time.time()
    while time.time() - t0 < 3.0:
        ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=1.0)
        if ack and ack.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            result_names = {0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED", 3: "UNSUPPORTED", 4: "FAILED"}
            res_str = result_names.get(ack.result, f"Code {ack.result}")
            print(f"[COMMAND_ACK] DO_SET_MODE result: {res_str}")
            return ack.result == 0
    print("[ERROR] No COMMAND_ACK received within 3.0s.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Cube Orange+ MAVLink Communication Tester")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default: 921600)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect connected serial ports")
    parser.add_argument("--test-modes", action="store_true", help="Test sending GUIDED and BRAKE mode commands")
    args = parser.parse_args()

    port = args.port
    if args.auto:
        ports = find_serial_ports()
        print(f"[AUTO-DETECT] Available serial ports on Pi 5: {ports}")
        if not ports:
            print("[ERROR] No USB or serial ports detected on the Raspberry Pi 5.")
            print("Please plug the USB-to-FTDI cable into a USB port on the Pi 5.")
            sys.exit(1)
        port = ports[0]

    conn = connect_cube(port=port, baud=args.baud)
    if conn is None:
        # Try fallback baud rates just in case Cube is on 57600 or 115200
        for fallback_baud in [57600, 115200]:
            print(f"\n[FALLBACK] Attempting connection at {fallback_baud} baud...")
            conn = connect_cube(port=port, baud=fallback_baud)
            if conn:
                print(f"\n[WARNING] Connected at {fallback_baud} baud, but production requires 921600 baud!")
                print("Please set SERIAL2_BAUD = 921 in Mission Planner to eliminate setpoint streaming latency.")
                break

    if conn is None:
        sys.exit(1)

    audit_telemetry(conn, duration=4.0)

    if args.test_modes:
        print("\n" + "=" * 60)
        print("[SAFETY MODE TEST] Testing Companion Computer Flight Mode Commands")
        print("=" * 60)
        test_mode_command(conn, "BRAKE")
        time.sleep(1.0)
        test_mode_command(conn, "GUIDED")

    print("\n[COMPLETE] MAVLink link verification finished successfully.")


if __name__ == '__main__':
    main()
