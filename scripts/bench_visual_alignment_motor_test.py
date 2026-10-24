#!/usr/bin/env python3
"""
Aerothon 2026 — Indoor Benchtop Visual Servoing & Motor Response Test (Props OFF! ⚠️)
------------------------------------------------------------------------------------
Allows verifying closed-loop motor reactions to QR movement without GPS.

How it Works:
  1. Checks Ethernet link to SIYI camera (192.168.144.25).
  2. Arms ArduPilot in GUIDED_NOGPS mode at safe bench thrust (Props REMOVED).
  3. Ingests live SIYI camera feed via zero-latency SiyiStreamWorker (or --mock-qr).
  4. Tracks QR centroid in real time.
  5. Translates pixel offset (ex, ey) into commanded Roll (phi) and Pitch (theta) tilt angles.
  6. Continuously streams SET_ATTITUDE_TARGET at 20 Hz to drive motor differentials:
     - Move QR to RIGHT -> Drone commands Roll Right -> Left motors spin faster.
     - Move QR to LEFT  -> Drone commands Roll Left  -> Right motors spin faster.
     - Move QR FORWARD  -> Drone commands Pitch Down -> Rear motors spin faster.
     - Move QR BACKWARD -> Drone commands Pitch Up   -> Front motors spin faster.
     - Center QR        -> Roll=0, Pitch=0 -> All 4 motors spin at uniform hover speed.
  7. Keeps motors active against ArduPilot's ground land detector auto-disarm.
  8. Runs for --duration (default 30s), then automatically ramps down thrust and disarms!

Usage:
  python3 scripts/bench_visual_alignment_motor_test.py
  python3 scripts/bench_visual_alignment_motor_test.py --thrust 0.25 --duration 30
  python3 scripts/bench_visual_alignment_motor_test.py --mock-qr
  python3 scripts/bench_visual_alignment_motor_test.py --no-display
"""

import argparse
import math
import os
import subprocess
import sys
import threading
import time

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.*=false"

import cv2
import numpy as np
from pymavlink import mavutil

CAMERA_IP = "192.168.144.25"
DEFAULT_RTSP = f"rtsp://{CAMERA_IP}:8554/main.264"


# ----------------------------------------------------------------------
# 1. Threaded Zero-Latency Camera Worker
# ----------------------------------------------------------------------

class SiyiStreamWorker:
    def __init__(self, capture_source=DEFAULT_RTSP):
        self.capture_source = capture_source
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self):
        print(f"[CAMERA] Opening video stream: {self.capture_source}")
        if isinstance(self.capture_source, int):
            self.cap = cv2.VideoCapture(self.capture_source)
        else:
            # Let OpenCV use default TCP/UDP fallback without forcing broken UDP
            self.cap = cv2.VideoCapture(self.capture_source, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.connected = True
            return True
        return False

    def _update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    time.sleep(1.0)
                    continue
            ret, img = self.cap.read()
            if ret and img is not None:
                with self.lock:
                    self.frame = img
                    self.connected = True
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ----------------------------------------------------------------------
# 2. Math Utility: Euler Angles to Quaternion
# ----------------------------------------------------------------------

def euler_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float = 0.0):
    """Converts Euler roll, pitch, yaw (radians) into a MAVLink unit quaternion [w, x, y, z]."""
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


# ----------------------------------------------------------------------
# 3. Main Bench Test Runner
# ----------------------------------------------------------------------

def run_motor_alignment_test(args):
    print("\n" + "=" * 70)
    print(" AEROTHON 2026 — INDOOR BENCHTOP VISUAL SERVOING & MOTOR TEST")
    print(" CRITICAL SAFETY RULE: ENSURE ALL 4 PROPELLERS ARE REMOVED! ⚠️")
    print("=" * 70 + "\n")

    # 1. Check Network Connectivity to SIYI Camera
    if not args.mock_qr:
        print(f"[NETWORK CHECK] Pinging SIYI Camera at {CAMERA_IP}...")
        res = subprocess.run(["ping", "-c", "1", "-W", "1", CAMERA_IP], capture_output=True, text=True)
        if res.returncode != 0:
            print("\n" + "!" * 68)
            print("[CAMERA WARNING] Cannot reach SIYI Camera at 192.168.144.25!")
            print("  * Is the RJ45 Ethernet cable plugged into the Pi 5's eth0 port?")
            print("  * Is the SIYI camera powered with 12V-24V? (Check gimbal LEDs)")
            print("  * Tip: You can run with '--mock-qr' to test motor reactions now.")
            print("!" * 68 + "\n")
            if not args.force:
                print("Exiting. Power on the camera or run with --mock-qr to simulate.")
                return

    # 2. Connect to ArduPilot on Serial
    port = args.port
    if not port or not os.path.exists(port):
        import glob
        detected = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        port = detected[0] if detected else "/dev/ttyACM0"

    print(f"[MAVLINK] Connecting to ArduPilot on {port} @ {args.baud}...")
    m = mavutil.mavlink_connection(port, baud=args.baud)
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    hb = m.wait_heartbeat(timeout=5.0)
    if hb is None:
        print("[ERROR] Heartbeat timeout. Could not connect to ArduPilot.")
        sys.exit(1)

    print(f"[MAVLINK SUCCESS] Connected to System {m.target_system}! Initial Mode: {m.flightmode}")

    # Request high-rate motor servo outputs
    m.mav.request_data_stream_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 20, 1
    )

    # 3. Initialize Camera
    worker = None
    if not args.mock_qr:
        cam_src = args.url or f"rtsp://{CAMERA_IP}:8554/{args.stream}.264"
        worker = SiyiStreamWorker(capture_source=cam_src)
        print("[CAMERA] Waiting for initial frame...")
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if worker.get_frame() is not None:
                break
            time.sleep(0.2)
        if worker.get_frame() is None:
            print("[CAMERA WARN] No video frame received yet. Proceeding with stream worker running.")

    detector = cv2.QRCodeDetector()
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        has_pyzbar = True
    except ImportError:
        has_pyzbar = False

    # Safety Confirmation
    print("\n" + "#" * 65)
    print("TEST PROCEDURE SUMMARY:")
    print(f"  1. Drone will switch to GUIDED_NOGPS mode.")
    print(f"  2. Motors will ARM at safe bench thrust ({args.thrust*100:.0f}% throttle).")
    print(f"  3. For {args.duration:.0f} seconds, present and move the QR under the camera.")
    print(f"  4. Watch your physical motors tilt and compensate in real time!")
    print(f"  5. Press Ctrl+C at any moment to immediately stop and DISARM.")
    print("#" * 65)

    print("\nStarting in 3 seconds... (Props MUST be OFF!)")
    for sec in [3, 2, 1]:
        print(f"  Countdown: {sec}...", flush=True)
        time.sleep(1.0)

    # Switch to GUIDED_NOGPS (Mode 20)
    mode_id = 20  # GUIDED_NOGPS
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id, 0, 0, 0, 0, 0
    )
    time.sleep(0.2)

    # Prime attitude stream before arming so GUIDED_NOGPS doesn't time out
    q_neutral = euler_to_quaternion(0.0, 0.0, 0.0)
    for _ in range(5):
        m.mav.set_attitude_target_send(
            0, m.target_system, m.target_component,
            0b00000111, q_neutral,
            0, 0, 0, args.thrust
        )
        time.sleep(0.04)

    # Command ARM
    print("\n[DRONE] Arming motors in GUIDED_NOGPS...")
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0
    )
    time.sleep(0.3)

    motor_pwms = [1000, 1000, 1000, 1000]
    is_armed = False

    show_gui = not args.no_display and ("DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
    if show_gui:
        cv2.namedWindow("Visual Servoing Motor Response HUD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Visual Servoing Motor Response HUD", 960, 540)

    start_time = time.time()
    last_loop = time.time()
    last_log = time.time()
    last_arm_cmd = time.time()

    commanded_roll_deg = 0.0
    commanded_pitch_deg = 0.0
    status_tag = "SEARCHING FOR QR"

    # Mock QR simulation variables
    mock_t0 = time.time()

    try:
        while time.time() - start_time < args.duration:
            loop_dt = time.time() - last_loop
            last_loop = time.time()
            elapsed = time.time() - start_time
            remaining = max(0.0, args.duration - elapsed)

            # Poll MAVLink messages (motor PWMs and heartbeat)
            while True:
                msg = m.recv_match(blocking=False)
                if msg is None:
                    break
                mtype = msg.get_type()
                if mtype == "SERVO_OUTPUT_RAW":
                    motor_pwms = [msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw]
                elif mtype == "HEARTBEAT":
                    is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

            # Keep-Alive: If ArduPilot ground land detector attempts to auto-disarm, re-arm immediately!
            if not is_armed and (time.time() - last_arm_cmd > 2.0):
                m.mav.command_long_send(
                    m.target_system, m.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 21196, 0, 0, 0, 0, 0
                )
                last_arm_cmd = time.time()

            # Grab video frame or generate mock frame
            frame = worker.get_frame() if worker else None
            qr_detected = False
            pts = None
            err_x_px = 0.0
            err_y_px = 0.0
            roll_rad = 0.0
            pitch_rad = 0.0

            # Mock QR generation if requested
            if args.mock_qr:
                # Synthesize a 960x540 frame with a QR moving in a circle
                h, w = 540, 960
                frame = np.full((h, w, 3), 40, dtype=np.uint8)
                t_m = time.time() - mock_t0
                cx, cy = w // 2, h // 2
                # Circular motion: radius 160 px, period 8 seconds
                u_m = int(cx + 180 * math.cos(t_m * 0.8))
                v_m = int(cy + 140 * math.sin(t_m * 0.8))
                # Draw mock QR square
                sz = 80
                cv2.rectangle(frame, (u_m - sz//2, v_m - sz//2), (u_m + sz//2, v_m + sz//2), (255, 255, 255), -1)
                cv2.rectangle(frame, (u_m - sz//2, v_m - sz//2), (u_m + sz//2, v_m + sz//2), (0, 0, 0), 4)
                cv2.putText(frame, "QR TARGET", (u_m - 35, v_m + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                pts = np.array([
                    [u_m - sz//2, v_m - sz//2], [u_m + sz//2, v_m - sz//2],
                    [u_m + sz//2, v_m + sz//2], [u_m - sz//2, v_m + sz//2]
                ], dtype=np.float32).reshape(1, 4, 2)
                ok = True

            elif frame is not None:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, pts = detector.detect(gray)

                if (not ok or pts is None) and has_pyzbar:
                    decoded = pyzbar_decode(gray)
                    if decoded and len(decoded[0].polygon) == 4:
                        pts = np.array([[p.x, p.y] for p in decoded[0].polygon], dtype=np.float32).reshape(1, 4, 2)
                        ok = True
            else:
                ok = False

            if ok and pts is not None and len(pts) > 0:
                qr_detected = True
                p = pts.reshape(-1, 2)
                u_c, v_c = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))

                # Pixel displacement from optical center
                err_x_px = u_c - cx  # (+) = QR is to the Right -> Roll Right
                err_y_px = v_c - cy  # (-) = QR is Forward (Top) -> Pitch Down/Forward

                # Proportional attitude control: Max 18 degrees tilt
                max_tilt_rad = math.radians(18.0)
                norm_x = np.clip(err_x_px / (w * 0.45), -1.0, 1.0)
                norm_y = np.clip(err_y_px / (h * 0.45), -1.0, 1.0)

                # Camera Nadir orientation:
                # Top of image (-norm_y) is Forward -> negative pitch tilts forward
                # Right of image (+norm_x) is Right -> positive roll tilts right
                roll_rad = norm_x * max_tilt_rad
                pitch_rad = norm_y * max_tilt_rad

                commanded_roll_deg = math.degrees(roll_rad)
                commanded_pitch_deg = -math.degrees(pitch_rad)

                dist_px = math.hypot(err_x_px, err_y_px)
                if dist_px < 35:
                    status_tag = f"LOCKED DEAD-CENTER ({dist_px:.1f}px)"
                else:
                    status_tag = f"ALIGNING (Offset: {dist_px:.0f}px)"
            else:
                roll_rad = 0.0
                pitch_rad = 0.0
                commanded_roll_deg = 0.0
                commanded_pitch_deg = 0.0
                status_tag = "SEARCHING FOR QR (Level Hover)"

            # Send attitude command to ArduPilot at 20 Hz
            q_target = euler_to_quaternion(roll_rad, pitch_rad, 0.0)
            m.mav.set_attitude_target_send(
                0, m.target_system, m.target_component,
                0b00000111, q_target,
                0, 0, 0, args.thrust
            )

            # Terminal telemetry bar
            if time.time() - last_log >= 0.2:
                print(f"[TIME: {remaining:4.1f}s] {status_tag} | "
                      f"Cmd Tilt: R={commanded_roll_deg:+5.1f}° P={commanded_pitch_deg:+5.1f}° | "
                      f"ESC PWM: M1={motor_pwms[0]} M2={motor_pwms[1]} M3={motor_pwms[2]} M4={motor_pwms[3]}", end='\r')
                last_log = time.time()

            # OpenCV Live HUD
            if show_gui and frame is not None:
                vis = frame.copy()
                vh, vw = vis.shape[:2]
                vcx, vcy = vw // 2, vh // 2

                # Reticle
                cv2.drawMarker(vis, (vcx, vcy), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
                cv2.circle(vis, (vcx, vcy), 18, (0, 255, 255), 1)

                if qr_detected and pts is not None:
                    p = np.int32(pts).reshape(-1, 2)
                    cv2.polylines(vis, [p], True, (0, 255, 0), 3)
                    qcx, qcy = int(np.mean(p[:, 0])), int(np.mean(p[:, 1]))
                    cv2.circle(vis, (qcx, qcy), 6, (0, 0, 255), -1)
                    cv2.line(vis, (vcx, vcy), (qcx, qcy), (255, 0, 255), 2)
                    cv2.putText(vis, f"Err: ({err_x_px:+.0f}, {err_y_px:+.0f})px", (qcx - 70, qcy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Top banner
                cv2.rectangle(vis, (0, 0), (vw, 75), (20, 20, 20), -1)
                cv2.putText(vis, f"BENCH MOTOR SERVOING: {status_tag}", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(vis, f"Commanded Tilt: Roll={commanded_roll_deg:+5.1f} deg  |  Pitch={commanded_pitch_deg:+5.1f} deg",
                            (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(vis, f"Motor ESC PWMs:  M1={motor_pwms[0]} us   M2={motor_pwms[1]} us   M3={motor_pwms[2]} us   M4={motor_pwms[3]} us",
                            (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

                # Countdown bottom box
                cd_txt = f"BENCH ALIGNMENT TIME REMAINING: {remaining:4.1f}s"
                cv2.rectangle(vis, (vcx - 190, vh - 45), (vcx + 190, vh - 10), (0, 100, 0), -1)
                cv2.putText(vis, cd_txt, (vcx - 180, vh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                cv2.imshow("Visual Servoing Motor Response HUD", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[STOP] Operator pressed 'q'. Exiting...")
                    break

            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Operator interrupted test.")
    finally:
        print("\n\n" + "=" * 60)
        print("[SHUTDOWN] Safely ramping down motors and DISARMING...")
        # Ramp thrust to 0
        for _ in range(5):
            m.mav.set_attitude_target_send(
                0, m.target_system, m.target_component,
                0b00000111, q_neutral,
                0, 0, 0, 0.0
            )
            time.sleep(0.04)

        # Disarm
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        time.sleep(0.2)

        # Restore mode to STABILIZE
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0, 0, 0, 0, 0, 0
        )
        if worker:
            worker.stop()
        if show_gui:
            cv2.destroyAllWindows()
        print("[COMPLETE] Multirotor disarmed and restored to STABILIZE mode.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Indoor Visual Servoing Motor Response Test (Props OFF)")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--url", default=None, help="Custom RTSP URL")
    parser.add_argument("--stream", choices=["main", "sub"], default="main", help="SIYI stream")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds (default: 30)")
    parser.add_argument("--thrust", type=float, default=0.25, help="Bench idle thrust [0.15-0.35] (default: 0.25)")
    parser.add_argument("--mock-qr", action="store_true", help="Synthesize moving QR on screen (test motors without camera)")
    parser.add_argument("--force", action="store_true", help="Continue even if ping fails")
    parser.add_argument("--no-display", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    run_motor_alignment_test(args)


if __name__ == '__main__':
    main()
