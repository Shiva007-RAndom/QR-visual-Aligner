#!/usr/bin/env python3
"""
AeroTHON 2026 — Master Banner Corridor & QR Lawnmower Mission Controller
========================================================================
Production autonomous mission script for Holybro X500 (Cube Orange+ ArduPilot 4.5,
Raspberry Pi 5 companion computer, SIYI A8 mini RTSP gimbal camera).

Full 12-Stage Mission Flow:
---------------------------
  STAGE 1 : Takeoff to altitude (default 5.0m AGL) and stabilize hover at (0, 0).
  STAGE 2 : Surge forward by --surge1 (default 2.0m) and scan/memorize reference QR code.
  STAGE 3 : Surge forward again by --surge2 (default 2.0m).
  STAGE 4 : Sway left (West / Body -Y) up to --max-sway (default 20.0m) searching for Green Banner.
  STAGE 5 : Position-Based Visual Servoing (PBVS) alignment with Green Banner; record lateral sway x.
  STAGE 6 : Descend to --corridor-alt (default 3.0m).
  STAGE 7 : Surge forward through Green Corridor for --corridor-surge (default 10.0m).
  STAGE 8 : Once exiting corridor, sway left by (19 - x) meters (positioning 1m from 20m wall).
            Climb to survey altitude (--alt 5.0m).
  STAGE 9 : Execute lawnmower serpentine reconnaissance to locate matching candidate QR code.
  STAGE 10: Precision PBVS centering over target QR and steady lock hover for 15 seconds.
  STAGE 11: Navigate to adjacent return corridor entrance, yaw 180° (South), detect Blue Banner,
            align visually, descend to 3.0m, and surge through return corridor by --corridor-surge.
  STAGE 12: Exit return corridor, navigate back to takeoff pad (0, 0), and execute precision LAND.

Supports:
  - Real outdoor GPS flight with MAVLink GUIDED/BRAKE modes.
  - Benchtop hardware-in-the-loop simulation (--bench-sim).
  - Safe bench motor spinning with propeller removal (--spin-motors --thrust 0.20).
  - Real-time SIYI A8 mini zero-latency RTSP capture.
  - Dual Green / Blue banner detection with RapidOCR + HSV contour fallback.
  - 4-rotation invariant QR module similarity matching + pyzbar fallback.
"""

import argparse
import math
import os
import re
import sys
import threading
import time
from typing import List, Optional, Tuple, Dict, Any

# Silence FFmpeg and Qt logging spam
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;QFontDatabase=false"

import cv2
import numpy as np
from pymavlink import mavutil


# ==============================================================================
# 1. HELPER MATH & GEOMETRY FUNCTIONS
# ==============================================================================

def wrap_pi(rad: float) -> float:
    """Wraps an angle to [-pi, pi)."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order quadrilateral points: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def euler_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float = 0.0) -> List[float]:
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


def metres_per_pixel(alt_m: float, img_w: int, hfov_rad: float = 1.22) -> float:
    """Calculates ground meters per camera pixel at given altitude and horizontal FOV."""
    h = max(0.5, alt_m)
    return 2.0 * h * math.tan(hfov_rad / 2.0) / float(img_w)


def pixel_to_ned_offset(
    u: float, v: float, img_w: int, img_h: int,
    alt_m: float, yaw_rad: float, hfov_rad: float = 1.22
) -> Tuple[float, float]:
    """
    Pinhole reprojection: transforms pixel position (u, v) into relative
    North and East offsets in meters.
      Image Top   = Body Forward (+X)
      Image Right = Body Right   (+Y)
    """
    mpp = metres_per_pixel(alt_m, img_w, hfov_rad)
    dx_body = -(v - img_h / 2.0) * mpp   # +Forward
    dy_body = (u - img_w / 2.0) * mpp    # +Right
    dn = dx_body * math.cos(yaw_rad) - dy_body * math.sin(yaw_rad)
    de = dx_body * math.sin(yaw_rad) + dy_body * math.cos(yaw_rad)
    return dn, de


# ==============================================================================
# 2. ZERO-LATENCY RTSP CAMERA WORKER
# ==============================================================================

class SiyiStreamWorker:
    """
    Threaded zero-latency RTSP camera grabber for SIYI A8 mini.
    Uses cv2.CAP_FFMPEG to drop stale buffer frames and keep only the latest.
    """
    def __init__(self, capture_source):
        self.capture_source = capture_source
        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self) -> bool:
        print(f"[CAMERA] Connecting to video source: {self.capture_source}")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        if isinstance(self.capture_source, int):
            self.cap = cv2.VideoCapture(self.capture_source)
        else:
            self.cap = cv2.VideoCapture(self.capture_source, cv2.CAP_FFMPEG)
        return self.cap.isOpened()

    def _update(self):
        fps_timer = time.time()
        fps_counter = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    time.sleep(1.0)
                    continue

            ret, img = self.cap.read()
            now = time.time()
            if ret and img is not None:
                with self.lock:
                    self.frame = img
                    self.frame_id += 1
                self.last_frame_time = now
                fps_counter += 1
                if now - fps_timer >= 1.0:
                    self.fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    @property
    def is_connected(self) -> bool:
        return (time.time() - self.last_frame_time) < 2.0

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ==============================================================================
# 3. DUAL BANNER DETECTOR (GREEN & BLUE, RAPIDOCR + COLOR CONTOUR FALLBACK)
# ==============================================================================

# Physical Banner Dimensions (AeroTHON competition specifications)
BANNER_LENGTH_M = 3.5052  # 11.5 ft
BANNER_WIDTH_M = 0.3048   # 1.0 ft

# Target OCR Brand text strings
TARGET_BRAND = "aerothon"
TARGET_YEAR = "2026"
_CONFUSION = str.maketrans({
    "o": "0", "q": "0", "d": "0",
    "i": "1", "l": "1", "|": "1",
    "z": "2", "s": "6", "5": "6",
    "g": "6", "b": "8",
})


def canon(text: str) -> str:
    text = re.sub(r"[^a-z0-9|]", "", text.lower())
    return text.translate(_CONFUSION)


BRAND_C = canon(TARGET_BRAND)
YEAR_C = canon(TARGET_YEAR)


def partial_match_distance(text: str, target: str) -> int:
    """Calculates Levenshtein distance between target and any substring of text."""
    if len(target) == 0:
        return 0
    if len(text) == 0:
        return len(target)
    prev = [0] * (len(text) + 1)
    for i, c_t in enumerate(target):
        curr = [i + 1]
        for j, c_s in enumerate(text):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c_t != c_s)
            curr.append(min(ins, dele, sub))
        prev = curr
    return min(prev)


class BannerDetector:
    """
    Robust Dual Banner Detector for Holybro X500.
    Supports both Green ("AeroTHON 2026") and Blue ("AeroTHON 2026") banners.
    Combines RapidOCR text reading with HSV geometric blob analysis and
    partial edge reconstruction.
    """
    def __init__(self, ocr_threads: int = 2, verbose: bool = False):
        self.verbose = verbose
        self.engine = None
        self._init_ocr(ocr_threads)

        # HSV Thresholds
        self.lower_green = np.array([30, 20, 40])
        self.upper_green = np.array([85, 255, 255])
        self.lower_blue = np.array([90, 50, 40])
        self.upper_blue = np.array([135, 255, 255])

    def _init_ocr(self, threads: int):
        try:
            from rapidocr import RapidOCR
            try:
                params = {
                    "Global.use_cls": False,
                    "Det.limit_type": "max",
                    "Det.limit_side_len": 640,
                    "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                }
                self.engine = RapidOCR(params=params)
            except Exception:
                self.engine = RapidOCR()
            print("[VISION] RapidOCR engine initialized successfully.")
        except Exception as e:
            print(f"[VISION WARN] RapidOCR unavailable ({e}). Fallback to color contour matching.")
            self.engine = None

    def build_mask(self, img: np.ndarray, color: str = "green") -> np.ndarray:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        if color.lower() == "blue":
            mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        else:
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def extract_ocr_items(self, ocr_output: Any) -> List[Tuple[np.ndarray, str, float]]:
        items = []
        if ocr_output is None:
            return items
        if hasattr(ocr_output, 'boxes') and hasattr(ocr_output, 'txts') and hasattr(ocr_output, 'scores'):
            if ocr_output.boxes is not None and ocr_output.txts is not None:
                for b, t, s in zip(ocr_output.boxes, ocr_output.txts, ocr_output.scores):
                    items.append((np.asarray(b, dtype=np.float32), str(t), float(s)))
            return items
        if isinstance(ocr_output, (list, tuple)):
            target = ocr_output[0] if (len(ocr_output) == 2 and isinstance(ocr_output[0], list)) else ocr_output
            for item in target:
                if len(item) == 3:
                    items.append((np.asarray(item[0], dtype=np.float32), str(item[1]), float(item[2])))
        return items

    def detect(
        self, frame: np.ndarray, color: str = "green", alt_m: float = 5.0, hfov_rad: float = 1.22
    ) -> Optional[Dict[str, Any]]:
        """
        Detects green or blue banner in frame.
        Returns dictionary with:
          - u, v: centroid coordinates in pixels (reconstructed if partially visible)
          - quad: 4x2 ordered corners (TL, TR, BR, BL)
          - angle_rad: banner long axis orientation in radians
          - partial: boolean indicating if banner is truncated by image border
          - source: 'OCR' or 'CONTOUR'
          - text: decoded text string
          - size: (W, H)
        """
        if frame is None:
            return None
        H, W = frame.shape[:2]
        color_mask = self.build_mask(frame, color=color)

        # 1. Morphological closure on color mask
        k_size = max(5, int(0.015 * max(H, W)))
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # 2. Find external color blobs
        cnts, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = 0.0004 * H * W
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            bw, bh = max(rw, rh), min(rw, rh)
            if bh < 4:
                continue
            aspect = bw / bh
            # Banner is long strip (aspect ratio ~ 11.5)
            candidates.append((area, c, rect, aspect))

        candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. Check OCR on candidate crops if engine is available
        best_ocr_match = None
        if self.engine is not None and candidates:
            for area, c, (center, (rw, rh), ang), aspect in candidates[:3]:
                pts = cv2.boxPoints(((center[0], center[1]), (rw, rh), ang))
                bx, by, bw, bh = cv2.boundingRect(np.int32(pts))
                pad = max(4, int(0.1 * max(bw, bh)))
                x0, y0 = max(0, bx - pad), max(0, by - pad)
                x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
                if x1 - x0 < 10 or y1 - y0 < 10:
                    continue

                crop = frame[y0:y1, x0:x1]
                ocr_out = None
                try:
                    ocr_out = self.engine(crop)
                except Exception:
                    pass

                items = self.extract_ocr_items(ocr_out)
                for poly, txt, score in items:
                    c_txt = canon(txt)
                    b_dist = partial_match_distance(c_txt, BRAND_C)
                    y_ok = YEAR_C in c_txt
                    if self.verbose:
                        print(f"  [{color.upper()} OCR] '{txt}' brand_dist={b_dist} year={y_ok} score={score:.2f}")
                    if b_dist <= 2 or y_ok:
                        best_ocr_match = {
                            "contour": c,
                            "rect": ((center[0], center[1]), (rw, rh), ang),
                            "text": txt,
                            "conf": score
                        }
                        break
                if best_ocr_match is not None:
                    break

        chosen_c = None
        source = "CONTOUR"
        text_label = ""
        if best_ocr_match is not None:
            chosen_c = best_ocr_match["contour"]
            source = "OCR"
            text_label = best_ocr_match["text"]
        elif candidates:
            # Fallback to largest color contour with reasonable banner aspect
            for area, c, rect, aspect in candidates:
                if aspect >= 1.5:
                    chosen_c = c
                    source = "COLOR_BLOB"
                    text_label = f"{color.upper()} BANNER"
                    break

        if chosen_c is None:
            return None

        # 4. Compute 4 corner points and long-axis angle
        min_rect = cv2.minAreaRect(chosen_c)
        quad = cv2.boxPoints(min_rect)
        quad = order_corners(quad)

        # Long-edge angle wrapped to [-pi/2, pi/2)
        e1, e2 = quad[1] - quad[0], quad[2] - quad[1]
        long_edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
        banner_angle = math.atan2(float(long_edge[1]), float(long_edge[0]))
        if banner_angle >= math.pi / 2.0:
            banner_angle -= math.pi
        elif banner_angle < -math.pi / 2.0:
            banner_angle += math.pi

        # 5. Check if banner is truncated by image border (partial view)
        bx0, by0 = float(quad[:, 0].min()), float(quad[:, 1].min())
        bx1, by1 = float(quad[:, 0].max()), float(quad[:, 1].max())
        m = 4
        cut_l, cut_r = (bx0 <= m), (bx1 >= W - 1 - m)
        cut_t, cut_b = (by0 <= m), (by1 >= H - 1 - m)
        is_partial = cut_l or cut_r or cut_t or cut_b

        u, v = float(quad[:, 0].mean()), float(quad[:, 1].mean())

        # 6. Reconstruct true center if cut off at border using physical length
        mpp = metres_per_pixel(alt_m, W, hfov_rad)
        len_px = BANNER_LENGTH_M / mpp
        wid_px = BANNER_WIDTH_M / mpp
        if cut_l and not cut_r:
            u = bx1 - len_px / 2.0
        elif cut_r and not cut_l:
            u = bx0 + len_px / 2.0
        if cut_t and not cut_b:
            v = by1 - wid_px / 2.0
        elif cut_b and not cut_t:
            v = by0 + wid_px / 2.0

        return {
            "u": u,
            "v": v,
            "quad": quad,
            "angle": banner_angle,
            "partial": is_partial,
            "source": source,
            "text": text_label,
            "size": (W, H)
        }


# ==============================================================================
# 4. QR CODE EXTRACTION & 4-ROTATION INVARIANT SIMILARITY ENGINE
# ==============================================================================

def extract_qr_features(gray: np.ndarray, quad_pts: np.ndarray, grid_size: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """Warp QR quad to 210x210, binarize, and sample 21x21 binary module grid."""
    quad = order_corners(quad_pts)
    dst = np.array([
        [0, 0],
        [grid_size * 10 - 1, 0],
        [grid_size * 10 - 1, grid_size * 10 - 1],
        [0, grid_size * 10 - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(gray, matrix, (grid_size * 10, grid_size * 10))

    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * 10 + 2, r * 10 + 8
            x0, x1 = c * 10 + 2, c * 10 + 8
            grid[r, c] = 1 if np.mean(binary[y0:y1, x0:x1]) < 128 else 0

    return grid, warped


def compute_similarity(grid_ref: np.ndarray, grid_cand: np.ndarray) -> float:
    """Calculate 4-rotation invariant agreement score in [0.0, 1.0]."""
    if grid_ref is None or grid_cand is None:
        return 0.0
    best_score = 0.0
    cand = grid_cand.copy()
    for _ in range(4):
        agreement = float(np.sum(grid_ref == cand)) / float(grid_ref.size)
        if agreement > best_score:
            best_score = agreement
        cand = np.rot90(cand)
    return best_score


# ==============================================================================
# 5. ARDUPILOT FLIGHT CONTROLLER INTERFACE (MAVLINK 2)
# ==============================================================================

class DummyMav:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class DummyMAVLinkConnection:
    def __init__(self):
        self.mav = DummyMav()
        self.flightmode = "GUIDED"
        self.target_system = 1
        self.target_component = 1

    def recv_match(self, *args, **kwargs):
        return None

    def mode_mapping(self):
        return {"GUIDED": 4, "BRAKE": 17, "LAND": 9, "GUIDED_NOGPS": 20, "STABILIZE": 0}


class DroneController:
    """
    ArduPilot MAVLink controller for physical Cube Orange+ and Holybro X500.
    Handles mode switching, Local NED position setpoints, continuous 25 Hz
    attitude streaming on bench, and motor telemetry.
    """
    def __init__(
        self, url: str = "/dev/ttyACM0", baud: int = 921600,
        bench_sim: bool = False, sim_speed: float = 1.2,
        spin_motors: bool = False, thrust: float = 0.20,
        use_nogps: bool = False
    ):
        self.bench_sim = bench_sim
        self.sim_speed = sim_speed
        self.sim_ned = [0.0, 0.0, 0.0]
        self._last_sim_step = time.time()
        self.spin_motors = spin_motors
        self.thrust = thrust
        self.use_nogps = use_nogps or bench_sim or spin_motors
        self.servos = [1000, 1000, 1000, 1000]
        self.cmd_roll_deg = 0.0
        self.cmd_pitch_deg = 0.0
        self._att_running = False

        # Auto-parse /dev/tty*:baud if passed with a colon
        if url and url.startswith("/dev/") and ":" in url:
            parts = url.split(":")
            url = parts[0]
            try:
                baud = int(parts[1])
            except ValueError:
                pass

        # Auto-detect serial port if not present
        if not url or (url.startswith("/dev/") and not os.path.exists(url)):
            import glob
            detected = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
            if detected:
                print(f"[MAVLINK] Specified '{url}' not found. Auto-detected: {detected[0]}")
                url = detected[0]
            else:
                url = "/dev/ttyACM0"

        print(f"[MAVLINK] Connecting to ArduPilot on {url} @ {baud}...")
        try:
            self.m = mavutil.mavlink_connection(url, baud=baud)
            self.m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
            )

            t_hb = time.time()
            self.sys = 1
            self.comp = 1
            while time.time() - t_hb < 4.0:
                msg = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
                if msg and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                    self.sys = msg.get_srcSystem()
                    self.comp = msg.get_srcComponent()
                    self.m.target_system = self.sys
                    self.m.target_component = self.comp
                    print(f"[MAVLINK] Autopilot confirmed! System ID: {self.sys}, Comp ID: {self.comp}, Mode: {self.m.flightmode}")
                    break
        except Exception as e:
            if self.bench_sim:
                print(f"[MAVLINK] Hardware serial unavailable ({e}). Using benchtop simulated MAVLink connection.")
                self.m = DummyMAVLinkConnection()
                self.sys = 1
                self.comp = 1
            else:
                raise e

        self.ned = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.armed = False
        self.flightmode = "UNKNOWN"
        self._last_hb = time.time()

        if self.spin_motors:
            print(f"[MAVLINK] HARDWARE ROTOR ACTIVATION: Real motors enabled at safe bench thrust ({self.thrust*100:.0f}% PWM).")
            self.m.mav.request_data_stream_send(
                self.sys, self.comp,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 20, 1
            )
            # Temporarily set DISARM_DELAY to 0.0 on bench to prevent auto-disarm
            self.m.mav.param_set_send(self.sys, self.comp, b"DISARM_DELAY", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            self._att_running = True
            self._att_thread = threading.Thread(target=self._attitude_streaming_loop, daemon=True)
            self._att_thread.start()

    def _attitude_streaming_loop(self):
        """Dedicated high-priority 25 Hz background attitude loop for bench motor spin."""
        while self._att_running:
            if self.armed and self.spin_motors:
                now_ms = int(time.time() * 1000) & 0xFFFFFFFF
                q = euler_to_quaternion(
                    math.radians(self.cmd_roll_deg),
                    math.radians(self.cmd_pitch_deg),
                    0.0
                )
                try:
                    self.m.mav.set_attitude_target_send(
                        now_ms,
                        self.sys, self.comp,
                        0b00000111, q,
                        0, 0, 0, self.thrust
                    )
                except Exception:
                    pass
            time.sleep(0.04)

    def set_attitude(self, roll_deg: float, pitch_deg: float):
        self.cmd_roll_deg = max(-10.0, min(10.0, roll_deg))
        self.cmd_pitch_deg = max(-10.0, min(10.0, pitch_deg))

    def poll(self):
        now = time.time()
        if now - self._last_hb > 1.0:
            self.m.mav.heartbeat_send(1, 0, 0, 0, 0)
            self._last_hb = now

        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "SERVO_OUTPUT_RAW":
                self.servos = [msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw]
            elif mtype == "LOCAL_POSITION_NED" and not self.bench_sim:
                self.ned = [msg.x, msg.y, msg.z]
            elif mtype == "ATTITUDE":
                self.yaw = msg.yaw
            elif mtype == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.flightmode = self.m.flightmode

        if self.bench_sim:
            self.ned = list(self.sim_ned)

    def set_mode(self, mode_name: str, timeout: float = 3.0) -> bool:
        if self.use_nogps and mode_name in ("GUIDED", "BRAKE"):
            mode_name = "GUIDED_NOGPS"
            if mode_name == "BRAKE":
                self.set_attitude(0.0, 0.0)

        mode_id = self.m.mode_mapping().get(mode_name)
        if mode_id is None and mode_name == "GUIDED_NOGPS":
            mode_id = 20
        if mode_id is None:
            print(f"[MODE ERROR] Unknown flight mode: {mode_name}")
            return False

        print(f"[MODE] Switching to {mode_name} (ID: {mode_id})...")
        for _ in range(4):
            self.m.mav.command_long_send(
                self.sys, self.comp,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id, 0, 0, 0, 0, 0
            )
            t_check = time.time()
            while time.time() - t_check < 0.6:
                msg = self.m.recv_match(type=['COMMAND_ACK', 'HEARTBEAT'], blocking=True, timeout=0.1)
                if msg:
                    if msg.get_type() == 'COMMAND_ACK' and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                        if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                            print(f"[MODE CONFIRMED] Autopilot acknowledged mode {mode_name}!")
                            self.flightmode = mode_name
                            return True
                    elif msg.get_type() == 'HEARTBEAT' and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                        if msg.custom_mode == mode_id or self.m.flightmode == mode_name:
                            print(f"[MODE CONFIRMED] Autopilot verified in {mode_name} mode!")
                            self.flightmode = mode_name
                            return True
            time.sleep(0.1)
        self.flightmode = mode_name
        return False

    def arm_and_takeoff(self, target_alt: float = 5.0):
        if self.use_nogps:
            print("\n" + "#" * 68)
            print(" BENCHTOP MOTOR ACTIVATION PROCEDURE (PROPELLERS MUST BE OFF! ⚠️)")
            print(f"  1. Switching Cube Orange+ to GUIDED_NOGPS mode.")
            print(f"  2. Arming physical motors at safe bench throttle ({self.thrust*100:.0f}% PWM).")
            print(f"  3. Simulating takeoff climb to {target_alt:.1f}m AGL.")
            print("#" * 68 + "\n")
            self.set_mode("GUIDED_NOGPS")
            time.sleep(0.2)
            print(f"[DRONE] Arming motors at bench thrust {self.thrust*100:.0f}% in GUIDED_NOGPS...")
            for _ in range(8):
                self.m.mav.command_long_send(
                    self.sys, self.comp,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 21196, 0, 0, 0, 0, 0
                )
                time.sleep(0.25)
                self.poll()
                if self.armed:
                    rps1 = max(0, int((self.servos[0] - 1000) * 0.12))
                    rps2 = max(0, int((self.servos[1] - 1000) * 0.12))
                    rps3 = max(0, int((self.servos[2] - 1000) * 0.12))
                    rps4 = max(0, int((self.servos[3] - 1000) * 0.12))
                    print(f"[DRONE] Motors ARMED! Active ESC PWM: M1={self.servos[0]}(~{rps1}rps) M2={self.servos[1]}(~{rps2}rps) M3={self.servos[2]}(~{rps3}rps) M4={self.servos[3]}(~{rps4}rps)")
                    break
            t0 = time.time()
            climb_time = 3.5
            while time.time() - t0 < climb_time:
                frac = min(1.0, (time.time() - t0) / climb_time)
                self.sim_ned[2] = -target_alt * frac
                self.ned = list(self.sim_ned)
                self.poll()
                time.sleep(0.05)
            self.sim_ned[2] = -target_alt
            self.ned = list(self.sim_ned)
            print(f"[BENCH-SIM] Target altitude reached: {target_alt:.1f}m AGL")
        else:
            self.set_mode("GUIDED")
            time.sleep(0.5)
            print("[DRONE] Arming motors in GUIDED mode...")
            for _ in range(8):
                self.m.mav.command_long_send(
                    self.sys, self.comp,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 21196, 0, 0, 0, 0, 0
                )
                time.sleep(0.3)
                self.poll()
                if self.armed:
                    print("[DRONE] Motors ARMED successfully!")
                    break

            print(f"[DRONE] Commanding takeoff to {target_alt:.1f}m AGL...")
            self.m.mav.command_long_send(
                self.sys, self.comp,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, target_alt
            )

    def rearm(self):
        """Re-assert arming if autopilot disarms unexpectedly on bench."""
        self.set_mode("GUIDED_NOGPS")
        time.sleep(0.1)
        self.m.mav.command_long_send(
            self.sys, self.comp,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0
        )
        time.sleep(0.2)
        self.poll()

    def setpoint(self, north: float, east: float, down: float, yaw: Optional[float] = None):
        """Sends MAVLink local NED position target with optional heading lock."""
        if not self.spin_motors:
            if yaw is not None:
                type_mask = 0b100111111000  # use position + yaw
                yaw_cmd = yaw
            else:
                type_mask = 0b110111111000  # use position only
                yaw_cmd = 0.0
            self.m.mav.set_position_target_local_ned_send(
                0, self.sys, self.comp,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                north, east, down,
                0, 0, 0,
                0, 0, 0,
                yaw_cmd, 0
            )

        if self.bench_sim:
            now = time.time()
            dt = max(0.001, min(0.2, now - self._last_sim_step))
            self._last_sim_step = now
            d_n = north - self.sim_ned[0]
            d_e = east - self.sim_ned[1]
            dist_horiz = math.hypot(d_n, d_e)
            step_h = min(self.sim_speed * dt, dist_horiz)
            if dist_horiz > 1e-4:
                self.sim_ned[0] += (d_n / dist_horiz) * step_h
                self.sim_ned[1] += (d_e / dist_horiz) * step_h

            d_d = down - self.sim_ned[2]
            dist_vert = abs(d_d)
            step_v = min(0.8 * dt, dist_vert)
            if dist_vert > 1e-4:
                self.sim_ned[2] += math.copysign(step_v, d_d)
            else:
                self.sim_ned[2] = down

            if yaw is not None:
                self.yaw = yaw
            self.ned = list(self.sim_ned)

    def land(self):
        print("[DRONE] Commanding LAND...")
        if self.spin_motors:
            print("[DRONE] Ramping down motor throttle safely...")
            t_ramp = time.time()
            init_thrust = self.thrust
            while time.time() - t_ramp < 2.0:
                frac = max(0.0, 1.0 - (time.time() - t_ramp) / 2.0)
                self.thrust = init_thrust * frac
                self.set_attitude(0.0, 0.0)
                self.poll()
                time.sleep(0.05)
            self.thrust = 0.0
            self._att_running = False

            print("[DRONE] Disarming motors...")
            for _ in range(5):
                self.m.mav.command_long_send(
                    self.sys, self.comp,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                time.sleep(0.1)

            self.m.mav.param_set_send(self.sys, self.comp, b"DISARM_DELAY", 10.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            self.set_mode("STABILIZE")
            self.armed = False
            print("[COMPLETE] Multirotor safely disarmed!")
        else:
            self.set_mode("LAND")


# ==============================================================================
# 6. LAWNMOWER SERPENTINE ROUTE GENERATOR
# ==============================================================================

def build_lawnmower_route(
    north_start: float, north_end: float,
    east_start: float, east_end: float,
    step: float = 4.0
) -> List[Tuple[float, float]]:
    """Generates a serpentine survey grid starting at (north_start, east_start)."""
    route = []
    east_cols = np.arange(east_start, east_end + (0.1 if east_end >= east_start else -0.1),
                          step if east_end >= east_start else -step)
    heading_north = True
    for e in east_cols:
        n0, n1 = (north_start, north_end) if heading_north else (north_end, north_start)
        route.append((float(n0), float(e)))
        route.append((float(n1), float(e)))
        heading_north = not heading_north
    return route


# ==============================================================================
# 7. MASTER FLIGHT MISSION CONTROLLER (12-STAGE ORCHESTRATION)
# ==============================================================================

class BannerCorridorQRMission:
    def __init__(self, args):
        self.args = args
        if args.spin_motors:
            args.bench_sim = True

        self.drone = DroneController(
            url=args.url,
            baud=args.baud,
            bench_sim=args.bench_sim,
            sim_speed=args.speed,
            spin_motors=args.spin_motors,
            thrust=args.thrust,
            use_nogps=(args.guided_nogps or args.spin_motors or args.bench_sim)
        )

        # Video stream
        cam_src = args.camera_src
        if not cam_src:
            cam_src = f"rtsp://192.168.144.25:8554/{args.stream}.264"
        self.worker = SiyiStreamWorker(capture_source=cam_src)

        # Vision Engines
        print("[VISION] Initializing Dual Banner Detector & QR Engines...")
        self.banner_detector = BannerDetector(ocr_threads=args.ocr_threads, verbose=args.verbose)
        self.qr_detector = cv2.QRCodeDetector()
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            self.pyzbar_decode = pyzbar_decode
        except ImportError:
            self.pyzbar_decode = None

        # Mission State Caches
        self.ref_grid = None
        self.ref_warped = None
        self.ref_text = ""
        self.state_text = "INITIALIZING"
        self.sway_x = 0.0
        self.green_banner_ned = [0.0, 0.0, 0.0]
        self.blue_banner_ned = [0.0, 0.0, 0.0]
        self.hud_countdown = 0.0
        self.hud_score = 0.0
        self.active_banner_color = "green"

        # Bounding box visual retention cache
        self.last_target_pts = None
        self.last_target_type = None
        self.last_score_or_dist = 0.0
        self.last_subtext = ""
        self.last_target_time = 0.0

        self.show_gui = args.show_camera and ("DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
        if self.show_gui:
            cv2.namedWindow("AeroTHON Real-World Mission HUD", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AeroTHON Real-World Mission HUD", 960, 540)

    # ------------------------------------------------------------------
    # Visual HUD Overlay
    # ------------------------------------------------------------------
    def update_hud(
        self,
        target_pts: Optional[np.ndarray] = None,
        target_type: Optional[str] = None,
        score_or_dist: float = 0.0,
        subtext: str = ""
    ):
        if not self.show_gui:
            return
        frame = self.worker.get_frame()
        if frame is None:
            vis = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(vis, "WAITING FOR CAMERA STREAM...", (240, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("AeroTHON Real-World Mission HUD", vis)
            cv2.waitKey(1)
            return

        vis = frame.copy()
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2

        # Reticle at image optical center
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.circle(vis, (cx, cy), 18, (0, 255, 255), 1)

        # Update or retain bounding box cache
        now = time.time()
        if target_pts is not None and len(target_pts) > 0:
            self.last_target_pts = target_pts
            self.last_target_type = target_type
            self.last_score_or_dist = score_or_dist
            self.last_subtext = subtext
            self.last_target_time = now
        elif (now - self.last_target_time) < 1.2 and self.last_target_pts is not None:
            target_pts = self.last_target_pts
            target_type = self.last_target_type
            score_or_dist = self.last_score_or_dist
            subtext = self.last_subtext

        # Highlight detected object (Banner or QR) with clear bounding boxes and corner tags
        if target_pts is not None and len(target_pts) > 0:
            p = np.int32(target_pts).reshape(-1, 2)
            if target_type == "QR":
                col = (0, 255, 0) if score_or_dist >= self.args.match_thresh else (0, 165, 255)
                lbl = f"QR TARGET [Sim: {score_or_dist*100:.1f}%]" if score_or_dist > 0 else f"QR CODE [{subtext}]"
            elif target_type == "BLUE_BANNER":
                col = (255, 140, 0)  # Bright Blue/Cyan in BGR
                lbl = f"BLUE BANNER [{subtext}]"
            else:
                col = (0, 255, 0)    # Green in BGR
                lbl = f"GREEN BANNER [{subtext}]"

            # 1. Bounding polygon outline
            cv2.polylines(vis, [p], True, col, 3)

            # 2. Corner markers with coordinates
            corner_names = ("TL", "TR", "BR", "BL")
            corner_offsets = ((-25, -10), (10, -10), (10, 20), (-25, 20))
            for i, (pt, name, (dx, dy)) in enumerate(zip(p[:4], corner_names, corner_offsets)):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 7, col, 1)
                cv2.putText(vis, f"{name}({int(pt[0])},{int(pt[1])})", (int(pt[0]) + dx, int(pt[1]) + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            # 3. Centroid and line from reticle
            tcx, tcy = int(np.mean(p[:, 0])), int(np.mean(p[:, 1]))
            cv2.circle(vis, (tcx, tcy), 7, (0, 0, 255), -1)
            cv2.circle(vis, (tcx, tcy), 12, (255, 255, 255), 1)
            cv2.line(vis, (cx, cy), (tcx, tcy), (255, 255, 0), 2)

            # 4. Text banner badge with solid background
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            lbl_x = max(10, min(w - tw - 15, tcx - tw // 2))
            lbl_y = max(35, tcy - 20)
            cv2.rectangle(vis, (lbl_x - 5, lbl_y - th - 5), (lbl_x + tw + 5, lbl_y + 5), (20, 20, 20), -1)
            cv2.rectangle(vis, (lbl_x - 5, lbl_y - th - 5), (lbl_x + tw + 5, lbl_y + 5), col, 1)
            cv2.putText(vis, lbl, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)

            # 5. Centroid offset label
            c_lbl = f"C({tcx},{tcy}) Offset: dx={tcx-cx:+d}px dy={tcy-cy:+d}px"
            cv2.putText(vis, c_lbl, (lbl_x, lbl_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

        # Header status bar
        cv2.rectangle(vis, (0, 0), (w, 38), (20, 20, 20), -1)
        cv2.putText(vis, f"MISSION: {self.state_text}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        alt_str = f"Alt: {-self.drone.ned[2]:.2f}m | NED: ({self.drone.ned[0]:+.1f}, {self.drone.ned[1]:+.1f}) | {self.drone.flightmode}"
        cv2.putText(vis, alt_str, (max(300, w - 460), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # Reference QR thumbnail cache
        if self.ref_warped is not None:
            thumb = cv2.resize(self.ref_warped, (80, 80))
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR) if len(thumb.shape) == 2 else thumb
            vis[45:125, w - 90:w - 10] = thumb_bgr
            cv2.rectangle(vis, (w - 92, 43), (w - 8, 127), (0, 255, 0), 2)
            cv2.putText(vis, "REF QR", (w - 85, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

        # Alignment countdown timer overlay
        if self.hud_countdown > 0:
            cv2.rectangle(vis, (cx - 160, h - 52), (cx + 160, h - 12), (0, 110, 0), -1)
            cv2.putText(vis, f"ALIGN HOLD: {self.hud_countdown:.1f}s", (cx - 145, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Motor ESC PWM & RPM/RPS telemetry bar
        if self.drone.spin_motors or self.args.bench_sim:
            cv2.rectangle(vis, (10, h - 72), (540, h - 8), (20, 20, 20), -1)
            cv2.rectangle(vis, (10, h - 72), (540, h - 8), (0, 255, 255), 1)
            rps1 = max(0, int((self.drone.servos[0] - 1000) * 0.12))
            rps2 = max(0, int((self.drone.servos[1] - 1000) * 0.12))
            rps3 = max(0, int((self.drone.servos[2] - 1000) * 0.12))
            rps4 = max(0, int((self.drone.servos[3] - 1000) * 0.12))
            m_txt = f"MOTORS: M1={self.drone.servos[0]}(~{rps1}rps) M2={self.drone.servos[1]}(~{rps2}rps) M3={self.drone.servos[2]}(~{rps3}rps) M4={self.drone.servos[3]}(~{rps4}rps)"
            tilt_txt = f"Cmd Tilt: Roll={self.drone.cmd_roll_deg:+.1f} deg | Pitch={self.drone.cmd_pitch_deg:+.1f} deg | Mode: {self.drone.flightmode}"
            cv2.putText(vis, m_txt, (15, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
            cv2.putText(vis, tilt_txt, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)

        cv2.imshow("AeroTHON Real-World Mission HUD", vis)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # QR Detection Helper
    # ------------------------------------------------------------------
    def detect_qr(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray, str]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, pts = self.qr_detector.detect(gray)
        text = ""
        if (not ok or pts is None) and self.pyzbar_decode is not None:
            decoded = self.pyzbar_decode(gray)
            if decoded:
                poly = decoded[0].polygon
                if len(poly) == 4:
                    pts = np.array([[p.x, p.y] for p in poly], dtype=np.float32).reshape(1, 4, 2)
                    ok = True
                    text = decoded[0].data.decode('utf-8', errors='ignore')
        if ok and pts is not None and not text:
            try:
                txt, _, _ = self.qr_detector.detectAndDecode(gray)
                if txt:
                    text = txt
            except Exception:
                pass
        return ok, pts, gray, text

    # ------------------------------------------------------------------
    # Slewed Setpoint Flight Helper
    # ------------------------------------------------------------------
    def fly_to(
        self, target_ned: List[float], timeout: float = 25.0,
        tol: float = 0.35, speed: Optional[float] = None, yaw: Optional[float] = None
    ) -> bool:
        sp_speed = speed or self.args.speed
        cur_sp = list(self.drone.ned)
        dt = 0.05
        t0 = time.time()
        last_log = 0.0
        while time.time() - t0 < timeout:
            self.drone.poll()
            self.update_hud()
            dist_3d = math.dist(target_ned, self.drone.ned)
            dist_h = math.hypot(target_ned[0] - self.drone.ned[0], target_ned[1] - self.drone.ned[1])
            dist_v = abs(target_ned[2] - self.drone.ned[2])
            if dist_h < tol and dist_v < 0.4:
                return True

            if self.drone.spin_motors and dist_h > 0.25:
                dn = target_ned[0] - self.drone.ned[0]
                de = target_ned[1] - self.drone.ned[1]
                t_pitch = max(-6.0, min(6.0, -(dn / dist_h) * 5.5))
                t_roll = max(-6.0, min(6.0, (de / dist_h) * 5.5))
                self.drone.set_attitude(t_roll, t_pitch)

            now_t = time.time()
            if self.drone.spin_motors and (now_t - last_log >= 1.5):
                last_log = now_t
                print(f"  [MOTORS FLY] ESC: M1={self.drone.servos[0]} M2={self.drone.servos[1]} M3={self.drone.servos[2]} M4={self.drone.servos[3]} | Tilt: Pitch={self.drone.cmd_pitch_deg:+.1f} Roll={self.drone.cmd_roll_deg:+.1f}")

            # Slew horizontal setpoint
            d_sp_h = math.hypot(target_ned[0] - cur_sp[0], target_ned[1] - cur_sp[1])
            step_h = min(sp_speed * dt, d_sp_h)
            if d_sp_h > 1e-4:
                cur_sp[0] += (target_ned[0] - cur_sp[0]) * step_h / d_sp_h
                cur_sp[1] += (target_ned[1] - cur_sp[1]) * step_h / d_sp_h

            # Slew vertical setpoint
            d_sp_v = abs(target_ned[2] - cur_sp[2])
            step_v = min(0.6 * dt, d_sp_v)
            if d_sp_v > 1e-4:
                cur_sp[2] += math.copysign(step_v, target_ned[2] - cur_sp[2])
            else:
                cur_sp[2] = target_ned[2]

            self.drone.setpoint(*cur_sp, yaw=yaw)
            time.sleep(dt)
        return False

    # ------------------------------------------------------------------
    # Closed-Loop PBVS Centering for Banners
    # ------------------------------------------------------------------
    def center_over_banner(
        self, color: str = "green", target_ned: Optional[List[float]] = None,
        max_iters: int = 6, tol_px: float = 25.0
    ) -> Tuple[List[float], bool]:
        self.state_text = f"{color.upper()} BANNER CENTERING"
        target = list(target_ned if target_ned is not None else self.drone.ned)
        locked = False

        for iteration in range(1, max_iters + 1):
            self.drone.poll()
            frame = self.worker.get_frame()
            if frame is None:
                time.sleep(0.08)
                continue

            meas = self.banner_detector.detect(
                frame, color=color, alt_m=-self.drone.ned[2], hfov_rad=self.args.hfov_rad
            )
            if meas is None:
                print(f"[CENTERING] Iter {iteration}/{max_iters}: {color} banner not visible, holding...")
                if self.drone.spin_motors:
                    self.drone.set_attitude(0.0, 0.0)
                self.fly_to(target, timeout=1.0)
                continue

            fw, fh = meas["size"]
            cx, cy = fw / 2.0, fh / 2.0
            err_u = meas["u"] - cx
            err_v = meas["v"] - cy
            err_px = math.hypot(err_u, err_v)

            # Active visual servoing tilt commanded directly to motors
            if self.drone.spin_motors:
                tilt_r = max(-8.0, min(8.0, (err_u / (fw * 0.5)) * 7.5))
                tilt_p = max(-8.0, min(8.0, -(err_v / (fh * 0.5)) * 7.5))
                self.drone.set_attitude(tilt_r, tilt_p)
                rps1 = max(0, int((self.drone.servos[0] - 1000) * 0.12))
                rps2 = max(0, int((self.drone.servos[1] - 1000) * 0.12))
                rps3 = max(0, int((self.drone.servos[2] - 1000) * 0.12))
                rps4 = max(0, int((self.drone.servos[3] - 1000) * 0.12))
                print(f"  [ALIGN TILT] {color.upper()} Banner Offset: ({err_u:+.1f}, {err_v:+.1f})px | Cmd Tilt: Roll={tilt_r:+.1f}° Pitch={tilt_p:+.1f}° | ESC: M1={self.drone.servos[0]}(~{rps1}rps) M2={self.drone.servos[1]}(~{rps2}rps) M3={self.drone.servos[2]}(~{rps3}rps) M4={self.drone.servos[3]}(~{rps4}rps)")

            print(f"[CENTERING] Iter {iteration}/{max_iters}: centroid ({meas['u']:.1f}, {meas['v']:.1f})px | Offset: {err_px:.1f}px | {'PARTIAL' if meas['partial'] else 'FULL'}")

            self.update_hud(target_pts=meas["quad"], target_type=f"{color.upper()}_BANNER", subtext=meas["source"])

            if err_px <= tol_px and not meas["partial"]:
                print(f"[CENTERING] Lock acquired! Centroid offset ({err_px:.1f}px) <= tol ({tol_px:.1f}px).")
                if self.drone.spin_motors:
                    self.drone.set_attitude(0.0, 0.0)
                locked = True
                break

            dn, de = pixel_to_ned_offset(
                meas["u"], meas["v"], fw, fh, -self.drone.ned[2], self.drone.yaw, self.args.hfov_rad
            )
            target[0] += dn * 0.70
            target[1] += de * 0.70

            t_settle = time.time()
            while time.time() - t_settle < 1.0:
                self.drone.poll()
                self.drone.setpoint(*target)
                self.update_hud(target_pts=meas["quad"], target_type=f"{color.upper()}_BANNER")
                time.sleep(0.05)

        return target, locked

    # ------------------------------------------------------------------
    # Master Mission Execution Loop
    # ------------------------------------------------------------------
    def run(self):
        a = self.args

        # ==============================================================
        # STAGE 1: Autonomous Takeoff to 5.0m AGL
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 1: TAKEOFF TO {a.alt:.1f}m AGL\n{'='*70}")
        self.state_text = f"STAGE 1: TAKEOFF ({a.alt:.1f}m)"
        self.drone.arm_and_takeoff(a.alt)

        t0 = time.time()
        while time.time() - t0 < 25.0:
            self.drone.poll()
            self.update_hud()
            curr_alt = -self.drone.ned[2]
            print(f"[STATUS] Altitude: {curr_alt:.2f}m / {a.alt:.1f}m", end='\r')
            if curr_alt >= a.alt * 0.85:
                print(f"\n[STATUS] Target altitude reached: {curr_alt:.2f}m")
                break
            time.sleep(0.08)

        # Hover stabilization at (0, 0)
        t_stab = time.time()
        while time.time() - t_stab < 1.5:
            self.drone.poll()
            self.drone.setpoint(0.0, 0.0, -a.alt, yaw=0.0)
            self.update_hud()
            time.sleep(0.05)

        # ==============================================================
        # STAGE 2: Surge 1 & Memorize Reference QR
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 2: SURGE 1 ({a.surge1:.1f}m) & SCAN REFERENCE QR\n{'='*70}")
        self.state_text = f"STAGE 2: SURGE 1 ({a.surge1:.1f}m)"
        sp_surge1 = [a.surge1, 0.0, -a.alt]
        self.fly_to(sp_surge1, timeout=12.0)

        self.state_text = "STAGE 2: SCANNING REFERENCE QR"
        print("[DRONE] Scanning takeoff reference QR code...")
        t_qr = time.time()
        while self.ref_grid is None and time.time() - t_qr < 12.0:
            self.drone.poll()
            self.drone.setpoint(*sp_surge1, yaw=0.0)
            frame = self.worker.get_frame()
            if frame is not None:
                ok, pts, gray, text = self.detect_qr(frame)
                if ok and pts is not None and len(pts) > 0:
                    grid, warped = extract_qr_features(gray, pts, grid_size=21)
                    self.ref_grid = grid
                    self.ref_warped = warped
                    self.ref_text = text
                    save_path = "/home/suasrasp/aerothon_ws/reference_qr_memorized.png"
                    cv2.imwrite(save_path, warped)
                    print(f"\n[SUCCESS] REFERENCE QR MEMORIZED! Payload: '{self.ref_text}'")
                    self.update_hud(target_pts=pts, target_type="QR", score_or_dist=1.0)
                    break
                self.update_hud()
            time.sleep(0.08)

        if self.ref_grid is None:
            print("[WARN] Visual scan timed out. Loading reference fallback model...")
            ref_path = "/home/suasrasp/aerothon_ws/reference_qr_memorized.png"
            if not os.path.exists(ref_path):
                ref_path = "/home/suasrasp/aerothon_ws/src/aerothon_sim/models/qr_initial/materials/textures/qr_initial.png"
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                ok, pts = self.qr_detector.detect(ref_img)
                if ok and pts is not None:
                    self.ref_grid, self.ref_warped = extract_qr_features(ref_img, pts, grid_size=21)
                    self.ref_text = "AEROTHON_REF"
                    print("[VISION] Reference QR loaded from cached file fallback!")

        # ==============================================================
        # STAGE 3: Surge 2 Forward
        # ==============================================================
        surge2_north = a.surge1 + a.surge2
        print(f"\n{'='*70}\nSTAGE 3: SURGE 2 ({a.surge2:.1f}m) -> North={surge2_north:.1f}m\n{'='*70}")
        self.state_text = f"STAGE 3: SURGE 2 ({a.surge2:.1f}m)"
        sp_surge2 = [surge2_north, 0.0, -a.alt]
        self.fly_to(sp_surge2, timeout=12.0)

        # ==============================================================
        # STAGE 4: Sway Left (West / Body -Y) & Search Green Banner
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 4: SWAY LEFT (max {a.max_sway:.1f}m) & DETECT GREEN BANNER\n{'='*70}")
        self.state_text = "STAGE 4: SWAY LEFT (SEARCH GREEN BANNER)"
        found_banner = None
        cur_e = 0.0
        dt = 0.05
        t_sway = time.time()

        while abs(cur_e) <= a.max_sway:
            self.drone.poll()
            frame = self.worker.get_frame()
            if frame is not None:
                meas = self.banner_detector.detect(
                    frame, color="green", alt_m=-self.drone.ned[2], hfov_rad=a.hfov_rad
                )
                if meas is not None:
                    found_banner = meas
                    self.update_hud(target_pts=meas["quad"], target_type="GREEN_BANNER", subtext=meas["source"])
                    print(f"\n[BANNER DETECTED!] Type: GREEN | Source: {meas['source']} | Current Sway: {abs(self.drone.ned[1]):.2f}m")
                    break
                self.update_hud()

            cur_e -= a.sway_speed * dt
            sp_sway = [surge2_north, cur_e, -a.alt]
            self.drone.setpoint(*sp_sway, yaw=0.0)
            if self.drone.spin_motors:
                self.drone.set_attitude(-4.5, 0.0)  # Tilt left
            time.sleep(dt)

        if found_banner is None:
            if a.bench_sim:
                print(f"[BENCH-SIM] Green Banner simulated at benchmark sway (North={surge2_north:+.1f}m, East=-7.40m, x=7.4m)")
                found_banner = {
                    "u": 640.0, "v": 360.0, "size": (1280, 720),
                    "source": "BENCH_SIM", "text": "AeroTHON 2026", "partial": False,
                    "quad": np.array([[200, 300], [900, 300], [900, 380], [200, 380]], dtype=np.float32)
                }
                self.drone.sim_ned[1] = -7.40
                self.drone.ned[1] = -7.40
            else:
                print("\n[RESULT] Green Banner not detected within max sway. Commanding LAND...")
                self.drone.land()
                return

        # Instant BRAKE to stop sideways momentum
        print("[BRAKE] Halting lateral momentum...")
        if self.drone.spin_motors:
            self.drone.set_attitude(0.0, 0.0)
        else:
            self.drone.set_mode("BRAKE")
        time.sleep(1.2)

        # ==============================================================
        # STAGE 5: Green Banner Visual Servoing & Measure Sway x
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 5: GREEN BANNER ALIGNMENT & MEASURE SWAY x\n{'='*70}")
        self.state_text = "STAGE 5: GREEN BANNER ALIGNMENT"
        self.drone.set_mode("GUIDED")
        time.sleep(0.2)

        target_banner_ned = list(self.drone.ned)
        if found_banner is not None:
            dn, de = pixel_to_ned_offset(
                found_banner["u"], found_banner["v"],
                found_banner["size"][0], found_banner["size"][1],
                -self.drone.ned[2], self.drone.yaw, a.hfov_rad
            )
            target_banner_ned[0] += dn
            target_banner_ned[1] += de
            self.fly_to(target_banner_ned, timeout=8.0)

        # Closed-loop centering
        target_banner_ned, _ = self.center_over_banner(
            color="green", target_ned=target_banner_ned, max_iters=a.center_iters, tol_px=a.center_tol
        )
        self.green_banner_ned = list(target_banner_ned)

        # Measure sway distance x (leftward distance from takeoff line)
        self.sway_x = -self.drone.ned[1]
        print(f"\n{'*'*70}")
        print(f"[MEASUREMENT] Green Banner Position: North={self.drone.ned[0]:+.2f}m, East={self.drone.ned[1]:+.2f}m")
        print(f"[MEASUREMENT] Total leftward sway distance x = {self.sway_x:.2f} meters")
        print(f"{'*'*70}\n")

        # ==============================================================
        # STAGE 6: Descend to Corridor Altitude (3.0m AGL)
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 6: DESCEND TO CORRIDOR ALTITUDE ({a.corridor_alt:.1f}m AGL)\n{'='*70}")
        self.state_text = f"STAGE 6: DESCENDING TO {a.corridor_alt:.1f}m"
        target_banner_ned[2] = -a.corridor_alt
        self.fly_to(target_banner_ned, timeout=12.0)
        time.sleep(1.0)

        # ==============================================================
        # STAGE 7: Surge Forward Through Green Corridor
        # ==============================================================
        corridor_exit_north = target_banner_ned[0] + a.corridor_surge
        print(f"\n{'='*70}\nSTAGE 7: SURGE {a.corridor_surge:.1f}m THROUGH CORRIDOR -> North={corridor_exit_north:.1f}m\n{'='*70}")
        self.state_text = f"STAGE 7: SURGE CORRIDOR ({a.corridor_surge:.1f}m)"
        sp_corridor_end = [corridor_exit_north, target_banner_ned[1], -a.corridor_alt]
        self.fly_to(sp_corridor_end, timeout=a.corridor_surge / 0.8 + 15.0, speed=0.9)
        print(f"[CORRIDOR EXIT] Exited corridor at North={self.drone.ned[0]:+.2f}m, East={self.drone.ned[1]:+.2f}m")

        # ==============================================================
        # STAGE 8: Lateral Positioning: Go Left by (19 - x) meters
        # ==============================================================
        wall_sway_step = a.wall_offset - self.sway_x
        target_wall_east = self.drone.ned[1] - wall_sway_step  # further left (negative East)
        print(f"\n{'='*70}\nSTAGE 8: LATERAL SHIFT LEFT BY ({a.wall_offset:.1f} - x) = {wall_sway_step:.2f}m (1m from wall)\n{'='*70}")
        self.state_text = f"STAGE 8: MOVE LEFT {wall_sway_step:.2f}m"
        print(f"[NAVIGATION] Target East coordinate: {target_wall_east:+.2f}m (1m away from wall at -20m)")

        # Climb back to survey altitude
        sp_wall = [self.drone.ned[0] + 1.0, target_wall_east, -a.alt]
        self.fly_to(sp_wall, timeout=20.0)
        print(f"[SURVEY ZONE ENTRY] Positioned at North={self.drone.ned[0]:+.2f}m, East={self.drone.ned[1]:+.2f}m")

        # ==============================================================
        # STAGE 9: Lawnmower Search Pattern for Candidate QR
        # ==============================================================
        n_start = self.drone.ned[0]
        n_end = n_start + a.lawnmower_length
        e_start = self.drone.ned[1]
        e_end = -1.0  # Sweep from near-wall East (-19m) across toward center

        route = build_lawnmower_route(
            north_start=n_start, north_end=n_end,
            east_start=e_start, east_end=e_end,
            step=a.step
        )
        print(f"\n{'='*70}\nSTAGE 9: LAWNMOWER QR RECONNAISSANCE ({len(route)} WAYPOINTS)\n{'='*70}")
        self.state_text = "STAGE 9: LAWNMOWER QR SEARCH"

        target_qr_ned = None
        for wp_idx, (wp_n, wp_e) in enumerate(route):
            tgt_ned = [wp_n, wp_e, -a.alt]
            print(f"[WAYPOINT {wp_idx+1}/{len(route)}] Target: North={wp_n:+.1f}m, East={wp_e:+.1f}m")
            cur_sp = list(self.drone.ned)
            dt = 0.05

            while True:
                self.drone.poll()
                if self.drone.spin_motors and not self.drone.armed:
                    self.drone.rearm()

                frame = self.worker.get_frame()
                detected_pts = None
                sim_score = 0.0
                is_match = False

                if frame is not None and self.ref_grid is not None:
                    ok, detected_pts, gray, cand_txt = self.detect_qr(frame)
                    if ok and detected_pts is not None and len(detected_pts) > 0:
                        cand_grid, _ = extract_qr_features(gray, detected_pts, grid_size=21)
                        sim_score = compute_similarity(self.ref_grid, cand_grid)

                        if cand_txt and self.ref_text and cand_txt == self.ref_text:
                            is_match = True
                            sim_score = 1.0
                        elif sim_score >= a.match_thresh:
                            is_match = True

                        if is_match:
                            p = detected_pts.reshape(-1, 2)
                            q_u, q_v = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
                            fw, fh = frame.shape[1], frame.shape[0]
                            dn, de = pixel_to_ned_offset(
                                q_u, q_v, fw, fh, -self.drone.ned[2], self.drone.yaw, a.hfov_rad
                            )
                            target_qr_ned = [self.drone.ned[0] + dn, self.drone.ned[1] + de, -a.alt]

                            print(f"\n{'!'*70}")
                            print(f"[MATCH FOUND!] Candidate Similarity: {sim_score*100:.1f}% | Payload: '{cand_txt}'")
                            print(f"[TARGET QR ESTIMATE] North={target_qr_ned[0]:+.2f}m, East={target_qr_ned[1]:+.2f}m")
                            print(f"{'!'*70}\n")

                            print("[BRAKE] Target QR matched! Freezing momentum...")
                            if self.drone.spin_motors:
                                self.drone.set_attitude(0.0, 0.0)
                            else:
                                self.drone.set_mode("BRAKE")
                            self.update_hud(target_pts=detected_pts, target_type="QR", score_or_dist=sim_score)
                            break

                self.update_hud(target_pts=detected_pts, target_type="QR", score_or_dist=sim_score)
                if target_qr_ned is not None:
                    break

                if a.bench_sim and wp_idx >= 1 and target_qr_ned is None:
                    print(f"\n[BENCH-SIM] Candidate Twin QR match simulated at Waypoint {wp_idx+1} ({wp_n:+.1f}m, {wp_e:+.1f}m)!")
                    target_qr_ned = [wp_n, wp_e, -a.alt]
                    self.drone.sim_ned[0] = wp_n
                    self.drone.sim_ned[1] = wp_e
                    self.drone.ned[0] = wp_n
                    self.drone.ned[1] = wp_e
                    break

                dist_to_wp = math.hypot(tgt_ned[0] - self.drone.ned[0], tgt_ned[1] - self.drone.ned[1])
                if dist_to_wp < 0.6:
                    break

                if self.drone.spin_motors:
                    wp_dn = tgt_ned[0] - self.drone.ned[0]
                    wp_de = tgt_ned[1] - self.drone.ned[1]
                    dist_m = math.hypot(wp_dn, wp_de)
                    if dist_m > 0.3:
                        wp_pitch = max(-6.0, min(6.0, -(wp_dn / dist_m) * 5.5))
                        wp_roll = max(-6.0, min(6.0, (wp_de / dist_m) * 5.5))
                        self.drone.set_attitude(wp_roll, wp_pitch)

                d_sp = math.hypot(tgt_ned[0] - cur_sp[0], tgt_ned[1] - cur_sp[1])
                step = min(a.speed * dt, d_sp)
                if d_sp > 1e-4:
                    cur_sp[0] += (tgt_ned[0] - cur_sp[0]) * step / d_sp
                    cur_sp[1] += (tgt_ned[1] - cur_sp[1]) * step / d_sp
                cur_sp[2] = -a.alt
                self.drone.setpoint(*cur_sp)
                time.sleep(dt)

            if target_qr_ned is not None:
                break

        if target_qr_ned is None:
            print("[WARN] Lawnmower sweep finished without matching candidate QR. Using current position.")
            target_qr_ned = list(self.drone.ned)

        # ==============================================================
        # STAGE 10: Precision QR Centering & 15-Second Hover Hold
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 10: PRECISION CENTERING & 15-SECOND HOVER HOLD\n{'='*70}")
        self.state_text = "STAGE 10: QR CENTERING & HOVER"
        self.drone.set_mode("GUIDED")
        time.sleep(0.2)
        self.fly_to(target_qr_ned, timeout=10.0, tol=0.25)

        # Fine Closed-Loop Centering
        for iteration in range(a.center_iters):
            self.drone.poll()
            frame = self.worker.get_frame()
            if frame is None:
                continue
            ok, pts, _, cand_txt = self.detect_qr(frame)
            if ok and pts is not None and len(pts) > 0:
                p = pts.reshape(-1, 2)
                u, v = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
                fw, fh = frame.shape[1], frame.shape[0]
                cx, cy = fw / 2.0, fh / 2.0
                err_u = u - cx
                err_v = v - cy
                err_px = math.hypot(err_u, err_v)

                if self.drone.spin_motors:
                    tilt_r = max(-8.0, min(8.0, (err_u / (fw * 0.5)) * 7.5))
                    tilt_p = max(-8.0, min(8.0, -(err_v / (fh * 0.5)) * 7.5))
                    self.drone.set_attitude(tilt_r, tilt_p)
                    rps1 = max(0, int((self.drone.servos[0] - 1000) * 0.12))
                    rps2 = max(0, int((self.drone.servos[1] - 1000) * 0.12))
                    rps3 = max(0, int((self.drone.servos[2] - 1000) * 0.12))
                    rps4 = max(0, int((self.drone.servos[3] - 1000) * 0.12))
                    print(f"  [ALIGN TILT] QR Centering Iter {iteration+1}/{a.center_iters}: Offset: ({err_u:+.1f}, {err_v:+.1f})px | Cmd Tilt: Roll={tilt_r:+.1f}° Pitch={tilt_p:+.1f}° | ESC: M1={self.drone.servos[0]}(~{rps1}rps) M2={self.drone.servos[1]}(~{rps2}rps) M3={self.drone.servos[2]}(~{rps3}rps) M4={self.drone.servos[3]}(~{rps4}rps)")

                self.update_hud(target_pts=pts, target_type="QR", score_or_dist=1.0)
                if err_px <= a.center_tol:
                    print(f"[QR CENTERING] Lock acquired! (Offset {err_px:.1f}px <= tol {a.center_tol:.1f}px)")
                    if self.drone.spin_motors:
                        self.drone.set_attitude(0.0, 0.0)
                    break
                dn, de = pixel_to_ned_offset(u, v, fw, fh, -self.drone.ned[2], self.drone.yaw, a.hfov_rad)
                target_qr_ned[0] += dn * 0.70
                target_qr_ned[1] += de * 0.70
                self.fly_to(target_qr_ned, timeout=1.5)

        # 15-Second Alignment Hover
        print(f"[ALIGN] Commencing rock-solid hover hold for {a.align_duration:.1f} seconds...")
        t_hold = time.time()
        last_log = 0.0
        while time.time() - t_hold < a.align_duration:
            rem = max(0.0, a.align_duration - (time.time() - t_hold))
            self.hud_countdown = rem
            self.drone.poll()
            frame = self.worker.get_frame()
            pts = None
            if frame is not None:
                ok, pts, _, _ = self.detect_qr(frame)
                if ok and pts is not None and len(pts) > 0:
                    p = pts.reshape(-1, 2)
                    u, v = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
                    fw, fh = frame.shape[1], frame.shape[0]
                    cx, cy = fw / 2.0, fh / 2.0
                    err_u, err_v = u - cx, v - cy
                    if self.drone.spin_motors:
                        tilt_r = max(-8.0, min(8.0, (err_u / (fw * 0.5)) * 7.5))
                        tilt_p = max(-8.0, min(8.0, -(err_v / (fh * 0.5)) * 7.5))
                        self.drone.set_attitude(tilt_r, tilt_p)
                elif self.drone.spin_motors:
                    self.drone.set_attitude(0.0, 0.0)

            self.drone.setpoint(*target_qr_ned)
            self.update_hud(target_pts=pts, target_type="QR", score_or_dist=1.0)
            now_t = time.time()
            if self.drone.spin_motors and (now_t - last_log >= 1.0):
                last_log = now_t
                rps1 = max(0, int((self.drone.servos[0] - 1000) * 0.12))
                rps2 = max(0, int((self.drone.servos[1] - 1000) * 0.12))
                rps3 = max(0, int((self.drone.servos[2] - 1000) * 0.12))
                rps4 = max(0, int((self.drone.servos[3] - 1000) * 0.12))
                print(f"[ALIGN HOVER] Remaining: {rem:4.1f}s | Cmd Tilt: Roll={self.drone.cmd_roll_deg:+.1f}° Pitch={self.drone.cmd_pitch_deg:+.1f}° | ESC: M1={self.drone.servos[0]}(~{rps1}rps) M2={self.drone.servos[1]}(~{rps2}rps) M3={self.drone.servos[2]}(~{rps3}rps) M4={self.drone.servos[3]}(~{rps4}rps)")
            time.sleep(0.05)
        self.hud_countdown = 0.0
        print(f"\n[ALIGN HOVER] 15-second alignment hold complete!")

        # ==============================================================
        # STAGE 11: Return Corridor (Blue Banner) Detection & Surge Back
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 11: RETURN CORRIDOR (BLUE BANNER) DETECTION & SURGE BACK\n{'='*70}")
        self.state_text = "STAGE 11: TRANSITION TO RETURN CORRIDOR"

        # Return corridor is adjacent to green corridor:
        # Green corridor lateral position was East = -sway_x
        # Return corridor offset: default +3.5m (to the right / East of green corridor)
        offset_dir = 1.0 if a.return_corridor_side == "right" else -1.0
        blue_corridor_east = -self.sway_x + (offset_dir * abs(a.return_corridor_offset))
        blue_entrance_north = corridor_exit_north + 2.0  # Just outside north exit, facing South

        print(f"[RETURN CORRIDOR] Known spatial relationship: Adjacent to Green Corridor (-{self.sway_x:.2f}m)")
        print(f"[RETURN CORRIDOR] Return Corridor East: {blue_corridor_east:+.2f}m | Entrance North: {blue_entrance_north:+.2f}m")

        # 1. Fly to standoff position outside the return corridor entrance
        sp_blue_approach = [blue_entrance_north, blue_corridor_east, -a.alt]
        self.fly_to(sp_blue_approach, timeout=18.0)

        # 2. Yaw drone 180 degrees (South) to look directly into the return corridor
        print("[RETURN CORRIDOR] Yawing 180 degrees (South) to face return corridor entrance...")
        self.state_text = "STAGE 11: YAWING SOUTH (180 deg)"
        yaw_south = math.pi
        t_yaw = time.time()
        while time.time() - t_yaw < 3.0:
            self.drone.poll()
            self.drone.setpoint(*sp_blue_approach, yaw=yaw_south)
            self.update_hud()
            time.sleep(0.05)

        # 3. Detect Blue Banner visually
        self.state_text = "STAGE 11: SEARCHING BLUE BANNER"
        print("[VISION] Searching for BLUE BANNER ('AeroTHON 2026')...")
        found_blue = None
        t_search_blue = time.time()
        while time.time() - t_search_blue < 10.0:
            self.drone.poll()
            frame = self.worker.get_frame()
            if frame is not None:
                meas = self.banner_detector.detect(
                    frame, color="blue", alt_m=-self.drone.ned[2], hfov_rad=a.hfov_rad
                )
                if meas is not None:
                    found_blue = meas
                    self.update_hud(target_pts=meas["quad"], target_type="BLUE_BANNER", subtext=meas["source"])
                    print(f"[BLUE BANNER DETECTED!] Source: {meas['source']} | Offset: u={meas['u']:.1f}, v={meas['v']:.1f}")
                    break
                self.update_hud()
            time.sleep(0.08)

        if found_blue is None and a.bench_sim:
            print(f"[BENCH-SIM] Simulated Blue Banner detected at Return Corridor entrance (North={blue_entrance_north:+.1f}m, East={blue_corridor_east:+.1f}m)")
            found_blue = {
                "u": 640.0, "v": 360.0, "size": (1280, 720),
                "source": "BENCH_SIM", "text": "AeroTHON 2026", "partial": False,
                "quad": np.array([[200, 300], [900, 300], [900, 380], [200, 380]], dtype=np.float32)
            }

        # 4. Center over Blue Banner
        blue_target_ned = list(self.drone.ned)
        if found_blue is not None:
            dn, de = pixel_to_ned_offset(
                found_blue["u"], found_blue["v"],
                found_blue["size"][0], found_blue["size"][1],
                -self.drone.ned[2], self.drone.yaw, a.hfov_rad
            )
            blue_target_ned[0] += dn
            blue_target_ned[1] += de
            self.fly_to(blue_target_ned, timeout=6.0, yaw=yaw_south)

        blue_target_ned, _ = self.center_over_banner(
            color="blue", target_ned=blue_target_ned, max_iters=4, tol_px=a.center_tol
        )
        self.blue_banner_ned = list(blue_target_ned)

        # 5. Descend to corridor altitude (3.0m AGL)
        print(f"[RETURN CORRIDOR] Descending to {a.corridor_alt:.1f}m AGL...")
        blue_target_ned[2] = -a.corridor_alt
        self.fly_to(blue_target_ned, timeout=10.0, yaw=yaw_south)

        # 6. Surge through return corridor (surging forward in body frame = South towards takeoff)
        # Heading is South (yaw = pi), so body forward (-X_NED) surges by corridor_surge
        return_exit_north = blue_target_ned[0] - a.corridor_surge
        print(f"[RETURN CORRIDOR] Surging {a.corridor_surge:.1f}m South through return corridor -> North={return_exit_north:.1f}m...")
        self.state_text = f"STAGE 11: SURGE RETURN CORRIDOR ({a.corridor_surge:.1f}m)"
        sp_return_exit = [return_exit_north, blue_target_ned[1], -a.corridor_alt]
        self.fly_to(sp_return_exit, timeout=a.corridor_surge / 0.8 + 15.0, speed=0.9, yaw=yaw_south)
        print(f"[RETURN CORRIDOR] Successfully emerged from return corridor at North={self.drone.ned[0]:+.2f}m, East={self.drone.ned[1]:+.2f}m")

        # ==============================================================
        # STAGE 12: Return to Takeoff Pad (0, 0) & Precision Land
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 12: RETURN TO TAKEOFF POINT & PRECISION LAND\n{'='*70}")
        self.state_text = "STAGE 12: RETURNING TO TAKEOFF PAD"

        # Climb to safe transit altitude and yaw North
        sp_home_transit = [0.0, 0.0, -a.alt]
        self.fly_to(sp_home_transit, timeout=20.0, yaw=0.0)

        # Descend to low hover over takeoff pad
        sp_home_low = [0.0, 0.0, -2.0]
        self.fly_to(sp_home_low, timeout=8.0, yaw=0.0)

        print("\n" + "#"*70)
        print(" MISSION OBJECTIVES FULLY ACHIEVED! COMMANDING PRECISION LAND...")
        print("#"*70 + "\n")
        self.state_text = "STAGE 12: LANDING"
        self.drone.land()

        t_land = time.time()
        while time.time() - t_land < 8.0:
            self.drone.poll()
            self.update_hud()
            time.sleep(0.1)

        print(f"\n{'='*70}\nMISSION COMPLETED SUCCESSFULLY!\n{'='*70}")

    def shutdown(self):
        self.worker.stop()
        if self.show_gui:
            cv2.destroyAllWindows()


# ==============================================================================
# 8. COMMAND LINE ARGUMENT PARSING & ENTRYPOINT
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description="AeroTHON 2026 Master Banner Corridor & QR Real-World Mission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Autopilot & Serial MAVLink
    p.add_argument("--url", default="/dev/ttyACM0", help="MAVLink serial port or URL (e.g. /dev/ttyACM0, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=921600, help="MAVLink baud rate")

    # SIYI A8 mini RTSP Camera Stream
    p.add_argument("--camera-src", default=None, help="Custom RTSP URL or device index (defaults to SIYI A8 mini RTSP)")
    p.add_argument("--stream", choices=["main", "sub"], default="main", help="SIYI RTSP stream profile")
    p.add_argument("--hfov-rad", type=float, default=1.22, help="Camera Horizontal FOV in radians (~70 deg for SIYI A8 mini)")
    p.add_argument("--ocr-threads", type=int, default=2, help="Number of RapidOCR inference threads")

    # Flight Stage Distances & Geometric Parameters (User-Configurable)
    p.add_argument("--alt", type=float, default=5.0, help="Takeoff and survey altitude in meters AGL")
    p.add_argument("--surge1", type=float, default=2.0, help="Surge 1 forward distance to scan reference QR (meters)")
    p.add_argument("--surge2", type=float, default=2.0, help="Surge 2 forward distance after scanning reference QR (meters)")
    p.add_argument("--max-sway", type=float, default=20.0, help="Maximum left sway distance to search for green banner (meters)")
    p.add_argument("--sway-speed", type=float, default=0.5, help="Sideways search speed (m/s)")
    p.add_argument("--corridor-alt", type=float, default=3.0, help="Altitude inside corridors in meters AGL")
    p.add_argument("--corridor-surge", type=float, default=10.0, help="Forward surge distance through corridor (meters)")
    p.add_argument("--wall-offset", type=float, default=19.0, help="Reference wall distance: lateral shift is (wall_offset - x) to land 1m from 20m wall")

    # Lawnmower & QR Matching Parameters
    p.add_argument("--lawnmower-length", type=float, default=18.0, help="Northward search span for lawnmower grid (meters)")
    p.add_argument("--step", type=float, default=4.0, help="Lawnmower row spacing in meters")
    p.add_argument("--match-thresh", type=float, default=0.88, help="QR similarity matching threshold [0.0 to 1.0]")
    p.add_argument("--align-duration", type=float, default=15.0, help="Hover hold duration over matched QR in seconds")
    p.add_argument("--center-tol", type=float, default=25.0, help="Visual servoing centering tolerance in pixels")
    p.add_argument("--center-iters", type=int, default=6, help="Maximum visual servoing iterations")

    # Return Corridor & Blue Banner
    p.add_argument("--return-corridor-offset", type=float, default=3.5, help="Lateral distance to adjacent return corridor relative to green corridor (meters)")
    p.add_argument("--return-corridor-side", choices=["right", "left"], default="right", help="Side of return corridor relative to forward flight (right/East or left/West)")

    # General Speed & Benchtop Testing Options
    p.add_argument("--speed", type=float, default=1.2, help="General fly-to speed in m/s")
    p.add_argument("--guided-nogps", action="store_true", help="Force GUIDED_NOGPS flight mode for bench testing without GPS lock")
    p.add_argument("--bench-sim", action="store_true", help="Run in benchtop simulation mode (emulates kinematics while keeping real SIYI camera and real Cube MAVLink commands active)")
    p.add_argument("--spin-motors", action="store_true", help="Spin physical rotors safely on bench during simulation (Props MUST be OFF!)")
    p.add_argument("--thrust", type=float, default=0.20, help="Bench motor thrust [0.10 to 0.35]")
    p.add_argument("--show-camera", action="store_true", default=True, help="Display live visual HUD overlay")
    p.add_argument("--no-camera", dest="show_camera", action="store_false", help="Run headless without GUI window")
    p.add_argument("--verbose", action="store_true", help="Print verbose vision and navigation diagnostics")

    args = p.parse_args()

    mission = BannerCorridorQRMission(args)
    try:
        mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] User interrupted mission. Commanding immediate LAND...")
        mission.drone.land()
    finally:
        mission.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
