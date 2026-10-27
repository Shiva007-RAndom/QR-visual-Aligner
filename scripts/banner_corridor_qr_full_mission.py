#!/usr/bin/env python3
"""
AeroTHON 2026 — Master Banner Corridor & QR Lawnmower Mission Controller
========================================================================
Production autonomous mission script for Holybro X500 (Cube Orange+ ArduPilot 4.5,
Raspberry Pi 5 companion computer, SIYI A8 mini RTSP gimbal camera).

Full Architecture & Sequence:
-----------------------------
  STAGE 1 : Autonomous Takeoff to transit altitude (relative to takeoff point).
            Records takeoff NED position and heading (no hard-coded North/West).
  STAGE 2 : Surge forward by --surge1 along takeoff heading & scan reference QR.
            Reference QR must come ONLY from the camera (NO loading from disk).
            If reference QR is not acquired within timeout, field QR reconnaissance
            and hover are skipped entirely, proceeding straight to corridor transit.
  STAGE 3 : Surge forward again by --surge2 along takeoff heading.
  STAGE 4 : Sway left along takeoff left axis up to --max-sway searching for Green Banner.
            Uses strict color segmentation (#00B140) + rotated rectangle classification
            (ratio >= 8.0, fill >= 0.60, full-view check, persistence >= 3 frames).
            Async RapidOCR verifies "AeroTHON" AND "2026".
            If Green Banner is NOT found: flies back to takeoff pad and lands.
  STAGE 5 : Position-Based Visual Servoing (PBVS) alignment with Green Banner.
            Reconstructs partial views along rotated long axis.
            Determines corridor heading perpendicular to banner, resolving 180° ambiguity
            and rejecting corrections > 25°.
            Measures lateral sway x along takeoff left axis.
  STAGE 6 : Descend to --corridor-alt (relative to takeoff point).
  STAGE 7 : Surge forward through Green Corridor for --corridor-surge along corridor heading.
  STAGE 8 : Move left by (19 - x) meters along corridor lateral axis (1m from 20m wall).
            Climb back to transit altitude.
  STAGE 9 : Lawnmower serpentine reconnaissance for candidate twin QR (skipped if ref QR missed).
  STAGE 10: Precision PBVS centering over target QR and 15-second steady hover lock.
  STAGE 11: Return Corridor: Predict Blue Banner (#0000FF) position from measured green banner,
            corridor heading, spacing and forward offset. Fly there at transit altitude,
            yaw to return heading (corridor_heading + 180°), and search locally (±3m lat, ±1.5m fwd).
            If Blue Banner is NOT found: does NOT descend or surge blind; flies home and lands.
            If found: PBVS align, descend, surge through return corridor by --corridor-surge.
  STAGE 12: Fly back to takeoff point at transit altitude and execute precision LAND.

Vision Test Mode:
-----------------
  python3 scripts/banner_corridor_qr_full_mission.py --vision-test green
  python3 scripts/banner_corridor_qr_full_mission.py --vision-test blue
  (Runs live camera detection with diagnostic rotated rectangles, corners, ratio, fill,
   verdicts, and 'm' key to toggle color mask overlay. No MAVLink, no flight.)
"""

import argparse
import math
import os
import queue
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
# 1. GEOMETRY, MATH & COORDINATE HELPERS
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


def get_body_axes(yaw_rad: float) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Returns (forward, left, right) unit vectors in Local NED for given heading.
    NED convention: +X = North, +Y = East.
    Yaw is measured clockwise from North (+X).
    """
    fwd = (math.cos(yaw_rad), math.sin(yaw_rad))
    left = (math.sin(yaw_rad), -math.cos(yaw_rad))
    right = (-math.sin(yaw_rad), math.cos(yaw_rad))
    return fwd, left, right


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
# 3. ASYNCHRONOUS RAPIDOCR WORKER (NON-BLOCKING)
# ==============================================================================

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


def extract_ocr_items(ocr_output: Any) -> List[Tuple[np.ndarray, str, float]]:
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


class AsyncOCRWorker:
    """
    Dedicated background thread for RapidOCR. Never blocks the flight control loop.
    Strictly verifies that BOTH 'AeroTHON' (edit distance <= 2) AND '2026' are present.
    """
    def __init__(self, threads: int = 2):
        self.lock = threading.Lock()
        self.job_queue = queue.Queue(maxsize=1)
        self.latest_match = None
        self.running = True
        self.engine = None
        self._init_ocr(threads)
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

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
            print("[VISION] RapidOCR background engine initialized.")
        except Exception as e:
            print(f"[VISION WARN] RapidOCR unavailable ({e}). Running shape-only classification.")
            self.engine = None

    def _worker_loop(self):
        while self.running:
            try:
                crop, meta = self.job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.engine is None or crop is None or crop.size == 0:
                continue

            try:
                ocr_out = self.engine(crop)
                items = extract_ocr_items(ocr_out)
                if items:
                    combined_text = " ".join([txt for _, txt, _ in items])
                    c_txt = canon(combined_text)
                    b_dist = partial_match_distance(c_txt, BRAND_C)
                    has_year = YEAR_C in c_txt
                    # Strict check: BOTH brand (dist <= 2) AND year (2026) required!
                    if b_dist <= 2 and has_year:
                        mean_score = float(np.mean([score for _, _, score in items]))
                        with self.lock:
                            self.latest_match = {
                                "meta": meta,
                                "text": combined_text,
                                "score": mean_score,
                                "time": time.time()
                            }
            except Exception:
                pass

    def submit_job(self, crop: np.ndarray, meta: Dict[str, Any]):
        if self.engine is None or crop is None:
            return
        try:
            try:
                self.job_queue.get_nowait()
            except queue.Empty:
                pass
            self.job_queue.put_nowait((crop.copy(), meta))
        except queue.Full:
            pass

    def get_latest_match(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.latest_match

    def clear_match(self):
        with self.lock:
            self.latest_match = None

    def stop(self):
        self.running = False


# ==============================================================================
# 4. COLOR-FIRST & STRICT SHAPE BANNER DETECTOR
# ==============================================================================

# Physical Banner Dimensions (AeroTHON specifications: 11.5 ft x 1 ft)
BANNER_LENGTH_M = 3.5052  # 11.5 ft
BANNER_WIDTH_M = 0.3048   # 1.0 ft


class BannerDetector:
    """
    Robust Banner Detector:
      1. Color segmentation: Dynamic RGB -> HSV target with configurable tolerance.
      2. Morphological smoothing: Open 3x3 (speckle) + Close ellipse ~1.2% width (bridge text & border).
      3. Strict Rotated-Rectangle classification (full-view, ratio >= 8.0, fill >= 0.60, persistence >= 3).
      4. Optional non-blocking RapidOCR (requires BOTH 'AeroTHON' and '2026').
      5. Pinhole center reconstruction along rotated long axis for partial views.
    """
    def __init__(self, args, verbose: bool = False):
        self.args = args
        self.verbose = verbose
        self.ocr_worker = AsyncOCRWorker(threads=args.ocr_threads)

        # Dynamic HSV target calculations from exact competition RGB colors
        # GREEN: #00B140 = RGB(0, 177, 64) -> BGR: (64, 177, 0)
        # BLUE:  #0000FF = RGB(0, 0, 255) -> BGR: (255, 0, 0)
        self.green_bgr = np.uint8([[[64, 177, 0]]])
        self.blue_bgr = np.uint8([[[255, 0, 0]]])

        self.green_hsv_target = cv2.cvtColor(self.green_bgr, cv2.COLOR_BGR2HSV)[0][0]
        self.blue_hsv_target = cv2.cvtColor(self.blue_bgr, cv2.COLOR_BGR2HSV)[0][0]

        # Persistence tracking state
        self.tracked_center: Optional[Tuple[float, float]] = None
        self.consecutive_hits: int = 0
        self.last_color: str = ""

    def get_hsv_ranges(self, color: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Calculates OpenCV HSV lower/upper bounds with wrap-around handling."""
        if color.lower() == "blue":
            target_h = int(self.blue_hsv_target[0])
            h_tol = self.args.blue_h_tol
            s_min = self.args.blue_s_min
            v_min = self.args.blue_v_min
        else:
            target_h = int(self.green_hsv_target[0])
            h_tol = self.args.green_h_tol
            s_min = self.args.green_s_min
            v_min = self.args.green_v_min

        h_low = target_h - h_tol
        h_high = target_h + h_tol

        ranges = []
        if h_low < 0:
            ranges.append((np.array([0, s_min, v_min], dtype=np.uint8), np.array([h_high, 255, 255], dtype=np.uint8)))
            ranges.append((np.array([180 + h_low, s_min, v_min], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)))
        elif h_high > 179:
            ranges.append((np.array([h_low, s_min, v_min], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)))
            ranges.append((np.array([0, s_min, v_min], dtype=np.uint8), np.array([h_high - 180, 255, 255], dtype=np.uint8)))
        else:
            ranges.append((np.array([h_low, s_min, v_min], dtype=np.uint8), np.array([h_high, 255, 255], dtype=np.uint8)))
        return ranges

    def build_color_mask(self, img_bgr: np.ndarray, color: str) -> np.ndarray:
        blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        ranges = self.get_hsv_ranges(color)
        mask = None
        for low, high in ranges:
            m = cv2.inRange(hsv, low, high)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # Morphological OPEN (3x3 rect removes noise/speckle)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # Morphological CLOSE (elliptical kernel of ~1.2% image width, joins white border & text)
        w_img = img_bgr.shape[1]
        k = max(3, int(round(0.012 * w_img)))
        k = k if k % 2 == 1 else k + 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        return closed

    def detect(
        self, frame: np.ndarray, color: str = "green", alt_rel: float = 5.0, hfov_rad: float = 1.22
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
        """
        Executes color segmentation, region extraction, strict shape checks, and persistence.
        Returns:
          - best_banner (Dict or None)
          - all_candidates (List of region diagnostic dicts)
          - closed_color_mask (for vision-test overlay)
        """
        if frame is None:
            return None, [], np.zeros((100, 100), dtype=np.uint8)

        full_h, full_w = frame.shape[:2]

        # 1. Downscale frame to ~640 px wide work frame
        work_w = 640
        scale = float(work_w) / float(full_w)
        work_h = int(round(full_h * scale))
        work_img = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)

        # 2. Color segmentation
        color_mask = self.build_color_mask(work_img, color=color)

        # 3. Extract regions (contours)
        min_area_frac = self.args.green_min_area if color.lower() == "green" else self.args.blue_min_area
        min_area_px = min_area_frac * float(work_w * work_h)
        cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_regions = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area >= min_area_px:
                candidate_regions.append((area, c))
        candidate_regions.sort(key=lambda x: x[0], reverse=True)

        all_diagnostics = []
        shape_passing_candidates = []
        margin_px = max(8.0, 0.02 * min(work_w, work_h))

        # Check OCR background match
        ocr_match = self.ocr_worker.get_latest_match()
        ocr_confirmed_quad = None
        ocr_confirmed_text = ""

        for area, c in candidate_regions:
            rect = cv2.minAreaRect(c)
            (cx_w, cy_w), (rw_w, rh_w), ang = rect
            long_side_w = max(rw_w, rh_w)
            short_side_w = max(1e-3, min(rw_w, rh_w))
            ratio = long_side_w / short_side_w
            rect_area = long_side_w * short_side_w
            fill = area / max(1e-3, rect_area)

            box_w = cv2.boxPoints(rect)
            box_w = order_corners(box_w)

            # Full-frame edge check: all 4 corners must be >= margin_px from image edges
            touching_edge = False
            for px, py in box_w:
                if px < margin_px or px > (work_w - margin_px) or py < margin_px or py > (work_h - margin_px):
                    touching_edge = True
                    break

            # Scale back up to FULL-frame pixel coordinates
            box_full = box_w / scale
            center_full = (cx_w / scale, cy_w / scale)
            long_side_full = long_side_w / scale

            # Physical length sanity check
            mpp = metres_per_pixel(alt_rel, full_w, hfov_rad)
            length_m = long_side_full * mpp

            # Long-side angle in image coordinates
            e1 = box_full[1] - box_full[0]
            e2 = box_full[3] - box_full[0]
            long_edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
            banner_angle = math.atan2(float(long_edge[1]), float(long_edge[0]))
            if banner_angle >= math.pi / 2.0:
                banner_angle -= math.pi
            elif banner_angle < -math.pi / 2.0:
                banner_angle += math.pi

            diag = {
                "quad": box_full,
                "center": center_full,
                "contour": c,
                "ratio": ratio,
                "fill": fill,
                "touching_edge": touching_edge,
                "length_m": length_m,
                "angle": banner_angle,
                "verdict": "REJECTED",
                "reason": ""
            }

            # Submit crop to background OCR worker
            bx, by, bw, bh = cv2.boundingRect(np.int32(box_full))
            pad = int(0.1 * max(bw, bh))
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(full_w, bx + bw + pad), min(full_h, by + bh + pad)
            if x1 > x0 + 10 and y1 > y0 + 10:
                crop = frame[y0:y1, x0:x1]
                self.ocr_worker.submit_job(crop, {"center": center_full, "color": color})

            # Check if this region matches background OCR hit
            if ocr_match is not None and (time.time() - ocr_match["time"]) < 2.0:
                meta = ocr_match["meta"]
                if meta.get("color") == color and math.dist(meta["center"], center_full) < (80.0 / scale):
                    ocr_confirmed_quad = box_full
                    ocr_confirmed_text = ocr_match["text"]
                    diag["verdict"] = "BANNER (OCR)"
                    diag["reason"] = f"OCR verified: '{ocr_confirmed_text}'"
                    all_diagnostics.append(diag)
                    continue

            # B2 Shape Classification Rules
            # 1. Never classify region that covers most of frame or spans full dimension
            if area / float(work_w * work_h) > 0.35:
                diag["reason"] = "Covers >35% of frame"
            elif long_side_w > 0.92 * max(work_w, work_h):
                diag["reason"] = "Spans full image dimension"
            # 2. Full-view check: must not touch edge
            elif touching_edge:
                diag["reason"] = "Border cut / touches margin"
            # 3. Ratio check (default >= 8.0)
            elif ratio < self.args.min_banner_ratio:
                diag["reason"] = f"Ratio {ratio:.1f} < {self.args.min_banner_ratio:.1f}"
            # 4. Fill check (default >= 0.60)
            elif fill < self.args.min_banner_fill:
                diag["reason"] = f"Fill {fill:.2f} < {self.args.min_banner_fill:.2f}"
            # 5. Physical size check (optional, on by default unless bench-sim or disabled)
            elif self.args.check_physical_size and not self.args.bench_sim and alt_rel > 1.5 and not (1.8 <= length_m <= 5.5):
                diag["reason"] = f"Physical length {length_m:.1f}m outside [1.8m, 5.5m]"
            else:
                diag["verdict"] = "BANNER (SHAPE)"
                diag["reason"] = f"Shape PASS: ratio={ratio:.1f}, fill={fill:.2f}"
                shape_passing_candidates.append(diag)

            all_diagnostics.append(diag)

        # 4. Immediate OCR classification if text was read
        if ocr_confirmed_quad is not None:
            if self.verbose:
                print(f"[BANNER DETECTED via OCR] Color={color.upper()} | Text='{ocr_confirmed_text}'")
            return self._build_banner_output(ocr_confirmed_quad, frame.shape, source="OCR",
                                              text=ocr_confirmed_text, alt_rel=alt_rel, hfov_rad=hfov_rad), all_diagnostics, color_mask

        # 5. Selection among passing shape candidates: best ratio/fill closest to 11.5:1 and 0.85
        if not shape_passing_candidates:
            self.consecutive_hits = 0
            self.tracked_center = None
            return None, all_diagnostics, color_mask

        # Sort by distance to ideal banner shape (ratio=11.5, fill=0.85)
        shape_passing_candidates.sort(key=lambda d: abs(d["ratio"] - 11.5) + 5.0 * abs(d["fill"] - 0.85))
        best_candidate = shape_passing_candidates[0]

        # 6. Persistence check: must pass in N consecutive frames
        cand_c = best_candidate["center"]
        track_dist_thresh = 100.0  # max pixels center can move between frames
        if (self.tracked_center is not None and self.last_color == color and
                math.dist(cand_c, self.tracked_center) < track_dist_thresh):
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 1
        self.tracked_center = cand_c
        self.last_color = color

        if self.consecutive_hits >= self.args.banner_persistence:
            if self.verbose:
                print(f"[BANNER CLASSIFIED via SHAPE] Color={color.upper()} | Ratio={best_candidate['ratio']:.1f} | Fill={best_candidate['fill']:.2f} | Consecutive={self.consecutive_hits}")
            return self._build_banner_output(best_candidate["quad"], frame.shape, source="SHAPE",
                                              text=f"{color.upper()} BANNER", alt_rel=alt_rel, hfov_rad=hfov_rad), all_diagnostics, color_mask

        return None, all_diagnostics, color_mask

    def _build_banner_output(
        self, quad_full: np.ndarray, frame_shape: Tuple[int, int, int], source: str,
        text: str, alt_rel: float, hfov_rad: float
    ) -> Dict[str, Any]:
        """Calculates center, corner coordinates, and rotated long-axis reconstruction if partial."""
        full_h, full_w = frame_shape[:2]
        pts = order_corners(quad_full)

        # Identify long edge and unit vector
        e1 = pts[1] - pts[0]
        e2 = pts[3] - pts[0]
        len1 = np.hypot(e1[0], e1[1])
        len2 = np.hypot(e2[0], e2[1])
        if len1 >= len2:
            v_long = e1
            l_vis = len1
        else:
            v_long = e2
            l_vis = len2

        u_long = v_long / max(1e-4, l_vis)
        banner_angle = math.atan2(float(v_long[1]), float(v_long[0]))
        if banner_angle >= math.pi / 2.0:
            banner_angle -= math.pi
        elif banner_angle < -math.pi / 2.0:
            banner_angle += math.pi

        # Check border cut
        bx0, by0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        bx1, by1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        m = 8.0
        cut = (bx0 <= m or bx1 >= full_w - 1 - m or by0 <= m or by1 >= full_h - 1 - m)

        u = float(pts[:, 0].mean())
        v = float(pts[:, 1].mean())

        # If partial and OCR detected: reconstruct center along rotated long axis
        if cut and source == "OCR":
            mpp = metres_per_pixel(alt_rel, full_w, hfov_rad)
            l_exp = BANNER_LENGTH_M / mpp
            # Find the two ends of the rotated board
            end_a = (pts[0] + pts[3]) * 0.5
            end_b = (pts[1] + pts[2]) * 0.5
            # Margin distance for each end
            d_a = min(end_a[0], full_w - end_a[0], end_a[1], full_h - end_a[1])
            d_b = min(end_b[0], full_w - end_b[0], end_b[1], full_h - end_b[1])
            if d_a < d_b:  # End A is cut, End B is inside
                dir_vec = (end_a - end_b) / max(1e-4, np.hypot(*(end_a - end_b)))
                true_c = end_b + dir_vec * (l_exp * 0.5)
            else:          # End B is cut, End A is inside
                dir_vec = (end_b - end_a) / max(1e-4, np.hypot(*(end_b - end_a)))
                true_c = end_a + dir_vec * (l_exp * 0.5)
            u, v = float(true_c[0]), float(true_c[1])

        return {
            "u": u,
            "v": v,
            "quad": pts,
            "angle": banner_angle,
            "partial": cut,
            "source": source,
            "text": text,
            "size": (full_w, full_h)
        }


# ==============================================================================
# 5. QRDET YOLO QR ENGINE (PORTED FROM THE qr_match ROS 2 SIM NODE)
# ==============================================================================

WARP_SIZE = 300
VALID_MODULE_COUNTS = [21 + 4 * (v - 1) for v in range(1, 15)]  # QR v1..v14


def warp_qr(image_bgr: np.ndarray, det: Dict[str, Any], padding: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Perspective-correct one qrdet detection into a square grayscale image.

    Returns (gray_warp, quad); quad is None when no polygon was available and
    a padded bounding-box crop was used instead.
    """
    height, width = image_bgr.shape[:2]

    quad = None
    for key in ("quad_xy", "padded_quad_xy", "polygon_xy"):
        raw = det.get(key)
        if raw is None:
            continue
        p = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        if p.shape[0] == 4:
            quad = order_corners(p)
        elif p.shape[0] > 4:  # many-vertex polygon -> its min-area box
            quad = order_corners(cv2.boxPoints(cv2.minAreaRect(p)))
        break

    if quad is not None:
        dst = np.array([[0, 0], [WARP_SIZE - 1, 0],
                        [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
                       dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(image_bgr, matrix, (WARP_SIZE, WARP_SIZE))
        return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), quad

    x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
    cy1, cy2 = max(0, y1 - padding), min(height, y2 + padding)
    cx1, cx2 = max(0, x1 - padding), min(width, x2 + padding)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((WARP_SIZE, WARP_SIZE), np.uint8), None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (WARP_SIZE, WARP_SIZE), interpolation=cv2.INTER_AREA), None


def normalize_qr(gray: np.ndarray) -> np.ndarray:
    """Binarise a warped QR so module colours survive uneven lighting."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


def estimate_module_count(binary: np.ndarray) -> int:
    """Estimate the module count from run lengths along a few scan lines.

    The most common same-colour run is one module wide, so the median run
    length estimates module size; snap the result to a legal QR size.
    """
    size = binary.shape[0]
    runs: List[int] = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        idx = min(int(size * frac), size - 1)
        for line in (binary[idx, :], binary[:, idx]):
            changes = np.flatnonzero(np.diff((line < 128).astype(np.uint8)))
            if changes.size >= 2:
                runs.extend(np.diff(changes).tolist())
    if not runs or np.median(runs) <= 0:
        return 25
    modules = size / float(np.median(runs))
    return min(VALID_MODULE_COUNTS, key=lambda v: abs(v - modules))


def extract_module_grid(binary: np.ndarray, grid_size: Optional[int] = None) -> np.ndarray:
    """Sample a binarised QR into an NxN uint8 grid (1 = dark module).

    Each module is read as the median of a small patch, which is far less
    sensitive to sensor noise than a single-pixel sample.
    """
    size = binary.shape[0]
    if grid_size is None:
        grid_size = estimate_module_count(binary)
    cell = size / grid_size
    half = max(1, int(cell * 0.25))
    grid = np.zeros((grid_size, grid_size), np.uint8)
    for r in range(grid_size):
        cy = min(int(r * cell + cell / 2), size - 1)
        y0, y1 = max(0, cy - half), min(size, cy + half + 1)
        for c in range(grid_size):
            cx = min(int(c * cell + cell / 2), size - 1)
            x0, x1 = max(0, cx - half), min(size, cx + half + 1)
            grid[r, c] = 1 if np.median(binary[y0:y1, x0:x1]) < 128 else 0
    return grid


def compute_match_score(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Fraction of modules that agree, maximised over the 4 rotations."""
    if grid_a.shape != grid_b.shape:
        size = max(grid_a.shape[0], grid_b.shape[0])
        grid_a = cv2.resize(grid_a, (size, size), interpolation=cv2.INTER_NEAREST)
        grid_b = cv2.resize(grid_b, (size, size), interpolation=cv2.INTER_NEAREST)
    best, cand = 0.0, grid_b.copy()
    for _ in range(4):
        best = max(best, float(np.sum(grid_a == cand) / grid_a.size))
        cand = np.rot90(cand)
    return best


class QRMatcher:
    """qrdet YOLO detector + module-grid matcher (same pipeline as the sim's qr_match node).

    Weights load in the constructor so the model is ready (and a missing
    install fails) before takeoff, not on the first frame in flight.
    """
    def __init__(self, model_size: str = "s", conf_th: float = 0.5, trust_text: bool = True):
        try:
            from qrdet import QRDetector
        except ImportError as e:
            raise RuntimeError("qrdet is required for QR detection (pip3 install qrdet)") from e
        try:
            self.detector = QRDetector(model_size=model_size, conf_th=conf_th)
        except TypeError:  # older qrdet has no conf_th
            self.detector = QRDetector(model_size=model_size)
        self.trust_text = trust_text
        self.cv_decoder = cv2.QRCodeDetector()
        try:
            from pyzbar import pyzbar
            self.pyzbar = pyzbar
        except ImportError:
            self.pyzbar = None

    def decode(self, crop_bgr: np.ndarray) -> str:
        """Decode a QR crop to text, or '' on failure. A failure is never fatal."""
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        try:
            data, _, _ = self.cv_decoder.detectAndDecode(crop_bgr)
            if data:
                return data
            if self.pyzbar is not None:
                found = self.pyzbar.decode(crop_bgr)
                if found:
                    return found[0].data.decode("utf-8", errors="ignore")
        except Exception:
            pass
        return ""

    def detect(self, frame_bgr: np.ndarray, grid_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find every QR in the frame. Each result: quad (4,2), bbox, conf, grid, warped, text.

        Pass grid_size=<reference grid size> so candidate grids are directly comparable.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_bgr.shape[:2]
        out = []
        for det in self.detector.detect(image=rgb, is_bgr=False):
            if "bbox_xyxy" not in det:
                continue
            warped, quad = warp_qr(frame_bgr, det)
            grid = extract_module_grid(normalize_qr(warped), grid_size)
            x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if quad is None:  # bbox-only fallback: use the box corners for centring/HUD
                quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            crop = frame_bgr[max(0, y1 - 10):min(height, y2 + 10),
                             max(0, x1 - 10):min(width, x2 + 10)]
            out.append({
                "quad": quad,
                "bbox": (x1, y1, x2, y2),
                "conf": float(det.get("confidence", 0.0)),
                "grid": grid,
                "warped": warped,
                "text": self.decode(crop),
            })
        return out

    def score(self, ref_grid: np.ndarray, ref_text: str, det: Dict[str, Any]) -> float:
        """Match score in [0, 1]. If both QRs decoded, an exact payload test beats the fuzzy grid."""
        if self.trust_text and ref_text and det["text"]:
            return 1.0 if det["text"] == ref_text else 0.0
        return compute_match_score(ref_grid, det["grid"])

    def best_match(
        self, frame_bgr: np.ndarray, ref_grid: np.ndarray, ref_text: str
    ) -> Tuple[float, Optional[Dict[str, Any]], int]:
        """Score every QR in the frame against the reference. Returns (best_score, best_det, n_detected)."""
        dets = self.detect(frame_bgr, grid_size=int(ref_grid.shape[0]))
        if not dets:
            return 0.0, None, 0
        scored = [(self.score(ref_grid, ref_text, d), d) for d in dets]
        best_score, best_det = max(scored, key=lambda s: s[0])
        return best_score, best_det, len(dets)


# ==============================================================================
# 6. ARDUPILOT MAVLINK DRONE CONTROLLER (STRICT GUIDED_NOGPS ON BENCH)
# ==============================================================================

class DummyMav:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class DummyMAVLinkConnection:
    def __init__(self):
        self.mav = DummyMav()
        self.flightmode = "GUIDED_NOGPS"
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
        self.flightmode = "GUIDED_NOGPS" if self.use_nogps else "UNKNOWN"
        self._last_hb = time.time()

        if self.spin_motors:
            print(f"[MAVLINK] HARDWARE ROTOR ACTIVATION: Real motors enabled at safe bench thrust ({self.thrust*100:.0f}% PWM).")
            self.m.mav.request_data_stream_send(
                self.sys, self.comp,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 20, 1
            )
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

    def setpoint(self, north: float, east: float, down: float, yaw: float):
        """
        Sends Local NED position target with heading lock.
        CRITICAL: ALWAYS sends yaw with 0b100111111000 so ArduPilot holds heading.
        """
        type_mask = 0b100111111000  # use position + yaw; ignore vel, accel, yaw-rate
        if not self.spin_motors:
            self.m.mav.set_position_target_local_ned_send(
                0, self.sys, self.comp,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                north, east, down,
                0, 0, 0,
                0, 0, 0,
                yaw, 0
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

            self.yaw = yaw
            self.ned = list(self.sim_ned)

    def land(self):
        print("[DRONE] Commanding LAND...")
        if self.spin_motors or self.use_nogps:
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
# 7. MASTER FLIGHT MISSION CONTROLLER
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
        print("[VISION] Initializing Banner Detector & QR Engines...")
        self.banner_detector = BannerDetector(args, verbose=args.verbose)
        self.qr_matcher = QRMatcher(model_size=args.qr_model_size, conf_th=args.qr_conf)

        # Mission State Variables
        self.ref_grid = None
        self.ref_warped = None
        self.ref_text = ""
        self.skip_field_qr = False

        self.state_text = "INITIALIZING"
        self.takeoff_ned = [0.0, 0.0, 0.0]
        self.takeoff_yaw = 0.0
        self.current_yaw_lock = 0.0
        self.corridor_heading = 0.0
        self.sway_x = 0.0

        self.green_banner_ned = [0.0, 0.0, 0.0]
        self.blue_banner_ned = [0.0, 0.0, 0.0]
        self.hud_countdown = 0.0

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

    @property
    def alt_rel(self) -> float:
        """Altitude relative to the takeoff pad in meters."""
        return -(self.drone.ned[2] - self.takeoff_ned[2])

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
        alt_str = f"Alt: {self.alt_rel:.2f}m | NED: ({self.drone.ned[0]:+.1f}, {self.drone.ned[1]:+.1f}) | {self.drone.flightmode}"
        cv2.putText(vis, alt_str, (max(300, w - 460), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # Reference QR thumbnail cache (saved during flight)
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
    def find_target_qr(self, frame: np.ndarray) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Best reference-matching QR in the frame: (score, detection or None)."""
        score, det, _ = self.qr_matcher.best_match(frame, self.ref_grid, self.ref_text)
        return score, det

    # ------------------------------------------------------------------
    # Slewed Setpoint Flight Helper (Heading Locked)
    # ------------------------------------------------------------------
    def fly_to(
        self, target_ned: List[float], timeout: float = 25.0,
        tol: float = 0.35, speed: Optional[float] = None, yaw: Optional[float] = None
    ) -> bool:
        sp_speed = speed or self.args.speed
        yaw_cmd = self.current_yaw_lock if yaw is None else yaw
        cur_sp = list(self.drone.ned)
        dt = 0.05
        t0 = time.time()
        last_log = 0.0
        while time.time() - t0 < timeout:
            self.drone.poll()
            self.update_hud()
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
                rps1 = max(0, int((self.drone.servos[0] - 1000) * 0.12))
                rps2 = max(0, int((self.drone.servos[1] - 1000) * 0.12))
                rps3 = max(0, int((self.drone.servos[2] - 1000) * 0.12))
                rps4 = max(0, int((self.drone.servos[3] - 1000) * 0.12))
                print(f"  [MOTORS FLY] ESC: M1={self.drone.servos[0]}(~{rps1}rps) M2={self.drone.servos[1]}(~{rps2}rps) M3={self.drone.servos[2]}(~{rps3}rps) M4={self.drone.servos[3]}(~{rps4}rps) | Cmd Tilt: Pitch={self.drone.cmd_pitch_deg:+.1f} Roll={self.drone.cmd_roll_deg:+.1f}")

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

            self.drone.setpoint(cur_sp[0], cur_sp[1], cur_sp[2], yaw=yaw_cmd)
            time.sleep(dt)
        return False

    # ------------------------------------------------------------------
    # Closed-Loop PBVS Centering for Banners (Re-evaluates B2 Every Step)
    # ------------------------------------------------------------------
    def center_over_banner(
        self, color: str = "green", target_ned: Optional[List[float]] = None,
        max_iters: int = 6, tol_px: float = 25.0, expected_heading: float = 0.0
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

            meas, diags, _ = self.banner_detector.detect(
                frame, color=color, alt_rel=self.alt_rel, hfov_rad=self.args.hfov_rad
            )
            if meas is None:
                print(f"[CENTERING] Iter {iteration}/{max_iters}: {color.upper()} banner not classified, holding position...")
                if self.drone.spin_motors:
                    self.drone.set_attitude(0.0, 0.0)
                self.fly_to(target, timeout=1.0, yaw=self.current_yaw_lock)
                continue

            fw, fh = meas["size"]
            cx, cy = fw / 2.0, fh / 2.0
            err_u = meas["u"] - cx
            err_v = meas["v"] - cy
            err_px = math.hypot(err_u, err_v)

            # Active motor tilt proportional to visual error
            if self.drone.spin_motors:
                tilt_r = max(-8.0, min(8.0, (err_u / (fw * 0.5)) * 7.5))
                tilt_p = max(-8.0, min(8.0, -(err_v / (fh * 0.5)) * 7.5))
                self.drone.set_attitude(tilt_r, tilt_p)
                rps1 = max(0, int((self.drone.servos[0] - 1000) * 0.12))
                rps2 = max(0, int((self.drone.servos[1] - 1000) * 0.12))
                rps3 = max(0, int((self.drone.servos[2] - 1000) * 0.12))
                rps4 = max(0, int((self.drone.servos[3] - 1000) * 0.12))
                print(f"  [ALIGN TILT] {color.upper()} Banner Offset: ({err_u:+.1f}, {err_v:+.1f})px | Cmd Tilt: Roll={tilt_r:+.1f}° Pitch={tilt_p:+.1f}° | ESC: M1={self.drone.servos[0]}(~{rps1}rps) M2={self.drone.servos[1]}(~{rps2}rps) M3={self.drone.servos[2]}(~{rps3}rps) M4={self.drone.servos[3]}(~{rps4}rps)")

            print(f"[CENTERING] Iter {iteration}/{max_iters}: centroid ({meas['u']:.1f}, {meas['v']:.1f})px | Offset: {err_px:.1f}px | {'PARTIAL' if meas['partial'] else 'FULL-VIEW'}")

            self.update_hud(target_pts=meas["quad"], target_type=f"{color.upper()}_BANNER", subtext=meas["source"])

            # Resolve 180° banner long-axis ambiguity to determine corridor heading
            # Long-side angle in world frame:
            theta_world = wrap_pi(self.drone.yaw + meas["angle"])
            psi1 = wrap_pi(theta_world + math.pi / 2.0)
            psi2 = wrap_pi(theta_world - math.pi / 2.0)
            diff1 = abs(wrap_pi(psi1 - expected_heading))
            diff2 = abs(wrap_pi(psi2 - expected_heading))
            cand_heading = psi1 if diff1 < diff2 else psi2
            min_diff = min(diff1, diff2)

            # Reject heading corrections > 25° from expected heading
            if min_diff <= math.radians(25.0):
                self.current_yaw_lock = cand_heading
                if color.lower() == "green":
                    self.corridor_heading = cand_heading
                print(f"  [HEADING ALIGN] Heading locked to corridor axis: {math.degrees(self.current_yaw_lock):+.1f}° (Corr: {math.degrees(min_diff):.1f}° <= 25°)")
            else:
                print(f"  [HEADING WARN] Heading diff {math.degrees(min_diff):.1f}° > 25°. Retaining expected heading {math.degrees(expected_heading):.1f}°.")

            # Lock declared ONLY when fully inside frame and centered within tolerance
            if err_px <= tol_px and not meas["partial"]:
                print(f"[CENTERING] Lock acquired! Centroid offset ({err_px:.1f}px) <= tol ({tol_px:.1f}px) & Full-View verified.")
                if self.drone.spin_motors:
                    self.drone.set_attitude(0.0, 0.0)
                locked = True
                break

            dn, de = pixel_to_ned_offset(
                meas["u"], meas["v"], fw, fh, self.alt_rel, self.drone.yaw, self.args.hfov_rad
            )
            target[0] += dn * 0.70
            target[1] += de * 0.70

            t_settle = time.time()
            while time.time() - t_settle < 1.0:
                self.drone.poll()
                self.drone.setpoint(target[0], target[1], target[2], yaw=self.current_yaw_lock)
                self.update_hud(target_pts=meas["quad"], target_type=f"{color.upper()}_BANNER")
                time.sleep(0.05)

        return target, locked

    # ------------------------------------------------------------------
    # Master Mission Execution Loop
    # ------------------------------------------------------------------
    def run(self):
        a = self.args

        # ==============================================================
        # STAGE 1: Autonomous Takeoff & Heading Recording
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 1: TAKEOFF TO {a.alt:.1f}m AGL\n{'='*70}")
        self.state_text = f"STAGE 1: TAKEOFF ({a.alt:.1f}m)"
        self.drone.arm_and_takeoff(a.alt)

        t0 = time.time()
        while time.time() - t0 < 25.0:
            self.drone.poll()
            self.update_hud()
            print(f"[STATUS] Altitude: {-self.drone.ned[2]:.2f}m / {a.alt:.1f}m", end='\r')
            if -self.drone.ned[2] >= a.alt * 0.85:
                print(f"\n[STATUS] Target altitude reached: {-self.drone.ned[2]:.2f}m")
                break
            time.sleep(0.08)

        # Record takeoff reference frame (No hard-coded North/West)
        self.takeoff_ned = list(self.drone.ned)
        self.takeoff_yaw = self.drone.yaw
        self.current_yaw_lock = self.takeoff_yaw
        print(f"[REFERENCE FRAME] Takeoff Origin: N={self.takeoff_ned[0]:+.2f}m, E={self.takeoff_ned[1]:+.2f}m, Heading={math.degrees(self.takeoff_yaw):+.1f}°")

        # Hover stabilization over takeoff pad
        t_stab = time.time()
        while time.time() - t_stab < 1.5:
            self.drone.poll()
            self.drone.setpoint(self.takeoff_ned[0], self.takeoff_ned[1], self.takeoff_ned[2] - a.alt, yaw=self.takeoff_yaw)
            self.update_hud()
            time.sleep(0.05)

        # Body axes relative to takeoff heading
        fwd_tk, left_tk, right_tk = get_body_axes(self.takeoff_yaw)

        # ==============================================================
        # STAGE 2: Surge 1 & Scan Reference QR (Camera Only, No Disk Loading)
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 2: SURGE 1 ({a.surge1:.1f}m) & SCAN REFERENCE QR (CAMERA ONLY)\n{'='*70}")
        self.state_text = f"STAGE 2: SURGE 1 ({a.surge1:.1f}m)"
        sp_surge1 = [
            self.takeoff_ned[0] + fwd_tk[0] * a.surge1,
            self.takeoff_ned[1] + fwd_tk[1] * a.surge1,
            self.takeoff_ned[2] - a.alt
        ]
        self.fly_to(sp_surge1, timeout=12.0, yaw=self.takeoff_yaw)

        self.state_text = "STAGE 2: SCANNING REFERENCE QR"
        print(f"[DRONE] Scanning takeoff reference QR code from camera (timeout: {a.ref_qr_timeout:.1f}s)...")
        t_qr = time.time()
        while self.ref_grid is None and (time.time() - t_qr) < a.ref_qr_timeout:
            self.drone.poll()
            self.drone.setpoint(sp_surge1[0], sp_surge1[1], sp_surge1[2], yaw=self.takeoff_yaw)
            frame = self.worker.get_frame()
            if frame is not None:
                dets = self.qr_matcher.detect(frame)
                if dets:
                    best = max(dets, key=lambda d: d["conf"])
                    self.ref_grid = best["grid"]
                    self.ref_warped = best["warped"]
                    self.ref_text = best["text"]
                    # Save snapshot for logging only (NEVER read back from disk)
                    save_path = "/home/suasrasp/aerothon_ws/reference_qr_memorized.png"
                    cv2.imwrite(save_path, best["warped"])
                    print(f"\n[SUCCESS] REFERENCE QR MEMORIZED FROM CAMERA! "
                          f"{self.ref_grid.shape[0]}x{self.ref_grid.shape[0]} modules, conf={best['conf']:.2f}, Payload: '{self.ref_text}'")
                    self.update_hud(target_pts=best["quad"], target_type="QR", score_or_dist=1.0)
                    break
                self.update_hud()
            time.sleep(0.08)

        # Section A3: If reference QR not seen within timeout, skip field QR search & 15s hover entirely!
        if self.ref_grid is None:
            print("\n" + "!" * 70)
            print("[WARN] Reference QR NOT detected from camera within scan timeout!")
            print("[WARN] Field QR reconnaissance (Stage 9) and 15s hover (Stage 10) will be SKIPPED.")
            print("[WARN] Continuing directly to corridor transit & return corridor stage.")
            print("!" * 70 + "\n")
            self.skip_field_qr = True

        # ==============================================================
        # STAGE 3: Surge 2 Forward along Takeoff Heading
        # ==============================================================
        total_surge_dist = a.surge1 + a.surge2
        print(f"\n{'='*70}\nSTAGE 3: SURGE 2 ({a.surge2:.1f}m) -> Total Forward = {total_surge_dist:.1f}m\n{'='*70}")
        self.state_text = f"STAGE 3: SURGE 2 ({a.surge2:.1f}m)"
        sp_surge2 = [
            self.takeoff_ned[0] + fwd_tk[0] * total_surge_dist,
            self.takeoff_ned[1] + fwd_tk[1] * total_surge_dist,
            self.takeoff_ned[2] - a.alt
        ]
        self.fly_to(sp_surge2, timeout=12.0, yaw=self.takeoff_yaw)

        # ==============================================================
        # STAGE 4: Sway Left (Takeoff Frame) & Search Green Banner
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 4: SWAY LEFT (max {a.max_sway:.1f}m) & DETECT GREEN BANNER (#00B140)\n{'='*70}")
        self.state_text = "STAGE 4: SWAY LEFT (SEARCH GREEN BANNER)"
        found_banner = None
        cur_sway_dist = 0.0
        dt = 0.05

        while cur_sway_dist <= a.max_sway:
            self.drone.poll()
            frame = self.worker.get_frame()
            if frame is not None:
                meas, diags, _ = self.banner_detector.detect(
                    frame, color="green", alt_rel=self.alt_rel, hfov_rad=a.hfov_rad
                )
                if meas is not None:
                    found_banner = meas
                    self.update_hud(target_pts=meas["quad"], target_type="GREEN_BANNER", subtext=meas["source"])
                    print(f"\n[BANNER CLASSIFIED!] Type: GREEN | Source: {meas['source']} | Current Sway: {cur_sway_dist:.2f}m")
                    break
                # If a region touches the edge, optionally slow down sway
                for d in diags:
                    if d.get("touching_edge"):
                        pass
                self.update_hud()

            cur_sway_dist += a.sway_speed * dt
            sp_sway = [
                sp_surge2[0] + left_tk[0] * cur_sway_dist,
                sp_surge2[1] + left_tk[1] * cur_sway_dist,
                self.takeoff_ned[2] - a.alt
            ]
            self.drone.setpoint(sp_sway[0], sp_sway[1], sp_sway[2], yaw=self.takeoff_yaw)
            if self.drone.spin_motors:
                self.drone.set_attitude(-4.5, 0.0)  # Tilt left
            time.sleep(dt)

        # Short hover at max sway
        if found_banner is None:
            print("[SEARCH] Hovering at max sway to complete persistence check...")
            t_hover = time.time()
            while time.time() - t_hover < 3.0:
                self.drone.poll()
                frame = self.worker.get_frame()
                if frame is not None:
                    meas, _, _ = self.banner_detector.detect(
                        frame, color="green", alt_rel=self.alt_rel, hfov_rad=a.hfov_rad
                    )
                    if meas is not None:
                        found_banner = meas
                        break
                time.sleep(0.08)

        # Section B5: Green banner NOT classified -> Fly back to takeoff point and land
        if found_banner is None:
            print("\n" + "!" * 70)
            print("[MISSION ABORT] Green Banner was NOT classified within max sway distance.")
            print("[MISSION ABORT] Returning safely to takeoff pad to land (NOT landing in field).")
            print("!" * 70 + "\n")
            self.state_text = "GREEN BANNER NOT FOUND - RETURNING HOME"
            if self.drone.spin_motors:
                self.drone.set_attitude(0.0, 0.0)
            sp_home = [self.takeoff_ned[0], self.takeoff_ned[1], self.takeoff_ned[2] - a.alt]
            self.fly_to(sp_home, timeout=25.0, yaw=self.takeoff_yaw)
            self.drone.land()
            return

        # Halting lateral momentum
        print("[BRAKE] Green Banner acquired! Halting lateral momentum...")
        if self.drone.spin_motors:
            self.drone.set_attitude(0.0, 0.0)
        else:
            self.drone.set_mode("GUIDED_NOGPS" if self.drone.use_nogps else "BRAKE")
        time.sleep(1.2)

        # ==============================================================
        # STAGE 5: Green Banner Visual Servoing & Measure Sway x
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 5: GREEN BANNER ALIGNMENT & MEASURE SWAY x\n{'='*70}")
        self.state_text = "STAGE 5: GREEN BANNER ALIGNMENT"
        self.drone.set_mode("GUIDED_NOGPS" if self.drone.use_nogps else "GUIDED")
        time.sleep(0.2)

        target_banner_ned = list(self.drone.ned)
        dn, de = pixel_to_ned_offset(
            found_banner["u"], found_banner["v"],
            found_banner["size"][0], found_banner["size"][1],
            self.alt_rel, self.drone.yaw, a.hfov_rad
        )
        target_banner_ned[0] += dn
        target_banner_ned[1] += de
        self.fly_to(target_banner_ned, timeout=8.0, yaw=self.takeoff_yaw)

        # Closed-loop centering re-evaluating B2 shape classification
        target_banner_ned, _ = self.center_over_banner(
            color="green", target_ned=target_banner_ned, max_iters=a.center_iters,
            tol_px=a.center_tol, expected_heading=self.takeoff_yaw
        )
        self.green_banner_ned = list(target_banner_ned)

        # Section B4: Measure sway distance x along takeoff frame's left axis
        d_n = self.green_banner_ned[0] - self.takeoff_ned[0]
        d_e = self.green_banner_ned[1] - self.takeoff_ned[1]
        self.sway_x = d_n * left_tk[0] + d_e * left_tk[1]

        print(f"\n{'*'*70}")
        print(f"[MEASUREMENT] Green Banner Position: North={self.green_banner_ned[0]:+.2f}m, East={self.green_banner_ned[1]:+.2f}m")
        print(f"[MEASUREMENT] Measured corridor heading = {math.degrees(self.corridor_heading):+.1f}°")
        print(f"[MEASUREMENT] Total leftward sway distance x = {self.sway_x:.2f} meters")
        print(f"{'*'*70}\n")

        # Direction vectors along measured corridor heading
        fwd_corr, left_corr, right_corr = get_body_axes(self.corridor_heading)

        # ==============================================================
        # STAGE 6: Descend to Corridor Altitude (Relative to Takeoff Point)
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 6: DESCEND TO CORRIDOR ALTITUDE ({a.corridor_alt:.1f}m AGL)\n{'='*70}")
        self.state_text = f"STAGE 6: DESCENDING TO {a.corridor_alt:.1f}m"
        target_banner_ned[2] = self.takeoff_ned[2] - a.corridor_alt
        self.fly_to(target_banner_ned, timeout=12.0, yaw=self.corridor_heading)
        time.sleep(1.0)

        # ==============================================================
        # STAGE 7: Surge Forward Through Green Corridor
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 7: SURGE {a.corridor_surge:.1f}m THROUGH CORRIDOR (Heading: {math.degrees(self.corridor_heading):+.1f}°)\n{'='*70}")
        self.state_text = f"STAGE 7: SURGE CORRIDOR ({a.corridor_surge:.1f}m)"
        sp_corridor_end = [
            target_banner_ned[0] + fwd_corr[0] * a.corridor_surge,
            target_banner_ned[1] + fwd_corr[1] * a.corridor_surge,
            self.takeoff_ned[2] - a.corridor_alt
        ]
        self.fly_to(sp_corridor_end, timeout=a.corridor_surge / 0.8 + 15.0, speed=0.9, yaw=self.corridor_heading)
        print(f"[CORRIDOR EXIT] Exited corridor at N={self.drone.ned[0]:+.2f}m, E={self.drone.ned[1]:+.2f}m")

        # ==============================================================
        # STAGE 8: Lateral Positioning: Go Left by (19 - x) meters along corridor axis
        # ==============================================================
        wall_sway_step = a.wall_offset - self.sway_x
        print(f"\n{'='*70}\nSTAGE 8: LATERAL SHIFT LEFT BY ({a.wall_offset:.1f} - x) = {wall_sway_step:.2f}m (1m from 20m wall)\n{'='*70}")
        self.state_text = f"STAGE 8: MOVE LEFT {wall_sway_step:.2f}m"

        sp_wall = [
            self.drone.ned[0] + left_corr[0] * wall_sway_step + fwd_corr[0] * 1.0,
            self.drone.ned[1] + left_corr[1] * wall_sway_step + fwd_corr[1] * 1.0,
            self.takeoff_ned[2] - a.alt  # Climb back to transit altitude
        ]
        self.fly_to(sp_wall, timeout=20.0, yaw=self.corridor_heading)
        print(f"[SURVEY ZONE ENTRY] Positioned at N={self.drone.ned[0]:+.2f}m, E={self.drone.ned[1]:+.2f}m")

        # ==============================================================
        # STAGES 9 & 10: QR Lawnmower Reconnaissance & 15s Hover
        # ==============================================================
        target_qr_ned = None

        if self.skip_field_qr or self.ref_grid is None:
            print("\n" + "=" * 70)
            print("[STAGE 9 & 10 SKIPPED] Reference QR was not acquired. Proceeding straight to Return Corridor.")
            print("=" * 70 + "\n")
        else:
            # Build lawnmower grid aligned with corridor heading and lateral stepping
            print(f"\n{'='*70}\nSTAGE 9: LAWNMOWER QR RECONNAISSANCE ALONG CORRIDOR AXIS\n{'='*70}")
            self.state_text = "STAGE 9: LAWNMOWER QR SEARCH"

            # Create waypoints spanning lawnmower_length forward and stepping rightwards
            n_rows = 5
            row_step = a.step
            lawnmower_wps = []
            heading_fwd = True
            for r in range(n_rows):
                lat_offset = r * row_step
                p0 = [
                    sp_wall[0] + right_corr[0] * lat_offset,
                    sp_wall[1] + right_corr[1] * lat_offset,
                    self.takeoff_ned[2] - a.alt
                ]
                p1 = [
                    p0[0] + fwd_corr[0] * a.lawnmower_length,
                    p0[1] + fwd_corr[1] * a.lawnmower_length,
                    self.takeoff_ned[2] - a.alt
                ]
                if heading_fwd:
                    lawnmower_wps.append(p0)
                    lawnmower_wps.append(p1)
                else:
                    lawnmower_wps.append(p1)
                    lawnmower_wps.append(p0)
                heading_fwd = not heading_fwd

            for wp_idx, tgt_ned in enumerate(lawnmower_wps):
                print(f"[WAYPOINT {wp_idx+1}/{len(lawnmower_wps)}] Target: N={tgt_ned[0]:+.1f}m, E={tgt_ned[1]:+.1f}m")
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
                        sim_score, best_qr, _ = self.qr_matcher.best_match(frame, self.ref_grid, self.ref_text)
                        if best_qr is not None:
                            detected_pts = best_qr["quad"]
                            cand_txt = best_qr["text"]
                            is_match = sim_score >= a.match_thresh

                            if is_match:
                                p = detected_pts.reshape(-1, 2)
                                q_u, q_v = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
                                fw, fh = frame.shape[1], frame.shape[0]
                                dn, de = pixel_to_ned_offset(
                                    q_u, q_v, fw, fh, self.alt_rel, self.drone.yaw, a.hfov_rad
                                )
                                target_qr_ned = [self.drone.ned[0] + dn, self.drone.ned[1] + de, self.takeoff_ned[2] - a.alt]

                                print(f"\n{'!'*70}")
                                print(f"[MATCH FOUND!] Candidate Similarity: {sim_score*100:.1f}% | Payload: '{cand_txt}'")
                                print(f"[TARGET QR ESTIMATE] N={target_qr_ned[0]:+.2f}m, E={target_qr_ned[1]:+.2f}m")
                                print(f"{'!'*70}\n")

                                print("[BRAKE] Target QR matched! Freezing momentum...")
                                if self.drone.spin_motors:
                                    self.drone.set_attitude(0.0, 0.0)
                                else:
                                    self.drone.set_mode("GUIDED_NOGPS" if self.drone.use_nogps else "BRAKE")
                                self.update_hud(target_pts=detected_pts, target_type="QR", score_or_dist=sim_score)
                                break

                    self.update_hud(target_pts=detected_pts, target_type="QR", score_or_dist=sim_score)
                    if target_qr_ned is not None:
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
                    cur_sp[2] = self.takeoff_ned[2] - a.alt
                    self.drone.setpoint(cur_sp[0], cur_sp[1], cur_sp[2], yaw=self.corridor_heading)
                    time.sleep(dt)

                if target_qr_ned is not None:
                    break

            # STAGE 10: Precision QR Centering & 15-Second Hover Hold
            if target_qr_ned is not None:
                print(f"\n{'='*70}\nSTAGE 10: PRECISION CENTERING & 15-SECOND HOVER HOLD\n{'='*70}")
                self.state_text = "STAGE 10: QR CENTERING & HOVER"
                self.drone.set_mode("GUIDED_NOGPS" if self.drone.use_nogps else "GUIDED")
                time.sleep(0.2)
                self.fly_to(target_qr_ned, timeout=10.0, tol=0.25, yaw=self.corridor_heading)

                # Fine Closed-Loop Centering over candidate QR
                for iteration in range(a.center_iters):
                    self.drone.poll()
                    frame = self.worker.get_frame()
                    if frame is None:
                        continue
                    _, target_det = self.find_target_qr(frame)
                    if target_det is not None:
                        pts = target_det["quad"]
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
                        dn, de = pixel_to_ned_offset(u, v, fw, fh, self.alt_rel, self.drone.yaw, a.hfov_rad)
                        target_qr_ned[0] += dn * 0.70
                        target_qr_ned[1] += de * 0.70
                        self.fly_to(target_qr_ned, timeout=1.5, yaw=self.corridor_heading)

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
                        _, target_det = self.find_target_qr(frame)
                        if target_det is not None:
                            pts = target_det["quad"]
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

                    self.drone.setpoint(target_qr_ned[0], target_qr_ned[1], target_qr_ned[2], yaw=self.corridor_heading)
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
        # STAGE 11: Return Corridor (Blue Banner Prediction, Search & Entry)
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 11: RETURN CORRIDOR (BLUE BANNER) DETECTION & TRANSIT\n{'='*70}")
        self.state_text = "STAGE 11: TRANSITION TO RETURN CORRIDOR"

        # Predict Blue Banner return corridor entrance:
        # Based on measured Green Banner position, corridor heading, spacing and forward offset
        side_mult = 1.0 if a.return_corridor_side.lower() == "right" else -1.0
        return_heading = wrap_pi(self.corridor_heading + math.pi)  # 180° opposite (facing South)

        n_blue_pred = (self.green_banner_ned[0] + fwd_corr[0] * a.return_corridor_forward +
                       right_corr[0] * (side_mult * a.return_corridor_spacing))
        e_blue_pred = (self.green_banner_ned[1] + fwd_corr[1] * a.return_corridor_forward +
                       right_corr[1] * (side_mult * a.return_corridor_spacing))

        print(f"[RETURN PREDICTION] Green Banner Position: N={self.green_banner_ned[0]:+.2f}m, E={self.green_banner_ned[1]:+.2f}m")
        print(f"[RETURN PREDICTION] Corridor Spacing: {a.return_corridor_spacing:.1f}m ({a.return_corridor_side}) | Forward Offset: {a.return_corridor_forward:.1f}m")
        print(f"[RETURN PREDICTION] Predicted Blue Entrance: N={n_blue_pred:+.2f}m, E={e_blue_pred:+.2f}m | Return Heading={math.degrees(return_heading):+.1f}°")

        # 1. Fly to predicted entrance at transit altitude
        sp_blue_pred = [n_blue_pred, e_blue_pred, self.takeoff_ned[2] - a.alt]
        self.fly_to(sp_blue_pred, timeout=20.0, yaw=self.corridor_heading)

        # 2. Turn to return heading (corridor_heading + 180°)
        print("[RETURN CORRIDOR] Turning to return heading (180° opposite)...")
        self.state_text = "STAGE 11: TURNING TO RETURN HEADING"
        self.current_yaw_lock = return_heading
        t_turn = time.time()
        while time.time() - t_turn < 3.0:
            self.drone.poll()
            self.drone.setpoint(sp_blue_pred[0], sp_blue_pred[1], sp_blue_pred[2], yaw=return_heading)
            self.update_hud()
            time.sleep(0.05)

        # 3. Active Blue Banner search: Check prediction, then local search pattern
        self.state_text = "STAGE 11: SEARCHING BLUE BANNER"
        print("[VISION] Searching for BLUE BANNER (#0000FF) at predicted entrance...")
        found_blue = None
        t_check_pred = time.time()
        while time.time() - t_check_pred < 3.0:
            self.drone.poll()
            frame = self.worker.get_frame()
            if frame is not None:
                meas, _, _ = self.banner_detector.detect(
                    frame, color="blue", alt_rel=self.alt_rel, hfov_rad=a.hfov_rad
                )
                if meas is not None:
                    found_blue = meas
                    break
            time.sleep(0.08)

        # Local search pattern (±3m lateral, ±1.5m longitudinal) if not classified at prediction
        if found_blue is None:
            print("[RETURN CORRIDOR] Blue banner not classified at initial point. Flying search pattern (±3m lat, ±1.5m fwd)...")
            search_offsets = [
                (0.0, -2.5),
                (0.0, 2.5),
                (1.5, 0.0),
                (-1.5, 0.0),
                (1.5, 2.0),
                (-1.5, -2.0)
            ]
            for fwd_off, lat_off in search_offsets:
                wp_search = [
                    n_blue_pred + fwd_corr[0] * fwd_off + right_corr[0] * lat_off,
                    e_blue_pred + fwd_corr[1] * fwd_off + right_corr[1] * lat_off,
                    self.takeoff_ned[2] - a.alt
                ]
                self.fly_to(wp_search, timeout=8.0, yaw=return_heading)
                t_dwell = time.time()
                while time.time() - t_dwell < 2.0:
                    self.drone.poll()
                    frame = self.worker.get_frame()
                    if frame is not None:
                        meas, _, _ = self.banner_detector.detect(
                            frame, color="blue", alt_rel=self.alt_rel, hfov_rad=a.hfov_rad
                        )
                        if meas is not None:
                            found_banner = meas
                            found_blue = meas
                            print(f"[BLUE BANNER DETECTED!] In search pattern at offset fwd={fwd_off:+.1f}m, lat={lat_off:+.1f}m")
                            break
                    time.sleep(0.08)
                if found_blue is not None:
                    break

        # Section B5: If Blue Banner is STILL not found -> DO NOT descend and DO NOT surge blind! Fly home above walls!
        if found_blue is None:
            print("\n" + "!" * 70)
            print("[WARN] Blue Banner was NOT classified after local search pattern.")
            print("[WARN] ABORTING corridor entry: will NOT descend and will NOT surge corridor blind!")
            print("[WARN] Flying home at transit altitude (above walls) and landing at takeoff pad.")
            print("!" * 70 + "\n")
            self.state_text = "BLUE BANNER NOT FOUND - RETURNING HOME"
            if self.drone.spin_motors:
                self.drone.set_attitude(0.0, 0.0)
            sp_home = [self.takeoff_ned[0], self.takeoff_ned[1], self.takeoff_ned[2] - a.alt]
            self.fly_to(sp_home, timeout=25.0, yaw=self.takeoff_yaw)
            self.drone.land()
            return

        # 4. Center over Blue Banner
        print("[RETURN CORRIDOR] Centering over classified Blue Banner...")
        blue_target_ned = list(self.drone.ned)
        dn, de = pixel_to_ned_offset(
            found_blue["u"], found_blue["v"],
            found_blue["size"][0], found_blue["size"][1],
            self.alt_rel, self.drone.yaw, a.hfov_rad
        )
        blue_target_ned[0] += dn
        blue_target_ned[1] += de
        self.fly_to(blue_target_ned, timeout=6.0, yaw=return_heading)

        blue_target_ned, _ = self.center_over_banner(
            color="blue", target_ned=blue_target_ned, max_iters=a.center_iters,
            tol_px=a.center_tol, expected_heading=return_heading
        )
        self.blue_banner_ned = list(blue_target_ned)

        # 5. Descend to corridor altitude (3.0m relative to takeoff point)
        print(f"[RETURN CORRIDOR] Descending to corridor altitude {a.corridor_alt:.1f}m AGL...")
        blue_target_ned[2] = self.takeoff_ned[2] - a.corridor_alt
        self.fly_to(blue_target_ned, timeout=10.0, yaw=self.current_yaw_lock)

        # 6. Surge through return corridor along return heading
        fwd_return, _, _ = get_body_axes(self.current_yaw_lock)
        print(f"[RETURN CORRIDOR] Surging {a.corridor_surge:.1f}m through return corridor (Heading: {math.degrees(self.current_yaw_lock):+.1f}°)...")
        self.state_text = f"STAGE 11: SURGE RETURN CORRIDOR ({a.corridor_surge:.1f}m)"
        sp_return_exit = [
            blue_target_ned[0] + fwd_return[0] * a.corridor_surge,
            blue_target_ned[1] + fwd_return[1] * a.corridor_surge,
            self.takeoff_ned[2] - a.corridor_alt
        ]
        self.fly_to(sp_return_exit, timeout=a.corridor_surge / 0.8 + 15.0, speed=0.9, yaw=self.current_yaw_lock)
        print(f"[RETURN CORRIDOR] Successfully emerged from return corridor at N={self.drone.ned[0]:+.2f}m, E={self.drone.ned[1]:+.2f}m")

        # ==============================================================
        # STAGE 12: Return to Takeoff Pad (0, 0) & Precision Land
        # ==============================================================
        print(f"\n{'='*70}\nSTAGE 12: RETURN TO TAKEOFF POINT & PRECISION LAND\n{'='*70}")
        self.state_text = "STAGE 12: RETURNING TO TAKEOFF PAD"

        # Climb to safe transit altitude and rotate to takeoff heading
        self.current_yaw_lock = self.takeoff_yaw
        sp_home_transit = [self.takeoff_ned[0], self.takeoff_ned[1], self.takeoff_ned[2] - a.alt]
        self.fly_to(sp_home_transit, timeout=20.0, yaw=self.takeoff_yaw)

        # Descend to low hover over takeoff pad
        sp_home_low = [self.takeoff_ned[0], self.takeoff_ned[1], self.takeoff_ned[2] - 2.0]
        self.fly_to(sp_home_low, timeout=8.0, yaw=self.takeoff_yaw)

        print("\n" + "#" * 70)
        print(" MISSION OBJECTIVES FULLY ACHIEVED! COMMANDING PRECISION LAND...")
        print("#" * 70 + "\n")
        self.state_text = "STAGE 12: LANDING"
        self.drone.land()

        t_land = time.time()
        while time.time() - t_land < 8.0:
            self.drone.poll()
            self.update_hud()
            time.sleep(0.1)

        print(f"\n{'='*70}\nMISSION COMPLETED SUCCESSFULLY!\n{'='*70}")

    def shutdown(self):
        self.banner_detector.ocr_worker.stop()
        self.worker.stop()
        if self.show_gui:
            cv2.destroyAllWindows()


# ==============================================================================
# 8. STANDALONE VISION TEST TOOL (NO MAVLINK, NO FLIGHT)
# ==============================================================================

def run_vision_test(args):
    """
    Interactive standalone tool to tune HSV thresholds and banner classification on the field.
    No MAVLink connection or flight commands.
    Shows rotated rectangles, 4 corners, center, ratio, fill, full-view status, and verdicts.
    Press 'm' to toggle color mask overlay. Press 'q' or ESC to exit.
    """
    target_color = args.vision_test.lower()
    print("\n" + "=" * 70)
    print(f" AEROTHON 2026 — STANDALONE VISION TUNING TOOL ({target_color.upper()} BANNER)")
    print("=" * 70)
    print(" Controls:")
    print("   'm' : Toggle Color Mask / Raw Camera Feed overlay")
    print("   'q' : Quit vision test tool")
    print("=" * 70 + "\n")

    cam_src = args.camera_src
    if not cam_src:
        cam_src = f"rtsp://192.168.144.25:8554/{args.stream}.264"
    worker = SiyiStreamWorker(capture_source=cam_src)
    detector = BannerDetector(args, verbose=True)

    show_mask = False
    window_name = f"AeroTHON Vision Test — {target_color.upper()} Banner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            frame = worker.get_frame()
            if frame is None:
                vis = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(vis, f"CONNECTING TO {cam_src}...", (160, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, vis)
                if cv2.waitKey(50) & 0xFF in (ord('q'), 27):
                    break
                continue

            banner, diags, mask = detector.detect(frame, color=target_color, alt_rel=args.alt, hfov_rad=args.hfov_rad)

            if show_mask:
                mask_bgr = cv2.cvtColor(cv2.resize(mask, (frame.shape[1], frame.shape[0])), cv2.COLOR_GRAY2BGR)
                vis = cv2.addWeighted(frame, 0.4, mask_bgr, 0.6, 0)
            else:
                vis = frame.copy()

            h, w = vis.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

            # Draw every detected region with rotated rectangle diagnostics
            for diag in diags:
                quad = np.int32(diag["quad"]).reshape(-1, 2)
                is_banner = "BANNER" in diag["verdict"]
                col = (0, 255, 0) if is_banner else (0, 140, 255)

                cv2.polylines(vis, [quad], True, col, 2 if not is_banner else 3)
                tcx, tcy = int(diag["center"][0]), int(diag["center"][1])
                cv2.circle(vis, (tcx, tcy), 5, (0, 0, 255), -1)

                # Diagnostic label
                edge_str = "CUT" if diag["touching_edge"] else "FULL"
                status_txt = f"R:{diag['ratio']:.1f} F:{diag['fill']:.2f} [{edge_str}] -> {diag['verdict']}"
                cv2.putText(vis, status_txt, (tcx - 60, max(20, tcy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            # Draw top info banner
            cv2.rectangle(vis, (0, 0), (w, 42), (20, 20, 20), -1)
            h_tol = args.green_h_tol if target_color == "green" else args.blue_h_tol
            s_min = args.green_s_min if target_color == "green" else args.blue_s_min
            v_min = args.green_v_min if target_color == "green" else args.blue_v_min
            info_txt = f"COLOR: {target_color.upper()} | H_tol: +/-{h_tol} S_min: {s_min} V_min: {v_min} | Mask: {'ON' if show_mask else 'OFF'} ('m' to toggle)"
            cv2.putText(vis, info_txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if banner is not None:
                b_pts = np.int32(banner["quad"]).reshape(-1, 2)
                cv2.polylines(vis, [b_pts], True, (0, 255, 0), 4)
                cv2.putText(vis, f"*** CLASSIFIED {target_color.upper()} BANNER ({banner['source']}) ***",
                            (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                show_mask = not show_mask
            elif key in (ord('q'), 27):
                break
    finally:
        detector.ocr_worker.stop()
        worker.stop()
        cv2.destroyAllWindows()


# ==============================================================================
# 9. CLI ARGUMENT PARSING & ENTRYPOINT
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

    # Reference QR Parameters
    p.add_argument("--ref-qr-timeout", type=float, default=12.0, help="Camera scan timeout for initial reference QR (seconds)")

    # Color Segmentation Parameters (B1)
    p.add_argument("--green-h-tol", type=int, default=15, help="Green hue tolerance (+/- deg) around #00B140")
    p.add_argument("--green-s-min", type=int, default=100, help="Green minimum saturation [0-255]")
    p.add_argument("--green-v-min", type=int, default=70, help="Green minimum brightness/value [0-255]")
    p.add_argument("--green-min-area", type=float, default=0.0004, help="Green minimum area fraction of frame")

    p.add_argument("--blue-h-tol", type=int, default=15, help="Blue hue tolerance (+/- deg) around #0000FF")
    p.add_argument("--blue-s-min", type=int, default=100, help="Blue minimum saturation [0-255]")
    p.add_argument("--blue-v-min", type=int, default=70, help="Blue minimum brightness/value [0-255]")
    p.add_argument("--blue-min-area", type=float, default=0.0004, help="Blue minimum area fraction of frame")

    # Shape Classification Parameters (B2)
    p.add_argument("--min-banner-ratio", type=float, default=8.0, help="Minimum long-to-short aspect ratio for banner")
    p.add_argument("--min-banner-fill", type=float, default=0.60, help="Minimum contour area to rotated rect area ratio")
    p.add_argument("--banner-persistence", type=int, default=3, help="Required consecutive frames for shape detection")
    p.add_argument("--check-physical-size", dest="check_physical_size", action="store_true", default=True, help="Check 2-5m physical length sanity")
    p.add_argument("--no-check-physical-size", dest="check_physical_size", action="store_false", help="Disable physical size sanity check")

    # Flight Stage Distances & Geometric Parameters (User-Configurable)
    p.add_argument("--alt", type=float, default=5.0, help="Takeoff and transit altitude in meters AGL")
    p.add_argument("--surge1", type=float, default=2.0, help="Surge 1 forward distance to scan reference QR (meters)")
    p.add_argument("--surge2", type=float, default=2.0, help="Surge 2 forward distance after scanning reference QR (meters)")
    p.add_argument("--max-sway", type=float, default=20.0, help="Maximum left sway distance to search for green banner (meters)")
    p.add_argument("--sway-speed", type=float, default=0.5, help="Sideways search speed (m/s)")
    p.add_argument("--corridor-alt", type=float, default=3.0, help="Altitude inside corridors in meters AGL")
    p.add_argument("--corridor-surge", type=float, default=10.0, help="Forward surge distance through corridor (meters)")
    p.add_argument("--wall-offset", type=float, default=19.0, help="Reference wall distance: lateral shift is (wall_offset - x) to land 1m from 20m wall")

    # Lawnmower & QR Matching Parameters (Preserved)
    p.add_argument("--lawnmower-length", type=float, default=18.0, help="Forward search span for lawnmower grid (meters)")
    p.add_argument("--step", type=float, default=4.0, help="Lawnmower row spacing in meters")
    p.add_argument("--match-thresh", type=float, default=0.85, help="QR similarity matching threshold [0.0 to 1.0]")
    p.add_argument("--qr-model-size", choices=["n", "s", "m", "l"], default="s", help="qrdet YOLO model size")
    p.add_argument("--qr-conf", type=float, default=0.5, help="qrdet detector confidence floor")
    p.add_argument("--align-duration", type=float, default=15.0, help="Hover hold duration over matched QR in seconds")
    p.add_argument("--center-tol", type=float, default=25.0, help="Visual servoing centering tolerance in pixels")
    p.add_argument("--center-iters", type=int, default=6, help="Maximum visual servoing iterations")

    # Return Corridor & Blue Banner
    p.add_argument("--return-corridor-spacing", type=float, default=4.0, help="Center-to-center spacing between corridors (meters)")
    p.add_argument("--return-corridor-forward", type=float, default=7.0, help="Green banner to blue banner along corridor offset (meters)")
    p.add_argument("--return-corridor-side", choices=["right", "left"], default="right", help="Side of return corridor relative to forward flight")

    # General Speed & Benchtop Testing Options
    p.add_argument("--speed", type=float, default=1.2, help="General fly-to speed in m/s")
    p.add_argument("--guided-nogps", action="store_true", help="Force GUIDED_NOGPS flight mode for bench testing without GPS lock")
    p.add_argument("--bench-sim", action="store_true", help="Run in benchtop simulation mode (emulates kinematics while keeping real SIYI camera and real Cube MAVLink commands active)")
    p.add_argument("--spin-motors", action="store_true", help="Spin physical rotors safely on bench during simulation (Props MUST be OFF!)")
    p.add_argument("--thrust", type=float, default=0.20, help="Bench motor thrust [0.10 to 0.35]")
    p.add_argument("--show-camera", action="store_true", default=True, help="Display live visual HUD overlay")
    p.add_argument("--no-camera", dest="show_camera", action="store_false", help="Run headless without GUI window")
    p.add_argument("--verbose", action="store_true", help="Print verbose vision and navigation diagnostics")

    # Standalone Vision Tuning Tool
    p.add_argument("--vision-test", choices=["green", "blue"], default=None,
                   help="Run standalone vision test tool to tune HSV thresholds on field without MAVLink")

    args = p.parse_args()

    # Section C: Vision Test Tool Mode
    if args.vision_test is not None:
        run_vision_test(args)
        return

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
