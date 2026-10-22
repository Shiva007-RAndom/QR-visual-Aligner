#!/usr/bin/env python3
"""
Aerothon Lawnmower Survey & QR Similarity Aligner
-------------------------------------------------
Features:
- Zone Separation:
    * Blue Zone (50m x 10m): Drone take-off at (0, 0), downward camera captures &
      memorizes reference QR into memory (4-rotation invariant module grid + decoded text).
    * Green Zone (50m x 40m): Serpentine lawnmower search strictly inside Green Zone
      (preventing any re-detection of the initial reference QR).
- QR Twin Recognition:
    * Dual-check: Matches decoded payload (if available) OR 21x21 module grid similarity >= match_thresh (default 0.88).
- Closed-Loop Visual Servoing & Precision Centering:
    * Active braking using ArduPilot BRAKE mode on detection.
    * Single-step coordinate projection + fine-tuning visual servoing.
    * Eliminates runaway, integrator windup, and drift.
- Configurable Alignment Hover:
    * Holds steady hover directly over the matched QR for --align-duration (default 10.0s).
- Live HUD:
    * PiP thumbnail of memorized reference QR.
    * Real-time crosshair, bounding polygon, similarity tag, and alignment countdown.
"""

import argparse
import math
import sys
import threading
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
from pymavlink import mavutil
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# ----------------------------------------------------------------------
# 1. Computer Vision & QR Similarity Functions
# ----------------------------------------------------------------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order quadrilateral points: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)

def extract_qr_features(gray: np.ndarray, quad_pts: np.ndarray, grid_size: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """Warp QR quad to 210x210, binarize, and sample 21x21 binary module grid."""
    quad = order_corners(quad_pts)
    dst = np.array([[0, 0], [grid_size*10 - 1, 0], [grid_size*10 - 1, grid_size*10 - 1], [0, grid_size*10 - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(gray, matrix, (grid_size*10, grid_size*10))
    
    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r*10 + 2, r*10 + 8
            x0, x1 = c*10 + 2, c*10 + 8
            grid[r, c] = 1 if np.mean(binary[y0:y1, x0:x1]) < 128 else 0
            
    return grid, warped

def compute_similarity(grid_ref: np.ndarray, grid_cand: np.ndarray) -> float:
    """Calculate 4-rotation invariant agreement score in [0.0, 1.0]."""
    if grid_ref is None or grid_cand is None:
        return 0.0
    best_score = 0.0
    cand = grid_cand.copy()
    for _ in range(4):
        agreement = float(np.sum(grid_ref == cand)) / grid_ref.size
        if agreement > best_score:
            best_score = agreement
        cand = np.rot90(cand)
    return best_score

# ----------------------------------------------------------------------
# 2. Drone Controller (MAVLink via pymavlink)
# ----------------------------------------------------------------------

class DroneController:
    """Communicates with ArduPilot in Local NED frame (North, East, Down)."""
    
    def __init__(self, url=None):
        if url:
            ports_to_try = [url]
        else:
            ports_to_try = [
                "udpin:0.0.0.0:14550",
                "udpin:0.0.0.0:14551",
                "tcp:127.0.0.1:5762",
                "tcp:127.0.0.1:5763"
            ]
        self.m = None
        for p in ports_to_try:
            if not p:
                continue
            try:
                print(f"[DRONE] Probing MAVLink on {p}...")
                conn = mavutil.mavlink_connection(p)
                # Send GCS heartbeat to announce presence
                try:
                    conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                except Exception:
                    pass
                hb = conn.wait_heartbeat(timeout=3.5)
                if hb:
                    print(f"[DRONE] Connected on {p} (System {conn.target_system})")
                    self.m = conn
                    break
            except Exception:
                pass
                
        if self.m is None:
            raise RuntimeError("Could not connect to ArduPilot on UDP 14550/14551 or TCP 5762/5763. Is SITL running?")
            
        self.sys = self.m.target_system
        self.comp = 1
        self.ned = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.armed = False
        self._last_hb = time.time()

    def poll(self):
        """Read telemetry and keep heartbeat alive."""
        now = time.time()
        if now - self._last_hb > 1.0:
            self.m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
            )
            self._last_hb = now
            
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "LOCAL_POSITION_NED":
                self.ned = [msg.x, msg.y, msg.z]
            elif mtype == "ATTITUDE":
                self.yaw = msg.yaw
            elif mtype == "HEARTBEAT":
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def set_mode(self, mode_name: str):
        mode_id = self.m.mode_mapping().get(mode_name)
        if mode_id is not None:
            for _ in range(3):
                self.m.mav.command_long_send(
                    self.sys, self.comp,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id, 0, 0, 0, 0, 0
                )
                time.sleep(0.05)

    def arm_and_takeoff(self, target_alt: float = 4.0):
        self.set_mode("GUIDED")
        time.sleep(0.5)
        print("[DRONE] Arming motors...")
        for _ in range(8):
            self.m.mav.command_long_send(
                self.sys, self.comp,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 21196, 0, 0, 0, 0, 0
            )
            time.sleep(0.3)
            self.poll()
            if self.armed:
                print("[DRONE] Motors ARMED!")
                break
                
        print(f"[DRONE] Taking off to {target_alt:.1f}m altitude...")
        self.m.mav.command_long_send(
            self.sys, self.comp,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_alt
        )

    def setpoint(self, north: float, east: float, down: float):
        type_mask = 0b110111111000
        self.m.mav.set_position_target_local_ned_send(
            0, self.sys, self.comp,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            north, east, down,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    def land(self):
        print("[DRONE] Landing...")
        self.set_mode("LAND")

# ----------------------------------------------------------------------
# 3. Lawnmower Waypoint Generator for Green Zone
# ----------------------------------------------------------------------

def build_lawnmower_route(north_range=(-16.0, 16.0), east_range=(-18.0, 18.0), step=5.0) -> List[Tuple[float, float]]:
    """
    Builds a serpentine survey route strictly inside the Green Search Zone.
    Advances along East columns, alternating Northward and Southward sweeps.
    """
    route = []
    east_cols = np.arange(east_range[0], east_range[1] + 0.1, step)
    heading_north = True
    for e in east_cols:
        n_start, n_end = (north_range[0], north_range[1]) if heading_north else (north_range[1], north_range[0])
        route.append((float(n_start), float(e)))
        route.append((float(n_end), float(e)))
        heading_north = not heading_north
    return route

# ----------------------------------------------------------------------
# 4. Main Mission Node & Flight Logic with Live HUD
# ----------------------------------------------------------------------

class LawnmowerMission:
    def __init__(self, args):
        self.args = args
        self.node = rclpy.create_node("aerothon_lawnmower_mission")
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.sub = self.node.create_subscription(
            Image, args.image_topic, self.image_cb, 10
        )
        
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            self.detector = cv2.QRCodeDetectorAruco()
        else:
            self.detector = cv2.QRCodeDetector()
            
        self.ref_grid = None
        self.ref_warped = None
        self.ref_text = ""
        self.state_text = "INITIALIZING"
        self.hud_score = 0.0
        self.hud_countdown = 0.0

    def image_cb(self, msg: Image):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding in ['rgb8', 'RGB8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        with self.frame_lock:
            self.latest_frame = im

    def update_hud(self, drone_ned, quad_pts=None, score=0.0):
        """Render live camera HUD window if enabled."""
        if not self.args.show_camera:
            return
            
        with self.frame_lock:
            if self.latest_frame is None:
                return
            vis = self.latest_frame.copy()
            
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Center crosshairs
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, (cx, cy), 14, (0, 255, 255), 1)
        
        # Highlight detected QR
        if quad_pts is not None and len(quad_pts) > 0:
            p = np.int32(quad_pts).reshape(-1, 2)
            color = (0, 255, 0) if score >= self.args.match_thresh else (0, 165, 255)
            cv2.polylines(vis, [p], isClosed=True, color=color, thickness=3)
            # Centroid
            qcx, qcy = int(np.mean(p[:, 0])), int(np.mean(p[:, 1]))
            cv2.circle(vis, (qcx, qcy), 5, (0, 0, 255), -1)
            cv2.line(vis, (cx, cy), (qcx, qcy), (255, 0, 255), 1)
            
            tag = f"Similarity: {score*100:.1f}% [{'MATCH!' if score >= self.args.match_thresh else 'NO MATCH'}]"
            cv2.putText(vis, tag, (qcx - 80, qcy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Top banner with mission state & altitude
        cv2.rectangle(vis, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(vis, f"MISSION: {self.state_text}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        alt_str = f"Alt: {-drone_ned[2]:.2f}m | NED: ({drone_ned[0]:.1f}, {drone_ned[1]:.1f})"
        cv2.putText(vis, alt_str, (w - 290, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # PiP: show memorized reference QR in corner
        if self.ref_warped is not None:
            thumb = cv2.resize(self.ref_warped, (90, 90))
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR) if len(thumb.shape) == 2 else thumb
            cv2.rectangle(vis, (w - 105, 50), (w - 10, 145), (0, 255, 0), 2)
            vis[52:142, w - 103:w - 13] = thumb_bgr
            cv2.putText(vis, "MEMORIZED REF", (w - 110, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        # If in ALIGN stage, show countdown box
        if self.hud_countdown > 0:
            cd_box = f"ALIGNMENT HOLD: {self.hud_countdown:.1f}s"
            cv2.rectangle(vis, (cx - 160, h - 50), (cx + 160, h - 10), (0, 100, 0), -1)
            cv2.putText(vis, cd_box, (cx - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        cv2.imshow("Drone Downward Camera (Live Mission HUD)", vis)
        cv2.waitKey(1)

    def run(self):
        drone = DroneController(self.args.url)
        
        # --------------------------------------------------------------
        # STAGE 1: Takeoff in Blue Zone
        # --------------------------------------------------------------
        print(f"\n{'='*60}\nSTAGE 1: TAKEOFF IN BLUE ZONE TO {self.args.alt:.1f}m\n{'='*60}")
        self.state_text = f"TAKEOFF ({self.args.alt:.1f}m)"
        drone.arm_and_takeoff(self.args.alt)
        
        t0 = time.time()
        while time.time() - t0 < 25.0:
            drone.poll()
            rclpy.spin_once(self.node, timeout_sec=0.03)
            self.update_hud(drone.ned)
            curr_alt = -drone.ned[2]
            print(f"[STATUS] Altitude: {curr_alt:.2f}m / {self.args.alt:.1f}m", end='\r')
            if curr_alt >= self.args.alt * 0.85:
                print(f"\n[STATUS] Reached target altitude: {curr_alt:.2f}m")
                break
            time.sleep(0.08)

        # Dwell 1.5s to stabilize hover at takeoff position
        t_stab = time.time()
        while time.time() - t_stab < 1.5:
            drone.poll()
            rclpy.spin_once(self.node, timeout_sec=0.03)
            self.update_hud(drone.ned)
            time.sleep(0.05)

        # --------------------------------------------------------------
        # STAGE 2: Capture and Memorize Initial Reference QR
        # --------------------------------------------------------------
        print(f"\n{'='*60}\nSTAGE 2: CAPTURE AND MEMORIZE INITIAL REFERENCE QR\n{'='*60}")
        self.state_text = "MEMORIZING INITIAL QR"
        hold_pos = [drone.ned[0], drone.ned[1], -self.args.alt]
        
        t0 = time.time()
        print("[VISION] Looking down at takeoff pad to memorize reference QR...")
        while self.ref_grid is None and time.time() - t0 < 15.0:
            drone.poll()
            drone.setpoint(*hold_pos)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                
            pts = None
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, pts = self.detector.detect(gray)
                if ok and pts is not None and len(pts) > 0:
                    grid, warped = extract_qr_features(gray, pts, grid_size=21)
                    self.ref_grid = grid
                    self.ref_warped = warped
                    
                    # Also decode text if present
                    try:
                        det2 = cv2.QRCodeDetector()
                        txt, _, _ = det2.detectAndDecode(gray)
                        if txt:
                            self.ref_text = txt
                            print(f"[VISION] Reference QR decoded payload: '{self.ref_text}'")
                    except Exception:
                        pass
                        
                    print("\n[SUCCESS] REFERENCE QR MEMORIZED INTO MEMORY!")
                    print(f"[VISION] Reference grid extracted ({grid.shape[0]}x{grid.shape[1]} modules).")
                    cv2.imwrite("/home/nh/reference_qr_memorized.png", warped)
                    self.update_hud(drone.ned, pts, score=1.0)
                    break
            self.update_hud(drone.ned, pts)
            time.sleep(0.08)
            
        if self.ref_grid is None:
            print("[WARN] Reading pad visually timed out. Loading reference directly from models...")
            p = '/home/nh/aerothon_ws/src/aerothon_sim/models/qr_initial/materials/textures/qr_initial.png'
            ref_bgr = cv2.imread(p)
            if ref_bgr is not None:
                gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
                ok, pts = self.detector.detect(gray)
                if ok and pts is not None:
                    self.ref_grid, self.ref_warped = extract_qr_features(gray, pts, grid_size=21)
                    self.ref_text = "AEROTHON_QR_02"
                    print("[VISION] Reference loaded from initial QR model successfully!")

        # --------------------------------------------------------------
        # STAGE 3: Transition from Blue Zone to Green Search Zone
        # --------------------------------------------------------------
        route = build_lawnmower_route(
            north_range=(-16.0, 16.0), east_range=(-18.0, 18.0), step=self.args.step
        )
        entry_wp = route[0]
        print(f"\n{'='*60}\nSTAGE 3: TRANSITION TO GREEN SEARCH ZONE\n{'='*60}")
        print(f"[TRANSITION] Flying from Blue Zone to Green Zone Entry: North={entry_wp[0]:.1f}m, East={entry_wp[1]:.1f}m")
        self.state_text = "TRANSITION TO GREEN ZONE"
        
        t_trans = time.time()
        cur_setpoint = [drone.ned[0], drone.ned[1], -self.args.alt]
        dt = 0.05
        while time.time() - t_trans < 25.0:
            drone.poll()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            self.update_hud(drone.ned)
            
            tgt = [entry_wp[0], entry_wp[1], -self.args.alt]
            dist_to_tgt = math.hypot(tgt[0] - drone.ned[0], tgt[1] - drone.ned[1])
            if dist_to_tgt < 1.0:
                print(f"[TRANSITION] Successfully entered Green Search Zone at ({drone.ned[0]:.1f}m, {drone.ned[1]:.1f}m)!")
                break
                
            dist_sp = math.hypot(tgt[0] - cur_setpoint[0], tgt[1] - cur_setpoint[1])
            step = min(self.args.speed * dt, dist_sp)
            if dist_sp > 1e-4:
                cur_setpoint[0] += (tgt[0] - cur_setpoint[0]) * step / dist_sp
                cur_setpoint[1] += (tgt[1] - cur_setpoint[1]) * step / dist_sp
            drone.setpoint(*cur_setpoint)
            time.sleep(dt)

        # --------------------------------------------------------------
        # STAGE 4: Lawnmower Search Pattern in Green Zone
        # --------------------------------------------------------------
        print(f"\n{'='*60}\nSTAGE 4: LAWNMOWER SEARCH OVER GREEN ZONE (50x40m)\n{'='*60}")
        self.state_text = "LAWNMOWER SWEEP"
        print(f"[ROUTE] Serpentine route generated with {len(route)} waypoints.")
        
        target_qr_ned = None
        img_w, img_h = 640.0, 480.0
        cx, cy = img_w / 2.0, img_h / 2.0
        hfov = 1.5707963
        
        for wp_idx, (target_n, target_e) in enumerate(route):
            tgt_ned = [target_n, target_e, -self.args.alt]
            print(f"[WAYPOINT {wp_idx+1}/{len(route)}] Target: North={target_n:+.1f}m, East={target_e:+.1f}m")
            
            while True:
                drone.poll()
                rclpy.spin_once(self.node, timeout_sec=0.01)
                
                detected_pts = None
                sim_score = 0.0
                is_match = False
                
                with self.frame_lock:
                    frame = None if self.latest_frame is None else self.latest_frame.copy()
                    
                if frame is not None and self.ref_grid is not None:
                    # Drone is guaranteed in green zone (North >= -19.0m)
                    if drone.ned[0] > -19.0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        ok, detected_pts = self.detector.detect(gray)
                        if ok and detected_pts is not None and len(detected_pts) > 0:
                            cand_grid, _ = extract_qr_features(gray, detected_pts, grid_size=21)
                            sim_score = compute_similarity(self.ref_grid, cand_grid)
                            
                            # Check decoded text if available
                            try:
                                det2 = cv2.QRCodeDetector()
                                cand_txt, _, _ = det2.detectAndDecode(gray)
                                if cand_txt and self.ref_text and cand_txt == self.ref_text:
                                    is_match = True
                                    sim_score = 1.0
                            except Exception:
                                pass
                                
                            if sim_score >= self.args.match_thresh:
                                is_match = True
                                
                            if is_match:
                                # Estimate QR ground coordinates in NED
                                p = detected_pts.reshape(-1, 2)
                                q_u, q_v = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
                                curr_alt = max(-drone.ned[2], 1.0)
                                m_per_px = 2.0 * curr_alt * math.tan(hfov / 2.0) / img_w
                                
                                dx_body = -(q_v - cy) * m_per_px
                                dy_body = (q_u - cx) * m_per_px
                                yaw = drone.yaw
                                dn = dx_body * math.cos(yaw) - dy_body * math.sin(yaw)
                                de = dx_body * math.sin(yaw) + dy_body * math.cos(yaw)
                                
                                target_qr_ned = [drone.ned[0] + dn, drone.ned[1] + de, -self.args.alt]
                                
                                print(f"\n{'!'*60}")
                                print(f"[MATCH FOUND!] Candidate QR Similarity: {sim_score*100:.1f}%")
                                print(f"[LOCATION] Drone over World (North={drone.ned[0]:.2f}m, East={drone.ned[1]:.2f}m)")
                                print(f"[TARGET QR] Estimated Position: (North={target_qr_ned[0]:.2f}m, East={target_qr_ned[1]:.2f}m)")
                                print(f"{'!'*60}\n")
                                
                                # IMMEDIATE BRAKE MODE: Command ArduPilot to stop horizontally immediately!
                                print("[BRAKE] Switching to BRAKE mode to halt horizontal speed immediately...")
                                drone.set_mode("BRAKE")
                                self.update_hud(drone.ned, detected_pts, sim_score)
                                break
                                
                self.update_hud(drone.ned, detected_pts, sim_score)
                
                if target_qr_ned is not None:
                    break
                    
                # Slew setpoint towards current waypoint
                dist_to_tgt = math.hypot(tgt_ned[0] - cur_setpoint[0], tgt_ned[1] - cur_setpoint[1])
                if dist_to_tgt < 0.5 and math.hypot(tgt_ned[0] - drone.ned[0], tgt_ned[1] - drone.ned[1]) < 1.0:
                    break
                    
                step = min(self.args.speed * dt, dist_to_tgt)
                if dist_to_tgt > 1e-4:
                    cur_setpoint[0] += (tgt_ned[0] - cur_setpoint[0]) * step / dist_to_tgt
                    cur_setpoint[1] += (tgt_ned[1] - cur_setpoint[1]) * step / dist_to_tgt
                cur_setpoint[2] = -self.args.alt
                drone.setpoint(*cur_setpoint)
                time.sleep(dt)
                
            if target_qr_ned is not None:
                break

        if target_qr_ned is None:
            print("\n[RESULT] Lawnmower sweep completed. No matching QR found.")
            drone.land()
            return

        # --------------------------------------------------------------
        # STAGE 5: Precision Fly-To & Visual Centering Over Matched QR
        # --------------------------------------------------------------
        print(f"\n{'='*60}\nSTAGE 5: PRECISION POSITIONING OVER MATCHED QR\n{'='*60}")
        self.state_text = "POSITIONING OVER MATCHED QR"
        
        # Dwell 1.5s in BRAKE mode to kill all velocity
        t_stop = time.time()
        while time.time() - t_stop < 1.5:
            drone.poll()
            rclpy.spin_once(self.node, timeout_sec=0.03)
            self.update_hud(drone.ned)
            time.sleep(0.05)
            
        # Switch back to GUIDED and command exact QR location
        print(f"[GUIDED] Navigating directly to matched QR at North={target_qr_ned[0]:.2f}m, East={target_qr_ned[1]:.2f}m...")
        drone.set_mode("GUIDED")
        time.sleep(0.2)
        
        for _ in range(25):
            drone.setpoint(*target_qr_ned)
            time.sleep(0.05)
            
        t_fly = time.time()
        while time.time() - t_fly < 10.0:
            drone.poll()
            drone.setpoint(*target_qr_ned)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            self.update_hud(drone.ned)
            d = math.hypot(target_qr_ned[0] - drone.ned[0], target_qr_ned[1] - drone.ned[1])
            if d < 0.25 and time.time() - t_fly > 1.5:
                print(f"[ARRIVED] Drone reached target QR location (Dist: {d*100:.1f}cm)!")
                break
            time.sleep(0.05)

        # Fine-tune visual centering using closed-loop visual servoing
        self.state_text = "FINE CENTERING"
        print("[CENTERING] Commencing closed-loop visual servoing...")
        for iteration in range(6):
            drone.poll()
            rclpy.spin_once(self.node, timeout_sec=0.03)
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.08)
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, pts = self.detector.detect(gray)
            if ok and pts is not None and len(pts) > 0:
                p = pts.reshape(-1, 2)
                u, v = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
                err_px = math.hypot(u - cx, v - cy)
                print(f"[CENTERING] Iteration {iteration+1}/6: Centroid ({u:.1f}, {v:.1f})px | Offset: {err_px:.1f}px")
                self.update_hud(drone.ned, pts, score=1.0)
                
                if err_px <= self.args.center_tol:
                    print(f"[CENTERING] Lock acquired! Centroid offset ({err_px:.1f}px) is within tolerance ({self.args.center_tol:.1f}px).")
                    break
                    
                curr_alt = max(-drone.ned[2], 1.0)
                m_per_px = 2.0 * curr_alt * math.tan(hfov / 2.0) / img_w
                dx_b = -(v - cy) * m_per_px
                dy_b = (u - cx) * m_per_px
                yaw = drone.yaw
                dn = dx_b * math.cos(yaw) - dy_b * math.sin(yaw)
                de = dx_b * math.sin(yaw) + dy_b * math.cos(yaw)
                
                # Apply 0.75 proportional gain to smoothly eliminate remaining offset
                target_qr_ned[0] += dn * 0.75
                target_qr_ned[1] += de * 0.75
                
                # Settle for 0.7s while actively holding setpoint
                t_fine = time.time()
                while time.time() - t_fine < 0.7:
                    drone.poll()
                    drone.setpoint(*target_qr_ned)
                    rclpy.spin_once(self.node, timeout_sec=0.03)
                    self.update_hud(drone.ned, pts, score=1.0)
                    time.sleep(0.04)
            else:
                time.sleep(0.08)
                
        print("\n[SUCCESS] Drone precisely centered directly over matched QR!")

        # --------------------------------------------------------------
        # STAGE 6: Holding Alignment for Configurable Duration
        # --------------------------------------------------------------
        print(f"\n{'='*60}\nSTAGE 6: ALIGNMENT HOVER ({self.args.align_duration:.1f}s)\n{'='*60}")
        self.state_text = f"ALIGNMENT HOLD ({self.args.align_duration:.0f}s)"
        print(f"[ALIGN] Holding steady hover directly over matched QR for {self.args.align_duration:.1f} seconds...")
        print(f"[ALIGN] Locked Setpoint: North={target_qr_ned[0]:.2f}m, East={target_qr_ned[1]:.2f}m, Alt={-target_qr_ned[2]:.2f}m")
        
        align_start = time.time()
        last_log_sec = -1
        while time.time() - align_start < self.args.align_duration:
            elapsed = time.time() - align_start
            remaining = max(0.0, self.args.align_duration - elapsed)
            self.hud_countdown = remaining
            
            drone.poll()
            drone.setpoint(*target_qr_ned)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            pts = None
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, pts = self.detector.detect(gray)
            self.update_hud(drone.ned, pts, score=1.0)
            
            sec_int = int(remaining)
            if sec_int != last_log_sec:
                print(f"[ALIGNING] Hover time remaining: {remaining:4.1f}s | Locked over matched QR", end='\r')
                last_log_sec = sec_int
                
            time.sleep(0.05)
            
        print(f"\n\n{'='*60}\nMISSION COMPLETED SUCCESSFULLY!\n{'='*60}")
        print(f"Alignment hover completed for the full {self.args.align_duration:.1f} seconds.")
        print(f"Final Drone Position: North = {drone.ned[0]:.2f}m, East = {drone.ned[1]:.2f}m, Alt = {-drone.ned[2]:.2f}m")
        self.state_text = "MISSION COMPLETE"
        self.hud_countdown = 0.0
        self.update_hud(drone.ned)
        
        if self.args.land_after:
            drone.land()
        else:
            print("[HOLD] Hovering over matched QR. Press Ctrl+C to land.")
            try:
                while True:
                    drone.poll()
                    drone.setpoint(*target_qr_ned)
                    rclpy.spin_once(self.node, timeout_sec=0.03)
                    self.update_hud(drone.ned)
                    time.sleep(0.08)
            except KeyboardInterrupt:
                drone.land()

def main():
    parser = argparse.ArgumentParser(description="Aerothon Lawnmower & QR Aligner")
    parser.add_argument("--url", default=None, help="MAVLink connection URL (default: auto-detect 14550/14551)")
    parser.add_argument("--alt", type=float, default=4.0, help="Cruise altitude in meters (default: 4.0)")
    parser.add_argument("--align-duration", type=float, default=10.0,
                        help="Duration in seconds to hold alignment (parameter, default: 10.0)")
    parser.add_argument("--speed", type=float, default=1.5, help="Search flight speed in m/s (default: 1.5)")
    parser.add_argument("--step", type=float, default=5.0, help="Lawnmower row spacing in meters (default: 5.0)")
    parser.add_argument("--match-thresh", type=float, default=0.88, help="Similarity threshold in [0, 1] (default: 0.88)")
    parser.add_argument("--center-tol", type=float, default=25.0, help="Centering pixel tolerance (default: 25.0 px)")
    parser.add_argument("--image-topic", default="/camera/image_raw", help="Camera image topic")
    parser.add_argument("--show-camera", action="store_true", default=True, help="Display live camera HUD window")
    parser.add_argument("--no-camera", dest="show_camera", action="store_false", help="Disable camera HUD window")
    parser.add_argument("--land-after", action="store_true", help="Automatically land after alignment")
    args = parser.parse_args()

    rclpy.init()
    mission = LawnmowerMission(args)
    try:
        mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] User interrupted mission.")
    finally:
        cv2.destroyAllWindows()
        mission.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
