#!/usr/bin/env python3
"""
SIYI Camera Feed Verifier & Latency Benchmark
---------------------------------------------
Uses the threaded zero-latency SiyiStreamWorker from AGENTS.md / Real-World Guide.
Continuously flushes RTSP decode buffers in a dedicated background worker to
provide the freshest video frame at 25-30 FPS with sub-frame latency.

Usage:
  python3 verify_siyi_stream.py
  python3 verify_siyi_stream.py --stream sub
  python3 verify_siyi_stream.py --no-display
  python3 verify_siyi_stream.py --url rtsp://192.168.144.25:8554/main.264
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import cv2
import numpy as np

CAMERA_IP = "192.168.144.25"
MAIN_RTSP = f"rtsp://{CAMERA_IP}:8554/main.264"
DEFAULT_RTSP = MAIN_RTSP
SUB_RTSP = f"rtsp://{CAMERA_IP}:8554/sub.264"


class SiyiStreamWorker:
    """Threaded SIYI RTSP Stream Worker (Zero Frame Latency).
    
    Continuously drains ffmpeg network buffer so consumer code always
    receives the absolute latest camera frame.
    """
    def __init__(self, capture_source=DEFAULT_RTSP):
        self.capture_source = capture_source
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.fps = 0.0
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_stream(self):
        print(f"[SIYI WORKER] Opening stream: {self.capture_source}")
        # Optimize OpenCV FFMPEG capture backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        self.cap = cv2.VideoCapture(self.capture_source, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.connected = True
            print(f"[SIYI WORKER] RTSP stream opened successfully.")
            return True
        else:
            self.connected = False
            print(f"[SIYI WORKER] Failed to open stream {self.capture_source}")
            return False

    def _update(self):
        fps_timer = time.time()
        fps_counter = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open_stream():
                    time.sleep(1.0)
                    continue

            ret, img = self.cap.read()
            now = time.time()
            if ret and img is not None:
                with self.lock:
                    self.frame = img
                    self.frame_count += 1
                    self.last_frame_time = now
                    self.connected = True
                
                fps_counter += 1
                if now - fps_timer >= 1.0:
                    with self.lock:
                        self.fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now
            else:
                self.connected = False
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_stats(self):
        with self.lock:
            return {
                "connected": self.connected,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "last_frame_age_s": (time.time() - self.last_frame_time) if self.last_frame_time > 0 else 999.0,
                "has_frame": self.frame is not None
            }

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


def check_network(ip=CAMERA_IP):
    print("=" * 60)
    print(f"[NETWORK CHECK] Probing SIYI Camera at IP: {ip}")
    print("=" * 60)
    res = subprocess.run(["ping", "-c", "2", "-W", "1", ip], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[NETWORK CHECK] SUCCESS! Camera responded to ICMP ping.")
        return True
    else:
        print(f"[NETWORK CHECK] PING FAILED (Camera not reachable at {ip}).")
        print("\nTroubleshooting Checks:")
        print("1. Is the Ethernet cable connected to the Pi 5's RJ45 eth0 port?")
        print("2. Is the SIYI camera powered with 12V-24V? (Check indicator LEDs)")
        print("3. Is eth0 configured with static IP 192.168.144.50/24?")
        print("   Run: sudo nmcli con up siyi-eth")
        return False


def main():
    parser = argparse.ArgumentParser(description="SIYI Camera Zero-Latency Feed Verifier")
    parser.add_argument("--url", default=None, help="Custom RTSP URL")
    parser.add_argument("--stream", choices=["main", "sub"], default="main",
                        help="Select main (1080p) or sub (720p) stream")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (no OpenCV window)")
    parser.add_argument("--skip-ping", action="store_true", help="Skip initial ping check")
    args = parser.parse_args()

    rtsp_url = args.url or (MAIN_RTSP if args.stream == "main" else SUB_RTSP)

    if not args.skip_ping:
        net_ok = check_network(CAMERA_IP)
        if not net_ok:
            print("\nProceeding with RTSP probe anyway in case ping is blocked...")

    print(f"\n[STREAM] Launching SiyiStreamWorker on: {rtsp_url}")
    worker = SiyiStreamWorker(capture_source=rtsp_url)

    # Wait for first frame
    print("[STREAM] Waiting for video stream frames (up to 10s)...")
    t0 = time.time()
    first_frame = None
    while time.time() - t0 < 10.0:
        stats = worker.get_stats()
        if stats["has_frame"]:
            first_frame = worker.get_frame()
            break
        time.sleep(0.2)

    if first_frame is None:
        print("\n[ERROR] No video frames received within 10 seconds.")
        print("Please check power to the SIYI camera, physical Ethernet cabling, and static IP configuration.")
        worker.stop()
        sys.exit(1)

    h, w = first_frame.shape[:2]
    print(f"[STREAM OK] First frame received successfully! Resolution: {w}x{h}")
    snapshot_path = "/home/suasrasp/aerothon_ws/siyi_test_capture.jpg"
    cv2.imwrite(snapshot_path, first_frame)
    print(f"[SNAPSHOT] Saved initial snapshot to: {snapshot_path}")

    show_gui = not args.no_display and ("DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
    if show_gui:
        print("\n[DISPLAY] Showing live video window. Press 'q' to quit, 's' to save snapshot.")
        cv2.namedWindow("SIYI Camera Live Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SIYI Camera Live Feed", 960, 540)

    try:
        last_log = time.time()
        while True:
            frame = worker.get_frame()
            stats = worker.get_stats()

            if frame is not None and show_gui:
                vis = frame.copy()
                vh, vw = vis.shape[:2]
                cx, cy = vw // 2, vh // 2

                # Center crosshairs
                cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.circle(vis, (cx, cy), 16, (0, 255, 0), 1)

                # Info overlay
                fps_txt = f"SIYI A8 mini | Stream: {args.stream.upper()} | {vw}x{vh} @ {stats['fps']:.1f} FPS"
                cv2.rectangle(vis, (0, 0), (vw, 36), (20, 20, 20), -1)
                cv2.putText(vis, fps_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("SIYI Camera Live Feed", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    p = f"/home/suasrasp/aerothon_ws/siyi_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(p, frame)
                    print(f"[SNAPSHOT] Manually saved snapshot to: {p}")

            if time.time() - last_log >= 2.0:
                print(f"[HEALTH] Receiving: {stats['frame_count']} frames | Rate: {stats['fps']:.1f} FPS | Resolution: {w}x{h} | Latency age: {stats['last_frame_age_s']*1000:.1f} ms")
                last_log = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[STOP] User stopped verifier.")
    finally:
        worker.stop()
        if show_gui:
            cv2.destroyAllWindows()
        print("[VERIFIER] Stream worker stopped cleanly.")


if __name__ == '__main__':
    main()
