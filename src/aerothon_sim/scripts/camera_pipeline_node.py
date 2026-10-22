#!/usr/bin/env python3
"""
Camera Pipeline Node for Aerothon
----------------------------------
Subscribes to /camera/image_raw (bridged from Gazebo Sim)
Converts image using robust pure-numpy conversion (no cv_bridge numpy ABI issues)
Provides a modular detector hook where custom algorithms can be inserted.
Displays the live feed with crosshairs and any detections.
Publishes annotated image to /camera/debug_image.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import time

class CameraPipelineNode(Node):
    def __init__(self):
        super().__init__('camera_pipeline_node')
        
        # Parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('show_window', True)
        self.declare_parameter('enable_qr_detector', True)
        
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self.enable_qr = self.get_parameter('enable_qr_detector').get_parameter_value().bool_value
        
        # Detector initialization (OpenCV built-in QR detector)
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            self.qr_detector = cv2.QRCodeDetectorAruco()
        else:
            self.qr_detector = cv2.QRCodeDetector()
            
        # Subscription
        self.sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        
        # Publisher for processed/annotated stream
        self.debug_pub = self.create_publisher(Image, '/camera/debug_image', 10)
        
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        self.get_logger().info(f"Camera Pipeline Node started. Subscribed to: {image_topic}")
        self.get_logger().info("Publishing annotated debug feed on: /camera/debug_image")

    def msg_to_cv2(self, msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image to BGR numpy array without cv_bridge dependency."""
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding in ['rgb8', 'RGB8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im.copy()

    def cv2_to_msg(self, cv_img: np.ndarray, header) -> Image:
        """Convert BGR numpy array to sensor_msgs/Image."""
        msg = Image()
        msg.header = header
        msg.height, msg.width = cv_img.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = cv_img.tobytes()
        return msg

    def detect_and_annotate(self, frame: np.ndarray):
        """
        ========================================================================
        DETECTOR PIPELINE HOOK:
        You can plug in your own custom detector here (YOLO, ArUco, color, etc.)
        ========================================================================
        """
        detections = []
        
        if self.enable_qr:
            try:
                # Detect QR codes
                val, pts, _ = self.qr_detector.detectAndDecode(frame)
                if pts is not None and len(pts) > 0:
                    pts = np.int32(pts).reshape(-1, 2)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                    
                    # Compute centroid
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    
                    label = val if val else "QR DETECTED"
                    cv2.putText(frame, label, (cx - 40, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    detections.append({'text': val, 'center': (cx, cy)})
            except Exception as e:
                pass
                
        return frame, detections

    def image_callback(self, msg: Image):
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now
            
        # 1. Convert to OpenCV
        frame = self.msg_to_cv2(msg)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # 2. Run detector pipeline
        annotated, detections = self.detect_and_annotate(frame)
        
        # 3. Draw crosshair at camera center (drone optical center)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(annotated, (cx, cy), 15, (0, 255, 255), 1)
        
        # 4. Display telemetry overlay
        info_text = f"{w}x{h} | FPS: {self.fps:.1f}"
        cv2.putText(annotated, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status_text = f"Detections: {len(detections)}"
        cv2.putText(annotated, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if len(detections) > 0 else (180, 180, 180), 2)
        
        # 5. Publish annotated image
        self.debug_pub.publish(self.cv2_to_msg(annotated, msg.header))
        
        # 6. Show OpenCV GUI window if requested
        if self.show_window:
            cv2.imshow("Aerothon Downward Camera Feed", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                snap_name = f"/home/nh/camera_snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_name, annotated)
                self.get_logger().info(f"Saved snapshot to: {snap_name}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraPipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
