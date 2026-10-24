#!/usr/bin/env python3
"""
QR Test Card Display Utility for Benchtop Testing
-------------------------------------------------
Displays each test QR tile fullscreen on your screen or HDMI monitor
so you can hold it in front of the SIYI camera during bench testing.

Controls:
  Press '0' -> Reference Initial QR (qr_initial: AEROTHON_QR_02)
  Press '1' -> Distractor QR 1 (qr_tile_1: AEROTHON_QR_01)
  Press '2' -> Matching Twin QR 2 (qr_tile_2: AEROTHON_QR_02 - 100% MATCH)
  Press '3' -> Distractor QR 3 (qr_tile_3: AEROTHON_QR_03)
  Press '4' -> Distractor QR 4 (qr_tile_4: AEROTHON_QR_04)
  Press 'q' -> Exit viewer
"""

import os
import sys
import cv2
import numpy as np

TILES = {
    '0': ("REFERENCE QR (AEROTHON_QR_02)", "/home/suasrasp/aerothon_ws/src/aerothon_sim/models/qr_initial/materials/textures/qr_initial.png"),
    '1': ("DISTRACTOR 1 (AEROTHON_QR_01)", "/home/suasrasp/aerothon_ws/src/aerothon_sim/models/qr_tile_1/materials/textures/qr_tile_1.png"),
    '2': ("TWIN MATCH 2 (AEROTHON_QR_02 - TARGET)", "/home/suasrasp/aerothon_ws/src/aerothon_sim/models/qr_tile_2/materials/textures/qr_tile_2.png"),
    '3': ("DISTRACTOR 3 (AEROTHON_QR_03)", "/home/suasrasp/aerothon_ws/src/aerothon_sim/models/qr_tile_3/materials/textures/qr_tile_3.png"),
    '4': ("DISTRACTOR 4 (AEROTHON_QR_04)", "/home/suasrasp/aerothon_ws/src/aerothon_sim/models/qr_tile_4/materials/textures/qr_tile_4.png")
}

def main():
    win = "QR Bench Test Cards (Press 0, 1, 2, 3, 4 | 'q' to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 700, 750)

    cur_key = '0'
    while True:
        title, path = TILES[cur_key]
        img = cv2.imread(path)
        if img is None:
            canvas = np.zeros((600, 600, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Missing: {path}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            img_resized = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
            canvas = np.full((720, 640, 3), 255, dtype=np.uint8)
            canvas[80:680, 20:620] = img_resized
            col = (0, 150, 0) if cur_key in ['0', '2'] else (0, 0, 200)
            cv2.putText(canvas, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            cv2.putText(canvas, "Keys: [0]=Ref | [1]=Distractor 1 | [2]=Twin Match | [3]=Distractor 3 | [q]=Quit", 
                        (20, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in TILES:
            cur_key = chr(key)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
