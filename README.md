# Aerothon 2026 — Lawnmower Survey, QR Similarity & Precision Visual Servoing

Autonomous aerial reconnaissance, rotation-invariant QR similarity matching, and precision visual servoing for ArduPilot Copter in Gazebo Sim 8 (Harmonic) with ROS 2 Humble.

📄 **Full Technical Systems Manual:** See [aerothon_system_guide.pdf](aerothon_system_guide.pdf) *(HTML version: [aerothon_system_guide.html](aerothon_system_guide.html))*.

---

## Key Capabilities

1. **Dual-Zone Spatial Isolation:**
   - **Blue Takeoff Zone ($50\text{m} \times 10\text{m}$):** Drone spawns over initial reference pad at $(N=-25.0, E=0.0)$, takes off to 4.0m, and captures reference QR into memory.
   - **Green Search Zone ($50\text{m} \times 40\text{m}$):** Lawnmower survey strictly inside the search zone, isolated by a 5m buffer to eliminate false self-detections.
2. **Rotation-Invariant Visual Similarity (No Text Decoding Required):**
   - Perspective-warps candidate QR quad to canonical $210 \times 210$ px space.
   - Samples a $21 \times 21$ boolean module matrix.
   - Computes rotational correlation across all 4 discrete rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
   - Triggers match when similarity $S \ge 0.88$ (or matching payload `AEROTHON_QR_02`).
3. **Position-Based Visual Servoing (PBVS):**
   - Targeted `MAV_CMD_DO_SET_MODE(BRAKE)` halts horizontal speed immediately to 0.0 m/s.
   - Pinhole camera reprojection calculates absolute world coordinates $(N_{\text{target}}, E_{\text{target}})$.
   - Multi-iteration closed-loop servoing achieves sub-centimeter alignment ($0.7\text{ px}$ optical error).
4. **Configurable Alignment Hover:**
   - Holds steady position lock over the matched QR for `--align-duration` (default 10.0s).
   - Live camera HUD with crosshair, dynamic bounding quad, centroid displacement vector, and countdown timer.

---

## Repository Structure

```
aerothon_ws/
├── aerothon_system_guide.pdf  # Comprehensive 6-page technical report
├── aerothon_system_guide.html # Standalone printable HTML report
├── README.md                  # Project overview and quick start
├── scripts/
│   ├── lawnmower_qr_mission.py # Core autonomous state machine & vision aligner
│   ├── fly_to.py              # Precision NED waypoint navigation utility
│   ├── run_sim.sh             # Gazebo Sim 8 world launcher
│   ├── run_sitl.sh            # ArduPilot Copter SITL launcher
│   ├── run_mission.sh         # Mission launcher with ros_gz_bridge daemon
│   └── run_cam_view.sh        # Standalone camera viewer
└── src/
    └── aerothon_sim/          # ROS 2 simulation package
        ├── CMakeLists.txt
        ├── package.xml
        ├── models/            # 2m x 2m QR tile SDF models
        └── worlds/            # 50x50m blue/green field SDF world
```

---

## Quick Start Guide

### 1. Build ROS 2 Package
```bash
cd ~/aerothon_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch Simulation (Three Terminals)

**Terminal 1: Gazebo Environment**
```bash
cd ~/aerothon_ws
./scripts/run_sim.sh
```

**Terminal 2: ArduPilot SITL**
```bash
cd ~/aerothon_ws
./scripts/run_sitl.sh
```

**Terminal 3: Autonomous Mission**
```bash
cd ~/aerothon_ws
./scripts/run_mission.sh --align-duration 15.0 --speed 2.0
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--alt` | float | `4.0` | Flight survey altitude in meters AGL. |
| `--align-duration` | float | `10.0` | Hold duration in seconds for steady visual alignment hover. |
| `--speed` | float | `1.5` | Lawnmower search traversal velocity in m/s. |
| `--step` | float | `5.0` | East column spacing between serpentine sweeps in meters. |
| `--match-thresh` | float | `0.88` | Minimum similarity threshold [0.0 – 1.0]. |
| `--center-tol` | float | `25.0` | Centering pixel tolerance before declaring lock. |
| `--land-after` | flag | `False` | Automatically commands landing after alignment completes. |
| `--no-camera` | flag | `False` | Disable live OpenCV HUD window. |

---

## Telemetry & Benchmarks

| Metric | Original Run | Corrected Run (V2.0) | Result |
| :--- | :--- | :--- | :--- |
| **Twin Recognition** | 90.6% | 100.0% (`AEROTHON_QR_02`) | Confirmed |
| **Braking Response** | Dropped by autopilot | `COMMAND_ACK: 176 Result: 0` | Instant halt |
| **Target Estimation** | Unlatched drift | $(N=7.93\text{m}, E=-11.99\text{m})$ | Error $< 7\text{ cm}$ |
| **Centroid Error** | 228 px $\to$ 160 px | **0.7 pixels from center** | **Sub-centimeter (0.8 cm)** |
| **Final Location** | Landed on grass at $(-1.8, 3.0)$ | Locked hover over $(7.93, -11.99)$ | **Dead-center over QR 2** |
