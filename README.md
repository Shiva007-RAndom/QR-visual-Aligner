# Aerothon 2026 — Lawnmower Survey, QR Similarity & Precision Visual Servoing

[![Platform](https://img.shields.io/badge/Autopilot-ArduPilot%20Copter%204.5%2B-blue.svg)](https://ardupilot.org/copter/)
[![Hardware](https://img.shields.io/badge/Airframe-Holybro%20X500%20Multirotor-orange.svg)](https://holybro.com/)
[![Companion](https://img.shields.io/badge/Companion-Raspberry%20Pi%205%20(8GB)-red.svg)](https://www.raspberrypi.com/)
[![Sensor](https://img.shields.io/badge/Optics-SIYI%20A8%20mini%204K%20Ethernet-brightgreen.svg)](https://siyi.biz/)
[![Simulation](https://img.shields.io/badge/Sim-Gazebo%20Sim%208%20(Harmonic)%20%2B%20ROS%202-purple.svg)](https://gazebosim.org/)

Autonomous aerial reconnaissance, 4-rotation invariant QR similarity matching, nadir pinhole reprojection, and closed-loop visual servoing for both **physical hardware deployment** (Holybro X500 + Cube Orange+ + Raspberry Pi 5 + SIYI A8 mini) and **Gazebo Sim 8 (Harmonic) + ROS 2 Humble SITL simulation**.

---

## 📚 Technical Manuals & Guides

Comprehensive technical reports with wiring schematics, concurrency state diagrams, and mathematical derivations are available in the repository:

| Document | Format | Description |
| :--- | :--- | :--- |
| **Hardware Architecture & Testing Guide** | [PDF](docs/aerothon_hardware_architecture_and_testing_guide.pdf) \| [HTML](docs/aerothon_hardware_architecture_and_testing_guide.html) | Physical wiring schematics, companion computer thread concurrency, MAVLink baud rates, and bench test procedures. |
| **Real-World Deployment Manual** | [PDF](docs/real_world_deployment_guide.pdf) \| [HTML](docs/real_world_deployment_guide.html) | Complete field deployment handbook for Holybro X500, Cube Orange+, Pi 5, and SIYI camera. |
| **Visual Alignment Mathematical Deep Dive** | [PDF](docs/visual_alignment_deep_dive.pdf) \| [HTML](docs/visual_alignment_deep_dive.html) | Mathematical derivations for nadir ray reprojection, 4-rotation invariant module grid correlation, and PBVS control laws. |
| **Simulation & System Architecture Guide** | [PDF](aerothon_system_guide.pdf) \| [HTML](aerothon_system_guide.html) | Gazebo Sim 8 world generation, ROS 2 bridges, and ArduPilot SITL setup. |

---

## 🎬 Flight Demo Video

<p align="center">
  <img src="assets/demo_highlight.gif" alt="Autonomous QR Detection & Precision Alignment Flight Demo" width="88%">
</p>
<p align="center">
  <em>Live Flight Sequence: Blue Zone takeoff &bull; Reference QR memorization &bull; Lawnmower survey &bull; 100% similarity lock &bull; Instant BRAKE &bull; Sub-centimeter alignment hold &bull; Precision touchdown.</em>
  <br>
  🎥 <strong><a href="assets/demo.mp4">▶ Click here to view/download the full high-definition demo video (assets/demo.mp4)</a></strong>
</p>

<details>
<summary>▶ <strong>Click to expand embedded HTML5 video player</strong></summary>
<br>
<p align="center">
  <video src="assets/demo.mp4" controls="controls" width="100%" poster="assets/demo_thumbnail.jpg">
    Your browser does not support the video tag. You can <a href="assets/demo.mp4">download the MP4 directly</a>.
  </video>
</p>
</details>

---

## 🚀 Key System Capabilities

1. **Dual-Check Rotation-Invariant QR Recognition:**
   - **Visual Grid Agreement:** Warps candidate quad to a normalized $210 \times 210\text{ px}$ image, extracts the $21 \times 21$ boolean module matrix, and tests rotational correlation across all 4 discrete angles ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
   - **Dual-Verification:** Accepts candidates via visual module agreement ($S \ge 0.88$) or payload string match (`AEROTHON_QR_02`). Robust against optical distortion, blur, and lighting changes.
2. **Instant Horizontal Braking (`BRAKE` Mode):**
   - Commands ArduPilot `DO_SET_MODE(BRAKE)` via MAVLink 2 upon match confirmation, actively cancelling multirotor horizontal drift and preventing target overshoot.
3. **Nadir Pinhole Reprojection & Visual Servoing (PBVS):**
   - Projects camera centroid displacement $(u - c_x, v - c_y)$ to physical meters using current AGL altitude and horizontal FOV:
     $$\Delta N = -\Delta v \cdot S_m \cos(\psi) - \Delta u \cdot S_m \sin(\psi), \quad \Delta E = -\Delta v \cdot S_m \sin(\psi) + \Delta u \cdot S_m \cos(\psi)$$
   - Iterative closed-loop visual servoing converges to sub-centimeter accuracy ($< 0.8\text{ cm}$ physical error).
4. **Configurable Alignment Hover & Autonomous Touchdown:**
   - Maintains steady alignment hover for a configurable duration (`--align-duration 15.0`) directly over the target QR with live HUD telemetry.
   - Automatically switches to `LAND` mode directly on top of the matched QR upon hold completion.
5. **Pure-Python Architecture on Companion Computer:**
   - Extremely lightweight: runs on Raspberry Pi 5 under Ubuntu 24.04 without ROS 2 overhead (<15% single-core CPU usage).
   - Zero-latency threaded video grabber via OpenCV FFMPEG RTSP worker.

---

## 🛠️ Physical Hardware Architecture

```
                                  HOLYBRO X500 MULTIROTOR
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                       │
 │   ┌───────────────────────┐       RTSP (H.265 / H.264)        ┌───────────────────┐   │
 │   │    SIYI A8 mini /     │ ────────────────────────────────> │   Raspberry Pi 5  │   │
 │   │    ZR10 Gimbal Cam    │     192.168.144.25:8554/main.264  │   (8GB, Ubuntu)   │   │
 │   └───────────────────────┘                                   └─────────┬─────────┘   │
 │       │                                                                 │             │
 │       │ Nadir (-90°)                                    MAVLink 2 Serial│             │
 │       ▼                                                 /dev/ttyACM0    │             │
 │   Ground QR Target                                      @ 921,600 baud  ▼             │
 │                                                               ┌───────────────────┐   │
 │                                                               │   Cube Orange+    │   │
 │                                                               │  (ArduPilot 4.5+) │   │
 │                                                               └─────────┬─────────┘   │
 │                                                                         │             │
 │                                                                    DShot / PWM        │
 │                                                                         ▼             │
 │                                                                    4x ESCs/Motors     │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

### Hardware Pinout & Wiring Specifications

| Connection | Origin Port | Destination Port | Protocol / Signal | Configuration Details |
| :--- | :--- | :--- | :--- | :--- |
| **Telemetry** | Cube Orange+ USB / TELEM2 | Pi 5 USB-A (`/dev/ttyACM0`) | MAVLink 2 (Serial) | `921600 baud`, 8-N-1, `SERIALx_BAUD = 921` |
| **Ethernet** | SIYI A8 mini Gimbal Port | Pi 5 `eth0` | RTSP over UDP/TCP | Camera: `192.168.144.25`, Pi 5: `192.168.144.50/24` |
| **Power** | 4S/6S LiPo Battery | Holybro Power Module (PDB) | 5.3V / 3A Regulated | Powers Cube Orange+ via Power1 port |
| **Pi Power** | 5V/5A Buck Regulator | Pi 5 USB-C Power In | 5.1V / 5.0A DC | Dedicated isolated DC-DC converter |

---

## 💻 Hardware Software Suite (`scripts/`)

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| [`lawnmower_qr_mission_real.py`](scripts/lawnmower_qr_mission_real.py) | **Production Autonomous Mission Script**: Executes complete 7-stage mission on real hardware or in bench simulation. | `python3 scripts/lawnmower_qr_mission_real.py --url /dev/ttyACM0:921600` |
| [`stage1_bench_test.py`](scripts/stage1_bench_test.py) | **Stage 1 Benchtop Validation**: Caches reference QR, verifies twin QR matching, commands MAVLink `BRAKE` to Cube Orange+. | `python3 scripts/stage1_bench_test.py` |
| [`bench_visual_alignment_motor_test.py`](scripts/bench_visual_alignment_motor_test.py) | **Prop-Off Motor Servoing Test**: Arm in `GUIDED_NOGPS`, present QR to camera, observe motor ESC differential response. | `python3 scripts/bench_visual_alignment_motor_test.py --duration 30` |
| [`verify_siyi_stream.py`](scripts/verify_siyi_stream.py) | **Camera Stream Benchmark**: Zero-latency threaded RTSP reader, FPS and latency analyzer. | `python3 scripts/verify_siyi_stream.py` |
| [`test_mavlink_cube.py`](scripts/test_mavlink_cube.py) | **Autopilot Telemetry & Mode Diagnostics**: Validates Cube Orange+ battery, GPS, local NED, compass, and RC modes. | `python3 scripts/test_mavlink_cube.py` |
| [`show_qr_cards.py`](scripts/show_qr_cards.py) | **Visual Card Test Utility**: Fullscreen visual cards for bench testing (Reference, Distractors, Twin). | `python3 scripts/show_qr_cards.py` |
| [`fly_to.py`](scripts/fly_to.py) | **Direct Waypoint Utility**: Standalone coordinate navigation utility for quick NED repositioning. | `python3 scripts/fly_to.py --target-n 5.0 --target-e 2.0 --alt 4.0` |

---

## 🧪 Testing Procedures

### 1. Benchtop Hardware Simulation Test (`--bench-sim`)
Run the full 7-stage mission on the bench with **propellers removed**, real live camera, and real Cube Orange+ without needing outdoor GPS:

```bash
python3 scripts/lawnmower_qr_mission_real.py --bench-sim
```

**Bench Test Sequence:**
1. **Stage 1 (Takeoff):** Emulates climb to 4.0m altitude in software; HUD reticle initializes.
2. **Stage 2 (Memorization):** Hold your printed Reference QR card under the SIYI camera lens. System extracts 21×21 module grid and saves `reference_qr_memorized.png`.
3. **Stage 3 & 4 (Sweep):** Virtual drone flies the serpentine grid. Present a distractor QR $\to$ rejected ($S < 88\%$). Present the Twin Target QR $\to$ instant match ($S \ge 88\%$)! Cube Orange+ physically switches to `BRAKE` mode.
4. **Stage 5 (Visual Centering):** System enters fine-centering loop. Slightly move the QR to watch the centroid offset align to $< 25\text{ px}$.
5. **Stage 6 (Alignment Hold):** Large green HUD banner counts down the 15-second hold lock.
6. **Stage 7 (Landing):** System automatically commands `LAND` to Cube Orange+ and shuts down cleanly.

### 2. Physical Field Flight
When at the outdoor flying field with GPS lock:

```bash
python3 scripts/lawnmower_qr_mission_real.py \
  --url /dev/ttyACM0:921600 \
  --alt 4.0 \
  --speed 1.5 \
  --step 4.5 \
  --align-duration 15.0 \
  --land-after
```

---

## 🌐 Gazebo Sim 8 (Harmonic) & ROS 2 Simulation

To run the mission in Gazebo Sim 8 simulation with ArduPilot SITL:

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

## 📊 Verification & Telemetry Results

| Evaluation Metric | Target Spec | Simulated (SITL) | Hardware Bench Test | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Twin QR Recognition** | $> 88\%$ visual agreement | 100.0% (`AEROTHON_QR_02`) | **99.5% agreement** | ✅ Verified |
| **Braking Response** | Instant halt on detection | `COMMAND_ACK: 176 Result: 0` | MAVLink `BRAKE` confirmed | ✅ Verified |
| **Camera Latency** | $< 100\text{ ms}$ glass-to-glass | Threaded ROS bridge (<20ms) | **~35 ms** via RTSP UDP grabber | ✅ Verified |
| **Visual Servoing Accuracy** | $< 25\text{ px}$ optical error | 0.7 pixels ($0.8\text{ cm}$) | Closed-loop convergence | ✅ Verified |
| **Steady Alignment Hold** | 15.0 seconds continuous lock | 15.0s held | 15.0s HUD countdown | ✅ Verified |
| **Touchdown Positioning** | Directly over matched QR | Centered on target pad | Automated `LAND` mode switch | ✅ Verified |

---

## 📄 License & Attribution
Developed for the **Aerothon 2026 Autonomous Drone Reconnaissance Challenge**. Distributed under the MIT License.
