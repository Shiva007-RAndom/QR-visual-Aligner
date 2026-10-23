# Autonomous UAV QR Survey & Precision Visual Servoing
## Antigravity Project Context & Field Operational Instructions

This document provides complete system context, architectural specifications, flight controller parameters, hardware interfaces, and test runbooks for any Antigravity session operating on this repository.

---

## 1. System Overview & Hardware Architecture

- **Airframe:** Holybro X500 Quadcopter (Carbon fiber, 500mm wheelbase).
- **Flight Controller:** CubePilot Cube Orange Plus running **ArduPilot Copter 4.5+**.
- **Companion Computer:** Raspberry Pi 5 (4GB / 8GB) running Debian Linux & Google Antigravity.
- **Optics & Vision:** SIYI Ethernet Camera (A8 mini / ZR10), mounted facing downward at nadir (-90° pitch).
  - Video stream: RTSP H.264 / H.265 at default IP `192.168.144.25`.
  - Pi 5 Ethernet (`eth0`): Configured with static IP `192.168.144.50/24`.
- **Telemetry Link:** Cube Orange+ `TELEM2` port connected to Raspberry Pi 5 via USB-to-FTDI UART (`/dev/ttyUSB0`) at **921,600 baud** running **MAVLink 2**.
- **Power Architecture (CRITICAL):**
  - Raspberry Pi 5 is powered by an independent, dedicated **5V / 5A Step-Down BEC** wired directly from the main LiPo.
  - **NEVER** power the Pi 5 from the Cube's 5V internal rail (Cube rail is strictly rated for 1A max).
  - SIYI camera is powered by 12V–26V (do NOT power from 5V).
  - All subsystems share a common ground (GND) to prevent floating reference levels.

---

## 2. Mission Logic & Visual Alignment Algorithm

The autonomous mission follows an 8-phase execution pipeline implemented in `scripts/lawnmower_qr_mission.py`:

1. **Takeoff & Local NED Origin:**
   - Outdoor ArduPilot EKF3 initializes Local NED coordinate origin $(N = 0.0\text{m}, E = 0.0\text{m})$ at the physical takeoff pad.
   - Drone arms in `GUIDED` mode and ascends to 4.0 meters above ground level.
2. **Reference QR Extraction & Module Grid Caching:**
   - Drone hovers at 4m directly above the takeoff pad for 1.5 seconds.
   - Downward SIYI camera captures the reference QR tile (2m × 2m).
   - Computer vision pipeline decodes the QR, crops the module matrix, downsamples to an exact 21×21 binary grid, and caches it in memory.
3. **Transition to Search Area:**
   - Drone transitions smoothly forward across a 5m safety buffer into the designated search zone (e.g., North from +5.0m to +30.0m, East from -12.0m to +12.0m).
4. **Lawnmower Serpentine Survey:**
   - Traverses parallel North-South search legs spaced by 4.0m–4.5m (providing 35%–50% optical swath overlap at 4m altitude) at a cruise speed of 1.5 m/s.
5. **Twin Candidate Identification:**
   - Continuous camera stream scans ground QR candidates.
   - Computes normalized bitwise Hamming correlation:
     $$\text{Score} = \frac{1}{441} \sum_{i=0}^{20} \sum_{j=0}^{20} \mathbb{I}(M_{\text{ref}}[i,j] == M_{\text{candidate}}[i,j])$$
   - When score $\ge 88\%$, the target twin is positively confirmed.
6. **Instant BRAKE & Momentum Arrest:**
   - Mission controller immediately issues `DO_SET_MODE` -> `BRAKE`.
   - Halts cruise momentum and settles the camera frame.
7. **Pinhole Ray Projection & Relative Re-referencing:**
   - Projects normalized image pixel error $(x_n, y_n)$ through the nadir pinhole model:
     $$\begin{pmatrix} \Delta x_B \\ \Delta y_B \end{pmatrix} = z \begin{pmatrix} \frac{u_c - c_x}{f_x} \\ \frac{v_c - c_y}{f_y} \end{pmatrix}$$
   - Transforms body displacement to local NED using yaw heading $\psi$:
     $$\begin{pmatrix} \Delta N \\ \Delta E \end{pmatrix} = \begin{pmatrix} \cos\psi & -\sin\psi \\ \sin\psi & \cos\psi \end{pmatrix} \begin{pmatrix} \Delta x_B \\ \Delta y_B \end{pmatrix}$$
   - Calculates target ground coordinates:
     $$N_{\text{target}} = N_{\text{drone}} + \Delta N, \quad E_{\text{target}} = E_{\text{drone}} + \Delta E$$
   - Drone switches to `GUIDED` and flies directly to $(N_{\text{target}}, E_{\text{target}})$.
8. **Closed-Loop Visual Servoing & Precision Hover:**
   - At target waypoint, drone performs optical servoing driven by pixel error $e = (u_c - c_x, v_c - c_y) \to 0$.
   - Locks optical crosshair over QR centroid with sub-centimeter accuracy ($< 1\text{ cm}$, $0.7\text{ px}$ error).
   - Holds steady alignment lock for 15.0 seconds.

---

## 3. Flight Controller Parameters (Cube Orange+)

Configure these parameters in Mission Planner before flight:

```ini
SERIAL2_PROTOCOL = 2       # MAVLink 2 on TELEM2
SERIAL2_BAUD     = 921     # 921,600 baud rate (zero latency)
ARMING_CHECK     = 1       # Strict safety pre-arm checks
FS_GCS_ENABLE    = 1       # RTL on MAVLink connection drop
FENCE_ENABLE     = 1       # Hard safety geofence enabled
FENCE_RADIUS     = 35.0    # 35-meter maximum radius boundary
FENCE_ALT_MAX    = 10.0    # 10-meter flight ceiling
WPNAV_SPEED      = 150.0   # 1.5 m/s survey cruise speed
MNT1_TYPE        = 4       # Gimbal locked facing Nadir (-90°)
```

---

## 4. Emergency Manual Takeover Configuration

The safety pilot must always be able to override companion computer automation:
- **RC Channel 5 (3-Position Switch):**
  - Position 1: `ALT_HOLD` (Full manual throttle & attitude)
  - Position 2: `LOITER` (Rock-solid GPS position hover)
  - Position 3: `GUIDED` (Companion computer autonomous control)
- **RC Channel 7 or 8 (Toggle Switch):**
  - Assigned to `BRAKE` mode for instantaneous freeze in place.

---

## 5. Software Stack & Integration Details

### Pure Python Architecture (No ROS 2 Required on Pi 5)
The production mission script runs entirely in lightweight Python using `pymavlink`, OpenCV, and `pyzbar` (<15% CPU on Raspberry Pi 5).

### Connecting the SIYI Camera Feed (Threaded Worker)
To eliminate RTSP buffer lag, use a threaded frame reader:
```python
import threading
import cv2

class SiyiStreamWorker:
    def __init__(self, capture_source="rtsp://192.168.144.25:8554/main.264"):
        self.cap = cv2.VideoCapture(capture_source, cv2.CAP_FFMPEG)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, img = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = img

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
```

---

## 6. Progressive Testing Protocol (Runbook)

1. **Stage 1: Benchtop Test (PROPELLERS OFF!):**
   ```bash
   python3 scripts/lawnmower_qr_mission.py --url /dev/ttyUSB0:921600 --alt 4.0
   ```
   - Present reference QR to camera -> verify `[SUCCESS] REFERENCE QR MEMORIZED`.
   - Present candidate twin QR -> verify `[MATCH FOUND!] 100.0%` and mode switches to `BRAKE`.
2. **Stage 2: Manual Flight & Loiter Stability:**
   - Outdoor takeoff in `ALT_HOLD`, switch to `LOITER`.
   - Verify minimum 14 satellites and HDOP < 0.8. Test RC `BRAKE` switch.
3. **Stage 3: Companion Single-Waypoint Test:**
   ```bash
   python3 scripts/fly_to.py --target-n 5.0 --target-e 0.0 --target-d -4.0 --url /dev/ttyUSB0:921600
   ```
   - Hover in `LOITER`, switch to `GUIDED`, execute 5m forward translation.
   - Flip RC back to `LOITER` to confirm manual override works instantly.
4. **Stage 4: Full Autonomous Mission:**
   ```bash
   python3 scripts/lawnmower_qr_mission.py \
     --url /dev/ttyUSB0:921600 \
     --alt 4.0 \
     --speed 1.5 \
     --step 4.5 \
     --align-duration 15.0 \
     --no-camera
   ```

---

## 7. Key Documentation Files in Repository
- `docs/real_world_deployment_guide.pdf`: Complete 4-page print-ready hardware deployment manual.
- `docs/visual_alignment_deep_dive.pdf`: Mathematical formulation of ray projection and visual servoing.
- `aerothon_system_guide.pdf`: 6-page comprehensive SITL simulation and system guide.
