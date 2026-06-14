# Matter — Markerless Motion Capture Pipeline

Real-time markerless human motion capture using MediaPipe. Extracts body, face, and hand landmarks from a live webcam or uploaded video, applies uncertainty-aware frame filtering, drives a 3D avatar in the browser, and exports Blender-ready BVH files.

Three operating modes are available in a single browser UI: **Live** (webcam → avatar), **Video** (uploaded clip → pose extraction), and **Gesture** (gesture-controlled avatar in a 3D obstacle course).

---

https://github.com/user-attachments/assets/08544eb9-13c4-4bd3-b37b-e0b75369e806

## Project Structure

```
Matter/
├── backend/
│   ├── main.py             # FastAPI app + WebSocket loop
│   ├── pose.py             # MediaPipe Pose / Face Mesh / Hands extractor
│   ├── uncertainty.py      # Aleatoric + epistemic uncertainty scorer
│   ├── bvh_writer.py       # Uncertainty-gated BVH exporter
│   ├── action_detector.py  # Pose-to-action detector (Live mode)
│   ├── actions.py          # Action types, controller, keyboard input
│   └── requirements.txt
├── frontend/
│   ├── index.html          # Three-tab UI (Live / Video / Gesture)
│   ├── avatar.js           # Three.js Mixamo avatar + IK + camera
│   ├── action_controller.js
│   ├── uncertainty_ui.js   # Real-time uncertainty bars
│   ├── pose_action_mapper.js  # Gesture → move mapper (Gesture tab only)
│   └── environments/
│       └── obstacle_course.js  # 3D obstacle course scene
├── models/
│   └── avatar.glb          # Mixamo character (GLB format)
└── output/                 # BVH recordings saved here
```

---

## Installation

### 1. Install Conda

Download and install Miniconda or Anaconda:  
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### 2. Create and activate the environment

```bash
conda create -n sim python=3.10 -y
conda activate sim
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

Or individually:

```bash
pip install mediapipe==0.10.21 opencv-python fastapi "uvicorn[standard]" websockets numpy python-multipart pygame
```

### 4. Download the avatar model

- Go to https://www.mixamo.com
- Pick any character, download as **glTF/GLB** (not FBX)
- Save the file to `models/avatar.glb`

> The server will still start without it, but the 3D view will show a placeholder.

---

## Running the System

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open a browser and navigate to:

```
http://localhost:8000
```

Allow camera access when the browser prompts. The three-tab dashboard appears immediately.

---

## Modes and Usage

### Live Tab

The default mode. Your webcam feed is processed in real-time by the backend.

| Step | Action |
|---|---|
| 1 | Allow camera access in the browser |
| 2 | The skeleton overlay and 3D avatar appear automatically |
| 3 | Click **Start Recording** to begin capturing |
| 4 | Click **Stop** then **Save BVH** to export the animation |

Keyboard controls while Live is active:

| Key | Action |
|---|---|
| `W / S` | Walk forward / backward |
| `A / D` | Strafe left / right |
| `Q / E` | Turn left / right |
| `Space` | Jump |
| `Shift` | Sprint |
| `Ctrl` | Crouch |

**Pose-driven control (Live tab):** toggle the **Pose Mode** switch in the dashboard. After a ~1.5 s calibration (stand still, facing the camera), your body movements drive the avatar:
- Walking in place → avatar walks
- Leaning forward / backward → walk direction
- Rotating your shoulders → avatar turns
- Crouching → avatar crouches
- Jumping → avatar jumps

### Video Tab

Upload a recorded video clip for offline pose extraction.

| Step | Action |
|---|---|
| 1 | Click **Choose File** and select a `.mp4`, `.mov`, `.avi`, or `.webm` file |
| 2 | Click **Process Video** |
| 3 | The backend extracts one landmark frame per video frame and returns the full sequence |
| 4 | Landmark positions are displayed per frame; the JSON sequence is available via the API at `/process_video` |

Accepted formats: `.mp4 .mov .avi .webm .mkv .m4v`

### Gesture Tab

Controls a Mixamo avatar through a 3D obstacle course using hand gestures. No keyboard needed.

| Gesture | Action |
|---|---|
| Right hand raised above shoulder | Walk forward |
| Left hand raised above shoulder | Crouch-walk forward (pass under the low barrier) |
| Both hands raised above shoulders | Sprint forward |
| Both hands raised above head | Jump (one-shot on rising edge) |
| Right hand thumbs-up | Turn right (35°, 1 s cooldown) |
| Left hand thumbs-up | Turn left (35°, 1 s cooldown) |

**Obstacle course layout** (auto-resets when you reach the finish):

```
Start → [Crouch barrier] → [Two-wall weave maze] → [Jump gap] → [Ramp] → Finish
```

The camera switches to third-person follow view automatically in this tab.

---

## Uncertainty Scoring (Paper 2)

Every frame is scored before it is written to BVH or sent to the avatar.

| Component | Source | Meaning |
|---|---|---|
| **Aleatoric** | MediaPipe visibility scores | How noisy/ambiguous is this observation |
| **Epistemic** | Monte Carlo (20 passes, Gaussian noise σ=0.01) | How uncertain the model is about this pose configuration |
| **Total** | aleatoric + epistemic, clamped to [0, 1] | Frames above 0.7 are **untrusted** and skipped |

The coloured bar in the dashboard shows live uncertainty:
- **Green** — trusted frame, avatar updates and BVH records
- **Red** — untrusted frame, avatar holds last known pose, BVH skips the frame

Coverage statistics (trusted % over the session) are available via the **Coverage** button in the UI.

---

## Detection Models

Three MediaPipe models run in parallel on every frame:

| Model | Landmarks | Used for |
|---|---|---|
| **Pose** | 33 body keypoints + world 3D | Avatar skeleton, action detection, BVH export |
| **Face Mesh** | 478 facial landmarks (irises included) | Head bone rotation (nod / turn / tilt) |
| **Hands** | 21 landmarks × 2 hands | Finger curl on 30 finger bones; gesture detection in Gesture tab |

All three are drawn on the live camera preview streamed to the browser.

---

## Expected Output

### Console (backend)

```
Starting Matter backend...
Open http://localhost:8000 in your browser.
Client connected.
Frame 100 | pose_mode=False | calibrated=False
[CALIB] Frame accepted | Progress: 23/45 (51.1%)
[ActionDetector] Calibrated — shoulder_w=0.312  torso=0.241  hip_y=0.583
Frame 200 | pose_mode=True | calibrated=True
```

### Browser UI

- **Camera panel (left):** live annotated video with green pose skeleton, orange face mesh contours, and cyan hand skeletons
- **3D panel (right):** Mixamo avatar mirroring detected pose in real-time
- **Dashboard (right sidebar):** uncertainty bars, recording controls, BVH frame counter, active action indicators

### BVH file (`output/`)

```
recording_20260531_143022.bvh
  Frames recorded : 312
  Frames skipped  : 18
  Duration        : 10.40 seconds
  Coverage        : 94.6% trusted frames used
```

The BVH file imports directly into Blender via **File → Import → BVH**. The skeleton matches Mixamo's joint naming so retargeting is straightforward.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `GET` | `/health` | Server status, recording state, frame count |
| `GET` | `/output/list` | Lists saved BVH files |
| `POST` | `/process_video` | Upload a video, returns full landmark sequence as JSON |
| `WS` | `/ws` | Real-time WebSocket: streams annotated frames + landmark data + avatar state |

---

## Troubleshooting

**Camera not accessible**
```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Import errors**
```bash
python -c "import mediapipe, cv2, fastapi; print('OK')"
```

**Port 8000 already in use**
```bash
uvicorn main:app --reload --port 8001
```

**Avatar model not found**
- Ensure `models/avatar.glb` exists
- Format must be `.glb` (not `.fbx`)

**Gesture tab — turns are too fast / too slow**  
Edit `TURN_STEP_DEG` and `TURN_COOLDOWN_MS` in `frontend/pose_action_mapper.js`.

**Pose mode — avatar drifts sideways at rest**  
Reset calibration via the **Reset Calibration** button and stand still for ~1.5 s.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| mediapipe | 0.10.21 | Pose, face, hand detection |
| opencv-python | latest | Camera capture, frame encoding |
| fastapi | ≥ 0.95 | REST API + static file serving |
| uvicorn | ≥ 0.20 | ASGI server |
| websockets | ≥ 11.0 | Real-time data streaming |
| numpy | ≥ 1.21 | Numerical operations |
| python-multipart | ≥ 0.0.5 | Video upload handling |
| Three.js | CDN | 3D rendering (frontend) |
